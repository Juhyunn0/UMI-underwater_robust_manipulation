# Control Methodology Log — BlueROV2 robust control

A **dated journal of the control development** for this simulator: which method we
introduced, **why** (what the previous method couldn't do), the **theory** behind it,
**how** we implemented it (and the key design decisions + their rationale), and the
**result**. Read top-to-bottom to follow the whole logic chain at a glance.

This is the *narrative* — distinct from the topic reference docs ([00_OVERVIEW](00_OVERVIEW.md),
[03_THRUSTERS](03_THRUSTERS.md), [04_HYDRO](04_HYDRO.md), [07_DISTURBANCES](07_DISTURBANCES.md), …),
which describe *what exists and how it works*. This log records *why we decided each step*.

**How it's maintained:** updated on every **major** control change / milestone (not small
bugfixes or refactors). Each update appends a dated entry below and keeps the Korean
twin [CONTROL_METHODOLOGY.ko.md](CONTROL_METHODOLOGY.ko.md) in sync. Entry format:
**Why → What (theory, *with the governing equations*) → How (implementation + decisions) →
Result**. Every entry includes its math.

---

## The setup — plant, disturbances, and what the controller controls

Shared context for every entry below. (Reference: [04_HYDRO](04_HYDRO.md),
[07_DISTURBANCES](07_DISTURBANCES.md); the plant physics is independently verified in
[HYDRO_VERIFICATION](HYDRO_VERIFICATION.md).)

**Plant (the controlled system).** BlueROV2, **FLU** body frame (x fwd, y left, z up),
gravity (0,0,−9.81). Rigid body m=11.2 kg, inertia diag(0.30375, 0.626, 0.5769), COM at the
body origin. State = pose η and body velocity ν; the 6-DOF Fossen model the controllers assume:

```
η = [x y z  φ θ ψ]ᵀ  (world position + roll/pitch/yaw)     ν = [u v w  p q r]ᵀ  (body lin+ang vel)
η̇ = J(η) ν
M ν̇ + C(ν)ν + D(ν)ν + g(η) = τ + w
  M = M_RB + M_A                  (rigid-body + added mass; diagonal M_A here)
  C(ν) = C_RB(ν) + C_A(ν)         (Coriolis/centripetal)
  D(ν) = D_L + D_NL·|ν|           (linear + quadratic drag, diagonal)
  g(η) = restoring (buoyancy B at CB = COM + coBM·ẑ_body, weight at COM; net +1.1 N up)
  τ = control wrench,   w = disturbance wrench
```
(In the sim M_A is applied as an external lagged force, not in MuJoCo's mass matrix — see
[HYDRO_VERIFICATION](HYDRO_VERIFICATION.md); the controllers still reason with M = M_RB+M_A.)

**Disturbance w (the environment forcing).** Three FLU layers — current + waves enter as a
*water velocity* through the relative velocity (so they modulate both drag and added mass,
Morison-like); kicks are a direct external force:

```
v_water(t,d) = v_current + v_wave(t,d)
v_r = ν_lin − Rᵀ v_water                                  → used in place of ν_lin in D(·), M_A(·)
v_wave(t,d) = Σ_i U_i e^(−k_i d)[ dir_i cos(ω_i t+φ_i) + ẑ sin(ω_i t+φ_i) ],  k_i = ω_i²/g
F_kick(t)  = Poisson-timed impulsive world-frame force (gusts), applied directly at the COM
```
So w is **DC (current) + oscillatory wave-band + impulses (kicks)** — the spectrum each
controller is judged against. (JONSWAP spectrum: see the 2026-06-14 eval-env entry.)

**Two boundaries — "input" means different things for the *plant* and the *controller*.**
The **BlueROV2 plant's input is the thrust** `τ` — the body wrench produced by the 6 thrusters
(the `τ` on the RHS of the Fossen equation); its output is the state (η, ν). The **controller's
input** is the measured state + reference, and its **output is the wrench command**, which — after
allocation + the T200 thrust curve — *becomes* that thrust. So **controller output = plant input =
thrust.** The closed loop:

```
 p_ref, ψ_ref, v_ref ┐
                     ├──►[ controller ]──► wrench cmd  τ_c = [Fx Fy Fz 0 0 Mz]
 measured η, ν ──────┘                            │ allocate:  f = B⁺ τ_c   (6 thruster forces, N)
        ▲                                         │ T200:      throttle = curve⁻¹(f) → data.ctrl
        │                                         ▼
        │                                [ thrusters → MuJoCo + hydro ]
        │     plant input = thrust τ = B·f ─►  M ν̇ + C(ν)ν + D(ν)ν + g(η) = τ + w  ─► new η, ν
        └─────────────────────────────────────────────────────────────────────────────────┘
```

**Controller I/O** (what each method below reads/writes — its input is *measurements*, not the thrust):

```
controller INPUT  (measured each step):  p (world pos), R (orientation → φ,θ,ψ),
        v (world lin vel), ω (body ang vel);  DOB-MPC also ν̇ (finite difference, for the EAOB).
        + reference: p_ref, ψ_ref, and v_ref for trajectories.
controller OUTPUT (= the plant input):  body wrench  τ_c = [Fx Fy Fz  Mx My Mz],  Mx = My = 0
        → 6 thruster forces  f = B⁺ τ_c   (B = 6×6 allocation, rank 5),  → data.ctrl (via T200)
        → realized thrust into the plant:  τ = B f   (uncommandable pitch My projected out)
```
**Rank-5 underactuation.** The 4 horizontal thrusters sit 0.0725 m below the COM, so surge
couples to pitch: `My ≈ −0.0725·Fx`. Pitch is never commanded; it floats to the trim where
buoyancy restoring balances the coupling: `sin θ* = 0.0725·Fx / (coBM·B)` (≈23° at 6 N). Every
method below outputs τ_c to this same allocation and inherits this constraint.

### The matrices — values, structure, and provenance

**Where the numbers actually come from (verified upstream, not just our files).**
- **Hydrodynamic coefficients — added mass M_A and damping D_L, D_NL:** identified experimentally
  (tow-tank static + dynamic tests) in **Wu, C-J. (2018), *6-DoF Modelling and Control of a Remotely
  Operated Vehicle*, MEng thesis, Flinders University — Tables 5.2 (added mass) & 5.3 (linear &
  quadratic damping)**. Every value below matches that thesis *exactly* (checked against the document).
  The same set is re-used by the peer-reviewed BlueROV2 benchmark **von Benzon et al. (2022, *J. Mar.
  Sci. Eng.* 10(12):1898)** and adopted into **MarineGym** (Chu et al., IROS 2025) → our
  [`BlueROV.yaml`](../marinegym_assets/BlueROV.yaml).
- **Rigid-body mass & inertia M_RB, geometry, the 6 thruster mounts:** the **`bluerov2_description` ROS
  URDF** (`BlueROV.urdf`) via MarineGym's Isaac asset → our [`bluerov.xml`](../bluerov.xml). *Note:*
  m = 11.2 kg comes from this CAD/URDF; **Wu's thesis used 11.5 kg** — the rigid-body and the hydro
  parameters have *different origins*, which is why we cite them separately.
- **volume (0.0113459 m³), coBM (0.01 m):** CAD-derived, in MarineGym's `BlueROV.yaml`.
- **T200 thrust curve & rotor config:** **Blue Robotics' published T200 performance data**, fit in
  MarineGym's `actuators/t200.py`.

**The matrices** (each 6-vector ordered **[surge, sway, heave, roll, pitch, yaw]**; units kg / kg·m²):

```
            surge  sway  heave    roll     pitch     yaw
          ┌ 11.2    0     0        0         0        0      ┐ Fx-axis
          │   0   11.2    0        0         0        0      │
M_RB  =   │   0     0   11.2       0         0        0      │   rigid body  (bluerov2_description URDF)
          │   0     0     0      0.30375     0        0      │   COM at body origin ⇒ diagonal,
          │   0     0     0        0       0.626      0      │   no m·z_g surge–pitch coupling
          └   0     0     0        0         0      0.5769   ┘

          ┌ 5.5    0      0      0      0      0    ┐   added mass  (Wu 2018, Table 5.2)
          │  0   12.7     0      0      0      0    │   (Xu̇,Yv̇,Zẇ,Kṗ,Mq̇,Nṙ), diagonal —
M_A   =   │  0     0    14.57    0      0      0    │   off-diag (e.g. Yṙ, Nv̇) dropped (MarineGym).
          │  0     0      0    0.12     0      0    │   heave 14.57 > m 11.2 ⇒ applied as the
          │  0     0      0      0    0.12     0    │   EMA-lagged external force, NOT in MuJoCo's
          └  0     0      0      0      0    0.12   ┘   mass matrix (see HYDRO_VERIFICATION)

M = M_RB + M_A = diag(16.70, 23.90, 25.77, 0.42375, 0.746, 0.6969)          — SPD (verified, T1.3)

D_L   = diag( 4.03,  6.22,  5.18, 0.07, 0.07, 0.07)   linear drag      (Wu 2018, Table 5.3)
D_NL  = diag(18.18, 21.66, 36.99, 1.55, 1.55, 1.55)   quadratic drag   (Wu 2018, Table 5.3)
   D(ν) = D_L + D_NL·|ν|  →  applied as the dissipative force  −(D_L·ν + D_NL·|ν|·ν)
   (both coefficient sets recovered back out of the sim to 0.00 %, T4.3)

C_RB(ν)·ν = [ m(qw−rv),  m(ru−pw),  m(pv−qu),  (I_z−I_y)qr,  (Iₓ−I_z)pr,  (I_y−Iₓ)pq ]ᵀ   (from M_RB)

            ┌  0     0     0      0    −a₃w   a₂v ┐   a = (a₁…a₆) = M_A diagonal
            │  0     0     0    a₃w     0    −a₁u │     = (5.5, 12.7, 14.57, 0.12, 0.12, 0.12)
C_A(ν)  =   │  0     0     0   −a₂v   a₁u     0   │   ν = [u v w  p q r]
            │  0   −a₃w   a₂v    0    −a₆r   a₅q │   skew-symmetric (Fossen 2011 Eq. 6.44);
            │ a₃w    0   −a₁u   a₆r     0    −a₄p │   verified C_A = −C_Aᵀ and == sim to
            └−a₂v   a₁u    0   −a₅q   a₄p     0   ┘   1e-14 (T1.1–1.2)

g(η)  restoring (FLU):  B = ρgV = 997·9.81·0.0113459 = 110.97 N  (up, at CB)
                        W = mg  = 11.2·9.81           = 109.87 N  (down, at COM)
                        net = B − W = +1.10 N up ;  CB = COM + coBM·ẑ_body,  coBM = 0.01 m
                        restoring moment = k·sinθ_tilt ,  k = coBM·B = 1.110 N·m/rad
   (volume, coBM from BlueROV.yaml; ρ = 997 fresh water; m, g from the URDF / model.opt.gravity)

τ = B · f   (plant input: body wrench from the 6 thruster forces f [N])
        thr0    thr1    thr2    thr3    thr4    thr5
      ┌ 0.707   0.707  −0.707  −0.707   0       0     ┐ Fx
      │ 0.707  −0.707   0.707  −0.707   0       0     │ Fy
B  =  │ 0       0       0       0       1       1     │ Fz
      │ 0.051  −0.051   0.051  −0.051  −0.110   0.110 │ Mx
      │−0.051  −0.051   0.051   0.051  −0.002  −0.002 │ My  ← only ±0.002 from the verticals
      └ 0.167  −0.167  −0.175   0.175   0       0     ┘ Mz     ⇒ rank 5, pitch ~uncommandable
   column i = [ d_i ; r_i × d_i ],  d_i = thruster axis (site +X),  r_i = pos − COM.
   4 horizontal thrusters at z = −0.0725 m (vectored ±45°) ⇒ the surge→pitch coupling; 2 vertical.
   (bluerov.xml sites)  ·  T200 curve (force ↔ throttle, the real driver layer): u∈[−1,1] → rpm
   (0.075 deadband, ±3900 rpm) → thrust via Blue Robotics' asymmetric T200 fit, t200_thrust(+1)=
   +64.13 N, t200_thrust(−1)=−51.55 N (~1.24 fwd/rev asymmetry). Allocation/curve: thrusters.py.
```

---

## 2026-06-14 — Baseline PID/PD station-keeping controller

**Why.** Before investing in advanced control we needed a *baseline* to quantify how
well a simple, model-light controller holds station and rejects ocean disturbances — a
yardstick every later method is measured against.

**What (theory).** A PID/PD setpoint regulator drives the vehicle to a world pose. P (and
D) give a spring–damper to the target; the **integral** term accumulates steady error to
cancel an *unknown constant* disturbance (e.g. a steady current) — the classic reason PID
"rejects DC": the sensitivity function S(0)→0 with integral action, so a constant input
disturbance leaves **zero steady-state error**. It does *not* reject time-varying (wave-band)
disturbances or impulses, because S(jω) is only small near DC.

**How (implementation).** [controller.py](../controller.py) `PoseController`: world-frame
position PD with the force rotated into the body frame (avoids anisotropic-gain "crabbing");
net-buoyancy feed-forward (+1.1 N); a **gated anti-windup integral** (integrate only near the
setpoint, then clamp) for the current bias; surge saturation + slew-rate limit + a soft pitch
guard. Decision: **never command pitch** — the BlueROV2 vectored-6 is rank-5 underactuated and
all 4 horizontal thrusters sit 0.0725 m below the COM, so surge couples to pitch (My≈−0.0725·Fx);
we leave roll/pitch to passive buoyancy restoring.

**Equations.**
```
e = p_ref − p                                            (world position error)
F_world = K_p e − K_d (v − v_ref) + K_i ∫e dt            (∫ gated: integrate only when |e|<e_gate)
F_world,z += −net_buoy                                   (buoyancy feed-forward)
F_body = Rᵀ F_world         (rotate to body; surge then slew-limited + saturated + pitch-guarded)
M_z = k_pψ·wrap(ψ_ref−ψ) − k_dψ·r + k_iψ ∫e_ψ dt         (yaw PD+I)
τ = [F_body,x, F_body,y, F_body,z, 0, 0, M_z]
```
*Why the integral rejects DC:* closed-loop sensitivity `S(jω) = 1/(1+L(jω))`; integral action makes
`S(0) = 0` → **zero steady-state error to a constant w**. But `|S(jω)|` is only small near DC, so the
wave-band and impulses pass through.

**Result.** Holds the origin; under a 0.2 m/s current the integral nulls the DC bias to
~0.5 cm. But the **wave band (≈13 cm radial std), impulsive kicks (~30 cm transients), and a
steady 9° trim pitch** remain — exactly the residual a model-based, disturbance-aware controller
should attack. → motivates MPC.

---

## 2026-06-14 — Evaluation environment: square mission + irregular JONSWAP waves

**Why.** To stress controllers we needed (a) a *moving* reference and (b) *realistic* sea
disturbance. The original 3-sinusoid wave model was too regular (a clear repeat period), so it
under-tested disturbance rejection.

**What (theory).** A JONSWAP wave spectrum sampled with **equal-energy frequency bins + a random
frequency per bin** kills the artificial repeat period (the key to "looks random"); `cos^(2s)`
directional spreading adds yaw excitation. Waves enter as a **water velocity** (orbital motion
with depth decay e^(−k·depth)), so they drive both wave drag and wave added-mass through the
relative-velocity hydro — a Morison-like model with no extra term.

**How (implementation).** [mission.py](../mission.py) `SquareMission` (approach → track → done,
auto-record CSV); `disturbances.jonswap_wave_specs(...)`. The square uses a continuously moving
setpoint with velocity feed-forward into the controller's D-term.

**Equations.**
```
JONSWAP:  S(ω) ∝ ω⁻⁵ exp(−1.25 (ω_p/ω)⁴) · γ^r,   r = exp(−(ω−ω_p)²/(2σ²ω_p²)),  ω_p = 2π/T_p
equal-energy bins → ω_i (one random ω per bin);  a_i = (H_s/4)√(2/N);  U_i = ω_i a_i   so 4√(Σa_i²/2)=H_s
v_wave, v_r:  as in "The setup" above (the components feed v_wave; v_r drives drag + added mass)
square reference (origin = corner, CCW, side S, speed c):  s(t) = c·(t − t₀)
   p_ref(s) traces the 4 edges of the S×S square;   v_ref = c · tangent(s)   (velocity feed-forward)
```

**Result.** A realistic test bed. Confirmed the PID tracks the square but with phase lag and the
underactuated pitch transients grow under disturbance — the same limitations seen in
station-keeping, now under motion.

---

## 2026-06-15 — MPC and DOB-MPC (ported from the paper)

**Why.** The PID baseline's limits were now quantified: it rejects the **DC current** but not the
**wave band** or **impulsive kicks**, and the underactuated surge→pitch coupling limits how hard
it can hold while moving. We wanted a controller that (a) respects actuator/state **constraints**
explicitly, (b) **anticipates** the future via a model, and (c) **actively rejects** disturbance.

**What (theory).**
- **MPC (Model Predictive Control):** at each control step, solve a finite-horizon optimal control
  problem — minimize tracking error + control effort over N steps subject to the dynamics model and
  input/state constraints — apply only the *first* optimal input, then re-solve next step (receding
  horizon). It beats PID because it uses the **model to look ahead** and handles **constraints**
  natively. But *plain* MPC has no integral action, so against a constant unmodeled disturbance it
  leaves a **gain-limited steady offset**.
- **DOB-MPC (Disturbance-Observer-Based MPC):** add an **Extended Active Observer (EAOB)** — an
  augmented-state Continuous-Discrete EKF (state = [pose η; velocity ν; disturbance w], 18-dim, with
  the internal model ẇ=0) that estimates the disturbance wrench **w online from measurements + the
  Fossen model**. The estimate `w_hat` is fed **into the MPC's prediction model** each step (held over
  the horizon), so the MPC plans *against* the estimated disturbance. This is what nulls the steady
  offset plain MPC leaves — and, unlike "add a feed-forward to the control", incorporating w into the
  *prediction* makes it a parameter-varying model the optimizer reasons with.

**Paper.** Hu, Li, Jiang, Han, Wen, "Disturbance Observer-Based Model Predictive Control for an
Unmanned Underwater Vehicle," *J. Mar. Sci. Eng.* 2024 ([docs PDF](Disturbance%20Observer-Based%20Model%20Predictive%20Control.pdf)).
We reused the validated EAOB + NMPC math from the standalone `bluerov2_mujoco_dobmpc/` package
and ported it into the marinegym (FLU) sim.

**How (implementation + key decisions).** [dobmpc_controller.py](../dobmpc_controller.py) +
[dobmpc/](../dobmpc/) (fossen/eaob/mpc copied verbatim; only params + frames are marinegym-specific).
Solver: CasADi + IPOPT NLP, multiple shooting, N=60, state 12 / control 4, analytical Fossen RK4
prediction, `w_hat` as a parameter. Design decisions and *why*:
- **Frame:** the observer/MPC run in the paper's NED/FRD; marinegym is FLU. A fixed `S=diag(1,−1,−1)`
  conjugation (`R_ned = S·R_flu·S`) maps state in and the 4-DOF wrench out — never hand-flip Euler
  (subtle-bug source). ([dobmpc/frames.py](../dobmpc/frames.py))
- **Params rebuilt from marinegym `BlueROV.yaml`** so the prediction model matches *this* plant and
  only the true current/wave/kick is left as `w`. Two traps fixed: **damping sign flip** (marinegym
  stores damping positive, Fossen wants it negative — get it backwards and the model is *anti-damped*);
  **ZG_MASS=0** (marinegym's COM is at the body origin, so it has no m·zg surge↔pitch *inertial*
  coupling, while keeping the buoyancy restoring ZG=0.01). ([dobmpc/params.py](../dobmpc/params.py))
- **Acceleration by finite difference, NOT `data.qacc`** — marinegym applies added mass as an
  *external force*, so qacc already contains it and would double-count it in the EKF measurement model.
- **Underactuation = "option (a)":** zero the MPC pitch/roll *position* weights, let pitch float to
  its physical trim, and let the EAOB absorb the steady surge→pitch coupling into `w`.
- **20 Hz control with zero-order hold** between physics substeps; feed the EAOB the *commanded* NED
  wrench it actually held.

**Equations.**
```
MPC — receding-horizon OCP, solved every step, apply only u₀:
  min_{x,u}  Σ_{k=0}^{N−1} ‖x_k − x_ref,k‖²_Q + ‖u_k‖²_R  +  ‖x_N − x_ref,N‖²_QN
   s.t.  x_{k+1} = f_d(x_k, u_k, ŵ),   |u_k| ≤ u_max,   |ν_lin| ≤ v_max,   |φ|,|θ| ≤ 1.2 rad
  prediction model (Fossen, ŵ a constant parameter over the horizon):
     ẋ = [ J(η)ν ;  M⁻¹( τ(u) + ŵ − C(ν)ν − D(ν)ν − g(η) ) ],   τ(u) = [u₁,u₂,u₃, 0,0, u₄]
  plain MPC sets ŵ = 0  → a gain-limited steady offset against a constant w.

EAOB — augmented continuous-discrete EKF, state x_a = [η; ν; w], internal model  ẇ = 0:
  predict:  ẋ_a = f(x_a, τ),   P⁺ = Φ P Φᵀ + Q,   Φ = exp(F·dt),  F = ∂f/∂x_a
  update:   z = [η; ν; τ],   ŵ enters via   h_τ(x_a) = M ν̇ + C(ν)ν + D(ν)ν + g(η) − w
            K = P Hᵀ(H P Hᵀ + R)⁻¹,   x_a ← x_a + K (z − h(x_a))
DOB-MPC = MPC with ŵ = (EAOB w-estimate) injected as the prediction parameter each step.
```

**Result.** Dynamic-positioning comparison under JONSWAP+current+kicks (radial RMS / DC bias):
**PID 13.3 cm / −0.1 · MPC 3.6 cm / +2.3 · DOB-MPC 3.7 cm / +0.3** (cm). Both MPC variants cut the
wave-band residual ~5× vs PID; **DOB-MPC's EAOB nulls the DC bias that plain MPC leaves (+2.3→+0.3 cm)**
— the paper's central result, reproduced on realistic irregular waves. The wave band itself stays a
shared residual (the ẇ=0 model can't track a 4 s wave) → future work: an **oscillator disturbance state**
(internal model principle / Fossen Ch. 8 wave filter).

---

## 2026-06-15 — DOB-MPC runtime (lag) diagnosis + acados recommendation

**Why.** Viewing the sim over viser from a MacBook felt laggy (slow-motion, occasional freezes). We
profiled to find the real bottleneck rather than guess.

**What we found (profile, DP under disturbance, 120 warm ticks).** Per control tick (20 Hz, budget
50 ms): **NMPC.solve ≈ 83 ms (≈79%)**, EAOB.update ≈ 22 ms (≈21%), full tick ≈ 106 ms = **0.47×  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
real-time**; rare **2.2 s freezes** (IPOPT cold-restart on solver failure). `cProfile` confirms the
time is *inside* the IPOPT solve (`casadi.Function_call`), not the Python rollout/Jacobian assembly.

**Why it's slow (root cause).** IPOPT is a general interior-point NLP solver that, every step, drives
the full nonlinear problem **to convergence (tol 1e-5)** — many interior-point iterations, each
factorizing a large sparse KKT system (N=60 → 732 state + 240 control vars + shooting constraints).
That "fully converge every step" is overkill: the system barely changes in 50 ms, yet we re-optimize
from scratch each tick. The three usual cheap wins are **already in place** — warm-start ✅, analytical
(CasADi autodiff) Jacobians ✅, light analytical-Fossen model ✅ — so the profile proves there's no
easy gain left on the MPC; only *how the NLP is solved* can move the median.

**Recommendation (decided method, port pending).** Move the solve to **acados**:
- **Real-Time Iteration (RTI):** do **one** SQP iteration per step (not solve-to-convergence), warm-
  started from the previous step. Because the system moves slowly, one step per tick accumulates and
  tracks the optimum → **fixed iteration count → deterministic, short solve, no freezes**.
- **HPIPM + (partial) condensing:** solve RTI's linear QP with a structure-exploiting QP solver that
  uses the OCP's time-banded block structure (condensing shrinks the KKT) — far faster than general MUMPS.
- **C code generation:** model/derivatives/solver compiled to native C → no Python/CasADi overhead.
- Expected **~83 ms → ~2–5 ms (15–40×)**, keeping N=60, the Fossen model, and the DOB structure
  (`w_hat` stays a horizon parameter). Our model is already CasADi-symbolic and the paper used acados,
  so it's a 1:1 re-encoding, not a re-derivation. Secondary win: replace the EAOB's finite-difference
  Jacobians with CasADi autodiff + a Cholesky solve (≈22→4 ms).
- **acados downsides:** C-library build + `acados_template` + env setup (not `pip`-only); RTI is a
  one-step *approximation* (needs a warm-up and can be less accurate on strong nonlinear transients;
  validate against the IPOPT solution); codegen means edits require regeneration; weaker globalization.
  Fallback if avoiding the toolchain: hand-rolled **LTV-QP + OSQP** (~5–15 ms, lighter, less robust).

**Equations.**
```
per tick now (IPOPT): solve the OCP above to tol 1e-5 — many interior-point iterations, each a sparse
   KKT factorization of size ~ N·(n_x+n_u) = 60·(12+4) = 960 vars + shooting constraints  → ~83 ms.
RTI (acados) instead: ONE Gauss-Newton SQP step per tick, warm-started from the shifted previous z:
   linearize  x_{k+1}=f_d(x_k,u_k,ŵ)  about the previous trajectory  →  one structured QP
   solve with HPIPM + (partial) condensing (exploits the time-banded KKT block structure)
   → fixed 1 iteration → deterministic ~2–5 ms (no convergence loop, no freezes).
```

**Result.** Analysis only this turn (no code changed). Constraint kept: **N=60** and the DOB structure
+ correctness must be preserved. Next step when we implement: prototype acados on this exact OCP, verify
its `u` matches the current IPOPT `u` on logged states, measure, then wire behind a `solver="acados"`
switch with IPOPT as the reference/fallback.

---

## 2026-06-15 — Trajectory-tracking comparison (square): MPC's pitch cost

**Why.** The DP comparison showed DOB-MPC's win on *station-keeping* (DC-bias rejection). We then ran
the **square trajectory** (1 m, 10 laps, JONSWAP+current+kicks) for all three controllers to see how
they behave on a *moving cornered reference* — the harder, always-transient case.

**What we found (steady window, skip lap 1; geometric off-path = distance to the 1 m square).**

| ctrl | off-path rms | off-path max | setpoint err | depth std | **pitch rms / max** |
|---|---|---|---|---|---|
| PID | 14.3 cm | 45.0 | 39.8 cm | 3.8 cm | 14.2° / **33.5°** |
| MPC | 2.3 cm | 12.7 | 12.7 cm | 1.7 cm | 20.2° / **62.0°** |
| DOB-MPC | 2.1 cm | 17.4 | 12.0 cm | 1.8 cm | 20.5° / **67.2°** |

Three findings: **(1)** MPC/DOB-MPC track the square **~7× tighter** than PID (off-path 2 cm vs 14 cm,
setpoint error 12 vs 40 cm) and hold depth tighter. **(2) The cost is pitch:** the MPC variants reach
**62–67°** (near-tumble) vs PID's 33°. PID deliberately caps surge + slew-limits + pitch-guards, so it
*trades tracking for bounded pitch*; the MPC (option (a): pitch unpenalized, no slew limit) pushes surge
hard to corner against the current → tight tracking **causes** the large pitch. **(3) DOB-MPC ≈ MPC**
here (2.1 vs 2.3 cm) — the observer's distinctive benefit is DC-bias rejection, which is a station-keeping
phenomenon; on an always-moving setpoint it's marginal.

**Implication (direction).** MPC trades the accuracy↔pitch axis *opposite* to PID, and 67° pitch is a
real loss-of-authority risk on hardware. This concretely motivates **option (b)** — put the surge→pitch
coupling (`My=−0.0725·Fx`) + restoring into the prediction model so the MPC *anticipates* and self-limits
pitch — and/or an explicit **pitch (or surge-slew) constraint** in the OCP, to keep the 7× tracking win
while taming pitch to PID levels.

**Equations (the metrics + the pitch cost).**
```
off-path error  = min over the 4 square edges of  dist(p_xy, edge)        (geometric shape error)
setpoint error  = ‖p_xy − p_ref(s)‖,   s = c·(t−t₀)                       (includes phase lag)
underactuation cost:  to corner against the current the MPC raises Fx, and  My ≈ −0.0725·Fx
   → trim pitch  sin θ* = 0.0725·Fx / (coBM·B)   (unbounded by the OCP under option (a) → 62–67°)
```

**Result.** Analysis only (no code change). Files: `recordings/20260615/square_{pid,mpc,dobmpc}_*.csv`,
comparison plot `square_compare_*.png`.

---

## 2026-06-15 — Orientation error diagnosis → option (b): pitch-aware MPC

**Why.** "x/y/z track well but the orientation errs" — decomposed across all three rotational
channels: **pitch is the dominant orientation error** (RMS 10–20°, square max 62–67° = near-tumble);
roll ≈1° (low excitation, and roll *is* controllable so it stays ~0); yaw <1° in our `yaw_ref=0`
runs (it only blows up on *turning* trajectories — a separate issue, below). Root cause of pitch:
the rank-5 surge→pitch coupling `My ≈ −0.0725·Fx` pitches the vehicle whenever the MPC raises surge to
track position, and option (a) neither **models** that coupling as a function of the surge *decision*
(DOB-MPC only saw the realized pitch moment as a frozen disturbance `ŵ[pitch]`, held constant over the
horizon) nor **bounds** pitch.

**What (theory).** Option (b): make the prediction model **anticipate its own surge's pitch and bound
it.** The MPC now foresees `more surge → more pitch` as an explicit function of the decision variable,
and a tightened pitch state bound implicitly caps the planned surge — the *optimal* equivalent of the
PID's hand-tuned surge limiter, but used only when tracking actually needs it.

**Equations.**
```
prediction model (NED): inject the coupling as a function of the surge DECISION u_surge:
   τ_My = +κ·u_surge ,   κ = SURGE_PITCH_COUPLING = 0.0725      (NED sign +, verified by the gate)
EAOB fed the same τ_My  ⇒  ŵ[pitch] → 0    (the coupling is now modeled, not double-counted as w)
pitch state constraint:  |θ_k| ≤ θ_max  ∀k ,   θ_max = 0.40 rad ≈ 23°
   ⇒ implicit optimal surge cap:  u_surge ≲ sin(θ_max)·zg·W / κ ≈ 5.9 N
```

**How (implementation, toggleable).** [dobmpc/mpc.py](../dobmpc/mpc.py) `_f_casadi` sets `τ_My=+κ·u0`
and tightens the `|θ|` bound 1.2→`THETA_MAX`; [dobmpc_controller.py](../dobmpc_controller.py) feeds the
EAOB the commanded wrench *with* the coupling so `w[pitch]→0` (the thruster command keeps `My=0` — the
rank-5 allocation realizes the coupling physically); [dobmpc/params.py](../dobmpc/params.py) adds
`PITCH_AWARE` (default on; off recovers option a) and `THETA_MAX`. The `+κ` NED sign is verified by the
equilibrium gate in `test_dobmpc.test_pitch_aware`.

**Result.** DOB-MPC option-a → option-b, disturbance ON:

| run | pitch_rms | pitch_max | position | w[pitch] |
|---|---|---|---|---|
| DP (15 s) | 15.0 → 13.4° | 30.0 → **22.9°** | radial 4.9 → 6.1 cm | 0.22 → **0.09** [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| square (2 laps) | 17.8 → 12.6° | 46.7 → **23.2°** | off-path 2.6 → 3.0 cm | — |

**Pitch max halved (capped at θ_max; the full-lap 67° → ~23°)** while position tracking is essentially
kept (off-path still ≪ PID's 14 cm), and `w[pitch]` drops (EAOB no longer absorbs the coupling). Cost:
~5% more solver fallbacks (the hard pitch constraint hardens the NLP) — a *soft* pitch constraint is a
future refinement. Remaining orientation work (deferred): **yaw on turning trajectories** = option (A)
rotate a world-frame `ŵ` to body at each predicted heading `ψ_k` (repeals the constant-body-`w`
Assumption 2 that goes stale at the yaw rate during turns), plus yaw weight 150→300; roll is already small.

---

## 2026-06-15 — acados SQP-RTI solver port (the lag fix, implemented)

**Why.** The runtime diagnosis above pinned the IPOPT NMPC as the bottleneck (≈83 ms/tick, 0.47×
real-time, 2.2 s cold-restart freezes) and recommended acados RTI. This turn **implements** that
recommendation, keeping IPOPT as the reference/fallback. Constraint kept: **N=60**, the DOB structure,
and correctness must be preserved (the acados `u` must match the validated IPOPT `u`).

**What (theory).** Same OCP, different *solve*. Replace IPOPT's solve-to-convergence with acados
**SQP-RTI**: one Gauss-Newton SQP iteration per tick, warm-started from the (internally shifted)
previous solution; the linearized QP solved by **PARTIAL_CONDENSING_HPIPM** exploiting the time-banded
KKT block structure; model/derivatives/solver compiled to **C**. The disturbance wrench `ŵ` stays an
on-line **parameter** (DOB structure preserved). N=60, dt, Q/R/QN, and the option-(b) bounds unchanged.

**Equations.** (same OCP as the MPC entry — only the solve changes)
```
per tick now (acados SQP-RTI):  ONE Gauss-Newton step about the previous trajectory z⁻:
   QP:  min_Δz  ½·Δzᵀ H Δz + gᵀ Δz   s.t.  linearized dynamics + bounds        (H = Gauss-Newton)
        H, g from the RK4(f(x,u,ŵ)) linearization;  HPIPM + partial condensing solves the banded KKT
   u₀ ← u₀⁻ + Δu₀ ,   shift z⁻ ← z for the next tick        → fixed 1 iteration → deterministic
integrator: ERK RK4, 2 substeps/interval (h = 25 ms) == mpc._rk4(n_int=2)
state bounds (roll, pitch=θ_max, |v_lin|) made SOFT (L2 slack) so a transient linearization can't make
   the RTI QP infeasible and stall the loop; control bounds stay HARD.   (IPOPT used hard state bounds.)
```

**How (implementation).** New [dobmpc/mpc_acados.py](../dobmpc/mpc_acados.py) `AcadosNMPC` reuses the
**exact** symbolic dynamics `dobmpc.mpc._f_casadi` (single source of truth) as the acados model;
LINEAR_LS cost `W=diag(Q,R)`, `W_e=QN` with per-stage time-varying `yref`. A factory
[`mpc.make_nmpc()`](../dobmpc/mpc.py) returns acados (`params.SOLVER="acados"`, default) or the IPOPT
`NMPC` (reference/fallback — and auto-fallback on any acados import/build failure);
[dobmpc_controller.py](../dobmpc_controller.py) calls it, and the `solve(x, ŵ, xref)→u` signature is
identical so the controller/EAOB/thruster path is unchanged.
[dobmpc/_acados_env.py](../dobmpc/_acados_env.py) pre-loads the acados shared libs with `ctypes
RTLD_GLOBAL` so the fast path works with **no shell `LD_LIBRARY_PATH`** (teleop users export nothing).
Toolchain: acados built into `/home/bdml/acados` (C lib + `acados_template` 0.5.1) inside the `robust`
env; numpy stays <2.

**Result.** Verified four ways ([verify_acados.py](../verify_acados.py)):

| check | IPOPT (reference) | acados SQP-RTI |
|---|---|---|
| solve / tick (N=60) | median **100 ms** (over the 50 ms budget) | median **0.97 ms**, max 1.1 ms [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| equivalence (interior states) | — | worst-case max\|Δu\| = **0.107 N** vs IPOPT (same optimum) |
| closed-loop DP (15 s, disturb) | radial 8.6 cm, pitch_max 22.9°, ŵ_x 3.19 N, **7 freezes** | radial 7.0 cm, pitch_max **22.9°**, ŵ_x 3.11 N, **0 freezes** |
| closed-loop square (1 m, 2 laps, disturb) | ~0.5× real-time | done, pitch_rms 14°, **0 freezes**, **1.2× real-time** |

**~103× median speedup**, deterministic (cold-restart freezes gone: `n_fail` 7→0), and the closed-loop
invariants match the validated IPOPT controller (option-b pitch cap 22.9°, EAOB estimate `ŵ_x`, DC
current rejection). Regression: `test_dobmpc.py`, `teleop --selftest`, `test_square_mission.py` all
pass. Trade-offs (per the recommendation): ~1 s codegen build at controller start; **soft** state bounds
(vs IPOPT hard) for RTI feasibility; RTI is a one-step approximation — validated against IPOPT here.
IPOPT stays selectable as the reference (`params.SOLVER="ipopt"`). Deferred: port the EAOB
finite-difference Jacobian to CasADi autodiff (≈22→4 ms).

---

## 2026-06-16 — Actuator-realism ablation (realistic T200 thrusters) + a discovered acados fragility

**Why.** All prior experiments command per-thruster force in N and assume it is realized exactly
(ideal force path). On the real BlueROV2 the low-level input is a normalized throttle/PWM; the T200
curve turns it into thrust, with a **deadband** (sub-~0.7 N lost, then a ~1.44 N minimum-spin jump),
**fwd/rev asymmetry**, **saturation**, **motor lag**, and a **voltage/wear gain error**. We asked
whether modelling these makes the sim meaningfully more realistic, and which controller is most robust.

**What (implementation, opt-in).** New `thrusters.ThrusterModel` (the real driver chain: T200 inverse →
motor lag → forward curve → `voltage_scale`), passed optionally through `set_wrench_command(actuator=)`
and the controllers (`actuator=None` default — the ideal path is unchanged). `ablation_thrusters.py`
runs DP (origin, disturbance ON, mean over 5 seeds) for PID / MPC / DOB-MPC under **ideal /
realistic / realistic-LV** (LV = ×0.85 thrust from battery sag).

**Result — actuator realism is a MODEST effect on DP (clean controllers).** PID and MPC (no solver
failures) station-keeping radial RMS [cm], mean over 5 seeds:

| ctrl | ideal | realistic | realistic-LV | jitter (std) ideal→LV |
|---|---|---|---|---|
| PID | 14.86 | 14.74 | 15.16 | 10.4 → 12.4 cm [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| MPC | 5.11 | 4.30 | 5.12 | 2.9 → 3.5 cm |

Radial RMS barely moves (within the ±7–9 cm seed scatter); the visible signature is **jitter
(position std) rising ~15–20 %** — the deadband limit-cycle. So adding realistic thrusters makes the
sim a bit more faithful (captures deadband chatter) **but does not change the DP controller ranking** —
the hold forces sit near/above the ~1.44 N deadband floor and the ~10 ms motor lag is well inside the
50 ms control tick. (A *moving* trajectory, where small per-thruster commands cross the deadband more
often, would stress it harder — a follow-up.)

**Result — the ablation incidentally exposed an acados DOB-MPC robustness bug.** Seed-averaging (which
the noise demanded) revealed that **on seed 3 the acados SQP-RTI cascades into `ACADOS_NAN_DETECTED` /
`MINSTEP` (n_fail 116) and blows up to 39 cm**, *independently of the actuator* (it happens on the ideal
path). Per-seed, ideal DOB-MPC: seeds 0/1/2/4 = 4.1 / 0.7 / 0.9 / 1.4 cm, n_fail 0 (excellent); **seed 3  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
= 39 cm, n_fail 116**. The single-seed acados verification (seed 0) missed this: a specific
wave/kick realization drives the EAOB `ŵ` into a regime where the RTI QP goes indefinite and, doing one
iteration, cannot recover (it holds a stale `u` → diverges → more failures). The IPOPT reference (full
convergence) is robust here. **Open fix (recommended): on repeated acados NaN, fall back to one IPOPT
solve for that tick** (IPOPT is already built as the reference), plus tighter `ŵ` clamping / QP
regularization. Until fixed, the DOB-MPC ablation numbers on seed 3 are a solver artifact, not an
actuator effect.

**Takeaway.** Modelling realistic thrusters is worth keeping as an opt-in sim-to-real stress test (it
adds the deadband jitter and the multiplicative-thrust robustness axis the additive DOB can't fully
cancel), but for DP it does not overturn the ideal-path comparison. The more urgent finding is the
acados DOB-MPC seed-3 NaN fragility — to be fixed with an IPOPT fallback.

---

## 2026-06-16 — Fix: acados DOB-MPC NaN fragility → IPOPT fallback + iterate re-init

**Why.** The ablation above found the acados SQP-RTI cascading into NaN and diverging on seed 3
(n_fail 116, 39 cm): a single failure leaves the RTI warm-started from a *corrupted* iterate, so every
later tick also fails and the held-stale `u` lets the vehicle drift away.

**What.** On any acados failure (NaN / min-step / non-finite u₀) `AcadosNMPC` now (1) **re-initialises
the acados iterate** (flat trajectory at the current x) so the next RTI restarts clean, and (2)
**recovers THIS tick with one IPOPT solve** — the validated full-convergence reference, built lazily on
the first failure. Previously it returned the stale `u₀` → divergence.

**How.** [dobmpc/mpc_acados.py](../dobmpc/mpc_acados.py): `fallback_ipopt=True` (default);
`_ipopt_fallback()` lazily builds the IPOPT `NMPC`; `n_fallback` counts recoveries; `_warm=False` forces
the clean acados restart. The no-failure path is untouched (same 0.97 ms RTI, same equivalence).

**Result.** Seed-3 ideal DOB-MPC: **39.04 cm / n_fail 116 → 12.82 cm / n_fail 1** — one fallback breaks
the cascade; the residual is now the genuine large-kick transient (bounded, recovered), not a solver
blow-up. Seeds 0/1/2/4 unchanged (0.7–4.1 cm, n_fail 0). Regression: `test_dobmpc`, `teleop --selftest`,
`verify_acados` (equivalence 0.107 N, 102.6× speedup) all pass. Trade-off: a failed tick costs one
~100 ms IPOPT solve (rare; pre-build the fallback for hard real-time). The acados DOB-MPC is now robust
across all five disturbance seeds.

---

## 2026-06-18 — Realistic T200 thrusters: datasheet-grounded `voltage_scale` + default-ON in teleop missions

**Why.** The realistic actuator (`ThrusterModel`: deadband / fwd-rev asymmetry / motor lag / voltage) was
**opt-in (ablation-only)**, so the autonomous teleop missions (`--square` / `--goto-origin`) — whose whole
point is to *predict the real BlueROV2* — ran the **ideal force path** (commanded == realized), which is a
non-physical idealisation. Separately, the `0.85` voltage loss used in the ablation was an **illustrative
value, not derived** from any datasheet.

**What.** (1) **Grounded the voltage scale** in the official datasheet; (2) made the realistic model the
**default for the mission paths** with an `--ideal-thrusters` opt-out (manual keyboard teleop, `eval_dp`,
and `ablation` are untouched — they use separate / explicit paths).

**Grounding (provenance).** Blue Robotics *T200 Public Performance Data 10–20 V (Sep 2019)*
(`marinegym_assets/*.xlsx`; reproduce with [analyze_t200_voltage.py](../analyze_t200_voltage.py), stdlib
zip/XML parse — no pandas):

| V | 10 | 12 | 14 | 16 | 18 | 20 |
|---|---|---|---|---|---|---|
| max fwd (kgf) | 2.93 | 3.71 | 4.53 | 5.25 | 6.02 | 6.72 |
| max rev (kgf) | −2.31 | −2.92 | −3.52 | −4.07 | −4.59 | −5.04 |

The MarineGym curve's max (`T200_MAX_FWD/REV` = +6.54 / −5.26 kgf) sits at the **top of the range** → its
`voltage_scale = 1.0` models a **~20 V** thruster. A real BlueROV2 runs a **4S Li-ion pack (nominal
14.8 V)**, where max thrust (interp 14↔16 V) is 4.81 / 3.74 kgf, so
`voltage_scale = 14.8V/base = 4.81/6.54 = 0.74 (fwd), 3.74/5.26 = 0.71 (rev)` → a single grounded scalar
**`NOMINAL_VOLTAGE_SCALE = 0.72`** (full-charge 16.8 V ≈ 0.83; near-empty 13 V ≈ 0.62). This **replaces the
illustrative 0.85**.

**How.** [thrusters.py](../thrusters.py): added `NOMINAL_VOLTAGE_SCALE = 0.72` (with the derivation in a
comment); the `ThrusterModel(voltage_scale=1.0)` constructor default is **left unchanged** so the ablation's
explicit `realistic` (V=1.0) scenario and other callers are not silently altered. [teleop.py](../teleop.py):
new `--ideal-thrusters` (opt-out) and `--thruster-voltage` (default `0.72`) flags; the mission branch builds
one `ThrusterModel(lag=True, voltage_scale=…)` and passes `actuator=` to both `DOBMPCController` and
`PoseController` (the actuator wiring + `reset()` already existed); a startup line prints the active path;
and the run manifest (`.meta.json`) now records `run.thrusters = {model, lag, voltage_scale}` so ideal vs
realistic runs are never confused.

**Result.** Closed-loop DP (dobmpc, seed 0, 20 s, disturbance ON): **ideal radial 5.02 cm / jitter 4.30 cm  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
→ realistic ×0.72 radial 7.76 cm / jitter 6.17 cm**, `n_fail 0` — the realistic stage degrades station-
keeping as expected (deadband jitter + a 28 % thrust deficit the additive DOB only partly cancels), with no
solver trouble. `analyze_t200_voltage.py` reproduces the table and `0.72` (MATCH). Regression: `eval_dp`
(ideal default) and `ablation_thrusters` scenarios unchanged; `teleop --selftest` passes.

**Scope / honesty.** The MarineGym curve is **not refitted** to 14.8 V (that would change `T200_MAX` /
`ctrlrange` everywhere); we keep the verbatim ~20 V curve and apply the scalar. The fwd/rev voltage ratios
(0.74 / 0.71) are approximated by the single 0.72. Inflow-velocity dependence (advance ratio: thrust drops
when moving), thermal, and fouling are **out of scope** — the realistic model is still the *static (bollard)*
curve. A *moving*-trajectory (square) run stresses the deadband harder than DP and is the natural follow-up.

---

## 2026-06-18 — BlueROV2 → BlueROV2 **Heavy**: 8 thrusters, fully actuated, 6-DOF MPC

**Why.** Moving from the standard vectored-6 BlueROV2 to the **Heavy** configuration. The headline is
actuation: Heavy adds **two more vertical thrusters** (4 total, at the corners), which makes the allocation
**rank 6 = fully actuated** — roll AND pitch become directly controllable, eliminating the rank-5
under-actuation that forced the whole option-(a)/(b) pitch workaround. Both variants are kept and selected
by the env var **`ROV_MODEL`** (`bluerov2` default | `heavy`); a new [rov_model.py](../rov_model.py) is the
single source of truth so the plant (MJCF/hydro) and the controller (params/NMPC) can never disagree.

**Provenance.** All values verified directly from the MarineGym USD
(`external/MarineGym/.../usd/BlueROVHeavy/BlueROVHeavy.usd`, parsed with `pxr`). Heavy keeps the SAME hydro
coefficients (added mass, linear/quadratic damping) and the SAME T200 thrusters as BlueROV2 — only mass,
inertia, buoyant volume, and the thruster layout differ. (The Heavy yaml lists a weaker `force_constants`
0.8e-7 which would scale thrust to ~18 %; we keep the validated T200 curve since the physical thruster is
unchanged — see [03_THRUSTERS.md](03_THRUSTERS.md).)

### What changed — values (기존 → 수정)

**Rigid-body mass matrix M_RB = diag(m, m, m, Ix, Iy, Iz):**
```
BlueROV2:  diag( 11.2, 11.2, 11.2,   0.30375, 0.626,  0.5769 )
Heavy:     diag( 11.5, 11.5, 11.5,   0.3291,  0.6347, 0.6109 )   (inertia derived — see below)
```

**Inertia I (diagonal):  bluerov2 → Heavy, DERIVED by parallel-axis:**
```
[ 0.30375                ]        [ 0.3291                ]
[         0.626          ]   →    [        0.6347         ]   I_heavy = I_bluerov2 + Δ
[                 0.5769 ]        [                0.6109 ]   Δ = [+0.0254, +0.0086, +0.0340]
```

> **⚠ Why NOT the Heavy USD's inertia — and how we derived this one.** The MarineGym/farol
> Heavy USD ships `[0.21, 0.245, 0.245]`, *smaller* than BlueROV2's despite Heavy being
> heavier — physically backwards. It is a **hand-tuned Gazebo-stability literal**: the farol
> source `bluerov_heavy_vehicle/urdf/base.xacro` hardcodes it with the comment *"... otherwise
> your model will become unstable on Gazebo"* (the physical ellipsoid formula right below is
> commented OUT; another dsor source even lists `[0.26, 0.23, 0.37]`, so it's inconsistent
> across sources too). BlueROV2's `[0.30375, 0.626, 0.5769]` comes from a different URDF
> (`bluerov2_description`).
>
> So instead of trusting the farol literal, we **derive a Heavy-specific tensor** from the
> BlueROV2 one. The 4 **horizontal** thrusters are at identical positions in both models, so
> they cancel exactly in the BlueROV2→Heavy inertia *difference*; the only change is the
> **vertical** layout — BlueROV2's 2 near-centre verticals (`±0.1105` y) → Heavy's 4 corner
> verticals (`±0.12` x, `±0.22` y). Treating each thruster as a **point mass of 0.15 kg**
> (model-consistent: the +0.3 kg / +2-thruster budget, 11.2→11.5), the difference is exactly
> the parallel-axis term `Δ = Σ_heavy m·(par-axis) − Σ_bluerov2 m·(par-axis)`:
> `I_heavy = I_bluerov2 + [+0.0254, +0.0086, +0.0340] = [0.3291, 0.6347, 0.6109]`. This holds
> whether or not the BlueROV2 base value includes its own thrusters (the hull + 4 horizontals
> cancel). Reproduce / change the thruster-mass assumption with
> [compute_heavy_inertia.py](../compute_heavy_inertia.py) (sensitivity: m_v 0.10→0.344 kg gives  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
> Ixx 0.321→0.362).
>
> **Honest limits:** this is a physically-motivated *estimate* (point-mass thrusters; the
> BlueROV2 base value's own CAD-vs-formula origin is itself unverified — its config.yaml names
> `bluerov2_description` upstream but a web search did not surface the exact tensor), **not** a
> Heavy CAD measurement. But it is Heavy-specific and strictly **≥ BlueROV2**, as physics
> requires — strictly better-founded than either the farol literal or a flat BlueROV2-reuse.

**Added mass M_A = diag(Xu̇, Yv̇, Zẇ, Kṗ, Mq̇, Nṙ):  UNCHANGED**
```
diag( 5.5, 12.7, 14.57, 0.12, 0.12, 0.12 )      (both variants)
```

**Total mass matrix M = M_RB + M_A:**
```
BlueROV2:  diag( 16.70, 23.90, 25.77,  0.42375, 0.746,  0.6969 )
Heavy:     diag( 17.00, 24.20, 26.07,  0.4491,  0.7547, 0.7309 )
```

**Damping D_L, D_NL:  UNCHANGED** — D_L = −diag(4.03, 6.22, 5.18, 0.07, 0.07, 0.07),
D_NL = −diag(18.18, 21.66, 36.99, 1.55, 1.55, 1.55).

**Buoyancy:** volume 0.0113459 → **0.0116499** m³ ⇒ B = ρgV 110.97 → **113.94** N. Heavy is both heavier
(W 109.87 → 112.82 N) and bigger, so **net buoyancy stays ~+1.1 N** (B−W = +1.10 → +1.13 N).

**Allocation B (wrench = B·thruster_forces):**
```
BlueROV2:  6×6,  rank 5   — pitch My is NOT independently controllable (under-actuated)
Heavy:     6×8,  rank 6   — FULLY ACTUATED (verified: a pure pitch wrench realizes My=1.000)
```
Thrusters: the 4 horizontal (thruster_0..3) are identical; the verticals change:
```
BlueROV2:  2 vertical at ( 0.0025, ±0.1105, −0.005)
Heavy:     4 vertical at (±0.12,   ±0.22,   −0.005)   (+Z, four corners → indep. Fz/roll/pitch)
```

**Controller (NMPC) — exploiting full actuation:**
```
NU (control dim):  4  [X,Y,Z,N]        →  6  [X,Y,Z,K,M,N]
tau mapping:       [X,Y,Z, 0, κ·X, N]  →  [X,Y,Z, K, M, N]   (κ = surge→pitch coupling, gone)
PITCH_AWARE (opt-b surge cap):  True   →  False   (pitch is a commanded DOF now)
MPC_Q roll/pitch position weight:  0,0 →  80,80   (the MPC actively levels the vehicle)
```

**Result (DP, dobmpc, seed 0, 20 s, disturbance ON):**
| variant | radial RMS | pitch mean | pitch max |
|---|---|---|---|
| BlueROV2 (rank-5) | 5.0 cm | **+11.8°** | 22.9° |
| Heavy (full 6-DOF, inertia derived) | **3.3 cm** | **+0.4°** | 5.4° |

Full actuation **actively levels pitch** (11.8° trim → 0.8°) and tightens station-keeping (5.0 → 3.3 cm).  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]

**How / blast radius.** [rov_model.py](../rov_model.py) (registry); [bluerov_heavy.xml](../bluerov_heavy.xml)
+ [marinegym_assets/BlueROVHeavy.yaml](../marinegym_assets/BlueROVHeavy.yaml); [params.py](../dobmpc/params.py)
(per-model MASS/I/VOL, NU, U_MAX, MPC_Q/R, PITCH_AWARE); [mpc.py](../dobmpc/mpc.py) (NU + NU-aware tau);
[mpc_acados.py](../dobmpc/mpc_acados.py) (per-model codegen dir); [thrusters.py](../thrusters.py) (allocation
discovers 6/8 thrusters from the model); [hydro.py](../hydro.py) (per-model yaml/volume);
[dobmpc_controller.py](../dobmpc_controller.py) (NU-aware wrench); [teleop.py](../teleop.py) /
[eval_dp.py](../dobmpc/eval_dp.py) (XML + ThrusterModel n from the model). BlueROV2 path is numerically
unchanged (regression: dobmpc DP 5.0 cm, test_thrusters/test_dobmpc/test_controller, `teleop --selftest` all
pass on both variants).

**Scope / follow-ups.** The PID baseline ([controller.py](../controller.py)) still commands only
surge/sway/heave/yaw on both variants (on Heavy the rank-6 allocation passively cancels the surge→pitch
coupling, so PID also levels better) — a full 6-DOF PID is a follow-up. The Heavy MJCF reuses the BlueROV2
visual meshes (visual only; dynamics are the Heavy inertial + 8 sites). Roll/pitch U_MAX (8 Nm) and Q
weights (80) are initial values, open to tuning. The actuator-realism ablation / verify_hydro suites still
target BlueROV2; extending them to Heavy is a follow-up.

## 2026-06-29 — Finite-depth disturbance environment + 12-run controller comparison

To fairly compare PID/MPC/DOB-MPC disturbance rejection at a shallow (h=4 m), swell-dominated Monterey Bay
site, added a **finite-depth directional irregular wave + ocean current (mean + drift) + Froude-Krylov inertia**
disturbance environment. The legacy `disturbances.py` (deep-water, k=ω²/g) is unfit for the shallow target
(kh≈0.34) → new `disturbance/` package.

**Physics decomposition (verified by the underwater-robotics & control-theory advisors).**
- Drag: the hull damping D(nu_r) (on relative velocity) is already the Fossen-equivalent of Morison drag →
  feed the fluid velocity through `water_velocity` ONLY; do **not** add `0.5ρC_D A v_rel|v_rel|` (double-count).
- Inertia: hydro's `−M_A·d(nu_r)/dt` already supplies the added-mass (C_a) part → inject only the missing
  **Froude-Krylov ρ∀·a_wave (C_M=1)** as an external force. (The slide's flat C_M=1.5 is right only on surge;
  per-axis C_a=[0.49,1.12,1.29].)
- Finite depth: `ω²=g·k·tanh(k·h)` (Newton-solved) + cosh/sinh depth profiles — fixes the ~3× k error and the
  non-vanishing seabed vertical velocity of deep-water at kh≈0.34.

**5 modes** (same seed → wave phases + GM drift bit-identical across modes, only the layer toggles differ):
NONE / C / CD / CW / CDW, where **NONE = still water** (current+drift+waves all off) is the disturbance-free
baseline, and C/CD/CW/CDW add current / +drift / +waves / +drift+waves. **Kicks excluded.**

**Result (smoke, DP, seed 0, 8 s — qualitative validation):**
| mode | PID | MPC | DOB-MPC | DRR=MPC/DOB |
|---|---|---|---|---|
| C (current) | 16.5 | 2.9 | **0.39 cm** | 7.4 |
| CDW (+drift+waves) | 13.3 | 3.2 | **0.47 cm** | 6.8 |

DOB-MPC nearly fully rejects the DC current/drift (EAOB estimates w_x≈1.5 N = the 0.2 m/s drag, est_err
0.01–0.06 N); only the wave-band residual grows (band_wave 0.21→0.44 cm). On square (tracking) the nu_ref=0
structural lag dominates → DOB≈MPC (11 cm).

**How / blast radius.** New [disturbance/](../disturbance/){waves,current,env,config,test_*}.py,
[experiments/run_compare.py](../experiments/run_compare.py), config/{base,scenario_square}.yaml. The only edit
to existing dynamics is a read-only `diag_wtrue` diagnostic in [hydro.py](../hydro.py) (`w_true_world` =
plant force − still-water model force + FK; forces unchanged) — the env is a drop-in for hydro's duck-typed
disturbance interface. 34 unit asserts + smoke pass. Run:  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
`python -m experiments.run_compare --config config/base.yaml`.

**Scope / follow-ups.** The slide's (von Benzon) M_RB/M_A differ from the current MarineGym-identified set
(mass 11.2 vs 13.5, etc.) → to be adopted consistently as a separate variant `ROV_MODEL=bluerov2_vonbenzon`
(mass/inertia/volume-buoyancy/added-mass + params + re-tune + re-verify). `fk_mode=morison_ca` (per-axis full
Morison) is kept for the verification sweep (default froude_krylov).

## 2026-06-29 — Heavy = default + DOB-MPC trajectory reference (velocity FF + heading-follow)

Three coupled changes, motivated by watching the square mission live (`experiments/run_viewer.py`): make the
fully-actuated **Heavy** the standard vehicle, make the ROV **face its travel direction** on the square, and
fix the **large DOB-MPC square tracking error**.

**1. Heavy is now the project-wide default.** [rov_model.py](../rov_model.py): `ROV_MODEL` default `bluerov2`
→ **`heavy`** (8 thrusters, rank-6 fully actuated, NU=6). Everything downstream (hydro YAML, `dobmpc/params`,
allocation, per-variant acados codegen) reads `RM.*` so it follows automatically — verified end-to-end
(`heavy 8 True`, allocation 6×8 rank 6, heavy acados RTI solver already built). `test_load.py` hardcodes
`bluerov.xml` (a bluerov2-specific phase-1 check) so it is unaffected. Override with `ROV_MODEL=bluerov2`.

**2. Heading-follow + corner smoothing** (`experiments/run_compare.py` square branch + `run_viewer.py`).
On the square the yaw reference now tracks the path tangent `atan2(ty,tx)` (the ROV faces where it's going),
**slew-rate-limited** so the 90° corner change ramps smoothly (`slew_heading`, default **60 °/s** ≈ 1.5 s/corner)
instead of stepping — the POSITION path stays the sharp square (heading only). Config knobs in the square block:
`heading_follow: true`, `yaw_rate_deg_s: 60`; viewer flags `--heading {follow,fixed}`, `--yaw-rate`. Verified:  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
the logged `yaw_deg` ramps at ≤3.1°/log (no 90° jump), progresses 0→90→±180→−90 through all four corners.

**3. DOB-MPC `_xref_ned`: constant-pose DP tile → horizon trajectory reference** (the load-bearing fix).
Root cause of the big square error: the NMPC reference set `nu_ref = 0` always and **ignored `v_ref`** — a pure
DP (station-keeping) regulator structurally lags a moving target (the code comment named it "a follow-up").
[dobmpc_controller.py](../dobmpc_controller.py) `_xref_ned()` now builds, per horizon step, all as **runtime
reference data (model + cost unchanged → no acados rebuild)**:
- **velocity feed-forward**: `nu_ref = [ S·(R(yaw_ref)ᵀ·v_ref) ; 0 ]` (world-FLU path velocity → reference body
  velocity, FRD) — the MPC stops fighting the motion;
- **position preview**: `p_k = p_ref + v_ref·(k·DT_CTRL)` extrapolated over the horizon (position & velocity
  references consistent = a point moving at `v_ref`);
- **yaw unwrap**: `psi_ref ← psi_now + wrap(psi_ref − psi_now)` so the NMPC cost (no angle wrapping) always turns
  the short way across ±π — required for heading-follow on the MPC (else a ~270° wrong-way spin at one corner).

`v_ref = 0` reduces `_xref_ned` **exactly** to the old constant-pose tile (unit-asserted `allclose 1e-12`) →
**DP / station-keeping results are unchanged** (regression-safe). Frame care: `nu` is body (FRD), `v_ref` is
world (FLU); the rotation is mandatory. EAOB is unaffected (it sees the actual commanded wrench).

**Result (heavy, DP T=20 s / square 2 laps, mode C, seed 0):**
| run | radial RMS |
|---|---|
| DP dobmpc (regression) | **0.00 cm** (fully-actuated + DOB rejects the current) [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| square dobmpc (FF + heading) | **1.87 cm** |
| square mpc (FF) | 3.63 cm |
| square pid (baseline) | 25.1 cm |

The DOB-MPC square error drops from lag-dominated (tens of cm pre-FF) to **~1.9 cm**; the remaining error is the
corner transient (linear preview overshoots past corners) + the wave/current disturbance residual (EAOB's job,
separate from the reference FF).

## 2026-06-29 — NONE (still-water) baseline mode + comparison-figure polish

Added a **5th disturbance mode `NONE` = still water** (current + drift + waves all off) as the disturbance-free
baseline, via a new `use_current` flag in [disturbance/env.py](../disturbance/env.py):
`MODES = ("NONE", "C", "CD", "CW", "CDW")`. `water_velocity` / `external_wrench` now return exactly zero in
NONE. The mode codes go through PyYAML `safe_load`, so they must avoid YAML-1.1 bool/null words
(`N` / `n` / `off` / `yes` / …) — hence the spelled-out `NONE`. Wired into `config/base.yaml`,
`config/scenario_square.yaml`, and the `--mode` argparse choices of `run_viewer.py` / `plot_trajectories.py`.

**Why:** every other mode bundles a controller's *intrinsic* tracking lag with its *disturbance rejection*. NONE
isolates the first — it anchors the bar chart with the best each controller can do when nothing pushes the ROV.

**Result (square, 10 current headings, seed 0; recording `compare_20260629_113356`) — radial RMS [cm]:**
| mode | PID | MPC | DOB-MPC |
|---|---|---|---|
| **NONE (still water)** | **19.0** | **1.9** | **1.9** (std 0 — deterministic) |
| C (current) | 29.5 | 16.4 | 14.6 |
| CDW (+drift+waves) | 67.5 | 31.1 | 21.4 |

NONE ≤ C for every controller (as it must). In still water MPC ≈ DOB-MPC (the EAOB has nothing to estimate,
`est_err ≈ 0.04 N`, DRR 0.99); the PID's 19 cm is its pure corner-tracking lag with no disturbance at all. NONE is
deterministic (no current to rotate, fixed seed) → identical across all 10 headings, so its bar has no error bar.

**Figure:** `fig_bars()` in [experiments/run_compare.py](../experiments/run_compare.py) rewritten for a
publication-quality look — refined palette, per-bar value labels, two-line x-labels that spell out each mode
(`NONE (still water)`, `C (current)`, …), top/right spines off, horizontal legend, 200 dpi. The recording's
`bar_square_radial_rms.png` + `results.csv` / `results_raw.csv` were regenerated to include NONE (the existing
4-mode data was preserved, NONE appended).

## 2026-07-03 — PID retuned by analytic pole placement (heavy) + yaw-rate reference FF

Applied the adversarially-verified 2026-07-02 pole-placement design to [controller.py](../controller.py).
`DEFAULT_GAINS` is now **per-variant**: `GAINS_HEAVY` (new) vs `GAINS_BLUEROV2` (the 2026-06-14 hand-tuned set,
unchanged), selected by `rov_model.FULLY_ACTUATED` — the property that motivated the split, since the old 6 N
surge cap is a rank-5 pitch-coupling artifact that heavy (rank-6) does not need but bluerov2 still does.

**Design (heavy):** per-axis hover linearization `m_eff·a = F − d_lin·v`; match the closed-loop char poly
`m·s³ + (d+Kd)·s² + Kp·s + Ki` to `(s² + 2ζωn·s + ωn²)(s + αωn)`:

```
Kp = m_eff·(1+2ζα)·ωn²   Kd = m_eff·(2ζ+α)·ωn − d_lin   Ki = m_eff·α·ωn³
ωn = 2.0 rad/s translation (above the JONSWAP energy band 0.45–1.2), 3.0 yaw; ζ = 0.9, α = 0.2
```

→ horizontal **isotropic** kp/kd/ki = 131.6/90.6/38.7 (designed on the sway m_eff = 24.2 kg; body-x realizes
ωn ≈ 2.78, ζ ≈ 1.0), heave 141.8/99.1/41.7, yaw 8.95/4.32/3.95. Isotropy is a validity condition, not polish:
world-frame PD only commutes with yaw when kp_x = kp_y, and the primary scenario is square + heading_follow.
Companion changes shipped with the gains (the design does not hold without them): surge `f_max` 6→30 N,
`e_gate` 0.5→0.15 m, `surge_slew` 30→120 N/s, and a **yaw-rate reference feed-forward** — `set_target()` on
both controllers now takes `r_ref` and the PID yaw law is `mz = kp·e_yaw − kd·(r − r_ref)`;
[run_compare.py](../experiments/run_compare.py) / [run_viewer.py](../experiments/run_viewer.py) pass the
heading-slew rate (60°/s at corners, 0 on straights) so the D term no longer fights the commanded turn.
`DOBMPCController.set_target` accepts `r_ref` for interface parity (the NMPC tracks `yaw_ref` and ignores it).

**Result (heavy):** `--smoke` DP radial 1.4–1.5 cm (NONE/C/CD; wave modes 25 cm are 5 s-transient numbers);
1-lap square NONE via run_viewer: **radial RMS 1.34 cm** (t>5 s) vs **17.8 cm** with the old gains in  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
`compare_20260702_222150` — ~13× — max 4.0 cm (corner transient), |pitch| 0.0°, no saturation chatter.
`verify_meta.py` passes; `meta.json` sidecars stamp the new gain set. bluerov2 `--smoke` unchanged (legacy path).

**Validity:** sim-only (ideal thrusters, perfect 500 Hz state). For hardware derate ωn to 1.0–1.5 rad/s and
filter the D term. Beyond the |S(1.6)| ≈ 0.41 wave-band residual, the answer is DOB-MPC, not more PID gain.  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]

---

## 2026-07-07 — DOB-MPC yaw reference completed (yaw preview + yaw-rate FF on turns)

**Why.** On the square, DOB-MPC's largest CW/CDW error concentrates at the **(1,1) upstream  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
corner** (17 cm, seed 0 / current 0° / wave 0°), a cross-track sag ~0.36 m past the corner. Root
cause is a half-finished feature: the 2026-06-30 `_xref_ned` upgrade (constant-pose → moving-path)
added the **translational** feed-forward (position preview + linear `v_ref`) but never the
**rotational** half. The yaw reference was held CONSTANT over the horizon and the angular-velocity
reference was hard `zeros(3)`; the 2026-07-03 `r_ref` yaw-rate FF was wired into the PID only
(`DOBMPCController.set_target` accepted it "for interface parity … ignores it"). So the NMPC chased
a 60°/s-slewing yaw target as a zero-rate position regulator → slew lag (1.8 s vs 1.5 s nominal) and
~199° overshoot, opening a mis-heading window that the upstream-edge wave drag exploits.

**What.** `_xref_ned` now fills the rotation reference on turns, mirroring the existing linear FF:
- **yaw-rate FF** → `xref[11,:] = -r_ref` (world-FLU +yaw-rate → NED/FRD `r` via the `S=diag(1,-1,-1)`
  sign flip; Q weight 10);
- **yaw-angle preview** → `xref[5,k]` ramps from the current NED command yaw at the reference rate
  toward the final edge heading `yaw_target`, **clamped** at it (Q weight 150) so it never predicts
  past the corner (an unclamped constant-rate ramp over the 3 s horizon would extrapolate a 90° turn
  to 180°+ → over-rotation). `run_compare`/`run_viewer` now pass `yaw_target = atan2(ty, tx)` (the
  edge tangent). `set_target(yaw_target=None)` defaults to the current yaw → no preview (safe).
- **Gate:** `r_ref == 0` (straights / DP) is an exact no-op → DP/station-keeping byte-identical
  (new `test_dobmpc.test_xref_yaw_preview` asserts DP-equivalence + the `-r_ref` sign + the clamp).
- Runtime-only (`yref`); no acados rebuild. Weights unchanged.

**Result (A/B, dobmpc, seed 0, current 0° / wave 0°, run_viewer headless).** Yaw slew-to-5° at the  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
(1,1) corner **1.85 → 1.45 s** (≈ nominal) in both NONE and CDW — the lag mechanism is fixed. CDW
(1,1) top-edge **max 17.2 → 13.2 cm (−23%)**; overall radial RMS unchanged (CDW 3.13→3.16, NONE
2.08→2.07 — no regression; still-water per-corner peaks ±0.3–1.7 cm, since a no-wave corner
transient is dominated by the *position*-reference kink, not the yaw lag). The residual ~13 cm is
the **relative-speed wave drag (M2) + EAOB wave-band/frame limits (M3)** — out of this change's
scope (M1/M4 only). The documented deferred "yaw on turns" item (rotate ŵ to each predicted ψ_k =
M3b) is complementary and still open. Unit suite passes (heavy: frames/predictor/EAOB/yaw-preview;
bluerov2: full). See memory `dobmpc-corner-deviation`.

---

## 2026-07-12 — heavy_gripper variant: the real payload (Newton gripper + MarineSitu C3) as a third plant

Added `ROV_MODEL=heavy_gripper` = heavy + the lab's actual payload, **composed, not hand-tuned**: vendor-verified
masses (Newton Subsea Gripper 524 g air / 267 g water, w/ cable; MarineSitu C3 1700 g / 430 g — both from the
Blue Robotics store pages, cross-checked) fed through a parallel-axis composition
([compute_payload_inertia.py](../compute_payload_inertia.py)); the MJCF is *generated* from `bluerov_heavy.xml`
by [gen_gripper_variant.py](../gen_gripper_variant.py) so it cannot drift from the baseline. mass 13.724 kg,
I = [0.363, 0.749, 0.699], displaced volume 0.0131815 m³ → **net buoyancy −5.7 N (sinks — no trim foam, by
decision)**; hydro added-mass/damping stay the heavy set (payload increments are a few %–20% per axis, below the
30–100% spread across published BlueROV2 sets; DNV-RP-C205 build-up estimates documented in the YAML, HOLD until
in-situ system ID). Jaws are two mirrored slide joints driven by ONE `position` actuator at **ctrl index 8** —
thruster code indexes actuators by name (`thr0..7`), so allocation/teleop are untouched. Three onboard cameras
(stereo pair 7.5 cm + center) look forward-45°-down at the tag floor.

**Two hard lessons (both now regression-guarded in [test_heavy_gripper.py](../test_heavy_gripper.py)):**

1. **origin == COM is a stack-wide assumption, not a convention.** First build kept the heavy body frame and let
   the payload shift the COM 3.3 cm off-origin; PID (model-free + new rp-leveling) held fine, but the NMPC closed
   loop diverged in <1 s — the predictor (`fossen.py`, `params.ZG_MASS=0`) and hydro's force application all
   assume the body origin is the COM, and the unmodeled m·r rotation–translation coupling (0.45 kg·m vs
   Ixx 0.36 kg·m²) is not small. Fix: the generated frame is **re-origined at the composite COM** (all sites/
   geoms shifted by −COM; thruster geometry relative to the vehicle unchanged; allocation B picks up the
   physically-correct new moment arms automatically).

2. **hydro.py implicitly requires body_iquat = identity (diagonal inertia).** Emitting the composite as
   `fullinertia` (Ixz = −0.0016) made MuJoCo diagonalize and *sort* the principal axes — for this payload the
   inertial frame becomes an axis PERMUTATION of the body frame — and `mj_objectVelocity(mjOBJ_BODY)` reports ν
   in that permuted frame while hydro applies the drag wrench via `xmat`. Crossed drag axes turn the dissipative
   term into an energy pump: a torque-free 0.5 rad/s pitch kick exploded to |q| > 60 rad/s in 1.5 s. Isolated by  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
   ablation (buoyancy-only stable, drag-only explodes; heavy control stable). Fix: emit **diaginertia** (the
   0.4% Ixz is far below coefficient uncertainty); the deeper hydro fix is logged in KNOWN_ISSUES.

**Controller updates:** `GAINS_HEAVY_GRIPPER` re-derives the same pole placement at the payload masses (sway
m_eff 26.42 → kp/kd/ki 143.7/99.5/42.3; heave 28.29 → 153.9/108.0/45.3; yaw I_eff 0.811 → 9.92/4.79/4.38) and
adds an optional **roll/pitch leveling PD** (`rp_kp=(4.3,7.7)`, `rp_kd=(2.6,4.6)`, gated on the gains dict so
heavy/bluerov2 are byte-identical): the payload's static attitude torque (jaw weight + CB_x offset ~1.3 N·m)
rivals the passive B·coBM restoring (~1.2 N·m/rad), so passive-only attitude walks off. dobmpc params flow from
the registry automatically (MPC weights untouched — the earlier "calm the attitude weights" hypothesis was a
misdiagnosis of lesson 2 and was reverted).

**Results:** PID DP hold 0.0 cm (still water, 20 s); DOB-MPC DP hold rms(12–20 s) **1.3 cm** still / **1.4 cm**  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
current+waves / 22 cm with 20–50 N Poisson kicks (impulse-recovery transients — EAOB cannot predict kicks,
consistent with the C-mode analysis). heavy reference on the same harness: 0.4 cm (cw). acados-vs-IPOPT  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
worst-case |Δu| 0.2717 N (marginally over the heavy-calibrated 0.25 N gate; ~0.9% of authority — logged in
KNOWN_ISSUES). Full three-variant regression (selftest / load / thrusters / controller / hydro / observe /
water_viz) passes; heavy and bluerov2 files untouched.

---

## 2026-07-19 — heavy_gripper: C3 moved to its MEASURED mount (front-bottom, lens forward) from the lab's Onshape CAD

The C3 placement used since 2026-07-12 (front-top, pitched 45° down, `C3_POS=[0.18, 0, 0.09]`) was a guess and
was wrong. The real mount was measured end-to-end from the lab's Onshape assembly (BROV2 Heavy + C3 on its
C3-BR bracket): exported with **onshape-to-robot** (v1.8.2, MuJoCo output; export kept in
`assets/CAD files/onshape_export/`), then the exported vehicle geometry was **registered to the sim base_link
frame** with the rotation constrained to a pure axis permutation (the CAD sits axis-aligned; its bbox equals the
vendor 575×254×457 mm exactly), a global voxel-occupancy grid search over translation (no ICP local minima),  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
scale 1.0233 accounted for (the MarineGym-derived skin is uniformly 2.3% large; placement is TRUE METRIC anchored
at the COM), and a trimmed-ICP translation polish: residual 1.6 mm, seed spread <0.1 mm, fore-aft disambiguated
by the electronics-tube dome. Pipeline constants frozen in [process_c3_mesh.py](../process_c3_mesh.py), which now
emits `meshes/c3_payload_frames.json` consumed by the generator (drift-guarded against `CP.C3_POS`).

**Result** (base_link FLU, origin = vehicle COM): C3 mesh centroid `[0.199, 0.008, −0.156]` — **front-bottom on
the centreline**, housing protruding ~9 mm past the frame nose and ~3 cm below the skid line; optical axis
`[1.000, 0, 0.0056]` (0.32° up-tilt kept verbatim from the CAD mates); stereo baseline horizontal. The C3-BR
bracket straddles the Newton-gripper tube with ~1 mm clearance — independently confirming the guessed  [UNVERIFIED: 예시/유도값 — docs/MEASUREMENT_AUDIT.md]
`GRIP_POS=[0.25, 0, −0.17]` is compatible with the real bracket. The 3 MJCF cameras now sit at the **lens plane**
(`x=0.2395`, `y=0.0055±0.0375`, `z=−0.1554`) looking forward-level; the gripper jaws appear dead ahead at ~18 cm.

**Model consequences:** composite inertia diag `[0.38154, 0.77780, 0.70954]` (was `[0.363, 0.749, 0.699]`),
TOTAL COM `[+0.0349, +0.0010, −0.0258]`, coBM **+0.01625 m** (was +0.00955 — COM dropped, CB barely moved),
net buoyancy unchanged −5.7 N. The dropped off-diagonal grew from Ixz −0.0016 (0.4%) to **+0.064 kg·m² (16.8% of
Ixx)** — a real roll-yaw product the plant cannot carry until hydro reads body-frame velocity (KNOWN_ISSUES).
Bracket mass is NOT yet composed (visual-only; pending real mass). Bonus fix: the mesh visual-orientation helper
used `conj(mesh_quat)`; MuJoCo's convention is `v_orig = R(mesh_quat)·v_stored + mesh_pos`, so the geom quat must
be `mesh_quat` itself — the old bake's ~180° principal rotation masked this (conj(q) = −q). Verified: vertex  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
reconstruction error 0.000 mm; `test_heavy_gripper.py` all green; renders `assets/screenshots/c3/c3new_*.png`.

---

## 2026-07-20 — heavy_c3 variant: reflect ONLY the Onshape assembly (heavy + C3, no gripper)

The lab's Onshape assembly contains the BlueROV2 Heavy + the MarineSitu C3 on its C3-BR bracket — but **not** the
Newton gripper (it isn't in CAD yet). Added `ROV_MODEL=heavy_c3` to reflect exactly that, so the sim never models
hardware the CAD doesn't have. It is `heavy_gripper` minus the gripper: `compute_payload_inertia.compose_c3()` /
`buoyancy_c3()` compose vehicle + C3 only (bracket is visual-only, mass unknown), `gen_c3_variant.py` generates
`bluerov_heavy_c3.xml` from `bluerov_heavy.xml` (frame re-origined at the composite COM, diagonal inertia, C3 mesh
+ bracket + 3 lens-plane cameras, **no** gripper cylinder / jaws / actuator). Numbers: mass 13.2 kg,
I = [0.37014, 0.73153, 0.67460], COM [+0.026, +0.001, −0.020], displaced volume 0.0129237 m³ →
**net buoyancy −3.1 N (sinks)**; coBM +0.01372. With no articulated bodies the composite COM is the whole-vehicle
COM, so `<inertial pos>` = 0. Gains: `heavy_c3` inherits `GAINS_HEAVY` (fully-actuated fallback) — PID holds the  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
origin at 0.0 cm; without the roll/pitch-leveling PD (`heavy_gripper` has it) the C3's forward-low COM leaves a
~1° residual pitch (position hold unaffected). The dropped off-diagonal is Ixz +0.046 (12.4% of Ixx), same
body-frame-hydro limitation as heavy_gripper (KNOWN_ISSUES). Regression `test_heavy_c3.py` (composition /
no-gripper / thrusters-identical-to-heavy / sinks / cameras-forward / PID-hold) passes; heavy, heavy_gripper,
and bluerov2 untouched. Renders: `assets/screenshots/c3/heavy_c3_*.png`. `heavy_gripper` stays as the future config
for when the gripper is added to Onshape.

---

## 2026-07-20 (2) — payload-mesh orientation bug: MuJoCo composes the mesh reframe into the geom — no cancellation quat, ever

The user compared the sim against their Onshape front view and caught the C3 rendering ~90°-twisted (panel not
facing forward). Root cause — and a correction to the 2026-07-19 "bonus fix", which was itself wrong: MuJoCo
re-orients each mesh to principal axes internally, but **the compiler composes that reframe back into every
referencing geom's pos/quat** (verified numerically: with XML quat = mesh_quat, the compiled geom_quat came out
exactly mesh_quat⊗mesh_quat). So an XML geom pose applies to the mesh **as authored**, and a pre-baked mesh needs
**no quat at all** — which is also why the quat-less `rovc_*` skins always rendered correctly. Both prior
"cancellation" schemes (conj(mesh_quat), then mesh_quat) double-handled the reframe; the C3's near-square cross  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
section (95×89 mm) made the error a clean-looking 90° twist that survived eyeballing. Fix: quat-less payload
geoms in both generators + a build-time guard `_verify_mesh_geoms()` (compile the emitted XML, reconstruct
rendered vertices from the compiled geom pos/quat, assert bbox == baked STL at the intended position, ±1 mm) so a
mis-oriented payload can never ship silently again. Also re-exported from Onshape to double-check the assembly:
C3/mount poses byte-identical to the frozen constants — the user's mates are fine; the error was purely in our
visualisation chain. Housing/bracket now use the exact Onshape appearance colors (blue 0.231/0.380/0.706, gray
0.753). Renders match the user's front view (landscape screwed panel forward, connector to port):
`assets/screenshots/c3/heavy_c3_front_full.png`. Dynamics untouched (visual-only geoms); both variant suites pass.

---

## 2026-07-21 — MPC reference preview (tracking mode): the horizon now sees corners

Motivated by the n=200 corner analysis (compare_20260720_230025): the dominant routine corner deviation for
mpc/dobmpc — outside corner-rounding, corner/edge RMS ratio ~2.4–2.9× for dobmpc, present identically in NONE —
traced to the horizon reference being a straight-line extrapolation `p_ref + v_ref·k·dt` through corners
(`_xref_ned`), i.e. the NMPC never saw the turn coming. That extrapolation was an artifact of growing the tracking
reference out of the constant-pose DP tile behind the PID-parity `set_target()` interface; DP (point stabilization)
and trajectory tracking are different problems and now have separate reference modes.

**Change** — standard receding-horizon reference preview (textbook tracking-MPC form; same pattern as rpg_mpc's
`setReferenceTrajectory`, acados stage-wise `yref`): `DOBMPCController.set_reference_traj(fn)` takes a mission
sampler `fn(ts) → (p (3,K), yaw (K), v (3,K), r (K))` in world FLU; each 20 Hz tick fills the acados stage
references from `fn(t + k·dt)`, k=0..N (3 s), so a corner inside the horizon bends the position preview, rotates
the stage-wise body-velocity reference `S·R(ψ_k)ᵀ·v_k` onto the new leg, and ramps the per-stage yaw/rate FF. Yaw
is unwrapped stage-to-stage anchored at the measured yaw (short-way across ±π, bounded over 10 CCW laps by the
per-tick re-anchor). Mission side: `run_compare.make_square_ref()` — position/tangent are exact `square_setpoint`
evaluations; the heading profile is precomputed with the SAME `slew_heading` recursion as the live loop (no vehicle
feedback → knowable future), verified equal to <1e-9 over a full lap (`test_square_ref_matches_live_loop`). Wired
in `run_one` and `run_viewer` for mpc/dobmpc square runs only; PID, DP, and teleop keep the plain `set_target`
path (legacy body verified bit-identical via the constant-sampler test). No OCP/model/cost change — runtime
reference data only, no acados rebuild; the sharp (corner-infeasible) square reference is kept deliberately as the
benchmark stress test (P2 review: reference smoothing would change the task; observed rounding = the controller's
optimal smoothing). Provenance: run meta now records `controller.ref_preview` (all runs before 2026-07-21 are
False-equivalent — do not pool square mpc/dobmpc results across this boundary). P2 review (simulation-advisor +
control-theory-advisor): no wrong findings; applied their minimal fixes (meta provenance, sampler shape gate,
`reset()` clears the sampler). Expected effect (theory): corner transient shrinks several-fold and flips sign to
anticipatory inside-cut; solver failures same or fewer (tick-to-tick reference now changes smoothly instead of the
all-stages-at-once 90° flip at the vertex). The comparison experiments themselves are deliberately left to be
rerun by the user.

## 2026-07-21 (2) — roll/pitch leveling PD→PID, and the rank-5 surge shaping stripped from the heavy path

Two `controller.py` (`PoseController`) changes for the rank-6 heavy family. Analysis + adversarial verification via a
control-theory workflow (both claim sets returned CONFIRMED).

**(1) Roll/pitch leveling PD → PID.** The optional leveling loop (enabled per variant by `rp_kp`; on for
`heavy_gripper`) was pure PD, which is type-0 against a *constant* attitude torque: the asymmetric payload (forward
COM + jaw weight, ~0.2–0.5 N·m) leaves a steady tilt `φ_ss = τ/(kp + B_restore)`. Measured at a 0.40 N·m constant  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
torque: PD holds roll 3.53° / pitch 3.58°; that is ~cm-scale end-effector error at manipulation reach. Adding any
`Ki>0` makes the loop **type-1** so `φ_ss → 0` exactly (verified 0.00°/0.00°), independent of `Ki` magnitude and of
the exact `B_restore` estimate. Design = **option (a)**: keep the validated `kp/kd`, add only `ki = I_eff·α·ωn³`
(same analytic pole placement as translation, α=0.2, ωn=3): `I_eff` roll 0.481 → 2.60, pitch 0.859 → 4.64
N·m/(rad·s). Anti-windup mirrors the position/yaw loops — integrate `-angle` only when near level (`|angle| <
rp_gate = 0.15 rad`), clamp `|ki·I| < rp_i_max = (1.5, 2.0)` N·m. Verified the 3rd-order closed loop  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
`I·s³+Kd·s²+(Kp+B)·s+Ki` has all roots in the LHP with dominant pair ζ≈0.83–0.89 (the integral does **not** erode PD
damping; it inserts a slow companion pole at −0.67/−0.84 rad/s → residual nulled in ~4.5–6 s). Gains added to
`GAINS_HEAVY_GRIPPER` (`rp_ki`, `rp_i_max`, `rp_gate`). Gated on `self.use_i` **and** presence of `rp_ki`, so
`mode="pd"` and rp_kp-only variants stay byte-identical PD. Provenance: `heavy`/`heavy_c3` leveling was off and stays
off (unchanged); `heavy_gripper` PID-leveling runs before 2026-07-21 used PD leveling (no integral) — do not pool
attitude-accuracy data across this boundary.

**(2) Body-force shaping generalized on rank-6.** The surge-**only** slew-rate limit is generalized to a **uniform**
slew on all three body-force axes (finite actuator/command bandwidth; rarely binds at 120 N/s — ~66 N/s at hover).  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
The soft **pitch guard** (scales surge down above |pitch|>15°) is **KEPT** for the heavy family, gated on the
`pitch_guard_deg` gain key. Correction to the first-pass analysis (which removed it): the rank-6 `pinv` of the
full-rank 6×8 `B` only STATICALLY decouples `Fx` from `My` (pure `Fx → My=0`), but the migrated `test_controller` on
the heavy plant showed a **DYNAMIC Coriolis/added-mass coupling** still excites the weakly-restored pitch mode on an
aggressive far-offset slew — pitch transiently swings to ~70° WITHOUT the guard vs ~45–56° WITH it (recovering to  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
level either way). So the guard IS useful on heavy: a dormant transient-pitch limit that fires only above 15° (never
in normal small-error operation; on `heavy_gripper` the rp-PID leveling handles attitude first and the guard only
backs it up on large excursions). Amplitude clamp unchanged (already isotropic 30/30/30). Docstring corrected: the
heavy family is rank-6 fully actuated → all 6 DOF independently commandable (the old "NEVER pitch / rank-5
underactuated" note was bluerov2-only). `test_controller`'s pitch thresholds were recalibrated for heavy (its
stress-start transient exceeds the old rank-5-tuned 45°; the gate is now "never flips <90°" + "recover to level <8°")
and the PD-vs-PID current-headroom reduced (stiff heavy already shrinks the PD offset to ~1 cm).

**(3) `bluerov2` variant removed — heavy-only (2026-07-21).** The legacy rank-5 vectored-6 `bluerov2` was removed as
a selectable variant: dropped from `rov_model._MODELS`/`_POOL_WRAP`, from the gains (`GAINS_BLUEROV2` + the
`DEFAULT_GAINS` under-actuated fallback, now `→ GAINS_HEAVY`), and from the docs; `ROV_MODEL=bluerov2` now fails fast.
The three root smoke tests (`test_load`, `test_controller`, `test_square_mission`) were migrated to the bare-heavy
plant. `bluerov.xml` + `marinegym_assets/BlueROV.yaml` are **kept as the fixed reference fixtures** for the
hydro/thruster verify_*/test_* scripts (`verify_hydro`, `verify_hydro_precise`, `test_hydro`, `test_thrusters`,
`test_disturbances`, `verify_meta`, `verify_gpu_mjx`) and the `compute_heavy_inertia` provenance — deleting them would
break 8 scripts. **Deferred** (per user): the dobmpc `NU=4`/option-(b) surge→pitch dead path and `test_dobmpc`'s
rank-5 trim assertions, to be collapsed to NU=6-only later (see KNOWN_ISSUES). Regression: `test_load`,
`test_controller`, `test_square_mission`, `test_hydro`, `test_thrusters`, `test_disturbances`, `verify_meta`,
`test_heavy_gripper`, `test_heavy_c3` all PASS.

## 2026-07-23 — EAOB retuned for realistic sensors: sigma-derived covariances ("perf" profile) + sensor-noise injection; NIS gate built but default-off

Verify-first review of the EAOB tuning (external review claims checked line-by-line against this repo before
any edit), then a retune. Everything below was measured on the heavy plant, seed 0, 20 Hz, `verify/verify_eaob.py`.

**(1) Diagnosis — the old tuning was near-deadbeat, and only "worked" because the sim feeds the observer
noise-free ground truth.** Confirmed: no measurement noise exists anywhere in the marinegym loop (the EAOB gets
raw `xpos/xmat/mj_objectVelocity` + a clean tick FD as `nudot`; no `<sensor>` elements; the disturbance package
perturbs forces only). The covariances were unit-blind DT templates (`Q_pose=R=DT⁴/4`, `Q_vel=Q_dist=DT²`,
`R = 1.5625e-6·I₁₈` — the same variance for a meter, a radian and a Newton). Steady-state Riccati analysis of
the real (F,H) at hover: per-axis w-error time constants 6.8–13.4 ms (tau-channel Kalman gains −0.976…−0.999),  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
i.e. the disturbance state re-locks in ~one tick — a deadbeat inverse-dynamics decoder, not a filter. Under any
realistic sensor noise it would pass that noise straight into `w_hat`.

**(2) The retune ("perf" profile, now the default).** `params.py` gains a single sensor-sigma block
(`EAOB_SIG_POS/ANG/LVEL/AVEL/ACC/AACC/ALLOC`) that is the SINGLE SOURCE for both the sim-side injection and the
filter R. R is block-diagonal per channel; the tau pseudo-measurement noise is the accelerometer noise pushed
through the mass matrix, `R_tau = M Σ_a Mᵀ + Σ_alloc` — physics sets the N/N·m scale, no hand-tuning.
`q_dist` comes from the scalar steady-state inversion `K = 1−exp(−DT/τ_dist)`, `q = diag(R_tau)·K²/(1−K)`, so
**`EAOB_TAU_DIST` (0.2 s, see the iteration in (3)) is the only tuning knob**. The old tuning survives as `profile="verify"`
(wiring checks with clean measurements; byte-identical construction). `DOBMPCController` injects Gaussian noise
from the same sigmas into the EAOB's eta/nu/nudot (`meas_noise`, ON by default with perf; the MPC still consumes
the clean plant state, so the observer is evaluated in isolation). Run meta records
`controller.eaob = {profile, meas_noise, noise_seed, tau_dist, gate, n_upd, n_gated}` — records without this key
are verify/deadbeat + clean-measurement equivalent; **don't pool results across this boundary** (all compare
sweeps ≤ 2026-07-22 are pre-boundary).

**(3) Validation + the τ_dist iteration (`verify/verify_eaob.py`: w_true = model residual M·a+C+D+g−τ at clean
states, cross-checked against hydro's independent `diag_wtrue` diagnostic — the two agree to ≤0.08 N).** At the  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
initial 0.5 s design point the perf profile passed only where the w_dot=0 model holds (DP CD: NIS 14.8 PASS) and
failed under waves (DP CDW: NIS 25.4, NEES 77 — the wave band outruns the w random walk). Iterating the single
knob, gate OFF (DP CDW, T=60 s, seed 0): τ = 0.5 / 0.3 / **0.2** / 0.1 → NIS 25.4 / 19.8 / **17.1** / 14.8,
NEES 77 / 26.1 / **15.0** / 11.5, w-RMSE_X 1.11 / 0.74 / **0.55** / 0.48 N. **Default set to τ_dist = 0.2 s** —
the only value passing BOTH ranges in CDW; in CD it gives NIS 14.4 PASS, RMSE X/Y/Z 0.29/0.41/0.48 N. CD NEES  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
sits at ~9 (conservative, safe direction) because R/Q deliberately model real-hardware terms the ideal sim doesn't
produce (allocation mismatch, process floors) — tuning them down would fit the sim, not the robot, so they stay.
Slow-disturbance-only experiments can raise τ (CD RMSE halves again by τ≈2 s) at the cost of wave-band lag.
Measurement caution: an early sweep accidentally run WITH the gate enabled reported wildly inflated NIS
(137/67/43/27 for τ 0.5→0.05) — gate rejections starve the filter, which inflates the next innovations
(a feedback artifact); filter consistency must be measured gate-off. Motion-correlation diagnostic: residual
slopes vs |nu| ≈ 0 with |r| ≤ 0.18 everywhere in the final config (no model error leaking; the small CDW Z/M  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
correlation is wave-coherent motion, not drag mismatch). Corner NIS spikes in square runs are dominated by the
psi/r channels, consistent with the placeholder rotational added mass.

**(4) Chi-square innovation gate: built, measured, and turned OFF by default.** `update()` now computes NIS
per tick (`last_nis`, `last_zscore`, `n_gated`) and can reject frames at χ²(0.999,18)=42.31
(`params.EAOB_GATE_ON`). Closed-loop A/B (square, 2 laps, seed 0) killed it as a default: the wave band + corner
maneuvers violate w_dot=0 ROUTINELY, so a consistency gate rejects 25–76% of frames and blocks exactly the
updates that track the disturbance — CDW radRMS 4.53→15.45 cm (gate off→on), CD 1.35→1.43 cm (measured at  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
τ=0.5; at the final τ=0.2 the gate never fires in the same scenarios, 0/1067, so it is harmless but also
useless). A gate cannot separate "spike" from "wave" under a model that calls every wave an outlier; anti-spike
defense belongs to the per-axis `W_HAT_CLIP` (KNOWN_ISSUES, still open).

**(5) What the honest numbers cost (final config: perf, noise on, τ=0.2, gate off; 2-lap square / 60 s DP,
seed 0).** DP mode C: DOB-MPC keeps its headline DC rejection under realistic sensors — dc offset  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
(−0.11, −0.01) cm, radial RMS 0.33 cm, vs plain MPC +2.13 cm (the old 0.00 cm was the noise-free fiction).
Square: CD 1.02 (verify, clean) → 1.21 cm, CDW 3.09 → 3.51 cm — the honest price of vision-grade sensing is
≈0.2–0.4 cm. Regression: `test_frames`/`test_predictor`/`test_eaob_no_accel_doublecount` PASS under the new
default; `test_casadi_matches_numpy` fails pre-existing (NU=4 assumption, tracked in KNOWN_ISSUES).

## 2026-07-23 (2) — MPC state input switched from plant truth to EAOB estimates (mpc_state_source)

Closes the honest-evaluation gap left by the morning's retune: the NMPC now consumes the EAOB's eta_hat/nu_hat
instead of the clean plant state (dobmpc mode, perf profile default). Verify-first protocol; all numbers seed 0,
`verify/verify_state_source.py`.

**(1) The switch (single point, provably unique).** `x_ned` in `_control_step` was a local with exactly one
consumer (`nmpc.solve`, dobmpc_controller.py) — the audit found no other NMPC state entry. With
`mpc_state_source="estimate"` the solver x0 AND the yaw-reference anchor `_psi_ned_now` both follow the estimate
(they must share a branch: the NMPC cost has no angle wrapping and the reference re-anchors to `_psi_ned_now`
every tick). Everything else stays on the plant readout by contract: the EAOB's own measurements
(meas/a_meas/_nu_prev_ned), `_read_state` (verify_eaob depends on it as clean truth), and every logging surface
(recorder/mission/run_compare/run_viewer/eval_dp/teleop all read `data.*` directly). Defaults: "estimate" iff
mode="dobmpc" AND profile="perf" (params.MPC_STATE_SOURCE); plain "mpc" has no observer and verify-profile runs
are wiring checks — both stay "truth". Run meta records `controller.mpc_state_source` (absent key = truth;
2026-07-23+ = the third provenance key after ref_preview and eaob).

**(2) Yaw continuity needed NO unwrap accumulator.** The EAOB yaw state is already the continuous signal (it is
integrated, never re-wrapped; only innovations are wrapped) and NEITHER solver bounds yaw or position
(acados idxbx=[3,4,6,7,8]; IPOPT bounds only roll/pitch/|v|), so a multi-lap accumulated yaw (2 laps ≈ 4π) is
safe — verified: max per-tick |Δpsi_hat| ≤ 0.12 rad across all runs (a branch jump would be ~6.3). Side benefit:  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
the estimate path removes the ±π measured-yaw jump the RTI warm start used to see each crossing. Startup needed
no warm-up either: the EAOB lazy-inits AT the first reading, so the estimate starts exactly at truth (startup
max radial identical to the initial offset in every A/B pair).

**(3) A/B results (perf + noise, τ_dist=0.2, seed 0; 60 s DP / 2-lap square; truth → estimate).** Under the
sea state the tuning was validated for (Hs 0.75/Tp 12): DP C 0.33→1.46 cm (dc −0.11→−0.45), CD identical to C;
DP CDW 1.51→2.22 cm; square C 1.20→1.88, CDW 3.51→3.92 cm. Observer health is UNAFFECTED by loop closure
(DP CDW NIS 17.1→16.8, NEES 15.0→14.5 — all PASS; truth rows reproduce the morning's verify_eaob numbers
exactly). Chatter (mean Σ|ΔF| per tick): DP C 0.71→1.18 N (estimate jitter enters the loop, +66%) but square
CDW 2.43→1.32 N — the observer acts as a state FILTER during maneuvers, trading a little tracking for a
smoother command; no saturation, n_fail 0, all runs finite. Acceptance PASS (estimate NIS in 14–24, NEES ≤ 24;
the ~9 NEES in C/CD is the documented conservative direction, reported non-gating).

**(4) Discovered during validation — the CURRENT base.yaml sea state breaks CDW consistency for BOTH sources.**
config/base.yaml's wave block was strengthened at 2026-07-23 19:28 (Hs 0.75→1.2 m, Tp 12→6 s, γ 5→2, s 30→10,  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
ω_max 1.6→3.0): under it, CDW NIS = 80(truth)/71(estimate) on DP and 44/42 on square, radRMS 9.6→10.1 /
15.7→17.5 cm — the w_dot=0 + τ_dist=0.2 observer cannot follow the new faster/bigger wave band. This is
orthogonal to the state-source switch (both sources fail identically; the old-waves A/B passes everything) and
is logged in KNOWN_ISSUES; retuning τ_dist (or harmonic-EAOB) for the new sea state is a deliberate human
decision, not part of this task.

## 2026-07-23 (3) — DOB plug-in architecture: NMPC x0 = the state-estimator output ("meas"), EAOB hands over only w_hat

Reframes the morning's state-source switch to the classic DOBC topology, per the control-theory-advisor consult
(P1, journal 2026-07-23): the EAOB becomes a PURE disturbance plug-in — its filtered eta_hat/nu_hat are no
longer used by the controller; the NMPC x0 comes from the state estimator, and only w_hat crosses from the EAOB
to the MPC. Endgame rationale: the diffusion policy will observe the SLAM/AI estimator, so the tracker's x0 must
share that estimator (planner/tracker state consistency), and "MPC vs DOB-MPC" then differ ONLY by w_hat.

**(1) The switch (`params.MPC_STATE_SOURCE`, now three-way).** `"meas"` (new default under noise, for BOTH mpc
and dobmpc) = the estimator output the loop actually has: in sim the clean reading + Gaussian sensor noise (the
SAME per-tick sample the EAOB consumes, params.EAOB_SIG_*); on hardware the SLAM / AI-model estimate. `"estimate"`
(the 2026-07-23 (2) default, now opt-in) = the EAOB-filtered state (offset-free-MPC style). `"truth"` = clean
plant (regression / observer isolation; also the no-noise path). One sample feeds both the EAOB measurement and
the x0; the yaw-ref anchor follows the chosen source. Plain mpc now consumes the noisy x0 too (was truth) — a
deliberate change so the mpc/dobmpc comparison is single-variable; the mpc baseline shifts accordingly (run meta
`controller.mpc_state_source` marks the boundary; absent key = truth).

**(2) A/B result — stable, but raw-noise x0 costs heavy chatter (advisor's prediction, quantified).** perf +
noise, τ_dist=0.2, seed 0, old sea state (Hs 0.75/Tp 12 — isolates the switch from the base.yaml wave issue),
truth → meas → estimate:

| scenario | radRMS [cm] | effF [N] | chatF [N/tick] | NIS |
|---|---|---|---|---|
| dp C | 0.33 / 1.44 / 1.46 | 3.0 / 8.3 / 3.4 | 0.71 / **10.89** / 1.18 | 14.4 / 14.9 / 14.4 |
| dp CDW | 1.51 / 2.58 / 2.22 | 5.2 / 10.0 / 5.6 | 0.88 / **10.66** / 1.32 | 17.1 / 18.5 / 16.8 |
| square C | 1.20 / 2.06 / 1.88 | 4.6 / 9.5 / 4.6 | 1.19 / **11.81** / 1.18 | 15.5 / 17.3 / 15.4 |
| square CDW | 3.51 / 3.98 / 3.92 | 7.5 / 11.4 / 6.9 | 2.43 / **12.02** / 1.32 | 20.1 / 22.9 / 17.9 |

Tracking is essentially the SAME across sources (meas ≈ estimate radRMS), but "meas" chatter is **9–15× truth**
and ~9× "estimate", and control effort ~2–3×: the MPC chases the 5 cm/tick @ 20 Hz position jitter directly.
No divergence anywhere — finite, no saturation, n_fail 0, EAOB yaw continuous (max |Δψ̂| ≤ 0.08 rad; "meas"
reintroduces the ±π measured-yaw warm-start transient that "estimate" uniquely smooths, but RTI absorbs it).
NEES for "meas" marginally exceeds 24 in maneuvers (square C/CD 24.3/24.4; CDW 46.5, though truth is already 30.8
there = the known wave-model limit) — the chatter couples back through the plant into the estimation residual.

**(3) Interpretation — this is the "who owns the state smoothing" question, answered empirically.** The plug-in
with a RAW noisy estimator is the correct hardware-honest topology but pays a large actuator-chatter tax; the
EAOB-filtered "estimate" delivers identical tracking at truth-like chatter because it IS a state filter. Per the
advisor: on real hardware the SLAM+IMU FUSION owns the smoothing (its fused output is ~1–2 cm colored, far less
punishing than the sim's 5 cm white @ 20 Hz), so the sim's "meas" is pessimistic. This is the deliberate,  [UNVERIFIED: 예시/유도값 — docs/MEASUREMENT_AUDIT.md]
temporary starting point (user-chosen while the estimator — SLAM vs learned model — is undecided); mitigation
(a SLAM-proxy KF feeding both EAOB and MPC, MPC rate weights, or reverting to "estimate") is a separate human
decision and was NOT applied here. EAOB tuning (R/Q/P0/τ_dist) untouched, as constrained.

## 2026-07-24 — Sensor-noise sigmas reduced below the tracking error (tames the "meas" x0 chatter)

Follow-up to the 2026-07-23 (3) plug-in switch: with `mpc_state_source="meas"` the raw x/y 5 cm (1σ) pose noise
sat ABOVE the ~1.2 cm closed-loop tracking error, so the MPC chased its own measurement jitter (9-15× command
chatter). Reduced `params.EAOB_SIG_*` (pose/vel channels only) to a good visual-inertial/SLAM+DVL+depth stack:
x/y **0.5 cm**, z 0.3 cm, attitude 0.3° (yaw 0.5°), lin-vel 0.5 cm/s, gyro 0.11°/s. The pose noise now sits at
~0.4× the tracking error. `ACC/AACC/ALLOC` (the disturbance pseudo-measurement → R_tau, q_dist) are UNCHANGED,
so w_hat quality is preserved (verify_eaob CD w-RMSE 0.24/0.34/0.39 N, if anything slightly better than the  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
0.29/0.41/0.48 at the old noise). Because `EAOB_SIG_*` is the single source for BOTH the injection and the perf
filter R, R shrinks with the injected noise — the filter stays consistent-or-conservative by construction.

**A/B (gentle sea state, seed 0, truth → meas → estimate; the 5 cm → 0.5 cm change):**

| scenario | radRMS [cm] | chatF [N/tick] (meas) |
|---|---|--- [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| dp C | 0.22 / 0.26 / 0.31 | 10.89 → **1.76** (truth 0.70) |
| square C | 1.13 / 1.20 / 1.16 | 11.81 → **2.91** (truth 1.19) |
| square CDW | 3.19 / 3.26 / 3.26 | 12.02 → **3.08** (truth 1.93) |

"meas" tracking is now truth-level (was 4-5× worse), the DC bias leak is gone, and chatter drops to ~2-3× the
truth-x0 level (was ~15×). State-estimate RMSE falls to pose ~1.6 mm / vel ~2.7 mm/s. Consistency: NIS/NEES now
sit ~13/9 (below the ~18 target) — the SAFE, conservative direction (the filter under-trusts itself; it is never
over-confident). The verify gates (verify_eaob, verify_state_source) were changed to gate on OVER-confidence only
(value ≤ 24), reporting the low side as conservative, so a legitimately low-noise config no longer spuriously
fails. Run meta now stamps `controller.eaob.sig_pos_xy` (0.005) + `mpc_state_source` — the 5 cm-era meas-noise
runs (sig_pos_xy 0.05, all internal A/B only) are distinguishable. NOTE: in ultra-tight DP hold the truth error
(0.22 cm) is itself below the 0.5 cm pose noise; the "< tracking error" target is met for the operational square
mission (~1.2 cm). Going to 0.25 cm would push chatter to the truth floor if desired (a sensor-spec choice).

## 2026-07-24 (2) — PID now consumes the SAME noisy state as the MPC (apples-to-apples comparison)

Until now the compare sweep gave the PID **ground-truth** state (`data.xpos/xmat/qvel`, 500 Hz) while both the MPC
and DOB-MPC ran off a **noisy 20 Hz** estimate (`mpc_state_source="meas"`). That asymmetry confounded every
PID-vs-MPC number: the analysis of the 2026-07-24 wave sweep had to prove *indirectly* (NONE control group +
DOB-MPC counterfactual) that state noise was not what made the MPC lose in C/CD — because the PID was silently
handed a better input. New `PoseController(meas_noise, noise_seed)` closes that gap: with `meas_noise` on the PID
SEES the clean plant state corrupted by the **same** Gaussian sensor model the MPC x0 uses (`params.EAOB_SIG_*`:
pose 0.5 cm / z 0.3 cm, attitude 0.3°/yaw 0.5°, lin-vel 0.5 cm/s, gyro 0.11°/s), **sampled at the control tick
(20 Hz) and held (ZOH)** between ticks — so the PID and the (DOB-)MPC eat an identically-degraded estimate and
differ ONLY by the control law. The noisy attitude is rebuilt into a full R (`_R_from_rpy`, the exact inverse of
the yaw/pitch/roll extraction) so both the angle read AND the body-force rotation `R.T` use the estimated
attitude consistently. `meas_noise` bundles the two coupled sensor effects the MPC already had: Gaussian noise
AND 20 Hz sampling lag. The logged truth (`px,py`) and the radial-RMS metric always stay on the plant, so the
metric still measures TRUE tracking error.

Default `params.PID_STATE_SOURCE = "meas"` (symmetric with `MPC_STATE_SOURCE`), so the sweep is apples-to-apples
out of the box; `"truth"` is the opt-in clean-state baseline. Interactive/eval tools (teleop, eval_dp) inherit
`meas` — consistent with their already-noisy MPC — while the four deterministic control-law regressions
(test_controller/test_square_mission/test_heavy_gripper/test_heavy_c3) pin `meas_noise=False` so they keep testing
the control law on clean state. Run meta now stamps `controller.meas = {state_source, noise_seed, sig_pos_xy}`
(absent key = truth); **PID records made before this date used truth state and are NOT comparable** to new
meas-state PID records — rerun the sweep to compare.

**A/B (heavy, seed 0, 3-lap square, truth → meas x0):**

| ctrl / mode | radRMS [cm] | dc [cm] | slew Σ\|dU\| [N/tick] |
|---|---|---|---|
| pid  NONE truth | 1.49 | 0.08 | 0.85 [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| pid  NONE meas  | 1.47 | 0.10 | 3.27 |
| pid  C    truth | 1.61 | 0.08 | 0.85 |
| pid  C    meas  | 1.58 | 0.10 | 3.24 |
| mpc  C   (meas) | 3.17 | **2.87** | 11.1 |

The key result is now shown DIRECTLY rather than inferred: even fed the identical noisy 20 Hz state, the PID keeps
its C-mode DC offset at ~0.1 cm (its integrator averages out the zero-mean noise) while the nominal MPC sits at
2.87 cm — so the C/CD gap is the **missing integral action**, not state quality. Noisy state costs the PID only
command chatter (slew ×3.8, still below the MPC's because the PID's 120 N/s slew limiter de-facto filters the
20 Hz noise steps); radRMS and offset are essentially unchanged. `pid/NONE/truth` reproduces the recorded sweep
`radial_max` 4.402 cm byte-for-byte, confirming `meas_noise=False` is the exact pre-change path. Verified:  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
`_R_from_rpy` round-trips to 4e-16; injected σ measured 5.24 mm / 4.82 mm·s⁻¹ / 0.482° vs the 5/5/0.5 targets;
refresh cadence 25 substeps (20 Hz), held within a tick.

## 2026-07-24 (3) — NMPC surge input box 8 → 30 N on heavy: the benchmark was not like-for-like

Investigating why the symmetric-state sweep showed PID beating the MPC in strong waves and losing in weak ones,
the cause turned out **not** to be a control-theoretic property but the NMPC's own hard input box. On heavy
(rank-6) `params.U_MAX` was `[8, 30, 30, 8, 8, 10]` — surge capped at **8 N while the PID's `f_max` is 30 N on
every axis**, a 3.75× handicap. The 8 N figure is a rank-5 relic: on bluerov2 the surge→pitch coupling
`My = −0.0725·Fx` tumbles the vehicle past sin θ = 1 at Fx ≈ 15 N, so 8 N is a real physical limit there; on
heavy the four vertical thrusters cancel that coupling. The old comment claimed 8 N was kept "for a like-for-like
compare", which was exactly backwards. The PID's identical relic (a 6 N surge cap) had already been removed on
2026-07-03 during the pole-placement redesign ("would bind ωn to ≤0.54"); the MPC's was left in place.

**How much it mattered** (storm CW, 10 laps, 4 paired headings, runtime box override, everything else identical):

| | PID | MPC box 8 N | MPC box 30 N |
|---|---|---|---|
| storm  | 26.32 | 30.55 (box active 60–66 % of ticks) | **17.72** (0.7–1.8 %) [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| gentle | 12.51 | 8.50 (5–9 %) | 7.64 (0.2–0.3 %) |

The box was active on **60 % of 20 Hz ticks at storm (dobmpc 75 %) vs 7 % at gentle** — an amplitude-dependent
throttle, which is why the *same* frequency band flipped sign with sea state (0.85–2.30 rad/s MPC/PID error ratio
1.38 at storm → 0.51 at gentle; no LTI bandwidth argument can do that). Releasing it moved storm PID−MPC from
**−4.24 cm to +8.60 cm** while gentle barely moved (+4.01 → +4.87) — i.e. **the entire "PID wins in strong waves"
crossover was this one constraint.** The recorded `sat_freq` metric is blind to it (0.0000 everywhere) because
8 N is a software bound, not thruster saturation.

**Applied:** heavy `U_MAX = [30, 30, 30, 8, 8, 10]`; the rank-5 branch keeps 8 N (the coupling is real there).
Validation with the rebuilt solver (storm/gentle CW, 10 laps, 3 paired headings):

| sea state | PID | MPC | DOB-MPC | max thruster | sat_freq |
|---|---|---|---|---|---|
| storm  | 26.14 | **17.86** (was 28.82) | **7.30** (was 17.36) | 40.3 N | 0.0000 [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| gentle | 11.56 | **7.63** (was 9.51)   | **1.87** (was 3.14)  | 36.1 N | 0.0000 |

MPC now wins at **both** ends (PID−MPC +8.28 storm, +3.93 gentle) — the crossover is gone — and DOB-MPC gains the
most (2.4× better at storm), since it was the most authority-starved. Still no physical saturation: peak
per-thruster demand 40.3 N against a T200 `ctrlrange` of −51.6…+64.1 N, and at a fixed heading the 30 N box
actually *lowers* peak thruster force (36.1 vs 36.9 N) because adequate authority prevents the violent late
corrections the 8 N box provoked. `verify_acados` still passes (acados↔IPOPT max|Δu| = 0.0615 N < 0.25 N gate,  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
1.04 ms median, n_fail = 0); closed-loop n_fail ≈ 1 per 5334-tick run, unchanged.

**Records boundary:** run meta now stamps `controller.u_max` (both `run_compare` and the teleop manifest, read off
the live solver). **Absent key = the old 8 N box.** Every mpc/dobmpc result recorded before this date — including
the 20260724 sweeps — is under 8 N and must not be pooled with new runs; in particular "PID beats MPC in strong
waves" from those records is a benchmark artifact and must not be cited. A full sweep rerun is pending
(KNOWN_ISSUES).

**What survives as real control findings** (unaffected by the box): the MPC's error PSD is 3–8× below the PID's at  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
every frequency under 0.7 rad/s in *every* sea state — its Fossen model plus 3 s reference preview — which is also
why it wins the disturbance-free NONE baseline (1.089 vs 1.492 cm). And in constant-current C mode the nominal
MPC still parks 2.70 cm downstream for want of integral action (PID 0.03 cm), the gap DOB-MPC's `w_hat` closes.
**Not established:** that the ladder's crossover was a *frequency* phenomenon — the sea-state ladder co-varies Hs  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
and Tp (corr(log Hs, log ω_p) = 0.967) plus γ and s, so a regression cannot separate them (substituting the
equally-collinear γ reverses the ranking, and the Hs coefficient is not significant on 6 rungs, p = 0.22). The
ablation settled the mechanism directly instead of the regression.

## 2026-07-28 — One steady window for every reported RMS (trajectory boxes == bar chart)

Two figures from the *same* 100 runs disagreed: `bar_square_radial_rms.png` put PID/CDW at **18.6 cm**, while the
boxed number on `trajectory_compare_CDW.png` read **18.2 ± 2.4 cm**. Not a bug in either number — two different
windows. `metrics()` scores the steady window (`t ≥ experiment.settle_s` = 10 s, clamped to half the run), which
is what lands in `results.csv` / `results_raw.csv` / every bar; the trajectory panel re-read `runs/traj_*.csv` and
scored the **whole** record. A square run starts on the reference with near-zero error, so those first 10 s — only
3.7 % of the 266.7 s record (196 / 5268 samples) — are much quieter than the rest and pull the pooled RMS down:

| controller (CDW) | first 10 s | after 10 s | pooled (old box) | steady (bar) |
|---|---|---|---|---|
| PID | 4.47 | 18.57 | 18.24 | **18.57** |
| MPC | 6.13 | 10.69 | 10.55 | **10.69** |
| DOB-MPC | 1.92 | 2.62 | 2.60 | **2.62** |

**Applied:** the window is now defined once — `run_compare.steady_mask(t, settle_s)` (+ `settle_start(cfg)`, the
single read of `experiment.settle_s`) — and called by both `metrics()` and `_draw_traj_panel`. `settle` is threaded
into `fig_trajectory_compare` / `fig_trajectory_allmodes` / `fig_xwave_trajectory`; every figure now *names* its
window in the caption (`fig_bars` grew a `note=` footnote, the trajectory legends read "lines = all laps · boxed
RMS = steady-window radial RMS (t ≥ 10 s, same metric as the bar chart)"). The standalone
`experiments/plot_trajectories.py` mirrors the same mask behind a `--settle` flag, so pointing it at a
`run_compare` `runs/` folder reproduces that run's `results.csv` numbers instead of a third value. The trajectory
*lines* still draw the entire run — only the scored number changed.

**Verification** (no sim re-run): re-rendered the `20260727/compare_20260727_000850/wave01_moderate` figures from
its existing per-run CSVs — all 15 (5 modes × 3 controllers) boxed values now equal `results_raw.csv` to  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
≤ 1.6e-6 cm, the `%.5f` m write precision of those CSVs. `plot_trajectories` on one run prints 17.26 / 10.88 /
2.79 cm against that run's recorded 17.26 / 10.88 / 2.79. Full `--smoke` pipeline green.

**Records boundary:** no recorded number moves — `results.csv`, `results_raw.csv`, `results_all_waves.csv` and the
bar charts were always the steady-window metric and are unchanged. Only the *trajectory-figure boxes* changed.
Figures generated **before this date** box a full-run RMS, ~1–2 % below the matching bar; read their caption
("full-run radial RMS (all laps)") or regenerate them.


<!-- MEASUREMENT AUDIT (2026-08-04): the following numbers appear inside code
     blocks above and could not be annotated inline without corrupting them. -->

> **[UNVERIFIED]** 아래 줄의 수치는 출처 감사에서 미검증으로 분류되었다 — 자세한 근거는 [docs/MEASUREMENT_AUDIT.md](docs/MEASUREMENT_AUDIT.md).
>
> - `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:130` — 산출물 없음 — `(both coefficient sets recovered back out of the sim to 0.00 %, T4.3)`
> - `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:139` — 산출물 없음 — `└−a₂v   a₁u    0   −a₅q   a₄p     0   ┘   1e-14 (T1.1–1.2)`
> - `bluerov2_mujoco_marinegym/docs/CONTROL_METHODOLOGY.md:704` — 예시/유도값 — `Heavy:     6×8,  rank 6   — FULLY ACTUATED (verified: a pure pitch wrench realiz`

## 2026-08-12 — The DOB-MPC leaves the simulator: closed-loop AprilTag MPC in rov_gui (`--mpc`)

The controller stack this document describes now has a REAL-VEHICLE consumer: `rov_gui/control/`
(the topside station's new closed-loop package) imports `dobmpc/` **verbatim** — params, fossen,
frames, eaob, mpc, mpc_acados, none of which touch MuJoCo — and replaces only the sim wrapper:
`_read_state` becomes AprilTag PnP off the C3 colour stream + ArduSub ATTITUDE/SCALED_IMU2
(`state_assembler.py`), and `set_wrench_command` becomes MANUAL_CONTROL axes through the station's
existing command sink (`allocation.py`; ArduSub in MANUAL, K/M dropped — no roll/pitch axis exists).
The reference logic (`set_target` / `set_reference_traj` / `_xref_ned` / `_xref_ned_traj`) is ported
line-for-line from `dobmpc_controller.py:126-285` into `rov_gui/control/mpc_bridge.py`; the square
generators are byte-copies of `run_compare.py:114-177` pinned by tests.

Decisions that differ from the sim wrapper, on purpose:
* **x0 is always "meas"** — the hardware has no truth branch; the DOB plug-in architecture
  (2026-07-23 entry) carries over unchanged: the EAOB hands the NMPC only w_hat.
* **tau_applied fed to the EAOB is the post-allocation estimate** (axis caps applied, K/M zeroed),
  not the raw solver output — otherwise the cap/drop mismatch double-counts as disturbance.
* **w_hat is clipped per-axis [15,45,45,5,5,8]** in the bridge before the solver — the KNOWN_ISSUES
  replacement for the scalar ±50; the sim code keeps its backstop untouched.
* Solver failures now log timestamp + status + |w_hat| every occurrence, and `solve_ms` is read
  live from acados (the sim wrapper's dead field is bypassed).

Verified 2026-08-12, bench only: acados SQP-RTI builds and runs in the GUI's numpy-2 env
(`rov_gui/control/smoke.py`: build 1.3 s, EAOB+solve p50 3.1 ms / p99 6.4 ms, n_fail 0/100);
the demo backend closes the full loop (engage → DP warm-up → 1 square lap → complete → DP → release)
with zero solver failures. Water status and the remaining [예측] items (axis gains, EAOB sigmas,
wall-tag preset) are tracked in the repo-root KNOWN_ISSUES.md entry of the same date.

**Records boundary:** hardware CSVs (`*_mpc.csv`) share the sim's 9-column prefix but are a NEW
population — `meta.json` carries `source: "hardware rov_gui.control"`, the tag-map sha1 and the
axis-gain provenance; never pool them with `runs/traj_*.csv` sim records.

## 2026-09-26 — 6-DoF variant: the policy emits roll/pitch and the NMPC tracks them (OFF by default)

A NEW variant selected by config + checkpoint, not a replacement. The diffusion policy may emit
`[dx, dy, dz, dyaw, droll, dpitch, width]` (`action_repr: pos_rpy_width`, width 7; label encoder
`umi/common/yaw_action.py::encode_pos_rpy`, ZYX Euler of `R_bt R_rel R_bt^T`, the dyaw column
bit-identical to the 5-dim label). The station transcribes the decoder in `policy_frames.py`
(parity-tested by path), `compose_plan(track_rp=True)` carries an ABSOLUTE NED roll/pitch per knot
(`PlanMsg.rp`, T1-clipped at `policy.rp_max_deg`), the filter gates magnitude / Euler rate / jump,
the stitcher samples attitude + rate (`sample_att`), and `HwDobMpc._xref_ned_plan` fills
xref[3:5] and the body-rate rows xref[9:11] (+ xref[11]) through `T(phi,theta)^-1` only when a plan
carries rp. The K/M torques reach ArduSub through MANUAL_CONTROL's extension axes (`s` = pitch,
`t` = roll, `enabled_extensions=0b11`; MAVLink 2 only) behind the engage gate `engage.attitude_axes`,
scaled by `axis_gain.roll_nm 13.2 / pitch_nm 7.2` [유도] and capped at 0.2 / 0.3 [예측]. The sim
wrapper (`dobmpc_controller.set_target(roll_ref, pitch_ref)`, 6-tuple sampler) got the same
reference path; `verify/verify_attitude_hold.py` runs the hold grid + step/ramp/dropped/sign-inversion/
gain/deadband/headroom scenarios.

Decisions worth keeping:
* **Variant OFF = byte-identical 4-DoF path.** Golden sha256 pins on `compose_plan` / `PlanFilter` /
  `PlanStitcher` (test_policy_frames, test_plan_stream), bitwise `_xref_ned` / `_xref_ned_plan` tiles
  vs a frozen legacy copy (test_attitude_axes, sim tests/test_dobmpc.py), 23-byte MAVLink frame
  equality, and the resolved `config/hw_mpc.yaml` compared key-for-key with the variant keys stripped.
* **Hold ramp, not a step, when rp disappears** (`_rp_hold` decays at `pq_max_rad_s`·DT per tick).
* **Rate rows through T^-1**: on a 0.35 rad/s ramp the level-rate reference lags 3.105 deg peak vs
  0.287 deg with the T^-1 rows; ramp peak error 7.252 deg
  (`bluerov2_mujoco_marinegym/recordings/20260926/attitude_hold_211921/results.csv`, rows `rate_*` / `ramp_*`).
  Two methods, two labels, **never pooled under one `rate_source`**: the sim's `attitude_meta.rate_source`
  is `fd_horizon_T_inv` (np.gradient over the sampled horizon); the station's `HwDobMpc` label is
  `stitcher_rate_T_inv` (the stitcher's sampled Euler rate through T^-1).
* **Steady window starts at t_step + settle = 20 s** (`steady_from_s=20`): step 15 deg, plain mpc, e_pitch
  RMS 0.0884 deg, overshoot 0.9991 %, 1-deg settle 1.0 s, accept 1
  (`recordings/20260926/attitude_hold_211813/results.csv`, row `step,mpc`). The earlier
  `attitude_hold_174607` step/ramp/rate/dropped rows are NOT citable for steady values (their window
  contained the transient; transients unchanged).
* **dobmpc stays gated (`dobmpc_allowed: false`)**: with a 2x torque-gain error the EAOB saturates the
  verticals (234 ticks, max|f| 52 N) where plain MPC shows +0.22 deg steady error
  (`recordings/20260926/attitude_hold_174607/results.csv`, rows `gain_*`); with K/M silently dropped
  (reference 20 deg = `policy.rp_max_deg`, threshold 15 deg = `policy.div_max_rp_deg`) both mpc and dobmpc
  trip `div_rp` at 0.45 s — the tick where the over-threshold streak reaches 10 samples = the station's
  0.5 s debounce; the hull never approaches 20 deg — and dobmpc drifts to 6.83 deg pitch through the
  credited-but-undelivered K/M, rad_max 14.5 cm
  (`recordings/20260926/attitude_hold_211917/results.csv`, rows `dropped_*`). The 174607 `dropped_*` rows
  had threshold = reference size and are degenerate — do not cite them.
* Heavy hold grid (mpc, dobmpc; NONE): e_roll/e_pitch RMS 0.07–0.08 deg, K/M 0.29/0.30 N·m at 15 deg,
  sat_ticks 0 (`recordings/20260926/attitude_hold_174544/results.csv`). heavy_gripper (0,20) deg: plain
  mpc steady pitch error −0.15 deg (+0.29 deg at level) vs dobmpc −0.04 deg, i.e. the predicted ZG·W vs
  B·coBM steady tilt is confirmed in sign only, sub-degree because Q_att 80 closes most of it
  (`recordings/20260926/attitude_hold_174602_gripper/results.csv`).
* **Cap tiers, outermost last** [예측 unless cited]: T0 label range (normaliser + DDIM clip) → T1
  `compose_plan` clip `rp_max_deg` 20 → T2 PlanFilter (`rp_reject_deg` 30 hard, `pq_max_rad_s` 0.35
  dilation, `rp_jump_max_deg` 5 soft) → T3 `set_path_plan_ned` ValueError above `rp_reject` → T4 wire
  caps `cap_roll 0.2 / cap_pitch 0.3` (`first_water_caps [0.1, 0.15]` for the first water session).
  The label statistics behind T1/T2: |dpitch| p99.9 16.7–18.6 deg, max 21.1–27.2 deg; |droll| p99.9
  7.8–9.3 deg (`data/20260926/rp_label_stats_174358.txt`, `rp_label_stats_174535.txt`); peak Euler rate
  |p|,|q| p90 0.453 rad/s (max over both axes) on the raw 66.7 ms grid, so the 0.35 rad/s rate gate dilates 2/1721 chunks at
  200 ms (`data/20260926/0926_175145_offline/gate_pos_rpy_width.log`).
* **MANUAL only.** In STABILIZE the firmware reinterprets s/t as lean-angle targets and runs its own
  attitude loop, so the station's K/M torque would be a second controller on the same axes; STABILIZE
  engage is refused whenever `attitude_axes.enabled` (the lean-angle cascade is a scope trim).
* **Ordering: plain mpc → torque-gain calibration → dobmpc.** `roll_nm 13.2 / pitch_nm 7.2` are derived
  from heave_n × lever arm [유도], never measured; the first water step is a policy-less plain-mpc
  attitude STEP that fits the gains, and only then may `dobmpc_allowed` be considered, because the EAOB
  credits a torque the wire does not carry (the `gain_*` / `dropped_*` evidence above).

Records: mission CSV gains trailing `rroll_deg, rpitch_deg, ax_roll, ax_pitch, rp_track`
(nan/0 when off), `policy_plan.csv` ends `,reason,roll_deg,pitch_deg`, meta schema 15 → 16 with
`run.attitude_axes`, `trajectory.attitude_track`, `policy.action_repr`, `controller.attitude_ref`.
Scope trims: `transport: rc_override`, the STABILIZE lean-angle cascade, the PID roll/pitch loop,
the run_compare attitude scenario and the armed in-water sign-probe mission are NOT in this cut.
**Zero hardware runs**: firmware s/t consumption, the K/M sign and the torque gains are unconfirmed
(KNOWN_ISSUES.md 2026-09-26).

## 2026-10-01 — A DVL becomes the estimator's third sensor: AprilTag + IMU + DVL (OFF by default; no DVL mounted yet)

An addition behind three switches, not a replacement. **Written before the hardware exists**: against the
Water Linked DVL A50 protocol pages and a 12 s capture of the vendor's public demo unit
(`rov_gui/tests/fixtures/dvl_a50_demo_capture_20261001.jsonl`); nothing below is a measurement on this vehicle.
Revised 2026-10-02 after a three-reviewer pass (safety / correctness / estimation theory); the points it changed
are marked ⟲. Revised again the same day after a timing simulation and three more independent passes, every
finding re-checked by a skeptic before it was acted on (round 2: four reviewers, 28 confirmed; round 3, on the
fixes: 9 confirmed; round 4, on those fixes: 8 — its skeptic did not run (session limit) and the main session re-ran
the scripts instead; round 5: 3 from the worker reviewer, plus 3 more from the other reviewer's interim runs —
neither round-5 skeptic finished, so the main session reproduced or derived each before acting; round 6, one
reviewer on the round-5 fixes: PASS_WITH_FIXES, 1 medium fixed + 1 low documented, intent-A parity re-measured at
0 differences incl. a fed IMU rung; none critical in any round); those points are marked ⟲⟲.

* **What changes.** Only the SOURCE of the HORIZONTAL body velocity. `StateAssembler.step(..., vel_xy=None,
  disp_xy=None)` (`rov_gui/control/state_assembler.py`): with a `vel_xy` the measured (surge, sway) replaces the
  low-passed tag finite difference as `nu[0:2]` and in the `nudot` transport term. Position (x, y, yaw from tag PnP),
  z, roll/pitch and the rates are untouched. `vel_xy=None` is the pre-DVL arithmetic, operation for operation
  (`test_dvl.py::test_without_a_dvl_the_assembler_is_bit_identical_to_before`, float64 `==`). This is the operator's
  choice of structure (2026-10-01): a velocity source in the existing assembler, not a new EKF. The EAOB's
  measurement sigmas are NOT changed with the source.
* ⟲ **The heave rate is not consumed.** `nu[2]` stays on the tag finite difference. The proof below sees surge and
  sway only, so a tilted bracket (a pitch error e only scales one row by cos e — 20 deg passes), a wrong vz sign
  convention or a wrong lever arm would otherwise put an unverified number into the heave loop with every gate
  silent. The DVL's w is recorded (`dvl_w`) for the day a vertical check exists.
* ⟲ **Fix-age bridging is an integral.** `p = p_tag(t_capture) + R · ∫ v_dvl dt` over the DVL history
  (`DvlAid.displacement`, piecewise-linear), not "newest velocity × age" — which is not an integral once the
  velocity keeps updating under an ageing fix (at a kick the estimate ran ahead, then back).
* **Why not an EKF.** The paper that motivated it ("Learning to Swim", arXiv:2410.00120v2) fuses AprilTag + DVL +
  IMU in an EKF and gives no state vector, noise or delay handling; its platform paper (CUREE, arXiv:2303.03943)
  says a Waterlinked DVL and an IMU give dead reckoning that an EKF fuses with global fixes. The role split is what
  transfers: tag = absolute position, DVL = direct velocity + the bridge when the tag is gone. An EKF tuned with no
  data would be all [예측] and would change the DVL-off path that every recorded run was flown on.
* **Three switches.** `dvl.enabled` / `--dvl` builds the TCP client (off = nothing constructed). `dvl.use` / the
  trajectory panel's DVL button decides whether the estimator CONSUMES it; OUT = shadow (received, gated, compared,
  recorded; commands bit-identical to no DVL). The button changes it only while disengaged and with no controller
  CSV open (operator decision: one run = one estimator). Third, per tick and automatic: a sample that fails a data
  gate makes THAT tick fall back to the tag finite difference.
* ⟲ **Automation may only take the DVL OUT.** (1) Whether an ENGAGEMENT may use the DVL is decided once, at engage
  (`_dvl_arm_run`: switch IN, proof satisfied, not tripped) — a proof that completes mid-run counts from the next
  engagement. (2) Evidence that the DVL is WRONG is a TRIP, latched until the operator sets the switch OUT then IN
  (which the proof gate must allow), and independent of `require_proven` (round 1 found the opt-out silently
  disabled the in-flight monitor too). (3) A missing lock / stale link / single outlier is not a trip; the DVL
  returns by itself after `reentry_s`. ⟲⟲ (4) Once this session's data has said the DVL is wrong — a trip, or a
  refuting fit seen in shadow — IN needs a proof made AFTER that, also under `require_proven: false`: the opt-out
  waives the proof a DVL with nothing against it would need, not the one evidence demands (the audit re-armed a
  tripped DVL one second later on nothing, the trip having deleted its own evidence). A trip discards the proof and
  its pairs; so does a change of the device's configuration in EVERY state (in shadow 59 old pairs + 1 new one
  completed a proof of a stream that no longer existed), and a proof re-made under a latch falls to the next
  evidence like any other (the first cut guarded every trip with "not already tripped"). (5) A trip needs the DVL
  to be wrong in a way that LASTS, and there are two such ways: the MOUNT — the tag velocity SEES the motion the
  DVL reports (at least 80 % of it, `COHERENT_MIN`) and puts it elsewhere (alignment monitor; a coherent refutation
  by the fit) — and a FALSE LOCK — the DVL stands apart from the tag velocity by more than 8 cm/s for about a
  second, whatever the tags saw (difference monitor; its message says "a false bottom lock rather than the mount").
  NOT a trip: a glitch of a few reports (once or every few seconds), zero-mean DVL noise, a fit that refutes only
  because the DVL "moves" and the tags do not. Those are HELD BACK without a latch, three ways, and the DVL returns
  by itself: an INNOVATION HOLD (a rejected sample keeps the DVL out until a tag velocity from a NEWER camera frame
  agrees — the same frame growing older does not rehabilitate it; keyed on "a fresh frame" instead, the hold never
  released once the tags ran 0.25 s late, or was never set — round 5; and a rejected sample stays out of the
  fix-age integral after the release, which otherwise put 9–11 cm into eta and the rung carried it — round 6), an
  ALIGNMENT HOLD (the alignment monitor reads
  the DVL's motion as going where the tags do not show it, however little of it the tags see), and a REFUTATION
  HOLD (the fit refutes it, with the newest refuting evaluation within `REFUTE_HOLD_S` 3 s). Each threshold here
  answers a reproduced failure: without "tags see it", two reports 12 cm/s off or 3 cm/s of DVL noise at rest
  latched out a correct, proven mount (round 2); at one half, 2–2.5 cm/s of noise, 4-report glitches every
  3–10 s (a "scale 1.3×" refutation) and one 4-report glitch at a square's corner still did (round 4); without the
  refutation hold a wrong mount under noise was flown while the fit refuted it (round 3), and without its 3 s
  staleness a DVL that went quiet stayed held for ~52 s (round 4); without the innovation hold a false lock the
  gate was rejecting went back in as the fix aged (round 4, below).
* **The "proven" gate.** A wrong mount is unobservable at rest and is wrong-signed velocity feedback in motion.
  IN is refused until (1) `dvl.mount_rpy_deg` / `mount_t_flu` are filled in from a tape measure and (2) this
  session's data proves the mount: least squares `v_tag_fd ≈ A v_dvl` over (surge, sway) pairs — raw two-point tag
  differences at their midpoint against the DVL velocity interpolated there, DVL as the regressor (the cleaner
  signal), pairs admitted only above 2 cm/s of DVL speed. ⟲ The verdict is THREE-WAY with a standard error
  σ = residual scatter / (√n · rms speed along the weaker axis): proven iff max|A − I| + 3σ ≤ 0.25, refuted iff
  max|A − I| − 3σ > 0.25, otherwise "not decided yet". One trimmed refit sets tag jumps aside; if fewer than 60
  pairs survive the trim the verdict is withheld (round 1 reproduced a correct mount called "rotated 27 deg" when
  the window shrank to the minimum with one tag jump in it); un-proving a proven mount takes 15 consecutive
  refuting fits. It reports the yaw to add to the mount and the scale. Synthetic prototype only [예측: tag noise
  0.8 mm at 15 Hz, DVL noise 3 mm/s assumed]: 8 s of 8–12 cm/s surge+sway gives A ± 0.01 and yaw ± 0.3 deg;
  20/90/180 deg mounts, a mirrored axis and a 1.33 scale are each refused by name; surge-only and station-keeping
  give no verdict. NOT seen by the proof: the vertical axis, mount tilt, the lever arm. ⟲⟲ The fit also reports
  `lag_s` — how much later the tag velocity describes the motion than the DVL velocity carrying the same stamp
  (the residual's slope on the DVL's acceleration) — as a commissioning number, logged with the proof: everything
  here pairs the two sensors by their stamps. It is biased toward zero by the noisy acceleration (+33 / +49 /
  +73 ms read for 50 / 100 / 150 ms in the simulation below [예측]) and reads zero on a single-frequency ellipse,
  where the lag is absorbed into A as a rotation — so it is taken on straight passes and read as a floor.
* ⟲⟲ **In flight: two trips, gates and holds.** Both trips (latched) are fed by the proof's own evidence — one
  TIME-ALIGNED pair per new tag fix: the raw two-point tag difference against the DVL velocity interpolated to its
  midpoint. Not by the tick's low-passed tag velocity against the newest DVL sample, which is what the first cut
  did: whatever the two latencies differ by is then booked as disagreement, and in the timing simulation below a
  CORRECT mount tripped once they differed by ~0.15 s and the vehicle oscillated at 0.5 Hz or faster [예측:
  observed while designing, 2026-10-02]. (a) The low-passed (1 s) horizontal difference VECTOR above 8 cm/s — a
  standing offset, a lasting false lock; a reversed mount is invisible to it below 4 cm/s [유도: the difference is
  twice the speed] and so is any oscillation. Each pair's difference is clipped at a FIXED 15 cm/s (innov_max):
  one jumped tag fix (metres per second in a raw difference) or one short burst cannot trip, a DVL grossly wrong
  pair after pair trips in about 0.8 s. ⟲⟲ (The clip used to be 4σ of the proof's scatter; with 3 mm tags that
  was 36 cm/s and ONE 3–4 report glitch tripped it in 0.4 s — round 4.) (b) ALIGNMENT, the proof in miniature
  over the last second: along each principal direction e of the DVL velocity that moves faster than 2 cm/s rms,
  `1 − |A e − e|` for the least-squares map A: tag ← DVL (1 the same motion, −1 reversed, −0.4 a quarter turn);
  below 0.5 by three standard errors — from that regression's own residual, so slow or noisy motion is "no
  evidence" — it holds the DVL back, and it trips when the tags also see ≥ 80 % of that motion for 0.4 s on end.
  Principal directions rather than body axes because a mount a few degrees out leaks surge into the DVL's sway
  and a per-axis slope reads the leak as "sway disagrees"; the hold because the first pair or two of a new motion
  are that direction's whole evidence, and a few tens of ms of stamp error make them read zero. This is what
  catches a station hold OSCILLATING on wrong-signed feedback, and a mount the proof accepts cannot trip it
  [유도: 14.5 deg of yaw is a deviation of 0.25]. One pair's weight in its sums is capped at 1/7, so it keeps
  working when tags come at 5 Hz or slower (uncapped, the effective pair count settled at 9 under a minimum of 10
  and the monitor was silently off). Gates (that tick only): an innovation gate, |v_dvl − v_tag| > 15 cm/s against
  the newest tag velocity (widened to 4σ of the proof's residual and by a_max × tag age beyond 0.25 s). That
  reference survives a tick or two without a tag state (up to 0.25 s, its age widening the limit) — clearing it on
  every such tick made the source flap once per fix cycle at slow tag rates — but a tick with NO tag velocity that
  fresh does not use the DVL at all: the first tick after a real tag gap, when a false lock picked up during the
  gap would otherwise go in unchecked; and a jump gate, |Δv| > a_max·Δt + 5 cm/s between
  good samples with a_max 2.0 m/s² [유도: 30 N / 19 kg ≈ 1.6]. fom and altitude are fail-closed. All thresholds
  [예측].
* ⟲⟲ **What a synthetic timing simulation says — NOT a measurement.** `rov_gui/tests/sim_dvl_monitors.py` runs
  the real `DvlAid` + `StateAssembler` on PRESCRIBED motion (open loop) with assumed sensor numbers (tags 15 Hz,
  0.13 s late, 0.8 mm; DVL 12 Hz, 0.19 s late, 3 mm/s, 0.2 m arm; ticks 20 Hz). Output of the 2026-10-02 run:
  `rov_gui/tests/fixtures/sim_dvl_monitors_20261002.txt`; every number here is [예측]. A correct mount never
  tripped: 9 motions × 12 seeds, the same with tag jumps, across five latency pairs, 21 glitch / noise conditions
  (one 4-report ±14 cm/s glitch incl. at a square's corner; 3–4 reports of 25–30 cm/s with 3 mm tags; 1–4 report
  glitches every 3–10 s; DVL noise 1–5 cm/s; a glitch with the tags 0.28 / 0.35 s late) on four motions (incl.
  0.3 m/s passes), and 300 s near the monitors' speed floor under the
  opt-out. Those glitches and noise cost availability instead — held back, not latched: glitches every 3 s leave
  39–60 % of the ticks on the DVL at rest, 2.5 cm/s of noise 48 %, 3 cm/s and more is hardly ever proven. A false
  lock that LASTS does trip, as intended: +14 cm/s for 1 s or +10 cm/s for 2 s, once or repeated; +10 cm/s for 1 s
  or +14 cm/s for 0.5 s does not. A reversed axis or
  a 90-deg mount flown under the opt-out tripped 0.8–1.9 s after IN (8.6 s for a reversed sway on a 5+2 cm/s
  diagonal). NOT caught under the opt-out: any wrong mount at station-keeping speeds (1–2 cm/s), a reversed axis
  moving below ~2 cm/s rms; a reversed axis on a STRAIGHT
  line within 14.5 deg of the other axis, at any speed (there a reversal is indistinguishable from a rotation by
  twice that angle [유도]); a mount 30 deg out below ~15 cm/s; a 2× scale below ~8 cm/s. In those cells 98 % of
  the ticks fly on the wrong DVL; where the fit can at least refute (a 2× scale on a 6 cm/s oscillation) the
  hold-back brings that to 16 %. On the default path the proof catches all of them (the straight line cannot even
  form a proof). A mount the proof ACCEPTS with 12 deg of
  residual yaw or a 1.2× scale trips at 0.4 m/s and above (the proof's tolerance is relative, the difference
  threshold absolute) — the proof's log line now says at what speed, and by how many degrees to correct the mount.
  A noisy DVL (3 cm/s) is mostly never proven — the fit ignores the DVL's own noise and the map is pulled toward
  zero — and stays OUT; at 1 cm/s nothing changes. Known sensitivity: a capture-stamp error on a sustained
  0.5–1.5 Hz ELLIPTICAL motion trips a correct mount (2/36 and 3/36 at ±50 ms, 8/36 at 100 ms, 8/36 at 150 ms;
  none on straight-line motions), because a time shift δ of a velocity that turns at ω is an apparent rotation
  ω·δ. A wrong lever arm (3 or 10 cm) trips nothing: the false velocity it makes is motion only the DVL sees, and
  3 cm/s of it [유도: 0.3 rad/s × 0.1 m] is under the difference threshold. The simulation has no controller in
  it, so wrong-signed feedback growing the motion is not represented.
* **Lever arm.** `v_origin = R_frd_dvl v_dvl − ω × r`. Device frame x forward (LED), y starboard, z down
  [스펙: docs.waterlinked.com/dvl/axes], so an LED-forward mount is the identity. 0.3 rad/s × 0.2 m = 6 cm/s
  [유도] — the size of the mission speed, hence not optional. ⟲ ω is interpolated from the ATTITUDE history at the
  sample's own time of validity; with the newest rate a changing yaw leaves up to ~7 cm/s behind [유도, review].
  ⟲⟲ With a lever arm and no rate for that moment the sample is not used (an uncorrected reading is not the
  body-origin velocity).
* **Tag dropout, STATION only.** `station_bridge.py` gains tier `dvl`: x/y integrate the DVL velocity from the
  last fix (rotated by autopilot roll/pitch and gyro-carried yaw), every axis held up to `dvl_hold_s` 10 s
  [예측], only while the DVL stays usable without a gap; then the existing IMU/COAST ladder on the same elapsed
  time, never climbing back. ⟲ Nothing cross-checks the DVL while the tag is gone, so the rung is bounded by what a
  station hold is: a DVL reading above `dvl_max_speed_m_s` 0.3 or integrating further than `dvl_max_disp_m` 0.5
  from the last fix closes the rung for that dropout. ⟲⟲ When a cap closes it, the rungs below start from the last
  FIX at rest, not from what that DVL integrated (the audit held 0.34 m of estimate error against the controller
  until `imu_hold_s` ran out); when the DVL merely goes away, they carry on from its last position. A tick the rung
  did not integrate — the bridge blind for a tick, a fault it does not carry — closes it until the next fresh fix.
  And the rung opens only for a DVL that was actually FLOWN on the last tick that had a tag: one the innovation
  gate was holding back while the tag contradicted it is not made trustworthy by the tag going away. ⟲⟲ Including
  the way tags really go: the fix stops arriving and the assembler bridges it until `tag_stale_s`, the gate's limit
  widening with its age — the innovation hold keeps such a DVL out through that phase (round 3 walked the estimate
  0.5 m on a false 0.25 m/s by the first lost tick, round 4 0.6 m by the widened gate). "The last fix" a closed rung
  returns to is the FIX's own position, not the last tagged tick's eta, which may already hold that DVL's integral
  (round 4: 0.125 m off). The cost: a hard kick (~1 m/s²) while the tags are in view holds an honest DVL out too,
  and if the tags then drop the pre-DVL ladder takes over — as without a DVL. ⟲⟲ The rung hands the IMU rung its
  POSITION only: the IMU dead reckoner is re-anchored at rest on rung ticks, and on tagged ticks with the TAG
  velocity even when the tick flew on the DVL (the pre-DVL value); seeded with a false lock's velocity that then
  simply ended, it integrated 0.67–1.2 m on a vehicle at rest (round 5).
  The ladder's own handling of such a blind tick is unchanged from the committed code (it restarts the hold clock;
  KNOWN_ISSUES): an intermediate cut that kept the clock running changed the tiers of runs with no DVL at all and
  was reverted. ⟲ The rung's `nudot` is built like the assembler's
  (accelerometer + gravity − ω × v) instead of zeros, so the EAOB is not told "moving, not accelerating". ⟲ The
  recovery log gives the dead-reckoning error only when the dropout ENDED on the DVL rung (otherwise the gap is
  mostly post-DVL drift), plus the gyro-yaw error over the same span. Path / policy / replay missions still
  disengage on a stale fix (operator decision).
* **Timing.** ⟲ Device stamps are mapped to the station clock through the fastest report of the last 30 s
  (`DeviceClock`: offset = min(t_arrival − time_of_transmission), `t_valid = time_of_validity + offset`), so the
  device clock needs no sync and a backlog after a TCP stall keeps its true age instead of being stamped fresh.
  The vendor's documented example report is 189 ms old at transmission [스펙 예시, one sample] — the same order
  as the tag path's latency (the estimation reviewer's budget: DVL ≈ 0.22–0.24 s vs tag FD ≈ 0.21–0.27 s [유도]),
  so the expected gain is noise, not lag, until measured.
* **Records.** Meta schema 16 → 17: `dvl` block always written (use_at_engage, armed_at_engage, proof, tripped,
  device config + changes, rows by `vel_src` and by fallback reason). CSV gains 15 trailing columns `vel_src,
  nu_u/v/w, fd_u/v/w, dvl_u/v/w, dvl_ok, dvl_valid, dvl_fom, dvl_alt_m, dvl_age_s` (body FRD) — the first body
  velocity this CSV has carried. `bridge_tier` gains `dvl`. Raw device lines go to `<stem>_dvl.jsonl`. NEVER pool
  velocity-derived statistics across `vel_src`, nor runs across `dvl.armed_at_engage`.
* **RL followers.** `rl` / `rl_pwm` were trained against a simulated tag-finite-difference estimator; a DVL
  velocity is an observation distribution they never saw. Not blocked; warned at engage and recorded.
* **Read-only.** The station sends `get_config` (a read) once per connection and every `config_poll_s` 10 s —
  ⟲ a change in the device's `mounting_rotation_offset` / `speed_of_sound` after the first answer is a trip — and
  nothing else: no config writes, no dead-reckoning reset, no ping trigger. The device's own `position_local`
  dead reckoning is logged raw and never consumed.
* ⟲ **Two pre-existing defects fixed on the way** (they apply without a DVL): `MpcStatus.bridge_tier` was filled
  only inside the `--imu-dr` block, so the panel's "NO TAG — COASTING" chip could not fire on an ordinary run; and
  `disengage` reset the ladder before the closing meta was written, so `station_bridge.dropouts` /
  `reached_coast` / `last_recovery` closed as 0 / 0 / null on disk in every run so far.

Verification: `rov_gui/tests/test_dvl.py` 96/96, `rov_gui/tests/demo_e2e_dvl.py` (correct and 90-deg-wrong
mount, real window + demo backend) and the timing simulation above, no device; the other `rov_gui/tests` suites
are where they were before this change. **Zero hardware runs**: the mount, every [예측] threshold, the latency,
the noise and the behaviour over this pool's floor are unmeasured (KNOWN_ISSUES.md 2026-10-01).

## 2026-10-02 — The paper's estimator beside ours: robot_localization's EKF, in shadow, with a switch to fly on it

The operator asked for the Learning-to-Swim estimator "exactly as the paper did", to compare with the 2026-10-01
velocity-source design. **Nothing below is a measurement on this vehicle** (no water run yet).

* **Which EKF.** The paper (arXiv:2410.00120v2, Fig. 2: "IMU, DVL, April Tag → Fused state estimation (50Hz)",
  an EKF "implemented in ROS") and its platform paper (CUREE, arXiv:2303.03943) give no state vector, noise or delay
  handling, and the released code (warplab/isaac-auv-env) is the simulator only. The lab's state-estimation
  repository (warplab/robot-state-estimation, empty) describes itself as "A state estimation codebase for marine
  robots using the ROS localization package". So "the paper's way" is taken to be ROS `robot_localization`'s
  `ekf_localization_node` — an inference, stated as one.
* **What was ported** (`rov_gui/control/rl_ekf.py`, from cra-ros-pkg/robot_localization noetic-devel @ 9ef26a57):
  the 15-state kinematic model and its Jacobian (`ekf.cpp::predict`; pinned against central differences of the
  transfer function, max error 3e-10), Joseph-form `correct` with angle-wrapped innovation and the Mahalanobis gate,
  `processMeasurement` (a measurement older than the filter is corrected in WITHOUT prediction), the measurement
  queue and the timer (`integrateMeasurements`, sensor-timeout prediction, "corrected data" not published),
  per-topic stamp ordering, the IMU's three-way split, `prepareTwist` (lever arm with the STATE's body rate, mask
  and covariance rotation), `prepareAcceleration` (gravity removed with the IMU MESSAGE's orientation), and
  `smooth_lagged_data` (`revertTo` / `clearExpiredHistory`). r_l defaults everywhere they exist (process noise,
  initial covariance 1e-9, sensor_timeout = 1/frequency, no smoothing); frequency 50 Hz from the paper. Frames are
  NED/FRD; the only sign that changes is gravity. The 50 Hz timer is replayed inside the 20 Hz tick
  (`RlEkfNode.advance`): each input is delivered before the first cycle at or after its ARRIVAL, as a ROS callback
  would have run.
* **Choices (not in the paper; [예측] in hw_mpc.yaml `ekf:`).** The tag feeds x, y, yaw (+ z when the station's z
  source is the tag), the vehicle IMU roll, pitch, rates and gravity-removed acceleration (compass yaw not fused),
  the pressure depth z with the assembler's own offset, the DVL (vx, vy) in its measured mount — the SAME
  information the assembler gets, so a difference is the fusion. IMU noise = mavros imu-plugin defaults. Tag/depth
  noise and the DVL fallback are guesses.
* **Shadow first, switch second.** On by default and recorded every tick (CSV schema 18: `est` + fifteen `ekf_*`
  columns; meta `ekf`; `<stem>_ekf_inputs.jsonl`, which `rov_gui.tools.ekf_replay` re-runs offline with any other
  configuration — with the live one it reproduces the live columns to < 1 mm once converged). The panel's EST button
  hands the controller the EKF's state instead (`meas_ctrl`, the imu_dr precedent); `meas` — plot, px/py, every
  interlock — stays the tag state. Disengaged-only, no CSV open; only `--ekf control` can start a session on it, and
  switching in / engaging needs the EKF and the tag state within 5 cm / 3 deg (every "here" target is set from the
  tag state). A run on the EKF ENDS, never falls back silently, when the EKF faults, when an input it fuses goes stale
  (per input — r_l's timeout prediction keeps its own stamp at "now"), on the FIRST station-bridge tick (a tag dropout
  is not carried: the bridge's ladder would leave it on uncorrected IMU integration and a coast retarget would write
  its drift into the target), or when it leaves a fresh tag state by 15 cm (x/y, z), 10 deg (yaw, roll/pitch) or
  15 cm/s (the difference low-passed, tau 0.5 s) for 0.25 s. The EKF fuses the DVL only while the DVL switch is IN and untripped (the DVL's own mount proof
  applies; `ekf_replay --dvl-all` shows the ungated filter offline).
* **Reviews (2026-10-02).** safety-code-reviewer: FAIL → the critical (station bridge carrying an EKF run) and three
  highs (dead output-age gate, DVL gates bypassed, xy/yaw-only divergence) fixed as above, plus the stale
  accelerometer and six lows. A port-fidelity check against an independent re-transcription of r_l: faithful with
  low/info issues (state ≤ 5.5e-14 apart over smoothing on/off, dynamic noise, rejection); the cheap ones fixed
  (angles::normalize_angle wrap, interleaved re-delivery dedupe, missing-compass handling, skip count). Safety
  re-audit of those fixes: PASS_WITH_FIXES, two new mediums fixed — the stale/bridged EKF fault is now DEBOUNCED by
  tag_stale_hold_s like the tag path (a frame landing just after the last 50 Hz cycle ended healthy runs), commanding
  nothing meanwhile (never the bridge's state instead); and the EKF's DVL feed also asks the proof gate
  (`DvlAid.refusal`), since a YAML `dvl.use: true` is IN without having passed it. The velocity gate judges the
  low-passed (tau 0.5 s) EKF-minus-tag difference vector: the tag finite difference is noisy. Round 3: PASS_WITH_FIXES,
  no command path from a held tick; one new medium fixed — a held tick no longer restarts the tag path's stale
  debounce (one counter for both kinds of no-step tick, so an episode is bounded by tag_stale_hold_s; interleaved late
  frames had stretched it to ~2.9 s [유도]); held rows carry est = hold and the observe sentinels.
* **What the synthetic comparison says** [예측: `rov_gui/tests/sim_ekf_compare.py`, output
  `rov_gui/tests/fixtures/sim_ekf_compare_20261002.txt`; open loop, every sensor number assumed]: with r_l's default
  delay handling the EKF corrects the current state with a tag frame ~0.12 s old, so its position trails by speed ×
  latency (0.3 m/s passes: 33 mm rms vs the assembler's 5 mm) and its yaw by yaw-rate × latency (1.7 deg vs 0.3 deg
  while yawing). `smooth_lagged_data: true` removes that (6 mm) and gives the best velocity of all (0.72 cm/s vs
  3.2–3.6), because the accelerometer is then fused at the right moment too. The mavros accelerometer noise trusts
  the accelerometer almost completely: a 0.1 m/s² bias with no DVL gives 15 cm/s of velocity error. r_l's default
  process noise makes the filter take each tag frame almost whole (cm-level 1-sigma even at rest).

Verification: `rov_gui/tests/test_rl_ekf.py` 38/38, `demo_e2e_ekf.py` (real window); `test_dvl.py` 96/96 after the schema-18 header/number update;
the other `rov_gui/tests` suites at their pre-change baselines. **Zero hardware runs.** KNOWN_ISSUES.md 2026-10-02.

## 2026-10-03 — Roll/pitch become a LOW-level choice: PID_7D, MPC_Tuned_7D, RL_7D (station; zero water runs)

Operator request: "pid_7d, mpc_tuned_7d, RL_7d 버전" — followers that also control pitch and roll, for the 7-dim
(`pos_rpy_width`) Diffusion Policy. No 7-dim checkpoint exists on disk (only 5-dim; `run_ablations.sh peg_7d_vitclip`
never ran), and the operator chose code only for RL (training deferred).

* **The LOW decides, not a config switch.** `geometry.LOW_7D_BASE = {pid_7d: pid, mpc_tuned_7d: mpc_tuned, rl_7d: rl}`.
  K/M leave the station (MANUAL_CONTROL s/t) iff the LOW is a `*_7d` follower AND `engage.attitude_axes.enabled`, which
  is now a station CAPABILITY (the sink reads it once at launch). Without it a `*_7d` LOW is refused on a vehicle; with
  it every other LOW engages 4-DoF (2026-09-26..10-02 plain mpc/mpc_tuned sent K/M under the switch and pid/rl/rl_pwm/mpcc
  were refused — zero water runs in that window). Tracking a 7-dim plan = `*_7d` LOW + 7-dim checkpoint;
  `policy.attitude_track` is now a VETO (shipped true); a 7-dim network on a 4-DoF LOW is dropped-and-logged.
  Meta: `controller.type` = the LOW (mpc_tuned_7d carries `base_type: mpc_tuned`), `run.attitude_axes.low/low_attitude`,
  `trajectory.attitude_track_low/_veto`.
* **mpc_tuned_7d** = the 2026-09-26 NMPC attitude path on HwDobMpc mode `mpc_tuned` (the worker maps the key; the bridge's
  `tuned` flag is `endswith("_tuned")`, so the key itself would fly the isotropic cost). No solver change.
* **pid_7d** (`rov_gui/control/pid.py HwPid7d`): the pid set for x/y/z/yaw unchanged plus
  `tau = k_ff [cos th_r sin ph_r, sin th_r] + Kp e + Kd (T^-1(ref) ref-rates - gyro p,q) + I`; Kp/Kd from the same
  3rd-order pole placement (formula k = d = 0) at **wn 0.5 rad/s** (zeta 0.9, alpha 0.2, heavy_gripper I_eff
  0.50/0.90 kg m^2 [유도]), Ki 0.15 set apart (the formula's I alpha wn^3 assumes k = 0), restoring FF 1.35 N m/rad on the
  REFERENCE angles (the NMPC's ZG*W), I stored in N m and clamped to 0.8 x the wire torque, conditional integration on the
  realized K/M (note_applied), heave trim rotated with the measured attitude, Euler yaw rate in the yaw D term, the D5
  hold ramp and T3 validation shared with HwDobMpc (`control/attitude_ref.py`). wn is set by the DELAY (ATTITUDE ~52 ms
  old at the tick [측정 data/20261001/1001_154825/mpc_154825.csv] + ZOH + sink phase + thruster lag), and `omega_derate`
  is not applied. Margins [유도: control-theory-advisor consult, re-computed by `rov_gui/tests/sim_pid7d_attitude.py` ->
  `rov_gui/tests/fixtures/sim_pid7d_attitude_20261003.txt`]: PM >= ~50 deg at 0.2 s total delay and actuator gain
  x0.4..x2.5 only while the restoring stiffness k <= ~2.3 (roll) / 3.6 (pitch) N m/rad; at k ~20 (the RL repo's lumped
  fit) 3-41 deg — hull resonance plus delay, not fixable by gain. Hence a pulse-and-release test (sign, delay, g/I,
  k/I, damping) must precede closing this loop in water. Closed loop with the real class (same artefact, first-water
  caps): k 1.35 -> 10 deg pitch hold error -0.03 deg, overshoot 2.7 deg; k 20 -> 4-9 deg steady error with the axis
  below its cap. The safety audit (2026-10-03, WARN, one HIGH) showed the same mechanism hides an INVERTED wire sign
  (the PID settles at a wrong tilt below the cap); fixed: HwPid7d reports "authority exhausted" (command clipped, or
  integral at its limit with > 1 deg error, or error beyond the 20 deg gate) and interlock (iii) judges it like a
  pinned axis (`test_low_7d.py::test_interlock_iii_trips_pid_7d_on_an_inverted_sign_below_the_cap`). Also from the
  audit: a vehicle engage on pid_7d needs the pulse-and-release record (`pid_7d.pulse_test`, free-decay periods >=
  3.2 s) or the recorded opt-out; non-finite attitude -> zero K/M + n_fail; rl_7d K/M clipped to its trained caps and
  its recorded reference = the envelope-clipped one it flies.
* **rl_7d** (`rov_gui/control/rl_policy.py HwRl7d`): an axes6 export whose `obs_spec.attitude_reference.trained` is true
  (fail-closed — a level-trained export has the same layout); `q_des = quat_from_euler_xyz(roll_ref, -pitch_ref,
  -yaw_ref)` clipped to the trained envelope; roll/pitch slewed even though the 4-axis slew is 0
  (`allocation.slew_axes`); refused while the caps in force are below the trained caps (opt-out recorded) — the
  2026-10-01 rl_pwm cap-0.25 run limit-cycled on a clipped policy. LOW `rl` no longer accepts axes6.
  RL_controller (not a git repo; pre-change copies in `RL_controller/_pre_att7_backup_20261003/`): `goals.py` optional
  roll/pitch reference (`att_*`, off = no RNG drawn: old vs new byte-identical over 12 seed/config cases,
  `RL_controller/logs/smoke/att7_goals_byte_identity_20261003.txt`), env q_des from it, refusal of `ang_error_mode yaw`
  / `rew_scale_tilt > 0` with att goals, `cob_trim_x_range_m`, preset `traj10att`, exporter `attitude_reference`,
  `scripts/_pipeline_att7_seed.sh` (NOT run). Isaac smoke (3 iterations, code check) passed:
  `RL_controller/logs/smoke/smoke_att7_20261003.log`. RL_PWM (pwm10_s1) was trained with a level q_des: it levels, it
  does not track a 7-dim policy's roll/pitch.

Verification: `rov_gui/tests/test_low_7d.py` 19/19; updated attitude tests; full station suite 1170 passed / 15 failed
(the same 15 as the pre-change baseline) / 8 skipped; `demo_e2e.py pid_7d policy rp` and `mpc_tuned_7d policy rp` OK
(`rov_gui/tests/fixtures/demo_e2e_*_policy_rp_20261003.log`, toy plant — wiring, not performance); RL_controller tests
53/53. **Zero hardware runs; every gain [예측].**

## 2026-10-03 (2) — RL_PWM learns to track roll/pitch (att7pwm, still water) + station LOW RL_PWM_7D (zero water runs)

Operator request: train RL_PWM now, without the waves/current disturbance. The level-trained pwm10_s1 had only ever seen a
LEVEL q_des, so a 7-dim policy's roll/pitch were out of its distribution.

* **Training** (RL_controller, still water): `scripts/_pipeline_att7pwm_seed.sh` = dr preset `traj10att` (traj10 DR +
  roll/pitch goal paths |roll| <= 10, |pitch| <= 20 deg, <= 0.35 rad/s, 30 % level + CB trim sampled per episode;
  `sea:` absent -> SeaStateCfg p_mode NONE) in action_mode pwm, ang_error_mode full, no absolute-tilt term, tumble 60 deg,
  tag dropout 35 deg, PPO std floor 0.05. Seeds 0..2, 2048 envs x 1200 it, 34:38-34:45 each [측정 `logs/train_att7pwm_s*.log`].
* **Evaluation** (new `scripts/eval_attitude.py`, still water, nominal dynamics, 15 programs x worlds cob_x 0 / -0.029 /
  -0.035; [측정 `RL_controller/logs/eval/ATT_COMPARE_20261003.md`]): retraining works — world 0 pitch RMS 1.46 deg (s0) vs
  10.93 deg for the level-trained pwm10_s1 fed the same tilted goals; but quality is short of a promotion: steady pitch errors
  up to 7-9 deg in the trim worlds, a +-3 deg roll wobble at level (e_roll RMS 2.8 vs 0.6 deg), vertical |throttle| ~0.7
  vs 0.3 and 24-38 % (s0) / 57-83 % (s1) / ~100 % (s2) saturated ticks. All three fail the [예측] gate (|ss pitch| <= 2,
  |ss roll| <= 1.5 deg). Best: s0 — shipped as a DISABLED candidate `rov_gui/control/rl_policies/att7pwm_s0/`.
* **Station LOW `rl_pwm_7d`** (`HwRlPwm7d`): the rl_pwm transport unchanged (pulses, Lua, MANUAL, vehicle gate, the
  transport's ONE cap `rl_pwm.pwm_cap` — HwRlPwm now reads it via PWM_CAP_KEY) + HwRl7d's q_des (plan roll/pitch clipped
  to the trained envelope, fail-closed on attitude_reference.trained). Not in LOW_7D_BASE (no s/t gates); a worker pulse bit
  `_attitude_pulses` arms the tilt CEILING for every pwm LOW (plain rl_pwm had none — behaviour change) and, for rl_pwm_7d,
  roll/pitch tracking (`rp_track_live`: policy truth table, div_rp). CSV `rp_track` 2 = tracked via pulses; meta
  `run.attitude_axes.rp_transport/ceiling_armed/rp_track_live`; pwm coast record corrected (eight 1500 us, nothing held).
  test_rl_pwm.py repaired (11 tests pinned the old shipped cap 0.25): 26/26; new test_rl_pwm_7d.py 13/13; suite 1199 passed
  / 4 pre-existing unrelated failures. **Zero water runs; the candidate is not enabled.**
* **Review fixes (same day, workflow wf_b9f3f143-d12, 6 low findings confirmed):** the RL 7d followers (rl_7d, rl_pwm_7d) now
  fly the TRAINED goal rate — the reference is envelope-clipped and then moved at most `attitude_reference.rate_max_rad_s`
  (0.35 rad/s in the att7 exports) × dt per tick, starting level at engage, as the training goal did; an export without that
  key is refused. The D5 hold-ramp decay moved to the START of step (pid_7d too), so the reference the CSV and interlock
  (iii) read after a tick is the one that tick used. CSV `rp_track` 2 only on ticks whose policy pulses actually left (a coast
  or gate-grace tick sends eight 1500 us and records 0). Suite 1212 passed / the same 4 pre-existing failures.
* **Enabled (same day, operator request):** `rl_pwm_7d.policy_dir: rov_gui/control/rl_policies/att7pwm_s0` — the station
  builds and selects RL_PWM_7D from the shipped config (transport cap 1.0 = trained 1.0, motor directions = the sink
  export's). The sim-gate shortfall above still stands; zero water runs.

## 2026-10-03 (3) — The policy can see in colour: an RGB-only checkpoint is fed the C3's colour frame (naive optics; zero water runs)

* **Why:** the RGB-only diffusion policy (`umi_rgb_5d_peg5d_rgb_dinov3b_9_27_9_28_20261001_211106`, DINOv3-B, 5-dim) was
  started three times in the pool and drew nothing: the station fed only `camera0_depth`, the session went READY on a warm-up
  that used the contract's own key, and every predict raised KeyError — zero plans [측정: `data/20261003/1003_151227/events.log`,
  `plans.jsonl`, meta `policy.worker.counters.infer_errors` 39→158].
* **What changed (rov_gui):** the PolicyWorker now feeds whichever image the LOADED checkpoint's contract names. `camera0_depth`
  takes the unchanged depth path (verified bit-identical: 173 plans over 8 random scenarios against the pre-change worker, review
  2026-10-03). `camera0_rgb` takes the C3 COLOUR frame — the centre lens, the localizer's own frame — through `ColorObsBuilder`:
  centre square crop, `INTER_AREA` to 224, BGR→RGB, /255, which reproduces the training store's `camera0_rgb` exactly (review:
  4 training frames, max |diff| 0). Each attempt is anchored on the colour frame of the newest tag FIX (stamps identical — the
  localizer's stamp is reused), so image and proprio rows are the same instant; the depth rule (newest frame) would have produced
  zero plans at realistic localizer latency (review simulation). Unsupported keys, an undeclared/BGR training channel order, a
  non-(3,R,R) image shape or a backend without a colour feed are an ERROR the controller refuses to arm on; the status says READY
  only after the contract is adopted.
* **What it is not:** the optics are NAIVE (operator decision): the training frames are the handheld colour camera in air,
  unwarped; underwater the C3 colour sees the scene ~1.8× magnified at 224 [유도: focal 178 vs 319 px]. Records: meta
  `policy.worker.obs_mode`/`obs.optics`, `depth_src = c3_rgb` per plan — do not pool with a warped-RGB retrain. The FS schedule
  does not apply to an RGB checkpoint (effective `free`). Offline replay/held-out tools are still depth-only (KNOWN_ISSUES).
* **Evidence:** test_policy_worker 9 new tests, test_policy_obs 2, demo e2e `mpc_tuned policy rgb` 24/24 plans from `c3_rgb`
  installed [측정: `rov_gui/tests/fixtures/demo_e2e_mpc_tuned_policy_rgb_20261003.log`]; safety audit PASS_WITH_FIXES (no critical/
  high), fixes applied. **Zero water runs** — first run in LOW None (observe), checking `obs_t − obs_fix_t[-1]` and the skip counters.

## 2026-10-03 (4) — `policy.action_repr: auto`: the checkpoint picked in the panel decides 5-dim vs 7-dim, pinned at ARM; first RL_PWM_7D water run diagnosed (2026-10-04)

* **What changed:** the operator asked that picking a 7-dim checkpoint should need no YAML edit. `policy.action_repr` gains the
  CONFIG sentinel `auto` (shipped in hw_mpc.yaml and land_dp.yaml; code default when absent stays pos_yaw_width). Under auto
  the mission's action representation is the contract of the checkpoint the policy worker holds at START, resolved by
  `geometry.resolve_policy_action_repr` from the SAME PolicyStatus object the ARM refusal judged (`_arm_policy` refuses
  if another status arrived in between), pinned in `rp["action_repr_pinned"]` / `scen["action_repr"]`; "" and pose10d are
  refused, never defaulted; the intake compares with the pin only. Explicit values behave and record exactly as before.
  The LOW still decides tracking (`geometry.policy_rp_tracked`); it is never switched automatically.
* **Operator visibility:** ckpt row `READY · 5-dim | 7-dim` (FAIL with why when START would refuse it); a pre-START forecast
  in the mission label (TRACKED in WARN colour / DROPPED-AND-LOGGED / holds level; respects the veto, the s/t capability, land
  dry-run, contouring LOWs and LOWs without an export); after ARM the chip shows `· 7d rp TRACKED|DROPPED`.
* **Records (schema stays 18):** trajectory.action_repr = the pin (pool on it); policy.action_repr = the live mission's pin,
  else the last pin armed while this CSV was open, else null; policy.config.action_repr echoes "auto".
* **Evidence:** test_policy auto tests, panel forecast test, 10/10 mutations killed, suite 1224 passed / 4 pre-existing
  failures; demo rows [측정: `rov_gui/tests/fixtures/demo_e2e_mpc_tuned_policy_auto_20261004.log` (304 rows rp_track 0),
  `demo_e2e_pid_7d_policy_rp_auto_20261004.log` (308/308 rows rp_track 1)] — synthetic plant, wiring only.
* **First water run of RL_PWM_7D + the 7-dim peg checkpoint (2026-10-03 20:27, `data/20261003/1003_202721/`) FAILED:** three
  engagements of 1.7 / 6.3 / 2.5 s, all released by the pilot. Diagnosis (records recomputed): att7pwm_s0 holds a learned
  vertical co-contraction (T5/T8 up vs T6/T7 down at +230..+340 us even at a level reference, the same as its sim level program
  — signs verified against the mixer flight in the same folder), drawing 22-42 A; the vehicle really sank (pressure depth
  0.66 → 0.83 m in the UI video HUD) under a nominal +19 N upward command (peg −12 N outside the DR; the net is the difference of
  two ~50 N numbers); and the policy's high gain on finite-difference velocity turned tag-fix z jumps (9 jumps >20 mm in
  engagement 3, none in engagement 2) into ±90 N heave reversals. The DP is not the main cause (engagement 1 had no plan; the
  engagement-3 excursion preceded the first plan); engagement 2's +14° nose-up plateau is most likely the stern on the floor.
  Station timing was normal. KNOWN_ISSUES has the full entry; a retrain needs a co-contraction penalty, payload buoyancy in the
  DR and tag-jump/fallback-camera noise in the observation model.

## 2026-10-04 — MPC_Tuned profiles: the follower's tuning is a file, picked on the panel like a DP checkpoint (zero water runs)

* **What changed:** operator request — save the mpc_tuned tuning that has been flown, and load/apply a tuning by picking it
  like a Diffusion Policy checkpoint, with today's values as the default. A profile (`config/mpc_tuned_profiles/NAME.yaml`,
  `rov_gui/control/mpc_tune.py`) holds three sections: `apply` (Q/QN/R, the along/cross split, policy q_scale/along_scale,
  vehicle_net_buoyancy_n, axis_cap, axis_slew_per_s — every key required), `compare` (rov_model, ctrl_hz, N, dt, u_max, plant
  damping, axis_gain — NOT applied: the damping is in the generated solver mpc/dobmpc share) and `record` (context). HwDobMpc
  HOLDS the profile and APPLIES it only in mode `mpc_tuned` (LOW mpc_tuned_7d flies that mode): baseline W written with
  cost_set, heave trim through the disturbance parameter, policy knobs via `policy_follow`, cap/slew via the controller hooks
  MpcWorker already reads (the cap may only tighten). Any other mode writes the build W and the config trim back at the switch.
  Operator decision: the heave trim is part of the applied set (so a profile carries its payload assumption).
* **Default = today:** `20261004_default` is a snapshot of hw_mpc.yaml + params.py as of 2026-10-04 (split 0.25/4.0, q_scale
  4.0, along_scale 3.0, trim −12 N peg-in-jaw, cap 0.5, slew 1.5/s, params Q/QN/R); hw_mpc.yaml `mpc_tuned_profile` names it,
  so mpc_tuned flies exactly what it flew before. Under mpc_tuned, the hw_mpc.yaml `mpc_tuned:` block, `policy.q_scale /
  along_scale` and `vehicle_net_buoyancy_n` are no longer read while a profile is held (`--mpc-tune none` restores that).
* **Refusals:** a pick is refused while engaged, with a controller CSV open, before READY, or for a file that does not
  load/fit (the previous profile stays held — the swap is transactional, a solver write that raises rolls it back). A launch
  profile that fails to load makes mpc_tuned REFUSE to engage rather than fly hw_mpc.yaml under a launch that named a profile.
* **Safety review (2026-10-04, WARN, no critical) → fixed:** axis_slew_per_s must be > 0 and is tighten-only like the cap (a
  profile could have switched the anti-chatter slew off); Q/QN pose rows must be > 0 (a zero x/y weight divided by zero in
  meta at CSV open); `path_cost` and `compare` must be complete (missing compare keys WARN); `name:` must equal the file name;
  a profile trim ≠ hw_mpc.yaml trim is a WARN (the peg trap, KNOWN_ISSUES); `save` defaults to the HELD profile's apply
  (`--from-config` / `--from`), because hw_mpc.yaml's blocks are not what mpc_tuned flies while one is held.
* **Records (boundary):** meta `controller.mpc_tune` (profile, path, sha1, active, apply, warnings), `controller.weights_source`,
  `controller.Q/QN/R` = the baseline THIS mode flew (was always params.py's), `trajectory.along_scale` +
  `follow_knobs_source`. Runs before 2026-10-04 have no `mpc_tune` block and flew hw_mpc.yaml's values.
* **Evidence:** `rov_gui/tests/test_mpc_tune.py` 16 passed (incl. a real acados bridge: a 4× position-weight profile changes
  the mpc_tuned wrench and plain mpc afterwards is bit-identical), test_path_cost 27/28 (the 1 failure is the pre-existing
  `|vehicle_net_buoyancy_n| ≤ 2` assertion vs the −12 N peg value), test_low_7d + test_control 161, policy/offline/replay/
  attitude/station-bridge/policy-worker 361; demo e2e `mpc_tuned square` and `mpc_tuned policy` OK with the default profile
  held (meta: profile 20261004_default, active, q_scale 4 / along 3 from the profile) — synthetic plant, wiring only.
  **Zero water runs** — but the default profile reproduces the previous values, so behaviour is unchanged until a different
  profile is picked.

## 2026-10-14 — The hardware square's yaw mode: fixed (crab) | face the direction of travel (turn in place at each vertex); panel default tag 37 · 1×1 m · 0.10 m/s (zero water runs)

* **Why:** operator request — choose on the panel between today's square (one heading, crabbing) and one that **faces the
  direction of travel** (+y side → face +y, +x → +x, −y → −y, −x → −x). `heading_follow` existed in the code, but on the
  hardware spatial follower (`PathCursor`, `path_fillet_m: 0.0`) it made the yaw reference **step 90° in one stage** at each
  vertex and put a (π/2)/0.05 = 31 rad/s spike into the NMPC plan's `r_ned` [유도]; `yaw_rate_deg_s` (60) is read only by the
  wall-clock sampler and the follow mission, so in path mode it bounded nothing. Every hardware square/circle had flown
  `heading_follow: false` [측정: `data/*/*/mpc_*.meta.json`]. The sim's heading_follow (slewing at 60°/s while moving)
  contradicts a sharp vertex's position reference (memory square-corner-error-floor); the hardware reference already
  **stops** at the vertex, so turning there is the one way that asks for nothing unrealizable on top.
* **What (theory):** a fillet-0 rectangle under heading_follow gets a `_Pivot` (position fixed, heading turning) at every
  vertex. The pivot's nominal arclength L is set from the turn rate, so the time the cursor takes to cross it at the creep
  speed IS the turn time.

  ```
  ψ_ref(s) = ψ_fix = yaw_map_deg − yaw0                       (yaw fixed: crab)
  ψ_ref(s) = ψ_path(s)                                        (yaw = travel)
  vertex i:  p(s) = v_i,  ψ(s) = ψ_in + Δψ·(s − a_i)/L,  s ∈ [a_i, a_i + L),  Δψ = +π/2 (NED, clockwise from above)
  L = v_creep·|Δψ| / ω_turn          (0.02 m/s, 90°, 20°/s → L 0.09 m, T = L/v_creep = 4.5 s) [유도]
  inside a pivot:  v_ref = 0
  r_ref(stage k ≥ 1) = (dψ/ds)·v_nominal              (a forecast, at the same nominal speed as the horizon's positions)
  r_ref(stage 0)     = [ψ(θ_cmd,k) − ψ(θ_cmd,k−1)] / Δt,  |·| ≤ nominal rate   (the turn the reference really made)
  r_ref(s ≥ s_end)   = 0
  approach/settle heading = ψ_path(0)  (the first side; map +x unrotated)  → no yaw step when the path arms
  lap time (1×1 m @0.10, a_long 0.05) ≈ 49 s → ≈ 65 s  (+ 4·4.5 s) [유도: speed_profile, reference time]
  ```
* **How:** `path_geometry.path_from_scenario` (+ `pivot_speed_m_s` = `path_creep_m_s`) builds `_pivoted_polygon` only for
  heading_follow ∧ square ∧ fillet ≈ 0 — crab, filleted, circle and line geometry unchanged. `PathCursor` became
  **pivot-aware**: (a) the projection may not pass the START of the first pivot the **setpoint** has not finished (every
  point of a pivot is the same xy, so the argmin tie let the projection skip it — an 88° one-tick step [유도: offline
  PathCursor]), (b) once the setpoint is inside the first pivot the **hull** has not finished and the hull has arrived
  (projection at the vertex, or within 5 cm of it — `PIVOT_CAPTURE_M`), the hull's progress follows the setpoint's (a turn
  is time, not distance; this removes the leash trapping the setpoint inside a long pivot and the path that ends in a pivot
  never reaching `complete` — RL parks ~1.8 cm short of a vertex [유도: offline PathCursor + Fossen plant], hence the
  distance test), (c) a pivot is crossed at its own speed exactly, (d) inside a pivot `v_ned = 0`, and `r_ned` as above.
  On a path without pivots, `step()` and a fixed-heading `plan()` are element-for-element what they were (the test's
  legacy-cursor oracle); a heading_follow plan's `r_ned` changed on purpose on every path (analytic, K=1 included). Worker:
  a square/circle that faces its travel approaches and settles on its first tangent (read off the same curve,
  `_path_start_yaw` — placed without the wall-clock sampler, a geometry error refuses START); START is refused **before the
  approach** under STABILIZE + `yaw_axis: hold`, on MPCC/DOBMPCC (square), with `path_fillet_m` > 0 and with
  `path_following: false` (the first two cannot honour the turn rate, the last two build no vertex pivots). HwRl clips a
  plan-carried r to `yaw_rate_max` (0.6 — not an export number: in-place 0.4 / heading-follow arcs 1.0 [스펙:
  obs_spec.json]) and keeps its previous yaw every tick (no one-tick rate spike after a turn). Panel: a NoFocus combo
  `yaw fixed +y | yaw = travel`, visible for Square only, riding the scenario as `heading_follow` (always false for the
  other shapes; deaf to the wheel), frozen from START, `· yaw = travel` on the chip; Esc on an open combo list is the
  E-STOP (all three combos). Turn rate `square.vertex_turn_deg_s` 20°/s — inside the RL policies' trained in-place yaw rate
  0.4 rad/s [스펙: `rov_gui/control/rl_policies/traj12_s2/obs_spec.json`]; ceiling 90°/s (refused at load), a warning
  above 22.9°/s on an RL LOW. Panel defaults (the YAML seed): tag 37, 1.0 × 1.0 m, 0.10 m/s, yaw fixed — the tag and speed
  boxes are shared by every shape, the distance box is remembered per shape (size/length/radius), the startup shape stays
  station, and the window sends the panel's scenario to the worker once the backend is attached.
* **Review (2026-10-14, 6 lenses + an adversarial verifier per finding) → fixed:** the capture is keyed on the **hull's**
  next pivot, not the setpoint's (a slow follower reaching the last pivot after the setpoint never completed — the shipped
  line behind a 0.03 m/s follower [유도]); the K=1 plan sent a nominal 20°/s while the leash held the setpoint still →
  stage 0 is the turn the reference really made; r and `ref_speed` are 0 at the end; fillet/path_following refusals; the
  panel's scenario never reached the worker at startup; the shared distance box opened Circle at r 1.00 m after a station
  start; combo Esc/wheel.
* **Side effect (record boundary):** a line also ends in a pivot and so, by the same defect, could **never complete**
  (offline a 1 m × 2-lap line stalled at θ 4.07 of 4.08, the shipped 2 m × 5 at 20.18 of 20.20, forever [유도: the
  pre-change cursor]; no hardware line has a completion on record). With the pivot-aware cursor a line completes
  (`demo_e2e.py pid line` → `line complete`; the KNOWN_ISSUES 2026-08-17 "budget" entry had the wrong cause and is
  deleted), and its turnarounds hold v = 0.
* **Records (boundary):** meta `trajectory.heading_follow`, `.vertex_turn_deg_s` (every square; FLOWN only where
  `vertex_pivot` is present), `.vertex_pivot` (len, speed, per_lap, capture), `reference_clock.strategy =
  "waypoint_vertex_pivot"`, `reference_clock.pivot_capture_m` (present = the new cursor, lines included). Never pool with
  crab squares or with the sim's heading_follow squares.
* **Result:** unit tests (cursor rate/completion/slow follower/real stage-0 r/byte-identity oracle, worker approach→settle→
  arm heading/refusals/completion/meta, config validation, RL r, panel yaw combo/per-shape distance box/Esc and wheel,
  window attach) pass. Fossen prediction-model closed loop, 1×1 m @0.10, 2 laps [예측/sim, no artifact]: PID, MPC_Tuned
  and RL all complete, reference yaw rate ≤ 20.0°/s, the target within 1 mm of the vertex while turning; heading error on
  the straights p95 PID ≤0.5°, MPC_Tuned ≤0.4°, RL ~12°. Demo toy-plant e2e `rov_gui/tests/demo_e2e.py <pid|mpc_tuned|rl>
  square_travel` (0.6×0.4 @0.15, 1 lap): all three LOWs `square complete` — a wiring check, not a performance figure
  [예측/sim]. **Zero water runs.** Remaining limits [예측]: on the ±x sides the C3 looks across the mat's short dimension
  and may lose tags (no dropout bridge on paths), the tether twists one turn per lap, the reported origin sits behind the
  true yaw pivot so an in-place turn may translate the hull, and START's first turn is an unramped yaw step (KNOWN_ISSUES
  "square `yaw = travel`").
