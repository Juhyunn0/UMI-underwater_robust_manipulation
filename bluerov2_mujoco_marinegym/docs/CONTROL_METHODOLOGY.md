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
50 ms): **NMPC.solve ≈ 83 ms (≈79%)**, EAOB.update ≈ 22 ms (≈21%), full tick ≈ 106 ms = **0.47×
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
| DP (15 s) | 15.0 → 13.4° | 30.0 → **22.9°** | radial 4.9 → 6.1 cm | 0.22 → **0.09** |
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
| solve / tick (N=60) | median **100 ms** (over the 50 ms budget) | median **0.97 ms**, max 1.1 ms |
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
| PID | 14.86 | 14.74 | 15.16 | 10.4 → 12.4 cm |
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
path). Per-seed, ideal DOB-MPC: seeds 0/1/2/4 = 4.1 / 0.7 / 0.9 / 1.4 cm, n_fail 0 (excellent); **seed 3
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

**Result.** Closed-loop DP (dobmpc, seed 0, 20 s, disturbance ON): **ideal radial 5.02 cm / jitter 4.30 cm
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
> [compute_heavy_inertia.py](../compute_heavy_inertia.py) (sensitivity: m_v 0.10→0.344 kg gives
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

Full actuation **actively levels pitch** (11.8° trim → 0.8°) and tightens station-keeping (5.0 → 3.3 cm).

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
disturbance interface. 34 unit asserts + smoke pass. Run:
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
`heading_follow: true`, `yaw_rate_deg_s: 60`; viewer flags `--heading {follow,fixed}`, `--yaw-rate`. Verified:
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
| DP dobmpc (regression) | **0.00 cm** (fully-actuated + DOB rejects the current) |
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
1-lap square NONE via run_viewer: **radial RMS 1.34 cm** (t>5 s) vs **17.8 cm** with the old gains in
`compare_20260702_222150` — ~13× — max 4.0 cm (corner transient), |pitch| 0.0°, no saturation chatter.
`verify_meta.py` passes; `meta.json` sidecars stamp the new gain set. bluerov2 `--smoke` unchanged (legacy path).

**Validity:** sim-only (ideal thrusters, perfect 500 Hz state). For hardware derate ωn to 1.0–1.5 rad/s and
filter the D term. Beyond the |S(1.6)| ≈ 0.41 wave-band residual, the answer is DOB-MPC, not more PID gain.

---

## 2026-07-07 — DOB-MPC yaw reference completed (yaw preview + yaw-rate FF on turns)

**Why.** On the square, DOB-MPC's largest CW/CDW error concentrates at the **(1,1) upstream
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

**Result (A/B, dobmpc, seed 0, current 0° / wave 0°, run_viewer headless).** Yaw slew-to-5° at the
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
   term into an energy pump: a torque-free 0.5 rad/s pitch kick exploded to |q| > 60 rad/s in 1.5 s. Isolated by
   ablation (buoyancy-only stable, drag-only explodes; heavy control stable). Fix: emit **diaginertia** (the
   0.4% Ixz is far below coefficient uncertainty); the deeper hydro fix is logged in KNOWN_ISSUES.

**Controller updates:** `GAINS_HEAVY_GRIPPER` re-derives the same pole placement at the payload masses (sway
m_eff 26.42 → kp/kd/ki 143.7/99.5/42.3; heave 28.29 → 153.9/108.0/45.3; yaw I_eff 0.811 → 9.92/4.79/4.38) and
adds an optional **roll/pitch leveling PD** (`rp_kp=(4.3,7.7)`, `rp_kd=(2.6,4.6)`, gated on the gains dict so
heavy/bluerov2 are byte-identical): the payload's static attitude torque (jaw weight + CB_x offset ~1.3 N·m)
rivals the passive B·coBM restoring (~1.2 N·m/rad), so passive-only attitude walks off. dobmpc params flow from
the registry automatically (MPC weights untouched — the earlier "calm the attitude weights" hypothesis was a
misdiagnosis of lesson 2 and was reverted).

**Results:** PID DP hold 0.0 cm (still water, 20 s); DOB-MPC DP hold rms(12–20 s) **1.3 cm** still / **1.4 cm**
current+waves / 22 cm with 20–50 N Poisson kicks (impulse-recovery transients — EAOB cannot predict kicks,
consistent with the C-mode analysis). heavy reference on the same harness: 0.4 cm (cw). acados-vs-IPOPT
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
vendor 575×254×457 mm exactly), a global voxel-occupancy grid search over translation (no ICP local minima),
scale 1.0233 accounted for (the MarineGym-derived skin is uniformly 2.3% large; placement is TRUE METRIC anchored
at the COM), and a trimmed-ICP translation polish: residual 1.6 mm, seed spread <0.1 mm, fore-aft disambiguated
by the electronics-tube dome. Pipeline constants frozen in [process_c3_mesh.py](../process_c3_mesh.py), which now
emits `meshes/c3_payload_frames.json` consumed by the generator (drift-guarded against `CP.C3_POS`).

**Result** (base_link FLU, origin = vehicle COM): C3 mesh centroid `[0.199, 0.008, −0.156]` — **front-bottom on
the centreline**, housing protruding ~9 mm past the frame nose and ~3 cm below the skid line; optical axis
`[1.000, 0, 0.0056]` (0.32° up-tilt kept verbatim from the CAD mates); stereo baseline horizontal. The C3-BR
bracket straddles the Newton-gripper tube with ~1 mm clearance — independently confirming the guessed
`GRIP_POS=[0.25, 0, −0.17]` is compatible with the real bracket. The 3 MJCF cameras now sit at the **lens plane**
(`x=0.2395`, `y=0.0055±0.0375`, `z=−0.1554`) looking forward-level; the gripper jaws appear dead ahead at ~18 cm.

**Model consequences:** composite inertia diag `[0.38154, 0.77780, 0.70954]` (was `[0.363, 0.749, 0.699]`),
TOTAL COM `[+0.0349, +0.0010, −0.0258]`, coBM **+0.01625 m** (was +0.00955 — COM dropped, CB barely moved),
net buoyancy unchanged −5.7 N. The dropped off-diagonal grew from Ixz −0.0016 (0.4%) to **+0.064 kg·m² (16.8% of
Ixx)** — a real roll-yaw product the plant cannot carry until hydro reads body-frame velocity (KNOWN_ISSUES).
Bracket mass is NOT yet composed (visual-only; pending real mass). Bonus fix: the mesh visual-orientation helper
used `conj(mesh_quat)`; MuJoCo's convention is `v_orig = R(mesh_quat)·v_stored + mesh_pos`, so the geom quat must
be `mesh_quat` itself — the old bake's ~180° principal rotation masked this (conj(q) = −q). Verified: vertex
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
COM, so `<inertial pos>` = 0. Gains: `heavy_c3` inherits `GAINS_HEAVY` (fully-actuated fallback) — PID holds the
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
"cancellation" schemes (conj(mesh_quat), then mesh_quat) double-handled the reframe; the C3's near-square cross
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
