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
