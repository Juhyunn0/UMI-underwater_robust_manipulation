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
**Why → What (theory) → How (implementation + decisions) → Result**.

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

**Result.** Analysis only (no code change). Files: `recordings/20260615/square_{pid,mpc,dobmpc}_*.csv`,
comparison plot `square_compare_*.png`.
