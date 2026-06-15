# Real BlueROV2 Hydro Verification — sim-to-real protocol (BLUEPRINT)

**Status: BLUEPRINT.** No hardware run yet — this is the protocol to execute once the physical
BlueROV2 + pool are available. It is the **real-world companion** to the in-sim
[HYDRO_VERIFICATION.md](HYDRO_VERIFICATION.md).

## Two different claims
- **Claim A — code correctness (DONE, [HYDRO_VERIFICATION.md](HYDRO_VERIFICATION.md), 32/32).**
  "Given the coefficients θ_sim, the simulator reproduces the Fossen equations." The coefficients are
  an *input*; we assumed them. This is a statement about the **software**.
- **Claim B — model fidelity / sim-to-real (THIS document).** "The Fossen structure *with these
  coefficients* predicts what the *real* BlueROV2 does." This is **system identification**: measure
  θ_real on the vehicle, compare to θ_sim, decide whether the gap matters for the controller.

**The honest premise:** the `marinegym_assets/BlueROV.yaml` numbers are *generic BlueROV2 literature
values, not identified on our hull.* Our vehicle has its own ballast, trim, the ZED housing, cabling,
and foam — so θ_sim is guaranteed to be off to some degree. Claim B is what "verify on the real
BlueROV2" means; this protocol identifies-then-compares. Mental model: claim A proved the calculator
computes correctly; claim B checks we typed the right numbers in.

**θ_sim (the targets to compare against, from `BlueROV.yaml` + `bluerov.xml`):**
mass 11.2 kg, I=[0.30375, 0.626, 0.5769], V=0.0113459 m³, coBM=0.01 m,
M_A=[5.5, 12.7, 14.57, 0.12, 0.12, 0.12], D_L=[4.03, 6.22, 5.18, 0.07, 0.07, 0.07],
D_NL=[18.18, 21.66, 36.99, 1.55, 1.55, 1.55]. ⇒ B=ρgV≈110.97 N, W≈109.87 N, net **+1.1 N**,
restoring k=coBM·B≈1.11 N·m/rad.

**Available signals:** BlueROV2 IMU (attitude, angular rate, linear accel), depth/pressure sensor,
commanded thrust (PWM → T200 curve, *with scatter*), and — the project's key asset — **ZED2 + AprilTag
SLAM pose ground truth** in the pool (refractive-corrected; see repo `claude.md` §3) + a gantry.

---

## Prerequisites (do these before any identification run)
1. **Recalibrate the T200 thrust curve with a load cell / thrust stand — highest-impact prerequisite.**
   Every experiment uses "known thrust", but we only know commanded PWM → a generic T200 curve with
   real scatter (battery-voltage sag, water temperature, unit-to-unit variation, ~10–20% weaker reverse).
   A 15% thrust error propagates directly into a 15% drag error and worse for M_A. Measure thrust vs
   command **for our thrusters at our operating voltage**; this replaces the `force_constants: 4.4e-7`
   assumption in `BlueROV.yaml` and tightens every downstream fit.
2. **Synchronized, timestamped logging** of thrust command + IMU + AprilTag pose. Bad time-sync silently
   corrupts every *dynamic* fit (experiments 4–5). Nail this first.
3. **Tether management.** The tether is an unmodeled, configuration-dependent, often dominant force and
   the #1 source of non-repeatability. Use a neutrally-buoyant section, a loose slack bight, route it
   identically across repeats, and deploy minimal tether for dynamic runs. Never identify drag with a
   taut tether (you'd measure the tether, not the hull).
4. **Reconcile pool geometry & clearances.** `config/config.yaml` lists pool width **1.8 m** while the
   working figure has been **2.438 m** — reconcile by measurement before computing run-up distances.
   Depth is only **1.143 m**: keep the vehicle as deep and as far from walls as possible during dynamic
   runs, and note free-surface/wall/blockage effects (below).

---

## sim test → real experiment mapping

| Term | θ_sim | Identifiable on our HW? | Method | Difficulty |
|---|---|---|---|---|
| Net buoyancy (W−B) | +1.1 N up | **Yes, easily** | dry weigh (air) + depth-hold thrust = W−B | easy |
| Restoring (coBM/GZ) | k≈1.11 N·m/rad | **Yes** | IMU equilibrium tilt + small-angle period | easy–med |
| Quadratic drag D_NL | [18.18,21.66,36.99,…] | **Yes — cleanest** | terminal-velocity sweep (AprilTag) | med |
| Linear drag D_L | [4.03,6.22,5.18,…] | Yes | low-speed / oscillation decay | med |
| Added mass M_A | [5.5,12.7,14.57,…] | **Partial** | transients only (see exp. 4–5) | hard |
| Added-mass Coriolis C_A | (derived) | No (structural) | constructed from M_A; check coupled maneuvers | — |

The static + steady-state terms (buoyancy, restoring, drag) — **most of the force the vehicle feels in
normal operation** — are directly measurable with the sensors we have. Added mass is the crux.

---

## Experiment protocol (priority order)

For each: **objective → coefficient → setup → excitation/commands → log → fit → expected confounds →
pass/fail vs θ_sim.** (Tolerances written as ±X — set the actual X when the controller's sensitivity
to each coefficient is known; a reasonable start: ±15% drag, order-of-magnitude on M_A.)

### Exp 1 — Static buoyancy & mass *(easy, high confidence)*
- **Coefficient:** net buoyancy W−B, mass m. **Setup:** dry-weigh the vehicle on a scale (→ W = mg).
  In water, command vertical thrust to hold a constant depth (depth sensor flat). **Log:** scale mass;
  the steady vertical thrust to hold depth; or the thrusters-off terminal ascent/descent rate.
- **Fit:** the depth-hold vertical thrust = W − B. **Confounds:** thrust calibration (Prereq 1).
- **Pass:** identified net buoyancy within ±X of sim's +1.1 N; mass within ±2% of 11.2 kg.

### Exp 2 — Restoring pendulum (roll & pitch) *(easy–med — the real analog of sim T4)*
- **Coefficient:** restoring stiffness k = coBM·B (and r_G−r_B offset). **Setup:** submerge neutrally,
  no thrust; displace to a small roll (then pitch) angle and release. **Log:** IMU attitude θ(t).
- **Fit:** equilibrium attitude → horizontal r_G−r_B offset; small-angle oscillation period
  T = 2π√((I+M_A_rot)/k) → k (this also yields the rotational added inertia, exp. 4); log-decrement → D_L_rot.
- **Confounds:** finite-amplitude D_NL_rot inflates damping (release ≤3°); slow heave drift from net buoyancy.
- **Pass:** k within ±X of 1.11 N·m/rad; equilibrium offset consistent with coBM = 0.01 m.

### Exp 3 — Terminal-velocity drag sweep (surge / sway / heave / yaw) *(the workhorse — direct sim-T2 replay)*
- **Coefficient:** D_L, D_NL per DOF. **Setup:** open run along the pool's long axis (after Prereq-4
  run-up), vehicle deep and away from walls. **Commands:** constant thrust at several levels per DOF.
  **Log:** AprilTag pose → differentiate to body velocity; start the measurement window only after v_∞
  is reached (pose/depth flat).
- **Fit:** per DOF, `thrust = D_L·v_∞ + D_NL·v_∞·|v_∞|`; fit D_L, D_NL across thrust levels. (This is
  exactly the sim T2 closed-form, now on hardware.)
- **Confounds:** thrust calibration (a consistent scale offset across *all* levels ⇒ T200 curve wrong,
  not drag — that's why this experiment also *exposes* Prereq-1 error); short run-up limits high-thrust
  v_∞; **pool blockage/walls make measured drag read systematically HIGH vs open water**; tether; wake
  recirculation (wait 30–60 s between runs, alternate directions).
- **Pass:** D_L/D_NL curves match θ_sim in scale & shape within ±X (note: pool values expected a bit
  high — a pool artifact, not the ocean truth).
- **Single most valuable measurement** — one curve per DOF tells you whether the sim drag is right and
  whether the thrust calibration is right.

### Exp 4 — Oscillation-decay added inertia (heave / roll / pitch) *(med — the restoring-equipped DOFs)*
- **Coefficient:** M_A[2], M_A[3], M_A[4]. **Setup:** reuse exp. 2 (these DOFs have a restoring "clock").
  **Log:** IMU/pose oscillation. **Fit:** from the natural period T = 2π√((I+M_A)/k) with k from exp. 2,
  back out the added inertia M_A; log-decrement gives D_L. **Confounds:** free-surface sensitivity of
  heave M_A at 1.14 m depth; finite-amplitude damping. **Pass:** M_A on these 3 DOFs within order-of-
  magnitude / ±X of sim. *(Note: surge & sway have no restoring → no natural oscillation → not reachable here.)*

### Exp 5 — Dynamic ID for surge/sway added mass (PRBS/chirp + LS/EKF) *(hard — estimate, then bound)*
- **Coefficient:** M_A[0], M_A[1] (the un-restored translational DOFs). **Setup:** after exp. 1–4 fix the
  static & drag terms, command a **PRBS or multi-sine chirp** thrust that keeps the vehicle accelerating
  back-and-forth within the run. **Log:** thrust, AprilTag pose→velocity, IMU accel (synchronized!).
- **Fit:** joint least-squares (or an EKF/UKF estimating parameters as augmented states — the same
  EAOB-style machinery already in this repo) for `[M_RB+M_A, D_L, D_NL]` with the others fixed.
- **Confounds:** short run-up → brief acceleration phases → **M_A and D_L become correlated** in the fit
  ⇒ wide confidence intervals on M_A; pose-differentiation noise. **No tow-tank / PMM available** (the
  gold standard — forced oscillation separating in-phase drag from quadrature added mass — is out of reach).
- **Pass:** report M_A[0], M_A[1] **with error bars**, not as a point measurement. Added mass ≈ 30–50% of
  dry mass for a compact ROV is the order-of-magnitude sanity check (sim surge 5.5/11.2 ≈ 49%, plausible).

---

## What is feasible vs not (be honest in the report)
- **Clean & high-confidence:** buoyancy, restoring, per-DOF drag (the static + steady-state terms = most
  of the operational force).
- **Constrained:** heave/roll/pitch added mass (oscillation tests).
- **Estimate with big error bars:** surge/sway added mass (PRBS+EKF in a short run).
- **Not feasible here:** tow-tank-quality M_A; a Planar Motion Mechanism; ocean-valid numbers — the small,
  shallow pool (1.14 m) makes drag & added mass read **systematically high** (wall/free-surface/blockage),
  so a pool-identified value is **not** ocean-valid. Identify the easy terms cleanly, bound M_A, and never
  report a pool number as open-water truth.

## Instrumentation upgrades (by impact)
1. **Load cell / thrust stand** — recalibrate the T200 curve; collapses the dominant confound and tightens
   every fit. **Get this first.**
2. **DVL (Doppler velocity log)** — clean body-frame velocity, removes pose-differentiation noise (the
   thing that most hurts M_A). Expensive; worth it only if pushing hard on dynamic ID.
3. (No hardware) **synchronized timestamped logging** — prerequisite for exp. 4–5.

## Results table template (fill after running)

| Term | θ_sim | θ_real (measured) | Δ% | Verdict | Notes |
|---|---|---|---|---|---|
| net buoyancy [N] | +1.1 | | | | |
| restoring k [N·m/rad] | 1.11 | | | | |
| D_L surge/sway/heave | 4.03/6.22/5.18 | | | | |
| D_NL surge/sway/heave | 18.18/21.66/36.99 | | | | |
| D_L/D_NL yaw | 0.07 / 1.55 | | | | |
| M_A heave/roll/pitch | 14.57/0.12/0.12 | | | | (exp. 4) |
| M_A surge/sway | 5.5/12.7 | | | | (exp. 5, ± error bar) |

## References (priors & sanity checks — every published number is *someone else's* vehicle)
- Fossen, *Handbook of Marine Craft Hydrodynamics and Motion Control*, 2nd ed. (2021) — the model and ID methods.
- Wu, *Towards a BlueROV2 Open-Source Model* — BlueROV2-specific parameter set to compare against.
- von Benzon et al., "An Open-Source Benchmark Simulator: Control of a BlueROV2," *J. Mar. Sci. Eng.* 2022.
- BlueRobotics T200 performance charts — thrust scatter & forward/reverse asymmetry to calibrate out.
- Cai et al., "Learning to Swim," ICRA 2025 — sim-to-real framing for the Fossen hydro coefficients.

---
*Companion:* [HYDRO_VERIFICATION.md](HYDRO_VERIFICATION.md) (claim A — the sim code is verified; the
coefficients here await real-world identification).
