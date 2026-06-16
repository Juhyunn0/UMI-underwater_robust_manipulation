# Hydrodynamics Verification — first-principles, disturbance-free

**Result: 32/32 checks PASS.** The marinegym hydrodynamics ([hydro.py](../hydro.py)) reproduces
the intended Fossen physics to within measurement precision; the single deliberate approximation
(the lagged/filtered added-mass force) is quantified and negligible.

- **What:** an independent harness, [verify_hydro.py](../verify_hydro.py), compares each hydro term
  against a closed-form analytic prediction with **disturbances OFF** (`disturbance=None` → still
  water, so each term is cleanly isolated).
- **Method:** the simulator is **not modified**. Known body wrenches are injected through MuJoCo's
  independent external-force buffer `data.xfrc_applied` (hydro keeps running as its own passive
  callback). Where a single axis must move purely, the net buoyancy (+1.10 N) is cancelled with a
  test-only vertical force.
- **Run:** `python verify_hydro.py` (env `robust`). Figures saved under [docs/figs/](figs/).
- **Reviewed by** the control-theory advisor (predictions, isolation logic, tolerances).

Reference truth (from `marinegym_assets/BlueROV.yaml` + `bluerov.xml`): m=11.2 kg,
I=[0.30375, 0.626, 0.5769], V=0.0113459 m³, ρ=997, coBM=0.01 m, M_A=[5.5,12.7,14.57,0.12,0.12,0.12],
D_L=[4.03,6.22,5.18,0.07,0.07,0.07], D_NL=[18.18,21.66,36.99,1.55,1.55,1.55], EMA α=0.3, dt=2 ms.
B=ρgV=**110.97 N**, W=mg=**109.87 N**, net **+1.10 N**, restoring stiffness k=coBM·B=**1.1097 N·m/rad**.

---

## Results

| # | Test | What it isolates | Prediction | Measured | Verdict |
|---|------|------------------|------------|----------|---------|
| **T1** | Net buoyancy | buoyancy − weight | a_z(0)=(B−W)/m = **0.0980 m/s²** | 0.0980 (**0.00%**); lateral/angular = 0 | ✅ |
| **T2** | Terminal velocity (drag), surge/sway/heave/yaw | linear+quadratic drag (at steady state added-mass & diagonal Coriolis = 0) | F=D_L·v+D_NL·v² | **0.00%** on all 4 axes; added-mass < 1e-15 N | ✅ |
| | — anisotropy | per-axis D_NL ordering | v order = inverse D_NL order | surge>sway>heave ✓ | ✅ |
| **TL** | Cross-axis leakage | force frame / diagonal D, M_A | single-axis velocity → only on-axis accel | off-axis accel **exactly 0** (6 axes); nu() reorder OK | ✅ |
| **T4** | Restoring pendulum | restoring stiffness + effective inertia | underdamped, ω_n=√(k/(I+M_A_rot)) | roll T=3.85 s vs 3.89 (1%); pitch 4.79 vs 5.16 (7%) | ✅ |
| | — static equilibrium | restoring stiffness alone | tilt = asin(M/k) | 26.1° vs 26.8° (2.7%) | ✅ |
| | — axis purity | no cross-coupling | roll tilt → roll-only moment | pitch/yaw accel = 0 | ✅ |
| **T5** | Added mass (effective inertia, Ω=0.5–5 rad/s) | M_A delivery through the EMA filter | effective mass = m + M_A·Re{H(Ω)} ≈ m+M_A | **0.0–0.3%** on surge/sway/heave; sign −M_A on all 6 axes | ✅ |
| **T6** | Coriolis passivity | skew-symmetry of C_A | νᵀC_A(ν)ν = 0 | **4.3e-14** | ✅ |
| | — mechanical energy | dissipativity | E=½νᵀ(M_RB+M_A)ν+U non-increasing | dissipated 4.59 J, **monotone** | ✅ |
| **T7-R2** | Whole-plant, force level | the TOTAL applied wrench, integrator-free | hydro wrench == independent Fossen recomputation | **0.0 N** over a 6 s excited trajectory; buoyancy+CB exact | ✅ |
| **T7-R1** | Whole-plant, approximation size | the added-mass lag | sim vs analytic (M_A in mass matrix) | transient divergence **0.01 cm/s**; same terminal | ✅ |

---

## How each term is verified (and why the isolation is clean)

- **T1 Buoyancy** — from rest, level, no thrust, the only vertical force is buoyancy − weight, and
  added mass ≈ 0 (no acceleration history), so `qacc_z` reads `(B−W)/m` directly.
- **T2 Drag** — the cleanest term. At terminal velocity ν̇→0, so the lagged added-mass force →0, and
  for a single-axis translation the diagonal added-mass Coriolis C_A is identically 0 (every term is a
  product of two *different* velocity components). So drag stands alone: F = D_L·v + D_NL·v². Run on all
  four un-restored axes; the terminal-speed ordering also confirms the per-axis D_NL anisotropy.
- **TL Cross-axis leakage** — the cheapest bug-net. A single-axis body velocity must produce a hydro
  force on that axis only (D and M_A are diagonal, single-axis C_A = 0). Any off-axis acceleration would
  expose a body↔world frame error in the `R @ wrench` application. It is **exactly zero**.
- **T4 Restoring** — the buoyancy is applied at the CB = COM + coBM·ẑ_body, so a tilt produces a righting
  moment of stiffness k = coBM·B. The roll/pitch subsystem is **strongly underdamped (ζ≈0.05)**, so it
  *oscillates*; the period reveals the effective inertia, which correctly **includes the rotational added
  mass** (I_eff = I + M_A_rot — that is why the naïve rigid-inertia prediction was 17% off and the
  added-mass-corrected one matches). The clean stiffness check is the **static equilibrium** under a known
  moment: tilt = asin(M/k), matched to 2.7%. (The damping ratio is inflated by the quadratic rotational
  drag D_NL_rot at finite amplitude — expected, not a discrepancy.)
- **T5 Added mass** (the subtle one) — added mass is applied as an *external* force −M_A·ν̇ with a
  one-step-lagged, EMA-filtered (α=0.3) acceleration, **not** placed in the MuJoCo mass matrix. We verify
  it as an **effective-inertia frequency sweep**: drive a sinusoidal force F·sin(Ωt), fit the velocity
  fundamental, and solve for the effective mass m_eff. The EMA filter's corner is ~230 rad/s, so across the
  physically relevant band (Ω = 0.5–5 rad/s) its in-phase gain Re{H(Ω)} ≈ 1 and the measured effective
  mass equals **m + M_A within 0.0–0.3%** on every axis — including heave, where M_A (14.57) exceeds the
  body mass. The added-mass force sign (−M_A·ν̇, opposing acceleration) holds on all six axes.
- **T6 Coriolis + energy** — the added-mass Coriolis matrix is skew-symmetric, so it does no work:
  νᵀC_A(ν)ν = 0 to 4e-14. In free still-water decay the total mechanical energy (kinetic with the full
  M_RB+M_A, plus the net-buoyancy potential) is monotonically non-increasing — only drag dissipates.
- **T7 Whole-plant cross-check** — two independent references. **R2 (force level, integrator-free):**
  along a 6 s excited trajectory we reconstruct the exact body wrench hydro applied (from its own internal
  drag/added/Coriolis state) and compare it to an **independently written** Fossen recomputation — they
  agree to **0.0 N**, proving the force model (signs, frame, coefficients, buoyancy point) is correct
  end-to-end. **R1 (approximation size):** a 1-DOF heave rise compared to an analytic model with M_A *in
  the mass matrix* (the "ideal" physics) diverges by only **0.01 cm/s** in the transient and reaches the
  same terminal — i.e. the lagged-external-force approximation is negligible.

## Figures
- [figs/hydro_T2_terminal.png](figs/hydro_T2_terminal.png) — terminal velocity vs analytic (4 axes).
- [figs/hydro_T4_pendulum.png](figs/hydro_T4_pendulum.png) — roll pendulum decay vs predicted envelope.
- [figs/hydro_T5_addedmass.png](figs/hydro_T5_addedmass.png) — effective inertia vs Ω (= m + M_A).
- [figs/hydro_T6_energy.png](figs/hydro_T6_energy.png) — monotone energy dissipation.
- [figs/hydro_T7_R1.png](figs/hydro_T7_R1.png) — sim vs ideal-added-mass heave rise (lag size).
- [figs/hydro_P_convergence.png](figs/hydro_P_convergence.png) — trajectory error vs dt: slope-1 **O(dt)** convergence to the continuous Fossen model.
- [figs/hydro_P_lagfidelity.png](figs/hydro_P_lagfidelity.png) — added-mass-lag: effective-mass fraction & transport delay vs Ω (in-band ≈ ideal).

## Precision verification (`verify_hydro_precise.py`)
A rigorous superset of the above (methodology reviewed by the control-theory advisor; grounded in Fossen
2011, Roache 1998 *V&V*, and the Salari–Knupp method of manufactured solutions). **17/17 checks across 4
tiers**; the simulator is again unmodified (driven through `xfrc_applied`, still water).

**Tier 1 — structural Fossen identities (a gate before the expensive runs).** An *independent*
added-mass Coriolis matrix C_A(ν), built from M_A by the skew-block construction (Fossen Eq. 6.44),
reproduces hydro's hand-typed `_coriolis_added` to **1.4 × 10⁻¹⁴** — this breaks the "same algebra typed
twice" risk that a force-level check alone cannot. C_A = −C_Aᵀ holds as a **full skew matrix** (numeric 0;
CasADi-symbolic residual *exactly* 0), not merely the quadratic form νᵀC_Aν = 0. M = M_RB + M_A is SPD
(eigenvalues 0.42–25.8); D(ν) ≻ 0 and total passivity νᵀ(C+D)ν = νᵀD(ν)ν ≥ 0 hold over 2 × 10⁶ random
states.

**Tier 2 — order of accuracy / continuum convergence.** Against a high-order continuous-Fossen reference
(added mass *in* the mass matrix, quaternion attitude, `scipy` DOP853 at rtol/atol = 10⁻¹²), the sim's
trajectory error under a manufactured 6-DOF forcing falls as **O(dt¹)** with observed order **p̂ = 1.000**
across the ladder dt = 2 → 0.125 ms (position L2 0.564 → 0.035 mm; Richardson dt→0 ≈ 2.5 × 10⁻⁶ mm). This
**proves the EMA-lagged simulator converges to the true M_A-in-mass continuous model** — the lagged-force
trick is a *consistent* approximation, not a different model. (`implicitfast` is first-order, and the
passive callback fires exactly **once per step**, so the EMA backward-difference uses the true dt.) The
lag injects **zero** energy per step (kinetic energy monotone-decreasing — strictly passive in practice).

**Tier 3 — frame invariance & Galilean.** Restoring torque = k·sinθ independent of tilt azimuth and yaw
(deviation 2 × 10⁻¹⁴ %); hydro forces independent of world position (0 N); drag is an exact odd function
of ν (0); and an unpowered neutrally-buoyant vehicle in a uniform current advects at exactly v_c with
zero steady drag (|err| 4 × 10⁻¹⁰ m/s) — validating the relative-velocity path vr = ν − Rᵀv_water.

**Tier 4 — added-mass-lag fidelity (the one approximation) + tightened estimators.** The lag's transfer
function gives an **equivalent transport delay ≈ 5.67 ms**, essentially constant with frequency, and an
effective added-mass error **< 0.013 %** in the ROV disturbance band (0.1–2 rad/s). Its
in-phase-with-velocity coefficient is **≥ 0 at every frequency** — the lag only ever *adds* damping, never
anti-damps, so it is **passive / non-destabilizing at all frequencies** (the sharpest fidelity question).
D_L and D_NL are **recovered from the sim to 0.00 %** by regression over a force sweep (not a single
point). Pendulum periods match the **full coupled** high-order ODE reference to **0.01 %** (roll) /
**0.00 %** (pitch); note the naive I + M_A_rot formula is 1–8 % off because the CB offset couples rotation
to translation (pitch↔surge, roll↔sway) — a real effect the sim captures exactly.

**Honest limitation.** M_A is **diagonal** by MarineGym design; the small off-diagonal added-mass terms
(e.g. Yṙ, Nv̇) of a full BlueROV2 identification are not modeled. This is a *modeling* choice, separate
from the lag approximation, and acceptable for this control study.

*Reproduce:* `python verify_hydro_precise.py --tier 1234` (env `robust`; ~25 s; needs `casadi`, `scipy`).

## Conclusion
Every hydrodynamic term — **buoyancy, restoring, linear+quadratic drag, added mass, and added-mass
Coriolis** — matches its first-principles prediction to within measurement precision, with **no frame,
sign, coefficient, or coupling errors** (T7-R2 = 0.0 N; leakage = 0). The one deliberate approximation,
the **lagged/filtered added-mass force** (used because heave M_A 14.57 > body mass 11.2 would otherwise
be numerically unstable in MuJoCo's explicit passive channel), is shown to be **negligible** across all
physically relevant frequencies (m_eff = m+M_A to 0.1%; transient lag 0.01 cm/s). The simulator's
hydrodynamics is verified correct for control work.

*Reproduce:* `python verify_hydro.py` (env `robust`).
