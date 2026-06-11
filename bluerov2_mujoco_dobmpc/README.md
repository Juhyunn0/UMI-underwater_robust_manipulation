# BlueROV2 DOBMPC in MuJoCo

MuJoCo replication of **"Disturbance Observer-Based Model Predictive Control
for an Unmanned Underwater Vehicle"** (Hu et al., *J. Mar. Sci. Eng.* 2024,
12, 94), built from the paper and the authors' reference implementation
(<https://github.com/HKPolyU-UAV/bluerov2>, Gazebo/uuv_simulator + acados).

```
measurement ──► EAOB (18-state EKF) ──► eta_hat, nu_hat, w_hat ──► NMPC ──► u
                  ▲                                                   │
                  └────────── tau_applied = K · t(u) ◄────────────────┘
```

* **Plant** - MuJoCo free body in a **NED world** (`gravity="0 0 9.81"`).
  MuJoCo natively provides `M_RB` (incl. the `m·z_G` coupling), `C_RB` and
  weight; buoyancy, linear+quadratic damping, added-mass Coriolis `C_A`, the
  added-mass inertial force `−M_A ν̇` (one-substep-lagged, low-pass-filtered -
  the same technique uuv_simulator uses), thruster wrench, and external
  disturbances are injected per 2 ms substep via `xfrc_applied`.
* **EAOB** - 18-state EKF `[η; ν; w]` with `ẇ = 0`, RK4 prediction, and the
  reference code's measurement model `z = [η, ν, τ]` where
  `h_τ = M a_meas + C_RB ν + D(ν)ν + g(η) − w`; the `−I` dependence on `w`
  makes the disturbance near-deadbeat observable. Q, R, P₀ follow
  `bluerov2_dob.cpp` exactly.
* **NMPC** - full Fossen model (Eq. 22) with the disturbance as a parameter,
  N = 60 × 0.05 s, weights from paper Table 4, multiple shooting + RK4,
  CasADi/Ipopt (~0.1 s per solve). **DOBMPC** = NMPC fed with EAOB state and
  `ŵ`; **baseline MPC** = same NMPC with `ŵ = 0`; **PID** = Table 5 gains.

## Install & run

```bash
pip install -r requirements.txt          # mujoco, casadi, numpy, scipy, matplotlib

python scripts/validate_plant.py         # model equivalence + MuJoCo fidelity
python scripts/run_dp.py        --controller all --dist constant  --T 25
python scripts/run_dp.py        --controller all --dist periodic  --T 25
python scripts/run_tracking.py  --traj circle     --dist mixed    --T 25
python scripts/run_tracking.py  --traj lemniscate --dist mixed    --T 25
python scripts/view_model.py             # interactive viewer (needs display)
```

Figures, RMSE tables → `results/`. Useful flags: `--plant analytic`
(pure-Python Fossen RK4 plant, same interface), `--N 30` (shorter horizon,
~2x faster), `--seed`, `--dist none`.

## Package layout

```
bluerov2mj/
  params.py        all physical/controller constants (paper + repo provenance)
  fossen.py        NumPy Fossen model: M, C_RB, C_A, D, g, J, RK4, quaternions
  allocation.py    u=[X,Y,Z,N] -> 6 thruster forces (pinv of K) -> wrench K·t
  bluerov2.xml     MJCF: NED world, free body, inertial pos (0 0 0.02)
  mujoco_env.py    MuJoCo plant with hydrodynamic force injection
  analytic_env.py  ground-truth Fossen RK4 plant (validation / fast prototyping)
  disturbances.py  periodic waves, constant current, superposition (Sec. 5)
  eaob.py          Extended Active Observer (CD-EKF, Eq. 25-37)
  experiment.py    closed-loop harness, references, RMSE
  plots.py         paper-style figures
  controllers/
    mpc.py         CasADi NMPC (disturbance-parameterised prediction model)
    pid.py         Table 5 PID baseline
scripts/
  validate_plant.py  run_dp.py  run_tracking.py  view_model.py
```

## Verified results (this exact code, seed 1, `--N 40`)

Dynamic positioning, constant current (10 N x/y/z + 5 Nm yaw at t = 10 s),
RMSE over 25 s:

| controller | x [m] | y [m] | z [m] | yaw [rad] |
|---|---|---|---|---|
| PID    | 1.2005 | 1.1649 | 0.8223 | 0.4801 |
| MPC    | 0.1045 | 0.1019 | 0.1430 | 0.0559 |
| DOBMPC | **0.0074** | **0.0031** | **0.0150** | **0.0015** |

Circle tracking (r = 2 m, 1 m/s) under mixed disturbance (3-6 N waves +
10 N / 3 Nm step at t = 4 s):

| controller | x [m] | y [m] | z [m] | yaw [rad] |
|---|---|---|---|---|
| PID    | 2.1096 | 2.0778 | 1.1192 | 0.2034 |
| MPC    | 0.1667 | 0.1524 | 0.2100 | **0.0520** |
| DOBMPC | **0.1030** | **0.0837** | **0.0820** | 0.0825 |

The EAOB tracks step and sinusoidal disturbances within 1-2 samples with
~0.5 N noise-induced jitter (see `results/*_eaob.png`), matching the
paper's Figs. 5/9. In tracking, DOBMPC's yaw RMSE is slightly *worse* than
the baseline's: a ~0.2 rad transient right after the disturbance step while
circling at 1 m/s (the body-frame `w_hat` is held constant over the horizon
- Assumption 2 - while the true world-frame step rotates in the body frame).
Position-channel gains of 1.6-2.6x dominate, as in the paper. Figures were
produced with `--N 40` for speed; the shipped default is the paper's N = 60
(closed-loop behaviour is indistinguishable, ~1.5x slower).

## Design decisions & caveats

1. **Frames.** Pure Fossen NED+FRD exactly as the paper; the MuJoCo world *is*
   NED (gravity +z). The interactive viewer therefore looks upside-down; the
   physics doesn't care. (The reference repo mixes Gazebo's ENU with
   NED-style equations - replicated *behaviour*, not its sign conventions.)
2. **Added mass.** Anisotropic translational added mass cannot be folded into
   a rigid-body inertia, so `−M_A ν̇` is applied with a one-substep-lagged,
   EMA-filtered (α = 0.3) acceleration. Validated open-loop against the
   analytic Fossen integrator: ≤ 0.2 cm / 0.4° divergence over 10 s of
   aggressive excitation (`validate_plant.py`).
3. **Thruster coupling.** The plant applies the *full* 6-DOF wrench `K t`,
   including the small roll/pitch moments the 4-DOF controller doesn't model -
   intentional unmodelled dynamics absorbed by the EAOB, as in the paper.
4. **MPC weights.** Q, N, dt follow paper Table 4. The paper's
   `R = [15 15 15 0.5]` lives in *thrust-normalised* control units
   (`u_scaled ≈ F/106` in the reference code), i.e. an effective penalty of
   ~1.3e-3 F² in Newtons; copying `15` verbatim into Newton units makes
   within-horizon drifting cheaper than cancelling a 10 N disturbance and
   even DOBMPC stops rejecting it (we verified this). `R = [0.5 0.5 0.5
   0.05]` reproduces the paper's behaviour: baseline MPC settles with the
   reported ~0.2-0.5 m offset, DOBMPC stays near zero.
5. **Solver.** CasADi/Ipopt (SX, multiple shooting, warm-started) ≈ 0.1 s per
   step at N = 60 - fine for simulation, not real-time. The model/cost are
   written 1:1 to the paper's acados formulation; port to acados SQP-RTI for
   hardware rates. CasADi's `sqpmethod` is kept as an option but its exact
   Hessian is not robust on this problem.
6. **EAOB measurement model** includes `C_A(ν)ν` (the reference code omits
   it), keeping `h()` consistent with the full-Fossen `f()` and the MPC
   model; otherwise `-C_A ν` is folded into `ŵ` as a phantom disturbance
   during sustained motion and double-counted by the MPC.
   **Noise covariances** follow the reference code (`R = I·dt⁴/4`),
   which trusts measurements heavily; with the default measurement noise the
   disturbance estimate carries a few-N jitter (visible in the paper's plots
   too). Tune `eaob_kwargs` in `experiment.run_closed_loop` for your sensors.
7. **RL-readiness.** All hydrodynamics live in `fossen.py` + a thin injection
   layer (`mujoco_env._apply_forces`), independent of the MJCF asset - the
   intended port path to MJX/Warp (vectorise `fossen.py`, keep the XML).
