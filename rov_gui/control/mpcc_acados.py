#!/usr/bin/env python3
"""mpcc_acados.py — Model Predictive CONTOURING Control for the ROV.

The difference from the tracking NMPC (``dobmpc/mpc_acados.py``) in one line:
there, the reference is a function of TIME and the solver is told "be here at
t"; here, the path position is a decision variable and the solver is told "stay
on this curve, and get along it as fast as the curve allows".

    state   x = [eta(6), nu(6), theta]          13
    input   u = [tau(6), v_theta]                7
    theta   arclength along the mission path [m], theta_dot = v_theta

The cost is the standard MPCC split of the position error into the path's own
frame, plus a progress term:

    e_c   CONTOURING error — distance off the curve. This is the mission
          requirement, so it carries the big weight.
    e_l   LAG error — how far theta has run from the vehicle's own foot point
          on the curve. It is NOT a mission requirement; its job is to keep
          theta an honest PROJECTION, because e_c is only the true
          distance-to-curve while theta sits at the foot point. Let theta run
          far enough ahead and the contouring frame quietly re-anchors to a
          piece of path the vehicle has not reached — measured on hardware
          2026-08-17, theta 29 cm ahead sat on a DIFFERENT SEGMENT for 44.6 %
          of ticks and quadrupled cross-track error there.
          The fix for that is NOT simply a bigger W_LAG, which is what the
          simulator suggested and the vehicle rejected (see the weights below).
          It is keeping the fillet radius comfortably larger than the
          cross-track error the vehicle actually achieves: the projection is
          degenerate once |e_c| > R, and the 44.6 % run had R = 0.06 m against
          an 8.7 cm p95 error.
    v_theta pushed toward a CONSTANT ceiling (the operator's commanded speed).
          Expanded, q*(v_theta - v_cmd)^2 is the classic linear progress reward
          plus a regularizer — and being a constant it encodes no clock, so
          there is nothing to be "late" against. Cornering speed is EMERGENT:
          pushing v_theta through a corner costs contouring error, so the
          optimizer backs off by exactly as much as the geometry demands.

THE PATH IS A RUNTIME PARAMETER, not baked into the generated C. Every mission
would otherwise mean a fresh code-generation and compile at the moment the
operator presses START. Instead the path enters as six CasADi B-splines over a
FIXED normalized grid, whose coefficients are acados parameters resampled each
tick over a sliding window ahead of the vehicle:

    px, py      the curve
    tcos, tsin  its unit tangent, carried explicitly rather than differentiated
                out of px/py — at an in-place reversal dp/dtheta is zero and a
                frame built from it is singular (path_geometry._Pivot)
    psi_cmd     the commanded heading (path tangent, or a fixed crab heading)

(The feasible SPEED profile is deliberately not a channel — it is a per-stage
yref constant. See N_CH.)

Window length must cover what the horizon can traverse: V_THETA_MAX * N * dt,
plus margin behind so the projection is defined when the vehicle is late.
"""

from __future__ import annotations

import os

import casadi as ca
import numpy as np

import sys
_SIM = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "bluerov2_mujoco_marinegym")
if _SIM not in sys.path:
    sys.path.insert(0, _SIM)

from dobmpc import _acados_env  # noqa: F401,E402  (ACADOS_SOURCE_DIR etc.)
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel  # noqa: E402

from dobmpc import params as P  # noqa: E402
from dobmpc.mpc import _f_casadi  # noqa: E402

GEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_mpcc_gen")

NX = 13                  # [eta(6), nu(6), theta]
NU = 7                   # [tau(6), v_theta]
# The window is sized for the HORIZON (metres of path), so its resolution is
# what has to resolve the smallest feature — the fillet. At 0.2 m/s the window
# is 2.1 m, so 41 points is 5.3 cm spacing and only ~1.8 samples across a 6 cm
# fillet arc. That sounds too coarse and is not: a B-spline through those
# samples reproduces the arc to 1.5 mm, and the measured closed loop is
# IDENTICAL to 61/81 points (err p50 6.9 / p95 25.4 mm either way, 2026-08-17)
# while costing a third of the solve — 11.8 ms p99 against 19.0 and 24.0.
# Raise it only if a fillet ever gets small enough that 1.5 mm matters.
# A CIRCLE has no fillet — the path IS the curved feature — and it turns out to
# be the EASIER case for this fit, not the harder one: 0.12 mm worst over the
# panel's whole radius x speed range, pinned by rov_gui/tests/test_mpcc.py ::
# test_the_window_spline_reproduces_a_circle_at_every_offered_radius. That
# holds only because HwMpcc._install sizes the window on the FEASIBLE speed;
# sized on the commanded one, a 0.15 m circle asked to fly at 0.50 m/s put 5.2
# laps inside one window and the same fit degraded to 3.2 mm.
N_GRID = 41              # window samples per channel
# FIVE channels, not six: v_ref is deliberately NOT one of them. Making the
# speed target a function of theta puts v_ref'(theta) into the Gauss-Newton
# Hessian for theta (H += W_PROGRESS * v_ref'^2), and a brisk profile turns
# that into a barrier the RTI's single iteration cannot cross — measured
# 2026-08-17: a_long 0.05 -> 0.118 m/s of progress, a_long 0.15 -> ZERO, same
# path, same everything else. The speed target is a per-STAGE yref constant
# instead (see solve/v_ref_schedule): identical schedule, no theta coupling.
N_CH = 5                 # px, py, tcos, tsin, psi_cmd
BACK_M = 0.25            # how much of the window sits BEHIND theta0
V_THETA_MAX = 0.60       # hard cap on progress rate [m/s]

# PROGRESSIVE shooting grid. A uniform 60 x 50 ms horizon is 3 s, and at the
# 0.05-0.12 m/s this vehicle actually flies that is only 0.15-0.36 m of PATH —
# less than one filleted corner, so the horizon cannot see the corner it is
# about to take. Stretching the later intervals buys preview without buying
# QP size: the first block stays at the control period (that is the part that
# becomes the command), the tail coarsens.
#
# The substep counts are NOT free tuning. The roll/pitch dynamics are stiff
# (|lambda| ~ 70-85 1/s, dobmpc/mpc._rk4), so RK4 needs h < 2.78/85 = 33 ms to
# stay INSIDE its stability region — a coarse interval integrated in one step
# would not be inaccurate, it would explode. Each block therefore carries
# enough substeps to hold h at ~25 ms, the same h the tracking NMPC uses.
GRID_BLOCKS = ((24, 0.05, 2),    # 1.2 s at the control period
               (20, 0.10, 4),    # +2.0 s
               (16, 0.25, 10))   # +4.0 s   -> 7.2 s total, 60 stages

SLACK_L2 = 1.0e3
SLACK_L1 = 1.0e2

# Residual layout: [e_c, e_l, e_z, e_psi, phi, pitch, p, q, r, dv, tau(6)]
NY = 16
NY_E = 9

# Weights. The lag/progress pair is the PATH-vs-TRAJECTORY knob, and it has now
# been decided by HARDWARE rather than by simulation. A 2026-08-17 sweep against
# a plant weakened to 0.15x thrust preferred 600/100 on the metric "distance to
# the curve":
#
#   W_LAG W_PROG | lag_p50 | curve p50/p95 | speed | theta >1/4 leg ahead
#      60    400 | 20.6 cm | 3.38/6.85 cm  | 0.150 | 32.7 %
#     600    100 |  2.5 cm | 0.77/4.17 cm  | 0.092 |  0.0 %
#
# The vehicle disagreed. Auditing all 118 recorded runs, the best full mission
# on file is 60/400 (0817_110145: 5/5 laps, 18.71 m, cross-track p50 2.13 /
# p95 6.59 cm, zero actuator saturation), while 600/100 without deadband
# compensation covered 1.16 m of a 19.5 m mission before it was stopped
# (0817_122659). The simulation's plant is optimistic in exactly the way that
# matters here — it has no ESC deadband and its drag is 8-12x too low — so its
# ranking of a knob that trades push against geometry does not transfer.
# 60/400 is therefore the shipped pair, and the sweep above is kept only as a
# record of what the simulator thought.
W_CONTOUR = 300.0
W_LAG = 60.0
W_Z = 150.0
W_PSI = 150.0
W_ROLL = 80.0
W_PITCH = 80.0
W_RATE = 10.0
# (History worth keeping: while the speed target was a FUNCTION of theta, a
# W_PROGRESS of 40 deadlocked the mission outright — theta never advanced, so
# v_ref never rose, and acados reported status 0 throughout. That coupling is
# gone: the target is a constant now, so this weight is a plain trade and no
# longer has a cliff in it.)
W_PROGRESS = 400.0
R_VTHETA = 0.05


def _splines():
    """One parametric B-spline shared by every channel (same fixed grid)."""
    grid = np.linspace(0.0, 1.0, N_GRID)
    f = ca.interpolant("mpcc_ch", "bspline", [grid.tolist()], 1)
    n_c = int(f.size_in(1)[0])
    # The coefficient vector is not the sampled values in general; the grid is
    # fixed, so the collocation matrix is too and inverting it once turns
    # "values I sampled" into "coefficients acados wants". (On CasADi 3.7 this
    # matrix comes out as the identity, but solving for it is version-proof.)
    A = np.zeros((N_GRID, n_c))
    for i in range(n_c):
        e = np.zeros(n_c)
        e[i] = 1.0
        for j, xj in enumerate(grid):
            A[j, i] = float(f(xj, e))
    return f, n_c, np.linalg.inv(A)


class AcadosMPCC:
    """SQP-RTI contouring controller. ``solve(x, w_hat, params) -> u``."""

    def __init__(self, build=True, blocks=GRID_BLOCKS):
        steps = np.concatenate([np.full(n, h) for n, h, _s in blocks])
        substeps = np.concatenate([np.full(n, s, dtype=int)
                                   for n, _h, s in blocks])
        self.time_steps = steps
        self.N = int(steps.size)
        self.dt = float(steps[0])            # the control period = stage 0
        self.horizon_s = float(steps.sum())
        self.spline, self.n_coeff, self._A_inv = _splines()
        self.back_m = BACK_M
        # w_hat(6) + theta0 + window length + the six channels. The window is
        # a PARAMETER, not a constant: its length has to cover what the horizon
        # can traverse (v * horizon_s), which the operator changes with the
        # speed box, and a window sized for 0.5 m/s would resolve a 0.15 m
        # fillet with 14 cm samples if it were fixed.
        self.n_param = 6 + 2 + N_CH * self.n_coeff

        x = ca.SX.sym("x", NX)
        u = ca.SX.sym("u", NU)
        p = ca.SX.sym("p", self.n_param)
        xdot = ca.SX.sym("xdot", NX)

        w = p[0:6]
        theta0 = p[6]
        window = p[7]
        ch = [p[8 + i * self.n_coeff: 8 + (i + 1) * self.n_coeff]
              for i in range(N_CH)]

        theta = x[12]
        # normalized window coordinate, clamped so a horizon that overruns the
        # window evaluates the edge instead of extrapolating a B-spline
        s = (theta - (theta0 - self.back_m)) / window
        s = ca.fmin(ca.fmax(s, 0.0), 1.0)
        px = self.spline(s, ch[0])
        py = self.spline(s, ch[1])
        tc = self.spline(s, ch[2])
        ts = self.spline(s, ch[3])
        psi_cmd = self.spline(s, ch[4])
        nrm = ca.sqrt(tc * tc + ts * ts + 1e-9)
        tc, ts = tc / nrm, ts / nrm

        dx, dy = x[0] - px, x[1] - py
        # Repo sign conventions (workers._path_split): cross is positive to the
        # LEFT of travel — in NED the left normal of heading u is (u_y, -u_x);
        # along is LAG, positive when the vehicle is BEHIND the path point.
        e_c = ts * dx - tc * dy
        e_l = -(tc * dx + ts * dy)
        e_psi = x[5] - psi_cmd

        f_expl = ca.vertcat(_f_casadi(x[0:12], u[0:6], w), u[6])

        m = AcadosModel()
        m.name = f"mpcc_{P.MODEL}"
        m.x, m.u, m.p, m.xdot = x, u, p, xdot
        m.f_expl_expr = f_expl
        m.f_impl_expr = xdot - f_expl
        # e_z rides yref (the depth setpoint is a scalar, not a path channel)
        # u[6] rides against a per-stage yref (the speed schedule), so the
        # residual carries no theta dependence at all.
        y = ca.vertcat(e_c, e_l, x[2], e_psi, x[3], x[4],
                       x[9], x[10], x[11], u[6], u[0:6])
        y_e = ca.vertcat(e_c, e_l, x[2], e_psi, x[3], x[4],
                         x[9], x[10], x[11])
        m.cost_y_expr, m.cost_y_expr_e = y, y_e

        ocp = AcadosOcp()
        ocp.model = m
        ocp.solver_options.N_horizon = self.N
        ocp.solver_options.tf = self.horizon_s
        ocp.solver_options.time_steps = steps
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        w_run = np.array([W_CONTOUR, W_LAG, W_Z, W_PSI, W_ROLL, W_PITCH,
                          W_RATE, W_RATE, W_RATE, W_PROGRESS]
                         + list(np.asarray(P.MPC_R, float)))
        ocp.cost.W = np.diag(w_run)
        ocp.cost.W_e = np.diag(w_run[:NY_E])
        ocp.cost.yref = np.zeros(NY)
        ocp.cost.yref_e = np.zeros(NY_E)
        ocp.parameter_values = np.zeros(self.n_param)

        u_max = np.concatenate([np.asarray(P.U_MAX, float), [V_THETA_MAX]])
        ocp.constraints.idxbu = np.arange(NU)
        ocp.constraints.lbu = np.concatenate(
            [-np.asarray(P.U_MAX, float), [0.0]])   # progress never negative
        ocp.constraints.ubu = u_max

        idxbx = np.array([3, 4, 6, 7, 8])           # roll, pitch, linear vel
        th = P.THETA_MAX if getattr(P, "PITCH_AWARE", False) else 1.2
        v_max = P.V_MAX
        lbx = np.array([-1.2, -th, -v_max, -v_max, -v_max])
        ubx = np.array([1.2, th, v_max, v_max, v_max])
        ocp.constraints.idxbx, ocp.constraints.lbx, ocp.constraints.ubx = \
            idxbx, lbx, ubx
        ocp.constraints.idxbx_e, ocp.constraints.lbx_e, \
            ocp.constraints.ubx_e = idxbx, lbx, ubx
        ocp.constraints.x0 = np.zeros(NX)
        ns = len(idxbx)
        ocp.constraints.idxsbx = np.arange(ns)
        ocp.constraints.idxsbx_e = np.arange(ns)
        ocp.cost.zl = SLACK_L1 * np.ones(ns)
        ocp.cost.zu = SLACK_L1 * np.ones(ns)
        ocp.cost.Zl = SLACK_L2 * np.ones(ns)
        ocp.cost.Zu = SLACK_L2 * np.ones(ns)
        ocp.cost.zl_e = SLACK_L1 * np.ones(ns)
        ocp.cost.zu_e = SLACK_L1 * np.ones(ns)
        ocp.cost.Zl_e = SLACK_L2 * np.ones(ns)
        ocp.cost.Zu_e = SLACK_L2 * np.ones(ns)

        so = ocp.solver_options
        so.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        so.hessian_approx = "GAUSS_NEWTON"
        so.integrator_type = "ERK"
        so.sim_method_num_stages = 4
        so.sim_method_num_steps = substeps      # per-interval: holds h ~ 25 ms
        so.nlp_solver_type = "SQP_RTI"
        so.qp_solver_iter_max = 50

        gen = os.path.join(GEN_DIR, P.MODEL)
        os.makedirs(gen, exist_ok=True)
        ocp.code_export_directory = gen
        self.solver = AcadosOcpSolver(
            ocp, json_file=os.path.join(gen, "acados_ocp_mpcc.json"),
            build=build, generate=build, verbose=False)
        self._u_prev = np.zeros(NU)
        self._warm = False
        self.n_fail = 0
        self.last_status = 0

    # ------------------------------------------------------------ parameters
    def window_for(self, v_cmd: float) -> float:
        """Window arclength for a commanded speed: everything the horizon can
        traverse, with 30 % margin, floored so a slow mission still resolves
        the corner it is heading for."""
        reach = 1.3 * max(0.0, float(v_cmd)) * self.horizon_s
        return float(min(6.0, max(1.0, self.back_m + reach)))

    def pack_params(self, w_hat, theta0: float, path, psi_fixed,
                    heading_follow: bool, s_grid, v_grid,
                    window_m: float, psi_branch: float = 0.0) -> np.ndarray:
        """Resample the path over the window and pack the acados parameter.

        ``psi_branch`` is the vehicle's CURRENT yaw. The commanded heading is
        shifted onto the 2*pi branch nearest it, because the OCP integrates psi
        as a plain state: an unshifted reference 2*pi away is a full extra
        turn's worth of cost, not a no-op (the tracking bridge does the same
        thing at mpc_bridge._xref_ned_traj)."""
        u = np.linspace(0.0, 1.0, N_GRID)
        # NOT clipped into [0, L] — ArcPath.sample extends the terminal
        # tangents on purpose, because a clamped (flat) window has zero path
        # gradient at theta0 and the mission never starts. See ArcPath.sample.
        s = (theta0 - self.back_m) + u * float(window_m)
        px, py, psi_path, _k = path.sample(s)
        if heading_follow:
            psi_cmd = np.asarray(psi_path, float).copy()
        else:
            psi_cmd = np.full(N_GRID, float(psi_fixed))
        psi_cmd = psi_cmd + 2.0 * np.pi * np.round(
            (float(psi_branch) - psi_cmd[0]) / (2.0 * np.pi))
        chans = (px, py, np.cos(psi_path), np.sin(psi_path), psi_cmd)
        out = np.empty(self.n_param)
        out[0:6] = np.clip(np.asarray(w_hat, float).ravel()[:6], -50.0, 50.0)
        out[6] = float(theta0)
        out[7] = float(window_m)
        for i, values in enumerate(chans):
            out[8 + i * self.n_coeff: 8 + (i + 1) * self.n_coeff] = \
                self._A_inv @ np.asarray(values, float)
        return out

    # ----------------------------------------------------------------- solve
    def solve(self, x, params, z_ref: float, v_cmd: float = 0.0):
        """x (13,), packed params, depth setpoint, speed CEILING -> u (7,).

        ``v_cmd`` is a CONSTANT — the operator's commanded speed — and that is
        the whole difference between path tracking and trajectory tracking in
        this controller.

        A time-varying schedule (what this took until 2026-08-17) says "you
        should be THIS far along by now", so falling behind is an error the
        cost fights, and the equilibrium is a haggle between the schedule and
        the lag term. That is trajectory tracking wearing a contouring hat, and
        the operator could see it: theta ran 29 cm ahead of the hull and spent
        44.6 % of ticks on a different segment.

        A constant target is the classic MPCC progress REWARD in least-squares
        clothing: expanding q*(v_theta - v_cmd)^2 gives q*v_theta^2 -
        2*q*v_cmd*v_theta + const, i.e. "go as fast along the path as you can"
        plus a regularizer. There is no clock, nothing to be late against, and
        no dependence on theta at all — so it also cannot rebuild the
        Gauss-Newton barrier that a theta-dependent v_ref created.

        Cornering speed is then EMERGENT rather than precomputed: pushing
        v_theta through a corner costs contouring error, so the optimizer backs
        off exactly as much as the geometry demands. That is the property the
        speed profile was faking."""
        x = np.asarray(x, float)
        s = self.solver
        s.set(0, "lbx", x)
        s.set(0, "ubx", x)
        yref = np.zeros(NY)
        yref[2] = float(z_ref)
        yref[9] = float(v_cmd)
        yref_e = np.zeros(NY_E)
        yref_e[2] = float(z_ref)
        for k in range(self.N):
            s.set(k, "p", params)
            s.set(k, "yref", yref)
        s.set(self.N, "p", params)
        s.set(self.N, "yref", yref_e)
        if not self._warm:
            for k in range(self.N + 1):
                s.set(k, "x", x)
            for k in range(self.N):
                s.set(k, "u", np.zeros(NU))
            self._warm = True
        status = int(s.solve())
        self.last_status = status
        u = np.asarray(s.get(0, "u"), float)
        if status in (0, 2) and np.isfinite(u).all():
            self._u_prev = u.copy()
            return u.copy()
        self.n_fail += 1
        self._warm = False
        return self._u_prev.copy()

    def horizon_theta(self) -> np.ndarray:
        """theta at every stage — how far along the path the horizon reaches."""
        return np.array([float(self.solver.get(k, "x")[12])
                         for k in range(self.N + 1)])

    def solve_ms(self) -> float:
        try:
            return 1.0e3 * float(self.solver.get_stats("time_tot"))
        except Exception:                                     # noqa: BLE001
            return float("nan")

    def reset(self) -> None:
        self._u_prev = np.zeros(NU)
        self._warm = False
        self.n_fail = 0
