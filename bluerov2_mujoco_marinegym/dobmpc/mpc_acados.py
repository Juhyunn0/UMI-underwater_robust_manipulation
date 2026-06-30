"""acados SQP-RTI port of the DOB-MPC NMPC (paper Sec. 4, Eq. 44).

Same OCP as dobmpc.mpc.NMPC -- same Fossen prediction model (reuses the EXACT
symbolic dynamics dobmpc.mpc._f_casadi as a single source of truth), same
N=60, dt, Q/R/QN, option-(b) bounds, and the disturbance wrench w as an on-line
parameter. Drop-in: AcadosNMPC.solve(x, w_hat, xref) -> u has the identical
signature to NMPC.solve, so DOBMPCController switches solvers via params.SOLVER.
The IPOPT NMPC (dobmpc.mpc.NMPC) stays as the reference / fallback solver.

By design, acados differs from the IPOPT NMPC in *how* the same OCP is solved:
  * SQP-RTI -- ONE Gauss-Newton SQP iteration per tick (not full convergence),
    warm-started from the previous (internally shifted) solution. Because the
    50 ms-apart problems are nearly identical, one step per tick tracks the
    optimal solution -> a fixed, deterministic ~2-5 ms solve, no IPOPT-style
    cold-restart freezes.
  * PARTIAL_CONDENSING_HPIPM -- a structure-exploiting QP solver for the
    time-banded KKT system (vs IPOPT's general sparse MUMPS factorisation).
  * ERK RK4, 2 substeps/interval -- matches mpc._rk4(n_int=2), h = 25 ms.
  * The roll / pitch(=THETA_MAX) / |v_lin| STATE bounds are SOFT (L2 slack with
    a large penalty) so a transient linearisation cannot make the RTI QP
    infeasible and stall the loop (the explicit goal of the port); the control
    bounds stay HARD. IPOPT uses hard state bounds -- in the bound-inactive
    interior the two solvers' optima coincide (the equivalence test checks this).

Cost note: acados LINEAR_LS carries a 1/2 factor the IPOPT cost does not; this
scales the whole objective uniformly and so leaves the minimiser u* unchanged.
"""
import os

import numpy as np
import casadi as ca

from . import _acados_env  # noqa: F401  (sets ACADOS_SOURCE_DIR / LD_LIBRARY_PATH)
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

from . import params as P
from .mpc import _f_casadi, NX, NU

GEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_acados_gen")

# soft-constraint slack penalties (near-hard but feasibility-preserving)
SLACK_L2 = 1.0e3     # Zl / Zu  (quadratic)
SLACK_L1 = 1.0e2     # zl / zu  (linear)


def _build_model(name="dobmpc_bluerov"):
    """AcadosModel wrapping the identical Fossen dynamics used by the IPOPT
    NMPC.  x=[eta;nu] (12), u=[X,Y,Z,N] (4), p=w_hat (6, NED body)."""
    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", NU)
    w = ca.SX.sym("w", 6)
    xdot = ca.SX.sym("xdot", NX)
    f_expl = _f_casadi(x, u, w)          # <-- single source of truth
    m = AcadosModel()
    m.name = name
    m.x, m.u, m.p, m.xdot = x, u, w, xdot
    m.f_expl_expr = f_expl
    m.f_impl_expr = xdot - f_expl
    return m


class AcadosNMPC:
    def __init__(self, N=P.MPC_N, dt=P.DT_CTRL, Q=P.MPC_Q, R=P.MPC_R, QN=P.MPC_QN,
                 u_max=P.U_MAX, v_max=P.V_MAX, rti=True, soft=True, build=True,
                 fallback_ipopt=True):
        self.N, self.dt = int(N), float(dt)
        # parallel run_compare pre-builds the solver once in the parent, then sets this
        # so forked workers LOAD the compiled solver instead of racing to recompile.
        if os.environ.get("DOBMPC_ACADOS_BUILD") == "0":
            build = False
        nx, nu = NX, NU
        ny, ny_e = nx + nu, nx

        # distinct model name / export dir per (rov model, solver variant) so the
        # bluerov2 (NU=4) and heavy (NU=6) C code -- and RTI vs full-SQP -- coexist
        # without overwriting each other (switching ROV_MODEL won't need a rebuild)
        variant = f"{P.MODEL}_{'rti' if rti else 'sqp'}"
        ocp = AcadosOcp()
        ocp.model = _build_model(name=f"dobmpc_{variant}")
        ocp.solver_options.N_horizon = self.N
        ocp.solver_options.tf = self.N * self.dt

        # ---- LINEAR_LS cost:  y = [x; u],  y_e = x
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"
        Vx = np.zeros((ny, nx)); Vx[:nx, :] = np.eye(nx)
        Vu = np.zeros((ny, nu)); Vu[nx:, :] = np.eye(nu)
        ocp.cost.Vx, ocp.cost.Vu = Vx, Vu
        ocp.cost.Vx_e = np.eye(nx)
        ocp.cost.W = np.diag(np.concatenate([np.asarray(Q, float), np.asarray(R, float)]))
        ocp.cost.W_e = np.diag(np.asarray(QN, float))
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ny_e)

        # ---- disturbance parameter (constant over horizon, set each solve)
        ocp.parameter_values = np.zeros(6)

        # ---- hard control bounds
        ocp.constraints.idxbu = np.arange(nu)
        ocp.constraints.lbu = -np.asarray(u_max, float)
        ocp.constraints.ubu = np.asarray(u_max, float)

        # ---- state bounds: roll(3), pitch(4)=THETA_MAX[option-b], lin-vel(6,7,8)
        idxbx = np.array([3, 4, 6, 7, 8])
        th = P.THETA_MAX if getattr(P, "PITCH_AWARE", False) else 1.2
        lbx = np.array([-1.2, -th, -v_max, -v_max, -v_max])
        ubx = np.array([1.2,  th,  v_max,  v_max,  v_max])
        ocp.constraints.idxbx, ocp.constraints.lbx, ocp.constraints.ubx = idxbx, lbx, ubx
        ocp.constraints.idxbx_e, ocp.constraints.lbx_e, ocp.constraints.ubx_e = idxbx, lbx, ubx
        ocp.constraints.x0 = np.zeros(nx)

        if soft:
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

        # ---- solver options
        so = ocp.solver_options
        so.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        so.hessian_approx = "GAUSS_NEWTON"
        so.integrator_type = "ERK"
        so.sim_method_num_stages = 4    # RK4
        so.sim_method_num_steps = 2     # h = dt/2 = 25 ms  (== mpc._rk4 n_int=2)
        so.nlp_solver_type = "SQP_RTI" if rti else "SQP"
        if not rti:
            so.nlp_solver_max_iter = 80
        so.qp_solver_iter_max = 50

        gen_dir = os.path.join(GEN_DIR, variant)
        ocp.code_export_directory = gen_dir
        os.makedirs(gen_dir, exist_ok=True)
        json_file = os.path.join(gen_dir, "acados_ocp_dobmpc.json")
        self.solver = AcadosOcpSolver(ocp, json_file=json_file,
                                      build=build, generate=build, verbose=False)

        self.rti = rti
        self._u_prev = np.zeros(nu)
        self._warm = False
        self.n_fail = 0
        self.n_fallback = 0
        self.last_status = 0
        # robustness: on a (rare) acados NaN/min-step failure, recover with one IPOPT
        # solve for that tick (IPOPT = the validated full-convergence reference) AND
        # re-init the acados iterate so it restarts clean -- without this, a single
        # NaN cascades (RTI warm-starts from the corrupted iterate and diverges; see
        # the 2026-06-16 seed-3 finding). Built lazily so the IPOPT cost is paid only
        # if a failure ever happens.
        self._fallback_enabled = bool(fallback_ipopt)
        self._fallback = None

    # ----------------------------------------------------------------- solve
    def solve(self, x, w_hat, xref_traj):
        """x (12,), w_hat (6,) NED body, xref_traj (12, N+1) -> u (4,)."""
        N = self.N
        x = np.asarray(x, float)
        w_hat = np.clip(np.asarray(w_hat, float), -50.0, 50.0)
        xref = np.asarray(xref_traj, float)
        s = self.solver

        s.set(0, "lbx", x)
        s.set(0, "ubx", x)
        for k in range(N):
            s.set(k, "p", w_hat)
            s.set(k, "yref", np.concatenate([xref[:, k], np.zeros(NU)]))
        s.set(N, "p", w_hat)
        s.set(N, "yref", xref[:, N])

        if not self._warm:                      # cold init: flat trajectory at x
            for k in range(N + 1):
                s.set(k, "x", x)
            for k in range(N):
                s.set(k, "u", np.zeros(NU))
            self._warm = True

        status = self.solver.solve()
        self.last_status = int(status)
        u = np.asarray(self.solver.get(0, "u"), float)
        if status in (0, 2) and np.isfinite(u).all():   # 0 ok; 2 = max_iter (usable)
            self._u_prev = u.copy()
            return u.copy()

        # ---- acados failed (NaN / min-step): recover, don't hold a stale u (->diverge)
        self.n_fail += 1
        self._warm = False                              # re-init the acados iterate next tick
        if self._fallback_enabled:
            uf = self._ipopt_fallback(x, w_hat, xref)
            if uf is not None and np.isfinite(uf).all():
                self.n_fallback += 1
                self._u_prev = np.asarray(uf, float).copy()
                return self._u_prev.copy()
        return self._u_prev.copy()

    def _ipopt_fallback(self, x, w_hat, xref):
        """One IPOPT (full-convergence) solve to recover from an acados failure.
        Built lazily on first use; returns None if even IPOPT can't solve."""
        try:
            if self._fallback is None:
                from .mpc import NMPC
                self._fallback = NMPC(N=self.N, dt=self.dt)
                self._fallback.reset()
            return self._fallback.solve(x, w_hat, xref)
        except Exception:
            return None

    def solve_ms(self):
        """Last solve wall time [ms] as reported by acados."""
        try:
            return 1.0e3 * float(self.solver.get_stats("time_tot"))
        except Exception:
            return float("nan")

    def reset(self):
        self._u_prev = np.zeros(NU)
        self._warm = False
        self.n_fail = 0
        self.n_fallback = 0
        if self._fallback is not None:
            self._fallback.reset()
