"""Disturbance-parameterised NMPC (paper Sec. 4, Eq. 44).

CasADi implementation of the OCP solved by acados in the reference code.
The prediction model is the full Fossen model (Eq. 22) with the disturbance
wrench w as a *parameter*, updated from the EAOB estimate at every control
step and held constant over the horizon (Assumption 2):

    min  sum ||x_k - xref_k||^2_Q + ||u_k||^2_R  +  ||x_N - xref_N||^2_QN
    s.t. x_{k+1} = RK4(f(x_k, u_k, w_hat)),  |u| <= u_max,  |v_lin| <= v_max

Baseline MPC = the same controller with w_hat = 0.

The NLP is built once with SX expressions (multiple shooting, bounds as
simple variable bounds) and solved warm-started by Ipopt
(~0.1 s per step at N = 60 - fast enough for simulation studies).  A
CasADi sqpmethod/QRQP path is kept for experimentation but is not robust
on this problem; for hard real-time use, port the identical model and cost
to acados SQP-RTI as in the paper's reference implementation.
"""
import casadi as ca
import numpy as np

from . import fossen
from . import params as P

NX = 12
NU = P.NU                    # 6 for heavy 


def make_nmpc(N=P.MPC_N, dt=P.DT_CTRL, solver=None):
    """Factory: return the requested NMPC backend with a uniform .solve(x, w_hat,
    xref) interface. `solver` defaults to params.SOLVER. "acados" yields the
    SQP-RTI/HPIPM AcadosNMPC (fast path); anything else (or an acados import/build
    failure) yields the IPOPT NMPC below, which is the reference & fallback."""
    solver = (solver or getattr(P, "SOLVER", "ipopt")).lower()
    if solver == "acados":
        try:
            from .mpc_acados import AcadosNMPC
            return AcadosNMPC(N=N, dt=dt)
        except Exception as e:                # missing build, codegen error, ...
            import warnings
            warnings.warn(f"acados NMPC unavailable ({type(e).__name__}: {e}); "
                          f"falling back to IPOPT NMPC", RuntimeWarning)
    return NMPC(N=N, dt=dt)


def _f_casadi(x, u, w):
    """Symbolic Eq. 22; mirrors fossen.f_state (validated in
    scripts/validate_plant.py)."""

    # state decomposition by JJ
    nu = x[6:]
    phi, theta, psi = x[3], x[4], x[5]
    uu, vv, ww, p, q, r = nu[0], nu[1], nu[2], nu[3], nu[4], nu[5]
    sph, cph = ca.sin(phi), ca.cos(phi)
    sth, cth = ca.sin(theta), ca.cos(theta)
    sps, cps = ca.sin(psi), ca.cos(psi)

    # kinematics (Eq. 2, 4, 5)
    Rib = ca.vertcat(
        ca.horzcat(cps * cth, -sps * cph + cps * sth * sph,
                   sps * sph + cps * cph * sth),
        ca.horzcat(sps * cth, cps * cph + sph * sth * sps,
                   -cps * sph + sth * sps * cph),
        ca.horzcat(-sth, cth * sph, cth * cph))
    T = ca.vertcat(
        ca.horzcat(1, sph * sth / cth, cph * sth / cth),
        ca.horzcat(0, cph, -sph),
        ca.horzcat(0, sph / cth, cph / cth))
    eta_dot = ca.vertcat(Rib @ nu[:3], T @ nu[3:])

    # kinetics (Eq. 7-17)
    m, Ix, Iy, Iz = P.MASS, P.IX, P.IY, P.IZ
    a = P.ADDED_MASS
    crb = ca.vertcat(m * (q * ww - r * vv),
                     m * (r * uu - p * ww),
                     m * (p * vv - q * uu),
                     (Iz - Iy) * q * r,
                     (Ix - Iz) * p * r,
                     (Iy - Ix) * p * q)
    cad = ca.vertcat(a[2] * ww * q - a[1] * vv * r,
                     -a[2] * ww * p + a[0] * uu * r,
                     a[1] * vv * p - a[0] * uu * q,
                     a[2] * ww * vv - a[1] * vv * ww + a[5] * r * q - a[4] * q * r,
                     -a[2] * ww * uu + a[0] * uu * ww - a[5] * r * p + a[3] * p * r,
                     a[1] * vv * uu - a[0] * uu * vv + a[4] * q * p - a[3] * p * q)
    damp = -(ca.DM(P.DL) * nu + ca.DM(P.DNL) * ca.fabs(nu) * nu)
    W, B, zg = P.WEIGHT, P.BUOYANCY, P.ZG
    g_eta = ca.vertcat((W - B) * sth,
                       -(W - B) * cth * sph,
                       -(W - B) * cth * cph,
                       zg * W * cth * sph,
                       zg * W * sth,
                       0.0)
    # control u -> body wrench tau: every loadable variant is the fully actuated
    # heavy family (NU=6, rank-6 allocation), so the full wrench is realizable
    # directly. (The rank-5 bluerov2 option-(b) surge->pitch mapping is gone with
    # the variant; see KNOWN_ISSUES "dobmpc NU=4" for the remaining deferred cleanup.)
    tau = u
    nu_dot = ca.DM(fossen.M_INV) @ (tau + w - crb - cad - damp - g_eta)
    return ca.vertcat(eta_dot, nu_dot)


def _rk4(x, u, w, dt, n_int=2):
    """RK4 with n_int substeps.  The roll/pitch dynamics are stiff
    (|lambda| ~ 70-85 1/s from D_L/I); a single 50 ms RK4 step lies outside
    the stability region (|lambda dt| > 2.78) and the prediction explodes,
    so each shooting interval is integrated in substeps.  n_int = 2
    (h = 25 ms, |lambda h| ~ 2.1) is stable with margin and ~2x faster than
    n_int = 4 (12.5 ms, the paper's effective acados step); raise it if you
    push the vehicle to large roll/pitch rates."""
    h = dt / n_int
    for _ in range(n_int):
        k1 = _f_casadi(x, u, w)
        k2 = _f_casadi(x + h / 2 * k1, u, w)
        k3 = _f_casadi(x + h / 2 * k2, u, w)
        k4 = _f_casadi(x + h * k3, u, w)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


class NMPC:
    def __init__(self, N=P.MPC_N, dt=P.DT_CTRL,
                 Q=P.MPC_Q, R=P.MPC_R, QN=P.MPC_QN,
                 u_max=P.U_MAX, v_max=P.V_MAX,
                 solver="ipopt", max_iter=15, n_int=2):
        self.N, self.dt = N, dt
        X = ca.SX.sym("X", NX, N + 1)
        U = ca.SX.sym("U", NU, N)
        Pp = ca.SX.sym("P", NX + 6 + NX * (N + 1))
        x0, what = Pp[:NX], Pp[NX:NX + 6]
        Xref = ca.reshape(Pp[NX + 6:], NX, N + 1)

        Qd, Rd, QNd = map(ca.DM, (np.diag(Q), np.diag(R), np.diag(QN)))
        cost, g = 0, [X[:, 0] - x0]
        for k in range(N):
            e = X[:, k] - Xref[:, k]
            cost += e.T @ Qd @ e + U[:, k].T @ Rd @ U[:, k]
            g.append(X[:, k + 1] - _rk4(X[:, k], U[:, k], what, dt,
                                        n_int=n_int))
        eN = X[:, N] - Xref[:, N]
        cost += eN.T @ QNd @ eN

        z = ca.vertcat(ca.vec(X), ca.vec(U))
        nlp = {"x": z, "p": Pp, "f": cost, "g": ca.vertcat(*g)}
        if solver == "ipopt":
            opts = {"print_time": False,
                    "ipopt": {"print_level": 0, "sb": "yes",
                              "max_iter": 50, "tol": 1e-5,
                              "warm_start_init_point": "yes",
                              "mu_init": 1e-3}}
            self.solver = ca.nlpsol("mpc", "ipopt", nlp, opts)
        else:  # SQP with active-set QP - the RTI-style work-horse
            opts = {"print_time": False, "print_header": False,
                    "print_iteration": False, "print_status": False,
                    "error_on_fail": False, "max_iter": max_iter,
                    "tol_du": 1e-4, "tol_pr": 1e-6,
                    # exact-Hessian SQP needs convexification, otherwise the
                    # QP turns indefinite away from the reference (this is
                    # what acados' Gauss-Newton Hessian provides for free)
                    "convexify_strategy": "eigen-reflect",
                    "qpsol": "qrqp",
                    "qpsol_options": {"print_iter": False,
                                      "print_header": False,
                                      "print_info": False,
                                      "error_on_fail": False}}
            self.solver = ca.nlpsol("mpc", "sqpmethod", nlp, opts)

        # variable bounds: velocities via state bounds, controls via u bounds
        lbx = np.full(NX * (N + 1) + NU * N, -np.inf)
        ubx = np.full_like(lbx, np.inf)
        for k in range(1, N + 1):                       # skip x0 (pinned)
            lbx[k * NX + 6:k * NX + 9] = -v_max
            ubx[k * NX + 6:k * NX + 9] = v_max
            lbx[k * NX + 3] = -1.2                       # |phi| (roll): T(eta) singularity
            ubx[k * NX + 3] = 1.2
            lbx[k * NX + 4] = -1.2                       # |theta| (pitch): T(eta) singularity
            ubx[k * NX + 4] = 1.2
        off = NX * (N + 1)
        for k in range(N):
            lbx[off + k * NU:off + (k + 1) * NU] = -np.asarray(u_max)
            ubx[off + k * NU:off + (k + 1) * NU] = np.asarray(u_max)
        self._lbx, self._ubx = lbx, ubx
        self._z0 = None
        self._lam = None
        self._u_prev = np.zeros(NU)
        self.n_fail = 0

    def solve(self, x, w_hat, xref_traj):
        """x (12,), w_hat (6,) body frame, xref_traj (12, N+1) -> u (6,)."""
        N = self.N
        x = np.asarray(x, float)
        w_hat = np.clip(np.asarray(w_hat, float), -50.0, 50.0)
        p = np.concatenate([x, w_hat,
                            np.asarray(xref_traj).flatten(order="F")])
        if self._z0 is None:
            self._z0 = np.concatenate([np.tile(x, N + 1), np.zeros(NU * N)])

        z, ok = self._try(p)
        if not ok:                       # cold restart once, then fall back
            self._z0 = np.concatenate([np.tile(x, N + 1),
                                       np.tile(self._u_prev, N)])
            self._lam = None
            z, ok = self._try(p)
        if not ok:
            self.n_fail += 1
            self.reset()
            return self._u_prev.copy()

        Xs = z[:NX * (N + 1)].reshape(NX, N + 1, order="F")
        Us = z[NX * (N + 1):].reshape(NU, N, order="F")
        # shift for warm start
        Xn = np.hstack([Xs[:, 1:], Xs[:, -1:]])
        Un = np.hstack([Us[:, 1:], Us[:, -1:]])
        self._z0 = np.concatenate([Xn.flatten(order="F"),
                                   Un.flatten(order="F")])
        self._u_prev = Us[:, 0].copy()
        return Us[:, 0].copy()

    def _try(self, p):
        kw = {"lam_g0": self._lam} if self._lam is not None else {}
        sol = self.solver(x0=self._z0, p=p, lbx=self._lbx, ubx=self._ubx,
                          lbg=0, ubg=0, **kw)
        z = np.array(sol["x"]).ravel()
        try:
            ok = bool(self.solver.stats()["success"])
        except RuntimeError:             # stats can be unavailable on failure
            ok = False
        ok = ok and np.isfinite(z).all()
        if ok:
            self._lam = np.array(sol["lam_g"]).ravel()
        return z, ok

    def reset(self):
        self._z0, self._lam = None, None
        self._u_prev = np.zeros(NU)
