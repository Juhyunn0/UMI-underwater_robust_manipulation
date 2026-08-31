#!/usr/bin/env python3
"""mpcc_bridge.py — the contouring controller behind the station's controller API.

Same surface ``MpcWorker`` already drives for ``HwDobMpc`` and ``HwPid`` (step /
set_target_ned / set_square_ned / set_line_ned / set_circle_ned /
ref_ned_at / note_applied /
reset / meta / solver_kind / realtime_ok / scenario), so switching the mode
combo to ``mpcc`` needs no special cases in the worker.

What is genuinely different, and why it is worth the extra module:

* **theta is a controller state, not a worker clock.** The tracking bridge is
  handed a time by the worker and asks "where should I be at t". This one owns
  its progress along the path and the SOLVER decides how fast to advance it,
  subject to the curvature-feasible speed profile. There is no governed clock,
  no wall clock, and no corner gate — the three heuristics that were
  approximating this by hand.
* **The horizon sees past corners.** That is the whole point (2026-08-16
  operator request): the reference is a curve, not a sequence of hidden legs.

DP hold (``set_target_ned``, used for the approach, the settle and station
mode) is NOT a contouring problem — there is no path to contour. It is served
by a degenerate one-point path so the same solver covers both, which keeps the
engaged/disengaged control path single.
"""

from __future__ import annotations

import math
import time

import numpy as np

from .path_geometry import ArcPath, _Line, path_from_scenario


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class HwMpcc:
    """MPCC in the NED datum frame the rest of the station speaks."""

    # the worker asks these before building a shared plan; MPCC needs neither
    # (it consumes the path itself), so it reports a one-stage plan and
    # ignores set_path_plan_ned.
    path_plan_steps = 1
    path_plan_dt = 0.05

    #: NO ``follow`` missions on this controller, and the reason is in
    #: :meth:`set_target_ned` three screens down: it opens with
    #: ``del v_ned, r_ned`` — the velocity feedforward is DISCARDED — and then
    #: rebuilds an ArcPath, a speed profile and a window spline from scratch,
    #: with ``scenario = None``. Handing that a setpoint that moves at 20 Hz
    #: means re-deriving the whole geometry every tick with no feedforward to
    #: show for it, which is precisely the case the measured leash limit
    #: (0.084 m/s, memory: approach-speed-leash-limited) makes unflyable. The
    #: fix is not a flag here; it is contouring against a moving path, which
    #: this solver does not do. So `follow` refuses in mpcc/dobmpcc.
    follow_ok = False

    def __init__(self, cfg, mode: str = "mpcc", log=print):
        from .mpcc_acados import AcadosMPCC

        self.cfg = cfg
        self.mode = str(mode)
        self.solver_kind = "acados-mpcc"
        self.scenario: dict | None = None
        self.w_hat = np.zeros(6)
        self.n_fail = 0
        self.last_status = 0
        self.solve_ms = 0.0
        self.eaob = None
        self._tau_ned_cmd = np.zeros(6)
        self.w_clip = np.asarray(cfg.w_hat_clip, float)

        t0 = time.perf_counter()
        self.mpcc = AcadosMPCC()
        self.build_s = time.perf_counter() - t0
        self.path: ArcPath | None = None
        self._s_grid = np.zeros(2)
        self._v_grid = np.zeros(2)
        self._window = 1.0
        self._theta = 0.0
        self._depth_ref = 0.0
        self._yaw_fixed = 0.0
        self._heading_follow = False
        self._hold = None                    # (p_ned, yaw) while DP-holding
        self._v_cmd = 0.0                    # constant progress ceiling

        # A probe solve at the mission-free hold path, so ``realtime_ok`` is
        # answered by this build on this machine rather than by hope.
        self.set_target_ned((0.0, 0.0, 0.0), 0.0)
        x = np.zeros(13)
        t0 = time.perf_counter()
        try:
            self._solve(x)
            self.probe_ms = 1e3 * (time.perf_counter() - t0)
        except Exception as e:                                # noqa: BLE001
            log(f"mpcc probe solve failed: {type(e).__name__}: {e}")
            self.probe_ms = None
        self.mpcc.reset()
        lim = float(cfg.engage.get("probe_ms_max", 25.0))
        self.realtime_ok = (self.probe_ms is not None
                            and self.probe_ms < lim)
        log(f"mpcc: {self.solver_kind}, {self.mpcc.N} stages / "
            f"{self.mpcc.horizon_s:.1f} s horizon, probe "
            f"{self.probe_ms and round(self.probe_ms, 1)} ms, "
            f"build {self.build_s:.1f}s, realtime_ok={self.realtime_ok}")

    # ------------------------------------------------------------- reference
    def _install(self, path: ArcPath, v_cmd: float, depth_ned: float,
                 yaw_fixed_ned: float, heading_follow: bool) -> None:
        self.path = path
        self._s_grid, self._v_grid = path.speed_profile(
            v_cmd,
            float(getattr(self.cfg, "path_lat_accel_m_s2", 0.05)),
            float(getattr(self.cfg, "path_long_accel_m_s2", 0.05)),
            v_creep=float(getattr(self.cfg, "path_creep_m_s", 0.02)))
        # THE WINDOW IS SIZED ON WHAT THE PATH PERMITS, not on what was asked
        # for. window_for answers "how much arclength can the horizon
        # traverse", and the honest input to that is the speed the curve
        # actually allows — v_grid already folds in the operator's number AND
        # the lateral-acceleration limit. On a polygon the two are identical
        # (the straights run at v_cmd), so no mission flown so far changes.
        #
        # A CIRCLE is where they diverge, because its curvature limit binds
        # over the WHOLE lap: ask for 0.50 m/s on a 0.15 m circle and the
        # vehicle can do 0.087, but a window sized for 0.50 spans 4.9 m — over
        # five laps of a 0.94 m loop — and the B-spline through 41 samples of
        # that reproduced the arc to only 3.2 mm, against 0.1 mm once the
        # window is sized on the feasible speed
        # (test_mpcc.test_the_window_spline_reproduces_a_circle_at_every_offered_radius).
        self._window = self.mpcc.window_for(float(np.max(self._v_grid)))
        self._v_cmd = float(v_cmd)
        self._theta = 0.0
        self._depth_ref = float(depth_ned)
        self._yaw_fixed = float(yaw_fixed_ned)
        self._heading_follow = bool(heading_follow)

    def set_target_ned(self, p_ned, yaw_ned, v_ned=None, r_ned=0.0) -> None:
        """DP hold: a one-point path the vehicle can only sit on.

        Built as a short straight through the hold point so the geometry stays
        well-formed (a zero-length primitive has no tangent), with a speed
        profile of zero — theta cannot advance, so the reference never leaves
        the point."""
        del v_ned, r_ned
        p = np.asarray(p_ned, float)
        yaw = float(yaw_ned)
        self._hold = (p.copy(), yaw)
        seg = _Line(float(p[0]), float(p[1]), yaw, 1.0)
        self.path = ArcPath([seg], laps=1, closed=False)
        self._s_grid = np.array([0.0, 1.0])
        self._v_grid = np.array([0.0, 0.0])
        self._window = self.mpcc.window_for(0.0)
        self._v_cmd = 0.0                    # a hold makes no progress
        self._theta = 0.0
        self._depth_ref = float(p[2])
        self._yaw_fixed = yaw
        self._heading_follow = False
        self.scenario = None

    def set_square_ned(self, square: dict, origin_ned_xy, yaw_fixed_ned: float,
                       depth_ned: float) -> dict:
        from .reference import place_square_ned

        _fn, self.scenario = place_square_ned(
            square, origin_ned_xy, yaw_fixed_ned, depth_ned,
            dt=self.mpcc.dt)
        self._arm(self.scenario, yaw_fixed_ned, depth_ned)
        return self.scenario

    def set_line_ned(self, line: dict, origin_ned_xy, yaw_fixed_ned: float,
                     depth_ned: float) -> dict:
        from .reference import place_line_ned

        _fn, self.scenario = place_line_ned(
            line, origin_ned_xy, yaw_fixed_ned, depth_ned, dt=self.mpcc.dt)
        self._arm(self.scenario, yaw_fixed_ned, depth_ned)
        return self.scenario

    def set_circle_ned(self, circle: dict, origin_ned_xy, yaw_fixed_ned: float,
                       depth_ned: float) -> dict:
        from .reference import place_circle_ned

        _fn, self.scenario = place_circle_ned(
            circle, origin_ned_xy, yaw_fixed_ned, depth_ned, dt=self.mpcc.dt)
        self._arm(self.scenario, yaw_fixed_ned, depth_ned)
        return self.scenario

    def _arm(self, scen: dict, yaw_fixed_ned: float, depth_ned: float) -> None:
        fillet = float(getattr(self.cfg, "path_fillet_m", 0.15))
        turn = float(getattr(self.cfg, "path_turn_radius_m", 0.0))
        path = path_from_scenario(scen, fillet_m=fillet, turn_radius_m=turn)
        self._install(path, float(scen.get("speed", 0.05)), depth_ned,
                      yaw_fixed_ned, bool(scen.get("heading_follow", False)))
        self._hold = None
        # The resolved CURVE, recorded beside the mission it came from — the
        # rectangle in the scenario is no longer what is flown, and a run whose
        # meta claims sharp corners could not be replayed.
        scen["path"] = {"kind": "mpcc-arc", "fillet_m": fillet,
                        "turn_radius_m": turn,
                        "lap_length_m": float(path.lap_length),
                        "total_length_m": float(path.total_length),
                        "v_cmd_m_s": float(scen.get("speed", 0.05)),
                        "v_min_m_s": float(self._v_grid.min()),
                        "v_max_m_s": float(self._v_grid.max())}

    def set_path_plan_ned(self, plan) -> None:
        """Ignored: MPCC owns its own reference (see the module docstring)."""

    def ref_ned_at(self, t: float):
        """(p_ned, yaw_ned, v_ned) of the CURRENT reference point.

        ``t`` is accepted for interface parity and NOT used — progress is
        ``theta``, a controller state. Feeding a time in here would be the very
        thing MPCC exists to stop doing."""
        del t
        if self.path is None:
            return np.zeros(3), 0.0, np.zeros(3)
        px, py, psi, _k = self.path.sample(np.array([self._theta]))
        v = float(np.interp(self._theta, self._s_grid, self._v_grid))
        yaw = (float(psi[0]) if self._heading_follow else self._yaw_fixed)
        return (np.array([px[0], py[0], self._depth_ref]), yaw,
                np.array([v * math.cos(psi[0]), v * math.sin(psi[0]), 0.0]))

    # ---------------------------------------------------------------- errors
    def path_errors(self, p_ned):
        """(along=LAG, cross) in the path frame — the same conventions the CSV
        and the plot already use (workers._path_split)."""
        if self.path is None:
            return None, None
        px, py, psi, _k = self.path.sample(np.array([self._theta]))
        dx = float(p_ned[0]) - px[0]
        dy = float(p_ned[1]) - py[0]
        c, s = math.cos(psi[0]), math.sin(psi[0])
        return -(c * dx + s * dy), (s * dx - c * dy)

    @property
    def progress_m(self) -> float:
        return float(self._theta)

    @property
    def complete(self) -> bool:
        return (self.path is not None and self.scenario is not None
                and self._theta >= self.path.total_length - 1e-3)

    def lap(self) -> int:
        if self.path is None or self.path.lap_length <= 0:
            return 0
        return int(self._theta // self.path.lap_length)

    # ------------------------------------------------------------------ tick
    def _solve(self, x13):
        params = self.mpcc.pack_params(
            self.w_hat, self._theta, self.path, self._yaw_fixed,
            self._heading_follow, self._s_grid, self._v_grid,
            self._window, float(x13[5]))
        return self.mpcc.solve(x13, params, self._depth_ref, self._v_cmd)

    def step(self, eta_ned, nu_ned, nudot_ned, t: float):
        """One 20 Hz tick -> (wrench (6,), info). ``t`` unused (see ref_ned_at)."""
        del t
        eta = np.asarray(eta_ned, float)
        nu = np.asarray(nu_ned, float)
        if self.mode == "dobmpcc":
            from .mpc_bridge import HwDobMpc

            if self.eaob is None:
                self.eaob = HwDobMpc._EAOB(eta0=eta, nu0=nu, profile="perf")
            _e, _n, w = self.eaob.update(
                {"eta": eta, "nu": nu, "nudot": np.asarray(nudot_ned, float)},
                self._tau_ned_cmd)
            self.w_hat = np.clip(w, -self.w_clip, self.w_clip)
        else:
            self.w_hat = np.zeros(6)
        x = np.concatenate([eta, nu, [self._theta]])
        t0 = time.perf_counter()
        u = self._solve(x)
        wall = 1e3 * (time.perf_counter() - t0)
        acms = self.mpcc.solve_ms()
        self.solve_ms = float(acms) if np.isfinite(acms) else wall
        self.n_fail = int(self.mpcc.n_fail)
        self.last_status = int(self.mpcc.last_status)
        # Advance progress by what the solver asked for. Clamped to the path so
        # a finished mission cannot walk the reference off the end.
        if self.path is not None:
            self._theta = float(min(self.path.total_length,
                                    self._theta + float(u[6]) * self.mpcc.dt))
        tau = np.asarray(u[:6], float)
        self._tau_ned_cmd = tau.copy()
        return tau.copy(), {
            "w_hat": self.w_hat.copy(), "solve_ms": self.solve_ms,
            "status": self.last_status, "n_fail": self.n_fail,
            "nis": (float(self.eaob.last_nis) if self.eaob is not None
                    else 0.0),
            "v_theta": float(u[6]), "theta": self._theta,
        }

    def note_applied(self, tau_ned_est) -> None:
        self._tau_ned_cmd = np.asarray(tau_ned_est, float).copy()

    def reset(self) -> None:
        self.eaob = None
        self.w_hat = np.zeros(6)
        self._tau_ned_cmd = np.zeros(6)
        self._theta = 0.0
        self.mpcc.reset()

    # ------------------------------------------------------------------ meta
    def meta(self) -> dict:
        from . import mpcc_acados as M

        return {
            "type": self.mode, "solver": self.solver_kind,
            "formulation": "MPCC (contouring): x=[eta,nu,theta], "
                           "u=[tau,v_theta]; cost = contour + lag + "
                           "progress-profile tracking",
            "N": int(self.mpcc.N), "horizon_s": float(self.mpcc.horizon_s),
            "shooting_grid": self.mpcc.time_steps.tolist(),
            "dt_s": float(self.mpcc.dt),
            "weights": {"contour": M.W_CONTOUR, "lag": M.W_LAG,
                        "depth": M.W_Z, "yaw": M.W_PSI,
                        "roll": M.W_ROLL, "pitch": M.W_PITCH,
                        "rates": M.W_RATE, "progress": M.W_PROGRESS},
            "R": list(np.asarray(__import__(
                "dobmpc.params", fromlist=["MPC_R"]).MPC_R, float)),
            "v_theta_max": M.V_THETA_MAX,
            "path_window_m": float(self._window),
            "path_grid_points": M.N_GRID,
            "fillet_m": float(getattr(self.cfg, "path_fillet_m", 0.15)),
            "turn_radius_m": float(getattr(self.cfg, "path_turn_radius_m",
                                           0.0)),
            "lat_accel_m_s2": float(getattr(self.cfg,
                                            "path_lat_accel_m_s2", 0.05)),
            "long_accel_m_s2": float(getattr(self.cfg,
                                             "path_long_accel_m_s2", 0.05)),
            "w_hat_clip": list(map(float, self.w_clip)),
            "ref_preview": True,
        }
