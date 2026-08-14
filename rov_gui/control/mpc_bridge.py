#!/usr/bin/env python3
"""mpc_bridge.py — the sim's EAOB + acados NMPC, wrapped for the real vehicle.

``bluerov2_mujoco_marinegym/dobmpc/`` is verified MuJoCo-free and is imported
AS-IS (params/fossen/frames/eaob/mpc/mpc_acados). What the sim's
``DOBMPCController`` wrapper did against MuJoCo, ``HwDobMpc`` does against
the state assembler: the reference-generation methods (``set_target``,
``set_reference_traj``, ``_xref_ned``, ``_xref_ned_traj``) are ported
line-for-line from ``dobmpc_controller.py:126-285`` (2026-08-12) with their
docstrings compressed — the originals carry the full reasoning.

Differences from the sim wrapper, each deliberate:
  * x0 is ALWAYS the measured/assembled state — the sim's "meas" semantic is
    the hardware reality, so the state-source switch is gone;
  * ``tau_applied`` fed to the EAOB is the wrench the ALLOCATION believes it
    realized (axis caps applied, K/M dropped — see allocation.py), reported
    back via :meth:`note_applied`, never the raw solver output; the axis-cap
    and dropped-moment mismatch would otherwise be double-counted as
    disturbance;
  * ``w_hat`` is clipped PER-AXIS to ``hw_mpc.yaml: w_hat_clip`` (default
    [15,45,45,5,5,8]) before the solver — the sim's scalar ±50 clip
    (mpc_acados.py:171) is 5-6x the real torque authority (KNOWN_ISSUES) and
    stays only as a backstop;
  * ``solve_ms``/``last_status``/``n_fail`` are read from the live acados
    solver every tick (the sim wrapper's ``solve_ms`` was a dead field).

Import discipline: importing THIS module is cheap; ``dobmpc``/casadi/acados
load inside :func:`import_dobmpc` only, and ``ROV_MODEL`` must be decided
before that (rov_model.py reads it at import time).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_MARINEGYM = Path(__file__).resolve().parents[2] / "bluerov2_mujoco_marinegym"

_dob = None          # dict of the imported dobmpc modules, once
_dob_model = None    # the ROV_MODEL it was imported under


def import_dobmpc(rov_model: str):
    """Import the sim's controller stack exactly once, pinned to one model.

    ``dobmpc/params.py`` does a plain ``import rov_model`` and rov_model.py
    reads ``$ROV_MODEL`` at import time, so both the path insert and the env
    var must be in place BEFORE the first import — and a second import under
    a different model would silently keep the first one's physics, hence the
    hard error instead."""
    global _dob, _dob_model
    if _dob is not None:
        if _dob_model != rov_model:
            raise RuntimeError(
                f"dobmpc already imported for ROV_MODEL={_dob_model!r}; "
                f"cannot re-import for {rov_model!r} in this process")
        return _dob
    if not _MARINEGYM.is_dir():
        raise FileNotFoundError(f"marinegym tree not found at {_MARINEGYM}")
    if str(_MARINEGYM) not in sys.path:
        sys.path.insert(0, str(_MARINEGYM))
    os.environ["ROV_MODEL"] = str(rov_model)
    from dobmpc import frames, params
    from dobmpc.eaob import EAOB
    from dobmpc.fossen import wrap_angle
    from dobmpc.mpc import make_nmpc
    _dob = {"frames": frames, "P": params, "EAOB": EAOB,
            "wrap_angle": wrap_angle, "make_nmpc": make_nmpc}
    _dob_model = rov_model
    return _dob


def apply_sigma_overrides(P, sig: dict) -> dict:
    """Overwrite params.EAOB_SIG_* with the hardware values BEFORE any EAOB is
    built (eaob._perf_covariances reads the module at construction time).
    Returns what was applied, for the run meta."""
    applied = {}
    if not sig:
        return applied
    if sig.get("pos") is not None:
        P.EAOB_SIG_POS = np.asarray(sig["pos"], float)
        applied["pos"] = list(P.EAOB_SIG_POS)
    if sig.get("ang_deg") is not None:
        P.EAOB_SIG_ANG = np.deg2rad(np.asarray(sig["ang_deg"], float))
        applied["ang_deg"] = list(np.rad2deg(P.EAOB_SIG_ANG))
    if sig.get("lvel") is not None:
        P.EAOB_SIG_LVEL = np.asarray(sig["lvel"], float)
        applied["lvel"] = list(P.EAOB_SIG_LVEL)
    if sig.get("avel") is not None:
        P.EAOB_SIG_AVEL = np.asarray(sig["avel"], float)
        applied["avel"] = list(P.EAOB_SIG_AVEL)
    if sig.get("acc") is not None:
        P.EAOB_SIG_ACC = float(sig["acc"])
        applied["acc"] = P.EAOB_SIG_ACC
    if sig.get("aacc") is not None:
        P.EAOB_SIG_AACC = float(sig["aacc"])
        applied["aacc"] = P.EAOB_SIG_AACC
    if sig.get("alloc") is not None:
        P.EAOB_SIG_ALLOC = np.asarray(sig["alloc"], float)
        applied["alloc"] = list(P.EAOB_SIG_ALLOC)
    return applied


def _guard_stale_solver(P) -> None:
    """The acados codegen cache is keyed only on (model, rti): N/dt/U_MAX
    changes load a STALE .so under DOBMPC_ACADOS_BUILD=0. A meta sidecar makes
    that combination loud: on mismatch the fast-load flag is dropped so the
    solver regenerates this once."""
    variant = f"{P.MODEL}_rti"
    gen = _MARINEGYM / "dobmpc" / "_acados_gen" / variant
    meta_p = gen / "build_meta_rovgui.json"
    want = {"model": P.MODEL, "N": int(P.MPC_N), "dt": float(P.DT_CTRL),
            "nu": int(P.NU), "u_max": [float(v) for v in P.U_MAX]}
    if os.environ.get("DOBMPC_ACADOS_BUILD") == "0":
        have = None
        if meta_p.exists():
            try:
                have = json.loads(meta_p.read_text())
            except (OSError, json.JSONDecodeError):
                have = None
        if have != want:
            os.environ.pop("DOBMPC_ACADOS_BUILD", None)   # force one rebuild
    try:
        gen.mkdir(parents=True, exist_ok=True)
        meta_p.write_text(json.dumps(want, indent=1))
    except OSError:
        pass


def _Rz_flu(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class HwDobMpc:
    """The hardware twin of the sim's DOBMPCController (see module docstring)."""

    def __init__(self, mode: str, mpc_cfg, log=print):
        assert mode in ("mpc", "dobmpc"), mode
        self.mode = mode
        self.cfg = mpc_cfg
        d = import_dobmpc(mpc_cfg.rov_model)
        self.frames = d["frames"]
        self.P = d["P"]
        self._EAOB = d["EAOB"]
        self.wrap_angle = d["wrap_angle"]
        P = self.P
        if not getattr(P, "FULLY_ACTUATED", False):
            raise RuntimeError(
                f"ROV_MODEL={P.MODEL!r} is rank-5 (NU=4); the hardware bridge "
                f"assumes the fully-actuated heavy wrench [X,Y,Z,K,M,N]")
        if abs(1.0 / float(mpc_cfg.ctrl_hz) - P.DT_CTRL) > 1e-9:
            raise RuntimeError(
                f"ctrl_hz {mpc_cfg.ctrl_hz} != 1/DT_CTRL ({1.0 / P.DT_CTRL}); "
                f"DT_CTRL is baked into the generated solver — keep them equal")
        self.sigma_applied = apply_sigma_overrides(P, mpc_cfg.eaob_sigmas)
        _guard_stale_solver(P)
        t0 = time.perf_counter()
        self.nmpc = d["make_nmpc"](N=P.MPC_N, dt=P.DT_CTRL)
        self.build_s = time.perf_counter() - t0
        self.solver_kind = ("acados" if type(self.nmpc).__name__ == "AcadosNMPC"
                            else "ipopt")
        self.w_clip = np.asarray(mpc_cfg.w_hat_clip, float)

        # references (world FLU — the sim wrapper's convention, kept verbatim)
        self.p_ref = np.zeros(3)
        self.yaw_ref = 0.0
        self.v_ref = np.zeros(3)
        self.r_ref = 0.0
        self.yaw_target = 0.0
        self._ref_traj = None
        self.scenario: dict | None = None

        self.eaob = None
        self.w_hat = np.zeros(6)
        self._tau_ned_cmd = np.zeros(6)     # what allocation says was realized
        self._psi_ned_now = 0.0
        self.n_fail = 0
        self.last_status = 0
        self.solve_ms = None
        self.probe_ms = None
        self.realtime_ok = False
        self._probe(log)

    # ------------------------------------------------------------------ probe
    def _probe(self, log) -> None:
        """One throwaway solve so 'which solver, how fast' is a fact, not a
        hope — make_nmpc downgrades to IPOPT silently on any acados problem."""
        x = np.zeros(12)
        xref = np.zeros((12, self.nmpc.N + 1))
        try:
            t0 = time.perf_counter()
            self.nmpc.solve(x, np.zeros(6), xref)
            self.probe_ms = 1e3 * (time.perf_counter() - t0)
            self.nmpc.reset()
        except Exception as e:                                   # noqa: BLE001
            log(f"mpc probe solve failed: {type(e).__name__}: {e}")
            self.probe_ms = None
        lim = float(self.cfg.engage.get("probe_ms_max", 25.0))
        self.realtime_ok = (self.solver_kind == "acados"
                            and self.probe_ms is not None
                            and self.probe_ms < lim)
        log(f"mpc solver: {self.solver_kind}, probe "
            f"{self.probe_ms if self.probe_ms is None else round(self.probe_ms, 1)} ms, "
            f"build {self.build_s:.1f}s, realtime_ok={self.realtime_ok}")

    # -------------------------------------------------- sim-ported reference API
    # PORTED from dobmpc_controller.py:126 (see module docstring)
    def set_target(self, p_ref=None, yaw_ref=None, v_ref=None, r_ref=None,
                   yaw_target=None):
        if p_ref is not None:
            self.p_ref = np.asarray(p_ref, float)
        if yaw_ref is not None:
            self.yaw_ref = float(yaw_ref)
        if v_ref is not None:
            self.v_ref = np.asarray(v_ref, float)
        if r_ref is not None:
            self.r_ref = float(r_ref)
        self.yaw_target = float(yaw_target) if yaw_target is not None else self.yaw_ref

    # PORTED from dobmpc_controller.py:141
    def set_reference_traj(self, fn):
        """fn(ts (K,)) -> p (3,K), yaw (K,), v (3,K), r (K,) — ALL WORLD FLU."""
        self._ref_traj = fn

    # PORTED from dobmpc_controller.py:183 (docstring compressed)
    def _xref_ned(self, t=None):
        if getattr(self, "_ref_traj", None) is not None and t is not None:
            return self._xref_ned_traj(float(t))
        frames, P, wrap_angle = self.frames, self.P, self.wrap_angle
        N = self.nmpc.N
        dt = P.DT_CTRL
        R_ref = _Rz_flu(self.yaw_ref)
        eta0 = frames.flu_to_ned_eta(self.p_ref, R_ref)
        eta0[5] = self._psi_ned_now + wrap_angle(eta0[5] - self._psi_ned_now)
        nu_ned = np.concatenate([frames.S @ (R_ref.T @ self.v_ref), np.zeros(3)])
        ks = np.arange(N + 1)
        pos_world = self.p_ref[:, None] + np.outer(self.v_ref, ks * dt)
        xref = np.zeros((12, N + 1))
        xref[0:3, :] = frames.S @ pos_world
        xref[3:6, :] = eta0[3:6][:, None]
        xref[6:12, :] = nu_ned[:, None]
        if self.r_ref != 0.0:
            r_ned = -self.r_ref
            psi0 = eta0[5]
            eta_t = frames.flu_to_ned_eta(self.p_ref, _Rz_flu(self.yaw_target))
            psi_t = psi0 + wrap_angle(eta_t[5] - psi0)
            delta = psi_t - psi0
            step = np.clip(r_ned * ks * dt, min(0.0, delta), max(0.0, delta))
            xref[5, :] = psi0 + step
            xref[11, :] = np.where(np.abs(step) < abs(delta) - 1e-12, r_ned, 0.0)
        return xref

    # PORTED from dobmpc_controller.py:243 (docstring compressed)
    def _xref_ned_traj(self, t0):
        frames, P, wrap_angle = self.frames, self.P, self.wrap_angle
        N = self.nmpc.N
        dt = P.DT_CTRL
        ts = t0 + np.arange(N + 1) * dt
        p_w, yaw_w, v_w, r_w = self._ref_traj(ts)
        p_w = np.asarray(p_w, float)
        v_w = np.asarray(v_w, float)
        yaw_w = np.asarray(yaw_w, float).ravel()
        r_w = np.asarray(r_w, float).ravel()
        assert p_w.shape == (3, N + 1) and v_w.shape == (3, N + 1), \
            f"sampler must return (3,{N + 1}) p/v, got {p_w.shape}/{v_w.shape}"
        assert yaw_w.size == N + 1 and r_w.size == N + 1, \
            f"sampler must return {N + 1} yaw/r samples, got {yaw_w.size}/{r_w.size}"
        xref = np.zeros((12, N + 1))
        xref[0:3, :] = frames.S @ p_w
        psi_prev = self._psi_ned_now
        for k in range(N + 1):
            Rk = _Rz_flu(yaw_w[k])
            eta_k = frames.flu_to_ned_eta(p_w[:, k], Rk)
            psi_prev = psi_prev + wrap_angle(eta_k[5] - psi_prev)
            xref[3:5, k] = eta_k[3:5]
            xref[5, k] = psi_prev
            xref[6:9, k] = frames.S @ (Rk.T @ v_w[:, k])
        xref[11, :] = -r_w
        return xref

    # ------------------------------------------------------- NED-facing helpers
    # The station's world is NED (the tag map); the ported methods above speak
    # the sim's world-FLU. The mirror is frames.S applied ONCE, here.
    def set_target_ned(self, p_ned, yaw_ned, v_ned=None, r_ned=0.0):
        S = self.frames.S
        self.set_target(p_ref=S @ np.asarray(p_ned, float),
                        yaw_ref=-float(yaw_ned),
                        v_ref=(np.zeros(3) if v_ned is None
                               else S @ np.asarray(v_ned, float)),
                        r_ref=-float(r_ned))
        self._ref_traj = None

    def set_square_ned(self, square: dict, origin_ned_xy, yaw_fixed_ned: float,
                       depth_ned: float):
        """Arm the tracking sampler with the square placed in NED. Returns the
        resolved scenario dict (recorded in meta / drawn by the UI)."""
        from .reference import make_square_ref_world

        size = float(square.get("size", 1.0))
        speed = float(square.get("speed", 0.12))
        laps = int(square.get("laps", 3))
        follow = bool(square.get("heading_follow", False))
        yaw_rate = float(np.radians(float(square.get("yaw_rate_deg_s", 60.0))))
        rot_ned = float(np.radians(float(square.get("rot_deg", 0.0))))
        T_run = laps * 4.0 * size / max(1e-6, speed)
        horizon_s = self.nmpc.N * self.P.DT_CTRL
        # NED -> sim-FLU mirror: y flips, angles negate, depth z_flu = -z_ned.
        fn = make_square_ref_world(
            size=size, speed=speed, depth=-float(depth_ned),
            heading_follow=follow, yaw_rate=yaw_rate, dt=self.P.DT_CTRL,
            T_total=T_run + horizon_s + 1.0,
            origin_xy=(float(origin_ned_xy[0]), -float(origin_ned_xy[1])),
            rot_rad=-rot_ned, yaw_fixed=-float(yaw_fixed_ned))
        self.set_reference_traj(fn)
        self.scenario = {"kind": "square", "size": size, "speed": speed,
                         "laps": laps, "depth_ned": float(depth_ned),
                         "heading_follow": follow,
                         "yaw_rate_deg_s": float(square.get("yaw_rate_deg_s", 60.0)),
                         "origin_ned": [float(origin_ned_xy[0]),
                                        float(origin_ned_xy[1])],
                         "rot_deg": float(square.get("rot_deg", 0.0)),
                         "yaw_fixed_ned_deg": float(np.degrees(yaw_fixed_ned)),
                         "T_run_s": T_run}
        return self.scenario

    def ref_ned_at(self, t: float) -> tuple[np.ndarray, float]:
        """(p_ned, yaw_ned) of the CURRENT reference — stage 0 of the horizon."""
        xref = self._xref_ned(t)
        return xref[0:3, 0].copy(), float(xref[5, 0])

    # ------------------------------------------------------------------ control
    def step(self, eta_ned, nu_ned, nudot_ned, t: float):
        """One 20 Hz tick: EAOB (dobmpc) -> per-axis w_hat clip -> NMPC.
        Returns (u_ned (6,), info dict). ``t`` is the trajectory clock."""
        eta = np.asarray(eta_ned, float)
        nu = np.asarray(nu_ned, float)
        if self.eaob is None:
            self.eaob = self._EAOB(eta0=eta, nu0=nu, profile="perf")
        if self.mode == "dobmpc":
            _eh, _nh, w = self.eaob.update(
                {"eta": eta, "nu": nu, "nudot": np.asarray(nudot_ned, float)},
                self._tau_ned_cmd)
            self.w_hat = np.clip(w, -self.w_clip, self.w_clip)
        else:
            self.w_hat = np.zeros(6)
        self._psi_ned_now = float(eta[5])
        t0 = time.perf_counter()
        u = self.nmpc.solve(np.concatenate([eta, nu]), self.w_hat,
                            self._xref_ned(t))
        wall_ms = 1e3 * (time.perf_counter() - t0)
        self.n_fail = int(self.nmpc.n_fail)
        self.last_status = int(getattr(self.nmpc, "last_status", 0))
        acms = self.nmpc.solve_ms() if hasattr(self.nmpc, "solve_ms") else None
        self.solve_ms = (float(acms) if acms is not None
                         and np.isfinite(acms) else wall_ms)
        # Until note_applied() reports the allocation's estimate, assume the
        # full wrench went out (correct for the sim, conservative here).
        self._tau_ned_cmd = np.asarray(u, float).copy()
        return np.asarray(u, float).copy(), {
            "w_hat": self.w_hat.copy(), "solve_ms": self.solve_ms,
            "status": self.last_status, "n_fail": self.n_fail,
            "nis": (float(self.eaob.last_nis) if self.eaob is not None else 0.0),
        }

    def note_applied(self, tau_ned_est) -> None:
        """The wrench the allocation believes the vehicle realized (axis caps
        applied, K/M zeroed). This is what the EAOB must see next tick —
        feeding it the raw solver output would count the cap/drop mismatch as
        disturbance."""
        self._tau_ned_cmd = np.asarray(tau_ned_est, float).copy()

    def reset(self) -> None:
        self.eaob = None
        self.w_hat = np.zeros(6)
        self._tau_ned_cmd = np.zeros(6)
        self._ref_traj = None
        self.scenario = None
        self.r_ref = 0.0
        self.nmpc.reset()
        self.n_fail = 0

    # ---------------------------------------------------------------- logging
    def eta_ned_to_flu(self, eta_ned):
        """(p_flu, yaw_flu, pitch_flu) for the sim-compatible CSV columns."""
        p_flu, R_flu = self.frames.ned_to_flu_eta(np.asarray(eta_ned, float))
        yaw = float(np.arctan2(R_flu[1, 0], R_flu[0, 0]))
        pitch = float(np.arcsin(np.clip(-R_flu[2, 0], -1.0, 1.0)))
        return p_flu, yaw, pitch

    def meta(self) -> dict:
        P = self.P
        m = {"type": self.mode, "solver": self.solver_kind,
             "ctrl_hz": float(self.cfg.ctrl_hz), "N": int(P.MPC_N),
             "u_max": [float(v) for v in P.U_MAX],
             "w_hat_clip": [float(v) for v in self.w_clip],
             "rov_model": P.MODEL, "ref_preview": True,
             "mpc_state_source": "meas",
             "probe_ms": self.probe_ms, "build_s": round(self.build_s, 2),
             "eaob_sigma_overrides": self.sigma_applied}
        if self.eaob is not None:
            m["eaob"] = {"profile": "perf", "tau_dist": float(P.EAOB_TAU_DIST),
                         "n_upd": int(self.eaob.n_upd),
                         "n_gated": int(self.eaob.n_gated)}
        return m
