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
    realized (axis caps applied; K/M sent only under ``engage.attitude_axes``,
    else dropped — see allocation.py), reported back via :meth:`note_applied`,
    never the raw solver output; the axis-cap and dropped-moment mismatch
    would otherwise be double-counted as disturbance;
  * ATTITUDE REFERENCE (2026-09-26, the 6-DoF variant): a streamed plan may
    carry ``rp_ned`` / ``rp_rate_ned`` (path_geometry.NedPlan), which fill
    xref rows 3:5 and — through T(phi, theta)^-1 — the body-rate rows 9:11
    (:meth:`_xref_ned_plan`); every hold entry after a tracked plan RAMPS
    the attitude reference back to level at ``policy.pq_max_rad_s`` inside
    this object (``_rp_hold``, design D5) instead of stepping — and the
    ramp is PREVIEWED along the horizon (:meth:`_rp_hold_preview`: stage k
    carries the angle the per-tick decay will have reached k ticks on, with
    the matching Euler rate through T^-1) in every reference tile, the
    trajectory sampler's included. With no plan ever carrying rp the
    reference is byte-identical to the 4-DoF one;
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
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from .state_assembler import rot_zyx

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
    rov_model = str(rov_model)
    if _dob is not None:
        # the model this process pinned (requested), or the one that was
        # actually loaded when they differ (see the warning below)
        if rov_model not in (_dob_model, _dob.get("model")):
            raise RuntimeError(
                f"dobmpc already imported for ROV_MODEL={_dob_model!r}; "
                f"cannot re-import for {rov_model!r} in this process")
        return _dob
    if not _MARINEGYM.is_dir():
        raise FileNotFoundError(f"marinegym tree not found at {_MARINEGYM}")
    os.environ["ROV_MODEL"] = rov_model
    for attempt in (0, 1):
        if str(_MARINEGYM) not in sys.path:
            sys.path.insert(0, str(_MARINEGYM))
        try:
            from dobmpc import frames, params
            from dobmpc.eaob import EAOB
            from dobmpc.fossen import wrap_angle
            from dobmpc.mpc import make_nmpc
            break
        except ModuleNotFoundError as e:
            # ``import rov_model`` inside dobmpc/params.py (name
            # "rov_model") — or the ``dobmpc`` package itself (name
            # "dobmpc") — can fail although the path was just inserted: a
            # concurrent ``import cv2`` in another thread
            # (rov_gui.qt.import_cv2, the PolicyWorker's setup) REPLACES
            # sys.path with a pre-import snapshot and drops the insert
            # (opencv __init__.py bootstrap; demo_e2e ``mpc policy`` 2/3
            # runs, 2026-09-26). Which name surfaces depends on where the
            # race lands. The failed submodules were removed from
            # sys.modules, so one re-insert + retry is exact; anything
            # else propagates.
            if e.name not in _RETRY_IMPORT_NAMES or attempt:
                raise
    actual = str(getattr(params, "MODEL", rov_model))
    if actual != rov_model:
        # Someone imported dobmpc BEFORE this call (e.g. an mpcc module's
        # bare ``from dobmpc import params``) under another $ROV_MODEL, so
        # the physics in memory are NOT the ones asked for. A warning, not
        # an error: raising here would take down a station that is
        # otherwise flyable; the mismatch is made visible in the returned
        # dict (``model`` vs ``model_requested``) and in HwDobMpc.meta().
        _warn_model_mismatch("mpc_bridge.import_dobmpc", rov_model, actual)
    _dob = {"frames": frames, "P": params, "EAOB": EAOB,
            "wrap_angle": wrap_angle, "make_nmpc": make_nmpc,
            "model": actual, "model_requested": rov_model}
    _dob_model = rov_model
    return _dob


#: ModuleNotFoundError names the cv2 sys.path race can surface as (see
#: :func:`import_dobmpc`): the package itself, or the shared ``rov_model``
#: module dobmpc/params.py imports.
_RETRY_IMPORT_NAMES = ("dobmpc", "rov_model")


def _warn_model_mismatch(where: str, requested: str, loaded: str) -> None:
    """dobmpc.params was already in memory under a different ROV_MODEL than
    the one asked for. Logged AND warned (never raised): the record must
    show it, the station must keep running."""
    msg = (f"{where}: dobmpc already imported as ROV_MODEL={loaded!r}; "
           f"{requested!r} was requested — the controller/plant record is "
           f"the LOADED model, not the requested one")
    logging.getLogger(__name__).warning(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=3)


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


#: Keys ``plant:`` may carry. Both are the marinegym POSITIVE damping
#: coefficients, order [surge sway heave roll pitch yaw]; params.py stores
#: them negated (``DL = -_LINEAR_DAMPING``) and this module does the negation,
#: so the config reads like the YAML the numbers came from.
PLANT_KEYS = {"linear_damping": "DL", "quadratic_damping": "DNL"}


def resolve_plant(plant: dict) -> dict:
    """Validate a ``plant:`` block at CONFIG time (geometry.MpcConfig), so a
    typo cannot reach a solver build. Unknown keys RAISE, same rule as
    station_bridge / mpc_tuned: a silently ignored ``linear_dampng:`` means
    the vehicle flies the SIM's drag while its meta says otherwise."""
    if not plant:
        return {}
    unknown = set(plant) - set(PLANT_KEYS)
    if unknown:
        raise ValueError(f"plant: unknown key(s) {sorted(unknown)}; "
                         f"allowed {sorted(PLANT_KEYS)}")
    out = {}
    for k in plant:
        v = np.asarray(plant[k], float).ravel()
        if v.size != 6 or not np.all(np.isfinite(v)) or np.any(v < 0.0):
            raise ValueError(
                f"plant.{k} must be 6 finite non-negative coefficients "
                f"[surge sway heave roll pitch yaw], got {plant[k]!r}")
        out[k] = [float(x) for x in v]
    return out


def apply_plant_overrides(P, plant: dict) -> dict:
    """Replace the PREDICTION MODEL's damping with values measured on THIS
    vehicle, BEFORE the solver is generated.

    Why this exists (2026-09-08). The generated plant carries marinegym's
    BlueROV.yaml damping, and against the hardware that is far too low: the
    whole-loop relation measured over 29 runs, ``speed = 0.692 (|axis| -
    0.096)`` (hw_mpc.yaml), implies a steady-state surge drag of
    ``axis_gain / 0.692 = 86.7 N.s/m`` where the model carries 4.03 — so the
    solver asked for ~2 N to move at the policy's 0.05 m/s, the T200 deadband
    ate all of it and the vehicle sat still. hw_mpc.yaml has said since
    2026-08-17 that the fix "belongs in the PREDICTION MODEL's drag"; this is
    that fix, and it is a FEEDFORWARD one — it does not touch Q, R, the axis
    gains or the observer, so the force appears without waiting for tracking
    error to grow (which ``policy.anchor: measured`` prevents anyway).

    DL/DNL are baked into the CasADi expression at build time, so this must
    run before ``make_nmpc`` AND the values must ride in the rebuild key
    (``_guard_stale_solver``) or a cached .so silently keeps the old drag.
    Returns what was applied, for the run meta."""
    applied = {}
    for key, attr in PLANT_KEYS.items():
        v = (plant or {}).get(key)
        if v is None:
            continue
        arr = np.asarray(v, float).ravel()
        if arr.size != 6 or not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
            raise ValueError(f"plant.{key} must be 6 finite non-negative "
                             f"coefficients, got {v!r}")
        was = np.abs(np.asarray(getattr(P, attr), float)).ravel()
        setattr(P, attr, -arr)          # params.py stores these NEGATED
        applied[key] = {
            "value": [float(x) for x in arr],
            "model_was": [float(x) for x in was],
            "ratio": [round(float(a / b), 3) if b else None
                      for a, b in zip(arr, was)],
        }
    return applied


def _guard_stale_solver(P) -> None:
    """The acados codegen cache is keyed only on (model, rti): N/dt/U_MAX or
    DAMPING changes load a STALE .so under DOBMPC_ACADOS_BUILD=0. A meta
    sidecar makes that combination loud: on mismatch the fast-load flag is
    dropped so the solver regenerates this once."""
    variant = f"{P.MODEL}_rti"
    gen = _MARINEGYM / "dobmpc" / "_acados_gen" / variant
    meta_p = gen / "build_meta_rovgui.json"
    want = {"model": P.MODEL, "N": int(P.MPC_N), "dt": float(P.DT_CTRL),
            "nu": int(P.NU), "u_max": [float(v) for v in P.U_MAX],
            # damping is IN the generated f_expl (mpc_acados._build_model),
            # so a plant: override that is not in this key would be silently
            # dropped by a cached build
            "dl": [float(v) for v in P.DL], "dnl": [float(v) for v in P.DNL]}
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

    #: every mode this ONE solver serves. The ``dob`` prefix decides whether
    #: the EAOB's w_hat reaches the solver; the ``_tuned`` suffix decides
    #: whether the position weight is rotated into the path frame
    #: (path_cost.py). Both are runtime flags on the same generated OCP —
    #: neither costs a second acados build.
    MODES = ("mpc", "dobmpc", "mpc_tuned", "dobmpc_tuned")

    #: May a ``follow`` mission be armed on this controller? Yes: this bridge
    #: takes a MOVING setpoint with a velocity feedforward (``set_target_ned``
    #: keeps ``v_ned`` and extrapolates it across the horizon), which is
    #: exactly what following a moving object needs. Contrast HwMpcc, which
    #: discards it. Read fail-closed by MpcWorker via ``getattr(..., False)``,
    #: so a controller that has not thought about this cannot be followed on.
    follow_ok = True

    @staticmethod
    def heave_trim_wrench(model_net_buoyancy_n: float,
                          vehicle_net_buoyancy_n: float) -> np.ndarray:
        """The constant NED-body wrench that turns the solver's plant into
        the vehicle actually flown. Both arguments are B - W (+ floats).

        The model reads ``nu_dot = M^-1(tau + w - ... - g)`` with
        ``g_z = -(W - B) cos(theta) cos(phi)``, so at rest the modelled net
        vertical force is ``w_z + (W - B)_model``; matching the real one,
        ``(W - B)_real``, needs ``w_z = (W - B)_real - (W - B)_model``
        ``= (B - W)_model - (B - W)_real``. A neutral vehicle on the -5.71 N
        heavy_gripper plant therefore gets w_z = -5.71 N.

        LEVEL APPROXIMATION: the exact correction rotates with attitude,
        ``delta * [sin(theta), -cos(theta) sin(phi), -cos(theta) cos(phi), 0, 0, 0]``;
        this constant body-z form is exact at level trim and leaves a
        ``delta * sin(theta)`` surge residual (1.0 N at 10 deg pitch for
        delta 5.71 N, under the T200 deadband) — fine while the pitch guard
        holds the vehicle near level."""
        w = np.zeros(6)
        w[2] = float(model_net_buoyancy_n) - float(vehicle_net_buoyancy_n)
        return w

    def __init__(self, mode: str, mpc_cfg, log=print):
        assert mode in self.MODES, mode
        self._mode = mode
        self.dob = mode.startswith("dobmpc")
        self.tuned = mode.endswith("_tuned")
        self._w_tuned = False       # is a rotated W currently in the solver?
        self._wt = None
        self._plan_q_scale = 1.0    # policy.q_scale while a policy plan is installed
        self._plan_q_scale_flown = 1.0   # max scale since reset(): what THIS engagement flew
        self._plan_along_scale = None      # policy.along_scale while a policy plan is installed
        self._plan_along_scale_flown = None   # the value THIS engagement flew (None = controller's own)
        self._wt_cfg_tune = None           # the config's along/cross tune, restored after a policy plan
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
        # The prediction model's damping, BEFORE the guard (which carries it in
        # the rebuild key) and before make_nmpc (which bakes it into the C code).
        self.plant_applied = apply_plant_overrides(
            P, getattr(mpc_cfg, "plant", None))
        if self.plant_applied:
            for k, a in self.plant_applied.items():
                log(f"mpc: plant {k} OVERRIDDEN {a['model_was']} -> "
                    f"{a['value']} (x{a['ratio']}) — the solver's drag is the "
                    f"vehicle's, not marinegym's; runs are not comparable "
                    f"across this boundary")
        _guard_stale_solver(P)
        t0 = time.perf_counter()
        self.nmpc = d["make_nmpc"](N=P.MPC_N, dt=P.DT_CTRL)
        self.build_s = time.perf_counter() - t0
        self.solver_kind = ("acados" if type(self.nmpc).__name__ == "AcadosNMPC"
                            else "ipopt")
        # NO IPOPT INSIDE THE 20 Hz TICK. acados' failure path builds the
        # casadi IPOPT NMPC lazily and solves it to full convergence, in the
        # calling thread. The sim can afford that; this loop cannot. 2026-08-23
        # one status-4 solve blocked the MPC worker ~6.5 s, and the sink's
        # 500 ms deadman plus the window's 1.5 s freshness watchdog released
        # control to the pilot in the middle of a follow. Holding the previous
        # wrench is the bounded answer, and `max_solver_fails` still disengages
        # on the third consecutive failure.
        if hasattr(self.nmpc, "disable_fallback"):
            self.nmpc.disable_fallback("real-time station loop")
            log("mpc: acados IPOPT fallback DISABLED — a failed solve holds "
                "the previous wrench; 3 in a row disengage")
        self.w_clip = np.asarray(mpc_cfg.w_hat_clip, float)
        # HEAVE TRIM (2026-09-07). The plant this solver was generated with
        # sinks at P.NET_BUOYANCY (B - W = -5.71 N for heavy_gripper); the
        # vehicle in the pool does not (mpc_cfg.vehicle_net_buoyancy_n, B - W of the
        # real vehicle, default 0). A nominal NMPC cancels its own model's
        # gravity, so the mismatch was a constant wrong-signed 5.7 N
        # up-force at zero depth error: with the T200 deadband (~5.8 N)
        # stacked on top, the follower needed ~0.19 m of depth error before
        # any thrust went DOWN, and the policy leash caps that error at
        # ~0.10-0.15 m — the vehicle drifted up under a plan that pointed
        # down (0907_180038, five missions, fit uZ = -5.6 N - 60 N/m * ez).
        # The model reads nu_dot = M^-1(tau + w - C - D - g) with w the
        # per-solve parameter, so the difference is a runtime input and no
        # rebuild is needed. mpc / mpc_tuned ONLY: under dobmpc the EAOB's
        # w_hat already carries this residual and adding it twice would
        # double count. Not an observer — a constant, recorded in meta().
        from .geometry import check_vehicle_net_buoyancy

        self.vehicle_net_buoyancy_n = check_vehicle_net_buoyancy(
            getattr(mpc_cfg, "vehicle_net_buoyancy_n", 0.0))
        self.w_trim = self.heave_trim_wrench(float(P.NET_BUOYANCY),
                                             self.vehicle_net_buoyancy_n)
        log(f"mpc: heave trim w_z {self.w_trim[2]:+.2f} N "
            f"(plant B-W {float(P.NET_BUOYANCY):+.2f} N, config vehicle_net_buoyancy_n "
            f"{self.vehicle_net_buoyancy_n:+.2f} N)"
            + (" — NOT applied under dobmpc: the EAOB estimates it"
               if self.dob else " — constant, through the disturbance parameter"))
        # The along/cross split. Built for EVERY mode, not just the tuned
        # ones, so the baseline W is always on hand to write back and so the
        # meta can state what the tuned modes WOULD have used.
        from .path_cost import PathFrameWeights

        self._wt_cfg_tune = getattr(mpc_cfg, "mpc_tuned", None)
        self._wt = PathFrameWeights(P.MPC_Q, P.MPC_R, P.MPC_QN, self._wt_cfg_tune)
        if self.tuned and self.solver_kind != "acados":
            raise RuntimeError(
                f"{mode} needs the acados solver (the per-stage weight is an "
                f"acados cost_set); make_nmpc fell back to {self.solver_kind}")
        if self.tuned:
            log(f"mpc: {mode} — path-frame Q, q_along {self._wt.q_along:.1f} "
                f"/ q_cross {self._wt.q_cross:.1f} "
                f"(x{self._wt.anisotropy:.1f}) on a {self._wt.q_xy:.1f} "
                f"isotropic baseline"
                + ("" if self._wt.tune["apply_terminal"]
                   else "; TERMINAL LEFT ISOTROPIC")
                + ("; velocity split ON" if self._wt.tune["split_velocity"]
                   else ""))

        # references (world FLU — the sim wrapper's convention, kept verbatim)
        self.p_ref = np.zeros(3)
        self.yaw_ref = 0.0
        self.v_ref = np.zeros(3)
        self.r_ref = 0.0
        self.yaw_target = 0.0
        self._ref_traj = None
        self._path_plan = None
        self.scenario: dict | None = None
        # ---- attitude reference (2026-09-26). `_rp_hold` is the NED (roll,
        # pitch) the hold reference carries while ramping back to level after
        # a tracked plan / an explicit rp_ref; None = never tracked = the
        # 4-DoF byte-identical path. Read from mpc_cfg.policy with .get so a
        # bare namespace config (tests) resolves the design defaults.
        pol = getattr(mpc_cfg, "policy", None) or {}
        _g = pol.get if hasattr(pol, "get") else (lambda k, d=None: d)
        self._rp_hold = None
        self._pq_max = float(_g("pq_max_rad_s", 0.35))
        self._rp_reject_rad = float(np.radians(float(_g("rp_reject_deg", 30.0))))
        self._rp_max_rad = float(np.radians(float(_g("rp_max_deg", 20.0))))
        self._heave_trim_rotated = bool(_g("heave_trim_attitude_rotated", False))
        self._attitude_axes = False          # set_attitude_axes (worker, at engage)
        self._u_max_wire_nm = None
        self._rp_tracked_any = False         # did THIS engagement ever fly an rp plan?

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
    # ``MpcWorker.set_mode`` assigns ``ctrl.mode = mode`` to flip between the
    # modes this one object serves, so the flags must move with it — and the
    # solver must not keep a rotated W after a switch back to the baseline.
    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        value = str(value)
        if value not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {value!r}")
        if value.endswith("_tuned") and self.solver_kind != "acados":
            raise RuntimeError(
                "mpc_tuned needs the acados solver (the per-stage weight is "
                f"an acados cost_set); this build fell back to "
                f"{self.solver_kind}")
        self._mode = value
        self.dob = value.startswith("dobmpc")
        self.tuned = value.endswith("_tuned")
        if not self.tuned:
            self._restore_base_weights()

    #: solves per probe after the warm-up one; the probe is their MEDIAN
    PROBE_N = 5

    def _probe(self, log) -> None:
        """Throwaway solves so 'which solver, how fast' is a fact, not a
        hope — make_nmpc downgrades to IPOPT silently on any acados problem.

        ONE cold solve is not a real-time measurement: the first call into a
        freshly loaded acados library pays page-faults and allocations, and
        on the station it lands while FoundationStereo captures its CUDA
        graph and the GUI thread is busy. 2026-09-08 pool: the single probe
        read 79.7 ms right after a pid -> mpc_tuned swap and refused every
        START for the rest of the session ("solver not real-time"), while
        the same solver runs the 20 Hz loop at a few ms. So: one warm-up
        solve (recorded as probe_ms_first), then PROBE_N timed solves whose
        MEDIAN is probe_ms. reprobe() repeats it on demand."""
        x = np.zeros(12)
        xref = np.zeros((12, self.nmpc.N + 1))
        times = []
        try:
            t0 = time.perf_counter()
            self.nmpc.solve(x, np.zeros(6), xref)          # warm-up (cold)
            self.probe_ms_first = 1e3 * (time.perf_counter() - t0)
            for _ in range(int(self.PROBE_N)):
                t0 = time.perf_counter()
                self.nmpc.solve(x, np.zeros(6), xref)
                times.append(1e3 * (time.perf_counter() - t0))
            self.probe_ms = float(np.median(times))
            self.probe_ms_worst = float(max(times))
            self.nmpc.reset()
        except Exception as e:                                   # noqa: BLE001
            log(f"mpc probe solve failed: {type(e).__name__}: {e}")
            self.probe_ms = None
            self.probe_ms_first = getattr(self, "probe_ms_first", None)
            self.probe_ms_worst = None
        lim = float(self.cfg.engage.get("probe_ms_max", 25.0))
        self.realtime_ok = (self.solver_kind == "acados"
                            and self.probe_ms is not None
                            and self.probe_ms < lim)
        self.probe_n = int(getattr(self, "probe_n", 0)) + 1
        fmt = lambda v: "None" if v is None else f"{v:.1f}"
        log(f"mpc solver: {self.solver_kind}, probe median {fmt(self.probe_ms)} ms "
            f"of {self.PROBE_N} (first/cold {fmt(getattr(self, 'probe_ms_first', None))}, "
            f"worst {fmt(getattr(self, 'probe_ms_worst', None))}), "
            f"build {self.build_s:.1f}s, realtime_ok={self.realtime_ok}")

    def reprobe(self, log=None) -> bool:
        """Measure again (a transient load spike must not lock the session
        out). Returns the fresh ``realtime_ok``. Only meaningful between
        engagements: it solves on the shared solver and resets it."""
        self._probe(log or (lambda m: None))
        return bool(self.realtime_ok)

    # -------------------------------------------------- sim-ported reference API
    # PORTED from dobmpc_controller.py:126 (see module docstring)
    def set_target(self, p_ref=None, yaw_ref=None, v_ref=None, r_ref=None,
                   yaw_target=None, rp_ref=None):
        """``rp_ref`` (2026-09-26): NED ``(roll, pitch)`` rad the hold
        reference starts from (then ramps to level, see ``_rp_hold``); None
        = keep whatever the dropped plan left (D5), or level if none."""
        if p_ref is not None:
            self.p_ref = np.asarray(p_ref, float)
        if yaw_ref is not None:
            self.yaw_ref = float(yaw_ref)
        if v_ref is not None:
            self.v_ref = np.asarray(v_ref, float)
        if r_ref is not None:
            self.r_ref = float(r_ref)
        self.yaw_target = float(yaw_target) if yaw_target is not None else self.yaw_ref
        self._drop_plan_keep_attitude()
        if rp_ref is not None:
            self._set_rp_hold(rp_ref)

    # PORTED from dobmpc_controller.py:141
    def set_reference_traj(self, fn):
        """fn(ts (K,)) -> p (3,K), yaw (K,), v (3,K), r (K,) — ALL WORLD FLU."""
        self._ref_traj = fn
        self._drop_plan_keep_attitude()

    # ------------------------------------------------ attitude hold ramp (D5)
    def _set_rp_hold(self, rp) -> None:
        rp = np.asarray(rp, float).ravel()
        if rp.shape != (2,) or not np.all(np.isfinite(rp)):
            raise ValueError(f"rp_ref must be a finite (roll, pitch) pair, got {rp!r}")
        lim = float(getattr(self, "_rp_reject_rad", np.radians(30.0)))
        if abs(float(rp[0])) > lim or abs(float(rp[1])) > lim:
            raise ValueError(f"rp_ref {np.degrees(rp)} deg exceeds rp_reject "
                             f"{np.degrees(lim):.1f} deg")
        self._rp_hold = rp.copy()
        self._rp_tracked_any = True

    def _drop_plan_keep_attitude(self) -> None:
        """Forget the installed plan; if it carried an attitude, the hold
        reference STARTS from its stage-0 attitude and ramps to level
        (``step``) rather than stepping — a Q=80 torque spike at every plan
        boundary otherwise (design D5). A plan without rp leaves ``_rp_hold``
        alone, so the 4-DoF path stays byte-identical (None)."""
        plan = getattr(self, "_path_plan", None)
        rp = None if plan is None else getattr(plan, "rp_ned", None)
        if rp is not None:
            self._rp_hold = np.asarray(rp, float)[:, 0].copy()
        self._path_plan = None

    def _decay_rp_hold(self) -> None:
        """One tick of the hold ramp: each axis moves toward 0 by at most
        ``pq_max * DT_CTRL``; None once both are there."""
        if getattr(self, "_rp_hold", None) is None:
            return
        step = float(getattr(self, "_pq_max", 0.35)) * float(self.P.DT_CTRL)
        rp = self._rp_hold
        self._rp_hold = np.sign(rp) * np.maximum(np.abs(rp) - step, 0.0)
        if not np.any(self._rp_hold):
            self._rp_hold = None

    def _rp_hold_preview(self, n_stages: int):
        """The hold ramp along the horizon, consistent with the per-tick
        decay in :meth:`_decay_rp_hold`: ``(rp_k (2, n), rpd_k (2, n))`` with

            rp_k  = sign(rp_hold) * max(|rp_hold| - pq_max * k * dt, 0)
            rpd_k = -sign(rp_hold) * pq_max   while |rp_k| > 0, else 0

        per axis, so stage k of the reference is exactly what the decay
        will have produced k ticks from now (a horizon-CONSTANT angle with
        zero rate rows — the 2026-09-26 first cut — biased the attitude to
        LAG the ramp: Q[9:11] = 10 pulled toward zero body rate while the
        angle moved at pq_max). None when no hold is running, the 4-DoF
        byte-identical path."""
        rp_hold = getattr(self, "_rp_hold", None)
        if rp_hold is None:
            return None
        rp = np.asarray(rp_hold, float).reshape(2)
        pq_max = float(getattr(self, "_pq_max", 0.35))
        dt = float(self.P.DT_CTRL)
        ks = np.arange(int(n_stages), dtype=float)
        sgn = np.sign(rp)[:, None]
        mag = np.maximum(np.abs(rp)[:, None] - pq_max * dt * ks[None, :], 0.0)
        rp_k = sgn * mag
        rpd_k = np.where(mag > 0.0, -sgn * pq_max, 0.0)
        return rp_k, rpd_k

    def _overlay_rp_hold(self, xref: np.ndarray) -> np.ndarray:
        """Write the hold-ramp preview into rows 3:5 (angles) and 9:12
        (body rates through T^-1, ``psid_k`` = whatever row 11 already
        carries: the yaw-ramp r_ned, a plan's r, a sampler's -r_w, or 0).
        A no-op — the SAME object, untouched — when ``_rp_hold`` is None."""
        prev = self._rp_hold_preview(xref.shape[1])
        if prev is None:
            return xref
        rp_k, rpd_k = prev
        xref[3:5, :] = rp_k
        pk, qk, rk = self.body_rates_from_euler(rp_k[0], rp_k[1],
                                                rpd_k[0], rpd_k[1],
                                                xref[11, :])
        xref[9, :] = pk
        xref[10, :] = qk
        xref[11, :] = rk
        return xref

    # PORTED from dobmpc_controller.py:183 (docstring compressed)
    def _xref_ned(self, t=None):
        if getattr(self, "_path_plan", None) is not None:
            return self._xref_ned_plan(self._path_plan)
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
        # the D5 ramp previewed along the horizon (no-op when never tracked)
        return self._overlay_rp_hold(xref)

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
        # A tracked plan that ended into a square/line/circle
        # (set_reference_traj -> _drop_plan_keep_attitude) must ramp here
        # too, not STEP to eta_k[3:5] = 0 while ref_attitude_ned_at reports
        # the ramp (the first cut did exactly that). No-op when never tracked.
        return self._overlay_rp_hold(xref)

    @staticmethod
    def body_rates_from_euler(phi, theta, phid, thetad, psid):
        """``(p, q, r) = T(phi, theta)^-1 (phid, thetad, psid)`` — the inverse
        of the kinematic T in dobmpc/mpc.py (Fossen eq. 2.28):
            p = phid − psid·sin(theta)
            q = thetad·cos(phi) + psid·cos(theta)·sin(phi)
            r = −thetad·sin(phi) + psid·cos(theta)·cos(phi)
        At phi = theta = 0 this is (phid, thetad, psid) exactly."""
        sf, cf = np.sin(phi), np.cos(phi)
        st, ct = np.sin(theta), np.cos(theta)
        return (phid - psid * st,
                thetad * cf + psid * ct * sf,
                -thetad * sf + psid * ct * cf)

    def _xref_ned_plan(self, plan):
        """Convert the worker's shared world-NED path plan to NMPC state xref.

        With ``plan.rp_ned`` (2026-09-26) rows 3:5 take the plan's absolute
        roll/pitch, rows 9:11 (and 11) the body rates from the Euler rates
        ``rp_rate_ned`` (0 when absent) and ``r_ned`` through T^-1, and the
        body-velocity rows use the full R(phi, theta, psi). ``rp_ned`` None
        = today's level reference, byte for byte (plus the D5 hold-ramp
        preview in rows 3:5 / 9:12 while one is running,
        :meth:`_overlay_rp_hold` — never on the 4-DoF path, where
        ``_rp_hold`` is None)."""
        N = self.nmpc.N
        p = np.asarray(plan.p_ned, float)
        yaw = np.asarray(plan.yaw_ned, float).ravel()
        v = np.asarray(plan.v_ned, float)
        r = np.asarray(plan.r_ned, float).ravel()
        expected = N + 1
        if (p.shape != (3, expected) or v.shape != (3, expected)
                or yaw.size != expected or r.size != expected):
            raise ValueError(
                f"MPC path plan must contain {expected} stages, got "
                f"p={p.shape}, v={v.shape}, yaw={yaw.size}, r={r.size}")
        rp = getattr(plan, "rp_ned", None)
        rpr = getattr(plan, "rp_rate_ned", None)
        if rp is not None:
            rp = np.asarray(rp, float)
            if rp.shape != (2, expected) or not np.all(np.isfinite(rp)):
                raise ValueError(f"MPC path plan rp_ned must be a finite "
                                 f"(2, {expected}) array, got {rp.shape}")
            if rpr is not None:
                rpr = np.asarray(rpr, float)
                if rpr.shape != (2, expected) or not np.all(np.isfinite(rpr)):
                    raise ValueError(f"MPC path plan rp_rate_ned must be a finite "
                                     f"(2, {expected}) array, got {rpr.shape}")
        xref = np.zeros((12, expected))
        xref[0:3, :] = p
        psi_prev = self._psi_ned_now
        if rp is None:
            for k in range(expected):
                psi_prev += self.wrap_angle(float(yaw[k]) - psi_prev)
                xref[5, k] = psi_prev
                Rk = rot_zyx(0.0, 0.0, psi_prev)
                xref[6:9, k] = Rk.T @ v[:, k]
            xref[11, :] = r
            return self._overlay_rp_hold(xref)
        for k in range(expected):
            psi_prev += self.wrap_angle(float(yaw[k]) - psi_prev)
            xref[5, k] = psi_prev
            phi, theta = float(rp[0, k]), float(rp[1, k])
            xref[3, k] = phi
            xref[4, k] = theta
            phid, thetad = ((float(rpr[0, k]), float(rpr[1, k]))
                            if rpr is not None else (0.0, 0.0))
            pk, qk, rk = self.body_rates_from_euler(phi, theta, phid, thetad,
                                                    float(r[k]))
            xref[9, k] = pk
            xref[10, k] = qk
            xref[11, k] = rk
            Rk = rot_zyx(phi, theta, psi_prev)
            xref[6:9, k] = Rk.T @ v[:, k]
        return xref

    # ------------------------------------------------------- NED-facing helpers
    # The station's world is NED (the tag map); the ported methods above speak
    # the sim's world-FLU. The mirror is frames.S applied ONCE, here.
    def set_target_ned(self, p_ned, yaw_ned, v_ned=None, r_ned=0.0,
                       rp_ref=None):
        """``rp_ref`` = NED ``(roll, pitch)`` rad, see :meth:`set_target`."""
        S = self.frames.S
        self.set_target(p_ref=S @ np.asarray(p_ned, float),
                        yaw_ref=-float(yaw_ned),
                        v_ref=(np.zeros(3) if v_ned is None
                               else S @ np.asarray(v_ned, float)),
                        r_ref=-float(r_ned), rp_ref=rp_ref)
        self._ref_traj = None

    @property
    def path_plan_steps(self) -> int:
        return int(self.nmpc.N) + 1

    @property
    def path_plan_dt(self) -> float:
        return float(self.P.DT_CTRL)

    def set_path_plan_ned(self, plan) -> None:
        """Install the geometry-driven plan produced once by MpcWorker.

        A plan carrying ``rp_ned`` (2026-09-26) is validated here (T3 of the
        cap tiers: shape (2, N+1), finite, |rp| <= ``policy.rp_reject_deg``,
        else ValueError — which the worker's install path turns into a
        disengage). When a plan WITH rp is replaced by one WITHOUT (or by
        None), the hold ramp starts from the outgoing plan's stage-0
        attitude (D5)."""
        if plan is not None:
            expected = self.path_plan_steps
            if np.asarray(plan.p_ned).shape != (3, expected):
                raise ValueError(f"MPC path plan needs {expected} stages")
            rp = getattr(plan, "rp_ned", None)
            if rp is not None:
                rp = np.asarray(rp, float)
                if rp.shape != (2, expected):
                    raise ValueError(f"MPC path plan rp_ned needs shape "
                                     f"(2, {expected}), got {rp.shape}")
                if not np.all(np.isfinite(rp)):
                    raise ValueError("MPC path plan rp_ned must be finite")
                worst = float(np.max(np.abs(rp)))
                lim = float(getattr(self, "_rp_reject_rad", np.radians(30.0)))
                if worst > lim:
                    raise ValueError(
                        f"MPC path plan attitude {np.degrees(worst):.1f} deg "
                        f"exceeds rp_reject {np.degrees(lim):.1f} deg")
                rpr = getattr(plan, "rp_rate_ned", None)
                if rpr is not None:
                    rpr = np.asarray(rpr, float)
                    if rpr.shape != (2, expected) or not np.all(np.isfinite(rpr)):
                        raise ValueError(f"MPC path plan rp_rate_ned needs a finite "
                                         f"(2, {expected}) array, got {rpr.shape}")
                self._rp_tracked_any = True
                self._rp_hold = None          # the plan owns the attitude now
                self._path_plan = plan
                return
        self._drop_plan_keep_attitude()
        self._path_plan = plan

    def set_square_ned(self, square: dict, origin_ned_xy, yaw_fixed_ned: float,
                       depth_ned: float):
        """Arm the tracking sampler with the square placed in NED. Returns the
        resolved scenario dict (recorded in meta / drawn by the UI).

        ``rot_deg`` rotates the rectangle in the MAP frame; 0 means the sides
        are parallel to the tag map's x and y axes, and the origin (the
        entered tag) is the min-x / min-y corner — bottom left of the
        top-down plot. Both properties come from ``mirror_y=True`` plus the
        map->datum rotation the caller folds into ``rot_deg``."""
        from .reference import place_square_ned

        fn, self.scenario = place_square_ned(
            square, origin_ned_xy, yaw_fixed_ned, depth_ned,
            dt=self.P.DT_CTRL, preview_s=self.nmpc.N * self.P.DT_CTRL)
        self.set_reference_traj(fn)
        return self.scenario

    def set_line_ned(self, line: dict, origin_ned_xy, yaw_fixed_ned: float,
                     depth_ned: float):
        """Arm the tracking sampler with an OUT-AND-BACK line placed in NED.

        ``line["dir_deg"]`` is the heading of the outbound leg in the NED map
        frame (0 = +x, 90 = +y), so "2 m along +y from tag 79" is dir_deg 90.
        Heading is held at ``yaw_fixed_ned`` throughout — the vehicle crabs,
        it does not turn around, so the camera keeps the same floor patch and
        the localizer's tag set stays continuous."""
        from .reference import place_line_ned

        fn, self.scenario = place_line_ned(
            line, origin_ned_xy, yaw_fixed_ned, depth_ned,
            dt=self.P.DT_CTRL, preview_s=self.nmpc.N * self.P.DT_CTRL)
        self.set_reference_traj(fn)
        return self.scenario

    def set_circle_ned(self, circle: dict, origin_ned_xy, yaw_fixed_ned: float,
                       depth_ned: float):
        """Arm the tracking sampler with a CIRCLE placed in NED.

        ``origin_ned_xy`` is a point on the RIM — the tag the operator entered
        — and ``circle["radius"]`` is the radius; the centre sits one radius
        along ``rot_deg`` from the tag, so rot 0 makes the tag the circle's
        minimum-x point (the bottom of the top-down plot). Placement is
        ``reference.place_circle_ned``, the same single definition HwPid and
        HwMpcc call, so the three controllers fly one geometry."""
        from .reference import place_circle_ned

        fn, self.scenario = place_circle_ned(
            circle, origin_ned_xy, yaw_fixed_ned, depth_ned,
            dt=self.P.DT_CTRL, preview_s=self.nmpc.N * self.P.DT_CTRL)
        self.set_reference_traj(fn)
        return self.scenario

    def ref_ned_at(self, t: float) -> tuple[np.ndarray, float, np.ndarray]:
        """Current ``(position, yaw, world velocity)`` reference in NED.

        NMPC stores linear velocity in the reference body frame, so rotate
        stage 0 back into world NED for this public NED-facing helper.
        """
        if getattr(self, "_path_plan", None) is not None:
            plan = self._path_plan
            return (np.asarray(plan.p_ned[:, 0], float).copy(),
                    float(plan.yaw_ned[0]),
                    np.asarray(plan.v_ned[:, 0], float).copy())
        xref = self._xref_ned(t)
        phi, theta, psi = (float(v) for v in xref[3:6, 0])
        v_ref_ned = rot_zyx(phi, theta, psi) @ xref[6:9, 0]
        return xref[0:3, 0].copy(), psi, v_ref_ned.copy()

    def ref_attitude_ned_at(self, t: float | None = None) -> tuple[float, float]:
        """Current NED ``(roll, pitch)`` reference, rad (2026-09-26): the
        installed plan's stage 0, else the hold ramp, else level. For the
        CSV `rroll/rpitch` columns and the interlocks; ``ref_ned_at`` keeps
        its 3-tuple."""
        plan = getattr(self, "_path_plan", None)
        rp = None if plan is None else getattr(plan, "rp_ned", None)
        if rp is not None:
            return float(rp[0][0]), float(rp[1][0])
        hold = getattr(self, "_rp_hold", None)
        if hold is not None:
            return float(hold[0]), float(hold[1])
        return 0.0, 0.0

    def attitude_ref_source(self) -> str:
        """``plan`` | ``hold_ramp`` | ``level`` — what rows 3:5 carry now."""
        plan = getattr(self, "_path_plan", None)
        if plan is not None and getattr(plan, "rp_ned", None) is not None:
            return "plan"
        if getattr(self, "_rp_hold", None) is not None:
            return "hold_ramp"
        return "level"

    def attitude_rate_source(self) -> str:
        """How the body-rate rows 9:11 were filled on THIS engagement:
        ``stitcher_rate_T_inv`` once a roll/pitch reference was ever
        carried — the stitcher's piecewise-constant ``rp_rate_ned`` (zero
        when absent) and the hold ramp's ``-sign * pq_max`` through
        T(phi, theta)^-1, NOT a finite difference (the sim's twin is
        ``fd_horizon_T_inv``: np.gradient over the sampled horizon) —
        else ``none`` (rows 9:11 are zero, the 4-DoF tile)."""
        return ("stitcher_rate_T_inv"
                if getattr(self, "_rp_tracked_any", False) else "none")

    def set_attitude_axes(self, enabled: bool, u_max_wire_nm=None) -> None:
        """The worker says at engage whether K/M actually leave the station
        (``engage.attitude_axes``), and what the wire can carry at most
        (``[cap_roll*roll_nm, cap_pitch*pitch_nm]``). Recorded in meta();
        it also arms the heave-trim attitude rotation (D13), which must not
        rotate a trim on a hull whose attitude nobody commands."""
        self._attitude_axes = bool(enabled)
        self._u_max_wire_nm = (None if u_max_wire_nm is None
                               else [float(v) for v in u_max_wire_nm])

    def _w_trim_now(self, eta) -> np.ndarray:
        """The heave trim this tick. Constant body-z (today) unless
        ``policy.heave_trim_attitude_rotated`` AND attitude axes are on: then
        the exact form, the NED-down net force rotated into the MEASURED
        body frame, ``w_z * [-sin(theta), cos(theta) sin(phi), cos(theta)
        cos(phi), 0, 0, 0]`` (= the docstring's ``delta * [sin, -cos sin,
        -cos cos]`` with ``delta = -w_z``). Level -> identical to the
        constant."""
        if not (getattr(self, "_heave_trim_rotated", False)
                and getattr(self, "_attitude_axes", False)):
            return self.w_trim
        phi, theta = float(eta[3]), float(eta[4])
        wz = float(self.w_trim[2])
        w = np.zeros(6)
        w[0] = -wz * np.sin(theta)
        w[1] = wz * np.cos(theta) * np.sin(phi)
        w[2] = wz * np.cos(theta) * np.cos(phi)
        return w

    # ------------------------------------------------------------------ control
    def step(self, eta_ned, nu_ned, nudot_ned, t: float):
        """One 20 Hz tick: EAOB (dobmpc) -> per-axis w_hat clip -> NMPC.
        Returns (u_ned (6,), info dict). ``t`` is the trajectory clock."""
        eta = np.asarray(eta_ned, float)
        nu = np.asarray(nu_ned, float)
        if self.eaob is None:
            self.eaob = self._EAOB(eta0=eta, nu0=nu, profile="perf")
        if self.dob:
            _eh, _nh, w = self.eaob.update(
                {"eta": eta, "nu": nu, "nudot": np.asarray(nudot_ned, float)},
                self._tau_ned_cmd)
            self.w_hat = np.clip(w, -self.w_clip, self.w_clip)
        else:
            self.w_hat = np.zeros(6)
        self._psi_ned_now = float(eta[5])
        self._apply_stage_weights()
        t0 = time.perf_counter()
        # dobmpc: the EAOB's (clipped) estimate alone. mpc / mpc_tuned: the
        # constant heave trim alone (w_hat is zero there). Never both.
        w_solver = self.w_hat if self.dob else self.w_hat + self._w_trim_now(eta)
        u = self.nmpc.solve(np.concatenate([eta, nu]), w_solver,
                            self._xref_ned(t))
        wall_ms = 1e3 * (time.perf_counter() - t0)
        self._decay_rp_hold()               # D5: after the solve that used it
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
            # what the solver was actually handed. Read by the tests only:
            # the CSV w0..w5 columns stay the EAOB estimate (w2 = 0 under
            # mpc), so a mpc CSV with and without the trim differ only in uZ
            # — meta `controller.heave_trim` is the record that tells them apart.
            "w_solver": np.asarray(w_solver, float).copy(),
            "status": self.last_status, "n_fail": self.n_fail,
            "nis": (float(self.eaob.last_nis) if self.eaob is not None else 0.0),
        }

    # ------------------------------------------------------- path-frame cost
    def set_path_cost(self, tune: dict | None) -> None:
        """Re-tune the along/cross split without rebuilding anything.

        The weights are a runtime ``cost_set``, so a sweep over q_along /
        q_cross costs ONE acados build for the whole sweep — which is the
        difference between a knob that gets swept and a knob that gets
        guessed at (rov_gui/tools/sweep_path_cost.py). Restores the baseline
        diagonal first, so a half-written sweep cannot leave the previous
        variant's weight in a stage the next one does not overwrite."""
        from .path_cost import PathFrameWeights

        self._restore_base_weights()
        P = self.P
        self._wt = PathFrameWeights(P.MPC_Q, P.MPC_R, P.MPC_QN, tune)

    #: State rows whose weight a ``NedPlan.w_stage`` mask scales: position
    #: (x, y, z) and body linear velocity (u, v, w) — the sim's state order
    #: ``[x y z, phi theta psi, u v w, p q r]`` (path_cost.IDX_*). Attitude,
    #: yaw and the rates keep their weight, so a masked stage still holds
    #: level and heading; R is never touched.
    W_STAGE_ROWS = (0, 1, 2, 6, 7, 8)
    #: State rows ``set_plan_cost_scale`` multiplies: POSITION only. Velocity,
    #: attitude, yaw, rates and R keep their weight, so the scaled cost still
    #: damps and levels — it only asks harder for the position the plan names.
    PLAN_SCALE_ROWS = (0, 1, 2)

    def set_plan_cost_scale(self, s: float) -> None:
        """Multiply the position weight of every stage while a streamed plan
        is installed (config ``policy.q_scale``; 1.0 = baseline).

        Exists because the plant the NMPC optimises has no thruster deadband:
        for the 8-10 cm error the policy leash allows it commands ~1 N, which
        the T200s turn into nothing (axis < ~0.096), so under mpc/mpc_tuned
        the vehicle sat still on 2026-09-07 while PID (6-7 N) moved. A runtime
        ``cost_set`` like the path-frame split, so no rebuild. The worker sets
        it at policy arm and puts 1.0 back at STOP / mode change; with no
        path plan installed the scale is dormant (station hold is unchanged).
        """
        s = float(s)
        if not np.isfinite(s) or s <= 0.0:
            raise ValueError(f"plan cost scale must be a positive finite number, got {s!r}")
        if s != 1.0 and self.solver_kind != "acados":
            # The scale is a per-stage cost_set; the IPOPT fallback has no
            # solver to write it into, so it would be recorded as flown and
            # never applied (safety review 2026-09-07) — refuse like the
            # tuned modes do.
            raise ValueError(f"plan cost scale {s:g} needs the acados solver "
                             f"(this controller runs {self.solver_kind})")
        if s != self._plan_q_scale:
            self._plan_q_scale = s
            self._plan_q_scale_flown = max(float(self._plan_q_scale_flown), s)
            # force a rewrite on the next tick even if nothing else changed
            self._w_tuned = True

    @property
    def plan_cost_scale(self) -> float:
        return float(self._plan_q_scale)

    def set_plan_along_scale(self, s) -> None:
        """``policy.along_scale``: the ALONG-track weight scale a path-frame
        (tuned) follower runs while a streamed policy plan is installed;
        ``None`` = the controller's own split (config ``mpc_tuned.along_scale``,
        0.25 — chosen for geometric path missions, where lagging the virtual
        target costs nothing). For a policy approach the along-track lag IS
        the task, and 0.25 leaves the surge command at 35 N/m·e + FF ≈ 3 N,
        inside the ESC deadband (0908_165517/mpc_165753: uX p10-p90
        2.5-3.5 N, 0 % of ticks above the 0.096 knee, 0.01 m in 45 s).

        Surge-only, which is why it exists beside :meth:`set_plan_cost_scale`:
        q_scale multiplies x, y AND z, so q_scale 4 also puts x4 on sway and
        heave (heave stiffness 55 -> 110 N/m against the STAB attitude loop's
        ±7 cm depth ripple). A runtime ``cost_set`` like the split itself —
        no rebuild. The worker sets it at policy arm and puts ``None`` back at
        STOP / mode change / disengage; ``reset()`` forgets it. A non-tuned
        follower records the value but has no split to change.
        """
        if s is not None:
            s = float(s)
            if not np.isfinite(s) or s <= 0.0:
                raise ValueError(f"plan along scale must be None or a positive "
                                 f"finite number, got {s!r}")
        if s == self._plan_along_scale:
            return
        if s is not None and self.tuned and self.solver_kind != "acados":
            raise ValueError(f"plan along scale {s:g} needs the acados solver "
                             f"(this controller runs {self.solver_kind})")
        self._plan_along_scale = s
        if s is not None:
            self._plan_along_scale_flown = s
        if not self.tuned:
            return                      # recorded; the baseline has no split
        if s is None:
            self.set_path_cost(self._wt_cfg_tune)
        else:
            tune = dict(self._wt.tune)
            tune["along_scale"] = float(s)
            tune["q_along"] = None      # the scale, not an absolute weight, wins
            self.set_path_cost(tune)
        self._w_tuned = True            # force a per-stage rewrite next tick

    @property
    def plan_along_scale(self):
        return self._plan_along_scale

    @classmethod
    def _scale_W(cls, W: np.ndarray, s: float) -> np.ndarray:
        """``W`` with the position block multiplied by ``s`` (rows/cols
        PLAN_SCALE_ROWS); symmetric and PSD stay intact."""
        out = np.array(W, dtype=float, copy=True)
        idx = list(cls.PLAN_SCALE_ROWS)
        out[np.ix_(idx, idx)] *= float(s)
        return out

    @classmethod
    def _mask_W(cls, W: np.ndarray, w: float) -> np.ndarray:
        """``W`` with the position/linear-velocity SUB-BLOCK scaled by ``w``.

        The sub-block (rows AND columns of :attr:`W_STAGE_ROWS`), not the rows
        alone: the tuned W carries off-diagonal terms inside the (x, y) and
        (u, v) pairs, and scaling rows only would leave an asymmetric W. On the
        diagonal baseline the two are the same thing. ``w = 1`` returns a
        copy equal to ``W``; ``w = 0`` zeroes the block (position and linear
        velocity free at that stage — acados accepts a positive
        SEMI-definite W; the yaw/attitude/rate rows and R keep the QP
        well-posed).
        """
        out = np.array(W, dtype=float, copy=True)
        idx = list(cls.W_STAGE_ROWS)
        out[np.ix_(idx, idx)] *= float(w)
        return out

    def _apply_stage_weights(self) -> None:
        """Write this tick's per-stage W: the path-frame rotation (``_tuned``
        modes, path_cost.py) and/or the plan's ``w_stage`` cost mask (v2 A10).

        Called every tick, before the solve. Ways this is a no-op, and each
        of them matters:
          * not a ``_tuned`` mode AND the plan carries no ``w_stage`` — the
            baseline diagonal is what was built;
          * no path plan (DP hold, the approach, the 10 s settle) — there is
            no path, so there is no along/cross to split and nothing to mask.
            Station keeping with an anisotropic weight would mean "hold this
            point, but only in one direction", which is not what station mode
            promises;
          * a tuned mode whose plan predates ``psi_path`` — fall back rather
            than guess (a mask alone still applies).
        In all of them the previously written weights are restored first, so
        a rotated or MASKED W can never outlive the mode, the mission or the
        plan that asked for it — a masked W left behind in a DP hold would be
        a station hold with no position weight on the far horizon.
        """
        plan = getattr(self, "_path_plan", None)
        psi = None if plan is None else getattr(plan, "psi_path", None)
        ws = None if plan is None else getattr(plan, "w_stage", None)
        rotate = bool(self.tuned and psi is not None)
        qs = float(self._plan_q_scale)
        scaled = qs != 1.0
        if plan is None or (not rotate and ws is None and not scaled):
            self._restore_base_weights()
            return
        solver = getattr(self.nmpc, "solver", None)
        if solver is None:                      # IPOPT fallback: nothing to set
            return
        N = int(self.nmpc.N)
        yaw = np.asarray(plan.yaw_ned, float).ravel()
        if rotate:
            psi = np.asarray(psi, float).ravel()
            if psi.size != N + 1:
                raise ValueError(f"path plan psi_path must be {N + 1} "
                                 f"stages, got {psi.size}")
        if ws is not None:
            ws = np.clip(np.asarray(ws, float).ravel(), 0.0, 1.0)
            if ws.size != N + 1:
                raise ValueError(f"path plan w_stage must be {N + 1} stages, "
                                 f"got {ws.size}")
        for k in range(N):
            W = (self._wt.stage_W(psi[k], yaw[k]) if rotate
                 else self._wt.W_base)
            if scaled:
                W = self._scale_W(W, qs)
            if ws is not None and ws[k] != 1.0:
                W = self._mask_W(W, ws[k])
            solver.cost_set(k, "W", W, api="new")
        We = (self._wt.terminal_W(psi[N], yaw[N]) if rotate
              else self._wt.We_base)
        if scaled:
            We = self._scale_W(We, qs)
        if ws is not None and ws[N] != 1.0:
            We = self._mask_W(We, ws[N])
        solver.cost_set(N, "W", We, api="new")
        self._w_tuned = True

    def _restore_base_weights(self) -> None:
        """Put the isotropic diagonal back. Idempotent and cheap when it is
        already there, because it runs on the untuned tick path. ``_w_tuned``
        means "a non-baseline W (rotated OR masked) is in the solver"."""
        if not self._w_tuned:
            return
        solver = getattr(getattr(self, "nmpc", None), "solver", None)
        if solver is None:
            self._w_tuned = False
            return
        N = int(self.nmpc.N)
        for k in range(N):
            solver.cost_set(k, "W", self._wt.W_base, api="new")
        solver.cost_set(N, "W", self._wt.We_base, api="new")
        self._w_tuned = False

    def note_applied(self, tau_ned_est) -> None:
        """The wrench the allocation believes the vehicle realized (axis caps
        applied; K/M as SENT — 0 unless ``engage.attitude_axes``, else the
        capped, slewed value). This is what the EAOB must see next tick —
        feeding it the raw solver output would count the cap/drop mismatch
        as disturbance. RECORD BOUNDARY: under the variant dobmpc's
        w_hat[3:5] books only the torque the wire could not carry."""
        self._tau_ned_cmd = np.asarray(tau_ned_est, float).copy()

    def reset(self) -> None:
        self._restore_base_weights()
        self._plan_q_scale = 1.0          # a fresh engagement never inherits a policy scale
        self._plan_q_scale_flown = 1.0
        if self._plan_along_scale is not None and self.tuned:
            self.set_path_cost(self._wt_cfg_tune)   # the config's own split is back
        self._plan_along_scale = None
        self._plan_along_scale_flown = None
        self.eaob = None
        self.w_hat = np.zeros(6)
        self._tau_ned_cmd = np.zeros(6)
        self._ref_traj = None
        self._path_plan = None
        self.scenario = None
        self.r_ref = 0.0
        self._rp_hold = None              # a fresh engagement starts level
        self._rp_tracked_any = False
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
        """EVERY constant this controller was flown with — the operator's
        counterpart to HwPid.meta() (2026-08-14). The stage weights are the
        MPC's equivalent of kp/kd/ki: they are what gets turned between runs,
        so a run whose meta omits them cannot be told apart from a run at
        different weights."""
        P = self.P
        m = {"type": self.mode, "solver": self.solver_kind,
             "ctrl_hz": float(self.cfg.ctrl_hz), "dt_s": float(P.DT_CTRL),
             "N": int(P.MPC_N), "horizon_s": round(P.MPC_N * P.DT_CTRL, 4),
             "u_max": [float(v) for v in P.U_MAX],
             "v_max_m_s": float(P.V_MAX),
             # x = [x y z, phi theta psi, u v w, p q r]
             "Q": [float(v) for v in P.MPC_Q],
             "QN": [float(v) for v in P.MPC_QN],
             "R": [float(v) for v in P.MPC_R],
             "state_order": "x = [x y z, phi theta psi, u v w, p q r] (NED/FRD)",
             "u_order": "u = [X Y Z K M N] (body wrench, N / N*m)",
             "w_hat_clip": [float(v) for v in self.w_clip],
             # RECORD BOUNDARY 2026-09-07: the constant heave trim (see
             # __init__). A run without this block flew the plant's own
             # -5.71 N sinking assumption as an up-force feedforward.
             "heave_trim": {
                 "vehicle_net_buoyancy_n": float(self.vehicle_net_buoyancy_n),
                 "model_net_buoyancy_n": float(P.NET_BUOYANCY),
                 "w_trim_ned_body": [float(v) for v in self.w_trim],
                 "applied": (not self.dob),
                 "channel": "acados parameter p (= w, NED body wrench), "
                            "constant over the horizon; dobmpc uses the "
                            "EAOB w_hat instead"},
             # THE COST SHAPE, not just its magnitudes. `cost_frame` is the
             # boundary between the tuned and the baseline records: a tuned
             # run's Q above is the isotropic baseline the split was derived
             # FROM, not what the solver ran, so pooling the two families on
             # the strength of the Q row alone would be wrong.
             "cost_frame": ("path (along/cross split)" if self.tuned
                            else "world (isotropic Q)"),
             "path_cost": (self._wt.meta() if self.tuned else None),
             # policy.q_scale in force at the time of writing (1.0 outside a
             # policy mission) — the trajectory block carries the armed value.
             "plan_q_scale": float(self._plan_q_scale),
             # the largest scale in force since reset() — what THIS CSV flew
             # (the value above is "now", 1.0 by the time STOP -> meta runs)
             "plan_q_scale_flown": float(getattr(self, "_plan_q_scale_flown", 1.0)),
             # policy.along_scale (path-frame followers): the along-track
             # scale in force now, and the one THIS CSV flew (None = the
             # controller's own split, i.e. path_cost.tune above).
             "plan_along_scale": getattr(self, "_plan_along_scale", None),
             "plan_along_scale_flown": getattr(self, "_plan_along_scale_flown", None),
             # RECORD BOUNDARY as of 2026-08-24: the sim keeps acados' IPOPT
             # recovery, this station refuses it (a lazy multi-second build
             # inside a 20 Hz tick), so the same solver behaves differently on
             # failure in the two places. A run that cannot say which is not
             # comparable with one that can.
             "ipopt_fallback": bool(getattr(self.nmpc, "_fallback_enabled",
                                            False)),
             "ipopt_fallback_off_reason": str(
                 getattr(self.nmpc, "_fallback_off_reason", "")),
             "rov_model": P.MODEL,
             # the model the config ASKED for; differs from rov_model above
             # only when dobmpc was already in memory under another
             # $ROV_MODEL (import_dobmpc warns, never raises) — then every
             # number in this record is the LOADED model's
             "rov_model_requested": str(getattr(self.cfg, "rov_model", P.MODEL)),
             "rov_model_mismatch": (str(getattr(self.cfg, "rov_model", P.MODEL))
                                    != str(P.MODEL)),
             "ref_preview": True,
             "path_reference": "shared spatial plan, full horizon",
             "mpc_state_source": "meas",
             "probe_ms": self.probe_ms, "build_s": round(self.build_s, 2),
             # median of PROBE_N warm solves (2026-09-08); the cold first
             # solve and the worst of the timed ones ride along
             "probe_ms_first": getattr(self, "probe_ms_first", None),
             "probe_ms_worst": getattr(self, "probe_ms_worst", None),
             "probe_n_solves": int(self.PROBE_N),
             "probe_runs": int(getattr(self, "probe_n", 0)),
             "eaob_sigma_overrides": self.sigma_applied,
             # {} = the solver flew marinegym's BlueROV.yaml damping. Non-empty
             # = the prediction model's drag was replaced with this vehicle's
             # (config plant:) — a RECORD BOUNDARY: uX/uY at a given error is
             # not comparable with a run that has {} here.
             "plant_overrides": self.plant_applied,
             # RECORD BOUNDARY 2026-09-26 (the 6-DoF variant). `tracked` =
             # did this engagement ever carry a roll/pitch reference;
             # `allocation.attitude` = did K/M leave the station (dobmpc's
             # w_hat[3:5] changes meaning with it). A run without these
             # blocks flew level with K/M dropped at the allocation.
             "attitude_ref": {
                 "tracked": bool(getattr(self, "_rp_tracked_any", False)),
                 "source": self.attitude_ref_source(),
                 # stitcher_rate_T_inv | none (see attitude_rate_source);
                 # the sim records fd_horizon_T_inv for its own rule
                 "rate_source": self.attitude_rate_source(),
                 "rp_max_rad": float(self._rp_max_rad),
                 "rp_reject_rad": float(self._rp_reject_rad),
                 "pq_max_rad_s": float(self._pq_max),
                 # W_STAGE_ROWS masks position/velocity only, so the hold
                 # tail keeps the endpoint attitude at full weight
                 "hold_tail": "endpoint",
                 # cap x gain per axis, or None when K/M never left the
                 # station; solver U_MAX[3:5] above is what the optimiser
                 # believed it had (the mismatch is recorded, not hidden)
                 "u_max_wire_nm": self._u_max_wire_nm},
             "allocation": {"attitude": bool(self._attitude_axes)}}
        m["heave_trim"]["attitude_rotated"] = bool(
            self._heave_trim_rotated and self._attitude_axes and not self.dob)
        # The EAOB is what mpc and dobmpc DIFFER by, so its tuning belongs in
        # the record whether or not one has been constructed yet (it is built
        # lazily, at the first tick of an engagement).
        m["eaob"] = {
            "active": bool(self.dob),
            "profile": "perf", "tau_dist_s": float(P.EAOB_TAU_DIST),
            "nis_gate": float(P.EAOB_NIS_GATE),
            "gate_on": bool(P.EAOB_GATE_ON),
            "sigma_pos": [float(v) for v in P.EAOB_SIG_POS],
            "sigma_ang_deg": [float(np.degrees(v)) for v in P.EAOB_SIG_ANG],
            "sigma_lvel": [float(v) for v in P.EAOB_SIG_LVEL],
            "sigma_avel": [float(v) for v in P.EAOB_SIG_AVEL],
            "sigma_acc": float(P.EAOB_SIG_ACC),
            "sigma_aacc": float(P.EAOB_SIG_AACC),
            "sigma_alloc": [float(v) for v in P.EAOB_SIG_ALLOC]}
        if self.eaob is not None:
            m["eaob"].update({"n_upd": int(self.eaob.n_upd),
                              "n_gated": int(self.eaob.n_gated)})
        return m
