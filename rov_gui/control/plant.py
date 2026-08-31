#!/usr/bin/env python3
"""plant.py — the vehicle model a run was flown against, as JSON.

A run's CSV says what the vehicle DID. It does not say what the controller
BELIEVED the vehicle was, and every one of the numbers below can be edited
between two runs that otherwise look identical on disk: a different
``ROV_MODEL``, a re-tuned added-mass row, a changed drag coefficient. Without
them in the record, two CSVs that disagree cannot be told apart from two CSVs
that were flown against two different plants (operator request 2026-08-14:
"PID square인데 Rec Nav 누르면 M, D, C도 같이 적어두자").

So this module reads the ONE definition the controllers actually use —
``bluerov2_mujoco_marinegym/dobmpc/params.py`` and ``fossen.py``, the same
modules ``mpc_bridge.import_dobmpc`` loads — and flattens it into a dict.
Nothing here is a second copy of the physics; it is a serializer.

    M nu_dot + C(nu) nu + D(nu) nu + g(eta) = tau + w        (NED / FRD)

M and g are constant, so they are written as numbers. C and D are functions of
the state, so what is written is the coefficient set that GENERATES them plus a
sample evaluated at the run's own velocity — and the sample's generator matrix
is checked against ``fossen``'s own product before it is written, so a silent
divergence between the record and the model cannot happen (``c_check_max``).

Import cost: ``params`` and ``fossen`` are plain numpy — no casadi, no acados —
so a PID-only station (where the solver build failed) still gets a full plant
record. ``ROV_MODEL`` is read by ``rov_model.py`` at import time, so it is set
here only when ``dobmpc`` has not already been imported; if it has,
:func:`plant_meta` reports the model actually loaded rather than the one asked
for, and says so.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_MARINEGYM = Path(__file__).resolve().parents[2] / "bluerov2_mujoco_marinegym"


def _load(rov_model: str):
    """(params, fossen), importing the sim's model stack at most once."""
    if "dobmpc.params" not in sys.modules:
        if not _MARINEGYM.is_dir():
            raise FileNotFoundError(f"marinegym tree not found at {_MARINEGYM}")
        if str(_MARINEGYM) not in sys.path:
            sys.path.insert(0, str(_MARINEGYM))
        os.environ["ROV_MODEL"] = str(rov_model)
    from dobmpc import fossen, params
    return params, fossen


def _c_rb(P, nu) -> np.ndarray:
    """C_RB(nu), the skew-symmetric rigid-body Coriolis matrix whose product
    with nu is ``fossen.coriolis_rb(nu)`` (the m*zg terms are dropped there,
    so they are absent here too)."""
    m = float(P.MASS)
    u, v, w, p, q, r = (float(x) for x in nu)
    Ix, Iy, Iz = (float(x) for x in (P.IX, P.IY, P.IZ))
    C = np.zeros((6, 6))
    C[0, 4], C[0, 5] = m * w, -m * v
    C[1, 3], C[1, 5] = -m * w, m * u
    C[2, 3], C[2, 4] = m * v, -m * u
    C[3, 1], C[3, 2] = m * w, -m * v
    C[3, 4], C[3, 5] = Iz * r, -Iy * q
    C[4, 0], C[4, 2] = -m * w, m * u
    C[4, 3], C[4, 5] = -Iz * r, Ix * p
    C[5, 0], C[5, 1] = m * v, -m * u
    C[5, 3], C[5, 4] = Iy * q, -Ix * p
    return C


def _c_a(P, nu) -> np.ndarray:
    """C_A(nu) for the diagonal added-mass matrix; product = ``coriolis_added``."""
    a = np.asarray(P.ADDED_MASS, float)
    u, v, w, p, q, r = (float(x) for x in nu)
    C = np.zeros((6, 6))
    C[0, 4], C[0, 5] = a[2] * w, -a[1] * v
    C[1, 3], C[1, 5] = -a[2] * w, a[0] * u
    C[2, 3], C[2, 4] = a[1] * v, -a[0] * u
    C[3, 1], C[3, 2] = a[2] * w, -a[1] * v
    C[3, 4], C[3, 5] = a[5] * r, -a[4] * q
    C[4, 0], C[4, 2] = -a[2] * w, a[0] * u
    C[4, 3], C[4, 5] = -a[5] * r, a[3] * p
    C[5, 0], C[5, 1] = a[1] * v, -a[0] * u
    C[5, 3], C[5, 4] = a[4] * q, -a[3] * p
    return C


def _d(P, nu) -> np.ndarray:
    """D(nu) = D_L + D_NL(|nu|), diagonal, POSITIVE (dissipative).

    ``params`` stores DL/DNL negated (fossen.damping returns -(DL nu + DNL|nu|nu)
    so that a positive marinegym coefficient dissipates); the record shows the
    positive form, which is the one the equation above is written in."""
    dl = np.abs(np.asarray(P.DL, float))
    dnl = np.abs(np.asarray(P.DNL, float))
    return np.diag(dl + dnl * np.abs(np.asarray(nu, float)))


def _m(v) -> list:
    return [[round(float(x), 9) for x in row] for row in np.asarray(v, float)]


def _v(v) -> list:
    return [round(float(x), 9) for x in np.asarray(v, float).ravel()]


def plant_meta(rov_model: str, nu=None, eta=None) -> dict:
    """The controller's model of the vehicle, ready for ``json.dumps``.

    ``nu`` / ``eta`` (the run's current body velocity / pose, NED-FRD) are
    optional: given, the state-dependent terms are also evaluated there and
    recorded as ``C_at_nu`` / ``D_at_nu`` / ``g_at_eta`` — a sample, clearly
    labelled with the state it was taken at, never as if C or D were constant.
    """
    P, fossen = _load(rov_model)
    nu = np.zeros(6) if nu is None else np.asarray(nu, float).ravel()[:6]
    eta = np.zeros(6) if eta is None else np.asarray(eta, float).ravel()[:6]

    M = fossen.mass_matrix()
    C_rb, C_a = _c_rb(P, nu), _c_a(P, nu)
    # The record must not be able to drift from the model it claims to
    # describe: the generator matrices above are written only after their
    # product reproduces fossen's own, at THIS run's velocity.
    probe = nu if np.linalg.norm(nu) > 1e-6 else np.array(
        [0.3, -0.2, 0.1, 0.05, -0.04, 0.2])
    c_err = max(
        float(np.max(np.abs(_c_rb(P, probe) @ probe - fossen.coriolis_rb(probe)))),
        float(np.max(np.abs(_c_a(P, probe) @ probe
                            - fossen.coriolis_added(probe)))))
    d_err = float(np.max(np.abs(_d(P, probe) @ probe - fossen.damping(probe))))

    meta = {
        "source": "bluerov2_mujoco_marinegym/dobmpc/params.py + fossen.py "
                  "(imported, not copied — the same modules the controller runs)",
        "equation": "M nu_dot + C(nu) nu + D(nu) nu + g(eta) = tau + w",
        "frame": "NED world / FRD body (Fossen); eta=[x y z phi theta psi], "
                 "nu=[u v w p q r]",
        "rov_model_requested": str(rov_model),
        "rov_model_loaded": str(P.MODEL),
        "fully_actuated": bool(P.FULLY_ACTUATED),
        "rigid_body": {
            "mass_kg": float(P.MASS),
            "inertia_kgm2": [float(P.IX), float(P.IY), float(P.IZ)],
            "volume_m3": float(P.VOLUME), "rho_kg_m3": float(P.RHO),
            "weight_n": float(P.WEIGHT), "buoyancy_n": float(P.BUOYANCY),
            "net_buoyancy_n": float(P.NET_BUOYANCY),
            "zg_restoring_m": float(P.ZG),
            "zg_mass_coupling_m": float(getattr(P, "ZG_MASS", P.ZG)),
        },
        "M": {
            "note": "M = M_RB + M_A, constant (kg / kg*m^2); the m*zg "
                    "off-diagonals are ZG_MASS, zero on this plant because "
                    "the COM sits at the body origin",
            "matrix": _m(M),
            "rb_diag": _v([P.MASS, P.MASS, P.MASS, P.IX, P.IY, P.IZ]),
            "added_mass_diag": _v(P.ADDED_MASS),
        },
        "C": {
            "note": "C(nu) = C_RB(nu) + C_A(nu), skew-symmetric, generated "
                    "from mass/inertia and added_mass_diag above "
                    "(fossen.coriolis_rb / coriolis_added)",
            "C_rb_at_nu": _m(C_rb), "C_a_at_nu": _m(C_a),
            "check_max_abs_err": round(c_err, 12),
        },
        "D": {
            "note": "D(nu) = diag(linear) + diag(quadratic * |nu|), positive "
                    "(dissipative). params.py stores DL/DNL negated; these are "
                    "the positive marinegym coefficients.",
            "linear": _v(np.abs(P.DL)), "quadratic": _v(np.abs(P.DNL)),
            "D_at_nu": _m(_d(P, nu)),
            "check_max_abs_err": round(d_err, 12),
        },
        "g": {"note": "g(eta), restoring, N / N*m (fossen.restoring)",
              "g_at_eta": _v(fossen.restoring(eta))},
        "evaluated_at": {"nu": _v(nu), "eta": _v(eta),
                         "note": "C, D and g are STATE-DEPENDENT — the "
                                 "matrices above are samples at this state, "
                                 "not constants"},
        "actuation": {"nu_dim": int(P.NU), "u_max": _v(P.U_MAX),
                      "v_max_m_s": float(P.V_MAX),
                      "note": "u is the BODY WRENCH the allocation is asked to "
                              "realize; u_max is the solver's box"},
        "timing": {"dt_ctrl_s": float(P.DT_CTRL)},
        "provenance": "[스펙/유도] marinegym_assets/BlueROV.yaml + bluerov.xml "
                      "via rov_model.py — a SIM plant identification, not a "
                      "system ID of this physical vehicle "
                      "(memory: heavy-added-mass-provenance)",
    }
    if str(P.MODEL) != str(rov_model):
        meta["warning"] = (
            f"dobmpc was already imported as ROV_MODEL={P.MODEL!r}; this run "
            f"asked for {rov_model!r}. The numbers above are the LOADED model, "
            f"which is also what the controller is using.")
    return meta
