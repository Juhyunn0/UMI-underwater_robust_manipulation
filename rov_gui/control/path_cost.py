#!/usr/bin/env python3
"""path_cost.py — split the tracking NMPC's position error into along-track
and cross-track, and weight the two differently.

WHY THIS EXISTS. The tracking MPC (``mpc`` / ``dobmpc``) penalises position
error in the WORLD frame with a diagonal ``Q``: ``Q[0]`` on north, ``Q[1]`` on
east, both 300. That is isotropic — a centimetre off the path costs exactly
what a centimetre behind on it costs — so when the reference turns a corner
the optimiser is free to trade one for the other, and it does: cutting the
corner buys a large along-track saving for a small cross-track price, which is
optimal for that cost and is the "smoothly turns early" behaviour the operator
wants suppressed.

WHAT IT DOES. Rotate the 2x2 position weight into the PATH frame at each
stage, so the cost reads

    q_along * (t_hat . e)^2  +  q_cross * (n_hat . e)^2

with ``t_hat`` the path tangent and ``n_hat`` its left normal. Because a
rotation is orthogonal this is EXACTLY a weight change, not a new cost type:

    ||e||^2_W   with   W = Rz(psi) diag(q_along, q_cross) Rz(psi)^T
                =  q_along * along^2 + q_cross * cross^2

so ``q_along == q_cross`` reproduces the isotropic diagonal to machine
precision (pinned by test_path_cost). That identity is the whole reason this
is implemented as a per-stage ``W`` rather than as a second solver: acados
takes ``cost_set(k, "W", W_k)`` at run time on the ALREADY GENERATED
``LINEAR_LS`` OCP, so ``mpc`` and ``mpc_tuned`` are the same compiled solver,
the same model, the same horizon, and differ in exactly one thing. There is no
second code generation and no second build wait.

WHICH ANGLE. The PATH's tangent (``NedPlan.psi_path``), never the vehicle's
heading. Under ``heading_follow: false`` — every mission flown so far — the
vehicle crabs: it holds one heading while the path turns 90 degrees under it,
so the two angles are unrelated. Deriving the tangent from the reference
VELOCITY instead would fail exactly at the corners this exists for, because
``speed_profile`` brakes the reference to a crawl there.

WHAT IT INTERACTS WITH, and this is not obvious. ``PathCursor`` already
bounds along-track error at ``path_lead_m`` (the leash), so lowering
``q_along`` does not let the reference run away — it cannot. What it changes
is how much authority the NMPC spends closing a gap that is bounded anyway.
On 2026-08-18 that leash was saturated on 83.6 % of engaged ticks
(sessions/low_level_controller_data/20260818/0818_143802/mpc_143938.csv), i.e.
the vehicle was already unable to keep up; tuning q_along DOWN in that regime
buys cross-track accuracy at the price of going even slower. Read the two
knobs together, not separately.

VELOCITY (optional, off by default). The same split can be applied to the
linear-velocity block, but that block lives in the reference BODY frame
(``HwDobMpc._xref_ned_plan`` writes ``Rk.T @ v``), so its rotation angle is
``psi_path - psi_body``, not ``psi_path``. Left off by default so an A/B
changes one thing.

Everything here is pure numpy and imports nothing from acados or dobmpc, so
the maths is testable without a solver build.
"""

from __future__ import annotations

import math

import numpy as np

# x = [x y z, phi theta psi, u v w, p q r]  (NED / FRD) — the sim's state order
IDX_POS_XY = (0, 1)
IDX_VEL_XY = (6, 7)

# Every key `mpc_tuned:` accepts. A typo here is the imu_dr failure mode: the
# run flies the BASELINE weights while its meta says "tuned", and the two
# CSVs are then indistinguishable from two runs of the same controller.
TUNE_KEYS = ("along_scale", "cross_scale", "q_along", "q_cross",
             "apply_terminal", "split_velocity",
             "v_along_scale", "v_cross_scale")

DEFAULT_TUNE = {
    # Multipliers on the isotropic baseline Q[0:2] (= 300 on the heavy).
    # 0.25 / 4.0 is a 16x anisotropy, which is a deliberate first guess and
    # NOT a measured optimum — sweep it with tools/sweep_path_cost.py.
    "along_scale": 0.25,
    "cross_scale": 4.0,
    # Absolute overrides. None = use the scales above. Setting either one
    # makes the run independent of whatever params.py's Q happens to be.
    "q_along": None,
    "q_cross": None,
    # Rotate the TERMINAL weight (QN) too. Off would leave the last stage
    # isotropic, which re-admits corner cutting at the end of the horizon —
    # the end of the horizon being exactly where the corner sits when it
    # matters. On by default; the flag exists to ablate that claim.
    "apply_terminal": True,
    # Split the linear-velocity weight the same way (see module docstring).
    "split_velocity": False,
    "v_along_scale": 1.0,
    "v_cross_scale": 1.0,
}


def path_frame_block(q_along: float, q_cross: float, psi: float) -> np.ndarray:
    """The 2x2 WORLD weight that equals ``diag(q_along, q_cross)`` in the
    frame whose x axis points along ``psi``.

        W = Rz(psi) @ diag(q_along, q_cross) @ Rz(psi).T

    Symmetric by construction, and positive definite whenever both weights
    are. Written out rather than assembled with matrix products because this
    runs 61 times per 50 ms tick.
    """
    c, s = math.cos(float(psi)), math.sin(float(psi))
    qa, qc = float(q_along), float(q_cross)
    cc, ss, cs = c * c, s * s, c * s
    off = (qa - qc) * cs
    return np.array([[qa * cc + qc * ss, off],
                     [off, qa * ss + qc * cc]], float)


def resolve_tune(raw: dict | None) -> dict:
    """Merge an ``mpc_tuned:`` block over the defaults, rejecting typos."""
    tune = dict(DEFAULT_TUNE)
    if not raw:
        return tune
    unknown = set(raw) - set(TUNE_KEYS)
    if unknown:
        raise ValueError(
            f"unknown mpc_tuned keys {sorted(unknown)}; "
            f"known: {sorted(TUNE_KEYS)}")
    tune.update(raw)
    for k in ("along_scale", "cross_scale", "v_along_scale", "v_cross_scale"):
        tune[k] = float(tune[k])
        if not tune[k] > 0.0:
            raise ValueError(f"mpc_tuned.{k} must be > 0, got {tune[k]}")
    for k in ("q_along", "q_cross"):
        if tune[k] is not None:
            tune[k] = float(tune[k])
            if not tune[k] > 0.0:
                raise ValueError(f"mpc_tuned.{k} must be > 0, got {tune[k]}")
    for k in ("apply_terminal", "split_velocity"):
        tune[k] = bool(tune[k])
    return tune


class PathFrameWeights:
    """Per-stage acados ``W`` / ``W_e`` for the along/cross split.

    Construct once from the solver's baseline ``Q``, ``R``, ``QN``; call
    :meth:`stage_W` / :meth:`terminal_W` with the path tangent at that stage.
    :attr:`W_base` / :attr:`We_base` are the untouched diagonals, which is
    what gets written back when the mode is switched off — leaving a rotated
    weight behind in a solver that is no longer running tuned mode would be a
    silent cross-mode contamination.
    """

    def __init__(self, Q, R, QN, tune: dict | None = None):
        self.Q = np.asarray(Q, float).copy()
        self.R = np.asarray(R, float).copy()
        self.QN = np.asarray(QN, float).copy()
        self.tune = resolve_tune(tune)
        self.nx = self.Q.size
        self.nu = self.R.size
        self.W_base = np.diag(np.concatenate([self.Q, self.R]))
        self.We_base = np.diag(self.QN)

        i, j = IDX_POS_XY
        # One isotropic baseline for the pair. They are equal on every shipped
        # model; averaging rather than picking one keeps a hand-edited
        # asymmetric Q from silently becoming "whichever axis I read first".
        self.q_xy = 0.5 * (float(self.Q[i]) + float(self.Q[j]))
        self.qn_xy = 0.5 * (float(self.QN[i]) + float(self.QN[j]))
        t = self.tune
        self.q_along = (t["q_along"] if t["q_along"] is not None
                        else t["along_scale"] * self.q_xy)
        self.q_cross = (t["q_cross"] if t["q_cross"] is not None
                        else t["cross_scale"] * self.q_xy)
        # The terminal weights follow the SAME ratio the stage weights use, so
        # QN != Q (if it ever is) stays a terminal-vs-stage decision instead of
        # quietly becoming a second anisotropy.
        scale_n = (self.qn_xy / self.q_xy) if self.q_xy > 0.0 else 1.0
        self.qn_along = self.q_along * scale_n
        self.qn_cross = self.q_cross * scale_n

        iv, jv = IDX_VEL_XY
        self.qv_xy = 0.5 * (float(self.Q[iv]) + float(self.Q[jv]))
        self.qv_along = t["v_along_scale"] * self.qv_xy
        self.qv_cross = t["v_cross_scale"] * self.qv_xy
        self.qnv_xy = 0.5 * (float(self.QN[iv]) + float(self.QN[jv]))
        self.qnv_along = t["v_along_scale"] * self.qnv_xy
        self.qnv_cross = t["v_cross_scale"] * self.qnv_xy

    # ------------------------------------------------------------------ ratio
    @property
    def anisotropy(self) -> float:
        """q_cross / q_along — the one number that says how hard this run
        prefers staying ON the line over staying ON schedule."""
        return float(self.q_cross / self.q_along)

    # ------------------------------------------------------------- the weights
    def _write_block(self, W, idx, block) -> None:
        (i, j) = idx
        W[i, i], W[j, j] = block[0, 0], block[1, 1]
        W[i, j] = W[j, i] = block[0, 1]

    def stage_W(self, psi_path: float, psi_body: float | None = None):
        """Running-cost weight (ny x ny, ny = nx + nu) at one stage."""
        W = self.W_base.copy()
        self._write_block(W, IDX_POS_XY,
                          path_frame_block(self.q_along, self.q_cross,
                                           psi_path))
        if self.tune["split_velocity"]:
            # The velocity block is expressed in the reference BODY frame, so
            # the path direction seen from there is psi_path - psi_body.
            d = float(psi_path) - float(psi_body or 0.0)
            self._write_block(W, IDX_VEL_XY,
                              path_frame_block(self.qv_along, self.qv_cross, d))
        return W

    def terminal_W(self, psi_path: float, psi_body: float | None = None):
        """Terminal-cost weight (nx x nx)."""
        W = self.We_base.copy()
        if not self.tune["apply_terminal"]:
            return W
        self._write_block(W, IDX_POS_XY,
                          path_frame_block(self.qn_along, self.qn_cross,
                                           psi_path))
        if self.tune["split_velocity"]:
            d = float(psi_path) - float(psi_body or 0.0)
            self._write_block(W, IDX_VEL_XY,
                              path_frame_block(self.qnv_along, self.qnv_cross,
                                               d))
        return W

    # ---------------------------------------------------------------- record
    def meta(self) -> dict:
        """What the run meta must carry: a tuned run whose weights are not
        written down cannot be told apart from a baseline one."""
        return {
            "q_xy_baseline": self.q_xy,
            "q_along": float(self.q_along),
            "q_cross": float(self.q_cross),
            "anisotropy_cross_over_along": round(self.anisotropy, 4),
            "qn_along": float(self.qn_along),
            "qn_cross": float(self.qn_cross),
            "apply_terminal": bool(self.tune["apply_terminal"]),
            "split_velocity": bool(self.tune["split_velocity"]),
            "qv_along": float(self.qv_along),
            "qv_cross": float(self.qv_cross),
            "angle_source": "NedPlan.psi_path (path tangent, not vehicle yaw)",
            "equivalent_cost": ("q_along*(t.e)^2 + q_cross*(n.e)^2, "
                                "W = Rz(psi) diag(q_along,q_cross) Rz(psi)^T"),
            "tune": dict(self.tune),
        }
