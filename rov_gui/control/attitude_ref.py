#!/usr/bin/env python3
"""attitude_ref.py — the roll/pitch REFERENCE layer of the 7d followers built on HwPid (LOW pid_7d, rl_7d; 2026-10-03).

HwDobMpc has carried this since the 6-DoF variant (2026-09-26, mpc_bridge.py ``_rp_hold`` / ``set_path_plan_ned`` T3 /
``ref_attitude_ned_at``); the PID and the RL follower consume the SAME worker plans (NedPlan, one stage), so they get the
SAME rules here, as a mixin placed BEFORE the follower class (``class HwPid7d(AttitudeRefMixin, HwPid)``):

  * a plan carrying ``rp_ned`` is validated (T3: shape (2, path_plan_steps), finite, |rp| <= ``policy.rp_reject_deg``,
    ``rp_rate_ned`` shape) — ValueError, which the worker's install path turns into a disengage, exactly as for the NMPC;
  * when a plan WITH rp is replaced by one without, by None, by a station target or by a placed shape, the reference
    does not STEP to level: it starts from the outgoing plan's stage-0 attitude and ramps to level at
    ``policy.pq_max_rad_s`` (design D5 — a kp x 20 deg torque step at every plan boundary otherwise);
  * ``ref_attitude_ned_at`` / ``attitude_ref_source`` / ``attitude_rate_source`` / ``set_attitude_axes`` — the surface
    the worker reads for the CSV ``rroll/rpitch`` columns, the cap-pinned interlock (iii) and the run meta. Without it
    the worker judges a LEVEL reference while the plan tilts (the record lie the 2026-10-03 survey warned about).

NED/FRD throughout: +roll = starboard down, +pitch = nose UP (allocation.py sign map). Nothing here commands anything:
a follower that never reads the reference behaves exactly as before, and with no rp ever installed every accessor
returns level (the 4-DoF path).
"""

from __future__ import annotations

import math

import numpy as np


def _cfg_get(block, key, default):
    """``block.get(key)`` with ``default`` for a missing OR null value (a bare namespace config resolves defaults)."""
    get = block.get if hasattr(block, "get") else (lambda k, d=None: d)
    v = get(key, None)
    return default if v is None else v


def body_rates_from_euler(phi, theta, phid, thetad, psid):
    """``(p, q, r) = T(phi, theta)^-1 (phid, thetad, psid)`` (Fossen eq. 2.28) — HwDobMpc.body_rates_from_euler."""
    sf, cf = math.sin(phi), math.cos(phi)
    st, ct = math.sin(theta), math.cos(theta)
    return (phid - psid * st,
            thetad * cf + psid * ct * sf,
            -thetad * sf + psid * ct * cf)


class AttitudeRefMixin:
    """Roll/pitch reference for a HwPid-family follower (one-stage plans). Call ``_att_init(cfg)`` in ``__init__``
    BEFORE the base class's ``reset()`` runs, and ``_att_decay()`` once per tick at the START of ``step()``, so the reference
    the worker reads after the tick (CSV rroll/rpitch, interlock (iii)) is the one the tick used (review 2026-10-03)."""

    def _att_init(self, cfg) -> None:
        pol = getattr(cfg, "policy", None) or {}
        self._pq_max = float(_cfg_get(pol, "pq_max_rad_s", 0.35))
        self._rp_reject_rad = math.radians(float(_cfg_get(pol, "rp_reject_deg", 30.0)))
        self._rp_max_rad = math.radians(float(_cfg_get(pol, "rp_max_deg", 20.0)))
        self._heave_trim_rotated = bool(_cfg_get(pol, "heave_trim_attitude_rotated", False))
        self._att_dt = 1.0 / float(cfg.ctrl_hz)
        self._rp_hold = None
        self._rp_tracked_any = False
        self._attitude_axes = False
        self._u_max_wire_nm = None

    # ------------------------------------------------------------------ plan / target installs
    def _att_validate_plan(self, plan) -> bool:
        """T3 for a plan's attitude; True when the plan carries rp (and it is valid), False when it carries none."""
        rp = getattr(plan, "rp_ned", None)
        if rp is None:
            return False
        steps = int(self.path_plan_steps)
        rp = np.asarray(rp, float)
        if rp.shape != (2, steps):
            raise ValueError(f"{self.mode} path plan rp_ned needs shape (2, {steps}), got {rp.shape}")
        if not np.all(np.isfinite(rp)):
            raise ValueError(f"{self.mode} path plan rp_ned must be finite")
        worst = float(np.max(np.abs(rp)))
        if worst > self._rp_reject_rad:
            raise ValueError(f"{self.mode} path plan attitude {math.degrees(worst):.1f} deg exceeds rp_reject "
                             f"{math.degrees(self._rp_reject_rad):.1f} deg")
        rpr = getattr(plan, "rp_rate_ned", None)
        if rpr is not None:
            rpr = np.asarray(rpr, float)
            if rpr.shape != (2, steps) or not np.all(np.isfinite(rpr)):
                raise ValueError(f"{self.mode} path plan rp_rate_ned needs a finite (2, {steps}) array, got {rpr.shape}")
        return True

    def _att_capture_hold(self) -> None:
        """The installed plan is about to go: if it carried an attitude, the hold reference STARTS there (D5)."""
        plan = getattr(self, "_path_plan", None)
        rp = None if plan is None else getattr(plan, "rp_ned", None)
        if rp is not None:
            self._rp_hold = np.asarray(rp, float)[:, 0].copy()

    def _set_rp_hold(self, rp) -> None:
        rp = np.asarray(rp, float).ravel()
        if rp.shape != (2,) or not np.all(np.isfinite(rp)):
            raise ValueError(f"rp_ref must be a finite (roll, pitch) pair, got {rp!r}")
        if abs(float(rp[0])) > self._rp_reject_rad or abs(float(rp[1])) > self._rp_reject_rad:
            raise ValueError(f"rp_ref {np.degrees(rp)} deg exceeds rp_reject {math.degrees(self._rp_reject_rad):.1f} deg")
        self._rp_hold = rp.copy() if np.any(rp) else None
        self._rp_tracked_any = True

    def set_path_plan_ned(self, plan) -> None:
        if plan is not None and self._att_validate_plan(plan):
            super().set_path_plan_ned(plan)          # the base checks p/v/yaw/r shapes first; flags only after success
            self._rp_tracked_any = True
            self._rp_hold = None                     # the plan owns the attitude now
            return
        self._att_capture_hold()
        super().set_path_plan_ned(plan)

    def set_target_ned(self, p_ned, yaw_ned, v_ned=None, r_ned=0.0, rp_ref=None):
        """``rp_ref`` = NED (roll, pitch) the hold starts from (then ramps to level) — HwDobMpc.set_target_ned's twin."""
        self._att_capture_hold()
        super().set_target_ned(p_ned, yaw_ned, v_ned=v_ned, r_ned=r_ned)
        if rp_ref is not None:
            self._set_rp_hold(rp_ref)

    def set_square_ned(self, *a, **kw):
        self._att_capture_hold()
        return super().set_square_ned(*a, **kw)

    def set_line_ned(self, *a, **kw):
        self._att_capture_hold()
        return super().set_line_ned(*a, **kw)

    def set_circle_ned(self, *a, **kw):
        self._att_capture_hold()
        return super().set_circle_ned(*a, **kw)

    # ------------------------------------------------------------------ the reference
    def _att_decay(self) -> None:
        """One tick of the D5 hold ramp: each axis moves toward 0 by at most pq_max * dt; None once both are there."""
        if self._rp_hold is None:
            return
        step = self._pq_max * self._att_dt
        rp = self._rp_hold
        self._rp_hold = np.sign(rp) * np.maximum(np.abs(rp) - step, 0.0)
        if not np.any(self._rp_hold):
            self._rp_hold = None

    def ref_attitude_ned_at(self, t: float | None = None) -> tuple[float, float]:
        """Current NED (roll, pitch) reference, rad: the installed plan's stage 0, else the hold ramp, else level."""
        plan = getattr(self, "_path_plan", None)
        rp = None if plan is None else getattr(plan, "rp_ned", None)
        if rp is not None:
            return float(rp[0][0]), float(rp[1][0])
        if self._rp_hold is not None:
            return float(self._rp_hold[0]), float(self._rp_hold[1])
        return 0.0, 0.0

    def ref_attitude_rate_ned_at(self, t: float | None = None) -> tuple[float, float]:
        """Euler (roll, pitch) rate of the reference: the plan's stage-0 ``rp_rate_ned`` (0 when absent), the hold
        ramp's -sign * pq_max while it runs, else 0 — the same rule HwDobMpc's rate rows use."""
        plan = getattr(self, "_path_plan", None)
        if plan is not None and getattr(plan, "rp_ned", None) is not None:
            rpr = getattr(plan, "rp_rate_ned", None)
            if rpr is None:
                return 0.0, 0.0
            return float(rpr[0][0]), float(rpr[1][0])
        if self._rp_hold is not None:
            s = np.sign(self._rp_hold)
            return float(-s[0] * self._pq_max), float(-s[1] * self._pq_max)
        return 0.0, 0.0

    def attitude_ref_source(self) -> str:
        """``plan`` | ``hold_ramp`` | ``level`` — what the attitude reference is now."""
        plan = getattr(self, "_path_plan", None)
        if plan is not None and getattr(plan, "rp_ned", None) is not None:
            return "plan"
        if self._rp_hold is not None:
            return "hold_ramp"
        return "level"

    def attitude_rate_source(self) -> str:
        """``stitcher_rate_T_inv`` once a roll/pitch reference was ever carried on this engagement, else ``none``."""
        return "stitcher_rate_T_inv" if self._rp_tracked_any else "none"

    def set_attitude_axes(self, enabled: bool, u_max_wire_nm=None) -> None:
        """The worker says at engage whether K/M leave the station, and what the wire can carry at most."""
        self._attitude_axes = bool(enabled)
        self._u_max_wire_nm = (None if u_max_wire_nm is None else [float(v) for v in u_max_wire_nm])

    def _att_reset(self) -> None:
        self._rp_hold = None                         # a fresh engagement starts level
        self._rp_tracked_any = False

    def _att_meta(self) -> dict:
        """meta controller.attitude_ref + allocation — the keys HwDobMpc.meta() writes (RECORD BOUNDARY 2026-09-26)."""
        return {"attitude_ref": {"tracked": bool(self._rp_tracked_any),
                                 "source": self.attitude_ref_source(),
                                 "rate_source": self.attitude_rate_source(),
                                 "rp_max_rad": float(self._rp_max_rad),
                                 "rp_reject_rad": float(self._rp_reject_rad),
                                 "pq_max_rad_s": float(self._pq_max),
                                 "hold_tail": "endpoint",
                                 "u_max_wire_nm": self._u_max_wire_nm},
                "allocation": {"attitude": bool(self._attitude_axes)}}
