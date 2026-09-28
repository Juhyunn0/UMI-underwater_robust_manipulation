#!/usr/bin/env python3
"""plan_stream.py — streamed reference-plan plumbing: safety filter, stitcher,
and replay-track helpers.

WHY THIS EXISTS. A plan SOURCE (today: one recorded handheld demo replayed
through this seam; later: a diffusion policy) emits reference-trajectory plans
at ~1 Hz, and a 20 Hz NMPC consumes a stitched, continuous reference. Three
hazards from hardware history shape everything here:

  * an UNBOUNDED FEEDFORWARD once caused a runaway, so every incoming plan is
    speed/accel/yaw-rate clamped BEFORE the controller ever sees it
    (``FilterLimits.v_max`` is a deployment-achievable cap, deliberately NOT
    the solver's U_MAX);
  * REFERENCE JUMPS excite the closed loop, so a new plan must anchor at the
    current active reference and is cosine-BLENDED in, never switched hard;
  * the GEOFENCE was removed from the station (see
    rov-gui-run-folder memory), so the workspace box gate here is the ONLY
    position-based protection left in the stack.

FRAME AND UNITS. World NED in the engage-datum frame, metres / rad / seconds,
same conventions as ``path_geometry.NedPlan``. Yaw is unwrapped internally —
callers may hand in wrapped yaw.

Everything here is pure numpy + stdlib: no Qt, no acados/casadi, no scipy —
same discipline as ``path_cost.py``, so the maths is testable without a
solver build or a display.

Margin keys written by :class:`PlanFilter` (positive = pass, by that much):
    "obs_age"   obs_max_age_s - (now - obs_t)                    [s]
    "anchor_m"  anchor_max_m  - |p_plan(now) - r_now|            [m]
    "jump_m"    jump_max_m    - max overlap deviation            [m]
    "speed"     v_max         - peak knot speed                  [m/s]
    "accel"     a_max         - peak knot acceleration           [m/s^2]
    "yaw_rate"  r_max         - peak knot yaw rate               [rad/s]
    "box_m"     min signed distance of any knot inside the box   [m]
    "need"      the time-dilation factor the kinematics WOULD need (<= 1
                means every cap was met as sent)
    "dilation"  (clip only) the uniform time-dilation factor alpha
    "yaw_jump"  yaw_jump_max_rad - max |yaw_new - yaw_cur| over the overlap
                (only when the limit is set)                     [rad]
  Only when the message CARRIES roll/pitch knots (``PlanMsg.rp``, the
  pos_rpy_width variant, 2026-09-26) -- a 4-DoF message never writes them:
    "rp_mag"    rp_reject_rad - max |rp|  (HARD reject, schema class) [rad]
    "rp_jump"   rp_jump_max_rad - max |rp_new - rp_cur| over the overlap
                (only when the limit is set and the current reference
                carries attitude)                                [rad]
    "rp_rate"   rp_rate_max - peak knot Euler rate (max over roll and
                pitch); joins ``need`` so the SAME dilation repairs it
                (only when rp_rate_max is set)                   [rad/s]
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

PLAN_STREAM_SCHEMA = 1

# A plan whose observation stamp lies further than this AHEAD of `now` was
# stamped on a different clock (spec v2 A1: a live source's obs_t is the
# monotonic now() clock and ONLY the consumer's intake converts it to the
# mission clock — a stamp that arrives here unconverted is in the future by
# the whole mission origin). Freshness already rejects the past; this is the
# other half of the domain check. 50 ms covers stamp-vs-tick jitter.
OBS_T_FUTURE_TOL_S = 0.05

# Sample spacing of the post-stitch blend preview (PlanStitcher.preview_install).
PREVIEW_DT_S = 0.05

# A speed violation up to this ratio is "minor": the plan's geometry is kept
# and its clock is dilated. Anything worse is a different plan than the source
# intended and is rejected outright.
CLIP_RATIO_MAX = 1.5

# load_replay_track refuses tracks with fewer valid pose rows than this — a
# handful of fixes cannot support smoothing, trimming, or a finite-diff speed.
MIN_FIX_COUNT = 10

_TWO_PI = 2.0 * math.pi


def _rz(a: float) -> np.ndarray:
    """Rotation by ``a`` about z (NED down axis), 3x3."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------
# messages
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PlanMsg:
    plan_id: int
    t0: float                 # time of first knot, caller's monotonic clock [s]
    dt: float                 # knot spacing [s]
    p_ned: np.ndarray         # (3, K) positions, engage-datum NED [m]
    yaw: np.ndarray           # (K,) [rad]; may be wrapped, unwrapped internally
    g: Optional[np.ndarray] = None   # (K,) gripper width [0,1], 0 = closed
    obs_t: Optional[float] = None    # timestamp of the conditioning observation
    arrival_t: float = 0.0
    # (2, K) [roll, pitch] rad, ABSOLUTE engage-datum NED body attitude,
    # wrapped; None = the consumer levels (every 4-DoF source). Set only by
    # the pos_rpy_width compose branch under policy.attitude_track and by a
    # replay track loaded with_rp (2026-09-26). Trailing so every positional
    # constructor in the codebase is untouched.
    rp: Optional[np.ndarray] = None

    @property
    def n_knots(self) -> int:
        return int(np.asarray(self.p_ned).shape[1])

    @property
    def t_end(self) -> float:
        return self.t0 + (self.n_knots - 1) * self.dt


# --------------------------------------------------------------------------
# safety filter
# --------------------------------------------------------------------------
@dataclass
class FilterLimits:
    v_max: float = 0.15       # [m/s] speed cap (deployment-achievable, NOT solver U_MAX)
    a_max: float = 0.30       # [m/s^2]
    r_max: float = 0.50       # [rad/s] yaw rate
    anchor_max_m: float = 0.30
    jump_max_m: float = 0.20  # max deviation from current reference over the overlap
    obs_max_age_s: float = 0.7
    box_ned_min: Optional[Tuple[float, float, float]] = None   # inclusive
    box_ned_max: Optional[Tuple[float, float, float]] = None
    reject_escalate: int = 3
    # Reject a message that carries NO obs_t. Off for the demo replay (its
    # plans are cut from a recording and have no observation); ON for a live
    # policy, whose plan without a stamp cannot be freshness-gated (v2 A1).
    require_obs_t: bool = False
    # Optional yaw-jump gate over the overlap window: max |yaw_new - yaw_cur|
    # (wrapped) against the CURRENT reference. None = off. A policy that
    # re-anchors its heading 30 degrees away passes every position gate and
    # still asks the stitcher for a yaw slew the blend cannot deliver (v2 A11).
    yaw_jump_max_rad: Optional[float] = None
    # Attitude gates (pos_rpy_width, 2026-09-26). ALL three act only on a
    # message whose ``rp`` is not None; a 4-DoF message yields the identical
    # margins dict and verdict as before they existed. None = off.
    #   rp_reject_rad  HARD reject (schema class, D3 T2) when max |rp| exceeds
    #                  it -- an attitude the hull must never be asked for.
    #   rp_rate_max    peak knot Euler rate [rad/s]; joins ``need`` so the one
    #                  uniform time dilation repairs it like the yaw rate.
    #   rp_jump_max_rad  max |rp_new - rp_cur| over the overlap vs the
    #                  CURRENT attitude reference (``cur_sample_att``); a
    #                  dynamics/continuity gate (_soft).
    rp_reject_rad: Optional[float] = None
    rp_rate_max: Optional[float] = None
    rp_jump_max_rad: Optional[float] = None
    # How far a MINOR kinematic violation may be repaired by time dilation
    # before the plan is a different plan and is rejected. 1.5x is the replay
    # band (CLIP_RATIO_MAX). A live policy trained on handheld demos that
    # move ~1.5-2x faster than this vehicle [측정: dp_policy_offline.py gate,
    # /home/bdml/Desktop/data collection/dataset_depth.zarr.zip — GT chunks
    # need p50 1.50 / p90 3.44 at v_max 0.08, a_max 0.20 on the 0.2 s grid]
    # would otherwise be refused most of the time; the policy block widens
    # this (policy.clip_ratio_max) and the caps above still bound the
    # vehicle — dilation keeps the geometry and only stretches the clock.
    clip_ratio_max: float = CLIP_RATIO_MAX
    #: THE DIVISION OF LABOUR (2026-09-02, operator decision). True = the plan
    #: doc's architecture taken literally: "policy는 embodiment-무관한 기하만
    #: 내고, 기체 dynamics/제약은 MPC follower가 담당"
    #: (docs/DP_TRAJECTORY_PLAN.ko.md §0). The NMPC already carries the
    #: actuator limits and optimises over a horizon, so it — not a
    #: finite-difference gate on six knots — is what decides whether a
    #: reference is trackable.
    #:
    #: What it changes: the DYNAMICS and CONTINUITY gates stop REJECTING.
    #: They still run, still compute every margin and still write their
    #: sentence into ``reasons``, because the attribution record is the whole
    #: point of this file — the plan is simply no longer thrown away for
    #: failing them. Kinematic dilation still happens (it preserves geometry
    #: and only re-times it for this embodiment, which IS the adaptation the
    #: division of labour asks for) but is no longer capped by
    #: ``clip_ratio_max``: dilating further makes the reference SLOWER, never
    #: more demanding.
    #:
    #: What it does NOT change — these are not dynamics, and stay hard:
    #:   * schema / finiteness / monotone knot times (a NaN is not geometry),
    #:   * freshness (a stale plan describes a world that has moved),
    #:   * the workspace box (the only position-based protection left after
    #:     the geofence was removed 2026-08-14),
    #:   * ``div_max_m`` in MpcWorker — the vehicle-vs-reference divergence
    #:     guard, which is what actually catches "the follower cannot keep
    #:     up". Handing dynamics to the follower makes that guard MORE
    #:     important, not less, so it is deliberately untouched.
    #: A RECORD BOUNDARY: runs with this on and off are not comparable, and
    #: the run meta carries it.
    follower_owns_dynamics: bool = False
    #: Does the workspace box REJECT, or only measure? True everywhere a
    #: wrench can leave the station — the box is the only position-based
    #: protection left after the geofence was removed (2026-08-14), and
    #: ``follower_owns_dynamics`` deliberately does not touch it.
    #:
    #: False ONLY in POLICY OBSERVE (--policy-observe), where the controller
    #: output is muted and the pilot flies by hand. There the box protects
    #: nothing — there is no wrench to bound — while REJECTING everything the
    #: moment the pilot leaves the +-1 m box around the engage datum, which is
    #: the whole point of that run. And a rejected plan draws as a faded
    #: dashed line, i.e. the screen would say "the network stopped
    #: predicting" (the exact misreading ``_emit_policy_plan_viz_refused``
    #: exists to prevent) about a vehicle that simply flew 1.2 m.
    #:
    #: ``box_m`` is computed and the sentence recorded either way, so
    #: plans.jsonl still answers "was this knot inside the box".
    box_enforced: bool = True


@dataclass
class Verdict:
    status: str                       # "accept" | "clip" | "reject"
    plan: Optional[PlanMsg]           # time-dilated copy on "clip"; None on reject
    margins: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    consec_rejects: int = 0
    escalate: bool = False


class PlanFilter:
    """Gates every incoming :class:`PlanMsg` before it may reach the stitcher.

    Check order (each gate rejects on its own; later gates are not evaluated
    after a reject, so ``reasons`` names the FIRST failure):
      1. schema / finiteness / monotone knot times,
      2. freshness (observation age vs ``obs_max_age_s``; skipped when the
         message carries no ``obs_t`` — the demo-replay source has none),
      3. anchor gate: the plan evaluated AT ``now`` must sit within
         ``anchor_max_m`` of the current ACTIVE reference pose,
      4. overlap jump gate: max deviation from the current reference over the
         whole overlap window (a plan can pass the anchor at knot 0 and still
         diverge wildly at knot 5 — this is the gate that catches it),
      5. kinematics on the knots (finite-diff speed / accel / yaw rate); a
         MINOR speed violation (<= ``CLIP_RATIO_MAX``x) is repaired by uniform
         time dilation t' = alpha t and returned as status "clip",
      6. workspace box — the ONLY position-based protection since the
         geofence was removed; any knot outside rejects.
    """

    def __init__(self, limits: FilterLimits):
        self.limits = limits
        self._consec_rejects = 0

    def reset(self) -> None:
        """Clear the consecutive-reject counter (e.g. after operator review)."""
        self._consec_rejects = 0

    # -- helpers -----------------------------------------------------------
    def _reject(self, margins: Dict[str, float],
                reasons: List[str]) -> Verdict:
        self._consec_rejects += 1
        return Verdict(status="reject", plan=None, margins=margins,
                       reasons=reasons,
                       consec_rejects=self._consec_rejects,
                       escalate=(self._consec_rejects
                                 >= self.limits.reject_escalate))

    def _pass(self, status: str, plan: PlanMsg, margins: Dict[str, float],
              reasons: List[str]) -> Verdict:
        self._consec_rejects = 0
        return Verdict(status=status, plan=plan, margins=margins,
                       reasons=reasons, consec_rejects=0, escalate=False)

    def _soft(self, margins: Dict[str, float], reasons: List[str],
              sentence: str) -> Optional[Verdict]:
        """A DYNAMICS/CONTINUITY failure. Rejects, unless the follower owns
        dynamics — then it is recorded and waved through.

        Returns a reject Verdict to hand straight back, or None to continue.
        The sentence is appended either way: what the gate SAW is the record,
        and losing it would make the two modes impossible to compare after the
        fact, which is the one thing this file exists to prevent.
        """
        if self.limits.follower_owns_dynamics:
            reasons.append(sentence + "  [not enforced: the NMPC follower "
                                      "owns dynamics]")
            return None
        reasons.append(sentence)
        return self._reject(margins, reasons)

    def reject_external(self, margins: Dict[str, float],
                        reasons: List[str]) -> Verdict:
        """A reject decided OUTSIDE the gate chain that must still count as a
        geometric strike — the post-stitch blend gate (v2 A11) lives in the
        consumer because it needs the stitcher's candidate blend, but a plan
        that fails it is as bad as one that failed the jump gate here, and
        three of them in a row must escalate the same way."""
        return self._reject(dict(margins), list(reasons))

    # -- the gate chain ----------------------------------------------------
    def evaluate(self, msg: PlanMsg, *,
                 r_now: Tuple[np.ndarray, float],
                 cur_sample: Optional[Callable[[np.ndarray], tuple]] = None,
                 now: float,
                 cur_sample_att: Optional[Callable[[np.ndarray], Optional[tuple]]] = None
                 ) -> Verdict:
        """``cur_sample_att`` (``PlanStitcher.sample_att``) is consulted only
        for the rp_jump gate of a message that carries ``rp``; it may return
        None (no installed plan ever carried attitude), which skips that
        gate. Everything attitude-related lives under ``if rp is not None``
        so a 4-DoF message runs exactly the pre-2026-09-26 chain."""
        lim = self.limits
        margins: Dict[str, float] = {}
        reasons: List[str] = []
        rp = getattr(msg, "rp", None)

        # 1. schema / finite / monotone -----------------------------------
        p = np.asarray(msg.p_ned, dtype=float)
        yaw = np.asarray(msg.yaw, dtype=float)
        if p.ndim != 2 or p.shape[0] != 3 or p.shape[1] < 2:
            reasons.append(f"schema: p_ned must be (3, K>=2), got {p.shape}.")
            return self._reject(margins, reasons)
        K = p.shape[1]
        if yaw.shape != (K,):
            reasons.append(f"schema: yaw must be ({K},), got {yaw.shape}.")
            return self._reject(margins, reasons)
        if not (np.isfinite(msg.dt) and msg.dt > 0.0
                and np.isfinite(msg.t0)):
            reasons.append("schema: dt must be a positive finite number and "
                           "t0 finite (knot times must be monotone).")
            return self._reject(margins, reasons)
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(yaw))):
            reasons.append("schema: non-finite value in p_ned or yaw.")
            return self._reject(margins, reasons)
        if msg.g is not None:
            g = np.asarray(msg.g, dtype=float)
            if g.shape != (K,) or not np.all(np.isfinite(g)) \
                    or g.min() < -1e-9 or g.max() > 1.0 + 1e-9:
                reasons.append("schema: g must be a finite (K,) array in "
                               "[0, 1].")
                return self._reject(margins, reasons)
        if rp is not None:
            rp = np.asarray(rp, dtype=float)
            if rp.shape != (2, K) or not np.all(np.isfinite(rp)):
                reasons.append(f"schema: rp must be a finite (2, {K}) array "
                               f"[roll, pitch] rad, got {rp.shape}.")
                return self._reject(margins, reasons)
            if lim.rp_reject_rad is not None:
                rp_abs = float(np.abs(rp).max())
                margins["rp_mag"] = float(lim.rp_reject_rad) - rp_abs
                if rp_abs > float(lim.rp_reject_rad):
                    # HARD (schema class): not a dynamics question, the hull
                    # must never be asked for this attitude (D3 tier T2).
                    reasons.append(
                        f"attitude: plan asks for |roll/pitch| up to "
                        f"{math.degrees(rp_abs):.1f} deg (reject limit "
                        f"{math.degrees(float(lim.rp_reject_rad)):.1f} deg).")
                    return self._reject(margins, reasons)
        yaw_u = np.unwrap(yaw)
        t_knots = msg.t0 + np.arange(K) * msg.dt

        # 2. freshness -----------------------------------------------------
        if msg.obs_t is None and lim.require_obs_t:
            reasons.append("no observation stamp: this source must stamp "
                           "every plan with obs_t (require_obs_t).")
            return self._reject(margins, reasons)
        if msg.obs_t is not None:
            obs_t = float(msg.obs_t)
            if not math.isfinite(obs_t):
                reasons.append("schema: obs_t is not finite.")
                return self._reject(margins, reasons)
            age = now - obs_t
            margins["obs_age"] = lim.obs_max_age_s - age
            if age < -OBS_T_FUTURE_TOL_S:
                # The past is caught by the age gate; the FUTURE means the
                # stamp is on another clock (an unconverted monotonic stamp
                # is ahead of the mission clock by the mission origin).
                reasons.append(
                    f"observation stamp {-age:.2f} s in the FUTURE of now — "
                    f"obs_t is on a different clock domain than the plan "
                    f"clock (tolerance {OBS_T_FUTURE_TOL_S:.2f} s).")
                return self._reject(margins, reasons)
            if age > lim.obs_max_age_s:
                reasons.append(
                    f"stale observation: plan conditioned on data "
                    f"{age:.2f} s old (limit {lim.obs_max_age_s:.2f} s).")
                return self._reject(margins, reasons)

        # 3. anchor gate ---------------------------------------------------
        p_ref, _yaw_ref = r_now
        p_ref = np.asarray(p_ref, dtype=float).reshape(3)
        t_at = min(max(now, t_knots[0]), t_knots[-1])
        p_at_now = np.array([np.interp(t_at, t_knots, p[i]) for i in range(3)])
        d_anchor = float(np.linalg.norm(p_at_now - p_ref))
        margins["anchor_m"] = lim.anchor_max_m - d_anchor
        if d_anchor > lim.anchor_max_m:
            v = self._soft(margins, reasons,
                           f"anchor gate: plan at t=now sits {d_anchor:.3f} m "
                           f"from the current reference "
                           f"(limit {lim.anchor_max_m:.3f} m).")
            if v is not None:
                return v

        # 4. overlap jump gate --------------------------------------------
        if cur_sample is not None:
            t_lo = max(now, t_knots[0])
            ts = np.concatenate(([t_lo], t_knots[t_knots > t_lo]))
            if ts.size >= 1 and ts[-1] >= t_lo:
                p_new = np.vstack(
                    [np.interp(ts, t_knots, p[i]) for i in range(3)])
                cur = cur_sample(ts)
                p_cur = np.asarray(cur[0], dtype=float)
                dev = float(np.max(np.linalg.norm(p_new - p_cur, axis=0)))
                margins["jump_m"] = lim.jump_max_m - dev
                if dev > lim.jump_max_m:
                    v = self._soft(
                        margins, reasons,
                        f"overlap jump: plan deviates up to {dev:.3f} m from "
                        f"the current reference over the overlap window "
                        f"(limit {lim.jump_max_m:.3f} m).")
                    if v is not None:
                        return v
                if lim.yaw_jump_max_rad is not None:
                    yaw_new = np.interp(ts, t_knots, yaw_u)
                    yaw_cur = np.asarray(cur[1], dtype=float)
                    dy = yaw_new - yaw_cur
                    dy = np.arctan2(np.sin(dy), np.cos(dy))     # wrapped
                    yaw_dev = float(np.max(np.abs(dy)))
                    margins["yaw_jump"] = float(lim.yaw_jump_max_rad) - yaw_dev
                    if yaw_dev > float(lim.yaw_jump_max_rad):
                        v = self._soft(
                            margins, reasons,
                            f"yaw jump: plan heading deviates up to "
                            f"{math.degrees(yaw_dev):.1f} deg from the current "
                            f"reference over the overlap window (limit "
                            f"{math.degrees(float(lim.yaw_jump_max_rad)):.1f} "
                            f"deg).")
                        if v is not None:
                            return v
                if (rp is not None and lim.rp_jump_max_rad is not None
                        and cur_sample_att is not None):
                    att = cur_sample_att(ts)
                    if att is not None:
                        rp_new = np.vstack(
                            [np.interp(ts, t_knots, rp[i]) for i in range(2)])
                        rp_cur = np.asarray(att[0], dtype=float).reshape(2, -1)
                        drp = rp_new - rp_cur
                        drp = np.arctan2(np.sin(drp), np.cos(drp))  # wrapped
                        rp_dev = float(np.max(np.abs(drp)))
                        margins["rp_jump"] = float(lim.rp_jump_max_rad) - rp_dev
                        if rp_dev > float(lim.rp_jump_max_rad):
                            v = self._soft(
                                margins, reasons,
                                f"attitude jump: plan roll/pitch deviates up "
                                f"to {math.degrees(rp_dev):.1f} deg from the "
                                f"current attitude reference over the overlap "
                                f"window (limit "
                                f"{math.degrees(float(lim.rp_jump_max_rad)):.1f}"
                                f" deg).")
                            if v is not None:
                                return v

        # 5. kinematics on the knots --------------------------------------
        v = np.diff(p, axis=1) / msg.dt                     # (3, K-1)
        speed = np.linalg.norm(v, axis=0)
        v_peak = float(speed.max()) if speed.size else 0.0
        a_peak = 0.0
        if K >= 3:
            a = np.diff(v, axis=1) / msg.dt                 # (3, K-2)
            a_peak = float(np.linalg.norm(a, axis=0).max())
        r_knots = np.diff(yaw_u) / msg.dt
        r_peak = float(np.abs(r_knots).max()) if r_knots.size else 0.0
        margins["speed"] = lim.v_max - v_peak
        margins["accel"] = lim.a_max - a_peak
        margins["yaw_rate"] = lim.r_max - r_peak
        rp_rate_peak = 0.0
        rp_rate_max = None
        if rp is not None and lim.rp_rate_max is not None:
            rp_rate_max = float(lim.rp_rate_max)
            if K >= 2:
                rp_rate_peak = float(np.abs(np.diff(rp, axis=1) / msg.dt).max())
            margins["rp_rate"] = rp_rate_max - rp_rate_peak

        # ONE uniform time dilation repairs all three at once: slowing the
        # clock by alpha scales speed by 1/alpha, acceleration by 1/alpha^2
        # and yaw rate by 1/alpha — so the alpha that fixes the worst
        # violation fixes the rest for free. A human demo's corners live in
        # the ACCEL term (a 90-degree turn at constant speed is an accel
        # spike, not a speed one), which is why dilation must cover it: the
        # first synthetic L-demo through this filter was rejected outright
        # for 0.302 vs 0.300 m/s^2 (2026-08-30) when a 1 % slowdown would
        # have repaired it.
        status = "accept"
        out = msg
        need = max(
            (v_peak / lim.v_max) if lim.v_max > 0.0 else 1.0,
            math.sqrt(a_peak / lim.a_max) if lim.a_max > 0.0 else 1.0,
            (r_peak / lim.r_max) if lim.r_max > 0.0 else 1.0)
        if rp_rate_max is not None:
            # Attitude Euler rate scales by 1/alpha exactly like the yaw
            # rate, so the same uniform dilation repairs it (D3 tier T2).
            need = max(need, (rp_rate_peak / rp_rate_max) if rp_rate_max > 0.0
                       else 1.0)
        margins["need"] = float(need)
        clip_max = float(getattr(lim, "clip_ratio_max", CLIP_RATIO_MAX))
        if need > 1.0:
            if need > clip_max and not lim.follower_owns_dynamics:
                reasons.append(
                    f"kinematics: repairing this plan needs a {need:.2f}x "
                    f"time dilation (peak speed {v_peak:.3f}/{lim.v_max:.3f} "
                    f"m/s, accel {a_peak:.3f}/{lim.a_max:.3f} m/s^2, yaw "
                    f"rate {r_peak:.3f}/{lim.r_max:.3f} rad/s) — beyond the "
                    f"{clip_max:g}x clip band, rejecting.")
                return self._reject(margins, reasons)
            if need > clip_max:
                # Dilate all the way instead of refusing. More dilation is a
                # SLOWER reference along the same knots, never a more
                # demanding one, so there is no band beyond which it becomes
                # unsafe — only one beyond which it stops resembling what the
                # policy asked for, and that is recorded rather than enforced.
                reasons.append(
                    f"kinematics: needed {need:.2f}x, past the "
                    f"{clip_max:g}x clip band — dilated anyway "
                    f"[not enforced: the NMPC follower owns dynamics].")
            # Minor: keep the geometry, dilate the clock uniformly about t0.
            alpha = need
            margins["speed"] = lim.v_max - v_peak / alpha
            margins["accel"] = lim.a_max - a_peak / (alpha * alpha)
            margins["yaw_rate"] = lim.r_max - r_peak / alpha
            if rp_rate_max is not None:
                margins["rp_rate"] = rp_rate_max - rp_rate_peak / alpha
            margins["dilation"] = alpha
            out = dataclasses.replace(msg, dt=msg.dt * alpha)
            status = "clip"
            reasons.append(
                f"kinematics: time-dilated by alpha={alpha:.3f} to meet the "
                f"caps (peaks: speed {v_peak:.3f} m/s, accel {a_peak:.3f} "
                f"m/s^2, yaw rate {r_peak:.3f} rad/s; geometry unchanged).")

        # 6. workspace box (the ONLY position-based protection) -----------
        if lim.box_ned_min is not None and lim.box_ned_max is not None:
            bmin = np.asarray(lim.box_ned_min, dtype=float).reshape(3, 1)
            bmax = np.asarray(lim.box_ned_max, dtype=float).reshape(3, 1)
            inside = np.minimum(p - bmin, bmax - p)         # (3, K), signed
            box_m = float(inside.min())
            margins["box_m"] = box_m
            if box_m < 0.0:
                if not lim.box_enforced:
                    # MEASURED, NOT ENFORCED (POLICY OBSERVE). Said out loud
                    # in the same shape as _soft's sentence so a reader of
                    # plans.jsonl sees the box was left and sees why nothing
                    # was done about it. There is no wrench to bound here.
                    reasons.append(
                        f"workspace box: a knot leaves the box by "
                        f"{-box_m:.3f} m  [not enforced: POLICY OBSERVE — "
                        f"the controller output is muted, so the box "
                        f"protects nothing].")
                else:
                    reasons.append(
                        f"workspace box: a knot leaves the box by "
                        f"{-box_m:.3f} m — rejecting (the box is the only "
                        f"position-based protection; the geofence was "
                        f"removed).")
                    return self._reject(margins, reasons)

        return self._pass(status, out, margins, reasons)


# --------------------------------------------------------------------------
# stitcher
# --------------------------------------------------------------------------
class _PlanRef:
    """One installed plan as a sampleable reference: piecewise-linear p /
    yaw / g over the knots, finite-diff (piecewise-constant) v / r, endpoint
    hold beyond the last knot (p = end, v = 0, r = 0, yaw = end, g = end) and
    symmetrically before the first knot.

    Attitude (2026-09-26): ``rp`` (2, K) is the message's own roll/pitch
    knots, else ``rp_default`` (the stitcher's last flown attitude)
    broadcast over the knots, else None -- a stream that never carried
    attitude has ``sample_att() is None`` and the consumer levels. Roll and
    pitch are NOT unwrapped (bounded far below pi by the compose clip and
    the filter's rp_reject gate)."""

    def __init__(self, msg: PlanMsg, g_default: float,
                 rp_default: Optional[np.ndarray] = None):
        p = np.asarray(msg.p_ned, dtype=float)
        K = p.shape[1]
        self.plan_id = int(msg.plan_id)
        self.t = msg.t0 + np.arange(K) * float(msg.dt)
        self.p = p.copy()
        self.yaw = np.unwrap(np.asarray(msg.yaw, dtype=float))
        if msg.g is not None:
            self.g = np.asarray(msg.g, dtype=float).copy()
        else:
            self.g = np.full(K, float(g_default))
        if K >= 2:
            self._vseg = np.diff(self.p, axis=1) / float(msg.dt)
            self._rseg = np.diff(self.yaw) / float(msg.dt)
        else:
            self._vseg = np.zeros((3, 1))
            self._rseg = np.zeros(1)
        rp_msg = getattr(msg, "rp", None)
        self.rp: Optional[np.ndarray] = None
        self._rpseg = np.zeros((2, 1))
        if rp_msg is not None:
            self.rp = np.asarray(rp_msg, dtype=float).reshape(2, K).copy()
        elif rp_default is not None:
            self.rp = np.repeat(np.asarray(rp_default, dtype=float)
                                .reshape(2, 1), K, axis=1)
        if self.rp is not None and K >= 2:
            self._rpseg = np.diff(self.rp, axis=1) / float(msg.dt)

    @property
    def t_end(self) -> float:
        return float(self.t[-1])

    def sample(self, ts: np.ndarray) -> tuple:
        ts = np.atleast_1d(np.asarray(ts, dtype=float))
        p = np.vstack([np.interp(ts, self.t, self.p[i]) for i in range(3)])
        yaw = np.interp(ts, self.t, self.yaw)
        g = np.interp(ts, self.t, self.g)
        if self.t.size >= 2:
            idx = np.clip(np.searchsorted(self.t, ts, side="right") - 1,
                          0, self.t.size - 2)
            inside = (ts >= self.t[0]) & (ts < self.t[-1])
            v = self._vseg[:, idx] * inside
            r = self._rseg[idx] * inside
        else:
            v = np.zeros((3, ts.size))
            r = np.zeros(ts.size)
        return p, yaw, v, r, g

    def sample_att(self, ts: np.ndarray) -> Optional[tuple]:
        """(rp (2, n), rp_rate (2, n)) at ``ts`` -- linear roll/pitch,
        piecewise-constant Euler rate with the same ``inside`` mask as
        :meth:`sample` (endpoint hold, rate 0) -- or None when this plan
        carries no attitude."""
        if self.rp is None:
            return None
        ts = np.atleast_1d(np.asarray(ts, dtype=float))
        rp = np.vstack([np.interp(ts, self.t, self.rp[i]) for i in range(2)])
        if self.t.size >= 2:
            idx = np.clip(np.searchsorted(self.t, ts, side="right") - 1,
                          0, self.t.size - 2)
            inside = (ts >= self.t[0]) & (ts < self.t[-1])
            rr = self._rpseg[:, idx] * inside
        else:
            rr = np.zeros((2, ts.size))
        return rp, rr


class _BlendRef:
    """Cosine crossfade w(t): 0 -> 1 over ``dur`` from ``old`` into ``new``.

    p / yaw / g are blended; v carries the CROSS TERM w_dot (p_new - p_old) so
    the returned velocity is the exact derivative of the returned position —
    the term whose absence would make the feedforward lie during every
    hand-over. The new plan's yaw was already shifted by a whole number of
    2 pi at install time (relative unwrap), so the blend turns the short way.
    """

    def __init__(self, old, new: _PlanRef, t_start: float, dur: float):
        self.old = old
        self.new = new
        self.t_start = float(t_start)
        self.dur = float(dur)

    @property
    def t_blend_end(self) -> float:
        return self.t_start + self.dur

    def sample(self, ts: np.ndarray) -> tuple:
        ts = np.atleast_1d(np.asarray(ts, dtype=float))
        po, yo, vo, ro, go = self.old.sample(ts)
        pn, yn, vn, rn, gn = self.new.sample(ts)
        s = np.clip((ts - self.t_start) / self.dur, 0.0, 1.0)
        w = 0.5 * (1.0 - np.cos(math.pi * s))
        wdot = (math.pi / (2.0 * self.dur)) * np.sin(math.pi * s)
        dyaw = yn - yo
        p = (1.0 - w) * po + w * pn
        yaw = yo + w * dyaw
        v = (1.0 - w) * vo + w * vn + wdot * (pn - po)
        r = (1.0 - w) * ro + w * rn + wdot * dyaw
        g = (1.0 - w) * go + w * gn
        return p, yaw, v, r, g

    def sample_att(self, ts: np.ndarray) -> Optional[tuple]:
        """The attitude mirror of :meth:`sample`: ``rp = rpo + w (rpn - rpo)``
        and ``rp_rate = (1-w) rro + w rrn + w_dot (rpn - rpo)`` (cross term =
        the true derivative). Exactly ONE side without attitude: that side
        is treated as holding the other side's rp(t) (so the hand-over is
        never a step to level) with the other side's rate -- the returned
        rate stays the derivative of the returned attitude. Both without:
        None."""
        ts = np.atleast_1d(np.asarray(ts, dtype=float))
        ao = self.old.sample_att(ts)
        an = self.new.sample_att(ts)
        if ao is None and an is None:
            return None
        if ao is None:
            ao = an
        elif an is None:
            an = ao
        rpo, rro = ao
        rpn, rrn = an
        s = np.clip((ts - self.t_start) / self.dur, 0.0, 1.0)
        w = 0.5 * (1.0 - np.cos(math.pi * s))
        wdot = (math.pi / (2.0 * self.dur)) * np.sin(math.pi * s)
        drp = rpn - rpo
        rp = rpo + w * drp
        rr = (1.0 - w) * rro + w * rrn + wdot * drp
        return rp, rr


class PlanStitcher:
    """Turns the accepted plan stream into ONE continuous reference.

    The first install is taken verbatim; every later install blends from the
    CURRENT sampled reference (whatever it is — plan, hold, or an unfinished
    earlier blend) into the new plan over ``blend_s``, so the reference the
    20 Hz consumer sees is C0-continuous at every hand-over and its velocity
    is the true derivative of its position (cross term included).
    """

    # Gripper value used when NO plan in the stream has ever carried g.
    # 0 = closed; a later plan WITH g takes over through the normal blend.
    G_DEFAULT = 0.0

    def __init__(self, blend_s: float = 0.4):
        self.blend_s = float(blend_s)
        self._ref = None            # _PlanRef | _BlendRef | None
        self._last_g = self.G_DEFAULT
        # Last flown attitude knot (2,) of the newest plan that carried rp;
        # None until any plan does. A later plan WITHOUT rp holds it (the g
        # pattern) -- never a step back to level inside the stream.
        self._last_rp: Optional[np.ndarray] = None

    # -- lifecycle ---------------------------------------------------------
    def install(self, msg: PlanMsg, now: float) -> None:
        new = _PlanRef(msg, g_default=self._last_g, rp_default=self._last_rp)
        self._last_g = float(new.g[-1])
        if new.rp is not None:
            self._last_rp = new.rp[:, -1].copy()
        if self._ref is None or self.blend_s <= 0.0:
            self._ref = new
            return
        cur = self._ref
        # A finished blend collapses to its new plan so the chain only grows
        # when installs outrun blend_s (pathological; bounded by the source).
        if isinstance(cur, _BlendRef) and now >= cur.t_blend_end:
            cur = cur.new
        # Relative unwrap: shift the whole new plan by the multiple of 2 pi
        # that puts its yaw at `now` nearest the current reference's, so a
        # +pi -> -pi pair blends 0.08 rad the short way, not 6.2 the long way.
        yaw_cur = float(cur.sample(np.array([now]))[1][0])
        yaw_new = float(new.sample(np.array([now]))[1][0])
        new.yaw = new.yaw - _TWO_PI * round((yaw_new - yaw_cur) / _TWO_PI)
        self._ref = _BlendRef(cur, new, t_start=now, dur=self.blend_s)

    def clear(self) -> None:
        self._ref = None
        self._last_g = self.G_DEFAULT
        self._last_rp = None

    def preview_install(self, msg: PlanMsg, now: float,
                        dt: float = PREVIEW_DT_S) -> Dict[str, float]:
        """What :meth:`install` WOULD hand the consumer over the blend window,
        without installing anything (v2 A11: the post-stitch blend gate).

        Builds the same candidate — relative unwrap included — that
        :meth:`install` would build, samples it over ``[now, now + blend_s]``
        at ``dt`` and returns the finite-difference peaks of the SAMPLED
        reference: ``v_peak`` [m/s], ``a_peak`` [m/s^2], ``r_peak`` [rad/s],
        plus ``n_samples``. The filter's kinematic gate sees the plan's KNOTS;
        the blend adds the cross term ``w_dot (p_new - p_old)`` on top, and
        that term is what this measures — an accepted plan can still ask the
        blend for more speed than the vehicle has (the analytic bound in
        test_plan_stream is v_max + jump_max_m / blend_s, and a policy's
        jump_max_m is sized for the vehicle, not for the blend). ``self`` is
        untouched: the caller decides whether to :meth:`install` afterwards.
        With no plan installed (or ``blend_s <= 0``) the candidate is the new
        plan alone, so the peaks are its own knot kinematics over the window.
        ``rp_rate_peak`` [rad/s] (the blended attitude's finite-difference
        Euler rate, max over roll and pitch) is added ONLY when ``msg``
        itself carries ``rp`` -- a 4-DoF candidate's record is unchanged.
        """
        dt = float(dt)
        if not (dt > 0.0):
            raise ValueError("preview_install: dt must be > 0")
        new = _PlanRef(msg, g_default=self._last_g, rp_default=self._last_rp)
        cur = self._ref
        if cur is None or self.blend_s <= 0.0:
            cand = new
            dur = max(self.blend_s, dt)
        else:
            if isinstance(cur, _BlendRef) and now >= cur.t_blend_end:
                cur = cur.new
            yaw_cur = float(cur.sample(np.array([now]))[1][0])
            yaw_new = float(new.sample(np.array([now]))[1][0])
            new.yaw = new.yaw - _TWO_PI * round((yaw_new - yaw_cur) / _TWO_PI)
            cand = _BlendRef(cur, new, t_start=now, dur=self.blend_s)
            dur = self.blend_s
        n = int(math.floor(dur / dt + 1e-9)) + 1
        ts = float(now) + np.arange(max(n, 3)) * dt
        p, yaw, _v, _r, _g = cand.sample(ts)
        p = np.asarray(p, dtype=float)
        yaw_u = np.unwrap(np.asarray(yaw, dtype=float))
        v = np.diff(p, axis=1) / dt
        speed = np.linalg.norm(v, axis=0)
        a = np.diff(v, axis=1) / dt
        rr = np.diff(yaw_u) / dt
        out = {
            "v_peak": float(speed.max()) if speed.size else 0.0,
            "a_peak": (float(np.linalg.norm(a, axis=0).max())
                       if a.size else 0.0),
            "r_peak": float(np.abs(rr).max()) if rr.size else 0.0,
            "n_samples": int(ts.size),
        }
        if getattr(msg, "rp", None) is not None:
            att = cand.sample_att(ts)
            rp_s = np.asarray(att[0], dtype=float)
            drp = np.diff(rp_s, axis=1) / dt
            out["rp_rate_peak"] = float(np.abs(drp).max()) if drp.size else 0.0
        return out

    # -- queries -----------------------------------------------------------
    def has_plan(self) -> bool:
        return self._ref is not None

    def active_plan_id(self) -> Optional[int]:
        if self._ref is None:
            return None
        ref = self._ref
        return (ref.new if isinstance(ref, _BlendRef) else ref).plan_id

    def end_time(self) -> float:
        """Time of the active (newest) plan's last knot."""
        if self._ref is None:
            raise RuntimeError("PlanStitcher.end_time: no plan installed")
        ref = self._ref
        return (ref.new if isinstance(ref, _BlendRef) else ref).t_end

    def end_velocity(self) -> np.ndarray:
        """Velocity (3,) of the active (newest) plan's LAST segment — what
        the plan was doing when it ran out. Zeros for a one-knot plan. The
        ``hold_tail: extrapolate`` rule continues the reference at this
        velocity past :meth:`end_time` (workers._extrapolate_tail); the
        stitcher itself keeps holding the endpoint."""
        if self._ref is None:
            raise RuntimeError("PlanStitcher.end_velocity: no plan installed")
        ref = self._ref
        nw = ref.new if isinstance(ref, _BlendRef) else ref
        return np.asarray(nw._vseg[:, -1], dtype=float).copy()

    def source_at(self, t: float) -> str:
        if self._ref is None:
            return "none"
        ref = self._ref
        if isinstance(ref, _BlendRef) and ref.t_start <= t < ref.t_blend_end:
            return "blend"
        newest = ref.new if isinstance(ref, _BlendRef) else ref
        return "hold" if t > newest.t_end else "plan"

    def sample(self, ts: np.ndarray) -> tuple:
        """(p (3,K), yaw (K,), v (3,K), r (K,), g (K,)) at times ``ts``."""
        if self._ref is None:
            raise RuntimeError("PlanStitcher.sample: no plan installed")
        return self._ref.sample(ts)

    def sample_att(self, ts: np.ndarray) -> Optional[tuple]:
        """(rp (2,K) [roll, pitch] rad, rp_rate (2,K) rad/s) at ``ts``, or
        None when no installed plan has ever carried attitude (the consumer
        levels). :meth:`sample` keeps its 5-tuple; this is the separate
        attitude channel of the pos_rpy_width variant (2026-09-26)."""
        if self._ref is None:
            raise RuntimeError("PlanStitcher.sample_att: no plan installed")
        return self._ref.sample_att(ts)


# --------------------------------------------------------------------------
# replay helpers (M0: one recorded handheld demo through the same seam)
# --------------------------------------------------------------------------
@dataclass
class ReplayTrack:
    t: np.ndarray             # (T,) seconds, 0 at first sample
    p: np.ndarray             # (3, T) in the BODY frame at t=0 (NED axes)
    yaw: np.ndarray           # (T,) relative to initial yaw, unwrapped [rad]
    g: Optional[np.ndarray]   # (T,) in [0,1], or None
    meta: dict
    # (2, T) [roll, pitch] rad relative to the first retained pose (Euler
    # difference), or None (the default: the replay consumer levels).
    rp: Optional[np.ndarray] = None


def _yaw_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """ZYX yaw from (T,4) w-first quaternions (body FRD in NED)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rpy_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """ZYX (roll, pitch) as (2, T) from (T,4) w-first quaternions (body FRD
    in NED) -- the two angles :func:`_yaw_from_quat_wxyz` leaves out; the
    same Hamilton convention, pitch clipped at the asin domain."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return np.vstack([roll, pitch])


def _moving_average(x: np.ndarray, n: int) -> np.ndarray:
    """Centered moving average, window shrinking at the edges. ``n`` odd."""
    if n <= 1 or x.size <= 2:
        return np.asarray(x, dtype=float).copy()
    c = np.concatenate(([0.0], np.cumsum(np.asarray(x, dtype=float))))
    half = n // 2
    idx = np.arange(x.size)
    lo = np.clip(idx - half, 0, x.size)
    hi = np.clip(idx + half + 1, 0, x.size)
    return (c[hi] - c[lo]) / (hi - lo)


def _count_csv_rows(path: Path) -> int:
    """Data rows in a CSV, tolerating one header line."""
    with path.open("r", newline="") as fh:
        rows = [r for r in csv.reader(fh) if r and any(c.strip() for c in r)]
    if not rows:
        return 0
    try:
        float(rows[0][0])
        return len(rows)
    except ValueError:
        return len(rows) - 1


def load_replay_track(session_dir: str, *, trim_still: bool = True,
                      still_speed: float = 0.02,
                      smooth_window_s: float = 0.3,
                      with_rp: bool = False) -> ReplayTrack:
    """Load one recorded handheld demo into a body-frame :class:`ReplayTrack`.

    Reads ``<session_dir>/poses.npy`` (float64 (T, 8) rows of
    [t_unix, x, y, z, qw, qx, qy, qz]; NaN rows = no fix), ``poses.json``
    (must declare schema ``umi_handheld_poses/1``), ``frames.csv``
    (row-aligned with poses.npy), and ``gripper_width.npy`` if present
    ((T, 1) float32 in [0, 1], 0 = closed, same row alignment).

    Processing order: drop no-fix rows -> quaternion -> yaw (ZYX) + unwrap ->
    light smoothing (centered moving average over ``smooth_window_s``;
    positions per axis, then yaw — already unwrapped, so a plain average is
    circular-safe) -> optional still head/tail trim (finite-diff speed below
    ``still_speed``) -> re-express in the body frame of the FIRST RETAINED
    pose: translate to it, then rotate by -yaw0 about z. By default
    (``with_rp=False``, the replay.track_attitude default) the demo's roll
    and pitch are IGNORED and ``rp`` is None — the replay consumer levels
    the vehicle itself, and tilting the whole demo by the operator's wrist
    pose at t=0 would bake that wrist error into every waypoint. With
    ``with_rp=True`` (2026-09-26, the attitude-tracking variant) the ZYX
    roll/pitch of every retained pose is ALSO carried as ``rp`` (2, T),
    smoothed like yaw and expressed RELATIVE to the first retained pose
    (an Euler difference, so ``rp[:, 0] == 0`` like ``yaw[0] == 0``) —
    the wrist attitude at t=0 is still not baked in; only its CHANGES are.
    ``t``, ``p``, ``yaw``, ``g`` are computed identically either way.
    Because the anchor is the first retained pose, ``p[:, 0] == 0`` and
    ``yaw[0] == 0`` hold whether or not the head was trimmed.
    """
    sdir = Path(session_dir)
    poses_path = sdir / "poses.npy"
    meta_path = sdir / "poses.json"
    frames_path = sdir / "frames.csv"
    if not poses_path.is_file():
        raise ValueError(f"{sdir}: poses.npy not found — not a pose session")
    if not meta_path.is_file():
        raise ValueError(f"{sdir}: poses.json not found")
    if not frames_path.is_file():
        raise ValueError(f"{sdir}: frames.csv not found")

    with meta_path.open("r") as fh:
        poses_meta = json.load(fh)
    if poses_meta.get("schema") != "umi_handheld_poses/1":
        raise ValueError(
            f"{meta_path}: schema {poses_meta.get('schema')!r} is not "
            f"'umi_handheld_poses/1' — refusing to guess the row layout")

    raw = np.asarray(np.load(poses_path), dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 8:
        raise ValueError(f"{poses_path}: expected (T, 8), got {raw.shape}")
    n_raw = raw.shape[0]

    n_frames = _count_csv_rows(frames_path)
    if n_frames != n_raw:
        raise ValueError(
            f"{sdir}: frames.csv has {n_frames} rows but poses.npy has "
            f"{n_raw} — the two are supposed to be row-aligned")

    g_raw = None
    g_path = sdir / "gripper_width.npy"
    if g_path.is_file():
        g_raw = np.asarray(np.load(g_path), dtype=float).reshape(-1)
        if g_raw.size != n_raw:
            raise ValueError(
                f"{g_path}: {g_raw.size} rows, poses.npy has {n_raw} — "
                f"row alignment broken")

    valid = np.all(np.isfinite(raw), axis=1)
    valid &= np.linalg.norm(raw[:, 4:8], axis=1) > 1e-6
    n_used = int(valid.sum())
    if n_used < MIN_FIX_COUNT:
        raise ValueError(
            f"{sdir}: only {n_used} valid pose rows of {n_raw} "
            f"(need >= {MIN_FIX_COUNT}) — track is unusable")

    rows = raw[valid]
    t = rows[:, 0] - rows[0, 0]
    if not np.all(np.diff(t) > 0.0):
        raise ValueError(
            f"{poses_path}: pose timestamps are not strictly increasing "
            f"after dropping no-fix rows")
    p = rows[:, 1:4].T.copy()                        # (3, T) world
    q = rows[:, 4:8]
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    yaw = np.unwrap(_yaw_from_quat_wxyz(q))
    g = np.clip(g_raw[valid], 0.0, 1.0) if g_raw is not None else None
    rp = _rpy_from_quat_wxyz(q) if with_rp else None

    # Light smoothing (positions, then the already-unwrapped yaw).
    if smooth_window_s > 0.0 and t.size >= 3:
        dt_med = float(np.median(np.diff(t)))
        n = int(round(smooth_window_s / max(dt_med, 1e-6)))
        if n % 2 == 0:
            n += 1
        if n > 1:
            p = np.vstack([_moving_average(p[i], n) for i in range(3)])
            yaw = _moving_average(yaw, n)
            if rp is not None:
                rp = np.vstack([_moving_average(rp[i], n) for i in range(2)])

    # Trim still head/tail on the SMOOTHED speed.
    if trim_still and t.size >= 3:
        speed = np.linalg.norm(np.diff(p, axis=1), axis=0) / np.diff(t)
        moving = speed >= still_speed
        if np.any(moving):
            first = int(np.argmax(moving))
            last = int(len(moving) - np.argmax(moving[::-1]) - 1)
            sl = slice(first, last + 2)              # speeds sit between rows
            t, p, yaw = t[sl], p[:, sl], yaw[sl]
            if g is not None:
                g = g[sl]
            if rp is not None:
                rp = rp[:, sl]
        # An entirely-still track is kept whole: anchored, it is a hold.

    # Re-express in the body frame of the first retained pose.
    yaw0 = float(yaw[0])
    p = _rz(-yaw0) @ (p - p[:, :1])
    yaw = yaw - yaw0
    t = t - t[0]
    if rp is not None:
        rp = rp - rp[:, :1]

    meta = {
        "session_dir": str(sdir),
        "n_raw": n_raw,
        "n_used": n_used,
        "duration_s": float(t[-1]),
        "trim_still": bool(trim_still),
        "still_speed": float(still_speed),
        "smooth_window_s": float(smooth_window_s),
        "poses_json": poses_meta,
        "with_rp": bool(with_rp),
    }
    return ReplayTrack(t=t, p=p, yaw=yaw, g=g, meta=meta, rp=rp)


def time_dilate(track: ReplayTrack, v_max: float) -> Tuple[ReplayTrack, float]:
    """Uniformly slow a track so its p95 speed fits ``v_max``.

    alpha = max(1, p95_speed / v_max); t *= alpha. Geometry (p, yaw, g) is
    untouched — only the clock stretches, so the demo's PATH is preserved
    exactly and only its tempo changes. p95 rather than max: one glitchy
    finite-diff spike must not slow the whole demo to a crawl.
    """
    alpha = 1.0
    if track.t.size >= 2:
        speed = (np.linalg.norm(np.diff(track.p, axis=1), axis=0)
                 / np.diff(track.t))
        p95 = float(np.percentile(speed, 95))
        alpha = max(1.0, p95 / float(v_max))
    meta = dict(track.meta)
    meta["time_dilation_alpha"] = alpha
    rp_t = getattr(track, "rp", None)
    out = ReplayTrack(t=track.t * alpha, p=track.p.copy(),
                      yaw=track.yaw.copy(),
                      g=None if track.g is None else track.g.copy(),
                      meta=meta,
                      rp=None if rp_t is None else np.asarray(rp_t).copy())
    return out, alpha


def _track_interp(track: ReplayTrack, t_rel: np.ndarray):
    """Sample a track's body-frame p / yaw / g / rp at relative times
    (clamped); ``rp`` is None when the track carries none."""
    p = np.vstack([np.interp(t_rel, track.t, track.p[i]) for i in range(3)])
    yaw = np.interp(t_rel, track.t, track.yaw)
    g = np.interp(t_rel, track.t, track.g) if track.g is not None else None
    rp_t = getattr(track, "rp", None)
    rp = (np.vstack([np.interp(t_rel, track.t, rp_t[i]) for i in range(2)])
          if rp_t is not None else None)
    return p, yaw, g, rp


def _rp_anchor(rp: Optional[np.ndarray], rp0) -> Optional[np.ndarray]:
    """The track's relative roll/pitch placed at the anchor attitude
    ``rp0`` (2,) (None = level, i.e. (0, 0)); None passes through."""
    if rp is None:
        return None
    if rp0 is None:
        return rp
    return rp + np.asarray(rp0, dtype=float).reshape(2, 1)


def anchor_track(track: ReplayTrack, p0_ned: np.ndarray, yaw0: float, *,
                 t0: float, dt: float = 0.25, plan_id: int = 0,
                 rp0=None) -> PlanMsg:
    """M0(a): the whole demo as ONE plan, anchored at (p0_ned, yaw0).

    The track's body-at-start frame is placed at the vehicle's current
    reference: rotate by ``yaw0`` about z, translate by ``p0_ned``, add
    ``yaw0`` to yaw, and resample onto the ``dt`` knot grid (the grid always
    covers the track's tail; the last knot clamps to the endpoint). A track
    that carries ``rp`` (``load_replay_track(with_rp=True)``) forwards it as
    ``PlanMsg.rp`` offset by the anchor attitude ``rp0`` (2,) (None = level);
    a track without rp yields ``rp None`` -- today's message, byte for byte.
    """
    p0 = np.asarray(p0_ned, dtype=float).reshape(3, 1)
    t_end = float(track.t[-1])
    K = max(2, int(math.ceil(t_end / dt - 1e-9)) + 1)
    t_rel = np.arange(K) * dt
    pb, yb, gb, rb = _track_interp(track, t_rel)
    return PlanMsg(plan_id=int(plan_id), t0=float(t0), dt=float(dt),
                   p_ned=_rz(yaw0) @ pb + p0, yaw=yb + yaw0, g=gb,
                   obs_t=None, arrival_t=float(t0), rp=_rp_anchor(rb, rp0))


def chop_track(track: ReplayTrack, p0_ned, yaw0: float, *, t0: float,
               horizon_s: float = 4.0, period_s: float = 1.0,
               dt: float = 0.25, rp0=None) -> List[PlanMsg]:
    """M0(b): the same anchoring as :func:`anchor_track`, cut into
    overlapping windows [k*period, k*period + horizon] and emitted as a list
    of :class:`PlanMsg` with increasing ``plan_id`` and
    ``t0 = t0 + k*period`` — the mock-planner feed that exercises the
    filter + stitcher seam exactly the way the diffusion policy will.
    Windows past the track's end clamp to the endpoint (terminal hold).
    ``rp0`` as in :func:`anchor_track`.
    """
    p0 = np.asarray(p0_ned, dtype=float).reshape(3, 1)
    R = _rz(yaw0)
    t_end = float(track.t[-1])
    Kw = max(2, int(round(horizon_s / dt)) + 1)
    n_win = int(math.floor(t_end / period_s + 1e-9)) + 1
    msgs: List[PlanMsg] = []
    for k in range(n_win):
        t_rel = k * period_s + np.arange(Kw) * dt
        pb, yb, gb, rb = _track_interp(track, t_rel)
        msgs.append(PlanMsg(plan_id=k, t0=float(t0 + k * period_s),
                            dt=float(dt), p_ned=R @ pb + p0, yaw=yb + yaw0,
                            g=gb, obs_t=None,
                            arrival_t=float(t0 + k * period_s),
                            rp=_rp_anchor(rb, rp0)))
    return msgs
