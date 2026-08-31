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
    "dilation"  (clip only) the uniform time-dilation factor alpha
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

    # -- the gate chain ----------------------------------------------------
    def evaluate(self, msg: PlanMsg, *,
                 r_now: Tuple[np.ndarray, float],
                 cur_sample: Optional[Callable[[np.ndarray], tuple]] = None,
                 now: float) -> Verdict:
        lim = self.limits
        margins: Dict[str, float] = {}
        reasons: List[str] = []

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
        yaw_u = np.unwrap(yaw)
        t_knots = msg.t0 + np.arange(K) * msg.dt

        # 2. freshness -----------------------------------------------------
        if msg.obs_t is not None:
            age = now - float(msg.obs_t)
            margins["obs_age"] = lim.obs_max_age_s - age
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
            reasons.append(
                f"anchor gate: plan at t=now sits {d_anchor:.3f} m from the "
                f"current reference (limit {lim.anchor_max_m:.3f} m).")
            return self._reject(margins, reasons)

        # 4. overlap jump gate --------------------------------------------
        if cur_sample is not None:
            t_lo = max(now, t_knots[0])
            ts = np.concatenate(([t_lo], t_knots[t_knots > t_lo]))
            if ts.size >= 1 and ts[-1] >= t_lo:
                p_new = np.vstack(
                    [np.interp(ts, t_knots, p[i]) for i in range(3)])
                p_cur = np.asarray(cur_sample(ts)[0], dtype=float)
                dev = float(np.max(np.linalg.norm(p_new - p_cur, axis=0)))
                margins["jump_m"] = lim.jump_max_m - dev
                if dev > lim.jump_max_m:
                    reasons.append(
                        f"overlap jump: plan deviates up to {dev:.3f} m from "
                        f"the current reference over the overlap window "
                        f"(limit {lim.jump_max_m:.3f} m).")
                    return self._reject(margins, reasons)

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
        if need > 1.0:
            if need > CLIP_RATIO_MAX:
                reasons.append(
                    f"kinematics: repairing this plan needs a {need:.2f}x "
                    f"time dilation (peak speed {v_peak:.3f}/{lim.v_max:.3f} "
                    f"m/s, accel {a_peak:.3f}/{lim.a_max:.3f} m/s^2, yaw "
                    f"rate {r_peak:.3f}/{lim.r_max:.3f} rad/s) — beyond the "
                    f"{CLIP_RATIO_MAX}x clip band, rejecting.")
                return self._reject(margins, reasons)
            # Minor: keep the geometry, dilate the clock uniformly about t0.
            alpha = need
            margins["speed"] = lim.v_max - v_peak / alpha
            margins["accel"] = lim.a_max - a_peak / (alpha * alpha)
            margins["yaw_rate"] = lim.r_max - r_peak / alpha
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
                reasons.append(
                    f"workspace box: a knot leaves the box by "
                    f"{-box_m:.3f} m — rejecting (the box is the only "
                    f"position-based protection; the geofence was removed).")
                return self._reject(margins, reasons)

        return self._pass(status, out, margins, reasons)


# --------------------------------------------------------------------------
# stitcher
# --------------------------------------------------------------------------
class _PlanRef:
    """One installed plan as a sampleable reference: piecewise-linear p /
    yaw / g over the knots, finite-diff (piecewise-constant) v / r, endpoint
    hold beyond the last knot (p = end, v = 0, r = 0, yaw = end, g = end) and
    symmetrically before the first knot."""

    def __init__(self, msg: PlanMsg, g_default: float):
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

    # -- lifecycle ---------------------------------------------------------
    def install(self, msg: PlanMsg, now: float) -> None:
        new = _PlanRef(msg, g_default=self._last_g)
        self._last_g = float(new.g[-1])
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


def _yaw_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """ZYX yaw from (T,4) w-first quaternions (body FRD in NED)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


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
                      smooth_window_s: float = 0.3) -> ReplayTrack:
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
    pose: translate to it, then rotate by -yaw0 about z. The anchor's roll
    and pitch are deliberately IGNORED — the replay consumer levels the
    vehicle itself, and tilting the whole demo by the operator's wrist pose
    at t=0 would bake that wrist error into every waypoint. Because the
    anchor is the first retained pose, ``p[:, 0] == 0`` and ``yaw[0] == 0``
    hold whether or not the head was trimmed.
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

    # Light smoothing (positions, then the already-unwrapped yaw).
    if smooth_window_s > 0.0 and t.size >= 3:
        dt_med = float(np.median(np.diff(t)))
        n = int(round(smooth_window_s / max(dt_med, 1e-6)))
        if n % 2 == 0:
            n += 1
        if n > 1:
            p = np.vstack([_moving_average(p[i], n) for i in range(3)])
            yaw = _moving_average(yaw, n)

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
        # An entirely-still track is kept whole: anchored, it is a hold.

    # Re-express in the body frame of the first retained pose.
    yaw0 = float(yaw[0])
    p = _rz(-yaw0) @ (p - p[:, :1])
    yaw = yaw - yaw0
    t = t - t[0]

    meta = {
        "session_dir": str(sdir),
        "n_raw": n_raw,
        "n_used": n_used,
        "duration_s": float(t[-1]),
        "trim_still": bool(trim_still),
        "still_speed": float(still_speed),
        "smooth_window_s": float(smooth_window_s),
        "poses_json": poses_meta,
    }
    return ReplayTrack(t=t, p=p, yaw=yaw, g=g, meta=meta)


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
    out = ReplayTrack(t=track.t * alpha, p=track.p.copy(),
                      yaw=track.yaw.copy(),
                      g=None if track.g is None else track.g.copy(),
                      meta=meta)
    return out, alpha


def _track_interp(track: ReplayTrack, t_rel: np.ndarray):
    """Sample a track's body-frame p / yaw / g at relative times (clamped)."""
    p = np.vstack([np.interp(t_rel, track.t, track.p[i]) for i in range(3)])
    yaw = np.interp(t_rel, track.t, track.yaw)
    g = np.interp(t_rel, track.t, track.g) if track.g is not None else None
    return p, yaw, g


def anchor_track(track: ReplayTrack, p0_ned: np.ndarray, yaw0: float, *,
                 t0: float, dt: float = 0.25, plan_id: int = 0) -> PlanMsg:
    """M0(a): the whole demo as ONE plan, anchored at (p0_ned, yaw0).

    The track's body-at-start frame is placed at the vehicle's current
    reference: rotate by ``yaw0`` about z, translate by ``p0_ned``, add
    ``yaw0`` to yaw, and resample onto the ``dt`` knot grid (the grid always
    covers the track's tail; the last knot clamps to the endpoint).
    """
    p0 = np.asarray(p0_ned, dtype=float).reshape(3, 1)
    t_end = float(track.t[-1])
    K = max(2, int(math.ceil(t_end / dt - 1e-9)) + 1)
    t_rel = np.arange(K) * dt
    pb, yb, gb = _track_interp(track, t_rel)
    return PlanMsg(plan_id=int(plan_id), t0=float(t0), dt=float(dt),
                   p_ned=_rz(yaw0) @ pb + p0, yaw=yb + yaw0, g=gb,
                   obs_t=None, arrival_t=float(t0))


def chop_track(track: ReplayTrack, p0_ned, yaw0: float, *, t0: float,
               horizon_s: float = 4.0, period_s: float = 1.0,
               dt: float = 0.25) -> List[PlanMsg]:
    """M0(b): the same anchoring as :func:`anchor_track`, cut into
    overlapping windows [k*period, k*period + horizon] and emitted as a list
    of :class:`PlanMsg` with increasing ``plan_id`` and
    ``t0 = t0 + k*period`` — the mock-planner feed that exercises the
    filter + stitcher seam exactly the way the diffusion policy will.
    Windows past the track's end clamp to the endpoint (terminal hold).
    """
    p0 = np.asarray(p0_ned, dtype=float).reshape(3, 1)
    R = _rz(yaw0)
    t_end = float(track.t[-1])
    Kw = max(2, int(round(horizon_s / dt)) + 1)
    n_win = int(math.floor(t_end / period_s + 1e-9)) + 1
    msgs: List[PlanMsg] = []
    for k in range(n_win):
        t_rel = k * period_s + np.arange(Kw) * dt
        pb, yb, gb = _track_interp(track, t_rel)
        msgs.append(PlanMsg(plan_id=k, t0=float(t0 + k * period_s),
                            dt=float(dt), p_ned=R @ pb + p0, yaw=yb + yaw0,
                            g=gb, obs_t=None,
                            arrival_t=float(t0 + k * period_s)))
    return msgs
