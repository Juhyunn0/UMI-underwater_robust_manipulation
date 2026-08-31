#!/usr/bin/env python3
"""object_nav.py — put the TRACKED OBJECT into the pool's frame, and keep it.

THE GAP THIS CLOSES. Two independent 6-DoF estimators run in this station and
neither has ever heard of the other:

  * ``--mpc``'s :class:`~rov_gui.control.workers.TagNavWorker` produces the
    VEHICLE's pose in the tag-map world (``bus.nav_fix`` -> ``NavFix.p_ned`` /
    ``R_ned_body``, +z DOWN);
  * ``--pose``'s ``PoseWorker`` produces the clicked OBJECT's pose relative to
    the CAMERA (``bus.pose`` -> ``PoseTrack.T_cam_obj``, OpenCV optical,
    object -> camera).

The pose log says so in as many words — *"NO camera->body transform is
applied; the vehicle's own pose is not in this file."* So nothing in the
station could say where in the pool a clicked object is, and nothing could be
done relative to it. This module is the bridge, and it is deliberately pure:
no Qt, no torch, no acados, numpy only, so the frame algebra is unit-tested
offline (rov_gui/tests/test_object_nav.py) instead of at the pool.

THE ONE PROPERTY WORTH PROTECTING — the extrinsic cancels.
``TagNav._solution`` DIVIDES the camera extrinsic out of the PnP camera pose
to report a BODY pose; this module MULTIPLIES it back in. When the object
observation and the tag fix come from THE SAME CAMERA FRAME, those two
operations are exact inverses:

    R_map_cam = R_ned_body . R_frd_cam = (R_nm R_mc)     <- extrinsic-free
    p_map_cam = p_ned + R_ned_body . t_frd_cam = (R_nm t_mc)

which matters because the C3's re-mount is only half measured: the 43.3 deg
tilt is applied as a PURE ROTATION and the translation that a real hinge also
moves (``cam_t_flu``, a 0.2855 m lever arm) is still unmeasured
(KNOWN_ISSUES 2026-08-17). Same-frame pairing makes that unknown drop out
entirely. Pair ACROSS frames and it comes straight back: a yaw difference of
theta between the two frames leaves a residual of
``|t_frd_cam| * 2*sin(theta/2)`` — 2.5 cm at 5 deg. That is what
:func:`pick_fix` and ``pair_tol_s`` exist for, and why ``pair_exact`` rides
into the CSV and onto the panel: it is the only warning that the cancellation
has stopped.

THE OTHER RULE — compose from a RAW ``NavFix``, never from ``meas["eta"]``.
The state assembler returns a HYBRID (z from the barometer, roll/pitch from
the autopilot's AHRS, x/y velocity-propagated between camera frames) and its
``tick`` datumizes as well. Composing an object pose out of that breaks the
cancellation on z/roll/pitch and resurrects the lever arm. It would also be
completely silent. Use the fix.

FRAMES, once, in words. Everything this module returns is in the MAP frame —
the tag-map NED world the pool, the mat and the plot are drawn in. It is NOT
the engage-datum frame the controller and ``MpcStatus`` live in, and it does
not move when START is pressed.
"""

from __future__ import annotations

import math

import numpy as np

#: The states a :class:`ObjectAnchor` can be in, in escalation order.
COLD = "cold"        # never locked on
LIVE = "live"        # a fresh pose from a TRACKING pipeline
STALE = "stale"      # the estimate is old, or the pipeline left "tracking"
LOST = "lost"        # stale for longer than hold_s

#: The only ``PoseTrack.state`` that counts as a lock. ``PoseWorker``
#: publishes at 10 Hz whatever the pipeline is doing, so a 30-second
#: ``registering`` arrives with a perfectly fresh stamp — age alone can never
#: catch it, and this is the check that does.
TRACKING = "tracking"

DEFAULTS = {
    # How far apart the object frame and the tag frame may be and still be
    # called "the same frame". [유도] 0.2855 m lever arm x 0.5 rad/s of yaw
    # rate is ~1.1 cm of residual at 0.08 s. Exact matches are preferred
    # whatever this says — see pick_fix.
    "pair_tol_s": 0.08,
    # The object leg is ALWAYS the C3: it is the only camera whose extrinsic
    # is measured, and the ROV RGB's whole second_cam block is [예측]. If the
    # tag fix comes from a different camera the cancellation above is not just
    # inexact, it is meaningless.
    "nav_source_required": "main",
    # Which object axis defines its heading, projected onto the map's
    # horizontal plane. "auto" = pick the most horizontal one at the first
    # lock and NEVER change it (a mid-run switch is a 90 deg step in the
    # heading reference). "none" = do not use the object's yaw at all and
    # keep the follow offset in the MAP frame — one whole failure class
    # disappears, so it is the recommended first bench mode.
    "yaw_axis": "auto",              # auto | x | y | z | none
    # Below this horizontal length the chosen axis is too close to vertical
    # for atan2 to mean anything (0.25 = within 14.5 deg of vertical).
    "yaw_min_horiz": 0.25,
    "max_jump_m": 0.35,              # [예측]
    "max_yaw_jump_deg": 45.0,        # [예측]
    # Consecutive rejections that mean the object really did move rather than
    # the estimator glitching once.
    "reseed_after_n": 5,
    # [측정: KNOWN_ISSUES 2026-08-09 — the pipeline works at 0.3-0.8 m on
    # stored data and fails to register at 2.4 m. The band here is wider than
    # the working one on purpose: this gate is for nonsense, not for quality.]
    "min_distance_m": 0.15,
    "max_distance_m": 1.20,
    "pos_lp_alpha": 0.5,             # [예측]
    "vel_lp_alpha": 0.3,             # [예측]
    # Constant-velocity extrapolation is clamped here; past it the point
    # STOPS and only the age keeps rising (an honest failure). Also the
    # freshness bound for LIVE — see ObjectAnchor.state.
    "max_extrap_s": 0.30,            # [예측]
    # Display honesty bound: past this the published ObjectFix carries
    # ok=False and the panel draws the marker hollow. Deliberately looser
    # than max_extrap_s, which governs the CONTROL freeze.
    "stale_s": 0.50,
    # Not LIVE for this long -> LOST, and a follow demotes to a station hold.
    "hold_s": 2.00,
    # How far the follow target may travel from where START was pressed
    # before the reference is CLAMPED (operator decision). Not a geofence
    # revival: it refuses nothing and stops nothing, it only stops asking.
    "max_excursion_m": 1.50,
}

KEYS = tuple(DEFAULTS)

_FLOAT_KEYS = ("pair_tol_s", "yaw_min_horiz", "max_jump_m",
               "max_yaw_jump_deg", "min_distance_m", "max_distance_m",
               "pos_lp_alpha", "vel_lp_alpha", "max_extrap_s", "stale_s",
               "hold_s", "max_excursion_m")

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def wrap_pi(a: float) -> float:
    return math.atan2(math.sin(float(a)), math.cos(float(a)))


def resolve(raw: dict | None) -> dict:
    """Merge an ``object_nav:`` block over the defaults, rejecting typos.

    Same rule as ``station_bridge`` and ``imu_dr``, for the same reason: a
    silently ignored key here means the operator flies believing a distance
    gate or an excursion clamp is armed when it is not.
    """
    cfg = dict(DEFAULTS)
    if not raw:
        return cfg
    unknown = set(raw) - set(KEYS)
    if unknown:
        raise ValueError(f"unknown object_nav keys {sorted(unknown)}; "
                         f"known: {sorted(KEYS)}")
    cfg.update(raw)
    for k in _FLOAT_KEYS:
        cfg[k] = float(cfg[k])
    cfg["reseed_after_n"] = int(cfg["reseed_after_n"])
    cfg["yaw_axis"] = str(cfg["yaw_axis"]).lower()
    cfg["nav_source_required"] = str(cfg["nav_source_required"]).lower()
    if cfg["yaw_axis"] not in ("auto", "x", "y", "z", "none"):
        raise ValueError(f"object_nav.yaw_axis must be auto|x|y|z|none, got "
                         f"{cfg['yaw_axis']!r}")
    if cfg["nav_source_required"] not in ("main", "second"):
        raise ValueError(f"object_nav.nav_source_required must be "
                         f"main|second, got {cfg['nav_source_required']!r}")
    if not cfg["pair_tol_s"] > 0.0:
        raise ValueError("object_nav.pair_tol_s must be > 0")
    if not 0.0 < cfg["yaw_min_horiz"] <= 1.0:
        raise ValueError("object_nav.yaw_min_horiz must be in (0, 1]")
    if not cfg["max_jump_m"] > 0.0:
        raise ValueError("object_nav.max_jump_m must be > 0")
    if not cfg["max_yaw_jump_deg"] > 0.0:
        raise ValueError("object_nav.max_yaw_jump_deg must be > 0")
    if cfg["reseed_after_n"] < 1:
        raise ValueError("object_nav.reseed_after_n must be >= 1")
    if not 0.0 <= cfg["min_distance_m"] < cfg["max_distance_m"]:
        raise ValueError("object_nav needs 0 <= min_distance_m < "
                         "max_distance_m")
    for k in ("pos_lp_alpha", "vel_lp_alpha"):
        if not 0.0 < cfg[k] <= 1.0:
            raise ValueError(f"object_nav.{k} must be in (0, 1]")
    if cfg["max_extrap_s"] < 0.0:
        raise ValueError("object_nav.max_extrap_s must be >= 0")
    for k in ("stale_s", "hold_s", "max_excursion_m"):
        if not cfg[k] > 0.0:
            raise ValueError(f"object_nav.{k} must be > 0")
    return cfg


# =============================================================================
# frames
# =============================================================================
def _T44(T_cam_obj) -> np.ndarray:
    """16 row-major floats (or a 4x4) -> a 4x4 array."""
    M = np.asarray(T_cam_obj, float)
    if M.size != 16:
        raise ValueError(f"T_cam_obj must be 16 floats, got {M.size}")
    return M.reshape(4, 4)


def compose_map_pose(T_cam_obj, p_ned, R_ned_body, R_frd_cam, t_frd_cam):
    """OBJECT pose in the MAP frame, from ONE camera frame's two estimates.

    ``T_cam_obj`` maps object -> camera (OpenCV optical, metres, row-major);
    ``p_ned`` / ``R_ned_body`` are a RAW :class:`~rov_gui.state.NavFix` (never
    ``meas["eta"]`` — see the module docstring); ``R_frd_cam`` / ``t_frd_cam``
    are ``NavConfig.R_t_frd_cam(source)``, the single definition of the
    extrinsic that already folds in ``cam_tilt_deg``.

    Returns ``(p_map (3,), R_map_obj (3,3))``.
    """
    M = _T44(T_cam_obj)
    R_cam_obj = M[0:3, 0:3]
    t_cam_obj = M[0:3, 3]
    R_nb = np.asarray(R_ned_body, float).reshape(3, 3)
    R_bc = np.asarray(R_frd_cam, float).reshape(3, 3)
    t_bc = np.asarray(t_frd_cam, float).reshape(3)
    p_nb = np.asarray(p_ned, float).reshape(3)

    R_map_cam = R_nb @ R_bc
    p_map_cam = p_nb + R_nb @ t_bc
    return p_map_cam + R_map_cam @ t_cam_obj, R_map_cam @ R_cam_obj


def pick_axis(R_map_obj) -> int:
    """Which object axis to read the heading off: the most HORIZONTAL one.

    Chosen ONCE, at the first lock, and pinned for the anchor's life. An
    on-site BundleSDF reconstruction has no canonical axes, so "x" means
    nothing in particular; what does mean something is "the axis that lies in
    the pool's horizontal plane", because that is the one whose projection is
    a stable heading. Re-picking mid-run would step the heading reference by
    up to 90 degrees.
    """
    R = np.asarray(R_map_obj, float).reshape(3, 3)
    return int(np.argmax([math.hypot(R[0, k], R[1, k]) for k in range(3)]))


def yaw_of(R_map_obj, axis: int | None):
    """``(yaw_rad, horizontal_length)`` of one object axis in the map frame.

    NOT ``geometry.yaw_from_R``: that is a ZYX psi extraction and it diverges
    as the object's +x approaches vertical, which is a perfectly ordinary
    thing for a reconstructed object to do. Projecting a CHOSEN axis onto the
    map's horizontal plane degrades gracefully instead — the projection simply
    gets short, and the caller can see that it has and refuse to use it.

    ``axis=None`` (``yaw_axis: none``) returns ``(None, 0.0)``: the object's
    heading is deliberately not part of the problem.
    """
    if axis is None:
        return None, 0.0
    R = np.asarray(R_map_obj, float).reshape(3, 3)
    a = R[:, int(axis)]
    h = math.hypot(float(a[0]), float(a[1]))
    return math.atan2(float(a[1]), float(a[0])), h


def pick_fix(hist, t_capture: float, tol_s: float):
    """The tag fix belonging to the SAME camera frame as an object pose.

    ``hist`` is an iterable of ``(t_capture, p_ned, R_ned_body)``. Returns
    ``(entry | None, dt_s, exact)``.

    An EXACT float match always wins, and comparing floats for equality is
    the right test here rather than a smell: ``C3VideoWorker._tap_pose``
    computes ``t_capture`` ONCE per colour frame and puts the identical float
    into both mailboxes, so equality answers exactly the question being
    asked — "did these two estimates come from one frame?" — with no
    tolerance to tune. The nearest-within-tolerance fallback is the
    degraded path, and ``dt_s`` is what says how degraded.
    """
    best = None
    best_dt = float("inf")
    for e in hist:
        t = float(e[0])
        if t == float(t_capture):
            return e, 0.0, True
        dt = abs(t - float(t_capture))
        if dt < best_dt:
            best, best_dt = e, dt
    if best is None or best_dt > float(tol_s):
        return None, best_dt, False
    return best, best_dt, False


# =============================================================================
# the anchor
# =============================================================================
class ObjectAnchor:
    """One tracked object, filtered, in the MAP frame.

    Owns the gates (distance, jump, reseed), the filter (position EMA plus a
    capture-time finite-difference velocity) and the freshness ladder. Every
    time quantity it consumes is a CAPTURE stamp, never a host clock: the
    pose arrives 10 Hz-published from a 30 fps camera through a GPU, so a
    wall-clock difference measures the pipeline, not the object.
    """

    def __init__(self, cfg: dict | None = None):
        self.cfg = resolve(cfg)
        self.reset()

    # ---------------------------------------------------------------- state
    def reset(self) -> None:
        self.axis = (None if self.cfg["yaw_axis"] == "none"
                     else _AXIS_INDEX.get(self.cfg["yaw_axis"]))
        # "auto" is the ONE mode that picks an axis at the first lock; every
        # other mode — including "none", whose answer is "no axis at all" — is
        # decided here and must stay decided. Testing `self.axis is not None`
        # instead conflated "none" with "auto" (both leave axis None) and
        # silently turned yaw_axis:"none" into an axis-pinned follow: the
        # object's own yaw then rotated the hold offset AND the heading
        # reference, so estimator yaw noise became commanded vehicle motion.
        # 2026-08-23 flew that way — meta said yaw_axis "none" / yaw_axis_used
        # "x" in the same breath (KNOWN_ISSUES 2026-08-24).
        self._axis_pinned = self.cfg["yaw_axis"] != "auto"
        self.p = None                  # filtered map position (3,)
        self.R = None                  # last accepted R_map_obj (3,3)
        self.yaw = None                # filtered heading, rad, or None
        self.yaw_ok = False
        self.v = np.zeros(3)           # filtered map velocity
        self.r = 0.0                   # filtered yaw rate
        self.t_cap = None              # capture stamp of the last accept
        self.t_live = None             # host time of the last LIVE observation
        self.distance_m = None
        self.track_ok = False          # was the last report a TRACKING one?
        self.note = ""
        self.n_obs = 0
        self.n_reject = 0              # CONSECUTIVE rejections
        self.n_reject_total = 0
        self.n_reseed = 0
        self.n_pair_exact = 0
        self.n_pair_loose = 0

    # ----------------------------------------------------------------- feed
    def note_pair(self, exact: bool) -> None:
        """Count how the last observation was paired to a tag fix. Kept here
        so the ratio lands in :meth:`meta` beside everything else the run has
        to record about this estimate."""
        if exact:
            self.n_pair_exact += 1
        else:
            self.n_pair_loose += 1

    def update(self, p_map, R_map_obj, t_capture: float, t_now: float,
               distance_m: float, pose_state: str) -> dict:
        """One object observation. Returns what was done with it.

        The gates run in the order a failure should be reported in: the
        pipeline's own verdict first (a mask being refined is not a lock), the
        distance sanity gate next, then the jump gate — which REJECTS one
        outlier but RESEEDS after ``reseed_after_n`` of them, because a run of
        rejections is what an object that genuinely moved looks like.
        """
        c = self.cfg
        out = {"accepted": False, "reseeded": False, "why": "",
               "state": self.state(t_now)}
        self.track_ok = str(pose_state) == TRACKING
        if not self.track_ok:
            out["why"] = f"pose state {pose_state!r}"
            self.note = out["why"]
            out["state"] = self.state(t_now)
            return out
        d = float(distance_m)
        if not (c["min_distance_m"] <= d <= c["max_distance_m"]):
            self.note = (f"object at {d:.2f} m, outside "
                         f"{c['min_distance_m']:.2f}-"
                         f"{c['max_distance_m']:.2f} m")
            out["why"] = self.note
            out["state"] = self.state(t_now)
            return out

        p_new = np.asarray(p_map, float).reshape(3)
        R_new = np.asarray(R_map_obj, float).reshape(3, 3)
        if not self._axis_pinned:
            self.axis = pick_axis(R_new)
            self._axis_pinned = True
        yaw_new, horiz = yaw_of(R_new, self.axis)
        yaw_valid = yaw_new is not None and horiz >= c["yaw_min_horiz"]

        if self.p is None:
            self._seed(p_new, R_new, yaw_new if yaw_valid else None,
                       t_capture, t_now, d)
            out.update(accepted=True, why="first lock",
                       state=self.state(t_now))
            return out

        jump = float(np.linalg.norm(p_new - self.p))
        yaw_jump = 0.0
        if yaw_valid and self.yaw is not None:
            yaw_jump = abs(wrap_pi(yaw_new - self.yaw))
        if (jump > c["max_jump_m"]
                or yaw_jump > math.radians(c["max_yaw_jump_deg"])):
            self.n_reject += 1
            self.n_reject_total += 1
            self.note = (f"rejected {jump * 100:.0f} cm / "
                         f"{math.degrees(yaw_jump):.0f} deg jump "
                         f"({self.n_reject}/{c['reseed_after_n']})")
            if self.n_reject < c["reseed_after_n"]:
                out["why"] = self.note
                out["state"] = self.state(t_now)
                return out
            # Enough of them in a row: the object moved, we did not.
            self.n_reseed += 1
            self._seed(p_new, R_new, yaw_new if yaw_valid else None,
                       t_capture, t_now, d)
            self.note = f"reseeded after {c['reseed_after_n']} rejections"
            out.update(accepted=True, reseeded=True, why=self.note,
                       state=self.state(t_now))
            return out

        dt = float(t_capture) - float(self.t_cap)
        if 1e-3 <= dt <= 1.0:
            v_raw = (p_new - self.p) / dt
            a = c["vel_lp_alpha"]
            self.v = (1.0 - a) * self.v + a * v_raw
            if yaw_valid and self.yaw is not None:
                r_raw = wrap_pi(yaw_new - self.yaw) / dt
                self.r = (1.0 - a) * self.r + a * r_raw
        ap = c["pos_lp_alpha"]
        self.p = (1.0 - ap) * self.p + ap * p_new
        if yaw_valid:
            if self.yaw is None:
                self.yaw = float(yaw_new)
            else:
                self.yaw = wrap_pi(self.yaw
                                   + ap * wrap_pi(yaw_new - self.yaw))
            self.yaw_ok = True
        else:
            # The heading is undefined THIS frame; the last good one is kept
            # (a follow holds its bearing) but nothing pretends it is fresh.
            self.yaw_ok = False
        self.R = R_new
        self.t_cap = float(t_capture)
        self.t_live = float(t_now)
        self.distance_m = d
        self.n_obs += 1
        self.n_reject = 0
        self.note = ""
        out.update(accepted=True, state=self.state(t_now))
        return out

    def _seed(self, p_new, R_new, yaw_new, t_capture, t_now, d) -> None:
        self.p = np.asarray(p_new, float).reshape(3).copy()
        self.R = np.asarray(R_new, float).reshape(3, 3).copy()
        self.yaw = None if yaw_new is None else float(yaw_new)
        self.yaw_ok = yaw_new is not None
        self.v = np.zeros(3)
        self.r = 0.0
        self.t_cap = float(t_capture)
        self.t_live = float(t_now)
        self.distance_m = float(d)
        self.n_obs += 1
        self.n_reject = 0

    # ----------------------------------------------------------------- read
    def age(self, t_now: float) -> float:
        return (float("inf") if self.t_cap is None
                else max(0.0, float(t_now) - float(self.t_cap)))

    def state(self, t_now: float) -> str:
        """cold | live | stale | lost.

        LIVE needs BOTH a fresh capture stamp and a pipeline that says
        ``tracking``: ``PoseWorker`` publishes at 10 Hz regardless of phase, so
        a 30-second registration arrives perfectly fresh and would otherwise
        read as a lock.
        """
        if self.p is None:
            return COLD
        if self.track_ok and self.age(t_now) <= self.cfg["max_extrap_s"]:
            return LIVE
        gap = (float("inf") if self.t_live is None
               else float(t_now) - float(self.t_live))
        return STALE if gap <= self.cfg["hold_s"] else LOST

    def predict(self, t_now: float) -> dict:
        """Where the object is NOW, at constant velocity, honestly clamped.

        Past ``max_extrap_s`` the point simply STOPS and only ``age_s`` keeps
        growing. An estimate that kept gliding would look most confident
        exactly when it knows least.
        """
        st = self.state(t_now)
        if self.p is None:
            return {"ok": False, "state": st, "p": None, "yaw": None,
                    "yaw_ok": False, "v": np.zeros(3), "r": 0.0,
                    "age_s": None, "extrapolated_s": 0.0,
                    "distance_m": None, "note": self.note}
        age = self.age(t_now)
        ex = max(0.0, min(age, self.cfg["max_extrap_s"]))
        p = self.p + self.v * ex
        yaw = (None if self.yaw is None else wrap_pi(self.yaw + self.r * ex))
        # `ok` needs BOTH a fresh stamp and a pipeline that is tracking. Age
        # alone cannot catch a 30-second `registering`, which arrives at 10 Hz
        # with a perfect stamp — and a marker drawn confidently through one
        # would be the exact dishonesty this field exists to prevent.
        return {"ok": bool(self.track_ok and age <= self.cfg["stale_s"]),
                "state": st,
                "p": p, "yaw": yaw, "yaw_ok": bool(self.yaw_ok),
                "v": self.v.copy(), "r": float(self.r),
                "age_s": age, "extrapolated_s": ex,
                "distance_m": self.distance_m, "note": self.note}

    def meta(self) -> dict:
        n = self.n_pair_exact + self.n_pair_loose
        return {"enabled": True,
                "yaw_axis": self.cfg["yaw_axis"],
                "yaw_axis_used": ("none" if self.axis is None
                                  else "xyz"[self.axis]),
                "nav_source_required": self.cfg["nav_source_required"],
                "pair_tol_s": float(self.cfg["pair_tol_s"]),
                "max_jump_m": float(self.cfg["max_jump_m"]),
                "max_extrap_s": float(self.cfg["max_extrap_s"]),
                "stale_s": float(self.cfg["stale_s"]),
                "hold_s": float(self.cfg["hold_s"]),
                "max_excursion_m": float(self.cfg["max_excursion_m"]),
                "min_distance_m": float(self.cfg["min_distance_m"]),
                "max_distance_m": float(self.cfg["max_distance_m"]),
                "observations": int(self.n_obs),
                "rejections": int(self.n_reject_total),
                "reseeds": int(self.n_reseed),
                # THE RECORD BOUNDARY for any statistic about where the object
                # was: only the exactly-paired rows are extrinsic-free.
                "pairs_exact": int(self.n_pair_exact),
                "pairs_loose": int(self.n_pair_loose),
                "pair_exact_ratio": (round(self.n_pair_exact / n, 4)
                                     if n else None),
                "frame": "MAP (tag-map NED, +z down) — NOT the engage datum",
                "provenance": "[예측] every threshold until the first pool "
                              "hold-only run measures them"}


# =============================================================================
# the follow geometry
# =============================================================================
def offset_in_object_frame(p_veh_map, yaw_veh_map, p_obj_map, yaw_obj):
    """Capture the vehicle's CURRENT pose as an offset held in the object's
    own yaw frame. ``yaw_obj=None`` keeps the offset in the MAP frame.

    The object's roll and pitch are DELIBERATELY discarded. A buoy that rocks
    must not translate the vehicle, and MANUAL_CONTROL carries no K/M axis
    anyway, so a roll the follow tried to answer would be a command the
    allocation drops (``station_bridge.release_horizontal`` records the same
    fact). The consequence is exactly the one an operator would want:
    object translates -> vehicle translates; object yaws -> vehicle orbits;
    object rolls or pitches -> nothing happens.
    """
    d = np.asarray(p_veh_map, float).reshape(3) - \
        np.asarray(p_obj_map, float).reshape(3)
    if yaw_obj is None:
        return d.copy(), wrap_pi(float(yaw_veh_map))
    c, s = math.cos(float(yaw_obj)), math.sin(float(yaw_obj))
    # Rz(yaw)^T applied to d — the map->object-yaw rotation.
    return (np.array([c * d[0] + s * d[1],
                      -s * d[0] + c * d[1],
                      d[2]]),
            wrap_pi(float(yaw_veh_map) - float(yaw_obj)))


def follow_goal(p_obj_map, yaw_obj, offset_obj, dyaw, v_obj, r_obj):
    """``(goal, goal_yaw, v_track)`` — where the vehicle should be, and how
    fast that point is itself moving.

        goal    = p_obj + Rz(yaw_obj) . offset_obj
        v_track = v_obj + r_obj x (Rz . offset_obj)

    The orbit term is ``d/dpsi (Rz d) = (-(Rz d)_y, (Rz d)_x, 0)``, in the
    same +z-DOWN convention ``yaw_from_R`` uses, so there is no sign trap in
    it. With ``yaw_obj=None`` (``yaw_axis: none``) the offset is a map-frame
    vector, the orbit term vanishes, and the vehicle simply translates with
    the object.
    """
    p = np.asarray(p_obj_map, float).reshape(3)
    d = np.asarray(offset_obj, float).reshape(3)
    v = np.asarray(v_obj, float).reshape(3)
    if yaw_obj is None:
        return p + d, wrap_pi(float(dyaw)), v.copy()
    c, s = math.cos(float(yaw_obj)), math.sin(float(yaw_obj))
    a = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])
    v_track = v + float(r_obj) * np.array([-a[1], a[0], 0.0])
    return p + a, wrap_pi(float(yaw_obj) + float(dyaw)), v_track


def clamp_excursion(goal, arm_p, limit_m: float):
    """Keep the follow target inside a sphere around where START was pressed.

    Returns ``(goal, leashed)``. A REFERENCE clamp, not a refusal and not a
    geofence: the vehicle is never stopped or disengaged by it, the station
    simply stops asking for anything further out. That is the form the
    operator allowed back after the fence was removed on 2026-08-14.
    """
    g = np.asarray(goal, float).reshape(3)
    a = np.asarray(arm_p, float).reshape(3)
    d = g - a
    n = float(np.linalg.norm(d))
    if not (n > float(limit_m) > 0.0):
        return g, False
    return a + d * (float(limit_m) / n), True
