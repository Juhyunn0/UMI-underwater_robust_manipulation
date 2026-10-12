#!/usr/bin/env python3
"""station_bridge.py — survive a tag dropout during a STATION hold.

THE PROBLEM. A disturbance kick blurs the camera or swings the tags out of
frame, the localizer stops producing a fix, and about 1.1 s later
(``tag_stale_s`` 0.7 + ``tag_stale_hold_s`` 0.4) the whole loop DISENGAGES.
That is worse than it sounds: disengaging drops DEPTH hold as well, and the
vehicle drifts with its trim of the day (the model says -5.7 N sinking;
2026-09-07 it floated — hw_mpc.yaml vehicle_net_buoyancy_n), so it starts moving
away from the tags — which
changes the attitude and the view and makes re-acquisition less likely, not
more. The operator then has to re-ARM.

Turning on the ``imu_dr`` experiment does NOT fix this, and the reason is
worth stating: ``MpcWorker._runtime_fault`` is fed ``meas`` — the TAG
solution, deliberately kept as ground truth — not ``meas_ctrl``, the state
the controller actually eats. So the interlock fires whatever the estimator
is doing. What has to be separated is not "tag vs IMU" but **"the estimate
stopped" from "give up controlling"**.

THE LADDER. Which axes may be carried, and for how long, is not one question
but four, because the sensors behind them are different:

    axis          source while the tag is gone        drift
    ------------  ----------------------------------  ---------------------
    z             barometer                           NONE (absolute)
    roll, pitch   ArduSub AHRS                        NONE (absolute)
    yaw           gyro integration                    slow, unbounded
    x, y          accelerometer double integration    t^2, fast

so:

    tier NONE    fix is fresh                     everything, as always
    tier IMU     0 .. imu_hold_s after the fix    all axes, on the bridge
                                                  estimate, full authority
    tier COAST   after imu_hold_s                 x/y/yaw RELEASED, depth
                                                  and attitude still held —
                                                  INDEFINITELY

and, since 2026-10-01 (control/dvl.py, HARDWARE-UNVERIFIED), one more rung
for a vehicle that carries a DVL the operator has switched IN:

    tier DVL     0 .. dvl_hold_s, only while the  all axes; x/y integrate the
                 DVL stays usable WITHOUT a gap   DVL's bottom velocity from
                 from the first lost tick         the last fix — drift grows
                                                  with t, not t^2

A DVL measures velocity over the floor, so x/y no longer rest on a double
integral and may be carried longer than ``imu_hold_s``. The moment it stops
being usable the ladder is exactly the one above, judged on the same elapsed
time — and it does NOT climb back onto the DVL later in the same dropout:
the position would then have a hole of unknown motion in it.

The DVL rung is the one stretch where NOTHING cross-checks the DVL (the tag
that proved it is the thing that is missing), so it is bounded by what a
station hold IS: a vehicle nominally at rest over one point. A DVL that
reports more than ``dvl_max_speed_m_s``, or whose integral has wandered more
than ``dvl_max_disp_m`` from the last fix, is describing something other
than a station hold — a false bottom lock off a wall, most likely — and the
rung is dropped for the rest of the dropout (review 2026-10-02: without
these, 10 s of a bogus 0.2 m/s is 2 m of full-authority push the wrong way).

COAST is the point of the design, and it is deliberately not a disengage.
Pushing on a horizontal estimate that is drifting actively drives the vehicle
somewhere wrong, and nothing in this station stops it by position any more
(the geofence was removed 2026-08-14). Releasing the horizontal axes lets it
drift with the water instead, which at pool speeds is slow — while the two
axes that never drift keep the vehicle at depth and level, which is the state
most likely to get the tags back. It is safe to sit there forever, so it does.

Yaw is released with x and y rather than held: gyro bias of 0.02 rad/s is
70 deg/min, and a vehicle slowly rotating to chase a drifting heading
reference is worse than one that simply sits. The camera looks DOWN
(``--nav-geometry floor``), so heading is not what keeps tags in view.

STATION ONLY, on purpose. A path mission that loses its fix has a moving
reference and a real velocity; this ladder assumes the vehicle is nominally
at rest over one point, which is what makes the x/y estimate defensible for
a few seconds and the coast tier safe for an unbounded one.

The OBJECT FOLLOW mission (2026-08-21, control/object_nav.py) obeys that
boundary rather than widening it: losing the tag fix demotes a follow to a
station hold on its last setpoint, and only THEN does this ladder take over.
Nothing in this file knows about it. Two things make the demotion the right
shape — a follow is structurally a moving mission, and its target is composed
FROM a NavFix, so a vehicle with no fix has nothing left to follow. Carrying
the hull on an IMU while the target is unobservable would be subtracting two
drifting quantities.

Nothing here is hardware- or Qt-aware, so the ladder is tested directly
(rov_gui/tests/test_station_bridge.py).
"""

from __future__ import annotations

import numpy as np

TIER_NONE = "none"      # the tag fix is fresh; not bridging
TIER_IMU = "imu"        # carrying every axis on the bridge estimate
TIER_DVL = "dvl"        # ...with x/y integrated from the DVL's velocity
TIER_COAST = "coast"    # depth + attitude only; horizontal released

#: Faults the bridge may carry. EVERYTHING else still disengages — most
#: importantly the two that would be bridged with a sensor that is itself
#: dead ("imu stale", "pressure depth stale") and the one that means the
#: autopilot stopped talking ("no vehicle imu").
BRIDGEABLE = ("no tag fix", "tag fix stale")

DEFAULTS = {
    "enabled": True,
    # How long every axis may be carried on the bridge estimate before the
    # horizontal ones are released. 3 s at a 0.02 m/s^2 residual accelerometer
    # bias is ~9 cm of position error [유도]; without a calibration the same
    # 3 s is 8 m, which is why xy_source refuses to use a raw sensor.
    "imu_hold_s": 3.0,
    # auto = use the accelerometer only when a calibration is loaded, else
    # freeze x/y at the last fix. imu / hold force it either way.
    "xy_source": "auto",
    # null = the mission's normal axis_cap (operator decision 2026-08-18).
    "axis_cap": None,
    # DVL rung (2026-10-01). Used ONLY when a DVL is built, switched IN and
    # usable at the dropout; otherwise these two keys change nothing.
    # 10 s: at the A50's stated +-1 % of speed and a few mm/s of noise the
    # integrated error over a station hold is millimetres, and the yaw it is
    # rotated by is the same gyro-carried one tier IMU uses [예측 — no DVL
    # has been in this pool; replace with the drift each recovery logs].
    "use_dvl": True,
    "dvl_hold_s": 10.0,
    # The rung's own plausibility bounds (see the module doc). A station
    # hold that is really moving at 0.3 m/s, or has really drifted half a
    # metre, has stopped being one. Both [예측].
    "dvl_max_speed_m_s": 0.3,
    "dvl_max_disp_m": 0.5,
}

KEYS = tuple(DEFAULTS)


def resolve(raw: dict | None) -> dict:
    """Merge a ``station_bridge:`` block over the defaults, rejecting typos.

    Same rule as ``imu_dr``: a silently ignored key here means the operator
    flies a pool session believing a safety ladder is armed when it is not."""
    cfg = dict(DEFAULTS)
    if not raw:
        return cfg
    unknown = set(raw) - set(KEYS)
    if unknown:
        raise ValueError(f"unknown station_bridge keys {sorted(unknown)}; "
                         f"known: {sorted(KEYS)}")
    cfg.update(raw)
    cfg["enabled"] = bool(cfg["enabled"])
    cfg["imu_hold_s"] = float(cfg["imu_hold_s"])
    if not cfg["imu_hold_s"] >= 0.0:
        raise ValueError(f"station_bridge.imu_hold_s must be >= 0, "
                         f"got {cfg['imu_hold_s']}")
    if str(cfg["xy_source"]) not in ("auto", "imu", "hold"):
        raise ValueError(f"station_bridge.xy_source must be auto|imu|hold, "
                         f"got {cfg['xy_source']!r}")
    cfg["xy_source"] = str(cfg["xy_source"])
    if cfg["axis_cap"] is not None:
        cfg["axis_cap"] = float(cfg["axis_cap"])
    if not isinstance(cfg["use_dvl"], bool):
        raise ValueError(f"station_bridge.use_dvl must be true or false, "
                         f"got {cfg['use_dvl']!r}")
    cfg["dvl_hold_s"] = float(cfg["dvl_hold_s"])
    if not cfg["dvl_hold_s"] >= 0.0:
        raise ValueError(f"station_bridge.dvl_hold_s must be >= 0, "
                         f"got {cfg['dvl_hold_s']}")
    for k in ("dvl_max_speed_m_s", "dvl_max_disp_m"):
        try:
            v = float(cfg[k])
        except (TypeError, ValueError):
            v = float("nan")
        if isinstance(cfg[k], bool) or not v > 0.0:     # also catches NaN
            raise ValueError(f"station_bridge.{k} must be a number > 0, "
                             f"got {cfg[k]!r}")
        cfg[k] = v
    return cfg


def is_bridgeable(why: str) -> bool:
    """Is this runtime fault one the bridge is allowed to carry?

    The assembler embeds the age in the string (``tag fix stale (0.83s)``),
    so an exact match will not do — but a bare ``startswith`` would also
    accept ``tag fix stale-ish but a different fault``, which is how a
    whitelist quietly stops being one. A whole-word boundary is the middle:
    the whitelisted phrase, then end of string or a space. Anything else is
    NOT bridgeable — a new fault string has to be opted in deliberately,
    never inherited."""
    w = str(why or "")
    return any(w == b or w.startswith(b + " ") for b in BRIDGEABLE)


class StationBridge:
    """The tier ladder. One instance per worker; reset at engage and STOP."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = resolve(cfg)
        # Does K/M actually leave the station this engagement
        # (engage.attitude_axes, 2026-09-26)? The worker sets it at engage;
        # False = the 4-DoF station, where the allocation drops them anyway.
        self.attitude_axes = False
        self.reset()

    # ---------------------------------------------------------------- state
    def reset(self) -> None:
        """A NEW engagement: the ladder and its counters start from zero."""
        self.n_entered = 0           # how many dropouts this engagement
        self.n_coast = 0             # ...and how many reached the coast tier
        self.n_dvl = 0               # ...and how many were carried on the DVL
        self.worst_s = 0.0           # longest dropout so far
        self.last_recover = None     # {"elapsed", "tier", "dvl_s", ...}
        self.end_dropout()

    def end_dropout(self) -> None:
        """Stop bridging, KEEP the counters. For a disengage or a STOP TRAJ:
        the closing meta is written after them, and until 2026-10-02 it was
        written from a ladder ``reset()`` had already zeroed — every run's
        ``dropouts`` / ``reached_coast`` / ``last_recovery`` closed as 0 /
        0 / null on disk whatever had happened in the water."""
        self.tier = TIER_NONE
        self._t_lost = None          # when the last fresh fix was seen
        self._dvl_chain = False      # DVL usable on EVERY lost tick so far
        self._dvl_s = 0.0            # seconds of THIS dropout on the DVL rung

    @property
    def enabled(self) -> bool:
        return bool(self.cfg["enabled"])

    @property
    def elapsed(self) -> float:
        """Seconds since the last fresh fix, 0 while not bridging."""
        return 0.0 if self._t_lost is None else float(self._t_lost)

    @property
    def active(self) -> bool:
        return self.tier != TIER_NONE

    # ----------------------------------------------------------------- step
    def note_fix(self, t: float) -> dict | None:
        """A fresh tag fix arrived. Returns a recovery record, or None if the
        bridge was not running — the record IS the measurement this feature
        gets for free (how far the estimate had drifted when truth came
        back), so the caller logs it rather than throwing it away."""
        if self._t_lost is None:
            self.tier = TIER_NONE
            return None
        rec = {"elapsed": float(self._t_lost), "tier": self.tier,
               # how much of it the DVL rung carried (0.0 = none of it)
               "dvl_s": round(float(self._dvl_s), 3)}
        self._t_lost = None
        self._dvl_chain = False
        self._dvl_s = 0.0
        self.tier = TIER_NONE
        self.last_recover = rec
        return rec

    def dvl_allowed(self, dt: float) -> bool:
        """May the NEXT lost tick still be carried on the DVL rung? True at
        the start of a dropout, and afterwards only while the chain is
        unbroken and ``dvl_hold_s`` has not run out. Asked BEFORE
        ``note_lost`` so the state the caller builds and the tier that call
        returns can never disagree."""
        if not self.cfg["use_dvl"]:
            return False
        if self._t_lost is None:
            return True
        return bool(self._dvl_chain) and (
            self._t_lost + max(0.0, float(dt)) <= self.cfg["dvl_hold_s"])

    def note_lost(self, dt: float, dvl_ok: bool = False) -> str:
        """One tick with no fresh fix. Returns the tier now in force.

        ``dvl_ok`` = the caller integrated a usable DVL velocity THIS tick.
        The DVL rung holds only while that has been true on every lost tick
        of this dropout (``_dvl_chain``)."""
        dvl_ok = bool(dvl_ok) and bool(self.cfg["use_dvl"])
        if self._t_lost is None:
            self._t_lost = 0.0
            self.n_entered += 1
            self.tier = TIER_IMU
            self._dvl_chain = dvl_ok
            if dvl_ok:
                self.n_dvl += 1
        else:
            self._t_lost += max(0.0, float(dt))
            self._dvl_chain = self._dvl_chain and dvl_ok
        self.worst_s = max(self.worst_s, self._t_lost)
        if self._dvl_chain and self._t_lost <= self.cfg["dvl_hold_s"]:
            self.tier = TIER_DVL
            self._dvl_s += max(0.0, float(dt))
        elif self._t_lost > self.cfg["imu_hold_s"]:
            if self.tier != TIER_COAST:
                self.n_coast += 1
            self.tier = TIER_COAST
        else:
            self.tier = TIER_IMU
        return self.tier

    # ------------------------------------------------------------ actuation
    def release_horizontal(self, u, attitude: bool | None = None):
        """Zero X, Y and N of a body wrench; leave Z alone. K and M too when
        ``attitude`` (default: :attr:`attitude_axes`).

        Belt and braces with the re-targeting the worker does in the coast
        tier: the reference is moved onto the vehicle so the controller has
        nothing to ask for, AND the surge/sway/yaw components are zeroed here
        so a wound-up integrator or a disturbance feedforward cannot put
        thrust out through the gap. On the 4-DoF station K and M are dropped
        by the allocation anyway (MANUAL_CONTROL's four axes carry no
        roll/pitch), so the coast tier commands the HEAVE axis and nothing
        else. Under ``engage.attitude_axes`` (2026-09-26) they WOULD reach
        the wire, and a coast is not the place for an attitude command on an
        estimate that stopped — so they are zeroed here as well and the meta
        says so (``released_in_coast``)."""
        v = np.asarray(u, float).copy()
        v[0] = v[1] = 0.0
        if v.size > 5:
            v[5] = 0.0
        att = self.attitude_axes if attitude is None else bool(attitude)
        if att and v.size > 4:
            v[3] = v[4] = 0.0
        return v

    def detail(self) -> str:
        """The one line the panel and the mission log show."""
        if self.tier == TIER_DVL:
            return f"DVL bridge {self.elapsed:.1f}s"
        if self.tier == TIER_IMU:
            return f"IMU bridge {self.elapsed:.1f}s"
        if self.tier == TIER_COAST:
            return f"COASTING {self.elapsed:.0f}s — depth + attitude only"
        return ""

    def meta(self) -> dict:
        return {"enabled": self.enabled,
                "imu_hold_s": float(self.cfg["imu_hold_s"]),
                "xy_source": self.cfg["xy_source"],
                "axis_cap": self.cfg["axis_cap"],
                "use_dvl": bool(self.cfg["use_dvl"]),
                "dvl_hold_s": float(self.cfg["dvl_hold_s"]),
                "dvl_max_speed_m_s": float(self.cfg["dvl_max_speed_m_s"]),
                "dvl_max_disp_m": float(self.cfg["dvl_max_disp_m"]),
                "dropouts": int(self.n_entered),
                "dropouts_on_dvl": int(self.n_dvl),
                "reached_coast": int(self.n_coast),
                "worst_dropout_s": round(float(self.worst_s), 3),
                "last_recovery": self.last_recover,
                "scope": "station hold only (an object follow is demoted "
                         "to a station hold first, then carried here)",
                "released_in_coast": ("X, Y, N, and K, M under attitude_axes "
                                      "(heave kept; attitude levels passively)"
                                      if self.attitude_axes else
                                      "X, Y, N (heave and attitude kept)")}
