#!/usr/bin/env python3
"""station_bridge.py — survive a tag dropout during a STATION hold.

THE PROBLEM. A disturbance kick blurs the camera or swings the tags out of
frame, the localizer stops producing a fix, and about 1.1 s later
(``tag_stale_s`` 0.7 + ``tag_stale_hold_s`` 0.4) the whole loop DISENGAGES.
That is worse than it sounds: disengaging drops DEPTH hold as well, and the
vehicle is negatively buoyant (net -5.7 N), so it starts sinking — which
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
        self.reset()

    # ---------------------------------------------------------------- state
    def reset(self) -> None:
        self.tier = TIER_NONE
        self._t_lost = None          # when the last fresh fix was seen
        self.n_entered = 0           # how many dropouts this engagement
        self.n_coast = 0             # ...and how many reached the coast tier
        self.worst_s = 0.0           # longest dropout so far
        self.last_recover = None     # {"elapsed", "err_m", "tier"}

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
        rec = {"elapsed": float(self._t_lost), "tier": self.tier}
        self._t_lost = None
        self.tier = TIER_NONE
        self.last_recover = rec
        return rec

    def note_lost(self, dt: float) -> str:
        """One tick with no fresh fix. Returns the tier now in force."""
        if self._t_lost is None:
            self._t_lost = 0.0
            self.n_entered += 1
            self.tier = TIER_IMU
        else:
            self._t_lost += max(0.0, float(dt))
        self.worst_s = max(self.worst_s, self._t_lost)
        if self._t_lost > self.cfg["imu_hold_s"]:
            if self.tier != TIER_COAST:
                self.n_coast += 1
            self.tier = TIER_COAST
        else:
            self.tier = TIER_IMU
        return self.tier

    # ------------------------------------------------------------ actuation
    def release_horizontal(self, u):
        """Zero X, Y and N of a body wrench; leave Z (and K/M) alone.

        Belt and braces with the re-targeting the worker does in the coast
        tier: the reference is moved onto the vehicle so the controller has
        nothing to ask for, AND the surge/sway/yaw components are zeroed here
        so a wound-up integrator or a disturbance feedforward cannot put
        thrust out through the gap. K and M are already dropped by the
        allocation (MANUAL_CONTROL carries no roll/pitch axis), so on this
        vehicle the coast tier commands the HEAVE axis and nothing else."""
        v = np.asarray(u, float).copy()
        v[0] = v[1] = 0.0
        if v.size > 5:
            v[5] = 0.0
        return v

    def detail(self) -> str:
        """The one line the panel and the mission log show."""
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
                "dropouts": int(self.n_entered),
                "reached_coast": int(self.n_coast),
                "worst_dropout_s": round(float(self.worst_s), 3),
                "last_recovery": self.last_recover,
                "scope": "station hold only (an object follow is demoted "
                         "to a station hold first, then carried here)",
                "released_in_coast": "X, Y, N (heave and attitude kept)"}
