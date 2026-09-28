#!/usr/bin/env python3
"""tilt_tracker.py — where the camera mount IS, from the best evidence there is.

The BlueROV2's tilt mount is open-loop from where this station sits: ArduSub
drives it from HELD joystick buttons and, on this vehicle, reports no angle
back (no MOUNT_STATUS arrives and no servo channel is named — the PAYLOAD
panel has said "no angle reported (open loop)" since the first pool day). The
mount still matters twice over: the ROV RGB camera's extrinsic depends on it,
and an accidental press leaves it wherever it stops with nothing on screen to
say so (operator, 2026-09-07).

So the angle is TRACKED from three kinds of evidence, best first:

  measured   the vehicle's own report (MOUNT_STATUS / a named servo's PWM),
             or the angle the tag localizer MEASURES when both cameras see
             the mat (control/nav_fusion.py BodyAlign.measured_tilt_deg);
  set        the operator typed it (they drove to the stop and know the
             number, or read it off the vehicle);
  dead-reckoned
             held UP/DOWN integrated at ``rate_deg_s`` [예측] since the last
             anchor, clamped to the mount's range; LEVEL is an anchor at
             ``level_deg`` (mount_center is a real position command).

Every reading carries a source and an uncertainty, and the uncertainty GROWS
with dead-reckoned travel — the rate is a guess until someone times a
sweep — so the panel can print "-18° ±6 (dead-reckoned)" rather than a
number that looks measured.

``epoch`` counts COMMANDED motion (a hold started, LEVEL, a manual set); it is
what tells the localizer "the extrinsic just changed, throw the alignment
away". A measurement does NOT bump it — otherwise the localizer's own
measurement would loop back and reset the alignment that produced it.

No Qt. Time is passed in, so tests can drive it deterministically.
"""

from __future__ import annotations

import math


class TiltTracker:
    #: Fraction of dead-reckoned travel added to the uncertainty [예측]: the
    #: rate is a guess, so half of any distance travelled on it is doubt.
    RATE_DOUBT = 0.5

    def __init__(self, rate_deg_s: float = 30.0, lo_deg: float = -45.0,
                 hi_deg: float = 45.0, level_deg: float = 0.0):
        self.rate_deg_s = float(rate_deg_s)
        self.lo_deg, self.hi_deg = float(lo_deg), float(hi_deg)
        self.level_deg = float(level_deg)
        self.deg: float | None = None        # None = no evidence yet
        self.src = "unknown"
        self.unc_deg: float = math.inf
        self.epoch = 0
        self.moving = False
        self._drive = 0.0
        self._t = None                        # last integration time

    # ----------------------------------------------------------- evidence
    def drive(self, direction: float, t: float) -> None:
        """A held UP (+1) / DOWN (-1) started or stopped (0)."""
        self._integrate(t)
        d = float(direction)
        if d and not self._drive:
            self.epoch += 1                   # motion begins
            if self.deg is None:
                # Nothing known: assume it was level, with the whole range
                # as doubt. The number is still better than nothing — it
                # says which WAY it went and roughly how far.
                self.deg, self.unc_deg = self.level_deg, self.hi_deg - self.lo_deg
                self.src = "dead-reckoned"
        self._drive = d
        self.moving = bool(d)
        self._t = float(t)

    def center(self, t: float) -> None:
        """mount_center: a real position command, so an ANCHOR."""
        self._integrate(t)
        self.epoch += 1
        self.deg, self.src, self.unc_deg = self.level_deg, "level", 1.0
        self.moving = False
        self._drive = 0.0
        self._t = float(t)

    def set_measured(self, deg: float, src: str, t: float,
                     unc_deg: float = 1.0) -> None:
        """The vehicle or the tag localizer measured it. No epoch bump."""
        self._integrate(t)
        self.deg = self._clamp(float(deg))
        self.src = f"measured:{src}"
        self.unc_deg = float(unc_deg)
        self._t = float(t)

    def set_manual(self, deg: float, t: float) -> None:
        """The operator typed it: an anchor AND a commanded change."""
        self._integrate(t)
        self.epoch += 1
        self.deg, self.src, self.unc_deg = self._clamp(float(deg)), "set", 2.0
        self._t = float(t)

    def update(self, t: float) -> None:
        """Advance dead reckoning to ``t`` (call from the sender's tick)."""
        self._integrate(t)
        self._t = float(t)

    # ------------------------------------------------------------ reading
    def note(self) -> str:
        if self.deg is None:
            return "unknown — press LEVEL or SET"
        unc = ("" if not math.isfinite(self.unc_deg)
               else f" ±{self.unc_deg:.0f}")
        return f"{self.deg:+.1f}°{unc} ({self.src})"

    # ------------------------------------------------------------ internal
    def _clamp(self, v: float) -> float:
        return max(self.lo_deg, min(self.hi_deg, v))

    def _integrate(self, t: float) -> None:
        if self._t is None or not self._drive or self.deg is None:
            return
        dt = max(0.0, float(t) - self._t)
        before = self.deg
        self.deg = self._clamp(self.deg + self._drive * self.rate_deg_s * dt)
        moved = abs(self.deg - before)
        if moved > 0.0:
            self.src = "dead-reckoned"
            self.unc_deg = (self.unc_deg if math.isfinite(self.unc_deg)
                            else 0.0) + self.RATE_DOUBT * moved


def tracker_from_opts(opts) -> TiltTracker:
    """A tracker sized from the CLI and hw_nav.yaml: --tilt-rate-deg-s wins,
    else second_cam.tilt_rate_deg_s, else the [예측] 30 deg/s; the range from
    --tilt-min/max-deg (the mount's own limits, ArduPilot default ±45)."""
    rate = getattr(opts, "tilt_rate_deg_s", None)
    if rate is None:
        try:
            from .geometry import NavConfig
            cfg = NavConfig.load(getattr(opts, "nav_config", "config/hw_nav.yaml"))
            rate = cfg.second_cam.get("tilt_rate_deg_s", 30.0)
        except Exception:                                   # noqa: BLE001
            rate = 30.0                                     # [예측]
    return TiltTracker(rate_deg_s=float(rate),
                       lo_deg=float(getattr(opts, "tilt_min_deg", -45.0)),
                       hi_deg=float(getattr(opts, "tilt_max_deg", 45.0)))
