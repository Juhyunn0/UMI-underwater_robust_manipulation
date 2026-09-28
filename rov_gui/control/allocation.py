#!/usr/bin/env python3
"""allocation.py — NED body wrench -> normalized MANUAL_CONTROL axes.

In MANUAL mode ArduSub maps the four MANUAL_CONTROL axes straight through its
mixer to the thrusters, so a wrench command becomes an axis command through a
per-axis gain: ``axis = wrench / gain``, where ``gain`` is the wrench the
vehicle produces at FULL deflection. Roll (K) and pitch (M) are DROPPED by
default — the Heavy is passively stable and the sim's square runs them near
zero anyway; a documented limitation of the 4-DoF MANUAL-mode experiment,
not an oversight. (The evidence for "near zero" is a SQUARE. A
heading-following CIRCLE is the first mission to hold a non-zero yaw rate
for a whole lap — v/R, 34 deg/s at 0.12 m/s and R 0.2 — so the sustained
Coriolis coupling into K and M on one is outside what that covers
[예측, 미검증].)

THE 6-DoF VARIANT (2026-09-26, ``engage.attitude_axes``). Sub 4.1.2+ reads two
MORE MANUAL_CONTROL axes — the v2 extension fields ``s`` (pitch) and ``t``
(roll) — as torque demands in MANUAL. ``wrench_to_axes(..., attitude=True)``
maps K/M through ``roll_nm`` / ``pitch_nm`` onto ``PilotInput.roll`` /
``PilotInput.pitch`` with their OWN cap (``attitude_cap``), and the command
sink puts them on the wire (backends/hardware.py). The flag is EXPLICIT and
defaults to False: HwDobMpc and HwMpcc already emit levelling torques every
hold, and a gains-only gate would have started transmitting them in every
4-DoF run. With ``attitude=False`` the returned object is byte-identical to
the pre-variant one (roll = pitch = 0.0, the dataclass defaults).

The gains in ``config/hw_mpc.yaml`` are [예측] — derived from the T200 curve
and mixer geometry, never measured on this vehicle — until the P4 step
calibration replaces them. ``roll_nm`` 13.2 / ``pitch_nm`` 7.2 N·m are
[유도: heave_n 60 N = 4 x 15 N per vertical thruster x lever arms |y| 0.22 /
|x| 0.12 m, bluerov2_mujoco_marinegym/bluerov_heavy.xml:116-119]. An error
here is a plant-gain error the DOB-MPC's w_hat absorbs at DC and plain MPC
shows as tracking offset; both outcomes are part of the experiment's story
rather than a safety issue, because the axis CAP below bounds authority
regardless of how wrong the gain is.

Sign map (PilotInput docstring in state.py vs NED/FRD):
    surge  +forward   =  +X_ned
    sway   +starboard =  +Y_ned
    heave  +up        =  -Z_ned      (NED z is DOWN-positive)
    yaw    +CW-from-above = +N_ned
    roll   +starboard-down = +K_ned  [가정 until the in-water sign probe]
    pitch  +nose-up        = +M_ned  [가정 until the in-water sign probe]
The wire sign (``engage.attitude_axes.sign``) is applied at the SINK, not
here, so the recorded ``ax_roll`` / ``ax_pitch`` stay in NED sign.
"""

from __future__ import annotations

import math

from ..state import PilotInput, now

#: [유도] see the module docstring; the config's ``axis_gain`` overrides.
ROLL_NM_DEFAULT = 13.2
PITCH_NM_DEFAULT = 7.2


def wrench_to_axes(u_ned, gains: dict, cap: float = 0.5,
                   stamp: float | None = None, *, attitude: bool = False,
                   attitude_cap=(0.2, 0.3)) -> PilotInput:
    """u_ned = [X, Y, Z, K, M, N] (N, N·m).

    K, M are dropped unless ``attitude`` (see module doc); with it they map
    to ``roll`` / ``pitch`` through ``gains['roll_nm']`` / ``gains['pitch_nm']``
    and are clipped to ``+-attitude_cap[0]`` / ``+-attitude_cap[1]`` — a cap
    of their own, separate from ``cap``, because the first-water values are
    a fraction of the translational one."""
    gx = max(1e-6, float(gains.get("surge_n", 60.0)))
    gy = max(1e-6, float(gains.get("sway_n", 60.0)))
    gz = max(1e-6, float(gains.get("heave_n", 60.0)))
    gn = max(1e-6, float(gains.get("yaw_nm", 20.0)))
    c = max(0.0, min(1.0, float(cap)))

    def lim(v: float) -> float:
        return max(-c, min(c, v))

    if attitude:
        gk = max(1e-6, float(gains.get("roll_nm", ROLL_NM_DEFAULT)))
        gm = max(1e-6, float(gains.get("pitch_nm", PITCH_NM_DEFAULT)))
        ck = max(0.0, min(1.0, float(attitude_cap[0])))
        cm = max(0.0, min(1.0, float(attitude_cap[1])))
        roll = max(-ck, min(ck, float(u_ned[3]) / gk))
        pitch = max(-cm, min(cm, float(u_ned[4]) / gm))
    else:
        roll = 0.0
        pitch = 0.0

    return PilotInput(
        surge=lim(float(u_ned[0]) / gx),
        sway=lim(float(u_ned[1]) / gy),
        heave=lim(-float(u_ned[2]) / gz),
        yaw=lim(float(u_ned[5]) / gn),
        active=frozenset(("mpc",)),
        source="mpc",
        stamp=now() if stamp is None else stamp,
        roll=roll, pitch=pitch,
    ).clamped()


def slew_axes(cmd: PilotInput, prev, max_rate: float, dt: float, *,
              rate_rp: float | None = None) -> PilotInput:
    """Bound how fast each axis may move, per second.

    Thrust that reverses faster than the hull can answer is heat and noise, not
    control — and the yaw axis has been doing exactly that in every hardware
    run on record (1.9-2.7 sign flips per second, 2026-08-14 onward, with and
    without deadband compensation). This is the one anti-chatter measure that
    does not depend on a model of the actuator: whatever the controller asks
    for, the command that leaves the station moves no faster than the vehicle
    can usefully follow.

    ``prev`` is the previously SENT axis tuple (or None to pass through):
    ``(surge, sway, heave, yaw)`` or, under ``engage.attitude_axes``,
    ``(surge, sway, heave, yaw, roll, pitch)``. Roll/pitch are slewed
    (``rate_rp`` per second, or ``max_rate`` when None) ONLY with the 6-tuple;
    with a 4-tuple they pass through unchanged — never zeroed, so a caller
    that has not opted in cannot silently lose them, and a 4-DoF caller
    (roll = pitch = 0.0) gets today's object back. Returns a new PilotInput;
    the caller keeps the tuple for next tick."""
    if prev is None or not (max_rate > 0.0) or not (dt > 0.0):
        return cmd
    step = float(max_rate) * float(dt)

    def lim(new: float, old: float, st: float = step) -> float:
        return old + max(-st, min(st, float(new) - float(old)))

    roll, pitch = cmd.roll, cmd.pitch
    if len(prev) >= 6:
        st_rp = step if rate_rp is None else float(rate_rp) * float(dt)
        if st_rp > 0.0:
            roll = lim(cmd.roll, prev[4], st_rp)
            pitch = lim(cmd.pitch, prev[5], st_rp)
    return PilotInput(
        surge=lim(cmd.surge, prev[0]), sway=lim(cmd.sway, prev[1]),
        heave=lim(cmd.heave, prev[2]), yaw=lim(cmd.yaw, prev[3]),
        active=cmd.active, source=cmd.source, stamp=cmd.stamp,
        roll=roll, pitch=pitch).clamped()


def axes_to_wrench(cmd: PilotInput, gains: dict):
    """The inverse map: what wrench do we BELIEVE the sent axes realize?

    This — not the raw solver output — is what the EAOB must be told was
    applied (``HwDobMpc.note_applied``): the axis cap, the slew and the
    dropped-or-capped K/M are known actuator limits, and crediting the EAOB
    with force that never went out would surface as a phantom disturbance.
    K = roll * roll_nm, M = pitch * pitch_nm — 0.0 when the fields are 0
    (every 4-DoF command), so a run without ``engage.attitude_axes`` books
    exactly what it did before. RECORD BOUNDARY: under the variant dobmpc's
    ``w_hat[3:5]`` no longer absorbs the whole levelling torque."""
    gx = float(gains.get("surge_n", 60.0))
    gy = float(gains.get("sway_n", 60.0))
    gz = float(gains.get("heave_n", 60.0))
    gn = float(gains.get("yaw_nm", 20.0))
    gk = float(gains.get("roll_nm", ROLL_NM_DEFAULT))
    gm = float(gains.get("pitch_nm", PITCH_NM_DEFAULT))
    return [cmd.surge * gx, cmd.sway * gy, -cmd.heave * gz,
            cmd.roll * gk, cmd.pitch * gm, cmd.yaw * gn]
