#!/usr/bin/env python3
"""allocation.py — NED body wrench -> normalized MANUAL_CONTROL axes.

In MANUAL mode ArduSub maps the four MANUAL_CONTROL axes straight through its
mixer to the thrusters, so a wrench command becomes an axis command through a
per-axis gain: ``axis = wrench / gain``, where ``gain`` is the wrench the
vehicle produces at FULL deflection. Roll (K) and pitch (M) have no
MANUAL_CONTROL axis and are DROPPED — the Heavy is passively stable and the
sim's square runs them near zero anyway; this is a documented limitation of
the MANUAL-mode experiment, not an oversight.

The gains in ``config/hw_mpc.yaml`` are [예측] — derived from the T200 curve
and mixer geometry, never measured on this vehicle — until the P4 step
calibration replaces them. An error here is a plant-gain error the DOB-MPC's
w_hat absorbs at DC and plain MPC shows as tracking offset; both outcomes are
part of the experiment's story rather than a safety issue, because the axis
CAP below bounds authority regardless of how wrong the gain is.

Sign map (PilotInput docstring in state.py vs NED/FRD):
    surge  +forward   =  +X_ned
    sway   +starboard =  +Y_ned
    heave  +up        =  -Z_ned      (NED z is DOWN-positive)
    yaw    +CW-from-above = +N_ned
"""

from __future__ import annotations

from ..state import PilotInput, now


def wrench_to_axes(u_ned, gains: dict, cap: float = 0.5,
                   stamp: float | None = None) -> PilotInput:
    """u_ned = [X, Y, Z, K, M, N] (N, N·m). K, M dropped (see module doc)."""
    gx = max(1e-6, float(gains.get("surge_n", 60.0)))
    gy = max(1e-6, float(gains.get("sway_n", 60.0)))
    gz = max(1e-6, float(gains.get("heave_n", 60.0)))
    gn = max(1e-6, float(gains.get("yaw_nm", 20.0)))
    c = max(0.0, min(1.0, float(cap)))

    def lim(v: float) -> float:
        return max(-c, min(c, v))

    return PilotInput(
        surge=lim(float(u_ned[0]) / gx),
        sway=lim(float(u_ned[1]) / gy),
        heave=lim(-float(u_ned[2]) / gz),
        yaw=lim(float(u_ned[5]) / gn),
        active=frozenset(("mpc",)),
        source="mpc",
        stamp=now() if stamp is None else stamp,
    ).clamped()


def axes_to_wrench(cmd: PilotInput, gains: dict):
    """The inverse map: what wrench do we BELIEVE the sent axes realize?

    This — not the raw solver output — is what the EAOB must be told was
    applied (``HwDobMpc.note_applied``): the axis cap and the dropped K/M are
    known actuator limits, and crediting the EAOB with force that never went
    out would surface as a phantom disturbance."""
    gx = float(gains.get("surge_n", 60.0))
    gy = float(gains.get("sway_n", 60.0))
    gz = float(gains.get("heave_n", 60.0))
    gn = float(gains.get("yaw_nm", 20.0))
    return [cmd.surge * gx, cmd.sway * gy, -cmd.heave * gz,
            0.0, 0.0, cmd.yaw * gn]
