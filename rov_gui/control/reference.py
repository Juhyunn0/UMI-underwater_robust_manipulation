#!/usr/bin/env python3
"""reference.py — the square trajectory, copied from the simulator.

``square_setpoint`` / ``slew_heading`` / ``make_square_ref`` are VERBATIM
copies of ``bluerov2_mujoco_marinegym/experiments/run_compare.py:114-177``
(2026-08-12). Copied, not imported, because importing run_compare drags in
matplotlib + mujoco — neither belongs in the control station's process — and
the three functions are pure numpy. ``tests/test_control.py`` pins their
output against hard-coded samples so the copy cannot drift silently.

Everything here speaks the SIM's mission convention: world FLU (x fwd, y
left, z up), CCW square with the origin as a corner, yaw positive CCW. The
hardware world is NED; ``mpc_bridge.HwDobMpc.set_square_ned`` owns the mirror
(S = diag(1,-1,-1)) so the sign juggling lives in exactly one place, next to
the frames module that defines it.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# VERBATIM from experiments/run_compare.py:114 (see module docstring)
# --------------------------------------------------------------------------
def square_setpoint(t, size, speed):
    """(point2, tangent2) at arclength s=speed*t around the CCW square (origin corner)."""
    P = 4.0 * size
    s = (speed * t) % P
    S = size
    if s < S:
        return (s, 0.0), (1.0, 0.0)
    if s < 2 * S:
        return (S, s - S), (0.0, 1.0)
    if s < 3 * S:
        return (3 * S - s, S), (-1.0, 0.0)
    return (0.0, 4 * S - s), (0.0, -1.0)


# VERBATIM from experiments/run_compare.py:128
def slew_heading(yaw_ref, tx, ty, rate_rad, dt):
    """Slew the held yaw reference toward the path tangent atan2(ty,tx) at most
    `rate_rad` per second (shortest angle). Smooths the 90-deg corner steps so the
    heading reference is continuous -- the ROV faces its travel direction without an
    instantaneous jump. The POSITION path stays the sharp square (heading only)."""
    target = np.arctan2(ty, tx)
    d = np.arctan2(np.sin(target - yaw_ref), np.cos(target - yaw_ref))   # wrap (-pi,pi]
    step = rate_rad * dt
    return yaw_ref + float(np.clip(d, -step, step))


# VERBATIM from experiments/run_compare.py:139
def make_square_ref(size, speed, depth, heading_follow, yaw_rate, dt, T_total):
    """Mission-side half of the MPC reference preview: a sampler
    fn(ts) -> (p (3,K), yaw (K,), v (3,K), r (K,)) in world FLU for the square
    mission, handed to DOBMPCController.set_reference_traj (tracking mode; the
    PID and the DP scenario keep the plain set_target interface).

    Position/velocity are exact square_setpoint evaluations at each requested
    time. The heading command is stateful (slew_heading recursion), but it has no
    vehicle feedback, so its future is knowable: precompute the profile on the
    physics-dt grid with the SAME recursion the live loop runs -- sampled future
    yaw == the yaw_cmd the loop will command when that time arrives. Pass
    T_total >= run duration + MPC horizon so end-of-run sampling stays in range
    (beyond the grid the profile holds its last value)."""
    tg = np.arange(0.0, T_total + dt, dt)
    yaw_prof = np.zeros(len(tg))
    r_prof = np.zeros(len(tg))
    if heading_follow:
        yc = 0.0
        for i, t in enumerate(tg):
            _, (tx, ty) = square_setpoint(t, size, speed)
            yn = slew_heading(yc, tx, ty, yaw_rate, dt)
            r_prof[i] = (yn - yc) / dt
            yaw_prof[i] = yn
            yc = yn

    def ref(ts):
        ts = np.atleast_1d(np.asarray(ts, float))
        K = ts.size
        p = np.empty((3, K))
        v = np.empty((3, K))
        for j, t in enumerate(ts):
            (rx, ry), (tx, ty) = square_setpoint(t, size, speed)
            p[:, j] = (rx, ry, depth)
            v[:, j] = (speed * tx, speed * ty, 0.0)
        yaw = np.interp(ts, tg, yaw_prof)              # continuous, accumulates CCW
        idx = np.clip(np.searchsorted(tg, ts, side="right") - 1, 0, len(tg) - 1)
        r = r_prof[idx]                                # piecewise-constant: no interp
        return p, yaw, v, r
    return ref


# --------------------------------------------------------------------------
# hardware additions (NOT from the sim)
# --------------------------------------------------------------------------
def make_square_ref_world(size, speed, depth, heading_follow, yaw_rate, dt,
                          T_total, origin_xy=(0.0, 0.0), rot_rad=0.0,
                          yaw_fixed=0.0):
    """The sim square, placed in the pool: rotate by ``rot_rad`` about the
    origin corner, translate to ``origin_xy`` — still world FLU.

    ``yaw_fixed`` (world FLU rad) is the heading command when
    ``heading_follow`` is False. The sim's generator hard-codes 0 there; the
    wall-tag geometry needs "keep facing the wall" to be a real, non-zero
    world heading, and it must NOT rotate with the square — the whole point
    of the crab square is that the camera keeps its tag."""
    base = make_square_ref(size, speed, depth, heading_follow, yaw_rate, dt,
                           T_total)
    c, s = float(np.cos(rot_rad)), float(np.sin(rot_rad))
    ox, oy = float(origin_xy[0]), float(origin_xy[1])

    def ref(ts):
        p, yaw, v, r = base(ts)
        p2 = np.empty_like(p)
        v2 = np.empty_like(v)
        p2[0] = c * p[0] - s * p[1] + ox
        p2[1] = s * p[0] + c * p[1] + oy
        p2[2] = p[2]
        v2[0] = c * v[0] - s * v[1]
        v2[1] = s * v[0] + c * v[1]
        v2[2] = v[2]
        if heading_follow:
            return p2, yaw + rot_rad, v2, r
        return p2, np.full(yaw.shape, float(yaw_fixed)), v2, np.zeros(r.shape)
    return ref


def square_corners_world(size, origin_xy=(0.0, 0.0), rot_rad=0.0):
    """The four corners of the placed square (world FLU, for display/geofence)."""
    c, s = float(np.cos(rot_rad)), float(np.sin(rot_rad))
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    out = []
    for x, y in ((0.0, 0.0), (size, 0.0), (size, size), (0.0, size)):
        out.append((c * x - s * y + ox, s * x + c * y + oy))
    return out


def lap_of(t, size, speed):
    """Lap counter, same definition as run_compare (perimeter = 4*size)."""
    if size <= 0 or speed <= 0:
        return 0
    return int((speed * max(0.0, float(t))) // (4.0 * size))
