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
def rect_setpoint(t, size_x, size_y, speed):
    """``square_setpoint`` generalized to a RECTANGLE (origin corner, CCW).

    The operator enters an x distance and a y distance, so the mission is a
    rectangle of which the square is the special case. The sim's function is
    left VERBATIM above (its hard-coded test samples are what detect drift
    from the simulator); ``test_rect_setpoint_reduces_to_the_square`` pins
    this one against it whenever size_x == size_y, so the generalization
    cannot quietly disagree with the copy it generalizes.
    """
    X, Y = float(size_x), float(size_y)
    P = 2.0 * (X + Y)
    s = (float(speed) * t) % P
    if s < X:
        return (s, 0.0), (1.0, 0.0)
    if s < X + Y:
        return (X, s - X), (0.0, 1.0)
    if s < 2 * X + Y:
        return (2 * X + Y - s, Y), (-1.0, 0.0)
    return (0.0, P - s), (0.0, -1.0)


def make_rect_ref(size_x, size_y, speed, depth, heading_follow, yaw_rate, dt,
                  T_total):
    """``make_square_ref`` for a rectangle. Same stateful-yaw reasoning."""
    tg = np.arange(0.0, T_total + dt, dt)
    yaw_prof = np.zeros(len(tg))
    r_prof = np.zeros(len(tg))
    if heading_follow:
        yc = 0.0
        for i, t in enumerate(tg):
            _, (tx, ty) = rect_setpoint(t, size_x, size_y, speed)
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
            (rx, ry), (tx, ty) = rect_setpoint(t, size_x, size_y, speed)
            p[:, j] = (rx, ry, depth)
            v[:, j] = (speed * tx, speed * ty, 0.0)
        yaw = np.interp(ts, tg, yaw_prof)
        idx = np.clip(np.searchsorted(tg, ts, side="right") - 1, 0, len(tg) - 1)
        return p, yaw, v, r_prof[idx]
    return ref


def rect_corners_world(size_x, size_y, origin_xy=(0.0, 0.0), rot_rad=0.0,
                       mirror_y=False):
    """The four corners of the placed rectangle (world FLU, for display).

    ``mirror_y`` reflects the rectangle about the origin corner's x axis
    BEFORE the rotation — see ``make_square_ref_world`` for why the NED
    wrapper needs it."""
    c, s = float(np.cos(rot_rad)), float(np.sin(rot_rad))
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    my = -1.0 if mirror_y else 1.0
    X, Y = float(size_x), float(size_y)
    return [(c * x - s * (my * y) + ox, s * x + c * (my * y) + oy)
            for x, y in ((0.0, 0.0), (X, 0.0), (X, Y), (0.0, Y))]


def rect_lap_of(t, size_x, size_y, speed):
    per = 2.0 * (float(size_x) + float(size_y))
    if per <= 0 or speed <= 0:
        return 0
    return int((float(speed) * max(0.0, float(t))) // per)


def make_square_ref_world(size, speed, depth, heading_follow, yaw_rate, dt,
                          T_total, origin_xy=(0.0, 0.0), rot_rad=0.0,
                          yaw_fixed=0.0, size_y=None, mirror_y=False):
    """The sim square, placed in the pool: rotate by ``rot_rad`` about the
    origin corner, translate to ``origin_xy`` — still world FLU.

    ``yaw_fixed`` (world FLU rad) is the heading command when
    ``heading_follow`` is False. The sim's generator hard-codes 0 there; the
    wall-tag geometry needs "keep facing the wall" to be a real, non-zero
    world heading, and it must NOT rotate with the square — the whole point
    of the crab square is that the camera keeps its tag.

    ``size_y`` makes it a RECTANGLE (``size`` is then the x side). None keeps
    the square, and with it the VERBATIM sim generator — so a square mission
    still runs the exact code the simulator's numbers came from.

    ``mirror_y`` reflects the generated path about the origin corner's x axis
    before rotating and translating. The sim's square grows into (+x, +y) of
    world FLU, and the NED wrapper's S = diag(1,-1,-1) mirror turns that into
    (+x, -y) of the tag map — i.e. the origin tag came out the corner with
    the LARGEST y, which on the top-down plot (up = +x, right = +y) is the
    bottom RIGHT. The operator asked for the entered tag to be the bottom
    LEFT corner (2026-08-14), so ``set_square_ned`` mirrors here and the
    rectangle grows into (+x, +y) of the MAP. Reflection reverses the sense
    of travel, so the heading profile is negated with it."""
    base = (make_square_ref(size, speed, depth, heading_follow, yaw_rate, dt,
                            T_total)
            if size_y is None or float(size_y) == float(size)
            else make_rect_ref(size, size_y, speed, depth, heading_follow,
                               yaw_rate, dt, T_total))
    return _placed(base, origin_xy, rot_rad, yaw_fixed, heading_follow,
                   mirror_y)


def _placed(base, origin_xy, rot_rad, yaw_fixed, heading_follow, mirror_y):
    """Rigidly place a BASE-frame sampler into the pool: optional y mirror,
    then rotate about the base origin, then translate. Still world FLU.

    ONE definition, because this repo has already paid for the alternative:
    ``reference.py``'s placement header records two occasions on which a
    second copy of mission arithmetic drifted from the first and made an A/B
    comparison measure two different missions. The square, the rectangle and
    the circle are three base curves sharing one placement, so a mission
    entered as "from tag 79" means the same physical thing whichever shape
    is selected.

    The mirror reverses the sense of travel, so the heading profile and the
    yaw rate are negated with it (``my``). ``yaw_fixed`` is a WORLD heading
    and therefore does NOT rotate with the path — the whole point of a crab
    mission is that the camera keeps its tag while the vehicle translates.
    """
    c, s = float(np.cos(rot_rad)), float(np.sin(rot_rad))
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    my = -1.0 if mirror_y else 1.0

    def ref(ts):
        p, yaw, v, r = base(ts)
        p2 = np.empty_like(p)
        v2 = np.empty_like(v)
        p2[0] = c * p[0] - s * (my * p[1]) + ox
        p2[1] = s * p[0] + c * (my * p[1]) + oy
        p2[2] = p[2]
        v2[0] = c * v[0] - s * (my * v[1])
        v2[1] = s * v[0] + c * (my * v[1])
        v2[2] = v[2]
        if heading_follow:
            return p2, my * yaw + rot_rad, v2, my * r
        return p2, np.full(yaw.shape, float(yaw_fixed)), v2, np.zeros(r.shape)
    return ref


def square_corners_world(size, origin_xy=(0.0, 0.0), rot_rad=0.0):
    """The four corners of the placed square (world FLU, for display)."""
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


# --------------------------------------------------------------------------
# LINE: an out-and-back traverse (hardware mission, not from the sim)
# --------------------------------------------------------------------------
def line_profile(t, length, speed, ramp_s):
    """(s, ds) along an out-and-back line of ``length``, at arclength time t.

    Trapezoidal in SPEED with cosine ramps at both ends, deliberately. A pure
    triangle wave reverses velocity in one sample at each turnaround, and this
    repo has already measured what an unrealizable reference costs: the square
    mission's 90-deg corners floor the tracking error near 2 cm no matter how
    the controller is tuned (memory: square-corner-error-floor). A 180-deg
    reversal is the same trap, worse. Ramping means the reference is something
    the vehicle can actually be, so reference-vs-actual measures the
    CONTROLLER rather than the geometry.

    One out-and-back "lap" takes 2 * (length / speed + ramp_s).
    """
    L = float(length)
    v = float(speed)
    tr = max(0.0, float(ramp_s))
    if L <= 0.0 or v <= 0.0:
        return 0.0, 0.0
    # A ramp covers v*tr/2; two of them must fit inside the traverse.
    if v * tr > L:
        tr = L / v
    T_half = tr + L / v                      # one traverse, ramps included
    tt = max(0.0, float(t)) % (2.0 * T_half)
    back = tt >= T_half
    u = tt - T_half if back else tt          # time within this traverse

    if tr <= 1e-9:
        s, ds = min(L, v * u), (v if u < L / v else 0.0)
    elif u < tr:                             # cosine ramp up
        s = v * (u / 2.0 - (tr / (2.0 * np.pi)) * np.sin(np.pi * u / tr))
        ds = v * 0.5 * (1.0 - np.cos(np.pi * u / tr))
    elif u < T_half - tr:                    # cruise
        s = v * tr / 2.0 + v * (u - tr)
        ds = v
    else:                                    # cosine ramp down
        w = u - (T_half - tr)
        s = (L - v * tr / 2.0
             + v * (w / 2.0 + (tr / (2.0 * np.pi)) * np.sin(np.pi * w / tr)))
        ds = v * 0.5 * (1.0 + np.cos(np.pi * w / tr))
    s = min(max(s, 0.0), L)
    return (L - s, -ds) if back else (s, ds)


def make_line_ref_world(length, speed, depth, dt, T_total,
                        origin_xy=(0.0, 0.0), dir_rad=0.0, yaw_fixed=0.0,
                        ramp_s=1.0):
    """Sampler fn(ts) -> (p (3,K), yaw (K,), v (3,K), r (K,)) in world FLU for
    the out-and-back line, same contract as ``make_square_ref_world``.

    Heading is HELD at ``yaw_fixed`` for the whole run — the vehicle crabs
    along the line instead of turning around at each end, which keeps the
    camera pointed at the same patch of floor and keeps the localizer's tag
    set continuous. ``dt``/``T_total`` are accepted for signature parity with
    the square sampler; this profile is closed-form and needs no grid.
    """
    del dt, T_total
    c, s_ = float(np.cos(dir_rad)), float(np.sin(dir_rad))
    ox, oy = float(origin_xy[0]), float(origin_xy[1])

    def ref(ts):
        ts = np.atleast_1d(np.asarray(ts, float))
        K = ts.size
        p = np.empty((3, K))
        v = np.empty((3, K))
        for j, t in enumerate(ts):
            s, ds = line_profile(t, length, speed, ramp_s)
            p[:, j] = (ox + c * s, oy + s_ * s, depth)
            v[:, j] = (c * ds, s_ * ds, 0.0)
        return (p, np.full(K, float(yaw_fixed)), v, np.zeros(K))
    return ref


def line_points_world(length, origin_xy=(0.0, 0.0), dir_rad=0.0):
    """The two endpoints of the placed line (world FLU, for display)."""
    c, s = float(np.cos(dir_rad)), float(np.sin(dir_rad))
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    return [(ox, oy), (ox + c * float(length), oy + s * float(length))]


def line_lap_of(t, length, speed, ramp_s):
    """Completed out-and-back laps at time t."""
    L, v = float(length), float(speed)
    if L <= 0 or v <= 0:
        return 0
    tr = min(max(0.0, float(ramp_s)), L / v)
    return int(max(0.0, float(t)) // (2.0 * (tr + L / v)))


def line_run_time(length, speed, laps, ramp_s):
    L, v = float(length), float(speed)
    if L <= 0 or v <= 0:
        return 0.0
    tr = min(max(0.0, float(ramp_s)), L / v)
    return float(laps) * 2.0 * (tr + L / v)


# --------------------------------------------------------------------------
# CIRCLE: a closed loop whose ENTERED TAG is on the RIM, not at the centre
#
# The operator's framing (2026-08-17): "the tag id you pick is the circle's
# bottom point, not its centre, and you pick the radius". Bottom means the
# bottom of the top-down plot, which draws screen up = +x_ned — so the tag is
# the circle's MINIMUM-x point and the centre sits one radius along +x from
# it. That is the same promise the rectangle already makes (the entered tag is
# its min-x/min-y corner, bottom LEFT of the plot), which is why it is worth
# stating in the same words: what you point at on the floor is where the
# vehicle starts, not a place it merely orbits.
#
# The circle is the one mission in this file with NO unrealizable feature: no
# 90-degree vertex (memory: square-corner-error-floor) and no 180-degree
# reversal (the reason line_profile needs cosine ramps). Curvature is constant,
# so the only speed limit it imposes is the lateral one, v <= sqrt(a_lat * R),
# and path_geometry.speed_profile applies that without a brake ramp anywhere.
# --------------------------------------------------------------------------
def circle_setpoint(t, radius, speed):
    """(point2, tangent2) at arclength s=speed*t around the circle.

    Same contract as ``square_setpoint`` / ``rect_setpoint``: the mission
    STARTS at this frame's origin and runs CCW, so the centre is at
    (radius, 0) and the origin is the rim point at minimum x. With rot_deg 0
    the placement below maps this frame onto the tag map one-for-one, so
    "origin = minimum x" survives into the pool as "the tag is the bottom of
    the drawn circle".

    CCW here, like the square, which after ``mirror_y`` and the NED display
    flip reads CLOCKWISE on the top-down plot for both shapes. A rectangle
    leaves its tag along +x (up the screen); a circle leaves its tag along
    -y (to the left) because that is where its tangent points. Both then go
    round the same way.
    """
    R = float(radius)
    v = float(speed)
    if R <= 0.0 or v <= 0.0:
        return (0.0, 0.0), (0.0, -1.0)
    th = (v * float(t) / R) % (2.0 * np.pi)
    return ((R * (1.0 - np.cos(th)), -R * np.sin(th)),
            (float(np.sin(th)), float(-np.cos(th))))


def make_circle_ref(radius, speed, depth, heading_follow, yaw_rate, dt,
                    T_total):
    """``make_square_ref`` for a circle. Same stateful-yaw reasoning.

    The heading recursion is kept even though a circle's tangent is already
    continuous (nothing here needs smoothing) for one reason that is not
    cosmetic: ``yaw_rate_deg_s`` is a RATE LIMIT the operator set, and a
    circle asks for a constant speed/R rad/s forever. A 0.3 m circle at
    0.2 m/s wants 38 deg/s of it. Running the same slew recursion means a
    heading-following circle that exceeds the limit falls behind visibly and
    identically in the preview and in the live loop, instead of the preview
    promising a turn rate the loop then refuses.
    """
    tg = np.arange(0.0, T_total + dt, dt)
    yaw_prof = np.zeros(len(tg))
    r_prof = np.zeros(len(tg))
    if heading_follow:
        yc = 0.0
        for i, t in enumerate(tg):
            _, (tx, ty) = circle_setpoint(t, radius, speed)
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
            (rx, ry), (tx, ty) = circle_setpoint(t, radius, speed)
            p[:, j] = (rx, ry, depth)
            v[:, j] = (speed * tx, speed * ty, 0.0)
        yaw = np.interp(ts, tg, yaw_prof)
        idx = np.clip(np.searchsorted(tg, ts, side="right") - 1, 0, len(tg) - 1)
        return p, yaw, v, r_prof[idx]
    return ref


def make_circle_ref_world(radius, speed, depth, heading_follow, yaw_rate, dt,
                          T_total, origin_xy=(0.0, 0.0), rot_rad=0.0,
                          yaw_fixed=0.0, mirror_y=False):
    """The circle, placed in the pool — same contract and same placement as
    ``make_square_ref_world`` (they share ``_placed``).

    ``rot_rad`` swings the CENTRE around the entered tag: the tag stays put on
    the rim and the circle rotates about it, so rot 0 puts the centre at
    +x of the tag and the tag at the bottom of the plot.
    """
    base = make_circle_ref(radius, speed, depth, heading_follow, yaw_rate, dt,
                           T_total)
    return _placed(base, origin_xy, rot_rad, yaw_fixed, heading_follow,
                   mirror_y)


def circle_points_world(radius, origin_xy=(0.0, 0.0), rot_rad=0.0, n=96):
    """The placed circle as a closed polyline (world FLU, for display).

    ``mirror_y`` is deliberately not a parameter: mirroring about the entered
    tag's x axis maps this circle onto itself (the centre is ON that axis), so
    it changes only the direction of travel, which a dotted outline does not
    show. The rectangle's helper needs the flag; this one would be lying about
    having used it.
    """
    R = float(radius)
    c, s = float(np.cos(rot_rad)), float(np.sin(rot_rad))
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    cx, cy = ox + c * R, oy + s * R                  # centre, one R along rot
    a = np.linspace(0.0, 2.0 * np.pi, int(max(8, n)), endpoint=False)
    return [(cx + R * float(np.cos(t)), cy + R * float(np.sin(t))) for t in a]


def circle_lap_of(t, radius, speed):
    """Completed circuits at time t (a lap is one full circle)."""
    R, v = float(radius), float(speed)
    if R <= 0 or v <= 0:
        return 0
    return int((v * max(0.0, float(t))) // (2.0 * np.pi * R))


def circle_run_time(radius, speed, laps):
    R, v = float(radius), float(speed)
    if R <= 0 or v <= 0:
        return 0.0
    return float(laps) * 2.0 * np.pi * R / v


# --------------------------------------------------------------------------
# PLACEMENT: where a mission sits in the NED tag map.
#
# ONE definition, shared by every controller. HwDobMpc and HwPid each used to
# carry their own copy of this arithmetic, and the copies drifted twice in two
# days: HwPid had no set_line_ned at all (a PID line reached its start point
# and then did nothing, 2026-08-14), and when the square became a RECTANGLE,
# HwPid kept ignoring size_y — so the operator entered 2.00 x 0.50 m, watched
# the MPC fly a rectangle and the PID fly a 2.00 x 2.00 m square, and the A/B
# comparison the two runs exist for was measuring two different missions.
#
# A controller difference must never be a GEOMETRY difference. Anything new
# that arms a path calls these; the only per-controller freedom is `dt` (the
# sampling grid) and `preview_s` (how far past the end the horizon must stay
# defined — an MPC needs its N*dt, a PID needs nothing).
# --------------------------------------------------------------------------
def place_square_ned(square: dict, origin_ned_xy, yaw_fixed_ned: float,
                     depth_ned: float, dt: float, preview_s: float = 0.0):
    """(world-FLU sampler, scenario dict) for a rectangle placed in NED.

    The NED->FLU mirror (S = diag(1,-1,-1): y flips, angles negate, depth
    z_flu = -z_ned) is applied ONCE, here. ``rot_deg`` is a MAP rotation; 0
    leaves the sides parallel to the tag map's axes and ``mirror_y`` puts the
    origin — the tag the operator entered — at the min-x / min-y corner.
    """
    size = float(square.get("size", 1.0))
    size_y = square.get("size_y")
    size_y = size if size_y in (None, "") else float(size_y)
    speed = float(square.get("speed", 0.12))
    laps = int(square.get("laps", 3))
    follow = bool(square.get("heading_follow", False))
    yaw_rate = float(np.radians(float(square.get("yaw_rate_deg_s", 60.0))))
    rot_ned = float(np.radians(float(square.get("rot_deg", 0.0))))
    T_run = laps * 2.0 * (size + size_y) / max(1e-6, speed)
    fn = make_square_ref_world(
        size=size, speed=speed, depth=-float(depth_ned),
        heading_follow=follow, yaw_rate=yaw_rate, dt=float(dt),
        T_total=T_run + float(preview_s) + 1.0,
        origin_xy=(float(origin_ned_xy[0]), -float(origin_ned_xy[1])),
        rot_rad=-rot_ned, yaw_fixed=-float(yaw_fixed_ned),
        size_y=size_y, mirror_y=True)
    scenario = {"kind": "square", "size": size, "size_y": size_y,
                "speed": speed, "laps": laps, "depth_ned": float(depth_ned),
                "heading_follow": follow,
                "yaw_rate_deg_s": float(square.get("yaw_rate_deg_s", 60.0)),
                "origin_ned": [float(origin_ned_xy[0]),
                               float(origin_ned_xy[1])],
                "rot_deg": float(square.get("rot_deg", 0.0)),
                "yaw_fixed_ned_deg": float(np.degrees(yaw_fixed_ned)),
                "T_run_s": T_run}
    return fn, scenario


def place_line_ned(line: dict, origin_ned_xy, yaw_fixed_ned: float,
                   depth_ned: float, dt: float, preview_s: float = 0.0):
    """(world-FLU sampler, scenario dict) for an OUT-AND-BACK line in NED.

    ``line["dir_deg"]`` is the outbound heading in the NED map frame (0 = +x,
    90 = +y), so "2 m along +y from tag 79" is dir_deg 90. Heading holds at
    ``yaw_fixed_ned`` throughout — the vehicle crabs rather than turning
    around, so the camera keeps the same floor patch and the localizer's tag
    set stays continuous.
    """
    length = float(line.get("length", 1.0))
    speed = float(line.get("speed", 0.05))
    laps = int(line.get("laps", 5))
    ramp_s = float(line.get("ramp_s", 1.0))
    dir_ned = float(np.radians(float(line.get("dir_deg", 90.0))))
    T_run = line_run_time(length, speed, laps, ramp_s)
    fn = make_line_ref_world(
        length=length, speed=speed, depth=-float(depth_ned), dt=float(dt),
        T_total=T_run + float(preview_s) + 1.0,
        origin_xy=(float(origin_ned_xy[0]), -float(origin_ned_xy[1])),
        dir_rad=-dir_ned, yaw_fixed=-float(yaw_fixed_ned), ramp_s=ramp_s)
    scenario = {"kind": "line", "length": length, "speed": speed, "laps": laps,
                "ramp_s": ramp_s,
                "dir_deg": float(line.get("dir_deg", 90.0)),
                "depth_ned": float(depth_ned),
                "origin_ned": [float(origin_ned_xy[0]),
                               float(origin_ned_xy[1])],
                "yaw_fixed_ned_deg": float(np.degrees(yaw_fixed_ned)),
                "T_run_s": T_run}
    return fn, scenario


def place_circle_ned(circle: dict, origin_ned_xy, yaw_fixed_ned: float,
                     depth_ned: float, dt: float, preview_s: float = 0.0):
    """(world-FLU sampler, scenario dict) for a CIRCLE placed in NED.

    ``circle["radius"]`` is the radius and ``origin_ned_xy`` is a point ON the
    rim — the tag the operator entered — not the centre. With ``rot_deg`` 0
    the centre sits one radius along the tag map's +x from it, which puts the
    tag at the circle's minimum x: the BOTTOM of the top-down plot, matching
    the promise the rectangle makes about its own entered tag.

    The same NED->FLU mirror as ``place_square_ned`` (S = diag(1,-1,-1)), and
    the same ``mirror_y=True``, so the circle grows into the MAP's frame the
    way the rectangle does and both are placed by one piece of arithmetic.
    """
    radius = float(circle.get("radius", 0.5))
    speed = float(circle.get("speed", 0.05))
    laps = int(circle.get("laps", 3))
    follow = bool(circle.get("heading_follow", False))
    yaw_rate = float(np.radians(float(circle.get("yaw_rate_deg_s", 60.0))))
    rot_ned = float(np.radians(float(circle.get("rot_deg", 0.0))))
    T_run = circle_run_time(radius, speed, laps)
    fn = make_circle_ref_world(
        radius=radius, speed=speed, depth=-float(depth_ned),
        heading_follow=follow, yaw_rate=yaw_rate, dt=float(dt),
        T_total=T_run + float(preview_s) + 1.0,
        origin_xy=(float(origin_ned_xy[0]), -float(origin_ned_xy[1])),
        rot_rad=-rot_ned, yaw_fixed=-float(yaw_fixed_ned), mirror_y=True)
    scenario = {"kind": "circle", "radius": radius, "speed": speed,
                "laps": laps, "depth_ned": float(depth_ned),
                "heading_follow": follow,
                "yaw_rate_deg_s": float(circle.get("yaw_rate_deg_s", 60.0)),
                "origin_ned": [float(origin_ned_xy[0]),
                               float(origin_ned_xy[1])],
                "rot_deg": float(circle.get("rot_deg", 0.0)),
                "yaw_fixed_ned_deg": float(np.degrees(yaw_fixed_ned)),
                "T_run_s": T_run}
    return fn, scenario
