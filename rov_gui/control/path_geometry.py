#!/usr/bin/env python3
"""path_geometry.py — the mission as a C1 curve parameterized by ARCLENGTH.

This is the geometry half of MPCC (``mpcc_acados.py`` is the solver half). The
contract it exists to satisfy: an MPCC cost projects the vehicle onto the path
through the path's own TANGENT frame, so the path must have a tangent
everywhere. The mission the operator enters does not — a rectangle's corner
turns 90 degrees at a point, and an out-and-back line reverses through 180.

Three primitives cover every mission this station flies (a CIRCLE is one
``_Arc`` of 2*pi and needs nothing else — it is the only mission with no
unrealizable feature to repair):

``_Line``   a straight run; constant heading, zero curvature.
``_Arc``    a circular fillet that replaces a polygon vertex. Tangent to both
            legs, so heading is continuous ACROSS the corner — this is what
            lets the NMPC horizon see, and plan through, the next side.
``_Pivot``  position stands still while the heading rotates. Zero-radius
            turnaround for an out-and-back line: the vehicle physically must
            stop to reverse, so the honest model of that is a stop, not a
            radius the mission never asked for. Given a nominal arclength so
            the parameterization stays monotonic in s.

Heading is carried EXPLICITLY (each primitive reports psi(s)) rather than
recovered by differentiating x(s), y(s). That is deliberate: at a pivot the
position derivative is zero and any frame built from it is singular, which is
exactly where a naive MPCC blows up. Supplying psi separately makes the
contouring frame well-defined at every s, including a full reversal.

A lap is CLOSED for every mission (a rectangle obviously; a circle likewise;
an out-and-back line returns to its start pointing the way it began), so
``sample`` wraps s by the lap length and ``laps`` only scales the total. On a
circle the heading advances a full 2*pi per lap and drops back at the wrap —
``sample`` unwraps the sequence it returns, so no consumer ever sees the jump.

The speed profile is part of the geometry, not the controller: ``v_max(s)``
is the largest speed the path itself permits, from three limits combined by a
forward/backward pass —
    * the operator's commanded speed,
    * lateral acceleration in a fillet, v <= sqrt(a_lat * R),
    * longitudinal acceleration, so the reference decelerates BEFORE a corner
      and accelerates after instead of stepping.
This is what removes the corner-error floor: the reference stops asking for a
velocity reversal the vehicle cannot deliver (the 2 cm floor measured on the
sharp-cornered square was the reference being unrealizable, not the
controller being badly tuned).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


# --------------------------------------------------------------- primitives
@dataclass(frozen=True)
class _Line:
    x0: float
    y0: float
    psi: float
    length: float

    def at(self, s: float):
        c, sn = math.cos(self.psi), math.sin(self.psi)
        return (self.x0 + c * s, self.y0 + sn * s, self.psi, 0.0)


@dataclass(frozen=True)
class _Arc:
    cx: float
    cy: float
    radius: float
    ang0: float          # angle of the START point as seen from the centre
    turn: float          # signed swept angle (+ = CCW)
    length: float

    def at(self, s: float):
        f = 0.0 if self.length <= 0 else s / self.length
        a = self.ang0 + self.turn * f
        sgn = 1.0 if self.turn >= 0 else -1.0
        # heading is the tangent of the circle, 90 deg off the radius
        return (self.cx + self.radius * math.cos(a),
                self.cy + self.radius * math.sin(a),
                a + sgn * math.pi / 2.0,
                sgn / max(1e-9, self.radius))


@dataclass(frozen=True)
class _Pivot:
    x0: float
    y0: float
    psi0: float
    turn: float          # signed heading change
    length: float        # nominal arclength so s stays monotonic

    def at(self, s: float):
        f = 0.0 if self.length <= 0 else s / self.length
        # Curvature is reported as inf so the speed profile pins v = 0 here.
        return (self.x0, self.y0, self.psi0 + self.turn * f, math.inf)


class ArcPath:
    """One lap of primitives + the arclength bookkeeping over ``laps`` of them."""

    def __init__(self, primitives, laps: int, closed: bool = True):
        prims = [p for p in primitives if p.length > 0.0]
        if not prims:
            raise ValueError("path has no primitives with positive length")
        self.prims = tuple(prims)
        self.laps = max(1, int(laps))
        self.closed = bool(closed)
        self._edges = np.concatenate((
            np.zeros(1), np.cumsum([p.length for p in self.prims])))
        self.lap_length = float(self._edges[-1])
        self.total_length = self.lap_length * self.laps

    # ------------------------------------------------------------- sampling
    def _at(self, s: float):
        """(x, y, psi, kappa) at arclength s WITHIN one lap."""
        i = int(np.searchsorted(self._edges, s, side="right") - 1)
        i = min(max(i, 0), len(self.prims) - 1)
        return self.prims[i].at(min(max(s - self._edges[i], 0.0),
                                    self.prims[i].length))

    def sample(self, s_abs):
        """(x, y, psi_unwrapped, kappa) at absolute arclength(s).

        Defined for EVERY s, including outside [0, total_length], by extending
        the terminal tangents as straight rays. That is not decoration: the
        MPCC window straddles the start (it deliberately reaches ``BACK_M``
        behind the vehicle so the projection is defined when the vehicle is
        late), and the obvious alternative — clamping s into range — makes the
        sampled path FLAT over the out-of-range part. A flat stretch has
        dp/dtheta = 0, and theta0 sits exactly at its edge, so the optimizer
        sees no first-order benefit from advancing theta at all: the mission
        never starts, silently, with a healthy solver status. A straight
        extension keeps |dp/dtheta| = 1 everywhere.

        ``psi`` is UNWRAPPED along the returned sequence — it is handed to a
        B-spline, and a 2*pi jump inside the horizon would be a false
        360-degree turn command."""
        s_abs = np.atleast_1d(np.asarray(s_abs, float))
        out = np.empty((4, s_abs.size))
        for j, s in enumerate(s_abs):
            s = float(s)
            if s < 0.0 or s > self.total_length:
                edge = 0.0 if s < 0.0 else self.total_length
                over = s - edge
                x0, y0, psi0, _k = self._at(
                    0.0 if s < 0.0 else self._wrapped(self.total_length))
                out[:, j] = (x0 + over * math.cos(psi0),
                             y0 + over * math.sin(psi0), psi0, 0.0)
                continue
            out[:, j] = self._at(self._wrapped(s))
        out[2] = np.unwrap(out[2])
        return out[0], out[1], out[2], out[3]

    def _wrapped(self, s: float) -> float:
        """Absolute arclength -> arclength within one lap."""
        if not self.closed:
            return min(s, self.lap_length)
        if s >= self.total_length:
            return self.lap_length
        return s % self.lap_length

    # -------------------------------------------------------- speed profile
    def speed_profile(self, v_cmd: float, a_lat: float, a_long: float,
                      ds: float = 0.02, v_creep: float = 0.02):
        """``(s_grid, v_grid)`` — the fastest FEASIBLE speed along the path.

        Three limits, combined the way a racing line is: the curvature limit
        first (a point-wise ceiling), then a forward pass so the reference can
        only speed up as fast as ``a_long`` allows, then a backward pass so it
        is already slow when it ARRIVES at a corner. That backward pass is what
        makes the corner realizable, and it only works because the horizon can
        see the corner coming — the two halves of this fix are the same fix.

        ``v_creep`` is a deadlock guard, and it is subtle enough to spell out.
        v_ref is the target for theta_dot, so wherever v_ref is exactly zero
        the optimizer's own solution is "do not advance", the horizon never
        leaves that point, and the mission never starts. That is harmless at
        the very end (the run IS over) but fatal at the start and at a
        ``_Pivot``. A pivot's POSITION is stationary, so letting theta creep
        through it costs no spatial motion at all — it just spends
        ``PIVOT_LEN_M / v_creep`` seconds turning, which is the turnaround
        dwell. The floor is therefore applied AFTER both passes and only where
        it cannot move the vehicle: pivots, and the first sample."""
        v_cmd = max(1e-3, float(v_cmd))
        a_lat = max(1e-3, float(a_lat))
        a_long = max(1e-3, float(a_long))
        v_creep = max(1e-4, float(v_creep))
        n = max(8, int(round(self.total_length / max(1e-3, ds))) + 1)
        s = np.linspace(0.0, self.total_length, n)
        _x, _y, psi, k = self.sample(s)
        pivot = ~np.isfinite(k)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.minimum(v_cmd, np.sqrt(a_lat / np.abs(k)))
        v = np.nan_to_num(v, nan=v_cmd, posinf=v_cmd)
        v[pivot] = 0.0                               # brake INTO a reversal
        # ...and brake into a CORNER the same way, when there is no fillet to
        # take it at speed. A vertex left sharp turns the heading through 90
        # degrees over zero arclength; the only reference that is realizable
        # there is one that arrives stopped, which is also what makes the
        # 90-degree change continuous in VELOCITY (the direction rotates while
        # |v| = 0). Detected from the sampled heading rather than from the
        # primitive list, so it holds for any polygon: on an arc the heading
        # turns by kappa*ds per sample, so anything far above that is a corner
        # and not curvature.
        ds_grid = float(s[1] - s[0])
        turn = np.abs(np.diff(np.unwrap(psi)))
        smooth = np.abs(np.nan_to_num(k[:-1], posinf=0.0, nan=0.0)) * ds_grid
        corner = np.zeros(n, bool)
        corner[:-1] |= turn > np.maximum(4.0 * smooth, math.radians(5.0))
        corner[1:] |= corner[:-1]
        v[corner] = 0.0
        v[0] = 0.0                                   # start from rest
        v[-1] = 0.0                                  # and finish at rest
        step = s[1] - s[0]
        for i in range(1, n):                        # forward: accel limit
            v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2.0 * a_long * step))
        for i in range(n - 2, -1, -1):               # backward: brake limit
            v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * a_long * step))
        v[pivot] = v_creep                           # ...then turn on the spot
        v[0] = max(v[0], v_creep)                    # ...and be able to leave
        # A corner gets the SAME creep floor as a pivot, and for the same
        # reason: v_ref is what advances the cursor, so an exact zero anywhere
        # the mission still has distance to cover is a fixed point the
        # reference never leaves — the run would stop dead at the first vertex
        # and wait forever. The braking ramp above is computed toward zero, so
        # the reference still arrives essentially stopped; the floor only lets
        # it cross. At 0.02 m/s the 90-degree direction change is a 2.8 cm/s
        # step in the velocity reference, which is below the vehicle's own
        # velocity noise.
        v[corner] = np.maximum(v[corner], v_creep)
        return s, v


# ------------------------------------------------------------- construction
def _fillet(prev_xy, vertex, next_xy, radius: float):
    """The arc that rounds ``vertex``, plus how much it eats off each leg.

    Returns ``(arc, trim)`` where ``trim`` is the distance removed from BOTH
    the incoming and outgoing legs, or ``(None, 0.0)`` when the turn is too
    tight or too straight to round."""
    p, v, q = (np.asarray(t, float) for t in (prev_xy, vertex, next_xy))
    u_in = v - p
    u_out = q - v
    n_in, n_out = np.linalg.norm(u_in), np.linalg.norm(u_out)
    if not (n_in > 1e-9 and n_out > 1e-9) or radius <= 0.0:
        return None, 0.0
    u_in, u_out = u_in / n_in, u_out / n_out
    cos_t = float(np.clip(np.dot(u_in, u_out), -1.0, 1.0))
    turn = math.atan2(float(u_in[0] * u_out[1] - u_in[1] * u_out[0]), cos_t)
    if abs(turn) < 1e-6 or abs(abs(turn) - math.pi) < 1e-6:
        return None, 0.0                      # straight through, or a reversal
    trim = radius / math.tan((math.pi - abs(turn)) / 2.0)
    trim = min(trim, 0.49 * n_in, 0.49 * n_out)
    if trim <= 1e-6:
        return None, 0.0
    r_eff = trim * math.tan((math.pi - abs(turn)) / 2.0)
    start = v - trim * u_in
    sgn = 1.0 if turn > 0 else -1.0
    # centre sits on the inward normal of the incoming leg
    normal = np.array([-u_in[1], u_in[0]]) * sgn
    centre = start + r_eff * normal
    ang0 = math.atan2(start[1] - centre[1], start[0] - centre[0])
    return _Arc(float(centre[0]), float(centre[1]), float(r_eff),
                float(ang0), float(turn), float(r_eff * abs(turn))), float(trim)


def _polygon(points, fillet_m: float, closed: bool, laps: int) -> ArcPath:
    """Straights + fillets through ``points`` (already in order)."""
    pts = [np.asarray(p, float)[:2] for p in points]
    n = len(pts)
    arcs: dict[int, tuple] = {}
    idx = range(n) if closed else range(1, n - 1)
    for i in idx:
        arc, trim = _fillet(pts[(i - 1) % n], pts[i], pts[(i + 1) % n],
                            fillet_m)
        if arc is not None:
            arcs[i] = (arc, trim)
    prims: list = []
    span = range(n) if closed else range(n - 1)
    for i in span:
        a, b = pts[i], pts[(i + 1) % n]
        d = b - a
        leg = float(np.linalg.norm(d))
        if leg <= 1e-9:
            continue
        psi = math.atan2(float(d[1]), float(d[0]))
        cut0 = arcs[i][1] if i in arcs else 0.0
        j = (i + 1) % n
        cut1 = arcs[j][1] if j in arcs else 0.0
        start = a + (d / leg) * cut0
        length = leg - cut0 - cut1
        if length > 1e-9:
            prims.append(_Line(float(start[0]), float(start[1]), psi, length))
        if j in arcs and (closed or j != n - 1):
            prims.append(arcs[j][0])
    return ArcPath(prims, laps=laps, closed=closed)


@dataclass(frozen=True)
class NedPlan:
    """One reference plan in world NED, K stages deep.

    The field names are not free: ``HwDobMpc._xref_ned_plan`` and
    ``HwPid.set_path_plan_ned`` already unpack exactly these, and both have
    been sitting unused since their only producer was deleted. K comes from the
    controller itself (``path_plan_steps``), so the SAME plan serves a PID that
    wants one stage and an NMPC that wants its whole horizon — which is what
    keeps "PID vs MPC" a controller comparison instead of a geometry one."""

    p_ned: np.ndarray       # (3, K)
    yaw_ned: np.ndarray     # (K,)
    v_ned: np.ndarray       # (3, K)   world frame
    r_ned: np.ndarray       # (K,)
    # The PATH's own tangent heading at each stage, ALWAYS the geometry's —
    # never the vehicle's. It is what ``yaw_ned`` would be under
    # heading_follow, and it stays populated when heading_follow is False,
    # which is the case that needs it: a crab square holds one vehicle
    # heading while the path turns 90 degrees under it, so the two angles
    # are unrelated. A cost that wants to split error into along-track and
    # cross-track (mpc_tuned, path_cost.py) needs THIS angle; deriving it
    # from ``v_ned`` fails wherever the speed profile brakes to zero, i.e.
    # exactly at the corners the split exists for.
    psi_path: np.ndarray = None     # (K,)

    def __post_init__(self):
        if self.psi_path is None:
            # Legacy construction (4 positional args). Fall back to the
            # heading, which is correct under heading_follow and the best
            # available guess otherwise — but say so is impossible in a
            # frozen dataclass, so producers in this repo always pass it.
            object.__setattr__(self, "psi_path",
                               np.asarray(self.yaw_ned, float).copy())


class PathCursor:
    """Monotonic projection onto an :class:`ArcPath`, for controllers that
    cannot optimize their own progress.

    The MPCC solves for theta; a PID cannot, so it gets this: project the hull
    onto the curve, never let progress run backwards, and hand it a setpoint
    ``lead_m`` further along. Deliberately the SAME ``ArcPath`` the MPCC flies
    — a controller difference must never become a geometry difference, which
    this repo has already paid for three times (reference.py's placement
    header). The search is LOCAL (a window around the current theta) so a
    self-crossing or doubled-back path cannot snap the cursor to the wrong
    lap."""

    def __init__(self, path: ArcPath, lead_m: float, s_grid, v_grid,
                 search_m: float = 0.60, ds: float = 0.01):
        self.path = path
        self.lead_m = max(0.0, float(lead_m))
        self._s = np.asarray(s_grid, float)
        self._v = np.asarray(v_grid, float)
        # THE SEARCH WINDOW IS BOUNDED BY THE LAP, not by a fixed 0.60 m.
        #
        # ``step`` takes the argmin over [theta, theta + search_m] and then
        # never lets theta go back, so whatever that argmin picks is permanent
        # progress. On a path whose lap is comparable to the window, the
        # nearest point in the window can be nowhere near the vehicle's actual
        # position along the curve, and the cursor jumps most of a lap in one
        # tick — dragging the leashed setpoint with it, which is a lunge.
        #
        # The shape that makes this reachable is the CIRCLE: its projection is
        # radial, so a hull displaced toward the CENTRE has no well-defined
        # arclength at all, and every point of the window is nearly the same
        # distance away. Measured on a 0.15 m circle (lap 0.94 m) with the old
        # fixed window: one tick took theta from 0 to 0.600 m — 64 % of a lap
        # — with the hull 0.14 m off the path
        # (test_cursor_cannot_jump_most_of_a_short_lap_in_one_tick).
        #
        # A quarter lap bounds the damage; the floor keeps the window several
        # ticks of hull motion wide (0.10 m is 4 s of travel at the shipped
        # 0.05 m/s and 40 ticks at the panel's 0.50 m/s ceiling), so the
        # projection can still keep up with anything the vehicle can do. On
        # every mission flown so far the lap is long enough that this changes
        # nothing.
        lap = float(getattr(path, "lap_length", 0.0) or 0.0)
        cap = max(0.10, 0.25 * lap) if lap > 0.0 else float(search_m)
        self.search_m = float(min(float(search_m), cap))
        self.ds = float(ds)
        self.theta = 0.0            # the HULL's progress (projection)
        self._theta_cmd = 0.0       # the SETPOINT's progress (see step)

    @property
    def target_theta(self) -> float:
        return float(self._theta_cmd)

    @property
    def complete(self) -> bool:
        return self.theta >= self.path.total_length - 1e-3

    def lap(self) -> int:
        if self.path.lap_length <= 0:
            return 0
        return min(self.path.laps,
                   int(self.theta / self.path.lap_length + 1e-9))

    def plan(self, steps: int, dt: float, depth: float, yaw_fixed: float,
             heading_follow: bool) -> "NedPlan":
        """Roll the reference forward ``steps`` stages -> :class:`NedPlan`.

        This is what gives a tracking NMPC its corners back. Until now the
        worker handed the controller a single setpoint every tick, and
        ``set_target_ned`` clears the trajectory sampler, so the NMPC fell back
        to extrapolating that one point along a STRAIGHT RAY for all 61 stages
        — it drove through every corner without knowing one was there.

        Integration starts at the LEASHED cursor ``_theta_cmd``, so a vehicle
        that has fallen behind still cannot have its reference walk away, and
        each stage advances by the path's own feasible speed. Where that speed
        is zero (an un-filleted vertex) the plan simply stops there and every
        remaining stage repeats the vertex — which is setpoint regulation at
        the corner, arrived at continuously rather than by switching modes.

        ``steps == 1`` reproduces exactly the setpoint the previous code sent,
        so the PID is unchanged by construction."""
        k = max(1, int(steps))
        dt = max(1e-6, float(dt))
        th = float(self._theta_cmd)
        p = np.empty((3, k))
        v = np.empty((3, k))
        yaw = np.empty(k)
        psi_p = np.empty(k)         # the PATH tangent, whatever the heading is
        for i in range(k):
            x, y, psi, _kap = self.path.sample(np.array([th]))
            vr = float(np.interp(th, self._s, self._v))
            c, sn = math.cos(psi[0]), math.sin(psi[0])
            p[:, i] = (x[0], y[0], float(depth))
            v[:, i] = (vr * c, vr * sn, 0.0)
            yaw[i] = float(psi[0]) if heading_follow else float(yaw_fixed)
            psi_p[i] = float(psi[0])
            th = min(self.path.total_length, th + vr * dt)
        # Yaw rate is the reference's own turn rate, and it is zero unless the
        # mission asks the vehicle to face along the path.
        r = np.zeros(k)
        if heading_follow and k > 1:
            r[:-1] = np.diff(np.unwrap(yaw)) / dt
            r[-1] = r[-2]
        return NedPlan(p, yaw, v, r, np.unwrap(psi_p))

    def step(self, p_ned, dt: float):
        """(target_ned, psi_path, e_along(LAG), e_cross, v_ref).

        TWO cursors, and conflating them is what made the operator's speed box
        do nothing (measured 2026-08-17: 0.2 m/s commanded, 0.026 m/s of path
        actually covered — sessions/low_level_controller_data/20260817/
        0817_103431/mpc_103624.csv).

        ``theta`` is where the HULL is: the monotonic projection, and the
        honest answer to "how far along are we".
        ``_theta_cmd`` is where the SETPOINT is: it advances at the path's own
        feasible speed v_ref, i.e. the operator's number, and is then leashed
        to within ``lead_m`` of the projection so a vehicle that falls behind
        is not abandoned. Advancing the setpoint by projection alone — which
        is what this did before — makes the reference a carrot held a fixed
        10 cm ahead, so the speed the vehicle settles at is whatever kp*0.10
        happens to produce against drag, and the commanded speed is not in the
        loop at all."""
        p = np.asarray(p_ned, float)
        n = max(2, int(self.search_m / max(1e-3, self.ds)) + 1)
        cand = np.clip(self.theta + np.linspace(0.0, self.search_m, n),
                       0.0, self.path.total_length)
        cx, cy, _psi, _k = self.path.sample(cand)
        j = int(np.argmin((cx - p[0]) ** 2 + (cy - p[1]) ** 2))
        self.theta = float(max(self.theta, cand[j]))

        v_ref = float(np.interp(self._theta_cmd, self._s, self._v))
        cmd = self._theta_cmd + v_ref * max(0.0, float(dt))
        cmd = min(cmd, self.theta + self.lead_m)      # leash: never abandon it
        self._theta_cmd = float(min(self.path.total_length,
                                    max(self.theta, cmd)))

        px, py, psi, _k2 = self.path.sample(np.array([self.theta]))
        dx, dy = float(p[0]) - px[0], float(p[1]) - py[0]
        s_p, c_p = math.sin(psi[0]), math.cos(psi[0])
        tx, ty, tpsi, _k3 = self.path.sample(np.array([self._theta_cmd]))
        # along is LAG (positive = behind the target), cross is positive to the
        # LEFT of travel — workers._path_split's conventions.
        return (np.array([tx[0], ty[0]]), float(tpsi[0]),
                -(c_p * (float(p[0]) - tx[0]) + s_p * (float(p[1]) - ty[0])),
                s_p * dx - c_p * dy, v_ref)


PIVOT_LEN_M = 0.02      # nominal arclength of an in-place 180 (see _Pivot)


def path_from_scenario(scenario: dict, fillet_m: float = 0.15,
                       turn_radius_m: float = 0.0) -> ArcPath:
    """Build the C1 mission path from ``place_*_ned``'s resolved scenario.

    ``fillet_m`` rounds rectangle corners. ``turn_radius_m`` optionally rounds
    an out-and-back line's turnarounds into a stadium; 0 (the default) keeps
    the mission's exact geometry and reverses in place through a ``_Pivot``,
    which is what the operator asked for when they typed a LINE. Neither knob
    touches a CIRCLE: it is already C1 everywhere, which is the whole reason
    it is worth flying."""
    kind = str(scenario.get("kind", "")).lower()
    if kind not in ("line", "square", "circle"):
        raise ValueError(f"MPCC path needs line|square|circle, got {kind!r}")
    laps = max(1, int(scenario.get("laps", 1)))
    ox, oy = (float(scenario["origin_ned"][0]),
              float(scenario["origin_ned"][1]))

    if kind == "circle":
        radius = float(scenario.get("radius", 0.0))
        if not radius > 0.0:
            raise ValueError("circle radius must be positive")
        # ONE primitive for the whole lap: the mission has no vertex to fillet
        # and no reversal to pivot through, so it needs neither. Constant
        # curvature 1/R also means speed_profile's only active limit is the
        # lateral one — v <= sqrt(a_lat * R) everywhere, with no braking ramp,
        # because there is nothing to brake INTO.
        #
        # ORIGIN IS ON THE RIM, not at the centre (operator, 2026-08-17): the
        # entered tag is the start point and the centre sits one radius along
        # the rot_deg direction, so rot 0 puts the tag at the circle's min x —
        # the BOTTOM of the top-down plot, matching what the rectangle
        # promises about its own entered tag. Travel is CCW in this NED frame,
        # the same sense the rectangle runs (o -> +x -> +x+y -> +y).
        rot = math.radians(float(scenario.get("rot_deg", 0.0)))
        cx, cy = ox + radius * math.cos(rot), oy + radius * math.sin(rot)
        return ArcPath([_Arc(cx, cy, radius, _wrap(rot + math.pi),
                             2.0 * math.pi, 2.0 * math.pi * radius)],
                       laps=laps, closed=True)

    if kind == "square":
        sx = float(scenario.get("size", 0.0))
        sy = float(scenario.get("size_y", sx) or sx)
        if not (sx > 0.0 and sy > 0.0):
            raise ValueError("rectangle sides must be positive")
        rot = math.radians(float(scenario.get("rot_deg", 0.0)))
        ux = np.array([math.cos(rot), math.sin(rot)])
        uy = np.array([-math.sin(rot), math.cos(rot)])
        o = np.array([ox, oy])
        corners = [o, o + sx * ux, o + sx * ux + sy * uy, o + sy * uy]
        # A fillet may not eat more than half the shorter side.
        return _polygon(corners, min(float(fillet_m), 0.45 * min(sx, sy)),
                        closed=True, laps=laps)

    length = float(scenario.get("length", 0.0))
    if not length > 0.0:
        raise ValueError("line length must be positive")
    d = math.radians(float(scenario.get("dir_deg", 90.0)))
    u = np.array([math.cos(d), math.sin(d)])
    a = np.array([ox, oy])
    b = a + length * u
    r = max(0.0, float(turn_radius_m))
    if r <= 1e-6:
        # Exact out-and-back: two straights joined by in-place reversals.
        prims = [_Line(float(a[0]), float(a[1]), d, length),
                 _Pivot(float(b[0]), float(b[1]), d, math.pi, PIVOT_LEN_M),
                 _Line(float(b[0]), float(b[1]), _wrap(d + math.pi), length),
                 _Pivot(float(a[0]), float(a[1]), _wrap(d + math.pi),
                        math.pi, PIVOT_LEN_M)]
        return ArcPath(prims, laps=laps, closed=True)
    # Stadium: the turnarounds become semicircles, so the vehicle never stops.
    nrm = np.array([-u[1], u[0]])
    a2, b2 = a + r * nrm, b + r * nrm
    prims = [
        _Line(float(a[0]), float(a[1]), d, length),
        _Arc(float(b[0] + r * nrm[0]), float(b[1] + r * nrm[1]), r,
             math.atan2(-nrm[1], -nrm[0]), math.pi, math.pi * r),
        _Line(float(b2[0]), float(b2[1]), _wrap(d + math.pi), length),
        _Arc(float(a[0] + r * nrm[0]), float(a[1] + r * nrm[1]), r,
             math.atan2(nrm[1], nrm[0]), math.pi, math.pi * r),
    ]
    return ArcPath(prims, laps=laps, closed=True)
