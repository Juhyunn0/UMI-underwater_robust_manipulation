#!/usr/bin/env python3
"""trajectory.py — the MPC panel: engage controls + reference-vs-actual plot.

Embedded in the main grid (window.py) — since 2026-08-14 in the BOTTOM ROW
spanning columns 2-3 (the old PROPULSION/SENSORS slots, which moved to column
3 under SYSTEM HEALTH), per operator request: the plot is the thing watched
during an experiment, so it gets the wide slot.

The plot is a top-down map of the NED tag-world (screen up = +x_ned, screen
right = +y_ned — the frame the pool is measured in), drawn with the house
QPainter pattern (no pyqtgraph/matplotlib, deliberately — indicators.py:5).
It shows the pool boundary, the placed reference square, the reference and
actual trails, and the vehicle's heading. (The geofence box it used to draw on
engage was removed on 2026-08-14 — operator request; see control/workers.py
for what that cost.) Everything numeric it prints comes
FROM MpcStatus/NavFix — this window computes nothing, so the plot and the CSV
can never disagree.

Engage discipline mirrors the ARM button: ENGAGE is a hold-to-confirm (the
vehicle starts holding position the moment it fires), DISENGAGE and STOP are
single clicks — a stop must never be gated (teleop.py's rule).
"""

from __future__ import annotations

import math
import time
from collections import deque

from .. import theme
from ..qt import QColor, QPainter, Qt, QtCore, QtGui, QtWidgets, Signal
from ..control.geometry import SHAPES
from ..state import MpcStatus, NavFix, ObjectFix
from .indicators import HoldToConfirmButton

TRAIL_MAX = 4800          # hard cap on stored points (memory bound)
TRAIL_AGE_S = 90.0        # points older than this fade out and are dropped


def _c(hex_str: str, alpha: int = 255) -> QColor:
    col = QColor(hex_str)
    col.setAlpha(alpha)
    return col


# Viridis anchor points (matplotlib's default perceptual ramp), so the GUI
# trail and the offline plot (rov_gui/tools/plot_nav_run.py, real viridis)
# read the same: dark purple = oldest, bright yellow = newest.
_VIRIDIS = ((68, 1, 84), (59, 82, 139), (33, 145, 140),
            (94, 201, 98), (253, 231, 37))


def _age_color(u: float, alpha: int = 255) -> QColor:
    """u in [0, 1]: 0 = oldest (about to disappear), 1 = newest."""
    u = max(0.0, min(1.0, u))
    seg = u * (len(_VIRIDIS) - 1)
    i = min(int(seg), len(_VIRIDIS) - 2)
    f = seg - i
    a, b = _VIRIDIS[i], _VIRIDIS[i + 1]
    col = QColor(int(a[0] + f * (b[0] - a[0])),
                 int(a[1] + f * (b[1] - a[1])),
                 int(a[2] + f * (b[2] - a[2])))
    # old points also fade toward transparent so the tail vanishes smoothly
    col.setAlpha(int(alpha * (0.25 + 0.75 * u)))
    return col


class TrajectoryView(QtWidgets.QWidget):
    """Top-down NED map. Fixed world, movable eye (pan/zoom), like pose3d."""

    def __init__(self):
        super().__init__()
        self.map_tags: list = []                # [(x, y, yaw, id, inst), ...]
        self.tag_size_m = 0.170                 # drawn edge; window sets from cfg
        # The vehicle's real footprint (along-heading, across), metres. Drawn
        # as an oriented rectangle around the position so the operator can see
        # whether the HULL clears the pool wall and the tags, not just whether
        # a dimensionless point does. hw_nav.yaml rov_footprint_m.
        self.rov_size_m = (0.4318, 0.5334)      # 17 x 21 inch
        self.pool: list | None = None           # 4 corners [(x,y),...] NED
        self.square_ned: list | None = None     # placed path points [(x,y),...]
        # "square" | "circle" | "arc" close; "line" is the only OPEN path.
        self.square_kind = "square"
        self.trail_act: deque = deque(maxlen=TRAIL_MAX)   # (t, x, y) NED
        self.trail_ref: deque = deque(maxlen=TRAIL_MAX)
        # The IMU dead-reckoned estimate: a THIRD series, in the same frame as
        # the other two so the gap between it and trail_act is read straight
        # off the picture. That gap is the experiment's whole output.
        self.trail_dr: deque = deque(maxlen=TRAIL_MAX)
        # ON whenever a dead reckoner is reporting. Running with --imu-dr IS
        # the request to see the estimate — the experiment is the two markers
        # side by side, and an overlay the operator has to discover and switch
        # on is one they will forget to switch on.
        #
        # It was off by default for one day (2026-08-17) for a real reason: an
        # unaided IMU drifts by KILOMETRES ("DR 140000 cm" on the first pool
        # run), and a trail drawn out to there crosses the whole plot and
        # buries the two series that matter. Hiding the series was the wrong
        # cure for that — the drawing is now CLIPPED to the viewport and a
        # runaway estimate is reported as an arrow on the border instead
        # (_clip_runs / the off-screen marker in paintEvent), so the overlay
        # can be on by default and still never take the plot away.
        self.show_dr = True
        # The tracked OBJECT, straight off bus.object_fix. It arrives ALREADY
        # in the map frame (state.ObjectFix), which is the frame this plot
        # draws in, so nothing here transforms it — the same "this window
        # computes nothing" rule the rest of the panel follows. Routing it
        # through MpcStatus instead would have meant mirroring FLU and
        # applying `_to_map` to a quantity that was never in the datum frame.
        self.trail_obj: deque = deque(maxlen=TRAIL_MAX)
        self.obj: ObjectFix | None = None
        # (ratio, n_tags) from window._check_depth_scale, or None until enough
        # mapped tags have been seen with depth behind them.
        self.depth_chk: tuple | None = None
        self.p_act = None                       # (x, y, z) NED
        self._act_t = 0.0                       # ...and when it was set
        self._nav_note = ""                     # why the localizer refused
        self.p_ref = None
        self.p_dr = None
        self.yaw_dr = None
        # (id, instance) of every tag in the LATEST accepted fix. The INSTANCE
        # matters: a duplicated id has two squares on the floor and only one of
        # them carried the fix — lighting both would be a lie about the very
        # thing the operator is watching for.
        self.used_ids: frozenset = frozenset()
        self._used_t = 0.0
        self.yaw_ned = None
        self.fix_hz = None
        self.fix_det_ms = 0.0
        self.fix_src = ""                       # which feed localizes
        self.status: MpcStatus | None = None
        # The plot lives in the MAP (tag-world) frame, ALWAYS: the mat stays
        # axis-aligned, the pool is the same rectangle every run, and a tag id
        # is where the operator can point at it. The controller works in the
        # ENGAGE-DATUM frame (START pose = origin, start heading = +x), so
        # everything arriving from MpcStatus gets rotated back through this.
        # Before 2026-08-14 the conversion ran the other way and the whole mat
        # visibly swung round the moment START was pressed.
        self.datum = None                       # (x0, y0, z0, yaw0), map frame
        self.zoom = 1.0
        self.pan = QtCore.QPointF(0.0, 0.0)     # screen px
        self._drag_from = None
        # 3-D view (the 3D button). Off = the top-down map this panel has
        # always been; the projection is continuous between them, so the
        # toggle starts from exactly the picture that was on screen.
        self.three_d = False
        self.azimuth_deg = 0.0
        self.elev_deg = 55.0
        self.z_centre = 0.0                     # the tag plane
        # Tiny minimum, deliberately: this view lives IN the main grid
        # (column 3, under SYSTEM HEALTH) and its minimum is a floor for
        # extreme shrink only — at the operator's real screen size it takes
        # the column's stretch space. A useful-looking minimum here would
        # push the whole window's minimum past a laptop screen.
        self.setMinimumSize(200, 110)
        self.setToolTip(
            "NED top-down: screen up = +x, screen right = +y\n"
            "drag to pan · mouse wheel to zoom · double-click to reset the view")

    def _dr_visible(self) -> bool:
        """Draw the DR series? Yes whenever one is reporting, unless the
        operator hid it — and never hideable in CONTROL, where that overlay is
        the instrument the vehicle is being flown on."""
        st = self.status
        if st is None or not st.dr_mode:
            return False
        if st.dr_mode == "control":
            return True
        return bool(self.show_dr)

    # ------------------------------------------------------------- data in
    def set_datum(self, d) -> None:
        """The engage datum (x0, y0, z0, yaw0) in MAP coordinates, or None."""
        self.datum = None if d is None else tuple(float(v) for v in d)
        self.update()

    def _to_map(self, x: float, y: float) -> tuple[float, float]:
        """Datum-frame xy -> MAP xy (the frame everything is drawn in)."""
        if self.datum is None:
            return x, y
        x0, y0, _z0, yaw0 = self.datum
        c, s = math.cos(yaw0), math.sin(yaw0)
        return x0 + c * x - s * y, y0 + s * x + c * y

    def _to_map_z(self, z: float) -> float:
        """Datum-frame z -> MAP z. The MISSING HALF of `_to_map`, added
        2026-08-23.

        The engage datum is a horizontal isometry with a z OFFSET: `_datumize`
        does `Rz @ (eta - p0)`, so a datum-frame z is measured from the depth
        the vehicle engaged at, not from the tag plane. `_to_map` translated
        and rotated x/y and left z alone — it even named the field `_z0` to say
        so — so from the moment anything engaged, the vehicle marker, its trail,
        the reference cross and the DR ghost were all drawn at z ~ 0, i.e.
        LYING ON THE TAG MAT, while the object kept its true map z and the chip
        kept printing the right depth (`_hold_z_text` added the offset back by
        hand, in the one place that did). In the top-down view nothing showed;
        tilt it and the vehicle sat on the floor (operator, 2026-08-23).

        Doing it HERE and not at each reader is the point: one boundary, after
        which every z on this widget is a map z and differences between them
        (`dz`, the error lines) keep working untouched."""
        return z + (self.datum[2] if self.datum else 0.0)

    def set_pool(self, corners: list | None) -> None:
        """The pool boundary's 4 corners (x, y) NED, ALREADY in the plot's
        frame (the window applies the engage datum). Also sets the view scale
        (_fit)."""
        self.pool = list(corners) if corners else None
        self.update()

    def set_map_tags(self, pts: list) -> None:
        """The tag map's (x, y, yaw, id, instance) entries, ALREADY in the
        frame the rest of the plot uses (the window applies the engage datum).
        yaw is the tag's in-plane rotation, so the square is drawn the way the
        tag actually lies on the floor — the old tagslam visualization. A
        duplicated id contributes one entry PER physical copy."""
        self.map_tags = list(pts)
        self.update()

    @staticmethod
    def _trail_push(trail: deque, x: float, y: float, eps: float,
                    z: float = 0.0) -> None:
        """One trail sample, ``(t, x, y, z)``.

        ``z`` joined the tuple with the 3-D view (2026-08-21): a top-down plot
        never needed it, but a trajectory you can tilt is not a trajectory
        unless the depth is in it. Every consumer indexes by position, so the
        tuple is the contract — do not reorder it."""
        if not trail or (abs(trail[-1][1] - x) > eps
                         or abs(trail[-1][2] - y) > eps):
            trail.append((time.monotonic(), x, y, z))

    def _prune_trails(self) -> None:
        cut = time.monotonic() - TRAIL_AGE_S
        for trail in (self.trail_act, self.trail_ref, self.trail_dr,
                      self.trail_obj):
            while trail and trail[0][0] < cut:
                trail.popleft()

    def set_depth_check(self, ratio: float, spread: float) -> None:
        """(median, p10..p90 spread) of depth-map / expected-floor-range over
        the mat — see window._check_depth_scale."""
        self.depth_chk = (float(ratio), float(spread))

    def set_object(self, fx: ObjectFix) -> None:
        """One object fix. STORE AND DRAW, nothing else.

        ``fx.p_map`` is already in this plot's frame, so there is no mirror
        and no datum transform here — and that is why the object rides its own
        signal rather than MpcStatus: it is the one quantity in this window
        whose natural home is the map frame the pool is drawn in, and it
        exists with nothing engaged at all.
        """
        self.obj = fx
        if fx is not None and fx.p_map is not None and fx.ok:
            self._trail_push(self.trail_obj, float(fx.p_map[0]),
                             float(fx.p_map[1]), 5e-3, float(fx.p_map[2]))
        self._prune_trails()
        self.update()

    def add_fix(self, f: NavFix) -> None:
        """Localizer health readout + WHICH tags carried the fix, and — while
        the MPC worker's state stream is NOT driving the plot (not engaged,
        e.g. bench runs with no ArduSub telemetry) — the marker itself. Once
        engaged, MpcStatus (20 Hz, velocity-bridged, same datum frame: the
        window transformed this fix already) takes over so the marker moves
        at the control rate."""
        if not f.ok:
            return
        self.fix_hz = f.hz
        self.fix_det_ms = f.detect_ms
        self.fix_src = f.source or self.fix_src
        insts = f.tag_insts or ((0,) * len(f.tag_ids))
        self.used_ids = frozenset(
            (int(i), int(k)) for i, k in zip(f.tag_ids, insts))
        self._used_t = time.monotonic()
        engaged = self.status is not None and self.status.engaged
        if not engaged:
            x, y = float(f.p_ned[0]), float(f.p_ned[1])
            self._trail_push(self.trail_act, x, y, 1e-3,
                             float(f.p_ned[2]))
            self._set_p_act(tuple(float(v) for v in f.p_ned))
            if f.yaw_ned is not None:
                self.yaw_ned = float(f.yaw_ned)
        self._prune_trails()
        self.update()

    def _set_p_act(self, p) -> None:
        """The vehicle marker, with the ONE clock that ages it.

        Two sources write it — `add_fix` while nothing is engaged, `add_status`
        once the control loop owns the state — and they run at similar rates.
        Ageing it here rather than letting whichever arrived last decide is
        what stops the 20 Hz stream with no state from erasing the marker the
        localizer had just placed (which is exactly what it did: the hull
        vanished on a bench run with the tags plainly in view, 2026-08-21)."""
        self.p_act = None if p is None else tuple(float(v) for v in p)
        self._act_t = time.monotonic()

    def _act_fresh(self) -> bool:
        """Is the vehicle marker recent enough to draw?

        ONE bound for both sources. Without it, "add_fix owns p_act while not
        engaged" would mean a marker that stays put forever after the
        localizer dies — the same lie the DR ghost has its own rule against."""
        return (self.p_act is not None
                and (time.monotonic() - self._act_t) < 1.5)

    def add_status(self, s: MpcStatus) -> None:
        self.status = s
        # ONE WRITER AT A TIME, and `s.engaged` is the switch. `_set_p_act`
        # has always documented the rule — "add_fix while nothing is engaged,
        # add_status once the control loop owns the state" — but this branch
        # did not honour it: MpcWorker._publish fills `p_flu` on every tick it
        # has a state, engaged or not, so outside an engagement BOTH sources
        # wrote the marker, at 17 Hz and at 20 Hz, with two different
        # estimates of the same pose:
        #
        #     add_fix      the tag fix RAW           z = tag PnP z
        #     add_status   the assembled state       z = pressure + offset,
        #                  x/y velocity-bridged across the fix age, yaw
        #                  gyro-bridged (control/state_assembler.py)
        #
        # The marker alternated between the two every tick and the trail took
        # a sample from each, which is what drew the vertical comb the
        # operator saw in the 3-D view on 2026-08-23: in an orthographic tilt
        # `_basis3` gives screen-right no z component, so an alternation
        # draws a stroke straight up and down and 90 s of them is a curtain.
        # Neither estimate was wrong; drawing both as one series was.
        if s.p_flu is not None and s.engaged:
            # FLU -> NED mirror, then datum -> map (see set_datum)
            x, y = self._to_map(s.p_flu[0], -s.p_flu[1])
            z = self._to_map_z(-s.p_flu[2])
            self._trail_push(self.trail_act, x, y, 1e-3, z)
            self._set_p_act((x, y, z))
            if s.yaw_flu_deg is not None:
                yaw = -math.radians(s.yaw_flu_deg)
                self.yaw_ned = yaw + (self.datum[3] if self.datum else 0.0)
        elif s.engaged:
            # The same honesty rule p_ref and p_dr already follow. Without it a
            # tag dropout freezes the green hull while the amber ghost keeps
            # moving, so the error line grows against a stale truth — and the
            # readout says "DR --" at the same moment the picture shows a
            # confident, wrong separation. Latent so far: every row of the
            # 2026-08-18 runs had a fix.
            #
            # ...but ONLY while engaged. This ran unconditionally until
            # 2026-08-21, and outside an engagement MpcStatus carries no
            # position at all — so a 20 Hz stream of `p_flu = None` erased the
            # marker `add_fix` had set from a perfectly good tag fix, and the
            # hull simply never appeared. Not engaged, `_act_fresh` is what
            # ages it instead.
            self._set_p_act(None)
        # The dead-reckoned estimate, through the SAME mirror and the SAME
        # datum transform as p_flu above — that identity is what makes the two
        # markers comparable, so it is spelled the same way on purpose.
        if s.p_dr_flu is not None and s.dr_ok:
            dx, dy = self._to_map(s.p_dr_flu[0], -s.p_dr_flu[1])
            dz = self._to_map_z(-float(s.p_dr_flu[2]))
            self._trail_push(self.trail_dr, dx, dy, 1e-3, dz)
            self.p_dr = (dx, dy, dz)
            if s.yaw_dr_flu_deg is not None:
                self.yaw_dr = (-math.radians(s.yaw_dr_flu_deg)
                               + (self.datum[3] if self.datum else 0.0))
        else:
            # Same honesty rule as p_ref below: no estimate, no marker. A
            # ghost hull frozen where the samples stopped would be the single
            # most misleading thing this plot could draw — a dead reckoner
            # that has died looks like one that is tracking perfectly.
            self.p_dr = None
        if s.ref_flu is not None:
            # FLU -> NED mirror for display (the map frame the pool is
            # measured in): (x, -y, -z).
            rx, ry = self._to_map(s.ref_flu[0], -s.ref_flu[1])
            r = (rx, ry, self._to_map_z(-s.ref_flu[2]))
            self.p_ref = r
            if s.engaged:
                self._trail_push(self.trail_ref, r[0], r[1], 1e-4, r[2])
        else:
            # NO reference means NO reference marker. MpcStatus only carries
            # ref_flu while engaged, and without this the cross (and the red
            # error line to it) stayed frozen wherever the run ended — a live
            # -looking target for a controller that is no longer driving to
            # anything. The reference TRAIL stays, so the path just flown is
            # still there to look at; only the "we are aiming here right now"
            # marker goes. Same honesty rule as the REC button and the ARM
            # label: the plot may only show what MpcStatus actually says.
            self.p_ref = None
        if s.scenario and s.scenario.get("kind") in ("follow", "replay"):
            # Neither has a PLACED geometry — a follow's path is wherever the
            # object goes, a replay's is a streamed plan drawn live by the
            # reference trail. Said explicitly rather than falling through,
            # so a stale rectangle outline from an earlier mission cannot
            # survive underneath either.
            self.square_ned = None
            self.square_kind = s.scenario.get("kind")
        elif s.scenario and s.scenario.get("kind") == "station":
            self.square_ned = [self._to_map(*s.scenario["origin_ned"])]
            self.square_kind = "station"
        elif s.scenario and s.scenario.get("kind") in ("square", "line",
                                                       "circle"):
            self.square_ned = [self._to_map(x, y)
                               for x, y in self._square_corners(s.scenario)]
            # "arc" = the sampled MPCC curve: one closed polyline, so it is
            # drawn closed and WITHOUT the line's turnaround end-markers (a
            # sampled lap already returns to its start).
            self.square_kind = ("arc"
                                if s.scenario.get("path", {}).get("kind")
                                == "mpcc-arc" else s.scenario.get("kind"))
        elif not s.traj_on:
            self.square_ned = self.square_ned if s.engaged else None
        self._prune_trails()
        self.update()

    @staticmethod
    def _square_corners(sc: dict) -> list:
        """The placed path in NED, as a polyline to draw.

        When the run is flying the MPCC curve the scenario carries a ``path``
        block, and this returns the ACTUAL filleted curve — densely sampled —
        rather than the sharp rectangle the operator typed. Drawing the
        rectangle would put a picture of a 90-degree corner under a vehicle
        that is deliberately rounding it, which is the same class of lie as
        the geofence box that was drawn after the fence was removed.
        Otherwise (legacy trajectory tracking) it is the polygon: a line's two
        endpoints, a rectangle's four corners, or — for a circle, which has no
        corners to name — a dense sampling of the rim."""
        if sc.get("path", {}).get("kind") == "mpcc-arc":
            import numpy as np

            from ..control.path_geometry import path_from_scenario

            p = path_from_scenario(
                sc, fillet_m=float(sc["path"].get("fillet_m", 0.15)),
                turn_radius_m=float(sc["path"].get("turn_radius_m", 0.0)))
            n = max(64, int(p.lap_length / 0.02))
            x, y, _psi, _k = p.sample(np.linspace(0.0, p.lap_length, n))
            return list(zip(x.tolist(), y.tolist()))
        from ..control.reference import (circle_points_world,
                                         line_points_world,
                                         rect_corners_world)

        ox, oy = sc["origin_ned"]
        if sc.get("kind") == "circle":
            # The entered tag is ON the rim (its min-x point), so the outline
            # is drawn about a centre one radius away — never about the tag.
            # Getting this backwards would draw a circle the operator's tag
            # sits in the middle of, which is precisely the shape they said
            # they did NOT want.
            flu = circle_points_world(sc["radius"], (ox, -oy),
                                      -math.radians(sc.get("rot_deg", 0.0)))
        elif sc.get("kind") == "line":
            flu = line_points_world(sc["length"], (ox, -oy),
                                    -math.radians(sc.get("dir_deg", 90.0)))
        else:
            sy = sc.get("size_y", sc["size"])
            # mirror_y matches HwDobMpc.set_square_ned: the entered tag is the
            # rectangle's min-x/min-y corner in the MAP frame.
            flu = rect_corners_world(sc["size"], sy, (ox, -oy),
                                     -math.radians(sc.get("rot_deg", 0.0)),
                                     mirror_y=True)
        return [(x, -y) for x, y in flu]

    def clear(self) -> None:
        self.trail_act.clear()
        self.trail_ref.clear()
        self.trail_dr.clear()
        self.trail_obj.clear()
        self.update()

    @staticmethod
    def _clip_runs(trail, xw0, xw1, yw0, yw1, pad: float = 0.25) -> list:
        """Contiguous stretches of a trail that lie inside the viewport.

        Point-wise rather than a true segment clip: a sample every 50 ms is
        far finer than the box, so the visible error is at most one sample of
        overshoot at each edge, and this cannot produce the long false chord a
        naive polyline draws when the series leaves the plot and comes back.
        """
        runs, cur = [], []
        for _t, x, y, z in trail:
            if (xw0 - pad) <= x <= (xw1 + pad) and (yw0 - pad) <= y <= (yw1 + pad):
                cur.append((x, y, z))
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        return runs

    def _draw_offscreen_dr(self, p, xw0, xw1, yw0, yw1) -> None:
        """A triangle on the border pointing at a dead reckoner that has left
        the plot. The estimate is still REPORTED (the readout has the metres);
        this only says which way it went."""
        ax, ay = self.p_act[0], self.p_act[1]
        dx, dy = self.p_dr[0] - ax, self.p_dr[1] - ay
        n = math.hypot(dx, dy)
        if n < 1e-9:
            return
        # Walk from the vehicle toward the estimate until the box edge.
        tmin = 1.0
        for lo, hi, o, d in ((xw0, xw1, ax, dx), (yw0, yw1, ay, dy)):
            if abs(d) > 1e-12:
                for edge in (lo, hi):
                    t = (edge - o) / d
                    if 0.0 < t < tmin:
                        q = (ax + t * dx, ay + t * dy)
                        inx = (xw0 - 1e-6) <= q[0] <= (xw1 + 1e-6)
                        iny = (yw0 - 1e-6) <= q[1] <= (yw1 + 1e-6)
                        if inx and iny:
                            tmin = t
        c = self._px(ax + tmin * dx, ay + tmin * dy)
        # Inset from the border so the whole triangle is on the widget: the
        # world bounds ARE the widget edge, so a marker centred on them is
        # drawn half outside and clipped to a sliver.
        m = 11.0
        c = QtCore.QPointF(min(max(c.x(), m), self.width() - m),
                           min(max(c.y(), m), self.height() - m))
        ang = math.atan2(dy / n, dx / n)          # NED
        # screen: +x_ned is UP, +y_ned is RIGHT
        sx, sy = math.sin(ang), -math.cos(ang)
        tri = QtGui.QPolygonF([
            QtCore.QPointF(c.x() + 9 * sx, c.y() + 9 * sy),
            QtCore.QPointF(c.x() - 5 * sx + 5 * sy, c.y() - 5 * sy - 5 * sx),
            QtCore.QPointF(c.x() - 5 * sx - 5 * sy, c.y() - 5 * sy + 5 * sx)])
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_c(theme.WARN, 220))
        p.drawPolygon(tri)

    def _hull(self, p_ned, yaw: float) -> list:
        """The vehicle's footprint rectangle in NED, oriented by heading.

        Shared by the tag marker and the dead-reckoned ghost so the two are
        the same size and the same shape — the operator is being asked to
        judge the distance BETWEEN them, and two differently drawn boxes would
        make that judgement about the drawing.
        """
        hl, hw = self.rov_size_m[0] / 2.0, self.rov_size_m[1] / 2.0
        ca, sa = math.cos(yaw), math.sin(yaw)
        return [(p_ned[0] + ca * dx - sa * dy, p_ned[1] + sa * dx + ca * dy)
                for dx, dy in ((hl, -hw), (hl, hw), (-hl, hw), (-hl, -hw))]

    # ---------------------------------------------------------- projection
    def _fit(self) -> tuple[float, float, float]:
        """(px_per_m, cx_ned, cy_ned). The POOL sets the scale when it is
        known (operator request 2026-08-14: the boundary is the pool and the
        axes are scaled to it); otherwise a 4 m box. (The geofence used to be
        the middle fallback; it was removed 2026-08-14.)"""
        if self.pool:
            xs = [c[0] for c in self.pool]
            ys = [c[1] for c in self.pool]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        else:
            x0, x1, y0, y1 = -2.0, 2.0, -2.0, 2.0
        w = max(0.5, y1 - y0)
        h = max(0.5, x1 - x0)
        s = 0.85 * min(self.width() / w, self.height() / h) * self.zoom
        return s, (x0 + x1) / 2.0, (y0 + y1) / 2.0

    def _px(self, x_ned: float, y_ned: float,
            z_ned: float = 0.0) -> QtCore.QPointF:
        """MAP metres -> screen pixels, in whichever mode the view is in.

        ONE projection for both, so every caller keeps working and the two
        views can never disagree about where a point is. ``z_ned`` is ignored
        top-down (it always was) and load-bearing in 3-D.

        The 3-D projection is ORTHOGRAPHIC — no perspective. At pool scale
        perspective buys nothing and costs a lot: two equal distances would
        stop looking equal depending on where they sat, which is exactly the
        judgement the operator is making off this plot. Orthographic also
        makes the two modes continuous: at ``elev = 90`` the basis below
        reduces to screen-up = +x_ned, screen-right = +y_ned, i.e. the
        top-down view, bit for bit.
        """
        s, cx, cy = self._fit()
        if not self.three_d:
            return QtCore.QPointF(
                self.width() / 2.0 + (y_ned - cy) * s + self.pan.x(),
                self.height() / 2.0 - (x_ned - cx) * s + self.pan.y())
        r, u = self._basis3()
        dx, dy, dz = x_ned - cx, y_ned - cy, z_ned - self.z_centre
        return QtCore.QPointF(
            self.width() / 2.0
            + (dx * r[0] + dy * r[1] + dz * r[2]) * s + self.pan.x(),
            self.height() / 2.0
            - (dx * u[0] + dy * u[1] + dz * u[2]) * s + self.pan.y())

    def _basis3(self):
        """(right, up) unit vectors of the 3-D eye, in MAP coordinates.

        Built from the two angles directly rather than from a cross product
        with a world up-vector, because that product degenerates exactly at
        the top-down pose this view has to reduce to. ``+z is DOWN`` in the
        map frame, so an elevation of +90 deg puts the eye ABOVE the pool.
        """
        az = math.radians(self.azimuth_deg)
        el = math.radians(self.elev_deg)
        ca, sa = math.cos(az), math.sin(az)
        ce, se = math.cos(el), math.sin(el)
        # Look direction (eye -> centre), map frame. The horizontal part is
        # NEGATIVE of the azimuth ray on purpose: azimuth 0 puts the eye on
        # the -x side, which is the only placement under which BOTH things
        # hold — screen-up is +x_ned at elevation 90 (so the mode reduces to
        # the top-down view), and the world's up (-z, since map z is DOWN) is
        # up on screen once tilted. Putting the eye at +x satisfies the first
        # and renders the world upside down under the second.
        f = (ce * ca, ce * sa, se)
        r = (-sa, ca, 0.0)
        u = (r[1] * f[2] - r[2] * f[1],
             r[2] * f[0] - r[0] * f[2],
             r[0] * f[1] - r[1] * f[0])
        return r, u

    def _depth_stick(self, p, painter, pen) -> None:
        """A vertical line from a marker down to the tag plane (z = 0).

        The one thing a tilted view genuinely adds over a top-down one is
        DEPTH, and a floating marker in an orthographic projection is
        ambiguous without a foot: the same pixel is every point along the
        view ray. The stick is what makes it readable, so every 3-D marker
        gets one.
        """
        if not self.three_d or p is None:
            return
        painter.setPen(pen)
        painter.drawLine(self._px(p[0], p[1], p[2]),
                         self._px(p[0], p[1], 0.0))

    def _world_bounds(self) -> tuple[float, float, float, float]:
        """(x_min, x_max, y_min, y_max) of the NED world visible right now —
        the exact inverse of _px, so grid lines span the viewport at any
        pan/zoom instead of only the fitted box.

        In 3-D there IS no such inverse (a screen pixel is a whole view ray),
        so the FITTED box plus a margin stands in. The grid and the trail
        clipping are the only two callers and both want "roughly the world we
        are looking at", which is what that is."""
        if self.three_d:
            if self.pool:
                xs = [c[0] for c in self.pool]
                ys = [c[1] for c in self.pool]
                m = 0.5
                return (min(xs) - m, max(xs) + m, min(ys) - m, max(ys) + m)
            return -2.5, 2.5, -2.5, 2.5
        s, cx, cy = self._fit()
        s = max(1e-6, s)
        w, h = self.width(), self.height()
        y_min = (0 - w / 2.0 - self.pan.x()) / s + cy
        y_max = (w - w / 2.0 - self.pan.x()) / s + cy
        x_min = (h / 2.0 + self.pan.y() - h) / s + cx
        x_max = (h / 2.0 + self.pan.y() - 0) / s + cx
        return x_min, x_max, y_min, y_max

    # --------------------------------------------------------------- mouse
    def mousePressEvent(self, ev):
        self._drag_from = ev.pos()

    def mouseMoveEvent(self, ev):
        if self._drag_from is None:
            return
        d = ev.pos() - self._drag_from
        self._drag_from = ev.pos()
        if self.three_d:
            # Drag ORBITS in 3-D and pans in 2-D. Panning a tilted view is
            # the gesture nobody reaches for first, and orbiting is the whole
            # reason the mode exists. Elevation is clamped short of 90 so the
            # view never lands on the degenerate straight-down pose — that is
            # what the button is for.
            self.azimuth_deg = (self.azimuth_deg + d.x() * 0.4) % 360.0
            self.elev_deg = max(5.0, min(89.0, self.elev_deg + d.y() * 0.3))
        else:
            self.pan += QtCore.QPointF(d.x(), d.y())
        self.update()

    def mouseReleaseEvent(self, _ev):
        self._drag_from = None

    def wheelEvent(self, ev):
        step = 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
        self.zoom = max(0.3, min(8.0, self.zoom * step))
        self.update()

    def mouseDoubleClickEvent(self, _ev):
        self.zoom = 1.0
        self.pan = QtCore.QPointF(0.0, 0.0)
        self.azimuth_deg, self.elev_deg = 0.0, 55.0
        self.update()

    def set_three_d(self, on: bool) -> None:
        self.three_d = bool(on)
        self.pan = QtCore.QPointF(0.0, 0.0)
        self.setToolTip(
            "3-D: drag orbits · wheel zooms · double-click resets the view\n"
            "the stick under each marker drops to the tag plane (z = 0)"
            if self.three_d else
            "NED top-down: screen up = +x, screen right = +y\n"
            "drag to pan · mouse wheel to zoom · double-click to reset the view")
        self.update()

    # --------------------------------------------------------------- paint
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), _c(theme.VIDEO_BG))

        def poly(points, pen, close=False):
            if len(points) < 2:
                return
            path = QtGui.QPainterPath()
            path.moveTo(self._px(*points[0]))
            for q in points[1:]:
                path.lineTo(self._px(*q))
            if close:
                path.closeSubpath()
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)

        # Adaptive metre grid across the whole viewport, with tick labels:
        # values of x_ned up the LEFT edge (horizontal lines are constant-x),
        # values of y_ned along the BOTTOM edge. The step is chosen so ticks
        # never crowd (>= 48 px apart) at any zoom.
        s, _cx, _cy = self._fit()
        xw0, xw1, yw0, yw1 = self._world_bounds()
        step = next((c for c in (0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
                     if c * s >= 48.0), 10.0)
        # Ticks accumulate by repeated addition, so a value that should be 0
        # arrives as -2.8e-16 and %g prints it in full. Snap to the step's own
        # precision before formatting.
        dec = max(0, -int(math.floor(math.log10(step))) + 1)

        def tick(v: float) -> str:
            return f"{round(v, dec) + 0.0:g}"

        grid_pen = QtGui.QPen(_c(theme.TEXT_FAINT, 34), 1)
        p.setFont(QtGui.QFont("monospace", 8))
        gx = math.floor(xw0 / step) * step
        while gx <= xw1 + 1e-9:
            poly([(gx, yw0), (gx, yw1)], grid_pen)
            q = self._px(gx, yw0)
            p.setPen(_c(theme.TEXT_DIM, 170))
            # Top-down the labels run up the left EDGE (a horizontal grid line
            # has one screen y). Tilted they do not — the line's two ends are
            # at different heights — so the label goes on the end it belongs
            # to instead of on a margin it no longer relates to.
            if self.three_d:
                p.drawText(int(q.x()) - 10, int(q.y()) + 3, tick(gx))
            else:
                p.drawText(4, int(q.y()) + 3, tick(gx))
            gx += step
        gy = math.floor(yw0 / step) * step
        while gy <= yw1 + 1e-9:
            poly([(xw0, gy), (xw1, gy)], grid_pen)
            q = self._px(xw0, gy)
            p.setPen(_c(theme.TEXT_DIM, 170))
            if self.three_d:
                p.drawText(int(q.x()) + 2, int(q.y()) + 10, tick(gy))
            else:
                p.drawText(int(q.x()) + 2, self.height() - 18, tick(gy))
            gy += step
        # Axis captions, in the corners the ticks accumulate toward.
        p.setPen(_c(theme.TEXT_DIM))
        if self.three_d:
            # Say WHERE the eye is. An orbited orthographic view has no other
            # cue for it, and "which way am I looking" is the first question
            # anyone asks of a plot they just tilted.
            p.drawText(4, 12, f"3D  az {self.azimuth_deg:.0f}  "
                              f"el {self.elev_deg:.0f}  (grid = tag plane)")
        else:
            p.drawText(4, 12, "x [m] ↑N")
            p.drawText(self.width() - 52, self.height() - 18, "y [m] →")

        # Pool boundary: the physical wall, drawn solid — the one line on this
        # plot the vehicle must never cross. Placement is config (hw_nav.yaml
        # pool_ned, [예측] until the map->wall offsets are taped).
        if self.pool:
            poly(self.pool, QtGui.QPen(_c(theme.TEXT_DIM, 220), 2), close=True)
            q = self._px(*self.pool[0])
            p.setPen(_c(theme.TEXT_FAINT, 180))
            p.drawText(int(q.x()) + 4, int(q.y()) - 4, "POOL")

        # (A dashed GEOFENCE box used to appear here the moment the run
        # engaged. Removed 2026-08-14 at the operator's request — the fence
        # itself is gone, so drawing one would promise a guard that no longer
        # exists. POOL, above, is now the only boundary on the plot, and it is
        # a picture of the wall rather than a limit anything enforces.)

        # The surveyed tag map, tagslam-viz style: each tag an oriented square
        # at its true printed size; the tags carrying the CURRENT fix fill
        # green, the rest stay faint. Highlight decays with the fix (1 s), so
        # a dead localizer cannot keep tags lit. Ids once zoomed in enough for
        # the text to be legible rather than confetti.
        if self.map_tags:
            show_ids = s * self.tag_size_m > 12.0   # tag edge spans >= 12 px
            fresh = (time.monotonic() - self._used_t) < 1.0
            half = self.tag_size_m / 2.0
            p.setFont(QtGui.QFont("monospace", 7))
            for tx, ty, tyaw, tid, tinst in self.map_tags:
                c, sn = math.cos(tyaw), math.sin(tyaw)
                corners = [(tx + c * dx - sn * dy, ty + sn * dx + c * dy)
                           for dx, dy in ((-half, -half), (-half, half),
                                          (half, half), (half, -half))]
                path = QtGui.QPainterPath()
                path.moveTo(self._px(*corners[0]))
                for q2 in corners[1:]:
                    path.lineTo(self._px(*q2))
                path.closeSubpath()
                used = fresh and (int(tid), int(tinst)) in self.used_ids
                if used:
                    p.setPen(QtGui.QPen(_c(theme.OK), 1))
                    p.setBrush(_c(theme.OK, 140))
                else:
                    p.setPen(QtGui.QPen(_c(theme.TEXT_FAINT, 120), 1))
                    p.setBrush(_c(theme.TEXT_DIM, 45))
                p.drawPath(path)
                if show_ids or used:
                    q = self._px(tx, ty)
                    p.setPen(_c(theme.TEXT, 220) if used
                             else _c(theme.TEXT_FAINT, 160))
                    p.drawText(int(q.x()) + 4, int(q.y()) + 3, str(tid))

        # The placed reference path. A line is OPEN (two endpoints); a
        # rectangle closes. End markers on the line so the turnarounds — the
        # only places the reference stops — are visible.
        if self.square_ned and self.square_kind == "station":
            q = self._px(*self.square_ned[0])
            p.setPen(QtGui.QPen(_c(theme.ACCENT, 200), 1, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(q, 10, 10)
            p.drawLine(q + QtCore.QPointF(-14, 0), q + QtCore.QPointF(14, 0))
            p.drawLine(q + QtCore.QPointF(0, -14), q + QtCore.QPointF(0, 14))
        elif self.square_ned:
            is_line = self.square_kind == "line"
            poly(self.square_ned, QtGui.QPen(_c(theme.TEXT_DIM, 160), 1,
                                             Qt.PenStyle.DotLine),
                 close=not is_line)
            if is_line:
                p.setPen(QtGui.QPen(_c(theme.TEXT_DIM, 200), 1))
                p.setBrush(Qt.BrushStyle.NoBrush)
                for e in self.square_ned:
                    p.drawEllipse(self._px(*e), 4, 4)

        # Trails. Reference: plain blue. Actual: colored by AGE (viridis,
        # dark purple = oldest, yellow = newest) so time order is readable on
        # a path that crosses itself; points older than TRAIL_AGE_S were
        # pruned on the way in. Drawn in chunks — one pen per ~METERED span —
        # so the cost stays one path per color, not one line per sample.
        self._prune_trails()
        poly([(x, y, z) for _t, x, y, z in self.trail_ref],
             QtGui.QPen(_c("#4d9dff", 150), 1))
        # The dead-reckoned trail, UNDER the actual one so ground truth is
        # never obscured by the estimate. Dashed amber: it collides with
        # neither the blue reference nor the viridis ramp nor the green hull,
        # and amber is already this station's colour for a number with a
        # caveat attached — which is exactly what an unaided IMU is.
        if self._dr_visible():
            # CLIPPED to the viewport. A dead reckoner that has run away to a
            # kilometre would otherwise draw one dashed line straight across
            # the plot, over the pool, the tags and both real series.
            for run in self._clip_runs(self.trail_dr, xw0, xw1, yw0, yw1):
                poly(run, QtGui.QPen(_c(theme.WARN, 170), 1,
                                     Qt.PenStyle.DashLine))
        # The object's own track: a plain 1 px solid line. Deliberately NOT a
        # second viridis ramp — one age ramp per plot is enough, and a second
        # would make the two impossible to tell apart at a glance.
        if len(self.trail_obj) >= 2:
            for run in self._clip_runs(self.trail_obj, xw0, xw1, yw0, yw1):
                poly(run, QtGui.QPen(_c(theme.ACCENT, 120), 1))
        act = list(self.trail_act)
        if len(act) >= 2:
            t_now = time.monotonic()
            chunk = max(2, len(act) // 48 + 1)
            for i0 in range(0, len(act) - 1, chunk):
                seg = act[i0:i0 + chunk + 1]     # overlap 1 pt = no gaps
                if len(seg) < 2:
                    continue
                age = t_now - seg[len(seg) // 2][0]
                u = 1.0 - min(1.0, age / TRAIL_AGE_S)
                poly([(x, y, z) for _t, x, y, z in seg],
                     QtGui.QPen(_age_color(u), 2))

        # Current reference: a cross — stage 0 of the controller's shared
        # geometric path plan. It is the vehicle's projection plus at most
        # path_lead_m on the ACTIVE segment, and it holds exactly at a corner
        # until the real hull captures that vertex at low speed. Drawn only
        # while there IS one — see add_status.
        if self.p_ref is not None:
            q = self._px(self.p_ref[0], self.p_ref[1], self.p_ref[2])
            self._depth_stick(self.p_ref, p,
                              QtGui.QPen(_c("#4d9dff", 90), 1,
                                         Qt.PenStyle.DotLine))
            p.setPen(QtGui.QPen(_c("#4d9dff"), 2))
            p.drawLine(q + QtCore.QPointF(-6, 0), q + QtCore.QPointF(6, 0))
            p.drawLine(q + QtCore.QPointF(0, -6), q + QtCore.QPointF(0, 6))

        # The vehicle: its actual FOOTPRINT, oriented by heading, with a dot
        # at the centre so it stays findable when zoomed out. NED yaw 0 = +x
        # = screen up; the long side of the rectangle is ACROSS the heading
        # (this hull is wider than it is long).
        if self._act_fresh():
            zv = float(self.p_act[2])
            q = self._px(self.p_act[0], self.p_act[1], zv)
            a = float(self.yaw_ned) if self.yaw_ned is not None else 0.0
            hl, hw = self.rov_size_m[0] / 2.0, self.rov_size_m[1] / 2.0
            ca, sa = math.cos(a), math.sin(a)
            hull = self._hull(self.p_act, a)
            self._depth_stick(self.p_act, p,
                              QtGui.QPen(_c(theme.OK, 90), 1,
                                         Qt.PenStyle.DotLine))
            path = QtGui.QPainterPath()
            path.moveTo(self._px(hull[0][0], hull[0][1], zv))
            for pt in hull[1:]:
                path.lineTo(self._px(pt[0], pt[1], zv))
            path.closeSubpath()
            p.setPen(QtGui.QPen(_c(theme.OK, 220), 2))
            p.setBrush(_c(theme.OK, 40))
            p.drawPath(path)
            # a nose mark on the leading edge, so heading reads at a glance
            p.setPen(QtGui.QPen(_c(theme.OK), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawLine(self._px(hull[0][0], hull[0][1], zv),
                       self._px(hull[1][0], hull[1][1], zv))
            if self.yaw_ned is not None:
                tip = self._px(self.p_act[0] + (hl + 0.18) * ca,
                               self.p_act[1] + (hl + 0.18) * sa, zv)
                p.drawLine(self._px(self.p_act[0] + hl * ca,
                                    self.p_act[1] + hl * sa, zv), tip)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(_c(theme.OK))
            p.drawEllipse(q, 3, 3)

        # Where the IMU alone thinks the vehicle is: the same hull, outline
        # only, no nose and no centre dot, so it reads as subordinate to the
        # solid green truth. Suppressed when the hull is only a few pixels
        # across — at that zoom two overlapping outlines are noise, and the
        # numeric readout below still carries the answer.
        if self.p_dr is not None and self._dr_visible():
            hull_px = abs(self._px(0.0, 0.0).x()
                          - self._px(0.0, self.rov_size_m[1]).x())
            zd = float(self.p_dr[2])
            if hull_px >= 10.0:
                hull = self._hull(self.p_dr,
                                  float(self.yaw_dr or self.yaw_ned or 0.0))
                path = QtGui.QPainterPath()
                path.moveTo(self._px(hull[0][0], hull[0][1], zd))
                for pt in hull[1:]:
                    path.lineTo(self._px(pt[0], pt[1], zd))
                path.closeSubpath()
                p.setPen(QtGui.QPen(_c(theme.WARN, 200), 1,
                                    Qt.PenStyle.DashLine))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPath(path)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(_c(theme.WARN, 200))
            p.drawEllipse(self._px(self.p_dr[0], self.p_dr[1], zd), 3, 3)

        # THE OBJECT: a DIAMOND, and the shape is the point. Tags are squares,
        # the hull is a rectangle, the reference is a cross — a fourth marker
        # has to be a fourth shape or the plot stops being readable at a
        # glance. Filled accent while the estimate is live, hollow amber while
        # it is stale, hollow faint once it is lost: the same honesty rule
        # p_dr and p_ref already follow, which is that a confident marker
        # requires a confident estimate. No estimate at all -> no marker.
        o = self.obj
        if o is not None and o.p_map is not None and o.state != "cold":
            q = self._px(float(o.p_map[0]), float(o.p_map[1]),
                         float(o.p_map[2]))
            self._depth_stick(o.p_map, p,
                              QtGui.QPen(_c(theme.ACCENT, 90), 1,
                                         Qt.PenStyle.DotLine))
            live = (o.state == "live" and o.ok)
            if live:
                p.setPen(QtGui.QPen(_c(theme.ACCENT, 230), 1))
                p.setBrush(_c(theme.ACCENT, 190))
            elif o.state == "stale":
                p.setPen(QtGui.QPen(_c(theme.WARN, 220), 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
            else:
                p.setPen(QtGui.QPen(_c(theme.TEXT_FAINT, 200), 1))
                p.setBrush(Qt.BrushStyle.NoBrush)
            r_px = 7.0
            p.drawPolygon(QtGui.QPolygonF([
                QtCore.QPointF(q.x(), q.y() - r_px),
                QtCore.QPointF(q.x() + r_px, q.y()),
                QtCore.QPointF(q.x(), q.y() + r_px),
                QtCore.QPointF(q.x() - r_px, q.y())]))
            # Which way it is FACING — the thing a follow's orbit term acts
            # on. An undefined heading gets a "?" instead of a tick, because
            # drawing a tick at yaw 0 would be an answer we do not have.
            pen = QtGui.QPen(_c(theme.ACCENT if live else theme.WARN, 210), 2)
            p.setBrush(Qt.BrushStyle.NoBrush)
            if o.yaw_map is None:
                p.setPen(_c(theme.TEXT_DIM, 200))
                p.drawText(int(q.x()) + 9, int(q.y()) - 6, "?")
            else:
                ca_o, sa_o = math.cos(o.yaw_map), math.sin(o.yaw_map)
                p.setPen(pen)
                p.drawLine(q, self._px(float(o.p_map[0]) + 0.15 * ca_o,
                                       float(o.p_map[1]) + 0.15 * sa_o,
                                       float(o.p_map[2])))
            # Hull -> object, with the number the pilot actually judges on:
            # the C3's pose pipeline works at 0.3-0.8 m and fails to register
            # at 2.4 m (KNOWN_ISSUES 2026-08-09), so "am I in the band?" is
            # the question, and it is a distance rather than a picture.
            #
            # 3-D SINCE 2026-08-23. It was `hypot(dx, dy)` — the HORIZONTAL
            # separation — while the question it answers is about the camera's
            # line of sight, and this vehicle flies about a metre ABOVE the
            # mat its objects sit on. So the plot said 85 cm where the depth
            # probe on the same object said 0.72 m, and the gap changed with
            # altitude instead of staying put (operator, 2026-08-23). Neither
            # number was wrong; they were different quantities, and only one of
            # them is the one the working band is written in.
            if self._act_fresh():
                pa = self._px(*self.p_act)
                p.setPen(QtGui.QPen(_c(theme.ACCENT, 120), 1,
                                    Qt.PenStyle.DotLine))
                p.drawLine(pa, q)
                d = math.dist((float(o.p_map[0]), float(o.p_map[1]),
                               float(o.p_map[2])), self.p_act)
                p.setPen(_c(theme.ACCENT, 200))
                p.drawText(int((pa.x() + q.x()) / 2) + 4,
                           int((pa.y() + q.y()) / 2) - 3, f"{d * 100:.0f} cm")

        # Error line to the estimate: SOLID amber (this is the measurement),
        # while the one to the reference is dashed red (that is the tracking
        # error). Two dashed lines on a widget this size would be one too many
        # to tell apart. Drawn only while the estimate is ON SCREEN — see the
        # border arrow below for when it is not.
        dr_on_screen = (self.p_dr is not None
                        and xw0 <= self.p_dr[0] <= xw1
                        and yw0 <= self.p_dr[1] <= yw1)
        if (self._act_fresh() and self.p_dr is not None
                and self._dr_visible() and dr_on_screen):
            p.setPen(QtGui.QPen(_c(theme.WARN, 200), 1))
            p.drawLine(self._px(*self.p_act), self._px(*self.p_dr))
        elif (self.p_dr is not None and self._act_fresh()
                and self._dr_visible()):
            # Off the plot. Say WHERE it went with a small arrow clamped to
            # the border, rather than either silently dropping the marker or
            # letting it drag a line across everything. The distance is in the
            # DR readout, so the arrow only has to carry the direction.
            self._draw_offscreen_dr(p, xw0, xw1, yw0, yw1)
        if self._act_fresh() and self.p_ref is not None:
            p.setPen(QtGui.QPen(_c(theme.FAIL, 130), 1, Qt.PenStyle.DashLine))
            p.drawLine(self._px(*self.p_act), self._px(*self.p_ref))

        # Corner readout — numbers from MpcStatus only (module docstring).
        # Starts below the axis caption so the two never overprint.
        p.setFont(QtGui.QFont("monospace", 8))
        p.setPen(_c(theme.TEXT_DIM))
        # DELIBERATELY SPARSE (operator, 2026-08-14). Position, localizer
        # health, solver timing, tag count and the disengage reason all left
        # this box — each already has a home that is better at showing it:
        # the SENSORS panel (TagNav row), the chip above the plot (phase +
        # reason), and the MISSION LOG under CAMERA TILT. What stays is the
        # only thing the plot is uniquely for: how far off the path we are.
        s = self.status
        lines = []
        # NO FIX IS NEWS. The plot draws nothing at all without a vehicle
        # position, which reads as "the panel is broken" rather than "the
        # localizer is rejecting" — and on 2026-08-21 that was exactly the
        # confusion, with nine mapped tags plainly outlined in the video. The
        # localizer's own reason for the miss rides NavFix.note; this is the
        # place the operator is already looking.
        if not self._act_fresh():
            why = (self._nav_note or "").strip()
            lines.append("NO FIX" + (f" — {why[:44]}" if why else ""))
        # Tag-implied roll/pitch vs the autopilot's ATTITUDE. The camera
        # extrinsic check, and the ONLY way to see a wrong mount angle from
        # the outside — a bad tilt does not show up in reprojection error at
        # all, because solvePnP recovers the CAMERA pose and the extrinsic
        # only maps camera -> body afterwards.
        #
        # Shown while NOT engaged (that is when the bench check happens, on a
        # level and still vehicle) and, once flying, only when it is large
        # enough to be news. The operator asked for a sparse box in 2026-08-14
        # and a permanently-visible healthy number would be clutter.
        if s is not None and s.rp_residual_deg is not None:
            rp = float(s.rp_residual_deg)
            if not s.engaged or rp > 5.0:
                rr = s.rp_residual_rp_deg
                detail = (f"roll {rr[0]:+5.1f}  pitch {rr[1]:+5.1f}"
                          if rr else f"{rp:4.1f} deg")
                lines.append(f"tag-vs-ATTITUDE {detail}"
                             + ("  <- extrinsic/tilt?" if rp > 5.0 else ""))
        # THE SECOND CROSS-CHECK, and it sits here because it is the same kind
        # of statement as the one above: two independent sensors describing one
        # world, printed as their disagreement. `window._check_depth_scale`
        # compares the C3 depth map against the tag PnP, which never touches
        # depth. 1.00x = the depth path is metric. Anything else scales the
        # reconstructed mesh AND the object's range together, which is exactly
        # the ambiguity this line exists to resolve.
        if self.depth_chk is not None:
            r, spread = self.depth_chk
            bad = abs(r - 1.0) > 0.10
            # THE SPREAD IS NOT DECORATION. A calibration scale is one number;
            # if the middle 80% of the floor disagrees by more than ~0.15 the
            # depth map is not merely mis-scaled, it is mis-SHAPED (or the mat
            # is not the only thing in view), and the headline ratio should not
            # be quoted as a constant.
            lines.append(f"depth-vs-MAP  {r:4.2f}x  (+/-{spread:4.2f})"
                         + ("  <- depth NOT metric" if bad else ""))
        if s is not None and s.engaged:
            # DEPTH belongs on the same line as the horizontal error. Until
            # 2026-08-18 no number on this panel was about z at all: err_xy is
            # HORIZONTAL by definition, so a vehicle holding station 20 cm
            # below its target read as a perfect hold. `dz` + = the vehicle is
            # DEEPER than the reference (NED z is down-positive).
            dz = ""
            if self._act_fresh() and self.p_ref is not None:
                dz = f"   dz {(self.p_act[2] - self.p_ref[2]) * 100:+5.1f} cm"
            if s.err_cross is not None and s.err_along is not None:
                # The split, not the magnitude: the spatial target is placed
                # ahead on purpose, so radial error includes configured
                # lookahead. `off` is geometric cross-track error; `lag` is
                # distance along the active segment to stage 0.
                lines.append(f"off {s.err_cross * 100:+5.1f} cm   "
                             f"lag {s.err_along * 100:+5.1f} cm{dz}   "
                             f"lap {s.lap}")
            elif s.err_xy is not None:
                # STATION lands here: there is no path to split against, so
                # this is the one line that says how the hold is going.
                lines.append(f"err {s.err_xy * 100:5.1f} cm{dz}   "
                             f"lap {s.lap}")
            if s.speed_m_s is not None:
                ref = ("" if s.ref_speed_m_s is None
                       else f"  ref {s.ref_speed_m_s:5.3f}")
                lines.append(f"v   {s.speed_m_s:5.3f} m/s{ref}")
            if len(s.w_hat) == 6 and any(abs(v) > 1e-9 for v in s.w_hat):
                w = s.w_hat
                lines.append(f"w_hat [{w[0]:+.1f} {w[1]:+.1f} {w[2]:+.1f}] N")
        # THE OBJECT READOUT, whenever a tracker is reporting at all — a
        # pipeline that is registering or an object that has gone out of range
        # has to be as visible as a position.
        #
        # `pair` gets a PERMANENT slot rather than appearing only when it goes
        # wrong. A non-zero pair_dt_ms is the single warning that the camera
        # extrinsic has stopped cancelling out of the composition, and that
        # the unmeasured 0.2855 m camera lever arm is back in the error
        # budget. Nothing else on this screen would say so.
        o = self.obj
        if o is not None:
            pair = ("pair --" if o.pair_dt_ms is None
                    else f"pair {o.pair_dt_ms:.0f} ms")
            if o.p_map is not None:
                # THE POSITION, in the MAP frame — the pool's frame, the same
                # one px/py/pz above are in, so the two can be subtracted by
                # eye. This is the object readout (2026-08-23): the camera-frame
                # x/y/z that used to sit on the C3 RGB panel was in a frame
                # nothing else here works in, so it moved to the map and to
                # this one place (widgets/video.py records the removal).
                lines.append("obj  x %+5.2f  y %+5.2f  z %+5.2f m"
                             % (float(o.p_map[0]), float(o.p_map[1]),
                                float(o.p_map[2])))
                # No vehicle position, no range: 0.00 m would be a number
                # nobody measured (the same rule p_act itself follows). 3-D —
                # the same quantity the line label draws, and the same one the
                # depth probe on the C3 DEPTH panel reads.
                d = ("d   -- m" if not self._act_fresh() else
                     "d %4.2f m" % math.dist(
                         (float(o.p_map[0]), float(o.p_map[1]),
                          float(o.p_map[2])), self.p_act))
                yaw = ("yaw   --" if o.yaw_map is None
                       else f"yaw {math.degrees(o.yaw_map):+4.0f}")
                lines.append(f"     {o.state:<5s} {d}  {yaw}  "
                             f"age {o.age_s or 0.0:4.2f} s  {pair}")
            else:
                # NO POSITION YET — and this line is now drawn for `cold` too.
                # It used to be suppressed, which meant the plot went silent
                # in exactly the case that needs explaining: the tracker is
                # locked on and drawing a mask, every observation is being
                # REJECTED by object_nav's gates, and the map has nothing to
                # put a diamond on. The gate that did it says so in `note`.
                lines.append(f"obj  {o.state}  {pair}")
            # THE REASON, whenever the estimate is not live — including the
            # frozen-but-stale case, where a position exists and is being drawn
            # and is nonetheless not what the camera can see any more. Before
            # 2026-08-23 the note was shown only when there was no position at
            # all, so "object at 1.53 m, outside 0.15-1.20 m" was invisible the
            # moment one earlier observation had got through.
            why = (o.note or o.pose_state or "").strip()
            if why and o.state != "live":
                lines.append(f"     {why[:52]}")
        if s is not None and s.follow_state:
            hold = ""
            sc = s.scenario or {}
            if sc.get("kind") == "follow":
                hold = (f"  hold {float(sc.get('hold_m', 0.0)):4.2f} m / "
                        f"{float(sc.get('dyaw_deg', 0.0)):+.0f}")
            err = ("--" if s.follow_err_m is None
                   else f"{s.follow_err_m * 100:.1f} cm")
            lines.append(f"follow {s.follow_state}  err {err}{hold}")
        # The dead-reckoning readout. Shown whenever an estimator is
        # configured, engaged or not, because "it stopped" has to be as
        # visible as a number — and in CONTROL mode this line is the
        # operator's instrument for deciding when to stop the run by hand.
        if s is not None and s.dr_mode and self._dr_visible():
            tag = f"[{s.dr_source}/{s.dr_attitude}"
            tag += "/CONTROL]" if s.dr_mode == "control" else "]"
            if s.dr_ok and s.dr_err_m is not None:
                lines.append(f"DR {s.dr_err_m * 100:5.1f} cm  "
                             f"{s.dr_elapsed_s or 0.0:4.0f} s  {tag}")
            else:
                why = (s.dr_note or "waiting").split(";")[0]
                lines.append(f"DR --  {tag} {why[:34]}")
        for i, ln in enumerate(lines):
            if ln.startswith("NO FIX"):
                p.setPen(_c(theme.FAIL))
            elif ln.endswith("extrinsic/tilt?"):
                p.setPen(_c(theme.WARN))
            elif ln.startswith("DR ") and s is not None and not s.dr_ok:
                p.setPen(_c(theme.WARN))
            elif ln.startswith("DR "):
                p.setPen(_c(theme.WARN, 230))
            elif ln.startswith("follow lost") or ln.startswith("obj lost"):
                p.setPen(_c(theme.FAIL))
            elif (ln.startswith("follow ") or ln.startswith("obj ")) and \
                    ("stale" in ln or "leashed" in ln):
                p.setPen(_c(theme.WARN))
            elif ln.startswith("obj ") or ln.startswith("follow "):
                p.setPen(_c(theme.ACCENT, 220))
            else:
                p.setPen(_c(theme.TEXT_DIM))
            p.drawText(8, 26 + 12 * i, ln)
        p.setPen(_c(theme.TEXT_DIM))
        # (No frame/mouse caption: the axes are labelled at their own ends and
        # the mouse gestures are on the widget's tooltip, not burned into the
        # picture — operator request 2026-08-14.)

        # Trail-age legend, bottom-right: the color ramp with its time span.
        # Laid out from the RIGHT EDGE inwards so the "now" label cannot fall
        # off the widget at small widths.
        fm = p.fontMetrics()
        old_lbl, new_lbl = f"-{TRAIL_AGE_S:.0f}s", "now"
        bar_w, bar_h = 56, 5
        bx = (self.width() - 6 - fm.horizontalAdvance(new_lbl) - 3 - bar_w)
        by = self.height() - 14
        for i in range(bar_w):
            p.setPen(_age_color(i / (bar_w - 1)))
            p.drawLine(bx + i, by, bx + i, by + bar_h)
        p.setPen(_c(theme.TEXT_FAINT, 190))
        p.drawText(bx - 4 - fm.horizontalAdvance(old_lbl), by + bar_h, old_lbl)
        p.drawText(bx + bar_w + 3, by + bar_h, new_lbl)
        # Two more swatches to the LEFT of the ramp, laid out right-to-left for
        # the same reason the ramp is: nothing may fall off a narrow widget.
        # Only drawn when there IS a second series to disambiguate.
        swatches = []
        if (self.p_dr is not None or self.trail_dr) and self._dr_visible():
            swatches += [("DR", QtGui.QPen(_c(theme.WARN, 200), 2,
                                           Qt.PenStyle.DashLine)),
                         ("tag", QtGui.QPen(_c(theme.OK, 220), 2))]
        if self.obj is not None and self.obj.p_map is not None:
            swatches.append(("obj", QtGui.QPen(_c(theme.ACCENT, 200), 2)))
        if swatches:
            x = bx - 4 - fm.horizontalAdvance(old_lbl) - 10
            for lbl, pen in swatches:
                w_lbl = fm.horizontalAdvance(lbl)
                x -= w_lbl
                p.setPen(_c(theme.TEXT_FAINT, 190))
                p.drawText(x, by + bar_h, lbl)
                x -= 14
                p.setPen(pen)
                p.drawLine(x + 1, by + bar_h - 2, x + 11, by + bar_h - 2)
                x -= 8


class TrajectoryWindow(QtWidgets.QWidget):
    """Controls + the view, now DOCKED in the main window (window.py builds a
    QDockWidget around it). Emits requests; never flips its own state — every
    label reflects what MpcStatus says actually happened, the same honesty
    rule the ARM button follows.

    Two ways to begin, deliberately:
      START (hold)  the one-button mission — engage, warm up, fly the square,
                    with the CSV recording open from the engage. What the
                    operator asked for: press once, it goes and it logs.
      HOLD  (hold)  engage only (DP hold) — the calibration / station-keeping
                    mode, and the safe place to stage before a manual START.
    STOP / DISENGAGE / E-STOP are single clicks: a stop is never gated."""

    engage_requested = Signal(bool)
    traj_requested = Signal(bool)
    mission_requested = Signal()
    mode_requested = Signal(str)
    estop_requested = Signal()
    # {shape, origin_tag, size|length, size_y, speed} merged over
    # hw_mpc.yaml's square block (station omits the distance/speed keys)
    scenario_requested = Signal(object)
    # Raw localization recording (map.json + fixes.csv), independent of the
    # engage-gated MPC CSV: the operator can log a hand-flown survey pass.
    record_requested = Signal(bool)

    _BTN_CSS = "QPushButton{font-size:11px; padding:3px 8px;}"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MPC — reference vs actual (NED tag world)")
        self.view = TrajectoryView()

        self.mode_box = QtWidgets.QComboBox()
        # mpcc = contouring (theta is a solver decision on a C1 filleted
        # curve); dobmpcc adds the EAOB disturbance estimate to it, exactly
        # as dobmpc does to mpc.
        # *_tuned = the SAME tracking solver as dobmpc/mpc with its position
        # weight rotated into the path frame (control/path_cost.py): cheap on
        # along-track lag, expensive on cross-track, which is the knob against
        # corner cutting. No extra build — the suffix is a runtime cost_set.
        self.mode_box.addItems(["mpcc", "dobmpcc", "dobmpc", "mpc",
                                "dobmpc_tuned", "mpc_tuned", "pid"])
        self.mode_box.setToolTip(
            "mpcc/dobmpcc: contouring — theta is a solver state\n"
            "mpc/dobmpc: tracking NMPC, isotropic world-frame Q\n"
            "*_tuned: the same tracking NMPC with Q split into along-track "
            "(cheap) and cross-track (expensive) — suppresses corner "
            "cutting; weights in hw_mpc.yaml mpc_tuned:\n"
            "pid: the pole-placed baseline")
        self.mode_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.mode_box.currentTextChanged.connect(self.mode_requested)

        # ---- the mission: shape, where it starts, how far, how fast --------
        # These are what the operator changes between runs, so they are
        # on the panel rather than in the YAML. They are sent as a scenario
        # OVERRIDE (cmd_mpc_scenario) and merged over hw_mpc.yaml's square
        # block, so laps/depth/ramp still come from the file.
        self.shape_box = QtWidgets.QComboBox()
        self.shape_box.addItems(list(SHAPES))
        self.shape_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.shape_box.setToolTip(
            "station: sit on the tag and hold, heading included — fly THIS "
            "first\n"
            "line: out and back along +y from the tag (1 lap = there AND back)\n"
            "square: a rectangle with the tag as its first corner\n"
            "circle: the tag is the BOTTOM of the circle, not its centre — "
            "the box sets the radius\n"
            "follow: hold the relative pose you have RIGHT NOW on the object "
            "you clicked (needs --pose; the speed box is the setpoint's "
            "speed cap). Arming it never moves the vehicle.\n"
            "replay: re-fly a recorded handheld demo starting from the "
            "CURRENT pose (session + limits in hw_mpc.yaml replay:, or "
            "--replay-session). Arming it never moves the vehicle first.")
        self.tag_box = QtWidgets.QSpinBox()
        self.tag_box.setRange(0, 999)
        self.tag_box.setToolTip("the path starts directly above THIS tag id — "
                                "click and type, or use the arrows")
        self.len_box = QtWidgets.QDoubleSpinBox()
        # Default corner capture radius is 5 cm; a leg must be longer than two
        # capture radii or its start/end gates overlap. The worker validates
        # against the loaded config too; 0.15 m keeps the default UI honest.
        self.len_box.setRange(0.15, 10.0)
        self.len_box.setSingleStep(0.1)
        self.len_box.setDecimals(2)
        self.len_box.setSuffix(" m")
        self.len_box.setToolTip("line: how far along +y · square: the x side "
                                "(the tag is the min-x/min-y corner — bottom "
                                "left of this plot — and the sides run along "
                                "the map's x and y axes) · circle: the RADIUS "
                                "(the tag is the circle's bottom point, so "
                                "the centre sits this far up-plot from it)")
        self.leny_box = QtWidgets.QDoubleSpinBox()
        self.leny_box.setRange(0.15, 10.0)
        self.leny_box.setSingleStep(0.1)
        self.leny_box.setDecimals(2)
        self.leny_box.setSuffix(" m")
        self.leny_box.setToolTip("square: the y side")
        self.spd_box = QtWidgets.QDoubleSpinBox()
        # 0.50 m/s GUI ceiling — well above every mission flown so far
        # (hw_mpc.yaml ships 0.05; the sim squares used 0.12). [예측] nothing
        # has been flown faster, so localizer robustness (motion blur, tag
        # loss) above that is unmeasured; the spatial follower's corner brake
        # and the axis caps still bound what the vehicle actually does.
        self.spd_box.setRange(0.01, 0.50)
        self.spd_box.setSingleStep(0.01)
        self.spd_box.setDecimals(2)
        self.spd_box.setSuffix(" m/s")
        # ONE default for this field, and it is the config layer's
        # (MpcConfig.square["speed"], geometry.py). Without it the box would
        # start at its range MINIMUM (Qt's 0.0, clamped to 0.01) whenever the
        # YAML seed fails — a mission speed nobody chose.
        self.spd_box.setValue(0.12)
        self.spd_box.setToolTip("desired speed along the active path segment; "
                                "the spatial follower brakes to zero and "
                                "captures every corner before continuing")
        self.lbl_y = QtWidgets.QLabel("×")
        # NOT setStyleSheet(): a widget-level sheet REPLACES the application
        # sheet for that widget, so styling these three individually is what
        # dropped them back to Qt's pale default look (operator, 2026-08-14).
        # Size them through the font and let theme.py paint them.
        small = QtGui.QFont()
        small.setPointSize(8)
        for w in (self.shape_box, self.tag_box, self.len_box, self.leny_box,
                  self.spd_box):
            w.setFont(small)
        self.shape_box.currentTextChanged.connect(self._shape_changed)
        # TYPEABLE, but the keyboard always belongs to the pilot.
        #
        # Everything else in this package is NoFocus because the window is the
        # only key handler (window.py "Keyboard") — a focused widget would eat
        # W/A/S/D and the vehicle would stop answering with no visible cause.
        # These fields have to accept digits, so instead:
        #   * ClickFocus — focus only when deliberately clicked, never by Tab;
        #   * keyboardTracking off — one scenario per COMMITTED value, not one
        #     per keystroke ("7" then "79");
        #   * an event filter that hands the keyboard straight back the moment
        #     a key that is not part of typing a number arrives, and re-sends
        #     that key to the window so the pilot action still happens.
        for w in (self.tag_box, self.len_box, self.leny_box, self.spd_box):
            w.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
            w.setKeyboardTracking(False)
            w.installEventFilter(self)
            w.editingFinished.connect(w.clearFocus)
            w.valueChanged.connect(self._emit_scenario)
        self._num_fields = (self.tag_box, self.len_box, self.leny_box,
                            self.spd_box)

        self.btn_start = HoldToConfirmButton(
            "START", hold_ms=1000,
            tooltip="hold: engage, warm up, fly the square — recording (CSV) "
                    "starts at engage")
        self.btn_start.setStyleSheet(self._BTN_CSS)
        self.btn_start.confirmed.connect(self._start_mission)
        self.btn_hold = HoldToConfirmButton(
            "HOLD", hold_ms=1000,
            tooltip="hold: engage only — MPC holds the current pose (DP)")
        self.btn_hold.setStyleSheet(self._BTN_CSS)
        self.btn_hold.confirmed.connect(
            lambda: self.engage_requested.emit(True))
        self.btn_release = QtWidgets.QPushButton("DISENG")
        self.btn_release.setObjectName("Danger")
        self.btn_release.setStyleSheet(self._BTN_CSS)
        self.btn_release.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_release.setToolTip("instant: back to pilot control")
        self.btn_release.clicked.connect(
            lambda: self.engage_requested.emit(False))
        self.btn_stop = QtWidgets.QPushButton("STOP TRAJ")
        self.btn_stop.setStyleSheet(self._BTN_CSS)
        self.btn_stop.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_stop.setToolTip("instant: abandon the mission, hold here (DP)"
                                 " — works during a follow too")
        self.btn_stop.clicked.connect(lambda: self.traj_requested.emit(False))
        self.btn_estop = QtWidgets.QPushButton("E-STOP")
        self.btn_estop.setObjectName("Danger")
        self.btn_estop.setStyleSheet(self._BTN_CSS)
        self.btn_estop.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_estop.setToolTip("zero all axes, disable transmission — the "
                                  "same E-STOP as the header (Esc)")
        self.btn_estop.clicked.connect(self.estop_requested)
        self.btn_dr = QtWidgets.QPushButton("DR")
        self.btn_dr.setCheckable(True)
        self.btn_dr.setStyleSheet(self._BTN_CSS)
        self.btn_dr.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_dr.setToolTip(
            "show the IMU dead-reckoning overlay (amber). ON whenever an "
            "estimator is running — the experiment IS the two markers side by "
            "side. Untick it if a drifted trail buries the plot. Always shown "
            "while imu_dr is in CONTROL mode.")
        self.btn_dr.setVisible(False)          # only when a DR is reporting
        self.btn_dr.setChecked(True)           # ...and checked when it appears
        self.btn_dr.toggled.connect(self._dr_toggled)
        self.btn_3d = QtWidgets.QPushButton("3D")
        self.btn_3d.setCheckable(True)
        self.btn_3d.setStyleSheet(self._BTN_CSS)
        self.btn_3d.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_3d.setToolTip(
            "tilt the plot into 3-D: the same map, the same markers, with "
            "DEPTH.\nDrag orbits, wheel zooms, double-click resets. Each "
            "marker drops a stick to the tag plane so its depth is readable.\n"
            "This replaces the old floating MAP window, which showed the "
            "object in the CAMERA frame instead of the pool's.")
        self.btn_3d.toggled.connect(self._three_d_toggled)
        clear = QtWidgets.QPushButton("CLR")
        clear.setStyleSheet(self._BTN_CSS)
        clear.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        clear.setToolTip("clear the trails")
        clear.clicked.connect(self.view.clear)
        self.btn_rec = QtWidgets.QPushButton("REC NAV")
        self.btn_rec.setObjectName("Rec")
        self.btn_rec.setCheckable(True)
        self.btn_rec.setStyleSheet(self._BTN_CSS)
        self.btn_rec.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_rec.setToolTip("record raw localization: tag map + every fix "
                                "to <run folder>/nav_<hhmmss>/ — replot later "
                                "with  python -m rov_gui.tools.plot_nav_run")
        self.btn_rec.toggled.connect(self.record_requested)

        # The last START refusal and when it arrived — see add_status.
        self._ref_msg = ""
        self._ref_t = 0.0
        self.chip = QtWidgets.QLabel("not engaged")
        self.chip.setToolTip(
            "off = cross-track (+ left of travel) · lag = along-track "
            "(+ behind)\n"
            "z = the DEPTH being held, MAP frame, NED down-positive — a "
            "vehicle above the floor tags is negative.\n"
            "The bracket is the error: (+n cm) = sitting n cm DEEPER than "
            "the setpoint.\n"
            "STATION holds the z it had when START was pressed "
            "(hw_mpc.yaml square.depth_ned pins it to a number instead).")
        self.chip.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 11px;")

        row0 = QtWidgets.QHBoxLayout()
        row0.setSpacing(4)
        row0.addWidget(self.shape_box)
        # Kept on self so `_shape_changed` can hide it WITH its box. A label
        # left behind when its field goes away reads as a field that failed to
        # draw, which is worse than either state.
        self.lbl_tag = QtWidgets.QLabel("from tag")
        self.lbl_tag.setStyleSheet(f"color:{theme.TEXT_DIM}; font-size:11px;")
        row0.addWidget(self.lbl_tag)
        row0.addWidget(self.tag_box)
        row0.addWidget(self.len_box)
        row0.addWidget(self.lbl_y)
        row0.addWidget(self.leny_box)
        row0.addWidget(self.spd_box)
        self.mission_lbl = QtWidgets.QLabel("")
        self.mission_lbl.setStyleSheet(
            f"color:{theme.TEXT_FAINT}; font-size:11px;")
        row0.addWidget(self.mission_lbl, 1)

        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(4)
        row1.addWidget(self.mode_box)
        row1.addWidget(self.btn_start, 1)
        row1.addWidget(self.btn_hold)
        row2 = QtWidgets.QHBoxLayout()
        row2.setSpacing(4)
        row2.addWidget(self.btn_stop)
        row2.addWidget(self.btn_release)
        row2.addWidget(self.btn_estop)
        row2.addWidget(clear)
        row2.addWidget(self.btn_3d)
        row2.addWidget(self.btn_dr)
        row2.addWidget(self.btn_rec)
        row2.addStretch(1)
        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(self.chip)
        bottom.addStretch(1)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 6)
        lay.setSpacing(4)
        lay.addLayout(row0)
        lay.addLayout(row1)
        lay.addLayout(row2)
        lay.addWidget(self.view, 1)
        lay.addLayout(bottom)
        self._traj_on = False
        self._engaged = False
        self._nav_note = ""            # the localizer's last refusal, and
        self._nav_note_t = 0.0         # when — see add_fix
        self._shape_changed(self.shape_box.currentText())

    # ------------------------------------------------------------- keyboard
    _EDIT_KEYS = frozenset((
        Qt.Key.Key_Backspace, Qt.Key.Key_Delete, Qt.Key.Key_Left,
        Qt.Key.Key_Right, Qt.Key.Key_Home, Qt.Key.Key_End,
        Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Period,
        Qt.Key.Key_Comma, Qt.Key.Key_Minus, Qt.Key.Key_Return,
        Qt.Key.Key_Enter, Qt.Key.Key_Tab))

    def eventFilter(self, obj, ev):
        """Digits stay in the field; anything else is a PILOT key — release
        the focus and re-deliver it to the window, so reaching for W after
        typing a tag number flies the vehicle instead of doing nothing."""
        if (ev.type() == QtCore.QEvent.Type.KeyPress
                and obj in getattr(self, "_num_fields", ())):
            k = ev.key()
            if not (Qt.Key.Key_0 <= k <= Qt.Key.Key_9
                    or k in self._EDIT_KEYS):
                obj.clearFocus()
                win = self.window()
                if win is not None and win is not self:
                    QtWidgets.QApplication.sendEvent(win, ev)
                return True
        return super().eventFilter(obj, ev)

    # ---------------------------------------------------------- the mission
    def _start_mission(self) -> None:
        """START, with any HALF-TYPED mission number committed first.

        The numeric fields are ``keyboardTracking(False)`` (one scenario per
        committed value, not one per keystroke) and every button here is
        NoFocus — so typing "0.25" into the speed box and then clicking START
        without pressing Enter used to fly the PREVIOUS speed while the box
        showed 0.25. ``interpretText`` commits the typed text, which fires
        valueChanged -> ``_emit_scenario``; both signals reach the same worker
        object, so the scenario is queued ahead of the start and the run flies
        what the panel shows."""
        for w in self._num_fields:
            if w.isEnabled():
                w.interpretText()
            w.clearFocus()
        self.mission_requested.emit()

    def set_mode_default(self, mode: str) -> None:
        """Show the mode the WORKER is actually in.

        The combo's first entry is whatever happens to be first in the list,
        and Qt selects it on construction — so before this existed the panel
        displayed ``mpcc`` while hw_mpc.yaml had the worker on ``dobmpc``, and
        a whole pool session was flown and logged under a mode nobody chose
        (2026-08-17, sessions/.../20260817/0817_103431 — every meta says
        dobmpc). Blocked, because the panel must REFLECT state here, not
        command it: emitting would make the label true by changing the
        vehicle, which is the wrong direction for the honesty rule."""
        m = str(mode).lower()
        if self.mode_box.findText(m) < 0:
            return
        self.mode_box.blockSignals(True)
        self.mode_box.setCurrentText(m)
        self.mode_box.blockSignals(False)

    def set_mission_defaults(self, sq: dict) -> None:
        """Seed the controls from hw_mpc.yaml so the panel and the file agree
        at startup (the panel then owns these fields)."""
        boxes = (self.shape_box, self.tag_box, self.len_box, self.leny_box,
                 self.spd_box)
        for w in boxes:
            w.blockSignals(True)
        shape = str(sq.get("shape", "square")).lower()
        self.shape_box.setCurrentText(shape if shape in SHAPES else "square")
        tag = sq.get("origin_tag")
        self.tag_box.setValue(0 if tag in (None, "") else int(tag))
        size = float(sq.get("size", 1.0) or 1.0)
        # ONE box, three meanings — so it is seeded from the key the SELECTED
        # shape actually reads, never from whichever key happens to exist.
        self.len_box.setValue({
            "line": float(sq.get("length", 2.0) or 2.0),
            "circle": float(sq.get("radius", 0.5) or 0.5),
        }.get(shape, size))
        sy = sq.get("size_y")
        self.leny_box.setValue(size if sy in (None, "") else float(sy))
        self.spd_box.setValue(float(sq.get("speed", 0.12) or 0.12))
        for w in boxes:
            w.blockSignals(False)
        self._shape_changed(self.shape_box.currentText())

    def _shape_changed(self, shape: str) -> None:
        station = (shape == "station")
        follow = (shape == "follow")
        # REPLAY takes NO panel numbers at all: the mission is a recorded
        # demo (hw_mpc.yaml replay.session / --replay-session), its speed cap
        # is replay.v_max_m_s, and it anchors at the vehicle's current pose —
        # so every field this shape does not use vanishes (the follow rule:
        # one greyed-out survivor reads as "this one still matters somehow").
        replay = (shape == "replay")
        self.leny_box.setVisible(shape == "square")
        self.lbl_y.setVisible(shape == "square")
        # FOLLOW has no distance and no origin tag: what it holds is the
        # offset the vehicle already has, and the "path" is wherever the
        # object goes. The SPEED box stays — for a follow it caps how fast
        # the setpoint may walk after the object, which is the one number the
        # operator still chooses.
        self.len_box.setVisible(not station and not follow and not replay)
        # ...and the tag box GOES AWAY for a follow rather than merely greying
        # out (2026-08-23, operator: "follow에서 왜 tag id 입력하는 란이 있어?").
        # It was disabled and its value dropped from the scenario, so it never
        # did anything — but every other field this shape does not use vanishes,
        # and one greyed-out survivor reads as "this one still matters somehow".
        # A follow is anchored to the OBJECT and to the pose the vehicle has at
        # arm; no tag id enters it. (The tag map is still what puts the object
        # in the map frame — but through whatever tags that FRAME saw, never
        # through one id the operator picked.)
        self.tag_box.setVisible(not follow and not replay)
        self.lbl_tag.setVisible(not follow and not replay)
        # station does not move; replay's speed is replay.v_max_m_s (config)
        self.spd_box.setVisible(not station and not replay)
        # The one distance box means a length, a side, or a RADIUS depending
        # on the shape, and 0.50 m of side and 0.50 m of radius are missions
        # of very different size. The prefix says which is on screen, so the
        # number is never ambiguous at a glance; mission_lbl spells it out.
        self.len_box.setPrefix("r " if shape == "circle" else "")
        self._emit_scenario()

    def _emit_scenario(self) -> None:
        shape = self.shape_box.currentText()
        tag = int(self.tag_box.value())
        d = {"shape": shape, "origin_tag": (None if tag == 0 else tag)}
        if shape == "follow":
            # The origin tag means nothing here, so it is not sent: a follow
            # anchored to a tag id would be a promise the mission cannot keep.
            d.pop("origin_tag", None)
            d["speed"] = float(self.spd_box.value())
            self.mission_lbl.setText(
                f"hold this offset on the clicked object, setpoint <= "
                f"{self.spd_box.value():.2f} m/s")
        elif shape == "replay":
            # No tag, no numbers: the demo anchors at the CURRENT pose and
            # everything else lives in hw_mpc.yaml's replay: block.
            d.pop("origin_tag", None)
            self.mission_lbl.setText(
                "re-fly the recorded demo from HERE "
                "(hw_mpc.yaml replay.session / --replay-session)")
        elif shape == "station":
            self.mission_lbl.setText(
                ("hold on tag %d" % tag if tag else "hold here")
                + ", facing +y")
        elif shape == "line":
            d["length"] = float(self.len_box.value())
            d["speed"] = float(self.spd_box.value())
            self.mission_lbl.setText(
                f"out {self.len_box.value():.2f} m +y and back @ "
                f"{self.spd_box.value():.2f} m/s"
                + (f", from tag {tag}" if tag else ", from here"))
        elif shape == "square":
            d["size"] = float(self.len_box.value())
            d["size_y"] = float(self.leny_box.value())
            d["speed"] = float(self.spd_box.value())
            self.mission_lbl.setText(
                f"{self.len_box.value():.2f} m +x × "
                f"{self.leny_box.value():.2f} m +y @ "
                f"{self.spd_box.value():.2f} m/s"
                + (f", SW corner at tag {tag}" if tag else ", SW corner here"))
        elif shape == "circle":
            d["radius"] = float(self.len_box.value())
            d["speed"] = float(self.spd_box.value())
            # Spelled out as "bottom of the circle", not "on the circle": the
            # operator's whole point (2026-08-17) was that the tag is NOT the
            # centre, and a label that only said "at tag N" would leave the
            # reader to guess which of the two it meant.
            self.mission_lbl.setText(
                f"circle r {self.len_box.value():.2f} m @ "
                f"{self.spd_box.value():.2f} m/s"
                + (f", tag {tag} at its bottom" if tag
                   else ", bottom of it here"))
        self.scenario_requested.emit(d)

    def _hold_z_text(self) -> str:
        """`· z -1.05 m (+2 cm)` — the depth being held, and the error.

        STATION mode already commanded all three axes (workers.py
        ``set_target_ned((x, y, depth), yaw)``, depth = the z at the moment
        START was pressed), but NOTHING on this panel was about z: ``err_xy``
        is horizontal by definition, so a hold that had sagged 20 cm read as
        a perfect one. Operator request 2026-08-18.

        The number is in the MAP frame — the same frame the plot, the tag map
        and the pool are drawn in — so it can be compared with the surveyed
        operating band (-1.30 .. -0.54 m, README). NED z is down-positive, so
        a vehicle swimming above the floor tags is NEGATIVE, and `(+n cm)`
        means it is sitting that much DEEPER than the setpoint.

        Since 2026-08-23 `p_ref` is ALREADY a map z — `_to_map_z` converts at
        the boundary, for every marker — so this method no longer adds the
        offset itself. It was the ONLY place that ever did, which is exactly
        why the text stayed right while the picture drew the vehicle lying on
        the tag mat."""
        v = self.view
        if v.p_ref is None:
            return ""                     # no reference, no readout
        # `z_d` SURVIVES the boundary change. Engaged with no datum yet — the
        # window has not handed one over, or this is a bench status — means
        # `_to_map_z` had nothing to add and the stored number is still
        # datum-relative. Saying so is the point: a datum-relative depth
        # printed as a map depth is off by however deep the vehicle engaged.
        tag = "z" if v.datum is not None else "z_d"
        z_map = float(v.p_ref[2])
        dz = ("" if v.p_act is None
              else f" ({(v.p_act[2] - v.p_ref[2]) * 100:+.0f} cm)")
        return f" · {tag} {z_map:+.2f} m{dz}"

    def _three_d_toggled(self, on: bool) -> None:
        self.view.set_three_d(on)

    def _dr_toggled(self, on: bool) -> None:
        self.view.show_dr = bool(on)
        self.view.update()

    def set_recording(self, on: bool) -> None:
        """Reflect what the window's recorder ACTUALLY did (honesty rule: the
        button never shows a recording that failed to open)."""
        self.btn_rec.blockSignals(True)
        self.btn_rec.setChecked(on)
        self.btn_rec.setText("REC ●" if on else "REC NAV")
        self.btn_rec.blockSignals(False)

    # ------------------------------------------------------------- data in
    def set_object(self, fx) -> None:
        """One ObjectFix, straight through — the panel adds nothing."""
        self.view.set_object(fx)

    def add_fix(self, f: NavFix) -> None:
        self.view.add_fix(f)
        # A localizer that is FAILING explains itself here (full reason — the
        # sensor row elides): outside an engagement the chip has nothing more
        # important to say.
        #
        # REMEMBERED, not just written. `add_status` runs at 20 Hz against
        # this method's camera rate and its else-branch writes the same label,
        # so a note set here used to be overwritten inside 50 ms — the reason
        # the localizer was rejecting reached the screen and was gone before
        # anyone could read it (2026-08-21: nine mapped tags in view, no fix,
        # and nothing on screen saying why).
        if not f.ok and f.note:
            self._nav_note = str(f.note)
            self._nav_note_t = time.monotonic()
            self.view._nav_note = self._nav_note
        elif f.ok:
            self._nav_note = ""
            self.view._nav_note = ""
        if not f.ok and f.note and not self._engaged:
            self.chip.setText(f"nav: {f.note}")
            self.chip.setStyleSheet(
                f"color: {theme.WARN}; font-size: 11px;")

    def add_status(self, s: MpcStatus) -> None:
        self.view.add_status(s)
        # The DR toggle only exists when there IS a dead-reckoner reporting;
        # in CONTROL mode it is shown checked and disabled, because that
        # overlay is the instrument the run is being flown on.
        live = bool(s.dr_mode)
        self.btn_dr.setVisible(live)
        if live and s.dr_mode == "control" and not self.btn_dr.isChecked():
            self.btn_dr.setChecked(True)
        self.btn_dr.setEnabled(live and s.dr_mode != "control")
        self._engaged = s.engaged
        self._traj_on = s.traj_on
        kind_now = (s.scenario or {}).get("kind")
        self.btn_start.setEnabled(not s.traj_on)
        self.btn_hold.setEnabled(not s.engaged)
        # A FOLLOW has no trajectory clock (traj_on stays False, exactly like
        # STATION), so keying STOP on traj_on alone left the only way to end
        # one as DISENGAGE — which drops depth hold on a negatively buoyant
        # vehicle. set_traj(False) already re-targets `self._eta`, so this
        # gives "stop following, hold right here" for free.
        self.btn_stop.setEnabled(s.traj_on or kind_now == "follow")
        self.mode_box.setEnabled(not s.engaged)
        # The mission is frozen from START, not from take-off. The worker
        # SNAPSHOTS the merged mission the moment START is pressed
        # (workers.py set_traj -> self._approach["sq"]) and arms the path from
        # that copy after the approach and the settle, so a field edited while
        # the vehicle is on its way to the tag updates this panel's label and
        # the worker's override and then never reaches the flight. Keyed on
        # traj_on alone that was a window of approach_max_s + settle_s (up to
        # ~190 s) in which the panel said one thing and the run flew another.
        busy = s.traj_on or s.phase in ("approach", "settle")
        for w in (self.shape_box, self.tag_box, self.len_box, self.leny_box,
                  self.spd_box):
            w.setEnabled(not busy)
        if s.engaged:
            warm = (f" · warm-up {s.warmup_left_s:.1f}s"
                    if s.warmup_left_s > 0 and s.phase != "warmup" else "")
            # WHICH PHASE, always — "on its way to the tag" and "flying the
            # path" looked identical before (operator, 2026-08-14).
            kind = (s.scenario or {}).get("kind", "square")
            what = {"approach": "GOING TO START",
                    "settle": "SETTLING",
                    "station": "STATION HOLD",
                    "follow": "FOLLOWING",
                    "warmup": "WARMING UP"}.get(
                        s.phase, kind.upper() if s.traj_on else "DP HOLD")
            det = f" {s.phase_detail}" if s.phase_detail else ""
            if s.err_cross is not None and s.err_along is not None:
                err = (f" · off {s.err_cross * 100:+.0f} cm"
                       f" · lag {s.err_along * 100:+.0f} cm")
            else:
                err = "" if s.err_xy is None else f" · err {s.err_xy * 100:.0f} cm"
            # Live SPEED, actual vs what the reference is asking for. The
            # 2026-08-17 session lost a run to a speed box that was not in the
            # loop; the two numbers side by side make that visible at a glance.
            if s.speed_m_s is not None:
                err += f" · {s.speed_m_s:.2f}"
                if s.ref_speed_m_s is not None:
                    err += f"/{s.ref_speed_m_s:.2f}"
                err += " m/s"
            err += self._hold_z_text()
            # THE BRIDGE HAS TO BE LOUD. "still engaged" and "engaged but no
            # longer holding position" look identical otherwise, and the coast
            # tier is indefinite — the operator is the thing deciding when it
            # has gone on too long, so the chip has to say so in a colour.
            tier = getattr(s, "bridge_tier", "none")
            if tier == "coast":
                what = "NO TAG — COASTING"
            elif tier == "imu":
                what = f"{what} · NO TAG (IMU)"
            # THE FOLLOW LADDER HAS TO BE LOUD, for the bridge's reason:
            # "still following" and "holding the last setpoint because the
            # object went away" look identical otherwise, and only one of them
            # is a run still doing what the operator asked for.
            fs = getattr(s, "follow_state", "")
            bad = tier != "none"
            if fs == "lost":
                what, bad = "OBJECT LOST — HOLDING", True
            elif fs == "stale":
                age = getattr(s, "follow_age_s", None)
                what = ("OBJECT STALE" if age is None
                        else f"OBJECT STALE {age:.1f}s")
            elif fs == "leashed":
                what = "FOLLOWING · EXCURSION LIMIT"
            # A REFUSED START HAS TO REACH THE CHIP. `_refuse` only ever set
            # `s.reason`, and while engaged this label shows the PHASE instead
            # — so pressing START, having it refused, and going on flying a DP
            # hold looked exactly like a follow that had armed. On 2026-08-23
            # the operator watched the thrusters work for minutes and read it
            # as following; the vehicle was station-keeping and the object was
            # never in the loop. Six seconds in red, over the phase.
            if s.reason.startswith("START refused") and s.reason != self._ref_msg:
                self._ref_msg, self._ref_t = s.reason, time.monotonic()
            fresh_refusal = (self._ref_msg
                             and time.monotonic() - self._ref_t < 6.0)
            if fresh_refusal:
                self.chip.setText(f"ENGAGED [{s.mode}] {what} · "
                                  f"{self._ref_msg}")
            else:
                self.chip.setText(f"ENGAGED [{s.mode}] {what}{det}{err}{warm}")
            self.chip.setStyleSheet(
                f"color: {theme.FAIL if (bad or fresh_refusal) else theme.WARN}; "
                f"font-size: 11px; font-weight: 600;")
        elif (self._nav_note
              and time.monotonic() - self._nav_note_t < 2.0):
            # The localizer is refusing and nothing is engaged: THAT is the
            # most important thing this label can say. Yielding it back to
            # `s.reason` (which is "not engaged", i.e. no information at all)
            # is what made the miss reason unreadable.
            self.chip.setText(f"nav: {self._nav_note}")
            self.chip.setStyleSheet(f"color: {theme.WARN}; font-size: 11px;")
        else:
            self.chip.setText(s.reason or "not engaged")
            self.chip.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 11px;")
