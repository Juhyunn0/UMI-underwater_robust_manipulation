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
import os
import time
from collections import deque

from .. import theme
from ..qt import (QColor, QPainter, QRectF, Qt, QtCore, QtGui, QtWidgets,
                  Signal)
from ..control.geometry import SHAPES
from ..state import MpcStatus, NavFix, ObjectFix, PolicyStatus
from .indicators import HoldToConfirmButton

#: HIGH level = the mission SHAPE: what generates the reference. The KEYS are
#: `geometry.SHAPES`, UNCHANGED — the worker, the scenario dict, hw_mpc.yaml
#: and every record keep saying `policy`/`station`/...; only the combo's TEXT
#: changed (2026-09-11, operator request: "high level: diffusion policy /
#: Station / Line ..."). Item order = SHAPES order, station first: it is the
#: shape to fly FIRST in a session and what a bare panel shows, and it keeps
#: the hold-to-confirm START as the default (a policy START is instant).
SHAPE_LABELS = {"station": "Station", "line": "Line", "square": "Square",
                "circle": "Circle", "follow": "Follow", "replay": "Replay",
                "policy": "Diffusion Policy"}
assert set(SHAPE_LABELS) == set(SHAPES), "a shape without a label, or a label without a shape"
#: LOW level = the FOLLOWER: who commands the thrusters. The KEYS are what
#: `bus.cmd_mpc_mode` carries and `MpcWorker.MODES` accepts. `none` is TELEOP
#: (2026-09-11, operator request): the station commands NOTHING, the pilot
#: flies on the joystick, and a Diffusion Policy mission still infers and
#: draws its plans — exactly the audited --policy-observe path, now a runtime
#: choice (the flag survives as a launch alias). First in the list because it
#: is the one LOW level that cannot move the vehicle by itself.
MODE_LABELS = {"none": "None", "pid": "PID", "mpc": "MPC",
               "mpc_tuned": "MPC_Tuned", "dobmpc": "DOBMPC",
               "dobmpc_tuned": "DOBMPC_Tuned", "mpcc": "MPCC",
               "dobmpcc": "DOBMPCC", "rl": "RL",
               # 2026-09-30: the per-thruster RL policy (8 raw pulses through
               # the vehicle's Lua override, mixer bypassed). Last, after RL.
               "rl_pwm": "RL_PWM"}
#: The ckpt row shows a BASENAME, elided past this many characters (the full
#: path is the tooltip). Character-based, not pixel-based, so the row's width
#: never depends on the font — the panel is a fixed grid cell.
CKPT_NAME_CHARS = 36


def elide_ckpt_name(name: str, limit: int = CKPT_NAME_CHARS) -> str:
    """Elide the MIDDLE, keeping head and tail. The data/checkpoints/ entries
    (2026-09-14) read ``<YYYYMMDD_HHMMSS>_<task>_<exp>.ckpt``: the stamp at
    the front and the experiment label at the back are the two halves that
    tell two runs apart, so a tail-elision (``20260907_113747_umi_depth_5d…``)
    would show the same text for every run of a day's ablation."""
    name = str(name)
    if len(name) <= limit:
        return name
    head = (limit - 1) // 2
    tail = limit - 1 - head
    return name[:head] + "…" + name[-tail:]

TRAIL_MAX = 4800          # hard cap on stored points (memory bound)
#: How far back the trails go. 3 s, not the 90 s this shipped with
#: (operator, 2026-09-03): at 90 s the vehicle's own history filled the plot
#: and buried the thing the panel is currently FOR — the policy's proposal,
#: which is about 1 s long. A 3 s tail still shows which way the hull is
#: moving and how fast, which is all the trail was ever read for at this
#: zoom, and it is the same order as the plan it now sits beside.
TRAIL_AGE_S = 3.0
#: How many recent diffusion-policy plans the plot keeps. Small on purpose:
#: they arrive every ~0.5 s and each is re-anchored within centimetres of the
#: hull, so ANY history draws a smear rather than a readable proposal. The
#: 2026-09-03 pool run settled it: at 60 (the observe history this briefly
#: had) 218 plans' worth of cyan filled the vehicle and the operator could
#: not tell the network's output from the hull's own motion. TWO — the
#: current one and the one before it — is what was asked for, and it is also
#: the only history that answers the question the panel is for: "is this
#: plan like the last one, or did the network just jump?"
POLICY_PLAN_KEEP = 2
#: Seconds before a plan is dropped even if nothing has replaced it. With a
#: 2-deep history the deque normally evicts long before this; the cap is what
#: clears the plot when the STREAM STOPS, so a halted policy does not leave
#: its last two proposals sitting there looking live.
POLICY_PLAN_AGE_S = 45.0
#: The proposals' own colour and weight. RED and thick (operator,
#: 2026-09-03): fluorescent yellow was the first try and it lost twice — to
#: the trail's own age ramp, whose "now" end is a yellow-green, and to the
#: green hull it is usually drawn across. A saturated red is the only strong
#: hue left on this plot; `theme.FAIL` (#ef4444) already carries the 1 px
#: tracking-error line, so this is a purer, brighter one and four times as
#: wide — width is what actually separates them.
POLICY_PLAN_COLOR = "#ff1f1f"
POLICY_PLAN_W = 5          # the current plan
POLICY_PLAN_W_PREV = 3     # the one before it

# --------------------------------------------------------------- the z gauge
#: THE DEPTH GAUGE'S FIXED WINDOW, map-frame metres as (shallow, deep). Map z
#: is DOWN-positive and everything this station flies is ABOVE the tag plane,
#: so the first number is the TOP of the scale and the more negative one.
#:
#: DESIGN CONSTANTS [예측] — chosen to bracket the record with headroom; they
#: are not themselves measurements. What they bracket, both re-measured
#: 2026-09-07 for this change:
#:   * every ACCEPTED tag fix of the post-extrinsic-fix era, z_ned
#:     -0.856 .. -0.165 m (p1 -0.802, p50 -0.229, p99 -0.192)
#:     [측정: data/2026090*/*/nav_*/fixes.csv,
#:      32 files, 44,548 rows with ok=1]
#:   * every DRAWN diffusion-policy plan knot, z_ned -0.754 .. +0.304 m
#:     [측정: data/2026090*/*/policy_plan.csv,
#:      18 files, 28,308 rows]
#: so the window clears the shallowest fix by 0.244 m and the deepest drawn
#: knot by 0.096 m.
#:
#: DO NOT cite config/hw_nav.yaml's -1.30 .. -0.54 m band here: those runs
#: predate the cam_t_flu fix (0.26 m, and the SIGN was wrong) and
#: `z_source: tag`, so memory c3-extrinsic-from-gripper forbids pooling them
#: with the numbers above.
Z_GAUGE_WINDOW_M = (-1.10, 0.40)
#: The tick ladder. No 0.25: thinning must never RENUMBER a surviving tick,
#: and 0.25 is the one rung that is not an integer multiple of its neighbours
#: — stepping 0.2 -> 0.25 moves every tick on the scale.
Z_GAUGE_STEPS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
Z_GAUGE_TICK_PX = 26.0     # minimum VERTICAL pitch between drawn ticks
Z_GAUGE_TOP_PX = 22.0      # below the "3D az .. el .." caption baseline (y=12)
Z_GAUGE_BOT_PX = 40.0      # above the trail-age legend row (height() - 14)
Z_GAUGE_MIN_H_PX = 40.0    # a band shorter than this is not a scale; draw none
Z_GAUGE_EDGE_PX = 6.0      # right margin, the same one the legend uses
Z_GAUGE_GAP_PX = 8.0       # tick arm -> its label


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
        # C3 mount angle, degrees DOWN from level. The window overrides it
        # from hw_nav.yaml `cam_tilt_deg` — the SAME key the localizer
        # applies — so the optical-axis ray on the plot cannot point
        # somewhere the solver does not think it points.
        self.cam_tilt_deg = 43.3
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
        # The same measurement against the LEARNED depth map (--fstereo), kept
        # apart from the device one on purpose: they are two instruments, and a
        # ratio printed without saying which one it came from is not evidence.
        self.depth_chk_fs: tuple | None = None
        # Latches once learned depth reports anything at all. The labels key on
        # THIS, not on whether both accumulators currently hold a number: a row
        # must not lose the instrument's name just because the other one was
        # cleared a moment ago.
        self._depth_two_source = False
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
        #: THE DEPTH GAUGE'S WINDOW, and the ONE thing about the gauge that is
        #: allowed to move. It starts at `Z_GAUGE_WINDOW_M` and only ever
        #: GROWS — never shrinks, never re-centres — so a vehicle that leaves
        #: the design band stays on the scale (operator's choice, 2026-09-07)
        #: without the every-frame renumbering that made the old ruler
        #: unreadable. Monotone is what makes it fixed IN PRACTICE: it can
        #: change a handful of times in a session and then never again.
        #: `clear()` puts it back.
        self._z_window = Z_GAUGE_WINDOW_M
        # THE DIFFUSION POLICY'S OWN OUTPUT (2026-09-02, operator request).
        # The last few composed plans, newest last, kept whatever the filter
        # decided — a REJECTED plan is the one worth looking at, and on the
        # runs so far that is nearly all of them. Bounded because they arrive
        # at ~2 Hz for the life of the mission.
        self.policy_plans: deque = deque(maxlen=POLICY_PLAN_KEEP)
        #: True while MpcStatus says this run's controller output is muted
        #: (POLICY OBSERVE). Drives the plan history depth and stops the
        #: vehicle-to-reference line from being drawn in the failure colour:
        #: with the loop open that gap is not an error, it is the operator
        #: standing where they chose to stand.
        self.observe = False
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

    def set_depth_check(self, ratio, spread, source: str = "dev",
                        held_s: float = 0.0) -> None:
        """(median, p10..p90 spread) of depth-map / expected-floor-range over
        the mat — see window._check_depth_scale.

        ``source`` is "dev" (the camera's own stereo) or "fs"
        (FoundationStereo). Only ONE of them can be live at a time, because
        only one depth map is on screen at a time; ``held_s`` is how long ago
        the other one was measured, and it is drawn, because a held number and
        a live number sitting on adjacent lines would otherwise read as two
        simultaneous measurements of one scene.

        ``ratio=None`` clears that source.
        """
        value = (None if ratio is None
                 else (float(ratio), float(spread), float(held_s)))
        if source == "fs":
            if value is not None:
                self._depth_two_source = True
            self.depth_chk_fs = value
        else:
            self.depth_chk = value

    def _readout_depth_lines(self) -> list:
        """The depth-vs-MAP rows, as drawn. One place, so a test can call it.

        Two independent sensors describing one world, printed as their
        disagreement: `window._check_depth_scale` compares the depth map on the
        panel against the tag PnP, which never touches depth. 1.00x = that
        depth path is metric.
        """
        out = []
        for tag, chk in (("dev", self.depth_chk), ("FS ", self.depth_chk_fs)):
            if chk is None:
                continue
            r, spread, held_s = chk
            bad = abs(r - 1.0) > 0.10
            # ALWAYS qualified once learned depth has been seen this session.
            # Keying the label on "is the other accumulator populated" dropped
            # the instrument's name in exactly the state that follows every
            # swap into FoundationStereo.
            label = ("depth-vs-MAP" if not self._depth_two_source
                     else f"depth-vs-MAP {tag}")
            # THE SPREAD IS NOT DECORATION. A calibration scale is one number;
            # if the middle 80% of the floor disagrees by more than ~0.15 the
            # depth map is not merely mis-scaled, it is mis-SHAPED (or the mat
            # is not the only thing in view), and the headline ratio should not
            # be quoted as a constant.
            age = f"  (held {held_s:.0f}s)" if held_s > 0.0 else ""
            # The NOT-metric flag belongs to the ratio, not to its freshness.
            out.append(f"{label}  {r:4.2f}x  (+/-{spread:4.2f}){age}"
                       + ("  <- depth NOT metric" if bad else ""))
        return out

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

    def add_policy_plan(self, v) -> None:
        """One composed diffusion-policy plan, in the DATUM frame.

        Transformed here, once, exactly like `add_status` does for the vehicle
        marker — the datum can move between plans, and storing raw would draw
        old plans in a frame that no longer exists.
        """
        try:
            px, py, pz = v.p_ned
            pts = []
            for x, y, z in zip(px, py, pz):
                mx, my = self._to_map(float(x), float(y))
                pts.append((mx, my, self._to_map_z(float(z))))
        except (TypeError, ValueError, AttributeError):
            return
        if len(pts) < 2:
            return
        self.policy_plans.append({
            "pts": pts,
            "status": str(getattr(v, "status", "")),
            "id": int(getattr(v, "plan_id", 0)),
            "t": time.monotonic(),
        })
        self.update()

    def _prune_policy_plans(self) -> None:
        t = time.monotonic()
        while (self.policy_plans
               and t - self.policy_plans[0]["t"] > POLICY_PLAN_AGE_S):
            self.policy_plans.popleft()


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
        self.observe = bool(getattr(s, "observe", False))
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
        if s.scenario and s.scenario.get("kind") in ("follow", "replay",
                                                     "policy"):
            # None has a PLACED geometry — a follow's path is wherever the
            # object goes, a replay's / a policy's is a streamed plan drawn
            # live by the reference trail. Said explicitly rather than
            # falling through, so a stale rectangle outline from an earlier
            # mission cannot survive underneath any of them.
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
        self.policy_plans.clear()
        # The gauge's window is a per-run latch, so a cleared plot gets a
        # cleared scale — otherwise one excursion would widen every later run.
        self._z_window = Z_GAUGE_WINDOW_M
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

    def _view_f(self):
        """The look direction (eye -> centre) in MAP coordinates.

        The third vector of `_basis3`'s frame, recomputed here rather than
        returned from it so that method's signature — and its callers — stay
        as they are. `dot(d, f)` is DEPTH: larger = farther from the eye,
        which is what the painter's-algorithm sort and the back-face test in
        `_draw_rov` both need.
        """
        az = math.radians(self.azimuth_deg)
        el = math.radians(self.elev_deg)
        ce, se = math.cos(el), math.sin(el)
        return (ce * math.cos(az), ce * math.sin(az), se)

    def _draw_rov(self, painter, p_ned, yaw: float, base: str,
                  ghost: bool = False) -> bool:
        """The vehicle as a BODY, gripper included, in whichever mode we are in.

        Returns True when it drew the full body; False when the vehicle is
        too few pixels across to be worth more than the flat footprint the
        panel has always drawn, so the caller can fall back.

        WHY THE GRIPPER IS HERE AT ALL: the diffusion policy's TCP is the
        JAW (`rov_shape.tcp_body_m` == hw_mpc.yaml `policy.tcp_body_flu_m`),
        0.50 m ahead of the COM this marker is centred on. Drawing only the
        hull put the vehicle's picture half a metre from the point its
        reference was composed at, which is the one relationship an operator
        watching a DP run is trying to read.

        THE 3-D PATH is orthographic like everything else in `_px`: faces are
        back-face culled per part (each part is convex, so that is exact
        within a part) and then sorted far-to-near across parts, which is the
        painter's algorithm — it can only be wrong where two parts
        interpenetrate, and these do not. Shading is |dot(normal, view)|,
        i.e. a face square to the eye is brightest. No light position, no
        specular: this is an instrument, and the only job of the shading is
        to make the box read as a box.
        """
        from .rov_shape import body_to_map, parts as rov_parts, tcp_body_m

        hull_px = abs(self._px(0.0, 0.0).x()
                      - self._px(0.0, self.rov_size_m[1]).x())
        if hull_px < 14.0:
            # Below this the four boxes are a smudge and the gripper is
            # sub-pixel — the flat outline says strictly more. Same threshold
            # spirit as the DR ghost's existing 10 px cut.
            return False

        model = rov_parts(hull_l=self.rov_size_m[0], hull_w=self.rov_size_m[1],
                          cam_tilt_deg=self.cam_tilt_deg)
        if hull_px < 48.0:
            # DETAIL FLOOR. Eight shrouds at 47 mm are ~4 px each here; drawn
            # they are a ring of specks around the hull that reads as noise,
            # and every one of them costs six more sorted quads at 20 Hz.
            # The gripper stays at every size the body is drawn at — it is
            # the part the operator is actually looking for.
            model = [q for q in model if q["kind"] != "thruster"]
        f = self._view_f()

        def _map(v):
            return body_to_map(v, p_ned, yaw)

        if not self.three_d:
            # TOP-DOWN. The hull keeps its historical outline exactly (the
            # test pins it corner for corner); the gripper is drawn flat on
            # top of it, which is what it looks like from above.
            self._draw_rov_flat(painter, p_ned, yaw, model, base, ghost)
            return True

        # 3-D: gather every visible face with its depth and shade.
        hull_faces, grip_faces = [], []
        for part in model:
            vm = [_map(v) for v in part["verts"]]
            for idx, n_flu in part["faces"]:
                # The normal rotates with the body but does not translate,
                # so it goes through the SAME conversion about the origin —
                # never a hand-written copy of the two sign flips.
                nm = body_to_map(n_flu, (0.0, 0.0, 0.0), yaw)
                facing = nm[0] * f[0] + nm[1] * f[1] + nm[2] * f[2]
                if facing >= 0.0:
                    continue                  # points away from the eye
                pts = [vm[i] for i in idx]
                depth = sum(q[0] * f[0] + q[1] * f[1] + q[2] * f[2]
                            for q in pts) / len(pts)
                # The camera HOUSING goes with the hull, translucent: it is
                # context. What the operator reads are the two LINES — the
                # heading line and the optical-axis ray — drawn after
                # everything at full strength; the C3 is a third of the
                # hull's length, and painted opaque on top it became the
                # loudest thing on the plot.
                bucket = (grip_faces if part["kind"] in ("tube", "jaw")
                          else hull_faces)
                bucket.append((depth, part["kind"], abs(facing), pts))
        hull_faces.sort(key=lambda t: -t[0])       # far first, within the hull
        grip_faces.sort(key=lambda t: -t[0])

        def _paint(group, alpha0, alpha1):
            for _d, _kind, shade, pts in group:
                path = QtGui.QPainterPath()
                path.moveTo(self._px(*pts[0]))
                for q in pts[1:]:
                    path.lineTo(self._px(*q))
                path.closeSubpath()
                if ghost:
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.setPen(QtGui.QPen(_c(base, 150), 1,
                                              Qt.PenStyle.DashLine))
                else:
                    painter.setBrush(
                        _c(base, int(alpha0 + (alpha1 - alpha0)
                                     * min(1.0, shade))))
                    painter.setPen(QtGui.QPen(_c(base, 210), 1))
                painter.drawPath(path)

        # THE GRIPPER IS DRAWN AFTER THE HULL, ALWAYS — not sorted with it.
        # Physically it hangs UNDER the hull's front, so from the default eye
        # (azimuth 0 = behind, elevation 55 = looking down) the hull really
        # would hide it. That is faithful and useless: the jaw is the
        # policy's TCP and the whole reason the gripper is drawn at all is
        # for the operator to see where it is relative to the reference. So
        # the hull stays translucent (40..150) and the tool is painted on
        # top, opaque (150..235). The cost is that the picture is a cutaway
        # rather than a photograph, and the tube crossing the hull outline is
        # what a cutaway looks like.
        _paint(hull_faces, 40, 150)
        _paint(grip_faces, 150, 235)
        # In 3-D both lines are drawn: the heading line lies in the
        # vehicle's horizontal plane through the COM, the optical-axis ray
        # leaves the lens pitched 43 deg down, and from an orbited eye the
        # two are visibly different things. (Top-down and the zoomed-out
        # fallback get their heading line from paintEvent instead, AFTER the
        # outline — see _draw_rov_heading.)
        if self.yaw_ned is not None:
            self._draw_rov_heading(painter, p_ned, yaw, base, ghost)
        self._draw_rov_optic(painter, p_ned, yaw, base, ghost)
        self._draw_rov_tcp(painter, _map(tcp_body_m()), base, ghost)
        return True

    def _draw_rov_heading(self, painter, p_ned, yaw: float, base: str,
                          ghost: bool) -> None:
        """The vehicle's axis: a dotted line through the COM along body
        +x, HEADING_AFT_M behind it and HEADING_FWD_M ahead of it.

        Operator request 2026-09-07. Until then the only line on the
        vehicle was the optical-axis ray, which from above starts at the
        lens (over the gripper tube) and stops 0.63 m ahead of the COM — it
        read as "the heading line", and a heading line that exists only in
        front of the gripper cannot be sighted along. What the operator is
        judging on a DP run is whether the plan, the reference and the
        object lie ON the vehicle's axis or beside it, and that needs the
        axis drawn through the whole body, both ways. Same pen as the ray
        (base colour, dotted, 1 px) so it reads as part of the vehicle and
        never as a series; the ghost draws neither, so it stays subordinate.

        WHO CALLS IT: `_draw_rov` in 3-D (nothing is painted over the body
        there), and paintEvent on every flat path — top-down with the body,
        and the zoomed-out fallback in either mode — AFTER the footprint
        outline, its 40-alpha fill and the solid nose tick. Drawn before
        them (the first cut) the axis was washed inside the footprint and
        overdrawn from the nose to +0.40 m by the tick, i.e. it did not
        visibly run through the COM; and behind the hull_px < 14 cut it
        vanished altogether, which is exactly the zoom a sight line is for.
        Only drawn with a KNOWN heading: an unknown yaw is drawn as 0 for
        the body's sake, but an axis pointing north by default would be a
        claim.
        """
        from .rov_shape import body_to_map, heading_line_body

        if ghost:
            return
        aft, fwd = heading_line_body()
        a = body_to_map(aft, p_ned, yaw)
        b = body_to_map(fwd, p_ned, yaw)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QtGui.QPen(_c(base, 190), 1, Qt.PenStyle.DotLine))
        painter.drawLine(self._px(*a), self._px(*b))

    def _draw_rov_optic(self, painter, p_ned, yaw: float, base: str,
                        ghost: bool) -> None:
        """A short ray out of the C3 along its optical axis.

        On a diffusion-policy run the operator's question about the vehicle
        is "is the object in frame", and nothing else on this plot answers
        it: the lens is 0.105 m ABOVE the COM and pitched 43.3 deg DOWN, so
        where it points is not guessable from a hull outline. Drawn as a bare
        RAY and not a frustum on purpose — a cone would be a claim about what
        is visible, and this panel does not know the range.
        """
        from .rov_shape import (CAM_RAY_M, CAM_T_FLU_M, body_to_map,
                                camera_axis_flu)

        if ghost:
            return
        ax = camera_axis_flu(self.cam_tilt_deg)
        tip = tuple(CAM_T_FLU_M[i] + CAM_RAY_M * ax[i] for i in range(3))
        a = body_to_map(CAM_T_FLU_M, p_ned, yaw)
        b = body_to_map(tip, p_ned, yaw)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QtGui.QPen(_c(base, 190), 1, Qt.PenStyle.DotLine))
        painter.drawLine(self._px(*a), self._px(*b))

    def _draw_rov_flat(self, painter, p_ned, yaw: float, model, base: str,
                       ghost: bool) -> None:
        """The top-down body: gripper (clipped to its overhang), thruster and
        camera silhouettes, and the TCP ring — everything but the hull and
        the heading line. The hull is left to the caller so its outline
        stays byte-identical to the one this panel has drawn since
        2026-08-14; the heading line is the caller's too, because it has to
        go on top of that outline (`_draw_rov_heading`). The optical-axis
        ray is NOT drawn from above — see the note at the end."""
        from .rov_shape import body_to_map, silhouette_xy, tcp_body_m

        z = float(p_ned[2])
        # CLIPPED AT THE HULL'S NOSE. Seen from above, the ~12 cm of tube
        # that runs back UNDER the hull is not visible — it is under 25 cm of
        # vehicle — and drawing it anyway put a bar across the hull that read
        # as a gripper mounted on the roof. Only the part that overhangs is
        # drawn, which is also exactly the part that can hit something.
        nose = float(self.rov_size_m[0]) / 2.0
        for part in model:
            if part["kind"] == "hull":
                continue
            xs = [v[0] for v in part["verts"]]
            ys = [v[1] for v in part["verts"]]
            if part["kind"] in ("thruster", "camera"):
                # THE TRUE SILHOUETTE, not a bounding box: four of the eight
                # are rotated 45 deg about z, and their box hid exactly the
                # thing worth drawing — which way they push. NOT clipped
                # either: the shrouds are what the footprint rectangle's own
                # edge is made of (its 0.5334 m width is 2 x (0.220 + the
                # shroud radius)), so from above they ARE the outline.
                quad = [(x, y, 0.0)
                        for x, y in silhouette_xy(part["verts"])]
                if len(quad) < 3:
                    continue
            else:
                # Axis-aligned along x, so the box IS the silhouette, and it
                # can be clipped to the overhang in one comparison.
                x0, x1 = max(min(xs), nose), max(xs)
                if x1 - x0 < 1e-4:
                    continue                  # entirely under the hull
                quad = ((x1, max(ys), 0.0), (x1, min(ys), 0.0),
                        (x0, min(ys), 0.0), (x0, max(ys), 0.0))
            path = QtGui.QPainterPath()
            first = True
            for v in quad:
                m = body_to_map(v, (p_ned[0], p_ned[1], 0.0), yaw)
                q = self._px(m[0], m[1], z)
                path.moveTo(q) if first else path.lineTo(q)
                first = False
            path.closeSubpath()
            painter.setPen(QtGui.QPen(_c(base, 150 if ghost else 220),
                                      1, Qt.PenStyle.DashLine if ghost
                                      else Qt.PenStyle.SolidLine))
            painter.setBrush(Qt.BrushStyle.NoBrush if ghost
                             else _c(base, 110))
            painter.drawPath(path)
        # FROM ABOVE the optical-axis ray and the heading line are the SAME
        # line: the ray's 43 deg pitch is invisible in a top-down projection,
        # and both run along body +x at y = 0. Drawing both put two dot
        # patterns on one segment (a denser, brighter stretch between the
        # lens and 0.63 m that read as a marker of its own), so the ray is
        # not drawn here at all — the heading line (paintEvent, after the
        # outline) covers its whole footprint and continues through the
        # body. The ray is still drawn in 3-D, where it is a different line.
        m = body_to_map(tcp_body_m(), (p_ned[0], p_ned[1], 0.0), yaw)
        self._draw_rov_tcp(painter, (m[0], m[1], z), base, ghost)

    def _draw_rov_tcp(self, painter, m, base: str, ghost: bool) -> None:
        """A small ring at the jaw-pair centre — the point the diffusion
        policy's plans are anchored and composed at. Deliberately a ring and
        not a filled dot: the vehicle's own centre dot is filled, and these
        two must not be confusable when they are 42 cm apart on screen."""
        if ghost:
            return
        q = self._px(m[0], m[1], m[2])
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QtGui.QPen(_c(base, 230), 1))
        painter.drawEllipse(q, 3.0, 3.0)

    # ------------------------------------------------------------- z gauge
    def _z_note(self) -> None:
        """Widen the gauge's window if anything DRAWN has left it. Monotone.

        The operator asked for a fixed band that grows rather than a hard one
        (2026-09-07), and growth is the only pose input the scale has. It is
        one-way and snapped OUTWARD to a 0.1 m lattice, so it settles after an
        excursion instead of breathing with the vehicle — which is the whole
        complaint this change answers.

        Only series that are actually PAINTED count. The old ruler took
        `p_dr` and the object with no visibility check while the drawing of
        both is gated (`_dr_visible`, `state != "cold"`), so a hidden DR ghost
        or a cold object could stretch a scale nothing was reading them on.
        """
        zs = []
        if self._act_fresh() and self.p_act is not None:
            zs.append(float(self.p_act[2]))
        if self.p_ref is not None:
            zs.append(float(self.p_ref[2]))
        if self.p_dr is not None and self._dr_visible():
            zs.append(float(self.p_dr[2]))
        o = self.obj
        if o is not None and o.p_map is not None and getattr(o, "state", "") != "cold":
            zs.append(float(o.p_map[2]))
        for pl in self.policy_plans:      # dicts, keyed "pts" — not tuples
            zs += [float(k[2]) for k in pl.get("pts", ())]
        if not zs:
            return
        lo, hi = self._z_window
        pad = 0.1
        want_lo = min(lo, math.floor(min(zs) / pad - 1e-9) * pad)
        want_hi = max(hi, math.ceil(max(zs) / pad + 1e-9) * pad)
        if want_lo < lo - 1e-9 or want_hi > hi + 1e-9:
            self._z_window = (want_lo, want_hi)

    def _z_gauge(self):
        """The depth gauge's geometry: (x, y_top, y_bot, ppm, lo, hi, step),
        or None when there is no room for one.

        READS: three_d, the widget size, and the latched window. NOTHING
        else — not the pose, not `pan`, not `zoom`, not the orbit angles. That
        is the operator's request (2026-09-07: "z축이 같이 안 움직이고
        고정되어있으면 좋겠어") expressed as a property a test can assert,
        rather than as a pixel diff.

        WHY THE GAUGE NO LONGER SHARES THE SCENE'S SCALE. The ruler this
        replaces was drawn THROUGH the projection, beside the vehicle, and
        that is what made it move — in three separate ways, all measured
        2026-09-07 at 700x500, az 0, el 55, zoom 1:
          1. the anchor was `p_act` stepped 118 px screen-left, so the whole
             ruler translated with the hull (1.0 m east + 0.5 m north moved it
             53 px across and 87 px up);
          2. because `u = (cos az sin el, sin az sin el, -cos el)`, a metre of
             HORIZONTAL travel slides a projected point tan(el) = 1.43x
             further up the screen than a metre of depth does — so most of the
             vertical wander was not depth at all;
          3. the span was re-centred every frame on the live z band, so the
             tick NUMBERS slid too (-0.10..+0.70 -> -0.10..+1.30 on a 0.6 m
             descent).
        Pinning only the anchor fixes (1) and (2); pinning only the span fixes
        (3). Both, together, mean the scale can no longer be a projection of
        the scene, because a projected scale is a function of the eye.

        SO IT IS AN INSTRUMENT, not an axis: a fixed window mapped onto a
        fixed pixel band, with the vehicle shown as a needle sliding on it.
        The price, stated plainly because it is real: a metre on the gauge is
        no longer a metre in the picture, so you cannot read a tick across to
        the hull. The needle is placed BY VALUE and prints its own number,
        which is exact where the old horizontal glance was only exact for a
        vehicle standing on the gauge's own world column.

        This is also what makes it usable on the REAL panel. The docked view
        is WIDE AND SHORT — measured 2026-09-07 by building MainWindow
        offscreen: 583x110 at the 1100x640 minimum window, and the height
        SATURATES at 338 px from 1920x1080 upward (window.py grid row 3 has
        `setRowStretch(3, 0)`), while the width goes on to 1896. A scene-
        linked scale at zoom 8 shows cos(55) * s = 746 px/m, i.e. only 0.38 m
        of window in a 282 px band — less than the 0.69 m the vehicle's own
        record spans. The fixed band gives 188 px/m there instead, at every
        zoom.
        """
        if not self.three_d:
            return None
        y_top = Z_GAUGE_TOP_PX
        y_bot = self.height() - Z_GAUGE_BOT_PX
        if y_bot - y_top < Z_GAUGE_MIN_H_PX:
            return None
        lo, hi = self._z_window
        if hi - lo < 1e-6:
            return None
        ppm = (y_bot - y_top) / (hi - lo)
        step = next((c for c in Z_GAUGE_STEPS if c * ppm >= Z_GAUGE_TICK_PX),
                    Z_GAUGE_STEPS[-1])
        # A CONSTANT label reserve, not the widest label actually drawn: the
        # column must not jump sideways when a tick gains a character.
        #
        # ALWAYS QFontMetricsF, never the painter's own QFontMetrics, even
        # though this only ever runs under a painter that set the same font.
        # The integer metric rounds ("-0.00" advances 33 px against 33.047),
        # so taking the painter's would make `_z_gauge()` answer differently
        # depending on who asked — and the test that pins the operator's
        # request asks without one.
        fm = QtGui.QFontMetricsF(QtGui.QFont("monospace", 8))
        x = (self.width() - Z_GAUGE_EDGE_PX
             - fm.horizontalAdvance("-0.00") - Z_GAUGE_GAP_PX)
        if x < 0.5 * self.width():
            return None
        return x, y_top, y_bot, ppm, lo, hi, step

    def _paint_z_axis(self, p) -> None:
        """The depth gauge: a fixed scale in the right margin, with a needle.

        RIGHT and not left because the left margin is taken — the corner
        readout starts at x = 8 and runs up to ~15 lines down the widget.
        Below the scale the bottom-right belongs to the trail-age legend
        (`height() - 14`), which `Z_GAUGE_BOT_PX` clears.

        NO BACKING STRIP. The gauge is painted late so the scene cannot cover
        it, and a strip that dimmed the scene behind it would re-open exactly
        the 2026-09-03 complaint that put the DP plan last in the paint order.
        Only a 1 px stem, short tick arms, outboard text and a 9 px caret ever
        land on the picture.
        """
        self._z_note()
        p.setFont(QtGui.QFont("monospace", 8))
        g = self._z_gauge()
        if g is None:
            return
        x, y_top, y_bot, ppm, lo, hi, step = g
        dec = max(0, -int(math.floor(math.log10(step))) + 1)

        def y_of(z: float) -> float:
            return y_top + (float(z) - lo) * ppm

        # DEEP IS DOWN, the same way round as the scene: map z is
        # down-positive and `_px`'s up-vector has u[2] = -cos(elev), so a
        # larger z draws lower there too. The gauge's MAGNITUDE is its own,
        # its DIRECTION is the picture's.
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QtGui.QPen(_c(theme.TEXT_DIM, 150), 1))
        p.drawLine(int(x), int(y_top), int(x), int(y_bot))
        # Ticks on an ABSOLUTE lattice by integer index — never by repeated
        # addition, which is why the horizontal grid above rounds before it
        # formats. The lattice is what lets the operator learn the scale:
        # -0.20 is always the same row until the window latches wider.
        fm = p.fontMetrics()
        k0 = int(math.ceil(lo / step - 1e-9))
        k1 = int(math.floor(hi / step + 1e-9))
        rows = []
        for k in range(k0, k1 + 1):
            gz = k * step
            y = y_of(gz)
            zero = abs(gz) < 1e-9
            arm = 7 if zero else 4
            p.setPen(_c(theme.TEXT_DIM if zero else theme.TEXT_FAINT,
                        200 if zero else 150))
            p.drawLine(int(x) - arm, int(y), int(x) + arm, int(y))
            lbl = f"{round(gz, dec) + 0.0:g}"
            rows.append((y, lbl))
        # THE NEEDLES, drawn before the labels so a label can stand aside for
        # the live one rather than the other way round.
        needles = []
        if self.p_ref is not None:
            needles.append((float(self.p_ref[2]), "#4d9dff", False, None))
        if self._act_fresh() and self.p_act is not None:
            zv = float(self.p_act[2])
            needles.append((zv, theme.OK, True, f"{zv:+.2f}"))
        texts = []
        for zv, hexc, filled, txt in needles:
            off = zv < lo - 1e-9 or zv > hi + 1e-9
            y = y_of(min(max(zv, lo), hi))
            p.setPen(QtGui.QPen(_c(hexc, 235), 1))
            p.setBrush(Qt.BrushStyle.NoBrush if (off or not filled)
                       else QtGui.QBrush(_c(hexc, 235)))
            tri = QtGui.QPolygonF([QtCore.QPointF(x - 1.0, y),
                                   QtCore.QPointF(x - 9.0, y - 4.0),
                                   QtCore.QPointF(x - 9.0, y + 4.0)])
            p.drawPolygon(tri)
            if txt:
                texts.append((y, txt, hexc))
        p.setBrush(Qt.BrushStyle.NoBrush)
        # Labels OUTBOARD — between the stem and the widget edge, the gutter
        # the reserve above was measured for. Nothing text-shaped is ever
        # drawn on the scene side of the stem.
        lx = int(x) + int(Z_GAUGE_GAP_PX)
        for y, txt, hexc in texts:
            p.setPen(_c(hexc, 235))
            p.drawText(lx, int(y) + 3, txt)
        # BRIGHTER THAN THE TICK ARMS, deliberately. With no backing strip
        # the numbers are read over whatever the scene put there — at zoom 8 a
        # single map tag can fill the gutter — and TEXT_FAINT at 150 vanished
        # on the grey of a tag. The arms stay faint; the numbers are the part
        # that has to survive a background.
        p.setPen(_c(theme.TEXT_DIM, 215))
        for y, lbl in rows:
            if any(abs(y - ty) < 9.0 for ty, _t, _c2 in texts):
                continue                      # the live number wins the row
            p.drawText(lx, int(y) + 3, lbl)
        # THE CAPS. Right-aligned to the widget edge so neither can run off a
        # narrow panel, and "deeper" belongs on the `hi` end because that is
        # the deep one — the first version of this ruler had them swapped.
        p.setPen(_c(theme.TEXT_DIM, 200))
        cap = "z [m]"
        p.drawText(int(self.width() - Z_GAUGE_EDGE_PX
                       - fm.horizontalAdvance(cap)), int(y_top) - 7, cap)
        deep = "↓ deeper"
        p.setPen(_c(theme.TEXT_FAINT, 190))
        p.drawText(int(self.width() - Z_GAUGE_EDGE_PX
                       - fm.horizontalAdvance(deep)), int(y_bot) + 13, deep)

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
            # THE BODY (2026-09-03): hull + Newton gripper + jaws, and in
            # 3-D an actual solid rather than a flat rectangle lying in the
            # water. `_draw_rov` returns False when the vehicle is only a few
            # pixels across, where the outline below says strictly more than
            # four smudged boxes would — so the flat path is not legacy, it
            # is the zoomed-out rendering.
            body = self._draw_rov(p, (self.p_act[0], self.p_act[1], zv),
                                  a, theme.OK)
            if not (body and self.three_d):
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
                    # THE HEADING LINE, last of the flat marker so the
                    # fill and the tick cannot cover it (docstring of
                    # _draw_rov_heading). On this path in both modes and at
                    # every zoom — the zoomed-out fallback is where a sight
                    # line earns its keep.
                    self._draw_rov_heading(
                        p, (self.p_act[0], self.p_act[1], zv), a, theme.OK,
                        False)
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
                yd = float(self.yaw_dr or self.yaw_ned or 0.0)
                # The SAME body as the truth marker, outline only — the
                # operator is judging the distance between the two, and two
                # differently drawn vehicles would make that judgement about
                # the drawing (the rule `_hull`'s docstring already states).
                ghost = self._draw_rov(p, (self.p_dr[0], self.p_dr[1], zd),
                                       yd, theme.WARN, ghost=True)
                if not (ghost and self.three_d):
                    hull = self._hull(self.p_dr, yd)
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

        # THE DIFFUSION POLICY'S PROPOSALS — what was ASKED FOR, never what
        # happened, and painted LAST so nothing can cover them.
        #
        # They used to go on before the trails and therefore before the
        # HULL, which on this vehicle is a 0.43 x 0.53 m body drawn with
        # filled faces — and the plan is composed at the jaw, i.e. right
        # where the hull is. The operator's report was exactly that: the
        # reference is hidden behind the live ROV (2026-09-03). Drawing
        # order is the fix; colour and width alone could not have been.
        # The cost is that the plan now covers the marker it overlaps
        # rather than the reverse, which is the right way round: the hull's
        # position is also given by the centre dot, the depth stick and the
        # readout, and the plan is given by nothing else.
        #
        # FLUORESCENT YELLOW, DASHED (operator, 2026-09-03, after the first
        # pool run). Every other series on this plot is solid: the hull is
        # green, the reference blue, the DR amber, the trail a viridis ramp.
        # Dashes are therefore unique to the policy, which matters more than
        # the hue does — the trail ramp ENDS in yellow-green at "now", so
        # colour alone would collide exactly where the two overlap. Width and
        # dashing carry the distinction; the hue is what makes it findable.
        #
        # NEWEST vs PREVIOUS, not a fade over 45 s: at two deep, "which of
        # these is the current one" is the whole reading, so the newest is
        # full-strength and 3 px and the older one is dimmer and thinner.
        # A dot marks knot 0 (the observation time) so the direction reads.
        self._prune_policy_plans()
        for i, pl in enumerate(self.policy_plans):
            newest = (i == len(self.policy_plans) - 1)
            pen = QtGui.QPen(_c(POLICY_PLAN_COLOR, 255 if newest else 130),
                             POLICY_PLAN_W if newest else POLICY_PLAN_W_PREV,
                             Qt.PenStyle.DashLine)
            poly(pl["pts"], pen)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(self._px(*pl["pts"][0]), 3 if newest else 2,
                          3 if newest else 2)

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
            # RED = tracking error, and it may only mean that. Under POLICY
            # OBSERVE nothing is tracking: the line still connects the hull to
            # the DP's current stage-0 target (which is exactly the "where
            # does it want me" the operator is here to read) but in the dim
            # neutral colour, so it cannot be read as a controller failing.
            p.setPen(QtGui.QPen(
                _c(theme.TEXT_DIM if self.observe else theme.FAIL, 130), 1,
                Qt.PenStyle.DashLine))
            p.drawLine(self._px(*self.p_act), self._px(*self.p_ref))

        # THE DEPTH GAUGE — 3-D only, because top-down has no z to show.
        # Operator request 2026-09-03: the tilted view had depth in it (every
        # marker sits at its own z, with a dotted stick to the tag plane) but
        # nothing to READ that depth against, so "how deep is it" was a
        # judgement about pixels.
        #
        # PAINTED HERE, after the scene and the plans, because 2026-09-07 made
        # it an instrument in the right margin rather than a ruler standing in
        # the scene: an orbited, zoomed view wanders into that margin, and a
        # scale that something else drew over is not a scale. The hull has
        # three other representations (centre dot, depth stick, readout); the
        # gauge has none. It draws no filled background, so nothing it does
        # can hide the plan the 2026-09-03 paint order exists to protect.
        if self.three_d:
            self._paint_z_axis(p)
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
        # With --fstereo there are two of them, and each is LABELLED. The
        # comparison is the experiment, so an unlabelled ratio would be worse
        # than no ratio: it would be attributed to whichever depth the reader
        # happened to have in mind.
        lines.extend(self._readout_depth_lines())
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
        # LIVE DEPTH, always — operator request 2026-09-03. Every other z on
        # this panel is conditional: the chip's `_hold_z_text` needs a
        # REFERENCE, and the 3-D ruler needs 3-D. But "how deep am I right
        # now" is the question a pilot flying by hand asks continuously, so
        # it gets its own line whenever there is a fresh fix, in the MAP
        # frame like everything else here. `-- m` rather than a stale number
        # when the marker has aged out: this panel's rule is that it may only
        # show what it currently knows.
        if self._act_fresh() and self.p_act is not None:
            zline = f"z  {self.p_act[2]:+6.3f} m"
            if self.p_ref is not None:
                zline += f"   ref {self.p_ref[2]:+6.3f}"
            lines.append(zline)
        elif self.p_act is not None:
            lines.append("z     -- m")

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
        if self.policy_plans:
            # The one series whose colour is deliberately close to the trail
            # ramp's "now" end, so it is the one that most needs naming.
            swatches.append(("DP", QtGui.QPen(_c(POLICY_PLAN_COLOR, 230), 2,
                                              Qt.PenStyle.DashLine)))
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


class ElidingLabel(QtWidgets.QLabel):
    """A QLabel that paints its text elided ("…") instead of demanding the
    width of its longest sentence.

    A plain QLabel's MINIMUM width is its full text, so the mission label —
    whose text is a sentence that grows with the mission (and, since
    2026-09-11, with the LOW None suffix: "… · LOW None: START will be
    refused (teleop runs only Diffusion Policy / Replay)") — set the whole
    panel's minimum width, and through the grid the window's, past the
    1366 px promise of test_control
    test_mpc_panel_lives_in_the_grid_and_fits_the_operator_screen (the test
    FAILED with the suffix in a plain QLabel on 2026-09-11; the width it
    reported was a transient failure message that no artifact stores, and the
    test asserts only the bound, so no figure is quoted here — review
    2026-09-11). Here the minimum is 0 and the layout's stretch gives it
    the leftover; the full sentence stays in `text()` (tests and the tooltip
    read it) and only the PAINT elides. Unlike indicators.ElidedLabel it keeps
    QLabel's text/tooltip/stylesheet behaviour — the sheet's colour and font
    reach the paint through the polished palette and font.

    Also the checkpoint NAME and STATE on the HIGH row (2026-09-11, review
    M1): each prefers its text and yields under a deficit, so neither a long
    checkpoint name nor a 48-char refusal can drive the panel's minimum."""

    def setText(self, text: str) -> None:          # noqa: N802 - Qt naming
        super().setText(text)
        self.setToolTip(text)

    def minimumSizeHint(self):                      # noqa: N802 - Qt naming
        h = super().minimumSizeHint()
        return QtCore.QSize(0, h.height())

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setFont(self.font())
        p.setPen(self.palette().color(self.foregroundRole()))
        fm = QtGui.QFontMetrics(self.font())
        r = self.contentsRect()
        p.drawText(r, int(self.alignment()) | int(Qt.AlignmentFlag.AlignVCenter),
                   fm.elidedText(self.text(), Qt.TextElideMode.ElideRight,
                                 max(0, r.width())))


class TrajectoryWindow(QtWidgets.QWidget):
    """Controls + the view, a GRID CELL of the main window (window.py ~450:
    `grid.addWidget(self._traj_panel, 3, 2, 1, 2)` — the bottom row's columns
    2-3; it was a QDockWidget before 2026-08-14). Emits requests; never flips
    its own state — every label reflects what MpcStatus / PolicyStatus says
    actually happened, the same honesty rule the ARM button follows.

    Two combos, two levels (2026-09-11, operator request):
      HIGH  the mission SHAPE — what generates the reference (SHAPE_LABELS);
      LOW   the FOLLOWER — who commands the thrusters (MODE_LABELS). `None`
            is teleop: the station commands nothing and the pilot flies; a
            Diffusion Policy mission keeps inferring and drawing.
    A Diffusion Policy mission also shows the CHECKPOINT widgets on the HIGH
    row (the numeric fields are hidden for that shape, so no row is added):
    the file is chosen in a dialog owned by the window (CkptDialog), the
    request goes to the
    PolicyWorker over `policy_ckpt_requested`, and the CONFIRMED name comes
    back only through `set_policy_status` (the worker may refuse the swap).

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
    # The `…` button on the ckpt row: "open the picker". The dialog belongs to
    # the WINDOW (MainWindow._pick_policy_ckpt), not to this panel — it has to
    # hand the keyboard back from the vehicle first and wire the dialog's Esc
    # to E-STOP, which only the window can do.
    policy_ckpt_pick_requested = Signal()
    # The chosen checkpoint PATH -> bus.cmd_policy_ckpt -> PolicyWorker.set_ckpt
    # (which loads it on the GPU, or refuses and says why in
    # PolicyStatus.ckpt_note). Replaces --policy-ckpt (2026-09-11).
    policy_ckpt_requested = Signal(str)

    _BTN_CSS = "QPushButton{font-size:11px; padding:3px 8px;}"
    # The ckpt row's button: same face, one line tall — the row sits between
    # the mission row and the LOW row and the window's height budget is ~6 px
    # (test_control: test_mpc_panel_lives_in_the_grid_and_fits_the_operator_screen).
    _CKPT_BTN_CSS = "QPushButton{font-size:11px; padding:1px 6px;}"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MPC — reference vs actual (NED tag world)")
        self.view = TrajectoryView()

        # A label per level, dim, so the two combos read as the two questions
        # they answer (operator, 2026-09-11: "high level ... / low level ...").
        self.lbl_high = QtWidgets.QLabel("HIGH")
        self.lbl_high.setStyleSheet(f"color:{theme.TEXT_DIM}; font-size:11px;")
        self.lbl_high.setToolTip(
            "HIGH level — the mission: what generates the reference")
        self.lbl_low = QtWidgets.QLabel("LOW")
        self.lbl_low.setStyleSheet(f"color:{theme.TEXT_DIM}; font-size:11px;")
        self.lbl_low.setToolTip(
            "LOW level — the follower: who commands the thrusters.\n"
            "None = nobody: YOU fly on the joystick (teleop); a Diffusion "
            "Policy mission still infers and draws — after START only (the "
            "PolicyWorker infers while a policy mission is ACTIVE).")
        self.mode_box = QtWidgets.QComboBox()
        # Items are (LABEL, KEY): the text is for the operator, the KEY is what
        # the worker speaks (MODE_LABELS). `currentData()` is the key — never
        # parse the text.
        # none = TELEOP (2026-09-11): nobody commands; see MODE_LABELS.
        # mpcc = contouring (theta is a solver decision on a C1 filleted
        # curve); dobmpcc adds the EAOB disturbance estimate to it, exactly
        # as dobmpc does to mpc.
        # *_tuned = the SAME tracking solver as dobmpc/mpc with its position
        # weight rotated into the path frame (control/path_cost.py): cheap on
        # along-track lag, expensive on cross-track, which is the knob against
        # corner cutting. No extra build — the suffix is a runtime cost_set.
        for key, label in MODE_LABELS.items():
            self.mode_box.addItem(label, key)
        self.mode_box.setToolTip(
            "None: TELEOP — the station commands nothing, you fly on the "
            "joystick; a Diffusion Policy mission still infers and draws its "
            "plans (START refused for every other shape but Replay). Frozen "
            "while engaged or while a controller CSV is open.\n"
            "MPCC/DOBMPCC: contouring — theta is a solver state\n"
            "MPC/DOBMPC: tracking NMPC, isotropic world-frame Q\n"
            "*_Tuned: the same tracking NMPC with Q split into along-track "
            "(cheap) and cross-track (expensive) — suppresses corner "
            "cutting; weights in hw_mpc.yaml mpc_tuned:\n"
            "PID: the pole-placed baseline")
        self.mode_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.mode_box.currentIndexChanged.connect(self._mode_changed)


        # ---- the mission: shape, where it starts, how far, how fast --------
        # These are what the operator changes between runs, so they are
        # on the panel rather than in the YAML. They are sent as a scenario
        # OVERRIDE (cmd_mpc_scenario) and merged over hw_mpc.yaml's square
        # block, so laps/depth/ramp still come from the file.
        self.shape_box = QtWidgets.QComboBox()
        # (LABEL, KEY) items in SHAPES order — the KEY is what `_emit_scenario`
        # sends and what `set_shape`/`current_shape` speak (SHAPE_LABELS).
        for key in SHAPES:
            self.shape_box.addItem(SHAPE_LABELS[key], key)
        self.shape_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.shape_box.setToolTip(
            "HIGH level — the mission. What each one does:\n"
            "Station: sit on the tag and hold, heading included — fly THIS "
            "first\n"
            "Line: out and back along +y from the tag (1 lap = there AND back)\n"
            "Square: a rectangle with the tag as its first corner\n"
            "Circle: the tag is the BOTTOM of the circle, not its centre — "
            "the box sets the radius\n"
            "Follow: hold the relative pose you have RIGHT NOW on the object "
            "you clicked (needs --pose; the speed box is the setpoint's "
            "speed cap). Arming it never moves the vehicle.\n"
            "Replay: re-fly a recorded handheld demo starting from the "
            "CURRENT pose (session + limits in hw_mpc.yaml replay:, or "
            "--replay-session). Arming it never moves the vehicle first.\n"
            "Diffusion Policy: let the diffusion policy fly from the CURRENT "
            "pose (needs --policy and --fstereo; the checkpoint is chosen on "
            "the ckpt row; limits in hw_mpc.yaml policy:). Arming it never "
            "moves the vehicle first; the policy does, one 1 s chunk at a "
            "time. With LOW = None it only infers and draws. "
            "HARDWARE-UNVERIFIED — a pipeline smoke, not a grasp test.")
        self.tag_box = QtWidgets.QSpinBox()
        self.tag_box.setRange(0, 999)
        self.tag_box.setToolTip("the path starts directly above THIS tag id — "
                                "click and type, or use the arrows")
        # STATION only (2026-09-11, operator: "station 에 있을 위치 tag id: 47,
        # heading 방향 tag id: 97"): hold the origin tag while FACING this
        # one. The worker turns the pair into a map bearing, so the same two
        # ids mean the same physical heading every run. 0 = off.
        self.face_box = QtWidgets.QSpinBox()
        self.face_box.setRange(0, 999)
        self.face_box.setToolTip("STATION only: hold the origin tag while "
                                 "FACING this tag id (0 = keep the configured "
                                 "heading)")
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
        for w in (self.shape_box, self.tag_box, self.face_box, self.len_box,
                  self.leny_box, self.spd_box):
            w.setFont(small)
        # (_shape_changed is connected at the END of __init__, next to the
        # call that seeds it: it now touches btn_start, which is built below,
        # so connecting here would leave a window where a setCurrentText
        # inserted between the two raises AttributeError.)
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
        for w in (self.tag_box, self.face_box, self.len_box, self.leny_box,
                  self.spd_box):
            w.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
            w.setKeyboardTracking(False)
            w.installEventFilter(self)
            w.editingFinished.connect(w.clearFocus)
            w.valueChanged.connect(self._emit_scenario)
        self._num_fields = (self.tag_box, self.face_box, self.len_box,
                            self.leny_box, self.spd_box)

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

        # ---- the checkpoint widgets (Diffusion Policy only) ----------------
        # 2026-09-11, operator request: the model is picked HERE, not with
        # --policy-ckpt. `ckpt_name` is what the PolicyWorker HOLDS (seeded
        # from hw_mpc.yaml policy.ckpt at startup by the window, then only
        # ever set from PolicyStatus.ckpt); a pick shows as "→ name" until the
        # worker confirms or refuses it. `ckpt_state` is loading…/READY/ERROR,
        # or the worker's refusal in red for a few seconds.
        # They live ON THE HIGH ROW (row0), right after the shape combo, and
        # show only for shape policy — for which every numeric field of row0
        # is hidden, so there is room — and a policy-seeded panel therefore
        # adds NO height (review 2026-09-11, M1: a separate row cost 22 px of
        # panel minimum height, 218 -> 240 px, measured offscreen in the
        # review's scratch script against the window's ~6 px height budget).
        self.lbl_ckpt = QtWidgets.QLabel("ckpt")
        self.lbl_ckpt.setStyleSheet(f"color:{theme.TEXT_DIM}; font-size:11px;")
        # Both PREFER their text and YIELD under a deficit (ElidingLabel:
        # minimum width 0, the paint elides, the full text is the tooltip):
        # the basename is already cut to CKPT_NAME_CHARS and the state to 48
        # chars, but a plain QLabel would still DEMAND that width and, through
        # the grid, the window's minimum (which resizes a maximised window).
        self.ckpt_name = ElidingLabel("")
        self.ckpt_name.setStyleSheet(f"color:{theme.TEXT_DIM}; font-size:11px;")
        self.ckpt_state = ElidingLabel("")
        self.ckpt_state.setStyleSheet(f"color:{theme.TEXT_DIM}; font-size:11px;")
        self.btn_ckpt = QtWidgets.QPushButton("…")
        self.btn_ckpt.setStyleSheet(self._CKPT_BTN_CSS)
        self.btn_ckpt.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_ckpt.setToolTip(
            "choose the diffusion-policy checkpoint (*.ckpt); loads on the "
            "GPU when chosen (refused while a policy mission is armed — STOP "
            "TRAJ / DISENG first — and disabled while a REC controller CSV is "
            "open: stop the recording first, its meta names one checkpoint).\n"
            "While the dialog is up the KEYBOARD is not flying the vehicle "
            "(Esc closes it AND is E-STOP); the joystick still flies.")
        self.btn_ckpt.clicked.connect(self.policy_ckpt_pick_requested)
        # Shown/hidden together by `_shape_changed` (tests read it too).
        self.ckpt_widgets = (self.lbl_ckpt, self.ckpt_name, self.ckpt_state,
                             self.btn_ckpt)
        self._ckpt_path = ""            # the seeded / confirmed path
        self._ckpt_note = ""            # the worker's last refusal, and when
        self._ckpt_note_t = 0.0
        self._ckpt_note_seen = ""       # last ckpt_note VALUE seen (incl. "")

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
            "(hw_mpc.yaml square.depth_ned pins it to a number instead).\n"
            "OBSERVE · DP NOT COMMANDING = LOW level None (teleop): the run "
            "is engaged but nothing leaves the station — you are flying.")
        self.chip.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 11px;")

        row0 = QtWidgets.QHBoxLayout()
        row0.setSpacing(4)
        row0.addWidget(self.lbl_high)
        row0.addWidget(self.shape_box)
        # The checkpoint widgets, right after the HIGH combo (visible only
        # for shape policy — see their construction above). No stretch: each
        # takes its text when there is room; the mission label yields first.
        for w in self.ckpt_widgets:
            row0.addWidget(w)
        # Kept on self so `_shape_changed` can hide it WITH its box. A label
        # left behind when its field goes away reads as a field that failed to
        # draw, which is worse than either state.
        self.lbl_tag = QtWidgets.QLabel("from tag")
        self.lbl_tag.setStyleSheet(f"color:{theme.TEXT_DIM}; font-size:11px;")
        row0.addWidget(self.lbl_tag)
        row0.addWidget(self.tag_box)
        self.lbl_face = QtWidgets.QLabel("facing tag")
        self.lbl_face.setStyleSheet(f"color:{theme.TEXT_DIM}; font-size:11px;")
        row0.addWidget(self.lbl_face)
        row0.addWidget(self.face_box)
        row0.addWidget(self.len_box)
        row0.addWidget(self.lbl_y)
        row0.addWidget(self.leny_box)
        row0.addWidget(self.spd_box)
        # Eliding, so a long mission sentence never sets the window's minimum
        # width (see ElidingLabel); the full text is its tooltip. Its width
        # is IGNORED and it takes the row's LEFTOVER (stretch 1): since the
        # checkpoint name and state share this row (review 2026-09-11, M1)
        # the explanation must be what gets squeezed when the row is short,
        # never the name of the model the worker holds.
        self.mission_lbl = ElidingLabel("")
        self.mission_lbl.setStyleSheet(
            f"color:{theme.TEXT_FAINT}; font-size:11px;")
        self.mission_lbl.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                       QtWidgets.QSizePolicy.Policy.Preferred)
        row0.addWidget(self.mission_lbl, 1)

        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(4)
        row1.addWidget(self.lbl_low)
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
        self._mission_base = ""        # mission_lbl before the LOW suffix
        self.shape_box.currentIndexChanged.connect(self._shape_index_changed)
        self._shape_changed(self.current_shape())

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
        (2026-08-17, data/20260817/0817_103431 — every meta says
        dobmpc). Blocked, because the panel must REFLECT state here, not
        command it: emitting would make the label true by changing the
        vehicle, which is the wrong direction for the honesty rule."""
        m = str(mode).lower()
        i = self.mode_box.findData(m)           # by KEY — the text is a label
        if i < 0:
            return
        self.mode_box.blockSignals(True)
        self.mode_box.setCurrentIndex(i)
        self.mode_box.blockSignals(False)
        # The mission label depends on the LOW level (LOW None changes what
        # START does), and the blocked combo fires nothing — so re-word it
        # here. Without this the window's seeding order (set_mission_defaults
        # with the bare panel on `none`, THEN set_mode_default from the file)
        # left the "LOW None: START will be refused" suffix on a panel whose
        # LOW combo said DOBMPC. Display only: nothing is emitted.
        self._refresh_mission_lbl()

    def current_mode(self) -> str:
        """The LOW level KEY the combo shows (MODE_LABELS) — never the text."""
        return str(self.mode_box.currentData() or "")

    def _mode_changed(self, _idx: int) -> None:
        """The operator picked a LOW level: send the KEY. The mission label
        depends on it too (LOW None changes what START will do), so it is
        re-worded here WITHOUT re-sending the scenario."""
        self._refresh_mission_lbl()
        key = self.current_mode()
        if key:
            self.mode_requested.emit(key)

    def current_shape(self) -> str:
        """The HIGH level KEY the combo shows (geometry.SHAPES) — never the
        text; `SHAPE_LABELS` is display only."""
        return str(self.shape_box.currentData() or "")

    def set_shape(self, key: str) -> None:
        """Select a shape BY KEY (a no-op for an unknown one). Fires the same
        path a click does unless the caller has blocked the combo's signals
        (set_mission_defaults)."""
        i = self.shape_box.findData(str(key).lower())
        if i >= 0:
            self.shape_box.setCurrentIndex(i)

    def _shape_index_changed(self, _idx: int) -> None:
        self._shape_changed(self.current_shape())

    def set_mission_defaults(self, sq: dict) -> None:
        """Seed the controls from hw_mpc.yaml so the panel and the file agree
        at startup (the panel then owns these fields)."""
        boxes = (self.shape_box, self.tag_box, self.face_box, self.len_box,
                 self.leny_box, self.spd_box)
        for w in boxes:
            w.blockSignals(True)
        shape = str(sq.get("shape", "square")).lower()
        self.set_shape(shape if shape in SHAPES else "square")
        tag = sq.get("origin_tag")
        self.tag_box.setValue(0 if tag in (None, "") else int(tag))
        face = sq.get("heading_tag")
        self.face_box.setValue(0 if face in (None, "") else int(face))
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
        self._shape_changed(self.current_shape())

    def _shape_changed(self, shape: str) -> None:
        station = (shape == "station")
        follow = (shape == "follow")
        # REPLAY takes NO panel numbers at all: the mission is a recorded
        # demo (hw_mpc.yaml replay.session / --replay-session), its speed cap
        # is replay.v_max_m_s, and it anchors at the vehicle's current pose —
        # so every field this shape does not use vanishes (the follow rule:
        # one greyed-out survivor reads as "this one still matters somehow").
        # POLICY hides the same fields for the same reason: the mission is
        # the live diffusion policy (hw_mpc.yaml policy:, --policy), its speed
        # cap is policy.v_max_m_s, and it anchors at the current pose.
        replay = shape in ("replay", "policy")
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
        # "facing tag" is a STATION idea only: a path's heading is its
        # direction of travel (or heading_follow), so the box vanishes with
        # the shape rather than greying out (the follow rule above).
        self.face_box.setVisible(station)
        self.lbl_face.setVisible(station)
        # station does not move; replay's speed is replay.v_max_m_s (config)
        self.spd_box.setVisible(not station and not replay)
        # The one distance box means a length, a side, or a RADIUS depending
        # on the shape, and 0.50 m of side and 0.50 m of radius are missions
        # of very different size. The prefix says which is on screen, so the
        # number is never ambiguous at a glance; mission_lbl spells it out.
        self.len_box.setPrefix("r " if shape == "circle" else "")
        # START is INSTANT for `policy`, hold-to-confirm for every other shape
        # (2026-09-02, operator request). The hold exists so a stray click
        # cannot launch a geometric mission that immediately flies the vehicle
        # somewhere; a policy mission does not have that shape of risk at the
        # button — arming it never moves the vehicle (the first plan has to
        # arrive, pass the filter and be installed first), and the operator is
        # watching a live scene where a second of hold is a second of the
        # object moving. Every ABORT (STOP TRAJ / DISENG / E-STOP / Esc /
        # DISARM) was already a single instant click, so this makes start and
        # stop symmetric for this one shape rather than weakening a gate.
        policy = (shape == "policy")
        # The checkpoint widgets exist only for the shape that has a
        # checkpoint. They share row0 with the numeric fields, ALL of which
        # are hidden for policy, so the panel's height is the same for every
        # shape (review 2026-09-11, M1 — no extra row). All four go together:
        # a stray "ckpt" label beside a Square mission would read as a field
        # that failed to draw (the follow rule above).
        for w in self.ckpt_widgets:
            w.setVisible(policy)
        self.btn_start.set_hold_ms(0 if policy else 1000)
        self.btn_start.setToolTip(
            "click: engage, warm up, run the diffusion policy — recording "
            "(CSV + plans.jsonl) starts at engage. INSTANT (no hold)."
            if policy else
            "hold: engage, warm up, fly the mission — recording (CSV) "
            "starts at engage")
        self._emit_scenario()

    def _emit_scenario(self) -> None:
        shape = self.current_shape()
        tag = int(self.tag_box.value())
        # heading_tag is written for EVERY shape (None unless station sets
        # it) so the panel's scenario always overrides a value left in
        # hw_mpc.yaml — a hidden box must not steer a path (safety review
        # 2026-09-11).
        d = {"shape": shape, "origin_tag": (None if tag == 0 else tag),
             "heading_tag": None}
        text = ""
        if shape == "follow":
            # The origin tag means nothing here, so it is not sent: a follow
            # anchored to a tag id would be a promise the mission cannot keep.
            d.pop("origin_tag", None)
            d["speed"] = float(self.spd_box.value())
            text = (f"hold this offset on the clicked object, setpoint <= "
                    f"{self.spd_box.value():.2f} m/s")
        elif shape == "replay":
            # No tag, no numbers: the demo anchors at the CURRENT pose and
            # everything else lives in hw_mpc.yaml's replay: block.
            d.pop("origin_tag", None)
            text = ("re-fly the recorded demo from HERE "
                    "(hw_mpc.yaml replay.session / --replay-session)")
        elif shape == "policy":
            # Same rule: the first plan is anchored at the pose the vehicle
            # has at START; every limit is in the policy: block. The wording
            # depends on the LOW level (teleop or a follower), so it is
            # composed in `_mission_text`.
            d.pop("origin_tag", None)
        elif shape == "station":
            face = int(self.face_box.value())
            d["heading_tag"] = None if face == 0 else face
            text = (("hold on tag %d" % tag if tag else "hold here")
                    + (f", facing tag {face}" if face else ", facing +y"))
        elif shape == "line":
            d["length"] = float(self.len_box.value())
            d["speed"] = float(self.spd_box.value())
            text = (f"out {self.len_box.value():.2f} m +y and back @ "
                    f"{self.spd_box.value():.2f} m/s"
                    + (f", from tag {tag}" if tag else ", from here"))
        elif shape == "square":
            d["size"] = float(self.len_box.value())
            d["size_y"] = float(self.leny_box.value())
            d["speed"] = float(self.spd_box.value())
            text = (f"{self.len_box.value():.2f} m +x × "
                    f"{self.leny_box.value():.2f} m +y @ "
                    f"{self.spd_box.value():.2f} m/s"
                    + (f", SW corner at tag {tag}" if tag
                       else ", SW corner here"))
        elif shape == "circle":
            d["radius"] = float(self.len_box.value())
            d["speed"] = float(self.spd_box.value())
            # Spelled out as "bottom of the circle", not "on the circle": the
            # operator's whole point (2026-08-17) was that the tag is NOT the
            # centre, and a label that only said "at tag N" would leave the
            # reader to guess which of the two it meant.
            text = (f"circle r {self.len_box.value():.2f} m @ "
                    f"{self.spd_box.value():.2f} m/s"
                    + (f", tag {tag} at its bottom" if tag
                       else ", bottom of it here"))
        self._mission_base = text
        self.mission_lbl.setText(self._mission_text(shape, text))
        self.scenario_requested.emit(d)

    def _mission_text(self, shape: str, base: str) -> str:
        """The mission label with the LOW level folded in.

        LOW None changes what START will DO, so the label has to say it
        before the button is pressed: a Diffusion Policy START then only
        infers and draws (the pilot flies), and any geometric shape / follow
        is refused by the worker (`set_traj`: teleop runs only Diffusion
        Policy / Replay). Wording per the 2026-09-11 design."""
        low_none = (self.current_mode() == "none")
        if shape == "policy":
            if low_none:
                return ("LOW None (teleop): press START — the DP infers + "
                        "draws, YOU fly; STOP TRAJ / DISENG to change LOW or "
                        "the ckpt")
            return ("run the diffusion policy from HERE (ckpt from the "
                    "picker; limits in hw_mpc.yaml policy:)")
        if low_none and shape != "replay":
            return base + (" · LOW None: START will be refused (teleop runs "
                           "only Diffusion Policy / Replay)")
        return base

    def _refresh_mission_lbl(self) -> None:
        """Re-word the mission label for the current LOW level, without
        re-sending the scenario (nothing about the mission changed)."""
        self.mission_lbl.setText(
            self._mission_text(self.current_shape(), self._mission_base))

    # ------------------------------------------------------- the checkpoint
    def current_ckpt(self) -> str:
        """The checkpoint path the row is SHOWING as held: the launch seed
        (hw_mpc.yaml policy.ckpt) until the PolicyWorker reports its own. A
        pending "→ name" request is NOT this — the honesty rule."""
        return self._ckpt_path

    def _show_ckpt_name(self, path: str, *, requested: bool) -> None:
        name = elide_ckpt_name(os.path.basename(str(path).rstrip("/")) or str(path))
        if requested:
            # A REQUEST, not a fact: the arrow says "asked for", and the held
            # path is untouched until the worker's status replaces it.
            self.ckpt_name.setText(f"→ {name}")
            self.ckpt_name.setToolTip(f"requested: {path}")
        else:
            self._ckpt_path = str(path)
            self.ckpt_name.setText(name)
            self.ckpt_name.setToolTip(str(path))

    def _set_ckpt_state(self, text: str, colour: str) -> None:
        # One line, always: a long error / refusal is elided here and kept
        # whole in the tooltip (and it is in the mission log in full).
        full = str(text)
        shown = full if len(full) <= 48 else full[:47] + "…"
        self.ckpt_state.setText(shown)
        self.ckpt_state.setToolTip(full)
        self.ckpt_state.setStyleSheet(f"color:{colour}; font-size:11px;")

    def set_policy_ckpt_default(self, path: str) -> None:
        """Seed the name from hw_mpc.yaml policy.ckpt — what the PolicyWorker
        loads at startup (window.py, beside set_mission_defaults). Replaced by
        the worker's own PolicyStatus.ckpt as soon as one arrives."""
        if str(path or "").strip():
            self._show_ckpt_name(str(path).strip(), requested=False)

    def _apply_ckpt_choice(self, path: str) -> None:
        """The dialog's result (a plain method so tests need no dialog):
        show it as REQUESTED and ask the PolicyWorker. Empty = cancelled."""
        path = str(path or "").strip()
        if not path:
            return
        self._show_ckpt_name(path, requested=True)
        # A NEW request: whatever the worker answers is news, even the
        # sentence it already gave once. The worker keeps `ckpt_note` until
        # its next swap / same-path pick and `_refuse_ckpt` overwrites it
        # with the identical text, so a second identical refusal ("still
        # loading X; wait…", "not a file: <same path>", "the controller is
        # silent…") is NO value change: the 6 s clock would not restart and
        # the row would show the held name + READY — the silent snap-back the
        # note exists to prevent (review 2026-09-11). So forget the last
        # value here, and drop the old note too: it answered the PREVIOUS
        # pick, and must not stay red over this one's "loading…".
        self._ckpt_note_seen = ""
        self._ckpt_note = ""
        self.policy_ckpt_requested.emit(path)

    def set_policy_status(self, st: PolicyStatus) -> None:
        """Reflect the PolicyWorker's truth on the ckpt row. The CONFIRMED name
        comes ONLY from here (`_apply_ckpt_choice` shows a `→ name` request
        until the worker says which file it holds).

        `st.ckpt` is the path the worker HOLDS (loading, ready or failed). It
        replaces the name only when NON-EMPTY: a crash status carries "" and
        the last name has to stay beside the ERROR, or a blank row reads as
        "no checkpoint" (review 2026-09-11). `st.ckpt_note` is why the last
        pick was REFUSED (mission armed / still loading / not a file /
        FS-parity / worker not running): shown in FAIL colour for 6 s — the
        `_ref_msg/_ref_t` pattern — then the row falls back to
        loading…/READY/ERROR. A refusal that only reached the log was
        invisible next to a name that silently snapped back."""
        ckpt = str(getattr(st, "ckpt", "") or "")
        if ckpt:
            self._show_ckpt_name(ckpt, requested=False)
        note = str(getattr(st, "ckpt_note", "") or "")
        if note != self._ckpt_note_seen:
            # New VALUE — or the first note after a new request, since
            # `_apply_ckpt_choice` forgets the last value (the worker keeps
            # the note until its next swap and repeats the identical sentence
            # for an identical refusal; review 2026-09-11). A stale 1 Hz idle
            # status still carrying the OLD note can restart the clock up to
            # ~1 s before the real answer; that answer then either keeps it
            # (same refusal) or withdraws it (below) — never a wrong colour
            # for longer than that.
            self._ckpt_note_seen = note
            if note:
                self._ckpt_note, self._ckpt_note_t = note, time.monotonic()
            else:
                # The worker WITHDREW its note (it clears it on a swap or a
                # same-path pick): the refusal is stale, and must not stay
                # red over the "loading…" that follows (review 2026-09-11).
                self._ckpt_note = ""
        if self._ckpt_note and time.monotonic() - self._ckpt_note_t < 6.0:
            self._set_ckpt_state(self._ckpt_note, theme.FAIL)
        elif getattr(st, "loading", False):
            self._set_ckpt_state("loading…", theme.WARN)
        elif getattr(st, "error", ""):
            self._set_ckpt_state(f"ERROR: {st.error}", theme.FAIL)
        elif getattr(st, "ready", False):
            self._set_ckpt_state("READY", theme.OK)
        else:
            self._set_ckpt_state("", theme.TEXT_DIM)

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

    def add_policy_plan(self, v) -> None:
        """One diffusion-policy proposal, straight through to the plot."""
        self.view.add_policy_plan(v)

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
        # LOW HONESTY RULE (review 2026-09-11): the combo shows the mode the
        # WORKER is on. A pick the worker refused (engaged, CSV open, still
        # building) or dropped never changes `s.mode`, so the combo snaps
        # back within one status instead of standing as a promise nobody
        # kept. Blocked (set_mode_default) — reflecting, not commanding.
        m = str(s.mode or "").lower()
        if m in MODE_LABELS and m != self.current_mode():
            self.set_mode_default(m)
            self._refresh_mission_lbl()
        # ...and it is frozen while engaged (the worker refuses a swap
        # mid-engagement) and while a controller CSV is open OUTSIDE one (a
        # REC-opened CSV pinned the run folder under the old mode's tree;
        # workers.set_mode refuses too — the combo says so by being dead).
        self.mode_box.setEnabled(not s.engaged
                                 and not getattr(s, "csv_open", False))
        # A checkpoint swap is refused by the PolicyWorker while a mission is
        # armed (anything engaged may be); the button says so the same way.
        # ...and while a REC-opened controller CSV is open (review 2026-09-11):
        # that CSV's meta names ONE checkpoint under `ckpt_loaded`, and a swap
        # under it would put rows from two networks into one file (the worker
        # records such a change as `ckpt_changes_while_csv_open` if it happens
        # anyway) — the same rule that freezes the LOW combo above.
        self.btn_ckpt.setEnabled(not s.engaged
                                 and not getattr(s, "csv_open", False))
        # The mission is frozen from START, not from take-off. The worker
        # SNAPSHOTS the merged mission the moment START is pressed
        # (workers.py set_traj -> self._approach["sq"]) and arms the path from
        # that copy after the approach and the settle, so a field edited while
        # the vehicle is on its way to the tag updates this panel's label and
        # the worker's override and then never reaches the flight. Keyed on
        # traj_on alone that was a window of approach_max_s + settle_s (up to
        # ~190 s) in which the panel said one thing and the run flew another.
        busy = s.traj_on or s.phase in ("approach", "settle")
        for w in (self.shape_box, self.tag_box, self.face_box, self.len_box,
                  self.leny_box, self.spd_box):
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
            # POLICY OBSERVE NEVER SAYS "ENGAGED". The word is what an
            # operator reads to answer "is the station flying it?", and here
            # the answer is no — the loop is open and the sticks are theirs.
            # `s.mode` is dropped with it: naming the solver would imply one
            # ran, and `ctrl.step()` was never called.
            head = ("OBSERVE · DP NOT COMMANDING"
                    if getattr(s, "observe", False) else f"ENGAGED [{s.mode}]")
            if fresh_refusal:
                self.chip.setText(f"{head} {what} · {self._ref_msg}")
            else:
                self.chip.setText(f"{head} {what}{det}{err}{warm}")
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


class CkptDialog(QtWidgets.QFileDialog):
    """The checkpoint picker (HIGH = Diffusion Policy, the `…` button).

    OWNED BY THE WINDOW (MainWindow._pick_policy_ckpt), not the panel: while
    it is up the keyboard belongs to the dialog, so the window hands the keys
    back from the vehicle first (teleop.all_stop) and says so in the log —
    the same rule as typing a recording name (window.py eventFilter). The
    JOYSTICK keeps flying throughout: its reader is a timer, and a modal
    exec keeps timers alive.

    NON-NATIVE on purpose: a native/portal dialog never runs any of this,
    and Esc here must be what Esc is everywhere else in the station — E-STOP
    (window.py "Keyboard") — as well as closing the dialog. It emits
    ``estop_requested`` (the window connects it to `estop`) and rejects.
    Cancel and the window close button just reject.

    Esc is caught by an APPLICATION-level event filter while the dialog is
    visible, not only by ``keyPressEvent`` (review 2026-09-11): Qt's
    non-native QFileDialog opens in Detail view, whose file list (the private
    QFileDialogTreeView, objectName "treeView") handles Esc ITSELF — its key
    handler goes through QFileDialogPrivate::itemViewKeyboardEvent, which
    calls reject() and accepts the event, so the dialog's keyPressEvent never
    ran once the operator had clicked a file; and the filename completer's
    popup eats the first Esc to close itself. Both closed the dialog WITHOUT
    the E-STOP, i.e. the promise held only while the filename box, the
    sidebar or a combo had focus. The app filter sees every KeyPress in the
    main thread before its target does, and the dialog is modal, so while
    it is visible every key press in the application is its (the popups
    included). Installed on show, removed on hide.
    """

    estop_requested = Signal()

    def __init__(self, parent=None, caption: str = "Diffusion policy checkpoint",
                 directory: str = "",
                 filt: str = "Checkpoints (*.ckpt);;All files (*)"):
        super().__init__(parent, caption, directory, filt)
        self.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog, True)
        self.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        self.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptOpen)
        self._filtering = False         # the app-level filter is installed

    # ---- the Esc contract, whichever child has the focus
    def showEvent(self, ev) -> None:
        super().showEvent(ev)
        app = QtWidgets.QApplication.instance()
        if app is not None and not self._filtering:
            app.installEventFilter(self)
            self._filtering = True

    def hideEvent(self, ev) -> None:
        self._drop_filter()
        super().hideEvent(ev)

    def _drop_filter(self) -> None:
        # Safe from inside eventFilter: QObject.removeEventFilter nulls the
        # entry rather than reshaping the list being walked.
        app = QtWidgets.QApplication.instance()
        if app is not None and self._filtering:
            app.removeEventFilter(self)
        self._filtering = False

    def eventFilter(self, obj, ev):
        if (self._filtering and self.isVisible()
                and ev.type() == QtCore.QEvent.Type.KeyPress
                and ev.key() == Qt.Key.Key_Escape):
            self._escape()
            return True                 # the tree view / popup never sees it
        return super().eventFilter(obj, ev)

    def _escape(self) -> None:
        # A popup open over the dialog (the filename completer's list, a
        # combo's list) would have eaten this Esc to close itself; the filter
        # took it instead, so close the popup here or it outlives the dialog
        # as an orphan window (seen in the review's offscreen probe,
        # 2026-09-11). Bounded: a popup that refuses to close must not hang
        # the E-STOP.
        app = QtWidgets.QApplication.instance()
        for _ in range(4):
            pw = app.activePopupWidget() if app is not None else None
            if pw is None:
                break
            pw.close()
        self.estop_requested.emit()
        self.reject()                   # hides -> hideEvent drops the filter

    def keyPressEvent(self, ev) -> None:
        # The filter answers first when it is installed; this is the same
        # contract for a key that reaches the dialog itself (kept: cheap, and
        # it is what a reader of QDialog expects to find).
        if ev.key() == Qt.Key.Key_Escape:
            self._escape()
            return
        super().keyPressEvent(ev)
