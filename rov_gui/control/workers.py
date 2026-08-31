#!/usr/bin/env python3
"""workers.py — the two Qt workers that close the MPC loop inside the station.

Both are :class:`TimerWorker`s and that is forced, not chosen: each one has
to RECEIVE (enable/engage/mode/reset/log slots), and only a TimerWorker has
the event loop that delivers a queued slot (backends/base.py; the package
already shipped one LoopWorker with a slot that silently never fired).

``TagNavWorker``   ~66 Hz poll of a one-slot RGB mailbox -> detect + PnP ->
                   ``bus.nav_fix`` at camera rate. Mirrors PoseWorker's shape
                   (hardware.py:2154) on a second mailbox, so tag navigation
                   and SAM2 tracking can coexist without stealing each
                   other's frames.

``MpcWorker``      50 ms tick = the controller's DT_CTRL (baked into the
                   generated acados solver — the interval is not tunable).
                   Assembles the state, runs the EAOB+NMPC bridge, emits
                   ``bus.cmd_pilot`` into the EXISTING command chain, and
                   writes the run CSV. Engage/trajectory arbitration and all
                   runtime interlocks live here, in one place.

Command-arbitration contract with the window (window._pump_commands): while
``MpcStatus.engaged`` is true the window stops pumping teleop frames and any
pilot axis input makes it emit ``cmd_mpc_engage(False)``. If this worker dies
mid-engagement the sink's 500 ms deadman drives the vehicle to neutral — the
same backstop the human pilot has.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np

from .. import runstore
from ..qt import Slot
from ..state import (Conn, MpcStatus, NavFix, ObjectFix, PilotInput,
                     TagOverlay, now)
from .allocation import axes_to_wrench, slew_axes, wrench_to_axes
from .geometry import MpcConfig, NavConfig, SHAPES, yaw_from_R
from .imu_dr import ImuCalibration, ImuDeadReckoner
from . import object_nav as ON
from . import station_bridge as SB
from .state_assembler import StateAssembler, rot_zyx
from ..backends.base import TimerWorker


def _wrap_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _scenario_lap(scen: dict, t: float) -> int:
    """Completed laps, whichever shape is running (a LINE lap is one out-and-
    back, a rectangle lap is one circuit, a CIRCLE lap is one revolution)."""
    from .reference import circle_lap_of, line_lap_of, rect_lap_of

    kind = scen.get("kind")
    if kind == "line":
        return line_lap_of(t, scen["length"], scen["speed"], scen["ramp_s"])
    if kind == "circle":
        return circle_lap_of(t, scen["radius"], scen["speed"])
    if "size" not in scen:
        return 0        # replay (a demo has no lap structure) / future kinds
    return rect_lap_of(t, scen["size"], scen.get("size_y", scen["size"]),
                       scen["speed"])


def _flu_of_ned(x, y, z):
    """NED -> world-FLU mirror (S = diag(1,-1,-1)), for the sim-schema CSV."""
    return float(x), -float(y), -float(z)


# =============================================================================
# tag navigation
# =============================================================================
class TagNavWorker(TimerWorker):
    """Frames in (one mailbox PER video feed) -> TagOverlay out per feed, and
    NavFix out for the ONE feed that can localize.

    Two kinds of feed, one worker:
      * the LOCALIZING feed (``hw_nav.yaml: nav_source`` — "main" = C3 with
        per-frame factory intrinsics, or "second" = the ROV RGB with the
        [예측] second_cam calibration) — detections there run PnP against the
        tag map and become the vehicle's NavFix;
      * every other feed is an OVERLAY ONLY — the pilot sees what that camera
        sees, and ``TagOverlay.localizes`` stays False so nobody mistakes a
        pretty outline for a state estimate.
    Each feed has its own on/off (the TAG button on its panel); off means the
    producer does not even copy a frame (mailbox ``wanted`` gate). Toggling
    the LOCALIZING feed off also resets the first-fix datum, so off->on is
    the "re-zero here" gesture."""

    def __init__(self, bus, mailboxes: dict, opts,
                 nav_cfg: NavConfig | None = None):
        super().__init__("tagnav", interval_ms=15)
        self.bus = bus
        self.mailboxes = dict(mailboxes)  # panel name -> RgbdMailbox
        self.opts = opts
        self.cfg = nav_cfg
        self.nav = None
        self.dets: dict = {}              # panel -> TagDetector (per-feed)
        self.solve_panel = "main"
        self.enabled: dict = {}           # panel -> bool
        self._imu = None                  # latest VehicleImu, for IPPE flips
        self._last_miss_note = 0.0
        self._n_fix = 0
        self._fix_marks: list[float] = []  # accepted-fix stamps -> measured Hz

    def setup(self) -> None:
        if self.cfg is None:
            self.cfg = NavConfig.load(
                getattr(self.opts, "nav_config", "config/hw_nav.yaml"),
                geometry_override=getattr(self.opts, "nav_geometry", None))
        from .tagnav import TagDetector, TagNav

        c = self.cfg
        self.solve_panel = c.nav_source
        # One detector PER FEED: decimation is a per-resolution decision
        # (640x360 C3 wants 1.0, 720p+ ROV RGB wants 2.0) and pupil pins it
        # at construction.
        self.dets = {panel: TagDetector(family=c.tag_family,
                                        backend=c.detector,
                                        quad_decimate=c.decimate_for(panel))
                     for panel in self.mailboxes}
        R_bc, t_bc = c.R_t_frd_cam(self.solve_panel)
        self.nav = TagNav(c.make_tag_map(), c.effective_tag_size(),
                          R_bc, t_bc, c.R_ned_map,
                          max_reproj_px=c.max_reproj_px,
                          min_tags=c.min_tags,
                          datum=c.datum,
                          tilt_gate_deg=c.tilt_gate_deg,
                          ambiguity_ratio=c.ambiguity_ratio,
                          duplicate_ids=c.duplicate_ids,
                          dup_confirm_px=c.dup_confirm_px)
        # Defaults: the localizing feed ON (the MPC needs it), extras OFF
        # (detection on a second 30 fps stream is CPU spent only when asked).
        # The window mirrors these onto the TAG buttons without re-emitting.
        for panel, mb in self.mailboxes.items():
            self.enabled[panel] = (panel == self.solve_panel)
            mb.set_wanted(self.enabled[panel])
        if self.solve_panel not in self.mailboxes:
            self.bus.log.emit(
                "error", f"tagnav: nav_source={self.solve_panel!r} has no "
                         f"feed (have {sorted(self.mailboxes)}) — no "
                         f"localization will be produced")
        if self.solve_panel == "second":
            self.bus.log.emit(
                "warn", "tagnav: localizing from the ROV RGB with the [예측] "
                        "second_cam calibration — distances scale with its "
                        "fx guess; calibrate before quoting numbers")
        backend = next(iter(self.dets.values())).backend if self.dets else "?"
        decs = {p: c.decimate_for(p) for p in sorted(self.mailboxes)}
        self.bus.log.emit(
            "info", f"tagnav: {backend}, geometry={c.geometry}, "
                    f"map={len(self.nav.map)} tag(s), size "
                    f"{c.effective_tag_size():.3f} m, decimate={decs}, "
                    f"localizing={self.solve_panel}, datum={c.datum}")

    @Slot(object)
    def set_imu_hint(self, imu) -> None:
        self._imu = imu

    @Slot(str, bool)
    def set_source_enabled(self, panel: str, on: bool) -> None:
        if panel not in self.mailboxes:
            return
        self.enabled[panel] = bool(on)
        self.mailboxes[panel].set_wanted(bool(on))
        if not on:
            # One explicit CLEAR so the canvas never shows a stale outline.
            self.bus.tag_overlay.emit(TagOverlay(panel=panel, enabled=False,
                                                 stamp=now()))
            if panel == self.solve_panel and self.nav is not None:
                # Off->on on the localizing feed is the RE-ZERO gesture: the
                # next accepted fix defines (0,0) and yaw 0 again (datum
                # first_fix; a no-op under datum map).
                self.nav.reset_datum()
        self.bus.log.emit("info", f"tagnav: {panel} detection "
                                  f"{'ON' if on else 'off'}")

    def tick(self) -> None:
        if self.nav is None:
            return
        for panel, mb in self.mailboxes.items():
            if not self.enabled.get(panel):
                continue
            item = mb.take()
            if item is not None:
                self._process(panel, item)

    def _intrinsics_for(self, panel: str, item: dict, gray):
        """(K, dist) for one feed. "main" = the C3's factory intrinsics that
        ride each frame; "second" = the [예측] config model, rescaled to the
        frame size actually received; anything else = no calibration."""
        if panel == "main":
            intr = item.get("K")
            if intr is None:
                return None, None
            d = np.asarray(getattr(intr, "distortion", ()) or (), float)
            return intr.K, (d if d.size else None)
        if panel == "second":
            return self.cfg.second_K(gray.shape[1], gray.shape[0])
        return None, None

    def _process(self, panel: str, item: dict) -> None:
        import cv2

        color = item["color"]
        gray = (color if color.ndim == 2
                else cv2.cvtColor(color, cv2.COLOR_BGR2GRAY))
        t0 = now()
        dets = self.dets[panel].detect(gray)
        detect_ms = 1e3 * (now() - t0)
        K, dist = self._intrinsics_for(panel, item, gray)
        localizes = (panel == self.solve_panel and K is not None)
        t_cap = float(item.get("t_capture", now()))
        self.bus.tag_overlay.emit(TagOverlay(
            panel=panel,
            quads=tuple(tuple((float(x), float(y)) for x, y in d.corners)
                        for d in dets),
            ids=tuple(d.tag_id for d in dets),
            mapped=tuple(d.tag_id in self.nav.map for d in dets),
            src_w=gray.shape[1], src_h=gray.shape[0],
            detect_ms=detect_ms, localizes=localizes, enabled=True,
            # The corners ride with the camera model and capture time that
            # produced them, so a REC NAV recording is a re-usable set of raw
            # observations (window.py writes detections.csv from this).
            t_capture=t_cap,
            K=((float(K[0][0]), float(K[1][1]), float(K[0][2]),
                float(K[1][2])) if K is not None else ()),
            dist=(tuple(float(v) for v in np.asarray(dist).ravel())
                  if dist is not None else ()),
            stamp=now()))
        if not localizes:
            return                          # overlay-only feed (or no K yet)
        hint = None
        imu = self._imu
        if imu is not None and imu.roll is not None:
            hint = (float(imu.roll), float(imu.pitch))
        sol = self.nav.solve(dets, K, dist, rp_hint=hint)
        if sol is None:
            t = now()
            if t - self._last_miss_note > 1.0:
                self._last_miss_note = t
                # The WHY travels with the miss — "seen, none usable" alone is
                # undebuggable at the pool (live finding, 2026-08-12).
                why = self.nav.last_reject or "none usable"
                self.bus.nav_fix.emit(NavFix(
                    t_capture=t_cap, n_tags=0,
                    # The ids ride the MISS too. Without them a rejected
                    # frame said only "39 seen" and no recording could ever
                    # name the culprit — which is exactly what blocked the
                    # 2026-08-13 diagnosis (0/114 and 0/146 rejected rows
                    # carried ids in sessions/nav_runs/20260813_17*).
                    tag_ids=tuple(d.tag_id for d in dets),
                    geometry=self.cfg.geometry, source=self.solve_panel,
                    conn=Conn.DEGRADED,
                    note=(f"{len(dets)} seen: {why}" if dets
                          else "no tags in view"),
                    src_w=gray.shape[1], src_h=gray.shape[0], stamp=now()))
            return
        self._n_fix += 1
        t = now()
        self._fix_marks.append(t)
        if len(self._fix_marks) > 30:
            self._fix_marks = self._fix_marks[-30:]
        span = self._fix_marks[-1] - self._fix_marks[0]
        hz = ((len(self._fix_marks) - 1) / span
              if len(self._fix_marks) > 1 and span > 0 else None)
        self.bus.nav_fix.emit(NavFix(
            t_capture=t_cap, n_tags=sol.n_tags, tag_ids=sol.tag_ids,
            tag_insts=sol.tag_insts,
            p_ned=tuple(float(v) for v in sol.p_ned),
            R_ned_body=tuple(float(v) for v in sol.R_ned_body.ravel()),
            yaw_ned=yaw_from_R(sol.R_ned_body),
            reproj_rms_px=sol.reproj_rms_px, detect_ms=sol.detect_ms,
            hz=hz, ambiguous=sol.ambiguous, geometry=self.cfg.geometry,
            source=self.solve_panel,
            src_w=gray.shape[1], src_h=gray.shape[0],
            conn=Conn.ONLINE, note=sol.note, stamp=t))

    def teardown(self) -> None:
        for mb in self.mailboxes.values():
            mb.set_wanted(False)


# =============================================================================
# the control loop
# =============================================================================
CSV_HEADER = ("t,px,py,pz,rx,ry,yaw_deg,pitch_deg,lap,"
              "rz,ryaw_deg,t_traj,mode,engaged,traj_on,"
              "solver,solver_status,solve_ms,"
              "n_tags,pnp_rms_px,tag_age_s,imu_age_s,z_src,ambig,"
              "w0,w1,w2,w3,w4,w5,uX,uY,uZ,uK,uM,uN,"
              "ax_surge,ax_sway,ax_heave,ax_yaw,nis,"
              "pwm_dev_us,e_along,e_cross,"
              "path_s_m,ref_speed_m_s,"
              "dr_px,dr_py,dr_pz,dr_pz_imu,dr_yaw_deg,dr_err_m,dr_err_z_m,"
              "dr_t_s,dr_hz,dr_n,dr_ok,rp_residual_deg,roll_deg,"
              "bridge_s,bridge_tier,"
              "obj_px,obj_py,obj_pz,obj_yaw_deg,obj_age_s,obj_pair_dt_ms,"
              "obj_pair_exact,obj_state,follow_state,follow_err_m,tick_ms,"
              "plan_id,ref_src,grip_cmd\n")
# NOTE for anyone loading an older CSV: runs before 2026-08-14 carry an extra
# `geofence_ok` column between `nis` and `pwm_dev_us`. The geofence was removed
# that day, so the column would have been a constant 1 and a lie about a guard
# that no longer exists. `e_along` / `e_cross` (the tracking error split along
# the path tangent — see MpcWorker._path_split) arrived the same day and are
# absent from every run before it; they are derivable from px,py,rx,ry only if
# you reconstruct the tangent yourself. The five `path_*`/corner/ref-speed
# audit fields arrived with the 2026-08-16 spatial follower. Read by NAME
# (pandas.read_csv handles every generation), never by position.
#
# 2026-08-17: the twelve `dr_*` / `rp_residual_deg` columns arrived with the
# IMU dead-reckoning experiment. `dr_px/py/pz` are world FLU in the datum
# frame, the SAME convention as `px/py/pz`, so the drift is `dr_px - px`
# straight off the file. They are `nan` on every run with `imu_dr.enabled:
# false`, which is every run before this date — and `meta.json`'s `imu_dr`
# block, not the column's presence, is what says whether an estimator ran.
# One more break with the older generations: when the DR is on, a row is
# written even on ticks with NO tag fix (the tag columns go `nan`), because
# drift during a dropout is exactly what such a run is recording. With the DR
# off, a fixless tick still writes nothing.
#
# 2026-08-18: `bridge_s` / `bridge_tier` arrived with the STATION BRIDGE
# (control/station_bridge.py). `bridge_s` is 0 on a normal tick and counts the
# seconds since the last fresh tag fix while a station hold is being carried;
# `bridge_tier` is none|imu|coast. THIS IS A RECORD BOUNDARY: rows with
# bridge_s > 0 were NOT flown on the tag — px/py there are an IMU estimate (or
# a frozen anchor, see meta station_bridge.xy_source) and the coast rows had
# surge/sway/yaw commanded to zero regardless of what the controller asked
# for. Never pool them with clean ticks.
#
# `roll_deg` (world FLU, like yaw_deg and pitch_deg) arrived at the same time
# and closes an older gap: the CSV recorded two of the three Euler angles, so
# no reader could reconstruct the attitude the run was flown at. Re-estimating
# an IMU track offline needs all three — anchoring level while the vehicle was
# rolled leaks g*sin(roll) straight into the horizontal.
#
# 2026-08-21: the ten `obj_*` / `follow_*` columns arrived with OBJECT FOLLOW
# (control/object_nav.py). `obj_px/py/pz` are world FLU in the DATUM frame —
# the SAME convention as `px/py/pz` and `dr_px/py/pz` — so the vehicle-to-
# object vector is `obj_px - px` straight off the file, with no transform in
# between. (The ObjectFix on the bus is MAP-frame; the conversion happens once,
# in `_obj_row`.) They are `nan` on every run without `--pose`, which is every
# run before this date.
#
# `obj_pair_exact` IS A RECORD BOUNDARY and the most important of the ten.
# The camera extrinsic cancels out of the object composition ONLY when the
# object pose and the tag fix came from the same camera frame; a row with
# `obj_pair_exact == 0` was composed across frames and carries up to
# |t_frd_cam| * 2*sin(dyaw/2) of extra position error from the unmeasured
# 0.2855 m camera lever arm. Never pool object-position statistics across
# rows that differ in it.
#
# 2026-08-30: `plan_id` / `ref_src` / `grip_cmd` arrived with the REPLAY
# mission (shape: replay, control/plan_stream.py). `plan_id` is the streamed
# plan the stitcher is actively sampling (nan outside a replay), `ref_src` is
# where this tick's reference came from (plan|blend|hold, "" outside a
# replay), and `grip_cmd` is the -1/0/+1 jaw drive the replay is holding (nan
# outside; the jaw path is open-loop, so this records the COMMAND, not the
# jaw). A replay run's reference is a streamed plan, not a placed geometry —
# a NEW reference family. Never pool replay rows with geometric-mission rows;
# `trajectory.kind == "replay"` in the meta is the boundary, and plans.jsonl
# beside the CSV carries the per-plan filter verdicts for planner-vs-tracker
# attribution.


class MpcWorker(TimerWorker):
    """The 20 Hz brain. See the module docstring for the arbitration contract.

    ``controller_factory`` exists for the offline tests and the demo: it must
    return an object with the :class:`~rov_gui.control.mpc_bridge.HwDobMpc`
    surface (step / set_target_ned / set_square_ned / set_line_ned /
    set_path_plan_ned / ref_ned_at / note_applied / reset / realtime_ok /
    solver_kind / meta). The default factory builds the real thing, which
    imports casadi+acados and may spend tens of seconds code-generating the
    solver — that is WHY it happens in setup() at startup and never at engage
    time.
    """

    def __init__(self, bus, opts, controller_factory=None):
        super().__init__("mpc", interval_ms=50)
        self.bus = bus
        self.opts = opts
        self._factory = controller_factory
        self.cfg: MpcConfig | None = None
        self.nav_cfg: NavConfig | None = None
        self.ctrl = None
        self._pid = None
        self._mpc_ctrl = None
        self.asm: StateAssembler | None = None
        # IMU dead reckoning, built in setup() only when it is enabled. None
        # is the off state and every call site checks it, so a build with the
        # experiment off runs the tick it always ran.
        self.dr: ImuDeadReckoner | None = None
        self.dr_control = False            # controller eats the DR state
        self._dr_q: deque = deque(maxlen=100)   # batches awaiting integration
        self._dr_last = None               # last state() result, for CSV/status
        self._dr_overflow = 0
        self._vimu_hist: deque = deque(maxlen=1200)
        self._ready = False
        self._setup_error = ""

        self.fix = None                    # latest NavFix
        # RAW tag fixes, kept ONLY so an object pose can be paired with the
        # fix from ITS OWN camera frame (control/object_nav.py). 150 entries
        # is ~5 s at camera rate — far more than pair_tol_s needs, and cheap.
        # Raw, never `meas["eta"]`: the assembler's state is a hybrid and
        # composing an object pose out of it silently breaks the extrinsic
        # cancellation the whole feature rests on.
        self._fix_hist: deque = deque(maxlen=150)
        self._obj = None                   # ON.ObjectAnchor, or None (no --pose)
        self._obj_ext = None               # (R_frd_cam, t_frd_cam) for its camera
        self._obj_src_ok = True            # do the object and fix cameras agree?
        self._obj_note = ""                # ...and why not, in words
        self._obj_last: ObjectFix | None = None
        self.imu = None                    # latest VehicleImu
        self.tel = None                    # latest Telemetry
        self.thr = None                    # latest ThrusterState (PWM feedback)
        self.cmd_enabled = False           # COMMAND ENABLE mirror
        # Injected by HardwareBackend: the command sink's status() — engage
        # must refuse while the sink CANNOT deliver (no peer / SYSID
        # mismatch). Absent in demo.
        self.sink_status_fn = None
        # commands-out-vs-motors-silent watchdog (2026-08-12 pool session:
        # 24 s of |axes|~0.2 with the vehicle not moving and nothing said so)
        self._cmd_active_ticks = 0
        self._last_noresp_warn = 0.0

        self.engaged = False
        self.traj_on = False
        self._axes_prev = None          # never ramp from a stale command
        self._auto_traj = False            # one-button flow: traj after warm-up
        # Mission datum, captured at ENGAGE: the engage pose in the TAG frame
        # becomes (0,0,0)/yaw 0 for everything downstream (controller, square,
        # CSV, plot). Set fresh at every engage — the operator's
        # "(0,0) = where I pressed START". Kept across disengage so the plot
        # and post-run CSV rows stay in the frame the run used.
        self._datum = None                 # {"p0": (3,), "yaw0": f, "Rz": 3x3}
        self.reason = ""
        self._warmup_left = 0
        self._t0_traj: float | None = None
        self._t_engage: float | None = None
        # STATION BRIDGE (station_bridge.py): the tier ladder that keeps a
        # station hold alive across a tag dropout instead of disengaging into
        # a sinking vehicle. `_bridge_dr` is its OWN estimator, RE-ANCHORED on
        # every fresh fix — deliberately not `self.dr`, whose whole purpose is
        # to never re-anchor so its drift can be measured.
        self._bridge = None
        self._bridge_dr = None
        self._bridge_anchor = None         # eta of the last fresh tag fix
        self._bridge_yaw = 0.0             # ...and its yaw, gyro-integrated
        self._bridge_z = 0.0               # barometer z, and its filtered
        self._bridge_vz = 0.0              # derivative, for the freeze state
        self._bridge_xy = "hold"
        self._bridge_dr_warned = False
        self._bridge_note = ""
        self._eta = None                   # last assembled state (np arrays)
        self._last_nu = None               # ...and the body velocity with it,
        # kept for the run record: C(nu) and D(nu) are state-dependent, so the
        # matrices written into controller.json are samples at a stated nu
        # rather than constants pretending otherwise (see control/plant.py).
        self._plant_meta: dict | None = None   # cached; read once per process
        self._nfail_prev = 0
        self._fail_streak = 0
        # Wall time of the last controller step, and how many in a row blew
        # the budget. Debounced like every other interlock in this class
        # (tag_stale_hold_s, max_solver_fails): this process shares a GIL with
        # Qt and the pose pipeline, so ONE late tick is a scheduling hiccup,
        # not a controller that stopped working — and the cure, a disengage,
        # drops depth hold on a -5.7 N vehicle.
        self._tick_ms = 0.0
        self._over_streak = 0
        #: The last follow that ENDED, kept so its constants survive into the
        #: run meta — `_follow_to_station` clears `self.follow`.
        self._follow_last = None
        self._scenario_override: dict = {}
        self._tagmap = None                # lazy: only for origin_tag lookups
        # Set between START and the path actually arming: fly to the origin
        # under DP, settle there, then go (see _tick_approach).
        self._approach: dict | None = None
        self.phase = ""                    # "" | warmup | approach | settle
        self.phase_detail = ""             #  | station | line | square | circle
        self.station: dict | None = None   # STATION mode: hold here, forever
        # FOLLOW mode: hold a captured relative pose on the tracked object.
        # Cleared everywhere `station` is (set_traj(False), _arm_path,
        # disengage, set_engaged) — a follow that outlived a disengage would
        # keep walking a setpoint for a loop that is no longer running.
        self.follow: dict | None = None
        # REPLAY mode (shape: replay): stream one recorded handheld demo
        # through the plan seam (control/plan_stream.py) — the same install
        # path a live diffusion policy will feed later. Cleared at the same
        # 4 sites as `follow`, and clearing it also NEUTRALS the jaw: a
        # gripper drive is a latched level in the command sink, so a replay
        # that dies without emitting 0.0 leaves the jaw driving forever.
        self.replay: dict | None = None
        #: The last replay that ENDED, kept for the run meta (the follow
        #: pattern: the runs worth a post-mortem are the ones no longer live).
        self._replay_last: dict | None = None
        self._plan_filter = None           # plan_stream.PlanFilter
        self._plan_stitcher = None         # plan_stream.PlanStitcher
        # In geometric path mode `_tau` is only the nominal time-coordinate
        # of the spatial target (for CSV/backward compatibility); vehicle
        # projection, not elapsed time, advances it.
        self._tau = 0.0
        self._tau_t = 0.0
        self._path_lag = 0.0
        self._path_cursor = None
        self._path_err = (None, None)
        self._path_depth = 0.0
        self._path_yaw_fixed = 0.0
        self._path_heading_follow = False
        self._axes_prev = None             # last SENT axes, for the slew limit
        self._stale_ticks = 0              # debounce for the stale-fix gate
        self._last_health: dict = {}

        # SESSION-level lines (the build fingerprint), held until a run folder
        # is opened for a real event — see _log_event(defer=True). `_events_path`
        # is the folder last written to, so each new folder gets the banner once.
        self._session_banner: list[str] = []
        self._events_path: Path | None = None

        self._csv = None
        self._csv_path: Path | None = None
        self._run_dir_pin: Path | None = None   # see _run_dir()
        self._csv_started = ""             # wall clock at open, for meta.json
        self._csv_auto = False             # opened by engage, closed by release
        self._t0_csv = 0.0
        self._rows = 0

    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        if self.stopping:                    # shut down before we even began
            return
        self.cfg = MpcConfig.load(getattr(self.opts, "mpc_config",
                                          "config/hw_mpc.yaml"))
        if getattr(self.opts, "mpc_mode", None):
            self.cfg.mode = str(self.opts.mpc_mode)
        if getattr(self.opts, "rov_model", None):
            self.cfg.rov_model = str(self.opts.rov_model)
        if getattr(self.opts, "replay_session", None):
            # --replay-session PATH: point shape `replay` at a demo folder
            # without editing hw_mpc.yaml at the pool.
            self.cfg.replay["session"] = str(self.opts.replay_session)
        # ONE root per run. --rec-dir governs the recorders and the nav
        # recording; without this line hw_mpc.yaml's log_dir governs the CSV
        # and events.log independently, and `--rec-dir /media/ssd/runs` would
        # split a run across two trees with nothing saying so. Explicit
        # `mpc_log_dir` still wins if someone genuinely wants them apart.
        root = (getattr(self.opts, "mpc_log_dir", None)
                or getattr(self.opts, "rec_dir", None))
        if root and str(root) != self.cfg.log_dir:
            self.bus.log.emit(
                "info", f"mpc: writing runs under {root} "
                        f"(hw_mpc.yaml log_dir {self.cfg.log_dir} overridden — "
                        f"one root per run)")
            self.cfg.log_dir = str(root)
        self.nav_cfg = NavConfig.load(
            getattr(self.opts, "nav_config", "config/hw_nav.yaml"),
            geometry_override=getattr(self.opts, "nav_geometry", None))
        self.asm = StateAssembler(
            z_source=self.nav_cfg.z_source,
            vel_lp_alpha=self.cfg.vel_lp_alpha,
            nudot_source=self.cfg.nudot_source,
            tag_stale_s=float(self.cfg.engage["tag_stale_s"]),
            imu_stale_s=float(self.cfg.engage["imu_stale_s"]),
            propagate=self.cfg.vel_propagation)
        self._setup_imu_dr()
        self._setup_station_bridge()
        self._setup_object_nav()
        # Both controller families are built up front: the PID costs nothing
        # and stays available even when the acados build fails, so the mode
        # combo can swap between them without a rebuild (never mid-engage —
        # set_mode refuses that).
        from .pid import HwPid

        self._pid = HwPid(self.cfg, log=lambda m: self.bus.log.emit("info", m))
        self._mpc_ctrl = None
        self._mpcc_ctrl = None
        try:
            if self._factory is not None:
                self._mpc_ctrl = self._factory(self.cfg)
            else:
                from .mpc_bridge import HwDobMpc
                # Build under the file's mode when that mode is one this
                # object serves; otherwise (pid, mpcc, dobmpcc) build the
                # default and let set_mode flip it later.
                self._mpc_ctrl = HwDobMpc(
                    self.cfg.mode if self.cfg.mode in HwDobMpc.MODES
                    else "dobmpc",
                    self.cfg, log=lambda m: self.bus.log.emit("info", m))
        except Exception as e:                                   # noqa: BLE001
            self._setup_error = f"{type(e).__name__}: {e}"
            self.bus.log.emit("error", f"mpc: MPC controller build FAILED — "
                                       f"{self._setup_error}. PID mode is "
                                       f"still available.")
        # MPCC is a SECOND acados solver (13 states, 7 inputs, a contouring
        # cost) and therefore a second build. Built here beside the others so
        # the mode combo never pays a code-generation wait mid-session; a
        # failure here costs only mpcc mode.
        if self._factory is None:
            try:
                from .mpcc_bridge import HwMpcc
                self._mpcc_ctrl = HwMpcc(
                    self.cfg, mode=("dobmpcc" if self.cfg.mode == "dobmpcc"
                                    else "mpcc"),
                    log=lambda m: self.bus.log.emit("info", m))
            except Exception as e:                               # noqa: BLE001
                self.bus.log.emit(
                    "warn", f"mpc: MPCC unavailable ({type(e).__name__}: {e}) "
                            f"— mpcc mode will refuse")
        self.ctrl = self._ctrl_for(self.cfg.mode)
        self._ready = True
        if self.ctrl is not None:
            k = self.ctrl.solver_kind
            ok = self.ctrl.realtime_ok
            self.bus.log.emit(
                "warn" if not ok else "info",
                f"mpc: {self.cfg.mode} ready, solver={k}, "
                f"probe={self.ctrl.probe_ms and round(self.ctrl.probe_ms, 1)} ms"
                + ("" if ok else " — NOT real-time capable, engage will refuse"))
        # A one-line BUILD FINGERPRINT in the mission log. On 2026-08-14 a pool
        # session was spent on a refusal that had already been removed from the
        # source — the GUI running was simply the older process, and nothing on
        # screen said so. Now the first line of every run states what this
        # build actually does, so "am I on the current code?" is answerable at
        # a glance instead of by reading a status bar mid-flight.
        #
        # DEFERRED to disk (2026-08-14, second pass). Writing it immediately
        # created a run folder for the mere act of launching the station: on
        # 2026-08-14 seven folders held nothing but this one line, and nothing
        # was ever flown in them. It still reaches the mission log at once, and
        # it still heads up events.log in EVERY run folder — it is just written
        # when there is finally a run to head up.
        sq = self.cfg.square
        self._event(
            f"ready: {self.cfg.mode}, "
            f"{'geometric-path/corner-gate' if self.cfg.path_following else 'traj-tracking'}"
            f", no geofence, mission={sq.get('shape')}"
            + (f" @tag{sq.get('origin_tag')}" if sq.get("origin_tag") else ""),
            defer=True)
        # AFTER the fingerprint, never before: that line has to head up every
        # run folder's events.log, and a deferred line emitted during setup
        # would slip in front of it.
        if self._bridge_note:
            self._event(self._bridge_note, defer=True)

    # -------------------------------------------------------- imu dead reckon
    def _setup_imu_dr(self) -> None:
        """Build the dead reckoner if the experiment is on, and say so LOUDLY.

        Two CLI overrides on top of the YAML, and one of them is a safety
        gate: ``--imu-dr`` picks off/shadow/control, but ``control`` ALSO
        needs ``--imu-dr-control``. Flying a closed loop on dead reckoning is
        not something a config file left in the wrong state should be able to
        start on its own — same reasoning as ``--allow-command``.
        """
        d = dict(self.cfg.imu_dr)
        want = getattr(self.opts, "imu_dr", None)
        if want:
            d["enabled"] = str(want) != "off"
            if str(want) in ("shadow", "control"):
                d["mode"] = str(want)
        if getattr(self.opts, "imu_dr_attitude", None):
            d["attitude"] = str(self.opts.imu_dr_attitude)
        self.cfg.imu_dr = d
        if not d.get("enabled"):
            self.dr, self.dr_control = None, False
            return

        calib = ImuCalibration.identity()
        cpath = d.get("calibration")
        if str(getattr(self.opts, "source", "")) == "demo":
            # A hardware calibration must never be applied to a FABRICATED
            # sensor. The demo's IMU is ideal by construction, so correcting
            # it with the real camera's 1.8 m/s^2 bias does not remove an
            # error, it INJECTS one — 243 m of drift in 9 s, which looks
            # exactly like a broken estimator and cost a debugging round.
            self.bus.log.emit(
                "info", "imu_dr: demo source — using an IDENTITY calibration "
                        f"({cpath!r} is for the real camera and would inject "
                        f"its bias into a synthetic sensor)")
            cpath = None
        if cpath and Path(cpath).exists():
            try:
                calib = ImuCalibration.from_json(cpath)
            except (OSError, ValueError, KeyError) as e:         # noqa: BLE001
                self.bus.log.emit("error", f"imu_dr: calibration {cpath} "
                                           f"unreadable ({e}) — running RAW")
        elif str(getattr(self.opts, "source", "")) != "demo":
            self.bus.log.emit(
                "warn", f"imu_dr: no calibration at {cpath!r} — running RAW. "
                        f"This camera's accelerometer carries a measured "
                        f"1.8 m/s^2 bias (90 m of drift at 10 s) and its "
                        f"mounting rotation is unknown; expect the estimate "
                        f"to be meaningless until "
                        f"`python -m rov_gui.tools.calib_c3_imu` has run.")
        self.dr = ImuDeadReckoner(
            calib=calib,
            attitude=str(d["attitude"]),
            ahrs_tau_s=float(d["ahrs_tau_s"]),
            accel_trust_m_s2=float(d["accel_trust_m_s2"]),
            z_source=str(d["z_source"]),
            max_dt_s=float(d["max_dt_s"]),
            static_window_s=float(d["static_window_s"]),
            gyro_static_std_max=float(d["gyro_static_std_max"]),
            gyro_bias_sem_max=float(d["gyro_bias_sem_max"]),
            accel_static_sd_max=float(d["accel_static_sd_max"]),
            stale_s=float(self.cfg.engage["tag_stale_s"]))

        # Typing `--imu-dr control` IS the explicit intent — asking for a
        # second flag on top of it was friction with no safety in it. What
        # the gate has to stop is a CONFIG FILE arming a closed loop on dead
        # reckoning because someone left it that way: that is the case with
        # no human in the moment, so `imu_dr.mode: control` in YAML still
        # needs --imu-dr-control (or --imu-dr control) on the command line.
        asked_cli = str(getattr(self.opts, "imu_dr", "") or "") == "control"
        self.dr_control = (str(d["mode"]) == "control"
                           and (asked_cli
                                or bool(getattr(self.opts, "imu_dr_control",
                                                False))))
        if str(d["mode"]) == "control" and not self.dr_control:
            self.bus.log.emit(
                "warn", "imu_dr: hw_mpc.yaml asks for mode 'control' but the "
                        "command line did not — running in SHADOW. Pass "
                        "--imu-dr control to fly on the estimate.")
        self._event(
            f"imu_dr {'CONTROL' if self.dr_control else 'shadow'}: "
            f"{d['source']}/{d['attitude']}"
            + (f" tau={d['ahrs_tau_s']:g}s" if d["attitude"] == "ahrs" else "")
            + f", z={d['z_source']}"
            + (f", calib {calib.sha1[:8]}" if calib.sha1 else ", NO CALIB"),
            defer=True)

    # ------------------------------------------------------------ object nav
    def _setup_object_nav(self) -> None:
        """Build the object anchor, but only with ``--pose`` (object_nav.py).

        Without the tracker there is nothing to anchor, so ``self._obj`` stays
        None and every follow gate reads that as "this station cannot follow".

        The one check worth making here rather than at arm time is WHICH
        CAMERA localizes. The extrinsic only cancels when the object pose and
        the tag fix come from one frame of one camera; if ``hw_nav.yaml``
        points the localizer at the ROV RGB while the object rides the C3, the
        composition is not merely inexact, it is meaningless — and it would
        still draw a confident diamond. So it is refused loudly and nothing is
        composed at all.
        """
        self._obj, self._obj_ext, self._obj_note = None, None, ""
        self._obj_src_ok = True
        if not bool(getattr(self.opts, "pose", False)):
            return
        try:
            cfg = ON.resolve(getattr(self.cfg, "object_nav", None))
        except ValueError as e:                                  # noqa: BLE001
            self.bus.log.emit("error", f"object_nav: {e} — DISABLED")
            return
        self._obj = ON.ObjectAnchor(cfg)
        want = cfg["nav_source_required"]
        self._obj_ext = self.nav_cfg.R_t_frd_cam(want)
        have = str(self.nav_cfg.nav_source)
        self._obj_src_ok = (have == want)
        if not self._obj_src_ok:
            # The anchor is deliberately left BUILT so `on_pose` keeps
            # publishing an ObjectFix that says why there is no position.
            # Tearing it down instead would leave the panel silent, which
            # reads as "no tracker" rather than "misconfigured".
            self._obj_note = (f"nav_source is {have!r} but the object rides "
                              f"{want!r} — the camera extrinsic cannot cancel")
            self.bus.log.emit(
                "error", f"object_nav: {self._obj_note}. No object position "
                         f"will be published and `follow` will refuse. Set "
                         f"hw_nav.yaml nav_source: {want}.")
            return
        self._event(f"object_nav ON: yaw_axis={cfg['yaw_axis']}, "
                    f"pair<={cfg['pair_tol_s'] * 1e3:.0f}ms, "
                    f"range {cfg['min_distance_m']:.2f}-"
                    f"{cfg['max_distance_m']:.2f} m, "
                    f"excursion {cfg['max_excursion_m']:.2f} m", defer=True)

    @Slot(object)
    def on_pose(self, track) -> None:
        """One object-tracker result -> one :class:`ObjectFix` in the MAP frame.

        Runs at the tracker's 10 Hz publish rate and NOT on the control tick:
        it is the arrival of a pose that makes an object position possible, and
        composing on the tick instead would pair whatever fix happened to be
        newest rather than the one from the pose's own frame.

        Every path here emits — including the unhappy ones. A pipeline that is
        registering, an object out of range and a pose with no fix to pair
        against are three different things the operator has to be able to tell
        apart, and silence tells them none of it.
        """
        if self._obj is None or track is None or self.nav_cfg is None:
            return
        t = now()
        state = str(getattr(track, "state", "") or "")
        T = getattr(track, "T_cam_obj", None)
        t_cap = float(getattr(track, "t_capture", 0.0) or 0.0)
        if not self._obj_src_ok:
            self._emit_object(t, t_cap, state, self._obj_note, None)
            return
        if state != ON.TRACKING or T is None:
            # Not a lock. `update` records that (so `state` can leave LIVE)
            # without touching the estimate.
            rec = self._obj.update(None, None, t_cap, t, 0.0, state)
            self._emit_object(t, t_cap, state, rec.get("why", ""), None)
            return
        entry, dt, exact = ON.pick_fix(self._fix_hist, t_cap,
                                       float(self._obj.cfg["pair_tol_s"]))
        if entry is None:
            # A pose with no tag fix from its own frame. The estimate is left
            # alone and simply ages — subtracting two drifting quantities is
            # the worst thing this could do instead.
            self._emit_object(t, t_cap, state,
                              ("no tag fix within "
                               f"{dt * 1e3:.0f} ms of this frame"), None)
            return
        R_bc, t_bc = self._obj_ext
        p_map, R_map_obj = ON.compose_map_pose(T, entry[1], entry[2],
                                               R_bc, t_bc)
        d = float(math.sqrt(float(T[3]) ** 2 + float(T[7]) ** 2
                            + float(T[11]) ** 2))
        self._obj.note_pair(exact)
        rec = self._obj.update(p_map, R_map_obj, t_cap, t, d, state)
        self._emit_object(t, t_cap, state, rec.get("why", ""), (dt, exact))

    def _emit_object(self, t_now, t_cap, pose_state, note, pair) -> None:
        """Publish what the anchor believes, MAP frame, honestly flagged."""
        a = self._obj
        st = a.predict(t_now)
        p = st["p"]
        fx = ObjectFix(
            t_capture=float(t_cap),
            ok=bool(st["ok"] and p is not None),
            state=str(st["state"]),
            p_map=(None if p is None else tuple(float(v) for v in p)),
            yaw_map=(None if st["yaw"] is None else float(st["yaw"])),
            R_map_obj=(None if a.R is None else
                       tuple(float(v) for v in np.asarray(a.R).ravel())),
            v_map=tuple(float(v) for v in st["v"]),
            r_map=float(st["r"]),
            distance_m=st["distance_m"],
            age_s=st["age_s"],
            extrapolated_s=float(st["extrapolated_s"]),
            pair_dt_ms=(None if pair is None else float(pair[0]) * 1e3),
            pair_exact=bool(pair is not None and pair[1]),
            # An UNLOCKED "auto" anchor also has axis None, and reporting that
            # as "none" is the same conflation that made yaw_axis:"none" fly
            # axis-pinned — the panel would say the object's heading is being
            # ignored while "auto" is one observation away from using it.
            yaw_axis=("xyz"[a.axis] if a.axis is not None
                      else ("none" if a.cfg["yaw_axis"] == "none"
                            else "auto (unlocked)")),
            pose_state=str(pose_state),
            n_obs=int(a.n_obs), n_reject=int(a.n_reject),
            note=str(note or st["note"] or ""),
            conn={ON.LIVE: Conn.ONLINE,
                  ON.STALE: Conn.DEGRADED,
                  ON.LOST: Conn.DEGRADED}.get(st["state"], Conn.OFFLINE),
            stamp=float(t_now))
        self._obj_last = fx
        self.bus.object_fix.emit(fx)

    def _setup_station_bridge(self) -> None:
        """Build the ladder and its own estimator (station_bridge.py).

        The estimator differs from the ``imu_dr`` experiment's in two ways,
        both deliberate:
          * ``attitude="vehicle"`` — roll/pitch from the autopilot's own AHRS.
            The experiment refuses that on purpose (it is measuring what the
            camera IMU alone can do); a SAFETY bridge has no such purity
            requirement and an absolute, drift-free attitude is strictly
            better for the job.
          * it is re-anchored on EVERY fresh fix, so a dropout always starts
            from the last tag pose with zero accumulated error.
        A raw (uncalibrated) accelerometer carries a measured 1.8 m/s^2 bias
        — 8 m of drift in 3 s — so without a calibration file the horizontal
        channel FALLS BACK to freezing x/y at the last fix rather than
        integrating a sensor that would take the vehicle across the pool.
        """
        try:
            cfg = SB.resolve(getattr(self.cfg, "station_bridge", None))
        except ValueError as e:                                  # noqa: BLE001
            self.bus.log.emit("error", f"station_bridge: {e} — DISABLED")
            self._bridge, self._bridge_dr = None, None
            return
        self._bridge = SB.StationBridge(cfg)
        if not cfg["enabled"]:
            self._bridge_dr = None
            self._bridge_note = ("station_bridge: OFF — a tag dropout during "
                                 "a station hold disengages, as before")
            return
        calib = ImuCalibration.identity()
        cpath = self.cfg.imu_dr.get("calibration")
        demo = str(getattr(self.opts, "source", "")) == "demo"
        if cpath and not demo and Path(cpath).exists():
            try:
                calib = ImuCalibration.from_json(cpath)
            except (OSError, ValueError, KeyError) as e:         # noqa: BLE001
                self.bus.log.emit("error", f"station_bridge: calibration "
                                           f"{cpath} unreadable ({e})")
        want = cfg["xy_source"]
        have_calib = bool(calib.sha1) or demo
        self._bridge_xy = ("hold" if want == "hold"
                           else "imu" if want == "imu"
                           else ("imu" if have_calib else "hold"))
        if want == "imu" and not have_calib:
            self.bus.log.emit(
                "warn", "station_bridge: xy_source 'imu' was asked for with "
                        "NO calibration — this accelerometer's measured "
                        "1.8 m/s^2 bias is 8 m of drift in 3 s. Flying it "
                        "because a config file said so.")
        self._bridge_dr = ImuDeadReckoner(
            calib=calib, attitude="vehicle",
            ahrs_tau_s=float(self.cfg.imu_dr["ahrs_tau_s"]),
            accel_trust_m_s2=float(self.cfg.imu_dr["accel_trust_m_s2"]),
            z_source="pressure",
            max_dt_s=float(self.cfg.imu_dr["max_dt_s"]),
            stale_s=float(self.cfg.engage["imu_stale_s"]))
        why_xy = (f"calib {calib.sha1[:8]}" if calib.sha1
                  else "demo: ideal sensor" if demo
                  else "NO CALIB — x/y frozen, not integrated")
        self._bridge_note = (
            f"station_bridge ON: hold every axis {cfg['imu_hold_s']:.1f}s "
            f"(xy={self._bridge_xy}, {why_xy}), then release x/y/yaw and keep "
            f"depth+attitude")

    def _station_bridge(self, meas, health, t, dt):
        """Carry a STATION hold across a tag dropout. Returns the state the
        rest of the tick should use, or ``meas`` unchanged.

        Substituting into ``meas`` rather than into ``meas_ctrl`` is the whole
        point: ``_runtime_fault`` reads ``meas``, so a bridged tick no longer
        trips the stale-fix interlock — while every OTHER interlock (disarm,
        flight mode, telemetry, solver) keeps running exactly as before,
        because they are evaluated on the same non-None state."""
        br = self._bridge
        if br is None or not br.enabled:
            return meas
        station = bool(self.engaged and self.station is not None)
        if meas is not None:
            rec = br.note_fix(t)
            if rec is not None:
                self._note_bridge_recovery(rec, meas)
            self._bridge_anchor = np.asarray(meas["eta"], float).copy()
            self._bridge_yaw = float(meas["eta"][5])
            self._bridge_z = float(meas["eta"][2])
            self._bridge_vz = float(
                (rot_zyx(*meas["eta"][3:6])
                 @ np.asarray(meas["nu"], float)[:3])[2])
            if self._bridge_dr is not None:
                self._drain_dr()
                self._bridge_dr.note_depth(*self._dr_depth())
                # nu is BODY frame; the anchor wants it in the world.
                v_w = (rot_zyx(*meas["eta"][3:6])
                       @ np.asarray(meas["nu"], float)[:3])
                self._bridge_dr.anchor(meas["eta"], nu_world_ned=v_w, t=t,
                                       zero_velocity=False)
            return meas
        if not station or self._bridge_anchor is None:
            return meas
        if not SB.is_bridgeable(health.get("why", "")):
            # imu stale / pressure stale / no autopilot: the very sensors the
            # bridge would fly on. Those still disengage.
            return meas
        m = None
        if self._bridge_xy == "imu" and self._bridge_dr is not None:
            self._drain_dr()
            self._bridge_dr.note_depth(*self._dr_depth())
            st = self._bridge_dr.state(t, imu_vehicle=self.imu)
            self._bridge_dr.note_tick(dt)
            if st["ok"]:
                m = st["meas"]
            elif not self._bridge_dr_warned:
                # The C3 sample stream died. Say so ONCE and drop to the
                # freeze, which needs nothing from that camera at all.
                self._bridge_dr_warned = True
                self.bus.log.emit(
                    "warn", f"station_bridge: no IMU estimate ({st['why']}) "
                            f"— falling back to freezing x/y")
        if m is None:
            # THE FREEZE. Deliberately reachable with no C3 IMU in the system
            # at all: x/y from the last fix, z from the barometer, roll/pitch
            # from the autopilot, yaw from its gyro. Nothing here is double
            # integrated, so it does not degrade with time the way the accel
            # path does — it is simply blind to real horizontal motion.
            m = self._bridge_meas_freeze(t, dt)
        if m is None:
            # Not even the freeze is available (autopilot or barometer gone).
            # Fall through to the normal interlock: a bridge that has itself
            # gone blind must never look like a hold.
            br.reset()
            return meas
        tier = br.note_lost(dt)
        self.phase_detail = br.detail()
        if tier == SB.TIER_COAST and br.n_coast and int(
                br.elapsed * self.cfg.ctrl_hz) % int(
                max(1, 5 * self.cfg.ctrl_hz)) == 0:
            self.bus.log.emit("warn", f"mpc: {br.detail()} — nothing is "
                                      f"holding position; take manual "
                                      f"control if it is drifting")
        return m

    def _bridge_meas_freeze(self, t, dt):
        """The no-accelerometer bridge state, or None if even this is gone.

            x, y        the last tag fix — for a vehicle that was station
                        keeping, the maximum-likelihood estimate absent a
                        trustworthy accelerometer, and a far better one than
                        integrating a RAW C3 (measured 1.8 m/s^2 bias = 8 m
                        in 3 s)
            z, w        barometer, absolute, low-passed derivative for heave
            roll, pitch autopilot AHRS, absolute
            yaw         last fix + integrated autopilot gyro
            u, v        zero — this state is BLIND to horizontal motion, and
                        pretending otherwise is what the accel path is for

        The freshness of both sources is re-checked here rather than trusted:
        the assembler returns "tag fix stale" BEFORE it looks at the IMU, so a
        bridgeable fault does not prove the autopilot is still talking.
        """
        imu = self.imu
        if imu is None or imu.roll is None or imu.p is None:
            return None
        t_att = imu.t_att if imu.t_att is not None else imu.stamp
        if t_att is None or t - float(t_att) > float(
                self.cfg.engage["imu_stale_s"]):
            return None
        z, t_baro = self._dr_depth()
        if z is None or t_baro is None or t - float(t_baro) > 1.5:
            return None
        a = float(self.cfg.vel_lp_alpha)
        vz = (z - self._bridge_z) / max(1e-3, dt)
        self._bridge_vz = (1.0 - a) * self._bridge_vz + a * vz
        self._bridge_z = float(z)
        self._bridge_yaw += float(imu.r) * dt
        eta = np.array([float(self._bridge_anchor[0]),
                        float(self._bridge_anchor[1]), float(z),
                        float(imu.roll), float(imu.pitch),
                        float(self._bridge_yaw)])
        R = rot_zyx(eta[3], eta[4], eta[5])
        nu = np.concatenate([R.T @ np.array([0.0, 0.0, self._bridge_vz]),
                             np.array([float(imu.p), float(imu.q),
                                       float(imu.r)])])
        return {"eta": eta, "nu": nu, "nudot": np.zeros(6)}

    def _note_bridge_recovery(self, rec: dict, meas) -> None:
        """The tag came back. Log how far the bridge had drifted — this is the
        one number that turns the [예측] IMU budget into a measurement, and it
        is free every time a dropout ends."""
        err = None
        if self._bridge_dr is not None and self._bridge_dr.anchored:
            p = np.asarray(self._bridge_dr.p, float)
            err = float(math.hypot(p[0] - float(meas["eta"][0]),
                                   p[1] - float(meas["eta"][1])))
            rec["err_m"] = err
        self.phase_detail = ""
        msg = (f"station_bridge: fix back after {rec['elapsed']:.2f}s "
               f"(tier {rec['tier']}"
               + (f", IMU was {err * 100:.1f} cm off" if err is not None
                  else "") + ")")
        self._event(msg)
        self.bus.log.emit("info", f"mpc: {msg}")

    def _coast_retarget(self, meas_ctrl) -> None:
        """In the coast tier the horizontal setpoint is moved ONTO the vehicle
        every tick, so the controller has nothing to ask for horizontally and
        no integrator winds against an error it is not allowed to correct.
        Depth stays at the station's depth — that is the axis still being
        held."""
        st = self.station or {}
        z = float(st.get("depth_ned", float(meas_ctrl["eta"][2])))
        self.ctrl.set_target_ned((float(meas_ctrl["eta"][0]),
                                  float(meas_ctrl["eta"][1]), z),
                                 float(meas_ctrl["eta"][5]))

    @Slot(object)
    def on_camera_imu(self, batch) -> None:
        """One C3 drain. Queued, integrated on the next tick.

        The queue is BOUNDED, and that is a decision rather than a default: if
        this thread stalls, an unbounded queue turns the stall into a memory
        leak plus a burst of stale integration when it recovers. Dropping the
        oldest keeps the estimate current and makes the loss countable, which
        ``dr_note`` then reports rather than hiding.
        """
        if batch is None or batch.n <= 0:
            return
        if self.dr is None and self._bridge_dr is None:
            return
        if len(self._dr_q) == self._dr_q.maxlen:
            self._dr_overflow += 1
        self._dr_q.append(batch)

    def _drain_dr(self) -> None:
        """One queue, up to two consumers. The experiment's estimator and the
        station bridge integrate the SAME samples — splitting the stream would
        make their disagreement a plumbing artefact instead of a measurement."""
        while self._dr_q:
            b = self._dr_q.popleft()
            if self.dr is not None:
                self.dr.integrate(b.samples, dropped=int(b.dropped))
            if self._bridge_dr is not None:
                self._bridge_dr.integrate(b.samples, dropped=int(b.dropped))

    def _dr_depth(self) -> tuple:
        """Barometer depth in the DATUM frame, for the dead reckoner's z.

        Two conversions, and both matter. The SAME session offset the state
        assembler anchored (``asm.z_offset``), so the two estimates share a
        world rather than sitting a constant apart; and the SAME mission datum
        the DR was anchored in, since everything downstream of engage lives
        there. Getting either wrong shows up as a fixed depth error that looks
        like sensor bias.
        """
        imu, off = self.imu, (self.asm.z_offset if self.asm else None)
        if imu is None or imu.depth_m is None or off is None:
            return None, None
        z = float(imu.depth_m) + float(off)
        if self._datum is not None:
            z -= float(self._datum["p0"][2])
        return z, imu.t_baro

    def _axis_cap(self) -> float:
        """The authority ceiling in force right now.

        Flying closed-loop on dead reckoning may want a lower one than a
        tag-guided run, so it is separately settable — but changing it also
        changes what `note_applied` feeds the EAOB, which means a DR run and a
        tag run at different caps are not comparable AS CONTROLLER runs. The
        value actually used goes into the run meta for that reason.
        """
        cap = self.cfg.axis_cap
        if self.dr_control:
            alt = self.cfg.imu_dr.get("axis_cap_dr")
            if alt is not None:
                cap = float(alt)
        # ...and the station bridge may derate while it is carrying a dropout.
        # Shipped as null (operator decision 2026-08-18: keep full authority,
        # because a derated bridge recovers more slowly from the very kick
        # that caused the dropout).
        if self._bridge is not None and self._bridge.active:
            alt = self._bridge.cfg.get("axis_cap")
            if alt is not None:
                cap = float(alt)
        return cap

    # ---------------------------------------------------------------- inputs
    @Slot(object)
    def on_nav_fix(self, fix) -> None:
        if getattr(fix, "ok", False):
            self.fix = fix
            if self._obj is not None:
                # RAW and undatumized, keyed by the frame's CAPTURE stamp —
                # the only form in which the camera extrinsic still cancels
                # (object_nav.py). Kept only when there is an object tracker
                # to pair it with.
                self._fix_hist.append(
                    (float(fix.t_capture),
                     np.asarray(fix.p_ned, float).reshape(3),
                     np.asarray(fix.R_ned_body, float).reshape(3, 3)))

    @Slot(object)
    def on_vehicle_imu(self, imu) -> None:
        self.imu = imu
        # A short history of the autopilot's ATTITUDE message, kept ONLY for
        # the dead reckoner's settle-window calibration: the rates so the gyro
        # bias can be differenced against the vehicle's real rotation, and the
        # roll/pitch so the accel offset is measured per sample instead of
        # against one frozen attitude. 20 Hz x 60 s covers any settle.
        if imu is not None and imu.p is not None and imu.roll is not None:
            t = imu.t_att if imu.t_att is not None else imu.stamp
            self._vimu_hist.append((float(t), float(imu.p), float(imu.q),
                                    float(imu.r), float(imu.roll),
                                    float(imu.pitch)))

    @Slot(object)
    def on_telemetry(self, tel) -> None:
        self.tel = tel

    @Slot(object)
    def on_thrusters(self, thr) -> None:
        self.thr = thr

    def _pwm_dev_us(self) -> float | None:
        """Mean |PWM - 1500| over the thrusters the vehicle reports — the
        actuation chain's own answer to "did anything spin"."""
        t = self.thr
        if t is None or not t.pwm_us:
            return None
        vals = [abs(p - 1500) for p in t.pwm_us if p is not None]
        return (sum(vals) / len(vals)) if vals else None

    @Slot(bool)
    def on_enable(self, on: bool) -> None:
        self.cmd_enabled = bool(on)
        if not on and self.engaged:
            self.disengage("COMMAND ENABLE off")

    # mpc_tuned / dobmpc_tuned are the SAME solver object as mpc / dobmpc —
    # the suffix rotates the position weight into the path frame at run time
    # (control/path_cost.py), so there is no third build and the A/B differs
    # in exactly one thing.
    MODES = ("mpc", "dobmpc", "mpc_tuned", "dobmpc_tuned",
             "mpcc", "dobmpcc", "pid")

    def _ctrl_for(self, mode: str):
        """The controller object a mode name selects, or None if unavailable.

        ``mpc``/``dobmpc`` share ONE tracking solver and ``mpcc``/``dobmpcc``
        share ONE contouring solver — in each pair the prefix only decides
        whether the EAOB's w_hat reaches the solver, exactly as in the sim."""
        if mode == "pid":
            return self._pid
        if mode in ("mpcc", "dobmpcc"):
            return self._mpcc_ctrl
        return self._mpc_ctrl

    @Slot(str)
    def set_mode(self, mode: str) -> None:
        mode = str(mode).lower()
        if mode not in self.MODES:
            return
        if self.engaged:
            self.bus.log.emit("warn", "mpc: mode change refused while engaged")
            return
        if self.cfg is None or not self._ready:
            return
        ctrl = self._ctrl_for(mode)
        if ctrl is None:
            self.bus.log.emit("error", f"mpc: mode {mode} unavailable "
                                       f"({self._setup_error or 'not built'})"
                                       f" — staying on {self.cfg.mode}")
            return
        self.ctrl = ctrl
        if mode != "pid" and mode != getattr(ctrl, "mode", mode):
            ctrl.mode = mode
        self.ctrl.reset()
        self._path_cursor = None
        self._path_err = (None, None)
        self._path_depth = 0.0
        self._path_yaw_fixed = 0.0
        self._path_heading_follow = False
        self.cfg.mode = mode
        self.bus.log.emit("info", f"mpc: mode = {mode} "
                                  f"({self.ctrl.solver_kind})")

    @Slot(object)
    def set_scenario(self, d: dict) -> None:
        if isinstance(d, dict):
            self._scenario_override = dict(d)

    # ------------------------------------------------------------ engagement
    @Slot(bool)
    def set_engaged(self, on: bool) -> None:
        if not on:
            self.disengage("released")
            return
        if self.engaged:
            return
        why = self._engage_refusal()
        if why:
            self.reason = f"engage refused: {why}"
            self.bus.log.emit("warn", f"mpc: {self.reason}")
            self._log_event(self.reason)
            return
        self.asm.reset()
        self.asm.calibrate_z_offset(self.fix, self.imu)
        meas, h = self.asm.step(self.fix, self.imu, now(), 1.0 / self.cfg.ctrl_hz)
        if meas is None:
            self.reason = f"engage refused: {h['why']}"
            self.bus.log.emit("warn", f"mpc: {self.reason}")
            self._log_event(self.reason)
            return
        # THE datum moment: this pose, in the tag frame, is (0,0,0)/yaw 0 for
        # the whole run. The EAOB and NMPC are (re)built in the datum frame
        # from their first tick, so nothing ever jumps mid-engagement.
        eta_tag = meas["eta"]
        self._datum = {"p0": eta_tag[:3].copy(), "yaw0": float(eta_tag[5]),
                       "Rz": rot_zyx(0.0, 0.0, -float(eta_tag[5]))}
        meas["eta"] = self._datumize(eta_tag)
        self.ctrl.reset()
        if self._bridge is not None:
            self._bridge.reset()
        self._bridge_anchor = None
        self.ctrl.set_target_ned(meas["eta"][:3], meas["eta"][5])
        self._eta = meas["eta"]
        self.engaged = True
        self.traj_on = False
        # 3 of 4 (see MpcWorker.follow). Belt and braces — disengage already
        # cleared both — but an engagement that inherited a mission from the
        # previous one is exactly the kind of state leak that is invisible
        # until it is not.
        self.station = None
        self.follow = None
        self._clear_replay()
        self._replay_last = None       # a fresh engagement is a fresh record
        self._t0_traj = None
        self._path_cursor = None
        self._path_err = (None, None)
        self._path_depth = 0.0
        self._path_yaw_fixed = 0.0
        self._path_heading_follow = False
        self._t_engage = now()
        self._warmup_left = max(1, int(round(
            float(self.cfg.engage["warmup_s"]) * self.cfg.ctrl_hz)))
        self._fail_streak = 0
        self._nfail_prev = 0
        self.reason = "engaged (DP hold)"
        # ONE announcement, not two. This used to fire a short mission event
        # ("ENGAGED (dobmpc)") AND a warn log line carrying the hold point, so
        # the operator's log showed the same event twice, on two lines, in two
        # styles (2026-08-23). The mission event now carries the hold point and
        # the log copy drops to `debug` — still on stdout, no longer a second
        # line in the panel.
        self._event(f"ENGAGED ({self.cfg.mode}) — holding "
                    f"({meas['eta'][0]:+.2f}, {meas['eta'][1]:+.2f}, "
                    f"{meas['eta'][2]:+.2f}) NED")
        self._log_event(f"ENGAGED ({self.cfg.mode}) datum p0="
                        f"{[round(float(v), 3) for v in self._datum['p0']]} "
                        f"yaw0 {math.degrees(self._datum['yaw0']):.1f} deg")
        self.bus.log.emit(
            "debug", f"mpc: ENGAGED ({self.cfg.mode}) — holding "
                     f"({meas['eta'][0]:+.2f}, {meas['eta'][1]:+.2f}, "
                     f"{meas['eta'][2]:+.2f}) NED. START TRAJ begins the "
                     f"square.")
        if self._csv is None:
            # Into the run folder the screen recording and the nav recording
            # are already writing to (rov_gui/runstore.py): three writers, one
            # dated folder, no handshake between them. PINNED here for the life
            # of the engagement — see _run_dir().
            self._run_dir_pin = runstore.run_dir(self.cfg.log_dir)
            stem = self._run_dir_pin / f"mpc_{runstore.stamp('%H%M%S')}"
            self._open_csv(stem, auto=True)

    def _event(self, msg: str, defer: bool = False) -> None:
        """A short, timestamped mission line: to the on-screen log AND to
        events.log, so the two can never tell different stories.

        ``defer`` holds the FILE half back until there is a run folder worth
        opening (see :meth:`_log_event`); the screen half is never deferred."""
        self.bus.mpc_event.emit(msg)
        self._log_event(msg, defer=defer)

    def _run_dir(self):
        """The folder this engagement's files belong in.

        PINNED while a CSV is open. Resolving it per event instead would put a
        refusal at 18:44 into `.../1844/events.log` while the CSV it explains
        sits in `.../1841/` — and, because run_dir creates as it goes, would
        leave a stray minute folder holding one line. Outside an engagement
        there is nothing to pin to, so a lone refusal opens (or joins) the
        folder for the moment it happened, which is the right answer for it.
        """
        return self._run_dir_pin or runstore.run_dir(self.cfg.log_dir)

    def _log_event(self, msg: str, defer: bool = False) -> None:
        """Every refusal/engage/disengage into ONE persistent file — a pool
        session run without a terminal must still leave its story behind
        (2026-08-12: the 19:29 refusals lived only on a screen recording).

        ``defer=True`` is for lines that describe the SESSION rather than a run
        — today just the build fingerprint from :meth:`setup`. They are held in
        memory and written as the header of each run folder's events.log the
        first time that folder is opened for a real line. That is what stops a
        station being launched and never flown from leaving a folder behind
        (seven of them on 2026-08-14), while still putting the fingerprint at
        the top of every run that does happen — including the second and third
        run of the same session, which is why the banner is kept rather than
        flushed once.
        """
        if self.cfg is None:
            return
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        if defer:
            self._session_banner.append(line)
            return
        try:
            p = self._run_dir() / "events.log"
            header = (self._session_banner
                      if p != self._events_path else ())
            with open(p, "a", encoding="utf-8") as f:
                for h in header:
                    f.write(h + "\n")
                f.write(line + "\n")
            # Only after the write succeeded — a failed open must not silently
            # consume the banner for every folder that follows.
            self._events_path = p
        except OSError:
            pass

    def _engage_refusal(self) -> str:
        e = self.cfg.engage if self.cfg else {}
        if not self._ready:
            return "worker not ready"
        if self.ctrl is None:
            return f"no controller ({self._setup_error})"
        if not self.ctrl.realtime_ok:
            return (f"solver not real-time ({self.ctrl.solver_kind}, "
                    f"probe {self.ctrl.probe_ms} ms)")
        if not self.cmd_enabled:
            return "COMMAND ENABLE is off"
        if self.sink_status_fn is not None:
            conn, note = self.sink_status_fn()
            if conn is Conn.FAULT:
                # The sink itself says it cannot deliver (no peer on the
                # command port, or SYSID_MYGCS mismatch): engaging would
                # compute a mission nothing receives.
                return f"command sink fault: {note}"
        tel = self.tel
        if tel is None or tel.armed is not True:
            return "vehicle not armed (or no telemetry)"
        # A stale snapshot still says armed=True forever; silence must not
        # count as consent (bus.py's Freshness rule, applied here too).
        if now() - float(tel.stamp) > 2.0:
            return "vehicle telemetry stale"
        want = str(e.get("require_mode", "MANUAL")).upper()
        mode = (tel.mode or "").upper()
        if want and not mode.startswith(want):
            return f"flight mode is {tel.mode or '?'}, need {want}"
        t = now()
        if self.fix is None or not self.fix.ok:
            return "no tag fix"
        if t - float(self.fix.t_capture) > float(e["tag_stale_s"]):
            return "tag fix stale"
        if self.imu is None or self.imu.roll is None:
            return "no vehicle attitude"
        # No position gate here, and since 2026-08-14 none anywhere: the
        # geofence that used to gate START and abort at runtime was removed.
        return ""

    def _datumize(self, eta):
        """TAG-frame eta -> datum-frame eta (horizontal isometry: translate
        by the engage position, rotate by the engage yaw). Body-frame
        quantities (nu, nudot, wrench) are invariant; roll/pitch unchanged."""
        d = self._datum
        if d is None:
            return eta
        out = eta.copy()
        out[0:3] = d["Rz"] @ (eta[0:3] - d["p0"])
        out[5] = math.atan2(math.sin(eta[5] - d["yaw0"]),
                            math.cos(eta[5] - d["yaw0"]))
        return out

    @Slot()
    def start_mission(self) -> None:
        """The one-button flow (the panel's START): engage if needed, and the
        square starts BY ITSELF the moment the warm-up completes. The CSV
        opens at engage as always, so one press = fly the square + log it.
        Every gate still applies — this is a convenience ordering of the same
        two steps, not a bypass."""
        self._auto_traj = True
        if not self.engaged:
            self.set_engaged(True)
            if not self.engaged:
                self._auto_traj = False       # refusal reason already logged
        elif self._warmup_left <= 0 and not self.traj_on:
            self._auto_traj = False
            self.set_traj(True)

    @Slot(bool)
    def set_traj(self, on: bool) -> None:
        if not on:
            if self.ctrl is not None and self._eta is not None:
                # Normal completion holds the captured final vertex. A manual
                # stop/timeout holds the measured pose, so STOP never causes a
                # last-moment move toward an unfinished target.
                #
                # NOT `if self.traj_on`: a FOLLOW runs with traj_on False
                # (`_arm_follow`), so that guard skipped exactly the mission
                # whose reference is MOVING. Clearing `self.follow` below then
                # left the controller holding the last follow target WITH its
                # velocity feedforward, and `_xref_ned` rebuilds the whole
                # 61-stage horizon as p_ref + v_ref*k*dt every tick — so STOP
                # TRAJ read as "stopped" on the panel while the vehicle kept
                # being commanded along a velocity ray, with nothing left to
                # walk or clamp it. `_tick_follow`'s own excursion warning
                # tells the operator to press this button.
                # A stationary target is the right answer for every mission
                # kind; station/path targets already carried v = 0.
                self.ctrl.set_target_ned(self._eta[:3], self._eta[5])
            self.traj_on = False
            self._t0_traj = None
            self._path_cursor = None
            self._path_err = (None, None)
            self._approach = None
            self.station = None
            self.follow = None            # 1 of 4: see MpcWorker.follow
            self._clear_replay()
            if self._bridge is not None:
                self._bridge.reset()      # the ladder is a STATION feature
            self.phase, self.phase_detail = "", ""
            self.reason = "trajectory stopped (DP hold)"
            self.bus.log.emit("info", "mpc: trajectory stopped — DP hold")
            return
        if not self.engaged or self.ctrl is None:
            self._refuse("not engaged")
            return
        if self._warmup_left > 0:
            self._refuse(f"warming up "
                         f"({self._warmup_left / self.cfg.ctrl_hz:.1f}s left)")
            return
        if self._eta is None:
            return
        sq = dict(self.cfg.square)
        sq.update(self._scenario_override)
        shape = str(sq.get("shape", "square")).lower()
        if shape not in SHAPES:
            # MpcConfig.load validates the FILE; a panel scenario override can
            # still carry anything. Same rule either way: never silently fly a
            # rectangle because a name was misspelled (`_arm_path`'s else
            # branch is "square", which is what made that failure silent).
            self._refuse(f"unknown shape {sq.get('shape')!r}; "
                         f"known: {list(SHAPES)}")
            return
        if shape == "follow":
            why = self._follow_refusal()
            if why:
                self._refuse(why)
                return
            # NO approach and NO settle, and that is a PROPERTY rather than an
            # omission: the offset is captured from the vehicle's CURRENT
            # pose, so the target the moment it arms is where the vehicle
            # already is. There is nowhere to fly to first, and pressing START
            # must not move the vehicle at all.
            self._event("FOLLOW requested")
            self._arm_path(sq, shape,
                           (float(self._eta[0]), float(self._eta[1])),
                           float(self._eta[5]), float(self._eta[2]))
            return
        if shape == "replay":
            why = self._replay_refusal()
            if why:
                self._refuse(why)
                return
            # Like FOLLOW: no approach and no settle, as a PROPERTY. The
            # recorded track is re-anchored at the vehicle's CURRENT pose, so
            # the reference the moment it arms is where the vehicle already
            # is and START must not move it anywhere first.
            self._event("REPLAY requested")
            self._arm_path(sq, shape,
                           (float(self._eta[0]), float(self._eta[1])),
                           float(self._eta[5]), float(self._eta[2]))
            return
        origin_xy = self._mission_origin(sq)
        if origin_xy is None:
            return                            # reason already logged
        yaw_fixed = self._mission_yaw(sq)
        depth = sq.get("depth_ned")
        depth = float(self._eta[2]) if depth is None else float(depth)
        # NO GEOFENCE CHECK HERE — the fence was removed entirely on
        # 2026-08-14 at the operator's explicit request (it was refusing
        # placements they wanted to fly, and the box on the plot was reading
        # as clutter). What used to happen here: every corner of the placed
        # path was tested against hw_nav.yaml's geofence_ned box and START was
        # refused if any of them fell outside.
        #
        # What still stops the vehicle, so this is on the record: E-STOP (Esc
        # / two buttons), DISENGAGE, the command sink's 500 ms deadman, the
        # arm/mode/tag-fix/telemetry-staleness engage gates, the runtime
        # tag-loss and disarm aborts, and the per-axis authority caps. What
        # does NOT: anything that knows where the pool WALL is. A mission
        # placed past the wall will now be flown at it.
        self._event(
            f"{shape.upper()} requested"
            + (f" @ tag {sq['origin_tag']}" if sq.get("origin_tag") else "")
            + (f", {float(sq.get('length', 0)):.2f} m "
               f"{float(sq.get('dir_deg', 90)):.0f} deg x{sq.get('laps')}"
               if shape == "line" else "")
            + (f", r {float(sq.get('radius', 0)):.2f} m x{sq.get('laps')}"
               if shape == "circle" else ""))
        err0 = math.hypot(self._eta[0] - origin_xy[0],
                          self._eta[1] - origin_xy[1])
        if (err0 <= float(self.cfg.engage["start_err_max_m"])
                and self._settle_s(shape) <= 0.0):
            self._arm_path(sq, shape, origin_xy, yaw_fixed, depth)
            return
        # GO THERE FIRST, then settle, then fly (operator 2026-08-14). The
        # path is anchored to a tag, which is generally NOT where the vehicle
        # engaged, so START means "take station over the tag and hold until
        # the estimate is quiet". DP-hold does the travelling — it is the same
        # controller and the same interlocks, so there is no separate,
        # less-guarded motion mode.
        self._approach = {"sq": sq, "shape": shape, "xy": origin_xy,
                          "yaw": yaw_fixed, "depth": depth,
                          "t0": now(), "t_in": None, "t_prev": now(),
                          # the setpoint STARTS at the vehicle and walks to
                          # the tag (see _tick_approach) — stepping it 2 m in
                          # one tick asks the controller for a lunge, and the
                          # vehicle answers with the overshoot-and-wallow the
                          # operator saw on 2026-08-14.
                          "sp": [float(self._eta[0]), float(self._eta[1])]}
        self.ctrl.set_target_ned((self._eta[0], self._eta[1], depth), yaw_fixed)
        d0 = math.hypot(self._eta[0] - origin_xy[0],
                        self._eta[1] - origin_xy[1])
        where = (f"tag {sq['origin_tag']}" if sq.get("origin_tag")
                 else "the start point")
        self.reason = f"approaching {where} ({d0:.2f} m)"
        self._event(f"going to {where} ({d0:.2f} m)")
        self._log_event(f"APPROACH {where} d={d0:.2f} m then settle "
                        f"{self._settle_s(shape):.0f} s")
        self.bus.log.emit(
            "warn", f"mpc: heading for {where} ({d0:.2f} m away), then "
                    f"holding {self._settle_s(shape):.0f} s before the {shape} "
                    f"starts")
        return

    def _refuse(self, why: str) -> None:
        """A refused START must reach the PANEL, not just the log line at the
        bottom of the window. Before 2026-08-14 the chip kept saying "engaged
        (DP hold)" and the operator had no idea the path had been rejected."""
        self.reason = f"START refused: {why}"
        self.bus.log.emit("warn", f"mpc: START TRAJ refused — {why}")

    def _mission_yaw(self, sq: dict) -> float:
        """The heading to hold, as a DATUM-frame yaw.

        ``yaw_map_deg`` is an ABSOLUTE heading in the tag-map frame (90 =
        facing +y), which is the only way to say "sit on the tag facing +y":
        a datum-relative angle means something different every time the
        vehicle engages on a different heading."""
        ym = sq.get("yaw_map_deg")
        if ym not in (None, "", "current"):
            yaw0 = float(self._datum["yaw0"]) if self._datum else 0.0
            return _wrap_pi(math.radians(float(ym)) - yaw0)
        yf = sq.get("yaw_fixed_deg", "current")
        return (float(self._eta[5]) if yf in ("current", None)
                else math.radians(float(yf)))

    def _advance_path_clock(self, t: float, meas) -> float:
        """Update the spatial path cursor (or legacy wall-time trajectory).

        In path mode this method does not advance a clock. It projects the
        measured hull onto the active segment, builds one segment-gated NED
        plan, and installs that exact plan in either controller. ``_tau`` is
        merely the nominal sampler time corresponding to the spatial target,
        retained for CSV compatibility. ``path_following: false`` restores
        the old wall-clock trajectory tracker.
        """
        if self._t0_traj is None:
            return 0.0
        wall = t - self._t0_traj
        # MPCC owns theta: progress is a DECISION of its own solver, so there
        # is nothing here to advance and nothing to install. Asking it "where
        # should I be at t" is exactly the trajectory-tracking question it
        # exists to stop answering.
        if self.ctrl is not None and hasattr(self.ctrl, "progress_m"):
            self._tau = float(self.ctrl.progress_m)
            if meas is not None:
                a, c = self.ctrl.path_errors(np.asarray(meas["eta"], float))
                self._path_err = (a, c)
                self._path_lag = abs(float(a)) if a is not None else 0.0
            return self._tau
        if not self.cfg.path_following or self._path_cursor is None:
            self._path_err = (None, None)
            self._tau = wall
            return wall
        if self.ctrl is None or meas is None:
            return self._tau
        dt = max(1e-3, min(0.5, t - self._tau_t))
        self._tau_t = t
        eta = np.asarray(meas["eta"], float)
        _target, psi_path, along, cross, v_ref = self._path_cursor.step(
            eta[:3], dt)
        # THE WHOLE HORIZON, not one setpoint. Handing the controller a single
        # target here is what silently blinded the tracking NMPC: the last line
        # of set_target_ned is `self._ref_traj = None`, so the sampler armed
        # milliseconds earlier by set_square_ned was destroyed on the first
        # tick and the solver fell back to extrapolating that one point along a
        # STRAIGHT RAY for all 61 stages. It drove through every corner without
        # knowing one was there. The plan is asked for at the size the
        # controller declares, so a PID (1 stage) and an NMPC (N+1) receive the
        # same geometry and the A/B stays a controller comparison.
        if hasattr(self.ctrl, "set_path_plan_ned"):
            plan = self._path_cursor.plan(
                int(getattr(self.ctrl, "path_plan_steps", 1)),
                float(getattr(self.ctrl, "path_plan_dt",
                              1.0 / self.cfg.ctrl_hz)),
                self._path_depth, self._path_yaw_fixed,
                self._path_heading_follow)
            self.ctrl.set_path_plan_ned(plan)
        else:
            yaw = (psi_path if self._path_heading_follow
                   else self._path_yaw_fixed)
            v_ned = (v_ref * math.cos(psi_path), v_ref * math.sin(psi_path),
                     0.0)
            self.ctrl.set_target_ned(
                (_target[0], _target[1], self._path_depth), yaw, v_ned=v_ned)
        self._path_err = (along, cross)
        self._path_lag = abs(float(along))
        self._tau = float(self._path_cursor.theta)
        return self._tau

    def _note_w_hat_rail(self, w_hat) -> None:
        """Say so when the disturbance observer is living on its limiter.

        A DOB pinned to ``w_hat_clip`` is not estimating a disturbance, it is
        reporting that its own loop has diverged — and the NMPC then spends
        full authority cancelling a number that is a rail. This is what the
        2026-08-17 pool session looked like from the pilot's seat: thrust
        reversing at ~3.6 Hz with the vehicle lurching (|ax_surge| pinned at
        the 0.5 cap 52 % of ticks, w0 flipping +/-15 N tick to tick,
        sessions/low_level_controller_data/20260817/0817_103431/mpc_103538.csv).
        An audit of EVERY dobmpc run on disk shows the X rail hit 66-95 % of
        the time going back to 2026-08-12, so this is long-standing and NOT a
        regression — it was simply never surfaced. Prime suspects, both
        unmeasured: axis_gain is [예측] (a wrong plant gain becomes phantom
        disturbance the moment note_applied reports a wrench the vehicle never
        produced) and the ~0.27 s tag latency inside an observer that assumes a
        fresh measurement.
        """
        if w_hat is None or self.cfg is None or not self.engaged:
            self._rail_ticks = 0
            return
        w = np.abs(np.asarray(w_hat, float).ravel())
        clip = np.asarray(self.cfg.w_hat_clip, float)
        if w.size < clip.size or not np.any(w[:clip.size] >= 0.98 * clip):
            self._rail_ticks = 0
            return
        self._rail_ticks = getattr(self, "_rail_ticks", 0) + 1
        n = int(round(3.0 * self.cfg.ctrl_hz))          # 3 s of solid railing
        if self._rail_ticks == n:
            axes = ", ".join(f"{n_}={v:.0f}" for n_, v in
                             zip("XYZKMN", w[:clip.size]) if v >= 0.98 * clip[
                                 "XYZKMN".index(n_)])
            self.bus.log.emit(
                "error",
                f"mpc: w_hat has been PINNED to its clip for 3 s ({axes}) — "
                f"the disturbance observer has diverged, and the solver is "
                f"spending full authority on a rail. Fly 'mpcc' or 'mpc' "
                f"(no w_hat) until the axis gains are calibrated.")
            self._event("w_hat railed: DOB diverged")

    def _ref_speed(self):
        """What the reference is asking for right now [m/s], or None.

        Read from whoever owns progress — the MPCC's own v_theta, or the
        cursor's v_ref — rather than recomputed, so the number on the screen
        and the number in the loop cannot disagree."""
        # A FOLLOW runs with traj_on False, and its feedforward is now the
        # load-bearing diagnostic — an object estimate asking for more than the
        # vehicle can swim is the whole 2026-08-23 failure. Read the value the
        # follow actually ISSUED, for this docstring's reason.
        if self.follow is not None:
            return float(self.follow.get("ff_issued_m_s", 0.0))
        if not self.traj_on:
            return None
        if self.ctrl is not None and hasattr(self.ctrl, "progress_m"):
            return float(getattr(self, "_mpcc_v_theta", 0.0))
        if self._path_cursor is not None:
            return float(np.interp(self._path_cursor.target_theta,
                                   self._path_cursor._s,
                                   self._path_cursor._v))
        return None

    def _path_theta(self):
        """Arclength progress along the mission curve, whoever owns it."""
        if self.ctrl is not None and hasattr(self.ctrl, "progress_m"):
            return float(self.ctrl.progress_m)
        if self._path_cursor is not None:
            return float(self._path_cursor.theta)
        return None

    def _path_lap(self, t_traj) -> int:
        for owner in (self.ctrl, self._path_cursor):
            if owner is not None and hasattr(owner, "lap"):
                return int(owner.lap())
        if self.ctrl is not None and self.ctrl.scenario:
            return _scenario_lap(self.ctrl.scenario, t_traj or 0.0)
        return 0

    def _path_done(self, t_traj, T_run) -> bool:
        """Mission complete — by ARCLENGTH when a curve owns the run, by the
        wall clock only in the legacy trajectory-tracking mode."""
        for owner in (self.ctrl, self._path_cursor):
            if owner is not None and hasattr(owner, "complete"):
                return bool(owner.complete)
        return bool(t_traj is not None and t_traj > T_run)

    _TANGENT_DT = 0.25          # s of lookahead used to read the path tangent

    def _path_split(self, meas, p_ref, t_traj):
        """(along, cross) of the tracking error, in the path's own frame.

        Two different questions, so two different sign conventions, each the
        one an operator would read off without translating:
          along  LAG — how far the vehicle is BEHIND the virtual target, so
                 positive means late (it is ``+u . (ref - veh)``).
          cross  which SIDE of the path the vehicle is on, positive to the
                 LEFT of the direction of travel. In NED (x north, y east, z
                 down) the left normal of heading u is (u_y, -u_x), so this
                 is ``(u_y, -u_x) . (veh - ref)``.

        This is the split that says whether path following is working: stage 0
        deliberately sits ahead of the projection, so |p - ref| alone reads
        as failure even when the vehicle is exactly on the line. Returns
        (None, None) with no usable tangent — station hold, or legacy reference
        instants where the tangent is undefined.
        """
        if not self.traj_on or self.ctrl is None or t_traj is None:
            return None, None
        if self._path_err[0] is not None:
            return (float(self._path_err[0]), float(self._path_err[1]))
        try:
            p_ahead, _yaw, _v_ref = self.ctrl.ref_ned_at(
                float(t_traj) + self._TANGENT_DT)
        except Exception:                                     # noqa: BLE001
            return None, None
        tx = float(p_ahead[0]) - float(p_ref[0])
        ty = float(p_ahead[1]) - float(p_ref[1])
        n = math.hypot(tx, ty)
        if not (n > 1e-4):
            return None, None
        tx, ty = tx / n, ty / n
        dx = float(p_ref[0]) - float(meas["eta"][0])
        dy = float(p_ref[1]) - float(meas["eta"][1])
        return tx * dx + ty * dy, -(ty * dx - tx * dy)

    def _settle_s(self, shape: str | None = None) -> float:
        """How long to hold over the origin before arming the mission.

        A PATH mission needs it: the square starts the instant it expires, and
        starting one 30 cm off the first corner is a bad first metre.

        A STATION mission does not, and the reason is worth spelling out
        because it is the opposite of intuition. During the settle the
        setpoint is still LEASHED to ``approach_lead_m`` ahead of the hull;
        arming station replaces that with the tag itself. So the settle does
        not help the vehicle arrive — it holds it in the SLOWER of the two
        regimes for another ten seconds. There is also nothing to start
        afterwards: the armed station target is the same point the approach
        was already walking to.

        The one thing it does buy is the ``imu_dr`` static window
        (``_anchor_dr`` measures the gyro bias over the last seconds of it),
        so when that experiment is on the full settle comes back.
        """
        base = float(self.cfg.engage.get("settle_s", 10.0))
        if str(shape or "").lower() != "station":
            return base
        if self.cfg.imu_dr.get("enabled"):
            return base
        return float(self.cfg.engage.get("settle_station_s", 0.0))

    def _tick_approach(self, t: float) -> None:
        """Fly to the path origin under DP, hold there, then arm the path."""
        ap = self._approach
        if ap is None or self._eta is None:
            return
        err = math.hypot(self._eta[0] - ap["xy"][0], self._eta[1] - ap["xy"][1])
        tol = float(self.cfg.engage["start_err_max_m"])
        where = (f"tag {ap['sq']['origin_tag']}" if ap["sq"].get("origin_tag")
                 else "the start point")
        # Walk the SETPOINT toward the tag at a bounded speed, and never let
        # it run more than `approach_lead_m` ahead of the vehicle: a position
        # controller chases the error it is given, so a setpoint that teleports
        # is a full-authority step command.
        dt = max(1e-3, min(0.5, t - ap.get("t_prev", t)))
        ap["t_prev"] = t
        v = float(self.cfg.engage.get("approach_speed_m_s", 0.20))
        lead = float(self.cfg.engage.get("approach_lead_m", 0.50))
        sp = ap["sp"]
        dx, dy = ap["xy"][0] - sp[0], ap["xy"][1] - sp[1]
        rem = math.hypot(dx, dy)
        v_ned = (0.0, 0.0, 0.0)
        if rem > 1e-6:
            step = min(v * dt, rem)
            sp[0] += step * dx / rem
            sp[1] += step * dy / rem
            # THE SETPOINT'S OWN SPEED, handed to the controller. Without it
            # the approach settles wherever kp*lead happens to balance drag —
            # measured 0.084 m/s against a 0.10 command (2026-08-18, 4 runs,
            # |axis| p50 0.18 against a 0.50 cap: not thrust-limited, just not
            # asked). This is the same v_ned the path follower has always
            # passed; the approach simply never did.
            #
            # Tapered by the CONTROLLER'S OWN PREVIEW: an NMPC extrapolates
            # the reference along v_ref for its whole horizon, so a full-speed
            # feedforward within one horizon of the target is a request to
            # drive past it. A PID (one stage, 0.05 s) is unaffected by the
            # taper, which is what keeps the two comparable.
            preview = max(1e-3, (int(getattr(self.ctrl, "path_plan_steps", 1))
                                 * float(getattr(self.ctrl, "path_plan_dt",
                                                 dt))))
            v_eff = min(v, rem / preview)
            v_ned = (v_eff * dx / rem, v_eff * dy / rem, 0.0)
        # Hard leash: the setpoint may never sit further than `lead` ahead of
        # the hull, whatever the tick timing did. Gating the step alone lets
        # one long tick jump the leash; clamping the RESULT cannot.
        ahead = math.hypot(sp[0] - self._eta[0], sp[1] - self._eta[1])
        if ahead > lead > 0.0:
            k = lead / ahead
            sp[0] = self._eta[0] + (sp[0] - self._eta[0]) * k
            sp[1] = self._eta[1] + (sp[1] - self._eta[1]) * k
        self.ctrl.set_target_ned((sp[0], sp[1], ap["depth"]), ap["yaw"],
                                 v_ned=v_ned)
        if err > tol:
            ap["t_in"] = None
            self.phase, self.phase_detail = "approach", f"{err:.2f} m to go"
            self.reason = f"approaching {where} ({err:.2f} m)"
        else:
            if ap["t_in"] is None:
                ap["t_in"] = t
            held = t - ap["t_in"]
            self.phase = "settle"
            settle = self._settle_s(ap["shape"])
            self.phase_detail = f"{max(0.0, settle - held):.0f} s"
            if held >= settle:
                self._event(f"arrived, held {settle:.0f} s")
                self._approach = None
                self._arm_path(ap["sq"], ap["shape"], ap["xy"], ap["yaw"],
                               ap["depth"])
                return
            self.reason = (f"settling over {where} "
                           f"({settle - held:.0f} s)")
        limit = float(self.cfg.engage.get("approach_max_s", 180.0))
        if t - ap["t0"] > limit:
            self._approach = None
            self.reason = f"approach timed out ({err:.2f} m from {where})"
            self.bus.log.emit(
                "error", f"mpc: gave up approaching {where} after "
                         f"{limit:.0f} s ({err:.2f} m away) — holding "
                         f"position. Fly closer and press START again.")

    # ========================================================= object follow
    # Keep a relative pose on the object the pilot clicked. The mission has no
    # geometry of its own: the "path" is wherever the object goes, and what is
    # held is the offset the vehicle happened to have when START was pressed.
    #
    # THREE FRAMES MEET HERE and getting them mixed is the way this feature
    # goes quietly wrong, so the division is absolute:
    #   * the ANCHOR (control/object_nav.py) works entirely in the MAP frame —
    #     the tag world the pool and the plot are drawn in;
    #   * the CONTROLLER works entirely in the ENGAGE-DATUM frame;
    #   * `_issue_follow_target` is the ONE place that crosses between them.
    # Everything in `self.follow` is MAP-frame, and it is named so.
    def _follow_refusal(self) -> str:
        """Why a follow may NOT be armed — each cause with its OWN sentence.

        A bare "follow refused" is a pool session spent guessing, and four of
        these five causes are fixed by a different config or a different mode
        rather than by flying somewhere else.
        """
        if self._obj is None:
            return ("no object tracker — this station was started without "
                    "--pose (follow needs --pose AND --mpc)")
        if not self._obj_src_ok:
            return self._obj_note
        if not bool(getattr(self.ctrl, "follow_ok", False)):
            # fail-closed: a controller that has never considered a moving
            # setpoint does not get one by default.
            return (f"{self.cfg.mode} cannot follow a moving setpoint — it "
                    f"discards the velocity feedforward and rebuilds its path "
                    f"every tick. Fly dobmpc, mpc or pid.")
        st = self._obj.predict(now())
        if st["state"] != ON.LIVE:
            return ("no object lock (" + str(st["state"])
                    + (f": {st['note']}" if st["note"] else "") + ")")
        if self._obj.axis is not None and not self._obj.yaw_ok:
            return ("the object's heading axis is within "
                    f"{math.degrees(math.asin(min(1.0, float(self._obj.cfg['yaw_min_horiz'])))):.0f}"
                    " deg of vertical, so its yaw is undefined — set "
                    "object_nav.yaw_axis: none to hold the offset in the map "
                    "frame instead")
        if self._obj_last is None or self._obj_last.pair_dt_ms is None:
            return ("the object pose and the tag fix are not paired — no fix "
                    "from the object's own camera frame")
        if not self._obj_last.pair_exact:
            # NOT a refusal: a loose pair is degraded, not wrong, and the
            # operator may well want to fly it anyway. But it must be said,
            # because it is the moment the camera extrinsic stopped
            # cancelling and the 0.2855 m lever arm re-entered the budget.
            self.bus.log.emit(
                "warn", f"mpc: arming follow on a LOOSE pair "
                        f"({self._obj_last.pair_dt_ms:.0f} ms) — the camera "
                        f"extrinsic is no longer cancelling; expect up to "
                        f"|t_frd_cam| x 2 sin(dyaw/2) of object-position "
                        f"error on top of everything else")
        return ""

    def _arm_follow(self, sq: dict) -> None:
        """Capture the CURRENT relative pose and start holding it.

        The offset is stored in the OBJECT's own yaw frame, so the object
        translating moves the vehicle with it and the object yawing makes the
        vehicle ORBIT and keep the same face. The object's roll and pitch are
        deliberately thrown away (object_nav.offset_in_object_frame explains
        why: MANUAL_CONTROL has no K/M axis, so a roll the follow tried to
        answer would be a command the allocation drops).

        Because the offset comes from where the vehicle IS, the armed target
        is the vehicle's own position: START never causes a lunge. That is
        pinned by a test rather than left as a comment.
        """
        a = self._obj
        t = now()
        st = a.predict(t)
        p_veh = self._datum_to_map_p(self._eta[:3])
        yaw_veh = _wrap_pi(float(self._eta[5]) + self._datum_yaw0())
        yaw_obj = st["yaw"] if a.axis is not None else None
        off, dyaw = ON.offset_in_object_frame(p_veh, yaw_veh, st["p"], yaw_obj)
        cap = float(sq.get("speed", 0.05) or 0.05)
        self.follow = {
            "kind": "follow",
            # MAP frame, all of it.
            "offset_obj": [float(v) for v in off],
            "dyaw_deg": math.degrees(dyaw),
            "hold_m": float(np.linalg.norm(off)),
            "arm_p_map": [float(v) for v in p_veh],
            "sp": [float(v) for v in p_veh],       # the walked setpoint
            "sp_yaw": float(yaw_veh),
            "speed_cap_m_s": cap,
            "yaw_rate_deg_s": float(sq.get("yaw_rate_deg_s", 60.0) or 60.0),
            "yaw_axis": ("none" if a.axis is None else "xyz"[a.axis]),
            "max_excursion_m": float(a.cfg["max_excursion_m"]),
            "approach_lead_m": float(self.cfg.engage.get("approach_lead_m",
                                                         0.50)),
            "ff_max_m_s": float(self.cfg.engage.get("follow_ff_max_m_s",
                                                    0.30)),
            # THE JUMP GATE'S LAST WORD. object_nav's gate rejects one outlier
            # but RESEEDS after `reseed_after_n` of them — an auto-capitulation
            # that is right for a display marker and wrong for a live follow,
            # where it teleports the target. Latch the count so `_tick_follow`
            # can end the follow instead of chasing the new anchor.
            "n_reseed_at_arm": int(a.n_reseed),
            "range_at_arm_m": st["distance_m"],
            "t_arm": float(t),
            "t_prev": float(t),
            "state": "following",
            "err_m": 0.0,
            "ff_clipped_n": 0,
            "t_ff_warn": 0.0,
        }
        # Like STATION: no trajectory clock, no laps, no timeout. There is
        # nothing to complete.
        self.traj_on = False
        self.station = None
        self.phase, self.phase_detail = "follow", ""
        self._issue_follow_target(p_veh, yaw_veh, np.zeros(3))
        d = self.follow["hold_m"]
        what = (f"FOLLOW: holding {d * 100:.0f} cm off the object"
                f" ({'map-frame offset' if a.axis is None else 'object frame'}"
                f", heading {self.follow['dyaw_deg']:+.0f} deg), "
                f"excursion limit {self.follow['max_excursion_m']:.2f} m")
        self.reason = "following the object"
        # ARMED WIDER THAN THE CLAMP. The excursion limit is a sphere about
        # where START was pressed, so a follow whose hold distance already
        # exceeds it can only track the object through a fraction of a turn
        # before the reference stops moving — and it does that silently, as a
        # clamp rather than a refusal. 2026-08-23: armed at 202 cm against a
        # 150 cm limit and it was `leashed` inside 1.5 s, which reads from the
        # cockpit as "follow does not work" rather than "the geometry cannot".
        if d > float(a.cfg["max_excursion_m"]):
            self.bus.log.emit(
                "warn", f"mpc: FOLLOW armed at {d * 100:.0f} cm but the "
                        f"excursion clamp is "
                        f"{float(a.cfg['max_excursion_m']) * 100:.0f} cm — the "
                        f"target will clamp almost immediately. Get closer to "
                        f"the object, or raise object_nav.max_excursion_m.")
        self._event(f"FOLLOW armed ({d * 100:.0f} cm)")
        self._log_event(what)
        self.bus.log.emit("warn", f"mpc: {what}")

    def _tick_follow(self, t: float) -> None:
        """One follow tick: object -> goal -> walked setpoint -> controller.

        Four layers sit between the object estimate and the vehicle, and each
        one answers a failure the others cannot:

          1. EXCURSION CLAMP — the goal may not leave a sphere of
             ``max_excursion_m`` about where START was pressed. A reference
             clamp, never a refusal and never a stop: it is the form of
             position limit the operator allowed back after the geofence was
             removed on 2026-08-14.
          2. RATE-LIMITED WALK — the setpoint travels at ``speed_cap_m_s``, so
             an object estimate that jumps moves the target at walking pace
             instead of handing the controller a step command.
          3. HARD LEASH ON THE RESULT — the setpoint may never sit further
             than ``approach_lead_m`` from the hull, whatever the tick timing
             did. Gating the STEP alone is not enough: one long tick jumps the
             leash, which is the lesson `_tick_approach` already carries.
          4. FEEDFORWARD — the goal's own velocity (including the orbit term)
             plus a catch-up term tapered by the controller's preview. This
             layer is the answer to the measured leash limit: with no
             feedforward the approach settles where kp*lead balances drag —
             0.084 m/s (2026-08-18, memory: approach-speed-leash-limited) —
             and a follow with no feedforward simply cannot keep up with
             anything faster than that.
        """
        f = self.follow
        if f is None or self._eta is None or self._obj is None:
            return
        dt = max(1e-3, min(0.5, t - float(f.get("t_prev", t))))
        f["t_prev"] = float(t)
        st = self._obj.predict(t)
        p_veh = self._datum_to_map_p(self._eta[:3])
        sp = np.asarray(f["sp"], float)
        f["age_s"] = st["age_s"]

        if st["state"] in (ON.LOST, ON.COLD):
            self._follow_to_station(
                "object lost" if st["state"] == ON.LOST
                else "object never locked")
            return
        # THE ANCHOR GAVE UP AND RE-ANCHORED. A reseed means the jump gate saw
        # `reseed_after_n` rejections in a row and concluded the object moved.
        # At 10 Hz that verdict costs 0.5 s and is unfalsifiable — a genuine
        # drag arrives as ACCEPTED small steps (0.35 m/frame is 3.5 m/s), so a
        # reseed under a follow is an estimator re-registration, not a moving
        # object. 2026-08-23 it consumed a 1.32 m snap onto a phantom 0.44 m
        # from the lens and flew at it. Ending the follow is the honest answer:
        # the operator re-arms once the pose is trustworthy again.
        if int(self._obj.n_reseed) > int(f.get("n_reseed_at_arm", 0)):
            self._follow_to_station("object estimate re-seeded — pose jumped")
            return
        if st["state"] != ON.LIVE:
            # THE FREEZE. Re-issue the last setpoint with NO feedforward and
            # say so in amber. Nothing here extrapolates: an estimate that has
            # gone stale is exactly the one whose velocity should not be
            # trusted to keep driving the vehicle.
            f["state"] = "stale"
            f["err_m"] = float(np.linalg.norm(sp - p_veh))
            self.phase_detail = (f"OBJECT STALE {st['age_s']:.1f}s"
                                 if st["age_s"] is not None else "OBJECT STALE")
            self.reason = "object stale — holding the last setpoint"
            self._issue_follow_target(sp, f["sp_yaw"], np.zeros(3))
            return

        yaw_obj = st["yaw"] if self._obj.axis is not None else None
        goal, goal_yaw, v_track = ON.follow_goal(
            st["p"], yaw_obj, f["offset_obj"], math.radians(f["dyaw_deg"]),
            st["v"], st["r"])
        goal, leashed = ON.clamp_excursion(goal, f["arm_p_map"],
                                           f["max_excursion_m"])
        # (2) walk the setpoint
        v_cap = float(f["speed_cap_m_s"])
        step = goal - sp
        n = float(np.linalg.norm(step))
        if n > 1e-9:
            sp = sp + step * (min(v_cap * dt, n) / n)
        # ...and the heading, at its own rate limit
        r_cap = math.radians(float(f["yaw_rate_deg_s"])) * dt
        dyaw = _wrap_pi(float(goal_yaw) - float(f["sp_yaw"]))
        f["sp_yaw"] = _wrap_pi(float(f["sp_yaw"])
                               + max(-r_cap, min(r_cap, dyaw)))
        # (3) hard leash, on the RESULT
        lead = float(f["approach_lead_m"])
        ahead = float(np.linalg.norm(sp - p_veh))
        if ahead > lead > 0.0:
            sp = p_veh + (sp - p_veh) * (lead / ahead)
        f["sp"] = [float(v) for v in sp]
        # (4) feedforward: the goal's own velocity, plus catch-up tapered by
        # the controller's own preview. The TRACK term is NOT tapered — by the
        # time the vehicle converges the goal really will be moving that fast —
        # while driving the catch-up at full speed inside one horizon of the
        # target is a request to overshoot it (`_tick_approach`'s lesson).
        preview = max(1e-3, int(getattr(self.ctrl, "path_plan_steps", 1))
                      * float(getattr(self.ctrl, "path_plan_dt", dt)))
        rem = goal - sp
        rn = float(np.linalg.norm(rem))
        v_ff = np.asarray(v_track, float).copy()
        if rn > 1e-9:
            v_ff = v_ff + rem * (min(v_cap, rn / preview) / rn)
        # (5) THE EXCURSION CLAMP MUST BIND THE FEEDFORWARD TOO.
        # `clamp_excursion` stops the GOAL at the sphere, but the feedforward
        # is a separate channel into the same reference, so a clamped follow
        # could sit at the limit with v_ff pinned outward on every tick — the
        # horizon leaning ~0.9 m past a boundary the operator set, forever,
        # with only a log line. Remove the component that points further out
        # and keep the tangential part, so a leashed follow still tracks
        # sideways motion.
        if leashed:
            out = np.asarray(goal, float) - np.asarray(f["arm_p_map"], float)
            n_out = float(np.linalg.norm(out))
            if n_out > 1e-9:
                n_hat = out / n_out
                v_ff = v_ff - max(0.0, float(v_ff @ n_hat)) * n_hat
        # (6) HARD CAP ON THE FEEDFORWARD. `set_target_ned` keeps v_ned and
        # `_xref_ned` extrapolates the WHOLE horizon along it
        # (pos_world = p_ref + v_ref*k*dt), so an unbounded track term does not
        # nudge the reference — it relocates the far end of it. 2026-08-23 a
        # spinning phantom drove ref_speed_m_s to 3.15 m/s while the logged
        # stage-0 setpoint sat frozen within 4 cm of the datum, which is both
        # how the vehicle was dragged 0.54 m and how it stayed invisible in the
        # CSV; in the run after, the same term handed acados a static position
        # target with 2.97 m/s of velocity and it returned status 4.
        # The bound is what the vehicle can actually do, so asking for more is
        # never information: [유도] full-loop F = 86.7 v + 5.76 N
        # (.claude/journal/consults.md 2026-08-18) at U_MAX 30 N -> 0.28 m/s.
        ff_cap = float(f.get("ff_max_m_s", 0.30))
        if not (math.isfinite(ff_cap) and ff_cap > 0.0):
            ff_cap = 0.30                       # fail CLOSED, never uncapped
        ff_n = float(np.linalg.norm(v_ff))
        if ff_n > ff_cap:
            v_ff = v_ff * (ff_cap / ff_n)
            f["ff_clipped_n"] = int(f.get("ff_clipped_n", 0)) + 1
            # Clipping is a DIAGNOSIS, not just a limit. Say so out loud, at
            # most once every 5 s.
            if t - float(f.get("t_ff_warn", 0.0)) > 5.0:
                f["t_ff_warn"] = float(t)
                self.bus.log.emit(
                    "warn", f"mpc: FOLLOW feedforward clipped "
                            f"{ff_n:.2f} -> {ff_cap:.2f} m/s — the reference "
                            f"is asking for more than this vehicle can swim.")
        # ...and a cap is not an answer on its own. `_follow_to_station` above
        # catches the DISCONTINUOUS failure (a reseed); a phantom that slides
        # SMOOTHLY stays under the 0.35 m jump gate, never reseeds, and would
        # otherwise buy a permanent pull at the cap.
        #
        # The test is the OBJECT's own apparent velocity, never |v_ff|: v_ff
        # also carries the catch-up term, which is large and legitimate
        # whenever the vehicle is simply behind its setpoint (at arm, after a
        # leashed stretch, or in any test where the hull does not move). Judging
        # the total would end healthy follows for being slow.
        # 3x the cap is gross implausibility with room to spare — a real orbit
        # in these tests demands ~0.2 m/s and the 2026-08-23 phantoms demanded
        # 3.15 and 2.97 m/s, ten times the cap.
        v_obj_n = float(np.linalg.norm(np.asarray(v_track, float)))
        if v_obj_n > 3.0 * ff_cap:
            f["obj_fast_streak"] = int(f.get("obj_fast_streak", 0)) + 1
            if int(f["obj_fast_streak"]) >= int(round(self.cfg.ctrl_hz)):
                self._follow_to_station(
                    f"object velocity implausible ({v_obj_n:.1f} m/s)")
                return
        else:
            f["obj_fast_streak"] = 0
        f["state"] = "leashed" if leashed else "following"
        f["err_m"] = float(np.linalg.norm(sp - p_veh))
        self._issue_follow_target(sp, f["sp_yaw"], v_ff)
        self.phase, self.phase_detail = "follow", (
            "EXCURSION LIMIT" if leashed
            else f"{f['err_m'] * 100:.0f} cm")
        self.reason = ("following the object (excursion limit)" if leashed
                       else "following the object")
        if leashed and int(t * self.cfg.ctrl_hz) % int(
                max(1, 5 * self.cfg.ctrl_hz)) == 0:
            self.bus.log.emit(
                "warn", f"mpc: the object has taken the target "
                        f"{f['max_excursion_m']:.2f} m from where START was "
                        f"pressed — the reference is CLAMPED there. Nothing "
                        f"stops the vehicle by position; STOP TRAJ to hold.")

    def _issue_follow_target(self, sp_map, yaw_map, v_ff_map) -> None:
        """THE one crossing from the MAP frame into the controller's datum
        frame. Every follow setpoint goes through here so the two frames have
        exactly one place they can be got wrong.

        ``r_ned`` is deliberately NOT passed. ``HwDobMpc.set_target_ned``
        forwards it as ``r_ref`` but leaves ``yaw_target = yaw_ref``, so
        ``_xref_ned`` computes ``delta = 0`` and forces ``xref[11,:] = 0``:
        the yaw-rate feedforward is a no-op on the default controller and
        live only on the PID. A feedforward that one of two controllers
        silently ignores makes the two incomparable for no gain; the heading
        rate limit above does the same job for both.
        """
        p = self._map_to_datum_p(sp_map)
        v = self._map_to_datum_v(v_ff_map)
        if self.follow is not None:
            # What was ISSUED, for the panel's reference-speed readout — the
            # number on the screen and the number in the loop are then the
            # same object (`_ref_speed`).
            self.follow["ff_issued_m_s"] = float(np.linalg.norm(v[:2]))
        self.ctrl.set_target_ned(
            (float(p[0]), float(p[1]), float(p[2])),
            self._map_to_datum_yaw(yaw_map),
            v_ned=(float(v[0]), float(v[1]), float(v[2])))

    def _follow_to_station(self, why: str) -> None:
        """Demote a follow to a STATION hold on its last setpoint.

        A DEMOTION and not a disengage, for exactly the reason
        ``station_bridge.py`` argues at length: disengaging drops DEPTH hold
        too, this vehicle is negatively buoyant (net -5.7 N), and a sinking
        vehicle's view changes — which makes re-acquiring the object LESS
        likely, not more. Holding the last known good pose keeps the object in
        frame if anything will.
        """
        f = self.follow
        if f is None:
            return
        # Keep the follow's own record. `_object_nav_meta` reads self.follow,
        # which this method is about to clear — so the runs that ENDED badly,
        # the only ones worth a post-mortem, recorded nothing but the reason.
        # Everything the whitelist wants (the cap, the clip count, the armed
        # offset) lives here and nowhere else.
        self._follow_last = dict(f)
        sp = np.asarray(f["sp"], float)
        p = self._map_to_datum_p(sp)
        yaw = self._map_to_datum_yaw(f["sp_yaw"])
        held = f.get("err_m")
        # Install the stationary target BEFORE dropping the follow. The order
        # matters on the unhappy path this method exists for: if set_target_ned
        # raised with self.follow already None, nothing would own the reference
        # and the moving one (feedforward included) would stay installed.
        self.ctrl.set_target_ned((float(p[0]), float(p[1]), float(p[2])), yaw)
        self.follow = None
        self.station = {
            "kind": "station",
            "origin_ned": [float(p[0]), float(p[1])],
            "depth_ned": float(p[2]),
            "yaw_fixed_ned_deg": math.degrees(yaw),
            "yaw_map_deg": math.degrees(float(f["sp_yaw"])),
            "origin_tag": None,
            # So a CSV/meta reader can tell this station apart from one the
            # operator asked for.
            "from_follow": str(why)}
        self.phase, self.phase_detail = "station", f"follow ended: {why}"
        self.reason = f"object follow ended ({why}) — station hold"
        self._event(f"FOLLOW -> STATION: {why}")
        self._log_event(f"FOLLOW ENDED - {why}; holding the last setpoint"
                        + (f" (err {held * 100:.0f} cm)" if held else ""))
        self.bus.log.emit(
            "warn", f"mpc: object follow ended ({why}) — holding the last "
                    f"setpoint as a STATION. Depth and heading are still "
                    f"held; nothing was disengaged.")

    # ============================================================ demo replay
    # Re-fly one recorded handheld demonstration (shape: replay). The track is
    # loaded from a session dir (umi_handheld.extract_pose output), speed-
    # limited by time dilation, re-anchored at the vehicle's pose at arm, and
    # streamed through PlanFilter -> PlanStitcher -> set_path_plan_ned — the
    # SAME seam a live diffusion policy will feed at 1 Hz later, which is the
    # point: M0 validates the label pipeline AND the integration in one run.
    # Everything here is in the ENGAGE-DATUM frame; the recorded track is
    # body-relative and never sees the tag map at all.
    def _clear_replay(self) -> None:
        """Drop the replay mission state — and NEUTRAL the jaw.

        ``grip_drive`` is a latched LEVEL in the command sink (hardware.py
        ``set_gripper_drive``): whoever set it non-zero must set it back, or
        the jaw keeps driving after the mission that asked for it is gone.
        Every mission-clearing site calls this, so the 4-site rule for
        ``follow`` covers the jaw too.
        """
        rp = self.replay
        if rp is not None:
            self._replay_last = dict(rp)
            if rp.get("gripper_on"):
                self.bus.cmd_gripper_drive.emit(0.0)
        self.replay = None
        self._plan_filter = None
        self._plan_stitcher = None

    def _replay_refusal(self) -> str:
        """Why a replay may NOT be armed — each cause with its own sentence
        (the ``_follow_refusal`` rule)."""
        if hasattr(self.ctrl, "progress_m"):
            return (f"{self.cfg.mode} contours its own path and ignores "
                    f"streamed plans (set_path_plan_ned is a no-op there) — "
                    f"fly dobmpc, mpc, *_tuned or pid")
        if not hasattr(self.ctrl, "set_path_plan_ned"):
            return f"{self.cfg.mode} cannot consume a streamed plan"
        sess = str(self.cfg.replay.get("session") or "")
        if not sess:
            return ("no replay session configured — set replay.session in "
                    "hw_mpc.yaml (or pass --replay-session) to a demo folder "
                    "holding poses.npy")
        if not (Path(sess) / "poses.npy").exists():
            return (f"replay session {sess!r} has no poses.npy — run "
                    f"python -m umi_handheld.extract_pose over it first")
        return ""

    def _arm_replay(self, sq: dict) -> None:
        import hashlib

        from .plan_stream import (FilterLimits, PlanFilter, PlanStitcher,
                                  anchor_track, chop_track, load_replay_track,
                                  time_dilate)

        r = self.cfg.replay
        try:
            track = load_replay_track(str(r["session"]))
        except (OSError, ValueError, KeyError) as e:
            self.ctrl.set_target_ned(self._eta[:3], self._eta[5])
            self._refuse(f"replay track unusable ({e})")
            return
        track, alpha = time_dilate(track, float(r["v_max_m_s"]))
        p0 = np.asarray(self._eta[:3], float).copy()
        yaw0 = float(self._eta[5])
        box = r.get("workspace_box_ned")
        lim = FilterLimits(
            v_max=float(r["v_max_m_s"]),
            anchor_max_m=float(r["anchor_max_m"]),
            jump_max_m=float(r["jump_max_m"]),
            box_ned_min=(tuple(float(v) for v in box[0]) if box else None),
            box_ned_max=(tuple(float(v) for v in box[1]) if box else None))
        self._plan_filter = PlanFilter(lim)
        self._plan_stitcher = PlanStitcher(blend_s=float(r["blend_s"]))
        period = float(r.get("stream_period_s", 0.0) or 0.0)
        if period > 0.0:
            plans = chop_track(track, p0, yaw0, t0=0.0,
                               horizon_s=float(r["horizon_s"]),
                               period_s=period)
        else:
            plans = [anchor_track(track, p0, yaw0, t0=0.0)]
        if box is None:
            # Not a fence — honesty. The filter's box is the ONLY position
            # gate since the geofence removal, and the first wet replay will
            # otherwise fly with it silently off (safety review 2026-08-30).
            self.bus.log.emit(
                "warn", "mpc: REPLAY workspace box is OFF "
                        "(replay.workspace_box_ned unset) — no position gate "
                        "on streamed plans")
        # The jaw starts in the state the DEMO starts in, without an edge: a
        # demo that opens on frame one must not drive an already-open jaw
        # against its stop for gripper_hold_max_s (safety review 2026-08-30).
        grip0 = "neutral"
        g_arr = getattr(track, "g", None)
        if g_arr is not None and np.asarray(g_arr).size:
            g0 = float(np.asarray(g_arr).ravel()[0])
            if g0 < float(r["gripper_close_below"]):
                grip0 = "close"
            elif g0 > float(r["gripper_open_above"]):
                grip0 = "open"
        try:
            poses_sha = hashlib.sha1(
                (Path(str(r["session"])) / "poses.npy")
                .read_bytes()).hexdigest()[:12]
        except OSError:
            poses_sha = None
        duration = float(track.t[-1])
        self.replay = {
            "kind": "replay",
            "session": str(r["session"]),
            "poses_sha1": poses_sha,
            "pending": list(plans),
            "n_plans": len(plans),
            "released": 0, "installed": 0, "clipped": 0, "rejected": 0,
            "period_s": period,
            "time_dilation": float(alpha),
            "duration_s": duration,
            "gripper_on": bool(r["gripper"]),
            "grip_state": grip0,
            "grip_since": 0.0,
            "grip_drive": 0.0,
            "grip_events": 0,
        }
        scen = {
            "kind": "replay",
            "session": str(r["session"]),
            "poses_sha1": poses_sha,
            "n_demo_samples": int(np.asarray(track.t).size),
            "duration_s": duration,
            "time_dilation": float(alpha),
            "v_max_m_s": float(r["v_max_m_s"]),
            "stream_period_s": period,
            "blend_s": float(r["blend_s"]),
            "origin_ned": [float(p0[0]), float(p0[1])],
            "depth_ned": float(p0[2]),
            "yaw_fixed_ned_deg": math.degrees(yaw0),
            "gripper": bool(r["gripper"]),
            "T_run_s": duration,
        }
        scen["path_timeout_s"] = duration * max(
            1.0, float(self.cfg.traj_timeout_factor))
        self.ctrl.scenario = scen
        # Never inherit a stale plan from an earlier mission (the _arm_path
        # rule); the first _tick_replay installs the real one.
        self.ctrl.set_path_plan_ned(None)
        self.traj_on = True
        self.station = None
        self._t0_traj = now()
        self._tau, self._tau_t, self._path_lag = 0.0, now(), 0.0
        self.phase, self.phase_detail = "replay", ""
        self.reason = "replay running"
        what = (f"REPLAY {scen['session']}: {duration:.0f} s, "
                f"{scen['n_demo_samples']} samples, dilation x{alpha:.2f} "
                f"(v_max {scen['v_max_m_s']:.2f} m/s), "
                f"{'ONE-SHOT plan' if period <= 0.0 else f'{len(plans)} plans @ {period:.1f} s'}"
                f", gripper {'ON' if scen['gripper'] else 'off'}")
        self._event("REPLAY started")
        self._log_event(f"REPLAY started - {what}")
        self.bus.log.emit("warn", f"mpc: {what}")

    def _replay_ref_now(self, t_rel: float):
        """The pose the anchor gate compares a new plan against: the ACTIVE
        reference if one exists, else the vehicle (first plan = anchored at
        the hull, which is where `_arm_replay` built it)."""
        st = self._plan_stitcher
        if st is not None and st.has_plan():
            p, yaw, _v, _r, _g = st.sample(np.array([float(t_rel)]))
            return np.asarray(p[:, 0], float), float(yaw[0])
        return np.asarray(self._eta[:3], float), float(self._eta[5])

    def _tick_replay(self, t_rel: float) -> None:
        """One replay tick: release due plans through the filter, then install
        the stitched horizon. Runs between `_advance_path_clock` and
        ``ctrl.step``, so the plan the solver sees is this tick's."""
        rp = self.replay
        st = self._plan_stitcher
        if rp is None or st is None or self.ctrl is None or t_rel is None:
            return
        while rp["pending"] and rp["pending"][0].t0 <= float(t_rel) + 1e-9:
            msg = rp["pending"].pop(0)
            rp["released"] += 1
            r_now = self._replay_ref_now(t_rel)
            verdict = self._plan_filter.evaluate(
                msg, r_now=r_now,
                cur_sample=(st.sample if st.has_plan() else None),
                now=float(t_rel))
            self._log_plan(msg, verdict, t_rel, r_now)
            if verdict.status in ("accept", "clip") and verdict.plan is not None:
                st.install(verdict.plan, float(t_rel))
                rp["installed"] += 1
                if verdict.status == "clip":
                    rp["clipped"] += 1
            else:
                rp["rejected"] += 1
                one_shot = float(rp.get("period_s", 0.0) or 0.0) <= 0.0
                if one_shot or verdict.escalate:
                    rp["pending"].clear()
                    if rp["installed"] == 0:
                        # NOTHING was ever flown: say so and END the mission.
                        # A one-shot reject has no escalation ladder to climb
                        # (n_plans == 1), and letting the wall clock later
                        # announce "complete" over a vehicle that never moved
                        # is the silent-status-0 failure mode this repo keeps
                        # re-finding (safety review 2026-08-30).
                        why = "; ".join(verdict.reasons) or "rejected"
                        self._event(f"REPLAY plan rejected — {why}")
                        self.bus.log.emit(
                            "error", f"mpc: REPLAY plan rejected ({why}) — "
                                     f"nothing to fly; holding HERE. Margins "
                                     f"are in plans.jsonl.")
                        self.set_traj(False)
                        self.reason = "replay plan rejected (DP hold)"
                        return
                    # The stream went bad mid-mission. Stop feeding it; the
                    # stitcher holds its endpoint (v=0), which is a DP hold,
                    # and the operator decides what happens next.
                    self._event("REPLAY plans rejected repeatedly — holding")
                    self.bus.log.emit(
                        "error", "mpc: REPLAY escalated after consecutive "
                                 "plan rejections — reference is holding the "
                                 "last plan's endpoint. STOP TRAJ to hold "
                                 "here instead.")
        if not st.has_plan():
            return
        # DIVERGENCE GUARD (safety review 2026-08-30). Geometric missions get
        # this from the PathCursor's projection leash; a wall-clock replay has
        # to check it itself, or a snagged vehicle watches the reference walk
        # the whole demo away and lunges at full authority on release — the
        # workspace box bounds the PLAN's knots, never the overshooting
        # vehicle, and a saturated solve is still status 0 (the PID
        # corner-deadlock lesson). Debounced like every other interlock.
        div = float(np.linalg.norm(
            self._replay_ref_now(t_rel)[0]
            - np.asarray(self._eta[:3], float)))
        lim_div = 2.0 * float(self.cfg.replay["anchor_max_m"])
        if div > lim_div:
            rp["div_streak"] = int(rp.get("div_streak", 0)) + 1
            if rp["div_streak"] >= max(2, int(round(0.5 * self.cfg.ctrl_hz))):
                self._event(f"REPLAY diverged ({div:.2f} m) — stopped")
                self.bus.log.emit(
                    "error", f"mpc: REPLAY reference is {div:.2f} m from the "
                             f"vehicle (> {lim_div:.2f} m for 0.5 s) — the "
                             f"vehicle cannot stay with the demo. Stopping "
                             f"and holding HERE.")
                self.set_traj(False)
                self.reason = "replay diverged (DP hold)"
                return
        else:
            rp["div_streak"] = 0
        K = int(getattr(self.ctrl, "path_plan_steps", 1))
        dt = float(getattr(self.ctrl, "path_plan_dt", 1.0 / self.cfg.ctrl_hz))
        ts = float(t_rel) + np.arange(K) * dt
        p, yaw, v, rr, g = st.sample(ts)
        speed = np.hypot(np.asarray(v[0], float), np.asarray(v[1], float))
        psi_path = np.where(speed > 1e-3,
                            np.arctan2(np.asarray(v[1], float),
                                       np.asarray(v[0], float)),
                            np.asarray(yaw, float))
        from .path_geometry import NedPlan

        self.ctrl.set_path_plan_ned(NedPlan(
            p_ned=np.asarray(p, float), yaw_ned=np.asarray(yaw, float),
            v_ned=np.asarray(v, float), r_ned=np.asarray(rr, float),
            psi_path=np.asarray(psi_path, float)))
        self.phase_detail = (f"{t_rel:.0f}/{rp['duration_s']:.0f} s"
                             + (" [hold]" if st.source_at(float(t_rel)) ==
                                "hold" else ""))
        self._tick_replay_gripper(
            (float(g[0]) if g is not None else None), float(t_rel))

    def _tick_replay_gripper(self, g, t_rel: float) -> None:
        """Replay the demo's jaw-width channel onto the momentary drive.

        Width (0 = closed) crosses ``gripper_close_below`` -> drive CLOSE;
        crosses ``gripper_open_above`` -> drive OPEN; in between, hold the
        current state (hysteresis — a borderline width must not toggle the
        drive at 20 Hz). The drive auto-neutrals after ``gripper_hold_max_s``
        (anti-stall; the jaw keeps its position when neutral) and every emit
        is EDGE-based, so a pilot G/H press between our edges still wins.
        OFF unless replay.gripper is true — the jaw path is open-loop (no
        position feedback exists) and early replays prove the trajectory
        before the jaw moves at all.
        """
        rp = self.replay
        if rp is None or not rp.get("gripper_on") or g is None:
            return
        r = self.cfg.replay
        want = rp["grip_state"]
        if g < float(r["gripper_close_below"]):
            want = "close"
        elif g > float(r["gripper_open_above"]):
            want = "open"
        if want != rp["grip_state"]:
            rp["grip_state"] = want
            rp["grip_since"] = float(t_rel)
            drive = {"close": -1.0, "open": +1.0}.get(want, 0.0)
            self._set_replay_grip_drive(drive)
            if drive:
                rp["grip_events"] += 1
                self._event(f"REPLAY gripper {want.upper()}")
        elif (rp["grip_drive"] != 0.0
              and (float(t_rel) - float(rp["grip_since"]))
              > float(r["gripper_hold_max_s"])):
            self._set_replay_grip_drive(0.0)   # anti-stall auto-neutral

    def _set_replay_grip_drive(self, v: float) -> None:
        rp = self.replay
        if rp is None:
            return
        if float(v) != float(rp.get("grip_drive", 0.0)):
            rp["grip_drive"] = float(v)
            self.bus.cmd_gripper_drive.emit(float(v))

    def _log_plan(self, msg, verdict, t_rel, r_now) -> None:
        """One JSON line per plan offered to the filter -> plans.jsonl in the
        run folder. This is the planner-vs-tracker attribution record: the raw
        plan (pre-filter), the anchor snapshot, and every margin NUMBER (a
        boolean verdict cannot be threshold-tuned afterwards)."""
        import hashlib

        try:
            p = np.asarray(msg.p_ned, float)
            small = p.shape[1] <= 64
            rec = {
                "t_rel": round(float(t_rel), 4),
                "wall": time.strftime("%Y-%m-%d %H:%M:%S"),
                "plan_id": int(msg.plan_id),
                "t0": float(msg.t0), "dt": float(msg.dt),
                "n_knots": int(p.shape[1]),
                "status": str(verdict.status),
                "reasons": [str(x) for x in verdict.reasons],
                "margins": {str(k): (float(v) if math.isfinite(float(v))
                                     else None)
                            for k, v in dict(verdict.margins).items()},
                "consec_rejects": int(verdict.consec_rejects),
                "anchor": {
                    "eta": [float(x) for x in self._eta[:3]],
                    "yaw": float(self._eta[5]),
                    "r_now": [float(x) for x in r_now[0]] + [float(r_now[1])],
                    "bridge_tier": (self._bridge.tier if self._bridge
                                    else SB.TIER_NONE)},
                # The RAW plan (pre-filter). Full knots when small (a streamed
                # 4 s window); a one-shot whole-demo plan is summarized by its
                # ends + a hash so the line stays one line.
                "raw": ({"p_ned": p.tolist(),
                         "yaw": np.asarray(msg.yaw, float).tolist()}
                        if small else
                        {"p_first": [float(x) for x in p[:, 0]],
                         "p_last": [float(x) for x in p[:, -1]],
                         "sha1": hashlib.sha1(
                             p.tobytes()).hexdigest()[:12]}),
            }
            with open(self._run_dir() / "plans.jsonl", "a",
                      encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except (OSError, TypeError, ValueError) as e:      # noqa: BLE001
            self.bus.log.emit("warn", f"mpc: plans.jsonl not written ({e})")

    def _plan_stream_meta(self) -> dict:
        """ALWAYS written, the imu_dr rule: an absent key cannot tell 'this
        build had no plan stream' from 'no replay was flown'."""
        if self.cfg is None:
            return {"enabled": False}
        r = self.cfg.replay
        rp = self.replay if self.replay is not None else self._replay_last
        out = {"enabled": rp is not None,
               "config": {k: r.get(k) for k in
                          ("session", "v_max_m_s", "blend_s",
                           "stream_period_s", "horizon_s", "anchor_max_m",
                           "jump_max_m", "workspace_box_ned", "gripper")}}
        if rp is not None:
            out["run"] = {k: rp.get(k) for k in
                          ("session", "poses_sha1", "n_plans", "released",
                           "installed", "clipped", "rejected", "period_s",
                           "time_dilation", "duration_s", "gripper_on",
                           "grip_events")}
        return out

    # ------------------------------------------------------- frame crossings
    def _datum_yaw0(self) -> float:
        return float(self._datum["yaw0"]) if self._datum is not None else 0.0

    def _datum_to_map_p(self, p):
        """Datum-frame position -> MAP (tag-world). The inverse of
        :meth:`_datumize`'s translation+rotation, in three dimensions."""
        d = self._datum
        v = np.asarray(p, float).reshape(3)
        if d is None:
            return v.copy()
        return np.asarray(d["p0"], float) + np.asarray(d["Rz"], float).T @ v

    def _map_to_datum_p(self, p):
        d = self._datum
        v = np.asarray(p, float).reshape(3)
        if d is None:
            return v.copy()
        return np.asarray(d["Rz"], float) @ (v - np.asarray(d["p0"], float))

    def _map_to_datum_v(self, v):
        """A VELOCITY, so the datum translation must not apply — only the
        rotation. Sharing `_map_to_datum_p` here would add p0 to a speed."""
        d = self._datum
        w = np.asarray(v, float).reshape(3)
        if d is None:
            return w.copy()
        return np.asarray(d["Rz"], float) @ w

    def _map_to_datum_yaw(self, yaw_map) -> float:
        return _wrap_pi(float(yaw_map) - self._datum_yaw0())

    def _anchor_dr(self) -> None:
        """Start dead reckoning HERE — the end of the settle, before the shape
        branch, so all three missions anchor at the same point in the flow.

        Once only, by the operator's decision (2026-08-16): no re-anchoring
        for the rest of the run, so what the plot shows is one continuous
        estimate diverging from one known point. Re-anchored drift statistics
        (a p50/p95 over many short windows) come out of the raw JSONL offline
        instead, where they cost no in-flight complexity.
        """
        if self.dr is None or self._eta is None:
            return
        R = rot_zyx(*[float(v) for v in self._eta[3:6]])
        # The autopilot's own body rates over the same window, so a vehicle
        # that is station-keeping rather than truly still does not get its
        # real rotation absorbed into "gyro bias".
        # The autopilot's rates over the WHOLE settle window, as a series.
        # The single latest sample used to be passed here, which is wrong
        # twice over: the bias is a mean-vs-mean quantity, and a constant
        # cannot remove the time-varying rotation the precision gate has to
        # see past (imu_dr.calibrate_static explains the measured difference).
        # ...and the same window's roll/pitch, because the accel offset is a
        # per-sample residual: a station-keeping vehicle moves its attitude
        # while it holds its position, so pinning the whole window's mean
        # specific force to one attitude leaks the difference in as bias.
        gyro_ref = att_ref = None
        if len(self._vimu_hist) >= 4:
            hist = np.asarray(self._vimu_hist, float)
            gyro_ref = hist[:, :4]
            att_ref = hist[:, [0, 4, 5]]
        out = self.dr.calibrate_static(R, gyro_ref=gyro_ref, att_ref=att_ref)
        if not out.get("ok"):
            self.bus.log.emit(
                "warn", f"imu_dr: static calibration skipped — "
                        f"{out.get('why', 'unknown')}. The estimate will carry "
                        f"the raw accelerometer offset (measured 1.8 m/s^2).")
        # Velocity anchors at ZERO: after a settle the vehicle is supposed to
        # be stopped, and the tag velocity is a low-passed difference carrying
        # ~44 ms of lag whose 0.01 m/s error would be 0.1 m by 10 s on its own.
        v = float(np.linalg.norm(np.asarray(self._last_nu, float)[:2])) \
            if self._last_nu is not None else 0.0
        self.dr.anchor(self._eta, t=now(), zero_velocity=True)
        self._event(f"imu_dr ANCHORED (settle speed {v * 100:.1f} cm/s, "
                    f"gyro bias from {self.dr.gyro_bias_source}, "
                    f"accel offset vs {self.dr.accel_ref_source})")
        if v > 0.02:
            self.bus.log.emit(
                "warn", f"imu_dr: anchored at {v * 100:.1f} cm/s but the "
                        f"velocity was zeroed anyway — the settle did not "
                        f"stop the vehicle, so the run starts with that much "
                        f"velocity error ({v * 10 * 100:.0f} cm by 10 s).")

    def _arm_path(self, sq, shape, origin_xy, yaw_fixed, depth) -> None:
        self._anchor_dr()
        self.follow = None                # 4 of 4: see MpcWorker.follow
        self._clear_replay()              # never inherit a streamed plan (and
                                          # never leave a jaw drive latched)
        if shape == "replay":
            self.station = None
            self._approach = None         # anchored at the current pose:
            self._path_cursor = None      # nothing to approach (follow's rule)
            self._path_err = (None, None)
            self._arm_replay(sq)
            return
        if shape == "follow":
            self.station = None
            self._approach = None      # nothing to approach: see set_traj
            self._path_cursor = None
            self._path_err = (None, None)
            self._arm_follow(sq)
            return
        if shape == "station":
            # No trajectory at all: sit on the tag and hold, heading included.
            # This is the mission to fly FIRST — if the vehicle cannot hold a
            # point it certainly cannot track a line, and every number the
            # line run would produce would be measuring that instead.
            self.ctrl.set_target_ned((origin_xy[0], origin_xy[1], depth),
                                     yaw_fixed)
            self._path_cursor = None
            self._path_err = (None, None)
            yaw_map = math.degrees(_wrap_pi(
                yaw_fixed + (self._datum["yaw0"] if self._datum else 0.0)))
            self.station = {
                "kind": "station",
                "origin_ned": [float(origin_xy[0]), float(origin_xy[1])],
                "depth_ned": float(depth),
                "yaw_fixed_ned_deg": math.degrees(yaw_fixed),
                "yaw_map_deg": yaw_map,
                "origin_tag": (int(sq["origin_tag"])
                               if sq.get("origin_tag") else None)}
            self.traj_on = False
            self.phase, self.phase_detail = "station", ""
            what_short = ("tag %d" % self.station["origin_tag"]
                          if self.station["origin_tag"] else "this point")
            what = (f"STATION hold at "
                    + (f"tag {self.station['origin_tag']}"
                       if self.station["origin_tag"] else "this point")
                    + f", heading {yaw_map:+.0f} deg (map)")
            self.reason = "station hold"
            self._event(f"on station: {what_short}")
            self._log_event(f"STATION - {what}")
            self.bus.log.emit("warn", f"mpc: {what}")
            return
        if shape == "line":
            # dir_deg is a MAP heading (90 = the pool's +y). The reference is
            # built in the DATUM frame, so it has to be rotated by the engage
            # yaw — without this the line ran along datum +y, which is a
            # different physical direction every run (operator: "+y is the
            # pool's long side but it drew along -x", 2026-08-14).
            sq = dict(sq)
            sq["dir_deg"] = math.degrees(_wrap_pi(
                math.radians(float(sq.get("dir_deg", 90.0)))
                - (self._datum["yaw0"] if self._datum else 0.0)))
            scen = self.ctrl.set_line_ned(sq, origin_xy, yaw_fixed, depth)
            scen["dir_map_deg"] = float(
                (self._scenario_override.get("dir_deg")
                 if "dir_deg" in self._scenario_override
                 else self.cfg.square.get("dir_deg", 90.0)))
            what = (f"LINE {scen['length']:.2f} m @ {scen['speed']} m/s, "
                    f"{scen['laps']} round trip(s), dir {scen['dir_deg']:.0f} deg")
        elif shape == "circle":
            # The entered tag is a point ON THE RIM, not the centre (operator,
            # 2026-08-17): "the tag is the bottom of the circle and you pick
            # the radius". rot_deg swings the CENTRE around that tag and is a
            # MAP rotation for the same reason the rectangle's is — it is
            # datumized here so rot 0 always means the same physical circle,
            # centre at +x of the tag, tag at the bottom of the plot.
            sq = dict(sq)
            rot_map = float(sq.get("rot_deg", 0.0))
            sq["rot_deg"] = math.degrees(_wrap_pi(
                math.radians(rot_map)
                - (self._datum["yaw0"] if self._datum else 0.0)))
            scen = self.ctrl.set_circle_ned(sq, origin_xy, yaw_fixed, depth)
            scen["rot_map_deg"] = rot_map
            what = (f"CIRCLE r {scen['radius']:.2f} m @ {scen['speed']} m/s, "
                    f"{scen['laps']} lap(s), tag on the rim, "
                    f"heading_follow={scen['heading_follow']}")
        else:
            # rot_deg is a MAP rotation, exactly like the line's dir_deg: 0
            # means the sides are parallel to the tag map's x and y axes, and
            # the entered tag is the min-x/min-y corner. The reference is
            # built in the DATUM frame, so subtract the engage yaw — without
            # this the rectangle came out tilted by however the vehicle
            # happened to be pointing at START (operator screenshot,
            # 2026-08-14).
            sq = dict(sq)
            rot_map = float(sq.get("rot_deg", 0.0))
            sq["rot_deg"] = math.degrees(_wrap_pi(
                math.radians(rot_map)
                - (self._datum["yaw0"] if self._datum else 0.0)))
            scen = self.ctrl.set_square_ned(sq, origin_xy, yaw_fixed, depth)
            scen["rot_map_deg"] = rot_map
            what = (f"SQUARE {scen['size']:.2f} x "
                    f"{scen.get('size_y', scen['size']):.2f} m @ "
                    f"{scen['speed']} m/s, {scen['laps']} lap(s), "
                    f"heading_follow={scen['heading_follow']}")
        # ONE curve for every controller. MPCC built its own inside the
        # bridge (it needs it as an optimizer parameter); a PID gets the same
        # ArcPath through a projection cursor, so "PID vs MPCC" is a
        # controller comparison and not a geometry comparison.
        self._path_cursor = None
        self._path_err = (None, None)
        if hasattr(self.ctrl, "set_path_plan_ned"):
            self.ctrl.set_path_plan_ned(None)     # never inherit a stale plan
        self._path_depth = float(depth)
        self._path_yaw_fixed = float(yaw_fixed)
        self._path_heading_follow = bool(scen.get("heading_follow", False))
        path = getattr(self.ctrl, "path", None)
        if self.cfg.path_following and not hasattr(self.ctrl, "progress_m"):
            from .path_geometry import PathCursor, path_from_scenario

            try:
                path = path_from_scenario(
                    scen, fillet_m=self.cfg.path_fillet_m,
                    turn_radius_m=self.cfg.path_turn_radius_m)
                s_grid, v_grid = path.speed_profile(
                    float(scen.get("speed", 0.05)),
                    self.cfg.path_lat_accel_m_s2,
                    self.cfg.path_long_accel_m_s2,
                    v_creep=self.cfg.path_creep_m_s)
                self._path_cursor = PathCursor(path, self.cfg.path_lead_m,
                                               s_grid, v_grid)
            except (KeyError, TypeError, ValueError) as e:
                self._path_cursor = None
                self.ctrl.set_target_ned(self._eta[:3], self._eta[5])
                self._refuse(f"invalid path geometry ({e})")
                return
        nominal_s = float(scen.get("T_run_s", 0.0))
        timeout_factor = float(self.cfg.traj_timeout_factor)
        if path is not None:
            # The wall-clock backstop is sized on the FEASIBLE traversal of the
            # actual curve (the speed profile already encodes cornering), not
            # on perimeter/speed, which no longer describes what is flown.
            s_g, v_g = path.speed_profile(
                float(scen.get("speed", 0.05)), self.cfg.path_lat_accel_m_s2,
                self.cfg.path_long_accel_m_s2,
                v_creep=self.cfg.path_creep_m_s)
            # time = integral of ds/v along the curve. Written out rather than
            # via np.trapz/np.trapezoid: the first is gone in numpy 2.x and the
            # second does not exist in the <2.0 this env is pinned to for gtsam
            # (memory: environment-numpy-constraint), and this station is
            # imported from both.
            v_mid = np.maximum(0.5 * (v_g[:-1] + v_g[1:]), 1e-3)
            minimum_s = float(np.sum(np.diff(s_g) / v_mid))
            timeout_s = minimum_s * max(1.0, timeout_factor)
            scen["path_min_duration_s"] = minimum_s
            # WHOSE NUMBER IS THE SPEED? On a polygon the straights run at the
            # operator's v_cmd and only the corners are slower, so the box on
            # the panel is the mission speed. On a CIRCLE the curvature limit
            # v <= sqrt(a_lat*R) applies to the WHOLE lap, so a small radius
            # silently overrides the box everywhere — a 0.3 m circle at the
            # shipped a_lat 0.05 caps at 0.12 m/s however fast you ask. That
            # has to be said out loud: the 2026-08-17 session already lost a
            # run to a speed box that was not in the loop, and a number the
            # operator typed that the geometry then ignores is the same
            # failure wearing a different hat.
            v_cap = float(np.max(v_g))
            v_cmd = float(scen.get("speed", 0.05))
            scen["path_v_max_m_s"] = v_cap
            if v_cap < 0.97 * v_cmd:
                self.bus.log.emit(
                    "warn",
                    f"mpc: curvature caps this path at {v_cap:.3f} m/s — the "
                    f"{v_cmd:.3f} m/s you asked for is not flyable on it "
                    f"(v <= sqrt(path_lat_accel_m_s2 * R)); the run will take "
                    f"{minimum_s:.0f} s")
        else:
            timeout_s = nominal_s * timeout_factor
        scen["path_timeout_s"] = timeout_s
        if sq.get("origin_tag") is not None:
            scen["origin_tag"] = int(sq["origin_tag"])
            what += f", origin tag {int(sq['origin_tag'])}"
        self.traj_on = True
        self.station = None
        self._t0_traj = now()
        self._tau, self._tau_t, self._path_lag = 0.0, now(), 0.0
        self.phase, self.phase_detail = shape, ""
        self.reason = f"{shape} running"
        self._event(f"{shape.upper()} started")
        self._log_event(f"{shape.upper()} started - {what}")
        self.bus.log.emit("warn", f"mpc: {what}")

    def _mission_origin(self, sq: dict):
        """Where the path starts, in the DATUM frame.

        Three ways to say it, in priority order: ``origin_tag`` (a tag id —
        the path is anchored to a physical tag, which is how the operator
        thinks about it), an explicit ``origin`` pair in the tag frame, or
        "current" = wherever the vehicle is now. The first two are TAG-frame
        and get datumized here, because everything downstream (controller,
        plot, CSV) lives in the engage-datum frame."""
        tag = sq.get("origin_tag")
        if tag not in (None, "", 0) or (tag == 0 and "origin_tag" in sq):
            tmap = self._tag_map()
            poses = None if tmap is None else tmap.instances.get(int(tag))
            if not poses:
                self._refuse(f"tag {tag} is not in the map")
                return None
            if len(poses) > 1:
                self._refuse(f"tag {tag} is at {len(poses)} places — ambiguous")
                return None
            p_tag = np.asarray(poses[0][1], float)
            eta_tag = np.zeros(6)
            eta_tag[0:3] = (p_tag[0], p_tag[1], 0.0)
            d = self._datumize(eta_tag)
            return (float(d[0]), float(d[1]))
        origin = sq.get("origin", "current")
        if origin in ("current", None):
            return (float(self._eta[0]), float(self._eta[1]))
        eta_tag = np.zeros(6)
        eta_tag[0:2] = (float(origin[0]), float(origin[1]))
        d = self._datumize(eta_tag)
        return (float(d[0]), float(d[1]))

    def _to_map_xy(self, x, y):
        """Datum-frame xy -> MAP (tag-world) xy."""
        d = self._datum
        if d is None:
            return float(x), float(y)
        c, s = math.cos(d["yaw0"]), math.sin(d["yaw0"])
        return (float(d["p0"][0]) + c * x - s * y,
                float(d["p0"][1]) + s * x + c * y)

    def _tag_map(self):
        if self._tagmap is None and self.nav_cfg.geometry == "floor":
            try:
                from .tagnav import TagMap
                self._tagmap = TagMap.load(self.nav_cfg.tag_map_path)
            except (OSError, ValueError) as e:                   # noqa: BLE001
                self.bus.log.emit("error", f"mpc: tag map unreadable ({e})")
        return self._tagmap

    def disengage(self, why: str) -> None:
        was = self.engaged
        self.engaged = False
        self.traj_on = False
        if self._bridge is not None:
            self._bridge.reset()
        self._bridge_anchor = None
        self._axes_prev = None          # never ramp from a stale command
        self._auto_traj = False
        self._t0_traj = None
        self._path_cursor = None
        self._path_err = (None, None)
        self._path_depth = 0.0
        self._path_yaw_fixed = 0.0
        self._path_heading_follow = False
        self._approach = None
        self.station = None
        self.follow = None                # 2 of 4: see MpcWorker.follow
        self._clear_replay()
        self.phase, self.phase_detail = "", ""
        if was:
            # One explicit neutral so the vehicle stops NOW; the window then
            # resumes the teleop pump, and the sink deadman covers the gap.
            self.bus.cmd_pilot.emit(PilotInput(source="mpc"))
            self.reason = f"disengaged: {why}"
            # `debug` for the same reason the ENGAGE copy is: the mission event
            # on the next line is the one the operator reads, and two lines for
            # one disengage is what made the log hard to scan.
            self.bus.log.emit("debug", f"mpc: DISENGAGED — {why}")
            self._event(f"DISENGAGED: {why}")
            self._log_event(f"DISENGAGED - {why}")
            if self._csv_auto:
                self._close_csv()

    @Slot()
    def estop(self) -> None:
        self.disengage("E-STOP")

    # ------------------------------------------------------------------ tick
    def tick(self) -> None:
        if not self._ready:
            return
        t = now()
        dt = 1.0 / self.cfg.ctrl_hz
        meas, health = self.asm.step(self.fix, self.imu, t, dt)
        if meas is not None and self._datum is not None:
            meas["eta"] = self._datumize(meas["eta"])
        # A FOLLOW THAT LOSES THE TAG IS DEMOTED FIRST, then bridged.
        # Three reasons, and the order matters:
        #   1. the bridge's own premise excludes this mission — its docstring
        #      says the ladder "assumes the vehicle is nominally at rest over
        #      one point", and a follow is structurally a moving one;
        #   2. losing the tag loses the OBJECT too. T_map_obj is composed FROM
        #      a NavFix, so with no fix there is nothing to follow. Carrying
        #      the vehicle on an IMU while the target is unobservable is
        #      subtracting two drifting quantities — the worst option
        #      available;
        #   3. the argument for not DISENGAGING is unchanged, so it does not.
        # The moment `self.station` is non-None the existing ladder picks it
        # up exactly as it always has, and station_bridge.py itself is
        # untouched.
        if (self.engaged and self.follow is not None and meas is None
                and SB.is_bridgeable(health.get("why", ""))):
            self._follow_to_station("tag fix lost")
        # A TAG DROPOUT IS NOT A REASON TO STOP CONTROLLING (station only).
        # Substituted into `meas` on purpose — that is what `_runtime_fault`
        # reads, so the stale-fix interlock stops firing while every other
        # interlock keeps working. See station_bridge.py.
        meas = self._station_bridge(meas, health, t, dt)
        self._last_health = health
        u = np.zeros(6)
        info = {}
        axes = None
        t_traj = None
        if meas is not None:
            self._eta = meas["eta"]
            self._last_nu = meas["nu"]

        # Dead reckoning, every tick and independent of engagement, so the
        # settle window is already full when _arm_path anchors. `meas` stays
        # the TAG state throughout: the plot's actual trail, the CSV's px/py
        # and every interlock read ground truth even when the controller is
        # flying on the estimate. Only `meas_ctrl` diverges.
        meas_ctrl = meas
        if self.dr is not None:
            self._drain_dr()
            self.dr.note_depth(*self._dr_depth())
            self._dr_last = self.dr.state(t, meas_tag=meas,
                                          imu_vehicle=self.imu)
            self.dr.note_tick(dt)
            if self.dr_control and self._dr_last["ok"]:
                meas_ctrl = self._dr_last["meas"]

        if self.engaged:
            why = self._runtime_fault(meas, health, t)
            if why:
                self.disengage(why)
            elif meas is None:
                # Debouncing a brief staleness (see _runtime_fault): there is
                # no state to control on, so this tick commands NOTHING and
                # the sink's 500 ms deadman keeps the vehicle honest. What we
                # do NOT do is run the controller on a stale or absent state.
                self._tau_t = t
                self.phase_detail = f"waiting for a fix ({self._stale_ticks})"
            else:
                if self._warmup_left > 0:
                    self._warmup_left -= 1
                if self._warmup_left > 0:
                    self.phase = "warmup"
                    self.phase_detail = (
                        f"{self._warmup_left / self.cfg.ctrl_hz:.1f} s")
                elif self.phase == "warmup":
                    self.phase, self.phase_detail = "", ""
                if self._approach is not None:
                    self._tick_approach(t)
                if self.follow is not None:
                    # Before the controller step, so the setpoint this tick
                    # issues is the one this tick's object estimate implies.
                    self._tick_follow(t)
                if (self._auto_traj and self._warmup_left <= 0
                        and not self.traj_on):
                    # One attempt only: a refusal (square outside the fence)
                    # must not retry at 20 Hz — the operator repositions and
                    # presses START again.
                    self._auto_traj = False
                    self.set_traj(True)
                # BOTH of these take meas_ctrl, and they have to agree: the
                # path cursor projects the vehicle onto the active segment, so
                # advancing it on the truth while the controller flies on the
                # estimate would measure neither one.
                coasting = (self._bridge is not None
                            and self._bridge.tier == SB.TIER_COAST)
                if coasting:
                    self._coast_retarget(meas_ctrl)
                t_traj = self._advance_path_clock(t, meas_ctrl)
                if self.replay is not None and self.traj_on:
                    # After the clock, before the step: the plan the solver
                    # consumes this tick is sampled at this tick's t_traj.
                    # GUARDED: a poisoned replay raising at 20 Hz would
                    # otherwise abort the rest of tick() every time — no
                    # step, no watchdogs, no CSV — while engaged=True and
                    # the jaw possibly latched (safety review 2026-08-30).
                    # A broken mission gets the same escalation as any other.
                    try:
                        self._tick_replay(t_traj)
                    except Exception as e:                       # noqa: BLE001
                        self.disengage(f"replay tick raised: "
                                       f"{type(e).__name__}: {e}")
                if not self.engaged:
                    # the replay guard disengaged us mid-tick; the neutral is
                    # already out (disengage), so publish/record and stop.
                    self._publish(meas, health, u, info, axes, t_traj)
                    self._object_heartbeat(t)
                    self._write_row(meas, health, u, info, axes, t_traj, t)
                    return
                # WALL TIME, MEASURED HERE. Every controller used to report
                # its own, and none of them reported the truth: HwDobMpc and
                # HwMpcc both prefer acados' `time_tot`, which times only the
                # QP it managed to solve (the 2026-08-23 tick that blocked
                # 6.5 s inside the failure path recorded 1.15 ms), and HwPid
                # returns the constant 0.05. One bracket around the step
                # covers all three and cannot be gamed by the thing it times.
                _t_step = time.perf_counter()
                u, info = self.ctrl.step(meas_ctrl["eta"], meas_ctrl["nu"],
                                         meas_ctrl["nudot"], t_traj)
                if coasting:
                    u = self._bridge.release_horizontal(u)
                # MPCC decides its own progress rate; keep it for the readout.
                if "v_theta" in info:
                    self._mpcc_v_theta = float(info["v_theta"])
                self._note_w_hat_rail(info.get("w_hat"))
                nf = info.get("n_fail", 0)
                failed_now = nf > self._nfail_prev
                self._fail_streak = self._fail_streak + 1 if failed_now else 0
                self._nfail_prev = nf
                self._tick_ms = 1e3 * (time.perf_counter() - _t_step)
                if failed_now:
                    # Timestamped, with w_hat — the exact logging KNOWN_ISSUES
                    # asked for before any hardware run. The MEASURED time
                    # rides along: 2026-08-23's failure line said nothing about
                    # the 6.5 s the same call had just spent.
                    self.bus.log.emit(
                        "error", f"mpc: solver FAILURE status "
                                 f"{info.get('status')} at t={t_traj:.2f}s "
                                 f"(n_fail={nf}, tick {self._tick_ms:.0f} ms, "
                                 f"|w_xyz|="
                                 f"{np.linalg.norm(self.ctrl.w_hat[:3]):.1f} N)")
                budget = float(self.cfg.engage["tick_overrun_ms"])
                over = not (self._tick_ms <= budget)   # NaN counts as over
                self._over_streak = self._over_streak + 1 if over else 0
                if self._fail_streak >= int(self.cfg.engage["max_solver_fails"]):
                    self.disengage(f"solver failed {self._fail_streak}x")
                elif over and self.follow is not None:
                    # A RUNG BEFORE THE DROP. On a follow the reference is the
                    # most likely thing that broke, and disengaging cures it by
                    # dropping depth hold on a -5.7 N vehicle. Demote first:
                    # `_follow_to_station` re-targets with no feedforward, i.e.
                    # it removes the reference that caused this.
                    self._follow_to_station(
                        f"tick {self._tick_ms:.0f} ms > budget")
                elif self._over_streak >= 2:
                    self.disengage(f"tick {self._tick_ms:.0f} ms > budget "
                                   f"({self._over_streak}x)")
                else:
                    axes = wrench_to_axes(u, self.cfg.axis_gain,
                                          self._axis_cap())
                    axes = slew_axes(axes, self._axes_prev,
                                     self.cfg.axis_slew_per_s,
                                     1.0 / self.cfg.ctrl_hz)
                    self._axes_prev = (axes.surge, axes.sway, axes.heave,
                                       axes.yaw)
                    # The EAOB is told what ACTUALLY went out — the slew limit
                    # is a real actuator limit like the cap, so crediting the
                    # unlimited wrench would book the difference as
                    # disturbance (axes_to_wrench's docstring).
                    self.ctrl.note_applied(axes_to_wrench(axes,
                                                          self.cfg.axis_gain))
                    self.bus.cmd_pilot.emit(axes)
                    self._watch_actuation(axes, t)
                    wall_traj = (t - self._t0_traj
                                 if self._t0_traj is not None else 0.0)
                    T_run = (self.ctrl.scenario or {}).get("T_run_s", 0.0)
                    timeout_s = (self.ctrl.scenario or {}).get(
                        "path_timeout_s",
                        T_run * float(self.cfg.traj_timeout_factor))
                    if (self.traj_on and self.ctrl.scenario
                            and wall_traj > timeout_s):
                        # A spatial cursor/corner gate can wait forever for a
                        # vehicle that cannot capture the path. This is the
                        # wall-clock backstop, and it is a FAULT, not success.
                        self.set_traj(False)
                        self._event(f"{(self.ctrl.scenario or {}).get('kind','path')} "
                                    f"TIMED OUT (path progress {t_traj:.0f}s of "
                                    f"{T_run:.0f}s after {wall_traj:.0f}s; "
                                    f"limit {timeout_s:.0f}s)")
                        self.reason = "path timed out (DP hold)"
                    elif (self.traj_on and self.ctrl.scenario
                          and self._path_done(t_traj, T_run)):
                        kind = self.ctrl.scenario.get("kind", "square")
                        self.set_traj(False)
                        self._event(f"{kind.upper()} complete")
                        self.reason = f"{kind} complete (DP hold)"
                        self.bus.log.emit("warn", f"mpc: {kind} COMPLETE — "
                                                  "holding the final pose")

        self._publish(meas, health, u, info, axes, t_traj)
        self._object_heartbeat(t)
        self._write_row(meas, health, u, info, axes, t_traj, t)

    def _object_heartbeat(self, t: float) -> None:
        """Age the object marker even when the TRACKER goes silent.

        An ObjectFix is only published when a pose arrives, so a PoseWorker
        that dies would leave a confident diamond frozen on the plot forever —
        the same failure mode `bus.Freshness` exists to stop, and the same one
        the dead reckoner's `dr_ok` had to be made loud about. The control
        side is already safe (``_tick_follow`` re-predicts every tick); this
        is for the operator's eyes. 2 Hz, and only once the estimate is no
        longer live, so a healthy run publishes nothing extra.
        """
        a = self._obj
        prev = self._obj_last
        if a is None or prev is None or a.state(t) == ON.LIVE:
            return
        if t - float(prev.stamp) < 0.5:
            return
        pair = (None if prev.pair_dt_ms is None
                else (float(prev.pair_dt_ms) / 1e3, bool(prev.pair_exact)))
        self._emit_object(t, prev.t_capture, prev.pose_state, "", pair)

    def _watch_actuation(self, axes, t) -> None:
        """Commands leaving, motors silent — say so LOUDLY. Found live
        2026-08-12: 24 s of sustained ~0.2 axes with every thruster parked at
        1500 us and nothing on screen naming the mismatch (likely SYSID or
        the vehicle-side pilot gain). Warn, do not disengage: a vehicle that
        ignores MANUAL_CONTROL is stationary, and the pilot may be mid-fix."""
        strong = max(abs(axes.surge), abs(axes.sway), abs(axes.heave),
                     abs(axes.yaw)) > 0.12
        self._cmd_active_ticks = self._cmd_active_ticks + 1 if strong else 0
        if self._cmd_active_ticks < int(2.0 * self.cfg.ctrl_hz):
            return
        dev = self._pwm_dev_us()
        if dev is not None and dev < 8.0 and t - self._last_noresp_warn > 10.0:
            self._last_noresp_warn = t
            msg = (f"mpc: commands are leaving (axes ~"
                   f"{max(abs(axes.surge), abs(axes.sway)):.2f}) but every "
                   f"thruster sits at neutral (PWM dev {dev:.0f} us) — the "
                   f"vehicle is NOT acting on MANUAL_CONTROL. Check: armed? "
                   f"SYSID_MYGCS? vehicle pilot gain? QGC also connected?")
            self.bus.log.emit("error", msg)
            self._log_event(msg)

    def _runtime_fault(self, meas, health, t) -> str:
        if meas is None:
            why = health.get("why", "state lost")
            # DEBOUNCE the stale-fix gate. Measured 2026-08-14: tag_age is
            # dominated by CAMERA LATENCY, not by the fix rate — it sits at
            # 0.14 s (p95 0.19) under mpc/pid and 0.26 s (p95 0.40) under
            # dobmpc, whose acados solve spikes to 17-32 ms and slows the
            # shared process. Both dobmpc runs died on a SINGLE sample at
            # 0.51 / 0.54 s. One late frame is not a lost localizer, and every
            # other interlock here already debounces (max_solver_fails: 3).
            if "stale" in why:
                self._stale_ticks += 1
                need = max(1, int(float(self.cfg.engage.get(
                    "tag_stale_hold_s", 0.4)) * self.cfg.ctrl_hz))
                if self._stale_ticks < need:
                    return ""
            return why
        self._stale_ticks = 0
        tel = self.tel
        if tel is None or tel.armed is not True:
            return "vehicle disarmed"
        if t - float(tel.stamp) > 2.0:
            return "vehicle telemetry stale"
        want = str(self.cfg.engage.get("require_mode", "MANUAL")).upper()
        if want and not (tel.mode or "").upper().startswith(want):
            return f"flight mode left {want}"
        # No position abort. Until 2026-08-14 leaving the geofence box (plus a
        # margin) disengaged the controller here; the fence was removed at the
        # operator's request, so the runtime faults are now exactly: disarm,
        # telemetry stale, flight mode left MANUAL, and tag-fix loss above.
        return ""

    # ------------------------------------------------------------- reporting
    def _publish(self, meas, health, u, info, axes, t_traj) -> None:
        s = MpcStatus(
            engaged=self.engaged, traj_on=self.traj_on,
            mode=self.cfg.mode if self.cfg else "",
            phase=self.phase, phase_detail=self.phase_detail,
            # A COPY: `self.follow` is rewritten every tick on this thread
            # while the GUI thread reads what was emitted, and state.py's
            # contract is that a payload is immutable once it is on the bus.
            # `self.station` needs no copy — it is built once and never
            # touched again.
            scenario=(dict(self.follow) if self.follow is not None else
                      self.station if self.station is not None else
                      self.ctrl.scenario if (self.ctrl and self.traj_on)
                      else None),
            solver=(self.ctrl.solver_kind if self.ctrl else ""),
            solver_status=int(info.get("status", 0)),
            solve_ms=info.get("solve_ms"),
            n_fail=int(info.get("n_fail", 0)),
            w_hat=tuple(float(v) for v in info.get("w_hat", ())),
            u_cmd=tuple(float(v) for v in u) if self.engaged else (),
            axes=((axes.surge, axes.sway, axes.heave, axes.yaw)
                  if axes is not None else ()),
            n_tags=(self.fix.n_tags if self.fix is not None else 0),
            tag_age_s=health.get("tag_age"),
            imu_age_s=health.get("imu_age"),
            warmup_left_s=self._warmup_left / (self.cfg.ctrl_hz if self.cfg else 20.0),
            datum=(None if self._datum is None else
                   (float(self._datum["p0"][0]), float(self._datum["p0"][1]),
                    float(self._datum["p0"][2]), self._datum["yaw0"])),
            t_traj=t_traj, reason=self.reason,
            conn=(Conn.ONLINE if self.engaged
                  else (Conn.CONNECTING if meas is not None else Conn.DEGRADED)),
            stamp=now())
        if meas is not None:
            px, py, pz = _flu_of_ned(*meas["eta"][:3])
            s.p_flu = (px, py, pz)
            s.yaw_flu_deg = math.degrees(-meas["eta"][5])
        s.rp_residual_deg = health.get("rp_residual_deg")
        s.rp_residual_rp_deg = health.get("rp_residual_rp_deg")
        # THE FOLLOW LOOP's own three numbers. The OBJECT's position is
        # deliberately NOT here — it rides `bus.object_fix` in the map frame,
        # because it exists with nothing engaged and must not move on screen
        # when START is pressed.
        if self.follow is not None:
            s.follow_state = str(self.follow.get("state", "following"))
            s.follow_age_s = self.follow.get("age_s")
            s.follow_err_m = self.follow.get("err_m")
        elif self.station is not None and self.station.get("from_follow"):
            s.follow_state = "lost"
        if meas is not None:
            v_w = rot_zyx(*meas["eta"][3:6]) @ np.asarray(meas["nu"], float)[:3]
            s.speed_m_s = float(math.hypot(v_w[0], v_w[1]))
        s.ref_speed_m_s = self._ref_speed()
        self._publish_dr(s, meas)
        if self.engaged and self.ctrl is not None:
            p_ref, yaw_ref, _v_ref = self.ctrl.ref_ned_at(t_traj or 0.0)
            rx, ry, rz = _flu_of_ned(*p_ref)
            s.ref_flu = (rx, ry, rz)
            if meas is not None:
                s.err_xy = math.hypot(meas["eta"][0] - p_ref[0],
                                      meas["eta"][1] - p_ref[1])
                s.err_along, s.err_cross = self._path_split(meas, p_ref,
                                                            t_traj)
            if self.traj_on and self.ctrl.scenario:
                s.lap = self._path_lap(t_traj)
        self.bus.mpc_status.emit(s)

    def _publish_dr(self, s: MpcStatus, meas) -> None:
        """Fill the dead-reckoning half of a status row.

        The plot may only draw what this says, so every field here is either a
        real value or None — including on the unhappy paths. A DR that stopped
        receiving samples must arrive as ``dr_ok=False`` with a reason, never
        as a stale position that looks like a vehicle holding perfectly still.
        """
        if self.dr is None:
            return
        d = self.cfg.imu_dr
        s.dr_source = str(d.get("source", "c3"))
        s.dr_attitude = str(d.get("attitude", ""))
        s.bridge_tier = (self._bridge.tier if self._bridge is not None
                         else SB.TIER_NONE)
        s.bridge_s = (self._bridge.elapsed if self._bridge is not None
                      else 0.0)
        s.dr_mode = "control" if self.dr_control else "shadow"
        st = self._dr_last
        if st is None:
            s.dr_note = "no state yet"
            return
        s.dr_hz = st["hz"]
        s.dr_n = int(st["n"])
        s.dr_ok = bool(st["ok"])
        s.dr_elapsed_s = st["elapsed"]
        notes = []
        if st["why"]:
            notes.append(st["why"])
        if self._dr_overflow:
            notes.append(f"{self._dr_overflow} batches dropped (queue full)")
        if st["rejected"]:
            notes.append(f"{st['rejected']} samples out of dt range")
        if self.dr.static_note:
            notes.append(self.dr.static_note)
        s.dr_note = "; ".join(notes)
        if not st["ok"]:
            return
        p = st["p_ned"]
        px, py, pz = _flu_of_ned(float(p[0]), float(p[1]), float(p[2]))
        s.p_dr_flu = (px, py, pz)
        s.yaw_dr_flu_deg = math.degrees(-float(st["yaw"]))
        if meas is not None:
            s.dr_err_m = math.hypot(float(p[0]) - meas["eta"][0],
                                    float(p[1]) - meas["eta"][1])
            s.dr_err_z_m = float(st["pz_imu"]) - float(meas["eta"][2])

    # ------------------------------------------------------------------- CSV
    @Slot(bool, str)
    def set_sensor_log(self, on: bool, stem: str) -> None:
        """Rides the shared recording stem, like the other sensor logs."""
        if on and self._csv is None:
            # The stem already lives in the run folder (the depth recording
            # that owns it opened there), so pin THAT rather than re-resolving.
            stem_path = Path(f"{stem}_mpc")
            self._run_dir_pin = stem_path.parent
            self._open_csv(stem_path, auto=False)
        elif not on and self._csv is not None and not self._csv_auto:
            self._close_csv()

    def _open_csv(self, stem: Path, auto: bool) -> None:
        try:
            stem.parent.mkdir(parents=True, exist_ok=True)
            path = stem.with_suffix(".csv")
            self._csv = open(path, "w", encoding="utf-8")
            self._csv.write(CSV_HEADER)
            self._csv_path = path
            self._csv_auto = auto
            self._t0_csv = now()
            # Wall clock, for meta.json. Separate from _t0_csv (monotonic, the
            # CSV's own t=0) because only the wall clock lines a run up against
            # a video file or a note.
            self._csv_started = time.strftime("%Y-%m-%d %H:%M:%S")
            self._rows = 0
            self._write_meta()
            self.bus.log.emit("info", f"mpc csv -> {path}")
            self._raw_logs(True, stem)
        except OSError as e:
            self._csv = None
            self.bus.log.emit("error", f"mpc csv: {e}")

    def _raw_logs(self, on: bool, stem: Path | None) -> None:
        """Ask the C3 and the vehicle to keep their RAW streams for this run.

        Only while the dead reckoner is enabled, because that is the only
        consumer that needs them and 40 kB/s of JSONL is not free. It is what
        makes `plot_imu_dr --from-jsonl` possible: the same pool run can be
        re-estimated afterwards with a different attitude mode, a different
        calibration or a different anchor, instead of costing another session.
        """
        if self.dr is None:
            return
        self.bus.cmd_log_raw_sensors.emit(bool(on),
                                          str(stem) if stem else "")

    def _close_csv(self) -> None:
        if self._csv is None:
            return
        self._raw_logs(False, None)
        try:
            self._csv.close()
        except OSError:
            pass
        self._write_meta()                    # final row count
        self.bus.log.emit("info", f"mpc csv closed: {self._rows} rows "
                                  f"({self._csv_path})")
        self._csv = None
        self._run_dir_pin = None   # see _run_dir()
        self._csv_auto = False

    def _plant(self) -> dict:
        """The M / C(nu) / D(nu) / g(eta) the controller believes in, cached.

        Read from the sim's params+fossen (plain numpy — no casadi), so a
        PID-only station whose acados build failed still records a full plant.
        Cached because only the STATE-dependent samples change, and re-reading
        the module every recording would be pure work; the sample is refreshed
        below from whatever the loop last assembled."""
        if self._plant_meta is None:
            from .plant import plant_meta
            try:
                self._plant_meta = plant_meta(
                    self.cfg.rov_model if self.cfg else "heavy_gripper")
            except Exception as e:                               # noqa: BLE001
                self._plant_meta = {"error": f"{type(e).__name__}: {e}",
                                    "note": "plant model could not be read; "
                                            "the run is NOT self-describing"}
                self.bus.log.emit("warn", f"mpc: plant model not recorded ({e})")
        return self._plant_meta

    def _plant_at_state(self) -> dict:
        """``_plant()``, re-evaluated at the state the loop is actually in."""
        base = self._plant()
        if "error" in base or self._eta is None:
            return base
        try:
            from .plant import plant_meta
            nu = self._last_nu
            return plant_meta(self.cfg.rov_model if self.cfg else "heavy_gripper",
                              nu=nu, eta=self._eta)
        except Exception:                                        # noqa: BLE001
            return base

    def _run_meta(self, trigger: str = "csv") -> dict:
        """EVERYTHING needed to reproduce this run, in one definition.

        ONE builder, two readers: the CSV's ``.meta.json`` sidecar and the
        ``controller.json`` a REC press drops into the run folder
        (:meth:`dump_run_meta`). Two copies of this dict would drift, and the
        drift would be invisible — the two files would simply disagree about
        which gains a run was flown with, with nothing to say which is right.
        """
        import hashlib

        nav = self.nav_cfg
        tag_map_sha = None
        if nav and nav.geometry == "floor":
            try:
                tag_map_sha = hashlib.sha1(
                    Path(nav.tag_map_path).read_bytes()).hexdigest()[:12]
            except OSError:
                pass
        return {
            "schema_version": 8,          # 2: + plant, mission, meta_trigger
                                          # 3: + imu_dr, hardware.cam_tilt_deg
                                          # 4: + circle; `trajectory` is now
                                          #    PER-SHAPE (a circle carries
                                          #    radius and NO size/size_y), so
                                          #    a reader must switch on
                                          #    trajectory.kind before touching
                                          #    a dimension key. `mission.
                                          #    config_square` keeps its name
                                          #    for continuity and is no longer
                                          #    only a rectangle.
                                          # 5: + station_bridge, and the CSV
                                          #    gains bridge_s / bridge_tier.
                                          #    Rows with bridge_s > 0 were NOT
                                          #    flown on the tag — see
                                          #    CSV_HEADER's note before pooling
                                          #    them with anything.
                                          # 6: + object_nav, the `follow`
                                          #    mission shape, and ten CSV
                                          #    columns (obj_* / follow_*).
                                          #    obj_pair_exact == 1 is the
                                          #    boundary: only those rows are
                                          #    extrinsic-free, so object
                                          #    positions must not be pooled
                                          #    across it.
                                          # 7: + CSV `tick_ms` (MEASURED wall
                                          #    time of the controller step;
                                          #    `solve_ms` is the solver's own
                                          #    account and both acados bridges
                                          #    under-report a stall), and a
                                          #    THREE-WAY follow boundary:
                                          #    yaw_axis "none" now really
                                          #    means none (it silently ran
                                          #    axis-pinned before), the
                                          #    feedforward is capped
                                          #    (controller.follow.ff_max_m_s),
                                          #    and the IPOPT recovery is off
                                          #    on hardware
                                          #    (controller.ipopt_fallback).
                                          #    Do not pool follow runs across
                                          #    it — the vehicle obeyed a
                                          #    different reference before.
                                          # 8: + plan_stream (ALWAYS written),
                                          #    the `replay` mission shape, CSV
                                          #    plan_id/ref_src/grip_cmd, and
                                          #    reference_clock.strategy
                                          #    "plan_stream_replay". A replay's
                                          #    reference is a streamed plan
                                          #    (recorded demo through the
                                          #    filter/stitcher), not a placed
                                          #    geometry — a NEW reference
                                          #    family; never pool replay runs
                                          #    with geometric missions. Raw
                                          #    per-plan verdicts: plans.jsonl.
            "source": "hardware rov_gui.control (real vehicle)",
            "meta_trigger": trigger,
            "written": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rov_model": self.cfg.rov_model if self.cfg else None,
            # STARTED, not "whenever this sidecar was last written". _write_meta
            # runs twice — once at open, once at close for the final row count —
            # and re-stamping here made every shipped meta.json claim the run
            # began at the moment it ENDED. Captured at _open_csv instead.
            "run": {"started": self._csv_started,
                    "log_hz": self.cfg.ctrl_hz if self.cfg else None,
                    "csv": (str(self._csv_path) if self._csv_path else None),
                    "rows": self._rows,
                    "engaged": bool(self.engaged), "traj_on": bool(self.traj_on),
                    "phase": self.phase, "reason": self.reason,
                    "frame_note": "px,py,pz / rx,ry,rz are world FLU "
                                  "(sim-compatible); w*/u* are NED body"},
            "controller": (self.ctrl.meta() if self.ctrl is not None else None),
            # The vehicle model the controller was flown against: M, C(nu),
            # D(nu), g(eta) and the coefficients that generate them. Added
            # 2026-08-14 at the operator's request — a CSV says what the vehicle
            # DID, this says what the controller thought it WAS.
            "plant": self._plant_at_state(),
            "trajectory": (self.ctrl.scenario if self.ctrl is not None else None),
            # What the PANEL was asking for, which is not the same thing: the
            # scenario above is null until a path is armed, and a REC press
            # before START would otherwise record no mission at all.
            "mission": {
                "config_square": (dict(self.cfg.square) if self.cfg else None),
                "panel_override": dict(self._scenario_override)},
            # HOW path progress/reference generation ran. Keep the historical
            # `reference_clock` key because offline plots already read it, but
            # strategy names the new spatial semantics unambiguously.
            "reference_clock": {
                "path_following": (bool(self.cfg.path_following)
                                   if self.cfg else None),
                "path_lead_m": (float(self.cfg.path_lead_m)
                                if self.cfg else None),
                # THE BOUNDARY between three different things a run could mean
                # by "path following". Never pool records across them.
                #   wall_clock_trajectory        be at p(t) at time t
                #   active_segment_projection_corner_gate  (2026-08-14..16)
                #        spatial cursor, full stop + dwell at every vertex
                #   mpcc_contouring              (2026-08-16 on) theta is a
                #        solver decision on a C1 filleted curve
                #   arc_projection_cursor        the PID on that same curve
                #   plan_stream_replay           (2026-08-30 on) a recorded
                #        demo streamed through PlanFilter/PlanStitcher into
                #        set_path_plan_ned — the diffusion-policy seam
                # Keyed on the ARMED scenario, not on _replay_last: a square
                # armed after a replay in the same engagement must not keep
                # wearing the replay label (scenario.kind is what flips).
                "strategy": (
                    "plan_stream_replay"
                    if (self.replay is not None
                        or (self.ctrl is not None
                            and (self.ctrl.scenario or {}).get("kind")
                            == "replay"))
                    else "mpcc_contouring"
                    if self.ctrl is not None
                    and hasattr(self.ctrl, "progress_m")
                    else (("waypoint_vertex_stop"
                           if self.cfg and self.cfg.path_fillet_m <= 1e-9
                           else "arc_projection_cursor")
                          if self.cfg and self.cfg.path_following
                          else "wall_clock_trajectory")),
                "fillet_m": (float(self.cfg.path_fillet_m)
                             if self.cfg else None),
                "turn_radius_m": (float(self.cfg.path_turn_radius_m)
                                  if self.cfg else None),
                "lat_accel_m_s2": (float(self.cfg.path_lat_accel_m_s2)
                                   if self.cfg else None),
                "long_accel_m_s2": (float(self.cfg.path_long_accel_m_s2)
                                    if self.cfg else None)},
            "disturbance": None,
            "hardware": {
                "geometry": nav.geometry if nav else None,
                "tag_map": (nav.tag_map_path if nav and nav.geometry == "floor"
                            else f"single wall tag {nav.wall_tag_id}" if nav else None),
                "tag_map_sha1": tag_map_sha,
                "tag_size_m": nav.effective_tag_size() if nav else None,
                "cam_t_flu": list(nav.cam_t_flu) if nav else None,
                "cam_xyaxes_flu": list(nav.cam_xyaxes_flu) if nav else None,
                # Degrees the MAIN camera is pitched DOWN from the axes above.
                # A RECORD BOUNDARY: cam_xyaxes_flu is the level Onshape
                # registration, so runs at different values of this are in
                # different world estimates and their positions and yaws must
                # not be pooled. Applied as a pure rotation about the camera
                # origin — cam_t_flu is assumed unmoved by the re-mount.
                "cam_tilt_deg": (float(nav.cam_tilt_deg) if nav else None),
                "z_source": nav.z_source if nav else None,
                "axis_gain": (dict(self.cfg.axis_gain) if self.cfg else None),
                "axis_cap": (self.cfg.axis_cap if self.cfg else None),
                "axis_slew_per_s": (self.cfg.axis_slew_per_s
                                    if self.cfg else None),
                # What the loop ACTUALLY capped at, which differs from
                # axis_cap when a DR-control run lowers it.
                "axis_cap_used": self._axis_cap() if self.cfg else None,
                "axis_gain_provenance": "[예측] T200 curve + mixer geometry — "
                                        "replace after the P4 step calibration",
                "intrinsics": "C3 EEPROM factory calibration (UNDERWATER; "
                              "calib/FOV_AUDIT.md)",
                # (0,0)/yaw0 of this run, expressed in the TAG frame — the
                # bridge back to absolute coordinates if anyone needs it.
                "datum_tag_frame": (None if self._datum is None else {
                    "p0": [float(v) for v in self._datum["p0"]],
                    "yaw0_deg": math.degrees(self._datum["yaw0"])}),
            },
            # IMU DEAD RECKONING. Always written, even when off: an ABSENT key
            # cannot tell "this build had no estimator" from "the estimator
            # was switched off", and the schema version cannot either once
            # both kinds of run exist at version 3.
            "imu_dr": self._imu_dr_meta(),
            # STATION BRIDGE. Always written, for the imu_dr reason: an absent
            # key cannot tell "this build had no ladder" from "the ladder was
            # switched off", and the dropout counters are the run's own answer
            # to "did the localizer actually drop out".
            "station_bridge": (self._bridge.meta() if self._bridge is not None
                               else {"enabled": False,
                                     "why": "not built"}),
            # PLAN STREAM (replay / future policy source). Always written,
            # same rule: filter counts are the run's own answer to "did the
            # stream ever get clipped or rejected".
            "plan_stream": self._plan_stream_meta(),
            # OBJECT FOLLOW. ALWAYS written, for the imu_dr reason: an absent
            # key cannot tell "this build had no object nav" from "the station
            # was started without --pose", and the schema version cannot
            # either once both kinds of run exist at version 6.
            "object_nav": self._object_nav_meta(),
        }

    def _object_nav_meta(self) -> dict:
        if self._obj is None:
            return {"enabled": False,
                    "why": (self._obj_note or
                            ("no --pose" if not getattr(self.opts, "pose",
                                                        False)
                             else "not built"))}
        m = self._obj.meta()
        m["source_ok"] = bool(self._obj_src_ok)
        # The mission itself, if one is (or was) armed. The offset is what a
        # reader needs to reconstruct what was being held; the excursion
        # clamp is what says whether the reference was ever limited.
        # ...falling back to the last one that ENDED, because the interesting
        # runs are exactly the ones no longer holding a live follow.
        f = self.follow if self.follow is not None else self._follow_last
        if f is not None:
            m["follow"] = {k: f[k] for k in
                           ("kind", "offset_obj", "dyaw_deg", "hold_m",
                            "arm_p_map", "speed_cap_m_s", "yaw_axis",
                            "max_excursion_m", "approach_lead_m",
                            # THE RECORD BOUNDARY for the feedforward. Runs
                            # before 2026-08-24 had no cap at all and reached
                            # 3.15 m/s; without this key in the meta a reader
                            # cannot tell an uncapped run from a capped one.
                            # `ff_clipped_n` is the run's own answer to "did
                            # the object estimate ask for the impossible".
                            "ff_max_m_s", "ff_clipped_n",
                            "range_at_arm_m", "state")
                           if k in f}
            m["follow"]["frame"] = (
                "offset held in the OBJECT yaw frame; arm_p_map is MAP frame"
                if f.get("yaw_axis") != "none" else
                "offset held in the MAP frame (yaw_axis none); "
                "arm_p_map is MAP frame")
            why = (self.station or {}).get("from_follow")
            if why and self.follow is None:
                m["follow"]["ended"] = str(why)
        elif (self.station or {}).get("from_follow"):
            m["follow"] = {"kind": "follow",
                           "ended": (self.station or {})["from_follow"]}
        return m

    def _imu_dr_meta(self) -> dict:
        if self.cfg is None:
            return {"enabled": False}
        d = dict(self.cfg.imu_dr)
        out = {"enabled": bool(d.get("enabled")) and self.dr is not None,
               "mode": ("control" if self.dr_control else "shadow"),
               "source": d.get("source"),
               "requested_rate_hz": int(getattr(self.opts, "c3_imu_rate", 0)
                                        or 0),
               "abort_err_m": d.get("abort_err_m"),
               "abort_max_s": d.get("abort_max_s"),
               "axis_cap_dr": d.get("axis_cap_dr"),
               "anchor": "once, at the end of settle (no re-anchoring)",
               "raw_log": bool(self.dr is not None)}
        if self.dr is None:
            return out
        out.update(self.dr.meta())
        st = self._dr_last
        out["achieved_rate_hz"] = (round(st["hz"], 1) if st else None)
        out["anchored"] = bool(self.dr.anchored)
        out["elapsed_s"] = (round(st["elapsed"], 2)
                            if st and st["elapsed"] is not None else None)
        out["queue_overflows"] = int(self._dr_overflow)
        return out

    def _write_meta(self) -> None:
        if self._csv_path is None:
            return
        try:
            self._csv_path.with_suffix(".meta.json").write_text(
                json.dumps(self._run_meta("csv"), indent=1, ensure_ascii=False))
        except OSError as e:
            self.bus.log.emit("error", f"mpc meta: {e}")

    @Slot(str)
    def dump_run_meta(self, dirpath: str) -> None:
        """``<dirpath>/controller.json`` — the run's constants, on demand.

        Wired to the REC buttons (window.py): pressing REC NAV or REC UI during
        a PID / MPC / DOB-MPC session drops the plant (M, C, D, g) and the
        controller's own constants (PID kp/kd/ki + gates and limits, MPC
        N/Q/R/u_max + the EAOB tuning) beside the data they explain. Operator
        request, 2026-08-14.

        Deliberately NOT gated on being engaged: the point is to be able to
        press REC, fly, and have the record be complete afterwards — including
        for a hand-flown survey pass the controller never touched. Overwrites
        on a second press in the same folder, because the second press is the
        more recent truth about what the panel was set to.
        """
        if not self._ready or self.cfg is None:
            return
        try:
            p = Path(dirpath)
            p.mkdir(parents=True, exist_ok=True)
            out = p / "controller.json"
            out.write_text(json.dumps(self._run_meta("rec_button"), indent=1,
                                      ensure_ascii=False))
            mode = self.cfg.mode
            self.bus.log.emit(
                "info", f"mpc: {mode} constants + plant model -> {out}")
        except OSError as e:
            self.bus.log.emit("error", f"mpc: controller.json not written ({e})")

    def _dr_row(self, health, meas) -> list:
        """The twelve dead-reckoning columns, in CSV_HEADER order.

        Positions are world FLU in the datum frame — the SAME convention as
        px/py/pz — so `dr_px - px` is the drift with no transform in between.
        """
        rp = health.get("rp_residual_deg")
        rp_s = f"{rp:.3f}" if rp is not None else "nan"
        st = self._dr_last
        if self.dr is None or st is None or not st["ok"]:
            hz = f"{st['hz']:.1f}" if st else "nan"
            n = str(int(st["n"])) if st else "nan"
            t_s = (f"{st['elapsed']:.3f}"
                   if st and st["elapsed"] is not None else "nan")
            return (["nan"] * 7) + [t_s, hz, n, "0", rp_s]
        p = st["p_ned"]
        px, py, pz = _flu_of_ned(float(p[0]), float(p[1]), float(p[2]))
        # pz_imu is the IMU's OWN integrated depth even when z_source is
        # pressure — kept so the "could the IMU have held depth?" question is
        # answerable afterwards without a second run.
        _, _, pz_imu = _flu_of_ned(0.0, 0.0, float(st["pz_imu"]))
        # Against THIS tick's tag state, never against the last one that
        # worked: during a dropout the truth is frozen, and differencing
        # against a frozen point reports the tag standing still as IMU drift.
        err = err_z = float("nan")
        if meas is not None:
            err = math.hypot(float(p[0]) - meas["eta"][0],
                             float(p[1]) - meas["eta"][1])
            err_z = float(st["pz_imu"]) - float(meas["eta"][2])
        return [f"{px:.5f}", f"{py:.5f}", f"{pz:.5f}", f"{pz_imu:.5f}",
                f"{math.degrees(-float(st['yaw'])):.3f}",
                f"{err:.5f}", f"{err_z:.5f}",
                f"{st['elapsed']:.3f}", f"{st['hz']:.1f}", str(int(st["n"])),
                "1", rp_s]

    def _obj_row(self) -> list:
        """The ten object/follow columns, in CSV_HEADER order.

        The ObjectFix lives in the MAP frame; `obj_p*` here are world FLU in
        the DATUM frame, the same convention as `px/py/pz`. That conversion
        happens exactly once — here — so a reader can subtract the two columns
        without knowing anything about either frame.
        """
        fx = self._obj_last
        f = self.follow
        f_state = (str(f.get("state", "following")) if f is not None
                   else ("lost" if (self.station or {}).get("from_follow")
                         else ""))
        f_err = f.get("err_m") if f is not None else None
        f_err_s = f"{float(f_err):.5f}" if f_err is not None else "nan"
        if fx is None or fx.p_map is None:
            return (["nan"] * 6) + ["0", (fx.state if fx is not None else ""),
                                    f_state, f_err_s]
        p = self._map_to_datum_p(fx.p_map)
        px, py, pz = _flu_of_ned(float(p[0]), float(p[1]), float(p[2]))
        yaw_s = ("nan" if fx.yaw_map is None else
                 f"{math.degrees(-self._map_to_datum_yaw(fx.yaw_map)):.3f}")
        return [f"{px:.5f}", f"{py:.5f}", f"{pz:.5f}", yaw_s,
                (f"{fx.age_s:.3f}" if fx.age_s is not None else "nan"),
                (f"{fx.pair_dt_ms:.1f}" if fx.pair_dt_ms is not None
                 else "nan"),
                str(int(bool(fx.pair_exact))), str(fx.state),
                f_state, f_err_s]

    def _write_row(self, meas, health, u, info, axes, t_traj, t) -> None:
        if self._csv is None:
            return
        if meas is None:
            # Normally a tick with no fix writes nothing — there is no state to
            # describe. With the dead reckoner running there is: drift across a
            # tag dropout is precisely what the experiment records, and a gap
            # in the file exactly where the tag was lost would delete it.
            if self.dr is None:
                return
            px = py = pz = yaw_flu = pitch_flu = roll_flu = float("nan")
        else:
            px, py, pz = _flu_of_ned(*meas["eta"][:3])
            yaw_flu = -meas["eta"][5]
            pitch_flu = -meas["eta"][4]
            roll_flu = meas["eta"][3]        # FLU->FRD flips y and z, not x
        rx = ry = rz = ryaw = float("nan")
        e_along = e_cross = None
        ref_speed = None
        lap = 0
        if self.engaged and self.ctrl is not None:
            p_ref, yaw_ref_ned, v_ref = self.ctrl.ref_ned_at(t_traj or 0.0)
            rx, ry, rz = _flu_of_ned(*p_ref)
            ryaw = math.degrees(-yaw_ref_ned)
            ref_speed = float(np.linalg.norm(np.asarray(v_ref, float)[:2]))
            if meas is not None:
                e_along, e_cross = self._path_split(meas, p_ref, t_traj)
            if self.traj_on and self.ctrl.scenario:
                lap = self._path_lap(t_traj)
        w = info.get("w_hat", np.zeros(6))
        ax = (axes.surge, axes.sway, axes.heave, axes.yaw) if axes else \
            (float("nan"),) * 4
        row = [
            f"{t - self._t0_csv:.4f}", f"{px:.5f}", f"{py:.5f}", f"{pz:.5f}",
            f"{rx:.5f}", f"{ry:.5f}", f"{math.degrees(yaw_flu):.3f}",
            f"{math.degrees(pitch_flu):.3f}", str(lap),
            f"{rz:.5f}", f"{ryaw:.3f}",
            (f"{t_traj:.4f}" if t_traj is not None else "nan"),
            self.cfg.mode, str(int(self.engaged)), str(int(self.traj_on)),
            (self.ctrl.solver_kind if self.ctrl else ""),
            str(int(info.get("status", 0))),
            (f"{info['solve_ms']:.2f}" if info.get("solve_ms") is not None
             else "nan"),
            str(self.fix.n_tags if self.fix else 0),
            (f"{self.fix.reproj_rms_px:.3f}"
             if self.fix and self.fix.reproj_rms_px is not None else "nan"),
            (f"{health.get('tag_age'):.3f}"
             if health.get("tag_age") is not None else "nan"),
            (f"{health.get('imu_age'):.3f}"
             if health.get("imu_age") is not None else "nan"),
            str(health.get("z_src", "")),
            # single-tag IPPE ambiguity flag: a bad stretch must be auditable
            str(int(bool(self.fix.ambiguous)) if self.fix else 0),
        ]
        row += [f"{float(v):.3f}" for v in (list(w) + list(u))]
        row += [f"{float(v):.4f}" for v in ax]
        dev = self._pwm_dev_us()
        row += [f"{float(info.get('nis', 0.0)):.2f}",
                (f"{dev:.0f}" if dev is not None else "nan"),
                (f"{e_along:.5f}" if e_along is not None else "nan"),
                (f"{e_cross:.5f}" if e_cross is not None else "nan")]
        # path_s = arclength progress (theta). The MPCC OPTIMIZES it, so it is
        # the run's independent variable rather than a derived quantity.
        row += [
            (f"{self._path_theta():.5f}"
             if self._path_theta() is not None else "nan"),
            (f"{ref_speed:.5f}" if ref_speed is not None else "nan")]
        row += self._dr_row(health, meas)
        row += [f"{math.degrees(roll_flu):.3f}"]
        # STATION BRIDGE: 0 / "none" on a normal tick. Non-zero marks a row
        # whose px,py did NOT come from the tag (see CSV_HEADER's note).
        row += [(f"{self._bridge.elapsed:.2f}" if self._bridge else "0.00"),
                (self._bridge.tier if self._bridge else SB.TIER_NONE)]
        # OBJECT FOLLOW: all `nan`/"" without --pose. obj_pair_exact is the
        # record boundary — see CSV_HEADER's note before pooling anything.
        row += self._obj_row()
        # WALL time of the controller step (schema 7). `solve_ms` beside it is
        # the solver's own account of itself and both acados bridges prefer
        # `time_tot`, which times only the QP — the 2026-08-23 tick that blocked
        # 6.5 s logged 1.15 ms there, which is why nothing in that run's CSV
        # showed the stall that lost the vehicle.
        row += [f"{float(self._tick_ms):.2f}"]
        # REPLAY (schema 8): which streamed plan the reference came from, and
        # the jaw drive being held. See CSV_HEADER's 2026-08-30 note.
        rp, pst = self.replay, self._plan_stitcher
        if rp is not None and pst is not None and pst.has_plan():
            pid = pst.active_plan_id()
            row += [(str(int(pid)) if pid is not None else "nan"),
                    (str(pst.source_at(float(t_traj)))
                     if t_traj is not None else ""),
                    f"{float(rp.get('grip_drive', 0.0)):.0f}"]
        else:
            row += ["nan", "", "nan"]
        try:
            self._csv.write(",".join(row) + "\n")
            self._rows += 1
            if self._rows % 200 == 0:
                self._csv.flush()
        except OSError:
            pass

    def teardown(self) -> None:
        self.disengage("shutdown")
        self._close_csv()
