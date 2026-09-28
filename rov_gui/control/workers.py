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

import dataclasses
import hashlib
import inspect
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np

from .. import runstore
from ..qt import Slot
from ..state import (ACTION_REPR_POS_RPY_WIDTH, Conn, MpcStatus, NavFix,
                     ObjectFix, PilotInput, POLICY_ACTION_REPRS_FLYABLE,
                     POLICY_GRID_WHY_IDLE, PolicyPlanViz, PolicyState,
                     TagOverlay, now)
from .allocation import axes_to_wrench, slew_axes, wrench_to_axes
from .geometry import (HANDHELD_TCP_OFFSET_CAM_M, MpcConfig, NavConfig,
                       POLICY_ACTION_REPR, POLICY_DOWN_SAMPLE_STEPS,
                       POLICY_PROVENANCE, SHAPES, S_FLU_FRD, yaw_from_R)
from .imu_dr import ImuCalibration, ImuDeadReckoner
from . import object_nav as ON
from . import station_bridge as SB
from .state_assembler import StateAssembler, rot_zyx
from ..backends.base import TimerWorker


def _num_or_none(v):
    """A status number for the meta: float (numpy scalars included) or None
    — a np.float32 hz from the worker must not break the meta write."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _rov_drawn_geometry_meta():
    """`rov_shape.rov_geometry_meta()` for the run meta, or a note when the
    widgets package cannot be imported (headless tools importing this
    module without the GUI tree) — an absent key could not say which."""
    try:
        from ..widgets.rov_shape import rov_geometry_meta
    except Exception as e:                                   # noqa: BLE001
        return {"error": f"rov_shape unavailable: {e}"}
    return rov_geometry_meta()


def _json_default(o):
    """json.dumps fallback for the plan record AND the run meta: numpy
    scalars/arrays and tuples of them become lists; anything else its repr
    (never a crash in the 20 Hz tick over a log line, never a lost
    meta.json over one numpy scalar — verify 2026-09-02)."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (deque, set, tuple)):
        return list(o)
    return repr(o)


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
        # SECOND-CAMERA FALLBACK (control/nav_fusion.py). ``navs`` holds one
        # TagNav per feed that can localize — the primary (``solve_panel``)
        # and, with hw_nav.yaml `fallback.enabled`, the other calibrated
        # feed. The two solvers share the map and the gates; what differs is
        # the camera->body extrinsic, and the RGB's follows the tracked
        # mount tilt. ``align`` learns the constant body-frame offset between
        # their answers, ``arbiter`` decides when the fallback's fix is the
        # vehicle's state.
        self.navs: dict = {}              # panel -> TagNav
        self.fallback_panel: str | None = None
        self.align = None                 # nav_fusion.BodyAlign
        self.arbiter = None               # nav_fusion.FallbackArbiter
        self._tilt_used: float | None = None    # tilt the RGB extrinsic assumes
        self._tilt_epoch_seen: int | None = None
        self._tilt_est: float | None = None
        self._t_tilt_emit = 0.0
        self._t_hold_log = 0.0
        self._n_held = 0
        # Set by HardwareBackend when --slam builds a REAL fix producer for
        # this bench run; see _emit_land_fix's caller.
        self.suppress_land_fix = False
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
                          dup_confirm_px=c.dup_confirm_px,
                          outlier_max_frac=c.outlier_max_frac,
                          outlier_min_ratio=c.outlier_min_ratio,
                          min_border_px=c.min_border_px)
        self.navs = {self.solve_panel: self.nav}
        # The fallback solver: the OTHER calibrated feed, when it is present
        # and the config asks for it. Built with datum "map" — the primary's
        # first-fix datum, if any, is applied by hand in _route_fallback so
        # both solvers report in ONE world (two independent datums would make
        # the offset between them pose-dependent and unlearnable).
        fb = dict(getattr(c, "fallback", {}) or {})
        other = "second" if self.solve_panel == "main" else "main"
        if fb.get("enabled") and other in self.mailboxes:
            from .nav_fusion import BodyAlign, FallbackArbiter
            self._tilt_used = (float(c.second_cam.get("tilt_deg", 0.0))
                               if other == "second" else None)
            R2, t2 = c.R_t_frd_cam(other, tilt_deg=self._tilt_used)
            gate = fb.get("max_reproj_px")
            self.navs[other] = TagNav(
                c.make_tag_map(), c.effective_tag_size(), R2, t2, c.R_ned_map,
                max_reproj_px=(c.max_reproj_px if gate is None else float(gate)),
                min_tags=c.min_tags,
                datum="map", tilt_gate_deg=c.tilt_gate_deg,
                ambiguity_ratio=c.ambiguity_ratio,
                duplicate_ids=c.duplicate_ids, dup_confirm_px=c.dup_confirm_px,
                outlier_max_frac=c.outlier_max_frac,
                outlier_min_ratio=c.outlier_min_ratio,
                min_border_px=c.min_border_px)
            self.fallback_panel = other
            self.align = BodyAlign(pair_tol_s=fb["pair_tol_s"],
                                   min_pairs=fb["min_pairs"],
                                   window=fb["window"])
            self.arbiter = FallbackArbiter(after_s=fb["after_s"],
                                           allow_unaligned=fb["allow_unaligned"])
            self._tilt_reset_deg = float(fb["tilt_reset_deg"])
        elif fb.get("enabled"):
            self.bus.log.emit(
                "warn", f"tagnav: fallback.enabled but the {other!r} feed is "
                        f"not running (have {sorted(self.mailboxes)}) — no "
                        f"second-camera fallback this session")
        # Defaults: the localizing feed ON (the MPC needs it), the fallback
        # feed ON too (it has to see tags BEFORE the primary loses them, or
        # there is nothing learned to hand over with), extras OFF (detection
        # on a second 30 fps stream is CPU spent only when asked). The window
        # mirrors these onto the TAG buttons without re-emitting.
        for panel, mb in self.mailboxes.items():
            self.enabled[panel] = panel in self.navs
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
                    f"localizing={self.solve_panel}, datum={c.datum}"
                    + (f", fallback={self.fallback_panel} after "
                       f"{self.arbiter.after_s:g} s, aligned at "
                       f"{self.align.min_pairs} pairs"
                       if self.fallback_panel else ""))

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
            if panel == self.fallback_panel and self.align is not None:
                # What was learned describes fixes that stop arriving; the
                # next on is a fresh start rather than a stale offset.
                self.align.reset("fallback feed switched off")
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
        role = ("primary" if panel == self.solve_panel
                else "fallback" if panel in self.navs else "overlay")
        # ``localizes`` keeps its meaning — THESE detections are the vehicle's
        # state — so the recording and the depth cross-check (window.py) keep
        # reading the primary feed only; the fallback rides ``role``.
        localizes = (role == "primary" and K is not None)
        t_cap = float(item.get("t_capture", now()))
        self.bus.tag_overlay.emit(TagOverlay(
            panel=panel,
            quads=tuple(tuple((float(x), float(y)) for x, y in d.corners)
                        for d in dets),
            ids=tuple(d.tag_id for d in dets),
            mapped=tuple(d.tag_id in self.nav.map for d in dets),
            src_w=gray.shape[1], src_h=gray.shape[0],
            detect_ms=detect_ms, localizes=localizes, enabled=True,
            role=(role if K is not None else "overlay"),
            # The corners ride with the camera model and capture time that
            # produced them, so a REC NAV recording is a re-usable set of raw
            # observations (window.py writes detections.csv from this).
            t_capture=t_cap,
            K=((float(K[0][0]), float(K[1][1]), float(K[0][2]),
                float(K[1][2])) if K is not None else ()),
            dist=(tuple(float(v) for v in np.asarray(dist).ravel())
                  if dist is not None else ()),
            stamp=now()))
        hint = None
        imu = self._imu
        if imu is not None and imu.roll is not None:
            hint = (float(imu.roll), float(imu.pitch))
        if role == "fallback" and K is not None:
            if not bool(getattr(self.opts, "land_dry_run", False)):
                self._route_fallback(dets, K, dist, hint, t_cap, gray.shape,
                                     detect_ms)
            return
        if not localizes:
            return                          # overlay-only feed (or no K yet)
        if bool(getattr(self.opts, "land_dry_run", False)):
            # ...unless something else is producing a REAL fix. --slam builds
            # an ORB-SLAM3 bridge that publishes onto this same signal, and
            # bus.nav_fix is last-writer-wins in MpcWorker: two producers would
            # give the policy a proprio history alternating between a moving
            # pose and a stationary one, i.e. a motion cue that flips at the
            # fix rate. The bench needs the synthetic fix only when it has no
            # other pose at all.
            if not self.suppress_land_fix:
                self._emit_land_fix(t_cap, gray.shape)
            return
        sol = self.nav.solve(dets, K, dist, rp_hint=hint)
        if sol is None:
            t = now()
            if t - self._last_miss_note > 1.0:
                self._last_miss_note = t
                # The WHY travels with the miss — "seen, none usable" alone is
                # undebuggable at the pool (live finding, 2026-08-12).
                why = self.nav.last_reject or "none usable"
                if self.arbiter is not None and self.arbiter.covering(t):
                    why += " · RGB covering"
                self.bus.nav_fix.emit(NavFix(
                    t_capture=t_cap, n_tags=0,
                    # The ids ride the MISS too. Without them a rejected
                    # frame said only "39 seen" and no recording could ever
                    # name the culprit — which is exactly what blocked the
                    # 2026-08-13 diagnosis (0/114 and 0/146 rejected rows
                    # carried ids in data/20260813/0813_17*/nav_*).
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
        if self.align is not None:
            from .nav_fusion import T_of
            self.align.note_main(t_cap, T_of(sol.R_ned_body, sol.p_ned))
            self.arbiter.note_main_fix(t)
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

    # ------------------------------------------------- second-camera fallback
    def _route_fallback(self, dets, K, dist, hint, t_cap: float, shape,
                        detect_ms: float) -> None:
        """Solve the fallback feed; learn the offset; forward only when the
        primary is silent AND the offset is known (control/nav_fusion.py)."""
        from .nav_fusion import T_of
        nav = self.navs[self.fallback_panel]
        sol = nav.solve(dets, K, dist, rp_hint=hint)
        if sol is None:
            # Silent while the primary is live (its fix is the state). When
            # the primary has NOTHING, this is the fix the operator was
            # counting on, and WHY it failed has to reach the screen and the
            # recording — a 2026-09-07 pool run lost every multi-tag RGB
            # frame to the 3 px gate with no line anywhere saying so.
            t = now()
            if (dets and self.arbiter.main_silent(t)
                    and t - self._last_miss_note > 1.0):
                self._last_miss_note = t
                self.bus.nav_fix.emit(NavFix(
                    t_capture=t_cap, n_tags=0,
                    tag_ids=tuple(d.tag_id for d in dets),
                    geometry=self.cfg.geometry, source=self.fallback_panel,
                    conn=Conn.DEGRADED,
                    note=f"{len(dets)} seen: {nav.last_reject or 'none usable'}",
                    src_w=shape[1], src_h=shape[0], stamp=t))
            return
        # ONE world: the primary's first-fix datum, if it has one.
        R, p = sol.R_ned_body, sol.p_ned
        d = self.nav.datum_transform()
        if d is not None:
            Rz, p0 = d
            p, R = Rz @ (p - p0), Rz @ R
        elif self.nav.datum == "first_fix":
            return                          # no world yet to express it in
        T_raw = T_of(R, p)
        self.align.note_second(t_cap, T_raw)
        t = now()
        self._maybe_emit_measured_tilt(t)
        verdict = self.arbiter.decide(t, self.align.ready)
        if verdict != "forward":
            self._n_held += 1
            if t - self._t_hold_log > 5.0 and self.arbiter.main_silent(t):
                # Only worth a line when it would have MATTERED: the primary
                # is silent and the fallback had a fix it was not allowed to
                # use. While the C3 is live, holding is the normal state.
                self._t_hold_log = t
                self.bus.log.emit(
                    "warn", f"tagnav: RGB fix held back — {verdict} "
                            f"({self.align.n_pairs}/{self.align.min_pairs} "
                            f"pairs; {self.align.reset_reason})")
            return
        T = self.align.correct(T_raw)
        aligned = self.align.ready
        note = (f"RGB fallback · aligned n={self.align.n_pairs}"
                f" offset {self.align.offset_mm():.0f} mm"
                f" spread {self.align.spread_mm() or 0.0:.0f} mm"
                if aligned else "RGB fallback · UNALIGNED raw pose")
        if sol.reproj_rms_px > self.nav.max_reproj_px:
            # Passed the fallback's own (looser) gate, not the main one:
            # say so, so a run's RGB fixes can be filtered by that later.
            note += f" · reproj {sol.reproj_rms_px:.1f}px > main gate"
        if sol.note:
            note += "; " + sol.note
        self.bus.nav_fix.emit(NavFix(
            t_capture=t_cap, n_tags=sol.n_tags, tag_ids=sol.tag_ids,
            tag_insts=sol.tag_insts,
            p_ned=tuple(float(v) for v in T[:3, 3]),
            R_ned_body=tuple(float(v) for v in T[:3, :3].ravel()),
            yaw_ned=yaw_from_R(T[:3, :3]),
            reproj_rms_px=sol.reproj_rms_px, detect_ms=detect_ms,
            hz=None, ambiguous=sol.ambiguous, geometry=self.cfg.geometry,
            source=self.fallback_panel,
            src_w=shape[1], src_h=shape[0],
            # DEGRADED on purpose: the state is coming from the backup camera
            # and the SENSORS row should say so in colour.
            conn=Conn.DEGRADED, note=note, stamp=t))

    def _maybe_emit_measured_tilt(self, t: float) -> None:
        """The alignment measures the RGB mount's real tilt; hand it to the
        tilt tracker (via the payload sender) about once a second."""
        if (self.fallback_panel != "second" or not self.align.ready
                or t - self._t_tilt_emit < 1.0):
            return
        self._t_tilt_emit = t
        R2, t2 = self.cfg.R_t_frd_cam("second", tilt_deg=self._tilt_used)
        deg = self.align.measured_tilt_deg(R2, t2)
        if deg is not None:
            self.bus.tilt_measured.emit(float(deg), int(self.align.n_pairs))

    @Slot(object)
    def on_payload(self, st) -> None:
        """The tracked mount tilt (state.PayloadState). A COMMANDED change —
        the epoch moved and the mount has stopped — re-points the RGB
        extrinsic at the new angle and forgets the alignment: the offset it
        learned belonged to the mount position that no longer exists. A
        measurement changes neither (the alignment already absorbs whatever
        the extrinsic gets wrong, exactly)."""
        if self.fallback_panel != "second" or self.align is None:
            return
        epoch = int(getattr(st, "tilt_epoch", 0))
        est = getattr(st, "tilt_est_deg", None)
        self._tilt_est = est
        if self._tilt_epoch_seen is None:
            self._tilt_epoch_seen = epoch
            return
        if epoch == self._tilt_epoch_seen or bool(getattr(st, "tilt_moving", False)):
            return
        self._tilt_epoch_seen = epoch
        if est is None:
            return
        if (self._tilt_used is not None
                and abs(float(est) - self._tilt_used) < self._tilt_reset_deg):
            return
        self._tilt_used = float(est)
        R2, t2 = self.cfg.R_t_frd_cam("second", tilt_deg=self._tilt_used)
        self.navs["second"].set_extrinsic(R2, t2)
        self.align.reset(f"mount moved to {self._tilt_used:+.0f}°")
        self.bus.log.emit(
            "info", f"tagnav: RGB extrinsic now assumes tilt "
                    f"{self._tilt_used:+.1f}° ({getattr(st, 'tilt_est_src', '?')}); "
                    f"C3-vs-RGB alignment reset — relearning")

    def fallback_meta(self) -> dict | None:
        """For controller.json: the fallback's configuration and what it had
        learned when the record was written. None = no fallback this run."""
        if self.align is None:
            return None
        E = self.align.E()
        return {
            "panel": self.fallback_panel,
            "after_s": self.arbiter.after_s,
            "allow_unaligned": self.arbiter.allow_unaligned,
            "min_pairs": self.align.min_pairs,
            "n_pairs": self.align.n_pairs,
            "aligned": self.align.ready,
            "offset_mm": self.align.offset_mm(),
            "spread_mm": self.align.spread_mm(),
            "E_body": (None if E is None else [[float(v) for v in row] for row in E]),
            "tilt_assumed_deg": self._tilt_used,
            "tilt_est_deg": self._tilt_est,
            "n_forwarded": self.arbiter.n_forwarded,
            "n_held": self._n_held,
            "reset_reason": self.align.reset_reason,
        }

    def _emit_land_fix(self, t_cap: float, shape) -> None:
        """A SYNTHETIC stationary fix, for --land-dry-run. Bench only.

        There are no tags on a bench, and without a fix nothing downstream
        runs: the state assembler returns None, engage is refused ("no tag
        fix"), and the policy never gets a proprio row. This supplies the one
        thing the chain actually needs.

        It is not a fabricated measurement. The policy's proprio is RELATIVE —
        ``inv(T_now) @ T_i`` — so for a vehicle that does not move it is
        identically zero/identity NO MATTER what absolute pose is used; two
        poses 15 m apart give bit-identical observations (verified). The only
        thing that has to be real is the CLOCK, because ``EtaHistory``'s
        degenerate test reads stamps and never values: a fix re-emitted at one
        frozen stamp is deduped to a single row and inference is skipped
        entirely. So the pose is the origin and the stamp is the camera's own
        capture time, which advances at the frame rate.

        ``n_tags`` and ``reproj_rms_px`` stay 0 and the note says LAND DRY-RUN,
        so the CSV columns that would otherwise carry fabricated tag counts are
        empty and no reader can mistake this for a localization that happened.
        """
        eye = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        self.bus.nav_fix.emit(NavFix(
            t_capture=t_cap, n_tags=0, tag_ids=(), tag_insts=(),
            p_ned=(0.0, 0.0, 0.0), R_ned_body=eye, yaw_ned=0.0,
            reproj_rms_px=0.0, detect_ms=0.0, hz=None, ambiguous=False,
            geometry=self.cfg.geometry, source=self.solve_panel,
            src_w=shape[1], src_h=shape[0], conn=Conn.ONLINE,
            note="LAND DRY-RUN — synthetic stationary fix, no tags",
            stamp=now()))

    def teardown(self) -> None:
        for mb in self.mailboxes.values():
            mb.set_wanted(False)


# =============================================================================
# the control loop
# =============================================================================
#: `solver_status` on a POLICY OBSERVE row: the controller step never ran.
#: Deliberately a value no solver can produce — acados and IPOPT both report
#: >= 0, and 0 is "converged", which is precisely the reading a muted loop
#: must not be able to pass for.
OBSERVE_SOLVER_STATUS = -1

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
              "plan_id,ref_src,grip_cmd,grip_w_est,grip_g,hold_frac,observe,"
              "pwm1,pwm2,pwm3,pwm4,pwm5,pwm6,pwm7,pwm8,pwm_age_s,"
              "batt_v,batt_a,"
              "rroll_deg,rpitch_deg,ax_roll,ax_pitch,rp_track\n")
# 2026-09-08: the eleven trailing columns. `pwm1..pwm8` are the vehicle's OWN
# SERVO_OUTPUT_RAW microseconds per thruster (1500 = neutral; EMPTY when that
# output is unconfigured or the vehicle has not reported yet), `pwm_age_s` is
# how old that message was at this row, and `batt_v` / `batt_a` are the pack
# voltage and current. Until then the only actuation record was `pwm_dev_us`,
# the MEAN of |pwm-1500| over the eight — which cannot answer "did any single
# thruster clear its deadband", the question every 2026-09 stall analysis ran
# into (.../0908_112101/diag/taskE_out.txt had to assume a x2 factor because
# only four thrusters drive a horizontal axis). `pwm_dev_us` is kept, so
# nothing that reads it breaks. WHY VOLTAGE: T200 thrust at a given PWM is a
# function of pack voltage, so a newton figure derived without it is a guess.
# The station stores the MEASUREMENT (us) and leaves the curve to the reader.
# The raw stream still exists too and is strictly richer (verbatim
# SERVO_OUTPUT_RAW + battery, 10 Hz, `<stem>_rov.jsonl` via SensorLog) — but it
# is written ONLY when the depth feed is recording or `--imu-dr` is on, so no
# 2026-09-07/08 run has one. These columns are the unconditional, tick-aligned
# record; the sidecar is the fuller one when it happens to be there.
# WHY THE AGE COLUMN: bus.thrusters is republished at 10 Hz into this 20 Hz
# CSV and `self.thr` is never cleared, so a dead feed would otherwise write its
# last value forever — the same trap the propulsion panel was fixed for on
# 2026-08-07. `pwm_age_s` is the reader's guard: a row whose age keeps growing
# is a repeat, not a measurement.
#
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
#
# 2026-09-02: `grip_w_est` / `hold_frac` arrived with the LIVE POLICY mission
# (shape: policy, schema 10). `grip_w_est` is the OPEN-LOOP jaw-width
# estimate in metres (policy_frames.GripperWidthEstimator: an integrator of
# the drive levels the jaw was sent, clamped to the dataset's open/closed
# levels — there is no jaw feedback, so this is what the POLICY WAS TOLD, not
# what the jaw did; `nan` when no policy worker is present). `hold_frac` is
# the fraction of the NMPC horizon's stages that lay PAST the active plan's
# last knot this tick (nan outside a policy run): with `policy.hold_tail:
# mask` those stages carry no position/velocity weight (NedPlan.w_stage), so
# a row with hold_frac 0.66 was flown with two thirds of the horizon free.
# The mask is applied ONLY while a plan is live (t_rel < its last knot) and
# the mission is not latched (verify 2026-09-02): a row with hold_frac 1.000
# was flown on the stitcher's endpoint hold at FULL weight — the 1.0 is a
# record of where the horizon sat, not of a weight-free horizon. Earlier
# dry-run records (2026-09-02 before the fix) masked the whole horizon there.
# `plan_id` / `ref_src` / `grip_cmd` keep their meaning for a policy run. A
# policy run's reference is a streamed LEARNED plan — `trajectory.kind ==
# "policy"` and `reference_clock.strategy == "plan_stream_policy"` in the
# meta are the boundary; never pool policy rows with replay rows or with
# geometric missions, and never quote a policy run flown under
# `source: demo` (SYNTHETIC plant) as a measurement of anything.
#
# 2026-09-03: `observe` arrived with POLICY OBSERVE (--policy-observe, schema
# 11) and is the LOUDEST boundary in this file. `observe == 1` means THE
# CONTROL LOOP WAS NEVER CLOSED on that row: `ctrl.step()` was not called, no
# wrench was computed and nothing was sent to the vehicle — the pilot flew it
# by hand on the joystick while the diffusion policy's proposals were
# composed and drawn. What that does to the other columns, deliberately:
#   * `uX..uN`, `w0..w5`, `ax_*`  -> nan. NOT 0.0: a zero would read as "the
#     controller commanded zero", which is a different and false claim.
#   * `solver_status` -> -1, a value no solver can return (acados and IPOPT
#     both report >= 0, and 0 means CONVERGED — exactly the reading a muted
#     loop must not be able to pass for). `solve_ms` / `tick_ms` -> nan.
#   * `e_along`, `e_cross`, `ref_speed_m_s` -> nan. They are TRACKING errors
#     and there was no tracker; the distance between `px,py` and `rx,ry` here
#     is just where the pilot chose to fly.
#   * `px,py,pz` are the real measured pose (a MANUALLY flown trajectory) and
#     `rx,ry,rz,ryaw` are the real composed DP reference — both are kept,
#     because "what did the network ask for, from where" is the run's whole
#     output. They are simply not a controller's input and output.
# `engaged` stays 1: the mission machinery genuinely was running. Read
# `observe`, never `engaged`, to ask whether the station was driving.
# Observe runs also live in their OWN tree (data/*/*_observe/), so a
# file from one cannot reach a water-run folder by accident.
#
# 2026-09-07: schema 12 adds NO CSV column. What changed is the policy's
# ACTION: (16, 5) [dx, dy, dz, dyaw, gripper_width_m] (`action_repr`
# pos_yaw_width) instead of (16, 10) [pos, rot6d, width], and that lives in
# plans.jsonl (`action_raw` rows are 5 wide from here on, `action_repr` on
# every line, `dropped_rp_deg` is 0 by construction) and in the meta
# (`policy.action_repr`, `policy.run.action_repr` / `reject_action_repr`).
# The CSV's rx,ry,rz,ryaw are the COMPOSED reference either way and mean
# what they meant; a policy run's rows are still not poolable across the
# boundary because the network that produced the reference is a different
# network.
#
# 2026-09-07 (2): `grip_g` arrived with schema 13, between `grip_w_est` and
# `hold_frac` (the ONE exception to the append-at-the-end rule, kept beside
# the estimate it is judged against; every by-name reader is unaffected).
# It is the jaw-width channel value (0 = closed, 1 = open) THE HYSTERESIS
# ACTUALLY SAW this tick: the stitcher sampled at t_traj +
# policy/replay.gripper_lookahead_s (0.0 = at now, the pre-13 behaviour),
# clamped inside the covered span. `nan` when no stream mission is live,
# when the mission's jaw is off (gripper false / forced off by OBSERVE) or
# when no plan is installed yet. Written because the 0907_145206 pool run
# (mission mpc_145239) could not answer "what did the jaw logic see" — the
# policy asked to close at the far end of every chunk and the sample at now
# only crossed close_below during plan gaps; `grip_cmd` beside it is what
# the logic then DID. A run with lookahead > 0 fed the hysteresis a
# different signal from a run with 0: `policy.config.gripper_lookahead_s`
# (or `plan_stream.config.gripper_lookahead_s` for a replay) in the meta is
# the boundary for any jaw-timing statistic.
#
# 2026-09-26: the five trailing columns `rroll_deg, rpitch_deg, ax_roll,
# ax_pitch, rp_track` arrived with schema 16, the 6-DoF VARIANT
# (engage.attitude_axes: the NMPC's K/M torques go to ArduSub as the
# MANUAL_CONTROL extension axes s = pitch / t = roll; policy.attitude_track:
# a 7-dim [dx, dy, dz, dyaw, droll, dpitch, width] policy's roll/pitch are
# TRACKED instead of levelled). Three-state pattern, always present:
#   * `rp_track` 1 = this row was flown with the attitude axes SENT (K/M
#     left the station); 0 = the 4-DoF path, where K/M were dropped at
#     allocation exactly as every run before this date. THE BOUNDARY: never
#     pool attitude statistics across it, and never pool dobmpc `w0..w5`
#     across it either — with the axes sent, w3/w4 (w_hat K/M) book the
#     residual of a CREDITED torque; without them they booked the whole
#     torque the plant never got.
#   * `rroll_deg` / `rpitch_deg` = the attitude REFERENCE the follower was
#     tracking, in the SAME world-FLU sign convention as the measured
#     `roll_deg` / `pitch_deg` columns (roll_flu = roll_ned, pitch_flu =
#     -pitch_ned), so the attitude error is `roll_deg - rroll_deg` straight
#     off the file. `nan` when rp_track is 0 (no attitude reference was ever
#     flown before 16; the follower levelled).
#   * `ax_roll` / `ax_pitch` = the normalised -1..+1 K/M axis commands SENT
#     (NED sign: +roll = starboard-down, +pitch = nose-up; the sink's own
#     sign map is applied AFTER these), `nan` when off — the same rule as
#     `ax_surge..ax_yaw` under observe: nan says "not commanded", 0 would say
#     "commanded zero".
# `uK` / `uM` and the measured `roll_deg` / `pitch_deg` existed before 16 —
# but until this date K/M were never applied, so a pre-16 `uK` is a torque
# the solver asked for and nobody delivered.


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
        # Bench mode: no command sink exists, so the vehicle gates below are
        # skipped and the divergence guard is off. Read once, here, so no gate
        # can disagree with another about which mode this run is in.
        self.land_dry_run = bool(getattr(opts, "land_dry_run", False))
        # POLICY OBSERVE (--policy-observe, 2026-09-03). The DIFFUSION POLICY
        # is the subject of the run: it infers, its chunks are composed into
        # datum-NED plans, filtered, stitched and DRAWN — and the controller
        # output is MUTED. `ctrl.step()` is never called and not one byte
        # leaves this worker toward the vehicle; the pilot flies by hand on
        # the joystick the whole time and checks, from wherever they like,
        # whether the reference the network draws makes sense.
        #
        # A DIFFERENT AXIS FROM land_dry_run, and they compose: that one says
        # "the vehicle cannot move and there is no sink"; this one says "the
        # vehicle really is flying, we are simply not the ones driving it".
        # So the sink is REAL here (the pilot needs it) and the localizer is
        # real — which is exactly why the record boundary below has to be as
        # loud as it is: everything about an observe run looks quotable and
        # none of it is.
        #
        # 2026-09-11 (operator request: HIGH/LOW split): OBSERVE IS THE LOW
        # LEVEL "none" — TELEOP — chosen in the trajectory panel at runtime,
        # and `--policy-observe` is only the launch alias that preselects it.
        # `_observe` is the backing field of the read-only `observe`
        # property; its ONLY writers are setup() (the launch normalisation)
        # and set_mode() (the panel), so — as before — no two gates can
        # disagree about which mode this run is in. Seeded from the alias
        # here so a fixture that sets `Opts.policy_observe = True` still
        # builds an observe worker; setup() then makes it consistent with
        # cfg.mode (observe == (cfg.mode == "none")).
        self._observe = bool(getattr(opts, "policy_observe", False))
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
        self._flight_mode_at_engage = None
        self._flight_modes_seen = []
        self._mode_changed_at = None       # wall time of the last flight-mode change
        self._tel_mode_prev = None
        self._yaw_hold = False             # STABILIZE: yaw axis sent as 0 (live)
        self._yaw_hold_at_engage = False   # ...as decided at engage (record)
        # THE 6-DoF VARIANT (2026-09-26, engage.attitude_axes). `_attitude_axes`
        # is the ONE bit every K/M actuation site reads: True only between an
        # engage whose gates all passed (MANUAL, LOW in the allow-list, a v2
        # link, firmware >= min_firmware, the sink configured, the probe
        # artefacts pinned) and the next disengage. False = today's 4-DoF
        # path, byte-identical: wrench_to_axes is called with its pre-variant
        # arguments, roll/pitch stay 0.0, `_axes_prev` stays a 4-tuple.
        self._attitude_axes = False
        self._attitude_cap = (0.2, 0.3)    # (roll, pitch) wire caps in force
        self._attitude_at_engage: dict = {}   # the record (survives disengage)
        self._att_ceiling_ticks = 0        # abort_deg debounce (interlock ii)
        self._att_sat: dict = {0: None, 1: None}   # cap-pinned windows (iii)
        self._attitude_sat_ticks = 0
        self._att_degraded_at = None       # wall time the sink degraded (v)
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

        # ENGAGED = the mission MACHINERY is live (datum captured, CSV open,
        # missions armable, reference real, PolicyState fed). It is TRUE in
        # POLICY OBSERVE too — every one of those things is exactly what an
        # observe run needs, and switching it off instead would have silently
        # disabled the policy mission itself: `set_traj` refuses when not
        # engaged (:1772), `_tick_replay` only runs inside `if self.engaged`
        # (:4410), the proprio history only fills while engaged (:3854) and
        # `PolicyWorker._active()` requires `PolicyState.engaged`
        # (backends/policy.py:582), so the GPU would never have inferred.
        #
        # Whether a WRENCH LEAVES is the separate question `commanding`
        # answers. Anything about actuation must read that one.
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
        # drops depth hold on a vehicle that is never neutral (model -5.7 N; it floated on 2026-09-07).
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
        # LIVE POLICY (shape: policy). `policy_present` is set by the backend
        # when a PolicyWorker exists (--policy); `policy_meta_fn` is its
        # meta() for the run record (the fstereo_meta_fn pattern). The inbox
        # is LATEST-WINS: the worker emits one PolicyPlan per inference and
        # the intake (`_tick_policy_intake`) consumes it on the next tick —
        # a plan that was superseded before the tick is a plan the vehicle
        # never needed. `_policy_epoch` increments at every ENGAGE and every
        # policy ARM; a plan built in an older epoch is dropped (v2 A8).
        self.policy_present = False
        self.policy_meta_fn = None
        self._policy_status = None         # last PolicyStatus (stamped)
        self._policy_inbox = None          # latest PolicyPlan, or None
        self._policy_last_seen = -1        # plan_id watermark (this epoch)
        self._policy_epoch = 0
        self._policy_counts = {"drop_epoch": 0, "drop_old": 0,
                               "drop_inactive": 0, "received": 0}
        self._policy_eta_start = None      # (6,) datum eta at policy START
        # Measured-eta history keyed on the FIX stamp (v2 A6), for the anchor
        # at obs_t (A4: x_meas(obs_t) is the interpolated history, NOT the
        # tick eta). One row per fix, 3 s deep, cleared at engage.
        self._policy_eta_hist = None       # policy_frames.EtaHistory
        self._policy_hist_last_fix = None
        # Open-loop jaw width (v2 A12), alive from setup() for the whole
        # process — every cmd_gripper_drive (pilot AND policy) integrates it.
        self._grip_est = None              # policy_frames.GripperWidthEstimator
        self._policy_T_bt = None           # (4,4) T_body_tcp, body FRD
        self._policy_tcp_off_cam = None    # (3,) TCP offset in the C3 frame
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
        # ONE-SHOT mission lines written outside any run (a LOW combo click
        # at the bench, 2026-09-11 review): held here and flushed as the
        # first lines of the NEXT events.log that gets a real line, then
        # dropped — unlike `_session_banner`, which heads EVERY folder.
        # See _log_event(pending=True).
        self._pending_events: list[str] = []
        # Checkpoint changes the PolicyStatus reported WHILE a controller
        # CSV was open (REC across DISENG + a panel pick, 2026-09-11
        # review): exported in policy meta so one CSV that spans two
        # networks says so itself. Reset per CSV (_open_csv / _close_csv).
        self._ckpt_changes: list[dict] = []

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
        # LOW LEVEL NONE (2026-09-11, operator request): ONE source of truth.
        # `observe` is TRUE iff the LOW level is "none". Three ways say so at
        # launch — the `--policy-observe` alias (opts.policy_observe, or a
        # fixture's Opts attribute), `--mpc-mode none`, or `mode: none` in
        # the file — and they are folded into cfg.mode here, once, so that
        # from this line on set_mode() is the only writer of either.
        # Invariant (pinned by test_policy, never asserted at runtime):
        # observe == (cfg.mode == "none") after setup.
        if (self._observe
                or str(getattr(self.opts, "mpc_mode", "") or "").lower() == "none"
                or str(self.cfg.mode).lower() == "none"):
            self.cfg.mode = "none"
            self._observe = True
        else:
            self._observe = False
        if getattr(self.opts, "rov_model", None):
            self.cfg.rov_model = str(self.opts.rov_model)
        if getattr(self.opts, "replay_session", None):
            # --replay-session PATH: point shape `replay` at a demo folder
            # without editing hw_mpc.yaml at the pool.
            self.cfg.replay["session"] = str(self.opts.replay_session)
        # `opts.policy_ckpt` (set DIRECTLY by tests/tools — test_policy_worker,
        # demo_e2e, policy_dryrun — since the `--policy-ckpt` flag was removed
        # on 2026-09-11 in favour of the panel picker) / --policy-repo
        # override the block the same way. NOTE the run meta no longer takes
        # the flown checkpoint from here: `_arm_policy` pins the one the
        # PolicyStatus HOLDS (the picker's choice); this is only the seed.
        if getattr(self.opts, "policy_ckpt", None):
            self.cfg.policy["ckpt"] = str(self.opts.policy_ckpt)
        if getattr(self.opts, "policy_repo", None):
            self.cfg.policy["repo"] = str(self.opts.policy_repo)
        # ONE root per run. --rec-dir governs the recorders and the nav
        # recording; without this line hw_mpc.yaml's log_dir governs the CSV
        # and events.log independently, and `--rec-dir /media/ssd/runs` would
        # split a run across two trees with nothing saying so. Explicit
        # `mpc_log_dir` still wins if someone genuinely wants them apart.
        root = (getattr(self.opts, "mpc_log_dir", None)
                or getattr(self.opts, "rec_dir", None))
        if root and str(root) != self.cfg.log_dir:
            self._log(
                "info", f"ctrl: writing runs under {root} "
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
        self._setup_policy()
        # Both controller families are built up front: the PID costs nothing
        # and stays available even when the acados build fails, so the mode
        # combo can swap between them without a rebuild (never mid-engage —
        # set_mode refuses that).
        from .pid import HwPid

        self._pid = HwPid(self.cfg, log=lambda m: self._log("info", m))
        # RL follower (2026-09-21): numpy-only, built up front like the PID; a missing/mismatched policy file only
        # costs the ``rl`` mode (set_mode refuses it with the error), never the session.
        self._rl = None
        try:
            from .rl_policy import HwRl
            self._rl = HwRl(self.cfg, log=lambda m: self._log("info", m))
        except Exception as e:                                   # noqa: BLE001
            self._rl_error = f"{type(e).__name__}: {e}"
            self._log("warn", f"ctrl: RL controller unavailable ({self._rl_error}) — rl mode will refuse")
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
                    self.cfg, log=lambda m: self._log("info", m))
        except Exception as e:                                   # noqa: BLE001
            self._setup_error = f"{type(e).__name__}: {e}"
            self._log("error", f"ctrl: MPC controller build FAILED — "
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
                    log=lambda m: self._log("info", m))
            except Exception as e:                               # noqa: BLE001
                self._log(
                    "warn", f"ctrl: MPCC unavailable ({type(e).__name__}: {e}) "
                            f"— mpcc mode will refuse")
        self.ctrl = self._ctrl_for(self.cfg.mode)
        self._ready = True
        if self.cfg.mode == "none":
            # Preselected at launch (alias / --mpc-mode none / mode: none).
            # The same DISARM sentence set_mode("none") logs, so a session
            # that never touched the combo still has it in the mission log.
            # No `_event` here: that would open a run folder for the mere
            # act of launching (the 2026-08-14 seven-empty-folders lesson).
            self._log("warn", self._low_none_banner())
        elif self.ctrl is not None:
            k = self.ctrl.solver_kind
            ok = self.ctrl.realtime_ok
            self._log(
                "warn" if not ok else "info",
                f"ctrl: {self.cfg.mode} ready, solver={k}, "
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

    # ------------------------------------------------------------- policy
    def _setup_policy(self) -> None:
        """The policy mission's process-lifetime state: the jaw-width
        estimator (v2 A12 — it integrates every drive from process start, so
        the width the policy is told at ARM is the width the pilot left the
        jaw at, not a fresh assumption) and the TCP geometry (v2 A3 — the
        ROV's OWN jaw, derived from body FLU through the C3 extrinsic unless
        ``tcp_offset_cam_m`` overrides it explicitly)."""
        from .policy_frames import (EtaHistory, GripperWidthEstimator,
                                    T_body_tcp, tcp_offset_from_body)

        pc = self.cfg.policy
        self._grip_est = GripperWidthEstimator(
            w_open=float(pc["gripper_width_open_m"]),
            w_closed=float(pc["gripper_width_closed_m"]),
            w_init=float(pc["gripper_width_init_m"]),
            travel_s=float(pc["gripper_travel_s"]))
        self._policy_eta_hist = EtaHistory(keep_s=3.0)
        R_bc, t_bc = self.nav_cfg.R_t_frd_cam("main")
        off = pc.get("tcp_offset_cam_m")
        if off is None:
            t_bt = S_FLU_FRD @ np.asarray(pc["tcp_body_flu_m"], float)
            off = tcp_offset_from_body(R_bc, t_bc, t_bt)
        self._policy_tcp_off_cam = np.asarray(off, float).reshape(3)
        self._policy_T_bt = T_body_tcp(R_bc, t_bc, self._policy_tcp_off_cam)

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
            self._log(
                "info", "imu_dr: demo source — using an IDENTITY calibration "
                        f"({cpath!r} is for the real camera and would inject "
                        f"its bias into a synthetic sensor)")
            cpath = None
        if cpath and Path(cpath).exists():
            try:
                calib = ImuCalibration.from_json(cpath)
            except (OSError, ValueError, KeyError) as e:         # noqa: BLE001
                self._log("error", f"imu_dr: calibration {cpath} "
                                           f"unreadable ({e}) — running RAW")
        elif str(getattr(self.opts, "source", "")) != "demo":
            self._log(
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
            self._log(
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
            self._log("error", f"object_nav: {e} — DISABLED")
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
            self._log(
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
            self._log("error", f"station_bridge: {e} — DISABLED")
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
                self._log("error", f"station_bridge: calibration "
                                           f"{cpath} unreadable ({e})")
        want = cfg["xy_source"]
        have_calib = bool(calib.sha1) or demo
        self._bridge_xy = ("hold" if want == "hold"
                           else "imu" if want == "imu"
                           else ("imu" if have_calib else "hold"))
        if want == "imu" and not have_calib:
            self._log(
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
                self._log(
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
            self._log("warn", f"ctrl: {br.detail()} — nothing is "
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
        self._log("info", f"ctrl: {msg}")

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
        # ...and a controller may carry its own, LOWER ceiling (HwRl: rl.axis_cap, 2026-09-21 — the policy
        # saturates at 20 cm of error and reached 0.45 m/s under 0.5). It can only tighten, never widen.
        alt = getattr(self.ctrl, "axis_cap", None)
        if alt is not None:
            cap = min(cap, float(alt))
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

    @staticmethod
    def _fmt_ms(v) -> str:
        return "?" if v is None else f"{float(v):.1f}"

    def _log_probe(self, msg: str) -> None:
        self._log("info", f"ctrl: {msg}")

    def _allowed_flight_modes(self) -> tuple:
        """``engage.require_mode`` as a tuple of ArduSub mode names (empty =
        no gate). Normalised here too, so a hand-built MpcConfig() whose
        default is the plain string "MANUAL" gates the same way a loaded
        YAML does."""
        from .geometry import normalize_require_mode

        e = self.cfg.engage if self.cfg is not None else {}
        return normalize_require_mode(e.get("require_mode", "MANUAL"))

    # ------------------------------------------------ 6-DoF attitude axes
    # engage.attitude_axes and the policy attitude keys, WITH DEFAULTS
    # (2026-09-26, design D7-D10): geometry.py owns the validated block in
    # config/hw_mpc.yaml; this worker reads through `.get` so a hand-built
    # MpcConfig() (every offline test) and a config file that predates the
    # variant both resolve to the OFF state. Numbers are [예측] unless the
    # config's own provenance says otherwise.
    ATTITUDE_AXES_DEFAULTS = {
        "enabled": False, "transport": "manual_control_ext",
        "cap_roll": 0.2, "cap_pitch": 0.3, "first_water_caps": [0.1, 0.15],
        "slew_per_s": 1.5, "sign": {"roll": 1.0, "pitch": 1.0},
        "abort_deg": 35.0, "sat_ineffective_s": 1.0,
        "require_manual": True, "require_probe": True,
        "require_sign_probe": True,
        "probe": None, "sign_probe": None, "dobmpc_allowed": False,
        "min_firmware": "4.1.2",
    }
    #: The `tool` a bench probe artefact must name (rov_gui/tools/
    #: attitude_axes_probe.py writes it); any other JSON is not a probe.
    ATTITUDE_PROBE_TOOL = "rov_gui.tools.attitude_axes_probe"
    POLICY_ATTITUDE_DEFAULTS = {
        "attitude_track": False, "rp_max_deg": 20.0, "rp_reject_deg": 30.0,
        "pq_max_rad_s": 0.35, "rp_jump_max_deg": 5.0,
        "anchor_leash_rp_deg": 3.0, "div_max_rp_deg": 15.0,
        "blend_rp_rate_max": 0.5, "heave_trim_attitude_rotated": False,
        "rp_ref_filter": None, "attitude_q_scale": 1.0,
    }
    #: LOW levels that may fly the attitude axes without an extra flag
    #: (D10): plain NMPC, whose K/M go out uncredited to any observer. The
    #: dobmpc family needs engage.attitude_axes.dobmpc_allowed (crediting a
    #: [유도] roll_nm/pitch_nm to the EAOB is a multiplicative plant-gain
    #: error at the DOB bandwidth); pid / rl / mpcc have no K/M path at all.
    ATTITUDE_LOW_ALLOWED = ("mpc", "mpc_tuned")
    RP_RESIDUAL_ARM_MAX_DEG = 10.0     # [예측] interlock (iv)

    def _attitude_cfg(self) -> dict:
        e = self.cfg.engage if self.cfg is not None else {}
        raw = e.get("attitude_axes") or {}
        d = dict(self.ATTITUDE_AXES_DEFAULTS)
        if isinstance(raw, dict):
            d.update(raw)
        sign = d.get("sign") or {}
        d["sign"] = {"roll": float((sign or {}).get("roll", 1.0)),
                     "pitch": float((sign or {}).get("pitch", 1.0))}
        return d

    def _policy_att(self, key: str):
        """A policy attitude knob, the config's value or the default."""
        pc = self.cfg.policy if self.cfg is not None else {}
        v = pc.get(key, self.POLICY_ATTITUDE_DEFAULTS[key])
        return self.POLICY_ATTITUDE_DEFAULTS[key] if v is None and key != "rp_ref_filter" else v

    def _policy_pinned_repr(self) -> str:
        """The action representation THIS side flies: policy.action_repr
        (config, default state.POLICY_ACTION_REPR), pinned per mission at
        ARM into rp["action_repr_pinned"] / scen["action_repr"]."""
        pc = self.cfg.policy if self.cfg is not None else {}
        return str(pc.get("action_repr") or POLICY_ACTION_REPR)

    @staticmethod
    def _parse_fw(v) -> tuple | None:
        """'4.1.2' (or '4.1.2-dev', 'ArduSub 4.5.1') -> (4, 1, 2); None when
        no a.b.c triple can be read — and None REFUSES ("" means the
        AUTOPILOT_VERSION never arrived, not 'any version')."""
        import re
        m = re.search(r"(\d+)\.(\d+)\.(\d+)", str(v or ""))
        return tuple(int(x) for x in m.groups()) if m else None

    @staticmethod
    def _alloc_has_attitude() -> bool:
        """Does THIS build's allocation carry the K/M path (wrench_to_axes
        `attitude=` kwarg)? The worker calls the pre-variant signature
        whenever the axes are off, so a build without it is still a valid
        4-DoF station — it just cannot fly the variant."""
        try:
            return "attitude" in inspect.signature(wrench_to_axes).parameters
        except (TypeError, ValueError):
            return False

    def _attitude_refusal(self, tel) -> str:
        """Why engage.attitude_axes.enabled may NOT be honoured at this
        engage ("" = every gate passed). Each cause its own sentence, the
        `_follow_refusal` rule. Called ONLY when the block is enabled and the
        vehicle gates apply (not land_dry_run, not observe)."""
        a = self._attitude_cfg()
        tr = str(a.get("transport") or "manual_control_ext")
        if tr != "manual_control_ext":
            return (f"engage.attitude_axes.transport {tr!r} is not implemented "
                    f"(manual_control_ext only; rc_override is a documented "
                    f"fallback, not built)")
        mode = str(getattr(tel, "mode", "") or "").upper()
        if bool(a.get("require_manual", True)) and not mode.startswith("MANUAL"):
            return (f"attitude_axes needs MANUAL (in {getattr(tel, 'mode', None) or '?'} "
                    f"the MANUAL_CONTROL s/t extension axes are LEAN-ANGLE "
                    f"targets and the autopilot would fight the NMPC's K/M "
                    f"torques) — press MANUAL or set engage.attitude_axes."
                    f"enabled: false")
        low = str(self.cfg.mode)
        if low.startswith("dobmpc"):
            if not bool(a.get("dobmpc_allowed", False)):
                return (f"attitude_axes on {low} needs engage.attitude_axes."
                        f"dobmpc_allowed: true (the EAOB would be credited an "
                        f"uncalibrated roll_nm/pitch_nm — calibrate on plain "
                        f"mpc first, D10)")
        elif low not in self.ATTITUDE_LOW_ALLOWED:
            return (f"attitude_axes has no K/M path on {low} — pick "
                    f"{'|'.join(self.ATTITUDE_LOW_ALLOWED)} (dobmpc* with "
                    f"dobmpc_allowed)")
        if not self._alloc_has_attitude():
            return ("this build's allocation has no attitude path "
                    "(wrench_to_axes lacks `attitude=`) — the variant cannot "
                    "be flown on it")
        if tel is None:
            return "attitude_axes: no telemetry"
        if not bool(getattr(tel, "attitude_axes_enabled", False)):
            return ("attitude_axes: the command sink is NOT configured to send "
                    "the s/t extension axes (Telemetry.attitude_axes_enabled "
                    "False — the sink reads engage.attitude_axes at setup; "
                    "restart the station with the block enabled)")
        if bool(getattr(tel, "attitude_axes_degraded", False)):
            return ("attitude_axes: the command sink already DEGRADED to the "
                    "4-axis frame (extension kwargs raised) — restart the link")
        wire = str(getattr(tel, "mavlink_wire_version", "") or "")
        if wire != "2.0":
            return (f"attitude_axes needs a MAVLink 2 command link "
                    f"(wire version {wire or 'unknown'}; the v1 dialect's "
                    f"manual_control_send has no s/t)")
        fw_s = str(getattr(tel, "firmware_version", "") or "")
        fw = self._parse_fw(fw_s)
        need = self._parse_fw(a.get("min_firmware", "4.1.2")) or (4, 1, 2)
        if fw is None:
            return ("attitude_axes: firmware version unknown (no "
                    "AUTOPILOT_VERSION received) — ArduSub < 4.1.2 silently "
                    "ignores s/t, so an unread version is refused")
        if fw < need:
            return (f"attitude_axes needs ArduSub >= "
                    f"{'.'.join(str(v) for v in need)} (vehicle reports "
                    f"{fw_s}) — s/t are ignored below it")
        chk = self._attitude_artefacts(a, fw_s)
        if chk["why"]:
            return chk["why"]
        if not chk["sign_proven"] and a.get("first_water_caps") is None:
            return ("attitude_axes: the wire sign is NOT proven (no sign_probe "
                    "artefact with sign_proven true) and engage.attitude_axes."
                    "first_water_caps is null — nothing would bound an "
                    "inverted sign; pin first_water_caps or the artefact")
        return ""

    @staticmethod
    def _read_json_artefact(path) -> tuple:
        """``(dict, sha1_hex, "")`` for a readable JSON object at ``path``,
        else ``(None, sha1_or_None, why)``. The sha1 is of the BYTES on disk
        (what the meta pins), computed whenever the file could be read."""
        try:
            raw = Path(str(path)).read_bytes()
        except OSError as e:
            return None, None, f"cannot be read ({type(e).__name__}: {e})"
        sha = hashlib.sha1(raw).hexdigest()
        try:
            d = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as e:
            return None, sha, f"is not valid JSON ({type(e).__name__}: {e})"
        if not isinstance(d, dict):
            return None, sha, f"is not a JSON object (got {type(d).__name__})"
        return d, sha, ""

    def _attitude_artefacts(self, a: dict, fw_vehicle) -> dict:
        """Read and JUDGE the two artefacts engage.attitude_axes pins
        (safety audit 2026-09-26: the gate used to be ``Path.exists()``, so
        any file — an empty one — armed the variant).

        ``probe`` (bench, disarmed; rov_gui.tools.attitude_axes_probe): when
        ``require_probe`` it must be a JSON object whose ``tool`` is
        :attr:`ATTITUDE_PROBE_TOOL`, ``pass`` is true, ``mavlink_wire_version``
        is "2.0" and whose ``firmware_version`` parses to the SAME a.b.c as
        the vehicle's (a probe from another firmware proves nothing about
        this one; both are named in the refusal).

        ``sign_probe`` (armed, in water): ``sign_proven`` true proves the wire
        sign and unlocks cap_roll / cap_pitch. When ``require_sign_probe`` it
        must be pinned and proven; ``require_sign_probe: false`` is the
        RECORDED opt-out — the run then flies first_water_caps
        (:meth:`set_engaged`). A pinned artefact that cannot be read or is
        not a JSON object refuses in every case: a path that names a
        non-artefact is a config error, not an opt-out.

        Returns ``{"why": "" | refusal, "probe_sha1", "sign_probe_sha1",
        "sign_proven", "probe_firmware_version"}``; the sha1s are recorded
        in meta run.attitude_axes whenever the file was readable."""
        out = {"why": "", "probe_sha1": None, "sign_probe_sha1": None,
               "sign_proven": False, "probe_firmware_version": None}
        need_probe = bool(a.get("require_probe", True))
        need_sign = bool(a.get("require_sign_probe", True))
        # ---- 1. the bench probe
        path = a.get("probe")
        if need_probe and not path:
            out["why"] = ("attitude_axes: engage.attitude_axes.probe is unset "
                          "— run the disarmed bench probe (python -m "
                          "rov_gui.tools.attitude_axes_probe) and pin its "
                          "artefact path (or require_probe: false, recorded)")
            return out
        if path:
            if not Path(str(path)).exists():
                out["why"] = (f"attitude_axes: engage.attitude_axes.probe "
                              f"{path!r} does not exist")
                return out
            d, sha, why = self._read_json_artefact(path)
            out["probe_sha1"] = sha
            if why:
                out["why"] = (f"attitude_axes: engage.attitude_axes.probe "
                              f"{path!r} {why} — not a probe artefact")
                return out
            out["probe_firmware_version"] = (
                None if d.get("firmware_version") is None
                else str(d.get("firmware_version")))
            if need_probe:
                why = self._judge_probe(d, fw_vehicle)
                if why:
                    out["why"] = (f"attitude_axes: engage.attitude_axes.probe "
                                  f"{path!r} {why}")
                    return out
        # ---- 2. the armed in-water sign probe
        path = a.get("sign_probe")
        if need_sign and not path:
            out["why"] = ("attitude_axes: engage.attitude_axes.sign_probe is "
                          "unset — the armed in-water sign probe has no "
                          "mission in this cut, so nothing writes one: pin a "
                          "real artefact (sign_proven true) or opt out with "
                          "require_sign_probe: false (RECORDED; the run then "
                          "flies first_water_caps)")
            return out
        if path:
            if not Path(str(path)).exists():
                out["why"] = (f"attitude_axes: engage.attitude_axes.sign_probe "
                              f"{path!r} does not exist")
                return out
            d, sha, why = self._read_json_artefact(path)
            out["sign_probe_sha1"] = sha
            if why:
                out["why"] = (f"attitude_axes: engage.attitude_axes.sign_probe "
                              f"{path!r} {why} — not a sign-probe artefact")
                return out
            out["sign_proven"] = d.get("sign_proven") is True
            if need_sign and not out["sign_proven"]:
                out["why"] = (f"attitude_axes: engage.attitude_axes.sign_probe "
                              f"{path!r} has sign_proven "
                              f"{d.get('sign_proven')!r}, not true — the wire "
                              f"sign is not proven (the bench probe never "
                              f"proves it); re-run the armed in-water sign "
                              f"probe or opt out with require_sign_probe: "
                              f"false (RECORDED, first_water_caps)")
                return out
        return out

    def _judge_probe(self, d: dict, fw_vehicle) -> str:
        """Why a parsed bench-probe JSON does NOT clear the gate ("" = it
        does): tool, pass, wire, and a firmware a.b.c equal to the vehicle's."""
        tool = d.get("tool")
        if tool != self.ATTITUDE_PROBE_TOOL:
            return (f"names tool {tool!r}, not {self.ATTITUDE_PROBE_TOOL!r} — "
                    f"not a bench probe artefact")
        if d.get("pass") is not True:
            return (f"records pass {d.get('pass')!r}, not true (verdict "
                    f"{d.get('verdict')!r}: {d.get('verdict_note') or '-'}) "
                    f"— the firmware did not demonstrably read s/t")
        wire = str(d.get("mavlink_wire_version") or "")
        if wire != "2.0":
            return (f"was recorded on a MAVLink {wire or 'unknown'} link, not "
                    f"2.0 — s/t never left the station during that probe")
        fw_p_s = d.get("firmware_version")
        fw_p = self._parse_fw(fw_p_s)
        if fw_p is None:
            return (f"records firmware_version {fw_p_s!r}, which does not "
                    f"read as a.b.c — the probe did not receive "
                    f"AUTOPILOT_VERSION")
        fw_v = self._parse_fw(fw_vehicle)
        if fw_v is None or fw_p != fw_v:
            return (f"was recorded on ArduSub {fw_p_s!r} but the vehicle now "
                    f"reports {str(fw_vehicle or '') or 'unknown'!r} — a probe "
                    f"proves only the firmware it ran on; re-run the bench "
                    f"probe on this vehicle")
        return ""

    def _ref_attitude(self, t_traj) -> tuple:
        """The attitude REFERENCE (phi, theta) NED rad the follower is
        tracking now: the controller's own (plan rp | hold ramp | level)
        when it has the accessor, else level."""
        fn = getattr(self.ctrl, "ref_attitude_ned_at", None)
        if callable(fn):
            try:
                v = fn(float(t_traj or 0.0))
                return (float(v[0]), float(v[1]))
            except (TypeError, ValueError, IndexError):
                return (0.0, 0.0)
        return (0.0, 0.0)

    def _reset_attitude_interlocks(self) -> None:
        self._att_ceiling_ticks = 0
        self._att_sat = {0: None, 1: None}
        self._attitude_sat_ticks = 0

    #: Interlock (iii): an axis leaves its cap-pinned window only after this
    #: fraction of a second of CONSECUTIVE unpinned ticks (safety audit
    #: 2026-09-26: one unpinned tick used to reset the window, so an axis
    #: toggling in and out of the cap every few ticks was never judged).
    ATTITUDE_SAT_RESET_S = 0.25

    def _attitude_sat_watch(self, meas, axes, t_traj, t) -> str:
        """Interlock (iii), D9: an attitude axis pinned at its wire cap for
        >= engage.attitude_axes.sat_ineffective_s while the torque shows NO
        EFFECT means it is not doing what the model says — sign inverted,
        s/t ignored by the firmware, or saturated against a static moment
        with no progress. Returns the disengage reason, or "".

        "No effect" over the window, judged on the MEASURED attitude
        (safety audit 2026-09-26 — the first cut compared |e| against its
        value at the first pinned tick, which false-tripped on a MOVING
        reference: the ramp at pq_max outruns a capped axis, |e| grows, and
        the torque was effective all along):
          * progress = the measured angle's change along the sign of the
            axis command; > 1 deg [예측] = the torque moved the hull the way
            it pushes -> a fresh window from here;
          * |e| decreased by > 1 deg -> the same;
          * the reference itself moved > 1 deg over the window and the hull
            did not move AGAINST the command -> ambiguous (a slow follower or
            an ineffective torque look alike while the target runs away):
            the window is restarted, never tripped, until the reference
            holds still for one window;
          * the hull moved against the command by > 1 deg -> trip at once
            when the window fills, moving reference or not (an inverted
            sign is exactly this);
          * otherwise (pinned, no progress, |e| not shrinking, reference
            still) -> trip. A CONSTANT saturated error (the pitch cap 0.3 x
            7.2 = 2.2 N*m [유도] against the gripper's static moment) trips
            here on purpose.
        A window is reset only after ATTITUDE_SAT_RESET_S of consecutive
        unpinned ticks; a single unpinned tick keeps the pinned count. A
        command that flips sign while pinned starts a new window."""
        if meas is None or axes is None:
            return ""
        a = self._attitude_cfg()
        hold = float(a.get("sat_ineffective_s", 1.0))
        # Counted in TICKS like every other interlock here (the ceiling, the
        # divergence guard): the window is hold x ctrl_hz controller steps,
        # not wall time — a stalled process must not be judged by a clock
        # that kept running while no torque was being applied.
        hz = float(self.cfg.ctrl_hz)
        need = max(1, int(round(hold * hz)))
        reset_after = max(1, int(round(self.ATTITUDE_SAT_RESET_S * hz)))
        eta = np.asarray(meas["eta"], float)
        ref = self._ref_attitude(t_traj)
        tol = math.radians(1.0)
        for i, name in enumerate(("roll", "pitch")):
            val = float(getattr(axes, name, 0.0))
            cap = float(self._attitude_cap[i])
            ref_i = float(ref[i])
            att_i = float(eta[3 + i])
            e = abs(_wrap_pi(ref_i - att_i))
            pinned = cap > 0.0 and abs(val) >= cap - 1e-9
            st = self._att_sat[i]
            if not pinned:
                if st is not None:
                    st["unpinned"] = int(st.get("unpinned", 0)) + 1
                    if st["unpinned"] >= reset_after:
                        self._att_sat[i] = None
                continue
            self._attitude_sat_ticks += 1
            sgn = 1.0 if val > 0.0 else -1.0
            if st is not None and st.get("sgn") != sgn:
                st = None                      # a reversed command: new window
            if st is None:
                self._att_sat[i] = {"n": 1, "unpinned": 0, "sgn": sgn,
                                    "e0": e, "ref0": ref_i, "att0": att_i}
                continue
            st["unpinned"] = 0
            st["n"] = int(st["n"]) + 1
            if st["n"] < need:
                continue
            d_ref = _wrap_pi(ref_i - float(st["ref0"]))
            progress = _wrap_pi(att_i - float(st["att0"])) * sgn
            e_shrank = (e - float(st["e0"])) < -tol
            if e_shrank or progress > tol:
                # the torque is doing something: judge the NEXT window
                self._att_sat[i] = {"n": 0, "unpinned": 0, "sgn": sgn,
                                    "e0": e, "ref0": ref_i, "att0": att_i}
                continue
            if abs(d_ref) > tol and progress >= -tol:
                # the reference moved and the hull did not go the wrong way:
                # not judgeable yet — restart, do not trip
                self._att_sat[i] = {"n": 0, "unpinned": 0, "sgn": sgn,
                                    "e0": e, "ref0": ref_i, "att0": att_i}
                continue
            how = ("moved AGAINST the command"
                   if progress < -tol else "did not move with the command")
            return (f"attitude {name} axis pinned at its cap "
                    f"({val:+.2f}, cap {cap:.2f}) for {hold:.1f} s while the "
                    f"measured {name} {how} "
                    f"({math.degrees(progress):+.1f} deg along it; error "
                    f"{math.degrees(st['e0']):.1f} -> {math.degrees(e):.1f} "
                    f"deg, reference moved {math.degrees(d_ref):+.1f} deg): "
                    f"torque ineffective (sign / firmware / saturation) — "
                    f"disengaging")
        return ""

    @Slot(str)
    def on_mode_request(self, name: str) -> None:
        """The station asked the vehicle for a flight mode (teleop panel /
        ARM). HEARTBEAT confirms it up to 1 s later, so start the
        engage.mode_settle_s clock at the REQUEST, not at the echo — a START
        inside that second would otherwise be judged on the old mode. A
        request for MANUAL starts nothing (nothing settles there). The
        gamepad's own mode buttons bypass the station entirely and cannot be
        seen here."""
        m = str(name or "").upper()
        if m and not m.startswith("MANUAL"):
            self._mode_changed_at = now()

    @Slot(object)
    def on_telemetry(self, tel) -> None:
        self.tel = tel
        # Flight-mode change clock for engage.mode_settle_s: the first
        # telemetry sets the baseline without starting the timer.
        m_now = str(tel.mode) if (tel is not None and tel.mode) else None
        if m_now is not None:
            if self._tel_mode_prev is not None and m_now != self._tel_mode_prev:
                # Only a change INTO a self-driving mode starts the clock:
                # entering MANUAL neutralises every override and runs no
                # attitude loop, so there is nothing to settle (and ARM
                # itself requests MANUAL — a timer there would delay every
                # START by mode_settle_s for no reason).
                self._mode_changed_at = (None if m_now.upper().startswith("MANUAL")
                                         else now())
            self._tel_mode_prev = m_now
        # RECORD BOUNDARY (2026-09-07): the flight mode decides how ArduSub
        # interprets every axis the station sends (in STABILIZE the yaw axis
        # is a rate demand and roll/pitch are levelled by the autopilot), so
        # a run must say which mode(s) it flew under. Tracked while engaged,
        # written to meta run.flight_mode_at_engage / flight_modes_seen.
        if self.engaged and tel is not None:
            m = str(tel.mode or "?")
            seen = getattr(self, "_flight_modes_seen", None)
            if seen is not None and m not in seen:
                seen.append(m)
                self._log_event(f"flight mode changed to {m} while engaged "
                                f"(seen: {'>'.join(seen)})")

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

    #: How many thruster columns the CSV carries. Fixed, because a header
    #: cannot grow a column when the vehicle reports a ninth output.
    N_PWM_COLS = 8

    def _thruster_row(self) -> list[str]:
        """The eleven schema-14 columns: per-thruster PWM as reported, the age
        of that report, and the pack voltage / current.

        The MEAN (`_pwm_dev_us`) cannot say whether ANY single thruster cleared
        its deadband — only four of the eight drive a horizontal axis, so a
        mean of 5 us is consistent both with nothing spinning and with two
        thrusters at 20 us. Every 2026-09 stall analysis hit that wall. The age
        rides along because ThrusterState is republished at 10 Hz into a 20 Hz
        CSV: without it a repeated row looks like a fresh measurement."""
        t, tel = self.thr, self.tel
        pwm = list(getattr(t, "pwm_us", None) or [])
        out = [("" if (i >= len(pwm) or pwm[i] is None) else f"{int(pwm[i])}")
               for i in range(self.N_PWM_COLS)]
        stamp = getattr(t, "stamp", None) if t is not None else None
        out.append(f"{max(0.0, now() - float(stamp)):.3f}"
                   if stamp else "nan")
        for v in ((getattr(tel, "battery_v", None),
                   getattr(tel, "current_a", None)) if tel is not None
                  else (None, None)):
            out.append(f"{float(v):.2f}" if v is not None else "nan")
        return out

    @Slot(bool)
    def on_enable(self, on: bool) -> None:
        self.cmd_enabled = bool(on)
        if not on and self.engaged and not self.observe:
            # Not under POLICY OBSERVE: COMMAND ENABLE governs the ACTUATION
            # path, and an observe run has none. Killing the run when the
            # pilot safes their own commands would end the DP check for a
            # reason that has nothing to do with the DP.
            self.disengage("COMMAND ENABLE off")
        if not on and self._grip_est is not None:
            # ENABLE off zeroes the command sink, jaw drive included — the
            # estimator must stop integrating with it (v2 A12).
            self._grip_est.release(now())

    # -------------------------------------------------------- policy slots
    @Slot(object)
    def on_policy_plan(self, plan) -> None:
        """A PolicyPlan from the policy worker: latest-wins inbox.

        Dropped, and counted, when its ``epoch`` is not the current one
        (built before the last ENGAGE/ARM — its anchor pose and its history
        belong to a different datum or a different run, v2 A8) or when its
        ``plan_id`` does not advance (a re-delivery). Nothing is composed
        here: the intake runs inside the guarded tick, where a bad action
        can be rejected with a reason instead of raising in a slot.
        """
        try:
            epoch = int(getattr(plan, "epoch", -1))
            pid = int(getattr(plan, "plan_id", -1))
        except (TypeError, ValueError):
            return
        rp = self.replay
        counts = (rp if (rp is not None and rp.get("kind") == "policy")
                  else self._policy_counts)
        if epoch != self._policy_epoch:
            counts["drop_epoch"] = int(counts.get("drop_epoch", 0)) + 1
            return
        if pid <= self._policy_last_seen:
            counts["drop_old"] = int(counts.get("drop_old", 0)) + 1
            return
        self._policy_last_seen = pid
        if rp is None or rp.get("kind") != "policy":
            self._policy_counts["drop_inactive"] += 1
            return
        rp["received"] = int(rp.get("received", 0)) + 1
        self._policy_inbox = plan

    @Slot(object)
    def on_policy_status(self, st) -> None:
        """The worker's stamped status (v2 A13). A backend routes the
        worker's ``failed`` signal here as ``PolicyStatus(error=...)``.

        A checkpoint CHANGE while a controller CSV is open (REC pressed on
        the video feed, DISENG, a panel pick — the picker is gated on
        ``engaged`` only, review 2026-09-11) is announced and recorded:
        the rows before this moment inferred on the OLD network and the
        CSV cannot tell, so events.log and `policy.ckpt_changes_while_csv_open`
        (meta) carry the boundary."""
        prev = str(getattr(self._policy_status, "ckpt", "") or "")
        cur = str(getattr(st, "ckpt", "") or "")
        self._policy_status = st
        if self._csv is not None and prev and cur and prev != cur:
            self._ckpt_changes.append({
                "t_csv_s": (round(now() - self._t0_csv, 3)
                            if self._t0_csv else None),
                "rows_before": int(self._rows),
                "from": prev, "to": cur,
                "to_sha1": str(getattr(st, "ckpt_sha1", "") or "")})
            self._event(f"POLICY checkpoint changed while the controller "
                        f"CSV is open: {Path(prev).name} -> {Path(cur).name} "
                        f"(rows before: {self._rows}); rows before this line "
                        f"inferred on the OLD network")
            self._log("warn", f"ctrl: checkpoint {Path(prev).name} -> "
                              f"{Path(cur).name} while the controller CSV "
                              f"is open — this CSV now spans two networks; "
                              f"plans.jsonl (per-plan ckpt_sha1) and meta "
                              f"policy.ckpt_changes_while_csv_open separate "
                              f"them; stop REC before picking to avoid it")

    @Slot(float)
    def on_gripper_drive(self, v: float) -> None:
        """EVERY jaw drive — pilot G/H, the replay/policy edges, the neutrals
        — feeds the open-loop width estimator, and NOTHING else happens here
        (v2 A12): when the emitter is this worker the call is direct, and a
        slot that touched mission state would re-enter the tick."""
        if self._grip_est is not None:
            self._grip_est.drive(float(v), now())

    # mpc_tuned / dobmpc_tuned are the SAME solver object as mpc / dobmpc —
    # the suffix rotates the position weight into the path frame at run time
    # (control/path_cost.py), so there is no third build and the A/B differs
    # in exactly one thing.
    #
    # "none" (2026-09-11, operator request) is the LOW level of the panel's
    # HIGH/LOW split: TELEOP. Not a controller — see _ctrl_for. It lives in
    # THIS tuple (the panel's vocabulary) and in MpcConfig.load's; it must
    # NEVER join HwDobMpc.MODES, whose setter raises for non-members and
    # whose members are all things that get stepped.
    MODES = ("none", "mpc", "dobmpc", "mpc_tuned", "dobmpc_tuned",
             "mpcc", "dobmpcc", "pid", "rl")

    def _ctrl_for(self, mode: str):
        """The controller object a mode name selects, or None if unavailable.

        ``mpc``/``dobmpc`` share ONE tracking solver and ``mpcc``/``dobmpcc``
        share ONE contouring solver — in each pair the prefix only decides
        whether the EAOB's w_hat reaches the solver, exactly as in the sim.

        ``none`` (LOW level None = TELEOP, 2026-09-11) returns the PID as a
        REFERENCE HOLDER, never as a follower: it keeps ``scenario``,
        ``set_path_plan_ned`` / ``ref_ned_at`` / ``set_target_ned`` so a
        Diffusion Policy or Replay mission has somewhere to install the plan
        the panel draws — and it is never stepped (the tick's observe branch
        returns before ``ctrl.step``). THE PID, NOT ``_mpc_ctrl``, and on
        purpose: it is always built (no acados needed, ``realtime_ok`` True);
        ``_tick_replay`` sizes the NedPlan by ``path_plan_steps`` and
        ``ref_ned_at`` is stage 0 on both, so the drawn reference is
        identical; and it has NO ``set_plan_cost_scale``, so the run record's
        ``plan_q_scale_flown`` cannot claim a scale that never flew.
        ``set_mode("none")`` never touches ``_mpc_ctrl.mode``."""
        if mode in ("pid", "none"):
            return self._pid
        if mode == "rl":
            return getattr(self, "_rl", None)
        if mode in ("mpcc", "dobmpcc"):
            return self._mpcc_ctrl
        return self._mpc_ctrl

    def _low_none_banner(self) -> str:
        """The full DISARM list for LOW level None, one sentence — the same
        content as __main__'s --policy-observe launch banner. Logged at WARN
        by set_mode("none") (once per switch) and by setup() when the launch
        preselected it, so an operator who never saw the terminal still reads
        in the mission log which protections are simply not there."""
        # The kind is READ from `_run_tree`, not spelled here (review
        # 2026-09-11): a `--land-dry-run` station files under `_landdry`
        # BEFORE observe is consulted, and this is exactly the line that
        # exists to say where the record went.
        kind = (self._run_tree().kind if self.cfg is not None else "observe")
        return ("ctrl: LOW level = None — TELEOP. The station commands NOTHING "
                "(no ctrl.step, no axis/wrench/jaw leaves this worker); you "
                "fly on the joystick. A Diffusion Policy mission still infers "
                "and draws its plans. DISARMED (each assumes a follower): "
                "divergence guard, escalation latch, bridge-too-long stop, "
                "workspace box, tag-loss disengage. Jaw forced off; run clock "
                f"policy.observe_max_run_s; records go to data/*/*_{kind}/ "
                "(observe=1) and are NOT closed-loop results.")

    @Slot(str)
    def set_mode(self, mode: str) -> None:
        """The panel's LOW combo (bus.cmd_mpc_mode). ``none`` = TELEOP
        (2026-09-11, operator request): observe on, the PID kept only as a
        REFERENCE HOLDER (see _ctrl_for). The ONLY runtime writer of
        ``_observe``. Every refusal below is logged AND put in ``reason``:
        the panel re-syncs its combo from MpcStatus.mode (honesty rule), so
        a silently dropped request would read as a combo that snapped back
        for no reason (review 2026-09-11)."""
        mode = str(mode).lower()
        if mode not in self.MODES:
            return
        if self.engaged:
            self.reason = "mode change refused while engaged"
            self._log("warn", "ctrl: mode change refused while engaged")
            return
        if self.cfg is None or not self._ready:
            self.reason = ("mode change refused: controller still building — "
                           "pick again when READY")
            self._log("warn", f"ctrl: {self.reason}")
            return
        if self._csv is not None:
            # A REC-opened CSV (set_sensor_log) PINNED the run folder under
            # the tree of the mode it was opened in (`_run_dir_pin`). A
            # switch into or out of "none" would put the rest of the record
            # — meta, plans.jsonl, events.log — in a folder that belongs to
            # the OTHER kind, and the kind is the pooling guard (`_run_tree`).
            self.reason = ("mode change refused while a controller CSV is "
                           "open (REC) — stop the recording first")
            self._log("warn", f"ctrl: {self.reason}")
            return
        self._apply_plan_cost_scale(1.0)       # the outgoing follower forgets it
        self._apply_plan_path_cost(None)
        ctrl = self._ctrl_for(mode)
        if ctrl is None:
            why = (getattr(self, "_rl_error", None) if mode == "rl" else None) or self._setup_error or "not built"
            self.reason = f"mode {mode} unavailable ({why})"
            self._log("error", f"ctrl: mode {mode} unavailable ({why})"
                                       f" — staying on {self.cfg.mode}")
            return
        prev = str(self.cfg.mode)
        self.ctrl = ctrl
        # "pid" has no mode to set; "none" must NOT touch anything — the
        # holder is the PID and `_mpc_ctrl.mode` stays whatever it was
        # (HwDobMpc's setter raises for a name outside its MODES).
        if mode not in ("pid", "none", "rl") and mode != getattr(ctrl, "mode", mode):
            ctrl.mode = mode
        self.ctrl.reset()
        self._path_cursor = None
        self._path_err = (None, None)
        self._path_depth = 0.0
        self._path_yaw_fixed = 0.0
        self._path_heading_follow = False
        self.cfg.mode = mode
        # The PREVIOUS run's counters must not outlive the LOW switch (review
        # 2026-09-11): `_policy_meta` derives observe / synthetic /
        # gates_enforced from the CURRENT level but `run` from
        # `_replay_last`, and a REC NAV press (dump_run_meta is not gated on
        # engagement) between engagements would write a controller.json
        # saying `policy.observe: false` beside `policy.run.observe: true`.
        # Those counters were already written to their own meta.json when
        # that run's CSV closed (set_mode refuses while one is open).
        self._replay_last = None
        if mode == "none":
            self._observe = True
            # WHAT IS DISARMED, ONCE PER SWITCH, IN FULL — WARN, because the
            # operator is about to fly a run whose every automatic stop is
            # off, and the combo label "None" cannot carry that.
            self._log("warn", self._low_none_banner())
            # Screen half now; FILE half held for the next run folder
            # (pending): a plain `_event` here opened a one-line
            # data/<date>/<hhmmss>_observe/events.log per combo
            # click at the bench — the 2026-08-14 "seven folders left
            # behind" pattern (review 2026-09-11).
            self._event("LOW None (teleop) selected", pending=True)
            self._slam_tree_caveat()
            return
        self._observe = False
        if prev == "none":
            self._log("info", f"ctrl: LOW level = {mode} — the station "
                              f"commands again")
            # The other direction leaves the same trace in the NEXT
            # events.log (symmetry with the None line; file half only).
            self._log_event(f"LOW {mode} selected — the station commands "
                            f"again", pending=True)
        self._log("info", f"ctrl: mode = {mode} "
                                  f"({self.ctrl.solver_kind})")
        self._slam_tree_caveat()

    def _slam_tree_caveat(self) -> None:
        """After a successful LOW switch under ``--slam``: the ORB-SLAM3
        outputs do NOT follow the switch (review 2026-09-11). slam.py
        resolves the worker's ``_run_tree`` ONCE, in ``_start_child`` on
        the first stereo pair (``proc.start()`` runs once; ``_faulted`` is
        terminal), so `orbslam.yaml` / `traj*` stay under whichever tree
        that moment resolved to — e.g. the water tree for a session that
        launched on DOBMPC and later flies an observe run under None."""
        if not bool(getattr(self.opts, "slam", False)):
            return
        self._log("warn", "ctrl: SLAM outputs stay under the tree resolved "
                          "when the ORB-SLAM3 child started (first stereo "
                          "pair) — a LOW change does not move them")

    @Slot(object)
    def set_scenario(self, d: dict) -> None:
        if isinstance(d, dict):
            self._scenario_override = dict(d)

    # ------------------------------------------------------------ engagement
    @property
    def observe(self) -> bool:
        """POLICY OBSERVE: is the controller output MUTED?

        TRUE iff the LOW level is ``none`` (TELEOP; ``--policy-observe`` is
        the launch alias that preselects it, 2026-09-11). READ-ONLY on
        purpose — no setter: the two writers are setup() (launch) and
        set_mode() (the panel), and nothing else may flip the one bit every
        actuation guard, the record boundary and the run tree all hang on.
        The mission machinery runs at full speed under it (engaged, datum,
        CSV, PolicyState feed, plans composed and drawn); `ctrl.step` never
        runs and not one byte leaves this worker toward the vehicle.
        """
        return bool(self._observe)

    @property
    def commanding(self) -> bool:
        """Is this worker DRIVING the vehicle right now?

        The predicate every actuation site must use. `engaged` is not it:
        under POLICY OBSERVE (LOW level None; --policy-observe is the launch
        alias) the mission machinery runs at full speed and nothing goes
        out. A property rather than a second stored bool so the two can
        never be set apart from each other, and so the only way to stop
        commanding is the one way that also stops the mission.
        """
        return bool(self.engaged) and not self.observe

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
            self._log("warn", f"ctrl: {self.reason}")
            self._log_event(self.reason)
            return
        self.asm.reset()
        self.asm.calibrate_z_offset(self.fix, self.imu)
        meas, h = self.asm.step(self.fix, self.imu, now(), 1.0 / self.cfg.ctrl_hz)
        if meas is None:
            self.reason = f"engage refused: {h['why']}"
            self._log("warn", f"ctrl: {self.reason}")
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
        # A NEW EPOCH (v2 A8): the datum just changed, so every PolicyPlan
        # in flight — and every eta row the anchor could interpolate — was
        # built in a frame that no longer exists.
        self._policy_epoch += 1
        self._policy_inbox = None
        self._policy_last_seen = -1
        if self._policy_eta_hist is not None:
            self._policy_eta_hist.clear()
            self._policy_hist_last_fix = None
        self._t0_traj = None
        self._path_cursor = None
        self._path_err = (None, None)
        self._path_depth = 0.0
        self._path_yaw_fixed = 0.0
        self._path_heading_follow = False
        self._t_engage = now()
        fm = str(self.tel.mode) if (self.tel is not None and self.tel.mode) else "?"
        self._flight_mode_at_engage = fm
        self._flight_modes_seen = [fm]
        st = (self.cfg.engage.get("stabilize") or {}) if self.cfg else {}
        self._yaw_hold = bool(
            fm.upper().startswith("STABILIZE")
            and str(st.get("yaw_axis", "hold")).strip().lower() == "hold"
            and not self.land_dry_run and not self.observe)
        # The RECORD of that decision outlives the engagement (disengage
        # clears the live flag before the closing meta is written).
        self._yaw_hold_at_engage = self._yaw_hold
        # THE 6-DoF VARIANT'S ONE BIT (2026-09-26). `_engage_refusal` has
        # already run every gate (MANUAL, LOW allow-list, v2 link, firmware,
        # sink configured, probe artefacts), so here it is only resolved and
        # RECORDED. Off under observe / land dry-run: nothing is actuated
        # there, so K/M would be a claim about a wire that carries nothing.
        att = self._attitude_cfg()
        want = bool(att.get("enabled", False))
        self._attitude_axes = bool(want and not self.land_dry_run
                                   and not self.observe)
        # THE CAPS IN FORCE (safety audit 2026-09-26: first_water_caps was
        # validated and never applied). The wire sign is PROVEN only by a
        # sign_probe artefact with sign_proven true; without it the caps on
        # the wire are first_water_caps (elementwise min with cap_roll /
        # cap_pitch), whatever the config's full caps say. The gate has
        # already refused an unproven sign with first_water_caps null, so
        # the last branch is reachable only where nothing is actuated.
        fw_s = str(getattr(self.tel, "firmware_version", "") or "")
        chk = (self._attitude_artefacts(att, fw_s) if want else
               {"why": "", "probe_sha1": None, "sign_probe_sha1": None,
                "sign_proven": False, "probe_firmware_version": None})
        caps_full = (float(att.get("cap_roll", 0.2)),
                     float(att.get("cap_pitch", 0.3)))
        fwc = att.get("first_water_caps")
        fwc = (None if fwc is None else
               (float(fwc[0]), float(fwc[1])))
        if chk["sign_proven"]:
            self._attitude_cap = caps_full
            caps_set = "full"
            caps_why = (f"wire sign proven: sign_probe "
                        f"{att.get('sign_probe')!r} sign_proven true "
                        f"(sha1 {str(chk['sign_probe_sha1'] or '')[:12]})")
        elif fwc is not None:
            self._attitude_cap = (min(caps_full[0], fwc[0]),
                                  min(caps_full[1], fwc[1]))
            caps_set = "first_water"
            caps_why = ("wire sign NOT proven ("
                        + (f"sign_probe {att.get('sign_probe')!r} sign_proven "
                           f"false"
                           if att.get("sign_probe") else
                           "no sign_probe artefact pinned; "
                           "require_sign_probe false = recorded opt-out")
                        + ") — first_water_caps in force, cap_roll/cap_pitch "
                          "withheld")
        else:
            self._attitude_cap = caps_full
            caps_set = "full"
            caps_why = ("first_water_caps null with an unproven sign — the "
                        "engage gate refuses this on a vehicle; reached only "
                        "with nothing actuated (observe / land dry-run)")
        if not self._attitude_axes:
            caps_set = None
            caps_why = ("axes not live: nothing on the wire ("
                        + ("variant off" if not want else
                           "observe" if self.observe else "land dry-run")
                        + ")")
        self._reset_attitude_interlocks()
        self._att_degraded_at = None
        if self._attitude_axes:
            # The slew memory starts at NEUTRAL (safety audit 2026-09-26):
            # with None the first frame passed through unslewed and a
            # follower asking for full K/M put +-cap on the wire on tick 1.
            # Seeded ONLY under the variant — the 4-DoF path keeps None and
            # its pass-through first frame, byte-identical.
            self._axes_prev = (0.0,) * 6
        if self._bridge is not None:
            # The station bridge zeroes K/M on its own coast rows only when
            # it knows the axes are live, and its meta
            # ``released_in_coast`` reports that same bit — keep it in step
            # with the worker's LIVE flag (integration seam, 2026-09-26).
            self._bridge.attitude_axes = bool(self._attitude_axes)
        rc = getattr(self.tel, "rc_chan_raw", None) if self.tel is not None else None
        gains = dict(self.cfg.axis_gain) if self.cfg else {}
        roll_nm = float(gains.get("roll_nm", 13.2))
        pitch_nm = float(gains.get("pitch_nm", 7.2))
        self._attitude_at_engage = {
            "enabled": bool(self._attitude_axes),
            "configured": want,
            "transport": str(att.get("transport") or "manual_control_ext"),
            "roll_nm": roll_nm, "pitch_nm": pitch_nm,
            "cap_roll": self._attitude_cap[0],
            "cap_pitch": self._attitude_cap[1],
            "slew_per_s": float(att.get("slew_per_s", 1.5)),
            "sign": dict(att.get("sign") or {}),
            "mavlink_wire_version": str(getattr(self.tel, "mavlink_wire_version", "") or ""),
            "firmware_version": fw_s,
            "probe": att.get("probe"), "sign_probe": att.get("sign_probe"),
            # The artefacts as READ at this engage (audit 2026-09-26): what
            # was pinned, byte for byte, and what it proved.
            "probe_sha1": chk["probe_sha1"],
            "sign_probe_sha1": chk["sign_probe_sha1"],
            "sign_proven": bool(chk["sign_proven"]),
            "probe_firmware_version": chk["probe_firmware_version"],
            "require_probe": bool(att.get("require_probe", True)),
            "require_sign_probe": bool(att.get("require_sign_probe", True)),
            "cap_roll_configured": caps_full[0],
            "cap_pitch_configured": caps_full[1],
            "first_water_caps": (None if fwc is None else list(fwc)),
            "caps_in_force": caps_set,
            "caps_in_force_why": caps_why,
            "rc_trim_at_engage": (None if not rc else
                                  [int(v) for v in list(rc)[:2]]),
            "enabled_extensions_sent": bool(self._attitude_axes),
            "degraded_at": None, "sat_ticks": 0,
            "u_max_wire_nm": ([self._attitude_cap[0] * roll_nm,
                               self._attitude_cap[1] * pitch_nm]
                              if self._attitude_axes else None),
        }
        set_att = getattr(self.ctrl, "set_attitude_axes", None)
        if callable(set_att):
            # The follower learns whether its K/M reach the wire (meta
            # controller.allocation.attitude, the w_hat[3:5] boundary).
            set_att(bool(self._attitude_axes),
                    u_max_wire_nm=self._attitude_at_engage["u_max_wire_nm"])
        if want and not self._attitude_axes:
            self._log("info", f"ctrl: engage.attitude_axes is enabled but the "
                              f"axes are NOT sent on this run "
                              f"({'observe' if self.observe else 'land dry-run'}"
                              f": nothing is actuated)")
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
                        f"yaw0 {math.degrees(self._datum['yaw0']):.1f} deg; "
                        f"vehicle flight mode {fm}")
        if not self.land_dry_run and not self.observe and \
                not fm.upper().startswith("MANUAL"):
            # The gate let a self-driving mode through (engage.require_mode
            # lists it). Say what that changes about the axes this loop
            # sends, once, where the operator reads.
            yaw_note = (
                "the station's yaw axis is HELD at 0 — the autopilot's heading "
                "hold owns heading, so heading_follow / circle / policy dyaw "
                "will NOT turn the vehicle (engage.stabilize.yaw_axis: hold)"
                if self._yaw_hold else
                "the station's yaw axis goes out as TORQUE, which ArduSub reads "
                "as a yaw-RATE demand: |axis| <= 0.10 (N <= 2 N*m) is ignored "
                "and heading is held 250 ms later (engage.stabilize.yaw_axis: "
                "torque — experiment)")
            self._log("warn",
                      f"ctrl: ENGAGED with the vehicle in {fm} — the autopilot "
                      f"is driving too: it levels roll/pitch itself; "
                      f"surge/sway/heave pass through; {yaw_note}. This run is "
                      f"NOT comparable with a MANUAL run (meta "
                      f"run.flight_mode_at_engage).")
        self._log(
            "debug", f"ctrl: ENGAGED ({self.cfg.mode}) — holding "
                     f"({meas['eta'][0]:+.2f}, {meas['eta'][1]:+.2f}, "
                     f"{meas['eta'][2]:+.2f}) NED. START TRAJ begins the "
                     f"square.")
        if self._attitude_axes:
            ae = self._attitude_at_engage
            self._log("warn",
                      f"ctrl: ATTITUDE AXES ON — the NMPC's K/M go out as "
                      f"MANUAL_CONTROL s/t (caps IN FORCE: {ae['caps_in_force']} "
                      f"roll {ae['cap_roll']:.2f} / pitch {ae['cap_pitch']:.2f} "
                      f"— {ae['caps_in_force_why']}; configured "
                      f"{ae['cap_roll_configured']:.2f}/"
                      f"{ae['cap_pitch_configured']:.2f}, gains {ae['roll_nm']:.1f} / "
                      f"{ae['pitch_nm']:.1f} N*m [유도], firmware "
                      f"{ae['firmware_version'] or '?'}, wire "
                      f"{ae['mavlink_wire_version'] or '?'}, probe sha1 "
                      f"{str(ae['probe_sha1'] or '-')[:12]}, sign_probe sha1 "
                      f"{str(ae['sign_probe_sha1'] or '-')[:12]}). Level reference "
                      f"= ACTIVE LEVELLING; a tracked attitude needs "
                      f"policy.attitude_track. Interlocks: ceiling "
                      f"{float(att.get('abort_deg', 35.0)):.0f} deg, "
                      f"cap-pinned-without-progress "
                      f"{float(att.get('sat_ineffective_s', 1.0)):.1f} s, "
                      f"link degrade. RECORD BOUNDARY (meta "
                      f"run.attitude_axes, CSV rp_track).")
            self._log_event(f"ATTITUDE AXES ON (caps {ae['caps_in_force']} "
                            f"{ae['cap_roll']:.2f}/{ae['cap_pitch']:.2f}, "
                            f"sign_proven {ae['sign_proven']}, fw "
                            f"{ae['firmware_version']}, rc trim "
                            f"{ae['rc_trim_at_engage']})")
        if self._csv is None:
            # Into the run folder the screen recording and the nav recording
            # are already writing to (rov_gui/runstore.py): three writers, one
            # dated folder, no handshake between them. PINNED here for the life
            # of the engagement — see _run_dir().
            self._run_dir_pin = runstore.run_dir(self._run_tree())
            stem = self._run_dir_pin / f"mpc_{runstore.stamp('%H%M%S')}"
            self._open_csv(stem, auto=True)

    def _log(self, level: str, msg: str) -> None:
        """Every bus.log line of this worker goes through here so the tag says
        WHO is flying. A message that starts with the neutral ``ctrl: `` is
        re-tagged ``ctrl[<mode>]: `` with the follower actually selected
        (pid / mpc / dobmpc / *_tuned / mpcc) — the old fixed ``mpc:`` tag
        read as "the MPC is running" even under PID (2026-09-07 pool session,
        operator confusion). Messages with another tag (``station_bridge:``,
        ``imu_dr:``, a controller's own log) pass through unchanged."""
        if msg.startswith("ctrl: "):
            mode = str(getattr(self.cfg, "mode", "") or "") if self.cfg is not None else ""
            msg = (f"ctrl[{mode}]: " if mode else "ctrl: ") + msg[len("ctrl: "):]
        self.bus.log.emit(level, msg)

    def _apply_plan_cost_scale(self, s: float) -> None:
        """policy.q_scale onto the follower (HwDobMpc.set_plan_cost_scale); a
        follower without the hook (PID, MPCC, a test stub) is left alone."""
        ctrl = self.ctrl
        if ctrl is None or not hasattr(ctrl, "set_plan_cost_scale"):
            return
        before = float(getattr(ctrl, "plan_cost_scale", 1.0))
        ctrl.set_plan_cost_scale(float(s))
        if float(s) != before:
            self._log("info", f"ctrl: plan position-weight scale {before:g} -> {float(s):g}"
                              + (" (baseline restored)" if float(s) == 1.0 else ""))

    def _apply_plan_path_cost(self, along) -> None:
        """policy.along_scale onto a path-frame follower (HwDobMpc.
        set_plan_along_scale); a follower without the hook (PID, MPCC, a
        test stub) is left alone. Same lifecycle as q_scale: set at policy
        arm, None back at STOP / mode change / disengage."""
        ctrl = self.ctrl
        if ctrl is None or not hasattr(ctrl, "set_plan_along_scale"):
            return
        before = getattr(ctrl, "plan_along_scale", None)
        ctrl.set_plan_along_scale(None if along is None else float(along))
        after = getattr(ctrl, "plan_along_scale", None)
        if after != before:
            self._log("info", f"ctrl: plan along-track weight scale "
                              f"{'own' if before is None else f'{before:g}'} -> "
                              f"{'own (restored)' if after is None else f'{after:g}'}")

    def _extrapolate_tail(self, st, ts, t_end: float, p, v):
        """``policy.hold_tail: extrapolate`` — horizon stages past the newest
        plan's last knot continue at that plan's LAST-SEGMENT velocity
        (clipped to ``v_max_m_s``) for at most ``hold_tail_extrap_max_m``,
        then hold; positions stay inside ``workspace_box_ned``. The stitcher
        itself keeps holding the endpoint (``_replay_ref_now``, the divergence
        guard and the jaw sample are untouched) — only the reference handed
        to the follower changes. Returns ``(p, v, n_extrapolated_stages)``.

        Why: a 1 s plan arrives ~0.5 s old, so under ``track`` 85-90 % of the
        3 s NMPC horizon is "reach the endpoint and stop" (hold_frac p50
        0.87-0.89) and the velocity feedforward the drag override multiplies
        is zero there; the optimum is a ~3 N push inside the ESC deadband
        [측정: 0908_165517/mpc_165753.csv uX p10-p90 2.5-3.5 N, 0.01 m in
        45 s]. With the tail moving, the same solver on the same plans
        reaches 0.055 m/s (64 % of the plan) at along_scale 1.0 [측정 offline:
        data/20260908/0908_165517/diag/
        fwd_C_out.txt]. Last-segment velocity, not the plan mean: a chunk
        that decelerates toward its end (the approach's last 0.35 m) is
        continued gently, a chunk that stops is not extrapolated at all.
        """
        pc = self.cfg.policy
        p = np.array(p, dtype=float, copy=True)
        v = np.array(v, dtype=float, copy=True)
        past = np.asarray(ts, float) - float(t_end)
        m = past > 1e-9
        if not np.any(m):
            return p, v, 0
        v_end = np.asarray(st.end_velocity(), float)
        n = float(np.linalg.norm(v_end))
        if not (n > 1e-6):
            return p, v, 0
        vmax = float(pc["v_max_m_s"])
        if vmax > 0.0 and n > vmax:
            v_end = v_end * (vmax / n)
            n = vmax
        cap = float(pc["hold_tail_extrap_max_m"])
        t_cap = cap / n
        d = np.minimum(past[m], t_cap)
        p_end = p[:, m][:, 0].copy()            # the stitcher's endpoint hold
        p[:, m] = p_end[:, None] + v_end[:, None] * d[None, :]
        v[:, m] = v_end[:, None] * (past[m] <= t_cap + 1e-9)[None, :]
        box = pc.get("workspace_box_ned")
        if box:
            lo = np.asarray(box[0], float)[:, None]
            hi = np.asarray(box[1], float)[:, None]
            p[:, m] = np.clip(p[:, m], lo, hi)
        return p, v, int(np.sum(m))

    def _event(self, msg: str, defer: bool = False,
               pending: bool = False) -> None:
        """A short, timestamped mission line: to the on-screen log AND to
        events.log, so the two can never tell different stories.

        ``defer`` / ``pending`` hold the FILE half back until there is a run
        folder worth opening (see :meth:`_log_event`); the screen half is
        never deferred."""
        self.bus.mpc_event.emit(msg)
        self._log_event(msg, defer=defer, pending=pending)

    def _run_dir(self):
        """The folder this engagement's files belong in.

        PINNED while a CSV is open. Resolving it per event instead would put a
        refusal at 18:44 into `.../1844/events.log` while the CSV it explains
        sits in `.../1841/` — and, because run_dir creates as it goes, would
        leave a stray minute folder holding one line. Outside an engagement
        there is nothing to pin to, so a lone refusal opens (or joins) the
        folder for the moment it happened, which is the right answer for it.
        """
        return self._run_dir_pin or runstore.run_dir(self._run_tree())

    def _run_tree(self) -> "runstore.Tree":
        """The run TREE: the root plus this engagement's KIND. A land dry-run
        gets its own kind, and that is a safety property of the RECORD, not a
        tidiness preference.

        runstore.run_dir JOINS any folder OF THE SAME KIND whose newest file
        is under JOIN_WINDOW_S (90 s) old, so without the kind a bench run
        started shortly after a water run would land in ITS folder and append
        to the same plans.jsonl. And nothing in the meta distinguishes them:
        `source` and `policy.synthetic` read only `opts.source`, which is
        "hw" both times, so a land record would carry `synthetic: false`, the
        real tag_map_sha1 and a plausible datum. The 2026-09-02 pool runs are
        already cited as the [측정] provenance for `num_inference_steps: 8`
        and for the safety limit `v_max_m_s: 0.15`; plot_runs.py filters on
        controller.type and the leaf's kind, never on source. A separate kind
        is the only thing that makes the two impossible to pool by accident.

        Until 2026-09-14 the kind was a separate TREE (sessions/land_dryruns
        and sessions/policy_observe, siblings of the water tree — 2026-09-14
        moved both under the one dated root as the leaf suffix `_landdry` /
        `_observe`, rov_gui/runstore.py). The guarantee is unchanged:
        whatever the water leaf is, these two are never it.
        """
        base = str(self.cfg.log_dir)
        # A tool that owns its record (policy_dryrun: synthetic plant) names
        # the kind outright; it wins over the flags, because a dryrun with
        # `--mpc-mode none` is still a dryrun, not an observe run.
        forced = str(getattr(self.opts, "run_kind", "") or "")
        if forced:
            return runstore.Tree(base, forced)
        if self.land_dry_run:
            return runstore.Tree(base, "landdry")
        if self.observe:
            # POLICY OBSERVE (LOW level None; --policy-observe is the launch
            # alias), for word-for-word the reason above. RULE: LOW None
            # puts EVERY engagement here regardless of shape — a policy run,
            # a replay, or a bare HOLD with no mission — because the kind,
            # not the shape, is the pooling guard (plot_runs.py reads
            # controller.type and the kind only, and under None that is
            # "none"). The JOIN_WINDOW_S argument bites HARDER here than it
            # does for a land dry-run: an observe run is the natural thing
            # to start right after a closed-loop attempt ("that went badly,
            # let me just watch it") — i.e. inside the 90 s join window —
            # and `plans.jsonl` and `policy_plan.csv` are APPENDED to per
            # folder. Both runs' lines say `kind: "policy"`, so a pooled
            # folder is not separable afterwards even in principle. And the
            # pooling would be worse than a land run's: hw_mpc.yaml already
            # cites `plans.jsonl` from the water runs as the [측정]
            # provenance for `num_inference_steps: 8` and for the SAFETY
            # limit `v_max_m_s: 0.15`, and an observe run's
            # accept/clip/reject rates come from an anchor the pilot was
            # dragging around by hand. A separate kind is the only thing
            # that makes it impossible by accident.
            return runstore.Tree(base, "observe")
        return runstore.Tree(base, "")

    def _log_event(self, msg: str, defer: bool = False,
                   pending: bool = False) -> None:
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

        ``pending=True`` (2026-09-11 review) is for a MISSION line that
        happens outside any run — a LOW combo click between engagements —
        where opening a folder for it would be the same seven-folders
        failure. It keeps its own stamp, is written ONCE as the head of the
        next events.log that gets a real line, and is then dropped.
        """
        if self.cfg is None:
            return
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        if defer:
            self._session_banner.append(line)
            return
        if pending:
            self._pending_events.append(line)
            return
        try:
            p = self._run_dir() / "events.log"
            header = (self._session_banner
                      if p != self._events_path else ())
            with open(p, "a", encoding="utf-8") as f:
                for h in header:
                    f.write(h + "\n")
                for h in self._pending_events:
                    f.write(h + "\n")
                f.write(line + "\n")
            # Only after the write succeeded — a failed open must not silently
            # consume the banner (or the pending lines) for every folder that
            # follows.
            self._events_path = p
            self._pending_events.clear()
        except OSError:
            pass

    def _engage_refusal(self) -> str:
        e = self.cfg.engage if self.cfg else {}
        if not self._ready:
            return "worker not ready"
        if self.ctrl is None:
            return f"no controller ({self._setup_error})"
        # POLICY OBSERVE (LOW level None) skips every COMMAND-PATH gate below
        # — the real-time probe included (moved inside on 2026-09-11: it is
        # a command-path gate, the holder is never stepped) — and the three
        # VEHICLE gates further down, for the land_dry_run reason, arrived
        # at from the other side: those gates exist to stop a wrench
        # reaching a vehicle that is not ready for it, and here no wrench is
        # computed at all. Requiring COMMAND ENABLE, a healthy sink, an ARM
        # and MANUAL would be requiring the operator to prepare an actuation
        # path this mode structurally refuses to use — and would refuse the
        # very run whose point is to watch the network with the loop open.
        #
        # THE LOCALIZER GATES BELOW STILL APPLY, and must: the plan is
        # composed against the tag pose, so without one there is nothing
        # honest to draw.
        if not self.observe:
            if not self.ctrl.realtime_ok:
                # A cold or load-spiked probe at construction must not lock
                # the session out (2026-09-08: 79.7 ms right after a pid ->
                # mpc_tuned swap, then nine refusals). Measure again NOW,
                # warm, and refuse only if the solver is still slow.
                reprobe = getattr(self.ctrl, "reprobe", None)
                ok = bool(reprobe(self._log_probe)) if callable(reprobe) else False
                if not ok:
                    return (f"solver not real-time ({self.ctrl.solver_kind}, "
                            f"probe {self._fmt_ms(self.ctrl.probe_ms)} ms median"
                            f"{'' if reprobe is None else ' after re-probe'}; "
                            f"limit {float(self.cfg.engage.get('probe_ms_max', 25.0)):.0f} ms)")
                self._log("info", f"ctrl: solver re-probed OK "
                                  f"({self.ctrl.solver_kind}, probe "
                                  f"{self._fmt_ms(self.ctrl.probe_ms)} ms median, "
                                  f"cold {self._fmt_ms(getattr(self.ctrl, 'probe_ms_first', None))} ms)")
            if not self.cmd_enabled:
                return "COMMAND ENABLE is off"
            if self.sink_status_fn is not None:
                conn, note = self.sink_status_fn()
                if conn is Conn.FAULT:
                    # The sink itself says it cannot deliver (no peer on the
                    # command port, or SYSID_MYGCS mismatch): engaging would
                    # compute a mission nothing receives.
                    return f"command sink fault: {note}"
        # LAND DRY-RUN skips the three VEHICLE gates below. They exist to stop
        # a wrench reaching a vehicle that is not ready for it; in this mode no
        # sink is built, so there is no wrench to stop — and requiring an ARM
        # on a bench would be the actively dangerous reading, because ArduSub
        # spins thrusters by itself in any mode but MANUAL. The localizer gates
        # further down still apply: the synthetic fix satisfies them honestly.
        if not self.land_dry_run and not self.observe:
            tel = self.tel
            if tel is None or tel.armed is not True:
                return "vehicle not armed (or no telemetry)"
            # A stale snapshot still says armed=True forever; silence must not
            # count as consent (bus.py's Freshness rule, applied here too).
            if now() - float(tel.stamp) > 2.0:
                return "vehicle telemetry stale"
            allowed = self._allowed_flight_modes()
            mode = (tel.mode or "").upper()
            key = next((m for m in allowed if mode.startswith(m)), None)
            if allowed and key is None:
                return (f"flight mode is {tel.mode or '?'}, need "
                        f"{'|'.join(allowed)} — press MANUAL on the teleop "
                        f"panel, or list it in engage.require_mode")
            if not allowed and not mode.startswith("MANUAL"):
                # require_mode: [] — the operator removed the gate. Say what
                # that admits, once per attempt, where it is read.
                self._log("warn", f"ctrl: flight-mode gate is EMPTY and the "
                                  f"vehicle is in {tel.mode}: the autopilot "
                                  f"drives too (engage.require_mode: [])")
            # STABILIZE (or any self-driving mode) lets ArduSub's own
            # attitude / yaw-rate / depth loops drive the thrusters alongside
            # this loop. The EAOB would book those torques as disturbance
            # (w_hat K/M/N), so the observer family stays MANUAL-only —
            # whatever the list above says, including an empty one
            # (design review + safety audit 2026-09-08).
            if not mode.startswith("MANUAL") and \
                    str(self.cfg.mode).startswith("dobmpc"):
                return (f"{self.cfg.mode} needs MANUAL (in {tel.mode or '?'} "
                        f"the observer would absorb the autopilot's attitude "
                        f"and heading torques as disturbance) — pick "
                        f"pid/mpc/mpc_tuned or press MANUAL")
            # A programmatic stabilize block skips the YAML validator: refuse
            # anything but the two implemented yaw handlings here too.
            ya = str((e.get("stabilize") or {}).get("yaw_axis", "hold")).strip().lower()
            if not mode.startswith("MANUAL") and ya not in ("hold", "torque"):
                return (f"engage.stabilize.yaw_axis {ya!r} is not implemented "
                        f"(hold | torque)")
            settle = float(e.get("mode_settle_s", 0.0) or 0.0)
            if settle > 0.0 and self._mode_changed_at is not None:
                dt = now() - float(self._mode_changed_at)
                if dt < settle:
                    return (f"flight mode changed {dt:.1f} s ago — wait "
                            f"{settle:.1f} s for the autopilot to settle, "
                            f"then press START again")
            # THE 6-DoF VARIANT (2026-09-26, D7/D8/D10): with
            # engage.attitude_axes.enabled every gate must pass or the
            # engage is REFUSED — never a silent 4-DoF fallback, because the
            # follower would then be told (set_attitude_axes) that it can
            # tilt when it cannot. Off = none of this runs.
            if bool(self._attitude_cfg().get("enabled", False)):
                why = self._attitude_refusal(tel)
                if why:
                    return why
        elif not self.ctrl.realtime_ok:
            # Once per engage attempt, so a reader of the log knows the gate
            # was not forgotten, it was structurally moot.
            self._log("info", f"ctrl: solver real-time probe SKIPPED (LOW "
                              f"level None — the {self.ctrl.solver_kind} "
                              f"holder is never stepped)")
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
            self._apply_plan_cost_scale(1.0)   # policy.q_scale never outlives the mission
            self._apply_plan_path_cost(None)   # nor policy.along_scale
            self.station = None
            self.follow = None            # 1 of 4: see MpcWorker.follow
            self._clear_replay()
            if self._bridge is not None:
                self._bridge.reset()      # the ladder is a STATION feature
            self.phase, self.phase_detail = "", ""
            self.reason = "trajectory stopped (DP hold)"
            self._log("info", "ctrl: trajectory stopped — DP hold")
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
        # A fresh mission never inherits a policy cost scale; _arm_policy
        # re-applies its own right after this dispatch (safety review 2026-09-07).
        self._apply_plan_cost_scale(1.0)
        self._apply_plan_path_cost(None)
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
        if self.observe and shape not in ("policy", "replay"):
            # LOW level None (2026-09-11): the geometric shapes and follow
            # need a follower — approach, settle and the corner gate are all
            # statements about a vehicle being driven to the reference, and
            # nobody is driving it. Engage (HOLD) under None stays allowed:
            # datum, CSV and the PolicyState feed are what observe needs.
            self._refuse(f"LOW None (teleop) runs only Diffusion Policy / "
                         f"Replay — pick a low-level controller (PID/MPC/…) "
                         f"for {shape}")
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
        if shape == "policy":
            why = self._policy_refusal()
            if why:
                self._refuse(why)
                return
            # The replay rule again: the first plan is anchored at the
            # vehicle's CURRENT pose (leash of zero offset, v2 A4), so START
            # does not move it — the policy does, one 1 s chunk at a time.
            self._event("POLICY requested")
            self._arm_path(sq, shape,
                           (float(self._eta[0]), float(self._eta[1])),
                           float(self._eta[5]), float(self._eta[2]))
            return
        origin_xy = self._mission_origin(sq)
        if origin_xy is None:
            return                            # reason already logged
        # heading_tag is a STATION idea (a path's heading is its direction of
        # travel, or heading_follow). A value left in hw_mpc.yaml must not
        # steer a line/square/circle from a box the panel hides for them
        # (safety review 2026-09-11, HIGH).
        yaw_fixed = self._mission_yaw(
            sq if shape == "station" else {**sq, "heading_tag": None},
            origin_xy)
        if yaw_fixed is None:
            return                            # reason already logged
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
        self._log(
            "warn", f"ctrl: heading for {where} ({d0:.2f} m away), then "
                    f"holding {self._settle_s(shape):.0f} s before the {shape} "
                    f"starts")
        return

    def _refuse(self, why: str) -> None:
        """A refused START must reach the PANEL, not just the log line at the
        bottom of the window. Before 2026-08-14 the chip kept saying "engaged
        (DP hold)" and the operator had no idea the path had been rejected."""
        self.reason = f"START refused: {why}"
        self._log("warn", f"ctrl: START TRAJ refused — {why}")

    def _mission_yaw(self, sq: dict, origin_xy):
        """The heading to hold, as a DATUM-frame yaw — or None after a refusal.

        Three ways to say it, in priority order: ``heading_tag`` (a tag id —
        face THAT tag from the origin, i.e. the map bearing origin -> tag, so
        "sit on tag 47 looking at tag 97" is one line of the mission and the
        same physical heading every run; 2026-09-11, operator request),
        ``yaw_map_deg`` (an ABSOLUTE heading in the tag-map frame, 90 =
        facing +y), or ``yaw_fixed_deg`` ("current" = whatever the vehicle
        has). The first two are MAP-frame and get datumized here: a
        datum-relative angle means something different every time the
        vehicle engages on a different heading.

        ``origin_xy`` is what :meth:`_mission_origin` resolved (DATUM frame),
        so the bearing starts from the very point the mission holds — tag,
        explicit pair or current pose — with one rule, not a second copy of
        the origin_tag predicate that could disagree about tag 0."""
        yaw0 = float(self._datum["yaw0"]) if self._datum else 0.0
        ht = sq.get("heading_tag")
        if ht not in (None, "", 0):             # the panel's 0 = off
            look = self._tag_map_xy(ht, "heading tag")
            if look is None:
                return None                     # reason already logged
            frm = self._to_map_xy(float(origin_xy[0]), float(origin_xy[1]))
            dx, dy = look[0] - frm[0], look[1] - frm[1]
            # The origin tag itself, or a tag a few cm from it: a bearing
            # that is all map error, so no direction to face.
            if math.hypot(dx, dy) < 0.05:
                self._refuse(f"heading tag {ht} sits on the origin — "
                             f"no direction to face")
                return None
            return _wrap_pi(math.atan2(dy, dx) - yaw0)
        ym = sq.get("yaw_map_deg")
        if ym not in (None, "", "current"):
            return _wrap_pi(math.radians(float(ym)) - yaw0)
        yf = sq.get("yaw_fixed_deg", "current")
        return (float(self._eta[5]) if yf in ("current", None)
                else math.radians(float(yf)))

    def _tag_map_xy(self, tag, what: str):
        """MAP-frame (x, y) of one tag id, or None after a logged refusal:
        a tag the map does not know, or one printed at two places, cannot
        anchor anything."""
        tmap = self._tag_map()
        if tmap is None:
            self._refuse(f"{what} {tag}: tag map not loaded")
            return None
        try:
            tag = int(tag)
        except (TypeError, ValueError):       # a YAML typo, not a panel value
            self._refuse(f"{what} {tag!r} is not a tag id")
            return None
        poses = tmap.instances.get(tag)
        if not poses:
            self._refuse(f"{what} {tag} is not in the map")
            return None
        if len(poses) > 1:
            self._refuse(f"{what} {tag} is at {len(poses)} places — ambiguous")
            return None
        p = np.asarray(poses[0][1], float)
        return float(p[0]), float(p[1])

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
        data/20260817/0817_103431/mpc_103538.csv).
        An audit of EVERY dobmpc run on disk shows the X rail hit 66-95 % of
        the time going back to 2026-08-12, so this is long-standing and NOT a
        regression — it was simply never surfaced. Prime suspects, both
        unmeasured: axis_gain is [예측] (a wrong plant gain becomes phantom
        disturbance the moment note_applied reports a wrench the vehicle never
        produced) and the ~0.27 s tag latency inside an observer that assumes a
        fresh measurement.
        """
        if w_hat is None or self.cfg is None or not self.commanding:
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
            self._log(
                "error",
                f"ctrl: w_hat has been PINNED to its clip for 3 s ({axes}) — "
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
            self._log(
                "error", f"ctrl: gave up approaching {where} after "
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
            self._log(
                "warn", f"ctrl: arming follow on a LOOSE pair "
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
            self._log(
                "warn", f"ctrl: FOLLOW armed at {d * 100:.0f} cm but the "
                        f"excursion clamp is "
                        f"{float(a.cfg['max_excursion_m']) * 100:.0f} cm — the "
                        f"target will clamp almost immediately. Get closer to "
                        f"the object, or raise object_nav.max_excursion_m.")
        self._event(f"FOLLOW armed ({d * 100:.0f} cm)")
        self._log_event(what)
        self._log("warn", f"ctrl: {what}")

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
                self._log(
                    "warn", f"ctrl: FOLLOW feedforward clipped "
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
            self._log(
                "warn", f"ctrl: the object has taken the target "
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
        too, this vehicle drifts with its trim of the day (model -5.7 N sinking,
        2026-09-07 it floated), and a drifting vehicle's view changes — which makes re-acquiring the object LESS
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
            "heading_tag": None,          # same schema as an armed station
            # So a CSV/meta reader can tell this station apart from one the
            # operator asked for.
            "from_follow": str(why)}
        self.phase, self.phase_detail = "station", f"follow ended: {why}"
        self.reason = f"object follow ended ({why}) — station hold"
        self._event(f"FOLLOW -> STATION: {why}")
        self._log_event(f"FOLLOW ENDED - {why}; holding the last setpoint"
                        + (f" (err {held * 100:.0f} cm)" if held else ""))
        self._log(
            "warn", f"ctrl: object follow ended ({why}) — holding the last "
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
    def _clear_replay(self, end_reason: str = "stop") -> None:
        """Drop the streamed-plan mission state (replay OR policy) — and
        NEUTRAL the jaw.

        ``grip_drive`` is a latched LEVEL in the command sink (hardware.py
        ``set_gripper_drive``): whoever set it non-zero must set it back, or
        the jaw keeps driving after the mission that asked for it is gone.
        Every mission-clearing site calls this, so the 4-site rule for
        ``follow`` covers the jaw too — and the width ESTIMATOR is released
        with it (v2 A12), so the width the next policy is told stops moving
        the moment the drive does. ``end_reason`` is what a policy run's
        record says when nothing more specific was written first (the intake
        writes max_run/diverged/escalated/bridge/worker_silent itself; STOP
        TRAJ arrives here as "stop", disengage as "disengaged").
        """
        rp = self.replay
        if rp is not None:
            if rp.get("kind") == "policy" and not rp.get("end_reason"):
                # Prefixed for the `_policy_end` reason: "disengaged" reads
                # the same in both kinds of run, and this is the path a
                # STOP / DISENGAGE / release takes.
                rp["end_reason"] = (f"observe_{end_reason}" if self.observe
                                    else str(end_reason))
            self._replay_last = dict(rp)
            if rp.get("gripper_on"):
                # Emit the neutral only when THIS mission holds a drive: a
                # 0.0 over a jaw the mission never drove is a spurious
                # command (and a spurious estimator "drive"); the estimator
                # is released either way (verify 2026-09-02).
                # `not self.observe` is belt and braces — an observe mission
                # is armed with gripper_on False and never leaves grip_drive
                # 0.0, so this is already unreachable. It is guarded because
                # it is an ACTUATION emit, and the rule for those is that the
                # mode is checked at the emit, not inferred from upstream.
                if (float(rp.get("grip_drive", 0.0) or 0.0) != 0.0
                        and not self.observe):
                    self.bus.cmd_gripper_drive.emit(0.0)
                if self._grip_est is not None:
                    self._grip_est.release(now())
        self.replay = None
        self._plan_filter = None
        self._plan_stitcher = None
        self._policy_inbox = None

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
        # replay.track_attitude (2026-09-26): a recorded demo's roll/pitch
        # ride the plan ONLY with the attitude axes live and a loader that
        # can carry them; the default replay path stays level (rp None).
        load_kw = {}
        track_att = bool(r.get("track_attitude", False)) and self._attitude_axes
        if track_att:
            try:
                if "with_rp" in inspect.signature(load_replay_track).parameters:
                    load_kw = {"with_rp": True}
            except (TypeError, ValueError):
                load_kw = {}
            if not load_kw:
                self._log("warn", "ctrl: replay.track_attitude set but this "
                                  "build's load_replay_track has no with_rp — "
                                  "flying the replay LEVEL")
                track_att = False
        try:
            track = load_replay_track(str(r["session"]), **load_kw)
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
            self._log(
                "warn", "ctrl: REPLAY workspace box is OFF "
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
            # The mode decides, not the YAML — the `_arm_policy` rule. A
            # replay IS armable inside an observe session (the flag is
            # session-wide and no refusal restricts the shape), and without
            # this it would log jaw CLOSE/OPEN events and a non-zero
            # `grip_cmd` for a jaw that structurally cannot have moved.
            "gripper_on": bool(r["gripper"]) and not self.observe,
            "grip_state": grip0,
            "grip_since": 0.0,
            "grip_drive": 0.0,
            "grip_events": 0,
            "grip_lookahead_s": float(r["gripper_lookahead_s"]),
            "grip_g": float("nan"),      # (schema 13) set per tick
            "attitude_track": bool(track_att),
        }
        scen = {
            "kind": "replay",
            "session": str(r["session"]),
            "poses_sha1": poses_sha,
            "attitude_track": bool(track_att),
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
            "gripper_lookahead_s": float(r["gripper_lookahead_s"]),
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
                f", gripper {'ON' if scen['gripper'] else 'off'}"
                + (f", lookahead {float(r['gripper_lookahead_s']):g} s"
                   if scen['gripper'] else ""))
        self._event("REPLAY started")
        self._log_event(f"REPLAY started - {what}")
        self._log("warn", f"ctrl: {what}")

    def _replay_ref_now(self, t_rel: float):
        """The pose the anchor gate compares a new plan against: the ACTIVE
        reference if one exists, else the vehicle (first plan = anchored at
        the hull, which is where `_arm_replay` built it)."""
        st = self._plan_stitcher
        if st is not None and st.has_plan():
            p, yaw, _v, _r, _g = st.sample(np.array([float(t_rel)]))
            return np.asarray(p[:, 0], float), float(yaw[0])
        return np.asarray(self._eta[:3], float), float(self._eta[5])

    def _replay_ref_att_now(self, t_rel: float):
        """The attitude the ACTIVE reference holds at ``t_rel``: (roll,
        pitch) NED rad as a (2,) array from the stitcher's ``sample_att``, or
        None when no installed plan ever carried roll/pitch (the consumer
        levels) or the stitcher predates the accessor. For the attitude
        leash and the div_rp interlock — never the vehicle: a 4-DoF stream
        has no attitude reference to compare against."""
        st = self._plan_stitcher
        fn = getattr(st, "sample_att", None) if st is not None else None
        if st is None or not callable(fn) or not st.has_plan():
            return None
        try:
            att = fn(np.array([float(t_rel)]))
        except (TypeError, ValueError):
            return None
        if att is None:
            return None
        rp = np.asarray(att[0], float)
        if rp.ndim != 2 or rp.shape[0] != 2 or rp.shape[1] < 1:
            return None
        return np.array([float(rp[0, 0]), float(rp[1, 0])])

    def _stream_att_tracked(self) -> bool:
        """Is the live stream mission composing/tracking roll-pitch?"""
        rp = self.replay
        return bool(rp is not None and rp.get("attitude_track"))

    def _stream_cfg(self) -> dict:
        """The config block the ACTIVE stream mission reads its gripper
        thresholds / hold_max from: ``cfg.policy`` for kind policy, else
        ``cfg.replay`` (v2 A21)."""
        rp = self.replay
        if rp is not None and rp.get("kind") == "policy":
            return self.cfg.policy
        return self.cfg.replay

    def _div_limit(self) -> float:
        """The divergence guard's distance [m]: the policy's OWN
        ``div_max_m`` (v2 A21); a replay keeps 2 x anchor_max_m."""
        rp = self.replay
        if rp is not None and rp.get("kind") == "policy":
            return float(self.cfg.policy["div_max_m"])
        return 2.0 * float(self.cfg.replay["anchor_max_m"])

    def _tick_replay(self, t_rel: float) -> None:
        """One streamed-plan tick (kind replay OR policy): release due plans
        through the filter, then install the stitched horizon. Runs between
        `_advance_path_clock` and ``ctrl.step``, so the plan the solver sees
        is this tick's. For kind policy the intake runs FIRST (v2 A23: inside
        this guarded tick, never in the slot) and may end the mission."""
        rp = self.replay
        st = self._plan_stitcher
        if rp is None or st is None or self.ctrl is None or t_rel is None:
            return
        # (schema 13) nan until the jaw logic at the bottom of this tick
        # samples the channel, so a tick that returns early (no plan yet,
        # diverged, halted) records that nothing was seen.
        rp["grip_g"] = float("nan")
        policy = rp.get("kind") == "policy"
        if policy:
            self._tick_policy_intake(float(t_rel))
            if self.replay is None:          # the intake ended the mission
                return
        while rp["pending"] and rp["pending"][0].t0 <= float(t_rel) + 1e-9:
            if policy and rp.get("halted"):
                # The latch (A7) is re-read EVERY iteration: the 3rd reject
                # sets it mid-loop, and a plan still due behind it must not
                # be evaluated, let alone installed (verify 2026-09-02).
                rp["pending"].clear()
                rp.get("extras", {}).clear()
                break
            msg = rp["pending"].pop(0)
            extra = rp.get("extras", {}).pop(int(msg.plan_id), None)
            rp["released"] += 1
            r_now = self._replay_ref_now(t_rel)
            ev_kw = {}
            if (getattr(msg, "rp", None) is not None
                    and callable(getattr(st, "sample_att", None))):
                # The attitude overlap gate needs the CURRENT reference's
                # roll/pitch; passed only for an rp-carrying plan, so a
                # 4-DoF stream's evaluate call is the pre-variant one.
                ev_kw["cur_sample_att"] = (st.sample_att if st.has_plan()
                                           else None)
            verdict = self._plan_filter.evaluate(
                msg, r_now=r_now,
                cur_sample=(st.sample if st.has_plan() else None),
                now=float(t_rel), **ev_kw)
            if (policy and verdict.status in ("accept", "clip")
                    and verdict.plan is not None):
                verdict = self._policy_blend_gate(verdict, float(t_rel),
                                                  extra)
            if policy and getattr(msg, "rp", None) is not None:
                # The attitude gate counters (meta policy.run): a HARD
                # reject on rp magnitude, and a clip whose dilation was set
                # by the attitude rate.
                mg = dict(verdict.margins or {})
                if verdict.status == "reject" and mg.get("rp_mag") is not None \
                        and float(mg["rp_mag"]) < 0.0:
                    rp["reject_rp_mag"] = int(rp.get("reject_rp_mag", 0)) + 1
                if verdict.status == "clip" and any(
                        ("rp_rate" in str(r) or "attitude rate" in str(r))
                        for r in verdict.reasons):
                    rp["clip_rp_rate"] = int(rp.get("clip_rp_rate", 0)) + 1
            self._log_plan(msg, verdict, t_rel, r_now, extra)
            if verdict.status in ("accept", "clip") and verdict.plan is not None:
                st.install(verdict.plan, float(t_rel))
                rp["installed"] += 1
                rp["t_last_accept"] = float(t_rel)
                if verdict.status == "clip":
                    rp["clipped"] += 1
                continue
            rp["rejected"] += 1
            if policy:
                # ESCALATION LATCH (v2 A7): three geometric rejects in a row
                # and the stream is no longer trusted. The reference holds
                # the last installed plan's endpoint (v = 0, a DP hold that
                # already passed the anchor gate), the intake drops
                # everything from here on, and the GPU idles
                # (PolicyState.active False). Only STOP TRAJ / START clears
                # it — a later GOOD plan must not un-latch what three bad
                # ones latched.
                if verdict.escalate and self.observe:
                    # NOT UNDER OBSERVE. The latch exists because a stream
                    # that keeps failing geometric gates must stop reaching a
                    # vehicle — but nothing reaches the vehicle here, and
                    # halting would stop the GPU inferring
                    # (PolicyState.active goes False), i.e. it would end the
                    # very measurement the run exists to take. A run of
                    # rejects IS the finding in observe, not a fault. Counted
                    # and said out loud once per escalation, never latched.
                    rp["escalations"] = int(rp.get("escalations", 0)) + 1
                    if now() - float(rp.get("t_escalate_warn", -1e9)) > 5.0:
                        rp["t_escalate_warn"] = now()
                        self._log(
                            "warn",
                            f"ctrl: POLICY OBSERVE — {verdict.consec_rejects} "
                            f"consecutive plan rejections "
                            f"({'; '.join(verdict.reasons) or 'rejected'}). "
                            f"NOT halted (observe): the network keeps "
                            f"inferring and every proposal keeps being drawn. "
                            f"This is the finding, not a fault.")
                    continue
                if verdict.escalate and not rp.get("halted"):
                    self._policy_halt(
                        "escalated",
                        f"POLICY plans rejected {verdict.consec_rejects}x — "
                        f"halted",
                        f"ctrl: POLICY escalated after "
                        f"{verdict.consec_rejects} consecutive plan rejections "
                        f"({'; '.join(verdict.reasons) or 'rejected'}) — "
                        f"reference is holding the last plan's endpoint, no "
                        f"more plans will be installed. STOP TRAJ to hold "
                        f"HERE instead; START re-arms.")
                    rp["pending"].clear()
                    rp.get("extras", {}).clear()
                    break
                continue
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
                    self._log(
                        "error", f"ctrl: REPLAY plan rejected ({why}) — "
                                 f"nothing to fly; holding HERE. Margins "
                                 f"are in plans.jsonl.")
                    self.set_traj(False)
                    self.reason = "replay plan rejected (DP hold)"
                    return
                # The stream went bad mid-mission. Stop feeding it; the
                # stitcher holds its endpoint (v=0), which is a DP hold,
                # and the operator decides what happens next.
                self._event("REPLAY plans rejected repeatedly — holding")
                self._log(
                    "error", "ctrl: REPLAY escalated after consecutive "
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
        # The policy has its OWN limit (policy.div_max_m, v2 A21).
        div = float(np.linalg.norm(
            self._replay_ref_now(t_rel)[0]
            - np.asarray(self._eta[:3], float)))
        lim_div = self._div_limit()
        # LAND DRY-RUN: no guard. It compares the reference against the
        # VEHICLE, and on a bench the vehicle is a synthetic pose that by
        # construction never moves — so the moment a plan installs and the
        # reference starts walking, the "error" grows without bound and the
        # mission would stop within seconds. There is nothing to protect: no
        # sink, no motion. The divergence is still COMPUTED and recorded, it
        # simply does not end the run.
        # ...and POLICY OBSERVE, for a stronger version of the same reason:
        # the pilot is deliberately flying AWAY from the reference — that is
        # the experiment — and 0.25 m is one or two seconds of manual flight.
        # Nothing is being commanded, so there is no lunge-on-release to
        # protect against. RECORDED rather than dropped: `div_max_seen_m`
        # goes into the run block, so a reader can still answer "how far
        # apart did they get" instead of taking the guard's silence for
        # agreement.
        rp["div_max_seen_m"] = max(float(rp.get("div_max_seen_m", 0.0)),
                                   float(div))
        if div > lim_div and not self.land_dry_run and not self.observe:
            rp["div_streak"] = int(rp.get("div_streak", 0)) + 1
            if rp["div_streak"] >= max(2, int(round(0.5 * self.cfg.ctrl_hz))):
                what = "POLICY" if policy else "REPLAY"
                if policy:
                    rp["halted"] = "diverged"
                    rp["end_reason"] = "diverged"
                self._event(f"{what} diverged ({div:.2f} m) — stopped")
                self._log(
                    "error", f"ctrl: {what} reference is {div:.2f} m from the "
                             f"vehicle (> {lim_div:.2f} m for 0.5 s) — the "
                             f"vehicle cannot stay with the "
                             f"{'policy' if policy else 'demo'}. Stopping "
                             f"and holding HERE.")
                self.set_traj(False)
                self.reason = (f"{'policy' if policy else 'replay'} diverged "
                               f"(DP hold)")
                return
        else:
            rp["div_streak"] = 0
        if policy and rp.get("attitude_track"):
            # Interlock (i), D9: the ATTITUDE divergence guard. The tracked
            # roll/pitch reference more than div_max_rp_deg from the
            # measured attitude for 0.5 s -> the policy is DROPPED to a
            # level DP hold (set_traj(False) -> set_target_ned, and the
            # follower ramps its attitude reference to level at pq_max, D5)
            # while K/M KEEP FLOWING: losing the levelling torque on a
            # tilted hull is worse than holding level. Recorded either way
            # (div_rp_max_seen_deg); disarmed with the position guard.
            ref_rp = self._replay_ref_att_now(t_rel)
            if ref_rp is not None:
                eta = np.asarray(self._eta, float)
                d_rp = max(abs(_wrap_pi(float(ref_rp[0]) - float(eta[3]))),
                           abs(_wrap_pi(float(ref_rp[1]) - float(eta[4]))))
                rp["div_rp_max_seen_deg"] = max(
                    float(rp.get("div_rp_max_seen_deg", 0.0)),
                    math.degrees(d_rp))
                lim_rp = math.radians(float(self._policy_att("div_max_rp_deg")))
                if (d_rp > lim_rp and not self.land_dry_run
                        and not self.observe):
                    rp["div_rp_streak"] = int(rp.get("div_rp_streak", 0)) + 1
                    if rp["div_rp_streak"] >= max(2, int(round(0.5 * self.cfg.ctrl_hz))):
                        rp["halted"] = "div_rp"
                        self._policy_end(
                            "div_rp",
                            f"POLICY attitude diverged "
                            f"({math.degrees(d_rp):.1f} deg) — level hold",
                            "error",
                            f"ctrl: POLICY attitude reference is "
                            f"{math.degrees(d_rp):.1f} deg from the measured "
                            f"roll/pitch (> {math.degrees(lim_rp):.0f} deg for "
                            f"0.5 s) — dropping the policy to a LEVEL DP hold; "
                            f"K/M keep flowing (active levelling).",
                            "policy attitude diverged (level DP hold)")
                        return
                else:
                    rp["div_rp_streak"] = 0
        K = int(getattr(self.ctrl, "path_plan_steps", 1))
        dt = float(getattr(self.ctrl, "path_plan_dt", 1.0 / self.cfg.ctrl_hz))
        ts = float(t_rel) + np.arange(K) * dt
        p, yaw, v, rr, g = st.sample(ts)
        if policy and self.cfg.policy["hold_tail"] == "extrapolate":
            # HOLD TAIL = EXTRAPOLATE (2026-09-08). Gated exactly like the
            # mask: only while the newest plan is LIVE and the mission is
            # not latched — an expired stream must hold its endpoint.
            t_end_x = float(st.end_time())
            if float(t_rel) < t_end_x - 1e-9 and not rp.get("halted"):
                p, v, n_ext = self._extrapolate_tail(st, ts, t_end_x, p, v)
                if n_ext:
                    rp["extrap_ticks"] = int(rp.get("extrap_ticks", 0)) + 1
        speed = np.hypot(np.asarray(v[0], float), np.asarray(v[1], float))
        psi_path = np.where(speed > 1e-3,
                            np.arctan2(np.asarray(v[1], float),
                                       np.asarray(v[0], float)),
                            np.asarray(yaw, float))
        from .path_geometry import NedPlan

        w_stage = None
        t_end = float(st.end_time())
        if policy:
            # HOLD TAIL = COST MASK (v2 A10). Stages past the active plan's
            # last knot weigh nothing on position/linear velocity, so the
            # NMPC does not brake for an endpoint the next plan will move.
            # ONLY WHILE A PLAN IS LIVE (t_rel < t_end) AND THE MISSION IS
            # NOT LATCHED: once the newest plan has expired (no plan in
            # 0.8 s: the stream stalled, the worker is late, or the
            # escalation latch dropped everything) the stitcher's endpoint
            # hold is the reference the vehicle must actually keep, so it
            # gets FULL weight — a mask there zeroed x/y/z AND heave on the
            # whole horizon and the "reference is holding" claim was false
            # (verify 2026-09-02, BLOCKER). `hold_frac` is still written
            # as a record (1.0 = the whole horizon is past the plan).
            rp["hold_frac"] = float(np.mean(ts > t_end + 1e-9))
            pc = self.cfg.policy
            live = (float(t_rel) < t_end - 1e-9) and not rp.get("halted")
            if pc["hold_tail"] == "mask" and live:
                past = ts - t_end
                taper = float(pc["hold_tail_taper_s"])
                if taper > 0.0:
                    w_stage = np.where(
                        past <= 0.0, 1.0,
                        np.where(past >= taper, 0.0,
                                 0.5 * (1.0 + np.cos(math.pi * past / taper))))
                else:
                    w_stage = (past <= 1e-9).astype(float)
        # ATTITUDE (2026-09-26): the stitcher's roll/pitch sample rides the
        # SAME horizon grid into NedPlan.rp_ned / rp_rate_ned. None (no
        # installed plan ever carried rp, or a stitcher without the
        # accessor) = the pre-variant NedPlan, and the follower levels. The
        # hold tail is NOT extrapolated for attitude: endpoint hold.
        plan_kw = {}
        sample_att = getattr(st, "sample_att", None)
        if callable(sample_att):
            att = sample_att(ts)
            if att is not None:
                plan_kw["rp_ned"] = np.asarray(att[0], float)
                plan_kw["rp_rate_ned"] = (None if att[1] is None
                                          else np.asarray(att[1], float))
        self.ctrl.set_path_plan_ned(NedPlan(
            p_ned=np.asarray(p, float), yaw_ned=np.asarray(yaw, float),
            v_ned=np.asarray(v, float), r_ned=np.asarray(rr, float),
            psi_path=np.asarray(psi_path, float), w_stage=w_stage,
            **plan_kw))
        hold = st.source_at(float(t_rel)) == "hold"
        if policy:
            self.phase_detail = (
                f"{t_rel:.0f}/{rp['duration_s']:.0f} s, "
                f"{rp['installed']} plans"
                + (" [hold]" if hold else "")
                + (f" [halted: {rp['halted']}]" if rp.get("halted") else "")
                + (" [policy stale]" if rp.get("stale_flag") else ""))
        else:
            self.phase_detail = (f"{t_rel:.0f}/{rp['duration_s']:.0f} s"
                                 + (" [hold]" if hold else ""))
        # JAW SAMPLE INSTANT (2026-09-07, schema 13). The hysteresis reads
        # the width channel at t_rel + gripper_lookahead_s, not at t_rel:
        # the 0907_145206 run showed the policy placing its "close" at the
        # far end of every ~0.45 s-lived chunk while knot 0 stayed open, so
        # the sample at now crossed close_below only in plan gaps (see
        # geometry.default_policy_block). `st.sample` clamps to the covered
        # span (endpoint hold, no extrapolation). la == 0.0 takes g[0] of
        # the horizon sample above VERBATIM — the pre-13 value, bit for bit.
        la = float(self._stream_cfg()["gripper_lookahead_s"])
        if g is None:
            g_jaw = None
        elif la > 0.0:
            g_jaw = float(st.sample(np.array([float(t_rel) + la]))[4][0])
        else:
            g_jaw = float(g[0])
        rp["grip_g"] = (float(g_jaw)
                        if (g_jaw is not None and rp.get("gripper_on"))
                        else float("nan"))
        self._tick_replay_gripper(g_jaw, float(t_rel))

    def _tick_replay_gripper(self, g, t_rel: float) -> None:
        """Replay the stream's jaw-width channel onto the momentary drive.

        Width (0 = closed) crosses ``gripper_close_below`` -> drive CLOSE;
        crosses ``gripper_open_above`` -> drive OPEN; in between, hold the
        current state (hysteresis — a borderline width must not toggle the
        drive at 20 Hz). The drive auto-neutrals after ``gripper_hold_max_s``
        (anti-stall; the jaw keeps its position when neutral) and every emit
        is EDGE-based, so a pilot G/H press between our edges still wins.
        OFF unless the active stream's block (``replay`` / ``policy``,
        :meth:`_stream_cfg`) has gripper true — the jaw path is open-loop (no
        position feedback exists) and early runs prove the trajectory before
        the jaw moves at all. ``g`` is the caller's sample at
        ``t_rel + gripper_lookahead_s`` (schema 13; 0 = at now), and is
        what the CSV records as ``grip_g``.
        """
        rp = self.replay
        if rp is None or not rp.get("gripper_on") or g is None:
            return
        r = self._stream_cfg()
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
                self._event(f"{rp.get('kind', 'replay').upper()} gripper "
                            f"{want.upper()}")
        elif (rp["grip_drive"] != 0.0
              and (float(t_rel) - float(rp["grip_since"]))
              > float(r["gripper_hold_max_s"])):
            self._set_replay_grip_drive(0.0)   # anti-stall auto-neutral

    def _set_replay_grip_drive(self, v: float) -> None:
        rp = self.replay
        if rp is None:
            return
        if float(v) != float(rp.get("grip_drive", 0.0)):
            if self.observe:
                # THE SECOND ACTUATOR, refused BEFORE the latch. The jaw does
                # not go through cmd_pilot, so muting the wrench does not
                # mute it — and the sink LATCHES a drive level, which under
                # observe nothing would ever clear (the deadman that would is
                # fed continuously by the pilot's own frames). Returning
                # before `rp["grip_drive"]` is written matters as much as
                # returning before the emit: that field is what the CSV's
                # `grip_cmd` column and the run meta report as the drive
                # being HELD, and recording a held drive that was never sent
                # is the same class of claim as a 0.000 N wrench on a muted
                # loop. Unreachable while both arm paths force `gripper_on`
                # False, and guarded here anyway: a guard at the emitter
                # cannot be routed around by a future caller.
                return
            rp["grip_drive"] = float(v)
            self.bus.cmd_gripper_drive.emit(float(v))
            # The emitter is this worker, so the estimator hears it DIRECTLY
            # (v2 A12) — the bus copy the backend routes back to
            # on_gripper_drive re-latches the same level at the same time,
            # which the integrator treats as a no-op.
            self.on_gripper_drive(float(v))

    #: `extra` keys _log_plan has already placed elsewhere in the record —
    #: never duplicated top-level: the anchor attitude is nested under
    #: `anchor` (rp_meas / rp_used / rp_ref) and `rp` (the resampled tracked
    #: knots) is `raw.rp` — the same (2,J) array the message carries.
    _PLAN_EXTRA_NESTED = frozenset({"anchor_rp_meas", "anchor_rp_used",
                                    "anchor_rp_ref", "rp"})

    def _log_plan(self, msg, verdict, t_rel, r_now, extra=None) -> None:
        """One JSON line per plan offered to the filter -> plans.jsonl in the
        run folder. This is the planner-vs-tracker attribution record: the raw
        plan (pre-filter), the anchor snapshot, and every margin NUMBER (a
        boolean verdict cannot be threshold-tuned afterwards). ``msg`` may be
        None for a policy plan that never reached the filter (late / skipped
        / rejected at composition — the line then carries the reason and
        whatever ``extra`` knows about it, v2 A7/A9/A23)."""
        import hashlib

        rp = self.replay
        try:
            rec = {
                "kind": (rp.get("kind") if rp is not None else None),
                # WHO WAS GOING TO FLY THIS. `status: "accept"` means "the
                # filter passed it and it was installed" — which in a control
                # run also means the NMPC then tracked it, and in an OBSERVE
                # run means only that it was drawn. Same word, two claims, in
                # files that are appended to per run folder; this field is
                # what keeps them apart line by line even if the two ever
                # share a folder.
                "follower": ("observe" if self.observe else "mpc"),
                "t_rel": round(float(t_rel), 4),
                "wall": time.strftime("%Y-%m-%d %H:%M:%S"),
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
                # ALWAYS (schema 16): was this plan's roll/pitch TRACKED (a
                # (2,K) rp on the message) rather than levelled? False on
                # every 4-DoF line, and on a 7-dim plan flown in the
                # dropped-and-logged mode (dropped_rp_deg then says what was
                # levelled away).
                "rp_tracked": bool(msg is not None
                                   and getattr(msg, "rp", None) is not None),
            }
            if extra:
                # The attitude anchor lives UNDER `anchor`, beside eta / yaw,
                # and ONLY there (record audit 2026-09-26, Q6: rp_ref used to
                # be dropped top-level as `anchor_rp_ref` and rp_meas /
                # rp_used duplicated top-level by the setdefault loop below).
                if extra.get("anchor_rp_meas") is not None:
                    rec["anchor"]["rp_meas"] = [float(v) for v in extra["anchor_rp_meas"]]
                if extra.get("anchor_rp_used") is not None:
                    rec["anchor"]["rp_used"] = [float(v) for v in extra["anchor_rp_used"]]
                if "anchor_rp_ref" in extra:
                    v = extra["anchor_rp_ref"]
                    rec["anchor"]["rp_ref"] = (None if v is None else
                                               [float(x) for x in v])
            if msg is not None:
                p = np.asarray(msg.p_ned, float)
                small = p.shape[1] <= 64
                rec.update({
                    "plan_id": int(msg.plan_id),
                    "t0": float(msg.t0), "dt": float(msg.dt),
                    "obs_t": (None if msg.obs_t is None
                              else float(msg.obs_t)),
                    "n_knots": int(p.shape[1]),
                    # The RAW plan (pre-PlanFilter; since 2026-09-14 its yaw
                    # is post yaw_ref_filter when that is on — the policy's
                    # own heading is action_raw + anchor). Full knots when small (a
                    # streamed window); a one-shot whole-demo plan is
                    # summarized by its ends + a hash so the line stays one
                    # line.
                    "raw": ({"p_ned": p.tolist(),
                             "yaw": np.asarray(msg.yaw, float).tolist(),
                             "g": (None if msg.g is None else
                                   np.asarray(msg.g, float).tolist())}
                            if small else
                            {"p_first": [float(x) for x in p[:, 0]],
                             "p_last": [float(x) for x in p[:, -1]],
                             "sha1": hashlib.sha1(
                                 p.tobytes()).hexdigest()[:12]}),
                })
                rp_msg = getattr(msg, "rp", None)
                if rp_msg is not None and small:
                    # the tracked (2,K) roll/pitch, rad, beside the geometry
                    rec["raw"]["rp"] = np.asarray(rp_msg, float).tolist()
            if extra:
                for k, v in dict(extra).items():
                    if k in self._PLAN_EXTRA_NESTED:
                        continue
                    rec.setdefault(k, v)
            with open(self._run_dir() / "plans.jsonl", "a",
                      encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False,
                                   default=_json_default) + "\n")
        except (OSError, TypeError, ValueError) as e:      # noqa: BLE001
            self._log("warn", f"ctrl: plans.jsonl not written ({e})")
        # DRAW IT, and write the flat per-knot CSV beside the JSONL (2026-09-02,
        # operator request). Both cover EVERY plan including rejected ones —
        # jsonl already had the geometry under `raw`, but one JSON line per plan
        # is not something you can look at, and on these runs nearly every plan
        # is rejected, so "what did the network actually ask for" had no answer
        # on screen at all. Outside the try above: a plot must not be lost
        # because the log file could not be opened, nor the reverse.
        if msg is not None and (rp is not None
                                and rp.get("kind") == "policy"):
            self._emit_policy_plan_viz(msg, verdict, t_rel)

    def _emit_policy_plan_viz(self, msg, verdict, t_rel: float) -> None:
        """Publish one composed plan to the panel and append it to
        ``policy_plan.csv``. Never raises into the control tick."""
        try:
            p = np.asarray(msg.p_ned, float)
            if p.ndim != 2 or p.shape[0] != 3 or p.shape[1] == 0:
                return
            yaw = np.asarray(msg.yaw, float).reshape(-1)
            status = str(verdict.status)
            reason = str(verdict.reasons[0]) if verdict.reasons else ""
            rp_m = getattr(msg, "rp", None)
            rp_v = None
            if rp_m is not None:
                rp_a = np.asarray(rp_m, float)
                if rp_a.ndim == 2 and rp_a.shape[0] == 2 \
                        and rp_a.shape[1] == p.shape[1]:
                    rp_v = (tuple(float(v) for v in rp_a[0]),
                            tuple(float(v) for v in rp_a[1]))
            self.bus.policy_plan_viz.emit(PolicyPlanViz(
                plan_id=int(msg.plan_id), status=status,
                p_ned=tuple(tuple(float(v) for v in row) for row in p),
                yaw=tuple(float(v) for v in yaw),
                t0=float(msg.t0), dt=float(msg.dt), reason=reason,
                rp=rp_v))
        except (ValueError, TypeError, AttributeError):        # noqa: BLE001
            return
        try:
            path = self._run_dir() / "policy_plan.csv"
            new = not path.exists()
            with open(path, "a", encoding="utf-8") as f:
                if new:
                    # One row per KNOT, not per plan: this is meant to be
                    # plotted, and a plan-per-row file would need the reader to
                    # unpack a list column before it could be.
                    # `follower` is the same boundary plans.jsonl carries:
                    # in an OBSERVE run `status: accept` means the plan was
                    # DRAWN, not flown, and this file is appended to per run
                    # folder.
                    # ALWAYS the schema-16 header (append-at-end rule):
                    # roll_deg / pitch_deg AFTER reason, nan on a plan
                    # without roll/pitch (every 4-DoF plan).
                    f.write("plan_id,status,follower,t_rel,k,t_knot,"
                            "x_ned,y_ned,z_ned,yaw_deg,reason,"
                            "roll_deg,pitch_deg\n")
                safe = reason.replace('"', "'")
                follower = "observe" if self.observe else "mpc"
                for k in range(p.shape[1]):
                    rp_cols = ("nan,nan" if rp_v is None else
                               f"{math.degrees(rp_v[0][k]):.3f},"
                               f"{math.degrees(rp_v[1][k]):.3f}")
                    f.write(f"{int(msg.plan_id)},{status},{follower},"
                            f"{t_rel:.4f},{k},"
                            f"{float(msg.t0) + k * float(msg.dt):.4f},"
                            f"{p[0, k]:.5f},{p[1, k]:.5f},{p[2, k]:.5f},"
                            f"{math.degrees(float(yaw[k])):.3f},"
                            f"\"{safe if k == 0 else ''}\",{rp_cols}\n")
        except (OSError, ValueError) as e:                     # noqa: BLE001
            self._log("warn", f"ctrl: policy_plan.csv not written ({e})")

    def _emit_policy_plan_viz_refused(self, plan, status: str, reason: str,
                                      t_rel: float,
                                      obs_t_rel=None) -> None:
        """Draw a plan the INTAKE refused, before it ever reached the filter.

        ``_log_plan`` draws only when a composed ``msg`` exists (:3089-3091),
        and the six intake refusals in ``_tick_policy_intake`` have none —
        they return before ``compose_plan`` runs. So on the only end-to-end
        hardware policy run on record, 12 of 29 plans were dropped as ``late``
        and drew NOTHING [측정: data/20260902/
        0902_215928/plans.jsonl — 29 plans, late 12 / clip 8 / reject 9,
        accept 0], while ``PolicyStatus`` went on saying ready. A map that
        goes blank while the counter climbs reads exactly like "the network
        stopped predicting", which is the ONE conclusion this rig exists to
        test — so the refused geometry has to be on screen too.

        THE COMPOSITION HERE IS FOR DRAWING ONLY. It never reproduces the
        leashed anchor ``_policy_anchor`` would have produced, because
        several of these refusals fire before an anchor is computable at all
        (the clock faults have no usable ``obs_t``). What it CAN do, and now
        does, is put the polyline where the vehicle WAS when the observation
        was taken: the fix-keyed eta history interpolated at ``obs_t``, the
        same series the real anchor reads (v2 A6), falling back to the tick
        eta when there is no usable stamp or no history.

        That fallback used to be the only path, and on a STATION-KEEPING
        control run the difference is millimetres. Under POLICY OBSERVE the
        pilot is flying: at 0.15 m/s the dominant refusal (``late``, 12 of 29
        plans on the only hardware run on record) is drawn 5-10 cm from where
        it belongs — further than the plan is long — so the map would show
        the network reaching at the wrong place and the operator would read
        the drawing, not the network (review 2026-09-03).

        It never touches ``rp['pending']``, never advances the
        epoch, and never counts toward ``n_plans``: nothing it produces can be
        flown. The polyline says only "this is what the network asked for,
        and the station refused it" — which is why every reason it writes is
        prefixed ``intake:``, so a row in ``policy_plan.csv`` drawn at the
        measured anchor can never be mistaken for one that was a real flight
        candidate.

        Silent on any failure: a plan refused BECAUSE its action is
        degenerate cannot be composed here either, and a missing polyline is
        the honest outcome — the reason still reaches ``plans.jsonl`` through
        the ``_log_plan`` call this one accompanies.
        """
        from .plan_stream import Verdict
        from .policy_frames import compose_plan

        rp = self.replay
        if (not (rp and rp.get("kind") == "policy") or self.cfg is None
                or self._policy_T_bt is None):
            return
        pc = self.cfg.policy
        try:
            action = np.asarray(getattr(plan, "action", None), dtype=float)
            if action.ndim != 2 or action.shape[0] < 2:
                return
            anchor = np.asarray(self._eta, float).copy()
            t0 = float(t_rel)
            if obs_t_rel is not None and math.isfinite(float(obs_t_rel)):
                t0 = float(obs_t_rel)
                hist = self._policy_eta_hist
                if hist is not None and len(hist) and self._t0_traj is not None:
                    try:
                        anchor = hist.interp(t0 + float(self._t0_traj))
                    except ValueError:
                        pass          # outside the kept window: tick eta
            msg, _info = compose_plan(
                action, anchor, anchor[3:5], self._policy_T_bt,
                obs_dt=float(rp["obs_dt_s"]), knot_dt=float(pc["knot_dt_s"]),
                t0=t0, plan_id=int(getattr(plan, "plan_id", -1)),
                obs_t_rel=t0,
                w_open=float(pc["gripper_width_open_m"]),
                w_closed=float(pc["gripper_width_closed_m"]))
        except (ValueError, TypeError, AttributeError, KeyError,
                IndexError):                                   # noqa: BLE001
            return
        self._emit_policy_plan_viz(
            msg, Verdict(str(status), None, {}, [f"intake: {reason}"]),
            float(t_rel))

    def _plan_stream_meta(self) -> dict:
        """ALWAYS written, the imu_dr rule: an absent key cannot tell 'this
        build had no plan stream' from 'no replay was flown'. The ``run``
        counters cover BOTH stream kinds (``run.kind`` says which); a policy
        run's own knobs and counters live in the ``policy`` block."""
        if self.cfg is None:
            return {"enabled": False}
        r = self.cfg.replay
        rp = self.replay if self.replay is not None else self._replay_last
        out = {"enabled": rp is not None,
               "config": {k: r.get(k) for k in
                          ("session", "v_max_m_s", "blend_s",
                           "stream_period_s", "horizon_s", "anchor_max_m",
                           "jump_max_m", "workspace_box_ned", "gripper",
                           "gripper_lookahead_s", "track_attitude")}}
        if rp is not None:
            out["run"] = {k: rp.get(k) for k in
                          ("kind", "session", "poses_sha1", "n_plans",
                           "released", "installed", "clipped", "rejected",
                           "period_s", "time_dilation", "duration_s",
                           "gripper_on", "grip_events", "grip_lookahead_s",
                           "attitude_track")}
        return out

    # ---------------------------------------------------------- live policy
    # shape: policy (2026-09-02, HARDWARE-UNVERIFIED). The trained UMI
    # diffusion policy's action chunks arrive as PolicyPlan on the bus
    # (backends/policy.py), are composed into datum-NED PlanMsgs here
    # (policy_frames.compose_plan) and go through the SAME filter/stitcher/
    # NedPlan seam the replay uses. What is DIFFERENT from a replay, each
    # with its spec decision: the clock conversion happens in ONE place (A1),
    # roll/pitch are dropped in TCP space (A2), the TCP is the ROV's own jaw
    # (A3), the anchor is leashed (A4), raw knots are resampled to 0.2 s
    # (A5), the anchor pose is the fix-keyed history at obs_t (A6), late
    # plans are skipped without a strike and escalation LATCHES (A7), epochs
    # drop stale plans (A8), a bridge/DR skips intake (A9), the hold tail is
    # a cost mask (A10), the blend is gated post-stitch (A11), the jaw width
    # is an open-loop estimate (A12), the worker's status must be fresh
    # (A13), the source string says whether the plant was real (A15), the
    # policy has its own limits incl. a REQUIRED box (A21), the end wording
    # never says "complete" (A22), and a bad action is a reject, never a
    # disengage (A23).
    def _policy_refusal(self) -> str:
        """Why a policy mission may NOT be armed — each cause its own
        sentence (the ``_follow_refusal`` rule)."""
        if hasattr(self.ctrl, "progress_m"):
            return (f"{self.cfg.mode} contours its own path and ignores "
                    f"streamed plans (set_path_plan_ned is a no-op there) — "
                    f"fly dobmpc, mpc, *_tuned or pid")
        if not hasattr(self.ctrl, "set_path_plan_ned"):
            return f"{self.cfg.mode} cannot consume a streamed plan"
        if not self.policy_present:
            return ("no policy worker — start the station with --policy "
                    "(and --mpc)")
        st = self._policy_status
        if st is None:
            return ("policy worker has not reported yet (no PolicyStatus "
                    "received)")
        age = now() - float(getattr(st, "stamp", 0.0))
        if age > 2.0:
            return f"policy worker silent for {age:.1f} s"
        if getattr(st, "error", ""):
            return f"policy worker error: {st.error}"
        if not getattr(st, "ready", False):
            what = ("still loading the checkpoint"
                    if getattr(st, "loading", False) else "not ready")
            note = getattr(st, "note", "") or ""
            return f"policy {what}" + (f" ({note})" if note else "")
        # The GRID's own state (verify 2026-09-02): a grid the builder
        # REFUSED (coverage < min_obs_coverage, live camera model != the
        # training one) left ready=True / error="" / depth_src set, and the
        # mission armed and could never receive a plan. `grid_ok` is False
        # by default, so a status from a producer that never filled it AND
        # never offered a grid still reads as "no grid yet" below — the
        # builder's IDLE sentinel (`DepthObsBuilder.why == "no grid set"`
        # before the first frame, policy_obs.py) is "no grid yet" too, not a
        # refusal (integration 2026-09-02).
        grid_why = str(getattr(st, "grid_why", "") or "")
        depth_src = str(getattr(st, "depth_src", "") or "")
        if grid_why == POLICY_GRID_WHY_IDLE:
            grid_why = ""
        if not bool(getattr(st, "grid_ok", False)) and (depth_src or grid_why):
            return (f"policy depth grid REFUSED — "
                    f"{grid_why or 'builder gave no reason'}"
                    + (f" (grid {depth_src})" if depth_src else ""))
        if not depth_src:
            note = getattr(st, "note", "") or ""
            return ("no depth grid yet — the policy worker has not received "
                    "a depth frame" + (f" ({note})" if note else ""))
        # The time base of every plan (verify 2026-09-02): the worker
        # derives obs_dt from the CHECKPOINT (down_sample_steps / dataset
        # fps); this side placed the raw knots with the config-derived one.
        # A different checkpoint would silently rescale every plan's speed.
        obs_dt_cfg = self._policy_obs_dt_cfg()
        obs_dt_st = float(getattr(st, "obs_dt_s", 0.0) or 0.0)
        if obs_dt_st > 0.0 and abs(obs_dt_st - obs_dt_cfg) > 1e-6:
            return (f"policy obs_dt mismatch — worker (checkpoint contract) "
                    f"{obs_dt_st * 1e3:.3f} ms vs config "
                    f"{obs_dt_cfg * 1e3:.3f} ms (POLICY_DOWN_SAMPLE_STEPS "
                    f"{int(POLICY_DOWN_SAMPLE_STEPS)} / policy.dataset_fps "
                    f"{float(self.cfg.policy['dataset_fps']):g}); fix "
                    f"policy.dataset_fps or the checkpoint")
        # The ACTION contract (2026-09-07): the worker reports the
        # checkpoint's action_repr; this side flies exactly ONE
        # (state.POLICY_ACTION_REPR). A stale policy.ckpt pointing at the
        # 10-dim network would otherwise arm and be decoded by width alone.
        # Unlike obs_dt an UNSET value is refused too: "" means the worker
        # never read a contract, and that is not a checkpoint to fly.
        # 2026-09-26: the ONE this side flies is policy.action_repr (config,
        # default state.POLICY_ACTION_REPR = pos_yaw_width), allow-listed by
        # state.POLICY_ACTION_REPRS_FLYABLE and pinned per mission at ARM.
        repr_st = str(getattr(st, "action_repr", "") or "")
        pinned = self._policy_pinned_repr()
        if pinned not in POLICY_ACTION_REPRS_FLYABLE:
            return (f"policy.action_repr {pinned!r} is not flyable "
                    f"({'|'.join(POLICY_ACTION_REPRS_FLYABLE)})")
        if repr_st != pinned:
            return (f"policy action_repr mismatch — worker (checkpoint "
                    f"contract) {repr_st or '(unset)'} vs policy.action_repr "
                    f"{pinned} (POLICY_ACTION_REPR default "
                    f"{POLICY_ACTION_REPR}); fix policy.ckpt or "
                    f"policy.action_repr")
        if not bool(getattr(st, "mount_ok", True)):
            why = str(getattr(st, "mount_why", "") or "")
            return ("policy mount mismatch — "
                    + (why or "the checkpoint's yaw axis (yaw_axis_R_frd_cam) "
                              "disagrees with hw_nav's R_frd_cam('main')"))
        if self.cfg.policy.get("workspace_box_ned") is None:
            return ("policy.workspace_box_ned is unset — the policy shape "
                    "REQUIRES the box (the only position-based protection "
                    "since the geofence removal); set it in hw_mpc.yaml")
        # THE TRUTH TABLE (2026-09-26, design safety): attitude_track needs
        # a 7-dim network, the K/M path live on THIS engagement, the mount
        # keys (mount_ok above — for pos_rpy_width the yaw_axis keys are
        # REQUIRED by backends/policy.py) and a trusted attitude measurement.
        # Any failure REFUSES ARM; the explicit fallback is attitude_track:
        # false with the 7-dim network (dropped-and-logged), the mandatory
        # first flight of any 7-dim checkpoint.
        if bool(self._policy_att("attitude_track")):
            if pinned != ACTION_REPR_POS_RPY_WIDTH:
                return (f"policy.attitude_track needs action_repr "
                        f"{ACTION_REPR_POS_RPY_WIDTH} (a {pinned} checkpoint "
                        f"has nothing to track) — set attitude_track: false")
            if self.observe:
                return ("policy.attitude_track is refused under LOW None "
                        "(observe): nothing is actuated, fly the 7-dim network "
                        "with attitude_track: false (dropped-and-logged) to "
                        "watch its roll/pitch")
            if not self._attitude_axes:
                return ("policy.attitude_track needs engage.attitude_axes "
                        "live on this engagement (K/M path) — engaged without "
                        "it; enable the block, re-engage")
            rr = self._last_health.get("rp_residual_deg")
            if rr is None or not math.isfinite(float(rr)):
                return ("policy.attitude_track needs the tag-vs-IMU attitude "
                        "residual (rp_residual_deg) — none measured yet")
            if float(rr) > self.RP_RESIDUAL_ARM_MAX_DEG:
                return (f"policy.attitude_track: tag-vs-IMU attitude residual "
                        f"{float(rr):.1f} deg > {self.RP_RESIDUAL_ARM_MAX_DEG:.0f} "
                        f"deg — the attitude measurement is not trusted "
                        f"enough to track (interlock iv)")
        return ""

    def _policy_obs_dt_cfg(self) -> float:
        """obs_dt as THIS side derives it (POLICY_DOWN_SAMPLE_STEPS /
        policy.dataset_fps) — the number the arm check and the intake
        compare the worker's checkpoint-derived one against."""
        return POLICY_DOWN_SAMPLE_STEPS / float(self.cfg.policy["dataset_fps"])

    def _arm_policy(self, sq: dict) -> None:
        """Arm the live policy mission at the vehicle's CURRENT pose.

        Mirrors :meth:`_arm_replay` with no session: the filter limits are
        ALL explicit from the policy block (v2 A5, incl. require_obs_t and
        the yaw-jump gate), the stitcher blends over ``blend_s``, nothing is
        pending until the first PolicyPlan arrives, and the scenario is a
        wall-clock run of ``max_run_s``. A NEW EPOCH starts here (A8): any
        plan the worker built against the pre-arm state is dropped.
        """
        from .plan_stream import FilterLimits, PlanFilter, PlanStitcher

        pc = self.cfg.policy
        box = pc.get("workspace_box_ned")
        if box is None:                   # _policy_refusal already said no
            self._refuse("policy.workspace_box_ned is unset")
            return
        # ATTITUDE (2026-09-26): the rp gates are set ONLY for a tracked
        # mission and only when this build's FilterLimits carries them;
        # otherwise the limits object is the pre-variant one, field for
        # field (test_policy_filter_limits_equal_config).
        attitude_track = (bool(self._policy_att("attitude_track"))
                          and bool(self._attitude_axes) and not self.observe)
        rp_lim = {}
        if attitude_track:
            have = {f.name for f in dataclasses.fields(FilterLimits)}
            want = {
                "rp_reject_rad": math.radians(float(self._policy_att("rp_reject_deg"))),
                "rp_rate_max": float(self._policy_att("pq_max_rad_s")),
                "rp_jump_max_rad": math.radians(float(self._policy_att("rp_jump_max_deg"))),
            }
            rp_lim = {k: v for k, v in want.items() if k in have}
            if len(rp_lim) != len(want):
                self._log("warn", f"ctrl: this build's PlanFilter has no "
                                  f"attitude gates ({sorted(set(want) - have)}) "
                                  f"— roll/pitch plans pass the filter "
                                  f"unchecked (T2 missing; T1 clip + T3 "
                                  f"set_path_plan_ned still apply)")
        lim = FilterLimits(
            v_max=float(pc["v_max_m_s"]),
            a_max=float(pc["a_max_m_s2"]),
            r_max=float(pc["r_max_rad_s"]),
            anchor_max_m=float(pc["anchor_max_m"]),
            jump_max_m=float(pc["jump_max_m"]),
            obs_max_age_s=float(pc["obs_max_age_s"]),
            box_ned_min=tuple(float(v) for v in box[0]),
            box_ned_max=tuple(float(v) for v in box[1]),
            require_obs_t=True,
            yaw_jump_max_rad=math.radians(float(pc["yaw_jump_max_deg"])),
            clip_ratio_max=float(pc["clip_ratio_max"]),
            reject_escalate=int(pc.get("reject_escalate", 3)),
            follower_owns_dynamics=bool(
                pc.get("follower_owns_dynamics", False)),
            **rp_lim,
            # THE BOX STOPS REJECTING UNDER OBSERVE, and only there. It is
            # the last position-based protection on a run that can command,
            # so it is hard everywhere else — but a muted controller has
            # nothing to protect, and the box is drawn around the ENGAGE
            # DATUM: the pilot flying 1.2 m to look at the bottle from
            # somewhere else would put every knot outside it, reject every
            # plan, and leave a screen of faded dashed lines that reads
            # exactly like "the network died". The margin is still computed
            # and the sentence still written to plans.jsonl.
            box_enforced=not self.observe)
        self._plan_filter = PlanFilter(lim)
        self._plan_stitcher = PlanStitcher(blend_s=float(pc["blend_s"]))
        self._policy_epoch += 1
        self._policy_inbox = None
        self._policy_last_seen = -1
        p0 = np.asarray(self._eta[:3], float).copy()
        yaw0 = float(self._eta[5])
        self._policy_eta_start = tuple(float(v) for v in self._eta)
        t = now()
        w0 = self._grip_est.width(t)
        g0 = self._grip_est.g(t)
        grip0 = ("close" if g0 < float(pc["gripper_close_below"])
                 else "open" if g0 > float(pc["gripper_open_above"])
                 else "neutral")
        max_run = float(pc["observe_max_run_s"] if self.observe
                        else pc["max_run_s"])
        # THE JAW IS OFF BECAUSE THIS IS AN OBSERVE RUN, not because the
        # config happened to say so. `_tick_replay_gripper` short-circuits on
        # this key, so forcing it here kills the whole jaw branch structurally
        # rather than relying on hw_mpc.yaml's `gripper: false` surviving
        # someone's edit. Mirrored into `scen` below so the record says WHY.
        gripper_on = bool(pc["gripper"]) and not self.observe
        obs_dt = self._policy_obs_dt_cfg()
        # CHECKPOINT TRUTH = the PolicyStatus, not the config (2026-09-11:
        # the panel picker can swap the network between engagements, so
        # `policy.ckpt` is only the launch seed). Pinned here at ARM, the
        # obs_dt pattern: the intake rejects a plan whose ckpt_sha1 is not
        # this one. `_policy_refusal` already required a READY status, so a
        # real checkpoint has its sha here; the stub has "" on both sides.
        st = self._policy_status
        ckpt = str(getattr(st, "ckpt", "") or pc["ckpt"])
        ckpt_sha1 = str(getattr(st, "ckpt_sha1", "") or "")
        self.replay = {
            "kind": "policy",
            "observe": bool(self.observe),
            "epoch": int(self._policy_epoch),
            "ckpt_sha1": ckpt_sha1,      # the armed network (intake gate)
            "pending": [], "extras": {},
            "halted": "", "end_reason": "",
            "n_plans": 0, "received": 0, "released": 0, "installed": 0,
            "clipped": 0, "rejected": 0, "late": 0, "skip_bridge": 0,
            "skip_halted": 0, "drop_epoch": 0, "drop_old": 0,
            "reject_compose": 0, "reject_blend": 0, "reject_clock": 0,
            "reject_obs_dt": 0, "reject_action_repr": 0, "reject_ckpt": 0,
            "stale_events": 0,
            "action_repr": "",           # of the last COMPOSED plan
            # THE PIN (2026-09-26): what this mission flies; the intake
            # rejects a plan whose contract is not this.
            "action_repr_pinned": self._policy_pinned_repr(),
            # roll/pitch TRACKED (7-dim + attitude axes live) vs levelled.
            "attitude_track": bool(attitude_track),
            "reject_rp_mag": 0, "clip_rp_rate": 0,
            "div_rp_max_seen_deg": 0.0, "div_rp_streak": 0,
            "stale_flag": False, "t_last_accept": None,
            "t_late_warn": -1e9, "bridge_since": None, "div_streak": 0,
            "escalations": 0, "t_escalate_warn": -1e9,
            "t_bridge_warn": -1e9,
            "hold_frac": float("nan"), "infer_ms": deque(maxlen=1000),
            "last_intake": None,
            "period_s": float(pc["period_s"]),
            "duration_s": max_run,
            "obs_dt_s": float(obs_dt),
            "anchor_mode": str(pc["anchor"]),
            "hold_tail": str(pc["hold_tail"]),
            "gripper_on": gripper_on,
            "div_max_seen_m": 0.0,
            "grip_state": grip0, "grip_since": 0.0, "grip_drive": 0.0,
            "grip_events": 0, "grip_w_at_arm": float(w0),
            "grip_lookahead_s": float(pc["gripper_lookahead_s"]),
            "grip_g": float("nan"),      # (schema 13) set per tick
            # YAW REFERENCE FILTER state (2026-09-14): the low-passed turn
            # rate carried plan to plan, the t0 it was last advanced at, and
            # the counters the meta reports. Reset every arm.
            "yaw_filt": {"omega": 0.0, "t0_prev": None, "n": 0,
                         "guard_clamps": 0, "rate_clips": 0},
        }
        scen = {
            "kind": "policy",
            "ckpt": ckpt,                # what the worker HOLDS (status), else config
            "ckpt_sha1": ckpt_sha1,      # "" for the stub / a not-yet-hashed load
            "epoch": int(self._policy_epoch),
            "obs_dt_s": float(obs_dt),
            "knot_dt_s": float(pc["knot_dt_s"]),
            "period_s": float(pc["period_s"]),
            "anchor": str(pc["anchor"]),
            "anchor_leash_m": float(pc["anchor_leash_m"]),
            # THE POOLING KEYS of the 6-DoF variant (schema 16): the
            # representation this mission was pinned to, and whether its
            # roll/pitch were tracked. Absent (older meta) = pos_yaw_width
            # / false.
            "action_repr": self._policy_pinned_repr(),
            "attitude_track": bool(attitude_track),
            "rp_max_deg": float(self._policy_att("rp_max_deg")),
            "anchor_leash_rp_deg": float(self._policy_att("anchor_leash_rp_deg")),
            "hold_tail": str(pc["hold_tail"]),
            "v_max_m_s": float(pc["v_max_m_s"]),
            "blend_s": float(pc["blend_s"]),
            "origin_ned": [float(p0[0]), float(p0[1])],
            "depth_ned": float(p0[2]),
            "yaw_fixed_ned_deg": math.degrees(yaw0),
            "gripper": gripper_on,
            "gripper_lookahead_s": float(pc["gripper_lookahead_s"]),
            "q_scale": float(pc["q_scale"]),
            # The panel and every reader of `trajectory` in the run meta learn
            # from ONE key that this reference was drawn and never followed.
            "observe": bool(self.observe),
            "T_run_s": max_run,
            # The generic wall-clock backstop; the intake ends the run at
            # max_run_s itself with the A22 wording well before this.
            "path_timeout_s": max_run * max(1.0,
                                            float(self.cfg.traj_timeout_factor)),
        }
        self.ctrl.scenario = scen
        # Never inherit a stale plan from an earlier mission (the _arm_path
        # rule); the first accepted PolicyPlan installs the real one.
        self.ctrl.set_path_plan_ned(None)
        if not self.observe:
            # NOT under LOW None: nothing is stepped, so a cost scale on the
            # holder would be a record claim (`plan_q_scale_flown`) about a
            # follower that never flew (the holder is the PID anyway, which
            # has neither hook — this guard says so in words).
            self._apply_plan_cost_scale(float(pc["q_scale"]))
            self._apply_plan_path_cost(pc.get("along_scale"))
        self.traj_on = True
        self.station = None
        self._t0_traj = now()
        self._tau, self._tau_t, self._path_lag = 0.0, now(), 0.0
        self.phase, self.phase_detail = "policy", "waiting for the first plan"
        self.reason = ("policy OBSERVE running (not commanding)"
                       if self.observe else "policy running")
        head = "POLICY OBSERVE" if self.observe else "POLICY"
        what = (f"{head} {Path(ckpt).name}"
                f"{f' (sha1 {ckpt_sha1[:12]})' if ckpt_sha1 else ''}: "
                f"action {self._policy_pinned_repr()}"
                + (f", roll/pitch TRACKED (rp_max "
                   f"{float(self._policy_att('rp_max_deg')):.0f} deg, reject "
                   f"{float(self._policy_att('rp_reject_deg')):.0f} deg, "
                   f"div {float(self._policy_att('div_max_rp_deg')):.0f} deg)"
                   if attitude_track else
                   (", roll/pitch DROPPED-AND-LOGGED (attitude_track false)"
                    if self._policy_pinned_repr() == ACTION_REPR_POS_RPY_WIDTH
                    else ""))
                + f", max {max_run:.0f} s, "
                f"anchor {pc['anchor']} (leash {pc['anchor_leash_m']:.2f} m), "
                f"v_max {pc['v_max_m_s']:.2f} m/s, knots {pc['knot_dt_s']:.1f} s, "
                f"hold tail {pc['hold_tail']}, box "
                f"{[list(b) for b in box]}"
                + ("" if not self.observe else " [NOT ENFORCED]")
                + f", gripper {'ON' if gripper_on else 'off'}"
                + (f", lookahead {float(pc['gripper_lookahead_s']):g} s"
                   if gripper_on else "")
                + ("" if not self.observe
                   else " (forced off by OBSERVE)")
                + f"; jaw width assumed "
                f"{w0 * 1e3:.0f} mm at START (open-loop estimate, init "
                f"{float(pc['gripper_width_init_m']) * 1e3:.0f} mm = OPEN)")
        if self.observe:
            # WHAT IS DISARMED, ONCE, IN FULL. An observe run deliberately
            # removes every automatic stop, so the operator must be able to
            # read from the log which protections are simply not there — and
            # `_policy_meta.gates_enforced` says the same thing in the record.
            what += ("; DISARMED for OBSERVE: divergence guard, escalation "
                     "latch, bridge-too-long stop, workspace box, tag-loss "
                     "disengage. NOTHING is commanded — you have the sticks.")
        self._event("POLICY OBSERVE started" if self.observe
                    else "POLICY started")
        self._log_event(f"{head} started - {what}")
        self._log("warn", f"ctrl: {what}")
        zh = (self.cfg.policy or {}).get("z_hold_above_floor_m")
        if zh is not None:
            # Loud, once per arm: a reference the policy did not ask for is
            # flying. If this line is in a run's log and nobody meant it, the
            # key was left in hw_mpc.yaml past the day it was for.
            self._log("warn", f"ctrl: Z HOLD — every policy plan knot pinned "
                              f"to {float(zh):.2f} m above the tag floor "
                              f"(policy.z_hold_above_floor_m); the policy's "
                              f"own dz is DISCARDED. Temporary, 2026-09-12.")
            self._log_event(f"Z HOLD {float(zh):.2f} m above floor (temporary)")
        yf = (self.cfg.policy or {}).get("yaw_ref_filter")
        if yf is not None and self.observe:
            self._log("info", "ctrl: yaw ref filter configured but SKIPPED "
                              "under POLICY OBSERVE — the drawn heading is "
                              "the network's raw dyaw (nothing is flown)")
        elif yf is not None:
            self._log("warn", f"ctrl: YAW REF FILTER — plan yaw = flown "
                              f"reference + low-passed turn rate (tau "
                              f"{float(yf['tau_s']):.1f} s, <= "
                              f"{float(yf['rate_deg_s']):.1f} deg/s, guard "
                              f"{float(yf['guard_deg']):.0f} deg from the "
                              f"measured yaw); the policy's dyaw is filtered, "
                              f"never re-anchored to the measurement "
                              f"(policy.yaw_ref_filter).")
            self._log_event(f"YAW REF FILTER tau {float(yf['tau_s']):.1f} s, "
                            f"rate {float(yf['rate_deg_s']):.1f} deg/s, guard "
                            f"{float(yf['guard_deg']):.0f} deg")

    def _policy_bridged(self) -> bool:
        """Is the controller flying on something other than a fresh tag fix
        (station bridge tier != none, or DR control)? (v2 A9)"""
        if self._bridge is not None and self._bridge.tier != SB.TIER_NONE:
            return True
        return bool(self.dr_control and self._dr_last is not None
                    and self._dr_last.get("ok"))

    def _policy_worker_fault(self, t_now: float) -> str:
        """A policy worker that stopped talking, or reported an error,
        mid-mission (v2 A13). Empty string = fine."""
        st = self._policy_status
        if st is None:
            return "no PolicyStatus received"
        age = t_now - float(getattr(st, "stamp", 0.0))
        if age > 2.0:
            return f"policy worker silent for {age:.1f} s"
        if getattr(st, "error", ""):
            return f"policy worker error: {st.error}"
        if getattr(st, "loading", False) and not getattr(st, "ready", False):
            # A checkpoint SWAP under an armed mission (2026-09-11, the panel
            # picker): PolicyWorker.set_ckpt refuses while a mission is
            # engaged, so reaching this means the guard on that side was
            # bypassed or raced — the armed network is gone either way.
            return ("policy worker reloading a checkpoint (a panel swap under "
                    "an armed mission)")
        return ""

    def _policy_end(self, end_reason: str, event: str, level: str,
                    log: str, reason: str) -> None:
        """End the policy mission: record WHY (v2 A22), announce it, and
        hold the MEASURED pose (set_traj(False) — never the far endpoint of
        a stream that just failed)."""
        rp = self.replay
        if rp is None:
            return
        # PREFIXED under observe so no reader can find `end_reason: max_run`
        # in an observe record and take it for a clean closed-loop run that
        # simply ran its clock out. Every guard that could have ended this
        # run early was disarmed; the ending says which kind of run it was.
        rp["end_reason"] = (f"observe_{end_reason}" if self.observe
                            else str(end_reason))
        self._event(event)
        self._log(level, log)
        self.set_traj(False)
        self.reason = reason

    def _policy_halt(self, why: str, event: str, log: str) -> None:
        """The escalation LATCH (v2 A7): the mission stays armed and the
        reference holds the last endpoint, but no plan is installed again
        until STOP TRAJ / START re-arms."""
        rp = self.replay
        if rp is None:
            return
        rp["halted"] = str(why)
        if not rp.get("end_reason"):
            rp["end_reason"] = str(why)
        self._event(event)
        self._log("error", log)
        qs = float(getattr(self.ctrl, "plan_cost_scale", 1.0))
        self.reason = (f"policy halted ({why}) — STOP/START to re-arm"
                       + (f" (position weight x{qs:g} still in force)" if qs != 1.0 else ""))

    def _policy_anchor(self, obs_t_mono: float, obs_t_rel: float):
        """The pose a PolicyPlan is composed against (v2 A4).

        ``x_meas`` = the fix-keyed eta history interpolated at the plan's
        obs_t (monotonic clock, v2 A6) — the vehicle where it WAS when the
        depth frame was taken, not where it is now; falls back to the tick
        eta when the history is empty. The reference ``r`` is the active
        stitched reference sampled at ``obs_t_rel`` (both branches coincide
        with no plan installed). ``leash`` starts at the measurement and
        moves toward the reference by at most the leash; ``reference`` takes
        the reference outright (the plan doc's B1 hazard in reverse: the
        reference can run away); ``measured`` resets the tracking error every
        replan. Roll/pitch are always the MEASURED ones — they become the
        level the plan is projected at.
        """
        from .policy_frames import leash_anchor

        pc = self.cfg.policy
        hist = self._policy_eta_hist
        src = "tick"
        x_meas = np.asarray(self._eta, float).copy()
        if hist is not None and len(hist):
            try:
                x_meas = hist.interp(float(obs_t_mono))
                src = "history"
            except ValueError:
                pass
        st = self._plan_stitcher
        if st is not None and st.has_plan():
            p_ref, yaw_ref = self._replay_ref_now(float(obs_t_rel))
        else:
            p_ref, yaw_ref = x_meas[:3].copy(), float(x_meas[5])
        mode = str(pc["anchor"])
        # ATTITUDE LEASH (2026-09-26, D12): under attitude_track the anchor's
        # roll/pitch move from the MEASURED attitude toward the flown
        # attitude reference by at most anchor_leash_rp_deg — the same
        # reason the position/yaw leash exists (a reference re-anchored to
        # a 0.5 s-stale measurement every replan fed the 2026-09-14 yaw
        # wobble). 0 deg / no reference / not tracked = measured, the
        # pre-variant call, byte-identical.
        rp_kw = {}
        rp_ref = None
        if self._stream_att_tracked() and mode == "leash":
            leash_rp = math.radians(float(self._policy_att("anchor_leash_rp_deg")))
            if st is not None and st.has_plan() and leash_rp > 0.0:
                rp_ref = self._replay_ref_att_now(float(obs_t_rel))
            if rp_ref is not None:
                try:
                    if "rp_ref" in inspect.signature(leash_anchor).parameters:
                        rp_kw = {"rp_ref": rp_ref, "leash_rp": leash_rp}
                except (TypeError, ValueError):
                    rp_kw = {}
        if mode == "leash":
            anchor, info = leash_anchor(
                x_meas, p_ref, yaw_ref, float(pc["anchor_leash_m"]),
                math.radians(float(pc["anchor_leash_yaw_deg"])), **rp_kw)
        else:
            anchor = x_meas.copy()
            off = np.asarray(p_ref, float) - x_meas[:3]
            dyaw = _wrap_pi(float(yaw_ref) - float(x_meas[5]))
            if mode == "reference":
                anchor[:3] = p_ref
                anchor[5] = _wrap_pi(float(yaw_ref))
            info = {"offset_ned": off, "offset_m": float(np.linalg.norm(off)),
                    "dyaw": dyaw,
                    "offset_applied_m": (float(np.linalg.norm(off))
                                         if mode == "reference" else 0.0),
                    "dyaw_applied": (dyaw if mode == "reference" else 0.0),
                    "clipped_pos": False, "clipped_yaw": False}
        info["mode"] = mode
        info["x_meas_src"] = src
        info["anchor_pose_meas"] = [float(v) for v in x_meas]
        info["anchor_pose_used"] = [float(v) for v in anchor]
        if self._stream_att_tracked():
            info["anchor_rp_meas"] = [float(x_meas[3]), float(x_meas[4])]
            info["anchor_rp_used"] = [float(anchor[3]), float(anchor[4])]
            info["anchor_rp_ref"] = (None if rp_ref is None
                                     else [float(v) for v in rp_ref])
        return anchor, info

    def _policy_blend_gate(self, verdict, t_rel: float, extra):
        """Post-stitch gate (v2 A11): what the BLEND would ask of the
        vehicle over the next blend_s, from the stitcher's own preview. A
        failure is a geometric strike (reject_external), so three in a row
        escalate like any other reject."""
        pc = self.cfg.policy
        st = self._plan_stitcher
        peaks = st.preview_install(verdict.plan, float(t_rel))
        v_lim = float(pc["blend_v_max_m_s"])
        a_lim = float(pc["blend_a_max_m_s2"])
        r_lim = 2.0 * float(pc["r_max_rad_s"])
        margins = dict(verdict.margins)
        margins["blend_v"] = v_lim - peaks["v_peak"]
        margins["blend_a"] = a_lim - peaks["a_peak"]
        margins["blend_r"] = r_lim - peaks["r_peak"]
        if extra is not None:
            extra["blend"] = dict(peaks)
        bad = []
        if peaks["v_peak"] > v_lim:
            bad.append(f"speed {peaks['v_peak']:.3f}/{v_lim:.3f} m/s")
        if peaks["a_peak"] > a_lim:
            bad.append(f"accel {peaks['a_peak']:.3f}/{a_lim:.3f} m/s^2")
        if peaks["r_peak"] > r_lim:
            bad.append(f"yaw rate {peaks['r_peak']:.3f}/{r_lim:.3f} rad/s")
        if peaks.get("rp_rate_peak") is not None:
            # ONLY an rp-carrying candidate has this peak (the 4-DoF blend
            # record is byte-identical): the attitude rate the stitch asks.
            rp_lim = float(self._policy_att("blend_rp_rate_max"))
            margins["blend_rp_rate"] = rp_lim - float(peaks["rp_rate_peak"])
            if float(peaks["rp_rate_peak"]) > rp_lim:
                bad.append(f"attitude rate {float(peaks['rp_rate_peak']):.3f}/"
                           f"{rp_lim:.3f} rad/s")
        if not bad:
            verdict.margins = margins
            return verdict
        # The post-stitch blend is a CONTINUITY gate, so it goes the same way
        # as the filter's own under `follower_owns_dynamics`: measured, said
        # out loud, not enforced. The NMPC gets the blended reference either
        # way and its actuator limits decide what actually happens.
        if bool(self._plan_filter.limits.follower_owns_dynamics):
            verdict.margins = margins
            verdict.reasons = list(verdict.reasons) + [
                f"blend: stitching this plan in over {pc['blend_s']:.2f} s "
                f"peaks at {', '.join(bad)} (post-stitch gate)  "
                f"[not enforced: the NMPC follower owns dynamics]"]
            return verdict
        rp = self.replay
        if rp is not None:
            rp["reject_blend"] = int(rp.get("reject_blend", 0)) + 1
        return self._plan_filter.reject_external(
            margins, list(verdict.reasons) + [
                f"blend: stitching this plan in over "
                f"{pc['blend_s']:.2f} s would peak at {', '.join(bad)} "
                f"(post-stitch gate) — rejecting."])

    def _tick_policy_intake(self, t_rel: float) -> None:
        """Turn the latest PolicyPlan into a pending PlanMsg — or end the
        mission. Runs at the top of :meth:`_tick_replay`, INSIDE the tick's
        try/except (v2 A23), before the filter.

        Order: wall-clock completion (A22) -> worker fault (A13) -> bridge
        watch (A9) -> stale watchdog -> latch/bridge skips (A7/A9) -> clock
        conversion + freshness (A1/A7: a late plan is skipped, NOT a strike)
        -> anchor (A4/A6) -> compose in TCP space onto the 0.2 s grid
        (A2/A5; a bad action is a reject with a reason, A23) -> pending.
        """
        from .plan_stream import OBS_T_FUTURE_TOL_S, Verdict
        from .policy_frames import (action_repr_of, compose_plan,
                                    filter_plan_yaw, guard_yaw)

        rp = self.replay
        pc = self.cfg.policy
        t_now = now()
        # OBSERVE runs its own, much longer clock (policy.observe_max_run_s):
        # the operator is flying to several places and looking, and max_run_s
        # (500 s since 2026-09-11; 120 before) is sized for ONE closed-loop
        # attempt. The key that was used is in the record.
        key = "observe_max_run_s" if self.observe else "max_run_s"
        max_run = float(pc[key])
        if t_rel >= max_run:
            n = int(rp["installed"])
            head = "POLICY OBSERVE" if self.observe else "POLICY"
            self._policy_end(
                "max_run",
                f"{head} {key} reached ({n} plans installed)", "warn",
                f"ctrl: {head} {key} ({max_run:.0f} s) reached — {n} plans "
                f"installed; holding HERE.",
                f"policy {key} reached (DP hold)")
            return
        why = self._policy_worker_fault(t_now)
        if why:
            if why.startswith("policy worker reloading"):
                # Its own end_reason (A22 wording): "silent" would send the
                # post-mortem looking for a dead thread when the record's
                # own ckpt pin says the network was swapped out from under
                # the mission.
                self._policy_end(
                    "worker_reloading", f"POLICY worker reloading — {why}",
                    "error",
                    f"ctrl: POLICY worker is reloading a checkpoint ({why}) "
                    f"— no plan from the armed network can arrive; holding "
                    f"HERE.",
                    "policy worker reloading (DP hold)")
            else:
                self._policy_end(
                    "worker_silent", f"POLICY worker lost — {why}", "error",
                    f"ctrl: POLICY worker lost ({why}) — no plans can arrive; "
                    f"holding HERE.",
                    "policy worker lost (DP hold)")
            return
        bridged = self._policy_bridged()
        if bridged:
            if rp.get("bridge_since") is None:
                rp["bridge_since"] = float(t_rel)
            elif float(t_rel) - float(rp["bridge_since"]) > float(pc["stale_s"]):
                if self.observe:
                    # A WARNING, NOT A STOP. The premise of the stop is "the
                    # vehicle is being flown against an estimated pose"; here
                    # it is being flown by a human, and the only consequence
                    # of a bridged state is that plans are SKIPPED (the
                    # skip_bridge branch below still fires, so nothing is
                    # composed from a pose we do not trust). Ending the run
                    # for it would just cost the operator the whole session
                    # the first time the tag went behind the bottle.
                    if now() - float(rp.get("t_bridge_warn", -1e9)) > 5.0:
                        rp["t_bridge_warn"] = now()
                        self._log(
                            "warn",
                            f"ctrl: POLICY OBSERVE on a bridged/DR state for "
                            f"{float(t_rel) - float(rp['bridge_since']):.1f} s "
                            f"— plans are being SKIPPED (no fresh tag fix to "
                            f"anchor them to). Not stopping (observe).")
                else:
                    self._policy_end(
                        "bridge", "POLICY bridged too long — stopped", "error",
                        f"ctrl: POLICY flew on a bridged/DR state for more than "
                        f"{float(pc['stale_s']):.1f} s — a policy conditioned on "
                        f"an estimated pose is not this run's purpose; holding "
                        f"HERE.",
                        "policy bridge (DP hold)")
                    return
        else:
            rp["bridge_since"] = None
        plan = self._policy_inbox
        self._policy_inbox = None
        # STALE WATCHDOG: no accepted plan for stale_s. The reference already
        # holds (stitcher endpoint / the arm pose) — this is the operator's
        # cue, once per episode, not an abort.
        last = rp.get("t_last_accept")
        since = float(t_rel) - (float(last) if last is not None else 0.0)
        if since > float(pc["stale_s"]):
            if not rp.get("stale_flag"):
                rp["stale_flag"] = True
                rp["stale_events"] = int(rp.get("stale_events", 0)) + 1
                self._log(
                    "warn", f"ctrl: policy stale — no plan accepted for "
                            f"{since:.1f} s (reference is holding)")
        else:
            rp["stale_flag"] = False
        if plan is None:
            return
        if rp.get("halted"):
            rp["skip_halted"] = int(rp.get("skip_halted", 0)) + 1
            return
        r_now = self._replay_ref_now(float(t_rel))
        pid = int(getattr(plan, "plan_id", -1))
        base_extra = {"plan_id": pid, "epoch": int(getattr(plan, "epoch", -1)),
                      "infer_ms": float(getattr(plan, "infer_ms", float("nan"))),
                      "depth_src": str(getattr(plan, "depth_src", "")),
                      "depth_coverage": getattr(plan, "depth_coverage", None),
                      "depth_valid": getattr(plan, "depth_valid", None),
                      "obs_rows_t": getattr(plan, "obs_rows_t", None),
                      "obs_fix_t": getattr(plan, "obs_fix_t", None),
                      "obs_pair_dt_s": getattr(plan, "obs_pair_dt_s", None),
                      "pair_dup": getattr(plan, "pair_dup", None),
                      "ckpt_sha1": str(getattr(plan, "ckpt_sha1", "")),
                      "t_rel_at_intake": float(t_rel)}
        if bridged:
            rp["skip_bridge"] = int(rp.get("skip_bridge", 0)) + 1
            why = ("skipped: the controller is flying on a bridged/DR state, "
                   "not a fresh tag fix (policy plans need one).")
            self._log_plan(None, Verdict("skip_bridge", None, {}, [why]),
                           t_rel, r_now, base_extra)
            self._emit_policy_plan_viz_refused(plan, "skipped", why, t_rel)
            return
        # CLOCKS (A1): the ONE place a monotonic obs_t becomes mission time.
        # A missing / non-finite stamp is the WORKER's failure: a reject
        # line, never a disengage (A23; float(None) used to raise into the
        # tick guard — verify 2026-09-02).
        try:
            obs_t_mono = float(getattr(plan, "obs_t", None))
        except (TypeError, ValueError):
            obs_t_mono = float("nan")
        if not math.isfinite(obs_t_mono):
            rp["reject_clock"] = int(rp.get("reject_clock", 0)) + 1
            rp["rejected"] = int(rp.get("rejected", 0)) + 1
            why = (f"schema: obs_t {getattr(plan, 'obs_t', None)!r} is not a "
                   f"finite monotonic stamp — the plan cannot be placed on "
                   f"the mission clock; rejected, no strike.")
            self._log("error", f"ctrl: policy plan {pid} rejected — "
                                       f"{why}")
            self._log_plan(None, Verdict("reject", None, {}, [why]),
                           t_rel, r_now, base_extra)
            self._emit_policy_plan_viz_refused(plan, "reject", why, t_rel)
            return
        obs_t_rel = obs_t_mono - float(self._t0_traj)
        age = float(t_rel) - obs_t_rel
        plan_obs_dt = float(getattr(plan, "obs_dt_s", 0.0) or 0.0)
        base_extra.update({"obs_t_mono": obs_t_mono, "obs_t_rel": obs_t_rel,
                           "age_at_intake_s": age,
                           "obs_dt_s": (plan_obs_dt if plan_obs_dt > 0.0
                                        else None),
                           "obs_dt_cfg_s": float(rp["obs_dt_s"])})
        if age < -OBS_T_FUTURE_TOL_S:
            # The FUTURE half of the clock-domain check (the filter has the
            # same one, but a future t0 sits in `pending` until the clock
            # reaches it and the filter then sees age ~ 0 — verify
            # 2026-09-02). Not a strike: it is a clock fault, not geometry.
            rp["reject_clock"] = int(rp.get("reject_clock", 0)) + 1
            rp["rejected"] = int(rp.get("rejected", 0)) + 1
            why = (f"clock domain: obs_t is {-age:.2f} s in the FUTURE of the "
                   f"mission clock (tolerance {OBS_T_FUTURE_TOL_S:.2f} s) — "
                   f"the stamp is on another clock; rejected, no strike.")
            self._log("error", f"ctrl: policy plan {pid} rejected — "
                                       f"{why}")
            self._log_plan(None, Verdict(
                "reject", None, {"obs_age": float(pc["obs_max_age_s"]) - age},
                [why]), t_rel, r_now, base_extra)
            self._emit_policy_plan_viz_refused(plan, "reject", why, t_rel)
            return
        if plan_obs_dt > 0.0 and abs(plan_obs_dt - float(rp["obs_dt_s"])) > 1e-6:
            # The plan's own time base disagrees with the one the knots are
            # about to be placed on (verify 2026-09-02): composing it would
            # rescale its speed by the ratio. Reject with the two numbers.
            rp["reject_obs_dt"] = int(rp.get("reject_obs_dt", 0)) + 1
            rp["rejected"] = int(rp.get("rejected", 0)) + 1
            why = (f"obs_dt mismatch: plan {plan_obs_dt * 1e3:.3f} ms "
                   f"(checkpoint contract) vs config "
                   f"{float(rp['obs_dt_s']) * 1e3:.3f} ms "
                   f"(POLICY_DOWN_SAMPLE_STEPS / policy.dataset_fps) — "
                   f"composing would rescale the plan's speed x"
                   f"{float(rp['obs_dt_s']) / plan_obs_dt:.3f}; rejected, "
                   f"no strike.")
            self._log("error", f"ctrl: policy plan {pid} rejected — "
                                       f"{why}")
            self._log_plan(None, Verdict("reject", None, {}, [why]),
                           t_rel, r_now, base_extra)
            self._emit_policy_plan_viz_refused(plan, "reject", why, t_rel,
                                               obs_t_rel)
            return
        # The ACTION contract (2026-09-07, the obs_dt pattern): the plan says
        # which representation its columns are in (the checkpoint contract;
        # inferred from the width when the producer left it unset), and this
        # side composes exactly ONE. A (16, 10) pose10d plan is still
        # DECODABLE by compose_plan -- that is what makes this gate necessary:
        # without it a stale checkpoint would fly by width alone. No strike.
        action = np.asarray(getattr(plan, "action", None), dtype=float)
        plan_repr = str(getattr(plan, "action_repr", "") or "")
        if not plan_repr and action.ndim == 2:
            try:
                plan_repr = action_repr_of(action.shape[1])
            except ValueError:
                plan_repr = ""
        base_extra["action_repr"] = plan_repr or None
        pinned = str(rp.get("action_repr_pinned") or self._policy_pinned_repr())
        if plan_repr != pinned:
            rp["reject_action_repr"] = int(rp.get("reject_action_repr", 0)) + 1
            rp["rejected"] = int(rp.get("rejected", 0)) + 1
            why = (f"action_repr mismatch: plan {plan_repr or 'unknown'} "
                   f"(action {tuple(action.shape) if action.ndim else '?'}, "
                   f"checkpoint contract) vs policy.action_repr {pinned} "
                   f"(pinned at ARM; POLICY_ACTION_REPR default "
                   f"{POLICY_ACTION_REPR}) — this mission composes only the "
                   f"pinned action; rejected, no strike.")
            self._log("error", f"ctrl: policy plan {pid} rejected — "
                                       f"{why}")
            self._log_plan(None, Verdict("reject", None, {}, [why]),
                           t_rel, r_now, base_extra)
            self._emit_policy_plan_viz_refused(plan, "reject", why, t_rel,
                                               obs_t_rel)
            return
        # THE CHECKPOINT PIN (2026-09-11, the obs_dt pattern): the panel can
        # swap the network between engagements, so the sha1 this mission was
        # ARMED against (`rp["ckpt_sha1"]`, from the PolicyStatus at ARM) is
        # compared with the one each plan carries. A plan from another
        # checkpoint is not this mission's network: rejected, no strike.
        # Permissive only when BOTH are empty — the stub session has no
        # checkpoint (ckpt_sha1_head "") and neither do hand-built plans.
        armed_sha = str(rp.get("ckpt_sha1", "") or "")
        plan_sha = str(getattr(plan, "ckpt_sha1", "") or "")
        if (armed_sha or plan_sha) and armed_sha != plan_sha:
            rp["reject_ckpt"] = int(rp.get("reject_ckpt", 0)) + 1
            rp["rejected"] = int(rp.get("rejected", 0)) + 1
            why = (f"ckpt mismatch: plan sha1 {plan_sha[:12] or '(none)'} vs "
                   f"armed {armed_sha[:12] or '(none)'} — the checkpoint "
                   f"changed under an armed mission (panel swap); rejected, "
                   f"no strike.")
            self._log("error", f"ctrl: policy plan {pid} rejected — "
                                       f"{why}")
            self._log_plan(None, Verdict("reject", None, {}, [why]),
                           t_rel, r_now, base_extra)
            self._emit_policy_plan_viz_refused(plan, "reject", why, t_rel,
                                               obs_t_rel)
            return
        if age > float(pc["obs_max_age_s"]) or not math.isfinite(age):
            rp["late"] = int(rp.get("late", 0)) + 1
            if t_now - float(rp.get("t_late_warn", -1e9)) > 5.0:
                rp["t_late_warn"] = t_now
                self._log(
                    "warn", f"ctrl: policy plan late (age {age:.2f} s > "
                            f"{float(pc['obs_max_age_s']):.2f} s) — skipped "
                            f"without a strike ({rp['late']} so far)")
            self._log_plan(None, Verdict(
                "late", None,
                {"obs_age": float(pc["obs_max_age_s"]) - age},
                [f"late: plan conditioned on data {age:.2f} s old at intake "
                 f"(limit {float(pc['obs_max_age_s']):.2f} s) — skipped, no "
                 f"strike."]), t_rel, r_now, base_extra)
            # The DOMINANT refusal on the only hardware run on record (12 of
            # 29), and the one that empties the map for seconds at a time.
            self._emit_policy_plan_viz_refused(
                plan, "late",
                f"late by {age - float(pc['obs_max_age_s']):.2f} s "
                f"(age {age:.2f} s)", t_rel, obs_t_rel)
            return
        anchor, ainfo = self._policy_anchor(obs_t_mono, obs_t_rel)
        # YAW REFERENCE FILTER, part 1 (policy.yaw_ref_filter, 2026-09-14):
        # the plan is COMPOSED against the flown reference yaw at its t0
        # (guard-clamped to the fresh measured yaw), not the measured yaw
        # the anchor rule picked — so body knot 0 stays exactly on the
        # anchor after part 2 re-derives the body knots (safety review
        # 2026-09-14 HIGH: composed on the measured heading, the TCP lever
        # would have put the yaw tracking error into the sway reference as
        # a 0.13 m offset at the guard). Skipped under OBSERVE: nothing is
        # flown there and the operator wants the network's raw heading.
        yf = pc.get("yaw_ref_filter")
        yaw_filter_on = yf is not None and not self.observe
        yaw_ref0 = None
        yaw_ref0_clamped = False
        if yaw_filter_on:
            st_ = self._plan_stitcher
            if st_ is not None and st_.has_plan():
                yaw_ref0, yaw_ref0_clamped = guard_yaw(
                    float(self._replay_ref_now(float(obs_t_rel))[1]),
                    float(self._eta[5]),
                    math.radians(float(yf["guard_deg"])))
                anchor = np.asarray(anchor, float).copy()
                anchor[5] = yaw_ref0
                ainfo["anchor_pose_used"] = [float(v) for v in anchor]
        base_extra.update({
            "anchor_mode": ainfo["mode"],
            "anchor_x_meas_src": ainfo["x_meas_src"],
            "anchor_offset_ned": [float(v) for v in
                                  np.asarray(ainfo["offset_ned"], float)],
            "anchor_offset_m": float(ainfo["offset_m"]),
            "anchor_offset_applied_m": float(ainfo["offset_applied_m"]),
            "anchor_dyaw_deg": math.degrees(float(ainfo["dyaw"])),
            "anchor_clipped": bool(ainfo["clipped_pos"] or ainfo["clipped_yaw"]),
            "anchor_pose_meas": ainfo["anchor_pose_meas"],
            "anchor_pose_used": ainfo["anchor_pose_used"],
            "action_raw": (action.tolist() if action.ndim == 2 else None),
        })
        # ATTITUDE (2026-09-26, D12): a TRACKED mission asks compose_plan
        # for its third branch (track_rp, T1 clip at rp_max_deg); every
        # other mission makes the pre-variant call — a 7-dim plan with
        # attitude_track false is levelled and its roll/pitch LOGGED
        # (dropped_rp_deg).
        compose_kw = {}
        if rp.get("attitude_track"):
            # T2 (rp_reject_deg) is judged at COMPOSE on the raw attitude
            # (2026-09-26): T1 clips to rp_max first, so the filter's own
            # rp_mag margin can never see the excess. Passed ONLY when
            # tracking — the dropped-and-logged / 5-dim calls stay
            # byte-identical.
            compose_kw = {"track_rp": True,
                          "rp_max_rad": math.radians(
                              float(self._policy_att("rp_max_deg"))),
                          "rp_reject_rad": math.radians(
                              float(self._policy_att("rp_reject_deg")))}
        try:
            msg, info = compose_plan(
                action, anchor, anchor[3:5], self._policy_T_bt,
                obs_dt=float(rp["obs_dt_s"]), knot_dt=float(pc["knot_dt_s"]),
                t0=obs_t_rel, plan_id=pid, obs_t_rel=obs_t_rel,
                w_open=float(pc["gripper_width_open_m"]),
                w_closed=float(pc["gripper_width_closed_m"]),
                # The producer's contract, so a width/contract disagreement
                # is a reject_compose with the reason (None = infer).
                action_repr=(str(getattr(plan, "action_repr", "") or "")
                             or None), **compose_kw)
        except TypeError as e:
            if not compose_kw:
                raise                    # the 4-DoF path: untouched
            # This build's compose_plan predates the attitude branch: a
            # tracked mission cannot be flown on it. A reject with the
            # reason (A23) — never a disengage — and the run says why.
            rp["reject_compose"] = int(rp.get("reject_compose", 0)) + 1
            rp["rejected"] = int(rp.get("rejected", 0)) + 1
            why = (f"compose_plan has no attitude branch on this build "
                   f"({e}) — attitude_track cannot be flown; rejected")
            self._log("error", f"ctrl: policy plan {pid} rejected — {why}")
            self._log_plan(None, Verdict("reject", None, {}, [why]),
                           t_rel, r_now, base_extra)
            return
        except ValueError as e:
            # A NaN / degenerate action is the NETWORK's failure, not the
            # loop's: reject with the reason, keep flying (A23).
            rp["reject_compose"] = int(rp.get("reject_compose", 0)) + 1
            rp["rejected"] = int(rp.get("rejected", 0)) + 1
            if "rp_reject" in str(e):
                # the T2 attitude-magnitude reject, raised at compose since
                # 2026-09-26: the same counter the filter's rp_mag margin
                # feeds, so meta policy.run.reject_rp_mag counts every T2
                rp["reject_rp_mag"] = int(rp.get("reject_rp_mag", 0)) + 1
            self._log("error", f"ctrl: policy plan {pid} rejected at "
                                       f"composition — {e}")
            self._log_plan(None, Verdict("reject", None, {}, [str(e)]),
                           t_rel, r_now, base_extra)
            # The ONE intake refusal with nothing to draw: composition is what
            # failed, so _emit_policy_plan_viz_refused would raise the same
            # ValueError on the same action and return silently. The reason
            # reaches plans.jsonl and the log; the map correctly shows no
            # polyline, because there is no geometry to show.
            return
        # YAW REFERENCE FILTER, part 2 (policy.yaw_ref_filter, 2026-09-14):
        # the composed plan's yaw knots (reference yaw + dyaw_k) are replaced
        # by a reference that continues the FLOWN yaw and turns at a
        # low-passed, rate-capped version of the chunk's requested turn
        # rate; the body knots are re-derived from the TCP knots with that
        # yaw. This is what stops the reference from carrying the hull's own
        # stale yaw swing (anchor measured / leash 0) or the policy's
        # alternating dyaw (leash 10) back into an underdamped yaw loop —
        # see the docstring of policy_frames.filter_plan_yaw and the
        # yaw-step test it cites. The turn-rate state lives in
        # rp["yaw_filt"] (reset at arm) and is advanced at INTAKE, i.e. by
        # the policy's intent stream, whether or not a gate later refuses
        # the plan: the flown reference is what the next plan continues
        # from either way. A ValueError here is a reject with the reason,
        # like compose_plan's (A23: never a disengage). Runs before the
        # filter are not comparable on yaw (RECORD BOUNDARY: meta
        # policy.config.yaw_ref_filter, plan record `yaw_ref_filter`).
        if yaw_filter_on:
            yfs = rp["yaw_filt"]
            t0_prev = yfs.get("t0_prev")
            dt_prev = (float(pc["period_s"]) if t0_prev is None
                       else float(obs_t_rel) - float(t0_prev))
            try:
                msg, yinfo = filter_plan_yaw(
                    msg, info["p_tcp"], yaw_ref0, float(self._eta[5]),
                    anchor[3:5], self._policy_T_bt,
                    float(yfs.get("omega", 0.0)), dt_prev,
                    rate=math.radians(float(yf["rate_deg_s"])),
                    tau=float(yf["tau_s"]),
                    guard=math.radians(float(yf["guard_deg"])))
            except ValueError as e:
                rp["reject_compose"] = int(rp.get("reject_compose", 0)) + 1
                rp["rejected"] = int(rp.get("rejected", 0)) + 1
                self._log("error", f"ctrl: policy plan {pid} rejected at the "
                                   f"yaw reference filter — {e}")
                self._log_plan(None, Verdict("reject", None, {},
                                             [f"yaw_ref_filter: {e}"]),
                               t_rel, r_now, base_extra)
                return
            yinfo["yaw_ref0_clamped"] = bool(yaw_ref0_clamped)
            yfs["omega"] = math.radians(float(yinfo["omega_ref_deg_s"]))
            yfs["t0_prev"] = float(obs_t_rel)
            yfs["n"] = int(yfs.get("n", 0)) + 1
            if (yaw_ref0_clamped or yinfo["guard_clamped_start"]
                    or yinfo["guard_clamped_knots"]):
                yfs["guard_clamps"] = int(yfs.get("guard_clamps", 0)) + 1
                if t_now - float(rp.get("t_yaw_guard_warn", -1e9)) > 5.0:
                    rp["t_yaw_guard_warn"] = t_now
                    self._log("warn", f"ctrl: yaw ref filter GUARD — the yaw "
                                      f"reference was {float(yf['guard_deg']):.0f} "
                                      f"deg from the measured yaw and was "
                                      f"clamped (plan {pid}; "
                                      f"{yfs['guard_clamps']} so far)")
            if yinfo["rate_clipped"]:
                yfs["rate_clips"] = int(yfs.get("rate_clips", 0)) + 1
            if yinfo.get("omega_policy_nonfinite"):
                yfs["nonfinite"] = int(yfs.get("nonfinite", 0)) + 1
            base_extra["yaw_ref_filter"] = yinfo
        # TEMPORARY z HOLD (policy.z_hold_above_floor_m, 2026-09-12): replace
        # the composed plan's z — the policy's own dz, which the handheld
        # demos learned as descend-to-grasp — with one fixed height above the
        # tag floor. Map z is NED-like (+down), so a height h above the floor
        # is map z = -h; the datum shift is d["p0"][2] (Rz is yaw-only, so z
        # is a pure offset). Written into every plan record so the boundary
        # is visible in plans.jsonl, not only in the meta.
        zh = pc.get("z_hold_above_floor_m")
        if zh is not None:
            z_map = -float(zh)
            z_ned = z_map - (float(self._datum["p0"][2]) if self._datum else 0.0)
            msg.p_ned[2, :] = z_ned
            base_extra["z_hold_applied"] = {"above_floor_m": float(zh),
                                            "z_map": z_map, "z_ned_datum": z_ned}
        base_extra.update({
            "action_repr": str(info["action_repr"]),
            "dropped_rp_deg": float(info["dropped_rp_deg"]),
            "jitter_rms_mm": float(info["jitter_rms_mm"]),
            "n_raw": int(info["n_raw"]), "n_knots": int(info["n_knots"]),
            "p_tcp": np.asarray(info["p_tcp"], float).tolist(),
            "anchor_tcp": [float(v) for v in info["anchor_tcp"][0]]
            + [float(info["anchor_tcp"][1])],
            # ALWAYS (schema 16): tracked or levelled.
            "rp_tracked": bool(info.get("rp_tracked",
                                        getattr(msg, "rp", None) is not None)),
        })
        # The attitude record of a 7-dim plan, when compose_plan produced
        # one: decoded absolute roll/pitch per raw knot, the resampled
        # (clipped) knots that fly, and how much the T1 clip took.
        for k in ("rp_raw", "rp"):
            v = info.get(k)
            if v is not None:
                base_extra[k] = np.asarray(v, float).tolist()
        for k in ("rp_clipped_deg", "rp_clipped_n"):
            if info.get(k) is not None:
                base_extra[k] = (int(info[k]) if k.endswith("_n")
                                 else float(info[k]))
        for k in ("anchor_rp_meas", "anchor_rp_used", "anchor_rp_ref"):
            if ainfo.get(k) is not None:
                base_extra[k] = ainfo[k]
        rp["extras"][int(msg.plan_id)] = base_extra
        rp["action_repr"] = str(info["action_repr"])
        rp["pending"].append(msg)
        rp["n_plans"] = int(rp.get("n_plans", 0)) + 1
        rp["infer_ms"].append(float(base_extra["infer_ms"]))
        rp["last_intake"] = {"plan_id": pid, "obs_t_rel": obs_t_rel,
                             "age_s": age, "anchor": ainfo,
                             "msg": msg, "info": info}

    def _emit_policy_state(self, meas, health) -> None:
        """PolicyState every tick while a policy worker exists (v2 A6): the
        tick's eta with the FIX stamp behind it (the worker keys its proprio
        history on t_fix advancing, never on the tick), whether that eta is a
        fresh tag solution, the epoch, and whether a policy mission is
        actually running (active = armed AND traj_on AND not halted — the
        worker only infers then, so a halted or stopped mission idles the
        GPU). Also feeds this worker's own fix-keyed history (A4's anchor)."""
        t = now()
        fix = self.fix
        fix_ok = bool(fix is not None and fix.ok)
        t_fix = float(fix.t_capture) if fix_ok else None
        flying_dr = bool(self.dr_control and self._dr_last is not None
                         and self._dr_last.get("ok"))
        bridged = bool(self._bridge is not None
                       and self._bridge.tier != SB.TIER_NONE)
        fix_fresh = bool(meas is not None and fix_ok and not bridged
                         and not flying_dr)
        eta_src = meas["eta"] if meas is not None else self._eta
        eta = (tuple(float(v) for v in eta_src) if eta_src is not None
               else (float("nan"),) * 6)
        if (fix_fresh and t_fix is not None and self.engaged
                and t_fix != self._policy_hist_last_fix
                and self._policy_eta_hist is not None):
            # Datum-frame rows only (engaged): before ENGAGE the eta is in
            # the tag frame and the history is cleared at engage anyway.
            if self._policy_eta_hist.append(t_fix, eta):
                self._policy_hist_last_fix = t_fix
        rp = self.replay
        kind_policy = bool(rp is not None and rp.get("kind") == "policy")
        halted = bool(kind_policy and rp.get("halted"))
        active = bool(kind_policy and self.traj_on and not halted)
        self.bus.policy_state.emit(PolicyState(
            t=t, t_fix=t_fix, eta=eta, fix_fresh=fix_fresh,
            epoch=int(self._policy_epoch), active=active, halted=halted,
            t0_traj=(self._t0_traj if kind_policy else None),
            eta_start=(self._policy_eta_start if kind_policy else None),
            grip_width_m=(float(self._grip_est.width(t))
                          if self._grip_est is not None else float("nan")),
            engaged=bool(self.engaged), stamp=t))

    def _policy_meta(self) -> dict:
        """ALWAYS written (the imu_dr rule): the policy block's config with
        its provenance, the TCP geometry actually used (v2 A3), the jaw
        estimator, the worker's own meta when injected, and the run's
        counters + end_reason (v2 A22). ``synthetic`` is True for a demo
        plant OR a stub session (v2 A15) — a reader must not quote such a
        run as a measurement."""
        if self.cfg is None:
            return {"enabled": False}
        pc = self.cfg.policy
        src = str(getattr(self.opts, "source", "hw") or "hw")
        worker = None
        if self.policy_meta_fn is not None:
            try:
                worker = dict(self.policy_meta_fn())
            except Exception as e:                               # noqa: BLE001
                worker = {"error": f"{type(e).__name__}: {e}"}
        stub = bool(worker and (worker.get("stub")
                                or (worker.get("session") or {}).get("stub")))
        rp = self.replay if self.replay is not None else self._replay_last
        if rp is not None and rp.get("kind") != "policy":
            rp = None
        st = self._policy_status
        est = self._grip_est
        t = now()
        out = {
            "enabled": bool(self.policy_present),
            "why": ("" if self.policy_present else
                    ("no --policy" if not getattr(self.opts, "policy", False)
                     else "not built")),
            "source": src,
            # POLICY OBSERVE counts as SYNTHETIC for the same reason a demo
            # plant does, and by the same precedent backends/policy.py uses
            # for a land dry-run: readers already treat `synthetic` as
            # do-not-cite, and an observe run's numbers describe a loop that
            # was never closed.
            "synthetic": bool(src != "hw" or stub or self.observe),
            "observe": bool(self.observe),
            # WHICH GUARDS WERE ACTUALLY ARMED, as booleans, beside the config
            # values they came from. `config` prints hw_mpc.yaml verbatim, so
            # a runtime bypass is otherwise invisible — the existing
            # land_dry_run divergence bypass has exactly that defect today
            # (meta keeps printing `div_max_m: 0.25` for a run where the
            # guard could not fire). An observe run disarms five of these on
            # purpose; this is where a reader finds that out.
            "gates_enforced": {
                "divergence": not (self.observe or self.land_dry_run),
                "escalation_latch": not self.observe,
                "bridge_stop": not self.observe,
                "workspace_box": not self.observe,
                "tag_loss_disengage": not self.observe,
                "vehicle_arm_mode_gates": not (self.observe
                                               or self.land_dry_run),
                "max_run_key": ("observe_max_run_s" if self.observe
                                else "max_run_s"),
            },
            "config": {k: pc[k] for k in sorted(pc)},
            "config_provenance": dict(POLICY_PROVENANCE),
            # WHICH NETWORK (2026-09-11, the panel picker): `config.ckpt` is
            # only what loaded at launch; the worker's status says what it
            # HOLDS, and the armed mission pinned that (scen["ckpt"] /
            # ["ckpt_sha1"]). `ckpt_source` is the worker's own word
            # ("launch" | "panel") when its meta is injected.
            "ckpt_loaded": str(getattr(st, "ckpt", "") or ""),
            "ckpt_config": str(pc["ckpt"]),
            "ckpt_source": ((worker or {}).get("ckpt_source")
                            if isinstance(worker, dict) else None),
            # The note must stay TRUE across a REC-open swap (review
            # 2026-09-11): ckpt_loaded is the status at WRITE time, and a
            # panel pick between engagements can change it under an open
            # REC CSV. What inferred is the ARM pin.
            "config_note": ("trajectory.ckpt / ckpt_sha1 (pinned at ARM) is "
                            "what inferred; ckpt_loaded is what the worker "
                            "held when this meta was written; config.ckpt is "
                            "the launch seed"),
            "ckpt_changes_while_csv_open": [dict(c) for c in self._ckpt_changes],
            "down_sample_steps": int(POLICY_DOWN_SAMPLE_STEPS),
            "obs_dt_s": POLICY_DOWN_SAMPLE_STEPS / float(pc["dataset_fps"]),
            # The action representation THIS side flies (2026-09-07): since
            # 2026-09-26 the config's policy.action_repr (default
            # POLICY_ACTION_REPR), pinned per mission in trajectory /
            # run.action_repr_pinned; the checkpoint's own is under
            # status.action_repr / worker.session.
            "action_repr": self._policy_pinned_repr(),
            "action_repr_default": POLICY_ACTION_REPR,
            "action_reprs_flyable": list(POLICY_ACTION_REPRS_FLYABLE),
            # The attitude knobs with their defaults when the config predates
            # them (`config` above prints the file verbatim).
            "attitude": {k: self._policy_att(k)
                         for k in sorted(self.POLICY_ATTITUDE_DEFAULTS)},
            "anchor_mode": str(pc["anchor"]),
            "hold_tail": str(pc["hold_tail"]),
            "tcp": {
                "tcp_body_flu_m": list(pc["tcp_body_flu_m"]),
                "tcp_offset_cam_m_override": pc.get("tcp_offset_cam_m"),
                "tcp_offset_cam_m_used": (
                    [float(v) for v in self._policy_tcp_off_cam]
                    if self._policy_tcp_off_cam is not None else None),
                "handheld_tcp_offset_cam_m": list(HANDHELD_TCP_OFFSET_CAM_M),
                "T_body_tcp": (self._policy_T_bt.tolist()
                               if self._policy_T_bt is not None else None),
                "t_body_tcp_norm_m": (
                    float(np.linalg.norm(self._policy_T_bt[:3, 3]))
                    if self._policy_T_bt is not None else None),
                "note": ("TCP = the ROV's own jaw, anchored to the C3 lens: "
                         "cam_t_flu + [0.196, 0, -0.275] (x measured "
                         "2026-09-08 from data/20260908/"
                         "0908_180453 — a jar gripped in the jaws set on tag "
                         "58's centre; z = CAD vertical [유도]; y = 0 [예측]; "
                         "cam_t_flu x/z attribution open, KNOWN_ISSUES "
                         "2026-09-08). Handheld value is a comparison only "
                         "(v2 A3); the jaw-vs-handheld mismatch is NOT "
                         "compensated (hw_mpc.yaml policy.tcp_body_flu_m)."),
            },
            "estimator": (None if est is None else {
                "w_open_m": est.w_open, "w_closed_m": est.w_closed,
                "travel_s": est.travel_s,
                "w_init_m": float(pc["gripper_width_init_m"]),
                # WHEN on the plan the hysteresis sampled the channel
                # (schema 13): now + this; 0 = at now (pre-13).
                "lookahead_s": float(pc["gripper_lookahead_s"]),
                "width_now_m": float(est.width(t)),
                "n_drives": int(est.n_drives),
                "assumption": ("jaw OPEN (w_init) at process start — there is "
                               "no jaw feedback; the width is what the "
                               "policy was TOLD, not what the jaw did"),
            }),
            "epoch": int(self._policy_epoch),
            "drops_outside_mission": dict(self._policy_counts),
            "depth_src": (str(getattr(st, "depth_src", "")) if st else ""),
            "status": (None if st is None else {
                "ready": bool(getattr(st, "ready", False)),
                "loading": bool(getattr(st, "loading", False)),
                "error": str(getattr(st, "error", "")),
                "hz": _num_or_none(getattr(st, "hz", None)),
                "infer_ms": _num_or_none(getattr(st, "infer_ms", None)),
                "note": str(getattr(st, "note", "")),
                "n_plans": _num_or_none(getattr(st, "n_plans", None)),
                "n_skip": _num_or_none(getattr(st, "n_skip", None)),
                "grid_ok": bool(getattr(st, "grid_ok", False)),
                "grid_why": str(getattr(st, "grid_why", "") or ""),
                "obs_dt_s": _num_or_none(getattr(st, "obs_dt_s", None)),
                "action_repr": str(getattr(st, "action_repr", "") or ""),
                "mount_ok": bool(getattr(st, "mount_ok", True)),
                "mount_why": str(getattr(st, "mount_why", "") or ""),
                "age_s": (t - float(getattr(st, "stamp", t)))}),
            "worker": worker,
            "run": None,
        }
        if rp is not None:
            ms = [float(v) for v in list(rp.get("infer_ms") or [])
                  if math.isfinite(float(v))]
            out["run"] = {
                k: rp.get(k) for k in
                ("epoch", "n_plans", "received", "released", "installed",
                 "clipped", "rejected", "late", "skip_bridge", "skip_halted",
                 "drop_epoch", "drop_old", "reject_compose", "reject_blend",
                 "reject_clock", "reject_obs_dt", "reject_action_repr",
                 "reject_ckpt", "ckpt_sha1",
                 "action_repr",
                 # the 6-DoF variant (schema 16): the pin, the tracked
                 # flag and the attitude gate / divergence counters
                 "action_repr_pinned", "attitude_track",
                 "reject_rp_mag", "clip_rp_rate", "div_rp_max_seen_deg",
                 "stale_events", "halted", "end_reason", "period_s",
                 "duration_s", "obs_dt_s", "anchor_mode", "hold_tail",
                 "gripper_on", "grip_events", "grip_w_at_arm",
                 "grip_lookahead_s",
                 "t_last_accept",
                 # OBSERVE's own three. `observe` says which kind of run
                 # these counters describe; `escalations` is what the latch
                 # WOULD have fired on (the latch is disarmed, so the count
                 # is the finding); `div_max_seen_m` is how far the pilot
                 # actually got from the reference, which is the number the
                 # disarmed divergence guard would otherwise have taken to
                 # the grave.
                 "observe", "escalations", "div_max_seen_m",
                 # hold_tail: extrapolate — ticks whose horizon tail was
                 # continued past the plan (0 under mask/track).
                 "extrap_ticks")}
            out["run"]["hold_frac_last"] = (
                None if not math.isfinite(float(rp.get("hold_frac", float("nan"))))
                else float(rp["hold_frac"]))
            out["run"]["infer_ms_p50"] = (float(np.percentile(ms, 50))
                                          if ms else None)
            out["run"]["infer_ms_max"] = float(max(ms)) if ms else None
            out["run"]["live"] = bool(self.replay is not None)
            # YAW REFERENCE FILTER (2026-09-14): what it did this run — how
            # many plans it filtered, the turn-rate state it ended on, and
            # how often the guard / rate cap bound. None = filter off.
            yfs = rp.get("yaw_filt")
            out["run"]["yaw_ref_filter"] = (
                None if not yfs or pc.get("yaw_ref_filter") is None else
                {"skipped": "observe"} if rp.get("observe") else
                {"plans": int(yfs.get("n", 0)),
                 "omega_last_deg_s": math.degrees(float(yfs.get("omega", 0.0))),
                 "guard_clamps": int(yfs.get("guard_clamps", 0)),
                 "rate_clips": int(yfs.get("rate_clips", 0)),
                 "nonfinite": int(yfs.get("nonfinite", 0))})
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
            self._log(
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
            self._log(
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
        if shape == "policy":
            self.station = None
            self._approach = None
            self._path_cursor = None
            self._path_err = (None, None)
            self._arm_policy(sq)
            return
        if shape == "follow":
            self.station = None
            self._approach = None      # nothing to approach: see set_traj
            self._path_cursor = None
            self._path_err = (None, None)
            self._arm_follow(sq)
            return
        if shape == "station":
            if sq.get("heading_tag") and self._yaw_hold:
                # A heading the loop cannot fly is a promise the meta would
                # then record as kept (KNOWN_ISSUES "STABILIZE에서 engage ...
                # yaw는 hold"): in STABILIZE with yaw_axis: hold the yaw axis
                # goes out as 0 and the autopilot's heading hold owns heading,
                # so "facing tag N" would be written down and never happen.
                self._refuse(
                    f"facing tag {sq['heading_tag']} needs the yaw axis, but "
                    f"the vehicle engaged in {self._flight_mode_at_engage} with "
                    f"engage.stabilize.yaw_axis: hold — the autopilot's heading "
                    f"hold owns heading. Engage in MANUAL (or clear facing tag).")
                return
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
                               if sq.get("origin_tag") else None),
                # The tag the heading was derived from (None = yaw_map_deg /
                # yaw_fixed_deg), so a run's meta says WHY it faced that way.
                "heading_tag": (int(sq["heading_tag"])
                                if sq.get("heading_tag") else None)}
            self.traj_on = False
            self.phase, self.phase_detail = "station", ""
            what_short = ("tag %d" % self.station["origin_tag"]
                          if self.station["origin_tag"] else "this point")
            ht = self.station["heading_tag"]
            what = (f"STATION hold at "
                    + (f"tag {self.station['origin_tag']}"
                       if self.station["origin_tag"] else "this point")
                    + (f", facing tag {ht} = " if ht else ", heading ")
                    + f"{yaw_map:+.0f} deg (map)")
            self.reason = "station hold"
            self._event(f"on station: {what_short}")
            self._log_event(f"STATION - {what}")
            self._log("warn", f"ctrl: {what}")
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
                self._log(
                    "warn",
                    f"ctrl: curvature caps this path at {v_cap:.3f} m/s — the "
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
        self._log("warn", f"ctrl: {what}")

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
            p_tag = self._tag_map_xy(tag, "tag")
            if p_tag is None:
                return None
            eta_tag = np.zeros(6)
            eta_tag[0:2] = p_tag
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
            except Exception as e:                               # noqa: BLE001
                # yaml.YAMLError / KeyError from a corrupt map are not
                # ValueErrors; escaping from set_traj inside tick() would
                # cost the rest of that tick (safety review 2026-09-11).
                self._log("error", f"ctrl: tag map unreadable ({e})")
        return self._tagmap

    def disengage(self, why: str) -> None:
        was = self.engaged
        self.engaged = False
        self.traj_on = False
        # policy.q_scale must not survive E-STOP / fault / ENGAGE-off /
        # teardown into the next engagement (safety review 2026-09-07):
        # disengage does not pass through set_traj(False).
        self._apply_plan_cost_scale(1.0)
        self._apply_plan_path_cost(None)
        if self._bridge is not None:
            self._bridge.reset()
        self._bridge_anchor = None
        self._axes_prev = None          # never ramp from a stale command
        self._yaw_hold = False          # the next engagement decides again
        if self._attitude_axes or self._attitude_at_engage:
            # The RECORD keeps enabled/counters (the closing meta is written
            # after this); the LIVE bit drops so no K/M can leave again
            # until the next engage's gates pass. The follower is told too.
            self._attitude_at_engage["sat_ticks"] = int(self._attitude_sat_ticks)
            if self._att_degraded_at is not None:
                self._attitude_at_engage["degraded_at"] = self._att_degraded_at
        self._attitude_axes = False
        # NOT here: `self._bridge.attitude_axes = False` — StationBridge.meta()
        # derives `released_in_coast` from that bit AT CALL TIME, and the
        # closing meta is written below (_close_csv); dropping it first made
        # a variant run close saying K/M were kept in coast (safety audit
        # 2026-09-26, W1). It drops with the follower's bit, after the meta.
        self._reset_attitude_interlocks()
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
        self._clear_replay(end_reason="disengaged")
        if self._grip_est is not None:
            # The sink is zeroed with the disengage (neutral pilot input,
            # deadman); the jaw estimator stops integrating with it (A12).
            self._grip_est.release(now())
        self.phase, self.phase_detail = "", ""
        try:
            if was:
                if not self.observe:
                    # One explicit neutral so the vehicle stops NOW; the
                    # window then resumes the teleop pump, and the sink
                    # deadman covers the gap.
                    #
                    # NOT under POLICY OBSERVE, and the comment above says
                    # why: the premise is that the window's pump was
                    # YIELDING and needs the gap covered. In observe it never
                    # yielded — the pilot has been flying the whole time — so
                    # this frame is not a stop, it is a SECOND command source
                    # writing a neutral MANUAL_CONTROL into the middle of
                    # someone's manoeuvre (`bus.cmd_pilot` -> the sink's
                    # `set_pilot`, last writer wins). There is also nothing
                    # to neutralise: this worker has not commanded an axis
                    # all run.
                    self.bus.cmd_pilot.emit(PilotInput(source="mpc"))
                self.reason = f"disengaged: {why}"
                # `debug` for the same reason the ENGAGE copy is: the mission
                # event on the next line is the one the operator reads, and
                # two lines for one disengage is what made the log hard to
                # scan.
                self._log("debug", f"ctrl: DISENGAGED — {why}")
                self._event(f"DISENGAGED: {why}")
                self._log_event(f"DISENGAGED - {why}")
                if self._csv_auto:
                    self._close_csv()
        finally:
            # The follower and the bridge are told LAST (integration seam,
            # 2026-09-26): the follower's meta() is what the closing CSV
            # meta's `controller` block copies (`allocation.attitude`,
            # `attitude_ref.u_max_wire_nm`) and the bridge's meta() derives
            # `released_in_coast` from its bit, and a run that flew with K/M
            # must not close saying it did not. The LIVE bit above already
            # dropped, so no K/M can leave in between. In a `finally` so a
            # closing meta that RAISES cannot leave the follower believing
            # its K/M still reach the wire (safety audit 2026-09-26); the
            # CSV-then-reset order holds on the normal path.
            if self._bridge is not None:
                self._bridge.attitude_axes = False
            set_att = getattr(self.ctrl, "set_attitude_axes", None)
            if callable(set_att) and self._attitude_at_engage.get("enabled"):
                try:
                    set_att(False)
                except Exception as e:                           # noqa: BLE001
                    self._log("warn", f"ctrl: set_attitude_axes(False) raised "
                                      f"{type(e).__name__}: {e}")

    @Slot()
    def estop(self) -> None:
        fm = str(getattr(self, "_flight_mode_at_engage", None) or "")
        if self.engaged and fm and not fm.upper().startswith("MANUAL"):
            # Neutral is not a stop in a self-driving mode: the autopilot
            # keeps levelling and holding heading with the thrusters live
            # and the position drifts. The stop is DISARM (or MANUAL).
            self._log("error", f"ctrl: E-STOP sends NEUTRAL, which does NOT "
                               f"stop the vehicle in {fm} — the autopilot "
                               f"keeps the thrusters live; press DISARM (or "
                               f"MANUAL) to stop it")
        self.disengage("E-STOP")
        if self._grip_est is not None:
            self._grip_est.release(now())

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
                if not self.observe:
                    # Under OBSERVE `_runtime_fault` has already written the
                    # note that says what this costs — "NO FIX — plans
                    # skipped" — and it is the more useful of the two: an
                    # observe run does not disengage on a dropout, so the
                    # operator's question is not "is it about to stop" but
                    # "why has the map gone quiet". Overwriting it here made
                    # that note dead on every tick it was set.
                    self.phase_detail = (f"waiting for a fix "
                                         f"({self._stale_ticks})")
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
                if self.observe:
                    # POLICY OBSERVE ENDS THE TICK HERE. Everything above ran
                    # — the state assembler, the path clock, the policy
                    # intake, the filter, the stitcher and
                    # `ctrl.set_path_plan_ned` — so the reference the panel
                    # draws is the real composed plan. Everything below is
                    # actuation, and there is none.
                    #
                    # `ctrl.step()` is SKIPPED rather than called-and-
                    # discarded, and that is the safer of the two:
                    #   * `note_applied` (:4556) would otherwise have to be
                    #     told a wrench was applied that never left the
                    #     station, and the EAOB would book the pilot's entire
                    #     manual flight as disturbance and rail w_hat;
                    #   * `_axes_prev` (:4550) is the slew memory a later real
                    #     engagement ramps from, and it must not fill with
                    #     commands nobody sent;
                    #   * the solver-fail and tick-overrun ladders below both
                    #     end in `disengage`, i.e. an NMPC handed a reference
                    #     the pilot is deliberately 2 m away from would keep
                    #     ending the very run whose purpose is to watch it.
                    #
                    # The sentinels are NOT zeros. A CSV row of exactly 0.000
                    # wrench with solver_status 0 is this repo's recurring
                    # silent failure ("the solver said fine and nothing
                    # moved"); status -1 cannot come from acados or IPOPT and
                    # `nan` says "not computed" where 0 would say "commanded
                    # zero". CSV_HEADER's 2026-09-03 note carries the meaning.
                    u = np.full(6, np.nan)
                    info = {"status": OBSERVE_SOLVER_STATUS, "solve_ms": None,
                            "w_hat": np.full(6, np.nan)}
                    axes = None
                    self._tick_ms = float("nan")
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
                    if self._attitude_axes:
                        # COAST releases X, Y, N — and, with the attitude
                        # axes live, K and M too (2026-09-26): a coast holds
                        # depth and attitude on the IMU alone, and a torque
                        # against an attitude estimate nobody trusts is the
                        # same mistake as a surge against a frozen xy. The
                        # bridge itself is untouched; meta
                        # station_bridge.released_in_coast is the bridge's
                        # own sentence, run.attitude_axes says K/M were
                        # zeroed here on coast rows (rp_track stays 1: the
                        # frame still carried the axes, at 0).
                        u = np.asarray(u, float).copy()
                        u[3:5] = 0.0
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
                    self._log(
                        "error", f"ctrl: solver FAILURE status "
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
                    # dropping depth hold on a vehicle that is never neutral. Demote first:
                    # `_follow_to_station` re-targets with no feedforward, i.e.
                    # it removes the reference that caused this.
                    self._follow_to_station(
                        f"tick {self._tick_ms:.0f} ms > budget")
                elif self._over_streak >= 2:
                    self.disengage(f"tick {self._tick_ms:.0f} ms > budget "
                                   f"({self._over_streak}x)")
                else:
                    # A controller may carry its own slew (HwRl: 0 — trained without one); else the config's.
                    _slew = getattr(self.ctrl, "axis_slew_per_s", None)
                    _rate = (self.cfg.axis_slew_per_s if _slew is None
                             else float(_slew))
                    if self._attitude_axes:
                        # THE 6-DoF VARIANT (2026-09-26): K/M become the
                        # roll/pitch axes, clipped to the wire caps
                        # (attitude_cap, D4) — the solver's U_MAX[3:5] may
                        # plan more torque than the wire carries; the EAOB
                        # is credited the CAPPED value through
                        # axes_to_wrench below, so the mismatch is seen, not
                        # hidden. The roll/pitch slew is the block's own
                        # slew_per_s and `_axes_prev` widens to a 6-tuple,
                        # which is what makes slew_axes slew (not zero) them.
                        att = self._attitude_cfg()
                        axes = wrench_to_axes(u, self.cfg.axis_gain,
                                              self._axis_cap(), attitude=True,
                                              attitude_cap=self._attitude_cap)
                        if self._yaw_hold:
                            # Belt and braces: the variant is refused outside
                            # MANUAL, so this line never runs with K/M live.
                            axes = dataclasses.replace(axes, yaw=0.0,
                                                       roll=0.0, pitch=0.0)
                        axes = slew_axes(axes, self._axes_prev, _rate,
                                         1.0 / self.cfg.ctrl_hz,
                                         rate_rp=float(att.get("slew_per_s",
                                                                1.5)))
                        self._axes_prev = (axes.surge, axes.sway, axes.heave,
                                           axes.yaw, axes.roll, axes.pitch)
                    else:
                        # THE 4-DoF PATH, byte-identical to every run before
                        # the variant: the pre-variant call, the 4-tuple.
                        axes = wrench_to_axes(u, self.cfg.axis_gain,
                                              self._axis_cap())
                        if self._yaw_hold:
                            # STABILIZE + yaw_axis hold: the autopilot's heading
                            # hold owns heading. CSV ax_yaw reads 0 while uN keeps
                            # the solver's value; axes_to_wrench below then tells
                            # the controller N = 0 was realised.
                            axes = dataclasses.replace(axes, yaw=0.0)
                        axes = slew_axes(axes, self._axes_prev, _rate,
                                         1.0 / self.cfg.ctrl_hz)
                        self._axes_prev = (axes.surge, axes.sway, axes.heave,
                                           axes.yaw)
                    # The EAOB is told what ACTUALLY went out — the slew limit
                    # is a real actuator limit like the cap, so crediting the
                    # unlimited wrench would book the difference as
                    # disturbance (axes_to_wrench's docstring). Under the
                    # variant that includes the capped/slewed K/M.
                    self.ctrl.note_applied(axes_to_wrench(axes,
                                                          self.cfg.axis_gain))
                    self.bus.cmd_pilot.emit(axes)
                    self._watch_actuation(axes, t)
                    if self._attitude_axes:
                        # Interlock (iii), D9 — AFTER the emit: the frame that
                        # just left is the one being judged, and a disengage
                        # here sends its own neutral.
                        why_att = self._attitude_sat_watch(meas, axes, t_traj, t)
                        if why_att:
                            self._event("ATTITUDE torque ineffective — "
                                        "disengaged")
                            self._log("error", f"ctrl: {why_att}")
                            self.disengage(why_att)
                            self._publish(meas, health, u, info, axes, t_traj)
                            self._object_heartbeat(t)
                            self._write_row(meas, health, u, info, axes,
                                            t_traj, t)
                            return
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
                        self._log("warn", f"ctrl: {kind} COMPLETE — "
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
        if self.observe:
            # Its premise is the first words of its own error line: "commands
            # are leaving". Under OBSERVE none are, so every firing would be a
            # false alarm sending the operator after a SYSID/arming fault that
            # does not exist — during the one run whose purpose is to diagnose
            # something else. Unreachable today (the tick returns before the
            # call site), guarded here so a future re-order cannot resurrect
            # it, and the streak is reset so a later real engagement starts
            # from zero.
            self._cmd_active_ticks = 0
            return
        strong = max(abs(axes.surge), abs(axes.sway), abs(axes.heave),
                     abs(axes.yaw), abs(getattr(axes, "roll", 0.0)),
                     abs(getattr(axes, "pitch", 0.0))) > 0.12
        self._cmd_active_ticks = self._cmd_active_ticks + 1 if strong else 0
        if self._cmd_active_ticks < int(2.0 * self.cfg.ctrl_hz):
            return
        dev = self._pwm_dev_us()
        if dev is not None and dev < 8.0 and t - self._last_noresp_warn > 10.0:
            self._last_noresp_warn = t
            msg = (f"ctrl: commands are leaving (axes ~"
                   f"{max(abs(axes.surge), abs(axes.sway)):.2f}) but every "
                   f"thruster sits at neutral (PWM dev {dev:.0f} us) — the "
                   f"vehicle is NOT acting on MANUAL_CONTROL. Check: armed? "
                   f"SYSID_MYGCS? vehicle pilot gain? QGC also connected?")
            self._log("error", msg)
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
            if self.observe:
                # LOSING THE TAG DOES NOT END AN OBSERVE RUN. There is no
                # control to protect — the debounce above exists so a stale
                # state never reaches a solver, and no solver runs here. What
                # a disengage WOULD cost is the whole session: it drops the
                # mission, and the next engagement takes a NEW datum, which
                # makes the window clear the plot (window.py `_on_mpc_status`
                # -> `view.clear()`), so every plan drawn so far disappears.
                # The pilot flying behind the bottle for a second would reset
                # the experiment.
                #
                # The tick still takes the `meas is None` branch, so this
                # tick composes nothing: no intake, no anchor, no plan. That
                # is the honest behaviour — a plan anchored to a stale pose
                # would draw a confident line in the wrong place.
                self.phase_detail = "NO FIX — plans skipped (observe)"
                return ""
            return why
        self._stale_ticks = 0
        # The runtime twins of the three engage gates, skipped for the same
        # reason (_engage_refusal): a LAND DRY-RUN has no command sink, so
        # "disarmed" is the expected and desired state, not a fault. Without
        # this the mission would engage and then disengage on the next tick.
        # POLICY OBSERVE skips them because it computes no wrench; the pilot
        # may arm, disarm and change flight mode freely — that is their half
        # of the run, and none of it is this worker's business.
        if not self.land_dry_run and not self.observe:
            tel = self.tel
            if tel is None or tel.armed is not True:
                return "vehicle disarmed"
            if t - float(tel.stamp) > 2.0:
                return "vehicle telemetry stale"
            # PINNED to the mode it engaged in: the yaw handling (and the
            # record boundary) were chosen at engage, so a change into any
            # other mode — listed or not — ends the engagement.
            pinned = getattr(self, "_flight_mode_at_engage", None)
            mode = (tel.mode or "").upper()
            if pinned and pinned != "?" and not mode.startswith(pinned.upper()):
                return f"flight mode left {pinned} (now {tel.mode or '?'})"
        if self._attitude_axes:
            # THE 6-DoF VARIANT's runtime twins (D9). (v) the sink degraded
            # to the 4-axis frame mid-run: a run that silently lost its
            # attitude actuation must not continue as a 6-DoF record.
            tel = self.tel
            if tel is not None and bool(getattr(tel, "attitude_axes_degraded",
                                                False)):
                if self._att_degraded_at is None:
                    self._att_degraded_at = time.strftime("%Y-%m-%d %H:%M:%S")
                return ("attitude axes DEGRADED — the command sink fell back "
                        "to the 4-axis MANUAL_CONTROL frame (K/M no longer "
                        "reach the vehicle)")
            # (ii) the attitude CEILING: |roll| or |pitch| past abort_deg for
            # >= 0.25 s -> neutral (K/M = 0, passive righting). Measured
            # state, debounced like every other interlock here.
            a = self._attitude_cfg()
            lim = math.radians(float(a.get("abort_deg", 35.0)))
            eta = np.asarray(meas["eta"], float)
            tilt = max(abs(float(eta[3])), abs(float(eta[4])))
            if tilt > lim:
                self._att_ceiling_ticks += 1
                need = max(1, int(round(0.25 * self.cfg.ctrl_hz)))
                if self._att_ceiling_ticks >= need:
                    return (f"attitude ceiling: |roll/pitch| "
                            f"{math.degrees(tilt):.1f} deg > "
                            f"{float(a.get('abort_deg', 35.0)):.0f} deg for "
                            f"0.25 s — neutral, passive righting")
            else:
                self._att_ceiling_ticks = 0
        # No position abort. Until 2026-08-14 leaving the geofence box (plus a
        # margin) disengaged the controller here; the fence was removed at the
        # operator's request, so the runtime faults are now exactly: disarm,
        # telemetry stale, flight mode left MANUAL, and tag-fix loss above.
        return ""

    # ------------------------------------------------------------- reporting
    def _publish(self, meas, health, u, info, axes, t_traj) -> None:
        if self.policy_present:
            # Every tick, both tick exits (v2 A6) — cheap, and the worker's
            # proprio history depends on never missing a fix stamp.
            self._emit_policy_state(meas, health)
        s = MpcStatus(
            engaged=self.engaged,
            # THE FIELD EVERY ACTUATION-AWARE CONSUMER READS. `engaged` says
            # the machinery is live; this says a wrench is going out. They
            # differ only under POLICY OBSERVE, and that difference is the
            # whole mode: the window leaves the joystick with the pilot
            # because this is False.
            commanding=self.commanding,
            observe=bool(self.observe),
            traj_on=self.traj_on,
            mode=self.cfg.mode if self.cfg else "",
            # The panel disables the LOW combo on it (set_mode refuses too).
            csv_open=(self._csv is not None),
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
            # "none" under LOW None: the holder's solver_kind ("pid") would
            # claim a solver that is never stepped (the solver_status
            # sentinel rule, applied to the name too).
            solver=("none" if self.observe
                    else (self.ctrl.solver_kind if self.ctrl else "")),
            # The `_write_row` sentinel rule, for the panel: an observe tick
            # never ran a solver, so it must not publish one's verdict.
            solver_status=(OBSERVE_SOLVER_STATUS if self.observe
                           else int(info.get("status", 0))),
            solve_ms=(None if self.observe else info.get("solve_ms")),
            n_fail=int(info.get("n_fail", 0)),
            w_hat=(() if self.observe
                   else tuple(float(v) for v in info.get("w_hat", ()))),
            u_cmd=tuple(float(v) for v in u) if self.commanding else (),
            axes=((axes.surge, axes.sway, axes.heave, axes.yaw)
                  if axes is not None else ()),
            # (roll, pitch) SENT under the variant; () on the 4-DoF path so
            # the panel's attitude bars stay blank exactly as before.
            axes_rp=((float(getattr(axes, "roll", 0.0)),
                      float(getattr(axes, "pitch", 0.0)))
                     if (axes is not None and self._attitude_axes) else ()),
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
            # The REFERENCE stays, observe included — it is the DP's proposal
            # and drawing it is the entire point of the mode.
            s.ref_flu = (rx, ry, rz)
            if meas is not None and not self.observe:
                # THE ERRORS DO NOT. err_xy / err_along / err_cross are
                # TRACKING errors: "how far is the follower behind its
                # reference". With the loop open there is no follower, and
                # the number they would carry is how far the PILOT chose to
                # fly from a line nobody was following — which the panel
                # prints as "off 190 cm · lag 40 cm" and a later reader
                # quotes as a tracking result. Left None so the chip and the
                # CSV both say "not measured" rather than something false.
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
        """Rides the shared recording stem, like the other sensor logs.

        EXCEPT under a mode that owns its own run TREE. The stem comes from
        the depth recorder, which opened in `--rec-dir` (the water tree), and
        pinning it here would drag the whole controller record — CSV, meta,
        plans.jsonl, policy_plan.csv, events.log — out of
        data/*/*_observe/ and into the water tree, silently, just
        because the operator pressed REC on the video feed before engaging.
        That defeats the one record layer that cannot be pooled by accident
        (`_run_tree`), so a kind-owning mode resolves its own folder and lets
        the video recording keep its.
        """
        if on and self._csv is None:
            # The stem already lives in the run folder (the depth recording
            # that owns it opened there), so pin THAT rather than re-resolving.
            stem_path = Path(f"{stem}_mpc")
            if self.land_dry_run or self.observe:
                own = runstore.run_dir(self._run_tree())
                stem_path = own / stem_path.name
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
            if not self.engaged:
                # REC before START: the open-time meta must not show the
                # PREVIOUS engagement's flight mode as this run's.
                self._flight_mode_at_engage = None
                self._flight_modes_seen = []
                self._yaw_hold_at_engage = False
                self._attitude_at_engage = {}     # nor its attitude axes
            # Wall clock, for meta.json. Separate from _t0_csv (monotonic, the
            # CSV's own t=0) because only the wall clock lines a run up against
            # a video file or a note.
            self._csv_started = time.strftime("%Y-%m-%d %H:%M:%S")
            self._rows = 0
            self._ckpt_changes = []          # per CSV (on_policy_status)
            self._write_meta()
            self._log("info", f"mpc csv -> {path}")
            self._raw_logs(True, stem)
        except OSError as e:
            self._csv = None
            self._log("error", f"mpc csv: {e}")

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
        self._log("info", f"mpc csv closed: {self._rows} rows "
                                  f"({self._csv_path})")
        self._csv = None
        self._run_dir_pin = None   # see _run_dir()
        self._csv_auto = False
        self._ckpt_changes = []    # the closing meta above already has them

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
                self._log("warn", f"ctrl: plant model not recorded ({e})")
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
            "schema_version": 16,         # 2: + plant, mission, meta_trigger
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
                                          # 9: + fstereo (ALWAYS written). A
                                          #    SECOND depth instrument, not a
                                          #    setting on the first: with it
                                          #    enabled the middle panel and
                                          #    the depth-vs-MAP readout carry
                                          #    FoundationStereo's depth, which
                                          #    has its own disparity, its own
                                          #    host rectification, and NO
                                          #    --depth-scale correction. Any
                                          #    depth-derived distance must be
                                          #    attributed to the source named
                                          #    here before it is pooled with
                                          #    another run's.
                                          # 10: + policy (ALWAYS written), the
                                          #    `policy` mission shape (a LIVE
                                          #    diffusion policy's chunks
                                          #    through the plan seam), CSV
                                          #    grip_w_est/hold_frac,
                                          #    reference_clock.strategy
                                          #    "plan_stream_policy", and
                                          #    `source` now says whether the
                                          #    plant was real: a demo run
                                          #    reads "SYNTHETIC plant" and
                                          #    policy.synthetic is true for a
                                          #    demo OR a stub session (v2
                                          #    A15). Never pool policy runs
                                          #    with replay runs, and never
                                          #    quote a synthetic run as a
                                          #    measurement. Raw per-plan
                                          #    verdicts + actions:
                                          #    plans.jsonl (kind field).
                                          #    Within 10 (verify 2026-09-02,
                                          #    pre-hardware, no bump): the
                                          #    hold-tail mask is live-gated
                                          #    (hold_frac 1.0 = full-weight
                                          #    hold), plans.jsonl lines carry
                                          #    obs_dt_s (the plan's) beside
                                          #    obs_dt_cfg_s, and reject lines
                                          #    now include clock-domain /
                                          #    obs_dt / obs_t-schema rejects
                                          #    (policy.run.reject_clock /
                                          #    reject_obs_dt).
                                          # 11: + POLICY OBSERVE
                                          #    (--policy-observe): a fourth
                                          #    `source`, top-level
                                          #    `policy_observe`, CSV column
                                          #    `observe`, and
                                          #    `policy.gates_enforced`. On an
                                          #    observe run the control loop
                                          #    was NEVER CLOSED: u/w/axes are
                                          #    nan, solver_status is -1, the
                                          #    tracking-error columns are nan
                                          #    and the vehicle went where the
                                          #    PILOT flew it. Not comparable
                                          #    with any closed-loop run, and
                                          #    kept in its own tree
                                          #    (data/*/*_observe/) so
                                          #    it cannot be pooled by
                                          #    accident.
                                          # 12: + policy action_repr
                                          #    (contract / per-plan / run)
                                          #    and the 5-dim [dx, dy, dz,
                                          #    dyaw, width] action
                                          #    (2026-09-07): plans.jsonl
                                          #    action_raw rows are 5 wide
                                          #    from here on (10 before),
                                          #    dropped_rp_deg is 0 by
                                          #    construction on the new path,
                                          #    so old-vs-new dropped_rp_deg
                                          #    comparisons are meaningless.
                                          #    Intake rejects a plan whose
                                          #    action_repr is not the flown
                                          #    one (policy.run.
                                          #    reject_action_repr) and the
                                          #    arm check refuses a
                                          #    checkpoint that is not. No
                                          #    CSV column changed.
                                          # 13: + grip_g column (the
                                          #    look-ahead jaw sample the
                                          #    hysteresis saw) and policy/
                                          #    replay.gripper_lookahead_s
                                          #    (2026-09-07); 0.0 == pre-13
                                          #    behaviour. The knob is in
                                          #    policy.config /
                                          #    policy.estimator.lookahead_s
                                          #    / policy.run.grip_lookahead_s
                                          #    (plan_stream.config/run for
                                          #    a replay); a jaw-timing
                                          #    statistic must not be pooled
                                          #    across different values.
                                          # 14: + the eleven per-thruster CSV
                                          #    columns pwm1..pwm8, pwm_age_s,
                                          #    batt_v, batt_a (2026-09-08). The
                                          #    vehicle's OWN actuation telemetry,
                                          #    which the station had been reducing
                                          #    to the single mean `pwm_dev_us` and
                                          #    then throwing away. No existing
                                          #    column moved or changed meaning; a
                                          #    run with schema < 14 simply has no
                                          #    per-thruster record, so "did any
                                          #    one thruster clear its deadband"
                                          #    cannot be answered from it.
                                          # 15: + top-level `mode` (the LOW
                                          #    level every run) and, under
                                          #    LOW level None (TELEOP,
                                          #    2026-09-11 — the runtime form
                                          #    of --policy-observe),
                                          #    `controller` becomes {type:
                                          #    "none", role, holder: <the
                                          #    PID's meta>}: the holder kept
                                          #    the reference and was NEVER
                                          #    stepped. plot_runs.py filters
                                          #    on controller.type, so a None
                                          #    run drops out of every pool by
                                          #    construction. MpcStatus.solver
                                          #    and the CSV `solver` cell read
                                          #    "none" on such rows. policy.*
                                          #    gains ckpt_loaded / ckpt_config
                                          #    / ckpt_source and trajectory
                                          #    gains ckpt_sha1 (the panel
                                          #    picker); policy.run gains
                                          #    reject_ckpt.
                                          # 16: + the 6-DoF VARIANT
                                          #    (2026-09-26). CSV gains five
                                          #    trailing columns rroll_deg,
                                          #    rpitch_deg, ax_roll, ax_pitch,
                                          #    rp_track (nan/0 when off);
                                          #    run.attitude_axes (ALWAYS:
                                          #    enabled false = K/M dropped at
                                          #    allocation as before);
                                          #    run.stabilize.rp_axis;
                                          #    trajectory.action_repr /
                                          #    attitude_track on a policy
                                          #    mission (the pooling keys);
                                          #    policy.action_repr is the
                                          #    config's pinned value (was
                                          #    the constant); policy.run
                                          #    gains action_repr_pinned,
                                          #    attitude_track, reject_rp_mag,
                                          #    clip_rp_rate,
                                          #    div_rp_max_seen_deg;
                                          #    controller.attitude_ref /
                                          #    allocation.attitude via the
                                          #    follower's meta; plans.jsonl
                                          #    rp_tracked on every line and
                                          #    raw.rp / anchor.rp_* when
                                          #    present; policy_plan.csv
                                          #    header ends ",reason,
                                          #    roll_deg,pitch_deg" (nan on a
                                          #    4-DoF plan). NEVER POOL across
                                          #    run.attitude_axes.enabled,
                                          #    trajectory.attitude_track,
                                          #    policy.action_repr or the CSV
                                          #    rp_track bit; dobmpc w3/w4
                                          #    change meaning with the axes
                                          #    sent (credited K/M).
            # `source` answers "was the plant real". A LAND DRY-RUN is the
            # third answer and needs saying out loud: the vehicle IS the real
            # one, but it could not move and could not be commanded, and the
            # localizer was synthetic — so a tracking error or a plan-accept
            # rate from this file is not a measurement of anything. Written
            # into `source` itself rather than only a side key, because that is
            # the field every reader already looks at.
            # POLICY OBSERVE is the FOURTH answer, and the one that most
            # needs saying here rather than only in a side key: the vehicle
            # is real, the localizer is real and the command sink is LIVE —
            # every surface a reader checks looks like a quotable water run.
            # The one thing that is not real is the control loop.
            "source": (
                "hardware rov_gui.control — LAND DRY-RUN (real vehicle, but "
                "NO command sink, SYNTHETIC stationary localizer, vehicle "
                "cannot move: no measurement value)"
                if self.land_dry_run
                # THE PLANT CLAIM COMES FIRST and is never overridden: a demo
                # observe run must still say SYNTHETIC. Composing the two the
                # other way round (observe wins) made a `--source demo` run
                # announce "real vehicle", which is the one sentence
                # `test_policy_demo_meta_never_says_real_vehicle` exists to
                # forbid — the observe note is a statement about the LOOP,
                # not about the plant, so it is appended to whichever plant
                # claim is true.
                else (("hardware rov_gui.control (real vehicle)"
                       if str(getattr(self.opts, "source", "hw")
                              or "hw") == "hw"
                       else f"{getattr(self.opts, 'source', 'demo')} "
                            f"rov_gui.control (SYNTHETIC plant, no "
                            f"measurement value)")
                      + (" — POLICY OBSERVE: the controller output was "
                         "MUTED. ctrl.step() never ran; the pilot flew by "
                         "hand on the joystick and the diffusion policy's "
                         "plans were drawn, never followed. No tracking "
                         "result, plan accept/reject rate or timing from "
                         "this run measures the closed loop."
                         if self.observe else ""))),
            "land_dry_run": bool(self.land_dry_run),
            "policy_observe": bool(self.observe),
            # The LOW level (schema 15). "none" is TELEOP: read `controller`
            # for what held the reference and never flew it.
            "mode": (self.cfg.mode if self.cfg else None),
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
                    "engaged": bool(self.engaged),
                    "commanding": bool(self.commanding),
                    "observe": bool(self.observe), "traj_on": bool(self.traj_on),
                    "phase": self.phase, "reason": self.reason,
                    # RECORD BOUNDARY 2026-09-07: ArduSub flight mode(s) the
                    # engagement ran under (see on_telemetry) and the gate
                    # that admitted it. Absent in older runs = MANUAL-only.
                    "flight_mode_at_engage": getattr(self, "_flight_mode_at_engage", None),
                    "flight_modes_seen": list(getattr(self, "_flight_modes_seen", []) or []),
                    "require_mode": list(self._allowed_flight_modes()),
                    "mode_settle_s": (float(self.cfg.engage.get("mode_settle_s", 0.0))
                                      if self.cfg else None),
                    "stabilize": {
                        "yaw_axis": ((self.cfg.engage.get("stabilize") or {}).get("yaw_axis", "hold")
                                     if self.cfg else None),
                        "yaw_axis_held": bool(getattr(self, "_yaw_hold_at_engage", False)),
                        # "hold" is the only value (schema 16): the station
                        # zeroes roll/pitch under STABILIZE; the variant
                        # is refused there anyway (D7).
                        "rp_axis": ((self.cfg.engage.get("stabilize") or {}).get("rp_axis", "hold")
                                    if self.cfg else None)},
                    # THE 6-DoF VARIANT (schema 16). ALWAYS written: enabled
                    # false = K/M were dropped at allocation exactly as every
                    # run before 2026-09-26. The dict is the ENGAGE record
                    # (survives disengage; sat_ticks / degraded_at are filled
                    # at disengage and live here in between).
                    "attitude_axes": self._attitude_axes_meta(),
                    "frame_note": "px,py,pz / rx,ry,rz are world FLU "
                                  "(sim-compatible); w*/u* are NED body"},
            "controller": (
                # LOW level None (schema 15): the PID is a REFERENCE HOLDER.
                # Its meta is kept under `holder` so a reader can still see
                # what sized the drawn plan, but `type` says nothing flew.
                {"type": "none",
                 "role": "reference holder — never stepped (LOW level None "
                         "/ teleop)",
                 "holder": self.ctrl.meta()}
                if (self.cfg is not None and self.cfg.mode == "none"
                    and self.ctrl is not None)
                else (self.ctrl.meta() if self.ctrl is not None else None)),
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
                #   plan_stream_policy           (2026-09-02 on) the LIVE
                #        diffusion policy's chunks through that same seam
                # Keyed on the ARMED scenario, not on _replay_last: a square
                # armed after a replay in the same engagement must not keep
                # wearing the replay label (scenario.kind is what flips).
                "strategy": (
                    "plan_stream_policy"
                    if ((self.replay or {}).get("kind") == "policy"
                        or (self.ctrl is not None
                            and (self.ctrl.scenario or {}).get("kind")
                            == "policy"))
                    else "plan_stream_replay"
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
            # ALWAYS written, off included: an absent key cannot distinguish
            # "this build had no learned depth" from "it was switched off",
            # and the schema version cannot either once both exist at 9.
            "fstereo": self._fstereo_meta(),
            # SECOND-CAMERA FALLBACK (control/nav_fusion.py): its config and
            # what it had learned when this was written. Always present:
            # null = no fallback this run, which an absent key could not say.
            "nav_fallback": self._nav_fallback_meta(),
            # THE DRAWN VEHICLE (widgets/rov_shape.py): every dimension the
            # trajectory panel paints the ROV with, each with its provenance
            # tag. The module had promised this table to the run meta since
            # 2026-09-03 and nothing wrote it (review 2026-09-07); a
            # screenshot's hull, gripper, TCP ring and heading line are
            # measured against THESE numbers, so the record has to carry
            # them. Pure data — no Qt is imported for it.
            "rov_drawn_geometry": _rov_drawn_geometry_meta(),
            # LIVE POLICY. ALWAYS written (schema 10), same rule: config +
            # provenance, TCP geometry, jaw estimator, worker meta, the run's
            # counters and end_reason, and `synthetic`.
            "policy": self._policy_meta(),
        }

    def _attitude_axes_meta(self) -> dict:
        """run.attitude_axes (schema 16): the engage-time record of the
        6-DoF variant, or the OFF record with the config echoed. Every key
        is present in both so a reader never has to guess a default."""
        a = self._attitude_cfg()
        gains = dict(self.cfg.axis_gain) if self.cfg else {}
        rec = self._attitude_at_engage or {}
        out = {
            "enabled": bool(rec.get("enabled", False)),
            "configured": bool(rec.get("configured", a.get("enabled", False))),
            "live": bool(self._attitude_axes),
            "transport": str(rec.get("transport") or a.get("transport")
                             or "manual_control_ext"),
            "roll_nm": float(rec.get("roll_nm", gains.get("roll_nm", 13.2))),
            "pitch_nm": float(rec.get("pitch_nm", gains.get("pitch_nm", 7.2))),
            "gain_provenance": "[유도] heave_n / 4 x lever arms "
                               "(bluerov_heavy.xml) — replace after the "
                               "attitude step test",
            "cap_roll": float(rec.get("cap_roll", a.get("cap_roll", 0.2))),
            "cap_pitch": float(rec.get("cap_pitch", a.get("cap_pitch", 0.3))),
            "slew_per_s": float(rec.get("slew_per_s", a.get("slew_per_s", 1.5))),
            "sign": dict(rec.get("sign") or a.get("sign") or {}),
            "abort_deg": float(a.get("abort_deg", 35.0)),
            "sat_ineffective_s": float(a.get("sat_ineffective_s", 1.0)),
            "dobmpc_allowed": bool(a.get("dobmpc_allowed", False)),
            "min_firmware": str(a.get("min_firmware", "4.1.2")),
            "mavlink_wire_version": str(rec.get("mavlink_wire_version", "")),
            "firmware_version": str(rec.get("firmware_version", "")),
            "probe": rec.get("probe", a.get("probe")),
            "sign_probe": rec.get("sign_probe", a.get("sign_probe")),
            # What the pinned artefacts WERE and PROVED at engage (audit
            # 2026-09-26): sha1 of the bytes, sign_proven, and which cap set
            # bounded the wire (first_water until the sign is proven).
            "probe_sha1": rec.get("probe_sha1"),
            "sign_probe_sha1": rec.get("sign_probe_sha1"),
            "sign_proven": bool(rec.get("sign_proven", False)),
            "probe_firmware_version": rec.get("probe_firmware_version"),
            "require_probe": bool(rec.get("require_probe",
                                          a.get("require_probe", True))),
            "require_sign_probe": bool(rec.get("require_sign_probe",
                                               a.get("require_sign_probe", True))),
            "cap_roll_configured": float(rec.get("cap_roll_configured",
                                                 a.get("cap_roll", 0.2))),
            "cap_pitch_configured": float(rec.get("cap_pitch_configured",
                                                  a.get("cap_pitch", 0.3))),
            "first_water_caps": rec.get("first_water_caps",
                                        a.get("first_water_caps")),
            "caps_in_force": rec.get("caps_in_force"),
            "caps_in_force_why": rec.get("caps_in_force_why",
                                         "no engagement (variant off)"),
            "rc_trim_at_engage": rec.get("rc_trim_at_engage"),
            "enabled_extensions_sent": bool(rec.get("enabled_extensions_sent",
                                                    False)),
            "degraded_at": (rec.get("degraded_at") or self._att_degraded_at),
            "sat_ticks": int(rec.get("sat_ticks", 0) or 0) + (
                int(self._attitude_sat_ticks) if self._attitude_axes else 0),
            "u_max_wire_nm": rec.get("u_max_wire_nm"),
            "low_allowed": list(self.ATTITUDE_LOW_ALLOWED),
            "note": ("K/M = NMPC torques as MANUAL_CONTROL extension axes "
                     "s = pitch, t = roll (MANUAL only); absent/false = K/M "
                     "dropped at allocation (the pre-2026-09-26 behaviour)"),
        }
        return out

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
                json.dumps(self._run_meta("csv"), indent=1, ensure_ascii=False,
                           default=_json_default))
        except (OSError, TypeError, ValueError) as e:
            self._log("error", f"mpc meta: {e}")

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
                                      ensure_ascii=False,
                                      default=_json_default))
            mode = self.cfg.mode
            self._log(
                "info", f"ctrl: {mode} constants + plant model -> {out}")
        except OSError as e:
            self._log("error", f"ctrl: controller.json not written ({e})")

    def _nav_fallback_meta(self):
        """TagNavWorker.fallback_meta, injected by HardwareBackend (the
        fstereo_meta_fn pattern); None when there is no fallback."""
        fn = getattr(self, "nav_fallback_meta_fn", None)
        if fn is None:
            return None
        try:
            return fn()
        except Exception as e:                                  # noqa: BLE001
            return {"error": repr(e)}

    def _fstereo_meta(self) -> dict:
        """What the learned-depth path was, for this run.

        ``fstereo_meta_fn`` is set by HardwareBackend when --fstereo is on; it
        is FStereoWorker.meta. Reading it through a hook rather than importing
        the worker keeps this module free of the perception stack, which is the
        same reason `--pose` is reachable here only as a flag.
        """
        fn = getattr(self, "fstereo_meta_fn", None)
        if fn is None:
            return {"enabled": False,
                    "why": ("no --fstereo"
                            if not getattr(self.opts, "fstereo", False)
                            else "not built")}
        try:
            return dict(fn())
        except Exception as e:                                   # noqa: BLE001
            return {"enabled": None, "why": f"{type(e).__name__}: {e}"}

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
            # UNDER OBSERVE the reference columns are written and the
            # FOLLOWER columns are not — see CSV_HEADER's 2026-09-03 note.
            # rx/ry/rz/ryaw are what the network asked for; e_along, e_cross
            # and ref_speed are statements about a tracker that never ran.
            if not self.observe:
                ref_speed = float(np.linalg.norm(
                    np.asarray(v_ref, float)[:2]))
                if meas is not None:
                    e_along, e_cross = self._path_split(meas, p_ref, t_traj)
            if self.traj_on and self.ctrl.scenario:
                lap = self._path_lap(t_traj)
        w = info.get("w_hat", np.zeros(6))
        ax = (axes.surge, axes.sway, axes.heave, axes.yaw) if axes else \
            (float("nan"),) * 4
        nis = float(info.get("nis", 0.0))
        status = int(info.get("status", 0))
        solve_ms = info.get("solve_ms")
        tick_ms = float(self._tick_ms)
        if self.observe:
            # THE SENTINELS ARE APPLIED HERE, at the writer, not only on the
            # tick's observe path — because tick() reaches `_write_row`
            # through THREE routes and only one of them passed through that
            # branch: a tick with no fix (`elif meas is None`) and a tick a
            # guard ended mid-way both fall to the bottom carrying the
            # function-top defaults `u = zeros(6)` / `info = {}`. Those rows
            # came out as EXACTLY 0.000 N with solver_status 0 — "the
            # controller commanded zero and the solver converged" — which is
            # the one reading CSV_HEADER, the schema-11 note and the launch
            # banner all promise cannot appear in an observe file. A guard at
            # the emitter cannot be routed around by a caller.
            u = np.full(6, np.nan)
            w = np.full(6, np.nan)
            ax = (float("nan"),) * 4
            nis = float("nan")
            status = OBSERVE_SOLVER_STATUS
            solve_ms = None
            tick_ms = float("nan")
        row = [
            f"{t - self._t0_csv:.4f}", f"{px:.5f}", f"{py:.5f}", f"{pz:.5f}",
            f"{rx:.5f}", f"{ry:.5f}", f"{math.degrees(yaw_flu):.3f}",
            f"{math.degrees(pitch_flu):.3f}", str(lap),
            f"{rz:.5f}", f"{ryaw:.3f}",
            (f"{t_traj:.4f}" if t_traj is not None else "nan"),
            self.cfg.mode, str(int(self.engaged)), str(int(self.traj_on)),
            # "none" on an observe row (the solver_status sentinel rule).
            ("none" if self.observe
             else (self.ctrl.solver_kind if self.ctrl else "")),
            str(status),
            (f"{solve_ms:.2f}" if solve_ms is not None else "nan"),
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
        row += [f"{nis:.2f}",
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
        row += [f"{tick_ms:.2f}"]
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
        # POLICY (schema 10): the open-loop jaw width the policy is told
        # (nan without a policy worker) and the horizon fraction past the
        # active plan's end (nan outside a policy run). See CSV_HEADER's
        # 2026-09-02 note.
        row += [(f"{self._grip_est.width(t):.4f}"
                 if (self.policy_present and self._grip_est is not None)
                 else "nan"),
                # (schema 13) the jaw-channel sample the hysteresis saw
                # this tick (at t_traj + gripper_lookahead_s); nan when no
                # stream is live, its jaw is off, or no plan is installed.
                # See CSV_HEADER's 2026-09-07 (2) note.
                (f"{float(rp['grip_g']):.4f}"
                 if (rp is not None and self.traj_on
                     and math.isfinite(float(rp.get("grip_g",
                                                    float("nan")))))
                 else "nan"),
                (f"{float(rp['hold_frac']):.3f}"
                 if (rp is not None and rp.get("kind") == "policy"
                     and self.traj_on
                     and math.isfinite(float(rp.get("hold_frac",
                                                    float("nan")))))
                 else "nan")]
        # POLICY OBSERVE (schema 11). The per-ROW answer to "was the loop
        # closed", so a reader who never opens the meta still cannot mistake
        # this file for a control run. See CSV_HEADER's 2026-09-03 note.
        row += [str(int(bool(self.observe)))]
        # ACTUATION AS THE VEHICLE REPORTS IT (schema 14). Everything to the
        # left of here is what the station ASKED for; these eleven are what the
        # autopilot says it did. Empty (not "nan") for a thruster the vehicle
        # does not report, because that is a different fact from "at neutral".
        row += self._thruster_row()
        # THE 6-DoF VARIANT (schema 16): attitude reference (world-FLU signs,
        # like roll_deg / pitch_deg), the K/M axes SENT, and the per-row
        # boundary bit. nan/nan/nan/nan/0 whenever the axes were not sent —
        # off, observe, or a tick that commanded nothing.
        if self._attitude_axes and axes is not None and not self.observe:
            phi_n, th_n = self._ref_attitude(t_traj)
            row += [f"{math.degrees(phi_n):.3f}",
                    f"{math.degrees(-th_n):.3f}",
                    f"{float(getattr(axes, 'roll', 0.0)):.4f}",
                    f"{float(getattr(axes, 'pitch', 0.0)):.4f}", "1"]
        else:
            row += ["nan", "nan", "nan", "nan", "0"]
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
