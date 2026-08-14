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
from pathlib import Path

import numpy as np

from ..qt import Slot
from ..state import Conn, MpcStatus, NavFix, PilotInput, TagOverlay, now
from .allocation import axes_to_wrench, wrench_to_axes
from .geometry import (MpcConfig, NavConfig, geofence_contains, yaw_from_R)
from .state_assembler import StateAssembler, rot_zyx
from ..backends.base import TimerWorker


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
              "ax_surge,ax_sway,ax_heave,ax_yaw,nis,geofence_ok,"
              "pwm_dev_us\n")


class MpcWorker(TimerWorker):
    """The 20 Hz brain. See the module docstring for the arbitration contract.

    ``controller_factory`` exists for the offline tests and the demo: it must
    return an object with the :class:`~rov_gui.control.mpc_bridge.HwDobMpc`
    surface (step / set_target_ned / set_square_ned / ref_ned_at /
    note_applied / reset / realtime_ok / solver_kind / meta). The default
    factory builds the real thing, which imports casadi+acados and may spend
    tens of seconds code-generating the solver — that is WHY it happens in
    setup() at startup and never at engage time.
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
        self._ready = False
        self._setup_error = ""

        self.fix = None                    # latest NavFix
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
        self._auto_traj = False            # one-button flow: traj after warm-up
        # Mission datum, captured at ENGAGE: the engage pose in the TAG frame
        # becomes (0,0,0)/yaw 0 for everything downstream (controller, square,
        # geofence, CSV, plot). Set fresh at every engage — the operator's
        # "(0,0) = where I pressed START". Kept across disengage so the plot
        # and post-run CSV rows stay in the frame the run used.
        self._datum = None                 # {"p0": (3,), "yaw0": f, "Rz": 3x3}
        self.reason = ""
        self._warmup_left = 0
        self._t0_traj: float | None = None
        self._t_engage: float | None = None
        self._eta = None                   # last assembled state (np arrays)
        self._nfail_prev = 0
        self._fail_streak = 0
        self._scenario_override: dict = {}
        self._last_health: dict = {}

        self._csv = None
        self._csv_path: Path | None = None
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
        # Both controller families are built up front: the PID costs nothing
        # and stays available even when the acados build fails, so the mode
        # combo can swap between them without a rebuild (never mid-engage —
        # set_mode refuses that).
        from .pid import HwPid

        self._pid = HwPid(self.cfg, log=lambda m: self.bus.log.emit("info", m))
        self._mpc_ctrl = None
        try:
            if self._factory is not None:
                self._mpc_ctrl = self._factory(self.cfg)
            else:
                from .mpc_bridge import HwDobMpc
                self._mpc_ctrl = HwDobMpc(
                    self.cfg.mode if self.cfg.mode != "pid" else "dobmpc",
                    self.cfg, log=lambda m: self.bus.log.emit("info", m))
        except Exception as e:                                   # noqa: BLE001
            self._setup_error = f"{type(e).__name__}: {e}"
            self.bus.log.emit("error", f"mpc: MPC controller build FAILED — "
                                       f"{self._setup_error}. PID mode is "
                                       f"still available.")
        self.ctrl = self._pid if self.cfg.mode == "pid" else self._mpc_ctrl
        self._ready = True
        if self.ctrl is not None:
            k = self.ctrl.solver_kind
            ok = self.ctrl.realtime_ok
            self.bus.log.emit(
                "warn" if not ok else "info",
                f"mpc: {self.cfg.mode} ready, solver={k}, "
                f"probe={self.ctrl.probe_ms and round(self.ctrl.probe_ms, 1)} ms"
                + ("" if ok else " — NOT real-time capable, engage will refuse"))

    # ---------------------------------------------------------------- inputs
    @Slot(object)
    def on_nav_fix(self, fix) -> None:
        if getattr(fix, "ok", False):
            self.fix = fix

    @Slot(object)
    def on_vehicle_imu(self, imu) -> None:
        self.imu = imu

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

    @Slot(str)
    def set_mode(self, mode: str) -> None:
        mode = str(mode).lower()
        if mode not in ("mpc", "dobmpc", "pid"):
            return
        if self.engaged:
            self.bus.log.emit("warn", "mpc: mode change refused while engaged")
            return
        if self.cfg is None or not self._ready:
            return
        if mode == "pid":
            self.ctrl = self._pid
        else:
            if self._mpc_ctrl is None:
                self.bus.log.emit("error", f"mpc: mode {mode} unavailable "
                                           f"({self._setup_error}) — staying "
                                           f"on {self.cfg.mode}")
                return
            self.ctrl = self._mpc_ctrl
            if mode != self.ctrl.mode:
                # Same solver, same EAOB machinery — mpc/dobmpc only decides
                # whether w_hat reaches the solver (mirrors the sim).
                self.ctrl.mode = mode
        self.ctrl.reset()
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
        self.ctrl.set_target_ned(meas["eta"][:3], meas["eta"][5])
        self._eta = meas["eta"]
        self.engaged = True
        self.traj_on = False
        self._t0_traj = None
        self._t_engage = now()
        self._warmup_left = max(1, int(round(
            float(self.cfg.engage["warmup_s"]) * self.cfg.ctrl_hz)))
        self._fail_streak = 0
        self._nfail_prev = 0
        self.reason = "engaged (DP hold)"
        self._log_event(f"ENGAGED ({self.cfg.mode}) datum p0="
                        f"{[round(float(v), 3) for v in self._datum['p0']]} "
                        f"yaw0 {math.degrees(self._datum['yaw0']):.1f} deg")
        self.bus.log.emit(
            "warn", f"mpc: ENGAGED ({self.cfg.mode}) — holding "
                    f"({meas['eta'][0]:+.2f}, {meas['eta'][1]:+.2f}, "
                    f"{meas['eta'][2]:+.2f}) NED. START TRAJ begins the square.")
        if self._csv is None:
            stem = (Path(self.cfg.log_dir)
                    / time.strftime("%Y%m%d_%H%M%S"))
            self._open_csv(stem, auto=True)

    def _log_event(self, msg: str) -> None:
        """Every refusal/engage/disengage into ONE persistent file — a pool
        session run without a terminal must still leave its story behind
        (2026-08-12: the 19:29 refusals lived only on a screen recording)."""
        if self.cfg is None:
            return
        try:
            p = Path(self.cfg.log_dir) / "events.log"
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
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
        # No position gate here: the geofence is relative to the engage datum,
        # and the engage pose is (0,0,0) by definition. The runtime fault and
        # the START-TRAJ corner check are the fence's teeth.
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
            if self.traj_on and self._eta is not None:
                self.ctrl.set_target_ned(self._eta[:3], self._eta[5])
            self.traj_on = False
            self._t0_traj = None
            self.reason = "trajectory stopped (DP hold)"
            self.bus.log.emit("info", "mpc: trajectory stopped — DP hold")
            return
        if not self.engaged or self.ctrl is None:
            self.bus.log.emit("warn", "mpc: START TRAJ refused — not engaged")
            return
        if self._warmup_left > 0:
            self.bus.log.emit(
                "warn", f"mpc: START TRAJ refused — warming up "
                        f"({self._warmup_left / self.cfg.ctrl_hz:.1f}s left)")
            return
        if self._eta is None:
            return
        sq = dict(self.cfg.square)
        sq.update(self._scenario_override)
        origin = sq.get("origin", "current")
        if origin == "current" or origin is None:
            origin_xy = (float(self._eta[0]), float(self._eta[1]))
        else:
            origin_xy = (float(origin[0]), float(origin[1]))
        yf = sq.get("yaw_fixed_deg", "current")
        yaw_fixed = (float(self._eta[5]) if yf in ("current", None)
                     else math.radians(float(yf)))
        depth = sq.get("depth_ned")
        depth = float(self._eta[2]) if depth is None else float(depth)
        err = math.hypot(self._eta[0] - origin_xy[0], self._eta[1] - origin_xy[1])
        lim = float(self.cfg.engage["start_err_max_m"])
        if err > lim:
            self.bus.log.emit(
                "warn", f"mpc: START TRAJ refused — {err:.2f} m from the "
                        f"square origin (limit {lim:.2f}). Move closer or set "
                        f"square.origin: current.")
            return
        # The PLACED square must fit the geofence BEFORE it is armed — the
        # runtime fence abort (margin band) is a last defense with momentum,
        # not the plan (review 2026-08-12). Base box, margin 0: the outward
        # runtime margin is the tracking-overshoot allowance, not a placement
        # buffer.
        from .reference import square_corners_world

        flu = square_corners_world(float(sq.get("size", 1.0)),
                                   (origin_xy[0], -origin_xy[1]),
                                   -math.radians(float(sq.get("rot_deg", 0.0))))
        box = self.nav_cfg.geofence
        for cx, cy in ((x, -y) for x, y in flu):
            if not geofence_contains(box, (cx, cy, depth), margin=0.0):
                self.bus.log.emit(
                    "warn", f"mpc: START TRAJ refused — square corner "
                            f"({cx:+.2f}, {cy:+.2f}, {depth:+.2f}) NED is "
                            f"outside the geofence. Move the vehicle, shrink "
                            f"square.size, or fix geofence_ned.")
                return
        scen = self.ctrl.set_square_ned(sq, origin_xy, yaw_fixed, depth)
        self.traj_on = True
        self._t0_traj = now()
        self.reason = "square running"
        self._log_event(f"SQUARE started - {scen['size']} m @ "
                        f"{scen['speed']} m/s x{scen['laps']}")
        self.bus.log.emit(
            "warn", f"mpc: SQUARE started — {scen['size']} m @ "
                    f"{scen['speed']} m/s, {scen['laps']} lap(s), "
                    f"heading_follow={scen['heading_follow']}")

    def disengage(self, why: str) -> None:
        was = self.engaged
        self.engaged = False
        self.traj_on = False
        self._auto_traj = False
        self._t0_traj = None
        if was:
            # One explicit neutral so the vehicle stops NOW; the window then
            # resumes the teleop pump, and the sink deadman covers the gap.
            self.bus.cmd_pilot.emit(PilotInput(source="mpc"))
            self.reason = f"disengaged: {why}"
            self.bus.log.emit("warn", f"mpc: DISENGAGED — {why}")
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
        self._last_health = health
        u = np.zeros(6)
        info = {}
        axes = None
        t_traj = None
        if meas is not None:
            self._eta = meas["eta"]

        if self.engaged:
            why = self._runtime_fault(meas, health, t)
            if why:
                self.disengage(why)
            else:
                if self._warmup_left > 0:
                    self._warmup_left -= 1
                if (self._auto_traj and self._warmup_left <= 0
                        and not self.traj_on):
                    # One attempt only: a refusal (square outside the fence)
                    # must not retry at 20 Hz — the operator repositions and
                    # presses START again.
                    self._auto_traj = False
                    self.set_traj(True)
                t_traj = (t - self._t0_traj) if self._t0_traj is not None else 0.0
                u, info = self.ctrl.step(meas["eta"], meas["nu"],
                                         meas["nudot"], t_traj)
                nf = info.get("n_fail", 0)
                failed_now = nf > self._nfail_prev
                self._fail_streak = self._fail_streak + 1 if failed_now else 0
                self._nfail_prev = nf
                if failed_now:
                    # Timestamped, with w_hat — the exact logging KNOWN_ISSUES
                    # asked for before any hardware run.
                    self.bus.log.emit(
                        "error", f"mpc: solver FAILURE status "
                                 f"{info.get('status')} at t={t_traj:.2f}s "
                                 f"(n_fail={nf}, |w_xyz|="
                                 f"{np.linalg.norm(self.ctrl.w_hat[:3]):.1f} N)")
                if self._fail_streak >= int(self.cfg.engage["max_solver_fails"]):
                    self.disengage(f"solver failed {self._fail_streak}x")
                elif info.get("solve_ms") is not None and \
                        info["solve_ms"] > float(self.cfg.engage["tick_overrun_ms"]):
                    self.disengage(f"solve {info['solve_ms']:.0f} ms > budget")
                else:
                    axes = wrench_to_axes(u, self.cfg.axis_gain,
                                          self.cfg.axis_cap)
                    self.ctrl.note_applied(axes_to_wrench(axes,
                                                          self.cfg.axis_gain))
                    self.bus.cmd_pilot.emit(axes)
                    self._watch_actuation(axes, t)
                    if (self.traj_on and self.ctrl.scenario
                            and t_traj > self.ctrl.scenario["T_run_s"]):
                        self.set_traj(False)
                        self.reason = "square complete (DP hold)"
                        self.bus.log.emit("warn", "mpc: square COMPLETE — "
                                                  "holding the final pose")

        self._publish(meas, health, u, info, axes, t_traj)
        self._write_row(meas, health, u, info, axes, t_traj, t)

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
            return health.get("why", "state lost")
        tel = self.tel
        if tel is None or tel.armed is not True:
            return "vehicle disarmed"
        if t - float(tel.stamp) > 2.0:
            return "vehicle telemetry stale"
        want = str(self.cfg.engage.get("require_mode", "MANUAL")).upper()
        if want and not (tel.mode or "").upper().startswith(want):
            return f"flight mode left {want}"
        box = self.nav_cfg.geofence
        if not geofence_contains(box, meas["eta"][:3],
                                 margin=float(box.get("margin", 0.4))):
            return "geofence violated"
        return ""

    # ------------------------------------------------------------- reporting
    def _publish(self, meas, health, u, info, axes, t_traj) -> None:
        s = MpcStatus(
            engaged=self.engaged, traj_on=self.traj_on,
            mode=self.cfg.mode if self.cfg else "",
            scenario=(self.ctrl.scenario if (self.ctrl and self.traj_on)
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
            geofence_ok=(meas is None or geofence_contains(
                self.nav_cfg.geofence, meas["eta"][:3],
                margin=float(self.nav_cfg.geofence.get("margin", 0.4)))),
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
        if self.engaged and self.ctrl is not None:
            p_ref, yaw_ref = self.ctrl.ref_ned_at(t_traj or 0.0)
            rx, ry, rz = _flu_of_ned(*p_ref)
            s.ref_flu = (rx, ry, rz)
            if meas is not None:
                s.err_xy = math.hypot(meas["eta"][0] - p_ref[0],
                                      meas["eta"][1] - p_ref[1])
            if self.traj_on and self.ctrl.scenario:
                from .reference import lap_of
                s.lap = lap_of(t_traj or 0.0, self.ctrl.scenario["size"],
                               self.ctrl.scenario["speed"])
        self.bus.mpc_status.emit(s)

    # ------------------------------------------------------------------- CSV
    @Slot(bool, str)
    def set_sensor_log(self, on: bool, stem: str) -> None:
        """Rides the shared recording stem, like the other sensor logs."""
        if on and self._csv is None:
            self._open_csv(Path(f"{stem}_mpc"), auto=False)
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
            self._rows = 0
            self._write_meta()
            self.bus.log.emit("info", f"mpc csv -> {path}")
        except OSError as e:
            self._csv = None
            self.bus.log.emit("error", f"mpc csv: {e}")

    def _close_csv(self) -> None:
        if self._csv is None:
            return
        try:
            self._csv.close()
        except OSError:
            pass
        self._write_meta()                    # final row count
        self.bus.log.emit("info", f"mpc csv closed: {self._rows} rows "
                                  f"({self._csv_path})")
        self._csv = None
        self._csv_auto = False

    def _write_meta(self) -> None:
        if self._csv_path is None:
            return
        import hashlib

        nav = self.nav_cfg
        tag_map_sha = None
        if nav and nav.geometry == "floor":
            try:
                tag_map_sha = hashlib.sha1(
                    Path(nav.tag_map_path).read_bytes()).hexdigest()[:12]
            except OSError:
                pass
        meta = {
            "schema_version": 1,
            "source": "hardware rov_gui.control (real vehicle)",
            "rov_model": self.cfg.rov_model if self.cfg else None,
            "run": {"started": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "log_hz": self.cfg.ctrl_hz if self.cfg else None,
                    "csv": str(self._csv_path), "rows": self._rows,
                    "frame_note": "px,py,pz / rx,ry,rz are world FLU "
                                  "(sim-compatible); w*/u* are NED body"},
            "controller": (self.ctrl.meta() if self.ctrl is not None else None),
            "trajectory": (self.ctrl.scenario if self.ctrl is not None else None),
            "disturbance": None,
            "hardware": {
                "geometry": nav.geometry if nav else None,
                "tag_map": (nav.tag_map_path if nav and nav.geometry == "floor"
                            else f"single wall tag {nav.wall_tag_id}" if nav else None),
                "tag_map_sha1": tag_map_sha,
                "tag_size_m": nav.effective_tag_size() if nav else None,
                "cam_t_flu": list(nav.cam_t_flu) if nav else None,
                "cam_xyaxes_flu": list(nav.cam_xyaxes_flu) if nav else None,
                "z_source": nav.z_source if nav else None,
                "axis_gain": (dict(self.cfg.axis_gain) if self.cfg else None),
                "axis_cap": (self.cfg.axis_cap if self.cfg else None),
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
        }
        try:
            self._csv_path.with_suffix(".meta.json").write_text(
                json.dumps(meta, indent=1, ensure_ascii=False))
        except OSError as e:
            self.bus.log.emit("error", f"mpc meta: {e}")

    def _write_row(self, meas, health, u, info, axes, t_traj, t) -> None:
        if self._csv is None or meas is None:
            return
        px, py, pz = _flu_of_ned(*meas["eta"][:3])
        yaw_flu = -meas["eta"][5]
        pitch_flu = -meas["eta"][4]
        rx = ry = rz = ryaw = float("nan")
        lap = 0
        if self.engaged and self.ctrl is not None:
            p_ref, yaw_ref_ned = self.ctrl.ref_ned_at(t_traj or 0.0)
            rx, ry, rz = _flu_of_ned(*p_ref)
            ryaw = math.degrees(-yaw_ref_ned)
            if self.traj_on and self.ctrl.scenario:
                from .reference import lap_of
                lap = lap_of(t_traj or 0.0, self.ctrl.scenario["size"],
                             self.ctrl.scenario["speed"])
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
                str(int(geofence_contains(self.nav_cfg.geofence,
                                          meas["eta"][:3],
                                          margin=float(self.nav_cfg.geofence.get(
                                              "margin", 0.4))))),
                (f"{dev:.0f}" if dev is not None else "nan")]
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
