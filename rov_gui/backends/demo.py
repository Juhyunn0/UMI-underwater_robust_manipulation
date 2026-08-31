#!/usr/bin/env python3
"""
demo.py — a synthetic vehicle and synthetic cameras.

Purpose: the dashboard must be developable, testable and demonstrable without a
vehicle in the water, a camera on the bench, or ROS installed. This backend
exercises every path the real ones use — the mailboxes, the bus, the command
signals, the freshness watchdog — so a layout or threading bug shows up here
rather than on the boat.

EVERY NUMBER THIS MODULE PRODUCES IS FABRICATED.
------------------------------------------------
That is not a caveat, it is the module's contract, and this repo has already
been bitten by the opposite (a placeholder written beside a tool was later cited
as a measured device fact — see docs/MEASUREMENT_AUDIT.md and the measurement
rule in CLAUDE.md). So:

* the backend sets ``simulated = True``, which makes the window show a permanent
  ``SIMULATED DATA`` banner;
* every VideoStat carries ``note="synthetic"``;
* nothing here is written to disk or logged in a form that could be read back
  as a measurement.

The one exception is the tether link panel, which reads the *real* host NIC
counters even in demo mode (see :mod:`rov_gui.net`) — because those are free,
real, and the thing you most want to sanity-check before a dive. If there is no
interface on the tether subnet it reports OFFLINE rather than inventing traffic.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from .. import imaging
from ..bus import FrameMailbox
from ..net import NicMonitor
from ..qt import Slot
from ..sensorlog import SensorLog
from ..state import (Conn, ImuBatch, NavFix, PayloadState, PilotInput,
                     PoseTrack, SensorStat, Telemetry, ThrusterState,
                     VehicleImu, VideoStat, now)
from .base import Backend, TimerWorker

# Demo thruster mixer: rows are the 8 motors of a BlueROV2 Heavy, columns are
# (surge, sway, heave, yaw). SHAPE ONLY — this is a plausible vectored-4 +
# vertical-4 geometry written for the demo, NOT ArduSub's AP_Motors6DOF table
# and NOT this project's allocation matrix. Do not read motor numbering, signs,
# or scaling off it for anything real.
DEMO_MIX = np.array([
    #  surge  sway  heave  yaw
    [+0.71, -0.71, 0.0, +0.50],   # T1 front-right horizontal
    [+0.71, +0.71, 0.0, -0.50],   # T2 front-left
    [+0.71, +0.71, 0.0, +0.50],   # T3 rear-right
    [+0.71, -0.71, 0.0, -0.50],   # T4 rear-left
    [0.0, 0.0, +1.0, 0.0],        # T5 vertical
    [0.0, 0.0, +1.0, 0.0],        # T6
    [0.0, 0.0, +1.0, 0.0],        # T7
    [0.0, 0.0, +1.0, 0.0],        # T8
])


# =============================================================================
# synthetic imagery
# =============================================================================
class _Scene:
    """A cheap moving underwater-ish scene, generated once and animated.

    Regenerating turbulence per frame costs more CPU than the whole rest of the
    demo, so the base is built once and shifted; only the overlay text and a
    couple of moving shapes are drawn per frame.
    """

    def __init__(self, w: int = 640, h: int = 360, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.w, self.h = w, h
        # Blue-green vertical gradient with a vignette, i.e. what a wide-angle
        # camera in green water actually looks like.
        yy = np.linspace(0.25, 1.0, h, dtype=np.float32)[:, None]
        xx = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
        vign = np.clip(1.15 - 0.55 * (xx ** 2 + (np.linspace(-1, 1, h,
                       dtype=np.float32)[:, None]) ** 2), 0.25, 1.0)
        base = np.zeros((h, w, 3), np.float32)
        base[..., 0] = 95 * yy * vign        # B
        base[..., 1] = 120 * yy * vign       # G
        base[..., 2] = 55 * yy * vign        # R
        base += rng.normal(0, 3.0, base.shape).astype(np.float32)
        self.base = np.clip(base, 0, 255).astype(np.uint8)
        # Marine snow: particles drifting through the frame.
        self.snow = rng.random((140, 3)).astype(np.float32)   # x, y, size

    def render(self, t: float, w: int, h: int, mono: bool = False) -> np.ndarray:
        cv2 = imaging.import_cv2()
        img = self.base.copy()
        # a structure sliding past, so motion is visible on a still bench
        x = int((0.5 + 0.35 * math.sin(t * 0.25)) * self.w)
        y = int(self.h * 0.55 + 18 * math.sin(t * 0.7))
        cv2.rectangle(img, (x - 70, y - 46), (x + 70, y + 46), (58, 74, 92), -1)
        cv2.rectangle(img, (x - 70, y - 46), (x + 70, y + 46), (120, 150, 170), 2)
        cv2.circle(img, (x, y), 13, (40, 200, 255), -1)
        for px, py, ps in self.snow:
            sx = int((px * self.w + t * 26 * (0.4 + ps)) % self.w)
            sy = int((py * self.h + t * 9) % self.h)
            cv2.circle(img, (sx, sy), 1 + int(ps * 2), (210, 230, 235), -1)
        if mono:
            img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                               cv2.COLOR_GRAY2BGR)
        if (w, h) != (self.w, self.h):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        return img

    def depth(self, t: float, w: int, h: int) -> np.ndarray:
        """A uint16 millimetre map with a near object and invalid patches."""
        yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        d = (1200 + 4200 * (1.0 - yy)) * np.ones((1, w), np.float32)
        cx = int((0.5 + 0.35 * math.sin(t * 0.25)) * w)
        cy = int(h * 0.55)
        y0, y1 = max(0, cy - h // 8), min(h, cy + h // 8)
        x0, x1 = max(0, cx - w // 9), min(w, cx + w // 9)
        d[y0:y1, x0:x1] = 900 + 60 * math.sin(t)
        depth = d.astype(np.uint16)
        # Stereo always has holes: low texture, occlusion, and the near limit.
        depth[(np.random.random((h, w)) < 0.04)] = 0
        depth[:, : w // 32] = 0
        return depth


class DemoVideoWorker(TimerWorker):
    """Produces main / stereo / depth frames into their mailboxes."""

    def __init__(self, mailboxes: dict[str, FrameMailbox], fps: float = 15.0):
        super().__init__("demo-video", interval_ms=int(1000 / max(1.0, fps)))
        self.mailboxes = mailboxes
        self.fps = fps
        self.scene = _Scene()
        self._t0 = time.monotonic()
        self._seq = 0
        self._last_pub = 0.0
        self.bus = None            # set by the backend, for VideoStat

    def setup(self) -> None:
        self._t0 = time.monotonic()

    def tick(self) -> None:
        t = time.monotonic() - self._t0
        self._seq += 1
        for name, mb in self.mailboxes.items():
            tw, th = mb.target_size()
            if tw <= 0 or th <= 0:
                tw, th = 480, 270          # not laid out yet; produce something
            # Render at the source resolution the real camera would deliver,
            # then shrink for the panel — same two-step the hardware path takes,
            # so the cost profile of the demo matches it.
            aux = None
            if name == "depth":
                src = self.scene.depth(t, 640, 400)
                bgr = imaging.depth_to_bgr(src)
                res = (640, 400)
                enc = "16UC1"
                aux = src            # the probe works in the demo too
            elif name == "second":
                bgr = self.scene.render(t, 640, 400, mono=True)
                res = (640, 400)
                enc = "gray8"
            else:
                bgr = self.scene.render(t, 960, 540)
                res = (960, 540)
                enc = "mjpeg"
            small = imaging.scale_to_fit(bgr, tw, th)
            img = imaging.bgr_to_qimage(small)
            stat = VideoStat(name=name, width=res[0], height=res[1],
                             fps=self.fps, latency_ms=40.0 + 8 * math.sin(t),
                             drop_rate=0.0, mbps=None, encoding=enc,
                             conn=Conn.ONLINE, note="synthetic")
            mb.put(img, stat, aux=aux)
            if self.bus is not None and (t - self._last_pub) > 0.5:
                self.bus.video_stat.emit(stat)
        if (t - self._last_pub) > 0.5:
            self._last_pub = t


# =============================================================================
# synthetic vehicle
# =============================================================================
class DemoVehicleWorker(TimerWorker):
    """A toy 4-DOF vehicle, a battery that drains, and the real NIC counters.

    Runs at 20 Hz and publishes at 10 Hz. That split is the point: a real
    telemetry source produces at whatever rate the vehicle sends (50-200 Hz for
    ATTITUDE/RAW_IMU) and the UI must not see one signal per message. Aggregate
    in the worker, publish a snapshot at a human rate.
    """

    def __init__(self, bus, hz: float = 20.0, opts=None):
        super().__init__("demo-vehicle", interval_ms=int(1000 / hz))
        self.bus = bus
        self.hz = hz
        self.opts = opts
        self._t0 = time.monotonic()
        self._last_pub = 0.0
        self._last_link = 0.0
        self.nic = NicMonitor()
        # --mpc: the toy plant also plays "tag localization + autopilot IMU"
        # so the whole closed loop (assembler -> EAOB -> NMPC -> cmd_pilot ->
        # this plant) closes with zero hardware. SYNTHETIC like everything
        # else here; the NavFix carries geometry="demo" so a log can never be
        # mistaken for a pool run.
        self.mpc_on = bool(getattr(opts, "mpc", False))
        # NED world position of the toy vehicle. Starts at the origin — the
        # demo plays the role of a datum'd (first-fix-relative) world, so a
        # bench ENGAGE starts from a sane pose.
        self.x = 0.0
        self.y = 0.0
        self._nav_rng = np.random.default_rng(3)
        self._nu_prev = np.zeros(3)
        # The last synthetic fix, kept so the synthetic OBJECT can be
        # back-projected through the SAME vehicle pose and the SAME capture
        # stamp. Sharing the stamp is the point: it makes `pair_exact` a thing
        # the offline demo actually exercises instead of something only the
        # pool can ever show.
        self._nav_t = None
        self._nav_p = None
        self._nav_R = None
        self._navcfg = None                 # NavConfig, for the real extrinsic
        self._obj_p0 = None                 # object seed, MAP frame
        self._obj_t0 = 0.0
        self.demo_object = str(getattr(opts, "demo_object", "still")
                               or "still")
        if self.mpc_on:
            self.depth0 = 0.4
        else:
            self.depth0 = 0.0
        # Depth of the tag MAT below the surface. The tag world has z=0 at the
        # mat with +z DOWN, so a swimming vehicle sits at NEGATIVE tag z —
        # which is what the real recordings show (-0.54 .. -1.30 m,
        # sessions/nav_runs/*/fixes.csv). Modelling it keeps the demo in the
        # same world as the pool instead of one where the vehicle is under
        # the floor.
        self.mat_depth = 1.4

        # vehicle state
        self.pilot = PilotInput()
        self.enabled = False
        self.vel = np.zeros(4)        # surge, sway, heave, yaw rates
        self.depth = self.depth0
        self.yaw = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.rate_p = 0.0
        self.rate_q = 0.0
        self._imu_seq = 0
        self.mah = 0.0
        self.grip_cmd = 0.0
        self.grip_fb = 0.0
        self.grip_drive = 0.0
        self.lights = 0.0
        self.estopped = False
        self.armed = False
        self.tilt_deg = 0.0
        self.tilt_drive = 0.0
        # Deliberately NOT manual at start: the real vehicle keeps whatever it
        # was left in, and the demo has to be able to reproduce arming into a
        # mode that drives the thrusters by itself.
        self.mode = "STABILIZE"
        self.js_buttons = 0
        self.pose_on = False
        self.pose_locked = False
        self._log: SensorLog | None = None
        self._imu_log: SensorLog | None = None

    # -------------------------------------------------------------- commands
    @Slot(object)
    def set_pilot(self, cmd: PilotInput) -> None:
        self.pilot = cmd if not self.estopped else PilotInput()

    @Slot(float)
    def set_gripper(self, v: float) -> None:
        self.grip_cmd = max(0.0, min(1.0, float(v)))

    @Slot(float)
    def set_gripper_drive(self, direction: float) -> None:
        """Momentary jaw drive (-1 close / 0 idle / +1 open), HELD like the
        real BTN 76/77 path — integrated in tick(). Without this slot the demo
        silently dropped cmd_gripper_drive, so nothing driven through the
        replay mission's jaw channel could ever be seen offline."""
        self.grip_drive = float(direction)

    @Slot(float)
    def set_lights(self, v: float) -> None:
        self.lights = max(0.0, min(1.0, float(v)))

    @Slot(float)
    def set_tilt(self, direction: float) -> None:
        """Held, like the real mount: integrate while the button is down."""
        self.tilt_drive = float(direction)

    @Slot()
    def tilt_center(self) -> None:
        self.tilt_deg = 0.0

    @Slot(int)
    def set_js_buttons(self, mask: int) -> None:
        self.js_buttons = int(mask) & 0xFFFF

    @Slot(str)
    def set_mode(self, name: str) -> None:
        self.mode = str(name).upper()
        self.bus.log.emit("warn", f"demo: flight mode {self.mode}")

    # ------------------------------------------------------- object tracking
    # A synthetic tracker, so the overlay, the click path and the pose log can
    # be developed and tested on a machine with no GPU at all. It follows the
    # same moving rectangle the synthetic scene draws, so a click near the
    # object "locks on" and a click on empty water does not.
    @Slot(bool)
    def set_pose_enabled(self, on: bool) -> None:
        self.pose_on = bool(on)
        if not on:
            self.pose_locked = False

    @Slot(float, float)
    def set_pose_click(self, x: float, y: float) -> None:
        if not self.pose_on:
            return
        cx, cy = self._demo_object_centre()
        self.pose_locked = (abs(x - cx) < 90 and abs(y - cy) < 60)
        self.bus.log.emit(
            "info", f"demo: prompt at ({x:.0f},{y:.0f}) -> "
                    f"{'locked on' if self.pose_locked else 'nothing there'}")

    @Slot()
    def reset_pose(self) -> None:
        self.pose_locked = False

    def _demo_object_centre(self) -> tuple[float, float]:
        """Where the scene's moving box is, in SOURCE (960x540) pixels."""
        t = time.monotonic() - self._t0
        return (0.5 + 0.35 * math.sin(t * 0.25)) * 960.0, 0.55 * 540.0

    def _nav_extrinsic(self):
        """The REAL ``NavConfig.R_t_frd_cam("main")``, cached.

        The demo object is back-projected through the same extrinsic the
        station composes with, so the round trip in
        ``MpcWorker.on_pose`` -> ``object_nav.compose_map_pose`` is exercised
        end to end offline. Fabricated like the rest of this module — what is
        real here is the ALGEBRA, not the object.
        """
        if self._navcfg is None:
            from ..control.geometry import NavConfig
            self._navcfg = NavConfig.load(
                getattr(self.opts, "nav_config", "config/hw_nav.yaml"),
                geometry_override=getattr(self.opts, "nav_geometry", None))
        return self._navcfg.R_t_frd_cam("main")

    def _demo_object_map(self, t: float):
        """``(p_obj_map, R_map_obj)`` — the synthetic object in the MAP frame.

        Seeded 0.5 m down the camera's OWN optical axis at the first call, so
        it always starts in front of the lens and inside object_nav's distance
        gate whatever the extrinsic's tilt happens to be. After that it is a
        map-frame object with a life of its own:

            still   sits there (the first follow to fly: the vehicle must not
                    move at all when the offset is captured from its own pose)
            drift   translates slowly (the feedforward case)
            orbit   rotates in place (the orbit case, and the one that shows a
                    wrong yaw_axis fastest)
        """
        R_bc, t_bc = self._nav_extrinsic()
        R_nb = self._nav_R
        R_mc = R_nb @ R_bc
        p_cam = self._nav_p + R_nb @ t_bc
        if self._obj_p0 is None:
            self._obj_p0 = p_cam + 0.5 * R_mc[:, 2]
            self._obj_t0 = t
        dt = t - self._obj_t0
        p = np.asarray(self._obj_p0, float).copy()
        yaw = 0.0
        if self.demo_object == "drift":
            p[0] += 0.15 * math.sin(0.15 * dt)
            p[1] += 0.10 * math.sin(0.10 * dt)
        elif self.demo_object == "orbit":
            yaw = 0.30 * dt
        c, sn = math.cos(yaw), math.sin(yaw)
        R_map_obj = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
        return p, R_map_obj, R_mc, p_cam

    def _publish_pose(self) -> None:
        if not self.pose_on:
            self.bus.pose.emit(PoseTrack(state="off", note="TRACK is off",
                                         conn=Conn.OFFLINE, stamp=now()))
            return
        if not self.pose_locked:
            self.bus.pose.emit(PoseTrack(state="idle", src_w=960, src_h=540,
                                         note="click an object",
                                         conn=Conn.ONLINE, stamp=now()))
            return
        cx, cy = self._demo_object_centre()
        w, h = 70.0, 46.0
        box = ((cx - w, cy - h), (cx + w, cy - h), (cx + w, cy + h),
               (cx - w, cy + h))
        t = time.monotonic() - self._t0
        ca, sa = math.cos(t * 0.5), math.sin(t * 0.5)
        axes = ((cx, cy), (cx + 55 * ca, cy + 55 * sa),
                (cx - 40 * sa, cy + 40 * ca), (cx, cy + 46))
        # THE 6-DoF POSE, and the pixels above, are two independent fictions:
        # the contours/axes drive the video overlay and the click path, the
        # transform below drives object_nav. Only the second one has to be
        # geometrically consistent, and with --mpc it is — a MAP-frame object
        # back-projected through the SAME vehicle pose, the SAME real camera
        # extrinsic and the SAME capture stamp the synthetic NavFix used. That
        # is what makes the composition, the pairing and a follow mission
        # testable with no hardware. Without --mpc there is no vehicle pose to
        # project through, so the old free-floating transform is kept.
        t_cap = now()
        if self.mpc_on and self._nav_R is not None:
            p_obj, R_map_obj, R_mc, p_cam = self._demo_object_map(t)
            R_co = R_mc.T @ R_map_obj
            t_co = R_mc.T @ (p_obj - p_cam)
            T = (R_co[0, 0], R_co[0, 1], R_co[0, 2], t_co[0],
                 R_co[1, 0], R_co[1, 1], R_co[1, 2], t_co[1],
                 R_co[2, 0], R_co[2, 1], R_co[2, 2], t_co[2],
                 0.0, 0.0, 0.0, 1.0)
            T = tuple(float(v) for v in T)
            t_cap = float(self._nav_t)
        else:
            z = 0.55 + 0.05 * math.sin(t * 0.4)
            T = (ca, -sa, 0.0, (cx - 480.0) / 960.0 * 0.6,
                 sa, ca, 0.0, (cy - 270.0) / 540.0 * 0.4,
                 0.0, 0.0, 1.0, z,
                 0.0, 0.0, 0.0, 1.0)
        self.bus.pose.emit(PoseTrack(
            state="tracking", contours=(box,), T_cam_obj=T, axes_px=axes,
            # The demo emits at the panel's own rate, so the two numbers are
            # written the way a real run reads them: an ARRIVAL rate capped by
            # the feed, and a solve time well under the frame interval.
            score=7.1, mask_px=int(4 * w * h), sam_hz=10.0, sam_solve_ms=12.0,
            pose_hz=10.0, pose_solve_ms=23.0,
            n_register=1, src_w=960, src_h=540, t_capture=t_cap,
            note=f"synthetic ({self.demo_object})", conn=Conn.ONLINE,
            stamp=now()))

    @Slot(bool, str)
    def set_sensor_log(self, on: bool, stem: str) -> None:
        if on and self._log is None:
            self._log = SensorLog(Path(f"{stem}_demo.jsonl"), "demo")
            self.bus.log.emit("warn", f"demo sensor log -> {self._log.path} "
                                      "(SYNTHETIC — no measurement value)")
        elif not on and self._log is not None:
            meta = self._log.close()
            self._log = None
            self.bus.log.emit("info", f"demo sensor log closed: "
                                      f"{meta['records']} records")

    @Slot(bool, str)
    def set_raw_sensor_log(self, on: bool, stem: str) -> None:
        """The synthetic ``*_c3_imu.jsonl`` a controller run keeps.

        Same file NAME and same record shape the hardware writes, so
        ``plot_imu_dr --from-jsonl`` — the offline re-estimation that is the
        whole payoff of logging every sample — can be exercised here instead
        of first being tried on a pool recording that cost an evening.
        SYNTHETIC, like everything else in this module.
        """
        if on and self._imu_log is None:
            self._imu_log = SensorLog(Path(f"{stem}_c3_imu.jsonl"), "c3")
            self.bus.log.emit(
                "warn", f"demo c3 imu log -> {self._imu_log.path} "
                        "(SYNTHETIC — no measurement value)")
        elif not on and self._imu_log is not None:
            meta = self._imu_log.close()
            self._imu_log = None
            self.bus.log.emit("info", f"demo c3 imu log closed: "
                                      f"{meta['records']} samples")

    @Slot(bool)
    def set_enabled(self, on: bool) -> None:
        self.enabled = bool(on)
        if not on:
            self.pilot = PilotInput()
            self.grip_drive = 0.0     # a held jaw drive dies with the enable

    @Slot(bool)
    def set_arm(self, arm: bool) -> None:
        if arm and not self.enabled:
            self.bus.log.emit("warn", "demo: ARM ignored — COMMAND ENABLE is off")
            return
        self.armed = bool(arm)
        self.bus.log.emit("warn" if arm else "info",
                          f"demo: vehicle {'ARMED' if arm else 'disarmed'}")

    @Slot()
    def estop(self) -> None:
        self.estopped = True
        self.pilot = PilotInput()
        self.vel[:] = 0.0
        self.grip_drive = 0.0         # the real sink zeroes it too (estop())
        self.bus.log.emit("warn", "demo: E-STOP — all axes zeroed")

    # ------------------------------------------------------------------ step
    def tick(self) -> None:
        dt = 1.0 / self.hz
        t = time.monotonic() - self._t0

        cmd = self.pilot if (self.enabled and self.armed) else PilotInput()
        u = np.array([cmd.surge, cmd.sway, cmd.heave, cmd.yaw])
        # A stabilising mode drives the thrusters ITSELF once armed, with no
        # pilot input — that is the whole point of the mode, and out of the
        # water it reads as "the motors started on their own". Modelled here so
        # the demo can reproduce the 2026-08-07 report and so the ARM-into-
        # MANUAL fix is testable without a vehicle.
        if self.armed and self.mode not in ("MANUAL", "ACRO"):
            u = u + np.array([0.0, 0.0,
                              0.22 * math.sin(t * 0.8),
                              0.14 * math.sin(t * 0.5)])
        # First-order drag response. Not a hydrodynamic model and not claiming
        # to be one: the real plant lives in bluerov2_mujoco_marinegym/.
        self.vel += (u * 1.2 - self.vel * 1.6) * dt
        self.depth = max(0.0, self.depth - self.vel[2] * dt * 0.8)
        self.yaw = (self.yaw + self.vel[3] * dt * 1.4) % (2 * math.pi)
        # World (NED-like) position: yaw as psi, surge/sway as u/v body FRD.
        self.x += (self.vel[0] * math.cos(self.yaw)
                   - self.vel[1] * math.sin(self.yaw)) * dt
        self.y += (self.vel[0] * math.sin(self.yaw)
                   + self.vel[1] * math.cos(self.yaw)) * dt
        roll0, pitch0 = self.roll, self.pitch
        self.roll = 0.03 * math.sin(t * 0.9) + 0.05 * self.vel[1]
        self.pitch = 0.02 * math.sin(t * 0.6) - 0.04 * self.vel[0]
        # Body rates that MATCH that attitude. Until 2026-08-17 this plant
        # reported p = q = 0 while roll and pitch swung through a few degrees
        # — kinematically impossible, and invisible for years because every
        # consumer took roll/pitch straight off the message. A dead reckoner
        # INTEGRATES the rates, so it saw the gravity vector tilt under an
        # attitude estimate that had been told nothing was turning, and read
        # the difference as horizontal acceleration. Small-angle, so the Euler
        # coupling between (p, q) and (roll_dot, pitch_dot) is negligible here.
        self.rate_p = (self.roll - roll0) / max(dt, 1e-6)
        self.rate_q = (self.pitch - pitch0) / max(dt, 1e-6)
        if self.mpc_on:
            self._publish_nav(dt)

        thrust = DEMO_MIX @ u
        load = float(np.abs(thrust).sum())
        current = 1.8 + load * 3.2 + self.lights * 7.5
        self.mah += current * 1000.0 * dt / 3600.0
        volt = 16.8 - 0.09 * current - self.mah / 1400.0
        pct = max(0.0, min(100.0, (volt - 13.2) / (16.8 - 13.2) * 100.0))

        # gripper: first-order lag towards the command, i.e. it takes time and
        # can therefore visibly disagree with the command.
        # A held momentary DRIVE (cmd_gripper_drive: G/H keys, the replay
        # mission) integrates the command like the real jaw moves while the
        # button bit is held — the hardware path this stands in for is
        # hardware.py's BTN 76/77. +1 = open, -1 = close, 0.9/s = the panel's
        # simulated rate (payload.GRIPPER_RATE).
        if self.grip_drive:
            self.grip_cmd = max(0.0, min(1.0,
                                         self.grip_cmd
                                         + self.grip_drive * 0.9 * dt))
        self.grip_fb += (self.grip_cmd - self.grip_fb) * min(1.0, dt * 2.5)
        # mount tilt: a rate, not a position — held buttons drive it and it
        # stops against its stops, like the real one.
        self.tilt_deg = max(-45.0, min(45.0, self.tilt_deg + self.tilt_drive * 30.0 * dt))

        if self._log is not None:
            self._log.write("SIM_STATE", {
                "roll": self.roll, "pitch": self.pitch, "yaw": self.yaw,
                "depth_m": self.depth, "tilt_deg": self.tilt_deg,
                "vel": [float(v) for v in self.vel],
            })

        if (t - self._last_pub) < 0.1:
            return
        self._last_pub = t
        self._publish(thrust, current, volt, pct)
        self._publish_pose()

    def _publish_nav(self, dt: float) -> None:
        """Synthetic NavFix + VehicleImu, every tick (20 Hz, camera-like).

        The noise floor (1 cm position, 0.5 deg yaw) is in the ballpark the
        real single-tag PnP is EXPECTED to produce — close enough for the
        loop's behaviour to be representative, fabricated like everything
        else in this module."""
        from ..control.state_assembler import rot_zyx

        R = rot_zyx(self.roll, self.pitch, self.yaw)
        p = (np.array([self.x, self.y, self.depth - self.mat_depth])
             + self._nav_rng.normal(0.0, 0.01, 3))
        yaw_n = self.yaw + self._nav_rng.normal(0.0, math.radians(0.5))
        Rn = rot_zyx(self.roll, self.pitch, yaw_n)
        t = now()
        # Kept for the synthetic object: it is back-projected through the
        # vehicle's TRUE pose (so the fix's own noise flows through the
        # composition, as it will at the pool) but stamped with the SAME
        # t_capture, which is what a shared camera frame means.
        self._nav_t, self._nav_p, self._nav_R = t, np.asarray(
            [self.x, self.y, self.depth - self.mat_depth], float), R
        self.bus.nav_fix.emit(NavFix(
            t_capture=t, n_tags=1, tag_ids=(25,),
            p_ned=tuple(float(v) for v in p),
            R_ned_body=tuple(float(v) for v in Rn.ravel()),
            yaw_ned=float(math.atan2(Rn[1, 0], Rn[0, 0])),
            reproj_rms_px=0.4, detect_ms=2.0, geometry="demo",
            src_w=960, src_h=540, conn=Conn.ONLINE,
            note="synthetic", stamp=t))
        # Body velocity for the accel FD: u, v from the plant, w from the
        # depth integrator's own gain (see tick()).
        nu = np.array([self.vel[0], self.vel[1], -0.8 * self.vel[2]])
        a = (nu - self._nu_prev) / max(1e-6, dt)
        self._nu_prev = nu.copy()
        g = np.array([0.0, 0.0, 9.80665])
        # Specific force consistent with what the assembler inverts:
        # it computes a = f + R^T g - omega x v, so the plant must emit
        # f = (dnu/dt) + omega x v - R^T g. Level & still ~ (0, 0, -9.81).
        omega = np.array([self.rate_p, self.rate_q,
                          float(self.vel[3] * 1.4)])
        f = a + np.cross(omega, nu) - R.T @ g
        self.bus.vehicle_imu.emit(VehicleImu(
            roll=self.roll, pitch=self.pitch, yaw=self.yaw,
            p=float(self.rate_p), q=float(self.rate_q),
            r=float(self.vel[3] * 1.4),
            ax=float(f[0]), ay=float(f[1]), az=float(f[2]),
            depth_m=float(self.depth), t_att=t, t_imu=t, t_baro=t,
            conn=Conn.ONLINE, stamp=t))
        self._publish_camera_imu(f, omega, t, dt)

    # Demo C3 IMU: samples per 20 Hz tick, i.e. 200 Hz like the real BNO086.
    # The point is not realism, it is that the dead reckoner sees the SAME
    # specific force the plant is using — so a bias injected here must come
    # back out of the estimate as exactly 0.5*b*t^2 and the test can say so.
    IMU_SUBSAMPLES = 10

    def _publish_camera_imu(self, f, omega, t: float, dt: float) -> None:
        n = self.IMU_SUBSAMPLES
        b_a = np.asarray(getattr(self.opts, "demo_imu_bias", None)
                         or (0.0, 0.0, 0.0), float)
        b_g = np.asarray(getattr(self.opts, "demo_imu_gyro_bias", None)
                         or (0.0, 0.0, 0.0), float)
        arr = np.zeros((n, 7))
        # Held over the tick rather than interpolated: the plant only knows
        # one acceleration per step, and pretending to a smoother one would
        # make the demo flatter than the hardware it stands in for.
        arr[:, 0] = t - dt + (np.arange(1, n + 1) * (dt / n))
        arr[:, 1:4] = np.asarray(f, float) + b_a
        arr[:, 4:7] = np.asarray(omega, float) + b_g
        self.bus.camera_imu.emit(ImuBatch(
            source="c3", samples=arr, n=n, accuracy="MEDIUM",
            t_host_drain=t, t_device_last=float(arr[-1, 0]),
            conn=Conn.ONLINE, stamp=t))
        if self._imu_log is not None:
            for k in range(n):
                self._imu_log.write("IMU", {
                    "t_device": float(arr[k, 0]), "seq": self._imu_seq,
                    "ax": float(arr[k, 1]), "ay": float(arr[k, 2]),
                    "az": float(arr[k, 3]), "gx": float(arr[k, 4]),
                    "gy": float(arr[k, 5]), "gz": float(arr[k, 6]),
                    "accel_accuracy": "MEDIUM", "gyro_accuracy": "HIGH",
                }, t=float(arr[k, 0]))
                self._imu_seq += 1

    def _leak_now(self) -> bool:
        """Simulated flooding, for looking at the alert before trusting it.

        Off unless ``--demo-leak-after S`` was passed, and DEMO ONLY: there is
        no path from this flag to the hardware backend, whose leak state comes
        from ArduSub's STATUSTEXT and nothing else. It exists because an alarm
        nobody has ever seen fire is an alarm nobody knows the look of.
        """
        after = getattr(self.opts, "demo_leak_after", None)
        if after is None or float(after) < 0:
            return False
        return (now() - self._t0) >= float(after)

    def _publish(self, thrust, current, volt, pct) -> None:
        tel = Telemetry(
            battery_v=volt, battery_pct=pct, battery_pct_source="vehicle",
            battery_left_mah=max(0.0, 18000.0 - self.mah),
            current_a=current, consumed_mah=self.mah,
            roll=self.roll, pitch=self.pitch, yaw=self.yaw,
            depth_m=self.depth, heading_deg=math.degrees(self.yaw),
            water_temp_c=14.2, internal_temp_c=31.0,
            armed=self.armed,
            mode=f"{self.mode} (sim)",
            leak=self._leak_now(),
            sensors={
                # Two IMUs, named for their owners: the autopilot's and the
                # C3's BNO086. Different devices, different clocks, different
                # rates — the panel keeps them apart for the same reason the
                # hardware backend does.
                "IMU (ROV)": SensorStat("IMU (ROV)", 200.0, Conn.ONLINE),
                "IMU (C3)": SensorStat("IMU (C3)", 486.0, Conn.ONLINE, "BNO086"),
                "Barometer": SensorStat("Barometer", 10.0, Conn.ONLINE),
                "Compass": SensorStat("Compass", 20.0, Conn.ONLINE),
                "Leak": (SensorStat("Leak", None, Conn.FAULT,
                                    "LEAK DETECTED (simulated)")
                         if self._leak_now() else
                         SensorStat("Leak", None, Conn.ONLINE, "dry (sim)")),
                "Enclosure": SensorStat(
                    "Enclosure", None,
                    Conn.FAULT if self._leak_now() else Conn.ONLINE,
                    # "(sim)" on both, like the Leak row: a bare hPa figure
                    # beside a real-looking panel is exactly the synthetic
                    # number that gets quoted later as a measurement.
                    "1103 hPa internal (sim)  CRITICAL" if self._leak_now()
                    else "1013 hPa internal (sim)"),
                "EKF": SensorStat("EKF", None, Conn.ONLINE, "ok"),
                "Heartbeat": SensorStat("Heartbeat", 1.0, Conn.ONLINE),
            },
            conn=Conn.ONLINE, stamp=now())
        self.bus.telemetry.emit(tel)

        n = len(thrust)
        self.bus.thrusters.emit(ThrusterState(
            n=n,
            norm=[float(v) for v in thrust],
            pwm_us=[int(1500 + 400 * max(-1.0, min(1.0, float(v)))) for v in thrust],
            health=[Conn.ONLINE] * n,
            labels=[f"T{i + 1}" for i in range(n)],
            conn=Conn.ONLINE, stamp=now()))

        self.bus.payload.emit(PayloadState(
            gripper_cmd=self.grip_cmd, gripper_fb=self.grip_fb,
            gripper_current_a=abs(self.grip_cmd - self.grip_fb) * 3.0,
            gripper_conn=Conn.ONLINE,
            lights_cmd=self.lights, lights_fb=self.lights,
            lights_conn=Conn.ONLINE,
            tilt_deg=self.tilt_deg, tilt_drive=self.tilt_drive,
            tilt_conn=Conn.ONLINE, tilt_note="simulated mount",
            stamp=now()))

        # Real host counters, at 1 Hz. The only honest numbers in this module.
        t = time.monotonic()
        if (t - self._last_link) > 1.0:
            self._last_link = t
            self.bus.link.emit(self.nic.sample())


class DemoBackend(Backend):
    name = "demo"
    simulated = True

    def __init__(self, bus, mailboxes, opts):
        super().__init__(bus, mailboxes, opts)
        fps = float(getattr(opts, "fps", 15.0) or 15.0)
        self.video = DemoVideoWorker(mailboxes, fps=fps)
        self.video.bus = bus
        self.vehicle = DemoVehicleWorker(bus, opts=opts)
        self.workers = [self.video, self.vehicle]
        # --mpc closes the loop against the toy plant: the REAL MpcWorker
        # (same code as hardware; the acados solver builds at startup), fed by
        # the synthetic NavFix/VehicleImu the vehicle worker publishes.
        self.mpc = None
        if bool(getattr(opts, "mpc", False)):
            from ..control.workers import MpcWorker

            self.mpc = MpcWorker(bus, opts)
            self.workers.append(self.mpc)
            bus.nav_fix.connect(self.mpc.on_nav_fix)
            # The object tracker's second consumer — see the hardware backend.
            bus.pose.connect(self.mpc.on_pose)
            bus.vehicle_imu.connect(self.mpc.on_vehicle_imu)
            bus.camera_imu.connect(self.mpc.on_camera_imu)
            bus.telemetry.connect(self.mpc.on_telemetry)
            bus.thrusters.connect(self.mpc.on_thrusters)
            bus.cmd_mpc_engage.connect(self.mpc.set_engaged)
            bus.cmd_mpc_traj.connect(self.mpc.set_traj)
            bus.cmd_mpc_start.connect(self.mpc.start_mission)
            bus.cmd_mpc_mode.connect(self.mpc.set_mode)
            bus.cmd_mpc_scenario.connect(self.mpc.set_scenario)
            bus.cmd_mpc_dump_meta.connect(self.mpc.dump_run_meta)
            bus.cmd_enable.connect(self.mpc.on_enable)
            bus.cmd_estop.connect(self.mpc.estop)
            bus.cmd_log_sensors.connect(self.mpc.set_sensor_log)

        # Commands go straight to the simulated vehicle. Connections are made
        # before moveToThread on purpose: Qt resolves AutoConnection at emit
        # time from the receiver's *current* thread affinity, so this becomes a
        # queued connection automatically once the worker is on its thread.
        bus.cmd_pilot.connect(self.vehicle.set_pilot)
        bus.cmd_gripper.connect(self.vehicle.set_gripper)
        bus.cmd_gripper_drive.connect(self.vehicle.set_gripper_drive)
        bus.cmd_lights.connect(self.vehicle.set_lights)
        bus.cmd_enable.connect(self.vehicle.set_enabled)
        bus.cmd_estop.connect(self.vehicle.estop)
        bus.cmd_arm.connect(self.vehicle.set_arm)
        bus.cmd_tilt.connect(self.vehicle.set_tilt)
        bus.cmd_tilt_center.connect(self.vehicle.tilt_center)
        bus.cmd_buttons.connect(self.vehicle.set_js_buttons)
        bus.cmd_log_sensors.connect(self.vehicle.set_sensor_log)
        bus.cmd_log_raw_sensors.connect(
            self.vehicle.set_raw_sensor_log)
        bus.cmd_mode.connect(self.vehicle.set_mode)
        bus.cmd_pose_enable.connect(self.vehicle.set_pose_enabled)
        bus.cmd_pose_click.connect(self.vehicle.set_pose_click)
        bus.cmd_pose_reset.connect(self.vehicle.reset_pose)

    def describe(self) -> str:
        return "demo — synthetic vehicle and cameras; NO measurement value"
