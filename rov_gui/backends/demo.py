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
from ..state import (Conn, PayloadState, PilotInput, SensorStat, Telemetry,
                     ThrusterState, VideoStat, now)
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
            if name == "depth":
                src = self.scene.depth(t, 640, 400)
                bgr = imaging.depth_to_bgr(src)
                res = (640, 400)
                enc = "16UC1"
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
            mb.put(img, stat)
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

    def __init__(self, bus, hz: float = 20.0):
        super().__init__("demo-vehicle", interval_ms=int(1000 / hz))
        self.bus = bus
        self.hz = hz
        self._t0 = time.monotonic()
        self._last_pub = 0.0
        self._last_link = 0.0
        self.nic = NicMonitor()

        # vehicle state
        self.pilot = PilotInput()
        self.enabled = False
        self.vel = np.zeros(4)        # surge, sway, heave, yaw rates
        self.depth = 0.0
        self.yaw = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.mah = 0.0
        self.grip_cmd = 0.0
        self.grip_fb = 0.0
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
        self._log: SensorLog | None = None

    # -------------------------------------------------------------- commands
    @Slot(object)
    def set_pilot(self, cmd: PilotInput) -> None:
        self.pilot = cmd if not self.estopped else PilotInput()

    @Slot(float)
    def set_gripper(self, v: float) -> None:
        self.grip_cmd = max(0.0, min(1.0, float(v)))

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

    @Slot(bool)
    def set_enabled(self, on: bool) -> None:
        self.enabled = bool(on)
        if not on:
            self.pilot = PilotInput()

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
        self.roll = 0.03 * math.sin(t * 0.9) + 0.05 * self.vel[1]
        self.pitch = 0.02 * math.sin(t * 0.6) - 0.04 * self.vel[0]

        thrust = DEMO_MIX @ u
        load = float(np.abs(thrust).sum())
        current = 1.8 + load * 3.2 + self.lights * 7.5
        self.mah += current * 1000.0 * dt / 3600.0
        volt = 16.8 - 0.09 * current - self.mah / 1400.0
        pct = max(0.0, min(100.0, (volt - 13.2) / (16.8 - 13.2) * 100.0))

        # gripper: first-order lag towards the command, i.e. it takes time and
        # can therefore visibly disagree with the command.
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
            leak=False,
            sensors={
                # Two IMUs, named for their owners: the autopilot's and the
                # C3's BNO086. Different devices, different clocks, different
                # rates — the panel keeps them apart for the same reason the
                # hardware backend does.
                "IMU (ROV)": SensorStat("IMU (ROV)", 200.0, Conn.ONLINE),
                "IMU (C3)": SensorStat("IMU (C3)", 486.0, Conn.ONLINE, "BNO086"),
                "Barometer": SensorStat("Barometer", 10.0, Conn.ONLINE),
                "Compass": SensorStat("Compass", 20.0, Conn.ONLINE),
                "Leak": SensorStat("Leak", None, Conn.ONLINE, "dry"),
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
        self.vehicle = DemoVehicleWorker(bus)
        self.workers = [self.video, self.vehicle]

        # Commands go straight to the simulated vehicle. Connections are made
        # before moveToThread on purpose: Qt resolves AutoConnection at emit
        # time from the receiver's *current* thread affinity, so this becomes a
        # queued connection automatically once the worker is on its thread.
        bus.cmd_pilot.connect(self.vehicle.set_pilot)
        bus.cmd_gripper.connect(self.vehicle.set_gripper)
        bus.cmd_lights.connect(self.vehicle.set_lights)
        bus.cmd_enable.connect(self.vehicle.set_enabled)
        bus.cmd_estop.connect(self.vehicle.estop)
        bus.cmd_arm.connect(self.vehicle.set_arm)
        bus.cmd_tilt.connect(self.vehicle.set_tilt)
        bus.cmd_tilt_center.connect(self.vehicle.tilt_center)
        bus.cmd_buttons.connect(self.vehicle.set_js_buttons)
        bus.cmd_log_sensors.connect(self.vehicle.set_sensor_log)
        bus.cmd_mode.connect(self.vehicle.set_mode)

    def describe(self) -> str:
        return "demo — synthetic vehicle and cameras; NO measurement value"
