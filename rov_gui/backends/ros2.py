#!/usr/bin/env python3
"""
ros2.py — the ROS 2 seam.

Not installed on this machine (no ``/opt/ros``, no ``rclpy`` in the ``robust``
env as of 2026-08-06), so this module is written to be *correct by
construction* and is exercised only by its import guard. It is the integration
point the rest of the package was shaped around, and the shape is the useful
part: when rclpy does arrive, nothing outside this file changes.

How ROS 2 and Qt are made to coexist
------------------------------------
They are two event loops that must not block each other, and neither may touch
the other's objects.

* rclpy spins in **its own thread** (a :class:`~rov_gui.backends.base.LoopWorker`
  calling ``spin_once`` with a timeout, so the loop can notice a stop request).
  Never ``rclpy.spin()`` on the GUI thread: it blocks until shutdown and the UI
  freezes at startup.
* Subscription callbacks run on the executor thread. They may touch **only**
  the mailboxes (mutex-protected) and the bus (queued signals). Calling a widget
  method from a ROS callback is the crash this architecture exists to prevent.
* Commands travel the other way through :class:`_CommandBox`, a lock-protected
  struct: the Qt signal is connected to a plain function, so it runs on the GUI
  thread and does nothing but take a lock and assign. A ROS timer reads it at a
  fixed rate and publishes. The GUI thread therefore never calls into rclpy.

QoS
---
Image topics use ``SensorDataQoS`` — BEST_EFFORT, depth 1. That is the ROS-side
twin of the FrameMailbox: on a congested tether you want the newest frame, not
a reliable replay of an old one. Telemetry uses the default reliable QoS,
because dropping a battery reading to save bandwidth saves nothing.

Topic names below are DEFAULTS, not a standard. Remap them (``--ros-args -r``)
or pass ``--ros-topic-*``; nothing in this file assumes a particular vehicle
stack.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from .. import imaging
from ..state import (Conn, PayloadState, PilotInput, SensorStat, Telemetry,
                     ThrusterState, VideoStat, now)
from .base import Backend, LoopWorker

DEFAULT_TOPICS = {
    "main": "/camera/main/image_raw",
    "second": "/stereo/left/image_raw",
    "depth": "/stereo/depth/image_raw",
    "battery": "/vehicle/battery",
    "imu": "/vehicle/imu",
    "odom": "/vehicle/odom",
    "thrusters": "/vehicle/thruster_outputs",
    "cmd_vel": "/vehicle/cmd_vel",
    "gripper": "/payload/gripper/command",
    "lights": "/payload/lights/command",
    "arm": "/vehicle/arm",
}

IMPORT_HELP = """rov_gui --source ros2 needs rclpy on the PYTHONPATH.

    source /opt/ros/<distro>/setup.bash
    python -m rov_gui --source ros2

Note that a conda env (this repo's `robust`) and a sourced ROS 2 install rarely
share an interpreter: ROS 2 python packages are built against the system python.
Run the GUI from the ROS-sourced interpreter, or build rclpy into the env — do
not mix, or you get an ABI crash inside rclpy at the first spin."""


class _CommandBox:
    """Lock-protected latest command. Written by Qt, read by the ROS thread."""

    def __init__(self):
        self._lock = threading.Lock()
        self.pilot = PilotInput()
        self.gripper = 0.0
        self.lights = 0.0
        self.enabled = False
        self.arm_request: bool | None = None

    def set_pilot(self, cmd: PilotInput) -> None:
        with self._lock:
            self.pilot = cmd

    def set_gripper(self, v: float) -> None:
        with self._lock:
            self.gripper = float(v)

    def set_lights(self, v: float) -> None:
        with self._lock:
            self.lights = float(v)

    def set_enabled(self, on: bool) -> None:
        with self._lock:
            self.enabled = bool(on)
            if not on:
                self.pilot = PilotInput()

    def estop(self) -> None:
        with self._lock:
            self.enabled = False
            self.pilot = PilotInput()

    def set_arm(self, arm: bool) -> None:
        with self._lock:
            self.arm_request = bool(arm)

    def snapshot(self) -> tuple[PilotInput, float, float, bool]:
        with self._lock:
            return self.pilot, self.gripper, self.lights, self.enabled

    def take_arm_request(self) -> bool | None:
        """Read-and-clear: an arm request is an event, not a level."""
        with self._lock:
            req, self.arm_request = self.arm_request, None
            return req


def image_to_bgr(msg) -> np.ndarray | None:
    """sensor_msgs/Image -> BGR uint8 (or the raw uint16 depth map).

    Deliberately not cv_bridge: cv_bridge drags in a compiled dependency that
    is version-locked to both OpenCV and the ROS distro, and this conversion is
    six lines. ``step`` is honoured because a padded row is exactly the bug
    that produces a sheared image nobody can explain.
    """
    h, w, step = msg.height, msg.width, msg.step
    enc = msg.encoding.lower()
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if enc in ("bgr8", "rgb8"):
        arr = buf.reshape(h, step)[:, : w * 3].reshape(h, w, 3)
        return arr[:, :, ::-1] if enc == "rgb8" else arr
    if enc in ("mono8", "8uc1"):
        return buf.reshape(h, step)[:, :w]
    if enc in ("16uc1", "mono16"):
        arr16 = buf.view(np.uint16).reshape(h, step // 2)[:, :w]
        return arr16                     # caller colourises; still millimetres
    return None


class Ros2Worker(LoopWorker):
    """One node, one executor, one thread."""

    def __init__(self, bus, mailboxes, opts, box: _CommandBox):
        super().__init__("ros2")
        self.bus = bus
        self.mailboxes = mailboxes
        self.opts = opts
        self.box = box
        self.node = None
        self.exec_ = None
        self._rclpy = None
        self._pubs: dict = {}
        self._counts: dict[str, int] = {}
        self._rate_windows: dict[str, list] = {}
        self._last_stat = 0.0
        self.topics = dict(DEFAULT_TOPICS)
        for key in self.topics:
            override = getattr(opts, f"ros_topic_{key}", None)
            if override:
                self.topics[key] = override

    # ----------------------------------------------------------------- setup
    def setup(self) -> None:
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import BatteryState, Image, Imu
            from std_msgs.msg import Bool, Float32, Float32MultiArray
            from geometry_msgs.msg import Twist
        except ImportError as e:                                 # noqa: BLE001
            raise RuntimeError(f"{e}\n\n{IMPORT_HELP}") from e

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = rclpy.create_node("rov_gui_station")

        def sub(msg_type, topic, cb, qos):
            self.node.create_subscription(msg_type, topic, cb, qos)
            self.bus.log.emit("info", f"ros2: subscribed {topic}")

        for panel in ("main", "second", "depth"):
            if panel in self.mailboxes:
                sub(Image, self.topics[panel],
                    lambda m, p=panel: self._on_image(p, m),
                    qos_profile_sensor_data)
        sub(BatteryState, self.topics["battery"], self._on_battery, 10)
        sub(Imu, self.topics["imu"], self._on_imu, qos_profile_sensor_data)
        sub(Float32MultiArray, self.topics["thrusters"], self._on_thrusters, 10)

        self._pubs = {
            "arm": self.node.create_publisher(Bool, self.topics["arm"], 10),
            "cmd_vel": self.node.create_publisher(Twist, self.topics["cmd_vel"], 10),
            "gripper": self.node.create_publisher(Float32, self.topics["gripper"], 10),
            "lights": self.node.create_publisher(Float32, self.topics["lights"], 10),
        }
        self._Twist, self._Float32, self._Bool = Twist, Float32, Bool
        # Publishing runs on a ROS timer, on this thread: the GUI thread never
        # calls into rclpy.
        self.node.create_timer(0.05, self._publish_commands)      # 20 Hz
        self.node.create_timer(0.1, self._publish_telemetry)      # 10 Hz

        self.exec_ = SingleThreadedExecutor()
        self.exec_.add_node(self.node)

        # Latest values, filled by callbacks on this same thread.
        self._battery = None
        self._imu = None
        self._thrusters = None

    # ------------------------------------------------------------------ loop
    def run(self) -> None:
        while not self.stopping:
            # A timeout rather than spin(): spin() returns only at shutdown, and
            # then stop() has nothing to interrupt.
            self.exec_.spin_once(timeout_sec=0.1)

    def teardown(self) -> None:
        if self.exec_ is not None:
            self.exec_.shutdown()
            self.exec_ = None
        if self.node is not None:
            self.node.destroy_node()
            self.node = None
        if self._rclpy is not None and self._rclpy.ok():
            self._rclpy.shutdown()

    # ------------------------------------------------------------- callbacks
    def _on_image(self, panel: str, msg) -> None:
        arr = image_to_bgr(msg)
        if arr is None:
            self.bus.log.emit("warn", f"ros2: unsupported encoding {msg.encoding}")
            return
        is_depth = arr.dtype == np.uint16
        bgr = imaging.depth_to_bgr(arr) if is_depth else arr
        mb = self.mailboxes[panel]
        tw, th = mb.target_size()
        small = imaging.scale_to_fit(bgr, tw, th)
        self._counts[panel] = self._counts.get(panel, 0) + 1

        stamp = msg.header.stamp
        # Header stamps are the CAPTURE time. Latency measured against the host
        # clock is only meaningful if the two clocks are synced (chrony/PTP);
        # if they are not, this reads as a constant offset, not as jitter.
        age_ms = (time.time() - (stamp.sec + stamp.nanosec * 1e-9)) * 1000.0
        mb.put(imaging.bgr_to_qimage(small),
               VideoStat(name=panel, width=msg.width, height=msg.height,
                         fps=self._rate(panel), latency_ms=age_ms,
                         encoding=msg.encoding, conn=Conn.ONLINE, stamp=now()))

    def _rate(self, panel: str) -> float:
        """Rate over the last second, measured here rather than taken on trust.

        Per panel, not one shared counter: the three feeds run at different
        rates and a shared "last rate" would report whichever stream ticked
        most recently on all three overlays.
        """
        window = self._rate_windows.setdefault(panel, [time.monotonic(), 0, 0.0])
        prev_t, prev_n, last = window
        t, n = time.monotonic(), self._counts.get(panel, 0)
        if t - prev_t >= 1.0:
            last = (n - prev_n) / (t - prev_t)
            self._rate_windows[panel] = [t, n, last]
        return last

    def _on_battery(self, msg) -> None:
        self._battery = msg

    def _on_imu(self, msg) -> None:
        self._imu = msg

    def _on_thrusters(self, msg) -> None:
        self._thrusters = list(msg.data)

    # -------------------------------------------------------------- publish
    def _publish_telemetry(self) -> None:
        b, imu = self._battery, self._imu
        roll = pitch = yaw = None
        if imu is not None:
            q = imu.orientation
            roll, pitch, yaw = _quat_to_rpy(q.x, q.y, q.z, q.w)
        tel = Telemetry(
            battery_v=(b.voltage if b else None),
            battery_pct=((b.percentage * 100.0) if b and b.percentage >= 0 else None),
            current_a=(abs(b.current) if b and b.current == b.current else None),
            roll=roll, pitch=pitch, yaw=yaw,
            heading_deg=(np.degrees(yaw) % 360.0 if yaw is not None else None),
            sensors={
                "IMU": SensorStat("IMU", None,
                                  Conn.ONLINE if imu is not None else Conn.OFFLINE),
                "Battery": SensorStat("Battery", None,
                                      Conn.ONLINE if b is not None else Conn.OFFLINE),
            },
            conn=Conn.ONLINE, stamp=now())
        self.bus.telemetry.emit(tel)

        if self._thrusters:
            n = len(self._thrusters)
            self.bus.thrusters.emit(ThrusterState(
                n=n, norm=[float(v) for v in self._thrusters],
                pwm_us=[None] * n, health=[Conn.ONLINE] * n,
                labels=[f"T{i + 1}" for i in range(n)],
                conn=Conn.ONLINE, stamp=now()))

    def _publish_commands(self) -> None:
        pilot, grip, lights, enabled = self.box.snapshot()
        arm = self.box.take_arm_request()
        if arm is not None:
            # Published once per request, not repeated: arming is an event. A
            # latched "arm" republished at 20 Hz would re-arm a vehicle that a
            # failsafe just disarmed.
            self._pubs["arm"].publish(self._Bool(data=bool(arm)))
        self.bus.payload.emit(PayloadState(
            gripper_cmd=grip, gripper_fb=None, gripper_conn=Conn.ONLINE,
            lights_cmd=lights, lights_fb=None, lights_conn=Conn.ONLINE,
            stamp=now()))
        if not enabled:
            return
        # Deadman: stale input publishes a zero Twist rather than repeating the
        # last one. Same rule as the MAVLink sink, for the same reason.
        stale = (now() - pilot.stamp) > 0.5
        cmd = PilotInput() if stale else pilot
        msg = self._Twist()
        msg.linear.x, msg.linear.y, msg.linear.z = cmd.surge, cmd.sway, cmd.heave
        msg.angular.z = cmd.yaw
        self._pubs["cmd_vel"].publish(msg)
        self._pubs["gripper"].publish(self._Float32(data=float(grip)))
        self._pubs["lights"].publish(self._Float32(data=float(lights)))


def _quat_to_rpy(x: float, y: float, z: float, w: float):
    """Quaternion -> (roll, pitch, yaw) in radians, ZYX convention."""
    sinr = 2 * (w * x + y * z)
    cosr = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    siny = 2 * (w * z + x * y)
    cosy = 1 - 2 * (y * y + z * z)
    return float(roll), float(pitch), float(np.arctan2(siny, cosy))


class Ros2Backend(Backend):
    name = "ros2"
    simulated = False

    def __init__(self, bus, mailboxes, opts):
        super().__init__(bus, mailboxes, opts)
        self.box = _CommandBox()
        self.worker = Ros2Worker(bus, mailboxes, opts, self.box)
        self.workers = [self.worker]
        self.sink = self.box
        # Direct connections to plain functions: they run on the GUI thread and
        # do nothing but take a lock. See the module docstring.
        bus.cmd_pilot.connect(self.box.set_pilot)
        bus.cmd_gripper.connect(self.box.set_gripper)
        bus.cmd_lights.connect(self.box.set_lights)
        bus.cmd_enable.connect(self.box.set_enabled)
        bus.cmd_estop.connect(self.box.estop)
        bus.cmd_arm.connect(self.box.set_arm)

    def describe(self) -> str:
        return f"ros2 — node rov_gui_station, images on {DEFAULT_TOPICS['main']} etc."
