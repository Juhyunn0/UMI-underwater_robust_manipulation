"""rov_gui.backends — where the data actually comes from.

A backend owns threads and hardware; the UI owns widgets; they meet only at
:class:`rov_gui.bus.DataBus` and :class:`rov_gui.bus.FrameMailbox`. That is the
whole architecture, and it is what lets the same window run against:

    demo   synthetic vehicle + synthetic cameras, no hardware, no ROS.
           The default, and what the offline test drives.
    hw     the MarineSitu C3 over DepthAI (``c3_camera.source.C3Source``) plus
           ArduSub telemetry over MAVLink (``c3_camera.mavlink_log``).
    ros2   sensor_msgs/Image + telemetry topics through rclpy.

Adding a fourth is: emit the dataclasses in :mod:`rov_gui.state` onto the bus,
put QImages in the mailboxes, accept the command signals. No widget code changes.
"""

from __future__ import annotations

from .base import Backend, LoopWorker, TimerWorker

SOURCES = ("demo", "hw", "ros2")


def make_backend(source: str, bus, mailboxes, opts) -> Backend:
    """Construct the backend named by ``source``. Imports are deliberately lazy.

    ``hw`` pulls in depthai and pymavlink; ``ros2`` pulls in rclpy. Importing
    either at module level would make the demo path — the one that has to work
    on any laptop — fail on a machine that has neither.
    """
    if source == "demo":
        from .demo import DemoBackend
        return DemoBackend(bus, mailboxes, opts)
    if source == "hw":
        from .hardware import HardwareBackend
        return HardwareBackend(bus, mailboxes, opts)
    if source == "ros2":
        from .ros2 import Ros2Backend
        return Ros2Backend(bus, mailboxes, opts)
    raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")


__all__ = ["Backend", "LoopWorker", "TimerWorker", "make_backend", "SOURCES"]
