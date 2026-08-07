"""rov_gui — topside control-station dashboard for the BlueROV2 (PyQt).

One window, one screen, no scrolling: every module the pilot needs is visible at
all times, and every module says out loud whether its data is live, stale, or
gone. Nothing here owns hardware — the panels render whatever a *backend* feeds
onto the data bus, so the same UI runs against synthetic data, against the real
C3 + ArduSub link, or against ROS 2 topics.

Layout
------
    qt.py          the Qt binding shim, plus the cv2/Qt plugin-path landmine
    theme.py       dark palette + stylesheet, and the connection-state colours
    state.py       the dataclasses that cross the thread boundary (plain data)
    bus.py         DataBus signals, FrameMailbox (frame conflation), Freshness
    net.py         tether/fiber link health read from the topside NIC (sysfs)
    recorder.py    UI screen recording: grab on the GUI thread, encode on another
    window.py      MainWindow — the QGridLayout that must fit one screen
    widgets/       indicators, video, health, propulsion, payload, teleop
    backends/      demo (synthetic) | hw (C3 + MAVLink) | ros2 (seam)

Run
---
    python -m rov_gui                 # synthetic data, no hardware touched
    python -m rov_gui --source hw     # real C3 + ArduSub (runs preflight first)
    python -m rov_gui --source ros2   # ROS 2 topics (needs rclpy)

Command posture
---------------
The teleop panel is a command *source*, and this repo's standing rule (see
``c3_camera/control.py``) is that nothing transmits to the vehicle unless a
human explicitly enables it: two command sources at once is the hazard. So the
GUI ships with a NullCommandSink, the COMMAND ENABLE switch is off at startup,
and the real MAVLink sink is a stub that refuses to construct. Demo mode routes
commands to the simulated vehicle instead, so the whole path is still testable.
"""

__version__ = "0.1.0"

__all__ = ["qt", "theme", "state", "bus", "net", "recorder", "window",
           "widgets", "backends"]
