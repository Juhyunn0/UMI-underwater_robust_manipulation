#!/usr/bin/env python3
"""
bus.py — the only place threads meet.

Three mechanisms, each for a different traffic shape. Picking the wrong one is
how Qt dashboards end up laggy or crashy, so the reasoning is written down
rather than implied.

1. :class:`DataBus` — queued signals, for *snapshots*
    Low-rate, small, immutable payloads (telemetry, thruster state, link stats).
    A cross-thread ``emit`` posts an event to the receiver's event loop, so the
    slot runs on the GUI thread and may touch widgets. Cheap at 10-20 Hz.
    It is NOT cheap at 200 Hz with a 2 MB payload, which is why video does not
    use it.

2. :class:`FrameMailbox` — a one-slot conflating buffer, for *video*
    The camera produces frames on its own schedule; the screen consumes them at
    the refresh rate. Connecting those with a signal per frame builds an
    unbounded queue the moment the GUI falls behind — and an unbounded queue in
    a video path does not just use memory, it adds latency that never comes
    back, so the pilot ends up flying a picture from ten seconds ago. A mailbox
    keeps exactly the newest frame and counts what it dropped, so latency is
    bounded by construction and the drop count is visible instead of silent.

    Producer calls :meth:`FrameMailbox.put` from any thread; the GUI's paint
    timer calls :meth:`FrameMailbox.take`. ``QImage`` is safe to build off the
    GUI thread (it is implicitly shared data, not a QObject); ``QPixmap`` is
    NOT — it can touch the window system — so workers hand over QImages only.

3. :class:`Freshness` — a watchdog, for *absence*
    Nothing emits "I stopped working". A thread that deadlocks, a camera that
    stops sending, a tether that is cut — all of them look exactly like silence.
    So the UI never trusts a "connected" flag on its own: every panel ages its
    last update against a per-source timeout, and silence turns it amber then
    red on its own. This is the single most important behaviour in the file,
    because it is what makes the "Disconnected" indicators trustworthy.

Command traffic (GUI → vehicle) rides the same bus in the other direction, and
the same rule applies as for telemetry: the GUI thread must never touch a socket
or a device handle. It emits a command signal; the worker's slot, running in the
worker's own thread, does the blocking work.
"""

from __future__ import annotations

from collections import deque

from .qt import QImage, QMutex, QObject, Signal
from .state import Conn, LinkStat, PayloadState, PilotInput, Telemetry, ThrusterState, VideoStat, now


class DataBus(QObject):
    """Signals only. Owned by the GUI thread; emitted from anywhere.

    Payloads are the dataclasses in :mod:`rov_gui.state` and are treated as
    immutable once emitted. ``object`` rather than a typed signature keeps this
    binding-agnostic and avoids registering meta-types.
    """

    # ---- telemetry (backend -> UI) ----------------------------------------
    telemetry = Signal(object)        # state.Telemetry
    thrusters = Signal(object)        # state.ThrusterState
    payload = Signal(object)          # state.PayloadState
    link = Signal(object)             # state.LinkStat
    video_stat = Signal(object)       # state.VideoStat
    aux_servos = Signal(object)       # {channel: pwm_us} for 9..16 (payload)
    source_state = Signal(str, object)  # (source name, state.Conn)
    log = Signal(str, str)            # (level, message) -> status bar + stdout
    # One sensor row from a worker that is NOT the vehicle worker. The vehicle's
    # own sensors ride inside Telemetry, but the C3's IMU is a different device
    # on a different thread and a different clock, and routing it through the
    # vehicle worker would imply a relationship between them that does not
    # exist. The window merges these into the same panel by name.
    sensor_stat = Signal(object)      # state.SensorStat

    # ---- commands (UI -> backend) -----------------------------------------
    # Connected with Qt.QueuedConnection so the send happens in the worker's
    # thread even though the click happened in the GUI's.
    cmd_pilot = Signal(object)        # state.PilotInput
    cmd_gripper = Signal(float)       # 0 closed .. 1 open
    cmd_lights = Signal(float)        # 0 .. 1
    cmd_estop = Signal()              # zero everything, now
    cmd_enable = Signal(bool)         # transmission master switch
    # True = arm, False = disarm. A separate signal from cmd_enable on purpose:
    # "this station may command" and "the vehicle's motors are live" are
    # different facts, and collapsing them means one click does both.
    cmd_arm = Signal(bool)
    # ArduSub's joystick functions come in two flavours and they are NOT
    # interchangeable, so they get two signals:
    #   cmd_gripper_drive  servo_1_{min,max}_momentary — acts while HELD
    #   cmd_lights_step    lights1_{dimmer,brighter}   — one step per PRESS
    # Sending a level where a press is expected steps once and then sits there;
    # sending a press where a hold is expected twitches the jaw and stops.
    cmd_gripper_drive = Signal(float)   # -1 close / 0 idle / +1 open
    cmd_lights_step = Signal(int)       # +n brighter, -n dimmer (presses)
    # Camera mount tilt. HELD like the gripper, not stepped like the lights:
    # ArduSub services mount_tilt_up/down from its button-REPEAT path, so the
    # mount moves for as long as the bit stays set.
    cmd_tilt = Signal(float)            # -1 down / 0 hold / +1 up
    cmd_tilt_center = Signal()          # one press of mount_center
    # The gamepad's OWN button bitmask, forwarded verbatim so the vehicle's
    # BTNn_FUNCTION parameters decide what each button does — the same contract
    # QGC has. Without this the station had a second, private button map and a
    # pad button did one thing in QGC and something else here.
    cmd_buttons = Signal(int)
    # Flight mode by ArduSub name ("MANUAL", "STABILIZE", "ALT_HOLD", ...).
    # The mode decides how the vehicle interprets every axis AND whether it
    # drives the thrusters on its own, so it is the pilot's most consequential
    # switch after ARM — which is exactly why it is here rather than implied.
    cmd_mode = Signal(str)
    # Start/stop the raw sensor log that accompanies a depth recording.
    # (enabled, path stem) — the stem matches the video file it belongs to.
    cmd_log_sensors = Signal(bool, str)


class FrameMailbox:
    """Latest-frame-wins hand-off for one video stream.

    Not a QObject: it is deliberately signal-free so that a producer running
    flat out cannot flood the GUI event loop. The consumer polls it from a
    paint timer.

    ``target_size`` is the reverse channel. The panel publishes the pixel size
    it is about to draw into; the worker reads it and scales there, on the
    worker thread. Scaling 1920x1080 down to a 640-wide panel is genuinely
    expensive, and doing it in ``paintEvent`` is doing it on the one thread that
    must never be busy.
    """

    def __init__(self, name: str):
        self.name = name
        self._mutex = QMutex()
        self._image: QImage | None = None
        self._stat: VideoStat | None = None
        self._target = (0, 0)
        self._put_count = 0
        self._take_count = 0
        self._conflated = 0

    # ------------------------------------------------------------- producer
    def put(self, image: QImage, stat: VideoStat) -> None:
        """Publish a frame. Any thread. Overwrites whatever was pending."""
        self._mutex.lock()
        try:
            if self._image is not None:
                self._conflated += 1      # the previous one never got drawn
            self._image = image
            stat.conflated = self._conflated
            self._stat = stat
            self._put_count += 1
        finally:
            self._mutex.unlock()

    def target_size(self) -> tuple[int, int]:
        """The panel's current draw size, for the worker to scale to."""
        self._mutex.lock()
        try:
            return self._target
        finally:
            self._mutex.unlock()

    # ------------------------------------------------------------- consumer
    def take(self) -> tuple[QImage | None, VideoStat | None]:
        """Take the pending frame, or (None, None) if nothing new arrived.

        Returning None is the normal case at a paint rate above the frame rate,
        and it means "keep showing what you have" — not "the feed died". Only
        :class:`Freshness` decides that.
        """
        self._mutex.lock()
        try:
            img, stat = self._image, self._stat
            if img is None:
                return None, None      # nothing new; never hand back a repeat
            self._image = None
            self._take_count += 1
            return img, stat
        finally:
            self._mutex.unlock()

    def set_target_size(self, w: int, h: int) -> None:
        self._mutex.lock()
        try:
            self._target = (int(w), int(h))
        finally:
            self._mutex.unlock()

    def counters(self) -> dict[str, int]:
        self._mutex.lock()
        try:
            return {"put": self._put_count, "taken": self._take_count,
                    "conflated": self._conflated}
        finally:
            self._mutex.unlock()


class Freshness:
    """Age-based connection state for one source.

    ``warn_s`` / ``fail_s`` are per-source because the sources are not
    comparable: a 15 fps video feed is late after a third of a second, while
    BATTERY_STATUS at 2 Hz is perfectly healthy three seconds later. Using one
    global timeout means either the video panel lies or the battery panel
    blinks red at random.
    """

    def __init__(self, warn_s: float, fail_s: float):
        self.warn_s = warn_s
        self.fail_s = fail_s
        self.last: float | None = None
        self.reported: Conn = Conn.OFFLINE   # what the source itself claimed
        self._history: deque[float] = deque(maxlen=64)

    def mark(self, stamp: float | None = None, reported: Conn | None = None) -> None:
        t = now() if stamp is None else stamp
        self.last = t
        self._history.append(t)
        if reported is not None:
            self.reported = reported

    @property
    def age(self) -> float | None:
        return None if self.last is None else now() - self.last

    @property
    def hz(self) -> float:
        if len(self._history) < 2:
            return 0.0
        span = self._history[-1] - self._history[0]
        return (len(self._history) - 1) / span if span > 0 else 0.0

    def state(self) -> Conn:
        """The verdict the UI paints. Silence wins over any claimed state."""
        if self.reported in (Conn.FAULT, Conn.CONNECTING) and self.last is None:
            return self.reported
        age = self.age
        if age is None:
            return Conn.OFFLINE
        if age > self.fail_s:
            return Conn.STALE
        if age > self.warn_s:
            return Conn.DEGRADED
        # Fresh data, but the source may still be telling us it is unhappy.
        if self.reported in (Conn.DEGRADED, Conn.FAULT):
            return self.reported
        return Conn.ONLINE


__all__ = ["DataBus", "FrameMailbox", "Freshness", "Conn", "LinkStat",
           "PayloadState", "PilotInput", "Telemetry", "ThrusterState", "VideoStat"]
