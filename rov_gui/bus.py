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
from .state import (Conn, LinkStat, PayloadState, PilotInput, PoseTrack,
                    Telemetry, ThrusterState, VideoStat, now)


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
    pose = Signal(object)             # state.PoseTrack
    # Closed-loop MPC (rov_gui/control/). nav_fix is the AprilTag localization
    # at camera rate; vehicle_imu is the autopilot's inertial state at the MPC
    # tick rate; mpc_status is one row per control tick. They ride the bus and
    # not a mailbox because they are small immutable snapshots at <= 20-30 Hz —
    # exactly the traffic shape DataBus is for (module docstring, mechanism 1).
    nav_fix = Signal(object)          # state.NavFix
    vehicle_imu = Signal(object)      # state.VehicleImu
    mpc_status = Signal(object)       # state.MpcStatus
    tag_overlay = Signal(object)      # state.TagOverlay (per video feed)

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
    # Object tracking. `cmd_pose_click` carries SOURCE image pixels, not panel
    # pixels — the canvas converts before emitting, so nothing downstream needs
    # to know how the video happened to be letterboxed.
    cmd_pose_enable = Signal(bool)
    cmd_pose_click = Signal(float, float)
    cmd_pose_reset = Signal()
    # Closed-loop MPC commands. ENGAGE and START TRAJ are two separate gates on
    # purpose: engaging only makes the MPC hold the CURRENT pose (DP), and the
    # trajectory clock starts on an explicit second action once the hold has
    # settled — so the vehicle never lunges for a distant square corner the
    # moment a ring-fill button completes.
    cmd_mpc_engage = Signal(bool)     # True = engage (DP hold), False = release
    cmd_mpc_traj = Signal(bool)       # True = start the square, False = back to DP
    # The one-button mission flow: engage (if needed), warm up, then start the
    # square by itself. Recording opens at engage as always, so one press ==
    # "fly the square and log it". Kept SEPARATE from cmd_mpc_engage so the
    # DP-only hold (calibration, station-keeping tests) still exists.
    cmd_mpc_start = Signal()
    cmd_mpc_mode = Signal(str)        # "mpc" | "dobmpc"
    cmd_mpc_scenario = Signal(object) # dict of square overrides (size, speed, ...)
    # Per-feed AprilTag detection toggle: (video panel key, on). Turning the
    # C3 feed off mid-engagement starves the localizer and the MPC disengages
    # on the stale-fix gate — that is the intended, safe consequence.
    cmd_tag_enable = Signal(str, bool)


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
        self._aux = None
        self._target = (0, 0)
        self._put_count = 0
        self._take_count = 0
        self._conflated = 0

    # ------------------------------------------------------------- producer
    def put(self, image: QImage, stat: VideoStat, aux=None) -> None:
        """Publish a frame. Any thread. Overwrites whatever was pending.

        ``aux`` rides along with the frame it belongs to — for the depth panel
        it is the raw uint16 millimetre map, which is what lets the cursor
        read out a real distance instead of a colour. It travels IN the
        mailbox, not beside it, so the numbers under the cursor can never come
        from a different frame than the picture. Must already be a copy the
        producer will not touch again (DepthAI recycles its buffers).
        """
        self._mutex.lock()
        try:
            if self._image is not None:
                self._conflated += 1      # the previous one never got drawn
            self._image = image
            stat.conflated = self._conflated
            self._stat = stat
            self._aux = aux
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
    def take(self):
        """Take the pending frame as (image, stat, aux), or (None, None, None).

        Returning None is the normal case at a paint rate above the frame rate,
        and it means "keep showing what you have" — not "the feed died". Only
        :class:`Freshness` decides that.
        """
        self._mutex.lock()
        try:
            img, stat, aux = self._image, self._stat, self._aux
            if img is None:
                return None, None, None  # nothing new; never hand back a repeat
            self._image = None
            self._aux = None
            self._take_count += 1
            return img, stat, aux
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


class RgbdMailbox:
    """Latest RGB-D frame for the perception worker. One slot, newest wins.

    :class:`FrameMailbox`'s sibling, and it exists for the same reason: the
    camera produces at 30 fps while the tracker consumes at whatever the GPU
    manages, and connecting those with a queue converts a rate mismatch into
    unbounded, unrecoverable latency. Here the consequence would be worse than a
    late picture — a pose computed from a frame ten seconds old is a pose that
    describes where the object used to be.

    Carries numpy, not QImage, and therefore carries the ONE rule that matters:
    the producer must hand over arrays it will not touch again. DepthAI recycles
    its frame pool, so :meth:`put` is called with copies (see
    ``C3VideoWorker._tap_pose``); this class cannot enforce that, so it is
    written down at both ends.
    """

    def __init__(self):
        self._mutex = QMutex()
        self._item: dict | None = None
        self._put = 0
        self._taken = 0
        self._conflated = 0
        # The consumer's appetite, visible to the producer. The video worker
        # checks this BEFORE copying: with TRACK off the pose worker discards
        # every item unread, so copying 1.15 MB per frame to feed it was pure
        # waste — and contradicted the documented promise that tracking off
        # costs nothing.
        self._wanted = False

    def set_wanted(self, on: bool) -> None:
        self._mutex.lock()
        try:
            self._wanted = bool(on)
            if not on:
                self._item = None       # drop what nobody will ever read
        finally:
            self._mutex.unlock()

    def wanted(self) -> bool:
        self._mutex.lock()
        try:
            return self._wanted
        finally:
            self._mutex.unlock()

    def put(self, color, depth, intrinsics, t_capture: float,
            skew_ms: float = 0.0) -> None:
        """Publish a frame. Any thread. Overwrites whatever was pending."""
        self._mutex.lock()
        try:
            if self._item is not None:
                self._conflated += 1        # the previous one was never used
            self._put += 1
            self._item = {"color": color, "depth": depth, "K": intrinsics,
                          "t_capture": t_capture, "skew_ms": skew_ms,
                          "seq": self._put}
        finally:
            self._mutex.unlock()

    def take(self) -> dict | None:
        """Take the pending frame, or None. Never hands back a repeat."""
        self._mutex.lock()
        try:
            item, self._item = self._item, None
            if item is not None:
                self._taken += 1
            return item
        finally:
            self._mutex.unlock()

    def counters(self) -> dict[str, int]:
        self._mutex.lock()
        try:
            return {"put": self._put, "taken": self._taken,
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


__all__ = ["DataBus", "FrameMailbox", "RgbdMailbox", "Freshness", "Conn",
           "LinkStat", "PayloadState", "PilotInput", "PoseTrack", "Telemetry",
           "ThrusterState", "VideoStat"]
