#!/usr/bin/env python3
"""
state.py — the plain data that crosses the thread boundary.

Everything a backend hands to the UI is one of these frozen-ish dataclasses:
no Qt types, no numpy views into a buffer someone else is about to overwrite, no
device handles. That is what makes the hand-off safe. A worker thread builds a
snapshot, emits it, and never touches it again; the GUI thread owns it from
then on.

Two conventions that the panels rely on:

* ``stamp`` is ``time.monotonic()`` at the moment the *data* was true (not when
  it was emitted). The freshness watchdog ages every panel off this, which is
  how a silently wedged worker still turns the panel red.
* Unknown is ``None``, never 0.0 and never a plausible-looking default. A panel
  renders ``None`` as "--". A dashboard that shows 0.0 V for "I have no idea" is
  worse than one that shows nothing, and this repo has already been bitten once
  by a placeholder number being read back as a measurement
  (docs/MEASUREMENT_AUDIT.md).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class Conn(Enum):
    """Connection/health state, in the order the UI escalates through them."""

    OFFLINE = "offline"        # never seen, or explicitly gone
    CONNECTING = "connecting"  # opening, handshaking, waiting for first data
    ONLINE = "online"          # fresh data, within spec
    DEGRADED = "degraded"      # data arriving but wrong: drops, low rate, errors
    STALE = "stale"            # was online, nothing recently — watchdog verdict
    FAULT = "fault"            # the source reported a failure

    @property
    def ok(self) -> bool:
        return self is Conn.ONLINE

    @property
    def bad(self) -> bool:
        return self in (Conn.OFFLINE, Conn.STALE, Conn.FAULT)


def now() -> float:
    """The one clock everything in this package ages against."""
    return time.monotonic()


# =============================================================================
# video
# =============================================================================
@dataclass
class VideoStat:
    """What the overlay on a video panel shows. One per stream."""

    name: str
    width: int = 0
    height: int = 0
    fps: float = 0.0
    latency_ms: float | None = None
    drop_rate: float = 0.0          # 0..1, device+host drops
    mbps: float | None = None       # link cost of THIS stream
    # False = counted on the wire. True = DERIVED (tether total minus the
    # streams we can count), which the overlay prefixes with "~" so it is never
    # read as a measurement. See CLAUDE.md on measurement provenance.
    mbps_estimated: bool = False
    conflated: int = 0              # frames the UI skipped to stay current
    encoding: str = ""
    conn: Conn = Conn.OFFLINE
    note: str = ""
    stamp: float = field(default_factory=now)

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}" if self.width else "--"


# =============================================================================
# vehicle telemetry
# =============================================================================
@dataclass
class SensorStat:
    """One auxiliary sensor's liveness. ``hz`` is measured, not requested."""

    name: str
    hz: float | None = None
    conn: Conn = Conn.OFFLINE
    detail: str = ""


@dataclass
class Telemetry:
    """The vehicle's own state, in SI, already converted out of MAVLink units."""

    # power
    battery_v: float | None = None
    battery_pct: float | None = None     # 0..100
    # WHERE the percentage came from, because the three sources are not equally
    # trustworthy and a bare number hides that:
    #   "vehicle"  the autopilot's own BATTERY_STATUS.battery_remaining
    #   "mah"      DERIVED: (capacity - consumed) / capacity
    #   "volts"    DERIVED: pack voltage against a nominal cell curve, which
    #              reads LOW under thrust because the pack sags
    # The panel labels the derived ones so they are never quoted as the
    # vehicle's own figure (CLAUDE.md, docs/MEASUREMENT_AUDIT.md).
    battery_pct_source: str = ""
    battery_left_mah: float | None = None
    current_a: float | None = None
    consumed_mah: float | None = None
    # attitude (radians) and depth (metres, positive down)
    roll: float | None = None
    pitch: float | None = None
    yaw: float | None = None
    depth_m: float | None = None
    heading_deg: float | None = None
    water_temp_c: float | None = None
    internal_temp_c: float | None = None
    # flight controller
    armed: bool | None = None
    mode: str = ""
    leak: bool | None = None
    # auxiliary sensors, keyed by name
    sensors: dict[str, SensorStat] = field(default_factory=dict)
    conn: Conn = Conn.OFFLINE
    stamp: float = field(default_factory=now)


# =============================================================================
# propulsion
# =============================================================================
@dataclass
class ThrusterState:
    """Per-motor output and health.

    ``norm`` is signed −1..+1 (reverse..forward) because a unipolar bar cannot
    show a thruster that is pushing backwards, and on a vectored ROV that is
    half of normal operation. ``pwm_us`` is the raw servo output when the
    vehicle reports it, so the two can be cross-checked rather than assumed
    consistent.
    """

    n: int = 8
    norm: list[float] = field(default_factory=list)
    pwm_us: list[int | None] = field(default_factory=list)
    health: list[Conn] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    conn: Conn = Conn.OFFLINE
    # Measured arrival rate of the message these numbers came from. Shown on the
    # panel because "the motors are at neutral" and "nobody has told us what the
    # motors are doing" look identical otherwise — and on 2026-08-07 they were
    # confused for each other while the motors were actually turning.
    hz: float | None = None
    stamp: float = field(default_factory=now)

    @classmethod
    def blank(cls, n: int = 8, labels: list[str] | None = None) -> "ThrusterState":
        return cls(n=n,
                   norm=[0.0] * n,
                   pwm_us=[None] * n,
                   health=[Conn.OFFLINE] * n,
                   labels=labels or [f"T{i + 1}" for i in range(n)])


# =============================================================================
# payload
# =============================================================================
@dataclass
class PayloadState:
    """Gripper and lights: commanded value, measured value, and link state.

    ``gripper_fb`` is separate from ``gripper_cmd`` on purpose. The stock
    BlueROV2 gripper is an open-loop servo with no position feedback, so on the
    real vehicle ``gripper_fb`` is None and the panel says "cmd only" instead of
    drawing a feedback bar that is really just the command echoed back. Showing
    a command as if it were feedback is how a jammed jaw goes unnoticed.
    """

    gripper_cmd: float = 0.0             # 0 closed .. 1 open
    gripper_fb: float | None = None
    gripper_current_a: float | None = None
    gripper_conn: Conn = Conn.OFFLINE
    lights_cmd: float = 0.0              # 0 .. 1
    lights_fb: float | None = None
    lights_conn: Conn = Conn.OFFLINE
    # Camera mount tilt. ``tilt_deg`` is the vehicle's OWN report (MOUNT_STATUS
    # pointing_a, or the tilt servo's PWM converted to degrees) and stays None
    # when the vehicle does not report one — the same rule as gripper_fb: a
    # commanded direction is not a measured angle, and a mount that has hit its
    # stop looks identical to one still moving if you draw the command.
    tilt_deg: float | None = None
    tilt_drive: float = 0.0              # -1 down / 0 idle / +1 up, as commanded
    tilt_conn: Conn = Conn.OFFLINE
    tilt_note: str = ""
    # Free text from the backend about HOW the device is controlled on this
    # vehicle ("hold", "stepped, no feedback", "no button bit mapped"). The
    # panel prints it instead of implying a position that does not exist.
    gripper_note: str = ""
    lights_note: str = ""
    stamp: float = field(default_factory=now)


# =============================================================================
# tether / fiber link
# =============================================================================
@dataclass
class LinkStat:
    """Topside end of the tether, as the operating system sees it.

    On a fiber tether the topside media converter presents itself to the desktop
    as an ordinary Ethernet interface, so ``carrier``/``speed_mbps`` *are* the
    converter's link status — if the fiber drops or a connector is dirty, the
    copper side goes down or renegotiates, and the error counters climb before
    it does. That is the honest signal available without polling the converter
    over SNMP; ``rtt_ms`` (a TCP connect probe to the ROV) covers the rest of
    the path down the tether.
    """

    iface: str = ""
    up: bool | None = None
    speed_mbps: int | None = None        # negotiated link speed, not throughput
    rx_mbps: float | None = None
    tx_mbps: float | None = None
    rx_err_per_s: float | None = None
    tx_err_per_s: float | None = None
    rx_drop_per_s: float | None = None
    rtt_ms: float | None = None
    peer: str = ""
    conn: Conn = Conn.OFFLINE
    note: str = ""
    stamp: float = field(default_factory=now)


# =============================================================================
# object tracking / pose
# =============================================================================
@dataclass
class PoseTrack:
    """What the object tracker knows, in plain data.

    Everything here is already in SOURCE image pixels (the 640x360 stream), so
    the overlay never does geometry and never needs the camera model — the same
    division of labour the rest of this package uses: the worker computes, the
    GUI thread blits.

    ``contours`` rather than a mask, deliberately. A 640x360 bool mask is 230 kB
    per frame and would have to be drawn pixel by pixel on the GUI thread;
    ``cv2.findContours`` + ``approxPolyDP`` turn it into ~1 kB of polylines that
    QPainter draws as a path. Same picture, no per-pixel work where it hurts.

    ``T_cam_obj`` is 16 floats, ROW-MAJOR, object -> camera, metres, OpenCV
    optical axes (+X right, +Y down, +Z forward). None until a pose exists —
    and in the tracking-only build it stays None forever, which is a legitimate,
    useful state and not a failure.
    """

    # off | loading | idle | live | capturing | building | pose_loading |
    # registering | tracking | lost | failed | fault
    state: str = "off"
    contours: tuple = ()            # ((x, y), ...) per polyline, source pixels
    T_cam_obj: tuple | None = None  # 16 floats row-major, or None
    axes_px: tuple = ()             # 4 points: origin, +X tip, +Y tip, +Z tip
    box_px: tuple = ()              # 8 corners of the oriented box
    score: float | None = None      # SAM2 object-score logit; >0 visible
    mask_px: int = 0
    # TWO DIFFERENT QUESTIONS, and they were one number until 2026-08-23.
    #   *_hz        MEASURED update rate: how often a new mask / a new pose
    #               actually arrives (perception/session.py _RateMeter).
    #   *_solve_ms  how long the GPU takes on ONE frame — the upstream
    #               trackers' own `hz`, inverted back into the milliseconds it
    #               always was.
    # They differ whenever the pipeline is not the bottleneck (a 20 ms solve
    # fed 13 frames a second is 13 Hz, not 50) and they differ WILDLY across a
    # re-registration, which costs ~0.7 s of solve and produces exactly one
    # pose. A record written before 2026-08-23 has the old meaning in `pose_hz`
    # — the pose CSV renamed its column to say so.
    sam_hz: float = 0.0
    sam_solve_ms: float = 0.0
    pose_hz: float = 0.0
    pose_solve_ms: float = 0.0
    n_register: int = 0
    frame_seq: int = 0
    # The frame these pixels belong to. The overlay scales by this, so a change
    # of stream resolution mid-session cannot silently misplace the mask.
    src_w: int = 0
    src_h: int = 0
    load_s: float = 0.0             # how long the model has been loading
    # On-site reconstruction. Only meaningful in the capturing/building states,
    # and they are the two the pilot has to steer: the arc is a budget they
    # spend by orbiting (75 deg, past which the per-view pose error goes off a
    # cliff), and build_s is a wait with nothing to do but not lose the object.
    n_views: int = 0
    arc_deg: float = 0.0
    max_arc: float = 0.0
    max_views: int = 0
    build_s: float = 0.0
    distance_m: float = 0.0         # camera->object during capture, metres
    # HOW SMEARED THE DEPTH IS, as a multiple of the object's own size — the
    # measurement that predicts whether the reconstruction will come out long
    # (perception/session.py:_frame_depth_quality). None means "not enough
    # pixels to say". Live during capture, so the pilot can fix it by moving
    # closer while the orbit is still happening rather than reading it in the
    # summary after the mesh is already wrong.
    smear_ratio: float | None = None
    smear_max: float = 0.0          # the frame gate this is judged against
    fp_load_s: float = 0.0          # FoundationPose bring-up, seconds so far
    # Whether a 6-DoF pose is expected at all. Without this the overlay cannot
    # tell "no pose because you asked for a mask" from "no pose because
    # something went wrong", and those need to look different.
    pose_expected: bool = False
    # WHEN THE FRAME THIS POSE CAME FROM WAS TRUE. Not ``stamp`` (which is
    # when the snapshot was built) and not 0.0. It is the same float
    # ``C3VideoWorker._tap_pose`` put into the NAV mailbox for the same colour
    # frame, which is what lets ``object_nav`` pair a pose with the tag fix
    # from that exact frame and cancel the camera extrinsic out of the
    # composition. Without it the pairing has nothing to compare.
    t_capture: float = 0.0
    note: str = ""
    conn: Conn = Conn.OFFLINE
    stamp: float = field(default_factory=now)

    @property
    def has_mask(self) -> bool:
        return bool(self.contours)


# =============================================================================
# closed-loop MPC (AprilTag localization + acados NMPC) — see rov_gui/control/
# =============================================================================
@dataclass
class NavFix:
    """One AprilTag-PnP localization of the VEHICLE, already in the NED world.

    The frame chain (tag map -> camera -> body -> NED) is resolved by
    ``rov_gui.control.tagnav`` + ``rov_gui.control.geometry`` BEFORE this
    object exists, so everything downstream (the MPC, the plot, the CSV) reads
    one convention and cannot re-derive it differently:

    * ``p_ned`` / ``R_ned_body`` — vehicle position and body(FRD)->world(NED)
      rotation in the tag-map-derived NED world frame (x north-ish, z DOWN).
      For the floor map that world is the tag-25 frame (already +Z down); for
      the wall-tag geometry it is the configured remap of the tag frame.
    * ``t_capture`` — when the camera FRAME was true (stamp rule from the
      module docstring), not when PnP finished. Latency compensation and the
      freshness gate both hang off this.
    * ``ambiguous`` — single-tag IPPE returned two plausible poses and the
      disambiguation was not decisive; the state assembler may still use the
      fix but the flag rides into the CSV so a bad stretch is auditable.
    """

    t_capture: float = 0.0
    n_tags: int = 0
    tag_ids: tuple = ()
    tag_insts: tuple = ()            # which copy of each id (0 = the only one)
    p_ned: tuple | None = None       # (x, y, z) metres, NED world
    R_ned_body: tuple | None = None  # 9 floats row-major, body(FRD)->world(NED)
    yaw_ned: float | None = None     # rad, extracted from R_ned_body (ZYX psi)
    reproj_rms_px: float | None = None
    detect_ms: float = 0.0
    hz: float | None = None          # measured rate of ACCEPTED fixes
    ambiguous: bool = False
    geometry: str = ""               # "floor" | "wall"
    source: str = ""                 # which feed localized: "main" (C3) | "second"
    src_w: int = 0
    src_h: int = 0
    conn: Conn = Conn.OFFLINE
    note: str = ""
    stamp: float = field(default_factory=now)

    @property
    def ok(self) -> bool:
        return self.p_ned is not None and self.R_ned_body is not None


@dataclass
class ObjectFix:
    """The tracked OBJECT, placed in the pool — one composition of a
    :class:`PoseTrack` with the :class:`NavFix` from the SAME camera frame.

    THE FRAME IS THE MAP FRAME: the tag-map NED world (x north-ish, +z DOWN)
    that the pool rectangle, the tag mat and the trajectory plot are drawn in.
    It is deliberately NOT the engage-datum frame :class:`MpcStatus` uses and
    NOT world FLU. The object exists before anything engages and must not move
    on screen when START is pressed — which is exactly the bug the datum
    conversion caused once already (trajectory.py: "the whole mat visibly
    swung round the moment START was pressed").

    ``pair_dt_ms`` / ``pair_exact`` are the health of the composition itself.
    The camera extrinsic cancels EXACTLY when the object pose and the tag fix
    come from one frame (``pair_exact``); paired across frames, the unmeasured
    0.2855 m camera lever arm re-enters the error budget. A run whose
    ``pair_exact`` ratio is not ~1.0 must not have object-position statistics
    pooled with one whose is.
    """

    t_capture: float = 0.0           # the frame BOTH estimates came from
    ok: bool = False                 # is p_map worth drawing/using?
    state: str = "cold"              # cold | live | stale | lost
    p_map: tuple | None = None       # (x, y, z) metres, MAP frame
    yaw_map: float | None = None     # rad, chosen object axis projected flat
    R_map_obj: tuple | None = None   # 9 floats row-major, object -> map
    v_map: tuple | None = None       # (3,) m/s, MAP frame
    r_map: float | None = None       # rad/s about map +z (DOWN)
    distance_m: float | None = None  # camera -> object, straight off the pose
    age_s: float | None = None       # now - t_capture
    extrapolated_s: float = 0.0      # of that age, how much was extrapolated
    pair_dt_ms: float | None = None  # |t_pose - t_fix|; 0 on an exact pair
    pair_exact: bool = False
    yaw_axis: str = ""               # which object axis carries the heading
    pose_state: str = ""             # the PoseTrack state behind this
    n_obs: int = 0
    n_reject: int = 0
    note: str = ""
    conn: Conn = Conn.OFFLINE
    stamp: float = field(default_factory=now)


@dataclass
class TagOverlay:
    """AprilTag detections on ONE video feed, in that feed's SOURCE pixels.

    Display data only — the pose/localization result rides :class:`NavFix`.
    This exists so the pilot can SEE what the detector sees, per feed, and so
    an uncalibrated camera (the ROV's own RGB) can still show detections
    without pretending they localize anything: ``localizes`` says whether this
    feed's detections feed the MPC state estimate.
    """

    panel: str = ""                  # "main" | "second" — the video panel key
    quads: tuple = ()                # ((4 corner (x,y) tuples), ...) source px
    ids: tuple = ()
    mapped: tuple = ()               # per-quad: is this tag in the map?
    src_w: int = 0
    src_h: int = 0
    detect_ms: float = 0.0
    localizes: bool = False          # True = these detections drive the MPC
    enabled: bool = True             # False = one last CLEAR after toggle-off
    # The camera model and capture time these corners belong to. Present so a
    # recording can carry RAW OBSERVATIONS, not just solved poses: rebuilding
    # or extending the tag map needs corners + K, and a fix alone throws both
    # away. Empty K = this feed has no calibration (overlay-only).
    t_capture: float = 0.0
    K: tuple = ()                    # (fx, fy, cx, cy) for THIS frame's size
    dist: tuple = ()
    note: str = ""
    stamp: float = field(default_factory=now)


@dataclass
class VehicleImu:
    """The autopilot's inertial state, in SI, NED/FRD — the MPC's second sensor.

    All from ArduSub over MAVLink (ATTITUDE / SCALED_IMU2 / SCALED_PRESSURE2),
    which already speaks NED/FRD, so NOTHING here is re-signed. ``ax..az`` are
    SPECIFIC FORCE (what the accelerometer measures): a level, stationary
    vehicle reads (0, 0, -9.81). Gravity removal happens in the state
    assembler, next to the rotation that needs it. Per-record host stamps are
    kept separately because the three messages arrive at different rates and
    one shared stamp would hide a dead stream behind a live one.
    """

    roll: float | None = None        # rad, NED (ATTITUDE)
    pitch: float | None = None
    yaw: float | None = None         # compass yaw — consistency check ONLY,
    p: float | None = None           # rad/s, FRD body (ATTITUDE rollspeed)
    q: float | None = None
    r: float | None = None
    ax: float | None = None          # m/s^2 specific force, FRD (SCALED_IMU2)
    ay: float | None = None
    az: float | None = None
    depth_m: float | None = None     # pressure-DERIVED, positive down
    t_att: float | None = None       # host arrival stamps per message
    t_imu: float | None = None
    t_baro: float | None = None
    conn: Conn = Conn.OFFLINE
    stamp: float = field(default_factory=now)


@dataclass
class ImuBatch:
    """Every C3 BNO086 sample since the last drain — a DIFFERENT sensor from
    ``VehicleImu``, on a different board and a different clock.

    ``samples`` is an (N, 7) float64 array in ``imu_dr.SAMPLE_COLS`` order
    ``(t, ax, ay, az, gx, gy, gz)``: accel m/s^2, gyro rad/s, both in the IMU's
    own axes (the mounting rotation is the estimator's business, not the
    transport's). An array rather than a list of ``c3_camera.imu.ImuSample``
    for two reasons — ``rov_gui.state`` must import with no depthai present,
    and the consumer slices this far more often than it inspects a field.

    ``t`` is the DEVICE timestamp, which on this host shares a clock with the
    image frames and with ``time.monotonic()``. That is the whole reason this
    IMU is worth the wiring, so ``t_host_drain`` and ``t_device_last`` are kept
    side by side: their difference is the running check that the two clocks
    still agree, and a dead reckoner integrating a wrong dt fails silently.
    """

    source: str = "c3"
    samples: object = None           # (N, 7) float64, or None
    n: int = 0
    dropped: int = 0                 # sequence numbers missing in this batch
    accuracy: str = ""               # worst accel/gyro flag in the batch
    t_host_drain: float = 0.0
    t_device_last: float = 0.0
    conn: Conn = Conn.OFFLINE
    stamp: float = field(default_factory=now)


@dataclass
class MpcStatus:
    """One MPC control tick, as the UI and the CSV see it.

    Positions here are WORLD FLU (x fwd, y left, z up) — the sim's recording
    convention — so the TrajectoryWindow and the traj CSV read like the
    simulator's outputs. The controller itself runs in NED; the bridge
    converts at this boundary and nowhere else. ``w_hat``/``u_cmd`` stay in
    the controller's native NED/FRD units (N, N·m) because converting a
    diagnostic invites sign bugs in the thing meant to catch them.
    """

    engaged: bool = False
    traj_on: bool = False
    mode: str = ""                   # "mpc" | "dobmpc"
    # The armed mission, PER SHAPE — a circle carries `radius` and no
    # `size`/`size_y`, a line carries `length`/`dir_deg`. Switch on
    # `scenario["kind"]` before touching a dimension key (meta schema 4).
    scenario: dict | None = None     # square params once the trajectory starts
    solver: str = ""                 # "acados" | "ipopt" | "stub" | ""
    solver_status: int = 0
    solve_ms: float | None = None
    n_fail: int = 0
    w_hat: tuple = ()                # (6,) NED body wrench, N / N·m
    u_cmd: tuple = ()                # (6,) NED body wrench command
    axes: tuple = ()                 # (surge, sway, heave, yaw) sent, -1..+1
    p_flu: tuple | None = None       # measured position, world FLU
    ref_flu: tuple | None = None     # reference position, world FLU
    yaw_flu_deg: float | None = None
    err_xy: float | None = None      # |p - ref| horizontal, metres
    # err_xy split along the path's own tangent, metres (MpcWorker._path_split
    # owns the conventions). err_along is LAG: + = the vehicle is behind the
    # virtual target. err_cross is which SIDE of the path it is on: + = left
    # of the direction of travel. The two
    # answer different questions and only one of them is path following's
    # business: stage 0 deliberately sits ahead of the active-segment
    # projection, so quoting |p - ref| alone hides the result (2026-08-14 line
    # run: 12.5 cm err = 11 cm lag + 1 cm off the line).
    err_cross: float | None = None
    err_along: float | None = None
    # Speed, live. The operator sets a path speed and until 2026-08-17 had no
    # way to see what the vehicle was doing with it — the run that exposed the
    # PID's disconnected speed box (0.2 m/s asked, 0.026 m/s covered) could
    # have been read off the screen in seconds.
    speed_m_s: float | None = None       # measured horizontal ground speed
    ref_speed_m_s: float | None = None   # what the reference is asking for
    n_tags: int = 0
    tag_age_s: float | None = None
    imu_age_s: float | None = None
    warmup_left_s: float = 0.0
    # WHICH part of a mission is running, so the operator never has to guess
    # whether the vehicle is still on its way to the start tag or already
    # flying the path: "" (idle) | "warmup" | "approach" | "settle" |
    # "station" | "line" | "square".
    phase: str = ""
    phase_detail: str = ""           # "0.83 m to go", "6 s", "lap 2/5"
    # The mission datum, set at ENGAGE: (x, y, z, yaw) of the engage pose in
    # the TAG frame. Everything in this status (and the CSV, and the square,
    # and the square) is relative to it — (0,0) is where START was pressed.
    # None = no engagement yet this session.
    datum: tuple | None = None
    lap: int = 0
    t_traj: float | None = None      # seconds since the trajectory clock began
    reason: str = ""                 # why the last engage/disengage happened
    # ---- IMU dead reckoning (the "how far can the IMU carry us" experiment)
    # p_dr_flu is world FLU in the datum frame, exactly like p_flu, so the two
    # can be subtracted without thinking. p_flu stays the TAG solution even
    # when the controller is flying on the dead reckoner: the plot's actual
    # trail and the CSV's px/py must always be ground truth, or the run
    # measures nothing.
    p_dr_flu: tuple | None = None
    yaw_dr_flu_deg: float | None = None
    dr_err_m: float | None = None    # |p_dr - p_tag| horizontal
    dr_err_z_m: float | None = None
    dr_elapsed_s: float | None = None    # since the anchor
    dr_source: str = ""              # "c3"
    dr_attitude: str = ""            # "gyro" | "ahrs" | "vehicle"
    dr_mode: str = ""                # "" | "shadow" | "control"
    dr_hz: float | None = None
    dr_n: int = 0                    # samples integrated since the anchor
    # False whenever the estimate must not be believed — INCLUDING when the
    # sample stream died. A starved dead reckoner looks perfect (a frozen
    # point drifting not at all), so this is the field that has to be loud.
    dr_ok: bool = False
    dr_note: str = ""
    # STATION BRIDGE (control/station_bridge.py). "none" while the tag fix is
    # fresh; "imu" while every axis is being carried on the bridge estimate;
    # "coast" once the horizontal axes have been released and only depth and
    # attitude are still held. `bridge_s` is how long the fix has been gone.
    # A run whose CSV shows a non-zero bridge_s was NOT flying on the tag for
    # those ticks — do not pool them with clean ones.
    bridge_tier: str = "none"
    bridge_s: float = 0.0
    # OBJECT FOLLOW (control/object_nav.py). Three statements about the
    # CONTROL LOOP, which is why they ride here and the object's own position
    # rides :class:`ObjectFix` instead: this row is one control tick and goes
    # straight into the run CSV, while the object exists whether or not
    # anything is engaged and lives in the map frame, not this one.
    # "" while no follow is armed; otherwise following | leashed | stale |
    # lost. `follow_err_m` is |vehicle - the walked setpoint|.
    follow_state: str = ""
    follow_age_s: float | None = None
    follow_err_m: float | None = None
    # Tag-implied roll/pitch vs the autopilot's ATTITUDE. Computed since the
    # station was built and never surfaced until now: it is the check that the
    # camera extrinsic (position AND tilt) is right, and a wrong mount angle
    # reads here as that angle while the vehicle sits level.
    rp_residual_deg: float | None = None
    # ...and signed, split into (roll, pitch). The magnitude alone
    # cannot be acted on: cam_tilt_deg corrects a PITCH.
    rp_residual_rp_deg: tuple | None = None
    conn: Conn = Conn.OFFLINE
    stamp: float = field(default_factory=now)


# =============================================================================
# pilot input
# =============================================================================
@dataclass
class PilotInput:
    """One teleop command, in body axes, normalised −1..+1.

    Deliberately NOT MAVLink's ±1000 MANUAL_CONTROL units: the conversion (and
    the z-axis convention trap that ``c3_camera/control.py`` warns about) belongs
    in the command sink, next to the code that has to get it right, not spread
    across every widget that can nudge an axis.
    """

    surge: float = 0.0     # +forward
    sway: float = 0.0      # +starboard
    heave: float = 0.0     # +up
    yaw: float = 0.0       # +clockwise seen from above
    active: frozenset[str] = frozenset()   # which inputs are held right now
    source: str = ""                       # "keyboard" | "buttons" | "gamepad"
    stamp: float = field(default_factory=now)

    @property
    def any_axis(self) -> bool:
        return any(abs(v) > 1e-6 for v in (self.surge, self.sway, self.heave, self.yaw))

    def clamped(self) -> "PilotInput":
        def c(v: float) -> float:
            return max(-1.0, min(1.0, float(v)))
        return PilotInput(c(self.surge), c(self.sway), c(self.heave), c(self.yaw),
                          self.active, self.source, self.stamp)
