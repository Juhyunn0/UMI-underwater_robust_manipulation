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
from pathlib import Path


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
    #: WHICH INSTRUMENT produced this frame, when a panel can carry more than
    #: one ("c3_stereo" | "foundationstereo"). It travels here, in the same
    #: mailbox slot as the picture and the raw millimetres, for the reason the
    #: aux map does: a label on a separate signal can arrive before or after
    #: the frame it describes, and then a measurement gets credited to the
    #: wrong instrument. Cross-thread ordering cannot desynchronise a field
    #: that is inside the thing being labelled.
    instrument: str = ""
    encoding: str = ""
    conn: Conn = Conn.OFFLINE
    note: str = ""
    #: The millimetre range THIS frame was colourised over, when the producer
    #: auto-ranges it (FStereoWorker). None = the fixed imaging.DEPTH_*_MM
    #: scale. It rides in the stat, beside the picture and the raw map, for
    #: the same reason `instrument` does: a colour bar drawn from a range that
    #: arrived on another signal can be a frame out of step, and then the
    #: legend states a distance the picture does not mean.
    depth_lo_mm: float | None = None
    depth_hi_mm: float | None = None
    #: WHICH COLOUR RULE painted this frame, as "umi:<domain>:<cmap>" when the
    #: picture went through the SHARED rule (rov_gui/depth_colour.py: fixed
    #: 0.2-3.0 m, warm = NEAR, no per-frame adaptation — the same function that
    #: paints the land episode videos, so a screenshot is comparable with one).
    #: "" = the legacy adaptive palette, whose scale rides in depth_lo/hi_mm and
    #: depth_knots_mm instead. The legend needs it because the two rules put
    #: NEAR at opposite ends of the bar and place ticks by different formulas.
    depth_rule: str = ""
    #: Ascending millimetre quantiles when the picture was HISTOGRAM-EQUALISED
    #: (imaging.depth_palette_knots). The colour bar needs them to place a tick
    #: where that depth actually landed — under equalisation the bar is no
    #: longer linear in millimetres, and a linear one would mislabel it.
    depth_knots_mm: tuple | None = None
    #: WHICH GRID the raw map beside this picture is on: "color" (projected
    #: onto the colour camera, pixel-aligned with the RGB panel) or
    #: "rect_left" (the stereo network's own rectified-left frame). Anything
    #: that back-projects the map through the COLOUR intrinsics — the
    #: depth-vs-MAP calibration check — must read this first; it used to be
    #: safe to assume, and stopped being so with --fstereo-view.
    depth_grid: str = "color"
    stamp: float = field(default_factory=now)

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}" if self.width else "--"


# =============================================================================
# vehicle telemetry
# =============================================================================
@dataclass
class FStereoState:
    """What the learned-depth worker is doing, for the panel and the readout.

    Published on its own signal rather than inferred from the depth panel's
    VideoStat, because two different consumers need it for two reasons that
    must not drift apart: the window has to attribute the depth-vs-MAP ratio to
    the right INSTRUMENT, and the FS button has to show whether the feature is
    actually running. Reading either off the picture would make them guesses.

    ``live`` is the authority on who owns the depth panel. It is TIME-BASED at
    the source (StereoMailbox.live), so a worker that dies without cleaning up
    still hands the panel back to the camera's own depth.
    """

    live: bool = False              # learned depth is the panel's picture NOW
    enabled: bool = False           # the operator's switch position
    loading: bool = False
    load_s: float = 0.0
    error: str = ""
    note: str = ""
    hz: float = 0.0                 # ARRIVAL rate, not 1/solve_ms
    solve_ms: float = 0.0
    valid_native: float = 0.0       # % of the matcher's own grid
    valid_out: float = 0.0          # % MEASURED after projection onto the grid
    #: % the scatter-gap fill closed on top of ``valid_out``. Deliberately a
    #: separate number: those pixels carry a neighbour's millimetres, not their
    #: own, and a chip that added them to "valid" would be reporting a repair
    #: as a measurement.
    filled_out: float = 0.0
    frames: int = 0
    #: A CUDA graph was asked for but the run is on the eager path (capture
    #: or replay failed, or the upstream code could not be patched). Same
    #: numbers, slower, and the pilot should be able to see it on the chip.
    eager: bool = False
    #: --policy-fs-schedule when it is not `free` ("yield" / "only"), else "".
    #: On the chip because under `only` the panel refreshes in pairs with a
    #: gap between them, and that must not read as a struggling network.
    schedule: str = ""
    stamp: float = field(default_factory=now)

    @property
    def chip(self) -> str:
        """One line for the panel overlay. Says WHY when it is not running."""
        if self.error:
            return f"FS FAULT — {self.error[:52]}"
        if self.loading:
            return f"FS LOADING — {self.load_s:.0f}s"
        if not self.enabled:
            return "FS STOPPED — no depth this run"
        if not self.live:
            return f"FS STARVED — {self.note or 'no mono pair'}"
        return (f"FOUNDATIONSTEREO  {self.hz:.1f} Hz  {self.solve_ms:.0f} ms  "
                f"valid {self.valid_native:.0f}%→{self.valid_out:.0f}%"
                + (f"+{self.filled_out:.0f}" if self.filled_out >= 0.5 else "")
                + ("  eager" if self.eager else "")
                + (f"  [{self.schedule}]" if self.schedule else ""))


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
    # The 6-DoF variant's actuation gates (2026-09-26), filled by the hardware
    # backend from AUTOPILOT_VERSION / the command sink; "" / False until known.
    firmware_version: str = ""             # "4.5.1" style from AUTOPILOT_VERSION
    mavlink_wire_version: str = ""         # "1.0" | "2.0" of the command link
    attitude_axes_enabled: bool = False    # sink configured to send s/t (engage.attitude_axes)
    attitude_axes_degraded: bool = False   # sink fell back to the 4-axis frame mid-run
    rc_chan_raw: tuple | None = None       # RC_CHANNELS chan1..chan8 raw PWM (probe / trim)
    # LOW mode rl_pwm (2026-09-30): the command sink's per-thruster gate, read back from the vehicle
    # (control/rl_pwm.PwmGate.state + "active"/"n_sent"). None = no sink that speaks it (demo, ros2, null sink).
    pwm_gate: dict | None = None
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
    # The TRACKED mount angle (control/tilt_tracker.py): the vehicle's report
    # when there is one, else the tag localizer's measurement, else the
    # operator's typed value, else held UP/DOWN dead-reckoned at a [예측]
    # rate. ``tilt_est_src`` says which, ``tilt_est_unc_deg`` how much doubt,
    # and ``tilt_epoch`` counts COMMANDED motion — the localizer resets its
    # RGB-vs-C3 alignment when it changes (control/nav_fusion.py).
    tilt_est_deg: float | None = None
    tilt_est_src: str = "unknown"
    tilt_est_unc_deg: float | None = None
    tilt_epoch: int = 0
    tilt_moving: bool = False
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
    # "primary" = the localizing feed; "fallback" = a second calibrated feed
    # whose fix stands in when the primary has none (control/nav_fusion.py);
    # "overlay" = detections drawn, never solved.
    role: str = "overlay"
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

    # ENGAGED = the mission MACHINERY is live: a datum was captured, the CSV is
    # open, the state assembler is running, a mission may be armed and the
    # reference (`ref_flu`, the plan stream, PolicyState) is real.
    engaged: bool = False
    # COMMANDING = the station's controller is driving the vehicle. Until
    # 2026-09-03 this was the SAME bit as `engaged`, and four consumers read
    # `engaged` to mean this one: window.py's teleop pump/gate and command
    # bars, and the trajectory chip's tracking-error line. POLICY OBSERVE
    # (--policy-observe) is the first mode where they differ — the diffusion
    # policy runs, its plans are composed and drawn, and NOTHING leaves the
    # station because the pilot is flying by hand. Anything that means "the
    # station is driving" MUST read this field, never `engaged`: a consumer
    # that gets it wrong either steals the pilot's stick or prints a tracking
    # error for a loop that was never closed.
    commanding: bool = False
    # This run's controller output is MUTED — LOW level None (teleop) in the
    # trajectory panel; `--policy-observe` is the launch alias that preselects
    # it. True iff `mode == "none"`. A RECORD
    # BOUNDARY: nothing measured in an observe run is comparable with a
    # closed-loop run — no wrench was ever computed, and the vehicle went
    # where the pilot flew it, not where the reference asked.
    observe: bool = False
    traj_on: bool = False
    # The LOW level the worker is on: "none" | "pid" | "mpc" | "dobmpc" |
    # "mpc_tuned" | "dobmpc_tuned" | "mpcc" | "dobmpcc". "none" is TELEOP —
    # the station commands nothing (observe is True iff mode is "none"); the
    # panel's LOW combo is re-synced from this field (honesty rule).
    mode: str = ""
    # A controller CSV is open OUTSIDE an engagement (REC on the depth feed
    # -> set_sensor_log). The panel disables the LOW combo on it: the CSV
    # pinned the run folder under the tree of the mode it was opened in, and
    # set_mode refuses while it is open (2026-09-11 review).
    csv_open: bool = False
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
    axes_rp: tuple = ()              # (roll, pitch) sent, -1..+1; () unless engage.attitude_axes
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
@dataclass(frozen=True)
class PwmCommand:
    """Eight thruster pulses [us], ArduSub motor order 1..8 — LOW mode ``rl_pwm`` (2026-09-30).

    The ONLY thing ``bus.cmd_pwm`` carries. Always explicit 1100..1900; 1500 x 8 is "stop", sent by the worker on a
    disengage. The command sink puts them on RC_CHANNELS_OVERRIDE channels 9..16 for the vehicle's Lua timeout
    override (control/rl_pwm.py) and refuses them unless its gate holds; a MANUAL_CONTROL keeps going out NEUTRAL
    beside them, so what the autopilot falls back to when the override expires is 1500 us."""

    pulses: tuple = (1500,) * 8
    #: "mpc" = an engaged follower (pulses, or neutral while it waits out a gate fault); "stop" = its disengage;
    #: "idle" = RL_PWM is selected but not engaged (keepalive). Only "stop"/"idle" clear the sink's latched gate fault.
    source: str = ""
    stamp: float = field(default_factory=now)


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
    # The two attitude axes of the 6-DoF variant (2026-09-26). Trailing fields
    # so every positional constructor above them keeps its meaning; 0.0 unless
    # engage.attitude_axes is on (no pilot input drives them -- only the
    # controller's K/M through allocation.wrench_to_axes). Signs are NED/FRD:
    roll: float = 0.0      # +starboard-down (K, about x_FRD)
    pitch: float = 0.0     # +nose-up (M, about y_FRD)

    @property
    def any_axis(self) -> bool:
        return any(abs(v) > 1e-6 for v in (self.surge, self.sway, self.heave,
                                           self.yaw, self.roll, self.pitch))

    def clamped(self) -> "PilotInput":
        def c(v: float) -> float:
            return max(-1.0, min(1.0, float(v)))
        return PilotInput(surge=c(self.surge), sway=c(self.sway),
                          heave=c(self.heave), yaw=c(self.yaw),
                          active=self.active, source=self.source,
                          stamp=self.stamp, roll=c(self.roll),
                          pitch=c(self.pitch))


# =============================================================================
# diffusion policy (live plan source) — see rov_gui/backends/policy.py and
# MpcWorker's `policy` mission shape. Three plain-data messages, one per hop:
#   MpcWorker  -> PolicyWorker : PolicyState  (every 20 Hz tick when a policy
#                                              worker exists; the proprio feed)
#   PolicyWorker -> MpcWorker  : PolicyPlan   (one per inference; the RAW
#                                              action chunk, TCP-relative)
#   PolicyWorker -> window/MpcWorker : PolicyStatus (>= 1 Hz, stamped)
# All stamps are on `now()` (monotonic). The ONLY place a stamp crosses into
# the mission clock (t_traj) is MpcWorker._tick_policy_intake.
# =============================================================================
@dataclass
class PolicyState:
    """The vehicle as the policy must see it, one tick.

    ``t_fix`` is the CAPTURE stamp of the tag fix behind ``eta`` (None when
    there is no fix). The policy's proprio history is keyed on it, not on the
    tick time: the assembler holds the same fix across ticks (a 20 Hz
    staircase), so rows appended per tick would quantise the 66.7 ms motion cue
    to {0, 1, 2} fixes. ``fix_fresh`` is False whenever ``eta`` is not a fresh
    tag solution (station bridge active, dead-reckoning control, no fix).
    ``epoch`` increments at every ENGAGE and every policy ARM; a plan built in
    an older epoch is dropped by the consumer.
    """

    t: float
    t_fix: float | None
    eta: tuple                       # (6,) datum NED [x, y, z, roll, pitch, yaw]
    fix_fresh: bool
    epoch: int
    active: bool                     # a policy mission is armed AND running
    halted: bool                     # ...but latched off (escalation/divergence)
    t0_traj: float | None            # mission clock origin (monotonic), for logs only
    eta_start: tuple | None          # (6,) datum NED pose at policy START
    grip_width_m: float              # open-loop jaw width estimate, metres
    engaged: bool
    stamp: float = field(default_factory=now)


# ---------------------------------------------------------------------------
# The policy ACTION contract vocabulary (2026-09-07, DP action 10-dim -> 5-dim).
# stdlib-only module, so perception (dp_policy), control (policy_frames,
# geometry, workers) and the tools all spell the representation the same way.
# ---------------------------------------------------------------------------
#: Legacy (16, 10) [pos(3), rot6d(6), gripper_width_m] -- the 2026-09-01 checkpoint;
#: kept DECODABLE so old plans.jsonl still render, never flown again.
ACTION_REPR_POSE10D = "pose10d"
#: (16, 5) [dx, dy, dz, dyaw, gripper_width_m]: dp in the current TCP/camera frame
#: (bit-identical to the legacy columns 0:3), dyaw = the C3-mount-referenced ZYX
#: yaw of the relative TCP rotation (NED sign), width unchanged. Encoder:
#: universal_manipulation_interface/umi/common/yaw_action.py (transcribed in
#: control/policy_frames.py).
ACTION_REPR_POS_YAW_WIDTH = "pos_yaw_width"
#: (16, 7) [dx, dy, dz, dyaw, droll, dpitch, gripper_width_m] (2026-09-26, the
#: 6-DoF variant): cols 0:4 and 6 are bit-identical to pos_yaw_width's cols 0:4
#: and 4; (droll, dpitch, dyaw) = ZYX Euler of R_bt @ R_rel @ R_bt^T, the
#: relative TCP rotation carried into the body(FRD) frame of a vehicle wearing
#: the camera on the C3 mount (yaw_action.encode_pos_rpy; transcribed in
#: control/policy_frames.py). Signs FRD: +dyaw = CW from above, +dpitch =
#: nose-up, +droll = starboard-down. Whether the roll/pitch columns are FLOWN
#: or dropped-and-logged is config policy.attitude_track (default false).
ACTION_REPR_POS_RPY_WIDTH = "pos_rpy_width"
ACTION_REPR_BY_DIM = {10: ACTION_REPR_POSE10D, 5: ACTION_REPR_POS_YAW_WIDTH,
                      7: ACTION_REPR_POS_RPY_WIDTH}
ACTION_DIM_BY_REPR = {v: k for k, v in ACTION_REPR_BY_DIM.items()}
#: The DEFAULT of config ``policy.action_repr`` [스펙: ckpt cfg
#: shape_meta.action], pinned PER MISSION at ARM: the worker's checkpoint
#: contract must equal the pinned value to arm, and every plan is checked
#: against it at intake (the obs_dt pattern) -- so a stale policy.ckpt pointing
#: at the 10-dim network is refused instead of silently decoded.
POLICY_ACTION_REPR = ACTION_REPR_POS_YAW_WIDTH
#: The representations a mission may be armed on (config policy.action_repr
#: must be one of these; the legacy pose10d is decodable for old records only).
POLICY_ACTION_REPRS_FLYABLE = (ACTION_REPR_POS_YAW_WIDTH, ACTION_REPR_POS_RPY_WIDTH)
#: The ONE default checkpoint path (dp_policy.DEFAULT_CKPT, geometry's policy
#: block, the backend fallback block and the parity reference tool all derive
#: from here). Updated when a retrain is selected; the YAML policy.ckpt lines
#: (config/hw_mpc.yaml, config/land_dp.yaml) must carry the same string
#: (rov_gui/tests/test_policy_ckpt_paths.py).
#: The UMI training/inference checkout, VENDORED into this repo on 2026-09-09
#: (external/UMI_aquatic, a nested git repo with its own history and the
#: hao-l1/UMI_aquatic remote -- the same shape as external/iPhUMI). It carries
#: commit 7948a08 "umi_depth_5d", this project's own 5-dim action work, so the
#: training code and the station that flies its checkpoints now live together.
#: Training runs from here; dp_policy imports its encoder/denoiser from here.
#: The old ~/Desktop/universal_manipulation_interface copy is DORMANT -- edits
#: there reach nothing.
POLICY_UMI_REPO = str(Path(__file__).resolve().parents[1] / "external" / "UMI_aquatic")
#: Where the BEST checkpoint of every training run lives -- THIS repo's own
#: ``data/checkpoints`` (gitignored, .gitignore "data/"), one flat folder
#: (operator request 2026-09-14): ``<YYYYMMDD_HHMMSS>_<task>_<exp_name>.ckpt``
#: is a symlink into that run's own ``checkpoints/`` and the ``.json`` beside
#: it says how it was picked (``selected`` = a held-out replay, ``auto`` = the
#: top-1 by train_loss the workspace publishes as it trains). The runs
#: themselves are ``data/YYYYMMDD/MMDD_HHMMSS_train_<task>_<exp>/`` with
#: every topk checkpoint kept as before. Not the UMI checkout's: the two were
#: split on 2026-09-09 -- ``POLICY_UMI_REPO`` above stays a CODE dependency
#: (train.py, and the two modules dp_policy imports at inference), while the
#: checkpoints live next to the project that flies them. The UMI workspace
#: yamls' hydra.run.dir write new runs under data/<date>/ and publish their
#: best into this folder (diffusion_policy/common/best_ckpt.py).
POLICY_OUTPUTS_ROOT = str(Path(__file__).resolve().parents[1] / "data" / "checkpoints")
POLICY_CKPT_DEFAULT = (f"{POLICY_OUTPUTS_ROOT}/"
                       "20260907_113747_umi_depth_5d_can_grasp_depth_5d_v0.ckpt")
#: ^ epoch 195 of the 2026-09-07 5-dim retrain (task umi_depth_5d), selected on
#: the 219-window held-out set with the DEPLOYED weights (policy.weights: model,
#: not the EMA copy): pos RMSE 17.3 mm / yaw RMS 2.20 deg (zero-predictor 2.56,
#: corr +0.56) / width 7.3 mm vs the 2026-09-01 10-dim checkpoint's 21.0 / 2.24 /
#: 7.6 on the same tool and windows [측정: <run>/checkpoints/selected.json].
#: The run's 32-window val curve is computed on the EMA weights and ranks this
#: family BACKWARDS (it picked epoch 10, the worst measured epoch) -- never rank
#: on it. The 2026-09-01 run stays readable for old records; flying it again
#: would need pose10d in POLICY_ACTION_REPRS_FLYABLE as well (arm + intake).


@dataclass
class PolicyPlan:
    """One inference: the policy's RAW output plus everything needed to
    attribute it afterwards. ``action`` is (16, D) float32 in the checkpoint's
    ``action_repr``: D = 5 [dx, dy, dz, dyaw, gripper_width_m] (pos_yaw_width,
    the 2026-09-07 retrain), D = 7 [dx, dy, dz, dyaw, droll, dpitch, width]
    (pos_rpy_width, the 6-DoF variant) or the legacy D = 10 [pos(3), rot6d(6),
    width] (pose10d); every knot is relative to the TCP pose at ``obs_t`` (knot 0 =
    the observation time itself); composition into NED happens in MpcWorker,
    never here."""

    plan_id: int
    epoch: int
    obs_t: float                     # monotonic stamp of the newest depth frame
    t_emit: float
    infer_ms: float
    action: object                   # np.ndarray (16, D) float32, D per action_repr
    lowdim: dict                     # the lowdim obs rows fed to the network
    obs_rows_t: tuple                # (t_prev, t_now) the two proprio row stamps
    obs_fix_t: tuple                 # the fix stamps bracketing those rows
    obs_pair_dt_s: float             # actual spacing of the two depth frames
    pair_dup: bool                   # True when the newest frame was duplicated
    depth_src: str                   # rect_left | color_aligned | identity
    depth_coverage: float            # of the 224 obs, from the grid map
    depth_valid: float               # measured-valid fraction of THIS obs
    ckpt_sha1: str
    # The time base the 16 raw knots live on, from the CHECKPOINT's contract
    # (down_sample_steps / dataset fps), so the consumer can refuse a plan
    # whose clock disagrees with the config-derived one instead of silently
    # rescaling every speed (verify 2026-09-02).
    obs_dt_s: float = 0.0
    # The checkpoint contract's action representation (ACTION_REPR_*); "" =
    # unknown, in which case compose_plan infers it from the array width. The
    # consumer refuses a plan whose value is not POLICY_ACTION_REPR.
    action_repr: str = ""
    # WHERE THE TIME WENT (2026-10-01), on the same monotonic clock as obs_t.
    # `infer_ms` is timed inside the session and so contains neither the wait
    # for FoundationStereo nor the observation build; with these the plan's
    # age splits into capture -> depth arrived here -> attempt began ->
    # emitted. None = this producer did not stamp it (a hand-built plan).
    depth_arrive_t: float | None = None   # the newest depth frame reached the worker
    trigger_t: float | None = None        # the tick that began this attempt
    # --policy-fs-schedule as it was IN EFFECT for this plan ("free" when no
    # gate exists, whatever was typed), and how long the forward waited for
    # the FoundationStereo frame in flight. None, not 0.0, when nothing
    # waited: a zero would read as "waited, and it took no time".
    fs_schedule: str = "free"
    fs_wait_ms: float | None = None
    fs_wait_timeout: bool | None = None
    # FoundationStereo's network input as asked for on the command line
    # ("scale 0.75 it8" / "size 224x224 it8"; "" = unknown), so a plans.jsonl
    # that two launches appended to can still be split by network input. The
    # rest of the depth setting (alpha, checkpoint) is in the run meta.
    fs_input: str = ""
    stamp: float = field(default_factory=now)


@dataclass
class PolicyPlanViz:
    """One composed policy plan, for DRAWING. Emitted once per plan (~2 Hz).

    Separate from :class:`PolicyPlan` because that one is the network's raw
    (16, D) TCP-relative output, which is not a thing that can be plotted:
    only ``MpcWorker`` knows the anchor and the extrinsic that turn it into a
    datum-NED polyline. This carries the RESULT of that composition — the same
    array ``plans.jsonl`` records as ``raw.p_ned`` — plus the filter's verdict,
    because a plan the filter REJECTED is exactly the one the operator needs to
    see: on the runs so far that is nearly all of them, and nothing on screen
    showed what the network had actually asked for.

    ``p_ned``/``yaw`` are in the ENGAGE-DATUM frame (the panel's ``_to_map``
    turns them into map coordinates), and are plain nested lists rather than a
    numpy view, per this module's rule about buffers someone else may overwrite.
    """

    plan_id: int
    status: str                      # accept | clip | reject | late | skipped
    p_ned: tuple                     # ((x...), (y...), (z...)) datum NED [m]
    yaw: tuple                       # (K,) rad
    t0: float                        # first knot, mission clock [s]
    dt: float                        # knot spacing [s]
    reason: str = ""                 # the FIRST reason, for the on-plot label
    stamp: float = field(default_factory=now)
    rp: tuple | None = None          # ((roll...), (pitch...)) rad; None on a 4-DoF plan

    @property
    def n_knots(self) -> int:
        return len(self.p_ned[0]) if self.p_ned else 0


#: ``DepthObsBuilder.why`` before any grid was offered (perception/policy_obs.py
#: sets it in __init__/reset). Shared here so the controller's refusal reads it
#: as "no grid yet", NOT as a refused grid (integration 2026-09-02).
POLICY_GRID_WHY_IDLE = "no grid set"


@dataclass
class PolicyStatus:
    """What the policy worker is doing. Stamped: a consumer must refuse a
    stale one (a dead worker must not stay 'ready' forever)."""

    ready: bool = False
    loading: bool = False
    error: str = ""
    hz: float = 0.0
    infer_ms: float = 0.0
    depth_src: str = ""
    note: str = ""
    n_plans: int = 0
    n_skip: int = 0
    # The depth GRID's state, separately from the session's: a grid the
    # builder REFUSED (coverage below policy.min_obs_coverage, live camera
    # model != training model) leaves depth_src set and ready True, so the
    # controller must read THESE to refuse arming (verify 2026-09-02).
    grid_ok: bool = False
    grid_why: str = ""               # the builder's reason when not ok (POLICY_GRID_WHY_IDLE = no grid yet)
    obs_dt_s: float = 0.0            # the checkpoint's obs stride, for the arm check
    action_repr: str = ""           # the checkpoint's action representation, for the arm check
    # The C3 MOUNT check (2026-09-07): a pos_yaw_width checkpoint carries the
    # rotation its yaw label was defined on (shape_meta.action.yaw_axis_R_frd_cam)
    # and the worker compares it with hw_nav's R_frd_cam('main'); a mismatch
    # means every dyaw would be decoded about a different axis than it was
    # labelled on, so the controller refuses to arm. True with no yaw-axis keys
    # (legacy checkpoint, stub) -- the arm check then rests on action_repr.
    mount_ok: bool = True
    mount_why: str = ""
    # WHICH checkpoint the worker HOLDS right now (loading, ready or failed)
    # — the truth the trajectory panel's picker shows and the controller pins
    # at ARM (scen["ckpt"] / scen["ckpt_sha1"]); the config's policy.ckpt is
    # only the launch seed since the panel picker (2026-09-11). `ckpt_sha1`
    # is the session's sha1 head once READY, "" otherwise (the stub has none).
    ckpt: str = ""
    ckpt_sha1: str = ""
    # Why the last panel pick was REFUSED (mission armed, still loading, not
    # a file, FS-parity mismatch, worker not running) or "" — the panel shows
    # it in red for a few seconds; a refusal that only reached the log was
    # invisible next to a name that silently snapped back (review 2026-09-11).
    ckpt_note: str = ""
    # --policy-fs-schedule as it is IN EFFECT in this worker ("free" when no
    # gate was built — the default, the demo source, device depth). The
    # controller says it at ARM: runs on different schedules must not be
    # pooled on plan age or inference time, and `only` changes the
    # observation's pair spacing too.
    fs_schedule: str = "free"
    conn: Conn = Conn.OFFLINE
    stamp: float = field(default_factory=now)
