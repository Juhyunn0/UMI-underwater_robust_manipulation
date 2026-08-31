#!/usr/bin/env python3
"""
hardware.py — the real rig: MarineSitu C3 over DepthAI, ArduSub over MAVLink.

Reuses this repo's existing plumbing rather than re-implementing it:

    c3_camera.source.C3Source     device, queues, decode, per-stream metrics
    c3_camera.config.StreamConfig the pipeline knobs (and their measured costs)
    c3_camera.mavlink_log         passive ArduSub telemetry reader
    c3_camera.preflight           readiness gate (run in __main__, before this)

Three workers, three threads, three different reasons:

    C3VideoWorker      LoopWorker  — blocks in DepthAI's queue wait; no slots
    VehicleWorker      TimerWorker — drains the MAVLink reader's buffer at 10 Hz
    MavlinkCommandSink TimerWorker — the ONLY thing that transmits, and only
                                     when two independent gates are open

Transmission posture
--------------------
``c3_camera/control.py`` states this project's standing rule: ArduSub is always
the flight controller, and two command sources at once is the hazard — ArduSub
acts on whichever MANUAL_CONTROL arrived last, so a station commanding
alongside QGroundControl makes the vehicle jitter between them. This module
therefore:

* defaults to :class:`NullCommandSink`, which transmits nothing, ever;
* builds :class:`MavlinkCommandSink` only when ``--allow-command`` is passed
  AND the pilot turns on COMMAND ENABLE in the UI;
* arms and disarms (2026-08-06, at the user's request, after this file argued
  the other way). The risk that argument was about — arming by accident — is
  answered by making ARM a hold-to-confirm that zeroes the axes first, and by
  showing the vehicle's OWN heartbeat state rather than what we sent. DISARM
  is a single click and is NOT gated behind COMMAND ENABLE, because gating a
  stop is the more dangerous mistake;
* still never changes flight mode. That stays in QGroundControl;
* sends HEARTBEAT at 1 Hz while enabled — required, because ArduSub's GCS
  failsafe is what stops the vehicle if this station dies — and stops sending
  it the moment commands are disabled, so the station stops counting as the
  commanding GCS;
* implements a deadman: input older than ``deadman_ms`` sends NEUTRAL rather
  than holding the last value. "Stop sending" is not a stop command.

The telemetry reader keeps a *different* source_system from the sink for the
same reason ``mavlink_log.py`` does: a passive logger must never be mistaken for
the commanding GCS.
"""

from __future__ import annotations

import math
import queue
import shutil
import subprocess
import time
from collections import deque
from pathlib import Path

import numpy as np

from .. import imaging
from ..leak import PRESSURE_REPEAT_S, LeakMonitor
from ..net import NicMonitor
from ..qt import Qt, Slot, import_cv2
from ..sensorlog import SensorLog
from ..state import (Conn, ImuBatch, PayloadState, PilotInput, PoseTrack,
                     SensorStat, Telemetry, ThrusterState, VehicleImu,
                     VideoStat, now)
from ..widgets.propulsion import pwm_to_norm
from .base import Backend, LoopWorker, TimerWorker

def _sha1_file(path: Path) -> str:
    """Content hash of a file, streamed. Identifies a mesh in a log record.

    The path alone is not an identity: reference-view directories get reused,
    reconstructions get re-run, and "model.obj" is the same name every time.
    """
    import hashlib

    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_ratio(text) -> tuple[int, int] | None:
    """'1/3' -> (1, 3). None leaves the config default alone."""
    if not text:
        return None
    try:
        n, d = (int(v) for v in str(text).split("/"))
        return (n, d) if n > 0 and d > 0 else None
    except ValueError:
        return None


def _parse_size(text) -> tuple[int, int] | None:
    if not text:
        return None
    try:
        w, h = (int(v) for v in str(text).lower().split("x"))
        return w, h
    except ValueError:
        return None


# Which mailbox each C3 stream feeds. "second" is either the C3's left mono
# image or the ROV's own RGB camera, chosen with --panel2.
STREAM_TO_PANEL = {"color": "main", "left": "second", "depth": "depth"}

#: THE DEPTH CORRECTION, and it is ON unless someone turns it off.
#: The C3's stereo depth reads LONG in water [측정: 2026-08-23 — mesh longest
#: axis 187 mm against a caliper-measured 119.73 mm object = 1.56x;
#: sessions/low_level_controller_data/20260823/0823_210304/mission_log.txt
#: 21:02:29 "메시 73770 verts, 143 x 166 x 187 mm"], while colour PnP stays
#: mm-accurate. 0.64 = 1/1.56.
#: WARNING — 0.64 IS A SCALAR AND THE ERROR IS NOT. The same mesh line reads
#: 143 x 166 x 187 against one caliper number, i.e. 1.19 / 1.39 / 1.56 on three
#: axes: a constant scale error stretches every axis EQUALLY, so an anisotropy
#: of 1.31 is proof the error GROWS WITH RANGE. The operator confirmed
#: 2026-08-24 that depth-vs-MAP tracks vehicle height (1.28 low, 1.56 at
#: ~0.9 m). This constant is therefore exact at ONE range only — near 1.0-1.1 m
#: [유도] — and its residual changes sign either side. It is a stopgap that
#: keeps the station usable, NOT a model. The fix is a disparity-domain
#: correction fitted at the pool; see KNOWN_ISSUES 2026-08-24.
#: It defaulted to 1.0 until 2026-08-24, which made forgetting it SILENT: two
#: runs that morning placed the object 1.55-1.57x down the camera ray — two tag
#: rows past it and half a metre under the mat — and nothing in the log said the
#: correction was absent [측정: sessions/low_level_controller_data/20260824/
#: {0824_101807,0824_101251}, object at tag 58 reported nearest tags 11/10/52].
#: The flag still exists; `--depth-scale 1.0` is what an IN-AIR bench wants.
#: Kept here rather than only in __main__ so a hand-built opts object (a test,
#: an embedder) gets the same correction the CLI does.
DEPTH_SCALE_DEFAULT = 0.64

# ArduSub JSButton::button_function_t values for the payload functions this
# station drives. Read back from the vehicle at startup and compared against
# what --btn-* claims, so a button that means something else on some other
# vehicle is never pressed blind.
BTN_FUNCTION = {
    "gripper_open": (77, "servo_1_max_momentary"),
    "gripper_close": (76, "servo_1_min_momentary"),
    "lights_up": (32, "lights1_brighter"),
    "lights_down": (33, "lights1_dimmer"),
    # Camera mount. The function NUMBERS are ArduSub's JSButton enum [스펙];
    # the BUTTON numbers they are expected on come from this vehicle's own
    # QGC assignment. Both halves are verified against the vehicle at connect
    # (see _note_button_function), so a wrong assumption here disables the
    # control with a loud message instead of pressing something unknown.
    "tilt_up": (22, "mount_tilt_up"),
    "tilt_down": (23, "mount_tilt_down"),
    "tilt_center": (21, "mount_center"),
}

# ArduSub custom_mode numbers, from pymavlink's own table
# (mavutil.mode_mapping_sub). MANUAL is the one that matters here: it is the
# only mode in which an armed vehicle sits still with no stick input. Every
# other mode listed drives the thrusters by itself — STABILIZE holds attitude,
# ALT_HOLD holds depth, POSHOLD holds position — which is correct behaviour and
# also exactly what "I pressed ARM and the motors started" was.
MODE_NUMBERS = {
    "MANUAL": 19,
    "STABILIZE": 0,
    "ALT_HOLD": 2,
    "ACRO": 1,
    "POSHOLD": 16,
    "SURFACE": 9,
}
# Modes in which arming alone makes the thrusters turn.
SELF_DRIVING_MODES = frozenset(MODE_NUMBERS) - {"MANUAL", "ACRO"}

# Open-circuit state of charge against per-cell voltage, for the Li-ion 18650
# packs a BlueROV2 ships with. NOMINAL CHEMISTRY DATA, not a measurement of this
# pack — the curve's shape is right, its exact position is not, and it is
# labelled [유도]/"est" everywhere it reaches the screen.
#
# Two things it cannot do, both worth knowing before trusting a number off it:
# the pack SAGS under thrust (8 thrusters pulling hard drop it most of a volt,
# so the estimate reads low exactly when you are working the vehicle hardest),
# and the curve is nearly flat from 3.9 to 3.6 V, which is most of the useful
# capacity — a 0.05 V reading error there is worth about 10% of charge.
#
# The real fix is on the vehicle: set BATT_CAPACITY and a current monitor, and
# ArduSub reports battery_remaining itself. That number is preferred whenever
# it exists.
LIION_CELL_CURVE = (
    (4.20, 100.0), (4.10, 92.0), (4.00, 82.0), (3.90, 70.0), (3.80, 58.0),
    (3.70, 45.0), (3.60, 31.0), (3.50, 19.0), (3.40, 10.0), (3.30, 5.0),
    (3.20, 2.0), (3.00, 0.0),
)

# Fresh-water density unless told otherwise; the pressure->depth conversion is
# a DERIVED number and the panel labels it as such. 1025 for seawater.
WATER_RHO = 997.0


DEFAULT_CELLS = 4                  # BlueROV2 stock pack is 4S


def cells_from_voltage(volt: float, default: int = DEFAULT_CELLS) -> int:
    """Series cell count for a pack voltage, ASSUMING ``default`` if it fits.

    Inference from one voltage is genuinely ambiguous and the ambiguity is not
    symmetric. A 12.0 V pack is either a 4S at 3.00 V/cell (empty) or a 3S at
    4.00 V/cell (82%), and picking the smaller count — which a naive
    "first n that fits" loop does — reports the empty pack as nearly full. That
    is the one error direction that strands a vehicle.

    So the configured/stock count wins whenever the pack could plausibly be it,
    and only a voltage outside that window falls back to searching, lowest
    per-cell voltage first (the conservative reading).
    """
    if default > 0 and 3.0 <= volt / default <= 4.25:
        return default
    for n in range(12, 0, -1):
        if 3.0 <= volt / n <= 4.25:
            return n
    return default or DEFAULT_CELLS


def soc_from_voltage(volt: float, cells: int) -> float:
    """Pack voltage -> percent, by linear interpolation on the cell curve."""
    if cells <= 0:
        return 0.0
    v = volt / cells
    if v >= LIION_CELL_CURVE[0][0]:
        return 100.0
    if v <= LIION_CELL_CURVE[-1][0]:
        return 0.0
    for (v_hi, p_hi), (v_lo, p_lo) in zip(LIION_CELL_CURVE, LIION_CELL_CURVE[1:]):
        if v_lo <= v <= v_hi:
            t = (v - v_lo) / (v_hi - v_lo)
            return p_lo + t * (p_hi - p_lo)
    return 0.0
G = 9.80665
SEA_LEVEL_HPA = 1013.25

# BNO086 report accuracy, worst first. Reported per sample and carried on the
# batch as the WORST in it, because this camera's accelerometer has read
# UNRELIABLE/MEDIUM every time it has been looked at (KNOWN_ISSUES.md:452-468)
# and a dead-reckoning run needs that visible in its own record rather than in
# a document someone has to remember.
_ACCURACY_ORDER = ("", "UNRELIABLE", "LOW", "MEDIUM", "HIGH")


def _worse_accuracy(a: str, b: str) -> str:
    """The less trustworthy of two accuracy flags. An unknown name wins (is
    treated as worst): a firmware that starts reporting something new must not
    be silently upgraded to trustworthy."""
    if not a:
        return b
    if a not in _ACCURACY_ORDER or b not in _ACCURACY_ORDER:
        return a if a not in _ACCURACY_ORDER else b
    return a if _ACCURACY_ORDER.index(a) <= _ACCURACY_ORDER.index(b) else b


# =============================================================================
# video
# =============================================================================
class C3VideoWorker(LoopWorker):
    """Blocks on the camera, converts on this thread, publishes to mailboxes.

    Opens in :meth:`run`, not in :meth:`setup`, and retries with a backoff.
    That is not defensive coding for its own sake — it is the difference
    between a station that recovers and one that has to be restarted:

    * an OAK device accepts exactly ONE pipeline at a time, so any overlap with
      the BlueOS Madrona extension, a previous run that has not finished
      closing, or a second copy of this tool gives ``DeviceBusyError`` — and
      then the camera is free again a few seconds later;
    * PoE boot takes ~12 s, so a station started at the same moment as the
      vehicle will simply miss the camera on its first try;
    * a tether that blips mid-dive kills the XLink stream, and the pilot should
      get the picture back on its own rather than by relaunching the dashboard
      with the ROV in the water.

    Every attempt is reported into the panel ("reconnecting in 5s"), because a
    silent retry loop and a dead feed look identical from the pilot's chair.
    """

    # Backoff, then hold. Not unbounded: after a minute of failures the cause is
    # something a human has to fix, and hammering the device does not help.
    RETRY_S = (2.0, 3.0, 5.0, 8.0, 10.0)

    def __init__(self, bus, mailboxes, opts):
        super().__init__("c3-video")
        self.bus = bus
        self.mailboxes = mailboxes
        self.opts = opts
        self.src = None
        self.cfg = None
        self._attempt = 0
        self._last_stat = 0.0
        self._imu_marks: deque[float] = deque(maxlen=512)
        self._last_imu_pub = 0.0
        self._last_imu_seq: int | None = None
        self._imu_log = None
        # Thread-safe hand-off from the GUI thread; see request_sensor_log.
        self._log_q: queue.SimpleQueue = queue.SimpleQueue()
        # DEPTH SCALE CORRECTION (--depth-scale). Applied to the raw
        # millimetres ONCE, at the moment a depth frame arrives, so every
        # consumer — the panel and its cursor probe, the reference-view
        # capture, FoundationPose, the depth-vs-MAP check — sees the same
        # corrected stream and the check VERIFIES the correction instead of
        # measuring around it. Why it exists: 2026-08-23, the C3's stereo
        # depth measured 1.56x LONG in water against a caliper (mesh 187 mm
        # from an object measuring 119.73 mm) and 1.71x against the tag
        # solution over the whole mat — while the colour camera's PnP was
        # millimetre-accurate. The colour intrinsics are underwater-calibrated;
        # whatever the stereo pair's are, they do not produce metric water
        # ranges. This knob is the interim; the fix is an in-water stereo
        # recalibration (KNOWN_ISSUES).
        self._depth_scale = float(getattr(opts, "depth_scale",
                                          DEPTH_SCALE_DEFAULT)
                                  or DEPTH_SCALE_DEFAULT)
        self._depth_scale_said = False
        # Set by HardwareBackend when --pose is on. None = perception off, and
        # then _tap_pose does not even copy a frame.
        self.pose_mb = None
        # Set by HardwareBackend when --mpc is on: the AprilTag worker's own
        # mailbox. A SECOND slot, not a shared one, so tag navigation and SAM2
        # tracking cannot starve each other of frames — and colour-only,
        # because tag PnP has no use for depth (no 1.15 MB copy).
        self.nav_mb = None

    def setup(self) -> None:
        # Imported here, not at module import: c3_camera's __init__ requires
        # depthai 2.x and refuses 3.x, so importing it at module scope would
        # break the demo path on a machine with no depthai at all.
        from c3_camera.config import StreamConfig

        wanted = ["color"]
        if getattr(self.opts, "depth", True):
            wanted.append("depth")
        if getattr(self.opts, "panel2", "rov") == "stereo":
            wanted.append("left")
        # Colour and the stereo pair get SEPARATE rates, because they cost
        # wildly different amounts of the link and only one of them is what the
        # pilot flies on. Measured on this rig, 960x540 MJPEG + aligned depth
        # (2026-08-06, c3_camera.metrics):
        #
        #   colour 15 + depth 15 @640x360   colour  33.7 ms   61 Mb/s   0% drop
        #   colour 30 + depth 30 @640x360   colour 128.0 ms   81 Mb/s  33% drop  <- saturated
        #   colour 30 + depth 10 @640x360   colour  34.4 ms   48 Mb/s   0% drop
        #   colour 30 + depth 10 @640x360   colour  49.4 ms   66 Mb/s   0% drop  <- same config, busier scene
        #   colour 30, no depth             colour  33.1 ms   11 Mb/s   0% drop
        #
        # i.e. asking for 30 fps on both does not give a faster picture, it
        # gives a THIRD of the frames and four times the latency.
        #
        # The two colour-30 + depth-10 rows are the SAME configuration measured
        # on different days, and they disagree by 18 Mb/s. That is not noise:
        # colour is MJPEG, so it costs 46 kB/frame on a plain scene and 121 on a
        # detailed one. Depth is raw uint16 and costs exactly w*h*2*fps whatever
        # it is looking at (36.86 predicted, 36.9 measured).
        #
        # Which is what decides how the two share the link. Depth at 640x360@20
        # is 73.7 Mb/s — 82% of it — so colour has ~16 Mb/s to live in, and
        # dropping colour FPS is the worst way to find it: 30 -> 15 returns half
        # of a number that is already too small, and costs the pilot the thing
        # they notice first. Quality and pixels are far cheaper. Measured by
        # re-encoding 120 frames this camera produced
        # (c3_camera/datasets/dataset_20260805_174544/rgb):
        #
        #   960x540 q90  114.1 kB/frame     640x360 q80   38.0 kB/frame
        #   960x540 q75   64.6              640x360 q85   44.8
        #   800x450 q90   80.2              480x270 q90   36.9
        #
        # i.e. 640x360 q80 is a THIRD of the bytes of 960x540 q90 for a
        # difference that is hard to see at panel size. Hence the defaults:
        # --isp-scale 1/3, --mjpeg-quality 80, 30 fps = 9.3 Mb/s, and
        # 9.3 + 73.7 = 83.1 of ~90.
        #
        # That re-encode is trustworthy rather than merely indicative: at q90 it
        # produced 114.1 kB/frame against the device encoder's own 114.2 on the
        # same frames.
        isp = _parse_ratio(getattr(self.opts, "isp_scale", None))
        if isp is None:
            # Never fall back to a different resolution silently: a typo'd
            # --isp-scale would change the picture AND the link budget while the
            # log still showed the configuration we thought we asked for.
            isp = StreamConfig.isp_scale
            if getattr(self.opts, "isp_scale", None):
                self.bus.log.emit(
                    "warn", f"--isp-scale {self.opts.isp_scale!r} is not N/D — "
                            f"falling back to {isp[0]}/{isp[1]}")
        self.cfg = StreamConfig(
            ip=getattr(self.opts, "ip", None) or StreamConfig.ip,
            mxid=getattr(self.opts, "mxid", None),
            streams=tuple(wanted),
            fps=float(getattr(self.opts, "fps", 30.0) or 30.0),
            isp_scale=isp,
            mjpeg_quality=int(getattr(self.opts, "mjpeg_quality", 80) or 80),
            mono_fps=float(getattr(self.opts, "depth_fps", 20.0) or 20.0),
            depth_size=_parse_size(getattr(self.opts, "depth_size", None)),
            # The C3's on-board BNO086. A DIFFERENT sensor from the autopilot's
            # IMU, on the camera's own clock — which is exactly why it is worth
            # having: it is the only IMU stamped in the same time domain as the
            # images, so it is the one a VIO/SLAM pipeline can actually use.
            # Rate ceiling is accel+gyro only; enabling ROTATION_VECTOR or the
            # magnetometer collapses every stream to ~40 Hz (c3_camera/config.py).
            imu_enable=bool(getattr(self.opts, "c3_imu", True)),
            imu_rate=int(getattr(self.opts, "c3_imu_rate", 500) or 500),
            # Batching is not a latency/throughput trade here, it is a loss
            # threshold: at these rates a threshold of 1 drops samples, and 10
            # was measured lossless (c3_camera/c3_collect.py).
            imu_batch_threshold=10,
        )
        rc = self.cfg.resolve()
        # That budget is the C3's OWN link (camera -> switch, measured ~90
        # Mbit/s), not the tether. The fibre runs at 1000 and carries the ROV's
        # camera and MAVLink besides; only C3 streams count against this.
        #
        # Warn rather than inform when the estimate is over: going over does not
        # fail, it degrades — the device drops a third of the frames and the
        # colour feed's latency quadruples — so it looks like a slow camera
        # rather than a configuration mistake unless something says so.
        from c3_camera.config import POE_BUDGET_MBPS
        over = rc.bandwidth_mbps()["total"] >= POE_BUDGET_MBPS
        self.bus.log.emit("warn" if over else "info",
                          f"c3: target {self.cfg.ip} streams={wanted} "
                          f"— {rc.budget_note()} [C3 link only]"
                          + (" — expect drops and 3-4x colour latency; "
                             "shrink --depth-size or lower --depth-fps"
                             if over else ""))
        if "depth" not in wanted:
            self._status_for("depth", Conn.OFFLINE, "depth disabled (--no-depth)")
        if getattr(self.opts, "panel2", "rov") != "stereo":
            pass          # the ROV camera worker owns that panel

    # ------------------------------------------------------------ connection
    def _status_for(self, panel: str, conn: Conn, note: str) -> None:
        self.bus.video_stat.emit(
            VideoStat(name=panel, conn=conn, note=note, stamp=now()))

    def _status(self, conn: Conn, note: str) -> None:
        """Tell the panels this worker owns what the camera is doing."""
        for panel in self._panels():
            self._status_for(panel, conn, note)

    def _panels(self) -> list[str]:
        owned = ["main"]
        if getattr(self.opts, "depth", True):
            owned.append("depth")
        if getattr(self.opts, "panel2", "rov") == "stereo":
            owned.append("second")
        return [p for p in owned if p in self.mailboxes]

    def _open(self) -> bool:
        from c3_camera.source import C3Source

        self._status(Conn.CONNECTING, "opening camera...")
        try:
            self.src = C3Source(self.cfg, verbose=False).open()
        except Exception as e:                                   # noqa: BLE001
            self.src = None
            self._attempt += 1
            delay = self.RETRY_S[min(self._attempt - 1, len(self.RETRY_S) - 1)]
            self.bus.log.emit(
                "warn", f"c3: {type(e).__name__}: {e} — retry {self._attempt} "
                        f"in {delay:.0f}s")
            self._status(Conn.FAULT,
                         f"{type(e).__name__} — retry {self._attempt} in {delay:.0f}s")
            # Event.wait, not sleep: a stop request cuts the backoff short
            # instead of making shutdown wait out the full delay.
            self._stop.wait(delay)
            return False
        self._attempt = 0
        self.bus.log.emit("info", "c3: connected")
        self._status(Conn.ONLINE, "")
        return True

    def _close(self) -> None:
        if self.src is not None:
            try:
                self.src.close()
            except Exception:                                    # noqa: BLE001
                pass
            self.src = None

    # ------------------------------------------------------------------ loop
    def run(self) -> None:
        while not self.stopping:
            if self.src is None:
                if not self._open():
                    continue
            try:
                bundle = self.src.read(timeout_ms=1000.0)
            except Exception as e:                               # noqa: BLE001
                self.bus.log.emit("warn",
                                  f"c3: stream lost ({type(e).__name__}: {e})")
                self._close()
                continue
            # Drained every pass regardless of whether a frame arrived: the IMU
            # runs at 500 Hz against a 20-30 Hz picture, so tying one to the
            # other would throttle it to the frame rate.
            self._drain_imu()
            if bundle is None:
                continue                      # timeout: the loop re-checks stop
            # Only when FRESH: `bundle.frames` carries the latest of every
            # stream across passes, so an unconditional multiply would scale
            # the same frame once per pass until the next one arrived.
            if self._depth_scale != 1.0 and "depth" in bundle.fresh:
                df = bundle.frames.get("depth")
                if df is not None and df.image is not None:
                    df.image = self._scale_depth(df.image, self._depth_scale)
                if not self._depth_scale_said:
                    self._depth_scale_said = True
                    # Say WHERE the number came from. It is on by default now,
                    # so "(--depth-scale)" alone would credit the operator for
                    # a correction they may never have typed — and the whole
                    # 2026-08-24 failure was not knowing whether it was armed.
                    src = ("default" if self._depth_scale == DEPTH_SCALE_DEFAULT
                           else "--depth-scale")
                    self.bus.log.emit(
                        "warn", f"c3: DEPTH SCALED x{self._depth_scale:g} "
                                f"({src}) — every consumer sees the "
                                f"corrected millimetres; depth-vs-MAP should "
                                f"now read ~1.0x")
            for stream in bundle.fresh:
                panel = STREAM_TO_PANEL.get(stream)
                if panel is None or panel not in self.mailboxes:
                    continue
                frame = bundle.frames.get(stream)
                if frame is None or frame.image is None:
                    continue                  # encoded passthrough, not decoded
                self._publish(panel, stream, frame)
            self._tap_pose(bundle)

    @staticmethod
    def _scale_depth(img, k: float):
        """uint16 millimetres, scaled. Zeros stay zero — 0 is DepthAI's
        "no measurement" and a hole must not become a distance."""
        out = img.astype(np.float32)
        out *= float(k)
        np.clip(out, 0.0, 65535.0, out=out)
        return out.astype(np.uint16)

    # ------------------------------------------------------------ pose tap
    def _tap_pose(self, bundle) -> None:
        """Hand the newest RGB-D pair to the perception worker.

        After the publish loop and gated on COLOUR being fresh, on purpose:

        * colour is the rate that matters (30 fps against depth's 20), and one
          tap per colour frame is what the tracker wants;
        * ``bundle.frames`` is the latest of EVERY stream, so the depth here is
          the newest depth whether or not it arrived this pass;
        * ``bundle.skew_ms`` and ``bundle.intrinsics`` are available at exactly
          this point, so the frame, its pairing error and the camera model
          travel together and cannot get out of step.

        Costs nothing when perception is off — the mailbox is None with no
        --pose, and ``wanted()`` is False while TRACK is off. Both checks come
        BEFORE the copies: the pose worker discards every item unread while
        disabled, so copying 1.15 MB per frame to feed it was pure waste (and
        contradicted the documented "off costs nothing" promise). The price is
        that intrinsics reach the pose worker only after TRACK is first
        switched on, which is also the first moment anything needs them.
        """
        want_pose = self.pose_mb is not None and self.pose_mb.wanted()
        want_nav = self.nav_mb is not None and self.nav_mb.wanted()
        if (not (want_pose or want_nav)) or "color" not in bundle.fresh:
            return
        color = bundle.frames.get("color")
        depth = bundle.frames.get("depth")
        if color is None or color.image is None:
            return
        # The stamp is when the frame was TRUE, not when we got here
        # (state.py:13-15). age_ms() covers sensor->host.
        t_capture = now() - (color.age_ms() / 1000.0)
        intr = bundle.intrinsics.get("color") if bundle.intrinsics else None
        # .copy() is not optional: DepthAI recycles its frame pool, so an array
        # handed to another thread is a race the moment the next frame lands.
        # imaging.py:11-17 documents the same hazard for QImage. Each consumer
        # gets its OWN copy for the same reason — two threads reading one
        # array is fine, but the pose worker hands its copy to torch, which
        # may write into it.
        if want_pose:
            self.pose_mb.put(
                color.image.copy(),
                depth.image.copy() if (depth is not None
                                       and depth.image is not None) else None,
                intr, t_capture, bundle.skew_ms)
        if want_nav:
            self.nav_mb.put(color.image.copy(), None, intr, t_capture,
                            bundle.skew_ms)

    # -------------------------------------------------------------- C3 IMU
    def request_sensor_log(self, on: bool, stem: str) -> None:
        """Start/stop the C3 IMU log. Runs on the CALLER's thread.

        A plain method, NOT a ``@Slot``, and that distinction is the whole
        point. This is a :class:`LoopWorker`: ``run()`` blocks on the camera for
        the process lifetime, so the thread's event loop is never entered and a
        QUEUED slot invocation would sit in a queue nobody services
        (``backends/base.py`` module docstring). It was a slot until 2026-08-09,
        and the consequence was silent — ``_c3_imu.jsonl`` was never created on
        hardware while ``_rov.jsonl`` appeared every time, because
        :class:`VehicleWorker` is a TimerWorker and its identical slot works.
        Half a dataset, and the README claimed both files.

        A queue rather than a bare attribute so a start/stop pair issued inside
        one loop iteration cannot lose the start.
        """
        self._log_q.put((bool(on), str(stem)))

    def _service_log_request(self) -> None:
        try:
            on, stem = self._log_q.get_nowait()
        except queue.Empty:
            return
        if on and self._imu_log is None:
            self._imu_log = SensorLog(Path(f"{stem}_c3_imu.jsonl"), "c3")
            self.bus.log.emit("info", f"c3 imu log -> {self._imu_log.path}")
        elif not on and self._imu_log is not None:
            meta = self._imu_log.close()
            self._imu_log = None
            self.bus.log.emit(
                "info", f"c3 imu log closed: {meta['records']} samples, "
                        f"{meta['hz_measured'].get('IMU', 0):.0f} Hz measured")

    def _drain_imu(self) -> None:
        self._service_log_request()
        if self.src is None:
            return
        try:
            samples = self.src.drain_imu()
        except Exception:                                        # noqa: BLE001
            return                      # the frame path already reports faults
        t = time.monotonic()
        rows = []
        worst = ""
        dropped = 0
        for s in samples:
            self._imu_marks.append(t)
            if self._imu_log is not None:
                self._imu_log.write("IMU", {
                    "t_device": s.t_device, "seq": s.seq,
                    "ax": s.ax, "ay": s.ay, "az": s.az,
                    "gx": s.gx, "gy": s.gy, "gz": s.gz,
                    "accel_accuracy": s.accel_accuracy,
                    "gyro_accuracy": s.gyro_accuracy,
                }, t=t)
            # A packet can legitimately carry only one of the two reports;
            # half a sample cannot be integrated, so it is dropped here rather
            # than becoming a NaN somewhere downstream.
            row = (s.t_device, s.ax, s.ay, s.az, s.gx, s.gy, s.gz)
            if any(v != v for v in row):                         # NaN check
                continue
            if self._last_imu_seq is not None:
                gap = int(s.seq) - self._last_imu_seq - 1
                if 0 < gap < 1000:
                    dropped += gap
            self._last_imu_seq = int(s.seq)
            rows.append(row)
            for acc in (s.accel_accuracy, s.gyro_accuracy):
                if acc:
                    worst = _worse_accuracy(worst, acc)
        if rows:
            arr = np.asarray(rows, dtype=np.float64)
            self.bus.camera_imu.emit(ImuBatch(
                source="c3", samples=arr, n=arr.shape[0], dropped=dropped,
                accuracy=worst, t_host_drain=t,
                t_device_last=float(arr[-1, 0]), conn=Conn.ONLINE))
        if t - self._last_imu_pub < 0.5:
            return
        self._last_imu_pub = t
        if not bool(getattr(self.opts, "c3_imu", True)):
            self.bus.sensor_stat.emit(
                SensorStat("IMU (C3)", None, Conn.OFFLINE, "disabled (--no-c3-imu)"))
            return
        # Rate over the window we have, not since launch: a sensor that stopped
        # ten seconds ago still averages fine over its lifetime.
        recent = [m for m in self._imu_marks if t - m < 2.0]
        hz = len(recent) / 2.0 if len(recent) > 1 else 0.0
        want = float(getattr(self.opts, "c3_imu_rate", 500) or 500)
        # None of these details may contain "Hz": the panel treats a detail that
        # mentions a rate as one that REPLACES the measured rate (for the REST
        # transport's "sampled" qualifier), and here the measured rate is
        # exactly what the operator needs to see.
        if hz <= 0.0:
            conn, detail = Conn.OFFLINE, "no samples"
        elif hz < 0.5 * want:
            conn, detail = Conn.DEGRADED, f"BNO086, under the {want:.0f} requested"
        else:
            conn, detail = Conn.ONLINE, "BNO086 accel+gyro"
        self.bus.sensor_stat.emit(SensorStat("IMU (C3)", hz, conn, detail))

    def _publish(self, panel: str, stream: str, frame) -> None:
        mb = self.mailboxes[panel]
        img = frame.image
        h, w = img.shape[:2]
        tw, th = mb.target_size()
        aux = None
        if stream == "depth":
            # Shrink the millimetres first, then colourise the small image.
            # See imaging.scale_depth: this is both 4x cheaper and the only
            # correct downscale for a depth map.
            small = imaging.depth_to_bgr(imaging.scale_depth(img, tw, th))
            # The raw millimetres ride along so the panel can read a distance
            # out from under the cursor. .copy() is mandatory — DepthAI
            # recycles this buffer the moment the next frame lands.
            aux = img.copy()
        else:
            small = imaging.scale_to_fit(img, tw, th)
        st = self.src.metrics.streams.get(stream)
        stat = VideoStat(
            name=panel, width=w, height=h,
            fps=st.fps if st else 0.0,
            latency_ms=frame.latency_ms,
            drop_rate=st.drop_rate if st else 0.0,
            mbps=st.mbps if st else None,
            encoding=frame.encoding or ("16UC1" if stream == "depth" else ""),
            conn=Conn.ONLINE, stamp=now())
        mb.put(imaging.bgr_to_qimage(small), stat, aux=aux)

        t = time.monotonic()
        if t - self._last_stat > 0.5:
            self._last_stat = t
            self.bus.video_stat.emit(stat)

    def teardown(self) -> None:
        if self._imu_log is not None:
            self._imu_log.close()             # a killed station still gets a file
            self._imu_log = None
        if self.src is not None:
            self._close()
            self.bus.log.emit("info", "c3: closed")


# =============================================================================
# the ROV's own RGB camera
# =============================================================================
def gstreamer_available() -> bool:
    return shutil.which("gst-launch-1.0") is not None


class GstRovCamWorker(LoopWorker):
    """The ROV camera through GStreamer, which is what QGroundControl uses.

    Why this exists at all: on 2026-08-06 the FFmpeg path was measured clean by
    every proxy available — kernel receive queue empty, no drops, +1 ms drift
    over 20 s, frame gaps capped at 48 ms, no B-frames to reorder, a receive
    chain that a stall test showed can only hold ~430 ms — and the picture was
    still about two seconds behind, while BlueOS's own viewer and QGC showed the
    SAME stream instantly. When every measurement of a component says it is fine
    and the component is still wrong, the honest move is to stop defending it
    and use the implementation that demonstrably works on this machine.

    The pipeline is QGC's, with the two settings that matter:

        rtpjitterbuffer latency=0 drop-on-latency=true
            no jitter buffer at all. This is the one FFmpeg has no equivalent
            for: its RTP reader thread fills a multi-megabyte FIFO that nothing
            outside it can see or bound.
        fdsink sync=false
            emit frames as they decode instead of pacing them to timestamps.

    Scaling happens in the pipeline (``videoscale``) rather than in numpy: it
    keeps the raw frames coming over the pipe small and takes the cost off this
    thread. The size is fixed at launch, so the panel does the last few percent.
    """

    def __init__(self, bus, mailboxes, opts):
        super().__init__("rov-cam-gst")
        self.bus = bus
        self.mailboxes = mailboxes
        self.opts = opts
        self.proc = None
        self.w, self.h = self._size()
        self._frame_bytes = self.w * self.h * 3
        self._last_resize = 0.0
        self._count = 0
        self._rate_t = 0.0
        self._fps = 0.0
        self._last_frame_t = 0.0
        self._warned_quiet = False
        self._last_stat = 0.0
        # Set by HardwareBackend when --mpc is on: the AprilTag worker's tap
        # for this feed. This camera has NO calibration, so its detections
        # are an overlay only (TagOverlay.localizes=False) — intrinsics are
        # deliberately passed as None.
        self.nav_mb = None

    # The stream's own size. Scaling above it invents pixels; the pipeline is
    # capped here so a maximised panel cannot ask for more detail than exists.
    SRC_W, SRC_H = 1920, 1080
    # Hysteresis, so a window drag does not restart the pipeline on every pixel.
    GROW, SHRINK, RESIZE_COOLDOWN_S = 1.25, 0.6, 3.0

    def _size(self) -> tuple[int, int]:
        """Pipeline output size: explicit WxH, or matched to the panel."""
        text = str(getattr(self.opts, "rov_cam_size", "auto")).lower()
        if text != "auto":
            try:
                w, h = (int(v) for v in text.split("x"))
                return self._clamp(w, h)
            except ValueError:
                pass
        mb = self.mailboxes.get("second")
        tw, th = mb.target_size() if mb else (0, 0)
        if tw <= 0 or th <= 0:
            return 960, 540               # half the source, until the panel lays out
        return self._clamp(tw, th)

    def _clamp(self, w: int, h: int) -> tuple[int, int]:
        """Fit (w, h) inside the source, keep its aspect, keep both even.

        Aspect matters here and not in the FFmpeg path: videoscale will happily
        stretch to whatever caps it is given, and the panel letterboxes assuming
        the frame still has the camera's proportions.
        """
        w = max(320, min(self.SRC_W, w))
        h = max(180, min(self.SRC_H, h))
        # Snap to the source aspect so nothing is stretched.
        if w / h > self.SRC_W / self.SRC_H:
            w = int(h * self.SRC_W / self.SRC_H)
        else:
            h = int(w * self.SRC_H / self.SRC_W)
        return w - (w % 2), h - (h % 2)

    def _wants_resize(self) -> tuple[int, int] | None:
        """Has the panel grown or shrunk enough to be worth a restart?"""
        if str(getattr(self.opts, "rov_cam_size", "auto")).lower() != "auto":
            return None
        t = time.monotonic()
        if t - self._last_resize < self.RESIZE_COOLDOWN_S:
            return None
        w, h = self._size()
        ratio = w / max(1, self.w)
        if ratio > self.GROW or ratio < self.SHRINK:
            self._last_resize = t
            return w, h
        return None

    def _pipeline(self) -> list:
        port = int(getattr(self.opts, "rov_cam_port", 5600))
        caps = ("application/x-rtp,media=video,clock-rate=90000,"
                "encoding-name=H264,payload=96")
        return [
            "gst-launch-1.0", "-q",
            "udpsrc", f"port={port}", f"caps={caps}", "!",
            "rtpjitterbuffer", "latency=0", "drop-on-latency=true", "!",
            "rtph264depay", "!", "h264parse", "!",
            "avdec_h264", "!",
            # Scale BEFORE converting: avdec_h264 emits I420 at 1.5 bytes per
            # pixel, so shrinking there moves half the bytes that shrinking
            # after the BGR conversion would.
            "videoscale", "!", f"video/x-raw,width={self.w},height={self.h}", "!",
            "videoconvert", "!", "video/x-raw,format=BGR", "!",
            "fdsink", "fd=1", "sync=false",
        ]

    def setup(self) -> None:
        self.w, self.h = self._size()
        self._frame_bytes = self.w * self.h * 3
        self.bus.log.emit(
            "info", f"rov-cam: GStreamer, udp/{getattr(self.opts, 'rov_cam_port', 5600)}"
                    f" -> {self.w}x{self.h} BGR (jitterbuffer off)")

    def _open(self) -> bool:
        self._status(Conn.CONNECTING, "starting GStreamer...")
        try:
            self.proc = subprocess.Popen(
                self._pipeline(), stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, bufsize=0)
        except OSError as e:
            self._status(Conn.FAULT, f"gst-launch-1.0: {e}")
            self._stop.wait(3.0)
            return False
        self._last_frame_t = time.monotonic()
        self._warned_quiet = False
        return True

    def _status(self, conn: Conn, note: str) -> None:
        self.bus.video_stat.emit(
            VideoStat(name="second", conn=conn, note=note, stamp=now()))

    def run(self) -> None:
        buf = bytearray(self._frame_bytes)
        while not self.stopping:
            if self.proc is None:
                if not self._open():
                    continue
                buf = bytearray(self._frame_bytes)
            # The output size is fixed when the pipeline launches, so matching
            # the panel means relaunching it — cheap (~0.5 s) and only on a
            # deliberate layout change, e.g. double-clicking this feed into the
            # big slot, which is exactly when the extra detail is wanted.
            new = self._wants_resize()
            if new is not None:
                self.w, self.h = new
                self._frame_bytes = self.w * self.h * 3
                self.bus.log.emit("info", f"rov-cam: panel changed — restarting "
                                          f"GStreamer at {self.w}x{self.h}")
                self._close()
                continue
            view = memoryview(buf)
            got = 0
            while got < self._frame_bytes and not self.stopping:
                n = self.proc.stdout.readinto(view[got:])
                if not n:
                    self._fail()
                    break
                got += n
            if got == self._frame_bytes:
                self._publish(np.frombuffer(bytes(buf), np.uint8)
                              .reshape(self.h, self.w, 3))

    def _fail(self) -> None:
        err = b""
        if self.proc is not None:
            try:
                err = self.proc.stderr.read() or b""
            except Exception:                                    # noqa: BLE001
                pass
            self._close()
        port = int(getattr(self.opts, "rov_cam_port", 5600))
        msg = err.decode("utf-8", "replace").strip().splitlines()
        hint = msg[-1] if msg else f"no data on udp/{port}"
        self.bus.log.emit("warn", f"rov-cam: GStreamer stopped — {hint}")
        self._status(Conn.FAULT, hint[:60])
        self._stop.wait(3.0)

    def _publish(self, frame) -> None:
        mb = self.mailboxes.get("second")
        if mb is None:
            return
        # Tap BEFORE scaling, so tag corners land in the stream's own pixels
        # (the overlay scales by src dims). .copy() for the same reason as
        # _tap_pose: the buffer is reused the moment the next frame lands.
        if self.nav_mb is not None and self.nav_mb.wanted():
            self.nav_mb.put(frame.copy(), None, None, now(), 0.0)
        t = time.monotonic()
        self._last_frame_t = t
        self._count += 1
        if t - self._rate_t >= 1.0:
            self._fps = self._count / (t - self._rate_t)
            self._count, self._rate_t = 0, t
        tw, th = mb.target_size()
        small = imaging.scale_to_fit(frame, tw, th)
        stat = VideoStat(name="second", width=self.w, height=self.h,
                         fps=self._fps, latency_ms=None,
                         encoding="h264/rtp gst", conn=Conn.ONLINE, stamp=now())
        mb.put(imaging.bgr_to_qimage(small), stat)
        if t - self._last_stat > 0.5:
            self._last_stat = t
            self.bus.video_stat.emit(stat)

    def _close(self) -> None:
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2.0)
            except Exception:                                    # noqa: BLE001
                try:
                    self.proc.kill()
                except Exception:                                # noqa: BLE001
                    pass
            self.proc = None

    def teardown(self) -> None:
        self._close()


class RovCamWorker(LoopWorker):
    """The BlueROV2's standard camera, as BlueOS already streams it.

    This is NOT the C3. It is the low-light USB camera on the Raspberry Pi
    (``/dev/video2`` on this vehicle), which BlueOS's mavlink-camera-manager
    encodes to H.264 and pushes as RTP to ``udp://192.168.2.1:5600`` at
    1920x1080@30 — the same stream QGroundControl draws. Probed from
    ``http://192.168.2.2:6020/streams`` on 2026-08-06.

    Two consequences worth knowing before trusting this picture:

    * **It is not low latency.** It goes camera -> Pi -> H.264 encode -> RTP ->
      tether -> host decode. That whole path is exactly what ``c3_camera``
      exists to bypass for the C3 (see its README). Use it for situational
      awareness; fly on the C3 feed, whose sensor-to-host latency is measured
      at ~34 ms.
    * **One UDP port, one receiver.** If QGroundControl is running it holds
      5600, and this worker will bind but never see a datagram — the same
      "two processes cannot receive the same unicast UDP" problem as the
      MAVLink port. The fix is a second endpoint in BlueOS -> Video Streams
      (``udp://192.168.2.1:5601``) and ``--rov-cam-port 5601``; the worker says
      so itself after 8 quiet seconds rather than sitting there black.

    Decoding is FFmpeg through OpenCV, driven by a generated SDP file. cv2 in
    this env is built with FFMPEG=YES and GStreamer=NO (checked 2026-08-06), so
    the usual BlueROV2 ``gst-launch`` recipe is not available here.
    """

    # Told to FFmpeg before the capture is created. `nobuffer`/`low_delay` stop
    # it from holding frames back for reordering, and a small buffer means the
    # picture is late by the decoder, not by us.
    # Told to FFmpeg before the capture is created. Every one of these is here
    # because of a measurement, not a cargo-culted recipe:
    #
    #   probesize;32 / analyzeduration;0
    #       THE fix. With the defaults, avformat_find_stream_info buffers video
    #       while it works out what the stream is: open() took 1.7 s and the
    #       decoder started that far behind. Measured over 15 s against the
    #       stream's own PTS, the picture then fell a further 300 ms behind and
    #       kept going, because decoding 1080p costs more than the 33 ms budget.
    #       With a small probe, open takes 0.1 s and the drift over 20 s is
    #       +1 ms — the socket drops what we cannot decode instead of queueing
    #       it, which is the right trade for a live view.
    #   max_delay;0        no RTP reordering delay on top of that.
    #   fflags;nobuffer, flags;low_delay, reorder_queue_size;0
    #       the usual low-latency set; kept because they cost nothing.
    #   threads;1
    #       FFmpeg's FRAME threading holds (threads-1) decoded frames back
    #       before emitting any — 3 frames = 100 ms at 30 fps — and one thread
    #       decodes this stream at 31.0 fps against a 30 fps source anyway.
    FFMPEG_OPTS = ("protocol_whitelist;file,rtp,udp|"
                   "fflags;nobuffer|flags;low_delay|reorder_queue_size;0|"
                   "max_delay;0|probesize;32|analyzeduration;0|threads;1")
    QUIET_S = 8.0
    # A grab that returns in less than this fraction of a frame interval did not
    # wait for the wire — it came out of a buffer, so there is newer video
    # behind it. See _grab_latest().
    CATCHUP_FRACTION = 0.5
    # A backstop only: the real limit is the one-frame-interval time budget in
    # _grab_latest(). 8 discards is already a quarter second of video.
    MAX_SKIP = 8

    def __init__(self, bus, mailboxes, opts):
        super().__init__("rov-cam")
        self.bus = bus
        self.mailboxes = mailboxes
        self.opts = opts
        self.cap = None
        self.sdp = None
        self._last_frame_t = 0.0
        self._warned_quiet = False
        self._last_stat = 0.0
        self._count = 0
        self._rate_t = 0.0
        self._fps = 0.0
        self._skipped = 0
        self._skip_logged = 0.0
        self._debt = 0.0                 # frames we owe the discard loop
        self._last_loop = 0.0
        # AprilTag tap (see GstRovCamWorker): overlay-only, no calibration.
        self.nav_mb = None

    # --------------------------------------------------------------- session
    def setup(self) -> None:
        import tempfile

        host = getattr(self.opts, "rov_cam_host", "192.168.2.1")
        port = int(getattr(self.opts, "rov_cam_port", 5600))
        # An SDP file is how FFmpeg is told what a bare RTP stream contains;
        # without it the payload type and codec are unknown and it will not open.
        sdp = (f"v=0\n"
               f"o=- 0 0 IN IP4 {host}\n"
               f"s=BlueROV2 camera\n"
               f"c=IN IP4 {host}\n"
               f"t=0 0\n"
               f"m=video {port} RTP/AVP 96\n"
               f"a=rtpmap:96 H264/90000\n")
        fd = tempfile.NamedTemporaryFile("w", suffix=".sdp", delete=False)
        fd.write(sdp)
        fd.close()
        self.sdp = fd.name
        self.bus.log.emit("info", f"rov-cam: listening on udp/{port} (H.264 RTP)")

    def _open(self) -> bool:
        import os

        cv2 = import_cv2()
        self._status(Conn.CONNECTING, "waiting for RTP...")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = self.FFMPEG_OPTS
        cap = cv2.VideoCapture(self.sdp, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            # The observed failure is "bind failed: Address already in use" from
            # FFmpeg's udp protocol — i.e. QGroundControl, or a second copy of
            # this station, already holds the port. Naming it saves the hunt.
            port = int(getattr(self.opts, "rov_cam_port", 5600))
            self.bus.log.emit(
                "warn", f"rov-cam: could not bind udp/{port} — QGroundControl or "
                        f"another rov_gui probably holds it. Add a second "
                        f"endpoint udp://192.168.2.1:5601 in BlueOS > Video "
                        f"Streams and pass --rov-cam-port 5601, or use "
                        f"--panel2 stereo")
            self._status(Conn.FAULT, f"udp/{port} is already in use")
            self._stop.wait(3.0)
            return False
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:                                        # noqa: BLE001
            pass
        self.cap = cap
        self._last_frame_t = time.monotonic()
        self._warned_quiet = False
        return True

    def _status(self, conn: Conn, note: str) -> None:
        self.bus.video_stat.emit(
            VideoStat(name="second", conn=conn, note=note, stamp=now()))

    # ------------------------------------------------------------------ loop
    def run(self) -> None:
        while not self.stopping:
            if self.cap is None:
                if not self._open():
                    continue
            if not self._grab_latest():
                self._quiet_check()
                continue
            ok, frame = self.cap.retrieve()
            if not ok:
                self._quiet_check()
                continue
            self._publish(frame)
            # How far behind this iteration put us: everything the vehicle
            # produced while we were decoding, converting and rescaling, minus
            # the one frame we actually consumed.
            now = time.monotonic()
            interval = 1.0 / max(1.0, float(getattr(self.opts, "rov_cam_fps", 30.0)))
            if self._last_loop:
                self._debt += (now - self._last_loop) / interval - 1.0
            self._last_loop = now
            self._debt = max(0.0, min(self._debt, float(self.MAX_SKIP)))

    def _grab_latest(self) -> bool:
        """Discard whatever we could not keep up with, then take one frame.

        The measurement that drove this (2026-08-06, real vehicle): after
        reading for 5 s and then stalling for 3 s, **14 frames were already
        waiting** — a standing ~470 ms of buffer that never drains, because in
        steady state we consume at exactly the rate the vehicle produces. That
        constant offset IS the latency the pilot sees; it is invisible to a
        drift measurement, which is why "drift +1 ms" was not the whole story.

        Shrinking the UDP buffer (``buffer_size;32768``) does remove it — and
        shreds the picture, because packets are then dropped in the middle of a
        frame: measured, the decoder logged continuous "error while decoding MB"
        and the image broke up. Dropping WHOLE frames is the only acceptable way
        to drop.

        A timing threshold is not reliable either: a buffered ``grab`` still has
        to decode, and 1080p decode is a good fraction of a frame interval, so
        "fast" and "live" look alike. So this counts TIME instead, which needs
        no threshold:

            every frame interval that passes = one frame produced by the vehicle
            every frame we publish          = one frame consumed

        The difference accumulates as ``_debt`` in frames. Debt is paid off with
        ``grab`` alone — decode without the colour conversion and rescale, which
        is the expensive half — and each discard nets ``1 - decode/interval``
        frames of progress, so it converges instead of chasing its tail.
        """
        interval = 1.0 / max(1.0, float(getattr(self.opts, "rov_cam_fps", 30.0)))
        skipped = 0
        started = time.monotonic()
        while self._debt >= 1.0 and skipped < self.MAX_SKIP:
            # HARD time budget. Without it this loop can hold the picture for
            # seconds: catching up costs a decode per discarded frame, and if
            # decoding is slower than the stream, every discard adds more debt
            # than it clears. The freezes-then-STALE-every-second behaviour on
            # 2026-08-06 was exactly that runaway, and the panel was right to
            # call it stale — nothing had been published for over a second.
            if time.monotonic() - started > interval:
                break
            t0 = time.monotonic()
            if not self.cap.grab():
                return False
            dt = time.monotonic() - t0
            if dt >= interval:
                # Decoding one frame costs more than the stream takes to produce
                # one. No amount of discarding wins that race, so stop trying and
                # let the kernel drop whole datagrams for us; the alternative is
                # spending the whole budget on frames nobody will see.
                self._debt = 0.0
                break
            self._debt -= 1.0            # one frame of backlog thrown away
            self._debt += dt / interval  # ... but time passed while decoding it
            skipped += 1

        return self.cap.grab()

    def _publish(self, frame) -> None:
        mb = self.mailboxes.get("second")
        if mb is None:
            return
        t = time.monotonic()
        self._last_frame_t = t
        self._warned_quiet = False
        self._count += 1
        if t - self._rate_t >= 1.0:
            self._fps = self._count / (t - self._rate_t)
            self._count, self._rate_t = 0, t

        # Same tap as the GStreamer path: full-res, pre-scale, no intrinsics.
        if self.nav_mb is not None and self.nav_mb.wanted():
            self.nav_mb.put(frame.copy(), None, None, now(), 0.0)
        h, w = frame.shape[:2]
        tw, th = mb.target_size()
        small = imaging.scale_to_fit(frame, tw, th)
        stat = VideoStat(name="second", width=w, height=h, fps=self._fps,
                         # No honest latency figure: RTP timestamps are in the
                         # camera's own clock domain and nothing syncs it to
                         # this host, so a number here would be invented.
                         latency_ms=None, encoding="h264/rtp",
                         conn=Conn.ONLINE, stamp=now())
        mb.put(imaging.bgr_to_qimage(small), stat)
        if t - self._last_stat > 0.5:
            self._last_stat = t
            self.bus.video_stat.emit(stat)

    def teardown(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.sdp:
            import os
            try:
                os.unlink(self.sdp)
            except OSError:
                pass


# =============================================================================
# vehicle telemetry
# =============================================================================
class VehicleWorker(TimerWorker):
    """Drains the passive MAVLink reader and publishes 10 Hz snapshots.

    The reader (``c3_camera.mavlink_log.MavlinkLogger``) already runs its own
    thread and buffers records, so this worker never blocks — which is what
    lets it stay a TimerWorker and keep an event loop for command slots.
    """

    def __init__(self, bus, opts):
        # With --mpc the tick doubles to 20 Hz: the control loop needs
        # VehicleImu at its own rate, while the Telemetry snapshot stays
        # throttled to 10 Hz below (the UI does not need more, and doubling
        # the GUI-thread slot traffic buys nothing).
        self._mpc = bool(getattr(opts, "mpc", False))
        super().__init__("vehicle", interval_ms=50 if self._mpc else 100)
        self.bus = bus
        self.opts = opts
        self.logger = None
        self.nic = NicMonitor(peer=(getattr(opts, "blueos_host", "192.168.2.2"), 80))
        self._latest: dict[str, dict] = {}
        self._last_link = 0.0
        self._last_tel = 0.0
        self._rho = float(getattr(opts, "water_density", WATER_RHO) or WATER_RHO)
        self._log: SensorLog | None = None
        # Everything worth keeping beside a depth recording. Deliberately a
        # allow-list rather than "log everything": the link also carries our own
        # MANUAL_CONTROL echoes and parameter traffic, and a log that is 90%
        # protocol chatter is one nobody loads twice.
        self._log_msgs = ("RAW_IMU", "SCALED_IMU2", "SCALED_IMU3", "HIGHRES_IMU",
                          "SCALED_PRESSURE", "SCALED_PRESSURE2", "ATTITUDE",
                          "AHRS2", "VFR_HUD", "GLOBAL_POSITION_INT",
                          "LOCAL_POSITION_NED", "BATTERY_STATUS", "SYS_STATUS",
                          "EKF_STATUS_REPORT", "MOUNT_STATUS", "SERVO_OUTPUT_RAW",
                          # what the vehicle SAID, incl. "Leak Detected" — a
                          # recording that cannot show the leak warning is a
                          # recording that cannot explain the recovery
                          "STATUSTEXT")
        # Water in the enclosure. See rov_gui/leak.py: ArduSub reports it only
        # as STATUSTEXT, so this is fed from tick() rather than from _latest.
        # `leak_cfg_fn` is injected by HardwareBackend from the command sink,
        # which is the object that may read parameters (this worker is a
        # passive observer and never transmits) — same pattern as the MPC
        # worker's sink_status_fn.
        self._leak = LeakMonitor()
        self._leak_reported = 0
        self._press_reported = 0.0
        self.leak_cfg_fn = None

    def setup(self) -> None:
        from c3_camera.mavlink_log import MavlinkLogger

        self.logger = MavlinkLogger(
            connection=getattr(self.opts, "mavlink", "udpin:0.0.0.0:14551"),
            transport=getattr(self.opts, "mavlink_transport", "udp"),
            rest_url=getattr(self.opts, "mavlink_rest_url",
                             "http://192.168.2.2:6040"),
            source_system=200,        # passive observer, NOT the commanding GCS
            # Raising message rates IS a transmission to the autopilot, so it
            # stays opt-in (c3_camera/mavlink_log.py explains why). ArduSub's
            # defaults are low — 2 Hz IMU on this vehicle — which is fine for a
            # dashboard and useless for anything time-critical.
            set_rates=bool(getattr(self.opts, "mavlink_set_rates", False)),
            default_rate_hz=int(getattr(self.opts, "mavlink_rate", 50) or 50),
            verbose=False)
        ok = self.logger.start(wait_s=3.0)
        self.bus.log.emit("info" if ok else "warn",
                          f"mavlink: {'connected' if ok else 'no telemetry yet'} "
                          f"({self.logger.transport})")

    def tick(self) -> None:
        for rec in self.logger.drain():
            # HEARTBEAT arrives from every system on the link — the autopilot,
            # onboard components, and any GCS. Keeping the last one regardless
            # of source is how a disarmed vehicle came to display MOTORS ARMED
            # (component 194 heartbeats with base_mode 0x80). The REST transport
            # is already scoped to components/1 by its URL, so an absent
            # _srccomp is the autopilot by construction.
            if (rec["msg_type"] == "HEARTBEAT"
                    and rec.get("_srccomp") not in (None, 1)):
                continue
            # STATUSTEXT is an EVENT stream, so it cannot go through _latest
            # like everything else: keeping only the newest one means a "Leak
            # Detected" is erased by the next mode-change message that happens
            # to arrive first. Every record is inspected on the way past.
            if rec["msg_type"] == "STATUSTEXT":
                self._note_statustext(rec)
            self._latest[rec["msg_type"]] = rec
            if self._log is not None and rec["msg_type"] in self._log_msgs:
                self._log.write(rec["msg_type"], rec)
        t = time.monotonic()
        # Telemetry stays a 10 Hz human-rate snapshot even when the worker
        # ticks at 20 Hz for the MPC; VehicleImu goes out every tick.
        if t - self._last_tel >= 0.1:
            self._last_tel = t
            self._publish()
        if self._mpc:
            self._publish_imu()
        if t - self._last_link > 1.0:
            self._last_link = t
            self.bus.link.emit(self.nic.sample())

    # ----------------------------------------------------------------- leak
    def _note_statustext(self, rec: dict) -> None:
        """One STATUSTEXT: into the leak monitor, and onto the log if loud.

        ArduSub repeats "Leak Detected" every 20 s while wet, so this must not
        emit a log line per message — the mission log would fill with one
        sentence. The first report of an event is loud; the repeats only
        refresh the hold inside :class:`~rov_gui.leak.LeakMonitor`.
        """
        text = str(rec.get("text", "") or "").replace("\x00", "").strip()
        sev = rec.get("severity")
        kind = self._leak.note_statustext(text, sev, t=now())
        if kind == "leak":
            if self._leak.leaks_seen != self._leak_reported:
                self._leak_reported = self._leak.leaks_seen
                self.bus.log.emit(
                    "error", "LEAK DETECTED — water in the enclosure. "
                             "Surface and recover the vehicle.")
        elif kind == "pressure":
            if now() - self._press_reported > PRESSURE_REPEAT_S:
                self._press_reported = now()
                self.bus.log.emit(
                    "error", "internal pressure critical (ArduSub "
                             "FS_PRESS_MAX) — an enclosure is filling")
        elif kind == "config":
            self.bus.log.emit("warn", f"vehicle: {text}")
        elif text and int(sev or 6) <= 4:
            # Any other WARNING-or-worse the vehicle says. These were being
            # dropped one layer down until 2026-08-14, which is why prearm
            # refusals never appeared on this station.
            self.bus.log.emit("warn", f"vehicle: {text[:80]}")

    def _internal_pressure_hpa(self) -> float | None:
        """The ENCLOSURE barometer, absolute hPa.

        ``SCALED_PRESSURE`` (id 29) is the autopilot's own baro, inside the
        electronics tube; ``SCALED_PRESSURE2`` (id 137) is the external Bar30
        that measures depth. Rising internal pressure is the early sign of a
        breach, and it is also what ArduSub's FS_PRESS failsafe watches.
        """
        hpa = self._get("SCALED_PRESSURE", "press_abs")
        return None if hpa is None else float(hpa)

    # -------------------------------------------------------------- decoding
    def _get(self, msg: str, field: str, default=None):
        rec = self._latest.get(msg)
        if rec is None:
            return default
        v = rec.get(field, default)
        return default if v is None else v

    def _depth_m(self) -> float | None:
        """Depth, positive down. From the autopilot's own altitude if it gives
        one, else derived from absolute pressure.

        The pressure path is a DERIVATION, not a measurement of depth: it
        assumes surface pressure is 1013.25 hPa and a water density, and both
        are wrong by some amount on any given day. It is right to within a few
        centimetres for shallow work and it is the wrong number to log as a
        depth measurement.
        """
        alt_mm = self._get("GLOBAL_POSITION_INT", "relative_alt")
        if alt_mm is not None:
            return -float(alt_mm) / 1000.0
        # SCALED_PRESSURE2 (id 137) ONLY — the external Bar30. There used to be
        # a fallback to SCALED_PRESSURE (id 29) here, and that message is the
        # autopilot's own barometer INSIDE the electronics tube: a dropped
        # Bar30 would have silently turned enclosure pressure into a depth,
        # and that number does not stop at the display — it feeds
        # VehicleImu.depth_m and therefore the MPC's z whenever
        # hw_nav.yaml has z_source: pressure. A missing depth sensor must read
        # as missing (the panel shows "--", the state assembler goes stale and
        # says why), never as a plausible wrong depth. Removed 2026-08-14.
        hpa = self._get("SCALED_PRESSURE2", "press_abs")
        if hpa is None:
            return None
        return max(0.0, (float(hpa) - SEA_LEVEL_HPA) * 100.0 / (self._rho * G))

    def _sensors(self) -> dict[str, SensorStat]:
        stats = self.logger.stats if self.logger else {}
        # With --mavlink-transport rest we POLL MAVLink2Rest for the latest
        # value of each message, so the rate we measure is our own polling
        # rate, not the vehicle's transmit rate. Grading that against a
        # transmit-rate floor paints a healthy IMU amber forever, and an
        # indicator that is always amber is an indicator nobody reads. So the
        # number is labelled "sampled" and graded on liveness only.
        sampled = getattr(self.opts, "mavlink_transport", "udp") == "rest"

        def rate(*names: str) -> float | None:
            hz = [stats[n].hz for n in names if n in stats]
            return max(hz) if hz else None

        def state(hz: float | None, floor: float) -> Conn:
            if hz is None:
                return Conn.OFFLINE
            if sampled:
                return Conn.ONLINE if hz > 0.2 else Conn.DEGRADED
            return Conn.ONLINE if hz >= floor else Conn.DEGRADED

        def detail(hz: float | None) -> str:
            return f"{hz:.0f} Hz sampled" if (sampled and hz is not None) else ""

        imu = rate("RAW_IMU", "SCALED_IMU2", "SCALED_IMU3", "HIGHRES_IMU")
        baro = rate("SCALED_PRESSURE2", "SCALED_PRESSURE")
        att = rate("ATTITUDE", "AHRS2")
        hb = rate("HEARTBEAT")
        out = {
            # Named for its owner. This is the autopilot's IMU (RAW_IMU /
            # SCALED_IMU2 over MAVLink), NOT the C3's BNO086 — those are
            # different sensors on different clocks, and confusing them is how a
            # VIO pipeline ends up fusing the wrong one.
            "IMU (ROV)": SensorStat("IMU (ROV)", imu, state(imu, 10.0), detail(imu)),
            "Barometer": SensorStat("Barometer", baro, state(baro, 2.0), detail(baro)),
            "Compass": SensorStat("Compass", att, state(att, 5.0), detail(att)),
            "Heartbeat": SensorStat("Heartbeat", hb, state(hb, 0.5), detail(hb)),
        }
        # Leak — the row QGC does not have and this station now does. Its
        # state comes from STATUSTEXT (never from a rate), and it is DEGRADED
        # rather than green whenever the station cannot prove the vehicle
        # would report a leak at all. See rov_gui/leak.py.
        if self.leak_cfg_fn is not None:
            self._leak.note_config(**self.leak_cfg_fn())
        ls = self._leak.state()
        out["Leak"] = SensorStat("Leak", None, ls.conn, ls.detail)
        # ...and the enclosure barometer beside it, because a leak pad is a
        # LATE signal: it only trips once water reaches that spot, while
        # internal pressure starts moving as soon as the seal goes.
        p_int = self._internal_pressure_hpa()
        if p_int is not None:
            out["Enclosure"] = SensorStat(
                "Enclosure", None,
                Conn.FAULT if ls.pressure_alarm else Conn.ONLINE,
                f"{p_int:.0f} hPa internal"
                + ("  CRITICAL" if ls.pressure_alarm else ""))
        ekf = self._latest.get("EKF_STATUS_REPORT")
        if ekf is not None:
            var = max(float(ekf.get(k, 0.0) or 0.0) for k in
                      ("velocity_variance", "pos_horiz_variance",
                       "pos_vert_variance", "compass_variance"))
            out["EKF"] = SensorStat("EKF", None,
                                    Conn.ONLINE if var < 0.5 else Conn.DEGRADED,
                                    f"var {var:.2f}")
        return out

    def _publish(self) -> None:
        v_mv = self._get("BATTERY_STATUS", "voltages")
        volt = None
        if isinstance(v_mv, (list, tuple)) and v_mv and v_mv[0] not in (0, 65535):
            volt = v_mv[0] / 1000.0
        elif self._get("SYS_STATUS", "voltage_battery"):
            volt = float(self._get("SYS_STATUS", "voltage_battery")) / 1000.0

        amps = self._get("BATTERY_STATUS", "current_battery",
                         self._get("SYS_STATUS", "current_battery"))
        amps = None if amps in (None, -1) else float(amps) / 100.0
        mah = self._get("BATTERY_STATUS", "current_consumed")
        mah = None if mah in (None, -1) else float(mah)

        # Remaining charge, best source first. ArduSub reports -1 for
        # battery_remaining unless BATT_CAPACITY and a current monitor are
        # configured, which is why this panel used to show "--" on the real
        # vehicle while showing 95% in the demo.
        pct = self._get("BATTERY_STATUS", "battery_remaining",
                        self._get("SYS_STATUS", "battery_remaining"))
        pct = None if pct in (None, -1) else float(pct)
        source = "vehicle" if pct is not None else ""
        left_mah = None
        capacity = float(getattr(self.opts, "battery_capacity_mah", 0) or 0)
        if capacity > 0 and mah is not None:
            left_mah = max(0.0, capacity - mah)
            if pct is None:
                # Coulomb counting: a real integration of measured current, so
                # it beats the voltage curve. It does assume the pack started
                # full and that the vehicle's counter was reset with it.
                pct = max(0.0, min(100.0, 100.0 * left_mah / capacity))
                source = "mah"
        if pct is None and volt is not None:
            cells = int(getattr(self.opts, "battery_cells", 0) or 0)
            cells = cells if cells > 0 else cells_from_voltage(volt)
            pct = soc_from_voltage(volt, cells)
            source = "volts"
            if capacity > 0:
                left_mah = capacity * pct / 100.0

        temp_c = self._get("SCALED_PRESSURE2", "temperature")
        temp_c = None if temp_c is None else float(temp_c) / 100.0
        heading = self._get("VFR_HUD", "heading")
        yaw = self._get("ATTITUDE", "yaw")

        connected = bool(self.logger and self.logger.connected)
        armed, mode = self._arm_state()
        tel = Telemetry(
            battery_v=volt, battery_pct=pct, battery_pct_source=source,
            battery_left_mah=left_mah, current_a=amps, consumed_mah=mah,
            roll=self._get("ATTITUDE", "roll"),
            pitch=self._get("ATTITUDE", "pitch"),
            yaw=yaw,
            depth_m=self._depth_m(),
            heading_deg=(float(heading) if heading is not None
                         else (math.degrees(yaw) % 360.0 if yaw is not None else None)),
            water_temp_c=temp_c,
            armed=armed, mode=mode,
            # Tri-state on purpose: None means "this vehicle would not tell
            # us", which is NOT the same as dry and must not paint like it.
            leak=self._leak.state().leak,
            sensors=self._sensors(),
            conn=Conn.ONLINE if connected else Conn.OFFLINE,
            stamp=now())
        self.bus.telemetry.emit(tel)
        self._publish_thrusters()

    def _publish_imu(self) -> None:
        """The MPC's half of this worker: ATTITUDE + SCALED_IMU2 + pressure
        depth as one NED/FRD snapshot, at the tick rate (20 Hz with --mpc).

        ArduSub already speaks NED/FRD, so nothing is re-signed here —
        re-signing sensor data away from the message definition is how frame
        bugs start. SCALED_IMU2 carries SPECIFIC FORCE in mg (level and still
        reads ~(0,0,-1000)); the conversion to body acceleration (gravity,
        transport term) belongs to the state assembler, which owns the one
        rotation matrix it needs.
        """
        att = self._latest.get("ATTITUDE")
        imu = self._latest.get("SCALED_IMU2") or self._latest.get("RAW_IMU")
        baro = self._latest.get("SCALED_PRESSURE2") \
            or self._latest.get("SCALED_PRESSURE")
        v = VehicleImu(conn=Conn.ONLINE if att is not None else Conn.OFFLINE,
                       stamp=now())
        if att is not None:
            v.roll = att.get("roll")
            v.pitch = att.get("pitch")
            v.yaw = att.get("yaw")
            v.p = att.get("rollspeed")
            v.q = att.get("pitchspeed")
            v.r = att.get("yawspeed")
            v.t_att = att.get("t_host")
        if imu is not None:
            MG = 9.80665 / 1000.0            # SCALED_IMU2 accel is mg
            for src, dst in (("xacc", "ax"), ("yacc", "ay"), ("zacc", "az")):
                raw = imu.get(src)
                if raw is not None:
                    setattr(v, dst, float(raw) * MG)
            v.t_imu = imu.get("t_host")
        v.depth_m = self._depth_m()
        if baro is not None:
            v.t_baro = baro.get("t_host")
        self.bus.vehicle_imu.emit(v)

    def _arm_state(self) -> tuple[bool | None, str]:
        """(armed, mode) from the autopilot's own heartbeat, or (None, "").

        None means "I do not know", and the UI draws that as ``MOTORS ?`` — the
        only honest answer when no autopilot heartbeat has arrived. It must
        never fall back to a stale or foreign value: this indicator is what a
        pilot checks before putting hands near the thrusters.
        """
        rec = self._latest.get("HEARTBEAT")
        if rec is None:
            return None, ""
        # Age the RECORD, not this republish tick: `_latest` retains the last
        # heartbeat forever, so without this a died stream keeps reading as
        # "armed" — the same class of bug as the 2026-08-07 thruster-panel
        # incident (see _publish_thrusters), and the armed/mode interlock of
        # the MPC worker consumes this value. 3 s = 3 missed 1 Hz beats.
        if now() - float(rec.get("t_host", 0.0)) > 3.0:
            return None, ""
        from pymavlink import mavutil

        base = int(rec.get("base_mode", 0) or 0)
        armed = bool(base & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        custom = int(rec.get("custom_mode", -1) or -1)
        mapping = mavutil.mode_mapping_bynumber(int(rec.get("type", 12) or 12)) or {}
        return armed, mapping.get(custom, f"mode {custom}")

    def _publish_thrusters(self) -> None:
        rec = self._latest.get("SERVO_OUTPUT_RAW")
        n = int(getattr(self.opts, "thrusters", 8) or 8)
        if rec is None:
            self.bus.thrusters.emit(ThrusterState.blank(n))
            return
        pwm, health, norm = [], [], []
        for i in range(1, n + 1):
            raw = rec.get(f"servo{i}_raw")
            # 0 means "this output is not configured", which is different from
            # "this motor is at neutral" — the panel must show them differently.
            if raw in (None, 0):
                pwm.append(None)
                health.append(Conn.OFFLINE)
                norm.append(0.0)
            else:
                pwm.append(int(raw))
                health.append(Conn.ONLINE)
                norm.append(pwm_to_norm(int(raw)))
        # Age the panel against the MESSAGE, not against this tick.
        #
        # This worker republishes `_latest` every 100 ms whether or not a new
        # SERVO_OUTPUT_RAW arrived, and it used to stamp each one `now()`. So a
        # feed that had stopped entirely still looked perfectly live: the
        # watchdog was being fed our own heartbeat instead of the vehicle's
        # data, and the last values received — all 1500 us from before the
        # vehicle armed — stayed on screen with a green pill while the motors
        # were actually turning. Reported from the real vehicle 2026-08-07.
        #
        # The stamp is now the record's own arrival time, so silence ages the
        # panel exactly like every other source in this package.
        stamp = float(rec.get("t_host", now()))
        age = now() - stamp
        conn = Conn.ONLINE if age < 1.0 else (
            Conn.DEGRADED if age < 4.0 else Conn.STALE)
        stats = self.logger.stats if self.logger else {}
        srv = stats.get("SERVO_OUTPUT_RAW")
        self.bus.thrusters.emit(ThrusterState(
            n=n, norm=norm, pwm_us=pwm, health=health,
            labels=[f"T{i + 1}" for i in range(n)],
            conn=conn, hz=(srv.hz if srv else None), stamp=stamp))
        # Every output, not just 9-16: the payload usually lives past the
        # thrusters (ArduPilot reports 9-16 in this message's extension fields),
        # but the camera mount's channel depends on the frame — a 6-thruster
        # BlueROV2 can have tilt on 7 or 8 — and the sink only ever looks up the
        # one channel it was explicitly told about.
        aux: dict = {i: int(rec[f"servo{i}_raw"]) for i in range(1, 17)
                     if rec.get(f"servo{i}_raw")}
        # The mount's OWN reported angle, when it reports one. Preferred over
        # the servo PWM because it is degrees the vehicle believes, not a pulse
        # width we would have to map through an unknown servo range.
        # MOUNT_STATUS.pointing_a is pitch in centidegrees.
        mount = self._latest.get("MOUNT_STATUS")
        if mount is not None and mount.get("pointing_a") is not None:
            aux["mount_deg"] = float(mount["pointing_a"]) / 100.0
        if aux:
            self.bus.aux_servos.emit(aux)

    @Slot(bool, str)
    def set_sensor_log(self, on: bool, stem: str) -> None:
        """Queued onto this worker's event loop, so the file is opened, written
        and closed by the one thread that drains the vehicle."""
        if on and self._log is None:
            self._log = SensorLog(Path(f"{stem}_rov.jsonl"), "rov")
            self.bus.log.emit("info", f"rov sensor log -> {self._log.path}")
        elif not on and self._log is not None:
            meta = self._log.close()
            self._log = None
            rates = ", ".join(f"{k} {v:.0f} Hz" for k, v in
                              sorted(meta["hz_measured"].items())[:4])
            self.bus.log.emit("info", f"rov sensor log closed: "
                                      f"{meta['records']} records ({rates})")

    def teardown(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None
        if self.logger is not None:
            self.logger.close()
            self.logger = None


# =============================================================================
# command sinks
# =============================================================================
class NullCommandSink(TimerWorker):
    """The default. Accepts every command and transmits nothing.

    Not a no-op class: it still reports payload state back to the UI (so the
    gripper and lights panels show the *command* and say "no link"), and it
    still runs the deadman logic, so the same code path is exercised whether or
    not transmission is enabled.
    """

    def __init__(self, bus, opts):
        super().__init__("cmd-null", interval_ms=100)
        self.bus = bus
        self.opts = opts
        self.pilot = PilotInput()
        self.grip = 0.0
        self.lights = 0.0
        # tick() reads this like MavlinkCommandSink does; only that class ever
        # measured it off a servo. Leaving it undefined made every tick of a
        # no-command build raise AttributeError into the failed-log (found
        # 2026-08-12 while wiring the MPC).
        self.lights_fb: float | None = None
        self.enabled = False
        self._warned = False

    @Slot(object)
    def set_pilot(self, cmd: PilotInput) -> None:
        self.pilot = cmd
        if cmd.any_axis and self.enabled and not self._warned:
            self._warned = True
            self.bus.log.emit(
                "warn", "commands are ENABLED but this build has no command sink "
                        "— nothing is being transmitted (start with --allow-command)")

    @Slot(float)
    def set_gripper(self, v: float) -> None:
        self.grip = v

    @Slot(float)
    def set_lights(self, v: float) -> None:
        self.lights = v

    @Slot(float)
    def set_gripper_drive(self, direction: float) -> None:
        pass

    @Slot(int)
    def set_lights_step(self, steps: int) -> None:
        pass

    @Slot(float)
    def set_tilt(self, direction: float) -> None:
        pass

    @Slot()
    def tilt_center(self) -> None:
        pass

    @Slot(int)
    def set_js_buttons(self, mask: int) -> None:
        pass

    @Slot(str)
    def set_mode(self, name: str) -> None:
        self.bus.log.emit(
            "warn", f"flight mode {name.upper()} requested, but this build has "
                    f"no command sink — nothing was transmitted "
                    f"(start with --allow-command)")

    @Slot(object)
    def note_aux_servos(self, servos: dict) -> None:
        pass

    @Slot(bool)
    def set_enabled(self, on: bool) -> None:
        self.enabled = on
        self._warned = False

    @Slot()
    def estop(self) -> None:
        self.pilot = PilotInput()
        self.enabled = False

    @Slot(bool)
    def set_arm(self, arm: bool) -> None:
        self.bus.log.emit(
            "warn", f"{'ARM' if arm else 'DISARM'} requested, but this build has "
                    f"no command sink — nothing was transmitted "
                    f"(start with --allow-command)")

    def tick(self) -> None:
        self.bus.payload.emit(PayloadState(
            gripper_cmd=self.grip, gripper_fb=None, gripper_conn=Conn.OFFLINE,
            lights_cmd=self.lights, lights_fb=self.lights_fb, lights_conn=Conn.OFFLINE,
            stamp=now()))

    def status(self) -> tuple[Conn, str]:
        return Conn.OFFLINE, "no command sink"


class MavlinkCommandSink(TimerWorker):
    """MANUAL_CONTROL at a steady rate, with a heartbeat and a deadman.

    Every requirement listed in ``c3_camera/control.py`` is implemented here and
    labelled. Read that file before changing anything in this class.

    Flight-mode changes ARE implemented, which reverses what this docstring
    used to say. The original reasoning — "the station is a command source, not
    an authority, so mode stays in QGC" — did not survive contact with the
    vehicle: leaving the mode alone means arming into whatever mode the vehicle
    happened to be left in, and on 2026-08-07 that was a stabilising one, so
    pressing ARM ran the thrusters immediately with no stick input. A station
    that can arm a vehicle but cannot tell it how to interpret being armed is
    not safer, it is just less predictable. So the mode is explicit, shown from
    the vehicle's own HEARTBEAT, and never changed except when asked.

    The z-axis trap: ArduSub's MANUAL_CONTROL ``z`` has used both 0..1000 and
    −1000..1000 in different vehicle types, and getting it wrong makes the
    vehicle dive when you meant neutral. ``z_convention`` makes the choice
    explicit rather than assumed, and the default is the one ArduSub documents
    for Sub: 0 = full down, 500 = neutral, 1000 = full up.

    Where the commands actually go
    ------------------------------
    NOT ``udpout:192.168.2.2:14550``. That was this class's first default and it
    silently did nothing: probed on 2026-08-06, UDP 14550 on the vehicle answers
    ICMP port-unreachable, because BlueOS's inbound "GCS Server Link" endpoint is
    disabled and its outbound "GCS Client Link" pushes from an *ephemeral* source
    port. Nothing on the vehicle is listening on 14550.

    So this sink does what QGroundControl does: it **binds a port BlueOS pushes
    to, and replies to whoever it heard from**. pymavlink's ``udpin:`` connection
    tracks every client that has sent to it and ``write()`` fans out to them, so
    receiving is what makes transmitting possible. That means:

    * the socket must be DRAINED even though this class does not care about
      telemetry — no recv, no peer, no commands. :meth:`tick` does that first;
    * BlueOS must be pushing to that port, i.e. the endpoint has to exist:
      ``./c3 blueos_endpoint add --port 14552 --yes``;
    * until a datagram arrives there is nowhere to send, and the UI says so
      (CMD pill goes amber with "no vehicle on udp/14552") instead of pretending
      a command was delivered. That invisibility is exactly what made the first
      version look like it worked.
    """

    def __init__(self, bus, opts):
        super().__init__("cmd-mavlink", interval_ms=50)     # 20 Hz
        self.bus = bus
        self.opts = opts
        self.master = None
        self.pilot = PilotInput()
        self.enabled = False
        self.grip = 0.0
        self.lights = 0.0
        self.deadman_s = float(getattr(opts, "deadman_ms", 500)) / 1000.0
        self.z_convention = getattr(opts, "z_convention", "0..1000")
        self._last_hb = 0.0
        self._sent = 0
        self._last_rate_t = time.monotonic()
        self._rate_hz = 0.0
        self._neutral_latched = False
        # Button bits for gripper/lights. None => this station does not touch
        # them, and the payload panel says so, rather than sending a bit that
        # is mapped to something else on this vehicle.
        def _bit(name: str) -> int | None:
            # -1 (or absent) means "this station does not touch that function".
            v = getattr(opts, name, None)
            return None if v is None or int(v) < 0 else int(v)

        self.bit_grip_open = _bit("btn_gripper_open")
        self.bit_grip_close = _bit("btn_gripper_close")
        self.bit_lights_up = _bit("btn_lights_up")
        self.bit_lights_down = _bit("btn_lights_down")
        self.bit_tilt_up = _bit("btn_tilt_up")
        self.bit_tilt_down = _bit("btn_tilt_down")
        self.bit_tilt_center = _bit("btn_tilt_center")
        self._btn_checked: set[str] = set()
        self.grip_drive = 0.0            # -1 close / 0 idle / +1 open, HELD
        self.tilt_drive = 0.0            # -1 down / 0 idle / +1 up, HELD
        self.tilt_deg: float | None = None
        _tservo = getattr(opts, "tilt_servo", -1)
        self.tilt_servo = None if _tservo is None or int(_tservo) < 0 else int(_tservo)
        self.tilt_min_deg = float(getattr(opts, "tilt_min_deg", -45.0))
        self.tilt_max_deg = float(getattr(opts, "tilt_max_deg", 45.0))
        # The gamepad's raw bitmask, forwarded so the VEHICLE decides what each
        # button means. See set_js_buttons.
        self.js_buttons = 0
        self.js_passthrough = bool(getattr(opts, "js_passthrough", True))
        self.lights_steps = int(getattr(opts, "lights_steps", 8) or 8)
        self._lights_notch = 0           # our model of where the light is
        _servo = getattr(opts, "lights_servo", None)
        self.lights_servo = None if _servo is None or int(_servo) < 0 else int(_servo)
        self.lights_fb: float | None = None
        self._pulses: list[int] = []     # queued button PRESSES, one bit each
        self._pulse_bit = -1
        self._pulse_on = 0
        self._pulse_gap = 0
        # A udpout connection never receives, so pymavlink's target_system stays
        # 0 and would address "everyone". State it instead of inferring it.
        self.sysid = int(getattr(opts, "cmd_sysid", 255) or 255)
        self.sysid_mygcs: float | None = None    # the vehicle's, once we read it
        # ArduSub's leak-detector configuration, read back so the Leak row can
        # distinguish "dry" from "would never say" (rov_gui/leak.py).
        self._leak_cfg: dict = {}
        self.leak_cfg_read = False
        self._param_asked = 0.0
        self.target_sys = int(getattr(opts, "target_sysid", 1) or 1)
        self.target_comp = 1                          # MAV_COMP_ID_AUTOPILOT1
        self.listening = False

    def setup(self) -> None:
        from pymavlink import mavutil

        target = getattr(self.opts, "mavlink_out", "udpin:0.0.0.0:14552")
        self.master = mavutil.mavlink_connection(
            target, source_system=self.sysid, source_component=190)
        self.listening = target.startswith("udpin")
        self.bus.log.emit("warn",
                          f"command sink OPEN on {target} (sysid "
                          f"{self.opts.cmd_sysid}) — it transmits only while "
                          f"COMMAND ENABLE is on")
        if self.listening:
            self.bus.log.emit(
                "info", "command sink replies to whoever pushes to that port; "
                        "if nothing arrives run: ./c3 blueos_endpoint add "
                        f"--port {target.rsplit(':', 1)[-1]} --yes")

    # -------------------------------------------------------------- commands
    @Slot(object)
    def set_pilot(self, cmd: PilotInput) -> None:
        self.pilot = cmd

    @Slot(float)
    def set_gripper(self, v: float) -> None:
        self.grip = v                    # display only; the vehicle has no position

    @Slot(float)
    def set_lights(self, v: float) -> None:
        """Drive the light to an ABSOLUTE level, in vehicle notches.

        ArduSub's lights1_brighter/dimmer move one notch per button PRESS and
        report nothing back, so a station can only know the level by counting
        its own presses — and one lost press desynchronises it forever. That is
        what "70% shown, but the light looks off" was.

        Two things fix it. The level is quantised to the vehicle's own
        JS_LIGHTS_STEPS (read from the parameter; 8 on this vehicle, not the 10
        this code first assumed), so every slider position is reachable. And the
        two ends are RESYNC points: asking for 0 or full drives past the stop by
        two extra presses, which costs a moment and makes the model true again.
        """
        self.lights = max(0.0, min(1.0, float(v)))
        steps = max(1, int(self.lights_steps))
        target = int(round(self.lights * steps))
        if target <= 0:
            self._pulses.clear()
            self.queue_presses(self.bit_lights_down, steps + 2)   # to the stop
        elif target >= steps:
            self._pulses.clear()
            self.queue_presses(self.bit_lights_up, steps + 2)
        else:
            delta = target - self._lights_notch
            if delta > 0:
                self.queue_presses(self.bit_lights_up, delta)
            elif delta < 0:
                self.queue_presses(self.bit_lights_down, -delta)
        self._lights_notch = target

    @Slot(float)
    def set_gripper_drive(self, direction: float) -> None:
        self.grip_drive = float(direction)

    @Slot(int)
    def set_lights_step(self, steps: int) -> None:
        bit = self.bit_lights_up if steps > 0 else self.bit_lights_down
        self.queue_presses(bit, abs(int(steps)))

    @Slot(float)
    def set_tilt(self, direction: float) -> None:
        """Camera mount tilt, HELD like the gripper.

        ArduSub services mount_tilt_up/down from its button-REPEAT path, so the
        mount keeps moving while the bit is set and stops when it clears — the
        same shape as servo_1_*_momentary, and NOT the one-action-per-press
        shape the lights use.
        """
        self.tilt_drive = float(direction)

    @Slot()
    def tilt_center(self) -> None:
        self.queue_presses(self.bit_tilt_center, 1)

    @Slot(str)
    def set_mode(self, name: str) -> None:
        """Ask the vehicle for a flight mode, by ArduSub name.

        MAV_CMD_DO_SET_MODE with CUSTOM_MODE_ENABLED — the same thing QGC
        sends. Note what is NOT here: no local record of "the mode we asked
        for". The mode the panel shows always comes back from HEARTBEAT, so a
        request that the vehicle rejected (it refuses some transitions while
        armed) shows as the mode it actually stayed in, not the one we wanted.
        """
        if self.master is None:
            return
        num = MODE_NUMBERS.get(name.upper())
        if num is None:
            self.bus.log.emit("error", f"unknown flight mode {name!r}")
            return
        from pymavlink import mavutil

        try:
            self.master.mav.command_long_send(
                self.target_sys, self.target_comp,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                num, 0, 0, 0, 0, 0)
            self.bus.log.emit("warn", f"requested flight mode {name.upper()}")
        except Exception as e:                                   # noqa: BLE001
            self.bus.log.emit("error", f"set_mode({name}) failed: {e}")

    @Slot(int)
    def set_js_buttons(self, mask: int) -> None:
        """The gamepad's own bitmask, forwarded verbatim to the vehicle.

        This is the whole fix for "the pad does something different here than in
        QGC". QGC sends the raw button bitmask and lets the vehicle's
        BTNn_FUNCTION parameters decide what each one means; this station used to
        keep its OWN pad map (--js-btn-*, defaulting to buttons 0/3/4/5) and
        synthesise a different set of bits. So the same physical button was
        gripper-close here and `disarm` in QGC.

        Now there is one map and the vehicle owns it. Whatever the operator
        configured in QGC > Joystick > Button Assignment is what the pad does
        here, including the functions this station has no UI for (mode changes,
        gain, trim, input_hold_set, shift).
        """
        self.js_buttons = int(mask) & 0xFFFF

    @Slot(bool)
    def set_enabled(self, on: bool) -> None:
        self.enabled = bool(on)
        if not on:
            self.grip_drive = 0.0
            self.tilt_drive = 0.0
            self.js_buttons = 0
            self._pulses.clear()
            self._pulse_on = self._pulse_gap = 0
            self._send(PilotInput(), buttons=0)     # one explicit neutral
            self.bus.log.emit("info", "command sink disabled — sent neutral")

    @Slot()
    def estop(self) -> None:
        self.enabled = False
        self.pilot = PilotInput()
        self.grip_drive = 0.0
        self.tilt_drive = 0.0
        self.js_buttons = 0
        self._pulses.clear()
        self._pulse_on = self._pulse_gap = 0
        for _ in range(3):                          # neutral, three times
            self._send(PilotInput(), buttons=0)
        self.bus.log.emit("error", "E-STOP: neutral sent, commands disabled")

    @Slot(object)
    def note_aux_servos(self, servos: dict) -> None:
        """SERVO_OUTPUT_RAW for the aux channels, from the telemetry worker.

        If the operator told us which output the light is on, this turns the
        panel's level from a guess into a MEASUREMENT — and, on the way, tells
        us our press-counting model was wrong before the pilot notices.
        """
        # The mount's own angle first; the servo PWM only if the operator named
        # a channel. A pulse width is not an angle without knowing the servo's
        # range, so it is converted with the ArduPilot default 1100-1900 us over
        # the mount's configured limits and labelled [유도] in the panel.
        if "mount_deg" in servos:
            self.tilt_deg = float(servos["mount_deg"])
        elif self.tilt_servo is not None:
            pwm = servos.get(int(self.tilt_servo))
            if pwm:
                span = float(self.tilt_max_deg - self.tilt_min_deg)
                frac = max(0.0, min(1.0, (float(pwm) - 1100.0) / 800.0))
                self.tilt_deg = self.tilt_min_deg + frac * span

        if self.lights_servo is None:
            return
        pwm = servos.get(int(self.lights_servo))
        if not pwm:
            return
        self.lights_fb = max(0.0, min(1.0, (float(pwm) - 1100.0) / 800.0))

    @Slot(bool)
    def set_arm(self, arm: bool) -> None:
        """MAV_CMD_COMPONENT_ARM_DISARM. The one command here that is not a
        pilot axis, and the only one that changes what the vehicle *is*.

        Two asymmetries, both deliberate:

        * ARM requires the COMMAND ENABLE switch; DISARM does not. A stop must
          never be gated behind another control — if this station can reach the
          vehicle at all, it can stop it.
        * ARM sends the axes to neutral first. Arming while an input is held
          means the vehicle moves the moment the motors go live.

        The UI never shows "armed" because we sent this. It shows what the
        vehicle's HEARTBEAT says, so a command that was refused (failsafe, bad
        pre-arm check) looks like what it is: nothing happened.
        """
        from pymavlink import mavutil

        if self.master is None:
            return
        if arm and not self.enabled:
            self.bus.log.emit("warn", "ARM ignored — COMMAND ENABLE is off")
            return
        if self.listening and self._peer_count() == 0:
            port = getattr(self.opts, "mavlink_out", "?").rsplit(":", 1)[-1]
            self.bus.log.emit(
                "error", f"{'ARM' if arm else 'DISARM'} NOT SENT — nothing has "
                         f"pushed MAVLink to udp/{port}, so this station has no "
                         f"address to reply to. Run: ./c3 blueos_endpoint add "
                         f"--port {port} --yes")
            return
        if arm:
            self._send(PilotInput(), buttons=0)      # neutral before live
        self.master.mav.command_long_send(
            self.target_sys, self.target_comp,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
            1 if arm else 0,                          # param1: arm/disarm
            0,                                        # param2: 0 = no force
            0, 0, 0, 0, 0)
        self.bus.log.emit("warn" if arm else "info",
                          f"sent {'ARM' if arm else 'DISARM'} to sysid "
                          f"{self.target_sys} — watch MOTORS for the vehicle's "
                          f"own answer")

    # ------------------------------------------------------------------ send
    def tick(self) -> None:
        t = time.monotonic()
        self._drain()
        self._check_gcs_sysid()
        # The payload panel shows the COMMAND even when nothing is transmitted,
        # with no feedback bar — see widgets/payload.py on why those must not be
        # the same number.
        self.bus.payload.emit(PayloadState(
            gripper_cmd=self.grip, gripper_fb=None,
            gripper_conn=Conn.ONLINE if (self.enabled and self.bit_grip_open
                                         is not None) else Conn.OFFLINE,
            gripper_note=("hold" if self.bit_grip_open is not None
                          else "no button bit mapped"),
            lights_cmd=self.lights, lights_fb=self.lights_fb,
            lights_conn=Conn.ONLINE if (self.enabled and self.bit_lights_up
                                        is not None) else Conn.OFFLINE,
            lights_note=(f"{len(self._pulses)} press(es) queued"
                         if self._pulses else
                         (f"measured on servo{self.lights_servo}"
                          if self.lights_fb is not None else
                          (f"{self.lights_steps} steps, no feedback"
                           if self.bit_lights_up is not None
                           else "no button bit mapped"))),
            tilt_deg=self.tilt_deg, tilt_drive=self.tilt_drive,
            tilt_conn=(Conn.ONLINE if (self.enabled and self.bit_tilt_up is not None)
                       else Conn.OFFLINE),
            tilt_note=("hold" if self.tilt_deg is None and self.bit_tilt_up is not None
                       else ("MOUNT_STATUS" if self.tilt_deg is not None
                             and self.tilt_servo is None
                             else (f"[유도] servo{self.tilt_servo}"
                                   if self.tilt_deg is not None
                                   else "no button bit mapped"))),
            stamp=now()))
        if not self.enabled:
            return

        # HEARTBEAT at 1 Hz: ArduSub's GCS failsafe is what stops the vehicle
        # if this station dies mid-command. Only while enabled, so a passive
        # station is never counted as the commanding GCS.
        if t - self._last_hb >= 1.0:
            self._last_hb = t
            self._heartbeat()

        stale = (t - self.pilot.stamp) > self.deadman_s
        cmd = PilotInput() if stale else self.pilot
        if stale and not self._neutral_latched:
            self._neutral_latched = True
            # THE HELD DRIVES DIE WITH THE DEADMAN, not just the axes. They
            # are latched LEVELS (set_gripper_drive/set_tilt): whoever set one
            # non-zero must set it back, and a worker that died mid-hold never
            # will — while this sink thread keeps heartbeating, so ArduSub's
            # GCS failsafe never trips either. Before 2026-08-30 a dead
            # MpcWorker with a replay jaw CLOSE latched would have kept the
            # close bit on the wire at 20 Hz forever (safety review, replay
            # change set). Fresh pilot input re-latches whatever is still
            # being held, so a live G/H key simply re-asserts on the next
            # frame.
            self.grip_drive = 0.0
            self.tilt_drive = 0.0
            self.bus.log.emit("warn",
                              f"deadman: no input for {self.deadman_s * 1000:.0f} ms "
                              f"— sending neutral (held drives released)")
        elif not stale:
            self._neutral_latched = False
        self._send(cmd, buttons=self._buttons())

        if t - self._last_rate_t >= 1.0:
            self._rate_hz = self._sent / (t - self._last_rate_t)
            self._sent, self._last_rate_t = 0, t

    def _check_gcs_sysid(self) -> None:
        """Read SYSID_MYGCS and complain if it is not us.

        ArduPilot accepts pilot input from exactly ONE ground station:

            void GCS_MAVLINK::handle_manual_control(const mavlink_message_t &msg) {
                if (msg.sysid != sysid_my_gcs()) {
                    return;             // only accept control from our gcs
                }

        MANUAL_CONTROL from any other system id is dropped without a NAK, an
        error, or any other sign. COMMAND_LONG is NOT gated this way — which is
        why, on 2026-08-06, this station could ARM the vehicle and then move
        nothing: armed, transmitting at 20 Hz, every thruster at 1500 us.
        Measured on this vehicle: SYSID_MYGCS = 255, and this sink defaulted to
        245.

        So it is read once, and a mismatch becomes a red CMD pill instead of a
        mystery. Read-only: PARAM_REQUEST_READ asks, it does not set.
        """
        if self.master is None:
            return
        if (self.sysid_mygcs is not None
                and self.leak_cfg_read
                and len(self._btn_checked) >= len(self._mapped_buttons())):
            return
        t = time.monotonic()
        if t - self._param_asked > 3.0:
            self._param_asked = t
            try:
                # FS_LEAK_ENABLE / LEAK1_PIN answer the question a leak
                # indicator cannot answer for itself: would this vehicle say
                # anything if it flooded? Either at its disabling value makes
                # a flood completely silent on the link (rov_gui/leak.py), so
                # without them the panel can only say "unknown", never "dry".
                for name in (b"SYSID_MYGCS", b"JS_LIGHTS_STEPS",
                             b"FS_LEAK_ENABLE", b"LEAK1_PIN"):
                    self.master.mav.param_request_read_send(
                        self.target_sys, self.target_comp, name, -1)
                for key, bit in self._mapped_buttons().items():
                    if key in self._btn_checked:
                        continue
                    self.master.mav.param_request_read_send(
                        self.target_sys, self.target_comp,
                        f"BTN{bit}_FUNCTION".encode(), -1)
            except Exception:                                    # noqa: BLE001
                pass

    def _mapped_buttons(self) -> dict:
        return {k: v for k, v in (("gripper_open", self.bit_grip_open),
                                  ("gripper_close", self.bit_grip_close),
                                  ("lights_up", self.bit_lights_up),
                                  ("lights_down", self.bit_lights_down),
                                  ("tilt_up", self.bit_tilt_up),
                                  ("tilt_down", self.bit_tilt_down),
                                  ("tilt_center", self.bit_tilt_center))
                if v is not None}

    # Which attribute holds each function's button number, so a function that
    # fails its check can be switched off. Kept as one table rather than inline:
    # the inline version was missing the tilt entries, and a KeyError inside a
    # parameter callback would have taken the command sink down at exactly the
    # moment it was reporting a misconfiguration.
    _BIT_ATTR = {
        "gripper_open": "bit_grip_open", "gripper_close": "bit_grip_close",
        "lights_up": "bit_lights_up", "lights_down": "bit_lights_down",
        "tilt_up": "bit_tilt_up", "tilt_down": "bit_tilt_down",
        "tilt_center": "bit_tilt_center",
    }

    def _note_button_function(self, bit: int, value: int) -> None:
        """Confirm a button does what --btn-* says, or stop using it."""
        for key, mapped in list(self._mapped_buttons().items()):
            if mapped != bit:
                continue
            self._btn_checked.add(key)
            want, label = BTN_FUNCTION[key]
            if value == want:
                self.bus.log.emit("info", f"button {bit} = {label} — {key} ok")
                continue
            # Two things can be wrong here and the message must not pick one:
            # the button may be assigned to something else on this vehicle, or
            # OUR expected function number may be wrong. The mount functions in
            # particular are from ArduSub's JSButton enum and have not been
            # confirmed against this vehicle. Either way the safe move is the
            # same — do not press a button whose meaning we cannot confirm.
            self.bus.log.emit(
                "error",
                f"button {bit} is BTN{bit}_FUNCTION={value}, and this station "
                f"expected {label} ({want}) for {key} — NOT pressing it. Either "
                f"reassign it in QGC > Joystick > Button Assignment (or pass "
                f"--btn-{key.replace('_', '-')} <n>), or {value} is what {key} "
                f"really is on this firmware and BTN_FUNCTION here is wrong.")
            attr = self._BIT_ATTR.get(key)
            if attr:
                setattr(self, attr, None)

    def _note_param(self, msg) -> None:
        pid = getattr(msg, "param_id", "")
        pid = pid.strip("\x00") if isinstance(pid, str) else str(pid)
        if pid.startswith("BTN") and pid.endswith("_FUNCTION"):
            try:
                self._note_button_function(int(pid[3:-9]), int(msg.param_value))
            except ValueError:
                pass
            return
        if pid in ("FS_LEAK_ENABLE", "LEAK1_PIN"):
            self._leak_cfg[pid] = int(msg.param_value)
            if len(self._leak_cfg) >= 2 and not self.leak_cfg_read:
                self.leak_cfg_read = True
                fs = self._leak_cfg.get("FS_LEAK_ENABLE")
                pin = self._leak_cfg.get("LEAK1_PIN")
                if fs == 0 or (pin is not None and pin < 0):
                    self.bus.log.emit(
                        "warn",
                        f"leak detector is NOT armed on this vehicle "
                        f"(FS_LEAK_ENABLE={fs}, LEAK1_PIN={pin}) — a flood "
                        f"would be silent. Set LEAK1_PIN=27 (Navigator "
                        f"built-in) and FS_LEAK_ENABLE=1 in QGC to arm it.")
                else:
                    self.bus.log.emit(
                        "info", f"leak detector armed (FS_LEAK_ENABLE={fs}, "
                                f"LEAK1_PIN={pin})")
            return
        if pid == "JS_LIGHTS_STEPS":
            steps = max(1, int(msg.param_value))
            if steps != self.lights_steps:
                self.bus.log.emit(
                    "info", f"JS_LIGHTS_STEPS = {steps} (was assuming "
                            f"{self.lights_steps}) — lights quantised to that")
                self.lights_steps = steps   # the window reads this back on tick
            return
        if pid != "SYSID_MYGCS":
            return
        self.sysid_mygcs = float(msg.param_value)
        if int(self.sysid_mygcs) == self.sysid:
            self.bus.log.emit("info", f"SYSID_MYGCS = {int(self.sysid_mygcs)} — "
                                      f"this station is the accepted GCS")
        else:
            self.bus.log.emit(
                "error",
                f"SYSID_MYGCS = {int(self.sysid_mygcs)} but this station sends as "
                f"{self.sysid}: ArduSub will ARM but IGNORE every stick input. "
                f"Fix with --cmd-sysid {int(self.sysid_mygcs)}, or set "
                f"SYSID_MYGCS={self.sysid} on the vehicle.")

    def _drain(self) -> None:
        """Read whatever arrived, purely so pymavlink learns the peer address.

        A udpin connection can only transmit to clients it has heard from. This
        sink ignores the content — the telemetry worker owns that — but without
        this loop ``write()`` has an empty client set and every command is
        discarded in userspace with no error.
        """
        if self.master is None or not self.listening:
            return
        for _ in range(40):                  # bounded: never starve the tick
            try:
                msg = self.master.recv_msg()
            except Exception:                                    # noqa: BLE001
                break
            if msg is None:
                break
            if msg.get_type() == "PARAM_VALUE":
                self._note_param(msg)

    def _peer_count(self) -> int:
        clients = getattr(self.master, "clients", None)
        if clients is not None:
            return len(clients)
        return 1 if self.master is not None else 0     # udpout: assume a target

    def _heartbeat(self) -> None:
        from pymavlink import mavutil
        self.master.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)

    # Two ticks (=100 ms at 20 Hz) down and two up. ArduSub reacts to the
    # CHANGE between consecutive MANUAL_CONTROL messages, so a press has to be
    # visible in more than one message to survive a lost packet, and has to be
    # released again or the next press is not a change.
    PULSE_ON_TICKS = 2
    PULSE_OFF_TICKS = 2

    def queue_presses(self, bit: int | None, count: int) -> None:
        """Queue ``count`` presses of one button bit (for stepped functions)."""
        if bit is None or count <= 0:
            return
        # Bounded: a slider dragged end to end must not turn into a minute of
        # queued button presses that keep firing after the pilot let go.
        room = max(0, 40 - len(self._pulses))
        self._pulses.extend([int(bit)] * min(count, room))

    def _pulse_bits(self) -> int:
        """The press-queue's contribution to this tick's bitmask."""
        if self._pulse_on > 0:
            self._pulse_on -= 1
            return 1 << self._pulse_bit
        if self._pulse_gap > 0:
            self._pulse_gap -= 1
            return 0
        if self._pulses:
            self._pulse_bit = self._pulses.pop(0)
            self._pulse_on = self.PULSE_ON_TICKS - 1
            self._pulse_gap = self.PULSE_OFF_TICKS
            return 1 << self._pulse_bit
        return 0

    def _buttons(self) -> int:
        """Bitmask for this message: the pad's own bits, plus what the UI holds.

        Two contributors, and they mean the same thing because they use the same
        numbering — the VEHICLE's (SDL/QGC), not the kernel's. On this vehicle
        (QGC > Joystick > Button Assignment) that map is:

            11 servo_1_max_momentary   gripper open    HELD while the button is
            12 servo_1_min_momentary   gripper close   down
            9  mount_tilt_down         camera tilt     HELD
            10 mount_tilt_up           camera tilt     HELD
            7  mount_center            camera level    per PRESS
            13 lights1_dimmer          one notch dimmer   per PRESS
            14 lights1_brighter        one notch brighter per PRESS

        The gripper lives on 11/12 — the D-pad — because its original close
        button (15 = MISC1 / "Share") does not exist on this pad: the kernel
        reports buttons 0-14 only, so that function could never be pressed.

        which is why the gripper and tilt read held directions while the lights
        read a queue of presses. Guessing that all of them were levels is what
        made an earlier build appear to do nothing.

        The pad's bits pass through untouched (``js_buttons``), so buttons this
        station has no UI for — mode changes, gain, trim, shift — still work and
        still do exactly what QGC says they do.
        """
        bits = self._pulse_bits() | (self.js_buttons if self.js_passthrough else 0)
        if self.grip_drive > 0.5 and self.bit_grip_open is not None:
            bits |= 1 << int(self.bit_grip_open)
        elif self.grip_drive < -0.5 and self.bit_grip_close is not None:
            bits |= 1 << int(self.bit_grip_close)
        if self.tilt_drive > 0.5 and self.bit_tilt_up is not None:
            bits |= 1 << int(self.bit_tilt_up)
        elif self.tilt_drive < -0.5 and self.bit_tilt_down is not None:
            bits |= 1 << int(self.bit_tilt_down)
        return bits & 0xFFFF

    @staticmethod
    def _clamp(v: float, lo: int, hi: int) -> int:
        return int(max(lo, min(hi, round(v))))

    def _send(self, cmd: PilotInput, buttons: int) -> None:
        if self.master is None:
            return
        if self.listening and self._peer_count() == 0:
            return                # nowhere to send; status() is already shouting
        if self.z_convention == "0..1000":
            z = self._clamp(500 + cmd.heave * 500, 0, 1000)
        else:
            z = self._clamp(cmd.heave * 1000, -1000, 1000)
        self.master.mav.manual_control_send(
            self.target_sys,
            self._clamp(cmd.surge * 1000, -1000, 1000),
            self._clamp(cmd.sway * 1000, -1000, 1000),
            z,
            self._clamp(cmd.yaw * 1000, -1000, 1000),
            buttons)
        self._sent += 1

    def leak_config(self) -> dict:
        """ArduSub's leak-detector parameters as far as they have been read.

        Handed to VehicleWorker by HardwareBackend. Plain-attribute read of
        ints written by this thread — no lock, for the same reason
        :meth:`status` needs none: a torn read of an int is not a thing
        CPython does, and a one-tick-stale value costs a repaint.
        """
        return {"fs_leak_enable": self._leak_cfg.get("FS_LEAK_ENABLE"),
                "leak1_pin": self._leak_cfg.get("LEAK1_PIN")}

    def status(self) -> tuple[Conn, str]:
        if (self.sysid_mygcs is not None
                and int(self.sysid_mygcs) != self.sysid):
            return Conn.FAULT, (f"SYSID_MYGCS={int(self.sysid_mygcs)} != {self.sysid}"
                                f" — sticks ignored")
        if self.listening and self._peer_count() == 0:
            # The failure that hid for a whole session: enabled, armed, keys
            # held, and not one byte leaving the host.
            port = getattr(self.opts, "mavlink_out", "?").rsplit(":", 1)[-1]
            return Conn.FAULT, f"no vehicle on udp/{port} — nowhere to send"
        if not self.enabled:
            return Conn.OFFLINE, "enable to transmit"
        return Conn.ONLINE, f"tx {self._rate_hz:.0f} Hz"

    def teardown(self) -> None:
        if self.master is not None:
            try:
                self._send(PilotInput(), buttons=0)      # leave it neutral
                self.master.close()
            except Exception:                             # noqa: BLE001
                pass
            self.master = None


# =============================================================================
# object tracking
# =============================================================================
class PoseWorker(TimerWorker):
    """Drives the SAM2 tracker from the frames C3VideoWorker taps.

    A ``TimerWorker`` and not a ``LoopWorker``, and that is forced rather than
    chosen: this worker has to RECEIVE — clicks, the enable toggle, reset — and
    only a TimerWorker has the event loop that delivers a queued slot
    (``backends/base.py``). The package already shipped one LoopWorker with a
    slot on it, and the slot silently never fired; see
    ``C3VideoWorker.request_sensor_log``.

    It never blocks: the mailbox take is a mutex, and the session's own work
    happens on the tracker's thread upstream.
    """

    def __init__(self, bus, mailbox, opts):
        super().__init__("pose", interval_ms=20)
        self.bus = bus
        self.mailbox = mailbox
        self.opts = opts
        self.session = None
        self.enabled = False
        self._last_pub = 0.0
        self._logged_intrinsics = False
        self._fault = ""
        self._log: SensorLog | None = None
        self._log_intr = None
        self._camera_written = False
        self._mesh_logged = False        # console announcement, once per mesh
        self._mesh_record_written = False  # MESH row, once per LOG FILE
        self._mesh_desc_logged = False   # the estimator's own one-line summary
        self._csv = None                 # flat pose rows beside the jsonl
        self._last_t_cap = 0.0           # capture stamp of the newest frame
        self._last_state = ""            # cursor for the LOG panel
        self._last_event_state = ""      # cursor for the jsonl EVENT rows

    @staticmethod
    def _K_matrix(intr):
        """Intrinsics -> 3x3, or None. c3_camera.geometry.Intrinsics has .K."""
        if intr is None:
            return None
        try:
            return intr.K
        except Exception:                                        # noqa: BLE001
            return None

    def setup(self) -> None:
        # Imported here, not at module scope: this pulls torch and SAM2, which
        # must not be a requirement for starting the station.
        from ..perception.session import PoseSession

        mesh = getattr(self.opts, "pose_mesh", None) or None
        # No mesh and building not refused: a click starts the model-free path
        # (collect reference views -> reconstruct -> pose), which is what the
        # upstream tool does with --pose and no --mesh. --pose-no-build opts
        # back down to tracking only.
        build = not mesh and not bool(getattr(self.opts, "pose_no_build", False))
        self.session = PoseSession(
            src_dir=getattr(self.opts, "pose_src", None) or None,
            model=str(getattr(self.opts, "pose_model", "tiny") or "tiny"),
            ckpt=getattr(self.opts, "pose_ckpt", None) or None,
            mesh=str(mesh) if mesh else None,
            build=build,
            ref_dir=getattr(self.opts, "pose_ref_dir", None) or None,
            max_arc=float(getattr(self.opts, "pose_max_arc", 75.0)),
            max_views=int(getattr(self.opts, "pose_max_views", 20)),
            object_size_mm=getattr(self.opts, "pose_object_size", None),
            # Both were hard-coded until 2026-08-21. Underwater the mask is
            # lost far more often than on a bench, and a 15 s reacquire budget
            # spent mid-orbit ends the whole collection in "click again" —
            # which is not a knob anyone could reach.
            lost_grace=float(getattr(self.opts, "pose_lost_grace", 3.0)),
            lost_timeout=float(getattr(self.opts, "pose_lost_timeout", 15.0)))
        # Load eagerly. The model takes seconds to tens of seconds, and doing it
        # on the first click would make the click feel broken; doing it now
        # overlaps with the pilot's setup instead. An idle tracker costs VRAM
        # and nothing else.
        self.session.start_async(on_log=self._say)
        how = ("click an object and it will register against the mesh"
               if mesh else
               ("click an object, then orbit it slowly — the station collects "
                "reference views, reconstructs a mesh (~2 min) and then "
                "estimates pose" if build else
                "tracking only (--pose-no-build): a mask, no 6-DoF pose"))
        self.bus.log.emit("info", f"pose: tracker loading. Use TRACK on the "
                                  f"C3 RGB panel, then {how}")

    def _say(self, level: str, text: str) -> None:
        """The on_log callback handed to the session.

        Named ``_say`` and NOT ``_log``, and that is a scar: ``self._log`` is
        the SensorLog attribute, and a method of the same name is shadowed by
        it the moment ``__init__`` runs. Every ``on_log=self._log`` then passes
        None (logs silently vanish) — or, while a recording is running, the
        SensorLog object itself, which is not callable, so the first log call
        inside the capture finalizer raised TypeError on every tick and the
        pipeline could never leave COLLECTING VIEWS. The shadow test in
        tests/test_offline.py now bans the pattern package-wide.
        """
        self.bus.log.emit(level, text)

    # -------------------------------------------------------------- commands
    @Slot(bool)
    def set_enabled(self, on: bool) -> None:
        """TRACK on/off. Off is also the pilot's ABORT.

        Turning it off drops the object and, if a reconstruction is under way,
        kills the child process — a two-minute GPU job that the button that
        started it cannot stop is not something to put on a station that is
        flying a vehicle at the same time.
        """
        self.enabled = bool(on)
        # Tell the producer, not just ourselves: the video worker only copies
        # frames into the mailbox while somebody wants them.
        self.mailbox.set_wanted(self.enabled)
        if not on and self.session is not None:
            self.session.reset(on_log=self._say)

    @Slot(float, float)
    def set_click(self, x: float, y: float) -> None:
        """A prompt in SOURCE pixels. The canvas already converted."""
        if self.session is not None and self.enabled:
            self.session.click(x, y)
            if self._log is not None:
                self._log.write("PROMPT", {"x": float(x), "y": float(y)})

    @Slot()
    def reset(self) -> None:
        if self.session is not None:
            self.session.reset(on_log=self._say)

    @Slot(bool, str)
    def set_sensor_log(self, on: bool, stem: str) -> None:
        """Rides the C3 Depth recording, like the other two sensor logs.

        A TimerWorker, so this queued slot really is delivered — unlike the one
        C3VideoWorker used to declare. See request_sensor_log there.
        """
        if on and self._log is None:
            self._log = SensorLog(Path(f"{stem}_pose.jsonl"), "pose")
            self._camera_written = False
            self._mesh_record_written = False   # each file names its mesh
            self.bus.log.emit("info", f"pose log -> {self._log.path}")
            self._write_camera_record()
            self._open_csv(Path(f"{stem}_pose.csv"))
        elif not on and self._log is not None:
            meta = self._log.close()
            self._log = None
            self._close_csv()
            self.bus.log.emit(
                "info", f"pose log closed: {meta['records']} records "
                        f"({meta['hz_measured'].get('TRACK', 0):.0f} Hz)")

    # A flat file next to the jsonl, one row per estimated pose, because that
    # is what a plotting script or pandas wants to eat. Same numbers, same
    # capture-time stamps as the POSE rows — a second FORMAT, never a second
    # source (health.py's rule about drawing one number in two places).
    # `pose_hz` became `pose_hz_measured` on 2026-08-23 and `solve_ms` arrived
    # beside it. THE RENAME IS THE POINT: files written before that date have a
    # column called pose_hz holding 1/mean-compute-time, and a reader that looks
    # up "pose_hz" in a new file must fail loudly rather than quietly read a
    # different quantity under a familiar name.
    _CSV_HEADER = ("t,frame_seq,x_m,y_m,z_m,distance_m,"
                   "r00,r01,r02,r10,r11,r12,r20,r21,r22,"
                   "pose_hz_measured,solve_ms,n_register,state\n")

    def _open_csv(self, path: Path) -> None:
        try:
            self._csv = open(path, "w", encoding="utf-8")
            self._csv.write(self._CSV_HEADER)
            self.bus.log.emit("info", f"pose csv -> {path}")
        except OSError as e:
            self._csv = None
            self.bus.log.emit("error", f"pose csv: {e}")

    def _close_csv(self) -> None:
        if self._csv is not None:
            try:
                self._csv.close()
            except OSError:
                pass
            self._csv = None

    def _write_csv(self, track: PoseTrack, t_cap: float) -> None:
        if self._csv is None or track.T_cam_obj is None:
            return
        T = track.T_cam_obj
        d = (T[3] ** 2 + T[7] ** 2 + T[11] ** 2) ** 0.5
        row = [f"{t_cap:.4f}", str(track.frame_seq),
               f"{T[3]:.5f}", f"{T[7]:.5f}", f"{T[11]:.5f}", f"{d:.5f}",
               # rotation ROWS of T_cam_obj, row-major like the jsonl
               f"{T[0]:.6f}", f"{T[1]:.6f}", f"{T[2]:.6f}",
               f"{T[4]:.6f}", f"{T[5]:.6f}", f"{T[6]:.6f}",
               f"{T[8]:.6f}", f"{T[9]:.6f}", f"{T[10]:.6f}",
               f"{track.pose_hz:.2f}", f"{track.pose_solve_ms:.1f}",
               str(track.n_register), track.state]
        self._csv.write(",".join(row) + "\n")

    def _write_camera_record(self) -> None:
        """The camera model, ONCE, with its provenance spelled out.

        A pose is only as metric as the intrinsics behind it, and these are an
        in-air factory calibration. Writing that into the file rather than
        implying it is the difference between a dataset and a trap six months
        from now (CLAUDE.md on measurement provenance).
        """
        # Deferred until the intrinsics exist. Recording can be started before
        # the first frame arrives, and a CAMERA record without a camera model in
        # it is worse than none — it looks like the calibration was unavailable
        # rather than merely not read yet.
        if self._log is None or self._camera_written:
            return
        i = self._log_intr
        if i is None:
            return
        self._camera_written = True
        fields = {
            "source": "C3 factory EEPROM via C3Source.intrinsics",
            # UNDERWATER, not in-air — vendor-confirmed and measured
            # (calib/fov_audit.py: HFOV 63.7 deg vs 67.2 Snell-underwater vs
            # 95 in-air spec; KNOWN_ISSUES.md 2026-08-04). An earlier version
            # of this record said "in_air": True, copied from a stale comment
            # the same audit had already flagged as one of five wrong hardcoded
            # phrases — exactly the "our own assumption printed into an output
            # file comes back looking like the device said it" failure the
            # measurement rule exists for.
            "medium": "water", "rectified": False,
            "depth_align": "color", "depth_units": "mm",
            "note": "UNDERWATER factory calibration (vendor-confirmed; "
                    "measured HFOV 63.7 deg matches the Snell-underwater "
                    "prediction, not the in-air spec — calib/fov_audit.py, "
                    "KNOWN_ISSUES.md 2026-08-04). Distances ARE metric in "
                    "water. IN-AIR captures read ~1.33x LONG with this "
                    "calibration, so bench-test depths must be divided by "
                    "~1.33 before comparing against a tape measure.",
            # Spell the convention out. Implying it is how a dataset becomes
            # wrong six months later, and this one has two easy ways to be read
            # backwards (which direction the transform maps, and which way Y
            # points).
            "frame": "T_cam_obj maps OBJECT -> CAMERA. OpenCV optical axes: "
                     "+X right, +Y DOWN, +Z forward. Metres, row-major. NO "
                     "camera->body transform is applied; the vehicle's own "
                     "pose is not in this file.",
            "mesh": str(getattr(self.opts, "pose_mesh", "") or ""),
            # The correction the depth in THIS file's poses was multiplied by
            # before anything consumed it (C3VideoWorker, --depth-scale).
            # 1.0 = raw sensor millimetres.
            "depth_scale_applied": float(
                getattr(self.opts, "depth_scale", DEPTH_SCALE_DEFAULT)
                or DEPTH_SCALE_DEFAULT),
        }
        fields.update({"fx": i.fx, "fy": i.fy, "cx": i.cx, "cy": i.cy,
                       "width": i.width, "height": i.height,
                       "distortion": list(i.distortion)})
        self._log.write("CAMERA", fields)

    # ------------------------------------------------------------------ tick
    def tick(self) -> None:
        s = self.session
        if s is None:
            return
        item = self.mailbox.take()
        if item is not None and item.get("K") is not None:
            self._log_intr = item["K"]
            if not self._logged_intrinsics:
                self._logged_intrinsics = True
                self._log_intrinsics(item["K"])
            self._write_camera_record()   # no-op once written, or if not logging
        if item is not None and self.enabled and s.ready:
            # The session owns the phase machine from here: with a mesh it
            # brings FoundationPose up on the first frame that carries K (which
            # is why this cannot happen in setup()), and without one it runs
            # capture -> reconstruct -> pose off these same frames.
            s.submit(item["color"], item.get("depth"),
                     item.get("t_capture", 0.0), item.get("seq", 0),
                     K=self._K_matrix(item.get("K")), on_log=self._say)
            self._log_mesh_once(s)
        # WHICH FRAME THE POSE BELONGS TO. `_step_pipeline` runs inline inside
        # `submit`, so the pose published below was estimated from the frame
        # just submitted; on the ticks between camera frames (the 10 Hz
        # publish gate is independent of frame arrival) it is still that
        # frame's. One definition, used by the log rows, the CSV and — the
        # reason it moved onto the dataclass — the object/tag frame pairing in
        # MpcWorker.on_pose, which needs the SAME float C3VideoWorker._tap_pose
        # put into the nav mailbox.
        if item is not None and item.get("t_capture"):
            self._last_t_cap = float(item["t_capture"])
        t = now()
        if t - self._last_pub < 0.1:          # publish at 10 Hz, not 50
            return
        self._last_pub = t
        track = self._snapshot(s)
        track.t_capture = float(self._last_t_cap)
        self.bus.pose.emit(track)
        self._announce(track)
        self._log_track(track, item)

    # Stages worth a line in the LOG panel, and how loud. The overlay is where
    # the pilot watches this, but the overlay has no history — after the fact,
    # "did it ever start reconstructing?" can only be answered from the log.
    _STAGE_WORDS = {
        "capturing": ("info", "collecting reference views"),
        "building": ("info", "reconstructing the mesh"),
        "pose_loading": ("info", "loading FoundationPose"),
        "registering": ("info", "registering the pose"),
        "tracking": ("info", "tracking"),
        "lost": ("warn", "lost the object"),
        "failed": ("error", "stopped"),
        "fault": ("error", "fault"),
    }

    @staticmethod
    def _trim_note(word: str, note: str) -> str:
        """Drop the part of the note that merely repeats the stage word.

        The tracker's note and this table's word are written independently and
        say the same thing often enough that the log filled with lines like
        "pose: reconstructing the mesh — reconstructing the mesh (about 2
        minutes)" and "pose: lost the object — lost — reacquiring (0 s)"
        (mission_log.txt, 2026-08-23). Strip the overlap rather than the note:
        the tail ("about 2 minutes", "reacquiring") is the half with the
        information in it.
        """
        n = (note or "").strip()
        if not n:
            return ""
        for lead in (word, word.split()[0] if word else ""):
            if lead and n.lower().startswith(lead.lower()):
                n = n[len(lead):].lstrip(" —-:").strip()
        return n

    def _announce(self, track: PoseTrack) -> None:
        if track.state == self._last_state:
            return
        self._last_state = track.state
        level, word = self._STAGE_WORDS.get(track.state, (None, ""))
        if level is not None:
            trimmed = self._trim_note(word, track.note)
            note = f" — {trimmed}" if trimmed else ""
            self.bus.log.emit(level, f"pose: {word}{note}")

    def _log_mesh_once(self, s) -> None:
        """Record WHICH mesh the poses in this file were measured against.

        A pose is a statement about a specific 3-D model, and an on-site
        reconstruction is a different model every time. Without the path, the
        hash and the sanity-check lines in the file, a POSE row six months from
        now is a number with no object behind it (CLAUDE.md on provenance).
        """
        mesh = getattr(s, "mesh", None)
        if not mesh:
            return
        # TWO latches, never one. The console announcement happens once per
        # mesh; the file record happens once per LOG FILE. A single latch
        # consumed by the console path meant a mesh that existed before the
        # recording started was announced and then never written — the file
        # whose whole point is provenance was the one place without it.
        need_console = not self._mesh_logged
        need_record = self._log is not None and not self._mesh_record_written
        if not (need_console or need_record):
            return
        fields = {
            "path": str(mesh),
            "reconstructed_on_site": bool(not getattr(self.opts, "pose_mesh",
                                                      None)),
            "ref_views_dir": s.ref_dir,
            "checks": list(s.mesh_check_lines),
            "units": "metres",
        }
        try:
            fields["sha1"] = _sha1_file(Path(str(mesh)))
        except OSError as e:
            fields["sha1"] = f"unreadable: {e}"
        if need_console:
            self._mesh_logged = True
            self.bus.log.emit("info", f"pose: mesh {Path(str(mesh)).name} "
                                      f"({fields['sha1'][:12]})")
        if need_record:
            self._mesh_record_written = True
            self._log.write("MESH", fields)

    def _log_track(self, track: PoseTrack, item) -> None:
        """One row per published result, plus a row on every state change.

        Stamped with the frame's CAPTURE time, not now — that is what lines the
        rows up against the video they belong to (sensorlog.py on clocks).
        """
        if self._log is None:
            return
        # A separate cursor from _announce's: the log only exists while a
        # recording is running, and sharing one would make the LOG panel go
        # quiet the moment recording was off (or repeat itself the moment it
        # was on).
        if track.state != self._last_event_state:
            self._log.write("EVENT", {"from": self._last_event_state,
                                      "to": track.state, "note": track.note})
            self._last_event_state = track.state
        if track.state not in ("tracking", "lost", "idle", "live"):
            return
        # The frame's CAPTURE time, resolved ONCE in tick() and carried on the
        # snapshot — not 0.0 (which sorted every such row to the epoch) and not
        # now() (the clock rule sensorlog.py exists to stop). It used to be
        # re-derived here from `item`, which meant two definitions of the same
        # stamp; the pairing in object_nav needs exactly one.
        t_cap = float(track.t_capture or 0.0)
        if track.T_cam_obj is not None:
            T = track.T_cam_obj
            self._log.write("POSE", {
                "state": track.state, "frame_seq": track.frame_seq,
                # 16 floats, ROW-MAJOR, object -> camera, metres, OpenCV axes.
                # The manifest says so in words too; a convention that is only
                # implied is a convention that will be read wrong.
                "T_cam_obj": [round(v, 6) for v in T],
                "pos_m": [round(T[3], 5), round(T[7], 5), round(T[11], 5)],
                "distance_m": round(float(
                    (T[3] ** 2 + T[7] ** 2 + T[11] ** 2) ** 0.5), 5),
                "pose_hz_measured": round(track.pose_hz, 2),
                "solve_ms": round(track.pose_solve_ms, 1),
                "n_register": track.n_register,
                "mask_px": track.mask_px,
            }, t=t_cap or None)
            self._write_csv(track, t_cap)
        self._log.write("TRACK", {
            "state": track.state, "frame_seq": track.frame_seq,
            "mask_px": track.mask_px, "score": track.score,
            # Renamed with pose_hz and for the same reason: this used to be
            # SAM2's 1/compute-time and is now the measured mask rate.
            "sam_hz_measured": round(track.sam_hz, 2),
            "sam_solve_ms": round(track.sam_solve_ms, 1),
            "skew_ms": round(float(item.get("skew_ms") or 0.0), 2) if item else None,
            "src_w": track.src_w, "src_h": track.src_h,
            # The outline itself, so a recording can be re-analysed offline
            # without re-running SAM2. ~1 kB, and it is the actual result.
            "contours": [[[round(x, 1), round(y, 1)] for x, y in poly]
                         for poly in track.contours],
        }, t=t_cap or None)

    def _snapshot(self, s) -> PoseTrack:
        if s.error:
            return PoseTrack(state="fault", note=s.error, conn=Conn.FAULT,
                             stamp=now())
        if s.loading:
            return PoseTrack(state="loading", load_s=s.load_seconds,
                             note="loading SAM2", conn=Conn.CONNECTING,
                             stamp=now())
        if not s.ready:
            return PoseTrack(state="off", conn=Conn.OFFLINE, stamp=now())
        if not self.enabled:
            return PoseTrack(state="off", note="TRACK is off",
                             conn=Conn.OFFLINE, stamp=now())
        raw = s.poll() or {}
        state = raw.get("state", "idle")
        conn = {"tracking": Conn.ONLINE, "lost": Conn.DEGRADED,
                "live": Conn.ONLINE,
                # Work in progress, not a healthy steady state: CONNECTING is
                # what the rest of the station uses for "on its way, do not
                # read it yet". The colour is the fastest signal on the panel,
                # so these must NOT be the same green as a working pose.
                "capturing": Conn.CONNECTING,
                "building": Conn.CONNECTING,
                "pose_loading": Conn.CONNECTING,
                "registering": Conn.CONNECTING,
                # Gave up. Sticks until the pilot retries, and says why.
                "failed": Conn.FAULT,
                "fault": Conn.FAULT}.get(state, Conn.ONLINE)
        if not self._mesh_desc_logged and s.mesh_description:
            self._mesh_desc_logged = True
            self.bus.log.emit("info", f"pose: {s.mesh_description}")
        return PoseTrack(
            state=state, contours=raw.get("contours", ()),
            T_cam_obj=raw.get("T_cam_obj"),
            axes_px=raw.get("axes_px", ()), box_px=raw.get("box_px", ()),
            score=raw.get("score"), mask_px=int(raw.get("mask_px", 0)),
            sam_hz=float(raw.get("sam_hz", 0.0)),
            sam_solve_ms=float(raw.get("sam_solve_ms", 0.0)),
            pose_hz=float(raw.get("pose_hz", 0.0)),
            pose_solve_ms=float(raw.get("pose_solve_ms", 0.0)),
            n_register=int(raw.get("n_register", 0)),
            frame_seq=int(raw.get("frame_seq", 0)),
            src_w=int(raw.get("src_w", 0)), src_h=int(raw.get("src_h", 0)),
            n_views=int(raw.get("n_views", 0)),
            arc_deg=float(raw.get("arc_deg", 0.0)),
            max_arc=float(raw.get("max_arc", 0.0)),
            max_views=int(raw.get("max_views", 0)),
            build_s=float(raw.get("build_s", 0.0)),
            distance_m=float(raw.get("distance_m", 0.0)),
            # None means "not enough pixels to say" and the panel must show
            # that as "--", never as 0.0 (which reads as perfect).
            smear_ratio=(float(raw["smear_ratio"])
                         if raw.get("smear_ratio") is not None else None),
            smear_max=float(raw.get("smear_max", 0.0)),
            fp_load_s=float(raw.get("fp_load_s", 0.0)),
            pose_expected=bool(raw.get("pose_expected", False)),
            note=raw.get("message", ""), conn=conn, stamp=now())

    def _log_intrinsics(self, intr) -> None:
        """Say the camera model out loud, once, with its provenance.

        The C3's EEPROM calibration is an UNDERWATER one — vendor-confirmed
        and measured (calib/fov_audit.py; KNOWN_ISSUES.md 2026-08-04). That
        cuts the way an operator on a bench does not expect: in-water
        distances are metric, and it is the DESK test that reads long. An
        earlier version of this message said the opposite, quoting a stale
        comment the audit had already flagged — the operator would have
        distrusted exactly the numbers that were right.
        """
        try:
            self.bus.log.emit(
                "info",
                f"pose: C3 colour intrinsics fx={intr.fx:.1f} fy={intr.fy:.1f} "
                f"cx={intr.cx:.1f} cy={intr.cy:.1f} @{intr.width}x{intr.height} "
                f"— UNDERWATER factory calibration (vendor-confirmed, "
                f"KNOWN_ISSUES 2026-08-04): in-water distances are metric; "
                f"IN-AIR bench tests read ~1.33x long")
        except Exception:                                        # noqa: BLE001
            pass

    def teardown(self) -> None:
        if self._log is not None:
            self._log.close()             # a killed station still gets a file
            self._log = None
        self._close_csv()
        if self.session is not None:
            self.session.close()          # joins the tracker thread; mandatory
            self.session = None


# =============================================================================
# backend
# =============================================================================
class HardwareBackend(Backend):
    name = "hw"
    simulated = False

    def __init__(self, bus, mailboxes, opts):
        super().__init__(bus, mailboxes, opts)
        self.video = C3VideoWorker(bus, mailboxes, opts)
        self.vehicle = VehicleWorker(bus, opts)
        allow = bool(getattr(opts, "allow_command", False))
        self.sink = (MavlinkCommandSink(bus, opts) if allow
                     else NullCommandSink(bus, opts))
        # The vehicle worker never transmits (source_system 200, a passive
        # observer), so the parameter readback that tells the Leak row whether
        # the detector is even armed has to come from the sink. Same injection
        # pattern as mpc.sink_status_fn below.
        if hasattr(self.sink, "leak_config"):
            self.vehicle.leak_cfg_fn = self.sink.leak_config
        self.workers = [self.video, self.vehicle, self.sink]
        # Object tracking: opt-in, and when it is off nothing is constructed,
        # nothing is imported, and C3VideoWorker does not even copy a frame.
        self.pose = None
        if bool(getattr(opts, "pose", False)):
            from ..bus import RgbdMailbox

            mb = RgbdMailbox()
            self.video.pose_mb = mb
            self.pose = PoseWorker(bus, mb, opts)
            self.workers.append(self.pose)

        self.rovcam = None
        if getattr(opts, "panel2", "rov") == "rov":
            want = getattr(opts, "rov_cam_backend", "auto")
            use_gst = (want == "gstreamer"
                       or (want == "auto" and gstreamer_available()))
            self.rovcam = (GstRovCamWorker(bus, mailboxes, opts) if use_gst
                           else RovCamWorker(bus, mailboxes, opts))
            self.workers.insert(1, self.rovcam)

        # Closed-loop MPC: opt-in like --pose; off = nothing constructed and
        # no video worker copies a frame for it. Both new workers are
        # TimerWorkers, so every connection below is a queued slot that IS
        # delivered (backends/base.py rule). Built AFTER the rovcam worker so
        # its feed can be tapped too: the C3 colour feed localizes (factory
        # intrinsics ride each frame), the ROV RGB feed is detection-overlay
        # only (no calibration exists for it).
        self.tagnav = None
        self.mpc = None
        if bool(getattr(opts, "mpc", False)):
            from ..bus import RgbdMailbox
            from ..control.workers import MpcWorker, TagNavWorker

            nav_mb = RgbdMailbox()
            self.video.nav_mb = nav_mb
            nav_sources = {"main": nav_mb}
            if self.rovcam is not None:
                rov_mb = RgbdMailbox()
                self.rovcam.nav_mb = rov_mb
                nav_sources["second"] = rov_mb
            self.tagnav = TagNavWorker(bus, nav_sources, opts)
            self.mpc = MpcWorker(bus, opts)
            self.workers.append(self.tagnav)
            self.workers.append(self.mpc)

        bus.cmd_pilot.connect(self.sink.set_pilot)
        bus.cmd_gripper.connect(self.sink.set_gripper)
        bus.cmd_lights.connect(self.sink.set_lights)
        bus.cmd_enable.connect(self.sink.set_enabled)
        bus.cmd_estop.connect(self.sink.estop)
        bus.cmd_arm.connect(self.sink.set_arm)
        bus.cmd_gripper_drive.connect(self.sink.set_gripper_drive)
        bus.cmd_lights_step.connect(self.sink.set_lights_step)
        bus.cmd_tilt.connect(self.sink.set_tilt)
        bus.cmd_tilt_center.connect(self.sink.tilt_center)
        bus.cmd_buttons.connect(self.sink.set_js_buttons)
        bus.cmd_mode.connect(self.sink.set_mode)
        bus.aux_servos.connect(self.sink.note_aux_servos)
        # The sensor log that rides along with a C3 Depth recording. Both
        # sources get the same request and each writes its own file, because
        # they are different devices on different clocks — one merged file would
        # have to pick a clock and would be lying about the other.
        # VehicleWorker is a TimerWorker: it has an event loop, so a queued slot
        # is delivered between ticks and AutoConnection is correct.
        bus.cmd_log_sensors.connect(self.vehicle.set_sensor_log)
        # C3VideoWorker is a LoopWorker: it has NO event loop, so an
        # AutoConnection would resolve to Queued and never be delivered. The
        # receiver is a plain thread-safe method and the connection must be
        # DIRECT — it runs on the GUI thread and only puts to a queue.
        bus.cmd_log_sensors.connect(self.video.request_sensor_log,
                                    Qt.ConnectionType.DirectConnection)
        # ...and the same two sinks again for a CONTROLLER run's raw logs
        # (MpcWorker emits this one when its CSV opens). Same Direct/Auto
        # asymmetry, for the same reason, and it is an asymmetry of the
        # RECEIVERS — do not "tidy" one of them to match the other.
        bus.cmd_log_raw_sensors.connect(self.vehicle.set_sensor_log)
        bus.cmd_log_raw_sensors.connect(self.video.request_sensor_log,
                                        Qt.ConnectionType.DirectConnection)
        if self.pose is not None:
            # PoseWorker IS a TimerWorker, so these queued slots are delivered.
            bus.cmd_pose_enable.connect(self.pose.set_enabled)
            bus.cmd_pose_click.connect(self.pose.set_click)
            bus.cmd_pose_reset.connect(self.pose.reset)
            bus.cmd_log_sensors.connect(self.pose.set_sensor_log)
        if self.mpc is not None:
            # Sensor fan-in (worker -> bus -> worker: each hop is a queued
            # snapshot, so the MPC thread never touches another thread's data).
            bus.nav_fix.connect(self.mpc.on_nav_fix)
            # The object tracker's second consumer. TimerWorker -> TimerWorker,
            # so this queued slot IS delivered (backends/base.py rule). Wired
            # unconditionally: with no --pose nothing is emitted, and with no
            # --pose MpcWorker builds no anchor, so the slot is a no-op twice
            # over rather than a hidden dependency between two opt-in flags.
            bus.pose.connect(self.mpc.on_pose)
            bus.vehicle_imu.connect(self.mpc.on_vehicle_imu)
            # The C3's BNO086, in batches. Emitter is a LoopWorker and the
            # receiver a TimerWorker, so AutoConnection resolves to Queued and
            # IS delivered — the mirror image of the cmd_log_sensors case
            # above, where the LoopWorker was the receiver.
            bus.camera_imu.connect(self.mpc.on_camera_imu)
            bus.telemetry.connect(self.mpc.on_telemetry)
            bus.thrusters.connect(self.mpc.on_thrusters)
            bus.vehicle_imu.connect(self.tagnav.set_imu_hint)
            # The sink's own health gates engage: computing a mission that the
            # sink cannot deliver (no peer / SYSID mismatch) helps nobody.
            # status() only reads plain fields — safe cross-thread.
            self.mpc.sink_status_fn = self.sink.status
            # Commands + the safety fan-out: E-STOP and COMMAND ENABLE off
            # must reach the MPC as well as the sink.
            bus.cmd_mpc_engage.connect(self.mpc.set_engaged)
            bus.cmd_mpc_traj.connect(self.mpc.set_traj)
            bus.cmd_mpc_start.connect(self.mpc.start_mission)
            bus.cmd_mpc_mode.connect(self.mpc.set_mode)
            bus.cmd_mpc_scenario.connect(self.mpc.set_scenario)
            bus.cmd_mpc_dump_meta.connect(self.mpc.dump_run_meta)
            bus.cmd_enable.connect(self.mpc.on_enable)
            bus.cmd_estop.connect(self.mpc.estop)
            bus.cmd_log_sensors.connect(self.mpc.set_sensor_log)
            bus.cmd_tag_enable.connect(self.tagnav.set_source_enabled)

    def describe(self) -> str:
        kind = "MAVLink command sink" if isinstance(self.sink, MavlinkCommandSink) \
            else "receive-only (no command sink)"
        return f"hw — C3 @ {getattr(self.opts, 'ip', '?')} + ArduSub; {kind}"
