#!/usr/bin/env python3
"""
slam.py — live ORB-SLAM3 as a CHILD PROCESS of the station (``--slam``).

    C3VideoWorker ──_tap_slam──▶ SlamPairQueue ──▶ SlamNavWorker
      writer thread:  rectify ─▶ CLAHE ─▶ OAKX 'FRAM' ─▶ child stdin
      child:          TrackStereo ─▶ 'TWC ' ─▶ child stdout
      reader thread:  ─▶ pose queue ─▶ tick() ─▶ bus.nav_fix + bus.vehicle_imu
    bus.policy_plan_viz ──▶ tick() ──▶ OAKX 'KNOT' ──▶ child ──▶ Pangolin overlay

WHY A CHILD PROCESS AND NOT A LIBRARY CALL. ORB-SLAM3 has no Python binding
here, and the one that would have to be written owns a Pangolin GL context and
three long-lived threads. The driver at
``"data collection"/UMI_Underwater/slam/oakd_live_slam`` already exists, already
speaks a byte protocol on stdin, and already opens the map window; making it a
child costs one pipe pair and keeps every environment problem on the other side
of it (the child is a native binary — no depthai, no torch, no numpy, so
``rovgui-pose`` and the driver never have to agree about anything).

WHY THE STATION OWNS THE CAMERA AND THE CHILD DOES NOT. ``oakd_stream.py``, the
driver's usual producer, opens the device itself with a bare ``dai.Device()``
— on this network that grabs the C3 — and it is written against the depthai 3.x
API while this station runs 2.32. Both problems disappear by feeding the child
the pair the station has ALREADY pulled for FoundationStereo: one camera owner,
one pipeline, and SLAM costs ZERO extra PoE bandwidth because it rides frames
that are on the wire regardless.

CLAHE IS NOT OPTIONAL ON THIS BRANCH.
``Tracking::StereoInitialization`` refuses to create a map until a single frame
yields more than 500 ORB keypoints, and it says nothing at all when it does not
— the run simply sits at "KFs: 0, MPs: 0" forever. Measured on this machine
with ``feed_episode.py --check`` on recorded episode 0: with CLAHE (clip 3.0,
8x8) ORB keypoints median 1243 and 100 % of sampled frames clear the gate;
with ``--no-clahe`` median 384 and 26 % clear it [측정: 2026-09-03, the two
``feed_episode.py --episode 0 --check`` runs recorded in
.claude/journal/consults.md]. The training SLAM that produced the demonstration
pose labels also ran with it (``demonstration_processing/run_slam.py:83``,
``feed_episode.py:143-147``, ``"data collection"/slam/0/slam.log:4-7``), so
applying it here matches the instrument the labels came from as well as making
the room trackable. It is applied to the SLAM copy ONLY — FoundationStereo's
pair is untouched, because the policy's depth was not trained on equalised
images.

THE CLOCK. One clock, no conversions: ``state.now()`` (``time.monotonic()``).
``_tap_slam`` stamps each pair ``t_capture = now() - left.age_ms()/1000`` — the
SAME expression and the SAME value ``_tap_fstereo`` gives FoundationStereo
(``hardware.py:699-730``) — that float rides out in the FRAM record, ORB-SLAM3
takes it as its timestamp (it needs only a monotone increasing double), and the
child ECHOES it in the pose record. So the NavFix for exposure N and the policy's
depth frame for exposure N carry the same ``t_capture`` by construction, and
``PolicyWorker``'s ``skip_fix_lag`` (backends/policy.py, tolerance 0.5*obs_dt)
is structurally unable to fire on a clock mismatch.

WHAT THIS MODULE REFUSES TO HIDE. ORB-SLAM3 fails silently in two ways that both
look like a healthy 15 Hz pose stream, and both would read as "the diffusion
policy stopped making sense":
  * TRACKING LOSS re-initialises into a BRAND NEW map with a new origin
    (``Tracking.cc`` ResetActiveMap / CreateMapInAtlas), and
    ``GetTrackingState()`` returns OK again within one frame.
  * A LOOP CLOSURE or global BA shifts the whole world under the poses already
    consumed (``System::MapChanged()``).
``control/slam_frames.TrackMonitor`` classifies both; this worker turns either
into a LOUD ``bus.log`` line plus a deliberate not-ok ``NavFix`` beat, so the
station's own staleness machinery sees a dropout instead of a silent jump.

Conventions: metres, radians, seconds. Positions in the SLAM-derived NED map
frame (``slam_frames.P_NED_SLAM``); body FRD; camera optical for the extrinsic.
"""

from __future__ import annotations

import os
import queue
import shutil
import signal
import subprocess
import threading
from collections import deque
from pathlib import Path

import numpy as np

from ..qt import Signal, Slot, import_cv2
from ..state import Conn, NavFix, VehicleImu, now
from ..control.slam_frames import (EngageDatum, PoseRecord, SlamNav,
                                   TrackMonitor, TrackStatus)
from ..control import slam_wire as W
from .base import TimerWorker

#: CLAHE, the values the training pipeline used (feed_episode.py:87-88).
CLAHE_CLIP = 3.0
CLAHE_GRID = 8

#: How long without a pose record before the bridge calls itself FAULT. The
#: child tracks a 640x400 pair in ~12 ms [측정: "data collection"/slam/0/run.json
#: mean_track_ms 11.874], so 2 s is ~150 missed frames — long past "busy".
POSE_WATCHDOG_S = 2.0

#: Bounded FIFO, drop-OLDEST. SLAM wants CONTINUITY (consecutive frames are how
#: it triangulates), which is the opposite of the latest-wins rule the depth
#: mailbox uses — but an unbounded queue would let a stalled child grow the
#: station's heap without limit, so it is a FIFO with a lid rather than a slot.
SLAM_QUEUE_DEPTH = 4


class SlamPairQueue:
    """Rectified-pair hand-off from the camera thread to the writer thread.

    Deliberately NOT :class:`bus.StereoMailbox`: that one conflates to the
    newest frame, which is right for a depth network that only ever wants
    "now" and wrong for a tracker that loses its baseline when frames are
    skipped. ``wanted`` exists for the same reason it does there — with the
    feature off the producer must not pay for a copy.
    """

    def __init__(self, depth: int = SLAM_QUEUE_DEPTH):
        self._lock = threading.Lock()
        self._q: deque = deque(maxlen=int(depth))
        self._wanted = False
        self.put_n = 0
        self.dropped = 0

    def set_wanted(self, on: bool) -> None:
        with self._lock:
            self._wanted = bool(on)
            if not on:
                self._q.clear()

    def wanted(self) -> bool:
        with self._lock:
            return self._wanted

    def put(self, left, right, rig, t_capture: float, seq: int) -> None:
        with self._lock:
            if not self._wanted:
                return
            if len(self._q) == self._q.maxlen:
                self.dropped += 1
            self._q.append((left, right, rig, float(t_capture), int(seq)))
            self.put_n += 1

    def get(self):
        with self._lock:
            return self._q.popleft() if self._q else None

    def stats(self) -> tuple:
        with self._lock:
            return self.put_n, self.dropped, len(self._q)


class OrbSlamProc:
    """The child process, its writer thread and its reader thread.

    Pure Python + numpy + cv2 — no Qt — so the whole protocol path is testable
    against ``tests/data/fake_slam_driver.py`` without a camera, without
    ORB-SLAM3 and without an event loop.

    BACK-PRESSURE. The writer's ``write()`` is BLOCKING. That is deliberate:
    the queue in front of it is bounded and drops the oldest, so a child that
    falls behind loses frames at a defined place instead of growing a buffer
    somewhere undefined. The pipe is enlarged with F_SETPIPE_SZ where the
    kernel allows it, because a 512 kB frame does not fit a default 64 kB pipe
    and would otherwise make every write a multi-round-trip.
    """

    F_SETPIPE_SZ = 1031          # not exposed by the fcntl module

    def __init__(self, binary: Path, vocab: Path, settings, out_prefix,
                 width: int, height: int, viewer: bool = True, log=None):
        self.binary = Path(binary)
        self.vocab = Path(vocab)
        # settings/out_prefix are None until the caller knows the live rig and
        # the run folder; start() refuses without them rather than inventing
        # a path a later reader would have to guess the provenance of.
        self.settings = Path(settings) if settings is not None else None
        self.out_prefix = Path(out_prefix) if out_prefix is not None else None
        self.width = int(width)
        self.height = int(height)
        self.viewer = bool(viewer)
        self._log = log or (lambda level, msg: None)

        self.proc: subprocess.Popen | None = None
        self.poses: queue.Queue = queue.Queue(maxsize=256)
        self._frames: queue.Queue = queue.Queue(maxsize=SLAM_QUEUE_DEPTH)
        self._knots: queue.Queue = queue.Queue(maxsize=32)
        self._stop = threading.Event()
        self._writer: threading.Thread | None = None
        self._reader: threading.Thread | None = None
        self._stderr: threading.Thread | None = None
        self.n_frames_sent = 0
        self.n_poses = 0
        self.t_last_pose = 0.0
        self.stderr_tail: deque = deque(maxlen=40)

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if not self.binary.exists():
            raise FileNotFoundError(
                f"{self.binary} missing — build it with:\n"
                f"  cd '{self.binary.parent.parent}' && "
                f"ORB_SLAM3_ROOT=$HOME/ORB_SLAM3 cmake -S . -B build "
                f"-DCMAKE_BUILD_TYPE=Release && cmake --build build -j")
        if not self.vocab.exists():
            raise FileNotFoundError(f"{self.vocab} missing — "
                                    f"run slam/setup_orbslam3.sh")
        if self.settings is None or self.out_prefix is None:
            raise RuntimeError("OrbSlamProc.start before the run folder and "
                               "the settings were resolved from the live rig")
        if not self.settings.exists():
            raise FileNotFoundError(f"{self.settings} missing (generated)")
        if self.width < 1 or self.height < 1:
            raise RuntimeError(f"OrbSlamProc.start with image size "
                               f"{self.width}x{self.height}")
        self.out_prefix.parent.mkdir(parents=True, exist_ok=True)

        argv = [str(self.binary), str(self.vocab), str(self.settings),
                str(self.out_prefix)]
        if not self.viewer:
            argv.append("--no-viewer")

        env = dict(os.environ)
        # The viewer needs an X display, and a shell started outside the
        # desktop session inherits an empty DISPLAY — which aborts Pangolin
        # with "Failed to open X display" and reads exactly like a headless
        # box. run_live_slam.sh solves it by probing; do the same here.
        if self.viewer and not env.get("DISPLAY"):
            found = _find_display()
            if found:
                env["DISPLAY"] = found
                self._log("info", f"slam: DISPLAY was empty; using {found}")

        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0, env=env)
        self._enlarge_pipe()

        self.proc.stdin.write(W.encode_header(self.width, self.height))
        self.proc.stdin.flush()

        self._stop.clear()
        self._writer = threading.Thread(target=self._write_loop,
                                        name="slam-writer", daemon=True)
        self._reader = threading.Thread(target=self._read_loop,
                                        name="slam-reader", daemon=True)
        self._stderr = threading.Thread(target=self._stderr_loop,
                                        name="slam-stderr", daemon=True)
        for t in (self._writer, self._reader, self._stderr):
            t.start()

    def _enlarge_pipe(self) -> None:
        """A 640x400 pair is 512 kB; the default pipe is 64 kB."""
        try:
            import fcntl
            fcntl.fcntl(self.proc.stdin.fileno(), self.F_SETPIPE_SZ,
                        1 << 20)
        except (OSError, AttributeError, ImportError):
            pass                       # a smaller pipe is slower, not wrong

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self, timeout_s: float = 5.0) -> None:
        """BYE, then wait, then SIGINT, then kill.

        The driver saves its trajectories on the normal exit path and calls
        ``_exit`` afterwards (its own teardown races the Pangolin viewer), so a
        clean BYE is what puts ``*_frames.tum`` on disk. SIGINT is the driver's
        documented stop and reaches the same path; SIGKILL loses the files.
        """
        self._stop.set()
        p = self.proc
        if p is None:
            return
        try:
            if p.poll() is None and p.stdin is not None:
                try:
                    p.stdin.write(W.encode_bye())
                    p.stdin.flush()
                except (BrokenPipeError, OSError, ValueError):
                    pass
            try:
                p.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                p.send_signal(signal.SIGINT)
                try:
                    p.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    p.kill()
        finally:
            for stream in (p.stdin, p.stdout, p.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except (OSError, ValueError):
                    pass

    # --------------------------------------------------------------- submit
    def submit_frame(self, seq: int, t_capture: float, left, right) -> bool:
        """Queue one rectified GRAY8 pair. False when the queue is full."""
        try:
            self._frames.put_nowait((int(seq), float(t_capture), left, right))
            return True
        except queue.Full:
            return False

    def submit_knots(self, payload: bytes) -> bool:
        try:
            self._knots.put_nowait(payload)
            return True
        except queue.Full:
            return False

    # ---------------------------------------------------------------- threads
    def _write_loop(self) -> None:
        """One writer, so FRAM and KNOT records can never interleave.

        Knots first on every pass: they are ~400 B against a 512 kB frame and
        the whole point of the overlay is that it keeps up with the map.
        """
        stdin = self.proc.stdin
        while not self._stop.is_set():
            wrote = False
            try:
                while True:
                    stdin.write(self._knots.get_nowait())
                    wrote = True
            except queue.Empty:
                pass
            except (BrokenPipeError, OSError, ValueError):
                return
            try:
                seq, t_cap, left, right = self._frames.get(timeout=0.05)
            except queue.Empty:
                if wrote:
                    try:
                        stdin.flush()
                    except (BrokenPipeError, OSError, ValueError):
                        return
                continue
            try:
                stdin.write(W.encode_frame(seq, t_cap, left, right))
                stdin.flush()
                self.n_frames_sent += 1
            except (BrokenPipeError, OSError, ValueError):
                return

    def _read_loop(self) -> None:
        """Parse the child's stdout with the streaming decoder.

        The decoder is fed whatever ``read`` returns, which is NOT record
        aligned — that is exactly what ``slam_wire``'s partial-read handling is
        for, and why the parsing is not open-coded here.
        """
        dec = W.PoseDecoder()
        stdout = self.proc.stdout
        while not self._stop.is_set():
            try:
                chunk = stdout.read(4096)
            except (OSError, ValueError):
                return
            if not chunk:
                return                       # child exited
            try:
                for rec in dec.feed(chunk):
                    self.n_poses += 1
                    self.t_last_pose = now()
                    try:
                        self.poses.put_nowait(rec)
                    except queue.Full:
                        try:
                            self.poses.get_nowait()   # drop the OLDEST
                            self.poses.put_nowait(rec)
                        except (queue.Empty, queue.Full):
                            pass
            except W.WireFormatError as e:
                self._log("error", f"slam: wire desync — {e}")
                return

    def _stderr_loop(self) -> None:
        """The driver's whole human-readable side is on stderr, including the
        ">500 keypoints" explanation that is the only clue when a room will
        not initialise. Keep a tail so a failure can be reported with it."""
        stderr = self.proc.stderr
        while not self._stop.is_set():
            try:
                line = stderr.readline()
            except (OSError, ValueError):
                return
            if not line:
                return
            text = line.decode("utf-8", "replace").rstrip()
            if text:
                self.stderr_tail.append(text)


#: Where the driver, its settings generator and the vocabulary live. NOT in
#: this repo: the SLAM side is the UMI_Underwater tree that recorded the
#: demonstrations, and the settings generator there is the SAME function the
#: offline pipeline used to write the yaml for the 75 training episodes — so
#: the live run and the label run cannot describe the camera differently.
#: MIND THE SPACE in "data collection"; it is a real directory name.
SLAM_DIR = Path("/home/bdml/Desktop/data collection/UMI_Underwater/slam")
ORB_SLAM3_ROOT = Path(os.environ.get("ORB_SLAM3_ROOT",
                                     str(Path.home() / "ORB_SLAM3")))
DRIVER_BIN = SLAM_DIR / "build" / "oakd_live_slam"
VOCAB = ORB_SLAM3_ROOT / "Vocabulary" / "ORBvoc.txt"


def _load_write_yaml():
    """``orbslam_settings.write_yaml`` from :data:`SLAM_DIR`, by path.

    Loaded through importlib rather than sys.path because the directory name
    contains a space and is outside this repo: appending it to sys.path would
    also expose its other top-level modules (``oakd_stream``, which imports
    depthai 3.x) to every later import in this process.
    """
    import importlib.util

    path = SLAM_DIR / "orbslam_settings.py"
    spec = importlib.util.spec_from_file_location("_orbslam_settings", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"{path} not importable")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.write_yaml


def _status_code(status) -> int:
    """``PolicyPlanViz.status`` ("accept" | "clip" | "reject" | "late" |
    "skipped") -> the wire's u8.

    An unrecognised name maps to REJECT rather than to ACCEPT: a verdict this
    module does not understand must never be drawn in the colour that means
    "the controller flew this".
    """
    try:
        return int(W.STATUS_NAMES.index(str(status)))
    except ValueError:
        return int(W.KnotStatus.REJECT)


def _find_display() -> str:
    """The running X display, or "". ``run_live_slam.sh``'s probe, in Python."""
    xdpy = shutil.which("xdpyinfo")
    try:
        socks = sorted(Path("/tmp/.X11-unix").iterdir())
    except OSError:
        return ""
    for s in socks:
        if not s.name.startswith("X"):
            continue
        disp = ":" + s.name[1:]
        if xdpy is None:
            return disp
        try:
            ok = subprocess.run([xdpy, "-display", disp],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, timeout=3.0)
            if ok.returncode == 0:
                return disp
        except (OSError, subprocess.SubprocessError):
            continue
    return ""


class SlamNavWorker(TimerWorker):
    """The Qt side: pose records in -> NavFix/VehicleImu out, knots back.

    Ticks at 5 ms so a 15 Hz pose stream is never held for more than a tick
    (the same interval ``TagNavWorker`` uses for the same reason).

    IT IS THE FIX PRODUCER, AND IT MUST BE THE ONLY ONE. ``bus.nav_fix`` is
    last-writer-wins in ``MpcWorker``, so a synthetic ``--land-dry-run`` fix
    (``TagNavWorker._emit_land_fix``) racing a real SLAM fix would give the
    policy a proprio history that alternates between a moving pose and a
    stationary one — a motion cue that flips sign at 15 Hz. ``hardware.py``
    suppresses the synthetic one whenever this worker exists; the assertion is
    written down here because THIS is the class that breaks if it stops being
    true.
    """

    #: Emitted once when the child has produced its first tracked pose, so the
    #: window can say the map is live rather than leaving the operator to read
    #: an empty plot.
    tracking_started = Signal()

    def __init__(self, bus, pair_q: SlamPairQueue, proc: OrbSlamProc,
                 nav, opts, clahe: bool = True,
                 log_base=None):
        super().__init__("slamnav", interval_ms=5)
        self.bus = bus
        self.pair_q = pair_q
        self.proc = proc
        self.nav: SlamNav = nav
        self.opts = opts
        # A runstore.Tree / string, or a CALLABLE returning one
        # (MpcWorker._run_tree, since 2026-09-11: the LOW level — and with it
        # the run KIND — can change in the panel after launch, so the tree is
        # resolved at the ONE use in _start_child, when the child actually
        # starts). None = the water root, kind "".
        from .. import runstore
        self.log_base = (log_base if callable(log_base)
                         else runstore.DEFAULT_BASE if log_base is None
                         else log_base)
        self.use_clahe = bool(clahe)
        self._clahe = None
        self._cv2 = None
        self._mon = TrackMonitor()
        self._session = 0
        self._datum: EngageDatum | None = None
        self._pending_viz: deque = deque(maxlen=4)
        self._n_fix = 0
        self._fix_marks: list = []
        self._started = False
        self._faulted = False
        self._t_start = 0.0
        self._last_seq = -1

    # -------------------------------------------------------------- lifecycle
    def setup(self) -> None:
        """Prepare, but do NOT start the child yet — see :meth:`_start_child`."""
        self._cv2 = import_cv2()
        if self.use_clahe and self._cv2 is not None:
            self._clahe = self._cv2.createCLAHE(
                clipLimit=CLAHE_CLIP,
                tileGridSize=(CLAHE_GRID, CLAHE_GRID))
        self._t_start = now()
        self.pair_q.set_wanted(True)

    def _start_child(self, rig) -> bool:
        """Generate the settings from THIS rig, then spawn the child.

        LAZY, on the first pair, for a reason that is not just tidiness: the
        settings file has to carry the intrinsics of the images ORB-SLAM3 will
        actually receive, and those are ``P1`` from the ``cv2.stereoRectify``
        that built the remap tables — which do not exist until the camera has
        opened and ``_build_rig`` has run. Writing the yaml from the same
        ``StereoRig`` object that rectifies the pixels is what makes it
        impossible for the two to disagree; a yaml written at construction time
        would have to guess, and a guessed baseline is a silent metric scale
        error in every pose that follows.

        It also keeps the 145 MB vocabulary load off the startup path: the
        child begins loading when the camera is already producing.
        """
        try:
            from .. import runstore

            base = (self.log_base() if callable(self.log_base)
                    else self.log_base)
            out_dir = Path(runstore.run_dir(base)) / "slam"
            out_dir.mkdir(parents=True, exist_ok=True)
            self.proc.settings = out_dir / "orbslam.yaml"
            self.proc.out_prefix = out_dir / "traj"
            write_yaml = _load_write_yaml()
            h, w = self._rect_size(rig)
            self.proc.width, self.proc.height = int(w), int(h)
            write_yaml(
                self.proc.settings, rig.P1, float(rig.baseline_mm) / 1000.0,
                int(w), int(h), float(getattr(self.opts, "depth_fps", 15.0)
                                      or 15.0),
                int(getattr(self.opts, "slam_features", 1200)),
                int(getattr(self.opts, "slam_ini_fast", 20)),
                int(getattr(self.opts, "slam_min_fast", 7)),
                provenance="rov_gui/backends/slam.py from the live StereoRig")
            self.proc.start()
        except Exception as e:                                   # noqa: BLE001
            self._faulted = True
            self.bus.log.emit("error", f"slam: cannot start ORB-SLAM3 — "
                                       f"{type(e).__name__}: {e}")
            self.failed.emit(f"slamnav: {type(e).__name__}: {e}")
            return False
        self.bus.log.emit(
            "info",
            f"slam: ORB-SLAM3 child up (pid {self.proc.proc.pid}), "
            f"{self.proc.width}x{self.proc.height}, fx {rig.P1[0, 0]:.2f} px, "
            f"baseline {rig.baseline_mm:.2f} mm, "
            f"CLAHE {'on' if self._clahe is not None else 'OFF'} -> "
            f"{self.proc.settings.parent}")
        if self._clahe is None:
            # Measured, and the single most likely reason a run never leaves
            # "KFs: 0" — so it is a warning, not a footnote.
            self.bus.log.emit(
                "warn",
                "slam: CLAHE is OFF. On recorded episode 0 that took the ORB "
                "keypoint median from 1243 to 384 and the fraction of frames "
                "clearing ORB-SLAM3's 500-keypoint gate from 100 % to 26 % "
                "[측정: 2026-09-03 feed_episode.py --check]. Expect it not to "
                "initialise.")
        return True

    @staticmethod
    def _rect_size(rig) -> tuple:
        """(h, w) of the RECTIFIED pair — the remap tables' own shape.

        ``rig.mono_size`` is (w, h) of the input; the maps are built at that
        size here, but reading the table is the only statement that cannot be
        wrong if that ever stops being true.
        """
        ml = getattr(rig, "map_left", None)
        if ml is not None and ml[0] is not None:
            return int(ml[0].shape[0]), int(ml[0].shape[1])
        w, h = rig.mono_size
        return int(h), int(w)

    def teardown(self) -> None:
        self.pair_q.set_wanted(False)
        try:
            self.proc.stop()
        except Exception as e:                                   # noqa: BLE001
            self.bus.log.emit("warn", f"slam: child stop — "
                                      f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------ Qt inbound
    @Slot(object)
    def on_mpc_status(self, st) -> None:
        """Track the engage datum so knots can be sent back into SLAM world.

        The datum is what ``MpcWorker`` fixed at ENGAGE; without it a
        datum-frame knot cannot be placed on the map at all, and with a STALE
        one it would be placed confidently in the wrong spot. So the overlay is
        cleared whenever the datum changes.
        """
        d = getattr(st, "datum", None)
        if d is None:
            if self._datum is not None:
                self._datum = None
                self.proc.submit_knots(W.encode_clear())
            return
        try:
            # (x0, y0, z0, yaw0) in MAP coordinates — NOT the internal dict
            # MpcWorker keeps; the status flattens it (workers.py:4658-4660)
            # and the plot reads the same four floats (trajectory.py:203).
            x0, y0, z0, yaw0 = (float(v) for v in d)
            p0 = np.array([x0, y0, z0], float)
        except (TypeError, ValueError):
            return
        old = self._datum
        if (old is None or float(np.max(np.abs(old.p0 - p0))) > 1e-9
                or abs(old.yaw0 - yaw0) > 1e-12):
            self._datum = EngageDatum(p0, yaw0)
            self.proc.submit_knots(W.encode_clear())

    @Slot(object)
    def on_policy_plan_viz(self, viz) -> None:
        """Queue one composed plan for the map window. Never blocks the
        emitter's thread: the transform is done on OUR tick."""
        self._pending_viz.append(viz)

    # ------------------------------------------------------------------ tick
    def tick(self) -> None:
        self._pump_frames()
        self._pump_poses()
        self._pump_knots()
        self._watchdog()

    def _pump_frames(self) -> None:
        """Rectify + equalise + submit. At most a few pairs per tick."""
        cv2 = self._cv2
        for _ in range(SLAM_QUEUE_DEPTH):
            item = self.pair_q.get()
            if item is None:
                return
            left, right, rig, t_cap, seq = item
            if self.proc.proc is None:
                if self._faulted or not self._start_child(rig):
                    return
            if seq == self._last_seq:
                continue
            self._last_seq = seq
            try:
                ml, mr = rig.map_left, rig.map_right
                if ml is not None and mr is not None and cv2 is not None:
                    left = cv2.remap(left, ml[0], ml[1], cv2.INTER_LINEAR)
                    right = cv2.remap(right, mr[0], mr[1], cv2.INTER_LINEAR)
                if self._clahe is not None:
                    left = self._clahe.apply(left)
                    right = self._clahe.apply(right)
            except Exception as e:                               # noqa: BLE001
                self.bus.log.emit("warn", f"slam: rectify/CLAHE — "
                                          f"{type(e).__name__}: {e}")
                continue
            self.proc.submit_frame(seq, t_cap, left, right)

    def _pump_poses(self) -> None:
        while True:
            try:
                rec: PoseRecord = self.proc.poses.get_nowait()
            except queue.Empty:
                return
            self._on_pose(rec)

    def _on_pose(self, rec: PoseRecord) -> None:
        ev = self._mon.update(rec)
        if ev.status is not TrackStatus.OK:
            self._emit_miss(rec, ev)
            return
        if ev.session != self._session:
            # A NEW MAP. The world moved; everything composed against the old
            # one is meaningless. Say so LOUDLY and beat one not-ok fix so the
            # station sees a dropout rather than a silent teleport.
            self._session = int(ev.session)
            self.bus.log.emit(
                "error",
                f"slam: ORB-SLAM3 re-initialised into a NEW MAP "
                f"(session {self._session}) — the world origin moved; every "
                f"pose before this is in a different frame. STOP and re-arm.")
            self.proc.submit_knots(W.encode_clear())
            self._emit_miss(rec, ev)
            return
        try:
            fix = self.nav.fix_from_pose(rec, session=self._session)
        except ValueError as e:
            self._emit_miss(rec, ev, note=str(e))
            return
        if not self._started:
            self._started = True
            self.bus.log.emit(
                "info", f"slam: tracking — first pose after "
                        f"{now() - self._t_start:.1f} s, "
                        f"{rec.n_tracked} map points")
            self.tracking_started.emit()
        self._n_fix += 1
        t = now()
        self._fix_marks.append(t)
        if len(self._fix_marks) > 30:
            self._fix_marks = self._fix_marks[-30:]
        span = self._fix_marks[-1] - self._fix_marks[0]
        hz = ((len(self._fix_marks) - 1) / span
              if len(self._fix_marks) > 1 and span > 0 else None)
        self.bus.nav_fix.emit(NavFix(
            t_capture=float(fix.t_capture),
            # n_tags stays 0 and reproj_rms_px stays None: there are no tags in
            # this run and a number in those columns would be a fabricated
            # measurement. `note` carries what actually localized.
            n_tags=0, tag_ids=(), tag_insts=(),
            p_ned=tuple(float(v) for v in fix.p_ned),
            R_ned_body=tuple(float(v) for v in
                             np.asarray(fix.R_ned_body, float).ravel()),
            yaw_ned=float(fix.yaw),
            reproj_rms_px=None, detect_ms=0.0, hz=hz, ambiguous=False,
            geometry="slam", source="slam",
            src_w=int(self.proc.width), src_h=int(self.proc.height),
            conn=Conn.ONLINE,
            note=(f"ORB-SLAM3 session {self._session}, "
                  f"{rec.n_tracked} map points"),
            stamp=t))
        # ATTITUDE. StateAssembler needs a VehicleImu for roll/pitch and the
        # yaw rate; with ArduSub powered its ATTITUDE messages are the better
        # source and this must not race them, so the synthesised one is
        # published ONLY when nothing else is producing (hardware.py decides,
        # and passes slam_attitude accordingly).
        if bool(getattr(self.opts, "slam_attitude", False)):
            self.bus.vehicle_imu.emit(VehicleImu(
                roll=float(fix.roll), pitch=float(fix.pitch),
                yaw=float(fix.yaw), t_att=float(fix.t_capture),
                conn=Conn.ONLINE))

    def _emit_miss(self, rec: PoseRecord, ev, note: str = "") -> None:
        """A not-ok beat. ``ok`` is False by having no ``p_ned``, which is the
        same shape ``TagNavWorker`` uses for "seen, none usable"."""
        why = note or getattr(ev, "why", "") or str(ev.status)
        self.bus.nav_fix.emit(NavFix(
            t_capture=float(rec.t), n_tags=0, tag_ids=(),
            geometry="slam", source="slam", conn=Conn.DEGRADED,
            note=f"ORB-SLAM3 {why}",
            src_w=int(self.proc.width), src_h=int(self.proc.height),
            stamp=now()))

    def _pump_knots(self) -> None:
        """Datum-NED plan -> SLAM world -> the map window.

        The transform is the EXACT inverse of the one the fix came through
        (``EngageDatum.map_from_datum_p`` then ``SlamNav.slam_from_ned_p``), so
        a knot lands on the map with no registration step and no fitted
        transform — if the polyline is in the wrong place, the bug is in the
        algebra and not in a calibration.
        """
        if not self._pending_viz:
            return
        datum = self._datum
        while self._pending_viz:
            viz = self._pending_viz.popleft()
            if datum is None:
                continue              # not engaged: nothing to place it against
            try:
                p_datum = np.asarray(viz.p_ned, float)      # (3, K)
                if p_datum.ndim != 2 or p_datum.shape[0] != 3:
                    continue
                # slam_from_datum_p IS the composed inverse; calling the two
                # halves separately here would be the fifth copy of the datum
                # isometry that slam_frames.EngageDatum exists to prevent.
                p_slam = self.nav.slam_from_datum_p(p_datum, datum)
                payload = W.encode_knots(
                    int(viz.plan_id), _status_code(viz.status),
                    float(viz.stamp), p_slam.T, None, None)
            except (ValueError, TypeError, AttributeError, KeyError):
                continue
            self.proc.submit_knots(payload)

    def _watchdog(self) -> None:
        """A child that has died, or stopped producing, must not look healthy."""
        if self._faulted or self.proc.proc is None:
            return                      # not spawned yet: no pair has arrived
        if not self.proc.alive():
            self._faulted = True
            tail = " | ".join(list(self.proc.stderr_tail)[-6:])
            self.bus.log.emit(
                "error",
                f"slam: ORB-SLAM3 child EXITED "
                f"(rc {self.proc.proc.returncode}). {tail}")
            self.failed.emit("slamnav: ORB-SLAM3 child exited")
            return
        if not self._started:
            # Not yet tracking is NORMAL for the first seconds (the vocabulary
            # is 145 MB), so the watchdog only speaks once the pipe has been
            # fed and nothing has come back for a long time.
            if (self.proc.n_frames_sent > 60
                    and now() - self._t_start > 20.0):
                self._faulted = True
                tail = " | ".join(list(self.proc.stderr_tail)[-6:])
                self.bus.log.emit(
                    "error",
                    f"slam: {self.proc.n_frames_sent} frames sent and NO pose "
                    f"— ORB-SLAM3 has not initialised. It needs one frame with "
                    f">500 ORB keypoints: point it at a textured room 1-5 m "
                    f"away, add light, and check CLAHE is on. {tail}")
            return
        if now() - self.proc.t_last_pose > POSE_WATCHDOG_S:
            self._faulted = True
            self.bus.log.emit(
                "error",
                f"slam: no pose for {POSE_WATCHDOG_S:.0f} s while the child is "
                f"alive — the tracker is not producing.")

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict:
        put_n, dropped, depth = self.pair_q.stats()
        return {"frames_queued": put_n, "frames_dropped": dropped,
                "queue_depth": depth,
                "frames_sent": int(self.proc.n_frames_sent),
                "poses": int(self.proc.n_poses),
                "fixes": int(self._n_fix),
                "session": int(self._session),
                "clahe": bool(self._clahe is not None),
                "tracking": bool(self._started),
                "faulted": bool(self._faulted)}
