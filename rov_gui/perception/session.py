#!/usr/bin/env python3
"""
session.py — the station's adapter onto the SAM2 tracker.

What this is
------------
``~/Desktop/New Folder/track.py`` is a complete application: it owns an OAK-D,
an OpenCV window, a mouse callback and the orchestration that ties SAM2 and
FoundationPose together. We want the orchestration and nothing else — the
station already owns the camera (DepthAI allows exactly one pipeline per device)
and already owns the window. So this class re-implements that orchestration
against the same library modules, and the upstream project is never modified.

``PoseSession`` is deliberately a plain object with no Qt in it:

    start(K, w, h)      load the model (slow, seconds); safe to call once
    submit(...)         hand it the newest frame, and optionally a click
    poll()              latest result, or None if nothing new
    close()             join the worker threads — MANDATORY, see below

That shape is the hedge described in the plan: the station calls it directly
today (one process), and if the environment ever has to be split the same
methods can sit behind a socket without any caller changing.

Why the imports are where they are
----------------------------------
``torch``, ``sam2`` and the upstream package are imported inside :meth:`start`,
never at module scope. Importing this module must stay free on a machine with no
CUDA — ``rov_gui`` has to keep starting there. :meth:`start` runs on the
perception worker's thread, after the QApplication exists, which is also the
ordering that keeps torch from fighting Qt over library paths (``rov_gui/qt.py``
documents two separate incidents of exactly that).

Threads and shutdown
--------------------
``LiveTracker`` is a daemon thread upstream, and :meth:`close` joins it anyway.
That is not tidiness: ``track.py:246-261`` records that letting the interpreter
tear a daemon thread down mid-torch-call unwinds through C++ and aborts. Join,
then return.
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

# Where the upstream project lives. Overridable because it is outside this repo
# and outside version control; the default is where it actually is today.
DEFAULT_SRC = Path(os.environ.get(
    "SAM2_LIVE_ROOT", str(Path.home() / "Desktop" / "New Folder")))

# Contour simplification tolerance, in pixels. The mask crosses the process/
# thread boundary as polylines rather than 230 kB of bools; 1.5 px is under the
# width of the line the overlay draws with, so the simplification is invisible.
CONTOUR_EPSILON = 1.5
MIN_CONTOUR_AREA = 12.0        # px^2; below this it is speckle, not an object

# Re-registration watchdog, values from the upstream project (track.py:57-60).
#
# FoundationPose's track_one() does not use the mask, so once the pose slides
# off the object it has no way back on its own. The only recovery is to notice
# and re-register: compare the mesh silhouette against SAM2's mask every frame,
# and when they stop agreeing, register again. register() costs ~0.73 s, so it
# is rate-limited — and if several attempts in a row do not stick, the object
# and the mesh probably are not the same thing and the pilot is told instead.
POSE_MIN_IOU = 0.25
POSE_BAD_FRAMES = 8
POSE_REG_INTERVAL = 1.5        # seconds between re-registration attempts
POSE_GIVE_UP = 4               # after this many, say so rather than keep trying

# The model-free pipeline's phases, the same three the upstream app runs
# (``track.py::_step_pipeline``). With a mesh in hand the first two are skipped
# and a click goes straight to pose; without one they are the only way to get a
# mesh, and they are the reason a click can take two minutes to become a pose.
PHASE_OFF = "off"              # SAM2 only: a mask, no 3-D, no reconstruction
PHASE_CAPTURE = "capture"      # collecting reference views while the pilot orbits
PHASE_BUILD = "build"          # BundleSDF running in a child process
PHASE_POSE = "pose"            # FoundationPose has a mesh
# A giving-up state that STICKS until the pilot resets. Falling back to
# PHASE_OFF instead was a real bug: the overlay reverted to plain green
# TRACKING, so a capture that had refused to reconstruct looked exactly like a
# capture that had never been asked to, and the reason was never shown.
PHASE_FAILED = "failed"

# Below this many reference views, reconstruction produces a mesh that looks
# plausible and is wrong (``capture.py:61``, MIN_VIEWS). Saying so beats
# spending two minutes of GPU to find out.
MIN_REF_VIEWS = 6


# Upstream speaks Korean to its operator; this station's UI is English
# (rov_gui/README.md, and the repo convention: prose Korean, UI labels English).
# The upstream tree is read-only to us, so the translation lives here, at the
# boundary — which is also the right place for it: these strings are the ONLY
# explanation the pilot gets for why a capture is not progressing, so leaving
# them untranslated would make the most important line on the panel the one
# nobody can act on.
#
# Applied in order with re.sub, so composed messages ("collection finished —
# <stop reason>") come out fully translated without a second pass.
_MESSAGE_RULES = (
    # capture: what the pilot should be doing
    (r"물체를 천천히 좌우로 돌려주세요", "turn the object slowly, left and right"),
    (r"물체의 3D 점이 부족합니다 \(더 가까이\)",
     "not enough 3-D points on the object — move closer"),
    (r"물체의 3D 점이 부족 \(마스크가 작거나 뎁스가 비었다\)",
     "not enough 3-D points (small mask, or no depth there)"),
    (r"물체가 너무 멉니다 \(([\d.]+) cm\) — 30~80 cm 로 가져오세요",
     r"object too far (\1 cm) — bring it to 30-80 cm"),
    # capture: why registration of this frame failed
    (r"겹치는 부분을 찾지 못했다 \(더 천천히 / 더 가까이\)",
     "no overlap found — move slower, or closer"),
    (r"정합 결과가 유효하지 않다", "registration result is not valid"),
    (r"겹침 부족 ([\d.]+)", r"not enough overlap (\1)"),
    (r"정합 오차 큼 ([\d.]+)mm", r"registration error too large (\1 mm)"),
    (r"정합 실패 — 더 천천히 움직여 주세요",
     "registration failed — move more slowly"),
    (r"추적 시작 전", "not started yet"),
    # capture: why this frame was not kept as a reference view
    (r"마스크가 너무 작다", "mask too small"),
    (r"마스크가 갑자기 작아졌다", "mask suddenly shrank"),
    (r"마스크가 갑자기 커졌다 \(다른 물체로 옮겨탔을 수 있다\)",
     "mask suddenly grew — it may have jumped to another object"),
    (r"마스크가 조각나 있다", "mask is fragmented"),
    (r"마스크 없음", "no mask"),
    (r"정합 신뢰도 낮음 \(([\d.]+)\)", r"registration confidence too low (\1)"),
    # Upstream says only "implausible distance". Say the target instead: this
    # is the one reject the pilot can fix in a second, and "implausible" does
    # not tell them which way to move.
    (r"카메라-물체 거리가 이상하다 \((\d+) mm\)",
     r"object is \1 mm away — bring it to 300-800 mm"),
    (r"각도 차이가 작다 \((\d+)도\)",
     r"too close in angle to a view already taken (\1 deg)"),
    # capture: progress and endings
    (r"담았습니다 \((\d+)장\)", r"captured (\1 views)"),
    (r"수집 완료 — ", "collection finished — "),
    (r"수집 오류: ", "capture error: "),
    (r"누적 회전 ([\d.]+)도 도달", r"reached \1 deg of accumulated rotation"),
    (r"(\d+)장 채움", r"collected the full \1 views"),
    (r"담은 뷰 없음", "no views captured"),
    # capture: the summary line that goes to the log
    (r"회전 ([\d.]+)도", r"arc \1 deg"),
    (r"마스크 (\d+)~(\d+) px \(중앙 (\d+)\)",
     r"mask \1-\2 px (median \3)"),
    (r"거리 (\d+)~(\d+) mm", r"distance \1-\2 mm"),
    (r"(\d+)장", r"\1 views"),          # after the two rules that need '장'
    # tracker
    (r"물체를 클릭하세요", "click an object"),
    (r"SAM2 오류: ", "SAM2 error: "),
    (r"물체가 아닌 것 같습니다\. 다시 클릭하세요",
     "that does not look like an object — click again"),
    (r"놓침 — 재획득 중 \(([\d.]+)s\)", r"lost — reacquiring (\1 s)"),
    (r"놓침 — 재획득 중", "lost — reacquiring"),
    (r"재획득 실패 — 다시 클릭하세요", "could not reacquire — click again"),
)


def english(msg: str) -> str:
    """Translate an upstream operator message for this station's UI."""
    import re

    if not msg:
        return ""
    for pat, rep in _MESSAGE_RULES:
        msg = re.sub(pat, rep, msg)
    return msg


class PoseSessionError(RuntimeError):
    """Raised by start() only. Everything after that is reported, not raised."""


class PoseSession:
    """SAM2 object tracking driven by frames the station already has.

    Three ways to run, and which one you get depends only on what you passed:

    ``mesh=...``      a click registers against that mesh in ~0.73 s and 6-DoF
                      pose follows immediately. Nothing is reconstructed.
    ``build=True``    no mesh: a click starts collecting reference views while
                      the pilot orbits the object, then reconstructs one
                      (BundleSDF, ~2 min in a child process), then pose. This
                      is what ``track.py --pose`` does.
    neither           tracking only. ``T_cam_obj`` stays None, and that is a
                      useful mode on its own — a persistent marker on the thing
                      the pilot cares about, with no 3-D model, no 30-80 cm
                      working distance and no GPU-minutes.
    """

    def __init__(self, src_dir: Path | str | None = None,
                 model: str = "tiny", ckpt: str | None = None,
                 mesh: str | None = None,
                 lost_grace: float = 3.0, lost_timeout: float = 15.0,
                 build: bool = False, ref_dir: Path | str | None = None,
                 max_arc: float = 75.0, max_views: int = 20):
        self.src_dir = Path(src_dir) if src_dir else DEFAULT_SRC
        self.model = model
        self.ckpt = ckpt
        self.mesh = mesh
        self.lost_grace = float(lost_grace)
        self.lost_timeout = float(lost_timeout)
        self.ref_root = Path(ref_dir) if ref_dir else Path("sessions/pose_meshes")
        self.max_arc = float(max_arc)
        self.max_views = int(max_views)

        # A mesh short-circuits the first two phases; asking to build without
        # one enters them; neither leaves the session in tracking-only.
        self._wants_build = bool(build) and not mesh
        self.phase = (PHASE_POSE if mesh else
                      (PHASE_CAPTURE if build else PHASE_OFF))
        self._fp_t0 = 0.0              # when FoundationPose started loading
        self._capture = None           # sam2_live.capture.CaptureWorker
        self._builder = None           # _MeshBuild (our own, cancellable)
        self._geom = None              # _PinholeUnprojector, built on frame 1
        self._ref_dir = None           # this run's reference-view directory
        self._phase_note = ""
        self._n_views = 0
        self._arc_deg = 0.0
        self._build_t0 = 0.0
        self._build_s = 0.0
        self._cap_dist = 0.0           # camera->object, m, during capture
        self._mesh_lines: tuple = ()

        self._live = None              # sam2_live.tracker.LiveTracker
        self._pose = None              # sam2_live.pose.PoseWorker
        self._loader = None            # the pose-load thread, for close()
        self._closed = False           # set by close(); a closed session stays closed
        self._cv2 = None
        self._np_K = None              # 3x3 float64, set on the first frame
        self._lock = threading.Lock()
        self._loading = False
        self._ready = False
        self._error = ""
        self._t_load0 = 0.0
        self._load_s = 0.0
        self._last_seq = 0
        self._src_wh = (0, 0)
        self._pending_click: tuple[float, float] | None = None
        # pose watchdog state (see the constants above)
        self._prev_sam_state = ""
        self._bad_iou = 0
        self._failed_reg = 0
        self._last_reg = 0.0
        self._iou = None
        self._axes_px: tuple = ()
        self._box_px: tuple = ()
        self._pose_note = ""
        self._mesh_desc = ""

    # ------------------------------------------------------------- lifecycle
    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def loading(self) -> bool:
        return self._loading

    @property
    def error(self) -> str:
        return self._error

    @property
    def load_seconds(self) -> float:
        if self._ready:
            return self._load_s
        return (time.monotonic() - self._t_load0) if self._loading else 0.0

    def start_async(self, on_log=None) -> None:
        """Load the model on a private thread and return immediately.

        Loading is seconds to tens of seconds. Doing it inline would block the
        perception worker's tick, which is the thread that also has to keep
        answering the GUI — so the station would freeze on the click that turned
        tracking on. The UI shows ``loading`` with an elapsed counter instead.
        """
        with self._lock:
            if self._loading or self._ready:
                return
            self._loading = True
            self._t_load0 = time.monotonic()

        def _run():
            try:
                self._load(on_log)
            except Exception as e:                               # noqa: BLE001
                with self._lock:
                    self._error = f"{type(e).__name__}: {e}"
                    self._loading = False
                if on_log and not self._closed:
                    on_log("error", f"pose: {self._error}")
            else:
                with self._lock:
                    self._load_s = time.monotonic() - self._t_load0
                    self._loading = False
                    # A session closed mid-load must not come back to life:
                    # ready=True here would advertise a tracker close() never
                    # saw and will never join.
                    self._ready = not self._closed
                if on_log and self._ready:
                    on_log("info", f"pose: SAM2 '{self.model}' ready "
                                   f"in {self._load_s:.1f} s")

        t = threading.Thread(target=_run, name="pose-load", daemon=True)
        with self._lock:
            self._loader = t
        t.start()

    def _load(self, on_log=None) -> None:
        """The heavy import + model build. Runs on the loader thread."""
        if not self.src_dir.is_dir():
            raise PoseSessionError(
                f"perception source not found at {self.src_dir} — set "
                f"SAM2_LIVE_ROOT or pass --pose-src")
        # Prepend rather than append: this must win over anything similarly
        # named, and it is removed from nothing afterwards because the upstream
        # modules keep importing each other lazily.
        src = str(self.src_dir)
        if src not in sys.path:
            sys.path.insert(0, src)

        from ..qt import import_cv2
        self._cv2 = import_cv2()

        from sam2_live.tracker import LiveTracker, StreamingSam2Tracker

        if on_log:
            on_log("info", f"pose: loading SAM2 '{self.model}' from {src}")
        sam = StreamingSam2Tracker(model=self.model, ckpt=self.ckpt)
        live = LiveTracker(sam, lost_grace=self.lost_grace,
                           lost_timeout=self.lost_timeout)
        # Start UNDER the lock, only if nobody closed us while the model was
        # building. close() takes the same lock to set _closed, so either it
        # ran first and this thread never starts — or this store wins and
        # close() finds the tracker to join. There is no third ordering; the
        # old start-then-store version had one, and it leaked a running torch
        # daemon thread into interpreter exit (the abort track.py:246-261
        # documents) whenever the station quit inside the load window.
        with self._lock:
            if self._closed:
                raise PoseSessionError("session closed during load")
            live.start()
            self._live = live

    def close(self) -> None:
        """Stop and JOIN. Never leave these threads to the interpreter's exit.

        ``track.py:246-261`` records why: a daemon thread torn down mid-torch
        call unwinds through C++ and aborts the process. Pose first, because it
        is the one holding a CUDA context.
        """
        # Before the threads: a reconstruction child outlives its parent unless
        # someone kills it, and it is holding GB of VRAM.
        self._abandon_build()
        with self._lock:
            self._closed = True
            live, self._live = self._live, None
            pose, self._pose = self._pose, None
            self._ready = False
            loader = self._loader
        for worker, timeout in ((pose, 10.0), (live, 5.0)):
            if worker is None:
                continue
            try:
                worker.stop()
                worker.join(timeout=timeout)
            except Exception:                                    # noqa: BLE001
                pass
        # The loader too. It may be seconds deep in building SAM2, and that
        # build cannot be interrupted — but returning while it runs hands a
        # live torch thread to interpreter teardown, which is the abort this
        # whole method exists to prevent. _closed (set above, under the same
        # lock the loader checks) guarantees it will discard its result.
        if loader is not None and loader is not threading.current_thread():
            loader.join(timeout=30.0)

    # ---------------------------------------------------------------- frames
    def click(self, x: float, y: float) -> None:
        """Queue a prompt in SOURCE pixels; it goes with the next frame.

        Held rather than applied, because upstream is explicit that a click must
        travel with the frame it was made on (``tracker.py:400-401``) — applying
        it to whatever frame arrives next is how a prompt lands on the wrong
        pixel after the vehicle moves.
        """
        with self._lock:
            self._pending_click = (float(x), float(y))

    def reset(self, on_log=None) -> None:
        """Drop the object AND abandon anything being collected or built.

        This is the pilot's abort. Turning TRACK off during a reconstruction has
        to actually stop it — a BundleSDF child left running would keep several
        GB of VRAM and a GPU busy behind a station that says it is idle.
        """
        with self._lock:
            live, fp = self._live, self._pose
            self._pending_click = None
            self._axes_px = self._box_px = ()
            self._iou = None
        if live is not None:
            live.request_reset()
        # The estimator too, and the watchdog counters with it (track.py:441-448).
        # Without this the next click re-registers only after the IoU watchdog
        # has watched eight bad frames go by — the pose would follow the OLD
        # object for a second after the pilot picked a new one.
        if fp is not None:
            try:
                fp.request_reset()
            except Exception:                                    # noqa: BLE001
                pass
        self._prev_sam_state = ""
        self._bad_iou = self._failed_reg = 0
        self._pose_note = ""
        self._abandon_build(on_log)
        # Back to the top of whichever pipeline this session was configured for.
        # A mesh, once reconstructed, is kept: re-clicking should re-register
        # against it, not spend another two minutes rebuilding it. A FAILED
        # capture is retried, which is the only way out of that state and is
        # what the message on screen tells the pilot to do.
        if self.phase in (PHASE_CAPTURE, PHASE_BUILD, PHASE_FAILED):
            self.phase = PHASE_CAPTURE if self._wants_build else PHASE_OFF
            self._n_views, self._arc_deg, self._build_s = 0, 0.0, 0.0
            self._phase_note = ""

    def _abandon_build(self, on_log=None) -> None:
        with self._lock:
            cap, self._capture = self._capture, None
            bld, self._builder = self._builder, None
        if cap is not None:
            try:
                cap.stop()
                cap.join(timeout=2.0)
            except Exception:                                    # noqa: BLE001
                pass
        if bld is not None:
            bld.cancel()
            if on_log:
                on_log("warn", "pose: reconstruction cancelled")

    def submit(self, color, depth=None, t_capture: float = 0.0,
               seq: int = 0, K=None, on_log=None) -> None:
        """Hand over the newest frame. Never blocks; the tracker drops."""
        with self._lock:
            live = self._live
            click, self._pending_click = self._pending_click, None
            self._last_seq = seq
            self._src_wh = (int(color.shape[1]), int(color.shape[0]))
            if K is not None and self._np_K is None:
                self._np_K = np.asarray(K, dtype=np.float64)
        if live is None:
            return
        live.submit(color, click=click)
        self._step_pipeline(color, depth, on_log)

    # -------------------------------------------------------------- pipeline
    def _step_pipeline(self, color, depth, on_log=None) -> None:
        """One step of whichever phase we are in. Mirrors track.py:294-305.

        Everything below needs depth. Saying so once here keeps three separate
        "if depth is None" branches out of the phase steps.
        """
        if self.phase == PHASE_OFF:
            return
        if depth is None:
            self._phase_note = "no depth — 3-D needs an RGB-D pair"
            return
        if self.phase == PHASE_CAPTURE:
            self._step_capture(color, depth, on_log)
        elif self.phase == PHASE_BUILD:
            self._step_build(on_log)
        elif self.phase == PHASE_POSE:
            self._start_pose_if_needed(on_log)
            self._step_pose(color, depth)

    def _start_pose_if_needed(self, on_log=None) -> None:
        """Bring FoundationPose up the first time we have both mesh and K."""
        if self._pose is not None or not self.mesh:
            return
        with self._lock:
            if self._np_K is None:
                return
        try:
            self.start_pose(str(self.mesh), on_log=on_log)
        except Exception as e:                                   # noqa: BLE001
            self.phase = PHASE_FAILED
            self._phase_note = f"mesh {Path(str(self.mesh)).name}: {e}"
            if on_log:
                on_log("error", f"pose: {self._phase_note}")

    def _step_capture(self, color, depth, on_log=None) -> None:
        """Collect reference views while SAM2 holds the object.

        The pilot's half of this is a slow orbit. Upstream measured why it has
        to stay slow and short: the per-view camera pose comes from colored ICP
        against the accumulated cloud, and past ~80 deg of accumulated rotation
        the error goes off a cliff (9.7 mm at 80 deg, 87 mm at 100, 283 mm at
        120 — ``capture.py:7-20``). So the arc is capped at 75 deg and the
        capture ENDS there rather than collecting more and getting worse.
        """
        # Finalize FIRST, and without asking SAM2's permission. The pilot stops
        # orbiting the moment the counter says done, and SAM2 routinely loses
        # the object at exactly that moment — a finalize gated behind
        # state == TRACKING (as this was) then waits forever for a state that
        # no longer matters, with "collection finished" frozen on screen.
        cap = self._capture
        if cap is not None and cap.snapshot().get("done"):
            self._finish_capture(cap, on_log)
            return

        snap = self._live.snapshot() if self._live is not None else {}
        state, mask = snap.get("state"), snap.get("mask")
        if state != "TRACKING" or mask is None:
            self._phase_note = "click the object to start collecting views"
            return

        if cap is None:
            cap = self._begin_capture(color, depth, on_log)
            if cap is None:
                return
        if mask.any():
            cap.submit(color, depth, mask)

        cs = cap.snapshot()
        self._n_views = int(cs.get("n", 0))
        self._arc_deg = float(cs.get("arc", 0.0))
        self._cap_dist = float(cs.get("distance") or 0.0)
        # The REJECT reason wins over the generic "turn the object slowly",
        # because when the counter is stuck that is the only line that says
        # why. Upstream keeps them separate and shows both; the station has one
        # line, so it shows the one that is actionable.
        self._phase_note = english(str(cs.get("reject")
                                       or cs.get("message") or ""))
        if cs.get("done"):
            self._finish_capture(cap, on_log)

    def _finish_capture(self, cap, on_log=None) -> None:
        """Collection is over — either the arc or the view budget ran out."""
        with self._lock:
            self._capture = None
        cap.stop()
        cap.join(timeout=5.0)
        rec = cap.recorder
        # The recorder's own final numbers, not the last snapshot's: on the
        # finalize-first path the snapshot counters are one tick stale.
        self._n_views = int(rec.n)
        self._arc_deg = float(getattr(rec, "arc", self._arc_deg) or 0.0)
        if on_log:
            on_log("info", f"pose: views {english(rec.summary())}")
        if rec.n < MIN_REF_VIEWS:
            self.phase = PHASE_FAILED
            # Say what ran out, not just that something did. The arc is a
            # budget spent by ROTATION, so the usual cause is turning too fast:
            # 75 deg goes by while most frames are still being rejected, and
            # the count never reaches six.
            self._phase_note = (
                f"stopped after {self._arc_deg:.0f} deg with only {rec.n} "
                f"usable views (need {MIN_REF_VIEWS}) — turn the object more "
                f"slowly, keep it 30-80 cm away, then press TRACK again")
            if on_log:
                on_log("warn", f"pose: {self._phase_note}")
            return
        if rec.n < 10 and on_log:
            on_log("warn", f"pose: only {rec.n} views (10+ recommended); the "
                           f"mesh may be poor")
        path = rec.write()
        if on_log:
            on_log("info", f"pose: reference views -> {path}")
        builder = _MeshBuild(path, src_dir=self.src_dir)
        builder.start()
        with self._lock:
            self._builder = builder
        self._build_t0 = time.monotonic()
        self._build_s = 0.0
        # Clear the capture message first: it says "orbit the object slowly",
        # and leaving it under a RECONSTRUCTING chip tells the pilot to keep
        # doing something that no longer matters.
        self._phase_note = "reconstructing the mesh (about 2 minutes)"
        self.phase = PHASE_BUILD

    def _begin_capture(self, color, depth, on_log=None):
        """Create the upstream CaptureWorker. Returns it, or None on failure."""
        with self._lock:
            K = self._np_K
        if K is None:
            return None
        if depth.shape[:2] != color.shape[:2]:
            self.phase = PHASE_FAILED
            self._phase_note = (f"depth {depth.shape[1]}x{depth.shape[0]} does "
                                f"not match colour {color.shape[1]}x"
                                f"{color.shape[0]} — reconstruction needs them "
                                f"aligned and the same size")
            if on_log:
                on_log("error", f"pose: {self._phase_note}")
            return None
        try:
            ref = self._new_ref_dir()
            from sam2_live.capture import CaptureWorker
            geom = _PinholeUnprojector(K, depth.shape[1], depth.shape[0])
            cap = CaptureWorker(geom, str(ref), K, max_arc_deg=self.max_arc,
                                max_views=self.max_views)
            cap.start()
        except Exception as e:                                   # noqa: BLE001
            self.phase = PHASE_FAILED
            self._phase_note = f"cannot start capture: {e}"
            if on_log:
                on_log("error", f"pose: {self._phase_note}")
            return None
        with self._lock:
            self._capture, self._geom, self._ref_dir = cap, geom, ref
        if on_log:
            on_log("info", f"pose: collecting reference views in {ref} — "
                           f"orbit the object slowly, up to "
                           f"{self.max_arc:.0f} deg")
        return cap

    def _new_ref_dir(self) -> Path:
        """A fresh directory per attempt, so a retry cannot mix two objects.

        Reference views are numbered from zero and the reconstruction reads the
        whole directory; reusing one would silently reconstruct a chimera of
        this object and the last one.
        """
        root = self.ref_root
        # Upstream refuses any path containing 'rgb' and it is right to: the
        # reconstruction builds its sibling paths with str.replace on that
        # substring (``capture.py:254-256``), so a directory called .../rgb_test
        # would have its depth read out of a directory that does not exist.
        if "rgb" in str(root.resolve() if root.is_absolute() else
                       (Path.cwd() / root)):
            raise PoseSessionError(
                f"reference directory {root} contains 'rgb'; the reconstruction "
                f"derives sibling paths by replacing that substring")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = root / f"obj_{stamp}"
        n = 1
        while out.exists():
            n += 1
            out = root / f"obj_{stamp}_{n}"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _step_build(self, on_log=None) -> None:
        b = self._builder
        if b is None:
            self.phase = PHASE_OFF
            return
        self._build_s = b.elapsed
        if b.running:
            self._phase_note = "reconstructing the mesh (about 2 minutes)"
            return
        with self._lock:
            self._builder = None
        if b.error:
            self.phase = PHASE_FAILED
            self._phase_note = b.error
            if on_log:
                on_log("error", f"pose: reconstruction failed: {b.error}")
            return
        if on_log:
            on_log("info", f"pose: mesh built in {b.elapsed:.0f} s: "
                           f"{b.mesh_path}")
        # Verify before trusting it. Reconstruction fails QUIETLY — a pose that
        # drifted during capture makes rays get discarded rather than raise, and
        # what comes out is a confident-looking mesh of the wrong thing
        # (``nerf.py:7-13``). These lines are the only warning anyone gets.
        lines = tuple(self._verify(b.ref_dir, b.mesh_path))
        self._mesh_lines = lines
        if on_log:
            for ln in lines:
                on_log("info", f"pose: {ln}")
        self.mesh = b.mesh_path
        self.phase = PHASE_POSE
        self._phase_note = ""
        self._start_pose_if_needed(on_log)

    def _verify(self, ref_dir, mesh_path) -> list:
        try:
            from sam2_live.nerf import verify
            return list(verify(ref_dir, mesh_path))
        except Exception as e:                                   # noqa: BLE001
            return [f"mesh check unavailable: {e}"]

    # ------------------------------------------------------------------ pose
    def start_pose(self, mesh: str, on_log=None) -> None:
        """Bring up FoundationPose for a mesh. Needs K, so call after a frame.

        The estimator is built INSIDE its own thread upstream and must stay
        there: its nvdiffrast context is only valid on the thread that created
        it (``pose.py:262-267``). So all we hand over is a path.
        """
        with self._lock:
            if self._pose is not None or self._np_K is None:
                return
            K = self._np_K
        from sam2_live.pose import PoseWorker as FpWorker

        w = FpWorker(mesh, K)
        w.start()
        with self._lock:
            self._pose = w
            self.mesh = mesh
            self._fp_t0 = time.monotonic()
        self._phase_note = f"loading FoundationPose for {Path(mesh).name}"
        if on_log:
            on_log("info", f"pose: FoundationPose loading for {Path(mesh).name}")

    def _step_pose(self, color, depth) -> None:
        """The per-frame pose step: watchdog, register/track, project.

        A re-implementation of ``track.py::App._step_pose`` minus the
        rectification. The C3's colour camera is 63.9 deg with a measured
        maximum pinhole disagreement of 2.08 px at 640x360 — 2.4 mm at 0.6 m —
        so RGB, depth, mask, click and every projected point live in ONE
        coordinate system instead of two. The upstream dual-frame machinery
        exists for a 96 deg lens that is 232 px out at the corners; carrying it
        here would only carry its bugs.
        """
        with self._lock:
            w, live, K = self._pose, self._live, self._np_K
        if w is None or live is None or K is None or not w.ready.is_set():
            return
        # The loading note has served its purpose the moment the estimator is
        # ready. Left in place (as it was), it sat under a live POSE TRACKING
        # chip for the rest of the session, telling the pilot a load that
        # finished long ago was still running.
        if self._phase_note.startswith("loading FoundationPose"):
            self._phase_note = ""
        if w.fatal:
            with self._lock:
                self._pose_note = w.error or "FoundationPose failed to load"
            return
        if depth is None:
            with self._lock:
                self._pose_note = "no depth — pose needs an RGB-D pair"
            return

        snap = live.snapshot()
        state, mask = snap.get("state"), snap.get("mask")
        recovered = (self._prev_sam_state == "LOST" and state == "TRACKING")
        self._prev_sam_state = state
        if state != "TRACKING":
            return

        T = w.snapshot()["pose"]
        need_register = (w.est is not None and w.est.pose is None) or recovered

        # The watchdog. track_one() ignores the mask, so a pose that has slid
        # off the object stays off; comparing the mesh silhouette with SAM2's
        # mask is the only thing that notices.
        if T is not None and mask is not None and mask.any() and not need_register:
            sil = w.est.silhouette(T, K, mask.shape)
            if sil is not None:
                inter = int(np.count_nonzero(sil & mask))
                union = int(np.count_nonzero(sil | mask))
                iou = (inter / union) if union else 0.0
                with self._lock:
                    self._iou = iou
                if iou < POSE_MIN_IOU:
                    self._bad_iou += 1
                    if (self._bad_iou >= POSE_BAD_FRAMES
                            and time.monotonic() - self._last_reg > POSE_REG_INTERVAL):
                        need_register = True
                        self._bad_iou = 0
                        self._last_reg = time.monotonic()
                        self._failed_reg += 1
                else:
                    self._bad_iou = 0
                    self._failed_reg = 0

        if need_register and (mask is None or not mask.any()):
            return
        w.submit(color, depth, mask, register=need_register)
        with self._lock:
            self._pose_note = (
                "pose does not fit — wrong object, or the mesh is not this one"
                if self._failed_reg >= POSE_GIVE_UP else "")

        T = w.snapshot()["pose"]
        if T is None:
            with self._lock:
                self._axes_px = self._box_px = ()
            return
        try:
            axes = _project(w.est.axes_cam(T), K)
            box = _project(w.est.box_corners_cam(T), K)
        except Exception:                                        # noqa: BLE001
            axes = box = ()
        with self._lock:
            self._axes_px, self._box_px = axes, box

    def poll(self) -> dict | None:
        """Latest tracker state as plain data, or None if not running.

        Returns a dict rather than a PoseTrack so this module stays free of
        ``rov_gui.state`` — the boundary is data, and the worker builds the
        dataclass. Keeps this class usable from a subprocess unchanged.
        """
        with self._lock:
            live = self._live
            seq, (w, h) = self._last_seq, self._src_wh
        if live is None:
            return None
        snap = live.snapshot()
        mask = snap.get("mask")
        contours = self._contours(mask) if mask is not None else ()
        state = str(snap.get("state", "")).lower()
        out = {
            "state": state or "idle",
            "contours": contours,
            "score": _finite(snap.get("score")),
            "mask_px": int(mask.sum()) if mask is not None else 0,
            "sam_hz": float(snap.get("hz") or 0.0),
            "message": english(str(snap.get("message") or "")),
            "frame_seq": seq,
            "src_w": w,
            "src_h": h,
            "T_cam_obj": None,
            "axes_px": (),
            "box_px": (),
            "pose_hz": 0.0,
            "n_register": 0,
            "iou": None,
            "mesh": self._mesh_desc,
            "phase": self.phase,
            "n_views": self._n_views,
            "arc_deg": self._arc_deg,
            "max_arc": self.max_arc,
            "max_views": self.max_views,
            "build_s": self._build_s,
            "distance_m": self._cap_dist,
            "fp_load_s": 0.0,
            # Whether a 6-DoF pose is even expected. Without it the overlay
            # cannot tell "no pose because you asked for a mask" apart from
            # "no pose because something went wrong", and it has to.
            "pose_expected": bool(self.mesh) or self._wants_build,
        }
        with self._lock:
            fp = self._pose
            out["axes_px"], out["box_px"] = self._axes_px, self._box_px
            out["iou"] = self._iou
            fp_t0 = self._fp_t0
        ps = fp.snapshot() if fp is not None else {}
        if fp is not None:
            T = ps.get("pose")
            if T is not None:
                # Plain floats: what crosses the boundary must not be a numpy
                # view into something the worker is about to overwrite.
                out["T_cam_obj"] = tuple(float(v)
                                         for v in np.asarray(T).reshape(-1))
            out["pose_hz"] = float(ps.get("hz") or 0.0)
            out["n_register"] = int(ps.get("n_register") or 0)
            fp_ready = bool(fp.ready.is_set()) and not fp.fatal
            if not fp_ready and fp_t0:
                out["fp_load_s"] = time.monotonic() - fp_t0
        else:
            fp_ready = False

        # --- what the pilot is actually looking at -------------------------
        #
        # The phase OVERRIDES the tracker's own state, because during the
        # reconstruction phases SAM2 says TRACKING and that is true and useless:
        # the question on screen is how much of the orbit is left, how long the
        # mesh has been building, or why it stopped. Getting this wrong was a
        # real failure — a capture that gave up reverted to a plain green
        # TRACKING chip and the reason was never shown anywhere.
        if self.phase == PHASE_FAILED:
            out["state"] = "failed"
        elif self.phase == PHASE_CAPTURE and self._capture is not None:
            out["state"] = "capturing"
        elif self.phase == PHASE_BUILD:
            out["state"] = "building"
        elif self.phase == PHASE_POSE and not fp_ready:
            out["state"] = "fault" if (fp is not None and fp.fatal) \
                else "pose_loading"
        elif (self.phase == PHASE_POSE and state == "tracking"
                and out["T_cam_obj"] is None):
            # Registration is a distinct 0.73 s during which the object is
            # tracked and the pose is not there yet.
            out["state"] = "registering"
        if self._phase_note:
            out["message"] = self._phase_note
        if self._pose_note:
            out["message"] = self._pose_note
        if ps.get("fatal"):
            out["message"] = str(ps.get("error") or "FoundationPose failed")
        return out

    @property
    def pose_ready(self) -> bool:
        w = self._pose
        return bool(w is not None and w.ready.is_set() and not w.fatal)

    @property
    def ref_dir(self) -> str:
        """Where this run's reference views went, or '' if none were collected."""
        return str(self._ref_dir) if self._ref_dir else ""

    @property
    def mesh_check_lines(self) -> tuple:
        """Human-readable results of the post-reconstruction sanity checks."""
        return self._mesh_lines

    @property
    def mesh_description(self) -> str:
        w = self._pose
        if w is not None and w.est is not None and not self._mesh_desc:
            try:
                self._mesh_desc = str(w.est.describe())
            except Exception:                                    # noqa: BLE001
                pass
        return self._mesh_desc

    # ------------------------------------------------------------- internals
    def _contours(self, mask) -> tuple:
        """bool mask -> simplified polylines in source pixels.

        Done HERE, next to the mask, so what crosses to the GUI is ~1 kB of
        points instead of a 230 kB bitmap that the GUI thread would have to
        rasterise per frame.
        """
        cv2 = self._cv2
        if cv2 is None or mask is None:
            return ()
        m = np.ascontiguousarray(mask.astype(np.uint8))
        found, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in found:
            if cv2.contourArea(c) < MIN_CONTOUR_AREA:
                continue
            approx = cv2.approxPolyDP(c, CONTOUR_EPSILON, True)
            pts = tuple((float(p[0][0]), float(p[0][1])) for p in approx)
            if len(pts) >= 3:
                out.append(pts)
        return tuple(out)


class _PinholeUnprojector:
    """The one method the upstream capture path needs from a camera object.

    ``capture.mask_cloud`` calls ``rect.unproject(depth_mm, mask)`` and nothing
    else on what it is handed — the only other references to it are storing it
    (``capture.py:87,125,148,172``). Upstream that object is a
    ``RectifiedCamera`` carrying a whole remap; here it is nine numbers, because
    this camera is not rectified (see :meth:`PoseSession._step_pose`).

    Same convention as upstream: millimetres in, metres out, OpenCV axes, and
    the 200-4000 mm clamp that keeps the stereo module's zero and saturation
    values out of the cloud.
    """

    def __init__(self, K, width: int, height: int):
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.K = K
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        xs = (np.arange(width, dtype=np.float32) - float(cx)) / float(fx)
        ys = (np.arange(height, dtype=np.float32) - float(cy)) / float(fy)
        self.ray_x = np.tile(xs, (height, 1))
        self.ray_y = np.tile(ys[:, None], (1, width))
        self.width, self.height = int(width), int(height)

    def unproject(self, depth_mm, mask=None, min_mm=200, max_mm=4000):
        valid = (depth_mm >= min_mm) & (depth_mm <= max_mm)
        if mask is not None:
            valid = valid & mask
        ys, xs = np.nonzero(valid)
        if xs.size == 0:
            return np.zeros((0, 3), np.float32), (ys, xs)
        z = depth_mm[ys, xs].astype(np.float32) * 0.001
        pts = np.stack([self.ray_x[ys, xs] * z,
                        self.ray_y[ys, xs] * z, z], axis=1)
        return pts, (ys, xs)


class _MeshBuild(threading.Thread):
    """Run BundleSDF over a reference-view directory. Cancellable.

    A re-implementation of ``sam2_live.nerf.MeshBuilder`` for one reason:
    upstream uses ``subprocess.call``, which cannot be interrupted. On a station
    that is also flying a vehicle, a two-minute GPU job the pilot cannot abort
    is not acceptable — turning TRACK off has to actually stop it. ``Popen`` +
    :meth:`cancel` is the whole difference; everything else (the script path,
    the ``osmesa`` workaround, the log file, the checks on the way out) is
    upstream's and is kept identical on purpose.

    A CHILD process, not a thread, and that is upstream's choice worth keeping:
    reconstruction holds the GPU for minutes, and in-process it would stall the
    tracker and take the station down with it if it died. A child also returns
    its several GB of VRAM for certain when it exits.
    """

    def __init__(self, ref_dir, src_dir=None, fp_root: str | None = None,
                 dataset: str = "ycbv"):
        super().__init__(daemon=True, name="pose-mesh-build")
        self.ref_dir = os.path.abspath(str(ref_dir))
        self.fp_root = fp_root or os.environ.get(
            "FOUNDATIONPOSE_ROOT",
            "/home/bdml/Desktop/poseEstimation/FoundationPose")
        self.src_dir = str(src_dir) if src_dir else None
        self.dataset = dataset
        self.log_path = os.path.join(self.ref_dir, "nerf_train.log")
        self.mesh_path = None
        self.error = None
        self._proc = None
        self._t0 = 0.0
        self._t1 = 0.0
        self._cancelled = False

    @property
    def elapsed(self) -> float:
        if not self._t0:
            return 0.0
        return (self._t1 or time.monotonic()) - self._t0

    @property
    def running(self) -> bool:
        return self.is_alive()

    def cancel(self) -> None:
        self._cancelled = True
        p = self._proc
        if p is not None and p.poll() is None:
            try:
                p.terminate()
            except Exception:                                    # noqa: BLE001
                pass

    def run(self) -> None:
        import subprocess

        self._t0 = time.monotonic()
        script = os.path.join(self.fp_root, "bundlesdf", "run_nerf_single.py")
        if not os.path.exists(script):
            self.error = (f"reconstruction script not found: {script} — set "
                          f"FOUNDATIONPOSE_ROOT")
            self._t1 = time.monotonic()
            return
        env = dict(os.environ)
        # Upstream's note: the NVIDIA EGL path is unstable during pyrender's
        # texture bake, so force software GL for the child only (nerf.py:62-64).
        env.setdefault("PYOPENGL_PLATFORM", "osmesa")
        try:
            with open(self.log_path, "w") as log:
                self._proc = subprocess.Popen(
                    [sys.executable, script, "--ref_view_dir", self.ref_dir,
                     "--dataset", self.dataset],
                    cwd=os.path.dirname(script), stdout=log,
                    stderr=subprocess.STDOUT, env=env)
                rc = self._proc.wait()
        except Exception as exc:                                 # noqa: BLE001
            self.error = f"could not run the reconstruction: {exc}"
            self._t1 = time.monotonic()
            return
        finally:
            self._t1 = time.monotonic()
        if self._cancelled:
            self.error = "cancelled"
            return
        if rc != 0:
            self.error = f"reconstruction exited {rc} (log: {self.log_path})"
            return
        out = os.path.join(self.ref_dir, "model", "model.obj")
        if not os.path.exists(out):
            self.error = f"no mesh was produced (log: {self.log_path})"
            return
        self.mesh_path = out


def _project(pts, K) -> tuple:
    """Camera-frame metres -> source pixels, plain pinhole.

    No distortion term, deliberately and consistently: the same K was handed to
    FoundationPose, and on this camera the pinhole model is within 2.08 px of
    the real one across the whole frame (measured). Applying distortion here
    while feeding a pinhole K to the estimator would be the inconsistent
    choice, not the accurate one.

    Points behind the camera are dropped rather than projected to a mirrored
    position somewhere on screen.
    """
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    z = p[:, 2]
    out = []
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    for i in range(p.shape[0]):
        if z[i] <= 1e-6:
            return ()                  # any vertex behind the lens: draw none
        out.append((float(fx * p[i, 0] / z[i] + cx),
                    float(fy * p[i, 1] / z[i] + cy)))
    return tuple(out)


def _finite(v) -> float | None:
    """NaN is 'no score', and the panel must show that as '--', not as 0.0."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None
