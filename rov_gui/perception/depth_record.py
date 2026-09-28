#!/usr/bin/env python3
"""
depth_record.py — write the underwater depth observation to disk, off the hot path.

Why this exists
---------------
Until now a live run persisted no depth at all. Not the uint16 millimetre maps,
not the 224x224 observation the policy consumed, and not the raw stereo pairs it
was all computed from — so an underwater depth map could not be inspected after
the fact and could not even be regenerated offline
(``rov_gui/backends/hardware.py:3244-3352``). The 2026-09-03 policy run left
behind ``controller.json``, ``mission_log.txt``, ``ui_20260903_183555.mp4`` and
``ui_20260903_183555.json`` — and no depth
[측정: data/20260903/0903_183555/].

The only visual trace was that mp4, and it is a recording of the *displayed*
picture: already colourised by a per-frame auto range and a per-frame 64-knot
histogram equalisation, then nearest-downscaled to the widget. Two frames of it
with the same colour do not have the same distance. It is not depth.

(Since 2026-09-06 the panel's default palette is the fixed shared rule, so a
colour in a NEW ui_*.mp4 does map back to a distance — but it is still the
recording of a resized, JPEG-ish picture of depth, not the millimetres. This
module is still the only thing that writes depth itself.)

That gap is what made the 2026-09-04 debugging request — compare the land depth
and the water depth — unanswerable. This module closes it.

What it writes
--------------
Into ``<run>/policy_obs/``::

    index.csv     one row per recorded tick: seq, t_capture, obs_png[, depth_png]
    meta.json     the obs recipe (z_near/z_far, crop, resolution, grid), the
                  builder's full describe(), the counters below, and under
                  ``extra`` the checkpoint that consumed the frames:
                  ``extra.ckpt`` is the LAST one, and ``extra.ckpt_swaps``
                  lists every mid-recording swap with the ``next_seq`` from
                  which frames belong to the new network (2026-09-11 review;
                  see :meth:`DepthRecorder.note_ckpt_swap`)
    obs/000123.png    224x224x3 uint8 — the tensor the policy consumed, verbatim
    depth/000123.png  640x400 uint16 millimetres, 0 = no measurement (optional)

Both are LOSSLESS PNG. That is the whole point: a lossy codec on a depth map
turns "no measurement" into "some distance" along every hole edge, which is the
one distinction the obs's validity channel exists to preserve.

``rov_gui/tools/depth_compare.py`` reads exactly this layout, and renders it on
the same fixed colour rule as the land training obs.

Safety
------
The producer-side rules (no filesystem call on the worker thread, a bounded
queue that DROPS rather than blocks, failures that never end a dive, a failed
write never counted as a success, a bounded shutdown) live with the machinery
that implements them, in ``frame_record.py``. They were found by the 2026-09-06
safety audit of THIS file and are shared, not copied, with the stereo-pair
recorder — read that module before changing either.

Everything is off unless ``--record-depth`` is given.
"""

from __future__ import annotations

import csv
import json
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .frame_record import CLOSE_TIMEOUT_S, QUEUE_DEPTH, FrameRecorder  # noqa: F401

#: Frame budget. At the measured rate the policy feeds this recorder
#: [측정 2026-09-03: policy.worker.depth_hz 11.09, ~2 obs per inference at
#: period_s 0.5], and with an obs PNG of order 30 kB plus a depth PNG of order
#: 300 kB [예측: not measured on real frames], 4000 frames is very roughly 2 GB
#: and ~16 min. Both figures want replacing with one measured run's
#: ``du -sh policy_obs/``.
DEFAULT_MAX_FRAMES = 4000


class DepthRecorder(FrameRecorder):
    """Persist (obs, depth) pairs from the policy worker without blocking it.

    Usage::

        rec = DepthRecorder(lambda: runstore.run_dir(base), save_depth=True)
        rec.start(builder_describe=builder.describe())   # first frame
        rec.submit(t_capture, obs, depth_mm)             # returns immediately
        rec.close()                                      # teardown

    The plumbing and the five safety rules are :class:`~.frame_record.
    FrameRecorder`'s; this class only says what a frame is.
    """

    LABEL = "depth recorder"
    SUBDIR = "policy_obs"
    SUBFOLDERS = ("obs",)
    CSV_HEADER = ("seq", "t_capture", "obs_png", "depth_png", "rgb_jpg",
                  "rgb_dt_ms")
    THREAD_NAME = "depth-rec"
    BUDGET_FLAG = "--record-depth-max"
    META_SCHEMA = "rov_gui/policy_obs_recording/1"
    WHY_UNFED = "armed but never fed — no inference ran"

    def __init__(self, run_dir_fn: Callable[[], Path], *,
                 save_depth: bool = True, save_rgb: bool = True,
                 rgb_quality: int = 85,
                 max_frames: int = DEFAULT_MAX_FRAMES,
                 on_log=None):
        super().__init__(run_dir_fn, max_frames=max_frames, on_log=on_log)
        self.save_depth = bool(save_depth)
        #: The COLOUR frame that was current when the observation was built.
        #: JPEG, not PNG, and that is a deliberate asymmetry: the obs and the
        #: millimetre map are MEASUREMENTS and a lossy codec would move them,
        #: while this is a picture for the eye — the policy never sees it. A
        #: lossless colour frame is ~1 MB against ~120 kB [예측: not measured
        #: on this camera], which at 2854 frames is the difference between a
        #: recording and a filled disk.
        self.save_rgb = bool(save_rgb)
        self.rgb_quality = int(rgb_quality)
        subs = ["obs"]
        if self.save_depth:
            subs.append("depth")
        if self.save_rgb:
            subs.append("rgb")
        self.SUBFOLDERS = tuple(subs)

    # ------------------------------------------------------------------ start
    def start(self, *, builder_describe: dict | None = None,
              extra: dict | None = None) -> bool:
        return self._start(builder=builder_describe or {}, extra=extra or {})

    def _arm_note(self) -> str:
        return (f"depth recorder: armed (max {self.max_frames} frames, "
                f"depth map {'on' if self.save_depth else 'off'}, "
                f"colour {'on' if self.save_rgb else 'off'}). The run "
                f"folder is created by the writer thread on the first frame. "
                f"Frames are DROPPED, never queued indefinitely, if the disk "
                f"cannot keep up; the drop and error counts are in meta.json "
                f"and in the run record.")

    # ----------------------------------------------------------------- submit
    def submit(self, t_capture: float, obs: np.ndarray,
               depth_mm: np.ndarray | None = None,
               color=None, color_t: float = 0.0) -> None:
        """Offer one frame. Returns immediately; never raises, never blocks.

        ``color`` is the newest COLOUR frame and ``color_t`` its own capture
        stamp — kept apart from ``t_capture`` because they are different
        sensors at different rates, and the index records the gap rather than
        implying the two are simultaneous.
        """
        keep_depth = depth_mm is not None and self.save_depth
        keep_rgb = color is not None and self.save_rgb
        dt_ms = ((float(color_t) - float(t_capture)) * 1e3) if keep_rgb else None
        self._enqueue(lambda seq: (
            seq, float(t_capture), np.array(obs, copy=True),
            np.array(depth_mm, copy=True) if keep_depth else None,
            np.array(color, copy=True) if keep_rgb else None, dt_ms))

    # ------------------------------------------------- mid-recording notes
    @property
    def next_seq(self) -> int:
        """The ``seq`` the NEXT accepted frame will get (the producer's own
        counter, so it is exact on the producer thread): every frame with
        ``seq >= next_seq`` was offered after this moment. Counted frames
        that were dropped (full queue, budget) never got a seq, so the
        boundary holds in index.csv too."""
        return int(self._seq)

    def note_extra(self, key: str, value) -> bool:
        """Annotate meta.json's ``extra`` while the recording is running.
        PRODUCER THREAD; touches NO file.

        Why (review 2026-09-11): ``start()`` captured ``extra`` ONCE, at the
        first frame, and the panel checkpoint picker (PolicyWorker.set_ckpt)
        can swap the network while this recorder stays armed — the swap is
        allowed between engagements, and runstore JOINS the same run folder
        within 90 s — so meta.json named the earlier checkpoint for every
        frame the later one consumed, and index.csv has no ckpt column to
        contradict it.

        No filesystem call here (frame_record rule 1): meta.json is written
        only at ``close()``, and ``_meta_extra`` is read only there, on the
        same thread as this call, so a plain dict write is race-free
        (``describe()`` never reads it). Returns False when there is nothing
        to annotate — before ``start()`` (its own ``extra`` will carry the
        current values) or once closed/disabled (no meta.json will be
        written for what was not recorded).
        """
        if not self.started or self._closed:
            return False
        self._meta_extra.setdefault("extra", {})[str(key)] = value
        return True

    def note_ckpt_swap(self, old: str, new: str, **more) -> dict | None:
        """Record a checkpoint swap in meta.json: ``extra.ckpt`` follows
        ``new`` and ``extra.ckpt_swaps`` gains one entry::

            {"next_seq": <seq of the first frame under the new network>,
             "t_wall": <time.time()>, "from": old, "to": new,
             "obs_dt_s_before": ..., "ckpt_sha1_from": ..., **more}

        Frames are recorded only while a network infers, so every frame with
        ``seq >= next_seq`` was consumed by ``new``. A reader of meta.json
        ALONE can therefore attribute each frame; before this the only
        recovery was cross-referencing ``t_capture`` against each
        engagement's ``mpc_*.meta.json``. Returns the entry, or None when
        nothing is armed (same rule as :meth:`note_extra`).
        """
        if not self.started or self._closed:
            return None
        extra = self._meta_extra.setdefault("extra", {})
        entry = {"next_seq": self.next_seq, "t_wall": time.time(),
                 "from": str(old), "to": str(new),
                 "obs_dt_s_before": extra.get("obs_dt_s"),
                 "ckpt_sha1_from": extra.get("ckpt_sha1")}
        entry.update(more)
        swaps = list(extra.get("ckpt_swaps") or [])
        swaps.append(entry)
        extra["ckpt_swaps"] = swaps
        extra["ckpt"] = str(new)
        return entry

    # ------------------------------------------------------------------- meta
    def _describe_extra(self) -> dict:
        return {"save_depth_mm": self.save_depth, "save_rgb": self.save_rgb,
                "rgb_quality": self.rgb_quality}

    def _meta_payload(self, drained: bool = True) -> dict:
        b = self._meta_extra.get("builder") or {}
        return {
            "written_by": "rov_gui/perception/depth_record.py",
            "obs_png": "224x224x3 uint8, channels (inverse_depth, validity, "
                       "inverse_depth) — the tensor the policy consumed, "
                       "written verbatim and losslessly",
            # PRE-WARP, and say so. The array submitted is the mailbox frame on
            # the PRODUCER's grid (rect_left for FoundationStereo,
            # color_aligned for the device), not the 640x400 target grid the
            # obs is built on — DepthObsBuilder.warp is what crosses between
            # them. Under --fstereo the two sizes coincide, which is exactly
            # how a wrong label stays invisible; the recoverable geometry is in
            # `builder.grid`.
            "depth_png": (f"uint16 millimetres, 0 = no measurement, on the "
                          f"PRODUCER grid ({b.get('grid_kind') or 'unknown'}) "
                          f"BEFORE the warp to the obs target grid — see "
                          f"builder.grid for its geometry"
                          if self.save_depth else None),
            "rgb_jpg": (f"the colour frame current when the observation was "
                        f"built, JPEG q{self.rgb_quality} — a picture, not a "
                        f"measurement (the policy never sees colour). "
                        f"`rgb_dt_ms` is its own capture stamp minus the "
                        f"depth frame's: colour runs ~30 Hz and the recorded "
                        f"depth ~5 Hz, so they are near-simultaneous, never "
                        f"simultaneous."
                        if self.save_rgb else None),
            "read_with": "python -m rov_gui.tools.depth_compare water --run <run>",
            "obs": {"z_near_m": b.get("z_near_m"), "z_far_m": b.get("z_far_m"),
                    "out_res": b.get("out_res"), "crop": b.get("crop"),
                    "grid_kind": b.get("grid_kind"),
                    "coverage": b.get("coverage")},
            "builder": b,
            "extra": self._meta_extra.get("extra") or {},
        }

    # ----------------------------------------------------------------- writer
    def _write_item(self, cv2, png, item) -> tuple:
        seq, t, obs, depth, color, dt_ms = item
        obs_name = f"obs/{seq:06d}.png"
        depth_name = f"depth/{seq:06d}.png" if depth is not None else ""
        rgb_name = f"rgb/{seq:06d}.jpg" if color is not None else ""
        try:
            # imwrite RETURNS False on a full disk; it does not raise.
            # Counting an unwritten frame as written is the one failure
            # this recorder must not have (rule 4).
            ok = bool(cv2.imwrite(str(self.dir / obs_name), obs, png))
            if ok and depth is not None:
                ok = bool(cv2.imwrite(str(self.dir / depth_name),
                                      np.asarray(depth, dtype=np.uint16), png))
            if ok and color is not None:
                ok = bool(cv2.imwrite(
                    str(self.dir / rgb_name), color,
                    [cv2.IMWRITE_JPEG_QUALITY, self.rgb_quality]))
        except Exception:                                        # noqa: BLE001
            ok = False
        return ok, [seq, f"{t:.6f}", obs_name, depth_name, rgb_name,
                    "" if dt_ms is None else f"{dt_ms:.1f}"]


__all__ = ["DepthRecorder", "QUEUE_DEPTH", "DEFAULT_MAX_FRAMES",
           "CLOSE_TIMEOUT_S"]
