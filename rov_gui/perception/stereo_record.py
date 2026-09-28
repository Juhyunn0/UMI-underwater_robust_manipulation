#!/usr/bin/env python3
"""
stereo_record.py — the RAW mono pairs, so a run's depth can be made again.

Why this exists
---------------
``--record-depth`` writes FoundationStereo's OUTPUT (the 224 observation and
the uint16 millimetre map). That answers "what did the policy see", and it
cannot answer "what would it have seen at iters 16 instead of 8" — because the
output is not the input. Until this file existed a run was welded to the
FoundationStereo settings it flew with: the network's INPUT, the rectifiable
left/right mono pair, was never persisted anywhere
(``rov_gui/backends/hardware.py`` ``_tap_fstereo`` copies it into a
one-slot mailbox and the next pair overwrites it).

The land side has had this since the first collection: ``oakd_record.py:524``
keeps the lossless mono pair and runs no StereoDepth node at all, precisely so
depth can be regenerated offline with another rectification or another model
(``UMI_Underwater/replay_foundation_stereo.py``). This gives the water side the
same property.

With ``<run>/stereo/`` on disk, an offline pass can:

* re-run FoundationStereo at other ``iters`` / ``scale`` / checkpoint and
  compare against the depth that actually flew (which ``policy_obs/`` holds),
* re-rectify with a NEW calibration — the underwater re-calibration this
  project still owes itself — without re-diving,
* reproduce the exact numbers of a run whose GPU is no longer available.

What is written, and what it is
-------------------------------
Into ``<run>/stereo/``::

    index.csv     seq, t_capture, frame_seq, left_png, right_png
    meta.json     the rig (rectification geometry + provenance), the mono
                  geometry, the FoundationStereo settings the run FLEW with,
                  and the counters below
    left/000123.png    RAW mono, exactly as the camera delivered it
    right/000123.png   its partner from the SAME device exposure

RAW, not rectified, and that is the point: rectification is a function of the
calibration, and baking the current one in would destroy the ability to apply a
better one later. The rig that WAS used is in meta.json, so the flown depth is
still exactly reproducible.

Lossless PNG, for the same reason the obs is: a lossy codec would make the
regenerated depth a function of the codec, and the comparison it exists for
would measure the compression rather than the settings.

``frame_seq`` is the DEVICE sequence number the two images shared. The producer
only pairs frames that report the same one — left and right cross XLink as
separate streams and are NOT delivered in lockstep — so this column is what
proves, offline, that a stored pair is one exposure and not two.

Size — the reason for a budget
------------------------------
A 640x400 mono pair is **296 KiB** on disk as lossless PNG at compression
level 1 [측정 2026-09-06: 10 real pairs from ``~/Desktop/data collection/
videos/0/{left,right}.mp4`` written through THIS recorder, total / 10; a single
``imencode`` of frame 150 gave 266 KiB at level 1, 257 at level 3, 243 at level
6 — the content is land, and underwater scenes with less texture should
compress better]. At the measured 7.26 Hz FoundationStereo rate [측정:
rov_gui/tools/fstereo_bench_out/policy_bench_20260902_141224.json] that is
**~2.1 MB/s, ~0.13 GB/min**, so the 6000-pair default is about **1.7 GB and
~14 minutes**. Raise it with ``--record-stereo-max`` when the dive is longer
and the disk is bigger.

``--record-stereo-every SECONDS`` samples instead, and costs something specific
that the help repeats: the policy's observation is built from a PAIR of
consecutive depth frames, so a sampled recording can no longer reconstruct the
observations the policy actually consumed — only individual frames. Sample for
a survey, keep every pair for anything that will be compared with
``policy_obs/``.

Safety
------
Every producer-side rule is :class:`~.frame_record.FrameRecorder`'s, shared with
the depth recorder and covered by ``test_depth_record.py``. The one thing this
class adds on the hot path is the interval check, which is two floats and a
comparison before the copy — deliberately before, so a sampled run does not pay
for the arrays it is about to drop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .frame_record import CLOSE_TIMEOUT_S, QUEUE_DEPTH, FrameRecorder  # noqa: F401

#: Pair budget. 6000 x 296 KiB ~ 1.7 GB, ~14 min at the measured 7.26 Hz
#: [측정: see the module docstring for both numbers]. Chosen so a flag left on
#: cannot fill a disk mid-dive; reaching it is not an error.
DEFAULT_MAX_PAIRS = 6000


class StereoRecorder(FrameRecorder):
    """Persist raw (left, right) mono pairs from the learned-depth worker.

    Usage::

        rec = StereoRecorder(lambda: runstore.run_dir(base), every_s=0.0)
        rec.start(rig_describe=rig.describe(), fstereo=session.describe())
        rec.submit(t_capture, left, right, frame_seq)     # returns immediately
        rec.close()
    """

    LABEL = "stereo recorder"
    SUBDIR = "stereo"
    SUBFOLDERS = ("left", "right")
    CSV_HEADER = ("seq", "t_capture", "frame_seq", "left_png", "right_png")
    THREAD_NAME = "stereo-rec"
    BUDGET_FLAG = "--record-stereo-max"
    META_SCHEMA = "rov_gui/stereo_pair_recording/1"
    WHY_UNFED = "armed but never fed — no mono pair reached the worker"

    def __init__(self, run_dir_fn: Callable[[], Path], *,
                 every_s: float = 0.0,
                 max_frames: int = DEFAULT_MAX_PAIRS,
                 on_log=None):
        super().__init__(run_dir_fn, max_frames=max_frames, on_log=on_log)
        #: 0 = every pair the worker took. Anything else is a sampling
        #: interval, and `dropped_interval` counts what it skipped so a sparse
        #: recording is never mistaken for a starved feed.
        self.every_s = max(0.0, float(every_s))
        self.counters["dropped_interval"] = 0
        self._last_kept = 0.0
        self._shape = None
        self._dtype = ""

    def _arm_note(self) -> str:
        how = ("every pair" if self.every_s <= 0.0
               else f"one pair every {self.every_s:g} s")
        gb = self.max_frames * 296 / 1e6
        size = f"{gb:.1f} GB" if gb >= 1.0 else f"{gb * 1000:.0f} MB"
        return (f"stereo recorder: armed ({how}, max {self.max_frames} pairs "
                f"~ {size} at 640x400 [유도 from the measured 296 KiB pair]). "
                f"RAW mono, lossless, so this run's depth can be recomputed "
                f"offline at other FoundationStereo settings. The run folder "
                f"is created by the writer thread on the first pair; pairs are "
                f"DROPPED, never queued indefinitely, if the disk cannot keep "
                f"up.")

    # ----------------------------------------------------------------- submit
    def submit(self, t_capture: float, left: np.ndarray, right: np.ndarray,
               frame_seq: int = 0) -> None:
        """Offer one pair. Returns immediately; never raises, never blocks.

        Called from the learned-depth worker's tick, which is already carrying
        a ~46 ms inference [측정: rov_gui/README fstereo bench], so the interval
        gate is checked BEFORE the copy: a sampled run should not pay two
        640x400 memcpys for a pair it is about to throw away.
        """
        if self.every_s > 0.0 and self.started and not self._closed:
            t = float(t_capture)
            # `<` not `<=`, and against the last KEPT stamp rather than a
            # deadline that accumulates: a feed that stalls then resumes must
            # not fire a burst of catch-up frames into a bounded queue.
            if self._last_kept and (t - self._last_kept) < self.every_s:
                self.counters["offered"] += 1
                self.counters["dropped_interval"] += 1
                return
            self._last_kept = t
        if self._shape is None and getattr(left, "shape", None) is not None:
            self._shape = tuple(int(v) for v in left.shape)
            self._dtype = str(getattr(left, "dtype", ""))
        self._enqueue(lambda seq: (
            seq, float(t_capture), int(frame_seq),
            np.array(left, copy=True), np.array(right, copy=True)))

    # ------------------------------------------------------------------- meta
    def _describe_extra(self) -> dict:
        return {"every_s": self.every_s, "mono_shape": self._shape,
                "mono_dtype": self._dtype}

    def _meta_payload(self, drained: bool = True) -> dict:
        return {
            "written_by": "rov_gui/perception/stereo_record.py",
            "images": ("RAW mono, exactly as the camera delivered them — NOT "
                       "rectified. Rectification is a function of the "
                       "calibration, and the rig that was used is below, so "
                       "the flown depth is reproducible and a future "
                       "calibration can still be applied to the same pixels."),
            "mono_shape": self._shape,
            "mono_dtype": self._dtype,
            "every_s": self.every_s,
            "pairs_are_one_exposure": ("index.csv frame_seq is the device "
                                       "sequence number both images reported; "
                                       "the producer only pairs equal ones"),
            "regenerate_with": ("FoundationStereo on the rig below: rectify "
                                "left/right with rig.map_left / map_right, "
                                "match, then depth = k_mm_px / disparity — the "
                                "same chain as rov_gui/perception/fstereo.py "
                                "infer(). The settings this run FLEW with are "
                                "in `fstereo` so a comparison can name both."),
            "rig": self._meta_extra.get("rig") or {},
            "fstereo": self._meta_extra.get("fstereo") or {},
            "extra": self._meta_extra.get("extra") or {},
        }

    def start(self, *, rig_describe: dict | None = None,
              fstereo: dict | None = None, extra: dict | None = None) -> bool:
        return self._start(rig=rig_describe or {}, fstereo=fstereo or {},
                           extra=extra or {})

    # ----------------------------------------------------------------- writer
    def _write_item(self, cv2, png, item) -> tuple:
        seq, t, frame_seq, left, right = item
        left_name = f"left/{seq:06d}.png"
        right_name = f"right/{seq:06d}.png"
        try:
            # Rule 4: imwrite returns False on a full disk rather than raising.
            # BOTH halves must land — half a pair is not a stereo frame, and a
            # row naming a file that does not exist is worse than no row.
            ok = bool(cv2.imwrite(str(self.dir / left_name), left, png))
            if ok:
                ok = bool(cv2.imwrite(str(self.dir / right_name), right, png))
        except Exception:                                        # noqa: BLE001
            ok = False
        return ok, [seq, f"{t:.6f}", frame_seq, left_name, right_name]


__all__ = ["StereoRecorder", "DEFAULT_MAX_PAIRS", "QUEUE_DEPTH",
           "CLOSE_TIMEOUT_S"]
