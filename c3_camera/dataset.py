#!/usr/bin/env python3
"""
dataset.py — write a SLAM-ready dataset in TUM RGB-D layout.

The timeline (the part that decides whether the dataset is usable)
------------------------------------------------------------------
Measured on this machine: `dai.Clock.now()` and `time.monotonic()` agree to within
8 microseconds — they are both CLOCK_MONOTONIC. That means all three sources are
*already* on one clock and need no cross-calibration:

    camera frame   ImgFrame.getTimestamp()   -> CLOCK_MONOTONIC (capture time,
                                                kept in the host domain by
                                                DepthAI's timesync service)
    camera IMU     IMUReport.getTimestamp()  -> CLOCK_MONOTONIC, same domain
    MAVLink        time.monotonic() on receipt -> CLOCK_MONOTONIC

Everything written here is converted to **Unix epoch seconds** using a single
offset captured at session start, so the files look like ordinary TUM data and are
comparable to wall-clock events. `frames.csv` also keeps the raw monotonic values,
so nothing is lost.

One honest asymmetry, recorded in metadata: camera and camera-IMU timestamps are
*capture* times, while MAVLink timestamps are *arrival* times and therefore lag the
true measurement by the link and routing delay (a few ms over this Ethernet path).
Each MAVLink record also carries ArduSub's own `time_boot_ms`, so a consumer who
needs better than a few ms can regress that against `t_unix` and recover the
autopilot clock properly.

depth_scale
-----------
Written as **1000.0**, meaning `metres = pixel_value / 1000.0`, because DepthAI
emits uint16 millimetres.

TUM's own datasets use 5000, and that number gets copied around as if it were the
convention — it is not, it is specific to how TUM scaled their Kinect data. Writing
5000 for millimetre data would make every depth come out 5x too small, and it would
look plausible rather than broken. So metadata states the number, the direction of
the operation, and the per-tool parameter name.
"""

from __future__ import annotations

import csv
import json
import math
import queue
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# metres = pixel_value / DEPTH_SCALE  (DepthAI depth is uint16 millimetres)
DEPTH_SCALE = 1000.0

FRAME_FIELDS = (
    "idx", "t_unix", "t_monotonic",
    "rgb_file", "depth_file", "left_file", "right_file",
    "rgb_seq", "depth_seq", "left_seq", "right_seq",
    "rgb_t_device", "depth_t_device", "left_t_device", "right_t_device",
    "rgb_latency_ms", "depth_latency_ms", "skew_ms", "exposure_ms",
    "rgb_encoding", "rgb_frame_type",
)

RGB_VIDEO_FIELDS = (
    "frame_idx", "t_unix", "t_monotonic", "rgb_seq", "rgb_file",
    "byte_offset", "byte_size", "frame_type",
)


@dataclass
class WriterStats:
    written: int = 0
    dropped: int = 0
    waited: int = 0
    images: int = 0
    bytes_written: int = 0
    encode_ms_total: float = 0.0
    started_at: float = field(default_factory=time.monotonic)

    @property
    def mb(self) -> float:
        return self.bytes_written / 1e6

    @property
    def mb_per_s(self) -> float:
        el = time.monotonic() - self.started_at
        return self.mb / el if el > 0 else 0.0

    @property
    def drop_rate(self) -> float:
        total = self.written + self.dropped
        return self.dropped / total if total else 0.0


@dataclass
class _Job:
    idx: int
    images: list           # [(relative_path, ndarray|bytes, image|depth|raw)]
    row: dict


class DatasetWriter:
    """TUM RGB-D writer with a background thread pool.

    Lossless PNG encoding is CPU-bound (several ms per image, four images per
    frame), so it runs on worker threads — cv2.imwrite releases the GIL, so
    threads genuinely parallelise here. Capture must never wait on disk or on
    zlib.
    """

    MODES = ("review", "research")

    def __init__(self, out_dir: Path, mode: str = "research",
                 threads: int = 3, queue_size: int = 12,
                 block_ms: float = 200.0,
                 png_compression: int = 1,
                 mp4_fps: float | None = None,
                 write_mp4: bool = True,
                 color_encode: str = "none",
                 verbose: bool = True):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")

        self.dir = Path(out_dir)
        self.mode = mode
        self.verbose = verbose
        self.stats = WriterStats()
        # PNG level 1 is a deliberate choice: still lossless, but ~4x faster to
        # encode than the default 3 for maybe 10% more bytes. The link caps us
        # near 11 MB/s while the NVMe here does far more, so spending CPU to save
        # disk would be optimising the wrong end.
        self.png_params = [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)]
        self.block_s = max(0.0, block_ms / 1000.0)
        self.write_mp4 = write_mp4
        self.mp4_fps = mp4_fps
        self.color_encode = color_encode
        self._interframe = color_encode in ("h264", "h265")

        self._idx = 0
        self._closed = False
        self._last_primary_seq: int | None = None
        self._last_name: str | None = None
        # Reorder buffer, so the index files stay in frame order even though
        # several encoder threads complete out of order. See _queue_index_row.
        self._pending_rows: dict[int, dict | None] = {}
        self._next_emit = 1
        self._mp4: cv2.VideoWriter | None = None
        self._mp4_size: tuple[int, int] | None = None
        self._video_fh = None
        self._video_frames = 0
        self._encoded_started = False
        self._encoded_seqs: set[int] = set()
        self._codec_header_written = False
        self._t_offset = time.time() - time.monotonic()   # monotonic -> unix
        self._write_lock = threading.Lock()

        self.dir.mkdir(parents=True, exist_ok=True)
        self._subdirs: dict[str, Path] = {}
        if mode == "research":
            for name in ("rgb", "depth", "left", "right"):
                d = self.dir / name
                d.mkdir(exist_ok=True)
                self._subdirs[name] = d

        # index files
        self._files: dict[str, object] = {}
        self._csv: dict[str, csv.DictWriter] = {}
        self._lists: dict[str, object] = {}
        if mode == "research":
            self._open_tum_lists()
            self._open_csv("frames", FRAME_FIELDS)
        if self._interframe:
            self._open_csv("rgb_video", RGB_VIDEO_FIELDS)
            self._video_fh = (self.dir / f"rgb.{color_encode}").open("wb")

        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._threads = [
            threading.Thread(target=self._run, name=f"ds-writer{i}", daemon=True)
            for i in range(max(1, threads))
        ]
        for t in self._threads:
            t.start()

    # --------------------------------------------------------------- plumbing
    def to_unix(self, t_monotonic: float) -> float:
        return t_monotonic + self._t_offset

    def _open_csv(self, name: str, fields) -> csv.DictWriter:
        fh = (self.dir / f"{name}.csv").open("w", newline="")
        w = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        self._files[name] = fh
        self._csv[name] = w
        return w

    def _open_tum_lists(self) -> None:
        stamp = datetime.now().isoformat(timespec="seconds")
        for name, what in (("rgb", "color images"), ("depth", "depth maps")):
            fh = (self.dir / f"{name}.txt").open("w")
            fh.write(f"# {what}\n")
            fh.write(f"# recorded {stamp} from a MarineSitu C3 (OAK-D-W-POE)\n")
            fh.write("# timestamp filename\n")
            self._lists[name] = fh
        fh = (self.dir / "associations.txt").open("w")
        fh.write("# rgb_timestamp rgb_file depth_timestamp depth_file\n")
        self._lists["assoc"] = fh

    # ---------------------------------------------------------------- submit
    def submit(self, bundle, imu_hint: float | None = None) -> bool:
        """Queue one frame bundle. Returns False if it had to be dropped.

        A "dataset frame" is defined by the COLOUR frame advancing, because that is
        what TUM RGB-D is indexed by and what names the files. This matters: the
        source returns a bundle whenever *any* stream has something new, so at 8 fps
        across four streams the loop sees ~26 bundles/s carrying the same colour
        frame repeatedly. Writing all of them produced duplicate filenames that
        overwrote each other and an index whose timestamps did not increase — so
        advancement is checked by sequence number, not by arrival.
        """
        if self._closed:
            return False

        c, d = bundle.color, bundle.depth
        l, r = bundle.left, bundle.right
        present = [f for f in (c, d, l, r) if f is not None]
        if not present:
            return True

        # Skip bundles that carry no new primary frame.
        primary = c if c is not None else present[0]
        if primary.seq == self._last_primary_seq:
            return True

        # An Annex-B stream started on a P frame is not independently decodable:
        # its SPS/PPS and reference image were discarded before the operator
        # pressed record. Wait for the next I frame, which the pipeline requests
        # once per second by default, and start RGB-D together there.
        if self._interframe:
            if c is not None:
                # Direct DatasetWriter callers may not use the separate encoded
                # drain. Sequence dedup makes this harmless when c3_collect did.
                self.encoded_rgb_frames([c])
            if not self._encoded_started:
                self._last_primary_seq = primary.seq
                return True

        self._last_primary_seq = primary.seq

        self._idx += 1
        idx = self._idx
        t_mono = c.t_device if c is not None else present[0].t_device
        t_unix = self.to_unix(t_mono)
        name = f"{t_unix:.6f}"
        # Belt and braces: filenames are timestamps, so two frames landing in the
        # same microsecond would overwrite each other silently. Nudge rather than
        # clobber, and keep the index consistent with the file actually written.
        if name == self._last_name:
            t_unix += 1e-6
            name = f"{t_unix:.6f}"
        self._last_name = name

        images: list = []
        row = {
            "idx": idx,
            "t_unix": f"{t_unix:.6f}",
            "t_monotonic": f"{t_mono:.6f}",
            "skew_ms": (f"{bundle.skew_ms:.3f}"
                        if bundle.skew_ms == bundle.skew_ms else ""),
        }
        for key, fr, sub, is_depth in (("rgb", c, "rgb", False),
                                       ("depth", d, "depth", True),
                                       ("left", l, "left", False),
                                       ("right", r, "right", False)):
            if fr is None:
                row[f"{key}_file"] = ""
                continue
            ext = "jpg" if key == "rgb" and fr.encoding == "mjpeg" else "png"
            rel = f"{sub}/{name}.{ext}"
            row[f"{key}_file"] = rel if self.mode == "research" else ""
            row[f"{key}_seq"] = fr.seq
            row[f"{key}_t_device"] = f"{self.to_unix(fr.t_device):.6f}"
            if key in ("rgb", "depth"):
                row[f"{key}_latency_ms"] = f"{fr.latency_ms:.2f}"
            if self.mode == "research":
                if key == "rgb" and self._interframe:
                    # The future PNG name is already in the index. The extractor
                    # creates it after capture from rgb.h264/rgb.h265.
                    pass
                elif key == "rgb" and fr.encoding == "mjpeg" and fr.raw is not None:
                    images.append((rel, fr.raw, "raw"))
                elif fr.image is not None:
                    images.append((rel, fr.image, "depth" if is_depth else "image"))
        if c is not None:
            row["rgb_encoding"] = c.encoding
            row["rgb_frame_type"] = c.frame_type
        if c is not None and c.exposure_ms == c.exposure_ms:
            row["exposure_ms"] = f"{c.exposure_ms:.3f}"

        # mp4 preview is written on the capture thread: one already-decoded BGR
        # frame into a video writer is sub-millisecond, and keeping it here means
        # the preview cannot end up out of order relative to the PNGs.
        if self.write_mp4 and c is not None and c.image is not None:
            self._append_mp4(c.image)

        if self.mode == "review":
            # Review mode keeps no images and no index; the mp4 plus the
            # telemetry CSVs are the whole point of it.
            self.stats.written += 1
            return True

        job = _Job(idx=idx, images=images, row=row)
        try:
            self._q.put_nowait(job)
            return True
        except queue.Full:
            pass
        try:
            if self.block_s > 0:
                self._q.put(job, timeout=self.block_s)
                self.stats.waited += 1
                return True
        except queue.Full:
            pass
        self.stats.dropped += 1
        self._idx -= 1
        return False

    def encoded_rgb_frames(self, frames) -> int:
        """Preserve every H.264/H.265 packet independently of RGB-D pairing."""
        if not self._interframe or self._closed:
            return 0
        written = 0
        for frame in frames:
            if frame.raw is None or frame.seq in self._encoded_seqs:
                continue
            if not self._encoded_started:
                if frame.frame_type != "I":
                    continue
                self._encoded_started = True
                if self.verbose:
                    print("  encoded RGB: first keyframe received; "
                          "dataset timeline starts")
            if not self._codec_header_written and frame.codec_header:
                # The operator may press record long after pipeline start. Cache
                # and prepend VPS/SPS/PPS so that a later keyframe still forms a
                # standalone Annex-B stream. Parameter NALs decode to no image.
                self._video_fh.write(frame.codec_header)
                with self._write_lock:
                    self.stats.bytes_written += len(frame.codec_header)
                self._codec_header_written = True
            t_mono = frame.t_device
            t_unix = self.to_unix(t_mono)
            rgb_file = f"rgb/{t_unix:.6f}.png" if self.mode == "research" else ""
            self._append_encoded(frame, rgb_file, t_unix, t_mono)
            self._encoded_seqs.add(frame.seq)
            written += 1
        return written

    def _append_encoded(self, frame, rgb_file: str,
                        t_unix: float, t_monotonic: float) -> None:
        """Append one H.264/H.265 packet and its capture-time sidecar row."""
        if self._video_fh is None or frame.raw is None:
            raise RuntimeError("encoded RGB stream is not open")
        offset = self._video_fh.tell()
        self._video_fh.write(frame.raw)
        self._video_frames += 1
        with self._write_lock:
            self.stats.bytes_written += len(frame.raw)
            self._csv["rgb_video"].writerow({
                "frame_idx": self._video_frames,
                "t_unix": f"{t_unix:.6f}",
                "t_monotonic": f"{t_monotonic:.6f}",
                "rgb_seq": frame.seq,
                "rgb_file": rgb_file,
                "byte_offset": offset,
                "byte_size": len(frame.raw),
                "frame_type": frame.frame_type,
            })

    # ---------------------------------------------------------------- writers
    def _run(self) -> None:
        while True:
            job = self._q.get()
            if job is None:
                self._q.task_done()
                return
            t0 = time.monotonic()
            try:
                nbytes = 0
                for rel, data, kind in job.images:
                    path = self.dir / rel
                    if kind == "raw":
                        path.write_bytes(data)
                        ok = True
                    elif kind == "depth":
                        if data.dtype != np.uint16:
                            raise ValueError(
                                f"depth must be uint16 millimetres, got {data.dtype}")
                        ok = cv2.imwrite(str(path), data, self.png_params)
                    else:
                        ok = cv2.imwrite(str(path), data, self.png_params)
                    if not ok:
                        raise IOError(f"cv2.imwrite failed for {path}")
                    nbytes += path.stat().st_size
                    self.stats.images += 1

                with self._write_lock:
                    self.stats.written += 1
                    self.stats.bytes_written += nbytes
                    self._queue_index_row(job.idx, job.row)
            except Exception as e:  # noqa: BLE001 - one bad frame must not stop the dive
                if self.verbose:
                    print(f"  writer: frame {job.idx} failed: "
                          f"{type(e).__name__}: {e}")
                # Tombstone, so the in-order emitter is not blocked forever
                # waiting for an index that will never arrive.
                with self._write_lock:
                    self._queue_index_row(job.idx, None)
            finally:
                self.stats.encode_ms_total += (time.monotonic() - t0) * 1000.0
                self._q.task_done()

    def _queue_index_row(self, idx: int, row: dict | None) -> None:
        """Emit index lines strictly in frame order. Call with _write_lock held.

        Several encoder threads run concurrently, so frame 3 can finish before
        frame 2. Writing each index line as its frame completes therefore produces
        a TUM index whose timestamps do not increase monotonically — which a
        consumer either rejects or, worse, silently mis-associates. So completed
        rows wait in a small reorder buffer until every earlier frame has been
        emitted. A frame that failed to write leaves a None tombstone, so the
        sequence advances past it instead of stalling.
        """
        self._pending_rows[idx] = row
        while self._next_emit in self._pending_rows:
            r = self._pending_rows.pop(self._next_emit)
            self._next_emit += 1
            if r is None:
                continue
            self._csv["frames"].writerow(r)
            ts = r["t_unix"]
            if r.get("rgb_file"):
                self._lists["rgb"].write(f"{ts} {r['rgb_file']}\n")
            if r.get("depth_file"):
                self._lists["depth"].write(f"{ts} {r['depth_file']}\n")
            if r.get("rgb_file") and r.get("depth_file"):
                # TUM associate.py order: rgb first, then depth — which is also
                # the order ORB-SLAM3's rgbd_tum parser expects. Column 3 is the
                # depth frame's own capture time, so the true skew stays visible.
                self._lists["assoc"].write(
                    f"{ts} {r['rgb_file']} "
                    f"{r.get('depth_t_device', ts)} {r['depth_file']}\n")

    def _append_mp4(self, bgr: np.ndarray) -> None:
        if self._mp4 is None:
            h, w = bgr.shape[:2]
            self._mp4_size = (w, h)
            fps = self.mp4_fps or 15.0
            self._mp4 = cv2.VideoWriter(str(self.dir / "rgb.mp4"),
                                        cv2.VideoWriter_fourcc(*"mp4v"),
                                        fps, (w, h))
            if not self._mp4.isOpened():
                if self.verbose:
                    print("  note: could not open rgb.mp4 for writing; "
                          "continuing without the preview video")
                self._mp4 = None
                self.write_mp4 = False
                return
        if self._mp4_size and bgr.shape[1::-1] != self._mp4_size:
            bgr = cv2.resize(bgr, self._mp4_size)
        self._mp4.write(bgr)

    # ------------------------------------------------------ telemetry streams
    def imu_camera_rows(self, samples, fields) -> None:
        if "imu_camera" not in self._csv:
            self._open_csv("imu_camera", ("t_unix",) + tuple(fields))
        w = self._csv["imu_camera"]
        with self._write_lock:
            for s in samples:
                row = s.row()
                row["t_unix"] = f"{self.to_unix(s.t_device):.6f}"
                w.writerow(row)

    def mavlink_rows(self, records: list[dict]) -> None:
        """Route each MAVLink record into the CSV its message belongs to.

        Fields are written in native MAVLink units without conversion; metadata
        documents the units. Converting here would bake in an assumption that a
        consumer cannot undo.
        """
        from .mavlink_log import CONTROL_MESSAGES, IMU_MESSAGES

        for rec in records:
            mtype = rec.get("msg_type", "?")
            if mtype in IMU_MESSAGES:
                group = "imu_rov"
            elif mtype in CONTROL_MESSAGES and mtype != "SERVO_OUTPUT_RAW":
                group = "control"
            else:
                group = "telemetry"
            row = dict(rec)
            row["t_unix"] = f"{self.to_unix(rec['t_host']):.6f}"
            row["t_monotonic"] = f"{rec['t_host']:.6f}"
            row.pop("t_host", None)
            if group not in self._csv:
                # Header is fixed on first sight of each group. Messages arriving
                # later with extra fields would be truncated by extrasaction, so
                # start from a generous union of what ArduSub sends per group.
                self._open_csv(group, ["t_unix", "t_monotonic", "msg_type"]
                               + sorted(k for k in row
                                        if k not in ("t_unix", "t_monotonic",
                                                     "msg_type")))
            with self._write_lock:
                self._csv[group].writerow(row)

    # -------------------------------------------------------------- artefacts
    def write_calibration(self, calib_dict: dict) -> None:
        (self.dir / "calibration.json").write_text(
            json.dumps(calib_dict, indent=2, default=str))

    def write_metadata(self, meta: dict, notes: str = "") -> None:
        (self.dir / "metadata.txt").write_text(_render_metadata(meta, notes))
        (self.dir / "metadata.json").write_text(
            json.dumps(meta, indent=2, default=str))

    def write_orbslam3_yaml(self, intr, depth_baseline_mm: float | None,
                            fps: float, filename: str = "orbslam3_rgbd.yaml") -> None:
        """A ready-to-use ORB-SLAM3 RGB-D config, so the colleague has a starting point.

        Depth is aligned to the colour camera and both are the same size, so the
        RGB intrinsics apply to the depth image too — which is the whole reason
        for aligning to colour rather than to the left mono.
        """
        # ORB-SLAM3's Settings::readRGBD requires Stereo.b and Stereo.ThDepth and
        # calls exit(-1) on a missing required parameter, so neither key may be
        # dropped. The baseline is read from the device when possible; when that
        # read fails, the C3's nominal 75 mm is written and labelled, because a
        # nominal baseline is recoverable and a config that aborts is not.
        if depth_baseline_mm:
            # ORB-SLAM3 Stereo.b is metres; ThDepth is in units of b*f.
            bf = (f"Stereo.b: {depth_baseline_mm / 1000.0:.6f}\n")
        else:
            bf = ("# NOMINAL baseline: the device calibration could not be read.\n"
                  "Stereo.b: 0.075000\n")
        text = f"""%YAML:1.0
# ORB-SLAM3 RGB-D config generated by c3_camera for this dataset.
# Intrinsics are the factory values for the COLOUR camera at the streamed
# resolution; depth is aligned to colour and the same size, so they apply to both.
File.version: "1.0"
Camera.type: "PinHole"
Camera1.fx: {intr.fx:.6f}
Camera1.fy: {intr.fy:.6f}
Camera1.cx: {intr.cx:.6f}
Camera1.cy: {intr.cy:.6f}
Camera1.k1: {(intr.distortion[0] if len(intr.distortion) > 0 else 0.0):.8f}
Camera1.k2: {(intr.distortion[1] if len(intr.distortion) > 1 else 0.0):.8f}
Camera1.p1: {(intr.distortion[2] if len(intr.distortion) > 2 else 0.0):.8f}
Camera1.p2: {(intr.distortion[3] if len(intr.distortion) > 3 else 0.0):.8f}
Camera1.k3: {(intr.distortion[4] if len(intr.distortion) > 4 else 0.0):.8f}
Camera.width: {intr.width}
Camera.height: {intr.height}
# Written as an integer: ORB-SLAM3 reads Camera.fps with readParameter<int>,
# which aborts on a real-valued node — so a 7.5 fps capture must not emit "7.5".
Camera.fps: {int(round(fps))}
Camera.RGB: 1
{bf}# metres = pixel / RGBD.DepthMapFactor. Our PNGs are uint16 millimetres.
RGBD.DepthMapFactor: {DEPTH_SCALE:.1f}
# Stereo.ThDepth is the name the RGB-D parser looks for; the bare ThDepth below
# is kept for the older ORB-SLAM2-style loader, which reads that one instead.
Stereo.ThDepth: 40.0
ThDepth: 40.0
ORBextractor.nFeatures: 1250
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0

# NOTE: these are IN-AIR factory intrinsics. Underwater, the port changes the
# effective focal length and adds radial error, so re-calibrate in water for
# metric work.
"""
        (self.dir / filename).write_text(text)

    def write_orbslam3_mono_yaml(self, intr, fps: float,
                                 filename: str = "orbslam3_mono.yaml") -> None:
        """The monocular counterpart, for a dataset recorded WITHOUT depth.

        A configuration sweep records some cells with `depth` off, and those
        datasets are still perfectly good monocular SLAM input. Handing them the
        RGB-D yaml instead would be worse than handing them nothing: ORB-SLAM3
        would look for a depth image per frame, and `RGBD.DepthMapFactor` in the
        file would assert that depth exists at all. So the config that ships is
        the one the recorded streams can actually satisfy — see the caller in
        c3_option_sweep.py, which picks between the two.
        """
        text = f"""%YAML:1.0
# ORB-SLAM3 MONOCULAR config generated by c3_camera for this dataset.
# This dataset was recorded WITHOUT a depth stream, so there is no depth/ to
# feed an RGB-D run — scale is unobservable and the trajectory is up to a
# similarity transform.
File.version: "1.0"
Camera.type: "PinHole"
Camera1.fx: {intr.fx:.6f}
Camera1.fy: {intr.fy:.6f}
Camera1.cx: {intr.cx:.6f}
Camera1.cy: {intr.cy:.6f}
Camera1.k1: {(intr.distortion[0] if len(intr.distortion) > 0 else 0.0):.8f}
Camera1.k2: {(intr.distortion[1] if len(intr.distortion) > 1 else 0.0):.8f}
Camera1.p1: {(intr.distortion[2] if len(intr.distortion) > 2 else 0.0):.8f}
Camera1.p2: {(intr.distortion[3] if len(intr.distortion) > 3 else 0.0):.8f}
Camera1.k3: {(intr.distortion[4] if len(intr.distortion) > 4 else 0.0):.8f}
Camera.width: {intr.width}
Camera.height: {intr.height}
# Integer: ORB-SLAM3 reads Camera.fps with readParameter<int> and aborts on a
# real-valued node.
Camera.fps: {int(round(fps))}
Camera.RGB: 1
ORBextractor.nFeatures: 1250
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0

# NOTE: these are IN-AIR factory intrinsics. Underwater, the port changes the
# effective focal length and adds radial error, so re-calibrate in water for
# metric work.
"""
        (self.dir / filename).write_text(text)

    # ------------------------------------------------------------------ close
    def close(self, timeout_s: float = 120.0) -> WriterStats:
        if self._closed:
            return self.stats
        self._closed = True
        pending = self._q.qsize()
        if pending and self.verbose:
            print(f"  flushing {pending} queued frames to disk ...")
        for _ in self._threads:
            try:
                self._q.put(None, timeout=5.0)
            except queue.Full:
                pass
        deadline = time.monotonic() + timeout_s
        for t in self._threads:
            t.join(timeout=max(0.5, deadline - time.monotonic()))

        # Anything still in the reorder buffer is stranded behind a frame that
        # never completed. Emit it in index order rather than losing it — the
        # timestamps stay monotonic because the buffer is keyed by frame index.
        with self._write_lock:
            if self._pending_rows and "frames" in self._csv:
                for idx in sorted(self._pending_rows):
                    r = self._pending_rows[idx]
                    if r is None:
                        continue
                    self._csv["frames"].writerow(r)
                    ts = r["t_unix"]
                    if r.get("rgb_file"):
                        self._lists["rgb"].write(f"{ts} {r['rgb_file']}\n")
                    if r.get("depth_file"):
                        self._lists["depth"].write(f"{ts} {r['depth_file']}\n")
                    if r.get("rgb_file") and r.get("depth_file"):
                        self._lists["assoc"].write(
                            f"{ts} {r['rgb_file']} "
                            f"{r.get('depth_t_device', ts)} {r['depth_file']}\n")
                self._pending_rows.clear()

        for fh in list(self._files.values()) + list(self._lists.values()):
            try:
                fh.close()
            except Exception:  # noqa: BLE001
                pass
        if self._video_fh is not None:
            self._video_fh.close()
            self._video_fh = None
        if self._mp4 is not None:
            self._mp4.release()
        return self.stats

    # -------------------------------------------------------------------- HUD
    def hud_line(self) -> str:
        s = self.stats
        el = time.monotonic() - s.started_at
        free_gb = shutil.disk_usage(self.dir).free / 1e9
        return (f"REC {s.written:6d} frames {el:6.1f}s  {s.mb:7.1f} MB "
                f"({s.mb_per_s:5.1f} MB/s)  q{self._q.qsize()}  "
                f"drop {s.dropped}  free {free_gb:5.1f} GB")

    def summary(self) -> dict:
        s = self.stats
        return {
            "dir": str(self.dir),
            "mode": self.mode,
            "frames": s.written,
            "images": s.images,
            "encoded_rgb_frames": self._video_frames,
            "color_encode": self.color_encode,
            "dropped": s.dropped,
            "drop_rate": round(s.drop_rate, 4),
            "waited_for_disk": s.waited,
            "megabytes": round(s.mb, 1),
            "mb_per_s": round(s.mb_per_s, 2),
            "mean_encode_ms_per_frame": (round(s.encode_ms_total / s.written, 2)
                                         if s.written else 0.0),
            "seconds": round(time.monotonic() - s.started_at, 1),
            "disk_free_gb": round(shutil.disk_usage(self.dir).free / 1e9, 1),
        }


# =============================================================================
# metadata
# =============================================================================
def _depth_lines(meta: dict, res: dict) -> list[str]:
    """What metadata.txt may claim about depth, given what was actually recorded.

    Three genuinely different datasets come out of this writer and only one of
    them is pixel-aligned RGB-D. Saying so unconditionally is the kind of error a
    consumer cannot detect: they index depth[u,v] with a colour pixel and get a
    plausible number from the wrong ray.
    """
    streams = meta.get("streams")
    align = meta.get("depth_align")
    color_out, depth_out = res.get("color_out"), res.get("depth_out")
    if streams is not None and "depth" not in streams:
        return [
            "depth: NOT RECORDED in this session.",
            "    depth/ and depth.txt exist but are empty, and associations.txt has",
            "    no pairs. This is monocular (or stereo, if left/ and right/ are",
            "    populated) data — not RGB-D.",
        ]
    if align == "color" and color_out is not None and color_out == depth_out:
        return [
            f"depth aligned to: {align}",
            "    depth is warped into the COLOUR camera's frame and emitted at the",
            "    same resolution, so rgb/ and depth/ correspond pixel-for-pixel and",
            "    the colour intrinsics apply to both. That is what RGB-D SLAM needs.",
        ]
    return [
        f"depth aligned to: {align}",
        f"    depth is {depth_out} while colour is {color_out}, so rgb[u,v] and",
        "    depth[u,v] are NOT the same ray and the colour intrinsics do NOT",
        "    describe the depth image. Rescale depth to the colour grid (NEAREST",
        "    only — interpolating depth invents distances) before treating this as",
        "    RGB-D, or use the depth intrinsics in calibration.json.",
    ]


def _render_preflight(report: dict | None) -> list[str]:
    """The readiness checks that were not OK, for metadata.txt.

    Separate from :func:`_render_metadata` so the shape of a preflight report is
    testable on its own, and so a report from an older or newer preflight cannot
    take the metadata file down with it — anything unrecognised renders as
    nothing rather than raising.
    """
    if not isinstance(report, dict):
        return []
    if report.get("ran") is False:
        return ["", "== preflight ==",
                "not run (--no-preflight): the state of the rig at the start of "
                "this run was not recorded."]
    checks = [c for c in (report.get("checks") or [])
              if isinstance(c, dict) and c.get("status") in ("warn", "fail")]
    if not checks:
        return ["", "== preflight ==",
                f"all checks passed ({report.get('elapsed_s', '?')} s)."]
    out = ["", "== preflight ==",
           "These readiness checks were NOT clean when this run started. They are",
           "the first thing to suspect if something in this dataset looks wrong:"]
    for c in checks:
        out.append(f"  [{str(c.get('status')).upper():<4}] {c.get('name')}: "
                   f"{str(c.get('detail', '')).splitlines()[0] if c.get('detail') else ''}")
    return out


def _render_metadata(meta: dict, notes: str = "") -> str:
    cam = meta.get("camera", {})
    res = meta.get("resolved", {})
    color_encode = meta.get("color_encode", "none")
    lines = [
        "# C3 underwater RGB-D / visual-SLAM dataset",
        f"# written by c3_camera at {datetime.now().isoformat(timespec='seconds')}",
        "",
        "== READ THIS FIRST: depth units ==",
        f"depth_scale: {DEPTH_SCALE:.1f}",
        f"    metres = png_pixel_value / {DEPTH_SCALE:.1f}",
        "    depth/*.png are 16-bit single-channel, values are MILLIMETRES,",
        "    and 0 means NO MEASUREMENT (not zero distance) — mask it out.",
        "",
        "    Per-tool parameter names for this data:",
        f"      ORB-SLAM3   RGBD.DepthMapFactor: {DEPTH_SCALE:.1f}",
        f"      Open3D      depth_scale={DEPTH_SCALE:.1f} (it divides, same direction)",
        "      RTAB-Map     reads 16-bit mm natively; no scale parameter needed",
        "",
        "    Do NOT use 5000. TUM's own sequences use 5000 because that is how",
        "    their Kinect data was scaled; it is not a convention and it is not",
        "    right for millimetre data. Using 5000 here makes every depth 5x too",
        "    small, which looks plausible instead of broken.",
        "",
        "    Read depth with cv2.IMREAD_UNCHANGED. Without it OpenCV silently",
        "    returns 8-bit and every value is wrong by a factor of 257.",
        "",
        "== timestamps ==",
        "All timestamps in every file are UNIX EPOCH SECONDS, 6 decimal places.",
        "",
        "Measured on the recording host: dai.Clock.now() and time.monotonic() are",
        "the same clock (CLOCK_MONOTONIC), agreeing to within 8 microseconds. So",
        "the camera, the camera IMU and the MAVLink log were already on ONE",
        "timeline before conversion; the files here differ from it by a single",
        "constant offset, recorded below.",
        "",
        f"clock_offset_monotonic_to_unix: {meta.get('clock', {}).get('offset', 'unknown')}",
        "    t_unix = t_monotonic + clock_offset_monotonic_to_unix",
        f"clock_offset_captured_at_unix: {meta.get('clock', {}).get('captured_at', '')}",
        "",
        "  rgb/, depth/, left/, right/ filenames and rgb.txt/depth.txt/",
        "  associations.txt  -> t_unix of the COLOUR frame's capture time",
        "",
        "  NOTE ON FILE NAMING: all four images of one frame share the COLOUR",
        "  frame's timestamp as their filename, rather than each being named by its",
        "  own capture time. That makes association exact and unambiguous — pairing",
        "  rgb.txt against depth.txt needs no nearest-neighbour search at all. Each",
        "  stream's true capture time is not lost: it is in associations.txt column",
        "  3 for depth, and in frames.csv (*_t_device) for every stream.",
        "  frames.csv        -> t_unix plus raw t_monotonic and each stream's own",
        "                       device capture time, so per-stream skew is visible",
        "  imu_camera.csv    -> capture time of each BNO086 sample (same clock)",
        "  imu_rov.csv, telemetry.csv, control.csv",
        "                    -> ARRIVAL time on the host, not measurement time.",
        "",
        "  The asymmetry matters: camera and camera-IMU stamps are capture times,",
        "  MAVLink stamps are arrival times and lag the true measurement by the",
        "  tether/routing delay (a few ms on this Ethernet path). Every MAVLink row",
        "  also carries ArduSub's own time_boot_ms; regress that against t_unix if",
        "  you need better than a few milliseconds.",
        "",
        "== which IMU to use ==",
    ]
    imu = meta.get("imu", {})
    if imu.get("present"):
        lines += [
            f"imu_camera: {imu.get('model', '?')} on the camera board, "
            f"{imu.get('rate_hz', '?')} Hz achieved",
            "    PREFER THIS ONE for visual-inertial work: it is rigidly mounted",
            "    with the image sensors and its timestamps come from the same",
            "    device clock as the frames.",
            "",
            "    TWO CAVEATS, both measured, both unresolved in this dataset:",
            "    1. IMU-to-camera extrinsics are NOT available from the device",
            f"       ({imu.get('extrinsics_note', 'unavailable')})",
            "       T_imu_cam is therefore UNKNOWN. Calibrate it (Kalibr) or",
            "       measure it mechanically before trusting a VIO scale.",
            "    2. Accelerometer scale is off: static |a| measured about",
            f"       {imu.get('static_accel_magnitude', '~11.8')} m/s^2 where gravity is 9.81",
            "       (~20% high), with the accuracy flag reading UNRELIABLE/MEDIUM.",
            "       The per-sample accuracy field is logged so you can see this.",
            "       IMU intrinsics need calibration before VIO use.",
            "",
            "    Axis convention relative to the camera optical frame is NOT",
            "    documented by the vendor and was not established here; treat the",
            "    IMU-camera rotation as unknown along with the translation.",
            "",
        ]
    else:
        lines += ["imu_camera: not present / not enabled this session", ""]
    lines += [
        "imu_rov: the ArduSub/Navigator IMU via MAVLink (RAW_IMU, SCALED_IMU2).",
        "    Metres away from the camera, on a different board, arriving over UDP",
        "    with arrival-time stamps. Useful as context and for cross-checking",
        "    attitude; not a good primary IMU for VIO.",
        "    Fields are in native MAVLink units, NOT converted:",
        "      RAW_IMU      xacc/yacc/zacc, xgyro/ygyro/zgyro, xmag/ymag/zmag",
        "                   (raw sensor units as ArduPilot reports them)",
        "      SCALED_IMU2  xacc etc. in mg (milli-g); gyro in mrad/s; mag in mgauss",
        "      ATTITUDE     roll/pitch/yaw in radians, *speed in rad/s",
        "      SCALED_PRESSURE2  press_abs/press_diff in hPa, temperature in cdegC",
        "                        (this is the Bar30 on a BlueROV2)",
        "      VFR_HUD      alt in metres; on a submarine this tracks depth",
        "      SERVO_OUTPUT_RAW  servo1..8 = thrusters, servo9 = Newton gripper, PWM us",
        "    Check the MAVLink XML for your dialect before relying on any unit.",
        "",
        "== camera ==",
        f"model: {cam.get('product', '?')}  mxid: {cam.get('mxid', '?')}",
        f"colour: {res.get('color_out')}  depth: {res.get('depth_out')}  "
        f"mono: {res.get('mono')}",
        f"fps_requested: {meta.get('fps_requested')}   "
        f"fps_achieved: {meta.get('fps_achieved')}",
        f"colour wire format: {color_encode if color_encode != 'none' else meta.get('color_wire')} "
        f"({'hardware encoded at ' + str(meta.get('video_bitrate_kbps')) + ' kbit/s' if color_encode in ('h264', 'h265') else ('lossy per-frame JPEG' if color_encode == 'mjpeg' else ('4:2:0 chroma subsampled; luma full resolution' if meta.get('color_wire') == 'nv12' else 'full chroma'))})",
        f"mono source: {meta.get('mono_source')} "
        f"({'straight off the sensors' if meta.get('mono_source') == 'raw' else 'rectified by StereoDepth'})",
        # Conditional, because a configuration sweep deliberately records cells
        # with no depth and cells whose depth is on a different grid. Stating
        # "pixel-for-pixel" unconditionally made the two most important
        # dataset-level facts unreadable from the file that exists to state them.
        *_depth_lines(meta, res),
        "",
        ("RGB is lossy hardware-encoded; depth remains bit-exact 16-bit PNG."
         if color_encode in ("mjpeg", "h264", "h265") else
         "PNG images are lossless. depth is 16-bit; rgb/left/right are 8-bit."),
        ("For H.264/H.265, run c3_camera/c3_extract_rgb_video.py on this folder "
         "before SLAM unless rgb/ has already been extracted."
         if color_encode in ("h264", "h265") else
         "rgb.mp4 is a lossy convenience preview only — never use it as data."),
        "",
        "== calibration ==",
        "calibration.json holds the on-device factory calibration: per-camera",
        "intrinsics and distortion at the streamed resolution, the stereo",
        "baseline, and the camera-to-camera extrinsics. The C3 is individually",
        "calibrated, so prefer these over any nominal values.",
        "",
        "These are IN-AIR values. Underwater the port changes the effective focal",
        "length and adds radial distortion, so for metric in-water work they are a",
        "starting point, not an answer.",
        "",
        f"{meta.get('sweep', {}).get('slam_config') or 'orbslam3_rgbd.yaml'} is a "
        f"generated starting config using these intrinsics.",
        "",
        "== vehicle ==",
        f"ArduSub: {meta.get('ardusub_version', 'unknown')}   "
        f"frame: {meta.get('vehicle_type', 'unknown')}",
        f"mavlink transport: {meta.get('mavlink', {}).get('transport', 'none')}  "
        f"source: {meta.get('mavlink', {}).get('source', '-')}",
        "",
        "== session ==",
        f"duration_s: {meta.get('duration_s')}",
        f"frames_written: {meta.get('frames_written')}   "
        f"frames_dropped: {meta.get('frames_dropped')}",
    ]
    if meta.get("frames_dropped"):
        lines += [
            "    NOTE: frames were dropped. Timestamps stay correct — a drop is a",
            "    gap, not a shift — but the effective frame rate is lower than",
            "    requested and gaps are visible as jumps in frames.csv seq columns.",
        ]
    # Anything preflight was unhappy about when this run started. Only the
    # not-OK checks, and only when there are any: a clean rig needs no paragraph,
    # but "the MAVLink endpoint was missing" is usually the whole explanation for
    # an empty telemetry.csv, and metadata.json alone is not where anyone looks.
    lines += _render_preflight(meta.get("preflight"))
    if notes:
        lines += ["", "== operator notes ==", notes]
    lines += ["", "== files ==",
              ("  rgb.h264/rgb.h265 + rgb_video.csv  encoded RGB + frame timestamps"
               if color_encode in ("h264", "h265") else
               "  rgb/, depth/, left/, right/   per-frame images"),
              "  depth/                         lossless uint16 PNG millimetres",
              "  rgb.txt, depth.txt            TUM lists: 'timestamp filename'",
              "  associations.txt              rgb_ts rgb_file depth_ts depth_file",
              "  frames.csv                    per-frame index and per-stream times",
              "  imu_camera.csv                BNO086 (preferred for VIO)",
              "  imu_rov.csv                   ArduSub IMU via MAVLink",
              "  telemetry.csv                 attitude, pressure/depth, battery, servo, mode",
              "  control.csv                   pilot inputs (RC_CHANNELS / MANUAL_CONTROL)",
              "  calibration.json              on-device factory calibration",
              "  orbslam3_rgbd.yaml            generated starting config",
              "  metadata.txt / metadata.json  this file, and the same as JSON",
              ""]
    return "\n".join(lines)


def estimate_rate(color_wh, depth_wh, mono_wh, fps: float,
                  streams, color_wire: str = "nv12",
                  color_encode: str = "none",
                  video_bitrate_kbps: int = 4000) -> dict:
    """Wire and disk cost of a research-mode configuration.

    Both numbers matter and they are very different here: the PoE link caps out
    around 91.5 Mbit/s (~11 MB/s) while the NVMe on this host does orders of
    magnitude more, so the LINK is the binding constraint even for four lossless
    streams. The disk figure is for planning capacity, not throughput.
    """
    per_frame_wire = 0.0
    per_frame_disk = 0.0
    if "color" in streams:
        if color_encode == "mjpeg":
            # Device-side JPEG: measured ~6:1 against raw NV12 at quality 90.
            from .config import MJPEG_RATIO
            per_frame_wire += color_wh[0] * color_wh[1] * 1.5 / MJPEG_RATIO
        elif color_encode in ("h264", "h265"):
            # Convert the configured rate back to an equivalent per-frame size
            # so the common totals and link-limited-fps calculation stay valid.
            per_frame_wire += video_bitrate_kbps * 1000 / 8 / max(fps, 1e-9)
        else:
            per_frame_wire += color_wh[0] * color_wh[1] * (3.0 if color_wire == "bgr"
                                                          else 1.5)
        if color_encode in ("h264", "h265"):
            per_frame_disk += video_bitrate_kbps * 1000 / 8 / max(fps, 1e-9)
        else:
            per_frame_disk += color_wh[0] * color_wh[1] * 3 * 0.55
    if "depth" in streams:
        per_frame_wire += depth_wh[0] * depth_wh[1] * 2
        per_frame_disk += depth_wh[0] * depth_wh[1] * 2 * 0.45   # lots of zeros
    for s in ("left", "right"):
        if s in streams:
            per_frame_wire += mono_wh[0] * mono_wh[1]
            per_frame_disk += mono_wh[0] * mono_wh[1] * 0.75
    return {
        "wire_mbit_s": per_frame_wire * fps * 8 / 1e6,
        "wire_kb_frame": per_frame_wire / 1024,
        "disk_mb_min": per_frame_disk * fps * 60 / 1e6,
        "disk_gb_hour": per_frame_disk * fps * 3600 / 1e9,
        "link_limited_fps": (91.5e6 / 8) / per_frame_wire if per_frame_wire else math.inf,
    }
