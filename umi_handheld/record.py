"""record.py — produce a recording session, from the bench camera or from synthetic input.

A session is the pipeline's only input contract. Whatever produced it, it looks the same:

    <session>/
      left/<t_unix>.png     mono8, the rectified-left grid
      depth/<t_unix>.png    uint16 millimetres, aligned to rectified-left, 0 = invalid
      frames.csv            per-frame index and timestamps
      session.json          camera model provenance, the resolved config, the source

Two sources
-----------
--source synthetic   Consumes a capture from tools/make_synthetic_capture.py. Useful when
                     the camera is not attached, and the only path with ground truth.

--source device      DepthAI on a bench OAK-D W over USB. Run against hardware.

Recording UX (device)
---------------------
A preview window shows rectified-left beside a depth colourmap, with the record state,
frame count, elapsed time, fps, and — the one that decides whether a take is usable —
the percentage of pixels carrying valid depth. It turns red below --valid-warn, because
the usual cause is standing closer than MinZ, where stereo simply cannot triangulate.

    SPACE   start / stop a take          K   switch demonstration <-> grippercalibration
    E       near <-> far mode            Q, ESC  quit
                                         takes auto-number: sessions/<kind>_0001, _0002 …

E switches to --config-extended, which defaults to <config>_extended.yaml if that file
exists. It halves MinZ by turning extended disparity on, and that is a pipeline
BUILD-time setting with no runtime config field, so the device is closed and reopened —
about a second, and the same thing test_oak_depth.py does. Refused mid-take, because the
two modes have different depth geometry and one session.json can only describe one.

--no-preview records one fixed-length take with no window, for machines with no display.

The device path's one non-obvious setting: depth must be aligned to RECTIFIED_LEFT, the
DepthAlign enum member, not to the CameraBoardSocket for the left camera. The socket
overload aligns to the RAW left camera, which is a different grid from the rectified one
the mono stream is served on.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from umi_handheld.camera_model import REPO, CameraModel        # noqa: E402


def _repair_qt_plugin_path() -> None:
    """Point Qt back at a plugin directory that actually has the xcb plugin.

    `cv2/config-3.py:14-18` overwrites QT_QPA_PLATFORM_PLUGIN_PATH at import with
    `<cv2>/qt/plugins`, and in the conda `robust` env that directory is renamed
    to `qt/plugins.disabled` so PyQt5/PySide6 can coexist — leaving the preview
    window to die with "no Qt platform plugin could be initialized". Qt reads the
    variable at first QGuiApplication, so fixing it here is early enough.

    Deliberately a copy of `c3_camera/viz.py:repair_qt_plugin_path` rather than an
    import: this is eight lines of display plumbing, and importing `c3_camera`
    for it would drag that package's depthai-2.x import guard into this one.
    """
    qxcb = os.path.join("platforms", "libqxcb.so")
    current = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH")
    if current and os.path.exists(os.path.join(current, qxcb)):
        return
    root = os.path.join(os.path.dirname(os.path.abspath(cv2.__file__)), "qt")
    for name in ("plugins", "plugins.disabled"):
        candidate = os.path.join(root, name)
        if os.path.exists(os.path.join(candidate, qxcb)):
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = candidate
            fonts = os.path.join(root, "fonts")
            if os.path.isdir(fonts):
                os.environ["QT_QPA_FONTDIR"] = fonts
            return


_repair_qt_plugin_path()

FRAME_FIELDS = ("idx", "t_unix", "left_file", "depth_file", "right_file",
                "color_file", "color_t_unix", "color_skew_ms",
                "depth_valid_fraction", "source")


def _deep_merge(base: dict, over: dict) -> dict:
    """over wins, but dicts merge key by key so a profile states only its deltas."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_pipeline_config(path, _seen=None) -> dict:
    """Load a pipeline config, resolving `extends:` so profiles carry only deltas.

    Without this a second profile has to duplicate the gripper/warp/zarr sections, and
    the two copies drift the first time one of them is edited.
    """
    path = Path(path)
    if not path.is_absolute():
        path = REPO / path
    path = path.resolve()
    _seen = _seen or []
    if path in _seen:
        raise ValueError(f"extends: cycle {' -> '.join(p.name for p in _seen + [path])}")
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    parent = cfg.pop("extends", None)
    if parent:
        ppath = Path(parent)
        if not ppath.is_absolute():
            ppath = path.parent / ppath
        base = load_pipeline_config(ppath, _seen + [path])
        chain = base.pop("_extends_chain", []) + [base.pop("_path")]
        cfg = _deep_merge(base, cfg)
        cfg["_extends_chain"] = chain

    if cfg.get("schema") != "umi_handheld_pipeline/1":
        raise ValueError(f"{path}: unexpected schema {cfg.get('schema')!r}")
    try:
        cfg["_path"] = str(path.relative_to(REPO))
    except ValueError:                      # a config from outside the repo tree
        cfg["_path"] = str(path)
    return cfg


# ------------------------------------------------------------------- synthetic

def read_synthetic(capture: Path):
    """Yield (t_unix, left_path, depth_path) from a synthetic capture directory."""
    fcsv = capture / "frames.csv"
    if not fcsv.exists():
        raise FileNotFoundError(f"{capture}: no frames.csv — not a capture directory")
    with fcsv.open() as fh:
        for row in csv.DictReader(fh):
            yield (float(row["t_unix"]),
                   capture / row["left_file"],
                   capture / row["depth_file"])


def record_synthetic(capture: Path, out: Path, cfg: dict, src_cam: CameraModel,
                     link: bool) -> list[dict]:
    rows = []
    for i, (t, lp, dp) in enumerate(read_synthetic(capture)):
        stamp = f"{t:.6f}"
        for sub, srcp in (("left", lp), ("depth", dp)):
            dst = out / sub / f"{stamp}.png"
            if link:
                if dst.exists():
                    dst.unlink()
                dst.symlink_to(srcp.resolve())
            else:
                shutil.copyfile(srcp, dst)
        d = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
        if d is None:
            raise RuntimeError(f"unreadable depth frame: {dp}")
        rows.append({
            "idx": i + 1, "t_unix": stamp,
            "left_file": f"left/{stamp}.png", "depth_file": f"depth/{stamp}.png",
            "depth_valid_fraction": f"{float(np.count_nonzero(d)) / d.size:.6f}",
            "source": "synthetic",
        })
    return rows


# ---------------------------------------------------------------------- device

def build_device_pipeline(cfg: dict):
    """StereoDepth wired for rectified-left mono + depth on the same grid."""
    import depthai as dai                                    # noqa: F401  (device only)

    sd = cfg["record"]["stereo_depth"]
    # `streams` decides what crosses the link. left+depth are the pipeline's input
    # contract and are always sent; right and colour are optional and cost real
    # latency, because everything shares one XLink. Both mono cameras are still
    # created either way — stereo needs both eyes regardless of what is streamed out.
    streams = [str(s).lower() for s in
               (cfg["record"].get("streams") or ["left", "right", "depth", "color"])]
    want_right = "right" in streams

    pipeline = dai.Pipeline()
    mono_l = pipeline.create(dai.node.MonoCamera)
    mono_r = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    xout_l = pipeline.create(dai.node.XLinkOut)
    xout_d = pipeline.create(dai.node.XLinkOut)
    xout_l.setStreamName("left")
    xout_d.setStreamName("depth")
    if want_right:
        xout_r = pipeline.create(dai.node.XLinkOut)
        xout_r.setStreamName("right")

    res = {"400p": dai.MonoCameraProperties.SensorResolution.THE_400_P,
           "720p": dai.MonoCameraProperties.SensorResolution.THE_720_P,
           "800p": dai.MonoCameraProperties.SensorResolution.THE_800_P}[sd["mono_resolution"]]
    # The sensors may run faster than the stereo core can consume. Feeding it more
    # frames than it outputs is not waste: the temporal filter gets more evidence, and
    # measured +4 points of valid depth at mono 60. It shortens the auto-exposure
    # ceiling to 1/fps, which on this unit is not binding (AE chose 10.4 ms at 30 fps).
    mono_fps = float(cfg["record"].get("mono_fps") or cfg["record"]["fps"])
    for m, sock in ((mono_l, dai.CameraBoardSocket.CAM_B),
                    (mono_r, dai.CameraBoardSocket.CAM_C)):
        m.setResolution(res)
        m.setBoardSocket(sock)
        m.setFps(mono_fps)

    # THE setting. The enum overload aligns to the rectified stream; the
    # CameraBoardSocket overload would align to the RAW left camera instead.
    stereo.setDepthAlign(dai.RawStereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT)
    stereo.setLeftRightCheck(sd["lr_check"])
    stereo.setExtendedDisparity(sd["extended_disparity"])
    stereo.setSubpixel(sd["subpixel"])
    if sd.get("confidence_threshold") is not None:
        stereo.initialConfig.setConfidenceThreshold(int(sd["confidence_threshold"]))
    stereo.initialConfig.setMedianFilter(
        {"off": dai.MedianFilter.MEDIAN_OFF, "MEDIAN_OFF": dai.MedianFilter.MEDIAN_OFF,
         "3x3": dai.MedianFilter.KERNEL_3x3, "5x5": dai.MedianFilter.KERNEL_5x5,
         "7x7": dai.MedianFilter.KERNEL_7x7}[str(sd["median_filter"]).lower()
                                             if str(sd["median_filter"]).lower() == "off"
                                             else sd["median_filter"]])

    if sd.get("subpixel") and sd.get("subpixel_bits"):
        stereo.setSubpixelFractionalBits(int(sd["subpixel_bits"]))
    if sd.get("hardware_resources"):
        hr = sd["hardware_resources"]
        stereo.setPostProcessingHardwareResources(int(hr["shaves"]), int(hr["slices"]))

    icfg = stereo.initialConfig.get()
    icfg.algorithmControl.disparityShift = int(sd["disparity_shift"])
    pp, ppc = icfg.postProcessing, sd["post_processing"]
    pp.speckleFilter.enable = ppc["speckle"]["enable"]
    pp.speckleFilter.speckleRange = int(ppc["speckle"]["speckle_range"])
    pp.temporalFilter.enable = ppc["temporal"]["enable"]
    pp.spatialFilter.enable = ppc["spatial"]["enable"]
    pp.spatialFilter.holeFillingRadius = int(ppc["spatial"]["hole_filling_radius"])
    pp.spatialFilter.numIterations = int(ppc["spatial"]["num_iterations"])
    pp.spatialFilter.alpha = float(ppc["spatial"]["alpha"])
    pp.spatialFilter.delta = int(ppc["spatial"]["delta"])
    pp.temporalFilter.alpha = float(ppc["temporal"]["alpha"])
    pp.temporalFilter.delta = int(ppc["temporal"]["delta"])
    # Decimation runs FIRST, so every later filter works on a quarter of the pixels.
    # That is where the frame rate comes from: the spatial filter is all-or-nothing
    # (measured 18.5 fps at radius 1 and 18.3 at radius 3), and decimating is what
    # makes affording it possible. It shrinks depth only — the mono streams stay full
    # size, and the disparity search is unchanged, so MinZ does NOT move.
    dec = int((ppc.get("decimation") or {}).get("factor", 1) or 1)
    if dec > 1:
        DM = dai.RawStereoDepthConfig.PostProcessing.DecimationFilter.DecimationMode
        pp.decimationFilter.decimationFactor = dec
        pp.decimationFilter.decimationMode = {
            "non_zero_median": DM.NON_ZERO_MEDIAN,
            "non_zero_mean": DM.NON_ZERO_MEAN,
            "pixel_skipping": DM.PIXEL_SKIPPING,
        }[str(ppc["decimation"].get("mode", "non_zero_median")).lower()]
    if ppc["threshold"]["enable"]:
        pp.thresholdFilter.minRange = int(ppc["threshold"]["min_range_mm"])
        pp.thresholdFilter.maxRange = int(ppc["threshold"]["max_range_mm"])
    stereo.initialConfig.set(icfg)

    mono_l.out.link(stereo.left)
    mono_r.out.link(stereo.right)
    # LATENCY, and the single biggest lever on it. These default to queueSize 8,
    # non-blocking. Non-blocking sounds safe — a full queue drops the oldest — but the
    # node still CONSUMES from the front, so once the stereo core is at its compute
    # limit the queue sits permanently full and every frame it works on is 8 frames old.
    #
    # Measured on this unit at 1280x800, drops counted by sequence number so they are
    # exact (umi_handheld/bench_stereo.py builds the same pipeline; 10 s per row):
    #
    #   queueSize   age at host   frames dropped
    #       1         153.2 ms        3.33%        <- default here
    #       2         242.9 ms        4.65%
    #       3         248.0 ms        4.33%
    #       4         241.2 ms        4.33%
    #       8         251.9 ms        4.33%        <- the DepthAI default
    #
    # Size 1 is better on BOTH axes, and it is a cliff rather than a gradient: 2 is
    # already back at 8's latency. The surplus frames were being discarded either way;
    # a deep queue only discarded them later. Watching wall-clock gaps instead of
    # sequence numbers hides this — size 1 trades a few 2-frame gaps for more 1-frame
    # gaps, which looks worse and is not.
    #
    # It does NOT undo the mono_fps > fps trick: the temporal filter still sees every
    # frame the stereo core manages to consume, which is what the density came from.
    #
    # What is left after this is resolution, not configuration: 152 ms at 1280x800
    # against 36 ms at 640x400 with the same filter chain. Turning the filters off
    # changes it by 11 ms, and dropping streams off the link changes it by none.
    lat = cfg["record"].get("stereo_input_queue") or {}
    for inp in (stereo.left, stereo.right):
        inp.setBlocking(bool(lat.get("blocking", False)))
        inp.setQueueSize(int(lat.get("size", 1)))
    stereo.rectifiedLeft.link(xout_l.input)
    stereo.depth.link(xout_d.input)
    if want_right:
        stereo.rectifiedRight.link(xout_r.input)

    # The OTHER end of the same lever. Until 2026-08-06 these were left at the DepthAI
    # default (8, blocking) while only the stereo INPUT queues were tuned — and the
    # stereo inputs were where 95 ms was hiding. A deep blocking queue here holds
    # finished frames on the device waiting for XLink, so the host receives frames that
    # are already several periods old, for the same reason as the table above.
    # `bench_stereo.py --config <profile>` reports `age` and `drop%`; change this with a
    # measurement in hand, not on the argument.
    xq = cfg["record"].get("xlink_out_queue") or {}
    for node in (xout_l, xout_d) + ((xout_r,) if want_right else ()):
        node.input.setBlocking(bool(xq.get("blocking", False)))
        node.input.setQueueSize(int(xq.get("size", 1)))

    # Colour on CAM_A. On this unit that is an OV9782 global-shutter 1 MP sensor, not
    # the 12 MP IMX378 the C3 carries. It is a SEPARATE camera: colour is not aligned
    # to depth and does not share the stereo pair's sequence numbers, so it is paired
    # by nearest timestamp and its own stamp is written to frames.csv.
    ccfg = cfg["record"].get("colour") or {}
    if ccfg.get("enable", False) and "color" in streams:
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam.setResolution({"800p": dai.ColorCameraProperties.SensorResolution.THE_800_P,
                           "720p": dai.ColorCameraProperties.SensorResolution.THE_720_P,
                           "1080p": dai.ColorCameraProperties.SensorResolution.THE_1080_P}[
                              ccfg.get("resolution", "800p")])
        cam.setFps(cfg["record"]["fps"])
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        xout_c = pipeline.create(dai.node.XLinkOut)
        xout_c.setStreamName("color")
        xout_c.input.setBlocking(bool(xq.get("blocking", False)))
        xout_c.input.setQueueSize(int(xq.get("size", 1)))
        # Encoding colour on the device is the single biggest frame-rate win here.
        # Writing 1280x800 BGR as PNG on the host measured 45.6 ms per frame — a 21.9
        # fps ceiling on its own, which is what actually capped recording at 17 fps.
        # The device's hardware encoder does it for free and the host writes bytes.
        if ccfg.get("encode", "mjpeg") == "mjpeg":
            enc = pipeline.create(dai.node.VideoEncoder)
            enc.setDefaultProfilePreset(
                cam.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
            enc.setQuality(int(ccfg.get("quality", 97)))
            cam.video.link(enc.input)
            enc.bitstream.link(xout_c.input)
        else:
            cam.video.link(xout_c.input)

    return pipeline


KINDS = ("demonstration", "grippercalibration")


def SUBDIRS(has_colour: bool, has_right: bool = True):
    return (("left", "depth") + (("right",) if has_right else ())
            + (("color",) if has_colour else ()))


def next_session_dir(base: Path, kind: str) -> Path:
    """<base>/<kind>_0001, then _0002 … so takes never overwrite each other."""
    base.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in base.glob(f"{kind}_*"):
        try:
            n = max(n, int(p.name.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            pass
    return base / f"{kind}_{n + 1:04d}"


class FrameWriter:
    """Encode and write frames on worker threads, so the capture loop never waits.

    Serialised in the capture loop, the three mono/depth PNGs measured 12.0 ms per
    frame — an 83 fps ceiling, close enough to the device's rate to steal frames from
    it. cv2.imwrite releases the GIL, so plain threads get real parallelism here.

    The queue is bounded: if the disk cannot keep up, `put` blocks and the recording
    slows down. That is the honest failure mode — the alternative is buffering until
    memory runs out and losing the whole take.
    """

    def __init__(self, workers: int = 4, depth: int = 96):
        import queue
        import threading
        self.q: queue.Queue = queue.Queue(maxsize=depth)
        self.errors: list[str] = []
        self._stop = object()
        self.threads = [threading.Thread(target=self._run, daemon=True)
                        for _ in range(workers)]
        for t in self.threads:
            t.start()

    def _run(self):
        while True:
            item = self.q.get()
            try:
                if item is self._stop:
                    return
                path, payload, params = item
                if isinstance(payload, (bytes, bytearray)):
                    Path(path).write_bytes(payload)
                elif not cv2.imwrite(path, payload, params):
                    self.errors.append(f"imwrite failed: {path}")
            except Exception as exc:                     # keep the take, report after
                self.errors.append(f"{path}: {exc}")
            finally:
                self.q.task_done()

    def put(self, path: Path, payload, params=None):
        self.q.put((str(path), payload, params or []))

    def flush(self):
        self.q.join()

    def close(self):
        self.flush()
        for _ in self.threads:
            self.q.put(self._stop)
        for t in self.threads:
            t.join(timeout=2.0)


class TagProbe:
    """Live AprilTag readout for the preview, built from the config the EXTRACTOR uses.

    It answers the one question a recordist cannot answer from a grey image: will this
    framing actually yield tag detections when extract_gripper_width.py runs on it
    later? A 2026-08-06 trial take came back at 42% both-tags, and the reason — the
    tags were 2.9 px per module, under the ~3 needed to decode — was only visible
    afterwards, from the recording. `px_per_module` puts that number on screen while
    the rig is still in the operator's hand.

    The detector and the marker geometry come from `gripper.aruco` via
    extract_gripper_width's own helpers, so the preview cannot answer for a different
    dictionary or marker size than the extractor will use.
    """

    HISTORY = 60                                # 2 s at 30 fps

    def __init__(self, acfg: dict, cam: CameraModel):
        # Deferred: extract_gripper_width imports load_pipeline_config from this
        # module, so a module-level import here would be circular.
        from umi_handheld.extract_gripper_width import make_detector   # noqa: E402
        self.det = make_detector(acfg)
        dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, acfg["dictionary"]))
        # + 1 black border module on each side; 36h11 is 6x6 of data -> 8 across, and
        # that total is what a detected side is divided by.
        self.modules = int(dic.markerSize) + 2
        self.ids = (int(acfg["left_tag_id"]), int(acfg["right_tag_id"]))
        self.length_m = float(acfg["marker_length_m"])
        self.cam = cam
        self.recent: deque = deque(maxlen=self.HISTORY)

    def annotate(self, panel: np.ndarray, mono: np.ndarray) -> dict:
        """Detect on `mono`, draw onto `panel` (same grid), return the readout."""
        from umi_handheld.extract_gripper_width import tag_pose         # noqa: E402
        corners, ids, _ = self.det.detectMarkers(mono)
        id_list = [] if ids is None else ids.ravel().tolist()
        if id_list:
            cv2.aruco.drawDetectedMarkers(panel, corners, ids)
        H, W = mono.shape[:2]
        K = self.cam.K(W, H)
        seen, sides, zs = set(), [], []
        for q, mid in zip(corners, id_list):
            if mid not in self.ids:
                continue                        # drawn, but not counted as ours
            seen.add(mid)
            p = q.reshape(4, 2)
            sides.append(float(np.mean([np.linalg.norm(p[i] - p[(i + 1) % 4])
                                        for i in range(4)])))
            pose = tag_pose(q, self.length_m, K, self.cam.dist)
            if pose is not None:
                zs.append(float(pose[1][2]) * 1000.0)
        self.recent.append(len(seen) == len(self.ids))
        return {
            "seen": {i: (i in seen) for i in self.ids},
            "px_per_module": (float(np.mean(sides)) / self.modules) if sides else None,
            "z_mm": float(np.mean(zs)) if zs else None,
            "rate": float(np.mean(self.recent)) if self.recent else 0.0,
        }


def _colour_panel(colour, W: int, H: int) -> np.ndarray:
    """The colour frame fitted to the mono panel, or a labelled placeholder.

    A placeholder rather than a dropped panel: the window must not resize under the
    recordist when colour is off or its first frame has not arrived yet.
    """
    if colour is None:
        cp = np.full((H, W, 3), 30, np.uint8)
        cv2.putText(cp, "no colour", (12, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (120, 120, 120), 2)
    elif colour.shape[:2] != (H, W):
        cp = cv2.resize(colour, (W, H), interpolation=cv2.INTER_AREA)
    else:
        cp = colour.copy()
    # CAM_A is a separate sensor: this view is NOT registered to the depth beside it,
    # and side by side is exactly where that gets forgotten.
    cv2.putText(cp, "CAM_A - not aligned to depth", (12, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    return cp


def draw_overlay(mono: np.ndarray, colour, depth: np.ndarray, st: dict,
                 probe: "TagProbe | None" = None) -> np.ndarray:
    """mono | colour | depth side by side, with the state a recordist needs.

    Tag outlines go on the MONO panel, not the colour one: rectified-left is the grid
    extract_gripper_width.py consumes, so it is the only panel whose detections predict
    what the pipeline will get.
    """
    left = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
    tags = probe.annotate(left, mono) if probe is not None else None
    H, W = left.shape[:2]

    v = depth > 0
    n = np.zeros(depth.shape, np.float32)
    if v.any():
        lo, hi = st["z_lo_mm"], st["z_hi_mm"]
        n[v] = np.clip((depth[v].astype(np.float32) - lo) / max(hi - lo, 1), 0, 1) * 255
    dc = cv2.applyColorMap(n.astype(np.uint8), cv2.COLORMAP_TURBO)
    dc[~v] = (0, 0, 0)                                   # invalid stays black
    if dc.shape[:2] != (H, W):                           # decimated depth is smaller
        dc = cv2.resize(dc, (W, H), interpolation=cv2.INTER_NEAREST)
    view = np.hstack([left, _colour_panel(colour, W, H), dc])

    rec = st["recording"]
    banner = (0, 0, 200) if rec else (60, 60, 60)
    cv2.rectangle(view, (0, 0), (view.shape[1], 34), banner, -1)
    head = (f"[REC]  {st['session']}" if rec else "IDLE   -   press SPACE to start")
    cv2.putText(view, head, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)

    vf = st["valid_fraction"] * 100
    warn = vf < st["valid_warn_pct"]
    lines = [
        (f"kind      {st['kind']}", (255, 255, 255)),
        # The mode decides how close the recordist can stand, so MinZ rides with it.
        (f"mode      {st.get('mode', '-')}  MinZ {st['min_z_mm']:.0f} mm",
         (255, 220, 120)),
        (f"frames    {st['frames']}", (255, 255, 255)),
        (f"elapsed   {st['elapsed']:5.1f} s", (255, 255, 255)),
        (f"fps       {st['fps']:5.1f}", (255, 255, 255)),
        (f"depth ok  {vf:5.1f} %", (0, 0, 255) if warn else (0, 220, 0)),
    ]
    if tags is not None:
        ok = all(tags["seen"].values())
        lines.append((
            "tags      " + "  ".join(f"{i}:{'ok' if s else '--'}"
                                     for i, s in tags["seen"].items())
            + f"   {tags['rate'] * 100:3.0f}% last 2s",
            (0, 220, 0) if ok else (0, 0, 255)))
        ppm = tags["px_per_module"]
        if ppm is None:
            lines.append(("px/module    -    (need 3, want 5)", (140, 140, 140)))
        else:
            # 3 is roughly where 36h11 stops decoding and 5 is where the corners get
            # steady enough to trust the pose; both are rules of thumb, not measured
            # here — what IS measured is that a take at 2.9 detected 42% of frames.
            lines.append((f"px/module {ppm:5.1f}  (need 3, want 5)",
                          (0, 0, 255) if ppm < 3 else
                          (0, 200, 255) if ppm < 5 else (0, 220, 0)))
        z = tags["z_mm"]
        lines.append((f"tag Z     {z:5.0f} mm" if z is not None else "tag Z        -",
                      (255, 220, 120)))
    # NOTE: cv2.putText only draws ASCII with the Hershey fonts; anything else
    # comes out as "???", so keep every string here plain.
    for i, (txt, col) in enumerate(lines):
        cv2.putText(view, txt, (10, 62 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2)
    if warn:
        cv2.putText(view, f"< {st['valid_warn_pct']:.0f}%  too close? MinZ "
                          f"{st['min_z_mm']:.0f} mm",
                    (10, 62 + len(lines) * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 255), 2)
    keys = ("SPACE start/stop    K switch kind    "
            + ("E near/far mode    " if st.get("can_toggle") else "")
            + "Q quit")
    cv2.putText(view, keys, (10, view.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (200, 200, 200), 1)
    # At 1280x800 the composed view is 2560 px wide — wider than most screens, so the
    # window manager rescales every frame on the way to the display. Doing it here once
    # with INTER_AREA is cheaper and predictable. Text is drawn BEFORE the resize so it
    # scales with the image instead of getting thinner as the window shrinks.
    wmax = int(st.get("preview_width") or 0)
    if wmax and view.shape[1] > wmax:
        h = int(round(view.shape[0] * wmax / view.shape[1]))
        view = cv2.resize(view, (wmax, h), interpolation=cv2.INTER_AREA)
    return view


STREAM_WIDTH = {"400p": 640, "720p": 1280, "800p": 1280}


def find_device(mxid: str | None):
    """The DeviceInfo for `mxid`, or None to let DepthAI choose.

    Both the bench OAK-D-W (USB) and the C3 (PoE, on the lab network) answer device
    discovery, and dai.Device() with no argument takes whichever it enumerates first.
    Opening the wrong one records frames that session.json then describes with the other
    camera's intrinsics — silently, because both are OAK-D-W variants and both stream.
    Seen for real while benchmarking: a run reported 2.25 fps and 8.8% valid depth
    because it had grabbed the C3 over PoE.
    """
    import depthai as dai
    if not mxid:
        return None
    found = {d.getMxId(): d for d in dai.Device.getAllAvailableDevices()}
    if mxid not in found:
        raise RuntimeError(
            f"camera model names device {mxid}, which is not reachable. Visible: "
            + (", ".join(f"{k} ({v.protocol.name})" for k, v in found.items()) or "none")
            + ". Pass --any-device to record on whatever is attached, accepting that "
              "the camera model may not describe it.")
    return found[mxid]


def open_device(pipeline, src_cam: CameraModel, any_device: bool = False):
    """dai.Device pinned to the unit the source camera model was read off."""
    import depthai as dai
    info = None if any_device else find_device(src_cam.device_mxid)
    if info is None:
        return dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS)
    return dai.Device(pipeline, info, maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS)


def mode_geometry(cfg: dict, src_cam: CameraModel) -> tuple[int, float]:
    """(max_disparity, MinZ in mm) for a config.

    MinZ scales with the STREAMING fx, not the calibration's native fx. Using the
    native 1280 width here reported 224.5 mm while frames actually carried 201 mm.
    """
    max_disp = int(cfg["record"]["max_disparity"])
    width = STREAM_WIDTH[cfg["record"]["stereo_depth"]["mono_resolution"]]
    return max_disp, src_cam.min_z_m(max_disp, width=width) * 1000.0


def record_device(base_out: Path, modes: list, src_cam: CameraModel,
                  tgt_cam: CameraModel, args) -> list[Path]:
    """Live preview with SPACE to start/stop. Returns the session dirs written.

    `modes` is [(label, cfg), ...]; E cycles through them. Extended disparity is a
    pipeline-BUILD-time setting — there is no runtime config field for it — so
    switching means closing the device and reopening on the other pipeline, which is
    what test_oak_depth.py does too. Each mode is a full config file rather than a
    computed delta, so `--dry-run` and bench_stereo.py can address either one.

    Headless (--no-preview) records one fixed-length take instead, which is the path
    that can be exercised without a display attached.
    """
    written: list[Path] = []
    # Survives across device restarts, so a toggle does not silently reset the kind.
    state = {"mode_i": 0, "kind_i": KINDS.index(args.kind)}
    while True:
        action = _device_session(base_out, modes, src_cam, tgt_cam, args, state, written)
        if action != "toggle":
            break
        state["mode_i"] = (state["mode_i"] + 1) % len(modes)
        print(f"  mode -> {modes[state['mode_i']][0]}  (reopening device)")

    if not written:
        raise RuntimeError("no take was recorded")
    return written


def _device_session(base_out: Path, modes: list, src_cam: CameraModel,
                    tgt_cam: CameraModel, args, state: dict,
                    written: list) -> str:
    """One device open/close. Returns "quit" or "toggle"."""
    import depthai as dai

    label, cfg = modes[state["mode_i"]]
    pipeline = build_device_pipeline(cfg)
    max_disp, min_z_mm = mode_geometry(cfg, src_cam)

    with open_device(pipeline, src_cam, getattr(args, "any_device", False)) as device:
        mx = device.getMxId()
        pinned = "pinned" if mx == src_cam.device_mxid else "NOT the camera model's unit"
        print(f"device       : {device.getDeviceName()} usb={device.getUsbSpeed().name} "
              f"mxid={mx} ({pinned})")
        print(f"mode         : {label}  MinZ {min_z_mm:.1f} mm  "
              f"(max_disparity {max_disp}"
              + (f", E switches to {modes[(state['mode_i'] + 1) % len(modes)][0]})"
                 if len(modes) > 1 else ", no second mode configured)"))
        names = device.getOutputQueueNames()
        q_left = device.getOutputQueue("left", 4, False)
        q_depth = device.getOutputQueue("depth", 4, False)
        has_right = "right" in names
        q_right = device.getOutputQueue("right", 4, False) if has_right else None
        # A pair is complete when every stereo stream this config streams has arrived —
        # not a hardcoded 3, which would never complete with `streams:` minus right.
        stereo_qs = [(q_left, "left"), (q_depth, "depth")]
        if has_right:
            stereo_qs.append((q_right, "right"))
        n_stereo = len(stereo_qs)
        has_colour = "color" in names
        q_color = device.getOutputQueue("color", 8, False) if has_colour else None
        ccfg = cfg["record"].get("colour") or {}
        colour_encoded = has_colour and ccfg.get("encode", "mjpeg") == "mjpeg"
        colour_ext = "jpg" if colour_encoded else "png"
        # Built once: the detector and the marker geometry are config, not per-frame.
        probe = (None if (args.no_preview or args.no_tag_overlay)
                 else TagProbe(cfg["gripper"]["aruco"], src_cam))
        wcfg = cfg["record"].get("writer") or {}
        writer = FrameWriter(workers=int(wcfg.get("workers", 4)),
                             depth=int(wcfg.get("queue_depth", 96)))
        # Colour is a different sensor on its own clock, so it is paired by NEAREST
        # timestamp, not by "whatever arrived last". Taking the latest frame made the
        # colour run ~440 ms AHEAD of the stereo frame it was filed with, because the
        # stereo pair had been waiting in `pending` while colour kept arriving.
        colour_buf: list = []          # (t_unix, ImgFrame), oldest first
        # dai's clock is CLOCK_MONOTONIC; one constant offset puts frames on unix time.
        offset = time.time() - time.monotonic()
        pending: dict = {}
        # Depth is NOT the same size as the mono streams once decimation is on, and
        # everything downstream has to know that rather than assume it.
        sizes: dict = {}
        kind_i = state["kind_i"]
        recording = False
        rows: list[dict] = []
        out: Path | None = None
        t_start = 0.0
        stamps: list[float] = []
        vf = 0.0
        action = ""                    # "" while running, then "quit" or "toggle"
        headless = args.no_preview
        if headless:
            recording, out = True, next_session_dir(base_out, KINDS[kind_i])
            for sub in SUBDIRS(has_colour, has_right):
                (out / sub).mkdir(parents=True, exist_ok=True)
            t_start = time.monotonic()
            print(f"headless take: {out.name}  ({args.seconds} s)")

        def close_take():
            nonlocal recording, rows, out
            if out is not None and rows:
                writer.flush()                     # every frame on disk before the index
                if writer.errors:
                    print(f"  !! {len(writer.errors)} write errors, first: "
                          f"{writer.errors[0]}")
                write_session(out, rows, cfg, src_cam, tgt_cam, KINDS[kind_i],
                              "device", min_z_mm / 1000.0, max_disp, args,
                              sizes=sizes)
                written.append(out)
                print(f"  saved {out.name}: {len(rows)} frames")
            recording, rows, out = False, [], None

        while not action:
            if q_color is not None:
                while True:
                    c = q_color.tryGet()
                    if c is None:
                        break
                    colour_buf.append((c.getTimestamp().total_seconds() + offset, c))
                if len(colour_buf) > 12:
                    del colour_buf[:-12]
            # Drain each queue rather than taking one packet per iteration: with a
            # display attached the loop runs at the window's pace, and one-per-iteration
            # would let the host fall permanently behind the device.
            for q, key in stereo_qs:
                while True:
                    pkt = q.tryGet()
                    if pkt is None:
                        break
                    pending.setdefault(pkt.getSequenceNum(), {})[key] = pkt
            mono = depth = preview_cpkt = None
            ready = sorted(k for k, v in pending.items() if len(v) == n_stereo)
            for seq in ready:
                # Hold a pair back until a colour frame NEWER than it exists, so the
                # nearest-neighbour search can bracket it instead of always picking
                # the last one before. Skipped once the buffer is clearly behind.
                if q_color is not None and recording and colour_buf:
                    lt = pending[seq]["left"].getTimestamp().total_seconds() + offset
                    if colour_buf[-1][0] < lt and len(pending) < 6:
                        continue
                pair = pending.pop(seq)
                lf, df, rf = pair["left"], pair["depth"], pair.get("right")
                mono, depth = lf.getFrame(), df.getFrame()
                vf = float(np.count_nonzero(depth)) / depth.size
                # Hoisted out of the `if recording:` below because the preview needs it
                # too: the colour panel is paired to the displayed stereo frame by the
                # same nearest-timestamp rule the recording uses, not by "newest".
                t_unix = lf.getTimestamp().total_seconds() + offset
                if colour_buf:
                    preview_cpkt = min(colour_buf, key=lambda cb: abs(cb[0] - t_unix))[1]
                if not sizes:
                    sizes = {"left": [mono.shape[1], mono.shape[0]],
                             "depth": [depth.shape[1], depth.shape[0]]}
                    if rf is not None:
                        sizes["right"] = [mono.shape[1], mono.shape[0]]
                if recording:
                    stamp = f"{t_unix:.6f}"
                    png = [cv2.IMWRITE_PNG_COMPRESSION, 1]
                    # .copy() because the arrays alias the device packet's buffer, which
                    # is recycled as soon as this loop drops its reference — the writer
                    # thread would otherwise encode whatever landed there next.
                    writer.put(out / "left" / f"{stamp}.png", mono.copy(), png)
                    writer.put(out / "depth" / f"{stamp}.png", depth.copy(), png)
                    if rf is not None:
                        writer.put(out / "right" / f"{stamp}.png", rf.getFrame().copy(), png)
                    cfile, ct, skew = "", "", ""
                    if colour_buf:
                        ct_unix, cframe = min(colour_buf, key=lambda cb: abs(cb[0] - t_unix))
                        cfile = f"color/{stamp}.{colour_ext}"
                        writer.put(out / cfile,
                                   cframe.getData().tobytes() if colour_encoded
                                   else cframe.getCvFrame().copy(), png)
                        ct = f"{ct_unix:.6f}"
                        skew = f"{(ct_unix - t_unix) * 1000:.1f}"
                    rows.append({
                        "idx": len(rows) + 1, "t_unix": stamp,
                        "left_file": f"left/{stamp}.png",
                        "depth_file": f"depth/{stamp}.png",
                        "right_file": (f"right/{stamp}.png" if rf is not None
                                       else ""),
                        "color_file": cfile, "color_t_unix": ct,
                        "color_skew_ms": skew,
                        "depth_valid_fraction": f"{vf:.6f}",
                        "source": "device",
                    })
                    stamps.append(time.monotonic())
            for seq in [k for k in pending if k < (max(pending) - 8 if pending else 0)]:
                pending.pop(seq, None)

            stamps = [s for s in stamps if time.monotonic() - s < 2.0]
            fps = len(stamps) / 2.0

            if headless:
                if time.monotonic() - t_start >= args.seconds:
                    close_take()
                    action = "quit"
                time.sleep(0.001)
                continue

            if mono is not None:
                st = {"recording": recording, "kind": KINDS[kind_i],
                      "session": out.name if out else "",
                      "frames": len(rows),
                      "elapsed": (time.monotonic() - t_start) if recording else 0.0,
                      "fps": fps, "valid_fraction": vf,
                      "valid_warn_pct": float(args.valid_warn),
                      "min_z_mm": min_z_mm, "mode": label,
                      "can_toggle": len(modes) > 1,
                      "preview_width": args.preview_width,
                      "z_lo_mm": min_z_mm, "z_hi_mm": float(args.preview_far_mm)}
                # One decode per DISPLAYED frame, not per packet drained. At 400p the
                # mono panel is 640 wide and colour arrives at 1280, so
                # IMREAD_REDUCED_COLOR_2 decodes straight to the panel size — the
                # cheapest JPEG path there is, and it skips the resize as well.
                colour_view = None
                if preview_cpkt is not None:
                    if colour_encoded:
                        buf = np.frombuffer(preview_cpkt.getData(), np.uint8)
                        flag = (cv2.IMREAD_REDUCED_COLOR_2 if mono.shape[1] <= 640
                                else cv2.IMREAD_COLOR)
                        colour_view = cv2.imdecode(buf, flag)
                    else:
                        colour_view = preview_cpkt.getCvFrame()
                cv2.imshow("umi_handheld record",
                           draw_overlay(mono, colour_view, depth, st, probe))

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                close_take()
                action = "quit"
            elif k == ord(" "):
                if recording:
                    close_take()
                else:
                    out = next_session_dir(base_out, KINDS[kind_i])
                    for sub in SUBDIRS(has_colour, has_right):
                        (out / sub).mkdir(parents=True, exist_ok=True)
                    rows, stamps, t_start, recording = [], [], time.monotonic(), True
                    print(f"  [REC] {out.name}")
            elif k == ord("k") and not recording:
                kind_i = (kind_i + 1) % len(KINDS)
                print(f"  kind -> {KINDS[kind_i]}")
            elif k == ord("e"):
                # Not mid-take: the two modes have different MinZ and different depth
                # geometry, and one session.json can only describe one of them. Same
                # rule as K, which is also refused while recording.
                if recording:
                    print("  press SPACE to close the take before switching mode")
                elif len(modes) > 1:
                    action = "toggle"
                else:
                    print("  no second mode: pass --config-extended, or use a config "
                          "with a <name>_extended.yaml beside it")
        cv2.destroyAllWindows()
        writer.close()

    state["kind_i"] = kind_i           # survive the device restart on a mode switch
    return action or "quit"


# ------------------------------------------------------------------------ main

def write_session(out: Path, rows: list, cfg: dict, src_cam: CameraModel,
                  tgt_cam: CameraModel, kind: str, source: str, min_z: float,
                  max_disp: int, args, sizes: dict | None = None) -> None:
    """frames.csv + session.json. One place, so every take is described the same way."""
    with (out / "frames.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FRAME_FIELDS)
        w.writeheader()
        w.writerows(rows)
    vf = [float(r["depth_valid_fraction"]) for r in rows]
    session = {
        "kind": kind,
        "source": source,
        "source_input": str(args.input) if source == "synthetic" else None,
        "frames": len(rows),
        "fps": cfg["record"]["fps"],
        "depth_scale": cfg["record"]["depth_scale"],
        "depth_units": "millimetre, uint16, 0 = invalid",
        "depth_valid_fraction": {"min": min(vf), "max": max(vf),
                                 "mean": float(sum(vf) / len(vf))},
        "min_z_m": min_z,
        "min_z_provenance": (
            "derived: fx*baseline/max_disparity at the DISPARITY grid width "
            "(decimation shrinks the depth image afterwards and does not change it). "
            "Confirmed on this device 2026-08-04 by setting the threshold floor BELOW "
            "the derived value and observing where returns actually stop: 112 mm over "
            "297 frames against 112.3 derived (sessions/demonstration_0010), 449 mm "
            "over 296 frames against 449.0 derived (sessions/demonstration_0012), and "
            "224 mm over 228 extended-disparity frames against 224.5 derived, floor at "
            "150 mm (sessions/demonstration_0019). Three points across both disparity "
            "ranges and two resolutions."),
        "max_disparity": max_disp,
        "stream_sizes": sizes,
        "camera_model_source": src_cam.provenance(),
        "camera_model_target": tgt_cam.provenance(),
        "config": cfg["_path"],
        "resolved_config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "device_path_executed": source == "device",
        "warning": ("Synthetic session. No hardware was connected."
                    if source == "synthetic" else None),
    }
    (out / "session.json").write_text(json.dumps(session, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--config-extended", default=None,
                    help="config the E key switches to. Defaults to "
                         "<config>_extended.yaml if that file exists; pass 'none' to "
                         "disable the key.")
    ap.add_argument("--source", choices=["synthetic", "device"], default="synthetic")
    ap.add_argument("--input", type=Path, default=REPO / "captures/synthetic_0001",
                    help="synthetic capture directory (--source synthetic)")
    ap.add_argument("--out", type=Path, default=REPO / "sessions/session_0001",
                    help="session directory (--source synthetic)")
    ap.add_argument("--sessions-dir", type=Path, default=REPO / "sessions",
                    help="where auto-numbered device takes go")
    ap.add_argument("--kind", choices=list(KINDS), default="demonstration")
    ap.add_argument("--seconds", type=float, default=10.0,
                    help="length of the headless device take (--no-preview)")
    ap.add_argument("--no-preview", action="store_true",
                    help="record one fixed-length take with no window (no display needed)")
    ap.add_argument("--valid-warn", type=float, default=40.0,
                    help="depth-valid %% below which the preview turns red")
    ap.add_argument("--preview-far-mm", type=float, default=2000.0,
                    help="far end of the preview depth colourmap")
    ap.add_argument("--preview-width", type=int, default=1600,
                    help="downscale the preview window to this width (0 = native). "
                         "At 1280x800 the three panels are 3840 px wide.")
    ap.add_argument("--no-tag-overlay", action="store_true",
                    help="skip AprilTag detection on the preview. It runs once per "
                         "displayed frame and costs time; turn it off if the preview "
                         "is the bottleneck.")
    ap.add_argument("--copy", action="store_true",
                    help="copy frames instead of symlinking them (synthetic path)")
    ap.add_argument("--any-device", action="store_true",
                    help="record on whatever DepthAI device is found instead of the one "
                         "the source camera model was read off. The session's camera "
                         "model may then not describe the frames.")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg = load_pipeline_config(a.config)
    src_cam = CameraModel.load(cfg["cameras"]["source"])
    tgt_cam = CameraModel.load(cfg["cameras"]["target"])
    max_disp = int(cfg["record"]["max_disparity"])
    stream_w = STREAM_WIDTH[cfg["record"]["stereo_depth"]["mono_resolution"]]
    min_z = src_cam.min_z_m(max_disp, width=stream_w)

    # The E-key mode. Auto-detecting the sibling means the documented invocation stays
    # one --config, and a profile without a near variant simply has no second mode.
    ext_path = a.config_extended
    if ext_path is None:
        sibling = Path(str(cfg["_path"]).removesuffix(".yaml") + "_extended.yaml")
        if not sibling.is_absolute():
            sibling = REPO / sibling
        ext_path = str(sibling) if sibling.exists() else "none"
    cfg_ext = None if str(ext_path).lower() == "none" else load_pipeline_config(ext_path)

    print(f"config       : {cfg['_path']}")
    print(f"source model : {src_cam.rel}  ({src_cam.name}, medium={src_cam.medium}, "
          f"verified={src_cam.verified}, rectified={src_cam.rectified})")
    print(f"target model : {tgt_cam.rel}  ({tgt_cam.name}, medium={tgt_cam.medium}, "
          f"verified={tgt_cam.verified}, rectified={tgt_cam.rectified})")
    print(f"source       : {a.source}"
          + (f"  <- {a.input}" if a.source == "synthetic" else ""))
    print(f"kind         : {a.kind}")
    print(f"MinZ         : {min_z*1000:.1f} mm  at {cfg['record']['stereo_depth']['mono_resolution']} "
          f"({stream_w} px wide)  [derived: fx*B/{max_disp}, not measured]")
    if cfg_ext is not None:
        _, ext_min_z_mm = mode_geometry(cfg_ext, src_cam)
        print(f"E mode       : {cfg_ext['_path']}  MinZ {ext_min_z_mm:.1f} mm")
    else:
        print("E mode       : none (no <config>_extended.yaml, E does nothing)")
    print(f"out          : {a.out if a.source == 'synthetic' else a.sessions_dir}")

    if a.dry_run:
        problems = []
        if a.source == "synthetic":
            for need in ("frames.csv", "left", "depth"):
                if not (a.input / need).exists():
                    problems.append(f"missing {a.input / need}")
        # Both modes are checked: a broken E config should not wait until the operator
        # presses E mid-session to announce itself.
        for c in (cfg, cfg_ext):
            if c is None:
                continue
            for key in ("record", "gripper", "warp", "zarr"):
                if key not in c:
                    problems.append(f"{c['_path']}: config section '{key}' missing")
            sd, md = c["record"]["stereo_depth"], int(c["record"]["max_disparity"])
            if sd["extended_disparity"] != (md == 190):
                problems.append(f"{c['_path']}: extended_disparity disagrees with "
                                "max_disparity")
            # extended x 3-bit subpixel is 1520 levels; the median filter caps at 1024
            # and the device disables it, logging an error every frame.
            levels = md * (2 ** int(sd.get("subpixel_bits", 3)) if sd["subpixel"] else 1)
            if str(sd.get("median_filter", "off")).lower() != "off" and levels > 1024:
                problems.append(f"{c['_path']}: median_filter with {levels} disparity "
                                "levels exceeds the 1024 limit — device will drop it")
        print("\n--dry-run: " + ("OK, nothing written" if not problems
                                  else "PROBLEMS:\n  " + "\n  ".join(problems)))
        return 1 if problems else 0

    if a.source == "device":
        modes = [(cfg.get("profile", "base"), cfg)]
        if cfg_ext is not None:
            modes.append((cfg_ext.get("profile", "extended"), cfg_ext))
        written = record_device(a.sessions_dir, modes, src_cam, tgt_cam, a)
        print("\nsessions:")
        for p in written:
            print(f"  {p}")
        return 0

    for sub in ("left", "depth"):
        (a.out / sub).mkdir(parents=True, exist_ok=True)
    rows = record_synthetic(a.input, a.out, cfg, src_cam, link=not a.copy)
    write_session(a.out, rows, cfg, src_cam, tgt_cam, a.kind, "synthetic",
                  min_z, max_disp, a)
    vf = [float(r["depth_valid_fraction"]) for r in rows]
    print(f"\nwrote {len(rows)} frames  depth valid "
          f"{min(vf)*100:.1f}%-{max(vf)*100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
