#!/usr/bin/env python3
"""
c3_encode_quality.py — what each colour encoder costs on the link, and what it
costs you in the pixels.

c3_bench.py answers "how many fps and how much latency does this configuration
deliver". It keeps no pixels, so it cannot answer the question that follows: at
the *same* frame rate, is the cheaper stream still good enough to run AprilTag
pose and an ORB front end on? This script is its sibling. It sweeps colour
encoder settings exactly the way c3_bench sweeps resolutions — one connection per
setting, one CSV row per setting — but it also keeps frames and scores them
against a raw reference run.

    # the default ladder: raw reference + MJPEG q90 + H.264/H.265 at 4 Mbit/s
    python c3_camera/c3_encode_quality.py --out c3_camera/encode_quality/run1

    # the question that actually matters: which curve is on top at a fixed budget
    python c3_camera/c3_encode_quality.py --codec none,mjpeg,h264,h265 \\
        --mjpeg-quality 80,90,97 --bitrate-kbps 2000,4000,8000,16000 \\
        --target-fps 19 --budget-mbps 25

    # plan the sweep without touching the camera
    python c3_camera/c3_encode_quality.py --codec none,mjpeg,h265 --dry-run

    # re-score an existing capture (no hardware, repeat as often as you like)
    python c3_camera/c3_encode_quality.py --offline c3_camera/encode_quality/run1

Why the encoder question is worth asking at all
-----------------------------------------------
At the *current* dataset operating point the colour stream is not the problem:
measured 41.2 kB/frame at 20 fps = 6.8 Mbit/s, i.e. 14% of a 48 Mbit/s total, of
which depth's uint16 is 86%. A perfect codec that made colour free would recover
7.4% of the link and buy exactly zero fps. The encoder only becomes binding when
colour resolution goes up — 1080p MJPEG q90 at 20 fps needs 61.8 Mbit/s and does
not fit alongside depth, while 1080p H.265 at 16 Mbit/s does. So the honest
question is not "which codec makes smaller files" (true by definition, since the
bitrate is an input parameter, not a result) but **"for a given Mbit/s budget,
which (codec, rate) gives the most usable detail per second"**. That is what the
ranking at the bottom of the output answers, and it is why the table is ranked by
a quality metric at a target fps rather than by bandwidth.

Frame correspondence — read this before quoting a PSNR number
--------------------------------------------------------------
A DepthAI pipeline is fixed at device boot, so every encoder setting is a
separate connection and therefore a separate run. There is no way, in this
script, for the reference frame and the test frame to be the same photons. What
it does instead is the strategy the design calls for when paired dual-output is
not available: the raw (`--codec none`) run is collapsed into a **temporal median
stack**, and every frame of every run is scored against that single reference
image. Shot noise averages out of the reference, so it is closer to the scene
than any single raw frame is — but it is still not the same frame, and the number
below is NOT a same-frame PSNR. `CORRESPONDENCE_NOTE` is printed with every
summary and copied into meta.json so the number cannot travel without its
caveat. Use these numbers to RANK settings, never as an absolute fidelity claim.

The raw run is scored against the median too, and its row is the **noise floor**:
any encoder that lands near it is at or below the sensor's own frame-to-frame
variation, and the difference is not measurable by this method.

Safety
------
`--dry-run` and `--offline` never construct a C3Source and never open the device.
Everything else in this file does connect — an OAK accepts one pipeline at a
time, so running it will take the camera from whoever else has it.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from c3_camera import config as C
from c3_camera import preflight as PF
from c3_camera.c3_extract_rgb_video import extract_rgb_video
from c3_camera.config import STREAM_COLOR, STREAM_DEPTH, StreamConfig
from c3_camera.dataset import RGB_VIDEO_FIELDS

# =============================================================================
# The caveats. Printed, not just commented — a table without them is misleading.
# =============================================================================
CORRESPONDENCE_NOTE = (
    "FRAME CORRESPONDENCE: these are NOT same-frame PSNR/SSIM numbers. Each "
    "encoder setting is a separate device connection (a DepthAI pipeline is "
    "fixed at boot), so reference and test frames are different exposures of a "
    "static scene. Every frame is scored against a temporal MEDIAN STACK of the "
    "raw run. Rank with these numbers; do not quote them as absolute fidelity."
)

CAVEATS = (
    CORRESPONDENCE_NOTE,
    "The raw arm's own row is the NOISE FLOOR, not a perfect score: it is scored "
    "against the median of its own run (disjoint frames), so its PSNR is what "
    "sensor noise alone costs. An encoder at that level is indistinguishable "
    "from raw by this method.",
    "Median stacking removes noise from the reference but not from the test "
    "frames. An encoder that denoises can therefore score ABOVE the raw arm. "
    "That is a real property of the encoder, not an error, but it means high "
    "PSNR here does not mean 'closer to the truth'.",
    "The reference is the ISP output, not the sensor: demosaic, AWB, sharpening "
    "and NV12 4:2:0 chroma subsampling are common-mode and invisible here. "
    "Metrics are computed on the LUMA plane, which NV12 keeps at full "
    "resolution, so chroma subsampling does not contaminate them.",
    "The ISP output is common-mode but the HOST DECODE is not, and it puts a "
    "hard ceiling on the H.264/H.265 rows. Raw arrives as NV12 and is converted "
    "by OpenCV (COLOR_YUV2BGR_NV12, limited range); MJPEG is decoded by libjpeg; "
    "H.26x is decoded by ffmpeg/swscale inside c3_extract_rgb_video and re-read "
    "from PNG. Measured on this host: a BIT-EXACT H.264 stream scored through "
    "that path against the raw arm's own conversion of the same pixels gives "
    "PSNR 45.6 dB, SSIM 0.9962, ORB inlier 0.916 — at ZERO codec loss. So an "
    "H.26x row can never exceed those values however high the bitrate, the top "
    "rungs of a wide ladder (8-16 Mbit/s) saturate against them, and any "
    "MJPEG-vs-H.26x comparison near the top is reading the decoder, not the "
    "encoder. Compare rungs WITHIN a codec; treat across-codec gaps smaller than "
    "this offset as unresolved. MJPEG paid no such offset in that test (as-scored "
    "matched its true loss to ~0.1 dB), but that assumed the encoder converts "
    "limited-range NV12 to full-range JFIF. If this camera's fixed-function JPEG "
    "block instead compresses the ISP planes verbatim, MJPEG picks up a floor "
    "near 28 dB / SSIM 0.82 and its whole column is unusable — visible on the "
    "first real capture as an MJPEG q97 row that cannot beat about 28 dB.",
    "Settings run in ladder order (none, then MJPEG, then H.264, then H.265) and "
    "the reference comes from the raw run, which is FIRST. A wide ladder takes "
    "~10 minutes, so codec is perfectly confounded with elapsed time: any drift "
    "in illumination, exposure or the scene is charged to whatever ran late, and "
    "the H.265 rows are always the ones that ran last. static_scene_warning() "
    "only inspects a few seconds inside the raw run and cannot see a drift that "
    "slow. To bracket it, run the ladder twice with --codec ordered differently, "
    "or capture a second raw arm at the end and compare the two rows.",
    "Bandwidth is an INPUT for H.264/H.265, not a result: given a target "
    "bitrate the encoder raises QP until it hits it. 'H.265 was smaller at the "
    "same settings' compares nothing. Only the rate-vs-quality curves compare.",
    "MJPEG's knob is quality, not bitrate, so its bandwidth is NOT controlled — "
    "measured compression on this camera ranged 4.05:1 to 6.35:1 across "
    "identical sessions. The burst column (p95/mean bytes per frame) is where "
    "that risk shows up on a link with a hard ceiling.",
    "H.264/H.265 streams carry a 1.5 s blocking queue on both the device "
    "(pipeline.py) and the host (source.py) to protect inter-frame integrity, "
    "while raw/MJPEG use a non-blocking depth-1 queue. Latency is therefore "
    "compared across DIFFERENT queue policies: under saturation raw/MJPEG shows "
    "up as drops and H.26x shows up as latency.",
    "Laplacian variance is reported for reference only and is NEVER used for "
    "ranking: it is non-monotonic under compression (blocking artefacts add "
    "high-frequency energy), and its direction flips with scene texture.",
    "A static scene is the BEST case for inter-frame prediction, so H.264/H.265 "
    "are flattered here. Re-run with the rig moving before trusting these "
    "numbers for a swimming vehicle — backscatter under motion is exactly where "
    "P-frames fail.",
    "pipeline.py pins H.264 to the BASELINE profile (no CABAC), which is "
    "measurably worse rate-distortion than MAIN/HIGH. The H.264 rows here are a "
    "lower bound on what this hardware can do.",
)


# =============================================================================
# CSV schemas — c3_bench.py's FIELDS, plus the encoder axis and the quality
# aggregates. One row per setting, which is what makes the two files readable
# side by side.
# =============================================================================
FIELDS = (
    "timestamp", "ok", "error", "run", "label",
    # encoder axis
    "codec", "mjpeg_quality", "bitrate_kbps", "keyframe",
    # fixed pipeline settings, echoed so a row is self-describing
    "color_res", "isp_scale", "color_out", "mono_res", "requested_fps",
    "streams", "scene",
    # link cost
    "est_mbps", "measured_mbps", "budget_frac",
    "color_fps", "color_lat_p50", "color_lat_p95", "color_lat_max",
    "color_frames", "color_dropped", "color_drop_rate",
    "color_kb_frame", "color_kb_frame_p95", "burstiness", "color_mbps",
    "depth_fps", "depth_kb_frame", "depth_mbps",
    # capture bookkeeping
    "frames_saved", "frames_extracted", "extraction_ok",
    "loop_hz", "duration_s",
    # quality (filled in by the scoring pass, blank until then)
    "scored_frames", "under_textured",
    "psnr_y_db", "psnr_y_sd", "psnr_y_p05", "ssim_y", "ssim_y_sd", "ssim_y_p05",
    "orb_n", "orb_inlier_rate", "orb_inlier_sd", "orb_inlier_p05",
    "tags_detected", "tags_known", "tags_false", "tag_backend",
    "lap_var", "quality_note",
)

SCORE_FIELDS = (
    "run", "label", "codec", "frame_idx", "file",
    "psnr_y_db", "ssim_y", "orb_n", "orb_inlier_rate",
    "tags_detected", "tags_known", "tags_false", "lap_var",
)

# Per-run frame index. Written for every codec so the scoring pass has one
# uniform entry point; H.264/H.265 runs additionally get the rgb_video.csv that
# c3_extract_rgb_video.py requires, because that file's schema belongs to the
# dataset path and is not ours to redefine.
FRAME_FIELDS = ("idx", "seq", "t_device", "latency_ms", "wire_bytes",
                "exposure_ms", "file")

# Metrics that may decide a ranking. Laplacian variance is deliberately absent:
# it is non-monotonic under compression, so ranking by it can invert the answer.
RANKABLE_METRICS = ("orb_inlier_rate", "ssim_y", "psnr_y_db")

# The per-frame scatter that belongs with each ranking metric. The DECISION line
# quotes it because adjacent rungs of a ladder routinely land inside it, and a
# winner announced without it reads as a result when it is a coin toss.
SCATTER_OF = {"orb_inlier_rate": "orb_inlier_sd", "ssim_y": "ssim_y_sd",
              "psnr_y_db": "psnr_y_sd"}


# =============================================================================
# Encoder settings — the sweep axis
# =============================================================================
@dataclass(frozen=True)
class EncodeSetting:
    """One point on the encoder ladder.

    The axes are folded per codec on purpose: a bitrate means nothing to MJPEG
    and a JPEG quality means nothing to H.26x, so the product of all axes would
    spend minutes re-running identical configurations under different names.
    """

    codec: str                      # none | mjpeg | h264 | h265
    quality: int | None = None      # MJPEG only
    bitrate_kbps: int | None = None  # h264/h265 only
    keyframe: int | None = None     # h264/h265 only; 0 = one per second

    @property
    def label(self) -> str:
        if self.codec == "none":
            return "none (raw NV12)"
        if self.codec == "mjpeg":
            return f"mjpeg q{self.quality}"
        kf = "1/s" if not self.keyframe else str(self.keyframe)
        return f"{self.codec} {self.bitrate_kbps}k kf={kf}"

    @property
    def slug(self) -> str:
        if self.codec == "none":
            return "none"
        if self.codec == "mjpeg":
            return f"mjpeg_q{self.quality}"
        return f"{self.codec}_{self.bitrate_kbps}k_kf{self.keyframe}"

    @property
    def interframe(self) -> bool:
        return self.codec in ("h264", "h265")

    def apply(self, base: dict) -> dict:
        out = dict(base)
        out["color_encode"] = self.codec
        if self.codec == "mjpeg":
            out["mjpeg_quality"] = int(self.quality)
        if self.interframe:
            out["video_bitrate_kbps"] = int(self.bitrate_kbps)
            out["video_keyframe_frequency"] = int(self.keyframe)
        return out

    def to_dict(self) -> dict:
        return {"codec": self.codec, "quality": self.quality,
                "bitrate_kbps": self.bitrate_kbps, "keyframe": self.keyframe,
                "label": self.label, "slug": self.slug}


def _csv_list(text: str) -> list[str]:
    return [s.strip() for s in str(text).split(",") if s.strip()]


def _parse_isp(text: str):
    if text.lower() in ("none", "off", "1", "1/1"):
        return None
    n, d = text.split("/", 1)
    return int(n), int(d)


def expand_settings(codecs, qualities, bitrates, keyframes) -> list[EncodeSetting]:
    """Fold the axes per codec and drop duplicates, preserving order."""
    out: list[EncodeSetting] = []
    for codec in codecs:
        if codec not in ("none", "mjpeg", "h264", "h265"):
            raise ValueError(f"unknown codec {codec!r}; "
                             "choose from none,mjpeg,h264,h265")
        if codec == "none":
            out.append(EncodeSetting("none"))
        elif codec == "mjpeg":
            out.extend(EncodeSetting("mjpeg", quality=int(q)) for q in qualities)
        else:
            for br in bitrates:
                for kf in keyframes:
                    out.append(EncodeSetting(codec, bitrate_kbps=int(br),
                                             keyframe=int(kf)))
    return list(dict.fromkeys(out))


def parse_setting(text: str) -> EncodeSetting:
    """'none' | 'mjpeg@90' | 'h265@8000' | 'h265@8000/kf1' -> EncodeSetting."""
    spec = text.strip()
    keyframe = 0
    if "/kf" in spec:
        spec, kf = spec.split("/kf", 1)
        keyframe = int(kf)
    if "@" in spec:
        codec, value = spec.split("@", 1)
    else:
        codec, value = spec, None
    codec = codec.strip().lower()
    if codec == "none":
        return EncodeSetting("none")
    if codec == "mjpeg":
        return EncodeSetting("mjpeg", quality=int(value if value else 90))
    if codec in ("h264", "h265"):
        if value is None:
            raise ValueError(f"{codec} needs a bitrate, e.g. {codec}@8000")
        return EncodeSetting(codec, bitrate_kbps=int(value), keyframe=keyframe)
    raise ValueError(f"unknown codec {codec!r} in {text!r}")


# =============================================================================
# Annex-B keyframe detection
#
# source.py reads frame_type from `ImgFrame.getFrameType()`, which does not exist
# on depthai 2.32's ImgFrame (only on EncodedFrame), so every packet is typed "".
# Recording code that gates on "start at the first I frame" therefore never
# starts. The NAL type is in the bytes we already hold, so read it from there and
# stay independent of that bug.
#
# bytes.find() rather than a per-byte Python loop: the loop in source.py costs
# ~19 ms on a 190 kB packet, which at 20 fps is a third of a core spent inside
# the very poll thread whose latency this script is trying to measure.
# =============================================================================
def _annexb_nal_types(data: bytes, codec: str):
    i = data.find(b"\x00\x00\x01")
    while i != -1:
        payload = i + 3
        if payload < len(data):
            b0 = data[payload]
            yield (b0 & 0x1F) if codec == "h264" else ((b0 >> 1) & 0x3F)
        i = data.find(b"\x00\x00\x01", payload)


def annexb_frame_type(data: bytes, codec: str) -> str:
    """'I' if the packet carries a keyframe NAL, else 'P'.

    H.264 nal_unit_type 5 is an IDR slice; H.265 types 16-23 are the IRAP class
    (BLA/IDR/CRA), any of which is a valid random-access point.
    """
    for nal in _annexb_nal_types(data or b"", codec):
        if codec == "h264" and nal == 5:
            return "I"
        if codec == "h265" and 16 <= nal <= 23:
            return "I"
    return "P"


# =============================================================================
# Quality maths — all of it on the luma plane, all of it cv2/numpy only
# (this venv has no skimage, no scipy, no matplotlib)
# =============================================================================
def to_luma(image: np.ndarray) -> np.ndarray:
    """BT.601 luma as float64. Colour is dropped on purpose: NV12 carries luma at
    full resolution and chroma at quarter, so luma is the plane where a
    difference is attributable to the encoder rather than to the wire format."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.astype(np.float64)


def psnr_db(ref: np.ndarray, test: np.ndarray, peak: float = 255.0) -> float:
    """Peak signal-to-noise ratio in dB; +inf for bit-identical inputs."""
    if ref.shape != test.shape:
        raise ValueError(f"shape mismatch: {ref.shape} vs {test.shape}")
    mse = float(np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2))
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10(peak * peak / mse))


def ssim(ref: np.ndarray, test: np.ndarray, peak: float = 255.0) -> float:
    """Mean gaussian-weighted SSIM (11x11, sigma 1.5), Wang et al. 2004.

    Hand-rolled because skimage is not in this venv and cv2.quality is not in
    this (non-contrib) build. Border handling is cv2's BORDER_REFLECT_101 rather
    than skimage's valid-region crop, so values differ from skimage in the ~1e-3
    range — irrelevant for ranking, worth knowing before comparing to a paper.
    """
    if ref.shape != test.shape:
        raise ValueError(f"shape mismatch: {ref.shape} vs {test.shape}")
    a = ref.astype(np.float64)
    b = test.astype(np.float64)
    c1, c2 = (0.01 * peak) ** 2, (0.03 * peak) ** 2
    k, s = (11, 11), 1.5

    mu_a = cv2.GaussianBlur(a, k, s)
    mu_b = cv2.GaussianBlur(b, k, s)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    var_a = cv2.GaussianBlur(a * a, k, s) - mu_a2
    var_b = cv2.GaussianBlur(b * b, k, s) - mu_b2
    cov = cv2.GaussianBlur(a * b, k, s) - mu_ab

    num = (2.0 * mu_ab + c1) * (2.0 * cov + c2)
    den = (mu_a2 + mu_b2 + c1) * (var_a + var_b + c2)
    return float(np.mean(num / den))


def laplacian_var(image: np.ndarray) -> float:
    """Global high-frequency energy. REFERENCE ONLY — see CAVEATS: compression
    can raise it (blocking artefacts are high frequency), so it does not rank."""
    g = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g.astype(np.uint8), cv2.CV_64F).var())


class OrbScorer:
    """ORB keypoint count and inlier rate against a fixed reference image.

    Count alone is a weak metric: measured on this camera's frames, pushing
    compression to 8.7 kB moved the count 1833 -> 1816 while the inlier rate
    collapsed 1.000 -> 0.827. Blocking artefacts manufacture corners, so a count
    can even rise as quality falls. The inlier rate cannot be gamed that way,
    because a manufactured corner has no match in the reference.

    Static rig plus static scene means the true homography is the identity, so
    no RANSAC is needed (and no degenerate-homography risk is taken): a match is
    an inlier when the two keypoints are within `radius_px` of each other.
    """

    def __init__(self, reference: np.ndarray, nfeatures: int = 2000,
                 radius_px: float = 1.5):
        self.orb = cv2.ORB_create(nfeatures=nfeatures)
        self.radius = float(radius_px)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.ref_gray = self._gray(reference)
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(self.ref_gray, None)

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image if image.dtype == np.uint8 else image.astype(np.uint8)

    @property
    def reference_keypoints(self) -> int:
        return len(self.ref_kp or [])

    def score(self, image: np.ndarray) -> tuple[int, float]:
        kp, des = self.orb.detectAndCompute(self._gray(image), None)
        n = len(kp or [])
        if not n or self.ref_des is None or des is None or not len(self.ref_kp):
            return n, 0.0
        inliers = 0
        for m in self.matcher.match(self.ref_des, des):
            (x0, y0) = self.ref_kp[m.queryIdx].pt
            (x1, y1) = kp[m.trainIdx].pt
            if (x0 - x1) ** 2 + (y0 - y1) ** 2 <= self.radius ** 2:
                inliers += 1
        return n, inliers / max(1, min(len(self.ref_kp), n))


class TagDetector:
    """AprilTag detection, degrading gracefully when no library is available.

    Detection *count* is a contaminated metric on its own: measured on this
    camera's frames, a compressed copy detected 13 tags where the reference
    detected 11. Compression artefacts manufacture false positives, and a false
    tag ID is far worse for a pose pipeline than a missing one. So when a tag map
    is supplied, known and false detections are counted separately.
    """

    def __init__(self, family: str = "36h11", known_ids: set[int] | None = None):
        self.family = family
        self.known_ids = known_ids
        self.backend = "none"
        self._impl = None
        try:                                    # preferred: the real AprilTag lib
            import pupil_apriltags

            self._impl = pupil_apriltags.Detector(families=f"tag{family}")
            self.backend = "pupil_apriltags"
            return
        except Exception:                       # noqa: BLE001
            pass
        try:                                    # fallback: OpenCV's ArUco port
            dict_id = getattr(cv2.aruco, f"DICT_APRILTAG_{family}")
            params = cv2.aruco.DetectorParameters()
            # Subpixel corner refinement, which is what a pose pipeline consumes.
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
            self._impl = cv2.aruco.ArucoDetector(
                cv2.aruco.getPredefinedDictionary(dict_id), params)
            self.backend = "cv2.aruco"
        except Exception:                       # noqa: BLE001
            self._impl = None
            self.backend = "none"

    @property
    def available(self) -> bool:
        return self._impl is not None

    def detect(self, image: np.ndarray) -> tuple[int, int, int]:
        """Return (detected, known, false). All zero when no backend exists."""
        if self._impl is None:
            return 0, 0, 0
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = gray if gray.dtype == np.uint8 else gray.astype(np.uint8)
        if self.backend == "pupil_apriltags":
            ids = [int(d.tag_id) for d in self._impl.detect(gray)]
        else:
            _, ids_arr, _ = self._impl.detectMarkers(gray)
            ids = [] if ids_arr is None else [int(i) for i in ids_arr.ravel()]
        if self.known_ids is None:
            return len(ids), len(ids), 0
        known = sum(1 for i in ids if i in self.known_ids)
        return len(ids), known, len(ids) - known


def median_reference(images: list[np.ndarray]) -> np.ndarray:
    """Temporal median of a static scene: the closest thing to a clean reference
    that separate runs allow. Median, not mean, so a single outlier frame (an
    auto-exposure step, a passing particle) cannot drag the reference."""
    if not images:
        raise ValueError("no reference frames")
    stack = np.stack([im.astype(np.uint8) for im in images], axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


# =============================================================================
# Capture — one connection per setting, exactly like c3_bench.run_one
# =============================================================================
class _EncodedWriter:
    """Append H.264/H.265 packets and the sidecar c3_extract_rgb_video.py wants.

    Recording starts at the first keyframe: a stream opening on a P frame decodes
    to fewer images than there are sidecar rows, and the extractor treats that
    mismatch as a hard failure (correctly — it means the dataset would be
    misaligned). The parameter sets are prepended so the file stands alone even
    though capture began mid-stream.
    """

    def __init__(self, run_dir: Path, codec: str, max_frames: int):
        self.codec = codec
        self.max_frames = max_frames
        self.path = run_dir / f"rgb.{codec}"
        self._fh = self.path.open("wb")
        self._csv_fh = (run_dir / "rgb_video.csv").open("w", newline="")
        self._csv = csv.DictWriter(self._csv_fh, fieldnames=RGB_VIDEO_FIELDS)
        self._csv.writeheader()
        self._started = False
        self._header_written = False
        self._seqs: set[int] = set()
        self.rows: list[dict] = []

    @property
    def count(self) -> int:
        return len(self.rows)

    def add(self, frame) -> bool:
        if frame.raw is None or frame.seq in self._seqs or self.full:
            return False
        ftype = annexb_frame_type(frame.raw, self.codec)
        if not self._started:
            if ftype != "I":
                return False
            self._started = True
        if not self._header_written and frame.codec_header:
            self._fh.write(frame.codec_header)
            self._header_written = True
        offset = self._fh.tell()
        self._fh.write(frame.raw)
        self._seqs.add(frame.seq)
        idx = len(self.rows) + 1
        self.rows.append({
            "frame_idx": idx,
            "t_unix": f"{frame.t_device:.6f}",
            "t_monotonic": f"{frame.t_device:.6f}",
            "rgb_seq": frame.seq,
            "rgb_file": f"rgb/{idx:06d}.png",
            "byte_offset": offset,
            "byte_size": len(frame.raw),
            "frame_type": ftype,
            "_latency_ms": frame.latency_ms,
        })
        return True

    @property
    def full(self) -> bool:
        return len(self.rows) >= self.max_frames

    def close(self) -> None:
        for row in self.rows:
            self._csv.writerow({k: v for k, v in row.items()
                                if k in RGB_VIDEO_FIELDS})
        self._csv_fh.close()
        self._fh.close()


def _write_frames_csv(run_dir: Path, rows: list[dict]) -> None:
    with (run_dir / "frames.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FRAME_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FRAME_FIELDS})


def run_one(cfg: StreamConfig, setting: EncodeSetting, run_dir: Path,
            duration: float, warmup: float, save_frames: int,
            scene: str = "static", verbose: bool = True) -> dict:
    """Stream one encoder setting, keep `save_frames` frames, return a CSV row."""
    from c3_camera.source import C3Source   # imported here so --offline/--dry-run
                                            # never pull in the device path

    rc_cfg = cfg.resolve()
    bw = rc_cfg.bandwidth_mbps()
    row = {f: "" for f in FIELDS}
    row.update({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run": run_dir.name,
        "label": setting.label,
        "codec": setting.codec,
        "mjpeg_quality": "" if setting.quality is None else setting.quality,
        "bitrate_kbps": "" if setting.bitrate_kbps is None else setting.bitrate_kbps,
        "keyframe": "" if setting.keyframe is None else setting.keyframe,
        "color_res": cfg.color_res,
        "isp_scale": ("none" if cfg.isp_scale is None
                      else f"{cfg.isp_scale[0]}/{cfg.isp_scale[1]}"),
        "color_out": f"{rc_cfg.color_out_size[0]}x{rc_cfg.color_out_size[1]}",
        "mono_res": cfg.mono_res,
        "requested_fps": cfg.fps,
        "streams": "+".join(cfg.streams),
        "scene": scene,
        "est_mbps": round(bw["total"], 1),
        "ok": 0,
        "extraction_ok": "",
    })

    run_dir.mkdir(parents=True, exist_ok=True)
    frame_rows: list[dict] = []
    wire_bytes: list[int] = []
    encoded: _EncodedWriter | None = None
    seen: set[int] = set()
    summary: dict = {}

    t0 = time.monotonic()
    try:
        # keep_raw makes MJPEG frames carry their original JPEG bytes, so the
        # saved file is the wire payload and not a re-encode of a decode.
        with C3Source(cfg, verbose=verbose, keep_raw=True) as src:
            if setting.interframe:
                encoded = _EncodedWriter(run_dir, setting.codec, save_frames)
            else:
                (run_dir / "frames").mkdir(exist_ok=True)

            reset_done = warmup <= 0
            warm_until = time.monotonic() + max(0.0, warmup)
            deadline = warm_until + duration
            while time.monotonic() < deadline:
                bundle = src.read(timeout_ms=2000)
                # Drain unconditionally: c3_bench never calls this, so the list
                # grows for the whole run. At 30 Mbit/s that is ~49 MB per config.
                packets = src.drain_encoded_rgb()
                if bundle is None:
                    continue
                src.metrics.mark_loop()
                if not reset_done:
                    if time.monotonic() < warm_until:
                        continue
                    src.metrics = type(src.metrics)(cfg.streams)
                    reset_done = True

                if setting.interframe:
                    for f in packets:
                        wire_bytes.append(len(f.raw or b""))
                        if encoded.add(f):
                            r = encoded.rows[-1]
                            frame_rows.append({
                                "idx": r["frame_idx"], "seq": f.seq,
                                "t_device": f"{f.t_device:.6f}",
                                "latency_ms": round(f.latency_ms, 1),
                                "wire_bytes": len(f.raw),
                                "exposure_ms": "",
                                "file": r["rgb_file"],
                            })
                    continue

                f = bundle.color
                if f is None or f.seq in seen:
                    continue
                seen.add(f.seq)
                nbytes = (len(f.raw) if f.raw is not None
                          else (f.image.shape[0] * f.image.shape[1] * 3) // 2)
                wire_bytes.append(nbytes)
                if len(frame_rows) >= save_frames or f.image is None:
                    continue
                idx = len(frame_rows) + 1
                if setting.codec == "mjpeg" and f.raw is not None:
                    name = f"frames/{idx:06d}.jpg"
                    (run_dir / name).write_bytes(f.raw)
                else:
                    # PNG: the raw arm is the reference, so it must not acquire
                    # any loss of its own on the way to disk.
                    name = f"frames/{idx:06d}.png"
                    cv2.imwrite(str(run_dir / name), f.image)
                frame_rows.append({
                    "idx": idx, "seq": f.seq, "t_device": f"{f.t_device:.6f}",
                    "latency_ms": round(f.latency_ms, 1), "wire_bytes": nbytes,
                    "exposure_ms": ("" if f.exposure_ms != f.exposure_ms
                                    else round(f.exposure_ms, 2)),
                    "file": name,
                })

            for tag, name in (("color", STREAM_COLOR), ("depth", STREAM_DEPTH)):
                st = src.metrics.streams.get(name)
                if st is None or not st.count:
                    continue
                s = st.summary()
                if tag == "color":
                    row["color_fps"] = s["fps_lifetime"]
                    row["color_lat_p50"] = s["latency_p50_ms"]
                    row["color_lat_p95"] = s["latency_p95_ms"]
                    row["color_lat_max"] = s["latency_max_ms"]
                    row["color_frames"] = s["frames"]
                    row["color_dropped"] = s["dropped"]
                    row["color_drop_rate"] = s["drop_rate"]
                    row["color_kb_frame"] = s["mean_frame_kb"]
                    row["color_mbps"] = s["mbps_measured"]
                else:
                    row["depth_fps"] = s["fps_lifetime"]
                    row["depth_kb_frame"] = s["mean_frame_kb"]
                    row["depth_mbps"] = s["mbps_measured"]
            row["measured_mbps"] = round(src.metrics.mbps_total, 1)
            row["budget_frac"] = round(src.metrics.mbps_total / C.POE_BUDGET_MBPS, 3)
            row["loop_hz"] = round(src.metrics.loop_hz, 2)
            row["ok"] = 1
            summary = src.summary()
    except Exception as e:      # noqa: BLE001 — one bad setting must not end a sweep
        row["error"] = f"{type(e).__name__}: {e}".replace("\n", " ")[:300]
    finally:
        # Also on Ctrl-C: the sidecar rows only reach disk on close(), so an
        # unclosed writer leaves an elementary stream with no index. meta.json is
        # written last and outside this, so an interrupted run stays incomplete
        # and the scoring pass skips it instead of half-reading it.
        if encoded is not None:
            encoded.close()
        _write_frames_csv(run_dir, frame_rows)
    row["frames_saved"] = len(frame_rows)

    # Burstiness: p95 over mean bytes per frame. config.bandwidth_mbps() reports
    # only a mean, so a CBR I-frame peak (5-10x a P frame) is invisible there
    # while it is exactly what fills the 1.5 s blocking queue and spikes latency.
    if wire_bytes:
        mean_b = statistics.fmean(wire_bytes)
        p95_b = _percentile(wire_bytes, 95)
        row["color_kb_frame_p95"] = round(p95_b / 1024, 1)
        row["burstiness"] = round(p95_b / mean_b, 2) if mean_b else ""

    # Decode H.26x through the dataset path's own extractor, not a private
    # decoder: what is being measured is this codec *as the dataset pipeline
    # would deliver it*, including whether extraction succeeds at all.
    if setting.interframe and row["ok"]:
        try:
            info = extract_rgb_video(run_dir)
            row["frames_extracted"] = info["frames_decoded"]
            row["extraction_ok"] = 1
        except Exception as e:   # noqa: BLE001
            row["frames_extracted"] = 0
            row["extraction_ok"] = 0
            row["error"] = (row["error"] or
                            f"extraction failed: {type(e).__name__}: {e}")[:300]
    elif row["ok"]:
        row["frames_extracted"] = len(frame_rows)
        row["extraction_ok"] = 1

    row["duration_s"] = round(time.monotonic() - t0, 1)
    (run_dir / "meta.json").write_text(json.dumps({
        "setting": setting.to_dict(),
        "scene": scene,
        "row": row,
        "source": summary,
    }, indent=2, default=str) + "\n")
    return row


def _percentile(values, pct: float) -> float:
    s = sorted(values)
    if not s:
        return float("nan")
    i = min(len(s) - 1, max(0, int(round((pct / 100.0) * (len(s) - 1)))))
    return float(s[i])


# =============================================================================
# Scoring — hardware-free, so it can be re-run and unit-tested
# =============================================================================
def load_run(run_dir: Path) -> dict:
    """Read one captured run back off disk (no device, no depthai)."""
    meta = json.loads((run_dir / "meta.json").read_text())
    rows = []
    frames_csv = run_dir / "frames.csv"
    if frames_csv.is_file():
        with frames_csv.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
    return {"dir": run_dir, "meta": meta, "frames": rows,
            "setting": meta.get("setting", {}), "row": meta.get("row", {})}


def run_images(run: dict, start: int = 0, limit: int | None = None,
               sample: int = 1) -> list[tuple[int, Path, np.ndarray]]:
    """Decoded BGR frames for a run, as (frame_idx, path, image)."""
    out = []
    rows = run["frames"][start:]
    for k, r in enumerate(rows):
        if k % max(1, sample):
            continue
        rel = r.get("file") or ""
        if not rel:
            continue
        p = run["dir"] / rel
        if not p.is_file():
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        out.append((int(r["idx"]), p, img))
        if limit is not None and len(out) >= limit:
            break
    return out


def reference_is_lossless(run: dict) -> bool:
    """Only a `--codec none` run may be the reference.

    Measured, not theoretical: scoring against JPEG frames inverted the ranking —
    q95 (46.1 kB) came out at 47.49 dB against q90 (34.2 kB) at 52.22 dB, because
    a re-encode at the same quantiser is nearly idempotent, so the metric was
    measuring distance to one particular quantiser grid rather than quality.
    """
    return str(run.get("setting", {}).get("codec", "")) == "none"


def reference_images(run: dict, n_frames: int) -> list[tuple[int, Path, np.ndarray]]:
    """The exact frames that go into the median stack, with their frame indices.

    The indices are returned, not just the images, because the raw arm's own row
    has to be scored on frames that are DISJOINT from these. Taking "everything
    from row n_frames onward" instead is only equivalent while every file is
    readable: one missing or corrupt PNG early in the run shifts the selection by
    one and the raw arm ends up scored against a median it contributed to, which
    inflates the noise floor — and the noise floor is the yardstick every encoder
    row is read against.
    """
    if not reference_is_lossless(run):
        raise ValueError(
            f"refusing to use {run['dir'].name} as the reference: it is "
            f"{run.get('setting', {}).get('codec')!r}, not raw. A compressed "
            "reference inverts the ranking (see reference_is_lossless).")
    frames = run_images(run, start=0, limit=n_frames)
    if not frames:
        raise ValueError(f"no readable reference frames in {run['dir']}")
    return frames


def build_reference(run: dict, n_frames: int) -> np.ndarray:
    return median_reference([im for _, _, im in reference_images(run, n_frames)])


def score_runs(runs: list[dict], reference_run: dict, *, reference_frames: int,
               orb_features: int = 2000, match_radius_px: float = 1.5,
               min_ref_keypoints: int = 500, sample: int = 1,
               tag_family: str = "36h11", known_ids: set[int] | None = None,
               verbose: bool = True) -> tuple[list[dict], dict]:
    """Score every run against the reference median stack.

    Returns (per-frame rows, {run_dir_name: aggregate}).
    """
    ref_frames = reference_images(reference_run, reference_frames)
    ref_used = {idx for idx, _, _ in ref_frames}
    ref = median_reference([im for _, _, im in ref_frames])
    ref_luma = to_luma(ref)
    orb = OrbScorer(ref, nfeatures=orb_features, radius_px=match_radius_px)
    tags = TagDetector(tag_family, known_ids)
    under_textured = orb.reference_keypoints < min_ref_keypoints
    ref_tags = tags.detect(ref)

    if verbose:
        print(f"  reference   : median of {reference_frames} raw frames from "
              f"{reference_run['dir'].name}, {ref.shape[1]}x{ref.shape[0]}")
        print(f"  ORB on ref  : {orb.reference_keypoints} keypoints"
              + ("   *** UNDER-TEXTURED: point the camera at something with "
                 "structure, these rankings are noise ***" if under_textured else ""))
        print(f"  tags on ref : {ref_tags[0]} detected via {tags.backend}"
              if tags.available else
              "  tags on ref : no AprilTag backend available — skipping tag metrics")

    per_frame: list[dict] = []
    aggregates: dict[str, dict] = {}
    for run in runs:
        name = run["dir"].name
        label = run.get("setting", {}).get("label", name)
        codec = run.get("setting", {}).get("codec", "")
        agg = {"scored_frames": 0, "under_textured": int(under_textured),
               "tag_backend": tags.backend, "quality_note": ""}
        # The reference run is scored on frames it did NOT contribute to the
        # median, otherwise its row would be comparing an image to itself.
        # Excluded by frame INDEX rather than by row offset, so an unreadable
        # frame cannot slide the two sets into overlapping.
        images = run_images(run, start=0, sample=sample)
        if run is reference_run:
            images = [t for t in images if t[0] not in ref_used]
        if not images:
            agg["quality_note"] = "no frames"
            aggregates[name] = agg
            continue
        if images[0][2].shape[:2] != ref.shape[:2]:
            agg["quality_note"] = (f"size {images[0][2].shape[1]}x"
                                   f"{images[0][2].shape[0]} != reference")
            aggregates[name] = agg
            continue

        psnrs, ssims, orb_ns, orb_rates, laps = [], [], [], [], []
        tdet, tknown, tfalse = [], [], []
        for idx, path, img in images:
            p = psnr_db(ref_luma, to_luma(img))
            s = ssim(ref_luma, to_luma(img))
            n, rate = orb.score(img)
            lv = laplacian_var(img)
            d, k, fa = tags.detect(img)
            psnrs.append(p)
            ssims.append(s)
            orb_ns.append(n)
            orb_rates.append(rate)
            laps.append(lv)
            tdet.append(d)
            tknown.append(k)
            tfalse.append(fa)
            per_frame.append({
                "run": name, "label": label, "codec": codec,
                "frame_idx": idx, "file": str(path.relative_to(run["dir"])),
                "psnr_y_db": ("inf" if p == float("inf") else round(p, 3)),
                "ssim_y": round(s, 5),
                "orb_n": n, "orb_inlier_rate": round(rate, 4),
                "tags_detected": d, "tags_known": k, "tags_false": fa,
                "lap_var": round(lv, 1),
            })

        finite = [p for p in psnrs if p != float("inf")]
        agg.update({
            "scored_frames": len(images),
            "psnr_y_db": ("inf" if not finite else round(statistics.fmean(finite), 2)),
            "psnr_y_sd": ("" if len(finite) < 2 else round(statistics.pstdev(finite), 3)),
            "psnr_y_p05": ("inf" if not finite else round(_percentile(finite, 5), 2)),
            "ssim_y": round(statistics.fmean(ssims), 4),
            "ssim_y_sd": round(statistics.pstdev(ssims), 4),
            "ssim_y_p05": round(_percentile(ssims, 5), 4),
            "orb_n": round(statistics.fmean(orb_ns), 1),
            "orb_inlier_rate": round(statistics.fmean(orb_rates), 4),
            "orb_inlier_sd": round(statistics.pstdev(orb_rates), 4),
            "orb_inlier_p05": round(_percentile(orb_rates, 5), 4),
            "tags_detected": round(statistics.fmean(tdet), 2),
            "tags_known": round(statistics.fmean(tknown), 2),
            "tags_false": round(statistics.fmean(tfalse), 2),
            "lap_var": round(statistics.fmean(laps), 1),
        })
        if run is reference_run:
            agg["quality_note"] = "noise floor (raw vs median of its own run)"
        aggregates[name] = agg
    return per_frame, aggregates


# =============================================================================
# Report
# =============================================================================
def _f(value, spec: str = "", dash: str = "-") -> str:
    """Format a possibly-missing cell, keeping the column width either way.

    A blank cell has to occupy the same space as a number or the table's columns
    stop lining up exactly when a run failed — which is when it is being read
    most carefully.
    """
    head = spec.split(".")[0] if spec else ""
    digits = "".join(ch for ch in head if ch.isdigit())
    width = int(digits) if digits else 0
    if value is None or value == "":
        return dash.rjust(width)
    try:
        return format(float(value), spec) if spec else str(value)
    except (TypeError, ValueError):
        return str(value).rjust(width)


def _metric(row: dict, key: str, default: float = float("-inf")) -> float:
    """One CSV cell as a number. Blank means 'not measured', not zero."""
    v = row.get(key, "")
    if v in ("", None):
        return default
    if v == "inf":
        return float("inf")
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def decision_margin(good: list[dict], rank_by: str) -> str:
    """How the winner's lead compares with the metric's own frame-to-frame spread.

    Without this the table announces a winner whether it led by 0.20 or by
    0.004, and adjacent rungs of a bitrate ladder routinely differ by less than
    one standard deviation. Note what this does NOT establish: the spread is
    measured across frames of a single connection, while the settings being
    compared are separate connections with their own exposure and AWB state, so
    even a margin outside the scatter is not a repeatability claim.
    """
    if not good:
        return ""
    best = good[0]
    sd = _metric(best, SCATTER_OF.get(rank_by, ""), float("nan"))
    sd_txt = "unknown" if sd != sd else f"{sd:.4f}"
    if len(good) < 2:
        return (f"margin: no runner-up qualified. Per-frame scatter (sd) of "
                f"{rank_by} on the winner: {sd_txt}.")
    lead = _metric(best, rank_by, float("nan")) - _metric(good[1], rank_by, float("nan"))
    verdict = "UNRESOLVED — the two settings are tied by this capture"
    if sd == sd and lead == lead and lead > sd:
        verdict = ("outside the per-frame scatter, but the settings were never "
                   "re-captured, so run-to-run variation is still unmeasured")
    return (f"margin over {good[1].get('label', '?')}: {lead:+.4f} vs per-frame "
            f"scatter (sd) {sd_txt} — {verdict}")


def rank_rows(rows: list[dict], target_fps: float, budget_mbps: float,
              rank_by: str) -> tuple[list[dict], list[dict]]:
    """Split into (qualifying, rejected) and sort the qualifying by quality.

    Qualifying means: the setting actually delivered the frame rate asked for,
    stayed inside the bandwidth budget, and produced frames that can still be
    read back. Ranking by quality alone would hand the crown to a setting that
    reaches it at 4 fps; ranking by bandwidth alone answers a question nobody
    asked, since the bitrate is an input.
    """
    if rank_by not in RANKABLE_METRICS:
        raise ValueError(f"--rank-by must be one of {RANKABLE_METRICS} "
                         f"(Laplacian variance is deliberately not rankable)")
    _num = _metric
    good, bad = [], []
    for r in rows:
        reasons = []
        if not r.get("ok"):
            reasons.append("run failed")
        if _num(r, "color_fps", 0.0) < target_fps:
            reasons.append(f"fps {_f(r.get('color_fps'), '.1f')} < {target_fps:g}")
        if _num(r, "measured_mbps", 0.0) > budget_mbps:
            reasons.append(f"{_f(r.get('measured_mbps'), '.1f')} Mbit/s > budget")
        if str(r.get("extraction_ok")) == "0":
            reasons.append("frames not recoverable")
        if not _num(r, "scored_frames", 0.0):
            reasons.append("not scored")
        r = dict(r)
        r["_reject"] = "; ".join(reasons)
        (bad if reasons else good).append(r)
    good.sort(key=lambda r: _num(r, rank_by), reverse=True)
    return good, bad


def static_scene_warning(rows: list[dict], floor: float = 0.7) -> str | None:
    """Flag a raw arm whose own frames do not agree with each other.

    The raw arm is scored against the median of its OWN run, so if the rig is
    fixed the only thing left to move its ORB inlier rate is sensor noise. A low
    value means the frames of a single run disagree — and if they disagree
    within one run they certainly disagree between runs, which is the assumption
    every cross-run number here rests on.

    Two things this check is NOT. It does not separate motion from noise:
    measured on a perfectly static synthetic scene, the rate is 0.92 at 0.5 DN
    of read noise, 0.86 at 2.5 DN and drops below this 0.7 floor at ~12 DN — a
    plausible high-gain underwater exposure. So a fired warning means "something
    is inconsistent", which may be a noisy sensor rather than a moving rig; raise
    --min-ref-keypoints and inspect the raw frames before blaming the mount. And
    it cannot see slow drift: it compares frames a few seconds apart inside one
    run, while the ladder itself spans minutes (see the run-order caveat).
    """
    raw = next((r for r in rows if r.get("codec") == "none"
                and r.get("orb_inlier_rate") not in ("", None)), None)
    if raw is None:
        return None
    try:
        rate = float(raw["orb_inlier_rate"])
    except (TypeError, ValueError):
        return None
    if rate >= floor:
        return None
    return (f"SCENE WAS NOT STATIC (or the sensor is very noisy): the raw arm "
            f"scores only {rate:.3f} ORB inlier rate against the median of its "
            f"own run (expected > {floor:g} for a fixed rig at moderate gain). "
            f"The frames of a single run already disagree, so the cross-run "
            f"fidelity columns compare different images and the ranking below is "
            f"not meaningful. Check the mount first, then the gain/exposure — "
            f"this floor is also reached by read noise alone at ~12 DN.")


def print_report(rows: list[dict], *, target_fps: float, budget_mbps: float,
                 rank_by: str, scene: str) -> None:
    good, bad = rank_rows(rows, target_fps, budget_mbps, rank_by)
    moved = static_scene_warning(rows)
    if moved:
        print("\n" + "!" * 78 + f"\n{moved}\n" + "!" * 78)

    width = 132
    print("\n" + "=" * width)
    print(f"{'setting':<22} {'fps':>6} {'p50':>6} {'p95':>6} {'Mbit/s':>7} "
          f"{'%ceil':>6} {'kB/f':>7} {'brst':>5} {'PSNR':>7} {'SSIM':>7} "
          f"{'ORBinl':>7} {'ORBn':>6} {'tags':>5} {'lap':>7}")
    print("-" * width)
    for r in good + bad:
        mark = " " if not r.get("_reject") else "x"
        budget = r.get("budget_frac")
        print(f"{mark}{r.get('label', '?'):<21} "
              f"{_f(r.get('color_fps'), '6.1f')} "
              f"{_f(r.get('color_lat_p50'), '6.0f')} "
              f"{_f(r.get('color_lat_p95'), '6.0f')} "
              f"{_f(r.get('measured_mbps'), '7.1f')} "
              f"{('-' if budget in ('', None) else format(float(budget) * 100, '5.0f') + '%'):>6} "
              f"{_f(r.get('color_kb_frame'), '7.1f')} "
              f"{_f(r.get('burstiness'), '5.2f')} "
              f"{_f(r.get('psnr_y_db'), '7.2f')} "
              f"{_f(r.get('ssim_y'), '7.4f')} "
              f"{_f(r.get('orb_inlier_rate'), '7.4f')} "
              f"{_f(r.get('orb_n'), '6.0f')} "
              f"{_f(r.get('tags_detected'), '5.1f')} "
              f"{_f(r.get('lap_var'), '7.0f')}")
    print("=" * width)
    print("'x' = does not qualify at the target frame rate / budget:")
    for r in bad:
        print(f"    x {r.get('label', '?'):<22} {r['_reject']}")
    if not bad:
        print("    (none — every setting met the target)")

    print(f"\nranked by {rank_by} among settings reaching >= {target_fps:g} fps "
          f"within {budget_mbps:g} Mbit/s  [scene: {scene}]")
    if good:
        best = good[0]
        print(f"\nDECISION: {best.get('label')} — {rank_by} "
              f"{_f(best.get(rank_by), '.4f')} at "
              f"{_f(best.get('color_fps'), '.1f')} fps, "
              f"{_f(best.get('measured_mbps'), '.1f')} Mbit/s "
              f"({_f(best.get('color_kb_frame'), '.1f')} kB/frame, "
              f"p50 {_f(best.get('color_lat_p50'), '.0f')} ms)")
        print(f"          {decision_margin(good, rank_by)}")
        if best.get("codec") == "none":
            print("          the raw arm won: at this resolution and target the "
                  "link is not the constraint, so there is nothing an encoder "
                  "can buy. Raise the resolution before choosing a codec.")
        raw = next((r for r in rows if r.get("codec") == "none"
                    and r.get("scored_frames")), None)
        if raw is not None and raw.get(rank_by) not in ("", None):
            print(f"          noise floor (raw arm): {rank_by} "
                  f"{_f(raw.get(rank_by), '.4f')} at "
                  f"{_f(raw.get('color_fps'), '.1f')} fps — an encoder at or "
                  f"above this is indistinguishable from raw by this method")
    else:
        print("\nDECISION: none of the settings met the target. Lower "
              "--target-fps, raise --budget-mbps, or widen the ladder.")

    print("\n" + "-" * width)
    print("HOW TO READ THIS")
    for i, note in enumerate(CAVEATS, 1):
        print(f"  {i:2d}. {note}")
    print("-" * width)


def write_report(path: Path, rows: list[dict], *, target_fps: float,
                 budget_mbps: float, rank_by: str, scene: str) -> None:
    good, bad = rank_rows(rows, target_fps, budget_mbps, rank_by)
    lines = ["# c3 encoder bandwidth-vs-quality", "",
             f"scene: `{scene}`  |  target fps: {target_fps:g}  |  "
             f"budget: {budget_mbps:g} Mbit/s  |  ranked by: `{rank_by}`", "",
             "| setting | fps | p50 ms | Mbit/s | kB/f | burst | PSNR-Y | SSIM-Y "
             "| ORB inlier | ORB n | tags | lap var | qualifies |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in good + bad:
        lines.append(
            f"| {r.get('label', '?')} | {_f(r.get('color_fps'), '.1f')} "
            f"| {_f(r.get('color_lat_p50'), '.0f')} "
            f"| {_f(r.get('measured_mbps'), '.1f')} "
            f"| {_f(r.get('color_kb_frame'), '.1f')} "
            f"| {_f(r.get('burstiness'), '.2f')} "
            f"| {_f(r.get('psnr_y_db'), '.2f')} | {_f(r.get('ssim_y'), '.4f')} "
            f"| {_f(r.get('orb_inlier_rate'), '.4f')} | {_f(r.get('orb_n'), '.0f')} "
            f"| {_f(r.get('tags_detected'), '.1f')} | {_f(r.get('lap_var'), '.0f')} "
            f"| {'yes' if not r.get('_reject') else r['_reject']} |")
    lines += ["", "## Decision", ""]
    if good:
        lines.append(f"**{good[0].get('label')}** — {rank_by} "
                     f"{_f(good[0].get(rank_by), '.4f')} at "
                     f"{_f(good[0].get('color_fps'), '.1f')} fps, "
                     f"{_f(good[0].get('measured_mbps'), '.1f')} Mbit/s")
        lines += ["", decision_margin(good, rank_by)]
    else:
        lines.append("No setting reached the target frame rate inside the budget.")
    moved = static_scene_warning(rows)
    if moved:
        lines += ["", f"> **{moved}**"]
    lines += ["", "## Caveats — these travel with the numbers", ""]
    lines += [f"{i}. {note}" for i, note in enumerate(CAVEATS, 1)]
    path.write_text("\n".join(lines) + "\n")


# =============================================================================
# CLI
# =============================================================================
def _load_tag_map(path: Path | None) -> set[int] | None:
    if path is None:
        return None
    data = json.loads(Path(path).read_text())
    ids = data.get("ids", data) if isinstance(data, dict) else data
    return {int(i) for i in ids}


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_connection_args(p)

    g = p.add_argument_group("encoder ladder (comma-separated; axes fold per codec)")
    g.add_argument("--codec", default="none,mjpeg,h264,h265",
                   help="none,mjpeg,h264,h265 (default: %(default)s). 'none' is "
                        "the raw reference arm and is not optional — without it "
                        "there is no fidelity baseline at all.")
    g.add_argument("--mjpeg-quality", default="90",
                   help="MJPEG quality ladder, e.g. 70,80,90,97 (default: %(default)s)")
    g.add_argument("--bitrate-kbps", default="4000",
                   help="H.264/H.265 bitrate ladder in kbit/s, e.g. "
                        "2000,4000,8000,16000 (default: %(default)s). Space these "
                        "logarithmically: rate-distortion is roughly linear in "
                        "log(rate), so linear spacing wastes runs at the top.")
    g.add_argument("--keyframe", default="0",
                   help="H.264/H.265 keyframe interval in frames; 0 = one per "
                        "second (default: %(default)s). 1 means all-intra, which "
                        "is MJPEG's direct competitor: one-frame latency, "
                        "one-frame loss, no GOP to lose.")
    g.add_argument("--settings", default=None,
                   help="explicit ladder instead of the axes above, e.g. "
                        "'none,mjpeg@90,h265@8000/kf1'")

    g = p.add_argument_group("fixed pipeline settings (NOT swept — the reference "
                             "is only valid at one resolution)")
    g.add_argument("--color-res", default="1080p", choices=sorted(C.COLOR_RESOLUTIONS))
    g.add_argument("--isp-scale", default="1/2", help="e.g. 1/4, 1/2, none "
                                                      "(default: %(default)s)")
    g.add_argument("--mono-res", default="400p", choices=sorted(C.MONO_RESOLUTIONS))
    g.add_argument("--fps", type=float, default=20.0,
                   help="requested colour fps (default: %(default)s)")
    g.add_argument("--streams", default="color",
                   help="stream set (default: %(default)s). Keep depth OFF: at "
                        "640x360 uint16 it is ~74 Mbit/s and swamps a 2-30 "
                        "Mbit/s colour ladder, so the encoder axis stops being "
                        "visible. Use color,depth only for a confirmation run.")
    g.add_argument("--depth-align", default="color",
                   choices=("color", "right", "left", "center", "none"))

    g = p.add_argument_group("run control")
    g.add_argument("--duration", type=float, default=10.0,
                   help="measured seconds per setting (default: %(default)s)")
    g.add_argument("--warmup", type=float, default=3.0,
                   help="discarded seconds per setting (default: %(default)s)")
    g.add_argument("--settle", type=float, default=5.0,
                   help="pause between settings, for the PoE device to reset "
                        "(default: %(default)s)")
    g.add_argument("--save-frames", type=int, default=60,
                   help="frames kept per setting for scoring (default: %(default)s)")
    g.add_argument("--scene-label", default="static",
                   help="'static' or 'motion'; recorded in every row because a "
                        "static scene is the best case for inter-frame codecs "
                        "(default: %(default)s)")

    g = p.add_argument_group("quality scoring")
    g.add_argument("--reference-frames", type=int, default=25,
                   help="raw frames median-stacked into the reference image "
                        "(default: %(default)s). The raw arm is then scored on "
                        "its REMAINING frames, so it never scores against itself.")
    g.add_argument("--orb-features", type=int, default=2000)
    g.add_argument("--match-radius-px", type=float, default=1.5,
                   help="an ORB match counts as an inlier within this radius "
                        "(default: %(default)s px)")
    g.add_argument("--min-ref-keypoints", type=int, default=500,
                   help="fewer ORB keypoints than this on the reference flags "
                        "the whole capture as under-textured (default: %(default)s)")
    g.add_argument("--tag-family", default="36h11")
    g.add_argument("--tag-map", type=Path, default=None,
                   help="JSON list of the tag IDs really present, so detections "
                        "outside it are counted as false positives rather than "
                        "credited as detections")
    g.add_argument("--sample", type=int, default=1,
                   help="score every Nth saved frame (default: %(default)s)")
    g.add_argument("--rank-by", default="orb_inlier_rate", choices=RANKABLE_METRICS,
                   help="deciding metric (default: %(default)s — it is what a "
                        "SLAM front end actually consumes, and it separates "
                        "settings that PSNR ranks identically)")
    g.add_argument("--target-fps", type=float, default=None,
                   help="a setting must reach this to be ranked at all "
                        "(default: --fps minus 5%%)")
    g.add_argument("--budget-mbps", type=float, default=C.POE_BUDGET_MBPS,
                   help="measured Mbit/s ceiling a setting must fit inside "
                        "(default: %(default)s)")

    g = p.add_argument_group("output / modes")
    g.add_argument("--out", type=Path, default=None,
                   help="output directory (default: "
                        "c3_camera/encode_quality/<YYYYMMDD_HHMMSS>)")
    g.add_argument("--offline", type=Path, default=None, metavar="DIR",
                   help="re-score an existing capture directory and exit. Never "
                        "opens the camera, so scoring can be iterated freely.")
    g.add_argument("--no-score", action="store_true",
                   help="capture only; score later with --offline")
    g.add_argument("--dry-run", action="store_true",
                   help="resolve the ladder, print the estimated cost and time, "
                        "and exit without connecting")
    g.add_argument("--quiet", action="store_true", help="less per-run chatter")
    PF.add_preflight_args(p)
    return p


def _settings_from_args(a: argparse.Namespace) -> list[EncodeSetting]:
    if a.settings:
        return list(dict.fromkeys(parse_setting(s) for s in _csv_list(a.settings)))
    return expand_settings(_csv_list(a.codec), _csv_list(a.mjpeg_quality),
                           _csv_list(a.bitrate_kbps), _csv_list(a.keyframe))


def score_directory(out: Path, *, reference_frames: int, orb_features: int,
                    match_radius_px: float, min_ref_keypoints: int, sample: int,
                    tag_family: str, known_ids: set[int] | None,
                    verbose: bool = True) -> list[dict]:
    """Load every run in `out`, score it, and merge quality into runs.csv."""
    run_dirs = sorted(d for d in out.iterdir()
                      if d.is_dir() and (d / "meta.json").is_file())
    if not run_dirs:
        raise RuntimeError(f"no captured runs under {out}")
    runs = [load_run(d) for d in run_dirs]

    reference = next((r for r in runs if reference_is_lossless(r)
                      and r["frames"]), None)
    if reference is None:
        raise RuntimeError(
            "no raw (--codec none) run with frames in this directory. The raw "
            "arm is the experiment's premise, not an option: without it there "
            "is no uncompressed reference and the fidelity columns are "
            "meaningless.")
    if len(reference["frames"]) <= reference_frames:
        raise RuntimeError(
            f"the raw run has {len(reference['frames'])} frames but "
            f"--reference-frames is {reference_frames}; it would have nothing "
            f"left to be scored on. Capture more frames or lower the setting.")

    # H.26x runs may have been captured but not yet extracted (or the images
    # cleaned away). Re-extract through the dataset path so offline scoring is
    # self-sufficient.
    for run in runs:
        codec = run["setting"].get("codec")
        if codec in ("h264", "h265") and run["frames"]:
            first = run["dir"] / run["frames"][0]["file"]
            if not first.is_file():
                try:
                    extract_rgb_video(run["dir"])
                except Exception as e:      # noqa: BLE001
                    if verbose:
                        print(f"  {run['dir'].name}: extraction failed ({e})")

    per_frame, aggregates = score_runs(
        runs, reference, reference_frames=reference_frames,
        orb_features=orb_features, match_radius_px=match_radius_px,
        min_ref_keypoints=min_ref_keypoints, sample=sample,
        tag_family=tag_family, known_ids=known_ids, verbose=verbose)

    with (out / "scores.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SCORE_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(per_frame)

    rows = []
    for run in runs:
        row = dict(run["row"])
        row.update(aggregates.get(run["dir"].name, {}))
        rows.append(row)
    _write_rows(out / "runs.csv", rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    known_ids = _load_tag_map(a.tag_map)
    target_fps = a.target_fps if a.target_fps is not None else a.fps * 0.95

    # ---------------------------------------------------------------- offline
    if a.offline is not None:
        out = Path(a.offline)
        print("=" * 78)
        print(f"C3 encoder quality: re-scoring {out} (no device is opened)")
        print("=" * 78)
        rows = score_directory(
            out, reference_frames=a.reference_frames, orb_features=a.orb_features,
            match_radius_px=a.match_radius_px, min_ref_keypoints=a.min_ref_keypoints,
            sample=a.sample, tag_family=a.tag_family, known_ids=known_ids,
            verbose=not a.quiet)
        scene = next((r.get("scene") for r in rows if r.get("scene")), a.scene_label)
        print_report(rows, target_fps=target_fps, budget_mbps=a.budget_mbps,
                     rank_by=a.rank_by, scene=scene or a.scene_label)
        write_report(out / "report.md", rows, target_fps=target_fps,
                     budget_mbps=a.budget_mbps, rank_by=a.rank_by,
                     scene=scene or a.scene_label)
        print(f"\nwrote {out/'runs.csv'}, {out/'scores.csv'}, {out/'report.md'}")
        return 0

    # ---------------------------------------------------------------- capture
    settings = _settings_from_args(a)
    if not settings:
        print("no encoder settings to run")
        return 2
    if not any(s.codec == "none" for s in settings):
        print("WARNING: no raw (--codec none) arm in this ladder. Nothing will "
              "be scored: PSNR/SSIM/ORB-inlier all need an uncompressed "
              "reference, and using a compressed frame as one has been measured "
              "to invert the ranking.")
    if a.reference_frames >= a.save_frames:
        print(f"WARNING: --reference-frames {a.reference_frames} >= "
              f"--save-frames {a.save_frames}; the raw arm would have no frames "
              f"left to score. Raise --save-frames.")

    base = dict(
        color_res=a.color_res, isp_scale=_parse_isp(a.isp_scale),
        mono_res=a.mono_res, fps=a.fps,
        streams=tuple(_csv_list(a.streams)), depth_align=a.depth_align,
    )

    est_total = 0.0
    print("=" * 78)
    print(f"C3 encoder quality: {len(settings)} settings, scene "
          f"'{a.scene_label}', {a.save_frames} frames kept each")
    print("=" * 78)
    for s in settings:
        rc = StreamConfig(**s.apply(base)).resolve()
        bw = rc.bandwidth_mbps()["total"]
        est_total += a.duration + a.warmup + a.settle + 14
        print(f"  {s.label:<24} est {bw:6.1f} Mbit/s "
              f"({bw / C.POE_BUDGET_MBPS * 100:3.0f}% of ceiling)  "
              f"{rc.color_out_size[0]}x{rc.color_out_size[1]}")
    print(f"  ~{est_total/60:.1f} min total")
    if a.dry_run:
        print("\n--dry-run: nothing was opened, nothing was written.")
        return 0

    out = a.out or (Path("c3_camera/encode_quality") /
                    datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Before the output directory exists, so a blocked run leaves no empty
    # skeleton behind for the next reader to wonder about.
    print()
    rc_pf = PF.gate(a, title="c3 encode-quality", vehicle=False, out_dir=out)
    if rc_pf is not None:
        return rc_pf
    print()

    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps({
        "created": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv[1:] if argv is None else argv,
        "scene": a.scene_label,
        "settings": [s.to_dict() for s in settings],
        "base": {k: (list(v) if isinstance(v, tuple) else v)
                 for k, v in base.items()},
        "caveats": list(CAVEATS),
    }, indent=2, default=str) + "\n")

    rows: list[dict] = []
    try:
        for i, s in enumerate(settings, 1):
            cfg = StreamConfig(ip=a.ip, mxid=a.mxid, **s.apply(base))
            run_dir = out / f"run_{i:02d}_{s.slug}"
            print(f"\n[{i}/{len(settings)}] {s.label}")
            row = run_one(cfg, s, run_dir, a.duration, a.warmup, a.save_frames,
                          scene=a.scene_label, verbose=not a.quiet)
            rows.append(row)
            _write_rows(out / "runs.csv", rows)
            if row["ok"]:
                print(f"    -> {row['color_fps'] or '-'} fps, "
                      f"p50 {row['color_lat_p50'] or '-'} ms, "
                      f"{row['measured_mbps'] or '-'} Mbit/s, "
                      f"{row['color_kb_frame'] or '-'} kB/frame, "
                      f"{row['frames_saved']} frames kept")
            else:
                print(f"    -> FAILED: {row['error']}")
            if i < len(settings):
                time.sleep(a.settle)
    except KeyboardInterrupt:
        print("\ninterrupted — partial results kept")

    if a.no_score:
        print(f"\ncapture only. Score it with:\n  {sys.executable} "
              f"{Path(__file__).name} --offline {out}")
        return 0

    try:
        rows = score_directory(
            out, reference_frames=a.reference_frames, orb_features=a.orb_features,
            match_radius_px=a.match_radius_px, min_ref_keypoints=a.min_ref_keypoints,
            sample=a.sample, tag_family=a.tag_family, known_ids=known_ids,
            verbose=not a.quiet)
    except Exception as e:      # noqa: BLE001 — never lose the capture over scoring
        print(f"\nscoring failed: {e}")
        print(f"the capture is intact; retry with --offline {out}")
        return 1

    print_report(rows, target_fps=target_fps, budget_mbps=a.budget_mbps,
                 rank_by=a.rank_by, scene=a.scene_label)
    write_report(out / "report.md", rows, target_fps=target_fps,
                 budget_mbps=a.budget_mbps, rank_by=a.rank_by, scene=a.scene_label)
    print(f"\nwrote {out/'runs.csv'}, {out/'scores.csv'}, {out/'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
