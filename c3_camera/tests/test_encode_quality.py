#!/usr/bin/env python3
"""
test_encode_quality.py — the quality maths in c3_encode_quality.py, on synthetic
images, with no camera anywhere near it.

    ~/.venvs/c3-depthai/bin/python c3_camera/tests/test_encode_quality.py
    ~/.venvs/c3-depthai/bin/python -m pytest c3_camera/tests/test_encode_quality.py -v

Why this file exists: a bug in scoring code is silent. It produces a plausible
table, and that table is what decides the dataset's pixel format. So the
properties the metrics are *chosen for* are pinned here as tests rather than
left as comments — in particular the two that were measured on real frames and
would otherwise be easy to regress:

  * ORB keypoint COUNT can rise as quality falls, while the inlier RATE falls
    monotonically. Ranking by count would therefore pick the worse encoder.
  * A compressed reference inverts the ranking outright (a heavier q95 file
    scored *lower* PSNR than a lighter q90 one, because re-encoding at the same
    quantiser is nearly idempotent). Refusing a compressed reference is a
    correctness requirement, not hygiene.

The reference image here is synthetic on purpose: it is generated from a fixed
seed, so a failure is a code change and never a scene change.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from c3_camera.c3_encode_quality import (
    CAVEATS,
    CORRESPONDENCE_NOTE,
    FIELDS,
    RANKABLE_METRICS,
    EncodeSetting,
    OrbScorer,
    TagDetector,
    annexb_frame_type,
    build_reference,
    expand_settings,
    laplacian_var,
    main,
    median_reference,
    parse_setting,
    print_report,
    psnr_db,
    rank_rows,
    reference_is_lossless,
    score_directory,
    ssim,
    to_luma,
)

JPEG_LADDER = (97, 90, 80, 60, 40, 20)


# =============================================================================
# synthetic imagery
# =============================================================================
def synthetic_scene(w: int = 320, h: int = 240, seed: int = 7) -> np.ndarray:
    """A deterministic textured scene: smooth background plus hard-edged blocks.

    Both parts matter. The smooth part is what JPEG reproduces well and the hard
    edges are what it destroys first, so a metric that responds to neither is
    visibly broken. It also yields enough corners for ORB to be meaningful.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    base = 60.0 + 120.0 * np.sin(x / 23.0) * np.cos(y / 17.0)
    img = np.stack([base, base * 0.9 + 20.0, base * 1.1 - 10.0], axis=-1)
    for _ in range(40):
        x0, y0 = int(rng.integers(0, w - 40)), int(rng.integers(0, h - 40))
        ww, hh = int(rng.integers(8, 40)), int(rng.integers(8, 40))
        img[y0:y0 + hh, x0:x0 + ww] = rng.integers(0, 255, 3)
    img += rng.normal(0.0, 4.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def jpeg_roundtrip(image: np.ndarray, quality: int) -> tuple[np.ndarray, int]:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok, "cv2.imencode failed"
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), int(buf.nbytes)


def tag_scene(ids=(3, 17)) -> np.ndarray:
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    canvas = np.full((260, 500, 3), 230, np.uint8)
    for k, tid in enumerate(ids):
        marker = cv2.aruco.generateImageMarker(d, int(tid), 100)
        canvas[70:170, 60 + k * 220:160 + k * 220] = \
            cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return canvas


# =============================================================================
# PSNR
# =============================================================================
def test_identical_images_give_infinite_psnr():
    """Zero error is not 'very high quality', it is no error at all. Returning a
    large finite number here would let a bit-exact arm be beaten by a lossy one."""
    ref = to_luma(synthetic_scene())
    assert psnr_db(ref, ref) == float("inf")
    assert psnr_db(ref, ref.copy()) == float("inf")


def test_psnr_matches_the_hand_computed_definition():
    a = np.zeros((4, 4), np.float64)
    b = np.zeros((4, 4), np.float64)
    b[0, 0] = 4.0                      # mse = 16/16 = 1 -> 10*log10(255^2) dB
    assert abs(psnr_db(a, b) - 10.0 * np.log10(255.0 ** 2)) < 1e-9


def test_psnr_falls_monotonically_as_jpeg_quality_falls():
    """The headline property: worse compression must score worse, every step."""
    ref = synthetic_scene()
    ref_y = to_luma(ref)
    values = []
    for q in JPEG_LADDER:
        decoded, nbytes = jpeg_roundtrip(ref, q)
        values.append((q, nbytes, psnr_db(ref_y, to_luma(decoded))))
    for (q_hi, _, p_hi), (q_lo, _, p_lo) in zip(values, values[1:]):
        assert p_hi > p_lo, f"PSNR rose from q{q_hi} ({p_hi}) to q{q_lo} ({p_lo})"
    # ... and every one of them is below the bit-exact case.
    assert all(p < float("inf") for _, _, p in values)
    # sanity on the range: a q97 JPEG of a textured scene is good but not perfect
    assert 30.0 < values[0][2] < 60.0, values[0][2]


def test_psnr_and_ssim_reject_mismatched_shapes():
    """Silently resizing would compare a resample to a compression artefact."""
    a, b = np.zeros((8, 8)), np.zeros((8, 16))
    for fn in (psnr_db, ssim):
        try:
            fn(a, b)
        except ValueError:
            continue
        raise AssertionError(f"{fn.__name__} accepted mismatched shapes")


# =============================================================================
# SSIM
# =============================================================================
def test_ssim_is_exactly_one_for_identical_images():
    ref = to_luma(synthetic_scene())
    assert abs(ssim(ref, ref) - 1.0) < 1e-9


def test_ssim_stays_inside_the_unit_interval_across_the_ladder():
    ref = synthetic_scene()
    ref_y = to_luma(ref)
    for q in JPEG_LADDER:
        s = ssim(ref_y, to_luma(jpeg_roundtrip(ref, q)[0]))
        assert 0.0 <= s <= 1.0, f"SSIM {s} at q{q} is outside [0, 1]"


def test_ssim_falls_monotonically_as_jpeg_quality_falls():
    ref = synthetic_scene()
    ref_y = to_luma(ref)
    values = [(q, ssim(ref_y, to_luma(jpeg_roundtrip(ref, q)[0])))
              for q in JPEG_LADDER]
    for (q_hi, s_hi), (q_lo, s_lo) in zip(values, values[1:]):
        assert s_hi > s_lo, f"SSIM rose from q{q_hi} ({s_hi}) to q{q_lo} ({s_lo})"


def _ssim_from_the_definition(a, b, peak=255.0):
    """Wang et al. 2004 eq. 13, written out independently of the shipped code.

    The point of the duplication is the constant PLACEMENT: C1 = (K1 L)^2 goes
    with the luminance (mean) terms and C2 = (K2 L)^2 with the contrast/structure
    (variance/covariance) terms. Swapping them leaves SSIM(x, x) == 1 and leaves
    it monotone in JPEG quality, so every property-style test still passes while
    every number is wrong. Only a value check catches it, and a value check is
    only trustworthy against a second derivation.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    k1, k2, win, sigma = 0.01, 0.03, 11, 1.5
    c1, c2 = (k1 * peak) ** 2, (k2 * peak) ** 2
    g = cv2.getGaussianKernel(win, sigma)
    kernel = g @ g.T                       # separable -> the full 2-D window

    def w(x):
        return cv2.filter2D(x, cv2.CV_64F, kernel, borderType=cv2.BORDER_REFLECT_101)

    mu_x, mu_y = w(a), w(b)
    sigma_x2 = w(a * a) - mu_x ** 2
    sigma_y2 = w(b * b) - mu_y ** 2
    sigma_xy = w(a * b) - mu_x * mu_y
    luminance = (2.0 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1)
    contrast_structure = (2.0 * sigma_xy + c2) / (sigma_x2 + sigma_y2 + c2)
    return float(np.mean(luminance * contrast_structure))


def test_ssim_matches_an_independent_write_out_of_wang_2004():
    """Pins the VALUES, not just the shape of the curve — see the docstring
    above for why the property tests alone cannot do it."""
    ref = synthetic_scene()
    ref_y = to_luma(ref)
    for q in JPEG_LADDER:
        test_y = to_luma(jpeg_roundtrip(ref, q)[0])
        mine, theirs = ssim(ref_y, test_y), _ssim_from_the_definition(ref_y, test_y)
        assert abs(mine - theirs) < 1e-9, (q, mine, theirs)


def test_ssim_constants_are_not_transposed():
    """C1 belongs to the mean term and C2 to the variance term. On constant
    patches the variance term is exactly 1, so SSIM collapses to the closed form
    (2*mu_a*mu_b + C1) / (mu_a^2 + mu_b^2 + C1) and C1 is the only free number
    left — which is what makes this the case that separates the two placements.
    """
    peak = 255.0
    c1, c2 = (0.01 * peak) ** 2, (0.03 * peak) ** 2
    a = np.full((64, 64), 10.0)
    b = np.full((64, 64), 200.0)
    correct = (2 * 10 * 200 + c1) / (10 ** 2 + 200 ** 2 + c1)
    transposed = (2 * 10 * 200 + c2) / (10 ** 2 + 200 ** 2 + c2)

    got = ssim(a, b)
    assert abs(got - correct) < 1e-9, (got, correct)
    # the test is only meaningful if the two placements actually differ here
    assert abs(correct - transposed) > 1e-5, (correct, transposed)
    assert abs(got - transposed) > 1e-5, "this case cannot tell the two apart"


def test_ssim_is_computed_on_luma_not_on_a_colour_average():
    """to_luma must be BT.601, not a plain channel mean: a pure-blue and a
    pure-red patch have very different luma and identical channel means."""
    red = np.zeros((16, 16, 3), np.uint8)
    red[..., 2] = 255                                  # BGR
    blue = np.zeros((16, 16, 3), np.uint8)
    blue[..., 0] = 255
    assert abs(float(to_luma(red).mean()) - 0.299 * 255) < 1.0
    assert abs(float(to_luma(blue).mean()) - 0.114 * 255) < 1.0


# =============================================================================
# task-level proxies
# =============================================================================
def test_orb_inlier_rate_is_one_against_the_same_image():
    ref = synthetic_scene()
    orb = OrbScorer(ref, nfeatures=2000, radius_px=1.5)
    assert orb.reference_keypoints > 100, orb.reference_keypoints
    n, rate = orb.score(ref)
    assert n == orb.reference_keypoints
    assert abs(rate - 1.0) < 1e-9, rate


def test_orb_inlier_rate_falls_faster_than_orb_count():
    """Measured on real frames: count moved 1833 -> 1816 while the inlier rate
    collapsed 1.000 -> 0.827, and count can even RISE because blocking artefacts
    manufacture corners. Ranking on count would pick the worse encoder, so the
    asymmetry is pinned here."""
    ref = synthetic_scene()
    orb = OrbScorer(ref, nfeatures=2000, radius_px=1.5)
    n_hi, r_hi = orb.score(jpeg_roundtrip(ref, 97)[0])
    n_lo, r_lo = orb.score(jpeg_roundtrip(ref, 20)[0])

    count_change = abs(n_hi - n_lo) / max(1, n_hi)
    rate_change = abs(r_hi - r_lo) / max(1e-9, r_hi)
    assert r_lo < r_hi, (r_hi, r_lo)
    assert rate_change > count_change, (
        f"count moved {count_change:.3f} but the inlier rate only "
        f"{rate_change:.3f}; the rate is supposed to be the sensitive one")


def test_laplacian_variance_is_never_a_ranking_metric():
    """It is non-monotonic under compression — measured here as well as on real
    frames — so the ranker must refuse it outright rather than let a caller pass
    it in and get a confidently wrong answer."""
    ref = synthetic_scene()
    values = [laplacian_var(jpeg_roundtrip(ref, q)[0]) for q in JPEG_LADDER]
    assert not all(a > b for a, b in zip(values, values[1:])), \
        f"Laplacian variance came out monotone ({values}); the caveat needs a rethink"

    assert "lap_var" not in RANKABLE_METRICS
    try:
        rank_rows([], target_fps=1.0, budget_mbps=100.0, rank_by="lap_var")
    except ValueError:
        return
    raise AssertionError("rank_rows accepted Laplacian variance as a ranking metric")


def test_apriltag_false_ids_are_counted_separately_from_detections():
    """A compressed frame detected 13 tags where the reference detected 11 — the
    extras were false positives, which are worse for pose than a miss. So the
    known-ID map must subtract them rather than credit them."""
    detector = TagDetector("36h11")
    if not detector.available:            # no backend: degrade, do not fail
        assert detector.detect(tag_scene()) == (0, 0, 0)
        return
    scene = tag_scene(ids=(3, 17))
    detected, known, false = detector.detect(scene)
    assert detected == 2, detected

    partial = TagDetector("36h11", known_ids={3})
    detected, known, false = partial.detect(scene)
    assert (detected, known, false) == (2, 1, 1), (detected, known, false)


def test_tag_detector_degrades_gracefully_when_no_backend_exists():
    detector = TagDetector("36h11")
    detector._impl = None                 # simulate a build without aruco
    detector.backend = "none"
    assert detector.available is False
    assert detector.detect(synthetic_scene()) == (0, 0, 0)


# =============================================================================
# reference construction
# =============================================================================
def test_median_stack_recovers_the_clean_scene_from_noisy_frames():
    """This is what makes a separate raw run usable as a reference at all."""
    clean = synthetic_scene()
    rng = np.random.default_rng(11)
    noisy = [np.clip(clean.astype(np.float64) + rng.normal(0, 6, clean.shape),
                     0, 255).astype(np.uint8) for _ in range(15)]
    stacked = median_reference(noisy)
    single = psnr_db(to_luma(clean), to_luma(noisy[0]))
    pooled = psnr_db(to_luma(clean), to_luma(stacked))
    assert pooled > single + 3.0, (single, pooled)


def test_a_compressed_run_is_refused_as_the_reference():
    """The measured failure this guards: scoring against JPEG frames inverted the
    ranking — q95 (46.1 kB) scored 47.49 dB against q90 (34.2 kB) at 52.22 dB."""
    raw = {"setting": {"codec": "none"}}
    for codec in ("mjpeg", "h264", "h265"):
        assert not reference_is_lossless({"setting": {"codec": codec}})
    assert reference_is_lossless(raw)

    with tempfile.TemporaryDirectory() as td:
        run = {"dir": Path(td), "setting": {"codec": "mjpeg"}, "frames": []}
        try:
            build_reference(run, 5)
        except ValueError:
            return
    raise AssertionError("build_reference accepted a compressed reference")


# =============================================================================
# sweep axes and stream parsing
# =============================================================================
def test_encoder_axes_fold_per_codec_instead_of_multiplying_out():
    """A bitrate means nothing to MJPEG and a quality means nothing to H.26x, so
    the full product would re-run identical configurations under new names."""
    settings = expand_settings(["none", "mjpeg", "h264"],
                               ["80", "90"], ["2000", "8000"], ["0"])
    slugs = [s.slug for s in settings]
    assert slugs == ["none", "mjpeg_q80", "mjpeg_q90",
                     "h264_2000k_kf0", "h264_8000k_kf0"], slugs
    assert len(slugs) == len(set(slugs))
    # 'none' has no rate axis at all, however many rates are requested
    assert len(expand_settings(["none"], ["80", "90"], ["1000", "2000"], ["0", "1"])) == 1


def test_setting_strings_round_trip():
    assert parse_setting("none") == EncodeSetting("none")
    assert parse_setting("mjpeg@97") == EncodeSetting("mjpeg", quality=97)
    assert parse_setting("h265@8000/kf1") == \
        EncodeSetting("h265", bitrate_kbps=8000, keyframe=1)
    assert parse_setting("h264@4000").keyframe == 0
    for bad in ("h264", "av1@100", "mjpeg@x"):
        try:
            parse_setting(bad)
        except (ValueError, TypeError):
            continue
        raise AssertionError(f"parse_setting({bad!r}) should have been rejected")


def test_settings_carry_their_knob_into_the_stream_config():
    base = {"fps": 20.0}
    assert EncodeSetting("mjpeg", quality=97).apply(base)["mjpeg_quality"] == 97
    h = EncodeSetting("h265", bitrate_kbps=8000, keyframe=1).apply(base)
    assert h["video_bitrate_kbps"] == 8000 and h["video_keyframe_frequency"] == 1
    assert "video_bitrate_kbps" not in EncodeSetting("mjpeg", quality=90).apply(base)


# =============================================================================
# Annex-B keyframe detection
# =============================================================================
def test_keyframe_detection_reads_the_nal_type_not_the_missing_api():
    """depthai 2.32's ImgFrame has no getFrameType(), so source.py types every
    packet "" and any 'start at the first I frame' gate stays shut. The bytes
    still say which it is."""
    idr = b"\x00\x00\x00\x01\x67sps\x00\x00\x01\x68pps\x00\x00\x01\x65idr"
    assert annexb_frame_type(idr, "h264") == "I"
    assert annexb_frame_type(b"\x00\x00\x00\x01\x41slice", "h264") == "P"
    # H.265 NAL type lives in bits 1..6: IDR_W_RADL is 19 -> 0x26.
    assert annexb_frame_type(b"\x00\x00\x00\x01\x26idr", "h265") == "I"
    assert annexb_frame_type(b"\x00\x00\x00\x01\x02trail", "h265") == "P"
    assert annexb_frame_type(b"", "h264") == "P"


# =============================================================================
# ranking — quality at a target frame rate, not bandwidth
# =============================================================================
def _row(label, fps, mbps, inlier, **extra):
    row = {f: "" for f in FIELDS}
    row.update({"ok": 1, "label": label, "run": label, "codec": "h265",
                "color_fps": fps, "measured_mbps": mbps,
                "orb_inlier_rate": inlier, "scored_frames": 10,
                "extraction_ok": 1})
    row.update(extra)
    return row


def _packet(seq: int, raw: bytes, header: bytes = b"") -> "Frame":
    from c3_camera.source import Frame
    return Frame(name="color", image=None, seq=seq, t_device=1.0 + seq * 0.05,
                 t_recv=1.01 + seq * 0.05, latency_ms=10.0, raw=raw,
                 encoding="h264", codec_header=header)


def test_encoded_recording_starts_at_a_keyframe_and_caps_at_save_frames():
    """A stream that opens on a P frame decodes to fewer images than there are
    sidecar rows, which c3_extract_rgb_video treats as a hard failure — correctly,
    because it means the dataset would be misaligned. So the leading P frames
    have to be dropped here rather than discovered later."""
    from c3_camera.c3_encode_quality import _EncodedWriter

    idr = b"\x00\x00\x00\x01\x65idr-payload"
    p_frame = b"\x00\x00\x00\x01\x41p-payload"
    header = b"\x00\x00\x00\x01\x67sps\x00\x00\x01\x68pps"
    with tempfile.TemporaryDirectory() as td:
        run = Path(td) / "run"
        run.mkdir()
        w = _EncodedWriter(run, "h264", max_frames=3)
        assert w.add(_packet(1, p_frame, header)) is False   # before any keyframe
        assert w.add(_packet(2, idr, header)) is True
        assert w.add(_packet(3, p_frame)) is True
        assert w.add(_packet(3, p_frame)) is False           # duplicate seq
        assert w.add(_packet(4, p_frame)) is True
        assert w.add(_packet(5, p_frame)) is False           # max_frames reached
        w.close()

        assert (run / "rgb.h264").read_bytes() == header + idr + p_frame * 2
        rows = list(csv.DictReader((run / "rgb_video.csv").open()))
        assert [r["frame_type"] for r in rows] == ["I", "P", "P"]
        assert [r["rgb_file"] for r in rows] == [f"rgb/{i:06d}.png" for i in (1, 2, 3)]
        assert [int(r["byte_offset"]) for r in rows] == \
            [len(header), len(header) + len(idr), len(header) + len(idr) + len(p_frame)]


def test_a_recorded_h264_run_decodes_through_the_dataset_extractor():
    """What is being measured is the codec *as the dataset pipeline delivers it*,
    so the recording must survive the very extractor c3_collect datasets use —
    including its refusal to proceed when the decoded count does not match."""
    import shutil
    import subprocess
    from c3_camera.c3_encode_quality import _EncodedWriter
    from c3_camera.c3_extract_rgb_video import extract_rgb_video

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        src.mkdir()
        for i in range(6):
            cv2.imwrite(str(src / f"{i + 1:03d}.png"),
                        synthetic_scene(w=128, h=96, seed=i))
        stream = Path(td) / "in.h264"
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                        "-framerate", "20", "-i", str(src / "%03d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-bf", "0",
                        "-g", "6", "-f", "h264", str(stream)], check=True)
        units = _split_access_units(stream.read_bytes())
        assert len(units) == 6, f"expected 6 access units, split gave {len(units)}"

        run = Path(td) / "run"
        run.mkdir()
        w = _EncodedWriter(run, "h264", max_frames=len(units))
        # A leading P packet, as arrives when capture starts mid-GOP.
        w.add(_packet(0, b"\x00\x00\x00\x01\x41stale"))
        for seq, unit in enumerate(units, 1):
            assert w.add(_packet(seq, unit)) is True
        w.close()

        summary = extract_rgb_video(run)
        assert summary["frames_decoded"] == len(units)
        for i in range(1, len(units) + 1):
            img = cv2.imread(str(run / f"rgb/{i:06d}.png"))
            assert img is not None and img.shape[:2] == (96, 128)


def _split_access_units(data: bytes) -> list[bytes]:
    """Split an Annex-B H.264 stream into one packet per coded picture.

    The hardware encoder hands the host one message per frame; ffmpeg hands us
    one file. Splitting on the first VCL NAL of each picture reconstructs the
    packet boundaries the device would have produced.
    """
    starts = []
    i = data.find(b"\x00\x00\x01")
    while i != -1:
        begin = i - 1 if i > 0 and data[i - 1] == 0 else i
        starts.append((begin, i + 3))
        i = data.find(b"\x00\x00\x01", i + 3)

    units: list[bytes] = []
    group_start: int | None = None
    for n, (begin, payload) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(data)
        if group_start is None:
            group_start = begin             # parameter sets / SEI open a group
        if (data[payload] & 0x1F) in (1, 5):        # a coded slice closes it
            units.append(data[group_start:end])
            group_start = None
    if group_start is not None:
        units.append(data[group_start:])
    return units


def test_ranking_prefers_quality_at_the_target_fps_over_small_files():
    """The cheapest stream always wins on bytes — bytes are an input parameter.
    A setting that only reaches the quality by dropping to 9 fps must not rank."""
    rows = [
        _row("cheap-but-bad", 20.0, 2.0, 0.61),
        _row("good", 20.0, 8.0, 0.94),
        _row("best-but-slow", 9.1, 124.0, 0.99),
        _row("over-budget", 20.0, 80.0, 0.97),
    ]
    good, bad = rank_rows(rows, target_fps=19.0, budget_mbps=25.0,
                          rank_by="orb_inlier_rate")
    assert [r["label"] for r in good] == ["good", "cheap-but-bad"]
    rejected = {r["label"]: r["_reject"] for r in bad}
    assert "fps" in rejected["best-but-slow"]
    assert "budget" in rejected["over-budget"]


def test_unrecoverable_frames_disqualify_a_setting_however_good_it_looks():
    """A codec whose stream does not decode afterwards is not a better codec."""
    rows = [_row("pretty", 20.0, 4.0, 0.99, extraction_ok=0),
            _row("boring", 20.0, 6.0, 0.90)]
    good, bad = rank_rows(rows, target_fps=19.0, budget_mbps=90.0,
                          rank_by="orb_inlier_rate")
    assert [r["label"] for r in good] == ["boring"]
    assert "recoverable" in bad[0]["_reject"]


def test_a_scene_that_moved_is_reported_instead_of_ranked_confidently():
    """Every cross-run number rests on the scene not changing between runs. The
    raw arm scored against the median of its own run is the one place that
    assumption can be checked from the data itself, so it must be checked."""
    from c3_camera.c3_encode_quality import static_scene_warning

    steady = [_row("raw", 9.0, 124.0, 0.97, codec="none"),
              _row("h265", 20.0, 8.0, 0.90)]
    assert static_scene_warning(steady) is None

    moved = [_row("raw", 9.0, 124.0, 0.11, codec="none"),
             _row("h265", 20.0, 8.0, 0.09)]
    text = static_scene_warning(moved)
    assert text and "NOT STATIC" in text and "0.110" in text

    buf = io.StringIO()
    with redirect_stdout(buf):
        print_report(moved, target_fps=1.0, budget_mbps=200.0,
                     rank_by="orb_inlier_rate", scene="static")
    assert "SCENE WAS NOT STATIC" in buf.getvalue()


def test_the_correspondence_caveat_is_printed_with_every_table():
    """The number must not be able to travel without the sentence that says it is
    not a same-frame PSNR."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_report([_row("good", 20.0, 8.0, 0.94)], target_fps=19.0,
                     budget_mbps=25.0, rank_by="orb_inlier_rate", scene="static")
    out = buf.getvalue()
    assert CORRESPONDENCE_NOTE in out
    assert "NOT same-frame" in out
    for note in CAVEATS:
        assert note in out
    assert "DECISION" in out


# =============================================================================
# offline scoring, end to end, with no hardware
# =============================================================================
def _fake_capture(root: Path, n_raw: int = 12, n_test: int = 6,
                  qualities=(95, 30)) -> Path:
    """Write a capture directory of the shape run_one() produces."""
    root.mkdir(parents=True, exist_ok=True)
    clean = synthetic_scene()
    rng = np.random.default_rng(3)

    def noisy():
        return np.clip(clean.astype(np.float64) + rng.normal(0, 3, clean.shape),
                       0, 255).astype(np.uint8)

    def write_run(slug: str, setting: dict, images, ext: str, quality=None):
        run = root / slug
        (run / "frames").mkdir(parents=True, exist_ok=True)
        rows = []
        for i, img in enumerate(images, 1):
            name = f"frames/{i:06d}.{ext}"
            if ext == "jpg":
                cv2.imwrite(str(run / name), img,
                            [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
            else:
                cv2.imwrite(str(run / name), img)
            rows.append({"idx": i, "seq": 100 + i, "t_device": f"{i * 0.05:.6f}",
                         "latency_ms": 40.0, "wire_bytes": 1000,
                         "exposure_ms": 30.0, "file": name})
        with (run / "frames.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        row = {f: "" for f in FIELDS}
        row.update({"ok": 1, "run": slug, "label": setting["label"],
                    "codec": setting["codec"], "color_fps": 20.0,
                    "measured_mbps": 8.0, "extraction_ok": 1,
                    "frames_saved": len(rows), "scene": "static"})
        (run / "meta.json").write_text(json.dumps(
            {"setting": setting, "scene": "static", "row": row}, indent=2))

    write_run("run_01_none",
              {"codec": "none", "quality": None, "bitrate_kbps": None,
               "keyframe": None, "label": "none (raw NV12)", "slug": "none"},
              [noisy() for _ in range(n_raw)], "png")
    for k, q in enumerate(qualities, start=2):
        write_run(f"run_{k:02d}_mjpeg_q{q}",
                  {"codec": "mjpeg", "quality": q, "bitrate_kbps": None,
                   "keyframe": None, "label": f"mjpeg q{q}",
                   "slug": f"mjpeg_q{q}"},
                  [noisy() for _ in range(n_test)], "jpg", quality=q)
    return root


def test_offline_scoring_reproduces_the_quality_ordering_without_a_camera():
    with tempfile.TemporaryDirectory() as td:
        out = _fake_capture(Path(td) / "cap")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rows = score_directory(
                out, reference_frames=6, orb_features=2000,
                match_radius_px=1.5, min_ref_keypoints=100, sample=1,
                tag_family="36h11", known_ids=None, verbose=True)
        by_label = {r["label"]: r for r in rows}
        assert set(by_label) == {"none (raw NV12)", "mjpeg q95", "mjpeg q30"}

        hi, lo = by_label["mjpeg q95"], by_label["mjpeg q30"]
        assert float(hi["psnr_y_db"]) > float(lo["psnr_y_db"]), (hi, lo)
        assert float(hi["ssim_y"]) > float(lo["ssim_y"])
        assert float(hi["orb_inlier_rate"]) > float(lo["orb_inlier_rate"])
        for r in rows:
            assert 0.0 <= float(r["ssim_y"]) <= 1.0
            assert r["scored_frames"] == 6

        # the raw arm is scored on the frames it did NOT contribute to the median
        raw = by_label["none (raw NV12)"]
        assert "noise floor" in raw["quality_note"]

        assert (out / "scores.csv").is_file() and (out / "runs.csv").is_file()
        scores = list(csv.DictReader((out / "scores.csv").open()))
        assert len(scores) == 18, len(scores)
        assert {s["run"] for s in scores} == {
            "run_01_none", "run_02_mjpeg_q95", "run_03_mjpeg_q30"}


def test_the_noise_floor_never_scores_a_frame_it_put_into_the_median():
    """The raw arm's row is the yardstick every encoder row is read against, so
    it must not be compared with an image it helped build. Selecting its scoring
    frames by row offset is only equivalent while every file is readable — one
    unreadable PNG early in the run slides the two sets into overlapping and
    quietly flatters the floor."""
    from c3_camera.c3_encode_quality import (load_run, reference_images,
                                             run_images, score_runs)

    with tempfile.TemporaryDirectory() as td:
        out = _fake_capture(Path(td) / "cap", n_raw=12)
        raw_dir = out / "run_01_none"
        (raw_dir / "frames" / "000003.png").unlink()      # a dropped frame
        run = load_run(raw_dir)

        used = {i for i, _, _ in reference_images(run, 6)}
        assert len(used) == 6 and 3 not in used, used

        buf = io.StringIO()
        with redirect_stdout(buf):
            _, aggregates = score_runs([run], run, reference_frames=6,
                                       min_ref_keypoints=100, verbose=False)
        scored = aggregates["run_01_none"]["scored_frames"]
        readable = len(run_images(run, start=0))
        assert scored == readable - len(used), (scored, readable, used)


def test_the_decision_line_reports_its_own_margin_and_scatter():
    """A winner announced without a spread reads as a result when it may be a
    coin toss: adjacent rungs of a bitrate ladder routinely differ by less than
    one per-frame standard deviation."""
    from c3_camera.c3_encode_quality import decision_margin

    tie = [_row("h264 4000k", 20.0, 4.0, 0.8535, orb_inlier_sd=0.0089),
           _row("h265 4000k", 20.0, 4.0, 0.8494, orb_inlier_sd=0.0087)]
    text = decision_margin(tie, "orb_inlier_rate")
    assert "+0.0041" in text or "+0.0042" in text, text
    assert "UNRESOLVED" in text, text

    clear = [_row("h265 8000k", 20.0, 8.0, 0.94, orb_inlier_sd=0.005),
             _row("mjpeg q60", 20.0, 3.0, 0.65, orb_inlier_sd=0.006)]
    assert "UNRESOLVED" not in decision_margin(clear, "orb_inlier_rate")

    buf = io.StringIO()
    with redirect_stdout(buf):
        print_report(tie, target_fps=19.0, budget_mbps=25.0,
                     rank_by="orb_inlier_rate", scene="static")
    assert "UNRESOLVED" in buf.getvalue()


def test_the_host_decode_path_bias_is_printed_not_just_known():
    """The raw arm is converted by OpenCV, MJPEG by libjpeg and H.26x by ffmpeg
    inside the extractor. Measured on this host, a BIT-EXACT H.264 stream scored
    through that path lands at 45.6 dB / SSIM 0.9962 / ORB 0.916 — a ceiling the
    H.26x rows cannot beat at any bitrate, and which MJPEG does not pay. A reader
    of the table has to be told."""
    text = "\n".join(CAVEATS)
    assert "45.6 dB" in text and "ZERO codec loss" in text
    assert "c3_extract_rgb_video" in text
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_report([_row("good", 20.0, 8.0, 0.94)], target_fps=19.0,
                     budget_mbps=25.0, rank_by="orb_inlier_rate", scene="static")
    assert "45.6 dB" in buf.getvalue()


def test_the_run_order_confound_is_printed():
    """Settings run none -> mjpeg -> h264 -> h265 and the reference is captured
    first, so on a ~10 minute ladder codec is confounded with elapsed time and
    the H.265 rows always ran last."""
    text = "\n".join(CAVEATS)
    assert "confounded with elapsed time" in text
    assert "ran late" in text or "ran last" in text


def test_offline_scoring_refuses_a_capture_with_no_raw_arm():
    """Without the raw arm there is no fidelity baseline; failing loudly beats
    quietly producing a table of numbers that mean nothing."""
    with tempfile.TemporaryDirectory() as td:
        out = _fake_capture(Path(td) / "cap")
        (out / "run_01_none" / "meta.json").write_text(json.dumps(
            {"setting": {"codec": "mjpeg", "label": "mjpeg q90"},
             "scene": "static", "row": {}}))
        try:
            score_directory(out, reference_frames=6, orb_features=500,
                            match_radius_px=1.5, min_ref_keypoints=100,
                            sample=1, tag_family="36h11", known_ids=None,
                            verbose=False)
        except RuntimeError as e:
            assert "raw" in str(e)
            return
    raise AssertionError("scoring proceeded without an uncompressed reference")


def test_offline_scoring_refuses_to_eat_its_whole_reference_run():
    with tempfile.TemporaryDirectory() as td:
        out = _fake_capture(Path(td) / "cap", n_raw=6)
        try:
            score_directory(out, reference_frames=6, orb_features=500,
                            match_radius_px=1.5, min_ref_keypoints=100,
                            sample=1, tag_family="36h11", known_ids=None,
                            verbose=False)
        except RuntimeError as e:
            assert "reference-frames" in str(e)
            return
    raise AssertionError("the reference run was consumed entirely and still scored")


def test_offline_mode_runs_end_to_end_through_the_cli():
    """--offline must never construct a C3Source; this is the whole reason
    scoring is a separate pass."""
    with tempfile.TemporaryDirectory() as td:
        out = _fake_capture(Path(td) / "cap")
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(["--offline", str(out), "--reference-frames", "6",
                         "--min-ref-keypoints", "100", "--target-fps", "19"])
        assert code == 0
        text = buf.getvalue()
        assert "no device is opened" in text
        assert CORRESPONDENCE_NOTE in text
        assert (out / "report.md").is_file()
        report = (out / "report.md").read_text()
        assert CORRESPONDENCE_NOTE in report and "| setting |" in report


def test_dry_run_resolves_the_ladder_without_touching_the_camera():
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(["--dry-run", "--codec", "none,mjpeg,h265",
                     "--bitrate-kbps", "2000,8000"])
    assert code == 0
    text = buf.getvalue()
    assert "nothing was opened" in text
    for expected in ("none (raw NV12)", "mjpeg q90", "h265 2000k", "h265 8000k"):
        assert expected in text, expected


# =============================================================================
# bare runner, so this works without pytest installed
# =============================================================================
def _main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
