#!/usr/bin/env python3
"""
test_offline.py — everything that can be checked without the camera attached.

    ~/.venvs/c3-depthai/bin/python -m pytest c3_camera/tests/test_offline.py -v
    ~/.venvs/c3-depthai/bin/python c3_camera/tests/test_offline.py      # no pytest

Covers the parts where a bug would be silent: the deprojection maths, the
resolution/bandwidth resolution logic, and the CLI parsing. Anything that needs
a live device belongs in a bench run, not here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from c3_camera import viz
from c3_camera.config import (
    COLOR_RESOLUTIONS,
    MONO_RESOLUTIONS,
    POE_BUDGET_MBPS,
    StreamConfig,
    _parse_fraction,
    _parse_size,
    _parse_streams,
)
from c3_camera.geometry import Intrinsics, depth_to_xyz, rgbd_to_pointcloud

INTR = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=180.0, width=640, height=360)


# =============================================================================
# geometry
# =============================================================================
def _depth_with(points: dict[tuple[int, int], int]) -> np.ndarray:
    d = np.zeros((360, 640), np.uint16)
    for (row, col), mm in points.items():
        d[row, col] = mm
    return d


def test_deprojection_matches_hand_computed_values():
    """x = (u - cx) * z / fx, and the invalid/range filters actually filter."""
    d = _depth_with({
        (180, 320): 1000,   # principal point at 1 m      -> (0, 0, 1)
        (180, 420): 1000,   # 100 px right at 1 m         -> (0.2, 0, 1)
        (280, 320): 2000,   # 100 px down at 2 m          -> (0, 0.4, 2)
        (10, 10): 50,       # closer than min_mm          -> dropped
        (11, 11): 30000,    # further than max_mm         -> dropped
    })
    xyz = depth_to_xyz(d, INTR, stride=1)
    assert len(xyz) == 3
    want = sorted([[0.0, 0.0, 1.0], [0.2, 0.0, 1.0], [0.0, 0.4, 2.0]])
    assert np.allclose(sorted(xyz.tolist()), want, atol=1e-6)


def test_zero_depth_is_no_measurement_not_zero_distance():
    assert len(depth_to_xyz(np.zeros((16, 16), np.uint16),
                            Intrinsics(1, 1, 8, 8, 16, 16))) == 0


def test_stride_does_not_shift_geometry():
    """Subsampling must keep true pixel coordinates, not renumber them."""
    d = _depth_with({(180, 420): 1000})
    assert np.allclose(depth_to_xyz(d, INTR, stride=2), [[0.2, 0.0, 1.0]], atol=1e-6)


def test_coloured_cloud_resamples_and_converts_bgr_to_rgb():
    colour = np.zeros((720, 1280, 3), np.uint8)
    colour[360, 840] = (10, 20, 30)            # BGR, at 2x the depth grid
    d = _depth_with({(180, 420): 1000})
    xyz, rgb = rgbd_to_pointcloud(colour, d, INTR, stride=1)
    assert np.allclose(xyz, [[0.2, 0.0, 1.0]], atol=1e-6)
    assert rgb.tolist() == [[30, 20, 10]]


def test_intrinsics_rescale():
    s = INTR.scaled_to(320, 180)
    assert (s.fx, s.fy, s.cx, s.cy) == (250.0, 250.0, 160.0, 90.0)


def test_depth_map_must_be_2d():
    for bad in (np.zeros((4, 4, 3), np.uint16), np.zeros(4, np.uint16)):
        try:
            depth_to_xyz(bad, INTR)
        except ValueError:
            continue
        raise AssertionError("expected ValueError for a non-2-D depth map")


# =============================================================================
# visualisation
# =============================================================================
def test_invalid_depth_renders_black_and_valid_does_not():
    d = _depth_with({(180, 320): 1000})
    img = viz.colorize_depth(d, 300, 6000)
    assert tuple(img[0, 0]) == (0, 0, 0)
    assert tuple(img[180, 320]) != (0, 0, 0)


def test_depth_stats_survives_an_empty_map():
    st = viz.depth_stats(np.zeros((8, 8), np.uint16))
    assert st["valid_pct"] == 0.0 and st["median_mm"] == 0


def test_hstack_pad_mixes_sizes_and_channel_counts():
    out = viz.hstack_pad([np.zeros((100, 200, 3), np.uint8),
                          np.zeros((50, 50), np.uint8)])
    assert out.ndim == 3 and out.shape[0] == 100


# =============================================================================
# configuration
# =============================================================================
def test_isp_scale_shrinks_output_and_keeps_even_dimensions():
    rc = StreamConfig(color_res="1080p", isp_scale=(1, 3)).resolve()
    assert rc.color_out_size == (640, 360)
    for n, d in ((1, 2), (1, 3), (1, 4), (2, 3), (3, 4)):
        out = StreamConfig(color_res="1080p", isp_scale=(n, d)).resolve().color_out_size
        assert out[0] % 2 == 0 and out[1] % 2 == 0, (n, d, out)


def test_isp_scale_beyond_hardware_scaler_limits_is_rejected():
    for bad in ((17, 1), (1, 64), (0, 2)):
        try:
            StreamConfig(isp_scale=bad).resolve()
        except ValueError:
            continue
        raise AssertionError(f"--isp-scale {bad} should have been rejected")


def test_depth_aligned_to_colour_takes_the_colour_aspect_ratio():
    rc = StreamConfig(color_res="1080p", isp_scale=(1, 2),
                      mono_res="400p", depth_align="color").resolve()
    dw, dh = rc.depth_out_size
    assert (dw, dh) == (640, 360)
    assert abs(dw / dh - rc.color_out_size[0] / rc.color_out_size[1]) < 0.01


def test_unaligned_depth_keeps_the_mono_size():
    rc = StreamConfig(mono_res="720p", depth_align="none").resolve()
    assert rc.depth_out_size == (1280, 720)


def test_bandwidth_arithmetic_is_bytes_times_fps():
    """NV12 is 1.5 B/px and depth16 is 2 B/px; the totals must say so.

    color_encode is pinned explicitly: the default is mjpeg, whose size is
    scene-dependent, so the raw arithmetic has to be asked for.
    """
    rc = StreamConfig(color_res="1080p", isp_scale=(1, 2), mono_res="400p",
                      streams=("color", "depth"), fps=10.0,
                      color_encode="none").resolve()
    bw = rc.bandwidth_mbps()
    expect_color = 960 * 540 * 1.5 * 10 * 8 / 1e6
    expect_depth = 640 * 360 * 2.0 * 10 * 8 / 1e6
    assert abs(bw["color"] - expect_color) < 0.1
    assert abs(bw["depth"] - expect_depth) < 0.1
    assert abs(bw["total"] - (expect_color + expect_depth)) < 0.2


def test_measured_link_ceiling_predicts_the_observed_fps():
    """Raw 960x540+depth measured 9.1-9.26 fps; the budget model must agree.

    This is the check that keeps POE_BUDGET_MBPS honest: bytes/frame at the
    ceiling should land within a few percent of what the camera really delivered
    across three separate bench runs.
    """
    rc = StreamConfig(color_res="1080p", isp_scale=(1, 2), mono_res="400p",
                      color_encode="none", fps=1.0).resolve()
    per_frame_mbit = rc.bandwidth_mbps()["total"]        # fps=1 -> Mbit per frame
    predicted_fps = POE_BUDGET_MBPS / per_frame_mbit
    assert 8.5 <= predicted_fps <= 10.0, predicted_fps   # measured 9.1-9.26

    # Full 1080p raw + depth measured 3.18-3.21 fps.
    rc2 = StreamConfig(color_res="1080p", isp_scale=None, mono_res="400p",
                       color_encode="none", fps=1.0).resolve()
    predicted2 = POE_BUDGET_MBPS / rc2.bandwidth_mbps()["total"]
    assert 2.9 <= predicted2 <= 3.5, predicted2


def test_over_budget_configs_are_visible_as_such():
    tight = StreamConfig(color_res="1080p", isp_scale=None, fps=30.0,
                         color_encode="none").resolve()
    assert tight.bandwidth_mbps()["total"] > POE_BUDGET_MBPS
    assert "OVER BUDGET" in tight.budget_note()
    ok = StreamConfig(color_res="1080p", isp_scale=(1, 4), fps=10.0,
                      color_encode="none", streams=("color",)).resolve()
    assert "ok" in ok.budget_note()


def test_mjpeg_size_warning_does_not_fire_on_verified_working_sizes():
    """960x540 and 480x270 both encode fine on this device, so neither should
    raise an encoder-alignment warning even though their heights are not
    multiples of 8."""
    for scale in ((1, 2), (1, 4)):
        rc = StreamConfig(color_res="1080p", isp_scale=scale,
                          color_encode="mjpeg").resolve()
        assert not any("multiple of 32" in w for w in rc.warnings), (scale, rc.warnings)


def test_default_config_fits_the_measured_link_budget():
    """The shipped default must not be an over-budget setting."""
    rc = StreamConfig().resolve()
    assert rc.bandwidth_mbps()["total"] <= POE_BUDGET_MBPS, rc.budget_note()


def test_h26x_bandwidth_uses_configured_bitrate():
    """Inter-frame encoder traffic is bitrate-controlled, not pixels times fps."""
    rc = StreamConfig(
        color_encode="h264", video_bitrate_kbps=4000,
        color_res="1080p", isp_scale=(1, 4), depth_size=(480, 270),
        streams=("color", "depth"), fps=20.0,
    ).resolve()
    bw = rc.bandwidth_mbps()
    assert abs(bw["color"] - 4.0) < 1e-9
    assert 45.0 < bw["total"] < 46.0, bw
    assert "ok" in rc.budget_note()


def test_hardware_video_pipeline_profiles_build_offline():
    from c3_camera.pipeline import build_pipeline

    for codec in ("h264", "h265"):
        cfg = StreamConfig(color_encode=codec, streams=("color", "depth"),
                           isp_scale=(1, 4), fps=20.0)
        pipeline, formats = build_pipeline(cfg.resolve())
        assert formats["color"] == codec
        assert formats["depth"] == "depth16"
        assert pipeline.getAllNodes()


def test_annexb_parameter_sets_are_recoverable_for_midstream_recording():
    from c3_camera.source import _annexb_parameter_sets

    h264 = (b"\x00\x00\x00\x01\x67sps"
            b"\x00\x00\x01\x68pps"
            b"\x00\x00\x01\x65idr")
    assert set(_annexb_parameter_sets(h264, "h264")) == {7, 8}

    # H.265 NAL type occupies bits 1..6: 32/33/34 -> 0x40/0x42/0x44.
    h265 = (b"\x00\x00\x00\x01\x40vps"
            b"\x00\x00\x01\x42sps"
            b"\x00\x00\x01\x44pps"
            b"\x00\x00\x01\x26idr")
    assert set(_annexb_parameter_sets(h265, "h265")) == {32, 33, 34}


def test_unknown_names_are_rejected_rather_than_guessed():
    for kwargs in ({"color_res": "1440p"}, {"mono_res": "1080p"},
                   {"depth_preset": "turbo"}, {"median": "9x9"},
                   {"color_encode": "av1"}, {"video_bitrate_kbps": 0},
                   {"video_keyframe_frequency": -1}, {"streams": ()},
                   {"streams": ("colour",)}):
        try:
            StreamConfig(**kwargs).resolve()
        except ValueError:
            continue
        raise AssertionError(f"{kwargs} should have been rejected")


def test_fps_above_the_sensor_maximum_warns_but_does_not_crash():
    rc = StreamConfig(color_res="12mp", fps=60.0).resolve()   # IMX378 caps 12MP at 30
    assert any("maximum" in w for w in rc.warnings)


def test_probed_tables_match_the_real_sensor_modes():
    """Guard against silently adding a mode this camera does not have."""
    assert set(COLOR_RESOLUTIONS) == {"1080p", "1352x1012", "2024x1520", "4k", "12mp"}
    assert set(MONO_RESOLUTIONS) == {"400p", "720p", "800p"}
    assert COLOR_RESOLUTIONS["1080p"][1:3] == (1920, 1080)
    assert MONO_RESOLUTIONS["400p"][1:3] == (640, 400)


def test_pair_tolerance_defaults_to_a_fraction_of_the_frame_interval():
    rc = StreamConfig(fps=10.0, pair_tolerance_ms=None).resolve()
    assert abs(rc.pair_tolerance_s() - 0.075) < 1e-9
    rc2 = StreamConfig(fps=10.0, pair_tolerance_ms=20.0).resolve()
    assert abs(rc2.pair_tolerance_s() - 0.020) < 1e-9


# =============================================================================
# CLI parsing
# =============================================================================
def test_cli_parsers_stream():
    assert _parse_fraction("1/2") == (1, 2)
    assert _parse_fraction("none") is None
    assert _parse_fraction("1/1") is None          # 1/1 means "do not scale"
    assert _parse_size("640x360") == (640, 360)
    assert _parse_size("640,360") == (640, 360)
    assert _parse_streams("color,depth") == ("color", "depth")
    for bad, fn in (("half", _parse_fraction), ("640", _parse_size),
                    ("rgb", _parse_streams)):
        try:
            fn(bad)
        except Exception:
            continue
        raise AssertionError(f"{fn.__name__}({bad!r}) should have been rejected")


# =============================================================================
# recorder — the parts where a bug would silently corrupt research data
# =============================================================================
def _synthetic_recording(tmp: Path, n: int = 24, queue_size: int = 8,
                         depth_format: str = "png"):
    """Record n synthetic frames with known depth values; return (dir, depth, jpg)."""
    import cv2
    from c3_camera.recorder import Recorder
    from c3_camera.source import Bundle, Frame

    colour = np.full((360, 640, 3), 40, np.uint8)
    cv2.circle(colour, (320, 180), 80, (0, 0, 255), -1)
    jpg = bytes(cv2.imencode(".jpg", colour, [cv2.IMWRITE_JPEG_QUALITY, 95])[1])

    depth = np.zeros((270, 480), np.uint16)
    depth[100, 200] = 1234        # ordinary millimetre reading
    depth[150, 300] = 65535       # top of the uint16 range
    depth[200, 400] = 1           # smallest non-zero

    rec = Recorder(tmp, depth_format=depth_format, queue_size=queue_size,
                   verbose=False)
    rec.write_meta({"config": {"color_res": "1080p", "fps": 20},
                    "resolved": {"color_out": [640, 360], "depth_out": [480, 270]},
                    "device": {"product": "OAK-D-W-POE", "mxid": "synthetic"}})
    for i in range(n):
        c = Frame("color", colour, 100 + i, 10.0 + i * 0.05,
                  10.035 + i * 0.05, 35.0, 8.0, jpg)
        d = Frame("depth", depth, 100 + i, 10.0 + i * 0.05,
                  10.068 + i * 0.05, 68.0)
        rec.submit(Bundle(frames={"color": c, "depth": d},
                          fresh=frozenset({"color", "depth"})))
    stats = rec.close()
    return rec, stats, depth, jpg


def test_recorded_depth_is_bit_exact_millimetres():
    """A lossy or 8-bit write would silently corrupt every distance."""
    import cv2
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "rec"
        _, stats, depth, _ = _synthetic_recording(out, n=6)
        assert stats.written == 6
        back = cv2.imread(str(out / "depth/000001.png"), cv2.IMREAD_UNCHANGED)
        assert back.dtype == np.uint16, back.dtype
        assert np.array_equal(back, depth)
        assert back[100, 200] == 1234 and back[150, 300] == 65535
        assert back[200, 400] == 1


def test_recorded_mjpeg_colour_is_not_re_encoded():
    """With MJPEG on the wire the bytes should reach disk untouched."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "rec"
        _, _, _, jpg = _synthetic_recording(out, n=4)
        assert (out / "color/000001.jpg").read_bytes() == jpg


def test_recorder_does_not_write_the_same_frame_twice():
    """pair_mode=latest repeats a stream's frame across bundles; sequence
    numbers are what distinguishes a new frame from a repeat."""
    import tempfile
    from c3_camera.recorder import Recorder
    from c3_camera.source import Bundle, Frame
    with tempfile.TemporaryDirectory() as td:
        rec = Recorder(Path(td) / "rec", queue_size=8, verbose=False)
        colour = np.zeros((8, 8, 3), np.uint8)
        depth = np.zeros((8, 8), np.uint16)
        for _ in range(5):      # same seq every time
            b = Bundle(frames={"color": Frame("color", colour, 7, 1.0, 1.0, 1.0),
                               "depth": Frame("depth", depth, 7, 1.0, 1.0, 1.0)},
                       fresh=frozenset())
            rec.submit(b)
        stats = rec.close()
        assert stats.written == 1, f"wrote {stats.written} copies of one frame"


def test_recorder_burst_does_not_lose_frames():
    """Submitting far faster than the disk briefly fills the queue; the bounded
    wait should absorb that rather than dropping most of the burst."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        _, stats, _, _ = _synthetic_recording(Path(td) / "rec", n=40, queue_size=4)
        assert stats.written == 40, f"only {stats.written}/40 survived the burst"
        assert stats.dropped == 0, f"dropped {stats.dropped} frames"


def test_recording_index_and_meta_are_complete():
    import csv as _csv
    import json as _json
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "rec"
        _synthetic_recording(out, n=10)
        rows = list(_csv.DictReader((out / "frames.csv").open()))
        assert len(rows) == 10
        assert [int(r["color_seq"]) for r in rows] == list(range(100, 110))
        # timestamps must survive as capture times, not be replaced by write times
        assert abs(float(rows[3]["color_t_device"]) - (10.0 + 3 * 0.05)) < 1e-9
        for r in rows:
            assert (out / r["color_file"]).exists()
            assert (out / r["depth_file"]).exists()
        meta = _json.loads((out / "meta.json").read_text())
        # without the clock offset, t_device is meaningless after the process ends
        assert "dai_now_s" in meta["clock"] and "time_time_s" in meta["clock"]
        assert meta["depth_units"].startswith("millimetre")


def test_replay_reads_back_a_recording():
    import tempfile
    from c3_camera.c3_replay import Recording
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "rec"
        _, _, depth, _ = _synthetic_recording(out, n=8)
        r = Recording(out)
        assert len(r) == 8
        assert abs(r.mean_fps() - 20.0) < 0.5, r.mean_fps()
        colour, back = r.load(r.rows[0])
        assert colour is not None and colour.shape == (360, 640, 3)
        assert back is not None and back.dtype == np.uint16
        assert np.array_equal(back, depth)


def test_npy_depth_format_round_trips():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "rec"
        _, stats, depth, _ = _synthetic_recording(out, n=3, depth_format="npy")
        assert stats.written == 3
        assert np.array_equal(np.load(out / "depth/000001.npy"), depth)


def test_disk_estimate_is_in_the_right_ballpark():
    from c3_camera.recorder import estimate_disk_mb_per_min
    # 67 kB colour + 480x270 depth PNG at 20 fps -> ~150 MB/min
    mb = estimate_disk_mb_per_min(67.0, (480, 270), 20.0, "png")
    assert 100 < mb < 220, mb
    # npy is bigger than png for the same config
    assert estimate_disk_mb_per_min(67.0, (480, 270), 20.0, "npy") > mb


# =============================================================================
# dataset writer — TUM conformance and the bugs that were actually hit
# =============================================================================
def _fake_bundle(seq: int, t: float, size=(64, 48), mono=(80, 50)):
    """A bundle with distinct per-stream seqs, as the real source produces."""
    import cv2  # noqa: F401  (imported for parity with the real path)
    from c3_camera.source import Bundle, Frame

    w, h = size
    colour = np.full((h, w, 3), seq % 200, np.uint8)
    depth = np.full((h, w), 1000 + seq, np.uint16)
    depth[0, 0] = 0                     # an invalid pixel must survive as 0
    left = np.full((mono[1], mono[0]), 7, np.uint8)
    right = np.full((mono[1], mono[0]), 9, np.uint8)
    return Bundle(frames={
        "color": Frame("color", colour, seq, t, t + 0.03, 30.0, 8.0),
        "depth": Frame("depth", depth, seq, t + 0.0004, t + 0.04, 40.0),
        "left": Frame("left", left, seq, t, t + 0.03, 30.0),
        "right": Frame("right", right, seq, t, t + 0.03, 30.0),
    }, fresh=frozenset({"color", "depth", "left", "right"}))


def _write_dataset(tmp: Path, n: int = 5, repeats: int = 3):
    """Write n frames, submitting each `repeats` times as the live loop does."""
    from c3_camera.dataset import DatasetWriter

    w = DatasetWriter(tmp, mode="research", threads=2, queue_size=8, verbose=False,
                      write_mp4=False)
    t = 1785370000.0 - w._t_offset      # so t_unix lands on a realistic epoch
    for i in range(n):
        b = _fake_bundle(i + 100, t + i * 0.125)
        for _ in range(repeats):        # the same bundle arrives repeatedly
            w.submit(b)
    w.write_metadata({"depth_scale": 1000.0, "clock": {"offset": w._t_offset}})
    w.close()
    return w


def test_h264_writer_preserves_stream_and_lossless_depth():
    """H.264 bytes are concatenated untouched while depth remains uint16 PNG."""
    import csv
    import cv2
    import tempfile
    from c3_camera.dataset import DatasetWriter
    from c3_camera.source import Bundle, Frame

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "ds"
        w = DatasetWriter(out, mode="research", threads=1, queue_size=8,
                          write_mp4=False, color_encode="h264", verbose=False)
        depth = np.full((12, 16), 2345, np.uint16)
        packets = []
        for i, frame_type in enumerate(("P", "I", "P")):
            raw = f"packet-{i}".encode()
            c = Frame("color", None, 20 + i, 1.0 + i * 0.05,
                      1.01 + i * 0.05, 10.0, raw=raw,
                      encoding="h264", frame_type=frame_type)
            d = Frame("depth", depth, 20 + i, 1.0 + i * 0.05,
                      1.02 + i * 0.05, 20.0, encoding="depth16")
            w.submit(Bundle(frames={"color": c, "depth": d},
                            fresh=frozenset({"color", "depth"})))
            if frame_type != "P" or packets:
                packets.append(raw)
        w.close()

        # The leading P frame is deliberately omitted; recording starts on I.
        assert (out / "rgb.h264").read_bytes() == b"".join(packets)
        sidecar = list(csv.DictReader((out / "rgb_video.csv").open()))
        assert [r["frame_type"] for r in sidecar] == ["I", "P"]
        assert len(list((out / "rgb").glob("*.png"))) == 0
        depth_files = sorted((out / "depth").glob("*.png"))
        assert len(depth_files) == 2
        back = cv2.imread(str(depth_files[0]), cv2.IMREAD_UNCHANGED)
        assert back.dtype == np.uint16 and np.array_equal(back, depth)


def test_h264_extractor_restores_timestamp_named_rgb_images():
    """The post-capture decoder must preserve one sidecar timestamp per frame."""
    import csv
    import shutil
    import subprocess
    import tempfile
    import cv2
    from c3_camera.c3_extract_rgb_video import extract_rgb_video

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "ds"
        src = out / "source"
        src.mkdir(parents=True)
        rows = []
        for i in range(3):
            image = np.full((48, 64, 3), 30 + i * 60, np.uint8)
            cv2.imwrite(str(src / f"{i + 1:03d}.png"), image)
            stamp = f"178537000{i}.000000"
            rows.append({
                "frame_idx": i + 1, "t_unix": stamp,
                "t_monotonic": f"{100 + i * 0.05:.6f}",
                "rgb_seq": 100 + i, "rgb_file": f"rgb/{stamp}.png",
                "byte_offset": "", "byte_size": "",
                "frame_type": "I" if i == 0 else "P",
            })
        subprocess.run([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-framerate", "20", "-i", str(src / "%03d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-bf", "0", "-g", "3",
            "-f", "h264", str(out / "rgb.h264"),
        ], check=True)
        with (out / "rgb_video.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        summary = extract_rgb_video(out)
        assert summary["frames_decoded"] == 3
        for row in rows:
            image = cv2.imread(str(out / row["rgb_file"]))
            assert image is not None and image.shape[:2] == (48, 64)


def test_writer_ignores_repeated_bundles_for_the_same_frame():
    """The live loop sees ~3 bundles per frame because any stream arriving makes a
    new bundle. Writing all of them produced duplicate filenames that overwrote
    each other and a non-monotonic index — this is that regression."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "ds"
        w = _write_dataset(out, n=5, repeats=3)
        assert w.stats.written == 5, f"wrote {w.stats.written}, expected 5"
        for sub in ("rgb", "depth", "left", "right"):
            got = len(list((out / sub).glob("*.png")))
            assert got == 5, f"{sub}/ has {got} files, expected 5"


def test_tum_index_files_are_wellformed_and_monotonic():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "ds"
        _write_dataset(out, n=6)
        for name in ("rgb", "depth"):
            lines = (out / f"{name}.txt").read_text().splitlines()
            heads = [l for l in lines if l.startswith("#")]
            body = [l for l in lines if l and not l.startswith("#")]
            assert len(heads) == 3, f"{name}.txt should have 3 comment lines"
            assert len(body) == 6
            ts = [float(l.split()[0]) for l in body]
            assert all(b > a for a, b in zip(ts, ts[1:])), \
                f"{name}.txt timestamps must strictly increase"
            assert all(l.split()[1].startswith(f"{name}/") for l in body)


def test_associations_column_order_is_rgb_then_depth():
    """ORB-SLAM3's rgbd_tum reads rgb_ts rgb_file depth_ts depth_file, in that
    order. Reversing it silently feeds depth as if it were colour."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "ds"
        _write_dataset(out, n=4)
        rows = [l.split() for l in (out / "associations.txt").read_text().splitlines()
                if l.strip() and not l.startswith("#")]
        assert len(rows) == 4
        for r in rows:
            assert len(r) == 4, r
            float(r[0]); float(r[2])
            assert r[1].startswith("rgb/") and r[3].startswith("depth/"), r


def test_written_depth_is_uint16_millimetres_and_bit_exact():
    import cv2
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "ds"
        _write_dataset(out, n=3)
        p = sorted((out / "depth").glob("*.png"))[0]
        back = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        assert back.dtype == np.uint16, back.dtype
        assert back[0, 0] == 0, "invalid pixel must stay 0, not be filled in"
        assert back[1, 1] in (1100, 1101, 1102), back[1, 1]


def test_mono_streams_are_written_losslessly():
    import cv2
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "ds"
        _write_dataset(out, n=3)
        for sub, val in (("left", 7), ("right", 9)):
            p = sorted((out / sub).glob("*.png"))[0]
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            assert img is not None and img.dtype == np.uint8
            assert int(img.min()) == val and int(img.max()) == val, \
                f"{sub} pixels changed: {img.min()}..{img.max()} != {val}"


def test_depth_scale_is_1000_not_tums_5000():
    """A dataset written with 5000 would make every depth 5x too small, and look
    plausible while doing it."""
    from c3_camera.dataset import DEPTH_SCALE
    assert DEPTH_SCALE == 1000.0
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "ds"
        _write_dataset(out, n=2)
        txt = (out / "metadata.txt").read_text()
        assert "depth_scale: 1000.0" in txt
        assert "Do NOT use 5000" in txt
        assert "RGBD.DepthMapFactor: 1000.0" in txt
        assert "IMREAD_UNCHANGED" in txt


def test_research_mode_makes_rgb_and_depth_the_same_size():
    """RGB-D SLAM needs rgb[u,v] and depth[u,v] to be the same ray, which needs
    identical resolution as well as alignment."""
    from c3_camera.c3_collect import RESEARCH_DEFAULTS
    cfg = StreamConfig(**RESEARCH_DEFAULTS)
    rc = cfg.resolve()
    assert rc.color_out_size == rc.depth_out_size, \
        f"colour {rc.color_out_size} != depth {rc.depth_out_size}"
    assert cfg.depth_align == "color"
    assert cfg.pair_mode == "timestamp", "a dataset should pair, not take latest"


def test_review_and_research_modes_fit_the_link_budget():
    from c3_camera.c3_collect import RESEARCH_DEFAULTS, REVIEW_DEFAULTS
    from c3_camera.dataset import estimate_rate
    for label, d in (("research", RESEARCH_DEFAULTS), ("review", REVIEW_DEFAULTS)):
        cfg = StreamConfig(**d)
        rc = cfg.resolve()
        est = estimate_rate(rc.color_out_size, rc.depth_out_size, rc.mono_size,
                            cfg.fps, cfg.streams, cfg.color_wire, cfg.color_encode)
        assert est["wire_mbit_s"] <= POE_BUDGET_MBPS, \
            f"{label} default needs {est['wire_mbit_s']:.0f} Mbit/s"


def test_imu_continuity_uses_the_gyro_counter():
    """The accelerometer's sequence numbers skip on this BNO086 without any data
    loss; counting gaps on them invents drops."""
    from c3_camera.imu import ImuSample, ImuStats
    st = ImuStats()
    # gyro seq contiguous, accel seq skipping — as measured on the real device
    for i, acc_seq in enumerate([8, 9, 11, 12, 13, 15, 16]):
        st.update(ImuSample(t_device=i * 0.005, t_host=0.0, t_accel=i * 0.005,
                            t_gyro=i * 0.005, seq=i, seq_accel=acc_seq,
                            ax=0.0, ay=0.0, az=9.81))
    assert st.count == 7
    assert st.dropped == 0, f"reported {st.dropped} phantom drops"
    assert 190 < st.hz < 210, st.hz


def test_imu_extrinsics_absence_is_reported_as_data():
    """'the device has no IMU extrinsics' is a fact the dataset must carry, not an
    exception that stops a dive."""
    from c3_camera.imu import read_imu_extrinsics

    class Broken:
        def getImuToCameraExtrinsics(self, *_):
            raise RuntimeError("IMU calibration data is not available on device yet.")
        getCameraToImuExtrinsics = getImuToCameraExtrinsics

    out = read_imu_extrinsics(Broken(), socket=None)
    assert out["available"] is False
    assert "Kalibr" in out["note"]


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
