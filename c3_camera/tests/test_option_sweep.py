#!/usr/bin/env python3
"""
test_option_sweep.py — everything in c3_option_sweep.py that runs without a camera.

    ~/.venvs/c3-depthai/bin/python -m pytest c3_camera/tests/test_option_sweep.py -v
    ~/.venvs/c3-depthai/bin/python c3_camera/tests/test_option_sweep.py   # no pytest

The sweep's expensive half needs hardware, but the half that decides *what gets
recorded* — axis parsing, the fold that drops indistinguishable cells, the
StreamConfig each cell builds, the slug that names its directory, and the choice
between the RGB-D and monocular SLAM configs — is pure and is exactly where a
silent bug would cost a whole afternoon of camera time. Also covered: the
Annex-B keyframe detection added to source.py, because without it an H.264 or
H.265 cell records zero frames and says nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from c3_camera import c3_option_sweep as S
from c3_camera.config import (STREAM_COLOR, STREAM_DEPTH, STREAM_LEFT,
                              STREAM_RIGHT)
from c3_camera.source import _annexb_frame_type


def _args(**over):
    """A parsed namespace, so tests exercise the real defaults and parsers."""
    argv = []
    for k, v in over.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return S.parse_args(argv)


# =============================================================================
# axis parsing
# =============================================================================
def test_onoff_accepts_the_spellings_the_help_text_promises():
    assert S._parse_onoff("on") == [True]
    assert S._parse_onoff("off") == [False]
    assert S._parse_onoff("on,off") == [True, False]
    assert S._parse_onoff("off,on") == [False, True]
    # Duplicates collapse: "on,on" is one cell, not two identical recordings.
    assert S._parse_onoff("on,on") == [True]


def test_onoff_rejects_anything_it_would_otherwise_have_to_guess():
    for bad in ("maybe", "1,2", "", "onoff"):
        try:
            S._parse_onoff(bad)
        except Exception:
            continue
        raise AssertionError(f"--depth {bad!r} was accepted")


def test_depth_size_keeps_policies_and_sizes_apart():
    assert S._parse_depth_sizes("match-color,derived,320x180") == \
        ["match-color", "derived", "320x180"]
    assert S._parse_depth_sizes("match") == ["match-color"]
    assert S._parse_depth_sizes("auto") == ["derived"]
    for bad in ("640,360", "640", "axb"):
        try:
            S._parse_depth_sizes(bad)
        except Exception:
            continue
        raise AssertionError(f"--depth-size {bad!r} was accepted")


def test_isp_scale_none_is_a_value_not_an_absence():
    assert S._parse_isp_list("1/2,none,2/3") == [(1, 2), None, (2, 3)]


# =============================================================================
# the fold
# =============================================================================
def test_axes_an_encoder_cannot_see_do_not_multiply_the_cells():
    """1 raw + 2 mjpeg qualities + 2 h265 bitrates = 5 runs, not 12."""
    a = _args(color_encode="none,mjpeg,h265", mjpeg_quality="80,90",
              video_bitrate_kbps="4000,8000", depth="on",
              depth_size="match-color", isp_scale="1/2", fps="20")
    combos = S.build_combos(a)
    assert len(combos) == 5
    labels = [c.encode_label for c in combos]
    assert sorted(labels) == ["h265@4000k", "h265@8000k", "mjpeg-q80",
                              "mjpeg-q90", "raw"]


def test_depth_size_does_not_multiply_cells_that_ship_no_depth():
    """With depth off the size changes nothing on the wire or on disk."""
    a = _args(depth="off", depth_size="match-color,derived,320x180",
              color_encode="mjpeg", isp_scale="1/2", fps="20")
    assert len(S.build_combos(a)) == 1


def test_depth_off_and_on_are_both_kept_when_both_are_asked_for():
    a = _args(depth="on,off", depth_size="match-color,derived",
              color_encode="mjpeg", isp_scale="1/2", fps="20")
    combos = S.build_combos(a)
    # 2 depth-on grids + 1 depth-off = 3.
    assert len(combos) == 3
    assert sum(1 for c in combos if not c.depth_on) == 1


def test_every_cell_gets_its_own_directory_name():
    a = _args(color_res="1080p,4k", isp_scale="1/2,1/4",
              color_encode="none,mjpeg,h265", mjpeg_quality="80,90",
              video_bitrate_kbps="4000,8000", depth="on,off",
              depth_size="match-color,derived,320x180", mono="on,off",
              fps="10,20")
    combos = S.build_combos(a)
    slugs = [c.slug(i) for i, c in enumerate(combos, 1)]
    assert len(set(slugs)) == len(slugs), "two cells would share a directory"
    # The leading index is what guarantees it; the rest must still be readable.
    assert all(s[:2].isdigit() for s in slugs)


# =============================================================================
# the StreamConfig each cell builds
# =============================================================================
def test_match_color_asks_for_the_colour_grid_and_nothing_else():
    a = _args(depth="on", depth_size="match-color", isp_scale="1/2",
              color_encode="mjpeg", fps="20")
    cfg = S.build_combos(a)[0].cfg
    assert cfg.depth_match_color is True
    assert cfg.depth_size is None          # the two must never both be set
    rc = cfg.resolve()
    assert rc.depth_out_size == rc.color_out_size


def test_an_explicit_depth_size_never_collides_with_match_color():
    """resolve() prefers depth_size, so the two must not both be requested."""
    a = _args(depth="on", depth_size="320x180", isp_scale="1/2",
              color_encode="mjpeg", fps="20")
    cfg = S.build_combos(a)[0].cfg
    assert cfg.depth_size == (320, 180)
    assert cfg.depth_match_color is False
    assert cfg.resolve().depth_out_size == (320, 180)


def test_derived_leaves_config_to_size_depth_itself():
    a = _args(depth="on", depth_size="derived", isp_scale="1/2",
              color_encode="mjpeg", fps="20")
    cfg = S.build_combos(a)[0].cfg
    assert cfg.depth_size is None and cfg.depth_match_color is False
    rc = cfg.resolve()
    assert rc.depth_out_size != rc.color_out_size   # mono-width default


def test_streams_follow_the_depth_and_mono_axes():
    on = S.build_combos(_args(depth="on", mono="on", depth_size="match-color",
                              color_encode="mjpeg", isp_scale="1/2",
                              fps="20"))[0]
    assert set(on.cfg.streams) == {STREAM_COLOR, STREAM_DEPTH,
                                   STREAM_LEFT, STREAM_RIGHT}
    off = S.build_combos(_args(depth="off", mono="off", color_encode="mjpeg",
                               isp_scale="1/2", fps="20"))[0]
    assert tuple(off.cfg.streams) == (STREAM_COLOR,)


def test_every_cell_records_a_dataset_pairing_not_a_live_view_pairing():
    """"latest" never waits, which is right for a HUD and wrong for a dataset."""
    a = _args(depth="on,off", depth_size="match-color,derived",
              color_encode="none,mjpeg", isp_scale="1/4,1/2", fps="20")
    assert all(c.cfg.pair_mode == "timestamp" for c in S.build_combos(a))


def test_the_camera_imu_is_on_by_default_because_a_slam_dataset_wants_it():
    c = S.build_combos(_args())[0]
    assert c.cfg.imu_enable is True
    assert c.cfg.imu_batch_threshold == 10      # completeness over latency
    assert S.build_combos(S.parse_args(["--no-camera-imu"]))[0].cfg.imu_enable \
        is False


def test_a_rejected_combination_raises_where_the_sweep_can_catch_it():
    """resolve() must fail per cell, not abort a sweep holding the camera."""
    a = _args(color_encode="mjpeg", mono_encode="mjpeg", mono="on",
              depth="on", depth_size="match-color", isp_scale="1/2", fps="20")
    # colour + two mono encoders = 3 contexts: allowed, but at the documented
    # ceiling, so it must warn rather than raise.
    rc = S.build_combos(a)[0].cfg.resolve()
    assert any("VideoEncoder" in w for w in rc.warnings)


# =============================================================================
# which SLAM config ships
# =============================================================================
class _FakeWriter:
    def __init__(self):
        self.written = []

    def write_orbslam3_yaml(self, intr, baseline_mm, fps):
        self.written.append("rgbd")

    def write_orbslam3_mono_yaml(self, intr, fps):
        self.written.append("mono")


class _FakeSource:
    def __init__(self, intr):
        self.intrinsics = {STREAM_COLOR: intr} if intr is not None else {}
        self.device = None


def _slam_for(**over):
    from c3_camera.geometry import Intrinsics
    combo = S.build_combos(_args(**over))[0]
    w, src = _FakeWriter(), _FakeSource(
        Intrinsics(fx=1, fy=1, cx=1, cy=1, width=8, height=8))
    name, reason = S._slam_config(w, src, combo, combo.cfg.resolve(),
                                  verbose=False)
    return w.written, name, reason


def test_depth_on_the_colour_grid_gets_the_rgbd_config():
    written, name, reason = _slam_for(depth="on", depth_size="match-color",
                                      isp_scale="1/2", color_encode="mjpeg",
                                      fps="20")
    assert written == ["rgbd"] and name == "orbslam3_rgbd.yaml" and reason == ""


def test_depth_off_gets_the_monocular_config_not_a_lying_rgbd_one():
    written, name, reason = _slam_for(depth="off", isp_scale="1/2",
                                      color_encode="mjpeg", fps="20")
    assert written == ["mono"] and name == "orbslam3_mono.yaml"
    assert "no depth stream" in reason


def test_depth_off_the_colour_grid_gets_the_monocular_config_and_says_why():
    """An RGB-D yaml here would put colour intrinsics on a 320x180 depth map."""
    written, name, reason = _slam_for(depth="on", depth_size="320x180",
                                      isp_scale="1/2", color_encode="mjpeg",
                                      fps="20")
    assert written == ["mono"] and name == "orbslam3_mono.yaml"
    assert "320x180" in reason and "960x540" in reason


def test_no_intrinsics_means_no_config_rather_than_a_crash():
    combo = S.build_combos(_args())[0]
    w = _FakeWriter()
    name, reason = S._slam_config(w, _FakeSource(None), combo,
                                  combo.cfg.resolve(), verbose=False)
    assert w.written == [] and name == "" and "intrinsics" in reason


# =============================================================================
# Annex-B keyframe detection (source.py)
# =============================================================================
def test_keyframe_detection_reads_the_nal_type_not_the_missing_api():
    """depthai 2.32 has no ImgFrame.getFrameType().

    Without this, frame_type is "" forever, DatasetWriter's "start at the first
    I frame" gate never opens, and an H.264/H.265 cell writes ZERO frames while
    reporting no error at all.
    """
    assert _annexb_frame_type(b"\x00\x00\x00\x01\x65idr", "h264") == "I"
    assert _annexb_frame_type(b"\x00\x00\x00\x01\x41slice", "h264") == "P"
    assert _annexb_frame_type(b"\x00\x00\x00\x01\x26idr", "h265") == "I"
    assert _annexb_frame_type(b"\x00\x00\x00\x01\x02trail", "h265") == "P"
    # Unparseable input must not be trusted to start a decodable stream.
    assert _annexb_frame_type(b"", "h264") == "P"
    assert _annexb_frame_type(b"\xde\xad\xbe\xef", "h264") == "P"


def test_a_keyframe_is_found_behind_its_parameter_sets():
    """The encoder prepends SPS/PPS (or VPS/SPS/PPS) to the IDR access unit."""
    h264 = (b"\x00\x00\x00\x01\x67sps" b"\x00\x00\x00\x01\x68pps"
            b"\x00\x00\x00\x01\x65idr")
    assert _annexb_frame_type(h264, "h264") == "I"
    h265 = (b"\x00\x00\x00\x01\x40vps" b"\x00\x00\x00\x01\x42sps"
            b"\x00\x00\x00\x01\x44pps" b"\x00\x00\x00\x01\x28idr")
    assert _annexb_frame_type(h265, "h265") == "I"


def test_three_byte_start_codes_are_read_too():
    assert _annexb_frame_type(b"\x00\x00\x01\x65idr", "h264") == "I"
    assert _annexb_frame_type(b"\x00\x00\x01\x41p", "h264") == "P"


def test_parameter_sets_alone_are_not_a_keyframe():
    """A packet of VPS/SPS/PPS decodes to no image; it cannot open the timeline."""
    assert _annexb_frame_type(b"\x00\x00\x00\x01\x67sps", "h264") == "P"
    assert _annexb_frame_type(b"\x00\x00\x00\x01\x40vps", "h265") == "P"


def test_the_h265_irap_range_is_the_whole_class_not_just_idr():
    """16-23 = BLA/IDR/CRA, any of which is a valid random-access point."""
    for nal in range(16, 24):
        pkt = b"\x00\x00\x00\x01" + bytes([nal << 1]) + b"x"
        assert _annexb_frame_type(pkt, "h265") == "I", nal
    for nal in (0, 1, 9, 15, 32, 33, 34):
        pkt = b"\x00\x00\x00\x01" + bytes([nal << 1]) + b"x"
        assert _annexb_frame_type(pkt, "h265") == "P", nal


# =============================================================================
# index counting
# =============================================================================
def test_tum_index_counting_ignores_the_comment_header():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "rgb.txt"
        p.write_text("# color images\n# timestamp filename\n"
                     "1.0 rgb/1.0.png\n2.0 rgb/2.0.png\n\n")
        assert S._count_list(p) == 2
        assert S._count_list(Path(td) / "missing.txt") == 0


def test_index_lines_and_images_on_disk_are_counted_separately():
    """They diverge exactly when it matters: a failed H.26x extraction."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "rgb").mkdir()
        (d / "rgb.txt").write_text("# h\n1.0 rgb/1.0.png\n2.0 rgb/2.0.png\n")
        (d / "rgb" / "1.0.png").write_bytes(b"x")
        assert S._count_list(d / "rgb.txt") == 2      # what the index promises
        assert S._count_files(d / "rgb") == 1         # what SLAM would find
        assert S._count_files(d / "nope") == 0


# =============================================================================
# end to end, against a fake source — no camera
# =============================================================================
class _FakeC3Source:
    """Enough of C3Source to drive run_one()'s whole dataset path."""

    def __init__(self, cfg, verbose=True, strict=False, keep_raw=False):
        import numpy as np

        from c3_camera.geometry import Intrinsics
        from c3_camera.imu import ImuStats
        from c3_camera.metrics import Metrics

        self._np = np
        self.cfg = cfg
        self.rc = cfg.resolve()
        self.metrics = Metrics(cfg.streams)
        self.imu_stats = ImuStats()
        self.device = None
        self.device_report = {"product": "FAKE", "mxid": "0", "bootloader": "-"}
        self.imu_extrinsics = {}
        cw, ch = self.rc.color_out_size
        self.intrinsics = {STREAM_COLOR: Intrinsics(
            fx=cw * 0.7, fy=cw * 0.7, cx=cw / 2, cy=ch / 2,
            width=cw, height=ch, distortion=(0.1, -0.02, 0.0, 0.0, 0.0))}
        self.formats = {s: "nv12" for s in cfg.streams}
        self._seq = 0

    def __enter__(self):
        return self

    def __exit__(self, *e):
        return None

    def read(self, timeout_ms=1000.0):
        import time

        from c3_camera.source import Bundle, Frame
        self._seq += 1
        t = time.monotonic()
        cw, ch = self.rc.color_out_size
        dw, dh = self.rc.depth_out_size
        frames = {STREAM_COLOR: Frame(
            name="color",
            image=self._np.zeros((ch, cw, 3), self._np.uint8),
            seq=self._seq, t_device=t, t_recv=t, latency_ms=12.0,
            encoding="nv12")}
        if STREAM_DEPTH in self.cfg.streams:
            frames[STREAM_DEPTH] = Frame(
                name="depth",
                image=self._np.full((dh, dw), 1500, self._np.uint16),
                seq=self._seq, t_device=t, t_recv=t, latency_ms=15.0,
                encoding="depth16")
        time.sleep(0.02)
        return Bundle(frames=frames, fresh=frozenset(frames),
                      intrinsics=self.intrinsics)

    def drain_encoded_rgb(self):
        return []

    def drain_imu(self):
        return []


def _record(tmp: Path, name: str, extra: list[str]):
    """Run one cell against the fake source and return (row, dir, verdict)."""
    import contextlib
    import io

    real = S.C3Source
    S.C3Source = _FakeC3Source
    try:
        a = S.parse_args(["--duration", "1.5", "--warmup", "0.3",
                          "--isp-scale", "1/4", "--color-encode", "none",
                          "--fps", "20", "--mavlink-transport", "none",
                          "--no-mp4"] + extra)
        combo = S.build_combos(a)[0]
        out = tmp / name
        with contextlib.redirect_stdout(io.StringIO()):
            row = S.run_one(combo, out, a, None, 1, 1)
            verdict = (S.check_dataset(out, combo, 2) if row["frames_written"]
                       else "-")
        return row, out, verdict, a, combo
    finally:
        S.C3Source = real


REFERENCE_ARTEFACTS = ("rgb", "depth", "left", "right", "rgb.txt", "depth.txt",
                       "associations.txt", "frames.csv", "calibration.json",
                       "metadata.json", "metadata.txt")


def test_a_depth_on_cell_writes_the_whole_reference_dataset_layout():
    """The promise of this tool: every cell is a dataset, not a pile of images."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        row, out, verdict, _, _ = _record(
            Path(td), "rgbd", ["--depth", "on", "--depth-size", "match-color"])
        assert row["ok"] == 1, row["error"]
        missing = [f for f in REFERENCE_ARTEFACTS if not (out / f).exists()]
        assert not missing, missing
        assert (out / "orbslam3_rgbd.yaml").exists()
        assert row["rgb_index"] == row["rgb_on_disk"] == row["assoc_lines"]
        assert verdict == "pass", verdict


def test_a_depth_off_cell_is_still_validated_rather_than_waved_through():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        row, out, verdict, _, _ = _record(Path(td), "mono", ["--depth", "off"])
        assert row["ok"] == 1, row["error"]
        assert (out / "orbslam3_mono.yaml").exists()
        assert not (out / "orbslam3_rgbd.yaml").exists()
        assert row["depth_on_disk"] == 0 and row["assoc_lines"] == 0
        # The RGB-D checker's two depth-specific failures are expected here and
        # are subtracted; everything else it checks still has to pass.
        assert verdict == "pass-mono", verdict


def test_metadata_never_claims_pixel_alignment_a_cell_did_not_deliver():
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        _, on_grid, _, _, _ = _record(
            Path(td), "rgbd", ["--depth", "on", "--depth-size", "match-color"])
        _, off_grid, v, _, _ = _record(
            Path(td), "offgrid", ["--depth", "on", "--depth-size", "320x180"])
        _, nodepth, _, _, _ = _record(Path(td), "mono", ["--depth", "off"])

        assert "pixel-for-pixel" in (on_grid / "metadata.txt").read_text()
        off_txt = (off_grid / "metadata.txt").read_text()
        assert "pixel-for-pixel" not in off_txt
        assert "NOT the same ray" in off_txt
        assert v == "problems-offgrid", v
        no_txt = (nodepth / "metadata.txt").read_text()
        assert "NOT RECORDED" in no_txt and "pixel-for-pixel" not in no_txt
        # depth_scale is what c3_dataset_check rejects a dataset for lacking.
        for d in (on_grid, off_grid, nodepth):
            assert json.loads((d / "metadata.json").read_text())["depth_scale"] \
                == 1000.0


def test_recording_into_a_used_directory_is_refused_not_merged():
    """Two sessions in one directory orphan images and inflate the size column."""
    import contextlib
    import io
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        row, out, _, a, combo = _record(
            Path(td), "cell", ["--depth", "on", "--depth-size", "match-color"])
        assert row["ok"] == 1
        real = S.C3Source
        S.C3Source = _FakeC3Source
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                again = S.run_one(combo, out, a, None, 1, 1)
        finally:
            S.C3Source = real
        assert again["ok"] == 0
        assert "already holds a recording" in again["error"]
        # ... and the failed cell must not be credited with the existing data.
        assert again["dataset_mb"] == "" and again["rgb_index"] == ""


def test_the_generated_rgbd_yaml_has_the_keys_orbslam3_actually_reads():
    """Camera.fps must be an int node, and the RGB-D parser needs Stereo.*."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        _, out, _, _, _ = _record(
            Path(td), "rgbd", ["--depth", "on", "--depth-size", "match-color",
                               "--fps", "20"])
        y = (out / "orbslam3_rgbd.yaml").read_text()
        assert "Camera.fps: 20\n" in y
        assert "Stereo.ThDepth:" in y
        assert "Stereo.b:" in y          # required even without a device read
        assert "RGBD.DepthMapFactor: 1000.0" in y


def test_a_non_integral_fps_still_writes_an_integer_camera_fps():
    """7.5 fps would emit `Camera.fps: 7.5`, which ORB-SLAM3 exits(-1) on."""
    from c3_camera.dataset import DatasetWriter
    from c3_camera.geometry import Intrinsics
    import tempfile

    intr = Intrinsics(fx=1, fy=1, cx=1, cy=1, width=8, height=8)
    with tempfile.TemporaryDirectory() as td:
        w = DatasetWriter(Path(td), mode="research", threads=1, verbose=False)
        try:
            w.write_orbslam3_yaml(intr, 75.0, 7.5)
            w.write_orbslam3_mono_yaml(intr, 7.5)
            assert "Camera.fps: 8\n" in (Path(td) / "orbslam3_rgbd.yaml").read_text()
            assert "Camera.fps: 8\n" in (Path(td) / "orbslam3_mono.yaml").read_text()
        finally:
            w.close()


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
