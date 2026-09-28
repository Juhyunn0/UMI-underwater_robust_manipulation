#!/usr/bin/env python3
"""
test_fstereo.py — the learned-depth option, checked without a GPU or a camera.

    QT_QPA_PLATFORM=offscreen \
      ~/miniforge3/envs/rovgui-pose/bin/python rov_gui/tests/test_fstereo.py

Everything here runs offscreen with no depthai device, no torch and no weights.
The network itself is NOT exercised — it needs 3 GiB of weights and CUDA. What
is pinned is the part that is easy to get wrong and impossible to see: that the
feature is genuinely off by default, that the link budget still closes with the
mono pair added, that a mismatched stereo pair is never matched, that the
exclusivity latch cannot leave the depth panel dark, and that a depth ratio is
never printed without saying which instrument produced it.
"""

from __future__ import annotations

import inspect
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from rov_gui.__main__ import build_parser


# --------------------------------------------------------------- opt-in
def test_learned_depth_is_off_unless_asked_for():
    """The flag, and everything that hangs off it, defaults to absent.

    Same contract as --pose and for a stronger reason: this one also changes
    the CAMERA configuration, so a default-on version would alter the link
    budget of every run anyone ever flew.
    """
    o = build_parser().parse_args([])
    assert o.fstereo is False
    assert o.panel2 == "rov"
    assert o.depth_fps == 20.0
    # The FS settings are ALSO dependent defaults now (--policy moves them
    # to the training store's, __main__.POLICY_DEPENDENT_DEFAULTS); without
    # --policy they resolve to the values every earlier run flew.
    assert o.fstereo_iters == 8 and o.fstereo_scale == 0.5 and o.fstereo_alpha == 0.0


def test_the_flag_never_takes_the_middle_panel():
    """--fstereo replaces the DEPTH panel's content and nothing else.

    It used to commandeer --panel2, which cost the pilot the ROV's own camera
    for the whole dive. The learned map belongs where the depth map already is:
    same colour bar, same cursor probe, same recording, so the comparison is a
    toggle rather than two views the eye has to reconcile.
    """
    o = build_parser().parse_args(["--fstereo"])
    assert o.panel2 == "rov"
    choices = [a.choices for a in build_parser()._actions if a.dest == "panel2"]
    assert choices and "fstereo" not in choices[0], choices


def test_the_layout_is_identical_with_and_without_the_flag():
    """The flag swaps a panel's CONTENT, never the screen's shape."""
    from rov_gui.window import panel_specs

    on = panel_specs(build_parser().parse_args(["--fstereo"]))
    off = panel_specs(build_parser().parse_args([]))
    assert on == off
    assert dict((k, t) for k, t, _ in on)["second"] == "Default RGB"


def test_the_flag_buys_link_back_for_the_mono_pair():
    """--fstereo adds two uncompressible streams, so it must pay for them.

    The pair is 2.05 Mbit/s per camera per fps at 640x400 and cannot be
    encoded. At the normal 20 fps the pair ALONE is 82 of the ~90 Mbit/s the
    C3's link carries, which is why the flag moves --depth-fps rather than
    hoping.
    """
    from c3_camera.config import POE_BUDGET_MBPS, StreamConfig

    from rov_gui.backends.hardware import _parse_ratio, _parse_size

    o = build_parser().parse_args(["--fstereo"])
    assert o.depth_fps == 15.0, o.depth_fps

    def cfg(streams, mono_fps):
        return StreamConfig(streams=streams, fps=o.fps,
                            isp_scale=_parse_ratio(o.isp_scale),
                            mjpeg_quality=o.mjpeg_quality, mono_fps=mono_fps,
                            depth_size=_parse_size(o.depth_size)).resolve()

    # What --fstereo actually asks for: colour + the mono pair, no device depth.
    shipped = cfg(("color", "left", "right"), o.depth_fps)
    assert shipped.bandwidth_mbps()["total"] < POE_BUDGET_MBPS, \
        shipped.budget_note()
    # Dropping the device depth stream is what pays for the rate: keeping it
    # at this mono rate does not fit.
    assert cfg(("color", "depth", "left", "right"),
               o.depth_fps).bandwidth_mbps()["total"] > POE_BUDGET_MBPS
    # ...and the rate is at the edge of what the link allows even without it.
    assert cfg(("color", "left", "right"),
               20.0).bandwidth_mbps()["total"] > POE_BUDGET_MBPS


def test_a_typed_depth_fps_beats_both_defaults():
    assert build_parser().parse_args(
        ["--fstereo", "--depth-fps", "12"]).depth_fps == 12.0


def test_the_parser_never_hands_out_the_none_sentinel():
    """--depth-fps defaults to None so --fstereo can move it.

    Downstream reads the namespace with getattr(opts, name, default), which
    does NOT fire for a key that exists and is None, so a leaked sentinel
    surfaces far from here as a format or arithmetic error. It leaked exactly
    once, into the link-budget guard.
    """
    for argv in ([], ["--fstereo"], ["--source", "hw"], ["--policy"]):
        o = build_parser().parse_args(argv)
        for name in ("depth_fps", "depth_size", "panel2", "fstereo_iters",
                     "fstereo_scale", "fstereo_alpha"):
            assert getattr(o, name) is not None, (argv, name)
    ns, _rest = build_parser().parse_known_args(["--fstereo", "--bogus"])
    assert ns.depth_fps == 15.0
    assert ns.fstereo_iters == 8 and ns.fstereo_scale == 0.5


def test_the_flag_says_so_when_it_can_do_nothing():
    """Only the hardware backend builds a learned-depth worker.

    On demo/ros2 the depth panel looks entirely normal and the FS button is not
    built, so without a word at launch the flag reads as "I asked for learned
    depth" and delivers silence — the same class of nothing-happens this
    feature already produced once.
    """
    from rov_gui.__main__ import check_fstereo

    assert check_fstereo(build_parser().parse_args([])) is None
    assert check_fstereo(build_parser().parse_args(
        ["--fstereo", "--source", "hw"])) is None
    warn = check_fstereo(build_parser().parse_args(["--fstereo"]))
    assert warn and "--source demo" in warn


# ------------------------------------------------------- stream ownership
def test_the_mono_pair_is_input_not_a_picture_unless_asked_for():
    """`left` is published as a picture only with --panel2 stereo.

    With --fstereo it is ALSO an input to the learned-depth worker, which is
    fine: the tap copies rather than consumes.
    """
    from rov_gui.backends.hardware import stream_to_panel

    fs = stream_to_panel(build_parser().parse_args(["--fstereo"]))
    assert "left" not in fs
    assert fs["depth"] == "depth" and fs["color"] == "main"
    assert stream_to_panel(
        build_parser().parse_args(["--panel2", "stereo"]))["left"] == "second"


def test_a_mismatched_stereo_pair_is_never_matched():
    """left and right cross XLink separately and are NOT in lockstep.

    Measured p50 host latency 87.0 ms (left) vs 69.5 ms (right) on this camera,
    so a bundle routinely holds one new frame and one retained one. Matching
    them does not fail — it returns a plausible, WRONG disparity field.
    """
    from rov_gui.backends.hardware import C3VideoWorker
    from rov_gui.bus import StereoMailbox

    w = C3VideoWorker.__new__(C3VideoWorker)      # no Qt, no device
    mb = StereoMailbox()
    mb.set_wanted(True)
    w.fstereo_mb = mb
    w._rig = object()
    w._fs_last_seq = None
    w._color_out_size = (640, 360)      # the grid the learned map is delivered on
    img = np.zeros((400, 640), np.uint8)

    def frame(seq):
        return types.SimpleNamespace(image=img, seq=seq, age_ms=lambda: 0.0)

    w._tap_fstereo(types.SimpleNamespace(
        frames={"left": frame(7), "right": frame(8)}))
    assert mb.take() is None

    bundle = types.SimpleNamespace(frames={"left": frame(9), "right": frame(9)})
    w._tap_fstereo(bundle)
    item = mb.take()
    assert item is not None and item["frame_seq"] == 9

    w._tap_fstereo(bundle)          # same exposure: not a new frame
    assert mb.take() is None


def test_disparity_to_millimetres_is_the_rectified_geometry():
    """K = fx_rect * baseline, and fx_rect may only come from stereoRectify's
    P1. The raw EEPROM fx differs by a fraction of a percent — invisible in a
    picture, fatal in a map — so the rig refuses to substitute it."""
    from c3_camera.host_depth import bench_rig, depth_from_disparity

    rig = bench_rig((640, 400), with_color=True, color_size=(640, 360))
    assert rig.fx_rect_source == "cv2.stereoRectify:P1[0,0]"
    assert rig.fx_rect != rig.provenance["fx_left_raw"]

    d = np.array([[24.0, 12.0, 0.0, -1.0]], dtype=np.float32)
    mm = depth_from_disparity(d, rig.k_mm_px, (200.0, 15000.0))
    assert mm.dtype == np.uint16
    assert abs(int(mm[0, 0]) - rig.k_mm_px / 24.0) <= 1
    assert abs(int(mm[0, 1]) - rig.k_mm_px / 12.0) <= 1
    # 0 disparity and the matcher's "no match" flag are both NO MEASUREMENT,
    # never a very large distance.
    assert mm[0, 2] == 0 and mm[0, 3] == 0


def test_the_rig_builds_the_same_way_under_opencv_4_and_5():
    """cv2.stereoRectify wants T as (3,1); 4.x tolerated a flat vector and 5.x
    aborts in gemm. The GUI env ships cv2 5 and `robust` ships 4, so a flat
    vector passes every test and fails on the vehicle."""
    from c3_camera.host_depth import bench_rig

    rig = bench_rig((640, 400), with_color=True, color_size=(640, 360))
    assert rig.rectifies and 10.0 <= rig.baseline_mm <= 500.0
    assert rig.provenance["T_left_to_right_mm"] == [-75.0, 0.0, 0.0]


# ------------------------------------------------------------- the readout
def test_a_depth_ratio_is_never_printed_without_its_source():
    """The comparison IS the experiment, so an unlabelled ratio is worse than
    none: it gets attributed to whichever depth the reader had in mind."""
    from rov_gui.widgets.trajectory import TrajectoryView

    v = TrajectoryView.__new__(TrajectoryView)
    v.depth_chk = v.depth_chk_fs = None

    TrajectoryView.set_depth_check(v, 1.42, 0.08, source="dev")
    assert v.depth_chk == (1.42, 0.08, 0.0) and v.depth_chk_fs is None
    TrajectoryView.set_depth_check(v, 1.03, 0.05, source="fs")
    assert v.depth_chk_fs == (1.03, 0.05, 0.0)
    TrajectoryView.set_depth_check(v, None, None, source="fs")
    assert v.depth_chk_fs is None and v.depth_chk == (1.42, 0.08, 0.0)


def test_a_held_ratio_is_marked_as_held():
    """Only one instrument can measure at a time — there is one depth map.

    Two labelled rows sitting adjacent would otherwise read as a simultaneous
    comparison, and the pair would get cited as one.
    """
    from rov_gui.widgets.trajectory import TrajectoryView

    v = TrajectoryView.__new__(TrajectoryView)
    v.depth_chk = v.depth_chk_fs = None
    TrajectoryView.set_depth_check(v, 1.42, 0.08, source="dev", held_s=37.0)
    TrajectoryView.set_depth_check(v, 1.03, 0.05, source="fs")

    # Call the WIDGET's own renderer, not a copy of its format string in the
    # test — a copy passes forever after the real one stops labelling anything.
    lines = _readout_lines(v)
    dev = [ln for ln in lines if "dev" in ln]
    fs = [ln for ln in lines if "FS" in ln]
    assert dev and fs, lines
    assert "held 37s" in dev[0], dev[0]          # the one not measuring now
    assert "held" not in fs[0], fs[0]            # the live one
    assert "1.42" in dev[0] and "1.03" in fs[0]


def _readout_lines(view):
    """The depth-vs-MAP lines the trajectory readout would actually draw.

    Runs the production loop out of the widget's own source so a test cannot
    drift from what the operator sees.
    """
    return list(type(view)._readout_depth_lines(view))


def test_the_device_row_keeps_its_label_when_the_fs_row_is_empty():
    """The state right after every swap into FoundationStereo.

    The FS accumulator is cleared on the swap, so a label keyed on "is the
    other row populated" printed the held DEVICE ratio bare — beside a panel
    captioned FOUNDATIONSTEREO. That is the attribution error the feature
    exists to avoid, drawn on the readout meant to prevent it.
    """
    from rov_gui.widgets.trajectory import TrajectoryView

    v = TrajectoryView.__new__(TrajectoryView)
    v.depth_chk = v.depth_chk_fs = None
    v._depth_two_source = False
    TrajectoryView.set_depth_check(v, 1.03, 0.05, source="fs")   # FS seen once
    TrajectoryView.set_depth_check(v, 1.62, 0.09, source="dev", held_s=3.0)
    TrajectoryView.set_depth_check(v, None, None, source="fs")   # ...then cleared

    lines = _readout_lines(v)
    assert len(lines) == 1, lines
    assert "dev" in lines[0], lines[0]          # still says which instrument
    assert "NOT metric" in lines[0], lines[0]   # ...and still flags 1.62x


def test_the_ratio_is_never_computed_twice_from_one_depth_map():
    """The measurement bug this design exists to make impossible.

    There is ONE depth map on screen. Selecting the input by panel key while
    calling the check once per source fed the same array to both accumulators
    and printed a fabricated agreement between two instruments — on the one
    cross-check nothing else on this station can make.
    """
    from rov_gui.window import MainWindow

    src = inspect.getsource(MainWindow._check_depth_scale)
    assert 'self.videos.get("depth")' in src   # input is the panel, always
    assert '"second"' not in src               # never the old side-by-side

    whole = inspect.getsource(MainWindow)
    calls = [ln for ln in whole.splitlines() if "_check_depth_scale(t" in ln]
    assert len(calls) == 1, calls              # exactly one call site...
    assert "_depth_instrument()" in calls[0], calls[0]
    # ...and the instrument is FIXED FOR THE RUN, not observed per frame: with
    # --fstereo the device stream is not requested at all, so there is no
    # moment at which a sample could be credited to the other sensor.
    inst = inspect.getsource(MainWindow._depth_instrument)
    assert "_fstereo_on" in inst


# ------------------------------------------------------------------ panels
def test_the_fs_chip_survives_a_drawn_frame():
    """Every explanation the worker produces has to outlive the first frame.

    `_note` renders only in the NO-SIGNAL branch, so once one learned frame had
    been shown, "no mono pair" and every inference error became invisible and
    the panel read as a frozen STALE rectangle with no reason — indistinguish-
    able from "never started", which is the confusion that started all this.
    """
    from rov_gui.state import FStereoState
    from rov_gui.widgets.video import VideoCanvas

    src = inspect.getsource(VideoCanvas.paintEvent)
    head, _, tail = src.partition('"NO SIGNAL"')
    assert "_paint_fs" in tail, "the FS chip must not live in the NO-SIGNAL arm"

    # ...and it explains every state rather than going blank in any of them.
    for kw in ({}, {"enabled": True, "loading": True}, {"enabled": True},
               {"enabled": True, "note": "no mono pair"}, {"error": "boom"},
               {"enabled": True, "live": True, "hz": 6.4}):
        assert FStereoState(**kw).chip.strip()


def test_the_meta_block_is_written_even_when_the_feature_is_off():
    """An absent key cannot tell "this build had no learned depth" from "it was
    switched off", and the schema version cannot either once both exist at 9."""
    from rov_gui.control.workers import MpcWorker

    # A stand-in `self`, not MpcWorker.__new__: that is a QObject, and reading
    # an attribute off one whose __init__ never ran raises rather than
    # answering. The method only touches self.opts and the hook.
    w = types.SimpleNamespace(opts=types.SimpleNamespace(fstereo=False))
    m = MpcWorker._fstereo_meta(w)
    assert m["enabled"] is False and m["why"] == "no --fstereo"

    w = types.SimpleNamespace(
        opts=types.SimpleNamespace(fstereo=True),
        fstereo_meta_fn=lambda: {"enabled": True, "model": "vitl @ 23-51-11",
                                 "depth_scale_applied": 1.0})
    m = MpcWorker._fstereo_meta(w)
    assert m["enabled"] is True
    # The device correction must NOT ride along: 0.64 was fitted to the
    # on-device block matcher's bias, and this path computes its own disparity.
    assert m["depth_scale_applied"] == 1.0


def test_the_device_depth_stream_is_not_even_requested():
    """One instrument per run, enforced at the pipeline.

    Not "preferred" — absent. Nothing can silently mix two depth sensors in a
    recording, a log or a ratio if only one of them exists in the process, and
    the bandwidth the device stream would have used is what buys the mono pair
    its frame rate.
    """
    from rov_gui.backends.hardware import C3VideoWorker

    src = inspect.getsource(C3VideoWorker.setup)
    assert 'if getattr(self.opts, "depth", True) and not self._fstereo_on:' in src
    # ...and the panel-status owner follows the same rule.
    panels = inspect.getsource(C3VideoWorker._panels)
    assert "not self._fstereo_on" in panels


def test_every_depth_consumer_takes_the_learned_map():
    """--fstereo is a depth SOURCE switch, not a display option.

    FoundationPose is the consumer that matters: it reads depth at the mask, so
    feeding it the camera's own map while the panel showed the learned one
    would make the pose and the picture disagree about the same object.
    """
    from rov_gui.backends.hardware import C3VideoWorker

    src = inspect.getsource(C3VideoWorker._tap_pose)
    assert "self._fs_depth_for(color)" in src
    assert 'bundle.frames.get("depth")' in src      # ...only on the other branch
    i_fs = src.index("_fs_depth_for")
    i_dev = src.index('bundle.frames.get("depth")')
    assert i_fs < i_dev, "the learned branch must come first, under the flag"


def test_a_stale_learned_map_is_refused_rather_than_paired():
    """The learned map lags its mono pair by an inference and arrives at ~13 Hz
    against colour's 30, so pairing is by capture time with a bound. Handing a
    tracker a map from half a second ago puts the object at a distance it no
    longer occupies — and FoundationPose has a path for "no depth", none for
    "confidently wrong depth"."""
    from rov_gui.backends.hardware import C3VideoWorker
    from rov_gui.bus import StereoMailbox
    from rov_gui.state import now

    w = C3VideoWorker.__new__(C3VideoWorker)
    w.fstereo_mb = StereoMailbox()
    w.FS_PAIR_MAX_MS = 250.0
    colour = types.SimpleNamespace(age_ms=lambda: 0.0)

    assert w._fs_depth_for(colour) == (None, float("inf")) or \
        w._fs_depth_for(colour)[0] is None          # nothing published yet

    fresh = np.zeros((250, 400), np.uint16)
    w.fstereo_mb.set_result(fresh, now())
    depth, skew = w._fs_depth_for(colour)
    assert depth is fresh and skew < 50.0

    w.fstereo_mb.set_result(fresh, now() - 1.0)     # a second old
    depth, skew = w._fs_depth_for(colour)
    assert depth is None and skew > 250.0


def test_a_failure_never_falls_back_to_the_other_sensor():
    """Stopping is the correct behaviour; substituting is not.

    --fstereo names the run's instrument. Quietly continuing on the camera's
    own depth after the learned path dies would put two sensors in one
    recording under one name — the exact attribution error the whole design is
    arranged to make impossible.
    """
    from rov_gui.backends.hardware import FStereoWorker
    from rov_gui.bus import DataBus, StereoMailbox

    logs = []
    w = types.SimpleNamespace(
        bus=DataBus(), mailbox=StereoMailbox(), mailboxes={}, enabled=True,
        _panel="depth", _fault="", _note="", _last_state_pub=0.0,
        _last_arrival=0.0, _hz=0.0, _frames=0, _last=None, IDLE_S=2.0,
        session=types.SimpleNamespace(error="", loading=False, ready=True,
                                      load_seconds=1.0))
    w.bus.log.connect(lambda lvl, m: logs.append((lvl, m)))
    w._publish_state = lambda note="": FStereoWorker._publish_state(w, note)
    w.mailbox.set_wanted(True)

    FStereoWorker._disarm(w, "checkpoint vanished")
    assert w.enabled is False
    assert w.mailbox.wanted() is False              # stops paying for it
    assert logs and logs[-1][0] == "error"
    # ...and it SAYS there is no other depth, instead of quietly finding one.
    assert "no other depth source" in logs[-1][1]
    assert "--fstereo" in logs[-1][1]


def test_the_flag_adds_no_control_to_the_gui():
    """No button, no toggle, nothing to press.

    The first hardware run failed because the feature waited behind a control
    the operator had no reason to look for. A launch flag that means "this run
    is about learned depth" needs no second confirmation.
    """
    from rov_gui.qt import QtWidgets
    from rov_gui.bus import FrameMailbox
    from rov_gui.widgets.video import VideoPanel
    from rov_gui.bus import DataBus

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = VideoPanel("depth", "C3 Depth", FrameMailbox("depth"))
    assert not hasattr(panel, "fs_btn")
    assert not hasattr(panel, "fstereo_toggled")
    assert not hasattr(DataBus, "cmd_fstereo_enable")
    del app


def main() -> int:
    """Run without pytest."""
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except Exception as e:                                   # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


# ------------------------------------------------------------ cuda graph
def test_the_cuda_graph_is_on_by_default_and_the_flag_turns_it_off():
    """Replay is the default because it is the same numbers for less time;
    the opt-out exists for a torch/driver where capture misbehaves."""
    o = build_parser().parse_args(["--fstereo"])
    assert o.fstereo_graph is True
    assert o.fstereo_size is None
    o = build_parser().parse_args(
        ["--fstereo", "--no-fstereo-graph", "--fstereo-size", "224x224"])
    assert o.fstereo_graph is False
    assert o.fstereo_size == "224x224"


def test_an_anisotropic_input_rescales_disparity_by_width_only():
    """224x224 from 640x400 squashes width by 640/224 and height by 400/224.

    Disparity is a horizontal length: the first ratio must be undone and the
    second must not touch it. The scale path is the same rule, isotropic.
    """
    from rov_gui.perception.fstereo import FStereoSession
    from rov_gui.qt import import_cv2
    cv2 = import_cv2()

    s = FStereoSession(size=(224, 224))
    s._cv2 = cv2
    assert s._infer_size(640, 400) == (224, 224)
    out = s._from_output(np.full((224, 224), 10.0, np.float32), 640, 400, 224, 224)
    assert out.shape == (400, 640)
    assert np.allclose(out, 10.0 * 640 / 224)

    s2 = FStereoSession(scale=0.5)
    s2._cv2 = cv2
    assert s2._infer_size(640, 400) == (320, 200)
    out = s2._from_output(np.full((200, 320), 4.0, np.float32), 640, 400, 320, 200)
    assert np.allclose(out, 8.0)

    same = np.full((224, 224), 1.0, np.float32)
    assert s2._from_output(same, 224, 224, 224, 224) is same

    # an explicit size wins over the scale, and nonsense sizes are refused
    assert FStereoSession(scale=0.5, size=(256, 160))._infer_size(640, 400) == (256, 160)
    try:
        FStereoSession(size=(32, 32))
    except ValueError:
        pass
    else:
        raise AssertionError("a 32x32 network input was accepted")


def test_a_failed_capture_falls_back_to_eager_once_and_stays_there():
    """Capture failing must not fail the frame, must not be retried on every
    frame (each attempt is three warm-up passes), and must be said out loud.
    """
    import contextlib
    from rov_gui.perception.fstereo import FStereoSession

    s = FStereoSession(graph=True)
    said = []
    s._on_log = lambda lvl, msg: said.append((lvl, msg))

    class FakeTorch:                      # what _eager and _capture touch
        float16 = "fp16"

        @staticmethod
        def no_grad():
            return contextlib.nullcontext()

        @staticmethod
        def autocast(*a, **k):
            return contextlib.nullcontext()
        # no .cuda -> _capture raises AttributeError, as a real failure would

    class FakeTensor:
        shape = (1, 3, 224, 320)

        def contiguous(self):
            return self

        def clone(self):
            return self

    calls = {"eager": 0, "capture": 0}

    def forward(l, r, iters, test_mode):
        calls["eager"] += 1
        return "disparity"

    s._torch = FakeTorch()
    s._model = types.SimpleNamespace(forward=forward)
    real_capture = s._capture

    def counted_capture(*a):
        calls["capture"] += 1
        return real_capture(*a)

    s._capture = counted_capture

    t = FakeTensor()
    assert s._forward(t, t) == "disparity"          # the frame still came out
    assert s.graph is False and s._graph is None
    assert "AttributeError" in s._graph_error
    assert any(lvl == "warn" and "eager" in msg for lvl, msg in said), said
    assert s._forward(t, t) == "disparity"
    assert calls["capture"] == 1, "capture was retried"
    assert calls["eager"] == 2

    d = s.describe()
    assert d["cuda_graph"] is False and d["cuda_graph_error"] == s._graph_error
    assert d["cuda_graph_input"] is None


def test_a_malformed_fstereo_size_is_refused_at_launch():
    """A typo must not quietly become --fstereo-scale. It used to: _parse_size
    returned None and the session took the scale path, with only the startup
    line's "scale=0.5" to say so."""
    import contextlib
    import io
    for bad in ("224*224", "224x", "224", "32x32", "abc", "224x224x1"):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                build_parser().parse_args(["--fstereo", "--fstereo-size", bad])
        except SystemExit:
            pass
        else:
            raise AssertionError(f"{bad!r} was accepted")
    o = build_parser().parse_args(["--fstereo", "--fstereo-size", "224X224"])
    assert o.fstereo_size == "224x224"


def test_a_captured_graph_is_replayed_and_a_failing_replay_falls_back():
    """The replay branch with a fake torch: capture on the first frame (three
    warm-ups + one captured pass), replay after that with the inputs copied
    into the static buffers, re-capture on a new shape, and a replay that
    raises drops the graph for the rest of the run — recomputing that frame
    eagerly rather than losing it."""
    import contextlib
    from rov_gui.perception.fstereo import FStereoSession

    class FakeStream:
        def wait_stream(self, other):
            pass

    class FakeGraph:
        replays = 0
        fail = False

        def replay(self):
            FakeGraph.replays += 1
            if FakeGraph.fail:
                raise RuntimeError("cudaErrorIllegalAddress")

    class FakeCuda:
        Stream = FakeStream
        CUDAGraph = FakeGraph

        @staticmethod
        def current_stream():
            return FakeStream()

        @staticmethod
        def stream(s):
            return contextlib.nullcontext()

        @staticmethod
        def graph(g, **kw):
            return contextlib.nullcontext()

        @staticmethod
        def synchronize():
            pass

        @staticmethod
        def empty_cache():
            pass

        @staticmethod
        def set_stream(s):
            pass

    class FakeTorch:
        float16 = "fp16"
        cuda = FakeCuda

        @staticmethod
        def no_grad():
            return contextlib.nullcontext()

        @staticmethod
        def autocast(*a, **k):
            return contextlib.nullcontext()

    class FakeTensor:
        shape = (1, 3, 224, 320)

        def __init__(self, tag):
            self.tag = tag
            self.copied = []

        def contiguous(self):
            return self

        def clone(self):
            return FakeTensor(self.tag + "-static")

        def copy_(self, other):
            self.copied.append(other.tag)

    class OtherShape(FakeTensor):
        shape = (1, 3, 160, 224)

    s = FStereoSession(graph=True)
    said = []
    s._on_log = lambda lvl, msg: said.append((lvl, msg))
    s._torch = FakeTorch()
    out = FakeTensor("out")
    eager = {"n": 0}

    def forward(l, r, iters, test_mode):
        eager["n"] += 1
        return out

    s._model = types.SimpleNamespace(forward=forward)
    t = FakeTensor("in")

    assert s._forward(t, t) is out
    assert s._graph is not None and s._graph_key == ((1, 3, 224, 320), s.iters)
    assert eager["n"] == 4                       # three warm-ups + the captured pass
    assert FakeGraph.replays == 1
    assert any("captured" in msg for _, msg in said), said
    assert s._forward(t, t) is out
    assert eager["n"] == 4 and FakeGraph.replays == 2   # replayed, nothing eager
    sl, sr, _ = s._static
    assert sl.copied == ["in", "in"] and sr.copied == ["in", "in"]
    d = s.describe()
    assert d["cuda_graph"] is True and d["cuda_graph_requested"] is True
    assert d["cuda_graph_input"] == [1, 3, 224, 320] and d["cuda_graph_error"] is None

    o = OtherShape("in2")
    assert s._forward(o, o) is out               # a new shape re-captures
    assert s._graph_key[0] == (1, 3, 160, 224) and eager["n"] == 8

    FakeGraph.fail = True
    assert s._forward(t, t) is out               # the frame still comes out
    assert s.graph is False and s._graph is None and s._graph_key is None
    assert s._graph_error.startswith("replay:")
    assert any("replay failed" in msg for lvl, msg in said if lvl == "warn"), said
    n = eager["n"]
    s._forward(t, t)
    assert eager["n"] == n + 1 and s.graph is False   # eager from here on, no re-capture
    d = s.describe()
    assert d["cuda_graph"] is False and d["cuda_graph_requested"] is True
    assert d["cuda_graph_input"] is None and d["cuda_graph_error"].startswith("replay:")


def test_disparity_composes_resize_forward_unpad_and_width_rescale():
    """disparity() end to end with a fake torch: the network sees the shrunk
    3-channel pair, and the result comes back at the rectified size scaled by
    w/w_in — the composition the unit tests of the pieces cannot pin."""
    import contextlib
    from rov_gui.perception.fstereo import FStereoSession
    from rov_gui.qt import import_cv2

    class T:
        def __init__(self, arr):
            self.arr = arr

        @property
        def shape(self):
            return self.arr.shape

        def cuda(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.arr

        def __getitem__(self, k):
            return T(self.arr[None]) if k is None else T(self.arr[k])

        def permute(self, *dims):
            return T(np.transpose(self.arr, dims))

    class FakeTorch:
        float16 = "fp16"

        @staticmethod
        def as_tensor(a):
            return T(np.asarray(a))

        @staticmethod
        def no_grad():
            return contextlib.nullcontext()

        @staticmethod
        def autocast(*a, **k):
            return contextlib.nullcontext()

    seen = {}

    class Padder:
        def __init__(self, shape, divis_by=32, force_square=False):
            seen["shape"] = tuple(shape)

        def pad(self, *xs):
            return list(xs)

        def unpad(self, x):
            return x

    def forward(l, r, iters, test_mode):
        return T(np.full((1, 1) + tuple(l.shape[-2:]), 7.0, np.float32))

    s = FStereoSession(size=(224, 224), graph=False)
    s._torch, s._cv2, s._padder = FakeTorch(), import_cv2(), Padder
    s._model = types.SimpleNamespace(forward=forward)
    L = np.zeros((400, 640), np.uint8)
    d = s.disparity(L, L.copy())
    assert seen["shape"] == (1, 3, 224, 224)
    assert d.shape == (400, 640)
    assert np.allclose(d, 7.0 * 640 / 224)
    assert s._last_in_size == (224, 224)
    assert s.describe()["infer_size_actual"] == [224, 224]
    assert s.describe()["scale_applied"] is False

    big = FStereoSession(size=(800, 400), graph=False)
    big._torch, big._cv2, big._padder, big._model = s._torch, s._cv2, Padder, s._model
    try:
        big.disparity(L, L.copy())
    except Exception as e:                                       # noqa: BLE001
        assert "larger" in str(e), e
    else:
        raise AssertionError("an enlarging --fstereo-size was accepted")


def test_the_stereo_import_does_not_keep_foundationposes_utils_name():
    """--fstereo --pose share one process and BOTH checkouts import a
    top-level `Utils` bare. Two fake checkouts stand in for them: after the
    stereo import, the bare name must belong to the pose side again, a fresh
    bare import must find the pose side's file (the stereo checkout was moved
    to the end of sys.path), and the stereo modules must still hold the names
    they bound at import."""
    import importlib
    import sys
    import tempfile
    import textwrap
    from rov_gui.perception import fstereo

    tmp = Path(tempfile.mkdtemp(prefix="fstereo_utils_"))
    fs, fp = tmp / "stereo", tmp / "pose"
    (fs / "core" / "utils").mkdir(parents=True)
    (fs / "Utils.py").write_text("SIDE = 'stereo'\n"
                                 "def get_resize_keep_aspect_ratio(): return 'stereo'\n"
                                 "def freeze_model(m): return m\n")
    (fs / "core" / "__init__.py").write_text("")
    (fs / "core" / "extractor.py").write_text(
        "from Utils import get_resize_keep_aspect_ratio, freeze_model\n")
    (fs / "core" / "foundation_stereo.py").write_text(textwrap.dedent("""
        from core.extractor import get_resize_keep_aspect_ratio
        class FoundationStereo:
            side = 'stereo'
        """))
    (fs / "core" / "utils" / "__init__.py").write_text("")
    (fs / "core" / "utils" / "utils.py").write_text("class InputPadder: pass\n")
    fp.mkdir()
    (fp / "Utils.py").write_text("SIDE = 'pose'\ndef depth2xyzmap(): return 'pose'\n")

    saved = {k: sys.modules.get(k) for k in ("Utils", "core", "core.extractor",
                                             "core.foundation_stereo",
                                             "core.utils", "core.utils.utils")}
    saved_path = list(sys.path)
    try:
        for k in saved:
            sys.modules.pop(k, None)
        # the pose side loaded first, the way sam2_live does it: front of the path
        sys.path.insert(0, str(fp))
        pose_utils = importlib.import_module("Utils")
        assert pose_utils.SIDE == "pose"

        FS, Padder = fstereo._import_upstream(str(fs))
        assert FS.side == "stereo" and Padder.__name__ == "InputPadder"
        assert sys.modules["Utils"] is pose_utils          # handed back
        assert sys.modules["core.extractor"].get_resize_keep_aspect_ratio() == "stereo"
        assert sys.modules[fstereo.FSTEREO_UTILS_ALIAS
                           if hasattr(fstereo, "FSTEREO_UTILS_ALIAS")
                           else "fstereo_upstream.Utils"].SIDE == "stereo"
        assert sys.path.index(str(fs)) > sys.path.index(str(fp))   # demoted

        # the other order: nobody had the name, the pose side imports later
        sys.modules.pop("Utils")
        for k in ("core", "core.extractor", "core.foundation_stereo",
                  "core.utils", "core.utils.utils"):
            sys.modules.pop(k, None)
        sys.path.remove(str(fp))
        FS, _ = fstereo._import_upstream(str(fs))
        assert "Utils" not in sys.modules                  # not left behind
        sys.path.insert(0, str(fp))                        # sam2_live's insert
        assert importlib.import_module("Utils").SIDE == "pose"
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        sys.modules.pop("fstereo_upstream.Utils", None)
        sys.path[:] = saved_path


def test_a_capture_that_dies_inside_the_body_names_the_real_error_and_restores_the_stream():
    """torch.cuda.graph.__exit__ raises 'operation failed due to a previous
    error during capture' from capture_end() AFTER the real error, with no
    try/finally around the stream context. The session must log the innermost
    cause and put the thread back on the stream it was on."""
    import contextlib
    from rov_gui.perception.fstereo import FStereoSession

    calls = {"set_stream": []}

    class FakeStream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, other):
            pass

    class DyingGraphCtx:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            if a[0] is None:
                try:
                    raise RuntimeError("Cannot copy between CPU and CUDA tensors during CUDA graph capture")
                except RuntimeError as inner:
                    raise RuntimeError("operation failed due to a previous error during capture") from inner
            return False

    class FakeCuda:
        Stream = lambda: FakeStream("side")            # noqa: E731
        CUDAGraph = lambda: types.SimpleNamespace(replay=lambda: None)   # noqa: E731

        @staticmethod
        def current_stream():
            return FakeStream("default")

        @staticmethod
        def stream(s):
            return contextlib.nullcontext()

        @staticmethod
        def graph(g, **kw):
            return DyingGraphCtx()

        @staticmethod
        def synchronize():
            pass

        @staticmethod
        def empty_cache():
            pass

        @staticmethod
        def set_stream(s):
            calls["set_stream"].append(s.name)

    class FakeTorch:
        float16 = "fp16"
        cuda = FakeCuda

        @staticmethod
        def no_grad():
            return contextlib.nullcontext()

        @staticmethod
        def autocast(*a, **k):
            return contextlib.nullcontext()

    class FakeTensor:
        shape = (1, 3, 224, 320)

        def contiguous(self):
            return self

        def clone(self):
            return self

        def copy_(self, other):
            pass

    s = FStereoSession(graph=True)
    said = []
    s._on_log = lambda lvl, msg: said.append((lvl, msg))
    s._torch = FakeTorch()
    s._model = types.SimpleNamespace(forward=lambda l, r, iters, test_mode: "disp")
    t = FakeTensor()
    assert s._forward(t, t) == "disp"                       # eager fallback for that frame
    assert s.graph is False and s._graph is None
    assert "Cannot copy between CPU and CUDA" in s._graph_error, s._graph_error
    assert "surfaced as RuntimeError" in s._graph_error
    assert calls["set_stream"] == ["default"], calls
    assert any("capture failed" in msg for lvl, msg in said if lvl == "warn"), said


def test_the_gap_fill_never_touches_a_measured_pixel_and_never_averages():
    """The projection's black web is closed with a NEIGHBOUR'S millimetres.

    Three properties, because each one is a different way to get this wrong:
    filling could overwrite a measurement, it could average two depths that
    straddle an edge into a distance at which nothing exists (the objection
    resize_depth_nearest is built around), and it could pull a foreground
    surface outward over a disocclusion that belongs to the background.
    """
    from c3_camera.host_depth import fill_scatter_gaps

    # a slanted surface with the 1-px scatter gaps the warp actually leaves
    d = np.zeros((9, 9), np.uint16)
    d[:] = 1000
    d[:, 3] = 0
    d[4, :] = 0
    out = fill_scatter_gaps(d, 2)
    assert (out[d != 0] == d[d != 0]).all(), "a measured pixel was overwritten"
    assert (out == 1000).all(), out
    # every value present must be a value that was present before: no averaging
    assert set(np.unique(out)).issubset(set(np.unique(d)) | {0})

    # a hole between two DIFFERENT surfaces takes the far one (the background),
    # never something in between
    e = np.zeros((5, 7), np.uint16)
    e[:, :3] = 500          # near
    e[:, 4:] = 4000         # far
    filled = fill_scatter_gaps(e, 1)
    assert filled[:, 3].tolist() == [4000] * 5, filled[:, 3]
    assert 500 not in set(np.unique(filled[:, 3]))
    assert not (500 < filled[:, 3]).all() or (filled[:, 3] == 4000).all()

    # iters=0 is the identity, and it is what every pre-existing caller gets
    same = fill_scatter_gaps(e, 0)
    assert same is e
    assert fill_scatter_gaps(np.zeros((4, 4), np.uint16), 3).max() == 0

    # the contract is uint16 millimetres, like every other depth in this repo
    try:
        fill_scatter_gaps(np.zeros((4, 4), np.float32), 1)
    except TypeError:
        pass
    else:
        raise AssertionError("a float depth map was accepted")


def test_filled_pixels_are_reported_apart_from_measured_ones():
    """valid_out must stay MEASURED coverage; the fill gets its own number.

    Folding them together would turn a resampling repair into a measurement on
    the one line the pilot reads to judge the depth — and this repo has a
    standing rule against exactly that.
    """
    from rov_gui.state import FStereoState

    st = FStereoState(live=True, enabled=True, hz=15.0, solve_ms=47.0,
                      valid_native=97.3, valid_out=95.3, filled_out=4.7,
                      frames=9)
    assert "95%+5" in st.chip, st.chip
    assert st.valid_out == 95.3 and st.filled_out == 4.7

    off = FStereoState(live=True, enabled=True, hz=15.0, solve_ms=47.0,
                       valid_native=97.3, valid_out=95.3, filled_out=0.0,
                       frames=9)
    assert "+" not in off.chip.split("valid")[1], off.chip

    o = build_parser().parse_args(["--fstereo"])
    assert o.fstereo_fill == 2
    assert build_parser().parse_args(["--fstereo", "--fstereo-fill", "0"]).fstereo_fill == 0

    # infer() must compute valid_out BEFORE filling, or the two collapse
    import inspect
    from rov_gui.perception import fstereo as mod
    src = inspect.getsource(mod.FStereoSession.infer)
    i_measured = src.index("measured = ")
    i_fill = src.index("fill_scatter_gaps(")
    assert i_measured < i_fill, "valid_out is computed after the fill"
    assert '"valid_out": 100.0 * measured' in src


if __name__ == "__main__":
    raise SystemExit(main())
