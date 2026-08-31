#!/usr/bin/env python3
"""
test_gantry_map_pose_panel.py — the "Tag map position" readout, offscreen.

    QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/robust/bin/python src/tests/test_gantry_map_pose_panel.py

Covers the wiring rather than the geometry (test_gantry_map_pose.py owns the
geometry): that GantryPanel actually builds the readout and feeds it frames,
that each display state is distinguishable, and that a setup error surfaces its
REASON instead of an indefinite "no fix".

The last one is the point of the resolution gate. The panel's camera resolution
is a combo box and the fisheye calibration is only valid at one size; picking a
different one makes K wrong by the ratio, which does not look like a failure —
every position is just quietly scaled. So the estimator refuses, and the refusal
has to reach the operator's eye.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

_SRC = Path(__file__).resolve().parents[1]
_REPO = _SRC.parent
for p in (str(_SRC), str(_REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import os                                                        # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication                         # noqa: E402

_APP = QApplication.instance() or QApplication([])

import gantry_panel as gp                                        # noqa: E402
from gantry_map_pose import MapPose                              # noqa: E402

CALIB = _REPO / "config" / "fisheye_calibration.yaml"
TAG_MAP = _REPO / "config" / "tag_map.yaml"


def _pump(seconds: float = 0.35) -> None:
    """Let the worker thread's queued signals land."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        _APP.processEvents()
        time.sleep(0.01)
    _APP.processEvents()


def _panel():
    return gp.MapPosePanel()


def test_starts_blank_rather_than_zero():
    """0.0 would be a lie: the map origin is a real place on the pool floor."""
    mp = _panel()
    try:
        assert "—" in mp.pos_label.text()
        assert "0.000" not in mp.pos_label.text()
        assert "camera off" in mp.status_label.text()
    finally:
        mp.shutdown()


def test_a_fix_renders_signed_metres_and_its_quality():
    mp = _panel()
    try:
        mp._on_pose_ready(
            MapPose(0.4123, -1.0834, -0.8341, 11, (4, 5, 25), 0.94, False, ""), "")
        pos = mp.pos_label.text()
        assert "+0.412" in pos and "-1.083" in pos and "-0.834" in pos, pos
        det = mp.detail_label.text()
        assert "tags 11" in det and "0.94 px" in det, det
        assert "locked" in mp.status_label.text()
        assert mp.latest() is not None
    finally:
        mp.shutdown()


def test_ambiguous_is_called_out_not_shown_as_locked():
    """A single-tag IPPE flip puts the camera on the wrong side of the tag
    plane. The number still renders; it must not render as trustworthy."""
    mp = _panel()
    try:
        mp._on_pose_ready(MapPose(0.4, -1.0, -0.83, 1, (25,), 1.8, True, ""), "")
        assert "AMBIGUOUS" in mp.status_label.text()
        assert "locked" not in mp.status_label.text()
    finally:
        mp.shutdown()


def test_no_fix_keeps_the_last_position_but_says_so():
    mp = _panel()
    try:
        mp._on_pose_ready(MapPose(0.4, -1.0, -0.83, 9, (4, 5), 0.9, False, ""), "")
        held = mp.pos_label.text()
        for _ in range(6):
            mp._on_pose_ready(None, "no tags in view")
        assert mp.pos_label.text() == held, "a dropout must not blank the position"
        assert "no tags in view" in mp.status_label.text()
    finally:
        mp.shutdown()


def test_wrong_resolution_reaches_the_operator():
    """The recorded frames are 960x540; the calibration is 1280x720. The gate
    must fire AND the reason must land in the status label."""
    import cv2
    import glob

    frames = sorted(glob.glob(str(
        _REPO / "data" / "20260528" / "20260528_215858_recording" / "frames" / "*.jpg")))
    if not frames:
        print("    SKIP (no recorded frames — data/ is gitignored)")
        return
    img = cv2.imread(frames[0])

    mp = _panel()
    try:
        mp.configure(CALIB, TAG_MAP, 0.170)
        _pump(0.2)
        mp._last_forward_t = 0.0
        mp.on_frame(img, time.monotonic())
        _pump()
        st = mp.status_label.text()
        assert "CalibrationMismatch" in st, st
        assert "960x540" in st and "1280x720" in st, st

        # And the latch must clear on the operator's fix: change the camera
        # resolution and reconnect, which re-configures with the SAME calib.
        mp.configure(CALIB, TAG_MAP, 0.170)
        _pump(0.2)
        assert mp._worker._fatal == "", "latched error survived a reconfigure"
    finally:
        mp.shutdown()


def test_disconnect_blanks_the_readout():
    mp = _panel()
    try:
        mp._on_pose_ready(MapPose(0.4, -1.0, -0.83, 9, (4, 5), 0.9, False, ""), "")
        mp.set_state("disconnected")
        assert "—" in mp.pos_label.text()
        assert "camera off" in mp.status_label.text()
    finally:
        mp.shutdown()


def test_gantry_panel_builds_and_feeds_it():
    """The structural wiring: the panel exists, and connecting the camera
    subscribes it to frame_ready alongside the preview."""
    w = gp.GantryPanel(controller=gp.MockFMC4030Controller(), is_mock=True)
    try:
        mp = getattr(w, "map_pose_panel", None)
        assert mp is not None, "GantryPanel did not build the readout"
        assert isinstance(mp, gp.MapPosePanel)
        # It must not fight the layout: the left pane is inside a QScrollArea
        # with a 440 px minimum, and this sits under the splitter.
        assert mp.minimumSizeHint().height() < 200, mp.minimumSizeHint().height()

        w._is_mock_camera = True
        w._connect_camera()
        _pump(0.2)
        assert w._camera is not None
        recv = w._camera.receivers(w._camera.frame_ready)
        assert recv >= 2, f"frame_ready has {recv} receivers, expected preview + readout"
    finally:
        try:
            w.close()
        except Exception:
            pass


def main() -> int:
    tests = [
        test_starts_blank_rather_than_zero,
        test_a_fix_renders_signed_metres_and_its_quality,
        test_ambiguous_is_called_out_not_shown_as_locked,
        test_no_fix_keeps_the_last_position_but_says_so,
        test_wrong_resolution_reaches_the_operator,
        test_disconnect_blanks_the_readout,
        test_gantry_panel_builds_and_feeds_it,
    ]
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


if __name__ == "__main__":
    raise SystemExit(main())
