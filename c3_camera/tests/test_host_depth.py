#!/usr/bin/env python3
"""
test_host_depth.py — the host stereo matcher's maths, on synthetic data only.

    ~/.venvs/c3-depthai/bin/python c3_camera/tests/test_host_depth.py     # no pytest
    ~/.venvs/c3-depthai/bin/python -m pytest c3_camera/tests/test_host_depth.py -v

No camera, no dataset fixture, no recorded frame. Every input here is built from
a number this file already knows the answer to, because that is the only way to
test a depth pipeline: a real wall has no ground truth better than a tape
measure, and a recorded frame proves only that the code still does what it did
last week.

What each group defends:

  * a known disparity must come back as the known distance
                                     -> test_synthetic_plane_*
  * d == 0 is "no measurement", never inf/NaN/65535
                                     -> test_zero_disparity_*, test_negative_*
  * uint16 millimetres, 0 = invalid — the same contract dataset.py enforces
                                     -> test_output_is_uint16_*, test_depth_scale_*
  * NEAREST resampling never invents a value
                                     -> test_nearest_*, test_warp_*
  * the rectified focal length is never substituted by the raw EEPROM one
                                     -> test_*refuses*
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from c3_camera.host_depth import (
    DEPTH_SCALE,
    INVALID_MM,
    MAX_UINT16_MM,
    MONO_INPUT_RAW,
    MONO_INPUT_RECTIFIED,
    AsyncHostStereo,
    CalibrationIncomplete,
    ColorRig,
    HostDepthError,
    HostStereoMatcher,
    MatcherParams,
    StereoRig,
    apply_mm_lut,
    depth_from_disparity,
    disparity_mm_lut,
    resize_depth_nearest,
    synthetic_pair,
    warp_depth_to_color,
)

# A rig with round numbers, so every expected value in this file can be checked
# by hand: K = 380 * 75 = 28500 mm*px, so d = 24 px is exactly 1187.5 mm.
RIG = StereoRig.synthetic((640, 400), fx_rect=380.0, baseline_mm=75.0)
K = RIG.k_mm_px


# =============================================================================
# the guard: d == 0 must be 0
# =============================================================================
def test_zero_disparity_is_invalid_not_infinite():
    """+0.0, -0.0 and 0/0 all have to land on the invalid code."""
    d = np.array([[0.0, -0.0, 24.0]], dtype=np.float32)
    z = depth_from_disparity(d, K)
    assert z.dtype == np.uint16
    assert z[0, 0] == INVALID_MM, f"+0.0 disparity gave {z[0, 0]}"
    assert z[0, 1] == INVALID_MM, f"-0.0 disparity gave {z[0, 1]}"
    assert z[0, 2] == 1188, z[0, 2]  # 28500/24 = 1187.5, rounds to nearest


def test_negative_disparity_is_invalid_and_not_a_plausible_distance():
    """SGBM writes (min_disparity - 1) * 16 into unmatched pixels — i.e. -16 in
    fixed point, -1.0 px. Naively dividing and casting turns that into 37036 mm:
    an in-range, confident-looking 37 m reading. This is the exact failure the
    guard exists to prevent, so it is pinned as a value, not as a shrug."""
    naive = np.array([K / np.float32(-1.0)], np.float32).astype(np.uint16)[0]
    assert naive == 37036, ("the naive cast changed; the guard's comment cites "
                            f"37036 and this build gives {naive}")
    z = depth_from_disparity(np.array([[-1.0]], np.float32), K)
    assert z[0, 0] == INVALID_MM
    # And the same value arriving as SGBM's own fixed-point int16.
    fixed = np.array([[-16]], np.int16)
    assert apply_mm_lut(fixed, disparity_mm_lut(K))[0, 0] == INVALID_MM


def test_nan_and_inf_disparity_are_invalid():
    d = np.array([[np.nan, np.inf, -np.inf]], dtype=np.float32)
    z = depth_from_disparity(d, K)
    assert (z == INVALID_MM).all(), z


def test_no_output_is_ever_inf_or_nan_over_the_whole_int16_range():
    """Sweep every disparity SGBM can physically emit, not a sample of them."""
    d = np.arange(-32768, 32768, dtype=np.int32).astype(np.float32) / 16.0
    z = depth_from_disparity(d, K)
    assert z.dtype == np.uint16
    assert np.isfinite(z.astype(np.float64)).all()
    assert (z[d <= 0] == INVALID_MM).all()


def test_lut_entry_zero_is_the_invalid_code():
    """The LUT is built by the guarded function, so entry 0 inherits the guard."""
    lut = disparity_mm_lut(K)
    assert lut.dtype == np.uint16
    assert lut.size == 1 << 15
    assert lut[0] == INVALID_MM
    assert lut[24 * 16] == 1188            # d = 24 px
    assert lut[95 * 16] == round(K / 95)   # d = 95 px, the device's default max


def test_lut_and_direct_division_agree_everywhere():
    lut = disparity_mm_lut(K)
    d_px = np.arange(1 << 15, dtype=np.float32) / 16.0
    direct = depth_from_disparity(d_px, K)
    assert np.array_equal(lut, direct)


def test_apply_lut_folds_the_matchers_invalid_flag_onto_zero():
    disp = np.array([[-16, 0, 24 * 16, 32767]], dtype=np.int16)
    out = apply_mm_lut(disp, disparity_mm_lut(K))
    assert out.dtype == np.uint16
    assert out[0, 0] == INVALID_MM
    assert out[0, 1] == INVALID_MM
    assert out[0, 2] == 1188


def test_apply_lut_refuses_a_float_disparity():
    """A float image here means somebody already divided by 16, and indexing
    with it would silently take the floor."""
    try:
        apply_mm_lut(np.zeros((4, 4), np.float32), disparity_mm_lut(K))
    except TypeError:
        return
    raise AssertionError("accepted a float disparity image")


# =============================================================================
# the uint16 / millimetre contract
# =============================================================================
def test_depth_scale_matches_the_dataset_writer():
    """host_depth must not drift from the module that enforces the contract."""
    from c3_camera import dataset
    assert DEPTH_SCALE == dataset.DEPTH_SCALE == 1000.0


def test_output_survives_the_writers_dtype_check():
    """dataset.py:400-404 raises on anything that is not uint16; run that exact
    check against real matcher output rather than trusting the annotation."""
    m = HostStereoMatcher(RIG, MONO_INPUT_RECTIFIED)
    left, right = synthetic_pair((640, 400), 24, seed=1)
    depth = m.compute(left, right)
    if depth.dtype != np.uint16:  # the writer's literal condition
        raise AssertionError(f"depth must be uint16 millimetres, got {depth.dtype}")
    assert depth.shape == (400, 640)


def test_output_round_trips_through_a_16_bit_png():
    """The dataset stores these as 16-bit PNGs; the values must come back
    unchanged, including the 0s."""
    m = HostStereoMatcher(RIG, MONO_INPUT_RECTIFIED)
    depth = m.compute(*synthetic_pair((640, 400), 24, seed=2))
    with tempfile.TemporaryDirectory() as td:
        p = str(Path(td) / "d.png")
        assert cv2.imwrite(p, depth, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        back = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    assert back.dtype == np.uint16
    assert np.array_equal(back, depth)


def test_values_beyond_uint16_are_dropped_not_saturated():
    """A distance that cannot be represented is not a measurement. Clamping it
    to 65535 would report the ceiling as if it had been measured there."""
    tiny = np.array([[0.1]], dtype=np.float32)     # K/0.1 = 285000 mm
    assert depth_from_disparity(tiny, K)[0, 0] == INVALID_MM
    edge = np.array([[K / MAX_UINT16_MM]], dtype=np.float32)
    assert depth_from_disparity(edge, K)[0, 0] == MAX_UINT16_MM


def test_z_range_drops_rather_than_clamps():
    d = np.array([[95.0, 24.0, 2.0]], dtype=np.float32)   # 300, 1187.5, 14250 mm
    z = depth_from_disparity(d, K, z_range_mm=(500.0, 5000.0))
    assert z[0, 0] == INVALID_MM, "too near should be dropped, not clamped to 500"
    assert z[0, 1] == 1188
    assert z[0, 2] == INVALID_MM, "too far should be dropped, not clamped to 5000"


# =============================================================================
# a known disparity must reconstruct the known distance
# =============================================================================
def _plane_stats(disparity_px: int, size=(640, 400), nd=96, seed=3):
    """Match a synthetic plane and return (recovered mm, truth mm, coverage)."""
    rig = StereoRig.synthetic(size, fx_rect=380.0, baseline_mm=75.0)
    m = HostStereoMatcher(rig, MONO_INPUT_RECTIFIED,
                          MatcherParams(num_disparities=nd))
    left, right = synthetic_pair(size, disparity_px, seed=seed)
    depth = m.compute(left, right)
    b = MatcherParams().block_size
    # Trim the two regions where a correct matcher still has nothing to say:
    # the leftmost `nd` columns (no search window fits) and the block border.
    core = depth[b:-b, nd + b:-b]
    valid = core[core > 0]
    truth = rig.k_mm_px / disparity_px
    return valid, truth, valid.size / core.size


def test_synthetic_plane_reconstructs_the_known_depth():
    """d = 24 px at K = 28500 is 1187.5 mm. Recover it to the millimetre."""
    valid, truth, cover = _plane_stats(24)
    assert cover > 0.95, f"only {cover:.1%} of the core matched"
    assert abs(float(np.median(valid)) - truth) <= 1.0, \
        f"median {np.median(valid)} vs truth {truth}"
    assert abs(float(valid.mean()) - truth) <= 2.0
    # Worst pixel: 1/16 px of disparity error at d=24 is 3.1 mm, so allow a
    # couple of quantisation steps and nothing more.
    assert float(np.abs(valid.astype(np.float64) - truth).max()) <= 10.0


def test_synthetic_plane_reconstructs_at_several_distances():
    """One distance can be a coincidence; a sweep across the range cannot.
    Tolerance scales with distance because a fixed disparity error does: at
    K = 28500, one pixel is 3.2 mm at 300 mm and 151 mm at 2 m."""
    for d_px, tol_frac in ((95, 0.01), (48, 0.01), (24, 0.01), (12, 0.02), (6, 0.03)):
        valid, truth, cover = _plane_stats(d_px)
        assert cover > 0.9, f"d={d_px}: coverage {cover:.1%}"
        err = abs(float(np.median(valid)) - truth)
        assert err <= tol_frac * truth, \
            f"d={d_px}: median {np.median(valid):.1f} vs truth {truth:.1f} " \
            f"({err / truth:.2%} > {tol_frac:.0%})"


def test_disparity_image_is_the_disparity_that_was_built_in():
    """Check the intermediate too, so a compensating error in K cannot hide a
    matching error."""
    m = HostStereoMatcher(RIG, MONO_INPUT_RECTIFIED)
    disp = m.compute_disparity(*synthetic_pair((640, 400), 24, seed=4))
    core = disp[8:-8, 104:-8]
    v = core[core > 0]
    assert abs(float(np.median(v)) - 24.0) <= 1.0 / 16.0
    assert float(np.abs(v - 24.0).max()) <= 0.25


def test_min_z_is_k_over_max_disparity():
    """MinZ = K / (min_disparity + num_disparities - 1). nd=96 searches d in
    [0, 95], which is exactly the device's default max disparity, so both give
    28500/95 = 300 mm."""
    m96 = HostStereoMatcher(RIG, MONO_INPUT_RECTIFIED, MatcherParams(num_disparities=96))
    assert abs(m96.min_z_mm - K / 95.0) < 1e-9
    assert abs(m96.min_z_mm - 300.0) < 1e-9
    m192 = HostStereoMatcher(RIG, MONO_INPUT_RECTIFIED, MatcherParams(num_disparities=192))
    assert abs(m192.min_z_mm - K / 191.0) < 1e-9
    assert m192.min_z_mm < 150.0        # device extended is 190 -> 150.0 mm


def test_nothing_nearer_than_min_z_is_ever_reported():
    m = HostStereoMatcher(RIG, MONO_INPUT_RECTIFIED, MatcherParams(num_disparities=96))
    depth = m.compute(*synthetic_pair((640, 400), 95, seed=5))
    v = depth[depth > 0]
    assert v.size > 0
    # Subpixel refinement can go a fraction past the last integer level; one
    # quantisation step of slack, not a free pass.
    assert v.min() >= m.min_z_mm - 5.0, f"{v.min()} < MinZ {m.min_z_mm}"


# =============================================================================
# NEAREST resampling invents nothing
# =============================================================================
def test_nearest_resampling_invents_no_values():
    """Downscale and upscale a depth map with holes: every output value must
    already exist in the input. A bilinear resize fails this immediately."""
    rng = np.random.default_rng(11)
    src = rng.integers(300, 5000, (400, 640)).astype(np.uint16)
    src[rng.random((400, 640)) < 0.3] = 0          # holes, the dangerous case
    allowed = set(np.unique(src).tolist())
    for size in ((480, 270), (320, 200), (640, 400), (200, 120)):
        out = resize_depth_nearest(src, size)
        assert out.dtype == np.uint16
        assert (out.shape[1], out.shape[0]) == size
        new = set(np.unique(out).tolist()) - allowed
        assert not new, f"{size} invented {sorted(new)[:5]}"


def test_bilinear_would_invent_values_and_bridge_holes():
    """The negative control. Without it, the test above could pass for a trivial
    reason (e.g. every value already present)."""
    src = np.zeros((8, 8), np.uint16)
    src[:, 4:] = 1000                        # a clean depth edge next to invalid
    allowed = set(np.unique(src).tolist())
    lin = cv2.resize(src, (16, 16), interpolation=cv2.INTER_LINEAR)
    assert set(np.unique(lin).tolist()) - allowed, \
        "bilinear failed to invent a value; this control is not controlling"
    near = resize_depth_nearest(src, (16, 16))
    assert not set(np.unique(near).tolist()) - allowed


def test_matcher_downscale_path_upsamples_with_nearest():
    """The performance path must not smuggle interpolation back in."""
    m = HostStereoMatcher(RIG, MONO_INPUT_RECTIFIED, downscale=2, upscale_output=True)
    depth = m.compute(*synthetic_pair((640, 400), 24, seed=6))
    assert depth.shape == (400, 640)
    small = HostStereoMatcher(RIG, MONO_INPUT_RECTIFIED, downscale=2,
                              upscale_output=False)
    depth_small = small.compute(*synthetic_pair((640, 400), 24, seed=6))
    assert depth_small.shape == (200, 320)
    assert not set(np.unique(depth).tolist()) - set(np.unique(depth_small).tolist())


def test_downscale_keeps_the_distance_it_reports():
    """Halving the images halves fx AND the disparity, so K/d is unchanged. If
    only one of the two were scaled, every distance would be out by 2x."""
    truth = K / 24.0
    m = HostStereoMatcher(RIG, MONO_INPUT_RECTIFIED, downscale=2)
    depth = m.compute(*synthetic_pair((640, 400), 24, seed=7))
    v = depth[100:-20, 120:-20]
    v = v[v > 0]
    assert v.size > 0
    assert abs(float(np.median(v)) - truth) <= 0.03 * truth, \
        f"downscaled median {np.median(v):.1f} vs {truth:.1f}"


# =============================================================================
# colour alignment
# =============================================================================
def _color_rig(mono=(640, 400), color=(480, 270), tx_mm=37.0) -> StereoRig:
    w, h = color
    K_c = np.array([[385.0 * w / 480.0, 0.0, (w - 1) / 2.0],
                    [0.0, 385.0 * h / 270.0, (h - 1) / 2.0],
                    [0.0, 0.0, 1.0]])
    col = ColorRig(K=K_c, dist=np.zeros(14), size=(w, h),
                   R=np.eye(3), t_mm=np.array([tx_mm, 0.0, 0.0]))
    return StereoRig.synthetic(mono, fx_rect=380.0, baseline_mm=75.0, color=col)


def test_warp_invents_no_depth_values():
    """Forward scatter with NEAREST placement: every warped value came from the
    source map, plus the 0 fill."""
    rig = _color_rig()
    rng = np.random.default_rng(13)
    src = rng.integers(400, 4000, (400, 640)).astype(np.uint16)
    src[rng.random((400, 640)) < 0.4] = 0
    out = warp_depth_to_color(src, rig)
    assert out.dtype == np.uint16
    assert out.shape == (270, 480)
    new = set(np.unique(out).tolist()) - set(np.unique(src).tolist()) - {0}
    assert not new, f"warp invented {sorted(new)[:5]}"


def test_warp_z_buffer_keeps_the_near_surface():
    """Two surfaces that project to the same colour pixel: the near one wins.
    Without the z-buffer the answer depends on scatter order, and a background
    reading can overwrite the object in front of it."""
    rig = _color_rig(tx_mm=0.0)   # colour co-located, so the mapping is 1:many
    src = np.zeros((400, 640), np.uint16)
    # A 2x2 block of mono pixels lands on one colour pixel at 480x270 scale.
    src[200:202, 320:322] = 5000
    src[200, 320] = 800           # one near sample among far ones
    out = warp_depth_to_color(src, rig)
    hit = out[out > 0]
    assert hit.size > 0
    assert hit.min() == 800
    # The near sample must actually be the one written where they collide.
    assert 800 in set(np.unique(out).tolist())


def test_warp_places_a_point_where_the_geometry_says():
    """One pixel, hand-computed. Colour is offset +37 mm in x from rectified
    left, so a point at 1000 mm on the left optical axis lands 37 mm to the left
    of the colour axis: u = cx_c + fx_c * (-37/1000)."""
    rig = _color_rig(tx_mm=37.0)
    src = np.zeros((400, 640), np.uint16)
    u0, v0 = int(round(rig.cx_rect)), int(round(rig.cy_rect))
    src[v0, u0] = 1000
    out = warp_depth_to_color(src, rig)
    vs, us = np.nonzero(out)
    assert vs.size == 1, f"expected one point, got {vs.size}"
    col = rig.color
    x_left = (u0 - rig.cx_rect) * 1000.0 / rig.fx_rect
    u_exp = col.K[0, 0] * ((x_left + 37.0) / 1000.0) + col.K[0, 2]
    v_exp = col.K[1, 1] * (((v0 - rig.cy_rect) * 1000.0 / rig.fy_rect) / 1000.0) \
        + col.K[1, 2]
    assert abs(us[0] - u_exp) <= 0.5, f"u {us[0]} vs {u_exp:.2f}"
    assert abs(vs[0] - v_exp) <= 0.5, f"v {vs[0]} vs {v_exp:.2f}"
    assert out[vs[0], us[0]] == 1000


def test_warp_refuses_without_a_colour_rig():
    try:
        warp_depth_to_color(np.zeros((400, 640), np.uint16), RIG)
    except HostDepthError:
        return
    raise AssertionError("warped without colour calibration")


def test_warp_applies_the_colour_distortion():
    """Skipping distortion changed 21.8% of the grid in a direct measurement, so
    the code must actually use the coefficients it is given."""
    rng = np.random.default_rng(17)
    src = rng.integers(500, 3000, (400, 640)).astype(np.uint16)
    base = _color_rig()
    distorted = StereoRig(**{**base.__dict__,
                             "color": ColorRig(K=base.color.K,
                                               dist=np.array([-0.3, 0.1, 0.0, 0.0,
                                                              0.0]),
                                               size=base.color.size,
                                               R=base.color.R,
                                               t_mm=base.color.t_mm)})
    a = warp_depth_to_color(src, base)
    b = warp_depth_to_color(src, distorted)
    assert not np.array_equal(a, b), "distortion coefficients were ignored"


# =============================================================================
# refusals
# =============================================================================
def test_rig_refuses_an_illegal_focal_length_provenance():
    """The invariant that stops any future constructor from smuggling the raw
    EEPROM fx in as if it were rectified."""
    try:
        StereoRig(mono_size=(640, 400), fx_rect=378.56, fy_rect=378.75,
                  cx_rect=320.0, cy_rect=200.0, baseline_mm=75.0,
                  fx_rect_source="eeprom", baseline_source="spec")
    except CalibrationIncomplete as e:
        assert "0.4%" in str(e)
        return
    raise AssertionError("accepted a raw-EEPROM focal length")


def test_rig_refuses_a_centimetre_baseline():
    """7.5 instead of 75 makes everything exactly ten times nearer, uniformly —
    which looks like a scene, not like a bug."""
    try:
        StereoRig.pre_rectified(mono_size=(640, 400), fx_rect=380.0, fy_rect=380.0,
                                cx_rect=320.0, cy_rect=200.0, baseline_mm=7.5)
    except CalibrationIncomplete as e:
        assert "centimetres" in str(e)
        return
    raise AssertionError("accepted a 7.5 mm baseline")


def test_rig_refuses_a_negative_baseline():
    try:
        StereoRig.pre_rectified(mono_size=(640, 400), fx_rect=380.0, fy_rect=380.0,
                                cx_rect=320.0, cy_rect=200.0, baseline_mm=-75.0)
    except CalibrationIncomplete:
        return
    raise AssertionError("accepted a negative baseline")


def test_calibration_json_without_extrinsics_is_refused():
    """Datasets recorded before extrinsics were saved cannot be reprocessed.
    Saying so is the correct behaviour; reaching for the EEPROM fx is not."""
    doc = {
        "streamed_intrinsics": {
            "left": {"fx": 378.56, "fy": 378.75, "cx": 324.43, "cy": 190.64,
                     "width": 640, "height": 400, "distortion": [0.0] * 14},
            "right": {"fx": 380.14, "fy": 380.08, "cx": 324.44, "cy": 193.81,
                      "width": 640, "height": 400, "distortion": [0.0] * 14},
        },
        "device": {"baseline_cm": 7.5},
    }
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "calibration.json"
        p.write_text(json.dumps(doc))
        try:
            StereoRig.from_calibration_json(p)
        except CalibrationIncomplete as e:
            assert "extrinsics" in str(e)
            assert "0.4%" in str(e)
            return
    raise AssertionError("built a rig with no extrinsics")


def test_calibration_json_refuses_an_unlabelled_translation():
    """DepthAI returns centimetres. An unlabelled vector is a 10x bug waiting."""
    doc = _json_doc_with_extrinsics({"rotation": np.eye(3).tolist(),
                                     "translation": [-75.0, 0.0, 0.0]})
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "calibration.json"
        p.write_text(json.dumps(doc))
        try:
            StereoRig.from_calibration_json(p)
        except CalibrationIncomplete as e:
            assert "unit" in str(e).lower()
            return
    raise AssertionError("accepted a translation with no unit in its key name")


def test_calibration_json_converts_centimetres_to_millimetres():
    """t_cm = -7.5 must become a 75 mm baseline, not a 7.5 mm one."""
    doc = _json_doc_with_extrinsics({"rotation": np.eye(3).tolist(),
                                     "t_cm": [-7.5, 0.0, 0.0]})
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "calibration.json"
        p.write_text(json.dumps(doc))
        rig = StereoRig.from_calibration_json(p)
    assert abs(rig.baseline_mm - 75.0) < 1e-6, rig.baseline_mm
    assert rig.fx_rect_source == "cv2.stereoRectify:P1[0,0]"


def test_calibration_json_takes_fx_from_stereo_rectify_not_from_eeprom():
    """The whole point. With real distortion the rectified fx is neither the
    left EEPROM value nor the right one, and the delta is recorded."""
    doc = _json_doc_with_extrinsics(
        {"rotation": np.eye(3).tolist(), "t_mm": [-75.0, 0.0, 0.0]},
        dist_left=[-1.40532, -0.62655, -9e-06, -0.001516, 2.000602,
                   -1.174531, -1.184813, 2.438596] + [0.0] * 6,
        dist_right=[-1.385192, -0.562238, 0.000664, 0.002364, 1.280538,
                    -1.159886, -1.073795, 1.60905] + [0.0] * 6)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "calibration.json"
        p.write_text(json.dumps(doc))
        rig = StereoRig.from_calibration_json(p, alpha=0.0)
    assert rig.fx_rect != 378.560547 and rig.fx_rect != 380.137756
    d = rig.describe()
    assert "fx_rect_vs_left_raw_pct" in d and "fx_rect_vs_right_raw_pct" in d
    assert abs(d["fx_rect_vs_left_raw_pct"]) > 0.0


def test_alpha_changes_fx_rect_and_is_recorded():
    """Different alpha, different focal length, different MinZ — so two datasets
    built with different alpha are not comparable, and alpha must be in the
    metadata for that to be discoverable."""
    doc = _json_doc_with_extrinsics(
        {"rotation": np.eye(3).tolist(), "t_mm": [-75.0, 0.0, 0.0]},
        dist_left=[-1.40532, -0.62655, -9e-06, -0.001516, 2.000602,
                   -1.174531, -1.184813, 2.438596] + [0.0] * 6,
        dist_right=[-1.40532, -0.62655, -9e-06, -0.001516, 2.000602,
                    -1.174531, -1.184813, 2.438596] + [0.0] * 6)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "calibration.json"
        p.write_text(json.dumps(doc))
        r0 = StereoRig.from_calibration_json(p, alpha=0.0)
        r1 = StereoRig.from_calibration_json(p, alpha=-1.0)
        r2 = StereoRig.from_calibration_json(p, alpha=1.0)
    assert abs(r0.fx_rect - r1.fx_rect) > 1.0, (r0.fx_rect, r1.fx_rect)
    assert r2.fx_rect < r0.fx_rect < r1.fx_rect, "alpha should order fx_rect"
    assert r0.describe()["alpha"] == 0.0 and r1.describe()["alpha"] == -1.0
    # And MinZ moves with it — which is why two alphas are two datasets.
    assert abs(r0.min_z_mm(96) - r1.min_z_mm(96)) > 5.0


def _json_doc_with_extrinsics(block: dict, dist_left=None, dist_right=None) -> dict:
    return {
        "streamed_intrinsics": {
            "left": {"fx": 378.560547, "fy": 378.753113, "cx": 324.433746,
                     "cy": 190.638, "width": 640, "height": 400,
                     "distortion": dist_left if dist_left is not None else [0.0] * 14},
            "right": {"fx": 380.137756, "fy": 380.079559, "cx": 324.441711,
                      "cy": 193.807693, "width": 640, "height": 400,
                      "distortion": dist_right if dist_right is not None else [0.0] * 14},
        },
        "device": {"baseline_cm": 7.5, "extrinsics": {"left_to_right": block}},
    }


def test_mono_input_is_required_and_never_guessed():
    """Raw and rectified frames are the same shape and dtype; nothing in the
    pixels says which one you have, and matching unrectified images does not
    fail, it just returns wrong distances."""
    try:
        HostStereoMatcher(RIG)          # type: ignore[call-arg]
    except TypeError:
        pass
    else:
        raise AssertionError("mono_input was optional")
    try:
        HostStereoMatcher(RIG, "auto")
    except ValueError as e:
        assert "never inferred" in str(e)
        return
    raise AssertionError("accepted mono_input='auto'")


def test_raw_input_requires_rectification_maps():
    """A synthetic rig has no maps, so it cannot accept raw frames."""
    try:
        HostStereoMatcher(RIG, MONO_INPUT_RAW)
    except HostDepthError as e:
        assert "rectification maps" in str(e)
        return
    raise AssertionError("accepted raw input with no maps")


def test_rectified_input_refuses_a_host_computed_rig():
    """Device-rectified images with this module's fx_rect would pair pixels from
    one rectification with a focal length from another — visible only as a scale
    error."""
    from c3_camera.host_depth import bench_rig
    rig = bench_rig((640, 400), with_color=False)
    assert rig.rectifies
    try:
        HostStereoMatcher(rig, MONO_INPUT_RECTIFIED)
    except HostDepthError as e:
        assert "pre_rectified" in str(e)
        return
    raise AssertionError("mixed two rectifications")


def test_raw_input_round_trip_through_real_rectification_maps():
    """The raw path end to end: distortion, remap, match, mm — on a rig built by
    cv2.stereoRectify. Only the shape/contract is asserted here, because a
    synthetic pair warped through real distortion has no closed-form disparity."""
    from c3_camera.host_depth import bench_rig
    rig = bench_rig((640, 400), with_color=False)
    m = HostStereoMatcher(rig, MONO_INPUT_RAW)
    depth = m.compute(*synthetic_pair((640, 400), 24, seed=19))
    assert depth.dtype == np.uint16 and depth.shape == (400, 640)
    assert (depth[depth > 0] >= m.min_z_mm - 5.0).all()


def test_matcher_refuses_a_size_it_was_not_built_for():
    """Intrinsics do not survive a resize by accident."""
    m = HostStereoMatcher(RIG, MONO_INPUT_RECTIFIED)
    try:
        m.compute(*synthetic_pair((320, 200), 12, seed=21))
    except ValueError as e:
        assert "rebuild the rig" in str(e)
        return
    raise AssertionError("matched frames of the wrong size")


def test_matcher_refuses_non_gray8_input():
    m = HostStereoMatcher(RIG, MONO_INPUT_RECTIFIED)
    l16 = np.zeros((400, 640), np.uint16)
    try:
        m.compute(l16, l16)
    except TypeError:
        return
    raise AssertionError("accepted a 16-bit mono pair")


def test_negative_min_disparity_is_refused():
    """Negative disparities would collide with SGBM's own invalid flag, which
    apply_mm_lut folds onto 0."""
    try:
        MatcherParams(min_disparity=-16).create()
    except ValueError:
        return
    raise AssertionError("accepted min_disparity < 0")


# =============================================================================
# metadata and threading
# =============================================================================
def test_describe_carries_everything_needed_to_tell_two_datasets_apart():
    m = HostStereoMatcher(RIG, MONO_INPUT_RECTIFIED, MatcherParams(num_disparities=192),
                          downscale=1)
    d = m.describe()
    for key in ("depth_source", "depth_scale", "mono_input", "k_mm_px",
                "min_z_mm", "max_z_mm", "matcher", "rig", "opencv_version",
                "downscale", "align_to_color"):
        assert key in d, f"describe() is missing {key}"
    assert d["depth_source"] == "host"
    assert d["depth_scale"] == 1000.0
    assert d["matcher"]["num_disparities"] == 192
    assert d["rig"]["fx_rect_source"] in ("explicit", "cv2.stereoRectify:P1[0,0]")
    json.dumps(d)   # must survive going into metadata.json


def test_async_pool_matches_the_single_threaded_result():
    """A worker pool is a throughput device, not a different algorithm."""
    pairs = [synthetic_pair((640, 400), 24, seed=100 + i) for i in range(4)]
    single = HostStereoMatcher(RIG, MONO_INPUT_RECTIFIED)
    want = [single.compute(l, r) for l, r in pairs]
    with AsyncHostStereo(lambda: HostStereoMatcher(RIG, MONO_INPUT_RECTIFIED),
                         workers=3) as pool:
        got = list(pool.map(pairs))
    assert len(got) == len(want)
    for a, b in zip(want, got):
        assert np.array_equal(a, b), "pooled result differs from single-threaded"


def test_benchmark_helper_runs_headless():
    """--benchmark must work with no camera and no dataset."""
    from c3_camera.host_depth import run_benchmark
    rows = run_benchmark(frames=2, workers=(1,), align_to_color=True,
                         cases=(("smoke", (320, 200), 48),))
    assert rows and rows[0]["fps"] > 0
    assert rows[0]["min_z_mm"] > 0


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
