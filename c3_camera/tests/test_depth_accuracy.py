#!/usr/bin/env python3
"""
test_depth_accuracy.py — the depth-accuracy statistics, on synthetic data only.

    ~/.venvs/c3-depthai/bin/python -m pytest c3_camera/tests/test_depth_accuracy.py -v
    ~/.venvs/c3-depthai/bin/python c3_camera/tests/test_depth_accuracy.py   # no pytest

Every test here builds a depth map whose answer is already known — a plane at a
chosen distance, tilted by a chosen angle, with a chosen amount of noise and a
chosen hole pattern — and then checks that c3_depth_accuracy.py recovers it.
That is the only way to know the measurement is sound, because the camera cannot
be opened to check it and a real wall has no ground truth better than the tape.

The choices being defended, and the test that defends each:

  * rho from a plane fit, rather than the centre pixel or the mean ROI depth,
    because rho survives camera tilt   -> test_plane_fit_survives_tilt_*
  * residuals reported along Z rather than along the plane normal
                                        -> test_residual_is_reported_along_z
  * spatial and temporal noise separated, because averaging removes one and not
    the other                           -> test_measure_stack_separates_*
  * holes reported per third, because the near-field loss is an edge band
                                        -> test_edge_band_and_speckle_*
  * the origin offset fitted, not assumed
                                        -> test_line_fit_recovers_*
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from c3_camera.c3_depth_accuracy import (
    FX_COLOR_3840,
    Gates,
    append_row,
    apply_gates,
    band_vanish_mm,
    column_fill_profile,
    fit_reported_vs_truth,
    fx_mono_px,
    measure_stack,
    measured_quantisation_mm,
    min_z_mm,
    overlap_band_frac,
    overlap_band_px,
    predict,
    quantisation_mm,
    read_rows,
    resid_limit_mm,
    robust_plane_fit,
    roi_slice,
    scale_from_differences,
    tan_half_fov,
    thirds_hole_fraction,
)
from c3_camera.geometry import Intrinsics

# Depth intrinsics of the existing dataset (colour-aligned, 480x270).
INTR = Intrinsics(fx=385.04, fy=385.09, cx=238.14, cy=134.83, width=480, height=270)
RNG = np.random.default_rng(20260803)


# =============================================================================
# Synthetic scenes
# =============================================================================
def plane_depth(intr: Intrinsics, rho_mm: float, tilt_deg: float = 0.0,
                axis: str = "yaw") -> np.ndarray:
    """Exact depth map of the plane n.X = rho, in float millimetres.

    A pixel's ray is d = ((u-cx)/fx, (v-cy)/fy, 1); the plane is hit at
    t = rho / (n.d), and since d_z = 1 that t *is* the depth. Nothing here uses
    the code under test, so the two can genuinely disagree.
    """
    t = math.radians(tilt_deg)
    n = (math.sin(t), 0.0, math.cos(t)) if axis == "yaw" \
        else (0.0, math.sin(t), math.cos(t))
    vv, uu = np.meshgrid(np.arange(intr.height, dtype=np.float64),
                         np.arange(intr.width, dtype=np.float64), indexing="ij")
    dx = (uu - intr.cx) / intr.fx
    dy = (vv - intr.cy) / intr.fy
    return rho_mm / (n[0] * dx + n[1] * dy + n[2])


def plane_points(intr: Intrinsics, rho_mm: float, tilt_deg: float = 0.0,
                 noise_mm: float = 0.0, stride: int = 3) -> np.ndarray:
    """(N,3) millimetre points on that plane, deprojected the same way the
    script does but written out longhand so a sign error cannot cancel."""
    z = plane_depth(intr, rho_mm, tilt_deg)[::stride, ::stride]
    if noise_mm:
        z = z + RNG.normal(0.0, noise_mm, z.shape)
    vv, uu = np.meshgrid(np.arange(intr.height, dtype=np.float64)[::stride],
                         np.arange(intr.width, dtype=np.float64)[::stride],
                         indexing="ij")
    x = (uu - intr.cx) * z / intr.fx
    y = (vv - intr.cy) * z / intr.fy
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)


def plane_stack(intr: Intrinsics, rho_mm: float, n_frames: int = 20,
                tilt_deg: float = 0.0, noise_mm: float = 0.0,
                fixed_pattern: np.ndarray | None = None,
                hole_mask: np.ndarray | None = None) -> np.ndarray:
    """(N,H,W) uint16 stack: one plane, optional fixed ripple, optional holes.

    `fixed_pattern` is added identically to every frame (calibration-style error
    that averaging cannot remove); `noise_mm` is redrawn per frame (sensor noise
    that averaging can). Keeping the two separable in the generator is what makes
    the decomposition testable at all.
    """
    base = plane_depth(intr, rho_mm, tilt_deg)
    if fixed_pattern is not None:
        base = base + fixed_pattern
    frames = []
    for _ in range(n_frames):
        d = base + (RNG.normal(0.0, noise_mm, base.shape) if noise_mm else 0.0)
        d = np.clip(np.round(d), 0, 65535).astype(np.uint16)
        if hole_mask is not None:
            d[hole_mask] = 0          # 0 = "no measurement", never 0 mm
        frames.append(d)
    return np.stack(frames)


def _metrics(**over) -> dict:
    """A plausible healthy measurement, for poking one gate at a time.

    `resid_rms_mm` is 10.5 rather than a prettier number on purpose: at 1 m on
    400p one disparity step is 35.1 mm, so a real fronto-parallel wall cannot come
    back flatter than dZ/sqrt(12) = 10.1 mm without subpixel. A fixture that
    claimed 2 mm here would be a configuration the camera cannot produce — and it
    would trip the quantisation-limited gate, correctly.
    """
    m = {
        "n_frames": 40, "n_fits": 40, "fit_reason": "",
        "fill_roi": 0.92, "inlier_frac": 0.95,
        "resid_rms_mm": 10.5, "tilt_deg": 1.5,
        "rho_iqr_mm": 0.4, "rho_mm": 1000.0,
    }
    m.update(over)
    return m


PRED_400P = predict("400p", False, 1000.0)


# =============================================================================
# Predicted geometry — these numbers are measured facts about this camera
# =============================================================================
def test_min_z_reproduces_the_three_measured_cliffs():
    """MinZ = fx*b/max_disparity, and there are only three distinct values on
    this camera. 400p + extended is the ONLY combination that reaches the
    vendor's claimed 18 cm.

    The 400p pair is not merely derived: every depth PNG this camera has written
    floors at exactly 300 mm, and every --extended one at exactly 150 mm, with
    zero pixels below either (c3_camera/datasets/*/depth/,
    c3_camera/recordings/*/depth/ — 47,270 frames scanned 2026-08-05). That is
    what fixes fx on CAM_C's 380.14 rather than CAM_B's 378.56, which would have
    printed a floor of 298.
    """
    fx400, fx1280 = fx_mono_px("400p"), fx_mono_px("800p")
    assert abs(fx400 - 380.14) < 0.01
    assert abs(fx1280 - 760.28) < 0.01
    assert abs(min_z_mm(fx400, False) - 300.1) < 0.1
    assert abs(min_z_mm(fx400, True) - 150.1) < 0.1
    assert abs(min_z_mm(fx1280, False) - 600.2) < 0.1
    assert abs(min_z_mm(fx1280, True) - 300.1) < 0.1
    assert min_z_mm(fx400, True) < 180.0 <= min_z_mm(fx400, False)
    # The observed floors are the integer millimetres these produce.
    assert int(min_z_mm(fx400, False)) == 300
    assert int(min_z_mm(fx400, True)) == 150
    # CAM_B's value would have floored at 298, which never appears in the data.
    assert int(min_z_mm(fx_mono_px("400p", 757.12), False)) == 298


def test_720p_and_800p_are_the_same_camera_as_far_as_depth_is_concerned():
    """Both are 1280 wide, so they share fx, MinZ and the quantisation step;
    720p is a vertical crop, hence a control rather than an independent rung."""
    assert fx_mono_px("720p") == fx_mono_px("800p")
    for ext in (False, True):
        assert min_z_mm(fx_mono_px("720p"), ext) == min_z_mm(fx_mono_px("800p"), ext)


def test_quantisation_grows_with_the_square_of_range():
    """dZ = Z^2/(fx*b): 3.2 mm at 300 mm becomes 140 mm at 2 m on 400p, and the
    1280-wide modes pay exactly half of both."""
    fx400, fx1280 = fx_mono_px("400p"), fx_mono_px("800p")
    for z, want in ((300, 3.16), (500, 8.77), (1000, 35.07), (2000, 140.30)):
        got = quantisation_mm(z, fx400)
        assert abs(got - want) < 0.1, (z, got, want)
        assert abs(quantisation_mm(z, fx1280) - want / 2) < 0.06


def test_overlap_band_matches_the_predicted_table_and_scales_with_range():
    """The un-matchable edge band is fx*b/Z mono pixels wide — 34.3% of the
    frame at 130 mm — and its *fraction* does not depend on mono resolution,
    because fx/width does not."""
    fx400 = fx_mono_px("400p")
    assert abs(overlap_band_px(130, fx400) - 219.3) < 0.5
    assert abs(overlap_band_px(130, fx400) / 640 - 0.343) < 0.002
    for z, frac in ((150, 0.297), (185, 0.241), (300, 0.149)):
        assert abs(overlap_band_px(z, fx400) / 640 - frac) < 0.002
    for res in ("400p", "720p", "800p"):
        p = predict(res, False, 130.0)
        assert abs(p.band_frac - 0.343) < 0.002, res


def test_depth_align_color_crops_the_band_out_of_the_delivered_frame():
    """Aligning depth to colour narrows the horizontal field of view from 80.2 to
    63.9 degrees, so most of the overlap band never reaches the delivered image
    and beyond ~343 mm none of it does. Band numbers taken under align=color are
    therefore not evidence about stereo overlap at all."""
    t_stereo = tan_half_fov(640, fx_mono_px("400p"))
    t_colour = tan_half_fov(480, FX_COLOR_3840 * 480 / 3840)
    assert abs(2 * math.degrees(math.atan(t_stereo)) - 80.2) < 0.1
    assert abs(2 * math.degrees(math.atan(t_colour)) - 63.9) < 0.1

    for z, frac in ((130, 0.288), (185, 0.150), (280, 0.040), (330, 0.007)):
        assert abs(overlap_band_frac(z, t_colour, t_stereo) - frac) < 0.002, z
    assert overlap_band_frac(400, t_colour, t_stereo) == 0.0
    assert abs(band_vanish_mm(t_colour, t_stereo) - 343.3) < 1.0

    # And when the delivered frame IS the mono frame, the general formula has to
    # collapse back onto fx*b/(Z*W).
    fx400 = fx_mono_px("400p")
    assert abs(overlap_band_frac(200, t_stereo, t_stereo)
               - overlap_band_px(200, fx400) / 640) < 1e-9


def test_predict_reports_the_delivered_band_when_told_the_output_optics():
    p = predict("400p", False, 185.0, fx_delivered=FX_COLOR_3840 * 480 / 3840,
                width_delivered=480)
    assert abs(p.band_frac - 0.241) < 0.002          # mono frame
    assert abs(p.delivered_band_frac - 0.150) < 0.002  # colour-aligned frame
    assert abs(p.delivered_band_px - 0.150 * 480) < 1.0
    assert abs(p.delivered_vanish_mm - 343.3) < 1.0


def test_prediction_carries_the_extended_flag_into_max_disparity():
    assert predict("400p", False, 300).max_disparity == 95
    assert predict("400p", True, 300).max_disparity == 190
    assert predict("400p", True, 300).min_z_mm < predict("400p", False, 300).min_z_mm


# =============================================================================
# Plane fit
# =============================================================================
def test_plane_fit_recovers_a_fronto_parallel_wall():
    pts = plane_points(INTR, 1000.0)
    fit = robust_plane_fit(pts)
    assert fit.ok, fit.reason
    assert abs(fit.rho_mm - 1000.0) < 0.05
    assert fit.tilt_deg < 0.05
    assert fit.resid_rms_mm < 0.05


def test_plane_fit_recovers_the_noise_it_was_given():
    fit = robust_plane_fit(plane_points(INTR, 1000.0, noise_mm=2.0))
    assert fit.ok
    assert abs(fit.rho_mm - 1000.0) < 0.5, fit.rho_mm
    # Trimming at 3 sigma clips ~0.3% of a Gaussian, so a small underestimate is
    # expected and correct; a large one would mean the fit is eating real signal.
    assert 1.80 < fit.resid_rms_mm < 2.15, fit.resid_rms_mm


def test_plane_fit_survives_tilt_that_would_wreck_a_mean_depth():
    """The reason rho exists. Tilted 8 degrees at 1 m, the ROI spans tens of
    millimetres of true depth, so any mean/centre statistic is badly wrong while
    the perpendicular distance is still right."""
    pts = plane_points(INTR, 1000.0, tilt_deg=8.0)
    fit = robust_plane_fit(pts)
    assert fit.ok
    assert abs(fit.rho_mm - 1000.0) < 0.5, fit.rho_mm
    assert abs(fit.tilt_deg - 8.0) < 0.2, fit.tilt_deg
    # What the naive statistic would have reported instead:
    spread = float(pts[:, 2].std())
    assert spread > 20.0, spread
    assert fit.resid_rms_mm < 0.5


def test_residual_is_reported_along_z_not_along_the_plane_normal():
    """Depth error is a Z-axis quantity. Noise injected along Z projects onto the
    normal as sigma*cos(tilt), so the fit divides by n_z; without that division a
    tilted rung would report artificially low noise."""
    sigma = 3.0
    flat = robust_plane_fit(plane_points(INTR, 800.0, noise_mm=sigma))
    tilted = robust_plane_fit(plane_points(INTR, 800.0, tilt_deg=25.0, noise_mm=sigma))
    assert flat.ok and tilted.ok
    assert abs(tilted.resid_rms_mm - sigma) < 0.25, tilted.resid_rms_mm
    # Without the n_z correction this would have come back ~cos(25) = 0.906x low.
    assert tilted.resid_rms_mm > flat.resid_rms_mm * 0.95


def test_plane_fit_trims_outliers_instead_of_being_dragged_by_them():
    pts = plane_points(INTR, 1000.0, noise_mm=1.0)
    n_bad = pts.shape[0] // 5
    pts[:n_bad, 2] += 400.0            # a fifth of the region on a nearer object
    fit = robust_plane_fit(pts)
    assert fit.ok
    assert abs(fit.rho_mm - 1000.0) < 2.0, fit.rho_mm
    assert 0.75 < fit.inlier_frac < 0.85, fit.inlier_frac
    # The un-trimmed least squares answer, for contrast:
    assert abs(float(pts[:, 2].mean()) - 1000.0) > 60.0


def test_plane_fit_is_deterministic():
    """A measurement script must give the same answer twice on the same data;
    that is why the fit trims deterministically instead of using RANSAC."""
    pts = plane_points(INTR, 1000.0, noise_mm=2.0)
    a, b = robust_plane_fit(pts), robust_plane_fit(pts)
    assert a.rho_mm == b.rho_mm and a.resid_rms_mm == b.resid_rms_mm


def test_plane_fit_refuses_rather_than_guesses_when_starved():
    assert not robust_plane_fit(np.zeros((10, 3))).ok
    assert "10 valid points" in robust_plane_fit(np.zeros((10, 3))).reason
    assert not robust_plane_fit(np.zeros((5, 2))).ok


def test_plane_fit_floor_keeps_heavily_quantised_depth_fittable():
    """At 2 m on 400p one disparity step is 141 mm, so a whole region can land on
    a single depth level and MAD becomes exactly 0. A pure 3*MAD threshold would
    then reject everything."""
    z = np.full((90, 160), 2000.0)
    vv, uu = np.meshgrid(np.arange(90.0), np.arange(160.0), indexing="ij")
    pts = np.stack([(uu.ravel() - 80) * 2000 / INTR.fx,
                    (vv.ravel() - 45) * 2000 / INTR.fy, z.ravel()], axis=-1)
    fit = robust_plane_fit(pts, floor_mm=141.0)
    assert fit.ok, fit.reason
    assert abs(fit.rho_mm - 2000.0) < 1.0


# =============================================================================
# measure_stack
# =============================================================================
def test_measure_stack_reports_a_planted_bias():
    """A camera that reports 25 mm long must come back as +25 mm, not as 'about
    right'. This is the whole measurement in one assertion."""
    stack = plane_stack(INTR, 1025.0, n_frames=12, noise_mm=1.0)
    m = measure_stack(stack, INTR)
    assert abs((m["rho_mm"] - 1000.0) - 25.0) < 0.5, m["rho_mm"]
    assert m["fill_roi"] == 1.0
    assert m["n_frames"] == 12 and m["n_fits"] == 12


def test_measure_stack_separates_temporal_noise_from_fixed_pattern():
    """Random noise averages away with more frames; a calibration ripple does
    not. They cost SLAM completely different amounts, so the script splits the
    single-frame residual into the two using the per-pixel time series."""
    vv, uu = np.meshgrid(np.arange(INTR.height, dtype=np.float64),
                         np.arange(INTR.width, dtype=np.float64), indexing="ij")
    ripple = 4.0 * np.sin(2 * np.pi * uu / 9.0) * np.sin(2 * np.pi * vv / 11.0)
    r0, r1, c0, c1 = roi_slice(INTR.height, INTR.width, (0.4, 0.4))
    planted_fixed = float(ripple[r0:r1, c0:c1].std())
    sigma = 2.0

    m = measure_stack(plane_stack(INTR, 1000.0, n_frames=24, noise_mm=sigma,
                                  fixed_pattern=ripple), INTR)
    assert abs(m["temporal_rms_mm"] - sigma) < 0.2, m["temporal_rms_mm"]
    assert abs(m["fixed_rms_mm"] - planted_fixed) < 0.35, \
        (m["fixed_rms_mm"], planted_fixed)
    # And the single-frame residual really is the two added in quadrature.
    assert abs(m["resid_rms_mm"] - math.hypot(sigma, planted_fixed)) < 0.35


def test_measure_stack_would_report_zero_temporal_noise_on_duplicate_frames():
    """Documents the trap the capture loop exists to avoid: C3Source.read() hands
    back the newest frame of each stream, so polling faster than the depth rate
    yields the same image repeatedly. Stacked, it looks like a perfect camera."""
    one = plane_stack(INTR, 1000.0, n_frames=1, noise_mm=3.0)
    duplicated = np.repeat(one, 15, axis=0)
    m = measure_stack(duplicated, INTR)
    assert m["temporal_rms_mm"] == 0.0
    assert m["resid_rms_mm"] > 2.0           # the spatial noise is still there
    # The honest stack, same sigma, reports it:
    honest = measure_stack(plane_stack(INTR, 1000.0, n_frames=15, noise_mm=3.0), INTR)
    assert honest["temporal_rms_mm"] > 2.5


def test_measure_stack_does_not_let_rig_creep_masquerade_as_sensor_noise():
    """Common mode is removed before the per-pixel time series is taken, so a
    slowly drifting rig shows up in rho_range and not in the noise figure."""
    frames = [plane_stack(INTR, 1000.0 + 0.6 * i, n_frames=1, noise_mm=0.5)[0]
              for i in range(16)]
    m = measure_stack(np.stack(frames), INTR)
    assert m["rho_range_mm"] > 8.0, m["rho_range_mm"]
    assert m["temporal_rms_mm"] < 1.0, m["temporal_rms_mm"]


def test_measure_stack_prefers_the_plane_residual_over_the_naive_roi_spread():
    """Both are reported so the difference is visible. Tilted, the naive ROI
    standard deviation is an order of magnitude larger and is measuring the aim,
    not the camera."""
    m = measure_stack(plane_stack(INTR, 1000.0, n_frames=8, tilt_deg=8.0,
                                  noise_mm=1.0), INTR)
    assert m["resid_rms_mm"] < 1.5, m["resid_rms_mm"]
    assert m["patch_std_mm"] > 10 * m["resid_rms_mm"], m["patch_std_mm"]
    assert abs(m["tilt_deg"] - 8.0) < 0.3


def test_measure_stack_counts_holes_and_ignores_them_in_the_fit():
    holes = RNG.random((INTR.height, INTR.width)) < 0.30
    m = measure_stack(plane_stack(INTR, 1000.0, n_frames=8, noise_mm=1.0,
                                  hole_mask=holes), INTR)
    assert abs(m["fill_all"] - 0.70) < 0.02, m["fill_all"]
    assert abs(m["fill_roi"] - 0.70) < 0.03, m["fill_roi"]
    assert abs(m["rho_mm"] - 1000.0) < 0.6      # holes must not move the answer


def test_zero_is_excluded_from_every_reduction_not_just_the_plane_fit():
    """The single most likely bug in a depth script, checked reduction by
    reduction. Depth 0 means 'no measurement', so a frame that is 60% holes must
    return the SAME distance as the same frame with no holes at all. One missed
    mask anywhere would drag rho toward 0.4 x 1000 mm."""
    clean = plane_stack(INTR, 1000.0, 20, noise_mm=1.5)
    holes = RNG.random((INTR.height, INTR.width)) < 0.60
    holey = plane_stack(INTR, 1000.0, 20, noise_mm=1.5, hole_mask=holes)
    a, b = measure_stack(clean, INTR), measure_stack(holey, INTR)

    # Every statistic that averages depths, one at a time.
    assert abs(b["rho_mm"] - a["rho_mm"]) < 1.0, (a["rho_mm"], b["rho_mm"])
    assert abs(b["z_center_mm"] - a["z_center_mm"]) < 2.0
    assert abs(b["patch_std_mm"] - a["patch_std_mm"]) < 0.5
    assert abs(b["temporal_rms_mm"] - a["temporal_rms_mm"]) < 0.5
    assert abs(b["resid_rms_mm"] - a["resid_rms_mm"]) < 0.5
    assert abs(b["z_p50_mm"] - a["z_p50_mm"]) < 2.0
    assert b["z_min_mm"] > 900.0            # a 0 leaking in would make this 0
    # ... while the coverage statistics DO see the holes, which is the control
    # proving the assertions above are not passing by accident.
    assert abs(b["fill_roi"] - 0.40) < 0.03 and a["fill_roi"] == 1.0


def test_measure_stack_reports_a_camera_that_reads_SHORT_as_a_negative_bias():
    """The companion to the +25 mm test: a sign error in rho or in the deprojec-
    tion would show up as a bias of the wrong sign, which is worse than no
    measurement because it points the correction the wrong way."""
    m = measure_stack(plane_stack(INTR, 975.0, 12, noise_mm=1.0), INTR)
    assert m["rho_mm"] < 1000.0
    assert abs((m["rho_mm"] - 1000.0) + 25.0) < 0.5, m["rho_mm"]


def test_noise_does_not_grow_with_tilt_across_the_whole_working_range():
    """The claim the plane fit exists to make, swept rather than spot-checked:
    a flat wall carries the same noise however it is aimed, while the naive ROI
    spread grows without limit. A variance-for-standard-deviation slip or a
    missing 1/n_z would break the flat line."""
    sigma = 2.0
    got = []
    for tilt in (0.0, 5.0, 15.0, 25.0):
        m = measure_stack(plane_stack(INTR, 1000.0, 12, tilt_deg=tilt,
                                      noise_mm=sigma), INTR)
        got.append((tilt, m["resid_rms_mm"], m["patch_std_mm"]))
        assert abs(m["resid_rms_mm"] - sigma) < 0.35, got[-1]
        assert abs(m["rho_mm"] - 1000.0) < 1.0
    # The naive statistic, over the same sweep, is a function of the aim alone.
    assert got[-1][2] > 30 * got[0][2], got


def test_measure_stack_min_mm_keeps_the_range_extended_disparity_unlocks():
    """c3_collect.py's display floor is 300 mm, which is exactly the band
    extended disparity opens up. The default here is 100 mm for that reason."""
    stack = plane_stack(INTR, 200.0, n_frames=6, noise_mm=0.5)
    kept = measure_stack(stack, INTR, min_mm=100.0)
    dropped = measure_stack(stack, INTR, min_mm=300.0)
    assert abs(kept["rho_mm"] - 200.0) < 1.0
    assert kept["fill_roi"] == 1.0
    assert dropped["fill_roi"] == 0.0 and dropped["n_fits"] == 0


def test_measured_quantisation_recovers_the_disparity_grid():
    """Settles what subpixel/extended did to the depth grid by measurement
    instead of by argument about the firmware."""
    grid = (np.round(plane_depth(INTR, 1000.0, tilt_deg=20.0) / 8.0) * 8.0)
    assert measured_quantisation_mm(grid.astype(np.uint16)) == 8.0
    # Too few distinct levels to say anything: report nothing, not 0.
    assert math.isnan(measured_quantisation_mm(np.full((40, 40), 1000, np.uint16)))


def test_measured_quantisation_survives_a_sparse_garbage_roi():
    """The regression that cost a whole captured rung. A below-MinZ negative
    control, or a texture-less wall, leaves a handful of scattered depths whose
    every gap is metres wide. Reporting nothing is right; raising is not, because
    this runs AFTER the hold is over and takes the capture with it."""
    scattered = np.zeros((40, 40), np.uint16)
    for i, v in enumerate((500, 1600, 2800, 4100, 5300, 7000, 9000, 12000)):
        scattered[i, i] = v
    assert math.isnan(measured_quantisation_mm(scattered))
    # And the same thing reached through the real entry point.
    stack = np.zeros((12, INTR.height, INTR.width), np.uint16)
    for i, v in enumerate((500, 1600, 2800, 4100, 5300, 7000, 9000, 12000, 15000)):
        stack[:, INTR.height // 2 + i, INTR.width // 2 + i] = v
    m = measure_stack(stack, INTR)          # must not raise
    assert math.isnan(m["quant_dz_meas_mm"])


def test_measure_stack_rejects_a_stack_that_is_not_a_stack():
    for bad in (np.zeros((10, 10), np.uint16), np.zeros((2, 4, 4, 3), np.uint16)):
        try:
            measure_stack(bad, INTR)
        except ValueError:
            continue
        raise AssertionError("expected ValueError for a non-(N,H,W) stack")


# =============================================================================
# Spatial structure of the holes
# =============================================================================
def test_edge_band_and_speckle_have_the_same_fill_but_different_thirds():
    """The requirement in one test. The near-field stereo loss is a band down one
    side, and an overall fill percentage cannot tell it apart from the same
    number of holes sprinkled everywhere — so the thirds are reported instead."""
    h, w = INTR.height, INTR.width
    band = np.zeros((h, w), bool)
    band[:, :int(0.25 * w)] = True                     # structural: one edge
    speckle = RNG.random((h, w)) < 0.25                # cosmetic: everywhere

    band_m = measure_stack(plane_stack(INTR, 1000.0, 6, noise_mm=1.0,
                                       hole_mask=band), INTR)
    speck_m = measure_stack(plane_stack(INTR, 1000.0, 6, noise_mm=1.0,
                                        hole_mask=speckle), INTR)

    assert abs(band_m["fill_all"] - speck_m["fill_all"]) < 0.02   # indistinguishable
    assert band_m["hole_left"] > 0.70 and band_m["hole_right"] < 0.02
    assert band_m["hole_asym"] > 0.70                             # sign names the side
    assert abs(speck_m["hole_asym"]) < 0.03                       # symmetric
    assert abs(speck_m["hole_left"] - 0.25) < 0.02


def test_thirds_hole_fraction_on_an_exact_pattern():
    fill = np.ones((30, 300))
    fill[:, :100] = 0.0
    t = thirds_hole_fraction(fill)
    assert t["hole_left"] == 1.0 and t["hole_centre"] == 0.0 and t["hole_right"] == 0.0
    assert t["hole_asym"] == 1.0
    fill_r = np.ones((30, 300))
    fill_r[:, 200:] = 0.0
    assert thirds_hole_fraction(fill_r)["hole_asym"] == -1.0


def test_column_fill_profile_locates_the_band():
    fill = np.ones((20, 480))
    fill[:, :120] = 0.0
    prof = column_fill_profile(fill, bins=24)
    assert len(prof) == 24
    assert all(p == 0.0 for p in prof[:6])
    assert all(p == 1.0 for p in prof[6:])


def test_roi_stays_clear_of_the_predicted_band_at_the_default_settings():
    """The centre ROI has to sit outside the overlap band at every rung, or the
    bias curve becomes a function of the band as well as of distance. At the
    closest rung the margin is only about 7 px, so this is worth pinning."""
    _, _, c0, _ = roi_slice(270, 480, (0.4, 0.4))
    assert c0 == 144
    worst = predict("400p", True, 130.0, fx_delivered=FX_COLOR_3840 * 480 / 3840,
                    width_delivered=480)
    assert worst.delivered_band_px < c0, (worst.delivered_band_px, c0)
    assert c0 - worst.delivered_band_px < 15.0     # margin, not comfort


# =============================================================================
# Reported vs truth: the origin offset is a result, not an assumption
# =============================================================================
def test_line_fit_recovers_a_known_scale_and_origin_offset():
    """The camera measures from an optical centre nobody has located, so the
    offset is fitted. Planted here as +25 mm behind the tape datum."""
    truth = np.array([450, 560, 660, 1000, 1500, 2000], float)
    reported = 1.0 * truth + 25.0 + RNG.normal(0, 0.8, truth.size)
    fit = fit_reported_vs_truth(truth, reported)
    assert fit.ok and fit.n == 6
    assert abs(fit.slope - 1.0) < 0.005, fit.slope
    assert abs(fit.intercept_mm - 25.0) < 3.0, fit.intercept_mm
    assert fit.intercept_se_mm < 3.0
    assert fit.r2 > 0.999


def test_line_fit_recovers_a_scale_error_too():
    """A 1.33 scale is what an uncorrected in-air calibration gives underwater,
    so the slope has to be a fitted quantity rather than assumed to be 1."""
    truth = np.array([500, 700, 1000, 1400, 2000], float)
    fit = fit_reported_vs_truth(truth, 1.33 * truth - 10.0)
    assert abs(fit.slope - 1.33) < 1e-6
    assert abs(fit.intercept_mm + 10.0) < 1e-6


def test_line_fit_admits_when_it_cannot_state_an_uncertainty():
    """Two rungs define a line exactly. Reporting a standard error of 0 there
    would be a lie, so it reports nothing."""
    fit = fit_reported_vs_truth([500.0, 1000.0], [525.0, 1025.0])
    assert fit.ok and fit.n == 2
    assert abs(fit.slope - 1.0) < 1e-9 and abs(fit.intercept_mm - 25.0) < 1e-9
    assert math.isnan(fit.slope_se) and math.isnan(fit.intercept_se_mm)


def test_line_fit_standard_error_grows_with_scatter():
    truth = np.array([450, 560, 660, 1000, 1500, 2000], float)
    tight = fit_reported_vs_truth(truth, truth + 25 + RNG.normal(0, 0.5, 6))
    loose = fit_reported_vs_truth(truth, truth + 25 + RNG.normal(0, 20.0, 6))
    assert loose.intercept_se_mm > 5 * tight.intercept_se_mm


def test_line_fit_refuses_degenerate_input():
    assert not fit_reported_vs_truth([1000.0], [1025.0]).ok
    assert not fit_reported_vs_truth([1000.0, 1000.0], [1025.0, 1030.0]).ok
    assert not fit_reported_vs_truth([1000.0, np.nan], [1025.0, 1030.0]).ok


def test_scale_from_differences_cannot_see_the_origin_offset():
    """An independent read on the scale: differencing adjacent rungs removes the
    datum offset entirely, so it cannot be contaminated by a wrong guess at it."""
    truth = np.array([450, 560, 660, 1000, 1500, 2000], float)
    a1 = scale_from_differences(truth, 1.05 * truth + 25.0)
    a2 = scale_from_differences(truth, 1.05 * truth - 300.0)
    assert abs(a1 - 1.05) < 1e-9 and abs(a2 - 1.05) < 1e-9


# =============================================================================
# Refusals
# =============================================================================
def test_gates_pass_a_healthy_rung():
    ok, warns, fails = apply_gates(_metrics(), 1000.0, PRED_400P, Gates(),
                                   exposure_ms=8.0, exposure_spread_ms=0.1,
                                   motion=1.0)
    assert ok == 1 and not fails and not warns


def test_gates_refuse_a_sparse_roi_instead_of_reporting_its_bias():
    ok, _, fails = apply_gates(_metrics(fill_roi=0.12), 1000.0, PRED_400P, Gates())
    assert ok == 0
    assert any("12.0% valid" in f for f in fails), fails


def test_gates_refuse_a_target_that_is_not_flat():
    """The limit is one disparity step (35.2 mm at 1 m on 400p), so a residual of
    a couple of steps passes and a region that is not one plane does not."""
    ok, _, fails = apply_gates(_metrics(resid_rms_mm=200.0), 1000.0, PRED_400P,
                               Gates())
    assert ok == 0 and any("not flat" in f for f in fails), fails
    ok2, _, _ = apply_gates(_metrics(resid_rms_mm=20.0), 1000.0, PRED_400P, Gates())
    assert ok2 == 1


def test_gates_refuse_when_the_depths_do_not_lie_on_one_plane():
    ok, _, fails = apply_gates(_metrics(inlier_frac=0.35), 1000.0, PRED_400P, Gates())
    assert ok == 0 and any("fitted plane" in f for f in fails), fails


def test_gates_refuse_a_moving_rig():
    ok, _, fails = apply_gates(_metrics(), 1000.0, PRED_400P, Gates(), motion=20.0)
    assert ok == 0 and any("moved during the hold" in f for f in fails), fails


def test_gates_call_out_motion_blur_and_escalate_when_it_actually_moved():
    _, warns, fails = apply_gates(_metrics(), 1000.0, PRED_400P, Gates(),
                                  exposure_ms=45.0, motion=0.5)
    assert not fails and any("smears the stereo pair" in w for w in warns), warns
    ok, _, fails = apply_gates(_metrics(), 1000.0, PRED_400P, Gates(),
                               exposure_ms=45.0, motion=4.0)
    assert ok == 0 and any("the scene moved" in f for f in fails), fails


def test_gates_notice_auto_exposure_hunting():
    _, warns, _ = apply_gates(_metrics(), 1000.0, PRED_400P, Gates(),
                              exposure_ms=10.0, exposure_spread_ms=6.0)
    assert any("hunting" in w for w in warns), warns


def test_gates_treat_a_below_min_z_rung_as_a_negative_control():
    """At 130 mm the default configuration has no disparity to find. Empty is the
    correct result, and the refusal has to say so rather than imply a defect."""
    ok, warns, fails = apply_gates(_metrics(fill_roi=0.0), 130.0, PRED_400P, Gates())
    assert ok == 0
    assert any("negative control" in w for w in warns), warns
    assert any("expected" in f for f in fails), fails


def test_gates_shout_when_depth_appears_below_the_geometric_minimum():
    """The single most important check in the protocol: valid depth below MinZ
    means something is inventing values, which invalidates every fill number in
    the run — not just this rung's."""
    _, warns, _ = apply_gates(_metrics(fill_roi=0.9, rho_mm=140.0), 130.0,
                              PRED_400P, Gates())
    assert any("inventing depth" in w for w in warns), warns


def test_gates_flag_a_gross_disagreement_with_the_tape():
    _, warns, _ = apply_gates(_metrics(rho_mm=1600.0), 1000.0, PRED_400P, Gates())
    assert any("more than the datum offset can explain" in w for w in warns), warns


def test_gates_warn_about_aim_but_still_report_rho():
    """Tilt does not invalidate rho, so it warns instead of refusing — the whole
    reason rho was chosen over a mean depth."""
    ok, warns, fails = apply_gates(_metrics(tilt_deg=25.0), 1000.0, PRED_400P,
                                   Gates())
    assert ok == 1 and not fails
    assert any("tilted" in w for w in warns), warns


def test_gates_refuse_a_starved_capture():
    ok, _, fails = apply_gates(_metrics(n_frames=3), 1000.0, PRED_400P, Gates())
    assert ok == 0 and any("unique depth frames" in f for f in fails), fails


def test_gates_catch_the_rung_that_landed_on_one_disparity_bin():
    """The failure that looks like a perfect measurement. At 2 m on 400p a
    disparity step is 140 mm while a wall three degrees off square only spreads
    the ROI over ~50 mm, so every pixel lands in the same bin: residual ~0,
    inliers 1.00, every gate green — and an rho that is the bin rather than the
    wall, carrying +-70 mm the residual cannot show."""
    pred2m = predict("400p", False, 2000.0)
    assert abs(pred2m.quant_mm - 140.3) < 0.5
    ok, warns, fails = apply_gates(_metrics(resid_rms_mm=0.0, rho_mm=2028.0,
                                            inlier_frac=1.0, rho_iqr_mm=0.0),
                                   2000.0, pred2m, Gates(),
                                   exposure_ms=8.0, motion=1.0)
    assert not fails                       # nothing about it is invalid, only blind
    assert any("quantisation-limited" in w for w in warns), warns
    assert any("+-70 mm" in w for w in warns), warns

    # A rung that IS sampling the grid — resid ~ dZ/sqrt(12) — must stay quiet, or
    # the warning fires on every honest long-range rung and gets ignored.
    _, quiet, _ = apply_gates(_metrics(resid_rms_mm=40.0), 2000.0, pred2m, Gates(),
                              exposure_ms=8.0, motion=1.0)
    assert not any("quantisation-limited" in w for w in quiet), quiet

    # And it must stay quiet under --subpixel, where the integer-disparity step
    # this warning is named after has been subdivided and no longer exists.
    sub = predict("400p", False, 2000.0, subpixel=True)
    _, sub_warns, _ = apply_gates(_metrics(resid_rms_mm=0.0), 2000.0, sub, Gates(),
                                  exposure_ms=8.0, motion=1.0)
    assert not any("quantisation-limited" in w for w in sub_warns), sub_warns


def test_gates_notice_when_the_overlap_band_reaches_into_the_roi():
    """The band warning has to fire during a real capture, not only in --dry-run:
    a bias fitted partly to pixels the stereo pair never saw is exactly the kind
    of confident wrong number this script exists to refuse."""
    near = predict("400p", True, 130.0, fx_delivered=FX_COLOR_3840 * 480 / 3840,
                   width_delivered=480)
    # A wide ROI leaves a 48 px margin, narrower than the 137 px band.
    wide = _metrics(depth_w=480, roi_w_px=384, resid_rms_mm=1.0)
    _, warns, _ = apply_gates(wide, 130.0, near, Gates(),
                              exposure_ms=8.0, motion=1.0)
    assert any("overlap band" in w for w in warns), warns

    # The shipped default ROI (0.4 of the width) clears it, and must not warn.
    tight = _metrics(depth_w=480, roi_w_px=192, resid_rms_mm=1.0)
    _, quiet, _ = apply_gates(tight, 130.0, near, Gates(),
                              exposure_ms=8.0, motion=1.0)
    assert not any("overlap band" in w for w in quiet), quiet


def test_resid_limit_scales_with_range_and_with_quantisation():
    """A fixed flatness threshold would pass everything at 2 m and fail
    everything at 150 mm, because both the disparity step and the projected tilt
    grow with distance."""
    fx = fx_mono_px("400p")
    near = resid_limit_mm(150.0, quantisation_mm(150.0, fx))
    far = resid_limit_mm(2000.0, quantisation_mm(2000.0, fx))
    assert near < far
    # At 2 m the quantisation step (141 mm) is what binds, and it must leave room
    # above the dZ/sqrt(12) = 41 mm that quantisation alone contributes.
    assert far >= quantisation_mm(2000.0, fx) - 1e-6
    assert far > quantisation_mm(2000.0, fx) / math.sqrt(12) * 2
    assert resid_limit_mm(2000.0, 141.0, override=5.0) == 5.0


# =============================================================================
# The accumulating CSV
# =============================================================================
def test_csv_appends_rungs_and_writes_missing_numbers_as_empty_cells():
    """A missing measurement must not be stored as the string 'nan', which would
    read back as a number in half the tools that open this file."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "rungs.csv"
        append_row(path, {"truth_mm": 300.0, "label": "default", "rho_mm": 325.5,
                          "bias_mm": 25.5, "ok": 1, "extended": 0})
        append_row(path, {"truth_mm": 185.0, "label": "ext", "rho_mm": float("nan"),
                          "bias_mm": float("nan"), "ok": 0, "extended": 1,
                          "refused": "below MinZ"})
        rows = read_rows(path)
        assert len(rows) == 2
        assert rows[0]["rho_mm"] == "325.5" and rows[0]["ok"] == "1"
        assert rows[1]["rho_mm"] == "" and rows[1]["refused"] == "below MinZ"
        # A second invocation appends rather than overwriting the first rung.
        append_row(path, {"truth_mm": 1000.0, "label": "default", "ok": 1})
        assert len(read_rows(path)) == 3


def test_settings_that_change_the_answer_are_not_pooled_into_one_group():
    """--analyze groups rungs by setting_key and then cross-checks that the fitted
    datum offset agrees between groups. If the key cannot see a knob, incomparable
    rungs are averaged and the cross-check agrees for the wrong reason. A preset
    sets several matcher parameters besides the confidence threshold, and the
    delivered size decides how much resampling stands between the disparity grid
    and the fitted number, so both have to be in the key."""
    from c3_camera.c3_depth_accuracy import base_key, setting_key
    base = {"mono_res": "400p", "extended": "0", "subpixel": "0", "lr_check": "1",
            "median": "off", "confidence": "245", "depth_align": "color",
            "depth_preset": "robotics", "depth_w": "640", "depth_h": "360"}
    for field_, other in (("depth_preset", "high_accuracy"),
                          ("depth_w", "320"), ("mono_res", "800p"),
                          ("median", "5x5"), ("depth_align", "none")):
        assert setting_key(base) != setting_key(dict(base, **{field_: other})), field_
    # ... except the extended flag, which is the axis base_key deliberately pools.
    assert setting_key(base) != setting_key(dict(base, extended="1"))
    assert base_key(base) == base_key(dict(base, extended="1"))
    assert base_key(base) != base_key(dict(base, depth_preset="high_density"))


def test_the_measurement_defaults_resolve_and_build_a_pipeline():
    """The one part of the capture path that can be checked without a camera:
    the configuration this script pins really does produce a valid pipeline,
    within the PoE budget, on both arms of the comparison. (Building a pipeline
    does not open a device on depthai 2.x — the package refuses to load on 3.x,
    where constructing one would.)"""
    from c3_camera.c3_depth_accuracy import build_config, parse_args
    from c3_camera.config import POE_BUDGET_MBPS
    from c3_camera.pipeline import build_pipeline

    for extra in ([], ["--extended"], ["--median", "5x5"],
                  ["--depth-align", "none", "--streams", "depth,left"]):
        cfg = build_config(parse_args(["--truth-mm", "300"] + extra))
        rc = cfg.resolve()
        pipeline, formats = build_pipeline(rc)
        assert formats["depth"] == "depth16", extra
        assert pipeline.getAllNodes()
        assert rc.bandwidth_mbps()["total"] < POE_BUDGET_MBPS, (extra, rc.budget_note())


def test_measurement_defaults_pin_the_settings_that_would_otherwise_drift():
    """median off (it fills holes and flatters the residual), confidence pinned
    (the robotics preset silently leaves it at 245/255), and the left stream on
    (it is where the exposure, texture and motion gates get their evidence)."""
    from c3_camera.c3_depth_accuracy import build_config, parse_args
    cfg = build_config(parse_args(["--truth-mm", "300"]))
    assert cfg.median == "off"
    assert cfg.confidence == 245
    assert "left" in cfg.streams
    assert cfg.subpixel is False


def test_measure_stack_output_fits_the_csv_schema():
    """Everything the measurement computes has somewhere to be stored; a metric
    with no column is a metric that silently vanishes."""
    from c3_camera.c3_depth_accuracy import FIELDS
    m = measure_stack(plane_stack(INTR, 1000.0, 6, noise_mm=1.0), INTR)
    internal = {"fit_reason", "n_fits", "inlier_frac"}
    missing = [k for k in m if k not in FIELDS and k not in internal]
    assert not missing, f"measured but never stored: {missing}"


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
