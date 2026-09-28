#!/usr/bin/env python3
"""Tests for the second-camera fallback (control/nav_fusion.py) and the mount
tilt tracker (control/tilt_tracker.py). Pure numpy — no Qt, no camera.

Run:  python -m rov_gui.tests.test_nav_fusion
"""

from __future__ import annotations

import math
import sys

import numpy as np

from rov_gui.control.nav_fusion import (BodyAlign, FallbackArbiter, R_of_rotvec,
                                        T_inv, T_of, rotvec_of, tilt_down_deg)
from rov_gui.control.tilt_tracker import TiltTracker


def _rot_zyx(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
    Ry = np.array([[cp, 0, sp], [0, 1.0, 0], [-sp, 0, cp]])
    Rx = np.array([[1.0, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def _second_cam_extrinsic(tilt_deg: float):
    """R_frd_cam for the ROV RGB the way geometry.R_t_frd_cam('second')
    builds it: level optical axis along body +x, then pitched DOWN."""
    from rov_gui.control.geometry import NavConfig
    cfg = NavConfig()
    cfg.second_cam = dict(cfg.second_cam)
    cfg.second_cam["tilt_deg"] = float(tilt_deg)
    return cfg.R_t_frd_cam("second")


# =============================================================================
# SE(3) helpers
# =============================================================================
def test_rotvec_round_trip():
    for w in ([0.0, 0.0, 0.0], [0.3, -0.2, 0.1], [0.0, 1.2, 0.0], [2.0, 0.5, -0.4]):
        R = R_of_rotvec(w)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.allclose(rotvec_of(R), w, atol=1e-9), (w, rotvec_of(R))


def test_tilt_down_reads_the_extrinsic_convention():
    """geometry._tilt_flu's 30-deg-down mount must read as +30 here — the
    same check test_control makes on z_cv, expressed as an angle."""
    R_bc, _t = _second_cam_extrinsic(30.0)
    assert abs(tilt_down_deg(R_bc) - 30.0) < 1e-9
    R_bc, _t = _second_cam_extrinsic(0.0)
    assert abs(tilt_down_deg(R_bc)) < 1e-9


# =============================================================================
# BodyAlign
# =============================================================================
def _simulate_pair_stream(align: BodyAlign, E_true: np.ndarray, n: int,
                          noise_m: float = 0.0, seed: int = 0):
    """Feed n time-paired fixes: the main is the truth, the second is the
    truth composed with a constant body-frame error E (exactly the model)."""
    rng = np.random.default_rng(seed)
    for k in range(n):
        t = 10.0 + 0.1 * k
        p = np.array([0.3 * math.sin(0.2 * k), 0.4 * math.cos(0.13 * k), -0.8])
        R = _rot_zyx(0.05 * math.sin(k), 0.04 * math.cos(k), 0.3 * k)
        T_main = T_of(R, p)
        T_second = T_main @ E_true
        if noise_m:
            T_second = T_second @ T_of(np.eye(3), rng.normal(0, noise_m, 3))
        align.note_main(t, T_main)
        assert align.note_second(t + 0.02, T_second)


def test_body_align_learns_a_constant_extrinsic_error_exactly():
    """The whole premise: the RGB's wrong extrinsic is a constant body-frame
    transform, so it is recoverable from pairs at ANY vehicle pose and divides
    back out to the C3's answer."""
    E_true = T_of(_rot_zyx(0.02, math.radians(18.0), -0.03),   # an 18 deg tilt error
                  [0.12, -0.03, 0.05])                          # and 13 cm of lever arm
    align = BodyAlign(pair_tol_s=0.05, min_pairs=5)
    assert not align.ready and align.E() is None
    _simulate_pair_stream(align, E_true, 12)
    assert align.ready and align.n_pairs == 12
    E = align.E()
    assert np.allclose(E, E_true, atol=1e-9), E - E_true
    # a fresh RGB pose at a pose never seen during learning
    T_true = T_of(_rot_zyx(0.1, -0.05, 2.0), [1.0, -0.7, -0.5])
    T_rgb = T_true @ E_true
    assert np.allclose(align.correct(T_rgb), T_true, atol=1e-9)
    assert abs(align.offset_mm() - 1e3 * np.linalg.norm([0.12, -0.03, 0.05])) < 1e-6
    assert align.spread_mm() < 1e-6


def test_body_align_measures_the_real_tilt():
    """T_bc_true = E · T_bc_assumed, so the mount's actual angle falls out
    of the alignment — the only tilt feedback this vehicle has."""
    R_assumed, t_assumed = _second_cam_extrinsic(0.0)      # nav thinks LEVEL
    R_true, t_true = _second_cam_extrinsic(25.0)           # it is 25 deg down
    E_true = T_of(R_true, t_true) @ T_inv(T_of(R_assumed, t_assumed))
    align = BodyAlign(min_pairs=3)
    _simulate_pair_stream(align, E_true, 6)
    assert abs(align.measured_tilt_deg(R_assumed, t_assumed) - 25.0) < 1e-6
    R_m, t_m = align.measured_extrinsic(R_assumed, t_assumed)
    assert np.allclose(R_m, R_true, atol=1e-9) and np.allclose(t_m, t_true, atol=1e-9)


def test_body_align_pairs_only_within_tolerance_and_averages_noise():
    E_true = T_of(np.eye(3), [0.10, 0.0, 0.0])
    align = BodyAlign(pair_tol_s=0.05, min_pairs=10, window=200)
    align.note_main(1.0, np.eye(4))
    assert not align.note_second(1.2, E_true), "0.2 s apart must not pair"
    assert align.n_pairs == 0
    _simulate_pair_stream(align, E_true, 150, noise_m=0.01, seed=3)
    E = align.E()
    assert np.allclose(E[:3, 3], [0.10, 0.0, 0.0], atol=3e-3), E[:3, 3]
    assert 5.0 < align.spread_mm() < 25.0, align.spread_mm()


def test_body_align_reset_forgets_everything():
    align = BodyAlign(min_pairs=2)
    _simulate_pair_stream(align, T_of(np.eye(3), [0.1, 0, 0]), 4)
    assert align.ready
    align.reset("tilt changed")
    assert not align.ready and align.E() is None and align.n_pairs == 0
    assert align.reset_reason == "tilt changed"
    assert np.allclose(align.correct(np.eye(4)), np.eye(4))


# =============================================================================
# FallbackArbiter
# =============================================================================
def test_arbiter_holds_while_main_is_live_and_until_aligned():
    arb = FallbackArbiter(after_s=0.2, allow_unaligned=False)
    # nothing from the main yet: silent, but not aligned -> hold
    assert arb.decide(0.0, aligned=False) == "hold: not aligned yet"
    assert arb.n_forwarded == 0
    arb.note_main_fix(1.0)
    assert arb.decide(1.1, aligned=True) == "hold: C3 live"
    assert arb.decide(1.19, aligned=True) == "hold: C3 live"
    assert arb.decide(1.25, aligned=True) == "forward"
    assert arb.covering(1.3)
    # the main comes back: its fix is younger, the RGB is held again
    arb.note_main_fix(1.3)
    assert arb.decide(1.35, aligned=True) == "hold: C3 live"
    assert not arb.covering(1.35)


def test_arbiter_can_be_told_to_forward_unaligned():
    arb = FallbackArbiter(after_s=0.2, allow_unaligned=True)
    assert arb.decide(5.0, aligned=False) == "forward"


# =============================================================================
# TiltTracker
# =============================================================================
def test_tilt_tracker_dead_reckons_clamps_and_grows_its_doubt():
    tr = TiltTracker(rate_deg_s=30.0, lo_deg=-45.0, hi_deg=45.0)
    assert tr.deg is None and tr.src == "unknown" and tr.epoch == 0
    tr.center(0.0)                       # LEVEL: an anchor
    assert tr.deg == 0.0 and tr.src == "level" and tr.epoch == 1
    tr.drive(-1.0, 1.0)                  # hold DOWN for 1 s
    assert tr.epoch == 2 and tr.moving
    tr.update(2.0)
    assert abs(tr.deg - (-30.0)) < 1e-9 and tr.src == "dead-reckoned"
    assert tr.unc_deg > 1.0, "doubt must grow with dead-reckoned travel"
    tr.drive(0.0, 2.0)
    assert not tr.moving and tr.epoch == 2, "stopping is not a new motion"
    tr.drive(-1.0, 3.0)
    tr.update(10.0)                      # would be -240: clamps at the stop
    assert tr.deg == -45.0
    tr.drive(0.0, 10.0)


def test_tilt_tracker_measurement_anchors_without_bumping_the_epoch():
    """A measurement (MOUNT_STATUS, the tag localizer) replaces the estimate
    and its doubt, but must NOT count as commanded motion — the localizer's
    own measurement would otherwise reset the alignment that made it."""
    tr = TiltTracker()
    tr.center(0.0)
    tr.drive(+1.0, 0.0)
    tr.update(0.5)
    tr.drive(0.0, 0.5)
    epoch = tr.epoch
    tr.set_measured(-22.5, "tags", 1.0, unc_deg=0.7)
    assert tr.deg == -22.5 and tr.src == "measured:tags" and tr.unc_deg == 0.7
    assert tr.epoch == epoch
    # ...and dead reckoning continues FROM the measured anchor
    tr.drive(-1.0, 2.0)
    tr.update(2.5)
    assert abs(tr.deg - (-22.5 - 15.0)) < 1e-9
    assert tr.epoch == epoch + 1


def test_tilt_tracker_manual_set_is_an_anchor_and_a_command():
    tr = TiltTracker()
    tr.set_manual(-30.0, 0.0)
    assert tr.deg == -30.0 and tr.src == "set" and tr.epoch == 1
    assert "set" in tr.note() and "-30.0" in tr.note()
    tr.set_manual(90.0, 1.0)
    assert tr.deg == 45.0, "a typed value is still clamped to the mount"


def test_tilt_tracker_unknown_start_assumes_level_with_full_range_doubt():
    tr = TiltTracker(lo_deg=-45.0, hi_deg=45.0)
    assert "unknown" in tr.note()
    tr.drive(-1.0, 0.0)
    tr.update(1.0)
    assert abs(tr.deg - (-30.0)) < 1e-9 and tr.unc_deg >= 90.0


# =============================================================================
if __name__ == "__main__":
    names = [n for n in list(globals()) if n.startswith("test_")]
    failed = 0
    for n in names:
        try:
            globals()[n]()
            print(f"  ok    {n}")
        except Exception as e:                            # noqa: BLE001
            failed += 1
            print(f"  FAIL  {n}: {e!r}")
    print(f"\n{len(names) - failed}/{len(names)} passed")
    sys.exit(1 if failed else 0)
