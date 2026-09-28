#!/usr/bin/env python3
"""
test_policy_obs.py — the policy's depth observation builder, offline.

    ~/miniforge3/envs/rovgui-pose/bin/python rov_gui/tests/test_policy_obs.py
    ~/miniforge3/envs/robust/bin/python      rov_gui/tests/test_policy_obs.py

No camera, no GPU, no torch. What is pinned:

* identity grid: the builder's recipe equals the training script's per-frame
  code (umi_handheld/build_dp_depth_zarr.py) BIT FOR BIT on a synthetic
  frame, with the training ``normalise_depth`` where it is importable (an env
  with the real ``zarr`` package — ``umi2``; neither ``robust`` nor
  ``rovgui-pose`` has one, and the repo-root ``zarr/`` data directory shadows
  it under pytest, so that one test skips there and says so);
* rect_left: a fronto-parallel plane at 1 m rendered into a ROTATED rectified
  frame comes back as 1000 mm on the target grid — centre AND edge — because
  of the per-pixel z correction, whose size at the edge is the predicted
  factor; the map equals cv2.undistortPoints(R=R1, P=P1) on the C3 model;
  coverage at alpha 0 vs 0.5 is printed and asserted; a recalibrated camera
  (K/D off the yaml) is refused by name;
* color_aligned: a plane scatters into a band with an invalid border, gaps
  are filled and COUNTED apart from measured pixels, coverage < 1;
* set_grid is idempotent for one fingerprint and refuses another;
* an invalid (0) pixel stays 0 with valid = 0 through every grid.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Probe the REAL zarr before the repo root leads sys.path (the repo has a
# `zarr/` data directory that otherwise shadows the package).
try:
    import zarr as _zarr_probe                      # noqa: F401
except Exception:                                   # noqa: BLE001
    _zarr_probe = None

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np                                  # noqa: E402

try:
    from rov_gui.qt import import_cv2               # noqa: E402
    cv2 = import_cv2()
except ImportError:                                 # a Qt-less zarr env (fstereo)
    import cv2                                      # noqa: E402

try:
    from c3_camera.host_depth import StereoRig      # noqa: E402  (pulls depthai)
    _HOST_DEPTH_ERR = ""
except Exception as _e:                             # noqa: BLE001
    StereoRig = None
    _HOST_DEPTH_ERR = f"{type(_e).__name__}: {_e}"

from rov_gui.perception.policy_obs import (         # noqa: E402
    GRID_COLOR_ALIGNED, GRID_IDENTITY, GRID_RECT_LEFT, ColorAlignedGrid,
    DepthObsBuilder, GridError, IdentityGrid, RectLeftGrid, normalise_depth)

TARGET_YAML = "configs/target_camera_underwater.yaml"
W, H = 640, 400


class _Skip(Exception):
    """Raised by _skip under the plain runner so a test body stops there."""


def _skip(reason: str) -> None:
    """Skip under pytest; print and abort the test under the plain runner."""
    print(f"  skip: {reason}")
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(reason)
    raise _Skip(reason)


def _need_host_depth() -> None:
    """The rig fixtures come from c3_camera.host_depth, whose package imports
    depthai; an env without it (the zarr env) skips those tests and says so."""
    if StereoRig is None:
        _skip(f"c3_camera.host_depth unavailable here ({_HOST_DEPTH_ERR}); the rig "
              f"fixtures need it — run under rovgui-pose or robust")


# --------------------------------------------------------------- fixtures
def _c3_pair():
    """The C3's left/right models from the yaml, at 640x400."""
    import yaml
    b = DepthObsBuilder(TARGET_YAML)
    cfg = yaml.safe_load(open(b.model.path, encoding="utf-8"))
    r = cfg["stereo"]["right"]
    K_l, D_l = b.model.K(W, H), b.model.dist.ravel().copy()
    K_r = np.array([[r["intrinsics"]["fx"] / 2, 0, r["intrinsics"]["cx"] / 2],
                    [0, r["intrinsics"]["fy"] / 2, r["intrinsics"]["cy"] / 2],
                    [0, 0, 1.0]])
    D_r = np.asarray(r["distortion"], dtype=np.float64)
    return K_l, D_l, K_r, D_r


def _rig(alpha: float, deg=(0.3, 1.0, 0.2)):
    """A synthetic C3-like rig: yaml K/D, a ~1 deg left->right rotation,
    T = -75 mm — enough R1 to make the z correction measurable."""
    _need_host_depth()
    K_l, D_l, K_r, D_r = _c3_pair()
    R = cv2.Rodrigues(np.deg2rad(np.asarray(deg, dtype=np.float64)))[0]
    return StereoRig.from_stereo_pair(
        mono_size=(W, H), K_left=K_l, dist_left=D_l, K_right=K_r, dist_right=D_r,
        R_left_to_right=R, T_left_to_right_mm=np.array([-75.0, 0.0, 0.0]),
        alpha=alpha)


def _rect_grid(rig, with_live=True) -> RectLeftGrid:
    K_l, D_l, _, _ = _c3_pair()
    return RectLeftGrid(R1=rig.R1, P1=rig.P1, mono_size=rig.mono_size, alpha=rig.alpha,
                        K_live=K_l if with_live else None,
                        D_live=np.r_[D_l, np.zeros(6)] if with_live else None)


def _plane_in_rectified(rig, z_raw_mm: float = 1000.0) -> np.ndarray:
    """A fronto-parallel plane z = const in the RAW CAM_B frame, rendered as
    the depth the rectified-left frame would store for it.

    Rectified ray (x_n, y_n, 1) is raw ray R1^T (x_n, y_n, 1), whose z is
    c = R1[0,2] x_n + R1[1,2] y_n + R1[2,2]; the plane is hit at scale
    z_raw / c, and the RECTIFIED z is that scale itself."""
    R1, P1 = rig.R1, rig.P1
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    xn = (u - P1[0, 2]) / P1[0, 0]
    yn = (v - P1[1, 2]) / P1[1, 1]
    c = R1[0, 2] * xn + R1[1, 2] * yn + R1[2, 2]
    return np.rint(z_raw_mm / c).astype(np.uint16), c


def _synthetic_depth(seed: int = 3) -> np.ndarray:
    """A textured uint16 depth with holes, in the range the recipe spans."""
    rng = np.random.default_rng(seed)
    d = rng.uniform(150.0, 3500.0, size=(H, W))
    d[rng.random((H, W)) < 0.03] = 0.0                # scattered invalid pixels
    d[300:340, 100:180] = 0.0                          # a hole block
    return np.rint(d).astype(np.uint16)


def _reference_obs(depth_u16: np.ndarray, norm, z_near=0.20, z_far=3.00, R=224):
    """VERBATIM per-frame code of umi_handheld/build_dp_depth_zarr.py (lines
    ~258-270), stage = None (identity), including its float32 cast."""
    dm = depth_u16.astype(np.float32)
    w = dm
    s = min(w.shape[:2])
    y0, x0 = (w.shape[0] - s) // 2, (w.shape[1] - s) // 2
    w = w[y0:y0 + s, x0:x0 + s]
    w = cv2.resize(w, (R, R), interpolation=cv2.INTER_NEAREST)
    v = norm(w, z_near, z_far)
    valid = (w > 0).astype(np.uint8) * 255
    u8 = np.clip(v * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return np.stack([u8, valid, u8], axis=-1)


# =============================================================================
# 1. identity grid == the training script, bit for bit
# =============================================================================
def test_identity_parity_with_build_dp_depth_zarr_training_normaliser():
    """The training ``normalise_depth`` (umi_handheld.build_zarr) drives the
    reference; needs the REAL zarr package. Neither ``robust`` nor
    ``rovgui-pose`` has one installed (checked 2026-09-02: only umi / umi2 /
    umi_day* / fstereo / aquabot do), and with the repo root leading sys.path
    the repo's ``zarr/`` data directory shadows it anyway — so this runs
    under ``~/miniforge3/envs/umi2/bin/python`` and skips elsewhere, saying so."""
    if _zarr_probe is None or not hasattr(_zarr_probe, "open_group"):
        _skip("umi_handheld.build_zarr needs the real `zarr` package; this env has none "
              "(or the repo-root zarr/ directory shadows it under pytest) — run "
              "~/miniforge3/envs/umi2/bin/python rov_gui/tests/test_policy_obs.py")
        return
    from umi_handheld.build_zarr import normalise_depth as train_norm
    b = DepthObsBuilder(TARGET_YAML)
    b.set_grid(IdentityGrid(size=(W, H)))
    d = _synthetic_depth()
    obs, st = b.build(d)
    ref = _reference_obs(d, train_norm)
    assert obs.dtype == np.uint8 and obs.shape == (224, 224, 3)
    assert np.array_equal(obs, ref), "identity recipe differs from the training script"
    assert st["coverage"] == 1.0 and st["filled"] == 0
    print(f"  bit-for-bit vs build_dp_depth_zarr (training normalise_depth): OK, "
          f"obs_valid {st['obs_valid']:.4f}")


def test_identity_parity_with_inline_reference_and_local_normaliser():
    """The same parity with this module's ``normalise_depth`` copy — runs in
    every env, and pins the copy against the training one when importable."""
    b = DepthObsBuilder(TARGET_YAML)
    b.set_grid(IdentityGrid(size=(W, H)))
    d = _synthetic_depth(seed=7)
    obs, _ = b.build(d)
    assert np.array_equal(obs, _reference_obs(d, normalise_depth))
    # the channel layout the policy was trained on
    assert np.array_equal(obs[..., 0], obs[..., 2])
    assert set(np.unique(obs[..., 1]).tolist()) <= {0, 255}
    try:
        from umi_handheld.build_zarr import normalise_depth as train_norm
    except Exception as e:                           # noqa: BLE001
        _skip(f"normalise_depth parity vs umi_handheld.build_zarr: {type(e).__name__}: {e}")
        return
    for zn, zf in ((0.20, 3.00), (0.10, 5.00)):
        a = normalise_depth(d, zn, zf)
        r = train_norm(d, zn, zf)
        assert a.dtype == r.dtype == np.float32 and np.array_equal(a, r), (zn, zf)
    print("  normalise_depth copy == umi_handheld.build_zarr.normalise_depth")


# =============================================================================
# 2. rect_left: a 1 m plane comes back at 1000 mm, corrected at the edges
# =============================================================================
def test_rect_left_plane_and_z_correction_and_coverage():
    rig05 = _rig(0.5)
    b = DepthObsBuilder(TARGET_YAML)
    b.set_grid(_rect_grid(rig05))
    assert b.grid_kind == GRID_RECT_LEFT and b.usable, b.why
    d, c = _plane_in_rectified(rig05, 1000.0)
    w, st = b.warp(d)
    assert w.dtype == np.uint16 and w.shape == (H, W)
    centre = w[H // 2 - 5:H // 2 + 5, W // 2 - 5:W // 2 + 5].astype(float)
    assert np.abs(centre - 1000.0).max() <= 1.0, centre
    inside = w > 0
    err = np.abs(w[inside].astype(float) - 1000.0)
    assert err.max() <= 1.5, f"plane error max {err.max()} mm"   # rint in + rint out

    # The correction at an edge pixel is the predicted factor: without it the
    # sampled rectified value is 1000/c of the SAMPLED pixel, not 1000.
    ui = int(np.rint(b._map_x[H // 2, 3]))
    vi = int(np.rint(b._map_y[H // 2, 3]))
    assert 0 <= ui < W and 0 <= vi < H
    uncorrected = float(d[vi, ui])
    predicted_factor = float(c[vi, ui])
    assert abs(uncorrected * predicted_factor - float(w[H // 2, 3])) <= 1.0
    assert abs(predicted_factor - 1.0) > 2e-3, "the synthetic R1 is too small to test"
    assert abs(uncorrected - 1000.0) > 2.0, "edge pixel not affected by R1; no test"
    print(f"  edge pixel: rectified {uncorrected:.0f} mm x factor {predicted_factor:.5f} "
          f"-> {w[H // 2, 3]} mm (correction {uncorrected - w[H // 2, 3]:+.1f} mm)")

    # Coverage: alpha 0 crops the rectified image inside the raw FOV.
    b0 = DepthObsBuilder(TARGET_YAML)
    rig0 = _rig(0.0)
    try:
        b0.set_grid(_rect_grid(rig0))
        refused = ""
    except GridError as e:
        refused = str(e)
    cov0, cov05 = b0.coverage, b.coverage
    print(f"  coverage of the 224 obs: alpha 0.0 -> {100 * cov0:.2f}%   "
          f"alpha 0.5 -> {100 * cov05:.2f}%   (min_coverage {100 * b.min_coverage:.1f}%)")
    assert cov0 > 0.9, cov0
    assert cov05 > 0.995, cov05
    assert b0.usable == (cov0 >= b0.min_coverage)
    if not b0.usable:
        assert "coverage" in refused and "--fstereo-alpha 0.5" in refused, refused
        assert b0.why == refused
    # the obs after the recipe carries the same validity as the map says
    obs0, st0 = (b0.build(_plane_in_rectified(rig0)[0]))
    assert abs(st0["obs_valid"] - cov0) < 1e-9
    assert st0["filled"] == 0 and st0["valid_measured"] == st0["valid_total"]
    desc = b.describe()
    assert desc["fill"].startswith("none") and desc["model_check"]["status"] == "OK"
    assert desc["z_correction_factor"]["min"] < 1.0 < desc["z_correction_factor"]["max"]


# =============================================================================
# 3. the map is cv2.undistortPoints(R=R1, P=P1) on the C3 model
# =============================================================================
def test_rect_left_map_equals_undistortpoints_with_R_and_P():
    rig = _rig(0.5)
    b = DepthObsBuilder(TARGET_YAML)
    b.set_grid(_rect_grid(rig))
    K_l, D_l, _, _ = _c3_pair()
    ref = cv2.undistortPoints(b._px.reshape(-1, 1, 2), K_l, D_l,
                              R=rig.R1, P=rig.P1).reshape(H, W, 2)
    dx = np.abs(ref[..., 0] - b._map_x.astype(np.float64))
    dy = np.abs(ref[..., 1] - b._map_y.astype(np.float64))
    m = b._inside
    # the raw image's corners fall outside the rectified one even at alpha
    # 0.5 (they are outside the crop, so the obs coverage is still 100 %)
    assert m.mean() > 0.9, f"inside fraction of the full 640x400 map {m.mean():.4f}"
    assert dx[m].max() < 1e-3 and dy[m].max() < 1e-3, (dx[m].max(), dy[m].max())
    print(f"  max |map - undistortPoints(R,P)| = {max(dx[m].max(), dy[m].max()):.2e} px")


# =============================================================================
# 4. a recalibrated camera is refused by name
# =============================================================================
def test_rect_left_refuses_a_live_model_off_the_yaml():
    rig = _rig(0.5)
    K_l, D_l, _, _ = _c3_pair()
    ok = RectLeftGrid(R1=rig.R1, P1=rig.P1, mono_size=rig.mono_size, alpha=rig.alpha,
                      K_live=K_l, D_live=np.r_[D_l, np.zeros(6)])
    DepthObsBuilder(TARGET_YAML).set_grid(ok)             # 14 coefficients, tail 0: fine

    def refused(**over):
        kw = dict(R1=rig.R1, P1=rig.P1, mono_size=rig.mono_size, alpha=rig.alpha,
                  K_live=K_l, D_live=np.r_[D_l, np.zeros(6)])
        kw.update(over)
        b = DepthObsBuilder(TARGET_YAML)
        try:
            b.set_grid(RectLeftGrid(**kw))
        except GridError as e:
            assert "recalibrat" in str(e).lower(), str(e)
            assert not b.usable and b.why == str(e)
            assert b.describe()["model_check"]["status"].startswith("MISMATCH")
            return str(e)
        raise AssertionError(f"accepted {list(over)}")

    K_bad = K_l.copy(); K_bad[0, 2] += 0.5
    m1 = refused(K_live=K_bad)
    D_bad = np.r_[D_l, np.zeros(6)]; D_bad[0] += 1e-3
    assert "dD" in refused(D_live=D_bad)
    D_tail = np.r_[D_l, np.zeros(6)]; D_tail[9] = 1e-3
    assert "9..14" in refused(D_live=D_tail)
    assert "coefficients" in refused(D_live=D_l[:5])
    print(f"  refusals: '{m1[:60]}...' / dK / dD / tail / short")
    # a grid without live K/D is accepted but the check is recorded as skipped
    b = DepthObsBuilder(TARGET_YAML)
    b.set_grid(_rect_grid(rig, with_live=False))
    assert b.usable and b.describe()["model_check"]["status"].startswith("SKIPPED")


# =============================================================================
# 5. color_aligned: scatter, fill counted apart, coverage < 1
# =============================================================================
def _color_grid(size_a=(320, 200), hfov_deg=63.7) -> ColorAlignedGrid:
    aw, ah = size_a
    fx = (aw / 2.0) / np.tan(np.deg2rad(hfov_deg / 2.0))
    K_a = np.array([[fx, 0, (aw - 1) / 2.0], [0, fx, (ah - 1) / 2.0], [0, 0, 1.0]])
    D_a = np.array([-0.05, 0.01, 0.0, 0.0, 0.0])
    R_ab = cv2.Rodrigues(np.deg2rad(np.array([0.0, 0.5, 0.0])))[0]
    t_ab_mm = np.array([-37.5, 0.0, 2.0])          # CAM_A -> CAM_B, millimetres
    return ColorAlignedGrid(K_a=K_a, D_a=D_a, size_a=size_a, R_ab=R_ab, t_ab_mm=t_ab_mm)


def test_color_aligned_scatter_fill_and_coverage():
    _need_host_depth()                     # the scatter uses host_depth's projector/fill
    g = _color_grid()
    b = DepthObsBuilder(TARGET_YAML, min_coverage=0.0, fill_iters=2)
    b.set_grid(g)
    assert b.grid_kind == GRID_COLOR_ALIGNED
    assert 0.3 < b.coverage < 1.0, b.coverage
    aw, ah = g.size_a
    d = np.full((ah, aw), 1000, dtype=np.uint16)
    w, st = b.warp(d)
    assert st["filled"] > 0, "a coarse colour grid must leave scatter gaps"
    assert st["valid_total"] == st["valid_measured"] + st["filled"]
    assert st["valid_measured"] == int(np.count_nonzero(w)) - st["filled"]
    # the written value is z in CAM_B: the plane's z_a = 1000 plus t_z, with
    # the small rotation folded in (< 1 mm across the band)
    vals = w[w > 0].astype(float)
    assert abs(np.median(vals) - (1000.0 + g.t_ab_mm[2])) < 2.0, np.median(vals)
    # the colour FOV sits inside CAM_B's: the border is invalid
    assert w[0, 0] == 0 and w[H - 1, W - 1] == 0 and w[H // 2, 0] == 0
    obs, st2 = b.build(d)
    assert obs[0, 0, 1] == 0 and obs[0, 0, 0] == 0
    assert obs[112, 112, 1] == 255
    assert st2["coverage"] == b.coverage and st2["filled"] == st["filled"]
    assert abs(st2["obs_valid"] - b.coverage) < 0.02, (st2["obs_valid"], b.coverage)
    print(f"  colour {aw}x{ah} -> target: measured {st['valid_measured']}, "
          f"filled {st['filled']} ({100 * st['filled'] / st['valid_total']:.2f}% of valid), "
          f"coverage {100 * b.coverage:.2f}%, obs valid {100 * st2['obs_valid']:.2f}%")
    # a measured pixel is never overwritten by the fill (host_depth's rule)
    b0 = DepthObsBuilder(TARGET_YAML, min_coverage=0.0, fill_iters=0)
    b0.set_grid(g)
    w0, st0 = b0.warp(d)
    assert st0["filled"] == 0
    m = w0 > 0
    assert np.array_equal(w[m], w0[m])
    d2 = b.describe()
    assert d2["fill_iters"] == 2 and d2["grid"]["size_a"] == [aw, ah]


# =============================================================================
# 6. set_grid idempotence via fingerprint
# =============================================================================
def test_set_grid_idempotent_for_one_fingerprint_and_refuses_another():
    rig = _rig(0.5)
    b = DepthObsBuilder(TARGET_YAML)
    g1 = _rect_grid(rig)
    g1b = _rect_grid(rig)                      # rebuilt from the same numbers
    assert g1.fingerprint == g1b.fingerprint and len(g1.fingerprint) == 16
    b.set_grid(g1)
    mx = b._map_x
    b.set_grid(g1b)
    assert b._map_x is mx and b.describe()["n_set_grid"] == 1
    g2 = _rect_grid(_rig(0.5, deg=(0.3, 1.2, 0.2)))
    assert g2.fingerprint != g1.fingerprint
    try:
        b.set_grid(g2)
        raise AssertionError("a different grid was adopted silently")
    except GridError as e:
        assert "grid changed" in str(e) and g1.fingerprint in str(e)
    assert b.usable and b._map_x is mx          # the first grid is untouched
    try:
        b.set_grid(IdentityGrid(size=(W, H)))
        raise AssertionError("a different KIND was adopted silently")
    except GridError:
        pass
    b.reset()
    b.set_grid(IdentityGrid(size=(W, H)))
    assert b.grid_kind == GRID_IDENTITY and b.describe()["n_set_grid"] == 2
    # identity refuses a size that is not the target grid
    try:
        DepthObsBuilder(TARGET_YAML).set_grid(IdentityGrid(size=(640, 360)))
        raise AssertionError("identity at 640x360 accepted")
    except GridError as e:
        assert "640x360" in str(e)


# =============================================================================
# 7. invalid pixels stay invalid through every grid
# =============================================================================
def test_invalid_pixels_stay_zero_with_valid_zero_on_every_grid():
    # identity: an all-zero frame and a frame with a hole block
    b = DepthObsBuilder(TARGET_YAML)
    b.set_grid(IdentityGrid(size=(W, H)))
    obs, st = b.build(np.zeros((H, W), np.uint16))
    assert not obs.any() and st["obs_valid"] == 0.0
    d = np.full((H, W), 800, np.uint16)
    d[100:200, 200:300] = 0
    obs, _ = b.build(d)
    hole = obs[..., 1] == 0
    assert hole.any() and not obs[..., 0][hole].any() and not obs[..., 2][hole].any()
    assert (obs[..., 1][~hole] == 255).all()

    # rect_left: zeros in the rectified frame land as zeros wherever they are
    # sampled, and nowhere else (nearest, no blending, no fill)
    rig = _rig(0.5)
    br = DepthObsBuilder(TARGET_YAML)
    br.set_grid(_rect_grid(rig))
    dr, _ = _plane_in_rectified(rig, 1000.0)
    dr[150:250, 250:350] = 0
    w, st = br.warp(dr)
    mask_src = (dr > 0).astype(np.uint16)
    sampled = cv2.remap(mask_src, br._map_x, br._map_y, cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    assert np.array_equal(w > 0, sampled > 0)
    assert st["filled"] == 0
    obs, st2 = br.build(dr)
    z = obs[..., 1] == 0
    assert z.any() and not obs[..., 0][z].any()
    obs0, _ = br.build(np.zeros((H, W), np.uint16))
    assert not obs0.any()

    # color_aligned: an empty frame scatters nothing, and a half-empty one
    # leaves the other half empty except for the <= fill_iters px dilation rim
    g = _color_grid()
    bc = DepthObsBuilder(TARGET_YAML, min_coverage=0.0, fill_iters=2)
    bc.set_grid(g)
    obs0, st0 = bc.build(np.zeros((g.size_a[1], g.size_a[0]), np.uint16))
    assert not obs0.any() and st0["valid_total"] == 0
    dh = np.full((g.size_a[1], g.size_a[0]), 1000, np.uint16)
    dh[:, : g.size_a[0] // 2] = 0
    wh, _ = bc.warp(dh)
    cols = np.nonzero(wh.any(axis=0))[0]
    # where the first measured colour column (aw//2, centre row, 1 m) lands
    # on the target grid, through the same K_a/D_a -> R,t -> K_b/D_b chain
    from c3_camera.host_depth import _project_with_distortion
    n = cv2.undistortPoints(np.array([[[g.size_a[0] // 2, (g.size_a[1] - 1) / 2.0]]],
                                     dtype=np.float64), g.K_a, g.D_a).reshape(2)
    P_b = g.R_ab @ np.array([n[0] * 1000.0, n[1] * 1000.0, 1000.0]) + g.t_ab_mm
    u_b, _ = _project_with_distortion(P_b[None, :], bc.K_b, bc.D_b)
    edge = int(np.rint(u_b[0]))
    assert edge - 2 - 1 <= cols.min() <= edge + 1, (cols.min(), edge)   # fill_iters=2 rim
    print(f"  half-empty colour frame: first valid target column {cols.min()} "
          f"(predicted edge {edge}, fill rim <= 2 px)")
    obs, _ = bc.build(dh)
    z = obs[..., 1] == 0
    assert not obs[..., 0][z].any() and not obs[..., 2][z].any()
    print("  identity / rect_left / color_aligned: 0 -> (0, valid 0, 0)")


# =============================================================================
# describe() carries what the run record needs
# =============================================================================
def test_describe_is_json_serialisable_and_names_the_recipe():
    import json
    rig = _rig(0.5)
    b = DepthObsBuilder(TARGET_YAML)
    b.set_grid(_rect_grid(rig))
    b.build(_plane_in_rectified(rig)[0])
    d = b.describe()
    s = json.dumps(d)
    assert "normalise_depth" in s and "INTER_NEAREST" in s
    assert d["crop"] == {"y0": 0, "x0": 120, "size": 400}
    assert d["target_model"]["config"] == TARGET_YAML and d["target_model"]["verified"]
    assert d["grid_fingerprint"] == b.grid.fingerprint
    assert d["model_check"]["yaml_fingerprint"] and d["model_check"]["live_fingerprint"]
    assert d["n_builds"] == 1 and d["usable"] and d["why"] == ""
    assert d["z_near_m"] == 0.20 and d["z_far_m"] == 3.00 and d["out_res"] == 224
    # dtype and shape refusals never come back as a wrong image
    try:
        b.build(_plane_in_rectified(rig)[0].astype(np.float32))
        raise AssertionError("float32 depth accepted")
    except TypeError:
        pass
    try:
        b.build(np.zeros((360, 640), np.uint16))
        raise AssertionError("wrong-size depth accepted")
    except GridError:
        pass


# =============================================================================
# runner (same shape as test_replay.py — works with or without pytest)
# =============================================================================
def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except _Skip:
            print(f"  skip  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:                                   # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
