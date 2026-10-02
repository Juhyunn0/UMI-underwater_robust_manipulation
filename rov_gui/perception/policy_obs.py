"""
policy_obs.py — the depth observation the diffusion policy was trained on, rebuilt live.

Why this exists
---------------
The policy (spec v1 §0) was trained on ONE depth recipe, frozen into the
training store's ``.zattrs`` and ``umi_handheld/build_dp_depth_zarr.py``:

    FoundationStereo depth (mm) on the land rectified-left grid
      -> WarpStage to configs/target_camera_underwater.yaml
         (RAW CAM_B model, 1280x800 scaled to 640x400, nearest)
      -> centre crop to 400x400  (y0 = 0, x0 = 120)
      -> cv2.resize 224x224, INTER_NEAREST
      -> normalise_depth(z_near=0.20, z_far=3.00)   (inverse depth, near = 1)
      -> u8 = clip(v*255 + 0.5); valid = 255 where mm > 0; channels (u8, valid, u8)

Everything the policy sees is that 224x224x3 uint8 image, so deployment has
to produce it from whatever depth the station has — and the station's depth
never arrives on the target grid. It arrives on one of three:

    rect_left      FoundationStereo's ``depth_native`` — the HOST rectified-left
                   grid (StereoRig R1/P1 at mono size), the parity path.
    color_aligned  the device's StereoDepth, aligned to the COLOUR camera
                   (CAM_A, 63.7 deg HFOV) — a different camera, a narrower
                   FOV, and the on-device matcher. Bench experiments only.
    identity       already on the 640x400 target grid (the demo backend).

This module owns the grid adapters (``set_grid``) and the recipe (``build``),
and nothing else: numpy + cv2 (through ``rov_gui.qt.import_cv2``, lazily) +
``umi_handheld.camera_model.CameraModel`` for the yaml. No torch, no Qt
widgets, no camera.

The target rays are VERBATIM the training call
----------------------------------------------
``WarpStage._build`` (umi_handheld/warp.py) computes, for every target pixel,
``CameraModel.unproject(px, W, H)`` on the target model and then asks the
SOURCE model where that ray lands. This file computes the same
``tgt.unproject(px, 640, 400)`` on the same yaml — literally the same method
on the same pixel list — and then replaces "source model" by the live grid's
geometry. Rebuilding the rays from K/D by hand would be a second
implementation that could drift from the training one; there is exactly one.

rect_left: inverse rectification with a per-pixel z correction
--------------------------------------------------------------
A rectified pixel is ``P1 · R1 · ray`` for a raw-CAM_B ray (``R1`` maps raw
-> rectified; verified against ``cv2.undistortPoints(px, K, D, R=R1, P=P1)``,
which is the exact inverse of ``initUndistortRectifyMap`` — test 3). The map
is then a plain ``cv2.remap(INTER_NEAREST)``, so an invalid (0) sample stays
invalid and no two distances are ever averaged (the same rule as
``host_depth.resize_depth_nearest``).

The VALUE needs a correction the map does not give: ``depth_native`` stores
z in the RECTIFIED frame while the target grid's z is in the RAW frame, and
the two frames differ by R1. A rectified point ``P_r = z_rect · (x_n, y_n, 1)``
(P1-normalised coordinates of the SAMPLED rectified pixel) is
``P_raw = R1^T · P_r``, whose z is::

    z_raw = z_rect · (R1[0,2]·x_n + R1[1,2]·y_n + R1[2,2])

R1 is a fraction of a degree on this camera, so the factor is within ~1 % of
1 — about 1 u8 LSB at >= 1 m [유도: dv/dz = 255/(4.667 z²) per metre], and
implemented rather than argued away. ``(x_n, y_n)`` are obtained by remapping
the normalised-coordinate grids through the SAME nearest map as the depth, so
the factor belongs to the pixel that was actually sampled, whatever rounding
``cv2.remap`` applied.

NO fill on rect_left: the training frames were not filled either, and the
rectified map is dense wherever it is valid.

The live model must be the training model
-----------------------------------------
The rays come from the yaml; the depth comes from a rig built off the live
EEPROM. If the camera were recalibrated the two would silently disagree, so
``set_grid`` asserts the live ``K_B`` (at mono size) against the yaml
(|ΔK| < 1e-2 px) and the live distortion against the yaml's 8 coefficients
(|ΔD| < 1e-4, coefficients 9..14 exactly 0) and raises ``GridError`` naming a
recalibration otherwise. Both fingerprints go into ``describe()``.

Coverage
--------
At alpha 0 the rectified image is cropped to fully-valid pixels, i.e. it is
NARROWER than the raw CAM_B image, so raw-grid pixels near the edge sample
outside it and come back 0. The fraction of the 224 obs that has a source
pixel at all — computed from the map, after the centre crop and the nearest
resize, before any depth is seen — is ``coverage``; below ``min_coverage``
(0.985 [유도: 0.985 sits at the 0.06 % tail of the training obs validity
(per-frame min 0.9777, p1 1.000, 99.94 % of frames ≥ 0.985) [측정:
rov_gui/tools/dp_policy_out/training_validity_20260902.txt]]) the grid is
marked unusable and the controller's refusal names it (PolicyStatus.grid_ok /
grid_why — verify 2026-09-02).
alpha 0.5 restores the edges (test 2 prints both numbers).

color_aligned: forward scatter, filled, counted separately
----------------------------------------------------------
The inverse of ``host_depth.warp_depth_to_color``: deproject the colour-grid
pixels through K_A/D_A (``cv2.undistortPoints``), move them to CAM_B with the
EEPROM extrinsics (mm), project through the yaml K_B/D_B with
``host_depth._project_with_distortion``, and scatter NEAREST with a z-buffer
(far first, near overwrites). The written value is z in CAM_B — the depth the
target grid means — not the colour-frame z. The scatter leaves resampling
gaps; ``host_depth.fill_scatter_gaps(fill_iters)`` closes them without
inventing a distance, and the stats report ``valid_measured`` (landed) and
``filled`` (dilated in) separately so a frame's honesty is visible. The
colour camera's FOV sits inside CAM_B's, so the border of the obs is
invalid by construction; ``coverage`` reports how much of the 224 obs the
colour FOV reaches (evaluated at a 1 m reference range [유도: the ~mm-scale
CAM_A->CAM_B baseline moves the footprint by <1 px at that range]).

view shift: a TEMPORARY virtual viewpoint (2026-09-30, default OFF)
-------------------------------------------------------------------
``view_shift_cam_m`` (None = off, and then ``build`` is byte-for-byte what it
was) re-renders the target-grid depth from a camera displaced by that vector,
expressed in the REAL camera's frame, same orientation: every valid pixel is
deprojected along its training ray (``z * ray / ray_z``), moved by ``-t``,
projected back through the yaml K_B/D_B and scattered NEAREST with a z-buffer
(near wins, each sample written to the 2x2 pixels around it so a magnified
near surface stays closed); what is left empty — the disocclusions — takes
the FARTHEST of the nearest measured pixels along the four image axes, i.e.
the background (``_fill_farthest_4dir``). It is the ``color_aligned`` scatter
with a translation instead of an extrinsic.

Why it exists: on the ROV the jaw sits ~5 cm further along the gripper axis
from the camera than on the handheld the peg demos were recorded with (peg
root 0.254 m vs 0.208 m along the axis, same tilt, same height [측정:
data/20260930/0930_183320_observe/diag/rig_geometry.json]), so the held peg
lands ~43 px higher in the obs and the peg checkpoint reads the scene as
"released -> retreat". Moving the viewpoint 5 cm forward is PREDICTED to turn
dz<0 from 0.90 to 0.31 of plans, 0.18 with the proprio width at the demos'
held value [예측: .../diag/validate_builder_shift.json, this code on the
2026-09-30 obs rebuilt from the UI recording]; it has not flown. A single view cannot see behind the peg, so what the fill writes
there is the floor's distance, not a measurement. RECORD BOUNDARY:
``describe()['view_shift']`` (run meta ``policy.worker.obs`` and
``policy_obs/meta.json``) and the controller's per-arm ``OBS VIEW SHIFT``
event. The per-frame ``view_landed`` / ``view_filled`` counts ride on the
stats ``build`` returns and are NOT written to plans.jsonl; ``depth_valid``
there is the validity AFTER the fill, so it reads 1.0 on a shifted run.

Provenance of the numbers above: crop origin and sizes are the training
script's own arithmetic on a 640x400 frame; z_near/z_far are the training
store's ``.zattrs['obs']``; the tolerances are spec v2 A16.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..state import POLICY_GRID_WHY_IDLE

REPO = Path(__file__).resolve().parents[2]

#: The three producers' names for the projection their millimetres live on.
GRID_RECT_LEFT = "rect_left"
GRID_COLOR_ALIGNED = "color_aligned"
GRID_IDENTITY = "identity"
GRID_KINDS = (GRID_RECT_LEFT, GRID_COLOR_ALIGNED, GRID_IDENTITY)

#: Model-mismatch tolerances [spec v2 A16].
K_TOL_PX = 1e-2
D_TOL = 1e-4
#: Number of distortion coefficients the yaml carries (c3_camera/device.py:399
#: truncates to 8; the live handler returns 14, the tail of which must be 0).
D_YAML_N = 8

#: View shift (module docstring). The cap is where a single view stops being
#: re-renderable [예측].
VIEW_SHIFT_MAX_M = 0.15
#: A sample closer than this to the VIRTUAL lens is dropped rather than drawn:
#: it would be a saturated speck (z_near is 200 mm) made of something inside
#: the stereo matcher's minimum range [예측].
VIEW_MIN_Z_MM = 50.0

#: The training store's recipe, spelled out for describe() (build_dp_depth_zarr.py).
RECIPE = ("depth (uint16 mm) on the producer's grid",
          "grid adapter -> raw CAM_B 640x400 target grid (nearest, no averaging)",
          "centre crop to square (y0=0, x0=120 for 640x400)",
          "cv2.resize to out_res x out_res, INTER_NEAREST",
          "normalise_depth(z_near, z_far) -> [0,1], near = 1",
          "u8 = clip(v*255 + 0.5); channel 1 = 255 where mm > 0; channels (u8, valid, u8)")


class GridError(RuntimeError):
    """The grid cannot be used for the policy; the message names why."""


# =============================================================================
# normalise_depth — a copy, parity-tested against umi_handheld.build_zarr
# =============================================================================
def normalise_depth(depth_mm: np.ndarray, z_near: float, z_far: float) -> np.ndarray:
    """uint16 millimetres -> normalised inverse depth in [0, 1], near = large.

    A verbatim copy of ``umi_handheld.build_zarr.normalise_depth`` (the
    training function) rather than an import: that module imports ``zarr``
    at module scope, and in the station env the repo-root ``zarr/`` data
    directory shadows the package. ``test_policy_obs.py`` asserts the two
    agree bit-for-bit wherever the original is importable.
    """
    z = depth_mm.astype(np.float32) / 1000.0
    valid = z > 0
    inv_near, inv_far = 1.0 / z_near, 1.0 / z_far
    v = np.zeros_like(z, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        v[valid] = (1.0 / z[valid] - inv_far) / (inv_near - inv_far)
    return np.clip(v, 0.0, 1.0)


# =============================================================================
# Grid descriptors
# =============================================================================
def _round_list(v, nd: int = 9):
    a = np.asarray(v, dtype=np.float64)
    return np.round(a, nd).tolist()


def grid_fingerprint(kind: str, **parts) -> str:
    """A short stable hash of a grid's geometry (arrays rounded to 1e-9).

    Used by the mailbox and the builder to tell "the same grid again" (a
    no-op) from "a different grid" (a reconnect that changed calibration,
    which must be visible, never silently rebuilt into).
    """
    h = hashlib.sha1()
    h.update(str(kind).encode())
    for key in sorted(parts):
        val = parts[key]
        if val is None:
            s = "None"
        elif isinstance(val, (np.ndarray, list, tuple)):
            s = json.dumps(_round_list(val))
        else:
            s = repr(val)
        h.update(f"|{key}={s}".encode())
    return h.hexdigest()[:16]


def _fill_farthest_4dir(depth_mm: np.ndarray) -> np.ndarray:
    """Fill the empty (0) pixels of a re-rendered depth map with the FARTHEST
    of the nearest measured pixels to the left, right, above and below.

    What is empty after the 2x2 scatter is a disocclusion: ground the real
    camera could not see because something near stood in front of it. The
    truth there is the background, so the farthest of the four wins — the
    same rule as ``host_depth.fill_scatter_gaps``. That function is not used
    here because it grows inwards one pixel per pass from EVERY side: a hole
    pixel next to the near object has only near neighbours on its first pass,
    takes the object's distance, and the object comes out with a pointed tail
    reaching into the band it vacated (a 14-row triangle on the test box).
    Looking along the four axes sees past the hole to both of its banks at
    once. No measured pixel is touched and no two distances are averaged; a
    pixel with nothing measured in any of the four directions stays 0.
    """
    valid = depth_mm > 0
    if valid.all():
        return depth_mm
    H, W = depth_mm.shape
    best = np.zeros_like(depth_mm)
    for axis in (1, 0):
        n = W if axis == 1 else H
        ar = np.arange(n).reshape((1, n) if axis == 1 else (n, 1))
        # nearest measured index at or before / at or after each pixel
        before = np.maximum.accumulate(np.where(valid, ar, -1), axis=axis)
        after = np.flip(np.minimum.accumulate(
            np.flip(np.where(valid, ar, n), axis=axis), axis=axis), axis=axis)
        for idx, none in ((before, -1), (after, n)):
            got = np.take_along_axis(depth_mm, np.clip(idx, 0, n - 1), axis=axis)
            best = np.maximum(best, np.where(idx == none, 0, got))
    return np.where(valid, depth_mm, best)


@dataclass(frozen=True)
class RectLeftGrid:
    """FoundationStereo's ``depth_native``: the host rectified-left grid.

    ``R1``/``P1`` are ``StereoRig.R1``/``StereoRig.P1`` (cv2.stereoRectify at
    ``mono_size`` with ``alpha``). ``K_live``/``D_live`` are the raw CAM_B
    intrinsics the rig was built from, at ``mono_size``; when the producer
    cannot supply them the model check is SKIPPED and ``describe()`` says so
    — it is never silently passed.
    """

    R1: np.ndarray
    P1: np.ndarray
    mono_size: tuple
    alpha: float
    K_live: np.ndarray | None = None
    D_live: np.ndarray | None = None
    fingerprint: str = ""
    #: Optional: the rig's ``provenance`` dict (recorded, never computed from).
    provenance: dict | None = None

    def __post_init__(self):
        object.__setattr__(self, "R1", np.asarray(self.R1, dtype=np.float64).reshape(3, 3))
        P1 = np.asarray(self.P1, dtype=np.float64)
        if P1.shape == (3, 3):
            P1 = np.concatenate([P1, np.zeros((3, 1))], axis=1)
        object.__setattr__(self, "P1", P1.reshape(3, 4))
        object.__setattr__(self, "mono_size",
                           (int(self.mono_size[0]), int(self.mono_size[1])))
        object.__setattr__(self, "alpha", float(self.alpha))
        if self.K_live is not None:
            object.__setattr__(self, "K_live",
                               np.asarray(self.K_live, dtype=np.float64).reshape(3, 3))
        if self.D_live is not None:
            object.__setattr__(self, "D_live",
                               np.asarray(self.D_live, dtype=np.float64).ravel())
        if not self.fingerprint:
            object.__setattr__(self, "fingerprint", grid_fingerprint(
                GRID_RECT_LEFT, R1=self.R1, P1=self.P1, mono_size=self.mono_size,
                alpha=self.alpha, K_live=self.K_live, D_live=self.D_live))

    @property
    def kind(self) -> str:
        return GRID_RECT_LEFT


@dataclass(frozen=True)
class ColorAlignedGrid:
    """The device's StereoDepth aligned to CAM_A, at the DEPTH array size.

    ``K_a``/``D_a`` at ``size_a`` (= the depth array's size, never the colour
    stream's), ``R_ab``/``t_ab_mm`` from ``calib.getCameraExtrinsics(CAM_A,
    CAM_B)`` through ``host_depth._split_extrinsics_cm`` (cm -> mm):
    ``P_b = R_ab @ P_a + t_ab_mm``.
    """

    K_a: np.ndarray
    D_a: np.ndarray
    size_a: tuple
    R_ab: np.ndarray
    t_ab_mm: np.ndarray
    fingerprint: str = ""

    def __post_init__(self):
        object.__setattr__(self, "K_a", np.asarray(self.K_a, dtype=np.float64).reshape(3, 3))
        object.__setattr__(self, "D_a", np.asarray(self.D_a, dtype=np.float64).ravel())
        object.__setattr__(self, "size_a", (int(self.size_a[0]), int(self.size_a[1])))
        object.__setattr__(self, "R_ab", np.asarray(self.R_ab, dtype=np.float64).reshape(3, 3))
        object.__setattr__(self, "t_ab_mm", np.asarray(self.t_ab_mm, dtype=np.float64).ravel())
        if self.t_ab_mm.shape != (3,):
            raise ValueError(f"t_ab_mm must have 3 entries, got {self.t_ab_mm.shape}")
        if not self.fingerprint:
            object.__setattr__(self, "fingerprint", grid_fingerprint(
                GRID_COLOR_ALIGNED, K_a=self.K_a, D_a=self.D_a, size_a=self.size_a,
                R_ab=self.R_ab, t_ab_mm=self.t_ab_mm))

    @property
    def kind(self) -> str:
        return GRID_COLOR_ALIGNED


@dataclass(frozen=True)
class IdentityGrid:
    """Depth already on the target grid (the demo's synthetic 640x400)."""

    size: tuple
    fingerprint: str = ""

    def __post_init__(self):
        object.__setattr__(self, "size", (int(self.size[0]), int(self.size[1])))
        if not self.fingerprint:
            object.__setattr__(self, "fingerprint",
                               grid_fingerprint(GRID_IDENTITY, size=self.size))

    @property
    def kind(self) -> str:
        return GRID_IDENTITY


def grid_kind_of(grid) -> str:
    if isinstance(grid, RectLeftGrid):
        return GRID_RECT_LEFT
    if isinstance(grid, ColorAlignedGrid):
        return GRID_COLOR_ALIGNED
    if isinstance(grid, IdentityGrid):
        return GRID_IDENTITY
    raise GridError(f"unknown grid descriptor {type(grid).__name__}; expected one of "
                    f"RectLeftGrid / ColorAlignedGrid / IdentityGrid")


# =============================================================================
# The builder
# =============================================================================
def _cv2():
    """cv2 through ``rov_gui.qt.import_cv2`` (it must not repoint the station's
    Qt plugin path); on a box with no Qt binding at all — an offline
    reprocessing env such as the one that holds the training ``zarr`` — a
    plain import, because this module needs numpy + cv2 and nothing of Qt."""
    try:
        from rov_gui.qt import import_cv2
    except ImportError:
        import cv2                                  # noqa: PLC0415
        return cv2
    return import_cv2()


class DepthObsBuilder:
    """Depth on a producer's grid -> the policy's (out_res, out_res, 3) uint8 obs.

    Construct once, ``set_grid`` once (idempotent for the same fingerprint),
    ``build`` per frame. ``usable``/``why`` are what the controller's refusal
    reads; ``describe()`` is what the run record keeps.
    """

    def __init__(self, target_model="configs/target_camera_underwater.yaml",
                 z_near: float = 0.20, z_far: float = 3.00, out_res: int = 224,
                 warp_size=(640, 400), min_coverage: float = 0.985,
                 fill_iters: int = 2, view_shift_cam_m=None):
        if not (0.0 < float(z_near) < float(z_far)):
            raise ValueError(f"need 0 < z_near < z_far, got {z_near}, {z_far}")
        cv2 = _cv2()                          # before camera_model imports cv2 itself
        from umi_handheld.camera_model import CameraModel
        self._cv2 = cv2
        self.z_near, self.z_far = float(z_near), float(z_far)
        self.out_res = int(out_res)
        self.W, self.H = (int(v) for v in warp_size)
        self.min_coverage = float(min_coverage)
        self.fill_iters = int(fill_iters)
        self.model = CameraModel.load(target_model)
        self.K_b = self.model.K(self.W, self.H)
        self.D_b = self.model.dist.ravel().copy()

        # The training rays, verbatim (umi_handheld/warp.py WarpStage._build).
        u, v = np.meshgrid(np.arange(self.W, dtype=np.float64),
                           np.arange(self.H, dtype=np.float64))
        self._px = np.stack([u.ravel(), v.ravel()], 1)
        self._rays = self.model.unproject(self._px, self.W, self.H)   # (N,3) unit
        self.rays_behind = int(np.count_nonzero(self._rays[:, 2] <= 0))

        # The crop the training script computes from the frame shape.
        s = min(self.H, self.W)
        self.crop_size = s
        self.crop_y0, self.crop_x0 = (self.H - s) // 2, (self.W - s) // 2

        # view shift (module docstring): None = off, the recipe is untouched.
        self._view_t_mm = None
        self._dirn = None
        if view_shift_cam_m is not None:
            t = np.asarray(view_shift_cam_m, dtype=np.float64).reshape(-1)
            if t.shape != (3,) or not np.all(np.isfinite(t)):
                raise ValueError(f"view_shift_cam_m must be 3 finite metres, "
                                 f"got {view_shift_cam_m!r}")
            if float(np.linalg.norm(t)) > VIEW_SHIFT_MAX_M + 1e-9:
                raise ValueError(f"view_shift_cam_m |t| {np.linalg.norm(t):.3f} m "
                                 f"> {VIEW_SHIFT_MAX_M} m: a single depth view "
                                 f"cannot be re-rendered that far")
            if self.rays_behind:
                raise ValueError("view shift needs every target ray in front of "
                                 f"the camera; {self.rays_behind} are not")
            if np.any(t != 0.0):
                self._view_t_mm = t * 1000.0
                self._dirn = self._rays / self._rays[:, 2:3]      # (N,3), z == 1
                from c3_camera import host_depth
                self._host_depth = host_depth

        self._grid = None
        self._kind = ""
        self._coverage = 0.0
        self._usable = False
        self._why = POLICY_GRID_WHY_IDLE
        self._model_check: dict = {"status": "not run"}
        self._n_builds = 0
        self._n_set_grid = 0
        # rect_left state
        self._map_x = self._map_y = None
        self._zfac = None
        self._inside = None
        # color_aligned state
        self._xn_a = self._yn_a = None
        if self._view_t_mm is None:
            self._host_depth = None

    # ---------------------------------------------------------------- props
    @property
    def grid_kind(self) -> str:
        return self._kind

    @property
    def grid(self):
        return self._grid

    @property
    def coverage(self) -> float:
        return float(self._coverage)

    @property
    def usable(self) -> bool:
        return bool(self._usable)

    @property
    def why(self) -> str:
        return self._why

    # ------------------------------------------------------------- set_grid
    def set_grid(self, grid) -> None:
        """Build the adapter for ``grid``. Raises ``GridError`` naming the cause.

        Idempotent for an identical fingerprint. A DIFFERENT grid after one is
        set is refused (the mailbox refuses it too): a camera that reconnected
        with another calibration must be seen, not rebuilt into. ``reset()``
        clears explicitly.
        """
        kind = grid_kind_of(grid)
        if self._grid is not None:
            if kind == self._kind and grid.fingerprint == self._grid.fingerprint:
                return
            raise GridError(
                f"grid changed: {self._kind}:{self._grid.fingerprint} is set, "
                f"{kind}:{grid.fingerprint} offered. A different rectification or "
                f"calibration mid-run is not adopted silently; call reset() first.")
        self._n_set_grid += 1
        self._grid, self._kind = grid, kind
        self._usable, self._why, self._coverage = False, "", 0.0
        try:
            if kind == GRID_RECT_LEFT:
                self._build_rect_left(grid)
            elif kind == GRID_COLOR_ALIGNED:
                self._build_color_aligned(grid)
            else:
                self._build_identity(grid)
        except GridError as e:
            self._why = str(e)
            raise
        if self._coverage < self.min_coverage:
            self._usable = False
            self._why = (f"obs coverage {100.0 * self._coverage:.2f}% < "
                         f"{100.0 * self.min_coverage:.2f}% on grid {kind}"
                         + (": pass --fstereo-alpha 0.5" if kind == GRID_RECT_LEFT
                            else ""))
            raise GridError(self._why)
        self._usable, self._why = True, ""

    def reset(self) -> None:
        self._grid, self._kind = None, ""
        self._usable, self._why, self._coverage = False, POLICY_GRID_WHY_IDLE, 0.0
        self._map_x = self._map_y = self._zfac = self._inside = None
        self._xn_a = self._yn_a = None
        self._model_check = {"status": "not run"}

    # ---- rect_left ---------------------------------------------------------
    def _check_model(self, grid: RectLeftGrid) -> None:
        """Live K_B/D_B vs the yaml the rays came from [spec v2 A16 tolerances]."""
        w, h = grid.mono_size
        K_yaml = self.model.K(w, h)
        D_yaml = self.D_b
        rec = {"status": "", "K_yaml": _round_list(K_yaml, 6),
               "D_yaml": _round_list(D_yaml, 9),
               "K_live": None if grid.K_live is None else _round_list(grid.K_live, 6),
               "D_live": None if grid.D_live is None else _round_list(grid.D_live, 9),
               "K_tol_px": K_TOL_PX, "D_tol": D_TOL,
               "yaml": self.model.rel, "yaml_fingerprint": grid_fingerprint(
                   "yaml", K=K_yaml, D=D_yaml),
               "live_fingerprint": None if grid.K_live is None else grid_fingerprint(
                   "live", K=grid.K_live, D=grid.D_live)}
        if grid.K_live is None or grid.D_live is None:
            rec["status"] = ("SKIPPED: producer supplied no live K/D; the rays are the "
                             "yaml's and the rig's agreement with it is UNCHECKED")
            self._model_check = rec
            return
        dK = np.abs(grid.K_live - K_yaml)
        dK_max = float(max(dK[0, 0], dK[1, 1], dK[0, 2], dK[1, 2]))
        D_live = grid.D_live
        n = min(D_live.size, D_yaml.size, D_YAML_N)
        dD_max = float(np.abs(D_live[:n] - D_yaml[:n]).max()) if n else float("inf")
        tail_ok = bool(np.all(np.abs(D_live[D_YAML_N:]) == 0.0)) if D_live.size > D_YAML_N else True
        n_ok = D_live.size >= D_YAML_N and D_yaml.size >= D_YAML_N
        rec.update({"dK_max_px": dK_max, "dD_max": dD_max, "D_tail_zero": tail_ok})
        bad = []
        if not n_ok:
            bad.append(f"distortion has {D_live.size} live / {D_yaml.size} yaml coefficients "
                       f"(need >= {D_YAML_N})")
        if dK_max >= K_TOL_PX:
            bad.append(f"|dK| max {dK_max:.4f} px >= {K_TOL_PX} px")
        if dD_max >= D_TOL:
            bad.append(f"|dD[:8]| max {dD_max:.2e} >= {D_TOL:.0e}")
        if not tail_ok:
            bad.append("live distortion coefficients 9..14 are not all 0")
        if bad:
            rec["status"] = "MISMATCH: " + "; ".join(bad)
            self._model_check = rec
            raise GridError(
                f"live CAM_B intrinsics disagree with the training model {self.model.rel} "
                f"({'; '.join(bad)}). The camera was recalibrated (or a different unit "
                f"is connected); the policy's depth would be warped through the wrong "
                f"optics. Re-derive the target model from this unit's EEPROM "
                f"(calib/fov_audit.py) and retrain or re-warp before flying the policy.")
        rec["status"] = "OK"
        self._model_check = rec

    def _build_rect_left(self, grid: RectLeftGrid) -> None:
        cv2 = self._cv2
        self._check_model(grid)
        SW, SH = grid.mono_size
        R1, P1 = grid.R1, grid.P1
        # rectified pixel = P1 · R1 · ray (== cv2.undistortPoints(px, K, D, R=R1, P=P1))
        r = self._rays @ R1.T
        q = r @ P1[:, :3].T
        ok = q[:, 2] > 1e-9
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.where(ok, q[:, 0] / q[:, 2], -1e4)
            v = np.where(ok, q[:, 1] / q[:, 2], -1e4)
        self._map_x = u.reshape(self.H, self.W).astype(np.float32)
        self._map_y = v.reshape(self.H, self.W).astype(np.float32)
        ui, vi = np.rint(self._map_x), np.rint(self._map_y)
        self._inside = ((ui >= 0) & (ui <= SW - 1) & (vi >= 0) & (vi <= SH - 1)
                        & ok.reshape(self.H, self.W))
        # z correction, keyed to the pixel remap actually samples.
        fx1, fy1, cx1, cy1 = P1[0, 0], P1[1, 1], P1[0, 2], P1[1, 2]
        xn = ((np.arange(SW, dtype=np.float64) - cx1) / fx1)[None, :].repeat(SH, 0)
        yn = ((np.arange(SH, dtype=np.float64) - cy1) / fy1)[:, None].repeat(SW, 1)
        xn_s = cv2.remap(xn, self._map_x, self._map_y, cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        yn_s = cv2.remap(yn, self._map_x, self._map_y, cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        self._zfac = R1[0, 2] * xn_s + R1[1, 2] * yn_s + R1[2, 2]
        self._coverage = self._obs_coverage(self._inside)

    # ---- color_aligned -----------------------------------------------------
    def _build_color_aligned(self, grid: ColorAlignedGrid) -> None:
        cv2 = self._cv2
        from c3_camera import host_depth
        self._host_depth = host_depth
        aw, ah = grid.size_a
        u, v = np.meshgrid(np.arange(aw, dtype=np.float64), np.arange(ah, dtype=np.float64))
        p = np.stack([u.ravel(), v.ravel()], 1).reshape(-1, 1, 2)
        n = cv2.undistortPoints(p, grid.K_a, grid.D_a).reshape(-1, 2)
        self._xn_a = n[:, 0].reshape(ah, aw)
        self._yn_a = n[:, 1].reshape(ah, aw)
        # Coverage: which target pixels can a colour-grid measurement reach at all?
        # Evaluated at a 1 m reference range through the inverse transform.
        P_b = self._rays * (1000.0 / self._rays[:, 2:3])
        P_a = (P_b - grid.t_ab_mm[None, :]) @ grid.R_ab      # R_ab^T applied
        front = P_a[:, 2] > 0
        ua = np.full(P_a.shape[0], -1.0)
        va = np.full(P_a.shape[0], -1.0)
        if front.any():
            uu, vv = host_depth._project_with_distortion(P_a[front], grid.K_a, grid.D_a)
            ua[front], va[front] = uu, vv
        inside = ((np.rint(ua) >= 0) & (np.rint(ua) <= aw - 1)
                  & (np.rint(va) >= 0) & (np.rint(va) <= ah - 1) & front)
        self._inside = inside.reshape(self.H, self.W)
        self._coverage = self._obs_coverage(self._inside)

    # ---- identity ----------------------------------------------------------
    def _build_identity(self, grid: IdentityGrid) -> None:
        if grid.size != (self.W, self.H):
            raise GridError(f"identity grid is {grid.size[0]}x{grid.size[1]} but the "
                            f"target grid is {self.W}x{self.H}; an identity grid means "
                            f"'already on the target grid', so the sizes must match")
        self._inside = np.ones((self.H, self.W), dtype=bool)
        self._coverage = 1.0

    # ---- shared ------------------------------------------------------------
    def _obs_coverage(self, inside: np.ndarray) -> float:
        """Fraction of the 224 obs pixels whose source exists — crop + nearest
        resize of the map's inside mask, i.e. exactly the obs pixels that can
        ever be non-zero."""
        cv2 = self._cv2
        s, y0, x0 = self.crop_size, self.crop_y0, self.crop_x0
        m = inside[y0:y0 + s, x0:x0 + s].astype(np.uint8)
        m = cv2.resize(m, (self.out_res, self.out_res), interpolation=cv2.INTER_NEAREST)
        return float(m.mean())

    # ----------------------------------------------------------------- warp
    def warp(self, depth_mm: np.ndarray) -> tuple[np.ndarray, dict]:
        """Producer-grid depth -> target-grid depth (uint16 mm), plus the
        per-frame counts. ``build`` calls this and then applies the recipe."""
        if depth_mm.dtype != np.uint16:
            raise TypeError(f"depth must be uint16 millimetres, got {depth_mm.dtype}")
        if self._grid is None:
            raise GridError("no grid set; call set_grid(...) first")
        kind = self._kind
        if kind == GRID_RECT_LEFT:
            if self._map_x is None:
                raise GridError(f"rect_left maps were not built: {self._why}")
            self._expect_shape(depth_mm, self._grid.mono_size)
            cv2 = self._cv2
            z = cv2.remap(depth_mm, self._map_x, self._map_y, cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            out = np.rint(z.astype(np.float64) * self._zfac)
            out = np.clip(out, 0, 65535).astype(np.uint16)
            out[z == 0] = 0                    # belt and braces: 0 stays 0
            measured = int(np.count_nonzero(out))
            return out, {"valid_measured": measured, "valid_total": measured, "filled": 0}
        if kind == GRID_COLOR_ALIGNED:
            self._expect_shape(depth_mm, self._grid.size_a)
            return self._scatter(depth_mm)
        self._expect_shape(depth_mm, self._grid.size)
        measured = int(np.count_nonzero(depth_mm))
        return depth_mm, {"valid_measured": measured, "valid_total": measured, "filled": 0}

    def _expect_shape(self, depth_mm, size_wh) -> None:
        w, h = size_wh
        if depth_mm.shape != (h, w):
            raise GridError(f"depth is {depth_mm.shape[1]}x{depth_mm.shape[0]} but the "
                            f"{self._kind} grid was declared at {w}x{h}")

    def _scatter(self, depth_mm: np.ndarray) -> tuple[np.ndarray, dict]:
        g: ColorAlignedGrid = self._grid
        hd = self._host_depth
        out = np.zeros((self.H, self.W), dtype=np.uint16)
        vs, us = np.nonzero(depth_mm)
        stats = {"valid_measured": 0, "valid_total": 0, "filled": 0}
        if vs.size == 0:
            return out, stats
        z = depth_mm[vs, us].astype(np.float64)
        P_a = np.stack([self._xn_a[vs, us] * z, self._yn_a[vs, us] * z, z], axis=1)
        P_b = P_a @ g.R_ab.T + g.t_ab_mm[None, :]
        zb = P_b[:, 2]
        front = zb > 0.0
        if not front.all():
            if not front.any():
                return out, stats
            P_b, zb = P_b[front], zb[front]
        u_b, v_b = hd._project_with_distortion(P_b, self.K_b, self.D_b)
        ui = np.rint(u_b).astype(np.int32)
        vi = np.rint(v_b).astype(np.int32)
        inside = (ui >= 0) & (ui < self.W) & (vi >= 0) & (vi < self.H)
        if not inside.all():
            if not inside.any():
                return out, stats
            ui, vi, zb = ui[inside], vi[inside], zb[inside]
        mm = np.clip(np.rint(zb), 1, 65535).astype(np.uint16)   # z in CAM_B, never 0
        order = np.argsort(-zb, kind="stable")                  # far first, near wins
        out[vi[order], ui[order]] = mm[order]
        measured = int(np.count_nonzero(out))
        filled = hd.fill_scatter_gaps(out, self.fill_iters)
        total = int(np.count_nonzero(filled))
        stats.update(valid_measured=measured, valid_total=total, filled=total - measured)
        return filled, stats

    # ----------------------------------------------------------- view shift
    @property
    def view_shift_cam_m(self):
        """The virtual camera's position in the real camera frame [m], or None."""
        return None if self._view_t_mm is None else (self._view_t_mm / 1000.0)

    def _shift_view(self, tgt_mm: np.ndarray) -> tuple[np.ndarray, dict]:
        """Target-grid depth (uint16 mm, z in raw CAM_B) seen from the shifted
        viewpoint, on the same grid. 0 stays "no measurement" going in; coming
        out, a pixel INSIDE THE CENTRE CROP is 0 only where neither a sample
        nor the fill reached (outside the crop the holes are left as they
        are — ``build`` discards those columns). ``view_landed`` /
        ``view_filled`` count crop pixels.

        Each sample is written to the 2x2 pixels around its projection, not to
        the nearest one. Moving towards a surface magnifies it (z / (z - t_z),
        1.13 at the jaw for 5 cm), so nearest-pixel samples of a NEAR surface
        land more than a pixel apart while the far samples behind it land
        densely in between — and the z-buffer cannot win a pixel no near
        sample was written to. The surface came out with the background
        showing through it as a grid (safety review 2026-09-30). The 2x2
        footprint closes any magnification below 2 at the cost of growing a
        near silhouette by at most one pixel of the 640x400 grid.
        """
        hd = self._host_depth
        H, W = tgt_mm.shape
        st = {"view_landed": 0, "view_filled": 0}
        flat = np.flatnonzero(tgt_mm)
        if flat.size == 0:
            return np.zeros_like(tgt_mm), st
        z = tgt_mm.ravel()[flat].astype(np.float64)
        P = self._dirn[flat] * z[:, None] - self._view_t_mm[None, :]
        P = P[P[:, 2] > VIEW_MIN_Z_MM]
        if P.shape[0] == 0:
            return np.zeros_like(tgt_mm), st
        u, v = hd._project_with_distortion(P, self.K_b, self.D_b)
        u0 = np.floor(u).astype(np.int64)
        v0 = np.floor(v).astype(np.int64)
        zs = P[:, 2].astype(np.float32)            # mm; exact to 0.004 mm at 65 m
        zb = np.full(H * W, np.inf, dtype=np.float32)
        for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):
            ui, vi = u0 + du, v0 + dv
            inside = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            np.minimum.at(zb, vi[inside] * W + ui[inside], zs[inside])   # near wins
        landed = np.isfinite(zb)
        out = np.zeros(H * W, dtype=np.uint16)
        out[landed] = np.clip(np.rint(zb[landed]), 1, 65535).astype(np.uint16)
        out = out.reshape(H, W)
        # Fill only what ``build`` keeps (the centre crop): the fill is the
        # slowest step and the columns outside the crop are thrown away on the
        # next line of the recipe. Outside the crop an empty pixel stays 0.
        y0, x0, c = self.crop_y0, self.crop_x0, self.crop_size
        if (H, W) == (self.H, self.W):
            crop = out[y0:y0 + c, x0:x0 + c]
            n_before = int(np.count_nonzero(crop))
            out[y0:y0 + c, x0:x0 + c] = _fill_farthest_4dir(crop)
            n_after = int(np.count_nonzero(out[y0:y0 + c, x0:x0 + c]))
        else:                                       # not the builder's grid: fill it all
            n_before = int(np.count_nonzero(out))
            out = _fill_farthest_4dir(out)
            n_after = int(np.count_nonzero(out))
        st.update(view_landed=n_before, view_filled=n_after - n_before)
        return out, st

    # ---------------------------------------------------------------- build
    def build(self, depth_mm: np.ndarray) -> tuple[np.ndarray, dict]:
        """The obs the policy consumes, and its stats.

        The recipe below is the per-frame loop of
        ``umi_handheld/build_dp_depth_zarr.py`` (lines ~258-270) line for line
        — crop from the frame shape, nearest resize, normalise, u8, validity —
        so a synthetic frame pushed through both comes out bit-identical
        (test 1). Only the first line (the grid adapter) is this module's.
        """
        w, st = self.warp(depth_mm)
        vst = None
        if self._view_t_mm is not None:
            w, vst = self._shift_view(w)
        cv2 = self._cv2
        R = self.out_res
        s = min(w.shape[:2])
        y0, x0 = (w.shape[0] - s) // 2, (w.shape[1] - s) // 2
        w = w[y0:y0 + s, x0:x0 + s]
        w = cv2.resize(w, (R, R), interpolation=cv2.INTER_NEAREST)
        v = normalise_depth(w, self.z_near, self.z_far)
        valid = (w > 0).astype(np.uint8) * 255
        u8 = np.clip(v * 255.0 + 0.5, 0, 255).astype(np.uint8)
        obs = np.stack([u8, valid, u8], axis=-1)
        self._n_builds += 1
        n = float(w.size)
        stats = {"valid_measured": int(st["valid_measured"]),
                 "valid_total": int(st["valid_total"]),
                 "filled": int(st["filled"]),
                 "coverage": self.coverage,
                 "obs_valid": float(np.count_nonzero(valid)) / n}
        if vst is not None:
            stats.update(vst)
        return obs, stats

    # ------------------------------------------------------------- describe
    def describe(self) -> dict:
        d = {
            "recipe": list(RECIPE),
            "target_model": self.model.provenance(),
            "target_rays": "umi_handheld.camera_model.CameraModel.unproject(px, W, H) "
                           "(verbatim umi_handheld/warp.py WarpStage._build)",
            "warp_size": [self.W, self.H],
            "crop": {"y0": self.crop_y0, "x0": self.crop_x0, "size": self.crop_size},
            "out_res": self.out_res,
            "z_near_m": self.z_near, "z_far_m": self.z_far,
            "rays_behind_camera": self.rays_behind,
            "grid_kind": self._kind,
            "grid_fingerprint": self._grid.fingerprint if self._grid is not None else "",
            "coverage": self.coverage,
            "min_coverage": self.min_coverage,
            "usable": self.usable,
            "why": self.why,
            "fill_iters": self.fill_iters if self._kind == GRID_COLOR_ALIGNED else 0,
            "n_builds": self._n_builds,
            "n_set_grid": self._n_set_grid,
        }
        if self._view_t_mm is not None:
            d["view_shift"] = {
                "t_cam_m": _round_list(self._view_t_mm / 1000.0, 6),
                "frame": "virtual camera centre in the REAL camera frame "
                         "(x right, y down, z optical), same orientation",
                "method": "deproject target-grid z along the training rays, "
                          "translate, project through the yaml K_B/D_B, 2x2 "
                          "scatter with z-buffer (near wins), then empty "
                          "pixels take the farthest of the nearest measured "
                          "pixels left/right/above/below",
                "min_z_mm": VIEW_MIN_Z_MM,
                "note": "TEMPORARY (2026-09-30). The obs is NOT the training "
                        "recipe's any more: filled pixels behind near objects "
                        "carry the background's distance, not a measurement. "
                        "Do not pool with unshifted runs.",
            }
        g = self._grid
        if isinstance(g, RectLeftGrid):
            d["grid"] = {"mono_size": list(g.mono_size), "alpha": g.alpha,
                         "R1": _round_list(g.R1), "P1": _round_list(g.P1),
                         "provenance": g.provenance}
            d["model_check"] = dict(self._model_check)
            if self._zfac is not None and self._inside is not None and self._inside.any():
                zf = self._zfac[self._inside]
                d["z_correction_factor"] = {"min": float(zf.min()), "max": float(zf.max())}
            d["fill"] = "none (rect_left is never filled)"
        elif isinstance(g, ColorAlignedGrid):
            d["grid"] = {"size_a": list(g.size_a), "K_a": _round_list(g.K_a, 6),
                         "D_a": _round_list(g.D_a), "R_ab": _round_list(g.R_ab),
                         "t_ab_mm": _round_list(g.t_ab_mm),
                         "K_b": _round_list(self.K_b, 6), "D_b": _round_list(self.D_b)}
            d["coverage_reference_range_mm"] = 1000.0
            d["depth_value"] = "z in CAM_B after the extrinsic transform (not the CAM_A z)"
        elif isinstance(g, IdentityGrid):
            d["grid"] = {"size": list(g.size)}
        return d


__all__ = ["GridError", "RectLeftGrid", "ColorAlignedGrid", "IdentityGrid",
           "DepthObsBuilder", "normalise_depth", "grid_fingerprint", "grid_kind_of",
           "GRID_RECT_LEFT", "GRID_COLOR_ALIGNED", "GRID_IDENTITY", "GRID_KINDS",
           "RECIPE", "K_TOL_PX", "D_TOL", "VIEW_SHIFT_MAX_M", "VIEW_MIN_Z_MM"]
