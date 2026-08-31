#!/usr/bin/env python3
"""
gantry_map_pose.py — where the gantry camera is, in the ROV's own world frame.

The gantry's downward fisheye sees the surveyed AprilTag floor. That floor map
(``config/tag_map.yaml``, anchor tag 25) is the SAME map the ROV localizes
against, and it was built by this very rig. So the camera can report its
position directly in the ROV's coordinates — no gantry-to-map calibration in
the loop at all.

WHY THIS IS NOT THE OBVIOUS PATH
--------------------------------
The obvious path is the ENCODER: read the gantry's XYZ and rotate it into the
map frame with ``R_gantry_to_slam`` / ``gantry_to_slam_scale`` from
``config/fisheye_calibration.yaml``. That path is broken three ways, and all
three are in KNOWN_ISSUES:

  * ``R_gantry_to_slam`` was fit against a per-run anchor (tag 67), not tag 25
    — 179.81 deg out (derived from tag 67's pose in config/tag_map.yaml).
  * there is NO translation calibration; ``gantry_anchor_offset_mm`` is absent
    from the YAML, so every plot falls back to first-sample zeroing.
  * ``gantry_to_slam_scale`` = 1.0357 is anisotropic (per-axis 0.9891 / 1.0465 /
    1.0440, recomputed from data/20260528/20260528_215858_recording), i.e. it is
    absorbing a real error, not a metric conversion.

All three relate the ENCODER frame to the map. Measuring with the CAMERA skips
every one of them.

WHY IT BORROWS THE ROV'S CODE
-----------------------------
Detection and PnP come from ``rov_gui/control/tagnav.py`` — the module the ROV
uses for its own fixes. Not a reimplementation: the same corner ordering, the
same tag object points, the same duplicate-id handling, the same map file. Frame
identity is then a property of the construction rather than something to verify
after the fact. ``tagnav`` imports only stdlib + numpy (cv2 and pupil_apriltags
are lazy), so it loads in `robust` next to PyQt5 without dragging in torch.

``datum="map"`` is the default and makes ``TagNav._apply_datum`` a no-op, so
``NavSolution.p_ned`` IS the anchor-25 map position. ``R_ned_map`` is identity
because the floor map is already NED-like (+z INTO the tag face, floor tags
print-up => +z down); see rov_gui/control/geometry.py.

WHAT IT DELIBERATELY DOES NOT IMPORT
------------------------------------
``fisheye_gantry_tagslam.build_fisheye_undistort_maps`` is the same six lines of
cv2 as ``_build_maps`` below, but importing it pulls in tagslam_core and
therefore gtsam. The panel already pays that cost lazily for the Experiment tab;
a live readout should not pay it at all.

WHAT IT REPORTS, AND WHAT IT DOES NOT
-------------------------------------
``p_ned`` is the CAMERA's optical centre. ``T_gantry_camera`` in the calibration
is identity — an UNMEASURED assumption that the lens sits exactly at the gantry
tool point. That does not affect this readout (it never touches the encoder),
but any use of it as "where the payload is" needs the lens-to-payload lever arm
measured first.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rov_gui.control.tagnav import (                            # noqa: E402
    Detection, NavSolution, TagDetector, TagMap, TagNav,
)

# The rectification balance every recorder in this repo uses (0 = tight crop,
# no black border). Kept a constant, not an argument: a readout rectified with a
# different balance than the recordings would not be comparable to them.
UNDISTORT_BALANCE = 0.0

DEFAULT_CALIB_PATH = "config/fisheye_calibration.yaml"
DEFAULT_TAG_MAP_PATH = "config/tag_map.yaml"
# The physical tag edge, confirmed for the survey that built the map
# (data/20260528/20260528_202333_survey/user_actions.csv logs tag_size_m 0.17)
# and matching config/hw_nav.yaml on the ROV side. NOT tagslam_core's
# DEFAULT_TAG_SIZE_M, which is 0.085 — exactly half, and a silent 2x scale error.
DEFAULT_TAG_SIZE_M = 0.170


class CalibrationMismatch(RuntimeError):
    """The frame size does not match the size the intrinsics were calibrated at.

    Raised rather than rescaled: K would be wrong by the ratio, and a position
    readout that is quietly 1.5x off is worse than no readout.
    """


@dataclass(frozen=True)
class MapPose:
    """One frame's answer, flattened for a UI that should not import tagnav."""

    x_m: float
    y_m: float
    z_m: float
    n_tags: int
    tag_ids: tuple
    reproj_rms_px: float
    ambiguous: bool
    note: str

    @classmethod
    def from_solution(cls, sol: NavSolution) -> "MapPose":
        p = np.asarray(sol.p_ned, float).ravel()
        return cls(float(p[0]), float(p[1]), float(p[2]),
                   int(sol.n_tags), tuple(sol.tag_ids),
                   float(sol.reproj_rms_px), bool(sol.ambiguous),
                   str(sol.note or ""))


def load_fisheye_KD(path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """(K, D, (width, height)) from a fisheye calibration YAML.

    Reads the same three keys as ``fisheye_gantry_tagslam.load_fisheye_calibration``
    and none of the gantry-to-slam ones, which this module has no use for.
    """
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: not a calibration YAML")
    for key in ("K", "D", "image_size"):
        if key not in raw:
            raise ValueError(f"{path}: missing '{key}'")
    K = np.asarray(raw["K"], dtype=np.float64).reshape(3, 3)
    D = np.asarray(raw["D"], dtype=np.float64).reshape(-1, 1)
    if D.shape[0] != 4:
        raise ValueError(f"{path}: D must be 4 fisheye coefficients, got {D.shape[0]}")
    w, h = (int(v) for v in raw["image_size"])
    return K, D, (w, h)


class MapPoseEstimator:
    """Frames in, anchor-25 map position out.

    Not thread-safe by itself — one instance per worker thread. It holds the
    remap LUTs (two 1280x720 arrays) and the detector, both of which are
    expensive to build and cheap to keep.
    """

    def __init__(self,
                 calib_path=DEFAULT_CALIB_PATH,
                 tag_map_path=DEFAULT_TAG_MAP_PATH,
                 tag_size_m: float = DEFAULT_TAG_SIZE_M,
                 *,
                 max_reproj_px: float = 3.0,
                 min_tags: int = 1,
                 min_decision_margin: float = 20.0,
                 quad_decimate: float = 1.0):
        self.calib_path = Path(calib_path)
        self.tag_map_path = Path(tag_map_path)
        self.tag_size_m = float(tag_size_m)

        self.K, self.D, self.calib_wh = load_fisheye_KD(self.calib_path)
        self.tag_map = TagMap.load(self.tag_map_path)
        self.detector = TagDetector(family="tag36h11",
                                    quad_decimate=float(quad_decimate),
                                    min_decision_margin=float(min_decision_margin))
        self.nav = TagNav(
            tag_map=self.tag_map,
            tag_size_m=self.tag_size_m,
            # "body" IS the camera: identity extrinsic, so NavSolution.p_ned is
            # the lens position. The gantry has no FRD body and no autopilot
            # attitude, so rp_hint stays None and the gravity/tilt gate — which
            # only fires on the single-tag path WITH a hint — never applies.
            R_frd_cam=np.eye(3),
            t_frd_cam=np.zeros(3),
            # The floor map is already NED-like (geometry.py: R_ned_map = I).
            R_ned_map=np.eye(3),
            max_reproj_px=float(max_reproj_px),
            min_tags=int(min_tags),
            datum="map",                       # no re-zero: absolute map coords
        )

        self._maps = None                      # (map1, map2) or None if rectified
        self._K_use = None                     # intrinsics matching the frames
        self._frame_wh = None
        self._rectified_source = False

    # ------------------------------------------------------------------ setup
    def configure(self, frame_wh: tuple[int, int], *, rectified: bool = False) -> None:
        """Prepare for frames of this size.

        ``rectified=False`` (live camera): frames are raw fisheye, and their size
        MUST equal the calibrated size or K is wrong — CalibrationMismatch.

        ``rectified=True`` (recorded frames): the recorder already undistorted
        them and wrote them scaled down (tagslam_core's frame saver), so no remap
        is applied and the rectified K is scaled to the frame instead. The scale
        is uniform in each axis, which is exactly how the intrinsics scale.
        """
        import cv2

        w, h = int(frame_wh[0]), int(frame_wh[1])
        cw, ch = self.calib_wh

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, self.D, (cw, ch), np.eye(3), balance=UNDISTORT_BALANCE)

        if rectified:
            sx, sy = float(w) / float(cw), float(h) / float(ch)
            K_use = new_K.copy()
            K_use[0, 0] *= sx
            K_use[0, 2] *= sx
            K_use[1, 1] *= sy
            K_use[1, 2] *= sy
            self._maps = None
        else:
            if (w, h) != (cw, ch):
                raise CalibrationMismatch(
                    f"frames are {w}x{h} but {self.calib_path.name} was "
                    f"calibrated at {cw}x{ch} — K would be wrong by "
                    f"{w / cw:.3f}x. Set the camera to {cw}x{ch}, or "
                    f"recalibrate at {w}x{h}.")
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                self.K, self.D, np.eye(3), new_K, (cw, ch), cv2.CV_16SC2)
            self._maps = (map1, map2)
            K_use = new_K

        self._K_use = np.asarray(K_use, dtype=np.float64)
        self._frame_wh = (w, h)
        self._rectified_source = bool(rectified)

    @property
    def configured(self) -> bool:
        return self._K_use is not None

    # ---------------------------------------------------------------- per frame
    def rectify(self, frame: np.ndarray) -> np.ndarray:
        """Undistorted gray. Auto-configures on the first frame it sees."""
        import cv2

        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        if not self.configured:
            h, w = gray.shape[:2]
            self.configure((w, h))
        if self._maps is not None:
            gray = cv2.remap(gray, self._maps[0], self._maps[1],
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)
        return gray

    def detect(self, gray_rect: np.ndarray) -> list[Detection]:
        return self.detector.detect(gray_rect)

    def solve(self, detections: list[Detection]) -> NavSolution | None:
        # dist=None: the frame is rectified, so there is no distortion left.
        return self.nav.solve(detections, self._K_use, None)

    def process(self, frame: np.ndarray) -> NavSolution | None:
        """rectify -> detect -> PnP. None when the map could not be resolved;
        ``self.last_reject`` then says why."""
        return self.solve(self.detect(self.rectify(frame)))

    # ------------------------------------------------------------------ status
    @property
    def last_reject(self) -> str:
        return getattr(self.nav, "last_reject", "")

    def describe(self) -> str:
        wh = f"{self._frame_wh[0]}x{self._frame_wh[1]}" if self._frame_wh else "?"
        src = "rectified" if self._rectified_source else "raw"
        return (f"tag_map={self.tag_map_path.name} "
                f"anchor={self.tag_map.anchor_id} tags={len(self.tag_map)} "
                f"size={self.tag_size_m:.3f}m frame={wh}({src}) "
                f"detector={self.detector.backend}")
