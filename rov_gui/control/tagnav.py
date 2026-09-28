#!/usr/bin/env python3
"""tagnav.py — AprilTag detection + PnP against a LOCKED tag map. No SLAM.

Why no GTSAM here: the pool map is already surveyed and pinned
(``config/tag_map.yaml``), and in that regime ``tagslam_core``'s iSAM2
backend adds essentially nothing over plain multi-tag solvePnP while its
graph grows without bound (measured 0.4 -> 36 ms/update over one survey —
data/20260528/20260528_202333_survey/slam_internals.csv). It also cannot be
imported into this process at all: gtsam pins numpy<2 and this station's env
is numpy 2 for torch. So this module reuses tagslam_core's *conventions* —
copied with line citations, never imported — on top of cv2 alone.

Conventions carried over (source: src/tagslam_core.py):
  * tag object points, :233 — corners (-h,+h),(+h,+h),(+h,-h),(-h,-h) in the
    tag's own frame, OpenCV handedness (+x right, +y DOWN, +z INTO the face).
    A tag lying print-up on the pool floor therefore has +z pointing DOWN,
    which is exactly why the floor map's world frame is already NED-like.
  * solvePnP returns camera_T_tag (:1576): a point in tag coordinates maps to
    camera coordinates. World poses come from composing with the map's
    world_T_tag, same as tagslam's PnP-only mode.
  * map format (:1529 load_tag_map): {anchor_tag_id, tags: {id:
    {position_m, quaternion_wxyz}}}, poses in the anchor-tag frame.

Refraction: NONE, deliberately. tagslam's refractive machinery models a
camera IN AIR above a flat interface; the C3 is submerged and its EEPROM
factory calibration is an UNDERWATER one (calib/FOV_AUDIT.md, vendor
confirmed), so port refraction is already inside the intrinsics. The known
consequence — in-AIR bench tests read ~1.33x long — is expected, not a bug.

DUPLICATED IDS (2026-08-14). The pool mat reuses 12 ids at a second physical
place, while the map holds ONE pose per id — so a detection of the "other"
copy contributes corners that belong somewhere else entirely, and the joint
solve splits the difference (measured 59-131 px against a 3 px gate, two
thirds of the rejections in data/20260813/0813_17*/nav_*). The mat is NOT
a repeated sheet, though: the operator's cell-by-cell survey shows the two
copies of an id share NO neighbour except their own pair partner, so the
tags detected ALONGSIDE a duplicated id say which copy it is. That is what
``duplicate_ids`` buys here — solve on the UNIQUE tags alone, then keep a
duplicated tag only if it reprojects where the map says it should
(``dup_confirm_px``). Confirm-or-drop, never guess: the wrong copy lands
many centimetres away and is thrown out, and no new survey is needed
because we only ever have to RECOGNISE the wrong copy, not locate it.

MIRROR POSE (2026-09-14). A planar target has an EXACT duplicate PnP
solution: negate every point's camera-frame coordinates and the pinhole
projection is unchanged (x/z = -x/-z), and for coplanar points that negation
is a proper pose — R·Rz(180°) with -t, the camera reflected through the tag
plane with its yaw flipped. Every tag then sits BEHIND the camera, which is
the one thing a reprojection error cannot see. solvePnP's cold start lands
there freely, and once it is the warm start it stays there: 42 consecutive
two-tag frames of data/20260914/0914_163103/nav_163112 (t 88.5-92.1 s)
reported the vehicle 0.44 m BELOW the floor, 0.7 m away and 180° round, at
2.2 px against a 3 px gate, and the NMPC chased it to full thrust. The
same frames solve to the true pose at 2.2 px from the IPPE seed. So every
joint fit is now CHEIRALITY-gated — a start whose points are not all in
front of the camera can neither win nor pass — and the mirror can never
become a warm start. No knob: a tag that was imaged is in front of the lens.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np


def tag_object_points(tag_size_m: float) -> np.ndarray:
    """VERBATIM convention from src/tagslam_core.py:233 (see module docstring)."""
    half = float(tag_size_m) / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )


def quat_wxyz_to_R(q) -> np.ndarray:
    """Unit quaternion (w, x, y, z) -> 3x3 rotation matrix (body->parent)."""
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


# =============================================================================
# tag map
# =============================================================================
class TagMap:
    """A locked map: tag id -> pose(s). Never optimized here.

    An id may carry MORE THAN ONE pose, because the pool mat physically
    reuses 12 ids (config/hw_nav.yaml duplicate_ids, confirmed by
    rov_gui/tools/build_tag_map.py finding two well-separated clusters for
    each). ``instances[id]`` is the full list; ``poses[id]`` is the first one
    and exists so every caller that only ever wants "a" pose for an id — the
    plot, the meta sidecar, the map builder's anchor lookup — keeps working
    unchanged. Code that must be correct in the presence of duplicates reads
    ``instances`` and ``is_unique``.
    """

    def __init__(self, poses: dict, anchor_id: int | None, source: str = "",
                 instances: dict | None = None):
        self.instances = ({int(k): list(v) for k, v in instances.items()}
                          if instances is not None
                          else {int(k): [v] for k, v in poses.items()})
        self.poses = {k: v[0] for k, v in self.instances.items()}
        self.anchor_id = anchor_id
        self.source = source

    def __contains__(self, tag_id: int) -> bool:
        return int(tag_id) in self.instances

    def __len__(self) -> int:
        return len(self.instances)

    def is_unique(self, tag_id: int) -> bool:
        return len(self.instances.get(int(tag_id), ())) == 1

    @property
    def duplicate_ids(self) -> frozenset:
        return frozenset(k for k, v in self.instances.items() if len(v) > 1)

    @classmethod
    def load(cls, path) -> "TagMap":
        """Read the tag_map.yaml format (src/tagslam_core.py:1529's schema).

        Keys are tag ids, except that a duplicated id writes one entry per
        physical tag as ``"<id>#<k>"`` (build_tag_map.py). Both forms load
        into the same ``instances`` dict.
        """
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict) or "tags" not in raw:
            raise ValueError(f"{path}: not a tag map (no 'tags' key)")
        inst: dict[int, list] = {}
        for key, entry in raw["tags"].items():
            tag_id = int(str(key).split("#")[0])
            t = np.asarray(entry["position_m"], float)
            R = quat_wxyz_to_R(entry["quaternion_wxyz"])
            inst.setdefault(tag_id, []).append((R, t))
        anchor = raw.get("anchor_tag_id")
        return cls({}, None if anchor is None else int(anchor), str(path),
                   instances=inst)

    @classmethod
    def single(cls, tag_id: int) -> "TagMap":
        """A one-tag map: the world frame IS that tag's frame (wall geometry)."""
        return cls({int(tag_id): (np.eye(3), np.zeros(3))}, int(tag_id),
                   f"single tag {tag_id}")


# =============================================================================
# detection
# =============================================================================
@dataclass
class Detection:
    tag_id: int
    corners: np.ndarray               # (4,2) float32, pupil-apriltags order
    decision_margin: float = 0.0
    # Signed pixels from the NEAREST corner to the image rectangle: positive
    # = the whole quad is inside, NEGATIVE = at least one corner lies outside
    # the frame, i.e. the detector EXTRAPOLATED it past the sensor edge and
    # those coordinates are not a measurement. Measured cost of trusting one:
    # 16 of the 18 reprojection-gate rejections on 2026-09-06 were carried by
    # a clipped quad, e.g. frame 1198 of
    # data/20260906/0906_194856/nav_194856,
    # where tag 27's corner sits at y=362.5 in a 360-row image and reprojects
    # at 7.12 px while the other six tags average 1.9 px.
    # ``inf`` = not measured (a Detection built by hand, e.g. in tests); such
    # a detection is never demoted for clipping.
    edge_px: float = float("inf")


class TagDetector:
    """pupil_apriltags when available, cv2.aruco otherwise. Same output order.

    cv2.aruco reports corners top-left, top-right, bottom-right, bottom-left;
    pupil reports the reverse of that (its (-h,+h) first corner is the
    BOTTOM-left of an upright tag because its tag frame has +y DOWN). The
    aruco path therefore reverses corner order so both backends feed the SAME
    object-point correspondence.
    """

    def __init__(self, family: str = "tag36h11", backend: str = "auto",
                 nthreads: int = 2, quad_decimate: float = 1.0,
                 max_hamming: int = 1, min_decision_margin: float = 20.0):
        self.family = family
        self.max_hamming = int(max_hamming)
        self.min_decision_margin = float(min_decision_margin)
        self.backend = ""
        self._pupil = None
        self._aruco = None
        if backend in ("auto", "pupil_apriltags"):
            try:
                from pupil_apriltags import Detector
                self._pupil = Detector(families=family, nthreads=int(nthreads),
                                       quad_decimate=float(quad_decimate),
                                       quad_sigma=0.0, refine_edges=1,
                                       decode_sharpening=0.25, debug=0)
                self.backend = "pupil_apriltags"
            except ImportError:
                if backend == "pupil_apriltags":
                    raise
        if self._pupil is None:
            import cv2
            if family != "tag36h11":
                raise ValueError(f"cv2.aruco fallback only maps tag36h11, "
                                 f"got {family!r}")
            dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
            par = cv2.aruco.DetectorParameters()
            par.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
            self._aruco = cv2.aruco.ArucoDetector(dic, par)
            self.backend = "cv2.aruco"

    @staticmethod
    def _edge_px(corners: np.ndarray, shape) -> float:
        """Distance from the nearest corner to the image rectangle (see
        ``Detection.edge_px``). Negative when a corner is off the sensor."""
        h, w = int(shape[0]), int(shape[1])
        c = np.asarray(corners, np.float64)
        return float(min(c[:, 0].min(), c[:, 1].min(),
                         (w - 1) - c[:, 0].max(), (h - 1) - c[:, 1].max()))

    def detect(self, gray: np.ndarray) -> list[Detection]:
        if self._pupil is not None:
            out = []
            for d in self._pupil.detect(gray, estimate_tag_pose=False):
                if int(d.hamming) > self.max_hamming:
                    continue
                if float(d.decision_margin) < self.min_decision_margin:
                    continue
                cs = np.asarray(d.corners, np.float32).reshape(4, 2)
                out.append(Detection(int(d.tag_id), cs,
                                     float(d.decision_margin),
                                     self._edge_px(cs, gray.shape)))
            return out
        corners, ids, _rej = self._aruco.detectMarkers(gray)
        if ids is None:
            return []
        out = []
        for cs, tid in zip(corners, ids.ravel()):
            # TL,TR,BR,BL -> BL,BR,TR,TL (the pupil order; module docstring)
            cs = np.asarray(cs, np.float32).reshape(4, 2)[::-1].copy()
            out.append(Detection(int(tid), cs, 0.0,
                                 self._edge_px(cs, gray.shape)))
        return out


# =============================================================================
# PnP
# =============================================================================
@dataclass
class NavSolution:
    """The vehicle pose in the NED world frame, plus everything needed to
    decide whether to trust it."""

    p_ned: np.ndarray
    R_ned_body: np.ndarray
    n_tags: int
    tag_ids: tuple
    # WHICH physical copy of each id was used (0 for every unique tag). Rides
    # alongside tag_ids so the plot can light the copy that actually carried
    # the fix instead of both squares that share the number.
    tag_insts: tuple = ()
    reproj_rms_px: float = 0.0
    ambiguous: bool = False
    note: str = ""
    rvec: np.ndarray | None = None    # camera_T_map, for the next warm start
    tvec: np.ndarray | None = None
    detect_ms: float = 0.0


def _reproj_rms(obj_pts, img_pts, rvec, tvec, K, dist) -> float:
    import cv2
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    err = proj.reshape(-1, 2) - img_pts.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))


def _all_in_front(obj_pts, rvec, tvec) -> bool:
    """Cheirality: is every object point in FRONT of the camera (z_cam > 0)?
    The pinhole projection cannot tell a point from its reflection through
    the lens, so this is the test the reprojection error lacks (module
    docstring, MIRROR POSE)."""
    import cv2
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    z = (R @ np.asarray(obj_pts, float).reshape(-1, 3).T
         + np.asarray(tvec, float).reshape(3, 1))[2]
    return bool(np.all(z > 0.0))


def _quad_area(corners) -> float:
    """Shoelace area of a detection's quad, in px² — the stand-in for "which
    tag is nearest / best conditioned" when picking a seed."""
    c = np.asarray(corners, np.float64)
    x, y = c[:, 0], c[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _rp_from_R_ned_body(R: np.ndarray) -> tuple[float, float]:
    """ZYX roll/pitch, same extraction as dobmpc.frames._euler_from_R."""
    theta = math.asin(max(-1.0, min(1.0, -float(R[2, 0]))))
    phi = math.atan2(float(R[2, 1]), float(R[2, 2]))
    return phi, theta


class TagNav:
    """Detections -> vehicle pose in NED. Stateless except the PnP warm start."""

    def __init__(self, tag_map: TagMap, tag_size_m: float,
                 R_frd_cam: np.ndarray, t_frd_cam: np.ndarray,
                 R_ned_map: np.ndarray,
                 max_reproj_px: float = 3.0,
                 ambiguity_ratio: float = 1.5,
                 tilt_gate_deg: float = 10.0,
                 min_tags: int = 1,
                 datum: str = "map",
                 duplicate_ids=(),
                 dup_confirm_px: float = 6.0,
                 outlier_max_frac: float = 0.30,
                 outlier_min_ratio: float = 1.8,
                 min_border_px: float | None = 0.0):
        self.map = tag_map
        self.obj_tag = tag_object_points(tag_size_m).astype(np.float64)
        # camera(optical) -> body(FRD): x_body = R_frd_cam x_cam + t_frd_cam
        self.R_bc = np.asarray(R_frd_cam, float)
        self.t_bc = np.asarray(t_frd_cam, float)
        self.R_cb = self.R_bc.T
        self.t_cb = -self.R_bc.T @ self.t_bc
        self.R_nm = np.asarray(R_ned_map, float)
        self.max_reproj_px = float(max_reproj_px)
        self.ambiguity_ratio = float(ambiguity_ratio)
        self.tilt_gate_rad = math.radians(float(tilt_gate_deg))
        self.min_tags = max(1, int(min_tags))
        # Ids that exist TWICE on the mat (module docstring). They never
        # anchor a solve; they may only join one the unique tags already
        # built, and only if they reproject within dup_confirm_px of where
        # the map puts them. A map that already holds two poses for an id
        # says so itself, so the config list only has to cover ids whose
        # second copy has not been surveyed yet.
        self.duplicate_ids = (frozenset(int(i) for i in duplicate_ids)
                              | tag_map.duplicate_ids)
        self.dup_confirm_px = float(dup_confirm_px)
        # OUTLIER RESCUE (see _drop_outliers). The joint fit pools its RMS
        # over every corner, so ONE tag with bad corners rejects a frame whose
        # other tags agree to ~1 px. Rather than loosen the gate, drop the
        # worst tag and refit — bounded so a uniformly-bad frame can never be
        # peeled down to a lucky subset that passes with a wrong pose.
        self.outlier_max_frac = float(outlier_max_frac)
        self.outlier_min_ratio = float(outlier_min_ratio)
        # Demote a detection whose quad reaches within this many pixels of the
        # image edge (0.0 = only quads that actually cross it; None = off).
        self.min_border_px = (None if min_border_px is None
                              else float(min_border_px))
        # Per-frame bookkeeping for the operator's screen.
        self.last_dropped: tuple = ()
        self.last_outliers: tuple = ()     # (tag_id, px) dropped by the rescue
        self.last_clipped: tuple = ()      # tag ids dropped at the image edge
        # datum="first_fix": the first accepted solve defines the world's
        # origin AND yaw zero — every later pose is expressed relative to
        # where (and which way) the run began. A pure horizontal isometry, so
        # roll/pitch (and the gravity gate) are untouched. reset_datum()
        # re-zeros; the worker calls it when the localizing feed is toggled.
        assert datum in ("map", "first_fix"), datum
        self.datum = datum
        self._datum_p0 = None
        self._datum_Rz = None                   # Rz(-yaw0)
        self._rvec = None                       # camera_T_map warm start
        self._tvec = None
        self._pnp_fail = ""                     # why _joint_pnp returned None
        # WHY the last solve returned None. "tags seen, none usable" with no
        # reason is undebuggable at the pool — this string reaches the sensor
        # row and the plot chip (found live 2026-08-12: the gravity gate was
        # rejecting every frame and nothing on screen said so).
        self.last_reject = ""

    def reset_datum(self) -> None:
        self._datum_p0 = None
        self._datum_Rz = None

    def datum_transform(self):
        """(Rz, p0) of the first-fix datum, or None (none yet / datum map).
        A second solver on another camera applies THIS one's datum so both
        report in one world (control/nav_fusion.py)."""
        if self.datum != "first_fix" or self._datum_p0 is None:
            return None
        return self._datum_Rz.copy(), self._datum_p0.copy()

    def set_extrinsic(self, R_frd_cam, t_frd_cam) -> None:
        """Re-point the camera->body extrinsic at runtime — the ROV RGB rides
        a tilt mount, so its extrinsic follows the tracked mount angle. The
        PnP warm start is camera_T_map and does not depend on this."""
        self.R_bc = np.asarray(R_frd_cam, float)
        self.t_bc = np.asarray(t_frd_cam, float)
        self.R_cb = self.R_bc.T
        self.t_cb = -self.R_bc.T @ self.t_bc

    def _apply_datum(self, sol: "NavSolution | None") -> "NavSolution | None":
        if sol is None or self.datum != "first_fix":
            return sol
        if self._datum_p0 is None:
            yaw0 = math.atan2(float(sol.R_ned_body[1, 0]),
                              float(sol.R_ned_body[0, 0]))
            c, s = math.cos(-yaw0), math.sin(-yaw0)
            self._datum_p0 = sol.p_ned.copy()
            self._datum_Rz = np.array([[c, -s, 0.0], [s, c, 0.0],
                                       [0.0, 0.0, 1.0]])
        sol.p_ned = self._datum_Rz @ (sol.p_ned - self._datum_p0)
        sol.R_ned_body = self._datum_Rz @ sol.R_ned_body
        return sol

    # ------------------------------------------------------------- composition
    def _solution(self, R_cm, t_cm, n_tags, ids, rms, ambiguous, note,
                  rvec, tvec, detect_ms, insts=None) -> NavSolution:
        """camera_T_map -> NED body pose, through the ONE extrinsic."""
        R_mc = R_cm.T
        t_mc = -R_cm.T @ t_cm
        R_map_body = R_mc @ self.R_cb
        t_map_body = R_mc @ self.t_cb + t_mc
        return NavSolution(
            p_ned=self.R_nm @ t_map_body,
            R_ned_body=self.R_nm @ R_map_body,
            n_tags=n_tags, tag_ids=tuple(ids),
            tag_insts=tuple(insts if insts is not None else [0] * len(ids)),
            reproj_rms_px=rms,
            ambiguous=ambiguous, note=note, rvec=rvec, tvec=tvec,
            detect_ms=detect_ms)

    # ------------------------------------------------------- joint PnP helpers
    def _obj_img(self, dets, insts=None) -> tuple[np.ndarray, np.ndarray]:
        """Map object points and image corners for a set of detections.

        ``insts`` optionally names WHICH instance of each id to use (for
        duplicated ids); it defaults to the first, which is the only one for
        every unique tag."""
        obj, img = [], []
        for k, d in enumerate(dets):
            i = 0 if insts is None else insts[k]
            R_mt, t_mt = self.map.instances[d.tag_id][i]
            obj.append((R_mt @ self.obj_tag.T).T + t_mt)
            img.append(d.corners.astype(np.float64))
        return (np.concatenate(obj).astype(np.float64),
                np.concatenate(img).astype(np.float64))

    def _seed_from_tag(self, dets, K, dist, insts=None):
        """Initial guesses for camera_T_map from the LARGEST tag's IPPE square,
        composed with that tag's map pose. Both IPPE branches are returned —
        the joint fit over the other tags is what decides between them."""
        import cv2

        k = int(np.argmax([_quad_area(d.corners) for d in dets]))
        d = dets[k]
        try:
            n_sol, rvecs, tvecs, _e = cv2.solvePnPGeneric(
                self.obj_tag, d.corners.astype(np.float64), K, dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except cv2.error:
            return []
        R_mt, t_mt = self.map.instances[d.tag_id][0 if insts is None
                                                  else insts[k]]
        out = []
        for i in range(n_sol):
            R_ct, _ = cv2.Rodrigues(rvecs[i])
            R_cm = R_ct @ R_mt.T
            rvec, _ = cv2.Rodrigues(R_cm)
            out.append((rvec, (tvecs[i].ravel() - R_cm @ t_mt).reshape(3, 1)))
        return out

    def _joint_pnp(self, dets, K, dist, rvec0=None, tvec0=None, insts=None):
        """Joint solvePnP over every given tag -> (rvec, tvec, rms) or None.

        BEST of up to three starts, in order and stopping as soon as one
        clears the gate: the caller's warm start, a cold solve, and a seed
        built from one tag's IPPE square (``_seed_from_tag``).

        Why three. A wildly wrong warm start can wedge LM in a bad basin, so a
        warm solve outside the gate is retried cold — but the COLD start has
        its own failure, and on a two-tag coplanar frame it is spectacular:
        492 px on frame 1214 of data/20260907/
        0907_133858/nav_134116, a frame whose own tags agree to 0.94 px. The
        old code only ever survived that by happening to hold a warm start
        from the previous frame. Seeding from a single tag's IPPE pose is the
        well-posed way to start a planar fit, so it is the last resort here.

        Why BEST rather than last: the cold retry used to REPLACE the warm
        answer unconditionally, so a 3.5 px warm solve could be overwritten by
        a 492 px cold one and reported as the fit.

        CHEIRALITY. A start that converges with any tag BEHIND the camera is
        the planar mirror pose (module docstring): it reprojects exactly as
        well as the truth, so it is neither kept as best nor allowed to pass,
        and the remaining starts run. A frame where EVERY start lands there
        returns None with ``_pnp_fail`` saying so. Measured: the 42 mirror
        frames of data/20260914/0914_163103/nav_163112 all recover the true
        pose (2.2 px) from the IPPE seed once the warm and cold mirrors are
        refused.
        """
        import cv2

        obj, img = self._obj_img(dets, insts)
        best = None
        behind = 0
        self._pnp_fail = ""

        def attempt(r0, t0) -> bool:
            """Run one start; keep it if it is the best so far. -> passed gate?"""
            nonlocal best, behind
            guess = r0 is not None
            ok, rvec, tvec = cv2.solvePnP(
                obj, img, K, dist,
                rvec=(np.array(r0, float).reshape(3, 1) if guess else None),
                tvec=(np.array(t0, float).reshape(3, 1) if guess else None),
                useExtrinsicGuess=guess, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return False
            if not _all_in_front(obj, rvec, tvec):
                behind += 1
                return False
            rms = _reproj_rms(obj, img, rvec, tvec, K, dist)
            if best is None or rms < best[2]:
                best = (rvec, tvec, rms)
            return rms <= self.max_reproj_px

        if rvec0 is not None and attempt(rvec0, tvec0):
            return best
        if attempt(None, None):
            return best
        for r0, t0 in self._seed_from_tag(dets, K, dist, insts):
            if attempt(r0, t0):
                break
        if best is None and behind:
            self._pnp_fail = (f"mirror pose only ({behind} start(s) put the "
                              f"tags behind the camera)")
        return best

    def _tag_reproj(self, d: Detection, rvec, tvec, K, dist,
                    inst: int = 0) -> float:
        """How far ONE tag's corners land from ONE of its map poses, in px."""
        R_mt, t_mt = self.map.instances[d.tag_id][inst]
        obj = ((R_mt @ self.obj_tag.T).T + t_mt).astype(np.float64)
        return _reproj_rms(obj, d.corners.astype(np.float64), rvec, tvec,
                           K, dist)

    def _pick_instance(self, d: Detection, rvec, tvec, K, dist):
        """WHICH physical copy of a duplicated id is this? -> (index, err).

        Scores every instance the map holds against the pose the unique tags
        already established. The winner must both fit (``dup_confirm_px``) and
        clearly beat the runner-up — two copies that both look plausible mean
        the pose is not good enough to arbitrate, and the detection is
        dropped rather than guessed.
        """
        errs = [self._tag_reproj(d, rvec, tvec, K, dist, i)
                for i in range(len(self.map.instances[d.tag_id]))]
        order = sorted(range(len(errs)), key=lambda i: errs[i])
        best = order[0]
        if not (errs[best] <= self.dup_confirm_px):
            return None, errs[best]
        if len(order) > 1 and errs[order[1]] < 2.0 * errs[best] + 1.0:
            return None, errs[best]          # the copies are not separable here
        return best, errs[best]

    def _drop_notes(self, wrong_copy=()) -> list:
        """What this frame threw away, for the solution's note field. Every
        discarded tag is named — a fix built on a pruned tag set must say so."""
        out = []
        if self.last_outliers:
            out.append("outlier tag(s) dropped: "
                       + ",".join(f"{t}@{e:.1f}px"
                                  for t, e in self.last_outliers))
        if wrong_copy:
            out.append("wrong-copy tag(s) dropped: "
                       + ",".join(str(t) for t, _e in self.last_dropped))
        if self.last_clipped:
            out.append("clipped at frame edge: "
                       + ",".join(str(t) for t in self.last_clipped))
        return out

    def _drop_outliers(self, anchors, K, dist, rvec, tvec, rms):
        """Rescue a frame the joint gate rejected, by dropping outlier tags.

        WHY. ``_joint_pnp`` pools its RMS over every corner with equal weight,
        so ONE tag whose corners are wrong — a quad clipped by the image edge,
        a mis-surveyed map entry — drags the pooled number past the gate while
        the pose itself is fine and every other tag sits near 1 px. Measured on
        the recordings of 2026-09-06: dropping the single worst tag and
        refitting rescues 55/55 gate rejections in
        data/20260906/0906_194856/nav_194856
        (median 4.63 -> 1.16 px) and 18/18 in .../0906_192348/nav_192348
        (3.30 -> 0.15 px). Example, frame 1198 of the first: per-tag residuals
        {27: 7.12, 28: 3.06, 44: 2.40, 29: 1.95, 58: 1.51, 30: 1.21, 31: 1.13}
        px for a joint 3.27 px against a 3.0 px gate — tag 27's quad runs to
        y=362.5 in a 360-row image.

        SAFETY. Peeling is how a bad fit turns into a confident wrong one, so
        three bounds apply and all three must hold for every drop:
          * at most ``outlier_max_frac`` of the anchors may go (always >= 1),
          * at least ``max(1, min_tags)`` anchors must remain,
          * the tag must STAND OUT — its residual must be at least
            ``outlier_min_ratio`` x the median of the others. A frame that is
            uniformly bad (wrong map, changed extrinsic, wrong camera model)
            has no standout tag and is REJECTED rather than peeled, which is
            the case this bound exists for.

        Returns ``(kept, rvec, tvec, rms)``, where a single survivor comes back
        with ``rvec=None`` for the caller to re-solve on the single-tag path,
        or ``None`` to reject (``last_reject`` explains which tag and why).
        """
        kept = list(anchors)
        dropped: list[tuple[int, float]] = []
        n_max = max(1, int(self.outlier_max_frac * len(anchors)))
        floor = max(1, self.min_tags)
        while rms > self.max_reproj_px:
            errs = [self._tag_reproj(d, rvec, tvec, K, dist) for d in kept]
            j = int(np.argmax(errs))
            others = [e for i, e in enumerate(errs) if i != j]
            med = float(np.median(others)) if others else 0.0
            stands_out = errs[j] >= self.outlier_min_ratio * med
            if len(dropped) >= n_max or len(kept) - 1 < floor or not stands_out:
                why = ("no single tag stands out — the whole frame disagrees "
                       "with the map" if not stands_out
                       else f"{len(dropped)}/{n_max} dropped, floor {floor}")
                self.last_reject = (
                    f"reproj {rms:.1f}px > {self.max_reproj_px:g}px "
                    f"({len(kept)} unique tags, worst {kept[j].tag_id} "
                    f"@{errs[j]:.1f}px: {why})")
                return None
            dropped.append((int(kept[j].tag_id), float(errs[j])))
            kept.pop(j)
            if len(kept) < 2:
                # One survivor: the planar 4-point fit is exactly the flip
                # ambiguity the single-tag IPPE branch exists to resolve, so
                # hand it there instead of guessing here.
                self.last_outliers = tuple(dropped)
                return kept, None, None, float("inf")
            got = self._joint_pnp(kept, K, dist, rvec, tvec)
            if got is None:
                self.last_reject = ((self._pnp_fail or "multi-tag PnP failed")
                                    + " after dropping "
                                    + ",".join(str(t) for t, _e in dropped))
                return None
            rvec, tvec, rms = got
        self.last_outliers = tuple(dropped)
        return kept, rvec, tvec, rms

    # ------------------------------------------------------------------- solve
    def solve(self, detections: list[Detection], K: np.ndarray, dist,
              rp_hint: tuple[float, float] | None = None) -> NavSolution | None:
        """PnP over every mapped detection, datum applied. ``rp_hint`` =
        (roll, pitch) from the autopilot, used ONLY to break the single-tag
        IPPE ambiguity (before the datum — the datum never changes tilt)."""
        return self._apply_datum(self._solve_raw(detections, K, dist, rp_hint))

    def _solve_raw(self, detections: list[Detection], K: np.ndarray, dist,
                   rp_hint: tuple[float, float] | None = None) -> NavSolution | None:
        import cv2

        t0 = time.perf_counter()
        K = np.asarray(K, np.float64)
        dist = None if dist is None else np.asarray(dist, np.float64).ravel()
        if dist is not None and len(dist) not in (0, 4, 5, 8, 12, 14):
            dist = dist[:8]                    # OpenCV accepts 4/5/8/12/14
        mapped = [d for d in detections if d.tag_id in self.map]
        self.last_dropped = ()
        self.last_outliers = ()
        self.last_clipped = ()
        # A quad that crosses the image edge has EXTRAPOLATED corners, not
        # measured ones (Detection.edge_px), and they are wrong by pixels —
        # the measured carrier of 16 of the 18 reprojection-gate rejections on
        # 2026-09-06. Drop them before anything counts tags, so a clipped copy
        # can neither anchor nor trip the seen-twice rule below.
        if self.min_border_px is not None:
            clipped = [d for d in mapped if d.edge_px < self.min_border_px]
            if clipped:
                self.last_clipped = tuple(int(d.tag_id) for d in clipped)
                mapped = [d for d in mapped if d.edge_px >= self.min_border_px]
        # An id detected TWICE in one frame means both physical copies are in
        # view (or one is a misread). Neither corner set can be trusted, so
        # both go — including for ids nobody declared duplicated, which is how
        # an UNDECLARED duplicate announces itself instead of poisoning the
        # solve. This also makes the min_tags count below honest: before, two
        # detections of one id satisfied min_tags=2.
        twice = sorted(i for i, n in Counter(d.tag_id for d in mapped).items()
                       if n > 1)
        if twice:
            mapped = [d for d in mapped if d.tag_id not in twice]
        # Duplicated ids never ANCHOR: the map holds one pose per id, so a
        # detection of the other copy is corners from somewhere else.
        anchors = [d for d in mapped if d.tag_id not in self.duplicate_ids]
        ambig = [d for d in mapped if d.tag_id in self.duplicate_ids]
        if len(anchors) < self.min_tags:
            # hw_nav.yaml min_tags: a floor-map run can demand >=2 tags so a
            # single grazing detection never carries the whole state estimate.
            bits = []
            if ambig:
                bits.append(f"{len(ambig)} dup held back")
            if twice:
                bits.append(f"{len(twice)} id(s) seen twice")
            if self.last_clipped:
                bits.append(f"{len(self.last_clipped)} clipped at frame edge")
            extra = f" ({', '.join(bits)})" if bits else ""
            self.last_reject = (f"{len(anchors)}/{self.min_tags} unique tags"
                                + extra if detections else "no tags")
            return None

        if len(anchors) >= 2:
            got = self._joint_pnp(anchors, K, dist, self._rvec, self._tvec)
            if got is None:
                self.last_reject = self._pnp_fail or "multi-tag PnP failed"
                return None
            rvec, tvec, rms = got
            if rms > self.max_reproj_px:
                # One bad tag must not cost the whole frame: drop it and
                # refit, under the bounds documented on _drop_outliers.
                res = self._drop_outliers(anchors, K, dist, rvec, tvec, rms)
                if res is None:
                    return None              # last_reject names the culprit
                anchors, rvec, tvec, rms = res

        if len(anchors) >= 2:
            # WHICH COPY is this? Score every instance the map holds against
            # the pose the unique tags just built. When the map knows both
            # copies (build_tag_map found them) the right one is recovered
            # and USED; when it knows only the surveyed one, the other copy
            # simply fails to fit and is dropped. Same code either way.
            used, insts, dropped = list(anchors), [0] * len(anchors), []
            for d in ambig:
                i, e = self._pick_instance(d, rvec, tvec, K, dist)
                if i is None:
                    dropped.append(d)
                    self.last_dropped += ((d.tag_id, float(e)),)
                else:
                    used.append(d)
                    insts.append(i)
            if len(used) > len(anchors):
                # Refit with the confirmed ones for the extra geometry, but
                # keep the anchor-only answer if the refit is not better.
                got2 = self._joint_pnp(used, K, dist, rvec, tvec, insts)
                if got2 is not None and got2[2] <= self.max_reproj_px:
                    rvec, tvec, rms = got2
                else:
                    used, insts = list(anchors), [0] * len(anchors)
            note = "; ".join(self._drop_notes(dropped))
            self._rvec, self._tvec = rvec.copy(), tvec.copy()
            R_cm, _ = cv2.Rodrigues(rvec)
            self.last_reject = ""
            return self._solution(R_cm, tvec.ravel(), len(used),
                                  [d.tag_id for d in used], rms,
                                  False, note, rvec, tvec,
                                  1e3 * (time.perf_counter() - t0), insts)
        mapped = anchors

        # ---- single tag: IPPE gives (up to) two solutions; disambiguate.
        d = mapped[0]
        img = d.corners.astype(np.float64)
        try:
            n_sol, rvecs, tvecs, errs = cv2.solvePnPGeneric(
                self.obj_tag, img, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except cv2.error:
            self.last_reject = "IPPE solve error"
            return None
        if n_sol < 1:
            self.last_reject = "IPPE found no pose"
            return None
        R_mt, t_mt = self.map.poses[d.tag_id]
        cands = []
        for i in range(n_sol):
            if not _all_in_front(self.obj_tag, rvecs[i], tvecs[i]):
                continue                       # mirror branch (module docstring)
            R_ct, _ = cv2.Rodrigues(rvecs[i])
            t_ct = tvecs[i].ravel()
            # camera_T_map = camera_T_tag o tag_T_map
            R_cm = R_ct @ R_mt.T
            t_cm = t_ct - R_cm @ t_mt
            rms = _reproj_rms(self.obj_tag, img, rvecs[i], tvecs[i], K, dist)
            cands.append((rms, R_cm, t_cm, rvecs[i], tvecs[i]))
        if not cands:
            self.last_reject = "IPPE: every pose puts the tag behind the camera"
            return None
        cands.sort(key=lambda c: c[0])
        best = cands[0]
        ambiguous = (len(cands) > 1
                     and cands[1][0] < best[0] * self.ambiguity_ratio)
        note = ""
        if ambiguous and rp_hint is not None:
            # The two IPPE solutions differ by a tag-plane flip, which shows
            # up as a large roll/pitch difference of the implied BODY pose.
            # The autopilot's gravity-referenced attitude is the arbiter.
            def rp_err(c):
                sol = self._solution(c[1], c[2], 1, (d.tag_id,), c[0], True,
                                     "", None, None, 0.0)
                phi, th = _rp_from_R_ned_body(sol.R_ned_body)
                return abs(_wrap(phi - rp_hint[0])) + abs(_wrap(th - rp_hint[1]))

            cands2 = sorted(cands, key=rp_err)
            if rp_err(cands2[0]) < rp_err(best) - 1e-9:
                best = cands2[0]
                note = "ippe flip resolved by attitude"
            if rp_err(cands2[0]) > 2 * self.tilt_gate_rad:
                self.last_reject = (
                    f"both IPPE poses tilt >"
                    f"{math.degrees(2 * self.tilt_gate_rad):.0f}° off gravity "
                    f"— tag vertical? camera tilt LEVEL?")
                return None                    # neither pose matches gravity
            ambiguous = rp_err(cands2[1]) < 2 * rp_err(cands2[0]) + 1e-9
        if best[0] > self.max_reproj_px:
            self.last_reject = (f"reproj {best[0]:.1f}px > "
                                f"{self.max_reproj_px:g}px")
            return None
        if rp_hint is not None and not ambiguous:
            sol = self._solution(best[1], best[2], 1, (d.tag_id,), best[0],
                                 False, "", None, None, 0.0)
            phi, th = _rp_from_R_ned_body(sol.R_ned_body)
            dphi = abs(_wrap(phi - rp_hint[0]))
            dth = abs(_wrap(th - rp_hint[1]))
            if dphi > self.tilt_gate_rad or dth > self.tilt_gate_rad:
                self.last_reject = (
                    f"gravity gate: Δroll {math.degrees(dphi):.0f}° / Δpitch "
                    f"{math.degrees(dth):.0f}° > "
                    f"{math.degrees(self.tilt_gate_rad):.0f}° — tag vertical? "
                    f"camera tilt LEVEL? (hw_nav tilt_gate_deg)")
                return None                    # pose disagrees with gravity
        # The lone anchor has fixed the pose, so the duplicated tags held back
        # above can now be CONFIRMED against it exactly as on the multi-tag
        # path. This is what turns "1 unique tag, the rest held back" — the
        # largest single cause of fix dropout measured on 2026-09-06, 372 of
        # 390 rejections in data/20260906/
        # 0906_192348/nav_192348 including one 10.9 s outage, the camera
        # seeing only ids {60, 65, 141} — into a three-tag fix. Confirming a
        # copy within dup_confirm_px of a ONE-tag pose is also independent
        # evidence that the IPPE branch chosen above is the right one: the
        # flipped pose puts every other tag centimetres from where it lands.
        rvec_map, _ = cv2.Rodrigues(np.asarray(best[1], float))
        tvec_map = np.asarray(best[2], float).reshape(3, 1)
        used, insts, wrong = [d], [0], []
        for a in ambig:
            i, e = self._pick_instance(a, rvec_map, tvec_map, K, dist)
            if i is None:
                wrong.append(a)
                self.last_dropped += ((a.tag_id, float(e)),)
            else:
                used.append(a)
                insts.append(i)
        R_cm, t_cm, rms = best[1], best[2], best[0]
        rvec_out = tvec_out = None
        if len(used) > 1:
            got = self._joint_pnp(used, K, dist, rvec_map, tvec_map, insts)
            if got is not None and got[2] <= self.max_reproj_px:
                rvec_out, tvec_out, rms = got
                R_cm, _ = cv2.Rodrigues(rvec_out)
                t_cm = tvec_out.ravel()
                note = "; ".join([n for n in
                                  [note, f"{len(used) - 1} duplicated tag(s) "
                                         f"confirmed"] if n])
            else:
                used, insts = [d], [0]         # keep the one-tag answer
        note = "; ".join([n for n in [note] + self._drop_notes(wrong) if n])
        # A map-frame refit CAN warm-start the next frame; a bare tag-frame
        # IPPE pose cannot (its vectors are in the tag's frame, not the map's).
        self._rvec = None if rvec_out is None else rvec_out.copy()
        self._tvec = None if tvec_out is None else tvec_out.copy()
        self.last_reject = ""
        return self._solution(R_cm, t_cm, len(used),
                              [a.tag_id for a in used], rms,
                              ambiguous, note, rvec_out, tvec_out,
                              1e3 * (time.perf_counter() - t0), insts)


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))
