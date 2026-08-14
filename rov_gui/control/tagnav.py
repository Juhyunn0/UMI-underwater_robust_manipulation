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
thirds of the rejections in sessions/nav_runs/20260813_17*). The mat is NOT
a repeated sheet, though: the operator's cell-by-cell survey shows the two
copies of an id share NO neighbour except their own pair partner, so the
tags detected ALONGSIDE a duplicated id say which copy it is. That is what
``duplicate_ids`` buys here — solve on the UNIQUE tags alone, then keep a
duplicated tag only if it reprojects where the map says it should
(``dup_confirm_px``). Confirm-or-drop, never guess: the wrong copy lands
many centimetres away and is thrown out, and no new survey is needed
because we only ever have to RECOGNISE the wrong copy, not locate it.
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

    def detect(self, gray: np.ndarray) -> list[Detection]:
        if self._pupil is not None:
            out = []
            for d in self._pupil.detect(gray, estimate_tag_pose=False):
                if int(d.hamming) > self.max_hamming:
                    continue
                if float(d.decision_margin) < self.min_decision_margin:
                    continue
                out.append(Detection(int(d.tag_id),
                                     np.asarray(d.corners, np.float32).reshape(4, 2),
                                     float(d.decision_margin)))
            return out
        corners, ids, _rej = self._aruco.detectMarkers(gray)
        if ids is None:
            return []
        out = []
        for cs, tid in zip(corners, ids.ravel()):
            # TL,TR,BR,BL -> BL,BR,TR,TL (the pupil order; module docstring)
            out.append(Detection(int(tid),
                                 np.asarray(cs, np.float32).reshape(4, 2)[::-1].copy()))
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
                 dup_confirm_px: float = 6.0):
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
        # Per-frame bookkeeping for the operator's screen.
        self.last_dropped: tuple = ()
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
        # WHY the last solve returned None. "tags seen, none usable" with no
        # reason is undebuggable at the pool — this string reaches the sensor
        # row and the plot chip (found live 2026-08-12: the gravity gate was
        # rejecting every frame and nothing on screen said so).
        self.last_reject = ""

    def reset_datum(self) -> None:
        self._datum_p0 = None
        self._datum_Rz = None

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

    def _joint_pnp(self, dets, K, dist, rvec0=None, tvec0=None, insts=None):
        """One joint solvePnP -> (rvec, tvec, rms), or None if it failed.

        A wildly wrong warm start can wedge LM in a bad basin, so a solve
        that starts warm and lands outside the gate gets ONE cold retry."""
        import cv2

        obj, img = self._obj_img(dets, insts)
        guess = rvec0 is not None
        ok, rvec, tvec = cv2.solvePnP(
            obj, img, K, dist,
            rvec=(rvec0.copy() if guess else None),
            tvec=(tvec0.copy() if guess else None),
            useExtrinsicGuess=guess, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        rms = _reproj_rms(obj, img, rvec, tvec, K, dist)
        if rms > self.max_reproj_px and guess:
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return None
            rms = _reproj_rms(obj, img, rvec, tvec, K, dist)
        return rvec, tvec, rms

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
            extra = ""
            if ambig or twice:
                extra = (f" ({len(ambig)} dup held back"
                         + (f", {len(twice)} id(s) seen twice" if twice else "")
                         + ")")
            self.last_reject = (f"{len(anchors)}/{self.min_tags} unique tags"
                                + extra if detections else "no tags")
            return None

        if len(anchors) >= 2:
            got = self._joint_pnp(anchors, K, dist, self._rvec, self._tvec)
            if got is None:
                self.last_reject = "multi-tag PnP failed"
                return None
            rvec, tvec, rms = got
            if rms > self.max_reproj_px:
                self.last_reject = (f"reproj {rms:.1f}px > "
                                    f"{self.max_reproj_px:g}px "
                                    f"({len(anchors)} unique tags)")
                return None
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
            note = ("" if not dropped else
                    "wrong-copy tag(s) dropped: "
                    + ",".join(str(t) for t, _e in self.last_dropped))
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
            R_ct, _ = cv2.Rodrigues(rvecs[i])
            t_ct = tvecs[i].ravel()
            # camera_T_map = camera_T_tag o tag_T_map
            R_cm = R_ct @ R_mt.T
            t_cm = t_ct - R_cm @ t_mt
            rms = _reproj_rms(self.obj_tag, img, rvecs[i], tvecs[i], K, dist)
            cands.append((rms, R_cm, t_cm, rvecs[i], tvecs[i]))
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
        self._rvec, self._tvec = None, None    # tag-frame vecs; don't warm-start map PnP
        self.last_reject = ""
        return self._solution(best[1], best[2], 1, (d.tag_id,), best[0],
                              ambiguous, note, None, None,
                              1e3 * (time.perf_counter() - t0))


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))
