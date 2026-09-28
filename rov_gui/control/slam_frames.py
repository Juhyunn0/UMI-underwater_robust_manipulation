#!/usr/bin/env python3
"""slam_frames.py — the station side of the ORB-SLAM3 driver link: the frame
algebra between the SLAM world and the station's NED frames, and the health
machine that catches the way ORB-SLAM3 fails.

WHY A SECOND NAV FRONT-END. On land the C3 sees a can, not the tag mat, so the
6-DoF pose that ``TagNav`` normally supplies comes from stereo ORB-SLAM3
instead. Everything downstream of the fix — ``StateAssembler``, the EAOB, the
NMPC, the plot — already speaks ONE convention (NED world, body FRD,
``rot_zyx``), so the job here is to land the SLAM pose in exactly that
convention and nowhere else, the same way ``geometry.py``/``tagnav.py`` do it
for tags. This module is the ONLY place the SLAM world is mentioned.

THE THREE FRAMES.

    SLAM world      the FIRST KEYFRAME's camera optical frame: +x right,
                    +y down, +z forward. It is what Pangolin draws map points
                    in, so it is also what the overlay wire carries.
    NED "map"       the SLAM-derived world the station navigates in, standing
                    in for the tag map: x north-ish, y east-ish, z DOWN.
    datum NED       the engage frame MpcWorker takes at ENGAGE
                    (workers.py:1485-1491): a horizontal isometry of the map.

    P = R_ned_slam = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]

is a cyclic permutation (det +1, so a rotation and not a mirror), i.e.
``p_ned = (z_slam, x_slam, y_slam)``: north <- forward, east <- right,
down <- down. A first keyframe taken with the C3 level therefore starts the
NED world pointing where the operator was pointing, which is what makes the
engage datum's yaw meaningful.

THE FORWARD CHAIN is ``TagNav._solution`` (tagnav.py:336-350) with ``R_ned_map``
replaced by ``P`` and the driver's ``Twc`` playing the part of the inverted PnP
solve — the algebra is TRANSCRIBED from there rather than re-derived, and the
inverse extrinsic is precomputed once in ``__init__`` exactly as
tagnav.py:279-283 does it, so the two front-ends cannot drift apart:

    R_slam_body = R_wc @ R_cb            R_cb = R_bc.T
    t_slam_body = R_wc @ t_cb + t_wc     t_cb = -R_bc.T @ t_bc
    R_ned_body  = P @ R_slam_body
    p_ned       = P @ t_slam_body

with ``(R_bc, t_bc) = NavConfig.R_t_frd_cam("main")`` (geometry.py:467-479,
contract ``x_bodyFRD = R_bc x_cam + t_bc``).

THE BACKWARD CHAIN is the same algebra run backwards, so a knot the station
sends lands on the Pangolin map with zero registration error rather than
"close enough": :meth:`SlamNav.twc_from_ned` is the exact inverse of
:meth:`SlamNav.ned_from_twc`, and :meth:`SlamNav.slam_from_datum_p` composes
``P.T`` with MpcWorker's engage isometry (forward ``_datumize``
workers.py:1678-1691, inverse ``_datum_to_map_p`` workers.py:3874-3880; the
plot does the same crossing at trajectory.py:206-213):

    p_map  = p0 + Rz(yaw0) @ p_datum
    p_slam = P.T @ p_map

WHY THE HEALTH MACHINE IS NOT OPTIONAL. ORB-SLAM3 does not report that the
world moved. On tracking loss ``Tracking::Track`` runs the block at
Tracking.cc:2270-2289 (verified 2026-09-03; the task brief said 2271-2288,
which is the body without the leading comment and the closing brace): with
<= 10 keyframes it calls ``System::ResetActiveMap``, otherwise
``CreateMapInAtlas`` (Tracking.cc:2286). Both set ``mState = NO_IMAGES_YET``
(``CreateMapInAtlas`` at Tracking.cc:2671, ``Tracking::ResetActiveMap`` at
Tracking.cc:3877 — NOT 3818, which is the whole-atlas ``Tracking::Reset``,
Tracking.cc:3779) and both start a BRAND NEW map whose origin is the next
keyframe. Stereo re-initialisation then needs a SINGLE frame with more
than 500 keypoints (``Tracking::StereoInitialization``, Tracking.cc:2335-2338),
so ``GetTrackingState()`` can be back to 2 (OK) one frame later — in a
different world, with no signal.

``MapChanged()`` does NOT cover that, and assuming it does is the trap:
System.cc:490-501 returns true only when ``Atlas::GetLastBigChangeIdx()``
increases, and the only writer is ``LoopClosing`` (LoopClosing.cc:1192, 2488),
i.e. a loop closure / GBA. ``CreateMapInAtlas`` informs no big change, and each
new ``Map`` starts ``mnBigChangeIdx`` at 0 (Map.cc:29, 36) while
``System::MapChanged``'s counter is a function-local ``static int`` that is
never reset — so after a map switch the flag can stay false for the rest of the
process. It is a useful "the map you already drew has been deformed" edge (it
is also CONSUMED by the reader that sees it first), and it is not a
new-world signal.

Hence :class:`TrackMonitor`: a session counter that increments on every
OK -> (not OK) -> OK transition. A session increment is an END OF RUN, never a
gap to bridge — the datum, the plan and the overlay ring all refer to a world
that no longer exists.

Pure numpy + stdlib. Imports only ``.state_assembler.rot_zyx`` so it loads (and
is testable) without Qt, cv2, torch or acados — the same discipline as
``policy_frames.py`` and ``plan_stream.py``. It deliberately does NOT import
``rov_gui.state``: :meth:`SlamFix.as_navfix_kwargs` returns the field names
instead, so the nav layer stays importable from a driver process that has no
GUI package on its path.

Conventions: metres, radians, seconds; SLAM world for ``Twc`` and for
everything the overlay wire carries; NED map for a fix; datum NED for plan
knots; body FRD throughout; ``rot_zyx(roll, pitch, yaw)`` = R_ned_body. All
angle outputs are wrapped to (-pi, pi].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from .state_assembler import rot_zyx

__all__ = [
    "P_NED_SLAM", "POSE_MAGIC", "POSE_VERSION", "TAG_TWC", "POSE_RECORD_STRUCT",
    "POSE_RECORD_LEN", "TRACK_STATE_OK", "KNOT_FLAG_ANCHOR",
    "DEFAULT_MAX_SPEED_MPS", "DEFAULT_GAP_S", "ORTHONORMAL_TOL",
    "wrap_pi", "rpy_from_R", "check_twc",
    "PoseRecord", "SlamFix", "SlamNav", "EngageDatum",
    "TrackStatus", "TrackEvent", "TrackMonitor",
]

# =============================================================================
# wire constants — the codec imports these, it does not re-spell them
# =============================================================================
#: Driver -> station stream header: b"POSE" then u32 version.
POSE_MAGIC = b"POSE"
POSE_VERSION = 1

#: The one record tag on that stream. FOUR ASCII BYTES with a trailing space —
#: compare with memcmp/== on the bytes, never by reinterpreting as an integer
#: (that would make the comparison endianness-dependent).
TAG_TWC = b"TWC "

#: The 'TWC ' payload, little-endian and UNPADDED ('<' switches struct to
#: standard sizes with no alignment, so the 3 reserved bytes are explicit):
#:     u32 seq | f64 t | i32 state | u8 map_changed | u8 rsv[3]
#:     | u32 n_tracked | f32 Twc[12]
POSE_RECORD_STRUCT = "<IdiB3xI12f"

#: 4 + 8 + 4 + 1 + 3 + 4 + 48 = 72. Written out because a wrong length does not
#: throw — it desynchronises the stream FOREVER, every later record reading
#: bytes from the middle of its predecessor. The test asserts both the
#: arithmetic and ``struct.calcsize(POSE_RECORD_STRUCT)`` against this.
POSE_RECORD_LEN = 72

#: ``Tracking::eTrackingState::OK`` (ORB_SLAM3/include/Tracking.h:121-129:
#: SYSTEM_NOT_READY=-1, NO_IMAGES_YET=0, NOT_INITIALIZED=1, OK=2,
#: RECENTLY_LOST=3, LOST=4, OK_KLT=5). Only 2 is a usable pose: RECENTLY_LOST
#: still emits a pose, but it is a motion-model extrapolation, not a
#: measurement, so this rig treats it as LOST.
TRACK_STATE_OK = 2

#: 'KNOT' flags bit0 — "the anchor field is valid" (the wire protocol). Kept
#: here because :meth:`SlamNav.slam_anchor` is what fills that field.
KNOT_FLAG_ANCHOR = 1 << 0

# =============================================================================
# frames
# =============================================================================
#: R_ned_slam. Cyclic permutation of the axes, det +1: p_ned = (z, x, y)_slam.
#: Rows read as "north takes forward, east takes right, down takes down".
P_NED_SLAM = np.array([[0.0, 0.0, 1.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0]])

#: ``check_twc`` calls R orthonormal when max |R.T R - I| is under this. Loose
#: enough for the f32 the wire carries (~1e-7 relative) and for a Sophus SE3f
#: that has been composed a few thousand times, tight enough that a garbage or
#: half-written buffer never passes.
ORTHONORMAL_TOL = 1e-3

#: Jump-guard defaults, reusing the reasoning of the trajectory plotter
#: ("/home/bdml/Desktop/data collection/UMI_Underwater/slam/plot_trajectory.py":54-60,
#: which breaks the drawn line on the same two tests): a pair of poses further
#: apart in time than ``gap_s`` is a tracking LOSS and implies no speed at all,
#: and above ``max_speed`` the step is a relocalisation snap rather than motion
#: — "Handheld peaks near 2-3 m/s; a relocalisation snap is an order of
#: magnitude above that" [스펙: that file's --max-speed help, default 3.0].
#: Not a measured handheld speed; nothing here has measured one.
DEFAULT_MAX_SPEED_MPS = 3.0
DEFAULT_GAP_S = 0.5


def wrap_pi(a: float) -> float:
    """Wrap to (-pi, pi] — the station's one wrap (workers.py:1687-1688,
    policy_frames._wrap).

    The closed end is the maths, not the arithmetic: ``sin(pi)`` is 1.2e-16
    rather than 0, so an input exactly ON the seam comes back with whichever
    sign that residue carries (``wrap_pi(-pi)`` is -pi). What is guaranteed is
    the angle and the magnitude, so never branch on the sign at +-pi.
    """
    return float(math.atan2(math.sin(a), math.cos(a)))


def rpy_from_R(R: np.ndarray) -> Tuple[float, float, float]:
    """ZYX Euler (roll, pitch, yaw) of R = Rz(yaw) Ry(pitch) Rx(roll), i.e. the
    inverse of :func:`state_assembler.rot_zyx` away from the pitch = +-90 deg
    singularity.

    Kept local for the reason ``state_assembler`` keeps ``rot_zyx`` local: this
    module must import from a driver process with nothing else on its path. It
    is the same extraction as ``tagnav._rp_from_R_ned_body`` (tagnav.py:257-261)
    for roll/pitch and ``TagNav._apply_datum`` (tagnav.py:325-326) for yaw, and
    the test pins it against ``rot_zyx`` on random rotations.
    """
    R = np.asarray(R, dtype=float)
    pitch = float(math.asin(max(-1.0, min(1.0, -float(R[2, 0])))))
    roll = float(math.atan2(float(R[2, 1]), float(R[2, 2])))
    yaw = float(math.atan2(float(R[1, 0]), float(R[0, 0])))
    return roll, pitch, yaw


def _twc_parts(Twc) -> Tuple[np.ndarray, np.ndarray]:
    """(R_wc (3,3), t_wc (3,)) from a (3,4) or (4,4) transform."""
    T = np.asarray(Twc, dtype=float)
    if T.shape not in ((3, 4), (4, 4)):
        raise ValueError(f"Twc must be (3, 4) or (4, 4), got {T.shape}")
    return T[:3, :3], T[:3, 3]


def check_twc(Twc) -> str:
    """"" when ``Twc`` is a usable rigid transform, else WHY it is not.

    Three failures this rig actually produces, in the order they are cheapest
    to test: a non-finite entry (a diverged solve, or a NaN that walked in
    through the f32 cast), an all-zero / non-orthonormal rotation (an
    uninitialised or half-written 48-byte block reads as exactly this, and
    ``state`` next to it can still say OK), and a reflection (an axis
    convention applied twice). Returning a REASON rather than a bool is what
    lets :class:`TrackMonitor` put the cause in the fix note instead of
    reporting a bare "lost" that nobody can debug at the poolside — the same
    lesson as ``TagNav.last_reject`` (tagnav.py:311-315).

    A wrong SHAPE still raises ``ValueError``: the codec always builds (3, 4),
    so that is a programmer error and not something the stream can produce.
    """
    R, t = _twc_parts(Twc)
    if not (np.all(np.isfinite(R)) and np.all(np.isfinite(t))):
        return "non-finite Twc"
    err = float(np.abs(R.T @ R - np.eye(3)).max())
    if err > ORTHONORMAL_TOL:
        return f"R not orthonormal (max |RtR-I| = {err:.3g})"
    if float(np.linalg.det(R)) < 0.0:
        return f"R is a reflection (det = {float(np.linalg.det(R)):.3f})"
    return ""


# =============================================================================
# the driver's POSE record
# =============================================================================
@dataclass(frozen=True)
class PoseRecord:
    """One decoded 'TWC ' record, still in the SLAM world.

    Field-for-field the wire (:data:`POSE_RECORD_STRUCT`); the codec builds
    these and hands them here, so the frame algebra never sees bytes.

    ``Twc`` is the row-major 3x4 [R|t] the driver computed as
    ``Tcw.inverse()`` — the CAMERA pose IN the SLAM world, which is the
    direction everything below assumes. ``t`` is the capture stamp the station
    stamped on the frame it sent (it round-trips through the driver so a fix
    ages off the same clock as every other sensor, ``NavFix.t_capture``).
    """

    seq: int
    t: float
    state: int
    map_changed: bool
    n_tracked: int
    Twc: np.ndarray                  # (3, 4) row-major [R|t], SLAM world

    @property
    def R_wc(self) -> np.ndarray:
        return _twc_parts(self.Twc)[0]

    @property
    def t_wc(self) -> np.ndarray:
        return _twc_parts(self.Twc)[1]

    @property
    def tracking_ok(self) -> bool:
        """``state`` says OK. Says NOTHING about the pose being usable, or
        about which world it is in — see the module docstring."""
        return int(self.state) == TRACK_STATE_OK


# =============================================================================
# a fix, NavFix-shaped
# =============================================================================
@dataclass(frozen=True)
class SlamFix:
    """The vehicle, in the SLAM-derived NED map frame. The SLAM twin of
    ``tagnav.NavSolution``, carrying the extra fields ``NavFix`` wants."""

    t_capture: float
    p_ned: np.ndarray                # (3,) metres, NED map frame
    R_ned_body: np.ndarray           # (3, 3) body(FRD) -> NED map
    roll: float
    pitch: float
    yaw: float
    n_tracked: int
    state: int
    session: int
    note: str = ""

    def eta6(self) -> np.ndarray:
        """``[x, y, z, roll, pitch, yaw]`` in the NED MAP frame (not the
        engage datum — the datum crossing is :class:`EngageDatum`)."""
        return np.array([self.p_ned[0], self.p_ned[1], self.p_ned[2],
                         self.roll, self.pitch, self.yaw], dtype=float)

    def as_navfix_kwargs(self) -> dict:
        """Keyword arguments for ``rov_gui.state.NavFix`` (state.py:416-458).

        Built as a dict rather than a ``NavFix`` so this module stays free of
        the GUI package (module docstring). Three deliberate choices:

        * ``n_tags=0`` — there are no tags. The map-point count is real
          information, so it goes in ``note`` where it cannot be misread as a
          tag count.
        * ``reproj_rms_px=None``, not 0.0. There is no reprojection residual
          on this path, and a 0.0 would print through window.py:1016 as
          "0.0 px" — a fabricated measurement (CLAUDE.md). ``None`` takes the
          else-branch at window.py:1017-1018 and prints the note instead.
        * ``geometry="slam"`` — a free display/CSV string; the two tag values
          ("floor"/"wall") are only read off ``NavConfig``
          (geometry.py:513-520), never off a fix, so a third value here tells
          the operator and the CSV which front-end produced the row.
        """
        return {
            "t_capture": float(self.t_capture),
            "n_tags": 0,
            "tag_ids": (),
            "tag_insts": (),
            "p_ned": tuple(float(v) for v in self.p_ned),
            "R_ned_body": tuple(float(v) for v in np.asarray(self.R_ned_body).ravel()),
            "yaw_ned": float(self.yaw),
            "reproj_rms_px": None,
            "geometry": "slam",
            "source": "main",
            "note": self.note or f"sess {self.session}, {self.n_tracked} pts",
        }


# =============================================================================
# the two directions
# =============================================================================
class SlamNav:
    """SLAM world <-> NED, through the ONE extrinsic. Stateless.

    ``R_frd_cam``/``t_frd_cam`` are ``NavConfig.R_t_frd_cam("main")``
    (geometry.py:467-479), contract ``x_bodyFRD = R_bc x_cam + t_bc``. The
    inverse extrinsic is precomputed here for the same reason tagnav.py:279-283
    precomputes it: it is used on every frame, and a second in-line derivation
    is a second chance to transpose it.

    ``R_ned_slam`` defaults to :data:`P_NED_SLAM` and is an argument only so a
    test can prove the algebra with a different (non-permutation) rotation.
    """

    def __init__(self, R_frd_cam: np.ndarray, t_frd_cam: np.ndarray,
                 R_ned_slam: np.ndarray = P_NED_SLAM):
        # camera(optical) -> body(FRD): x_body = R_bc x_cam + t_bc
        self.R_bc = np.asarray(R_frd_cam, dtype=float).reshape(3, 3)
        self.t_bc = np.asarray(t_frd_cam, dtype=float).reshape(3)
        self.R_cb = self.R_bc.T
        self.t_cb = -self.R_bc.T @ self.t_bc
        self.R_ns = np.asarray(R_ned_slam, dtype=float).reshape(3, 3)
        self.R_sn = self.R_ns.T

    # ------------------------------------------------------------- forward
    def ned_from_twc(self, Twc, check: bool = True
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """``Twc`` (SLAM world) -> ``(p_ned (3,), R_ned_body (3,3))``.

        ``TagNav._solution`` (tagnav.py:336-350) transcribed, with the SLAM
        world in the map's place. There the PnP gives camera_T_map and the
        first two lines INVERT it; here the driver already sends
        world_T_camera, so ``(R_wc, t_wc)`` enters where ``(R_mc, t_mc)`` did
        and the rest is line-for-line identical::

            R_map_body = R_mc @ R_cb          ->  R_slam_body = R_wc @ R_cb
            t_map_body = R_mc @ t_cb + t_mc   ->  t_slam_body = R_wc @ t_cb + t_wc
            p_ned      = R_nm @ t_map_body    ->  p_ned       = P @ t_slam_body
            R_ned_body = R_nm @ R_map_body    ->  R_ned_body  = P @ R_slam_body

        Raises ``ValueError`` with :func:`check_twc`'s reason unless
        ``check=False`` (the round-trip test's inner loop, where the input is
        constructed rather than received).
        """
        if check:
            why = check_twc(Twc)
            if why:
                raise ValueError(f"ned_from_twc: {why}")
        R_wc, t_wc = _twc_parts(Twc)
        R_slam_body = R_wc @ self.R_cb
        t_slam_body = R_wc @ self.t_cb + t_wc
        return self.R_ns @ t_slam_body, self.R_ns @ R_slam_body

    def fix_from_pose(self, rec: PoseRecord, session: int = 0,
                      note: str = "") -> SlamFix:
        """A :class:`PoseRecord` -> a :class:`SlamFix`. Raises ``ValueError``
        on a pose :func:`check_twc` rejects; ``state`` is carried through
        UNJUDGED (:class:`TrackMonitor` is what judges it, and it needs the
        fix to exist before it can jump-guard the position)."""
        p_ned, R_ned_body = self.ned_from_twc(rec.Twc)
        roll, pitch, yaw = rpy_from_R(R_ned_body)
        return SlamFix(t_capture=float(rec.t), p_ned=p_ned,
                       R_ned_body=R_ned_body, roll=roll, pitch=pitch, yaw=yaw,
                       n_tracked=int(rec.n_tracked), state=int(rec.state),
                       session=int(session), note=note)

    # ------------------------------------------------------------ backward
    def twc_from_ned(self, p_ned, R_ned_body) -> np.ndarray:
        """The EXACT inverse of :meth:`ned_from_twc`: -> ``Twc`` (3, 4).

        Solving the forward chain backwards, and simplifying the translation
        so the extrinsic is applied once instead of twice::

            R_slam_body = P.T @ R_ned_body
            t_slam_body = P.T @ p_ned
            R_wc = R_slam_body @ R_bc            (from R_slam_body = R_wc @ R_bc.T)
            t_wc = t_slam_body - R_wc @ t_cb
                 = t_slam_body + R_slam_body @ t_bc   (R_wc @ R_bc.T == R_slam_body)

        This is what puts a station-side pose back on the Pangolin map with
        zero registration error: not a fitted alignment, the algebra run in
        reverse.
        """
        p_ned = np.asarray(p_ned, dtype=float).reshape(3)
        R_ned_body = np.asarray(R_ned_body, dtype=float).reshape(3, 3)
        R_slam_body = self.R_sn @ R_ned_body
        t_slam_body = self.R_sn @ p_ned
        Twc = np.zeros((3, 4))
        Twc[:3, :3] = R_slam_body @ self.R_bc
        Twc[:3, 3] = t_slam_body + R_slam_body @ self.t_bc
        return Twc

    # -------------------------------------------------- position-only pair
    def ned_from_slam_p(self, p_slam) -> np.ndarray:
        """SLAM-world position(s) -> NED map. Accepts (3,) or (3, K)."""
        return self.R_ns @ np.asarray(p_slam, dtype=float)

    def slam_from_ned_p(self, p_ned) -> np.ndarray:
        """NED map position(s) -> SLAM world. Accepts (3,) or (3, K)."""
        return self.R_sn @ np.asarray(p_ned, dtype=float)

    def slam_from_datum_p(self, p_datum, datum: "Optional[EngageDatum]" = None
                          ) -> np.ndarray:
        """Datum-NED knot position(s) -> SLAM world. Accepts (3,) or (3, K).

        ``p_slam = P.T @ (p0 + Rz(yaw0) @ p_datum)`` — :meth:`slam_from_ned_p`
        after :meth:`EngageDatum.map_from_datum_p`. ``datum=None`` means "not
        engaged", where the datum is the identity and a datum position IS a
        map position (the same convention ``MpcWorker._datum_to_map_p`` uses
        when ``self._datum is None``, workers.py:3874-3880).

        This is the KNOT path: what the station sends in a 'KNOT' record's
        ``plan``/``raw`` arrays.
        """
        p = np.asarray(p_datum, dtype=float)
        p_map = p if datum is None else datum.map_from_datum_p(p)
        return self.R_sn @ p_map

    def slam_anchor(self, eta_datum6, datum: "Optional[EngageDatum]" = None
                    ) -> np.ndarray:
        """Datum-NED ``eta6`` -> the 'KNOT' record's ``anchor``: row-major 3x4
        [R|t] of the BODY frame in the SLAM world.

        Body, not camera — Pangolin draws the vehicle the plan is anchored to,
        and the camera is where the map points come from, so drawing the
        camera frame there would just re-draw the map's own origin trail. The
        rotation is the inverse of the forward chain's LAST step only
        (``R_slam_body = P.T @ R_ned_body``, no extrinsic), which is what makes
        the drawn triad the same triad ``eta[3:6]`` describes.

        Send it with :data:`KNOT_FLAG_ANCHOR` set; the reader keys off the flag
        because identity is a legitimate pose, not a missing one
        (ORB_SLAM3/include/PolicyOverlay.h, ``has_anchor``).
        """
        e = np.asarray(eta_datum6, dtype=float).reshape(6)
        R_ned_body = rot_zyx(float(e[3]), float(e[4]),
                             float(e[5] if datum is None
                                   else datum.map_from_datum_yaw(float(e[5]))))
        p_map = e[:3] if datum is None else datum.map_from_datum_p(e[:3])
        out = np.zeros((3, 4))
        out[:3, :3] = self.R_sn @ R_ned_body
        out[:3, 3] = self.R_sn @ p_map
        return out


# =============================================================================
# the engage datum (MpcWorker's isometry, transcribed)
# =============================================================================
class EngageDatum:
    """The horizontal isometry MpcWorker takes at ENGAGE, as a value.

    ``MpcWorker`` builds it inline at workers.py:1485-1491 —
    ``{"p0": eta_tag[:3], "yaw0": eta_tag[5], "Rz": rot_zyx(0, 0, -yaw0)}`` —
    and crosses it in four places (``_datumize`` workers.py:1678-1691,
    ``_datum_to_map_p``/``_map_to_datum_p`` workers.py:3874-3885, and the plot's
    own copy at trajectory.py:206-213 + ``_to_map_z``). This class is that same
    isometry so the SLAM path crosses it by CALLING it, not by writing a fifth
    copy; the plot's 2026-08-23 bug (``_to_map`` rotated x/y and silently left
    z alone, drawing every engaged run flat on the mat) is what a fifth copy
    costs.

    Note the ``Rz`` stored by the worker is ``Rz(-yaw0)``, i.e. the MAP -> DATUM
    direction, so ``map_from_datum_p`` transposes it. Positions translate AND
    rotate; a velocity must only rotate (``_map_to_datum_v``,
    workers.py:3887-3893) — that direction is deliberately not offered here,
    because the only thing this module sends back is positions.
    """

    def __init__(self, p0, yaw0: float):
        self.p0 = np.asarray(p0, dtype=float).reshape(3)
        self.yaw0 = float(yaw0)
        self.Rz = rot_zyx(0.0, 0.0, -self.yaw0)      # MAP -> DATUM

    @classmethod
    def from_eta(cls, eta_map6) -> "EngageDatum":
        """The datum a station engaging at ``eta_map6`` (MAP frame) would take
        — workers.py:1488-1490 verbatim."""
        e = np.asarray(eta_map6, dtype=float).reshape(6)
        return cls(e[:3], float(e[5]))

    # ------------------------------------------------------------ map -> datum
    def datum_from_map_p(self, p_map) -> np.ndarray:
        """``Rz @ (p - p0)``; (3,) or (3, K). ``_map_to_datum_p``,
        workers.py:3882-3885."""
        p = np.asarray(p_map, dtype=float)
        off = self.p0 if p.ndim == 1 else self.p0[:, None]
        return self.Rz @ (p - off)

    def datum_from_map_yaw(self, yaw_map: float) -> float:
        """``wrap(yaw - yaw0)`` — ``_datumize``, workers.py:1687-1688."""
        return wrap_pi(float(yaw_map) - self.yaw0)

    def datum_from_eta(self, eta_map6) -> np.ndarray:
        """A whole ``eta6``: position through the isometry, yaw wrapped,
        roll/pitch UNCHANGED (the isometry is horizontal) —
        ``_datumize``, workers.py:1678-1691."""
        e = np.asarray(eta_map6, dtype=float).reshape(6).copy()
        e[0:3] = self.datum_from_map_p(e[0:3])
        e[5] = self.datum_from_map_yaw(float(e[5]))
        return e

    # ------------------------------------------------------------ datum -> map
    def map_from_datum_p(self, p_datum) -> np.ndarray:
        """``p0 + Rz.T @ p``; (3,) or (3, K). ``_datum_to_map_p``,
        workers.py:3874-3880 — "in three dimensions", z included."""
        p = np.asarray(p_datum, dtype=float)
        off = self.p0 if p.ndim == 1 else self.p0[:, None]
        return off + self.Rz.T @ p

    def map_from_datum_yaw(self, yaw_datum: float) -> float:
        """``wrap(yaw + yaw0)`` — the inverse of
        :meth:`datum_from_map_yaw`."""
        return wrap_pi(float(yaw_datum) + self.yaw0)

    def map_from_eta(self, eta_datum6) -> np.ndarray:
        """The inverse of :meth:`datum_from_eta`."""
        e = np.asarray(eta_datum6, dtype=float).reshape(6).copy()
        e[0:3] = self.map_from_datum_p(e[0:3])
        e[5] = self.map_from_datum_yaw(float(e[5]))
        return e


# =============================================================================
# health — the silent-failure machine
# =============================================================================
class TrackStatus(Enum):
    """Per-record verdict, in the order the operator escalates through them
    (the ordering convention of ``state.Conn``)."""

    OK = "ok"                    # state OK, pose usable, same world as before
    MAP_CHANGED = "map_changed"  # OK, but LoopClosing deformed the map
    LOST = "lost"                # not OK, or a pose check_twc rejects


@dataclass(frozen=True)
class TrackEvent:
    """What :meth:`TrackMonitor.update` concluded about one record.

    ``session_started`` is the dangerous one: this record is OK and the
    previous world is GONE. ``clear_overlay`` is the union of the two reasons
    the drawn overlay is now stale (a new world, or a deformed map) and maps
    straight onto the wire's 'CLRO' tag.
    """

    status: TrackStatus
    session: int
    session_started: bool
    map_changed: bool
    jump: bool
    speed_mps: Optional[float]     # None when unmeasurable (see TrackMonitor)
    dt: Optional[float]
    n_tracked: int
    note: str = ""

    @property
    def ok(self) -> bool:
        """The pose may be consumed. True for MAP_CHANGED as well — the pose
        itself is valid, it is the DRAWING that is stale."""
        return self.status is not TrackStatus.LOST

    @property
    def clear_overlay(self) -> bool:
        return bool(self.session_started or self.map_changed)


class TrackMonitor:
    """Consecutive :class:`PoseRecord`s -> :class:`TrackEvent`s. Stateful; one
    per driver connection, ``reset()`` returns it to cold.

    THE SESSION COUNTER. It increments on every OK -> (not OK) -> OK
    transition, and the arming flag is what makes it fire ONCE per cycle
    rather than once per lost frame: the OK -> not-OK edge ARMS it and the
    next OK spends it. A steady LOST run therefore leaves it alone. See the
    module docstring for why this counter, and not ``GetTrackingState()`` or
    ``MapChanged()``, is the thing that tells you the world moved.

    A session increment is an END OF RUN. The datum was taken in the old
    world, the installed plan is expressed against it, and the overlay ring
    holds knots drawn in it — none of that survives, and nothing here tries to
    bridge it.

    THE JUMP GUARD reuses the trajectory plotter's two tests
    ("…/UMI_Underwater/slam/plot_trajectory.py":121-125): a step is a jump when
    the implied speed exceeds ``max_speed_mps`` AND the pair is not separated
    by a gap (``dt > gap_s``), because across a gap the pose pair spans a
    tracking loss and implies no speed at all. It is measured on the CAMERA
    position ``t_wc``, which is the trajectory that plotter draws and measures.
    The body position differs by the extrinsic lever arm rotated into the
    world, so a pure body rotation in place swings the camera and registers
    HERE as motion the body never made — an over-report, which is the safe
    direction for a guard looking for relocalisation snaps and the reason the
    threshold sits an order of magnitude above handheld speed rather than
    close to it. ``speed_mps`` is None whenever it is not measurable: the
    first OK record, the first record of a new session (different origin — the
    difference is meaningless, not fast), the record after a loss, and a
    non-advancing stamp.

    A jump does NOT force LOST. It is a flag on an otherwise-OK record,
    because the honest reading of one is "either the pose relocalised or the
    operator moved fast", and only the consumer knows which of those its gate
    should treat as fatal.
    """

    def __init__(self, max_speed_mps: float = DEFAULT_MAX_SPEED_MPS,
                 gap_s: float = DEFAULT_GAP_S,
                 ok_state: int = TRACK_STATE_OK):
        self.max_speed_mps = float(max_speed_mps)
        self.gap_s = float(gap_s)
        self.ok_state = int(ok_state)
        self.reset()

    def reset(self) -> None:
        self.session = 0
        self.n_ok = 0
        self.n_lost = 0
        self.n_jumps = 0
        self.n_map_changed = 0
        self.last_ok_t: Optional[float] = None
        self._was_ok = False
        self._armed = False              # an OK -> not-OK edge is owed a session
        self._prev_t: Optional[float] = None
        self._prev_p: Optional[np.ndarray] = None

    # ------------------------------------------------------------- update
    def update(self, rec: PoseRecord) -> TrackEvent:
        """Classify one record and advance the machine."""
        why = check_twc(rec.Twc)
        ok = (int(rec.state) == self.ok_state) and not why

        if not ok:
            # ARM on the OK -> not-OK EDGE only. Arming on every lost record
            # would be harmless (the flag is idempotent), but arming HERE and
            # spending it on the next OK is what makes the counter one per
            # cycle by construction rather than by accident.
            if self._was_ok:
                self._armed = True
            self._was_ok = False
            self.n_lost += 1
            # Drop the speed reference: the next OK pose is either in a new
            # world or across a gap, and a difference against a pre-loss pose
            # would be a fabricated speed either way.
            self._prev_t = None
            self._prev_p = None
            note = why or f"tracking state {int(rec.state)}"
            return TrackEvent(status=TrackStatus.LOST, session=self.session,
                              session_started=False, map_changed=False,
                              jump=False, speed_mps=None, dt=None,
                              n_tracked=int(rec.n_tracked), note=note)

        session_started = False
        if self._armed:
            self.session += 1
            self._armed = False
            session_started = True

        t = float(rec.t)
        p = np.asarray(rec.t_wc, dtype=float)
        dt: Optional[float] = None
        speed: Optional[float] = None
        jump = False
        note = "new session (world moved)" if session_started else ""
        # INVARIANT: at a session start ``_prev_t`` is None, so no speed is
        # measured across the origin change. It holds because the only place
        # that arms a session also drops the reference (the not-ok branch
        # above), and every record between the arming and this one was not-ok.
        # Stated rather than re-cleared here so the counters cannot disagree
        # with the returned event.
        if self._prev_t is not None and math.isfinite(t):
            d = t - self._prev_t
            if d > 0.0:
                dt = float(d)
                speed = float(np.linalg.norm(p - self._prev_p) / d)
                if dt <= self.gap_s and speed > self.max_speed_mps:
                    jump = True
                    self.n_jumps += 1
                    note = f"jump {speed:.1f} m/s over {dt * 1e3:.0f} ms"
            else:
                note = f"stamp did not advance (dt = {d:+.4f} s)"

        self._prev_t = t if math.isfinite(t) else None
        self._prev_p = p if math.isfinite(t) else None
        map_changed = bool(rec.map_changed)
        if map_changed:
            self.n_map_changed += 1
        self._was_ok = True
        self.n_ok += 1
        self.last_ok_t = t
        status = TrackStatus.MAP_CHANGED if map_changed else TrackStatus.OK
        return TrackEvent(status=status, session=self.session,
                          session_started=session_started,
                          map_changed=map_changed, jump=jump,
                          speed_mps=speed, dt=dt,
                          n_tracked=int(rec.n_tracked), note=note)

    # -------------------------------------------------------------- report
    def summary(self) -> dict:
        """Counters for the run meta / the operator's chip. ``sessions`` is a
        COUNT (1 for a clean run), ``session`` the current index."""
        return {"session": int(self.session), "sessions": int(self.session) + 1,
                "n_ok": int(self.n_ok), "n_lost": int(self.n_lost),
                "n_jumps": int(self.n_jumps),
                "n_map_changed": int(self.n_map_changed),
                "max_speed_mps": float(self.max_speed_mps),
                "gap_s": float(self.gap_s)}
