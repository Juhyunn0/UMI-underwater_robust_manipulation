#!/usr/bin/env python3
"""slam_wire.py — the station <-> ORB-SLAM3 driver pipe protocol, in one file.

WHY THIS EXISTS. The land-validation rig runs stereo ORB-SLAM3 in its OWN
process (it owns Pangolin's GL context and the ORB vocabulary; the station
owns the C3, the policy and Qt). They talk over the driver's stdin/stdout:

    station --stdin--> driver   rectified stereo frames + policy overlays
    driver --stdout--> station  the camera pose in the SLAM world frame

Both ends parse those bytes BY HAND — one in Python here, one in C++ in the
driver — so the layout is written down exactly once, in this module, and every
record length is recomputed and checked at import (:func:`_verify_layout`).
That check is the whole point of the file: a length that is wrong by one byte
does not corrupt one record, it slides the reader off the record boundary and
EVERY following record is read at the wrong offset. There is no resync marker
in either direction, so a desync is not a glitch, it is the end of the run.

Two rules that keep the two implementations from drifting apart:

  * TAGS ARE FOUR ASCII BYTES, compared as bytes (``memcmp`` in C++, ``==`` on
    a 4-byte ``bytes`` here — ``struct`` unpacks them with ``4s``, which is
    byte-order independent, so nobody has to remember which end the magic
    lands on). Never reinterpret a tag as an integer: that answer depends on
    the machine's endianness and the two sides would disagree silently.
    Note ``TAG_BYE`` ends in a SPACE, and so does ``TAG_POSE``.
  * UNKNOWN TAGS ARE SKIPPED, not fatal. Both decoders here honour that
    (``n_unknown`` counts them), which is what lets one side gain a record
    type without a flag day — the reader that does not know it steps over it
    by its length and stays in sync.

WHAT IS AND IS NOT HERE. This module is a CODEC and nothing else: no frame
algebra, no process handling, no numpy tricks beyond a dtype/shape check.
Positions handed to :func:`encode_knots` are already in the SLAM world frame
(the first keyframe's camera optical frame, +x right / +y down / +z forward —
the frame Pangolin draws map points in, PolicyOverlay.h:14-18). The NED<->SLAM
conversion belongs to the caller, because it owns the two pieces this file
must not guess at: the camera extrinsic (``TagNav._solution``,
control/tagnav.py:336-350, with its precomputed inverse at tagnav.py:279-283)
and the engage isometry (``MpcWorker._datumize`` forward, control/workers.py:
1678-1691; ``_datum_to_map_p`` inverse, workers.py:3874-3882).

Wire format, little-endian, NO implicit padding anywhere (every ``struct``
format below starts with ``<``, which turns alignment padding off; the ``3x``
pad bytes are EXPLICIT ``u8 rsv[3]`` fields, written as zeros).

    station -> driver (driver stdin)
      header once   "OAKX" | u32 version=2 | u32 width | u32 height
      record        u32 tag | u32 len | u8 payload[len]
        'FRAM'  len = 12 + 2*w*h : u32 seq | f64 t_capture | u8 L[w*h] | u8 R[w*h]
        'KNOT'  u32 plan_id | u32 flags | u8 status | u8 rsv[3] | f64 t_emit
                | u32 n_plan | u32 n_raw | f32 plan[3*n_plan] | f32 raw[3*n_raw]
                | f32 anchor[12]      (row-major 3x4 [R|t]; flags bit0 = valid)
        'CLRO'  len 0 : clear the overlay ring (datum change / re-arm)
        'BYE '  len 0 : clean shutdown

    driver -> station (driver stdout)
      header once   "POSE" | u32 version=1
      record        u32 tag='TWC ' | u32 len | u32 seq | f64 t | i32 state
                    | u8 map_changed | u8 rsv[3] | u32 n_tracked | f32 Twc[12]

``Twc`` is row-major 3x4 [R|t] = ``Tcw.inverse()``: the camera pose IN the SLAM
world frame. ``state`` is ``System::GetTrackingState()`` (:data:`TRACKING_OK`
= 2), ``map_changed`` is ``System::MapChanged()`` polled once per frame, and
``n_tracked`` is ``SLAM.GetTrackedMapPoints().size()``.

LEGACY. An "OAKD" header selects the older UNTAGGED frame stream (header then
bare ``u32 seq | f64 t | L | R`` records, no pose out, no overlays) that the
recorded-episode pipeline speaks — documented at
"/home/bdml/Desktop/data collection/UMI_Underwater/slam/oakd_live_slam.cc":11-14.
The driver still accepts it; :class:`DriverStreamDecoder` deliberately does
NOT (it would have to guess the frame size from the header alone and could
never see a tag again), and says so instead of half-parsing.

Pure stdlib + numpy: no Qt, no cv2, no torch, no depthai, no subprocess — the
same discipline as ``plan_stream.py`` and ``policy_frames.py``, so the
protocol is testable without a camera, a display or a SLAM build.
"""

from __future__ import annotations

import enum
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "SLAM_WIRE_VERSION_IN", "SLAM_WIRE_VERSION_OUT",
    "MAGIC_IN", "MAGIC_IN_LEGACY", "MAGIC_OUT",
    "TAG_FRAME", "TAG_KNOT", "TAG_CLEAR", "TAG_BYE", "TAG_POSE",
    "HEADER_IN_LEN", "HEADER_OUT_LEN", "REC_HEADER_LEN",
    "FRAME_FIXED_LEN", "KNOT_FIXED_LEN", "ANCHOR_LEN", "POSE_PAYLOAD_LEN",
    "MAX_RECORD_LEN", "TRACKING_OK", "KnotStatus", "STATUS_NAMES",
    "WireFormatError",
    "encode_header", "encode_frame", "encode_knots", "encode_clear",
    "encode_bye", "encode_pose", "encode_pose_header",
    "PoseRecord", "FrameRecord", "KnotRecord", "ClearRecord", "ByeRecord",
    "PoseDecoder", "DriverStreamDecoder",
]

# --------------------------------------------------------------------------
# magics, tags, versions
# --------------------------------------------------------------------------
#: Station -> driver stream. Version 2 is the TAGGED format described above;
#: version 1 never existed as a tagged stream (the untagged legacy carried no
#: version field at all), so the number also says "this header has 4 words".
MAGIC_IN = b"OAKX"
MAGIC_IN_LEGACY = b"OAKD"
SLAM_WIRE_VERSION_IN = 2

#: Driver -> station stream.
MAGIC_OUT = b"POSE"
SLAM_WIRE_VERSION_OUT = 1

TAG_FRAME = b"FRAM"
TAG_KNOT = b"KNOT"
TAG_CLEAR = b"CLRO"
TAG_BYE = b"BYE "        # trailing SPACE — tags are 4 bytes, always
TAG_POSE = b"TWC "       # trailing SPACE

#: ``Tracking::OK``. The driver forwards ``System::GetTrackingState()``
#: verbatim, including the pre-initialisation and lost states, so the station
#: can tell "not initialised yet" from "lost" instead of one boolean.
TRACKING_OK = 2


class KnotStatus(enum.IntEnum):
    """The station's filter verdict, carried as the wire's ``u8 status``.

    The values are the wire contract and are mirrored in the C++ struct
    (PolicyOverlay.h:39-43), which keeps the field as a plain ``int`` on
    purpose: an unknown future code must still DRAW (as "not accepted")
    rather than be dropped at the parser. :func:`encode_knots` therefore
    accepts any ``u8``, not just these five.
    """
    ACCEPT = 0
    CLIP = 1
    REJECT = 2
    LATE = 3
    SKIPPED = 4


#: Index-by-value names, for captions and logs.
STATUS_NAMES = ("accept", "clip", "reject", "late", "skipped")

#: flags bit0 of a 'KNOT' record: the anchor 3x4 is a real pose.
KNOT_FLAG_ANCHOR = 0x1

# --------------------------------------------------------------------------
# struct layouts — every one of them starts with '<': little-endian AND no
# alignment padding, which is what makes the byte counts below arithmetic
# rather than a property of the compiler on either side.
# --------------------------------------------------------------------------
_HDR_IN = struct.Struct("<4sIII")        # magic, version, width, height
_HDR_OUT = struct.Struct("<4sI")         # magic, version
_REC = struct.Struct("<4sI")             # tag, payload length
_FRAME_FIXED = struct.Struct("<Id")      # seq, t_capture
_KNOT_FIXED = struct.Struct("<IIB3xdII")  # plan_id, flags, status, rsv[3],
#                                          t_emit, n_plan, n_raw
_POSE_FIXED = struct.Struct("<IdiB3xI")  # seq, t, state, map_changed, rsv[3],
#                                          n_tracked

HEADER_IN_LEN = 4 + 4 + 4 + 4            # = 16
HEADER_OUT_LEN = 4 + 4                   # = 8
REC_HEADER_LEN = 4 + 4                   # = 8
FRAME_FIXED_LEN = 4 + 8                  # = 12, then 2*w*h image bytes
KNOT_FIXED_LEN = 4 + 4 + 1 + 3 + 8 + 4 + 4       # = 28
ANCHOR_LEN = 12 * 4                              # = 48 (f32 row-major 3x4)
POSE_PAYLOAD_LEN = 4 + 8 + 4 + 1 + 3 + 4 + 48    # = 72

#: A desynchronised reader gets GARBAGE in the length word, and a reader that
#: trusts it buffers until the machine dies. The cap turns that into a loud
#: failure AT the record where sync was lost. It must stay comfortably above
#: the largest legitimate record, which is a 'FRAM' at 12 + 2*w*h bytes.
MAX_RECORD_LEN = 64 * 1024 * 1024


class WireFormatError(ValueError):
    """The stream said something that cannot be parsed at this offset.

    A ``ValueError`` so a caller can catch encoder argument errors and stream
    corruption with one ``except``; a distinct type so the driver-supervision
    layer can tell "the pipe is broken, kill the child" from "my own call was
    wrong".
    """


def _verify_layout() -> None:
    """Re-derive every documented length and refuse to import if one moved.

    Deliberately NOT written with ``assert``: ``python -O`` strips those, and
    this is the single check that keeps the Python and the C++ transcription
    of the protocol in sync. It costs microseconds, once.
    """
    checks: Tuple[Tuple[str, int, int], ...] = (
        ("HEADER_IN_LEN", HEADER_IN_LEN, _HDR_IN.size),
        ("HEADER_OUT_LEN", HEADER_OUT_LEN, _HDR_OUT.size),
        ("REC_HEADER_LEN", REC_HEADER_LEN, _REC.size),
        ("FRAME_FIXED_LEN", FRAME_FIXED_LEN, _FRAME_FIXED.size),
        ("KNOT_FIXED_LEN", KNOT_FIXED_LEN, _KNOT_FIXED.size),
        # 4 (seq) + 8 (t) + 4 (state) + 1 (map_changed) + 3 (rsv) + 4
        # (n_tracked) = 24 fixed, + 48 for f32 Twc[12] = 72. The C++ writer
        # states the same arithmetic beside its own writev.
        ("POSE_PAYLOAD_LEN", POSE_PAYLOAD_LEN, _POSE_FIXED.size + ANCHOR_LEN),
    )
    for name, documented, computed in checks:
        if documented != computed:
            raise AssertionError(
                f"slam_wire layout drift: {name} documented {documented} but "
                f"struct computes {computed} — the C++ side was written "
                "against the documented number, fix the format string")
    for tag in (TAG_FRAME, TAG_KNOT, TAG_CLEAR, TAG_BYE, TAG_POSE):
        if len(tag) != 4:
            raise AssertionError(f"slam_wire: tag {tag!r} is not 4 bytes")
    for magic in (MAGIC_IN, MAGIC_IN_LEGACY, MAGIC_OUT):
        if len(magic) != 4:
            raise AssertionError(f"slam_wire: magic {magic!r} is not 4 bytes")
    if POSE_PAYLOAD_LEN != 72:
        raise AssertionError("slam_wire: the pose payload is 72 bytes by "
                             "contract; both sides hard-code it")


_verify_layout()


# ==========================================================================
# encoders — station -> driver
# ==========================================================================
def _u32(value, what: str) -> int:
    """A u32 field, range-checked. No silent masking: a counter that wrapped
    into the next field is exactly the kind of bug this protocol cannot see."""
    v = int(value)
    if not (0 <= v <= 0xFFFFFFFF):
        raise ValueError(f"{what} must fit in u32, got {v}")
    return v


def _check_image(img, name: str) -> np.ndarray:
    """A stereo plane as the wire wants it: 2-D, uint8, C-contiguous.

    Contiguity is CHECKED, not repaired. ``np.ascontiguousarray`` here would
    hide a full-plane copy per camera per frame inside a hot path, and the
    caller that handed in a slice (a crop view, a flipped view) almost never
    meant to pay for it silently — it usually means the rectifier's output was
    sliced instead of re-rectified.
    """
    arr = img if isinstance(img, np.ndarray) else np.asarray(img)
    if arr.dtype != np.uint8:
        raise ValueError(f"{name} image must be uint8 (the wire carries one "
                         f"byte per pixel), got dtype {arr.dtype}")
    if arr.ndim != 2:
        raise ValueError(f"{name} image must be 2-D (H, W) greyscale, got "
                         f"shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} image is empty (shape {arr.shape}); the "
                         "header's geometry is never zero, so neither is a "
                         "frame")
    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError(
            f"{name} image is not C-contiguous; the encoder will not copy it "
            "for you (a per-frame hidden copy). Pass the rectifier's own "
            "output, or np.ascontiguousarray(...) at the call site")
    return arr


def encode_header(width: int, height: int) -> bytes:
    """The one-shot "OAKX" header: magic, version, frame geometry.

    Sent BEFORE the SLAM system exists on the far side — the driver reads the
    geometry first so a producer that failed to start reports itself now,
    rather than after the vocabulary has loaded (the legacy driver's reason,
    oakd_live_slam.cc:145-148, and it still holds).
    """
    w = _u32(width, "width")
    h = _u32(height, "height")
    if w == 0 or h == 0:
        raise ValueError(f"frame geometry must be non-zero, got {w}x{h}")
    return _HDR_IN.pack(MAGIC_IN, SLAM_WIRE_VERSION_IN, w, h)


def encode_frame(seq: int, t_capture: float, left, right) -> bytes:
    """One 'FRAM' record: ``u32 seq | f64 t_capture | u8 L[w*h] | u8 R[w*h]``.

    Both planes must be the SAME shape and match the geometry announced by
    :func:`encode_header` — this function cannot check the latter (it never
    saw the header), so the decoder does, against ``12 + 2*w*h``.
    ``t_capture`` is the station's own clock at capture; ORB-SLAM3 only ever
    uses it as a monotonically increasing timestamp.
    """
    l_img = _check_image(left, "left")
    r_img = _check_image(right, "right")
    if l_img.shape != r_img.shape:
        raise ValueError(f"left {l_img.shape} and right {r_img.shape} planes "
                         "must have the same shape")
    payload_len = FRAME_FIXED_LEN + 2 * l_img.size
    return b"".join((
        _REC.pack(TAG_FRAME, _u32(payload_len, "frame payload length")),
        _FRAME_FIXED.pack(_u32(seq, "seq"), float(t_capture)),
        l_img.tobytes(), r_img.tobytes()))


def _xyz_block(pts, name: str) -> Tuple[np.ndarray, int]:
    """(N, 3) float32 xyz -> (contiguous array, N).

    (N, 3) and NOT (3, N): the station's own plans are (3, K)
    (``PlanMsg.p_ned``, plan_stream.py:94), so the caller must transpose, and
    the two shapes are indistinguishable at K = 3. Passing ``p_ned`` straight
    in would therefore go unnoticed on a three-knot plan and draw a transposed
    scribble on every other one, so the docstring says it loudly and the
    error names the axis.
    """
    if pts is None:
        return np.zeros((0, 3), dtype=np.float32), 0
    arr = np.asarray(pts, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.float32), 0
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must be (N, 3) xyz rows, got shape "
                         f"{arr.shape} (a (3, N) plan needs .T)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} has non-finite entries; a NaN reaches GL as "
                         "an undrawable vertex and takes the whole polyline "
                         "with it")
    return np.ascontiguousarray(arr), int(arr.shape[0])


def _anchor_block(anchor) -> Tuple[np.ndarray, int]:
    """(3, 4) row-major [R|t] float32 + the flags word.

    ``None`` sends IDENTITY with the flag clear, not zeros: the C++ default is
    ``Matrix4f::Identity()`` (PolicyOverlay.h:56) and a consumer that ignored
    ``has_anchor`` would then draw an axis triad at the world origin instead
    of collapsing one onto a singular zero matrix. The record always carries
    the 48 bytes so the length arithmetic has no branch in it.
    """
    if anchor is None:
        eye = np.zeros((3, 4), dtype=np.float32)
        eye[0, 0] = eye[1, 1] = eye[2, 2] = 1.0
        return eye, 0
    arr = np.asarray(anchor, dtype=np.float32)
    if arr.shape == (4, 4):
        arr = arr[:3, :4]
    if arr.shape != (3, 4):
        raise ValueError(f"anchor must be (3, 4) [R|t] (or (4, 4)), got shape "
                         f"{arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("anchor has non-finite entries")
    return np.ascontiguousarray(arr), KNOT_FLAG_ANCHOR


def encode_knots(plan_id: int, status: int, t_emit: float, plan_xyz,
                 raw_xyz=None, anchor=None) -> bytes:
    """One 'KNOT' record: the plan the controller got, plus what the network
    said before the safety filter, plus the anchor pose the plan hangs off.

    ``plan_xyz`` / ``raw_xyz`` are (N, 3) rows in the SLAM WORLD frame (see
    the module docstring — the NED->SLAM rotation and the engage isometry are
    the caller's job, and doing them there is what makes a knot land on the
    map with zero registration error). ``raw_xyz=None`` sends an empty raw
    polyline, which is the honest encoding of "the filter had nothing to
    compare against", not a copy of the plan.

    ``status`` is a ``u8`` (:class:`KnotStatus` names 0..4); any other code in
    range is passed through on purpose, because the drawing side treats an
    unknown code as "not accepted" instead of dropping the overlay
    (PolicyOverlay.h:39-43).
    """
    plan_arr, n_plan = _xyz_block(plan_xyz, "plan_xyz")
    raw_arr, n_raw = _xyz_block(raw_xyz, "raw_xyz")
    anchor_arr, flags = _anchor_block(anchor)
    st = int(status)
    if not (0 <= st <= 0xFF):
        raise ValueError(f"status must fit in u8, got {st}")
    payload_len = KNOT_FIXED_LEN + 12 * (n_plan + n_raw) + ANCHOR_LEN
    return b"".join((
        _REC.pack(TAG_KNOT, _u32(payload_len, "knot payload length")),
        _KNOT_FIXED.pack(_u32(plan_id, "plan_id"), _u32(flags, "flags"), st,
                         float(t_emit), n_plan, n_raw),
        plan_arr.tobytes(), raw_arr.tobytes(), anchor_arr.tobytes()))


def encode_clear() -> bytes:
    """'CLRO', empty: drop every overlay the driver is holding.

    Sent on a datum change or a re-arm, where the stored overlays are still
    valid geometry in a world frame that no longer means the same thing —
    leaving them up would draw last engage's plan against this engage's map.
    """
    return _REC.pack(TAG_CLEAR, 0)


def encode_bye() -> bytes:
    """'BYE ', empty: a clean shutdown request (note the trailing space).

    Closing the pipe also stops the driver, but only after it has blocked in
    a read; the explicit record lets it shut the SLAM system and save its
    trajectory the same way an operator's window close would.
    """
    return _REC.pack(TAG_BYE, 0)


# ==========================================================================
# encoder — driver -> station (the driver's own side, in Python)
# ==========================================================================
def encode_pose_header() -> bytes:
    """The one-shot "POSE" header the driver writes before its first record."""
    return _HDR_OUT.pack(MAGIC_OUT, SLAM_WIRE_VERSION_OUT)


def encode_pose(seq: int, t: float, state: int, map_changed: bool,
                n_tracked: int, Twc) -> bytes:
    """One 'TWC ' record, payload length :data:`POSE_PAYLOAD_LEN` = 72.

    The C++ driver is what writes these in the live rig; this exists so the
    station can be exercised (and this codec round-tripped) against a fake
    driver with no SLAM build, and so the layout has a second, executable
    transcription to disagree with if either side drifts.

    ``Twc`` is (3, 4) row-major [R|t] — or (4, 4), whose bottom row is dropped
    because it is (0, 0, 0, 1) by construction and 4 wasted floats per frame
    is 4 more chances to disagree about row/column order.
    """
    arr = np.asarray(Twc, dtype=np.float32)
    if arr.shape == (4, 4):
        arr = arr[:3, :4]
    if arr.shape != (3, 4):
        raise ValueError(f"Twc must be (3, 4) [R|t] (or (4, 4)), got shape "
                         f"{arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Twc has non-finite entries")
    payload = _POSE_FIXED.pack(_u32(seq, "seq"), float(t), int(state),
                               1 if map_changed else 0,
                               _u32(n_tracked, "n_tracked"))
    payload += np.ascontiguousarray(arr).tobytes()
    if len(payload) != POSE_PAYLOAD_LEN:                 # belt and braces
        raise AssertionError(f"pose payload is {len(payload)} B, contract says "
                             f"{POSE_PAYLOAD_LEN}")
    return _REC.pack(TAG_POSE, POSE_PAYLOAD_LEN) + payload


# ==========================================================================
# records
# ==========================================================================
@dataclass(frozen=True)
class PoseRecord:
    """One 'TWC ' record: where the driver put the camera for frame ``seq``.

    ``Twc`` is (3, 4) float32 row-major [R|t] in the SLAM world frame. The
    rotation and translation helpers are ``R_wc`` / ``t_wc`` rather than
    ``R`` / ``t`` because ``t`` is already this record's TIMESTAMP; the names
    also match the frame algebra in the module docstring, where ``t_wc`` is
    the camera position and ``R_wc`` the camera orientation IN the world.
    """
    seq: int
    t: float
    state: int
    map_changed: bool
    n_tracked: int
    Twc: np.ndarray

    @property
    def R_wc(self) -> np.ndarray:
        return self.Twc[:, :3]

    @property
    def t_wc(self) -> np.ndarray:
        return self.Twc[:, 3]

    @property
    def tracking_ok(self) -> bool:
        """``state == Tracking::OK``. Anything else (lost, not initialised,
        relocalising) means the pose is stale or absent, NOT merely poor."""
        return int(self.state) == TRACKING_OK

    def as_4x4(self) -> np.ndarray:
        """The same pose as a homogeneous 4x4 (bottom row 0,0,0,1)."""
        out = np.zeros((4, 4), dtype=np.float32)
        out[:3, :4] = self.Twc
        out[3, 3] = 1.0
        return out


@dataclass(frozen=True)
class FrameRecord:
    """One 'FRAM' record, planes reshaped with the header's geometry."""
    seq: int
    t_capture: float
    left: np.ndarray          # (H, W) uint8
    right: np.ndarray         # (H, W) uint8


@dataclass(frozen=True)
class KnotRecord:
    """One 'KNOT' record. ``anchor`` is None when flags bit0 was clear —
    the wire always carried 48 bytes, but identity is a legitimate pose and
    must not be confused with "absent" (PolicyOverlay.h:54-57)."""
    plan_id: int
    status: int
    t_emit: float
    flags: int
    plan: np.ndarray          # (n_plan, 3) float32
    raw: np.ndarray           # (n_raw, 3) float32
    anchor: Optional[np.ndarray] = None    # (3, 4) float32 or None

    @property
    def status_name(self) -> str:
        i = int(self.status)
        return STATUS_NAMES[i] if 0 <= i < len(STATUS_NAMES) else f"code{i}"


@dataclass(frozen=True)
class ClearRecord:
    """'CLRO' — no payload."""


@dataclass(frozen=True)
class ByeRecord:
    """'BYE ' — no payload."""


# ==========================================================================
# streaming decoders
# ==========================================================================
class _RecordStream:
    """Tag/length record splitter that tolerates ARBITRARY chunk boundaries.

    A pipe read returns whatever happened to be in the kernel buffer: half a
    header, a record and a half, one byte. Everything that is not yet a whole
    record stays in ``self._buf`` and is re-examined on the next
    :meth:`feed`, so the byte-at-a-time case and the whole-buffer case decode
    identically (the test asserts exactly that — it is the property the live
    rig depends on and the one that is easy to get subtly wrong).

    :meth:`feed` RETURNS A LIST and is not a generator on purpose. A lazy
    generator that the caller drops on the floor (``dec.feed(chunk)`` with no
    loop, a plausible line in a Qt slot) would consume nothing, and the
    buffer would grow forever while the station quietly saw no poses.
    """

    MAGIC: bytes = b""
    VERSION: int = 0
    HEADER: struct.Struct = _HDR_OUT
    MAX_PAYLOAD: int = MAX_RECORD_LEN
    NAME: str = "stream"

    def __init__(self) -> None:
        self._buf = bytearray()
        self.header_seen = False
        self.version: Optional[int] = None
        self.n_records = 0                    # decoded, known tags
        self.n_unknown = 0                    # skipped, unknown tags
        self.unknown_tags: Dict[bytes, int] = {}

    # ------------------------------------------------------------- state
    @property
    def pending(self) -> int:
        """Bytes held back because they are not yet a whole record."""
        return len(self._buf)

    def reset(self) -> None:
        """Forget the buffer and the header (a restarted child process
        sends its header again; a decoder that kept ``header_seen`` would
        parse "POSE" as a tag and desync on the first record)."""
        self._buf.clear()
        self.header_seen = False
        self.version = None

    # ------------------------------------------------------------- parse
    def feed(self, chunk) -> List[object]:
        """Add ``chunk`` and return every COMPLETE record it finished."""
        if chunk:
            self._buf += bytes(chunk)
        out: List[object] = []
        pos = 0
        n = len(self._buf)
        if not self.header_seen:
            if n < self.HEADER.size:
                return out
            self._read_header(bytes(self._buf[:self.HEADER.size]))
            self.header_seen = True
            pos = self.HEADER.size
        while n - pos >= REC_HEADER_LEN:
            tag, length = _REC.unpack_from(self._buf, pos)
            if length > self.MAX_PAYLOAD:
                raise WireFormatError(
                    f"{self.NAME}: record {tag!r} claims {length} B, cap is "
                    f"{self.MAX_PAYLOAD} B — the reader is off a record "
                    "boundary and cannot resynchronise")
            if n - pos - REC_HEADER_LEN < length:
                break                          # partial record: keep waiting
            start = pos + REC_HEADER_LEN
            payload = bytes(self._buf[start:start + length])
            pos = start + length
            rec = self._decode(tag, payload)
            if rec is None:
                # Forward compat: step OVER the record by its own length. The
                # length word is why an unknown tag is survivable at all.
                self.n_unknown += 1
                self.unknown_tags[tag] = self.unknown_tags.get(tag, 0) + 1
            else:
                self.n_records += 1
                out.append(rec)
        if pos:
            del self._buf[:pos]
        return out

    # --------------------------------------------------------- overrides
    def _read_header(self, blob: bytes) -> None:
        magic, version = self.HEADER.unpack(blob)[:2]
        self._check_magic(magic)
        if int(version) != self.VERSION:
            raise WireFormatError(
                f"{self.NAME}: protocol version {int(version)}, this build "
                f"speaks {self.VERSION} — the record layout differs, refusing "
                "to guess")
        self.version = int(version)

    def _check_magic(self, magic: bytes) -> None:
        if magic != self.MAGIC:              # 4-byte memcmp, never an int
            raise WireFormatError(
                f"{self.NAME}: expected the {self.MAGIC!r} header, got "
                f"{magic!r}")

    def _decode(self, tag: bytes, payload: bytes):
        raise NotImplementedError


class PoseDecoder(_RecordStream):
    """driver -> station: "POSE" header, then 'TWC ' records.

    The live consumer. Records are tiny (:data:`POSE_PAYLOAD_LEN` bytes plus
    an 8-byte record header), so the payload cap is tightened well below
    :data:`MAX_RECORD_LEN`: on this stream a large length is never legitimate
    and is worth failing on immediately.
    """

    MAGIC = MAGIC_OUT
    VERSION = SLAM_WIRE_VERSION_OUT
    HEADER = _HDR_OUT
    MAX_PAYLOAD = 1 << 20
    NAME = "pose stream"

    def _decode(self, tag: bytes, payload: bytes):
        if tag != TAG_POSE:
            return None
        if len(payload) != POSE_PAYLOAD_LEN:
            raise WireFormatError(
                f"pose stream: 'TWC ' payload is {len(payload)} B, contract "
                f"says {POSE_PAYLOAD_LEN} (4 seq + 8 t + 4 state + 1 "
                "map_changed + 3 rsv + 4 n_tracked + 48 Twc) — one side's "
                "length arithmetic is wrong and the stream is now desynced")
        seq, t, state, changed, n_tracked = _POSE_FIXED.unpack_from(payload, 0)
        twc = np.frombuffer(payload, dtype="<f4", count=12,
                            offset=_POSE_FIXED.size).reshape(3, 4).copy()
        return PoseRecord(seq=int(seq), t=float(t), state=int(state),
                          map_changed=bool(changed), n_tracked=int(n_tracked),
                          Twc=twc)


class DriverStreamDecoder(_RecordStream):
    """station -> driver: "OAKX" header, then 'FRAM' / 'KNOT' / 'CLRO' / 'BYE '.

    The C++ driver is the real consumer of this direction; this decoder is the
    executable statement of what it must see. It is what the round-trip tests
    read back, and it lets a recorded stream be replayed and inspected without
    a SLAM build.

    The legacy "OAKD" header is REFUSED rather than half-supported: that
    stream has no tags at all, so a decoder that accepted the header would
    have to infer every following record from the geometry and could never
    report an unknown tag again.
    """

    MAGIC = MAGIC_IN
    VERSION = SLAM_WIRE_VERSION_IN
    HEADER = _HDR_IN
    MAX_PAYLOAD = MAX_RECORD_LEN
    NAME = "frame stream"

    def __init__(self) -> None:
        super().__init__()
        self.width = 0
        self.height = 0

    def reset(self) -> None:
        super().reset()
        self.width = 0
        self.height = 0

    def _check_magic(self, magic: bytes) -> None:
        if magic == MAGIC_IN_LEGACY:
            raise WireFormatError(
                "frame stream: this is the legacy untagged 'OAKD' stream "
                "(header, then bare seq/t/L/R records with no tag or length). "
                "It carries no records this decoder can step over; feed it to "
                "the driver, which still accepts it, not to DriverStreamDecoder")
        super()._check_magic(magic)

    def _read_header(self, blob: bytes) -> None:
        magic, version, width, height = self.HEADER.unpack(blob)
        self._check_magic(magic)
        if int(version) != self.VERSION:
            raise WireFormatError(
                f"{self.NAME}: protocol version {int(version)}, this build "
                f"speaks {self.VERSION}")
        if width == 0 or height == 0:
            raise WireFormatError(
                f"{self.NAME}: header geometry {width}x{height} is degenerate")
        self.version = int(version)
        self.width = int(width)
        self.height = int(height)

    def _decode(self, tag: bytes, payload: bytes):
        if tag == TAG_FRAME:
            return self._decode_frame(payload)
        if tag == TAG_KNOT:
            return self._decode_knot(payload)
        if tag == TAG_CLEAR:
            self._expect_empty(payload, TAG_CLEAR)
            return ClearRecord()
        if tag == TAG_BYE:
            self._expect_empty(payload, TAG_BYE)
            return ByeRecord()
        return None

    def _expect_empty(self, payload: bytes, tag: bytes) -> None:
        if payload:
            raise WireFormatError(f"{self.NAME}: {tag!r} takes no payload, got "
                                  f"{len(payload)} B")

    def _decode_frame(self, payload: bytes) -> FrameRecord:
        plane = self.width * self.height
        want = FRAME_FIXED_LEN + 2 * plane
        if len(payload) != want:
            raise WireFormatError(
                f"{self.NAME}: 'FRAM' payload is {len(payload)} B, the "
                f"header's {self.width}x{self.height} geometry needs "
                f"{want} = {FRAME_FIXED_LEN} + 2*{plane}")
        seq, t_capture = _FRAME_FIXED.unpack_from(payload, 0)
        off = _FRAME_FIXED.size
        left = np.frombuffer(payload, dtype=np.uint8, count=plane, offset=off)
        right = np.frombuffer(payload, dtype=np.uint8, count=plane,
                              offset=off + plane)
        shape = (self.height, self.width)
        return FrameRecord(seq=int(seq), t_capture=float(t_capture),
                           left=left.reshape(shape).copy(),
                           right=right.reshape(shape).copy())

    def _decode_knot(self, payload: bytes) -> KnotRecord:
        if len(payload) < KNOT_FIXED_LEN + ANCHOR_LEN:
            raise WireFormatError(
                f"{self.NAME}: 'KNOT' payload is {len(payload)} B, shorter "
                f"than the {KNOT_FIXED_LEN} B header + {ANCHOR_LEN} B anchor")
        (plan_id, flags, status, t_emit, n_plan,
         n_raw) = _KNOT_FIXED.unpack_from(payload, 0)
        # The counts are cross-checked against the RECORD LENGTH before a
        # single element is read, so a corrupted count can never drive an
        # allocation or walk off the payload.
        want = KNOT_FIXED_LEN + 12 * (int(n_plan) + int(n_raw)) + ANCHOR_LEN
        if len(payload) != want:
            raise WireFormatError(
                f"{self.NAME}: 'KNOT' says n_plan={int(n_plan)} "
                f"n_raw={int(n_raw)}, which needs {want} B, but the record is "
                f"{len(payload)} B")
        off = _KNOT_FIXED.size
        plan = np.frombuffer(payload, dtype="<f4", count=3 * int(n_plan),
                             offset=off).reshape(int(n_plan), 3).copy()
        off += 12 * int(n_plan)
        raw = np.frombuffer(payload, dtype="<f4", count=3 * int(n_raw),
                            offset=off).reshape(int(n_raw), 3).copy()
        off += 12 * int(n_raw)
        anchor_arr = np.frombuffer(payload, dtype="<f4", count=12,
                                   offset=off).reshape(3, 4).copy()
        anchor = anchor_arr if (int(flags) & KNOT_FLAG_ANCHOR) else None
        return KnotRecord(plan_id=int(plan_id), status=int(status),
                          t_emit=float(t_emit), flags=int(flags),
                          plan=plan, raw=raw, anchor=anchor)
