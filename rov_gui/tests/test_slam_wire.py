#!/usr/bin/env python3
"""test_slam_wire.py — the station <-> ORB-SLAM3 driver pipe codec, offline.

    ~/miniforge3/envs/robust/bin/python rov_gui/tests/test_slam_wire.py

Pure stdlib + numpy (no Qt, no cv2, no SLAM build, no pytest dependency —
plain asserts, so the same file collects under pytest where it exists).

This is the layer where a mistake is not a glitch: neither direction has a
resync marker, so ONE wrong length slides the reader off the record boundary
and every later record — every pose, every overlay — is parsed at the wrong
offset for the rest of the run. What is pinned here:

  * the documented byte arithmetic, recomputed from the field list rather
    than from the module's own constants, above all the pose payload
    4 + 8 + 4 + 1 + 3 + 4 + 48 = 72 that both sides hard-code;
  * a SECOND, hand-rolled transcription of the 'TWC ', 'FRAM' and 'KNOT'
    layouts (independent ``struct.pack`` calls straight off the protocol
    text) compared byte-for-byte with the encoders — an encoder/decoder pair
    that agrees with itself on the WRONG layout passes a round-trip and
    still desyncs against the C++;
  * round-trips of every record type, including the anchor flag, the empty
    raw polyline and a status code outside the known enum;
  * the partial-read property the pipe forces on us: fed ONE BYTE AT A TIME,
    and at random chunk boundaries, the decoder produces exactly what the
    whole-buffer decode produced;
  * a truncated tail yields nothing and stays buffered until the rest
    arrives;
  * an unknown tag is stepped over by its length and the stream stays in
    sync (the forward-compat rule both sides promise);
  * every corruption that must be LOUD instead of silent: wrong magic, the
    legacy untagged "OAKD" stream, a version bump, a 71-byte pose payload,
    a knot whose counts disagree with its length, a frame that disagrees
    with the header geometry, and an absurd length word;
  * the encoder's input gates — wrong dtype, wrong shape, non-contiguous
    plane, mismatched stereo pair, (3, N) knots, non-finite values.
"""

from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rov_gui.control.slam_wire import (  # noqa: E402
    ANCHOR_LEN, FRAME_FIXED_LEN, HEADER_IN_LEN, HEADER_OUT_LEN,
    KNOT_FIXED_LEN, MAGIC_IN, MAGIC_IN_LEGACY, MAGIC_OUT, MAX_RECORD_LEN,
    POSE_PAYLOAD_LEN, REC_HEADER_LEN, SLAM_WIRE_VERSION_IN,
    SLAM_WIRE_VERSION_OUT, STATUS_NAMES, TAG_BYE, TAG_CLEAR, TAG_FRAME,
    TAG_KNOT, TAG_POSE, TRACKING_OK, ByeRecord, ClearRecord,
    DriverStreamDecoder, FrameRecord, KnotRecord, KnotStatus, PoseDecoder,
    PoseRecord, WireFormatError, encode_bye, encode_clear, encode_frame,
    encode_header, encode_knots, encode_pose, encode_pose_header)

W, H = 8, 6                      # a tiny frame: the codec does not care, and
#                                  a 640x400 plane in a unit test buys nothing


# --------------------------------------------------------------------- utils
def _img(seed: int) -> np.ndarray:
    """A deterministic (H, W) uint8 plane."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(H, W), dtype=np.uint8)


def _twc(seed: int) -> np.ndarray:
    """A (3, 4) [R|t] whose entries survive float32 exactly (powers of two and
    halves), so a byte comparison is not fighting rounding."""
    rng = np.random.default_rng(seed)
    vals = rng.integers(-8, 9, size=(3, 4)).astype(np.float32) / 4.0
    return np.ascontiguousarray(vals, dtype=np.float32)


def _knots(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(
        (rng.integers(-64, 65, size=(n, 3)) / 8.0).astype(np.float32))


def _raises(fn, exc, needle: str = "") -> str:
    """Call ``fn`` expecting ``exc``; returns the message (asserts on it)."""
    try:
        fn()
    except exc as e:                                      # noqa: BLE001
        msg = str(e)
        if needle:
            assert needle in msg, f"message {msg!r} lacks {needle!r}"
        return msg
    raise AssertionError(f"expected {exc.__name__}, nothing raised")


def _rec_eq(a, b) -> bool:
    """Structural equality — the dataclasses hold arrays, so ``==`` on them
    would raise the ambiguous-truth-value error instead of comparing."""
    if type(a) is not type(b):
        return False
    if isinstance(a, PoseRecord):
        return (a.seq == b.seq and a.t == b.t and a.state == b.state
                and a.map_changed == b.map_changed
                and a.n_tracked == b.n_tracked
                and np.array_equal(a.Twc, b.Twc))
    if isinstance(a, FrameRecord):
        return (a.seq == b.seq and a.t_capture == b.t_capture
                and np.array_equal(a.left, b.left)
                and np.array_equal(a.right, b.right))
    if isinstance(a, KnotRecord):
        same_anchor = ((a.anchor is None and b.anchor is None)
                       or (a.anchor is not None and b.anchor is not None
                           and np.array_equal(a.anchor, b.anchor)))
        return (a.plan_id == b.plan_id and a.status == b.status
                and a.t_emit == b.t_emit and a.flags == b.flags
                and np.array_equal(a.plan, b.plan)
                and np.array_equal(a.raw, b.raw) and same_anchor)
    return True                                  # ClearRecord / ByeRecord


def _mixed_in_stream() -> bytes:
    """A whole station->driver session: header, frames, both knot flavours,
    a clear, a goodbye."""
    return b"".join((
        encode_header(W, H),
        encode_frame(0, 1.5, _img(1), _img(2)),
        encode_knots(7, KnotStatus.ACCEPT, 2.25, _knots(16, 3), _knots(16, 4),
                     anchor=_twc(5)),
        encode_knots(8, KnotStatus.REJECT, 2.75, _knots(16, 6)),
        encode_clear(),
        encode_frame(1, 3.5, _img(7), _img(8)),
        encode_bye()))


def _mixed_out_stream(n: int = 5) -> bytes:
    parts = [encode_pose_header()]
    for i in range(n):
        parts.append(encode_pose(i, 100.0 + 0.25 * i, TRACKING_OK if i else 1,
                                 bool(i % 2), 300 + i, _twc(10 + i)))
    return b"".join(parts)


# ====================================================================== sizes
def test_layout_arithmetic():
    """Every documented length, recomputed from the FIELD LIST here rather
    than from the module's constants — a constant that agrees with itself
    proves nothing about the C++."""
    assert HEADER_IN_LEN == 4 + 4 + 4 + 4 == 16          # magic, ver, w, h
    assert HEADER_OUT_LEN == 4 + 4 == 8                  # magic, ver
    assert REC_HEADER_LEN == 4 + 4 == 8                  # tag, len
    assert FRAME_FIXED_LEN == 4 + 8 == 12                # seq, t_capture
    # plan_id, flags, status, rsv[3], t_emit, n_plan, n_raw
    assert KNOT_FIXED_LEN == 4 + 4 + 1 + 3 + 8 + 4 + 4 == 28
    assert ANCHOR_LEN == 12 * 4 == 48                    # f32 [R|t]
    # seq, t, state, map_changed, rsv[3], n_tracked, Twc[12]
    assert POSE_PAYLOAD_LEN == 4 + 8 + 4 + 1 + 3 + 4 + 48
    assert POSE_PAYLOAD_LEN == 72, "both sides hard-code 72"
    assert MAX_RECORD_LEN > FRAME_FIXED_LEN + 2 * 4096 * 4096

    # ...and the encoders emit exactly those lengths.
    hdr = encode_header(W, H)
    assert len(hdr) == HEADER_IN_LEN
    assert len(encode_pose_header()) == HEADER_OUT_LEN
    frame = encode_frame(0, 0.0, _img(1), _img(2))
    assert len(frame) == REC_HEADER_LEN + FRAME_FIXED_LEN + 2 * W * H
    knot = encode_knots(1, 0, 0.0, _knots(16, 1), _knots(16, 2), _twc(3))
    assert len(knot) == REC_HEADER_LEN + KNOT_FIXED_LEN + 12 * 32 + ANCHOR_LEN
    assert len(encode_clear()) == REC_HEADER_LEN
    assert len(encode_bye()) == REC_HEADER_LEN
    pose = encode_pose(0, 0.0, 2, False, 0, _twc(4))
    assert len(pose) == REC_HEADER_LEN + 72

    # The length WORD inside each record header, read back independently.
    for blob, want in ((frame, FRAME_FIXED_LEN + 2 * W * H),
                       (knot, KNOT_FIXED_LEN + 12 * 32 + ANCHOR_LEN),
                       (encode_clear(), 0), (encode_bye(), 0),
                       (pose, 72)):
        tag, length = struct.unpack("<4sI", blob[:8])
        assert len(tag) == 4 and length == want, (tag, length, want)
        assert len(blob) == 8 + length


def test_tags_and_magics_are_four_bytes():
    """Four ASCII bytes each, and the two tags with a TRAILING SPACE keep
    it — 'BYE' or 'TWC' at three bytes would shift every later field."""
    for tag in (TAG_FRAME, TAG_KNOT, TAG_CLEAR, TAG_BYE, TAG_POSE):
        assert isinstance(tag, bytes) and len(tag) == 4
        assert all(32 <= b < 127 for b in tag), tag
    assert TAG_BYE == b"BYE " and TAG_POSE == b"TWC "
    assert TAG_BYE.endswith(b" ") and TAG_POSE.endswith(b" ")
    for magic in (MAGIC_IN, MAGIC_IN_LEGACY, MAGIC_OUT):
        assert len(magic) == 4
    assert MAGIC_IN == b"OAKX" and MAGIC_OUT == b"POSE"
    assert MAGIC_IN != MAGIC_IN_LEGACY
    assert SLAM_WIRE_VERSION_IN == 2 and SLAM_WIRE_VERSION_OUT == 1


def test_status_enum_values():
    """0..4, in the documented order, and int-valued so ``struct`` takes
    them as the wire's u8 without a cast at every call site."""
    assert [int(s) for s in KnotStatus] == [0, 1, 2, 3, 4]
    assert (KnotStatus.ACCEPT, KnotStatus.CLIP, KnotStatus.REJECT,
            KnotStatus.LATE, KnotStatus.SKIPPED) == (0, 1, 2, 3, 4)
    assert isinstance(KnotStatus.ACCEPT, int)
    assert STATUS_NAMES == ("accept", "clip", "reject", "late", "skipped")
    for s in KnotStatus:
        assert STATUS_NAMES[int(s)] == s.name.lower()


# ============================================ independent layout transcription
def test_pose_bytes_match_a_handrolled_record():
    """The 'TWC ' bytes, packed field by field straight off the protocol
    text. If this and ``encode_pose`` ever disagree, the C++ agrees with one
    of them and the pipe desyncs on record 1."""
    Twc = _twc(11)
    seq, t, state, n_tracked = 4242, 12.5, 2, 917
    body = (struct.pack("<I", seq) + struct.pack("<d", t)
            + struct.pack("<i", state) + struct.pack("<B", 1) + b"\x00" * 3
            + struct.pack("<I", n_tracked)
            + struct.pack("<12f", *Twc.reshape(-1).tolist()))
    assert len(body) == 72
    hand = struct.pack("<4sI", b"TWC ", 72) + body
    assert encode_pose(seq, t, state, True, n_tracked, Twc) == hand
    # and the rsv bytes really are zeros (the C++ reads 3 bytes it ignores)
    assert hand[8 + 17:8 + 20] == b"\x00\x00\x00"


def test_frame_and_knot_bytes_match_handrolled_records():
    """Same independent transcription for the two variable-length records."""
    left, right = _img(21), _img(22)
    hand_frame = (struct.pack("<4sI", b"FRAM", 12 + 2 * W * H)
                  + struct.pack("<I", 5) + struct.pack("<d", 9.25)
                  + left.tobytes() + right.tobytes())
    assert encode_frame(5, 9.25, left, right) == hand_frame

    plan, raw, anchor = _knots(3, 23), _knots(2, 24), _twc(25)
    payload = (struct.pack("<I", 77) + struct.pack("<I", 1)
               + struct.pack("<B", 1) + b"\x00" * 3 + struct.pack("<d", 4.5)
               + struct.pack("<I", 3) + struct.pack("<I", 2)
               + struct.pack("<9f", *plan.reshape(-1).tolist())
               + struct.pack("<6f", *raw.reshape(-1).tolist())
               + struct.pack("<12f", *anchor.reshape(-1).tolist()))
    assert len(payload) == 28 + 12 * 5 + 48
    hand_knot = struct.pack("<4sI", b"KNOT", len(payload)) + payload
    assert encode_knots(77, KnotStatus.CLIP, 4.5, plan, raw, anchor) == hand_knot

    assert encode_clear() == struct.pack("<4sI", b"CLRO", 0)
    assert encode_bye() == struct.pack("<4sI", b"BYE ", 0)
    assert encode_header(W, H) == struct.pack("<4sIII", b"OAKX", 2, W, H)
    assert encode_pose_header() == struct.pack("<4sI", b"POSE", 1)


def test_little_endian_is_explicit():
    """The magic must land byte-for-byte as written (no word-swapping) and a
    multi-byte field must be little-endian regardless of the host."""
    assert encode_header(1, 0x02010000)[:4] == b"OAKX"
    assert encode_header(0x04030201, 1)[8:12] == b"\x01\x02\x03\x04"


# ================================================================ round-trips
def test_roundtrip_station_to_driver():
    dec = DriverStreamDecoder()
    recs = dec.feed(_mixed_in_stream())
    assert dec.version == SLAM_WIRE_VERSION_IN
    assert (dec.width, dec.height) == (W, H)
    assert [type(r).__name__ for r in recs] == [
        "FrameRecord", "KnotRecord", "KnotRecord", "ClearRecord",
        "FrameRecord", "ByeRecord"]
    assert dec.pending == 0 and dec.n_unknown == 0 and dec.n_records == 6
    assert isinstance(recs[0], FrameRecord) and isinstance(recs[1], KnotRecord)
    assert isinstance(recs[3], ClearRecord) and isinstance(recs[5], ByeRecord)

    f0 = recs[0]
    assert f0.seq == 0 and f0.t_capture == 1.5
    assert np.array_equal(f0.left, _img(1)) and np.array_equal(f0.right, _img(2))
    assert f0.left.dtype == np.uint8 and f0.left.shape == (H, W)

    k0 = recs[1]
    assert k0.plan_id == 7 and k0.status == int(KnotStatus.ACCEPT)
    assert k0.t_emit == 2.25 and k0.flags == 1
    assert np.array_equal(k0.plan, _knots(16, 3))
    assert np.array_equal(k0.raw, _knots(16, 4))
    assert k0.anchor is not None and np.array_equal(k0.anchor, _twc(5))
    assert k0.plan.dtype == np.float32 and k0.plan.shape == (16, 3)
    assert k0.status_name == "accept"

    k1 = recs[2]
    assert k1.flags == 0 and k1.anchor is None      # identity != "present"
    assert k1.raw.shape == (0, 3), "no raw polyline is empty, not a copy"
    assert k1.status_name == "reject"


def test_roundtrip_driver_to_station():
    dec = PoseDecoder()
    recs = dec.feed(_mixed_out_stream(3))
    assert dec.version == SLAM_WIRE_VERSION_OUT and dec.pending == 0
    assert len(recs) == 3 and all(isinstance(r, PoseRecord) for r in recs)
    r1 = recs[1]
    assert r1.seq == 1 and r1.t == 100.25 and r1.state == TRACKING_OK
    assert r1.map_changed is True and r1.n_tracked == 301
    assert np.array_equal(r1.Twc, _twc(11))
    assert recs[0].state == 1 and recs[0].map_changed is False
    assert recs[0].tracking_ok is False and r1.tracking_ok is True


def test_pose_record_helpers():
    """``R_wc`` / ``t_wc`` (not ``R`` / ``t``: ``t`` is the timestamp) and the
    4x4 promotion."""
    Twc = np.array([[0.0, 0.0, 1.0, 2.0],
                    [1.0, 0.0, 0.0, -3.0],
                    [0.0, 1.0, 0.0, 0.5]], dtype=np.float32)
    rec = PoseDecoder_single(encode_pose(9, 1.0, 2, False, 12, Twc))
    assert np.array_equal(rec.R_wc, Twc[:, :3])
    assert np.array_equal(rec.t_wc, np.array([2.0, -3.0, 0.5], np.float32))
    assert abs(float(np.linalg.det(rec.R_wc)) - 1.0) < 1e-6
    T = rec.as_4x4()
    assert T.shape == (4, 4) and np.array_equal(T[:3, :4], Twc)
    assert np.array_equal(T[3], np.array([0, 0, 0, 1], np.float32))


def PoseDecoder_single(record_bytes: bytes) -> PoseRecord:
    """Decode one 'TWC ' record with a fresh decoder (helper, not a test)."""
    dec = PoseDecoder()
    out = dec.feed(encode_pose_header() + record_bytes)
    assert len(out) == 1
    return out[0]


def test_unknown_status_code_survives_the_round_trip():
    """A code outside the enum must still DRAW as 'not accepted', so it is
    passed through, not clamped or rejected (PolicyOverlay.h keeps an int)."""
    dec = DriverStreamDecoder()
    recs = dec.feed(encode_header(W, H)
                    + encode_knots(1, 9, 0.5, _knots(2, 30)))
    assert recs[0].status == 9 and recs[0].status_name == "code9"
    assert 9 not in [int(s) for s in KnotStatus]


def test_anchor_accepts_4x4_and_defaults_to_identity():
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = (1.0, 2.0, 3.0)
    a = encode_knots(1, 0, 0.0, _knots(2, 31), anchor=T)
    b = encode_knots(1, 0, 0.0, _knots(2, 31), anchor=T[:3, :4])
    assert a == b, "(4, 4) must drop the constant bottom row, not reorder"
    # anchor=None still ships 48 bytes (identity) with the flag CLEAR, so the
    # record length has no branch in it.
    none_rec = encode_knots(1, 0, 0.0, _knots(2, 31))
    assert len(none_rec) == len(a)
    dec = DriverStreamDecoder()
    recs = dec.feed(encode_header(W, H) + none_rec)
    assert recs[0].anchor is None and recs[0].flags == 0
    ident = np.frombuffer(none_rec[-48:], dtype="<f4").reshape(3, 4)
    assert np.array_equal(ident, np.hstack([np.eye(3, dtype=np.float32),
                                            np.zeros((3, 1), np.float32)]))


# ============================================================ partial reads
def _decode_in_chunks(dec, blob: bytes, sizes):
    out = []
    pos = 0
    for n in sizes:
        out.extend(dec.feed(blob[pos:pos + n]))
        pos += n
    if pos < len(blob):
        out.extend(dec.feed(blob[pos:]))
    return out


def test_one_byte_at_a_time_matches_whole_buffer():
    """The property the pipe forces on us: the kernel picks the chunk
    boundaries, so the decoder must not depend on them. One byte per feed is
    the worst case — it splits every header, every length word and every
    float."""
    for blob, factory in ((_mixed_out_stream(4), PoseDecoder),
                          (_mixed_in_stream(), DriverStreamDecoder)):
        whole = factory().feed(blob)
        drip = factory()
        one_at_a_time = _decode_in_chunks(drip, blob, [1] * len(blob))
        assert len(one_at_a_time) == len(whole)
        for a, b in zip(one_at_a_time, whole):
            assert _rec_eq(a, b), (type(a).__name__, type(b).__name__)
        assert drip.pending == 0


def test_random_chunk_boundaries_match_whole_buffer():
    rng = np.random.default_rng(20260903)
    for blob, factory in ((_mixed_out_stream(6), PoseDecoder),
                          (_mixed_in_stream(), DriverStreamDecoder)):
        whole = factory().feed(blob)
        for _ in range(20):
            sizes = rng.integers(0, 37, size=len(blob) + 8).tolist()
            dec = factory()
            got = _decode_in_chunks(dec, blob, sizes)
            assert len(got) == len(whole)
            assert all(_rec_eq(a, b) for a, b in zip(got, whole))
            assert dec.pending == 0


def test_truncated_tail_yields_nothing_and_stays_buffered():
    """A half-arrived record must be HELD, not guessed at and not dropped."""
    blob = _mixed_out_stream(3)
    cut = len(blob) - 30                       # inside the last record
    dec = PoseDecoder()
    got = dec.feed(blob[:cut])
    assert len(got) == 2
    assert dec.pending == (REC_HEADER_LEN + POSE_PAYLOAD_LEN) - 30
    rest = dec.feed(blob[cut:])
    assert len(rest) == 1 and dec.pending == 0
    assert _rec_eq(rest[0], PoseDecoder().feed(blob)[-1])

    # a truncated HEADER is equally patient
    part = DriverStreamDecoder()
    assert part.feed(_mixed_in_stream()[:HEADER_IN_LEN - 1]) == []
    assert part.header_seen is False and part.pending == HEADER_IN_LEN - 1
    # ...and an empty feed is a no-op, not a parse attempt
    assert part.feed(b"") == [] and part.pending == HEADER_IN_LEN - 1


def test_partial_record_header_is_held():
    """Fewer than 8 bytes cannot even name a tag; nothing may be inferred."""
    blob = _mixed_out_stream(2)
    dec = PoseDecoder()
    got = dec.feed(blob[:HEADER_OUT_LEN + REC_HEADER_LEN - 1])
    assert got == [] and dec.pending == REC_HEADER_LEN - 1
    assert dec.feed(blob[HEADER_OUT_LEN + REC_HEADER_LEN - 1:])


def test_reset_forgets_the_header():
    """A restarted driver sends its header again; a decoder that kept
    ``header_seen`` would read 'POSE' as a tag and desync immediately."""
    dec = PoseDecoder()
    assert len(dec.feed(_mixed_out_stream(2))) == 2
    dec.reset()
    assert dec.header_seen is False and dec.pending == 0
    assert len(dec.feed(_mixed_out_stream(3))) == 3


# ========================================================== forward compat
def test_unknown_tag_is_skipped_and_the_stream_stays_in_sync():
    """The length word is what makes an unknown record survivable: step over
    it and keep parsing. The poses on BOTH sides of it must decode."""
    payload = b"\xde\xad\xbe\xef" * 9                    # 36 bytes, no meaning
    unknown = struct.pack("<4sI", b"XTRA", len(payload)) + payload
    stream = (encode_pose_header()
              + encode_pose(0, 1.0, 2, False, 10, _twc(41))
              + unknown
              + encode_pose(1, 2.0, 2, True, 11, _twc(42)))
    dec = PoseDecoder()
    recs = dec.feed(stream)
    assert len(recs) == 2 and dec.n_unknown == 1 and dec.n_records == 2
    assert dec.unknown_tags == {b"XTRA": 1} and dec.pending == 0
    assert recs[0].seq == 0 and recs[1].seq == 1
    assert np.array_equal(recs[1].Twc, _twc(42))

    # ...and identically when the chunk boundaries fall inside the stranger
    drip = PoseDecoder()
    got = _decode_in_chunks(drip, stream, [1] * len(stream))
    assert len(got) == 2 and drip.n_unknown == 1
    assert all(_rec_eq(a, b) for a, b in zip(got, recs))

    # an unknown record on the other direction too, with a zero length
    empty_unknown = struct.pack("<4sI", b"ZZZZ", 0)
    d2 = DriverStreamDecoder()
    r2 = d2.feed(encode_header(W, H) + empty_unknown
                 + encode_frame(3, 1.0, _img(43), _img(44)))
    assert len(r2) == 1 and d2.n_unknown == 1 and r2[0].seq == 3


# ============================================================== corruption
def test_bad_magic_and_version_are_refused():
    _raises(lambda: PoseDecoder().feed(struct.pack("<4sI", b"NOPE", 1)),
            WireFormatError, "POSE")
    _raises(lambda: PoseDecoder().feed(struct.pack("<4sI", b"POSE", 2)),
            WireFormatError, "version")
    _raises(lambda: DriverStreamDecoder().feed(
        struct.pack("<4sIII", b"OAKX", 3, W, H)), WireFormatError, "version")
    _raises(lambda: DriverStreamDecoder().feed(
        struct.pack("<4sIII", b"OAKX", 2, 0, H)), WireFormatError, "degenerate")


def test_legacy_oakd_header_is_refused_with_a_reason():
    """The legacy stream is untagged, so there is nothing to step over — say
    so rather than half-parse it into a desync."""
    msg = _raises(lambda: DriverStreamDecoder().feed(
        struct.pack("<4sIII", MAGIC_IN_LEGACY, 2, W, H)),
        WireFormatError, "OAKD")
    assert "legacy" in msg.lower()


def test_wrong_pose_payload_length_is_fatal_and_names_72():
    """The single most destructive failure mode: a payload length that is not
    72 means one side's arithmetic is wrong and every later record is lost."""
    bad = struct.pack("<4sI", b"TWC ", 71) + b"\x00" * 71
    msg = _raises(lambda: PoseDecoder().feed(encode_pose_header() + bad),
                  WireFormatError, "72")
    assert "71" in msg


def test_absurd_length_fails_instead_of_buffering_forever():
    """A desynced reader gets garbage in the length word. Failing at the
    record where sync was lost beats buffering until the machine dies."""
    huge = struct.pack("<4sI", b"TWC ", 1 << 30)
    _raises(lambda: PoseDecoder().feed(encode_pose_header() + huge),
            WireFormatError, "cannot resynchronise")
    over = struct.pack("<4sI", b"FRAM", MAX_RECORD_LEN + 1)
    _raises(lambda: DriverStreamDecoder().feed(encode_header(W, H) + over),
            WireFormatError, "cap")


def test_frame_length_must_match_the_header_geometry():
    """The header is the only place the geometry is stated; a frame that
    disagrees with it would reshape into garbage (or crash)."""
    good = encode_frame(0, 0.0, _img(51), _img(52))
    other = DriverStreamDecoder()
    msg = _raises(lambda: other.feed(struct.pack("<4sIII", b"OAKX", 2, W + 1, H)
                                     + good), WireFormatError, "FRAM")
    assert str(FRAME_FIXED_LEN + 2 * (W + 1) * H) in msg


def test_knot_counts_must_agree_with_the_record_length():
    """A corrupted n_plan must be caught by the LENGTH cross-check before a
    single element is read — never trusted enough to size an allocation."""
    rec = encode_knots(1, 0, 0.0, _knots(4, 61), _knots(4, 62), _twc(63))
    n_plan_off = REC_HEADER_LEN + 4 + 4 + 1 + 3 + 8          # = 28
    assert struct.unpack_from("<I", rec, n_plan_off)[0] == 4
    tampered = bytearray(rec)
    struct.pack_into("<I", tampered, n_plan_off, 4096)
    msg = _raises(lambda: DriverStreamDecoder().feed(encode_header(W, H)
                                                    + bytes(tampered)),
                  WireFormatError, "n_plan=4096")
    assert "KNOT" in msg
    short = struct.pack("<4sI", b"KNOT", 4) + b"\x00" * 4
    _raises(lambda: DriverStreamDecoder().feed(encode_header(W, H) + short),
            WireFormatError, "shorter")


def test_empty_records_reject_a_payload():
    fake = struct.pack("<4sI", b"CLRO", 2) + b"\x00\x00"
    _raises(lambda: DriverStreamDecoder().feed(encode_header(W, H) + fake),
            WireFormatError, "no payload")


# ========================================================= encoder gates
def test_image_gates():
    """dtype, rank, emptiness, contiguity and the stereo pair, each with a
    message that names what to fix."""
    ok = _img(71)
    _raises(lambda: encode_frame(0, 0.0, ok.astype(np.uint16), ok),
            ValueError, "uint8")
    _raises(lambda: encode_frame(0, 0.0, ok[None], ok), ValueError, "2-D")
    _raises(lambda: encode_frame(0, 0.0, np.zeros((0, 0), np.uint8),
                                 np.zeros((0, 0), np.uint8)),
            ValueError, "empty")
    view = np.ascontiguousarray(np.zeros((H, 2 * W), np.uint8))[:, ::2]
    assert not view.flags["C_CONTIGUOUS"] and view.shape == (H, W)
    msg = _raises(lambda: encode_frame(0, 0.0, view, ok), ValueError,
                  "C-contiguous")
    assert "ascontiguousarray" in msg
    _raises(lambda: encode_frame(0, 0.0, ok, _img(72)[:H - 1]),
            ValueError, "same shape")
    # a contiguous COPY of the same view encodes fine
    assert encode_frame(0, 0.0, np.ascontiguousarray(view), ok)


def test_knot_gates():
    _raises(lambda: encode_knots(1, 0, 0.0, np.zeros((3, 5), np.float32)),
            ValueError, "(N, 3)")
    msg = _raises(lambda: encode_knots(1, 0, 0.0, np.zeros((3, 16),
                                                           np.float32)),
                  ValueError, ".T")
    assert "(3, 16)" in msg, "the error must name the offending shape"
    bad = _knots(4, 81).copy()
    bad[2, 1] = math.nan
    _raises(lambda: encode_knots(1, 0, 0.0, bad), ValueError, "non-finite")
    _raises(lambda: encode_knots(1, 0, 0.0, _knots(4, 82),
                                 anchor=np.zeros((2, 4), np.float32)),
            ValueError, "(3, 4)")
    _raises(lambda: encode_knots(1, 300, 0.0, _knots(4, 83)),
            ValueError, "u8")
    _raises(lambda: encode_knots(-1, 0, 0.0, _knots(4, 84)),
            ValueError, "u32")
    # an empty plan is legal (a skipped inference still wants its caption)
    dec = DriverStreamDecoder()
    recs = dec.feed(encode_header(W, H)
                    + encode_knots(5, KnotStatus.SKIPPED, 1.0, None))
    assert recs[0].plan.shape == (0, 3) and recs[0].raw.shape == (0, 3)


def test_pose_and_header_gates():
    _raises(lambda: encode_pose(0, 0.0, 2, False, 0, np.zeros((2, 4),
                                                             np.float32)),
            ValueError, "(3, 4)")
    nan = _twc(91).copy()
    nan[1, 3] = math.inf
    _raises(lambda: encode_pose(0, 0.0, 2, False, 0, nan),
            ValueError, "non-finite")
    _raises(lambda: encode_pose(0, 0.0, 2, False, -1, _twc(92)),
            ValueError, "u32")
    _raises(lambda: encode_header(0, H), ValueError, "non-zero")
    _raises(lambda: encode_header(2 ** 32, H), ValueError, "u32")


# ------------------------------------------------------------------- runner
def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as e:                                # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
