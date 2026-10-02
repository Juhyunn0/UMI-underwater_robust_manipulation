"""The wire between ``pool_cam.py --serve`` and the control station (rov_gui).

Standard library only, on purpose: rov_gui imports this module on its GUI
thread's side, and pulling cv2 in through it would repoint Qt's plugin path
(rov_gui/qt.py, "cv2 hijacks QT_QPA_PLATFORM_PLUGIN_PATH").

Two directions, two formats:

  station -> pool_cam   stdin, one JSON object per line ("commands")
                            {"cmd": "record", "on": true, "t0": 1234.5, "outdir": "..."}
                            {"cmd": "record", "on": false}
                            {"cmd": "arm", "cam": "cam0", "on": false}
                            {"cmd": "preview", "sizes": {"cam0": [640, 360]}, "hz": 10}
                            {"cmd": "quit"}            (or just close stdin)

  pool_cam -> station   stdout, framed messages: an 8-byte header (JSON length,
                        payload length; big-endian uint32), the JSON, the payload
                            {"t": "hello", "cams": [...], "encoder": ...}
                            {"t": "status", "cams": [...], "session": ...}
                            {"t": "log", "level": "info", "msg": ...}
                            {"t": "frame", "cam": "cam0", "w": W, "h": H, ...} + W*H*3 BGR bytes
                            {"t": "bye", ...}

Previews travel as raw BGR rather than JPEG: at tile size that is ~1 MB a
frame, and copying it is cheaper for the station than decoding a JPEG would be
-- the station's side is the one that must stay light.
"""

import json
import struct

HEADER = struct.Struct(">II")
MAX_JSON = 1 << 20          # a status message is ~2 KB; anything near this is garbage
MAX_PAYLOAD = 16 << 20      # a 1280x720 BGR preview is 2.8 MB


def pack(msg, payload=b""):
    """One framed message, ready to write."""
    head = json.dumps(msg, separators=(",", ":")).encode()
    return HEADER.pack(len(head), len(payload)) + head + payload


def _read_exact(stream, n):
    """n bytes, or None at end of stream (also mid-message: a dead writer)."""
    chunks, got = [], 0
    while got < n:
        part = stream.read(n - got)
        if not part:
            return None
        chunks.append(part)
        got += len(part)
    return b"".join(chunks)


def read_message(stream):
    """(dict, payload bytes) from a binary stream, or None once it has ended.

    Raises ValueError on a header that cannot be a message -- the stream is out
    of step and nothing after it can be trusted."""
    raw = _read_exact(stream, HEADER.size)
    if raw is None:
        return None
    n_json, n_payload = HEADER.unpack(raw)
    if n_json > MAX_JSON or n_payload > MAX_PAYLOAD:
        raise ValueError(f"bad frame header ({n_json}, {n_payload})")
    head = _read_exact(stream, n_json)
    payload = _read_exact(stream, n_payload) if n_payload else b""
    if head is None or payload is None:
        return None
    return json.loads(head), payload


def command(cmd, **fields):
    """One command line for pool_cam's stdin."""
    return (json.dumps(dict(cmd=cmd, **fields), separators=(",", ":")) + "\n").encode()


def parse_command(line):
    """A command line -> dict, or None if it is not one (ignored, not fatal)."""
    try:
        msg = json.loads(line)
    except (ValueError, UnicodeDecodeError):
        return None
    return msg if isinstance(msg, dict) and isinstance(msg.get("cmd"), str) else None
