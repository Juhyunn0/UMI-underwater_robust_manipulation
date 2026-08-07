#!/usr/bin/env python3
"""
preflight.py — one screen that says whether the rig is ready, before anything connects.

Why this exists
---------------
The failures in this stack all look the same from the operator's chair — "it
didn't work" — but they live in five different places, and until now each one
announced itself at a different moment, some of them only after a 12 s PoE boot:

    wrong interpreter          conda base has depthai 3.x, which claims the camera
                               merely by constructing a Pipeline
    wrong network path         traffic leaving over a VPN / the wrong NIC
    camera has an owner        Madrona (or another c3 tool) holds the one pipeline
                               an OAK device allows
    no MAVLink to the recorder BlueOS pushes 14550 and QGroundControl holds it, so
                               a second endpoint is needed or the dataset has no
                               telemetry
    no room / no display       the run dies twenty minutes in, or at the first imshow

Preflight asks all of them up front, in a couple of seconds, and prints a table
with the fix beside anything that is not OK.

    ./c3 preflight                       # just the report
    ./c3 preflight --json-out pf.json    # machine readable
    ./c3 collect --profile research_near # runs it automatically first

What it deliberately does NOT do
--------------------------------
**It never opens the camera.** Everything here stays at the TCP / UDP-discovery
level, which costs milliseconds and takes nothing from anybody. Opening the
device means uploading firmware and booting it (`device.TYPICAL_POE_BOOT_S`) and
claiming the single pipeline slot, so a check that did it would be slower than
the thing it is checking and would itself become a reason the next tool fails.
That is what `discover_c3.py --probe` is for — run it when preflight says the
camera is free but you want the sensor list and calibration too.

It also never commands the vehicle. The MAVLink checks read; they do not send.

Severity
--------
    OK      nothing to do
    WARN    the run can proceed but something is degraded (no telemetry, no
            window, unexpected mx_id). Printed with a fix; does not block.
    FAIL    the run cannot succeed (camera owned, link down, disk full). Blocks,
            unless --no-preflight.
    SKIP    not applicable to this invocation (e.g. telemetry checks under
            --mavlink-transport none)

`--preflight-strict` promotes every WARN to blocking, for unattended runs where
a dataset without telemetry is not worth the dive.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
import textwrap
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from c3_camera import config as C
from c3_camera import device as D

# BlueOS lives at a fixed address on this rig; both of these already exist as
# module constants next to the code that talks to them, and are imported rather
# than re-typed so there is one place to change if the vehicle moves.
from c3_camera.blueos_endpoint import DEFAULT_HOST as BLUEOS_HOST
from c3_camera.madrona import MADRONA_ID

# MAVLink2Rest's port on BlueOS. Same default as MavlinkLogger's --mavlink-rest-url.
MAVLINK_REST_PORT = 6040

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"

# Exit code a tool returns when preflight blocked it. Distinct from the codes the
# tools already use for their own failures (1 error, 2 bad arguments, 3 device
# busy) so a script can tell "never started" from "started and failed".
EXIT_PREFLIGHT = 5

_LABEL_W = 18
_RULE_W = 78


# =============================================================================
# result types
# =============================================================================
@dataclass
class Check:
    """One question, its answer, and what to do if the answer is bad."""

    name: str
    status: str
    detail: str = ""
    fix: tuple[str, ...] = ()
    data: dict = field(default_factory=dict)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        out = {"name": self.name, "status": self.status, "detail": self.detail,
               "elapsed_s": round(self.elapsed_s, 3)}
        if self.fix:
            out["fix"] = list(self.fix)
        if self.data:
            out["data"] = self.data
        return out


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)
    elapsed_s: float = 0.0
    title: str = ""

    def count(self, status: str) -> int:
        return sum(1 for c in self.checks if c.status == status)

    @property
    def ok(self) -> bool:
        """No FAIL. Warnings are not failures unless the caller says so."""
        return self.count(FAIL) == 0

    def blocking(self, strict: bool = False) -> list[Check]:
        bad = {FAIL, WARN} if strict else {FAIL}
        return [c for c in self.checks if c.status in bad]

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "elapsed_s": round(self.elapsed_s, 3),
            "verdict": "ready" if self.ok else "not_ready",
            "counts": {s: self.count(s) for s in (OK, WARN, FAIL, SKIP)},
            "checks": [c.to_dict() for c in self.checks],
        }


# Convenience constructors, so a check body reads as one return statement.
def _ok(detail: str, **data) -> Check:
    return Check("", OK, detail, data=data)


def _warn(detail: str, *fix: str, **data) -> Check:
    return Check("", WARN, detail, fix=tuple(fix), data=data)


def _fail(detail: str, *fix: str, **data) -> Check:
    return Check("", FAIL, detail, fix=tuple(fix), data=data)


def _skip(detail: str, **data) -> Check:
    return Check("", SKIP, detail, data=data)


# =============================================================================
# rendering
# =============================================================================
_COLOURS = {OK: "\033[32m", WARN: "\033[33m", FAIL: "\033[31m", SKIP: "\033[90m"}
_RESET = "\033[0m"


def _use_colour(stream) -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    try:
        return bool(stream.isatty())
    except Exception:  # noqa: BLE001
        return False


def _badge(status: str, colour: bool) -> str:
    text = {OK: "OK  ", WARN: "WARN", FAIL: "FAIL", SKIP: "skip"}[status]
    return f"{_COLOURS[status]}{text}{_RESET}" if colour else text


def _wrap(text: str, indent: int, width: int) -> list[str]:
    if not text:
        return [""]
    lines: list[str] = []
    for para in text.split("\n"):
        lines += textwrap.wrap(para, width=max(20, width - indent)) or [""]
    return lines


# =============================================================================
# the runner
# =============================================================================
def run(specs, title: str = "preflight", stream=None,
        show_ok_fix: bool = False) -> Report:
    """Run `specs` — a list of (label, callable) — printing as it goes.

    Printing the label *before* running the check is the point: a check that
    blocks for its whole timeout should be visibly the thing that is blocking,
    not an unexplained pause. Each callable takes no arguments and returns a
    :class:`Check` (its ``name`` is filled in from the label).

    A callable that raises becomes a FAIL rather than killing the run — the
    remaining checks still have something to say, and an exception inside a
    diagnostic is itself a diagnostic.
    """
    stream = stream or sys.stdout
    colour = _use_colour(stream)
    width = min(shutil.get_terminal_size((100, 24)).columns, 110)
    report = Report(title=title)
    t0 = time.monotonic()

    note = "(skip with --no-preflight)"
    print("=" * _RULE_W, file=stream)
    head = f"preflight — {title}"
    print(f"{head}{note:>{max(1, _RULE_W - len(head))}}", file=stream)
    print("=" * _RULE_W, file=stream)

    n = len(specs)
    counter_w = len(str(n))
    for i, (label, fn) in enumerate(specs, 1):
        head = f"  {i:>{counter_w}}/{n} {label:<{_LABEL_W}} "
        print(head, end="", flush=True, file=stream)
        t = time.monotonic()
        try:
            check = fn()
        except Exception as e:  # noqa: BLE001 — a broken check must not hide the rest
            check = _fail(f"the check itself failed: {type(e).__name__}: {e}")
        check.name = label
        check.elapsed_s = time.monotonic() - t
        report.checks.append(check)

        indent = len(head) + 5
        body = _wrap(check.detail, indent, width)
        print(f"{_badge(check.status, colour)} {body[0]}", file=stream)
        for extra in body[1:]:
            print(" " * indent + extra, file=stream)
        if check.fix and (check.status != OK or show_ok_fix):
            for j, line in enumerate(check.fix):
                tag = "fix: " if j == 0 else "     "
                for k, wrapped in enumerate(_wrap(line, indent + 5, width)):
                    print(" " * indent + (tag if k == 0 else "     ") + wrapped,
                          file=stream)

    report.elapsed_s = time.monotonic() - t0

    print("-" * _RULE_W, file=stream)
    counts = (f"{report.count(OK)} ok, {report.count(WARN)} warning"
              f"{'' if report.count(WARN) == 1 else 's'}, "
              f"{report.count(FAIL)} failed")
    if report.count(SKIP):
        counts += f", {report.count(SKIP)} skipped"
    verdict = "READY" if report.ok else "NOT READY"
    if colour:
        verdict = f"{_COLOURS[OK if report.ok else FAIL]}{verdict}{_RESET}"
    print(f"{verdict} — {counts}  ({report.elapsed_s:.1f} s)", file=stream)
    failed = [c.name for c in report.checks if c.status == FAIL]
    if failed:
        print(f"blocking: {', '.join(failed)}", file=stream)
    print("=" * _RULE_W, file=stream)
    return report


# =============================================================================
# checks — host
# =============================================================================
def check_python_env(need_mavlink: bool = False, need_yaml: bool = False) -> Check:
    """The interpreter and the packages that decide whether anything can run.

    The one that bites on this desktop is depthai: the DEFAULT python is conda
    base with depthai 3.x, which opens a device merely by constructing a
    Pipeline. ``c3_camera/__init__.py`` already refuses to import there, so by
    the time this runs the version is known good — reporting it anyway is what
    makes "which python am I actually in" answerable from the output.
    """
    import depthai as dai

    parts = [f"python {sys.version.split()[0]}", f"depthai {dai.__version__}"]
    missing: list[str] = []
    for mod, label, required in (("cv2", "cv2", True),
                                 ("numpy", "numpy", True),
                                 ("yaml", "PyYAML", need_yaml),
                                 ("pymavlink", "pymavlink", need_mavlink)):
        try:
            m = __import__(mod)
            parts.append(f"{label} {getattr(m, '__version__', '?')}")
        except Exception:  # noqa: BLE001
            if required:
                missing.append(label)

    # numpy temporary-elision probe. On python 3.14 with a source-built numpy
    # 1.26 the OPERAND of a large-array expression gets overwritten — silently
    # wrong rather than loudly broken. See the header of the ./c3 wrapper.
    elide_note = ""
    try:
        import numpy as np
        a = np.full(100_000, 3.0)
        _ = a * a
        if float(a[0]) != 3.0:
            return _fail(
                f"numpy CORRUPTS OPERANDS on this interpreter (a[0] became "
                f"{float(a[0])} after `a * a`) — every array result from this "
                f"process is suspect",
                "use the robust env (python 3.12): ./c3 <tool>",
                "see ./c3 env for what this interpreter resolved to",
                python=sys.executable)
    except Exception as e:  # noqa: BLE001
        elide_note = f"; numpy elision unprobed ({type(e).__name__})"

    # The interpreter path on its own line: "which python am I actually in" is
    # the question this check exists to answer, and it is the one the wrapper
    # was written to stop people getting wrong.
    detail = "  ".join(parts) + f"\n{sys.executable}{elide_note}"
    if missing:
        return _warn(
            f"{detail}\nmissing: {', '.join(missing)}",
            f"pip install {' '.join(m.lower() for m in missing)}",
            "or use the wrapper, which picks an interpreter that has them: ./c3 <tool>",
            interpreter=sys.executable, missing=missing)
    return _ok(detail, interpreter=sys.executable)


def check_host_route(ip: str) -> Check:
    """Which local address the kernel would use to reach the camera.

    Catches the case where the camera is reachable but over the wrong interface
    (a VPN, a second NIC), which produces latency and drops that look like a
    camera problem and are not.
    """
    local, note = D.local_route_to(ip)
    if local is None:
        return _fail(
            f"no route to {ip} ({note})",
            "check the fibre NIC holds 192.168.2.1:  ip -brief addr",
            "check the GigaBlox switch and PoE power")
    if not local.startswith("192.168.2."):
        return _warn(
            f"{local} -> {ip}, but {local} is not on the camera's 192.168.2.x "
            f"subnet — traffic may be leaving over the wrong interface",
            "ip -brief addr   # which interface holds 192.168.2.1?",
            "ip route get " + ip,
            local=local)
    return _ok(f"{local} -> {ip}", local=local)


# =============================================================================
# checks — camera
# =============================================================================
def local_c3_processes() -> list[tuple[int, str]]:
    """Other processes on THIS host that look like they are holding the camera.

    When the device reports X_LINK_BOOTED the useful question is *who*, and the
    answer is very often "the c3 tool you left running in the other terminal"
    rather than the Madrona extension everyone reaches for first. /proc is free
    to read and exact about it; anything non-Linux just gets an empty list.
    """
    out: list[tuple[int, str]] = []
    me = os.getpid()
    try:
        entries = os.listdir("/proc")
    except OSError:
        return out
    for name in entries:
        if not name.isdigit() or int(name) == me:
            continue
        try:
            raw = Path(f"/proc/{name}/cmdline").read_bytes()
        except OSError:
            continue                      # exited between listdir and read
        cmd = raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()
        if not cmd or "c3_camera" not in cmd:
            continue
        # A preflight run is not an owner; neither is a grep for one.
        if "c3_camera.preflight" in cmd or "preflight.py" in cmd:
            continue
        out.append((int(name), cmd[:100]))
    return out


def check_camera_link(ip: str, timeout_s: float, state: dict) -> Check:
    """Is the camera's XLink TCP port accepting connections?

    Independent of DepthAI on purpose: it separates "the network is broken" from
    "DepthAI cannot claim it", which is the single most useful split in this
    stack because the two have completely different fixes.
    """
    t = time.monotonic()
    reachable, note = D.tcp_reachable(ip, timeout_s=timeout_s)
    state["tcp_reachable"] = reachable
    if not reachable:
        return _fail(
            note,
            "check PoE power and the GigaBlox switch",
            f"check this host still holds 192.168.2.1:  ip -brief addr",
            f"the camera answers on TCP {D.XLINK_TCP_PORT}; nothing else on the "
            f"path does, so a closed port is the link, not DepthAI",
            reachable=False)
    return _ok(f"TCP {ip}:{D.XLINK_TCP_PORT} open in "
               f"{(time.monotonic() - t) * 1000:.0f} ms", reachable=True)


def check_camera_owner(ip: str, mxid: str | None, timeout_s: float,
                       state: dict) -> Check:
    """Discovery, ownership, and identity — the check that usually fires.

    An OAK device runs exactly one pipeline at a time, so "already owned" is a
    hard stop and worth naming the owner for. Discovery itself is best-effort:
    it is a UDP broadcast and a switch or a VPN can eat it, which is not fatal
    because every connection in this package names its device by IP.
    """
    found = D.discover(timeout_s=timeout_s)
    target = D.find_c3(ip=ip, mxid=mxid)
    state["discovered"] = len(found)

    if target is None:
        others = ", ".join(f"{f.name}({f.state})" for f in found) or "none"
        if state.get("tcp_reachable"):
            return _warn(
                f"{mxid or ip} did not answer UDP discovery (saw: {others}), but "
                f"its TCP port is open — connecting by --ip needs no discovery, "
                f"so tools will still work",
                "harmless unless you use --mxid, which does need discovery",
                "broadcast is blocked by a switch, a firewall, or a VPN interface",
                discovered=[f.name for f in found])
        if state.get("tcp_reachable") is False:
            # A second FAIL here would name the same root cause twice and put two
            # entries in "blocking:" for one broken link. The finding is still
            # worth printing — discovery agreeing with the TCP probe is what
            # rules out "the port is filtered but the device is fine".
            return _warn(
                f"not found by discovery either (saw: {others}) — consistent "
                f"with the link failure above, which is the one to fix",
                discovered=[f.name for f in found])
        return _fail(
            f"{mxid or ip} not found by discovery and its TCP state is unknown "
            f"(saw: {others})",
            "python c3_camera/discover_c3.py   # the full report",
            discovered=[f.name for f in found])

    ident = f"mxid {target.mxid}"
    if not target.is_free:
        owners = local_c3_processes()
        fix = []
        if owners:
            fix.append("a local process already holds it: " +
                       "; ".join(f"pid {p}: {c}" for p, c in owners[:3]))
            fix.append("stop it and re-run — an OAK device allows one pipeline")
        else:
            fix.append(f"the BlueOS {MADRONA_ID} extension is the usual owner:")
            fix.append("    python -m c3_camera.madrona stop --yes")
            fix.append("    (then ~5-10 s for the device to reset into bootloader)")
            fix.append("    restore it afterwards with: python -m c3_camera.madrona start")
        return _fail(f"OWNED — a pipeline is already loaded ({target.state}), "
                     f"{ident}", *fix, state=target.state, mxid=target.mxid)

    detail = f"free ({target.state}), {ident}"
    if mxid is None and target.mxid.upper() != C.DEFAULT_MXID:
        return _warn(
            f"{detail}\nthis is NOT the expected C3 mx_id ({C.DEFAULT_MXID}) — "
            f"a different camera would carry different calibration",
            "pin the one you mean with --mxid, or update config.DEFAULT_MXID",
            mxid=target.mxid, expected=C.DEFAULT_MXID)
    return _ok(detail, mxid=target.mxid, state=target.state)


# =============================================================================
# checks — vehicle
# =============================================================================
def _http_json(url: str, timeout_s: float):
    with urllib.request.urlopen(url, timeout=timeout_s) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def vehicle_info(host: str = BLUEOS_HOST, timeout_s: float = 4.0) -> dict:
    """ArduSub version and frame type off BlueOS. Never raises."""
    out: dict = {}
    for key, path in (("ardusub_version", "/v1.0/firmware_info"),
                      ("vehicle_type", "/v1.0/vehicle_type")):
        try:
            out[key] = _http_json(f"http://{host}:8000{path}", timeout_s)
        except Exception:  # noqa: BLE001
            out[key] = None
    v = out.get("ardusub_version")
    if isinstance(v, dict):
        out["ardusub_version"] = f"{v.get('version')} ({v.get('type')})"
    return out


def check_blueos(host: str, timeout_s: float, state: dict | None = None) -> Check:
    """Is BlueOS answering, and what is it running?

    Probes the TCP port before asking for anything over HTTP. When the vehicle
    is simply off, that turns four separate service timeouts into one: the
    result lands in `state` and every later check that needs BlueOS reports the
    same cause immediately instead of re-discovering it three seconds at a time.
    """
    state = state if state is not None else {}
    up, note = D.tcp_reachable(host, port=8000, timeout_s=timeout_s)
    state["blueos_up"] = up
    if not up:
        return _warn(
            f"BlueOS at {host}:8000 is not answering ({note}) — the camera path "
            f"does not need it, but telemetry and the Madrona controls do",
            "is the ROV powered and the tether up?",
            f"ping {host}",
            "--mavlink-transport none if you meant camera-only")

    info = vehicle_info(host, timeout_s)
    if info.get("ardusub_version") is None and info.get("vehicle_type") is None:
        return _warn(
            f"BlueOS at {host}:8000 accepted the connection but answered neither "
            f"/firmware_info nor /vehicle_type — it may still be starting up",
            f"curl -s http://{host}:8000/v1.0/vehicle_type")
    bits = [b for b in (info.get("ardusub_version"), info.get("vehicle_type"))
            if b]
    return _ok(f"{host} — " + "  ".join(str(b) for b in bits), **info)


def _flag_bits(value) -> int | None:
    """MAVLink2Rest serialises bitfields as {"bits": N}; pymavlink gives an int."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, dict) and isinstance(value.get("bits"), int):
        return int(value["bits"])
    return None


def _sub_mode_name(custom_mode) -> str:
    """ArduSub flight-mode number -> name, via pymavlink's own table.

    Deliberately not a hand-written table: the mapping belongs to pymavlink and
    copying it here would be a number without a source.
    """
    try:
        from pymavlink import mavutil
        table = getattr(mavutil, "mode_mapping_sub", None) or {}
        for name, num in table.items():
            if num == custom_mode:
                return str(name)
    except Exception:  # noqa: BLE001
        pass
    return f"mode {custom_mode}"


def summarize_vehicle(messages: dict) -> tuple[str, dict]:
    """One human line + the structured facts, from a MAVLink2Rest message map.

    Split out from the check so it can be tested against a recorded payload
    without a vehicle: everything fiddly here (bitfields that are sometimes
    dicts, invalid sentinels, missing messages) is exactly what breaks silently.
    """
    def body(name: str) -> dict:
        entry = messages.get(name) or {}
        return (entry.get("message") or {}) if isinstance(entry, dict) else {}

    facts: dict = {}
    parts: list[str] = []

    hb = body("HEARTBEAT")
    if hb:
        base = _flag_bits(hb.get("base_mode"))
        if base is None:
            parts.append("armed state unknown")
        else:
            armed = bool(base & 128)     # MAV_MODE_FLAG_SAFETY_ARMED
            facts["armed"] = armed
            parts.append("ARMED" if armed else "disarmed")
        mode = hb.get("custom_mode")
        if isinstance(mode, int):
            facts["mode"] = _sub_mode_name(mode)
            parts.append(facts["mode"])
    else:
        parts.append("no HEARTBEAT")

    sysst = body("SYS_STATUS")
    mv = sysst.get("voltage_battery")
    if isinstance(mv, int) and 0 < mv < 65535:
        facts["battery_v"] = round(mv / 1000.0, 2)
        parts.append(f"{facts['battery_v']:.2f} V")
    pct = sysst.get("battery_remaining")
    if isinstance(pct, int) and 0 <= pct <= 100:
        facts["battery_pct"] = pct
        parts.append(f"{pct}%")

    att = body("ATTITUDE")
    if all(isinstance(att.get(k), (int, float)) for k in ("roll", "pitch")):
        import math
        facts["roll_deg"] = round(math.degrees(att["roll"]), 1)
        facts["pitch_deg"] = round(math.degrees(att["pitch"]), 1)
        parts.append(f"roll {facts['roll_deg']:+.0f}° pitch {facts['pitch_deg']:+.0f}°")

    # External barometer, reported raw. Turning hPa into a depth needs a water
    # density and a surface datum, neither of which this knows, so it does not
    # invent one — ArduSub's own depth is in the telemetry log either way.
    baro = body("SCALED_PRESSURE2")
    if isinstance(baro.get("press_abs"), (int, float)):
        facts["ext_press_hpa"] = round(float(baro["press_abs"]), 1)
        parts.append(f"ext baro {facts['ext_press_hpa']:.0f} hPa")

    return "  ".join(parts) if parts else "no messages", facts


def check_rov(rest_url: str, timeout_s: float, state: dict | None = None) -> Check:
    """Is the ROV alive right now, and in what state?

    Uses MAVLink2Rest rather than the UDP stream because it answers instantly,
    needs no vehicle configuration, and cannot contend with QGroundControl for a
    port. Liveness is proved by sampling HEARTBEAT's counter twice: a stale
    cached value looks identical to a live one in a single GET, and "the ROV was
    here a minute ago" is exactly the failure this is meant to catch.
    """
    if state is not None and state.get("blueos_up") is False:
        return _warn("BlueOS is not answering (see above), so vehicle state is "
                     "unknown — not re-tried here")
    url = f"{rest_url.rstrip('/')}/v1/mavlink/vehicles/1/components/1/messages"
    try:
        first = _http_json(url, timeout_s)
    except Exception as e:  # noqa: BLE001
        return _warn(
            f"MAVLink2Rest at {rest_url} did not answer ({type(e).__name__}) — "
            f"vehicle state unknown",
            "is the ROV powered and the tether connected?",
            f"curl -s {rest_url}/v1/mavlink | head -c 200",
            "camera-only work is unaffected; --mavlink-transport none silences this")
    if not isinstance(first, dict) or not first:
        return _warn(f"MAVLink2Rest answered but knows no messages — the "
                     f"autopilot link inside BlueOS may be down",
                     "check the Pi <-> Navigator link in the BlueOS UI")

    def counter(payload) -> int | None:
        entry = payload.get("HEARTBEAT") or {}
        status = entry.get("status") if isinstance(entry, dict) else None
        return ((status or {}).get("time") or {}).get("counter")

    c0 = counter(first)
    time.sleep(min(0.6, max(0.2, timeout_s / 5.0)))
    try:
        second = _http_json(url, timeout_s)
    except Exception:  # noqa: BLE001
        second = first
    c1 = counter(second)

    detail, facts = summarize_vehicle(second)
    if c0 is not None and c1 is not None and c1 == c0:
        return _warn(
            f"HEARTBEAT is STALE — its counter did not advance ({c0}); the last "
            f"known state was: {detail}",
            "the vehicle stopped talking to BlueOS; check power and the tether",
            stale=True, **facts)
    if c0 is None:
        return _warn(f"no HEARTBEAT in MAVLink2Rest; last known: {detail}",
                     "the autopilot is not reaching BlueOS", **facts)
    return _ok(detail, **facts)


def parse_udpin(connection: str) -> tuple[str, int] | None:
    """('0.0.0.0', 14551) from 'udpin:0.0.0.0:14551'; None for anything else.

    Only udpin is bindable-by-us; udpout/tcp/serial connection strings describe
    something this check has no business opening.
    """
    parts = str(connection).split(":")
    if len(parts) != 3 or parts[0].lower() not in ("udpin", "udp"):
        return None
    try:
        return parts[1], int(parts[2])
    except ValueError:
        return None


def check_mavlink_endpoint(port: int, host: str, place: str,
                           timeout_s: float, state: dict | None = None) -> Check:
    """Does BlueOS push a copy of MAVLink to the port this recorder will bind?

    QGroundControl consumes the default 14550 stream and two processes cannot
    receive the same unicast datagrams, so without a second endpoint the
    recorder simply never hears anything — silently, and only visible as an
    empty telemetry CSV after the dive.
    """
    from c3_camera import blueos_endpoint as BE

    if state is not None and state.get("blueos_up") is False:
        return _warn(f"BlueOS is not answering (see above), so whether it pushes "
                     f"to {place}:{port} is unknown — not re-tried here")
    try:
        endpoints = BE.list_endpoints(host)
    except Exception as e:  # noqa: BLE001
        return _warn(f"could not read BlueOS endpoints ({type(e).__name__}) — "
                     f"cannot tell whether {place}:{port} is configured",
                     f"python -m c3_camera.blueos_endpoint list --host {host}")
    match = [e for e in endpoints
             if e.get("argument") in (port, str(port))
             and e.get("connection_type") == "udpout"]
    if not match:
        others = ", ".join(f"{e.get('connection_type')}:{e.get('place')}:"
                           f"{e.get('argument')}" for e in endpoints) or "none"
        return _warn(
            f"BlueOS does not push to {place}:{port}, so this recorder will "
            f"receive nothing (configured: {others})",
            f"python -m c3_camera.blueos_endpoint add --port {port} --yes",
            "or avoid vehicle config entirely: --mavlink-transport rest",
            "(adding an endpoint may blip QGC's link — do it before the dive)",
            configured=others)
    e = match[0]
    if not e.get("enabled", True):
        return _warn(f"the endpoint to {place}:{port} exists but is DISABLED",
                     f"enable it in BlueOS, or remove and re-add it: "
                     f"python -m c3_camera.blueos_endpoint add --port {port} --yes")
    return _ok(f"BlueOS pushes udpout -> {e.get('place')}:{port} "
               f"(persistent={'yes' if e.get('persistent') else 'no'})")


def check_mavlink_udp(connection: str, wait_s: float,
                      state: dict | None = None) -> Check:
    """Bind the recorder's own port and wait for a real datagram.

    This is the only check that exercises the path the recorder will actually
    use, and it catches the two things the endpoint check cannot: the port
    already being held by something else, and packets that are configured but
    not arriving. It binds and closes; the logger re-binds afterwards without
    trouble because UDP has no lingering state to fight.

    The bind happens even when BlueOS is known to be down, because a port
    conflict is worth finding either way and costs nothing; only the *waiting*
    is skipped, since a dark vehicle is not going to send anything.
    """
    target = parse_udpin(connection)
    if target is None:
        return _skip(f"{connection} is not a udpin address; nothing to bind")
    host, port = target

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        try:
            sock.bind((host, port))
        except OSError as e:
            holders = ", ".join(f"pid {p}" for p, _ in local_c3_processes())
            return _warn(
                f"UDP {host}:{port} is not bindable ({e.strerror or e}) — "
                f"something else already holds it"
                + (f"; local c3 processes: {holders}" if holders else ""),
                f"ss -lunp 'sport = :{port}'   # who has it",
                "QGroundControl holds 14550; the recorder needs its own port",
                port=port)
        if state is not None and state.get("blueos_up") is False:
            return _warn(
                f"UDP {host}:{port} is free to bind, but BlueOS is not answering "
                f"(see above) so nothing would arrive — not waiting {wait_s:g} s "
                f"to confirm it",
                port=port)
        sock.settimeout(min(0.5, wait_s))
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            try:
                data, addr = sock.recvfrom(2048)
            except socket.timeout:
                continue
            if data:
                elapsed = wait_s - (deadline - time.monotonic())
                return _ok(f"{len(data)} B from {addr[0]}:{addr[1]} on "
                           f"{host}:{port} after {elapsed:.1f} s",
                           port=port, source=f"{addr[0]}:{addr[1]}")
        return _warn(
            f"nothing arrived on UDP {host}:{port} in {wait_s:g} s — telemetry "
            f"would be empty for this run",
            f"python -m c3_camera.blueos_endpoint add --port {port} --yes",
            "or --mavlink-transport rest (no vehicle configuration needed)",
            "or --mavlink-transport none if you meant camera-only",
            port=port)
    finally:
        sock.close()


# =============================================================================
# checks — storage and display
# =============================================================================
def _nearest_existing(path: Path) -> Path:
    path = Path(path).resolve()
    while not path.exists() and path.parent != path:
        path = path.parent
    return path


def check_storage(out_dir, min_free_gb: float) -> Check:
    """Free space and write permission where the run will write.

    Checked against the nearest existing ancestor, because the output directory
    itself usually does not exist yet — and that ancestor is what carries the
    filesystem the run will actually fill.
    """
    if out_dir is None:
        return _skip("this tool writes nothing that needs checking")
    anchor = _nearest_existing(Path(out_dir))
    if not os.access(anchor, os.W_OK):
        return _fail(f"{anchor} is not writable by this user",
                     f"ls -ld {anchor}",
                     "pick another --out")
    free_gb = shutil.disk_usage(anchor).free / 1e9
    detail = f"{free_gb:.0f} GB free at {anchor}"
    if min_free_gb and free_gb < min_free_gb:
        return _fail(f"{detail}, below the {min_free_gb:g} GB floor — recording "
                     f"would stop almost immediately",
                     "free space, or lower --min-free-gb if you mean it",
                     free_gb=round(free_gb, 1))
    if min_free_gb and free_gb < min_free_gb * 3:
        return _warn(f"{detail}, within 3x of the {min_free_gb:g} GB floor — a "
                     f"long take will hit it",
                     free_gb=round(free_gb, 1))
    return _ok(detail, free_gb=round(free_gb, 1))


def check_display(want_window: bool) -> Check:
    """Will cv2.imshow open a window, or kill the process at the first frame?

    Two independent ways this fails on this rig: no X/Wayland server at all
    (ssh without -X), and cv2's Qt plugin directory being renamed in the conda
    `robust` env — see viz._qt_plugin_path, which repairs the second one at
    import and whose result is reported here so it is visible rather than magic.
    """
    if not want_window:
        return _skip("headless run requested")
    server = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    if not server:
        return _warn(
            "neither DISPLAY nor WAYLAND_DISPLAY is set, so opening a window "
            "will fail — this is what an ssh session without -X looks like",
            "pass --no-display and drive the run from the console",
            "or reconnect with ssh -X / run on the desktop session")
    try:
        from c3_camera import viz
        plugins = viz.repair_qt_plugin_path()
    except Exception as e:  # noqa: BLE001
        return _warn(f"display {server}, but cv2/Qt could not be prepared: "
                     f"{type(e).__name__}: {e}",
                     "pass --no-display to run without a window")
    note = f" (Qt plugins repaired -> {plugins})" if plugins else ""
    return _ok(f"display {server}{note}", display=server)


# =============================================================================
# check-set builders
# =============================================================================
def host_checks(*, ip: str, need_mavlink: bool = False,
                need_yaml: bool = False) -> list:
    return [
        ("python + deps", lambda: check_python_env(need_mavlink, need_yaml)),
        ("host route", lambda: check_host_route(ip)),
    ]


def camera_checks(*, ip: str, mxid: str | None, timeout_s: float) -> list:
    # The two share the TCP verdict: "not discovered" means something completely
    # different depending on whether the port is open, and asking twice would
    # both cost a second timeout and risk the two disagreeing.
    state: dict = {}
    return [
        ("camera link", lambda: check_camera_link(ip, timeout_s, state)),
        ("camera owner", lambda: check_camera_owner(ip, mxid, timeout_s, state)),
    ]


def vehicle_checks(*, transport: str, connection: str, rest_url: str,
                   blueos_host: str = BLUEOS_HOST, timeout_s: float = 3.0,
                   place: str = "192.168.2.1") -> list:
    """BlueOS, the ROV itself, and the recorder's own telemetry path.

    `transport` decides which of the last two apply: the UDP path needs an
    endpoint and a bindable port, the REST path needs neither, and `none` means
    the operator has said they do not want telemetry at all.
    """
    if transport == "none":
        return [
            ("BlueOS", lambda: _skip("--mavlink-transport none")),
            ("ROV state", lambda: _skip("--mavlink-transport none")),
            ("telemetry path", lambda: _skip("--mavlink-transport none")),
        ]

    # One reachability verdict, shared: when the vehicle is simply off, three
    # more service timeouts add nine seconds and no information.
    state: dict = {}
    checks = [
        ("BlueOS", lambda: check_blueos(blueos_host, timeout_s, state)),
        ("ROV state", lambda: check_rov(rest_url, timeout_s, state)),
    ]
    udp = parse_udpin(connection) if transport == "udp" else None
    if udp is None:
        checks.append(("telemetry path",
                       lambda: _ok(f"MAVLink2Rest {rest_url} (sampled, not "
                                   f"captured — see mavlink_log.py)")
                       if transport == "rest" else
                       _skip(f"transport {transport}")))
    else:
        port = udp[1]
        checks += [
            ("MAVLink endpoint",
             lambda: check_mavlink_endpoint(port, blueos_host, place, timeout_s,
                                            state)),
            ("telemetry path",
             lambda: check_mavlink_udp(connection, timeout_s, state)),
        ]
    return checks


def storage_checks(*, out_dir, min_free_gb: float) -> list:
    return [("disk", lambda: check_storage(out_dir, min_free_gb))]


def display_checks(*, want_window: bool) -> list:
    return [("display", lambda: check_display(want_window))]


def standard_checks(a: argparse.Namespace, *, camera: bool = True,
                    vehicle: bool | None = None, out_dir=None,
                    min_free_gb: float | None = None,
                    display: bool | None = None) -> list:
    """The usual set, read off a tool's parsed arguments.

    Anything not present on the namespace is simply not checked, so one call
    site works for tools with very different flag sets. `vehicle=None` means
    "check it if this tool has a --mavlink-transport", which is the honest
    default: a tool that cannot record telemetry has no reason to wait on it.
    """
    ip = getattr(a, "ip", C.DEFAULT_IP)
    timeout = float(getattr(a, "preflight_timeout", 3.0) or 3.0)
    transport = getattr(a, "mavlink_transport", None)
    if vehicle is None:
        vehicle = transport is not None
    if min_free_gb is None:
        min_free_gb = float(getattr(a, "min_free_gb", 0.0) or 0.0)

    specs = host_checks(ip=ip,
                        need_mavlink=bool(vehicle) and transport != "none",
                        need_yaml=getattr(a, "profile", None) is not None)
    if camera:
        specs += camera_checks(ip=ip, mxid=getattr(a, "mxid", None),
                               timeout_s=timeout)
    if vehicle:
        specs += vehicle_checks(
            transport=transport or "none",
            connection=getattr(a, "mavlink", ""),
            rest_url=getattr(a, "mavlink_rest_url",
                             f"http://{BLUEOS_HOST}:{MAVLINK_REST_PORT}"),
            timeout_s=timeout)
    if out_dir is not None:
        specs += storage_checks(out_dir=out_dir, min_free_gb=min_free_gb)
    if display is not None:
        specs += display_checks(want_window=bool(display))
    return specs


# =============================================================================
# the seam every tool uses
# =============================================================================
def add_preflight_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("preflight (readiness checks before anything connects)")
    g.add_argument("--no-preflight", action="store_true",
                   help="skip the readiness checks and go straight to connecting")
    g.add_argument("--preflight-only", action="store_true",
                   help="run the readiness checks and exit, without connecting")
    g.add_argument("--preflight-strict", action="store_true",
                   help="treat warnings as failures (unattended runs, where a "
                        "dataset without telemetry is not worth the dive)")
    g.add_argument("--preflight-timeout", type=float, default=3.0,
                   help="per-check network timeout in seconds (default: %(default)s)")
    g.add_argument("--preflight-json", type=Path, default=None,
                   help="also write the readiness report here as JSON")


def gate(a: argparse.Namespace, *, title: str, specs=None, **kw):
    """Run preflight and decide whether the caller may continue.

    Returns None to continue, or an exit code the caller should return —
    0 for ``--preflight-only``, :data:`EXIT_PREFLIGHT` when something blocked.
    The report is stashed on the namespace as ``a.preflight_report`` so a tool
    can record it beside its results without threading it through its own code.

        rc = PF.gate(a, title="c3 collect", out_dir=out_dir, display=not a.no_display)
        if rc is not None:
            return rc
    """
    if getattr(a, "no_preflight", False) and not getattr(a, "preflight_only", False):
        return None
    report = run(specs if specs is not None else standard_checks(a, **kw),
                 title=title)
    a.preflight_report = report

    out = getattr(a, "preflight_json", None)
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report.to_dict(), indent=2, default=str))
        print(f"  wrote {out}")

    blocking = report.blocking(strict=getattr(a, "preflight_strict", False))
    if getattr(a, "preflight_only", False):
        return 0 if not blocking else EXIT_PREFLIGHT
    if blocking:
        names = ", ".join(c.name for c in blocking)
        print(f"\nstopping before anything connects — blocked by: {names}")
        print("each blocking check printed what to run; --no-preflight overrides "
              "this if you know better.")
        return EXIT_PREFLIGHT
    return None


# =============================================================================
# standalone
# =============================================================================
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m c3_camera.preflight",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_connection_args(p)
    p.add_argument("--mavlink", default="udpin:0.0.0.0:14551",
                   help="the udpin address a recorder would bind (default: %(default)s)")
    p.add_argument("--mavlink-transport", default="udp",
                   choices=("udp", "rest", "none"),
                   help="which telemetry path to check (default: %(default)s)")
    p.add_argument("--mavlink-rest-url",
                   default=f"http://{BLUEOS_HOST}:{MAVLINK_REST_PORT}")
    p.add_argument("--out", type=Path, default=Path("c3_camera/datasets"),
                   help="directory to check for space (default: %(default)s)")
    p.add_argument("--min-free-gb", type=float, default=10.0,
                   help="free-space floor to check against (default: %(default)s)")
    p.add_argument("--no-display", action="store_true",
                   help="check as a headless run would")
    p.add_argument("--json-out", type=Path, default=None,
                   help="write the report here as JSON")
    add_preflight_args(p)
    a = p.parse_args(argv)

    report = run(standard_checks(a, out_dir=a.out, display=not a.no_display),
                 title="c3 rig")
    target = a.json_out or a.preflight_json
    if target is not None:
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_text(json.dumps(report.to_dict(), indent=2, default=str))
        print(f"wrote {target}")
    if report.ok:
        print("\nnext:  ./c3 collect --profile research_near"
              "        (or ./c3 discover --probe for sensors + calibration,"
              " which does open the camera)")
    return 0 if not report.blocking(strict=a.preflight_strict) else EXIT_PREFLIGHT


if __name__ == "__main__":
    raise SystemExit(main())
