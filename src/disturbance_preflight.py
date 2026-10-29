#!/usr/bin/env python3
"""
disturbance_preflight.py — one screen that says whether the DISTURBANCE RIG is
ready, before ./c3 disturb connects to anything.

Same contract as c3_camera/preflight.py (the camera/ROV preflight): print the
label before each check, OK / WARN / FAIL / skip with the fix beside anything
that is not OK, WARN never blocks, FAIL blocks unless --no-preflight, exit code 5
when blocked, --preflight-only to run the report alone, --preflight-json for a
machine-readable copy that the panel stores in meta.json. The result types and
the renderer are COPIED from c3_camera/preflight.py:97-263 rather than imported:
that module imports c3_camera.device, which needs depthai, and the disturbance
panel runs in the `whisker` conda env (python-can + pyserial) where depthai does
not exist.

Devices are judged as a group (2026-10-27, user request): a FAIL on a device
check (gantry client / gantry route -> gantry, can bus -> motor, thruster port
-> thruster) means only "that device cannot connect yet". While at least one of
the three devices passes, those FAILs become WARNs and the panel opens without
the missing ones — their Connect button reports the error until the device is
fixed (plugged in, cabled, can0 up: whisker's damiao raises can0 itself on
Connect motor) and then connects. Only when ALL three fail does preflight
block, as does any FAIL of a non-device check (interpreter, whisker_vision,
data root, display).

What it deliberately does NOT do: it never opens a device. No TCP connect to
the FMC4030 (the controller allows ONE client, and a probe would take it), no
CAN socket, no serial port. Everything is read from the filesystem, `ip`, `ss`,
`sudo -n -l` and importlib.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path

from disturbance_core import REPO_ROOT, gantry_link_held, git_head, whisker_dir

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"
#: exit code a tool returns when preflight blocked it (same as c3_camera)
EXIT_PREFLIGHT = 5
_LABEL_W = 18
_RULE_W = 78
GANTRY_BITRATE = 1_000_000
#: device checks -> the device they gate (see the module docstring)
DEVICE_CHECKS = {"gantry client": "gantry", "gantry route": "gantry",
                 "can bus": "motor", "thruster port": "thruster"}
DEVICES = ("gantry", "motor", "thruster")


# =============================================================================
# result types (copied from c3_camera/preflight.py:97-160)
# =============================================================================
@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    fix: tuple = ()
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
    checks: list = field(default_factory=list)
    elapsed_s: float = 0.0
    title: str = ""
    #: {"available": [...], "unavailable": [...], "relaxed": bool} — see relax_devices
    devices: dict = field(default_factory=dict)

    def count(self, status: str) -> int:
        return sum(1 for c in self.checks if c.status == status)

    @property
    def ok(self) -> bool:
        return self.count(FAIL) == 0

    def blocking(self, strict: bool = False) -> list:
        bad = {FAIL, WARN} if strict else {FAIL}
        return [c for c in self.checks if c.status in bad]

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "elapsed_s": round(self.elapsed_s, 3),
            "verdict": "ready" if self.ok else "not_ready",
            "counts": {s: self.count(s) for s in (OK, WARN, FAIL, SKIP)},
            "devices": dict(self.devices),
            "checks": [c.to_dict() for c in self.checks],
        }


def relax_devices(report: Report) -> dict:
    """Judge the devices as a group: a device is unavailable when any of its
    checks FAILED. While at least one device is available, each unavailable
    device's FAILs become WARNs (the panel opens; that Connect fails until the
    device is fixed, then works). All unavailable -> nothing changes (blocks).
    Devices whose checks were all skipped (--sim) are neither."""
    seen, down = set(), set()
    for c in report.checks:
        dev = DEVICE_CHECKS.get(c.name)
        if dev is None or c.status == SKIP:
            continue
        seen.add(dev)
        if c.status == FAIL:
            down.add(dev)
    available = [d for d in DEVICES if d in seen and d not in down]
    unavailable = [d for d in DEVICES if d in down]
    relaxed = bool(available and unavailable)
    if relaxed:
        for c in report.checks:
            dev = DEVICE_CHECKS.get(c.name)
            if dev in down and c.status == FAIL:
                c.status = WARN
                c.detail += (f"\n-> {dev} UNAVAILABLE: the panel opens without it; fix it, then "
                             f"press Connect {dev}")
                c.data["unavailable_device"] = dev
    report.devices = {"available": available, "unavailable": unavailable, "relaxed": relaxed}
    return report.devices


def _ok(detail: str, **data) -> Check:
    return Check("", OK, detail, data=data)


def _warn(detail: str, *fix: str, **data) -> Check:
    return Check("", WARN, detail, fix=tuple(fix), data=data)


def _fail(detail: str, *fix: str, **data) -> Check:
    return Check("", FAIL, detail, fix=tuple(fix), data=data)


def _skip(detail: str, **data) -> Check:
    return Check("", SKIP, detail, data=data)


# =============================================================================
# rendering + runner (copied from c3_camera/preflight.py:166-263)
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


def _wrap(text: str, indent: int, width: int) -> list:
    if not text:
        return [""]
    lines = []
    for para in text.split("\n"):
        lines += textwrap.wrap(para, width=max(20, width - indent)) or [""]
    return lines


def run(specs, title: str = "preflight", stream=None,
        show_ok_fix: bool = False) -> Report:
    """Run `specs` — a list of (label, callable) — printing as it goes."""
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
    devices = relax_devices(report)
    print("-" * _RULE_W, file=stream)
    if devices["unavailable"]:
        if devices["relaxed"]:
            print(f"devices: {', '.join(devices['available'])} ok; "
                  f"{', '.join(devices['unavailable'])} UNAVAILABLE — the panel opens without "
                  "them (their FAIL above counts as a warning); Connect works once fixed",
                  file=stream)
        else:
            print("devices: gantry, motor and thruster are ALL unavailable — nothing to drive",
                  file=stream)
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
# the checks — all read-only, none opens a device
# =============================================================================
def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def check_interpreter(sim: bool) -> Check:
    parts = [f"python {sys.version.split()[0]}"]
    if not _has("PyQt5"):
        return _fail(f"{' '.join(parts)}: PyQt5 missing\n{sys.executable}",
                     "DISTURB_PY=~/miniforge3/envs/whisker/bin/python ./c3 disturb",
                     interpreter=sys.executable)
    parts.append("PyQt5")
    missing = [name for mod, name in (("can", "python-can"), ("serial", "pyserial"))
               if not _has(mod)]
    detail = "  ".join(parts) + f"\n{sys.executable}"
    if missing and not sim:
        return _warn(
            f"{detail}\nmissing: {', '.join(missing)} — gantry only; motor/thruster "
            "Connect will fail",
            "DISTURB_PY=~/miniforge3/envs/whisker/bin/python ./c3 disturb   "
            "(python-can pins wrapt~=1.10: do NOT pip install it into robust)",
            interpreter=sys.executable, missing=missing)
    if missing:
        detail += f"\n(no {', '.join(missing)}: fine for --sim)"
    return _ok(detail, interpreter=sys.executable)


def check_whisker(root: Path, sim: bool = False) -> Check:
    if not (root / "motion_control.py").is_file():
        return _fail(f"whisker_vision not found at {root}",
                     "export WHISKER_VISION_DIR=/path/to/whisker_vision  (or --whisker-dir)",
                     path=str(root))
    g = git_head(root)
    so = root / "motion" / "vendor" / "ubuntu" / "lib" / "64" / "libFMC4030-Lib.so"
    detail = f"{root}  branch {g.get('branch')}  {str(g.get('sha') or '?')[:10]}"
    if g.get("dirty"):
        detail += "  (dirty worktree)"
    if not so.is_file():
        if sim:
            return _warn(detail + f"\nvendored FMC4030 library missing: {so} (fine for --sim)",
                         "copy it from another checkout before a real session",
                         path=str(root), git=g)
        return _fail(detail + f"\nvendored FMC4030 library missing: {so}",
                     "the .so is gitignored — copy it from another checkout "
                     "(src/gantry/vendor/ubuntu/lib/64/ has the same file)",
                     path=str(root), git=g)
    return _ok(detail, path=str(root), git=g)


def check_gantry_client(ip: str, port: int, sim: bool, probe=gantry_link_held) -> Check:
    if sim:
        return _skip("simulated gantry")
    held, note = probe(ip, port)
    if held:
        return _fail(f"another process already holds {ip}:{port} — the controller "
                     f"accepts ONE client\n{note}",
                     "close ./c3 gantry, whisker's motion_panel.py or gantry_soft_limits.py first",
                     socket=note)
    if held is None:
        return _warn(note, "ss -tn | grep " + f"{ip}:{port}")
    return _ok(note)


def _ip_json(args: list):
    from disturbance_core import _run  # noqa: PLC0415
    r = _run(["ip", "-json"] + args)
    if r is None or r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except ValueError:
        return None


def check_gantry_route(ip: str, sim: bool, route=None) -> Check:
    if sim:
        return _skip("simulated gantry")
    data = route if route is not None else _ip_json(["route", "get", ip])
    if not data:
        return _fail(f"no route to {ip}",
                     "plug the USB-Ethernet dongle (static 192.168.0.35/24) or join the "
                     "192.168.0.x WiFi;  ip -brief addr;  ip route get " + ip)
    entry = data[0] if isinstance(data, list) else data
    dev = str(entry.get("dev", "?"))
    src = str(entry.get("prefsrc", "?"))
    if dev.startswith("wl"):
        return _warn(f"{src} -> {ip} via {dev} (WiFi): STOP ALL reaches the controller "
                     "only while WiFi holds; a one-shot move runs to completion on loss",
                     "prefer the wired USB-Ethernet dongle (NM profile 'Wired connection 2/3', "
                     "192.168.0.35/24)", dev=dev, local=src)
    return _ok(f"{src} -> {ip} via {dev}", dev=dev, local=src)


def _sudo_can_rule(channel: str) -> "bool | None":
    from disturbance_core import _run  # noqa: PLC0415
    r = _run(["sudo", "-n", "-l"])
    if r is None or r.returncode != 0:
        return None
    return f"ip link set {channel}" in r.stdout


def check_can(channel: str, sim: bool, link=None, sudo_rule=None) -> Check:
    if sim:
        return _skip("simulated motor")
    data = link if link is not None else _ip_json(["-details", "link", "show", channel])
    if not data:
        return _fail(f"no {channel} interface — is the USB-CAN (gs_usb/candleLight) "
                     "adapter plugged in?",
                     "lsusb | grep -i can;  then once per boot:  sudo ip link set "
                     f"{channel} type can bitrate 1000000 && sudo ip link set {channel} up")
    entry = data[0] if isinstance(data, list) else data
    flags = entry.get("flags", []) or []
    info = (entry.get("linkinfo") or {}).get("info_data") or {}
    state = str(info.get("state", "")).upper()
    bitrate = (info.get("bittiming") or {}).get("bitrate")
    rule = sudo_rule if sudo_rule is not None else _sudo_can_rule(channel)
    bring_up = (f"sudo ip link set {channel} down; sudo ip link set {channel} type can "
                f"bitrate {GANTRY_BITRATE}; sudo ip link set {channel} up; "
                f"sudo ip link set {channel} txqueuelen 1000")
    if "UP" not in flags or bitrate is None:
        if rule:
            return _warn(f"{channel} is DOWN — the panel raises it on Connect motor "
                         "(sudo -n, NOPASSWD rule present)", bring_up, state=state)
        return _fail(f"{channel} is DOWN and no NOPASSWD sudo rule for `ip link set "
                     f"{channel}` — the automatic bring-up would fail", bring_up,
                     state=state)
    if int(bitrate) != GANTRY_BITRATE:
        return _fail(f"{channel} at {bitrate} bps, the rig runs at {GANTRY_BITRATE}",
                     bring_up, bitrate=bitrate)
    if state in ("BUS-OFF", "STOPPED"):
        return _fail(f"{channel} is {state} — the adapter has no restart-ms; cycle it by hand",
                     f"sudo ip link set {channel} down && sudo ip link set {channel} up",
                     state=state)
    if state in ("ERROR-PASSIVE", "ERROR-WARNING"):
        return _warn(f"{channel} is {state}: frames go out but nothing ACKs — motor DC "
                     "supply off or CAN-H/L unconnected?",
                     "check the motor supply and termination; python3 "
                     "~/Desktop/whisker_vision/can_power_check.py (status requests only)",
                     state=state)
    return _ok(f"{channel} up at {int(bitrate) // 1000} kbps, state {state or 'n/a'}",
               state=state, bitrate=bitrate)


def check_thruster_port(port: str, sim: bool) -> Check:
    if sim:
        return _skip("simulated thruster")
    if Path(port).exists():
        return _ok(port)
    by_id = Path("/dev/serial/by-id")
    seen = sorted(p.name for p in by_id.iterdir()) if by_id.is_dir() else []
    return _fail(f"{port} not present" + (f"\nseen: {', '.join(seen)}" if seen
                                             else "\n(no /dev/serial/by-id at all — "
                                                  "Arduino unplugged?)"),
                 "plug the thruster Arduino; ls -l /dev/serial/by-id/; "
                 "--thruster-port <path> if the board changed", seen=seen)


def check_ld_library_path() -> Check:
    val = os.environ.get("LD_LIBRARY_PATH", "")
    if any(s in val.lower() for s in ("openfoam", "cuda")):
        return _warn("LD_LIBRARY_PATH carries OpenFOAM/CUDA paths — PyQt5's xcb plugin can "
                     "bind the system Qt and die; ./c3 disturb unsets it, a direct launch "
                     "must too", "env -u LD_LIBRARY_PATH python src/disturbance_panel.py ...")
    return _ok("clean" if not val else "set, no OpenFOAM/CUDA entries")


def check_data_root(root: Path, min_free_gb: float = 1.0) -> Check:
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".preflight_write_test"
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        return _fail(f"{root} not writable: {exc}", "--data-root <writable dir>")
    free_gb = shutil.disk_usage(root).free / 1e9
    if free_gb < min_free_gb:
        return _warn(f"{root}: {free_gb:.1f} GB free", "free some space or --data-root elsewhere",
                     free_gb=round(free_gb, 2))
    return _ok(f"{root}  ({free_gb:.0f} GB free)", free_gb=round(free_gb, 2))


def check_display() -> Check:
    if os.environ.get("QT_QPA_PLATFORM", "") == "offscreen":
        return _warn("QT_QPA_PLATFORM=offscreen — no window will appear (test shell "
                     "leftover?)", "unset QT_QPA_PLATFORM")
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return _ok(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return _fail("no DISPLAY / WAYLAND_DISPLAY — a Qt window cannot open",
                 "run from the desktop session, or ssh -X")


def disturb_checks(*, sim: bool, whisker_root: Path, gantry_ip: str, gantry_port: int,
                   can_channel: str, thruster_port: str, data_root: Path) -> list:
    return [
        ("interpreter", lambda: check_interpreter(sim)),
        ("whisker_vision", lambda: check_whisker(whisker_root, sim)),
        ("gantry client", lambda: check_gantry_client(gantry_ip, gantry_port, sim)),
        ("gantry route", lambda: check_gantry_route(gantry_ip, sim)),
        ("can bus", lambda: check_can(can_channel, sim)),
        ("thruster port", lambda: check_thruster_port(thruster_port, sim)),
        ("LD_LIBRARY_PATH", check_ld_library_path),
        ("data root", lambda: check_data_root(Path(data_root))),
        ("display", check_display),
    ]


# =============================================================================
# argparse + gate (same flags as c3_camera/preflight.py:996-1044)
# =============================================================================
def add_preflight_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("preflight (readiness checks before anything connects)")
    g.add_argument("--no-preflight", action="store_true",
                   help="skip the readiness checks and go straight to the panel")
    g.add_argument("--preflight-only", action="store_true",
                   help="run the readiness checks and exit, without opening the panel")
    g.add_argument("--preflight-strict", action="store_true",
                   help="treat warnings as failures")
    g.add_argument("--preflight-json", type=Path, default=None,
                   help="also write the readiness report here as JSON")


def gate(a: argparse.Namespace, *, title: str, specs, stream=None) -> "int | None":
    """Run preflight; None = continue, else the exit code to return."""
    stream = stream or sys.stdout
    if getattr(a, "no_preflight", False) and not getattr(a, "preflight_only", False):
        a.preflight_report = None
        return None
    report = run(specs, title=title, stream=stream)
    a.preflight_report = report
    out = getattr(a, "preflight_json", None)
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report.to_dict(), indent=2, default=str))
        print(f"  wrote {out}", file=stream)
    blocking = report.blocking(strict=getattr(a, "preflight_strict", False))
    if getattr(a, "preflight_only", False):
        return 0 if not blocking else EXIT_PREFLIGHT
    if blocking:
        names = ", ".join(c.name for c in blocking)
        print(f"\nstopping before anything connects — blocked by: {names}", file=stream)
        print("each blocking check printed what to run; --no-preflight overrides "
              "this if you know better.", file=stream)
        return EXIT_PREFLIGHT
    return None


def main(argv: "list[str] | None" = None) -> int:
    """Standalone: the report for the current environment (no panel)."""
    ap = argparse.ArgumentParser(description="disturbance rig readiness report")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--whisker-dir", default=None)
    ap.add_argument("--gantry-ip", default="192.168.0.30")
    ap.add_argument("--gantry-port", type=int, default=8088)
    ap.add_argument("--can-channel", default="can0")
    ap.add_argument("--thruster-port",
                    default="/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_"
                            "75330303934351903171-if00")
    ap.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    add_preflight_args(ap)
    a = ap.parse_args(argv)
    a.preflight_only = True
    rc = gate(a, title="disturbance rig", specs=disturb_checks(
        sim=a.sim, whisker_root=whisker_dir(a.whisker_dir), gantry_ip=a.gantry_ip,
        gantry_port=a.gantry_port, can_channel=a.can_channel,
        thruster_port=a.thruster_port, data_root=Path(a.data_root)))
    return 0 if rc is None else rc


if __name__ == "__main__":
    raise SystemExit(main())
