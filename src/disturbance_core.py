#!/usr/bin/env python3
"""
disturbance_core.py — the Qt-free half of the pool DISTURBANCE RIG panel.

The rig: a pool-top FMC4030 3-axis gantry (TCP 192.168.0.30:8088) carrying a
Damiao rotation motor (SocketCAN can0) on which an external T200 thruster
(Arduino Uno + ESC, serial line protocol -100..100 %) is mounted. The jet is a
disturbance SOURCE for BlueROV2 control experiments — this is NOT the DOB/EAOB
estimated disturbance that rov_gui calls "disturbance".

Everything that touches a device lives in whisker_vision (the labmate's repo,
imported BY PATH and never modified):

    motion_control.GantryLink / MotorLink / ThrusterLink   device-owning threads
    motion_config.MotionSettings + build_*_link             sim <-> real factories
    motion_simulator.Sim*                                   --sim fakes

This module holds what the panel needs that is not a widget: the sys.path shim,
the rig config (config/disturbance_rig.yaml) and CLI -> MotionSettings, the
thruster factory with a serial write timeout, the run-folder logger
(data/YYYYMMDD/MMDD_HHMMSS_disturb/), and the git / ss helpers. It imports no
Qt, so it is tested without a display and runs in any interpreter that can
import whisker_vision (python-can / pyserial are only needed for real devices,
and only inside the links' lazy imports).

sys.path order matters: src/ first, then the repo root (rov_gui.runstore),
then whisker_vision LAST. whisker_vision's root holds top-level tests/,
simulator.py, cameras.py, reader.py ... — inserted at index 0 it would shadow
same-named modules of this repo.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SRC_DIR.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_RIG_CONFIG = REPO_ROOT / "config" / "disturbance_rig.yaml"
DEFAULT_WHISKER_DIR = Path.home() / "Desktop" / "whisker_vision"
AXIS_NAMES = ("X", "Y", "Z")
#: taught soft limits ("−X home" / "+X home"), kept beside the run folders
LIMITS_FILE_NAME = "disturbance_rig_limits.json"
#: FMC4030_Stop_Single_Axis mode: 1 = decelerate (减速停止), 2 = immediate (立即停止)
#: [스펙: src/gantry/mannual/FMC4030二次开发库详解V1.0.pdf, Stop_Single_Axis].
#: 1 is what the vendor Qt demo and ./c3 gantry send on a jog release. On THIS
#: controller it stopped 18/18 hold-jog releases at 20-50 mm/s, at rest within
#: 1.43 s (data/20261026/1026_140909_disturb/events.csv, jog_stopped rows). Inside STOP ALL it is not yet exercised.
JOG_STOP_MODE = 1
#: gantry speed cap of this rig (jog, Send, pattern). whisker's links clamp to
#: their own MAX_SPEED_MM_S = 50 (motion_config.py:72, a policy value — no
#: hardware reason given); ./c3 gantry drives this same gantry at 10 cm/s by
#: default (src/gantry_panel.py:3131). The release run-on grows with v²/2a
#: (acc 50 mm/s²: ~100 mm at 100 mm/s [유도]).
RIG_MAX_SPEED_MM_S = 100.0
#: per-axis caps under it. Z: its whole counter span to the + firmware stop is
#: 100 mm, and the release run-on at 100 mm/s is ~100 mm [유도] — so 50. X and Y:
#: the rig cap. Nothing here has run above 50 mm/s yet (2026-10-27).
AXIS_MAX_SPEED_MM_S = {"X": RIG_MAX_SPEED_MM_S, "Y": RIG_MAX_SPEED_MM_S, "Z": 50.0}
#: Send (absolute moves) stays at whisker's 50 until a Send has run on this
#: controller (KNOWN_ISSUES 2026-10-26: Send / STOP ALL in motion unmeasured)
SEND_MAX_SPEED_MM_S = 50.0


def axis_max_speed(axis) -> float:
    """The speed cap of one axis (an Axis or its name)."""
    return AXIS_MAX_SPEED_MM_S[getattr(axis, "name", axis)]
#: pattern path: the worker polls Check_Axis_Is_Stop this often between segments
PATH_POLL_S = 0.05
#: an axis whose share of a path segment is smaller than this is not commanded
PATH_MIN_AXIS_MM = 0.05
#: shorter laps are refused (the UI's smallest pattern is a 5 mm back-and-forth)
PATH_MIN_LAP_MM = 1.0
#: a path segment that reports "stopped" further than this from its vertex is an
#: error (an early stop flag or a drive fault must not count as a lap)
PATH_REACH_TOL_MM = 1.0
#: the worker aborts a path when an axis reads further than this outside the
#: pattern's envelope (the targets themselves can never leave it)
PATH_ENVELOPE_SLACK_MM = 3.0
WHISKER_DIR_ENV = "WHISKER_VISION_DIR"
#: runstore kind of a disturbance-rig folder: data/YYYYMMDD/MMDD_HHMMSS_disturb/
LOG_KIND = "disturb"
#: whisker's motion_config SystemExits at import on any other WHISKER_PROFILE
VALID_PROFILES = ("", "night", "daytime")

SAMPLE_COLUMNS = ("host_epoch_s", "t_monotonic_s",
                  "x_mm", "y_mm", "z_mm", "gantry_busy",
                  "motor_deg", "torque_nm", "motor_enabled", "motor_busy",
                  "thr_pct_cmd", "pwm_us", "ramp_target_pct", "latched")
EVENT_COLUMNS = ("host_epoch_s", "t_monotonic_s", "kind", "detail")
#: the links' own buffer (ThrusterLink.stop_recording rows): command values,
#: one row per staircase step + a 1 Hz heartbeat — NOT a measurement.
THRUSTER_COLUMNS = ("host_epoch_s", "throttle_pct", "pwm_us")
_PRECISION = {"host_epoch_s": 6, "t_monotonic_s": 6, "x_mm": 3, "y_mm": 3,
              "z_mm": 3, "motor_deg": 2, "torque_nm": 3}


class WhiskerMissing(RuntimeError):
    """whisker_vision is not where we expect it (or has no motion_control.py)."""


# =============================================================================
# sys.path shim
# =============================================================================
def whisker_dir(override: "str | Path | None" = None) -> Path:
    """Where whisker_vision lives: --whisker-dir > $WHISKER_VISION_DIR > ~/Desktop."""
    raw = override or os.environ.get(WHISKER_DIR_ENV) or DEFAULT_WHISKER_DIR
    return Path(raw).expanduser()


def add_whisker_path(override: "str | Path | None" = None) -> Path:
    """Make whisker_vision's motion stack importable; returns its root.

    Never called at import time by anything under src/tests (./c3 test runs
    every src/tests/test_*.py on a machine that may not have whisker_vision):
    it raises WhiskerMissing / RuntimeError with the fix instead of exiting.
    """
    root = whisker_dir(override)
    if not (root / "motion_control.py").is_file():
        raise WhiskerMissing(
            f"whisker_vision not found at {root} (no motion_control.py). "
            f"Set {WHISKER_DIR_ENV}=/path/to/whisker_vision or pass --whisker-dir.")
    profile = os.environ.get("WHISKER_PROFILE", "")
    if profile.strip().lower() not in VALID_PROFILES:
        # motion_config.py:238-247 would SystemExit inside the import; say why
        raise RuntimeError(
            f"WHISKER_PROFILE={profile!r} is not night/daytime — whisker's "
            "motion_config refuses to import. Unset it (env -u WHISKER_PROFILE) "
            "or export WHISKER_PROFILE=night.")
    src, repo, wv = str(_SRC_DIR), str(REPO_ROOT), str(root.resolve())
    # enforce the order even when a launcher (python -m pytest, a cwd entry)
    # already put the repo root first: src, then the repo root, at the front
    for entry in (repo, src):
        while entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    if wv not in sys.path:
        sys.path.append(wv)             # LAST — see the module docstring
    return root


def sys_path_order_ok() -> bool:
    """src/ before the repo root before whisker_vision (when all three are in)."""
    try:
        i_src = sys.path.index(str(_SRC_DIR))
        i_repo = sys.path.index(str(REPO_ROOT))
        i_wv = sys.path.index(str(whisker_dir().resolve()))
    except ValueError:
        return False
    return i_src < i_repo < i_wv


# =============================================================================
# rig config (config/disturbance_rig.yaml) and CLI
# =============================================================================
@dataclass
class RigConfig:
    gantry_ip: str = "192.168.0.30"
    gantry_port: int = 8088
    speed_mm_s: float = 50.0
    can_channel: str = "can0"
    motor_vel_deg_s: float = 30.0
    motor_vel_max_deg_s: float = 90.0
    #: a hold-jog toward a side with NO taught home is one controller move of at
    #: most this length; while the button stays down the next one follows (a
    #: brief stop at each boundary). It bounds how far that axis runs if the
    #: release stop is not taken or the panel process dies mid-hold. Toward a
    #: taught home the move ends AT the home instead.
    jog_piece_mm: float = 300.0
    #: gantry patterns (X stays put): circle in the Y-Z plane / Y back-and-forth,
    #: centred on where the gantry stands at Start
    pattern_speed_mm_s: float = 10.0
    circle_radius_mm: float = 30.0
    line_range_mm: float = 100.0
    path_segment_mm: float = 8.0
    thruster_port: str = ("/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_"
                          "75330303934351903171-if00")
    thruster_max_pct: int = 100
    ramp_step_pct: int = 10
    ramp_interval_s: float = 3.0
    source: str = "defaults"

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def load_rig_config(path: "str | Path | None" = None) -> RigConfig:
    """config/disturbance_rig.yaml -> RigConfig; defaults when absent/unreadable."""
    cfg = RigConfig()
    p = Path(path) if path else DEFAULT_RIG_CONFIG
    if not p.is_file():
        cfg.source = f"defaults ({p} not found)"
        return cfg
    try:
        import yaml  # noqa: PLC0415 — optional
    except ImportError:
        cfg.source = f"defaults (PyYAML missing; {p} ignored)"
        return cfg
    try:
        raw = yaml.safe_load(p.read_text()) or {}
    except Exception as exc:  # noqa: BLE001
        cfg.source = f"defaults ({p}: {type(exc).__name__}: {exc})"
        return cfg
    g = raw.get("gantry") or {}
    m = raw.get("motor") or {}
    t = raw.get("thruster") or {}
    pt = raw.get("pattern") or {}
    cfg.gantry_ip = str(g.get("ip", cfg.gantry_ip))
    cfg.gantry_port = int(g.get("port", cfg.gantry_port))
    cfg.speed_mm_s = float(g.get("speed_mm_s", cfg.speed_mm_s))
    cfg.jog_piece_mm = float(g.get("jog_piece_mm", cfg.jog_piece_mm))
    cfg.pattern_speed_mm_s = float(pt.get("speed_mm_s", cfg.pattern_speed_mm_s))
    cfg.circle_radius_mm = float(pt.get("circle_radius_mm", cfg.circle_radius_mm))
    cfg.line_range_mm = float(pt.get("line_range_mm", cfg.line_range_mm))
    cfg.path_segment_mm = float(pt.get("segment_mm", cfg.path_segment_mm))
    cfg.can_channel = str(m.get("can_channel", cfg.can_channel))
    cfg.motor_vel_deg_s = float(m.get("vel_deg_s", cfg.motor_vel_deg_s))
    cfg.motor_vel_max_deg_s = float(m.get("vel_max_deg_s", cfg.motor_vel_max_deg_s))
    cfg.thruster_port = str(t.get("port", cfg.thruster_port))
    cfg.thruster_max_pct = int(t.get("max_pct", cfg.thruster_max_pct))
    cfg.ramp_step_pct = int(t.get("ramp_step_pct", cfg.ramp_step_pct))
    cfg.ramp_interval_s = float(t.get("ramp_interval_s", cfg.ramp_interval_s))
    cfg.source = str(p)
    return cfg


def rig_config_path_from_argv(argv: "list[str] | None" = None) -> "str | None":
    """Pre-scan for --rig-config so its values can seed the real parser's defaults."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--rig-config", default=None)
    known, _ = pre.parse_known_args(sys.argv[1:] if argv is None else argv)
    return known.rig_config


def build_arg_parser(cfg: "RigConfig | None" = None) -> argparse.ArgumentParser:
    cfg = cfg or RigConfig()
    ap = argparse.ArgumentParser(
        prog="disturbance_panel",
        description="Pool disturbance rig — gantry + rotation motor + T200 in one "
                    "manual panel, on whisker_vision's motion links.")
    ap.add_argument("--sim", action="store_true",
                    help="simulated devices (whisker_vision/motion_simulator); "
                         "no gantry / CAN / serial needed")
    ap.add_argument("--whisker-dir", default=None,
                    help=f"whisker_vision checkout (default ${WHISKER_DIR_ENV} "
                         f"or {DEFAULT_WHISKER_DIR})")
    ap.add_argument("--rig-config", default=str(DEFAULT_RIG_CONFIG),
                    help="rig YAML (default: %(default)s)")
    g = ap.add_argument_group("gantry")
    g.add_argument("--gantry-ip", default=cfg.gantry_ip)
    g.add_argument("--gantry-port", type=int, default=cfg.gantry_port)
    g.add_argument("--speed", type=float, default=cfg.speed_mm_s,
                   metavar="MM_S",
                   help=f"default move/jog speed (clamped 0.1..{RIG_MAX_SPEED_MM_S:g})")
    g.add_argument("--jog-piece", type=float, default=cfg.jog_piece_mm, metavar="MM",
                   help="hold-jog toward a side with no taught home: one continuous move "
                        "covers at most this distance; the next follows while the button "
                        "is held (default %(default)s)")
    g.add_argument("--limits-file", default=None, metavar="JSON",
                   help="where the taught homes (soft limits) are saved for 'Restore saved' "
                        f"(default <data-root>/{LIMITS_FILE_NAME})")
    m = ap.add_argument_group("rotation motor")
    m.add_argument("--can-channel", default=cfg.can_channel)
    m.add_argument("--motor-vel", type=float, default=cfg.motor_vel_deg_s,
                   metavar="DEG_S", help="default Go-theta speed; also the STOP hold speed")
    m.add_argument("--motor-vel-max", type=float, default=cfg.motor_vel_max_deg_s,
                   metavar="DEG_S", help="UI cap for the motor speed")
    m.add_argument("--no-motor-interlock", action="store_true",
                   help="allow thrust while the motor is NOT enabled/holding "
                        "(bench tests only — a limp mount swings under the jet)")
    t = ap.add_argument_group("thruster")
    t.add_argument("--thruster-port", default=cfg.thruster_port)
    t.add_argument("--thruster-max", type=int, default=cfg.thruster_max_pct,
                   metavar="PCT", help="cap |throttle| for this session (1..100)")
    t.add_argument("--ramp-step", type=int, default=cfg.ramp_step_pct,
                   metavar="PCT", help="staircase step per interval (1..100)")
    t.add_argument("--ramp-interval", type=float, default=cfg.ramp_interval_s,
                   metavar="S", help="seconds between staircase steps (0 = no staircase; "
                                     "the ESC/supply misbehaved on abrupt changes)")
    lg = ap.add_argument_group("logging")
    lg.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT),
                    help="run folders go to <data-root>/YYYYMMDD/MMDD_HHMMSS_disturb/ "
                         "(default: the repo's data/)")
    lg.add_argument("--no-autolog", action="store_true",
                    help="do not start a log folder when the first device connects")
    lg.add_argument("--rig-note", action="append", default=[], metavar="TEXT",
                    help="free text into meta.json rig_notes (which way +%% pushes, "
                         "what motor 0 deg points at, mount offset); repeatable")
    lg.add_argument("--companion", default=None, metavar="RUN_DIR",
                    help="the ROV run folder this disturbance session belongs to "
                         "(recorded in meta.json companion_run)")
    return ap


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(float(value), lo), hi)


@dataclass
class PanelOptions:
    """Panel-side policy (not whisker's MotionSettings)."""
    motor_interlock: bool = True
    autolog: bool = True
    data_root: Path = DEFAULT_DATA_ROOT
    rig_notes: list = field(default_factory=list)
    motor_vel_max_deg_s: float = 90.0
    rig_config_source: str = "defaults"
    companion_run: "str | None" = None       # the ROV run this disturbance belongs to
    jog_piece_mm: float = 300.0
    limits_file: Path = DEFAULT_DATA_ROOT / LIMITS_FILE_NAME
    pattern_speed_mm_s: float = 10.0
    circle_radius_mm: float = 30.0
    line_range_mm: float = 100.0
    path_segment_mm: float = 8.0

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["data_root"] = str(self.data_root)
        d["limits_file"] = str(self.limits_file)
        return d


def settings_from_args(args: argparse.Namespace, cfg: "RigConfig | None" = None):
    """CLI (+ rig config) -> whisker MotionSettings with every rig value explicit.

    Requires add_whisker_path() first (motion_config is imported here).
    """
    from motion_config import MAX_MOTOR_VEL_DEG_S, MotionSettings  # noqa: PLC0415
    cfg = cfg or RigConfig()
    settings = MotionSettings(
        simulator=bool(args.sim),
        gantry_ip=str(args.gantry_ip).strip(),
        gantry_port=int(args.gantry_port),
        can_channel=str(args.can_channel).strip(),
        thruster_port=str(args.thruster_port).strip(),
        # no client limits: this controller's firmware pair is invalid (200/200),
        # so every axis connects relative-only and the panel installs the TAUGHT
        # homes per axis (RigGantryLink.set_axis_limits)
        client_soft_limits_mm={})
    settings.speed_mm_s = _clamp(args.speed, 0.1, RIG_MAX_SPEED_MM_S)
    vel_max = _clamp(args.motor_vel_max, 0.5, MAX_MOTOR_VEL_DEG_S)
    settings.motor_vel_deg_s = _clamp(args.motor_vel, 0.5, vel_max)
    settings.thruster_max_pct = int(_clamp(args.thruster_max, 1, 100))
    settings.thruster_ramp_step_pct = int(_clamp(args.ramp_step, 1, 100))
    settings.thruster_ramp_interval_s = max(0.0, float(args.ramp_interval))
    # whisker-demo plumbing: never feed the live whisker demo / monitor from here
    settings.pose_udp_port = 0
    settings.estimate_udp_port = 0
    settings.monitor_port = 0
    return settings


def options_from_args(args: argparse.Namespace, cfg: "RigConfig | None" = None) -> PanelOptions:
    cfg = cfg or RigConfig()
    data_root = Path(args.data_root).expanduser()
    limits_file = getattr(args, "limits_file", None)
    return PanelOptions(
        motor_interlock=not bool(args.no_motor_interlock),
        autolog=not bool(args.no_autolog),
        data_root=data_root,
        rig_notes=list(args.rig_note or []),
        motor_vel_max_deg_s=_clamp(args.motor_vel_max, 0.5, 180.0),
        rig_config_source=cfg.source,
        companion_run=(str(args.companion) if getattr(args, "companion", None) else None),
        jog_piece_mm=_clamp(getattr(args, "jog_piece", cfg.jog_piece_mm), 5.0, 2000.0),
        limits_file=(Path(limits_file).expanduser() if limits_file
                     else data_root / LIMITS_FILE_NAME),
        pattern_speed_mm_s=_clamp(cfg.pattern_speed_mm_s, PATTERN_SPEED_RANGE[0],
                                  PATTERN_SPEED_RANGE[1]),
        circle_radius_mm=_clamp(cfg.circle_radius_mm, CIRCLE_RADIUS_RANGE[0],
                                CIRCLE_RADIUS_RANGE[1]),
        line_range_mm=_clamp(cfg.line_range_mm, LINE_RANGE_RANGE[0], LINE_RANGE_RANGE[1]),
        path_segment_mm=_clamp(cfg.path_segment_mm, 2.0, 20.0))


def settings_summary(settings, options: PanelOptions) -> dict:
    """What meta.json records about the commanded configuration."""
    base = settings.summary()
    base.update({
        "speed_mm_s": settings.speed_mm_s,
        "motor_vel_deg_s": settings.motor_vel_deg_s,
        "motor_vel_max_deg_s": options.motor_vel_max_deg_s,
        "thruster_max_pct": settings.thruster_max_pct,
        "thruster_ramp_step_pct": settings.thruster_ramp_step_pct,
        "thruster_ramp_interval_s": settings.thruster_ramp_interval_s,
        "client_soft_limits_mm": {k: list(v) for k, v in
                                  settings.client_soft_limits_mm.items()},
        "motor_interlock": options.motor_interlock,
        "autolog": options.autolog,
        "rig_config": options.rig_config_source,
        "jog_piece_mm": options.jog_piece_mm,
        "jog_stop_mode": JOG_STOP_MODE,
        "pattern_defaults": {"speed_mm_s": options.pattern_speed_mm_s,
                             "circle_radius_mm": options.circle_radius_mm,
                             "line_range_mm": options.line_range_mm,
                             "segment_mm": options.path_segment_mm},
        "limits_file": str(options.limits_file),
        "acc_mm_s2": settings.acc_mm_s2,
        "companion_run": options.companion_run,
    })
    return base


# =============================================================================
# gantry patterns (pure geometry; the rig link executes them)
# =============================================================================
#: UI ranges. Small boxes on purpose: a pattern runs unattended. The speed is the
#: commanded cruise; on short chords the acceleration caps it (a chord of L mm
#: never exceeds sqrt(acc·L): ~20 mm/s on the 8 mm circle chords at 50 mm/s²
#: [유도]).
PATTERN_SPEED_RANGE = (1.0, RIG_MAX_SPEED_MM_S)
CIRCLE_RADIUS_RANGE = (5.0, 50.0)
LINE_RANGE_RANGE = (5.0, 100.0)       # Y back-and-forth, peak to peak


def circle_points(center: "tuple[float, float]", radius_mm: float,
                  segment_mm: float = 8.0) -> "list[tuple[float, float]]":
    """One lap of a circle as polygon vertices, counter-clockwise from angle 0
    (center + (r, 0)), ending back on that start vertex. The controller only
    runs straight single-axis moves, so a circle is a chain of short chords:
    at r 30 / 8 mm the 24 chords stay within 0.26 mm of the circle [유도:
    r(1 - cos(pi/N))]."""
    c0, c1 = float(center[0]), float(center[1])
    r = float(radius_mm)
    if not (math.isfinite(r) and r > 0 and math.isfinite(c0) and math.isfinite(c1)):
        raise ValueError("circle needs a finite centre and a positive radius")
    n = max(12, int(math.ceil(2.0 * math.pi * r / max(float(segment_mm), 1.0))))
    return [(c0 + r * math.cos(2.0 * math.pi * k / n), c1 + r * math.sin(2.0 * math.pi * k / n))
            for k in range(1, n + 1)]


def line_points(center: float, range_mm: float) -> "list[tuple[float]]":
    """One lap of a back-and-forth: centre + range/2, then centre - range/2."""
    half = float(range_mm) / 2.0
    c = float(center)
    if not (math.isfinite(half) and half > 0 and math.isfinite(c)):
        raise ValueError("back-and-forth needs a finite centre and a positive range")
    return [(c + half,), (c - half,)]


def path_envelope(start, points) -> "list[tuple[float, float]]":
    """Per-axis (lo, hi) of the start position and every vertex."""
    cols = list(zip(tuple(start), *[tuple(p) for p in points]))
    return [(min(c), max(c)) for c in cols]


# =============================================================================
# taught soft limits ("−X home" / "+X home")
# =============================================================================
def clean_taught_limits(raw: Any) -> dict:
    """{axis: {"lo": mm|None, "hi": mm|None, "lo_epoch_s", "hi_epoch_s"}} with
    anything malformed dropped; an axis whose pair is not lo < hi is dropped
    whole (a half-trusted pair is worse than none)."""
    out: dict = {}
    if not isinstance(raw, dict):
        return out
    for axis in AXIS_NAMES:
        entry = raw.get(axis)
        if not isinstance(entry, dict):
            continue
        clean: dict = {}
        for side in ("lo", "hi"):
            v = entry.get(side)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                clean[side] = float(v)
                t = entry.get(f"{side}_epoch_s")
                clean[f"{side}_epoch_s"] = (float(t) if isinstance(t, (int, float))
                                            and math.isfinite(t) else None)
        if "lo" in clean and "hi" in clean and not clean["lo"] < clean["hi"]:
            continue
        if clean:
            out[axis] = clean
    return out


def _clean_position(raw: Any) -> "tuple | None":
    if (isinstance(raw, (list, tuple)) and len(raw) == 3
            and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) for v in raw)):
        return tuple(float(v) for v in raw)
    return None


def load_taught_limits(path: "str | Path") -> dict:
    """The saved homes -> {"axes": {...}, "saved_epoch_s": float|None,
    "last_pos_mm": (x, y, z)|None, "error": str|None}. Missing file = no
    axes, no error. last_pos_mm is the last position the panel saw with
    these homes in force: a restore must find the gantry exactly there."""
    empty = {"axes": {}, "saved_epoch_s": None, "last_pos_mm": None, "error": None}
    p = Path(path)
    if not p.is_file():
        return empty
    try:
        raw = json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        return dict(empty, error=f"{p}: {exc}")
    if not isinstance(raw, dict):
        return dict(empty, error=f"{p}: not a JSON object")
    saved = raw.get("saved_epoch_s")
    return {"axes": clean_taught_limits(raw.get("axes")),
            "saved_epoch_s": float(saved) if isinstance(saved, (int, float)) else None,
            "last_pos_mm": _clean_position(raw.get("last_pos_mm")),
            "error": None}


def save_taught_limits(path: "str | Path", axes: dict,
                       last_pos_mm: "tuple | None" = None) -> "str | None":
    """Atomic write of the taught homes and the position the gantry stood at;
    an error text instead of raising (the panel keeps running without it)."""
    p = Path(path)
    now = time.time()
    doc = {
        "what": "disturbance rig taught soft limits (./c3 disturb: −X home / +X home)",
        "frame": "FMC4030 controller counter, mm. The counter restarts at 0 wherever the "
                 "axes stand when the controller is powered up: after a power cycle "
                 "these numbers point somewhere else — Clear and re-teach.",
        "saved_epoch_s": now,
        "saved_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
        "axes": clean_taught_limits(axes),
        "last_pos_mm": list(_clean_position(last_pos_mm) or ()) or None,
    }
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(doc, indent=2) + "\n")
        os.replace(tmp, p)
    except OSError as exc:
        return f"{p}: {exc}"
    return None


# =============================================================================
# links
# =============================================================================
_RIG_GANTRY_CLS = None


def rig_gantry_link_class():
    """GantryLink + what this rig needs, built lazily (motion_control is only
    importable after add_whisker_path()).

    set_axis_limits(): the link reads its per-axis limits ONCE, at connect,
    but this rig's limits are TAUGHT while connected ("−X home" / "+X home" =
    the current counter position). The controller's firmware pair is invalid
    (200/200), so every axis connects relative-only and the panel installs an
    axis's taught pair once both ends are set.

    request_move(): whisker's, plus the jog-stop check; capped at
    SEND_MAX_SPEED_MM_S (whisker's 50) until a Send has run on this controller.

    Speeds: jog and path clamp per axis (AXIS_MAX_SPEED_MM_S: X/Y 100, Z 50).

    request_jog_relative(): request_relative_move() caps every relative move
    at 10 mm (MAX_RELATIVE_STEP_MM), which turns a held jog into a stutter of
    10 mm hops. A hold-jog is ONE controller-relative move per press: to the
    taught home on that side, or at most jog_piece_mm where none is taught.

    jog_stop(): the release. GantryLink only has the LATCHING stop_all(), and
    its stop sequence (stop_run, then Stop_Single_Axis mode 2 on each axis)
    did NOT stop a relative jog on this controller: three 300 mm jogs ran to
    their end through release, STOP ALL and a disconnect
    (data/20261026/1026_131512_disturb/samples.csv, X −90→−390, −390→−690,
    −690→−990 mm at 20 mm/s). Which call failed (stop_run blocking what
    follows, or mode 2 being ignored on a jog) is not separated. The vendor's
    Qt demo and ./c3 gantry release a jog with Stop_Single_Axis mode 1
    (decelerate); jog_stop() sends that, without latching the link, so the
    next press works at once. It stopped 18/18 releases on this controller the
    same afternoon (data/20261026/1026_140909_disturb/events.csv).

    _run_stop_sequence(): STOP ALL / stop() / Recover send mode 1 to every axis,
    then mode 2 — and NOT stop_run (only needed for homing/interpolation, which
    this panel never uses, and one of the two unseparated 10-26 suspects).

    request_path(): a gantry PATTERN (Y-Z circle, Y back-and-forth) as ONE
    long command the worker runs lap after lap until cancelled. Each segment is
    the same absolute single-axis jog request_move() sends, issued for every
    moving axis inside one SDK-lock block, with speed AND acceleration scaled by
    the axis's share of the segment, so the axes start and end together and
    the tool moves on the straight chord. Not the controller's own
    Line_2Axis / Arc_2Axis: they interpolate in controller UNITS, and Y
    (2.5 mm/unit) and Z (0.5 mm/unit) differ 5x, so a unit circle is an
    ellipse in mm; nor have they ever run on this rig. Every target is checked
    against the pattern's envelope (start position and vertices) and any
    installed limits under the SDK lock right before it is sent, so whatever
    happens the gantry never gets a target outside that box. Cancel: jog_stop()
    (non-latching, mode 1 on every axis) or STOP ALL — both bump the generation
    the worker re-checks under the SDK lock, so a segment is on the wire
    before the stop or never.

    Internals used (whisker_vision is not modified; test_disturbance_core pins
    them so a refactor there fails loudly): _state_lock, _sdk_lock,
    _soft_min_mm, _soft_max_mm, _generation, _pending, _controller, _stop_done,
    _shutdown, _settings (motion_control.py:75-90), _require_commandable, _queue
    (:326, :337), _command_current, _finish_command, _poll_positions, _execute
    (:364, overridden for the "path" kind only), _run_stop_sequence (:289-314)
    and the "relative" command kind executed by _execute_relative (:399-425,
    which re-checks the endpoint against any limits under the SDK lock).
    """
    global _RIG_GANTRY_CLS
    if _RIG_GANTRY_CLS is None:
        import threading  # noqa: PLC0415

        from motion_config import mm_to_units, units_to_mm  # noqa: PLC0415
        from motion_control import AXES, GantryLink  # noqa: PLC0415

        class RigGantryLink(GantryLink):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self._jog_stop_start_lock = threading.Lock()
                self._jog_stop_done = threading.Event()
                self._jog_stop_done.set()
                #: set when the last jog_stop() could not be sent; cleared by the next
                self.jog_stop_error: "str | None" = None
                #: (lap, vertex index, vertices per lap) of the running path
                self._path_progress: "tuple | None" = None
                #: the running path's axes: STOP ALL stops them first
                self._path_axes: tuple = ()

            def set_axis_limits(self, axis, lo: "float | None", hi: "float | None") -> None:
                """Replace one axis's software limits; (None, None) = relative-only."""
                if (lo is None) != (hi is None):
                    raise ValueError("set both limits or neither")
                if lo is not None and not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
                    raise ValueError(f"bad limits {lo}..{hi}")
                with self._state_lock:
                    self._soft_min_mm[int(axis)] = None if lo is None else float(lo)
                    self._soft_max_mm[int(axis)] = None if hi is None else float(hi)

            def request_jog_relative(self, axis, delta_mm: float, speed_mm_s: float,
                                     max_mm: float) -> None:
                """request_relative_move() with the 10 mm cap replaced by max_mm.

                Same checks otherwise: commandable (connected, not latched, not
                busy), finite non-zero delta, fresh live position, endpoint inside
                any limits the axis has; the worker re-checks under the SDK lock."""
                self._require_commandable()
                if not self._jog_stop_done.is_set():
                    raise RuntimeError("the previous jog stop is still being sent")
                delta = float(delta_mm)
                if not math.isfinite(delta) or delta == 0.0:
                    raise ValueError("jog must be a finite non-zero distance")
                if not (math.isfinite(max_mm) and max_mm > 0) or abs(delta) > max_mm:
                    raise ValueError(f"jog of {delta:+.1f} mm exceeds {max_mm:g} mm")
                positions = self.positions_mm()
                if positions is None or not self.telemetry_fresh():
                    raise RuntimeError("jog requires a fresh live position")
                current = positions[int(axis)]
                if not math.isfinite(current):
                    raise RuntimeError(f"{axis.name} live position is invalid")
                lo, hi = self.soft_limits_mm()
                a_lo, a_hi = lo[int(axis)], hi[int(axis)]
                endpoint = current + delta
                if a_lo is not None and a_hi is not None and not (a_lo <= endpoint <= a_hi):
                    raise ValueError(f"{axis.name} jog endpoint {endpoint:.1f} mm outside "
                                     f"soft limits {a_lo:.1f}..{a_hi:.1f} mm")
                speed = float(speed_mm_s)
                if not math.isfinite(speed):
                    raise ValueError("gantry speed must be finite")
                speed = min(max(speed, 0.1), axis_max_speed(axis))
                self._queue({"kind": "relative", "axis": axis, "delta_mm": delta,
                             "speed_mm_s": speed})

            def request_move(self, targets_mm, speed_mm_s: float) -> None:
                """GantryLink.request_move() (motion_control.py:183-205) plus the
                jog-stop check; same checks: both limits of every target axis,
                target inside them, finite; speed ≤ SEND_MAX_SPEED_MM_S."""
                self._require_commandable()
                if not self._jog_stop_done.is_set():
                    raise RuntimeError("the previous jog stop is still being sent")
                if not targets_mm:
                    return
                lo, hi = self.soft_limits_mm()
                for axis, target in targets_mm.items():
                    if not math.isfinite(target):
                        raise ValueError(f"{axis.name} target is not finite")
                    a_lo, a_hi = lo[int(axis)], hi[int(axis)]
                    if a_lo is None or a_hi is None:
                        raise RuntimeError(f"absolute {axis.name} move disabled: soft limits "
                                           "are missing")
                    if not (a_lo <= target <= a_hi):
                        raise ValueError(f"{axis.name} target {target:.1f} mm outside soft "
                                         f"limits {a_lo:.1f}..{a_hi:.1f} mm")
                speed = float(speed_mm_s)
                if not math.isfinite(speed):
                    raise ValueError("gantry speed must be finite")
                speed = min(max(speed, 0.1), SEND_MAX_SPEED_MM_S,
                            min(axis_max_speed(a) for a in targets_mm))
                self._queue({"kind": "move", "targets": dict(targets_mm),
                             "speed_mm_s": speed})

            @property
            def jog_stop_running(self) -> bool:
                return not self._jog_stop_done.is_set()

            def jog_stop(self, axis, *more) -> bool:
                """Release of a hold-jog (or a pattern's Stop) — NOT latching (the
                next press works).

                The generation bump cancels a queued or dequeued-but-unsent jog:
                the worker re-checks the generation under the SDK lock before it
                sends, so a jog is either on the wire before this stop or never.
                The stop itself (Stop_Single_Axis mode 1 on ``axis`` and ``more``
                — the moving ones — then the rest) goes out from a thread like
                stop_all(wait=False):
                the GUI thread never waits on the SDK. False when there is no
                controller or a jog stop is already being sent (that one covers
                every axis)."""
                with self._state_lock:
                    self._generation += 1
                    self._pending = None
                controller = self._controller
                if controller is None:
                    return False
                with self._jog_stop_start_lock:
                    if not self._jog_stop_done.is_set():
                        return False
                    self._jog_stop_done.clear()
                    self.jog_stop_error = None
                    try:
                        first = [axis] + [a for a in more if a != axis]
                        threading.Thread(target=self._run_jog_stop, args=(controller, first),
                                         name="gantry-jog-stop", daemon=True).start()
                    except Exception as exc:  # noqa: BLE001 — e.g. no threads left
                        self.jog_stop_error = f"gantry jog stop: thread start failed: {exc}"
                        self._jog_stop_done.set()
                        return False
                return True

            def _run_jog_stop(self, controller, first) -> None:
                failures = []
                try:
                    with self._sdk_lock:
                        for a in list(first) + [b for b in AXES if b not in first]:
                            try:
                                controller.stop_axis(a, mode=JOG_STOP_MODE)
                            except Exception as exc:  # noqa: BLE001
                                failures.append(f"{a.name}: {exc}")
                    if failures:
                        message = "gantry jog stop: " + "; ".join(failures)
                        self.jog_stop_error = message
                        self.last_command_error = message
                finally:
                    self._jog_stop_done.set()

            def _run_stop_sequence(self) -> None:
                """Every axis: the decelerating per-axis stop (the one that stops a
                jog release on this controller), then the immediate one.
                whisker's stop_run is left out: it is only needed while homing or
                interpolating (neither happens here) and it is one of the two
                unseparated suspects of 10-26 — it may block what follows."""
                controller = self._controller
                if controller is None:
                    self._stop_done.set()
                    return
                failures = []
                first = tuple(self._path_axes)        # a running pattern's moving axes
                order = list(first) + [a for a in AXES if a not in first]
                try:
                    with self._sdk_lock:
                        for axis in order:
                            try:
                                controller.stop_axis(axis, mode=JOG_STOP_MODE)
                            except Exception as exc:  # noqa: BLE001
                                failures.append(f"{axis.name} mode {JOG_STOP_MODE}: {exc}")
                        for axis in order:
                            try:
                                controller.stop_axis(axis, mode=2)
                            except Exception as exc:  # noqa: BLE001
                                failures.append(f"{axis.name}: {exc}")
                    if failures:
                        message = "gantry stop: " + "; ".join(failures)
                        self.error = message
                        self.last_command_error = message
                finally:
                    self._stop_done.set()

            # ---------------------------------------------------------- patterns
            @property
            def path_progress(self) -> "tuple | None":
                with self._state_lock:
                    return self._path_progress

            def request_path(self, axes, points, speed_mm_s: float, envelope) -> None:
                """Queue a pattern: ``points`` is ONE lap of absolute vertices
                (tuples aligned with ``axes``), repeated until cancelled;
                ``envelope`` a (lo, hi) per axis that every target and every
                live reading of those axes must stay in."""
                self._require_commandable()
                if not self._jog_stop_done.is_set():
                    raise RuntimeError("the previous jog stop is still being sent")
                axes = list(axes)
                if not axes or len(set(axes)) != len(axes):
                    raise ValueError("a path needs distinct axes")
                pts = [tuple(float(v) for v in p) for p in points]
                if len(pts) < 2 or any(len(p) != len(axes) for p in pts):
                    raise ValueError("a path needs at least two vertices, one value per axis")
                lap_mm = sum(math.dist(pts[i - 1], pts[i]) for i in range(len(pts)))
                if not lap_mm >= PATH_MIN_LAP_MM:
                    raise ValueError(f"a path lap must be at least {PATH_MIN_LAP_MM:g} mm long")
                env = [(float(lo), float(hi)) for lo, hi in envelope]
                if len(env) != len(axes):
                    raise ValueError("one envelope pair per axis")
                lo_lim, hi_lim = self.soft_limits_mm()
                for k, axis in enumerate(axes):
                    e_lo, e_hi = env[k]
                    if not (math.isfinite(e_lo) and math.isfinite(e_hi) and e_lo <= e_hi):
                        raise ValueError(f"{axis.name} envelope {e_lo}..{e_hi} is invalid")
                    a_lo, a_hi = lo_lim[int(axis)], hi_lim[int(axis)]
                    for p in pts:
                        v = p[k]
                        if not (math.isfinite(v) and e_lo - 1e-6 <= v <= e_hi + 1e-6):
                            raise ValueError(f"{axis.name} vertex {v} outside the envelope "
                                             f"{e_lo:.1f}..{e_hi:.1f} mm")
                        if a_lo is not None and a_hi is not None and not (a_lo <= v <= a_hi):
                            raise ValueError(f"{axis.name} vertex {v:.1f} mm outside soft "
                                             f"limits {a_lo:.1f}..{a_hi:.1f} mm")
                positions = self.positions_mm()
                if positions is None or not self.telemetry_fresh():
                    raise RuntimeError("a path requires a fresh live position")
                for k, axis in enumerate(axes):
                    cur = positions[int(axis)]
                    if not (env[k][0] - 1e-6 <= cur <= env[k][1] + 1e-6):
                        raise ValueError(f"{axis.name} at {cur:.1f} mm is outside the path "
                                         "envelope (it must start inside it)")
                speed = float(speed_mm_s)
                if not math.isfinite(speed):
                    raise ValueError("gantry speed must be finite")
                speed = min(max(speed, 0.1), RIG_MAX_SPEED_MM_S)
                with self._state_lock:
                    self._path_progress = None
                self._queue({"kind": "path", "axes": axes, "points": pts, "envelope": env,
                             "speed_mm_s": speed})

            def _execute(self, command: dict) -> None:
                if command.get("kind") != "path":
                    super()._execute(command)
                    return
                self._path_axes = tuple(command["axes"])
                try:
                    self._execute_path(command)
                except Exception as exc:  # noqa: BLE001 — same contract as GantryLink._execute
                    message = f"gantry command: {exc}"
                    self.error = message
                    self.last_command_error = message
                finally:
                    self._path_axes = ()
                    self._finish_command()

            def _execute_path(self, command: dict) -> None:
                """Vertex after vertex, lap after lap. Each segment starts from the
                axes' position read under the SDK lock (so a segment that ended
                short does not bend the next chord) and is checked against the
                envelope and the limits right before it is sent."""
                axes, pts, env = command["axes"], command["points"], command["envelope"]
                speed = float(command["speed_mm_s"])
                acc = float(self._settings.acc_mm_s2)
                n = len(pts)
                lap = 0
                with self._state_lock:
                    self._path_progress = (0, 0, n)
                while not self._shutdown.is_set():
                    for i, target in enumerate(pts):
                        moving = []
                        with self._sdk_lock:
                            if not self._command_current(command):
                                return
                            lo_lim, hi_lim = self.soft_limits_mm()
                            now_mm = []
                            for k, axis in enumerate(axes):
                                e_lo, e_hi = env[k]
                                cur = units_to_mm(
                                    axis, float(self._controller.get_axis_position(axis)))
                                if not (e_lo - PATH_ENVELOPE_SLACK_MM <= cur
                                        <= e_hi + PATH_ENVELOPE_SLACK_MM):
                                    raise RuntimeError(
                                        f"{axis.name} reads {cur:.1f} mm, outside the "
                                        f"pattern envelope {e_lo:.1f}..{e_hi:.1f} mm")
                                t = target[k]
                                a_lo, a_hi = lo_lim[int(axis)], hi_lim[int(axis)]
                                if not (e_lo - 1e-6 <= t <= e_hi + 1e-6) or (
                                        a_lo is not None and a_hi is not None
                                        and not (a_lo <= t <= a_hi)):
                                    raise ValueError(f"{axis.name} target {t:.1f} mm "
                                                     "outside the envelope / soft limits")
                                now_mm.append(cur)
                            deltas = [t - q for t, q in zip(target, now_mm)]
                            length = math.sqrt(sum(d * d for d in deltas))
                            # slow the whole chord so no axis exceeds its own cap
                            seg_speed = speed
                            for k, axis in enumerate(axes):
                                if length > 0 and abs(deltas[k]) > 0:
                                    seg_speed = min(seg_speed, axis_max_speed(axis)
                                                    * length / abs(deltas[k]))
                            for k, axis in enumerate(axes):
                                d = abs(deltas[k])
                                if length < PATH_MIN_AXIS_MM or d < PATH_MIN_AXIS_MM:
                                    continue
                                share = d / length
                                scale = float(units_to_mm(axis, 1.0))
                                self._controller.jog_single_axis(
                                    axis, mm_to_units(axis, target[k]),
                                    seg_speed * share / scale, acc * share / scale,
                                    acc * share / scale, relative=False)
                                moving.append(axis)
                        if moving and not self._wait_axes_stopped(moving, command):
                            return
                        if moving:
                            pos = self.positions_mm()       # polled at the stop just now
                            for k, axis in enumerate(axes):
                                miss = abs(pos[int(axis)] - target[k]) if pos else float("inf")
                                if axis in moving and miss > PATH_REACH_TOL_MM:
                                    raise RuntimeError(
                                        f"{axis.name} reported stopped {miss:.1f} mm from its "
                                        f"vertex {target[k]:.1f} mm")
                        if not moving:
                            # already there: still poll and pace, never spin
                            self._poll_positions()
                            time.sleep(PATH_POLL_S)
                        with self._state_lock:
                            self._path_progress = (lap, i + 1, n)
                    lap += 1

            def _wait_axes_stopped(self, axes, command: dict) -> bool:
                """Poll until every axis reports stopped, refreshing the cached
                positions at least once per segment (the run loop does not poll
                while a path runs); False once the command was cancelled."""
                time.sleep(PATH_POLL_S)
                while not self._shutdown.is_set():
                    with self._sdk_lock:
                        if not self._command_current(command):
                            return False
                        stopped = all(self._controller.is_axis_stopped(a) for a in axes)
                    self._poll_positions()
                    if stopped:
                        return self._command_current(command)
                    time.sleep(PATH_POLL_S)
                return False

        _RIG_GANTRY_CLS = RigGantryLink
    return _RIG_GANTRY_CLS


def rig_gantry_link(controller_factory, settings):
    return rig_gantry_link_class()(controller_factory, settings)


def make_gantry_link(settings):
    """Same devices as whisker's build_gantry_link, wrapped in RigGantryLink."""
    if settings.simulator:
        from motion_simulator import SimGantryController  # noqa: PLC0415
        factory = SimGantryController
    else:
        def factory():
            from motion.fmc4030 import FMC4030Controller  # noqa: PLC0415
            return FMC4030Controller()
    return rig_gantry_link(factory, settings)


def make_motor_link(settings):
    from motion_config import build_motor_link  # noqa: PLC0415
    return build_motor_link(settings)


def make_thruster_link(settings):
    """ThrusterLink with a serial WRITE timeout on the real port.

    whisker's build_thruster_link opens pyserial with timeout=0.2 and no
    write_timeout (motion/thruster.py:48-49); ThrusterLink.stop_all() then
    writes on the GUI thread under the SDK lock (motion_control.py:1047-1052),
    so a USB CDC device that vanished mid-run could block STOP ALL forever.
    ThrusterSerial accepts a serial_factory, so the panel supplies one.
    """
    from motion_control import ThrusterLink  # noqa: PLC0415
    if settings.simulator:
        from motion_config import build_thruster_link  # noqa: PLC0415
        return build_thruster_link(settings)
    port, baud = settings.thruster_port, settings.thruster_baud

    def factory():
        import serial  # noqa: PLC0415 — lazy like whisker's own real branch
        from motion.thruster import ThrusterSerial  # noqa: PLC0415
        return ThrusterSerial(port, baud, serial_factory=lambda: serial.Serial(
            port, baud, timeout=0.2, write_timeout=1.0, exclusive=True))
    return ThrusterLink(factory, settings)


# =============================================================================
# helpers: git, ss
# =============================================================================
def _run(argv: list, timeout: float = 5.0) -> "subprocess.CompletedProcess | None":
    """subprocess.run with close_fds (the default) — never QProcess/fork: the
    FMC4030 library opens its socket without close-on-exec and a child would
    keep the controller's single connection alive (whisker motion_panel.py:1180-1198)."""
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None


def git_head(path: "str | Path") -> dict:
    """{'sha', 'branch', 'dirty'} of a checkout; None fields when unknown. Never raises."""
    out = {"sha": None, "branch": None, "dirty": None}
    p = str(path)
    r = _run(["git", "-C", p, "rev-parse", "HEAD"])
    if r is not None and r.returncode == 0:
        out["sha"] = r.stdout.strip() or None
    r = _run(["git", "-C", p, "rev-parse", "--abbrev-ref", "HEAD"])
    if r is not None and r.returncode == 0:
        out["branch"] = r.stdout.strip() or None
    r = _run(["git", "-C", p, "status", "--porcelain", "--untracked-files=no"])
    if r is not None and r.returncode == 0:
        out["dirty"] = bool(r.stdout.strip())
    return out


def parse_ss_estab(text: str, ip: str, port: int) -> "str | None":
    """The `ss -tn` line holding an ESTABLISHED connection to ip:port, else None."""
    needle = f"{ip}:{port}"
    for line in text.splitlines():
        if needle in line and "ESTAB" in line:
            return line.strip()
    return None


def gantry_link_held(ip: str, port: int) -> "tuple[bool | None, str]":
    """Is some process on this host holding the controller's single TCP client slot?

    (True, line) when an ESTABLISHED socket exists, (False, note) when not,
    (None, why) when it could not be checked. Read-only: never connects.
    """
    r = _run(["ss", "-tn"])
    if r is None or r.returncode != 0:
        return None, "ss -tn unavailable — occupancy unknown"
    line = parse_ss_estab(r.stdout, ip, port)
    if line:
        return True, line
    return False, f"no established connection to {ip}:{port} on this host"


# =============================================================================
# the run-folder logger
# =============================================================================
def _fmt(column: str, value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.{_PRECISION.get(column, 4)}f}"
    return str(value)


class DisturbLog:
    """One run folder: samples.csv (streamed, flushed per row), events.csv,
    thruster.csv (the link's own command rows, written at close) and meta.json.

    Written from the GUI thread from CACHED link getters only — logging never
    touches a device. Streaming + flush-per-row is deliberate: the links' own
    start_recording() buffers live in memory until stop_recording(), so a crash
    (PyQt5 aborts on an unhandled slot exception; this host has hard-crashed)
    would lose the whole run.
    """

    def __init__(self, data_root: "str | Path", meta: dict, kind: str = LOG_KIND,
                 when: "float | None" = None) -> None:
        from rov_gui import runstore  # noqa: PLC0415 — needs the repo root on sys.path
        self.dir: Path = runstore.run_dir(Path(data_root), when=when, kind=kind,
                                          join=False)
        self.t0_epoch_s = time.time()
        self.t0_monotonic_s = time.monotonic()
        self._samples_fh = open(self.dir / "samples.csv", "w", newline="")
        self._samples = csv.writer(self._samples_fh)
        self._samples.writerow(SAMPLE_COLUMNS)
        self._samples_fh.flush()
        self._events_fh = open(self.dir / "events.csv", "w", newline="")
        self._events = csv.writer(self._events_fh)
        self._events.writerow(EVENT_COLUMNS)
        self._events_fh.flush()
        self.n_samples = 0
        self.n_events = 0
        self.closed = False
        #: first write error (disk full, vanished mount); set once, then the
        #: logger goes quiet instead of raising into a Qt slot every 100 ms
        self.failed: "str | None" = None
        self.meta: dict = dict(meta)
        self.meta.update({
            "schema": "disturbance_rig/1",
            "kind": kind,
            "start_epoch_s": self.t0_epoch_s,
            "t0_epoch_s": self.t0_epoch_s,
            "t0_monotonic_s": self.t0_monotonic_s,
            "stop_epoch_s": None,
            "hostname": socket.gethostname(),
            "files": {"samples": "samples.csv", "events": "events.csv",
                      "thruster": "thruster.csv"},
            "notes": {"thrust_is_commanded_not_measured": True,
                      "latched_column_is_authoritative_after_stop": True},
        })
        self.write_meta()

    # ------------------------------------------------------------- rows
    def _write_row(self, writer, fh, row, fname: str) -> None:
        if self.failed is not None:
            return
        try:
            writer.writerow(row)
            fh.flush()
        except (OSError, ValueError) as exc:          # ValueError: closed file
            self.failed = f"{fname}: {type(exc).__name__}: {exc}"
            self.meta.setdefault("write_errors", []).append(self.failed)

    def sample(self, values: dict) -> None:
        if self.closed:
            return
        self._write_row(self._samples, self._samples_fh,
                        [_fmt(c, values.get(c)) for c in SAMPLE_COLUMNS], "samples.csv")
        self.n_samples += 1

    def event(self, kind: str, detail: str = "") -> None:
        if self.closed:
            return
        self._write_row(self._events, self._events_fh,
                        [f"{time.time():.6f}", f"{time.monotonic():.6f}", kind, detail],
                        "events.csv")
        self.n_events += 1

    # ------------------------------------------------------------- meta
    def write_meta(self, extra: "dict | None" = None) -> bool:
        """Atomic meta.json rewrite; False (never raises) when the disk refuses."""
        if extra:
            self.meta.update(extra)
        try:
            tmp = self.dir / "meta.json.tmp"
            tmp.write_text(json.dumps(self.meta, indent=2, default=str))
            tmp.replace(self.dir / "meta.json")
            return True
        except OSError as exc:
            self.failed = self.failed or f"meta.json: {exc}"
            return False

    def close(self, thruster_rows=None, extra: "dict | None" = None) -> None:
        """Write thruster.csv (even if empty) and the final meta; idempotent."""
        if self.closed:
            return
        try:
            with open(self.dir / "thruster.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(THRUSTER_COLUMNS)
                for epoch_s, pct, pwm in (thruster_rows or []):
                    w.writerow([f"{epoch_s:.6f}", int(pct),
                                "" if pwm is None else int(pwm)])
        except OSError as exc:
            self.meta.setdefault("write_errors", []).append(f"thruster.csv: {exc}")
        self.meta["stop_epoch_s"] = time.time()
        self.meta["n_samples"] = self.n_samples
        self.meta["n_events"] = self.n_events
        if self.failed:
            self.meta["log_failed"] = self.failed
        try:
            self.write_meta(extra)
        finally:
            for fh in (self._samples_fh, self._events_fh):
                try:
                    fh.close()
                except OSError:
                    pass
            self.closed = True


def jet_calibration(whisker_root: "Path | None") -> "dict | None":
    """whisker's aim_calibration.json (jet heading vs motor angle), if present.

    Reference only: it is valid while the mount and the motor zero are those of
    the calibration (2026-09-08: thruster re-mounted reversed, useful jet =
    NEGATIVE throttle, jet faces +X at motor 90 deg). The panel shows it as a
    hint and copies it into meta.json; it never drives an interlock.
    """
    if whisker_root is None:
        return None
    path = Path(whisker_root) / "aim_calibration.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    data["file"] = str(path)
    return data


def provenance(whisker_root: "Path | None") -> dict:
    """git state of both checkouts — six subprocesses, so the panel takes it
    ONCE before any device connects, never from a slot while the jet runs."""
    return {"repo_git": git_head(REPO_ROOT),
            "whisker_git": git_head(whisker_root) if whisker_root else None,
            "probed_epoch_s": time.time()}


def base_meta(settings, options: PanelOptions, whisker_root: Path,
              preflight: "dict | None" = None, prov: "dict | None" = None) -> dict:
    """The meta.json skeleton the panel fills: provenance + configuration."""
    prov = prov or provenance(whisker_root)
    return {
        "jet_calibration": jet_calibration(whisker_root),
        "panel": {"file": str(Path(__file__).resolve().parent / "disturbance_panel.py"),
                  "repo_git": prov.get("repo_git")},
        "whisker_vision": {"dir": str(whisker_root), "git": prov.get("whisker_git")},
        "provenance_probed_epoch_s": prov.get("probed_epoch_s"),
        "settings": settings_summary(settings, options),
        "rig_notes": list(options.rig_notes),
        "simulator": bool(settings.simulator),
        "preflight": preflight,
        "interpreter": sys.executable,
    }


def whisker_dir_from_argv(argv: "list[str] | None" = None) -> "str | None":
    """Pre-scan for --whisker-dir: the panel must add whisker_vision to sys.path
    before its own imports, i.e. before argparse runs."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--whisker-dir", default=None)
    known, _ = pre.parse_known_args(sys.argv[1:] if argv is None else argv)
    return known.whisker_dir
