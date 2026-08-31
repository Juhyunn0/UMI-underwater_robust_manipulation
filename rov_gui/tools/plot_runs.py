#!/usr/bin/env python3
"""plot_runs.py — presentation-quality trajectory figures from the run folders.

    python -m rov_gui.tools.plot_runs                      # config/traj_plots.yaml
    python -m rov_gui.tools.plot_runs --dates 20260814          # the whole day
    python -m rov_gui.tools.plot_runs --dates 20260814_212021   # one run
    python -m rov_gui.tools.plot_runs --dates 20260814 --list

Give it DATES — that is the whole config. It finds every controller run
recorded under ``sessions/low_level_controller_data/<YYYYMMDD>/<run>/``
(rov_gui/runstore.py), draws one figure per run, and says plainly which entries
produced nothing and why. Everything else — the frame, the axis limits, the
colours — is fixed so that two figures from two different days can be put side
by side on a slide and compared by eye without reading the axes.

A date entry may name the day, the run folder, or the single CSV::

    20260814          every run of that day
    20260814_2120     the 2120 run folder
    20260814_212021   the one CSV stamped 212021
    all | latest

The figure itself carries NO date: it goes in the file name, where a caption
cannot contradict it —

    figures/trajectories/<date>/<date>_<run>_<controller>_<hhmmss>.png

and the CONTROLLER in that name comes from meta.json, not from the CSV's stem.
The station writes every controller's log as ``mpc_<hhmmss>.csv`` — the prefix
names the worker — so a PID run would otherwise ship as ``..._mpc_....png``.

What it reads per run
---------------------
    mpc_<hhmmss>.csv        the 20 Hz controller log (px,py / rx,ry / obj_px,
                            obj_py / traj_on / mode ...). A log opened next to
                            a feed recording is named after it instead:
                            c3_depth_<date>_<hhmmss>_mpc.csv
    mpc_<hhmmss>.meta.json  controller type, solver, trajectory spec, and the
                            ENGAGE DATUM in tag-map coordinates
    nav_<hhmmss>/map.json   (optional) the pool box the run itself knew — used
                            only to CHECK the box this tool draws

The frame, and why it is not the CSV's frame
--------------------------------------------
The CSV is written in the run's own datum frame: at every ENGAGE the current
pose becomes (0,0,0)/yaw 0 (workers.py ``_datumize``), and the columns are then
mirrored to world-FLU. Two runs from the same afternoon therefore share no
origin — plotting them raw puts two squares on top of each other that were
metres apart in the pool.

So every run is lifted back into the MAP frame (the tag map's anchor-tag frame,
NED-like, +z down) with the rigid transform its meta.json carries:

    x_ned = px,  y_ned = -py                       (FLU mirror, undone)
    x_map = x0 + cos(yaw0) x_ned - sin(yaw0) y_ned
    y_map = y0 + sin(yaw0) x_ned + cos(yaw0) y_ned      (workers.py _to_map_xy)

with (x0, y0, yaw0) = ``meta.hardware.datum_tag_frame``. A run recorded without
a datum cannot be placed in the pool and is REPORTED, not silently drawn
somewhere plausible.

Screen orientation follows the GUI plot and plot_nav_run.py: screen up = +x_map,
screen right = +y_map.

The axes are the pool, always
-----------------------------
Every figure — every run, every date — uses the same limits: the pool box,
derived from the tag map exactly the way the station derives it (outermost tag
EDGES + ``pool_margin_m``, config/hw_nav.yaml, geometry.py ``pool_from_tags``).
Nothing autoscales. If a run's own ``nav_*/map.json`` recorded a different box,
that is printed as a warning rather than quietly changing one figure's scale.

What is on the figure, and what is deliberately not
---------------------------------------------------
The vehicle, the reference, the tracked object when there is one, one error
number, and one key. Nothing else: operator request 2026-08-15, after the
first version carried a stats table and a provenance footer that competed with
the slide's own caption.

Colour carries two things at once. HUE says which track — blue is the vehicle,
orange the tracked object — and LIGHTNESS within each hue says when, light at
the start of the run and dark at the end. So the key is not a legend and not a
colour bar but both: one gradient row per track over a shared elapsed-time
axis. The reference gets no ramp, because it is a command rather than a
measurement and shading it would imply it drifted. ``time_color: false``
falls back to flat colours and an ordinary legend.

Two things make that readable rather than merely true (2026-08-23: the first
version's shades were too close to tell apart):

* the ramps are WIDE and EVENLY SPACED. Each walks its hue's analogous
  neighbours instead of holding one hue, and :func:`_cmap` re-parameterises it
  by OKLab arc length so a minute of run time is the same amount of colour
  change wherever it falls. See the note above THEMES for the measurements.
* ROUND-TIME DOTS on the vehicle path, labelled with their second count. A
  path that laps over itself buries its own early colours — the last lap draws
  on top of the first four — and no palette fixes that. An isolated ringed dot
  survives the overdraw, its label removes the need to judge colour at all,
  and the spacing between dots is speed for free. A mark with no sample near
  it (a released stretch, `segment: engaged`) is dropped rather than snapped
  to the nearest row. ``time_ticks: false`` turns them off; ``time_bands: N``
  quantises the ramp into N flat steps if discrete reads better than smooth.

The object track (``--pose``, control/object_nav.py)
----------------------------------------------------
``obj_px/py/pz`` are world FLU in the DATUM frame, the same convention as
``px/py/pz``, so the object rides the SAME lift into the map as the vehicle
and the two stay comparable (workers.py:341). It is drawn as points rather
than a line: the tracker re-seeds, and a line would draw a journey between two
sightings that never happened.

Three filters, each answering something the recording actually does — the
counts are printed per run, never applied silently:

    live_only         `obj_state` also takes stale / cold / lost, where the
                      position is the last good one held over, not a
                      measurement.
    pair_exact_only   the camera extrinsic cancels only when the object pose
                      and the tag fix came from the SAME camera frame; the
                      CSV calls `obj_pair_exact == 0` a record boundary.
    min_step_m        the log runs at 20 Hz and the tracker at ~9, so most
                      rows repeat the previous pose exactly.

Because the axes are the pool and the tracker can put the object outside it,
the count of samples that fall off the figure is printed too.

The one number is CROSS-TRACK RMS whenever the run used path following, because
the spatial follower deliberately places stage 0 up to ``path_lead_m`` ahead
of the vehicle's projection — radial |p - ref| then reads as failure on a
vehicle sitting exactly on the line (see MpcWorker._path_split). ``error:
radial`` overrides that; nothing on the figure is ever typed in by hand.

Provenance moved off the picture rather than away: the file name pins the day,
the run and the controller, and the console prints
``<date>/<run>/<csv stem>  →  <figure>`` for every figure written, so the CSV
behind any figure is still one lookup away (CLAUDE.md's citation rule).

matplotlib is fine HERE (offline analysis, its own process); the no-matplotlib
house rule applies to the live GUI only (widgets/indicators.py:5).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _rel(p: Path) -> str:
    """Repo-relative when it is inside the repo — paths on screen stay short
    and stay quotable (they end up in the figure footer)."""
    p = Path(p)
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)

# ---------------------------------------------------------------- defaults
#: Every knob, with the value used when the config file omits it. A config
#: file is merged ONTO this, key by key, so a two-line config is legal.
DEFAULTS = {
    # ---- what to plot: this is the only key the config file normally carries
    "dates": [],                 # "20260814" | "20260814_212021" | all | latest
    "base": "sessions/low_level_controller_data",
    "out_dir": "figures/trajectories",
    "controllers": [],           # ["mpc", "pid", "dobmpc"]; empty = all
    "segment": "auto",           # auto | traj | engaged | all
    "min_points": 20,            # fewer usable samples than this = skipped
    "frame": "map",              # map | datum
    "error": "auto",             # auto | cross | radial — the ONE number shown

    # ---- the tracked object (--pose; the CSV's obj_* columns, 2026-08-21+)
    "object": {
        "show": "auto",          # auto = draw it when the run has one
        "live_only": True,       # obj_state == live; drop stale/cold/lost
        "pair_exact_only": True, # see the note on Run.object_xy
        "min_step_m": 0.002,     # drop repeats: the log is 20 Hz, poses ~9 Hz
    },

    # ---- the box the axes are locked to
    "pool": {
        "source": "auto",        # auto | tag_map | map_json | config
        "x": None,               # [x0, x1] map frame, used when source: config
        "y": None,               # [y0, y1]
        "nav_yaml": "config/hw_nav.yaml",
        "margin_m": None,        # None = hw_nav.yaml pool_margin_m
        "pad_m": 0.05,           # breathing room outside the wall, drawing only
        "fallback": {"x": [-2.0, 2.0], "y": [-2.0, 2.0]},
    },

    # ---- how it looks
    "style": {
        "theme": "light",        # light | dark
        "actual_color": None,    # None = the theme's series blue
        "ref_color": None,       # None = the theme's ink (black on light)
        "actual_lw": 2.1,
        "ref_lw": 1.5,
        "ref_dashes": [5.0, 3.5],
        "ref_single_lap": True,  # draw one lap, so the dashes stay dashes
        "tick_m": 0.5,
        "panel_width_in": 11.0,
        "dpi": 300,
        "formats": ["png"],      # png | pdf | svg
        "time_color": True,      # shade both tracks light -> dark by elapsed t
        "time_bands": 0,         # >1 quantises the ramp into that many steps
        "time_ticks": True,      # dots at round times, so laps stay readable
        "time_tick_ms": 52,      # their marker area
        "time_tick_labels": True,   # ...with the second count beside them
        "object_ms": 22,         # object scatter marker area
        "show_markers": True,
        "show_tags": False,      # faint tag map behind the path
        "show_context": False,   # the approach/hold path, faint
        "stats_box": True,       # the one error number, in a corner
        "font": "DejaVu Sans",
    },
}

#: Chart surfaces and ink, from the validated reference palette (the dataviz
#: skill's `references/palette.md`): series slot 1 for the vehicle, primary ink
#: for the reference. "Reference in black" on light becomes white on dark —
#: it is the INK role that is fixed, not the hex.
#:
#: ``ramp`` / ``ramp2`` are the SEQUENTIAL pair — one hue family each, light to
#: dark, carrying elapsed time within a series while the hue keeps saying which
#: series it is (blue = vehicle, orange = object).
#:
#: They are wider than the palette's tabled blue scale, deliberately, and this
#: is the reason: a two-step ramp inside the ORDINAL window spans ~43 units of
#: OKLab and the operator could not read time off it (2026-08-23). Two changes
#: bought ~25-40% more separation without breaking any rule:
#:
#: * ANALOGOUS DRIFT. Each ramp walks its hue's neighbours rather than holding
#:   one hue — blue through cyan-blue to indigo, orange through amber to deep
#:   red. That is the documented multi-hue exception for sequential scales
#:   (neighbours only, and always with a scale legend, which the key is).
#: * EVEN SPACING. :func:`_cmap` resamples every ramp to equal OKLab
#:   arc length, so a minute of run time looks like the same amount of colour
#:   change wherever it falls. Before that the orange ramp spent its first half
#:   almost still (steps 5.7, 5.1, 13.1, 13.8) — the run's first two minutes
#:   were one colour.
#:
#: The pale end still has to be a visible LINE, not a receding heat-map cell,
#: so both are cut at ~2:1 against their surface — measured, not eyeballed:
#: light 2.07 / 2.08, dark 2.17 / 2.36 at the ends that face their surface.
#: Resulting quartile steps ~14 units, end to end ~53 / 49.
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
              "ink3": "#8a8985", "grid": "#e6e6e3", "wall": "#9a9a95",
              "series": "#2a78d6", "box_fc": "#ffffff", "box_ec": "#dedddb",
              "series2": "#eb6834",
              "ramp": ["#6bbdd8", "#2f8fd8", "#2a5ac2", "#232e88", "#111545"],
              "ramp2": ["#e3a64e", "#e46b2c", "#bd3a23", "#851e17", "#4d100f"]},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
             "ink3": "#8a8980", "grid": "#2e2e2c", "wall": "#6f6f68",
             "series": "#3987e5", "box_fc": "#242422", "box_ec": "#3a3a37",
             "series2": "#e0763f",
             "ramp": ["#d6f2f7", "#7dc8eb", "#4693e6", "#3f5ecb", "#3c44ad"],
             "ramp2": ["#fbe3b8", "#f5af60", "#eb7439", "#d3452b", "#a02a1e"]},
}

#: The five controller modes (mpc_bridge.MODES + pid). ``*_tuned`` is the same
#: compiled solver with the position weights rotated into the path frame, so
#: it is named as a variant of its base rather than as a separate controller.
CTRL_LABEL = {"pid": "PID", "mpc": "MPC", "dobmpc": "DOB-MPC",
              "mpc_tuned": "MPC path-frame",
              "dobmpc_tuned": "DOB-MPC path-frame"}


# ============================================================== config
def deep_merge(base: dict, over: dict) -> dict:
    """``over`` wins, one key at a time, recursing into dicts only."""
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path | None) -> dict:
    if path is None:
        return dict(DEFAULTS)
    if not path.exists():
        raise SystemExit(f"plot_runs: config not found: {path}")
    import yaml
    user = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(user, dict):
        raise SystemExit(f"plot_runs: {path} must be a mapping at the top level")
    unknown = sorted(set(user) - set(DEFAULTS))
    if unknown:
        print(f"[warn] {path.name}: ignoring unknown key(s) {', '.join(unknown)}")
    return deep_merge(DEFAULTS, user)


#: A config entry: the day, and optionally which run of it.
#:
#:   20260814          the whole day
#:   20260814_2120     the 2120 run folder
#:   20260814_212021   the one CSV stamped 212021, and nothing else
#:
#: The tail is matched as a SUBSTRING against both the run folder's name and
#: the CSV's, because the two carry different resolutions of the same instant:
#: the folder is ``2120`` (or ``0814_212021`` since the rename) while the CSV
#: beside it is ``mpc_212021``. One rule reaches both.
_SPEC = re.compile(r"^(\d{8})(?:[_\-](\S+))?$")


def resolve_dates(cfg: dict, base: Path) -> tuple[list[tuple[str, str]], list[str]]:
    """([(date, run token or "")], entries whose DAY does not exist)."""
    want = cfg["dates"]
    if isinstance(want, (str, int)):
        want = [want]
    want = [str(d).strip() for d in (want or []) if str(d).strip()]
    have = sorted(d.name for d in base.iterdir()
                  if d.is_dir() and d.name.isdigit()) if base.is_dir() else []
    if not want:
        raise SystemExit(
            "plot_runs: no dates. Put them in the config (dates: [20260814]) "
            "or pass --dates 20260814"
            + (f"\n  available: {', '.join(have) or '(none)'}" if have else ""))
    if any(d.lower() == "all" for d in want):
        return [(d, "") for d in have], []
    if any(d.lower() == "latest" for d in want):
        return ([(have[-1], "")], []) if have else ([], ["latest"])
    specs, missing = [], []
    for entry in want:
        m = _SPEC.match(entry)
        if m is None:
            missing.append(f"{entry} (not a YYYYMMDD[_HHMMSS] entry)")
        elif m.group(1) not in have:
            missing.append(entry)
        else:
            specs.append((m.group(1), m.group(2) or ""))
    return specs, missing


# ============================================================== the pool box
def _tag_map_extent(tag_map_path: Path) -> tuple[np.ndarray, int]:
    import yaml
    m = yaml.safe_load(tag_map_path.read_text(encoding="utf-8")) or {}
    pts = []
    for v in (m.get("tags") or {}).values():
        for e in (v if isinstance(v, list) else [v]):
            p = e.get("position_m")
            if p is not None:
                pts.append([float(p[0]), float(p[1])])
    return np.asarray(pts, float), len(pts)


def resolve_pool(cfg: dict, run_dirs: list[Path]) -> tuple[dict, str]:
    """The one box every figure is scaled to, plus a one-line provenance.

    ``auto`` derives it from the tag map named in hw_nav.yaml, which is what
    the station itself draws (geometry.py ``pool_from_tags``): the outermost
    tag EDGES plus ``pool_margin_m`` on every side. Config beats that, a
    recorded map.json is the fallback, and a plain box is the last resort.
    """
    pc = cfg["pool"]
    src = str(pc.get("source", "auto")).lower()

    if src in ("config", "auto") and pc.get("x") and pc.get("y"):
        box = {"x": [float(v) for v in pc["x"]], "y": [float(v) for v in pc["y"]]}
        return box, "config/traj_plots.yaml pool.x / pool.y"

    if src in ("auto", "tag_map"):
        try:
            import yaml
            navp = REPO / str(pc["nav_yaml"])
            nav = yaml.safe_load(navp.read_text(encoding="utf-8")) or {}
            margin = pc.get("margin_m")
            margin = float(nav.get("pool_margin_m") if margin is None else margin)
            size = float(nav.get("tag_size_m", 0.17))
            tm = REPO / str(nav.get("tag_map", "config/tag_map_full.yaml"))
            P, n = _tag_map_extent(tm)
            if n:
                pad = size / 2.0 + margin
                box = {"x": [float(P[:, 0].min() - pad), float(P[:, 0].max() + pad)],
                       "y": [float(P[:, 1].min() - pad), float(P[:, 1].max() + pad)]}
                return box, (f"{_rel(tm)} ({n} tags) + tag_size/2 + "
                             f"pool_margin_m {margin:g} m [유도: geometry.py "
                             f"pool_from_tags]")
        except (OSError, ValueError, KeyError) as e:            # noqa: BLE001
            print(f"[warn] pool: tag-map derivation failed ({e})")

    if src in ("auto", "map_json"):
        for d in run_dirs:
            for mj in sorted(d.glob("nav_*/map.json")):
                try:
                    pn = (json.loads(mj.read_text(encoding="utf-8")) or {}).get("pool_ned")
                except (OSError, ValueError):
                    continue
                if pn and pn.get("x") and pn.get("y"):
                    return ({"x": [float(v) for v in pn["x"]],
                             "y": [float(v) for v in pn["y"]]},
                            f"{_rel(mj)} pool_ned (as the run recorded it)")

    fb = pc["fallback"]
    return ({"x": [float(v) for v in fb["x"]], "y": [float(v) for v in fb["y"]]},
            "fallback box — NO tag map and NO recorded pool_ned was readable")


def outside_box(box: dict, x, y) -> int:
    """How many samples the fixed axes cannot show.

    The pool box is the axis range by design, so anything beyond it is
    silently clipped by matplotlib. A tracked object CAN land outside — a
    re-seed onto the wrong thing puts it metres away — and a figure that just
    drops those points without a word is a figure that lies by omission.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size == 0:
        return 0
    inb = ((x >= box["x"][0]) & (x <= box["x"][1])
           & (y >= box["y"][0]) & (y <= box["y"][1]))
    return int((~inb).sum())


def check_pool(box: dict, run: "Run", tol: float = 0.01) -> str | None:
    """The run's own recorded pool box, when it disagrees with the drawn one."""
    for mj in sorted(run.run_dir.glob("nav_*/map.json")):
        try:
            pn = (json.loads(mj.read_text(encoding="utf-8")) or {}).get("pool_ned")
        except (OSError, ValueError):
            continue
        if not pn:
            continue
        d = max(abs(float(a) - float(b))
                for k in ("x", "y") for a, b in zip(pn[k], box[k]))
        if d > tol:
            return (f"{run.name}: {mj.parent.name}/map.json recorded a pool box "
                    f"{d * 100:.0f} cm from the one drawn — the map changed?")
    return None


# ============================================================== loading a run
@dataclass
class Run:
    """One controller CSV, lifted into the map frame and ready to draw."""
    date: str
    run_dir: Path
    csv_path: Path
    meta: dict
    x: np.ndarray = field(default_factory=lambda: np.empty(0))   # map frame [m]
    y: np.ndarray = field(default_factory=lambda: np.empty(0))
    rx: np.ndarray = field(default_factory=lambda: np.empty(0))
    ry: np.ndarray = field(default_factory=lambda: np.empty(0))
    t: np.ndarray = field(default_factory=lambda: np.empty(0))
    lap: np.ndarray = field(default_factory=lambda: np.empty(0))
    cx: np.ndarray = field(default_factory=lambda: np.empty(0))  # context path
    cy: np.ndarray = field(default_factory=lambda: np.empty(0))
    ox: np.ndarray = field(default_factory=lambda: np.empty(0))  # object, map
    oy: np.ndarray = field(default_factory=lambda: np.empty(0))
    ot: np.ndarray = field(default_factory=lambda: np.empty(0))  # its own clock
    modes: tuple = ()            # the controller(s) the plotted rows ran under
    stats: dict = field(default_factory=dict)

    def ref_xy(self, single_lap: bool = True):
        """The reference polyline to DRAW.

        A 5-lap square traces the same four sides five times, and five dashed
        copies at five different phases fill each other's gaps in — the line
        renders solid and the reader loses the one cue that says "this is the
        command, not the vehicle". So the drawn reference is the FIRST lap
        only. The error statistics are still computed against every sample.
        """
        if single_lap and self.lap.size == self.rx.size and self.lap.size:
            m = self.lap == self.lap[0]
            if 2 <= int(m.sum()) < self.rx.size:
                return self.rx[m], self.ry[m]
        return self.rx, self.ry

    @property
    def name(self) -> str:
        return f"{self.date}/{self.run_dir.name}/{self.csv_path.stem}"

    @property
    def ctrl(self) -> str:
        return str((self.meta.get("controller") or {}).get("type", "?"))

    @property
    def ctrl_label(self) -> str:
        """What the title says.

        ``meta.controller.type`` is the controller the run ENDED on, and the
        panel can be switched mid-run — 2026-08-23 has runs that went
        dobmpc -> pid -> mpc_tuned inside one CSV. The rows know: the `mode`
        column is per-tick. When the plotted rows disagree with each other the
        title lists them in the order they were flown, because "PID" over a
        picture two thirds of which is not PID is simply wrong.
        """
        if len(self.modes) > 1:
            return "  →  ".join(CTRL_LABEL.get(m, m.upper()) for m in self.modes)
        c = self.modes[0] if self.modes else self.ctrl
        lab = CTRL_LABEL.get(c, c.upper())
        solver = (self.meta.get("controller") or {}).get("solver")
        return f"{lab} · {solver}" if solver and c != "pid" else lab

    @property
    def has_object(self) -> bool:
        return self.ox.size > 1

    @property
    def stamp(self) -> str:
        """The HHMMSS in the CSV's name.

        Two namings reach here. A standalone controller log is
        ``mpc_<hhmmss>``; one opened alongside a feed recording inherits that
        recording's stem, ``c3_depth_<YYYYMMDD>_<hhmmss>_mpc``. So take the
        LAST underscore-delimited field that is exactly six digits — which
        skips the eight-digit date and does not care where in the name it sits.
        """
        six = [f for f in self.csv_path.stem.split("_")
               if len(f) == 6 and f.isdigit()]
        return six[-1] if six else self.csv_path.stem

    @property
    def slug(self) -> str:
        """The figure's file name.

        The station writes EVERY controller's log as ``mpc_<hhmmss>.csv`` —
        the prefix names the worker, not the controller — so a PID run's CSV
        is ``mpc_212021.csv``. Naming the figure after the stem would ship a
        PID result as ``..._mpc_....png``. The controller comes out of
        meta.json and goes in the name instead.
        """
        d = self.run_dir.name
        # The stamp is dropped only when it would repeat the folder AND there
        # is nothing else in that folder to confuse it with — a folder holding
        # three CSVs needs all three names to differ.
        alone = len(list(self.run_dir.glob("*.meta.json"))) == 1
        tail = "" if (alone and self.stamp in d) else f"_{self.stamp}"
        return f"{self.date}_{d}_{self.ctrl}{tail}"

    def headline(self, mode: str = "auto") -> tuple[str, float]:
        """(label, cm) — the ONE error number on the figure.

        ``auto`` picks cross-track for a path-following run: stage 0 is placed
        up to ``path_lead_m`` ahead of the vehicle projection on purpose, so
        radial |p - ref| is dominated by a SETTING and reads as failure on a
        vehicle sitting exactly on the line
        (MpcWorker._path_split). Set ``error: radial`` to override.
        """
        if mode == "auto":
            # A STATION hold has no path — "cross-track" against a fixed point
            # is a direction with no meaning, whatever path_following says.
            mode = ("cross" if (self.stats["path_following"]
                                and self.stats.get("shape") != "station")
                    else "radial")
        if mode == "cross" and np.isfinite(self.stats["cross_rms_cm"]):
            return "cross-track RMS", self.stats["cross_rms_cm"]
        return "radial RMS", self.stats["radial_rms_cm"]


#: Columns this tool needs. Read by NAME — runs before 2026-08-14 carry an
#: extra `geofence_ok` column and no `e_along` / `e_cross` (workers.py:290),
#: and the `dr_*` dead-reckoning columns exist only from 2026-08-17. Missing
#: names are simply absent from the returned dict, so old runs still load.
_NUM = ("t", "px", "py", "pz", "rx", "ry", "rz", "yaw_deg", "ryaw_deg",
        "t_traj", "engaged", "traj_on", "lap", "e_along", "e_cross",
        "dr_px", "dr_py", "dr_pz", "dr_pz_imu", "dr_err_m", "dr_err_z_m",
        "dr_t_s", "dr_hz", "dr_ok",
        "roll_deg", "pitch_deg", "tag_age_s",
        "obj_px", "obj_py", "obj_pz", "obj_age_s", "obj_pair_exact")

#: The columns that are words, not numbers.
_TXT = ("mode", "obj_state", "follow_state", "bridge_tier")


def read_csv(path: Path) -> dict[str, np.ndarray]:
    """Every wanted column as an array, float for `_NUM` and str for `_TXT`."""
    cols: dict[str, list] = {}
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        names = [n for n in (rd.fieldnames or []) if n in _NUM or n in _TXT]
        for n in names:
            cols[n] = []
        for row in rd:
            for n in names:
                v = row.get(n)
                if n in _TXT:
                    cols[n].append("" if v is None else str(v))
                    continue
                try:
                    cols[n].append(float(v))
                except (TypeError, ValueError):
                    cols[n].append(float("nan"))
    return {k: (np.asarray(v, dtype=object).astype(str) if k in _TXT
                else np.asarray(v, float)) for k, v in cols.items()}


def _datum(meta: dict):
    d = ((meta.get("hardware") or {}).get("datum_tag_frame")) or None
    if not d or d.get("p0") is None or d.get("yaw0_deg") is None:
        return None
    p0 = [float(v) for v in d["p0"]]
    return p0[0], p0[1], math.radians(float(d["yaw0_deg"]))


def _to_map(px, py, datum):
    """world-FLU datum frame -> MAP (tag-world) NED xy. Mirrors workers.py."""
    xn, yn = np.asarray(px, float), -np.asarray(py, float)
    if datum is None:
        return xn, yn
    x0, y0, yaw0 = datum
    c, s = math.cos(yaw0), math.sin(yaw0)
    return x0 + c * xn - s * yn, y0 + s * xn + c * yn


def _along_cross(P: np.ndarray, R: np.ndarray):
    """(along-lag, cross) from the reference path's own tangent — the fallback
    for runs older than the CSV's e_along / e_cross columns. Same conventions
    as MpcWorker._path_split: along + = behind the virtual target, cross + =
    left of the direction of travel."""
    dR = np.gradient(R, axis=0)
    n = np.hypot(dR[:, 0], dR[:, 1])
    ok = n > 1e-9
    u = np.zeros_like(dR)
    u[ok] = dR[ok] / n[ok, None]
    e = P - R
    along = -(u[:, 0] * e[:, 0] + u[:, 1] * e[:, 1])
    cross = u[:, 1] * e[:, 0] - u[:, 0] * e[:, 1]
    along[~ok] = np.nan
    cross[~ok] = np.nan
    return along, cross


def _finite(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v[np.isfinite(v)]


def _rms(v) -> float:
    v = _finite(v)
    return float(np.sqrt(np.mean(v ** 2))) if v.size else float("nan")


def _peak(v) -> float:
    """max |v|, NaN when nothing is finite — a station hold has no path
    tangent, so its cross-track column is all NaN and np.nanmax would warn."""
    v = _finite(v)
    return float(np.max(np.abs(v))) if v.size else float("nan")


def _mean(v) -> float:
    v = _finite(v)
    return float(np.mean(v)) if v.size else float("nan")


def _object_track(col: dict, m: np.ndarray, datum, t_rel: np.ndarray,
                  cfg: dict):
    """(x, y, t, note) of the tracked object, in the map frame.

    `obj_px/py/pz` are world FLU in the DATUM frame — the same convention as
    `px/py/pz` — so the object rides the SAME lift into the map as the vehicle
    and the two stay comparable to the millimetre (workers.py:341).

    Three filters, each because of something the recording actually does:

    `live_only`   `obj_state` also takes stale / cold / lost, and on those the
                  position is the last good one held over, not a measurement.
                  Drawing them paints a dense blob wherever the tracker was
                  when it lost the object.
    `pair_exact_only`  the camera extrinsic cancels out of the object
                  composition ONLY when the object pose and the tag fix came
                  from the same camera frame. A row with `obj_pair_exact == 0`
                  was composed across frames and carries up to
                  |t_frd_cam| * 2 sin(dyaw/2) of extra position error from the
                  0.2855 m lever arm — the CSV calls it a record boundary and
                  so is this.
    `min_step_m`  the log runs at 20 Hz and the tracker at ~9, so most rows
                  repeat the previous pose exactly. Keeping them triples the
                  ink for no information and darkens the time ramp unevenly.
    """
    oc = cfg["object"]
    if str(oc.get("show", "auto")).lower() in ("off", "false", "no"):
        return np.empty(0), np.empty(0), np.empty(0), ""
    if "obj_px" not in col or "obj_py" not in col:
        return np.empty(0), np.empty(0), np.empty(0), ""

    keep = m & np.isfinite(col["obj_px"]) & np.isfinite(col["obj_py"])
    n_any = int(keep.sum())
    if not n_any:
        return np.empty(0), np.empty(0), np.empty(0), ""
    if oc.get("live_only", True) and "obj_state" in col:
        keep &= col["obj_state"] == "live"
    n_live = int(keep.sum())
    if oc.get("pair_exact_only", True) and "obj_pair_exact" in col:
        keep &= col["obj_pair_exact"] > 0.5
    n_exact = int(keep.sum())
    if n_live < 2:
        # Not a filter outcome: the tracker held `lost`/`cold` the whole time,
        # so this run has an object column and no object.
        states = sorted(set(col["obj_state"][m & np.isfinite(col["obj_px"])])) \
            if "obj_state" in col else []
        return (np.empty(0), np.empty(0), np.empty(0),
                f"object: never acquired — obj_state was "
                f"{'/'.join(states) or 'never live'} on all {n_any} row(s)")
    if n_exact < 2:
        return (np.empty(0), np.empty(0), np.empty(0),
                f"object: {n_live} live sample(s), none with an exact "
                f"camera-frame pair (object.pair_exact_only)")

    x, y = _to_map(col["obj_px"][keep], col["obj_py"][keep], datum)
    t = t_rel[keep[m]]
    step = float(oc.get("min_step_m", 0.0) or 0.0)
    if step > 0 and x.size > 1:
        d = np.hypot(np.diff(x), np.diff(y))
        fresh = np.concatenate(([True], d > step))
        x, y, t = x[fresh], y[fresh], t[fresh]
    note = (f"object: {x.size} point(s) drawn of {n_any} logged "
            f"({n_any - n_live} not live, {n_live - n_exact} loose pairs, "
            f"{n_exact - x.size} repeats)")
    return x, y, t, note


def load_run(date: str, run_dir: Path, meta_path: Path, cfg: dict):
    """(Run, None) or (None, "why this one produced no figure")."""
    csv_path = meta_path.with_suffix("").with_suffix(".csv")
    if not csv_path.exists():
        csv_path = meta_path.parent / (meta_path.name[:-len(".meta.json")] + ".csv")
    label = f"{date}/{run_dir.name}/{csv_path.stem}"
    if not csv_path.exists():
        return None, f"{label}: meta.json without a CSV beside it"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:                          # noqa: BLE001
        return None, f"{label}: meta.json unreadable ({e})"

    want = [c.lower() for c in (cfg["controllers"] or [])]
    ctrl = str((meta.get("controller") or {}).get("type", "?")).lower()
    if want and ctrl not in want:
        return None, None                          # filtered out, not a failure

    try:
        col = read_csv(csv_path)
    except OSError as e:
        return None, f"{label}: CSV unreadable ({e})"
    if not col or col.get("t") is None or col["t"].size == 0:
        return None, f"{label}: CSV has a header and no rows (nothing was flown)"

    datum = _datum(meta)
    if cfg["frame"] == "map" and datum is None:
        return None, (f"{label}: no hardware.datum_tag_frame in meta.json — the "
                      f"run cannot be placed in the pool (use frame: datum to "
                      f"plot it in its own frame)")
    if cfg["frame"] != "map":
        datum = None

    seg = str(cfg["segment"]).lower()
    n = col["t"].size
    on_traj = col.get("traj_on", np.zeros(n)) > 0.5
    on_eng = col.get("engaged", np.zeros(n)) > 0.5
    if seg == "auto":
        # A square or a line has a trajectory window and that window IS the
        # result. A station hold or an object follow never sets traj_on at
        # all (meta run.traj_on False, 2026-08-23), so asking for the
        # trajectory there means asking for nothing — fall through to the
        # engagement, and to the whole log for a run that only ever watched.
        if on_traj.sum() >= int(cfg["min_points"]):
            m, seg_note = on_traj, "trajectory"
        elif on_eng.sum() >= int(cfg["min_points"]):
            m, seg_note = on_eng, "engaged"
        else:
            m, seg_note = np.ones(n, bool), "whole log"
    elif seg == "traj":
        m, seg_note = on_traj, "trajectory"
    elif seg == "engaged":
        m, seg_note = on_eng, "engaged"
    else:
        m, seg_note = np.ones(n, bool), "whole log"
    m &= np.isfinite(col["px"]) & np.isfinite(col["py"])
    m &= np.isfinite(col.get("rx", np.full(n, np.nan)))
    m &= np.isfinite(col.get("ry", np.full(n, np.nan)))
    if int(m.sum()) < int(cfg["min_points"]):
        return None, (f"{label}: {int(m.sum())} usable {seg_note} samples "
                      f"(< min_points {cfg['min_points']}) — {n} rows logged, "
                      f"{int(on_eng.sum())} engaged, {int(on_traj.sum())} with "
                      f"the trajectory running")

    run = Run(date=date, run_dir=run_dir, csv_path=csv_path, meta=meta)
    run.x, run.y = _to_map(col["px"][m], col["py"][m], datum)
    run.rx, run.ry = _to_map(col["rx"][m], col["ry"][m], datum)
    run.t = col["t"][m] - float(col["t"][m][0])
    run.lap = col["lap"][m] if "lap" in col else np.empty(0)
    if "mode" in col:
        seen_modes, order = set(), []
        for v in col["mode"][m]:
            if v and v not in seen_modes:
                seen_modes.add(v)
                order.append(v)
        run.modes = tuple(order)
    run.ox, run.oy, run.ot, obj_note = _object_track(col, m, datum, run.t, cfg)
    if obj_note:
        run.stats["obj_note"] = obj_note
    if cfg["style"]["show_context"]:
        c = np.isfinite(col["px"]) & np.isfinite(col["py"]) & ~m
        run.cx, run.cy = _to_map(col["px"][c], col["py"][c], datum)

    P = np.column_stack([run.x, run.y])
    R = np.column_stack([run.rx, run.ry])
    radial = np.hypot(*(P - R).T)
    if "e_cross" in col and np.isfinite(col["e_cross"][m]).sum() >= 3:
        along, cross = col["e_along"][m], col["e_cross"][m]
        err_src = "CSV e_along / e_cross"
    else:
        along, cross = _along_cross(P, R)
        err_src = "reconstructed from the reference tangent"
    dz = (col["pz"][m] - col["rz"][m]) if "rz" in col else np.full(m.sum(), np.nan)
    dyaw = ((col["yaw_deg"][m] - col["ryaw_deg"][m] + 180.0) % 360.0 - 180.0) \
        if "ryaw_deg" in col else np.full(int(m.sum()), np.nan)
    lead = float(((meta.get("reference_clock") or {}).get("path_lead_m") or 0.0))
    obj_note = run.stats.get("obj_note", "")
    run.stats = {
        "n": int(m.sum()), "dur_s": float(run.t[-1]), "seg": seg_note,
        "obj_n": int(run.ox.size), "obj_note": obj_note,
        "cross_rms_cm": _rms(cross) * 100.0,
        "cross_max_cm": _peak(cross) * 100.0,
        "along_mean_cm": _mean(along) * 100.0,
        "radial_rms_cm": _rms(radial) * 100.0,
        "z_rms_cm": _rms(dz) * 100.0, "yaw_rms_deg": _rms(dyaw),
        "laps": int(_peak(col["lap"][m])) + 1 if "lap" in col else 0,
        "shape": _shape(meta),
        "path_following": bool((meta.get("reference_clock") or {})
                               .get("path_following", False)),
        "lead_cm": lead * 100.0, "err_src": err_src,
    }
    return run, None


def discover(base: Path, specs: list[tuple[str, str]], cfg: dict):
    """(runs, skip messages) for every config entry, chronological.

    An entry that names one run (``20260814_212021``) and finds nothing must
    say so by its own name — silently plotting the rest of the day would look
    like success.
    """
    runs, skips, seen = [], [], set()
    for date, token in specs:
        day = base / date
        entry = f"{date}_{token}" if token else date
        found_any = False
        for run_dir in sorted(d for d in day.iterdir() if d.is_dir()):
            metas = sorted(run_dir.glob("*.meta.json"))
            if token:
                metas = [m for m in metas
                         if token in run_dir.name or token in m.name]
            if not metas:
                continue
            found_any = True
            for mp in metas:
                if mp in seen:
                    continue                     # two entries, the same run
                seen.add(mp)
                run, why = load_run(date, run_dir, mp, cfg)
                if run is not None:
                    runs.append(run)
                elif why:
                    skips.append(why)
        if not found_any:
            skips.append(
                f"{entry}: nothing matches"
                + (" — no run folder or CSV carries that stamp"
                   if token else
                   " — no run folder holds a controller CSV (only station "
                   "launches / screen recordings)"))
    return runs, skips


# ============================================================== drawing
def _mpl(show: bool):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _free_corner(ax_box: dict, x, y, taken=()):
    """Which corner of the pool the data is farthest from — so the stats box
    and the legend land on empty water instead of on the path."""
    cx = np.asarray(x, float).mean()
    cy = np.asarray(y, float).mean()
    out = []
    for name, (px, py) in {
            "upper left": (ax_box["x"][1], ax_box["y"][0]),
            "upper right": (ax_box["x"][1], ax_box["y"][1]),
            "lower left": (ax_box["x"][0], ax_box["y"][0]),
            "lower right": (ax_box["x"][0], ax_box["y"][1])}.items():
        out.append((math.hypot(px - cx, py - cy), name))
    out.sort(reverse=True)
    return [n for _d, n in out if n not in taken]


def _shape(meta: dict) -> str:
    """What the run was flying: square / line / circle / follow / station."""
    tr = meta.get("trajectory") or {}
    if tr.get("kind"):
        return str(tr["kind"])
    mission = meta.get("mission") or {}
    return str((mission.get("panel_override") or {}).get("shape")
               or (mission.get("config_square") or {}).get("shape")
               or "station")


def _traj_line(meta: dict) -> str:
    """The subtitle: what the run was asked to do.

    ``trajectory`` is filled in for a square / line / circle. A station hold
    or an object follow leaves it null and says what it was in ``mission``
    instead (schema 6) — reading only ``trajectory`` labels every follow run
    "station hold (no trajectory)", which is the one thing it was not.
    """
    tr = meta.get("trajectory") or {}
    kind = tr.get("kind")
    if not kind:
        mission = meta.get("mission") or {}
        panel = mission.get("panel_override") or {}
        conf = mission.get("config_square") or {}
        shape = panel.get("shape") or conf.get("shape")
        if not shape:
            return "station hold (no trajectory)"
        bits = [str(shape).upper()]
        speed = panel.get("speed", conf.get("speed"))
        if shape == "follow":
            obj = meta.get("object_nav") or {}
            follow = obj.get("follow") or {}
            bits.append("tracked object")
            if speed is not None:
                bits.append(f"{float(speed):.2f} m/s")
            if follow.get("ended"):
                bits.append(f"ended: {follow['ended']}")
        elif speed is not None:
            bits.append(f"{float(speed):.2f} m/s")
        return "  ·  ".join(bits)
    bits = [str(kind).upper()]
    if tr.get("size") is not None:
        sy = tr.get("size_y", tr.get("size"))
        bits.append(f"{float(tr['size']):.2f} × {float(sy):.2f} m")
    if tr.get("radius") is not None:
        bits.append(f"r {float(tr['radius']):.2f} m")
    if tr.get("length") is not None:
        bits.append(f"{float(tr['length']):.2f} m")
    if tr.get("speed") is not None:
        bits.append(f"{float(tr['speed']):.2f} m/s")
    if tr.get("laps"):
        bits.append(f"{int(tr['laps'])} laps")
    if tr.get("origin_tag") is not None:
        bits.append(f"origin tag {int(tr['origin_tag'])}")
    return "  ·  ".join(bits)


def _tag_patches(cfg: dict):
    """(x, y) of every tag in the map, for the faint context layer."""
    try:
        import yaml
        nav = yaml.safe_load(
            (REPO / cfg["pool"]["nav_yaml"]).read_text(encoding="utf-8")) or {}
        P, n = _tag_map_extent(REPO / str(nav.get("tag_map",
                                                  "config/tag_map_full.yaml")))
        return (P, float(nav.get("tag_size_m", 0.17))) if n else (None, 0.0)
    except (OSError, ValueError, KeyError):
        return None, 0.0


def _setup_axes(ax, box, pad, th, st, cfg, tags=None):
    """The pool, the grid, the ticks — identical on every panel by construction."""
    from matplotlib import patches, ticker
    ax.set_facecolor(th["surface"])
    ax.set_xlim(box["y"][0] - pad, box["y"][1] + pad)       # screen right = +y
    ax.set_ylim(box["x"][0] - pad, box["x"][1] + pad)       # screen up    = +x
    ax.set_aspect("equal", "box")
    step = float(st["tick_m"])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(step))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(step))
    ax.grid(True, color=th["grid"], lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)                # the pool wall IS the frame
    ax.tick_params(length=0, colors=th["ink3"], labelsize=9)
    ax.set_xlabel("y  [m]", fontsize=10.5, color=th["ink2"], labelpad=5)
    ax.set_ylabel("x  [m]", fontsize=10.5, color=th["ink2"], labelpad=3)
    if tags is not None and st["show_tags"]:
        P, size = tags
        if P is not None:
            for tx, ty in P:
                ax.add_patch(patches.Rectangle(
                    (ty - size / 2, tx - size / 2), size, size,
                    facecolor=th["grid"], edgecolor="none", zorder=1))
    ax.add_patch(patches.Rectangle(
        (box["y"][0], box["x"][0]),
        box["y"][1] - box["y"][0], box["x"][1] - box["x"][0],
        fill=False, edgecolor=th["wall"], lw=1.6, zorder=2))


def _norm(t0, t1):
    from matplotlib.colors import Normalize
    return Normalize(vmin=t0, vmax=t1)


def _oklab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0-1, Nx3) -> OKLab. Small enough to carry; the point of having it
    is that ramp spacing is MEASURED rather than judged by eye."""
    c = np.asarray(rgb, float)[:, :3]
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = lin[:, 0], lin[:, 1], lin[:, 2]
    l = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return np.stack([0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
                     1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
                     0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s],
                    axis=-1)


_CMAP_CACHE: dict = {}


def _cmap(colors, name, bands: int = 0):
    """The theme's ramp, resampled so equal time reads as equal colour change.

    Interpolating hex anchors in sRGB bunches the change where the anchors sit:
    the orange ramp used to spend the first half of every run almost still. So
    the LUT is re-parameterised by OKLab arc length — constant perceptual speed
    from end to end — which is what makes "how far along is this" answerable by
    eye at all.

    ``bands`` > 0 quantises the result into that many flat steps. A banded ramp
    is easier to READ (this is the third band, not "somewhere past the middle")
    at the cost of pretending time is discrete, so it is opt-in.
    """
    from matplotlib.colors import LinearSegmentedColormap
    key = (tuple(colors), int(bands))
    if key in _CMAP_CACHE:
        return _CMAP_CACHE[key]
    raw = LinearSegmentedColormap.from_list(name, list(colors), N=256)
    rgb = raw(np.linspace(0.0, 1.0, 256))[:, :3]
    arc = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(_oklab(rgb), axis=0), axis=1))])
    if arc[-1] <= 0:
        return raw
    idx = np.interp(np.linspace(0.0, arc[-1], 256), arc, np.arange(256.0))
    even = rgb[np.round(idx).astype(int)]
    if bands and bands > 1:
        # one flat colour per band, sampled at the band's midpoint
        edges = np.linspace(0, 256, int(bands) + 1).astype(int)
        for a, b in zip(edges[:-1], edges[1:]):
            even[a:b] = even[(a + b) // 2]
    cm = LinearSegmentedColormap.from_list(f"{name}_even", even, N=256)
    _CMAP_CACHE[key] = cm
    return cm


#: The ladder a tick interval is snapped to, so the labels under the key read
#: 60 / 120 / 300 rather than 63.5.
_TICK_LADDER = (1, 2, 5, 10, 15, 20, 30, 60, 90, 120, 300, 600, 900)


def _tick_times(t: np.ndarray, target: int = 10) -> np.ndarray:
    """Round elapsed times to mark on the path, ~``target`` of them.

    Colour alone cannot survive a path that laps over itself: the last lap
    draws on top of the first four, so most of a 5-lap square ends up wearing
    the ramp's dark end whatever the palette does. Isolated marks do survive —
    a ringed dot is legible over any amount of overdraw — and they carry a
    second reading for free, because their SPACING is speed.
    """
    if t.size < 2:
        return np.empty(0)
    span = float(t[-1] - t[0])
    if span <= 0:
        return np.empty(0)
    raw = span / max(1, target)
    step = next((v for v in _TICK_LADDER if v >= raw), _TICK_LADDER[-1])
    marks = np.arange(step, span + 1e-9, step) + float(t[0])
    return marks[marks <= t[-1]]


def _time_span(run) -> tuple[float, float]:
    """The one clock both tracks are shaded against.

    The vehicle and the object are sampled on the SAME CSV rows, so the object
    can never outlast the run; taking the span from the vehicle alone keeps
    the two ramps aligned even when the tracker only saw part of the run.
    """
    t1 = float(run.t[-1]) if run.t.size else 1.0
    return 0.0, t1 if t1 > 1e-6 else 1.0


def _draw_time_ticks(ax, run, th, st, bands, t0, t1, lw_scale=1.0):
    """Round-time dots on the vehicle path, labelled with their second count.

    The dots survive overdraw where the line cannot (each is an isolated mark
    with a surface ring), and the LABEL is what finally makes the picture
    answerable without judging colour at all — the operator's complaint about
    the first version was that neighbouring shades were too close to read.
    Labels are pushed radially outward from the path's own centroid so they
    sit off the track rather than on it.
    """
    from matplotlib import patheffects

    marks = _tick_times(run.t)
    if not marks.size:
        return
    i = np.clip(np.searchsorted(run.t, marks), 0, run.x.size - 1)
    # A run's clock is not continuous: with `segment: engaged` the rows either
    # side of a disengagement are minutes apart, and the nearest sample to the
    # 120 s mark can be at 141 s. Drawing that dot would put a wrong time on a
    # right-looking place, so a mark with no sample near it gets NO tick.
    step = float(marks[1] - marks[0]) if marks.size > 1 else float(marks[0])
    near = np.abs(run.t[i] - marks) <= max(1.0, 0.05 * step)
    i = i[near]
    if not i.size:
        return
    ax.scatter(run.y[i], run.x[i], c=run.t[i],
               cmap=_cmap(th["ramp"], "veh", bands), norm=_norm(t0, t1),
               s=st["time_tick_ms"] * lw_scale ** 2, edgecolors=th["surface"],
               linewidths=1.6 * lw_scale, zorder=4.5)
    if not st["time_tick_labels"]:
        return
    cx, cy = float(run.x.mean()), float(run.y.mean())
    halo = [patheffects.withStroke(linewidth=2.6, foreground=th["surface"])]
    off = 14.0 * lw_scale
    placed: list[tuple[float, float]] = []
    # A path that laps can put two ticks in nearly the same place; two numbers
    # on top of each other are worse than one, so the later label is dropped
    # while its dot stays. Display coordinates, because the axes are equal
    # aspect but the pool is not square.
    min_gap = 34.0 * lw_scale
    for k, t in zip(i, run.t[i]):
        dx, dy = float(run.x[k] - cx), float(run.y[k] - cy)
        n = math.hypot(dx, dy) or 1.0
        px, py = ax.transData.transform((float(run.y[k]), float(run.x[k])))
        px, py = px + off * dy / n * 1.6, py + off * dx / n * 1.6
        if any(math.hypot(px - qx, py - qy) < min_gap for qx, qy in placed):
            continue
        placed.append((px, py))
        ax.annotate(f"{t:.0f}", (run.y[k], run.x[k]),
                    textcoords="offset points",
                    xytext=(off * dy / n, off * dx / n),
                    ha="center", va="center", fontsize=8.0 * lw_scale,
                    color=th["ink2"], zorder=4.6, path_effects=halo)


def _draw_paths(ax, run, th, st, lw_scale=1.0):
    """Vehicle, object, reference — in that z-order, the dashes on TOP.

    With ``time_color`` the vehicle is a LineCollection whose segments are
    shaded along the theme's blue ramp and the object a scatter on the orange
    one: hue keeps saying WHICH track, lightness says WHEN. The reference gets
    no ramp at all — it is a command, not a measurement, and shading it would
    imply it drifted.
    """
    from matplotlib.collections import LineCollection

    act = st["actual_color"] or th["series"]
    ref = st["ref_color"] or th["ink"]
    obj = th["series2"]
    bands = int(st.get("time_bands", 0) or 0)
    t0, t1 = _time_span(run)

    if run.cx.size:
        ax.plot(run.cy, run.cx, color=th["ink3"], lw=1.0 * lw_scale, alpha=0.45,
                solid_capstyle="round", zorder=3, label="approach / hold")

    if st["time_color"] and run.t.size == run.x.size and run.x.size > 1:
        pts = np.column_stack([run.y, run.x]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=_cmap(th["ramp"], "veh", bands),
                            norm=_norm(t0, t1), linewidths=st["actual_lw"] * lw_scale,
                            capstyle="round", joinstyle="round", zorder=4)
        lc.set_array(run.t[:-1])
        ax.add_collection(lc)
        act = th["ramp"][1]                       # the mid step, for markers
        if st["time_ticks"]:
            _draw_time_ticks(ax, run, th, st, bands, t0, t1, lw_scale)
    else:
        ax.plot(run.y, run.x, color=act, lw=st["actual_lw"] * lw_scale,
                alpha=0.95, solid_capstyle="round", solid_joinstyle="round",
                zorder=4, label="actual")

    if run.has_object:
        if st["time_color"]:
            ax.scatter(run.oy, run.ox, c=run.ot,
                       cmap=_cmap(th["ramp2"], "obj", bands),
                       norm=_norm(t0, t1), s=st["object_ms"] * lw_scale ** 2,
                       linewidths=0.0, zorder=3.5)
            obj = th["ramp2"][1]
        else:
            ax.scatter(run.oy, run.ox, color=obj,
                       s=st["object_ms"] * lw_scale ** 2, linewidths=0.0,
                       zorder=3.5, label="object")

    rx, ry = run.ref_xy(bool(st["ref_single_lap"]))
    if rx.size > 1:
        ax.plot(ry, rx, color=ref, lw=st["ref_lw"] * lw_scale,
                dashes=list(st["ref_dashes"]), dash_capstyle="round", zorder=5,
                label="reference")
    if st["show_markers"]:
        ax.plot(run.y[0], run.x[0], "o", ms=7.0 * lw_scale, mfc=act,
                mec=th["surface"], mew=1.8, zorder=6)
        ax.plot(run.y[-1], run.x[-1], "s", ms=6.5 * lw_scale, mfc=th["surface"],
                mec=act, mew=1.8, zorder=6)


def _time_scale(fig, rect, run, th, st, has_object: bool):
    """The time legend: one gradient row per track, sharing one t axis.

    A plain colorbar would say WHEN but not WHICH, and a plain legend the
    reverse — the tracks are colour-coded on two axes at once (hue = identity,
    lightness = time), so the key has to be too. Rows are stacked in draw
    order, vehicle first.
    """
    t0, t1 = _time_span(run)
    bands = int(st.get("time_bands", 0) or 0)
    rows = [("vehicle", th["ramp"])] + ([("object", th["ramp2"])]
                                        if has_object else [])
    cax = fig.add_axes(rect)
    cax.set_facecolor(th["surface"])
    grad = np.linspace(0.0, 1.0, 256).reshape(1, -1)
    for i, (_lab, ramp) in enumerate(rows):
        y = len(rows) - 1 - i                     # first row on top
        cax.imshow(grad, extent=(t0, t1, y, y + 0.72), aspect="auto",
                   cmap=_cmap(ramp, f"key{i}", bands), vmin=0.0, vmax=1.0)
    if bands > 1:
        # hairlines where the steps change, so the key can be read off
        for e in np.linspace(t0, t1, bands + 1)[1:-1]:
            cax.axvline(e, color=th["surface"], lw=1.0, zorder=3)
    cax.set_xlim(t0, t1)
    cax.set_ylim(0.0, len(rows) - 1 + 0.72)
    cax.set_yticks([len(rows) - 1 - i + 0.36 for i in range(len(rows))])
    cax.set_yticklabels([lab for lab, _ in rows], fontsize=9.5,
                        color=th["ink"])
    cax.tick_params(length=0, colors=th["ink3"], labelsize=8.5)
    for sp in cax.spines.values():
        sp.set_visible(False)
    tick_note = ""
    if st["time_ticks"]:
        marks = _tick_times(run.t)
        if marks.size > 1:
            tick_note = f"   ·   dot every {marks[1] - marks[0]:g} s"
    cax.set_xlabel(f"elapsed time  [s]{tick_note}", fontsize=9.5,
                   color=th["ink2"], labelpad=3)
    for lab in cax.get_yticklabels():
        lab.set_color(th["ink"])
    return cax


def _reference_key(fig, rect, th, st):
    """The reference's own key — a dash sample and a word, no ramp."""
    kax = fig.add_axes(rect)
    kax.set_axis_off()
    kax.set_xlim(0, 1)
    kax.set_ylim(0, 1)
    kax.plot([0.0, 0.26], [0.5, 0.5], color=st["ref_color"] or th["ink"],
             lw=st["ref_lw"], dashes=list(st["ref_dashes"]),
             dash_capstyle="round", clip_on=False)
    kax.text(0.32, 0.5, "reference", va="center", ha="left", fontsize=9.5,
             color=th["ink"])
    return kax


def plot_run(run: Run, box: dict, cfg: dict, out_dir: Path, plt) -> list[Path]:
    """One run, one figure. Two lines, one number, one legend, nothing else.

    Everything that identifies WHEN lives in the file name instead of on the
    picture (operator request, 2026-08-15): a slide has a caption, and a date
    stamped into the artwork only competes with it.
    """
    st, th = cfg["style"], THEMES[cfg["style"]["theme"]]
    pad = float(cfg["pool"]["pad_m"])
    W = (box["y"][1] - box["y"][0]) + 2 * pad
    H = (box["x"][1] - box["x"][0]) + 2 * pad

    # Margins in INCHES, so the axes rectangle is exactly the pool's aspect and
    # the key below always gets the same strip no matter how wide the pool.
    # The time key needs one gradient row per track, so the strip grows by a
    # row when there is an object to show.
    # The bottom strip, bottom-up in inches: the key's own tick labels and
    # axis title (0.40), the gradient rows themselves (0.20 each), a gap, and
    # the main axes' own x label (0.42). Computed rather than guessed because
    # the key grows a row when there is an object.
    keyed = bool(st["time_color"])
    key_rows = (2 if run.has_object else 1) if keyed else 0
    key_y0, key_h, key_gap, xlab = 0.40, 0.20 * key_rows, 0.14, 0.42
    L, Rr, T = 0.62, 0.28, 0.78
    B = (key_y0 + key_h + key_gap + xlab) if keyed else 1.05
    aw = float(st["panel_width_in"]) - L - Rr
    ah = aw * H / W
    fw, fh = float(st["panel_width_in"]), ah + T + B
    fig = plt.figure(figsize=(fw, fh), facecolor=th["surface"])
    ax = fig.add_axes([L / fw, B / fh, aw / fw, ah / fh])
    _setup_axes(ax, box, pad, th, st, cfg, _tag_patches(cfg))
    _draw_paths(ax, run, th, st)

    # --- title: WHICH CONTROLLER, then what it was asked to fly
    ax.text(0.0, 1.075, run.ctrl_label, transform=ax.transAxes, ha="left",
            va="bottom", fontsize=15.5, fontweight="bold", color=th["ink"])
    ax.text(0.0, 1.012, _traj_line(run.meta), transform=ax.transAxes,
            ha="left", va="bottom", fontsize=10, color=th["ink2"])

    # --- the one error number, in the corner the path is farthest from
    if st["stats_box"]:
        label, cm = run.headline(str(cfg["error"]))
        free = _free_corner(box, run.x, run.y)[0]
        ax.text(*({"upper left": (0.018, 0.962), "upper right": (0.982, 0.962),
                   "lower left": (0.018, 0.038),
                   "lower right": (0.982, 0.038)}[free]),
                f"{label}   {cm:.1f} cm", transform=ax.transAxes,
                ha="left" if "left" in free else "right",
                va="top" if "upper" in free else "bottom",
                fontsize=10, color=th["ink"], zorder=7,
                bbox=dict(boxstyle="round,pad=0.42", fc=th["box_fc"],
                          ec=th["box_ec"], lw=0.8, alpha=0.94))

    # --- the key, under the plot. With time colouring it is a gradient strip
    #     (identity x time in one object) plus the reference's own dash
    #     sample; without it, the plain legend the tracks already carry.
    if keyed:
        # Centre the whole key BLOCK, not the strip: the row labels sit
        # outside the strip axes and are part of what the eye centres on.
        lab_in, strip_in, gap_in, ref_in = 0.78, 3.10, 0.55, 1.35
        block = lab_in + strip_in + gap_in + ref_in
        x0 = max(0.2, (fw - block) / 2.0) + lab_in
        _time_scale(fig, [x0 / fw, key_y0 / fh, strip_in / fw, key_h / fh],
                    run, th, st, run.has_object)
        _reference_key(fig, [(x0 + strip_in + gap_in) / fw, key_y0 / fh,
                             ref_in / fw, key_h / fh], th, st)
    else:
        h, lab = ax.get_legend_handles_labels()
        leg = fig.legend(h, lab, loc="lower center", ncol=len(lab),
                         frameon=False, fontsize=10.5, handlelength=2.6,
                         columnspacing=2.6,
                         bbox_to_anchor=(L / fw + aw / fw / 2.0, 0.16 / fh))
        for txt in leg.get_texts():
            txt.set_color(th["ink"])

    if cfg["frame"] != "map":
        # The box is still the pool's SIZE, but the run was not placed in it —
        # say so on the figure, or a reader compares two of these by position.
        ax.text(0.5, 0.5, "DATUM FRAME — (0,0) is where ENGAGE was pressed;\n"
                          "the border is the pool's size, not its place",
                transform=ax.transAxes, ha="center", va="center", fontsize=11,
                color=th["ink3"], alpha=0.5, zorder=1)
    return _save(fig, out_dir / run.date, run.slug, st, plt)


def _save(fig, out_dir: Path, stem: str, st: dict, plt) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in st["formats"]:
        p = out_dir / f"{stem}.{fmt}"
        fig.savefig(p, dpi=int(st["dpi"]), facecolor=fig.get_facecolor())
        written.append(p)
    plt.close(fig)
    return written


# ============================================================== main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", type=Path,
                    default=REPO / "config" / "traj_plots.yaml",
                    help="YAML config (default config/traj_plots.yaml)")
    ap.add_argument("--dates", nargs="+", default=None,
                    help="override the config's dates: YYYYMMDD for the whole "
                         "day, YYYYMMDD_HHMMSS for one run, or all / latest")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="override out_dir")
    ap.add_argument("--segment", choices=["traj", "engaged", "all"], default=None)
    ap.add_argument("--error", choices=["auto", "cross", "radial"], default=None,
                    help="which single error number the figure shows")
    ap.add_argument("--theme", choices=["light", "dark"], default=None)
    ap.add_argument("--list", action="store_true",
                    help="say what would be plotted, write nothing")
    ap.add_argument("--show", action="store_true", help="open the figures too")
    args = ap.parse_args(argv)

    cfg = load_config(args.config if args.config.exists() else None)
    print(f"[cfg]  {_rel(args.config)}" if args.config.exists() else
          f"[cfg]  {_rel(args.config)} not found — using built-in defaults")
    if args.dates:
        cfg["dates"] = args.dates
    if args.out:
        cfg["out_dir"] = str(args.out)
    if args.segment:
        cfg["segment"] = args.segment
    if args.error:
        cfg["error"] = args.error
    if args.theme:
        cfg["style"]["theme"] = args.theme

    base = Path(cfg["base"])
    base = base if base.is_absolute() else REPO / base
    if not base.is_dir():
        print(f"[fail] no session tree at {base}")
        return 1

    specs, missing = resolve_dates(cfg, base)
    for d in missing:
        print(f"[none] {d}: no such date under {_rel(base)} — nothing to plot")
    if not specs:
        print("[fail] none of the requested dates exist")
        return 1

    runs, skips = discover(base, specs, cfg)
    box, prov = resolve_pool(cfg, sorted({r.run_dir for r in runs}))
    print(f"[pool] x [{box['x'][0]:+.3f}, {box['x'][1]:+.3f}] "
          f"y [{box['y'][0]:+.3f}, {box['y'][1]:+.3f}] m "
          f"= {box['x'][1] - box['x'][0]:.2f} × {box['y'][1] - box['y'][0]:.2f} m")
    print(f"       from {prov}")
    for r in runs:
        w = check_pool(box, r)
        if w:
            print(f"[warn] {w}")
        if r.stats.get("obj_note"):
            print(f"[obj]  {r.name}: {r.stats['obj_note']}")
        for who, xs, ys in (("vehicle", r.x, r.y), ("object", r.ox, r.oy)):
            n_out = outside_box(box, xs, ys)
            if n_out:
                print(f"[warn] {r.name}: {n_out} {who} sample(s) fall outside "
                      f"the pool box and are NOT drawn (the axes are the pool)")

    for s in skips:
        print(f"[none] {s}")

    if args.list:
        for r in runs:
            lab, cm = r.headline(str(cfg["error"]))
            print(f"[plot] {r.slug}.png   {r.ctrl_label}  {_traj_line(r.meta)}"
                  f"  {r.stats['n']} pts  {lab} {cm:.1f} cm")
        print(f"\n{len(runs)} figure(s) would be written, {len(skips)} skipped")
        return 0

    if not runs:
        print("\n[fail] nothing plottable in the requested date(s) — see the "
              "[none] lines above for why")
        return 1

    plt = _mpl(args.show)
    plt.rcParams.update({
        "font.family": cfg["style"]["font"], "savefig.dpi": cfg["style"]["dpi"],
        "figure.dpi": 130, "savefig.facecolor": THEMES[cfg["style"]["theme"]]["surface"],
    })

    out_root = Path(cfg["out_dir"])
    out_root = out_root if out_root.is_absolute() else REPO / out_root

    written = []
    for r in runs:
        written += plot_run(r, box, cfg, out_root, plt)
        print(f"[ok]   {r.name}  →  {written[-1].name}")

    print(f"\n{len(written)} file(s) under {out_root}")
    if skips:
        print(f"{len(skips)} run(s)/date(s) produced nothing — listed above")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
