#!/usr/bin/env python3
"""
replay_run.py — watch a recorded run back: the tag floor, where the vehicle
went, and the reference the diffusion policy was putting out at each update.

    ./c3 run-replay data/20260903/0903_150441
    ./c3 run-replay <RUN_DIR> --speed 2 --start 40
    ./c3 run-replay <RUN_DIR> --shots out/        # stills, no window

    (`run-replay`, not `replay` — that name already belongs to the depth
     recording player, c3_replay.py.)

    (or directly:
     QT_QPA_PLATFORM= ~/miniforge3/envs/rovgui-pose/bin/python \
         rov_gui/tools/replay_run.py <RUN_DIR>)

WHY IT EXISTS. A POLICY OBSERVE run is a diagnosis: the pilot flies slowly
past an object and the question afterwards is "what was the network drawing,
and where was the vehicle when it drew it". Live, that is 1-2 plans on screen
for half a second each; the answer is only visible if you can stop time.

WHAT IT DRAWS — and this is the whole design decision — is the REAL
``TrajectoryView``, the same widget the station flies with. That widget takes
exactly two inputs, ``add_status(MpcStatus)`` and
``add_policy_plan(PolicyPlanViz)``, and does the frame chain itself
(datum -> map, world FLU -> map z, body FLU -> map for the hull). So this
tool RECONSTRUCTS those two dataclasses from the recorded files and feeds
them; it never converts a coordinate of its own.

That is not laziness, it is the point. This repo has shipped frame and sign
errors into production twice (hw_nav.yaml's ``cam_t_flu`` block: a C3
extrinsic 0.26 m out; trajectory.py's ``_to_map_z`` block: an engage-datum z
offset that drew the vehicle lying on the tag mat). A replay tool with its
own transform would be a second implementation to drift, and a diagnosis
drawn in a frame the run was not flown in is worse than no diagnosis. Here,
if the picture is wrong, the LIVE picture was wrong the same way.

WHAT IT READS (all of it already written by an ordinary run):

  mpc_<HHMMSS>.csv       one row per 20 Hz control tick. ``px,py,pz`` is the
                         vehicle in world FLU of the engage-datum frame,
                         ``yaw_deg`` its heading, ``t_traj`` the mission
                         clock, ``rx,ry,rz`` the reference. Read BY NAME:
                         the header has grown five times and old runs are
                         missing columns this tool simply does without.
  mpc_<HHMMSS>.meta.json ``hardware.datum_tag_frame`` -> MpcStatus.datum,
                         which is what puts everything else in the map frame.
  policy_plan.csv        one row per KNOT of every plan the policy produced,
                         in datum NED, stamped ``t_rel`` on the same mission
                         clock as ``t_traj``. Includes the REFUSED ones,
                         which on these runs are most of them and are the
                         interesting half.
  nav_<HHMMSS>/map.json  (REC NAV) the tag floor and pool, self-contained.
                         Optional: without it the tag map named in the meta
                         is loaded instead, and without that the floor is
                         simply not drawn.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rov_gui.qt import Qt, QtCore, QtGui, QtWidgets          # noqa: E402
from rov_gui.state import MpcStatus, PolicyPlanViz           # noqa: E402
from rov_gui.widgets.trajectory import TrajectoryView        # noqa: E402
from rov_gui import theme                                         # noqa: E402

#: Replay tick. The SAME 20 Hz the station publishes MpcStatus at, so at
#: speed 1.0 the trail ages and the plans expire exactly as they did live —
#: `TrajectoryView` prunes both against wall time, and matching the cadence
#: is what makes "1x" mean it.
TICK_MS = 50


def _f(row: dict, key: str, default=float("nan")) -> float:
    """One CSV cell as a float. Missing column or empty cell -> default: the
    header has grown five times (schema 11 today) and a run recorded under an
    older one is still worth watching."""
    v = row.get(key, "")
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
class Run:
    """One recorded run, as the two streams the panel eats.

    ``ticks`` is the vehicle over time and ``plans`` is what the policy asked
    for, both keyed on the MISSION clock (``t_traj`` / ``t_rel``) — the only
    clock the two files share. Wall stamps are deliberately not used: the CSV
    counts from when the file opened and the plan log from when the mission
    armed, so aligning on them would slide the plans against the vehicle by
    however long the operator took to press START.
    """

    def __init__(self, run_dir: Path, csv_path: Path | None = None):
        self.dir = Path(run_dir)
        self.csv_path = csv_path or self._pick_csv()
        self.meta = self._load_meta()
        self.datum = self._load_datum()
        self.ticks = self._load_ticks()
        self.plans = self._load_plans()
        self.map_tags, self.pool, self.tag_size_m = self._load_map()

    # -- inputs ------------------------------------------------------------
    def _pick_csv(self) -> Path:
        """The BIGGEST mpc CSV in the folder, not the newest.

        A run folder routinely holds several: every engage opens one, and the
        two-second ones are the engagements that were refused or released
        immediately. The one worth replaying is the one with the flying in
        it, and size is the honest proxy for that.
        """
        cands = sorted(self.dir.glob("mpc_*.csv"),
                       key=lambda p: p.stat().st_size, reverse=True)
        if not cands:
            raise SystemExit(f"no mpc_*.csv in {self.dir}")
        return cands[0]

    def _load_meta(self) -> dict:
        p = self.csv_path.with_suffix(".meta.json")
        if not p.exists():
            return {}
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _load_datum(self):
        """(x, y, z, yaw_rad) — the engage pose in the TAG frame.

        Without it every drawn position is in the datum frame while the tag
        floor is in the map frame, i.e. the vehicle appears near the origin
        wherever it actually flew. Returning None is honest (the view then
        draws the datum frame and says so) but it is worth a warning.
        """
        d = ((self.meta.get("hardware") or {}).get("datum_tag_frame") or {})
        p0 = d.get("p0")
        if not p0 or d.get("yaw0_deg") is None:
            return None
        return (float(p0[0]), float(p0[1]), float(p0[2]),
                math.radians(float(d["yaw0_deg"])))

    def _load_ticks(self) -> list:
        """Every row that has a POSITION, in file order.

        Rows with no tag fix carry `nan` px/py (the dead-reckoning generation
        writes a row even then) and are dropped: a marker cannot be drawn at
        nan, and interpolating across the gap would invent a track through
        exactly the moments the localizer could not see.
        """
        out = []
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                px, py, pz = _f(row, "px"), _f(row, "py"), _f(row, "pz")
                if not (math.isfinite(px) and math.isfinite(py)):
                    continue
                t_traj = _f(row, "t_traj")
                out.append({
                    "t": _f(row, "t", 0.0),
                    "t_traj": t_traj,
                    "p_flu": (px, py, pz),
                    "yaw_deg": _f(row, "yaw_deg", 0.0),
                    "ref_flu": ((_f(row, "rx"), _f(row, "ry"), _f(row, "rz"))
                                if math.isfinite(_f(row, "rx")) else None),
                    "engaged": bool(int(_f(row, "engaged", 0) or 0)),
                    "traj_on": bool(int(_f(row, "traj_on", 0) or 0)),
                    "observe": bool(int(_f(row, "observe", 0) or 0)),
                    "mode": row.get("mode", ""),
                    "speed": _f(row, "speed_m_s", float("nan")),
                    "plan_id": row.get("plan_id", ""),
                })
        if not out:
            raise SystemExit(f"{self.csv_path} has no rows with a position")
        return out

    def _load_plans(self) -> list:
        """Every composed plan, knots in datum NED, newest last.

        Grouped by (plan_id, t_rel) rather than plan_id alone: ids restart at
        every re-arm, and a folder can hold more than one mission.
        """
        p = self.dir / "policy_plan.csv"
        if not p.exists():
            return []
        groups: dict = {}
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row.get("plan_id", ""), row.get("t_rel", ""))
                g = groups.setdefault(key, {
                    "plan_id": int(_f(row, "plan_id", -1)),
                    "status": row.get("status", ""),
                    "follower": row.get("follower", ""),
                    "t_rel": _f(row, "t_rel", 0.0),
                    "reason": "", "knots": [],
                })
                if row.get("reason"):
                    g["reason"] = g["reason"] or row["reason"]
                g["knots"].append((
                    int(_f(row, "k", 0)), _f(row, "t_knot", 0.0),
                    _f(row, "x_ned", 0.0), _f(row, "y_ned", 0.0),
                    _f(row, "z_ned", 0.0), _f(row, "yaw_deg", 0.0)))
        out = []
        for g in groups.values():
            ks = sorted(g["knots"])
            if len(ks) < 2:
                continue                 # a polyline needs two points
            t0 = ks[0][1]
            dt = ks[1][1] - ks[0][1]
            out.append({
                "plan_id": g["plan_id"], "status": g["status"],
                "follower": g["follower"], "t_rel": g["t_rel"],
                "reason": g["reason"], "t0": t0,
                "dt": dt if dt > 1e-9 else 0.2,
                "p_ned": (tuple(k[2] for k in ks), tuple(k[3] for k in ks),
                          tuple(k[4] for k in ks)),
                "yaw": tuple(math.radians(k[5]) for k in ks),
            })
        out.sort(key=lambda g: g["t_rel"])
        return out

    def _load_map(self):
        """The tag floor and pool, from REC NAV's map.json when it is there.

        map.json is SELF-CONTAINED — tags with their yaw, the pool extent and
        the tag size — which is what makes a replay reproducible: it is the
        map as it was that day, not as config/tag_map_full.yaml happens to be
        now. The named config is the fallback for a run flown without REC NAV.
        """
        navs = sorted(self.dir.glob("nav_*/map.json"))
        if navs:
            try:
                with open(navs[-1], encoding="utf-8") as f:
                    m = json.load(f)
                tags = [(float(v["x"]), float(v["y"]),
                         float(v.get("yaw_rad", 0.0)), int(tid), 0)
                        for tid, v in m.get("tags", {}).items()]
                pool = None
                pn = m.get("pool_ned") or {}
                if pn.get("x") and pn.get("y"):
                    x0, x1 = float(pn["x"][0]), float(pn["x"][1])
                    y0, y1 = float(pn["y"][0]), float(pn["y"][1])
                    pool = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                return tags, pool, float(m.get("tag_size_m", 0.17))
            except (OSError, ValueError, KeyError, TypeError):
                pass
        hw = self.meta.get("hardware") or {}
        path = hw.get("tag_map")
        if path and Path(path).exists():
            try:
                from rov_gui.control.tagnav import TagMap
                tm = TagMap.load(path)
                tags = [(float(t[0]), float(t[1]),
                         math.atan2(float(R[1, 0]), float(R[0, 0])), tid, k)
                        for tid, poses in sorted(tm.instances.items())
                        for k, (R, t) in enumerate(poses)]
                return tags, None, float(hw.get("tag_size_m", 0.17))
            except Exception:                                # noqa: BLE001
                pass
        return [], None, float(hw.get("tag_size_m", 0.17))

    # -- derived -----------------------------------------------------------
    @property
    def t_span(self) -> tuple:
        return self.ticks[0]["t"], self.ticks[-1]["t"]

    def describe(self) -> str:
        t0, t1 = self.t_span
        kind = ((self.meta.get("trajectory") or {}).get("kind")
                or (self.meta.get("mission") or {}).get("shape") or "?")
        obs = bool((self.meta.get("policy") or {}).get("observe"))
        return (f"{self.csv_path.name}  {t1 - t0:.0f} s  "
                f"{len(self.ticks)} ticks  {len(self.plans)} plans  "
                f"kind={kind}{'  OBSERVE' if obs else ''}  "
                f"tags={len(self.map_tags)}"
                + ("" if self.datum else "  [NO DATUM — datum frame]"))


# --------------------------------------------------------------------------
# feeding the real widget
# --------------------------------------------------------------------------
# Module level, not methods, so the still-renderer can drive a bare
# TrajectoryView without constructing a window (and without the
# `Window.__new__` trick the first draft used to borrow them).
def feed_tick(view, datum, r: dict) -> None:
    """One recorded control tick -> the panel, as MpcStatus.

    Every frame conversion in here is the widget's, not ours: `datum` is what
    turns the CSV's datum-frame FLU position back into map coordinates, and
    it is passed through untouched.

    THE DATUM IS SET SEPARATELY, and that is not obvious: `add_status` does
    NOT read `s.datum` — the live station calls `view.set_datum()` itself
    from `window._on_mpc_status`, because a new datum also means a new run
    and the trails have to be cleared with it. A replay that only filled the
    dataclass drew the whole run in the DATUM frame, i.e. a metre from the
    origin instead of where it flew, with the tag floor around it unmoved
    (caught by the round-trip test, 2026-09-03). So it is applied here too,
    idempotently.
    """
    if getattr(view, "datum", None) != datum:
        view.set_datum(datum)
    view.add_status(MpcStatus(
        engaged=r["engaged"], traj_on=r["traj_on"],
        observe=r["observe"], commanding=r["engaged"] and not r["observe"],
        mode=r["mode"],
        # `kind` only has to tell the panel this is a streamed-plan mission,
        # so it draws no placed geometry under the vehicle.
        scenario=({"kind": "policy"} if r["traj_on"] else None),
        datum=datum,
        p_flu=r["p_flu"], yaw_flu_deg=r["yaw_deg"],
        ref_flu=r["ref_flu"],
        t_traj=(r["t_traj"] if math.isfinite(r["t_traj"]) else None),
        speed_m_s=(r["speed"] if math.isfinite(r["speed"]) else None),
        reason="replay"))


def feed_plan(view, g: dict) -> None:
    """One recorded plan -> the panel, as PolicyPlanViz. The knots are
    already in the datum NED frame the live path handed over, so they go
    straight in."""
    view.add_policy_plan(PolicyPlanViz(
        plan_id=g["plan_id"], status=g["status"], p_ned=g["p_ned"],
        yaw=g["yaw"], t0=g["t0"], dt=g["dt"], reason=g["reason"]))


def _centre_on_vehicle(view) -> None:
    """Pan so the hull stays in the middle of the plot.

    A replay is watched ZOOMED IN — a 1 s plan spans about 4 cm, which on a
    5 m pool view is a red speck (the live panel had a magnified inset for
    exactly this and it was removed in favour of one plot). Zoomed in, a
    vehicle that is flying leaves the frame in seconds, so the view follows
    it. Done by projecting with no pan and cancelling the result, which works
    in both modes because `_px` is the only projection either uses.
    """
    if view.p_act is None:
        return
    view.pan = QtCore.QPointF(0.0, 0.0)
    q = view._px(*view.p_act)
    view.pan = QtCore.QPointF(view.width() / 2.0 - q.x(),
                              view.height() / 2.0 - q.y())


# --------------------------------------------------------------------------
# the window
# --------------------------------------------------------------------------
class ReplayWindow(QtWidgets.QWidget):
    """The panel plus a transport bar. Everything drawn is the live widget."""

    def __init__(self, run: Run, speed: float = 1.0, start_s: float = 0.0,
                 zoom: float = 6.0):
        super().__init__()
        self.run = run
        self.setWindowTitle(f"DP replay — {run.dir.name}/{run.csv_path.name}")
        self.resize(1000, 760)
        self.setStyleSheet(f"background:{theme.BG}; color:{theme.TEXT};")

        self.view = TrajectoryView()
        self.view.tag_size_m = run.tag_size_m
        if run.map_tags:
            self.view.set_map_tags(run.map_tags)
        if run.pool:
            self.view.set_pool(run.pool)

        self.btn_play = QtWidgets.QPushButton("PLAY")
        self.btn_play.setCheckable(True)
        self.btn_play.setChecked(True)
        self.btn_play.toggled.connect(
            lambda on: self.btn_play.setText("PAUSE" if on else "PLAY"))
        self.slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, max(0, len(run.ticks) - 1))
        self.slider.sliderMoved.connect(self._scrub)
        self.spd = QtWidgets.QDoubleSpinBox()
        self.spd.setRange(0.1, 20.0)
        self.spd.setSingleStep(0.5)
        self.spd.setValue(float(speed))
        self.spd.setSuffix("x")
        self.btn_3d = QtWidgets.QPushButton("3D")
        self.btn_3d.setCheckable(True)
        self.btn_3d.toggled.connect(self._set_3d)
        # FOLLOW is on by default and the zoom starts in close, because the
        # thing being diagnosed is centimetres long: a 1 s policy chunk spans
        # ~4 cm and the pool view is 5 m. Both are still operator knobs —
        # unchecking follow hands the pan back to the mouse.
        self.follow = QtWidgets.QCheckBox("follow")
        self.follow.setChecked(True)
        self.zoom = QtWidgets.QDoubleSpinBox()
        self.zoom.setRange(0.2, 40.0)
        self.zoom.setSingleStep(0.5)
        self.zoom.setPrefix("zoom ")
        self.zoom.valueChanged.connect(self._set_zoom)
        self.lbl = QtWidgets.QLabel("")
        self.lbl.setStyleSheet("font-family:monospace; font-size:11px;")

        bar = QtWidgets.QHBoxLayout()
        for w in (self.btn_play, self.slider, self.spd, self.zoom,
                  self.follow, self.btn_3d):
            bar.addWidget(w)
        bar.setStretch(1, 1)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.view, 1)
        lay.addLayout(bar)
        lay.addWidget(self.lbl)

        self._i = 0
        self._next_plan = 0
        self.zoom.setValue(float(zoom))          # also applies it to the view
        self._seek(self._index_at(start_s))
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(TICK_MS)

    # -- transport ---------------------------------------------------------
    def _index_at(self, t_s: float) -> int:
        for i, r in enumerate(self.run.ticks):
            if r["t"] >= t_s:
                return i
        return 0

    def _set_3d(self, on: bool) -> None:
        self.view.three_d = bool(on)
        if self.follow.isChecked():
            _centre_on_vehicle(self.view)
        self.view.update()

    def _set_zoom(self, z: float) -> None:
        self.view.zoom = float(z)
        if self.follow.isChecked():
            _centre_on_vehicle(self.view)
        self.view.update()

    def _scrub(self, i: int) -> None:
        self.btn_play.setChecked(False)
        self._seek(int(i))

    def _seek(self, i: int) -> None:
        """Jump anywhere and rebuild the picture from scratch.

        A seek CLEARS the view and re-feeds a pre-roll rather than trying to
        rewind: the trail is a deque pruned by wall time and the plans are the
        last two received, so there is no state to invert. The pre-roll is
        every tick in the last few seconds plus the last two plans due by
        then, which is exactly what would have been on screen.
        """
        self._i = max(0, min(i, len(self.run.ticks) - 1))
        self.view.clear()
        row = self.run.ticks[self._i]
        pre = [r for r in self.run.ticks
               if 0.0 <= row["t"] - r["t"] <= 3.0 and r["t"] <= row["t"]]
        for r in pre[:-1]:
            self._feed_tick(r)
        # the two most recent plans due at this instant, oldest first
        t_now = row["t_traj"]
        due = [g for g in self.run.plans
               if math.isfinite(t_now) and g["t_rel"] <= t_now]
        self._next_plan = len(due)
        for g in due[-2:]:
            self._feed_plan(g)
        self._feed_tick(row)
        self._refresh_label()

    def _tick(self) -> None:
        if not self.btn_play.isChecked():
            return
        step = max(1, int(round(float(self.spd.value()))))
        target = self._i + step
        if target >= len(self.run.ticks):
            self.btn_play.setChecked(False)
            return
        # Release every plan whose mission time we just passed, in order, so a
        # fast replay never skips one — the plans ARE the record here, and
        # dropping some because the slider moved quickly would quietly change
        # what the run looked like.
        t_to = self.run.ticks[target]["t_traj"]
        while (self._next_plan < len(self.run.plans)
               and math.isfinite(t_to)
               and self.run.plans[self._next_plan]["t_rel"] <= t_to):
            self._feed_plan(self.run.plans[self._next_plan])
            self._next_plan += 1
        self._i = target
        self._feed_tick(self.run.ticks[self._i])
        self.slider.setValue(self._i)
        self._refresh_label()

    # -- feeding the real widget -------------------------------------------
    def _feed_tick(self, r: dict) -> None:
        feed_tick(self.view, self.run.datum, r)
        if self.follow.isChecked():
            _centre_on_vehicle(self.view)

    def _feed_plan(self, g: dict) -> None:
        feed_plan(self.view, g)

    def _refresh_label(self) -> None:
        r = self.run.ticks[self._i]
        t0, t1 = self.run.t_span
        last = (self.run.plans[self._next_plan - 1]
                if self._next_plan else None)
        pl = ("plan --" if last is None else
              f"plan #{last['plan_id']} {last['status']}"
              f"{'  ' + last['reason'][:60] if last['reason'] else ''}")
        self.lbl.setText(
            f"t {r['t'] - t0:7.2f} / {t1 - t0:.1f} s   "
            f"mission {r['t_traj']:7.2f} s   "
            f"{self._next_plan}/{len(self.run.plans)} plans   {pl}")

    def keyPressEvent(self, ev) -> None:                     # noqa: N802
        k = ev.key()
        if k == Qt.Key.Key_Space:
            self.btn_play.setChecked(not self.btn_play.isChecked())
        elif k == Qt.Key.Key_Right:
            self.btn_play.setChecked(False)
            self._seek(self._i + 1)
            self.slider.setValue(self._i)
        elif k == Qt.Key.Key_Left:
            self.btn_play.setChecked(False)
            self._seek(self._i - 1)
            self.slider.setValue(self._i)
        else:
            super().keyPressEvent(ev)


# --------------------------------------------------------------------------
def _shots(run: Run, out: Path, n: int, three_d: bool,
           zoom: float = 6.0) -> list:
    """Stills at N moments spread over the plans, for a run you want to look
    at without sitting through it. Rendered off a STANDALONE view at a
    generous size — the docked one is whatever the layout left it."""
    out.mkdir(parents=True, exist_ok=True)
    saved = []
    if not run.plans:
        return saved
    picks = [run.plans[int(i * (len(run.plans) - 1) / max(1, n - 1))]
             for i in range(n)]
    for j, g in enumerate(picks):
        v = TrajectoryView()
        v.resize(900, 680)
        v.tag_size_m = run.tag_size_m
        if run.map_tags:
            v.set_map_tags(run.map_tags)
        if run.pool:
            v.set_pool(run.pool)
        v.three_d = three_d
        v.azimuth_deg, v.elev_deg = 25.0, 35.0
        v.zoom = zoom
        i = 0
        for i, r in enumerate(run.ticks):
            if math.isfinite(r["t_traj"]) and r["t_traj"] >= g["t_rel"]:
                break
        for r in run.ticks[max(0, i - 60):i + 1]:
            feed_tick(v, run.datum, r)
        prev = [x for x in run.plans if x["t_rel"] <= g["t_rel"]][-2:]
        for x in prev:
            feed_plan(v, x)
        _centre_on_vehicle(v)
        img = QtGui.QImage(v.size(), QtGui.QImage.Format.Format_RGB32)
        img.fill(QtGui.QColor(theme.BG))
        v.render(img)
        p = out / f"plan_{j:02d}_id{g['plan_id']}_{g['status']}.png"
        img.save(str(p))
        saved.append(p)
    return saved


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Replay a recorded run: tag floor, vehicle track, and the "
                    "diffusion policy's reference at every update.")
    ap.add_argument("run_dir", help="a run folder (data/YYYYMMDD/MMDD_HHMMSS[_kind])")
    ap.add_argument("--csv", default=None,
                    help="which mpc_*.csv (default: the biggest one)")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--start", type=float, default=0.0,
                    help="seconds into the CSV to start at")
    ap.add_argument("--3d", dest="three_d", action="store_true",
                    help="open in the tilted view")
    ap.add_argument("--shots", metavar="DIR", default=None,
                    help="render stills instead of opening a window")
    ap.add_argument("--n-shots", type=int, default=6)
    ap.add_argument("--zoom", type=float, default=6.0,
                    help="start zoom; a policy chunk is only ~4 cm long "
                         "(default 6)")
    a = ap.parse_args(argv)

    if a.shots:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    run = Run(Path(a.run_dir), Path(a.csv) if a.csv else None)
    print(run.describe())
    if not run.plans:
        print("  (no policy_plan.csv — the vehicle track will replay, but "
              "there are no policy references in this run)")
    if run.datum is None:
        print("  WARNING: no hardware.datum_tag_frame in the meta — the "
              "track is drawn in the DATUM frame, so it will not sit on the "
              "tag floor.")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    if a.shots:
        for p in _shots(run, Path(a.shots), max(1, a.n_shots), a.three_d,
                        a.zoom):
            print(f"  shot  {p}")
        return 0
    w = ReplayWindow(run, speed=a.speed, start_s=a.start, zoom=a.zoom)
    w.btn_3d.setChecked(bool(a.three_d))
    w.show()
    print("  space = play/pause,  arrows = step,  drag = pan,  wheel = zoom")
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
