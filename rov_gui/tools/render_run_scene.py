#!/usr/bin/env python3
"""
render_run_scene.py — a flown run as a video: the pool, the vehicle and the
policy's reference on the left; the depth the policy actually saw on the right.

    python -m rov_gui.tools.render_run_scene \
        --run data/20260906/0906_191935_observe

    +--------------------------------------------+------------------+
    |  AprilTag map (3-D) + DP reference + ROV    |  FoundationStereo|
    +--------------------------------------------+------------------+

Nothing here is re-simulated: every element is read back from what the run
wrote, and the four sources are joined on ONE clock.

    tag map        config/tag_map_full.yaml, checked against the run meta's
                   `hardware.tag_map_sha1` — the run stores the PATH and the
                   hash, not the map, so this refuses a map that has changed
                   under it rather than drawing a pool that was never flown.
    ROV pose       mpc_*.csv  (px, py, pz, yaw_deg) at 20 Hz
    DP reference   policy_plan.csv (per-knot x_ned/y_ned/z_ned) + plans.jsonl
                   (status, anchor, ckpt) — 1426 plans on this run
    depth          policy_obs/depth/*.png, the uint16 millimetre maps written
                   by --record-depth, drawn with the SHARED colour rule
                   (rov_gui/depth_colour.py) so a frame here and a frame of a
                   land episode video mean the same distance by the same red.

FRAMES — the part that silently ruins a picture
-----------------------------------------------
Three of the four live in DIFFERENT frames, and the conversions are taken from
the code that produced them rather than re-derived:

* The CSV and the plans are in the **datum frame**: the pose at engagement is
  its origin and its yaw zero (``MpcWorker._datumize``). The inverse is a plain
  horizontal isometry, and the run meta carries both halves of it
  (``hardware.datum_tag_frame`` = p0, yaw0_deg):

      p_map = p0 + Rz(+yaw0) @ p_datum          yaw_map = yaw_datum + yaw0

* The **map frame is the anchor tag's**, and it is NED-like: a tag lying
  print-up has +z INTO the floor, so +z is DOWN and the mat is z ~ 0
  (``control/tagnav.py``'s header; the map here measures z -0.050..+0.104 m
  over 140 instances). The view therefore takes world-up = -z.
* The **vehicle is body FLU** (+x forward, +y LEFT, +z UP) and
  ``widgets/rov_shape.body_to_map`` is the only thing that flips y and z. Its
  parts and that flip are unit-tested (``tests/test_rov_shape.py``), which is
  why this file imports them instead of drawing its own box.

The vehicle is drawn with YAW ONLY, exactly as the station's trajectory panel
draws it — roll and pitch are in the CSV and are shown as numbers, not as
attitude, so the picture cannot imply an attitude the panel never showed.

TIME — one clock, derived from the data
---------------------------------------
``policy_obs/index.csv`` stamps frames on a monotonic clock; the CSV and the
plans use run-relative seconds. Every plan carries BOTH (``obs_t_mono`` and
``obs_t``), so the offset is measured from the run itself rather than assumed,
and it is checked for drift before anything is drawn. On the 2026-09-06 run all
1426 plans' ``obs_t_mono`` land exactly on a recorded depth frame, so the join
is exact and no interpolation happens anywhere.

WHAT THIS CANNOT SHOW
---------------------
The depth on disk is the subset the POLICY consumed (2854 of 9469 frames on
that run, ~5 Hz of the ~10.8 Hz the network ran at), because that is what
``--record-depth`` is fed. It is not the panel's stream, and a gap here is a
gap in the recording, not in the depth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rov_gui.control.state_assembler import rot_zyx            # noqa: E402
from rov_gui.control.tagnav import TagMap                      # noqa: E402
from rov_gui.depth_colour import Z_FAR_M, Z_NEAR_M, mm_to_t_valid  # noqa: E402
from rov_gui.depth_panel import (BG, PANEL_H, PANEL_W,         # noqa: E402
                                 depth_panel, text)

def _rov_shape():
    """``rov_gui.widgets.rov_shape`` WITHOUT importing the widgets package.

    That module is deliberately Qt-free ("No Qt here on purpose: the frame
    algebra is the part that can be wrong in a way nobody notices"), but its
    package's ``__init__`` pulls the whole widget set and therefore a Qt
    binding — which this offline renderer must not need, since the env that
    holds FoundationStereo and PyAV has no PyQt. Loading the file directly is
    the smallest thing that keeps ONE geometry: the alternative is a second
    copy of the vehicle, which is exactly what rov_shape.py exists to prevent.
    """
    import importlib.util
    path = REPO / "rov_gui" / "widgets" / "rov_shape.py"
    spec = importlib.util.spec_from_file_location("rov_shape_offline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SHAPE = _rov_shape()
body_to_map, rov_parts = _SHAPE.body_to_map, _SHAPE.parts
rov_footprint, tcp_body_m = _SHAPE.footprint, _SHAPE.tcp_body_m
tcp_body_for_meta = _SHAPE.tcp_body_for_meta

SCENE = (1024, 720)          # left panel (w, h)
SIDE = (PANEL_W, 720)        # right column; the depth panel itself is 640x480

# BGR. The scene is an instrument, so the palette says WHAT a thing is rather
# than being decorative: the pool is grey, the vehicle is the station's green,
# and the policy's proposal is the only saturated colour on the left.
C_GRID = (48, 48, 48)
C_TAG = (66, 62, 58)
C_TAG_EDGE = (120, 112, 104)
C_ANCHOR = (90, 200, 255)
C_PATH_AHEAD = (74, 74, 74)
C_PLAN = {"accept": (120, 255, 120), "clip": (100, 190, 255),
          "late": (150, 150, 150), "reject": (90, 90, 235)}
C_ROV = {"hull": (120, 200, 140), "tube": (170, 190, 220), "jaw": (90, 200, 255),
         "thruster": (110, 110, 110), "camera": (220, 180, 90)}


# ============================================================== 3-D projection
class View:
    """A fixed pinhole view of a fixed scene: world points in, pixels out.

    Fitted ONCE, from every point that will ever be drawn, so the mapping is
    constant for the whole video. An autoscaling view would rescale as the path
    grows and the motion of the frame would be indistinguishable from the
    motion of the vehicle.
    """

    def __init__(self, R, eye, focal, centre):
        self.R, self.eye, self.focal, self.centre = R, eye, focal, centre

    @classmethod
    def fit(cls, points, size=SCENE, azimuth=-60.0, elevation=32.0,
            margin=0.88, world_up=(0.0, 0.0, -1.0)):
        """`world_up` defaults to -z because the map frame is NED."""
        pts = np.asarray(points, np.float64).reshape(-1, 3)
        centre_w = (pts.max(0) + pts.min(0)) / 2.0
        radius = max(float(np.linalg.norm(pts - centre_w, axis=1).max()), 1e-3)
        az, el = math.radians(azimuth), math.radians(elevation)
        up = np.asarray(world_up, np.float64)
        # Elevation is measured along world-up, which is -z here, so a positive
        # elevation looks DOWN at the floor whichever way the map's z points.
        direction = (np.array([math.cos(el) * math.cos(az),
                               math.cos(el) * math.sin(az), 0.0])
                     + up * math.sin(el))
        direction /= np.linalg.norm(direction)
        eye = centre_w + direction * (radius * 3.0)

        f = centre_w - eye
        f /= np.linalg.norm(f)
        # OpenCV's view basis: X right, Y DOWN, Z forward. Built in that
        # handedness or the picture comes out rotated 180 degrees — which still
        # looks like a plausible 3-D scene, just with up and left exchanged.
        right = np.cross(f, up)
        if np.linalg.norm(right) < 1e-6:
            right = np.cross(f, [1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        down = np.cross(f, right)
        R = np.stack([right, down, f])

        view = (R @ (pts - eye).T).T
        z = np.maximum(view[:, 2], 1e-6)
        w, h = size
        fx = margin * (w / 2) / max(np.abs(view[:, 0] / z).max(), 1e-6)
        fy = margin * (h / 2) / max(np.abs(view[:, 1] / z).max(), 1e-6)
        return cls(R, eye, min(fx, fy), (w / 2.0, h / 2.0))

    def __call__(self, points):
        """(N,3) world -> (N,2) int pixels, plus the view-space depth."""
        p = np.asarray(points, np.float64).reshape(-1, 3)
        v = (self.R @ (p - self.eye).T).T
        z = np.maximum(v[:, 2], 1e-6)
        u = self.centre[0] + self.focal * v[:, 0] / z
        w = self.centre[1] + self.focal * v[:, 1] / z
        return np.stack([u, w], axis=1).astype(np.int32), v[:, 2]


# =================================================================== the run
class Run:
    """Everything the video needs, read once and joined on one clock."""

    def __init__(self, run_dir: Path, tag_map_path: Path | None = None,
                 strict_map: bool = True, engagement: str | None = None):
        self.dir = Path(run_dir)
        if not self.dir.is_dir():
            raise FileNotFoundError(f"{self.dir} does not exist")
        self.metas = sorted(self.dir.glob("*.meta.json"))
        if not self.metas:
            raise FileNotFoundError(f"{self.dir}: no <csv>.meta.json — is this a run?")

        # A RUN FOLDER CAN HOLD MORE THAN ONE ENGAGEMENT, and the per-plan logs
        # are SHARED between them: data/20260906/0906_191935_observe
        # holds mpc_191935 (epoch 2, plans 1..1426) and mpc_200050 41 minutes
        # later (epoch 4, plans 1991..2203), while plans.jsonl and
        # policy_plan.csv carry both — and t_rel restarts at 0 for the second.
        # Reading them wholesale therefore draws the SECOND engagement's
        # reference at the first engagement's times, on any frame before the
        # second one ended (~230 s here). Everything below is filtered by
        # epoch, and the epoch comes from the chosen engagement's own meta.
        self._load_depth_index()
        self._load_plan_records()
        self.meta_path = self._pick_engagement(engagement)
        self.meta = json.loads(self.meta_path.read_text())
        self.csv_path = self.meta_path.with_suffix("").with_suffix(".csv")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"{self.csv_path} is missing")
        self.epoch = (self.meta.get("policy") or {}).get("epoch")
        # The TCP dot is THIS RUN's TCP (meta), not today's module: the jaw
        # moved on 2026-09-08 and plans.jsonl p_tcp / T_body_tcp were recorded
        # with the point of the day (rov_shape.tcp_body_for_meta).
        self.tcp_body, self.tcp_source = tcp_body_for_meta(self.meta)

        hw = self.meta.get("hardware", {})
        self.tag_size = float(hw.get("tag_size_m", 0.17))
        self.tag_map_path = Path(tag_map_path or hw.get("tag_map", ""))
        self._check_map_hash(hw.get("tag_map_sha1"), strict_map)
        self.tags = TagMap.load(str(self.tag_map_path))

        d = hw.get("datum_tag_frame") or {}
        if "p0" not in d:
            raise ValueError(f"{self.meta_path.name}: no hardware.datum_tag_frame "
                             f"— without the datum the CSV cannot be put on the "
                             f"map, and drawing it anyway would be a picture of "
                             f"the wrong pool")
        self.p0 = np.asarray(d["p0"], float)
        self.yaw0 = math.radians(float(d["yaw0_deg"]))
        self.Rz_inv = rot_zyx(0.0, 0.0, self.yaw0)     # datum -> tag frame

        self._load_csv()
        self._load_plans()
        self.clipped = 0
        if not self.has_depth:
            self._spine_from_csv()
        else:
            self._clip_spine_to_engagement()

    # ------------------------------------------------------- which engagement
    def _epoch_of(self, meta_path: Path):
        try:
            return (json.loads(meta_path.read_text()).get("policy") or {}).get("epoch")
        except Exception:                                        # noqa: BLE001
            return None

    def _pick_engagement(self, want: str | None) -> Path:
        """Choose which engagement in the folder this video is about.

        By DATA, not by filename order: the recorded depth frames belong to
        exactly one of them, so the engagement whose plans were stamped inside
        the depth-frame window is the one these pictures can be built from.
        """
        if want:
            hits = [m for m in self.metas if want in m.name]
            if not hits:
                raise ValueError(f"--engagement {want!r} matches none of "
                                 f"{[m.name for m in self.metas]}")
            return hits[0]
        if len(self.metas) == 1:
            return self.metas[0]
        if not self.has_depth:
            # No depth window to match against, so the honest tie-break is the
            # engagement with the most pose rows — and it is stated, not hidden.
            def rows(m):
                try:
                    return int((json.loads(m.read_text()).get("run") or {})
                               .get("rows") or 0)
                except Exception:                                # noqa: BLE001
                    return 0
            best = max(self.metas, key=rows)
            print(f"NOTE: {len(self.metas)} engagements and no recorded depth "
                  f"to date them by; picking the longest ({best.name}, "
                  f"{rows(best)} rows). --engagement overrides.")
            return best
        # ONE depth recording can span SEVERAL engagements: the recorder is
        # armed per station process while the CSVs are per engagement
        # [측정 2026-09-07, data/20260907/
        # 0907_164659: 1160 frames over 343 s across epochs 2,4,6,8,10]. So the
        # question is not "whose plans are inside the depth window" — with one
        # recording spanning all of them, every candidate scores high — but
        # "how many frames fall inside THIS engagement's window", which is
        # exactly how much of the page it can fill.
        best, scores = None, []
        for m in self.metas:
            e = self._epoch_of(m)
            w = self._plan_window(e)
            n = 0 if w is None else int(((self.t_depth >= w[0] - 1.0)
                                         & (self.t_depth <= w[1] + 1.0)).sum())
            scores.append((n, m.name, e, w))
            if best is None or n > best[0]:
                best = (n, m)
        print(f"NOTE: {len(self.metas)} engagements in this folder and ONE "
              f"depth recording across them (the per-plan logs are shared "
              f"too). Picking the one with the most recorded frames:")
        for n, name, e, w in scores:
            span = f"{w[1] - w[0]:6.0f} s" if w else "  no plans"
            print(f"        {name}  epoch {e}  {span}  {n:5d} frames"
                  f"{'   <- chosen' if best[1].name == name else ''}")
        if best[0] == 0:
            raise ValueError(
                "no engagement has a recorded depth frame inside its window; "
                "pass --engagement <name fragment> to choose explicitly")
        print("        --engagement <fragment> exports another; each is its "
              "own page")
        return best[1]

    def _plan_window(self, epoch):
        """(first, last) monotonic stamp of an epoch's plans, or None."""
        mono = [r["mono"] for r in self._plan_recs
                if r["epoch"] == epoch and r["mono"] is not None]
        return (min(mono), max(mono)) if mono else None

    # ------------------------------------------------------------- the map
    def _check_map_hash(self, want: str | None, strict: bool) -> None:
        self.map_sha1 = ""
        if not self.tag_map_path.exists():
            raise FileNotFoundError(
                f"tag map not found: {self.tag_map_path}. The run stores the "
                f"PATH and a hash, never the map itself — pass --tag-map if it "
                f"has moved.")
        self.map_sha1 = hashlib.sha1(
            self.tag_map_path.read_bytes()).hexdigest()[:12]
        self.map_matches = (want is None or self.map_sha1 == str(want)[:12])
        if not self.map_matches:
            msg = (f"tag map CHANGED since the run: {self.tag_map_path} is "
                   f"sha1 {self.map_sha1}, the run flew {str(want)[:12]}. The "
                   f"picture would show a pool that was never flown.")
            if strict:
                raise ValueError(msg + "  Pass --allow-map-mismatch to draw it "
                                       "anyway (it is stamped on every frame).")
            print(f"WARNING: {msg}", file=sys.stderr)

    # ------------------------------------------------------- datum -> map
    def to_map(self, p_datum) -> np.ndarray:
        """Datum-frame position -> tag-map frame. The inverse of _datumize."""
        p = np.asarray(p_datum, float)
        return self.p0 + (self.Rz_inv @ p.reshape(-1, 3).T).T.reshape(p.shape)

    def yaw_map(self, yaw_datum: float) -> float:
        return float(yaw_datum) + self.yaw0

    # ------------------------------------------------------------ the pose
    def _load_csv(self) -> None:
        rows = list(csv.DictReader(open(self.csv_path)))
        if not rows:
            raise ValueError(f"{self.csv_path} has no rows")

        def col(name, default=np.nan):
            out = np.full(len(rows), default, float)
            for i, r in enumerate(rows):
                v = r.get(name, "")
                if v not in ("", "nan", "None"):
                    try:
                        out[i] = float(v)
                    except ValueError:
                        pass
            return out

        self.t = col("t")
        # THE CSV IS FLU, THE PLANS ARE NED — both in the datum frame.
        # CSV_HEADER says so ("world FLU in the datum frame, the SAME
        # convention as px/py/pz") and this run proves it: at t 294.83 the CSV
        # reads (0.34335, +0.47377, +0.11847) yaw -41.727 deg while the plan
        # anchored one tick later at eta (0.3318, -0.4782, -0.1194) yaw
        # +41.43 deg — y, z and yaw negated, residual 4.5 mm / 1.0 mm / 0.3 deg
        # [측정 2026-09-06 on this run]. Without the flip the vehicle draws
        # BELOW the tag mat and half a metre from its own reference.
        self.p_datum = np.stack([col("px"), -col("py"), -col("pz")], axis=1)
        self.yaw = -np.radians(col("yaw_deg"))
        self.pitch_deg, self.roll_deg = col("pitch_deg"), col("roll_deg")
        self.n_tags, self.tag_age = col("n_tags", 0.0), col("tag_age_s", 0.0)
        self.pnp_rms = col("pnp_rms_px", 0.0)
        self.z_src = [r.get("z_src", "") for r in rows]
        # WHICH PLANS THIS ENGAGEMENT CONSUMED, from its own CSV column. The
        # strongest available statement of ownership: policy_plan.csv and
        # plans.jsonl are shared across engagements, this file is not.
        self.csv_plan_ids = {v for v in (r.get("plan_id", "") for r in rows)
                             if v not in ("", "0", None)}
        self.bridge_s = col("bridge_s", 0.0)
        self.p_map = self.to_map(self.p_datum)
        self.yaw_map_all = self.yaw + self.yaw0

    # ----------------------------------------------------------- the plans
    def _load_plan_records(self) -> None:
        """plans.jsonl, raw and UNFILTERED — the epoch is not known yet."""
        self._plan_recs = []
        self.jsonl = {}
        jl = self.dir / "plans.jsonl"
        if not jl.exists():
            return
        for line in open(jl):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            a, b = d.get("obs_t_mono"), d.get("obs_t")
            self._plan_recs.append({
                "id": str(d.get("plan_id")), "epoch": d.get("epoch"),
                "mono": a if isinstance(a, (int, float)) else None,
                "rel": b if isinstance(b, (int, float)) else None, "rec": d})

    def _load_plans(self) -> None:
        """policy_plan.csv is the reference the run actually drew: one row per
        KNOT, already through the filter, with the status that decided it.

        Filtered to THIS engagement. The plan ids of the chosen epoch come from
        plans.jsonl; the CSV's own `plan_id` column is the fallback, and it is
        the stronger statement of the two — it is what the controller consumed.
        """
        self.plans = []
        self.plan_t = np.zeros(0)
        keep = {r["id"] for r in self._plan_recs if r["epoch"] == self.epoch}
        keep |= self.csv_plan_ids
        self.jsonl = {r["id"]: r["rec"] for r in self._plan_recs
                      if not keep or r["id"] in keep}

        pp = self.dir / "policy_plan.csv"
        if pp.exists():
            by_id: dict[str, dict] = {}
            for r in csv.DictReader(open(pp)):
                pid = r["plan_id"]
                if keep and pid not in keep:
                    continue                # another engagement's reference
                p = by_id.setdefault(pid, {"plan_id": pid, "status": r["status"],
                                           "follower": r.get("follower", ""),
                                           "t_rel": float(r["t_rel"]), "knots": [],
                                           "t_knot": [], "yaw": [], "rp": [],
                                           "reason": r.get("reason", "")})
                p["knots"].append([float(r["x_ned"]), float(r["y_ned"]),
                                   float(r["z_ned"])])
                p["t_knot"].append(float(r["t_knot"]))
                p["yaw"].append(float(r["yaw_deg"]))
                # schema 16 (2026-09-26): trailing roll_deg,pitch_deg AFTER reason,
                # nan on a plan without an attitude reference; absent before 16.
                # Read by name, never by position; kept for the hover layer only
                # (the hull drawn is yaw-only).
                try:
                    p["rp"].append([float(r.get("roll_deg", "nan") or "nan"),
                                    float(r.get("pitch_deg", "nan") or "nan")])
                except ValueError:
                    p["rp"].append([float("nan"), float("nan")])
                if r.get("reason"):
                    p["reason"] = r["reason"]
            for p in by_id.values():
                p["knots_map"] = self.to_map(np.asarray(p["knots"], float))
            self._add_raw_horizon(by_id)
            self.plans = sorted(by_id.values(), key=lambda p: p["t_rel"])
            self.plan_t = np.array([p["t_rel"] for p in self.plans])

        # The two clocks, from THIS engagement's records only: mixing epochs
        # here is what made the offset spread read 2475 s instead of 0.
        offs = [r["mono"] - r["rel"] for r in self._plan_recs
                if r["epoch"] == self.epoch
                and r["mono"] is not None and r["rel"] is not None]
        if offs:
            self.t_offset = float(np.median(offs))
            self.offset_spread = float(np.max(offs) - np.min(offs))
        elif self.has_depth:
            # No plans.jsonl: fall back to aligning the two spans, and SAY so.
            self.t_offset = float(self.t_depth[0] - self.t[0])
            self.offset_spread = float("nan")
        else:
            self.t_offset = 0.0        # _spine_from_csv sets the real one
            self.offset_spread = 0.0

    def _add_raw_horizon(self, by_id: dict) -> None:
        """The policy's OWN 16 steps, recomputed from ``action_raw``.

        What gets logged as a reference is the plan AFTER ``resample_knots``
        puts it on the 0.2 s grid (spec A5): 16 steps at the training stride
        of 66.7 ms span 1.07 s, which lands on 6 knots — that is why a picture
        of policy_plan.csv shows six dots and not sixteen.

        The 16 are not stored anywhere, so they are recomposed here with the
        SAME function the run used (``policy_frames.compose_plan``) from the
        recorded action, anchor and T_body_tcp. That is checkable rather than
        hopeful, and it was checked: over 383 plans of the 2026-09-06 run the
        recomposed 6 knots match the recorded ones to **0.000 um**, so the 16
        beside them are the run's own arithmetic, not a re-derivation of it.

        Reads ALL record generations: the 10-column ``action_raw`` of the
        pose10d runs (meta ``schema_version`` <= 11, [pos(3), rot6d(6),
        width]), the 5-column one of the pos_yaw_width runs (schema 12,
        [dx, dy, dz, dyaw, width], 2026-09-07) and the 7-column one of the
        pos_rpy_width runs (schema 16, [dx, dy, dz, dyaw, droll, dpitch,
        width], 2026-09-26). The dispatch is the WIDTH of the action array
        — ``compose_plan`` branches on it — never the meta version, so a
        mixed plans.jsonl and a run whose meta predates the key both render;
        the anchor / T_body_tcp / width arguments are the same in every
        generation (``dropped_rp_deg`` is 0 by construction on the 5-column
        path and is not needed here). A 7-column record is recomposed the
        way the run flew it: ``track_rp`` = meta ``trajectory.attitude_track``
        (the record's own ``rp_tracked`` when present) with the run's
        ``policy.config.rp_max_deg`` clip — the two sub-modes place the body
        knots differently at a tilted anchor, so the flag matters for the
        picture. The hull drawn on the page stays yaw-only either way (no
        attitude is implied that the panel never showed).
        """
        pol = self.meta.get("policy") or {}
        tcp = (pol.get("tcp") or {}).get("T_body_tcp")
        cfg = pol.get("config") or {}
        w_open = cfg.get("gripper_width_open_m")
        w_closed = cfg.get("gripper_width_closed_m")
        if tcp is None or w_open is None or w_closed is None or not self.jsonl:
            return
        try:
            from rov_gui.control.policy_frames import compose_plan
        except Exception:                                        # noqa: BLE001
            return
        T_bt = np.asarray(tcp, float)
        track_rp = bool((self.meta.get("trajectory") or {}).get("attitude_track", False))
        rp_max_deg = cfg.get("rp_max_deg")
        n_ok = 0
        for pid, plan in by_id.items():
            rec = self.jsonl.get(str(pid))
            if not rec or "action_raw" not in rec:
                continue
            try:
                a = np.asarray(rec["action_raw"], float)
                anchor = np.asarray(rec["anchor_pose_used"], float)
                kw = {}
                if a.shape[-1] == 7:              # pos_rpy_width: as the run flew it
                    kw = dict(track_rp=bool(rec.get("rp_tracked", track_rp)),
                              rp_max_rad=(math.radians(float(rp_max_deg))
                                          if rp_max_deg is not None else None))
                _msg, info = compose_plan(
                    a, anchor, anchor[3:5], T_bt, float(rec["obs_dt_s"]),
                    float(rec["dt"]), float(rec["t0"]), int(pid),
                    float(rec["obs_t_rel"]), float(w_open), float(w_closed), **kw)
                raw = np.asarray(info["p_body_raw"], float)      # (3, K)
                plan["raw_map"] = self.to_map(raw.T)
                plan["n_raw"] = int(raw.shape[1])
                plan["obs_dt"] = float(rec["obs_dt_s"])
                n_ok += 1
            except Exception:                                    # noqa: BLE001
                continue                # one bad record must not cost the video
        self.n_raw_ok = n_ok

    # ----------------------------------------------------------- the depth
    def _load_depth_index(self) -> None:
        """The recorded depth frames, when there are any.

        A run flown without ``--record-depth`` has no depth on disk at all —
        not the observation, not the millimetre map, not the stereo pairs. That
        costs the right-hand panel and NOTHING ELSE: the pool, the vehicle and
        the policy's reference are in the CSV and the plans, so the scene is
        still worth drawing and the timeline is then taken from the pose log
        instead (see :meth:`_spine_from_csv`).
        """
        self.depth_rows = []
        self.depth_meta = {}
        self.t_depth = np.zeros(0)
        self.has_depth = False
        self.no_depth_why = ""
        idx = self.dir / "policy_obs" / "index.csv"
        if not idx.exists():
            self.no_depth_why = ("this run was flown without --record-depth, "
                                 "so no depth reached the disk")
            return
        self.depth_rows = list(csv.DictReader(open(idx)))
        if not self.depth_rows:
            self.no_depth_why = f"{idx} has no rows"
            return
        try:
            self.depth_meta = json.loads(
                (self.dir / "policy_obs" / "meta.json").read_text())
        except Exception:                                        # noqa: BLE001
            self.depth_meta = {}
        self.t_depth = np.array([float(r["t_capture"]) for r in self.depth_rows])
        self.has_depth = True

    def _clip_spine_to_engagement(self) -> None:
        """Keep only the frames this engagement can be drawn against.

        A frame captured during ANOTHER engagement has no pose in this CSV, and
        ``pose_at`` would clamp it to the nearest edge row — a frozen vehicle
        under a moving depth picture, which reads as a tracking failure rather
        than as a frame that does not belong here.
        """
        w = self._plan_window(self.epoch)
        if w is None or len(self.metas) < 2:
            return
        keep = (self.t_depth >= w[0] - 1.0) & (self.t_depth <= w[1] + 1.0)
        n_drop = int((~keep).sum())
        if not n_drop:
            return
        total = len(self.depth_rows)
        self.depth_rows = [r for r, k in zip(self.depth_rows, keep) if k]
        self.t_depth = self.t_depth[keep]
        self.clipped = n_drop
        print(f"        dropped {n_drop} of {total} recorded frames: captured "
              f"during another engagement, no pose in {self.csv_path.name}")

    def _spine_from_csv(self, hz: float = 5.0) -> None:
        """No depth: make the timeline out of the pose log at ~5 Hz.

        5 Hz because that is what a recorded run's depth cadence is, so a page
        made either way scrubs at the same speed and --every means the same
        thing. The rows are synthesised in the depth-row SHAPE so nothing
        downstream has to know which kind of run it is; an empty `depth_png`
        is what says there is no picture for this frame.
        """
        if len(self.t) < 2:
            raise ValueError(
                f"{self.csv_path.name} has {len(self.t)} pose rows — there is "
                f"no run here to draw. (A folder holding only a stream "
                f"recorder's companion CSV looks like this.)")
        step = max(1, int(round((1.0 / hz) / max(np.median(np.diff(self.t)),
                                                 1e-3))))
        idx = list(range(0, len(self.t), step))
        self.depth_rows = [{"seq": str(k), "t_capture": f"{self.t[i]:.6f}",
                            "obs_png": "", "depth_png": ""}
                           for k, i in enumerate(idx)]
        self.t_depth = np.array([self.t[i] for i in idx])
        self.t_offset = 0.0            # the spine IS run time; nothing to shift
        self.offset_spread = 0.0

    # ------------------------------------------------------------- lookups
    def pose_at(self, t_run: float):
        i = int(np.clip(np.searchsorted(self.t, t_run), 0, len(self.t) - 1))
        if i > 0 and abs(self.t[i - 1] - t_run) < abs(self.t[i] - t_run):
            i -= 1
        return i, float(self.t[i] - t_run)

    def plan_at(self, t_run: float):
        """The plan IN FORCE at t: the newest one that had already arrived."""
        if not self.plans:
            return None, 0.0
        j = int(np.searchsorted(self.plan_t, t_run, side="right")) - 1
        if j < 0:
            return None, 0.0
        return self.plans[j], float(t_run - self.plan_t[j])


# ================================================================== drawing
def draw_grid(img, view, bounds, step=0.5):
    """A 0.5 m grid on the mat plane (z = 0), the plane the tags lie in."""
    (x0, x1), (y0, y1) = bounds
    xs = np.arange(math.floor(x0 / step) * step, x1 + step, step)
    ys = np.arange(math.floor(y0 / step) * step, y1 + step, step)
    for x in xs:
        p, _ = view(np.array([[x, y0, 0.0], [x, y1, 0.0]]))
        cv2.line(img, tuple(p[0]), tuple(p[1]), C_GRID, 1, cv2.LINE_AA)
    for y in ys:
        p, _ = view(np.array([[x0, y, 0.0], [x1, y, 0.0]]))
        cv2.line(img, tuple(p[0]), tuple(p[1]), C_GRID, 1, cv2.LINE_AA)


def tag_quads(tags: TagMap, size: float):
    """Every tag INSTANCE as a world-frame square, in the tag's own plane.

    Instances, not ids: this mat reuses 12 ids at a second physical place
    (control/tagnav.py), and drawing one of the two would leave a hole in the
    picture exactly where the localiser had to disambiguate.
    """
    h = size / 2.0
    local = np.array([[-h, -h, 0.0], [h, -h, 0.0], [h, h, 0.0], [-h, h, 0.0]])
    out = []
    for tag_id, insts in sorted(tags.instances.items()):
        for R, t in insts:
            out.append((int(tag_id), (R @ local.T).T + np.asarray(t, float)))
    return out


def draw_tags(img, view, quads, anchor_id=None, ids=False):
    order = []
    for tag_id, world in quads:
        px, z = view(world)
        order.append((float(np.mean(z)), tag_id, px))
    for _, tag_id, px in sorted(order, key=lambda a: -a[0]):   # far first
        cv2.fillConvexPoly(img, px, C_TAG, cv2.LINE_AA)
        edge = C_ANCHOR if (anchor_id is not None and tag_id == anchor_id) \
            else C_TAG_EDGE
        cv2.polylines(img, [px], True, edge, 1, cv2.LINE_AA)
        if ids:
            c = px.mean(axis=0).astype(int)
            text(img, [str(tag_id)], org=(int(c[0]) - 6, int(c[1]) + 3),
                 scale=0.3, colour=(150, 150, 150))


def draw_rov(img, view, p_map, yaw, jaw_frac=0.5):
    """The vehicle from its own tested geometry, YAW ONLY (as the panel draws).

    Painter's algorithm over the part faces, shaded by |dot(normal, view)| so a
    box reads as a box. It can only be wrong where two parts interpenetrate,
    and these do not.
    """
    faces = []
    for part in rov_parts(jaw_open_frac=jaw_frac):
        base = C_ROV.get(part["kind"], (180, 180, 180))
        world = np.array([body_to_map(v, p_map, yaw) for v in part["verts"]])
        px, z = view(world)
        for idx, normal in part["faces"]:
            n_map = np.array(body_to_map(normal, (0.0, 0.0, 0.0), yaw))
            n = n_map / max(np.linalg.norm(n_map), 1e-9)
            shade = 0.45 + 0.55 * abs(float(np.dot(n, view.R[2])))
            faces.append((float(np.mean(z[list(idx)])), px[list(idx)],
                          tuple(int(min(255, c * shade)) for c in base)))
    for _, poly, col in sorted(faces, key=lambda a: -a[0]):
        cv2.fillConvexPoly(img, poly, col, cv2.LINE_AA)


def draw_plan(img, view, plan, colour):
    raw = plan.get("raw_map")
    if raw is not None and len(raw) > 1:
        # The policy's own 16 steps at the training stride, under the 6 knots
        # the controller was actually given. Thin, because the thick line is
        # the one that flew.
        pr, _ = view(raw)
        for i in range(len(pr) - 1):
            cv2.line(img, tuple(pr[i]), tuple(pr[i + 1]), colour, 1, cv2.LINE_AA)
        for q in pr:
            cv2.circle(img, tuple(q), 1, colour, -1, cv2.LINE_AA)
    k = plan["knots_map"]
    px, _ = view(k)
    for i in range(len(px) - 1):
        cv2.line(img, tuple(px[i]), tuple(px[i + 1]), colour, 3, cv2.LINE_AA)
    for i, p in enumerate(px):
        cv2.circle(img, tuple(p), 4 if i == 0 else 3, colour,
                   -1 if i else 2, cv2.LINE_AA)
    # A dropped vertical to the mat plane on the last knot: without it, "where
    # the reference ends" is unreadable in a projection.
    end = k[-1].copy()
    foot = end.copy()
    foot[2] = 0.0
    p2, _ = view(np.array([end, foot]))
    cv2.line(img, tuple(p2[0]), tuple(p2[1]), colour, 1, cv2.LINE_AA)


def draw_inset(img, run, i_pose, plan, box_m=1.0, size=300, pad=12):
    """A top-down zoom on the vehicle, because the two scales differ by 30x.

    The pool is 1.7 x 4.1 m and one diffusion-policy plan is about 0.1 m long
    [측정: this run, v_max 0.15 m/s over ~1.0 s of knots], so a view fitted to
    the map renders the reference as a few pixels — the thing the video exists
    to show. Rather than zooming the main view (and losing the pool), the
    reference gets its own panel at its own scale, with the grid stated.

    Orthographic and top-down on purpose: the plan's z is a separate readout
    below, because a perspective inset would let a knot's height read as a
    horizontal offset, which is the one confusion this panel must not add.
    """
    w = img.shape[1]
    x0, y0 = w - size - pad, pad + 76
    # Drawn into its OWN array and pasted: a knot outside the box would
    # otherwise be drawn at a clipped-looking pixel somewhere over the pool
    # behind it, which reads as a second reference that does not exist.
    panel = np.zeros((size, size, 3), np.uint8)
    panel[:] = (18, 18, 18)

    # Centred on the vehicle's BODY ORIGIN, because that is where the logged
    # reference lives: the policy plans at the TCP (the jaw) and
    # control/policy_frames.py subtracts the lever arm before the plan reaches
    # the controller, so policy_plan.csv holds a BODY-ORIGIN trajectory. The
    # jaw is drawn as a dot 0.50 m ahead so the two are never confused.
    yaw_c = float(run.yaw_map_all[i_pose])
    c = run.p_map[i_pose][:2]
    scale = (size - 24) / box_m

    def px(xy):
        # map x is NORTH and points UP on this inset; map y is EAST -> right.
        return (int(size / 2 + (float(xy[1]) - c[1]) * scale),
                int(size / 2 - (float(xy[0]) - c[0]) * scale))

    for tag_id, world in run.quads:
        w = np.asarray(world, float)
        if np.max(np.abs(w[:, :2].mean(axis=0) - c)) > box_m:
            continue                      # outside the box; skip the work
        pts = np.array([px(q) for q in w], np.int32)
        cv2.fillConvexPoly(panel, pts, C_TAG, cv2.LINE_AA)
        cv2.polylines(panel, [pts], True,
                      C_ANCHOR if tag_id == run.tags.anchor_id else C_TAG_EDGE,
                      1, cv2.LINE_AA)
        side = float(np.linalg.norm(pts[1] - pts[0]))
        if side >= 16:
            mid = pts.mean(axis=0).astype(int)
            text(panel, [str(tag_id)], org=(int(mid[0]) - 7, int(mid[1]) + 4),
                 scale=0.34, colour=(150, 150, 150))

    step = 0.1
    n = int(box_m / step / 2) + 1
    for k in range(-n, n + 1):
        gx, gy = px((c[0] + k * step, c[1] - box_m)), px((c[0] + k * step, c[1] + box_m))
        cv2.line(panel, gx, gy, (34, 34, 34), 1)
        gx, gy = px((c[0] - box_m, c[1] + k * step)), px((c[0] + box_m, c[1] + k * step))
        cv2.line(panel, gx, gy, (34, 34, 34), 1)

    hull = np.array([px(q) for q in rov_footprint(run.p_map[i_pose], yaw_c)],
                    np.int32)
    cv2.polylines(panel, [hull], True, C_ROV["hull"], 2, cv2.LINE_AA)
    cv2.line(panel, tuple(hull[0]), tuple(hull[1]), (255, 255, 255), 2, cv2.LINE_AA)
    tcp = body_to_map(run.tcp_body, run.p_map[i_pose], yaw_c)
    cv2.circle(panel, px(tcp[:2]), 4, C_ROV["jaw"], -1, cv2.LINE_AA)

    label = "no plan in force"
    if plan is not None:
        colour = C_PLAN.get(plan["status"], (200, 200, 200))
        pts = [px(q[:2]) for q in plan["knots_map"]]
        for a, b in zip(pts, pts[1:]):
            cv2.line(panel, a, b, colour, 2, cv2.LINE_AA)
        for j, q in enumerate(pts):
            cv2.circle(panel, q, 4 if j == 0 else 3, colour, -1 if j else 2,
                       cv2.LINE_AA)
        dz = plan["knots_map"][-1][2] - run.p_map[i_pose][2]
        span = float(np.linalg.norm(plan["knots_map"][-1][:2]
                                    - plan["knots_map"][0][:2]))
        label = (f"plan {plan['plan_id']} {plan['status']}   "
                 f"{span * 100:.0f} cm across, dz {dz * 100:+.0f} cm")
    img[y0:y0 + size, x0:x0 + size] = panel
    cv2.rectangle(img, (x0, y0), (x0 + size, y0 + size), (80, 80, 80), 1)
    # Two short lines rather than one long one: at 0.36 scale the inset is
    # ~58 characters wide and a single line was being cut off mid-word.
    text(img, [f"top-down zoom  {box_m:.1f} m box, 0.1 m grid"],
         org=(x0 + 6, y0 - 6), scale=0.36, colour=(160, 160, 160))
    tcp_tag = "run meta" if run.tcp_source.startswith("meta") else "live"
    text(img, [label, f"body origin centred; amber dot = TCP (jaw, {tcp_tag})"],
         org=(x0 + 6, y0 + size + 14), scale=0.38, gap=14,
         colour=(200, 200, 200))


class VideoOut:
    """h264 through PyAV where it exists, mp4v through cv2 where it does not.

    The station's own env (``rovgui-pose``, what ``./c3 gui`` runs) has no
    PyAV, and making the operator switch interpreters to render a picture of
    their own run is how a tool stops being used. The fallback is honest about
    what it cost: mp4v is a bigger file and some browsers will not play it, so
    it says so once instead of quietly writing something the caller did not
    ask for.
    """

    def __init__(self, path: Path, size, fps: float, crf: int):
        self.path, self.size = Path(path), (int(size[0]), int(size[1]))
        self.fps = max(1.0, float(fps))
        self.backend = ""
        self._container = self._stream = self._vw = None
        try:
            import av
            self._av = av
            self._container = av.open(str(self.path), mode="w")
            st = self._container.add_stream(
                "libx264", rate=Fraction(int(round(self.fps)), 1))
            st.width, st.height = self.size
            st.pix_fmt = "yuv420p"
            st.options = {"crf": str(int(crf)), "preset": "veryfast"}
            self._stream = st
            self.backend = "PyAV/libx264"
        except Exception:                                        # noqa: BLE001
            self._container = self._stream = None
            vw = cv2.VideoWriter(str(self.path),
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 self.fps, self.size)
            if not vw.isOpened():
                raise RuntimeError(
                    f"no video writer: PyAV is absent and cv2 could not open "
                    f"{self.path} for mp4v either")
            self._vw = vw
            self.backend = "cv2/mp4v"
            print("NOTE: PyAV is not in this interpreter, so the video is "
                  "written as mp4v rather than h264 — bigger, and some "
                  "browsers will not play it. The fstereo env has PyAV.",
                  file=sys.stderr)

    def write(self, frame) -> None:
        if self._stream is not None:
            for packet in self._stream.encode(
                    self._av.VideoFrame.from_ndarray(frame, format="bgr24")):
                self._container.mux(packet)
        else:
            self._vw.write(frame)

    def close(self) -> None:
        if self._stream is not None:
            try:
                for packet in self._stream.encode():
                    self._container.mux(packet)
            finally:
                self._container.close()
        elif self._vw is not None:
            self._vw.release()


# =============================================================== the frames
def scene_panel(run: Run, view, i_depth: int, t_run: float, size=SCENE):
    """The pool, the vehicle NOW, and the reference in force. No path history:
    a trail of where the vehicle has been competes with the one line this
    picture is about, and the CSV already holds it for anyone who wants it."""
    img = np.zeros((size[1], size[0], 3), np.uint8)
    img[:] = BG
    draw_grid(img, view, run.grid_bounds)
    draw_tags(img, view, run.quads, run.tags.anchor_id, run.draw_tag_ids)

    i, dt_pose = run.pose_at(t_run)
    plan, age = run.plan_at(t_run)
    if plan is not None:
        draw_plan(img, view, plan, C_PLAN.get(plan["status"], (200, 200, 200)))
    draw_rov(img, view, run.p_map[i], float(run.yaw_map_all[i]))
    draw_inset(img, run, i, plan, box_m=run.inset_box_m)

    p = run.p_map[i]
    lines = [
        f"{run.dir.name}   frame {i_depth + 1}/{len(run.depth_rows)}   "
        f"t {t_run:7.2f} s",
        f"ROV map  x {p[0]:+.3f}  y {p[1]:+.3f}  z {p[2]:+.3f} m   "
        f"yaw {math.degrees(run.yaw_map_all[i]):+6.1f} deg "
        f"(pitch {run.pitch_deg[i]:+.1f}  roll {run.roll_deg[i]:+.1f}, "
        f"drawn yaw-only)",
        f"tags {run.n_tags[i]:.0f}  age {run.tag_age[i]:.2f} s  "
        f"rms {run.pnp_rms[i]:.2f} px  z_src {run.z_src[i]}"
        + (f"  BRIDGED {run.bridge_s[i]:.1f} s" if run.bridge_s[i] > 0 else "")
        + (f"   pose is {abs(dt_pose) * 1e3:.0f} ms from this frame"
           if abs(dt_pose) > 0.1 else ""),
    ]
    if plan is not None:
        raw_n = plan.get("n_raw")
        lines.append(
            f"DP plan {plan['plan_id']}  {plan['status'].upper()}  "
            + (f"{raw_n} raw steps @ {plan.get('obs_dt', 0) * 1e3:.1f} ms -> "
               if raw_n else "")
            + f"{len(plan['knots'])} knots @ "
            f"{plan['t_knot'][1] - plan['t_knot'][0]:.1f} s over "
            f"{plan['t_knot'][-1] - plan['t_knot'][0]:.1f} s   "
            f"issued {age:.2f} s ago   follower {plan['follower']}")
    else:
        lines.append("DP plan: none in force")
    text(img, lines, org=(10, 16), scale=0.42, gap=16)

    legend = [
        f"tag map {run.tag_map_path.name} sha1 {run.map_sha1}"
        f"{'' if run.map_matches else '  != THE RUN (drawn anyway)'}   "
        f"{len(run.quads)} instances @ {run.tag_size * 1000:.0f} mm   "
        f"anchor {run.tags.anchor_id}",
        "grid 0.5 m on the mat plane (z=0)   map frame = anchor tag, NED "
        "(+z DOWN)   green = vehicle NOW (no path history), coloured line = "
        "DP reference",
        "the reference is the BODY-ORIGIN trajectory the controller was given "
        "(the policy plans at the TCP; the lever arm is removed upstream)",
    ]
    text(img, legend, org=(10, size[1] - 40), scale=0.38, gap=14,
         colour=(150, 150, 150))
    return img


def depth_side(run: Run, i: int, domain: str, cmap: str, size=SIDE):
    row = run.depth_rows[i]
    img = np.zeros((size[1], size[0], 3), np.uint8)
    img[:] = BG
    if not row.get("depth_png"):
        text(img, ["no depth in this run", run.no_depth_why or "",
                   "(the scene on the left is unaffected)"],
             org=(14, size[1] // 2 - 16), scale=0.44, gap=20,
             colour=(150, 150, 150))
        return img
    path = run.dir / "policy_obs" / row["depth_png"]
    mm = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mm is None:
        text(img, [f"missing {row['depth_png']}"], org=(10, 40), scale=0.5,
             colour=(120, 160, 255))
        return img
    t, valid = mm_to_t_valid(mm, domain, Z_NEAR_M, Z_FAR_M)
    panel = depth_panel(t, valid, domain=domain, cmap=cmap, z_near=Z_NEAR_M,
                        z_far=Z_FAR_M, lines=run.depth_header(i))
    y0 = (size[1] - PANEL_H) // 2
    img[y0:y0 + PANEL_H, :PANEL_W] = panel
    v = float((mm > 0).mean()) * 100.0
    d = mm[mm > 0]
    text(img, [
        f"valid {v:.1f}%   nearest {d.min() / 1000.0:.2f} m   "
        f"median {np.median(d) / 1000.0:.2f} m" if d.size else "no measurement",
        f"seq {row['seq']}   t_capture {float(row['t_capture']):.3f} (mono)",
    ], org=(10, y0 + PANEL_H + 20), scale=0.40, gap=16, colour=(170, 170, 170))
    return img


# ==================================================================== render
def render(run: Run, out: Path, *, every: int, limit: int | None, speed: float,
           domain: str, cmap: str, crf: int, azimuth: float, elevation: float,
           orbit: float) -> Path:
    idx = list(range(0, len(run.depth_rows), max(1, every)))
    if limit:
        idx = idx[:limit]
    dt = float(np.median(np.diff(run.t_depth))) if len(run.t_depth) > 1 else 0.2
    fps = max(1.0, (1.0 / max(dt, 1e-3)) / max(1, every) * max(speed, 0.01))

    pts = [q for _, q in run.quads]
    pts.append(run.p_map)
    for p in run.plans:
        pts.append(p["knots_map"])
    allp = np.concatenate([np.asarray(a, float).reshape(-1, 3) for a in pts])
    run.grid_bounds = ((float(allp[:, 0].min()) - 0.3, float(allp[:, 0].max()) + 0.3),
                       (float(allp[:, 1].min()) - 0.3, float(allp[:, 1].max()) + 0.3))
    corners = np.array([[x, y, 0.0] for x in run.grid_bounds[0]
                        for y in run.grid_bounds[1]])
    fit_pts = np.concatenate([allp, corners])
    # The encoder is chosen HERE, not at module scope: everything above this
    # function is frame algebra with no encoder in it, and the env that has
    # pytest has no PyAV. Nothing that can be unit-tested should need one.
    out_v = VideoOut(out, (SCENE[0] + SIDE[0], SCENE[1]), fps, crf)
    view = View.fit(fit_pts, SCENE, azimuth=azimuth, elevation=elevation)
    try:
        for n, i in enumerate(idx):
            t_run = float(run.t_depth[i] - run.t_offset)
            if orbit:
                view = View.fit(fit_pts, SCENE, azimuth=azimuth + orbit * t_run,
                                elevation=elevation)
            left = scene_panel(run, view, i, t_run)
            right = depth_side(run, i, domain, cmap)
            frame = np.hstack([left, right])
            frame[:, SCENE[0]] = (70, 70, 70)
            out_v.write(frame)
            if n % 200 == 0:
                print(f"  {n}/{len(idx)}  t {t_run:7.2f} s", flush=True)
    finally:
        out_v.close()
    return out


def describe_siblings(run_dir: Path) -> str:
    """What ELSE is in this date folder, when the run asked for is empty.

    A run folder under data/ can be a flight, a second engagement of one, or
    the companion CSV a stream recording leaves behind (0 rows, no plans) —
    and they are named alike. Refusing without saying which neighbours are
    flights makes the operator open them one by one.
    """
    d0 = Path(run_dir)
    out = []
    # A run folder carries its OWN date (0907_133358) INSIDE a date folder, so
    # the two can disagree by a day and the path still looks right. Say where
    # this exact name does live before listing anything else.
    if not d0.is_dir():
        # ...and the root can be wrong as well as the date
        # (data/20260906/0907_164659 is both), so search
        # every date folder under the root before giving up on the name.
        root = d0.parent.parent.parent
        found = sorted(d0.parent.parent.glob(f"*/{d0.name}")) if \
            d0.parent.parent.is_dir() else []
        if not found and root.is_dir():
            found = sorted(root.glob(f"*/*/{d0.name}"))
        if found:
            return ("\n  the same run name exists here:\n"
                    + "\n".join(f"    {c}" for c in found))
    if not d0.parent.is_dir():
        return ""
    for d in sorted(p for p in d0.parent.iterdir() if p.is_dir()):
        rows, has_depth = 0, (d / "policy_obs" / "index.csv").exists()
        for m in d.glob("*.meta.json"):
            n = 0
            try:
                n = int((json.loads(m.read_text()).get("run") or {})
                        .get("rows") or 0)
            except Exception:                                    # noqa: BLE001
                n = 0
            if not n:
                # Older metas, and any the writer did not finish, have no
                # run.rows — count the file rather than call the run empty.
                c = m.with_suffix("").with_suffix(".csv")
                try:
                    n = max(0, sum(1 for _ in c.open()) - 1)
                except OSError:
                    n = 0
            rows += n
        if rows:
            out.append(f"    {d.name}   {rows} pose rows"
                       + ("   + depth" if has_depth else "   (no depth)"))
    return ("\n  runs with something in them, beside it:\n" + "\n".join(out)) \
        if out else ""


# ======================================================================= cli
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="the run folder")
    p.add_argument("--out", type=Path, default=None,
                   help="output mp4 (default: <run>/scene.mp4)")
    p.add_argument("--tag-map", type=Path, default=None,
                   help="override the map path the run recorded")
    p.add_argument("--allow-map-mismatch", action="store_true",
                   help="draw even if the map's sha1 differs from the run's")
    p.add_argument("--engagement", default=None, metavar="NAME",
                   help="which engagement in the folder, as a fragment of its "
                        "csv name (e.g. 191935). A folder can hold several, "
                        "and plans.jsonl / policy_plan.csv are SHARED between "
                        "them; without this the one whose plans were stamped "
                        "inside the recorded depth window is chosen.")
    p.add_argument("--every", type=int, default=1,
                   help="use every Nth recorded depth frame (default: %(default)s)")
    p.add_argument("--limit", type=int, default=None, help="stop after N frames")
    p.add_argument("--speed", type=float, default=1.0,
                   help="playback speed (default: %(default)s = real time; the "
                        "clock is on every frame either way)")
    p.add_argument("--domain", choices=("obs", "metric"), default="obs",
                   help="depth colour spacing; match whatever it is compared with")
    p.add_argument("--cmap", default="turbo")
    p.add_argument("--azimuth", type=float, default=-60.0)
    p.add_argument("--elevation", type=float, default=32.0)
    p.add_argument("--orbit", type=float, default=0.0, metavar="DEG_PER_S",
                   help="rotate the viewpoint while playing; helps depth "
                        "perception at the cost of a moving frame")
    p.add_argument("--tag-ids", action=argparse.BooleanOptionalAction,
                   default=True, help="label every tag with its id "
                                      "(default: %(default)s)")
    p.add_argument("--inset-box", type=float, default=2.0, metavar="M",
                   help="width of the top-down zoom on the vehicle and its "
                        "reference (default: %(default)s m). The plans on the "
                        "2026-09-06 run are ~0.1 m long, so this is what makes "
                        "them legible next to a 4 m pool. Centred on the "
                        "vehicle, with the tags that fall inside it, so the "
                        "zoom still says where in the pool this is.")
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    try:
        run = Run(a.run, a.tag_map, strict_map=not a.allow_map_mismatch,
                  engagement=a.engagement)
    except (FileNotFoundError, ValueError) as e:
        # An empty or wrong folder is an ordinary mistake, not a crash: say
        # what is wrong and what is next door.
        print(f"{a.run}: {e}{describe_siblings(a.run)}", file=sys.stderr)
        return 2
    run.draw_tag_ids = bool(a.tag_ids)
    run.inset_box_m = float(a.inset_box)
    run.quads = tag_quads(run.tags, run.tag_size)

    fs = run.meta.get("fstereo", {})
    grid = (run.depth_meta.get("builder") or {}).get("grid") or {}
    fx = (grid.get("P1") or [[float("nan")]])[0][0]
    ck = Path(str(fs.get("ckpt", "?")))

    def header(i):
        row = run.depth_rows[i]
        if not row.get("depth_png"):
            return ["no depth recorded", run.no_depth_why or ""]
        return [f"FoundationStereo depth   policy_obs/{row['depth_png']}   "
                f"mm, 0 = invalid   seq {row['seq']}",
                f"iters {fs.get('iters', '?')}  scale {fs.get('scale', '?')}  "
                f"{ck.parent.name}/{ck.name}  grid "
                f"{(run.depth_meta.get('obs') or {}).get('grid_kind', '?')}  "
                f"fx_rect {fx:.2f} px"]
    run.depth_header = header

    out = a.out or (run.dir / "scene.mp4")
    if out.exists() and not a.overwrite:
        print(f"{out} exists; pass --overwrite", file=sys.stderr)
        return 1
    print(f"run      {run.dir}")
    print(f"tag map  {run.tag_map_path}  sha1 {run.map_sha1}"
          f"{'  (matches the run)' if run.map_matches else '  != THE RUN'}"
          f"   {len(run.quads)} instances, {len(run.tags.instances)} ids")
    print(f"datum    p0 {np.round(run.p0, 3).tolist()}  "
          f"yaw0 {math.degrees(run.yaw0):.2f} deg")
    print(f"engage   {run.meta_path.name}  epoch {run.epoch}")
    print(f"tcp      {np.round(run.tcp_body, 4).tolist()}  <- {run.tcp_source}")
    print(f"pose     {len(run.t)} rows, {run.t[0]:.2f}..{run.t[-1]:.2f} s")
    print(f"plans    {len(run.plans)} in policy_plan.csv"
          f"   clock offset {run.t_offset:.3f} s "
          f"(spread {run.offset_spread * 1e3:.1f} ms)")
    print(f"depth    {len(run.depth_rows)} frames, "
          f"{run.t_depth[0]:.2f}..{run.t_depth[-1]:.2f}"
          + (" (mono)" if run.has_depth else
             f"  — NO DEPTH ({run.no_depth_why}); the timeline comes from the "
             f"pose log and the right panel says so"))
    render(run, out, every=a.every, limit=a.limit, speed=a.speed,
           domain=a.domain, cmap=a.cmap, crf=a.crf, azimuth=a.azimuth,
           elevation=a.elevation, orbit=a.orbit)
    print(f"==> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
