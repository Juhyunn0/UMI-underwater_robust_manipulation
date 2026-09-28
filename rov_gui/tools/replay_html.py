#!/usr/bin/env python3
"""
replay_html.py — bake a recorded run into ONE self-contained HTML page you
can scrub in a browser: the tag floor, the vehicle, and the diffusion
policy's reference at every update.

    ./c3 replay-html data/20260903/0903_183405_observe
    ./c3 replay-html <RUN_DIR> -o /tmp/run.html --stride 2

WHY. `./c3 run-replay` needs a display and the rovgui-pose Qt env. A page
needs neither, so a run can be looked at from a laptop, a phone, or a shell
with no X — which is how the analysis of a pool session actually gets shared.

THE FRAME CHAIN IS NOT REIMPLEMENTED. That is the whole design constraint,
inherited from replay_run.py: this repo has shipped frame and sign errors
into production twice, and a second implementation of datum -> map is exactly
how a third one happens. So every metric quantity in the page is computed HERE
by the REAL ``TrajectoryView`` — the run is fed through ``feed_tick`` /
``feed_plan`` exactly as the replay window feeds it, and what gets written out
is what the widget itself put in ``p_act``, ``yaw_ned`` and
``policy_plans[i]["pts"]``, already in MAP coordinates. The browser therefore
never sees a datum, an FLU axis, or a z sign.

WHAT THE BROWSER DOES DO is the camera: orthographic projection (`_px`), the
eye basis (`_basis3`), the view direction (`_view_f`), and `body_to_map` for
the hull — four small pure functions, ported line for line. Those are the
cosmetic half, and they are also CHECKED: `_projection_probe` renders a
handful of points through the real widget's `_px` at a known camera and ships
the expected pixels with the data, so the page can verify its own projection
against Qt on load and say so in the header. A silent divergence is the one
failure mode this tool could have; it is not allowed to be silent.

WHAT IT READS is what an ordinary run already writes — see replay_run.Run:
mpc_<HHMMSS>.csv, its .meta.json (the datum), policy_plan.csv, and
nav_<HHMMSS>/map.json for the floor.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rov_gui import theme                                        # noqa: E402
from rov_gui.qt import QtWidgets                                 # noqa: E402
from rov_gui.tools.replay_run import Run, feed_plan, feed_tick   # noqa: E402
from rov_gui.widgets import rov_shape                            # noqa: E402
from rov_gui.widgets.trajectory import (POLICY_PLAN_COLOR,       # noqa: E402
                                        POLICY_PLAN_KEEP,
                                        POLICY_PLAN_W,
                                        POLICY_PLAN_W_PREV,
                                        TRAIL_AGE_S,
                                        TrajectoryView, _VIRIDIS)

HERE = Path(__file__).resolve().parent


def _r(v, nd: int = 4):
    """Round for transport. Millimetre-and-better on metres, which is two
    orders finer than anything this plot can resolve, and it halves the file."""
    return None if v is None or not math.isfinite(float(v)) else round(float(v), nd)


# --------------------------------------------------------------------------
def _bake(run: Run) -> dict:
    """The run, in MAP coordinates, as the REAL widget computes them."""
    view = TrajectoryView()
    view.resize(900, 680)
    view.tag_size_m = run.tag_size_m
    if run.map_tags:
        view.set_map_tags(run.map_tags)
    if run.pool:
        view.set_pool(run.pool)

    ticks = []
    for r in run.ticks:
        feed_tick(view, run.datum, r)
        if view.p_act is None:
            continue
        x, y, z = view.p_act
        ticks.append([_r(r["t_traj"] if math.isfinite(r["t_traj"]) else r["t"], 3),
                      _r(x), _r(y), _r(z), _r(math.degrees(view.yaw_ned), 2)])

    plans = []
    for g in run.plans:
        feed_plan(view, g)
        if not view.policy_plans:
            continue
        pl = view.policy_plans[-1]
        plans.append({
            "t": _r(g["t_rel"], 3), "id": int(g["plan_id"]),
            "status": g["status"], "reason": g["reason"],
            "pts": [[_r(a), _r(b), _r(c)] for a, b, c in pl["pts"]],
        })
    return {"ticks": ticks, "plans": plans}


def _body(run=None) -> dict:
    """The vehicle as static body-frame geometry. JS applies `body_to_map`
    per frame, which is the one transform it owns and the one it is checked
    on — the hull's footprint is pinned by test_rov_shape."""
    model = rov_shape.parts()
    tcp, tcp_src = (rov_shape.tcp_body_for_meta(run.meta) if run is not None
                    else (tuple(rov_shape.tcp_body_m()), "live rov_shape"))
    out = []
    for part in model:
        out.append({
            "kind": part["kind"],
            "verts": [[_r(a), _r(b), _r(c)] for a, b, c in part["verts"]],
            "faces": [{"i": list(idx), "n": [_r(n[0]), _r(n[1]), _r(n[2])]}
                      for idx, n in part["faces"]],
        })
    ax = rov_shape.camera_axis_flu()
    cam = rov_shape.CAM_T_FLU_M
    return {
        "parts": out,
        # the RUN's TCP, not today's: the jaw moved on 2026-09-08 and an
        # older run's plans were composed at its own point (tcp_source says)
        "tcp": [_r(v) for v in tcp],
        "tcp_source": tcp_src,
        "cam": [_r(v) for v in cam],
        "cam_tip": [_r(cam[i] + ax[i] * rov_shape.CAM_RAY_M) for i in range(3)],
        # the heading line's two ends (aft, fwd), body frame — the widget's
        # `_draw_rov_heading` (operator request 2026-09-07)
        "heading": [[_r(v) for v in end]
                    for end in rov_shape.heading_line_body()],
        "hull_w": rov_shape.HULL_W_M,
    }


def _projection_probe(w: int = 900, h: int = 680) -> dict:
    """Ground truth for the page's own projection, straight out of Qt.

    Points chosen to exercise all three axes and both signs, at a camera the
    page can reproduce exactly. If the ported `_px` ever drifts from the
    widget's, this is what says so — on the page, in the header, on load.
    """
    view = TrajectoryView()
    view.resize(w, h)
    view.set_pool([(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)])
    view.three_d, view.azimuth_deg, view.elev_deg = True, 25.0, 35.0
    view.zoom, view.z_centre = 2.0, 0.0
    pts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
           (0.0, 0.0, 1.0), (-1.5, 0.7, -0.4), (2.0, -2.0, 0.25)]
    return {
        "w": w, "h": h, "zoom": 2.0, "az": 25.0, "el": 35.0, "z_centre": 0.0,
        "pool": [[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]],
        "pts": [[_r(p[0]), _r(p[1]), _r(p[2])] for p in pts],
        "px": [[round(view._px(*p).x(), 3), round(view._px(*p).y(), 3)]
               for p in pts],
    }


# --------------------------------------------------------------------------
def build(run: Run, out: Path, stride: int = 1) -> Path:
    baked = _bake(run)
    ticks = baked["ticks"][::max(1, stride)]
    meta = run.meta
    hw = meta.get("hardware") or {}
    pol = meta.get("policy") or {}
    data = {
        "run": run.dir.name,
        "run_path": str(run.dir),
        "csv": run.csv_path.name,
        "kind": ((meta.get("trajectory") or {}).get("kind")
                 or (meta.get("mission") or {}).get("shape") or "?"),
        "observe": bool(pol.get("observe")),
        "source": str(meta.get("source", ""))[:400],
        "hz": round(20.0 / max(1, stride), 2),
        "stride": max(1, stride),
        "datum": ([_r(v) for v in run.datum] if run.datum else None),
        "tag_size_m": run.tag_size_m,
        "tags": [[_r(t[0]), _r(t[1]), _r(t[2], 5), int(t[3])]
                 for t in run.map_tags],
        "pool": ([[_r(c[0]), _r(c[1])] for c in run.pool] if run.pool else None),
        "ticks": ticks,
        "plans": baked["plans"],
        "body": _body(run),
        "probe": _projection_probe(),
        "style": {
            "bg": theme.BG, "panel": theme.PANEL, "border": theme.BORDER,
            "text": theme.TEXT, "dim": theme.TEXT_DIM, "faint": theme.TEXT_FAINT,
            "accent": theme.ACCENT, "ok": theme.OK, "warn": theme.WARN,
            "fail": theme.FAIL,
            "plan": POLICY_PLAN_COLOR, "plan_w": POLICY_PLAN_W,
            "plan_w_prev": POLICY_PLAN_W_PREV, "plan_keep": POLICY_PLAN_KEEP,
            "trail_age_s": TRAIL_AGE_S,
            "viridis": [list(c) for c in _VIRIDIS],
        },
    }
    tpl = (HERE / "replay_html_page.html").read_text(encoding="utf-8")
    blob = json.dumps(data, separators=(",", ":"))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tpl.replace("/*__RUN_DATA__*/null", blob), encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="a run folder (data/YYYYMMDD/MMDD_HHMMSS[_kind])")
    ap.add_argument("-o", "--out", default=None, help="output .html")
    ap.add_argument("--csv", default=None, help="which mpc_*.csv")
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every Nth tick (1 = all 20 Hz)")
    a = ap.parse_args(argv)

    # KEPT IN A NAME. Discarding it lets Python collect the QApplication
    # before the first QWidget is built, which fails as "Must construct a
    # QApplication before a QWidget" — with a QApplication on the line above.
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    run = Run(Path(a.run_dir), Path(a.csv) if a.csv else None)
    print(run.describe())
    out = Path(a.out) if a.out else (run.dir / "replay.html")
    p = build(run, out, a.stride)
    print(f"{p}  {p.stat().st_size / 1e6:.1f} MB")
    del app
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
