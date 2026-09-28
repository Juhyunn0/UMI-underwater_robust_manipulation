#!/usr/bin/env python3
"""plot_policy_map.py — 정책 런 하나를 태그맵 위에 **시간을 색으로** 그린다.

    ~/miniforge3/envs/robust/bin/python -m rov_gui.tools.plot_policy_map 0908_112101
    ... --plan-scale 0.25                 # 플랜을 1/4만 (10개면 2~3개)
    ... --object 0.147,0.379              # 물체 위치(MAP 좌표)를 알면
    ... --panels both                     # 지도 + yaw 시계열
    ... --span 3 | --view tags            # 지도 창을 넓게 / 태그맵 전체
    ... --list                            # 이 폴더의 미션 목록만

폴더는 날짜/시각만 주면 된다: ``0908_112101``, ``20260908/0908_112101``,
또는 런 폴더 경로 자체. 한 폴더에 engagement가 여러 개면(플랜 로그는 공유된다,
KNOWN_ISSUES.md) 정책 미션마다 그림을 하나씩 쓴다. ``--mission HHMMSS``로 하나만.

무엇을 그리나
-------------
* **ROV 경로**: 미션 시계로 색을 입힌 산점(옅은 회색 선은 궤적 연결).
* **policy plan-end yaw**: 그 플랜이 1 s 뒤에 원하는 헤딩. 광선의 시작점은 그
  플랜의 앵커(= 관측 시각의 기체 위치)이고 **색은 경로와 같은 시계·같은 컬러맵**이라
  "이 색 점에 있을 때 이 색 광선 쪽을 보고 싶어 했다"로 읽는다.
* **vehicle heading**: 같은 순간의 **실제** 기체 헤딩, 같은 색 **점선**. 실선(가고
  싶은 방향)과 점선(지금 보는 방향)이 벌어진 각이 그 플랜의 dyaw다. 헤딩은 AprilTag
  PnP 해의 yaw 성분 — 플랜 레코드의 ``anchor_pose_used[5]``(정책이 관측한 바로 그
  순간, 플랜 합성에 쓴 값이라 실선과 정확히 같은 기준)를 쓰고, 없으면 ``anchor.yaw``,
  그것도 없으면 CSV ``yaw_deg``를 그 시각에 보간한다(2026-09-11).
  **기본은 자동**: observe 런(meta ``policy.observe``)이면 켜진다 — 제어기가 명령을
  안 내니 헤딩과 요청이 갈라지고 그 차이가 정보다. 추종자가 요청을 그대로 따라가는
  런에서는 점선이 실선 밑에 가려 잉크만 늘어서 꺼진다. ``--heading`` / ``--no-heading``
  으로 강제.
* 태그맵 사각형, 그리고 ``--object``를 주면 물체.

지도 창은 런이 작아도 **최소 2 m 정사각**이라(``--min-span``) 태그맵이 늘 같이 보인다.
더 넓게는 ``--span 3``, 맵 전체는 ``--view tags``(4.1 x 1.7 m). yaw 광선 길이는 창의
0.14배로 따라간다(``--yaw-len``으로 고정).

병진 의도(translation intent) 화살표는 그리지 않는다 — 전진이 지배적이라 화살표가
경로를 가릴 뿐이고, 이 그림의 질문은 "어디를 **보려** 했나"이다.

시계 주의 (2026-09-08에 물린 함정)
----------------------------------
``plans.jsonl``의 ``t_rel``은 CSV의 ``t``가 아니라 ``t_traj``(미션 시작에 0으로
리셋되는 시계)다. 두 시계는 run 0908_112101에서 **1.478 s** 어긋난다
[측정: data/20260908/0908_112101/diag/
verifierC_redo_out.txt]. 이 도구는 ``t_traj > 0.5`` 인 행에서 ``median(t − t_traj)``
를 offset으로 잡아 경로와 플랜을 같은 미션 시계에 올린다. 색이 맞으려면 이게 맞아야
한다.

프레임
------
CSV ``px,py,pz``는 ENGAGE 데이텀의 world **FLU**. NED는 ``(px, −py, −pz)``,
yaw_ned = −``yaw_deg``. MAP(태그맵, NED-like, +z down)은

    x_map = x0 + cos(yaw0)·x_ned − sin(yaw0)·y_ned
    y_map = y0 + sin(yaw0)·x_ned + cos(yaw0)·y_ned

``(x0, y0, yaw0) = meta.hardware.datum_tag_frame`` — ``plot_runs.py:_to_map`` 및
``workers.py`` ``_to_map_xy``와 같은 식. 화면은 GUI/plot_runs와 같게
**위 = +x_map, 오른쪽 = +y_map**. 플랜의 ``raw.yaw``는 NED 데이텀 body yaw라
map yaw = ``yaw0 + raw.yaw``이고, 줄어들면 좌회전(CCW, 위에서 볼 때)이다.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
#: The dated root (rov_gui/runstore.py); a bare `MMDD_HHMMSS` spec is a
#: water run, `..._observe` / `..._landdry` name the other kinds explicitly.
SESSIONS = REPO / "data"


# --------------------------------------------------------------- run lookup
def resolve_run(spec: str) -> Path:
    """``0908_112101`` / ``20260908/0908_112101`` / 경로 → 런 폴더."""
    p = Path(spec).expanduser()
    if p.is_dir():
        return p
    if p.is_file():
        return p.parent
    pats = [spec, f"*/{spec}"] if "/" not in spec else [spec]
    for pat in pats:
        hits = sorted(q for q in SESSIONS.glob(pat) if q.is_dir())
        if hits:
            if len(hits) > 1:
                print(f"[warn] '{spec}' matches {len(hits)} folders; using {hits[-1].name}")
            return hits[-1]
    raise SystemExit(f"[fail] run folder not found for '{spec}' (looked under {SESSIONS})")


def missions(folder: Path, want: str | None):
    """[(csv_path, meta)] — 정책 미션만, 시간순. ``want``는 HHMMSS."""
    out = []
    for c in sorted(folder.glob("mpc_*.csv")):
        mp = c.with_suffix(".meta.json")
        if not mp.exists():
            continue
        meta = json.loads(mp.read_text())
        if want and c.stem.split("_")[-1] != want:
            continue
        out.append((c, meta))
    return out


# ------------------------------------------------------------------ loading
def read_csv(path: Path) -> dict:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    cols = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        try:
            cols[k] = np.array([float(v) if v not in ("", "nan") else np.nan
                                for v in vals])
        except ValueError:
            cols[k] = np.array(vals, dtype=object)
    return cols


def mission_clock(d: dict) -> tuple[np.ndarray, float]:
    """CSV t → 미션 시계(플랜의 t_rel과 같은 시계). (tm, offset)."""
    t = d["t"]
    tt = d.get("t_traj")
    off = 0.0
    if tt is not None:
        m = np.isfinite(tt) & (tt > 0.5) & np.isfinite(t)
        if m.sum() >= 5:
            off = float(np.median(t[m] - tt[m]))
    return t - off, off


def datum_of(meta: dict):
    d = ((meta.get("hardware") or {}).get("datum_tag_frame")) or None
    if not d or d.get("p0") is None or d.get("yaw0_deg") is None:
        raise SystemExit("[fail] meta.hardware.datum_tag_frame 없음 — MAP 프레임으로 "
                         "올릴 수 없다 (plot_runs.py와 같은 규칙: 조용히 아무 데나 "
                         "그리지 않는다)")
    p0 = [float(v) for v in d["p0"]]
    return p0[0], p0[1], math.radians(float(d["yaw0_deg"]))


def to_map(xn, yn, datum):
    x0, y0, yaw0 = datum
    c, s = math.cos(yaw0), math.sin(yaw0)
    return x0 + c * np.asarray(xn) - s * np.asarray(yn), \
           y0 + s * np.asarray(xn) + c * np.asarray(yn)


def load_plans(folder: Path, meta: dict) -> list[dict]:
    """이 미션의 설치된(accept + clip) 플랜만. 플랜 로그는 폴더 안에서 공유된다."""
    p = folder / "plans.jsonl"
    if not p.exists():
        return []
    run = meta.get("run") or {}
    try:
        lo = dt.datetime.strptime(run["started"], "%Y-%m-%d %H:%M:%S")
        hi = dt.datetime.strptime(meta["written"], "%Y-%m-%d %H:%M:%S")
    except (KeyError, TypeError, ValueError):
        lo, hi = dt.datetime.min, dt.datetime.max
    out = []
    for line in p.open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "raw" not in r:                     # late/rejected: 설치 안 됨
            continue
        try:
            w = dt.datetime.strptime(r["wall"], "%Y-%m-%d %H:%M:%S")
        except (KeyError, ValueError):
            w = lo
        if lo <= w <= hi:
            out.append(r)
    out.sort(key=lambda r: r.get("obs_t_rel", r.get("t0", 0.0)))
    return out


def load_tags(meta: dict):
    hw = meta.get("hardware") or {}
    path = Path(hw.get("tag_map") or (REPO / "config" / "tag_map_full.yaml"))
    if not path.exists():
        path = REPO / "config" / "tag_map_full.yaml"
    if not path.exists():
        return {}, 0.17
    tags = (yaml.safe_load(path.read_text()) or {}).get("tags", {})
    xy = {str(k): (float(v["position_m"][0]), float(v["position_m"][1]))
          for k, v in tags.items()}
    return xy, float(hw.get("tag_size_m") or 0.17)


def fit_window(win, lo, hi, keep_lo, keep_hi):
    """창 폭은 유지한 채 가능하면 [lo,hi](태그맵) 안으로 당긴다. 단 런 자체
    [keep_lo,keep_hi]는 절대 자르지 않는다 — 빈 여백보다 태그를 보여주는 쪽."""
    a, b = win
    w = b - a
    if hi - lo <= w:                       # 맵이 창보다 작다 → 맵 중심
        c = 0.5 * (lo + hi)
        a, b = c - w / 2, c + w / 2
    else:
        if a < lo:
            a, b = lo, lo + w
        if b > hi:
            a, b = hi - w, hi
    if a > keep_lo:
        a, b = keep_lo, keep_lo + w
    if b < keep_hi:
        a, b = keep_hi - w, keep_hi
    return (a, b)


def pick(n: int, scale: float, every: int | None) -> np.ndarray:
    """플랜 n개 중 scale 비율만 고르게 뽑는다 (10개 × 0.2 → 2개)."""
    if n == 0:
        return np.zeros(0, int)
    if every:
        return np.arange(0, n, max(1, every))
    k = max(1, int(round(n * float(scale))))
    return np.unique(np.linspace(0, n - 1, k).round().astype(int))


# ------------------------------------------------------------------ drawing
def draw(csv_path: Path, meta: dict, plans: list[dict], args, plt):
    from matplotlib import patches
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    d = read_csv(csv_path)
    dat = datum_of(meta)
    tm, off = mission_clock(d)
    xm, ym = to_map(d["px"], -d["py"], dat)
    yaw_map = math.degrees(dat[2]) - d["yaw_deg"]          # FLU yaw_deg = −yaw_ned

    ptm = np.array([p.get("obs_t_rel", p.get("t0", 0.0)) for p in plans], float)
    pax = np.array([p["anchor"]["eta"][0] for p in plans], float)
    pay = np.array([p["anchor"]["eta"][1] for p in plans], float)
    pxm, pym = (to_map(pax, pay, dat) if len(plans) else (np.zeros(0), np.zeros(0)))
    pyaw = np.array([math.degrees(dat[2] + np.asarray(p["raw"]["yaw"])[-1])
                     for p in plans], float)
    # The heading the vehicle HAD at each plan's observation instant — the
    # anchor yaw the plan was composed from, so (pyaw - phead) is exactly the
    # plan's dyaw. Falls back to the CSV's tag-solution yaw interpolated at
    # the plan time (same source, 20 Hz, so the difference is negligible).
    phead = np.full(len(plans), np.nan)
    for i, p in enumerate(plans):
        apu = p.get("anchor_pose_used")
        a = p.get("anchor") or {}
        if isinstance(apu, (list, tuple)) and len(apu) >= 6:
            phead[i] = math.degrees(dat[2] + float(apu[5]))
        elif a.get("yaw") is not None:
            phead[i] = math.degrees(dat[2] + float(a["yaw"]))
    sel = pick(len(plans), args.plan_scale, args.plan_every)
    heading_on = (bool(args.heading) if args.heading is not None
                  else bool((meta.get("policy") or {}).get("observe")
                            or meta.get("policy_observe")))

    live = np.isfinite(tm) & (tm >= (ptm.min() - 2.0 if len(sel) else -np.inf))
    lo = float(min(tm[live].min(), ptm[sel].min())) if len(sel) else float(np.nanmin(tm))
    hi = float(max(tm[live].max(), ptm[sel].max())) if len(sel) else float(np.nanmax(tm))
    norm, cmap = Normalize(lo, hi), plt.get_cmap(args.cmap)
    col = lambda t: cmap(norm(t))

    obj = None
    if args.object:
        obj = np.array([float(v) for v in args.object.replace(" ", "").split(",")])

    npan = 2 if args.panels == "both" else 1
    fig, axes = plt.subplots(1, npan, figsize=(8.4 * npan, 7.6))
    axes = np.atleast_1d(axes)
    ai = 0

    if args.panels in ("map", "both"):
        A = axes[ai]; ai += 1
        # 한계를 먼저 정한다. 런이 작아도 태그맵이 충분히 보이도록 정사각 창을
        # 쓰고(--min-span, 기본 2 m), --span으로 강제하거나 --view tags로 맵 전체.
        tags, ts = load_tags(meta)
        xs = np.r_[xm[live], (obj[0] if obj is not None else xm[live][0])]
        ys = np.r_[ym[live], (obj[1] if obj is not None else ym[live][0])]
        if args.view == "tags" and tags:
            T = np.array(list(tags.values()), float)
            xr = (T[:, 0].min() - ts, T[:, 0].max() + ts)
            yr = (T[:, 1].min() - ts, T[:, 1].max() + ts)
        else:
            span = float(args.span) if args.span else max(
                float(np.ptp(xs)) + 2 * args.pad,
                float(np.ptp(ys)) + 2 * args.pad, float(args.min_span))
            cx, cy = 0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max())
            xr = (cx - span / 2, cx + span / 2)
            yr = (cy - span / 2, cy + span / 2)
            if tags:                       # 빈 여백 대신 태그가 프레임을 채우게
                T = np.array(list(tags.values()), float)
                xr = fit_window(xr, T[:, 0].min() - ts, T[:, 0].max() + ts,
                                xs.min() - 0.05, xs.max() + 0.05)
                yr = fit_window(yr, T[:, 1].min() - ts, T[:, 1].max() + ts,
                                ys.min() - 0.05, ys.max() + 0.05)
        A.set_xlim(*yr); A.set_ylim(*xr)
        big = max(xr[1] - xr[0], yr[1] - yr[0])
        if args.tags:
            fs = float(np.clip(5.5 * 2.0 / big, 3.0, 7.0))
            for k, (tx, ty) in tags.items():
                if xr[0] - ts < tx < xr[1] + ts and yr[0] - ts < ty < yr[1] + ts:
                    A.add_patch(patches.Rectangle((ty - ts / 2, tx - ts / 2), ts, ts,
                                                  fill=False, ec="0.85", lw=0.6,
                                                  zorder=1, clip_on=True))
                    A.text(ty, tx, k, fontsize=fs, ha="center", va="center",
                           color="0.72", zorder=1, clip_on=True)

        A.plot(ym[live], xm[live], "-", color="0.6", lw=0.8, alpha=0.6, zorder=2)
        A.scatter(ym[live], xm[live], c=tm[live], cmap=cmap, norm=norm,
                  s=7, lw=0, zorder=3)

        L = float(args.yaw_len) if args.yaw_len else 0.14 * big
        for i in sel:
            c = col(ptm[i])
            A.plot([pym[i], pym[i] + L * math.sin(math.radians(pyaw[i]))],
                   [pxm[i], pxm[i] + L * math.cos(math.radians(pyaw[i]))],
                   "-", color=c, lw=1.7, solid_capstyle="round", zorder=5)
            A.plot([pym[i]], [pxm[i]], "o", ms=4.2, mfc=c, mec="k", mew=0.5, zorder=6)
            if heading_on:
                yv = (float(phead[i]) if np.isfinite(phead[i])
                      else float(np.interp(ptm[i], tm, yaw_map)))
                A.plot([pym[i], pym[i] + 0.62 * L * math.sin(math.radians(yv))],
                       [pxm[i], pxm[i] + 0.62 * L * math.cos(math.radians(yv))],
                       ":", color=c, lw=1.5, zorder=4)

        A.plot(ym[live][0], xm[live][0], "^", ms=9, mfc="none", mec="k", mew=1.4, zorder=7)
        A.plot(ym[live][-1], xm[live][-1], "s", ms=8, mfc="none", mec="k", mew=1.4, zorder=7)
        if obj is not None:
            A.plot(obj[1], obj[0], "o", ms=13, color="tab:orange", mec="k", mew=0.8, zorder=8)

        A.set_aspect("equal")
        A.set_xlim(*yr); A.set_ylim(*xr)
        A.set_xlabel("y_map (m)  →"); A.set_ylabel("x_map (m)  ↑")
        A.set_title(args.title or f"{csv_path.parent.name} / {csv_path.stem} — "
                                  f"map frame, colour = mission time")
        h = [Line2D([], [], color="0.6", lw=1.2, marker="o", ms=4, mfc="0.4", mec="none",
                    label="ROV COM path (colour = t)"),
             Line2D([], [], color="0.35", lw=1.7, label="policy plan-end yaw (1 s ahead)")]
        if heading_on:
            h.append(Line2D([], [], color="0.35", lw=1.5, ls=":",
                            label="vehicle heading at that instant (tag PnP yaw)"))
        h += [Line2D([], [], color="k", lw=0, marker="^", mfc="none", ms=8, label="start"),
              Line2D([], [], color="k", lw=0, marker="s", mfc="none", ms=7, label="end")]
        if obj is not None:
            h.append(Line2D([], [], color="tab:orange", lw=0, marker="o", ms=10, label="object"))
        A.legend(handles=h, loc="best", fontsize=8, framealpha=0.9)
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=A, fraction=0.035,
                     pad=0.02, label="mission time (s)")

    if args.panels in ("yaw", "both"):
        B = axes[ai]
        B.plot(tm[live], yaw_map[live], "-", color="0.15", lw=1.4, label="vehicle yaw_map")
        if "ryaw_deg" in d:
            B.plot(tm[live], math.degrees(dat[2]) - d["ryaw_deg"][live], "--",
                   color="tab:green", lw=1.0, label="follower reference yaw (ryaw)")
        if obj is not None:
            B.plot(tm[live], np.degrees(np.arctan2(obj[1] - ym[live], obj[0] - xm[live])),
                   "-", color="tab:orange", lw=1.4, label="bearing to object")
        if len(sel):
            B.scatter(ptm[sel], pyaw[sel], c=ptm[sel], cmap=cmap, norm=norm,
                      s=42, ec="k", lw=0.4, zorder=5, label="policy plan-end yaw (1 s ahead)")
        B.set_xlabel("mission time (s)"); B.set_ylabel("deg (map)")
        B.grid(alpha=0.3); B.legend(fontsize=8)
        B.set_title("yaw: vehicle vs policy request" + ("" if obj is None else " vs object bearing"))

    fig.tight_layout()
    out = args.out or (csv_path.parent /
                       f"plot_policy_{args.panels}_{csv_path.stem.split('_')[-1]}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[ok]   {csv_path.name}: {len(sel)}/{len(plans)} plans drawn "
          f"(scale {args.plan_scale}), t {lo:.1f}–{hi:.1f} s, clock offset "
          f"{off:+.3f} s  →  {out}")
    return out


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="0908_112101 | 20260908/0908_112101 | 폴더 경로")
    ap.add_argument("--mission", default=None, help="HHMMSS (기본: 폴더의 모든 미션)")
    ap.add_argument("--plan-scale", type=float, default=1.0,
                    help="플랜을 이 비율만 그린다 (0.25 = 1/4, 기본 1.0)")
    ap.add_argument("--plan-every", type=int, default=None,
                    help="대신 N개마다 하나 (지정하면 --plan-scale 무시)")
    ap.add_argument("--object", default=None, metavar="X,Y",
                    help="물체의 MAP 좌표 (예: 0.147,0.379)")
    ap.add_argument("--panels", choices=["map", "yaw", "both"], default="map")
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--yaw-len", type=float, default=None,
                    help="yaw 광선 길이 (m). 기본은 창 크기의 0.14배")
    ap.add_argument("--view", choices=["run", "tags"], default="run",
                    help="run = 런 주변(기본), tags = 태그맵 전체")
    ap.add_argument("--span", type=float, default=None,
                    help="지도 창을 이 크기(m)의 정사각으로 강제")
    ap.add_argument("--min-span", type=float, default=2.0,
                    help="런이 작아도 지도 창은 최소 이만큼 (m, 기본 2.0)")
    ap.add_argument("--heading", dest="heading", action="store_true", default=None,
                    help="그 순간의 실제 기체 헤딩을 같은 색 점선으로 겹쳐 그린다 "
                         "(기본: observe 런이면 자동으로 켜짐)")
    ap.add_argument("--no-heading", dest="heading", action="store_false",
                    help="헤딩 점선을 끈다")
    ap.add_argument("--no-tags", dest="tags", action="store_false")
    ap.add_argument("--pad", type=float, default=0.5, help="여백 (m)")
    ap.add_argument("--title", default=None)
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--list", action="store_true", help="미션 목록만 찍는다")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    folder = resolve_run(args.run)
    ms = missions(folder, args.mission)
    if not ms:
        raise SystemExit(f"[fail] {folder} 에 mpc_*.csv + meta 쌍이 없다"
                         + (f" (mission {args.mission})" if args.mission else ""))
    if args.list:
        for c, meta in ms:
            tr = (meta.get("trajectory") or {})
            print(f"{c.stem}  kind={tr.get('kind')}  ctrl={(meta.get('controller') or {}).get('type')}  "
                  f"mode={(meta.get('run') or {}).get('flight_mode_at_engage')}  "
                  f"rows={(meta.get('run') or {}).get('rows')}")
        return 0

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 140})

    n = 0
    for c, meta in ms:
        kind = (meta.get("trajectory") or {}).get("kind")
        plans = load_plans(folder, meta)
        if kind != "policy" and not plans:
            print(f"[skip] {c.stem}: trajectory.kind={kind}, 설치된 플랜 0개")
            continue
        if args.out and len(ms) > 1:
            print("[warn] --out은 미션이 여럿이면 마지막 그림만 남긴다")
        draw(c, meta, plans, args, plt)
        n += 1
    if not n:
        print("[fail] 그릴 정책 미션이 없다")
        return 1
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
