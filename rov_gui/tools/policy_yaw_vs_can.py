#!/usr/bin/env python3
"""policy_yaw_vs_can.py — 관측 런에서 정책의 yaw 의도가 **캔 쪽을 향하는가**, 체크포인트별로.

    P=~/miniforge3/envs/rovgui-pose/bin/python
    $P -m rov_gui.tools.policy_yaw_vs_can 0908_173106                 # orig + __replay_* 전부
    $P -m rov_gui.tools.policy_yaw_vs_can 0908_170428 0908_175151 --variants orig,E,A,C

`data/20260908/0908_173106_observe/diag/yaw_vs_bearing.py`(2026-09-08, 한 런
전용, 캔이 태그 61#0 모서리에 있다는 외부 지식에 의존)를 **여러 런·여러 체크포인트**로
일반화한 것이다. 캔이 어디 있었는지 적어둔 런이 없으므로(CSV `obj_px` 전부 비어 있음,
2026-09-11 확인) 위치는 관측에서 복원한다.

캔 위치를 아는 방법
-------------------
1. **depth 블롭** (diag 스크립트의 `detect` 그대로): 플랜의 depth PNG에서 RANSAC 바닥면을
   맞추고, 그 면보다 가까운 작은 덩어리(그리퍼 영역 제외)를 캔으로 잡는다. 카메라 기준
   방위 `cam_bear = atan((u − cx)/fx)`. 내재 파라미터는 `policy_obs/meta.json`의
   `builder.grid.P1`에서 읽는다(하드코딩 아님). 캔이 시야에 있고 검출이 맞은 플랜만.
2. **광선 교차로 맵 위치 복원**: 검출된 플랜마다 (ROV 맵 위치, 헤딩 + cam_bear) 광선을
   긋고 최소제곱 교점을 구한다. 캔이 고정돼 있었다면 한 점에 모이고, 그 점에서 **모든**
   플랜의 맵 기준 방위 `map_bear`가 나온다. 태그 가정이 없고, 거리도 외부 extrinsic도
   쓰지 않는다(카메라 yaw ≈ 기체 yaw, 렌즈는 기체 원점 근처라는 근사만).
   교점의 잔차(광선까지의 거리 RMS)가 크면 캔이 움직였거나 검출이 틀린 것이다 — 출력한다.

두 방위는 부호 수준에서만 일치한다(173106: |차| 16–19°, r +0.70, 좌/우 일치 86 %). 그리고
광선 교차는 **5개 런 중 2개(173106, 170428)에서만** 카메라와 맞았다 — 나머지는 검출이 멀거나
캔이 고정이 아니었던 듯. 그래서 요약표는 `cam_bear`(근거리 `--max-z` 이내) 기준이다.

읽는 법: **부호 일치율만 보면 속는다.** 상수를 내는 정책도 캔이 한쪽에만 있던 런에서는 높은
일치율을 얻는다(175004 의 C 84 %, 175151 의 orig 98 %). 반드시 r 과 함께, 그리고 캔이 좌우
양쪽에 있었던 런인지와 함께 읽어라. 기울기 소수점은 읽지 않는다.

무엇을 재나
-----------
플랜의 knot-15 dyaw(`action_raw[15, 3]`, deg, NED + = 우회전) vs 캔 방위(+ = 우측):
* 부호 일치율 — |방위| > `--min-bear`(기본 8°)인 플랜에서, 정책이 캔 쪽으로 돌자고
  했는가. 방위 오차 15–20°에 가장 강건한 잣대.
* r — 방위가 커질수록 더 돌자고 하는가.
* 기울기 [°/°] — 참고용. 학습 정답은 diag 스크립트 기준 +0.45.
* 구간별 dyaw 평균 — 왼쪽/정면/오른쪽에서 실제로 어느 쪽으로 돌자고 했나.
같은 ckpt를 다시 돌린 `verify` 변형이 있으면 orig 대비 dyaw RMS를 잡음 바닥으로 낸다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
#: The dated root; observe runs are the ``*_observe`` leaves in it
#: (rov_gui/runstore.py, 2026-09-14 — was the separate sessions/policy_observe tree).
SESSIONS = REPO / "data"
RUN_KIND = "observe"


# ------------------------------------------------------------------ run folder

def run_folder(spec: str) -> Path:
    p = Path(spec)
    if p.is_dir():
        return p
    # `0907_133358` / `20260907/0907_133358` name the run; the kind suffix
    # is added here so a spec never has to spell it.
    pats = ([spec, f"{spec}_{RUN_KIND}", f"*/{spec}", f"*/{spec}_{RUN_KIND}"]
            if "/" not in spec else [spec, f"{spec}_{RUN_KIND}"])
    for pat in pats:
        hits = sorted(q for q in SESSIONS.glob(pat)
                      if q.is_dir() and "__replay_" not in q.name)
        if hits:
            return hits[-1]
    raise SystemExit(f"[fail] run folder not found for {spec!r} (under {SESSIONS}, kind {RUN_KIND})")


def wrap_deg(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


# ------------------------------------------------------------------ can detector (diag/yaw_vs_bearing.py 그대로)

def detect(dep: np.ndarray, fx: float, cx: float, cy: float):
    """can = compact blob nearer than the local floor plane, upper 65 % of the frame.
    returns (u, v, z_m, area) or None."""
    z = dep.astype(np.float32) / 1000.0
    H, W = z.shape
    vv, uu = np.mgrid[0:H, 0:W]
    X = (uu - cx) / fx * z
    Y = (vv - cy) / fx * z
    ok = (z > 0.35) & (z < 3.0)
    ok &= ~((vv > 0.55 * H) & (np.abs(uu - cx) < 0.22 * W))      # gripper wedge
    pts = np.column_stack([X[ok], Y[ok], z[ok]])
    if len(pts) < 500:
        return None
    rng = np.random.default_rng(0)
    best, bi = None, 0
    for _ in range(120):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        dd = -n @ s[0]
        k = int((np.abs(pts @ n + dd) < 0.02).sum())
        if k > bi:
            bi, best = k, (n, dd)
    if best is None:
        return None
    n, dd = best
    inl = np.abs(pts @ n + dd) < 0.02
    A = pts[inl]
    cen = A.mean(0)
    _, _, vt = np.linalg.svd(A - cen, full_matrices=False)
    n = vt[2]
    dd = -n @ cen
    if n[2] < 0:
        n, dd = -n, -dd
    P = np.dstack([X, Y, z])
    dist = P @ n + dd
    if np.nanmedian(dist[ok]) > 0:
        dist = -dist
    near = (dist > 0.02) & ok & (vv < 0.65 * H)
    near = cv2.morphologyEx(near.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    nlab, lab, stats, cents = cv2.connectedComponentsWithStats(near, 8)
    cand = []
    for i in range(1, nlab):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if 25 <= a <= 4000:
            u, v = cents[i]
            cand.append((a, float(u), float(v), float(np.median(z[lab == i]))))
    if not cand:
        return None
    a, u, v, zz = max(cand)
    return u, v, zz, a


# ------------------------------------------------------------------ per-run geometry

class Run:
    def __init__(self, folder: Path):
        self.folder = folder
        self.name = folder.name
        csvs = sorted(folder.glob("mpc_*.csv"))
        if not csvs:
            raise SystemExit(f"[fail] no mpc_*.csv in {folder}")
        meta = json.loads(csvs[0].with_suffix(".meta.json").read_text())
        d = meta["hardware"]["datum_tag_frame"]
        self.x0, self.y0 = float(d["p0"][0]), float(d["p0"][1])
        self.yaw0 = math.radians(float(d["yaw0_deg"]))
        obs = folder / "policy_obs"
        om = json.loads((obs / "meta.json").read_text())
        P1 = np.asarray(om["builder"]["grid"]["P1"], float)
        self.fx, self.cx, self.cy = float(P1[0, 0]), float(P1[0, 2]), float(P1[1, 2])
        self.obs_dir = obs
        self.tcap, self.depth_png = [], []
        with (obs / "index.csv").open() as fh:
            for r in csv.DictReader(fh):
                self.tcap.append(float(r["t_capture"]))
                self.depth_png.append(r.get("depth_png") or "")
        self.tcap = np.asarray(self.tcap)

    def to_map(self, xn, yn):
        c, s = math.cos(self.yaw0), math.sin(self.yaw0)
        return self.x0 + c * xn - s * yn, self.y0 + s * xn + c * yn

    def plans(self, variant: str):
        p = (self.folder if variant == "orig"
             else self.folder.parent / f"{self.name}__replay_{variant}") / "plans.jsonl"
        if not p.exists():
            return None
        out = {}
        for r in map(json.loads, p.open()):
            if "raw" in r and r.get("action_raw") and r.get("obs_rows_t"):
                out[int(r["plan_id"])] = r
        return out

    def frame_for(self, t_now: float, tol: float = 1e-3):
        i = int(np.argmin(np.abs(self.tcap - t_now)))
        return i if abs(self.tcap[i] - t_now) <= tol and self.depth_png[i] else None


def can_bearings(run: Run, orig: dict):
    """plan_id -> dict(xm, ym, yaw_map, cam_bear|nan, z|nan). 광선 교차로 캔 맵 위치도."""
    geo = {}
    for pid, r in orig.items():
        eta = np.asarray(r["anchor_pose_used"], float)
        xm, ym = run.to_map(eta[0], eta[1])
        yaw_map = run.yaw0 + eta[5]
        g = {"xm": xm, "ym": ym, "yaw_map": yaw_map, "cam_bear": np.nan, "z": np.nan}
        i = run.frame_for(float(r["obs_rows_t"][1]))
        if i is not None:
            dep = cv2.imread(str(run.obs_dir / run.depth_png[i]), cv2.IMREAD_UNCHANGED)
            if dep is not None and dep.ndim == 2:
                det = detect(dep, run.fx, run.cx, run.cy)
                if det:
                    g["cam_bear"] = math.degrees(math.atan((det[0] - run.cx) / run.fx))
                    g["z"] = det[2]
        geo[pid] = g
    # ---- 광선 교차: 캔의 맵 위치 (검출된 플랜만) ----
    det = [g for g in geo.values() if np.isfinite(g["cam_bear"])]
    can = None
    if len(det) >= 5:
        O = np.array([[g["xm"], g["ym"]] for g in det])
        ang = np.array([g["yaw_map"] + math.radians(g["cam_bear"]) for g in det])
        D = np.column_stack([np.cos(ang), np.sin(ang)])
        # Σ (I − d dᵀ)(p − o) = 0
        Amat = np.zeros((2, 2)); b = np.zeros(2)
        for o, dvec in zip(O, D):
            M = np.eye(2) - np.outer(dvec, dvec)
            Amat += M; b += M @ o
        p = np.linalg.solve(Amat, b)
        # 잔차: 각 광선까지의 수직 거리
        res = np.array([np.linalg.norm((np.eye(2) - np.outer(dv, dv)) @ (p - o))
                        for o, dv in zip(O, D)])
        can = {"xy": p, "n": len(det), "res_rms_m": float(np.sqrt(np.mean(res ** 2))),
               "res_p90_m": float(np.quantile(res, 0.9))}
        for g in geo.values():
            b_abs = math.degrees(math.atan2(p[1] - g["ym"], p[0] - g["xm"]))
            g["map_bear"] = float(wrap_deg(b_abs - math.degrees(g["yaw_map"])))
    else:
        for g in geo.values():
            g["map_bear"] = np.nan
    return geo, can


# ------------------------------------------------------------------ metrics

#: Which ``action_raw`` column is ``dyaw`` per representation (design D1,
#: 2026-09-26: column 3 for BOTH flyable reprs, so 5- and 7-dim replays of one
#: run compare on the same column). A record without ``action_repr`` (pre-
#: 2026-09-07) is read by width; pose10d rows have no yaw column and are
#: derived through the same encoder the station uses.
DYAW_COL = {"pos_yaw_width": 3, "pos_rpy_width": 3}


def dyaw15(rec):
    a = np.asarray(rec["action_raw"], float)
    repr_ = str(rec.get("action_repr") or "")
    if not repr_:
        repr_ = {5: "pos_yaw_width", 7: "pos_rpy_width", 10: "pose10d"}.get(a.shape[-1], "")
    if repr_ == "pose10d":
        from rov_gui.control.policy_frames import encode_pos_yaw, pose10d_to_mat
        return math.degrees(float(encode_pos_yaw(pose10d_to_mat(a[15:16, :9]))[1][0]))
    return math.degrees(float(a[15, DYAW_COL.get(repr_, 3)]))


def metrics(bear: np.ndarray, dy: np.ndarray, min_bear: float):
    big = np.abs(bear) > min_bear
    out = {"n": int(len(bear)), "n_big": int(big.sum())}
    if big.sum() >= 3:
        out["sign_agree"] = float(np.mean(np.sign(dy[big]) == np.sign(bear[big])))
    else:
        out["sign_agree"] = float("nan")
    if len(bear) >= 3 and np.std(bear) > 0 and np.std(dy) > 0:
        out["r"] = float(np.corrcoef(bear, dy)[0, 1])
        out["slope"] = float(np.polyfit(bear, dy, 1)[0])
    else:
        out["r"] = out["slope"] = float("nan")
    out["mean"] = float(dy.mean()) if len(dy) else float("nan")
    return out


BINS = ((-90, -15), (-15, -5), (-5, 5), (5, 15), (15, 90))


def analyse(run: Run, variants, min_bear: float, max_bear: float = 45.0,
            max_z: float = 1.5):
    orig = run.plans("orig")
    if not orig:
        raise SystemExit(f"[fail] {run.name}: no installed plans")
    geo, can = can_bearings(run, orig)
    n_det = sum(1 for g in geo.values() if np.isfinite(g["cam_bear"]))
    print(f"\n{'=' * 78}\n{run.name}: {len(orig)} installed plans, can detected in {n_det}")
    if can is None:
        print("  [!] 검출이 5개 미만 — 캔 맵 위치 복원 불가, 카메라 방위만 사용")
    else:
        print(f"  can (ray intersection of {can['n']} detections): map ({can['xy'][0]:+.3f}, "
              f"{can['xy'][1]:+.3f}) m, residual RMS {can['res_rms_m'] * 100:.1f} cm "
              f"p90 {can['res_p90_m'] * 100:.1f} cm"
              + ("   [!] 잔차 큼: 캔이 움직였거나 오검출" if can["res_rms_m"] > 0.25 else ""))
    have = {v: run.plans(v) for v in variants}
    have = {v: p for v, p in have.items() if p}
    ids = sorted(set(orig) & set.intersection(*[set(p) for p in have.values()]))
    if "verify" in have:
        d = np.array([dyaw15(have["verify"][i]) - dyaw15(orig[i]) for i in ids])
        print(f"  noise floor (verify vs orig, same ckpt): dyaw RMS {np.sqrt(np.mean(d ** 2)):.2f}°")
    results = {}
    for src, key in (("map_bear (all plans, ray-intersection can)", "map_bear"),
                     ("cam_bear (detected plans only)", "cam_bear")):
        # 시야 안의 캔만: C3 수평 반시야 ≈ atan(320/fx) ≈ 43°. 뒤에 있는 캔을 안
        # 따라갔다고 감점하지 않는다 — 정책은 못 본 것에 반응할 수 없다.
        # cam_bear 는 근거리 검출만: 173106 에서 0.45 m 검출은 태그맵과 맞고 1.5 m
        # 검출은 어긋났다 — 먼 블롭은 캔이 아닐 가능성이 크다.
        sel = [i for i in ids if np.isfinite(geo[i][key]) and abs(geo[i][key]) <= max_bear
               and (key != "cam_bear" or geo[i]["z"] <= max_z)]
        if len(sel) < 5:
            continue
        b = np.array([geo[i][key] for i in sel])
        print(f"\n  -- {src}: n={len(sel)} (|bear|<={max_bear:g}°), range {b.min():+.0f}..{b.max():+.0f}°, "
              f"|bear|>{min_bear:g}°: {int((np.abs(b) > min_bear).sum())}")
        print(f"  {'variant':8} {'sign agree':>10} {'r':>6} {'slope':>7} {'mean dyaw':>9}   "
              + " ".join(f"{lo:+3d}..{hi:+3d}" for lo, hi in BINS))
        for v in variants:
            if v not in have or v == "verify":
                continue
            dy = np.array([dyaw15(have[v][i]) for i in sel])
            m = metrics(b, dy, min_bear)
            bins = []
            for lo, hi in BINS:
                msk = (b >= lo) & (b < hi)
                bins.append(f"{dy[msk].mean():+6.1f}" if msk.sum() else "     -")
            print(f"  {v:8} {m['sign_agree'] * 100:9.0f}% {m['r']:+6.2f} {m['slope']:+7.3f} "
                  f"{m['mean']:+9.2f}   " + " ".join(f"{x:>8}" for x in bins))
            results.setdefault(v, {})[key] = m
        # 정답 참고선
        print(f"  {'(GT ref)':8} {'':>10} {'':>6} {'+0.450':>7} {'':>9}   "
              + " ".join(f"{(0.45 * b[(b >= lo) & (b < hi)].mean() - 1.85 if ((b >= lo) & (b < hi)).sum() else float('nan')):+8.1f}"
                        for lo, hi in BINS))
    return {"run": run.name, "n_plans": len(ids), "n_det": n_det, "can": can, "results": results}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--variants", default="verify,orig,E,A,C",
                    help="orig | __replay_<NAME> 들 (콤마)")
    ap.add_argument("--min-bear", type=float, default=8.0,
                    help="부호 일치율을 잴 최소 |방위| [deg]")
    ap.add_argument("--max-bear", type=float, default=45.0,
                    help="이 |방위|를 넘는 플랜은 캔이 시야 밖 — 제외 [deg]")
    ap.add_argument("--max-z", type=float, default=1.5,
                    help="cam_bear: 이 거리[m]보다 먼 검출은 제외 (오검출 가능)")
    ap.add_argument("--json", type=Path, default=None, help="요약을 JSON으로")
    a = ap.parse_args(argv)
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    summary = []
    for spec in a.runs:
        run = Run(run_folder(spec))
        summary.append(analyse(run, variants, a.min_bear, a.max_bear, a.max_z))
    # ---- 런 통합 ----
    print(f"\n{'=' * 78}\nSUMMARY — sign agreement (|bear|>{a.min_bear:g}°, cam_bear z<={a.max_z:g} m) / r,  per run\n"
          f"  (map_bear 는 광선 교차가 카메라와 맞는 런에서만 믿을 것 — 2026-09-11: 173106, 170428 만)")
    vs = [v for v in variants if v != "verify"]
    print(f"  {'run':12} {'plans':>5} {'det':>4} " + " ".join(f"{v:>14}" for v in vs))
    for s in summary:
        cells = []
        for v in vs:
            m = s["results"].get(v, {}).get("cam_bear")
            cells.append(f"{m['sign_agree'] * 100:4.0f}% r{m['r']:+.2f}" if m else "      -       ")
        print(f"  {s['run']:12} {s['n_plans']:5d} {s['n_det']:4d} " + " ".join(f"{c:>14}" for c in cells))
    # 플랜 수 가중 평균
    print(f"  {'weighted':12} {'':>5} {'':>4} ", end="")
    for v in vs:
        num = den = 0.0; rs = []
        for s in summary:
            m = s["results"].get(v, {}).get("cam_bear")
            if m and np.isfinite(m["sign_agree"]):
                num += m["sign_agree"] * m["n_big"]; den += m["n_big"]; rs.append(m["r"])
        print(f"{(num / den * 100 if den else float('nan')):4.0f}% r{np.nanmean(rs) if rs else float('nan'):+.2f}".rjust(14), end=" ")
    print()
    if a.json:
        def conv(o):
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, (np.floating, np.integer)): return o.item()
            raise TypeError
        a.json.write_text(json.dumps(summary, default=conv, indent=1, ensure_ascii=False))
        print(f"  -> {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
