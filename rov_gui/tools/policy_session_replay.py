#!/usr/bin/env python3
"""policy_session_replay.py — 기록된 정책 런의 관측을 **다른 체크포인트**로 다시 돌린다.

    P=~/miniforge3/envs/rovgui-pose/bin/python
    $P -m rov_gui.tools.policy_session_replay 0908_175151 --verify
    $P -m rov_gui.tools.policy_session_replay 0908_175151 \
        --ckpt E=<run_E>/checkpoints/epoch=0195.ckpt \
        --ckpt A=<run_A>/checkpoints/epoch=0195.ckpt \
        --ckpt C=<run_C>/checkpoints/epoch=0195.ckpt
    # 6-DoF 변형(2026-09-26)의 OOD 프로브: 7-dim ckpt를 기존 수중 런에 먹여 roll/pitch 열을 본다
    $P -m rov_gui.tools.policy_session_replay 0908_175151 \
        --ckpt D7=<run_7d>/checkpoints/selected.ckpt [--attitude-track --rp-max-deg 20]

"이 데이터에 저 정책이 학습됐다면 reference trajectory를 어떻게 그렸을까"에 답한다.
런이 남긴 관측을 그대로 다시 먹여 플랜을 다시 짓고, ``plans.jsonl`` 호환 폴더를
체크포인트마다 하나씩 쓴다. 그림은 ``plot_policy_map.py``가 **그대로** 그린다 —
여기서 두 번째 렌더러를 만들지 않는 이유는 depth 영상 도구와 같다: 같은 플랜을 다른
픽셀에 찍는 그림 두 장은 정책이 아니라 렌더러를 비교하게 만든다.

무엇이 정확하고 무엇이 재구성인가 (이 도구의 핵심 고지)
--------------------------------------------------------
* **이미지는 정확하다.** ``policy_obs/obs/*.png``는 "the tensor the policy consumed,
  written verbatim and losslessly"(policy_obs/meta.json)이고, 플랜 레코드의
  ``obs_rows_t``가 ``index.csv``의 ``t_capture``와 **부동소수점까지 일치**하므로
  어느 두 장을 먹였는지가 추론이 아니라 조회로 정해진다. ``cv2.imwrite`` ->
  ``cv2.imread`` 왕복이라 배열이 비트 단위로 돌아온다.
* **low-dim proprio는 재구성이다.** 라이브 워커는 ``lowdim``을 ``PolicyPlan``에 실어
  나르지만 ``plans.jsonl``에는 **직렬화하지 않는다**. 그래서 여기서는 미션 CSV의
  ``px/py/pz/roll_deg/pitch_deg/yaw_deg``와 ``grip_w_est``를 ``t_traj``로 보간해
  ``policy_frames.lowdim_obs``에 넣는다. 라이브가 쓰던 자체 히스토리 버퍼와 보간
  방식이 다를 수 있다.

7-dim(``pos_rpy_width``) 체크포인트 — 비행 전 OOD 프로브 (2026-09-26)
-------------------------------------------------------------------
6-DoF 변형의 체크포인트를 기존 수중 런에 먹이면 ``plans.jsonl``의 ``action_raw``가
7열이 되고, 레코드마다 ``rp_tracked``(bool)·``rp_raw``(최상위: 디코드된 절대 roll/pitch,
rad, (2,K) — 워커의 plans.jsonl 과 같은 자리)·``rp_clipped_deg``/``rp_clipped_n``(T1 클립량)·
``dropped_rp_deg``가 붙고, ``raw.rp``(실린 (2,J))는 tracked 일 때만 있다. 기본은
**dropped-and-logged**(``compose_plan(track_rp=False)``: 앵커 자세로 레벨링, ``rp_raw``는
정보용) — 실기의 첫 7-dim 비행과 같은 모드다. ``--attitude-track``은 ``track_rp=True``로
합성해(T1 클립 ``--rp-max-deg``) 플랜이 실제로 실었을 자세를 쓴다. 7-dim ckpt이면 이름과
무관하게 "요청 자세 vs 그 순간의 측정 자세(CSV ``roll_deg/pitch_deg``)"의 roll/pitch RMS를
찍는다 — 학습 라벨은 손목이고 실기는 선체이므로 이 숫자는 "정책이 요청한 기울기"이지 추종
성능이 아니다. 4-DoF(5-dim) ckpt의 출력 레코드는 이 변경 전과 **키 하나 다르지 않다**
(``rp_tracked`` 등은 7열 레코드에만 붙는다).

그래서 ``--verify``가 있다
--------------------------
``--verify``는 **런이 실제로 쓴 그 체크포인트**를 같은 경로로 다시 돌려 재구성된
플랜을 기록된 ``raw``와 비교한다. 재구성이 옳으면 차이는 DDIM 샘플링 잡음 수준이고,
크면 이 도구가 그 사실을 숫자로 말한다. **E/A/C 그림을 믿기 전에 이걸 먼저 봐라** —
verify가 크게 어긋나면 세 그림의 차이는 체크포인트 차이가 아니라 재구성 오차다.

출력
----
``<run>__replay_<NAME>/`` 에 ``plans.jsonl``(원본 레코드에서 action 유래 필드만 교체)
과 미션 CSV/meta의 **복사본**을 둔다. 경로·태그맵·시계가 원본과 같은 파일에서 오므로
그림의 색과 배경이 원본과 정확히 비교된다. 그린 뒤:

    ~/miniforge3/envs/robust/bin/python -m rov_gui.tools.plot_policy_map \
        data/20260908/0908_175151_observe__replay_E
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rov_gui.control.policy_frames import compose_plan, lowdim_obs  # noqa: E402
from rov_gui.state import ACTION_DIM_BY_REPR, ACTION_REPR_POS_RPY_WIDTH  # noqa: E402

#: The dated root; observe runs are the ``*_observe`` leaves in it
#: (rov_gui/runstore.py, 2026-09-14 — was the separate sessions/policy_observe tree).
SESSIONS = REPO / "data"
RUN_KIND = "observe"

#: action 으로부터 다시 계산되는 필드. 원본 레코드에서 이것만 갈아끼우고 나머지
#: (시계, 앵커, 태그, wall 등)는 그대로 둔다 — 그림이 읽는 키를 잃지 않기 위해서다.
DERIVED = ("raw", "action_raw", "p_tcp", "n_knots", "t0", "dt", "plan_id",
           # 6-DoF 변형(2026-09-26): 7열 action에서만 다시 계산되는 필드
           "rp_tracked", "rp_clipped_deg", "rp_clipped_n", "dropped_rp_deg")


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
        hits = sorted(q for q in SESSIONS.glob(pat) if q.is_dir())
        if hits:
            return hits[-1]
    raise SystemExit(f"[fail] run folder not found for {spec!r} (under {SESSIONS}, kind {RUN_KIND})")


def mission_of(folder: Path):
    """(csv, meta_path, meta) — 이 폴더의 정책 미션. 여러 개면 첫 번째."""
    for c in sorted(folder.glob("mpc_*.csv")):
        mp = c.with_suffix(".meta.json")
        if mp.exists():
            return c, mp, json.loads(mp.read_text())
    raise SystemExit(f"[fail] no mpc_*.csv + .meta.json in {folder}")


# ------------------------------------------------------------------ the run's own state

class MissionState:
    """미션 CSV를 ``t_traj`` 위의 eta(6)와 그리퍼 폭으로 읽는다.

    ``t_traj``를 쓰는 이유는 플랜 로그의 시계가 그것이기 때문이다 (CSV의 ``t``가
    아니다 — plot_policy_map.py 의 "시계 주의" 참조). 두 시계를 우리가 다시 맞추지
    않고 CSV가 이미 들고 있는 열을 쓴다.
    """

    def __init__(self, csv_path: Path):
        t, eta, w = [], [], []
        with csv_path.open() as fh:
            for row in csv.DictReader(fh):
                try:
                    tt = float(row["t_traj"])
                    e = [float(row["px"]), float(row["py"]), float(row["pz"]),
                         math.radians(float(row["roll_deg"])),
                         math.radians(float(row["pitch_deg"])),
                         math.radians(float(row["yaw_deg"]))]
                except (KeyError, TypeError, ValueError):
                    continue
                if not all(math.isfinite(v) for v in e) or not math.isfinite(tt):
                    continue
                try:
                    gw = float(row.get("grip_w_est") or "nan")
                except ValueError:
                    gw = float("nan")
                t.append(tt), eta.append(e), w.append(gw)
        if not t:
            raise SystemExit(f"[fail] no usable rows in {csv_path}")
        o = np.argsort(np.asarray(t))
        self.t = np.asarray(t)[o]
        self.eta = np.asarray(eta)[o]
        w = np.asarray(w)[o]
        if np.isnan(w).all():
            w = np.zeros_like(w)
        else:                                   # 폭은 드물게 갱신된다: 앞으로 채운다
            idx = np.where(~np.isnan(w), np.arange(len(w)), 0)
            np.maximum.accumulate(idx, out=idx)
            w = w[idx]
            w[np.isnan(w)] = np.nanmin(w)
        self.w = w

    def eta_at(self, t_traj: float) -> np.ndarray:
        # yaw 는 랩을 도니 성분별 선형보간은 위험하다. 각도 셋은 unwrap 후 보간한다.
        out = np.empty(6)
        out[:3] = [np.interp(t_traj, self.t, self.eta[:, i]) for i in range(3)]
        for i in (3, 4, 5):
            out[i] = np.interp(t_traj, self.t, np.unwrap(self.eta[:, i]))
        return out

    def width_at(self, t_traj: float) -> float:
        return float(np.interp(t_traj, self.t, self.w))


# ------------------------------------------------------------------ observations

class ObsStore:
    """``policy_obs/`` — 정책이 먹은 224x224x3 uint8 텐서, 촬영 시각으로 찾는다."""

    def __init__(self, folder: Path):
        d = folder / "policy_obs"
        idx = d / "index.csv"
        if not idx.exists():
            raise SystemExit(
                f"[fail] {idx} 없음 — 이 런은 관측을 기록하지 않았다 "
                f"(--record-depth 없이 돌린 런은 이 도구로 재생할 수 없다)")
        self.dir = d
        self.t, self.png = [], []
        with idx.open() as fh:
            for row in csv.DictReader(fh):
                self.t.append(float(row["t_capture"]))
                self.png.append(row["obs_png"])
        self.t = np.asarray(self.t)
        self.meta = json.loads((d / "meta.json").read_text())

    def row_for(self, t_capture: float, tol: float = 1e-3):
        """가장 가까운 기록 시각. ``tol`` 밖이면 None — 추측해서 먹이지 않는다."""
        if not len(self.t):
            return None
        i = int(np.argmin(np.abs(self.t - t_capture)))
        return i if abs(self.t[i] - t_capture) <= tol else None

    def image(self, i: int) -> np.ndarray:
        """기록된 배열 그대로. ``cv2.imwrite`` 로 썼으니 ``cv2.imread`` 로 돌아온다."""
        a = cv2.imread(str(self.dir / self.png[i]), cv2.IMREAD_COLOR)
        if a is None:
            raise SystemExit(f"[fail] unreadable obs png: {self.png[i]}")
        return a


# ------------------------------------------------------------------ replay

def replay(folder: Path, name: str, ckpt: str, args, records, state, obs,
           mission_csv: Path, mission_meta_path: Path, meta: dict):
    """한 체크포인트로 전 플랜을 다시 짓는다. (out_dir, diffs) 반환."""
    from rov_gui.perception import dp_policy

    pol = meta.get("policy") or {}
    cfg = pol.get("config") or {}
    tcp = pol.get("tcp") or {}
    T_bt = np.asarray(tcp["T_body_tcp"], float).reshape(4, 4)
    knot_dt = float(cfg["knot_dt_s"])
    w_open = float(cfg["gripper_width_open_m"])
    w_closed = float(cfg["gripper_width_closed_m"])

    # action_repr 는 체크포인트 자신의 계약에서 온다 (생성자 인자가 아니다).
    sess = dp_policy.DpPolicySession(
        ckpt=ckpt, repo=args.repo, num_inference_steps=args.steps,
        weights=args.weights, device=args.device,
        eval_transforms=args.eval_transforms)
    sess.load()
    if not sess.ready:
        raise SystemExit(f"[fail] {name}: checkpoint did not load: {sess.error}")
    desc = sess.describe()
    sha = str(desc.get("ckpt_sha1_head") or "")
    print(f"  [{name}] {desc.get('model_name')}  weights={desc.get('weights')}  "
          f"batchnorm_kept={desc.get('encoder_batchnorm_kept')}  sha1={sha[:12]}")
    if name == "verify":
        # verify 가 런이 쓴 그 가중치가 아니면 비교는 의미가 없다. 조용히 넘기지 않는다.
        flown = str((records[0] or {}).get("ckpt_sha1") or "")
        if flown and sha and not flown.startswith(sha[:12]):
            print(f"  [warn] verify ckpt sha1 {sha[:12]} != 런이 쓴 {flown[:12]} — "
                  f"--ckpt verify=<그 체크포인트> 로 지정해야 비교가 성립한다")

    out_dir = folder.parent / f"{folder.name}__replay_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in (mission_csv, mission_meta_path):
        shutil.copy2(src, out_dir / src.name)

    made, skipped, diffs, rp_meas_diff = [], 0, [], []
    for rec in records:
        rows_t = rec.get("obs_rows_t")
        if not rows_t or len(rows_t) != 2:
            skipped += 1
            continue
        i_prev, i_now = (obs.row_for(float(rows_t[0])), obs.row_for(float(rows_t[1])))
        if i_prev is None or i_now is None:
            skipped += 1                       # 이 플랜의 관측은 기록되지 않았다
            continue
        img = np.stack([obs.image(i_prev), obs.image(i_now)], 0).astype(np.float32) / 255.0
        img = np.ascontiguousarray(img.transpose(0, 3, 1, 2))

        # 시계: 레코드가 같은 순간의 mono 와 mission 시각을 둘 다 들고 있다.
        off = float(rec["obs_t_mono"]) - float(rec["obs_t_rel"])
        t_prev, t_now = float(rows_t[0]) - off, float(rows_t[1]) - off
        low = lowdim_obs(state.eta_at(t_prev), state.eta_at(t_now),
                         state.eta_at(args.start_t), T_bt,
                         state.width_at(t_prev), state.width_at(t_now))

        obs_dict = {"camera0_depth": img}
        obs_dict.update(low)
        action, _ms = sess.predict(obs_dict, seed=args.seed)

        anchor = np.asarray(rec["anchor_pose_used"], float)
        t0 = float(rec["obs_t_rel"])
        action = np.asarray(action, np.float32)
        # 7열(pos_rpy_width)은 체크포인트 계약이 정하고, 기록 레코드의 action_repr(5-dim 런)
        # 은 그 계약과 다르므로 넘기지 않는다 — 폭이 곧 표현이다(compose_plan action_repr_of).
        rec_repr = str(rec.get("action_repr") or "") or None
        is7 = action.shape[-1] == ACTION_DIM_BY_REPR[ACTION_REPR_POS_RPY_WIDTH]
        kw = {}
        if is7:
            rec_repr = ACTION_REPR_POS_RPY_WIDTH
            kw = dict(track_rp=bool(args.attitude_track),
                      rp_max_rad=math.radians(float(args.rp_max_deg)))
        try:
            msg, info = compose_plan(
                action, anchor, anchor[3:5], T_bt,
                obs_dt=float(rec["obs_dt_s"]), knot_dt=knot_dt,
                t0=t0, plan_id=int(rec["plan_id"]), obs_t_rel=t0,
                w_open=w_open, w_closed=w_closed,
                action_repr=rec_repr, **kw)
        except ValueError as e:                 # NaN/degenerate: 네트워크의 실패
            skipped += 1
            print(f"    plan {rec.get('plan_id')}: compose rejected — {e}")
            continue
        except TypeError as e:
            if not kw:
                raise
            raise SystemExit("[fail] compose_plan does not take track_rp / rp_max_rad "
                             "(6-DoF 변형의 policy_frames 변경이 이 체크아웃에 없다): "
                             f"{e}")

        p = np.asarray(msg.p_ned, float)
        new = dict(rec)
        raw = {"p_ned": p.tolist(),
               "yaw": np.asarray(msg.yaw, float).tolist(),
               "g": (None if msg.g is None
                     else np.asarray(msg.g, float).tolist())}
        new.update({
            "raw": raw,
            "action_raw": np.asarray(action, float).tolist(),
            "n_knots": int(p.shape[1]),
            "t0": float(msg.t0), "dt": float(msg.dt),
            "replayed_from": str(ckpt), "replay_name": name,
        })
        if is7:
            # 6-DoF 변형 레코드 키(record_boundaries, 워커 plans.jsonl 과 같은 자리):
            # rp_tracked 항상; raw.rp 는 실린 플랜(2,J) — tracked 일 때만; rp_raw 는
            # 디코드된 원시 자세(2,K) 최상위(7-dim 이면 항상); 클립량은 있을 때.
            new["rp_tracked"] = bool(info.get("rp_tracked", False))
            msg_rp = getattr(msg, "rp", None)
            if msg_rp is not None:
                raw["rp"] = np.asarray(msg_rp, float).tolist()   # 최상위 rp 중복은 쓰지 않는다 (워커와 동일, 2026-09-26 검토)
            if info.get("rp_raw") is not None:
                new["rp_raw"] = np.asarray(info["rp_raw"], float).tolist()
            new["dropped_rp_deg"] = float(info.get("dropped_rp_deg", float("nan")))
            if "rp_clipped_deg" in info:
                new["rp_clipped_deg"] = float(info["rp_clipped_deg"])
                new["rp_clipped_n"] = int(info.get("rp_clipped_n", 0))
            rp_raw = info.get("rp_raw")
            if rp_raw is not None:
                rp_raw = np.asarray(rp_raw, float)
                meas = state.eta_at(t0)[3:5]
                rp_meas_diff.append(np.degrees(
                    np.arctan2(np.sin(rp_raw - meas[:, None]),
                               np.cos(rp_raw - meas[:, None]))))
        made.append(new)
        old = rec.get("raw") or {}
        if isinstance(old.get("p_ned"), list):
            o = np.asarray(old["p_ned"], float)
            if o.shape == p.shape:
                diffs.append(float(np.sqrt(np.mean((o - p) ** 2))))

    with (out_dir / "plans.jsonl").open("w") as fh:
        for r in made:
            fh.write(json.dumps(r) + "\n")
    print(f"  [{name}] {len(made)} plans -> {out_dir}"
          + (f"  (skipped {skipped})" if skipped else ""))
    if rp_meas_diff:
        # 7-dim ckpt 의 자세 요청 vs 그 순간의 측정 자세(CSV roll_deg/pitch_deg) — 정책이
        # 선체를 얼마나 기울이자고 했는지. 라벨은 손목·실기는 선체라 추종 성능이 아니다.
        D = np.concatenate(rp_meas_diff, axis=1)             # (2, sum K)
        rms = np.sqrt(np.mean(D ** 2, axis=1))
        print(f"  [{name}] roll/pitch requested vs measured at obs (deg, over "
              f"{D.shape[1]} raw knots): roll RMS {rms[0]:.2f} max {np.abs(D[0]).max():.2f} | "
              f"pitch RMS {rms[1]:.2f} max {np.abs(D[1]).max():.2f}  "
              f"(tracked={bool(args.attitude_track)}; requested-minus-measured, +pitch = nose-up NED)")
    return out_dir, diffs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="0908_175151 | 20260908/0908_175151 | 폴더 경로")
    ap.add_argument("--ckpt", action="append", default=[], metavar="NAME=PATH",
                    help="재생할 체크포인트. 여러 번 줄 수 있다")
    ap.add_argument("--verify", action="store_true",
                    help="런이 실제로 쓴 체크포인트로 재생해 기록된 플랜과 비교")
    ap.add_argument("--repo", default=None, help="UMI 코드 경로 (기본: state의 값)")
    ap.add_argument("--weights", default="model",
                    help="model | ema_model (기본 model — BatchNorm 결함, "
                         "KNOWN_ISSUES.md)")
    ap.add_argument("--steps", type=int, default=8,
                    help="DDIM steps (기본 8 = 0908 실기 설정)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-transforms", default="center")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start-t", type=float, default=0.0, metavar="T_TRAJ",
                    help="eta_start 를 읽을 미션 시각 (기본 0 = 미션 시작)")
    ap.add_argument("--attitude-track", action="store_true",
                    help="7-dim ckpt: compose_plan(track_rp=True) — 플랜이 roll/pitch 를 "
                         "싣는다 (기본은 dropped-and-logged: 레벨링, rp_raw 는 정보용)")
    ap.add_argument("--rp-max-deg", type=float, default=20.0,
                    help="--attitude-track 의 T1 클립 [예측 20, hw_mpc.yaml policy.rp_max_deg]")
    a = ap.parse_args(argv)

    if a.repo is None:
        from rov_gui.state import POLICY_UMI_REPO
        a.repo = POLICY_UMI_REPO

    folder = run_folder(a.run)
    mission_csv, mission_meta_path, meta = mission_of(folder)
    state = MissionState(mission_csv)
    obs = ObsStore(folder)
    records = [json.loads(l) for l in (folder / "plans.jsonl").open()]
    records = [r for r in records if "raw" in r and r.get("obs_rows_t")]
    print(f"{folder.name}: {len(records)} installed plans, "
          f"{len(obs.t)} recorded observations")

    jobs = []
    if a.verify:
        from rov_gui.state import POLICY_CKPT_DEFAULT
        jobs.append(("verify", POLICY_CKPT_DEFAULT))
    for spec in a.ckpt:
        if "=" not in spec:
            raise SystemExit(f"[fail] --ckpt wants NAME=PATH, got {spec!r}")
        n, _, p = spec.partition("=")
        jobs.append((n, p))
    if not jobs:
        raise SystemExit("[fail] nothing to do: pass --verify and/or --ckpt NAME=PATH")

    for name, ckpt in jobs:
        out, diffs = replay(folder, name, ckpt, a, records, state, obs,
                            mission_csv, mission_meta_path, meta)
        if name == "verify" and diffs:
            d = np.asarray(diffs)
            print(f"\n  VERIFY: 재구성 플랜 vs 기록된 플랜, knot 위치 RMS "
                  f"[m] over {len(d)} plans\n"
                  f"    p50 {np.median(d):.4f}   p90 {np.quantile(d, 0.9):.4f}   "
                  f"max {d.max():.4f}\n"
                  f"  이 값이 크면 아래 그림들의 차이는 체크포인트 차이가 아니라 "
                  f"low-dim 재구성 오차다.\n")
        print(f"  그리기: ~/miniforge3/envs/robust/bin/python -m "
              f"rov_gui.tools.plot_policy_map {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
