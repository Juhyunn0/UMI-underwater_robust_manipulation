#!/usr/bin/env python3
"""fstereo_two_process_bench.py — does putting FoundationStereo in its OWN
process (the lab's "depth server" proposal, 2026-10-11) change the policy
forward time on this GPU?

Same device (one RTX 5090), same two networks and the same inputs as
data/20260930/0930_220212/diag/policy_vs_fstereo_contention.py, which measured
the station's layout (ONE process, FS on a thread, policy on its own CUDA
stream): policy forward 19.3 ms alone -> 152.8 ms p50 beside FS back-to-back.

Arms (``--arms``):
  alone                 policy only (reference)
  inproc@<scale>@<hz>   one process: FS thread + policy loop  (the station today)
  twoproc@<scale>@<hz>  FS loop in a CHILD process, policy loop in this one
                        (the proposal, minus the frame transport)
``hz`` 0 = FS back-to-back (what --policy ran at scale 0.75 on 09-30: 75.7 ms
a frame > the 66.7 ms camera interval, so it never idles); otherwise FS is
paced to that rate (15 = the camera, 7.5 = the 09-30 pacing arm).

What this does NOT include: camera, GUI, vehicle, recorders, and — in the
two-process arm — any IPC: no depth frame crosses the process boundary, so
that arm is the BEST CASE a separate server could give the policy forward.
CUDA MPS is not started (the desktop display shares this GPU); the two-process
arm therefore measures the driver's default context time-slicing.

Usage (the GPU must be idle; refuses otherwise like fstereo_bench):
    P=~/miniforge3/envs/rovgui-pose/bin/python
    $P rov_gui/tools/fstereo_two_process_bench.py drive
    $P rov_gui/tools/fstereo_two_process_bench.py drive --arms alone,twoproc@0.5@15 --n 20
Writes rov_gui/tools/fstereo_bench_out/two_process_<stamp>.json and prints a table.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.chdir(REPO)

import numpy as np                                             # noqa: E402

CKPT = ("data/checkpoints/"
        "umi_depth_5d_peg5d_vitclip_9_27_9_28_20260929_012539.ckpt")
DEFAULT_ARMS = ("alone,inproc@0.75@0,twoproc@0.75@0,inproc@0.75@7.5,"
                "twoproc@0.75@7.5,inproc@0.5@15,twoproc@0.5@15")
OUT_DIR = REPO / "rov_gui" / "tools" / "fstereo_bench_out"


def q(a):
    a = np.asarray(a, float)
    if a.size == 0:
        return None
    return {"p10": round(float(np.percentile(a, 10)), 1),
            "p50": round(float(np.median(a)), 1),
            "p90": round(float(np.percentile(a, 90)), 1),
            "max": round(float(a.max()), 1), "n": int(a.size)}


# ------------------------------------------------------------------ FS side
def fs_setup(scale: float):
    from rov_gui.tools import fstereo_bench as fb
    from c3_camera.host_depth import bench_rig
    pairs = fb.load_pairs()
    rig = bench_rig((fb.MONO_W, fb.MONO_H), with_color=True, color_size=(400, 250))
    fs, _ = fb.session(iters=8, scale=scale, graph=True)
    for _ in range(3):
        fs.infer(*pairs[0], rig, out_size=(640, 360))
    return fs, pairs, rig


def fs_loop(fs, pairs, rig, period: float, stop: threading.Event, rec: list):
    """Append (t_wall_end, ms) per frame until `stop`."""
    i = 0
    while not stop.is_set():
        t = time.perf_counter()
        fs.infer(*pairs[i % len(pairs)], rig, out_size=(640, 360))
        i += 1
        dt = time.perf_counter() - t
        rec.append((time.time(), dt * 1e3))
        if period and period > dt:
            stop.wait(period - dt)


def cmd_fs(a) -> None:
    """Child process: FS server loop. Writes `ready` when warm, records every
    frame, stops when `stop_file` appears or `seconds` elapse."""
    fs, pairs, rig = fs_setup(a.scale)
    period = (1.0 / a.hz) if a.hz > 0 else 0.0
    stop = threading.Event()
    rec: list = []
    th = threading.Thread(target=fs_loop, args=(fs, pairs, rig, period, stop, rec), daemon=True)
    th.start()
    Path(a.ready).write_text(str(os.getpid()))
    t_end = time.time() + a.seconds
    while time.time() < t_end and not Path(a.stop_file).exists():
        time.sleep(0.1)
    stop.set()
    th.join()
    fs.close()
    json.dump({"pid": os.getpid(), "scale": a.scale, "hz_requested": a.hz,
               "frames": rec}, open(a.out, "w"))


# -------------------------------------------------------------- policy side
def policy_setup(steps: int):
    from rov_gui.control.geometry import MpcConfig
    from rov_gui.perception.dp_policy import (make_policy_session, ROT6D_IDENTITY,
                                              STUB_WIDTH_M)
    cfg = MpcConfig.load("config/hw_mpc.yaml")
    blk = dict(cfg.policy)
    blk["num_inference_steps"] = steps
    s = make_policy_session(blk, None, ckpt=CKPT)
    s.load(lambda *a: None)
    assert s.ready, s.error
    c = s.contract
    obs = {c["image_key"]: np.zeros((c["obs_horizon"],) + tuple(c["image_shape"]), np.float32)}
    for k, shp in c["lowdim_shapes"].items():
        row = np.zeros(shp, np.float32)
        if tuple(shp) == (6,):
            row[:] = ROT6D_IDENTITY
        elif tuple(shp) == (1,):
            row[:] = STUB_WIDTH_M
        obs[k] = np.repeat(row[None], c["obs_horizon"], axis=0)
    for _ in range(5):
        s.predict(obs, _count=False)
    return s, obs


def cmd_dp(a) -> None:
    """Policy loop. ``--fs-thread scale,hz`` adds the FS loop on a thread of
    THIS process (inproc arm); ``--wait-ready`` waits for a child FS server."""
    fs_rec: list = []
    stop = threading.Event()
    fs = None
    if a.fs_thread:
        scale, hz = (float(x) for x in a.fs_thread.split(","))
        fs, pairs, rig = fs_setup(scale)
        period = (1.0 / hz) if hz > 0 else 0.0
        th = threading.Thread(target=fs_loop, args=(fs, pairs, rig, period, stop, fs_rec),
                              daemon=True)
        th.start()
    s, obs = policy_setup(a.steps)
    if a.wait_ready:
        t0 = time.time()
        while not Path(a.wait_ready).exists():
            if time.time() - t0 > 180:
                sys.exit("FS server never became ready")
            time.sleep(0.1)
    time.sleep(1.5)                       # let the FS side settle at rate
    ms = []
    t_start = time.time()
    for _ in range(a.n):
        t = time.perf_counter()
        _, m = s.predict(obs, _count=False)
        ms.append(m)
        time.sleep(max(0.0, a.period - (time.perf_counter() - t)))
    t_end = time.time()
    if fs is not None:
        stop.set()
        th.join()
        fs.close()
    s.close()
    json.dump({"pid": os.getpid(), "steps": a.steps, "period_s": a.period,
               "t_start": t_start, "t_end": t_end, "policy_ms_raw": ms,
               "fs_frames_inproc": fs_rec}, open(a.out, "w"))


# ------------------------------------------------------------------ driver
def fs_stats(frames, t0, t1):
    """FS per-frame stats inside the policy's measurement window."""
    w = [(t, ms) for t, ms in frames if t0 <= t <= t1]
    if not w:
        return None, None
    hz = len(w) / max(1e-9, (t1 - t0))
    return q([ms for _, ms in w]), round(hz, 2)


def run_arm(arm: str, py: str, n: int, steps: int, tmp: Path) -> dict:
    me = str(Path(__file__).resolve())
    out_dp = tmp / f"{arm.replace('@', '_')}_dp.json"
    res = {"arm": arm}
    if arm == "alone":
        subprocess.run([py, me, "dp", "--n", str(n), "--steps", str(steps),
                        "--out", str(out_dp)], check=True)
        d = json.load(open(out_dp))
        res.update(policy_ms=q(d["policy_ms_raw"]), fs_ms=None, fs_hz=None)
        return res
    kind, scale, hz = arm.split("@")
    scale, hz = float(scale), float(hz)
    if kind == "inproc":
        subprocess.run([py, me, "dp", "--n", str(n), "--steps", str(steps),
                        "--fs-thread", f"{scale},{hz}", "--out", str(out_dp)], check=True)
        d = json.load(open(out_dp))
        fs_q, fs_hz = fs_stats(d["fs_frames_inproc"], d["t_start"], d["t_end"])
        res.update(policy_ms=q(d["policy_ms_raw"]), fs_ms=fs_q, fs_hz=fs_hz,
                   layout="one process: FS thread + policy (own CUDA stream)")
        return res
    if kind == "twoproc":
        ready = tmp / f"{arm}_ready"
        stop_file = tmp / f"{arm}_stop"
        out_fs = tmp / f"{arm.replace('@', '_')}_fs.json"
        for p in (ready, stop_file):
            if p.exists():
                p.unlink()
        child = subprocess.Popen([py, me, "fs", "--scale", str(scale), "--hz", str(hz),
                                  "--seconds", "240", "--ready", str(ready),
                                  "--stop-file", str(stop_file), "--out", str(out_fs)])
        try:
            subprocess.run([py, me, "dp", "--n", str(n), "--steps", str(steps),
                            "--wait-ready", str(ready), "--out", str(out_dp)], check=True)
        finally:
            stop_file.write_text("stop")
            child.wait(timeout=120)
        d = json.load(open(out_dp))
        f = json.load(open(out_fs))
        fs_q, fs_hz = fs_stats(f["frames"], d["t_start"], d["t_end"])
        res.update(policy_ms=q(d["policy_ms_raw"]), fs_ms=fs_q, fs_hz=fs_hz,
                   layout="two processes: FS child (own CUDA context) + policy here",
                   fs_pid=f["pid"], dp_pid=d["pid"])
        return res
    raise SystemExit(f"unknown arm {arm}")


def cmd_drive(a) -> None:
    from rov_gui.tools import fstereo_bench as fb
    fb.refuse_if_busy(a.force)
    py = sys.executable
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tmp = Path(a.tmp or (OUT_DIR / f"two_process_{stamp}_tmp"))
    tmp.mkdir(parents=True, exist_ok=True)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,utilization.gpu,memory.used",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    out = {"tool": "rov_gui/tools/fstereo_two_process_bench.py", "when": stamp, "gpu": gpu,
           "ckpt": CKPT, "fs": "iters 8, cuda graph, scale per arm", "n_per_arm": a.n,
           "steps": a.steps, "policy_period_s": 0.5, "mps": "not started",
           "note": ("no camera / GUI / vehicle / IPC; twoproc = best case for a separate "
                    "FS server (no frame transport); same inputs as "
                    "data/20260930/0930_220212/diag/policy_vs_fstereo_contention.py"),
           "arms": []}
    for arm in a.arms.split(","):
        arm = arm.strip()
        if not arm:
            continue
        print(f"== {arm}", flush=True)
        r = run_arm(arm, py, a.n, a.steps, tmp)
        out["arms"].append(r)
        print(json.dumps(r), flush=True)
        time.sleep(2.0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"two_process_{stamp}.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"\n{'arm':>18} | {'policy p50':>10} {'p90':>7} {'max':>7} | {'FS p50':>7} {'FS Hz':>6}")
    for r in out["arms"]:
        p = r["policy_ms"] or {}
        f = r.get("fs_ms") or {}
        print(f"{r['arm']:>18} | {p.get('p50', float('nan')):10.1f} {p.get('p90', float('nan')):7.1f} "
              f"{p.get('max', float('nan')):7.1f} | {f.get('p50', float('nan')):7.1f} "
              f"{(r.get('fs_hz') or float('nan')):6.2f}")
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fs")
    p.add_argument("--scale", type=float, required=True)
    p.add_argument("--hz", type=float, required=True)
    p.add_argument("--seconds", type=float, default=240)
    p.add_argument("--ready", required=True)
    p.add_argument("--stop-file", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_fs)
    p = sub.add_parser("dp")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--period", type=float, default=0.5)
    p.add_argument("--fs-thread", default="")
    p.add_argument("--wait-ready", default="")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_dp)
    p = sub.add_parser("drive")
    p.add_argument("--arms", default=DEFAULT_ARMS)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--tmp", default="")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_drive)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
