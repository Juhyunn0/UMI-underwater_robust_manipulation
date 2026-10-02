#!/usr/bin/env python3
"""policy_fs_schedule_bench.py — ``--policy-fs-schedule`` measured offline, on
the REAL workers.

    P=~/miniforge3/envs/rovgui-pose/bin/python
    QT_QPA_PLATFORM=offscreen $P rov_gui/tools/policy_fs_schedule_bench.py
    QT_QPA_PLATFORM=offscreen $P rov_gui/tools/policy_fs_schedule_bench.py \\
        --arms free@0.75,yield@0.75,only@0.75,free@0.5 --seconds 40

WHAT RUNS. The station's own ``FStereoWorker`` and ``PolicyWorker``, each on
its own QThread, joined by the same ``PolicyMailbox`` and (when the arm's
schedule is not ``free``) the same ``FsGate`` that ``HardwareBackend`` builds —
with the real FoundationStereo network and the real policy checkpoint on the
GPU. So this exercises the code the vehicle runs, not a model of it: the gate
calls in both ticks, the burst trigger, the mailboxes, the plan record.

WHAT IS SYNTHETIC — and therefore what the numbers do NOT say:
  * the stereo pairs are 9 recorded land pairs replayed at the camera's 15 fps
    (``fstereo_bench.load_pairs``), stamped ``t_capture = now`` when put: there
    is no camera transport latency in any age reported here [the vehicle's
    capture -> depth-ready was 0.196 s p50: data/20260930/0930_220212/diag/
    latency_breakdown.json], so ages are "since the pair reached the host";
  * the rig is ``c3_camera.host_depth.bench_rig`` (alpha 0.5), not the C3's;
  * the vehicle state is a constant-speed pose fed at 20 Hz, always a fresh
    fix; nothing consumes the plans (no controller, no intake, no vehicle);
  * no GUI window, no video panels, no recorders: the process is quieter than
    the station, so contention is, if anything, understated.
It measures what the schedule changes — the policy forward's wall time, the
wait for the frame in flight, the depth frame rate, the pair spacing, the plan
interval — under otherwise identical conditions. It says nothing about how the
policy flies.

One arm per subprocess (a fresh CUDA context each): ``<schedule>@<scale>``.
Writes one JSON under ``--out-dir`` (default rov_gui/tools/fstereo_bench_out)
and prints a table. Opens no camera, no vehicle link, no serial port.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np                                             # noqa: E402

DEFAULT_CKPT = ("data/checkpoints/"
                "umi_depth_5d_peg5d_vitclip_9_27_9_28_20260929_012539.ckpt")
DEFAULT_ARMS = "free@0.75,yield@0.75,only@0.75,free@0.5,yield@0.5"
CAMERA_FPS = 15.0
STATE_HZ = 20.0


class Opts:
    """What the two workers read from the CLI namespace (everything else is
    a getattr default inside them)."""
    source = "hw"
    fstereo = True
    fstereo_repo = None
    fstereo_ckpt = None
    fstereo_iters = 8
    fstereo_scale = 0.75
    fstereo_size = None
    fstereo_alpha = 0.5
    fstereo_graph = True
    fstereo_fill = 2
    fstereo_align_size = "400x250"
    fstereo_view = "native"
    depth_fps = CAMERA_FPS
    policy = True
    policy_ckpt = None
    policy_repo = None
    policy_fs_schedule = "free"
    policy_allow_device_depth = False
    policy_allow_fs_mismatch = False
    record_depth = False
    record_stereo = False
    land_dry_run = False
    nav_config = str(REPO / "config/hw_nav.yaml")
    mpc_config = str(REPO / "config/hw_mpc.yaml")
    nav_geometry = None


def _q(vals, digits=1):
    v = np.asarray([x for x in vals if x is not None], float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"p10": None, "p50": None, "p90": None, "max": None, "n": 0}
    return {"p10": round(float(np.percentile(v, 10)), digits),
            "p50": round(float(np.percentile(v, 50)), digits),
            "p90": round(float(np.percentile(v, 90)), digits),
            "max": round(float(v.max()), digits), "n": int(v.size)}


def run_arm(arm: str, seconds: float, ckpt: str, warm_s: float) -> dict:
    """One schedule@scale, in this process. Returns the result dict."""
    schedule, _, scale = arm.partition("@")
    scale = float(scale or 0.75)

    from rov_gui.qt import QtWidgets
    from rov_gui.bus import DataBus, FrameMailbox, PolicyMailbox, StereoMailbox
    from rov_gui.backends.hardware import FStereoWorker
    from rov_gui.backends.policy import PolicyWorker
    from rov_gui.perception.fs_gate import FsGate
    from rov_gui.state import PolicyState, now
    from rov_gui.tools import fstereo_bench as fb
    from c3_camera.host_depth import bench_rig

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    busy = fb.gpu_busy()
    opts = Opts()
    opts.fstereo_scale = scale
    opts.policy_fs_schedule = schedule
    opts.policy_ckpt = str(ckpt)

    pairs = fb.load_pairs()
    # alpha 0.5 = the --policy rectification (the obs crop is not covered at
    # 0); the colour size is only the panel grid here.
    rig = bench_rig((fb.MONO_W, fb.MONO_H), with_color=True,
                    color_size=(640, 360), alpha=0.5)

    bus = DataBus()
    smb, pmb = StereoMailbox(), PolicyMailbox()
    # A depth panel mailbox, so the stereo thread does the colourise/copy it
    # does in the station (CPU on that thread, between frames).
    fsw = FStereoWorker(bus, smb, opts, {"depth": FrameMailbox("depth")})
    pw = PolicyWorker(bus, pmb, opts)
    # exactly HardwareBackend.__init__'s wiring
    fsw.policy_mb = pmb
    pw.fstereo_meta_fn = fsw.meta
    gate = None
    if schedule != "free":
        gate = FsGate(schedule)
        fsw.fs_gate = gate
        pw.fs_gate = gate

    plans, logs, statuses = [], [], []
    bus.policy_state.connect(pw.on_policy_state)
    bus.policy_plan.connect(plans.append)
    bus.policy_status.connect(statuses.append)
    bus.log.connect(lambda lvl, msg: logs.append((lvl, msg)))
    fsw.failed.connect(lambda m: logs.append(("error", f"fstereo: {m}")))
    pw.failed.connect(lambda m: logs.append(("error", f"policy: {m}")))

    fsw.start()
    pw.start()

    t_begin = now()
    state = {"active": False, "n_pair": 0, "n_state": 0,
             "t_pair": t_begin, "t_state": t_begin, "epoch": 1}

    def pump(until, active):
        """Feed pairs at CAMERA_FPS and vehicle states at STATE_HZ, and run
        this thread's event loop (the plan signal is queued to it)."""
        state["active"] = active
        while now() < until:
            t = now()
            if t >= state["t_pair"]:
                L, R = pairs[state["n_pair"] % len(pairs)]
                smb.put(L, R, rig, t, frame_seq=state["n_pair"],
                        out_size=(640, 360))
                state["n_pair"] += 1
                state["t_pair"] += 1.0 / CAMERA_FPS
                if state["t_pair"] < t:                 # fell behind: no burst
                    state["t_pair"] = t + 1.0 / CAMERA_FPS
            if t >= state["t_state"]:
                x = 0.02 * (t - t_begin)                # 2 cm/s, +x
                bus.policy_state.emit(PolicyState(
                    t=t, t_fix=t, eta=(x, 0.0, 0.8, 0.0, 0.0, 0.1),
                    fix_fresh=True, epoch=state["epoch"], active=bool(active),
                    halted=False, t0_traj=t_begin, eta_start=(0.0,) * 6,
                    grip_width_m=0.007, engaged=True, stamp=t))
                state["n_state"] += 1
                state["t_state"] += 1.0 / STATE_HZ
                if state["t_state"] < t:
                    state["t_state"] = t + 1.0 / STATE_HZ
            app.processEvents()
            time.sleep(0.0005)

    # ---- load: both sessions ready, the graph captured, the grid adopted
    deadline = now() + 180.0
    while now() < deadline:
        pump(now() + 0.25, active=False)
        fs_ok = bool(fsw.session is not None and fsw.session.ready
                     and fsw._frames >= 3)
        pol_ok = bool(pw.session is not None and pw.session.ready
                      and pw.builder is not None and pw.builder.usable)
        if fs_ok and pol_ok:
            break
        if (fsw.session is not None and fsw.session.error) or (
                pw.session is not None and pw.session.error):
            break
    ready = {"fs_ready": bool(fsw.session is not None and fsw.session.ready),
             "fs_error": str(getattr(fsw.session, "error", "") or ""),
             "policy_ready": bool(pw.session is not None and pw.session.ready),
             "policy_error": str(getattr(pw.session, "error", "") or ""),
             "builder_usable": bool(pw.builder is not None and pw.builder.usable),
             "builder_why": str(getattr(pw.builder, "why", "")),
             "load_s": round(now() - t_begin, 1)}
    result = {"arm": arm, "schedule": schedule, "fstereo_scale": scale,
              "ckpt": str(ckpt), "seconds": seconds, "ready": ready,
              "gpu_other_processes_at_start": busy,
              # CPU load is part of the conditions: the policy's observation
              # build and both ticks are Python on the CPU.
              "loadavg_at_start": [round(v, 2) for v in os.getloadavg()],
              "cpu_count": os.cpu_count()}
    if not (ready["fs_ready"] and ready["policy_ready"] and ready["builder_usable"]):
        result["error"] = "workers did not come up"
        pw.stop()
        fsw.stop()
        return result

    # ---- idle warm-up (mission not active: FS free under every schedule)
    f0, t0 = int(fsw._frames), now()
    pump(now() + warm_s, active=False)
    idle_hz = (int(fsw._frames) - f0) / max(1e-6, now() - t0)

    # ---- the mission
    n_before = len(plans)
    f0, t0 = int(fsw._frames), now()
    pump(now() + seconds, active=True)
    t1 = now()
    fs_frames = int(fsw._frames) - f0
    run = plans[n_before:]
    # the first plans carry the mission start (free-running frames, the first
    # burst); report steady state from the third on, and say how many
    steady = run[2:] if len(run) > 4 else run

    # ---- mission over: does FS come back?
    f1, t2 = int(fsw._frames), now()
    pump(now() + 2.0, active=False)
    after_hz = (int(fsw._frames) - f1) / max(1e-6, now() - t2)

    fs_meta = fsw.meta()
    pol_meta = pw.meta()
    pw.stop()
    fsw.stop()

    emit = [p.t_emit for p in steady]
    wait = [p.fs_wait_ms for p in steady]
    result.update({
        "n_plans": len(run), "n_steady": len(steady),
        "plan_interval_s": _q(np.diff(emit) if len(emit) > 1 else [], 3),
        "infer_ms": _q([p.infer_ms for p in steady]),
        "fs_wait_ms": _q(wait),
        "fs_wait_timeouts": int(sum(1 for p in steady if p.fs_wait_timeout)),
        "pair_dt_ms": _q([1e3 * p.obs_pair_dt_s for p in steady]),
        "pair_consecutive_pct": (round(100.0 * float(np.mean(
            [abs(p.obs_pair_dt_s - 1.0 / CAMERA_FPS) < 0.02 for p in steady])), 1)
            if steady else None),
        "pair_dup": int(sum(1 for p in steady if p.pair_dup)),
        # ages since the pair reached the host (no camera latency here)
        "depth_ready_age_ms": _q([1e3 * (p.depth_arrive_t - p.obs_t)
                                  for p in steady if p.depth_arrive_t is not None]),
        "trigger_age_ms": _q([1e3 * (p.trigger_t - p.obs_t)
                              for p in steady if p.trigger_t is not None]),
        "emit_age_ms": _q([1e3 * (p.t_emit - p.obs_t) for p in steady]),
        "attempt_ms": _q([1e3 * (p.t_emit - p.trigger_t)
                          for p in steady if p.trigger_t is not None]),
        "fs_frames_per_s_mission": round(fs_frames / max(1e-6, t1 - t0), 2),
        "fs_frames_per_s_idle_before": round(idle_hz, 2),
        "fs_frames_per_s_idle_after": round(after_hz, 2),
        "fs_measured_hz_meta": fs_meta.get("measured_hz"),
        "fs_infer_size": fs_meta.get("infer_size_actual"),
        "plan_fs_schedule": sorted({p.fs_schedule for p in run}),
        "plan_fs_input": sorted({p.fs_input for p in run}),
        "policy_counters": dict(pol_meta.get("counters") or {}),
        "policy_pairing": pol_meta.get("pairing"),
        "policy_fs_schedule_meta": pol_meta.get("fs_schedule"),
        "fstereo_schedule_meta": fs_meta.get("schedule"),
        "loadavg_at_end": [round(v, 2) for v in os.getloadavg()],
        "stereo_mailbox": smb.counters(),
        "policy_mailbox": pmb.counters(),
        "warn_error_logs": [f"{lvl}: {msg}"[:200] for lvl, msg in logs
                            if lvl in ("warn", "error")][:20],
    })
    return result


def _table(results) -> str:
    def g(r, key, sub="p50"):
        v = r.get(key)
        if isinstance(v, dict):
            v = v.get(sub)
        return "-" if v is None else v
    rows = [("arm", "plans", "interval s", "infer ms p50/p90", "wait ms p50/p90",
             "emit age ms p50/p90", "pair ms p50", "consec %", "FS /s")]
    for r in results:
        if r.get("error"):
            rows.append((r["arm"], "ERROR: " + r["error"], "", "", "", "", "", "", ""))
            continue
        rows.append((r["arm"], str(r["n_plans"]), str(g(r, "plan_interval_s")),
                     f"{g(r, 'infer_ms')}/{g(r, 'infer_ms', 'p90')}",
                     f"{g(r, 'fs_wait_ms')}/{g(r, 'fs_wait_ms', 'p90')}",
                     f"{g(r, 'emit_age_ms')}/{g(r, 'emit_age_ms', 'p90')}",
                     str(g(r, "pair_dt_ms")), str(r.get("pair_consecutive_pct")),
                     str(r.get("fs_frames_per_s_mission"))))
    w = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w[i]) for i, c in enumerate(row)) for row in rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default=DEFAULT_ARMS,
                    help="comma list of <schedule>@<fstereo scale> "
                         "(default: %(default)s)")
    ap.add_argument("--seconds", type=float, default=30.0,
                    help="mission time per arm (default: %(default)s)")
    ap.add_argument("--warm", type=float, default=3.0)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out-dir", default="rov_gui/tools/fstereo_bench_out")
    ap.add_argument("--one", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--json-out", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    if a.one:                                    # the child: one arm
        r = run_arm(a.one, a.seconds, a.ckpt, a.warm)
        Path(a.json_out).write_text(json.dumps(r, indent=1, default=str))
        return 0 if not r.get("error") else 1

    os.chdir(REPO)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    results = []
    for arm in [s.strip() for s in a.arms.split(",") if s.strip()]:
        tmp = out_dir / f".policy_fs_schedule_{stamp}_{arm.replace('@', '_')}.json"
        print(f"--- {arm} ({a.seconds:.0f} s) ...", flush=True)
        rc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--one", arm,
             "--seconds", str(a.seconds), "--warm", str(a.warm),
             "--ckpt", a.ckpt, "--json-out", str(tmp)],
            cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True)
        if tmp.exists():
            results.append(json.loads(tmp.read_text()))
            tmp.unlink()
        else:
            results.append({"arm": arm, "error": f"child exited {rc.returncode}",
                            "tail": rc.stdout[-1500:]})
            print(rc.stdout[-1500:])
    out = {
        "tool": "rov_gui/tools/policy_fs_schedule_bench.py",
        "when": stamp,
        "what": ("real FStereoWorker + PolicyWorker threads, real networks; "
                 "synthetic pairs at 15 fps, synthetic vehicle state, no "
                 "camera latency, no GUI, no controller — see the module "
                 "docstring"),
        "camera_fps": CAMERA_FPS, "state_hz": STATE_HZ,
        "arms": results,
    }
    path = out_dir / f"policy_fs_schedule_{stamp}.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    print()
    print(_table(results))
    print(f"\n-> {path}")
    return 0 if all(not r.get("error") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
