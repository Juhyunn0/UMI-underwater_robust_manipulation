#!/usr/bin/env python3
"""
policy_dryrun.py — the live policy mission end to end, offline, with the
REAL checkpoint (or the stub): the strongest check there is without a vehicle.

    P=~/miniforge3/envs/rovgui-pose/bin/python
    QT_QPA_PLATFORM=offscreen $P rov_gui/tools/policy_dryrun.py                # real ckpt, 40 s
    QT_QPA_PLATFORM=offscreen $P rov_gui/tools/policy_dryrun.py --ckpt stub    # no torch
    QT_QPA_PLATFORM=offscreen $P rov_gui/tools/policy_dryrun.py --seconds 60 --mpc-mode pid

Equivalent to ``./c3 gui --source demo --mpc --policy`` driven the way an
operator would (MANUAL -> ENABLE -> ARM -> shape policy -> START), for N
seconds, then STOP TRAJ and disengage. Real MainWindow, real demo backend,
real MpcWorker with the real controller (acados dobmpc by default), the REAL
PolicyWorker with the checkpoint on the GPU, the demo's synthetic depth on the
identity grid. What it reports:

* when the worker became ready (checkpoint load + warm-up),
* plans EMITTED by the worker (its counters) vs INSTALLED / CLIPPED /
  REJECTED / LATE by the controller (the run's ``plans.jsonl`` + meta),
* inference time p50 / max, the pairing statistics, the skip counters,
* PASS / FAIL against spec v2 §C.3: worker ready < 30 s, >= 20 plans
  emitted, infer_ms p50 reported, zero worker errors, the reject fraction
  REPORTED (not gated — the demo plant is fabricated, and a rejection of a
  real policy's chunk against the real limits is information, not a bug).

When the controller ESCALATES (3 consecutive geometric rejects -> the mission
halts on the last plan's endpoint and the worker idles, spec v2 A7) the
driver does what the log tells the operator to do — STOP TRAJ, then START
re-arms — and counts it, so a 40 s dryrun keeps exercising the pipeline and
the report says how often the REAL policy's chunks were refused. On the
synthetic scene that happened at once on 2026-09-02 (peak speed 0.126 m/s
against v_max 0.08, successive chunks 7-12 cm apart against jump_max 0.06 —
out-of-distribution input, in-distribution safety). ``--no-rearm`` keeps the
first halt.

Everything here is SYNTHETIC (demo plant, demo depth) except the network:
the plans are the real policy's answer to a fabricated scene, which says
nothing about grasping and everything about whether the pipeline moves. The
record lands under ``--rec-dir`` (default ``data``) as its own KIND —
``data/YYYYMMDD/MMDD_HHMMSS_dryrun/`` — NEVER a water folder (spec v2 A15;
rov_gui/runstore.py), and the folder is printed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np                                        # noqa: E402

from rov_gui.qt import QtWidgets, QTimer                  # noqa: E402

READY_MAX_S = 30.0
PLANS_MIN = 20


class Opts:
    """The CLI namespace the station reads (demo_e2e's, plus --policy)."""
    source = "demo"
    mpc = True
    policy = True
    policy_ckpt = None
    policy_repo = None
    policy_allow_device_depth = False
    policy_allow_fs_mismatch = False
    fstereo = False
    fstereo_iters = 16
    fstereo_scale = 1.0
    fstereo_size = None
    fstereo_alpha = 0.5
    depth_scale = 0.64
    fps = 15.0
    ui_fps = 30.0
    thrusters = 8
    rec_dir = "data"
    run_kind = "dryrun"          # the leaf suffix; MpcWorker._run_tree honours it
    rec_fps = 12.0
    fullscreen = False
    joystick = "none"
    nav_config = str(REPO / "config/hw_nav.yaml")
    mpc_config = str(REPO / "config/hw_mpc.yaml")
    nav_geometry = None
    mpc_mode = "dobmpc"
    rov_model = "heavy_gripper"
    imu_dr = None
    imu_dr_attitude = None
    imu_dr_control = False
    c3_imu_rate = 200
    demo_imu_bias = None
    demo_imu_gyro_bias = None
    pose = False
    demo_object = "still"
    replay_session = None


def _read_plans(run_dir: Path) -> list[dict]:
    p = run_dir / "plans.jsonl"
    if not p.is_file():
        return []
    out = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return out


def _read_meta(run_dir: Path) -> dict:
    for name in ("controller.json",):
        p = run_dir / name
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    metas = sorted(run_dir.glob("*.meta.json"))
    if metas:
        return json.loads(metas[-1].read_text(encoding="utf-8"))
    return {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=40.0,
                    help="how long the policy mission runs after START "
                         "(default: %(default)s)")
    ap.add_argument("--ckpt", default=None, metavar="PATH|stub",
                    help="override hw_mpc.yaml policy.ckpt ('stub' = no torch)")
    ap.add_argument("--repo", default=None, help="override policy.repo")
    ap.add_argument("--mpc-mode", default="dobmpc",
                    choices=("dobmpc", "mpc", "dobmpc_tuned", "mpc_tuned", "pid", "rl"))
    ap.add_argument("--rec-dir", default="data",
                    help="ROOT of the dated run tree the record lands in "
                         "(default: %(default)s); the leaf is always "
                         "MMDD_HHMMSS_dryrun, never a water folder")
    ap.add_argument("--stop", default="stop", choices=("stop", "disengage"),
                    help="how the run ends after --seconds: STOP TRAJ then "
                         "disengage (default), or disengage straight away")
    ap.add_argument("--no-rearm", dest="rearm", action="store_false",
                    help="do NOT re-arm after an escalation halt (the run "
                         "then ends at the first halt; default re-arms and "
                         "counts)")
    a = ap.parse_args(argv)

    Opts.policy_ckpt = a.ckpt
    Opts.policy_repo = a.repo
    Opts.mpc_mode = a.mpc_mode
    # ONE folder per invocation. The run store JOINS an in-progress folder
    # of the same kind (memory: rov-gui-run-folder), so two dryruns a
    # minute apart wrote one plans.jsonl and the second report counted the
    # first run's plans. Opening a FRESH `_dryrun` folder now (join=False)
    # keeps every record attributable: it is the newest of its kind, so the
    # writers that follow join it and not the previous invocation's.
    from rov_gui import runstore
    Opts.rec_dir = str(a.rec_dir)
    run_folder = runstore.run_dir(Opts.rec_dir, kind=Opts.run_kind, join=False)
    print(f"policy_dryrun: record -> {run_folder} (SYNTHETIC run; its own "
          f"kind, never a water folder)  ckpt={a.ckpt or 'hw_mpc.yaml policy.ckpt'} "
          f"mode={a.mpc_mode} seconds={a.seconds:g}")

    from rov_gui.backends import make_backend
    from rov_gui.window import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow(Opts())
    backend = make_backend("demo", win.bus, win.mailboxes, Opts())
    assert backend.policy is not None and backend.mpc is not None
    statuses, pstatus, plans, logs, pstates = [], [], [], [], []
    win.bus.mpc_status.connect(statuses.append)
    win.bus.policy_status.connect(pstatus.append)
    win.bus.policy_plan.connect(plans.append)
    win.bus.policy_state.connect(pstates.append)
    win.bus.log.connect(lambda lvl, msg: logs.append((lvl, msg)))
    win.attach(backend)
    t_start = time.monotonic()
    state = {"phase": "boot", "t0": t_start, "ready_at": None, "traj_started": None,
             "run_dir": None, "end_reason": "", "worker_errors": [],
             "escalations": 0, "halt_reasons": [], "rearm_at": None}

    def step():
        t = time.monotonic() - state["t0"]
        s = statuses[-1] if statuses else None
        ps = pstatus[-1] if pstatus else None
        if state["ready_at"] is None and ps is not None and ps.ready:
            state["ready_at"] = t
            print(f"[{t:5.1f} s] policy worker READY")
        if ps is not None and ps.error and ps.error not in state["worker_errors"]:
            state["worker_errors"].append(ps.error)
            print(f"[{t:5.1f} s] policy worker ERROR: {ps.error}")
        ph = state["phase"]
        if ph == "boot" and t > 3.0:
            win.bus.cmd_mode.emit("MANUAL")
            win.bus.cmd_enable.emit(True)
            state["phase"] = "arm"
        elif ph == "arm" and t > 4.0:
            win.bus.cmd_arm.emit(True)
            win.bus.cmd_mpc_scenario.emit({"shape": "policy"})
            state["phase"] = "engage"
        elif ph == "engage":
            # START only once the worker is ready (the controller refuses
            # otherwise) — that wait IS the ready-time measurement.
            if state["ready_at"] is not None and t > 6.0:
                win.bus.cmd_mpc_start.emit()
                state["phase"] = "warmup"
                state["start_at"] = t
            elif t > READY_MAX_S + 6.0:
                print(f"FAIL: the policy worker was not ready after "
                      f"{READY_MAX_S:g} s "
                      f"({ps.error or ps.note if ps else 'no status'})")
                app.quit()
        elif ph == "warmup":
            if s is not None and s.traj_on:
                state["phase"] = "traj"
                # Stamped ONCE: the --seconds budget counts from the FIRST
                # start, not from every re-arm (a sliding deadline never ends
                # a run the controller keeps halting).
                if state["traj_started"] is None:
                    state["traj_started"] = t
                mpc = backend.mpc
                state["run_dir"] = str(mpc._run_dir()) if mpc._csv_path else None
                print(f"[{t:5.1f} s] policy mission RUNNING ({s.reason})")
            elif t > state["start_at"] + 25.0:
                print(f"FAIL: START did not reach the policy mission. last: "
                      f"{s.reason if s else 'no status'}")
                state["end_reason"] = s.reason if s else ""
                app.quit()
        elif ph == "rearm":
            if t >= state["rearm_at"]:
                win.bus.cmd_mpc_start.emit()
                state["phase"] = "warmup"
                state["start_at"] = t
        elif ph == "traj":
            halted = bool(pstates and pstates[-1].halted)
            if halted:
                state["escalations"] += 1
                why = s.reason if s else ""
                state["halt_reasons"].append(why)
                print(f"[{t:5.1f} s] HALTED by the controller (#"
                      f"{state['escalations']}): {why}")
                if a.rearm and t < state["traj_started"] + a.seconds:
                    # What the log tells the operator: STOP TRAJ, START re-arms.
                    win.bus.cmd_mpc_traj.emit(False)
                    state["phase"] = "rearm"
                    state["rearm_at"] = t + 0.6
                    return
            if s is not None and not s.engaged:
                print(f"[{t:5.1f} s] DISENGAGED mid-run: {s.reason}")
                state["end_reason"] = s.reason
                state["phase"] = "done"
                QTimer.singleShot(800, app.quit)
            elif s is not None and not s.traj_on:
                why = str(s.reason)
                # The A21 divergence guard ENDS the mission (holding the
                # measured pose) instead of latching it, and on the fabricated
                # plant a 2.4-2.7x-dilated chunk from an out-of-distribution
                # scene trips it within ~10 s (2026-09-02: 2 of 3 real-ckpt
                # dryruns stopped there and the report then read "only 6 plans
                # emitted", which says nothing about the pipeline). What the
                # operator is told is the same as after a latch - START
                # re-arms - so the driver does that and COUNTS it separately,
                # keeping the guard's rate visible instead of truncating the
                # run.
                if ("diverged" in why and a.rearm
                        and t < state["traj_started"] + a.seconds):
                    state["divergences"] = state.get("divergences", 0) + 1
                    state["halt_reasons"].append(why)
                    print(f"[{t:5.1f} s] DIVERGENCE stop by the controller (#"
                          f"{state['divergences']}): {why} - re-arming")
                    state["phase"] = "rearm"
                    state["rearm_at"] = t + 0.6
                    return
                print(f"[{t:5.1f} s] mission ended on its own: {why}")
                state["end_reason"] = why
                state["phase"] = "closing"
                win.bus.cmd_mpc_engage.emit(False)
                QTimer.singleShot(800, app.quit)
            elif t > state["traj_started"] + a.seconds:
                state["end_reason"] = f"operator {a.stop} after {a.seconds:g} s"
                print(f"[{t:5.1f} s] {a.stop.upper()} after {a.seconds:g} s")
                if a.stop == "stop":
                    win.bus.cmd_mpc_traj.emit(False)
                win.bus.cmd_mpc_engage.emit(False)
                state["phase"] = "closing"
                QTimer.singleShot(1200, app.quit)

    drv = QTimer()
    drv.setInterval(100)
    drv.timeout.connect(step)
    drv.start()
    QTimer.singleShot(int((a.seconds + 90.0) * 1000), app.quit)
    app.exec_() if hasattr(app, "exec_") else app.exec()
    # The worker's meta is read BEFORE the threads stop (a stopped session
    # has nothing left to describe); the backend is then stopped properly so
    # no QTimer dies on a foreign thread at interpreter exit.
    wmeta = backend.policy.meta()
    win.shutdown()

    # ---------------------------------------------------------------- report
    run_dir = Path(state["run_dir"]) if state["run_dir"] else None
    rec = _read_plans(run_dir) if run_dir else []
    meta = _read_meta(run_dir) if run_dir else {}
    pol = meta.get("policy") or {}
    run = pol.get("run") or {}
    verdicts = {}
    for r in rec:
        v = str(r.get("verdict") or r.get("status") or "?")
        verdicts[v] = verdicts.get(v, 0) + 1
    emitted = int(wmeta["counters"]["plans"])
    # plans.jsonl covers EVERY mission of this invocation (re-arms append);
    # the meta's policy.run block is the LAST mission only, so the fractions
    # come from the verdict counts and the meta block is shown beside them.
    installed = int(verdicts.get("accept", 0) + verdicts.get("clip", 0))
    rejected = int(verdicts.get("reject", 0))
    late = int(sum(1 for r in rec if "late" in str(r.get("status", ""))))
    offered = sum(verdicts.values())
    infer = wmeta.get("infer_ms") or {}
    pair = wmeta.get("pairing") or {}
    c = wmeta["counters"]
    # Every mission of this invocation appends to the same plans.jsonl
    # (re-arms), so plan_id restarts; count the rejects by verdict.
    skips = {k: v for k, v in c.items() if k.startswith("skip_") and v}
    sess = wmeta.get("session") or {}

    print()
    print(f"run folder        : {run_dir}")
    print(f"session           : {'STUB' if sess.get('stub') else sess.get('ckpt')} "
          f"(sha1 {sess.get('ckpt_sha1_head') or '-'}; eval_transforms "
          f"{sess.get('eval_transforms')}; steps {sess.get('num_inference_steps')})")
    print(f"worker ready      : {state['ready_at'] if state['ready_at'] is not None else 'NEVER'} s "
          f"after launch (load {sess.get('load_s', 0.0):.1f} s, warm-up "
          f"{sess.get('warmup_ms', float('nan')):.0f} ms)")
    print(f"plans emitted     : {emitted}  (worker, {wmeta.get('plan_hz', 0.0):.2f} Hz)")
    print(f"plans offered     : {offered}  verdicts {verdicts}  (plans.jsonl)")
    print(f"installed/rejected: {installed} / {rejected}  late {late}  "
          f"(all missions of this invocation, plans.jsonl); LAST mission per "
          f"meta policy.run: {json.dumps({k: run.get(k) for k in ('installed', 'clipped', 'rejected', 'late', 'skip_bridge', 'end_reason')})}")
    frac = (rejected / offered) if offered else float("nan")
    print(f"reject fraction   : {frac:.2f}  (REPORTED, not gated)")
    print(f"infer_ms          : p50 {infer.get('p50')}  max {infer.get('max')}  n {infer.get('n')}")
    print(f"pairing           : dt p50 {pair.get('dt_p50_s')} s, max {pair.get('dt_max_s')} s, "
          f"dup {pair.get('dup')}, fallback {pair.get('fallback')}, near {pair.get('near')}, "
          f"skip_pair {pair.get('skip')}, skip_fix_lag {pair.get('skip_fix_lag')} "
          f"(fallback <= {pair.get('fallback_max')}·obs_dt, fix lag <= "
          f"{pair.get('fix_lag_tol')}·obs_dt; trigger: {pair.get('trigger')})")
    print(f"skips             : {skips or 'none'}   (ALL skip_* counters; a skip "
          f"retries on the next depth frame, it does not burn a period)")
    print(f"depth             : {wmeta.get('depth_src')} @ {wmeta.get('depth_hz', 0.0):.1f} fps, "
          f"coverage {(wmeta.get('obs') or {}).get('coverage')}")
    print(f"worker errors     : {state['worker_errors'] or 'none'} "
          f"(infer_errors {c.get('infer_errors', 0)})")
    print(f"escalations       : {state['escalations']}  "
          f"divergence stops: {state.get('divergences', 0)} "
          f"({'re-armed each time' if a.rearm else 'no re-arm'})"
          + (f" — {state['halt_reasons'][-1][:110]}" if state["halt_reasons"] else ""))
    print(f"end reason        : {state['end_reason']!r}")
    errs = [s.err_xy for s in statuses if s.traj_on and s.err_xy is not None]
    if errs:
        e = np.array(errs)
        print(f"tracking (demo)   : err_xy mean {e.mean() * 100:.1f} cm, p95 "
              f"{np.percentile(e, 95) * 100:.1f} cm — FABRICATED plant, not a figure")

    fails = []
    if state["ready_at"] is None or state["ready_at"] > READY_MAX_S:
        fails.append(f"worker ready {state['ready_at']} s (limit {READY_MAX_S:g})")
    if emitted < PLANS_MIN:
        fails.append(f"only {emitted} plans emitted (need >= {PLANS_MIN})")
    if infer.get("p50") is None:
        fails.append("infer_ms p50 not reported")
    if state["worker_errors"] or c.get("infer_errors", 0):
        fails.append(f"worker errors {state['worker_errors']} / infer_errors "
                     f"{c.get('infer_errors', 0)}")
    if state["phase"] not in ("done", "closing"):
        fails.append(f"ended in phase {state['phase']}")
    if run_dir is None or not rec:
        fails.append("no plans.jsonl in the run folder")
    print()
    if fails:
        print("FAIL: " + "; ".join(fails))
        return 1
    print(f"PASS: policy pipeline moved end to end for {a.seconds:g} s "
          f"(SYNTHETIC scene; reject fraction {frac:.2f} and "
          f"{state['escalations']} escalation(s) REPORTED, not gated)")
    if state["escalations"]:
        print("NOTE: the controller refused the policy's chunks against the "
              "hw_mpc.yaml policy: limits on a FABRICATED scene — read the "
              "reasons in plans.jsonl before touching a limit; out-of-"
              "distribution input is the expected cause here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
