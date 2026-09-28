#!/usr/bin/env python3
"""
policy_bench_check.py — the POOL depth path of the policy mission, on the bench:
real C3, real FoundationStereo, real PolicyWorker, NO vehicle commands.

    P=~/miniforge3/envs/rovgui-pose/bin/python
    QT_QPA_PLATFORM=offscreen $P rov_gui/tools/policy_bench_check.py --seconds 45

Equivalent to ``./c3 gui --source hw --fstereo --policy --mpc
--mavlink-transport none`` (no ``--allow-command``: the NullCommandSink
transmits nothing; no telemetry is read either) held for N seconds and then
shut down. Nothing is engaged, so the policy never infers on the vehicle —
what this exercises is exactly the half no offline test can:

* ``C3VideoWorker._build_rig`` with the policy's alpha (0.5 under --policy),
  and the live CAM_B K/D stashed for the recalibration guard;
* ``FStereoWorker`` at the training instrument settings (iters 16, scale 1.0)
  publishing ``depth_native`` + rig to the ``PolicyMailbox``;
* ``PolicyWorker`` building the ``RectLeftGrid`` from the live rig —
  coverage of the 224 obs, the live-vs-yaml model check, the fingerprint —
  and its depth ring / status while idle;
* the checkpoint load + warm-up on this GPU beside FoundationStereo.

Reported (all [측정] for THIS bench session; the folder path is printed):
grid kind / coverage / model_check / fingerprint, FS measured Hz + solve ms +
alpha + k_mm_px, depth frames delivered to the policy mailbox, worker ready
time, the depth-pair spacing the measured rate implies (spec v2 A18: which of
near / fallback / skip a regular frame train at that rate lands in, and the
ratio to the 66.7 ms training stride), and PASS/FAIL against: rig built, grid
usable (coverage >= policy.min_obs_coverage), model_check not FAILED, FS Hz >
0, mailbox puts > 0, worker ready, zero worker errors. The report is written
as JSON beside the log so the numbers can be cited by path. Reference run:
7.26 Hz, solve 137 ms, panel latency 243 ms [측정: rov_gui/tools/
fstereo_bench_out/policy_bench_20260902_141224.json, C3 실기, alpha 0.5, scale
1.0, iters 16, 330 frames, GPU shared with a concurrent test run].
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rov_gui.qt import QtWidgets, QTimer                  # noqa: E402


def pair_expectation(measured_hz, obs_dt_s: float, *, tol=None, fallback=None) -> dict:
    """What the A18 pairing does to a REGULAR frame train at ``measured_hz``.

    The partner is the frame nearest ``t_d - obs_dt`` within ±tol·obs_dt
    ('near'); else the nearest OLDER frame within fallback·obs_dt ('fallback');
    else 'skip'. For frames k·Δ back (Δ = 1/hz) the nearest-to-target distance
    is min_k |k·Δ − obs_dt|; the fallback partner is the newest frame with
    k·Δ > tol·obs_dt beyond the target and k·Δ ≤ fallback·obs_dt."""
    from rov_gui.backends.policy import PAIR_FALLBACK, PAIR_TOL
    tol = PAIR_TOL if tol is None else float(tol)
    fallback = PAIR_FALLBACK if fallback is None else float(fallback)
    out = {"measured_hz": measured_hz, "obs_dt_s": obs_dt_s,
           "frame_spacing_s": None, "mode": "no rate", "pair_dt_s": None,
           "ratio_to_training_stride": None,
           "tol": tol, "fallback": fallback}
    if not measured_hz or float(measured_hz) <= 0.0:
        return out
    d = 1.0 / float(measured_hz)
    out["frame_spacing_s"] = d
    target = obs_dt_s
    ks = range(1, 64)
    k_near = min(ks, key=lambda k: abs(k * d - target))
    if abs(k_near * d - target) <= tol * obs_dt_s:
        out["mode"], out["pair_dt_s"] = "near", k_near * d
    else:
        far = [k for k in ks if k * d > target + tol * obs_dt_s
               and k * d <= fallback * obs_dt_s]
        if far:
            out["mode"], out["pair_dt_s"] = "fallback", min(far) * d
        else:
            out["mode"] = "skip"
    if out["pair_dt_s"] is not None:
        out["ratio_to_training_stride"] = out["pair_dt_s"] / obs_dt_s
    return out


class Opts:
    """The station's CLI namespace for a camera-only, command-free run."""
    source = "hw"
    ip = "192.168.2.191"
    mxid = None
    fps = 30.0
    isp_scale = "1/3"
    mjpeg_quality = 80
    depth_fps = 15.0
    depth_size = "640x360"
    depth = True
    panel2 = "stereo"            # no ROV RGB worker: keep this to the C3
    rov_cam_host = "192.168.2.1"
    rov_cam_fps = 30.0
    rov_cam_backend = "auto"
    rov_cam_size = "1280x720"
    rov_cam_port = 5600
    mavlink = "udpin:0.0.0.0:14551"
    mavlink_transport = "none"   # NOTHING sent to or read from the vehicle
    mavlink_rest_url = "http://192.168.2.2:6040"
    blueos_host = "192.168.2.2"
    water_density = 997.0
    mavlink_set_rates = False
    mavlink_rate = 200
    c3_imu = True
    c3_imu_rate = 200
    imu_dr = None
    imu_dr_attitude = None
    imu_dr_control = False
    battery_capacity_mah = 0.0
    battery_cells = 4
    thrusters = 8
    allow_command = False        # NullCommandSink
    mavlink_out = "udpin:0.0.0.0:14552"
    target_sysid = 1
    cmd_sysid = 255
    deadman_ms = 500
    z_convention = "0..1000"
    tilt_servo = -1
    tilt_min_deg = -45.0
    tilt_max_deg = 45.0
    arm_mode = "MANUAL"
    lights_servo = 13
    lights_steps = 8
    joystick = "none"
    js_deadzone = 0.08
    js_scale = 1.0
    js_translate = True
    js_passthrough = True
    js_remap = "11:0,12:15"
    fstereo = True
    fstereo_repo = None
    fstereo_ckpt = None
    fstereo_iters = None         # resolved by rov_gui.__main__.resolve_defaults
    fstereo_scale = None
    fstereo_size = None
    fstereo_alpha = None
    fstereo_graph = True
    fstereo_fill = 2
    fstereo_align_size = "400x250"
    pose = False
    depth_scale = 0.64
    mpc = True                   # MpcWorker emits PolicyState (idle, never engaged)
    nav_config = str(REPO / "config/hw_nav.yaml")
    mpc_config = str(REPO / "config/hw_mpc.yaml")
    nav_geometry = None
    mpc_mode = "dobmpc"
    rov_model = None
    replay_session = None
    policy = True
    policy_ckpt = None
    policy_repo = None
    policy_allow_device_depth = False
    policy_allow_fs_mismatch = False
    ui_fps = 30.0
    fullscreen = False
    rec_dir = "data"
    run_kind = "dryrun"          # the leaf suffix; MpcWorker._run_tree honours it
    rec_fps = 12.0
    demo_object = "still"
    demo_leak_after = None
    no_preflight = True
    preflight_only = False
    preflight_strict = False
    preflight_timeout = 3.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--ckpt", default=None, help="override policy.ckpt (or 'stub')")
    ap.add_argument("--out-dir", default="rov_gui/tools/fstereo_bench_out")
    ap.add_argument("--fstereo-alpha", type=float, default=None)
    ap.add_argument("--fstereo-scale", type=float, default=None)
    ap.add_argument("--fstereo-iters", type=int, default=None)
    a = ap.parse_args(argv)

    from rov_gui.__main__ import check_policy, resolve_defaults
    opts = Opts()
    if a.ckpt:
        opts.policy_ckpt = a.ckpt
    if a.fstereo_alpha is not None:
        opts.fstereo_alpha = a.fstereo_alpha
    if a.fstereo_scale is not None:
        opts.fstereo_scale = a.fstereo_scale
    if a.fstereo_iters is not None:
        opts.fstereo_iters = a.fstereo_iters
    resolve_defaults(opts)
    chk = check_policy(opts)
    print(f"policy_bench_check: fstereo iters={opts.fstereo_iters} "
          f"scale={opts.fstereo_scale} alpha={opts.fstereo_alpha}; "
          f"check_policy -> {chk}")
    if chk and chk[0] == "refuse":
        print("REFUSED by check_policy; nothing opened")
        return 2

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"policy_bench_{stamp}.log"
    json_path = out_dir / f"policy_bench_{stamp}.json"

    from rov_gui.backends import make_backend
    from rov_gui.window import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow(opts)
    backend = make_backend("hw", win.bus, win.mailboxes, opts)
    assert backend.policy is not None and backend.fstereo is not None
    logs, pstatus, vstats, sstats = [], [], {}, {}
    t0 = time.monotonic()

    def _log(lvl, msg):
        line = f"[{time.monotonic() - t0:6.1f} s] {lvl:5s} {msg}"
        logs.append(line)
        print(line, flush=True)

    win.bus.log.connect(_log)
    win.bus.policy_status.connect(pstatus.append)
    win.bus.video_stat.connect(lambda st: vstats.__setitem__(st.name, st))
    win.bus.sensor_stat.connect(lambda st: sstats.__setitem__(st.name, st))
    win.attach(backend)
    QTimer.singleShot(int(a.seconds * 1000), app.quit)
    app.exec_() if hasattr(app, "exec_") else app.exec()

    wmeta = backend.policy.meta()
    fmeta = backend.fstereo.meta()
    ready = [s for s in pstatus if s.ready]
    errors = sorted({s.error for s in pstatus if s.error})
    fs_stat = vstats.get("depth")
    win.shutdown()

    obs = wmeta.get("obs") or {}
    grid_kind = obs.get("grid_kind") or obs.get("grid", {}).get("kind") \
        if isinstance(obs, dict) else None
    coverage = obs.get("coverage") if isinstance(obs, dict) else None
    model_check = (obs.get("model_check") or {}).get("status") \
        if isinstance(obs, dict) else None
    min_cov = None
    try:
        from rov_gui.control.geometry import MpcConfig
        min_cov = float(MpcConfig.load(opts.mpc_config).policy["min_obs_coverage"])
    except Exception as e:                                   # noqa: BLE001
        print(f"note: could not read min_obs_coverage ({e})")

    report = {
        "when": stamp, "seconds": a.seconds,
        "fstereo": {"iters": opts.fstereo_iters, "scale": opts.fstereo_scale,
                    "alpha": opts.fstereo_alpha,
                    "measured_hz": fmeta.get("measured_hz"),
                    "frames": fmeta.get("frames"),
                    "cuda_graph": fmeta.get("cuda_graph"),
                    "rig": fmeta.get("rig"),
                    "panel_stat": (fs_stat.note if fs_stat else None),
                    "panel_fps": (fs_stat.fps if fs_stat else None),
                    "panel_latency_ms": (fs_stat.latency_ms if fs_stat else None)},
        "policy_worker": wmeta,
        "policy_status_last": (pstatus[-1].__dict__ if pstatus else None),
        "worker_ready_after_s": (ready[0].stamp - t0) if ready else None,
        "worker_errors": errors,
        "sensor_rows": {k: (v.hz, str(v.conn), v.detail) for k, v in sstats.items()},
        "log": logs,
    }
    json_path.write_text(json.dumps(report, indent=1, default=str))
    log_path.write_text("\n".join(logs) + "\n")

    counters = wmeta.get("counters") or {}
    mailbox = wmeta.get("mailbox") or {}
    checks = {
        "rig_built": bool(fmeta.get("rig")),
        "fs_frames_gt_0": int(fmeta.get("frames") or 0) > 0,
        "grid_rect_left": (str(grid_kind) == "rect_left"),
        "grid_usable": bool((obs.get("usable") if isinstance(obs, dict) else False)),
        "coverage_ok": (coverage is not None and min_cov is not None
                        and float(coverage) >= min_cov),
        "model_check_not_failed": not str(model_check or "").startswith("FAIL"),
        "mailbox_puts_gt_0": int(mailbox.get("put", 0)) > 0,
        "worker_ready": bool(ready),
        "no_worker_errors": not errors,
    }
    print("\n==================== policy_bench_check ====================")
    print(f"log  -> {log_path}\njson -> {json_path}")
    print(f"FS: iters {opts.fstereo_iters} scale {opts.fstereo_scale} alpha "
          f"{opts.fstereo_alpha}  measured {fmeta.get('measured_hz')} Hz  "
          f"frames {fmeta.get('frames')}  panel '{fs_stat.note if fs_stat else None}'")
    print(f"grid: kind={grid_kind} coverage={coverage} min={min_cov} "
          f"model_check={model_check}")
    obs_dt = float(wmeta.get("obs_dt_s") or (2.0 / 30.0))
    pe = pair_expectation(fmeta.get("measured_hz"), obs_dt)
    report["pair_expectation"] = pe
    json_path.write_text(json.dumps(report, indent=1, default=str))
    if pe["frame_spacing_s"] is not None:
        print(f"pairing at {float(fmeta.get('measured_hz')):.2f} Hz: frame spacing "
              f"{pe['frame_spacing_s'] * 1e3:.0f} ms vs obs_dt {obs_dt * 1e3:.1f} ms "
              f"-> every pair is '{pe['mode']}'"
              + (f", obs_pair_dt {pe['pair_dt_s'] * 1e3:.0f} ms = "
                 f"{pe['ratio_to_training_stride']:.2f}x the training stride"
                 if pe["pair_dt_s"] is not None else
                 f" (no older frame within {pe['fallback']:g}·obs_dt)")
              + f"  [near ±{pe['tol']:g}·obs_dt, fallback ≤ {pe['fallback']:g}·obs_dt]")
    else:
        print("pairing: no measured FS rate — no pair expectation")
    print(f"mailbox: {mailbox}   counters: {counters}")
    print(f"worker ready after: {report['worker_ready_after_s']}  errors: {errors}")
    for k, v in checks.items():
        print(f"  {'ok  ' if v else 'FAIL'} {k}")
    ok = all(checks.values())
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
