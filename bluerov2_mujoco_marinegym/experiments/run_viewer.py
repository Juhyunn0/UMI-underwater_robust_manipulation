#!/usr/bin/env python3
"""Real-time MuJoCo viewer for ONE (controller, mode) running the square mission.

Fix one seed + one current direction, then watch a single controller
(pid | mpc | dobmpc) track the square under one disturbance mode (C/CD/CW/CDW)
LIVE in MuJoCo. The physics is built with the SAME `build()` / `DisturbanceEnv`
as the headless batch (`experiments/run_compare.py`), so the trajectory you watch
is identical to the analysed results.

Run ONE combination per invocation (pick --ctrl and --mode):

  python -m experiments.run_viewer --config config/base.yaml --ctrl dobmpc --mode CDW

Outputs (accumulate in one folder, so 12 calls = 4 modes x 3 ctrls land together):
  recordings/<YYYYMMDD>/square_view/
    traj_square_<mode>_<ctrl>_seed<seed>_dir<deg>.csv   # full run, position+reference
    lap_square_<mode>_<ctrl>_seed<seed>_dir<deg>.mp4    # ONLY the recorded lap (last)
    meta_square_<mode>_<ctrl>_seed<seed>_dir<deg>.json  # reproduction metadata

The whole run is `--laps` laps (default = config square laps, 10); only ONE lap
(default the last, settled lap) is captured to mp4 to keep the file small.

  --headless   no on-screen window; offscreen render only (display-less check /
               deterministic video fallback if the GLFW viewer + offscreen
               Renderer clash on this machine).
  --no-arrows  skip the live current/wave/thrust force arrows.
  --no-video   skip the mp4 (CSV only).
"""
import argparse
import csv
import dataclasses
import json
import os
import sys
import time

import numpy as np
# Pick an offscreen GL backend for the video Renderer when there is no display
# (headless server / --headless smoke), BEFORE importing mujoco so its GL context
# honours it. With a display present we leave the default (GLFW) for both the live
# viewer and the offscreen render.
if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import mujoco.viewer   # import is display-safe; only launch*() needs a display

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # marinegym dir
sys.path.insert(0, HERE)

import hydro as H
import water_viz as WV     # animated pool water surface (VISUAL ONLY; POOL_TAGS scene)
import thrusters as Tt
from disturbance.config import load_config
from .run_compare import (build, make_controller, square_setpoint, slew_heading,
                          make_square_ref)
# teleop's overlay drawers (module import is display-safe: no viser/pyqtgraph at top)
from teleop import _draw_plan, draw_force_arrows


# --------------------------------------------------------------- helpers
def _square_block(cfg):
    """The first experiment block whose scenario is 'square' (primary/secondary/dp_later)."""
    exp = cfg.experiment
    for key in ("primary", "secondary", "dp_later"):
        b = exp.get(key)
        if isinstance(b, dict) and b.get("scenario") == "square":
            return b
    return {}


def _square_corners(size, depth):
    """CCW square corner polyline (closed), (5,3), for the path overlay."""
    return np.array([[0.0, 0.0, depth], [size, 0.0, depth], [size, size, depth],
                     [0.0, size, depth], [0.0, 0.0, depth]], float)


def _resolve_record_lap(spec, laps):
    if spec == "last":
        return laps - 1
    if spec == "first":
        return 0
    if spec == "middle":
        return laps // 2
    return max(0, min(int(spec), laps - 1))


class _ArrowShim:
    """Minimal stand-in for the teleop object that force_items()/draw_force_arrows()
    need: they only read `.B` (the allocation matrix, for the net-thrust arrow)."""
    def __init__(self, model):
        self.B, _ = Tt.allocation_matrix(model)


class LapVideo:
    """Offscreen mp4 recorder for a single lap. Renders with mujoco.Renderer and
    writes frames via cv2.VideoWriter (mp4v). Created eagerly so any GL failure
    surfaces BEFORE the interactive viewer opens; disables itself (CSV unaffected)
    on any error so live viewing never crashes because of the recorder."""

    def __init__(self, model, *, size, speed, depth, record_lap, out_path,
                 video_hz, width, height):
        self.model = model
        self.size = float(size)
        self.speed = float(speed)
        self.record_lap = int(record_lap)
        self.out_path = out_path
        self.video_hz = float(video_hz)
        self.width = int(width)
        self.height = int(height)
        self.corners = _square_corners(size, depth)
        self.n = 0
        self._last_t = -1e9
        self.enabled = False
        self.writer = None
        self.renderer = None
        self.cam = None
        self._surf = None          # animated water surface (VISUAL ONLY), injected by run()
        self._field = None
        try:
            import cv2
            self._cv2 = cv2
            self.renderer = mujoco.Renderer(model, self.height, self.width)
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [0.5 * size, 0.5 * size, depth]
            cam.distance = 3.0 * size + 1.5
            cam.azimuth = 90.0
            cam.elevation = -35.0
            self.cam = cam
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(out_path, fourcc, self.video_hz,
                                          (self.width, self.height))
            if not self.writer.isOpened():
                raise RuntimeError("cv2.VideoWriter failed to open")
            self.enabled = True
        except Exception as e:                                 # noqa: BLE001
            print(f"[run_viewer] video DISABLED ({type(e).__name__}: {e}). "
                  f"CSV still saved; for a display-less machine try MUJOCO_GL=egl "
                  f"or run with --headless.")
            self._safe_release()

    def capture(self, data):
        if not self.enabled:
            return
        lap = int((self.speed * data.time) // (4.0 * self.size))
        if lap != self.record_lap:
            return
        if data.time - self._last_t < 1.0 / self.video_hz:
            return
        self._last_t = data.time
        try:
            self.renderer.update_scene(data, self.cam)
            _draw_plan(self.renderer.scene, self.corners)      # draw the square in-frame
            if self._surf is not None and self._field is not None:  # animate water (VISUAL ONLY)
                self._surf.update(self._field, data.time,
                                  enabled=getattr(self._field, "enabled", False),
                                  renderer=self.renderer)
            frame = self.renderer.render()                     # (H,W,3) uint8 RGB
            self.writer.write(self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2BGR))
            self.n += 1
        except Exception as e:                                 # noqa: BLE001
            print(f"[run_viewer] video capture error ({type(e).__name__}: {e}); "
                  f"disabling video.")
            self.enabled = False
            self._safe_release()

    def _safe_release(self):
        try:
            if self.writer is not None:
                self.writer.release()
        except Exception:                                      # noqa: BLE001
            pass
        try:
            if self.renderer is not None:
                self.renderer.close()
        except Exception:                                      # noqa: BLE001
            pass

    def close(self):
        had = self.n
        self._safe_release()
        if had == 0 and os.path.exists(self.out_path):
            os.remove(self.out_path)                           # no frames -> no stub file
        return had


# --------------------------------------------------------------- the run
def run(model, data, hydro, ctrl, bid, *, size, speed, depth, laps, T_run,
        arrows, video, csv_path, log_hz, headless, heading_follow=True, yaw_rate=None):
    dt = float(model.opt.timestep)
    n_sub = max(1, round((1.0 / 60.0) / dt))      # ~60 fps render cadence
    shim = _ArrowShim(model) if arrows else None
    surf = WV.make_surface_from_env(model)         # live animated water (None unless hfield scene)
    if video is not None:                          # give the offscreen recorder its own surface
        video._surf = WV.make_surface_from_env(model)
        video._field = hydro.disturbance
    corners = _square_corners(size, depth)
    log_dt = 1.0 / float(log_hz)
    yaw_rate = np.radians(60.0) if yaw_rate is None else float(yaw_rate)
    yaw_cmd = [0.0]                                # slewed heading ref (square starts +x)

    f = open(csv_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["t", "px", "py", "pz", "rx", "ry", "yaw_deg", "pitch_deg", "lap"])
    last_log = [-log_dt]

    def step_once():
        (rx, ry), (tx, ty) = square_setpoint(data.time, size, speed)
        r_cmd = 0.0
        if heading_follow:
            yaw_new = slew_heading(yaw_cmd[0], tx, ty, yaw_rate, dt)
            r_cmd = (yaw_new - yaw_cmd[0]) / dt        # slew rate = yaw-rate ref
            yaw_cmd[0] = yaw_new
        ctrl.set_target((rx, ry, depth), yaw_ref=(yaw_cmd[0] if heading_follow else 0.0),
                        v_ref=(speed * tx, speed * ty, 0.0), r_ref=r_cmd,
                        yaw_target=(np.arctan2(ty, tx) if heading_follow else 0.0))
        ctrl.apply(model, data)
        mujoco.mj_step(model, data)
        return rx, ry

    def record(rx, ry):
        if video is not None:
            video.capture(data)
        if data.time - last_log[0] >= log_dt:
            last_log[0] = data.time
            p = np.asarray(data.xpos[bid], float)
            R = np.asarray(data.xmat[bid], float).reshape(3, 3)
            yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
            pitch = np.degrees(-np.arcsin(np.clip(R[2, 0], -1.0, 1.0)))
            lap = int((speed * data.time) // (4.0 * size))
            w.writerow([f"{data.time:.4f}", f"{p[0]:.5f}", f"{p[1]:.5f}",
                        f"{p[2]:.5f}", f"{rx:.5f}", f"{ry:.5f}",
                        f"{yaw:.3f}", f"{pitch:.3f}", lap])

    t_start = time.time()
    if headless:
        while data.time < T_run:
            rx, ry = step_once()
            record(rx, ry)
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = [0.5 * size, 0.5 * size, depth]
            viewer.cam.distance = 3.0 * size + 1.5
            viewer.cam.azimuth = 90.0
            viewer.cam.elevation = -35.0
            while viewer.is_running() and data.time < T_run:
                t0 = time.time()
                rx = ry = 0.0
                for _ in range(n_sub):
                    rx, ry = step_once()
                    record(rx, ry)
                    if data.time >= T_run:
                        break
                if arrows:
                    draw_force_arrows(viewer.user_scn, hydro, shim, data, bid)
                else:
                    viewer.user_scn.ngeom = 0
                _draw_plan(viewer.user_scn, corners)           # yellow square path
                if surf is not None and hydro.disturbance is not None:  # animate water (VISUAL ONLY)
                    surf.update(hydro.disturbance, data.time,
                                enabled=getattr(hydro.disturbance, "enabled", False),
                                viewer=viewer)
                viewer.sync()
                slack = n_sub * dt - (time.time() - t0)
                if slack > 0:
                    time.sleep(slack)
    f.close()
    n_frames = video.close() if video is not None else 0
    return time.time() - t_start, n_frames


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description="Real-time MuJoCo viewer: one controller x one mode on the square.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ctrl", required=True, choices=("pid", "mpc", "dobmpc"))
    ap.add_argument("--mode", required=True, choices=("NONE", "C", "CD", "CW", "CDW"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dir-deg", type=float, default=0.0,
                    help="fixed current heading [deg] (only the current rotates).")
    ap.add_argument("--laps", type=int, default=None, help="override square laps.")
    ap.add_argument("--size", type=float, default=None, help="override square edge [m].")
    ap.add_argument("--speed", type=float, default=None, help="override speed [m/s].")
    ap.add_argument("--record-lap", default="last",
                    help="lap to capture to mp4: last | first | middle | <int>.")
    ap.add_argument("--heading", choices=("follow", "fixed"), default="follow",
                    help="follow = face travel direction (yaw->path tangent); fixed = yaw 0.")
    ap.add_argument("--yaw-rate", type=float, default=60.0,
                    help="heading slew-rate limit [deg/s] for smooth corner turns.")
    ap.add_argument("--video-hz", type=float, default=30.0)
    ap.add_argument("--video-size", default="720x480", help="WIDTHxHEIGHT.")
    ap.add_argument("--no-arrows", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--headless", action="store_true",
                    help="no on-screen window; offscreen render + CSV only.")
    ap.add_argument("--out", default=None, help="output dir (default recordings/<date>/square_view).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sb = _square_block(cfg)
    size = float(args.size if args.size is not None else sb.get("size", 1.0))
    speed = float(args.speed if args.speed is not None else sb.get("speed", 0.15))
    depth = float(sb.get("depth", 0.0))
    laps = int(args.laps if args.laps is not None else sb.get("laps", 10))
    log_hz = float(cfg.sim.get("log_hz", 20.0))
    T_run = laps * (4.0 * size) / speed
    record_lap = _resolve_record_lap(args.record_lap, laps)

    # fix ONE current direction (waves' beta_bar stays put; only the current rotates)
    dist = dataclasses.replace(cfg.dist, theta_c=float(np.radians(args.dir_deg)))

    # output folder + filenames (accumulate across the 12 combos)
    out_dir = args.out or os.path.join(
        HERE, "recordings", time.strftime("%Y%m%d"), "square_view")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"square_{args.mode}_{args.ctrl}_seed{args.seed}_dir{int(round(args.dir_deg))}"
    csv_path = os.path.join(out_dir, f"traj_{tag}.csv")
    mp4_path = os.path.join(out_dir, f"lap_{tag}.mp4")
    meta_path = os.path.join(out_dir, f"meta_{tag}.json")

    print(f"[run_viewer] {args.ctrl} | mode {args.mode} | seed {args.seed} | "
          f"dir {args.dir_deg:.1f} deg | {laps} laps x {4*size:.1f} m / {speed} m/s "
          f"= {T_run:.0f} s | record lap {record_lap} "
          f"({'headless' if args.headless else 'live viewer'})")
    print(f"[run_viewer] out -> {out_dir}")

    # same plant/hydro/env/controller as the batch (build() uninstalls stale cb first)
    model, data, hydro, env, bid = build(cfg, args.mode, args.seed, args.ctrl, T_run, dist)
    ctrl = make_controller(args.ctrl, model, hydro)
    ctrl.reset()
    if args.ctrl in ("mpc", "dobmpc"):
        # tracking mode (same wiring as run_compare.run_one): the NMPC samples the
        # TRUE future square reference over its horizon instead of extrapolating.
        horizon_s = ctrl.nmpc.N * ctrl.ctrl_dt
        ctrl.set_reference_traj(make_square_ref(
            size, speed, depth, args.heading == "follow",
            np.radians(args.yaw_rate), float(model.opt.timestep),
            T_run + horizon_s + 1.0))
    data.qpos[:3] = [0.0, 0.0, depth]                          # start at the origin corner
    mujoco.mj_forward(model, data)

    width, height = (int(x) for x in args.video_size.lower().split("x"))
    video = None
    if not args.no_video:
        video = LapVideo(model, size=size, speed=speed, depth=depth,
                         record_lap=record_lap, out_path=mp4_path,
                         video_hz=args.video_hz, width=width, height=height)

    wall, n_frames = run(model, data, hydro, ctrl, bid, size=size, speed=speed,
                         depth=depth, laps=laps, T_run=T_run, arrows=not args.no_arrows,
                         video=video, csv_path=csv_path, log_hz=log_hz,
                         headless=args.headless,
                         heading_follow=(args.heading == "follow"),
                         yaw_rate=np.radians(args.yaw_rate))

    # metadata sidecar
    meta = dict(scenario="square", controller=args.ctrl, mode=args.mode,
                seed=args.seed, dir_deg=args.dir_deg, size=size, speed=speed,
                depth=depth, laps=laps, T_run=T_run, record_lap=record_lap,
                heading=args.heading, yaw_rate_deg_s=args.yaw_rate,
                ref_preview=(args.ctrl in ("mpc", "dobmpc")),   # tracking-mode provenance
                dt=float(model.opt.timestep), log_hz=log_hz,
                video=dict(hz=args.video_hz, width=width, height=height,
                           frames=n_frames, path=os.path.basename(mp4_path) if n_frames else None),
                config=os.path.abspath(args.config),
                started=time.strftime("%Y-%m-%d %H:%M:%S"), wall_s=round(wall, 1))
    try:
        meta["disturbance"] = env.to_meta()
    except Exception:                                          # noqa: BLE001
        pass
    with open(meta_path, "w") as mf:
        json.dump(meta, mf, indent=2, default=str)

    H.Hydrodynamics.uninstall()
    print(f"[run_viewer] done in {wall:.1f}s wall | CSV {os.path.basename(csv_path)} | "
          f"mp4 {n_frames} frames" + ("" if n_frames else " (none)"))


if __name__ == "__main__":
    main()
