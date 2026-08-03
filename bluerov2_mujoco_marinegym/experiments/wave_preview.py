#!/usr/bin/env python3
"""Offscreen ~10 s MP4 preview of the CDW pool environment.

Renders the POOL_TAGS scene (AprilTag floor + ROV + animated water surface) so you
can eyeball how the waves defined by a `waves:` block actually look and act — the same
view teleop/run_viewer show live, captured to a short clip.

The physics IS stepped, but with NO controller (thrusters at 0): the ROV is released
from rest and left to be carried by the real current + wave (Froude-Krylov) forces,
so the clip shows the true effect of the disturbance on the vehicle. The camera is
fixed on the pool, so the ROV may drift out of frame over the clip — that is expected
and fine (it is what "free-drift under the sea state" looks like). Only base_link sees
hydro; the pool floor / water surface are non-colliding visual geoms.

Used two ways:
  * run_compare calls render_wave_preview(...) once per wave config when CDW (or CW)
    is in the modes, writing wave_preview.mp4 into that run's output folder.
  * standalone, to (re)generate previews without a full compare:
      python -m experiments.wave_preview --config config/base.yaml
      python -m experiments.wave_preview --config config/base.yaml --wave 1 --duration 8

The clip fixes ONE heading (waves rotate per sweep point; a preview just needs one),
so `--theta-deg` / `--beta-deg` (or the first sweep point, when called from the batch)
pick the current + wave heading shown.

Wave surface: by DEFAULT the water surface uses the PHYSICALLY FAITHFUL wavenumber
(no exaggeration) — the same look the live viewer shows. Real ocean wavelengths dwarf
the ~8.5 m pool, so the faithful surface reads as a gentle heave/tilt; the honest
signal of the sea state is mostly in how the ROV is pushed around. To exaggerate the
ripples for a stylised clip, set WATER_WAVE_LAMBDA (visual wavelength in metres) and/or
WATER_WAVE_AMP (amplitude gain) — both render-only (physics wavenumber/amplitude
untouched), read by water_viz.make_surface_from_env.
"""
import argparse
import dataclasses
import os
import sys

import numpy as np
# Offscreen GL backend when there is no display (headless server), BEFORE importing
# mujoco so its GL context honours it. With a display we leave the default.
if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # marinegym dir
sys.path.insert(0, HERE)

import rov_model as RM
import water_viz as WV
import hydro as H
from disturbance.env import DisturbanceEnv
from disturbance.config import load_config

WATER_GEOM = WV.WATER_GEOM   # "pool_water_surface"


def _water_frame(model):
    """(cx, cy, cz, span) of the pool water geom for camera framing, or None if the
    scene has no water surface geom."""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, WATER_GEOM)
    if gid < 0:
        return None
    cx, cy, cz = (float(v) for v in model.geom_pos[gid])
    span = 4.0
    if int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_HFIELD):
        hfid = int(model.geom_dataid[gid])
        hx, hy = (float(v) for v in model.hfield_size[hfid][:2])
        span = 2.0 * max(hx, hy)
    return cx, cy, cz, span


def render_wave_preview(dist, out_path, *, mode="CDW", seed=0, theta_deg=0.0,
                        beta_deg=0.0, duration=10.0, fps=30, width=720, height=480,
                        label="", model_name=None):
    """Render a `duration`-second MP4 of the pool scene under `dist` at the (theta_deg
    current, beta_deg wave) heading, disturbance `mode` (default CDW). The ROV is
    released from rest with NO controller and stepped so the real current + wave forces
    carry it (it may drift out of the fixed frame — expected). The water surface is
    physically faithful by default (WATER_WAVE_LAMBDA/AMP exaggerate it). Returns the
    number of frames written (0 if skipped/failed — the batch must never crash because
    of the preview).

    `dist` is a DistConfig (e.g. cfg.wave_configs[i]); its theta_c/beta_bar are
    overridden by theta_deg/beta_deg here."""
    model_name = model_name or RM.MODEL
    pool_xml = RM.pool_xml_path(model_name)
    if not os.path.isfile(pool_xml):
        print(f"[wave_preview] SKIP: pool scene {os.path.basename(pool_xml)} not found "
              f"(regenerate with tools/gen_pool_apriltags.py). No preview written.",
              flush=True)
        return 0

    try:
        import cv2
    except Exception as e:                                    # noqa: BLE001
        print(f"[wave_preview] SKIP: cv2 unavailable ({type(e).__name__}: {e}). "
              f"No preview written.", flush=True)
        return 0

    writer = renderer = tmp_path = None
    hydro_on = False
    try:
        model = mujoco.MjModel.from_xml_path(pool_xml)
        data = mujoco.MjData(model)

        wf = _water_frame(model)
        # faithful surface by default (WATER_WAVE_LAMBDA/AMP exaggerate); same as viewer
        surf = WV.make_surface_from_env(model) if wf is not None else None
        if surf is None:
            print(f"[wave_preview] SKIP: scene has no animated water surface hfield "
                  f"(built with --no-water-anim?). No preview written.", flush=True)
            return 0

        dt = float(model.opt.timestep)
        env = DisturbanceEnv(
            dataclasses.replace(dist, theta_c=float(np.radians(theta_deg)),
                                beta_bar=float(np.radians(beta_deg))),
            mode=mode, seed=int(seed), dt=dt, T_sim=float(duration) + 1.0)
        env.enabled = True
        # step the REAL physics so the current + wave forces carry the ROV (no
        # controller -> ctrl stays 0). Only base_link sees hydro; the pool floor /
        # water are non-colliding visual geoms, so the ROV drifts freely.
        H.Hydrodynamics.uninstall()                          # drop any stale passive cb first
        H.Hydrodynamics(model, disturbance=env).install()
        hydro_on = True
        mujoco.mj_forward(model, data)                       # ROV at its rest pose, level

        renderer = mujoco.Renderer(model, height, width)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cx, cy, cz, span = wf
        cam.lookat[:] = [cx, cy, cz - 0.8]                    # between surface, ROV, floor
        cam.distance = 1.35 * span
        cam.azimuth = 60.0
        cam.elevation = -16.0

        # Encode into a .part.mp4 and atomically promote on success -> out_path only
        # ever exists as a COMPLETE clip (a mid-render crash/SIGKILL leaves the .part,
        # never a truncated final; a teardown error can't delete a promoted good file).
        tmp_path = out_path + ".part.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, float(fps), (int(width), int(height)))
        if not writer.isOpened():
            raise RuntimeError("cv2.VideoWriter failed to open")

        n_frames = int(round(float(duration) * float(fps)))
        for k in range(n_frames):
            # render the current state, then advance the physics to the next frame time
            renderer.update_scene(data, cam)
            surf.update(env, data.time, enabled=True, renderer=renderer)
            frame = renderer.render()                        # (H,W,3) uint8 RGB
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            target = (k + 1) / float(fps)
            while data.time < target:
                mujoco.mj_step(model, data)                  # ctrl==0: free drift under waves
        writer.release(); writer = None
        os.replace(tmp_path, out_path); tmp_path = None      # atomic promote
        try:
            renderer.close()                                 # a close() failure now is harmless
        except Exception:                                    # noqa: BLE001
            pass
        renderer = None
        tag = f" [{label}]" if label else ""
        print(f"[wave_preview] wrote {os.path.basename(out_path)}{tag}: {n_frames} frames "
              f"({duration:g}s @ {fps:g}fps, mode {mode}, current {theta_deg:g} deg, "
              f"wave {beta_deg:g} deg)", flush=True)
        return n_frames
    except Exception as e:                                    # noqa: BLE001
        print(f"[wave_preview] DISABLED ({type(e).__name__}: {e}). No preview written; "
              f"for a display-less machine try MUJOCO_GL=egl.", flush=True)
        try:
            if writer is not None:
                writer.release()
        except Exception:                                    # noqa: BLE001
            pass
        try:
            if renderer is not None:
                renderer.close()
        except Exception:                                    # noqa: BLE001
            pass
        # remove only the un-promoted partial temp; never a good, already-promoted out_path
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return 0
    finally:
        if hydro_on:
            H.Hydrodynamics.uninstall()                      # drop this preview's passive cb


def main():
    ap = argparse.ArgumentParser(
        description="Offscreen MP4 preview of the CDW pool wave environment.")
    ap.add_argument("--config", default=os.path.join(HERE, "config", "base.yaml"))
    ap.add_argument("--wave", type=int, default=None,
                    help="index into a multi-wave `waves:` list (default: all of them).")
    ap.add_argument("--mode", default="CDW", choices=("C", "CD", "CW", "CDW"),
                    help="disturbance mode shown (default CDW = current+drift+waves).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--theta-deg", type=float, default=0.0, help="current heading [deg].")
    ap.add_argument("--beta-deg", type=float, default=None,
                    help="wave heading [deg] (default: the wave spec's beta_bar_deg).")
    ap.add_argument("--duration", type=float, default=10.0, help="clip length [s].")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--video-size", default="720x480", help="WIDTHxHEIGHT.")
    ap.add_argument("--out", default=None, help="output dir (default recordings/<date>/wave_preview).")
    args = ap.parse_args()

    import time
    cfg = load_config(args.config)
    width, height = (int(x) for x in args.video_size.lower().split("x"))
    out_dir = args.out or os.path.join(HERE, "recordings", time.strftime("%Y%m%d"),
                                       "wave_preview")
    os.makedirs(out_dir, exist_ok=True)

    idxs = [args.wave] if args.wave is not None else list(range(len(cfg.wave_configs)))
    total = 0
    for i in idxs:
        dist = cfg.wave_configs[i]
        label = cfg.wave_labels[i]
        beta = float(args.beta_deg) if args.beta_deg is not None \
            else float(np.degrees(dist.beta_bar))
        stem = f"wave_preview_{i:02d}_{label}" if len(cfg.wave_configs) > 1 else "wave_preview"
        n = render_wave_preview(
            dist, os.path.join(out_dir, stem + ".mp4"), mode=args.mode, seed=args.seed,
            theta_deg=args.theta_deg, beta_deg=beta, duration=args.duration,
            fps=args.fps, width=width, height=height, label=label)
        total += n
    print(f"[wave_preview] {len([i for i in idxs])} preview(s) attempted -> {out_dir}",
          flush=True)


if __name__ == "__main__":
    main()
