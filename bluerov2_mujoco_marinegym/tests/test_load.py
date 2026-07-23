#!/usr/bin/env python3
"""
Phase-1 verification for the MarineGym BlueROV2 MJCF import.

Loads bluerov.xml with *base* MuJoCo (no MJX / JAX / CUDA needed), prints model
stats, lists the thruster mount sites, and runs a zero-control stability check
(a few seconds of free fall; with no buoyancy yet the vehicle simply sinks --
that is expected this phase). Optionally renders one headless frame to a PNG.

Usage
-----
    python test_load.py                 # load + stats + stability check
    python test_load.py --seconds 3     # longer stability check
    python test_load.py --render preview.png
    python test_load.py --viewer        # interactive viewer (needs a display)

Only `mujoco` and `numpy` are required. The PNG writer is pure stdlib, so
--render adds no dependency (it still needs a working offscreen GL context).
"""
import argparse
import os
import struct
import sys
import zlib

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "bluerov.xml")


def write_png(path, rgb):
    """Minimal RGB PNG writer (stdlib only). rgb: (H, W, 3) uint8."""
    h, w, _ = rgb.shape
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(typ, data):
        body = typ + data
        return struct.pack(">I", len(data)) + body + struct.pack(
            ">I", zlib.crc32(body) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 9)))
        f.write(chunk(b"IEND", b""))


def print_stats(model, data):
    mujoco.mj_forward(model, data)
    print("=" * 64)
    print(f"Loaded: {XML}")
    print("=" * 64)

    total_mass = float(model.body_mass.sum())
    print(f"  bodies (incl. world) : {model.nbody}")
    print(f"  geoms                : {model.ngeom}")
    print(f"  meshes               : {model.nmesh}")
    print(f"  sites                : {model.nsite}")
    print(f"  free joints / DoF    : {model.njnt}  /  nv={model.nv}, nq={model.nq}")
    print(f"  TOTAL MASS           : {total_mass:.4f} kg   (BlueROV2 ~ 10-11 kg)")

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    print(f"  base_link mass       : {model.body_mass[bid]:.4f} kg")
    print(f"  base_link inertia    : {np.array(model.body_inertia[bid]).round(5).tolist()}  (Ixx,Iyy,Izz)")
    print(f"  base_link com (local): {np.array(model.body_ipos[bid]).round(5).tolist()}")

    # geom breakdown by group
    vis = int(np.sum((model.geom_contype == 0) & (model.geom_conaffinity == 0)))
    col = model.ngeom - vis
    print(f"  visual / collision   : {vis} visual, {col} collision geom(s)")

    print("\n  Thruster mount sites (local pos in base_link frame; +X = thrust dir):")
    nthr = 0
    for sid in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid)
        if name and name.startswith("thruster_"):
            nthr += 1
            pos = np.array(model.site_pos[sid]).round(5)
            # local +X axis of the site, expressed in body frame -> thrust dir
            xmat = np.array(data.site_xmat[sid]).reshape(3, 3)
            xaxis = xmat[:, 0].round(3)
            print(f"    {name:<11} pos={pos.tolist()}   thrust_axis(world)={xaxis.tolist()}")
    print(f"  -> {nthr} thruster sites (expected 6)")
    print("=" * 64)
    return total_mass, nthr


def stability_check(model, data, seconds):
    steps = int(round(seconds / model.opt.timestep))
    z0 = float(data.qpos[2])
    print(f"\nZero-control stability check: {steps} steps "
          f"({seconds:.1f}s @ dt={model.opt.timestep*1000:.1f} ms), ctrl=0")
    data.ctrl[:] = 0  # (no actuators this phase; defensive)
    blew_up = False
    for i in range(steps):
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            print(f"  !! non-finite state at step {i}")
            blew_up = True
            break
    z1 = float(data.qpos[2])
    vz = float(data.qvel[2])
    finite = np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
    print(f"  start z = {z0:+.3f} m  ->  end z = {z1:+.3f} m   (Δz = {z1 - z0:+.3f} m)")
    print(f"  end vertical velocity vz = {vz:+.3f} m/s   (sinking under gravity = expected)")
    print(f"  all states finite: {finite}   |   no blow-up: {not blew_up}")
    # free fall from rest for `seconds` should give z ~ -0.5 g t^2
    expected = -0.5 * abs(model.opt.gravity[2]) * seconds ** 2
    print(f"  (ideal free-fall Δz ≈ {expected:+.2f} m; close confirms gravity acts, no NaN)")
    return finite and not blew_up


def render_frame(model, data, out_path):
    try:
        renderer = mujoco.Renderer(model, height=960, width=1280)
    except Exception as e:  # noqa: BLE001
        print(f"\n[render] offscreen GL unavailable ({e!r}); skipping PNG. "
              f"Use --viewer for interactive view instead.")
        return False
    mujoco.mj_forward(model, data)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[2] = 1          # body + thruster visuals live in group 2
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, -0.03]
    cam.distance = 1.1
    cam.azimuth = 130
    cam.elevation = -20
    renderer.update_scene(data, camera=cam, scene_option=opt)
    write_png(out_path, renderer.render())
    print(f"\n[render] wrote {out_path}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--render", metavar="PNG", default=None)
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)

    total_mass, nthr = print_stats(model, data)
    ok = stability_check(model, data, args.seconds)

    # sanity assertions for the phase-1 deliverable
    assert 9.0 <= total_mass <= 12.0, f"total mass {total_mass} out of expected range"
    assert nthr == 6, f"expected 6 thruster sites, found {nthr}"
    assert ok, "stability check failed (NaN / blow-up)"
    print("\nPHASE-1 CHECKS PASSED ✔  (loads, ~11 kg, 6 thrusters, stable under gravity)")

    if args.render:
        mujoco.mj_resetData(model, data)
        render_frame(model, data, args.render)

    if args.viewer:
        from mujoco import viewer as mj_viewer
        mujoco.mj_resetData(model, data)
        print("\nLaunching interactive viewer (close window to exit)...")
        mj_viewer.launch(model, data)


if __name__ == "__main__":
    sys.exit(0 if main() is None else 0)
