"""Live 3-D viewer of the *closed-loop* BlueROV2 underwater.

Unlike view_model.py (bare geometry, free-falls under gravity), this script
runs the real control stack - BlueROV2MujocoEnv (buoyancy + damping + added
mass + thrusters injected every substep) driven by PID / MPC / DOBMPC, with an
external disturbance - and renders it live in the MuJoCo passive viewer.  You
watch the vehicle actually hold station / track a trajectory while the current
pushes it around.

  python scripts/view_sim.py                      # DOBMPC, station-keeping, waves
  python scripts/view_sim.py --controller pid     # watch the baseline drift
  python scripts/view_sim.py --traj circle --dist mixed
  python scripts/view_sim.py --controller mpc --dist constant

Scene cues (decoration only - no effect on physics):
  * green translucent sphere = commanded reference (where the ROV should be)
  * blue dotted trail        = where the ROV has actually been
  * sandy slab / blue slab   = sea bed / water surface
The world is NED (+z points DOWN), so on first open the camera looks "up" at
the vehicle; drag with the left mouse to orbit, scroll to zoom.  Press SPACE
to pause/resume, BACKSPACE to restart the run.

Note: DOBMPC/MPC call Ipopt every step (~70-100 ms), so they play back a bit
slower than real time; PID runs at full real-time speed.
"""
import argparse
import os
import sys
import time
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujoco
import mujoco.viewer
import numpy as np

from bluerov2mj import allocation, disturbances, experiment
from bluerov2mj.controllers.mpc import NMPC
from bluerov2mj.controllers.pid import PID
from bluerov2mj.eaob import EAOB
from bluerov2mj.mujoco_env import BlueROV2MujocoEnv

NO_NOISE = dict(pos=0, ang=0, lin_vel=0, ang_vel=0, lin_acc=0, ang_acc=0)


def make_ref(traj):
    if traj == "station":
        return experiment.dp_reference()
    if traj == "circle":
        return experiment.circle_reference()
    if traj == "lemniscate":
        return experiment.lemniscate_reference()
    raise ValueError(traj)


def underwater_look(model):
    """Tint the scene like murky water (lighting only; physics untouched)."""
    model.vis.headlight.ambient[:] = (0.22, 0.34, 0.44)
    model.vis.headlight.diffuse[:] = (0.34, 0.52, 0.62)
    model.vis.headlight.specular[:] = (0.12, 0.12, 0.14)


def _add(scn, gtype, size, pos, rgba, mat=None):
    """Append one decorative geom to the viewer's user scene (guarded)."""
    if scn.ngeom >= scn.maxgeom:
        return
    mat = np.eye(3).flatten() if mat is None else np.asarray(mat, float).flatten()
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], int(gtype),
                        np.asarray(size, float), np.asarray(pos, float),
                        mat, np.asarray(rgba, np.float32))
    scn.ngeom += 1


def draw_scene(viewer, eta_ref, z0, trail, show_scene):
    scn = viewer.user_scn
    scn.ngeom = 0
    if show_scene:
        # sea bed 4 m below the nominal depth, water surface 5 m above
        # (NED: +z is down, so "below" is the larger z)
        _add(scn, mujoco.mjtGeom.mjGEOM_BOX, (25, 25, 0.05),
             (0, 0, z0 + 4.0), (0.55, 0.49, 0.38, 1.0))
        _add(scn, mujoco.mjtGeom.mjGEOM_BOX, (25, 25, 0.02),
             (0, 0, z0 - 5.0), (0.10, 0.34, 0.52, 0.16))
    # reference marker (where the controller is told to be)
    _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, (0.13, 0, 0),
         eta_ref[:3], (0.15, 0.95, 0.25, 0.55))
    # breadcrumb trail of the true path
    for p in trail:
        _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, (0.04, 0, 0), p,
             (0.30, 0.70, 1.0, 0.7))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", default="dobmpc",
                    choices=["pid", "mpc", "dobmpc"])
    ap.add_argument("--traj", default="station",
                    choices=["station", "circle", "lemniscate"])
    ap.add_argument("--dist", default="periodic",
                    choices=["none", "periodic", "constant", "mixed"])
    ap.add_argument("--T", type=float, default=30.0,
                    help="seconds per run before it loops")
    ap.add_argument("--N", type=int, default=40, help="MPC horizon")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--noise", action="store_true",
                    help="enable measurement noise (default: clean)")
    ap.add_argument("--no-scene", action="store_true",
                    help="hide the sea-bed / water-surface decoration")
    args = ap.parse_args()

    env = BlueROV2MujocoEnv(seed=args.seed,
                            meas_noise=None if args.noise else NO_NOISE)
    underwater_look(env.model)
    ref_fn = make_ref(args.traj)
    dist_fn = disturbances.make(args.dist, seed=args.seed)
    dt = env.dt_ctrl
    z0 = ref_fn(0.0)[0][2]

    mpc = (NMPC(N=args.N) if args.controller in ("mpc", "dobmpc") else None)
    pid = PID() if args.controller == "pid" else None

    # ----- shared run state (reset() rebuilds it; key_callback toggles pause)
    state = {"paused": False, "restart": False}

    def key_callback(keycode):
        if keycode == 32:                       # SPACE
            state["paused"] = not state["paused"]
        elif keycode in (259, 8):               # BACKSPACE
            state["restart"] = True

    def fresh_run():
        meas = env.reset(eta0=ref_fn(0.0)[0])
        obs = (EAOB(eta0=meas["eta"], nu0=meas["nu"])
               if args.controller == "dobmpc" else None)
        if mpc is not None:
            mpc.reset()
        if pid is not None:
            pid.reset()
        return meas, obs, np.zeros(4), deque(maxlen=200)

    print(f"[view_sim] {args.controller.upper()} | traj={args.traj} | "
          f"dist={args.dist} | noise={'on' if args.noise else 'off'}")
    print("  SPACE pause/resume   BACKSPACE restart   (close window to quit)")

    meas, obs, u, trail = fresh_run()
    with mujoco.viewer.launch_passive(env.model, env.data,
                                      key_callback=key_callback) as viewer:
        viewer.cam.type = int(mujoco.mjtCamera.mjCAMERA_TRACKING)
        viewer.cam.trackbodyid = env.bid
        viewer.cam.distance = 6.0
        viewer.cam.elevation = -18.0
        viewer.cam.azimuth = 130.0

        wall0 = time.time()
        last_print = 0.0
        while viewer.is_running():
            if state["restart"]:
                meas, obs, u, trail = fresh_run()
                state["restart"] = False
                wall0 = time.time()
            if state["paused"]:
                viewer.sync()
                time.sleep(0.02)
                continue

            t = env.t
            eta_r, nu_r = ref_fn(t)

            if obs is not None:
                eta_h, nu_h, w_h = obs.update(meas, allocation.wrench_from_u(u))
                x_ctrl = np.concatenate([eta_h, nu_h])
                w_mpc = w_h
            else:
                x_ctrl = np.concatenate([meas["eta"], meas["nu"]])
                w_mpc = np.zeros(6)

            if pid is not None:
                u = pid.solve(x_ctrl, np.concatenate([eta_r, nu_r]))
            else:
                refs = np.empty((12, mpc.N + 1))
                for j in range(mpc.N + 1):
                    er, nr = ref_fn(t + j * dt)
                    refs[:, j] = np.concatenate([er, nr])
                u = mpc.solve(x_ctrl, w_mpc, refs)

            meas, x_true = env.step(u, dist_fn(t))
            trail.append(x_true[:3].copy())

            draw_scene(viewer, ref_fn(env.t)[0], z0, trail, not args.no_scene)
            viewer.sync()

            # pace to real time (controllers slower than real time just lag)
            ahead = env.t - (time.time() - wall0)
            if ahead > 0:
                time.sleep(ahead)

            if env.t - last_print >= 0.5:
                last_print = env.t
                e = x_true[:6] - ref_fn(env.t)[0]
                e[5] = (e[5] + np.pi) % (2 * np.pi) - np.pi
                wn = np.linalg.norm(dist_fn(env.t)[:3])
                msg = (f"\r  t={env.t:5.1f}s  pos_err={np.linalg.norm(e[:3]):.3f} m"
                       f"  yaw_err={abs(e[5]):.3f} rad  |F_dist|={wn:5.1f} N")
                if obs is not None:
                    msg += f"  |w_est|={np.linalg.norm(obs.w_world()[:3]):5.1f} N"
                sys.stdout.write(msg + "   ")
                sys.stdout.flush()

            if env.t >= args.T:
                meas, obs, u, trail = fresh_run()
                wall0 = time.time()
    print("\n[view_sim] window closed.")


if __name__ == "__main__":
    main()
