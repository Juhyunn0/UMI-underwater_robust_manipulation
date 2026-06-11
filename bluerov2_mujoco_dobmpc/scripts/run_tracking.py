"""Trajectory tracking experiment (paper Sec. 5.2).

Circle (r = 2 m, one lap / 12.5 s, tangential yaw) or Gerono lemniscate
(2 m, 25 s period, yaw = 0) under external disturbances.

  python scripts/run_tracking.py --traj circle --dist mixed --T 25
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from bluerov2mj import disturbances, experiment, plots
from bluerov2mj.controllers.mpc import NMPC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="circle",
                    choices=["circle", "lemniscate"])
    ap.add_argument("--controller", default="all",
                    choices=["pid", "mpc", "dobmpc", "all"])
    ap.add_argument("--dist", default="mixed",
                    choices=["constant", "periodic", "mixed", "none"])
    ap.add_argument("--T", type=float, default=25.0)
    ap.add_argument("--plant", default="mujoco",
                    choices=["mujoco", "analytic"])
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--solver", default="ipopt",
                    choices=["sqpmethod", "ipopt"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--combine", action="store_true",
                    help="skip simulation; merge saved per-controller logs "
                         "from --out and produce figures/RMSE")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "..", "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ref = (experiment.circle_reference() if args.traj == "circle"
           else experiment.lemniscate_reference())
    names = (["pid", "mpc", "dobmpc"] if args.controller == "all"
             else [args.controller])
    mpc_kwargs = dict(solver=args.solver)
    if args.N:
        mpc_kwargs["N"] = args.N
    mpc = (NMPC(**mpc_kwargs)
           if not args.combine and any(n != "pid" for n in names) else None)

    tag = f"track_{args.traj}_{args.dist}_{args.plant}"
    logs = {}
    if args.combine:
        for name in ["pid", "mpc", "dobmpc"]:
            fp = os.path.join(args.out, f"{tag}_{name}.npz")
            if os.path.exists(fp):
                logs[name] = dict(np.load(fp))
        if not logs:
            raise SystemExit("no saved logs found to combine")
    else:
        print(f"tracking={args.traj} | disturbance={args.dist} | "
              f"plant={args.plant} | T={args.T}s")
        for name in names:
            env = experiment.make_env(args.plant, seed=args.seed)
            dist = disturbances.make(args.dist, seed=args.seed)
            logs[name] = experiment.run_closed_loop(name, env, ref, dist,
                                                    args.T, mpc_obj=mpc)
            np.savez(os.path.join(args.out, f"{tag}_{name}.npz"),
                     **logs[name])
    rmse = experiment.rmse_table(logs)
    txt = experiment.print_rmse(rmse)
    with open(os.path.join(args.out, f"{tag}_rmse.txt"), "w") as f:
        f.write(txt + "\n")
    plots.plot_errors(logs, os.path.join(args.out, f"{tag}_errors.png"),
                      title=f"{args.traj} tracking - {args.dist} disturbance")
    plots.plot_xy(logs, os.path.join(args.out, f"{tag}_xy.png"),
                  title=f"{args.traj} tracking (top view)")
    plots.plot_controls(logs, os.path.join(args.out, f"{tag}_controls.png"))
    if "dobmpc" in logs:
        plots.plot_disturbance(logs["dobmpc"],
                               os.path.join(args.out, f"{tag}_eaob.png"))
    print(f"figures -> {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
