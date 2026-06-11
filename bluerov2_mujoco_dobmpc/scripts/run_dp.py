"""Dynamic positioning experiment (paper Sec. 5.1).

Hold eta_d = [0, 0, -20, 0, 0, 0] while external disturbances act on the
vehicle.  Compares PID / MPC / DOBMPC and reproduces the paper's error,
control and disturbance-estimation figures plus the RMSE table (Table 6).

  python scripts/run_dp.py --controller all --dist constant --T 25
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
    ap.add_argument("--controller", default="all",
                    choices=["pid", "mpc", "dobmpc", "all"])
    ap.add_argument("--dist", default="constant",
                    choices=["constant", "periodic", "mixed", "none"])
    ap.add_argument("--T", type=float, default=25.0)
    ap.add_argument("--plant", default="mujoco",
                    choices=["mujoco", "analytic"])
    ap.add_argument("--N", type=int, default=None, help="MPC horizon")
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

    names = (["pid", "mpc", "dobmpc"] if args.controller == "all"
             else [args.controller])
    mpc_kwargs = dict(solver=args.solver)
    if args.N:
        mpc_kwargs["N"] = args.N
    mpc = (NMPC(**mpc_kwargs)
           if not args.combine and any(n != "pid" for n in names) else None)

    ref = experiment.dp_reference()
    tag = f"dp_{args.dist}_{args.plant}"
    logs = {}
    if args.combine:
        for name in ["pid", "mpc", "dobmpc"]:
            fp = os.path.join(args.out, f"{tag}_{name}.npz")
            if os.path.exists(fp):
                logs[name] = dict(np.load(fp))
        if not logs:
            raise SystemExit("no saved logs found to combine")
    else:
        print(f"DP | disturbance={args.dist} | plant={args.plant} | T={args.T}s")
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
                      title=f"Dynamic positioning - {args.dist} disturbance")
    plots.plot_controls(logs, os.path.join(args.out, f"{tag}_controls.png"))
    if "dobmpc" in logs:
        plots.plot_disturbance(logs["dobmpc"],
                               os.path.join(args.out, f"{tag}_eaob.png"))
    print(f"figures -> {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
