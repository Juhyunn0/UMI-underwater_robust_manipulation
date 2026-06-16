#!/usr/bin/env python3
"""acados (SQP-RTI) vs pre-acados (IPOPT) on the same 1m square mission.

Pre-acados runs (recorded before the 16:57 acados build) vs acados runs
(recorded ~19:30, params.SOLVER='acados'). PID never used the NMPC solver, so
its before/after is a noise-level sanity check; MPC/DOB-MPC swapped IPOPT->acados.

The CSV carries no solver flag or solve-time, so this confirms CLOSED-LOOP
TRACKING IS PRESERVED across the solver swap; the speed/determinism win
(103x, n_fail 7->0) is the runtime result in verify_acados.py.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "recordings", "20260615")
# controller -> (pre-acados file, acados file)
PAIRS = {
    "PID":     ("20260615_164104_square_pid.csv",    "20260615_194936_square_pid.csv"),
    "MPC":     ("20260615_160633_square_mpc.csv",    "20260615_194349_square_mpc.csv"),
    "DOB-MPC": ("20260615_152706_square_dobmpc.csv", "20260615_193023_square_dobmpc.csv"),
}
S, SPEED, PER, Z = 1.0, 0.15, 4.0, 0.0
SEGS = [((0, 0), (S, 0)), ((S, 0), (S, S)), ((S, S), (0, S)), ((0, S), (0, 0))]
COL = {n: i for i, n in enumerate(
    "t,px,py,pz,qw,qx,qy,qz,roll,pitch,yaw,vx,vy,vz,wx,wy,wz,"
    "cur_x,cur_y,cur_z,wav_x,wav_y,wav_z,kick_x,kick_y,kick_z,"
    "u0,u1,u2,u3,u4,u5,dist_on".split(","))}


def crosstrack(px, py):
    d = np.full(px.shape, np.inf)
    for (ax, ay), (bx, by) in SEGS:
        abx, aby = bx - ax, by - ay
        L2 = abx * abx + aby * aby
        tt = np.clip(((px - ax) * abx + (py - ay) * aby) / L2, 0.0, 1.0)
        d = np.minimum(d, np.hypot(px - (ax + tt * abx), py - (ay + tt * aby)))
    return d


def analyze(path):
    D = np.loadtxt(path, delimiter=",", skiprows=1)
    c = COL
    px, py, pz = D[:, c["px"]], D[:, c["py"]], D[:, c["pz"]]
    off = crosstrack(px, py)
    roll = np.abs(np.degrees(D[:, c["roll"]]))
    pitch = np.abs(np.degrees(D[:, c["pitch"]]))
    yaw = np.abs(np.degrees(D[:, c["yaw"]]))
    cx, cy = px.mean(), py.mean()                 # centroid (DC current bias)
    cur = np.hypot(D[:, c["cur_x"]], D[:, c["cur_y"]]).mean()
    wav = np.linalg.norm(D[:, [c["wav_x"], c["wav_y"], c["wav_z"]]], axis=1).mean()
    ueff = np.sqrt((D[:, [c["u0"], c["u1"], c["u2"], c["u3"], c["u4"], c["u5"]]] ** 2)
                   .sum(1)).mean()
    rms = lambda a: float(np.sqrt((a ** 2).mean()))
    return dict(
        n=len(px), T=D[-1, c["t"]] - D[0, c["t"]], dist=int(D[:, c["dist_on"]].mean() > .5),
        off_rms=rms(off) * 100, off_max=off.max() * 100,
        pitch_rms=rms(pitch), pitch_max=pitch.max(),
        roll_rms=rms(roll), yaw_rms=rms(yaw), yaw_max=yaw.max(),
        depth_rms=rms(np.abs(pz - Z)) * 100, dc_x=(cx - 0.5) * 100, dc_y=(cy - 0.5) * 100,
        cur=cur, wav=wav, ueff=ueff, px=px, py=py)


def main():
    res = {k: (analyze(os.path.join(DIR, b)), analyze(os.path.join(DIR, a)))
           for k, (b, a) in PAIRS.items()}

    print("\n================  acados (SQP-RTI)  vs  pre-acados (IPOPT)  — 1m square  ================")
    print("(PID has no NMPC solver -> its before/after is a noise-floor sanity check)\n")
    rows = [
        ("off-path rms [cm]",  "off_rms",  "{:6.2f}"),
        ("off-path max [cm]",  "off_max",  "{:6.2f}"),
        ("DC bias x [cm]",     "dc_x",     "{:+6.2f}"),
        ("pitch rms [deg]",    "pitch_rms","{:6.2f}"),
        ("pitch max [deg]",    "pitch_max","{:6.2f}"),
        ("yaw rms [deg]",      "yaw_rms",  "{:6.2f}"),
        ("yaw max [deg]",      "yaw_max",  "{:6.2f}"),
        ("roll rms [deg]",     "roll_rms", "{:6.2f}"),
        ("depth rms [cm]",     "depth_rms","{:6.2f}"),
        ("ctrl effort |u|",    "ueff",     "{:6.2f}"),
    ]
    for ctrl, (b, a) in res.items():
        print(f"--- {ctrl} ---   rows {b['n']}/{a['n']}  dur {b['T']:.0f}/{a['T']:.0f}s  "
              f"dist {b['dist']}/{a['dist']}  |  cur {b['cur']:.2f}/{a['cur']:.2f}  "
              f"wav {b['wav']:.3f}/{a['wav']:.3f}")
        print(f"    {'metric':<18}{'pre-acados':>12}{'acados':>12}{'Δ':>10}")
        for label, key, fmt in rows:
            pv, av = b[key], a[key]
            print(f"    {label:<18}{fmt.format(pv):>12}{fmt.format(av):>12}{av-pv:>+10.2f}")
        print()

    # plot: xy paths overlaid (pre vs acados) per controller
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    plan = np.array([[0, 0], [S, 0], [S, S], [0, S], [0, 0]], float)
    for ax, (ctrl, (b, a)) in zip(axes, res.items()):
        ax.plot(plan[:, 0], plan[:, 1], "--", color="0.6", lw=1.4, label="plan")
        ax.plot(b["px"], b["py"], lw=0.6, color="tab:orange", alpha=0.7,
                label=f"pre-acados (off {b['off_rms']:.1f}cm)")
        ax.plot(a["px"], a["py"], lw=0.6, color="tab:blue", alpha=0.7,
                label=f"acados (off {a['off_rms']:.1f}cm)")
        ax.set_title(f"{ctrl}  — square xy")
        ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(DIR, "acados_vs_before.png")
    fig.savefig(out, dpi=110)
    print(f"[plot] {out}")


if __name__ == "__main__":
    main()
