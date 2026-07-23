#!/usr/bin/env python3
"""Compare PID / MPC / DOB-MPC on the 1m square (CCW, 0.15 m/s, z=0).

Off-path = perpendicular distance to the square polyline (timing-independent).
Time-ref error = distance to the moving setpoint s=speed*(t-t0) (mod P).
Orientation in deg. READ-ONLY analysis; writes a PNG + prints a table.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(HERE, "recordings", "20260615")
RUNS = [
    ("PID",     "20260615_164104_square_pid.csv"),
    ("MPC",     "20260615_160633_square_mpc.csv"),
    ("DOB-MPC", "20260615_152706_square_dobmpc.csv"),
]
S, SPEED, PER, Z = 1.0, 0.15, 4.0, 0.0
SEGS = [((0, 0), (S, 0)), ((S, 0), (S, S)), ((S, S), (0, S)), ((0, S), (0, 0))]
COL = {n: i for i, n in enumerate(
    "t,px,py,pz,qw,qx,qy,qz,roll,pitch,yaw,vx,vy,vz,wx,wy,wz,"
    "cur_x,cur_y,cur_z,wav_x,wav_y,wav_z,kick_x,kick_y,kick_z,"
    "u0,u1,u2,u3,u4,u5,dist_on".split(","))}


def crosstrack(px, py):
    """Min perpendicular distance from each (px,py) to the square polyline."""
    d = np.full(px.shape, np.inf)
    for (ax, ay), (bx, by) in SEGS:
        abx, aby = bx - ax, by - ay
        L2 = abx * abx + aby * aby
        tt = np.clip(((px - ax) * abx + (py - ay) * aby) / L2, 0.0, 1.0)
        cx, cy = ax + tt * abx, ay + tt * aby
        d = np.minimum(d, np.hypot(px - cx, py - cy))
    return d


def ref_point(s):
    s = s % PER
    if s < S:     return s, 0.0
    if s < 2 * S: return S, s - S
    if s < 3 * S: return 3 * S - s, S
    return 0.0, 4 * S - s


def stats(a):
    a = np.asarray(a, float)
    return a.mean(), np.sqrt((a ** 2).mean()), a.max()


def load(path):
    return np.loadtxt(path, delimiter=",", skiprows=1)


def main():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    rows = []
    plan = np.array([[0, 0], [S, 0], [S, S], [0, S], [0, 0]], float)
    for ci, (name, fn) in enumerate(RUNS):
        D = load(os.path.join(DIR, fn))
        t = D[:, COL["t"]]
        px, py, pz = D[:, COL["px"]], D[:, COL["py"]], D[:, COL["pz"]]
        roll = np.degrees(D[:, COL["roll"]])
        pitch = np.degrees(D[:, COL["pitch"]])
        yaw = np.degrees(D[:, COL["yaw"]])
        # off-path (cross-track) to the square outline
        off = crosstrack(px, py)
        # time-referenced error
        t0 = t[0]
        sref = SPEED * (t - t0)
        ref = np.array([ref_point(s) for s in sref])
        terr = np.hypot(px - ref[:, 0], py - ref[:, 1])
        depth = np.abs(pz - Z)
        # disturbance magnitudes (confirm same conditions)
        cur = np.hypot(D[:, COL["cur_x"]], D[:, COL["cur_y"]])
        wav = np.linalg.norm(D[:, [COL["wav_x"], COL["wav_y"], COL["wav_z"]]], axis=1)
        spd = np.hypot(D[:, COL["vx"]], D[:, COL["vy"]])
        # control effort (thrust-channel RMS over the 6 generalized actuators)
        ueff = np.sqrt((D[:, [COL["u0"], COL["u1"], COL["u2"],
                              COL["u3"], COL["u4"], COL["u5"]]] ** 2).sum(1))

        rows.append(dict(
            name=name, n=len(t), T=t[-1] - t[0], dist=int(D[:, COL["dist_on"]].mean() > 0.5),
            off=stats(off), terr=stats(terr), depth=stats(depth),
            roll=stats(np.abs(roll)), pitch=stats(np.abs(pitch)), yaw=stats(np.abs(yaw)),
            cur=cur.mean(), wav=wav.mean(), spd=spd.mean(), ueff=ueff.mean()))

        # --- top row: xy path
        ax = axes[0, ci]
        ax.plot(plan[:, 0], plan[:, 1], "--", color="0.6", lw=1.5, label="plan")
        sc = ax.scatter(px, py, c=t, s=2, cmap="viridis")
        ax.set_title(f"{name}  xy path  (off-path rms {stats(off)[1]*100:.1f} cm)")
        ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.grid(alpha=0.3)
        # --- bottom row: pitch over time (the orientation focus)
        ax2 = axes[1, ci]
        ax2.plot(t, pitch, lw=0.7, color="tab:red", label="pitch")
        ax2.plot(t, roll, lw=0.6, color="tab:blue", alpha=0.7, label="roll")
        ax2.plot(t, yaw, lw=0.6, color="tab:green", alpha=0.7, label="yaw")
        ax2.set_title(f"{name}  orientation [deg]  (pitch rms {stats(np.abs(pitch))[1]:.1f}, "
                      f"max {stats(np.abs(pitch))[2]:.1f})")
        ax2.set_xlabel("t [s]"); ax2.set_ylabel("deg"); ax2.grid(alpha=0.3)
        ax2.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    out = os.path.join(DIR, "square3_compare.png")
    fig.savefig(out, dpi=110)

    # ---- table
    def f3(s): return f"{s[0]*100:6.2f} {s[1]*100:6.2f} {s[2]*100:6.2f}"  # cm
    def fd(s): return f"{s[0]:6.2f} {s[1]:6.2f} {s[2]:6.2f}"              # deg
    print("\n=== 1m SQUARE @ 0.15 m/s — PID vs MPC vs DOB-MPC ===")
    print(f"{'metric':<22}" + "".join(f"{r['name']:>22}" for r in rows))
    print(f"{'rows / dur[s]':<22}" + "".join(
        f"{r['n']:>8} /{r['T']:7.1f}" for r in rows))
    print(f"{'disturb on':<22}" + "".join(f"{r['dist']:>22}" for r in rows))
    print("-" * 90)
    print("POSITION  (mean rms max)")
    print(f"  {'off-path [cm]':<20}" + "".join(f"{f3(r['off']):>22}" for r in rows))
    print(f"  {'time-ref err [cm]':<20}" + "".join(f"{f3(r['terr']):>22}" for r in rows))
    print(f"  {'depth |z| [cm]':<20}" + "".join(f"{f3(r['depth']):>22}" for r in rows))
    print("ORIENTATION |deg|  (mean rms max)")
    print(f"  {'roll':<20}" + "".join(f"{fd(r['roll']):>22}" for r in rows))
    print(f"  {'pitch':<20}" + "".join(f"{fd(r['pitch']):>22}" for r in rows))
    print(f"  {'yaw':<20}" + "".join(f"{fd(r['yaw']):>22}" for r in rows))
    print("-" * 90)
    print(f"{'mean |current| [m/s]':<22}" + "".join(f"{r['cur']:>22.3f}" for r in rows))
    print(f"{'mean |wave| [m/s]':<22}" + "".join(f"{r['wav']:>22.3f}" for r in rows))
    print(f"{'mean speed [m/s]':<22}" + "".join(f"{r['spd']:>22.3f}" for r in rows))
    print(f"{'mean |u| (6-ch L2)':<22}" + "".join(f"{r['ueff']:>22.2f}" for r in rows))
    print(f"\n[plot] {out}")


if __name__ == "__main__":
    main()
