#!/usr/bin/env python3
"""
Phase-2 verification: thruster directions, allocation matrix, FLU signs.

Gravity is disabled to isolate thrust. For each test command we (a) apply it
through the MJCF actuators and MEASURE the realized body wrench from MuJoCo's
acceleration, and (b) compare it to the analytic allocation B @ f. Then we
assert the FLU directions: surge->+x, sway->+y (LEFT), heave->+z (UP),
yaw->+Mz, roll->+Mx, and document the geometric couplings (the 4 horizontal
thrusters sit 0.0725 m below the COM, so surge couples to pitch and sway to
roll; the vectored-6 layout is rank-5, i.e. pitch is underactuated).

    python tests/test_thrusters.py            # asserts + print allocation matrix
    python tests/test_thrusters.py --render   # gravity-off viewer for manual checks

Only mujoco + numpy required. (--render needs a display; on macOS run with
`mjpython tests/test_thrusters.py --render`.)
"""
import argparse
import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import thrusters as T

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML = os.path.join(HERE, "bluerov.xml")
np.set_printoptions(precision=4, suppress=True, floatmode="fixed")


def measure_wrench(model, data, forces_N):
    """Realized body wrench [Fx,Fy,Fz,Mx,My,Mz] (FLU) for thruster forces [N].

    Gravity off, body at identity pose & rest: from a single mj_forward the free
    joint gives linear accel of the COM in world (== body at identity) and
    angular accel in the body frame, so F = m a, M = I_diag alpha.
    """
    mujoco.mj_resetData(model, data)
    data.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
    data.qvel[:] = 0
    T.set_thruster_forces(model, data, forces_N)
    mujoco.mj_forward(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    m = model.body_mass[bid]
    I = np.array(model.body_inertia[bid])
    F = m * np.array(data.qacc[:3])
    M = I * np.array(data.qacc[3:6])
    return np.concatenate([F, M])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--T", type=float, default=20.0, help="test thrust magnitude (N)")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(XML)
    model.opt.gravity[:] = 0.0          # isolate thrust
    data = mujoco.MjData(model)

    # ---- T200 curve / actuator-range consistency ----
    print("T200 thrust curve (MarineGym):")
    print(f"  max forward (u=+1) = {T.T200_MAX_FWD:+.3f} N")
    print(f"  max reverse (u=-1) = {T.T200_MAX_REV:+.3f} N")
    print(f"  fwd/rev asymmetry  = {T.T200_MAX_FWD/abs(T.T200_MAX_REV):.3f}   "
          f"deadband |u|<=0.075")
    cr = model.actuator_ctrlrange[0]
    assert np.allclose([cr[1], cr[0]], [T.T200_MAX_FWD, T.T200_MAX_REV], atol=1e-2), \
        "actuator ctrlrange does not match the T200 curve limits"
    print("  actuator ctrlrange matches curve  ✔")

    # ---- allocation matrix ----
    B, sites = T.allocation_matrix(model, data)
    rank = np.linalg.matrix_rank(B)
    _, sv, vt = np.linalg.svd(B)
    print("\nAllocation matrix B  (wrench = B @ thruster_forces), "
          "rows [Fx,Fy,Fz,Mx,My,Mz], cols thruster_0..5:")
    print(B)
    print(f"rank(B) = {rank}/6   singular values = {sv}")
    if rank < 6:
        print(f"  -> UNDERACTUATED: unreachable body-wrench direction ~ "
              f"{vt[-1].round(3)}  (pitch My is coupled to Fx, Fz; not independent)")
    print("\npseudo-inverse B^+ (wrench -> thruster forces):")
    print(np.linalg.pinv(B))

    # ---- per-DOF direction tests ----
    Tm = args.T
    LBL = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    tests = {  # name: (forces, dominant_idx, expected FLU sign)
        "surge  (forward, +x)": ([+Tm, +Tm, -Tm, -Tm, 0, 0], 0, +1),
        "sway   (left,    +y)": ([+Tm, -Tm, +Tm, -Tm, 0, 0], 1, +1),
        "heave  (up,      +z)": ([0, 0, 0, 0, +Tm, +Tm],      2, +1),
        "yaw    (+Mz,  CCW/z)": ([+Tm, -Tm, -Tm, +Tm, 0, 0],  5, +1),
        "roll   (+Mx)        ": ([0, 0, 0, 0, -Tm, +Tm],      3, +1),
    }
    print(f"\nPer-DOF direction tests (gravity off, T = {Tm:.0f} N):")
    print(f"  {'test':22s}{'measured wrench [Fx Fy Fz | Mx My Mz]':46s} result")
    all_ok = True
    wmeas = {}
    for name, (f, dom, sgn) in tests.items():
        f = np.array(f, float)
        w = measure_wrench(model, data, f)
        wmeas[name] = w
        analytic = B @ f
        ok = True
        # 1) MuJoCo realized wrench matches the analytic allocation (pipeline)
        ok &= np.allclose(w, analytic, atol=1e-3)
        # 2) dominant component has the expected FLU sign and is large
        ok &= (np.sign(w[dom]) == sgn) and (abs(w[dom]) > 1.0)
        # 3) components that are zero by the geometry must be ~0; the rest are
        #    genuine geometric couplings (reported below). Threshold relative to
        #    the dominant term so float noise in the site axes reads as zero.
        thr = max(1e-5, 1e-4 * abs(analytic[dom]))
        zeros = [i for i in range(6) if i != dom and abs(analytic[i]) < thr]
        ok &= all(abs(w[i]) < max(1e-3, 1e-4 * abs(analytic[dom])) for i in zeros)
        couplings = {i: w[i] for i in range(6)
                     if i != dom and abs(analytic[i]) >= thr}
        all_ok &= ok
        wl = f"[{w[0]:7.2f}{w[1]:7.2f}{w[2]:7.2f} |{w[3]:7.2f}{w[4]:7.2f}{w[5]:7.2f}]"
        note = ""
        if couplings:
            note = "  coupling: " + ", ".join(
                f"{LBL[i]}={w[i]/w[dom]:+.4f}*{LBL[dom]}" for i in couplings)
        print(f"  {name:22s}{wl:46s} {'PASS' if ok else 'FAIL'}{note}")

    # ---- tie the main couplings to the actual thruster z-offset below COM ----
    sid0 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "thruster_0")
    z0 = float(data.site_xpos[sid0][2])     # horizontal thruster height vs COM
    surge, sway = wmeas["surge  (forward, +x)"], wmeas["sway   (left,    +y)"]
    geo_ok = (abs(surge[4] / surge[0] - z0) < 1e-3 and       # My/Fx  ==  z0
              abs(sway[3] / sway[1] + z0) < 1e-3)            # Mx/Fy  == -z0
    all_ok &= geo_ok
    print(f"\nGeometric coupling: horizontal thrusters sit z0 = {z0:+.4f} m vs COM")
    print(f"  surge->pitch  My/Fx = {surge[4]/surge[0]:+.4f}  (expect z0  = {z0:+.4f})")
    print(f"  sway ->roll   Mx/Fy = {sway[3]/sway[1]:+.4f}  (expect -z0 = {-z0:+.4f})  "
          f"{'PASS' if geo_ok else 'FAIL'}")

    # ---- stepped-motion check: exercise throttle -> T200 curve -> ctrl -> step
    #      (the accel test above applies forces directly, skipping the curve).
    #      Short window: with no hydro damping yet, the surge->pitch coupling
    #      slowly curves the open-loop path, so we only check the early motion. -
    dt_move = 0.2
    print(f"\nStepped-motion check (gravity off, throttle commands, {dt_move:.1f} s):")
    moves = {
        "surge u=[+.5,+.5,-.5,-.5,0,0]": ([.5, .5, -.5, -.5, 0, 0], 0, +1),
        "heave u=[0,0,0,0,+.5,+.5]    ": ([0, 0, 0, 0, .5, .5],     2, +1),
    }
    for name, (u, dom, sgn) in moves.items():
        mujoco.mj_resetData(model, data)
        data.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
        T.step(model, data, throttles=u, n=int(round(dt_move / model.opt.timestep)))
        p = np.array(data.qpos[:3])
        ok = (np.sign(p[dom]) == sgn) and (int(np.argmax(np.abs(p))) == dom)
        all_ok &= ok
        print(f"  {name}: displacement = {p.round(3)} m  "
              f"-> largest motion +{'xyz'[dom]}  {'PASS' if ok else 'FAIL'}")

    # ---- underactuation: pitch cannot be commanded independently ----
    My_des = np.array([0, 0, 0, 0, 5.0, 0.0])     # ask for pure +My (pitch)
    forces, realized = T.set_wrench_command(model, data, My_des, B)
    print("\nUnderactuation check — request pure pitch wrench [0,0,0,0,5,0]:")
    print(f"  realized wrench = {realized}")
    print(f"  -> pitch is not independently controllable; allocator yields a "
          f"near-zero / heavily-coupled result (My_realized={realized[4]:+.3f}).")

    print("\n" + ("PHASE-2 CHECKS PASSED  (T200 curve, allocation, FLU signs, "
                  "couplings verified)" if all_ok else "SOME CHECKS FAILED"))
    assert all_ok, "direction/allocation checks failed"

    if args.render:
        from mujoco import viewer as mj_viewer
        print("\nGravity-off viewer: open the Control panel and drag a 'thr*' "
              "slider (thrust in N) to push one thruster; watch the FLU motion.")
        mujoco.mj_resetData(model, data)
        data.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
        mj_viewer.launch(model, data)


if __name__ == "__main__":
    main()
