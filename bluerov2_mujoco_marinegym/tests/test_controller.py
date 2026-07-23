#!/usr/bin/env python3
"""Headless verification for the baseline go-to-origin PD/PID controller.

Starts the BlueROV2 at an offset pose and runs the controller to the global origin.
Checks convergence + stability + bounded pitch (no nose-over), and shows the PD-vs-PID
behaviour under a constant current (PID rejects it; PD keeps a steady-state offset).
Also prints metrics a future DOB-MPC will be compared against (RMS error, settling
time, max pitch, thrust energy).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import mujoco

import hydro as H
import disturbances as D
import controller as C

XML = os.path.join(HERE, "bluerov.xml")
START = (2.0, 1.5, -1.0, 45.0)        # x, y, z, yaw(deg)


def run_episode(mode, disturb, start=START, T_sec=40.0, seed=0):
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    field = D.DisturbanceField(seed=seed)
    field.enabled = disturb
    if disturb:                       # isolate the CONSTANT current (clean PID-vs-PD);
        field.use_waves = False       # waves/kicks are oscillatory/impulsive and would
        field.use_kicks = False       # swamp the steady-state offset both share.
    # This test drives the hard-coded rank-5 bluerov.xml plant, so pin BOTH of its
    # matching configs (they otherwise follow ROV_MODEL and would mismatch the plant):
    #  * hydro coeffs -> BlueROV.yaml (under ROV_MODEL=heavy_gripper the default YAML
    #    carries the payload's displaced volume -> +19 N buoyancy on this 11.2 kg
    #    plant, which blows the vehicle to the surface);
    #  * gains -> GAINS_BLUEROV2 (GAINS_HEAVY's 30 N surge on rank-5 would nose the
    #    vehicle over; the 6 N cap is load-bearing).
    hydro = H.Hydrodynamics(model, disturbance=field,
                            coeff_path=os.path.join(HERE, "marinegym_assets",
                                                    "BlueROV.yaml")).install()
    ctrl = C.PoseController(model, mode=mode, setpoint=(0, 0, 0), yaw_ref=0.0,
                            buoyancy_ff=hydro, gains=C.GAINS_BLUEROV2)
    sx, sy, sz, syaw = start
    data.qpos[:3] = [sx, sy, sz]
    half = np.radians(syaw) / 2.0
    data.qpos[3:7] = [np.cos(half), 0, 0, np.sin(half)]
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    n = int(T_sec / dt)
    bid = ctrl.bid
    errs = np.empty(n)
    max_pitch = 0.0
    energy = 0.0
    settle = None
    for k in range(n):
        forces, _ = ctrl.apply(model, data)
        mujoco.mj_step(model, data)
        e = ctrl.pos_error(data)
        errs[k] = e
        R = np.asarray(data.xmat[bid], float).reshape(3, 3)
        max_pitch = max(max_pitch, abs(C.PoseController._pitch_from_R(R)))
        energy += float(np.sum(np.abs(forces))) * dt
        if settle is None and e < 0.05:
            settle = k * dt
    H.Hydrodynamics.uninstall()
    tail = errs[-int(5.0 / dt):]                # last 5 s
    R_final = np.asarray(data.xmat[bid], float).reshape(3, 3)
    return dict(
        final_pos=float(errs[-1]),
        final_yaw=float(abs(ctrl.yaw_error(data))),
        rms_tail=float(np.sqrt(np.mean(tail ** 2))),
        max_pitch_deg=float(np.degrees(max_pitch)),
        final_pitch_deg=float(abs(np.degrees(C.PoseController._pitch_from_R(R_final)))),
        settle_s=settle,
        energy=energy,
        finite=bool(np.all(np.isfinite(data.qpos))),
        speed=float(np.linalg.norm(data.qvel)),
    )


def _row(name, m):
    st = f"{m['settle_s']:.1f}s" if m["settle_s"] is not None else "  -  "
    print(f"  {name:14s} final|p|={m['final_pos']:.3f} m  yaw={np.degrees(m['final_yaw']):+5.1f}°"
          f"  rms5s={m['rms_tail']:.3f}  settle={st}  pitch max={m['max_pitch_deg']:4.1f}°"
          f"/end={m['final_pitch_deg']:4.1f}°  E={m['energy']:6.1f}")


def main():
    print(f"Go-to-origin from start {START} (x,y,z,yawdeg); 40 s each\n")
    pid_still = run_episode("pid", disturb=False)
    pd_still = run_episode("pd", disturb=False)
    pid_dist = run_episode("pid", disturb=True)
    pd_dist = run_episode("pd", disturb=True)

    print("still water:")
    _row("PID", pid_still); _row("PD", pd_still)
    print("with constant current 0.2 m/s (waves/kicks off, to isolate SS rejection):")
    _row("PID", pid_dist); _row("PD", pd_dist)
    print()

    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")

    # still water: both converge to the origin, level, stable
    check(pid_still["finite"] and pid_still["final_pos"] < 0.10,
          f"PID still-water converges (<0.10 m): {pid_still['final_pos']:.3f}")
    check(pid_still["final_yaw"] < 0.10, f"PID yaw converges (<0.10 rad): {pid_still['final_yaw']:.3f}")
    # transient pitch from the surge->pitch coupling is inherent (underactuated hull);
    # what matters is it never tumbles and recovers to level by the end.
    check(pid_still["max_pitch_deg"] < 45.0,
          f"PID pitch never tumbled (<45°): {pid_still['max_pitch_deg']:.1f}")
    check(pid_still["final_pitch_deg"] < 8.0,
          f"PID recovered to level (end pitch <8°): {pid_still['final_pitch_deg']:.1f}")
    check(pid_still["speed"] < 0.3, f"PID came to rest (|qvel|<0.3): {pid_still['speed']:.3f}")
    check(pd_still["final_pos"] < 0.15, f"PD still-water converges (<0.15 m): {pd_still['final_pos']:.3f}")
    # disturbance: PID rejects the constant current; PD keeps a steady-state offset
    check(pid_dist["rms_tail"] < 0.20, f"PID rejects current (rms5s<0.20 m): {pid_dist['rms_tail']:.3f}")
    check(pd_dist["rms_tail"] > pid_dist["rms_tail"] + 0.03,
          f"PD worse than PID under current (headroom): PD={pd_dist['rms_tail']:.3f} > PID={pid_dist['rms_tail']:.3f}")
    check(pid_dist["max_pitch_deg"] < 45.0 and pd_dist["max_pitch_deg"] < 45.0,
          f"pitch never tumbled under disturbance (<45°): PID={pid_dist['max_pitch_deg']:.1f} PD={pd_dist['max_pitch_deg']:.1f}")

    print("\n" + ("CONTROLLER TEST PASSED" if ok else "CONTROLLER TEST FAILED"))
    assert ok, "baseline controller did not meet convergence criteria"


if __name__ == "__main__":
    main()
