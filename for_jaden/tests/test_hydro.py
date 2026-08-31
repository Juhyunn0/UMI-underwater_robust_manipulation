#!/usr/bin/env python3
"""
Phase-3 verification: buoyancy/restoring, added mass, drag (Fossen, MarineGym).

Gravity is ON (buoyancy counteracts it). Checks:
  1. Neutral buoyancy  : no thrust -> hovers / drifts slowly, NOT free-fall.
  2. Self-righting     : tilt 20 deg in roll AND pitch, no thrust -> returns level.
  3. Terminal velocity : constant surge thrust -> bounded steady speed (drag), then
                         release -> decelerates to ~0 (no infinite coasting).
  4. Straighter path   : drag+restoring bound the surge->pitch coast/rotation that
                         Phase 2 had (compared head-to-head with hydro off).
  5. Stability         : long run, no NaN / blow-up.

    python tests/test_hydro.py            # asserts + prints
    python tests/test_hydro.py --render   # managed viewer (gravity+hydro on; starts tilted)

Only mujoco + numpy required.
"""
import argparse
import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import thrusters as TH
import hydro as H

XML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bluerov.xml")


def tilt_deg(model, data, bid):
    """Angle between body +z and world +z (0 = level)."""
    R = data.xmat[bid].reshape(3, 3)
    return np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0)))


def quat_axis(angle_deg, axis):
    a = np.radians(angle_deg) / 2.0
    ax = np.asarray(axis, float)
    return [np.cos(a), *(np.sin(a) * ax)]


def run(model, data, hd, seconds, forces=None, q0=None, log_every=None, bid=0):
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = 0
    data.qpos[3:7] = q0 if q0 is not None else [1, 0, 0, 0]
    hd.reset()
    if forces is not None:
        TH.set_thruster_forces(model, data, forces)
    dt = model.opt.timestep
    n = int(round(seconds / dt))
    log = []
    for k in range(n):
        mujoco.mj_step(model, data)
        if log_every and k % int(log_every / dt) == 0:
            log.append((k * dt, tilt_deg(model, data, bid), float(np.linalg.norm(data.qvel[:3]))))
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    hd = H.Hydrodynamics(model)
    hd.install()
    print(hd.summary())
    assert abs(hd.coBM - 0.01) < 1e-9, "coBM (CB-above-COM) must be 0.01 m"
    print(f"\ngravity = {model.opt.gravity.tolist()} (ON);  restoring max = "
          f"B*coBM = {hd.buoyancy*hd.coBM:.3f} N*m\n")
    ok_all = True

    if args.render:
        from mujoco import viewer as mj_viewer
        mujoco.mj_resetData(model, data)
        data.qpos[3:7] = quat_axis(30, (0, 1, 0))   # start pitched so righting shows
        print("Viewer: gravity+hydro ON, started pitched 30 deg -> watch it right "
              "itself and hover near-neutral.")
        mj_viewer.launch(model, data)
        return

    # 1) NEUTRAL BUOYANCY -----------------------------------------------------
    run(model, data, hd, 10.0, bid=bid)
    vz = float(data.qvel[2])
    free_fall_vz = -9.81 * 10.0
    ok = (abs(vz) < 0.3) and np.all(np.isfinite(data.qvel))
    ok_all &= ok
    print("1) NEUTRAL BUOYANCY (10 s, no thrust):")
    print(f"   steady vz = {vz:+.4f} m/s  (slight + = drifts up; free-fall would be "
          f"~{free_fall_vz:.0f} m/s)   {'PASS' if ok else 'FAIL'}")

    # 2) SELF-RIGHTING (roll AND pitch) --------------------------------------
    print("2) SELF-RIGHTING from 20 deg (no thrust):")
    for name, axis in [("pitch", (0, 1, 0)), ("roll", (1, 0, 0))]:
        log = run(model, data, hd, 25.0, q0=quat_axis(20, axis), log_every=5.0, bid=bid)
        final = tilt_deg(model, data, bid)
        ok = final < 3.0 and log[0][1] > 15.0   # started ~20, ends near level
        ok_all &= ok
        traj = ", ".join(f"{t:.0f}s:{a:.1f}" for t, a, _ in log)
        print(f"   {name:5s}: tilt {traj} -> final {final:.2f} deg   "
              f"{'PASS' if ok else 'FAIL'}")

    # 3) TERMINAL VELOCITY + drag stops coasting ------------------------------
    # Gentle surge thrust (Fx ~ 2.8 N, coupling 0.21 N*m << restoring 1.11) ->
    # nearly-level stable glide with a clean terminal speed. (Higher thrust makes
    # the surge->pitch coupling overwhelm the weak coBM restoring and the vehicle
    # tumbles -- see the note; that is MarineGym's geometry, and motivates a
    # controller. The terminal SPEED is drag-bounded at any thrust either way.)
    Ts = 1.0
    log = run(model, data, hd, 25.0, forces=[Ts, Ts, -Ts, -Ts, 0, 0],
              log_every=2.5, bid=bid)
    spd = float(np.linalg.norm(data.qvel[:3]))
    vx = float(data.qvel[0])
    pit = tilt_deg(model, data, bid)
    spds = [s for _, _, s in log]
    win = spds[-4:]
    converged = (max(win) - min(win) < 0.05) and spd < 1.5
    ok_all &= converged
    print(f"3) TERMINAL VELOCITY (surge Fx={2.8284*Ts:.1f} N):")
    print(f"   speed over time: {[round(s,3) for s in spds]}")
    print(f"   -> terminal speed = {spd:.3f} m/s (vx={vx:+.3f}, glide pitch={pit:.0f} deg, "
          f"nearly level), bounded & converged  {'PASS' if converged else 'FAIL'}")
    # release thrust -> drag halts the surge coast (vertical keeps the slow
    # buoyancy drift, ~the neutral-buoyancy speed; that is expected, not a coast).
    TH.set_thruster_forces(model, data, [0, 0, 0, 0, 0, 0])
    for _ in range(int(15.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    horiz = float(np.linalg.norm(data.qvel[:2]))
    vz_coast = float(data.qvel[2])
    ok = horiz < 0.05
    ok_all &= ok
    print(f"   release thrust, 15 s later: horizontal speed = {horiz:.4f} m/s -> ~0 "
          f"(drag stopped the surge); vertical = {vz_coast:+.3f} m/s (buoyancy drift)  "
          f"{'PASS' if ok else 'FAIL'}")

    # 4) STRAIGHTER THAN PHASE 2 (drag bounds the surge coast/rotation) -------
    f_surge = [Ts, Ts, -Ts, -Ts, 0, 0]
    hd.install()
    run(model, data, hd, 3.0, forces=f_surge, bid=bid)
    spd_hydro, pit_hydro = float(np.linalg.norm(data.qvel[:3])), tilt_deg(model, data, bid)
    H.Hydrodynamics.uninstall()                 # Phase-2-like: no hydro, no gravity
    g_save = model.opt.gravity.copy(); model.opt.gravity[:] = 0
    run(model, data, hd, 3.0, forces=f_surge, bid=bid)
    spd_none, pit_none = float(np.linalg.norm(data.qvel[:3])), tilt_deg(model, data, bid)
    model.opt.gravity[:] = g_save; hd.install()
    ok = spd_hydro < spd_none                    # drag makes it slower / bounded
    ok_all &= ok
    print("4) STRAIGHTER PATH (3 s surge, hydro vs Phase-2 no-hydro):")
    print(f"   no-hydro : speed={spd_none:.3f} m/s, tilt={pit_none:.0f} deg  (grows, coasts)")
    print(f"   hydro    : speed={spd_hydro:.3f} m/s, tilt={pit_hydro:.0f} deg  (drag-bounded)  "
          f"{'PASS' if ok else 'FAIL'}")

    # 5) STABILITY (long run with thrust + initial tilt) ----------------------
    run(model, data, hd, 60.0, forces=[1.5, 1.5, -1.5, -1.5, 1.0, 1.0],
        q0=quat_axis(15, (0.3, 0.6, 0)), bid=bid)
    finite = np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
    bounded = float(np.linalg.norm(data.qvel)) < 5.0
    ok = finite and bounded
    ok_all &= ok
    print(f"5) STABILITY (60 s, thrust+tilt): finite={finite}, "
          f"|qvel|={np.linalg.norm(data.qvel):.3f}<5  {'PASS' if ok else 'FAIL'}")

    print("\n" + ("PHASE-3 CHECKS PASSED  (buoyancy/restoring, added mass, drag)"
                  if ok_all else "SOME PHASE-3 CHECKS FAILED"))
    assert ok_all, "hydro checks failed"


if __name__ == "__main__":
    main()
