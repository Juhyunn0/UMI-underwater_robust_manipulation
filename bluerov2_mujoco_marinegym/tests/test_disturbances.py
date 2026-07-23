#!/usr/bin/env python3
"""
Phase-4 verification: current / waves / kicks each distinct, DR bounded & stable.

All in FLU on top of Phase-3 hydro. Checks:
  1. Current  : no thrust -> vehicle drifts to the current velocity (vr, not v).
  2. Waves    : vehicle oscillates; amplitude decays with depth.
  3. Kicks    : occasional sudden velocity jolts at ~the set rate/magnitude.
  4. Distinct : each layer alone gives an identifiable signature.
  5. DR       : randomize() across seeds -> varied, bounded, NaN-free.
  6. Combined : all three for 60 s -> finite and bounded.

    python test_disturbances.py            # asserts + prints
    python test_disturbances.py --render    # viewer with all disturbances on

Only mujoco + numpy required.
"""
import argparse
import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hydro as H
import disturbances as D

XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bluerov.xml")


def make(disturbance=None):
    # A global mjcb_passive callback fires during from_xml_path's internal forward,
    # so uninstall any previous hydro before compiling a fresh model (else crash).
    H.Hydrodynamics.uninstall()
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    hd = H.Hydrodynamics(model, disturbance=disturbance)
    hd.install()
    return model, data, hd, bid


def simulate(model, data, hd, seconds, z0=0.0, record=False):
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = [0, 0, z0]
    data.qpos[3:7] = [1, 0, 0, 0]
    hd.reset()
    n = int(round(seconds / model.opt.timestep))
    log = np.zeros((n, 3)) if record else None
    for k in range(n):
        mujoco.mj_step(model, data)
        if record:
            log[k] = data.qvel[:3]
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()
    dt = mujoco.MjModel.from_xml_path(XML).opt.timestep
    ok_all = True

    if args.render:
        field = D.DisturbanceField(seed=0)
        model, data, hd, bid = make(field)
        from mujoco import viewer as mj_viewer
        mujoco.mj_resetData(model, data)
        print(field.summary())
        print("\nViewer: all disturbances ON (current+waves+kicks). Watch the drift, "
              "the wave oscillation, and the occasional kicks.")
        mj_viewer.launch(model, data)
        return

    # 1) CURRENT — drift to current velocity, proves vr (not v) is used ---------
    cur = np.array([0.20, 0.0, 0.0])
    field = D.DisturbanceField(current=cur, seed=1)
    field.use_waves = field.use_kicks = False
    model, data, hd, bid = make(field)
    # instantaneous: at rest, drag should push toward +current (vr = -cur != 0)
    mujoco.mj_resetData(model, data); hd.reset()
    mujoco.mj_step(model, data)
    a0 = np.array(data.qvel[:2])
    push_ok = a0[0] > 0 and abs(a0[1]) < 0.1 * abs(a0[0]) + 1e-6
    # steady state: horizontal velocity -> current velocity
    simulate(model, data, hd, 40.0)
    vxy = np.array(data.qvel[:2])
    drift_ok = abs(vxy[0] - cur[0]) < 0.03 and abs(vxy[1]) < 0.02
    ok = push_ok and drift_ok
    ok_all &= ok
    print("1) CURRENT (no thrust, current = [0.20, 0, 0] m/s):")
    print(f"   at rest, first-step horizontal vel = {a0.round(4).tolist()} -> pushed +x "
          f"(vr used, not v)  {'ok' if push_ok else 'FAIL'}")
    print(f"   after 40 s: horizontal vel = {vxy.round(3).tolist()} -> reaches current "
          f"velocity  {'PASS' if ok else 'FAIL'}")

    # 2) WAVES — oscillation + depth decay ------------------------------------
    # field-level decay of the default (multi-component) sea: unambiguous.
    deflt = D.DisturbanceField(seed=2)
    s1, s5 = deflt.wave_speed_at_depth(1.0), deflt.wave_speed_at_depth(5.0)
    field_decay_ok = s1 > s5 > 0
    # vehicle-level: a single moderate-period wave (strong decay), neutral
    # buoyancy so the vehicle holds its depth (no drift confound), compared at a
    # shallow vs deep depth -> the oscillation must be much weaker when deeper.
    wave1 = [dict(U=0.15, T=3.0, heading_deg=0.0, phase_deg=0.0)]   # surge-only
    field = D.DisturbanceField(waves=wave1, seed=2)
    field.use_current = field.use_kicks = False
    model, data, hd, bid = make(field)
    hd.buoyancy = hd.weight                       # neutral -> depth stays put
    field.z_surface = 1.5                         # body z=0 -> depth 1.5 (shallow)
    osc_sh = simulate(model, data, hd, 15.0, record=True)[:, 0].std()
    field.z_surface = 6.0                         # body z=0 -> depth 6 (deep)
    osc_dp = simulate(model, data, hd, 15.0, record=True)[:, 0].std()
    veh_decay_ok = osc_sh > 2 * osc_dp and osc_sh > 0.01
    ok = field_decay_ok and veh_decay_ok
    ok_all &= ok
    print("2) WAVES (no current/kicks):")
    print(f"   field wave speed (default sea): depth 1 m = {s1:.3f} > depth 5 m = "
          f"{s5:.3f} m/s (deep-water decay)  {'ok' if field_decay_ok else 'FAIL'}")
    print(f"   vehicle surge oscillation (single T=3 s wave): depth 1.5 m = {osc_sh:.3f} "
          f">> depth 6 m = {osc_dp:.3f} m/s  {'PASS' if ok else 'FAIL'}")

    # 3) KICKS — sudden jolts at ~set rate ------------------------------------
    field = D.DisturbanceField(kicks=dict(rate=0.4, fmin=25, fmax=45, duration=0.15),
                               seed=3)
    field.use_current = field.use_waves = False
    model, data, hd, bid = make(field)
    T = 40.0
    log = simulate(model, data, hd, T, record=True)
    spd = np.linalg.norm(log, axis=1)
    # detect jolts: speed rises by > 0.08 m/s within ~0.25 s, events >0.5 s apart
    w = int(0.25 / dt)
    rise = spd[w:] - spd[:-w]
    jolt_idx = np.where(rise > 0.08)[0]
    events = []
    for i in jolt_idx:
        if not events or (i - events[-1]) * dt > 0.5:
            events.append(i)
    n_sched = int(np.sum(field._kick_starts < T))
    rate_ok = 0.4 * n_sched <= len(events) <= 1.6 * n_sched + 1
    ok_all &= rate_ok
    print("3) KICKS (no current/waves, rate 0.4/s over 40 s):")
    print(f"   scheduled events = {n_sched}, detected jolts = {len(events)}, "
          f"max single-step speed = {spd.max():.3f} m/s  {'PASS' if rate_ok else 'FAIL'}")

    # 4) DISTINCTNESS — signatures differ -------------------------------------
    def sig(field_kwargs, layers, secs=25.0):
        f = D.DisturbanceField(seed=7, **field_kwargs)
        f.use_current, f.use_waves, f.use_kicks = layers
        m, d, h, _ = make(f)
        lg = simulate(m, d, h, secs, record=True)
        sp = np.linalg.norm(lg, axis=1)
        mean_h = np.linalg.norm(lg[:, :2].mean(0))      # steady drift
        osc = lg[len(lg)//2:, 0].std()                  # oscillation (late window)
        ww = int(0.25/dt); spike = (sp[ww:]-sp[:-ww]).max()  # biggest jolt
        return mean_h, osc, spike
    c_mean, c_osc, c_spk = sig(dict(current=(0.2, 0, 0)), (True, False, False))
    w_mean, w_osc, w_spk = sig({}, (False, True, False))
    k_mean, k_osc, k_spk = sig(dict(kicks=dict(rate=0.4, fmin=30, fmax=50, duration=0.15)),
                               (False, False, True))
    distinct_ok = (c_mean > 0.15 and c_mean > 3 * w_mean and          # current = drift
                   w_osc > 0.02 and w_osc > 3 * c_osc and             # wave = oscillation
                   k_spk > 0.1 and k_spk > 2 * c_spk)                 # kick = jolt
    ok_all &= distinct_ok
    print("4) DISTINCTNESS (drift / oscillation / jolt signatures):")
    print(f"   current: drift={c_mean:.3f}  osc={c_osc:.3f}  jolt={c_spk:.3f}")
    print(f"   wave   : drift={w_mean:.3f}  osc={w_osc:.3f}  jolt={w_spk:.3f}")
    print(f"   kick   : drift={k_mean:.3f}  osc={k_osc:.3f}  jolt={k_spk:.3f}  "
          f"{'PASS' if distinct_ok else 'FAIL'}")

    # 5) DOMAIN RANDOMIZATION — varied, bounded, stable -----------------------
    speeds, stable = [], True
    for seed in range(6):
        field, mp = D.randomize(seed)
        m, d, h, _ = make(field)
        simulate(m, d, h, 8.0)
        fin = np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))
        stable &= fin and np.linalg.norm(d.qvel) < 10
        speeds.append(np.linalg.norm(field.current))
    varied = np.std(speeds) > 0.02 and max(speeds) <= 0.4 + 1e-9
    ok = stable and varied
    ok_all &= ok
    print("5) DOMAIN RANDOMIZATION (6 seeds):")
    print(f"   sampled current speeds = {[round(s,3) for s in speeds]} (varied, <=0.4)")
    print(f"   all finite & bounded = {stable}  {'PASS' if ok else 'FAIL'}")

    # 6) COMBINED + STABILITY -------------------------------------------------
    field = D.DisturbanceField(seed=11)
    model, data, hd, bid = make(field)
    simulate(model, data, hd, 60.0)
    finite = np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
    bounded = np.linalg.norm(data.qvel) < 5.0
    ok = finite and bounded
    ok_all &= ok
    print("6) COMBINED all-on (60 s):")
    print(f"   finite={finite}, |qvel|={np.linalg.norm(data.qvel):.3f}<5  "
          f"{'PASS' if ok else 'FAIL'}")

    print("\n" + ("PHASE-4 CHECKS PASSED  (current / waves / kicks distinct, DR stable)"
                  if ok_all else "SOME PHASE-4 CHECKS FAILED"))
    assert ok_all, "disturbance checks failed"


if __name__ == "__main__":
    main()
