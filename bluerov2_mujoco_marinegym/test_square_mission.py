#!/usr/bin/env python3
"""Headless verification for (1) the irregular JONSWAP wave spectrum and
(2) the autonomous square-trajectory mission with auto-recording."""
import os
import sys
import csv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import mujoco

import hydro as H
import disturbances as D
import controller as C
from recorder import Recorder
from mission import SquareMission

XML = os.path.join(HERE, "bluerov.xml")


def test_spectrum():
    Hs, Tp, n = 0.20, 4.0, 30
    specs = D.jonswap_wave_specs(Hs=Hs, Tp=Tp, n=n, seed=0)
    assert len(specs) == n, f"expected {n} components, got {len(specs)}"
    # energy: a_i = U_i/omega_i ; Hs_est = 4*sqrt(sum a_i^2/2)
    U = np.array([s["U"] for s in specs])
    om = 2 * np.pi / np.array([s["T"] for s in specs])
    a = U / om
    Hs_est = 4.0 * np.sqrt(np.sum(a ** 2 / 2.0))
    assert abs(Hs_est - Hs) < 0.02, f"Hs mismatch: {Hs_est:.3f} vs {Hs}"
    # frequencies distinct (no equal-spacing repeat); reproducible
    assert np.allclose([s["U"] for s in D.jonswap_wave_specs(seed=0)],
                       [s["U"] for s in D.jonswap_wave_specs(seed=0)]), "not reproducible"

    # build a field, sample surface wave velocity over 300 s
    field = D.DisturbanceField(waves=specs, seed=0)
    t = np.arange(0.0, 300.0, 0.1)
    pos = np.array([0.0, 0.0, field.z_surface])           # surface -> no depth decay
    v = np.array([field.wave_velocity(ti, pos) for ti in t])
    mag = np.linalg.norm(v, axis=1)
    assert np.all(np.isfinite(v)) and mag.max() < 0.5, f"unbounded field max={mag.max():.3f}"
    # irregularity: autocorrelation of vx must NOT re-peak near 1 at long lags
    vx = v[:, 0] - v[:, 0].mean()
    full = np.correlate(vx, vx, "full")
    ac = full[full.size // 2:]                             # zero lag at index 0
    ac = ac / ac[0]                                        # normalize to 1 at lag 0
    long_lag = np.abs(ac[int(8.0 / 0.1):]).max()          # lags > 8 s
    assert long_lag < 0.5, f"signal recurs (autocorr {long_lag:.2f} at long lag) -> too regular"
    print(f"[spectrum] OK  n={n}  Hs_est={Hs_est:.3f}  max|v|={mag.max():.3f}  "
          f"long-lag autocorr={long_lag:.2f} (irregular)")
    H.Hydrodynamics.uninstall()


def test_square_mission():
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    field = D.DisturbanceField(waves=D.jonswap_wave_specs(seed=0), seed=0)
    field.enabled = False                                  # clean tracking assertions
    hydro = H.Hydrodynamics(model, disturbance=field).install()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    ctrl = C.PoseController(model, mode="pid", buoyancy_ff=hydro)
    rec = Recorder("/tmp/sqtest", tag="sq")
    S, laps, speed = 0.25, 2, 0.12
    mission = SquareMission(ctrl, rec, hydro, bid, size=S, laps=laps, speed=speed, log_hz=50)
    data.qpos[:3] = [0.15, 0.15, 0.0]                      # small offset -> quick approach
    mujoco.mj_forward(model, data)

    while not mission.done and data.time < 60.0:
        mission.step(model, data)
        mujoco.mj_step(model, data)
    assert mission.done, f"mission did not finish (phase={mission.phase}, t={data.time:.1f})"
    assert np.all(np.isfinite(data.qpos)), "sim went non-finite"

    with open(rec.path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 100, f"too few logged rows: {len(rows)}"
    px = np.array([float(r["px"]) for r in rows])
    py = np.array([float(r["py"]) for r in rows])
    # tracked the square: reaches the far +x/+y corner (~S) and returns near 0
    assert px.max() > 0.18 and py.max() > 0.18, f"never reached far corner ({px.max():.2f},{py.max():.2f})"
    assert px.min() < 0.08 and py.min() < 0.08, f"never returned near origin ({px.min():.2f},{py.min():.2f})"
    # stayed near the square (not wildly off)
    assert px.max() < 0.45 and py.max() < 0.45, "tracking diverged"
    H.Hydrodynamics.uninstall()
    print(f"[mission] OK  laps={laps}  rows={len(rows)}  "
          f"x∈[{px.min():.2f},{px.max():.2f}] y∈[{py.min():.2f},{py.max():.2f}]  saved {os.path.basename(rec.path)}")


def main():
    test_spectrum()
    test_square_mission()
    print("\nSQUARE+SPECTRUM TEST PASSED")


if __name__ == "__main__":
    main()
