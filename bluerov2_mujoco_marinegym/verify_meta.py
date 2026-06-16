#!/usr/bin/env python3
"""Verify the per-run disturbance/run-manifest sidecar: completeness + REPRODUCIBILITY.

1. DisturbanceField.to_meta() carries seed/config + the exact kick event schedule.
2. Reproduction round-trip: rebuild a field from the meta -> identical kicks + waves.
3. Recorder writes <run>.meta.json beside the CSV, CSV format unchanged.
4. teleop._controller_meta detects the live solver (acados/ipopt).
"""
import json
import os
import tempfile

import numpy as np

import disturbances as D
from recorder import Recorder, build_run_meta, record_row, RECORD_FIELDS


def test_to_meta_and_reproduce():
    field = D.DisturbanceField(waves=D.jonswap_wave_specs(seed=0), seed=0)
    meta = field.to_meta()
    assert meta["seed"] == 0 and meta["schema_version"] == 1
    assert len(meta["waves"]) == 30, meta["waves"]
    ke = meta["kicks"]
    assert ke["n_events"] == len(ke["events"]) > 0
    e0 = ke["events"][0]
    assert {"t_start", "t_end", "fx", "fy", "fz"} <= set(e0)
    assert abs((e0["t_end"] - e0["t_start"]) - ke["params"]["duration"]) < 1e-9

    # --- reproduction round-trip: rebuild from the meta alone
    rep = D.DisturbanceField(current=meta["current"], waves=meta["waves"],
                             kicks=meta["kicks"]["params"], z_surface=meta["z_surface"],
                             horizon=meta["horizon"], seed=meta["seed"])
    assert np.allclose(rep._kick_starts, field._kick_starts), "kick schedule not reproduced"
    assert np.allclose(rep._kick_forces, field._kick_forces), "kick forces not reproduced"
    # waves reproduce the water velocity field exactly
    for t, pos in [(3.3, np.zeros(3)), (12.7, np.array([0.4, -0.2, 0.0]))]:
        assert np.allclose(rep.wave_velocity(t, pos), field.wave_velocity(t, pos), atol=1e-9)
    print(f"[1+2] to_meta + reproduction OK  ({meta['kicks']['n_events']} kick events, "
          f"30 wave comps; kicks+waves reproduced exactly)")


def test_recorder_sidecar():
    field = D.DisturbanceField(waves=D.jonswap_wave_specs(seed=0), seed=0)
    with tempfile.TemporaryDirectory() as d:
        rec = Recorder(d, tag="square_dobmpc")
        rec.set_meta(build_run_meta(
            disturbance=field,
            controller=dict(type="dobmpc", solver="acados", N=60, ctrl_hz=20.0),
            trajectory=dict(kind="square", size=1.0, speed=0.15, laps=2),
            run=dict(started="2026-06-15 20:00:00", sim_dt=0.002)))
        p = rec.start()
        rec.log({k: 0.0 for k in RECORD_FIELDS})            # one dummy row
        rec.stop()
        mp = p[:-4] + ".meta.json"
        assert os.path.exists(mp), "sidecar not written"
        meta = json.load(open(mp))
        assert meta["controller"]["solver"] == "acados"
        assert meta["trajectory"]["kind"] == "square" and meta["trajectory"]["laps"] == 2
        assert meta["disturbance"]["seed"] == 0
        assert meta["disturbance"]["kicks"]["n_events"] > 0
        # CSV format unchanged: header == RECORD_FIELDS
        with open(p) as f:
            hdr = f.readline().strip().split(",")
        assert hdr == RECORD_FIELDS, "CSV header changed!"
    print(f"[3] recorder sidecar OK  (<run>.meta.json written, CSV header unchanged, "
          f"solver+trajectory+disturbance captured)")


def test_controller_meta_detects_solver():
    import mujoco
    import dobmpc.params as P
    from teleop import _controller_meta
    model = mujoco.MjModel.from_xml_path("bluerov.xml")
    import hydro as H
    field = D.DisturbanceField(waves=D.jonswap_wave_specs(seed=0), seed=0)
    hydro = H.Hydrodynamics(model, disturbance=field).install()
    P.SOLVER = "acados"
    from dobmpc_controller import DOBMPCController
    ctrl = DOBMPCController(model, hydro=hydro, mode="dobmpc")
    m = _controller_meta(ctrl, "dobmpc")
    assert m["solver"] == "acados" and m["N"] == P.MPC_N and m["mode"] == "dobmpc", m
    print(f"[4] controller meta OK  {m}")
    H.Hydrodynamics.uninstall()


if __name__ == "__main__":
    test_to_meta_and_reproduce()
    test_recorder_sidecar()
    test_controller_meta_detects_solver()
    print("\nMETA SIDECAR VERIFICATION PASSED")
