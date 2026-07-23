#!/usr/bin/env python3
"""Headless smoke test for teleop.py --viser plumbing (no browser needed).

Exercises the riskiest viser API assumptions and the refactor:
  * force_items / force_magnitudes / draw_force_arrows (local path unchanged)
  * build_viser_scene  (mjModel mesh extraction -> viser nodes, finite poses)
  * viser_draw_arrows  (validates add_arrows (N,2,3) shape)
  * build_viser_gui    (buttons/checkbox/slider/text construct without error)
  * the per-frame sync loop (step + update handle.position/.wxyz)
  * the GUI control path (on_key via the same Teleop used by the keyboard)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import mujoco
import teleop as TL
import hydro as H
import disturbances as D


def main():
    model = mujoco.MjModel.from_xml_path(TL.XML)
    data = mujoco.MjData(model)
    field = D.DisturbanceField(seed=0)
    field.enabled = True
    hydro = H.Hydrodynamics(model, disturbance=field).install()
    teleop = TL.Teleop(model, data, scale=1.0, verbose=False, disturbance=field)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    for _ in range(20):                       # populate hydro.components/.water
        mujoco.mj_step(model, data)

    # 1. shared force_items + magnitudes
    items = TL.force_items(hydro, teleop, data, bid)
    names = [n for n, *_ in items]
    assert "thrust" in names and "buoyancy" in names and "current" in names, names
    mags = TL.force_magnitudes(hydro, teleop, data, bid)
    assert mags["buoyancy"] > 100.0, mags     # ~111 N up
    print(f"[1] force_items OK ({len(items)} arrows); "
          f"buoy={mags['buoyancy']:.0f}N drag={mags['drag']:.1f}N")

    # 2. local mjv drawer still works (refactor regression)
    scn = mujoco.MjvScene(model, maxgeom=400)
    TL.draw_force_arrows(scn, hydro, teleop, data, bid)
    assert scn.ngeom > 0, "local arrows produced no geoms"
    print(f"[2] local draw_force_arrows OK (ngeom={scn.ngeom})")

    # 3. viser scene build (real server on a test port)
    import viser
    server = viser.ViserServer(host="127.0.0.1", port=8099, verbose=False)
    try:
        handles = TL.build_viser_scene(server, model, data)
        assert len(handles) >= 1, "no geoms synced to viser"
        for gid, h in handles:
            assert np.all(np.isfinite(np.asarray(h.position))), gid
            assert np.all(np.isfinite(np.asarray(h.wxyz))), gid
        print(f"[3] build_viser_scene OK ({len(handles)} geoms, finite poses)")

        # 4. GUI builds (buttons/checkbox/slider/text)
        class A:
            scale, no_arrows, port = 1.0, False, 8099
        status, rec_status = TL.build_viser_gui(server, model, data, teleop, hydro, A())
        print("[4] build_viser_gui OK")

        # 4b. the new browser monitor: construct, push samples, refresh (renders the
        # velocity uplot + the time-coloured projection image with a planned square)
        plan = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]], float)
        vmon = TL._ViserMonitor(server, window_s=10.0, plan=plan)
        for k in range(20):
            vmon.push(0.05 * k, (0.1, -0.05, 0.0), (0.02 * k, 0.01 * k, 0.0))
        vmon._img_n = 0
        vmon.refresh()
        for h in (vmon.img_xy, vmon.img_xz, vmon.img_yz):
            assert h.image.ndim == 3 and h.image.shape[2] == 3, "projection image bad"
        # recording-start resets the time origin to the current sample
        vmon.push(1.5, (0.1, 0.0, 0.0), (0.0, 0.0, 0.0), recording=True)
        assert vmon.t0 == 1.5 and len(vmon.t) == 1, "recording start did not reset the clock"
        print(f"[4b] _ViserMonitor OK (numeric-time vel uplot + 3 separate {vmon.img_xy.image.shape} "
              f"projection panels, plan overlaid, clock resets on record)")

        # 5. arrows API accepts (N,2,3)
        TL.viser_draw_arrows(server, hydro, teleop, data, bid)
        print("[5] viser_draw_arrows OK (add_arrows shape accepted)")

        # 6. GUI control path == keyboard path
        teleop.on_key("W")
        assert abs(teleop.wrench[0]) > 0, "surge not latched"
        teleop.on_key("X")
        assert np.allclose(teleop.wrench, 0.0), "STOP did not zero wrench"
        print("[6] on_key control path OK (W latches, X stops)")

        # 7. a few sync-loop frames
        for _ in range(5):
            for _ in range(8):
                mujoco.mj_step(model, data)
            for gid, h in handles:
                h.position = tuple(map(float, data.geom_xpos[gid]))
                h.wxyz = TL._mat2wxyz(data.geom_xmat[gid])
            TL.viser_draw_arrows(server, hydro, teleop, data, bid)
            status.value = TL._status_text(
                teleop, TL.force_magnitudes(hydro, teleop, data, bid))
        assert np.all(np.isfinite(data.qpos)), "sim went non-finite"
        print("[7] sync loop OK (5 frames stepped + handles updated)")
    finally:
        server.stop()

    print("\nVISER SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
