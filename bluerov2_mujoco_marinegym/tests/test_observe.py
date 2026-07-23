"""Tests for the --observe (free-drift) mode in teleop.py.

Run:  python tests/test_observe.py        (headless; no viewer needed)

The mode releases the ROV from rest and lets the current+waves carry it, with an
optional recenter to keep it in view. These checks pin down its two contracts:

  1. UNCONTROLLED — with `Teleop.observe = True`, drive keys (W/S/Q/E/R/F/A/D/Z/C/X)
     are ignored so no thrust is ever commanded; only G (disturbance) and H (recenter)
     do anything.
  2. RECENTER IS STATE-ONLY — `_maybe_recenter` restores the release qpos and zeros
     qvel (like the viewer's Reset button), triggers only past the radius (or when
     forced), and NEVER mutates the physics model. It's a viewing convenience, not a
     dynamics change.

Plus a determinism sanity check: a free-drift rollout (thrust 0, disturbances on) is
reproducible, and byte-identical to one run with the observe key-gate installed — i.e.
observe mode adds no dynamics side effects when recenter never fires.
"""
import os
os.environ.setdefault("ROV_MODEL", "heavy")
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import rov_model as RM          # noqa: E402
import hydro as H              # noqa: E402
import disturbances as D        # noqa: E402
import teleop as TL            # noqa: E402


def _load():
    m = mujoco.MjModel.from_xml_path(RM.XML_PATH)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def test_drive_keys_disabled():
    """In observe mode every drive key is a no-op -> ctrl stays exactly zero."""
    m, d = _load()
    field = D.DisturbanceField(seed=0)
    tp = TL.Teleop(m, d, verbose=False, disturbance=field)
    tp.observe = True
    for ch in "WSQERFADZCX":
        tp.on_key(ch)
        assert np.all(np.asarray(d.ctrl) == 0.0), f"key {ch} commanded thrust in observe mode"
    assert np.all(tp.wrench == 0.0), "observe mode latched a wrench"
    # H only requests a recenter (loop performs it); G still toggles the flow.
    tp.on_key("H")
    assert tp._recenter_request is True, "H did not request a recenter"
    was = field.enabled
    tp.on_key("G")
    assert field.enabled != was, "G no longer toggles disturbances in observe mode"
    print("[ok] drive keys disabled; H requests recenter; G still toggles the flow")


def test_recenter_state_only():
    """_maybe_recenter restores the release pose + zero velocity, fires only past the
    radius (or when forced), and leaves the physics MODEL untouched."""
    m, d = _load()
    field = D.DisturbanceField(seed=0); field.enabled = True
    hy = H.Hydrodynamics(m, disturbance=field).install()
    home = (d.qpos.copy(), d.qvel.copy())
    model_before = m.opt.timestep, m.body_mass.copy(), m.geom_size.copy(), hy.buoyancy, hy.mass

    # within the radius -> no recenter, state left alone
    d.qpos[0] = home[0][0] + 0.5           # 0.5 m < 1.2 m radius
    d.qvel[0] = 0.3
    assert _run_recenter(m, d, hy, home, 1.2) is False
    assert d.qpos[0] == home[0][0] + 0.5 and d.qvel[0] == 0.3, "recentered inside the radius"

    # drift past the radius -> recenter, qpos restored + qvel zeroed
    d.qpos[0] = home[0][0] + 2.0
    d.qvel[:] = 0.7
    assert _run_recenter(m, d, hy, home, 1.2) is True
    assert np.array_equal(d.qpos, home[0]), "qpos not restored to release pose"
    assert np.all(d.qvel == 0.0), "qvel not zeroed on recenter"

    # depth drift alone also triggers
    d.qpos[2] = home[0][2] + 1.5
    assert _run_recenter(m, d, hy, home, 1.2) is True

    # forced (manual H) triggers even at the release pose
    assert _run_recenter(m, d, hy, home, 1.2, force=True) is True

    model_after = m.opt.timestep, m.body_mass.copy(), m.geom_size.copy(), hy.buoyancy, hy.mass
    assert model_after[0] == model_before[0]
    assert np.array_equal(model_after[1], model_before[1])
    assert np.array_equal(model_after[2], model_before[2])
    assert model_after[3] == model_before[3] and model_after[4] == model_before[4]
    H.Hydrodynamics.uninstall()
    print("[ok] recenter restores state only (radius + force gates correct; model intact)")


def _run_recenter(m, d, hy, home, radius, force=False):
    return TL._maybe_recenter(m, d, hy, home, radius, force=force)


def test_free_drift_deterministic():
    """Free drift (thrust 0, disturbances on) is reproducible, and the observe key-gate
    adds no dynamics side effect (recenter never fires here)."""
    def rollout(gate, n=1500):
        m, d = _load()
        field = D.DisturbanceField(seed=0); field.enabled = True
        H.Hydrodynamics(m, disturbance=field).install()
        tp = TL.Teleop(m, d, verbose=False, disturbance=field)
        tp.observe = gate
        traj = np.empty((n, m.nq + m.nv))
        for k in range(n):
            if gate:                       # a mashed drive key must not perturb drift
                tp.on_key("W")
            mujoco.mj_step(m, d)           # ctrl stays 0 -> pure free drift
            traj[k, :m.nq] = d.qpos
            traj[k, m.nq:] = d.qvel
        H.Hydrodynamics.uninstall()
        return traj

    plain = rollout(gate=False)
    gated = rollout(gate=True)
    dmax = float(np.max(np.abs(plain - gated)))
    assert dmax == 0.0, f"observe key-gate changed the free-drift dynamics! max|delta|={dmax:.3e}"
    # sanity: the flow actually moves it (not a frozen body)
    disp = float(np.linalg.norm(gated[-1, :3] - gated[0, :3]))
    assert disp > 0.05, f"free drift barely moved ({disp:.3f} m); disturbance not acting?"
    print(f"[ok] free drift deterministic + observe-gate inert (drift {disp:.2f} m over 1500 steps)")


def test_water_boundary_recenter():
    """Default --observe boundary = the water volume: inside the water -> no recenter
    (even far past the old 1.2 m radius); leaving the water footprint or rising above
    the waterline -> recenter. Bare scenes (no water geom) yield bounds=None."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "scene_bluerov_heavy_tags.xml"))
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    b = TL._water_bounds(m)
    assert b is not None, "pool scene should expose water bounds"
    cx, cy, rx, ry, z_bot, z_top = b
    assert rx > 2.0 and ry > 2.0, f"water bounds unexpectedly small: {b}"
    home = (d.qpos.copy(), d.qvel.copy())

    d.qpos[0] = cx + 0.8 * rx          # deep inside the water, far past the old 1.2 m
    assert TL._maybe_recenter(m, d, None, home, None, bounds=b) is False, \
        "recentered while still inside the water"
    d.qpos[0] = cx + rx + 0.05         # just past the water edge
    assert TL._maybe_recenter(m, d, None, home, None, bounds=b) is True
    d.qpos[:] = home[0]
    d.qpos[2] = z_top + 0.05           # broaching above the waterline
    assert TL._maybe_recenter(m, d, None, home, None, bounds=b) is True

    plain = mujoco.MjModel.from_xml_path(os.path.join(HERE, RM._CFG["xml"]))
    assert TL._water_bounds(plain) is None, "bare plant scene must have no water bounds"
    print(f"[ok] water-boundary recenter (inside ±{rx:.1f}x±{ry:.1f} m free; edge/waterline snap)")


def test_direction_helpers():
    """Wave heading (beta) + current heading/speed (theta_c) are set correctly and IN
    PLACE (the object hydro references), so a re-draw takes effect without swapping fields."""
    import argparse
    a = argparse.Namespace(waves="spectrum", sea="0.20,4.0")

    # current vector: exact heading + speed
    v = TL._current_from_heading(0.25, 45.0)
    assert abs(np.hypot(v[0], v[1]) - 0.25) < 1e-9 and abs(v[2]) < 1e-12
    assert abs(np.degrees(np.arctan2(v[1], v[0])) - 45.0) < 1e-6

    # apply to a field IN PLACE (same object id)
    field = D.DisturbanceField(seed=0)
    oid = id(field)
    TL._apply_directions(field, a, 90.0, 135.0, 0.30)
    assert id(field) == oid, "directions must mutate the field in place (hydro holds the ref)"
    cx, cy, _ = field.current
    assert abs(np.hypot(cx, cy) - 0.30) < 1e-9
    assert abs(((np.degrees(np.arctan2(cy, cx)) - 135.0 + 180) % 360) - 180) < 1e-6
    # what hydro actually reads each step reflects the new heading
    field.enabled = True
    assert np.allclose(field.current_velocity(), field.current)
    # wave circular-mean heading ~ 90 deg (JONSWAP about beta with cos^2s spread)
    dirs = np.array([np.arctan2(d[1], d[0]) for (_U, _om, _k, d, _ph) in field.waves])
    cmean = np.degrees(np.arctan2(np.sin(dirs).mean(), np.cos(dirs).mean()))
    assert abs(((cmean - 90 + 180) % 360) - 180) < 20, f"wave mean {cmean:.1f}° not ~90°"

    # classic waves rotate rigidly by beta
    wc = TL._waves_at_heading(argparse.Namespace(waves="classic", sea="0.20,4.0"), 30.0)
    assert abs(wc[0]["heading_deg"] - 30.0) < 1e-9 and abs(wc[1]["heading_deg"] - 80.0) < 1e-9

    # default (+x) path unchanged: theta_c=0 -> (speed,0,0)
    assert np.allclose(TL._current_from_heading(0.20, 0.0), [0.20, 0.0, 0.0])
    print(f"[ok] direction helpers: current 135° / wave ~{cmean:.0f}° applied in place")


if __name__ == "__main__":
    test_drive_keys_disabled()
    test_recenter_state_only()
    test_free_drift_deterministic()
    test_water_boundary_recenter()
    test_direction_helpers()
    print("ALL OBSERVE TESTS PASSED")
