"""Tests for the heavy_c3 variant (BlueROV2 Heavy + MarineSitu C3, NO gripper).

heavy_c3 reflects exactly the lab's Onshape assembly (heavy + C3 on its C3-BR bracket;
the Newton gripper is not in Onshape yet). Run:
    ROV_MODEL=heavy_c3 python tests/test_heavy_c3.py      (env set below if unset)

Contracts pinned down:
  1. COMPOSITION — MuJoCo's composed subtree mass/COM/inertia match the transparent
     parallel-axis build-up in compute_payload_inertia.compose_c3() (no hand-tuned
     literals); frame re-origined at the composite COM (origin==COM), inertia DIAGONAL.
  2. NO GRIPPER — exactly 8 actuators (thr0..7), one free joint, no jaw bodies. This is
     the whole point of the variant vs heavy_gripper.
  3. HEAVY BASELINE UNTOUCHED — thruster sites preserved relative to the vehicle (frame
     shift only), actuators byte-identical, heavy XML not modified.
  4. BUOYANCY — net ~ -3.1 N (C3 sinks, no trim foam): hydro number + a free-sink rollout.
  5. CAMERAS — c3_center/left/right present, looking FORWARD and level (from the CAD).
  6. CONTROL — allocation rank 6 from the 8 thrusters; PID holds the origin.
"""
import os
os.environ.setdefault("ROV_MODEL", "heavy_c3")
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import rov_model as RM              # noqa: E402
import compute_payload_inertia as CP  # noqa: E402
import thrusters as T               # noqa: E402
import hydro as H                   # noqa: E402

assert RM.MODEL == "heavy_c3", "run with ROV_MODEL=heavy_c3"


def _load():
    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, RM._CFG["xml"]))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def test_composition():
    m, d = _load()
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    M, com, I, _ = CP.compose_c3()
    assert abs(float(m.body_subtreemass[bid]) - M) < 1e-6, "subtree mass mismatch"
    assert abs(M - RM.MASS) < 1e-6, "registry mass mismatch"
    # no articulated bodies -> composite COM IS the origin; subtree COM ~ 0
    assert np.linalg.norm(d.subtree_com[bid]) < 1e-5, "frame not re-origined at COM"
    assert np.allclose(m.body_ipos[bid], 0, atol=1e-5), "inertial pos != 0 (origin!=COM)"
    assert np.allclose(m.body_iquat[bid], [1, 0, 0, 0], atol=1e-6), \
        f"body_iquat {m.body_iquat[bid]} != identity (inertial frame permuted!)"
    assert np.allclose(np.asarray(m.body_inertia[bid], float), np.diag(I), atol=1e-5)
    assert np.allclose(np.diag(I), RM.INERTIA, atol=1e-4)
    print(f"[ok] composition: subtree {M:.4f} kg, origin==COM "
          f"(|subtree_com|={np.linalg.norm(d.subtree_com[bid]):.1e}), diag inertia + "
          f"identity iquat (Ixz={I[0,2]:+.5f} dropped, {abs(I[0,2])/I[0,0]*100:.1f}% of Ixx)")


def test_no_gripper():
    m, _ = _load()
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    assert names == [f"thr{i}" for i in range(8)], f"expected 8 thrusters only, got {names}"
    assert m.njnt == 1, f"expected 1 (free) joint, got {m.njnt} (jaw bodies present?)"
    for bad in ("jaw_left", "jaw_right"):
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bad) == -1, f"{bad} present"
    print("[ok] no gripper: 8 actuators (thr0..7), single free joint, no jaw bodies")


def test_thrusters_identical_to_heavy():
    mc, _ = _load()
    mh = mujoco.MjModel.from_xml_path(os.path.join(HERE, "bluerov_heavy.xml"))
    _, com, _, _ = CP.compose_c3()
    for i in range(8):
        sc = mujoco.mj_name2id(mc, mujoco.mjtObj.mjOBJ_SITE, f"thruster_{i}")
        sh = mujoco.mj_name2id(mh, mujoco.mjtObj.mjOBJ_SITE, f"thruster_{i}")
        assert np.allclose(mc.site_pos[sc], mh.site_pos[sh] - com, atol=1e-5), \
            f"site {i}: relative geometry not preserved"
        assert np.array_equal(mc.site_quat[sc], mh.site_quat[sh]), f"site {i} quat differs"
        ac = mujoco.mj_name2id(mc, mujoco.mjtObj.mjOBJ_ACTUATOR, f"thr{i}")
        ah = mujoco.mj_name2id(mh, mujoco.mjtObj.mjOBJ_ACTUATOR, f"thr{i}")
        assert np.array_equal(mc.actuator_ctrlrange[ac], mh.actuator_ctrlrange[ah])
        assert np.array_equal(mc.actuator_gear[ac], mh.actuator_gear[ah])
    assert mh.nu == 8 and mc.nu == 8, "both heavy and heavy_c3 have 8 actuators"
    print("[ok] 8 thruster sites preserved relative to the vehicle; heavy untouched")


def test_buoyancy_sinks():
    m, d = _load()
    hy = H.Hydrodynamics(m).install()
    g = getattr(H, "G", 9.81)
    net = hy.buoyancy - RM.MASS * g
    assert -3.6 < net < -2.6, f"net buoyancy {net:.2f} N (expected ~ -3.1)"
    for _ in range(int(2.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    H.Hydrodynamics.uninstall()
    assert d.qpos[2] < -0.03 and d.qvel[2] < 0, \
        f"C3 vehicle should sink (z={d.qpos[2]:.3f}, vz={d.qvel[2]:.3f})"
    print(f"[ok] negative buoyancy: net {net:+.2f} N, sank to z={d.qpos[2]:.3f} m in 2 s")


def test_cameras_forward():
    m, d = _load()
    for cn in ("c3_center", "c3_left", "c3_right"):
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cn)
        assert cid >= 0, f"camera {cn} missing"
        look = -d.cam_xmat[cid].reshape(3, 3)[:, 2]   # MuJoCo cameras look along -Z
        assert look[0] > 0.99, f"{cn} not looking forward: {np.round(look, 3)}"
        assert abs(look[2]) < 0.05, f"{cn} not level: {np.round(look, 3)}"
    # stereo pair straddles the centre in y (baseline horizontal)
    cl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "c3_left")
    cr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "c3_right")
    assert d.cam_xpos[cl][1] > d.cam_xpos[cr][1], "left/right cameras swapped in y"
    print("[ok] 3 cameras present, looking forward + level, stereo baseline horizontal")


def test_allocation_and_pid_hold():
    m, d = _load()
    B, _ = T.allocation_matrix(m)
    assert B.shape == (6, 8) and np.linalg.matrix_rank(B, tol=1e-6) == 6, \
        "allocation must be rank 6 from the 8 thrusters"
    import controller as C
    hy = H.Hydrodynamics(m).install()
    ctl = C.PoseController(m, mode="pid", setpoint=(0.0, 0.0, 0.0), yaw_ref=0.0,
                           buoyancy_ff=hy)
    d.qpos[:3] = [0.3, 0.2, -0.2]
    mujoco.mj_forward(m, d)
    for _ in range(int(20.0 / m.opt.timestep)):
        ctl.apply(m, d)
        mujoco.mj_step(m, d)
    H.Hydrodynamics.uninstall()
    err = float(np.linalg.norm(d.qpos[:3]))
    assert err < 0.08, f"PID failed to hold origin against -3.1 N (err {err:.3f} m)"
    print(f"[ok] allocation rank 6 (8 thrusters); PID holds origin at {err*100:.1f} cm")


if __name__ == "__main__":
    test_composition()
    test_no_gripper()
    test_thrusters_identical_to_heavy()
    test_buoyancy_sinks()
    test_cameras_forward()
    test_allocation_and_pid_hold()
    print("ALL HEAVY_C3 TESTS PASSED")
