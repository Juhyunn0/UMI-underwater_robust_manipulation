"""Tests for the heavy_gripper variant (BlueROV2 Heavy + Newton gripper + MarineSitu C3).

Run:  ROV_MODEL=heavy_gripper python test_heavy_gripper.py     (env set below if unset)

Contracts pinned down:
  1. COMPOSITION — MuJoCo's composed subtree mass/COM/inertia match the transparent
     parallel-axis build-up in compute_payload_inertia.py (no hand-tuned literals).
  2. HEAVY BASELINE UNTOUCHED — the thruster sites/actuators are byte-identical to
     bluerov_heavy.xml (same allocation), and the heavy XML itself is not modified.
  3. BUOYANCY — net ~ -5.7 N (payload sinks, no trim foam): hydro numbers + an actual
     free-sink rollout.
  4. GRIPPER — actuator named "gripper" at ctrl index 8 (invisible to the name-based
     thruster code), jaws open to the 62 mm spec and mirror each other, dynamics
     stable at dt=2 ms.
  5. CONTROL — allocation still rank 6 from the 8 thrusters; PID gains selected for
     the variant; short station-keep run holds depth against the negative buoyancy.
"""
import os
os.environ.setdefault("ROV_MODEL", "heavy_gripper")
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rov_model as RM              # noqa: E402
import compute_payload_inertia as CP  # noqa: E402
import thrusters as T               # noqa: E402
import hydro as H                   # noqa: E402

assert RM.MODEL == "heavy_gripper", "run with ROV_MODEL=heavy_gripper"


def _load():
    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, RM._CFG["xml"]))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def test_composition():
    m, d = _load()
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    M, com_baked, I, _ = CP.compose()
    m_total, com_total, I_total = CP.compose_total()
    assert abs(float(m.body_subtreemass[bid]) - m_total) < 1e-6, "subtree mass mismatch"
    assert abs(m_total - RM.MASS) < 1e-6, "registry mass mismatch"
    # the generated frame is RE-ORIGINED at the total COM -> subtree COM must be ~0
    # (the origin==COM assumption the dobmpc predictor / params.ZG_MASS=0 rely on)
    assert np.linalg.norm(d.subtree_com[bid]) < 1e-4, \
        f"origin != COM: subtree_com {d.subtree_com[bid]}"
    # DIAGONAL inertia with identity iquat — REGRESSION GUARD: a fullinertia here gets
    # axis-PERMUTED by MuJoCo's principal-axis sort (Iyy > Izz > Ixx), and hydro.py's
    # mj_objectVelocity(mjOBJ_BODY) then measures nu in that permuted inertial frame
    # while applying drag via xmat -> crossed drag axes pump energy and the plant
    # explodes from a torque-free angular kick. iquat MUST stay identity.
    assert np.allclose(m.body_iquat[bid], [1, 0, 0, 0]), \
        f"body_iquat {m.body_iquat[bid]} != identity (inertial frame permuted!)"
    assert np.allclose(np.asarray(m.body_inertia[bid], float), np.diag(I), atol=1e-5)
    # baked COM sits at (baked - total) in the new frame
    assert np.allclose(np.asarray(m.body_ipos[bid], float), com_baked - com_total,
                       atol=1e-5)
    # registry inertia = diagonal of the TOTAL composite about the total COM
    assert np.allclose(np.diag(I_total), RM.INERTIA, atol=1e-4)
    print(f"[ok] composition: subtree {m_total:.4f} kg, origin==COM "
          f"(|subtree_com|={np.linalg.norm(d.subtree_com[bid]):.1e}), diag inertia + "
          f"identity iquat (Ixz={I_total[0,2]:+.5f} dropped, "
          f"{abs(I_total[0,2])/I_total[0,0]*100:.1f}% of Ixx)")


def test_thrusters_identical_to_heavy():
    """Thruster geometry must be preserved RELATIVE TO THE VEHICLE: the generated
    frame is shifted by -COM_total, so each site sits at (heavy pos - com_total);
    quats/gear/ctrlrange are byte-identical. (Allocation B legitimately differs from
    heavy by exactly the physical COM shift - the moment arms really did change.)"""
    mg, _ = _load()
    mh = mujoco.MjModel.from_xml_path(os.path.join(HERE, "bluerov_heavy.xml"))
    _, com_total, _ = CP.compose_total()
    for i in range(8):
        sg = mujoco.mj_name2id(mg, mujoco.mjtObj.mjOBJ_SITE, f"thruster_{i}")
        sh = mujoco.mj_name2id(mh, mujoco.mjtObj.mjOBJ_SITE, f"thruster_{i}")
        assert np.allclose(mg.site_pos[sg], mh.site_pos[sh] - com_total, atol=1e-5), \
            f"site {i}: relative geometry not preserved"
        assert np.array_equal(mg.site_quat[sg], mh.site_quat[sh]), f"site {i} quat differs"
        ag = mujoco.mj_name2id(mg, mujoco.mjtObj.mjOBJ_ACTUATOR, f"thr{i}")
        ah = mujoco.mj_name2id(mh, mujoco.mjtObj.mjOBJ_ACTUATOR, f"thr{i}")
        assert np.array_equal(mg.actuator_ctrlrange[ag], mh.actuator_ctrlrange[ah])
        assert np.array_equal(mg.actuator_gear[ag], mh.actuator_gear[ah])
    assert mh.nu == 8 and mg.nu == 9, "heavy must stay 8; gripper variant adds 1"
    print("[ok] 8 thruster sites preserved relative to the vehicle (frame shift only); "
          "actuators identical; heavy untouched")


def test_buoyancy_sinks():
    m, d = _load()
    hy = H.Hydrodynamics(m).install()
    net = hy.buoyancy - RM.MASS * H.G if hasattr(H, "G") else hy.buoyancy - RM.MASS * 9.81
    assert -6.2 < net < -5.2, f"net buoyancy {net:.2f} N (expected ~ -5.7)"
    # free release from rest, thrust 0 -> it must SINK
    for _ in range(int(2.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    H.Hydrodynamics.uninstall()
    assert d.qpos[2] < -0.05 and d.qvel[2] < 0, \
        f"payload vehicle should sink (z={d.qpos[2]:.3f}, vz={d.qvel[2]:.3f})"
    print(f"[ok] negative buoyancy: net {net:+.2f} N, sank to z={d.qpos[2]:.3f} m in 2 s")


def test_gripper_actuation():
    m, d = _load()
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    assert names[8] == "gripper" and names[:8] == [f"thr{i}" for i in range(8)]
    jl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "jaw_left")
    jr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "jaw_right")
    al, ar = m.jnt_qposadr[jl], m.jnt_qposadr[jr]
    d.ctrl[8] = 0.031                              # command full 62 mm opening
    for _ in range(int(1.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    ql, qr = float(d.qpos[al]), float(d.qpos[ar])
    assert ql > 0.028, f"left jaw only opened to {ql:.4f} m"
    assert abs(ql + qr) < 2e-3, f"jaws not mirrored: {ql:.4f} vs {qr:.4f}"
    assert np.all(np.isfinite(d.qpos)) and np.all(np.abs(d.qvel) < 50), "instability"
    print(f"[ok] gripper: index 8, opened to ±{ql:.4f} m (62 mm spec), mirrored, stable")


def test_allocation_and_pid_hold():
    m, d = _load()
    B, sites = T.allocation_matrix(m)
    assert B.shape == (6, 8) and np.linalg.matrix_rank(B, tol=1e-6) == 6, \
        "allocation must ignore the gripper actuator and stay rank 6"
    import controller as C
    assert C.DEFAULT_GAINS is C.GAINS_HEAVY_GRIPPER, "variant gains not selected"
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
    assert err < 0.08, f"PID failed to hold origin against -5.7 N (err {err:.3f} m)"
    print(f"[ok] allocation rank 6 (8 thrusters); PID holds origin at {err*100:.1f} cm "
          f"despite the sinking payload")


if __name__ == "__main__":
    test_composition()
    test_thrusters_identical_to_heavy()
    test_buoyancy_sinks()
    test_gripper_actuation()
    test_allocation_and_pid_hold()
    print("ALL HEAVY_GRIPPER TESTS PASSED")
