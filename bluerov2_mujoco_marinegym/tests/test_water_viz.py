"""Tests for the animated pool water surface (water_viz.py) — VISUAL ONLY.

Run:  POOL_TAGS=1 python tests/test_water_viz.py     (env is set below if unset)

Checks:
  1. the POOL_TAGS scene exposes `pool_water_surface` as a heightfield geom;
  2. writing wave elevations into hfield_data + re-uploading actually changes the
     rendered surface, and successive times animate it (offscreen, EGL);
  3. animating the hfield EVERY physics step does NOT change dynamics — a rollout
     against the plain flat model is byte-identical (the core VISUAL-ONLY guarantee).
"""
import os
os.environ.setdefault("POOL_TAGS", "1")
os.environ.setdefault("ROV_MODEL", "heavy")
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import rov_model as RM          # noqa: E402
import water_viz                # noqa: E402
import disturbances as D        # noqa: E402
import hydro as H              # noqa: E402


def _load(path=None):
    m = mujoco.MjModel.from_xml_path(path or RM.XML_PATH)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def test_hfield_present():
    m, _ = _load()
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "pool_water_surface")
    assert gid >= 0, "pool_water_surface geom missing (regenerate with tools/gen_pool_apriltags.py)"
    assert int(m.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_HFIELD), \
        "water geom is not a heightfield (did --no-water-anim get used?)"
    assert water_viz.make_surface(m) is not None
    print("[ok] hfield water surface present")


def test_animation_renders():
    m, d = _load()
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "pool_water_surface")
    hfid = int(m.geom_dataid[gid])
    adr = int(m.hfield_adr[hfid]); nr = int(m.hfield_nrow[hfid]); nc = int(m.hfield_ncol[hfid])
    cx, cy, pz = (float(v) for v in m.geom_pos[gid])
    mean_z = pz + 0.5 * float(m.hfield_size[hfid][2])       # surface z is data-driven, not hardcoded
    r = mujoco.Renderer(m, 480, 720)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    # Fixed close-ish oblique framing (sees a ~few-metre patch around the centre) + a SHORT
    # synthetic wavelength (~nc/4 cycles, i.e. a few cells per crest) + near-full amplitude, so
    # several crests are always in view whether the pool is 2.6 m or the 48 m "infinite" floor.
    cam.lookat[:] = [cx, cy, mean_z]; cam.distance = 4.2; cam.azimuth = 56; cam.elevation = -22

    def render(vals):
        m.hfield_data[adr:adr + nr * nc] = np.clip(vals, 0, 1).astype(np.float32).ravel()
        mujoco.mjr_uploadHField(m, r._mjr_context, hfid)
        r.update_scene(d, cam)
        return r.render().astype(np.int16)

    c = np.linspace(0, 2 * np.pi * max(3, nc // 4), nc)     # ~nc/4 cycles -> a few cells per crest
    flat = np.full((nr, nc), 0.5)
    w1 = 0.5 + 0.45 * np.sin(c)[None, :].repeat(nr, 0)
    w2 = 0.5 + 0.45 * np.sin(c + 1.5)[None, :].repeat(nr, 0)
    img_flat, img_w1, img_w2 = render(flat), render(w1), render(w2)
    r.close()
    d_fw = float(np.abs(img_w1 - img_flat).mean())
    d_12 = float(np.abs(img_w1 - img_w2).mean())
    assert d_fw > 1.0, f"waves did not change the surface (diff {d_fw:.2f}); hfield upload broken?"
    assert d_12 > 0.5, f"phase shift did not animate (diff {d_12:.2f}); per-frame upload broken?"
    print(f"[ok] hfield animates (waves vs flat {d_fw:.1f}, phase1 vs phase2 {d_12:.1f})")


def test_dynamics_inert():
    ctrl = None

    def rollout(path, animate, n=2000):
        m, d = _load(path)
        field = D.DisturbanceField(seed=0); field.enabled = True
        H.Hydrodynamics(m, disturbance=field).install()
        surf = water_viz.make_surface(m, lam_target=0.9) if animate else None
        u = np.array([8., -6., 7., -5., 4., -3., 6., -4.], float)[:m.nu]
        traj = np.empty((n, m.nq + m.nv))
        for k in range(n):
            d.ctrl[:] = u
            mujoco.mj_step(m, d)
            if surf is not None:
                surf.update(field, d.time, enabled=True)   # write hfield_data every step
            traj[k, :m.nq] = d.qpos
            traj[k, m.nq:] = d.qvel
        H.Hydrodynamics.uninstall()
        return traj

    plain = os.path.join(HERE, RM._CFG["xml"])
    base = rollout(plain, False)
    anim = rollout(RM.XML_PATH, True)               # POOL_TAGS hfield scene, animated each step
    dmax = float(np.max(np.abs(base - anim)))
    assert dmax == 0.0, f"animated water perturbed dynamics! max|delta| = {dmax:.3e}"
    print(f"[ok] dynamics inert: max|delta(qpos,qvel)| = {dmax:.1e} over 2000 steps")


if __name__ == "__main__":
    test_hfield_present()
    test_animation_renders()
    test_dynamics_inert()
    print("ALL WATER_VIZ TESTS PASSED")
