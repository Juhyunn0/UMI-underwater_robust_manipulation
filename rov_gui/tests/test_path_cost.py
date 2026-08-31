#!/usr/bin/env python3
"""test_path_cost.py — the along/cross cost split (mode mpc_tuned).

Two halves, the same shape as test_mpcc.py. The ALGEBRA tests are pure numpy
and always run: they pin the one claim the whole feature rests on, that
rotating the weight matrix IS splitting the error, so an isotropic tuning is
bit-for-bit the baseline cost. The CLOSED-LOOP tests need acados and skip
without it.

READ THE CLOSED-LOOP NUMBERS AS DIRECTION, NOT MAGNITUDE. The offline plant
here is the controller's own prediction model, so it has no ESC deadband, no
tether, and (per the 2026-08-17 whole-loop fit in hw_mpc.yaml) 8-12x too
little horizontal drag. That is the regime where a push-vs-geometry knob
cannot be ranked on absolute numbers — what these tests can honestly show is
the SIGN of the effect and that the machinery is wired to the right angle.

    ~/miniforge3/envs/robust/bin/python rov_gui/tests/test_path_cost.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rov_gui.control.path_cost import (          # noqa: E402
    DEFAULT_TUNE, PathFrameWeights, path_frame_block, resolve_tune)
from rov_gui.control.path_geometry import (      # noqa: E402
    PathCursor, path_from_scenario)

SQUARE = {"kind": "square", "origin_ned": (0.0, 0.0), "size": 1.0,
          "size_y": 1.0, "rot_deg": 0.0, "laps": 1, "speed": 0.10}

# x = [x y z, phi theta psi, u v w, p q r]
Q_DEMO = np.array([300.0, 300.0, 150.0, 80.0, 80.0, 150.0,
                   20.0, 20.0, 20.0, 5.0, 5.0, 5.0])
R_DEMO = np.array([0.05, 0.05, 0.05, 0.01, 0.01, 0.005])


# ------------------------------------------------------------------- algebra
def test_the_rotated_weight_is_exactly_the_path_frame_split():
    """W = Rz diag(qa,qc) Rz^T  <=>  qa*along^2 + qc*cross^2.

    This is the feature. If it fails, `mpc_tuned` is penalising something
    that is not along-track and cross-track error, and every run flown under
    it is measuring an unnamed quantity."""
    rng = np.random.default_rng(7)
    for _ in range(500):
        psi = rng.uniform(-4.0, 4.0)
        qa, qc = rng.uniform(1.0, 2000.0), rng.uniform(1.0, 2000.0)
        e = rng.normal(size=2)
        t_hat = np.array([math.cos(psi), math.sin(psi)])
        n_hat = np.array([-math.sin(psi), math.cos(psi)])
        want = qa * (t_hat @ e) ** 2 + qc * (n_hat @ e) ** 2
        got = e @ path_frame_block(qa, qc, psi) @ e
        assert abs(got - want) < 1e-9 * max(1.0, abs(want)), \
            f"psi={psi:.3f} qa={qa:.1f} qc={qc:.1f}: {got} != {want}"


def test_isotropic_weights_reproduce_the_baseline_diagonal():
    """along_scale == cross_scale must be the UNTUNED cost, exactly.

    The parity that makes an A/B trustworthy: any difference a tuned run
    shows against the baseline has to come from the anisotropy and not from
    the machinery that applies it."""
    for psi in np.linspace(-math.pi, math.pi, 37):
        B = path_frame_block(123.0, 123.0, psi)
        assert np.allclose(B, 123.0 * np.eye(2), atol=1e-12), psi
    w = PathFrameWeights(Q_DEMO, R_DEMO, Q_DEMO,
                         {"along_scale": 1.0, "cross_scale": 1.0})
    for psi in np.linspace(-math.pi, math.pi, 17):
        assert np.allclose(w.stage_W(psi, 0.3), w.W_base, atol=1e-12)
        assert np.allclose(w.terminal_W(psi, 0.3), w.We_base, atol=1e-12)


def test_the_split_touches_the_xy_block_and_nothing_else():
    """Depth, attitude, velocity and the input weights must come through
    untouched — a corner-cutting knob that quietly re-tuned yaw would make
    every tuned run uninterpretable."""
    w = PathFrameWeights(Q_DEMO, R_DEMO, Q_DEMO, DEFAULT_TUNE)
    W = w.stage_W(0.7, 0.1)
    other = [i for i in range(W.shape[0]) if i not in (0, 1)]
    assert np.allclose(W[np.ix_(other, other)],
                       w.W_base[np.ix_(other, other)], atol=1e-12)
    assert np.allclose(W[np.ix_(other, (0, 1))], 0.0)
    # ...and the 2x2 block really is anisotropic
    ev = np.linalg.eigvalsh(W[:2, :2])
    assert abs(max(ev) / min(ev) - w.anisotropy) < 1e-9


def test_weights_stay_symmetric_and_positive_definite():
    """acados takes W into a Gauss-Newton Hessian; an asymmetric or
    indefinite one is a silently wrong QP, not an error."""
    w = PathFrameWeights(Q_DEMO, R_DEMO, Q_DEMO, DEFAULT_TUNE)
    for psi in np.linspace(-4.0, 4.0, 41):
        for W in (w.stage_W(psi, 0.2), w.terminal_W(psi, 0.2)):
            assert np.allclose(W, W.T, atol=1e-12)
            assert np.linalg.eigvalsh(W).min() > 0.0


def test_the_default_tuning_is_actually_anisotropic():
    w = PathFrameWeights(Q_DEMO, R_DEMO, Q_DEMO, None)
    assert w.anisotropy > 4.0, w.anisotropy
    assert w.q_along < w.q_xy < w.q_cross


def test_absolute_weights_override_the_scales():
    w = PathFrameWeights(Q_DEMO, R_DEMO, Q_DEMO,
                         {"q_along": 10.0, "q_cross": 2500.0,
                          "along_scale": 99.0, "cross_scale": 99.0})
    assert (w.q_along, w.q_cross) == (10.0, 2500.0)


def test_a_typo_in_the_tuning_block_raises():
    """The imu_dr rule. A silently ignored `cross_sale:` means the run flies
    the BASELINE cost while its meta says tuned, and the two CSVs are then
    indistinguishable from two runs of the same controller."""
    for bad in ({"cross_sale": 4.0}, {"crossscale": 4.0}, {"Q_cross": 1.0}):
        try:
            resolve_tune(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} was accepted")
    for bad in ({"cross_scale": 0.0}, {"along_scale": -1.0},
                {"q_cross": 0.0}):
        try:
            resolve_tune(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} was accepted")


def test_config_loads_the_block_and_the_new_mode_names():
    from rov_gui.control.geometry import MpcConfig

    cfg = MpcConfig.load(str(ROOT / "config" / "hw_mpc.yaml"))
    resolved = resolve_tune(cfg.mpc_tuned)
    assert resolved["cross_scale"] > resolved["along_scale"], resolved
    for m in ("mpc", "dobmpc", "mpc_tuned", "dobmpc_tuned", "pid"):
        MpcConfig(mode=m)          # dataclass accepts; load() validates
    import yaml

    raw = yaml.safe_load((ROOT / "config" / "hw_mpc.yaml").read_text())
    assert "mpc_tuned" in raw, "the shipped config must carry the block"


def test_the_worker_and_the_bridge_agree_on_the_mode_names():
    from rov_gui.control.mpc_bridge import HwDobMpc
    from rov_gui.control.workers import MpcWorker

    for m in ("mpc_tuned", "dobmpc_tuned"):
        assert m in MpcWorker.MODES, m
        assert m in HwDobMpc.MODES, m
    # the UI must offer them, or the mode exists only in a config file
    from rov_gui.widgets import trajectory as T

    src = Path(T.__file__).read_text()
    assert '"mpc_tuned"' in src and '"dobmpc_tuned"' in src


# --------------------------------------------------------------- the tangent
def test_the_plan_carries_the_path_tangent_under_a_fixed_heading():
    """The angle the split rotates by. Under heading_follow: false the
    vehicle crabs — one heading for the whole lap while the path turns 90
    degrees under it — so yaw_ned cannot stand in for the tangent."""
    path = path_from_scenario(SQUARE, fillet_m=0.15)
    s_g, v_g = path.speed_profile(0.10, 0.05, 0.05)
    cur = PathCursor(path, 0.10, s_g, v_g)
    # Park the cursor where the horizon actually reaches the corner. N*dt = 3 s
    # at ~0.1 m/s is only ~0.30 m of preview, and the first fillet starts at
    # s = 0.70 — worth knowing on its own: the split can only act on a corner
    # the horizon can see.
    px, py, _p, _k = path.sample(np.array([0.62]))
    for _ in range(80):
        cur.step([px[0], py[0], 0.5], 0.05)
    plan = cur.plan(61, 0.05, 0.5, yaw_fixed=0.3, heading_follow=False)
    assert plan.psi_path.shape == (61,)
    assert np.allclose(plan.yaw_ned, 0.3), "heading must stay fixed"
    turn = abs(float(plan.psi_path[-1] - plan.psi_path[0]))
    assert turn > math.radians(60.0), \
        f"the horizon crosses a corner; the tangent only turned {math.degrees(turn):.1f} deg"
    # ...and under heading_follow the two agree
    plan2 = cur.plan(61, 0.05, 0.5, yaw_fixed=0.3, heading_follow=True)
    assert np.allclose(np.unwrap(plan2.yaw_ned), plan2.psi_path, atol=1e-9)


def test_the_tangent_survives_the_corner_where_the_speed_brakes_to_zero():
    """Why the tangent is carried instead of derived from v_ned: with no
    fillet the speed profile pins v_ref at the creep floor through the
    vertex, so atan2(vy, vx) there is a direction taken from a number that
    was set by a deadlock guard."""
    path = path_from_scenario(SQUARE, fillet_m=0.0)
    s_g, v_g = path.speed_profile(0.10, 0.05, 0.05, v_creep=0.02)
    cur = PathCursor(path, 0.10, s_g, v_g)
    # park the cursor just before the first vertex
    px, py, _p, _k = path.sample(np.array([0.98]))
    for _ in range(3):
        cur.step([px[0], py[0], 0.5], 0.05)
    plan = cur.plan(61, 0.05, 0.5, yaw_fixed=0.0, heading_follow=False)
    speeds = np.hypot(plan.v_ned[0], plan.v_ned[1])
    assert speeds.min() <= 0.03, "the reference should be crawling at the vertex"
    d = np.abs(np.diff(plan.psi_path))
    assert d.max() > math.radians(45.0), \
        "the tangent must show the 90-degree vertex the speed hides"


# ----------------------------------------------------------------- closed loop
def _acados_or_skip():
    try:
        from rov_gui.control.mpc_bridge import HwDobMpc, import_dobmpc
        import_dobmpc("heavy_gripper")
        return HwDobMpc
    except Exception as e:                                    # noqa: BLE001
        print(f"  SKIP (no acados / dobmpc: {type(e).__name__}: {e})")
        return None


def _plant():
    """The controller's OWN prediction model as the plant. Honest about what
    that means: no deadband, no tether, 8-12x too little drag."""
    import casadi as ca
    from dobmpc.mpc import _f_casadi

    xs, us, ws = ca.SX.sym("x", 12), ca.SX.sym("u", 6), ca.SX.sym("w", 6)
    F = ca.Function("F", [xs, us, ws], [_f_casadi(xs, us, ws)])
    z = np.zeros(6)

    def rk4(x, u, dt=0.05, n=5):
        h = dt / n
        for _ in range(n):
            k1 = np.array(F(x, u, z)).ravel()
            k2 = np.array(F(x + h / 2 * k1, u, z)).ravel()
            k3 = np.array(F(x + h / 2 * k2, u, z)).ravel()
            k4 = np.array(F(x + h * k3, u, z)).ravel()
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x
    return rk4


_CTRL = {}


def _controller(HwDobMpc):
    """ONE acados build for the whole module: the weights are a runtime
    cost_set, so every variant reuses the same compiled solver."""
    if "c" not in _CTRL:
        from rov_gui.control.geometry import MpcConfig

        _CTRL["c"] = HwDobMpc("mpc", MpcConfig(), log=lambda m: None)
    return _CTRL["c"]


def fly(ctrl, mode, tune=None, fillet=0.15, speed=0.10, laps=1,
        lead_m=0.10, depth=0.5, max_ticks=4000):
    """Drive HwDobMpc exactly the way MpcWorker._advance_path_clock does.

    Returns ``(xy (T,2), theta (T,), u (T,6))``."""
    rk4 = _plant()
    path = path_from_scenario(dict(SQUARE, laps=laps), fillet_m=fillet)
    s_g, v_g = path.speed_profile(speed, 0.05, 0.05)
    cur = PathCursor(path, lead_m, s_g, v_g)
    ctrl.set_path_cost(tune)
    ctrl.mode = mode
    ctrl.reset()
    dt = float(ctrl.path_plan_dt)
    x = np.zeros(12)
    x0, y0, _p, _k = path.sample(np.array([0.0]))
    x[0], x[1], x[2] = float(x0[0]), float(y0[0]), depth
    xy, th, us = [], [], []
    for i in range(max_ticks):
        cur.step(x[:3], dt)
        plan = cur.plan(ctrl.path_plan_steps, dt, depth, 0.0, False)
        ctrl.set_path_plan_ned(plan)
        u, _info = ctrl.step(x[0:6], x[6:12], np.zeros(6), i * dt)
        x = rk4(x, np.asarray(u, float))
        xy.append((x[0], x[1]))
        th.append(cur.theta)
        us.append(np.asarray(u, float).copy())
        if cur.complete:
            break
    return np.array(xy), np.array(th), np.array(us)


def cross_track(xy, path, ds=0.002):
    """Signed-magnitude distance to the nearest point of ONE lap."""
    s = np.arange(0.0, path.lap_length, ds)
    cx, cy, _p, _k = path.sample(s)
    d = np.hypot(xy[:, 0][:, None] - cx[None, :], xy[:, 1][:, None] - cy[None, :])
    j = np.argmin(d, axis=1)
    return d[np.arange(len(xy)), j], s[j]


def corner_window(path, s_at, pad=0.20):
    """Mask of samples inside the first corner (its arc, plus ``pad`` either
    side) — where cutting happens and the only place it can be measured."""
    s = np.arange(0.0, path.lap_length, 0.002)
    _x, _y, _p, k = path.sample(s)
    arc = np.isfinite(k) & (np.abs(k) > 1e-6)
    if not arc.any():                       # un-filleted: the vertex itself
        a = b = path.lap_length / 4.0
    else:
        i0 = int(np.argmax(arc))            # the FIRST contiguous arc only —
        i1 = i0                             # s[arc][-1] would span the whole lap
        while i1 + 1 < arc.size and arc[i1 + 1]:
            i1 += 1
        a, b = float(s[i0]), float(s[i1])
    return (s_at > a - pad) & (s_at < b + pad), (a, b)


def test_isotropic_tuning_flies_the_same_trajectory_as_the_baseline():
    """The closed-loop half of the parity claim: mpc_tuned at 1.0/1.0 must
    reproduce mpc, so any later difference is the anisotropy and not the
    per-stage cost_set, the plan plumbing, or the mode switch."""
    H = _acados_or_skip()
    if H is None:
        return
    c = _controller(H)
    base, _t0, u0 = fly(c, "mpc")
    iso, _t1, u1 = fly(c, "mpc_tuned",
                       {"along_scale": 1.0, "cross_scale": 1.0})
    n = min(len(base), len(iso))
    dp = np.abs(base[:n] - iso[:n]).max()
    du = np.abs(u0[:n] - u1[:n]).max()
    assert dp < 1e-6, f"trajectories differ by {dp * 1000:.3f} mm"
    assert du < 1e-6, f"wrenches differ by {du:.2e} N"


def test_switching_back_to_the_baseline_restores_the_isotropic_weight():
    """A rotated W left behind after a mode switch would contaminate the
    NEXT run silently — the solver reports status 0 either way."""
    H = _acados_or_skip()
    if H is None:
        return
    c = _controller(H)
    a, _t, _u = fly(c, "mpc")
    fly(c, "mpc_tuned", {"along_scale": 0.1, "cross_scale": 20.0})
    b, _t2, _u2 = fly(c, "mpc")            # same as the first run?
    n = min(len(a), len(b))
    assert np.abs(a[:n] - b[:n]).max() < 1e-6, "baseline run was contaminated"


def test_the_split_cuts_the_corner_less_than_the_isotropic_cost():
    """The point of the mode. Direction only — see the module docstring on
    what this plant can and cannot rank."""
    H = _acados_or_skip()
    if H is None:
        return
    c = _controller(H)
    path = path_from_scenario(SQUARE, fillet_m=0.15)
    base, _tb, _ub = fly(c, "mpc")
    tuned, _tt, _ut = fly(c, "mpc_tuned", None)      # the shipped default
    eb, sb = cross_track(base, path)
    et, st = cross_track(tuned, path)
    mb, (a, b) = corner_window(path, sb)
    mt, _ = corner_window(path, st)
    cut_b, cut_t = float(eb[mb].max()), float(et[mt].max())
    print(f"    corner arc s=[{a:.2f},{b:.2f}]  cut base {cut_b * 1000:5.1f} mm"
          f"  ->  tuned {cut_t * 1000:5.1f} mm"
          f"   ({100 * (cut_t - cut_b) / cut_b:+.0f} %)")
    print(f"    lap p95 |cross|  base {np.percentile(eb, 95) * 1000:5.1f} mm"
          f"  ->  tuned {np.percentile(et, 95) * 1000:5.1f} mm")
    assert cut_t < cut_b, (
        f"tuned cut {cut_t * 1000:.1f} mm is not below baseline "
        f"{cut_b * 1000:.1f} mm")


def test_a_tuned_run_records_the_cost_it_actually_flew():
    """A tuned run whose meta says only `Q: [300, 300, ...]` is a run whose
    cost cannot be reconstructed — the measurement rule applied to a knob."""
    H = _acados_or_skip()
    if H is None:
        return
    c = _controller(H)
    c.set_path_cost({"along_scale": 0.2, "cross_scale": 5.0})
    c.mode = "dobmpc_tuned"
    m = c.meta()
    assert m["type"] == "dobmpc_tuned"
    assert m["cost_frame"].startswith("path")
    pc = m["path_cost"]
    assert pc["q_along"] == 0.2 * pc["q_xy_baseline"]
    assert pc["q_cross"] == 5.0 * pc["q_xy_baseline"]
    assert abs(pc["anisotropy_cross_over_along"] - 25.0) < 1e-6
    assert m["eaob"]["active"] is True
    c.mode = "mpc"
    m2 = c.meta()
    assert m2["path_cost"] is None and m2["cost_frame"].startswith("world")
    assert m2["eaob"]["active"] is False


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except Exception as e:                                # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
