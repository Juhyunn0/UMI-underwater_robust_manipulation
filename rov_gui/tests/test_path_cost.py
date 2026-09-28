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


def test_plan_cost_scale_touches_the_position_block_only():
    """policy.q_scale (2026-09-07): HwDobMpc._scale_W multiplies the (x, y, z)
    block by s and nothing else; s = 1 is the identity; on the W the solver
    actually sees (diagonal Q with at most an xy rotation INSIDE the block —
    PathFrameWeights.stage_W) the result stays symmetric, positive definite,
    and composes with the w_stage mask. A dense W with position/other cross
    terms is NOT the contract (block-scaling is not a congruence there), and
    the solver never builds one."""
    from rov_gui.control.mpc_bridge import HwDobMpc, import_dobmpc
    from rov_gui.control.path_cost import PathFrameWeights
    # Through import_dobmpc like every other test here: a bare
    # ``bluerov2_mujoco_marinegym.dobmpc.params`` import fails in isolation
    # (params.py does ``import rov_model`` off sys.path) and, worse, would
    # pin the shared ``rov_model`` module to the env default before the
    # heavy_gripper physics the rest of this file relies on (2026-09-26).
    P = import_dobmpc("heavy_gripper")["P"]
    wt = PathFrameWeights(P.MPC_Q, P.MPC_R, P.MPC_QN,
                          {"along_scale": 0.25, "cross_scale": 4.0})
    for W in (wt.W_base, wt.stage_W(0.7, 0.1), wt.terminal_W(-2.0, 0.3)):
        W = np.asarray(W, float)
        n = W.shape[0]
        S = HwDobMpc._scale_W(W, 16.0)
        idx = list(HwDobMpc.PLAN_SCALE_ROWS)
        assert idx == [0, 1, 2]
        for i in range(n):
            for j in range(n):
                f = 16.0 if (i in idx and j in idx) else 1.0
                assert abs(S[i, j] - f * W[i, j]) < 1e-9, (i, j)
        assert np.array_equal(HwDobMpc._scale_W(W, 1.0), W)
        assert np.allclose(S, S.T)
        # the xy rotation lives inside the scaled block, so scaling is a
        # congruence there: eigenvalues of the block scale by exactly 16
        ev_w = np.sort(np.linalg.eigvalsh(W[np.ix_(idx, idx)]))
        ev_s = np.sort(np.linalg.eigvalsh(S[np.ix_(idx, idx)]))
        assert np.allclose(ev_s, 16.0 * ev_w)
        assert np.linalg.eigvalsh(S).min() > 0.0                     # still PD
        # mask after scale == scale after mask (both are block multiplications)
        M1 = HwDobMpc._mask_W(HwDobMpc._scale_W(W, 16.0), 0.0)
        M2 = HwDobMpc._scale_W(HwDobMpc._mask_W(W, 0.0), 16.0)
        assert np.allclose(M1, M2)
        assert np.linalg.eigvalsh(M1).min() >= -1e-9 * np.abs(M1).max()   # PSD survives
        assert HwDobMpc._scale_W(W, 16.0) is not W                       # a copy
        # nothing outside the position block moved, cross terms included
        mask = np.ones((n, n), bool); mask[np.ix_(idx, idx)] = False
        assert np.array_equal(S[mask], W[mask])


def test_plan_cost_scale_setter_validates_and_flags_a_rewrite():
    """set_plan_cost_scale refuses non-positive / non-finite scales and marks
    the solver weights dirty so the next tick rewrites them; the getter
    reports the live value. Exercised on a bare instance (no solver)."""
    from rov_gui.control.mpc_bridge import HwDobMpc
    obj = HwDobMpc.__new__(HwDobMpc)
    obj._plan_q_scale = 1.0
    obj._plan_q_scale_flown = 1.0
    obj._w_tuned = False
    obj.solver_kind = "acados"
    assert obj.plan_cost_scale == 1.0
    obj.set_plan_cost_scale(16)
    assert obj.plan_cost_scale == 16.0 and obj._w_tuned is True
    obj._w_tuned = False
    obj.set_plan_cost_scale(16.0)                                    # no change: not dirtied
    assert obj._w_tuned is False
    assert obj._plan_q_scale_flown == 16.0                            # what this engagement flew
    # the IPOPT fallback has no per-stage cost_set: a non-unit scale is refused
    # instead of being recorded as flown and silently ignored
    ip = HwDobMpc.__new__(HwDobMpc)
    ip._plan_q_scale = 1.0; ip._plan_q_scale_flown = 1.0; ip._w_tuned = False
    ip.solver_kind = "ipopt"
    ip.set_plan_cost_scale(1.0)                                       # unit is fine
    try:
        ip.set_plan_cost_scale(16.0)
    except ValueError as e:
        assert "acados" in str(e)
    else:
        raise AssertionError("ipopt accepted a cost scale it cannot apply")
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        try:
            obj.set_plan_cost_scale(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {bad!r}")


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
        _SKIPPED.append(f"acados: {type(e).__name__}: {e}")
        print(f"  SKIP (no acados / dobmpc: {type(e).__name__}: {e})")
        return None


def _plant(w_ext=None):
    """The controller's OWN prediction model as the plant. Honest about what
    that means: no deadband, no tether, 8-12x too little drag. ``w_ext`` is a
    constant NED-body wrench the plant feels and the controller is not told
    about (a buoyancy trim the model does not have, for instance)."""
    import casadi as ca
    from dobmpc.mpc import _f_casadi

    xs, us, ws = ca.SX.sym("x", 12), ca.SX.sym("u", 6), ca.SX.sym("w", 6)
    F = ca.Function("F", [xs, us, ws], [_f_casadi(xs, us, ws)])
    z = np.zeros(6) if w_ext is None else np.asarray(w_ext, float).copy()

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
_SKIPPED: list = []      # every solver-dependent test that returned early


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


# ------------------------------------------------------------- heave trim
# 2026-09-07: the solver's plant sinks at 5.71 N (heavy_gripper), the pool
# vehicle floated, and a nominal NMPC cancels its own model's gravity — so
# mpc/mpc_tuned pushed UP 5.7 N at zero depth error and the policy follower
# rose under a plan that pointed down (0907_180038). hw_mpc.yaml
# vehicle_net_buoyancy_n feeds the difference through the disturbance parameter.
def _dobmpc_or_skip():
    try:
        from rov_gui.control.mpc_bridge import HwDobMpc, import_dobmpc
        import_dobmpc("heavy_gripper")
        from dobmpc import fossen, params
        return HwDobMpc, fossen, params
    except Exception as e:                                    # noqa: BLE001
        _SKIPPED.append(f"dobmpc: {type(e).__name__}: {e}")
        print(f"  SKIP (no dobmpc: {type(e).__name__}: {e})")
        return None


def test_heave_trim_turns_the_models_gravity_into_the_real_one():
    """At rest the trimmed model must accelerate exactly as the real vehicle
    does: not at all when it is neutral, upward at B-W when it floats. Pure
    numpy on the same Fossen dynamics the solver was generated from, so a
    sign slip here is the sign slip the solver would fly."""
    d = _dobmpc_or_skip()
    if d is None:
        return
    HwDobMpc, fossen, P = d
    eta0, nu0, tau0 = np.zeros(6), np.zeros(6), np.zeros(6)
    a_model = fossen.nu_dot(eta0, nu0, tau0, np.zeros(6))[2]
    assert a_model > 0.1, "the untrimmed heavy_gripper plant must sink (NED +z)"
    a_neutral = fossen.nu_dot(eta0, nu0, tau0,
                              HwDobMpc.heave_trim_wrench(P.NET_BUOYANCY, 0.0))[2]
    assert abs(a_neutral) < 1e-12, a_neutral
    a_float = fossen.nu_dot(eta0, nu0, tau0,
                            HwDobMpc.heave_trim_wrench(P.NET_BUOYANCY, 2.0))[2]
    # same M^-1 row, force -2 N instead of +5.71 N
    assert abs(a_float / a_model - (-2.0 / -P.NET_BUOYANCY)) < 1e-9, \
        (a_float, a_model)
    assert a_float < 0.0, "a floating vehicle accelerates UP (NED -z)"


def test_heave_trim_reaches_the_solver_only_in_the_nominal_modes():
    """mpc / mpc_tuned hand the solver the trim; dobmpc hands it the EAOB's
    estimate alone — never both, or the residual is counted twice."""
    H = _acados_or_skip()
    if H is None:
        return
    c = _controller(H)
    cap = {}
    orig = c.nmpc.solve

    def spy(x, w, xref):
        cap["w"] = np.array(w, float).copy()
        return orig(x, w, xref)

    c.nmpc.solve = spy
    try:
        x0 = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        for mode in ("mpc", "mpc_tuned"):
            c.mode = mode
            c.reset()
            c.set_target_ned([0.0, 0.0, 0.5], 0.0)
            u, info = c.step(x0, np.zeros(6), np.zeros(6), 0.0)
            assert np.array_equal(cap["w"], c.w_trim), (mode, cap["w"])
            assert np.array_equal(info["w_solver"], c.w_trim)
            assert np.all(info["w_hat"] == 0.0), "w_hat stays the EAOB column"
            assert abs(u[2]) < 1e-6, \
                f"{mode}: zero error on the trimmed plant must ask for no heave, got {u[2]:+.3f} N"
            assert c.meta()["heave_trim"]["applied"] is True
        c.mode = "dobmpc"
        c.reset()
        c.set_target_ned([0.0, 0.0, 0.5], 0.0)
        _u, info = c.step(x0, np.zeros(6), np.zeros(6), 0.0)
        assert np.array_equal(cap["w"], info["w_hat"]), "dobmpc: EAOB only"
        assert np.array_equal(info["w_solver"], info["w_hat"])
        assert c.meta()["heave_trim"]["applied"] is False
        # reset() must not forget a config constant
        c.reset()
        assert c.w_trim[2] == c.heave_trim_wrench(c.P.NET_BUOYANCY, 0.0)[2]
    finally:
        c.nmpc.solve = orig
        c.mode = "mpc"
        c.reset()


def test_a_neutral_vehicle_holds_depth_only_with_the_trim():
    """The hardware failure on the model plant: a NEUTRAL vehicle (the plant
    plus the up-force that cancels its modelled sinking) under a station
    hold. Without the trim the nominal NMPC parks it ABOVE the setpoint by
    about 5.71 N / 60 N/m; with it the offset vanishes. No deadband here, so
    the real vehicle does worse than the untrimmed number, not better."""
    H = _acados_or_skip()
    if H is None:
        return
    c = _controller(H)
    neutral = H.heave_trim_wrench(c.P.NET_BUOYANCY, 0.0)   # what the pool did
    rk4 = _plant(w_ext=neutral)

    def hold(trim_on: bool, T=20.0):
        c.mode = "mpc"
        c.reset()
        c.set_target_ned([0.0, 0.0, 0.5], 0.0)
        saved = c.w_trim.copy()
        if not trim_on:
            c.w_trim = np.zeros(6)
        try:
            x = np.zeros(12)
            x[2] = 0.5
            zs = []
            for i in range(int(T / 0.05)):
                u, _info = c.step(x[0:6], x[6:12], np.zeros(6), i * 0.05)
                x = rk4(x, np.asarray(u, float))
                zs.append(x[2])
        finally:
            c.w_trim = saved
        return np.array(zs)

    z_off = hold(False)
    z_on = hold(True)
    high_off = 0.5 - z_off[-100:].mean()      # + = parked ABOVE the setpoint
    high_on = 0.5 - z_on[-100:].mean()
    print(f"    neutral vehicle, station hold: untrimmed parks {high_off * 100:+.1f} cm "
          f"high, trimmed {high_on * 100:+.2f} cm")
    assert high_off > 0.05, high_off
    assert abs(high_on) < 0.005, high_on
    assert abs(z_on[-100:].std()) < 1e-3, "trimmed hold must be still"


def test_net_buoyancy_is_validated_and_recorded():
    import os
    import tempfile

    import yaml

    from rov_gui.control.geometry import MpcConfig

    raw = yaml.safe_load(open(ROOT / "config" / "hw_mpc.yaml"))
    land = yaml.safe_load(open(ROOT / "config" / "land_dp.yaml"))
    v = float(raw["vehicle_net_buoyancy_n"])
    # measured 2026-09-08: +0.1 N (0908_105822/buoyancy_fit_20260908.txt);
    # both YAMLs carry the same vehicle constant and it stays a small number
    assert abs(v) <= 2.0 and float(land["vehicle_net_buoyancy_n"]) == v, (v, land["vehicle_net_buoyancy_n"])
    for bad in (16.0, -16.0, float("nan")):
        raw["vehicle_net_buoyancy_n"] = bad
        fd, tmp = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        with open(tmp, "w") as f:
            yaml.safe_dump(raw, f)
        try:
            try:
                MpcConfig.load(tmp)
                raise AssertionError(f"{bad!r} accepted")
            except ValueError:
                pass
        finally:
            os.unlink(tmp)
    d = _dobmpc_or_skip()
    if d is None:
        return
    HwDobMpc, _f, P = d
    # a hand-built config must not bypass the loader's bound
    from rov_gui.control.geometry import check_vehicle_net_buoyancy
    for bad in (16.0, float("inf")):
        try:
            check_vehicle_net_buoyancy(bad)
            raise AssertionError(f"{bad!r} accepted by the bridge-side check")
        except ValueError:
            pass
    w = HwDobMpc.heave_trim_wrench(P.NET_BUOYANCY, MpcConfig(vehicle_net_buoyancy_n=2.0).vehicle_net_buoyancy_n)
    assert abs(w[2] - (P.NET_BUOYANCY - 2.0)) < 1e-12
    assert np.all(w[[0, 1, 3, 4, 5]] == 0.0)
    H = _acados_or_skip()
    if H is None:
        return
    m = _controller(H).meta()["heave_trim"]
    assert m["model_net_buoyancy_n"] == float(P.NET_BUOYANCY)
    assert m["vehicle_net_buoyancy_n"] == 0.0
    assert m["w_trim_ned_body"][2] == float(P.NET_BUOYANCY)
    # a hand-built config out of range is refused at construction too
    try:
        H("mpc", MpcConfig(vehicle_net_buoyancy_n=16.0), log=lambda m: None)
        raise AssertionError("HwDobMpc accepted vehicle_net_buoyancy_n 16")
    except ValueError:
        pass


def test_mpcc_records_that_it_ignores_the_trim():
    """A session trim set in hw_mpc.yaml must not LOOK applied on a run
    flown under mpcc: its meta says so, in the same key the NMPC uses."""
    try:
        from rov_gui.control.mpcc_bridge import HwMpcc
        from rov_gui.control.geometry import MpcConfig
        c = HwMpcc(MpcConfig(vehicle_net_buoyancy_n=2.0), "mpcc", log=lambda m: None)
    except Exception as e:                                    # noqa: BLE001
        _SKIPPED.append(f"mpcc: {type(e).__name__}: {e}")
        print(f"  SKIP (no mpcc solver: {type(e).__name__}: {e})")
        return
    m = c.meta()["heave_trim"]
    assert m["applied"] is False and m["vehicle_net_buoyancy_n"] == 2.0
    assert "ignores" in m["reason"]


# ----------------------------------------------------------- real-time probe
def test_probe_is_a_warm_median_and_reprobe_refreshes_it():
    """The probe is the median of PROBE_N solves after one warm-up solve, the
    cold first solve is kept separately, and reprobe() measures again and
    returns the fresh verdict — so a one-off slow solve at construction
    cannot lock the session out."""
    H = _acados_or_skip()
    if H is None:
        return
    c = _controller(H)
    assert c.probe_ms is not None and c.probe_ms_first is not None
    assert c.probe_ms_worst is not None and c.probe_ms_worst >= c.probe_ms
    assert H.PROBE_N >= 3
    runs_before = c.probe_n
    ok = c.reprobe()
    assert ok is True and c.realtime_ok is True
    assert c.probe_n == runs_before + 1
    assert c.probe_ms < float(c.cfg.engage.get("probe_ms_max", 25.0))
    m = c.meta()
    assert m["probe_n_solves"] == H.PROBE_N and m["probe_runs"] == c.probe_n
    assert m["probe_ms_first"] is not None
    # a tighter limit than any real solve flips the verdict on re-probe
    old = c.cfg.engage["probe_ms_max"]
    try:
        c.cfg.engage["probe_ms_max"] = 1e-6
        assert c.reprobe() is False and c.realtime_ok is False
    finally:
        c.cfg.engage["probe_ms_max"] = old
        assert c.reprobe() is True


# ------------------------------------------------------- prediction-model drag
def test_the_shipped_plant_block_matches_the_measured_whole_loop_law():
    """``plant.linear_damping`` is not a free knob: it is axis_gain inverted
    through the 2026-08-17 whole-loop fit, so it goes stale the moment
    axis_gain moves. This test is the alarm for that."""
    import yaml

    raw = yaml.safe_load((ROOT / "config" / "hw_mpc.yaml").read_text())
    plant = raw.get("plant") or {}
    assert "linear_damping" in plant, \
        "the shipped config must carry the measured drag, or the solver flies " \
        "marinegym's 4.03 N.s/m and the vehicle sits still (0908_133049)"
    dl = plant["linear_damping"]
    assert len(dl) == 6, dl
    surge_want = float(raw["axis_gain"]["surge_n"]) / 0.692
    assert abs(dl[0] - surge_want) < 0.5, (
        f"plant.linear_damping[0] {dl[0]} != axis_gain.surge_n / 0.692 "
        f"= {surge_want:.1f}: the drag was derived from the axis gain, so one "
        f"cannot move without the other (hw_mpc.yaml)")
    # sway carries the SAME multiplier — the law cannot separate the axes, so
    # the model's own anisotropy is what is preserved
    assert abs(dl[1] / dl[0] - 6.22 / 4.03) < 0.02, (dl[0], dl[1])
    # heave and the angular axes have no equivalent measurement: unchanged.
    # yaw was tried at 4.0 on 2026-09-14 (station yaw-step fit) and REVERTED
    # the same night: overshoot doubled (0914_202238/mpc_202348, mpc_202502
    # vs 0914_190226/mpc_190335, mpc_190532) because the NMPC dropped its
    # rate feedback by the damping it was told the water provides. Pinned so
    # the sim placeholder is not silently "corrected" again without an A/B.
    assert dl[2:] == [5.18, 0.07, 0.07, 0.07], dl[2:]
    # the quadratic term is deliberately left alone (0.7 N at 0.2 m/s)
    assert "quadratic_damping" not in plant, \
        "raising the quadratic term extrapolates past the fit; if it is ever " \
        "set, this test and the hw_mpc.yaml note must be rewritten together"


def test_a_typo_in_the_plant_block_raises_instead_of_flying_the_sim_drag():
    from rov_gui.control.mpc_bridge import resolve_plant

    assert resolve_plant({}) == {} and resolve_plant(None) == {}
    ok = resolve_plant({"linear_damping": [86.7, 133.8, 5.18, 0.07, 0.07, 0.07]})
    assert ok["linear_damping"][0] == 86.7
    for bad in ({"linear_dampng": [1] * 6},        # the typo
                {"linear_damping": [1, 2, 3]},      # wrong length
                {"linear_damping": [-1] + [1] * 5},  # anti-damped
                {"linear_damping": [float("nan")] + [1] * 5}):
        try:
            resolve_plant(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r} — a silently ignored damping "
                             f"override flies the sim plant under a meta that "
                             f"says otherwise")


def test_the_acados_rebuild_key_carries_the_damping():
    """The codegen cache is keyed on (model, rti). Damping lives INSIDE the
    generated dynamics, so it must ride in the sidecar or a cached .so keeps
    the old drag while the meta reports the new one."""
    import json as _json
    import shutil
    import types

    from rov_gui.control import mpc_bridge as MB

    P = types.SimpleNamespace(
        MODEL="pytest_dragkey", MPC_N=60, DT_CTRL=0.05, NU=6,
        U_MAX=np.array([30.0] * 3 + [8.0, 8.0, 10.0]),
        DL=-np.array([86.7, 133.8, 5.18, 0.07, 0.07, 0.07]),
        DNL=-np.array([18.18, 21.66, 36.99, 1.55, 1.55, 1.55]))
    gen = MB._MARINEGYM / "dobmpc" / "_acados_gen" / f"{P.MODEL}_rti"
    try:
        MB._guard_stale_solver(P)
        want = _json.loads((gen / "build_meta_rovgui.json").read_text())
        assert "dl" in want and "dnl" in want, sorted(want)
        assert abs(abs(want["dl"][0]) - 86.7) < 1e-9, want["dl"]
        assert abs(abs(want["dnl"][0]) - 18.18) < 1e-9, want["dnl"]
    finally:
        shutil.rmtree(gen, ignore_errors=True)


def test_plant_overrides_change_the_dynamics_the_solver_is_generated_from():
    """Same argument as the heave-trim test above: the solver is generated
    FROM these params, so a change that shows up in ``fossen`` is the change
    the compiled OCP flies. Restores the module afterwards — every other
    closed-loop test in this file shares one build."""
    d = _dobmpc_or_skip()
    if d is None:
        return
    _HwDobMpc, fossen, P = d
    from rov_gui.control.mpc_bridge import apply_plant_overrides

    dl0, dnl0 = np.array(P.DL, copy=True), np.array(P.DNL, copy=True)
    try:
        v = 0.05
        nu = np.zeros(6)
        nu[0] = v
        drag_before = abs(fossen.damping(nu)[0])
        applied = apply_plant_overrides(
            P, {"linear_damping": [86.7, 133.8, 5.18, 0.07, 0.07, 0.07]})
        drag_after = abs(fossen.damping(nu)[0])
        assert P.DL[0] < 0, "params.py stores damping NEGATED; a positive DL " \
                            "is an anti-damped model"
        assert abs(applied["linear_damping"]["ratio"][0] - 21.5) < 0.1, applied
        assert abs(drag_before - 0.25) < 0.05, drag_before
        assert abs(drag_after - 4.38) < 0.1, drag_after
        # what the change is FOR: the feedforward at policy speed clears a
        # useful fraction of the T200 deadband instead of 4 % of it
        assert drag_after / 60.0 > 0.07, drag_after
    finally:
        P.DL, P.DNL = dl0, dnl0


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
    if _SKIPPED:
        # A skipped closed-loop test is a VACUOUS pass: the heave-trim sign
        # tests only mean something with the solver present. Say so, and
        # fail when the caller demands the real thing.
        import os
        print(f"  {len(_SKIPPED)} solver-dependent test(s) SKIPPED — those passes are vacuous "
              f"(set PATH_COST_REQUIRE_SOLVER=1 to fail instead)")
        if os.environ.get("PATH_COST_REQUIRE_SOLVER") == "1":
            return 2
    return 1 if failed else 0


def test_plan_along_scale_swaps_the_split_and_restores_the_config_tune():
    """policy.along_scale (2026-09-08): on a TUNED bridge, set_plan_along_scale(s)
    re-tunes the along/cross split with along_scale = s (q_along 75 -> 300 at
    s = 1.0 on the 300 baseline) leaving q_cross alone, marks the weights
    dirty, records the value flown; None puts the config's own tune back;
    reset() forgets it. A non-tuned bridge records the value but keeps its
    isotropic W (nothing to split). Bare instances, no solver."""
    from rov_gui.control.mpc_bridge import HwDobMpc
    from rov_gui.control.path_cost import PathFrameWeights
    from bluerov2_mujoco_marinegym.dobmpc import params as P

    class _NoSolver:                     # reset() calls nmpc.reset(); no cost_set here
        N = 60

        def reset(self):
            pass

    def bare(tuned):
        obj = HwDobMpc.__new__(HwDobMpc)
        obj.P = P
        obj.nmpc = _NoSolver()
        obj.tuned = tuned
        obj.solver_kind = "acados"
        obj._w_tuned = False
        obj._plan_q_scale = 1.0
        obj._plan_q_scale_flown = 1.0
        obj._plan_along_scale = None
        obj._plan_along_scale_flown = None
        obj._wt_cfg_tune = ({"along_scale": 0.25, "cross_scale": 4.0}
                            if tuned else None)
        obj._wt = PathFrameWeights(P.MPC_Q, P.MPC_R, P.MPC_QN, obj._wt_cfg_tune)
        return obj

    obj = bare(True)
    q_along0, q_cross0 = obj._wt.q_along, obj._wt.q_cross
    assert abs(q_along0 - 0.25 * obj._wt.q_xy) < 1e-9
    obj.set_plan_along_scale(1.0)
    assert obj.plan_along_scale == 1.0 and obj._plan_along_scale_flown == 1.0
    assert abs(obj._wt.q_along - obj._wt.q_xy) < 1e-9          # 75 -> 300
    assert abs(obj._wt.q_cross - q_cross0) < 1e-9              # cross untouched
    assert obj._w_tuned is True                                 # rewrite pending
    obj._w_tuned = False
    obj.set_plan_along_scale(1.0)                               # no change: not dirtied
    assert obj._w_tuned is False
    obj.set_plan_along_scale(None)                              # STOP: config tune back
    assert obj.plan_along_scale is None
    assert abs(obj._wt.q_along - q_along0) < 1e-9
    assert obj._plan_along_scale_flown == 1.0                   # what this engagement flew
    obj.set_plan_along_scale(2.0)
    obj.reset()                                                 # a fresh engagement forgets it
    assert obj.plan_along_scale is None and obj._plan_along_scale_flown is None
    assert abs(obj._wt.q_along - q_along0) < 1e-9
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        try:
            obj.set_plan_along_scale(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"along scale {bad!r} accepted")
    # IPOPT fallback: a tuned bridge cannot apply it -> refused, not recorded
    ip = bare(True)
    ip.solver_kind = "ipopt"
    try:
        ip.set_plan_along_scale(1.0)
    except ValueError as e:
        assert "acados" in str(e)
    else:
        raise AssertionError("ipopt accepted an along scale it cannot apply")
    # the isotropic baseline: recorded, W unchanged
    iso = bare(False)
    W0 = np.array(iso._wt.W_base, float)
    iso.set_plan_along_scale(1.0)
    assert iso.plan_along_scale == 1.0 and iso._plan_along_scale_flown == 1.0
    assert np.array_equal(np.asarray(iso._wt.W_base, float), W0)
    assert iso._w_tuned is False


if __name__ == "__main__":
    raise SystemExit(main())
