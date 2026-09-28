#!/usr/bin/env python3
"""test_attitude_axes.py — the 6-DoF variant's OUTPUT side (2026-09-26).

Group "output" of the pos_rpy_width design: allocation (K/M -> the
MANUAL_CONTROL extension axes), the NMPC bridge's attitude reference
(xref rows 3:5 / 9:11 through T^-1, the D5 hold ramp), config gating
(geometry), the command sink's wire frames, the null sink's slot parity, the
station bridge's coast tier, the demo plant and the bench probe.

THE PROMISE every test here pins: with ``engage.attitude_axes.enabled: false``
(the default) every control output, xref and wire frame is BYTE-IDENTICAL to
the 4-DoF station. No acados, no hardware, no torch: ``HwDobMpc`` is built
without ``__init__`` (the test_control.py pattern) and the sink talks to a
fake ``master``.

    cd <repo> && QT_QPA_PLATFORM=offscreen python -m pytest \\
        rov_gui/tests/test_attitude_axes.py -q -p no:cacheprovider
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rov_gui.control.allocation import (axes_to_wrench, slew_axes,  # noqa: E402
                                        wrench_to_axes)
from rov_gui.control.state_assembler import rot_zyx  # noqa: E402
from rov_gui.state import PilotInput  # noqa: E402

GAINS = {"surge_n": 60.0, "sway_n": 60.0, "heave_n": 60.0, "yaw_nm": 20.0,
         "roll_nm": 13.2, "pitch_nm": 7.2}


# =============================================================================
# allocation — K/M on and off, 6-tuple slew, inverse map
# =============================================================================
def test_allocation_off_is_the_pre_variant_object():
    """attitude=False: roll = pitch = 0.0 whatever K/M say, and the returned
    PilotInput equals one built the pre-variant way (the same fields, the
    dataclass defaults for the two new ones)."""
    u = [30.0, -15.0, 30.0, 5.0, 5.0, 10.0]
    cmd = wrench_to_axes(u, GAINS, cap=0.5, stamp=1.0)
    assert cmd.roll == 0.0 and cmd.pitch == 0.0
    legacy = PilotInput(surge=0.5, sway=-0.25, heave=-0.5, yaw=0.5,
                        active=frozenset(("mpc",)), source="mpc", stamp=1.0)
    assert cmd == legacy
    # the explicit-flag rule: gains present + non-zero K/M is NOT enough
    cmd2 = wrench_to_axes([0, 0, 0, 8.0, -8.0, 0], GAINS, stamp=1.0)
    assert cmd2.roll == 0.0 == cmd2.pitch
    back = axes_to_wrench(cmd2, GAINS)
    assert back[3] == 0.0 and back[4] == 0.0 and len(back) == 6


def test_allocation_on_maps_k_m_with_their_own_cap_and_round_trips():
    u = [0.0, 0.0, 0.0, 1.32, -0.72, 0.0]          # 0.1 of each gain
    cmd = wrench_to_axes(u, GAINS, cap=0.5, stamp=1.0, attitude=True)
    assert abs(cmd.roll - 0.1) < 1e-12 and abs(cmd.pitch + 0.1) < 1e-12
    assert cmd.surge == cmd.sway == cmd.heave == cmd.yaw == 0.0
    back = axes_to_wrench(cmd, GAINS)
    assert abs(back[3] - 1.32) < 1e-12 and abs(back[4] + 0.72) < 1e-12
    # the attitude cap is separate from `cap`: +8 N*m roll -> 0.2, pitch -> 0.3
    big = wrench_to_axes([0, 0, 0, 8.0, 8.0, 0], GAINS, cap=0.5, stamp=1.0,
                         attitude=True, attitude_cap=(0.2, 0.3))
    assert big.roll == 0.2 and big.pitch == 0.3
    assert axes_to_wrench(big, GAINS)[3] == pytest.approx(0.2 * 13.2)
    # signs: +K (starboard-down) -> +roll, +M (nose-up) -> +pitch
    assert wrench_to_axes([0, 0, 0, 1.0, 1.0, 0], GAINS, stamp=1.0,
                          attitude=True).roll > 0
    assert wrench_to_axes([0, 0, 0, 1.0, 1.0, 0], GAINS, stamp=1.0,
                          attitude=True).pitch > 0
    # missing roll_nm / pitch_nm fall back to the [유도] defaults
    g4 = {k: v for k, v in GAINS.items() if k not in ("roll_nm", "pitch_nm")}
    assert wrench_to_axes(u, g4, stamp=1.0, attitude=True).roll == pytest.approx(0.1)


def test_slew_keeps_roll_pitch_with_a_4_tuple_and_slews_with_a_6_tuple():
    cmd = PilotInput(surge=1.0, sway=0.0, heave=0.0, yaw=0.0, roll=0.2, pitch=-0.3,
                     active=frozenset(("mpc",)), source="mpc", stamp=1.0)
    out4 = slew_axes(cmd, (0.0, 0.0, 0.0, 0.0), 1.5, 0.05)
    assert out4.surge == pytest.approx(0.075)
    assert out4.roll == 0.2 and out4.pitch == -0.3, "never zeroed, never slewed with a 4-tuple"
    out6 = slew_axes(cmd, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 1.5, 0.05)
    assert out6.roll == pytest.approx(0.075) and out6.pitch == pytest.approx(-0.075)
    out6b = slew_axes(cmd, (0.0,) * 6, 1.5, 0.05, rate_rp=4.0)
    assert out6b.roll == pytest.approx(0.2) and out6b.pitch == pytest.approx(-0.2)
    # a 4-DoF command through the 6-tuple path is the 4-tuple result
    zero = PilotInput(surge=1.0, active=frozenset(("mpc",)), source="mpc", stamp=1.0)
    assert slew_axes(zero, (0.0,) * 6, 1.5, 0.05) == slew_axes(zero, (0.0,) * 4, 1.5, 0.05)
    # pass-through cases unchanged
    assert slew_axes(cmd, None, 1.5, 0.05) is cmd


# =============================================================================
# HwDobMpc — xref rows with / without rp, T^-1, the hold ramp
# =============================================================================
def _frames():
    """dobmpc.frames when importable (numpy-only), else a faithful stand-in."""
    try:
        # Through import_dobmpc, never a bare ``from dobmpc import frames``:
        # dobmpc/params.py imports the shared ``rov_model`` module, which
        # reads $ROV_MODEL ONCE — a bare import here pinned it to the env
        # default ("heavy") and every later heavy_gripper test in the same
        # process (test_path_cost heave trim) silently flew heavy physics
        # under a heavy_gripper label (full-suite run, 2026-09-26).
        from rov_gui.control.mpc_bridge import import_dobmpc
        return import_dobmpc(MpcConfig().rov_model)["frames"]
    except Exception:                                            # noqa: BLE001
        S = np.diag([1.0, -1.0, -1.0])

        def euler(R):
            theta = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
            phi = np.arctan2(R[2, 1], R[2, 2])
            psi = np.arctan2(R[1, 0], R[0, 0])
            return np.array([phi, theta, psi])

        def flu_to_ned_eta(p_flu, R):
            return np.concatenate([S @ np.asarray(p_flu, float),
                                   euler(S @ np.asarray(R, float) @ S)])
        return types.SimpleNamespace(S=S, flu_to_ned_eta=flu_to_ned_eta)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _ctrl(N: int = 10, dt: float = 0.05, pq_max: float = 0.35):
    from rov_gui.control.mpc_bridge import HwDobMpc

    c = HwDobMpc.__new__(HwDobMpc)
    c.nmpc = types.SimpleNamespace(N=N)
    c.P = types.SimpleNamespace(DT_CTRL=dt)
    c.frames = _frames()
    c.wrap_angle = _wrap
    c._psi_ned_now = 0.3
    c.p_ref = np.array([1.0, -0.5, 0.2])
    c.yaw_ref = -0.3
    c.v_ref = np.array([0.02, 0.0, 0.0])
    c.r_ref = 0.0
    c.yaw_target = -0.3
    c._ref_traj = None
    c._path_plan = None
    c._rp_hold = None
    c._pq_max = pq_max
    c._rp_reject_rad = math.radians(30.0)
    c._rp_max_rad = math.radians(20.0)
    c._rp_tracked_any = False
    c._attitude_axes = False
    c._u_max_wire_nm = None
    c._heave_trim_rotated = False
    return c


def _xref_ned_hold_legacy(self):
    """The pre-variant `_xref_ned` hold branch, verbatim (mpc_bridge.py
    2026-09-25) — the oracle the never-tracked path must match bitwise."""
    frames, P, wrap_angle = self.frames, self.P, self.wrap_angle
    N = self.nmpc.N
    dt = P.DT_CTRL
    c, s = np.cos(self.yaw_ref), np.sin(self.yaw_ref)
    R_ref = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    eta0 = frames.flu_to_ned_eta(self.p_ref, R_ref)
    eta0[5] = self._psi_ned_now + wrap_angle(eta0[5] - self._psi_ned_now)
    nu_ned = np.concatenate([frames.S @ (R_ref.T @ self.v_ref), np.zeros(3)])
    ks = np.arange(N + 1)
    pos_world = self.p_ref[:, None] + np.outer(self.v_ref, ks * dt)
    xref = np.zeros((12, N + 1))
    xref[0:3, :] = frames.S @ pos_world
    xref[3:6, :] = eta0[3:6][:, None]
    xref[6:12, :] = nu_ned[:, None]
    return xref


def _xref_ned_plan_legacy(self, plan):
    """The pre-variant `_xref_ned_plan`, verbatim."""
    N = self.nmpc.N
    p = np.asarray(plan.p_ned, float)
    yaw = np.asarray(plan.yaw_ned, float).ravel()
    v = np.asarray(plan.v_ned, float)
    r = np.asarray(plan.r_ned, float).ravel()
    xref = np.zeros((12, N + 1))
    xref[0:3, :] = p
    psi_prev = self._psi_ned_now
    for k in range(N + 1):
        psi_prev += self.wrap_angle(float(yaw[k]) - psi_prev)
        xref[5, k] = psi_prev
        Rk = rot_zyx(0.0, 0.0, psi_prev)
        xref[6:9, k] = Rk.T @ v[:, k]
    xref[11, :] = r
    return xref


def _plan(N: int, rp=None, rpr=None):
    from rov_gui.control.path_geometry import NedPlan

    K = N + 1
    ts = np.arange(K) * 0.05
    p = np.vstack([0.5 + 0.03 * ts, -0.2 + 0.01 * ts, 0.4 * np.ones(K)])
    v = np.vstack([0.03 * np.ones(K), 0.01 * np.ones(K), np.zeros(K)])
    yaw = 0.3 + 0.1 * ts
    r = 0.1 * np.ones(K)
    kw = {}
    if rp is not None:
        kw["rp_ned"] = np.asarray(rp, float)
    if rpr is not None:
        kw["rp_rate_ned"] = np.asarray(rpr, float)
    try:
        return NedPlan(p, yaw, v, r, psi_path=yaw.copy(), **kw)
    except TypeError:
        # the worker group's NedPlan (rp_ned / rp_rate_ned) not merged yet:
        # a look-alike with the same field names is what the bridge reads
        base = NedPlan(p, yaw, v, r, psi_path=yaw.copy())
        ns = types.SimpleNamespace(**{f.name: getattr(base, f.name)
                                      for f in dataclasses.fields(base)})
        ns.rp_ned = kw.get("rp_ned")
        ns.rp_rate_ned = kw.get("rp_rate_ned")
        return ns


def test_xref_plan_without_rp_is_bitwise_the_legacy_tile():
    c = _ctrl(N=10)
    plan = _plan(10)
    assert getattr(plan, "rp_ned", None) is None
    got = c._xref_ned_plan(plan)
    want = _xref_ned_plan_legacy(c, plan)
    assert np.array_equal(got, want), "the 4-DoF plan path must be byte-identical"
    assert not np.any(got[3:5]) and not np.any(got[9:11])
    c.set_path_plan_ned(plan)                 # no rp: nothing to validate, no hold
    assert c._rp_hold is None and c.ref_attitude_ned_at(0.0) == (0.0, 0.0)
    assert c.attitude_ref_source() == "level"


def _T(phi, theta):
    """The kinematic T(eta) of dobmpc/mpc.py, in numpy."""
    sph, cph = math.sin(phi), math.cos(phi)
    sth, cth = math.sin(theta), math.cos(theta)
    return np.array([[1.0, sph * sth / cth, cph * sth / cth],
                     [0.0, cph, -sph],
                     [0.0, sph / cth, cph / cth]])


def test_xref_plan_with_rp_fills_attitude_and_body_rate_rows_through_T_inverse():
    from rov_gui.control.mpc_bridge import HwDobMpc

    c = _ctrl(N=10)
    K = 11
    rp = np.vstack([np.linspace(0.0, 0.15, K), np.linspace(0.1, -0.1, K)])
    rpr = np.vstack([np.full(K, 0.15 / 0.5), np.full(K, -0.2 / 0.5)])
    plan = _plan(10, rp=rp, rpr=rpr)
    c.set_path_plan_ned(plan)
    x = c._xref_ned_plan(plan)
    assert np.array_equal(x[3:5], rp)
    # rows 0:3, 5 are the legacy ones; rows 6:9 use the full R
    legacy = _xref_ned_plan_legacy(c, plan)
    assert np.array_equal(x[0:3], legacy[0:3]) and np.array_equal(x[5], legacy[5])
    for k in range(K):
        phi, theta, psi = x[3, k], x[4, k], x[5, k]
        eul_dot = np.array([rpr[0, k], rpr[1, k], plan.r_ned[k]])
        pqr = np.linalg.solve(_T(phi, theta), eul_dot)
        assert np.allclose(x[9:12, k], pqr, atol=1e-12), k
        assert np.allclose(x[6:9, k], rot_zyx(phi, theta, psi).T @ plan.v_ned[:, k], atol=1e-12)
    # the static helper at level is the identity map
    assert HwDobMpc.body_rates_from_euler(0.0, 0.0, 0.3, -0.2, 0.1) == pytest.approx((0.3, -0.2, 0.1))
    # rp present but ZERO with zero rates: rows 3:5 / 9:11 zero and the rest
    # bitwise equal to the no-rp tile (the level case degenerates exactly)
    z = _plan(10, rp=np.zeros((2, K)), rpr=np.zeros((2, K)))
    xz = c._xref_ned_plan(z)
    assert np.array_equal(xz, _xref_ned_plan_legacy(c, z))
    # no rp_rate: Euler rates 0, r still through T^-1
    plan2 = _plan(10, rp=rp)
    x2 = c._xref_ned_plan(plan2)
    for k in range(K):
        pqr = np.linalg.solve(_T(x2[3, k], x2[4, k]), np.array([0.0, 0.0, plan.r_ned[k]]))
        assert np.allclose(x2[9:12, k], pqr, atol=1e-12)
    assert c.ref_attitude_ned_at(0.0) == (pytest.approx(rp[0, 0]), pytest.approx(rp[1, 0]))
    assert c.attitude_ref_source() == "plan"


def test_set_path_plan_ned_validates_rp_and_rejects_beyond_rp_reject():
    c = _ctrl(N=4)
    K = 5
    with pytest.raises(ValueError, match="rp_ned"):
        c.set_path_plan_ned(_plan(4, rp=np.zeros((2, K - 1))))
    bad = np.zeros((2, K)); bad[1, 2] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        c.set_path_plan_ned(_plan(4, rp=bad))
    big = np.zeros((2, K)); big[0, 3] = math.radians(31.0)
    with pytest.raises(ValueError, match="rp_reject"):
        c.set_path_plan_ned(_plan(4, rp=big))
    ok = np.zeros((2, K)); ok[0, 3] = math.radians(29.0)
    c.set_path_plan_ned(_plan(4, rp=ok))
    assert c._path_plan is not None and c._rp_tracked_any


def test_hold_ramp_starts_from_the_dropped_plan_and_decays_at_pq_max():
    c = _ctrl(N=4, dt=0.05, pq_max=0.35)
    K = 5
    rp = np.zeros((2, K)); rp[0, :] = 0.10; rp[1, :] = -0.05
    c.set_path_plan_ned(_plan(4, rp=rp))
    assert c._rp_hold is None                # the plan owns the attitude
    c.set_path_plan_ned(None)                # plan end -> ramp from stage 0
    assert np.allclose(c._rp_hold, [0.10, -0.05])
    assert c.attitude_ref_source() == "hold_ramp"
    assert c.ref_attitude_ned_at(0.0) == (pytest.approx(0.10), pytest.approx(-0.05))
    # the hold reference PREVIEWS the ramp along the horizon (2026-09-26
    # fix 2): stage 0 = the hold, then pq_max*dt = 0.0175 rad per stage to
    # 0 — what the per-tick decay below will produce, stage for tick
    x = c._xref_ned(0.0)
    assert np.allclose(x[3, :], [0.10, 0.0825, 0.065, 0.0475, 0.03])
    assert np.allclose(x[4, :], [-0.05, -0.0325, -0.015, 0.0, 0.0])
    legacy = _xref_ned_hold_legacy(c)
    keep = [i for i in range(12) if i not in (3, 4, 9, 10, 11)]
    assert np.array_equal(x[keep], legacy[keep])
    # decay: 0.35 * 0.05 = 0.0175 rad per tick; roll needs 6 ticks, pitch 3
    for _ in range(3):
        c._decay_rp_hold()
    assert c._rp_hold is not None and c._rp_hold[1] == 0.0
    assert c._rp_hold[0] == pytest.approx(0.10 - 3 * 0.0175)
    for _ in range(3):
        c._decay_rp_hold()
    assert c._rp_hold is None, "level -> None, the byte-identical path again"
    assert c.attitude_ref_source() == "level"
    assert np.array_equal(c._xref_ned(0.0), _xref_ned_hold_legacy(c))
    # a plan WITHOUT rp installed after a tracked plan starts the ramp too
    c.set_path_plan_ned(_plan(4, rp=rp))
    c.set_path_plan_ned(_plan(4))
    assert np.allclose(c._rp_hold, [0.10, -0.05])
    xp = c._xref_ned_plan(c._path_plan)
    assert np.allclose(xp[3, :], [0.10, 0.0825, 0.065, 0.0475, 0.03])
    assert np.allclose(xp[4, :], [-0.05, -0.0325, -0.015, 0.0, 0.0])
    # set_target_ned(rp_ref=...) seeds the ramp explicitly; None keeps the D5 capture
    c._rp_hold = None
    c.set_target_ned([0.0, 0.0, 0.5], 0.1, rp_ref=(0.05, 0.02))
    assert np.allclose(c._rp_hold, [0.05, 0.02])
    with pytest.raises(ValueError):
        c.set_target_ned([0.0, 0.0, 0.5], 0.1, rp_ref=(math.radians(40.0), 0.0))


def test_never_tracked_hold_xref_is_bitwise_legacy():
    c = _ctrl(N=12)
    for r_ref in (0.0, 0.2):
        c.r_ref = r_ref
        c.yaw_target = 0.4
        got = c._xref_ned(1.0)
        c2 = copy.copy(c)
        assert c._rp_hold is None
        # the legacy oracle has no yaw-ramp branch, so compare the rows it
        # covers when r_ref != 0 and everything when r_ref == 0
        want = _xref_ned_hold_legacy(c2)
        if r_ref == 0.0:
            assert np.array_equal(got, want)
        else:
            assert np.array_equal(np.delete(got, [5, 11], axis=0), np.delete(want, [5, 11], axis=0))
    # set_target with no rp_ref on a never-tracked controller keeps None
    c.set_target(p_ref=[0, 0, 0], yaw_ref=0.0)
    assert c._rp_hold is None
    assert c.ref_attitude_ned_at(0.0) == (0.0, 0.0)


def _traj_fn(ts):
    """A set_reference_traj sampler (world FLU): p (3,K), yaw (K,), v (3,K), r (K,)."""
    ts = np.asarray(ts, float)
    K = ts.size
    p = np.vstack([0.5 + 0.02 * ts, -0.1 * np.ones(K), 0.3 * np.ones(K)])
    v = np.vstack([0.02 * np.ones(K), np.zeros(K), np.zeros(K)])
    return p, -0.3 + 0.05 * ts, v, 0.05 * np.ones(K)


def _ramp_preview(hold, pq_max, dt, n):
    """The D5 ramp the per-tick decay produces, stage for tick."""
    hold = np.asarray(hold, float)
    ks = np.arange(n)
    mag = np.maximum(np.abs(hold)[:, None] - pq_max * dt * ks[None, :], 0.0)
    rp = np.sign(hold)[:, None] * mag
    rpd = np.where(mag > 0.0, -np.sign(hold)[:, None] * pq_max, 0.0)
    return rp, rpd


def _check_hold_preview(c, tile, hold, pq_max, dt):
    """(N4) rows 3:5 = the ramp preview; rows 9:12 = T(phi_k, theta_k)^-1
    (phid_k, thetad_k, psid_k) with psid_k what the SAME tile carried in
    row 11 without a hold; every other row bitwise the no-hold tile."""
    c._rp_hold = None
    base = tile()
    assert not np.any(base[9:11]), "rows 9:11 are exactly 0 with no hold"
    c._rp_hold = np.asarray(hold, float).copy()
    x = tile()
    n = x.shape[1]
    rp, rpd = _ramp_preview(hold, pq_max, dt, n)
    assert np.array_equal(x[3:5], rp)
    for k in range(n):
        eul_dot = np.array([rpd[0, k], rpd[1, k], base[11, k]])
        pqr = np.linalg.solve(_T(x[3, k], x[4, k]), eul_dot)
        assert np.allclose(x[9:12, k], pqr, atol=1e-12), (k, x[9:12, k], pqr)
    keep = [i for i in range(12) if i not in (3, 4, 9, 10, 11)]
    assert np.array_equal(x[keep], base[keep])
    # once an axis reached 0 its rate is 0; while moving it is -sign*pq_max
    moving = np.abs(rp) > 0.0
    assert np.all(rpd[~moving] == 0.0) and np.all(np.abs(rpd[moving]) == pq_max)
    return x


def test_traj_tile_after_a_tracked_plan_ramps_instead_of_stepping():
    """(N3, fix 1) a tracked plan ends into set_reference_traj (square /
    line / circle): the traj tile's stage 0 is the captured hold, the
    horizon previews the decay, and ref_attitude_ned_at agrees with what
    the solver sees (the first cut wrote eta_k[3:5] = 0 there — a STEP to
    level while the CSV reported the ramp)."""
    c = _ctrl(N=6, dt=0.05, pq_max=0.35)
    K = 7
    rp = np.zeros((2, K)); rp[0, :] = 0.12; rp[1, :] = -0.04
    c.set_path_plan_ned(_plan(6, rp=rp))
    c.set_reference_traj(_traj_fn)
    assert c._path_plan is None and c._ref_traj is _traj_fn
    assert np.allclose(c._rp_hold, [0.12, -0.04])
    x = c._xref_ned(0.0)
    assert x[3, 0] == 0.12 and x[4, 0] == -0.04
    assert c.ref_attitude_ned_at(0.0) == (pytest.approx(0.12), pytest.approx(-0.04))
    assert c.attitude_ref_source() == "hold_ramp"
    want_r, want_p = _ramp_preview([0.12, -0.04], 0.35, 0.05, K)[0]
    assert np.allclose(x[3, :], want_r) and np.allclose(x[4, :], want_p)
    assert np.all(np.diff(np.abs(x[3, :])) < 0.0)           # strictly decaying roll
    assert x[4, 3] == 0.0 and np.all(x[4, 3:] == 0.0)        # pitch reached level at stage 3
    # the ramp advances per tick exactly one stage: after one decay the
    # new stage 0 is the old stage 1
    c._decay_rp_hold()
    x1 = c._xref_ned(0.05)
    assert np.allclose(x1[3:5, 0], x[3:5, 1])
    # the never-tracked traj tile is untouched: rows 3:5 zero, 9:11 zero
    c2 = _ctrl(N=6, dt=0.05)
    c2.set_reference_traj(_traj_fn)
    x0 = c2._xref_ned(0.0)
    assert c2._rp_hold is None and not np.any(x0[3:5]) and not np.any(x0[9:11])
    assert np.allclose(x0[11, :], -0.05)


def test_hold_preview_rate_rows_through_T_inverse_on_every_tile():
    """(N4, fix 2) under a running hold, all three reference tiles — hold,
    plan without rp, trajectory sampler — carry the per-stage Euler rates
    of the ramp through T^-1 in rows 9:12 (psid = what row 11 carried:
    the yaw-ramp r_ned, the plan's r, the sampler's -r_w, or 0), and rows
    9:11 are exactly 0 whenever _rp_hold is None."""
    pq_max, dt = 0.35, 0.05
    c = _ctrl(N=8, dt=dt, pq_max=pq_max)
    hold = [0.10, -0.06]
    # hold tile, r_ref = 0 (psid 0) and r_ref != 0 (psid = r_ned on the ramp)
    c.r_ref = 0.0
    x = _check_hold_preview(c, lambda: c._xref_ned(0.0), hold, pq_max, dt)
    # psid = 0 here, so r = -thetad * sin(phi): the T^-1 coupling into
    # row 11 is real (a level-rate overlay would have left it 0)
    thetad = -np.sign(hold[1]) * pq_max * (np.abs(x[4, :]) > 0)
    assert np.allclose(x[11, :], -thetad * np.sin(x[3, :]), atol=1e-12)
    assert np.any(x[11, :] != 0.0)
    c.r_ref = 0.2
    c.yaw_target = 0.4
    _check_hold_preview(c, lambda: c._xref_ned(0.0), hold, pq_max, dt)
    c.r_ref = 0.0
    # plan-without-rp tile (psid = plan.r_ned)
    plan = _plan(8)
    _check_hold_preview(c, lambda: c._xref_ned_plan(plan), hold, pq_max, dt)
    # trajectory tile (psid = -r_w)
    c.set_reference_traj(_traj_fn)
    _check_hold_preview(c, lambda: c._xref_ned(0.3), hold, pq_max, dt)
    # the preview helper itself: None with no hold, the closed form with one
    c._rp_hold = None
    assert c._rp_hold_preview(9) is None
    c._rp_hold = np.array([0.0, 0.02])                      # roll already level
    rp, rpd = c._rp_hold_preview(4)
    assert np.all(rp[0] == 0.0) and np.all(rpd[0] == 0.0)
    assert np.allclose(rp[1], [0.02, 0.0025, 0.0, 0.0]) and np.allclose(rpd[1], [-0.35, -0.35, 0.0, 0.0])
    # the per-tick decay is consistent with the preview, tick for stage
    c._rp_hold = np.array(hold)
    rp, _ = c._rp_hold_preview(8)
    for k in range(1, 8):
        c._decay_rp_hold()
        got = np.zeros(2) if c._rp_hold is None else c._rp_hold
        assert np.allclose(got, rp[:, k], atol=1e-15), k


def _meta_ctrl(rov_model_cfg="heavy_gripper", model_loaded="heavy_gripper"):
    """A _ctrl() with everything HwDobMpc.meta() reads, no solver."""
    c = _ctrl(N=10)
    c._mode, c.solver_kind, c.dob, c.tuned, c._w_tuned = "mpc", "acados", False, False, False
    c.cfg = types.SimpleNamespace(ctrl_hz=20.0, rov_model=rov_model_cfg)
    c.P = types.SimpleNamespace(
        DT_CTRL=0.05, MPC_N=10, U_MAX=np.ones(6), V_MAX=1.0, MPC_Q=np.ones(12),
        MPC_QN=np.ones(12), MPC_R=np.ones(6), NET_BUOYANCY=-5.71,
        MODEL=model_loaded, EAOB_TAU_DIST=1.0, EAOB_NIS_GATE=9.0,
        EAOB_GATE_ON=False, EAOB_SIG_POS=np.ones(3), EAOB_SIG_ANG=np.ones(3),
        EAOB_SIG_LVEL=np.ones(3), EAOB_SIG_AVEL=np.ones(3), EAOB_SIG_ACC=1.0,
        EAOB_SIG_AACC=1.0, EAOB_SIG_ALLOC=np.ones(6))
    c.nmpc = types.SimpleNamespace(N=10, _fallback_enabled=False, _fallback_off_reason="")
    c.w_clip, c.w_trim = np.ones(6), np.zeros(6)
    c.vehicle_net_buoyancy_n, c._wt, c._plan_q_scale = 0.0, None, 1.0
    c.probe_ms, c.build_s, c.sigma_applied, c.plant_applied, c.eaob = None, 0.0, {}, {}, None
    return c


def test_rate_source_label_is_the_stitcher_not_a_finite_difference():
    """(fix 3) the station's rows 9:11 come from the stitcher's
    piecewise-constant rp_rate (or the ramp's -sign*pq_max) through T^-1,
    never a finite difference: the label says so, in the method and in
    meta(); never-tracked = "none"; the sim's fd label never appears."""
    c = _meta_ctrl()
    assert c.attitude_rate_source() == "none"
    m = c.meta()
    assert m["attitude_ref"]["rate_source"] == "none"
    assert m["attitude_ref"]["tracked"] is False
    c.set_path_plan_ned(_plan(10, rp=np.zeros((2, 11))))
    assert c._rp_tracked_any
    assert c.attitude_rate_source() == "stitcher_rate_T_inv"
    m = c.meta()
    assert m["attitude_ref"]["rate_source"] == "stitcher_rate_T_inv"
    assert m["attitude_ref"]["tracked"] is True and m["attitude_ref"]["source"] == "plan"
    assert "fd_euler" not in str(m["attitude_ref"])
    # (fix 4) the requested vs loaded model is visible in the record
    assert m["rov_model"] == "heavy_gripper" and m["rov_model_requested"] == "heavy_gripper"
    assert m["rov_model_mismatch"] is False
    m2 = _meta_ctrl(rov_model_cfg="heavy", model_loaded="heavy_gripper").meta()
    assert m2["rov_model"] == "heavy_gripper" and m2["rov_model_requested"] == "heavy"
    assert m2["rov_model_mismatch"] is True


def test_import_dobmpc_and_plant_load_warn_on_a_model_mismatch_never_raise():
    """(fix 4) dobmpc already in memory under another ROV_MODEL: both
    loaders return the LOADED model with a RuntimeWarning (a raise would
    take down a station whose mpcc module imported dobmpc bare first),
    and expose requested vs loaded. Skipped when the sim tree / dobmpc
    is not importable in this process."""
    import os
    import warnings

    from rov_gui.control import mpc_bridge, plant
    from rov_gui.control.geometry import MpcConfig

    try:
        d = mpc_bridge.import_dobmpc(MpcConfig().rov_model)
    except Exception as e:                                     # noqa: BLE001
        pytest.skip(f"dobmpc not importable here: {e}")
    loaded = str(d["P"].MODEL)
    assert d["model"] == loaded and d["model_requested"] == MpcConfig().rov_model
    other = "heavy" if loaded != "heavy" else "heavy_gripper"
    # plant._load: the module is cached under `loaded`; asking for `other`
    # warns and hands back the loaded params (plant_meta says so too)
    with pytest.warns(RuntimeWarning, match="already imported"):
        P, _f = plant._load(other)
    assert str(P.MODEL) == loaded
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pm = plant.plant_meta(other)
    assert pm["rov_model_requested"] == other and pm["rov_model_loaded"] == loaded
    assert "warning" in pm
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plant._load(loaded)                                    # no mismatch: silent
    # import_dobmpc with the cache cleared (dobmpc.params still in
    # sys.modules): warns, returns the loaded model, pins the request
    saved = (mpc_bridge._dob, mpc_bridge._dob_model, os.environ.get("ROV_MODEL"))
    try:
        mpc_bridge._dob, mpc_bridge._dob_model = None, None
        with pytest.warns(RuntimeWarning, match="already imported"):
            d2 = mpc_bridge.import_dobmpc(other)
        assert d2["model"] == loaded and d2["model_requested"] == other
        assert d2["P"] is d["P"]
        # a repeat call under EITHER name is the same dict, no raise
        assert mpc_bridge.import_dobmpc(other) is d2
        assert mpc_bridge.import_dobmpc(loaded) is d2
        with pytest.raises(RuntimeError, match="already imported"):
            mpc_bridge.import_dobmpc("bluerov2")
    finally:
        mpc_bridge._dob, mpc_bridge._dob_model = saved[0], saved[1]
        if saved[2] is None:
            os.environ.pop("ROV_MODEL", None)
        else:
            os.environ["ROV_MODEL"] = saved[2]


def test_heave_trim_rotation_is_off_by_default_and_exact_when_on():
    c = _ctrl()
    c.w_trim = np.array([0.0, 0.0, -5.71, 0.0, 0.0, 0.0])
    eta = np.array([0, 0, 0, 0.2, -0.3, 0.5])
    assert c._w_trim_now(eta) is c.w_trim                 # constant, the same object
    c._heave_trim_rotated = True
    assert c._w_trim_now(eta) is c.w_trim                 # needs attitude axes too
    c._attitude_axes = True
    w = c._w_trim_now(eta)
    R = rot_zyx(0.2, -0.3, 0.5)
    assert np.allclose(w[:3], R.T @ np.array([0.0, 0.0, -5.71]), atol=1e-12)
    assert np.allclose(c._w_trim_now(np.zeros(6)), c.w_trim)


def test_meta_and_set_attitude_axes_fill_the_record_blocks():
    c = _ctrl()
    c.set_attitude_axes(True, u_max_wire_nm=[0.2 * 13.2, 0.3 * 7.2])
    assert c._attitude_axes and c._u_max_wire_nm == pytest.approx([2.64, 2.16])
    c.set_attitude_axes(False)
    assert not c._attitude_axes and c._u_max_wire_nm is None


# =============================================================================
# geometry — config gating
# =============================================================================
def test_axis_gain_unknown_key_raises_and_defaults_carry_roll_pitch():
    import yaml
    from rov_gui.control.geometry import MpcConfig

    assert MpcConfig().axis_gain["roll_nm"] == 13.2
    assert MpcConfig().axis_gain["pitch_nm"] == 7.2
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "bad.yaml"
        f.write_text(yaml.safe_dump({"axis_gain": {"rol_nm": 13.2}}))
        with pytest.raises(ValueError, match="unknown axis_gain keys"):
            MpcConfig.load(f)
        f.write_text(yaml.safe_dump({"axis_gain": {"roll_nm": 0.0}}))
        with pytest.raises(ValueError, match="axis_gain.roll_nm"):
            MpcConfig.load(f)


def test_validate_engage_attitude_axes():
    from rov_gui.control.geometry import (default_attitude_axes_block,
                                          parse_firmware_version,
                                          validate_engage_attitude_axes)

    e = {}
    aa = validate_engage_attitude_axes(e)
    assert aa == default_attitude_axes_block() and e["attitude_axes"] is aa
    assert aa["enabled"] is False and aa["transport"] == "manual_control_ext"
    with pytest.raises(ValueError, match="not implemented"):
        validate_engage_attitude_axes({"attitude_axes": {"transport": "rc_override"}})
    with pytest.raises(ValueError, match="transport"):
        validate_engage_attitude_axes({"attitude_axes": {"transport": "serial"}})
    with pytest.raises(ValueError, match="unknown engage.attitude_axes keys"):
        validate_engage_attitude_axes({"attitude_axes": {"enable": True}})
    with pytest.raises(ValueError, match="not an existing file"):
        validate_engage_attitude_axes({"attitude_axes": {"probe": "/nonexistent/x.json"}})
    with pytest.raises(ValueError, match="sign"):
        validate_engage_attitude_axes({"attitude_axes": {"sign": {"roll": 0.5, "pitch": 1}}})
    with pytest.raises(ValueError, match="cap_roll"):
        validate_engage_attitude_axes({"attitude_axes": {"cap_roll": 1.5}})
    with pytest.raises(ValueError, match="require_manual"):
        validate_engage_attitude_axes({"attitude_axes": {"enabled": True, "require_manual": False}})
    with pytest.raises(ValueError, match="min_firmware"):
        validate_engage_attitude_axes({"attitude_axes": {"min_firmware": "four"}})
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "probe.json"
        f.write_text("{}")
        aa = validate_engage_attitude_axes({"attitude_axes": {
            "enabled": True, "probe": str(f), "sign": {"roll": -1, "pitch": 1},
            "first_water_caps": [0.1, 0.15]}})
        assert aa["sign"] == {"roll": -1.0, "pitch": 1.0} and aa["probe"] == str(f)
    assert parse_firmware_version("4.1.2") == (4, 1, 2)
    assert parse_firmware_version("4.5.1-rc1") == (4, 5, 1)
    assert parse_firmware_version("") is None and parse_firmware_version(None) is None
    assert parse_firmware_version("4.1.2") >= parse_firmware_version("4.1.2")
    assert parse_firmware_version("4.1.1") < parse_firmware_version("4.1.2")


def test_stabilize_rp_axis_whitelisted_only_as_hold_and_only_when_spelled():
    from rov_gui.control.geometry import check_engage_stabilize

    e = {"stabilize": {"yaw_axis": "hold"}}
    check_engage_stabilize(e)
    assert e["stabilize"] == {"yaw_axis": "hold"}, "a pre-variant block resolves as before"
    e = {"stabilize": {"yaw_axis": "hold", "rp_axis": "HOLD"}}
    check_engage_stabilize(e)
    assert e["stabilize"] == {"yaw_axis": "hold", "rp_axis": "hold"}
    with pytest.raises(ValueError, match="rp_axis"):
        check_engage_stabilize({"stabilize": {"rp_axis": "angle"}})


def test_policy_attitude_keys_validate():
    from rov_gui.control.geometry import (POLICY_PROVENANCE, default_policy_block,
                                          validate_policy_block)
    from rov_gui.state import (ACTION_REPR_POS_RPY_WIDTH, ACTION_REPR_POS_YAW_WIDTH,
                               POLICY_ACTION_REPR)

    pc = default_policy_block()
    assert pc["action_repr"] == POLICY_ACTION_REPR == ACTION_REPR_POS_YAW_WIDTH
    assert pc["attitude_track"] is False and pc["rp_ref_filter"] is None
    for k in ("action_repr", "attitude_track", "rp_max_deg", "rp_reject_deg",
              "pq_max_rad_s", "rp_jump_max_deg", "anchor_leash_rp_deg",
              "div_max_rp_deg", "blend_rp_rate_max", "heave_trim_attitude_rotated",
              "rp_ref_filter", "attitude_q_scale"):
        assert k in pc and k in POLICY_PROVENANCE, k
    validate_policy_block(pc)
    pc["action_repr"] = ACTION_REPR_POS_RPY_WIDTH
    validate_policy_block(pc)
    for bad in ({"action_repr": "pose10d"}, {"rp_max_deg": 0.0},
                {"rp_max_deg": 35.0, "rp_reject_deg": 30.0}, {"pq_max_rad_s": -1},
                {"anchor_leash_rp_deg": -1}, {"rp_ref_filter": {"rate_deg_s": 1}},
                {"attitude_q_scale": 0.0}, {"rp_reject_deg": 70.0}):
        p2 = default_policy_block()
        p2.update(bad)
        with pytest.raises(ValueError):
            validate_policy_block(p2)
    p3 = default_policy_block()
    p3["rp_ref_filter"] = {"rate_deg_s": 3, "tau_s": 2, "guard_deg": 15}
    validate_policy_block(p3)
    assert p3["rp_ref_filter"] == {"guard_deg": 15.0, "rate_deg_s": 3.0, "tau_s": 2.0}
    # a pre-variant block (no new keys at all) still resolves to the defaults
    p4 = {k: v for k, v in default_policy_block().items()
          if k not in ("action_repr", "attitude_track", "rp_max_deg")}
    validate_policy_block(p4)
    assert p4["action_repr"] == POLICY_ACTION_REPR and p4["rp_max_deg"] == 20.0


def _strip_variant_keys(raw: dict) -> dict:
    """The shipped YAML as it was BEFORE the variant: the added keys removed."""
    r = copy.deepcopy(raw)
    for k in ("roll_nm", "pitch_nm"):
        r.get("axis_gain", {}).pop(k, None)
    r.get("engage", {}).pop("attitude_axes", None)
    (r.get("engage", {}).get("stabilize") or {}).pop("rp_axis", None)
    for k in ("action_repr", "attitude_track", "rp_max_deg", "rp_reject_deg",
              "pq_max_rad_s", "rp_jump_max_deg", "anchor_leash_rp_deg",
              "div_max_rp_deg", "blend_rp_rate_max", "heave_trim_attitude_rotated",
              "rp_ref_filter", "attitude_q_scale"):
        r.get("policy", {}).pop(k, None)
    r.pop("replay", None)
    return r


def _assert_subset_equal(before, after, path="cfg"):
    """Every key/value in `before` appears unchanged in `after` (recursive)."""
    if isinstance(before, dict):
        assert isinstance(after, dict), path
        for k, v in before.items():
            assert k in after, f"{path}.{k} vanished"
            _assert_subset_equal(v, after[k], f"{path}.{k}")
    elif isinstance(before, np.ndarray):
        assert np.array_equal(before, after), path
    elif isinstance(before, float) and math.isnan(before):
        assert isinstance(after, float) and math.isnan(after), path
    else:
        assert before == after, f"{path}: {before!r} != {after!r}"


def test_4dof_resolved_config_is_unchanged_on_every_pre_existing_key():
    """The shipped hw_mpc.yaml with the variant keys STRIPPED (= the file as
    it was) resolves to the same value on every key it had; the shipped file
    resolves with the variant OFF."""
    import yaml
    from rov_gui.control.geometry import MpcConfig

    src = ROOT / "config" / "hw_mpc.yaml"
    raw = yaml.safe_load(src.read_text(encoding="utf-8"))
    full = MpcConfig.load(src)
    assert full.engage["attitude_axes"]["enabled"] is False
    assert full.policy["attitude_track"] is False
    assert full.engage["attitude_axes"]["transport"] == "manual_control_ext"
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "hw_mpc_4dof.yaml"
        f.write_text(yaml.safe_dump(_strip_variant_keys(raw)), encoding="utf-8")
        old = MpcConfig.load(f)
    for fld in dataclasses.fields(MpcConfig):
        if fld.name == "raw":
            continue
        b, a = getattr(old, fld.name), getattr(full, fld.name)
        if fld.name in ("axis_gain", "engage", "policy", "replay"):
            # the variant ADDS keys here; every pre-existing one must match
            b = {k: v for k, v in b.items()
                 if k not in ("roll_nm", "pitch_nm", "attitude_axes", "track_attitude",
                              "action_repr", "attitude_track", "rp_max_deg",
                              "rp_reject_deg", "pq_max_rad_s", "rp_jump_max_deg",
                              "anchor_leash_rp_deg", "div_max_rp_deg",
                              "blend_rp_rate_max", "heave_trim_attitude_rotated",
                              "rp_ref_filter", "attitude_q_scale")}
            if fld.name == "engage":
                b = dict(b, stabilize={k: v for k, v in b["stabilize"].items()
                                       if k != "rp_axis"})
        _assert_subset_equal(b, a, f"cfg.{fld.name}")
    # ...and a hand-built MpcConfig() keeps the pre-variant stabilize dict
    assert MpcConfig().engage["stabilize"] == {"yaw_axis": "hold"}


# =============================================================================
# command sink — wire frames
# =============================================================================
class _Mav:
    def __init__(self, raise_on_ext: bool = False):
        self.calls: list[tuple] = []
        self.raise_on_ext = raise_on_ext
        self.cmds: list[tuple] = []

    def manual_control_send(self, *args):
        if self.raise_on_ext and len(args) > 6:
            raise TypeError("manual_control_send() takes 7 positional arguments")
        self.calls.append(tuple(args))

    def command_long_send(self, *args):
        self.cmds.append(tuple(args))

    def heartbeat_send(self, *a):
        pass


class _Master:
    def __init__(self, v2: bool = True, raise_on_ext: bool = False):
        self.mav = _Mav(raise_on_ext)
        self.clients = {("1.2.3.4", 1)}
        self._v2 = v2
        self.WIRE_PROTOCOL_VERSION = "2.0" if v2 else "1.0"
        self.queue: list = []

    def mavlink20(self):
        return self._v2

    def recv_msg(self):
        return self.queue.pop(0) if self.queue else None

    def close(self):
        pass


class _Log:
    def __init__(self):
        self.lines: list[tuple[str, str]] = []

    def emit(self, lvl, msg):
        self.lines.append((lvl, msg))


def _sink(v2: bool = True, raise_on_ext: bool = False):
    from rov_gui.backends.hardware import MavlinkCommandSink
    from rov_gui.qt import QtWidgets

    global _APP
    _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    bus = types.SimpleNamespace(log=_Log())
    opts = types.SimpleNamespace(cmd_sysid=255, mavlink_out="udpin:0.0.0.0:14552")
    s = MavlinkCommandSink(bus, opts)
    s.master = _Master(v2=v2, raise_on_ext=raise_on_ext)
    s.listening = True
    return s, bus.log


_APP = None


def test_sink_off_sends_the_six_arg_frame_and_drops_k_m():
    s, log = _sink()
    cmd = PilotInput(surge=0.5, sway=-0.25, heave=0.1, yaw=0.3, roll=0.2, pitch=-0.3, stamp=1.0)
    s._send(cmd, buttons=7)
    assert s.master.mav.calls == [(1, 500, -250, 550, 300, 7)]
    assert s.attitude_axes_dropped == 1 and not s.attitude_axes_ext_sent
    # on, but a 4-DoF command: still the six-arg frame
    s.set_attitude_axes(True)
    s._send(PilotInput(surge=0.5, sway=-0.25, heave=0.1, yaw=0.3, stamp=1.0), buttons=7)
    assert s.master.mav.calls[-1] == (1, 500, -250, 550, 300, 7)
    assert s.attitude_axes_dropped == 1


def test_sink_on_sends_enabled_extensions_0b11_with_s_pitch_t_roll_and_sign():
    s, log = _sink()
    s.set_attitude_axes(True, sign=(1.0, 1.0))
    cmd = PilotInput(surge=0.5, sway=-0.25, heave=0.1, yaw=0.3, roll=0.2, pitch=-0.3, stamp=1.0)
    s._send(cmd, buttons=7)
    assert s.master.mav.calls[-1] == (1, 500, -250, 550, 300, 7, 0, 0b11, -300, 200)
    assert s.attitude_axes_ext_sent and s.attitude_axes_dropped == 0
    s.set_attitude_axes(True, sign=(-1.0, 1.0))
    s._send(cmd, buttons=0)
    assert s.master.mav.calls[-1] == (1, 500, -250, 550, 300, 0, 0, 0b11, -300, -200)
    s.set_attitude_axes(True, sign=(1.0, -1.0))
    s._send(cmd, buttons=0)
    assert s.master.mav.calls[-1][-2:] == (300, 200)
    # clamped at +-1000 whatever the axis says
    s._send(PilotInput(roll=1.0, pitch=-1.0, stamp=1.0), buttons=0)
    assert s.master.mav.calls[-1][-2:] == (1000, 1000)
    st = s.attitude_axes_state()
    assert st["enabled"] and not st["degraded"] and st["ext_sent"]
    assert st["sign"] == (1.0, -1.0)


def test_sink_type_error_degrades_to_the_six_arg_frame_once():
    s, log = _sink(raise_on_ext=True)
    s.set_attitude_axes(True)
    cmd = PilotInput(surge=0.5, roll=0.2, pitch=-0.3, stamp=1.0)
    s._send(cmd, buttons=0)
    assert s.master.mav.calls == [(1, 500, 0, 500, 0, 0)], "resent with K/M zero"
    assert s.attitude_axes_degraded and s.attitude_axes_degraded_at is not None
    assert any(lvl == "error" and "DEGRADED" in m for lvl, m in log.lines)
    n_err = sum(1 for lvl, _ in log.lines if lvl == "error")
    s._send(cmd, buttons=0)                       # degraded: never tries again
    assert s.master.mav.calls[-1] == (1, 500, 0, 500, 0, 0)
    assert s.attitude_axes_dropped == 1
    assert sum(1 for lvl, _ in log.lines if lvl == "error") == n_err, "one error log"
    assert s.attitude_axes_state()["degraded"]


def test_sink_refuses_the_extension_on_a_v1_link():
    s, log = _sink(v2=False)
    s.set_attitude_axes(True)
    s._send(PilotInput(roll=0.2, stamp=1.0), buttons=0)
    assert s.master.mav.calls == [(1, 0, 0, 500, 0, 0)]
    assert s.attitude_axes_degraded and s.attitude_axes_dropped == 1


def test_off_frame_bytes_equal_the_six_arg_frame_on_the_real_v20_packer():
    """The all-zero extension kwargs pack to the identical 23-byte MAVLink 2
    frame (pymavlink zero-truncates trailing extension fields), so the sink's
    OFF path is byte-identical on the wire; the ON frame is 30 bytes with
    enabled_extensions 3, s, t in the extension slots."""
    pytest.importorskip("pymavlink")
    from pymavlink.dialects.v20 import ardupilotmega as m20

    class Buf:
        def write(self, b):
            pass

    mav = m20.MAVLink(Buf(), srcSystem=255, srcComponent=190)
    mav.seq = 5
    a = mav.manual_control_encode(1, 100, -200, 500, 300, 7).pack(mav)
    mav.seq = 5
    b = mav.manual_control_encode(1, 100, -200, 500, 300, 7, 0, 0, 0, 0).pack(mav)
    assert a == b and len(a) == 23
    mav.seq = 5
    c = mav.manual_control_encode(1, 100, -200, 500, 300, 7, 0, 0b11, 300, -400).pack(mav)
    assert len(c) == 30 and c != a
    m = m20.MAVLink_manual_control_message
    assert m.fieldnames[6:10] == ["buttons2", "enabled_extensions", "s", "t"]


def test_sink_notes_autopilot_version_and_rc_channels_from_the_drain():
    from rov_gui.backends.hardware import firmware_version_str

    s, log = _sink()
    assert firmware_version_str((4 << 24) | (1 << 16) | (2 << 8) | 255) == "4.1.2"
    assert firmware_version_str(None) == "" and firmware_version_str(0) == ""

    class Msg:
        def __init__(self, t, **kw):
            self._t = t
            self.__dict__.update(kw)

        def get_type(self):
            return self._t

        def to_dict(self):
            return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    s.master.queue = [Msg("AUTOPILOT_VERSION", flight_sw_version=(4 << 24) | (5 << 16) | (1 << 8)),
                      Msg("RC_CHANNELS", **{f"chan{i}_raw": 1500 + i for i in range(1, 19)})]
    s._drain()
    assert s.firmware_version == "4.5.1"
    assert s.rc_chan_raw == tuple(1500 + i for i in range(1, 9))
    # the request goes out once a peer exists, as MAV_CMD_REQUEST_MESSAGE(148)
    s.firmware_version = ""
    s._request_autopilot_version()
    assert s.master.mav.cmds and s.master.mav.cmds[-1][2] == 512 and s.master.mav.cmds[-1][4] == 148
    # ...and not again within 5 s
    s._request_autopilot_version()
    assert len(s.master.mav.cmds) == 1


def test_null_sink_slot_parity_and_no_transmit_path():
    import inspect
    from rov_gui.backends import hardware

    for name in ("set_attitude_axes", "attitude_axes_state"):
        assert hasattr(hardware.NullCommandSink, name)
        assert hasattr(hardware.MavlinkCommandSink, name)
    src = inspect.getsource(hardware.NullCommandSink)
    for forbidden in ("manual_control_send", "mav.", "command_long_send"):
        assert forbidden not in src
    from rov_gui.qt import QtWidgets
    global _APP
    _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    n = hardware.NullCommandSink(types.SimpleNamespace(log=_Log()), types.SimpleNamespace())
    n.set_attitude_axes(True, (1.0, -1.0))
    st = n.attitude_axes_state()
    assert st["enabled"] and st["sign"] == (1.0, -1.0) and st["firmware_version"] == ""
    assert set(st) == set(_sink()[0].attitude_axes_state()), "one state surface on both sinks"


def test_telemetry_producer_carries_the_attitude_axes_facts():
    """VehicleWorker._publish fills Telemetry.firmware_version /
    mavlink_wire_version / attitude_axes_enabled / degraded / rc_chan_raw
    from the sink's state hook and its own AUTOPILOT_VERSION / RC_CHANNELS."""
    from rov_gui.backends.hardware import VehicleWorker
    from rov_gui.qt import QtWidgets
    from rov_gui.state import Telemetry

    global _APP
    _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    got: list[Telemetry] = []
    bus = types.SimpleNamespace(
        telemetry=types.SimpleNamespace(emit=got.append),
        thrusters=types.SimpleNamespace(emit=lambda *_: None),
        log=_Log(), link=types.SimpleNamespace(emit=lambda *_: None))
    vw = VehicleWorker(bus, types.SimpleNamespace(blueos_host="127.0.0.1"))
    assert "AUTOPILOT_VERSION" in vw._log_msgs and "RC_CHANNELS" in vw._log_msgs
    vw._latest["RC_CHANNELS"] = {f"chan{i}_raw": 1500 + i for i in range(1, 9)}
    vw._latest["AUTOPILOT_VERSION"] = {"flight_sw_version": (4 << 24) | (1 << 16) | (2 << 8)}
    vw._publish_thrusters = lambda: None
    vw._publish()
    t = got[-1]
    assert t.firmware_version == "4.1.2" and t.rc_chan_raw == tuple(1500 + i for i in range(1, 9))
    assert t.mavlink_wire_version == "" and not t.attitude_axes_enabled
    vw.attitude_axes_fn = lambda: {"enabled": True, "degraded": True,
                                   "firmware_version": "4.5.1", "mavlink_wire_version": "2.0"}
    vw._publish()
    t = got[-1]
    assert t.firmware_version == "4.5.1" and t.mavlink_wire_version == "2.0"
    assert t.attitude_axes_enabled and t.attitude_axes_degraded


# =============================================================================
# station bridge — coast tier
# =============================================================================
def test_station_bridge_coast_zeroes_k_m_only_under_attitude_axes():
    from rov_gui.control.station_bridge import StationBridge

    b = StationBridge()
    u = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert np.array_equal(b.release_horizontal(u), [0.0, 0.0, 3.0, 4.0, 5.0, 0.0])
    assert b.meta()["released_in_coast"] == "X, Y, N (heave and attitude kept)"
    b.attitude_axes = True
    assert np.array_equal(b.release_horizontal(u), [0.0, 0.0, 3.0, 0.0, 0.0, 0.0])
    assert np.array_equal(b.release_horizontal(u, attitude=False), [0.0, 0.0, 3.0, 4.0, 5.0, 0.0])
    assert "K, M" in b.meta()["released_in_coast"]


# =============================================================================
# demo plant
# =============================================================================
def test_demo_pendulum_is_inert_off_and_answers_k_m_on():
    from rov_gui.backends.demo import DemoVehicleWorker
    from rov_gui.qt import QtWidgets

    global _APP
    _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    bus = types.SimpleNamespace(log=_Log())
    w = DemoVehicleWorker(bus, opts=types.SimpleNamespace(mpc=False))
    assert not w.attitude_axes
    cmd = PilotInput(roll=0.5, pitch=0.5, stamp=1.0)
    for _ in range(50):
        w._step_pendulum(cmd, 0.05)
    assert np.array_equal(w._pend, [0.0, 0.0]), "OFF: K/M never reach the toy plant"
    w.set_attitude_axes(True, (1.0, 1.0))
    for _ in range(50):
        w._step_pendulum(cmd, 0.05)
    assert w._pend[0] > 0.05 and w._pend[1] > 0.05, "ON: +roll/+pitch axes tilt it"
    # the sign the sink applies is applied here too
    w2 = DemoVehicleWorker(bus, opts=types.SimpleNamespace(mpc=False))
    w2.set_attitude_axes(True, (-1.0, 1.0))
    for _ in range(50):
        w2._step_pendulum(cmd, 0.05)
    assert w2._pend[0] < 0 < w2._pend[1]
    # with the torque removed it settles back (a pendulum, not an integrator)
    for _ in range(400):
        w._step_pendulum(PilotInput(stamp=1.0), 0.05)
    assert abs(w._pend[0]) < 0.01 and abs(w._pend[1]) < 0.01
    st = w.attitude_axes_state()
    assert st["enabled"] and st["mavlink_wire_version"] == "2.0"


# =============================================================================
# the bench probe (fake link)
# =============================================================================
class _Msg:
    def __init__(self, t, comp=1, **kw):
        self._t = t
        self._comp = comp
        self.__dict__.update(kw)

    def get_type(self):
        return self._t

    def get_srcComponent(self):
        return self._comp


class _ProbeMaster:
    """Answers HEARTBEAT (disarmed), AUTOPILOT_VERSION, and RC_CHANNELS that
    follow the last s/t frame like a 4.1.2+ vehicle would [예측]."""

    def __init__(self, reads_st: bool = True, armed: bool = False, arm_after: int | None = None):
        self.reads_st = reads_st
        self.armed = armed
        self.arm_after = arm_after
        self.frames: list[tuple] = []
        self.s = self.t = 0
        self.mav = self
        self._n = 0

    def mavlink20(self):
        return True

    def manual_control_send(self, *args):
        self.frames.append(args)
        if len(args) > 6:
            self.s, self.t = args[8], args[9]
        else:
            self.s = self.t = 0
        if self.arm_after is not None and len(self.frames) > self.arm_after:
            self.armed = True

    def command_long_send(self, *args):
        pass

    def heartbeat_send(self, *args):
        pass

    def recv_msg(self):
        self._n += 1
        k = self._n % 3
        if k == 0:
            return _Msg("HEARTBEAT", base_mode=(128 if self.armed else 0), custom_mode=19)
        if k == 1:
            return _Msg("AUTOPILOT_VERSION", flight_sw_version=(4 << 24) | (5 << 16) | (1 << 8))
        gain = 0.4 if self.reads_st else 0.0
        return _Msg("RC_CHANNELS", **{"chan1_raw": int(1500 + gain * self.s),
                                      "chan2_raw": int(1500 + gain * self.t),
                                      **{f"chan{i}_raw": 1500 for i in range(3, 9)}})


def _fake_clock():
    t = [0.0]

    def clock():
        return t[0]

    def sleep(dt):
        t[0] += dt
    return clock, sleep


def test_probe_passes_on_a_firmware_that_reads_s_t_and_fails_otherwise():
    from rov_gui.tools import attitude_axes_probe as P

    clock, sleep = _fake_clock()
    m = _ProbeMaster(reads_st=True)
    res = P.run_probe(m, amplitude=300, hold_s=1.0, settle_s=0.5, rate_hz=10.0,
                      clock=clock, sleep=sleep, log=lambda *_: None)
    assert res["pass"] and res["verdict"] == "PASS"
    assert res["firmware_version"] == "4.5.1" and res["mavlink_wire_version"] == "2.0"
    assert res["deltas_us"]["chan1_under_s"] == pytest.approx(120.0)
    assert res["deltas_us"]["chan2_under_t"] == pytest.approx(120.0)
    assert res["sign_proven"] is False and res["armed_during_probe"] is False
    ext = [f for f in m.frames if len(f) > 6]
    assert ext and all(f[7] == 0b11 for f in ext)
    assert any(f[8] == 300 and f[9] == 0 for f in ext) and any(f[8] == 0 and f[9] == 300 for f in ext)
    assert m.frames[-1][:6] == (1, 0, 0, 500, 0, 0), "ends neutral"

    clock, sleep = _fake_clock()
    res2 = P.run_probe(_ProbeMaster(reads_st=False), hold_s=1.0, settle_s=0.5,
                       clock=clock, sleep=sleep, log=lambda *_: None)
    assert not res2["pass"] and "미확인" in res2["verdict_note"]


def test_probe_refuses_armed_before_and_aborts_armed_during():
    from rov_gui.tools import attitude_axes_probe as P

    clock, sleep = _fake_clock()
    with pytest.raises(P.ProbeAbort, match="ARMED"):
        P.run_probe(_ProbeMaster(armed=True), clock=clock, sleep=sleep, log=lambda *_: None)
    clock, sleep = _fake_clock()
    m = _ProbeMaster(arm_after=8)
    with pytest.raises(P.ProbeAbort, match="ARMED during"):
        P.run_probe(m, hold_s=1.0, settle_s=0.5, clock=clock, sleep=sleep, log=lambda *_: None)
    assert m.frames[-1] == (1, 0, 0, 500, 0, 0), "neutral sent on abort"


def test_probe_artefact_lands_in_a_dated_probe_run_folder():
    from rov_gui.tools import attitude_axes_probe as P

    with tempfile.TemporaryDirectory() as tmp:
        out = P.write_artefact({"pass": True, "verdict": "PASS"}, base=tmp, when=1_700_000_000.0)
        assert out.name == "attitude_axes_probe.json"
        assert out.parent.name.endswith("_probe") and out.parent.parent.name.isdigit()
        d = json.loads(out.read_text())
        assert d["pass"] is True and "written_at" in d
