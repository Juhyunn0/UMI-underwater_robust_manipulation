#!/usr/bin/env python3
"""LOW modes pid_7d / mpc_tuned_7d / rl_7d (2026-10-03, operator request "pid_7d, mpc_tuned_7d, RL_7d 버전").

Pinned here:
  * vocabulary: the three keys everywhere a LOW key is spelled, mpc_tuned_7d NEVER a HwDobMpc mode (its `tuned` flag is
    endswith("_tuned")) and mapped to "mpc_tuned" by the worker, an unknown key never falls through to the MPC;
  * HwPid7d: X/Y/Z/N bitwise HwPid's at level, K/M signs, the restoring FF on the reference, T3 plan validation, the D5
    hold ramp, conditional integration on the realized K/M, the wire-cap clamp, the recorded design;
  * HwRl7d: fail-closed acceptance (a level-trained export, a wrong layout, a non-axes6 export, no export), q_des from
    the plan's roll/pitch = the training convention (rl-controller parity), the envelope clip, non-finite -> n_fail;
  * the worker: THE LOW DECIDES — a *_7d LOW without engage.attitude_axes is refused, with it K/M go out; a 4-DoF LOW
    under the capability engages 4-DoF; rl_7d refused under caps below its training; a tracked 7-dim mission on pid_7d
    reaches the PID and the CSV reference columns; land dry-run refuses tracking; the meta stamps the LOW;
  * allocation.slew_axes slews roll/pitch even with the 4-axis slew off (rl_7d).

Run: QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/robust/bin/python -m pytest rov_gui/tests/test_low_7d.py -q
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rov_gui.control.geometry import LOW_7D_BASE, MpcConfig
from rov_gui.control.path_geometry import NedPlan
from rov_gui.control.pid import HwPid, HwPid7d
from rov_gui.state import ACTION_REPR_POS_RPY_WIDTH
from rov_gui.tests.test_control import (_alloc_has_attitude, _att_worker, _engage_att, _feed_att, _last_csv_row)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FIXTURE_SPEC = HERE / "fixtures" / "rl7d_smoke_obs_spec.json"


def _pulse_json(tmp, periods=(3.62, 4.71)) -> str:
    """A pulse-and-release record as the operator writes it from the open-loop test — SYNTHETIC (the default periods
    are the demo pendulum's 2 pi sqrt(I/k), not a vehicle's)."""
    f = Path(tmp) / "pulse_test_SYNTHETIC.json"
    f.write_text(json.dumps({"tool": "manual (test_low_7d SYNTHETIC)", "free_decay_period_s": list(periods),
                             "note": "synthetic — proves nothing about a vehicle"}))
    return str(f)


def _settle(o, n=60):
    """Advance an RL follower's flown reference to its target (trained goal rate 0.35 rad/s x 0.05 s per tick)."""
    for _ in range(n):
        o._advance_rp_flown()


def _plan1(rp=None, rp_rate=None, p=(0.0, 0.0, 0.8), yaw=0.0):
    """A one-stage NedPlan (what the worker installs on the PID family)."""
    kw = {}
    if rp is not None:
        kw["rp_ned"] = np.asarray(rp, float).reshape(2, 1)
    if rp_rate is not None:
        kw["rp_rate_ned"] = np.asarray(rp_rate, float).reshape(2, 1)
    return NedPlan(p_ned=np.asarray(p, float).reshape(3, 1), yaw_ned=np.array([yaw]),
                   v_ned=np.zeros((3, 1)), r_ned=np.zeros(1), psi_path=np.zeros(1), **kw)


# =============================================================================
# vocabulary
# =============================================================================
def test_the_7d_keys_are_spelled_everywhere_and_never_a_solver_mode():
    from rov_gui.control.mpc_bridge import HwDobMpc
    from rov_gui.control.workers import MpcWorker
    from rov_gui.widgets.trajectory import MODE_LABELS
    assert LOW_7D_BASE == {"pid_7d": "pid", "mpc_tuned_7d": "mpc_tuned", "rl_7d": "rl"}
    for k in LOW_7D_BASE:
        assert k in MpcWorker.MODES and k in MODE_LABELS and k in MpcWorker.ATTITUDE_LOW_ALLOWED
        assert k not in HwDobMpc.MODES
    assert set(MpcWorker.ATTITUDE_LOW_ALLOWED) == set(LOW_7D_BASE)
    assert not "mpc_tuned_7d".endswith("_tuned")          # why the worker maps it, never passes it
    src = (ROOT / "rov_gui" / "__main__.py").read_text(encoding="utf-8")
    for k in LOW_7D_BASE:
        assert f'"{k}"' in src
    with tempfile.TemporaryDirectory() as tmp:
        for k in LOW_7D_BASE:
            f = Path(tmp) / f"{k}.yaml"
            f.write_text(f"mode: {k}\n")
            assert MpcConfig.load(f).mode == k
        f = Path(tmp) / "bad.yaml"
        f.write_text("mode: mpc_7d\n")
        with pytest.raises(ValueError, match="pid_7d|mpc_tuned_7d|rl_7d"):
            MpcConfig.load(f)


def test_set_mode_maps_mpc_tuned_7d_to_its_base_and_unknown_keys_never_fly_the_mpc():
    """With a STRICT mode setter (HwDobMpc's rule: non-members raise) the worker sets the BASE, and a setter that
    raises leaves the worker whole on its previous LOW."""
    from rov_gui.tests.test_control import StubCtrl

    class Strict(StubCtrl):
        MODES = ("mpc", "dobmpc", "mpc_tuned", "dobmpc_tuned")

        def __init__(self, cfg):
            super().__init__(cfg)
            self._m = "dobmpc"

        @property
        def mode(self):
            return self._m

        @mode.setter
        def mode(self, v):
            if v not in self.MODES:
                raise ValueError(f"unknown mode {v!r}")
            self._m = v

    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs, st = _att_worker(tmp, mode=None, factory=Strict)
        w.set_mode("mpc_tuned_7d")
        assert w.cfg.mode == "mpc_tuned_7d" and w.ctrl is w._mpc_ctrl and w._mpc_ctrl.mode == "mpc_tuned"
        w.set_mode("pid_7d")
        assert w.cfg.mode == "pid_7d" and isinstance(w.ctrl, HwPid7d) and w.ctrl is not w._pid
        assert w._mpc_ctrl.mode == "mpc_tuned"              # untouched by a PID-family switch
        w.set_mode("pid")
        assert w.ctrl is w._pid and type(w.ctrl) is HwPid
        assert w._ctrl_for("mpc_7d") is None and w._ctrl_for("nonsense") is None
        w.set_mode("rl_7d")                                 # no export trained -> refused, reason shown
        assert w.cfg.mode == "pid" and "rl_7d unavailable" in w.reason and "no policy_dir" in w.reason
        w.teardown()


def test_the_real_hwdobmpc_flies_mpc_tuned_7d_tuned():
    """The bridge's own flags after the worker's mapping (skips without acados)."""
    try:
        from rov_gui.control.mpc_bridge import HwDobMpc
        ctrl = HwDobMpc("mpc", MpcConfig(), log=lambda m: None)
    except Exception as e:                                   # noqa: BLE001
        pytest.skip(f"no acados build here ({type(e).__name__}: {e})")
    ctrl.mode = LOW_7D_BASE["mpc_tuned_7d"]
    assert ctrl.mode == "mpc_tuned" and ctrl.tuned and not ctrl.dob
    with pytest.raises(ValueError):
        ctrl.mode = "mpc_tuned_7d"


# =============================================================================
# HwPid7d
# =============================================================================
def _pid_pair(**over):
    cfg = MpcConfig()
    cfg.vehicle_net_buoyancy_n = -12.0
    cfg.pid_7d = dict(over)
    return HwPid(cfg, log=lambda m: None), HwPid7d(cfg, log=lambda m: None)


def test_pid7d_translation_and_yaw_are_bitwise_the_pid_at_level():
    a, b = _pid_pair()
    rng = np.random.default_rng(3)
    for o in (a, b):
        o.set_target_ned([0.3, -0.2, 0.9], 0.4, v_ned=[0.05, 0.0, 0.0], r_ned=0.1)
    for k in range(60):
        eta = np.r_[rng.normal(0, 0.2, 3), 0.0, 0.0, rng.normal(0, 0.3)]
        nu = np.r_[rng.normal(0, 0.1, 3), 0.0, 0.0, rng.normal(0, 0.1)]
        ua, _ = a.step(eta, nu, np.zeros(6), 0.05 * k)
        ub, _ = b.step(eta, nu, np.zeros(6), 0.05 * k)
        assert np.array_equal(ua[[0, 1, 2, 5]], ub[[0, 1, 2, 5]]), (k, ua, ub)
        assert ua[3] == 0.0 and ua[4] == 0.0
    assert b.meta()["type"] == "pid_7d" and a.meta()["type"] == "pid"


def test_pid7d_km_signs_ff_and_design_record():
    _a, p = _pid_pair()
    p.set_attitude_axes(True, u_max_wire_nm=[2.64, 2.16])
    p.set_target_ned([0.0, 0.0, 0.8], 0.0)
    # level reference, hull rolled starboard-down 5 deg + pitched nose-up 5 deg -> push both back (negative K, M)
    u, _ = p.step(np.r_[0, 0, 0.8, math.radians(5), math.radians(5), 0], np.zeros(6), np.zeros(6), 0.0)
    assert u[3] < 0 and u[4] < 0
    p.reset()
    # a tracked 10 deg nose-up reference on a level hull: positive M, at least the restoring FF
    p.set_path_plan_ned(_plan1(rp=[0.0, math.radians(10)]))
    u, info = p.step(np.r_[0, 0, 0.8, 0, 0, 0], np.zeros(6), np.zeros(6), 0.0)
    ff = 1.35 * math.sin(math.radians(10))
    assert u[4] > ff and abs(u[3]) < 1e-12 and info["rp_err"][1] > 0
    m = p.meta()
    rl = m["rp_loop"]
    assert rl["k_ff_nm_per_rad"] == 1.35 and rl["omega_derate_applied"] is False
    assert abs(rl["kp_nm_per_rad"][1] - 0.8978 * 1.36 * 0.25) < 1e-9 and rl["ki_nm_per_rad_s"] == [0.15, 0.15]
    assert m["attitude_ref"]["tracked"] is True and m["allocation"]["attitude"] is True
    assert m["heave_trim"]["attitude_rotated"] is True
    json.dumps(m)


def test_pid7d_validates_the_plan_attitude_t3_and_its_config():
    _a, p = _pid_pair()
    with pytest.raises(ValueError, match="rp_reject"):
        p.set_path_plan_ned(_plan1(rp=[0.0, math.radians(31)]))
    class _Bad:                                            # a plan object whose rp has the wrong stage count
        p_ned, yaw_ned, v_ned, r_ned = np.zeros((3, 1)), np.zeros(1), np.zeros((3, 1)), np.zeros(1)
        rp_ned, rp_rate_ned = np.zeros((2, 3)), None
    with pytest.raises(ValueError, match="rp_ned needs shape"):
        p.set_path_plan_ned(_Bad())
    with pytest.raises(ValueError, match="unknown key"):
        _pid_pair(rp_kp_typo=1.0)
    with pytest.raises(ValueError, match="gains must be"):
        _pid_pair(rp_k_model=[5.0, 5.0])                  # a formula Kp below zero
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "m.yaml"
        f.write_text("pid_7d:\n  rp_wn: 0.5\n  rp_ki_typo: 1\n")
        with pytest.raises(ValueError, match="pid_7d: unknown"):
            MpcConfig.load(f)


def test_pid7d_hold_ramp_after_a_tracked_plan_and_level_otherwise():
    _a, p = _pid_pair()
    assert p.ref_attitude_ned_at() == (0.0, 0.0) and p.attitude_ref_source() == "level"
    assert p.attitude_rate_source() == "none"
    p.set_path_plan_ned(_plan1(rp=[0.0, math.radians(10)], rp_rate=[0.0, 0.1]))
    assert p.attitude_ref_source() == "plan" and p.ref_attitude_rate_ned_at() == (0.0, 0.1)
    p.set_target_ned([0.0, 0.0, 0.8], 0.0)                  # the plan ends: ramp, not a step
    assert p.attitude_ref_source() == "hold_ramp"
    th0 = p.ref_attitude_ned_at()[1]
    assert abs(th0 - math.radians(10)) < 1e-12 and p.ref_attitude_rate_ned_at()[1] == -0.35
    p.step(np.r_[0, 0, 0.8, 0, 0, 0], np.zeros(6), np.zeros(6), 0.0)
    assert abs(p.ref_attitude_ned_at()[1] - (th0 - 0.35 * 0.05)) < 1e-12
    for _ in range(20):
        p.step(np.r_[0, 0, 0.8, 0, 0, 0], np.zeros(6), np.zeros(6), 0.0)
    assert p.ref_attitude_ned_at() == (0.0, 0.0) and p.attitude_ref_source() == "level"
    assert p.attitude_rate_source() == "stitcher_rate_T_inv"
    p.reset()
    assert p.attitude_rate_source() == "none"


def test_pid7d_integral_clamped_to_the_wire_and_frozen_when_the_wire_cut_it():
    _a, p = _pid_pair()
    p.set_attitude_axes(True, u_max_wire_nm=[1.32, 1.08])   # first-water caps x roll_nm/pitch_nm
    p.set_path_plan_ned(_plan1(rp=[0.0, math.radians(15)]))
    eta = np.r_[0, 0, 0.8, 0, 0, 0]
    for _ in range(4000):                                   # 200 s of an unreachable error, the wire carrying all
        u, _ = p.step(eta, np.zeros(6), np.zeros(6), 0.0)
        p.note_applied(np.r_[0, 0, 0, u[3], u[4], 0])
    assert abs(p._i_rp[1] - 0.8 * 1.08) < 1e-9 and abs(u[4]) <= 1.08 + 1e-12
    p.reset()
    p.set_attitude_axes(True, u_max_wire_nm=[1.32, 1.08])
    p.set_path_plan_ned(_plan1(rp=[0.0, math.radians(15)]))
    u, _ = p.step(eta, np.zeros(6), np.zeros(6), 0.0)
    i0 = p._i_rp[1]
    for _ in range(50):                                     # the wire carried LESS than asked, toward the error
        p.note_applied(np.r_[0, 0, 0, 0, 0.5 * u[4], 0])
        u, _ = p.step(eta, np.zeros(6), np.zeros(6), 0.0)
    assert p._i_rp[1] == i0                                 # frozen (conditional integration)


def test_pid7d_trim_rotates_with_the_measured_attitude():
    a, b = _pid_pair()
    for o in (a, b):
        o.set_target_ned([0.0, 0.0, 0.8], 0.0)
    eta = np.r_[0, 0, 0.8, 0.0, math.radians(20), 0.0]
    for _ in range(30):                                     # past HwPid's 120 N/s uniform slew (6 N per tick)
        ua, _ = a.step(eta, np.zeros(6), np.zeros(6), 0.0)
        ub, _ = b.step(eta, np.zeros(6), np.zeros(6), 0.0)
    # same position error (same integrals), but pid_7d puts the WORLD-vertical -12 N through R^T:
    # body X = -wz sin(theta) = +12 sin 20, body Z = wz cos(theta) = -12 cos 20 (pid: 0 and -12)
    assert abs((ub[0] - ua[0]) - (12.0 * math.sin(math.radians(20)))) < 1e-6
    assert abs((ub[2] - ua[2]) - (-12.0 * math.cos(math.radians(20)) + 12.0)) < 1e-6


# =============================================================================
# HwRl7d
# =============================================================================
def _rl7d_export(tmp, spec_over=None, trained=True):
    """A FORMAT export: the real exporter's spec (fixtures/rl7d_smoke_obs_spec.json, a 3-iteration smoke) with zero
    weights and a constant non-zero action — never a policy."""
    spec = json.loads(FIXTURE_SPEC.read_text(encoding="utf-8"))
    if not trained:
        spec.pop("attitude_reference", None)
    spec.update(spec_over or {})
    pdir = Path(tmp) / "rl7d"
    pdir.mkdir(exist_ok=True)
    (pdir / "obs_spec.json").write_text(json.dumps(spec))
    n_in, n_act = int(spec["obs_dim"]), int(spec["num_actions"])
    np.savez(pdir / "policy_weights.npz", n_layers=np.array(2), activation=np.array("elu"),
             W0=np.zeros((8, n_in), np.float32), b0=np.zeros(8, np.float32),
             W1=np.zeros((n_act, 8), np.float32), b1=np.linspace(-0.3, 0.3, n_act).astype(np.float32))
    return pdir


def _rl7d(tmp, **kw):
    from rov_gui.control.rl_policy import HwRl7d
    cfg = MpcConfig()
    cfg.rl_7d = {"policy_dir": str(_rl7d_export(tmp, **kw))}
    return HwRl7d(cfg, log=lambda m: None)


def test_rl7d_refuses_everything_but_a_roll_pitch_trained_axes6_export():
    from rov_gui.control.rl_policy import HwRl7d
    cfg = MpcConfig()
    with pytest.raises(ValueError, match="no policy_dir"):
        HwRl7d(cfg, log=lambda m: None)
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="LEVEL goal"):
            _rl7d(tmp, trained=False)
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="frame_layout"):
            _rl7d(tmp, spec_over={"frame_layout": ["q_des(w,x,y,z)"]})
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="axes6"):
            _rl7d(tmp, spec_over={"action_mode": "pwm"})
    cfg.rl_7d = {"policy_dir": "x", "leash": 1.0}
    with pytest.raises(ValueError, match="unknown key"):
        HwRl7d(cfg, log=lambda m: None)


def test_rl7d_q_des_is_the_training_convention_and_clipped_to_the_envelope():
    from rov_gui.control.rl_policy import HwRl, _unique, quat_from_euler_xyz
    with tempfile.TemporaryDirectory() as tmp:
        rl = _rl7d(tmp)
        assert rl.mode == "rl_7d" and rl.n_act == 6
        assert abs(rl.att_pitch_max - 0.35) < 1e-12 and abs(rl.att_roll_max - 0.17) < 1e-12
        rl.set_target_ned([0.0, 0.0, 0.8], 0.3)
        lvl = rl._q_des(0.3)
        assert np.allclose(lvl, quat_from_euler_xyz(0.0, 0.0, -0.3))   # level = what HwRl builds
        rl.set_path_plan_ned(_plan1(rp=[0.1, 0.2], yaw=0.3))
        _settle(rl)
        q = _unique(rl._q_des(0.3))
        # RL_controller: q_des = quat_from_euler_xyz(roll_up, pitch_up, yaw_up), roll_up = roll, pitch_up = -pitch_ned
        try:
            import torch
            from importlib import util
            spec = util.spec_from_file_location("rlmath", "/home/bdml/Desktop/RL_controller/rov_rl/utils/math.py")
            m = util.module_from_spec(spec)
            spec.loader.exec_module(m)
            q_t = m.quat_unique(m.quat_from_euler_xyz(torch.tensor([0.1]), torch.tensor([-0.2]),
                                                      torch.tensor([-0.3])))[0].double().numpy()
            assert np.allclose(q, q_t, atol=1e-6), (q, q_t)
        except (ImportError, FileNotFoundError):
            assert np.allclose(q, _unique(quat_from_euler_xyz(0.1, -0.2, -0.3)))
        rl.set_path_plan_ned(_plan1(rp=[0.3, -0.45], yaw=0.0))          # beyond the trained envelope
        n0 = rl._rp_clip_ticks
        _settle(rl)
        q = rl._q_des(0.0)
        assert np.allclose(q, quat_from_euler_xyz(0.17, 0.35, 0.0)) and rl._rp_clip_ticks == n0 + 60
        # the frame really carries it (observation[0:4])
        obs = rl.observation(np.r_[0, 0, 0.8, 0, 0, 0], np.zeros(6), 0.0)
        assert np.allclose(obs[0:4], _unique(quat_from_euler_xyz(0.17, 0.35, 0.0)))
        m = rl.meta()
        assert m["type"] == "rl_7d" and m["attitude_reference_trained"]["trained"] is True
        assert m["attitude_cap_trained_station"] == [pytest.approx(0.2), pytest.approx(0.3)]
        json.dumps(m)
        assert issubclass(type(rl), HwRl)


def test_rl7d_non_finite_is_a_counted_failure_with_a_neutral_command():
    with tempfile.TemporaryDirectory() as tmp:
        rl = _rl7d(tmp)
        rl.set_target_ned([0.0, 0.0, 0.8], 0.0)
        u, info = rl.step(np.r_[0, 0, np.nan, 0, 0, 0], np.zeros(6), np.zeros(6), 0.0)
        assert np.all(u == 0.0) and info["status"] == 1 and rl.n_fail == 1


# =============================================================================
# allocation
# =============================================================================
def test_slew_axes_slews_roll_pitch_with_the_four_axis_slew_off():
    from rov_gui.control.allocation import slew_axes
    from rov_gui.state import PilotInput
    cmd = PilotInput(surge=0.4, sway=0.0, heave=0.0, yaw=0.0, roll=0.2, pitch=-0.3, source="mpc")
    out = slew_axes(cmd, (0.0,) * 6, 0.0, 0.05, rate_rp=1.5)
    assert out.surge == 0.4 and abs(out.roll - 0.075) < 1e-12 and abs(out.pitch + 0.075) < 1e-12
    assert slew_axes(cmd, (0.0,) * 4, 0.0, 0.05, rate_rp=1.5) is cmd     # 4-tuple / 4-DoF: untouched
    assert slew_axes(cmd, (0.0,) * 6, 0.0, 0.05) is cmd                  # no rp rate given: untouched


# =============================================================================
# the worker
# =============================================================================
def _skip_no_att():
    if not _alloc_has_attitude():
        pytest.skip("allocation without the attitude path (output group)")


def test_a_7d_low_needs_the_capability_and_a_4dof_low_ignores_it():
    _skip_no_att()
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs, st = _att_worker(tmp, mode="pid_7d")
        w.cfg.engage["attitude_axes"]["enabled"] = False
        w.on_enable(True)
        _feed_att(w)
        w.set_engaged(True)
        assert not w.engaged and "engage.attitude_axes.enabled is false" in w.reason and "pick pid" in w.reason
        w.set_mode("pid")
        _feed_att(w)
        w.set_engaged(True)
        assert w.engaged and not w._attitude_axes
        w.teardown()


def test_pid_7d_tracks_a_plan_attitude_end_to_end_through_the_worker():
    """Real HwPid7d on LOW pid_7d: the K/M it computes leave as roll/pitch axes within the caps in force, the CSV's
    reference columns carry the PID's (plan) reference, and the meta records the loop and the LOW."""
    _skip_no_att()
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs, st = _att_worker(tmp, mode="pid_7d")
        w._pid7d.pulse_test = _pulse_json(tmp)
        _engage_att(w)
        assert w._attitude_axes and isinstance(w.ctrl, HwPid7d)
        assert w.ctrl.pulse_test_record["free_decay_period_s"] == [3.62, 4.71]
        assert w.ctrl._attitude_axes is True and w.ctrl._u_max_wire_nm is not None
        # install a tracked 10 deg nose-up plan the way _tick_replay does (one stage)
        w.ctrl.set_path_plan_ned(_plan1(rp=[0.0, math.radians(10.0)], p=w._eta[:3], yaw=float(w._eta[5])))
        n0 = len(pilots)
        for _ in range(10):
            _feed_att(w)
            w.ctrl.set_path_plan_ned(_plan1(rp=[0.0, math.radians(10.0)], p=w._eta[:3], yaw=float(w._eta[5])))
            w.tick()
        sent = [p for p in pilots[n0:] if p.source == "mpc"]
        # nose-up torque on the wire: FF 1.35 sin 10 + P 0.305 x 0.175 ~ 0.29 N m -> ~0.04 axis (low-gain design)
        assert sent and max(p.pitch for p in sent) > 0.03
        assert all(abs(p.pitch) <= w._attitude_cap[1] + 1e-9 for p in sent)
        row = _last_csv_row(w)
        assert row["rp_track"] == "1" and abs(float(row["rpitch_deg"]) + 10.0) < 1e-6   # CSV pitch is FLU (-NED)
        m = w._run_meta()
        assert m["controller"]["type"] == "pid_7d" and "rp_loop" in m["controller"]
        assert m["run"]["attitude_axes"]["low"] == "pid_7d" and m["run"]["attitude_axes"]["low_attitude"] is True
        json.dumps(m, default=str)
        w.teardown()


def test_mpc_tuned_7d_meta_is_stamped_with_the_low():
    _skip_no_att()
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs, st = _att_worker(tmp)            # mpc_tuned_7d on the stub
        _engage_att(w)
        m = w._run_meta()
        assert m["controller"]["type"] == "mpc_tuned_7d" and m["controller"]["base_type"] == "mpc_tuned"
        assert m["mode"] == "mpc_tuned_7d"
        w.teardown()


def test_rl_7d_refused_under_caps_below_its_training_unless_opted_out():
    _skip_no_att()
    with tempfile.TemporaryDirectory() as tmp:
        pdir = _rl7d_export(tmp)
        w, bus, pilots, logs, st = _att_worker(tmp, mode=None, sign_probe=None, require_sign_probe=False)
        from rov_gui.control.rl_policy import HwRl7d
        w.cfg.rl_7d = {"policy_dir": str(pdir)}
        w._rl7d = HwRl7d(w.cfg, log=lambda m: None)
        w.set_mode("rl_7d")
        assert w.cfg.mode == "rl_7d", w.reason
        w.on_enable(True)
        _feed_att(w)
        w.set_engaged(True)                                  # first_water 0.1/0.15 < trained 0.2/0.3
        assert not w.engaged and "below what the policy was trained with" in w.reason, w.reason
        w._rl7d.allow_caps_below_trained = True
        _feed_att(w)
        w.set_engaged(True)
        assert w.engaged and w._attitude_axes, w.reason
        # the attitude slew applies although rl's 4-axis slew is 0 (first frame <= 1.5/s x dt)
        for _ in range(2):
            _feed_att(w)
            w.tick()
        sent = [p for p in pilots if p.source == "mpc"]
        assert sent and abs(sent[0].pitch) <= 1.5 * 0.05 + 1e-9 and abs(sent[0].roll) <= 1.5 * 0.05 + 1e-9
        w.teardown()


def test_rl_pwm_is_no_longer_refused_by_the_capability():
    """engage.attitude_axes.enabled is a station capability: rl_pwm's own gate decides (the attitude sentence is gone)."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs, st = _att_worker(tmp, mode=None)
        w.cfg.mode = "rl_pwm"
        w.ctrl = type("C", (), {"transport": "pwm"})()
        from rov_gui.tests.test_control import _att_tel
        why = w._pwm_refusal(_att_tel())
        assert "attitude_axes" not in why
        w.teardown()


@pytest.mark.parametrize("action_repr", [ACTION_REPR_POS_RPY_WIDTH, "auto"])
def test_tracking_on_a_7d_low_in_land_dry_run_is_refused_at_arm(action_repr):
    """A *_7d LOW + a 7-dim checkpoint wants to track; under land dry-run nothing is actuated, so ARM says why —
    the same under policy.action_repr auto (2026-10-03: land_dp.yaml ships auto too)."""
    _skip_no_att()
    from rov_gui.tests.test_policy import _feed_status
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs, st = _att_worker(tmp, mode="pid_7d")
        w.land_dry_run = True
        w.cfg.policy.update({"action_repr": action_repr, "attitude_track": True})
        w.policy_present = True
        w.set_scenario({"shape": "policy"})
        w.on_enable(True)
        _feed_att(w)
        w.set_engaged(True)
        assert w.engaged and not w._attitude_axes, w.reason
        for _ in range(4):
            _feed_att(w)
            _feed_status(w, action_repr=ACTION_REPR_POS_RPY_WIDTH)
            w.tick()
        w.set_traj(True)
        assert not w.traj_on and "K/M path is not live" in w.reason and "land dry-run" in w.reason, w.reason
        w.cfg.policy["attitude_track"] = False                 # the veto: dropped-and-logged instead
        w.set_traj(True)
        assert w.traj_on and w.replay["attitude_track"] is False, w.reason
        assert w.replay["action_repr_pinned"] == ACTION_REPR_POS_RPY_WIDTH
        w.teardown()


# =============================================================================
# safety audit 2026-10-03 follow-ups
# =============================================================================
def test_pid_7d_engage_needs_the_pulse_record_or_the_recorded_opt_out():
    _skip_no_att()
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs, st = _att_worker(tmp, mode="pid_7d")
        w.on_enable(True)

        def refuse(needle):
            _feed_att(w)
            w.set_engaged(True)
            assert not w.engaged and needle in w.reason, (needle, w.reason)

        refuse("no pulse-and-release record")
        w._pid7d.pulse_test = str(Path(tmp) / "missing.json")
        refuse("cannot be read")
        w._pid7d.pulse_test = _pulse_json(tmp, periods=(1.0, 1.3))        # a stiff hull (k ~ 20 N m/rad)
        refuse("free-decay period 1.00 s < 3.2 s")
        bad = Path(tmp) / "bad.json"
        bad.write_text(json.dumps({"free_decay_period_s": "slow"}))
        w._pid7d.pulse_test = str(bad)
        refuse("no finite free_decay_period_s")
        w._pid7d.pulse_test = None
        w._pid7d.require_pulse_test = False                              # the RECORDED opt-out
        _feed_att(w)
        w.set_engaged(True)
        assert w.engaged, w.reason
        assert w._run_meta()["controller"]["rp_loop"]["pulse_test"]["opt_out"] is True
        w.teardown()


def test_pid7d_reports_exhausted_authority_below_the_cap():
    """The integral at its ceiling (0.8 x wire) with an error left = exhausted, although |K|,|M| < the cap."""
    _a, p = _pid_pair()
    p.set_attitude_axes(True, u_max_wire_nm=[1.32, 1.08])
    p.set_path_plan_ned(_plan1(rp=[0.0, math.radians(6)]))           # FF 0.14 + P 0.03 + I 0.86 < the 1.08 cap
    eta = np.r_[0, 0, 0.8, 0, 0, 0]
    for _ in range(4000):
        u, _ = p.step(eta, np.zeros(6), np.zeros(6), 0.0)
        p.note_applied(np.r_[0, 0, 0, u[3], u[4], 0])
    assert abs(u[4]) < 1.08 and p.attitude_authority_exhausted() == (False, True)
    # an error beyond the integration gate (20 deg) is exhausted at once, integral or not
    p.reset()
    p.set_path_plan_ned(_plan1(rp=[0.0, math.radians(10)]))
    p.step(np.r_[0, 0, 0.8, 0, -math.radians(15), 0], np.zeros(6), np.zeros(6), 0.0)
    assert p.attitude_authority_exhausted() == (False, True)
    p.set_path_plan_ned(_plan1(rp=[0.0, 0.0]))                         # error gone: not exhausted any more
    for _ in range(2):
        p.step(np.r_[0, 0, 0.8, 0, 0, 0], np.zeros(6), np.zeros(6), 0.0)
    assert p.attitude_authority_exhausted()[1] is False


def test_interlock_iii_trips_pid_7d_on_an_inverted_sign_below_the_cap():
    """Safety audit 2026-10-03 HIGH: with the wire sign inverted the hull tilts AWAY from the reference; pid_7d's axis
    stays below its cap, and before the fix nothing disengaged. Now the exhausted integral counts as pinned and (iii)
    sees the hull move against the command."""
    _skip_no_att()
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs, st = _att_worker(tmp, mode="pid_7d", sat_ineffective_s=1.0)
        w._pid7d.require_pulse_test = False
        _engage_att(w)
        w.ctrl.set_path_plan_ned(_plan1(rp=[0.0, math.radians(10.0)], p=w._eta[:3], yaw=float(w._eta[5])))
        pitch = 0.0
        for k in range(3000):
            if not w.engaged:
                break
            # the plant: an INVERTED wire -> the hull pitches DOWN while the PID asks nose-up (bounded, stiff)
            sent = [p for p in pilots if p.source == "mpc"]
            ax = sent[-1].pitch if sent else 0.0
            pitch = max(-math.radians(20.0), min(0.0, pitch - 0.02 * ax))
            _feed_att(w, pitch=pitch)
            w.ctrl.set_path_plan_ned(_plan1(rp=[0.0, math.radians(10.0)], p=w._eta[:3], yaw=float(w._eta[5])))
            w.tick()
        assert not w.engaged and "torque ineffective" in w.reason, (k, w.reason)
        assert all(abs(p.pitch) < w._attitude_cap[1] for p in pilots if p.source == "mpc")   # the axis never pinned
        w.teardown()


def test_pid7d_non_finite_attitude_is_a_counted_failure_never_a_cap():
    _a, p = _pid_pair()
    p.set_target_ned([0.0, 0.0, 0.8], 0.0)
    u, info = p.step(np.r_[0, 0, 0.8, np.nan, 0, 0], np.zeros(6), np.zeros(6), 0.0)
    assert np.all(u == 0.0) and info["status"] == 1 and info["n_fail"] == 1
    u, info = p.step(np.r_[0, 0, 0.8, 0, 0, 0], np.zeros(6), np.zeros(6), 0.0)
    assert np.all(np.isfinite(u)) and info["status"] == 0 and np.all(np.isfinite(p._i_rp))
    p.reset()
    assert p.n_fail == 0


def test_rl7d_never_exceeds_its_trained_caps_and_reports_the_flown_reference():
    with tempfile.TemporaryDirectory() as tmp:
        rl = _rl7d(tmp, spec_over={"attitude_cap_trained": [0.1, 0.2]})   # trained BELOW the station's caps
        rl.set_target_ned([0.0, 0.0, 0.8], 0.0)
        u, _ = rl.step(np.r_[0, 0, 0.8, 0, 0, 0], np.zeros(6), np.zeros(6), 0.0)
        assert abs(u[3]) <= 0.4 * 0.1 * 13.2 + 1e-12 and abs(u[4]) <= 0.4 * 0.2 * 7.2 + 1e-12
        rl.set_path_plan_ned(_plan1(rp=[0.3, -0.45]))
        _settle(rl)
        assert rl.ref_attitude_ned_at() == (pytest.approx(0.17), pytest.approx(-0.35))
    with tempfile.TemporaryDirectory() as tmp:
        spec = json.loads(FIXTURE_SPEC.read_text(encoding="utf-8"))
        ar = dict(spec["attitude_reference"])
        ar.pop("pitch_max_rad")
        with pytest.raises(ValueError, match="envelope is unknown"):
            _rl7d(tmp, spec_over={"attitude_reference": ar})


def test_resolve_policy_action_repr_and_rp_tracked_truth_tables():
    """The ONE rule the worker's ARM refusal, its ARM pin and the panel's
    forecast share (2026-10-03, policy.action_repr auto)."""
    from rov_gui.control.geometry import (LOW_RP_TRACK_BASE, LOW_RP_TRACK_SIBLING,
                                          policy_rp_tracked, resolve_policy_action_repr)
    r = resolve_policy_action_repr
    assert r("pos_yaw_width", "pos_rpy_width") == ("pos_yaw_width", "config")
    assert r("pos_rpy_width", "") == ("pos_rpy_width", "config")
    assert r("auto", "pos_rpy_width") == ("pos_rpy_width", "checkpoint")
    assert r("auto", "pos_yaw_width") == ("pos_yaw_width", "checkpoint")
    assert r("auto", "") == ("", "checkpoint")            # never defaulted
    assert r("auto", None) == ("", "checkpoint")
    assert r("auto", "pose10d") == ("pose10d", "checkpoint")   # the caller refuses it
    assert r(None, "pos_rpy_width") == ("pos_yaw_width", "config")  # code default
    for low in LOW_RP_TRACK_BASE:
        assert policy_rp_tracked(low, "pos_rpy_width", True), low
        assert not policy_rp_tracked(low, "pos_rpy_width", False), low
        assert not policy_rp_tracked(low, "pos_yaw_width", True), low
        assert not policy_rp_tracked(low, "auto", True), low     # a sentinel is no pin
    for low in ("none", "pid", "mpc", "mpc_tuned", "dobmpc", "dobmpc_tuned",
                "mpcc", "dobmpcc", "rl", "rl_pwm"):
        assert not policy_rp_tracked(low, "pos_rpy_width", True), low
    assert LOW_RP_TRACK_SIBLING == {"pid": "pid_7d", "mpc_tuned": "mpc_tuned_7d",
                                    "rl": "rl_7d", "rl_pwm": "rl_pwm_7d"}
