"""--state-noise (control/state_noise.py + MpcWorker, 2026-10-19).

The contract: noise reaches ctrl.step and nothing else — the truth `meas`
(plot, CSV px/py, interlocks) is never touched, the switch only moves while
disengaged, and every row says what it added (CSV sn_*)."""

from __future__ import annotations

import math
import tempfile

import numpy as np
import pytest

from rov_gui.control import state_noise as SN
from rov_gui.tests.test_control import (StubCtrl, _app, _feed_good_state,
                                        _last_csv_row, _test_opts)


# =============================================================================
# the module
# =============================================================================
def _meas():
    return {"eta": np.array([1.0, 2.0, 0.5, 0.01, -0.02, 0.3]),
            "nu": np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.01]),
            "nudot": np.zeros(6), "extra": "kept"}


def test_resolve_refuses_typos_negatives_and_over_the_ceiling():
    assert SN.resolve({}) == SN.DEFAULTS
    with pytest.raises(ValueError, match="unknown"):
        SN.resolve({"yaw_dg": 1.0})
    with pytest.raises(ValueError, match=">= 0"):
        SN.resolve({"pos_m": -0.01})
    with pytest.raises(ValueError, match="ceiling"):
        SN.resolve({"pos_m": 0.01}, scale=6.0)          # 0.06 > 0.05
    with pytest.raises(ValueError, match="bias"):
        SN.resolve({"tau_s": 1e3})
    assert SN.resolve({"pos_m": 0.01}, scale=5.0)["scale"] == 5.0


def test_apply_never_mutates_the_input_and_adds_exactly_what_it_reports():
    n = SN.StateNoise({"seed": 3}, dt=0.05)
    m = _meas()
    keep = {k: (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in m.items()}
    out, d = n.apply(m)
    for k in ("eta", "nu", "nudot"):
        assert np.array_equal(m[k], keep[k])          # truth untouched
        assert out[k] is not m[k]
    assert np.allclose(out["eta"] - m["eta"], d[0:6])
    assert np.allclose(out["nu"] - m["nu"], d[6:12])
    assert np.allclose(out["nudot"][0:3] - m["nudot"][0:3], d[12:15])
    assert np.array_equal(out["nudot"][3:6], m["nudot"][3:6])
    assert out["extra"] == "kept"
    assert np.all(d[12:15] == 0.0)                    # acc_m_s2 0 by default


def test_white_noise_has_the_configured_sigma_and_a_seed_reproduces_it():
    n = SN.StateNoise({"seed": 11, "scale": 2.0}, dt=0.05)
    ds = np.array([n.apply(_meas())[1] for _ in range(20000)])
    std = ds.std(axis=0)
    nz = n.sigma > 0
    assert np.allclose(std[nz], n.sigma[nz], rtol=0.03)
    assert abs(n.sigma[0] - 0.010) < 1e-12            # 2 x 5 mm
    assert abs(n.sigma[5] - math.radians(2.0)) < 1e-12
    lag1 = np.corrcoef(ds[:-1, 0], ds[1:, 0])[0, 1]
    assert abs(lag1) < 0.03                           # white per tick
    n.reset(11)
    again = np.array([n.apply(_meas())[1] for _ in range(50)])
    assert np.array_equal(again, ds[:50])


def test_gauss_markov_keeps_sigma_and_correlates_at_exp_minus_dt_over_tau():
    n = SN.StateNoise({"seed": 5, "tau_s": 1.0}, dt=0.05)
    ds = np.array([n.apply(_meas())[1] for _ in range(60000)])
    a = math.exp(-0.05 / 1.0)
    assert abs(n.a - a) < 1e-12
    lag1 = np.corrcoef(ds[:-1, 0], ds[1:, 0])[0, 1]
    assert abs(lag1 - a) < 0.01
    assert abs(ds[:, 0].std() / n.sigma[0] - 1.0) < 0.1


def test_scale_zero_adds_nothing():
    n = SN.StateNoise({"scale": 0.0}, dt=0.05)
    assert n.zero
    out, d = n.apply(_meas())
    assert not np.any(d) and np.array_equal(out["eta"], _meas()["eta"])


# =============================================================================
# the worker
# =============================================================================
class _RecCtrl(StubCtrl):
    """StubCtrl that keeps the eta / nu each step was handed."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.seen = []

    def step(self, eta, nu, nudot, t):
        self.seen.append((np.array(eta, float), np.array(nu, float)))
        return super().step(eta, nu, nudot, t)


def _worker(tmp, state_noise=None, policy=None):
    from rov_gui.bus import DataBus
    from rov_gui.control.workers import MpcWorker

    _app()
    bus = DataBus()
    Opts = _test_opts(tmp)
    Opts.state_noise = state_noise
    Opts.state_noise_policy = policy
    w = MpcWorker(bus, Opts(), controller_factory=_RecCtrl)
    w.setup()
    w.cfg.log_dir = str(tmp)
    w.cfg.engage["warmup_s"] = 0.1
    w.cfg.engage["settle_s"] = 0.0
    w.cfg.engage["mode_settle_s"] = 0.0
    logs = []
    bus.log.connect(lambda lvl, msg: logs.append((lvl, msg)))
    return w, bus, logs


def _engage(w):
    w.on_enable(True)
    _feed_good_state(w)
    w.set_engaged(True)
    assert w.engaged, w.reason


def test_columns_are_appended_last_and_the_schema_is_24():
    import rov_gui.control.workers as W

    cols = W.CSV_HEADER.strip().split(",")
    n = len(W.SN_CSV_COLUMNS)
    assert cols[-n:] == list(W.SN_CSV_COLUMNS) and len(cols) == len(set(cols))
    with tempfile.TemporaryDirectory() as tmp:
        w, _b, _l = _worker(tmp)
        try:
            assert w._run_meta()["schema_version"] == 24
        finally:
            w.teardown()


def test_without_the_flag_nothing_is_built_and_the_switch_refuses():
    with tempfile.TemporaryDirectory() as tmp:
        w, _b, _l = _worker(tmp)
        try:
            assert w.snoise is None
            w.set_state_noise(True)
            assert not w.snoise_on and "--state-noise" in w.reason
            _engage(w)
            for _ in range(3):
                _feed_good_state(w)
                w.tick()
            row = _last_csv_row(w)
            assert row["sn_on"] == "" and row["sn_dx"] == "nan"
            assert w._run_meta()["state_noise"]["enabled"] is False
            # no noise: the controller was handed the truth
            eta, _nu = w.ctrl.seen[-1]
            assert np.array_equal(eta, w._eta)
        finally:
            w.teardown()


def test_noise_reaches_ctrl_step_only_and_every_row_says_what_it_added():
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, _l = _worker(tmp, state_noise=2.0)
        try:
            assert w.snoise is not None and not w.snoise_on   # OFF at start
            st = []
            bus.mpc_status.connect(st.append)
            w.set_state_noise(True)
            assert w.snoise_on, w.reason
            _engage(w)
            m = w._run_meta()["state_noise"]
            assert m["enabled"] and m["at_engage"]["on"] is True
            assert isinstance(m["at_engage"]["seed"], int)
            assert m["config"]["scale"] == 2.0
            # the switch does not move while engaged
            w.set_state_noise(False)
            assert w.snoise_on and "engaged" in w.reason
            for _ in range(5):
                _feed_good_state(w)
                w.tick()
                d = w._sn_last
                assert d is not None and np.any(d[0:6] != 0.0)
                eta, nu = w.ctrl.seen[-1]
                # handed = truth + exactly the reported noise
                assert np.allclose(eta - w._eta, d[0:6], atol=1e-12)
                assert np.allclose(nu - w._last_nu, d[6:12], atol=1e-12)
                row = _last_csv_row(w)
                assert row["sn_on"] == "1"
                # CSV px is the TRUTH (datum frame: origin), sn_dx the noise
                assert abs(float(row["px"])) < 1e-9
                assert abs(float(row["sn_dx"]) - d[0]) < 1e-6
                assert abs(float(row["sn_dy"]) + d[1]) < 1e-6      # FLU
                assert abs(float(row["sn_dyaw_deg"])
                           + math.degrees(d[5])) < 1e-4
                assert abs(float(row["sn_du"]) - d[6]) < 1e-6      # FRD
            assert st and st[-1].sn_state == "on" and st[-1].sn_desc
            w.disengage("test")
            w.set_state_noise(False)
            assert not w.snoise_on, w.reason
        finally:
            w.teardown()


def test_switch_off_engagement_hands_the_truth_and_writes_sn_on_0():
    with tempfile.TemporaryDirectory() as tmp:
        w, _b, _l = _worker(tmp, state_noise=True)
        try:
            assert w.snoise is not None and w.snoise.cfg["scale"] == 1.0
            _engage(w)
            for _ in range(3):
                _feed_good_state(w)
                w.tick()
            assert w._sn_last is None
            eta, _nu = w.ctrl.seen[-1]
            assert np.array_equal(eta, w._eta)
            assert _last_csv_row(w)["sn_on"] == "0"
            assert w._run_meta()["state_noise"]["at_engage"] == {"on": False}
        finally:
            w.teardown()


def test_a_scale_over_the_ceiling_is_not_built_and_says_so():
    with tempfile.TemporaryDirectory() as tmp:
        w, _b, logs = _worker(tmp, state_noise=100.0)
        try:
            assert w.snoise is None and "ceiling" in w._sn_why
            w.set_state_noise(True)
            assert not w.snoise_on and "ceiling" in w.reason
        finally:
            w.teardown()


def test_observe_engagement_records_no_noise_in_meta(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        w, _b, _l = _worker(tmp, state_noise=True)
        try:
            w.set_state_noise(True)
            assert w.snoise_on
            # observe is read-only (LOW none); patched on the class for the test
            monkeypatch.setattr(type(w), "observe", property(lambda _s: True))
            _engage(w)
            assert w._sn_at_engage == {"on": False, "observe": True}
            assert w._sn_apply({"eta": np.zeros(6), "nu": np.zeros(6),
                                "nudot": np.zeros(6)})["eta"].tolist() == [0.0] * 6
        finally:
            w.teardown()


def test_a_raising_noise_source_disengages_instead_of_aborting_the_tick():
    with tempfile.TemporaryDirectory() as tmp:
        w, _b, _l = _worker(tmp, state_noise=True)
        try:
            w.set_state_noise(True)
            _engage(w)

            def boom(_m):
                raise RuntimeError("bad shape")
            w.snoise.apply = boom
            _feed_good_state(w)
            w.tick()
            assert not w.engaged and "state noise raised" in w.reason
        finally:
            w.teardown()



# =============================================================================
# the DP (--state-noise-policy, 2026-10-19)
# =============================================================================
def test_resolve_policy_is_a_strict_bool_and_off_by_default():
    assert SN.resolve({})["policy"] is False
    assert SN.resolve({"policy": True})["policy"] is True
    for bad in ("yes", 1, None):
        with pytest.raises(ValueError, match="policy"):
            SN.resolve({"policy": bad})
    assert "DP" in SN.StateNoise({"policy": True}, dt=0.05).describe()
    assert "DP" not in SN.StateNoise({}, dt=0.05).describe()


def _dp_worker(tmp, policy=True, noise_on=True):
    """Engaged on shape=policy with a 'present' policy worker reporting ready
    (test_policy's feed), the NOISE switch as asked BEFORE engage."""
    from rov_gui.tests.test_policy import _feed

    w, bus, _logs = _worker(tmp, state_noise=True, policy=policy)
    states, status = [], []
    bus.policy_state.connect(states.append)
    bus.mpc_status.connect(status.append)
    w.policy_present = True
    w.set_scenario({"shape": "policy"})
    if noise_on:
        w.set_state_noise(True)
        assert w.snoise_on, w.reason
    w.on_enable(True)
    _feed(w)
    w.set_engaged(True)
    assert w.engaged, w.reason
    for _ in range(4):
        _feed(w)
        w.tick()
    return w, states, status


def test_policy_noise_hands_the_dp_the_controllers_own_draw():
    from rov_gui.tests.test_policy import _feed

    with tempfile.TemporaryDirectory() as tmp:
        w, states, status = _dp_worker(tmp)
        try:
            assert w._sn_policy_cfg() and w._policy_eta_hist_truth is not None
            m = w._run_meta()["state_noise"]
            assert m["at_engage"]["policy"] is True and m["cli_policy"] is True
            assert m["config"]["policy"] is True and "DP" in m["describe"]
            assert status[-1].sn_policy is True and "DP" in status[-1].sn_desc
            for _ in range(5):
                _feed(w)
                w.tick()
                d = w._sn_last
                assert d is not None and np.any(d[0:6] != 0.0)
                s = states[-1]
                truth = np.asarray(w._eta, float)
                # the DP's pose = the truth + exactly the controller's draw...
                assert np.allclose(np.asarray(s.eta) - truth, d[0:6],
                                   atol=1e-12)
                # ...i.e. the very pose the controller was handed (tag mode)
                eta_ctrl, _nu = w.ctrl.seen[-1]
                assert np.allclose(np.asarray(s.eta), eta_ctrl, atol=1e-12)
                # the anchor history keeps the noisy row, its twin the truth
                assert w._policy_hist_last_fix == s.t_fix
                assert np.allclose(w._policy_eta_hist.interp(s.t_fix),
                                   s.eta, atol=1e-9)
                assert np.allclose(w._policy_eta_hist_truth.interp(s.t_fix),
                                   truth, atol=1e-9)
                # CSV px is still the truth (datum origin), sn_dx the draw
                row = _last_csv_row(w)
                assert abs(float(row["px"])) < 1e-9
                assert abs(float(row["sn_dx"]) - d[0]) < 1e-6
            assert len(w._policy_eta_hist) == len(w._policy_eta_hist_truth)
        finally:
            w.policy_present = False
            w.teardown()


def test_policy_noise_off_by_default_follows_the_switch_and_needs_the_flag():
    from rov_gui.tests.test_policy import _feed

    # --state-noise alone: the LOW is noised, the DP still sees the truth
    with tempfile.TemporaryDirectory() as tmp:
        w, states, status = _dp_worker(tmp, policy=None)
        try:
            assert not w._sn_policy_cfg()
            assert w._policy_eta_hist_truth is None
            assert w._sn_at_engage["on"] is True
            assert w._sn_at_engage["policy"] is False
            assert status[-1].sn_policy is False
            for _ in range(3):
                _feed(w)
                w.tick()
                assert w._sn_last is not None
                assert np.allclose(states[-1].eta, np.asarray(w._eta, float))
        finally:
            w.policy_present = False
            w.teardown()
    # the flag, but NOISE OFF at engage: nothing anywhere
    with tempfile.TemporaryDirectory() as tmp:
        w, states, _status = _dp_worker(tmp, noise_on=False)
        try:
            assert w._sn_at_engage == {"on": False}
            for _ in range(3):
                _feed(w)
                w.tick()
                assert w._sn_last is None and w._sn_policy_d() is None
                assert np.allclose(states[-1].eta, np.asarray(w._eta, float))
        finally:
            w.policy_present = False
            w.teardown()
    # the flag without --state-noise builds nothing, and says why
    with tempfile.TemporaryDirectory() as tmp:
        w, _b, _l = _worker(tmp, state_noise=None, policy=True)
        try:
            assert w.snoise is None and "--state-noise-policy" in w._sn_why
            w.set_state_noise(True)
            assert not w.snoise_on and "--state-noise-policy" in w.reason
        finally:
            w.teardown()


def test_policy_noise_reaches_the_start_pose_and_each_plan_says_what_it_carried():
    from rov_gui.tests.test_policy import _arm, _mk_plan, _plans_jsonl, _ticks

    with tempfile.TemporaryDirectory() as tmp:
        w, states, _status = _dp_worker(tmp)
        try:
            truth0 = np.asarray(w._eta, float).copy()
            d0 = np.asarray(w._sn_last, float).copy()
            _arm(w)
            start = np.asarray(w._policy_eta_start, float)
            assert np.allclose(start - truth0, d0[0:6], atol=1e-12)
            ps = w._run_meta()["state_noise"]["policy_starts"]
            assert len(ps) == 1
            assert np.allclose(ps[0]["d"][0:3], d0[0:3])
            assert np.allclose(ps[0]["d"][3:6], np.degrees(d0[3:6]))
            _ticks(w, 3)
            assert np.allclose(states[-1].eta_start, start)
            plan = _mk_plan(w)
            # the intake reads the histories as they are NOW (this tick's row
            # is appended at publish, after the intake)
            hn, ht = w._policy_eta_hist, w._policy_eta_hist_truth

            def want(ts):
                d = np.asarray(hn.interp(ts)) - np.asarray(ht.interp(ts))
                return np.concatenate([d[0:3], np.degrees(d[3:6])])
            exp_anchor = want(plan.obs_t)
            exp_rows = [want(t) for t in plan.obs_rows_t]
            w.on_policy_plan(plan)
            _ticks(w, 2)
            recs = [r for r in _plans_jsonl(w)
                    if r.get("plan_id") == plan.plan_id]
            assert recs, "the plan reached plans.jsonl"
            sp = recs[-1]["sn_policy"]
            assert "noisy - truth" in sp["frame"]
            assert np.allclose(sp["x_meas_d"], exp_anchor, atol=1e-9)
            assert np.allclose(sp["rows_d"], exp_rows, atol=1e-9)
            assert np.any(np.asarray(sp["x_meas_d"]) != 0.0)
        finally:
            w.policy_present = False
            w.teardown()


def test_without_policy_noise_plans_carry_no_sn_policy():
    from rov_gui.tests.test_policy import _arm, _mk_plan, _plans_jsonl, _ticks

    with tempfile.TemporaryDirectory() as tmp:
        w, _states, _status = _dp_worker(tmp, policy=None)
        try:
            _arm(w)
            assert w._run_meta()["state_noise"]["policy_starts"] == []
            _ticks(w, 3)
            plan = _mk_plan(w)
            w.on_policy_plan(plan)
            _ticks(w, 2)
            recs = [r for r in _plans_jsonl(w)
                    if r.get("plan_id") == plan.plan_id]
            assert recs and "sn_policy" not in recs[-1]
        finally:
            w.policy_present = False
            w.teardown()


def test_launch_check_refuses_the_flag_without_state_noise():
    from rov_gui.__main__ import build_parser, check_state_noise_policy

    p = build_parser()
    a = p.parse_args(["--mpc", "--source", "demo"])
    assert a.state_noise_policy is None and check_state_noise_policy(a) is None
    a = p.parse_args(["--mpc", "--source", "demo", "--state-noise-policy"])
    level, text = check_state_noise_policy(a)
    assert level == "refuse" and "--state-noise" in text
    a = p.parse_args(["--mpc", "--source", "demo", "--state-noise",
                      "--state-noise-policy"])
    assert a.state_noise is True and a.state_noise_policy is True
    level, text = check_state_noise_policy(a)
    assert level == "warn" and "--policy" in text
    a = p.parse_args(["--mpc", "--source", "demo", "--policy",
                      "--state-noise", "2", "--state-noise-policy"])
    level, text = check_state_noise_policy(a)
    assert level == "warn" and "RECORD BOUNDARY" in text



def test_the_dp_noise_is_a_cli_switch_never_a_yaml_key():
    with tempfile.TemporaryDirectory() as tmp:
        w, _b, _l = _worker(tmp, state_noise=True)
        try:
            w.cfg.state_noise = {"policy": True}       # a yaml left at true
            w._setup_state_noise()
            assert w.snoise is None and "--state-noise-policy" in w._sn_why
            w.opts.state_noise_policy = True           # ...even with the flag
            w._setup_state_noise()
            assert w.snoise is None and "not a yaml key" in w._sn_why
            w.cfg.state_noise = {}
            w._setup_state_noise()
            assert w.snoise is not None and w._sn_policy_cfg()
        finally:
            w.teardown()


def test_dp_noise_stops_at_disengage_and_the_chip_says_what_the_engagement_adds(
        monkeypatch):
    from rov_gui.state import MpcStatus

    with tempfile.TemporaryDirectory() as tmp:
        w, _states, _status = _dp_worker(tmp)
        try:
            assert w._sn_last is not None and w._sn_policy_d() is not None
            w.disengage("test")
            # the last tick's draw is still held, the DP gets none of it
            assert w._sn_last is not None and w._sn_policy_d() is None
            s = MpcStatus()
            w._publish_sn(s)
            assert s.sn_state == "on" and s.sn_policy is True   # process setting
            # an OBSERVE engagement adds nothing, so no "+DP"
            monkeypatch.setattr(type(w), "observe", property(lambda _s: True))
            _engage(w)
            assert w._sn_at_engage == {"on": False, "observe": True}
            s = MpcStatus()
            w._publish_sn(s)
            assert s.sn_policy is False
            assert w._sn_policy_d() is None
        finally:
            w.policy_present = False
            w.teardown()


def test_an_attitude_tracked_policy_with_dp_noise_is_refused_at_arm():
    from rov_gui.state import ACTION_REPR_POS_RPY_WIDTH
    from rov_gui.tests.test_control import _alloc_has_attitude, _att_worker
    from rov_gui.tests.test_policy import _arm, _feed7, validate_policy_block

    if not _alloc_has_attitude():
        pytest.skip("allocation without the attitude path (output group)")
    with tempfile.TemporaryDirectory() as tmp:
        w, _bus, _p, _l, _st = _att_worker(tmp, mode="mpc_tuned_7d")
        try:
            w.opts.state_noise = True
            w.opts.state_noise_policy = True
            w._setup_state_noise()
            assert w._sn_policy_cfg()
            w.cfg.policy.update({"action_repr": ACTION_REPR_POS_RPY_WIDTH,
                                 "attitude_track": True})
            validate_policy_block(w.cfg.policy)
            w.policy_present = True
            w.set_scenario({"shape": "policy"})

            def engage():
                w.on_enable(True)
                _feed7(w)
                w.set_engaged(True)
                assert w.engaged and w._attitude_axes is True, w.reason
                for _ in range(4):
                    _feed7(w)
                    w.tick()

            w.set_state_noise(True)
            engage()
            assert w._sn_at_engage.get("policy") is True
            w.set_traj(True)
            assert not w.traj_on and "--state-noise-policy" in w.reason, \
                w.reason
            # the same mission with NOISE OFF flies tracked: the refusal is
            # about the combination, nothing else
            w.disengage("test")
            w.set_state_noise(False)
            assert not w.snoise_on, w.reason
            engage()
            _arm(w)
            assert w.replay["attitude_track"] is True
        finally:
            w.policy_present = False
            w.teardown()


def test_under_dvl_pose_control_the_dp_and_the_controller_share_one_draw():
    """--dvl_pose control (safety review 2026-10-19, MEDIUM-1): the policy is
    told the FLOWN pose and the controller is handed the same pose — with
    --state-noise-policy both carry the one draw, added once."""
    from rov_gui.control import dvl_pose as DVP
    from rov_gui.tests import test_dvl_pose_worker as DPW

    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, _logs, _ev = DPW._worker(tmp)
        sim = DPW.PoseSim(w, TC)
        got = []
        bus.policy_state.connect(got.append)
        try:
            w.opts.state_noise = True
            w.opts.state_noise_policy = True
            w._setup_state_noise()
            w.set_state_noise(True)
            assert w.snoise_on and w._sn_policy_cfg(), w.reason
            DPW._ready(sim, w)
            sim.engage()
            assert w.dvp_control and w._sn_at_engage.get("policy") is True
            seen = []
            step0 = w.ctrl.step

            def rec(eta, nu, nudot, t):
                seen.append(np.array(eta, float))
                return step0(eta, nu, nudot, t)
            w.ctrl.step = rec
            w.policy_present = True
            n = 0
            for _ in range(30):
                k = len(seen)
                sim.step()
                d, flown = w._sn_last, w._flown_meas
                if d is None or flown is None or len(seen) == k:
                    continue
                assert w._dvp_last[1].get("state") in (DVP.ST_TAG,
                                                       DVP.ST_BLEND)
                n += 1
                pol = np.asarray(got[-1].eta, float)
                # the DP = the flown pose + the draw, once...
                assert np.allclose(pol - np.asarray(flown["eta"], float),
                                   d[0:6], atol=1e-12)
                # ...the very pose the controller was handed
                assert np.allclose(pol, seen[-1], atol=1e-12)
            assert n > 10
        finally:
            w.policy_present = False
            sim.close()


def test_launch_check_asks_for_scale_one_on_the_first_water_runs():
    from rov_gui.__main__ import build_parser, check_state_noise_policy

    p = build_parser()
    a = p.parse_args(["--mpc", "--source", "demo", "--policy",
                      "--state-noise", "--state-noise-policy"])
    level, text = check_state_noise_policy(a)
    assert level == "warn" and "SCALE" not in text and "land" in text
    a = p.parse_args(["--mpc", "--source", "demo", "--policy",
                      "--state-noise", "2", "--state-noise-policy"])
    level, text = check_state_noise_policy(a)
    assert level == "warn" and "SCALE 2 > 1" in text
