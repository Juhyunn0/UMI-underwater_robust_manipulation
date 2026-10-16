#!/usr/bin/env python3
"""test_rl_ekf.py — the robot_localization EKF port (control/rl_ekf.py) and
its place in the station (MpcWorker's EST switch, CSV, meta, replay).

What is pinned, in the order a failure should be read:

    port     the prediction Jacobian IS the derivative of r_l's model; the
             first measurement initializes only what it measures; a message
             older than the filter is corrected in WITHOUT prediction (r_l's
             no-smoothing behaviour — the thing the comparison is about);
             Mahalanobis gate; Joseph form; angle wrap
    node     per-topic stamp ordering; the timer integrates only what had
             ARRIVED; sensor-timeout prediction; "corrected data" is not
             published; smooth_lagged_data rewinds to the in-order answer;
             the DVL mount rotates the mask and the covariance and the lever
             arm uses the STATE's rate; no transform = no DVL; gravity
             removal has the NED sign; a repeated snapshot is fused once
    worker   shadow changes no command; the EST switch's refusals; a run on
             the EKF flies on it and ends when it diverges or faults; the
             CSV/meta/inputs log; the replay reproduces the live columns
    panel    the EST button reflects the worker and never flips itself

None of this is evidence about the real vehicle: every sensor number here
is synthetic, and robot_localization as the paper's estimator is an
inference (module docstring).

    QT_QPA_PLATFORM=offscreen \\
      ~/miniforge3/envs/robust/bin/python rov_gui/tests/test_rl_ekf.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np                                           # noqa: E402

from rov_gui.control import rl_ekf as EK                     # noqa: E402
from rov_gui.state import MpcStatus                          # noqa: E402

G = 9.80665


def _meas(t, topic, fields, values, var=1e-4, thresh=EK.DBL_MAX):
    z = np.zeros(EK.N)
    R = np.zeros((EK.N, EK.N))
    u = np.zeros(EK.N, bool)
    for f, v in zip(fields, values):
        i = EK.NAMES.index(f)
        z[i], R[i, i], u[i] = v, var, True
    return EK.Measurement(t, topic, z, R, u, thresh)


def _fix(p, yaw, tc, roll=0.0, pitch=0.0):
    R = EK.rot_zyx(roll, pitch, yaw)
    return SimpleNamespace(ok=True, t_capture=tc, p_ned=tuple(p),
                           R_ned_body=tuple(R.ravel()))


def _imu(t, roll=0.0, pitch=0.0, yaw=0.0, pqr=(0.0, 0.0, 0.0), f=None):
    if f is None:
        f = EK.rot_zyx(roll, pitch, yaw).T @ np.array([0.0, 0.0, -G])
    return SimpleNamespace(roll=roll, pitch=pitch, yaw=yaw, p=pqr[0],
                           q=pqr[1], r=pqr[2], ax=f[0], ay=f[1], az=f[2],
                           t_att=t, t_imu=t, stamp=t)


def _dvl(t, v, cov=None, valid=True):
    return SimpleNamespace(t_valid=t, v_dvl=tuple(v), valid=valid,
                           kind="velocity", covariance=cov)


# ======================================================================= port
def test_the_prediction_jacobian_is_the_derivative_of_the_model():
    """The hand-written Jacobian, against central differences of r_l's own
    transfer function — a transcription error anywhere in ekf.cpp's
    fourteen partial derivatives shows up here."""
    rng = np.random.default_rng(3)
    f = EK.Ekf()
    worst = 0.0
    for _ in range(30):
        x = rng.normal(size=EK.N)
        x[EK.ROLL] = rng.uniform(-1.0, 1.0)
        x[EK.PITCH] = rng.uniform(-1.2, 1.2)
        x[EK.YAW] = rng.uniform(-3.0, 3.0)
        d = rng.uniform(0.005, 0.1)

        def fn(xx):
            f.x, f.P = xx.copy(), np.eye(EK.N)
            f.predict(0.0, d)
            return f.x.copy()
        # the analytic J, read back through P = J e_i e_i^T J^T with Q = 0
        Q = f.Q
        f.Q = np.zeros((EK.N, EK.N))
        J = np.zeros((EK.N, EK.N))
        for i in range(EK.N):
            f.x = x.copy()
            f.P = np.zeros((EK.N, EK.N))
            f.P[i, i] = 1.0
            f.predict(0.0, d)
            k = int(np.argmax(np.diag(f.P)))
            J[:, i] = f.P[:, k] / math.sqrt(f.P[k, k]) if f.P[k, k] > 0 else 0
        f.Q = Q
        for i in range(EK.N):
            e = np.zeros(EK.N)
            e[i] = 1e-6
            dd = fn(x + e) - fn(x - e)
            for k in (EK.ROLL, EK.PITCH, EK.YAW):
                dd[k] = math.atan2(math.sin(dd[k]), math.cos(dd[k]))
            col = dd / 2e-6
            k = int(np.argmax(np.abs(col)))
            if np.sign(J[k, i]) != np.sign(col[k]):
                J[:, i] *= -1.0
            worst = max(worst, float(np.max(np.abs(J[:, i] - col))))
    assert worst < 1e-7, worst


def test_the_first_measurement_initializes_only_what_it_measures():
    f = EK.Ekf()
    res, _ = f.process(_meas(1.0, "t", ("x", "yaw"), (2.0, 0.5), var=4e-4))
    assert res == "init" and f.initialized and f.last_meas_time == 1.0
    assert f.x[EK.X] == 2.0 and f.x[EK.YAW] == 0.5 and f.x[EK.Y] == 0.0
    assert f.P[EK.X, EK.X] == 4e-4 and f.P[EK.Y, EK.Y] == 1e-9


def test_an_old_measurement_is_corrected_in_without_prediction():
    """r_l with smooth_lagged_data false: delta < 0 -> no predict, correct
    the CURRENT state anyway, and the filter time does not go back."""
    f = EK.Ekf()
    f.process(_meas(1.0, "t", ("x", "vx"), (0.0, 0.3)))
    f.process(_meas(2.0, "i", ("vx",), (0.3,)))      # predicted to t = 2
    x_before = f.x[EK.X]
    assert abs(x_before - 0.3) < 0.02, x_before
    res, delta = f.process(_meas(1.9, "t", ("x",), (0.27,)))   # 0.1 s late
    assert res is True and delta < 0
    assert f.last_meas_time == 2.0
    # pulled BACK toward where the vehicle was 0.1 s ago — the lag
    assert f.x[EK.X] < x_before


def test_the_mahalanobis_gate_rejects_beyond_its_threshold_only():
    f = EK.Ekf()
    f.process(_meas(1.0, "t", ("x",), (0.0,)))
    x0 = f.x.copy()
    assert f.process(_meas(1.0, "t", ("x",), (5.0,), thresh=3.0))[0] is False
    assert np.array_equal(f.x, x0)
    # r_l's default threshold is the numeric maximum: nothing is rejected
    assert f.process(_meas(1.0, "t", ("x",), (5.0,)))[0] is True


def test_joseph_form_keeps_the_covariance_symmetric_and_positive():
    rng = np.random.default_rng(1)
    f = EK.Ekf()
    t = 1.0
    f.process(_meas(t, "a", ("x", "y", "yaw"), (0, 0, 0)))
    for k in range(400):
        t += 0.02
        flds = [("x", "y", "yaw"), ("vx", "vy"), ("roll", "pitch", "vyaw"),
                ("ax", "ay", "az")][k % 4]
        f.process(_meas(t, "a", flds, rng.normal(size=len(flds)) * 0.1,
                        var=rng.uniform(1e-8, 1e-2)))
    assert np.max(np.abs(f.P - f.P.T)) < 1e-12
    assert np.min(np.linalg.eigvalsh(0.5 * (f.P + f.P.T))) > -1e-12


def test_an_angle_innovation_takes_the_short_way_round():
    f = EK.Ekf()
    f.process(_meas(1.0, "t", ("yaw",), (math.pi - 0.05,)))
    f.process(_meas(1.0, "t", ("yaw",), (-math.pi + 0.05,), var=1e-4))
    y = f.x[EK.YAW]
    assert abs(abs(y) - math.pi) < 0.06, y    # near +-pi, not near 0


# ======================================================================= node
def _node(**cfg):
    return EK.RlEkfNode(cfg, z_source="pressure")


def test_a_late_message_on_its_own_topic_is_dropped_and_no_other():
    n = _node()
    n.feed_tag(_fix((0, 0, 0.5), 0.0, 10.0), 10.0)
    n.feed_tag(_fix((0, 0, 0.5), 0.0, 9.9), 10.01)   # older than the last
    n.feed_imu(_imu(9.95), 10.02)                     # another topic: fine
    n.advance(10.1)
    assert n.n[EK.TOPIC_TAG]["order_drop"] == 1
    assert n.n[EK.TOPIC_TAG]["used"] == 1
    assert n.n[EK.TOPIC_IMU_POSE]["used"] == 1


def test_the_timer_integrates_only_what_had_arrived_and_coasts_on_timeout():
    n = _node()
    n.feed_tag(_fix((1.0, 2.0, 0.5), 0.3, 10.0), 10.0)
    n.advance(10.0)                       # first cycle: phase starts here
    assert n.output()["stamp"] == 10.0
    # stamped 10.01 but ARRIVES at 10.05: the 10.02 and 10.04 cycles cannot
    # have seen it (a ROS callback had not run yet)
    n.feed_imu(_imu(10.01), 10.05)
    n.advance(10.04)
    assert n.n[EK.TOPIC_IMU_POSE]["used"] == 0
    # ...but with nothing queued for >= sensor_timeout they PREDICT to now
    assert n.n_timeout_predicts >= 1
    assert abs(n.output()["stamp"] - 10.04) < 1e-9
    n.advance(10.06)
    assert n.n[EK.TOPIC_IMU_POSE]["used"] == 1


def test_a_correction_that_does_not_move_the_stamp_is_not_published():
    """permit_corrected_publication false (r_l default): a lagged message
    corrects the state, but the published odometry keeps the previous
    message until the stamp advances."""
    n = _node(sensor_timeout_s=10.0)
    n.feed_imu(_imu(10.0), 10.0)
    n.feed_tag(_fix((0, 0, 0.5), 0.0, 10.0), 10.0)
    n.advance(10.0)
    n.feed_imu(_imu(10.05), 10.05)
    n.advance(10.06)
    pub = n.output()
    n.feed_tag(_fix((0.05, 0, 0.5), 0.0, 10.02), 10.07)   # lagged
    n.advance(10.08)
    assert n.n_corrected_suppressed >= 1
    assert n.output() is pub                 # the old message stands
    assert n.ekf.x[EK.X] > 0.0               # ...though the state moved


def test_smoothing_rewinds_to_the_in_order_answer():
    """smooth_lagged_data: a tag that arrives after later IMU messages is
    re-fused in stamp order — the same answer as if it had arrived on time.
    (sensor timeout long, so neither node coasts between messages.)"""
    msgs = []
    for k in range(10):
        msgs.append(("imu", 10.0 + 0.02 * k))
    msgs.append(("tag", 10.05))
    res = []
    for late in (False, True):
        n = _node(smooth_lagged_data=True, history_length_s=1.0,
                  sensor_timeout_s=10.0)
        n.feed_tag(_fix((0, 0, 0.5), 0.0, 9.99), 9.99)
        for kind, ts in msgs:
            arr = ts + (0.12 if (late and kind == "tag") else 0.0)
            if kind == "imu":
                n.feed_imu(_imu(ts, pqr=(0.0, 0.0, 0.1)), arr)
            else:
                n.feed_tag(_fix((0.02, 0.01, 0.5), 0.004, ts), arr)
        n._inbox.sort(key=lambda e: e[0])
        n.advance(9.99)
        for k in range(1, 40):
            n.advance(9.99 + 0.005 * k)
        res.append((n.ekf.x.copy(), n.n_reverts))
    assert res[1][1] >= 1, "the late tag did not trigger a rewind"
    assert np.max(np.abs(res[0][0] - res[1][0])) < 1e-9


def test_the_dvl_mount_rotates_the_mask_and_the_lever_arm_uses_the_state():
    R = EK.rot_zyx(0.0, 0.0, math.pi / 2)         # DVL x = body y
    r = np.array([0.2, 0.0, 0.0])
    n = EK.RlEkfNode({"dvl_fuse": ["vx"]}, dvl_mount=(R, r, True),
                     z_source="pressure")
    n.ekf.initialized = True
    n.ekf.x[EK.VYAW] = 0.5                         # the STATE's yaw rate
    n.ekf.last_meas_time = 10.0
    cov = (4e-6, 0, 0, 0, 9e-6, 0, 0, 0, 1e-6)
    n._deliver("dvl", (10.0, (0.1, 0.0, 0.0), cov))
    m = n._queue[0][2]
    # vx in the DVL frame is vy in the body: the mask follows the rotation
    assert not m.update[EK.VX] and m.update[EK.VY] and not m.update[EK.VZ]
    # v = R v_dvl + r x w_state = (0, 0.1, 0) + (0.2,0,0) x (0,0,0.5)
    assert np.allclose(m.z[EK.VX:EK.VZ + 1], [0.0, 0.1 - 0.1, 0.0])
    assert abs(m.R[EK.VY, EK.VY] - 4e-6) < 1e-15   # rotated covariance


def test_an_unmeasured_dvl_mount_means_no_dvl():
    n = EK.RlEkfNode({}, dvl_mount=(np.eye(3), np.zeros(3), False))
    n.feed_tag(_fix((0, 0, 0.5), 0.0, 10.0), 10.0)
    n.feed_dvl(_dvl(10.0, (0.1, 0, 0)), 10.0)
    n.advance(10.0)
    assert n.n_dvl_no_tf == 1 and n.n[EK.TOPIC_DVL]["in"] == 0


def test_gravity_is_removed_with_the_ned_sign():
    """A stationary vehicle at any attitude reads R^T (0, 0, -g) in FRD;
    after removal the acceleration measurement must be zero."""
    for roll, pitch in ((0.0, 0.0), (0.5, -0.3)):
        n = _node()
        n._deliver("imu", (10.0, roll, pitch, 1.0, (0.0, 0.0, 0.0),
                           tuple(EK.rot_zyx(roll, pitch, 1.0).T
                                 @ np.array([0.0, 0.0, -G]))))
        acc = [m for _t, _s, m in n._queue
               if m.topic == EK.TOPIC_IMU_ACC][0]
        assert np.max(np.abs(acc.z[EK.AX:EK.AZ + 1])) < 1e-12


def test_a_repeated_snapshot_is_fused_once():
    n = _node()
    fx = _fix((0, 0, 0.5), 0.0, 10.0)
    im = _imu(10.0)
    for k in range(3):                     # the station re-delivers objects
        n.feed_tag(fx, 10.0 + 0.001 * k)
        n.feed_imu(im, 10.0 + 0.001 * k)
    n.advance(10.01)
    assert n.n[EK.TOPIC_TAG]["used"] == 1
    assert n.n[EK.TOPIC_IMU_POSE]["used"] == 1


def test_the_config_refuses_typos_and_takes_z_from_the_tag_when_told():
    try:
        EK.resolve({"smooth_laged_data": True})
    except ValueError as e:
        assert "smooth_laged_data" in str(e)
    else:
        raise AssertionError("a typo was accepted")
    assert "z" not in EK.resolve({})["tag_fuse"]
    assert "z" in EK.resolve({}, z_source="tag")["tag_fuse"]
    assert EK.resolve({})["sensor_timeout_s"] == 1.0 / 50.0
    for bad in ({"dvl_fuse": ["x"]}, {"frequency_hz": 0},
                {"process_noise_diag": [1.0] * 14}):
        try:
            EK.resolve(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_the_controller_gets_d_nu_dt_not_the_inertial_acceleration():
    x = np.zeros(EK.N)
    x[EK.VX], x[EK.VYAW], x[EK.AY] = 0.2, 0.5, 0.1
    m = EK.to_meas({"x": x, "P": np.eye(EK.N), "stamp": 0.0, "t_cycle": 0.0,
                    "ang_acc": np.zeros(3)})
    # a = nudot + w x v  ->  nudot_y = 0.1 - (0.5 * 0.2) = 0
    assert abs(m["nudot"][1]) < 1e-12 and m["nu"][0] == 0.2


def test_a_vehicle_at_rest_converges_on_its_pose():
    n = _node()
    p, yaw = (1.5, -0.7, 0.9), 2.5
    t = 100.0
    for k in range(250):                 # 5 s
        t = 100.0 + 0.02 * k
        if k % 3 == 0:
            n.feed_tag(_fix(p, yaw, t - 0.12), t)
        n.feed_imu(_imu(t, yaw=yaw + 0.4), t)          # compass off: unfused
        n.feed_depth(p[2], t, t)
        n.advance(t)
    x = n.output()["x"]
    assert np.allclose(x[0:3], p, atol=2e-3), x[0:3]
    assert abs(x[EK.YAW] - yaw) < 1e-3
    assert np.max(np.abs(x[EK.VX:EK.VZ + 1])) < 5e-3
    # r_l's DEFAULT process noise (0.05 m^2/s on x and y) grows the
    # horizontal variance by 1e-3 m^2 every 20 ms cycle, so between tag
    # frames the filter's own 1-sigma is centimetres even at rest — and
    # each tag frame (1 cm) is taken almost whole.
    sig = EK.sigma(n.output(), (EK.X, EK.Y))
    assert 0.01 < sig < 0.1, sig


# ===================================================================== worker
def _td():
    from rov_gui.tests import test_dvl as TD
    return TD


def test_the_test_worker_builds_the_ekf_in_shadow():
    TD = _td()
    TC = TD._harness()
    with tempfile.TemporaryDirectory() as tmp:
        w, _bus, _p, _l = TC._worker(tmp)
        try:
            assert w.ekf is not None and w.ekf_control is False
            assert w.ekf.dvl_R is None          # no DVL built
        finally:
            w.teardown()


def test_a_shadow_ekf_does_not_change_one_command():
    TD = _td()
    out = []
    for with_ekf in (False, True):
        with tempfile.TemporaryDirectory() as tmp:
            TC = TD._harness()
            w, _bus, pilots, _logs = TC._worker(tmp)
            if not with_ekf:
                w.cfg.ekf = {"enabled": False}
                w._setup_ekf()
                assert w.ekf is None
            sim = TD._Sim(w, TC)
            try:
                sim.engage()
                for k in range(30):
                    sim.step(v=(0.2 * math.sin(0.3 * k), 0.04, 0.0))
                if with_ekf:
                    rows = TD._rows(w)
                    assert rows and all(r["est"] == "tag" for r in rows)
                    assert all(r["ekf_px"] != "nan" for r in rows[3:])
                out.append([(q.surge, q.sway, q.heave, q.yaw)
                            for q in pilots])
            finally:
                sim.close()
    assert out[0] and out[0] == out[1], "a shadow EKF perturbed the commands"


def test_the_estimator_switch_refuses_until_it_may_and_freezes_in_flight():
    TD = _td()
    TC = TD._harness()
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, _p, logs = TC._worker(tmp)
        sim = TD._Sim(w, TC)
        try:
            w.set_estimator("ekf")            # no measurement yet
            assert not w.ekf_control and "not ready" in w.reason, w.reason
            for _ in range(10):
                sim.step()
            w.set_estimator("ekf")
            assert w.ekf_control and "EKF" in w.reason, w.reason
            sim.engage()
            w.set_estimator("tag")            # one run = one estimator
            assert w.ekf_control and "engaged" in w.reason
            w.set_engaged(False)
            w._open_csv(Path(tmp) / "rec_mpc", auto=False)
            w.set_estimator("tag")
            assert w.ekf_control and "CSV" in w.reason
            w._close_csv()
            w.set_estimator("tag")
            assert not w.ekf_control
            # no EKF at all
            w.cfg.ekf = {"enabled": False}
            w._setup_ekf()
            w.set_estimator("ekf")
            assert not w.ekf_control and "no EKF" in w.reason
        finally:
            sim.close()


def test_a_run_on_the_ekf_flies_on_it_and_records_that_it_did():
    TD = _td()
    TC = TD._harness()
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, _p, _l = TC._worker(tmp)
        sim = TD._Sim(w, TC)
        try:
            for _ in range(10):
                sim.step()
            w.set_estimator("ekf")
            sim.engage()
            st = []
            bus.mpc_status.connect(st.append)
            for _ in range(8):
                sim.step(v=(0.05, 0.0, 0.0))
            assert w.engaged
            assert st[-1].ekf_control and st[-1].ekf_flew
            assert st[-1].ekf_state == "ready" and st[-1].p_ekf_flu
            m = w._run_meta()
            assert m["schema_version"] == 22
            assert m["ekf"]["estimator_at_engage"] == "ekf"
            assert m["ekf"]["ready_at_engage"] is True
            assert m["ekf"]["node"]["source"] == EK.RL_SOURCE
            stem = w._csv_path.with_suffix("")
            rows = TD._rows(w)
            from rov_gui.control.workers import CSV_HEADER
            assert all(len(r) == len(CSV_HEADER.strip().split(","))
                       for r in rows)
            assert rows and all(r["est"] == "ekf" for r in rows)
            assert m["ekf"]["rows"].get("ekf", 0) >= 8
            # the inputs log: header, the engage datum, and the inputs
            log = Path(f"{stem}_ekf_inputs.jsonl")
            lines = [json.loads(x) for x in log.read_text().splitlines()]
            assert lines[0]["k"] == "header"
            kinds = {x["k"] for x in lines}
            assert {"fix", "imu", "depth"} <= kinds
        finally:
            sim.close()


def test_a_run_on_the_ekf_ends_when_it_walks_away_from_the_tag_state():
    TD = _td()
    TC = TD._harness()
    with tempfile.TemporaryDirectory() as tmp:
        w, _bus, _p, _l = TC._worker(tmp)
        sim = TD._Sim(w, TC)
        try:
            for _ in range(10):
                sim.step()
            w.set_estimator("ekf")
            sim.engage()
            for _ in range(3):
                sim.step()
            assert w.engaged
            # the EKF now rejects every tag (Mahalanobis 3 sigma) while the
            # tag state jumps 0.5 m: the flown state and the truth part
            w.ekf.cfg["tag_rejection_threshold"] = 3.0
            sim.p = sim.p + np.array([0.5, 0.0, 0.0])
            for _ in range(10):
                sim.step()
                if not w.engaged:
                    break
            assert not w.engaged
            assert "diverged" in w.reason, w.reason
        finally:
            sim.close()


def test_a_fault_in_the_ekf_ends_a_run_on_it_and_not_a_tag_run():
    TD = _td()
    for fly_ekf in (True, False):
        TC = TD._harness()
        with tempfile.TemporaryDirectory() as tmp:
            w, _bus, _p, logs = TC._worker(tmp)
            sim = TD._Sim(w, TC)
            try:
                for _ in range(10):
                    sim.step()
                if fly_ekf:
                    w.set_estimator("ekf")
                sim.engage()

                def boom(t):
                    raise RuntimeError("synthetic")
                w.ekf.advance = boom
                for _ in range(3):
                    sim.step()
                assert w._ekf_fault and any("CODE FAULT" in m
                                            for _l, m in logs)
                assert w.engaged is (not fly_ekf), w.reason
                if fly_ekf:
                    assert "EKF" in w.reason
            finally:
                sim.close()


def test_the_replay_reproduces_the_live_columns():
    """rov_gui.tools.ekf_replay on a recorded run, live config: the replay
    starts cold at the CSV's first row (the live node was warm), so it is
    compared once both have converged."""
    TD = _td()
    TC = TD._harness()
    from rov_gui.tools import ekf_replay as RP
    with tempfile.TemporaryDirectory() as tmp:
        w, _bus, _p, _l = TC._worker(tmp)
        sim = TD._Sim(w, TC)
        try:
            for _ in range(10):
                sim.step()
            sim.engage()
            for k in range(160):
                sim.step(v=(0.1 * math.sin(0.1 * k), 0.05, 0.0))
            path = w._csv_path
            w._close_csv()
            rows, meta = RP.replay(path, {})
            late = [r for r in rows if "flu" in r][80:]
            d = [math.hypot(r["flu"][0] - float(r["row"]["ekf_px"]),
                            r["flu"][1] - float(r["row"]["ekf_py"]))
                 for r in late]
            assert late and max(d) < 1e-3, max(d)
            assert meta["counts"][EK.TOPIC_TAG]["used"] > 100
            # and a variant runs (smoothing on)
            rows2, meta2 = RP.replay(path, {"smooth_lagged_data": True,
                                            "history_length_s": 1.0})
            assert any("flu" in r for r in rows2)
        finally:
            sim.close()


def test_a_frozen_accelerometer_is_not_fused_as_new():
    """A dead SCALED_IMU2 under a live ATTITUDE (safety review 2026-10-02):
    the old specific force must not ride on every new message."""
    n = _node()
    for k in range(5):
        t = 10.0 + 0.04 * k
        im = _imu(t)
        im.t_imu = 10.0                    # the accelerometer stopped at 10.0
        n.feed_imu(im, t)
    n.advance(10.2)
    assert n.n[EK.TOPIC_IMU_ACC]["used"] <= int(0.3 / 0.04) + 1
    assert n.n[EK.TOPIC_IMU_POSE]["used"] == 5
    assert 10.9 - n.last_used_t[EK.TOPIC_IMU_ACC] > 0.3
    assert 10.9 - n.last_used_t[EK.TOPIC_IMU_POSE] < 0.8


def test_freshness_is_what_was_fused_not_the_published_stamp():
    """r_l predicts to now with nothing queued, so its stamp never ages;
    the station's gate asks per input."""
    n = _node()
    n.feed_tag(_fix((0, 0, 0.5), 0.0, 10.0), 10.0)
    n.feed_imu(_imu(10.0), 10.0)
    n.feed_depth(0.5, 10.0, 10.0)
    n.advance(10.0)
    assert n.stale_inputs(10.0, 0.7, 0.3) == ""
    for k in range(1, 60):                 # 1.2 s of silence
        n.advance(10.0 + 0.02 * k)
    assert 10.0 + 1.16 - n.output()["stamp"] < 0.03     # looks fresh...
    why = n.stale_inputs(10.0 + 1.18, 0.7, 0.3)
    assert "fused for" in why, why                      # ...but is not


def test_a_missing_compass_is_not_offered_and_cannot_poison_the_state():
    n = EK.RlEkfNode({"imu_fuse": ["roll", "pitch", "yaw"]})
    im = _imu(10.0)
    im.yaw = None
    n.feed_imu(im, 10.0)
    n.advance(10.0)
    assert EK.finite(n.output())


def test_an_interleaved_redelivery_is_still_fused_once():
    n = _node()
    a, b = _imu(10.0), _imu(10.04)
    for k, im in enumerate((a, b, a, b)):
        n.feed_imu(im, 10.04 + 0.001 * k)
    n.advance(10.1)
    assert n.n[EK.TOPIC_IMU_POSE]["used"] == 2


def test_a_tag_dropout_in_station_hold_ends_a_run_on_the_ekf():
    """The critical finding of the 2026-10-02 safety review: the station
    bridge must not carry an EKF run (uncorrected integration, a coast
    retarget writing its drift into the target). The first bridged tick
    ends it and steps nothing."""
    TD = _td()
    TC = TD._harness()
    with tempfile.TemporaryDirectory() as tmp:
        w, _bus, pilots, _l = TC._worker(tmp)
        w.cfg.engage["settle_s"] = 0.0
        sim = TD._Sim(w, TC)
        try:
            for _ in range(10):
                sim.step()
            w.set_estimator("ekf")
            assert w.ekf_control, w.reason
            sim.engage()
            w.set_scenario({"shape": "station"})
            w.set_traj(True)
            for _ in range(3):
                sim.step()
            assert w.engaged and w.station is not None
            stepped = []
            orig = w.ctrl.step

            def spy(*a, **k):
                stepped.append(w._bridge.tier if w._bridge else None)
                return orig(*a, **k)
            w.ctrl.step = spy
            for _ in range(40):            # the tags go away
                sim.step(fix=False)
                if not w.engaged:
                    break
            assert not w.engaged
            assert "station bridge does not carry an EKF run" in w.reason \
                or "no tag pose fused" in w.reason, w.reason
            assert all(t in (None, "none") for t in stepped), stepped
        finally:
            sim.close()


def test_a_stale_input_ends_a_run_on_the_ekf():
    """The accelerometer stops (SCALED_IMU2 dies, ATTITUDE lives): the
    assembler does not notice, the EKF's freshness gate does."""
    TD = _td()
    TC = TD._harness()
    import rov_gui.control.workers as W
    with tempfile.TemporaryDirectory() as tmp:
        w, _bus, _p, _l = TC._worker(tmp)
        sim = TD._Sim(w, TC)
        try:
            for _ in range(10):
                sim.step()
            w.set_estimator("ekf")
            sim.engage()
            for _ in range(3):
                sim.step()
            assert w.engaged
            t_dead = sim.t
            real_imu = TC._imu

            def dead_acc(**kw):
                im = real_imu(**kw)
                im.t_imu = t_dead
                return im
            TC._imu = dead_acc
            try:
                # accelerometer limit 1.0 s (2-3 Hz SCALED_IMU2 must not flap
                # the gate) + the 0.4 s hold
                for _ in range(60):
                    sim.step()
                    if not w.engaged:
                        break
            finally:
                TC._imu = real_imu
            assert not w.engaged and "accelerometer" in w.reason, w.reason
            assert W is not None
        finally:
            sim.close()


def test_the_ekf_takes_the_dvl_only_while_the_dvl_switch_is_in():
    TD = _td()
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, _bus, _p, _l = TD._dvl_worker(tmp, reentry_s=0.0)
        w._setup_ekf()     # the station builds the DVL first; the harness
        sim = TD._Sim(w, TC)        # adds it afterwards
        try:
            assert w.ekf is not None and w.ekf.dvl_R is not None
            for _ in range(20):
                sim.step(v=(0.03, 0.0, 0.0))
            assert w.ekf.n[EK.TOPIC_DVL]["in"] == 0     # OUT: not given
            sim.prove()
            w.set_dvl_use(True)
            assert w.dvl.use, w.reason
            for _ in range(10):
                sim.step()
            assert w.ekf.n[EK.TOPIC_DVL]["used"] > 0    # IN: fused
        finally:
            sim.close()


# LEGACY-PINNED (2026-10-16, SPEC §10.2b): the same, under dvl.selection legacy.
def test_the_ekf_takes_the_dvl_only_while_the_dvl_switch_is_in_under_legacy():
    TD = _td()
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, _bus, _p, _l = TD._dvl_worker(tmp, reentry_s=0.0,
                                             selection="legacy")
        w._setup_ekf()
        sim = TD._Sim(w, TC)
        try:
            assert w.ekf is not None and w.ekf.dvl_R is not None
            for _ in range(20):
                sim.step(v=(0.03, 0.0, 0.0))
            assert w.ekf.n[EK.TOPIC_DVL]["in"] == 0     # OUT: not given
            sim.prove()
            w.set_dvl_use(True)
            assert w.dvl.use, w.reason
            for _ in range(10):
                sim.step()
            assert w.ekf.n[EK.TOPIC_DVL]["used"] > 0    # IN: fused
        finally:
            sim.close()


def _spy_feed(w):
    """Record which DVL reports the worker hands the EKF (t_valid)."""
    fed = []
    real = w.ekf.feed_dvl

    def spy(sample, t):
        fed.append(float(sample.t_valid))
        return real(sample, t)
    w.ekf.feed_dvl = spy
    return fed


def test_the_ekf_takes_only_a_usable_dvl():
    """N36 (2026-10-16, selection primary). The EKF fuses the DVL with the
    device's ~1 mm/s covariance, so a fused wrong DVL dominates it: it is
    given a report only while the blend may use it — not during a
    disagreement hold, not a flagged report — instead of until a trip."""
    TD = _td()
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, _bus, _p, _l = TD._dvl_worker(tmp, reentry_s=0.0)
        w._setup_ekf()
        sim = TD._Sim(w, TC)
        try:
            assert w.dvl.selection == "primary"
            sim.prove()
            w.set_dvl_use(True)
            fed = _spy_feed(w)
            for _ in range(10):
                sim.step(v=(0.10, 0.0, 0.0))
            assert len(fed) == 10                     # agreeing: all of them
            # (b) a FLAGGED report (fom 4 mm/s) is not given, the next is
            sim.step(v=(0.10, 0.0, 0.0), dvl=False, tick=False)
            w.on_dvl(TD._smp((0.10, 0.0, 0.0), sim.t, fom=0.004))
            w.tick()
            assert w.dvl.usable(sim.t)[2] == "flag_fom"
            assert fed[-1] != sim.t
            sim.step(v=(0.10, 0.0, 0.0))
            assert fed[-1] == sim.t
            # (a) a disagreement hold: the reversed reading is not given
            # while the hold lasts
            held = []
            for _ in range(60):
                n0 = len(fed)
                sim.step(v=(0.10, 0.0, 0.0), dvl_v=(-0.10, 0.0, 0.0))
                if w.dvl.usable(sim.t)[2].startswith("dis_"):
                    held.append(len(fed) - n0)
            assert held and not any(held), held
        finally:
            sim.close()


def test_the_ekf_velocity_gate_reads_the_tag_difference():
    """N36 (c). Under primary nu[0:2] IS the DVL blend: the EKF's velocity
    divergence (the interlock of a run on the EKF) is measured against the
    tag finite difference, as its own comment says — an EKF that fused a
    wrong DVL would otherwise be compared with that same DVL. Legacy keeps
    the shipped reference (meas["nu"])."""
    TD = _td()
    for sel in ("primary", "legacy"):
        with tempfile.TemporaryDirectory() as tmp:
            TC, w, _bus, _p, _l = TD._dvl_worker(tmp, reentry_s=0.0,
                                                 selection=sel)
            w._setup_ekf()
            sim = TD._Sim(w, TC)
            try:
                sim.prove()
                w.set_dvl_use(True)
                sim.engage()
                for _ in range(30):           # DVL 3 cm/s faster: inside the
                    sim.step(v=(0.05, 0.0, 0.0), dvl_v=(0.08, 0.0, 0.0))
                m, h = w._ekf_meas, w._last_health
                assert m is not None and w._ekf_err is not None
                assert abs(float(w._last_nu[0]) - 0.08) < 1e-9, w._last_nu
                ref = (np.asarray(h["nu_fd"], float)[:3] if sel == "primary"
                       else np.asarray(w._last_nu, float)[:3])
                assert np.allclose(w._ekf_err["dv"],
                                   np.asarray(m["nu"][:3], float) - ref), sel
                assert abs(float(h["nu_fd"][0]) - 0.05) < 0.01
            finally:
                sim.close()


def test_a_run_on_the_ekf_ends_when_its_velocity_walks_away():
    """An accelerometer bias the mavros noise trusts completely: x/y stay
    tag-pinned, the velocity does not — the velocity gate ends the run."""
    TD = _td()
    TC = TD._harness()
    with tempfile.TemporaryDirectory() as tmp:
        w, _bus, _p, _l = TC._worker(tmp)
        sim = TD._Sim(w, TC)
        try:
            for _ in range(10):
                sim.step()
            w.set_estimator("ekf")
            sim.engage()
            for _ in range(60):
                sim.step(f=(1.5, 0.0, -9.80665))
                if not w.engaged:
                    break
            assert not w.engaged and "vel" in w.reason, w.reason
        finally:
            sim.close()


def test_switching_in_needs_the_ekf_and_the_tag_state_to_agree():
    TD = _td()
    TC = TD._harness()
    with tempfile.TemporaryDirectory() as tmp:
        w, _bus, _p, _l = TC._worker(tmp)
        sim = TD._Sim(w, TC)
        try:
            for _ in range(10):
                sim.step()
            w._ekf_err = dict(w._ekf_err, xy=0.08)
            w.set_estimator("ekf")
            assert not w.ekf_control and "disagree" in w.reason, w.reason
            sim.step()                       # recomputed: they agree
            w.set_estimator("ekf")
            assert w.ekf_control, w.reason
        finally:
            sim.close()


def test_a_short_tag_gap_holds_an_ekf_run_without_commanding_and_resumes():
    """The re-audit's N1: staleness and bridged ticks are DEBOUNCED like the
    tag path's stale gate (tag_stale_hold_s), commanding nothing meanwhile —
    a tag back inside the window resumes the run on the EKF."""
    TD = _td()
    TC = TD._harness()
    with tempfile.TemporaryDirectory() as tmp:
        w, _bus, _p, _l = TC._worker(tmp)
        w.cfg.engage["settle_s"] = 0.0
        sim = TD._Sim(w, TC)
        try:
            for _ in range(10):
                sim.step()
            w.set_estimator("ekf")
            sim.engage()
            w.set_scenario({"shape": "station"})
            w.set_traj(True)
            for _ in range(3):
                sim.step()
            stepped, held = [], []
            orig = w.ctrl.step

            def spy(*a, **k):
                stepped.append(sim.t)
                return orig(*a, **k)
            w.ctrl.step = spy
            for _ in range(12):            # 0.6 s gap: past tag_stale_s 0.5
                sim.step(fix=False)
                held.append(w._ekf_hold)
            assert w.engaged, w.reason
            assert any(held), "the gap never reached the EKF gate"
            n = len(stepped)
            for _ in range(5):             # the tags come back
                sim.step()
            assert w.engaged and len(stepped) > n and w._ekf_flew
            assert not w._ekf_hold
            rows = TD._rows(w)
            held_rows = [r for r in rows if r["est"] == "hold"]
            assert held_rows and all(r["solver_status"] == "-1"
                                     and r["uX"] == "nan" for r in held_rows)
            assert rows[-1]["est"] == "ekf"
        finally:
            sim.close()


def test_interleaved_late_frames_cannot_stretch_the_commandless_window():
    """Re-audit round 3: a held tick must not restart the tag path's stale
    debounce. Late frames on every third tick (fresh for the assembler on
    one tick, stale on two) while the EKF refuses every tag: ONE counter
    ends the run within tag_stale_hold_s of the EKF going stale."""
    TD = _td()
    TC = TD._harness()
    with tempfile.TemporaryDirectory() as tmp:
        w, _bus, _p, _l = TC._worker(tmp)
        sim = TD._Sim(w, TC)
        try:
            for _ in range(10):
                sim.step()
            w.set_estimator("ekf")
            sim.engage()
            for _ in range(3):
                sim.step()
            w.ekf.feed_tag = lambda *a, **k: None   # the EKF gets no tag
            n, first_stale = 0, None
            for k in range(80):
                sim.step(fix=(k % 3 == 0), lat=0.48)
                n += 1
                if first_stale is None and w._ekf_stale:
                    first_stale = n
                if not w.engaged:
                    break
            assert not w.engaged, "the commandless window never closed"
            assert first_stale is not None
            need = round(w.cfg.engage.get("tag_stale_hold_s", 0.4)
                         * w.cfg.ctrl_hz)
            assert n - first_stale <= need + 1, (n, first_stale, need)
        finally:
            sim.close()


def test_a_dvl_switched_in_from_the_yaml_unproven_is_not_given_to_the_ekf():
    """The re-audit's N2: `dvl.use: true` in the YAML starts IN without the
    proof gate; the EKF asks the gate itself."""
    TD = _td()
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, _bus, _p, _l = TD._dvl_worker(tmp, use=True, reentry_s=0.0)
        w._setup_ekf()
        sim = TD._Sim(w, TC)
        try:
            assert w.dvl.use and not w.dvl.proven
            for _ in range(20):
                sim.step(v=(0.03, 0.0, 0.0))
            assert w.ekf.n[EK.TOPIC_DVL]["in"] == 0
            sim.prove()
            for _ in range(15):
                sim.step()
            assert w.ekf.n[EK.TOPIC_DVL]["used"] > 0
        finally:
            sim.close()

# ====================================================================== panel
def test_the_est_button_reflects_the_worker_and_never_flips_itself():
    TC = _td()._harness()
    TC._app()
    from rov_gui.widgets.trajectory import TrajectoryWindow

    p = TrajectoryWindow()
    asked = []
    p.estimator_requested.connect(asked.append)
    p.add_status(MpcStatus())                 # no EKF built
    assert p.btn_est.isHidden()
    p.add_status(MpcStatus(ekf_state="not_ready", ekf_note="no tag yet"))
    assert not p.btn_est.isHidden() and p.btn_est.text() == "EST TAG"
    p.btn_est.click()                         # a REQUEST...
    assert asked == ["ekf"] and p.btn_est.text() == "EST TAG"
    assert not p.btn_est.isChecked()
    p.add_status(MpcStatus(ekf_state="ready", ekf_control=True))
    assert p.btn_est.isChecked() and p.btn_est.text() == "EST EKF"
    p.add_status(MpcStatus(engaged=True, mode="pid", ekf_state="ready",
                           ekf_control=True, ekf_flew=True))
    assert not p.btn_est.isEnabled() and "est:EKF" in p.chip.text()
    p.add_status(MpcStatus(csv_open=True, ekf_state="ready"))
    assert not p.btn_est.isEnabled()


def test_the_csv_header_ends_with_the_ekf_columns():
    from rov_gui.control.workers import CSV_HEADER

    cols = CSV_HEADER.strip().split(",")
    # schema 18's sixteen, located BY NAME: schema 19 (2026-10-10) appended
    # the three pwm-LOW columns after them, schema 20 (2026-10-10 b) the
    # pwm_guard code (test_rl_pwm_scale / test_rl_pwm_interlocks pin those)
    # and schema 21 (2026-10-15) the three attitude-age columns (test_att_age)
    # — append-at-end, so 18's stay where they were.
    i = cols.index("est")
    assert cols[i:i + 16] == ["est", "ekf_ok", "ekf_px", "ekf_py", "ekf_pz",
                              "ekf_roll_deg", "ekf_pitch_deg", "ekf_yaw_deg",
                              "ekf_u", "ekf_v", "ekf_w", "ekf_sig_xy_m",
                              "ekf_age_s", "ekf_tag_late_s", "ekf_n_tag",
                              "ekf_n_dvl"]
    assert cols[i + 16:] == ["i_pred_a", "null_v_us", "null_h_us", "pwm_guard",
                             "att_age_v_s", "att_tb_ms", "banner_age_s",
                             # schema 22 (2026-10-16): the DVL selection
                             "w_dvl", "dvl_code", "dvl_res_fast_m_s",
                             "dvl_res_slow_m_s", "dvl_esc_s", "dvl_align",
                             "dvl_scale"]
    assert len(cols) == len(set(cols))


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
