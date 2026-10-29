"""DvlPose (control/dvl_pose.py) on a synthetic vehicle — no Qt, no worker.

Every number here is a SYNTHETIC truth chosen for the test, not a
measurement: the truth vehicle flies a known body velocity and yaw rate at a
fixed roll/pitch, its gyro integral, DVL samples (validity times ~0.1 s in
the past) and tag states are generated from that truth, and the pose is
compared with it.
"""

from __future__ import annotations

import math
import re
from types import SimpleNamespace

import numpy as np
import pytest

from rov_gui.control import dvl_pose as DP
from rov_gui.control.state_assembler import rot_zyx
from rov_gui.state import VehicleImu


# =============================================================================
# a truth vehicle
# =============================================================================
class Truth:
    """Constant roll/pitch, body velocity v_b, yaw rate omega(t) — integrated
    on a 1 ms grid so position and yaw are exact to ~1e-6 at any time."""

    def __init__(self, T=200.0, v_b=(0.1, 0.0, 0.0), omega=lambda t: 0.0,
                 phi=0.02, theta=0.15, psi0=0.3, p0=(1.0, -2.0), z0=0.8,
                 depth0=0.5, t0=100.0):
        self.t0, self.phi, self.theta = t0, phi, theta
        dt = 1e-3
        n = int(T / dt) + 2
        tt = np.arange(n) * dt
        om = np.array([omega(s) for s in tt]) if callable(omega) else \
            np.full(n, float(omega))
        psi = psi0 + np.concatenate([[0.0], np.cumsum(om[:-1]) * dt])
        a = rot_zyx(phi, theta, 0.0) @ np.asarray(v_b, float)
        vx = np.cos(psi) * a[0] - np.sin(psi) * a[1]
        vy = np.sin(psi) * a[0] + np.cos(psi) * a[1]
        x = p0[0] + np.concatenate([[0.0], np.cumsum(vx[:-1]) * dt])
        y = p0[1] + np.concatenate([[0.0], np.cumsum(vy[:-1]) * dt])
        self.tt, self.psi, self.om, self.x, self.y = tt + t0, psi, om, x, y
        self.vz = a[2]
        self.z0, self.depth0 = z0, depth0
        self.v_b = np.asarray(v_b, float)
        # body rates from the Euler rates (phi, theta constant)
        st, ct = math.sin(theta), math.cos(theta)
        sp, cp = math.sin(phi), math.cos(phi)
        self.pqr = np.stack([-om * st, om * ct * sp, om * ct * cp], axis=1)
        self.gint = np.concatenate(
            [np.zeros((1, 3)), np.cumsum(self.pqr[:-1], axis=0) * dt])

    def _i(self, t):
        return float(np.interp(t, self.tt, np.arange(len(self.tt))))

    def at(self, t, arr):
        return float(np.interp(t, self.tt, arr))

    def pos(self, t):
        return np.array([self.at(t, self.x), self.at(t, self.y)])

    def yaw(self, t):
        return self.at(t, self.psi)

    def z(self, t):
        return self.z0 + self.vz * (t - self.t0)

    def depth(self, t):
        return self.depth0 + self.vz * (t - self.t0)

    def eta(self, t):
        p = self.pos(t)
        return np.array([p[0], p[1], self.z(t), self.phi, self.theta,
                         DP.wrap(self.yaw(t))])

    def pqr_at(self, t):
        return np.array([self.at(t, self.pqr[:, k]) for k in range(3)])

    def gint_at(self, t):
        return np.array([self.at(t, self.gint[:, k]) for k in range(3)])


class Feed:
    """Drives a DvlPose from a Truth at 20 Hz: IMU snapshots with a RAW_IMU
    integral (optional z bias), DVL samples at 10 Hz delivered 0.1 s late,
    tag states (truth now, fix captured 0.14 s ago) at 15 Hz."""

    def __init__(self, truth, pose, gyro_bias=0.0, dvl_bias=(0.0, 0.0),
                 dvl_hz=10.0, dvl_late=0.1, tag_hz=15.0, tag_late=0.14,
                 n_tags=6):
        self.T, self.pose = truth, pose
        self.gyro_bias = gyro_bias          # rad/s on the z axis
        self.dvl_bias = np.asarray(dvl_bias, float)
        self.dvl_dt, self.dvl_late = 1.0 / dvl_hz, dvl_late
        self.tag_dt, self.tag_late = 1.0 / tag_hz, tag_late
        self.n_tags = n_tags
        self.t = truth.t0
        self._next_dvl = truth.t0
        self._fix_t = None
        self.n_gyro = 0
        self.dvl_on = lambda t: True
        self.dvl_usable = lambda t: True
        self.tag_jump = lambda t: np.zeros(2)
        self.gyro_on = lambda t: True
        self._gyro_hold = None
        self.n_gaps = 0

    def imu(self, t):
        T = self.T
        g = T.gint_at(t)
        g = g + np.array([0.0, 0.0, self.gyro_bias * (t - T.t0)])
        if self.gyro_on(t):
            self.n_gyro += 1
            self._gyro_hold = (tuple(g), t)
        gi, tg = (self._gyro_hold if self._gyro_hold else (None, None))
        p, q, r = T.pqr_at(t)
        return VehicleImu(roll=T.phi, pitch=T.theta, yaw=0.0, p=p, q=q, r=r,
                          t_att=t, t_att_v=t, depth_m=T.depth(t), t_baro=t,
                          gyro_int=gi, t_gyro=tg, n_gyro=self.n_gyro,
                          n_gyro_gaps=self.n_gaps,
                          depth_sp2_m=T.depth(t), t_sp2=t, stamp=t)

    def tag(self, t, fresh=True):
        T = self.T
        tc = t - self.tag_late
        if self._fix_t is None or tc >= self._fix_t + self.tag_dt - 1e-9:
            self._fix_t = tc
        tcap = self._fix_t
        eta = T.eta(t)
        eta[0:2] += self.tag_jump(t)
        meas = {"eta": eta,
                "nu": np.concatenate([T.v_b, T.pqr_at(t)]),
                "nudot": np.zeros(6)}
        R = rot_zyx(T.phi, T.theta, T.yaw(tcap))
        fix = SimpleNamespace(ok=True, n_tags=self.n_tags, t_capture=tcap,
                              R_ned_body=R,
                              p_ned=np.r_[T.pos(tcap) + self.tag_jump(t),
                                          T.z(tcap)])
        return meas, fix

    def tick(self, t, tags=True, engaged=False, dt=0.05):
        T, pose = self.T, self.pose
        imu = self.imu(t)
        pose.note_imu(imu)
        while self._next_dvl + self.dvl_late <= t + 1e-9:
            tv = self._next_dvl
            if self.dvl_on(tv):
                v = T.v_b.copy()
                v[:2] += self.dvl_bias
                pose.note_dvl(tv, v, self.dvl_usable(tv),
                              "" if self.dvl_usable(tv) else "no_lock",
                              T.phi, T.theta, 0.35, t_rx=t)
            self._next_dvl += self.dvl_dt
        meas, fix = self.tag(t) if tags else (None, None)
        return pose.step(t, imu, meas, fix, engaged, dt), meas

    def run(self, seconds, tags=True, engaged=False, every=None):
        out = []
        n = int(round(seconds / 0.05))
        for _ in range(n):
            self.t += 0.05
            tg = tags(self.t) if callable(tags) else tags
            en = engaged(self.t) if callable(engaged) else engaged
            (m, h), tm = self.tick(self.t, tg, en)
            out.append((self.t, m, h, tm))
            if every:
                every(self.t, m, h)
        return out


def _pose(mode="control", **cfg):
    base = {"bias_min_span_s": 20.0, "bias_min_n": 50}
    base.update(cfg)
    return DP.DvlPose(base, mode=mode, nudot_source="fd")


def _xy_err(truth, t, m):
    return float(np.hypot(*(m["eta"][:2] - truth.pos(t))))


# =============================================================================
# configuration
# =============================================================================
def test_resolve_rejects_unknown_keys_and_bad_numbers():
    with pytest.raises(ValueError, match="unknown"):
        DP.resolve({"hold_sec": 1.0})
    with pytest.raises(ValueError, match="lost_s"):
        DP.resolve({"lost_s": -1})
    with pytest.raises(ValueError, match="hold_s <= max_interp_s"):
        DP.resolve({"hold_s": 0.9, "max_interp_s": 0.6})
    c = DP.resolve({"abort_err_m": None})
    assert c["abort_err_m"] is None
    assert DP.resolve(None)["mode"] == "off"


def test_mode_command_line_wins_and_yaml_control_flies_shadow():
    assert DP.mode_from("off", "control")[0] == "control"
    assert DP.mode_from("control", None)[0] == "shadow"
    assert "only the command line" in DP.mode_from("control", None)[1]
    assert DP.mode_from("shadow", None)[0] == "shadow"
    assert DP.mode_from("shadow", "off")[0] == "off"


# =============================================================================
# TAG FIX ON
# =============================================================================
def test_tag_on_passes_the_tag_state_through_unchanged():
    T = Truth()
    pose = _pose()
    f = Feed(T, pose)
    for t, m, h, tm in f.run(3.0):
        assert h["ok"] and h["state"] == DP.ST_TAG
        # the pose IS the tag state: the controller sees what it sees
        # without --dvl_pose
        np.testing.assert_array_equal(m["eta"], tm["eta"])
        np.testing.assert_array_equal(m["nu"], tm["nu"])
        np.testing.assert_array_equal(m["nudot"], tm["nudot"])


def test_the_first_anchor_needs_anchor_n_agreeing_fixes():
    T = Truth(v_b=(0.0, 0.0, 0.0))
    pose = _pose(anchor_n=5)
    f = Feed(T, pose)
    # every other fix jumps 0.3 m: never 5 agreeing in a row
    f.tag_jump = lambda t: np.array([0.3, 0.0]) if int(t * 15) % 2 else \
        np.zeros(2)
    f.run(2.0)
    assert not pose.anchored
    assert pose.tag_refusal(False, f.t).startswith("no anchor")
    f.tag_jump = lambda t: np.zeros(2)
    f.run(0.2)
    assert not pose.anchored             # 3 fixes so far
    f.run(0.4)
    assert pose.anchored


# =============================================================================
# dead reckoning
# =============================================================================
def test_a_tag_dropout_is_carried_by_the_dvl():
    T = Truth(v_b=(0.1, 0.03, 0.0))
    pose = _pose()
    f = Feed(T, pose)
    f.run(5.0)                            # anchored on the tags
    assert pose.anchored
    out = f.run(20.0, tags=False, engaged=True)
    for t, m, h, _ in out:
        assert h["ok"], h["why"]
        assert h["state"] == DP.ST_DR
        # exact synthetic velocity and heading: only the integration's own
        # error, and the 0.45 s extrapolation of a constant velocity
        assert _xy_err(T, t, m) < 2e-3
        assert abs(m["eta"][2] - T.z(t)) < 1e-6
    assert out[-1][2]["since_fix_s"] == pytest.approx(20.0, abs=0.06)
    assert out[-1][2]["dist_m"] == pytest.approx(
        20.0 * math.hypot(*(rot_zyx(T.phi, T.theta, 0) @ T.v_b)[:2]),
        rel=0.02)


def test_the_heading_is_the_euler_rate_not_the_body_rate():
    # a 10-deg pitch and a steady turn: r alone under-turns by cos(theta)
    T = Truth(v_b=(0.08, 0.0, 0.0), omega=0.2, theta=math.radians(10))
    pose = _pose()
    f = Feed(T, pose)
    f.run(3.0)
    out = f.run(30.0, tags=False)
    t, m, h, _ = out[-1]
    assert abs(DP.wrap(m["eta"][5] - T.yaw(t))) < math.radians(0.05)
    turned = 0.2 * 30.0
    assert turned * (1 - math.cos(math.radians(10))) > math.radians(5)


def test_the_gyro_bias_is_learnt_while_on_and_frozen_while_off():
    bias = math.radians(0.03)             # deg/s -> rad/s
    T = Truth(v_b=(0.08, 0.0, 0.0), omega=lambda s: 0.05 * math.sin(0.1 * s),
              T=400.0)
    pose = _pose(bias_min_span_s=60.0, bias_min_n=300)
    f = Feed(T, pose, gyro_bias=bias)
    f.run(90.0)
    b = pose.head.bias_info()
    assert b["state"] == "ok"
    assert b["dps"] == pytest.approx(0.03, abs=0.002)
    assert pose.tag_refusal(False, f.t) == ""
    pose.set_tag_on(False, f.t, engaged=False)
    out = f.run(60.0)                     # tags fed, but OFF
    t, m, h, _ = out[-1]
    assert h["state"] == DP.ST_DR
    assert abs(DP.wrap(m["eta"][5] - T.yaw(t))) < math.radians(0.2)
    # the same without the learnt bias: ~ bias x 60 s
    pose2 = _pose(bias_min_span_s=60.0, bias_min_n=300,
                  require_gyro_bias=False)
    f2 = Feed(T, pose2, gyro_bias=bias)
    f2.run(5.0)
    pose2.set_tag_on(False, f2.t, engaged=False)
    t2, m2, _h2, _ = f2.run(60.0)[-1]
    assert abs(DP.wrap(m2["eta"][5] - T.yaw(t2))) == pytest.approx(
        math.radians(1.8), rel=0.15)


def test_off_is_refused_until_the_dead_reckoning_can_carry_the_vehicle():
    T = Truth()
    pose = _pose(bias_min_span_s=30.0, bias_min_n=100)
    f = Feed(T, pose)
    f.run(5.0)
    assert pose.anchored
    why = pose.tag_refusal(False, f.t)
    assert "gyro bias is not learnt" in why
    assert pose.tag_refusal(False, f.t, dvl_ok=False,
                            dvl_why="DVL not proven").startswith(
        "the DVL cannot carry")
    shadow = _pose(mode="shadow", bias_min_span_s=30.0, bias_min_n=100)
    fs = Feed(T, shadow)
    fs.run(5.0)
    assert shadow.tag_refusal(False, fs.t) == ""   # shadow: a measurement
    assert pose.tag_refusal(True, f.t) == ""        # ON is always allowed
    f.dvl_on = lambda t: False
    f.run(1.0)
    assert "no usable DVL" in pose.tag_refusal(False, f.t)


# =============================================================================
# gaps
# =============================================================================
def test_a_dvl_gap_extrapolates_then_holds_then_loses_the_pose():
    T = Truth(v_b=(0.1, 0.0, 0.0))
    pose = _pose()
    f = Feed(T, pose)
    f.run(5.0)
    f.dvl_on = lambda t: t < f.t - 0.5     # samples stop ~now
    seen = []
    for t, m, h, _ in f.run(3.0, tags=False, engaged=True):
        seen.append((t, h["state"], h["ok"], h["why"]))
    states = [s for _t, s, _ok, _w in seen]
    assert DP.ST_DR in states and DP.ST_HELD in states
    lost = [(t, w) for t, s, ok, w in seen if not ok]
    assert lost, "the pose must be lost after lost_s"
    assert "stale" in lost[0][1]          # the worker debounces "stale"
    assert states.count(DP.ST_HELD) >= 5     # (positions: test_held_means_...)


def test_a_long_gap_is_committed_as_it_was_shown():
    """The last HELD tick against the tick the gap-ending sample lands on
    (review 2026-10-18: the earlier version read two ticks AFTER it and
    could not see the commit rule)."""
    T = Truth(v_b=(0.1, 0.0, 0.0))
    pose = _pose()
    f = Feed(T, pose)
    f.run(5.0)
    t_cut = f.t
    f.dvl_on = lambda tv: not (t_cut - 0.1 < tv < t_cut + 0.9)
    prev, jumps = None, []
    for _ in range(40):
        n0 = pose.n_usable
        f.t += 0.05
        (m, h), _tm = f.tick(f.t, tags=False, engaged=True)
        if prev is not None and pose.n_usable > n0 and prev[1] == DP.ST_HELD:
            jumps.append(float(np.hypot(*(m["eta"][:2] - prev[0]))))
        prev = (m["eta"][:2].copy(), h["state"])
    assert len(jumps) == 1, jumps
    # the sample is ~0.1 s old when it lands: only its own extrapolation
    assert jumps[0] < 0.1 * 0.15 + 1e-3, jumps


def test_held_means_the_position_stops():
    T = Truth(v_b=(0.1, 0.0, 0.0))
    pose = _pose()
    f = Feed(T, pose)
    f.run(5.0)
    t_cut = f.t
    f.dvl_on = lambda tv: tv < t_cut - 0.1
    held = [m["eta"][:2].copy() for _t, m, h, _ in
            f.run(1.2, tags=False, engaged=True) if h["state"] == DP.ST_HELD]
    assert len(held) >= 5
    assert max(float(np.hypot(*(q - held[0]))) for q in held) < 1e-9


# =============================================================================
# corrections
# =============================================================================
def _drift_then_on(engaged: bool, dvl_bias=(0.01, 0.0), seconds=20.0, **cfg):
    T = Truth(v_b=(0.05, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False, **cfg)
    f = Feed(T, pose)
    f.run(5.0)
    pose.set_tag_on(False, f.t, engaged)
    f.dvl_bias = np.asarray(dvl_bias, float)
    f.run(seconds, engaged=engaged)       # OFF: drifts dvl_bias x seconds
    return T, pose, f


def test_tags_back_on_while_engaged_agree_first_then_slew_in():
    T, pose, f = _drift_then_on(engaged=True)
    f.dvl_bias = np.zeros(2)
    off0 = None
    pose.set_tag_on(True, f.t, engaged=True)
    states, errs = [], []
    prev = None
    for t, m, h, tm in f.run(12.0, engaged=True):
        states.append(h["state"])
        e = _xy_err(T, t, m)
        errs.append(e)
        if off0 is None:
            off0 = e
        if prev is not None:
            # the pose never moves faster than the vehicle + the slew rate
            vstep = float(np.hypot(*(m["eta"][:2] - prev[1])))
            assert vstep <= 0.05 * 0.05 + 0.02 * 0.05 + 1e-4
        prev = (t, m["eta"][:2].copy())
    assert off0 == pytest.approx(0.2, abs=0.02)
    # the first fixes do not move it (anchor_n must agree)...
    assert states[:3] == [DP.ST_DR] * 3
    # ...then it slews (blend) and ends on the tags
    assert DP.ST_BLEND in states
    assert errs[-1] < 1e-3 and states[-1] == DP.ST_TAG
    rec = pose.corrections[-1]
    assert rec["applied"] == "slew" and rec["kind"] == "reacquire"
    assert rec["err_m"] == pytest.approx(0.2, abs=0.02)
    assert rec["since_fix_s"] == pytest.approx(20.0, abs=0.5)


def test_tags_back_on_while_disengaged_snap():
    T, pose, f = _drift_then_on(engaged=False)
    f.dvl_bias = np.zeros(2)
    pose.set_tag_on(True, f.t, engaged=False)
    out = f.run(1.0)
    assert out[-1][2]["state"] == DP.ST_TAG
    assert _xy_err(T, out[-1][0], out[-1][1]) < 1e-6
    assert pose.corrections[-1]["applied"] == "snap"


def test_a_large_disagreement_while_engaged_is_a_fault_not_a_slew():
    # (abort off: with it, the OFF divergence interlock ends this run at
    # 30 cm long before the tags come back — which is its job)
    T, pose, f = _drift_then_on(engaged=True, dvl_bias=(0.02, 0.0),
                                seconds=20.0, abort_err_m=None)
    f.dvl_bias = np.zeros(2)
    pose.set_tag_on(True, f.t, engaged=True)
    faults = [h["fault"] for _t, _m, h, _ in f.run(1.0, engaged=True)
              if h["fault"]]
    assert faults and "disagree" in faults[0]
    assert pose.corrections[-1]["applied"] == "refused"


def test_a_one_frame_tag_jump_never_reaches_the_pose():
    T = Truth(v_b=(0.05, 0.0, 0.0))
    pose = _pose()
    f = Feed(T, pose)
    f.run(5.0)
    t_j = f.t + 0.5
    f.tag_jump = lambda t: np.array([0.8, 0.0]) if t_j <= t < t_j + 0.06 \
        else np.zeros(2)
    for t, m, h, tm in f.run(2.0, engaged=True):
        assert _xy_err(T, t, m) < 5e-3, (t, h["state"], h["warn"])
    assert pose.n_jumps_held >= 1


def test_a_lasting_tag_level_change_is_believed_after_anchor_n():
    T = Truth(v_b=(0.05, 0.0, 0.0))
    pose = _pose()
    f = Feed(T, pose)
    f.run(5.0)
    t_j = f.t + 0.2
    f.tag_jump = lambda t: np.array([0.08 + 0.05, 0.0]) if t >= t_j \
        else np.zeros(2)
    out = f.run(10.0, engaged=True)
    assert pose.corrections and pose.corrections[-1]["kind"] == "jump"
    t, m, h, tm = out[-1]
    np.testing.assert_allclose(m["eta"][:2], tm["eta"][:2], atol=1e-6)


# =============================================================================
# OFF: the divergence interlock
# =============================================================================
def test_off_warns_then_faults_when_the_tags_show_a_divergence():
    T, pose, f = _drift_then_on(engaged=True, dvl_bias=(0.03, 0.0),
                                seconds=1.0)
    warned = faulted = None
    for t, m, h, _ in f.run(15.0, engaged=True):
        if warned is None and "from the tags" in h["warn"]:
            warned = (t, h["err_m"])
        if faulted is None and h["fault"]:
            faulted = (t, h["err_m"], h["fault"])
    assert warned and warned[1] == pytest.approx(0.10, abs=0.01)
    assert faulted and faulted[1] >= 0.25        # abort_err_m (= max_corr_m)
    # held for abort_hold_s first: the error passed 0.25 m a second earlier
    assert faulted[1] > 0.25 + 0.5 * 0.03
    assert "diverged" in faulted[2]


def test_off_the_pose_does_not_depend_on_the_tags():
    T = Truth(v_b=(0.07, 0.02, 0.0), omega=0.03)

    def go(with_tags: bool):
        pose = _pose(require_gyro_bias=False)
        f = Feed(T, pose, dvl_bias=(0.004, -0.002))
        f.run(5.0)
        pose.set_tag_on(False, f.t, engaged=True)
        return [m["eta"].copy() for _t, m, _h, _ in
                f.run(20.0, tags=with_tags, engaged=True)]

    a, b = go(True), go(False)
    for ea, eb in zip(a, b):
        np.testing.assert_array_equal(ea, eb)


# =============================================================================
# heading, z, record
# =============================================================================
def test_a_gyro_hole_after_the_anchor_loses_the_heading_until_reanchored():
    T = Truth(v_b=(0.05, 0.0, 0.0), omega=0.05)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(5.0)
    f.n_gaps = 1                          # the reader left an interval out
    out = f.run(0.5, tags=False, engaged=True)
    assert not out[-1][2]["ok"]
    assert "gap after the anchor" in out[-1][2]["why"]
    out = f.run(1.0, tags=True, engaged=True)
    assert out[-1][2]["ok"] and out[-1][2]["state"] in (DP.ST_TAG,
                                                         DP.ST_BLEND)
    out = f.run(1.0, tags=False, engaged=True)
    assert out[-1][2]["ok"]


def test_z_is_the_bar30_plus_the_tag_offset_and_w_its_derivative():
    T = Truth(v_b=(0.05, 0.0, 0.02), theta=0.0, phi=0.0)
    pose = _pose()
    f = Feed(T, pose)
    f.run(5.0)
    out = f.run(5.0, tags=False, engaged=True)
    t, m, h, _ = out[-1]
    assert h["z_src"] == "sp2"
    assert m["eta"][2] == pytest.approx(T.z(t), abs=1e-6)
    assert m["nu"][2] == pytest.approx(0.02, abs=2e-3)


def test_stale_inputs_say_why_in_words_the_worker_debounces():
    T = Truth()
    pose = _pose()
    f = Feed(T, pose)
    f.run(5.0)
    f.gyro_on = lambda t: False
    out = f.run(1.0, tags=False, engaged=True)
    assert not out[-1][2]["ok"] and "stale" in out[-1][2]["why"]
    imu = f.imu(f.t)
    imu.t_att = f.t - 1.0
    _m, h = pose.step(f.t, imu, None, None, True, 0.05)
    assert h["why"].startswith("imu stale")


def test_the_record_says_what_happened():
    T, pose, f = _drift_then_on(engaged=True)
    f.dvl_bias = np.zeros(2)
    pose.set_tag_on(True, f.t, engaged=True)
    f.run(12.0, engaged=True)
    md = pose.meta()
    assert [x["on"] for x in md["toggles"]] == [False, True]
    assert md["corrections"][-1]["applied"] == "slew"
    assert md["rows"].get(DP.ST_DR, 0) > 0
    assert md["max_err_m"] == pytest.approx(0.2, abs=0.03)
    assert md["config"]["hold_s"] == 0.45


# =============================================================================
# review 2026-10-18 — what the first version let through (ported mutation
# killers from the review's test-adequacy lens, then the fixes' own tests)
# =============================================================================
def test_a_one_frame_tag_jump_while_off_does_not_end_a_control_run():
    """abort_hold_s: OFF compares every fix; one bad frame is not a
    divergence (2026-10-15 had a 0.83 m one)."""
    T, pose, f = _drift_then_on(engaged=True, dvl_bias=(0.0, 0.0),
                                seconds=2.0)
    assert not pose.tag_on
    t_j = f.t + 0.5
    f.tag_jump = lambda t: np.array([0.8, 0.0]) if t_j <= t < t_j + 0.06 \
        else np.zeros(2)
    out = f.run(3.0, engaged=True)
    assert any(h["err_m"] is not None and h["err_m"] > 0.5
               for _t, _m, h, _ in out)          # the jump was compared
    assert not [h["fault"] for _t, _m, h, _ in out if h["fault"]]


def test_one_tag_fixes_never_place_or_correct_a_pose_that_demands_two():
    T = Truth()
    pose = _pose(min_tags=2)
    f = Feed(T, pose, n_tags=1)
    out = f.run(3.0)
    assert not pose.anchored and pose.head.bias_n == 0
    assert all(not h["ok"] and h["state"] == DP.ST_NO_ANCHOR
               for _t, _m, h, _ in out)


def test_agreement_counts_fixes_not_ticks():
    """The tag state repeats a fix between captures; anchor_n is FIXES."""
    T = Truth(v_b=(0.05, 0.0, 0.0))
    pose = _pose()
    f = Feed(T, pose)
    f.run(5.0)
    f.tag_dt = 0.5                       # a fix every 0.5 s
    f.run(1.0)
    t_jc = f._fix_t + 0.25               # keyed on the CAPTURE time
    f.tag_jump = lambda t: np.array([0.13, 0.0]) if f._fix_t >= t_jc \
        else np.zeros(2)
    n0 = len(pose.corrections)
    f.run(1.5, engaged=True)             # <= 4 new fixes: cannot agree yet
    assert len(pose.corrections) == n0, pose.corrections[n0:]
    f.run(2.0, engaged=True)
    assert len(pose.corrections) == n0 + 1
    assert pose.corrections[-1]["kind"] == "jump"


def test_off_a_heading_divergence_alone_faults():
    T = Truth(v_b=(0.0, 0.0, 0.0))
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose, gyro_bias=math.radians(2.0))   # not learnt (20 s span)
    f.run(5.0)
    pose.set_tag_on(False, f.t, engaged=True)
    out = f.run(8.0, engaged=True)
    faults = [(h["err_m"], h["err_yaw_deg"], h["fault"])
              for _t, _m, h, _ in out if h["fault"]]
    assert faults and "diverged" in faults[0][2], faults[:1]
    assert faults[0][0] < 0.01 and faults[0][1] >= 10.0


def test_a_large_heading_disagreement_while_engaged_is_a_fault():
    T = Truth(v_b=(0.0, 0.0, 0.0))
    pose = _pose(require_gyro_bias=False, abort_err_yaw_deg=None)
    f = Feed(T, pose, gyro_bias=math.radians(2.0))
    f.run(5.0)
    pose.set_tag_on(False, f.t, engaged=True)
    f.run(6.5, engaged=True)             # ~13 deg of heading drift
    pose.set_tag_on(True, f.t, engaged=True)
    faults = [h["fault"] for _t, _m, h, _ in f.run(1.0, engaged=True)
              if h["fault"]]
    assert faults and "disagree" in faults[0]
    assert pose.corrections[-1]["applied"] == "refused"


def test_a_heading_correction_is_slewed_while_engaged():
    T = Truth(v_b=(0.0, 0.0, 0.0))
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose, gyro_bias=math.radians(0.5))
    f.run(5.0)
    pose.set_tag_on(False, f.t, engaged=True)
    f.run(8.0, engaged=True)             # ~4 deg of heading drift
    pose.set_tag_on(True, f.t, engaged=True)
    yaws = [m["eta"][5] for _t, m, _h, _ in f.run(6.0, engaged=True)]
    steps = [abs(DP.wrap(b - a)) for a, b in zip(yaws, yaws[1:])]
    assert pose.corrections[-1]["applied"] == "slew"
    assert pose.corrections[-1]["err_yaw_deg"] > 3.0
    assert max(steps) <= math.radians((1.0 + 0.5) * 0.05) + 1e-6, \
        math.degrees(max(steps))


def test_w_includes_the_tilt_term():
    T = Truth(v_b=(0.05, 0.0, 0.02), theta=0.15, phi=0.0)
    pose = _pose()
    f = Feed(T, pose)
    f.run(5.0)
    t, m, h, _ = f.run(5.0, tags=False, engaged=True)[-1]
    assert h["state"] == DP.ST_DR
    assert m["nu"][2] == pytest.approx(0.02, abs=2e-3)


def test_dead_reckoning_in_a_turn_rotates_each_sample_by_its_own_heading():
    """The DVL is 0.1 s late: each sample must be rotated by the heading OF
    ITS MOMENT, interpolated from the gyro history."""
    T = Truth(v_b=(0.1, 0.0, 0.0), omega=0.15)
    pose = _pose()
    f = Feed(T, pose)
    f.run(5.0)
    out = f.run(10.0, tags=False, engaged=True)
    assert all(h["state"] == DP.ST_DR for _t, _m, h, _ in out)
    assert max(_xy_err(T, t, m) for t, m, _h, _ in out) < 5e-3


class YawBurstFeed(Feed):
    burst = None                         # (t0, t1, yaw offset rad)

    def tag(self, t, fresh=True):
        meas, fix = super().tag(t, fresh)
        b = self.burst
        if b and b[0] <= t < b[1]:
            meas["eta"][5] = DP.wrap(meas["eta"][5] + b[2])
            fix.R_ned_body = rot_zyx(self.T.phi, self.T.theta,
                                     self.T.yaw(fix.t_capture) + b[2])
        return meas, fix


def test_a_burst_of_wrong_tag_yaws_does_not_bend_the_bias():
    """4 deg passes the jump gate (5 deg): only the fit's own 2-deg
    rejection keeps it out of the bias."""
    T = Truth(v_b=(0.08, 0.0, 0.0), omega=lambda s: 0.05 * math.sin(0.1 * s),
              T=400.0)
    pose = _pose(bias_min_span_s=60.0, bias_min_n=300)
    f = YawBurstFeed(T, pose, gyro_bias=math.radians(0.03))
    f.burst = (T.t0 + 80.0, T.t0 + 85.0, math.radians(4.0))
    f.run(90.0)
    b = pose.head.bias_info()
    assert b["state"] == "ok"
    assert b["dps"] == pytest.approx(0.03, abs=0.002)


def test_off_is_refused_while_the_heading_is_stale():
    T = Truth()
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(5.0)
    assert pose.tag_refusal(False, f.t) == ""
    f.gyro_on = lambda t: False
    f.run(0.6)
    assert "heading" in pose.tag_refusal(False, f.t)


def test_after_a_one_frame_jump_the_next_agreeing_fix_is_the_pose_again():
    T = Truth(v_b=(0.05, 0.0, 0.0))
    pose = _pose()
    f = Feed(T, pose)
    f.run(5.0)
    t_j = f.t + 0.5
    f.tag_jump = lambda t: np.array([0.8, 0.0]) if t_j <= t < t_j + 0.06 \
        else np.zeros(2)
    out = f.run(2.0, engaged=True)
    i = next(k for k, (t, _m, _h, _tm) in enumerate(out) if t >= t_j)
    states = [h["state"] for _t, _m, h, _ in out[i:]]
    assert next(k for k, st in enumerate(states) if st == DP.ST_TAG) <= 3, \
        states[:12]


def test_a_stale_depth_is_said_in_the_word_the_worker_debounces():
    T = Truth()
    pose = _pose()
    f = Feed(T, pose)
    f.run(5.0)
    imu = f.imu(f.t)
    imu.t_sp2 = imu.t_baro = f.t - 5.0
    _m, h = pose.step(f.t, imu, None, None, True, 0.05)
    assert not h["ok"] and "stale" in h["why"], h["why"]


def test_max_unaided_s_ends_a_control_run():
    T, pose, f = _drift_then_on(engaged=True, dvl_bias=(0.0, 0.0),
                                seconds=0.5, max_unaided_s=3.0)
    faults = [h["fault"] for _t, _m, h, _ in f.run(4.0, tags=False,
                                                    engaged=True)
              if h["fault"]]
    assert faults and "without a tag correction" in faults[0]


def test_the_att_rate_fallback_is_the_euler_rate_too():
    T = Truth(v_b=(0.08, 0.0, 0.0), omega=0.2, theta=math.radians(10))
    pose = _pose(heading="att_rate")
    f = Feed(T, pose)
    f.run(3.0)
    t, m, h, _ = f.run(30.0, tags=False)[-1]
    assert h["state"] == DP.ST_DR
    assert abs(DP.wrap(m["eta"][5] - T.yaw(t))) < math.radians(0.1)


# ---- the fixes' own tests ---------------------------------------------------
def test_the_shipped_config_spells_every_key_and_resolves():
    """A key added to the code but not to config/hw_mpc.yaml, or a value the
    checks refuse, breaks the station's config load (it did once on
    2026-10-18: abort_err_m 0.30 > max_corr_m 0.25)."""
    from pathlib import Path

    from rov_gui.control.geometry import MpcConfig

    mc = MpcConfig.load(Path(__file__).resolve().parents[2] / "config" /
                        "hw_mpc.yaml")
    assert set(mc.dvl_pose) == set(DP.KEYS), set(DP.KEYS) ^ set(mc.dvl_pose)
    c = DP.resolve(mc.dvl_pose)
    assert c["mode"] == "off"
    assert all(c[k] == DP.DEFAULTS[k] for k in DP.KEYS if k != "mode"), \
        {k: (c[k], DP.DEFAULTS[k]) for k in DP.KEYS
         if k != "mode" and c[k] != DP.DEFAULTS[k]}


def test_an_abort_beyond_its_correction_limit_is_refused_at_load():
    with pytest.raises(ValueError, match="must not exceed"):
        DP.resolve({"abort_err_m": 0.30, "max_corr_m": 0.25})
    with pytest.raises(ValueError, match="must not exceed"):
        DP.resolve({"abort_err_z_m": 0.5})
    with pytest.raises(ValueError, match="must not exceed"):
        DP.resolve({"abort_err_yaw_deg": 12.0})          # > max_corr 10
    assert DP.resolve({"abort_err_m": None})["abort_err_m"] is None
    with pytest.raises(ValueError, match="agree_window_s"):
        DP.resolve({"agree_window_s": 0.0})


def test_blind_for_max_blind_s_ends_a_control_run():
    T = Truth(v_b=(0.03, 0.0, 0.0), T=300.0)
    pose = _pose(max_blind_s=5.0)
    f = Feed(T, pose)
    f.run(5.0)
    out = f.run(7.0, tags=False, engaged=True)
    faults = [(t, h["fault"]) for t, _m, h, _ in out if h["fault"]]
    assert faults and "without a tag in view" in faults[0][1], faults[:1]
    t0 = out[0][0]
    assert faults[0][0] - t0 == pytest.approx(5.0, abs=0.2)
    assert len([x for x in pose.faults if "without a tag in view" in x[1]]) \
        == 1                                       # meta: once per onset
    # shadow: the same stop is REPORTED (the run a control session would
    # have ended — the worker logs it so) and the pose dead-reckons on
    pose_s = _pose(mode="shadow", max_blind_s=5.0)
    f_s = Feed(T, pose_s)
    f_s.run(5.0)
    out_s = f_s.run(7.0, tags=False, engaged=True)
    fs = [h["fault"] for _t, _m, h, _ in out_s if h["fault"]]
    assert fs and "without a tag in view" in fs[0], fs[:1]
    assert all(h["ok"] for _t, _m, h, _ in out_s)
    assert len([x for x in pose_s.faults
                if "without a tag in view" in x[1]]) == 1


def test_a_dvl_faster_than_the_mission_while_dead_reckoning_is_a_fault():
    T = Truth(v_b=(0.02, 0.0, 0.0))
    pose = _pose()
    f = Feed(T, pose)
    f.run(5.0)
    f.dvl_bias = np.array([0.40, 0.0])           # a false lock, 0.42 m/s
    faults = [h["fault"] for _t, _m, h, _ in f.run(1.0, tags=False,
                                                    engaged=True)
              if h["fault"]]
    assert faults and "false lock" in faults[0], faults[:1]


def test_a_dvl_gap_past_lost_s_is_a_hole_until_the_tags_re_anchor():
    T = Truth(v_b=(0.05, 0.0, 0.0))
    pose = _pose()
    f = Feed(T, pose)
    f.run(5.0)
    t_cut = f.t
    f.dvl_on = lambda tv: not (t_cut < tv < t_cut + 2.5)
    f.run(5.0, tags=False, engaged=True)          # the DVL is back after 2.5 s
    _m, h = f.tick(f.t + 0.05, tags=False, engaged=True)[0]
    assert not h["ok"] and "after the anchor" in h["why"], h["why"]
    assert "after the anchor" in pose.tag_refusal(False, f.t)
    f.t += 0.05
    out = f.run(1.0, tags=True, engaged=False)    # the tags re-anchor it
    assert out[-1][2]["state"] == DP.ST_TAG
    assert pose._xy_why() == ""
    assert pose.n_xy_holes == 1


def test_repeated_short_gaps_add_up_to_a_hole():
    """Each 0.9 s gap leaves ~0.45 s at 5 cm/s unintegrated (~2 cm): the
    budget (here 4 cm) is spent by the second one."""
    T = Truth(v_b=(0.05, 0.0, 0.0), T=300.0)
    pose = _pose(max_unobserved_m=0.04)
    f = Feed(T, pose)
    f.run(5.0)
    t0 = f.t
    # every 2 s the DVL drops for 0.9 s: 0.45 s unintegrated each time
    f.dvl_on = lambda tv: ((tv - t0) % 2.0) > 0.9
    out = f.run(8.0, tags=False, engaged=True)
    lost = [h["why"] for _t, _m, h, _ in out if not h["ok"]]
    assert lost and any("unintegrated motion" in w for w in lost), lost[:2]


def test_a_latched_dvl_ends_off_at_once_and_is_a_debounced_dropout_under_on():
    """OFF has nothing but the DVL: a latch ends it at once. ON flies the
    tags whenever they are fresh, and one late frame is not a lost
    localizer: the same latch is a 'stale' (debounced) dropout there — it
    ended the run on a single late frame (fix verification 2026-10-18)."""
    for on in (False, True):
        T = Truth(v_b=(0.05, 0.0, 0.0))
        pose = _pose(require_gyro_bias=False)
        f = Feed(T, pose)
        f.run(5.0)
        if not on:
            pose.set_tag_on(False, f.t, engaged=True)
        f.dvl_usable = lambda tv: False
        orig = pose.note_dvl

        def note(tv, v, ok, code="", *a, _o=orig, **k):   # latched OUT
            return _o(tv, v, ok, "latched" if not ok else code, *a, **k)
        pose.note_dvl = note
        out = f.run(0.3, tags=False, engaged=True)
        bad = [h for _t, _m, h, _ in out if not h["ok"]]
        assert bad and "latched" in bad[0]["why"], (on, bad[:1])
        assert ("stale" in bad[0]["why"]) is on, (on, bad[0]["why"])


def test_off_a_z_divergence_faults_too():
    """The Bar30 under the pose wanders off the tag z: z is in the abort."""
    T = Truth(v_b=(0.0, 0.0, 0.0))
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(5.0)
    pose.set_tag_on(False, f.t, engaged=True)
    real = f.imu

    def drifting(t):
        im = real(t)
        im.depth_sp2_m = float(im.depth_sp2_m) + 0.05 * (t - f.T.t0 - 5.0)
        return im
    f.imu = drifting
    out = f.run(8.0, engaged=True)
    faults = [(h["err_z_m"], h["fault"]) for _t, _m, h, _ in out if h["fault"]]
    assert faults and "diverged" in faults[0][1], faults[:1]
    assert abs(faults[0][0]) >= 0.20


def test_a_large_z_disagreement_while_engaged_is_slewed_not_a_fault():
    T = Truth(v_b=(0.0, 0.0, 0.0))
    pose = _pose(require_gyro_bias=False, abort_err_z_m=None)
    f = Feed(T, pose)
    f.run(5.0)
    pose.set_tag_on(False, f.t, engaged=True)
    real = f.imu

    def offset(t):
        im = real(t)
        im.depth_sp2_m = float(im.depth_sp2_m) + 0.3
        return im
    f.imu = offset
    f.run(1.0, engaged=True)
    pose.set_tag_on(True, f.t, engaged=True)
    out = f.run(1.0, engaged=True)
    # z is the Bar30 under the pose: 30 cm against the tags is their z
    # against it, not a drift — slewed and said, never a run's end (fix
    # verification 2026-10-18)
    assert not [h for _t, _m, h, _ in out if h["fault"]]
    assert [h for _t, _m, h, _ in out if "slewed in, not refused" in h["warn"]]
    assert pose.corrections[-1]["applied"] == "slew"


def test_tags_back_with_no_dead_reckoning_are_judged_against_the_last_pose():
    """The DVL died DURING a dropout: when the tags return there is no dead
    reckoning to compare with — the drift the controller was shown is
    slewed in (engaged), never stepped in (review 2026-10-18)."""
    T = Truth(v_b=(0.05, 0.0, 0.0), T=300.0)
    pose = _pose()
    f = Feed(T, pose, dvl_bias=(0.0, 0.0))
    f.run(5.0)
    f.dvl_bias = np.array([0.02, 0.0])            # a slow drift...
    f.run(5.0, tags=False, engaged=True)          # ...10 cm in the dropout
    t_cut = f.t
    f.dvl_on = lambda tv: tv < t_cut              # ...then the DVL dies
    f.dvl_bias = np.zeros(2)
    out = f.run(3.0, tags=lambda t: t > t_cut + 1.6, engaged=True)
    prev = None
    for t, m, h, _tm in out:
        if m is not None and prev is not None:
            step = float(np.hypot(*(m["eta"][:2] - prev)))
            assert step < 0.05 * 0.05 + 0.02 * 0.05 + 2e-3, (t - t_cut, step,
                                                              h["state"])
        prev = None if m is None else m["eta"][:2].copy()
    blind = [c for c in pose.corrections if c["kind"] == "blind"]
    assert blind and blind[-1]["applied"] == "slew", pose.corrections[-2:]


def test_a_jump_right_after_an_accepted_correction_is_still_held():
    T, pose, f = _drift_then_on(engaged=True, dvl_bias=(0.01, 0.0),
                                seconds=10.0)
    f.dvl_bias = np.zeros(2)
    f.tag_dt = 0.05                      # a fix EVERY tick: the window is one
    pose.set_tag_on(True, f.t, engaged=True)
    n0 = len(pose.corrections)
    while len(pose.corrections) == n0:
        f.run(0.05, engaged=True)
    # the very NEXT fix after the acceptance jumps 0.8 m (review 2026-10-18:
    # the reference used to be None for that one fix, and it went straight in)
    tc0 = f._fix_t
    f.tag_jump = lambda t: (np.array([0.8, 0.0])
                            if tc0 < f._fix_t <= tc0 + f.tag_dt + 1e-9
                            else np.zeros(2))
    out = f.run(1.0, engaged=True)
    # what remains is the slew of the ~10 cm correction, never the jump
    assert max(_xy_err(T, t, m) for t, m, _h, _ in out) < 0.12


def test_off_is_refused_while_a_correction_is_still_slewing():
    T, pose, f = _drift_then_on(engaged=True, dvl_bias=(0.01, 0.0),
                                seconds=15.0)
    f.dvl_bias = np.zeros(2)
    pose.set_tag_on(True, f.t, engaged=True)
    f.run(0.6, engaged=True)
    assert pose.state == DP.ST_BLEND
    why = pose.tag_refusal(False, f.t)
    assert "still being applied" in why, why


def test_the_off_refusal_names_each_axis_still_slewing():
    """The z and yaw slews outlast the xy one: a refusal that said
    "0.0 cm left" gave the operator no reason (demo e2e, 2026-10-18)."""
    T, pose, f = _drift_then_on(engaged=True, dvl_bias=(0.0, 0.0),
                                seconds=2.0)
    pose.set_tag_on(True, f.t, engaged=True)
    f.run(3.0, engaged=True)
    pose.off[:] = [0.0, 0.0, 0.02, math.radians(2.0)]
    why = pose.tag_refusal(False, f.t)
    assert "(2.00 deg / 2.0 cm z left)" in why, why
    pose.off[:] = [0.03, 0.0, 0.0, 0.0]
    assert "(3.0 cm left)" in pose.tag_refusal(False, f.t)
    pose.off[:] = 0.0
    assert "still being applied" not in pose.tag_refusal(False, f.t)


def test_on_is_refused_engaged_when_it_would_end_the_run():
    T, pose, f = _drift_then_on(engaged=True, dvl_bias=(0.03, 0.0),
                                seconds=9.0, abort_err_m=None)
    f.run(0.1, engaged=True)
    assert pose.err_m is not None and pose.err_m > 0.25
    why = pose.tag_refusal(True, f.t, engaged=True)
    assert "would end the run" in why, why
    assert pose.tag_refusal(True, f.t, engaged=False) == ""


def test_shadow_records_a_refused_correction_once_and_snaps():
    shadow = DP.DvlPose({"bias_min_span_s": 20.0, "bias_min_n": 50,
                         "abort_err_m": None}, mode="shadow",
                        nudot_source="fd")
    T2 = Truth(v_b=(0.05, 0.0, 0.0), T=300.0)
    f2 = Feed(T2, shadow)
    f2.run(5.0)
    shadow.set_tag_on(False, f2.t, engaged=True)
    f2.dvl_bias = np.array([0.03, 0.0])
    f2.run(12.0, engaged=True)
    f2.dvl_bias = np.zeros(2)
    shadow.set_tag_on(True, f2.t, engaged=True)
    f2.run(3.0, engaged=True)
    refused = [c for c in shadow.corrections if c["applied"].startswith(
        "refused")]
    assert len(refused) == 1 and "shadow" in refused[0]["applied"]
    assert shadow.state == DP.ST_TAG


def test_a_lasting_divergence_is_one_fault_record():
    T, pose, f = _drift_then_on(engaged=True, dvl_bias=(0.05, 0.0),
                                seconds=1.0)
    f.run(12.0, engaged=True)
    div = [w for _t, w in pose.faults if "diverged" in w]
    assert len(div) == 1, len(div)


def test_a_nan_roll_does_not_poison_the_heading():
    T = Truth(v_b=(0.05, 0.0, 0.0), omega=0.05)
    pose = _pose()
    f = Feed(T, pose)
    f.run(5.0)
    real = f.imu

    def once(t, _n=[0]):
        im = real(t)
        if _n[0] == 0:
            im.roll = float("nan")
        _n[0] += 1
        return im
    f.imu = once
    f.run(0.1)
    f.imu = real
    assert math.isfinite(pose.head.raw)
    out = f.run(3.0, tags=False, engaged=True)
    assert out[-1][2]["ok"] and _xy_err(T, out[-1][0], out[-1][1]) < 0.01


def test_a_jump_that_began_before_a_dropout_does_not_outlive_it():
    T = Truth(v_b=(0.05, 0.0, 0.0), T=300.0)
    pose = _pose()
    f = Feed(T, pose)
    f.run(5.0)
    t_j = f.t + 0.3
    f.tag_jump = lambda t: np.array([0.5, 0.0]) if t_j <= t < t_j + 0.07 \
        else np.zeros(2)
    f.run(0.4, engaged=True)
    f.tag_jump = lambda t: np.zeros(2)
    f.dvl_bias = np.array([0.01, 0.0])
    f.run(6.0, tags=False, engaged=True)          # pending must not survive
    assert pose._pending == [] and pose._pending_kind == ""
    f.dvl_bias = np.zeros(2)
    out = f.run(1.0, engaged=True)
    states = [h["state"] for _t, _m, h, _ in out]
    assert states[0] == DP.ST_DR                  # a re-acquisition, agreed
    assert pose.corrections[-1]["kind"] == "reacquire"



# =============================================================================
# fix verification 2026-10-18 (round 3): each test below fails on the code
# that round verified and passes on its fix
# =============================================================================
def _flicker(t0, on_s, period_s):
    """Tags in view ``on_s`` of every ``period_s``."""
    return lambda t: ((t - t0) % period_s) < on_s


def _off_tick(f, seconds):
    """Engaged ticks until ``seconds`` have passed (one at a time)."""
    out = []
    for _ in range(int(round(seconds / 0.05))):
        out += f.run(0.05, engaged=True)
    return out


def test_a_flickering_view_still_re_anchors_the_pose_under_on():
    """In view 0.25 s of every second (3-4 fixes a burst, never anchor_n):
    the agreement collects its fixes across the gaps. Cleared per tag-less
    tick, the pose dead-reckoned uncorrected for as long as the view stayed
    like that — 4 mm/s of DVL error = 24 cm in a minute, no stop."""
    T = Truth(v_b=(0.05, 0.0, 0.0), T=300.0)
    pose = _pose()
    f = Feed(T, pose)
    f.run(5.0)
    f.dvl_bias = np.array([0.004, 0.0])
    t0 = f.t
    out = f.run(60.0, tags=_flicker(t0, 0.25, 1.0), engaged=True)
    re = [c for c in pose.corrections
          if c["kind"] == "reacquire" and c["t"] > t0]
    assert len(re) > 10, len(re)
    worst = max(_xy_err(T, t, m) for t, m, h, _tm in out if m is not None)
    assert worst < 0.05, worst
    assert not [h for _t, _m, h, _tm in out if h["fault"]]


def test_a_flickering_view_still_ends_an_off_run_that_diverged():
    """OFF, in view 0.6 s of every 1.4 s, the DVL 2 cm/s wrong: the abort's
    evidence is the fixes over the line, not an unbroken second of them —
    reset per tag-less tick, the abort never fired (1.18 m, no stop)."""
    T = Truth(v_b=(0.05, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(5.0)
    pose.set_tag_on(False, f.t, engaged=True)
    f.dvl_bias = np.array([0.02, 0.0])
    t0 = f.t
    out = f.run(40.0, tags=_flicker(t0, 0.6, 1.4), engaged=True)
    faults = [t for t, _m, h, _tm in out if "diverged" in h["fault"]]
    assert faults, max((h["err_m"] or 0.0) for _t, _m, h, _tm in out)
    # 25 cm at 2 cm/s is 12.5 s; the hold and the view's gaps add a little
    assert faults[0] - t0 < 16.0, faults[0] - t0
    assert len([x for x in pose.faults if "diverged" in x[1]]) == 1


def test_two_stray_frames_far_apart_are_not_a_divergence():
    """The divergence evidence survives gaps, but a stop needs anchor_n
    fixes over the line: two single 50 cm frames 4 s apart are not one."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=200.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(5.0)
    pose.set_tag_on(False, f.t, engaged=True)
    t0 = f.t
    f.tag_jump = lambda t: np.array([0.5, 0.0])

    def seen(t):
        return abs(t - (t0 + 1.0)) < 0.026 or abs(t - (t0 + 5.0)) < 0.026
    out = f.run(8.0, tags=seen, engaged=True)
    assert sum(1 for _t, _m, h, _tm in out if h["err_m"] is not None) == 2
    assert not [h for _t, _m, h, _tm in out if "diverged" in h["fault"]]


def test_the_unobserved_budget_is_judged_while_the_gap_is_still_open():
    """Two 0.9 s gaps commit ~4.5 cm of a 6 cm budget at 5 cm/s; the third
    gap ends the dead reckoning when the budget runs out — not when the
    sample that closes it arrives (a HELD pose flew up to lost_s - hold_s
    past it)."""
    T = Truth(v_b=(0.05, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False, max_unobserved_m=0.06)
    f = Feed(T, pose)
    f.run(5.0)
    pose.set_tag_on(False, f.t, engaged=True)
    t0 = f.t
    gaps = [(t0 + 1.0, t0 + 1.85), (t0 + 3.0, t0 + 3.85),
            (t0 + 5.0, t0 + 6.4)]
    f.dvl_on = lambda tv: not any(a < tv < b for a, b in gaps)
    out = f.run(8.0, tags=False, engaged=True)
    lost = [(t, h["why"]) for t, m, h, _tm in out if m is None]
    assert lost and "still open" in lost[0][1], lost[:1]
    assert lost[0][0] < t0 + 6.0, lost[0][0] - t0
    # ...and it is a hole: the DVL coming back does not resume the track
    assert not out[-1][2]["ok"] and pose.n_xy_holes == 1


def test_a_dvl_gap_longer_than_the_gyro_history_is_a_hole_too():
    """The heading history holds ~60 s; a gap reaching back past it used to
    skip the hole rule together with the integration, and the track
    resumed from before the outage with no hole recorded."""
    T = Truth(v_b=(0.05, 0.0, 0.0), T=400.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(5.0)
    t_last = pose.t_p
    for _ in range(int(70.0 / 0.05)):      # 70 s of gyro, no DVL, no step
        f.t += 0.05
        pose.note_imu(f.imu(f.t))
    assert pose._psi(t_last) is None       # the history no longer reaches it
    pose.note_dvl(t_last + 70.0, T.v_b, True, "", T.phi, T.theta, 0.35,
                  t_rx=t_last + 70.1)
    assert pose.n_xy_holes == 1
    # (measured from the track's anchor, a tick after the newest sample)
    assert re.search(r"a DVL gap of (69|70)\.\d s", pose._xy_why()), \
        pose._xy_why()
    assert "after the anchor" in pose.tag_refusal(False, f.t)


def test_a_slew_that_rides_out_a_short_dead_reckoning_loss_is_not_a_jump():
    """The Bar30 stale for 3 ticks mid-slew, tags in view: the tag state
    (+ the slewing correction) passes through, and when the dead reckoning
    is back its difference is -off — the reference used to be 0, which
    read the slew's remainder as a jump and left the tag state."""
    T, pose, f = _drift_then_on(engaged=True, dvl_bias=(0.02, 0.0),
                                seconds=9.0)
    f.dvl_bias = np.zeros(2)
    pose.set_tag_on(True, f.t, engaged=True)
    n0 = len(pose.corrections)
    while len(pose.corrections) == n0:
        f.run(0.05, engaged=True)
    assert pose.corrections[-1]["err_m"] > 0.12        # > jump_m
    f.run(0.5, engaged=True)
    assert pose.state == DP.ST_BLEND
    jumps0, real, t_s = pose.n_jumps_held, f.imu, f.t

    def stale(t):
        im = real(t)
        if t_s < t <= t_s + 0.16:
            im.depth_sp2_m = im.t_sp2 = None
            im.depth_m = im.t_baro = None
        return im
    f.imu = stale
    out = f.run(2.0, engaged=True)
    f.imu = real
    assert pose.n_jumps_held == jumps0
    states = [h["state"] for _t, _m, h, _tm in out]
    assert set(states) <= {DP.ST_BLEND, DP.ST_TAG}, states


def test_a_tag_dropout_with_no_dead_reckoning_shown_passes_the_tags_back():
    """The DVL out, the tags the pose: a short tag dropout has no drift to
    judge, so the returning tags are the pose at once — waiting for
    anchor_n of them ended runs the station without --dvl_pose survives."""
    T = Truth(v_b=(0.05, 0.0, 0.0), T=300.0)
    pose = _pose()
    f = Feed(T, pose)
    f.run(5.0)
    t_cut = f.t
    f.dvl_on = lambda tv: tv < t_cut               # the DVL gone for good
    f.run(3.0, engaged=True)
    t_d = f.t
    out = f.run(1.0, tags=lambda t: not (t_d < t <= t_d + 0.3),
                engaged=True)
    back = [h["state"] for t, _m, h, _tm in out if t > t_d + 0.3 + 1e-9]
    assert back and back[0] == DP.ST_TAG, back[:3]
    assert not [c for c in pose.corrections if c["kind"] == "blind"]


def test_a_blind_re_acquisition_is_slewed_across_the_lost_gap():
    """After a dead-reckoned stretch, the DVL dies: the first pose after the
    tags return is where the LAST one stood (the drift then slews), whatever
    the ticks in between showed. (At rest, so the drift is all of it.)"""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose()
    f = Feed(T, pose)
    f.run(5.0)
    f.dvl_bias = np.array([0.02, 0.0])
    f.run(5.0, tags=False, engaged=True)           # ~10 cm of drift shown
    t_cut = f.t
    f.dvl_on = lambda tv: tv < t_cut
    f.dvl_bias = np.zeros(2)
    out = f.run(4.0, tags=lambda t: t > t_cut + 1.6, engaged=True)
    before = [m for t, m, h, _tm in out if m is not None and t <= t_cut + 1.6]
    after = [m for t, m, h, _tm in out if m is not None and t > t_cut + 1.6]
    assert before and after
    step = float(np.hypot(*(after[0]["eta"][:2] - before[-1]["eta"][:2])))
    assert step < 0.01, step
    blind = [c for c in pose.corrections if c["kind"] == "blind"]
    assert blind and blind[-1]["applied"] == "slew" \
        and blind[-1]["err_m"] > 0.05, blind[-1:]


def test_a_blind_re_acquisition_too_big_to_slew_ends_a_control_run():
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose()
    f = Feed(T, pose)
    f.run(5.0)
    f.dvl_bias = np.array([0.06, 0.0])
    f.run(5.0, tags=False, engaged=True)           # ~30 cm of drift shown
    t_cut = f.t
    f.dvl_on = lambda tv: tv < t_cut
    out = f.run(4.0, tags=lambda t: t > t_cut + 1.6, engaged=True)
    faults = [h["fault"] for _t, _m, h, _tm in out if h["fault"]]
    assert faults and "(blind" in faults[0], faults[:1]
    assert not [m for t, m, h, _tm in out if m is not None
                and t > t_cut + 1.6]               # never stepped in


def test_a_slew_cut_short_by_a_dropout_finishes_on_the_track():
    """The correction is IN the track (anchored to tag + off) when the tags
    drop: it keeps slewing ON THE TRACK, so when off reaches 0 nothing is
    left unapplied — decaying off alone let OFF through with 14.5 cm still
    in the track; while some is left, OFF stays refused."""
    T, pose, f = _drift_then_on(engaged=True, dvl_bias=(0.02, 0.0),
                                seconds=8.0)
    f.dvl_bias = np.zeros(2)
    pose.set_tag_on(True, f.t, engaged=True)
    n0 = len(pose.corrections)
    while len(pose.corrections) == n0:
        f.run(0.05, engaged=True)
    f.run(0.5, engaged=True)
    assert pose.corr_m() > 0.10
    out = f.run(2.0, tags=False, engaged=True)     # part of the slew
    assert "cm left" in pose.tag_refusal(False, f.t)
    # the pose moves at the slew rate on the dead-reckoned ticks too
    steps = [float(np.hypot(*(b[1]["eta"][:2] - a[1]["eta"][:2])))
             for a, b in zip(out, out[1:])]
    assert max(steps) < 0.05 * 0.05 + 0.02 * 0.05 + 1e-3, max(steps)
    out = f.run(7.0, tags=False, engaged=True)     # longer than the slew
    assert pose.corr_m() == 0.0
    assert _xy_err(T, out[-1][0], out[-1][1]) < 0.01
    assert pose.tag_refusal(False, f.t) == ""


def test_off_is_refused_while_fixes_are_still_agreeing():
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(5.0)
    f.tag_jump = lambda t: np.array([0.15, 0.0])     # a new level
    for _ in range(40):
        f.run(0.05, engaged=True)
        if 0 < len(pose._pending) < 5:
            break
    assert 0 < len(pose._pending) < 5, len(pose._pending)
    why = pose.tag_refusal(False, f.t)
    assert "waiting to agree" in why, why


def test_a_jump_still_held_when_the_tags_drop_comes_back_judged():
    """A held jump does not survive a dropout as a JUMP (its 'back at the
    old level' pass-through would skip the judgement): it turns into a
    re-acquisition, keeping the fixes it has."""
    T = Truth(v_b=(0.05, 0.0, 0.0), T=300.0)
    pose = _pose()
    f = Feed(T, pose)
    f.run(5.0)
    f.tag_jump = lambda t: np.array([0.5, 0.0])
    for _ in range(20):
        f.run(0.05, engaged=True)
        if pose._pending_kind == "jump":
            break
    assert pose._pending_kind == "jump"
    n = len(pose._pending)
    f.tag_jump = lambda t: np.zeros(2)
    f.dvl_bias = np.array([0.01, 0.0])
    f.run(0.05, tags=False, engaged=True)
    assert pose._pending_kind == "reacquire" and len(pose._pending) == n
    f.run(4.0, tags=False, engaged=True)
    f.dvl_bias = np.zeros(2)
    out = f.run(1.0, engaged=True)
    assert out[0][2]["state"] == DP.ST_DR            # judged, not passed
    assert pose.corrections[-1]["kind"] == "reacquire"


def test_pending_fixes_expire_after_agree_window_s():
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(agree_window_s=2.0)
    f = Feed(T, pose)
    f.run(5.0)
    f.tag_jump = lambda t: np.array([0.3, 0.0])
    for _ in range(20):
        f.run(0.05, engaged=True)
        if len(pose._pending) >= 2:
            break
    assert len(pose._pending) >= 2
    f.run(3.0, tags=False, engaged=True)            # older than the window
    assert pose._pending_live(f.t) == 0
    assert "waiting to agree" not in pose.tag_refusal(False, f.t)
    f.run(0.05, engaged=True)                        # the next fix prunes them
    assert len(pose._pending) == 1


def test_the_dead_reckoning_z_never_switches_depth_source_mid_track():
    """Anchored on the Bar30: a tick with only the GPI depth (another sensor,
    another offset) is no dead-reckoned z, until the tags re-anchor on it."""
    T = Truth(v_b=(0.05, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(5.0)
    real = f.imu

    def gpi_only(t):
        im = real(t)
        im.depth_sp2_m, im.t_sp2 = None, None
        im.depth_m = float(im.depth_m) + 0.4
        return im
    f.imu = gpi_only
    out = f.run(1.0, tags=False, engaged=True)
    bad = [h["why"] for _t, _m, h, _tm in out if not h["ok"]]
    assert bad and "depth source changed (sp2 -> gpi)" in bad[0], bad[:1]
    assert "stale" in bad[0]                          # debounced, not a stop
    f.run(1.0, tags=True, engaged=False)              # the tags re-anchor
    out = f.run(1.0, tags=False, engaged=False)
    assert out[-1][2]["ok"], out[-1][2]["why"]
    assert out[-1][2]["z_src"] == "gpi"


def test_a_one_frame_z_jump_never_reaches_the_pose():
    """A PnP z outlier (in the fix, so in the state built from it) is held
    like any jump."""
    T = Truth(v_b=(0.05, 0.0, 0.0), T=300.0)
    pose = _pose()
    f = Feed(T, pose)
    f.run(5.0)
    real_tag, t_j = f.tag, f.t + 0.3

    def ztag(t, fresh=True):
        meas, fix = real_tag(t, fresh)
        if t_j <= t < t_j + 0.07:
            meas["eta"][2] += 0.15
            fix.p_ned = np.array(fix.p_ned, float)
            fix.p_ned[2] += 0.15
        return meas, fix
    f.tag = ztag
    out = f.run(1.0, engaged=True)
    worst = max(abs(m["eta"][2] - T.z(t)) for t, m, h, _tm in out
                if m is not None)
    assert worst < 0.02, worst
    assert pose.n_jumps_held >= 1


def test_agreement_needs_z_and_yaw_to_agree_too():
    pose = _pose()
    a = np.array([0.10, 0.00, 0.00, 0.0])
    pose._pending = [a.copy() for _ in range(5)]
    pose._pending_t = [0.0] * 5
    assert pose._agree()
    pose._pending[2] = a + np.array([0.0, 0.0, 0.15, 0.0])
    assert not pose._agree()
    pose._pending[2] = a + np.array([0.0, 0.0, 0.0, math.radians(8.0)])
    assert not pose._agree()
    r = np.array([1.0, 2.0, 0.8, 0.0, 0.0, 0.3])
    pose._pending = [r.copy() for _ in range(5)]
    assert pose._agree_raw()
    pose._pending[1] = r + np.array([0.0, 0.0, 0.15, 0.0, 0.0, 0.0])
    assert not pose._agree_raw()


def test_on_is_refused_engaged_on_a_yaw_or_z_beyond_a_correction():
    T, pose, f = _drift_then_on(engaged=True, dvl_bias=(0.0, 0.0),
                                seconds=2.0)
    pose.err_m, pose.err_yaw_deg, pose.err_z_m = 0.0, 12.0, 0.0
    why = pose.tag_refusal(True, f.t, engaged=True)
    assert "more than a correction may be" in why, why
    assert pose.tag_refusal(True, f.t, engaged=False) == ""
    # z refuses nothing: it is slewed whatever its size
    pose.err_m, pose.err_yaw_deg, pose.err_z_m = 0.0, 0.0, 0.25
    assert pose.tag_refusal(True, f.t, engaged=True) == ""


def test_the_reference_is_set_at_the_first_anchor_and_after_a_dr_loss():
    """A single 0.83 m frame right after the FIRST anchor, and right after a
    tick with no dead reckoning, is held like any other jump: both need the
    reference those ticks set."""
    for after in ("anchor", "dr_loss"):
        T = Truth(v_b=(0.05, 0.0, 0.0), T=300.0)
        pose = _pose()
        f = Feed(T, pose)
        f.tag_dt = 0.05                              # a fix every tick
        while not pose.anchored:
            f.run(0.05)
        if after == "dr_loss":
            f.run(0.5)
            real, t_s = f.imu, f.t

            def stale(t, _r=real, _t=t_s):
                im = _r(t)
                if _t < t <= _t + 0.06:
                    im.depth_sp2_m = im.t_sp2 = None
                    im.depth_m = im.t_baro = None
                return im
            f.imu = stale
            f.run(0.05, engaged=True)
            f.imu = real
        tc0 = f._fix_t
        f.tag_jump = lambda t, _c=tc0: (np.array([0.83, 0.0])
                                        if _c < f._fix_t <= _c + 0.05 + 1e-9
                                        else np.zeros(2))
        out = f.run(1.0, engaged=True)
        worst = max(_xy_err(T, t, m) for t, m, h, _tm in out
                    if m is not None)
        assert worst < 0.1, (after, worst)


def test_a_dvl_refused_on_evidence_is_not_extrapolated_past_the_refusal():
    """A disagreement hold (dis_*) stops the last good velocity at the
    refused sample, not hold_s later (DvlBlend's own rule)."""
    T = Truth(v_b=(0.1, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(5.0)
    pose.set_tag_on(False, f.t, engaged=True)
    f.run(1.0, tags=False, engaged=True)
    t_r = f.t
    f.dvl_usable = lambda tv: tv < t_r
    orig = pose.note_dvl

    def note(tv, v, ok, code="", *a, **k):
        return orig(tv, v, ok, "dis_align" if not ok else code, *a, **k)
    pose.note_dvl = note
    out = f.run(0.6, tags=False, engaged=True)
    t_stop = pose._stop_t
    assert t_stop is not None
    xs = [m["eta"][0] for t, m, h, _tm in out if m is not None]
    # frozen from the refused sample on: no more than one sample period of
    # motion past the last good one (hold_s would be 0.45 s = 4.5 cm)
    assert max(xs) - xs[0] < 0.1 * 0.25, (max(xs) - xs[0])


# =============================================================================
# fix verification 2026-10-18 (round 4): each test below fails on the code
# that round verified and passes on its fix
# =============================================================================
def test_a_hole_from_before_the_anchor_does_not_block_a_new_one():
    """A closing sample whose gap the tags covered can leave a hole stamped
    BEFORE the newest anchor; it is spent — it used to block every later
    hole until the next anchor."""
    T = Truth(v_b=(0.05, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(5.0)
    pose._xy_hole_t = pose._t_ta - 1.0             # a spent hole
    pose._xy_hole_why = "an old one"
    assert pose._xy_why() == ""
    n0 = pose.n_xy_holes
    t_last = pose.t_p
    pose.note_dvl(t_last + 3.0, T.v_b, True, "", T.phi, T.theta, 0.35,
                  t_rx=t_last + 3.1)
    assert pose.n_xy_holes == n0 + 1
    assert "a DVL gap of" in pose._xy_why(), pose._xy_why()


def test_glimpses_that_show_the_pose_wrong_do_not_keep_the_stops_away():
    """3-4 fixes every 15 s, the DVL 14 deg off (~2.4 cm/s across): each
    glimpse used to refresh 'a tag in view', the agreement and the abort
    evidence expired between them, and the pose walked metres off with no
    stop. A fix outside the abort limits vouches for nothing: blind counts.
    (max_blind_s set: it is off by default since 2026-10-18.)"""
    T = Truth(v_b=(0.10, 0.0, 0.0), T=400.0)
    pose = _pose(require_gyro_bias=False, max_blind_s=20.0)
    f = Feed(T, pose)
    f.run(5.0)
    f.dvl_bias = np.array([0.0, 0.024])
    t0 = f.t
    out = f.run(120.0, tags=_flicker(t0, 0.25, 15.0), engaged=True)
    faults = [(t, h["fault"]) for t, _m, h, _tm in out if h["fault"]]
    assert faults, max((h["err_m"] or 0.0) for _t, _m, h, _tm in out)
    worst = max(_xy_err(T, t, m) for t, m, h, _tm in out
                if m is not None and t <= faults[0][0])
    assert worst < 0.9, (worst, faults[0])


def test_a_bridged_tag_z_artifact_is_not_a_jump_and_not_a_fault():
    """The tag STATE's z between fixes is the fix's z + the tag FD heave x
    age: one 2-tag fix put 18 cm into it on 10-17. The fix itself is clean,
    so the pose neither holds a 'jump' when the next fix returns nor ends
    the run on a z correction (13 of the 111 ON runs of a replay did)."""
    T = Truth(v_b=(0.05, 0.0, 0.0), T=300.0)
    pose = _pose()
    f = Feed(T, pose)
    f.run(5.0)
    real_tag = f.tag
    k = {"n": 0}

    def bridged(t, fresh=True):
        meas, fix = real_tag(t, fresh)
        if k["n"] < 7:                              # 0.35 s, no new fix
            meas["eta"][2] += 0.02 * (k["n"] + 1)
        return meas, fix
    f.tag, f.tag_dt = bridged, 0.40                # no capture in the ramp
    jumps0 = pose.n_jumps_held
    states = []
    for _ in range(7):
        states.append(f.run(0.05, engaged=True)[0][2]["state"])
        k["n"] += 1
    f.tag, f.tag_dt = real_tag, 1.0 / 15.0
    out = f.run(2.0, engaged=True)
    states += [h["state"] for _t, _m, h, _tm in out]
    assert set(states) <= {DP.ST_TAG, DP.ST_BLEND}, states
    assert pose.n_jumps_held == jumps0
    assert not [h for _t, _m, h, _tm in out if h["fault"]]


def test_a_dropout_drops_the_bridged_z_onto_the_bar30_truth():
    """Tags drop while the shown (bridged) z carries 8 cm the dead
    reckoning's (the Bar30 on the FIX's offset) does not: the dead reckoning
    is the truth the bridged z drifted from, and the pose goes there and
    stays (round 4 slewed the split in through off and that ratcheted —
    fix verification 2026-10-18)."""
    T = Truth(v_b=(0.05, 0.0, 0.0), T=300.0)
    pose = _pose()
    f = Feed(T, pose)
    f.run(5.0)
    real_tag = f.tag

    def bridged(t, fresh=True):
        meas, fix = real_tag(t, fresh)
        meas["eta"][2] += 0.08
        return meas, fix
    f.tag, f.tag_dt = bridged, 0.40
    f.run(0.1, engaged=True)
    f.tag, f.tag_dt = real_tag, 1.0 / 15.0
    out = f.run(6.0, tags=False, engaged=True)
    zs = [(t, float(m["eta"][2])) for t, m, h, _tm in out if m is not None]
    assert max(abs(z - T.z(t)) for t, z in zs) < 0.01, \
        max(abs(z - T.z(t)) for t, z in zs)
    assert abs(pose.off[2]) < 1e-9


def test_glitches_under_the_jump_gate_do_not_ratchet_the_pose():
    """One fix a second 7 cm off (in z, or in x): each passes the gate (it
    is under jump_*), and its RETURN must pass too — compared against what
    the next difference will be once re-anchored, -off, not against the
    glitch itself (that made every return a 2x 'jump', held the glitch
    level and walked the flown z metres away, fix verification 2026-10-18)."""
    for axis in (2, 0):
        T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
        pose = _pose()
        f = Feed(T, pose)
        f.tag_dt = 0.05                              # a fix every tick
        f.run(5.0)
        real_tag = f.tag
        t0 = f.t

        def glitch(t, fresh=True, _r=real_tag, _a=axis, _t0=t0):
            meas, fix = _r(t, fresh)
            if ((t - _t0) % 1.0) < 0.05:
                meas["eta"][_a] += 0.07
                fix.p_ned = np.array(fix.p_ned, float)
                fix.p_ned[_a] += 0.07
            return meas, fix
        f.tag = glitch
        jumps0 = pose.n_jumps_held
        out = f.run(30.0, engaged=True)
        worst = max(abs(float(m["eta"][axis]) - T.eta(t)[axis])
                    for t, m, h, _tm in out if m is not None)
        assert worst < 0.08, (axis, worst)           # the glitch itself, once
        assert pose.n_jumps_held == jumps0, (axis, pose.n_jumps_held - jumps0)


def test_the_heave_is_no_finite_difference_over_a_burst_or_a_spike():
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose()
    f = Feed(T, pose)
    f.run(2.0)
    vz0 = pose.vz
    pose._note_depth(T.depth(f.t) + 0.002, f.t + 0.001, "sp2")  # 1 ms later
    assert abs(pose.vz - vz0) < 1e-9                 # a burst: skipped
    pose._note_depth(T.depth(f.t) + 1.0, f.t + 0.101, "sp2")    # 1 m in 0.1 s
    assert abs(pose.vz) <= 0.3 * 0.5 + 1e-9          # clamped at 0.5 m/s


def test_the_budget_is_motion_a_station_hold_pays_little_for_a_gap():
    for v, gaps, hole in ((0.0, 6, False), (0.10, 3, True)):
        T = Truth(v_b=(v, 0.0, 0.0), T=300.0)
        pose = _pose(require_gyro_bias=False)
        f = Feed(T, pose)
        f.run(5.0)
        pose.set_tag_on(False, f.t, engaged=True)
        t0 = f.t
        g = [(t0 + 1.0 + 2.0 * k, t0 + 1.9 + 2.0 * k) for k in range(gaps)]
        f.dvl_on = lambda tv, _g=g: not any(a < tv < b for a, b in _g)
        f.run(2.0 * gaps + 1.0, tags=False, engaged=True)
        assert (pose.n_xy_holes > 0) is hole, (v, pose.unobserved_m)


def test_a_gap_the_tags_covered_is_not_charged_to_the_next():
    """ON, tags in view, the DVL away 1.3 s (the tags re-anchor the track
    meanwhile), then a dropout with a short gap: nothing of the covered gap
    is unobserved motion."""
    T = Truth(v_b=(0.05, 0.0, 0.0), T=300.0)
    pose = _pose()
    f = Feed(T, pose)
    f.run(5.0)
    t_cut = f.t
    f.dvl_on = lambda tv: not (t_cut < tv < t_cut + 1.3)
    f.run(1.25, engaged=True)                      # tags in view all along
    f.run(1.0, tags=False, engaged=True)           # the gap closes here
    assert pose.unobserved_m < 0.01, pose.unobserved_m
    assert pose.n_xy_holes == 0


def test_a_z_pinned_on_the_gpi_stays_on_it_when_the_bar30_returns():
    T = Truth(v_b=(0.05, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    real = f.imu

    def gpi_only(t):
        im = real(t)
        im.depth_sp2_m, im.t_sp2 = None, None
        return im
    f.imu = gpi_only
    f.run(5.0)                                     # anchored on the GPI
    assert pose._z_src == "gpi"
    pose.set_tag_on(False, f.t, engaged=True)
    f.imu = real                                   # the Bar30 is back
    out = f.run(2.0, tags=False, engaged=True)
    assert all(h["ok"] for _t, _m, h, _tm in out), out[-1][2]["why"]
    assert out[-1][2]["z_src"] == "gpi"


def test_a_z_slew_with_no_depth_on_the_tag_ticks_leaves_no_step():
    """A 15 cm z correction slews while the depth is stale (tags in view):
    the offset follows off's z, so the dead reckoning that takes over is
    where the pose stood, and nothing is left unapplied when off is 0."""
    T = Truth(v_b=(0.05, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(5.0)
    pose.off[2] = 0.15                               # a z correction slewing
    real = f.imu
    f.imu = lambda t: VehicleImu(**{**real(t).__dict__, "depth_sp2_m": None,
                                    "t_sp2": None, "depth_m": None,
                                    "t_baro": None})
    out = f.run(3.0, engaged=True)                   # tags, no depth
    z_tag = float(out[-1][1]["eta"][2])
    f.imu = real                                     # the depth back AS the
    out = f.run(10.0, tags=False, engaged=True)      # tags drop: no anchor
    zs = [float(m["eta"][2]) for _t, m, h, _tm in out if m is not None]
    assert abs(zs[0] - z_tag) < 0.01, (zs[0], z_tag)
    # (the offset that followed off's z puts the dead reckoning on the
    # truth; left behind, 6 cm of the correction stayed in for good)
    assert abs(zs[-1] - T.z(out[-1][0])) < 0.01, zs[-1] - T.z(out[-1][0])


def test_disengaged_stops_are_not_recorded_as_a_runs_end():
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(max_blind_s=5.0)
    f = Feed(T, pose)
    f.run(5.0)
    out = f.run(7.0, tags=False, engaged=False)
    assert [h for _t, _m, h, _tm in out if h["fault"]]   # START would refuse
    assert not pose.faults                               # ...no run ended


def test_the_reference_after_a_disengaged_dr_loss_is_the_snapped_level():
    """Disengaged, off is zeroed on the tick: the reference set there is 0,
    not the -off that tick began with (it read the snap as a jump)."""
    T, pose, f = _drift_then_on(engaged=True, dvl_bias=(0.02, 0.0),
                                seconds=9.0)
    f.dvl_bias = np.zeros(2)
    pose.set_tag_on(True, f.t, engaged=True)
    n0 = len(pose.corrections)
    while len(pose.corrections) == n0:
        f.run(0.05, engaged=True)
    assert pose.corr_m() > 0.1
    jumps0, real, t_s = pose.n_jumps_held, f.imu, f.t

    def stale(t, _r=real, _t=t_s):
        im = _r(t)
        if _t < t <= _t + 0.06:
            im.depth_sp2_m = im.t_sp2 = None
            im.depth_m = im.t_baro = None
        return im
    f.imu = stale
    f.run(0.05, engaged=False)                       # disengaged, DR lost
    f.imu = real
    f.run(1.0, engaged=False)
    assert pose.n_jumps_held == jumps0


def test_a_re_acquisition_the_dead_reckoning_drops_out_of_goes_on_blind():
    """ON, the DVL out (HELD), tags back after a gap: the re-acquisition
    collects fixes; when the DVL gap passes lost_s it goes on as a blind one
    WITH those fixes — starting over cost anchor_n more ticks of no pose."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose()
    f = Feed(T, pose)
    f.run(5.0)
    t_cut = f.t
    f.dvl_on = lambda tv: tv < t_cut                 # the DVL gone
    f.run(1.25, tags=False, engaged=True)            # HELD shown
    out = f.run(1.5, engaged=True)                   # tags back, 2-3 fixes
    #                                                  before lost_s
    no_pose = [t for t, m, h, _tm in out if m is None]
    assert no_pose and len(no_pose) <= 5, len(no_pose)   # (9 starting over)
    assert out[-1][2]["state"] == DP.ST_TAG


def test_the_heading_is_anchored_at_the_gyros_own_newest_moment():
    """The newest RAW_IMU value is one link latency old: anchored at t_now
    it put yaw rate x latency into the heading (1 deg at 20 deg/s, 50 ms)."""
    lat = 0.05
    om = math.radians(20.0)
    # (still for the first anchor — 5 raw poses must agree within 5 deg —
    # then a 20 deg/s vertex pivot that ENDS during the dead reckoning)
    T = Truth(v_b=(0.0, 0.0, 0.0),
              omega=lambda s: om if 4.0 < s < 7.0 else 0.0, T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    real = f.imu

    def late(t):
        im = real(t)
        g = T.gint_at(t - lat)
        im.gyro_int, im.t_gyro = tuple(g), t
        im.t_gyro_v = t - lat
        return im
    f.imu = late
    f.run(6.0)
    assert pose.anchored
    out = f.run(3.0, tags=False, engaged=True)      # the pivot ends at s 7
    t_stop = T.t0 + 7.0
    err = [abs(DP.wrap(m["eta"][5] - T.yaw(t))) for t, m, h, _tm in out
           if m is not None and not (t_stop <= t <= t_stop + 0.15)]
    # through the turn and after it; only the latency right after the stop
    # (the rate's change is that late) may carry up to rate x latency
    assert err and math.degrees(max(err)) < 0.3, math.degrees(max(err))


#: The queued-link tests run with the cap an operator may raise
#: gyro_late_max_s to (1.0 s, its value before round 8): they pin the queue
#: machinery that still flies under it. The default, 0.1 s, gives a link more
#: than 0.3 s late no dead reckoning at all
#: (test_a_link_more_than_0p3_s_late_has_no_dead_reckoning, round 8).
QUEUE_CFG = {"gyro_late_max_s": 1.0}


def test_a_queued_downlink_is_late_not_stale():
    """The RAW_IMU samples 0.7 s old on the vehicle clock but ARRIVING all
    along: the heading is late, not stale (42 % of 10-13's ticks)."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False, **QUEUE_CFG)
    f = Feed(T, pose)
    real = f.imu

    def queued(t):
        im = real(t)
        im.gyro_int, im.t_gyro = tuple(T.gint_at(t - 0.7)), t
        im.t_gyro_v = t - 0.7
        return im
    f.imu = queued
    f.run(5.0)
    out = f.run(2.0, tags=False, engaged=True)
    assert all(h["ok"] for _t, _m, h, _tm in out), out[-1][2]["why"]


def test_the_speed_stop_is_a_seconds_mean_not_one_sample():
    T = Truth(v_b=(0.10, 0.0, 0.0), T=300.0)
    pose = _pose()
    f = Feed(T, pose)
    f.run(5.0)
    t1 = f.t + 0.5
    f.dvl_bias = np.array([0.0, 0.0])
    real_usable = f.dvl_usable
    orig = pose.note_dvl

    def spike(tv, v, ok, *a, **k):                    # ONE 0.36 m/s sample
        if t1 <= tv < t1 + 0.1:
            v = np.array(v, float)
            v[0] = 0.36
        return orig(tv, v, ok, *a, **k)
    pose.note_dvl = spike
    out = f.run(2.0, tags=False, engaged=True)
    assert not [h for _t, _m, h, _tm in out if "false lock" in h["fault"]]
    pose.note_dvl = orig
    f.dvl_usable = real_usable
    f.dvl_bias = np.array([0.30, 0.0])                # sustained: a lock
    out = f.run(2.0, tags=False, engaged=True)
    assert [h for _t, _m, h, _tm in out if "false lock" in h["fault"]]


def test_a_held_tag_jump_bigger_than_a_correction_is_slewed_not_a_fault():
    """The tags in view all along and their level moves 35 cm and stays:
    the station without --dvl_pose steps to it; the pose slews it in with a
    warning (refusing it ended real ON runs). A correction after DEAD
    RECKONING that big still ends a control run (the next test above)."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose()
    f = Feed(T, pose)
    f.run(5.0)
    f.tag_jump = lambda t: np.array([0.35, 0.0])
    out = f.run(3.0, engaged=True)
    assert not [h for _t, _m, h, _tm in out if h["fault"]]
    c = pose.corrections[-1]
    assert c["kind"] == "jump" and c["applied"] == "slew" and c["err_m"] > 0.3
    assert [h for _t, _m, h, _tm in out
            if "slewed in, not refused" in h["warn"]]
    steps = [float(np.hypot(*(b[1]["eta"][:2] - a[1]["eta"][:2])))
             for a, b in zip(out, out[1:])]
    assert max(steps) < 0.02 * 0.05 + 1e-3, max(steps)


def test_the_z_offset_is_retaken_on_fixes_as_the_vehicle_moves():
    """The map 5 cm/m off gravity: 2 m in y with the tags in view moves the
    tag z 10 cm against the Bar30. The offset is re-taken on the fixes, so a
    dropout there starts — and stays — at the tag level, not at the offset of
    the first anchor."""
    T = Truth(v_b=(0.0, 0.1, 0.0), psi0=0.0, T=300.0)
    pose = _pose()
    f = Feed(T, pose)
    real_tag = f.tag

    def tilted(t, fresh=True):
        meas, fix = real_tag(t, fresh)
        meas["eta"][2] += 0.05 * float(meas["eta"][1])
        fix.p_ned = np.array(fix.p_ned, float)
        fix.p_ned[2] += 0.05 * float(fix.p_ned[1])
        return meas, fix
    f.tag = tilted
    f.run(25.0)                                    # ~2.5 m in y, tags on
    out = f.run(4.0, tags=False, engaged=True)
    t_end, m_end = out[-1][0], out[-1][1]
    want = T.z(t_end) + 0.05 * float(T.pos(t_end)[1])   # the tag-frame z
    assert abs(float(m_end["eta"][2]) - want) < 0.03, \
        (float(m_end["eta"][2]), want)


# =============================================================================
# fix verification round 5 (2026-10-18)
# =============================================================================
def _queued_imu(f, T, late):
    """RAW_IMU records ``late`` s old on the vehicle clock, ARRIVING all
    along (a queued downlink)."""
    real = f.imu

    def queued(t):
        im = real(t)
        im.gyro_int, im.t_gyro = tuple(T.gint_at(t - late)), t
        im.t_gyro_v = t - late
        return im
    f.imu = queued


def test_a_late_gyro_in_a_turn_is_stale_at_rest_it_flies():
    """RAW_IMU 0.7 s late: at rest the heading is carried (late, not stale —
    the test above); in a 20 deg/s pivot the 0.5 s beyond the extrapolation
    is 10 deg of heading nothing knows — the pose stops until the turn's end
    has arrived (round 4 flew it)."""
    om = math.radians(20.0)
    T = Truth(v_b=(0.0, 0.0, 0.0),
              omega=lambda s: om if 8.0 < s < 11.0 else 0.0, T=300.0)
    pose = _pose(require_gyro_bias=False, **QUEUE_CFG)
    f = Feed(T, pose)
    _queued_imu(f, T, 0.7)
    f.run(6.0)
    assert pose.anchored
    out = f.run(8.0, tags=False, engaged=True)      # s 6..14, pivot 8..11
    s = [(t - T.t0, h) for t, _m, h, _tm in out]
    assert all(h["ok"] for x, h in s if x < 8.0), \
        [h["why"] for x, h in s if x < 8.0 and not h["ok"]][:1]
    turning = [h for x, h in s if 9.0 < x < 11.5]
    assert turning and all(not h["ok"] and "cannot be carried" in h["why"]
                           for h in turning), turning[0]["why"]
    # the turn's end arrived: carried again, on the gyro's own integral
    after = [(t, m) for t, m, h, _tm in out if t - T.t0 > 12.0]
    assert after and all(m is not None for _t, m in after)
    assert max(abs(DP.wrap(m["eta"][5] - T.yaw(t))) for t, m in after) \
        < math.radians(0.3)


def test_a_slow_correction_in_a_queue_flies():
    """The same 0.7 s queue under a 4 deg/s station correction: 2 deg past
    the extrapolation, under gyro_late_deg 3 (under what the tag gate calls
    one level, jump_yaw_deg 5) — carried. At 1 deg, 18 of the 23 10-13 OFF
    replay anchors stopped within seconds on such corrections."""
    om = math.radians(4.0)
    T = Truth(v_b=(0.0, 0.0, 0.0),
              omega=lambda s: om if 8.0 < s < 11.0 else 0.0, T=300.0)
    pose = _pose(require_gyro_bias=False, **QUEUE_CFG)
    f = Feed(T, pose)
    _queued_imu(f, T, 0.7)
    f.run(6.0)
    out = f.run(8.0, tags=False, engaged=True)
    assert all(h["ok"] for _t, _m, h, _tm in out), \
        [h["why"] for _t, _m, h, _tm in out if not h["ok"]][:1]


def test_a_gyro_later_than_gyro_late_max_s_is_stale_even_at_rest():
    """gyro_late_max_s past the 0.2 s extrapolation at any rate: stale — 0.1
    s by default since round 8, i.e. 0.3 s late."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(3.0)
    t = f.t
    hd = pose.head
    # (a vehicle-stamped history: the lateness is on the vehicle clock)
    hd.t = hd.t_v = t - 0.29
    hd.t_rx, hd.rate = t, 0.0                       # 0.09 s past the 0.2 s
    assert "cannot be carried" not in pose._heading_why(t)
    hd.t = hd.t_v = t - 0.31                        # 0.11 s past it
    assert "cannot be carried" in pose._heading_why(t)
    # (the cap is a setting: 1.0 s, its value before round 8)
    pose.cfg = dict(pose.cfg, gyro_late_max_s=1.0)
    assert "cannot be carried" not in pose._heading_why(t)
    hd.t = hd.t_v = t - 1.3                         # 1.1 s past it
    assert "cannot be carried" in pose._heading_why(t)


def test_a_burst_of_queued_gyro_records_does_not_inflate_the_rate():
    """Records arriving 1 ms apart, each carrying 50 ms of a 10 deg/s turn,
    stamped on arrival (no vehicle clock): differenced over 1 ms they read
    500 deg/s, and raw_at's 0.2 s extrapolation made that 100 deg."""
    pose = _pose(require_gyro_bias=False)
    hd = pose.head
    om = math.radians(10.0)

    def rec(t_stamp, gz, n):
        return VehicleImu(roll=0.0, pitch=0.0, yaw=0.0, p=0.0, q=0.0, r=om,
                          t_att=t_stamp, t_att_v=t_stamp, gyro_int=(0, 0, gz),
                          t_gyro=t_stamp, n_gyro=n, n_gyro_gaps=0,
                          stamp=t_stamp)
    n, gz, t = 0, 0.0, 100.0
    for _ in range(20):                             # 1 s at 20 Hz
        n, gz, t = n + 1, gz + om * 0.05, t + 0.05
        hd.note_imu(rec(t, gz, n))
    assert abs(hd.rate - om) < 1e-6
    rates = []
    for _ in range(6):                              # the burst
        n, gz, t = n + 1, gz + om * 0.05, t + 0.001
        hd.note_imu(rec(t, gz, n))
        rates.append(hd.rate)
    assert max(abs(r) for r in rates) <= om * 1.0001, \
        [math.degrees(r) for r in rates]
    assert abs(hd.raw_at(t + 0.2) - (hd.raw + om * 0.2)) < 1e-6
    # ...and a clock map that moved never makes it faster than 2.5 rad/s
    n, gz, t = n + 1, gz + math.radians(30.0), t + 0.03
    hd.note_imu(rec(t, gz, n))
    assert abs(hd.rate) <= 2.5 + 1e-9


def test_a_one_sample_depth_spike_never_reaches_the_dead_reckoned_z():
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(5.0)
    real = f.imu
    t_sp = f.t + 1.0

    def spiky(t):
        im = real(t)
        if abs(t - t_sp) < 0.025:                   # one Bar30 record
            im.depth_sp2_m = im.depth_sp2_m + 0.30
        return im
    f.imu = spiky
    out = f.run(3.0, tags=False, engaged=True)
    zerr = [abs(float(m["eta"][2]) - T.z(t)) for t, m, _h, _tm in out
            if m is not None]
    assert len(zerr) == len(out) and max(zerr) < 0.005, max(zerr)


@pytest.mark.parametrize("step", [0.30, 0.70])
def test_a_depth_step_that_holds_is_taken_within_a_second(step):
    """A step the vehicle cannot make (between two records) that STAYS: the
    sensor's level moved — refused while it could be a spike, taken within
    1 s (TAG FIX ON re-takes the offset on the next fixes). 30 cm is within
    the 0.5 m/s reach of the interval by then; 70 cm is taken on the 1 s
    hold."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(5.0)
    real = f.imu
    t_st = f.t + 0.5

    def stepped(t):
        im = real(t)
        if t >= t_st:
            im.depth_sp2_m = im.depth_sp2_m + step
        return im
    f.imu = stepped
    out = f.run(2.5, tags=False, engaged=True)
    dz = [(t - t_st, float(m["eta"][2]) - T.z(t)) for t, m, _h, _tm in out
          if m is not None]
    assert all(abs(d) < 0.005 for x, d in dz if x < 0.25), dz
    assert all(abs(d - step) < 0.005 for x, d in dz if x > 1.05), dz


def test_an_accepted_correction_keeps_the_pose_where_it_was_in_z_too():
    """Tags back after a dropout with the STATE's z bridged 5 cm off its
    fix: the agreement judges the fix's z (no z difference), but the slew
    starts from the pose the controller saw — off takes up the state's
    difference, so the accept tick does not step z by the bridging."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(5.0)
    f.run(1.0, tags=False, engaged=True)             # dead reckoning shown
    real_tag = f.tag

    def bridged(t, fresh=True):
        meas, fix = real_tag(t, fresh)
        meas["eta"][2] += 0.05                       # the state, not the fix
        return meas, fix
    f.tag = bridged
    n0 = len(pose.corrections)
    out = f.run(2.0, engaged=True)
    assert len(pose.corrections) > n0 and \
        pose.corrections[-1]["applied"] == "slew"
    zs = [float(m["eta"][2]) for _t, m, _h, _tm in out if m is not None]
    steps = [abs(b - a) for a, b in zip(zs, zs[1:])]
    assert max(steps) < 0.002, max(steps)


def test_a_sample_twice_the_speed_cap_stops_at_once():
    """The 1 s mean (the test above) lets one fast sample by — but one at
    twice max_dr_speed_m_s is a stop on its tick: the mean would integrate
    ~0.3 m of a false lock before it spoke."""
    T = Truth(v_b=(0.10, 0.0, 0.0), T=300.0)
    pose = _pose()
    f = Feed(T, pose)
    f.run(5.0)
    t1 = f.t + 0.5
    orig = pose.note_dvl

    def spike(tv, v, ok, *a, **k):                  # ONE 0.70 m/s sample
        if t1 <= tv < t1 + 0.1:
            v = np.array(v, float)
            v[0] = 0.70
        return orig(tv, v, ok, *a, **k)
    pose.note_dvl = spike
    out = f.run(2.0, tags=False, engaged=True)
    hits = [t for t, _m, h, _tm in out if "false lock" in h["fault"]]
    assert hits and hits[0] - t1 <= f.dvl_late + 0.1 + 1e-6, \
        (hits[:1], t1)


def test_a_sparse_view_does_not_refuse_off_for_as_long_as_it_lasts():
    """Fixes waiting for their agreement refuse OFF while they are still
    coming in; one fix and then nothing (a sparse view) refuses for 1 s, not
    for the whole agree_window_s it is still kept for."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(5.0)
    f.tag_jump = lambda t: np.array([0.15, 0.0])     # a new level
    for _ in range(40):
        f.run(0.05, engaged=True)
        if 0 < len(pose._pending) < 5:
            break
    assert 0 < len(pose._pending) < 5
    f.run(0.5, tags=False)
    assert pose.pending_n(f.t) >= 1
    assert "waiting to agree" in pose.tag_refusal(False, f.t)
    f.run(0.75, tags=False)
    assert pose.pending_n(f.t) == 0
    assert pose._pending_live(f.t) >= 1              # kept to agree with
    assert "waiting to agree" not in pose.tag_refusal(False, f.t)


def test_a_held_jump_the_dead_reckoning_drops_out_of_stays_a_jump():
    """The tags in view all along, their level moved 35 cm, and the DVL
    gone so the dead reckoning is lost mid-agreement: the re-acquisition
    goes on blind on the same fixes but is still the tags' JUMP — slewed,
    as the station steps to it — not a drift that ends the run."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose()
    f = Feed(T, pose)
    f.run(5.0)
    t_cut = f.t
    f.dvl_on = lambda tv: tv < t_cut                 # the DVL gone
    f.run(1.25, engaged=True)                        # the tags carry it
    f.tag_jump = lambda t: np.array([0.35, 0.0])     # (2-3 fixes before
    #                                                  the gap passes lost_s)
    out = f.run(3.0, engaged=True)
    assert [h for _t, _m, h, _tm in out if h["state"] == DP.ST_LOST]
    assert not [h for _t, _m, h, _tm in out if h["fault"]], \
        [h["fault"] for _t, _m, h, _tm in out if h["fault"]][:1]
    c = pose.corrections[-1]
    assert c["kind"] == "jump" and c["applied"] == "slew" \
        and c["err_m"] > 0.3, c


# =============================================================================
# fix verification round 6 (2026-10-18, wf_0e5903e0-a6d)
# =============================================================================
def _late_pivot(om_dps, t0, t1, late, tags_until, T_s=300.0, **cfg):
    """A pivot at om_dps between s t0 and t1 with the gyro ``late`` s queued
    (vehicle-stamped), tags in view and ENGAGED until s ``tags_until``, then
    a dropout. -> (Truth, pose, ticks after the dropout)."""
    om = math.radians(om_dps)
    T = Truth(v_b=(0.0, 0.0, 0.0),
              omega=lambda s: om if t0 < s < t1 else 0.0, T=T_s)
    pose = _pose(require_gyro_bias=False, **cfg)
    f = Feed(T, pose)
    _queued_imu(f, T, late)
    f.run(6.0)
    assert pose.anchored
    f.run(tags_until - 6.0, engaged=True)
    out = f.run(6.0, tags=False, engaged=True)
    return T, pose, out


def test_an_anchor_on_a_late_gyro_waits_for_the_history_to_reach_the_fix():
    """0.7 s queue; a 30 deg/s x 0.3 s burst with the tags in view, the tags
    dropping as it ends. Anchored at t_now through the extrapolation (round
    5) the whole burst — still in the queue — went into the offset, and the
    dead reckoning flew it once the backlog drained (9 deg). Anchored at the
    fixes the history covers, the heading is exact once it has drained."""
    T, pose, out = _late_pivot(30.0, 8.0, 8.3, 0.7, tags_until=8.3, **QUEUE_CFG)
    after = [(t, m) for t, m, h, _tm in out
             if m is not None and t - T.t0 > 9.3]
    assert after
    err = max(abs(DP.wrap(m["eta"][5] - T.yaw(t))) for t, m in after)
    assert math.degrees(err) < 0.5, math.degrees(err)


def test_an_anchor_late_in_a_pivot_end_is_exact_after_the_backlog():
    """20 deg/s pivot, the tags lost 0.35 s after its true end (round 5: 7
    stale ticks — under the debounce — then 3 deg flown for good)."""
    T, pose, out = _late_pivot(20.0, 8.0, 12.5, 0.7, tags_until=12.85, **QUEUE_CFG)
    after = [(t, m) for t, m, h, _tm in out
             if m is not None and t - T.t0 > 13.8]
    assert after
    err = max(abs(DP.wrap(m["eta"][5] - T.yaw(t))) for t, m in after)
    assert math.degrees(err) < 0.5, math.degrees(err)


def test_a_late_gyro_pivot_with_the_tags_in_view_is_no_yaw_jump():
    """30 deg/s, 0.7 s late, tags in view all along: the t_now anchor's
    error read as a > 5 deg 'jump' and held the dead reckoning against the
    right tags (round 5); anchored on covered fixes it is none."""
    om = math.radians(30.0)
    T = Truth(v_b=(0.0, 0.0, 0.0),
              omega=lambda s: om if 8.0 < s < 11.0 else 0.0, T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    _queued_imu(f, T, 0.7)
    f.run(6.0)
    j0 = pose.n_jumps_held
    out = f.run(8.0, engaged=True)
    assert pose.n_jumps_held == j0, pose.n_jumps_held - j0
    assert all(h["state"] in (DP.ST_TAG, DP.ST_BLEND)
               for _t, _m, h, _tm in out)


def test_the_bias_pairs_wait_for_the_history_too():
    """A 0.7 s queue with a gyro bias: pairs taken through the extrapolation
    during turns bent the fit; pairs on covered fixes learn the bias."""
    om = math.radians(20.0)
    T = Truth(v_b=(0.0, 0.0, 0.0), T=400.0,
              omega=lambda s: om * math.sin(0.5 * s))
    pose = _pose(require_gyro_bias=False, bias_min_span_s=40.0,
                 bias_min_n=100)
    f = Feed(T, pose, gyro_bias=math.radians(0.03))
    _queued_imu(f, T, 0.7)
    real = f.imu

    def biased(t):
        im = real(t)
        g = T.gint_at(t - 0.7)
        g = g + np.array([0.0, 0.0, f.gyro_bias * (t - 0.7 - T.t0)])
        im.gyro_int = tuple(g)
        return im
    f.imu = biased
    f.run(80.0)
    b = pose.head.bias_info()
    assert b["state"] == "ok"
    assert abs(b["dps"] - 0.03) < 0.01, b


def test_a_gyro_record_off_the_vehicle_clock_is_lateness_unknown():
    """Once the history is on the vehicle clock, a record the clock map
    cannot place (a queue past the reader's 2 s window) is lateness UNKNOWN:
    stale — it used to fall back to the arrival time, 'late 0.02 s'."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    _queued_imu(f, T, 0.3)
    f.run(5.0)
    out = f.run(1.0, tags=False, engaged=True)
    assert all(h["ok"] for _t, _m, h, _tm in out)
    real = f.imu

    def unplaced(t):
        im = real(t)
        im.t_gyro_v = None
        return im
    f.imu = unplaced
    out = f.run(1.0, tags=False, engaged=True)
    assert all(not h["ok"] and "cannot be placed" in h["why"]
               for _t, _m, h, _tm in out), out[0][2]["why"]
    f.imu = real                                     # placed again
    out = f.run(1.0, tags=False, engaged=True)
    assert out[-1][2]["ok"], out[-1][2]["why"]


def test_need_vclk_refuses_an_arrival_stamped_gyro_from_the_start():
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = DP.DvlPose({"bias_min_span_s": 20.0, "bias_min_n": 50,
                       "require_gyro_bias": False}, mode="control",
                      nudot_source="fd", need_vclk=True)
    f = Feed(T, pose)                                # t_gyro only
    out = f.run(5.0)
    assert not pose.anchored
    assert "cannot be placed" in pose._heading_why(f.t)


def test_the_att_rate_heading_is_late_on_the_vehicle_clock_too():
    """heading att_rate on a 0.7 s queue: stamped with the parse time the
    late rule could not fire, and 14 deg rode through a 20 deg/s pivot."""
    om = math.radians(20.0)
    T = Truth(v_b=(0.0, 0.0, 0.0),
              omega=lambda s: om if 8.0 < s < 11.0 else 0.0, T=300.0)
    pose = _pose(require_gyro_bias=False, heading="att_rate")
    f = Feed(T, pose)
    real = f.imu

    def queued_att(t):
        im = real(t)
        p, q, r = T.pqr_at(t - 0.7)
        im.p, im.q, im.r = p, q, r
        im.t_att_v = t - 0.7
        return im
    f.imu = queued_att
    f.run(6.0)
    assert pose.anchored
    out = f.run(6.0, tags=False, engaged=True)
    turning = [h for t, _m, h, _tm in out if 9.0 < t - T.t0 < 11.5]
    assert turning and all(not h["ok"] and "cannot be carried" in h["why"]
                           for h in turning), turning[0]["why"]


def test_a_re_acquisition_already_past_a_correction_refuses_off_past_1s():
    """A sparse view whose re-acquisition already shows the pose 30 cm off
    (> max_corr_m): OFF / START stay refused while those fixes are kept —
    the agreement can only end as a refusal once engaged; disengaged it
    snaps. (A small one is refused only while its fixes still come in.)"""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(5.0)
    f.run(1.0, tags=False)                           # dead reckoning
    f.tag_jump = lambda t: np.array([0.30, 0.0])
    for _ in range(40):                              # one or two fixes
        f.run(0.05)
        if 0 < len(pose._pending) < 3:
            break
    assert pose._pending_kind == "reacquire" and pose.pending_too_big()
    f.run(1.25, tags=False)
    assert pose.pending_n(f.t) >= 1
    assert "waiting to agree" in pose.tag_refusal(False, f.t)
    f.run(9.5, tags=False)                           # past agree_window_s
    assert pose.pending_n(f.t) == 0


def test_a_jump_whose_dead_reckoning_comes_back_mid_agreement_stays_a_jump():
    """35 cm tag jump held, the dead reckoning out for a few ticks (a
    RAW_IMU stall) and back before the blind agreement completed: the
    restart on innovations is the JUMP it was — slewed — not a
    re-acquisition refused as a drift."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(5.0)
    t_s = f.t
    # a 0.6 s RAW_IMU stall (stale past gyro_stale_s 0.5 s: 2 ticks)...
    f.gyro_on = lambda t: not (t_s < t < t_s + 0.6)
    f.run(0.45, engaged=True)
    # ...that the jump's first fixes are held into
    f.tag_jump = lambda t: np.array([0.35, 0.0])
    out = f.run(3.0, engaged=True)
    assert [h for _t, _m, h, _tm in out if h["state"] == DP.ST_LOST]
    assert not [h for _t, _m, h, _tm in out if h["fault"]], \
        [h["fault"] for _t, _m, h, _tm in out if h["fault"]][:1]
    c = pose.corrections[-1]
    assert c["kind"] == "jump" and c["applied"] == "slew", c


def test_a_blind_set_from_a_jump_is_a_drift_again_once_the_tags_dropped():
    """...but when the TAGS drop during that blind phase, they were not in
    view all along: its acceptance is a blind one, judged as a drift."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(5.0)
    pose._pending_kind, pose._blind_from_jump = "blind", True
    f.tick(f.t + 0.05, tags=False, engaged=True)
    assert pose._blind_from_jump is False


def test_a_refused_spike_is_not_the_depth_the_shown_pose_is_paired_with():
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(5.0)
    real = f.imu
    t_sp = f.t + 0.5

    def spiky(t):
        im = real(t)
        if abs(t - t_sp) < 0.025:
            im.depth_sp2_m = im.depth_sp2_m + 0.30
        return im
    f.imu = spiky
    worst = 0.0
    for _ in range(20):                              # through the spike tick
        f.run(0.05, tags=False, engaged=True)
        assert pose._shown_dz is not None
        worst = max(worst, abs(pose._shown_dz - (T.z(f.t) - T.depth(f.t))))
    assert worst < 0.005, worst


def test_a_bar30_level_change_under_on_is_not_flown_as_a_z_jump():
    """TAG FIX ON, tags in view, at rest: the Bar30 level steps 30 cm and
    stays. The new level is taken after its 1 s hold, the z offset re-taken
    on the next fix captured on it — not held as a z 'jump' that showed the
    stepped dead reckoning and slewed it back at 2 cm/s."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(5.0)
    real = f.imu
    t_st = f.t + 0.5

    def stepped(t):
        im = real(t)
        if t >= t_st:
            im.depth_sp2_m = im.depth_sp2_m + 0.30
        return im
    f.imu = stepped
    j0 = pose.n_jumps_held
    out = f.run(5.0, engaged=True)
    zerr = max(abs(float(m["eta"][2]) - T.z(t)) for t, m, _h, _tm in out
               if m is not None)
    assert zerr < 0.01, zerr
    assert pose.n_jumps_held == j0
    # ...and the dead reckoning after it is on the new level's offset
    out = f.run(2.0, tags=False, engaged=True)
    zerr = max(abs(float(m["eta"][2]) - T.z(t)) for t, m, _h, _tm in out
               if m is not None)
    assert zerr < 0.01, zerr


def test_a_disengage_mid_slew_is_not_read_as_a_held_jump():
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(5.0)
    f.tag_jump = lambda t: np.array([0.20, 0.0])
    f.run(1.5, engaged=True)                         # accepted, slewing
    assert pose.corr_m() > 0.05
    j0 = pose.n_jumps_held
    out = f.run(1.0, engaged=False)                  # disengaged
    assert pose.n_jumps_held == j0
    assert all(h["state"] == DP.ST_TAG for _t, _m, h, _tm in out[1:]), \
        [h["state"] for _t, _m, h, _tm in out]


def test_the_correction_record_says_what_the_slew_moves():
    """err_z_m is the FIX-based z that judged the correction; slew_z_m is
    what the slew moves (the state's difference) — they part in z (a
    replay summary read err_z_m as the slewed z, round 6)."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(5.0)
    f.run(1.0, tags=False, engaged=True)
    real_tag = f.tag

    def bridged(t, fresh=True):
        meas, fix = real_tag(t, fresh)
        meas["eta"][2] += 0.05
        return meas, fix
    f.tag = bridged
    f.run(2.0, engaged=True)
    c = [c for c in pose.corrections if c["applied"] == "slew"][-1]
    assert abs(c["err_z_m"]) < 0.01, c
    assert abs(c["slew_z_m"] - 0.05) < 0.01, c


def test_a_bar30_level_change_is_re_taken_on_the_state_without_fix_z():
    """z_source pressure (no fix z judged): the z offset is re-taken on the
    state each tag tick — a Bar30 level change ends its 'z jump test off'
    there too, instead of leaving the test off for good."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = DP.DvlPose({"bias_min_span_s": 20.0, "bias_min_n": 50,
                       "require_gyro_bias": False}, mode="control",
                      nudot_source="fd", z_from_fix=False)
    f = Feed(T, pose)
    f.run(5.0)
    real = f.imu
    t_st = f.t + 0.5

    def stepped(t):
        im = real(t)
        if t >= t_st:
            im.depth_sp2_m = im.depth_sp2_m + 0.70
        return im
    f.imu = stepped
    f.run(3.0, engaged=True)                         # taken (1 s hold)
    assert pose._z_retake is False



# =============================================================================
# fix verification round 7 (2026-10-18, wf_0a38afc8-48f)
# =============================================================================
def _queued_gyro(f, T, late, bias_dps=0.0, t_b=None, t_e=None):
    """RAW_IMU ``late`` s queued (a float or a function of t), vehicle-
    stamped, with a yaw-rate bias ``bias_dps`` on the vehicle time between
    ``t_b`` and ``t_e`` (a heading error growing there, then held)."""
    real = f.imu
    b = math.radians(bias_dps)

    def q(t):
        im = real(t)
        L = late(t) if callable(late) else late
        tv = t - L
        g = T.gint_at(tv)
        if t_b is not None and b:
            hi = tv if t_e is None else min(tv, t_e)
            g = g + np.array([0.0, 0.0, b * max(0.0, hi - t_b)])
        im.gyro_int, im.t_gyro = tuple(g), t
        im.t_gyro_v = tv
        return im
    f.imu = q


def _tag_level(f, dxy=(0.0, 0.0), dyaw_deg=0.0, dz_state=0.0):
    """Tag fixes (and states) on another level: xy and yaw on both, z
    ``dz_state`` on the STATE only (the assembler's bridging)."""
    real = f.tag
    dy = math.radians(dyaw_deg)
    dxy = np.asarray(dxy, float)

    def tag(t, fresh=True):
        meas, fix = real(t, fresh)
        meas["eta"][0:2] = meas["eta"][0:2] + dxy
        meas["eta"][5] = DP.wrap(meas["eta"][5] + dy)
        meas["eta"][2] += dz_state
        R = np.asarray(fix.R_ned_body, float).reshape(3, 3)
        fix.R_ned_body = rot_zyx(0.0, 0.0, dy) @ R
        p = np.asarray(fix.p_ned, float).copy()
        p[0:2] += dxy
        fix.p_ned = p
        return meas, fix
    return tag


def test_off_a_heading_divergence_is_caught_on_a_queued_link_too():
    """TAG FIX OFF on a 0.5 s queued link, the heading drifting 2 deg/s:
    the yaw is judged on the newest fix the gyro history covers, so the
    OFF 10 deg abort ends the run. Round 6 put the expected difference (~0)
    in its place, and the 12-15 deg heading flew with nothing to stop it."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False, **QUEUE_CFG)
    f = Feed(T, pose)
    _queued_gyro(f, T, 0.5)
    f.run(6.0)
    assert pose.anchored
    pose.set_tag_on(False, f.t, True)
    _queued_gyro(f, T, 0.5, bias_dps=2.0, t_b=f.t - 0.5)
    out = f.run(10.0, engaged=True)
    assert [h for _t, _m, h, _tm in out if "diverged" in h["fault"]], \
        max(h["err_yaw_deg"] or 0.0 for _t, _m, h, _tm in out)


def test_a_re_acquisition_past_the_yaw_limit_is_refused_on_a_queued_link():
    """ON, 0.7 s queued link: a dropout drifts the heading 15 deg; the tags
    come back. Judged (on covered fixes, or the state's difference until
    one is), 15 deg is past max_corr_yaw_deg — refused, as on a healthy
    link. Round 6 slewed it in with err_yaw_deg 0."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False, **QUEUE_CFG)
    f = Feed(T, pose)
    _queued_gyro(f, T, 0.7)
    f.run(6.0)
    assert pose.anchored
    t_b = f.t - 0.7
    _queued_gyro(f, T, 0.7, bias_dps=5.0, t_b=t_b, t_e=t_b + 3.0)
    f.run(3.7, tags=False, engaged=True)             # 15 deg, backlog in
    out = f.run(2.0, engaged=True)
    assert [h for _t, _m, h, _tm in out if h["fault"]], \
        [(c["kind"], c["applied"], c["err_yaw_deg"]) for c in pose.corrections]
    assert not [c for c in pose.corrections
                if c["kind"] == "reacquire" and c["applied"] == "slew"]


def test_a_flickering_lateness_does_not_stall_a_re_acquisition():
    """Lateness swinging 0.10-0.25 s around the 0.14 s capture age: round 6
    mixed judged and expected yaw in one agreement, which failed while 8
    deg of DR heading flew with the tags in view. Judged on covered fixes,
    the 8 deg re-acquisition is slewed in at once."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)

    def L(t):
        # (a 0.5 s swing — fast enough to mix covered and uncovered fixes in
        # one 5-fix agreement, slow enough that the vehicle stamps t - L(t)
        # still advance: 0.075 x 4 pi < 1)
        return 0.175 + 0.075 * math.sin(4.0 * math.pi * t)
    _queued_gyro(f, T, L)
    f.run(6.0)
    assert pose.anchored
    t_b = f.t - L(f.t)
    _queued_gyro(f, T, L, bias_dps=4.0, t_b=t_b, t_e=t_b + 2.0)
    f.run(2.5, tags=False, engaged=True)             # 8 deg off
    n0 = len(pose.corrections)
    t_back = f.t
    out = f.run(3.0, engaged=True)
    re = [c for c in pose.corrections[n0:] if c["kind"] == "reacquire"]
    assert re and re[0]["applied"] == "slew" and re[0]["t"] - t_back < 1.0, \
        (re[:1], [h["warn"] for _t, _m, h, _tm in out][:3])


def test_an_accepted_correction_on_a_late_link_keeps_the_heading_on_the_pose():
    """0.5 s queue: a tag level change of 15 cm and 8 deg, held, agreed and
    slewed in (engaged); the tags drop 0.2 s later. Fixes of the OLD level
    were still queued for the anchor: anchored with the NEW correction's
    yaw they put its whole 8 deg into the heading — the pose stepped 8 deg
    at the dropout (round 6). Queued with the correction in force then,
    they keep the heading on the pose."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False, **QUEUE_CFG)
    f = Feed(T, pose)
    _queued_gyro(f, T, 0.5)
    f.run(6.0)
    f.tag = _tag_level(f, dxy=(0.15, 0.0), dyaw_deg=8.0)
    n0 = len(pose.corrections)
    for _ in range(60):
        f.run(0.05, engaged=True)
        if len(pose.corrections) > n0:
            break
    assert pose.corrections[-1]["applied"] == "slew", pose.corrections[-1]
    # the pose stays where the slew puts it — truth (the DR it was) moving
    # toward the new level at corr_rate_deg_s — with the tags in view, and
    # through the dropout 0.25 s later (a heading 8 deg off showed as a held
    # 'jump' with the tags in view, or as an 8 deg step at the dropout)
    out = f.run(0.25, engaged=True) + f.run(0.3, tags=False, engaged=True)
    worst = max(abs(DP.wrap(float(m["eta"][5]) - T.yaw(t)))
                for t, m, h, _tm in out if m is not None)
    assert math.degrees(worst) < 1.5, math.degrees(worst)
    assert any(h["state"] in (DP.ST_DR, DP.ST_HELD)
               for _t, _m, h, _tm in out[-6:])          # (it did drop out)


def test_after_a_snap_the_heading_waits_for_the_new_level():
    """Disengaged, 0.5 s queue: a new tag level snaps in. The heading is on
    the old level until a fix of the new one is covered — no dead
    reckoning (no START) on it until then."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False, **QUEUE_CFG)
    f = Feed(T, pose)
    _queued_gyro(f, T, 0.5)
    f.run(6.0)
    f.tag = _tag_level(f, dxy=(0.15, 0.0), dyaw_deg=8.0)
    n0 = len(pose.corrections)
    for _ in range(60):
        f.run(0.05)
        if len(pose.corrections) > n0:
            break
    assert pose.corrections[-1]["applied"] == "snap"
    assert "new tag level" in pose._heading_why(f.t)
    f.run(1.0)
    assert pose._heading_why(f.t) == "", pose._heading_why(f.t)
    out = f.run(0.5, tags=False)
    dr = [(t, m) for t, m, h, _tm in out if m is not None]
    assert dr
    err = max(abs(DP.wrap(float(m["eta"][5]) - T.yaw(t) - math.radians(8.0)))
              for t, m in dr)
    assert math.degrees(err) < 0.5, math.degrees(err)


def test_a_queued_bar30_still_judges_z_on_the_fix():
    """Gyro and Bar30 both 0.7 s queued (vehicle-stamped), the state's z
    bridged 12 cm off its fixes: the Bar30 still ARRIVING is carried past
    its stamp to the capture, so z is judged on the fix — no held z 'jump'
    from the bridged state (round 6 fell back to it for 43 % of 10-13's
    fixes)."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    _queued_gyro(f, T, 0.7)
    gyro = f.imu

    def queued_bar30(t):
        im = gyro(t)
        im.depth_sp2_m, im.t_sp2, im.t_sp2_v = T.depth(t - 0.7), t, t - 0.7
        return im
    f.imu = queued_bar30
    f.run(6.0)
    assert pose.anchored
    f.tag = _tag_level(f, dz_state=0.12)
    j0 = pose.n_jumps_held
    out = f.run(8.0, engaged=True)
    assert pose.n_jumps_held == j0, pose.n_jumps_held - j0
    assert all(h["state"] in (DP.ST_TAG, DP.ST_BLEND)
               for _t, _m, h, _tm in out)


def test_the_first_anchor_leaves_no_anchoring_poses_pending():
    """A late gyro holds the first heading anchor back after the raw poses
    agreed; the raw poses pended meanwhile are no innovations — left over
    they held the tags on the first dead-reckoned tick or were taken as a
    141 cm 'jump'."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    _queued_gyro(f, T, 0.7)
    for _ in range(200):
        f.run(0.05)
        if pose.anchored:
            break
    assert pose.anchored
    assert pose._pending == []
    out = f.run(2.0, engaged=True)
    assert all(h["state"] in (DP.ST_TAG, DP.ST_BLEND)
               for _t, _m, h, _tm in out), \
        [h["warn"] for _t, _m, h, _tm in out if h["warn"]][:2]
    assert not [c for c in pose.corrections if c["err_m"] > 0.5]


def test_no_anchoring_pose_is_pended_while_the_heading_is_awaited():
    """The raw poses agreed (anchor ready) but the late gyro holds the first
    heading anchor back: no more raw poses are pended meanwhile."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    _queued_gyro(f, T, 0.7)
    waited = 0
    for _ in range(200):
        f.run(0.05)
        if pose._anchor_ready and not pose.anchored:
            waited += 1
            assert pose._pending == [], len(pose._pending)
        if pose.anchored:
            break
    assert pose.anchored and waited > 0, waited


def test_a_held_jump_is_no_divergence_stop_under_on():
    """ON, engaged, tags in view: a 35 cm level whose first fixes do not
    agree (a noisy view) is held past abort_hold_s. It is the tags' jump —
    slewed when it agrees, never 'diverged' (that is the OFF abort's)."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(5.0)
    k = {"n": 0, "tc": None}
    real = f.tag

    def noisy(t, fresh=True):
        meas, fix = real(t, fresh)
        if fix.t_capture != k["tc"]:
            k["tc"] = fix.t_capture
            k["n"] += 1
        d = 0.35 + (0.12 if (k["n"] < 12 and k["n"] % 2) else 0.0)
        meas["eta"][0] += d
        p = np.asarray(fix.p_ned, float).copy()
        p[0] += d
        fix.p_ned = p
        return meas, fix
    f.tag = noisy
    out = f.run(4.0, engaged=True)
    assert not [h for _t, _m, h, _tm in out if "diverged" in h["fault"]]
    assert not [h for _t, _m, h, _tm in out if h["fault"]]
    c = pose.corrections[-1]
    assert c["kind"] == "jump" and c["applied"] == "slew", c


def test_a_blind_correction_is_recorded_as_judged():
    """A blind correction compares the state with the shown pose — its yaw
    is judged, whatever the tick before it was."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(5.0)
    t_cut = f.t
    f.dvl_on = lambda tv: tv < t_cut
    f.run(2.5, tags=False, engaged=True)             # DR, then lost
    pose._yaw_judged = False                         # (a stale flag)
    n0 = len(pose.corrections)
    f.run(1.5, engaged=True)                         # blind re-acquisition
    new = pose.corrections[n0:]
    assert new and new[-1]["kind"] == "blind", new
    assert "yaw_judged" not in new[-1], new[-1]


def test_a_turn_started_within_the_lateness_is_no_heading_to_carry():
    """0.7 s queue: a 20 deg/s turn starts 0.4 s before the tags drop — its
    records have not arrived, the newest rate is still 0 and the late rule
    is silent, while the heading is already 8 deg behind. The tags measured
    that lag on their last ticks: the dead reckoning does not take over on
    it until the gyro history has caught up (10-13: up to 40 deg)."""
    om = math.radians(20.0)
    T = Truth(v_b=(0.0, 0.0, 0.0),
              omega=lambda s: om if 8.0 < s < 9.0 else 0.0, T=300.0)
    pose = _pose(require_gyro_bias=False, **QUEUE_CFG)
    f = Feed(T, pose)
    _queued_gyro(f, T, 0.7)
    f.run(6.0)
    assert pose.anchored
    f.run(2.4, engaged=True)                         # tags until s 8.4
    out = f.run(4.0, tags=False, engaged=True)
    flown = [(t, m) for t, m, h, _tm in out if m is not None]
    assert flown, [h["why"] for _t, _m, h, _tm in out][:3]
    worst = max(abs(DP.wrap(float(m["eta"][5]) - T.yaw(t))) for t, m in flown)
    assert math.degrees(worst) < 3.5, math.degrees(worst)


def test_a_lag_measured_on_a_level_change_is_spent_by_its_acceptance():
    """0.7 s queue: an 8 deg tag level change (with 15 cm) reads as an 8 deg
    'lag' (the judged yaw is on older fixes), and the jump agrees on xy
    before the yaw shows — accepted with that lag pending. Once the
    correction takes it up, there is no lag to wait out: a dropout 0.05 s
    later is carried — a lag kept past the acceptance left the pose LOST
    for 9 ticks, past the 8-tick stale debounce, and the run ended."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False, **QUEUE_CFG)
    f = Feed(T, pose)
    _queued_gyro(f, T, 0.7)
    f.run(6.0)
    f.tag = _tag_level(f, dxy=(0.15, 0.0), dyaw_deg=8.0)
    n0 = len(pose.corrections)
    for _ in range(60):
        f.run(0.05, engaged=True)
        if len(pose.corrections) > n0:
            break
    assert pose.corrections[-1]["applied"] == "slew", pose.corrections[-1]
    f.run(0.05, engaged=True)
    out = f.run(0.5, tags=False, engaged=True)
    lost = [h["why"] for _t, m, h, _tm in out if m is None]
    assert not lost, lost[:1]


def test_a_held_jump_on_a_late_link_keeps_the_dead_reckoning_shown():
    """0.7 s queue, a 15 cm + 8 deg tag level held as a jump: its yaw reads
    as an 8 deg 'lag' while it agrees. That is the tags' level change, not a
    turn — the dead reckoning stays the pose (it is what the jump is held
    against); dropping it sent the jump blind for the stale debounce."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False, **QUEUE_CFG)
    f = Feed(T, pose)
    _queued_gyro(f, T, 0.7)
    f.run(6.0)
    f.tag = _tag_level(f, dxy=(0.15, 0.0), dyaw_deg=8.0)
    n0 = len(pose.corrections)
    states = []
    for _ in range(60):
        out = f.run(0.05, engaged=True)
        states.append(out[-1][2]["state"])
        if len(pose.corrections) > n0:
            break
    assert DP.ST_LOST not in states, states
    assert pose.corrections[-1]["kind"] == "jump"


# =============================================================================
# fix verification round 8 (2026-10-18; wf_21856d6c-69a) and the operator's
# one-shot TAG FIX / no blind limit
# =============================================================================
def _yaw_outlier_once(f, dyaw_deg, k_at=1):
    """From now on, the ``k_at``-th new fix with its yaw (state and R) off by
    ``dyaw_deg`` — one single-tag PnP outlier — the rest clean."""
    real = f.tag
    st = {"tc": None, "k": 0}
    dy = math.radians(dyaw_deg)

    def tag(t, fresh=True):
        meas, fix = real(t, fresh)
        if fix.t_capture != st["tc"]:
            st["tc"] = fix.t_capture
            st["k"] += 1
        if st["k"] == k_at:
            meas["eta"][5] = DP.wrap(meas["eta"][5] + dy)
            fix.R_ned_body = rot_zyx(0.0, 0.0, dy) @ np.asarray(fix.R_ned_body)
        return meas, fix
    return tag


def test_one_tag_yaw_outlier_on_a_late_link_never_becomes_the_heading():
    """0.25 s queue (inside the cap), ON, engaged, at rest, ONE fix 8 deg off
    in yaw. It passes as the pose on its own tick (judged on an older
    covered fix — the station without --dvl_pose flies it too). Round 7
    then anchored the heading on it once covered, every clean fix after it
    read as a jump, slewed, re-anchored: 10-20 deg flown for 30 s with no
    fault (fix verification wf_21856d6c-69a, estimator F1). Judged before it
    may anchor, it is no anchor: no jump held, no correction, the pose on
    the truth — and a dropout after it is carried on the right heading."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    _queued_gyro(f, T, 0.25)
    f.run(8.0)
    assert pose.anchored
    f.tag = _yaw_outlier_once(f, 8.0)
    j0, n0 = pose.n_jumps_held, len(pose.corrections)
    r0 = pose.n_anchor_refused
    out = f.run(15.0, engaged=True)
    errs = [math.degrees(abs(DP.wrap(float(m["eta"][5]) - T.yaw(t))))
            for t, m, h, _tm in out if m is not None]
    assert sum(1 for e in errs if e > 2.0) <= 3, errs[:20]
    assert max(errs[10:]) < 1.0, max(errs[10:])
    assert pose.n_jumps_held == j0 and len(pose.corrections) == n0, \
        pose.corrections[n0:]
    assert pose.n_anchor_refused > r0
    out = f.run(1.0, tags=False, engaged=True)
    dr = [(t, m) for t, m, h, _tm in out if h["state"] == DP.ST_DR]
    assert dr, [h["why"] for _t, _m, h, _tm in out][:2]
    assert max(abs(DP.wrap(float(m["eta"][5]) - T.yaw(t))) for t, m in dr) \
        < math.radians(1.0)


def test_a_tag_yaw_level_the_gate_accepted_is_still_anchored():
    """The judgment before an anchor refuses a fix off the heading by more
    than jump_yaw_deg — not the fixes of a level change the gate held and
    accepted: the pose and their anchors carry the correction, so they are
    on the heading, the slew takes the heading to the new level, and a
    dropout after it steps nothing."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(8.0)
    f.tag = _tag_level(f, dyaw_deg=8.0)
    n0 = len(pose.corrections)
    for _ in range(80):
        f.run(0.05, engaged=True)
        if len(pose.corrections) > n0:
            break
    c = pose.corrections[-1]
    assert c["kind"] == "jump" and c["applied"] == "slew", c
    r0 = pose.n_anchor_refused
    f.run(10.0, engaged=True)                 # 8 deg at 1 deg/s: slewed out
    assert abs(float(pose.off[3])) < 1e-4
    assert pose.n_anchor_refused == r0
    out = f.run(0.5, tags=False, engaged=True)
    dr = [(t, m) for t, m, h, _tm in out if h["state"] == DP.ST_DR]
    assert dr
    for t, m in dr:
        assert abs(DP.wrap(float(m["eta"][5]) - (T.yaw(t) + math.radians(8.0)))) \
            < math.radians(0.5)


def _never_agreeing_level(f, dxy=(0.30, 0.0), wobble_m=0.06):
    """Tag fixes on another level whose consecutive fixes are 2 x wobble_m
    apart across it (y alternating +/- wobble_m): a level the agreement
    never believes (> jump_m), so the jump stays held."""
    real = f.tag
    st = {"tc": None, "k": 0}
    dxy = np.asarray(dxy, float)

    def tag(t, fresh=True):
        meas, fix = real(t, fresh)
        if fix.t_capture != st["tc"]:
            st["tc"] = fix.t_capture
            st["k"] += 1
        d = dxy + np.array([0.0, wobble_m if st["k"] % 2 else -wobble_m])
        meas["eta"][0:2] = meas["eta"][0:2] + d
        fix.p_ned = np.r_[np.asarray(fix.p_ned, float)[:2] + d,
                          float(np.asarray(fix.p_ned, float)[2])]
        return meas, fix
    return tag


def test_the_drift_during_a_held_tag_jump_ends_a_control_run():
    """A held jump is slewed whatever its size (round 7), so its own size is
    no stop — but the dead reckoning flies while it is held, and with the
    divergence watch off nothing but max_blind_s bounded that: 0.69-0.77 m
    in a synthetic sparse noisy hold, nothing at all with max_blind_s null
    (fix verification wf_21856d6c-69a, safety R7-1). The difference's MOVE
    since the hold began is the drift: past abort_err_m for abort_hold_s it
    ends the run, as the OFF abort does."""
    T = Truth(v_b=(0.10, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(6.0)
    f.run(1.0, engaged=True)
    f.tag = _never_agreeing_level(f)
    f.dvl_bias = np.array([-0.035, 0.0])       # the DR falls behind 3.5 cm/s
    out = f.run(20.0, engaged=True)
    faults = [(t, h["fault"]) for t, _m, h, _tm in out if h["fault"]]
    assert faults, [h["warn"] for _t, _m, h, _tm in out][-3:]
    t_f, why = faults[0]
    assert "held tag jump" in why, why
    worst = max(_xy_err(T, t, m) for t, m, h, _tm in out
                if m is not None and t <= t_f)
    assert 0.25 < worst < 0.40, worst
    assert pose.faults and "held tag jump" in pose.faults[-1][1]


def test_a_held_tag_jump_with_no_drift_is_no_stop():
    """The same never-agreeing level with the DVL right: the hold's drift
    stays the fixes' own wobble — 30 s held, no stop."""
    T = Truth(v_b=(0.10, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(6.0)
    f.run(1.0, engaged=True)
    f.tag = _never_agreeing_level(f)
    out = f.run(30.0, engaged=True)
    assert not [h["fault"] for _t, _m, h, _tm in out if h["fault"]]
    assert all(h["state"] == DP.ST_DR for _t, _m, h, _tm in out[5:])
    assert max(_xy_err(T, t, m) for t, m, h, _tm in out if m is not None) \
        < 0.02


def test_a_held_jump_accepted_after_the_dead_reckoning_drifted_is_refused():
    """With the drift watch off (abort_* null), the accept bounds it: a held
    jump that agrees only after the dead reckoning walked 35 cm is not
    slewed in as 'the tags' level' — the drift ends a control run, as a
    re-acquisition's would (max_corr_m)."""
    T = Truth(v_b=(0.10, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False, abort_err_m=None,
                 abort_err_yaw_deg=None, abort_err_z_m=None)
    f = Feed(T, pose)
    f.run(6.0)
    f.run(1.0, engaged=True)
    real = f.tag
    f.tag = _never_agreeing_level(f)
    f.dvl_bias = np.array([-0.035, 0.0])
    f.run(10.0, engaged=True)
    assert pose._pending_kind == "jump"
    # (the wobble ends: the same level, now agreeing)
    f.tag = _tag_level(SimpleNamespace(tag=real), dxy=(0.30, 0.0))
    n0 = len(pose.corrections)
    out = []
    for _ in range(40):
        out += f.run(0.05, engaged=True)
        if len(pose.corrections) > n0:
            break
    c = pose.corrections[-1]
    assert c["kind"] == "jump" and c["applied"] == "refused", c
    assert c["hold_drift_m"] > 0.25, c
    assert "while a tag jump was held" in out[-1][2]["fault"]


def _jump_slew_dropout(jump=(0.35, 0.0), slew_s=1.0, drop_s=1.0,
                       back=None, drop_dvl_bias=(0.0, 0.0)):
    """A held tag jump accepted and slewing, ``slew_s`` later a tag dropout
    of ``drop_s`` (the DVL off by drop_dvl_bias in it), then the tags back —
    on the jump's level, or ``back`` -> (pose, the corrections after it,
    the ticks after it)."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose()
    f = Feed(T, pose)
    f.run(5.0)
    assert pose.anchored
    f.tag_jump = lambda t: np.asarray(jump, float)
    n0 = len(pose.corrections)
    for _ in range(200):
        f.run(0.05, engaged=True)
        if len(pose.corrections) > n0:
            break
    acc = pose.corrections[-1]
    assert acc["kind"] == "jump" and acc["applied"] == "slew", acc
    f.run(slew_s, engaged=True)
    f.dvl_bias = np.asarray(drop_dvl_bias, float)
    f.run(drop_s, tags=False, engaged=True)
    f.dvl_bias = np.zeros(2)
    if back is not None:
        f.tag_jump = lambda t: np.asarray(back, float)
    nc = len(pose.corrections)
    out = f.run(3.0, engaged=True)
    return pose, pose.corrections[nc:], out


def test_a_dropout_mid_slew_is_judged_against_the_slews_target():
    """A 35 cm held jump slewing in, a 1 s tag dropout 1 s later, the tags
    back on the same level: the re-acquisition's difference is the slew's
    remainder, ~31 cm, which was judged as drift past max_corr_m — the run
    ended with nothing wrong, and the record showed a drift that never was
    (fix verification wf_21856d6c-69a, estimator F3 / safety R7-2). Judged
    against the level the correction was slewing to it is ~0: slewed on."""
    pose, new, out = _jump_slew_dropout()
    assert not [h["fault"] for _t, _m, h, _tm in out if h["fault"]]
    assert new and new[0]["kind"] == "reacquire", new
    assert new[0]["applied"] == "slew" and new[0]["err_m"] < 0.01, new[0]
    assert new[0].get("judged_on") == "slew_target"


def test_a_dropout_mid_slew_does_not_refuse_a_start_or_count_blind():
    """The same reading in what else judges a re-acquisition: the START /
    OFF refusal of a pending one past max_corr (pending_too_big), and the
    vouching behind max_blind_s — the slew's remainder is no drift there
    either (with max_blind_s 1.2 s, a 1 s dropout and the remainder read as
    drift, the run ended blind before the tags agreed)."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(max_blind_s=1.2)
    f = Feed(T, pose)
    f.run(5.0)
    f.tag_jump = lambda t: np.array([0.35, 0.0])
    n0 = len(pose.corrections)
    for _ in range(200):
        f.run(0.05, engaged=True)
        if len(pose.corrections) > n0:
            break
    assert pose.corrections[-1]["applied"] == "slew"
    f.run(1.0, engaged=True)
    f.run(1.0, tags=False, engaged=True)
    out = f.run(0.1, engaged=True)               # the first fixes back
    assert pose._pending_kind == "reacquire" and pose._pending
    assert not pose.pending_too_big()
    assert pose._vouches()
    out += f.run(3.0, engaged=True)
    assert not [h["fault"] for _t, _m, h, _tm in out if h["fault"]]


def test_the_tags_back_on_the_old_level_mid_slew_are_slewed_too():
    """...and the tags back on the OLD level early in the slew agree with
    the pose shown (the other no-drift reading): slewed, as before."""
    pose, new, out = _jump_slew_dropout(slew_s=0.2, drop_s=0.5,
                                        back=(0.0, 0.0))
    assert not [h["fault"] for _t, _m, h, _tm in out if h["fault"]]
    assert new and new[0]["applied"] == "slew" and new[0]["err_m"] < 0.05, new
    assert "judged_on" not in new[0]


def test_a_real_drift_in_a_dropout_mid_slew_still_ends_the_run():
    """...but a dead reckoning that drifted 30 cm across in the dropout is
    off both readings: refused, the run ends."""
    pose, new, out = _jump_slew_dropout(drop_s=3.0,
                                        drop_dvl_bias=(0.0, 0.10))
    assert new and new[0]["applied"] == "refused", new
    assert new[0]["err_m"] > 0.25, new[0]
    assert any("disagree" in h["fault"] for _t, _m, h, _tm in out)


def test_a_link_more_than_0p3_s_late_has_no_dead_reckoning():
    """gyro_late_max_s 0.1 s past the 0.2 s extrapolation = 0.3 s late: a
    turn that starts in the undelivered stretch is in no record, so no rule
    can see it. The healthy days never pass 0.13 s; 10-13's queue put 63 %
    of its engaged ticks past 0.3 s and carried 10-27 deg heading errors
    through dropouts at the old 1.0 s (round 8). At rest, tags gone: 0.25 s
    late carries the pose; 0.35 s late has none — and OFF is refused."""
    for late, ok in ((0.25, True), (0.35, False)):
        T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
        pose = _pose(require_gyro_bias=False)
        f = Feed(T, pose)
        _queued_gyro(f, T, late)
        f.run(6.0)
        assert pose.anchored
        why = pose.tag_refusal(False, f.t)
        assert (why == "") == ok, (late, why)
        out = f.run(1.0, tags=False, engaged=True)
        if ok:
            assert all(h["ok"] for _t, _m, h, _tm in out), \
                [h["why"] for _t, _m, h, _tm in out if not h["ok"]][:1]
        else:
            assert all(not h["ok"] and "late" in h["why"]
                       for _t, _m, h, _tm in out), out[0][2]["why"]


def test_a_one_shot_tag_fix_corrects_once_and_then_says_back_to_off():
    """TAG FIX pressed while OFF (the operator's use, 2026-10-18: clear the
    DVL's drift now and then): ON until ONE correction is in, then due back
    to OFF — at once when it snapped (disengaged), once the slew is out
    (engaged)."""
    T = Truth(v_b=(0.05, 0.0, 0.0), T=600.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(6.0)
    for engaged, applied in ((False, "snap"), (True, "slew")):
        pose.set_tag_on(False, f.t, engaged)
        f.dvl_bias = np.array([0.02, 0.0])
        f.run(5.0, engaged=engaged)                 # OFF: ~10 cm of drift
        f.dvl_bias = np.zeros(2)
        pose.start_oneshot(f.t, engaged)
        assert pose.tag_on and pose.oneshot is not None
        assert pose.toggles[-1].get("oneshot") is True
        assert pose.oneshot_due(f.t) == ""
        n0 = len(pose.corrections)
        for _ in range(60):
            f.run(0.05, engaged=engaged)
            if len(pose.corrections) > n0:
                break
        c = pose.corrections[-1]
        assert len(pose.corrections) == n0 + 1, pose.corrections[n0:]
        assert c["kind"] == "reacquire" and c["applied"] == applied, c
        assert 0.07 < c["err_m"] < 0.13, c
        if engaged:
            assert pose.oneshot_due(f.t) == ""      # still slewing
            f.run(6.0, engaged=True)
        assert pose.oneshot_due(f.t) == "done"
        pose.set_tag_on(False, f.t, engaged, auto="done")
        assert not pose.tag_on and pose.oneshot is None
        assert pose.toggles[-1]["auto"] == "done"


def test_a_one_shot_tag_fix_with_no_tags_times_out_uncorrected():
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(6.0)
    pose.set_tag_on(False, f.t, True)
    f.run(1.0, engaged=True)
    pose.start_oneshot(f.t, True)
    n0 = len(pose.corrections)
    to = pose.cfg["oneshot_timeout_s"]
    f.run(to - 0.5, tags=False, engaged=True)
    assert pose.oneshot_due(f.t) == ""
    f.run(0.6, tags=False, engaged=True)
    assert pose.oneshot_due(f.t) == "timeout"
    assert len(pose.corrections) == n0


def test_no_tag_in_view_for_a_minute_is_no_stop_by_default():
    """max_blind_s null since 2026-10-18 (the operator's choice: long
    tagless stretches, re-fixed by a one-shot press): OFF with no tag in
    view for 60 s and the DVL right — flown, and the panel warned after
    warn_unaided_s."""
    T = Truth(v_b=(0.05, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    assert pose.cfg["max_blind_s"] is None
    f = Feed(T, pose)
    f.run(6.0)
    pose.set_tag_on(False, f.t, True)
    t0 = f.t
    out = f.run(60.0, tags=False, engaged=True)
    assert all(h["ok"] and not h["fault"] for _t, _m, h, _tm in out), \
        [h["why"] or h["fault"] for _t, _m, h, _tm in out
         if not h["ok"] or h["fault"]][:1]
    assert any("since the last tag correction" in h["warn"]
               for t, _m, h, _tm in out if t - t0 > 25.0)


def test_after_a_gyro_hole_the_tags_re_anchor_a_heading_far_off():
    """A gyro hole whose interval held a 20 deg turn the integral never saw
    (the tags in view all along: the pose passes them through): the heading
    after it is that far off. The anchor's judgment (round 8) must not
    refuse the fixes that re-anchor it — across a hole there is no heading
    to judge by."""
    T = Truth(v_b=(0.0, 0.0, 0.0),
              omega=lambda s: math.radians(20.0) if 6.0 < s < 7.0 else 0.0,
              T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(6.0)
    f.gyro_on = lambda t: False
    f.run(1.0, engaged=True)                    # the turn, no records
    f.gyro_on = lambda t: True
    f.n_gaps = 1                                # the reader left it out
    f.run(0.05, engaged=True)
    assert pose.head.broken_since_anchor()
    out = f.run(1.0, engaged=True)              # re-anchored on the tags
    assert not pose.head.broken_since_anchor()
    out = f.run(1.0, tags=False, engaged=True)
    dr = [(t, m) for t, m, h, _tm in out if h["state"] == DP.ST_DR]
    assert dr, out[-1][2]["why"]
    assert max(abs(DP.wrap(float(m["eta"][5]) - T.yaw(t))) for t, m in dr) \
        < math.radians(1.0)


def test_a_never_agreeing_view_after_a_dropout_mid_slew_is_no_divergence():
    """The tags back on the slewing jump's level after a dropout, but never
    agreeing (a +/- 6 cm wobble): the re-acquisition stays pending and the
    divergence watch judges it meanwhile — on the drift (_jerr), not on the
    slew's ~30 cm remainder, which ended the run on 'diverged'."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose()
    f = Feed(T, pose)
    f.run(5.0)
    f.tag_jump = lambda t: np.array([0.35, 0.0])
    n0 = len(pose.corrections)
    for _ in range(200):
        f.run(0.05, engaged=True)
        if len(pose.corrections) > n0:
            break
    assert pose.corrections[-1]["applied"] == "slew"
    f.run(1.0, engaged=True)
    f.run(1.0, tags=False, engaged=True)
    f.tag = _never_agreeing_level(f, dxy=(0.0, 0.0))
    out = f.run(5.0, engaged=True)
    assert pose._pending_kind == "reacquire"
    assert not [h["fault"] for _t, _m, h, _tm in out if h["fault"]]


# =============================================================================
# fix verification round 9 (2026-10-18; wf_f143a3fa-24b)
# =============================================================================
def _view_cycle(t0, on_s, off_s):
    """Tags in view ``on_s`` then out ``off_s``, repeating, from ``t0``."""
    def tags(t):
        return ((t - t0) % (on_s + off_s)) < on_s
    return tags


def test_a_chain_of_re_acquisitions_cannot_slew_a_drift_past_the_limit():
    """ON, engaged, a station hold, the DVL 3.4 cm/s wrong (inside the
    proof's tolerance), tags in view 1 s of every 7. Each re-acquisition
    came while the last one's correction still slewed (2 cm/s is slower):
    round 8 judged each on the slew's target, so each read only the newest
    dropout's drift (~22 cm), every one was slewed, and the flown pose walked
    1.5 m off with no stop (fix verification wf_f143a3fa-24b, R8E-1). A
    re-acquisition's leftover is the error being flown: judged with it in,
    the chain ends the run by its second or third re-acquisition."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=600.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(6.0)
    f.run(1.0, engaged=True)
    f.dvl_bias = np.array([0.034, 0.0])
    out = f.run(60.0, tags=_view_cycle(f.t, 1.0, 6.0), engaged=True)
    faults = [(t - T.t0, h["fault"]) for t, _m, h, _tm in out if h["fault"]]
    assert faults, max(_xy_err(T, t, m) for t, m, h, _tm in out
                       if m is not None)
    assert "disagree" in faults[0][1] and "reacquire" in faults[0][1], faults[0]
    t_f = faults[0][0] + T.t0
    worst = max(_xy_err(T, t, m) for t, m, h, _tm in out
                if m is not None and t <= t_f)
    assert worst < 0.45, worst
    assert all(c.get("slew_m", 0.0) < 0.40 for c in pose.corrections
               if c.get("applied") == "slew"), pose.corrections


def test_a_one_shot_in_a_chain_of_dropouts_ends_like_any_re_acquisition():
    """...and the one-shot does not loop on it: pressed while OFF with the
    same DVL and view, it is refused (ends the run) once the leftover plus
    the newest drift passes max_corr, instead of slewing for ever."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=600.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(6.0)
    pose.set_tag_on(False, f.t, True)
    f.dvl_bias = np.array([0.034, 0.0])
    f.run(5.0, tags=False, engaged=True)            # 17 cm of drift
    pose.start_oneshot(f.t, True)
    out = f.run(60.0, tags=_view_cycle(f.t, 1.0, 6.0), engaged=True)
    faults = [(t, h["fault"]) for t, _m, h, _tm in out if h["fault"]]
    assert faults and "reacquire" in faults[0][1], faults[:1]
    # (the worker ends the run on that fault; the feed here flies on)
    assert max(_xy_err(T, t, m) for t, m, h, _tm in out
               if m is not None and t <= faults[0][0]) < 0.45


def test_the_hold_drift_does_not_count_the_slew_in_flight():
    """A 0.60 m tag level change accepted and slewing (2 cm/s, ~30 s), then
    a cluster 0.30 m across whose fixes never agree — a held jump on the DR,
    whose track the slew keeps moving. Round 8 read that movement as drift:
    'drifted 30 cm', the run ended with a perfect DVL (R8D-2 / R8-1). On the
    difference with the correction in flight added back it is no drift."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(6.0)
    real = f.tag
    f.tag = _tag_level(SimpleNamespace(tag=real), dxy=(0.60, 0.0))
    n0 = len(pose.corrections)
    for _ in range(80):
        f.run(0.05, engaged=True)
        if len(pose.corrections) > n0:
            break
    assert pose.corrections[-1]["kind"] == "jump"
    f.run(1.0, engaged=True)
    # (the second cluster 30 cm across the slew's direction: the slew moves
    # the difference along x, never back to the old level)
    first = SimpleNamespace(tag=_tag_level(SimpleNamespace(tag=real),
                                           dxy=(0.60, 0.0)))
    f.tag = _never_agreeing_level(first, dxy=(0.0, 0.30))
    out = f.run(20.0, engaged=True)
    assert pose._pending_kind == "jump"
    assert not [h["fault"] for _t, _m, h, _tm in out if h["fault"]], \
        [h["fault"] for _t, _m, h, _tm in out if h["fault"]][:1]


def test_tags_back_on_the_slewing_level_after_a_long_glimpse_are_no_jump():
    """...and the tags back on the level being slewed to after a 6 s glimpse
    of another one: the slew had moved the track past the stale jump
    reference, so the return read as a new jump whose 'drift' from the
    glimpse's level refused it (round 8). The reference moves with the
    track: the return is the same level, slewed on."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(6.0)
    real = f.tag
    level = _tag_level(SimpleNamespace(tag=real), dxy=(0.60, 0.0))
    f.tag = level
    n0 = len(pose.corrections)
    for _ in range(80):
        f.run(0.05, engaged=True)
        if len(pose.corrections) > n0:
            break
    f.run(1.0, engaged=True)
    f.tag = _never_agreeing_level(SimpleNamespace(tag=real), dxy=(0.90, 0.0))
    f.run(6.0, engaged=True)
    f.tag = level
    n1 = len(pose.corrections)
    out = f.run(3.0, engaged=True)
    assert not [h["fault"] for _t, _m, h, _tm in out if h["fault"]], \
        [h["fault"] for _t, _m, h, _tm in out if h["fault"]][:1]
    assert not [c for c in pose.corrections[n1:]
                if c["applied"] == "refused"], pose.corrections[n1:]
    assert out[-1][2]["state"] in (DP.ST_TAG, DP.ST_BLEND)


def test_a_held_jump_does_not_end_a_run_on_a_tag_z_move():
    """z is slewed whatever its size — the dead reckoning's z IS the Bar30,
    so a z 'drift' during a hold is the tags' (a cluster's map z): round 8's
    hold watch ended the run on a 25 cm tag-z move (docs R8D-3)."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(6.0)
    f.run(1.0, engaged=True)
    level = _never_agreeing_level(f)
    f.tag = level
    f.run(3.0, engaged=True)

    def z_moved(t, fresh=True):
        meas, fix = level(t, fresh)
        meas["eta"][2] += 0.25
        fix.p_ned = np.r_[np.asarray(fix.p_ned, float)[:2],
                          float(np.asarray(fix.p_ned, float)[2]) + 0.25]
        return meas, fix
    f.tag = z_moved
    out = f.run(6.0, engaged=True)
    assert pose._pending_kind == "jump"
    assert not [h["fault"] for _t, _m, h, _tm in out if h["fault"]], \
        [h["fault"] for _t, _m, h, _tm in out if h["fault"]][:1]


def test_the_hold_drift_survives_a_dead_reckoning_flicker():
    """A RAW_IMU stall of 0.3 s every 2.5 s during a never-agreeing hold:
    each sent the hold blind and the DR's return restarted it with a fresh
    reference — the drift so far forgiven, 1.4 m flown with no stop in a
    synthetic run (R8D-1, estimator lens). Kept across the round trip, the
    drift stops the run as without the flicker."""
    T = Truth(v_b=(0.10, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    st = {"t0": None}

    def late(t):
        # 0.36 s late for 2 ticks every 2.5 s (past the 0.3 s cap: no DR)
        if st["t0"] is None:
            return 0.07
        return 0.36 if ((t - st["t0"]) % 2.5) < 0.1 - 1e-9 else 0.07
    _queued_gyro(f, T, late)
    f.run(6.0)
    f.run(1.0, engaged=True)
    f.tag = _never_agreeing_level(f)
    f.dvl_bias = np.array([-0.035, 0.0])
    st["t0"] = f.t + 1.0
    out = f.run(30.0, engaged=True)
    assert [h for _t, m, h, _tm in out if m is None]      # (it flickered)
    faults = [(t, h["fault"]) for t, _m, h, _tm in out if h["fault"]]
    assert faults and "held tag jump" in faults[0][1], faults[:1]
    worst = max(_xy_err(T, t, m) for t, m, h, _tm in out
                if m is not None and t <= faults[0][0])
    assert worst < 0.45, worst
    assert pose.n_jumps_held == 1, pose.n_jumps_held


def test_a_due_one_shot_is_not_held_on_by_a_pending_agreement():
    """A glimpse's leftover fixes after the one-shot's correction, and a
    view whose fixes never agree before its timeout, each held TAG FIX ONCE
    ON (OFF's refusal waited for them: R8O-1, docs R8D-1). Due, it drops
    them: its OFF waits only for a slew in flight and the DR itself."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(6.0)
    pose.set_tag_on(False, f.t, True)
    f.run(2.0, tags=False, engaged=True)
    # (a) never-agreeing view: due at the timeout with fixes pending
    pose.start_oneshot(f.t, True)
    f.tag = _never_agreeing_level(f, dxy=(0.0, 0.0))
    f.run(pose.cfg["oneshot_timeout_s"] + 0.1, engaged=True)
    assert pose.oneshot_due(f.t) == "timeout" and pose._pending
    assert pose.tag_refusal(False, f.t) != ""        # (OFF's own: waits)
    assert pose.oneshot_refusal(f.t) == ""
    pose.drop_pending()
    pose.set_tag_on(False, f.t, True, auto="timeout")
    assert not pose.tag_on and not pose._pending


def test_a_re_acquisition_judged_on_the_pose_leaves_no_jump_level_behind():
    """A jump slewing in, a dropout, the tags back on the OLD level: judged
    on the pose shown (the tags left the jump's level), the jump part of the
    correction is gone — kept, it would stay a level the next
    re-acquisition could be discounted against (round 9)."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose()
    f = Feed(T, pose)
    f.run(5.0)
    f.tag_jump = lambda t: np.array([0.35, 0.0])
    n0 = len(pose.corrections)
    for _ in range(200):
        f.run(0.05, engaged=True)
        if len(pose.corrections) > n0:
            break
    assert pose.corrections[-1]["kind"] == "jump"
    assert np.allclose(pose.off_jump, pose.off)
    f.run(4.0, engaged=True)                    # ~8 cm of the 35 slewed
    f.run(0.5, tags=False, engaged=True)
    f.tag_jump = lambda t: np.zeros(2)          # back on the OLD level
    n1 = len(pose.corrections)
    for _ in range(40):
        f.run(0.05, engaged=True)
        if len(pose.corrections) > n1:
            break
    c = pose.corrections[-1]
    assert c["kind"] == "reacquire" and "judged_on" not in c, c
    assert not np.any(np.abs(pose.off_jump) > 1e-12), pose.off_jump


# =============================================================================
# fix verification round 10 (wf_ce59a697-8bd + the main session)
# =============================================================================
def _held_once(f, pose, n_ticks=80, engaged=True):
    """Run until one more correction is recorded; -> that correction."""
    n0 = len(pose.corrections)
    for _ in range(n_ticks):
        f.run(0.05, engaged=engaged)
        if len(pose.corrections) > n0:
            return pose.corrections[-1]
    raise AssertionError(f"no correction in {n_ticks} ticks")


def _z_level(src, dz):
    """Tag fixes AND states on a map z ``dz`` off (a tilted map, a cluster
    surveyed off in z)."""
    def tag(t, fresh=True):
        meas, fix = src(t, fresh)
        meas["eta"][2] += dz
        p = np.asarray(fix.p_ned, float).copy()
        p[2] += dz
        fix.p_ned = p
        return meas, fix
    return tag


def test_a_jump_inside_a_re_acquisitions_slew_does_not_launder_its_leftover():
    """The chain above, with the tag level stepping 15 cm (past the jump
    gate) half-way through every view: each view holds a re-acquisition,
    then a jump. Round 9 made the whole correction in flight the jump's
    level at its accept (off_jump = off), so the re-acquisition's leftover
    became a level the next re-acquisition was judged on — each read only
    its own dropout's drift (~22 cm) and the pose walked 1.6-1.9 m off in
    120 s with no stop (round 10, R10E-1, synthetic). Only the jump's own
    level change is a level: the chain ends the run."""
    for step in (-0.15, 0.15):
        T = Truth(v_b=(0.0, 0.0, 0.0), T=900.0)
        pose = _pose(require_gyro_bias=False)
        f = Feed(T, pose)
        f.run(6.0)
        f.run(1.0, engaged=True)
        f.dvl_bias = np.array([0.034, 0.0])
        t0 = f.t

        def level(t, t0=t0, step=step):
            k = int((t - t0) // 7.0) + (1 if (t - t0) % 7.0 >= 0.5 else 0)
            return np.array([step * (k % 2), 0.0])
        f.tag_jump = level
        out = f.run(120.0, tags=_view_cycle(t0, 1.0, 6.0), engaged=True)
        assert any(c["kind"] == "jump" and c["applied"] == "slew"
                   for c in pose.corrections), pose.corrections
        faults = [(t, h["fault"]) for t, _m, h, _tm in out if h["fault"]]
        assert faults, (step, max(_xy_err(T, t, m) for t, m, h, _tm in out
                                  if m is not None))
        assert "reacquire" in faults[0][1], faults[0]
        # (the tags' own 15 cm level is in the pose's error to the truth)
        worst = max(_xy_err(T, t, m) for t, m, h, _tm in out
                    if m is not None and t <= faults[0][0])
        assert worst < 0.5, (step, worst)


def test_a_jumps_level_is_its_own_step_not_the_correction_in_flight():
    """...the bookkeeping itself: a re-acquisition's ~20 cm leftover is
    slewing when the tags step 15 cm in view; the jump's level part is the
    step alone, the leftover stays drift."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(6.0)
    f.run(1.0, engaged=True)
    f.dvl_bias = np.array([0.04, 0.0])
    f.run(5.0, tags=False, engaged=True)            # ~20 cm of drift
    f.dvl_bias = np.zeros(2)
    c = _held_once(f, pose, 40)
    assert c["kind"] == "reacquire"
    assert not np.any(np.abs(pose.off_jump) > 1e-12)
    f.tag_jump = lambda t: np.array([-0.15, 0.0])
    c = _held_once(f, pose, 60)
    assert c["kind"] == "jump" and c["applied"] == "slew", c
    # off = the re-acquisition's leftover + the step; off_jump = the step
    # (pose - tag: the tags moved -x, so +0.15 m), within the hold's noise
    assert abs(float(pose.off_jump[0]) - 0.15) < 0.02, pose.off_jump
    assert float(pose.off[0]) - float(pose.off_jump[0]) > 0.10, (
        pose.off, pose.off_jump)


def test_a_held_jumps_own_drift_is_not_laundered_into_its_level():
    """The second path through the same line: a held jump whose dead
    reckoning drifts 20 cm (under the 25 cm watch) is accepted, then a
    dropout drifts 19 cm more and the tags come back on the jump's level.
    With the hold's drift in the level (round 9) the re-acquisition read
    only the 19 cm and slewed 32 cm; it is the error being flown (R10E-1).
    What the fixes cannot tell from their own noise — the jump gate, 10 cm —
    is not booked (round 10b, R11E-1)."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(6.0)
    f.run(1.0, engaged=True)
    real = f.tag
    f.tag = _never_agreeing_level(SimpleNamespace(tag=real), dxy=(0.0, 0.30))
    f.dvl_bias = np.array([0.035, 0.0])
    f.run(5.5, engaged=True)
    f.tag = _tag_level(SimpleNamespace(tag=real), dxy=(0.0, 0.30))
    c = _held_once(f, pose, 40)
    assert c["kind"] == "jump" and c["applied"] == "slew", c
    assert 0.15 < c["hold_drift_m"] < 0.25, c
    # (booked as drift: the hold's shift beyond the 10 cm jump gate)
    booked = float(pose.off[0] - pose.off_jump[0])
    assert abs(booked - (c["hold_drift_m"] - 0.10)) < 0.02, (booked, c)
    f.run(5.0, tags=False, engaged=True)
    c = _held_once(f, pose, 40)
    assert c["kind"] == "reacquire" and c["applied"] == "refused", c
    assert c["err_m"] > 0.25, c


def test_a_restored_hold_back_on_the_old_level_leaves_no_reference():
    """A held jump goes blind on a gyro 0.36 s late; the dead reckoning
    comes back on a tick whose tags read the OLD level. The hold restored on
    that tick stayed behind (its kind already cleared), and the next jump —
    at any later time — was judged against it: a clean 30 cm jump refused
    as 60 cm of drift with every sensor right (round 10, R10D-1)."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    st = {"late": 0.07}
    _queued_gyro(f, T, lambda t: st["late"])
    f.run(6.0)
    f.run(1.0, engaged=True)
    real = f.tag
    L0 = SimpleNamespace(tag=real)
    f.tag = _never_agreeing_level(L0, dxy=(0.30, 0.0))
    f.run(2.0, engaged=True)
    assert pose._pending_kind == "jump" and pose._hold_ref is not None
    st["late"] = 0.36
    for _ in range(2):
        f.run(0.05, engaged=True)
    assert pose._pending_kind == "blind"
    st["late"] = 0.07
    f.tag = real
    f.run(0.05, engaged=True)                       # the DR back, old level
    assert pose._pending_kind == "" and pose._hold_ref is None, (
        pose._pending_kind, pose._hold_ref)
    f.run(3.0, engaged=True)
    for j2 in ((-0.30, 0.0), (0.0, 0.15)):
        f.tag = _tag_level(L0, dxy=j2)
        out = []
        n0 = len(pose.corrections)
        for _ in range(60):
            out += f.run(0.05, engaged=True)
            if len(pose.corrections) > n0:
                break
        c = pose.corrections[-1]
        assert c["kind"] == "jump" and c["applied"] == "slew", (j2, c)
        assert c["hold_drift_m"] < 0.05, (j2, c)
        assert not any(h["fault"] for _t, _m, h, _tm in out), j2
        f.tag = real
        f.run(30.0, engaged=True)                   # (the slew back done)


def test_a_blind_accept_back_on_the_level_a_jump_left_is_no_hold_drift():
    """A 40 cm level is held as a jump; the gyro goes 0.36 s late (no DR)
    just as the camera is back on the OLD level, whose raw poses agree at
    once. Round 9 bounded that blind accept by the hold's drift against the
    jump's reference, so the whole jump read as drift: 'drifted 40 cm while
    a tag jump was held' with every sensor right and a correction of 0
    (round 10, R9-1). The jump is over: a blind re-acquisition of the old
    level, judged as drift (none)."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    st = {"late": 0.07}
    _queued_gyro(f, T, lambda t: st["late"])
    f.run(6.0)
    f.run(1.0, engaged=True)
    real = f.tag
    f.tag = _never_agreeing_level(SimpleNamespace(tag=real), dxy=(0.40, 0.0))
    f.run(2.0, engaged=True)
    assert pose._pending_kind == "jump"
    st["late"] = 0.36
    f.tag = real                                    # back on the OLD level
    out = f.run(0.5, engaged=True)
    st["late"] = 0.07
    out += f.run(3.0, engaged=True)
    assert not any(h["fault"] for _t, _m, h, _tm in out), [
        h["fault"] for _t, _m, h, _tm in out if h["fault"]][:1]
    c = pose.corrections[-1]
    assert c["kind"] == "blind" and c["applied"] == "slew", c
    assert c["err_m"] < 0.02, c


def test_z_never_ends_an_on_run_on_the_re_acquisition_watch():
    """ON, engaged, every sensor right. (a) A re-acquisition on a cluster
    whose map z is 30 cm off is accepted (z slewed); a second dropout, then
    the same cluster seen noisily (its fixes never agree): the watch read
    the estimator's own z slew left over as divergence and ended the run
    ('24 cm z', round 9 — round 10, R10Z-1). (b) A first re-acquisition on
    a 25 cm map-z cluster seen as noisily ended it on z too (rounds 7-9).
    z never ends an ON run: the dead reckoning's z IS the Bar30."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(6.0)
    f.run(1.0, engaged=True)
    real = f.tag
    f.run(2.0, tags=False, engaged=True)
    f.tag = _z_level(real, 0.30)
    c = _held_once(f, pose, 40)
    assert c["kind"] == "reacquire" and c["applied"] == "slew", c
    assert abs(c["err_z_m"] - 0.30) < 0.02, c
    f.run(1.0, engaged=True)
    f.run(1.0, tags=False, engaged=True)
    assert abs(float(pose.off[2])) > 0.2            # the z slew left over
    f.tag = _never_agreeing_level(SimpleNamespace(tag=_z_level(real, 0.30)),
                                  dxy=(0.0, 0.0), wobble_m=0.06)
    out = f.run(4.0, engaged=True)
    assert not any(h["fault"] for _t, _m, h, _tm in out), [
        h["fault"] for _t, _m, h, _tm in out if h["fault"]][:1]
    # (b)
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(6.0)
    f.run(1.0, engaged=True)
    real = f.tag
    f.run(2.0, tags=False, engaged=True)
    f.tag = _never_agreeing_level(SimpleNamespace(tag=_z_level(real, 0.25)),
                                  dxy=(0.0, 0.0))
    out = f.run(4.0, engaged=True)
    assert not any(h["fault"] for _t, _m, h, _tm in out), [
        h["fault"] for _t, _m, h, _tm in out if h["fault"]][:1]


def test_the_off_divergence_abort_still_ends_a_run_on_z():
    """...OFF keeps its z abort (20 cm): the tags 30 cm off in z, for 1 s,
    end an engaged OFF run (the fault names the z it judged)."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(6.0)
    pose.set_tag_on(False, f.t, True)
    real = f.tag
    f.tag = _z_level(real, 0.30)
    out = f.run(3.0, engaged=True)
    fl = [h["fault"] for _t, _m, h, _tm in out if h["fault"]]
    assert fl and "30 cm z" in fl[0] and "diverged" in fl[0], fl[:1]


def test_a_jump_levels_z_does_not_decide_which_reading_is_drift():
    """A 25 cm x jump slewing, a dropout, the tags back ON the jump's x
    level but on a map z 19 cm off. With z an equal partner in the score
    both readings tied on it (0.95), so the re-acquisition was judged on
    the pose and the jump's x leftover was zeroed from off_jump; the next
    dropout's 12 cm drift then read 26 cm and ended the run (round 10,
    R10E-2). z only breaks a tie: judged on the jump's level."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(6.0)
    real = f.tag
    st = {"dz": 0.0}

    def tag(t, fresh=True):
        meas, fix = real(t, fresh)
        meas["eta"][0] += 0.25
        meas["eta"][2] += st["dz"]
        p = np.asarray(fix.p_ned, float).copy()
        p[0] += 0.25
        p[2] += st["dz"]
        fix.p_ned = p
        return meas, fix
    f.tag = tag
    c = _held_once(f, pose, 80)
    assert c["kind"] == "jump" and c["applied"] == "slew", c
    f.run(1.0, engaged=True)
    f.run(1.0, tags=False, engaged=True)
    st["dz"] = 0.19
    c = _held_once(f, pose, 40)
    assert c["kind"] == "reacquire" and c.get("judged_on") == "slew_target", c
    assert abs(float(pose.off_jump[0])) > 0.15, pose.off_jump
    f.run(0.5, engaged=True)
    f.dvl_bias = np.array([-0.06, 0.0])
    f.run(2.0, tags=False, engaged=True)
    f.dvl_bias = np.zeros(2)
    out = []
    n0 = len(pose.corrections)
    for _ in range(40):
        out += f.run(0.05, engaged=True)
        if len(pose.corrections) > n0:
            break
    c = pose.corrections[-1]
    assert c["applied"] == "slew" and c["err_m"] < 0.16, c
    assert not any(h["fault"] for _t, _m, h, _tm in out)


def test_two_dropouts_in_a_big_jumps_slew_are_both_on_its_level():
    """A 0.80 m tag level change slewing (~40 s); two 0.5 s dropouts 4 s
    apart, the tags back on the jump's level each time, every sensor right.
    Each return is judged on the slew's target and slewed: the second one
    too — off_jump is KEPT at an on-target re-acquisition (round 9; mutant
    N31, zeroing it, refused the second as 57 cm: round 10, R10-2)."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(6.0)
    f.tag_jump = lambda t: np.array([0.80, 0.0])
    c = _held_once(f, pose, 80)
    assert c["kind"] == "jump" and c["applied"] == "slew", c
    for _k in range(2):
        f.run(4.0, engaged=True)
        f.run(0.5, tags=False, engaged=True)
        n1 = len(pose.corrections)
        out = f.run(2.0, engaged=True)
        assert not any(h["fault"] for _t, _m, h, _tm in out)
        cs = pose.corrections[n1:]
        assert cs and cs[0]["kind"] == "reacquire", cs
        assert cs[0]["applied"] == "slew", cs
        assert cs[0].get("judged_on") == "slew_target", cs
        assert cs[0]["err_m"] < 0.02, cs
        assert np.allclose(pose.off_jump[:2], pose.off[:2], atol=0.01), (
            pose.off, pose.off_jump)


def test_a_jump_held_long_during_a_slew_is_accepted_on_its_own_drift():
    """A 0.60 m jump slewing; 1 s later a second level 0.30 m across is
    held 12-16 s (its fixes never agree, the slew moving the track all
    along), then agrees. Every sensor right: its accept bound reads the
    hold's drift with each fix's own correction in flight added back
    (_pending_off, trimmed with the fixes) — about 0, and it is slewed.
    Mutants N8 / N16b refused it as 32-35 cm of drift (round 10, R10-3)."""
    for hold_s in (12.0, 16.0):
        T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
        pose = _pose(require_gyro_bias=False)
        f = Feed(T, pose)
        f.run(6.0)
        real = f.tag
        first = _tag_level(SimpleNamespace(tag=real), dxy=(0.60, 0.0))
        f.tag = first
        c = _held_once(f, pose, 80)
        assert c["kind"] == "jump" and c["applied"] == "slew", c
        f.run(1.0, engaged=True)
        f.tag = _never_agreeing_level(SimpleNamespace(tag=first),
                                      dxy=(0.0, 0.30))
        out = f.run(hold_s, engaged=True)
        assert pose._pending_kind == "jump"
        assert pose.corr_m() > 0.2                  # the slew still on
        f.tag = _tag_level(SimpleNamespace(tag=first), dxy=(0.0, 0.30))
        n1 = len(pose.corrections)
        out += f.run(3.0, engaged=True)
        assert not any(h["fault"] for _t, _m, h, _tm in out)
        cs = pose.corrections[n1:]
        assert cs and cs[0]["kind"] == "jump" and cs[0]["applied"] == "slew"
        assert cs[0]["hold_drift_m"] < 0.1, cs[0]


def test_a_resumed_hold_back_on_the_old_level_ends():
    """A held jump (cluster A) goes blind on a 2-tick gyro stall; the dead
    reckoning comes back on a tick with NO fix (the hold resumes with
    nothing pending); the next fixes are on the old level: the hold is over.
    A later cluster B is its own hold, with its own reference — mutant N19
    watched it against A's and stopped a right pose (round 10, R10-4)."""
    for phase in range(8):
        T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
        pose = _pose(require_gyro_bias=False)
        f = Feed(T, pose)
        st = {"a": None, "b": None}

        def late(t, st=st):
            if st["a"] is not None and st["a"] <= t < st["b"]:
                return 0.36
            return 0.07
        _queued_gyro(f, T, late)
        f.run(6.0)
        f.run(1.0 + 0.05 * phase, engaged=True)
        real = f.tag
        f.tag = _never_agreeing_level(SimpleNamespace(tag=real),
                                      dxy=(0.30, 0.0))
        f.run(2.0, engaged=True)
        st["a"], st["b"] = f.t + 0.05, f.t + 0.15
        seen_blind = resumed = False
        for _ in range(12):
            f.run(0.05, engaged=True)
            seen_blind |= pose._pending_kind == "blind"
            if seen_blind and pose._pending_kind == "jump" \
                    and not pose._pending:
                resumed = True
                break
        if not resumed:
            continue
        f.tag = real                                # back on the OLD level
        f.run(3.0, engaged=True)
        assert pose._pending_kind == "" and pose._hold_ref is None
        f.tag = _never_agreeing_level(SimpleNamespace(tag=real),
                                      dxy=(0.0, 0.30))
        out = f.run(6.0, engaged=True)
        assert not any(h["fault"] for _t, _m, h, _tm in out), [
            h["fault"] for _t, _m, h, _tm in out if h["fault"]][:1]
        assert pose.n_jumps_held == 2
        return
    raise AssertionError("no phase gave a no-fix return tick")


def test_a_blind_from_jump_accept_is_bounded_by_the_holds_drift():
    """A held jump (a cluster 0.30 m off that never agrees) on a DVL
    3.5 cm/s wrong: the hold's drift passes 25 cm, and before the watch's
    1 s is up the gyro goes 0.5 s late (no DR) while the cluster turns
    clean — the held jump is accepted BLIND on its level. That accept is
    bounded by the hold's drift too: refused (mutant N17, no bound, slewed
    the drift in as a 'jump': round 10, R10-5)."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    st = {"late_from": None}

    def late(t):
        return 0.5 if (st["late_from"] is not None
                       and t >= st["late_from"]) else 0.07
    _queued_gyro(f, T, late)
    f.run(6.0)
    f.run(1.0, engaged=True)
    real = f.tag
    f.tag = _never_agreeing_level(SimpleNamespace(tag=real), dxy=(0.30, 0.0))
    f.dvl_bias = np.array([-0.035, 0.0])
    f.run(8.4 - 0.35, engaged=True)
    st["late_from"] = f.t + 0.05
    n1 = len(pose.corrections)
    f.run(0.35, engaged=True)
    assert pose._pending_kind == "blind" and pose._blind_from_jump
    f.tag = _tag_level(SimpleNamespace(tag=real), dxy=(0.30, 0.0))
    out = f.run(1.5, engaged=True)
    cs = pose.corrections[n1:]
    assert cs and cs[0]["kind"] == "jump" and cs[0]["applied"] == "refused"
    assert cs[0]["hold_drift_m"] > 0.25, cs[0]
    fl = [h["fault"] for _t, _m, h, _tm in out if h["fault"]]
    assert fl and "while a tag jump was held" in fl[0], fl[:1]


def test_a_one_shot_is_done_with_a_glimpse_still_pending():
    """The 'done' half of round 9's one-shot OFF: its correction accepted
    and slewing, a dropout, a 2-fix glimpse (a re-acquisition 2/5, never
    completed), the slew ends on the dead reckoning. 'done' does not wait
    for those fixes (mutant N23, the round-8 rule, waited for as long as
    the tags stayed away: round 10, R10-6)."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(6.0)
    pose.set_tag_on(False, f.t, True)
    f.dvl_bias = np.array([0.03, 0.0])
    f.run(4.0, tags=False, engaged=True)            # 12 cm of drift
    f.dvl_bias = np.zeros(2)
    pose.start_oneshot(f.t, True)
    c = _held_once(f, pose, 80)
    assert c["kind"] == "reacquire" and c["applied"] == "slew", c
    f.run(1.0, tags=False, engaged=True)
    f.run(0.15, engaged=True)                       # a glimpse: 2 fixes
    f.run(12.0, tags=False, engaged=True)           # the slew ends on DR
    assert pose.corr_m() <= 1e-3 and pose._pending
    assert pose.oneshot_due(f.t) == "done"
    assert pose.oneshot_refusal(f.t) == ""


def test_a_yaw_jumps_level_decays_with_its_slew():
    """A 24 deg yaw tag level change slewing at 1 deg/s; at mid-slew a
    0.5 s dropout, the tags back on the jump's level, every sensor right.
    off_jump's yaw decays with off's: judged on the slew's target (mutant
    N3, not decaying it, refused it as 12 deg: round 10, R10-7)."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(6.0)
    f.tag = _tag_level(SimpleNamespace(tag=f.tag), dyaw_deg=24.0)
    c = _held_once(f, pose, 200)
    assert c["kind"] == "jump" and c["applied"] == "slew", c
    f.run(11.0, engaged=True)                       # to mid-slew
    assert abs(float(pose.off_jump[3]) - float(pose.off[3])) < 1e-6
    assert 8.0 < abs(math.degrees(float(pose.off[3]))) < 16.0
    f.run(0.5, tags=False, engaged=True)
    n1 = len(pose.corrections)
    out = f.run(3.0, engaged=True)
    assert not any(h["fault"] for _t, _m, h, _tm in out)
    cs = pose.corrections[n1:]
    assert cs and cs[0]["applied"] == "slew", cs
    assert cs[0].get("judged_on") == "slew_target", cs
    assert cs[0]["err_yaw_deg"] < 1.0, cs


def test_the_on_watch_counts_a_re_acquisitions_leftover():
    """ON, a station hold, the DVL 3.4 cm/s wrong: a 6 s dropout, a ~22 cm
    re-acquisition slewing (its leftover is the error being flown), another
    2 s dropout, then a view whose fixes never agree — only the ON watch can
    stop it. It judges the leftover in (only the jump part is discounted,
    _offj_e): stopped within 2.5 s of the view, the flown pose under 30 cm
    (mutant N12, discounting all of off, stopped at 6.75 s and 34 cm:
    round 10, R10-8)."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(6.0)
    f.run(1.0, engaged=True)
    f.dvl_bias = np.array([0.034, 0.0])
    f.run(6.0, tags=False, engaged=True)
    n0 = len(pose.corrections)
    f.run(0.5, engaged=True)
    cs = pose.corrections[n0:]
    assert cs and cs[0]["kind"] == "reacquire" and cs[0]["applied"] == "slew"
    f.run(2.0, tags=False, engaged=True)
    f.tag = _never_agreeing_level(SimpleNamespace(tag=f.tag), dxy=(0.0, 0.0),
                                  wobble_m=0.06)
    out = f.run(15.0, engaged=True)
    fl = [(t, h["fault"]) for t, _m, h, _tm in out if h["fault"]]
    assert fl and "diverged" in fl[0][1], fl[:1]
    assert fl[0][0] - out[0][0] < 2.5, fl[0]
    assert max(_xy_err(T, t, m) for t, m, h, _tm in out
               if m is not None and t <= fl[0][0]) < 0.30
    assert " cm z" not in fl[0][1]                  # (z not judged under ON)


# =============================================================================
# fix verification round 10b (the eleventh review, wf_1e670663-eb4)
# =============================================================================
def _wobble_x(src, dx, w, phase):
    """Fixes on a level ``dx`` off in x whose consecutive fixes alternate
    +/- ``w`` (the hold's first fixes then carry a median bias)."""
    st = {"tc": None, "k": phase}

    def tag(t, fresh=True):
        meas, fix = src(t, fresh)
        if fix.t_capture != st["tc"]:
            st["tc"] = fix.t_capture
            st["k"] += 1
        d = np.array([dx + (w if st["k"] % 2 else -w), 0.0])
        meas["eta"][0:2] = meas["eta"][0:2] + d
        p = np.asarray(fix.p_ned, float).copy()
        p[0:2] += d
        fix.p_ned = p
        return meas, fix
    return tag


def test_a_noisy_held_level_books_no_drift_into_the_correction():
    """A 0.60 m level held on fixes that wobble +/- 6 cm (the hold's first
    fixes biased), then a clean view of it: accepted. Round 10 took the
    jump's part from the hold's first fixes, so their bias stayed in the
    correction as 'drift' — the next on-target re-acquisition read 4 cm with
    no drift at all, slewed a true 27.5 cm drift as 23.5 and refused a true
    23.9 cm one as 27.8 (round 10b, R11E-1). The jump's part is what the
    accept changed: no drift is booked from the hold's noise."""
    for phase in (0, 1):
        for bias, want in ((0.0, "none"), (-0.0345, "refused"),
                           (-0.0300, "slew")):
            T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
            pose = _pose(require_gyro_bias=False)
            f = Feed(T, pose)
            f.run(6.0)
            f.run(1.0, engaged=True)
            real = f.tag
            f.tag = _wobble_x(real, 0.60, 0.06, phase)
            f.run(2.0, engaged=True)
            assert pose._pending_kind == "jump"
            f.tag = _tag_level(SimpleNamespace(tag=real), dxy=(0.60, 0.0))
            c = _held_once(f, pose, 80)
            assert c["kind"] == "jump" and c["applied"] == "slew", c
            assert np.allclose(pose.off_jump, pose.off, atol=1e-3), (
                phase, pose.off, pose.off_jump)
            f.run(1.0, engaged=True)
            f.dvl_bias = np.array([bias, 0.0])
            f.run(8.0, tags=False, engaged=True)
            f.dvl_bias = np.zeros(2)
            n0 = len(pose.corrections)
            f.run(3.0, engaged=True)
            r = pose.corrections[n0]
            assert r["kind"] == "reacquire", r
            if want == "none":
                assert r["err_m"] < 0.01 and r["applied"] == "slew", (phase, r)
            else:
                assert r["applied"] == want, (phase, bias, r)


def test_a_jump_levels_yaw_does_not_decide_which_reading_is_drift():
    """The yaw twin of the z test above: a 25 cm x jump slewing, a dropout,
    the tags back ON the jump's x level on a map yaw 9 deg off. Both
    readings carried the same yaw, which tied the larger of x-y and yaw
    (0.90), so the re-acquisition was judged on the pose and the jump's x
    leftover zeroed; the next dropout's 12 cm drift read 26 cm and ended the
    run (rounds 9-10; round 10b, R11E-2). The sum breaks the tie."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(6.0)
    real = f.tag
    st = {"dy": 0.0}

    def tag(t, fresh=True):
        meas, fix = real(t, fresh)
        dy = math.radians(st["dy"])
        meas["eta"][0] += 0.25
        meas["eta"][5] = DP.wrap(meas["eta"][5] + dy)
        R = np.asarray(fix.R_ned_body, float).reshape(3, 3)
        fix.R_ned_body = rot_zyx(0.0, 0.0, dy) @ R
        p = np.asarray(fix.p_ned, float).copy()
        p[0] += 0.25
        fix.p_ned = p
        return meas, fix
    f.tag = tag
    c = _held_once(f, pose, 80)
    assert c["kind"] == "jump" and c["applied"] == "slew", c
    f.run(1.0, engaged=True)
    f.run(1.0, tags=False, engaged=True)
    st["dy"] = 9.0
    c = _held_once(f, pose, 40)
    assert c["kind"] == "reacquire" and c.get("judged_on") == "slew_target", c
    assert abs(float(pose.off_jump[0])) > 0.15, pose.off_jump
    f.run(0.5, engaged=True)
    f.dvl_bias = np.array([-0.06, 0.0])
    f.run(2.0, tags=False, engaged=True)
    f.dvl_bias = np.zeros(2)
    out = []
    n0 = len(pose.corrections)
    for _ in range(40):
        out += f.run(0.05, engaged=True)
        if len(pose.corrections) > n0:
            break
    c = pose.corrections[-1]
    assert c["applied"] == "slew" and c["err_m"] < 0.16, c
    assert not any(h["fault"] for _t, _m, h, _tm in out)


def test_a_refusal_names_no_z_limit_it_does_not_have():
    """The too-big refusal printed the tags' z and a '0.2 m' z limit although
    only x-y and yaw refuse (round 10b, R11E-3)."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(6.0)
    f.run(1.0, engaged=True)
    f.dvl_bias = np.array([0.06, 0.0])
    f.run(6.0, tags=False, engaged=True)            # ~36 cm of drift
    f.dvl_bias = np.zeros(2)
    out = f.run(1.0, engaged=True)
    fl = [h["fault"] for _t, _m, h, _tm in out if h["fault"]]
    assert fl and "disagree" in fl[0], fl[:1]
    assert "cm z" not in fl[0] and "0.2 m" not in fl[0], fl[0]
    assert "z is slewed, never refused" in fl[0], fl[0]


def test_a_held_jumps_yaw_shift_beyond_the_gate_is_booked_as_drift():
    """The yaw twin of the x-y booking: a held jump whose agreeing fixes sit
    8 deg in yaw from the hold's first fixes, with a yaw correction in flight
    at the accept. That shift reads as the dead reckoning's heading drift
    over the hold: the 3 deg beyond the 5 deg jump gate is booked as drift,
    the rest stays the jump's (round 10b; a mutant that never booked yaw
    survived the round-10b battery)."""
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    pose = _pose(require_gyro_bias=False)
    f = Feed(T, pose)
    f.run(6.0)
    f.run(1.0, engaged=True)
    real = f.tag
    first = _tag_level(SimpleNamespace(tag=real), dyaw_deg=12.0)
    f.tag = _never_agreeing_level(SimpleNamespace(tag=first), dxy=(0.30, 0.0))
    f.run(2.0, engaged=True)
    assert pose._pending_kind == "jump"
    f.tag = _tag_level(SimpleNamespace(tag=real), dxy=(0.30, 0.0),
                       dyaw_deg=4.0)
    c = _held_once(f, pose, 60)
    assert c["kind"] == "jump" and c["applied"] == "slew", c
    assert abs(c["hold_drift_yaw_deg"] - 8.0) < 0.5, c
    booked = math.degrees(DP.wrap(float(pose.off[3] - pose.off_jump[3])))
    assert abs(booked - 3.0) < 0.3, (booked, pose.off, pose.off_jump)


# =============================================================================
# trust mode (2026-10-19, the operator's decision — module docstring, TRUST)
# =============================================================================
def test_resolve_knows_the_trust_keys():
    c = DP.resolve({"trust_dvl": True, "bias_seed_max_age_h": 0})
    assert c["trust_dvl"] is True and c["bias_seed_max_age_h"] == 0.0
    assert DP.resolve(None)["trust_dvl"] is False
    with pytest.raises(ValueError, match="trust_dvl"):
        DP.resolve({"trust_dvl": "yes"})
    with pytest.raises(ValueError, match="bias_seed_max_age_h"):
        DP.resolve({"bias_seed_max_age_h": -1})


def test_trust_mode_lets_off_fly_before_the_gyro_bias_is_learnt():
    """Strict: OFF waits for the bias fit (bias_min_span_s of TAG FIX ON —
    60 s on the station: 1019_211310 waited from 21:11:11 to 21:12:47).
    Trust: the anchor is enough; the bias is learnt whenever ON and used
    once ok."""
    for trust in (False, True):
        T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
        pose = _pose(trust_dvl=trust)
        f = Feed(T, pose)
        f.run(2.0)
        assert pose.anchored and pose.head.bias_state != "ok"
        why = pose.tag_refusal(False, f.t)
        if trust:
            assert why == "", why
            pose.set_tag_on(False, f.t, False)
            out = f.run(2.0)
            assert out[-1][2]["state"] == DP.ST_DR
        else:
            assert "gyro bias" in why, why


def test_trust_mode_flies_across_a_dvl_gap_with_a_warning_not_a_hole():
    """Strict: a DVL gap past lost_s after the anchor is a HOLE — no START
    until the tags re-anchor (1019_211310: 'a DVL gap of 4.9 s after the
    anchor (position unknown — TAG FIX ON to re-anchor)', the vehicle on
    the floor). Trust: the pose resumes from where the gap found it, says
    so, counts it, and a START is not refused."""
    for trust in (False, True):
        T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
        pose = _pose(trust_dvl=trust, require_gyro_bias=False)
        f = Feed(T, pose)
        f.run(6.0)
        pose.set_tag_on(False, f.t, False)
        f.run(1.0)
        t0 = f.t
        f.dvl_on = lambda t, t0=t0: not (t0 < t < t0 + 5.0)
        out = f.run(5.0, tags=False)                # the gap: LOST either way
        assert out[-1][2]["state"] == DP.ST_LOST, out[-1][2]
        out = f.run(10.0, tags=False)               # the DVL back
        h = out[-1][2]
        if trust:
            assert h["state"] == DP.ST_DR and h["ok"], h
            assert pose.dr_refusal(f.t) == "", pose.dr_refusal(f.t)
            assert pose.n_gaps_trusted == 1 and pose.n_xy_holes == 0
            # (the warning stays until the tags re-anchor — not 6 s)
            assert "flown across" in h["warn"], h["warn"]
            assert pose.meta()["gaps_trusted"] == 1
        else:
            assert h["state"] == DP.ST_LOST, h
            assert "position unknown" in pose.dr_refusal(f.t)
            assert pose.n_xy_holes == 1 and pose.n_gaps_trusted == 0


def test_a_seeded_bias_is_in_force_until_the_sessions_own_fit():
    """A bias another session saved (the worker's file) is the heading's
    from the start — 'saved' until this session's own fit is ok."""
    pose = _pose()
    pose.head.seed_bias(0.05, 0.01, 100.0, 2.0)
    assert pose.head.bias_state == "saved"
    assert pose.head.bias_info()["dps"] == pytest.approx(0.05)
    assert pose.meta()["bias_seed"] == {"dps": 0.05, "se_dps": 0.01,
                                        "age_h": 2.0}
    T = Truth(v_b=(0.0, 0.0, 0.0), T=300.0)
    f = Feed(T, pose)
    f.run(1.0)
    assert pose.anchored
    assert math.degrees(pose.head.b_anchor) == pytest.approx(0.05)
    assert pose.head.bias_state == "saved"      # the window is 20 s here
    f.run(25.0)
    assert pose.head.bias_state == "ok"
    assert abs(pose.head.bias_info()["dps"]) < 0.02   # the feed's gyro: none


def test_trust_mode_never_ends_a_run_on_the_unintegrated_motion():
    """Short DVL gaps (inside lost_s) at 0.10 m/s each hide ~7.5 cm of
    motion. Past max_corr_m the `unobserved` stop (safety review 2026-10-19
    #2, #3) ended an engaged run — until the operator, 2026-10-29: "I am
    always watching; I will stop it". Engaged it is now the gap warning
    only, the estimate growing past the line with nothing stopped;
    disengaged it is still the START refusal, until a ONCE with tags in
    view snaps it away."""
    for engaged in (True, False):
        T = Truth(v_b=(0.10, 0.0, 0.0), T=300.0)
        pose = _pose(trust_dvl=True, require_gyro_bias=False)
        f = Feed(T, pose)
        f.run(6.0)
        pose.set_tag_on(False, f.t, engaged)
        f.run(1.0, engaged=engaged)
        faults = []
        for _k in range(8):
            t0 = f.t
            f.dvl_on = lambda t, t0=t0: not (t0 < t < t0 + 1.2)
            out = f.run(2.2, tags=False, engaged=engaged)
            faults += [h["fault"] for _t, _m, h, _tm in out if h["fault"]]
            if pose.unobserved_m > 0.35:
                break
        assert pose.n_xy_holes == 0 and pose.unobserved_m > 0.35, \
            pose.unobserved_m
        h = out[-1][2]
        assert h["ok"] and h["state"] == DP.ST_DR, h
        # the warning carries the number (the operator's live cue)
        assert (f"{pose.unobserved_m * 100:.0f} cm of unintegrated motion"
                in h["warn"]) and "flown across" in h["warn"], h["warn"]
        assert not [w for _t, w in pose.faults if "unintegrated" in w], \
            pose.faults
        if engaged:
            assert not faults, faults[:1]
        else:
            assert faults and "unintegrated motion" in faults[0] \
                and "while disengaged" in faults[0], faults[:1]
            assert "unintegrated motion" in h["fault"], h
        # the tags back, ON. Disengaged: a snap re-anchors, and the gap
        # warning goes with the unobserved budget. Engaged: the pose really
        # lags the truth by that much, so the re-acquisition is past
        # max_corr and refused — the correction limit, which stays, ends
        # that run (an ON press with the tags already in view is refused in
        # words first: tag_refusal)
        pose.set_tag_on(True, f.t, engaged)
        f.dvl_on = lambda t: True
        n0 = len(pose.corrections)
        out = f.run(3.0, engaged=engaged)
        h = out[-1][2]
        if engaged:
            cs = pose.corrections[n0:]
            assert cs and cs[0]["applied"] == "refused" and cs[0]["err_m"] > 0.25, cs
        else:
            assert pose.unobserved_m == 0.0
            assert "unintegrated" not in h["fault"], h
            assert "flown across" not in h["warn"], h
