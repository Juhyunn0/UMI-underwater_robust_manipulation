#!/usr/bin/env python3
"""test_dvl.py — the DVL as the estimator's third sensor (control/dvl.py).

NO DVL HAS BEEN ON THIS VEHICLE. These tests pin the LOGIC — the wire
format, the frames, the gates, the switch rules, the record — against
synthetic data and against one capture of the vendor's public demo unit
(tests/fixtures/dvl_a50_demo_capture_20261001.jsonl). They are not, and cannot
be, evidence that any threshold in hw_mpc.yaml ``dvl:`` suits this pool.

Halves, in the order a failure should be read:

    wire      the vendor capture + the vendor's documented example parse
    geometry  mount rotation, lever arm, dead-reckoning step
    agree     the "proven" fit: a good mount passes, every wrong one is named
    assembler vel_body=None is BIT-identical to the pre-DVL arithmetic
    aid       gates, hysteresis, the switch; the trips are LATCHED and do
              not depend on require_proven; the device-clock map
    worker    the real MpcWorker (stub controller): off = unchanged; the
              switch is frozen while engaged and eligibility is decided at
              engage; fallback; the bridge's DVL rung and its bounds
    client    the TCP reader against a local fake server; the temperature
              poller against a local fake status page (GET only), and the
              SENSORS row it feeds — the temperature is said BEFORE the lock
    panel     the button reflects the worker and never flips itself, and an
              IN switch is readable (rendered, measured contrast)

    QT_QPA_PLATFORM=offscreen \\
      ~/miniforge3/envs/robust/bin/python rov_gui/tests/test_dvl.py
"""

from __future__ import annotations

import json
import math
import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np                                           # noqa: E402

from rov_gui.control import dvl as DV                        # noqa: E402
from rov_gui.control import station_bridge as SB             # noqa: E402
from rov_gui.control.state_assembler import (G_NED, StateAssembler,   # noqa: E402
                                             rot_zyx)
from rov_gui.state import Conn, DvlSample, MpcStatus, now    # noqa: E402

# In tests/fixtures/, not tests/data/: the repo ignores every directory named
# `data` (.gitignore), and this file has to exist on a fresh clone.
CAPTURE = (Path(__file__).parent / "fixtures"
           / "dvl_a50_demo_capture_20261001.jsonl")

#: The vendor's own documented example (docs.waterlinked.com/dvl/
#: dvl-json-protocol, read 2026-10-01), transducers trimmed to one.
DOC_EXAMPLE = {
    "time": 106.3935775756836, "vx": -3.713480691658333e-05,
    "vy": 5.703703573090024e-05, "vz": 2.4990416932269e-05,
    "fom": 0.00016016385052353144,
    "covariance": [[2.4471841442164077e-08, -3.3937477272871774e-09,
                    -1.6659699175747278e-09],
                   [-3.3937477272871774e-09, 1.4654466085062268e-08,
                    4.0409570134514183e-10],
                   [-1.6659699175747278e-09, 4.0409570134514183e-10,
                    1.5971971523143225e-09]],
    "altitude": 0.4949815273284912,
    "transducers": [{"id": 0, "velocity": 0.0001, "distance": 0.5568,
                     "rssi": -30.49, "nsd": -88.73, "beam_valid": True}],
    "velocity_valid": True, "status": 0, "format": "json_v3.3",
    "type": "velocity", "time_of_validity": 1638191471563017,
    "time_of_transmission": 1638191471752336,
}


def _harness():
    from rov_gui.tests import test_control as TC
    return TC


def _sample(v=(0.0, 0.0, 0.0), t=None, valid=True, fom=0.002, alt=0.5,
            kind="velocity"):
    t = now() if t is None else t
    return DvlSample(t_valid=t, t_rx=t, v_dvl=tuple(float(x) for x in v),
                     valid=valid, fom=fom, altitude=alt, n_beams=4,
                     kind=kind, hz=10.0, source="test", conn=Conn.ONLINE,
                     stamp=t)


MOUNT = {"mount_rpy_deg": [0.0, 0.0, 0.0], "mount_t_flu": [0.0, 0.0, 0.0]}


# ===================================================================== config
def test_the_shipped_config_spells_known_dvl_keys_with_the_code_defaults():
    """(2026-10-16: was ..._has_the_dvl_off_and_the_mount_unmeasured, which
    failed since the operator measured the mount.) The DVL is off and needs
    the proof; the file names only keys the code knows, and every key it does
    not name is one of the 2026-10-16 selection keys (none once the operator
    applies RL_controller logs/eval/dvl_trip_fix_20261015/
    hw_mpc_dvl_patch.diff); every spelled key but the measured mount says
    what the code would have assumed."""
    from rov_gui.control.geometry import MpcConfig

    mc = MpcConfig.load(ROOT / "config" / "hw_mpc.yaml")
    cfg = DV.resolve(mc.dvl)
    assert cfg["enabled"] is False and cfg["use"] is False
    assert cfg["require_proven"] is True
    assert cfg["port"] == 16171
    # The file SPELLS OUT every key, and says what the code would have
    # assumed: a number changed in one place and not the other is a run
    # flown on a gate nobody chose.
    assert set(mc.dvl) <= set(DV.KEYS), set(mc.dvl) - set(DV.KEYS)
    assert set(DV.KEYS) - set(mc.dvl) <= set(DV.KEYS_20261016), \
        set(DV.KEYS) - set(mc.dvl)
    mount = ("mount_rpy_deg", "mount_t_flu")
    assert all(cfg[k] == DV.DEFAULTS[k] for k in mc.dvl if k not in mount), \
        {k: cfg[k] for k in mc.dvl if k not in mount and cfg[k] != DV.DEFAULTS[k]}
    # the mount: both unmeasured, or both measured (3 finite numbers each)
    if cfg["mount_rpy_deg"] is None or cfg["mount_t_flu"] is None:
        assert cfg["mount_rpy_deg"] is None and cfg["mount_t_flu"] is None
    else:
        for k in mount:
            assert len(cfg[k]) == 3 and all(math.isfinite(x) for x in cfg[k])
    assert cfg["selection"] == "primary"
    sb = SB.resolve(mc.station_bridge)
    assert set(mc.station_bridge) == set(SB.KEYS)
    assert all(sb[k] == SB.DEFAULTS[k] for k in SB.KEYS)


def test_a_typo_in_the_dvl_block_is_refused_not_ignored():
    for bad in ({"mount_rpy": [0, 0, 0]}, {"enabled": "yes"},
                {"mount_rpy_deg": [0, 0]}, {"stale_s": 0},
                {"fom_max": float("nan")}, {"port": 0},
                {"agree_min_pairs": 2}, {"mount_t_flu": [0, "x", 0]},
                # the 2026-10-16 keys
                {"selection": "Primary"}, {"selection": None},
                {"selection": True}, {"blend_xfade_s": 0},
                {"blend_hold_s": -1}, {"flag_min_beams": 5},
                {"flag_min_beams": 3.5}, {"flag_min_beams": True},
                {"flag_reacquired": 1}, {"agree_release_m_s": 0.06},
                {"esc_latch_s": 0}, {"esc_leak": -0.1}, {"flag_fom_m_s": -1},
                {"esc_scale_ratio": 0.9}, {"esc_repeat_n": 1},
                {"esc_repeat_n": 2.0}):
        try:
            DV.resolve(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")
    assert DV.resolve({"mount_rpy_deg": [0, 0, 90]})["mount_rpy_deg"] == [
        0.0, 0.0, 90.0]
    for good in ({"selection": "legacy"}, {"flag_min_beams": 0},
                 {"blend_gross_m_s": 0.0}, {"esc_scale_ratio": 0.0},
                 {"esc_repeat_n": 0}, {"flag_reacquired": False}):
        c = DV.resolve(good)
        assert all(c[k] == v for k, v in good.items()), (good, c)


def test_the_command_line_overrides_the_file_for_both_readers():
    class O:
        dvl = True
        dvl_host = "10.0.0.7"
        dvl_port = 1234
    cfg = DV.resolve_with_opts({"enabled": False}, O())
    assert cfg["enabled"] and cfg["host"] == "10.0.0.7" and cfg["port"] == 1234
    assert DV.resolve_with_opts(None, object())["enabled"] is False


# ======================================================================= wire
def test_the_vendor_demo_capture_parses_line_for_line():
    """Real bytes off a real (demo) device: 179 CRLF lines, two report
    kinds. Every velocity line must become a sample; every position_local
    line must be recognised and NOT become one."""
    raw = CAPTURE.read_bytes()
    lines = raw.split(b"\n")
    assert lines[-1] == b""                       # the capture ends on \r\n
    reps = [DV.parse_line(ln) for ln in lines[:-1]]
    assert all(r is not None for r in reps) and len(reps) == 179
    kinds = [r["type"] for r in reps]
    assert kinds.count("velocity") == 60
    assert kinds.count("position_local") == 119
    n = 0
    for r in reps:
        f = DV.sample_fields(r, t_rx=100.0)
        if r["type"] != "velocity":
            assert f is None
            continue
        n += 1
        s = DvlSample(**f)
        assert s.valid and s.v_dvl is not None and len(s.v_dvl) == 3
        assert s.n_beams == 4 and s.covariance is not None
        assert s.tov_us is not None and s.tot_us >= s.tov_us
        # arrival minus the device's own validity->transmission delay
        assert 100.0 - 0.01 <= s.t_valid <= 100.0
    assert n == 60


def test_time_of_validity_comes_from_the_difference_of_the_device_stamps():
    f = DV.sample_fields(DOC_EXAMPLE, t_rx=50.0)
    # 1638191471752336 - 1638191471563017 = 189319 us: the report describes
    # a velocity that was true 0.19 s before it was sent.
    assert abs(f["t_valid"] - (50.0 - 0.189319)) < 1e-9
    assert f["valid"] and f["n_beams"] == 1
    assert abs(f["altitude"] - 0.4949815273284912) < 1e-12
    # a device clock that jumped (NTP) must not move the sample into the
    # far past or the future: arrival time, and a note saying so
    r = dict(DOC_EXAMPLE, time_of_transmission=DOC_EXAMPLE["time_of_validity"]
             - 5_000_000)
    f = DV.sample_fields(r, t_rx=50.0)
    assert f["t_valid"] == 50.0 and "out of range" in f["note"]
    r = dict(DOC_EXAMPLE)
    del r["time_of_validity"]
    f = DV.sample_fields(r, t_rx=50.0)
    assert f["t_valid"] == 50.0 and f["tov_us"] is None


def test_a_report_cannot_be_valid_with_a_velocity_that_is_not_a_number():
    r = dict(DOC_EXAMPLE, vx=float("nan"))
    f = DV.sample_fields(json.loads(json.dumps(r)), t_rx=1.0)
    assert f["valid"] is False and f["v_dvl"] is None
    assert DV.sample_fields(dict(DOC_EXAMPLE, velocity_valid=False),
                            1.0)["valid"] is False


def test_garbage_on_the_wire_is_dropped_never_raised():
    for bad in (b"", b"\r", b"not json", b'{"half": ', b"\xff\xfe\x00",
                b"[1, 2, 3]", b'{"no_type": 1}'):
        assert DV.parse_line(bad) is None, bad
    assert DV.parse_line(b'{"type":"response","response_to":"get_config"}\r')



def test_nothing_a_line_can_hold_makes_the_parser_raise():
    """review 2026-10-02: Infinity stamps raised OverflowError, a scalar
    `transducers` raised TypeError and a deeply nested line raised
    RecursionError — the last one killed the reader thread for good."""
    weird = [
        dict(DOC_EXAMPLE, time_of_transmission=float("inf")),
        dict(DOC_EXAMPLE, time_of_validity="soon"),
        dict(DOC_EXAMPLE, status=1e999),
        dict(DOC_EXAMPLE, transducers=5),
        dict(DOC_EXAMPLE, transducers=[1, "x", None]),
        dict(DOC_EXAMPLE, covariance=[[1, 2], [3]]),
        dict(DOC_EXAMPLE, covariance="no"),
        dict(DOC_EXAMPLE, fom="bad", altitude=None, time=[1]),
        dict(DOC_EXAMPLE, velocity_valid=1),          # not the JSON `true`
        dict(DOC_EXAMPLE, velocity_valid="true"),
    ]
    for r in weird:
        f = DV.sample_fields(r, t_rx=3.0)
        assert f is not None and f["t_valid"] <= 3.0, r
    assert DV.sample_fields(weird[0], 3.0)["tov_us"] is None
    assert DV.sample_fields(weird[3], 3.0)["n_beams"] == 0
    # only the JSON literal true is a lock
    assert DV.sample_fields(weird[8], 3.0)["valid"] is False
    assert DV.sample_fields(weird[9], 3.0)["valid"] is False
    deep = b'{"type":"velocity","x":' + b"[" * 30000 + b"]" * 30000 + b"}"
    assert DV.parse_line(deep) is None


def test_the_device_clock_gives_a_backlog_its_true_age():
    """After a TCP stall the queued reports arrive together. Stamped by
    arrival they would all look fresh; mapped through the device's own
    clock they are as old as they really are."""
    ck = DV.DeviceClock()
    dev0 = 1_700_000_000_000_000           # device epoch, microseconds
    lat = 190_000                          # validity -> transmission
    # steady state: 10 Hz, 4 ms on the wire; station clock = device + 5000 s
    out = []
    for k in range(30):
        tot = dev0 + k * 100_000
        rep = dict(DOC_EXAMPLE, time_of_validity=tot - lat,
                   time_of_transmission=tot)
        t_rx = 5000.0 + tot * 1e-6 - dev0 * 1e-6 + 0.004
        out.append((t_rx, DV.sample_fields(rep, t_rx, clock=ck)))
    for t_rx, f in out[1:]:
        # the device's own validity->transmission delay; the network hop is
        # NOT subtracted (it cannot be known) — as DvlSample says
        assert abs((t_rx - f["t_valid"]) - 0.190) < 1e-5
    # a 3 s stall: 30 reports land in ONE chunk, 3 s after the first of them
    t_burst = out[-1][0] + 3.1
    ages = []
    for k in range(30, 60):
        tot = dev0 + k * 100_000
        rep = dict(DOC_EXAMPLE, time_of_validity=tot - lat,
                   time_of_transmission=tot)
        f = DV.sample_fields(rep, t_burst, clock=ck)
        ages.append(t_burst - f["t_valid"])
    assert abs(ages[0] - 3.19) < 0.01 and abs(ages[-1] - 0.29) < 0.01, (
        ages[0], ages[-1])
    assert ages == sorted(ages, reverse=True)
    # without the clock the oldest of them would have claimed 0.19 s
    f = DV.sample_fields(dict(DOC_EXAMPLE), t_burst)
    assert abs((t_burst - f["t_valid"]) - 0.189319) < 1e-5
    # the device clock stepping (NTP) starts the map over instead of
    # leaving every later sample hours "old"
    tot = dev0 + 10 ** 10
    rep = dict(DOC_EXAMPLE, time_of_validity=tot - lat,
               time_of_transmission=tot)
    f = DV.sample_fields(rep, t_burst + 1.0, clock=ck)
    assert ck.n_steps == 1
    assert abs((t_burst + 1.0 - f["t_valid"]) - 0.19) < 1e-5


# =================================================================== geometry
def test_the_mount_rotation_and_the_lever_arm():
    # LED forward, no offset: the device frame IS body FRD
    R, r, measured = DV.mount_from_cfg(DV.resolve(MOUNT))
    assert measured and np.allclose(R, np.eye(3)) and np.allclose(r, 0)
    # unmeasured = identity, zero arm, and it says so
    R0, r0, m0 = DV.mount_from_cfg(DV.resolve(None))
    assert not m0 and np.allclose(R0, np.eye(3)) and np.allclose(r0, 0)
    # the device turned 90 deg clockwise (seen from above): its +x is the
    # body's +y (starboard), so a device reading of pure +x is pure SWAY
    R, r, _ = DV.mount_from_cfg(DV.resolve(
        {"mount_rpy_deg": [0, 0, 90], "mount_t_flu": [0.2, 0.0, -0.1]}))
    v = DV.body_velocity((0.1, 0.0, 0.0), R, r, None)
    assert np.allclose(v, [0.0, 0.1, 0.0], atol=1e-12), v
    # the arm is given in FLU and used in FRD: 0.2 m forward, 0.1 m BELOW
    assert np.allclose(r, [0.2, 0.0, 0.1])
    # a pure yaw at 0.5 rad/s about the body origin, vehicle not translating:
    # the DVL point 0.2 m ahead moves to starboard at 0.1 m/s. The device
    # (90 deg mount) reads that as +x; the corrected origin velocity is zero.
    v = DV.body_velocity((0.1, 0.0, 0.0), R, r, (0.0, 0.0, 0.5))
    assert np.allclose(v, 0.0, atol=1e-12), v


def test_dead_reckoning_step_rotates_by_the_heading():
    p = DV.integrate_xy([1.0, 2.0], [0.2, 0.0, 0.0], 0.0, 0.0, math.pi / 2,
                        0.5)
    assert np.allclose(p, [1.0, 2.1], atol=1e-12)      # nose along +y


# ====================================================================== agree
def _pairs(ag, n=240, theta_deg=0.0, flip_y=False, scale=1.0, kind="both",
           seed=0):
    """Feed ``ag`` pairs for a device that really sits ``theta_deg``
    clockwise of the configured mount."""
    rng = np.random.default_rng(seed)
    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    for i in range(n):
        t = i / 15.0
        if kind == "both":
            v = np.array([0.12 * math.sin(2 * math.pi * t / 8.0),
                          0.08 * math.sin(2 * math.pi * (t - 3.0) / 8.0)])
        elif kind == "surge":
            v = np.array([0.12 * math.sin(2 * math.pi * t / 6.0), 0.0])
        else:
            v = np.array([0.004 * math.sin(t), 0.004 * math.cos(t)])
        vd = scale * np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1]])
        if flip_y:
            vd[1] = -vd[1]
        vd = vd + rng.normal(0, 0.003, 2)
        vf = v + rng.normal(0, 0.017, 2)
        ag.add(t, vd, vf)


def _ag():
    c = DV.DEFAULTS
    return DV.Agreement(c["agree_min_pairs"], c["agree_min_excitation_m_s"],
                        c["agree_tol"], c["agree_v_min_m_s"],
                        c["agree_window_s"])



def test_a_correct_mount_is_proven_by_motion_in_both_axes():
    ag = _ag()
    _pairs(ag)
    f = ag.fit()
    assert f["verdict"] == "proven", f
    assert abs(f["mount_yaw_fix_deg"]) < 3.0 and abs(f["scale"] - 1.0) < 0.05
    assert f["mirrored"] is False
    # the verdict carries its own margin: three standard errors inside tol
    assert f["dev"] + 3.0 * f["sigma_A"] <= DV.DEFAULTS["agree_tol"]


def test_a_rotated_mount_is_refused_and_the_fix_has_the_right_sign():
    ag = _ag()
    _pairs(ag, theta_deg=30.0)
    f = ag.fit()
    assert f["verdict"] == "refuted", f
    # the device sits 30 deg clockwise of what the config says: ADD +30
    assert abs(f["mount_yaw_fix_deg"] - 30.0) < 3.0, f
    assert "30" in f["why"]
    ag = _ag()
    _pairs(ag, theta_deg=180.0)                       # LED pointing aft
    f = ag.fit()
    assert f["verdict"] == "refuted"
    assert abs(abs(f["mount_yaw_fix_deg"]) - 180.0) < 3.0


def test_a_mirrored_axis_is_named_as_a_frame_error_not_a_rotation():
    ag = _ag()
    _pairs(ag, flip_y=True)
    f = ag.fit()
    assert f["verdict"] == "refuted" and f["mirrored"] is True
    assert "MIRRORED" in f["why"]


def test_no_verdict_without_motion_in_both_axes():
    """At rest a wrong mount and a right one read the same, and surge alone
    cannot tell a starboard-y device from a port-y one: neither may prove —
    or refute — anything."""
    for kind, flip in (("surge", False), ("surge", True), ("still", False),
                       ("still", True)):
        ag = _ag()
        _pairs(ag, kind=kind, flip_y=flip)
        f = ag.fit()
        assert f["verdict"] == "insufficient", (kind, flip, f)
        assert f["why"]


def test_a_wrong_scale_is_refused():
    ag = _ag()
    _pairs(ag, scale=1.5)
    f = ag.fit()
    assert f["verdict"] == "refuted" and abs(f["scale"] - 1.5) < 0.08, f



def test_a_handful_of_noisy_pairs_decides_nothing():
    """The verdict is three-way. With little data the standard error is as
    large as the tolerance, and the honest answer is "not yet" — for a
    right mount AND for a wrong one (review 2026-10-02: the point estimate
    alone proved a mount at exactly the minimum pair count)."""
    for theta in (0.0, 20.0):
        ag = _ag()
        rng = np.random.default_rng(4)
        c, s_ = math.cos(math.radians(theta)), math.sin(math.radians(theta))
        for i in range(70):                   # slow, and the tags are noisy
            t = i / 15.0
            v = np.array([0.06 * math.sin(t), 0.06 * math.cos(1.3 * t)])
            vd = np.array([c * v[0] + s_ * v[1], -s_ * v[0] + c * v[1]])
            ag.add(t, vd, v + rng.normal(0, 0.08, 2))
        f = ag.fit()
        assert f["verdict"] == "insufficient", (theta, f)
        assert "not decided yet" in f["why"], f["why"]
        assert f["sigma_A"] > 0.05


def test_outliers_that_cannot_be_trimmed_are_not_judged():
    """review 2026-10-02 (reproduced there through the real worker): one tag
    jump in a window that has shrunk to the minimum left `kept < min_pairs`,
    the trim was skipped, and the contaminated fit called a good mount
    "rotated 27 degrees". It must say "wait" instead."""
    ag = DV.Agreement(60, 0.025, 0.25, 0.02, 60.0)
    rng = np.random.default_rng(9)
    for i in range(60):
        t = i / 15.0
        v = np.array([0.10 * math.sin(2 * math.pi * t / 4.0),
                      0.08 * math.cos(2 * math.pi * t / 4.0)])
        vf = v + rng.normal(0, 0.012, 2)
        if i == 30:
            vf = vf + np.array([0.9, -0.7])       # a 6 cm tag jump at 15 Hz
        ag.add(t, v + rng.normal(0, 0.003, 2), vf)
    f = ag.fit()
    assert f["n_outliers"] == 1 and f["verdict"] == "insufficient", f
    assert "after setting aside 1 outlier" in f["why"], f["why"]
    assert f["A"] is None                     # nothing to mis-quote
    ag.add(4.1, [0.1, 0.02], [0.1, 0.02])     # one more honest pair
    assert ag.fit()["verdict"] == "proven", ag.fit()


# ================================================================== assembler
def _fix(p, yaw, t_cap):
    return _harness()._fix(p, yaw, t_cap)


def _imu(**kw):
    return _harness()._imu(**kw)


def _reference_step(st, fix, imu, t_now, dt, alpha, tag_stale_s=0.5):
    """The PRE-DVL StateAssembler.step, frozen here (z_source tag,
    nudot_source imu, propagate on) — written out from the 2026-09 file so
    the comparison below does not depend on the code it is checking."""
    p_tag = np.asarray(fix.p_ned, float)
    R_tag = np.asarray(fix.R_ned_body, float).reshape(3, 3)
    psi = math.atan2(R_tag[1, 0], R_tag[0, 0])
    age = min(max(0.0, t_now - float(fix.t_capture)), tag_stale_s)
    psi += float(imu.r) * age
    phi, theta = float(imu.roll), float(imu.pitch)
    R = rot_zyx(phi, theta, psi)
    p = np.array([p_tag[0], p_tag[1], float(p_tag[2])])
    tc = float(fix.t_capture)
    if st["t"] is not None and tc > st["t"] + 1e-6:
        v_new = (p_tag - st["p"]) / (tc - st["t"])
        st["v"] = (1.0 - alpha) * st["v"] + alpha * v_new
    if st["t"] is None or tc > st["t"] + 1e-6:
        st["t"], st["p"] = tc, p_tag.copy()
    if age > 0.0:
        p[0] += st["v"][0] * age
        p[1] += st["v"][1] * age
        p[2] += st["v"][2] * age
    nu_lin = R.T @ st["v"]
    gyro = np.array([float(imu.p), float(imu.q), float(imu.r)])
    nu = np.concatenate([nu_lin, gyro])
    ang = np.zeros(3) if st["g"] is None else (gyro - st["g"]) / dt
    st["g"] = gyro.copy()
    f = np.array([float(imu.ax), float(imu.ay), float(imu.az)])
    a_lin = f + R.T @ G_NED - np.cross(gyro, nu_lin)
    return (np.array([p[0], p[1], p[2], phi, theta, psi]), nu,
            np.concatenate([a_lin, ang]))


def test_without_a_dvl_the_assembler_is_bit_identical_to_before():
    """THE PIN. ``vel_body=None`` must be the pre-DVL arithmetic, operation
    for operation — equality here is ``==`` on float64, not a tolerance."""
    rng = np.random.default_rng(5)
    asm = StateAssembler(z_source="tag", vel_lp_alpha=0.6, nudot_source="imu")
    st = {"t": None, "p": None, "v": np.zeros(3), "g": None}
    t = 100.0
    p = np.array([0.3, -0.2, -0.9])
    for k in range(200):
        t_cap = t - 0.13
        if k % 2 == 0:                       # camera slower than the loop
            p = p + rng.normal(0, 0.004, 3) + np.array([0.004, 0.002, 0.0])
            fix = _fix(list(p), 0.4 + 0.01 * k, t_cap)
        imu = _imu(roll=0.02 * math.sin(k), pitch=0.03 * math.cos(k),
                   r=0.1 * math.sin(0.3 * k), t=t - 0.01,
                   f=(0.1 * rng.normal(), 0.1 * rng.normal(), -9.8))
        m, h = asm.step(fix, imu, t, 0.05)
        eta, nu, nudot = _reference_step(st, fix, imu, t, 0.05, 0.6)
        assert m is not None and h["vel_src"] == "tag_fd"
        assert np.array_equal(m["eta"], eta), (k, m["eta"] - eta)
        assert np.array_equal(m["nu"], nu), (k, m["nu"] - nu)
        assert np.array_equal(m["nudot"], nudot), (k, m["nudot"] - nudot)
        assert np.array_equal(h["nu_fd"], nu[:3])
        t += 0.05



def test_a_dvl_velocity_replaces_surge_and_sway_and_nothing_else():
    asm = StateAssembler(z_source="tag", vel_lp_alpha=1.0, nudot_source="fd")
    t0 = now()
    yaw = math.pi / 2                        # nose along +y_ned
    asm.step(_fix([0.0, 0.0, 0.5], yaw, t0), _imu(yaw=yaw, t=t0),
             t0 + 0.01, 0.05)
    # the tags say 0.2 m/s surge and 0.1 m/s DOWN
    fix = _fix([0.0, 0.02, 0.51], yaw, t0 + 0.1)
    m, h = asm.step(fix, _imu(yaw=yaw, t=t0 + 0.1), t0 + 0.2, 0.05,
                    vel_xy=(0.30, 0.05))
    assert h["vel_src"] == "dvl"
    assert np.allclose(m["nu"][:2], [0.30, 0.05])
    # ...the HEAVE rate is still the tags': nothing verifies a DVL's z
    assert abs(m["nu"][2] - 0.1) < 1e-9, m["nu"]
    # ...the finite difference kept running underneath, and is reported
    assert abs(h["nu_fd"][0] - 0.2) < 1e-9
    # ...and the 0.1 s fix-age bridge used the DVL velocity, in the WORLD:
    # body (0.30, 0.05) with the nose along +y is world (-0.05, +0.30)
    assert abs(m["eta"][0] - (0.0 - 0.05 * 0.1)) < 1e-9, m["eta"]
    assert abs(m["eta"][1] - (0.02 + 0.30 * 0.1)) < 1e-9, m["eta"]
    assert abs(m["eta"][2] - (0.51 + 0.1 * 0.1)) < 1e-9, m["eta"]
    # position source, yaw and rates are untouched
    assert abs(m["eta"][5] - yaw) < 1e-9


def test_the_fix_age_bridge_is_an_integral_when_one_is_given():
    """`velocity x age` is not an integral once the velocity keeps updating
    under an ageing fix. With `disp_xy` the bridge is p_tag + R @ integral,
    whatever the newest velocity happens to be."""
    asm = StateAssembler(z_source="tag", vel_lp_alpha=1.0, nudot_source="fd")
    t0 = 50.0
    fix = _fix([1.0, 2.0, 0.5], 0.0, t0)
    asm.step(fix, _imu(t=t0), t0, 0.05)
    # 0.3 s after the fix; the vehicle moved 4 cm forward and has since
    # STOPPED — the newest velocity is zero
    m, _h = asm.step(fix, _imu(t=t0 + 0.3), t0 + 0.3, 0.05,
                     vel_xy=(0.0, 0.0), disp_xy=(0.04, -0.01))
    assert abs(m["eta"][0] - 1.04) < 1e-12 and abs(m["eta"][1] - 1.99) < 1e-12
    # without the integral the bridge would have left it at the fix
    m, _h = asm.step(fix, _imu(t=t0 + 0.3), t0 + 0.3, 0.05,
                     vel_xy=(0.0, 0.0))
    assert abs(m["eta"][0] - 1.0) < 1e-12


# ======================================================================== aid
def _proven_aid(**over):
    cfg = dict(MOUNT)
    cfg.update(over)
    d = DV.DvlAid(cfg)
    d.proven = True                           # skip the fit; tested above
    return d


def _vel(d, t):
    """The switch and the DATA gates only — the question the bridge's rung
    asks. A flying tick also needs a current tag velocity to check the
    sample against; that rule has its own test."""
    return d.velocity(t, need_ref=False)


def test_the_gates_each_hold_a_sample_back_and_say_why():
    t = 10.0
    d = _proven_aid(reentry_s=0.0)
    d.use = True
    assert _vel(d, t) == (None, "no DVL data")
    d.note_sample(_sample((0.1, 0.0, 0.0), t=t))
    v, why = _vel(d, t + 0.1)
    assert v is not None and why == "" and abs(v[0] - 0.1) < 1e-12
    v, why = _vel(d, t + 5.0)
    assert v is None and why.startswith("DVL stale (")
    for s, needle in ((_sample(valid=False, t=t), "DVL has no bottom lock"),
                      (_sample(fom=0.5, t=t), "DVL fom too high ("),
                      (_sample(alt=0.02, t=t), "DVL altitude too low ("),
                      (_sample((9.0, 0.0, 0.0), t=t), "DVL velocity implausible"),
                      (_sample(kind="velocity_water", t=t),
                       "DVL is in water-tracking mode"),
                      # FAIL-CLOSED: a lock with no quality figures tells the
                      # gates nothing, so it is not a pass
                      (_sample(fom=None, t=t), "DVL report lacks fom"),
                      (_sample(alt=None, t=t), "DVL report lacks fom")):
        d = _proven_aid(reentry_s=0.0)
        d.use = True
        d.note_sample(s)
        v, why = _vel(d, t)
        assert v is None and why.startswith(needle), (needle, why)
        # the run meta counts fallback rows by the phrase before " ("
        assert not any(ch.isdigit() for ch in why.split(" (")[0]), why


def test_a_spike_is_not_a_manoeuvre():
    """The jump gate: no thruster on this vehicle changes its velocity by
    half a metre per second in a tenth of a second."""
    d = _proven_aid(reentry_s=0.0)
    d.use = True
    t = 10.0
    for k in range(5):
        d.note_sample(_sample((0.05, 0.0, 0.0), t=t + 0.1 * k))
    assert _vel(d, t + 0.4)[0] is not None
    d.note_sample(_sample((0.60, 0.0, 0.0), t=t + 0.5))        # a spike
    v, why = _vel(d, t + 0.5)
    assert v is None and why.startswith("DVL velocity jump ("), why
    d.note_sample(_sample((0.06, 0.0, 0.0), t=t + 0.6))        # back to truth
    assert _vel(d, t + 0.6)[0] is not None
    # a real acceleration inside a_max passes: 0.15 m/s more in 0.1 s
    d.note_sample(_sample((0.21, 0.0, 0.0), t=t + 0.7))
    assert _vel(d, t + 0.7)[0] is not None
    # the lever-arm term is bounded too: a wild body rate cannot push the
    # BODY velocity past v_max while the raw reading looks fine
    d = DV.DvlAid({"mount_rpy_deg": [0, 0, 0], "mount_t_flu": [0.3, 0, 0],
                   "reentry_s": 0.0})
    d.proven, d.use = True, True
    d.note_sample(_sample((0.05, 0.0, 0.0), t=t), omega=(0.0, 0.0, 6.0))
    v, why = _vel(d, t)
    assert v is None and why == "DVL velocity implausible", why


def test_out_means_shadow_and_in_needs_the_mount_and_the_proof():
    d = DV.DvlAid(None)                       # mount unmeasured
    d.note_sample(_sample(t=1.0))
    assert "mount not measured" in d.set_use(True) and not d.use
    assert d.velocity(1.0)[0] is None
    d = DV.DvlAid(dict(MOUNT))
    assert d.set_use(True) == "no DVL data yet"
    d.note_sample(_sample(t=1.0))
    why = d.set_use(True)
    assert "not proven" in why and not d.use
    d.proven = True
    assert d.set_use(True) == "" and d.use
    assert d.set_use(False) == "" and not d.use     # OUT is never refused
    assert d.velocity(1.0) == (None, "DVL is OUT (shadow)")
    # the recorded opt-out waives the PROOF...
    d = DV.DvlAid(dict(MOUNT, require_proven=False))
    d.note_sample(_sample(t=1.0))
    assert d.set_use(True) == "" and d.meta()["require_proven"] is False
    # ...it does not overrule evidence: a mount the data REFUTES stays out
    d = DV.DvlAid(dict(MOUNT, require_proven=False))
    _fly(d, theta_deg=180.0)
    assert d.agree.fit()["verdict"] == "refuted"
    why = d.set_use(True)
    assert "REFUTES" in why and not d.use, why


# LEGACY-PINNED (2026-10-16, SPEC §10.2a): the shipped ladder = the rollback's test.
def test_reentry_hysteresis_keeps_a_flickering_lock_out():
    d = _proven_aid(selection="legacy", reentry_s=0.5)
    d.use = True
    t = 20.0
    d.note_sample(_sample(valid=False, t=t))
    for k in range(1, 5):                     # 0.1 .. 0.4 s of good samples
        d.note_sample(_sample((0.1, 0, 0), t=t + 0.1 * k))
        v, why = _vel(d, t + 0.1 * k)
        assert v is None and why.startswith("DVL re-entry ("), (k, why)
    for k in range(5, 8):
        d.note_sample(_sample((0.1, 0, 0), t=t + 0.1 * k))
    assert _vel(d, t + 0.7)[0] is not None
    d.note_sample(_sample(valid=False, t=t + 0.8))    # one bad report
    d.note_sample(_sample((0.1, 0, 0), t=t + 0.9))
    v, why = _vel(d, t + 0.9)
    assert v is None and why.startswith("DVL re-entry (")


# LEGACY-PINNED (2026-10-16, SPEC §10.2a): the shipped ladder = the rollback's test.
def test_a_silence_restarts_the_hysteresis_like_a_bad_sample_does():
    """review 2026-10-02: only an INVALID sample used to reset the clock, so
    after a link stall the first report back was used at once."""
    d = _proven_aid(selection="legacy", reentry_s=0.5)
    d.use = True
    t = 40.0
    for k in range(10):
        d.note_sample(_sample((0.1, 0, 0), t=t + 0.1 * k))
    assert _vel(d, t + 0.9)[0] is not None
    assert _vel(d, t + 5.0)[1].startswith("DVL stale (")
    d.note_sample(_sample((0.1, 0, 0), t=t + 10.0))       # 9 s of nothing
    v, why = _vel(d, t + 10.0)
    assert v is None and why.startswith("DVL re-entry (0.00"), why
    for k in range(1, 7):
        d.note_sample(_sample((0.1, 0, 0), t=t + 10.0 + 0.1 * k))
    assert _vel(d, t + 10.6)[0] is not None
    # a report OLDER than the one already held cannot become "the newest"
    d.note_sample(_sample((0.9, 0, 0), t=t + 3.0))
    assert d.n_out_of_order == 1 and abs(_vel(d, t + 10.6)[0][0] - 0.1) < 1e-12


def _fly(d, T=12.0, theta_deg=0.0, t0=0.0, imu=None, stamp_late=0.0,
         periods=(8.0, 8.0)):
    """Drive a DvlAid with consistent tag fixes (15 Hz) and DVL samples
    (10 Hz) of the same motion; the device really sits ``theta_deg``
    clockwise of the configured mount. ``stamp_late`` = the fixes carry a
    capture stamp that much AFTER the moment they were true. ``periods`` =
    of the surge and of the sway oscillation, seconds."""
    imu = imu or _imu(t=t0)
    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    w1, w2 = 2 * math.pi / periods[0], 2 * math.pi / periods[1]

    def vel(t):
        return np.array([0.12 * math.sin(w1 * t),
                         0.08 * math.sin(w2 * (t - 3.0))])

    def pos(t):
        return np.array([-0.12 / w1 * math.cos(w1 * t),
                         -0.08 / w2 * math.cos(w2 * (t - 3.0)), 0.5])

    ev = [(k / 15.0, "fix") for k in range(int(T * 15))]
    ev += [(0.03 + k / 10.0, "dvl") for k in range(int(T * 10))]
    for t, kind in sorted(ev):
        if kind == "fix":
            d.note_fix(_fix(list(pos(t)), 0.0, t0 + t + stamp_late), imu)
        else:
            v = vel(t)
            d.note_sample(_sample((c * v[0] + s * v[1],
                                   -s * v[0] + c * v[1], 0.0), t=t0 + t))


def test_the_aid_proves_a_good_mount_from_fixes_and_samples():
    d = DV.DvlAid(dict(MOUNT))
    _fly(d)
    assert d.proven, d.agree.fit()
    assert abs(d.proof["mount_yaw_fix_deg"]) < 2.0
    assert d.set_use(True) == ""
    m = d.meta()
    assert m["proven"] and m["proof"]["verdict"] == "proven"
    assert m["samples"] == 120 and m["samples_valid"] == 120


def test_the_fit_reports_how_far_apart_the_two_sensors_stamps_are():
    """A commissioning number, not a gate: everything pairs the two sensors
    by their stamps, so a stamp error has to be visible. 80 ms here."""
    d = DV.DvlAid(dict(MOUNT))
    _fly(d, T=20.0, periods=(8.0, 5.0))
    f = d.agree.fit()
    assert abs(f["lag_s"]) < 0.01 and f["lag_sigma_s"] < 0.01, f
    for late in (0.08, -0.08):
        d = DV.DvlAid(dict(MOUNT))
        _fly(d, T=20.0, periods=(8.0, 5.0), stamp_late=late)
        f = d.agree.fit()
        assert abs(f["lag_s"] - late) < 0.015, (late, f["lag_s"])
        assert d.proven and d.proof["lag_s"] is not None, d.proof
    # ONE frequency in both axes is the blind case: the velocity vector
    # turns, a lag is then exactly a linear map of it, and the 2x2 fit
    # absorbs it — the lag reads zero and the MOUNT reads a few degrees out
    # (w x lag = 3.6 deg for a circle; this ellipse reads 5.5)
    d = DV.DvlAid(dict(MOUNT))
    _fly(d, T=20.0, stamp_late=0.08)
    f = d.agree.fit()
    assert abs(f["lag_s"]) < 0.01, f["lag_s"]
    assert 2.0 < abs(f["mount_yaw_fix_deg"]) < 8.0, f["mount_yaw_fix_deg"]
    # ...and with no acceleration known there is no number, not a zero
    a = DV.Agreement(4, 0.01, 0.25, 0.01, 60.0)
    for k in range(40):
        v = (0.1 * math.sin(0.3 * k), 0.1 * math.cos(0.3 * k))
        a.add(0.1 * k, v, v)
    assert a.fit()["verdict"] == "proven" and a.fit()["lag_s"] is None


def test_the_aid_never_proves_a_mount_that_is_90_degrees_out():
    d = DV.DvlAid(dict(MOUNT))
    _fly(d, theta_deg=90.0)
    assert not d.proven
    f = d.agree.fit()
    assert f["verdict"] == "refuted" and abs(f["mount_yaw_fix_deg"] - 90) < 3
    why = d.set_use(True)
    assert "REFUTES" in why and "+90 deg" in why and not d.use, why
    # in shadow, never proven: there is nothing to trip
    assert d.n_trips == 0 and not d.tripped
    # ...and telling the config about it makes the same data agree
    d = DV.DvlAid({"mount_rpy_deg": [0, 0, 90], "mount_t_flu": [0, 0, 0]})
    _fly(d, theta_deg=90.0)
    assert d.proven, d.agree.fit()


class _Pairs:
    """Feeds a DvlAid the two streams the worker feeds it — DVL samples and
    tag FIXES, each stamped with the moment it was true — and lets the aid
    form its own time-aligned pairs, which is all its runtime monitors see.

    The tag position is the integral of ``v_tag`` at heading 0 (so body =
    world); ``noise_mm`` is the tag position noise. ``dvl_late`` /
    ``tag_late`` delay one stream's ARRIVAL without touching its stamps."""

    def __init__(self, d, t=30.0, seed=2):
        self.d, self.t = d, float(t)
        self.p = np.array([0.0, 0.0, 0.5])
        self.rng = np.random.default_rng(seed)
        self.imu = _imu(t=t)
        self._late: list = []                 # (arrival, order, kind, obj)

    def run(self, n, v_dvl, v_tag, noise_mm=0.0, fixes=True, dt=0.05,
            dvl_late=0.0, tag_late=0.0, stamp_late=0.0):
        """``n`` steps of ``dt``; returns the time. Stops at a NEW trip.
        ``stamp_late`` = the fixes' capture stamp is WRONG by that much."""
        trips = self.d.n_trips
        for _ in range(n):
            t0, self.t = self.t, self.t + dt
            tm = 0.5 * (t0 + self.t)
            vd = v_dvl(self.t) if callable(v_dvl) else v_dvl
            vt = v_tag(tm) if callable(v_tag) else v_tag
            self.p = self.p + np.asarray(vt, float) * dt
            self._late.append((self.t + dvl_late, len(self._late), "dvl",
                               _sample(vd, t=self.t)))
            if fixes:
                pn = self.p + self.rng.normal(0, noise_mm * 1e-3, 3)
                self._late.append((self.t + tag_late, len(self._late), "fix",
                                   _fix(list(pn), 0.0, self.t + stamp_late)))
            self._late.sort(key=lambda e: e[:2])
            while self._late and self._late[0][0] <= self.t + 1e-9:
                _, _, kind, obj = self._late.pop(0)
                if kind == "dvl":
                    self.d.note_sample(obj)
                else:
                    self.d.note_fix(obj, self.imu)
            if self.d.n_trips != trips:
                break
        return self.t

    def jump(self, dx, dy=0.0):
        """The tag solution steps and stays (a re-acquired, mis-solved fix)."""
        self.p = self.p + np.array([dx, dy, 0.0])

    def flush(self):
        """Deliver whatever is still on its way."""
        for _, _, kind, obj in self._late:
            if kind == "dvl":
                self.d.note_sample(obj)
            else:
                self.d.note_fix(obj, self.imu)
        self._late = []


# LEGACY-PINNED (2026-10-16, SPEC §10.2a): the shipped ladder = the rollback's test.
def test_a_sustained_disagreement_trips_and_the_trip_is_latched():
    d = _proven_aid(selection="legacy", reentry_s=0.0)
    d.use = True
    f = _Pairs(d)
    t = f.run(200, (0.10, 0, 0), (0.10, 0, 0), noise_mm=1.0)
    assert d.proven and d.n_trips == 0 and not d.tripped, d.residual_m_s
    assert d.n_monitor > 150 and d.residual_m_s < 0.02, d.residual_m_s
    t1 = f.run(60, (-0.10, 0, 0), (0.10, 0, 0))     # reads the reverse
    assert d.tripped and d.n_trips == 1 and not d.proven
    # 0.2 m/s apart is GROSS: each pair counts as innov_max (0.15) apart,
    # so the 1 s low-pass crosses 8 cm/s after ~0.8 s — not after one pair
    assert "sustained" in d.tripped and d.n_gross > 10, d.tripped
    assert "false bottom lock" not in d.tripped      # the tags moved too
    assert 0.5 < t1 - t < 1.5, t1 - t
    v, why = _vel(d, t1)
    assert v is None and why.startswith("DVL tripped ("), why
    # LATCHED: perfectly agreeing data afterwards does not put it back...
    t2 = f.run(100, (0.10, 0, 0), (0.10, 0, 0))
    assert d.tripped and _vel(d, t2)[0] is None
    assert d.state(t2, used=False)[0] == DV.ST_TRIPPED
    # ...only the operator does, and only through the proof gate
    why = d.set_use(True)
    assert "has to be PROVEN" in why and "sustained" in why and d.tripped, why
    d.proven = True
    assert d.set_use(True) == "" and not d.tripped
    assert _vel(d, t2)[0] is not None


# LEGACY-PINNED (2026-10-16, SPEC §10.2a): the shipped ladder = the rollback's test.
def test_a_standing_offset_trips_on_the_low_passed_difference():
    """Below the gross gate, above trip_m_s: a mount 60 degrees out at
    10 cm/s is 10 cm/s of difference that never goes away."""
    d = _proven_aid(selection="legacy", reentry_s=0.0, agree_ratio_v_min_m_s=50.0)   # alignment off
    d.use = True
    f = _Pairs(d)
    c, s = math.cos(math.radians(60)), math.sin(math.radians(60))
    t0 = f.t
    t = f.run(100, (0.10 * c, -0.10 * s, 0), (0.10, 0, 0), noise_mm=0.5)
    assert d.tripped and "sustained" in d.tripped, d.tripped
    assert 1.0 < t - t0 < 3.0, t - t0


# LEGACY-PINNED (2026-10-16, SPEC §10.2a): the shipped ladder = the rollback's test.
def test_the_opt_out_waives_the_proof_not_the_trips():
    """review 2026-10-02, both reviewers: with require_proven false the
    runtime monitor used to be gated on `proven` and so never fired — a
    mirrored mount flew on wrong-signed feedback until E-STOP."""
    d = DV.DvlAid(dict(MOUNT, selection="legacy", require_proven=False, reentry_s=0.0))
    d.note_sample(_sample(t=29.9))
    assert d.set_use(True) == "" and not d.proven
    f = _Pairs(d)
    t = f.run(60, (0.10, -0.06, 0), (0.10, 0.06, 0), noise_mm=0.5)  # y mirrored
    assert d.tripped and d.n_trips == 1, (d.residual_m_s, d.alignment())
    assert t - 30.0 < 1.5, t
    assert _vel(d, t)[0] is None


# LEGACY-PINNED (2026-10-16, SPEC §10.2a): the shipped ladder = the rollback's test.
def test_an_oscillating_wrong_sign_trips_on_alignment_not_on_the_average():
    """A station hold on wrong-signed velocity feedback does not drift, it
    OSCILLATES — and the low-passed difference of an oscillation averages
    toward zero. The alignment test is what catches it."""
    w = 4.0                                   # rad/s, the loop's own frequency
    amp = 0.06

    def tag(t):
        return (amp * math.sin(w * t), 0.0, 0.0)

    def flipped(t):
        return (-amp * math.sin(w * t), 0.0, 0.0)

    d = _proven_aid(selection="legacy", reentry_s=0.0, agree_trip_m_s=10.0)   # vector monitor off
    d.use = True
    f = _Pairs(d)
    t = f.run(200, flipped, tag, noise_mm=0.5)
    assert d.tripped and "alignment" in d.tripped, d.tripped
    assert "bearing +0 deg" in d.tripped or "bearing -0 deg" in d.tripped, \
        d.tripped
    assert t - 30.0 < 1.5, t
    # with the alignment test disabled the same data sails through the
    # vector monitor at its shipped threshold
    d = _proven_aid(selection="legacy", reentry_s=0.0, agree_ratio_v_min_m_s=50.0)
    d.use = True
    _Pairs(d).run(200, flipped, tag, noise_mm=0.5)
    assert not d.tripped, d.tripped
    # ...and a RIGHT mount oscillating the same way trips neither
    d = _proven_aid(selection="legacy", reentry_s=0.0)
    d.use = True
    _Pairs(d).run(200, tag, tag, noise_mm=1.0)
    assert not d.tripped, d.tripped


# LEGACY-PINNED (2026-10-16, SPEC §10.2a), with a twin under primary below.
def test_the_alignment_numbers_name_the_direction_and_the_error(selection="legacy"):
    """alignment() is 1 - |A e - e| along each direction the vehicle moves
    in: +1 the same motion, -1 reversed, 1 - sqrt(2) a quarter turn. In
    shadow with no proof there is nothing to trip, so the numbers can be
    read off every case."""
    def tag(t):                # surge the larger, sway the less; one lap of
        #                        the ellipse fits the test's 1 s memory
        return (0.08 * math.sin(7.0 * t), 0.05 * math.cos(7.0 * t), 0.0)

    d = DV.DvlAid(dict(MOUNT, selection=selection))
    assert d.alignment() == []                # no pairs yet
    _Pairs(d).run(200, tag, tag, noise_mm=0.5)
    al = d.alignment()
    assert len(al) == 2 and not d.tripped, al
    assert abs(al[0]["along_deg"]) < 20, al
    assert abs(abs(al[1]["along_deg"]) - 90) < 20, al
    assert all(a["align"] > 0.85 and a["sigma"] < 0.15 for a in al), al
    assert al[0]["speed_m_s"] > al[1]["speed_m_s"] > 0.02 and al[0]["n_eff"] > 20

    d = DV.DvlAid(dict(MOUNT, selection=selection))                # sway mirrored
    _Pairs(d).run(200, lambda t: (tag(t)[0], -tag(t)[1], 0.0), tag,
                  noise_mm=0.5)
    al = d.alignment()
    assert al[0]["align"] > 0.85 and al[1]["align"] < -0.85, al
    assert d.n_trips == 0                     # shadow, unproven: nothing to trip

    d = DV.DvlAid(dict(MOUNT, selection=selection))                # a quarter turn
    _Pairs(d).run(200, lambda t: (tag(t)[1], -tag(t)[0], 0.0), tag,
                  noise_mm=0.5)
    al = d.alignment()
    assert all(abs(a["align"] - (1 - math.sqrt(2))) < 0.15 for a in al), al

    d = DV.DvlAid(dict(MOUNT, selection=selection))                # a station hold: no evidence
    _Pairs(d).run(200, lambda t: (0.008 * math.sin(t), 0.0, 0.0),
                  lambda t: (0.008 * math.sin(t), 0.0, 0.0), noise_mm=0.5)
    assert d.alignment() == [] and d.n_monitor > 150
    m = d.meta()["monitor_now"]
    assert m["alignment"] == [] and m["residual_m_s"] < 0.02
    json.dumps(d.meta())


def test_the_alignment_numbers_name_the_direction_and_the_error_under_primary():
    """The same, under dvl.selection primary (2026-10-16): mode-independent in substance."""
    test_the_alignment_numbers_name_the_direction_and_the_error(selection="primary")


# LEGACY-PINNED (2026-10-16, SPEC §10.2a), with a twin under primary below.
def test_a_mount_inside_the_tolerance_cannot_trip_whichever_way_it_moves(selection="legacy"):
    """A mount the proof accepts (12 degrees out is inside agree_tol) leaks
    surge into the DVL's sway. A per-AXIS slope reads that leak as "sway
    disagrees" on a diagonal pass (design note 2026-10-02); along the
    direction of motion a rotation by D is only 2 sin(D/2) = 0.21."""
    for deg in (12.0, -12.0):
        c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))

        def tag(t):                           # a diagonal line, back and forth
            a = math.sin(1.5 * t)
            return (0.10 * a, 0.02 * a, 0.0)

        def dvl(t):
            v = tag(t)
            return (c * v[0] + s * v[1], -s * v[0] + c * v[1], 0.0)

        d = _proven_aid(selection=selection, reentry_s=0.0)
        d.use = True
        _Pairs(d).run(600, dvl, tag, noise_mm=0.8)
        assert not d.tripped, (deg, d.tripped)
        al = d.alignment()
        assert al and abs((1 - al[0]["align"]) - 0.21) < 0.12, al


def test_a_mount_inside_the_tolerance_cannot_trip_whichever_way_it_moves_under_primary():
    """The same, under dvl.selection primary (2026-10-16): mode-independent in substance."""
    test_a_mount_inside_the_tolerance_cannot_trip_whichever_way_it_moves(selection="primary")


# LEGACY-PINNED (2026-10-16, SPEC §10.2a): the shipped ladder = the rollback's test.
def test_the_first_pairs_of_a_new_motion_are_not_a_verdict():
    """When the vehicle starts moving, the first pair or two are that
    direction's whole evidence. With stamps 100 ms apart the DVL has stepped
    in them and the tag difference has not — a slope of zero on no residual
    — until the next few pairs outweigh them. The verdict has to STAND for
    ALIGN_HOLD_S (simulation 2026-10-02: a right mount tripped at the
    corners of a 10 cm/s square)."""
    def fly(d):
        f = _Pairs(d)
        f.run(40, (0.0, 0, 0), (0.0, 0, 0), noise_mm=0.2, stamp_late=0.10)
        for k in range(6):                    # six starts along +-x, +-y
            v = [(0.10, 0, 0), (0, 0.10, 0), (-0.10, 0, 0), (0, -0.10, 0),
                 (0.10, 0, 0), (0, 0, 0)][k]
            f.run(40, v, v, noise_mm=0.2, stamp_late=0.10)

    d = _proven_aid(selection="legacy", reentry_s=0.0)
    d.use = True
    fly(d)
    assert not d.tripped, d.tripped
    # ...and it is the hold (with the share rule, which keeps the barely
    # moving direction of a fresh motion out of the verdict) that does it
    hold, share = DV.ALIGN_HOLD_S, DV.ALIGN_MIN_SHARE
    DV.ALIGN_HOLD_S, DV.ALIGN_MIN_SHARE = 0.0, 0.0
    try:
        d = _proven_aid(selection="legacy", reentry_s=0.0)
        d.use = True
        fly(d)
        assert d.tripped and "alignment" in d.tripped, d.tripped
    finally:
        DV.ALIGN_HOLD_S, DV.ALIGN_MIN_SHARE = hold, share


# LEGACY-PINNED (2026-10-16, SPEC §10.2a), with a twin under primary below.
def test_a_tag_jump_is_set_aside_not_booked_against_the_dvl(selection="legacy"):
    """One mis-solved fix is one pair of metres per second in the raw tag
    difference. It counts as innov_max apart in the difference monitor and
    is left out of the alignment sums — it is the tag's error — so it takes
    most of a second of such pairs on end to trip."""
    d = _proven_aid(selection=selection, reentry_s=0.0)
    d.use = True
    f = _Pairs(d)
    f.run(100, (0.05, 0, 0), (0.05, 0, 0), noise_mm=0.5)
    res = d.residual_m_s
    for k in range(6):                        # six 4 cm steps, 0.5 s apart
        f.jump(0.04 * (-1) ** k, 0.03)
        f.run(10, (0.05, 0, 0), (0.05, 0, 0), noise_mm=0.5)
    assert not d.tripped and d.proven, d.tripped
    assert d.n_gross == 6, d.n_gross
    assert abs(d.residual_m_s - res) < 0.02, (res, d.residual_m_s)


def test_a_tag_jump_is_set_aside_not_booked_against_the_dvl_under_primary():
    """The same, under dvl.selection primary (2026-10-16): mode-independent in substance."""
    test_a_tag_jump_is_set_aside_not_booked_against_the_dvl(selection="primary")


# LEGACY-PINNED (2026-10-16, SPEC §10.2a), with a twin under primary below.
def test_how_late_either_stream_arrives_changes_nothing(selection="legacy"):
    """The monitors compare what the two sensors say about the SAME moment
    (a pair is the tag difference against the DVL interpolated to its
    midpoint). Fed by the tick's own tag velocity instead, a right mount
    tripped whenever one stream was ~0.15 s later than the other and the
    vehicle oscillated at 0.5 Hz or faster (simulation 2026-10-02)."""
    def v(t):                                 # 1 Hz, 6 cm/s
        return (0.06 * math.sin(2 * math.pi * t),
                0.04 * math.sin(2 * math.pi * t + 1.0), 0.0)

    ref = None
    for late in ({}, {"dvl_late": 0.30}, {"tag_late": 0.25},
                 {"dvl_late": 0.05, "tag_late": 0.30}):
        d = _proven_aid(selection=selection, reentry_s=0.0)
        d.use = True
        f = _Pairs(d)
        f.run(400, v, v, noise_mm=0.8, **late)
        f.flush()
        assert not d.tripped, (late, d.tripped)
        al = d.alignment()
        assert al and al[0]["align"] > 0.8 and d.n_monitor > 350, (late, al)
        # the same pairs whatever the arrival order, so the same numbers
        got = (round(d.residual_m_s, 9), round(al[0]["align"], 9))
        ref = ref or got
        assert got == ref, (late, got, ref)


def test_how_late_either_stream_arrives_changes_nothing_under_primary():
    """The same, under dvl.selection primary (2026-10-16): mode-independent in substance."""
    test_how_late_either_stream_arrives_changes_nothing(selection="primary")


# LEGACY-PINNED (2026-10-16, SPEC §10.2a), with a twin under primary below.
def test_without_new_fixes_there_is_no_evidence_to_trip_on(selection="legacy"):
    """After the LAST fix the tag side says nothing. Comparing the DVL
    against a tag velocity frozen at its last value books the vehicle's own
    acceleration as disagreement (review 2026-10-02: a kick that took the
    tags out of view un-proved the mount right before the dropout the DVL
    rung exists for)."""
    d = _proven_aid(selection=selection, reentry_s=0.0)
    d.use = True
    f = _Pairs(d)
    t = f.run(40, (0.0, 0, 0), (0.0, 0, 0))
    n = d.n_monitor
    assert n > 30
    # the vehicle is kicked to 0.3 m/s; the tags stopped updating
    t_kick = t
    f.run(40, lambda t: (min(0.3, 1.5 * (t - t_kick)), 0, 0), (0.0, 0, 0),
          fixes=False)
    assert d.n_monitor == n and not d.tripped and d.proven
    assert _vel(d, f.t)[0] is not None


def test_without_new_fixes_there_is_no_evidence_to_trip_on_under_primary():
    """The same, under dvl.selection primary (2026-10-16): mode-independent in substance."""
    test_without_new_fixes_there_is_no_evidence_to_trip_on(selection="primary")


# LEGACY-PINNED (2026-10-16, SPEC §10.2a): the shipped ladder = the rollback's test.
def test_an_outlier_tick_is_held_back_without_a_trip():
    """The innovation gate: with a current tag velocity to compare against,
    one sample far from it is simply not used that tick."""
    d = _proven_aid(selection="legacy", reentry_s=0.0, a_max_m_s2=50.0)       # jump gate aside
    d.use = True
    t = 30.0
    for _ in range(20):
        t += 0.05
        d.note_sample(_sample((0.05, 0, 0), t=t))
        d.note_tag_velocity(np.array([0.05, 0.0, 0.0]), t, tag_age=0.13)
    d.note_sample(_sample((0.40, 0.0, 0.0), t=t + 0.05))  # a false lock
    v, why = d.velocity(t + 0.05)
    assert v is None and why.startswith("DVL innovation ("), why
    assert not d.tripped and d.proven and d.n_innovation == 1
    d.note_sample(_sample((0.05, 0.0, 0.0), t=t + 0.10))
    d.note_tag_velocity(np.array([0.05, 0.0, 0.0]), t + 0.10, tag_age=0.13)
    assert d.velocity(t + 0.10)[0] is not None
    # an old tag velocity is a weak reference: 0.6 s stale widens the gate
    # by a_max x (age - 0.25 s)
    d.note_tag_velocity(np.array([0.05, 0.0, 0.0]), t + 0.15, tag_age=0.60)
    d.note_sample(_sample((0.40, 0.0, 0.0), t=t + 0.15))
    assert d.velocity(t + 0.15)[0] is not None
    # A tick or two with no tag state keeps the last reference (it ages out
    # by itself): the gate still stands, it does not flap to "unchecked"
    # (review 2026-10-02: clearing it there made the source switch once per
    # fix cycle whenever the fix age touched tag_stale_s).
    n = d.n_innovation
    d.note_tag_velocity(np.array([0.05, 0.0, 0.0]), t + 0.20, tag_age=0.13)
    d.note_tag_velocity(None, t + 0.25)
    d.note_sample(_sample((0.40, 0.0, 0.0), t=t + 0.30))
    assert d.velocity(t + 0.30)[1].startswith("DVL innovation (")
    assert d.n_innovation == n + 1
    # a display-only question is not another rejected tick
    assert d.state(t + 0.30, used=False)[0] == DV.ST_FALLBACK
    assert d.n_innovation == n + 1
    # ...but with NO tag velocity for longer than that (a real tag gap) a
    # flying tick does not take a sample nothing checked: the first tick
    # after the gap would otherwise use whatever the DVL had locked onto
    # during it. Only the bridge's rung, which has no tag by definition,
    # may — and within FD_REF_MAX_S of the last tag velocity even the rung
    # is still held to it.
    d.note_sample(_sample((0.40, 0.0, 0.0), t=t + 0.35))
    assert _vel(d, t + 0.35)[1].startswith("DVL innovation (")
    d.note_sample(_sample((0.40, 0.0, 0.0), t=t + 0.50))
    assert d.velocity(t + 0.50) == (
        None, "DVL unchecked (no current tag velocity)")
    # (and what the gate rejected above stays out of the rung as well —
    # the innovation hold, until a newer frame agrees)
    assert _vel(d, t + 0.50)[1].startswith("DVL innovation hold (")
    d.note_tag_velocity(np.array([0.05, 0.0, 0.0]), t + 0.55, tag_age=0.13)
    d.note_sample(_sample((0.05, 0.0, 0.0), t=t + 0.60))
    assert d.velocity(t + 0.60)[0] is not None
    # the tick's tag velocity is the gate's reference and nothing more: it
    # feeds no monitor, so a wildly different one cannot trip
    for k in range(60):
        d.note_tag_velocity(np.array([-0.5, 0.3, 0.0]), t + 0.25 + 0.05 * k,
                            tag_age=0.13)
    assert not d.tripped and d.n_monitor == 0


def test_the_device_changing_its_own_rotation_is_a_trip():
    d = _proven_aid(reentry_s=0.0)
    d.use = True
    d.note_sample(_sample((0.05, 0, 0), t=5.0))
    assert d.note_device_config({"mounting_rotation_offset": 0,
                                 "speed_of_sound": 1475}) == ""
    assert d.note_device_config({"mounting_rotation_offset": 0.0,
                                 "speed_of_sound": 1475.0,
                                 "dark_mode_enabled": True}) == ""
    assert not d.tripped and _vel(d, 5.0)[0] is not None
    what = d.note_device_config({"mounting_rotation_offset": 90,
                                 "speed_of_sound": 1475})
    assert "mounting_rotation_offset" in what
    assert d.tripped and "configuration changed" in d.tripped
    assert _vel(d, 5.0)[0] is None
    assert d.meta()["device_config_changes"] == [what]


# LEGACY-PINNED (2026-10-16, SPEC §10.2a): the shipped ladder = the rollback's test.
def test_a_latched_trip_does_not_shield_a_proof_made_under_it():
    """review 2026-10-02 (reproduced): every trip was guarded by "not
    already tripped". So a proof RE-MADE while the latch was still set could
    never be undone — it survived a change of the device's own rotation,
    and the operator could re-arm on it."""
    d = DV.DvlAid(dict(MOUNT, selection="legacy", reentry_s=0.0))
    _fly(d)
    assert d.proven and d.set_use(True) == ""
    assert d.note_device_config({"mounting_rotation_offset": 0,
                                 "speed_of_sound": 1475}) == ""
    f = _Pairs(d)
    f.run(60, (-0.10, 0, 0), (0.10, 0, 0))    # reads the reverse: a trip
    assert d.tripped and not d.proven and d.n_trips == 1
    _fly(d, t0=40.0)                          # honest again: RE-proven,
    assert d.proven and d.tripped             # ...and still latched
    what = d.note_device_config({"mounting_rotation_offset": 90,
                                 "speed_of_sound": 1475})
    assert what and not d.proven and d.proof is None and d.agree.n == 0
    assert "configuration changed" in d.tripped and d.n_trips == 2
    why = d.set_use(True)
    assert "has to be PROVEN" in why and d.tripped, why
    # the monitors can undo a proof made under a latch as well
    _fly(d, t0=60.0)
    assert d.proven and d.tripped
    f = _Pairs(d, t=80.0)
    f.run(60, (-0.10, 0, 0), (0.10, 0, 0))
    assert not d.proven and d.n_trips == 3 and "sustained" in d.tripped
    # ...and while latched with NO proof there is nothing more to take out:
    # the same wrong data does not trip again and again
    f.run(100, (-0.10, 0, 0), (0.10, 0, 0))
    assert d.n_trips == 3


# LEGACY-PINNED (2026-10-16, SPEC §10.2a): the shipped ladder = the rollback's test.
def test_evidence_against_the_dvl_outlives_its_window_and_its_trip():
    """require_proven false waives the proof a DVL with nothing against it
    would need — not the one evidence demands (review 2026-10-02: under the
    opt-out a refutation seen in shadow expired with the 60 s window, and a
    tripped DVL could be re-armed a second later, the trip having deleted
    the very pairs that caused it)."""
    # (a) refuted in shadow: remembered after the window has forgotten it
    d = DV.DvlAid(dict(MOUNT, selection="legacy", require_proven=False, agree_window_s=8.0))
    _fly(d, theta_deg=180.0)                  # switch OUT, mount reversed
    assert d.agree.fit()["verdict"] == "refuted"
    assert not d.tripped and d.n_trips == 0   # nothing to take out...
    assert "disagree" in d.against            # ...but it is remembered
    f = _Pairs(d)
    f.run(300, (0.0, 0, 0), (0.0, 0, 0))      # 15 s at rest: the window empties
    assert d.agree.fit()["verdict"] == "insufficient"
    why = d.set_use(True)
    assert "has to be PROVEN" in why and not d.use, why
    assert d.meta()["against"] and "shown wrong earlier" in d.state(
        f.t, used=False)[1]
    _fly(d, t0=60.0)                          # a proof made AFTERWARDS
    assert d.proven and d.against == "" and d.set_use(True) == ""
    # (b) a trip: the re-arm needs a proof made after it, opt-out or not
    d = DV.DvlAid(dict(MOUNT, selection="legacy", require_proven=False, reentry_s=0.0))
    d.note_sample(_sample(t=29.9))
    assert d.set_use(True) == ""              # the opt-out, nothing against
    f = _Pairs(d)
    f.run(60, (-0.10, 0, 0), (0.10, 0, 0))
    assert d.tripped and d.agree.n == 0       # the trip cleared its pairs
    why = d.set_use(True)                     # one second later, on nothing
    assert "has to be PROVEN" in why and d.tripped, why
    _fly(d, t0=50.0)
    assert d.proven and d.set_use(True) == "" and not d.tripped
    assert _vel(d, 50.0 + 11.9)[0] is not None


def test_a_device_config_change_discards_the_evidence_in_every_state():
    """In shadow and not yet proven there is nothing to trip — but the pairs
    so far describe a stream that no longer exists (review 2026-10-02,
    reproduced: 59 old pairs + 1 new one completed a PROOF of a mount the
    device had just been rotated 20 degrees away from)."""
    d = DV.DvlAid(dict(MOUNT))
    d.note_device_config({"mounting_rotation_offset": 0,
                          "speed_of_sound": 1475})
    _fly(d, T=3.5)
    assert 30 < d.agree.n < 60 and not d.proven and d.n_monitor > 30
    t_change = max(h[0] for h in d._hist)
    what = d.note_device_config({"mounting_rotation_offset": 20,
                                 "speed_of_sound": 1475})
    assert what and not d.tripped and d.n_trips == 0
    assert d.agree.n == 0 and d.alignment() == [] and not d._fd_pending
    assert d.residual_m_s is None
    # nothing pairs ACROSS the change either: the first pair of the new
    # stream lies wholly after it
    _fly(d, T=1.0, t0=3.5)
    assert 0 < d.agree.n < 16
    assert min(q[0] for q in d.agree._pairs) > t_change + 0.05


# LEGACY-PINNED (2026-10-16, SPEC §10.2a): the shipped ladder = the rollback's test.
def test_dvl_noise_or_a_glitch_is_not_evidence_about_the_mount():
    """A trip is about the MOUNT: both sensors see the motion and disagree
    about it. When only the DVL "moves" — its own noise, two reports off a
    wall — the tags saw nothing, and that is what the fom gate is for
    (review 2026-10-02, reproduced: a right, proven mount was latched out
    by two reports 12 cm/s off, and by 3 cm/s of DVL noise at rest)."""
    def glitch(d):
        f = _Pairs(d)
        f.run(60, (0.0, 0, 0), (0.0, 0, 0), noise_mm=0.5)
        f.run(2, (0.0, 0.12, 0), (0.0, 0, 0), noise_mm=0.5)
        f.run(60, (0.0, 0, 0), (0.0, 0, 0), noise_mm=0.5)

    def noisy(d):
        rng = np.random.default_rng(7)
        _Pairs(d).run(300, lambda t: tuple(rng.normal(0, 0.04, 3)),
                      (0.0, 0, 0), noise_mm=0.5, dt=0.1)

    for feed in (glitch, noisy):
        d = _proven_aid(selection="legacy", reentry_s=0.0)
        d.use = True
        feed(d)
        assert not d.tripped and d.proven, (feed.__name__, d.tripped)
    # the noisy one: the numbers say what it is
    al = d.alignment()
    assert al and all(a["gain"] < 0.3 and a["align"] < 0.4 for a in al), al
    ft = d.agree.fit()
    assert ft["verdict"] == "refuted" and ft["coherent"] is False, ft
    assert ft["gain_min"] < 0.2 and "nothing about the mount" in ft["why"]
    # ...which still keeps a switch from going IN on that data — but is
    # not REMEMBERED against the DVL once it has passed (shadow and IN must
    # agree on what counts as evidence)
    d2 = DV.DvlAid(dict(MOUNT, selection="legacy", require_proven=False, agree_window_s=8.0))
    noisy(d2)
    assert "nothing about the mount" in d2.set_use(True)
    assert d2.against == "" and not d2.tripped
    f2 = _Pairs(d2, t=70.0)
    f2.run(200, (0.0, 0, 0), (0.0, 0, 0), noise_mm=0.5)   # quiet again
    assert d2.agree.fit()["verdict"] == "insufficient"
    assert d2.set_use(True) == ""
    # and it IS the coherence rule that holds the trip back
    hold, DV.COHERENT_MIN = DV.COHERENT_MIN, 0.0
    try:
        for feed in (glitch, noisy):
            d = _proven_aid(selection="legacy", reentry_s=0.0)
            d.use = True
            feed(d)
            assert d.tripped, feed.__name__
    finally:
        DV.COHERENT_MIN = hold


# LEGACY-PINNED (2026-10-16, SPEC §10.2a): the shipped ladder = the rollback's test.
def test_a_dvl_the_fit_currently_refutes_is_held_back_without_a_latch():
    """A refutation that is not about the mount (the DVL "moves", the tags
    do not) does not trip — but the DVL is not flown on while it stands
    (review 2026-10-02: it was, for the rest of the engagement, on data
    refusal() would have refused IN for). It comes back by itself."""
    d = _proven_aid(selection="legacy", reentry_s=0.0, agree_window_s=20.0)
    d.use = True
    rng = np.random.default_rng(3)
    f = _Pairs(d)
    f.run(300, lambda t: tuple(rng.normal(0, 0.04, 3)), (0.0, 0, 0),
          noise_mm=0.5, dt=0.1)               # 30 s of a noisy DVL at rest
    ft = d.agree.fit()
    assert ft["verdict"] == "refuted" and ft["coherent"] is False, ft
    assert not d.tripped and d.proven and d.against == ""
    assert _vel(d, f.t) == (None, "DVL held back (the fit refutes it)")
    assert d.state(f.t, used=False)[0] == DV.ST_FALLBACK
    f.run(250, (0.0, 0, 0), (0.0, 0, 0), noise_mm=0.5, dt=0.1)   # quiet
    assert d.agree.fit()["verdict"] != "refuted"
    v, why = _vel(d, f.t)
    assert v is not None and why == "", why   # back, no operator involved
    assert d.n_trips == 0


# LEGACY-PINNED (2026-10-16, SPEC §10.2a): the shipped ladder = the rollback's test.
def test_a_dvl_that_stands_apart_for_a_second_is_a_trip_whatever_the_tags_saw():
    """The other half of the rule: a two-report glitch is not a trip, a
    false lock that lasts IS — a DVL that reports for a second a motion
    that is not there is not fit to fly on. The message says which of the
    two it looks like, because the operator's remedy differs."""
    d = _proven_aid(selection="legacy", reentry_s=0.0)
    d.use = True
    f = _Pairs(d)
    f.run(60, (0.0, 0, 0), (0.0, 0, 0), noise_mm=0.5)
    t0 = f.t
    f.run(60, (0.0, 0.12, 0), (0.0, 0, 0), noise_mm=0.5)   # at rest
    assert d.tripped and "sustained" in d.tripped, d.tripped
    assert "false bottom lock rather than the mount" in d.tripped
    assert 0.8 < f.t - t0 < 2.0, f.t - t0
    # shorter than that it is not (the fixture's table A5 has the boundary)
    d = _proven_aid(selection="legacy", reentry_s=0.0)
    d.use = True
    f = _Pairs(d)
    f.run(60, (0.0, 0, 0), (0.0, 0, 0), noise_mm=0.5)
    f.run(8, (0.0, 0.12, 0), (0.0, 0, 0), noise_mm=0.5)    # 0.4 s
    f.run(60, (0.0, 0, 0), (0.0, 0, 0), noise_mm=0.5)
    assert not d.tripped, d.tripped


# LEGACY-PINNED (2026-10-16, SPEC §10.2a): the shipped ladder = the rollback's test.
def test_a_rejected_dvl_stays_out_until_a_newer_frame_agrees():
    """The innovation gate's limit widens with the age of the fix — the
    vehicle may have accelerated since. But the same camera frame growing
    older must not rehabilitate a sample it contradicted (review 2026-10-02,
    reproduced: as the tags faded the limit grew past a false 0.25 m/s the
    gate had been rejecting, the DVL went back in, and the station bridge
    carried it 0.6 m). Rejected, the DVL stays out — for the bridge's rung
    too — until a tag velocity from a NEWER frame agrees."""
    def run(lat):
        d = _proven_aid(selection="legacy", reentry_s=0.0)
        d.use = True
        t = 30.0
        for _ in range(10):                   # agreeing tags, `lat` late
            t += 0.05
            d.note_sample(_sample((0.0, 0, 0), t=t))
            d.note_tag_velocity(np.zeros(3), t, tag_age=lat)
            assert d.velocity(t)[0] is not None
        for v in (0.1, 0.2, 0.25, 0.25):      # a false lock; tags in view
            t += 0.05
            d.note_sample(_sample((v, 0, 0), t=t))
            _v, why = d.velocity(t)
            d.note_tag_velocity(np.zeros(3), t, tag_age=lat)
        assert why.startswith("DVL innovation ("), (lat, why)
        frame = t - lat                       # the newest camera frame
        # the fixes stop: that frame freezes and ages, the limit widens past
        # 0.25 m/s — and the DVL stays out, on the tick and on the rung
        for _ in range(8):
            t += 0.05
            d.note_sample(_sample((0.25, 0, 0), t=t))
            v, why = d.velocity(t)
            assert v is None, (lat, why)
            d.note_tag_velocity(np.zeros(3), t, tag_age=t - frame)
        assert d._innov_limit(t - frame) > 0.25
        assert why.startswith("DVL innovation hold ("), (lat, why)
        assert _vel(d, t + 0.05)[0] is None   # the rung's question
        assert d.state(t, used=False) == (DV.ST_FALLBACK, d.velocity(
            t, count=False)[1])
        assert d.meta()["innovation_hold"].startswith("DVL innovation (")
        assert not d.tripped                  # held, not latched
        # the SAME frame agreeing (the vehicle sped up meanwhile) does not
        # release it either...
        d.note_tag_velocity(np.array([0.25, 0.0, 0.0]), t, tag_age=t - frame)
        t += 0.05
        d.note_sample(_sample((0.25, 0, 0), t=t))
        assert d.velocity(t)[1].startswith("DVL innovation hold (")
        # ...a NEWER frame does, late or not (round 5: keyed on "fresh", a
        # hold set by one glitch never released once the tags ran 0.25 s
        # late, and with tags that late it was never set at all)
        d.note_tag_velocity(np.array([0.25, 0.0, 0.0]), t, tag_age=lat)
        t += 0.05
        d.note_sample(_sample((0.25, 0, 0), t=t))
        v, why = d.velocity(t)
        assert v is not None and why == "", (lat, why)
        assert d.meta()["innovation_hold"] is None
        # a display-only question changes nothing
        d.note_tag_velocity(np.zeros(3), t, tag_age=lat)
        d.note_sample(_sample((0.60, 0, 0), t=t + 0.05))
        d.velocity(t + 0.05, count=False)
        assert d._innov_hold == ""

    # (0.28 s: no frame is ever "fresh" there. Much later than that the
    # gate itself widens past 0.25 m/s — a_max x the age — and only the
    # difference monitor is left to catch such a lock: KNOWN_ISSUES.)
    for lat in (0.13, 0.28):
        run(lat)


# LEGACY-PINNED (2026-10-16, SPEC §10.2a): the shipped ladder = the rollback's test.
def test_the_fix_age_integral_never_contains_a_rejected_sample():
    """The gate judges the NEWEST sample. Once it releases, the integral
    over the fix's age (displacement) must not still hold the samples it
    rejected a few ticks earlier (round 6, reproduced: 9-11 cm into eta with
    the tags in view, then carried through a dropout by the DVL rung)."""
    d = _proven_aid(selection="legacy", reentry_s=0.0, a_max_m_s2=50.0)       # jump gate aside
    d.use = True
    t = 30.0
    for _ in range(10):
        t += 0.05
        d.note_sample(_sample((0.0, 0, 0), t=t))
        d.note_tag_velocity(np.zeros(3), t, tag_age=0.13)
        assert d.velocity(t)[0] is not None
    t_bad0 = t + 0.05
    for v in (0.40, 0.40, 0.40):                          # rejected
        t += 0.05
        d.note_sample(_sample((v, 0, 0), t=t))
        assert d.velocity(t)[1].startswith("DVL innovation (")
        d.note_tag_velocity(np.zeros(3), t, tag_age=0.13)
    t_bad1 = t
    for _ in range(3):                                    # honest again
        t += 0.05
        d.note_sample(_sample((0.0, 0, 0), t=t))
        d.note_tag_velocity(np.zeros(3), t, tag_age=0.13)
        d.velocity(t)
    assert d.velocity(t)[0] is not None                   # released
    # an interval reaching back over the rejected samples: no integral
    assert d.displacement(t_bad0 - 0.1, t) is None
    assert d.displacement(t_bad1, t) is None
    # one that starts after them integrates as before
    dp = d.displacement(t_bad1 + 0.06, t)
    assert dp is not None and np.allclose(dp, 0.0), dp
    # (without the rule the first one would hold 3 x 0.05 s x 0.40 m/s)
    rej, d._innov_reject_t = d._innov_reject_t, None
    assert abs(d.displacement(t_bad0 - 0.1, t)[0] - 0.06) < 0.02
    d._innov_reject_t = rej


# LEGACY-PINNED (2026-10-16, SPEC §10.2a): the shipped ladder = the rollback's test.
def test_a_released_dvl_does_not_carry_what_it_was_rejected_for():
    """Worker level, round 6's scenario: tags 0.30 s late, a false lock
    ramps up and down (each step passes the jump gate), the gate rejects
    the large part, then releases; the tags drop right after. The estimate
    must not keep the rejected motion — in view or through the dropout."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, _pilots, _logs, sim = _station(tmp, selection="legacy")
        try:
            w._bridge.cfg["dvl_hold_s"] = 3.0
            st = []
            bus.mpc_status.connect(st.append)

            def step(v, fix=True):
                sim.step(lat=0.30, dvl=False, fix=fix, tick=False)
                w.on_dvl(_sample((v, 0.0, 0.0), t=sim.t))
                w.tick()
            for _ in range(20):
                step(0.0)
            x0 = float(w._eta[0])
            released, errs = None, []
            for k, v in enumerate([0.14, 0.28, 0.42, 0.56, 0.42, 0.28, 0.14]
                                  + [0.0] * 30):
                step(v, fix=released is None or k <= released + 1)
                if released is not None and k == released + 1:
                    w.fix.t_capture -= 5.0                 # the tags drop
                if released is None and k >= 4 and st[-1].vel_src == "dvl":
                    released = k
                errs.append(float(w._eta[0]) - x0)
            assert released is not None
            # one tick of a sample the (widened) gate accepted is the only
            # trace: 0.14 m/s x 0.30 s; the rejected 0.28-0.56 m/s is not
            assert max(abs(e) for e in errs) < 0.05, max(errs)
            assert abs(errs[-1]) < 0.005, errs[-1]           # 1.5 s into it
        finally:
            sim.close()


# LEGACY-PINNED (2026-10-16, SPEC §10.2a): the shipped ladder = the rollback's test.
def test_a_shrunken_map_is_held_back_not_called_a_mount():
    """What only part of the time is DVL-only motion — repeated short
    glitches, noise on slow motion — does not drive the fit's slope to zero
    but dilutes it to 0.55-0.75, which reads like "the DVL reads 1.4x". At
    COHERENT_MIN 0.5 that was a coherent refutation and a latched trip on a
    right mount (review 2026-10-02, reproduced). A map that only SHRINKS is
    not evidence about the mount: held back, not latched. A rotation or a
    mirror still is."""
    def feed(d, mapping, n=400, t=30.0):
        f = _Pairs(d, t=t)
        w = 2 * math.pi / 8.0

        def tag(tt):
            return (0.12 * math.sin(w * tt), 0.08 * math.sin(w * (tt - 3.0)),
                    0.0)

        def dvl(tt):
            v = np.asarray(tag(tt)[:2])
            return (*(np.linalg.solve(mapping, v)), 0.0)
        f.run(n, dvl, tag, noise_mm=0.3)
        return f

    shrink = np.diag([0.7, 0.7])              # tags see 0.7 of the DVL
    d = _proven_aid(selection="legacy", reentry_s=0.0)
    d.use = True
    f = feed(d, shrink)
    ft = d.agree.fit()
    assert ft["verdict"] == "refuted" and ft["coherent"] is False, ft
    assert 0.65 < ft["gain_min"] < 0.75 and not d.tripped
    assert _vel(d, f.t) == (None, "DVL held back (the fit refutes it)")
    # the same with the old threshold: a latched "scale" trip
    hold, DV.COHERENT_MIN = DV.COHERENT_MIN, 0.5
    try:
        d = _proven_aid(selection="legacy", reentry_s=0.0)
        d.use = True
        feed(d, shrink)
        assert d.tripped and "scale" in d.tripped, d.tripped
    finally:
        DV.COHERENT_MIN = hold
    # a mount 40 degrees out and a mirrored axis are still trips
    c, s_ = math.cos(math.radians(40)), math.sin(math.radians(40))
    for mapping in (np.array([[c, -s_], [s_, c]]), np.diag([1.0, -1.0])):
        d = _proven_aid(selection="legacy", reentry_s=0.0)
        d.use = True
        feed(d, mapping)
        assert d.tripped, mapping


# LEGACY-PINNED (2026-10-16, SPEC §10.2a): the shipped ladder = the rollback's test.
def test_an_alignment_the_tags_only_partly_see_holds_the_dvl_back():
    """The alignment monitor: the DVL's motion along a direction goes
    somewhere the tags do not show. If the tags see most of it (a mount),
    that is a trip; if they see little of it (a glitch at a corner, noise on
    slow motion), the DVL is held back while it lasts and nothing latches
    (review 2026-10-02: one 4-report glitch at a corner of a 5 cm/s square
    tripped a right mount through this monitor at gain 0.5-0.7)."""
    w = 4.0

    def tag(t):
        return (0.06 * math.sin(w * t), 0.0, 0.0)

    def run(mapping, n=200):
        d = _proven_aid(selection="legacy", reentry_s=0.0, agree_trip_m_s=10.0)   # vector off
        d.use = True
        f = _Pairs(d)
        held = 0

        def dvl(t):
            v = np.asarray(tag(t)[:2])
            return (*(np.linalg.solve(mapping, v)), 0.0)
        for _ in range(n):
            f.run(1, dvl, tag, noise_mm=0.3)
            if d.tripped:
                break
            if _vel(d, f.t)[1].startswith("DVL alignment hold ("):
                held += 1
        return d, held

    c, s_ = math.cos(math.radians(70)), math.sin(math.radians(70))
    rot70 = np.array([[c, -s_], [s_, c]])
    d, held = run(0.6 * rot70)                # the tags see 0.6 of it
    assert not d.tripped and held > 100, (d.tripped, held)
    al = d.alignment()
    assert al and al[0]["gain"] < DV.COHERENT_MIN and al[0]["align"] < 0.3
    d, _held = run(rot70)                     # ...or all of it: a mount
    assert d.tripped and "alignment" in d.tripped, d.tripped


# LEGACY-PINNED (2026-10-16, SPEC §10.2a): the shipped ladder = the rollback's test.
def test_a_burst_cannot_trip_the_difference_monitor_whatever_the_tag_noise():
    """Every pair counts as at most innov_max_m_s apart — a FIXED clip.
    Clipped at 4 sigma of the proof's scatter instead, noisy tags (3 mm:
    the proof's raw rms ~9 cm/s, clip 0.36 m/s) let ONE burst of 3-4
    reports trip it in 0.4 s (review 2026-10-02, reproduced). A false lock
    that LASTS still trips."""
    def make():
        d = _proven_aid(selection="legacy", reentry_s=0.0, a_max_m_s2=50.0)     # jump gate aside
        d.use = True
        d.proof = {"rms_m_s": 0.09}           # what 3 mm tags give a proof
        return d

    assert make()._gross_limit() > 0.35
    d = make()
    f = _Pairs(d)
    f.run(60, (0.0, 0, 0), (0.0, 0, 0), noise_mm=3.0)
    f.run(7, (0.30, 0, 0), (0.0, 0, 0), noise_mm=3.0)     # ~0.35 s burst
    f.run(60, (0.0, 0, 0), (0.0, 0, 0), noise_mm=3.0)
    assert not d.tripped, d.tripped
    d = make()
    f = _Pairs(d)
    f.run(60, (0.0, 0, 0), (0.0, 0, 0), noise_mm=3.0)
    t0 = f.t
    f.run(60, (0.30, 0, 0), (0.0, 0, 0), noise_mm=3.0)    # a false lock
    assert d.tripped and "false bottom lock" in d.tripped, d.tripped
    assert 0.6 < f.t - t0 < 1.6, f.t - t0


# LEGACY-PINNED (2026-10-16, SPEC §10.2a): the shipped ladder = the rollback's test.
def test_the_refutation_hold_ends_when_its_evidence_does():
    """The fit's window keeps refuting pairs for agree_window_s, and with
    the vehicle at rest no new pair replaces them: a DVL that went quiet
    again stayed held back for ~52 s (review 2026-10-02). The hold lasts
    REFUTE_HOLD_S past the newest refuting evaluation."""
    d = _proven_aid(selection="legacy", reentry_s=0.0)
    d.use = True
    rng = np.random.default_rng(3)
    f = _Pairs(d)
    f.run(300, lambda t: tuple(rng.normal(0, 0.04, 3)), (0.0, 0, 0),
          noise_mm=0.5, dt=0.1)               # 30 s of a noisy DVL at rest
    assert _vel(d, f.t) == (None, "DVL held back (the fit refutes it)")
    t_quiet = f.t
    released = None
    for _ in range(200):
        f.run(1, (0.0, 0, 0), (0.0, 0, 0), noise_mm=0.5, dt=0.1)
        if _vel(d, f.t)[0] is not None:
            released = f.t - t_quiet
            break
    assert released is not None and released <= DV.REFUTE_HOLD_S + 0.5, (
        released)
    assert d.agree.fit()["verdict"] == "refuted"     # the window still does
    assert not d.tripped


# LEGACY-PINNED (2026-10-16, SPEC §10.2a): the shipped ladder = the rollback's test.
def test_the_alignment_test_still_works_when_the_tags_come_slowly():
    """review 2026-10-02 (reproduced): with one pair every 0.2 s the
    effective pair count settled at 9, under the minimum of 10 — the
    monitor was silently off exactly when tag detection degrades."""
    w = 2 * math.pi * 0.25                    # a slow, wrong-signed sway

    for dt in (0.2, 0.3):
        d = _proven_aid(selection="legacy", reentry_s=0.0, agree_trip_m_s=10.0)
        d.use = True
        f = _Pairs(d)
        f.run(int(30 / dt), lambda t: (-0.06 * math.sin(w * t), 0.0, 0.0),
              lambda t: (0.06 * math.sin(w * t), 0.0, 0.0), noise_mm=0.5,
              dt=dt)
        assert d.tripped and "alignment" in d.tripped, (dt, d.alignment())
        assert f.t - 30.0 < 8.0, (dt, f.t)


def test_with_a_lever_arm_a_sample_needs_the_body_rate_of_its_moment():
    """v_origin = R v_dvl - omega x r. With r != 0 and no rate for the
    sample's moment, the uncorrected reading is not the body-origin
    velocity (0.3 rad/s on a 0.2 m arm is 6 cm/s of it) — it used to be
    stored and flown as good (review 2026-10-02)."""
    d = DV.DvlAid({"mount_rpy_deg": [0, 0, 0], "mount_t_flu": [0.2, 0, 0],
                   "reentry_s": 0.0})
    d.proven, d.use = True, True
    d.note_sample(_sample((0.05, 0.0, 0.0), t=5.0))
    assert _vel(d, 5.0) == (None, "no body rate for the DVL lever arm")
    assert d._hist[-1][2] is False            # a gap for pairs and integrals
    d.note_sample(_sample((0.05, 0.0, 0.0), t=5.1), omega=(0.0, 0.0, 0.3))
    v, why = _vel(d, 5.1)
    assert why == "" and np.allclose(v, [0.05, -0.06, 0.0]), (v, why)
    # no lever arm: nothing to correct, a sample is fine without a rate
    d = _proven_aid(reentry_s=0.0)
    d.use = True
    d.note_sample(_sample((0.05, 0.0, 0.0), t=5.0))
    assert _vel(d, 5.0)[0] is not None


def test_absurd_device_stamps_do_not_reach_the_clock():
    """json parses a 400-digit integer; float() of one raises. The parser's
    contract is "never raises" (review 2026-10-02)."""
    big = 10 ** 400
    for clock in (None, DV.DeviceClock()):
        f = DV.sample_fields(dict(DOC_EXAMPLE, time_of_validity=big,
                                  time_of_transmission=big + 1000),
                             12.0, clock)
        assert f["t_valid"] == 12.0 and f["tov_us"] is None, f
        assert "no device stamps" in f["note"] and f["valid"] is True


def test_the_velocity_integral_follows_the_samples_not_the_newest_one():
    d = _proven_aid()
    # 0.2 m/s for half a second, then a stop
    for k in range(6):
        d.note_sample(_sample((0.2, 0.0, 0.0), t=10.0 + 0.1 * k))
    for k in range(6, 9):
        d.note_sample(_sample((0.0, 0.0, 0.0), t=10.0 + 0.1 * k))
    dp = d.displacement(10.0, 10.8)
    # 0.5 s at 0.2, a 0.1 s linear ramp down (0.01 m), then nothing
    assert abs(dp[0] - (0.2 * 0.5 + 0.01)) < 1e-9 and abs(dp[1]) < 1e-12, dp
    assert np.allclose(d.displacement(10.2, 10.4), [0.04, 0, 0])
    assert np.allclose(d.displacement(10.8, 10.8), 0.0)
    # held flat past the newest sample, but not across a silence
    assert np.allclose(d.displacement(10.7, 11.0), 0.0)
    assert d.displacement(10.7, 12.0) is None
    # a dropout INSIDE the interval: no integral, the caller falls back
    d.note_sample(_sample(valid=False, t=10.9))
    d.note_sample(_sample((0.1, 0.0, 0.0), t=11.0))
    assert d.displacement(10.5, 11.0) is None
    assert d.displacement(10.95, 11.0) is None      # nothing covers its start


# ===================================================================== ladder
def test_the_dvl_rung_holds_only_while_the_chain_is_unbroken():
    br = SB.StationBridge({"imu_hold_s": 1.0, "dvl_hold_s": 2.0})
    assert br.dvl_allowed(0.05)
    assert br.note_lost(0.05, dvl_ok=True) == SB.TIER_DVL
    for _ in range(30):                       # 1.5 s — past imu_hold_s
        assert br.dvl_allowed(0.05)
        assert br.note_lost(0.05, dvl_ok=True) == SB.TIER_DVL
    assert "DVL" in br.detail()
    # the DVL drops out: judged on the SAME elapsed time -> straight to coast
    assert br.note_lost(0.05, dvl_ok=False) == SB.TIER_COAST
    # ...and it does not climb back on, however good the DVL looks again
    assert not br.dvl_allowed(0.05)
    assert br.note_lost(0.05, dvl_ok=True) == SB.TIER_COAST
    rec = br.note_fix(0.0)
    assert br.tier == SB.TIER_NONE and br.dvl_allowed(0.05)
    # the record says how much of the dropout the DVL carried — and that
    # it did NOT carry it to the end (tier coast)
    assert rec["tier"] == SB.TIER_COAST and abs(rec["dvl_s"] - 1.55) < 1e-9
    # the hold time itself runs out
    br.reset()
    tier = None
    for _ in range(45):
        tier = br.note_lost(0.05, dvl_ok=True)
    assert tier == SB.TIER_COAST and br.n_dvl == 1
    m = br.meta()
    assert m["dvl_hold_s"] == 2.0 and m["dropouts_on_dvl"] == 1
    assert m["dvl_max_speed_m_s"] == 0.3 and m["dvl_max_disp_m"] == 0.5
    # without a DVL the ladder is exactly the old one
    br = SB.StationBridge({"imu_hold_s": 1.0})
    assert br.note_lost(0.05) == SB.TIER_IMU
    br = SB.StationBridge({"use_dvl": False})
    assert not br.dvl_allowed(0.05)
    assert br.note_lost(0.05, dvl_ok=True) == SB.TIER_IMU
    for bad in ({"dvl_max_speed_m_s": 0}, {"dvl_max_disp_m": float("nan")},
                {"dvl_max_disp_m": True}):
        try:
            SB.resolve(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")


def test_a_blind_tick_ends_the_dropout_as_it_always_did_and_keeps_the_counters():
    """A tick the bridge could not carry at all (autopilot or barometer
    gone too) ends the dropout: the next lost tick starts a new one with a
    full imu_hold_s. That is what the committed, pre-DVL code does, and a
    build with a DVL in it must not change the ladder of a run that has
    none IN (review 2026-10-02: one cut of this change kept the clock
    running, and a station hold with no DVL anywhere coasted 2.5 s sooner
    than before). What a blind tick must NOT do any more is zero the
    engagement's counters. What keeps the DVL rung from climbing back on
    after the hole is the worker's _dvl_xy — tested with the worker."""
    br = SB.StationBridge({"imu_hold_s": 1.0, "dvl_hold_s": 2.0})
    for _ in range(15):
        assert br.note_lost(0.05) == SB.TIER_IMU      # 0.70 s on the IMU rung
    br.end_dropout()                          # the blind tick
    assert br.tier == SB.TIER_NONE and not br.active and br.elapsed == 0.0
    assert br.n_entered == 1                  # ...the counters are kept
    tier = None
    for _ in range(20):                       # a NEW dropout: 0.95 s more
        tier = br.note_lost(0.05)
    assert tier == SB.TIER_IMU and br.n_entered == 2, (tier, br.elapsed)
    for _ in range(3):
        tier = br.note_lost(0.05)
    assert tier == SB.TIER_COAST
    # disengage keeps the counters for the closing meta; engage zeroes them
    br.end_dropout()
    assert br.meta()["dropouts"] == 2 and br.meta()["reached_coast"] == 1
    assert br.tier == SB.TIER_NONE and br.elapsed == 0.0
    br.reset()
    assert br.meta()["dropouts"] == 0


# ===================================================================== worker
class _Sim:
    """The worker on a STUBBED clock (test_control's shadow-parity rule: the
    tag state is bridged by ``t_now - t_capture``, so on the real clock a
    test would measure scheduling jitter), with a vehicle whose tag fixes
    and DVL samples describe the same motion. Heading 0, so body == NED."""

    def __init__(self, w, TC):
        import rov_gui.control.workers as W
        self.W, self.w, self.TC = W, w, TC
        self.t = 10_000.0
        self.p = np.array([-2.0, 0.0, 0.8])
        self._real = W.now
        W.now = lambda: self.t

    def close(self):
        self.W.now = self._real
        self.w.teardown()

    def step(self, v=(0.0, 0.0, 0.0), fix=True, dvl=True, valid=True,
             tick=True, dvl_v=None, f=None, imu=True, lat=0.0):
        """``lat`` = how late the fix is: its capture stamp is that much
        before now (the position is the current one — use it at rest)."""
        w, TC = self.w, self.TC
        self.t += 0.05
        self.p = self.p + np.asarray(v, float) * 0.05
        if fix:
            w.on_nav_fix(TC._fix(list(self.p), 0.0, self.t - lat))
        if imu:
            w.on_vehicle_imu(TC._imu(depth=0.8, t=self.t, f=f))
        w.on_telemetry(TC.Telemetry(armed=True, mode="MANUAL",
                                    conn=Conn.ONLINE, stamp=self.t))
        if dvl and w.dvl is not None:
            w.on_dvl(_sample(v if dvl_v is None else dvl_v, t=self.t,
                             valid=valid))
        if tick:
            w.tick()

    def prove(self):
        """12 s of surge-and-sway through the worker's OWN slots."""
        for k in range(240):
            tt = k * 0.05
            self.step(v=(0.12 * math.sin(2 * math.pi * tt / 8.0),
                         0.08 * math.sin(2 * math.pi * (tt - 3.0) / 8.0),
                         0.0), tick=False)
        assert self.w.dvl.proven, self.w.dvl.agree.fit()

    def engage(self):
        # One DISENGAGED tick first, as on the station (the worker ticks at
        # 20 Hz whether engaged or not): it is what gives the innovation
        # gate its tag velocity, and without a current one a tick does not
        # fly on the DVL.
        self.step(tick=True)
        self.w.on_enable(True)
        self.w.set_engaged(True)
        assert self.w.engaged, self.w.reason
        self.w._warmup_left = 0


def _dvl_worker(tmp, **over):
    TC = _harness()
    w, bus, pilots, logs = TC._worker(tmp)
    cfg = {"enabled": True}
    cfg.update(MOUNT)
    cfg.update(over)
    w.cfg.dvl = cfg
    w._setup_dvl()
    assert w.dvl is not None
    return TC, w, bus, pilots, logs


def _rows(w):
    import csv
    path = w._csv_path
    w._close_csv()
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_with_no_dvl_the_worker_is_what_it_was_and_says_so():
    TC = _harness()
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, _pilots, _logs = TC._worker(tmp)
        assert w.dvl is None
        sim = _Sim(w, TC)
        try:
            sim.engage()
            st = []
            bus.mpc_status.connect(st.append)
            for _ in range(5):
                sim.step(v=(0.05, 0.0, 0.0))
            assert st and all(s.dvl_state == "" and s.vel_src == ""
                              for s in st)
            w.set_dvl_use(True)               # nothing to switch
            assert "no DVL" in w.reason
            m = w._run_meta()
            assert m["dvl"]["enabled"] is False and m["schema_version"] == 22
            rows = _rows(w)
            assert rows and all(r["vel_src"] == "tag_fd" for r in rows)
            assert all(r["dvl_u"] == "nan" and r["dvl_ok"] == "0"
                       for r in rows)
            # schema 22: no DVL, no weight and no usage code
            assert all(r["w_dvl"] == "0.000" and r["dvl_code"] == ""
                       and r["dvl_esc_s"] == "nan" for r in rows)
            # the body velocity is on disk for the first time, and without a
            # DVL it IS the tag finite difference
            assert all(r["nu_u"] == r["fd_u"] and r["nu_v"] == r["fd_v"]
                       for r in rows)
            assert float(rows[-1]["nu_u"]) > 0.03
        finally:
            sim.close()


def test_the_switch_needs_the_proof_and_is_frozen_while_engaged():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, _bus, _pilots, logs = _dvl_worker(tmp)
        sim = _Sim(w, TC)
        try:
            sim.step(tick=False)
            w.set_dvl_use(True)
            assert not w.dvl.use and "not proven" in w.reason, w.reason
            sim.prove()
            assert any("PROVEN" in m for _l, m in logs)
            w.set_dvl_use(True)
            assert w.dvl.use and w.reason == "DVL IN", w.reason
            sim.engage()
            w.set_dvl_use(False)              # one run = one estimator
            assert w.dvl.use and "refused while engaged" in w.reason
            w.set_engaged(False)
            w.set_dvl_use(False)
            assert not w.dvl.use
        finally:
            sim.close()


# LEGACY-PINNED (2026-10-16, SPEC §10.2a): the shipped ladder = the rollback's test.
def test_in_flies_on_the_dvl_and_falls_back_row_by_row():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, _pilots, _logs = _dvl_worker(tmp, selection="legacy", reentry_s=0.0)
        sim = _Sim(w, TC)
        try:
            sim.prove()
            w.set_dvl_use(True)
            sim.engage()
            st = []
            bus.mpc_status.connect(st.append)
            for _ in range(6):
                sim.step(v=(0.07, -0.02, 0.0))
            assert st[-1].vel_src == "dvl" and st[-1].dvl_state == "in_use"
            assert st[-1].dvl_use and st[-1].dvl_proven
            assert abs(w._last_nu[0] - 0.07) < 1e-9
            for _ in range(4):                # the DVL loses the bottom
                sim.step(v=(0.07, -0.02, 0.0), valid=False)
            assert st[-1].vel_src == "tag_fd"
            assert st[-1].dvl_state == "no_lock"
            assert w.engaged, "losing the DVL must never end the run"
            m = w._run_meta()
            assert m["dvl"]["use_at_engage"] is True
            assert m["dvl"]["proven_at_engage"] is True
            assert m["dvl"]["rows"]["dvl"] == 6
            assert m["dvl"]["rows"]["tag_fd"] == 4
            assert m["dvl"]["rows"]["fallback: DVL has no bottom lock"] == 4
            rows = _rows(w)
            assert [r["vel_src"] for r in rows] == ["dvl"] * 6 + ["tag_fd"] * 4
            assert all(abs(float(r["nu_u"]) - 0.07) < 1e-6 for r in rows[:6])
            assert all(r["dvl_ok"] == "1" for r in rows[:6])
            assert all(r["dvl_ok"] == "0" and r["dvl_valid"] == "0"
                       for r in rows[6:])
            # both instruments are on every row, whichever one flew
            assert all(r["fd_u"] != "nan" for r in rows)
        finally:
            sim.close()


def test_shadow_records_the_dvl_and_does_not_change_one_command():
    """OUT is the imu_dr shadow rule: the same inputs produce the same
    commands with a DVL attached as without one — even a DVL that reads
    nonsense."""
    out = []
    for with_dvl in (False, True):
        with tempfile.TemporaryDirectory() as tmp:
            if with_dvl:
                TC, w, _bus, pilots, _logs = _dvl_worker(tmp)
            else:
                TC = _harness()
                w, _bus, pilots, _logs = TC._worker(tmp)
            sim = _Sim(w, TC)
            try:
                sim.engage()
                for k in range(25):
                    sim.step(v=(0.2 * math.sin(0.3 * k), 0.04, 0.0),
                             dvl_v=(0.5, -0.5, 0.2))
                if with_dvl:
                    assert w._run_meta()["dvl"]["use"] is False
                    rows = _rows(w)
                    assert all(r["vel_src"] == "tag_fd" for r in rows)
                    assert all(abs(float(r["dvl_u"]) - 0.5) < 1e-9
                               for r in rows)
                out.append([(q.surge, q.sway, q.heave, q.yaw)
                            for q in pilots])
            finally:
                sim.close()
    assert out[0] and len(out[0]) == len(out[1])
    assert out[0] == out[1], "a shadow DVL perturbed the commands"


def _station(tmp, **over):
    """A DVL worker, proven, switched IN, holding station; -> (TC, w, bus,
    pilots, logs, sim)."""
    TC, w, bus, pilots, logs = _dvl_worker(tmp, reentry_s=0.0, **over)
    w._bridge.cfg["imu_hold_s"] = 0.2
    w._bridge.cfg["dvl_hold_s"] = 1.0
    w.cfg.engage["settle_s"] = 0.0
    sim = _Sim(w, TC)
    sim.prove()
    w.set_dvl_use(True)
    sim.engage()
    w.set_scenario({"shape": "station", "origin_tag": None,
                    "origin": "current", "yaw_map_deg": 0.0})
    w.set_traj(True)
    for _ in range(3):
        sim.step()
    return TC, w, bus, pilots, logs, sim


# LEGACY-PINNED (2026-10-16, SPEC §10.2a): the shipped ladder = the rollback's test.
def test_the_bridge_carries_a_station_hold_on_the_dvl_then_coasts():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, pilots, logs, sim = _station(tmp, selection="legacy")
        try:
            st = []
            bus.mpc_status.connect(st.append)
            x0 = float(w._eta[0])
            w.fix.t_capture -= 5.0            # the tags are gone
            for _ in range(10):               # 0.5 s, drifting 0.1 m/s ahead
                sim.step(v=(0.1, 0.0, 0.0), fix=False, f=(0.4, 0.0, -9.80665))
            assert w.engaged, w.reason
            assert w._bridge.tier == SB.TIER_DVL, w._bridge.tier
            # dead reckoning: 10 ticks x 0.05 s x 0.1 m/s = 5 cm, and well
            # past the 0.2 s at which the IMU rung would have let go
            assert abs(float(w._eta[0]) - (x0 + 0.05)) < 1e-6, (w._eta[0], x0)
            assert abs(float(w._last_nu[0]) - 0.1) < 1e-9
            assert "DVL bridge" in w._bridge.detail()
            # the panel is told — WITHOUT --imu-dr (it used to hear the
            # ladder's tier only when the dead-reckoning experiment was on)
            assert w.dr is None and st[-1].bridge_tier == "dvl"
            assert st[-1].bridge_s > 0.4 and st[-1].vel_src == "dvl"
            for _ in range(4):                # the DVL goes too
                sim.step(fix=False, valid=False)
            assert w.engaged
            assert w._bridge.tier == SB.TIER_COAST, w._bridge.tier
            assert st[-1].bridge_tier == "coast"
            a = pilots[-1]
            assert a.surge == 0.0 and a.sway == 0.0 and a.yaw == 0.0, a
            # the next rung holds where the DVL last put the vehicle, not
            # back at the fix it left 5 cm ago
            assert abs(float(w._eta[0]) - (x0 + 0.05)) < 1e-6, w._eta[0]
            # ...and a DVL that comes back does NOT reopen its rung
            for _ in range(3):
                sim.step(fix=False)
            assert w._bridge.tier == SB.TIER_COAST and w._dvl_xy is None
            sim.step()                        # the tags come back
            assert not w._bridge.active
            # The chain broke before the fix returned: the distance to the
            # fix is NOT the DVL's error and must not be written down as one.
            rec = w._bridge.last_recover
            assert rec["tier"] == "coast" and abs(rec["dvl_s"] - 0.5) < 1e-9
            assert "dvl_err_m" not in rec, rec
            assert not any("DVL dead reckoning was" in m for _l, m in logs)
            assert any("the DVL carried the first 0.5s" in m
                       for _l, m in logs), logs[-4:]
            # the run meta books bridged rows under the rung that carried
            # them, not under the pre-bridge "no tag velocity" answer
            mr = w._run_meta()["dvl"]["rows"]
            assert mr.get("bridge: tier coast") == 7, mr
            assert mr.get("fallback: DVL unchecked", 0) <= 1, mr
            rows = _rows(w)
            tiers = [r["bridge_tier"] for r in rows]
            assert tiers.count("dvl") == 10 and tiers.count("coast") == 7
            assert all(r["vel_src"] == "dvl" for r in rows
                       if r["bridge_tier"] == "dvl")
            assert all(r["vel_src"] == "bridge" for r in rows
                       if r["bridge_tier"] == "coast")
        finally:
            sim.close()


# LEGACY-PINNED (2026-10-16, SPEC §10.2a), with a twin under primary below.
def test_a_dropout_carried_to_its_end_on_the_dvl_measures_the_dead_reckoning(selection="legacy"):
    """Only THEN is the gap to the returning fix the DVL's own error — the
    number that will replace the [예측] dvl_hold_s."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, _bus, _pilots, logs, sim = _station(tmp, selection=selection)
        try:
            w.fix.t_capture -= 5.0
            for _ in range(8):
                sim.step(v=(0.1, 0.05, 0.0), fix=False)
            assert w._bridge.tier == SB.TIER_DVL
            sim.step(v=(0.1, 0.05, 0.0))      # the tags come back
            rec = w._bridge.last_recover
            assert rec["tier"] == "dvl" and rec["dvl_s"] > 0.35
            assert rec["dvl_err_m"] < 0.011 and abs(rec["yaw_err_deg"]) < 1e-6
            assert any("DVL dead reckoning was" in m and "gyro yaw" in m
                       for _l, m in logs), logs[-3:]
            # THE CLOSING META KEEPS IT. Until 2026-10-02 the disengage
            # reset the ladder before the meta was written, so every run
            # closed with dropouts 0 / last_recovery null on disk.
            path = w._csv_path
            w.set_engaged(False)
            sb = json.loads(Path(path).with_suffix(".meta.json").read_text()
                            )["station_bridge"]
            assert sb["dropouts"] == 1 and sb["dropouts_on_dvl"] == 1, sb
            assert sb["last_recovery"]["tier"] == "dvl"
            assert sb["last_recovery"]["dvl_err_m"] < 0.011
            sim.engage()                      # ...and a new engagement
            assert w._bridge.meta()["dropouts"] == 0      # starts from zero
        finally:
            sim.close()


def test_a_dropout_carried_to_its_end_on_the_dvl_measures_the_dead_reckoning_under_primary():
    """The same, under dvl.selection primary (2026-10-16): mode-independent in substance."""
    test_a_dropout_carried_to_its_end_on_the_dvl_measures_the_dead_reckoning(selection="primary")


# LEGACY-PINNED (2026-10-16, SPEC §10.2a), with a twin under primary below.
def test_the_dvl_rung_is_bounded_by_what_a_station_hold_is(selection="legacy"):
    """Nothing cross-checks the DVL while the tag is gone, so the rung
    refuses a DVL that describes anything but a vehicle nominally at rest
    (review 2026-10-02: 10 s of a false 0.2 m/s was 2 m of full-authority
    push the wrong way)."""
    for kind in ("speed", "displacement"):
        with tempfile.TemporaryDirectory() as tmp:
            TC, w, _bus, _pilots, logs, sim = _station(tmp, selection=selection)
            try:
                if kind == "displacement":
                    w._bridge.cfg["dvl_max_disp_m"] = 0.032
                x0 = float(w._eta[0])
                w.fix.t_capture -= 5.0
                if kind == "speed":
                    # (0.3 s into the dropout first: until the last tag
                    # velocity has aged out, the innovation gate would catch
                    # this before the cap had to)
                    for _ in range(6):
                        sim.step(dvl_v=(0.0, 0.0, 0.0), fix=False)
                    for v in (0.1, 0.2, 0.3):  # ramped, so it is no "jump"
                        sim.step(dvl_v=(v, 0.0, 0.0), fix=False)
                    assert w._bridge.tier == SB.TIER_DVL
                    sim.step(dvl_v=(0.4, 0.0, 0.0), fix=False)
                else:
                    for _ in range(6):
                        sim.step(dvl_v=(0.1, 0.0, 0.0), fix=False)
                    assert w._bridge.tier == SB.TIER_DVL
                    sim.step(dvl_v=(0.1, 0.0, 0.0), fix=False)
                assert w._bridge.tier != SB.TIER_DVL, (kind, w._bridge.tier)
                assert w._dvl_xy is None
                # The rung has just called its DVL implausible, so the
                # position that DVL integrated is NOT what the next rung
                # holds: back to the last fix (review 2026-10-02: the
                # ladder kept the vehicle pushing against up to
                # dvl_max_disp_m of it until imu_hold_s ran out).
                x1 = float(w._eta[0])
                assert abs(x1 - x0) < 1e-9, (kind, x1 - x0)
                assert abs(float(w._bridge_anchor[0]) - x0) < 1e-9
                assert sum(1 for _l, m in logs if "DVL rung closed" in m
                           and "back to the last fix" in m) == 1, logs[-3:]
                for _ in range(5):            # and it stays closed
                    sim.step(dvl_v=(0.0, 0.0, 0.0), fix=False)
                    assert w._bridge.tier != SB.TIER_DVL
                assert abs(float(w._eta[0]) - x0) < 1e-9
                assert w.engaged
            finally:
                sim.close()


def test_the_dvl_rung_is_bounded_by_what_a_station_hold_is_under_primary():
    """The same, under dvl.selection primary (2026-10-16): mode-independent in substance."""
    test_the_dvl_rung_is_bounded_by_what_a_station_hold_is(selection="primary")


# LEGACY-PINNED (2026-10-16, SPEC §10.2a), with a twin under primary below.
def test_the_dvl_rung_tells_the_observer_about_the_acceleration(selection="legacy"):
    """A bridge state that says "moving at 0.1 m/s, accelerating at zero"
    for ten seconds pulls the EAOB's disturbance estimate by M x the real
    acceleration. The rung builds nudot the way the assembler does."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, _bus, _pilots, _logs, sim = _station(tmp, selection=selection)
        try:
            seen = []
            real = w.ctrl.step

            def spy(eta, nu, nudot, t):
                seen.append(np.asarray(nudot, float).copy())
                return real(eta, nu, nudot, t)
            w.ctrl.step = spy
            w.fix.t_capture -= 5.0
            for _ in range(4):
                sim.step(v=(0.1, 0.0, 0.0), fix=False, f=(0.4, 0.0, -9.80665))
            assert w._bridge.tier == SB.TIER_DVL
            # level, no rates: a = f + R^T g - omega x v = (0.4, 0, 0)
            assert np.allclose(seen[-1][:3], [0.4, 0.0, 0.0], atol=1e-9), seen[-1]
        finally:
            sim.close()


def test_the_dvl_rung_tells_the_observer_about_the_acceleration_under_primary():
    """The same, under dvl.selection primary (2026-10-16): mode-independent in substance."""
    test_the_dvl_rung_tells_the_observer_about_the_acceleration(selection="primary")


# LEGACY-PINNED (2026-10-16, SPEC §10.2a): the shipped ladder = the rollback's test.
def test_a_tick_the_rung_did_not_integrate_closes_it_for_the_dropout():
    """The rung's position is the last fix plus the integral of EVERY tick
    since. A tick it did not integrate — the bridge blind for one tick, or
    a fault the bridge does not carry at all — leaves a hole, and only a
    fresh fix may open the rung again."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, _bus, _pilots, _logs, sim = _station(tmp, selection="legacy")
        try:
            w.fix.t_capture -= 5.0
            for _ in range(4):
                sim.step(v=(0.05, 0.0, 0.0), fix=False)
            assert w._bridge.tier == SB.TIER_DVL and w._dvl_xy is not None
            n = w._bridge.n_entered
            real = w._bridge_meas_freeze
            w._bridge_meas_freeze = lambda *a: None   # autopilot quiet a tick
            sim.step(v=(0.05, 0.0, 0.0), fix=False)
            w._bridge_meas_freeze = real
            assert w._bridge.tier == SB.TIER_NONE and w._dvl_xy is None
            assert w.engaged, w.reason
            tiers = []
            for _ in range(3):                # the DVL is fine again...
                sim.step(v=(0.05, 0.0, 0.0), fix=False)
                tiers.append(w._bridge.tier)
            assert SB.TIER_DVL not in tiers, tiers    # ...the rung is not
            # the ladder itself restarted, as it does without any DVL, and
            # the engagement's counters were not wiped by the blind tick
            assert tiers[0] == SB.TIER_IMU and w._bridge.n_entered == n + 1
            sim.step()                        # the tags are back: the first
            assert w._dvl_xy is None          # tick has nothing to check the
            sim.step()                        # DVL against, the second flies
            assert w._dvl_xy is not None      # on it — and re-opens the rung
            # a fault the bridge does not carry (the autopilot itself stale)
            w._station_bridge(None, {"why": "imu stale"}, sim.t, 0.05)
            assert w._dvl_xy is None
        finally:
            sim.close()


# LEGACY-PINNED (2026-10-16, SPEC §10.2a): the shipped ladder = the rollback's test.
def test_the_rung_does_not_take_a_dvl_the_gate_was_rejecting():
    """The rung has no tag to check the DVL against — so it only opens for
    a DVL that was actually FLOWN on the last tagged tick. One the
    innovation gate was holding back while the tag was there to contradict
    it is not made trustworthy by that tag going away (review 2026-10-02,
    reproduced: a false 0.25 m/s, rejected on the last tagged ticks, was
    taken by the rung on the first lost one and walked the estimate 0.5 m
    with the controller pushing against it)."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, _pilots, _logs, sim = _station(tmp, selection="legacy")
        try:
            st = []
            bus.mpc_status.connect(st.append)
            x0 = float(w._eta[0])
            for v in (0.1, 0.2, 0.25, 0.25, 0.25, 0.25):   # a false lock,
                sim.step(dvl_v=(v, 0.0, 0.0))              # tags in view
            assert st[-1].vel_src == "tag_fd" and not w.dvl.tripped
            assert st[-1].dvl_note.startswith("DVL innovation ("), (
                st[-1].dvl_note)
            assert w._dvl_xy is None
            w.fix.t_capture -= 5.0                         # the tags drop
            tiers = []
            for _ in range(8):
                sim.step(dvl_v=(0.25, 0.0, 0.0), fix=False)
                tiers.append(w._bridge.tier)
            assert SB.TIER_DVL not in tiers, tiers
            assert abs(float(w._eta[0]) - x0) < 1e-6, float(w._eta[0]) - x0
            assert w._bridge.meta()["dropouts_on_dvl"] == 0
        finally:
            sim.close()


# LEGACY-PINNED (2026-10-16, SPEC §10.2a): the shipped ladder = the rollback's test.
def test_the_tags_fading_out_do_not_let_a_rejected_dvl_back_in():
    """How tags REALLY disappear: the fix simply stops arriving, and the
    assembler bridges the last one until tag_stale_s — a tag state with an
    ageing fix, the innovation gate's limit widening all the while. A false
    lock the gate rejected while the tags were fresh stays out through that
    phase, and the rung does not open on it (review 2026-10-02, reproduced
    the opposite: the DVL went back in 0.35 s after the last fix, the rung
    carried it 0.6 m, and the closure went "back" to a point 0.125 m off)."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, pilots, logs, sim = _station(tmp, selection="legacy")
        try:
            w._bridge.cfg["imu_hold_s"] = 3.0
            w._bridge.cfg["dvl_hold_s"] = 10.0
            st = []
            bus.mpc_status.connect(st.append)
            x0 = float(w._eta[0])
            for v in (0.1, 0.2, 0.25, 0.25, 0.25, 0.25):   # rejected,
                sim.step(dvl_v=(v, 0.0, 0.0))              # tags in view
            assert st[-1].vel_src == "tag_fd"
            src, tiers, notes = [], [], []
            for _ in range(40):                            # the fix stops
                sim.step(dvl_v=(0.25, 0.0, 0.0), fix=False)
                src.append(st[-1].vel_src)
                tiers.append(w._bridge.tier)
                notes.append(st[-1].dvl_note)
            assert "dvl" not in src and SB.TIER_DVL not in tiers, (src, tiers)
            assert any(n.startswith("DVL innovation hold (") for n in notes)
            assert abs(float(w._eta[0]) - x0) < 1e-6, float(w._eta[0]) - x0
            assert not w.dvl.tripped and w.engaged
            assert not any("DVL rung closed" in m for _l, m in logs)
        finally:
            sim.close()


# LEGACY-PINNED (2026-10-16, SPEC §10.2a): the shipped ladder = the rollback's test.
def test_late_tags_fading_out_do_not_let_a_rejected_dvl_back_in_either():
    """The same with tags 0.28 s late — no frame is ever "fresh" there.
    Keyed on freshness, the hold was then never set, and the station bridge
    carried the rejected false lock 0.5 m (round 5, reproduced)."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, pilots, logs, sim = _station(tmp, selection="legacy")
        try:
            w._bridge.cfg["imu_hold_s"] = 3.0
            w._bridge.cfg["dvl_hold_s"] = 10.0
            w.asm.tag_stale_s = 0.7
            st = []
            bus.mpc_status.connect(st.append)
            for _ in range(12):                            # late tags, at rest
                sim.step(lat=0.28)
            assert st[-1].vel_src == "dvl", st[-1].dvl_note
            x0 = float(w._eta[0])
            for v in (0.1, 0.2, 0.25, 0.25, 0.25, 0.25):   # a false lock
                sim.step(dvl_v=(v, 0.0, 0.0), lat=0.28)
            assert st[-1].vel_src == "tag_fd", st[-1].dvl_note
            src, tiers = [], []
            for _ in range(40):                            # the fix stops
                sim.step(dvl_v=(0.25, 0.0, 0.0), fix=False)
                src.append(st[-1].vel_src)
                tiers.append(w._bridge.tier)
            assert "dvl" not in src and SB.TIER_DVL not in tiers, (src, tiers)
            assert abs(float(w._eta[0]) - x0) < 0.02, float(w._eta[0]) - x0
            assert not any("DVL rung closed" in m for _l, m in logs)
        finally:
            sim.close()


# LEGACY-PINNED (2026-10-16, SPEC §10.2a), with a twin under primary below.
def test_the_imu_rung_never_inherits_a_dvl_velocity(selection="legacy"):
    """The IMU rung's dead reckoner is anchored on every tagged tick and,
    while the DVL rung carries a dropout, on every rung tick. Neither may
    seed it with the DVL's velocity: a false lock that ENDS (lock lost,
    link silent, snapped back) rather than tripping a cap hands it a
    velocity nothing bounds any more, integrated for the rest of
    imu_hold_s — 0.67-1.2 m on a vehicle at rest (round 5). Tagged ticks
    seed it with the TAG velocity, as before the DVL; rung ticks at rest."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, _bus, _pilots, _logs, sim = _station(tmp, selection=selection)
        try:
            assert w._bridge_dr is not None
            seen = []
            real = w._bridge_dr.anchor

            def spy(eta, nu_world_ned=None, t=None, zero_velocity=True):
                seen.append((w._bridge.tier, None if zero_velocity else
                             np.asarray(nu_world_ned, float).copy()))
                return real(eta, nu_world_ned=nu_world_ned, t=t,
                            zero_velocity=zero_velocity)
            w._bridge_dr.anchor = spy
            # a tagged tick flying on a DVL that reads 0.12 while the tags
            # say 0.10: the anchor gets the tags' 0.10-ish, not 0.12
            for _ in range(10):
                sim.step(v=(0.10, 0.0, 0.0), dvl_v=(0.12, 0.0, 0.0))
            tagged = [v for tier, v in seen if tier == SB.TIER_NONE]
            assert tagged and tagged[-1] is not None
            assert abs(float(tagged[-1][0]) - 0.10) < 0.01, tagged[-1]
            assert abs(float(w._last_nu[0]) - 0.12) < 1e-6   # flew on the DVL
            # ...and on the DVL rung: at rest, every time
            seen.clear()
            w.fix.t_capture -= 5.0
            for _ in range(6):
                sim.step(v=(0.10, 0.0, 0.0), dvl_v=(0.12, 0.0, 0.0),
                         fix=False)
            on_rung = [v for tier, v in seen if tier == SB.TIER_DVL]
            assert on_rung and all(v is None for v in on_rung), on_rung
        finally:
            sim.close()


def test_the_imu_rung_never_inherits_a_dvl_velocity_under_primary():
    """The same, under dvl.selection primary (2026-10-16): mode-independent in substance."""
    test_the_imu_rung_never_inherits_a_dvl_velocity(selection="primary")


# LEGACY-PINNED (2026-10-16, SPEC §10.2a): the shipped ladder = the rollback's test.
def test_the_panel_gives_a_reason_on_bridged_and_tagless_ticks():
    """A fallback with no reason on the panel (round 5: about five ticks
    per dropout on the IMU rung, and on tagless ticks without a station)."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, _pilots, _logs, sim = _station(tmp, selection="legacy")
        try:
            w._bridge.cfg["use_dvl"] = False          # the IMU rung carries it
            st = []
            bus.mpc_status.connect(st.append)
            w.fix.t_capture -= 5.0
            for _ in range(3):
                sim.step(fix=False)
            assert st[-1].vel_src == "bridge"
            assert st[-1].dvl_state == DV.ST_FALLBACK
            assert "station bridge on tier" in st[-1].dvl_note, st[-1].dvl_note
        finally:
            sim.close()


# LEGACY-PINNED (2026-10-16, SPEC §10.2a): the shipped ladder = the rollback's test.
def test_the_panel_says_what_the_tick_was_told():
    """The panel's reason is the one the tick itself got, not a second
    question to the aid — which can answer differently, the gate's
    reference having moved on since (review 2026-10-02)."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, _pilots, _logs = _dvl_worker(tmp, selection="legacy", reentry_s=0.0)
        sim = _Sim(w, TC)
        try:
            sim.prove()
            w.set_dvl_use(True)
            sim.engage()
            st = []
            bus.mpc_status.connect(st.append)

            def two_answers(t, need_ref=True, count=True):
                return None, ("DVL tick reason (1)" if count else
                              "DVL second opinion (2)")
            w.dvl.velocity = two_answers
            sim.step(v=(0.05, 0.0, 0.0))
            assert st[-1].vel_src == "tag_fd"
            assert st[-1].dvl_state == DV.ST_FALLBACK
            assert st[-1].dvl_note == "DVL tick reason (1)", st[-1].dvl_note
        finally:
            sim.close()


# LEGACY-PINNED (2026-10-16, SPEC §10.2a), with a twin under primary below.
def test_a_healthy_dvl_carries_the_real_tag_dropout(selection="legacy"):
    """...and the same real sequence with an honest DVL: it flies through
    the ageing phase and the rung carries it, tracking a drift the frozen
    ladder would not see."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, _pilots, _logs, sim = _station(tmp, selection=selection)
        try:
            w._bridge.cfg["dvl_hold_s"] = 10.0
            st = []
            bus.mpc_status.connect(st.append)
            x0, true_x = float(w._eta[0]), 0.0
            for _ in range(30):                            # the fix stops
                sim.step(v=(0.1, 0.0, 0.0), fix=False)
                true_x += 0.005
            assert all(s.vel_src == "dvl" for s in st[-30:]), {
                s.vel_src for s in st[-30:]}
            assert w._bridge.tier == SB.TIER_DVL
            assert abs((float(w._eta[0]) - x0) - true_x) < 0.01, (
                float(w._eta[0]) - x0, true_x)
        finally:
            sim.close()


def test_a_healthy_dvl_carries_the_real_tag_dropout_under_primary():
    """The same, under dvl.selection primary (2026-10-16): mode-independent in substance."""
    test_a_healthy_dvl_carries_the_real_tag_dropout(selection="primary")


# LEGACY-PINNED (2026-10-16, SPEC §10.2a), with a twin under primary below.
def test_a_refused_rung_goes_back_to_the_fix_not_to_an_eta(selection="legacy"):
    """A false lock that starts only once the fix is old (the gate can no
    longer tell it from a real acceleration) is flown, and the rung carries
    it until a cap closes it. "Back to the last fix" then means the FIX —
    not the last tagged tick's eta, which by then holds the DVL's integral
    over the fix's age."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, _pilots, logs, sim = _station(tmp, selection=selection)
        try:
            w._bridge.cfg["imu_hold_s"] = 3.0
            w._bridge.cfg["dvl_hold_s"] = 10.0
            w._bridge.cfg["dvl_max_disp_m"] = 0.10
            x_fix = float(w._eta[0])                       # at rest
            for _ in range(5):                             # fix stops; the
                sim.step(dvl_v=(0.0, 0.0, 0.0), fix=False)  # DVL is honest
            v = 0.0
            closed_at = None
            for k in range(60):                            # then false-locks
                v = min(0.25, v + 0.05)
                sim.step(dvl_v=(v, 0.0, 0.0), fix=False)
                if any("DVL rung closed" in m for _l, m in logs):
                    closed_at = k
                    break
            assert closed_at is not None, w._bridge.tier
            assert abs(float(w._eta[0]) - x_fix) < 1e-6, (
                float(w._eta[0]) - x_fix)
            assert abs(float(w._dvl_xy0[0]) - x_fix) < 1e-6
        finally:
            sim.close()


def test_a_refused_rung_goes_back_to_the_fix_not_to_an_eta_under_primary():
    """The same, under dvl.selection primary (2026-10-16): mode-independent in substance."""
    test_a_refused_rung_goes_back_to_the_fix_not_to_an_eta(selection="primary")


def test_the_dvl_bookkeeping_cannot_abort_engage_the_switch_or_the_meta():
    """Intent E off the tick path: set_engaged asks the aid whether to arm,
    the switch asks it too, and the run meta — written from set_engaged and
    from _close_csv, which a disengage inside tick() calls — asks it for
    its record. None of the three may raise into the worker (review
    2026-10-02: an exception in meta() left a closed CSV handle behind and
    every later tick raised)."""
    def boom(*_a, **_k):
        raise RuntimeError("synthetic DVL bug")

    with tempfile.TemporaryDirectory() as tmp:                  # arm
        TC, w, _bus, pilots, _logs = _dvl_worker(tmp, reentry_s=0.0)
        sim = _Sim(w, TC)
        try:
            sim.prove()
            w.set_dvl_use(True)
            w.dvl.refusal = boom
            w.dvl.arm_refusal = boom          # (primary arms on this, 2026-10-16)
            sim.engage()
            assert w.engaged and "arm: RuntimeError" in w._dvl_fault
            assert w._dvl_run_ok is False
            n0 = len(pilots)
            for _ in range(3):
                sim.step(v=(0.05, 0.0, 0.0))
            assert len(pilots) == n0 + 3
        finally:
            sim.close()
    # 2026-10-16, selection primary: the blend's own entry points — at engage
    # (blend_reset), and on the tick (usable / blend_step) — latch the fault,
    # and the engagement and the tick go on, on the tag velocity (w 0)
    for name in ("blend_reset", "usable", "blend_step"):
        with tempfile.TemporaryDirectory() as tmp:
            TC, w, _bus, pilots, _logs = _dvl_worker(tmp, reentry_s=0.0)
            sim = _Sim(w, TC)
            try:
                assert w.dvl.selection == "primary"
                sim.prove()
                w.set_dvl_use(True)
                setattr(w.dvl, name, boom)
                sim.engage()
                n0 = len(pilots)
                for _ in range(3):
                    sim.step(v=(0.05, 0.0, 0.0))
                assert w.engaged and len(pilots) == n0 + 3, name
                assert "RuntimeError" in w._dvl_fault, (name, w._dvl_fault)
                rows = _rows(w)
                assert rows and all(r["w_dvl"] == "0.000" and
                                    r["vel_src"] == "tag_fd" for r in rows), \
                    (name, [(r["w_dvl"], r["vel_src"]) for r in rows])
                assert all(r["nu_u"] == r["fd_u"] for r in rows), name
            finally:
                sim.close()
    with tempfile.TemporaryDirectory() as tmp:                  # the switch
        TC, w, _bus, _pilots, _logs = _dvl_worker(tmp, reentry_s=0.0)
        sim = _Sim(w, TC)
        try:
            sim.prove()
            w.dvl.set_use = boom
            w.set_dvl_use(True)
            assert not w.dvl.use and "code fault" in w.reason, w.reason
            assert "switch: RuntimeError" in w._dvl_fault
        finally:
            sim.close()
    with tempfile.TemporaryDirectory() as tmp:                  # the meta
        TC, w, _bus, pilots, _logs = _dvl_worker(tmp, reentry_s=0.0)
        sim = _Sim(w, TC)
        try:
            sim.prove()
            w.set_dvl_use(True)
            w.dvl.meta = boom
            sim.engage()                      # writes the open-time meta
            assert w.engaged and "meta: RuntimeError" in w._dvl_fault
            m = w._run_meta()["dvl"]
            assert "synthetic DVL bug" in m["meta_error"]
            assert m["armed_at_engage"] is True and m["code_fault"]
            n0 = len(pilots)
            for _ in range(3):
                sim.step(v=(0.05, 0.0, 0.0))
            assert w.engaged and len(pilots) == n0 + 3
            w.set_engaged(False)              # closes the CSV, writes the meta
            assert w._csv is None
            sim.engage()                      # ...and the station can re-engage
            assert w.engaged
        finally:
            sim.close()


# LEGACY-PINNED (2026-10-16, SPEC §10.2a): the shipped ladder = the rollback's test.
def test_a_proof_at_the_edge_of_the_tolerance_says_at_what_speed_it_trips():
    """The proof's tolerance is relative (0.25 of the speed) and the
    difference monitor's threshold absolute (8 cm/s): a mount proven 10
    degrees out trips above ~0.4 m/s on that residual alone. The operator
    is told when the proof is made, with the yaw to correct by."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, _bus, _pilots, logs = _dvl_worker(tmp, selection="legacy")
        sim = _Sim(w, TC)
        try:
            c, s = math.cos(math.radians(10)), math.sin(math.radians(10))
            for k in range(240):
                tt = k * 0.05
                v = (0.12 * math.sin(2 * math.pi * tt / 8.0),
                     0.08 * math.sin(2 * math.pi * (tt - 3.0) / 8.0), 0.0)
                sim.step(v=v, tick=False,
                         dvl_v=(c * v[0] + s * v[1], -s * v[0] + c * v[1], 0))
            assert w.dvl.proven
            warn = [m for _l, m in logs
                    if "trips the difference monitor" in m]
            assert len(warn) == 1, warn
            import re
            mm = re.search(r"above ([0-9.]+) m/s .* yaw by \+([0-9.]+) deg",
                           warn[0])
            # (this harness stamps the DVL half a tick off the tag
            # difference, which on its elliptical pass reads as ~1.7 deg
            # more: the lag-is-rotation effect the fit's lag_s is there for)
            assert mm and 0.35 < float(mm.group(1)) < 0.50, warn[0]
            assert 9.0 < float(mm.group(2)) < 13.0, warn[0]
        finally:
            sim.close()
    with tempfile.TemporaryDirectory() as tmp:
        # a mount that is right says nothing of the kind
        TC, w, _bus, _pilots, logs = _dvl_worker(tmp, selection="legacy")
        sim = _Sim(w, TC)
        try:
            sim.prove()
            assert not any("trips the difference monitor" in m
                           for _l, m in logs)
        finally:
            sim.close()


def test_an_out_dvl_does_not_touch_the_ladder():
    """Shadow means shadow on the bridge too: with the switch OUT a dropout
    climbs the pre-DVL ladder, however good the DVL looks."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, _bus, _pilots, _logs = _dvl_worker(tmp, reentry_s=0.0)
        w._bridge.cfg["imu_hold_s"] = 0.2
        w.cfg.engage["settle_s"] = 0.0
        sim = _Sim(w, TC)
        try:
            sim.prove()                       # proven, but left OUT
            sim.engage()
            w.set_scenario({"shape": "station", "origin_tag": None,
                            "origin": "current", "yaw_map_deg": 0.0})
            w.set_traj(True)
            for _ in range(3):
                sim.step()
            w.fix.t_capture -= 5.0
            seen = set()
            calls = []
            real = w._bridge_meas_dvl
            w._bridge_meas_dvl = lambda *a: (calls.append(1), real(*a))[1]
            for _ in range(10):
                sim.step(fix=False)
                seen.add(w._bridge.tier)
            assert SB.TIER_DVL not in seen, seen
            assert w._bridge.tier == SB.TIER_COAST
            assert w._bridge.meta()["dropouts_on_dvl"] == 0
            # ...and the rung's code is not even entered: building its
            # state integrates the gyro yaw a tick early, which with an IMU
            # rung underneath is a number the no-DVL build does not have
            assert not calls
        finally:
            sim.close()


def test_the_dvl_rung_is_station_only_like_the_rest_of_the_ladder():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, _bus, _pilots, _logs = _dvl_worker(tmp, reentry_s=0.0)
        sim = _Sim(w, TC)
        try:
            sim.prove()
            w.set_dvl_use(True)
            sim.engage()                      # a plain DP hold, no station
            for _ in range(3):
                sim.step()
            w.fix.t_capture -= 5.0
            for _ in range(30):
                sim.step(fix=False)
            assert not w.engaged and "stale" in w.reason, w.reason
        finally:
            sim.close()


# LEGACY-PINNED (2026-10-16, SPEC §10.2a): the shipped ladder = the rollback's test.
def test_a_run_is_armed_for_the_dvl_at_engage_or_not_at_all():
    """One run = one estimator binds the automation too. A proof that
    completes in the middle of a mission — the switch left IN by the YAML,
    or a re-proof after a trip — must not move the velocity source under
    it (review 2026-10-02)."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, _pilots, logs, _ = (*_dvl_worker(tmp, selection="legacy", use=True,
                                                    reentry_s=0.0), None)
        sim = _Sim(w, TC)
        try:
            assert w.dvl.use and not w.dvl.proven   # IN from the file
            sim.engage()
            assert w._dvl_run_ok is False
            assert "not armed" in w._dvl_run_why.lower()
            assert any("flies on the tag finite difference" in m
                       for _l, m in logs)
            st = []
            bus.mpc_status.connect(st.append)
            for k in range(240):              # the proof completes MID-RUN
                tt = k * 0.05
                sim.step(v=(0.12 * math.sin(2 * math.pi * tt / 8.0),
                            0.08 * math.sin(2 * math.pi * (tt - 3.0) / 8.0),
                            0.0))
            assert w.dvl.proven
            assert all(s.vel_src == "tag_fd" for s in st), {
                s.vel_src for s in st}
            # the panel says WHY a proven, healthy DVL that is IN is not
            # flying (the aid alone would say "fallback" and nothing else)
            assert st[-1].dvl_state == "fallback"
            assert "not armed at engage" in st[-1].dvl_note, st[-1].dvl_note
            assert any("used from the next engagement" in m
                       for _l, m in logs)
            m = w._run_meta()["dvl"]
            assert m["use_at_engage"] is True and m["armed_at_engage"] is False
            assert m["not_armed_why"] and m["rows"].get("dvl", 0) == 0
            w.set_engaged(False)
            sim.engage()                      # the NEXT engagement is armed
            assert w._dvl_run_ok is True
            st.clear()
            for _ in range(5):
                sim.step(v=(0.05, 0.0, 0.0))
            assert all(s.vel_src == "dvl" for s in st)
            assert w._run_meta()["dvl"]["armed_at_engage"] is True
            # A recording with NO engagement in it does not inherit that
            # (review 2026-10-02: REC after an armed run wrote
            # armed_at_engage true into a shadow recording).
            w.set_engaged(False)
            w.set_dvl_use(False)
            w._open_csv(Path(tmp) / "shadow_only", auto=False)
            m = w._run_meta()["dvl"]
            assert m["use"] is False and m["use_at_engage"] is None
            assert m["armed_at_engage"] is None and m["rows"] == {}
            w._close_csv()
        finally:
            sim.close()


# LEGACY-PINNED (2026-10-16, SPEC §10.2a): the shipped ladder = the rollback's test.
def test_a_trip_in_flight_takes_the_dvl_out_until_the_operator_rearms():
    with tempfile.TemporaryDirectory() as tmp:
        # An 8 s agreement window, so the two seconds of wrong data below
        # have left it by the end of the 12 s that follow. (With the shipped
        # 60 s the mount stays un-provable for a minute after a DVL has
        # disagreed with the tags — which is the behaviour one wants.)
        TC, w, bus, _pilots, logs = _dvl_worker(tmp, selection="legacy", reentry_s=0.0,
                                                agree_window_s=8.0)
        sim = _Sim(w, TC)
        try:
            sim.prove()
            w.set_dvl_use(True)
            sim.engage()
            st = []
            bus.mpc_status.connect(st.append)
            for _ in range(10):
                sim.step(v=(0.10, 0.0, 0.0))
            assert st[-1].vel_src == "dvl"
            for _ in range(40):               # the DVL starts reading the
                sim.step(v=(0.10, 0.0, 0.0),  # reverse of the tags
                         dvl_v=(-0.10, 0.0, 0.0))
            assert w.dvl.tripped and w.engaged
            assert st[-1].vel_src == "tag_fd" and st[-1].dvl_state == "tripped"
            assert sum(1 for lvl, m in logs
                       if lvl == "error" and "TRIPPED" in m) == 1
            # the wrong-signed samples themselves never reached the loop:
            # the innovation gate held each one back, tick by tick
            rows_src = [s.vel_src for s in st[10:]]
            assert "dvl" not in rows_src, rows_src
            for k in range(240):              # honest again, and re-PROVEN
                tt = k * 0.05
                sim.step(v=(0.12 * math.sin(2 * math.pi * tt / 8.0),
                            0.08 * math.sin(2 * math.pi * (tt - 3.0) / 8.0),
                            0.0))
            assert w.dvl.proven and w.dvl.tripped
            assert st[-1].vel_src == "tag_fd"         # ...but still out
            m = w._run_meta()["dvl"]
            assert m["trips"] == 1 and m["tripped"]
            assert m["rows"]["fallback: DVL tripped"] > 200, m["rows"]
            assert not any(any(c.isdigit() for c in k) for k in m["rows"]), (
                m["rows"])
            w.set_engaged(False)
            sim.engage()                      # a new engagement does not
            assert w._dvl_run_ok is False     # re-arm it by itself
            assert "tripped" in w._dvl_run_why
            w.set_engaged(False)
            w.set_dvl_use(True)               # the operator does
            assert w.reason == "DVL IN (re-armed)" and not w.dvl.tripped
            sim.engage()
            assert w._dvl_run_ok is True
        finally:
            sim.close()


def test_the_heave_rate_never_comes_from_the_dvl():
    """The proof sees surge and sway. A DVL w that nothing has checked — a
    tilted bracket leaks surge into it, a wrong z convention flips it —
    stays out of nu[2]; it is recorded, not consumed."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, _bus, _pilots, _logs = _dvl_worker(tmp, reentry_s=0.0)
        sim = _Sim(w, TC)
        try:
            sim.prove()
            w.set_dvl_use(True)
            sim.engage()
            for _ in range(12):               # sinking 4 cm/s; the DVL
                sim.step(v=(0.05, 0.0, 0.04),  # claims 30 cm/s
                         dvl_v=(0.05, 0.0, 0.30))
            assert abs(w._last_nu[0] - 0.05) < 1e-9
            assert abs(w._last_nu[2] - 0.04) < 1e-3, w._last_nu
            rows = _rows(w)
            assert rows[-1]["vel_src"] == "dvl"
            assert abs(float(rows[-1]["nu_w"]) - float(rows[-1]["fd_w"])) < 1e-9
            assert abs(float(rows[-1]["dvl_w"]) - 0.30) < 1e-9
            assert "nu[2]" in w._run_meta()["dvl"]["consumes"]
        finally:
            sim.close()


def test_the_lever_arm_uses_the_body_rate_of_the_samples_own_moment():
    """A report describes a velocity ~0.2 s old. With the NEWEST rate in
    omega x r, a yaw that is speeding up leaves centimetres per second of
    false sway behind (review 2026-10-02)."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, _bus, _pilots, _logs = _dvl_worker(
            tmp, mount_t_flu=[0.2, 0.0, 0.0])
        t0 = 500.0
        for k in range(21):                   # yaw rate ramps 0 -> 0.4 rad/s
            w.on_vehicle_imu(TC._imu(r=0.02 * k, depth=0.8, t=t0 + 0.05 * k))
        assert w._omega_at(t0 + 0.5) == (0.0, 0.0, 0.2)
        assert abs(w._omega_at(t0 + 0.525)[2] - 0.21) < 1e-12
        assert w._omega_at(t0 - 5.0) is None and w._omega_at(t0 + 9.0) is None
        # the vehicle only YAWS: the DVL, 0.2 m ahead, sees pure sway r x 0.2.
        # A sample valid at t0+0.5 (rate 0.2) arrives at t0+0.7 (rate 0.28+)
        s = _sample((0.0, 0.2 * 0.2, 0.0), t=t0 + 0.5)
        s.t_rx = t0 + 0.7
        w.on_dvl(s)
        assert np.allclose(w.dvl.v_body, 0.0, atol=1e-12), w.dvl.v_body
        w.teardown()


# LEGACY-PINNED (2026-10-16, SPEC §10.2a): the shipped ladder = the rollback's test.
def test_the_log_never_ends_on_the_wrong_velocity_source():
    """The source-change line is rate-limited to one per 2 s; a change the
    limit swallowed is reported when the limit lifts, so the last line is
    never the opposite of what is flying."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, _bus, _pilots, logs = _dvl_worker(tmp, selection="legacy", reentry_s=0.0)
        sim = _Sim(w, TC)
        try:
            sim.prove()
            w.set_dvl_use(True)
            sim.engage()
            for _ in range(60):
                sim.step()
            for _ in range(5):
                sim.step(valid=False)         # 0.25 s without a lock
            for _ in range(60):
                sim.step()                    # back on the DVL for 3 s
            src = [m for _l, m in logs if "velocity source ->" in m]
            assert len(src) == 2, src
            assert "tag finite difference (DVL has no bottom lock)" in src[0]
            assert src[1].endswith("velocity source -> DVL"), src
        finally:
            sim.close()


# LEGACY-PINNED (2026-10-16, SPEC §10.2a): the shipped ladder = the rollback's test.
def test_a_bug_in_the_dvl_code_cannot_starve_the_control_tick():
    """If the DVL's own code ever raises on the tick path, the DVL is
    latched out and the loop flies on the tag finite difference — it must
    not abort tick() at 20 Hz with the vehicle engaged."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, pilots, logs = _dvl_worker(tmp, selection="legacy", reentry_s=0.0)
        sim = _Sim(w, TC)
        try:
            sim.prove()
            w.set_dvl_use(True)
            sim.engage()
            for _ in range(3):
                sim.step(v=(0.05, 0.0, 0.0))
            n0 = len(pilots)

            def boom(*_a, **_k):
                raise RuntimeError("synthetic DVL bug")
            w.dvl.velocity = boom
            st = []
            bus.mpc_status.connect(st.append)
            for _ in range(5):
                sim.step(v=(0.05, 0.0, 0.0))
            assert w.engaged and len(pilots) == n0 + 5, (w.reason, len(pilots))
            assert w._dvl_fault and "synthetic DVL bug" in w._dvl_fault
            assert all(s.vel_src == "tag_fd" for s in st)
            assert "CODE FAULT" in st[-1].dvl_note
            errs = [m for lvl, m in logs if lvl == "error" and "CODE FAULT" in m]
            assert len(errs) == 1, errs                  # said ONCE
            path = w._csv_path
            rows = _rows(w)
            # No short rows — read RAW: DictReader pads a short row with
            # None, so counting its keys can never fail (review 2026-10-02).
            import csv
            import rov_gui.control.workers as W
            with open(path, newline="") as fh:
                widths = {len(r) for r in csv.reader(fh)}
            assert widths == {len(W.CSV_HEADER.strip().split(","))}, widths
            assert [r["vel_src"] for r in rows[-5:]] == ["tag_fd"] * 5
        finally:
            sim.close()


def test_a_bug_in_the_dvl_code_cannot_starve_the_control_tick_under_primary():
    """N40 (2026-10-16), the primary half of the test above: the blend's
    entry points on the tick (usable, blend_step) and the bridge rung's
    (rung_velocity) raise — the DVL is latched out once, the loop flies on
    the tag finite difference with w 0, and no row is short."""
    import csv
    import rov_gui.control.workers as W

    def boom(*_a, **_k):
        raise RuntimeError("synthetic DVL bug")

    width = len(W.CSV_HEADER.strip().split(","))
    for name in ("usable", "blend_step"):
        with tempfile.TemporaryDirectory() as tmp:
            TC, w, bus, pilots, logs = _dvl_worker(tmp, reentry_s=0.0)
            sim = _Sim(w, TC)
            try:
                assert w.dvl.selection == "primary"
                sim.prove()
                w.set_dvl_use(True)
                sim.engage()
                for _ in range(3):
                    sim.step(v=(0.05, 0.0, 0.0))
                assert w._last_nu is not None and w._dvl_tick["w"] == 1.0
                n0 = len(pilots)
                setattr(w.dvl, name, boom)
                st = []
                bus.mpc_status.connect(st.append)
                for _ in range(5):
                    sim.step(v=(0.05, 0.0, 0.0))
                assert w.engaged and len(pilots) == n0 + 5, (name, w.reason)
                assert "synthetic DVL bug" in w._dvl_fault, name
                assert all(s.vel_src == "tag_fd" and s.dvl_w == 0.0
                           for s in st), name
                assert "CODE FAULT" in st[-1].dvl_note
                errs = [m for lvl, m in logs
                        if lvl == "error" and "CODE FAULT" in m]
                assert len(errs) == 1, errs
                path = w._csv_path
                rows = _rows(w)
                with open(path, newline="") as fh:
                    widths = {len(r) for r in csv.reader(fh)}
                assert widths == {width}, widths
                assert [(r["vel_src"], r["w_dvl"], r["dvl_code"])
                        for r in rows[-5:]] == [("tag_fd", "0.000", "fault")] * 5
                assert all(r["nu_u"] == r["fd_u"] for r in rows[-5:])
            finally:
                sim.close()
    with tempfile.TemporaryDirectory() as tmp:           # the rung's entry
        TC, w, bus, pilots, logs, sim = _station(tmp)
        try:
            assert w._dvl_xy is not None, "the rung should be open"
            w.dvl.rung_velocity = boom
            n0 = len(pilots)
            w.fix.t_capture -= 5.0            # the tags are gone
            for _ in range(5):
                sim.step(fix=False)
            assert w.engaged and len(pilots) == n0 + 5, w.reason
            assert "synthetic DVL bug" in w._dvl_fault
            assert w._bridge.tier != SB.TIER_DVL
            path = w._csv_path
            rows = _rows(w)
            with open(path, newline="") as fh:
                widths = {len(r) for r in csv.reader(fh)}
            assert widths == {width}, widths
            assert all(r["w_dvl"] == "0.000" and r["dvl_code"] == "bridge"
                       for r in rows[-5:]), [(r["w_dvl"], r["dvl_code"])
                                             for r in rows[-5:]]
        finally:
            sim.close()


def test_a_tag_jump_does_not_un_prove_a_good_mount():
    """One mis-solved frame is a finite difference of metres per second;
    the fit must set it aside rather than let it outvote the run."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, _bus, _pilots, _logs = _dvl_worker(tmp)
        sim = _Sim(w, TC)
        try:
            sim.prove()
            for k in range(60):
                if k % 20 == 5:
                    sim.p = sim.p + np.array([0.04, -0.03, 0.0])   # a jump
                sim.step(v=(0.10 * math.sin(0.2 * k), 0.06, 0.0), tick=False)
            assert w.dvl.proven and w.dvl.n_trips == 0, w.dvl.agree.fit()
            assert w.dvl.agree.fit()["n_outliers"] >= 1
        finally:
            sim.close()


def test_a_yaw_in_both_the_device_and_the_config_is_called_out():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, _bus, _pilots, logs = _dvl_worker(
            tmp, mount_rpy_deg=[0.0, 0.0, 90.0])
        w.on_dvl_info({"mounting_rotation_offset": 90.0, "range_mode": "auto"})
        assert any("TWICE" in m for lvl, m in logs if lvl == "error"), logs
        assert w._run_meta()["dvl"]["device_config"]["range_mode"] == "auto"
        w.teardown()


def test_the_demo_mount_is_the_identity_and_the_real_one_must_be_measured():
    TC = _harness()
    with tempfile.TemporaryDirectory() as tmp:
        w, _bus, _pilots, logs = TC._worker(tmp)      # Opts.source == "demo"
        w.cfg.dvl = {"enabled": True}
        w._setup_dvl()
        assert w.dvl.mount_measured
        assert any("IDENTITY mount" in m for _l, m in logs)
        w.opts.source = "hw"
        w.cfg.dvl = {"enabled": True}
        w._setup_dvl()
        assert not w.dvl.mount_measured
        w.on_dvl(_sample(t=1.0))
        w.set_dvl_use(True)
        assert not w.dvl.use and "mount not measured" in w.reason
        w.teardown()


# ===================================================================== client
class _FakeDvl:
    """A one-connection-at-a-time TCP server that plays the vendor capture."""

    def __init__(self, payload: bytes, hold_s: float = 0.3):
        self.payload = payload
        self.hold_s = float(hold_s)
        self.got = b""
        self.srv = socket.socket()
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        self.stop = threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        self.srv.settimeout(0.2)
        while not self.stop.is_set():
            try:
                c, _ = self.srv.accept()
            except OSError:
                continue
            with c:
                c.settimeout(0.05)
                try:
                    self.got += c.recv(4096)
                except OSError:
                    pass
                # torn deliberately: 100-byte chunks split lines mid-JSON
                for i in range(0, len(self.payload), 100):
                    try:
                        c.sendall(self.payload[i:i + 100])
                    except OSError:
                        break
                c.sendall(b'{"response_to":"get_config","success":true,'
                          b'"error_message":"","result":{"speed_of_sound":'
                          b'1475,"mounting_rotation_offset":0,"range_mode":'
                          b'"auto"},"format":"json_v3.3","type":"response"}'
                          b'\r\n')
                t_end = time.monotonic() + self.hold_s
                while time.monotonic() < t_end and not self.stop.is_set():
                    try:                      # keep listening: the client
                        self.got += c.recv(4096)   # may ask again
                    except OSError:
                        pass

    def close(self):
        self.stop.set()
        self.th.join(2.0)
        self.srv.close()


def test_the_client_reads_torn_lines_asks_config_once_and_reconnects():
    from rov_gui.backends.dvl import DvlClient

    srv = _FakeDvl(CAPTURE.read_bytes())
    c = DvlClient("127.0.0.1", srv.port, connect_timeout_s=1.0,
                  reconnect_s=0.1)
    c.start()
    got = []
    t_end = time.monotonic() + 6.0
    while time.monotonic() < t_end and (len(got) < 400 or c.n_connects < 2):
        got += c.drain()
        time.sleep(0.02)
    c.stop()
    srv.close()
    kinds = [rep["type"] for _t, rep, _raw, _c in got]
    assert kinds.count("velocity") >= 120, len(kinds)     # two full plays
    assert "response" in kinds
    assert c.n_connects >= 2 and c.n_bad == 0, (c.n_connects, c.n_bad)
    assert srv.got.count(b'"get_config"') >= 2
    # arrival stamps never go backwards, and each line says which connection
    ts = [t for t, _r, _raw, _c in got]
    assert ts == sorted(ts)
    assert {cid for _t, _r, _raw, cid in got} >= {1, 2}
    assert c.state is Conn.OFFLINE                        # after stop()


def test_the_client_asks_again_on_a_period_and_outlives_a_bad_line():
    """get_config is re-asked every config_poll_s (the device's own rotation
    can be changed under a live connection), and a line the parser chokes on
    ends nothing: not the reader thread, not the rest of the drain."""
    from rov_gui.backends import dvl as BD

    srv = _FakeDvl(b'{"type":"velocity","x":' + b"[" * 30000 + b"]" * 30000
                   + b"}\r\n" + CAPTURE.read_bytes(), hold_s=1.2)
    c = BD.DvlClient("127.0.0.1", srv.port, connect_timeout_s=1.0,
                     reconnect_s=0.1, config_poll_s=0.3)
    c.start()
    got = []
    t_end = time.monotonic() + 1.4
    while time.monotonic() < t_end:
        got += c.drain()
        time.sleep(0.02)
    alive = c._thread is not None and c._thread.is_alive()
    c.stop()
    srv.close()
    assert alive and 1 <= c.n_bad <= c.n_connects, (alive, c.n_bad)
    assert sum(1 for g in got if g[1]["type"] == "velocity") >= 60
    assert srv.got.count(b'"get_config"') >= 3, srv.got.count(b'"get_config"')

    # the WORKER: one report that blows up costs that report only
    TC = _harness()
    TC._app()
    from rov_gui.bus import DataBus

    class _Stub:
        host, port, reconnect_s, error, config_poll_s = "x", 1, 1.0, "", 0.0
        state = Conn.ONLINE

        def __init__(self, items):
            self.items = items

        def start(self):
            pass

        def stop(self):
            pass

        def drain(self):
            out, self.items = self.items, []
            return out

    class _Boom(dict):
        def get(self, *a, **k):
            raise RuntimeError("synthetic")

    good = DV.parse_line(CAPTURE.read_bytes().split(b"\n")[1])
    bus = DataBus()
    out, logs = [], []
    bus.dvl.connect(out.append)
    bus.log.connect(lambda lvl, m: logs.append((lvl, m)))
    w = BD.DvlWorker(bus, object(), client=_Stub(
        [(1.0, good, b"", 1), (1.1, _Boom(), b"", 1), (1.2, good, b"", 1)]))
    w.setup()
    w.tick()
    assert len(out) == 2 and w.n_bad_reports == 1
    assert sum(1 for lvl, m in logs if "could not be handled" in m) == 1


def test_a_dead_address_is_a_reported_state_not_an_exception():
    from rov_gui.backends.dvl import DvlClient

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()                                 # nothing listens here now
    c = DvlClient("127.0.0.1", port, connect_timeout_s=0.3, reconnect_s=0.1)
    c.start()
    time.sleep(0.6)
    assert c.drain() == [] and c.error and c.n_connects == 0
    c.stop()


def _wait(cond, timeout_s: float, what: str = "") -> None:
    t_end = time.monotonic() + timeout_s
    while time.monotonic() < t_end:
        if cond():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what or cond}")


class _FakeStatusPage:
    """The DVL's web server, as far as /api/v1/about/status goes. Records
    every request's METHOD, path and headers, whatever the method."""

    def __init__(self, body: bytes = b'{"cpu_load":1.64,"disk_free":{"data":'
                                      b'2.46,"root":0.211},"temperature":38.25}'):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        page = self
        self.reply = (200, body)
        self.requests: list = []

        class H(BaseHTTPRequestHandler):
            def parse_request(self):
                ok = super().parse_request()
                if ok:
                    page.requests.append((self.command, self.path,
                                          dict(self.headers)))
                return ok

            def do_GET(self):
                status, data = page.reply
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_a):
                pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        # the oversized page is hung up on mid-write, deliberately
        self.srv.handle_error = lambda _request, _address: None
        self.port = self.srv.server_address[1]
        self.th = threading.Thread(target=self.srv.serve_forever,
                                   kwargs={"poll_interval": 0.05}, daemon=True)
        self.th.start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.th.join(2.0)


class _RawPage:
    """A TCP server that accepts and then says nothing (``"mute"``), or
    starts an answer and drips one header byte every 50 ms for ever
    (``"drip"`` — every recv is inside any socket timeout)."""

    def __init__(self, mode: str):
        self.mode = mode
        self.stop = threading.Event()
        self.srv = socket.socket()
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(4)
        self.srv.settimeout(0.1)
        self.port = self.srv.getsockname()[1]
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        while not self.stop.is_set():
            try:
                c, _ = self.srv.accept()
            except OSError:
                continue
            threading.Thread(target=self._serve, args=(c,), daemon=True).start()

    def _serve(self, c):
        try:
            if self.mode == "drip":
                c.sendall(b"HTTP/1.1 200 OK\r\nX-Drip: ")
                while not self.stop.wait(0.05):
                    c.sendall(b"a")
            else:
                self.stop.wait()
        except OSError:
            pass
        finally:
            c.close()

    def close(self):
        self.stop.set()
        self.th.join(2.0)
        self.srv.close()


def test_the_temperature_is_a_get_of_the_status_page_and_nothing_else():
    """The number the vendor GUI shows, from the page it shows it from. A
    page that is missing, not JSON or not a temperature is a reported state:
    the thread lives, and the last good reading keeps ITS stamp (the worker
    stops showing it when it is too old — never a frozen number). A server
    that never answers is a timeout; one that keeps a request open by
    dripping bytes is reported STUCK, since no socket timeout ever fires."""
    from rov_gui.backends import dvl as BD

    srv = _FakeStatusPage()
    p = None
    try:
        p = BD.DvlTempPoller("127.0.0.1", port=srv.port, period_s=0.03,
                             timeout_s=0.5)
        p.start()
        _wait(lambda: p.latest()[0] is not None, 3.0, "a first reading")
        assert p.latest()[0] == 38.25 and p.latest()[2] == ""
        assert p.last_try()[1] is True
        pages = p.drain()
        assert pages and pages[0][1]["cpu_load"] == 1.64     # verbatim page
        # From here no page is good, so the last good stamp can no longer
        # move under the test.
        srv.reply = (404, b'{"code":"not_found"}')
        n = p.n_fail
        _wait(lambda: p.n_fail >= n + 2, 3.0, "HTTP 404")    # a whole poll
        temp, t_good, err = p.latest()
        assert temp == 38.25 and "HTTP 404" in err
        assert p.last_try()[1] is True        # a 404 is an ANSWER: it is up
        for status, body, needle in (
                (200, b"<html>", "JSONDecodeError"),
                (200, b"[38.0]", "not an object"),
                (200, b'{"cpu_load":1.0}', "not a temperature"),
                (200, b'{"temperature":true}', "not a temperature"),
                (200, b'{"temperature":NaN}', "not a temperature"),
                (200, b'{"temperature":38250}', "not a temperature"),  # m°C
                (200, b"{" + b" " * 20000 + b"}", "longer than")):
            srv.reply = (status, body)
            n = p.n_fail
            _wait(lambda: p.n_fail >= n + 2, 3.0, needle)
            temp, t_kept, err = p.latest()
            assert needle in err, (needle, err)
            assert temp == 38.25 and t_kept == t_good, (temp, t_kept)
            assert p.last_try()[1] is True, needle        # answered, uselessly
            assert p._thread is not None and p._thread.is_alive()
        srv.reply = (200, b'{"temperature":41.5}')
        _wait(lambda: p.latest()[0] == 41.5, 3.0, "recovery")
        assert p.latest()[2] == "" and p.last_try()[1] is True
        # poke(): now, not at the end of the period
        p.period_s = 30.0
        time.sleep(0.2)                       # the 30 s wait has begun
        n = len(srv.requests)
        p.poke()
        _wait(lambda: len(srv.requests) > n, 1.0, "a poked poll")
        th = p._thread
        t0 = time.monotonic()
        p.stop()
        assert time.monotonic() - t0 < 1.0 and not th.is_alive()
    finally:
        if p is not None:
            p.stop()
        srv.close()
    # A read and nothing else: GET, of that one page, every time — framed
    # like the urllib probe that read the real A50 (Connection: close).
    assert srv.requests
    assert {m for m, _path, _h in srv.requests} == {"GET"}, srv.requests
    assert {path for _m, path, _h in srv.requests} == {BD.STATUS_PATH}
    assert all(h.get("Connection") == "close" for _m, _p, h in srv.requests)

    mute, drip = _RawPage("mute"), _RawPage("drip")
    q = None
    try:
        q = BD.DvlTempPoller("127.0.0.1", port=mute.port, period_s=0.05,
                             timeout_s=0.3)
        q.start()
        _wait(lambda: q.n_fail >= 1, 3.0, "a timeout")
        assert "timed out" in q.latest()[2], q.latest()
        assert q.last_try()[1] is False                   # no answer at all
        q.stop()
        q = BD.DvlTempPoller("127.0.0.1", port=drip.port, period_s=0.05,
                             timeout_s=0.3)
        q.start()
        _wait(lambda: "stuck" in q.latest()[2], 3.0, "a stuck request")
        assert q.n_fail == 0 and q.last_try() == (None, False)
    finally:
        if q is not None:
            q.stop()
        mute.close()
        drip.close()

    # nothing listening: a state, not an exception
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    dead = s.getsockname()[1]
    s.close()
    q = BD.DvlTempPoller("127.0.0.1", port=dead, period_s=0.05, timeout_s=0.3)
    try:
        q.start()
        _wait(lambda: q.n_fail >= 1, 3.0, "a refused GET")
    finally:
        q.stop()
    assert q.latest()[0] is None and "Refused" in q.latest()[2], q.latest()
    assert q.last_try()[1] is False


class _TempRig:
    """A DvlWorker on a stub TCP client and a stub poller, on a FAKE clock
    (``backends.dvl.now`` patched, put back by ``close()``), so the 0.5 s
    stat, the 2 s link, the 20 s staleness and BIT0_CLEAR_S are steps, not
    sleeps."""

    _KEEP = object()

    def __init__(self):
        from rov_gui.backends import dvl as BD
        from rov_gui.bus import DataBus

        TC = _harness()
        TC._app()
        self.BD = BD
        self.t = 1000.0
        self._real_now = BD.now
        BD.now = lambda: self.t

        class Client:
            host, port, reconnect_s, error, config_poll_s = "x", 1, 1.0, "", 0.0
            state = Conn.ONLINE

            def __init__(self):
                self.items = []

            def start(self):
                pass

            def stop(self):
                pass

            def drain(self):
                out, self.items = self.items, []
                return out

        class Temp:
            host, period_s = "192.0.2.1", 5.0

            def __init__(self):
                self.reading = (None, None, "")
                self.tried = (None, False)
                self.pages = []
                self.n_fail = 0
                self.pokes = 0
                self.stopped = False

            def start(self):
                pass

            def stop(self):
                self.stopped = True

            def poke(self):
                self.pokes += 1

            def latest(self):
                return self.reading

            def last_try(self):
                return self.tried

            def drain(self):
                out, self.pages = self.pages, []
                return out

        lines = [DV.parse_line(ln) for ln in CAPTURE.read_bytes().split(b"\n")]
        self.lock = next(r for r in lines if r and r.get("type") == "velocity"
                         and r.get("velocity_valid"))
        self.no_lock = dict(self.lock, velocity_valid=False, altitude=-1.0)
        self.stats, self.logs = [], []
        bus = DataBus()
        bus.sensor_stat.connect(self.stats.append)
        bus.log.connect(lambda lvl, m: self.logs.append((lvl, m)))
        self.cl, self.tp = Client(), Temp()
        self.w = BD.DvlWorker(bus, object(), client=self.cl, temp=self.tp)
        self.w.setup()

    def row(self, rep=None, temp=_KEEP, dt=0.1):
        """Advance the clock by ``dt``, deliver ``rep`` (if any) and a fresh
        successful reading (if ``temp`` is given), publish one stat."""
        self.t += dt
        if rep is not None:
            self.cl.items.append((self.t, rep, b"", 1))
        if temp is not self._KEEP:
            self.tp.reading = (temp, self.t, "")
            self.tp.tried = (self.t, True)
        self.w._t_stat = -1e9
        self.w.tick()
        return self.stats[-1]

    def page_fails(self, err="TimeoutError: timed out", dt=0.1,
                   answered=False):
        """A request completed ``dt`` after now and failed — with no answer
        at all, or (``answered``) with a useless one; the reading stays."""
        self.tp.tried = (self.t + dt, answered)
        self.tp.reading = (self.tp.reading[0], self.tp.reading[1], err)

    def said(self, word):
        return [m for _lvl, m in self.logs if word in m]

    def close(self):
        self.BD.now = self._real_now
        self.w.teardown()


def test_the_sensors_row_says_the_temperature_before_the_lock():
    """Out of the water is the only place this DVL overheats, and there it
    never has a lock: HOT / HIGH TEMP must win over "no lock" (until
    2026-10-13 bit 0 was checked after it, out of sight) — and bit 0 with NO
    number (an older firmware's 404) too. HOT is the 45/44 °C band in the
    row and the log alike; bit 0 is held until clear for BIT0_CLEAR_S; each
    change is logged once; the rate leads every detail that carries a °C,
    or SensorRow would show the bare rate instead."""
    from rov_gui.widgets.health import SensorRow

    r = _TempRig()
    try:
        BD, lock, no_lock = r.BD, r.lock, r.no_lock
        assert any("temperature GET http://192.0.2.1/api/v1/about/status" in m
                   for _l, m in r.logs)
        r.row(lock)
        s = r.row(lock)                                  # two reports = a rate
        assert s.conn is Conn.ONLINE and s.detail.startswith("10 Hz alt")
        assert "°C" not in s.detail and not r.said("dvl: temperature ")
        s = r.row(lock, temp=21.4)
        assert s.conn is Conn.ONLINE and s.detail.startswith("10 Hz alt")
        assert s.detail.endswith(" 21.4°C"), s.detail
        s = r.row(no_lock, temp=38.2)                    # in air, still cool
        assert s.conn is Conn.DEGRADED and s.detail == "10 Hz no lock 38.2°C"
        s = r.row(no_lock, temp=46.3)                    # in air, HOT: wins
        assert s.conn is Conn.DEGRADED and s.detail == "10 Hz HOT 46.3°C"
        s = r.row(no_lock, temp=44.5)                    # inside the band
        assert s.detail == "10 Hz HOT 44.5°C" and not r.said("back to"), s
        s = r.row(no_lock, temp=43.9)
        assert s.detail == "10 Hz no lock 43.9°C", s.detail
        assert len(r.said("back to 43.9")) == 1
        s = r.row(lock, temp=46.5)                       # a lock changes nothing
        assert s.detail == "10 Hz HOT 46.5°C", s.detail
        hot = r.said("dvl: HOT")
        assert len(hot) == 2 and "acoustics OFF" in hot[0]
        assert len(r.said("°C (HOT at")) == 1            # once, not per stat
        # the device's own flag with NO number — an older firmware's 404
        r.tp.reading = (None, None, "RuntimeError: HTTP 404 for /api/v1/about")
        r.tp.n_fail = 1
        s = r.row(dict(no_lock, status=1))
        assert s.conn is Conn.DEGRADED and s.detail == "10 Hz HIGH TEMP", s
        assert len(r.said("no temperature (RuntimeError: HTTP 404")) == 1
        s = r.row(dict(no_lock, status=1), temp=40.0)   # ...or a cool one
        assert s.detail == "10 Hz HIGH TEMP 40.0°C", s.detail
        # held: a flag flickering at its threshold is ONE warning
        s = r.row(no_lock, temp=40.0)
        assert s.detail == "10 Hz HIGH TEMP 40.0°C", s.detail
        r.row(dict(no_lock, status=1), temp=40.0)
        s = r.row(no_lock, temp=40.0, dt=BD.BIT0_CLEAR_S - 1.0)
        assert "HIGH TEMP" in s.detail and not r.said("flag cleared")
        s = r.row(no_lock, temp=40.0, dt=1.5)
        assert s.detail.endswith("no lock 40.0°C"), s.detail
        assert len(r.said("status bit 0")) == 1
        assert len(r.said("flag cleared")) == 1
        # a valid report with no altitude still leads with the rate...
        r.row(lock, temp=30.0)
        s = r.row(dict(lock, altitude=None), temp=30.0)
        assert s.conn is Conn.ONLINE and s.detail.endswith(" Hz 30.0°C"), s
        assert "alt" not in s.detail
        # ...because the row shows a detail only if it carries "Hz"
        for st in (s, r.row(no_lock, temp=46.0)):
            sr = SensorRow("DVL")
            sr.update_from(st)
            assert sr.value.text() == st.detail and "°C" in sr.value.text()
        # a reading that stopped updating is not shown — said once, even
        # when no request ever FAILED (a hung one never does)
        r.tp.n_fail = 0
        r.tp.reading = (46.0, r.t - BD.TEMP_STALE_S - 1.0,
                        "status page request stuck 9 s")
        s = r.row(lock)
        assert "°C" not in s.detail and "HOT" not in s.detail, s.detail
        r.row(lock)
        assert len(r.said("no temperature (status page request stuck")) == 1
        # the pages go into the run's raw log, beside the reports
        with tempfile.TemporaryDirectory() as tmp:
            stem = str(Path(tmp) / "run")
            r.w.set_raw_log(True, stem)
            try:
                r.tp.pages = [(r.t, {"cpu_load": 1.2, "temperature": 43.1})]
                r.row(lock)
            finally:
                r.w.set_raw_log(False, "")
            recs = [json.loads(x) for x in
                    Path(stem + "_dvl.jsonl").read_text().splitlines()]
            pages = [x for x in recs if x["msg"] == "about_status"]
            assert len(pages) == 1 and pages[0]["temperature"] == 43.1
            assert any(x["msg"] == "velocity" for x in recs)
    finally:
        r.close()
    assert r.tp.stopped


def test_a_link_lost_while_hot_is_a_shutdown_only_once_the_page_fails_too():
    """The vendor restarts a DVL that reached 55 °C, so a velocity stream
    that stops while it was hot MAY be a thermal shutdown — or the operator
    turning acoustics OFF, as the station's own HOT line tells them to, or a
    TCP hiccup. The call is made only when the DVL was hot at the last
    linked stat (its own bit 0, or a reading at TEMP_NEAR_SHUTDOWN_C) AND a
    status-page request completed SINCE got no answer at all; any answer —
    even an older firmware's 404 — withdraws it. Logged once per outage,
    with no frozen number on the row."""
    r = _TempRig()
    try:
        BD, lock, no_lock = r.BD, r.lock, r.no_lock
        r.row(lock, temp=40.0)
        r.row(dict(no_lock, status=1), temp=40.0)        # hot: its bit 0
        # the stream stops; no request has completed since: nothing yet
        r.cl.state, r.cl.error = Conn.OFFLINE, "silent for 5 s"
        s = r.row(dt=2.5)
        assert s.conn is Conn.OFFLINE and s.detail == "no link 40.0°C", s
        assert r.tp.pokes == 1 and not r.said("SHUTDOWN")  # asked at once
        r.row(dt=0.5)
        assert r.tp.pokes == 1                             # once per loss
        # the page answers (acoustics OFF, a hiccup): the device is up
        r.tp.tried, r.tp.reading = (r.t + 0.1, True), (40.2, r.t + 0.1, "")
        s = r.row(dt=0.5)
        assert s.conn is Conn.OFFLINE and not r.said("SHUTDOWN"), s
        # ...then it stops answering too: now it is a likely thermal shutdown
        r.page_fails()
        s = r.row(dt=0.5)
        assert s.conn is Conn.FAULT and s.detail == "THERMAL SHUTDOWN?", s
        e = r.said("LIKELY THERMAL SHUTDOWN")
        assert len(e) == 1 and "its bit 0" in e[0], e
        r.row(dt=0.5)
        assert len(r.said("LIKELY THERMAL SHUTDOWN")) == 1   # once per outage
        # the page comes back: withdrawn (once); fails again: FAULT, no new
        # error line; the row stays bounded, the error text is in the log
        r.tp.tried, r.tp.reading = (r.t + 0.1, True), (52.0, r.t + 0.1, "")
        s = r.row(dt=0.5)
        assert s.conn is Conn.DEGRADED and s.detail == "no link HOT 52.0°C", s
        assert len(r.said("answers again")) == 1
        r.page_fails()
        s = r.row(dt=0.5)
        assert s.conn is Conn.FAULT
        assert len(r.said("LIKELY THERMAL SHUTDOWN")) == 1
        # back, and cooled: linked rows again, each change said once
        r.cl.state, r.cl.error = Conn.ONLINE, ""
        r.row(lock, temp=43.0, dt=BD.BIT0_CLEAR_S)
        s = r.row(lock, temp=43.0)
        assert s.conn is Conn.ONLINE and s.detail.endswith("43.0°C"), s
        assert r.said("back to 43.0") and r.said("flag cleared")
        # NOT hot at the last linked stat: the plain "no link", however
        # hot it was before and whatever the page does
        r.cl.state, r.cl.error = Conn.OFFLINE, "closed by the DVL"
        r.page_fails()
        s = r.row(dt=2.5)
        assert s.conn is Conn.OFFLINE and s.detail == "no link 43.0°C", s
        r.tp.tried, r.tp.reading = (r.t + 0.1, True), (47.0, r.t + 0.1, "")
        s = r.row(dt=0.5)
        assert s.conn is Conn.DEGRADED and s.detail == "no link HOT 47.0°C"
        assert r.tp.pokes == 1 and len(r.said("LIKELY THERMAL")) == 1
        # a reading at TEMP_NEAR_SHUTDOWN_C is hot-at-the-link without bit 0
        r.cl.state, r.cl.error = Conn.ONLINE, ""
        r.row(no_lock, temp=BD.TEMP_NEAR_SHUTDOWN_C + 0.5)
        r.row(no_lock, temp=BD.TEMP_NEAR_SHUTDOWN_C + 0.5)
        r.cl.state, r.cl.error = Conn.OFFLINE, "silent for 5 s"
        r.row(dt=2.5)
        r.page_fails()
        s = r.row(dt=0.5)
        assert s.conn is Conn.FAULT and r.tp.pokes == 2
        e = r.said("LIKELY THERMAL SHUTDOWN")
        assert len(e) == 2 and f"{BD.TEMP_NEAR_SHUTDOWN_C + 0.5:.1f} °C" in e[1]
        # ANY answer says the device is up: a 404 after the loss is no call
        hot = BD.TEMP_NEAR_SHUTDOWN_C + 0.5
        r.cl.state, r.cl.error = Conn.ONLINE, ""
        r.row(no_lock, temp=hot)
        r.row(no_lock, temp=hot)
        r.cl.state, r.cl.error = Conn.OFFLINE, "silent for 5 s"
        r.page_fails(err="RuntimeError: HTTP 404 for /api/v1/about/status",
                     answered=True)
        s = r.row(dt=2.5)
        assert s.conn is not Conn.FAULT, s
        # a page already silent BEFORE the loss decides nothing until a
        # request made AFTER it comes back silent too
        r.cl.state, r.cl.error = Conn.ONLINE, ""
        r.row(no_lock, temp=hot)
        r.page_fails(dt=0.0)
        r.row(no_lock)                                   # linked, page silent
        r.cl.state, r.cl.error = Conn.OFFLINE, "silent for 5 s"
        s = r.row(dt=2.5)
        assert s.conn is not Conn.FAULT, s
        r.page_fails()
        s = r.row(dt=0.5)
        assert s.conn is Conn.FAULT and len(r.said("LIKELY THERMAL")) == 3
    finally:
        r.close()


def test_a_worker_that_builds_its_client_builds_the_poller_beside_it():
    """Same host as the TCP stream (one place says where the DVL is); an
    injected client brings its own poller or none."""
    from rov_gui.backends import dvl as BD
    from rov_gui.bus import DataBus

    class O:
        mpc_config = str(ROOT / "config" / "hw_mpc.yaml")
        dvl = True
        dvl_host = "127.0.0.1"
        dvl_port = 9

    started = []
    real = (BD.DvlClient.start, BD.DvlTempPoller.start)
    BD.DvlClient.start = lambda self: started.append("tcp")
    BD.DvlTempPoller.start = lambda self: started.append("temp")
    try:
        w = BD.DvlWorker(DataBus(), O())
        w.setup()
        assert isinstance(w.temp, BD.DvlTempPoller)
        assert w.temp.host == "127.0.0.1" and w.temp.port == 80
        assert started == ["tcp", "temp"]
    finally:
        BD.DvlClient.start, BD.DvlTempPoller.start = real
    w.teardown()


def test_the_panel_button_reflects_the_worker_and_never_flips_itself():
    TC = _harness()
    TC._app()
    from rov_gui.widgets.trajectory import TrajectoryWindow

    p = TrajectoryWindow()
    asked = []
    p.dvl_use_requested.connect(asked.append)
    p.add_status(MpcStatus())                 # a station with no DVL
    assert p.btn_dvl.isHidden()
    p.add_status(MpcStatus(dvl_state="shadow", dvl_use=False,
                           dvl_note="not proven yet"))
    assert not p.btn_dvl.isHidden() and p.btn_dvl.text() == "DVL OUT"
    assert p.btn_dvl.isEnabled() and not p.btn_dvl.isChecked()
    p.btn_dvl.click()                         # a REQUEST...
    assert asked == [True]
    assert not p.btn_dvl.isChecked() and p.btn_dvl.text() == "DVL OUT"
    p.add_status(MpcStatus(dvl_state="in_use", dvl_use=True, dvl_proven=True,
                           vel_src="dvl"))    # ...the worker agreed
    assert p.btn_dvl.isChecked() and p.btn_dvl.text() == "DVL IN"
    assert "in use" in p.btn_dvl.toolTip()
    # frozen while engaged, and while a controller CSV is open
    p.add_status(MpcStatus(engaged=True, dvl_state="in_use", dvl_use=True,
                           vel_src="dvl", mode="pid"))
    assert not p.btn_dvl.isEnabled()
    assert "v:DVL" in p.chip.text()
    p.add_status(MpcStatus(engaged=True, dvl_state="no_lock", dvl_use=True,
                           vel_src="tag_fd", mode="pid"))
    assert "v:TAG" in p.chip.text() and "NO LOCK" in p.chip.text()
    p.add_status(MpcStatus(csv_open=True, dvl_state="shadow"))
    assert not p.btn_dvl.isEnabled()
    p.add_status(MpcStatus(engaged=True, mode="pid", bridge_tier="dvl",
                           dvl_state="in_use", dvl_use=True, vel_src="dvl"))
    assert "NO TAG (DVL)" in p.chip.text()


def _pixels(btn):
    """A rendered button: (its pixels as an [h][w] grid of RGB, background)
    — the background being the most common colour inside the border."""
    from collections import Counter

    from rov_gui.qt import QtGui

    img = btn.grab().toImage()
    grid = [[QtGui.QColor(img.pixel(x, y)).getRgb()[:3]
             for x in range(img.width())] for y in range(img.height())]
    inner = [c for row in grid[4:-4] for c in row[4:-4]]
    return grid, Counter(inner).most_common(1)[0][0]


def _contrast(a, b):
    def lum(c):
        def ch(v):
            v /= 255.0
            return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
        return 0.2126 * ch(c[0]) + 0.7152 * ch(c[1]) + 0.0722 * ch(c[2])
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _face(btn):
    """(background, text, WCAG contrast): the interior pixel furthest from
    the background is the text."""
    grid, bg = _pixels(btn)
    inner = [c for row in grid[4:-4] for c in row[4:-4]]
    fg = max(inner, key=lambda c: _contrast(c, bg))
    return bg, fg, _contrast(bg, fg)


def _ink_width(btn) -> int:
    """How many columns between the border carry label ink."""
    grid, bg = _pixels(btn)
    cols = [x for x in range(3, len(grid[0]) - 3)
            if any(_contrast(row[x], bg) >= 3.0 for row in grid[4:-4])]
    return 0 if not cols else cols[-1] - cols[0] + 1


def _clipped(btn) -> bool:
    """True when the label is cut: the same face drawn 80 px wider shows
    more ink. (The app sheet makes checked text bold while the button is
    sized for its regular label — "DVL TRIPPED" read ")VL TRIPPEI".)"""
    from rov_gui.qt import QtWidgets

    twin = QtWidgets.QPushButton(btn.text())
    twin.setCheckable(btn.isCheckable())
    twin.setChecked(btn.isChecked())
    twin.setEnabled(btn.isEnabled())
    twin.setStyleSheet(btn.styleSheet())
    twin.setFixedSize(btn.width() + 80, btn.height())
    twin.setDown(btn.isDown())
    try:
        return _ink_width(btn) + 1 < _ink_width(twin)
    finally:
        twin.deleteLater()


def test_an_in_switch_is_readable_in_every_state():
    """Operator screenshot 2026-10-13: "DVL IN" in amber on the app sheet's
    light-blue checked fill, ~1:1 [유도: WCAG on theme.WARN vs theme.ACCENT;
    1.07-1.87:1 rendered on the old faces, 측정: data/20261013/
    1013_dvl_gui_render/render_dvl_ui_out.txt] — the widget sheet's colour beat
    the app sheet's checked text. Rendered with the REAL app sheet, every
    face of both state switches — checked, frozen, unchecked, pressed —
    reads at 4.5:1 (WCAG AA); a checked one is FILLED with its state colour,
    a frozen one keeps the dark face with the state in text and border; and
    no label is cut."""
    from rov_gui import theme
    from rov_gui.widgets.trajectory import TrajectoryWindow

    TC = _harness()
    app = TC._app()
    prev = app.styleSheet()
    theme.apply(app)                       # the :checked fill IS the bug
    p = None

    def rgb(hex_):
        return tuple(int(hex_[i:i + 2], 16) for i in (1, 3, 5))

    def check(btn, kw, fill=None, frozen=None, down=False):
        btn.setDown(down)
        app.processEvents()
        try:
            bg, fg, cr = _face(btn)
            assert cr >= 4.5, (kw, down, bg, fg, round(cr, 2))
            if fill is not None:
                assert btn.isChecked() and bg == rgb(fill), (kw, bg)
            if frozen is not None:
                assert not btn.isEnabled() and bg == rgb(theme.PANEL), (kw, bg)
                assert fg == rgb(frozen), (kw, fg)
            if down:
                assert bg == rgb(theme.ACCENT), (kw, bg)
            assert not _clipped(btn), (kw, down, btn.text())
        finally:
            btn.setDown(False)

    try:
        p = TrajectoryWindow()
        p.show()
        app.processEvents()
        for kw, fill, frozen, down in (
                (dict(dvl_state="fallback", dvl_use=True), theme.WARN, None,
                 False),
                (dict(dvl_state="in_use", dvl_use=True, vel_src="dvl"),
                 theme.OK, None, False),
                (dict(dvl_state="tripped", dvl_use=True), theme.FAIL, None,
                 False),
                (dict(dvl_state="tripped", dvl_use=False), None, None, False),
                (dict(dvl_state="tripped", dvl_use=False), None, None, True),
                (dict(dvl_state="shadow", dvl_use=False), None, None, False),
                (dict(dvl_state="shadow", dvl_use=False), None, None, True),
                (dict(dvl_state="in_use", dvl_use=True, vel_src="dvl",
                      engaged=True, mode="pid"), None, theme.OK, False),
                (dict(dvl_state="fallback", dvl_use=True, csv_open=True),
                 None, theme.WARN, False),
                # 2026-10-16 (selection primary): the self-releasing hold,
                # a run not armed, a tick without a tag state — all "DVL IN"
                # on the WARN face, never TRIPPED
                (dict(dvl_state="held", dvl_use=True, dvl_code="dis_diff",
                      dvl_w=0.0, dvl_target=0.0), theme.WARN, None, False),
                (dict(dvl_state="not_armed", dvl_use=True,
                      dvl_code="not_armed", dvl_w=0.0, dvl_target=0.0),
                 theme.WARN, None, False),
                (dict(dvl_state="no_tag", dvl_use=True, dvl_code="no_tag"),
                 theme.WARN, None, False),
                (dict(dvl_state="held", dvl_use=True, dvl_code="flag_fom",
                      dvl_w=0.6, dvl_target=0.0, csv_open=True),
                 None, theme.WARN, False)):
            p.add_status(MpcStatus(**kw))
            app.processEvents()
            check(p.btn_dvl, kw, fill, frozen, down)
            if kw["dvl_state"] in ("held", "not_armed", "no_tag"):
                assert p.btn_dvl.text() == "DVL IN", (kw, p.btn_dvl.text())
        for kw, fill in ((dict(ekf_state="ready", ekf_control=True), theme.OK),
                         (dict(ekf_state="warming", ekf_control=True),
                          theme.WARN),
                         (dict(ekf_state="fault", ekf_control=True),
                          theme.FAIL),
                         (dict(ekf_state="fault", ekf_control=False), None),
                         (dict(ekf_state="ready", ekf_control=False), None)):
            p.add_status(MpcStatus(**kw))
            app.processEvents()
            check(p.btn_est, kw, fill)
    finally:
        if p is not None:
            p.close()
        app.setStyleSheet(prev)


def test_the_demo_dvl_is_synthetic_and_says_so():
    """--source demo --dvl: a fabricated stream good enough to drive the
    switch offline, labelled so it can never be read as a device's."""
    TC = _harness()
    TC._app()
    from rov_gui.bus import DataBus
    from rov_gui.backends.demo import DemoVehicleWorker

    with tempfile.TemporaryDirectory() as tmp:
        Opts = TC._test_opts(tmp)
        Opts.dvl = True
        bus = DataBus()
        got, stats = [], []
        bus.dvl.connect(got.append)
        bus.sensor_stat.connect(stats.append)
        v = DemoVehicleWorker(bus, opts=Opts())
        v.setup()
        assert v.dvl_on
        for _ in range(10):
            v.tick()
        assert len(got) == 5                          # every other tick
        assert all(s.source == "demo" and s.note == "synthetic" for s in got)
        assert all(s.valid and s.altitude > 0.05 for s in got)
        assert any(s.name == "DVL" and "(sim)" in s.detail for s in stats)
        Opts.dvl = False
        v2 = DemoVehicleWorker(DataBus(), opts=Opts())
        v2.setup()
        assert not v2.dvl_on


def test_the_csv_header_carries_the_velocity_and_dvl_columns_at_the_end():
    from rov_gui.control.workers import CSV_HEADER

    cols = CSV_HEADER.strip().split(",")
    # schema 17's fifteen, now followed by schema 18's sixteen EKF columns
    # (test_rl_ekf pins those) — append-at-end, so 17's stay where they were.
    i = cols.index("vel_src")
    assert cols[i:i + 15] == ["vel_src", "nu_u", "nu_v", "nu_w", "fd_u",
                              "fd_v", "fd_w", "dvl_u", "dvl_v", "dvl_w",
                              "dvl_ok", "dvl_valid", "dvl_fom", "dvl_alt_m",
                              "dvl_age_s"]
    # ...then schema 19's three pwm-LOW columns (2026-10-10, test_rl_pwm_scale pins them) and schema 20's
    # pwm_guard (2026-10-10 b, test_rl_pwm_interlocks pins it), schema 21's three attitude-age columns
    # (2026-10-15, test_att_age pins them) and schema 22's seven DVL-selection columns (2026-10-16): still
    # append-at-end
    assert cols[i + 15] == "est" and len(cols) == i + 15 + 16 + 3 + 1 + 3 + 7
    assert cols[-7:] == ["w_dvl", "dvl_code", "dvl_res_fast_m_s", "dvl_res_slow_m_s", "dvl_esc_s",
                         "dvl_align", "dvl_scale"]
    assert cols.index("vel_src") == cols.index("rp_track") + 1
    assert len(cols) == len(set(cols))


# ===================================================== selection primary (2026-10-16)
# DTF = RL_controller logs/eval/dvl_trip_fix_20261015. N-numbers are SPEC.md
# §10.3's; [bracketed] numbers are the prototype's on the same scenario
# (DTF design/out/spec_scenarios.json [측정]), which this implementation
# reproduces exactly (DTF impl/out/spec_scenarios_impl.json). The bounds leave
# room for the tag-noise seed, not for a different design.
BLEND_TRACE = (Path(__file__).parent / "fixtures"
               / "dvl_blend_trace_20261015.json")


def _smp(v, t, valid=True, fom=0.002, alt=0.5, n_beams=4):
    """A DvlSample with every field the R3 flags read (_sample has no beams)."""
    return DvlSample(t_valid=t, t_rx=t, valid=valid, fom=fom, altitude=alt,
                     v_dvl=(None if v is None else tuple(float(x) for x in v)),
                     n_beams=n_beams, kind="velocity", hz=10.0, source="test",
                     conn=Conn.ONLINE, stamp=t)


def _paid(**over):
    """SPEC §10.3's _aid(): proven, IN, the default selection (primary)."""
    d = _proven_aid(**over)
    d.proof = {"rms_m_s": 0.02}
    d.use = True
    return d


def _osc(t):
    """Surge-and-sway, 12 / 8 cm/s, 8-s periods (the _fly / prove motion)."""
    return (0.12 * math.sin(2 * math.pi * t / 8.0),
            0.08 * math.sin(2 * math.pi * (t - 3.0) / 8.0), 0.0)


class _Feed:
    """DTF design/spec_scenarios.py `Feed`: a DVL sample and a tag fix every
    0.05 s (heading 0: body = world), the usage question every step and —
    with ``blend`` — the blend stepped on the tag velocity (+1 cm/s noise) as
    the worker's tick would. ``v_dvl`` None = an invalid (no-lock) report,
    False = no report. ``log`` rows: (t, code, tripped, escalation_s,
    residual_slow, w, why)."""

    def __init__(self, d, t=30.0, seed=2, blend=False):
        self.d, self.t = d, float(t)
        self.p = np.array([0.0, 0.0, 0.5])
        self.rng = np.random.default_rng(seed)
        self.imu = _imu(t=t)
        self.log = []
        self.blend = blend
        if blend:
            d.blend_reset()

    def run(self, n, v_dvl, v_tag, noise_mm=1.0, dt=0.05, s_kw=None,
            tag=True):
        for _ in range(n):
            t0, self.t = self.t, self.t + dt
            tm = 0.5 * (t0 + self.t)
            vd = v_dvl(self.t) if callable(v_dvl) else v_dvl
            vt = v_tag(tm) if callable(v_tag) else v_tag
            self.p = self.p + np.asarray(vt, float) * dt
            kw = s_kw(self.t) if callable(s_kw) else (s_kw or {})
            if vd is not False:
                self.d.note_sample(_smp(vd, self.t, valid=vd is not None,
                                        **kw))
            if tag:
                pn = self.p + self.rng.normal(0, noise_mm * 1e-3, 3)
                self.d.note_fix(_fix(list(pn), 0.0, self.t), self.imu)
            ok, why, code = self.d.usable(self.t)
            w = None
            if self.blend:
                fd = (np.asarray(v_tag(self.t) if callable(v_tag) else v_tag,
                                 float) + self.rng.normal(0, 0.01, 3))
                _vb, w, _src, code, _why = self.d.blend_step(self.t, fd, ok,
                                                             code)
            self.log.append((round(self.t, 3), code, bool(self.d.tripped),
                             self.d.escalation_s, self.d.residual_slow_m_s, w,
                             why))
        return self.t

    def first(self, pred, after=None):
        return next((r[0] for r in self.log
                     if (after is None or r[0] > after) and pred(r)), None)

    def last(self, pred, after=None):
        return max((r[0] for r in self.log
                    if (after is None or r[0] > after) and pred(r)),
                   default=None)


def _is_dis(r):
    return str(r[1]).startswith("dis_")


def test_the_blend_is_the_trained_twin_tick_for_tick():
    """N1. DvlBlend = RL_controller deploy/obs_builder.py DvlBlendNp, the
    blend pwm11att13 trains on: five scripted streams (dropout, silence,
    gross, a good report mid-fade, engage on a bad sample), 500 ticks,
    reproduced tick for tick (fixture = DTF map/dvl_blend_trace.json, made by
    running BOTH the torch selector and the numpy twin) [측정: 1.7e-16]."""
    tr = json.loads(BLEND_TRACE.read_text())
    inp = tr["inputs"]
    fd = np.asarray(inp["fd_b_flu"], float)
    n = 0
    for name, sc in tr["scenarios"].items():
        b = DV.DvlBlend(dt=inp["dt"], **inp["blend"])
        for r in sc["rows"]:
            v, w = b.step(r["t"], fd, r["sample_v"], r["sample_t_valid"],
                          r["sample_ok"])
            assert abs(w - r["w"]) <= 1e-6, (name, r["k"], w, r["w"])
            assert float(np.max(np.abs(v - np.asarray(r["v_b"])))) <= 2e-6, \
                (name, r["k"], v, r["v_b"])
            assert b.src == r["src"], (name, r["k"], b.src, r["src"])
            n += 1
    assert len(tr["scenarios"]) == 5 and n == 500


def test_the_blend_constants_are_the_pwm11att13_values():
    """N2. The export's velocity_source (hold 0.45 s, fade 0.5 s, gross
    0.5 m/s against a 1-s low-pass, stale 0.6 s) [스펙], primary by default,
    and the aid's blend steps at the control period it is given."""
    D_ = DV.DEFAULTS
    assert (D_["blend_hold_s"], D_["blend_xfade_s"], D_["blend_gross_m_s"],
            D_["blend_gross_tau_s"], D_["stale_s"]) == (0.45, 0.5, 0.5, 1.0,
                                                         0.6)
    assert D_["selection"] == "primary"
    b = DV.DvlAid(dict(MOUNT), dt=0.05).blend
    assert (b.dt, b.hold_s, b.xfade_s, b.gross, b.gross_tau, b.stale_s,
            b.dvl_w) == (0.05, 0.45, 0.5, 0.5, 1.0, 0.6, False)
    assert len(DV.KEYS_20261016) == 20 and DV.KEYS_20261016[0] == "selection"
    assert set(DV.KEYS_20261016) <= set(DV.KEYS)


def test_the_assembler_blend_path_is_the_fd_path_at_w0():
    """N3. vel_fn answering w = 0 is the finite-difference path, bit for bit
    (the inputs of test_without_a_dvl_the_assembler_is_bit_identical_to_
    before); w > 0 flies the blended (u, v) with the heave and the bridge's
    velocity x age as trained; vel_xy and vel_fn are exclusive."""
    rng = np.random.default_rng(5)
    a0 = StateAssembler(z_source="tag", vel_lp_alpha=0.6, nudot_source="imu")
    a1 = StateAssembler(z_source="tag", vel_lp_alpha=0.6, nudot_source="imu")
    t = 100.0
    p = np.array([0.3, -0.2, -0.9])
    fix = None
    for k in range(200):
        t_cap = t - 0.13
        if k % 2 == 0:
            p = p + rng.normal(0, 0.004, 3) + np.array([0.004, 0.002, 0.0])
            fix = _fix(list(p), 0.4 + 0.01 * k, t_cap)
        imu = _imu(roll=0.02 * math.sin(k), pitch=0.03 * math.cos(k),
                   r=0.1 * math.sin(0.3 * k), t=t - 0.01,
                   f=(0.1 * rng.normal(), 0.1 * rng.normal(), -9.8))
        m0, h0 = a0.step(fix, imu, t, 0.05)
        m1, h1 = a1.step(fix, imu, t, 0.05,
                         vel_fn=lambda tt, fd: (None, "tag_fd", 0.0))
        for key in ("eta", "nu", "nudot"):
            assert np.array_equal(m0[key], m1[key]), (k, key)
        assert np.array_equal(h0["nu_fd"], h1["nu_fd"])
        assert h1["vel_src"] == "tag_fd" and h1["w_dvl"] == 0.0
        assert "w_dvl" not in h0
        t += 0.05
    seen = {}

    def fn(tt, fd):
        seen["fd"] = np.array(fd)
        return np.array([0.2, -0.1]), "blend", 0.4

    fix = _fix([1.0, 2.0, 0.5], 0.3, t - 0.1)
    imu = _imu(t=t)
    m, h = a1.step(fix, imu, t, 0.05, vel_fn=fn)
    assert np.array_equal(m["nu"][:2], [0.2, -0.1]), m["nu"]
    assert m["nu"][2] == h["nu_fd"][2] and np.array_equal(seen["fd"],
                                                          h["nu_fd"])
    assert h["vel_src"] == "blend" and h["w_dvl"] == 0.4
    step = rot_zyx(0.0, 0.0, 0.3) @ np.array([0.2, -0.1, h["nu_fd"][2]]) * 0.1
    assert abs(m["eta"][0] - (1.0 + step[0])) < 1e-12, m["eta"]
    assert abs(m["eta"][1] - (2.0 + step[1])) < 1e-12, m["eta"]
    try:
        a1.step(fix, imu, t, 0.05, vel_xy=(0.1, 0.0), vel_fn=fn)
    except ValueError:
        pass
    else:
        raise AssertionError("vel_xy and vel_fn together were accepted")


def test_a_three_second_false_lock_holds_and_returns_without_a_latch():
    """N4. THE OPERATOR'S CASE (the 10-15 20:25-20:27 trips were real false
    locks, DVL ~0 while the vehicle moved): a 3-s zero lock is HELD — the
    blend fades to the tag velocity — and comes back BY ITSELF; no latch, the
    proof kept, nothing for the operator to press."""
    for spd, hold_max, back_max in ((0.30, 2.0, 9.0), (0.10, 3.5, 4.0)):
        d = _paid()
        f = _Feed(d, blend=True)
        t0 = f.run(200, (spd, 0, 0), (spd, 0, 0))
        t1 = f.run(60, (0.0, 0, 0), (spd, 0, 0))
        first = f.first(_is_dis, t0)
        # held from <= hold_max s into the lock [1.65 s at 0.3, 2.8 s at 0.1]
        assert first is not None and first - t0 <= hold_max, (spd, first)
        ok, why, code = d.usable(t1)
        assert not ok and code == "dis_diff", (spd, code, why)
        st, note = d.state(t1, used=False, why=why, code=code, w=0.0)
        assert st == DV.ST_HELD and "returns by itself" in note, (st, note)
        t2 = f.run(400, (spd, 0, 0), (spd, 0, 0))
        back = f.last(_is_dis, t0)
        # back within back_max s of the lock's end [7.15 s at 0.3, 2.9 at 0.1]
        assert back - t1 <= back_max, (spd, back - t1)
        assert d.usable(t2) == (True, "", "ok")
        assert max(r[3] for r in f.log) < 4.0     # [3.2 s at 0.3]
        assert d.proven and d.n_trips == 0 and not d.tripped
        assert f.log[-1][5] == 1.0                # flown at full weight again


def test_a_lasting_disagreement_latches_and_says_what_to_do():
    """N5. A reversed reading held 15 s at 10 cm/s is not a glitch: the
    escalation latches it (the proof is thrown away), the text says so, the
    panel says how to re-arm, and nothing re-latches under the latch."""
    d = _paid()
    f = _Feed(d)
    t0 = f.run(200, (0.10, 0, 0), (0.10, 0, 0))
    f.run(300, (-0.10, 0, 0), (0.10, 0, 0))
    t_trip = f.first(lambda r: r[2], t0)
    assert t_trip is not None and 4.5 <= t_trip - t0 <= 7.5, t_trip  # [5.65]
    assert d.tripped.startswith("DVL disagreed with the tag velocity "
                                "persistently (") and "now: residual" in \
        d.tripped, d.tripped
    assert not d.proven and d.against
    assert "has to be PROVEN" in d.refusal(True)
    st, note = d.state(f.t, used=False)
    assert st == DV.ST_TRIPPED and "proving pass" in note \
        and "OUT then IN" in note, note
    assert d.usable(f.t)[2] == "latched"
    assert d.n_trips == 1 and d.n_esc["persistent"] == 1
    f.run(200, (-0.10, 0, 0), (0.10, 0, 0))   # the fault goes on
    assert d.n_trips == 1 and d.escalation_s == 0.0
    assert all(r[3] == 0.0 for r in f.log if r[2])   # frozen while latched


def test_a_three_cm_s_bias_is_neither_held_nor_latched():
    """N6. 3 cm/s is inside what the recordings' tag velocity disagrees by:
    no hold, no latch [slow residual max 3.08 cm/s]."""
    d = _paid()
    f = _Feed(d)
    f.run(200, (0.10, 0, 0), (0.10, 0, 0))
    f.run(1200, (0.13, 0, 0), (0.10, 0, 0))
    assert not any(_is_dis(r) for r in f.log) and not d.tripped
    assert d.r_slow_max < DV.DEFAULTS["agree_hold_m_s"], d.r_slow_max


def test_a_ten_cm_s_bias_is_held_then_latched():
    """N7. +10 cm/s held 30 s: a hold first [2.8 s], then the latch [6.8 s]."""
    d = _paid()
    f = _Feed(d)
    t0 = f.run(200, (0.10, 0, 0), (0.10, 0, 0))
    f.run(600, (0.20, 0, 0), (0.10, 0, 0))
    hold = f.first(_is_dis, t0)
    trip = f.first(lambda r: r[2], t0)
    assert hold is not None and 2.0 <= hold - t0 <= 4.0, hold
    assert trip is not None and 5.5 <= trip - t0 <= 8.5, trip
    assert "persistently" in d.tripped, d.tripped


def test_a_false_lock_that_lasts_six_seconds_latches():
    """N8. A zero lock that LASTS (6 s at 0.3 m/s) is latched [5.65 s]: past
    ~3.4 s no single constant lock stays under the 4-s bucket [유도]."""
    d = _paid()
    f = _Feed(d)
    t0 = f.run(200, (0.30, 0, 0), (0.30, 0, 0))
    f.run(120, (0.0, 0, 0), (0.30, 0, 0))
    trip = f.first(lambda r: r[2], t0)
    assert trip is not None and 4.5 <= trip - t0 <= 7.0, trip


def test_an_oscillating_wrong_sign_is_held_at_once_and_latched_only_if_it_lasts():
    """N9. The counterpart of the legacy alignment trip: a station hold on
    wrong-signed feedback (4 rad/s, 6 cm/s) is HELD at once by the alignment
    verdict; for one second that is all [no latch], for fifteen it latches
    [6.2 s] — the alignment's coherent verdict is escalation evidence."""
    w, amp = 4.0, 0.06

    def tag(t):
        return (amp * math.sin(w * t), 0.0, 0.0)

    def flipped(t):
        return (-amp * math.sin(w * t), 0.0, 0.0)

    for n, latches in ((20, False), (300, True)):
        d = _paid()
        f = _Feed(d)
        t0 = f.run(200, tag, tag, noise_mm=0.5)
        f.run(n, flipped, tag, noise_mm=0.5)
        t2 = f.run(400, tag, tag, noise_mm=0.5)
        assert any(r[1] == "dis_align" for r in f.log if r[0] > t0), n
        trip = f.first(lambda r: r[2], t0)
        if latches:
            assert trip is not None and 5.0 <= trip - t0 <= 8.0, trip
            assert "alignment" in d.tripped, d.tripped
        else:
            assert trip is None and d.usable(t2)[0], (trip, d.usable(t2))


def test_a_device_configuration_change_still_latches():
    """N10. R4a, unchanged: the device rotating its own output is a latch at
    once, and the proof made on the old stream is gone."""
    d = _paid()
    f = _Feed(d)
    f.run(100, (0.10, 0, 0), (0.10, 0, 0))
    d.note_device_config({"mounting_rotation_offset": 180.0,
                          "speed_of_sound": 1475.0})
    d.note_device_config({"mounting_rotation_offset": 0.0,
                          "speed_of_sound": 1475.0})
    assert d.tripped.startswith("the device's configuration changed ("), \
        d.tripped
    assert not d.proven
    ok, why, code = d.usable(f.t)
    assert not ok and why.startswith("DVL latched OUT (") \
        and code == "latched", (ok, why, code)


def test_quality_flags_hold_the_sample_not_the_evidence():
    """N11. R3: the report's own word that it may be a false lock — fom,
    an altitude jump, beams, a lock re-acquired — keeps the sample from being
    FLOWN, never trips, never moves the jump gate's reference, and leaves
    the monitors their pairs; an altitude that stays is believed after
    ~0.85 s; with the flags off the same reports fly."""
    d = _paid()
    f = _Feed(d)
    f.run(40, (0.10, 0, 0), (0.10, 0, 0))
    t = f.t
    d.note_sample(_smp((0.10, 0, 0), t + 0.05, fom=0.004))
    assert d.usable(t + 0.05) == (
        False, "DVL quality flag: fom (4.0 > 3.0 mm/s)", "flag_fom")
    assert abs(d._last_good[0] - t) < 1e-9     # not the jump reference
    d.note_sample(_smp((0.10, 0, 0), t + 0.10, alt=0.75))
    ok, why, code = d.usable(t + 0.10)
    assert code == "flag_alt" and "altitude jump" in why, why
    t = t + 0.10
    codes = []
    for k in range(1, 30):
        d.note_sample(_smp((0.10, 0, 0), t + 0.05 * k, alt=0.75))
        codes.append((round(0.05 * k, 2), d.usable(t + 0.05 * k)[2]))
    first_ok = next(a for a, c in codes if c == "ok")
    assert first_ok <= 1.0, codes             # [0.85 s]
    assert all(c == "ok" for a, c in codes if a >= first_ok), codes
    d.note_sample(_smp((0.10, 0, 0), t + 1.6, n_beams=3, alt=0.75))
    assert d.usable(t + 1.6) == (False, "DVL quality flag: beams (3 < 4)",
                                 "flag_beams")
    d.note_sample(_smp(None, t + 1.65, valid=False, alt=0.75))
    d.note_sample(_smp((0.10, 0, 0), t + 1.70, alt=0.75))
    ok, why, code = d.usable(t + 1.70)
    assert code == "flag_reacq" and "lock re-acquired" in why, why
    d.note_sample(_smp((0.10, 0, 0), t + 1.75, alt=0.75))
    assert d.usable(t + 1.75) == (True, "", "ok")
    # no flag is a phrase with a digit before " (" (the meta counts by it)
    assert not any(ch.isdigit() for ch in why.split(" (")[0])
    d2 = _paid(flag_fom_m_s=0.0, flag_alt_jump_m=0.0, flag_min_beams=0,
               flag_reacquired=False)
    _Feed(d2).run(40, (0.10, 0, 0), (0.10, 0, 0))
    d2.note_sample(_smp(None, 32.02, valid=False))
    d2.note_sample(_smp((0.10, 0, 0), 32.05, fom=0.004, n_beams=3))
    assert d2.usable(32.05) == (True, "", "ok")
    n0 = d.n_monitor
    f.t = t + 1.75
    f.run(40, (0.10, 0, 0), (0.10, 0, 0),
          s_kw=lambda tt: {"fom": 0.004, "alt": 0.75})
    assert d.n_monitor - n0 >= 35, d.n_monitor - n0      # [39 of 40]
    assert all(r[1] == "flag_fom" for r in f.log[-40:])
    m = d.meta()["flags"]["counts"]
    assert m["flag_fom"] >= 41 and m["flag_beams"] == 1 \
        and m["flag_reacq"] == 1 and m["flag_alt"] >= 1, m


def test_a_coherent_refutation_holds_and_escalates():
    """N12. A DVL mounted 180 deg out, in a surge-and-sway oscillation: held
    within ~1 s [0.9 s], latched once it has lasted [15.8 s]."""
    d = _paid()
    f = _Feed(d)
    f.run(100, (0.10, 0, 0), (0.10, 0, 0))
    t0 = f.t
    f.run(400, lambda t: (-_osc(t)[0], -_osc(t)[1], 0.0), _osc)
    hold = f.first(_is_dis, t0)
    trip = f.first(lambda r: r[2], t0)
    assert hold is not None and hold - t0 <= 2.0, hold
    assert trip is not None and trip - t0 <= 25.0, trip
    assert d.n_trips == 1


def test_a_2x_scale_in_oscillation_is_held_and_latched():
    """N13. A DVL reading twice the motion refutes INCOHERENTLY (the fit sees
    ~half of it) and never reaches the alignment threshold — the scale
    evidence latches it [held 3.7 s, latched 6.75 s, gain 0.55, DVL weight
    3.9 s of the fault]."""
    d = _paid()
    f = _Feed(d, blend=True)
    f.run(100, _osc, _osc)
    t0 = f.t
    fits = []
    for _ in range(1200):
        f.run(1, lambda t: (2.0 * _osc(t)[0], 2.0 * _osc(t)[1], 0.0), _osc)
        if not d.tripped:
            fits.append(d.agree.fit())
    hold = f.first(_is_dis, t0)
    trip = f.first(lambda r: r[2], t0)
    assert hold is not None and hold - t0 <= 10.0, hold
    assert trip is not None and trip - t0 <= 20.0, trip
    assert "the DVL reads" in d.tripped, d.tripped
    refuted = [x for x in fits if x["verdict"] == "refuted"]
    assert refuted and refuted[-1]["coherent"] is False, refuted[-1:]
    flown = sum(r[5] for r in f.log if r[0] > t0 and r[5] is not None) * 0.05
    assert flown <= 8.0, flown


def test_a_false_lock_at_rest_is_flown_not_tripped():
    """N14. The counterpart of the legacy "stands apart for a second is a
    trip" test: 12 cm/s for 1 s at rest is held by the alignment verdict at
    most, nothing latches [slow residual 2.6 cm/s]."""
    d = _paid()
    f = _Feed(d)
    t0 = f.run(100, (0.0, 0, 0), (0.0, 0, 0))
    f.run(20, (0.12, 0, 0), (0.0, 0, 0))
    f.run(100, (0.0, 0, 0), (0.0, 0, 0))
    assert not d.tripped and d.n_trips == 0
    assert d.r_slow_max < 0.05, d.r_slow_max
    assert {r[1] for r in f.log if r[0] > t0} <= {"dis_align", "ok"}, \
        {r[1] for r in f.log if r[0] > t0}


def test_a_flickering_false_lock_is_never_flown():
    """N27. A zero lock that alternates with no-lock reports: every valid
    report follows a gap marker, so every one is flag_reacq and the blend
    never takes it [w 0.078 over the window]."""
    d = _paid()
    f = _Feed(d, blend=True)
    f.run(200, (0.12, 0, 0), (0.12, 0, 0))
    t0 = f.t
    k = {"n": 0}

    def flick(t):
        k["n"] += 1
        return None if k["n"] % 2 == 0 else (0.0, 0.0, 0.0)
    t1 = f.run(200, flick, (0.12, 0, 0))
    rows = [r for r in f.log if t0 < r[0] <= t1]
    assert {r[1] for r in rows} <= {"flag_reacq", "no_lock", "ok"}, \
        {r[1] for r in rows}
    assert sum(1 for r in rows if r[1] == "ok") <= 2           # [2]
    assert float(np.mean([r[5] for r in rows])) <= 0.15
    assert not d.tripped


def test_a_disagreement_onset_expires_the_held_sample():
    """N28. A disagreement ends the hold on the sample it is about: the fade
    starts on the onset tick [w 1.0 -> 0.9], instead of flying it 0.45 s."""
    d = _paid()
    f = _Feed(d, blend=True)
    f.run(200, (0.30, 0, 0), (0.30, 0, 0))
    first = None
    for _ in range(80):
        f.run(1, (0.0, 0, 0), (0.30, 0, 0))
        if first is None and _is_dis(f.log[-1]):
            first = (f.log[-2][5], f.log[-1][5], d.blend.t_hold, d.blend.src)
    assert first is not None
    w0, w1, t_hold, src = first
    assert t_hold == -math.inf and src == "blend", first
    assert abs(w1 - (w0 - 0.1)) < 1e-9, first


def test_the_escalation_is_frozen_while_latched_and_reset_by_a_rearm():
    """N29. Under the latch nothing in the escalation moves; the proof made
    afterwards and the IN that clears the latch start it afresh — the
    shipped re-trip loop (4 trips in 2.2 min on mpc_202432) cannot happen —
    and evidence from before a proof does not count after it."""
    d = _paid()
    f = _Feed(d)
    f.run(200, _osc, _osc)
    n = 0
    while not d.tripped and n < 2000:
        f.run(1, lambda t: (_osc(t)[0], -_osc(t)[1], 0.0), _osc)
        n += 1
    assert d.tripped, "the reversed sway never latched"
    esc = []
    for _ in range(1800):                      # 90 s more of the fault
        f.run(1, lambda t: (_osc(t)[0], -_osc(t)[1], 0.0), _osc)
        esc.append(d.escalation_s)
    assert max(esc) == 0.0 and d.n_trips == 1
    k = 0
    while not d.proven and k < 6000:           # honest data: a new proof
        f.run(1, _osc, _osc)
        k += 1
    assert d.proven
    assert d.set_use(True) == "" and not d.tripped
    assert d.escalation_s == 0.0
    f.run(400, _osc, _osc)
    assert d.n_trips == 1 and not d.tripped    # no re-latch in the 20 s
    # ...and 60 s of +10 cm/s in SHADOW before the first proof
    d = _paid()
    d.proven, d.proof, d.use = False, None, False
    f = _Feed(d)
    f.run(1200, (0.20, 0.0, 0), (0.10, 0, 0))
    assert d.escalation_s == 0.0
    k = 0
    while not d.proven and k < 6000:
        f.run(1, _osc, _osc)
        k += 1
    assert d.proven and d.set_use(True) == ""
    f.run(400, _osc, _osc)
    assert not d.tripped and d.n_trips == 0


def test_a_false_lock_that_keeps_coming_back_latches_repeatedly():
    """N30. 3 s of zero lock every 12 s at 0.12 m/s: each one alone drains
    away, together they are a DVL that keeps disagreeing — a warning at the
    second [14.47 s], the latch at the third [26.55 s]."""
    d = _paid()
    f = _Feed(d, blend=True)
    t0 = f.run(200, (0.12, 0, 0), (0.12, 0, 0))

    def rec(t):
        return (0.0, 0.0, 0.0) if ((t - t0) % 12.0) < 3.0 else (0.12, 0.0, 0.0)
    f.run(1200, rec, (0.12, 0, 0))
    assert len(d.warn_log) == 1, d.warn_log
    tw, text = d.warn_log[0]
    assert 10.0 <= tw - t0 <= 20.0 and "again (2 times in the last 60 s)" \
        in text, (tw - t0, text)
    trip = f.first(lambda r: r[2], t0)
    assert trip is not None and 20.0 <= trip - t0 <= 32.0, trip
    assert "repeatedly (3 times in the last 60 s" in d.tripped, d.tripped
    assert d.take_warnings() == d.warn_log and d.take_warnings() == []


def test_a_reversed_axis_does_not_flap_back_in():
    """N31. A reversed sway at 5 cm/s keeps disappearing from view as the
    oscillation turns; the hold releases only once its evidence has DRAINED,
    so the DVL never flies again between the first hold [0.7 s] and the
    latch [9.8 s] [DVL weight 0.93 s]."""
    k = 0.05 / 0.12
    d = _paid()
    f = _Feed(d, blend=True)
    f.run(200, lambda t: tuple(k * x for x in _osc(t)),
          lambda t: tuple(k * x for x in _osc(t)))
    t0 = f.t
    f.run(600, lambda t: (k * _osc(t)[0], -k * _osc(t)[1], 0.0),
          lambda t: tuple(k * x for x in _osc(t)))
    rows = [r for r in f.log if r[0] > t0]
    i_hold = next(i for i, r in enumerate(rows) if _is_dis(r))
    i_trip = next(i for i, r in enumerate(rows) if r[2])
    assert rows[i_hold][0] - t0 <= 2.0 and rows[i_trip][0] - t0 <= 15.0
    assert not any(r[1] == "ok" for r in rows[i_hold:i_trip]), \
        [r[:2] for r in rows[i_hold:i_trip] if r[1] == "ok"]
    assert sum(r[5] for r in rows if r[5] is not None) * 0.05 <= 2.0


def test_sparse_tags_do_not_stall_the_escalation():
    """N32. Tags in bursts (0.3 s on, 0.7 s off — the high-tilt case where
    the 10-15 false locks sat): the 0.5-s pair gap restarted the old
    monitors on every burst; the slow monitors run across gaps up to
    SLOW_GAP_S. A 20-s zero lock at 0.12 m/s: held [4.25 s], latched
    [10.15 s] (DTF design/out/synth_v2.json sparse_tags)."""
    rng = np.random.default_rng(5)
    d = _paid()
    imu = _imu(t=30.0)
    t = 30.0
    p = np.array([0.0, 0.0, 0.5])
    nd = nt = tl = t
    last_fix = None
    first_hold = trip = None
    while t < 80.0:
        tn = t + 0.05
        while True:
            te = min(nd, nt)
            if te > tn:
                break
            p = p + np.array([0.12, 0.0, 0.0]) * (te - tl)
            tl = te
            if te == nd:
                v = (0.0, 0.0, 0.0) if 40.0 <= te <= 60.0 else (0.12, 0, 0)
                d.note_sample(_smp(v, te))
                nd += 0.1
            else:
                if ((te - 30.0) % 1.0) < 0.3:
                    d.note_fix(_fix(list(p + rng.normal(0, 1e-3, 3)), 0.0, te),
                               imu)
                    last_fix = te
                nt += 1 / 15
        t = tn
        fd = np.array([0.12, 0.0, 0.0]) + rng.normal(0, 0.01, 3)
        if last_fix is None or t - last_fix > 0.5:
            continue                           # no tag state: no blend step
        ok, _why, code = d.usable(t)
        _vb, _w, _src, code, _ = d.blend_step(t, fd, ok, code)
        if first_hold is None and t >= 40.0 and str(code).startswith("dis_"):
            first_hold = t - 40.0
        if trip is None and t >= 40.0 and d.tripped:
            trip = t - 40.0
    assert first_hold is not None and first_hold <= 6.0, first_hold
    assert trip is not None and trip <= 14.0, trip


def test_the_rung_admission_refuses_a_false_lock_the_blend_still_flies():
    """N33. A +0.25 m/s false lock at hover is still `ok` for the blend for
    the moment before a disagreement shows — and the station bridge's rung,
    which nothing would correct, does not take it: the shipped innovation
    limit (review 2026-10-02: such a lock walked a station hold 0.5 m)."""
    d = _paid()
    f = _Feed(d, blend=True)
    f.run(200, (0.0, 0, 0), (0.0, 0, 0))
    f.run(10, (0.25, 0, 0), (0.0, 0, 0))
    assert d.usable(f.t)[2] == "ok"
    assert d.rung_admit(f.t, np.zeros(3), 0.1) == (
        False, "DVL innovation (0.25 > 0.15 m/s)")


def test_the_blend_replays_ticks_without_a_tag_state():
    """N34. Ticks with no tag state do not step the blend; the next one
    replays their fade steps toward the current target [w 1.0 -> 0.4]."""
    b = DV.DvlBlend()
    b.reset()
    b.step(0.0, np.zeros(3), np.array([0.1, 0, 0]), 0.0, True)
    assert b.w == 1.0
    _v, w = b.step(1.0, np.zeros(3), np.array([0.1, 0, 0]), 0.0, False,
                   n_skip=5)
    assert abs(w - 0.4) < 1e-9 and b.src == "blend", (w, b.src)


def test_the_rung_holds_across_a_flag_then_refuses():
    """N18a. The rung carries the blend's own hold: a flagged newest sample on
    a lost tick is bridged on the last usable one while it was captured
    <= blend_hold_s ago, then refused (ages kept off 0.45 itself)."""
    d = _paid()
    f = _Feed(d, blend=True)
    t_last = f.run(100, (0.10, 0, 0), (0.10, 0, 0))
    assert d.rung_admit(t_last, np.array([0.10, 0.0, 0.0]), 0.05) == (True, "")
    for k in range(1, 12):
        tk = t_last + 0.05 * k
        d.note_sample(_smp((0.10, 0, 0), tk, fom=0.004))
        v, why = d.rung_velocity(tk)
        if 0.05 * k <= 0.40 + 1e-9:
            assert v is not None and abs(v[0] - 0.10) < 1e-12, (k, why)
            assert why.startswith("held: DVL quality flag: fom"), why
        elif 0.05 * k >= 0.5 - 1e-9:
            assert v is None and why.startswith("DVL quality flag: fom"), \
                (k, why)
    assert d.n_rung_held == 8
    # a usable sample is carried (and becomes the hold); a DISAGREEMENT then
    # refuses at once, however fresh the held sample is
    t2 = t_last + 0.6
    d.note_sample(_smp((0.10, 0, 0), t2))
    assert d.rung_velocity(t2)[1] == ""
    d._hold_diff, d._r_slow = True, 0.08       # the slow residual's hold
    d.note_sample(_smp((0.10, 0, 0), t2 + 0.05))
    v, why = d.rung_velocity(t2 + 0.05)
    assert v is None and why.startswith("DVL disagrees with the tag "
                                        "velocity (slow residual"), why


def test_arming_ignores_a_refutation_the_switch_does_not():
    """N35a. A refutation current at engage refuses the switch IN
    (refusal, unchanged) but no longer the RUN (arm_refusal): the per-tick
    dis_fit hold holds the DVL while it lasts and releases by itself. The
    latches are off here to build the state."""
    for name, fn, n in (("incoherent_2x",
                         lambda t: (2.0 * _osc(t)[0], 2.0 * _osc(t)[1], 0.0),
                         300),
                        ("coherent_180",
                         lambda t: (-_osc(t)[0], -_osc(t)[1], 0.0), None)):
        d = _paid(esc_scale_ratio=0.0, esc_latch_s=1e9, esc_repeat_n=0)
        f = _Feed(d)
        f.run(200, _osc, _osc)
        k = 0
        while k < 1600 and not d.tripped and not (
                n is None and d._fit_current(f.t) and d._refuted_coh):
            f.run(1, fn, _osc)
            k += 1
            if n is not None and k >= n:
                break
        assert d.agree.fit()["verdict"] == "refuted" and \
            d._fit_current(f.t), name
        assert "REFUTES" in d.refusal(True), name
        assert d.arm_refusal() == "", (name, d.arm_refusal())
        assert d.usable(f.t)[2] in ("dis_fit", "dis_diff"), \
            (name, d.usable(f.t))
        assert not d.tripped


def test_every_reason_the_ladder_or_the_worker_gives_has_a_code():
    """N38. The CSV's dvl_code is a closed vocabulary: every reason the
    legacy ladder, the data gates, usable(), the blend and the worker give
    maps to a code, never "other"."""
    phrases = [
        "", "DVL is OUT (shadow)", "DVL latched OUT (x)", "DVL tripped (x)",
        "DVL tripped before engage", "DVL not armed at engage (x)",
        "DVL mount not measured", "DVL not proven against the tag velocity",
        "no DVL data", "DVL has no bottom lock", "DVL is in water-tracking mode",
        "DVL report lacks fom or altitude", "DVL fom too high (0.1 > 0.05 m/s)",
        "DVL altitude too low (0.02 < 0.05 m)", "DVL velocity implausible",
        "DVL velocity jump (0.30 > 0.15 m/s)",
        "no body rate for the DVL lever arm", "DVL stale (0.70s)",
        "DVL velocity not finite", "DVL code fault",
        "DVL quality flag: fom (4.0 > 3.0 mm/s)",
        "DVL quality flag: altitude jump (0.75 m against 0.50 m over the "
        "last 1 s)", "DVL quality flag: beams (3 < 4)",
        "DVL quality flag: lock re-acquired (the report 0.05 s before it "
        "failed a data gate)",
        "DVL disagrees with the tag velocity (slow residual 6.0 > 5.0 cm/s)",
        "DVL disagreement releasing (slow residual 3.0 cm/s; it returns "
        "below 2.5)",
        "DVL disagreement releasing (waiting for 1.2 s of disagreement "
        "evidence to drain)",
        "DVL disagrees with the tag velocity: alignment (bearing +0 deg: "
        "alignment -0.90, the tags see 0.95 of it)",
        "DVL disagrees with the tag velocity: the fit refutes the mount (the "
        "tags see the motion elsewhere)",
        "DVL disagrees with the tag velocity: the fit refutes it (the tags "
        "see only 0.55 of the motion it reports)",
        "DVL gross (|v_dvl - LP(v_tag)| 0.80 > 0.50 m/s)",
        "no tag state this tick (the blend waits for it)",
        "no tag state this tick (the DVL needs one to be checked)",
        "station bridge on tier imu (the DVL rung is not carrying this "
        "dropout)", "held: DVL has no bottom lock",
        "DVL re-entry (0.10/0.50s)", "DVL innovation (0.25 > 0.15 m/s)",
        "DVL innovation hold (until a fresh tag velocity agrees; it was x)",
        "DVL unchecked (no current tag velocity)",
        "DVL held back (the fit refutes it)",
        "DVL alignment hold (bearing +0 deg: alignment -0.90, the tags see "
        "0.95 of it)"]
    bad = [p for p in phrases if DV.reason_code(p) == "other"]
    assert not bad, bad
    assert DV.reason_code("DVL tripped (x)") == "latched"
    assert DV.reason_code("DVL tripped before engage") == "not_armed"
    assert DV.reason_code("DVL held back (the fit refutes it)") == "refute_hold"
    assert DV.reason_code("") == "ok" and DV.reason_code("what?") == "other"
    # ...and the chip's words
    assert DV.reason_class("dis_fit") == "disagrees"
    assert DV.reason_class("flag_reacq") == "flag re-lock"
    assert DV.reason_class("ok") == "" and DV.reason_class("rung") == ""
    # a live aid's own reasons, every code it can answer
    d = _paid()
    f = _Feed(d)
    f.run(40, (0.10, 0, 0), (0.10, 0, 0))
    for why in [r[6] for r in f.log] + [d.usable(f.t + 9.0)[1]]:
        assert DV.reason_code(why) != "other", why


# ------------------------------------------------- selection primary: worker
def _fly_n15(tmp, **over):
    """N15's flight through the real worker: proven, IN, engaged; 6 ticks
    good at (0.07, -0.02); 20 with no bottom lock; 14 good again. -> (w,
    rows, meta dvl, logs, statuses). Sample ages are exact multiples of
    0.05 s (the _Sim clock)."""
    TC, w, bus, _pilots, logs = _dvl_worker(tmp, reentry_s=0.0, **over)
    sim = _Sim(w, TC)
    st = []
    try:
        assert w.dvl.selection == "primary"
        sim.prove()
        w.set_dvl_use(True)
        sim.engage()
        bus.mpc_status.connect(st.append)
        v = (0.07, -0.02, 0.0)
        for _ in range(6):
            sim.step(v=v)
        for _ in range(20):
            sim.step(v=v, valid=False)
        for _ in range(14):
            sim.step(v=v)
        assert w.engaged, w.reason
        meta = w._run_meta()["dvl"]
        rows = _rows(w)
    finally:
        sim.close()
    return w, rows, meta, logs, st


def test_primary_fades_row_by_row_and_never_hard_switches():
    """N15. A lost bottom lock is a tick without a DVL — under primary the
    last good sample is held 0.45 s and the weight fades 0.1 per tick to the
    tag velocity, and back: no row changes w by more than one fade step. The
    first report after the lock comes back is a re-acquired lock and is not
    flown; the next ones fade in."""
    with tempfile.TemporaryDirectory() as tmp:
        w, rows, meta, _logs, _st = _fly_n15(tmp)
    src = [r["vel_src"] for r in rows]
    wd = [float(r["w_dvl"]) for r in rows]
    code = [r["dvl_code"] for r in rows]
    assert len(rows) == 40, len(rows)
    assert src[:6] == ["dvl"] * 6 and wd[:6] == [1.0] * 6 and \
        code[:6] == ["ok"] * 6
    n_hold = 0
    while src[6 + n_hold] == "hold":
        n_hold += 1
    assert n_hold in (8, 9), src[6:20]            # 0.45 s of capture time
    assert wd[6:6 + n_hold] == [1.0] * n_hold
    i = 6 + n_hold
    assert src[i:i + 9] == ["blend"] * 9, src[i:i + 9]
    assert np.allclose(wd[i:i + 9], [0.9 - 0.1 * k for k in range(9)],
                       atol=1e-3), wd[i:i + 9]
    assert all(s == "tag_fd" and x == 0.0 for s, x in zip(src[i + 9:26],
                                                         wd[i + 9:26]))
    assert code[6:26] == ["no_lock"] * 20, code[6:26]
    # the lock is back: the first report is a RE-ACQUIRED lock, not flown...
    assert (src[26], wd[26], code[26]) == ("tag_fd", 0.0, "flag_reacq")
    # ...then the weight fades in, 0.1 per tick, to the DVL
    assert src[27:36] == ["blend"] * 9 and code[27:] == ["ok"] * 13
    assert np.allclose(wd[27:36], [0.1 * (k + 1) for k in range(9)],
                       atol=1e-3), wd[27:36]
    assert src[36:] == ["dvl"] * 4 and wd[36:] == [1.0] * 4
    # never a hard switch; the row's (u, v) IS the blend of the two it shows
    assert all(abs(b - a) <= 0.1 + 1e-6 for a, b in zip(wd, wd[1:])), wd
    for r, x in zip(rows, wd):
        exp = x * 0.07 + (1.0 - x) * float(r["fd_u"])
        assert abs(float(r["nu_u"]) - exp) < 2e-5, (r["vel_src"], x,
                                                     r["nu_u"], exp)
    m = meta["rows"]
    # tag_fd: what is left of the 20 after hold and fade, + the re-lock row
    assert m.get("dvl") == 10 and m.get("hold") == n_hold \
        and m.get("blend") == 18 and m.get("tag_fd") == 12 - n_hold, m
    assert m.get("fallback: DVL has no bottom lock") == 20, m
    assert m.get("fallback: DVL quality flag: lock re-acquired") == 1, m
    assert not any(any(c.isdigit() for c in k) for k in m), m


def test_the_csv_and_meta_record_the_selection():
    """N20. Schema 22: the seven columns on every row, parseable, the code a
    single token; the meta says which selection flew and with what."""
    with tempfile.TemporaryDirectory() as tmp:
        w, rows, meta, _logs, _st = _fly_n15(tmp)
        m_all = w._run_meta()
    assert m_all["schema_version"] == 22
    for r in rows:
        for k in ("w_dvl", "dvl_res_fast_m_s", "dvl_res_slow_m_s",
                  "dvl_esc_s", "dvl_align", "dvl_scale"):
            float(r[k])                           # parses ("nan" included)
        assert r["dvl_code"] and "," not in r["dvl_code"] \
            and " " not in r["dvl_code"], r["dvl_code"]
        assert math.isfinite(float(r["dvl_esc_s"]))       # primary: known
    assert meta["selection"] == "primary" and meta["reentry_s_used"] is False
    assert meta["blend"] == {"hold_s": 0.45, "xfade_s": 0.5, "gross_m_s": 0.5,
                             "gross_tau_s": 1.0, "stale_s": 0.6,
                             "dvl_w": False, "dt_s": 0.05}, meta["blend"]
    assert meta["flags"]["reacquired"] is True \
        and meta["flags"]["counts"]["flag_reacq"] >= 1
    for k in ("disagreement", "escalation", "inactive_keys", "governed_by",
              "station_bridge_dvl"):
        assert k in meta, k
    assert meta["inactive_keys"] == {"reentry_s": 0.0, "agree_trip_m_s": 0.08}
    assert meta["escalation"]["latches"] == {"persistent": 0, "repeated": 0}
    assert "residual_slow_m_s" in meta["monitor_now"]
    assert "a disagreement is a hold that releases by itself" in \
        meta["switch_rule"]
    assert "gates" in meta and "reentry_s_used" not in meta["gates"]
    DV.resolve(dict(meta["gates"], **meta["agreement_cfg"]))   # replayable
    json.dumps(m_all, default=str)


def test_primary_never_asks_the_legacy_ladder():
    """N21. Under primary the tick, the rung and the after-step never ask
    velocity(), displacement() or note_tag_velocity() — the legacy ladder
    raising changes nothing."""
    def boom(*_a, **_k):
        raise RuntimeError("the legacy ladder was asked")

    orig = (DV.DvlAid.velocity, DV.DvlAid.displacement,
            DV.DvlAid.note_tag_velocity)
    DV.DvlAid.velocity = DV.DvlAid.displacement = boom
    DV.DvlAid.note_tag_velocity = boom
    try:
        with tempfile.TemporaryDirectory() as tmp:
            w, rows, _meta, _logs, _st = _fly_n15(tmp)
            assert not w._dvl_fault, w._dvl_fault
    finally:
        (DV.DvlAid.velocity, DV.DvlAid.displacement,
         DV.DvlAid.note_tag_velocity) = orig
    assert [r["vel_src"] for r in rows][:6] == ["dvl"] * 6
    assert [r["dvl_code"] for r in rows][26] == "flag_reacq"


def test_legacy_keeps_the_shipped_tick():
    """N22. selection: legacy is the shipped tick: hard switches, w 1 / 0,
    no escalation; the seven new columns only describe it."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, _pilots, _logs = _dvl_worker(tmp, reentry_s=0.0,
                                                 selection="legacy")
        sim = _Sim(w, TC)
        try:
            assert w.dvl.selection == "legacy"
            sim.prove()
            w.set_dvl_use(True)
            sim.engage()
            for _ in range(6):
                sim.step(v=(0.07, -0.02, 0.0))
            for _ in range(4):
                sim.step(v=(0.07, -0.02, 0.0), valid=False)
            m = w._run_meta()["dvl"]
            rows = _rows(w)
        finally:
            sim.close()
    assert [r["vel_src"] for r in rows] == ["dvl"] * 6 + ["tag_fd"] * 4
    assert [r["w_dvl"] for r in rows] == ["1.000"] * 6 + ["0.000"] * 4
    assert [r["dvl_code"] for r in rows] == ["ok"] * 6 + ["no_lock"] * 4
    assert all(r["dvl_esc_s"] == "nan" for r in rows)
    assert m["rows"]["dvl"] == 6 and m["rows"]["tag_fd"] == 4
    assert m["rows"]["fallback: DVL has no bottom lock"] == 4
    assert m["selection"] == "legacy" and m["reentry_s_used"] is True
    assert m["inactive_keys"] == {} and "station_bridge_dvl" not in m
    assert m["flags"]["counts"]["flagged"] == 0


def test_a_disagreement_in_flight_holds_and_returns_by_itself():
    """N16. THE OPERATOR'S REQUEST: the DVL reading the reverse of the tags
    for two seconds is a HOLD — it fades to the tag velocity at once, comes
    back by itself when it agrees again, keeps its proof and its arming;
    no TRIPPED anywhere, nothing to press."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, _pilots, logs = _dvl_worker(tmp, reentry_s=0.0)
        sim = _Sim(w, TC)
        try:
            sim.prove()
            w.set_dvl_use(True)
            sim.engage()
            st = []
            bus.mpc_status.connect(st.append)
            for _ in range(20):
                sim.step(v=(0.10, 0.0, 0.0))
            assert st[-1].vel_src == "dvl" and st[-1].dvl_w == 1.0
            for _ in range(40):
                sim.step(v=(0.10, 0.0, 0.0), dvl_v=(-0.10, 0.0, 0.0))
            for _ in range(240):
                sim.step(v=(0.10, 0.0, 0.0))
            assert w.engaged and w._dvl_run_ok and not w.dvl.tripped
            assert w.dvl.proven
            assert not any("TRIPPED" in m for _l, m in logs), \
                [m for _l, m in logs if "TRIPPED" in m]
            held = [m for lvl, m in logs if lvl == "warn"
                    and m.startswith("dvl: HELD (not latched) — ")]
            assert len(held) == 1 and "nothing to press" in held[0], held
            rel = [m for lvl, m in logs if lvl == "info"
                   and m.startswith("dvl: hold released")]
            assert len(rel) == 1, rel
            assert all(s.dvl_state != "tripped" for s in st)
            assert any(s.dvl_state == "held" for s in st)
            i = next(k for k, s in enumerate(st)
                     if s.dvl_code.startswith("dis_"))
            assert st[i - 1].dvl_w == 1.0 and abs(st[i].dvl_w - 0.9) < 1e-9, \
                (st[i - 1].dvl_w, st[i].dvl_w)        # N28: no 0.45 s hold
            assert st[-1].dvl_w == 1.0 and st[-1].vel_src == "dvl"
            m = w._run_meta()["dvl"]
            assert m["trips"] == 0 and m["escalation"]["latches"] == {
                "persistent": 0, "repeated": 0}
            assert sum(m["disagreement"]["hold_onsets"].values()) >= 1
        finally:
            sim.close()


def test_a_lasting_disagreement_in_flight_latches_and_rearming_is_unchanged():
    """N17. The same reversed reading for 15 s is not a glitch: ONE latch
    line that says what to do, an event, the TRIPPED face, a fade (not a
    step) to the tag velocity, a frozen escalation — and the shipped
    re-arming: a new proof, then OUT and IN."""
    TC = _harness()
    TC._app()
    from rov_gui import theme
    from rov_gui.widgets.trajectory import TrajectoryWindow

    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, _pilots, logs = _dvl_worker(tmp, reentry_s=0.0,
                                                agree_window_s=8.0)
        sim = _Sim(w, TC)
        p = None
        try:
            sim.prove()
            w.set_dvl_use(True)
            sim.engage()
            st, ev = [], []
            bus.mpc_status.connect(st.append)
            bus.mpc_event.connect(ev.append)
            for _ in range(10):
                sim.step(v=(0.10, 0.0, 0.0))
            for _ in range(300):
                sim.step(v=(0.10, 0.0, 0.0), dvl_v=(-0.10, 0.0, 0.0))
            assert w.dvl.tripped and w.engaged
            errs = [m for lvl, m in logs if lvl == "error" and "TRIPPED" in m]
            assert len(errs) == 1, errs
            assert errs[0].startswith(
                "dvl: TRIPPED (latched) — DVL disagreed with the tag velocity "
                "persistently"), errs[0]
            assert "proving pass" in errs[0] and "OUT and IN" in errs[0]
            assert any(e.startswith("dvl: TRIPPED (latched)") for e in ev)
            assert st[-1].dvl_state == "tripped" and st[-1].dvl_w == 0.0
            assert st[-1].dvl_esc_s == 0.0 and w.dvl.escalation_s == 0.0
            ws = [s.dvl_w for s in st if s.dvl_w is not None]
            assert all(abs(b - a) <= 0.1 + 1e-6 for a, b in zip(ws, ws[1:]))
            p = TrajectoryWindow()
            p.add_status(st[-1])
            assert p.btn_dvl.text() == "DVL TRIPPED"
            assert p._dvl_css == p._state_btn_css(theme.FAIL)
            assert "TRIPPED = it disagreed" in p.btn_dvl.toolTip()
            # the shipped re-arming, unchanged
            w.set_engaged(False)
            w.set_dvl_use(True)
            assert w.dvl.tripped and "PROVEN" in w.reason, w.reason
            sim.prove()                       # a proof made after the latch
            w.set_dvl_use(False)
            w.set_dvl_use(True)
            assert not w.dvl.tripped and w.dvl.use, w.reason
            assert w.dvl.escalation_s == 0.0
            sim.engage()
            assert w._dvl_run_ok is True
        finally:
            if p is not None:
                p.close()
            sim.close()


def test_the_rung_takes_only_a_usable_agreeing_dvl():
    """N18. The station bridge's DVL rung under primary: opened only by a
    tagged tick that flew the DVL AND agrees with that tick's tag velocity
    (rung_admit); a flagged sample on a lost tick is carried on the last
    usable one for blend_hold_s; a disagreement closes it at once."""
    TC = _harness()
    TC._app()
    from rov_gui import theme
    from rov_gui.widgets.trajectory import TrajectoryWindow

    def lost(sim, w, v=(0.0, 0.0, 0.0), **kw):
        sim.step(fix=False, dvl=False, tick=False)
        w.on_dvl(_smp(v, sim.t, **kw))
        w.tick()

    # (a) + (b): opened by an agreeing DVL; a flag is bridged, then closes
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, _pilots, _logs, sim = _station(tmp)
        p = None
        try:
            st = []
            bus.mpc_status.connect(st.append)
            assert w._dvl_xy is not None, "an agreeing DVL opens the rung"
            w.fix.t_capture -= 5.0            # the tags are gone
            tiers = []
            for _ in range(14):
                lost(sim, w, fom=0.004)       # every report flagged
                tiers.append(w._bridge.tier)
            assert tiers[:8] == [SB.TIER_DVL] * 8, tiers
            assert SB.TIER_DVL not in tiers[10:], tiers
            assert w._dvl_xy is None
            assert st[0].dvl_state == "in_use" and st[0].dvl_code == "rung"
            assert st[0].dvl_w == 1.0
            p = TrajectoryWindow()
            p.add_status(st[0])
            assert p._dvl_css == p._state_btn_css(theme.OK)
            sb = w._run_meta()["dvl"]["station_bridge_dvl"]
            assert sb["dropouts_on_dvl"] == 1 and sb["held_ticks"] >= 7, sb
            assert sb["chain_breaks"].get("flag_fom", 0) >= 1, sb
            rows = _rows(w)
            rung = [r for r in rows if r["bridge_tier"] == "dvl"]
            assert rung and all(r["w_dvl"] == "1.000" and
                                r["dvl_code"] == "rung" for r in rung)
        finally:
            if p is not None:
                p.close()
            sim.close()
    # (c) the counterpart of the legacy rejected-DVL test: a false lock the
    # tags contradict on the last tagged ticks never opens the rung
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, _pilots, _logs, sim = _station(tmp)
        try:
            x0 = float(w._eta[0])
            for v in (0.1, 0.2, 0.25, 0.25, 0.25, 0.25):
                sim.step(dvl_v=(v, 0.0, 0.0))
            assert w._dvl_xy is None and not w.dvl.tripped
            w.fix.t_capture -= 5.0
            tiers = []
            for _ in range(8):
                sim.step(dvl_v=(0.25, 0.0, 0.0), fix=False)
                tiers.append(w._bridge.tier)
            assert SB.TIER_DVL not in tiers, tiers
            assert abs(float(w._eta[0]) - x0) < 1e-6
            assert w._bridge.meta()["dropouts_on_dvl"] == 0
            refused = w._run_meta()["dvl"]["station_bridge_dvl"][
                "rung_refusals"]
            assert refused.get("DVL innovation", 0) >= 1, refused
        finally:
            sim.close()
    # (d) a +0.25 m/s false lock starting 0.5 s before the tags go
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, _pilots, _logs, sim = _station(tmp)
        try:
            for _ in range(10):
                sim.step(dvl_v=(0.25, 0.0, 0.0))
            assert w._dvl_xy is None
            w.fix.t_capture -= 5.0
            for _ in range(6):
                sim.step(dvl_v=(0.25, 0.0, 0.0), fix=False)
                assert w._bridge.tier != SB.TIER_DVL
        finally:
            sim.close()
    # (e) a tier below the rung: the panel names it, never "held back"
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, _pilots, _logs, sim = _station(tmp)
        try:
            w._bridge.cfg["use_dvl"] = False
            st = []
            bus.mpc_status.connect(st.append)
            w.fix.t_capture -= 5.0
            for _ in range(3):
                sim.step(fix=False)
            assert st[-1].vel_src == "bridge" and st[-1].dvl_code == "bridge"
            assert st[-1].dvl_state == "no_tag" and st[-1].dvl_w == 0.0
            assert "station bridge on tier" in st[-1].dvl_note, st[-1].dvl_note
        finally:
            sim.close()
    # (f) a disagreement on a lost tick closes the rung at once
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, _pilots, _logs, sim = _station(tmp)
        try:
            w.fix.t_capture -= 5.0
            lost(sim, w)
            assert w._bridge.tier == SB.TIER_DVL
            w.dvl._hold_diff, w.dvl._r_slow = True, 0.08   # stage 1 holds
            lost(sim, w)
            assert w._bridge.tier != SB.TIER_DVL and w._dvl_xy is None
            cb = w._run_meta()["dvl"]["station_bridge_dvl"]["chain_breaks"]
            assert cb.get("dis_diff") == 1, cb
        finally:
            sim.close()


def test_the_panel_never_says_tripped_for_a_hold():
    """N19 (R9). A hold reads "held — returns by itself" on a DVL IN face,
    with the reason and the fade on the chip; a run not armed reads "not
    used this run", a tick without a tag state "no tag state"; a shadow aid
    logs no HELD line."""
    TC = _harness()
    TC._app()
    from rov_gui.widgets.trajectory import TrajectoryWindow

    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, _pilots, logs = _dvl_worker(tmp, reentry_s=0.0)
        sim = _Sim(w, TC)
        p = TrajectoryWindow()
        try:
            sim.prove()
            w.set_dvl_use(True)
            sim.engage()
            st = []
            bus.mpc_status.connect(st.append)
            for _ in range(20):
                sim.step(v=(0.10, 0.0, 0.0))
            for _ in range(40):
                sim.step(v=(0.10, 0.0, 0.0), dvl_v=(-0.10, 0.0, 0.0))
            held = [s for s in st if s.dvl_state == "held"]
            assert held, {s.dvl_state for s in st}
            assert all("returns by itself" in s.dvl_note for s in held)
            # (the first may be the re-acquired lock after the jump, still
            # flown on the held sample at w 1: "v:DVL"); the disagreement:
            dis = [s for s in held if s.dvl_code.startswith("dis_")]
            assert dis, [(s.dvl_code, s.dvl_w) for s in held]
            for s in (dis[0], dis[-1]):
                s.engaged, s.mode = True, "pid"
                p.add_status(s)
                assert p.btn_dvl.text() == "DVL IN"
                assert "HELD =" in p.btn_dvl.toolTip()
                chip = p.chip.text()
                assert ("v:TAG (DVL held: disagrees)" in chip
                        or "%↓ (disagrees)" in chip), (s.dvl_w, chip)
            for _ in range(60):               # agreeing again: the tail
                sim.step(v=(0.10, 0.0, 0.0))
            tail = [s for s in st if s.dvl_code == "dis_diff"
                    and "releasing" in s.dvl_note]
            assert tail, [s.dvl_note for s in st if s.dvl_code == "dis_diff"]
            out = [s for s in st if s.dvl_code.startswith("dis_")
                   and s.dvl_w == 0.0]
            assert out, [(s.dvl_code, s.dvl_w) for s in st[-60:]]
            out[0].engaged, out[0].mode = True, "pid"
            p.add_status(out[0])
            assert "v:TAG (DVL held: disagrees)" in p.chip.text(), \
                p.chip.text()
            # a tick with no tag state, engaged
            w.fix.t_capture -= 5.0
            w._bridge.cfg["enabled"] = False
            n0 = len(st)
            sim.step(fix=False)
            nt = [s for s in st[n0:] if s.dvl_code == "no_tag"]
            assert nt and nt[-1].dvl_state == "no_tag" and \
                "no tag state" in nt[-1].dvl_note, [(s.dvl_state, s.dvl_code)
                                                     for s in st[n0:]]
        finally:
            p.close()
            sim.close()
    # a run that is not armed
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, _pilots, logs = _dvl_worker(tmp, use=True, reentry_s=0.0)
        sim = _Sim(w, TC)
        p = TrajectoryWindow()
        try:
            sim.engage()                      # IN from the file, unproven
            assert w._dvl_run_ok is False
            st = []
            bus.mpc_status.connect(st.append)
            for k in range(240):              # the proof completes mid-run
                sim.step(v=_osc(k * 0.05))
            assert w.dvl.proven and st[-1].dvl_state == "not_armed"
            assert "not armed at engage" in st[-1].dvl_note, st[-1].dvl_note
            s = st[-1]
            s.engaged, s.mode = True, "pid"
            p.add_status(s)
            assert "v:TAG (DVL not used this run)" in p.chip.text(), \
                p.chip.text()
        finally:
            p.close()
            sim.close()
    # a SHADOW aid that disagrees logs nothing about holding
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, _pilots, logs = _dvl_worker(tmp, reentry_s=0.0)
        sim = _Sim(w, TC)
        try:
            sim.prove()
            sim.engage()                      # OUT: shadow
            for _ in range(60):
                sim.step(v=(0.10, 0.0, 0.0), dvl_v=(-0.10, 0.0, 0.0))
            assert w.dvl.residual_slow_m_s > 0.05
            assert not any("HELD" in m for _l, m in logs)
        finally:
            sim.close()


def test_a_refutation_at_engage_does_not_unarm_the_run():
    """N35. A fit refutation current at engage refuses the switch IN but no
    longer the run: the run is armed (the events.log engage line says
    ARMED), the tick holds the DVL while the refutation is current — the
    difference hold answers first (disagree_hold's order: difference,
    alignment, fit), then dis_fit while the fit alone refutes — and fades it
    in when it ends [first ok 6.8 s, coherent case]. The latches are OFF here
    to build the state: the test is about ARMING only. With them on, a 2x
    DVL latches through the scale evidence (N13) and a 180 deg one through
    the coherent evidence (N12) — that is the escalation's job, not a
    contradiction (fix round 1: the old "does not latch" line was vacuous)."""
    for name, fn, coh in (
            ("incoherent", lambda t: (2.0 * _osc(t)[0], 2.0 * _osc(t)[1],
                                      0.0), False),
            ("coherent", lambda t: (-_osc(t)[0], -_osc(t)[1], 0.0), True)):
        with tempfile.TemporaryDirectory() as tmp:
            TC, w, bus, _pilots, logs = _dvl_worker(
                tmp, reentry_s=0.0, agree_window_s=8.0, esc_scale_ratio=0.0,
                esc_latch_s=1e9, esc_repeat_n=0)
            sim = _Sim(w, TC)
            try:
                sim.prove()
                w.set_dvl_use(True)
                k = 0
                while k < 2000 and not (
                        w.dvl._fit_current(sim.t)
                        and w.dvl.agree.fit()["coherent"] is coh):
                    sim.step(v=_osc(k * 0.05), dvl_v=fn(k * 0.05))
                    k += 1
                f = w.dvl.agree.fit()
                assert w.dvl._fit_current(sim.t) and f["coherent"] is coh, \
                    (name, f)
                assert "REFUTES" in w.dvl.refusal(True)
                assert w.dvl.arm_refusal() == ""
                w.on_enable(True)
                w.set_engaged(True)
                w._warmup_left = 0
                assert w.engaged and w._dvl_run_ok is True, (name,
                                                             w._dvl_run_why)
                ev = "\n".join(p.read_text(encoding="utf-8")
                               for p in Path(tmp).rglob("events.log"))
                assert "DVL IN at engage — ARMED for this run" in ev, ev
                st = []
                bus.mpc_status.connect(st.append)
                sim.step(v=_osc(k * 0.05), dvl_v=fn(k * 0.05))
                assert st[-1].dvl_code == "dis_diff", (name, st[-1].dvl_code)
                assert w.dvl._fit_current(sim.t), name
                assert st[-1].dvl_w == 0.0, st[-1].dvl_w
                codes = []
                for j in range(400):          # honest again
                    sim.step(v=_osc((k + j) * 0.05))
                    codes.append(st[-1].dvl_code)
                assert "dis_fit" in codes, (name, sorted(set(codes)))
                assert st[-1].dvl_w == 1.0 and st[-1].vel_src == "dvl", \
                    (name, st[-1].dvl_code, st[-1].dvl_note)
                assert not w.dvl.tripped and w.dvl.n_trips == 0
                assert w._dvl_run_ok is True
            finally:
                sim.close()


def test_primary_names_the_keys_it_does_not_use():
    """N37. A legacy-only key tuned in the yaml would change nothing under
    primary: the setup line names it and what governs instead, a warning
    says so for a non-default value, and the meta records it."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, _bus, _pilots, logs = _dvl_worker(tmp, reentry_s=0.3)
        setup = [m for lvl, m in logs if m.startswith("dvl: built")]
        assert setup and "selection primary" in setup[-1]
        assert "reentry_s 0.3 (blend_xfade_s instead)" in setup[-1], setup[-1]
        assert "agree_trip_m_s 0.08" in setup[-1]
        warn = [m for lvl, m in logs if lvl == "warn" and
                "selection primary does not use it" in m]
        assert len(warn) == 1 and "dvl.reentry_s is 0.3" in warn[0], warn
        assert w.dvl.meta()["inactive_keys"]["reentry_s"] == 0.3
        TC, w, _bus, _pilots, logs = _dvl_worker(tmp, selection="legacy",
                                                 reentry_s=0.3)
        setup = [m for lvl, m in logs if m.startswith("dvl: built")]
        assert "selection LEGACY" in setup[-1]
        assert not any("does not use it" in m for _l, m in logs)


def test_a_proof_at_the_edge_of_the_tolerance_says_at_what_speed_it_holds():
    """N39. Primary's counterpart of the legacy edge-of-proof warning: the
    hold level (5 cm/s) and the escalation, not agree_trip_m_s."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, _bus, _pilots, logs = _dvl_worker(tmp)
        sim = _Sim(w, TC)
        try:
            c, s = math.cos(math.radians(10)), math.sin(math.radians(10))
            for k in range(240):
                v = _osc(k * 0.05)
                sim.step(v=v, tick=False,
                         dvl_v=(c * v[0] + s * v[1], -s * v[0] + c * v[1], 0))
            assert w.dvl.proven
            warn = [m for _l, m in logs if "holds the DVL back" in m]
            assert len(warn) == 1, warn
            assert "slow residual > 5 cm/s" in warn[0] and \
                "lasting 4 s, latches it" in warn[0], warn[0]
            assert "trips the difference monitor" not in warn[0]
            import re
            mm = re.search(r"above ([0-9.]+) m/s .* yaw by \+([0-9.]+) deg",
                           warn[0])
            assert mm and 0.20 < float(mm.group(1)) < 0.32, warn[0]
            assert 9.0 < float(mm.group(2)) < 13.0, warn[0]
        finally:
            sim.close()


def test_a_run_is_armed_for_the_dvl_at_engage_or_not_at_all_under_primary():
    """The primary twin of the (legacy-pinned) arming test: one run = one
    estimator binds the automation too — a proof completing mid-run does not
    move the velocity source; the panel says "not used this run" (never
    "held back") with the reason."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, _pilots, logs = _dvl_worker(tmp, use=True, reentry_s=0.0)
        sim = _Sim(w, TC)
        try:
            assert w.dvl.use and not w.dvl.proven
            sim.engage()
            assert w._dvl_run_ok is False
            assert "not armed" in w._dvl_run_why.lower()
            st = []
            bus.mpc_status.connect(st.append)
            for k in range(240):
                sim.step(v=_osc(k * 0.05))
            assert w.dvl.proven
            assert all(s.vel_src == "tag_fd" and s.dvl_w == 0.0 for s in st)
            assert st[-1].dvl_state == "not_armed"
            assert "not armed at engage" in st[-1].dvl_note, st[-1].dvl_note
            assert any("used from the next engagement" in m
                       for _l, m in logs)
            m = w._run_meta()["dvl"]
            assert m["armed_at_engage"] is False and m["rows"].get("dvl", 0) == 0
            w.set_engaged(False)
            sim.engage()
            assert w._dvl_run_ok is True
            st.clear()
            for _ in range(5):
                sim.step(v=(0.05, 0.0, 0.0))
            assert all(s.vel_src == "dvl" and s.dvl_w == 1.0 for s in st)
        finally:
            sim.close()


# ================================================ fix round 1 (2026-10-16)
# DTF fix1/ (IMPLEMENTATION.md "Fix round 1"): the verifiers' findings.
# [bracketed] numbers are measured on THIS implementation by the same
# scenario (DTF fix1/out [측정]).
def _glitch_run(m, dur, warm_s=60.0, after_s=10.0, flagged=False, seed=2,
                k=1.0):
    """k x `_osc` motion with a full 60-s fit window of honest pairs (as in
    flight), then ONE glitch: +m m/s on the DVL's u for dur s (its reports
    carrying fom 6 mm/s when ``flagged``), then honest for after_s."""
    d = _paid()
    f = _Feed(d, seed=seed, blend=True)

    def mot(t):
        return tuple(k * x for x in _osc(t))
    f.run(int(round(warm_s / 0.05)), mot, mot)
    t0 = f.t

    def dv(t):
        v = mot(t)
        return (v[0] + m, v[1], v[2]) if t0 < t <= t0 + dur + 1e-9 else v
    kw = ((lambda t: {"fom": 0.006} if t0 < t <= t0 + dur + 1e-9 else {})
          if flagged else None)
    f.run(int(round((dur + after_s) / 0.05)), dv, mot, s_kw=kw)
    return d, f, t0, t0 + dur


def test_a_short_glitch_while_moving_is_held_not_latched():
    """Fix round 1 (DTF verify/tests.md §6.1, parity.md §4): ONE large short
    DVL glitch while moving is a transient — the blend keeps it out (jump
    gate, gross check, or the fom flag), nothing latches, the proof stays,
    and the DVL is used again within a second [ok 0.05-0.45 s, w 1 by
    0.5-0.9 s]. The slow DVL power used to take the glitch UNclipped: its
    4-s memory held scale_ratio over 1.5 and latched "persistently ... the
    DVL reads 2.1x the motion the tags see" 4.05 s after the onset."""
    for m, dur, flagged in ((0.6, 1.0, False), (0.8, 0.5, False),
                            (0.8, 1.0, False), (0.5, 0.5, False),
                            (0.6, 1.0, True)):
        d, f, t0, te = _glitch_run(m, dur, flagged=flagged)
        case = (m, dur, flagged)
        assert d.n_trips == 0 and not d.tripped and d.proven, \
            (case, d.last_trip)
        assert d.esc_max < 0.5 and not d.ep_log, (case, d.esc_max)
        post = [r for r in f.log if r[0] > te]
        ok = next(r[0] - te for r in post if r[1] == "ok")
        assert ok <= 1.0, (case, ok)
        assert all(r[1] == "ok" and r[5] == 1.0 for r in post
                   if r[0] - te >= 1.5), \
            (case, sorted({r[1] for r in post if r[0] - te >= 1.5}))
        assert {r[1] for r in f.log if r[0] > t0} <= {
            "ok", "jump", "gross", "flag_reacq", "flag_fom"}, \
            (case, {r[1] for r in f.log if r[0] > t0})
    # Where the tags see just over 5 cm/s rms (6 / 4 cm/s), a 1.5-s glitch:
    # a glitch pair feeds no scale evidence at all [bucket 0.0; clipped but
    # kept, it left 2.9 s — 0.73 of the latch — DTF fix1/out/
    # glitch_slowmotion_aid.json]. (Its fit-memory hold is the next test's.)
    d, f, t0, te = _glitch_run(0.5, 1.5, k=0.5)
    assert d.n_trips == 0 and not d.tripped and d.proven, d.last_trip
    assert d.esc_max < 0.5 and not d.ep_log, d.esc_max


def test_a_long_glitch_is_held_by_the_fit_and_returns_without_a_latch():
    """Fix round 1, the cost that remains (DTF IMPLEMENTATION.md Fix round
    1): a glitch long enough to dominate the fit's 60-s window (+0.8 m/s for
    1.5 s, 30 pairs of 0.8 m/s regressor against 1200 honest ones) makes the
    fit refute INCOHERENTLY, and the dis_fit hold (R1's refuting-fit hold)
    keeps the DVL out until those pairs leave the window [59.3 s]. It never
    latches and it returns by itself; the proof is kept."""
    d, f, t0, te = _glitch_run(0.8, 1.5, after_s=62.0)
    assert d.n_trips == 0 and not d.tripped and d.proven, d.last_trip
    post = [r for r in f.log if r[0] > te]
    assert any(r[1] == "dis_fit" for r in post)
    back = max(r[0] - te for r in post if r[1] != "ok")
    assert back <= 61.0, back                       # the window, plus a tick
    assert post[-1][1] == "ok" and post[-1][5] == 1.0, post[-1]


def test_alignment_only_episodes_do_not_count_toward_the_repeated_latch():
    """Fix round 1 (DTF verify/REPLAY.md §8 item 1): every counted episode on
    the recordings was ALIGNMENT evidence alone, and on mpc_192708 two came
    55 s apart, the second while the DVL was the better sensor — one more
    and the "repeatedly" latch would have thrown the proof away. Three 1.5-s
    spells of a DVL reading the motion rotated 60 deg, 20 s apart: each one
    is HELD (dis_align) and fills the bucket [max 0.95 s], none counts
    toward esc_repeat_n — no warning, no latch [round 0: warning at 21.4 s,
    latched "repeatedly" at the third]; they are recorded as alignment-only
    episodes. A false lock that keeps coming back still latches (N30: its
    episodes carry difference evidence)."""
    d = _paid()
    f = _Feed(d, blend=True)
    f.run(1200, _osc, _osc)
    t0 = f.t
    c, s = math.cos(math.radians(60.0)), math.sin(math.radians(60.0))

    def dv(t):
        v = _osc(t)
        if t - t0 < 60.0 and (t - t0) % 20.0 < 1.5:
            return (c * v[0] - s * v[1], s * v[0] + c * v[1], 0.0)
        return v
    f.run(1600, dv, _osc)
    assert d.n_trips == 0 and not d.tripped and d.proven, d.last_trip
    assert d.warn_log == [] and d.ep_log == [], (d.warn_log, d.ep_log)
    assert len(d.ep_log_align) == 3, d.ep_log_align
    assert 0.5 <= d.esc_max < 2.0, d.esc_max
    spells = [k for k in range(3)
              if any(r[1] == "dis_align" for r in f.log
                     if t0 + 20.0 * k < r[0] <= t0 + 20.0 * k + 4.0)]
    assert spells == [0, 1, 2], spells
    m = d.meta()["escalation"]
    assert m["episodes"] == 0 and m["episodes_alignment_only"] == 3, m
    assert m["episode_kinds"] == ["difference", "fit", "scale"], m


def test_the_held_and_released_lines_pair_up():
    """Fix round 1 (DTF verify/tests.md §6.4, REPLAY.md §8 item 2, review.md
    F3): a hold is remembered only once its HELD line is written. A hold
    starting within 2 s of the last HELD line is announced once the 2 s are
    up, or — if it ends sooner — not at all; so the log reads HELD,
    released, HELD, released, never a "released" without its HELD. The
    release line is written only on a tick that flies the DVL back in
    (ok), never on a gross one [round 0: 19 released for 14 HELD on the
    armed recordings]."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, _pilots, logs = _dvl_worker(tmp, reentry_s=0.0)
        sim = _Sim(w, TC)
        try:
            sim.prove()
            w.set_dvl_use(True)
            sim.engage()
            d = w.dvl
            real = d.usable
            on = {"hold": False}

            def usable(t):
                if on["hold"]:
                    return (False, "DVL disagrees with the tag velocity "
                            "(slow residual 6.0 > 5.0 cm/s)", "dis_diff")
                return real(t)
            d.usable = usable

            def run(n, hold):
                on["hold"] = hold
                for _ in range(n):
                    sim.step(v=(0.10, 0.0, 0.0))
            run(20, False)
            run(10, True)       # A: HELD at once
            run(10, False)      # A released
            run(6, True)        # B: 1.0 s after A's line, 0.3 s long: silent
            run(10, False)
            run(40, True)       # C: 1.8 s after A's line, 2 s long
            run(20, False)      # C released
            lines = [(lvl, m) for lvl, m in logs
                     if m.startswith("dvl: HELD (not latched)")
                     or m.startswith("dvl: hold released")]
            assert [m.startswith("dvl: HELD") for _l, m in lines] == \
                [True, False, True, False], lines
            assert [lvl for lvl, _m in lines] == ["warn", "info"] * 2
            # a gross tick does not release an announced hold; ok does
            w._dvl_dis_logged = "dis_diff"
            n0 = len(logs)
            w._dvl_after_step_primary({}, {"dvl_code": "gross",
                                           "dvl_why": "DVL gross"}, sim.t)
            assert not any("hold released" in m for _l, m in logs[n0:])
            assert w._dvl_dis_logged == "dis_diff"
            w._dvl_after_step_primary({}, {"dvl_code": "ok"}, sim.t)
            assert any("hold released" in m for _l, m in logs[n0:])
            assert w._dvl_dis_logged == ""
        finally:
            sim.close()


def test_the_panel_says_returns_by_itself_only_when_it_can():
    """Fix round 1 (DTF verify/REPLAY.md §8 item 4, review.md F2): a lost
    lock reads "— returns by itself" only when the DVL would be used again
    with the lock back. OUT (shadow) and IN-but-unproven keep the plain
    words; a run the worker did not arm reads "not used this run" with its
    reason. The held note's episode count drops "of 0" (repeat latch off)
    and older onsets than the window."""
    t = 30.0
    d = _paid()
    d.note_sample(_smp((0.1, 0.0, 0.0), t, valid=False))
    st, note = d.state(t, used=False, code="no_lock", w=0.0)
    assert st == DV.ST_NO_LOCK and note.endswith("— returns by itself"), note
    d.use = False
    st, note = d.state(t, used=False, code="out", w=0.0)
    assert st == DV.ST_NO_LOCK and "returns by itself" not in note, note
    u = DV.DvlAid(dict(MOUNT))                # IN, not proven
    u.use = True
    u.note_sample(_smp((0.1, 0.0, 0.0), t, valid=False))
    st, note = u.state(t, used=False, code="no_lock", w=0.0)
    assert st == DV.ST_NO_LOCK and "returns by itself" not in note, note
    z = _paid(esc_repeat_n=0)
    z.note_sample(_smp((0.1, 0.0, 0.0), t))
    z._esc_onsets.append(t - 1.0)
    z._mon_t = t
    st, note = z.state(t, used=False, why="DVL quality flag: fom (4.0 > 3.0 "
                       "mm/s)", code="flag_fom", w=0.0)
    assert st == DV.ST_HELD and " of 0" not in note, note
    e = _paid()
    e._esc_onsets.append(t - 1.0)
    e._mon_t = t
    assert e.episodes_recent == 1
    e._mon_t = t + 60.5                       # frozen bucket, the window moves
    assert e.episodes_recent == 0
    # the worker: a run NOT armed (IN from the file, unproven) losing lock
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, _pilots, _logs = _dvl_worker(tmp, use=True, reentry_s=0.0)
        sim = _Sim(w, TC)
        try:
            sts = []
            bus.mpc_status.connect(sts.append)
            sim.engage()
            assert w._dvl_run_ok is False
            for _ in range(20):
                sim.step(v=(0.05, 0.0, 0.0), valid=False)
            assert sts[-1].dvl_state == "not_armed", sts[-1].dvl_state
            assert "not armed at engage" in sts[-1].dvl_note, sts[-1].dvl_note
            assert "returns by itself" not in sts[-1].dvl_note
        finally:
            sim.close()
    # ...and a SHADOW one
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, _pilots, _logs = _dvl_worker(tmp, reentry_s=0.0)
        sim = _Sim(w, TC)
        try:
            sts = []
            bus.mpc_status.connect(sts.append)
            for _ in range(20):
                sim.step(v=(0.05, 0.0, 0.0), valid=False)
            assert sts[-1].dvl_state == "no_lock" and not sts[-1].dvl_use
            assert "returns by itself" not in sts[-1].dvl_note, \
                sts[-1].dvl_note
        finally:
            sim.close()


def test_the_fade_catch_up_is_bounded_and_bit_identical():
    """Fix round 1 (DTF verify/review.md F1): the ticks without a tag state
    replay the fade only until w can have reached its (fixed) target, so an
    hour without tags (72 000 ticks) costs one ordinary tick [2.0 ms before]
    and the answer is the same, bit for bit, as replaying every one."""
    def two_steps(w0_ok, n_skip):
        """w after: a first step (ok = w0_ok: w snaps to 1 or 0), then a
        step 1 s later toward the other target with n_skip skipped ticks."""
        b = DV.DvlBlend()
        b.step(0.0, np.zeros(3), np.array([0.1, 0.0, 0.0]), 0.0, w0_ok)
        if w0_ok:                             # 1 -> 0: the sample went bad
            return b.step(1.0, np.array([0.02, 0.0, 0.0]), None, None, False,
                          n_skip=n_skip)[1]
        return b.step(1.0, np.array([0.02, 0.0, 0.0]),
                      np.array([0.1, 0.0, 0.0]), 1.0, True,
                      n_skip=n_skip)[1]       # 0 -> 1: a good sample

    stp = DV.DvlBlend().dt / DV.DvlBlend().xfade_s
    for w0_ok in (True, False):
        for n_skip in (0, 3, 9, 12, 72_000):
            w = 1.0 if w0_ok else 0.0         # the per-tick law, every tick
            target = 0.0 if w0_ok else 1.0
            for _ in range(1 + n_skip):
                if abs(target - w) <= stp + 1e-6:
                    w = target
                else:
                    w = w + stp if target > w else w - stp
            assert two_steps(w0_ok, n_skip) == w, (w0_ok, n_skip, w)
    t0 = time.perf_counter()
    assert two_steps(True, 2_000_000) == 0.0
    assert time.perf_counter() - t0 < 0.01     # uncapped: ~60 ms [유도]


def test_legacy_keeps_the_jump_reference_on_samples_primary_would_flag():
    """R5, fix round 1 (DTF verify/tests.md §6.2: a mutant that ran the R3
    flags under legacy survived the whole suite). Under ``selection:
    legacy`` the four R3 flags do not exist: a report with fom 4 mm/s, 3
    beams, after a no-lock report, or 0.25 m off the altitude median is a
    GOOD sample that becomes the jump gate's reference (the shipped
    _last_good), so the next report 0.14 m/s further on is not a "jump"
    (limit 0.15 m/s at 0.05 s). Under primary the same reports are
    flagged, the reference stays on the last unflagged one, and that next
    report — 0.28 m/s from it after 0.10 s, limit 0.25 — IS a jump (after a
    no-lock report the reference is 0.15 s old, limit 0.35: not a jump)."""
    for name, bad_kw, pre in (
            ("fom", {"fom": 0.004}, None),
            ("beams", {"n_beams": 3}, None),
            ("reacq", {}, "nolock"),
            ("alt", {"alt": 0.75}, None)):
        out = {}
        for sel in ("legacy", "primary"):
            d = _proven_aid(reentry_s=0.0, selection=sel)
            d.use = True
            t = 10.0
            for _ in range(12):                 # 0.6 s of good reports
                t += 0.05
                d.note_sample(_smp((0.10, 0.0, 0.0), t))
            if pre == "nolock":
                t += 0.05
                d.note_sample(_smp(None, t, valid=False))
            t += 0.05
            d.note_sample(_smp((0.24, 0.0, 0.0), t, **bad_kw))
            lg = d._last_good
            flag_b = d._s_flag
            t += 0.05
            d.note_sample(_smp((0.38, 0.0, 0.0), t, **{
                k: v for k, v in bad_kw.items() if k == "alt"}))
            out[sel] = (lg, flag_b, d._s_ok, d._s_why, dict(d.n_flag))
        lg, flag_b, ok, why, nflag = out["legacy"]
        assert abs(float(lg[1][0]) - 0.24) < 1e-12, (name, lg)
        assert flag_b == "" and ok and why == "" and nflag == {}, \
            (name, out["legacy"])
        lg, flag_b, ok, why, nflag = out["primary"]
        assert abs(float(lg[1][0]) - 0.10) < 1e-12, (name, lg)
        assert flag_b.startswith("DVL quality flag: "), (name, flag_b)
        if pre is None:
            assert not ok and why.startswith("DVL velocity jump"), \
                (name, out["primary"])


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
