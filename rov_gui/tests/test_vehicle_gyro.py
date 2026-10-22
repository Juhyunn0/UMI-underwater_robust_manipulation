"""The RAW_IMU gyro integral and the Bar30 depth VehicleImu carries for
--dvl_pose (2026-10-18), from the hardware reader and from the demo plant.

Every record here is SYNTHETIC (a constant rate, a ramp, a gap) — chosen so
the expected integral is exact, not measured.
"""

from __future__ import annotations

import math
import tempfile

import numpy as np
import pytest

from rov_gui.state import now


def _vehicle_worker():
    """A bare VehicleWorker (no link, no thread: the test_att_age pattern),
    with what the drain loop and _publish_imu read."""
    from rov_gui.backends.hardware import VehicleWorker, WATER_RHO
    from rov_gui.bus import DataBus
    from rov_gui.leak import LeakMonitor
    from rov_gui.tests.test_control import _app

    _app()
    bus = DataBus()
    logs, imus = [], []
    bus.log.connect(lambda lvl, m: logs.append((lvl, m)))
    bus.vehicle_imu.connect(imus.append)

    class O:
        mavlink_transport = "udp"

    w = VehicleWorker.__new__(VehicleWorker)
    w.bus, w.opts = bus, O()
    w._leak, w._leak_reported, w._press_reported = LeakMonitor(), 0, 0.0
    w.leak_cfg_fn = None
    w._latest, w.logger = {}, None
    w._rho = WATER_RHO
    w._aa_target = (1, 1)
    return w, logs, imus


def _rec(tu, g, t_host=None, sysid=1, comp=1):
    return {"msg_type": "RAW_IMU", "time_usec": tu, "xgyro": g[0],
            "ygyro": g[1], "zgyro": g[2], "xacc": 0, "yacc": 0, "zacc": -1000,
            "t_host": now() if t_host is None else t_host,
            "_srcsys": sysid, "_srccomp": comp}


# =============================================================================
# the hardware reader
# =============================================================================
def test_a_constant_rate_integrates_to_rate_times_time():
    w, _logs, imus = _vehicle_worker()
    g = (10.0, -5.0, 20.0)                     # mrad/s
    for k in range(41):                        # 40 intervals of 50 ms = 2 s
        w._note_raw_imu(_rec(1_000_000 + 50_000 * k, g))
    w._publish_imu()
    v = imus[-1]
    np.testing.assert_allclose(v.gyro_int, (0.020, -0.010, 0.040), atol=1e-12)
    assert v.n_gyro == 41 and v.n_gyro_gaps == 0
    assert v.gyro_raw == pytest.approx((0.010, -0.005, 0.020))


def test_a_ramp_is_integrated_exactly_by_the_trapezoid():
    w, _logs, imus = _vehicle_worker()
    for k in range(21):                        # z rate 0 -> 200 mrad/s over 1 s
        w._note_raw_imu(_rec(5_000_000 + 50_000 * k, (0.0, 0.0, 10.0 * k)))
    w._publish_imu()
    assert imus[-1].gyro_int[2] == pytest.approx(0.5 * 0.2 * 1.0, abs=1e-12)


def test_a_gap_and_a_reboot_are_counted_not_integrated():
    w, _logs, imus = _vehicle_worker()
    g = (0.0, 0.0, 100.0)
    w._note_raw_imu(_rec(1_000_000, g))
    w._note_raw_imu(_rec(1_050_000, g))        # +5 mrad
    w._note_raw_imu(_rec(1_850_000, g))        # 0.8 s: a gap, not integrated
    w._note_raw_imu(_rec(1_900_000, g))        # +5 mrad
    w._note_raw_imu(_rec(200_000, g))          # time_usec back: a reboot
    w._note_raw_imu(_rec(250_000, g))          # +5 mrad
    w._note_raw_imu(_rec(250_000, g))          # a repeat: dt 0
    w._publish_imu()
    v = imus[-1]
    assert v.gyro_int[2] == pytest.approx(0.015, abs=1e-12)
    assert v.n_gyro_gaps == 3 and v.n_gyro == 7


def test_one_record_is_a_starting_point_not_an_integral():
    w, _logs, imus = _vehicle_worker()
    w._publish_imu()
    assert imus[-1].gyro_int is None and imus[-1].n_gyro == 0
    w._note_raw_imu(_rec(1_000_000, (1.0, 2.0, 3.0)))
    w._publish_imu()
    assert imus[-1].gyro_int is None and imus[-1].n_gyro == 1
    w._note_raw_imu(_rec(1_050_000, (1.0, 2.0, 3.0)))
    w._publish_imu()
    assert imus[-1].gyro_int is not None


def test_a_record_that_is_no_reading_is_skipped_and_said_once():
    w, logs, imus = _vehicle_worker()
    w._note_raw_imu(_rec(1_000_000, (0.0, 0.0, 10.0)))
    w._note_raw_imu(_rec(None, (0.0, 0.0, 10.0)))
    w._note_raw_imu(_rec(1_050_000, (0.0, float("nan"), 10.0)))
    w._note_raw_imu(_rec(1_100_000, (0.0, 0.0, 10.0)))
    w._publish_imu()
    assert w._n_gyro_bad == 2
    assert sum("RAW_IMU record without" in m for _l, m in logs) == 1
    # the two bad records are not a gap: 1.0 -> 1.1 s integrates
    assert imus[-1].gyro_int[2] == pytest.approx(0.001, abs=1e-12)


def test_only_the_vehicles_raw_imu_is_integrated():
    w, _logs, imus = _vehicle_worker()

    class Feed:
        def __init__(self, recs):
            self.recs = recs

        def drain(self):
            r, self.recs = self.recs, []
            return r

    recs = [_rec(1_000_000 + 50_000 * k, (0.0, 0.0, 10.0)) for k in range(5)]
    # another component's IMU in between: another body
    recs += [_rec(1_000_000 + 50_000 * k, (0.0, 0.0, 999.0), comp=194)
             for k in range(5)]
    w.logger, w._log, w._mpc = Feed(recs), None, True
    w._last_tel = w._last_link = float("inf")
    w.tick()
    v = imus[-1]
    assert v.n_gyro == 5
    assert v.gyro_int[2] == pytest.approx(0.002, abs=1e-12)


def test_the_gyro_integral_carries_the_newest_records_host_arrival():
    """t_gyro is when the NEWEST record arrived (the staleness the pose
    judges), not the first one's or the publish moment's."""
    w, _logs, imus = _vehicle_worker()
    w._note_raw_imu(_rec(1_000_000, (0.0, 0.0, 10.0), t_host=5000.0))
    w._note_raw_imu(_rec(1_050_000, (0.0, 0.0, 10.0), t_host=5000.07))
    w._publish_imu()
    assert imus[-1].t_gyro == 5000.07


def test_the_gyro_sample_moment_maps_through_the_attitude_clock():
    """t_gyro_v: the record's time_usec on the host clock through
    ATTITUDE's t_valid / time_boot_ms map — a queued link delays the
    arrival, not the sample. No map, or a mapped time more than 2 s from
    the map's own arrival (a GPS-epoch time_usec, a stale map), is None."""
    w, _logs, imus = _vehicle_worker()
    w._note_raw_imu(_rec(1_000_000, (0.0, 0.0, 10.0), t_host=5000.0))
    w._publish_imu()
    assert imus[-1].t_gyro_v is None                     # no ATTITUDE yet
    # boot clock 1.000 s <-> host 4999.990 (t_valid of the ATTITUDE sample)
    w._latest["ATTITUDE"] = {"msg_type": "ATTITUDE", "time_boot_ms": 1000,
                             "t_valid": 4999.990, "t_host": 5000.02}
    w._note_raw_imu(_rec(1_050_000, (0.0, 0.0, 10.0), t_host=5000.30))
    w._publish_imu()
    v = imus[-1]
    assert v.t_gyro == 5000.30                           # the late arrival
    assert v.t_gyro_v == pytest.approx(4999.990 + 0.050, abs=1e-9)
    # a time_usec on another clock (GPS epoch microseconds): no mapping
    w._note_raw_imu(_rec(1_700_000_000_000_000, (0.0, 0.0, 10.0),
                         t_host=5000.35))
    w._publish_imu()
    assert imus[-1].t_gyro_v is None
    # an ATTITUDE without its boot stamp: no mapping
    w._latest["ATTITUDE"] = {"msg_type": "ATTITUDE", "t_valid": 4999.99,
                             "t_host": 5000.02}
    w._note_raw_imu(_rec(1_100_000, (0.0, 0.0, 10.0), t_host=5000.40))
    w._publish_imu()
    assert imus[-1].t_gyro_v is None


def test_dvl_pose_on_the_hardware_refuses_a_raw_imu_below_5_hz_at_launch():
    from rov_gui import __main__ as M

    p = M.build_parser()

    def check(*argv):
        return M.check_mavlink_rates(p.parse_args(list(argv)))

    for mode in ("--dvl_pose", "--dvl-pose=shadow"):
        lvl, txt = check("--source", "hw", mode, "--mavlink-imu-rate", "4")
        assert lvl == "refuse" and "needs RAW_IMU at >= 5 Hz" in txt, txt
        assert check("--source", "hw", mode, "--mavlink-imu-rate", "5") \
            is None
        assert check("--source", "hw", mode) is None     # the 20 Hz default
    # the legacy request resolves RAW_IMU too
    lvl, txt = check("--source", "hw", "--dvl_pose", "--mavlink-rate", "2")
    assert lvl == "refuse" and "resolves to 2 Hz" in txt, txt
    # not this check's business: no --dvl_pose, --dvl_pose off, the demo
    assert check("--source", "hw", "--mavlink-imu-rate", "4") is None
    assert check("--source", "hw", "--dvl_pose", "off",
                 "--mavlink-imu-rate", "4") is None
    lvl, txt = check("--dvl_pose", "--mavlink-imu-rate", "4")
    assert lvl == "warn" and "does nothing" in txt, txt
    # nothing requested at all: said, with what the pose needs
    lvl, txt = check("--source", "hw", "--dvl_pose", "--no-mavlink-set-rates",
                     "--mavlink-att-rate", "100")
    assert lvl == "warn" and "NOTHING is requested" in txt \
        and "needs RAW_IMU at >= 5 Hz" in txt, txt


def test_the_bar30_depth_is_its_own_record_and_not_clamped():
    from rov_gui.backends.hardware import G, SEA_LEVEL_HPA

    w, _logs, imus = _vehicle_worker()
    t = now()
    w._latest["SCALED_PRESSURE2"] = {"msg_type": "SCALED_PRESSURE2",
                                     "press_abs": SEA_LEVEL_HPA + 49.0,
                                     "t_host": t}
    w._publish_imu()
    v = imus[-1]
    assert v.depth_sp2_m == pytest.approx(49.0 * 100.0 / (w._rho * G))
    assert v.t_sp2 == t
    # above the surface (a surface pressure below 1013.25 hPa): negative,
    # where depth_m clamps at 0 — a pose's z must not sit still there
    w._latest["SCALED_PRESSURE2"]["press_abs"] = SEA_LEVEL_HPA - 5.0
    w._publish_imu()
    assert imus[-1].depth_sp2_m < 0.0
    # no pressure is no depth; and the enclosure barometer is never one
    w._latest["SCALED_PRESSURE2"]["press_abs"] = float("nan")
    w._publish_imu()
    assert imus[-1].depth_sp2_m is None
    del w._latest["SCALED_PRESSURE2"]
    w._latest["SCALED_PRESSURE"] = {"msg_type": "SCALED_PRESSURE",
                                    "press_abs": SEA_LEVEL_HPA + 49.0,
                                    "t_host": t}
    w._publish_imu()
    assert imus[-1].depth_sp2_m is None and imus[-1].t_sp2 is None


# =============================================================================
# the demo plant
# =============================================================================
def _demo(tmp, **opts):
    from rov_gui.backends.demo import DemoVehicleWorker
    from rov_gui.bus import DataBus
    from rov_gui.tests import test_control as TC

    TC._app()
    Opts = TC._test_opts(tmp)
    for k, v in opts.items():
        setattr(Opts, k, v)
    bus = DataBus()
    got = {"imu": [], "fix": [], "dvl": []}
    bus.vehicle_imu.connect(got["imu"].append)
    bus.nav_fix.connect(got["fix"].append)
    bus.dvl.connect(got["dvl"].append)
    w = DemoVehicleWorker(bus, opts=Opts())
    w.setup()
    return w, got


def _yawing(w, rate=0.4):
    """Hold a yaw command so the plant turns."""
    from rov_gui.state import PilotInput

    w.enabled, w.armed, w.mode = True, True, "MANUAL"
    w.pilot = PilotInput(yaw=rate, source="test")


def test_the_demo_gyro_is_the_plants_rotation_and_drifts_by_its_bias():
    with tempfile.TemporaryDirectory() as tmp:
        w, got = _demo(tmp, mpc=True)
        _yawing(w)
        for _ in range(60):
            w.tick()
        imus = got["imu"]
        a, b = imus[5], imus[-1]
        # roll/pitch swing a little: the Euler yaw rate is (q s + r c)/c
        # — with the plant's small angles, its z integral is the yaw change
        dz = b.gyro_int[2] - a.gyro_int[2]
        dyaw = math.atan2(math.sin(b.yaw - a.yaw), math.cos(b.yaw - a.yaw))
        assert abs(dyaw) > 0.3
        assert dz == pytest.approx(dyaw, abs=2e-3)
    with tempfile.TemporaryDirectory() as tmp:
        w, got = _demo(tmp, mpc=True, demo_gyro_bias_dps=0.5)
        for _ in range(41):
            w.tick()
        imus = got["imu"]
        # the first snapshot is the starting point (gyro_int None, the
        # hardware's one-record rule); at rest the z integral is the bias
        # alone: 0.5 deg/s x (40 - 1) ticks x 1/hz
        assert imus[0].gyro_int is None and imus[1].gyro_int is not None
        expect = math.radians(0.5) * (len(imus) - 2) / w.hz
        assert imus[-1].gyro_int[2] - imus[1].gyro_int[2] == pytest.approx(
            expect, rel=1e-9)
        assert imus[-1].depth_sp2_m == pytest.approx(imus[-1].depth_m)


def test_a_demo_tag_blackout_withholds_only_the_navfix():
    from rov_gui.backends.demo import parse_tag_blackout

    assert parse_tag_blackout(None) == ()
    with pytest.raises(ValueError):
        parse_tag_blackout([(5.0, 1.0)])
    with tempfile.TemporaryDirectory() as tmp:
        w, got = _demo(tmp, mpc=True, demo_tag_blackout=[(0.0, 1e9)])
        for _ in range(10):
            w.tick()
        assert not got["fix"] and len(got["imu"]) == 10
        assert w.n_fix_withheld == 10
    with tempfile.TemporaryDirectory() as tmp:
        w, got = _demo(tmp, mpc=True, demo_tag_blackout=[(1e8, 1e9)])
        for _ in range(10):
            w.tick()
        assert len(got["fix"]) == 10


def test_the_demo_dvl_reads_at_its_mount_and_the_station_recovers_the_origin():
    """With the DEPLOYED mount (config/hw_mpc.yaml dvl.mount_t_flu, a 0.232 m
    arm) the synthetic DVL reports the velocity of ITS point, so the
    station's lever-arm correction returns the body-origin velocity — not a
    false sway of ~0.232 r."""
    from rov_gui.control import dvl as DV

    from pathlib import Path

    deployed = str(Path(__file__).resolve().parents[2] / "config" /
                   "hw_mpc.yaml")
    with tempfile.TemporaryDirectory() as tmp:
        w, got = _demo(tmp, mpc=True, dvl=True, mpc_config=deployed)
        assert w.dvl_on
        R, r = w._dvl_R, w._dvl_r
        assert abs(r[0]) > 0.2                 # the deployed arm, not identity
        _yawing(w, rate=0.8)
        for _ in range(80):
            w.tick()
        res = []
        for s in got["dvl"][-20:]:
            imu = next(i for i in reversed(got["imu"]) if i.stamp <= s.t_valid
                       + 1e-6)
            omega = (imu.p, imu.q, imu.r)
            vb = DV.body_velocity(s.v_dvl, R, r, omega)
            res.append(vb[1])                   # sway of the body origin
        r_now = got["imu"][-1].r
        assert abs(r_now) > 0.3
        # the plant has no sway command: what is left is the 3 mm/s noise
        assert abs(float(np.mean(res))) < 0.004, (np.mean(res), 0.232 * r_now)


def test_the_bar30_sample_moment_maps_through_the_attitude_clock():
    """t_sp2_v: SCALED_PRESSURE2.time_boot_ms on the host clock — the DVL
    pose differences the depth on it (queued records arrive ~1 ms apart)."""
    from rov_gui.backends.hardware import SEA_LEVEL_HPA

    w, _logs, imus = _vehicle_worker()
    sp2 = {"msg_type": "SCALED_PRESSURE2", "press_abs": SEA_LEVEL_HPA + 49.0,
           "time_boot_ms": 1100, "t_host": 5000.15}
    w._latest["SCALED_PRESSURE2"] = sp2
    w._publish_imu()
    assert imus[-1].t_sp2_v is None                      # no ATTITUDE yet
    w._latest["ATTITUDE"] = {"msg_type": "ATTITUDE", "time_boot_ms": 1000,
                             "t_valid": 4999.99, "t_host": 5000.02}
    w._publish_imu()
    assert imus[-1].t_sp2 == 5000.15
    assert imus[-1].t_sp2_v == pytest.approx(5000.09, abs=1e-9)
    sp2["time_boot_ms"] = 9000                           # after its arrival
    w._publish_imu()
    assert imus[-1].t_sp2_v is None
    del sp2["time_boot_ms"]
    w._publish_imu()
    assert imus[-1].t_sp2_v is None
