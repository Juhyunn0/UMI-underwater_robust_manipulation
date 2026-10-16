#!/usr/bin/env python3
"""test_att_age.py — the attitude's VEHICLE-clock age (2026-10-15), offline.

Source: RL_controller/logs/eval/pool_trips_1013_rootcause (rootcause_results.json, fix-design station items 2, 3, 6;
reference implementation fix-design/scripts/age_guard_replay.py). On 2026-10-13 the rl_pwm policy flew on an ATTITUDE
0.43 s old at the median behind a vehicle-side MAVLink queue that the station's parse-time imu_age (~0.05 s) could not
see. The operator approved (2026-10-15): (1a) measure the age on the vehicle clock, (1b) an AGE interlock + engage
refusal for the pwm LOWs only, (1d) a banner detector.

    cd <repo> && QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/robust/bin/python -m pytest \\
        rov_gui/tests/test_att_age.py -q -p no:cacheprovider

What is pinned:

1. c3_camera/mavlink_log.VehicleClockMap on synthetic streams: it settles on the least-delayed sample and reads a
   vehicle-side queue as age; it follows a vehicle clock drifting up to 300 ppm either way (and over-reads, never
   under-reads, past it); it resets on a reboot (time_boot_ms back; or the host clock running ahead with no backward
   step) and NOT on an outage the vehicle clock ran through or a duplicate stamp; a map started inside a queue
   under-reads until it drains (the limit age_map_min_s covers). MavlinkLogger stamps every autopilot ATTITUDE with
   t_valid / vclk_t0 in the reader (UDP and REST), never another component's, and its summary carries the map.
2. hardware.VehicleWorker: an 'ArduSub V' STATUSTEXT records the banner (time + count) and says it on the log at most
   once a minute — the tail of a burst that ended inside the minute is said by tick() once the minute has passed
   (verify-replay F2, 2026-10-15); VehicleImu carries t_att_v / att_tb_ms / att_map_t0 / t_banner / n_banners.
3. state_assembler: health att_age_v = t - t_att_v beside the UNCHANGED imu_age and imu_stale_s rule, present on a
   fixless tick too, None when unknown; the assembled state is identical with or without it.
4. rl_pwm.PwmInterlocks AGE: 0.15 s / 0.5 s, 0.25 s / 0.2 s, unknown (None / nan) = over, one healthy tick resets,
   AGE is named first, judge_age=False skips it, meta; parse_per_thruster_block validates the eight keys.
5. workers.MpcWorker: AGE disengages rl_pwm and rl_pwm_7d with neutral pulses and a mission line (CSV imu_age_s stays
   small while att_age_v_s is stale — the 10-13 blind spot); a 4-axis LOW never sees any of it; the engage refusal
   (unknown, map < 30 s, banner < 30 s, ANY unknown tick in the last 10 s, rolling 10-s p99 > 0.12 s, the age NOW >
   age_s 0.15 s; in that order — the last two rules from verify-review F1 / verify-replay I1, 2026-10-15, RL_controller/
   logs/eval/station_fix_20261015); the CSV columns (schema 21, every
   LOW) and meta `att_age`; the banner mission line at most once a minute, held for the next events.log when no run
   folder is open.
6. tools/rl_pwm_interlock_replay: AGE judged only on a schema-21 file that recorded att_age_v_s.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import tempfile
from collections import deque
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from c3_camera import mavlink_log as ML                                                          # noqa: E402
from c3_camera.mavlink_log import (VCLK_DRIFT_PER_S, VCLK_REBOOT_GAP_S, MavlinkLogger,          # noqa: E402
                                   VehicleClockMap)
from rov_gui.control.rl_pwm import (NEUTRAL_PULSES, PER_THRUSTER_DEFAULTS, PwmInterlocks,       # noqa: E402
                                    parse_per_thruster_block)
from rov_gui.state import (BANNER_LINE, BANNER_PREFIX, BANNER_WARN_EVERY_S, Conn, NavFix,        # noqa: E402
                           Telemetry, VehicleImu, now)

HZ = 20.0
BANNER = "ArduSub V4.5.7 (b09fafe2)"          # the 10-13 text [측정: data/20261013/1013_211743/*_rov.jsonl]


# =============================================================================
# 1. the clock map
# =============================================================================
def _stream(n, rate_hz, t0_host, tb0_ms, transit, drift_ppm=0.0):
    """(t_host, time_boot_ms, true capture time on the host clock) of n samples generated at rate_hz on the
    vehicle; the vehicle clock runs (1 - drift_ppm 1e-6) as fast as the host's (positive = vehicle slow, so
    t_host - time_boot GROWS); `transit(k)` = that sample's delay to the host."""
    out = []
    for k in range(n):
        t_cap = t0_host + k / rate_hz                       # host-clock time the sample was taken
        tb = tb0_ms + (k / rate_hz) * (1.0 - drift_ppm * 1e-6) * 1000.0
        out.append((t_cap + transit(k), tb, t_cap))
    return out


def _sawtooth(k, rate_hz=200.0, rise_s=1.8, fall_s=1.0, peak_s=0.9):
    """A vehicle-side FIFO queue: the delay climbs 0 -> peak over rise_s, then drains over fall_s (a FIFO cannot
    drain faster than real time: arrival order is generation order, so the delay's slope is >= -1)."""
    t = (k / rate_hz) % (rise_s + fall_s)
    return peak_s * (t / rise_s) if t < rise_s else peak_s * (1.0 - (t - rise_s) / fall_s)


def test_the_map_settles_on_the_floor_and_reads_a_vehicle_side_queue_as_age():
    rng = np.random.default_rng(7)
    jit = rng.uniform(0.0, 0.004, 40000)
    m = VehicleClockMap()
    # 60 s healthy at 200 Hz: transit 3 ms + 0..4 ms jitter
    healthy = _stream(12000, 200.0, 5000.0, 7_200_000.0, lambda k: 0.003 + jit[k])
    errs = []
    for th, tb, tc in healthy:
        tv = m.update(th, tb, (1, 1))
        errs.append(tv - tc)
    errs = np.asarray(errs[2000:])                          # after the first 10 s
    # the envelope sits on the least-delayed samples: t_valid = capture + ~the minimum transit (3 ms) — the age it
    # gives is a LOWER bound by exactly that, never more (and never an over-read)
    assert np.all(errs >= 0.003 - 1e-9) and np.all(errs <= 0.0045), (errs.min(), errs.max())
    assert m.resets == 0 and m.t_start == pytest.approx(healthy[0][0])
    # then a vehicle-side queue (the 10-13 shape: up to 0.9 s, rising over ~2 s, draining), three teeth; no jitter
    # here so the FIFO stays in order
    queued = _stream(1680, 200.0, healthy[-1][2] + 0.005, healthy[-1][1] + 5.0, lambda k: 0.003 + _sawtooth(k))
    ages_read, ages_true = [], []
    for th, tb, tc in queued:
        tv = m.update(th, tb, (1, 1))
        ages_read.append(th - tv)                           # the age at ARRIVAL, as a tick right then would see it
        ages_true.append(th - tc)
    d = np.asarray(ages_true) - np.asarray(ages_read)
    assert np.all(d >= 0.003 - 1e-9) and np.all(d <= 0.0045), (d.min(), d.max())   # the queue IS read as age
    assert max(ages_read) > 0.85 and m.resets == 0
    # (the station's parse-time age of the same samples reads 0 at arrival: the 10-13 blind spot)


@pytest.mark.parametrize("drift_ppm", [240.0, -240.0])
def test_the_map_follows_a_vehicle_clock_drifting_within_300_ppm(drift_ppm):
    m = VehicleClockMap()
    s = _stream(90000, 50.0, 100.0, 1_000.0, lambda k: 0.002, drift_ppm=drift_ppm)       # 30 min
    worst = 0.0
    for i, (th, tb, tc) in enumerate(s):
        tv = m.update(th, tb, (1, 1))
        if i > 50:
            worst = max(worst, abs((tv - tc) - 0.002))
    # a rise of 240 ppm is inside the 300 ppm rate limit; a fall is followed by min() at once
    assert worst < 0.0005, worst
    assert VCLK_DRIFT_PER_S == pytest.approx(300e-6) and m.resets == 0


def test_a_clock_drifting_past_the_limit_over_reads_the_age_never_under():
    m = VehicleClockMap()
    s = _stream(5000, 50.0, 100.0, 1_000.0, lambda k: 0.002, drift_ppm=1000.0)          # 100 s at 1000 ppm
    over = []
    for th, tb, tc in s:
        tv = m.update(th, tb, (1, 1))
        over.append((th - tv) - (th - tc - 0.002))          # read age - (true age - min transit)
    over = np.asarray(over)
    assert np.all(over >= -1e-9) and over[-1] > 0.05        # 700 ppm x 100 s = 0.07 s too OLD: fail-safe side


def test_the_map_resets_on_a_reboot_and_not_on_an_outage_or_a_duplicate():
    m = VehicleClockMap()
    for th, tb, _tc in _stream(1500, 50.0, 100.0, 600_000.0, lambda k: 0.002):
        m.update(th, tb, (1, 1))
    t_last = th
    # a duplicate stamp (seen 0-209 times per recording) and an outage the vehicle clock ran through (the 23.6-s
    # ATTITUDE gap of 0809_202049: host +23.642 s, time_boot_ms +23.595 s): no reset, and the first sample after it
    # reads an age between 0 and its true extra delay (47 ms)
    m.update(t_last + 0.001, tb, (1, 1))
    tv = m.update(t_last + 23.642, tb + 23_595.0, (1, 1))
    assert m.resets == 0 and 0.0 <= (t_last + 23.642) - tv <= 0.047 + 1e-9
    t_last = t_last + 23.642
    # REBOOT 1: time_boot_ms back to 9 s, 12 s of host time later -> a new map, right at once
    tv = m.update(t_last + 12.0, 9_000.0, (1, 1))
    assert m.resets == 1 and "went back" in m.last_reset["why"] and m.t_start == pytest.approx(t_last + 12.0)
    assert tv == pytest.approx(t_last + 12.0)               # the first sample of a map defines it
    for k in range(1, 100):
        tv = m.update(t_last + 12.0 + 0.02 * k + 0.001, 9_000.0 + 20.0 * k, (1, 1))
    assert abs(tv - (t_last + 12.0 + 0.02 * 99)) < 1e-3 and m.resets == 1
    # REBOOT 2, landing ABOVE the old time_boot_ms (the old session was shorter than the new one's first sample):
    # host 10 s on, vehicle +1.5 s -> rule 2 (host ran > VCLK_REBOOT_GAP_S ahead)
    th2, tb2 = t_last + 12.0 + 0.02 * 99 + 0.001, 9_000.0 + 20.0 * 99
    m.update(th2 + 10.0, tb2 + 1_500.0, (1, 1))
    assert m.resets == 2 and "ahead" in m.last_reset["why"] and VCLK_REBOOT_GAP_S == 2.0
    # a new source (another system id) is another clock
    m.update(th2 + 10.02, tb2 + 1_520.0, (2, 1))
    assert m.resets == 3 and "source" in m.last_reset["why"]
    # unusable stamps change nothing
    n = m.n
    assert m.update(th2 + 11.0, None) is None and m.update(th2 + 11.0, "x") is None \
        and m.update(float("nan"), 5.0) is None and m.n == n
    s = m.summary()
    assert s["resets"] == 3 and s["drift_ppm"] == 300.0 and s["samples"] == n
    json.dumps(s)


def test_a_map_started_inside_a_queue_under_reads_until_the_queue_drains():
    """The known limit (fix-design check_online_err.py: up to 0.77 s for 6.8 s of 1013_211913) that
    engage.per_thruster.age_map_min_s (30 s) exists for."""
    m = VehicleClockMap()
    q = lambda k: 0.002 + max(0.0, 0.8 - 0.8 * k / 300.0)                    # noqa: E731  0.8 s draining over 6 s
    under = []
    for th, tb, tc in _stream(600, 50.0, 100.0, 50_000.0, q):
        tv = m.update(th, tb, (1, 1))
        under.append((th - tc) - (th - tv))                 # true - read
    assert under[0] == pytest.approx(0.802, abs=1e-6)        # the first read says "fresh"; it was 0.802 s old
    assert under[150] > 0.38                                 # still under-reading half way through the drain
    assert max(under[305:]) < 0.0025 and min(under) >= 0.002 - 1e-9   # right once drained; never an over-read


class _Msg:
    def __init__(self, mtype, src=(1, 1), **fields):
        self._t, self._src, self._f = mtype, src, fields

    def get_type(self):
        return self._t

    def get_srcSystem(self):
        return self._src[0]

    def get_srcComponent(self):
        return self._src[1]

    def to_dict(self):
        return {"mavpackettype": self._t, **self._f}


class _Master:
    """recv_match hands out the queued messages, then stops the reader."""

    def __init__(self, logger, msgs):
        self.logger, self.msgs = logger, list(msgs)

    def recv_match(self, blocking=True, timeout=0.5):
        if not self.msgs:
            self.logger._stop.set()
            return None
        return self.msgs.pop(0)


def test_the_reader_stamps_autopilot_attitude_only():
    lg = MavlinkLogger(verbose=False)
    lg.master = _Master(lg, [
        _Msg("ATTITUDE", time_boot_ms=600_000, roll=0.1, pitch=0.0, yaw=0.0, rollspeed=0.0, pitchspeed=0.0,
             yawspeed=0.0),
        _Msg("STATUSTEXT", severity=6, text=BANNER),
        _Msg("ATTITUDE", src=(1, 194), time_boot_ms=5, roll=0.0),      # a companion's clock: never mapped
        _Msg("ATTITUDE", time_boot_ms=600_020, roll=0.1),
        _Msg("ATTITUDE", roll=0.1),                                     # no stamp: cannot be placed
    ])
    lg._run_udp()
    recs = lg.drain()
    att = [r for r in recs if r["msg_type"] == "ATTITUDE"]
    assert len(att) == 4 and "t_valid" not in [r for r in recs if r["msg_type"] == "STATUSTEXT"][0]
    a0, foreign, a1, bare = att
    assert a0["t_valid"] == pytest.approx(a0["t_host"]) and a0["vclk_t0"] == pytest.approx(a0["t_host"])
    assert foreign["t_valid"] is None and foreign["vclk_t0"] is None
    assert a1["vclk_t0"] == a0["vclk_t0"] and a1["t_valid"] <= a1["t_host"] + 1e-9
    assert bare["t_valid"] is None and bare["vclk_t0"] is None
    assert lg.vclock.n == 2 and lg.summary()["vehicle_clock"]["samples"] == 2


def test_the_rest_reader_stamps_attitude_too(monkeypatch):
    lg = MavlinkLogger(transport="rest", verbose=False)
    calls = {"n": 0}

    class _Resp:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self.body

    def fake_urlopen(url, timeout=2.0):
        calls["n"] += 1
        if calls["n"] > 1:
            lg._stop.set()
            raise ML.urllib.error.URLError("done")
        payload = {"ATTITUDE": {"message": {"type": "ATTITUDE", "time_boot_ms": 123_000, "roll": 0.0},
                                "status": {"time": {"counter": 1}}}}
        return _Resp(json.dumps(payload).encode())

    monkeypatch.setattr(ML.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(ML.time, "sleep", lambda s: None)
    lg._run_rest()
    att = [r for r in lg.drain() if r["msg_type"] == "ATTITUDE"]
    assert len(att) == 1 and att[0]["t_valid"] == pytest.approx(att[0]["t_host"]) \
        and att[0]["vclk_t0"] == pytest.approx(att[0]["t_host"])


# =============================================================================
# 2. the vehicle worker: the banner, and what VehicleImu carries
# =============================================================================
def _vehicle_worker():
    from rov_gui.backends.hardware import VehicleWorker
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

    w = VehicleWorker.__new__(VehicleWorker)          # no link, no thread (the test_offline pattern)
    w.bus, w.opts = bus, O()
    w._leak, w._leak_reported, w._press_reported = LeakMonitor(), 0, 0.0
    w.leak_cfg_fn = None
    w._latest, w.logger = {}, None
    return w, logs, imus


def test_the_banner_is_recorded_and_said_at_most_once_a_minute():
    w, logs, _imus = _vehicle_worker()
    t0 = 1000.0
    # the 10-13 culprit re-asked every ~1.9 s: 40 banners over 74 s
    for k in range(40):
        w._note_statustext({"text": BANNER + "\x00\x00", "severity": 6, "t_host": t0 + 1.9 * k})
    warns = [m for lvl, m in logs if lvl == "warn" and BANNER_LINE in m]
    assert len(warns) == 2, warns                             # at t0 and at the first banner >= t0 + 60 s
    k2 = math.ceil(BANNER_WARN_EVERY_S / 1.9)                 # 32 -> 1060.8 s
    assert "1 since the last warning" in warns[0] and f"{k2} since the last warning" in warns[1], warns
    assert f"{k2 + 1} this session" in warns[1] and BANNER[:20] in warns[0]
    assert w._n_banners == 40 and w._t_banner == pytest.approx(t0 + 1.9 * 39)
    # other vehicle text is not a banner (the boot burst's other lines, a PreArm, a mid-text match)
    for text in ("ChibiOS: 6f2a4e5d", "Frame: VECTORED_6DOF", "PreArm: Compass not calibrated", "xx ArduSub V4"):
        w._note_statustext({"text": text, "severity": 6, "t_host": t0 + 200.0})
    assert w._n_banners == 40 and BANNER_PREFIX == "ArduSub V"
    # a worker built bare (__new__, no banner attributes) still takes one, with a missing arrival stamp
    w2, logs2, _ = _vehicle_worker()
    w2._note_statustext({"text": BANNER, "severity": 6})
    assert w2._n_banners == 1 and w2._t_banner is not None and len(logs2) == 1


def test_the_tail_of_a_banner_burst_is_said_once_the_minute_has_passed():
    """verify-replay F2 (2026-10-15): _note_banner only spoke at an ARRIVAL, so a burst that ended inside the minute
    after its first warning never had its count said. VehicleWorker.tick() now says it once the minute has passed."""
    import time as _time

    w, logs, _imus = _vehicle_worker()

    class _Quiet:                                   # a reader with nothing new this tick
        def drain(self):
            return []

    w.logger, w._log, w._mpc = _Quiet(), None, False
    w._last_tel = w._last_link = float("inf")       # no Telemetry publish, no link sample: only the banner bookkeeping

    def warns():
        return [m for lvl, m in logs if lvl == "warn" and BANNER_LINE in m]

    t = _time.monotonic()
    # a burst of 4 banners 70..64 s ago: the first is said at its arrival, the other 3 are held by the limit
    for k in range(4):
        w._note_statustext({"text": BANNER, "severity": 6, "t_host": t - 70.0 + 2.0 * k})
    assert len(warns()) == 1 and "1 since the last warning" in warns()[0]
    w.tick()                                        # 70 s after the first line: the held 3 are said, once
    assert len(warns()) == 2 and "3 since the last warning, 4 this session" in warns()[1], warns()
    assert BANNER[:20] in warns()[1]
    w.tick()
    assert len(warns()) == 2                        # nothing new, nothing said
    # a fresh burst inside the minute after that line: held at arrival AND at the tick (the limit still holds)
    for k in range(2):
        w._note_statustext({"text": BANNER, "severity": 6, "t_host": _time.monotonic()})
    w.tick()
    assert len(warns()) == 2 and w._n_banners == 6
    # ...and said once the minute has passed (the line's own time moved back 61 s here instead of waiting)
    w._banner_warned_t -= BANNER_WARN_EVERY_S + 1.0
    w.tick()
    assert len(warns()) == 3 and "2 since the last warning, 6 this session" in warns()[2], warns()


def test_vehicle_imu_carries_the_vehicle_clock_and_the_banner():
    w, _logs, imus = _vehicle_worker()
    t = now()
    w._latest["ATTITUDE"] = {"t_host": t, "msg_type": "ATTITUDE", "roll": 0.1, "pitch": -0.2, "yaw": 0.3,
                             "rollspeed": 0.01, "pitchspeed": 0.02, "yawspeed": 0.03, "time_boot_ms": 777_000,
                             "t_valid": t - 0.4, "vclk_t0": t - 90.0}
    w._t_banner, w._n_banners = t - 12.0, 3
    w._publish_imu()
    v = imus[-1]
    assert v.t_att == t and v.t_att_v == pytest.approx(t - 0.4) and v.att_tb_ms == 777_000
    assert v.att_map_t0 == pytest.approx(t - 90.0) and v.t_banner == pytest.approx(t - 12.0) and v.n_banners == 3
    # a record the map could not place (or a pre-2026-10-15 record) publishes UNKNOWN, never a guess
    del w._latest["ATTITUDE"]["t_valid"]
    w._publish_imu()
    assert imus[-1].t_att_v is None and imus[-1].roll == 0.1
    # no ATTITUDE at all: nothing to place, the banner still rides
    w._latest.clear()
    w._publish_imu()
    assert imus[-1].t_att_v is None and imus[-1].att_map_t0 is None and imus[-1].n_banners == 3


# =============================================================================
# 3. the state assembler
# =============================================================================
def _fix_at(t, n_tags=4, p=(-2.0, 0.0, 0.8)):
    from rov_gui.control.state_assembler import rot_zyx

    return NavFix(t_capture=t, n_tags=n_tags, tag_ids=tuple(range(25, 25 + n_tags)), p_ned=tuple(p),
                  R_ned_body=tuple(rot_zyx(0.0, 0.0, 0.0).ravel()), yaw_ned=0.0, reproj_rms_px=0.3,
                  conn=Conn.ONLINE, stamp=t)


def _imu_at(t, age=0.0, map_age=60.0, known=True, t_banner=None, n_banners=0, roll=0.0, pitch=0.0, p=0.0, q=0.0,
            tb_ms=4_242_000):
    """The station's parse time `t_att` is ALWAYS fresh here (= t): only the vehicle clock says how old the sample
    is — exactly the 10-13 situation."""
    return VehicleImu(roll=roll, pitch=pitch, yaw=0.0, p=p, q=q, r=0.0, ax=0.0, ay=0.0, az=-9.80665, depth_m=0.8,
                      t_att=t, t_imu=t, t_baro=t,
                      t_att_v=(t - age) if known else None, att_tb_ms=tb_ms if known else None,
                      att_map_t0=(t - map_age) if known else None, t_banner=t_banner, n_banners=n_banners,
                      conn=Conn.ONLINE, stamp=t)


def test_the_assembler_reports_the_true_age_beside_the_unchanged_imu_age():
    from rov_gui.control.state_assembler import StateAssembler, att_age_v

    t0 = 500.0
    a, b = StateAssembler(), StateAssembler()
    m1, h1 = a.step(_fix_at(t0), _imu_at(t0, age=0.4), t0 + 0.01, 0.05)
    m2, h2 = b.step(_fix_at(t0), _imu_at(t0, known=False), t0 + 0.01, 0.05)
    assert h1["imu_age"] == pytest.approx(0.01) and h1["att_age_v"] == pytest.approx(0.41)
    assert h2["att_age_v"] is None and h2["imu_age"] == pytest.approx(0.01)
    # the estimate is untouched by it: same state either way, and a 0.4-s-old attitude is NOT refused here —
    # imu_stale_s (0.3 s) still judges the parse-time age only
    assert m1 is not None and m2 is not None
    for k in ("eta", "nu", "nudot"):
        assert np.array_equal(m1[k], m2[k])
    m3, h3 = StateAssembler().step(_fix_at(t0), _imu_at(t0 - 0.35, age=0.0), t0, 0.05)
    assert m3 is None and "imu stale" in h3["why"] and h3["att_age_v"] == pytest.approx(0.35)
    # a fixless tick (the station bridge flies on those) still carries it; no snapshot -> None
    m4, h4 = StateAssembler().step(None, _imu_at(t0, age=0.2), t0, 0.05)
    assert m4 is None and h4["why"] == "no tag fix" and h4["att_age_v"] == pytest.approx(0.2)
    assert StateAssembler().step(_fix_at(t0), None, t0, 0.05)[1]["att_age_v"] is None
    # floored at 0 like imu_age; non-finite -> unknown
    assert att_age_v(_imu_at(t0 + 1.0), t0) == 0.0
    assert att_age_v(VehicleImu(t_att_v=float("nan")), t0) is None and att_age_v(None, t0) is None


# =============================================================================
# 4. the interlock and its keys
# =============================================================================
QUIET = dict(roll_rad=math.radians(5.0), pitch_rad=math.radians(-6.0), p_rad_s=math.radians(10.0),
             q_rad_s=math.radians(-8.0), batt_a=20.0, batt_v=15.5,
             pwm_us=(1580, 1500, 1480, 1500, 1530, 1470, 1510, 1490))


def _ticks(ilk, n, **over):
    kw = dict(QUIET)
    kw.update(over)
    for _ in range(n):
        why = ilk.step(**kw)
        if why:
            return why
    return ""


def test_the_age_interlock_soft_hard_unknown_and_its_debounce():
    ilk = PwmInterlocks(None, HZ, 1.0)
    assert ilk.meta()["hold_ticks"]["age"] == 10 and ilk.meta()["hold_ticks"]["age_hard"] == 4
    # soft: 0.18 s for 9 ticks holds, the 10th (0.5 s) trips
    assert _ticks(ilk, 9, att_age_s=0.18) == ""
    why = _ticks(ilk, 1, att_age_s=0.18)
    assert "AGE" in why and "0.18 s > 0.15 s for 0.5 s" in why and "neutral pulses" in why, why
    assert ilk.last_trip["name"] == "age" and ilk.last_trip["values"]["att_age_s"] == pytest.approx(0.18)
    # hard: 0.30 s for 3 ticks holds, the 4th (0.2 s) trips
    ilk = PwmInterlocks(None, HZ, 1.0)
    assert _ticks(ilk, 3, att_age_s=0.30) == ""
    why = _ticks(ilk, 1, att_age_s=0.30)
    assert "0.30 s > 0.25 s for 0.2 s" in why, why
    # UNKNOWN is over (fail-safe) — None, nan, a string — and trips on the hard rule; a caller that never passes
    # the age trips the same way
    for bad in (None, float("nan"), "?"):
        ilk = PwmInterlocks(None, HZ, 1.0)
        assert _ticks(ilk, 3, att_age_s=bad) == ""
        why = _ticks(ilk, 1, att_age_s=bad)
        assert "UNKNOWN" in why and "fail-safe" in why and "for 0.2 s" in why, why
    ilk = PwmInterlocks(None, HZ, 1.0)
    assert "AGE" in _ticks(ilk, 4, **{k: v for k, v in QUIET.items()})
    assert ilk.peak["att_age_unknown_ticks"] == 4 and ilk.peak["att_age_s_max"] is None
    # one healthy tick resets both counters; exactly at the threshold is not over
    ilk = PwmInterlocks(None, HZ, 1.0)
    for _ in range(10):
        assert _ticks(ilk, 9, att_age_s=0.2) == ""
        assert _ticks(ilk, 1, att_age_s=0.05) == ""
    assert _ticks(ilk, 200, att_age_s=0.15) == "" and _ticks(ilk, 200, att_age_s=0.06) == ""
    assert ilk.n_trips == 0 and ilk.peak["att_age_s_max"] == pytest.approx(0.2) and ilk.ticks["age"] == 0
    ilk.reset()
    assert ilk.age_hard_ticks == 0 and ilk.ticks["age"] == 0 and ilk.peak["att_age_unknown_ticks"] == 0
    json.dumps(ilk.meta())


def test_age_is_named_first_and_judge_age_false_skips_it():
    # tilt over from tick 1 (trips at 5), age over from tick 2 (hard trips at 5): the same tick -> AGE named
    ilk = PwmInterlocks(None, HZ, 1.0)
    assert _ticks(ilk, 1, roll_rad=math.radians(30.0), att_age_s=0.05) == ""
    why = _ticks(ilk, 4, roll_rad=math.radians(30.0), att_age_s=0.4)
    assert "AGE" in why and ilk.ticks["tilt"] == 5 and ilk.last_trip["name"] == "age", why
    assert PwmInterlocks.NAMES[0] == "age"
    # the offline replay of a file that never logged the age: the item is skipped, everything else judged
    ilk = PwmInterlocks(None, HZ, 1.0, judge_age=False)
    assert _ticks(ilk, 100, att_age_s=None) == "" and ilk.meta()["age_judged"] is False
    assert "age NOT judged" in ilk.describe() and "TILT" in _ticks(ilk, 5, roll_rad=math.radians(30.0))
    ilk = PwmInterlocks(None, HZ, 1.0)
    d = ilk.describe()
    assert d.startswith("age 0.15 s/0.5 s | 0.25 s/0.2 s") and "unknown = over" in d and "tilt 20 deg/0.25 s" in d


def test_the_eight_age_keys_are_validated():
    d = parse_per_thruster_block(None)
    assert {k: d[k] for k in ("age_s", "age_hold_s", "age_hard_s", "age_hard_hold_s", "age_engage_p99_s",
                              "age_engage_window_s", "age_map_min_s", "banner_hold_s")} == {
        "age_s": 0.15, "age_hold_s": 0.5, "age_hard_s": 0.25, "age_hard_hold_s": 0.2, "age_engage_p99_s": 0.12,
        "age_engage_window_s": 10.0, "age_map_min_s": 30.0, "banner_hold_s": 30.0}
    # RATE / TILT unchanged by this change
    assert (d["rate_dps"], d["rate_s"], d["abort_deg"], d["abort_s"]) == (40.0, 0.15, 20.0, 0.25)
    for bad in ({"age_s": 0}, {"age_hold_s": float("nan")}, {"age_hard_s": 0.1}, {"banner_hold_s": True},
                {"age_engage_p99_s": "0.12"}, {"age_map_min_s": -1.0}, {"age_map_min": 30.0}):
        with pytest.raises(ValueError, match="engage.per_thruster"):
            parse_per_thruster_block(bad)
    assert parse_per_thruster_block({"age_s": 0.1, "age_hard_s": 0.2})["age_hard_s"] == 0.2
    # the shipped yaml carries the same eight, literally
    src = (ROOT / "config" / "hw_mpc.yaml").read_text(encoding="utf-8")
    for line in ("    age_s: 0.15", "    age_hold_s: 0.5", "    age_hard_s: 0.25", "    age_hard_hold_s: 0.2",
                 "    age_engage_p99_s: 0.12", "    age_engage_window_s: 10.0", "    age_map_min_s: 30.0",
                 "    banner_hold_s: 30.0"):
        assert line in src, line
    from rov_gui.control.geometry import MpcConfig

    assert MpcConfig.load(ROOT / "config" / "hw_mpc.yaml").engage["per_thruster"] == PER_THRUSTER_DEFAULTS


# =============================================================================
# 5. the worker
# =============================================================================
def _feed(w, age=0.0, map_age=60.0, known=True, t_banner=None, n_banners=0, roll=0.0, n_tags=4):
    from rov_gui.tests.test_rl_pwm_interlocks import _gate

    t = now()
    w.on_nav_fix(_fix_at(t, n_tags))
    w.on_vehicle_imu(_imu_at(t, age=age, map_age=map_age, known=known, t_banner=t_banner, n_banners=n_banners,
                             roll=roll))
    w.on_telemetry(Telemetry(armed=True, mode="MANUAL", conn=Conn.ONLINE, pwm_gate=_gate(), battery_v=15.5,
                             current_a=20.0))


def _pwm(tmp, mode, monkeypatch):
    """A pwm-LOW worker (test_rl_pwm_interlocks' builder), the follower's pulses pinned to a quiet frame (the real
    pwm10_s1 squeeze would trip NULL by itself after 0.5 s on this static frame — not what is tested here)."""
    from rov_gui.control.rl_policy import HwRlPwm7d
    from rov_gui.tests.test_rl_pwm_7d import _export
    from rov_gui.tests.test_rl_pwm_interlocks import _pwm_worker

    w, bus, pwms, logs, events = _pwm_worker(tmp, mode="rl_pwm")
    if mode == "rl_pwm_7d":
        w.cfg.rl_pwm_7d = {"policy_dir": str(_export(tmp))}
        w._rl_pwm7d = HwRlPwm7d(w.cfg, log=lambda m: None)
        w.set_mode("rl_pwm_7d")
        assert w.cfg.mode == "rl_pwm_7d", w.reason
    real_step = w.ctrl.step

    def pinned(eta, nu, nudot, t):
        u, info = real_step(eta, nu, nudot, t)
        return u, {**info, "pwm_us": QUIET["pwm_us"]}

    monkeypatch.setattr(w.ctrl, "step", pinned)
    return w, bus, pwms, logs, events


def _engage_healthy(w):
    w.on_enable(True)
    _feed(w)
    w.set_engaged(True)
    assert w.engaged, w.reason
    for _ in range(4):
        _feed(w)
        w.tick()
    assert w.engaged, w.reason


@pytest.mark.parametrize("mode", ["rl_pwm", "rl_pwm_7d"])
def test_a_stale_attitude_disengages_both_pwm_lows_while_imu_age_reads_fresh(mode, monkeypatch):
    from rov_gui.tests.test_control import _last_csv_row
    from rov_gui.tests.test_rl_pwm_interlocks import _events_log

    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pwms, logs, events = _pwm(tmp, mode, monkeypatch)
        _engage_healthy(w)
        assert any("PER-THRUSTER INTERLOCKS ARMED" in m and "age 0.15 s/0.5 s" in m for _l, m in logs)
        for _ in range(3):                           # 0.30 s on the vehicle clock, 3 ticks: the debounce holds
            _feed(w, age=0.30)
            w.tick()
        assert w.engaged, w.reason
        row = _last_csv_row(w)
        # THE 10-13 BLIND SPOT, on the row: the parse-time age is fresh, the vehicle clock says 0.30 s
        assert float(row["imu_age_s"]) < 0.05 and float(row["att_age_v_s"]) == pytest.approx(0.30, abs=0.03)
        assert row["att_tb_ms"] == "4242000" and row["banner_age_s"] == "nan"
        _feed(w, age=0.30)
        w.tick()
        assert not w.engaged and "per-thruster interlock AGE" in w.reason and "> 0.25 s for 0.2 s" in w.reason, \
            w.reason
        assert pwms[-1].pulses == NEUTRAL_PULSES and pwms[-1].source == "stop"
        assert any("PER-THRUSTER INTERLOCK AGE" in e for e in events) and any("DISENGAGED" in e for e in events)
        assert "PER-THRUSTER INTERLOCK AGE" in _events_log(w)
        c = w._run_meta()["controller"]["interlocks"]
        assert c["last_trip"]["name"] == "age" and c["last_trip"]["values"]["att_age_s"] == pytest.approx(0.3, abs=0.03)
        assert c["age_judged"] is True and c["thresholds"]["age_hard_s"] == 0.25
        w.teardown()


def test_the_soft_rule_and_an_unknown_age_trip_a_flight_and_quiet_data_does_not(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        # soft: 0.18 s -> 9 ticks hold, the 10th trips
        w, bus, pwms, logs, events = _pwm(tmp, "rl_pwm", monkeypatch)
        _engage_healthy(w)
        for _ in range(9):
            _feed(w, age=0.18)
            w.tick()
        assert w.engaged, w.reason
        _feed(w, age=0.18)
        w.tick()
        # (the age printed is 0.18 s + the feed-to-tick wall time: match the rule, not the last digit)
        assert not w.engaged and "attitude age 0.1" in w.reason and "> 0.15 s for 0.5 s" in w.reason, w.reason
        w.teardown()
        # the map vanishes mid-flight (a record the reader could not place): unknown = over, 4 ticks
        w, bus, pwms, logs, events = _pwm(tmp, "rl_pwm", monkeypatch)
        _engage_healthy(w)
        for _ in range(3):
            _feed(w, known=False)
            w.tick()
        assert w.engaged, w.reason
        _feed(w, known=False)
        w.tick()
        assert not w.engaged and "UNKNOWN" in w.reason, w.reason
        w.teardown()
        # a healthy link (0.06 s, the healthy recordings' p50) for 3 s: engaged throughout, a banner in flight
        # does not end it (the banner gates ENGAGE; the age is what flies)
        w, bus, pwms, logs, events = _pwm(tmp, "rl_pwm", monkeypatch)
        _engage_healthy(w)
        for k in range(60):
            _feed(w, age=0.06, t_banner=(now() - 1.0) if k > 30 else None, n_banners=int(k > 30))
            w.tick()
        assert w.engaged, w.reason
        assert w._run_meta()["controller"]["interlocks"]["peak"]["att_age_s_max"] == pytest.approx(0.06, abs=0.03)
        w.disengage("test over")
        w.teardown()


def test_a_4dof_low_never_sees_the_age_gates():
    from rov_gui.tests.test_control import _last_csv_row, _worker

    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _worker(tmp)
        w.set_mode("pid")
        w.on_enable(True)
        _feed(w, age=0.5, map_age=1.0, t_banner=now(), n_banners=1)        # every pwm refusal would fire
        w.set_engaged(True)
        assert w.engaged, w.reason
        for _ in range(20):
            _feed(w, known=False)
            w.tick()
        assert w.engaged and w._pwm_ilk is None, w.reason
        row = _last_csv_row(w)
        assert row["att_age_v_s"] == "nan" and row["att_tb_ms"] == "nan"     # the columns are on every LOW
        _feed(w, age=0.4)
        w.tick()
        assert float(_last_csv_row(w)["att_age_v_s"]) == pytest.approx(0.4, abs=0.03)
        w.disengage("test over")
        w.teardown()


def _shift_history(w, dt):
    w._att_age_hist = deque(((t - dt, a) for (t, a) in w._att_age_hist), maxlen=w._att_age_hist.maxlen)


@pytest.mark.parametrize("mode", ["rl_pwm", "rl_pwm_7d"])
def test_the_engage_refusal_rules_in_order(mode, monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pwms, logs, events = _pwm(tmp, mode, monkeypatch)
        w.on_enable(True)
        # 1. unknown (no ATTITUDE on the vehicle clock) — first, whatever else is wrong
        _feed(w, known=False, t_banner=now(), n_banners=1)
        w.set_engaged(True)
        assert not w.engaged and f"{mode}: attitude age UNKNOWN" in w.reason, w.reason
        # 2. a map 10 s old (station just started / vehicle rebooted)
        _feed(w, map_age=10.0, t_banner=now(), n_banners=1)
        w.set_engaged(True)
        assert not w.engaged and "vehicle-clock map is 10 s old" in w.reason and "age_map_min_s 30 s" in w.reason, \
            w.reason
        # 3. a banner 5 s ago; 31 s ago is fine
        _feed(w, t_banner=now() - 5.0, n_banners=1)
        w.set_engaged(True)
        assert not w.engaged and "'ArduSub V' banner 5 s ago" in w.reason and "banner_hold_s 30 s" in w.reason, \
            w.reason
        # 4. the rolling p99: 180 fresh ticks and 20 at 0.5 s (10 % > 1 %) -> refused, with the numbers
        for k in range(200):
            _feed(w, age=(0.5 if k % 10 == 0 else 0.0), t_banner=now() - 31.0, n_banners=1)
            w.tick()
        _feed(w, t_banner=now() - 31.0, n_banners=1)
        w.set_engaged(True)
        assert not w.engaged and "attitude age p99 0.5" in w.reason and "s over the last 10 s" in w.reason \
            and "age_engage_p99_s 0.12 s" in w.reason, w.reason
        # ...the stale ticks age out of the window -> the same link engages (now fresh)
        _shift_history(w, 11.0)
        _feed(w, t_banner=now() - 31.0, n_banners=1)
        w.set_engaged(True)
        assert w.engaged, w.reason
        w.disengage("test over")
        w.teardown()


def test_any_unknown_tick_in_the_window_refuses_and_the_p99_includes_the_engage_instant(monkeypatch):
    """verify-replay I1 (2026-10-15): with unknown ticks weighted by a sentinel inside an interpolated p99, 2 unknown
    ticks refused a 200-tick window and passed a 201-tick one. Now ONE unknown tick in the window refuses, whatever
    its length (fail-safe; the runtime AGE item trips an unknown age in 0.2 s anyway)."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pwms, logs, events = _pwm(tmp, "rl_pwm", monkeypatch)
        w.on_enable(True)
        # an empty window: the age NOW is the whole sample — 0.2 s refuses (by the p99, which comes first), 0.05 s not
        _feed(w, age=0.2)
        w.set_engaged(True)
        assert not w.engaged and "p99 0.2" in w.reason and "(1 ticks;" in w.reason, w.reason
        # 200 healthy ticks with ONE unknown: refused, and the reason says how many of how many
        for k in range(200):
            _feed(w, known=(k != 50))
            w.tick()
        _feed(w)
        w.set_engaged(True)
        assert not w.engaged and "1 of the last 200 ticks" in w.reason and "NO measurable attitude age" in w.reason \
            and "unknown is never fresh" in w.reason, w.reason
        # ...the unknown tick ages out of the window -> the same link engages
        _shift_history(w, 11.0)
        _feed(w)
        w.set_engaged(True)
        assert w.engaged, w.reason
        w.disengage("test over")
        # the window's length no longer decides it: 2 unknown among 200 and among 201 (the I1 pair), 1 among 1000
        _feed(w)
        for n_ok, n_unk in ((198, 2), (199, 2), (999, 1)):
            t = now()
            w._att_age_hist = deque([(t, 0.05)] * n_ok + [(t, None)] * n_unk, maxlen=w._att_age_hist.maxlen)
            why = w._att_age_refusal()
            assert f"{n_unk} of the last {n_ok + n_unk} ticks" in why, (n_ok, n_unk, why)
            w._att_age_hist = deque([(t, 0.05)] * (n_ok + n_unk), maxlen=w._att_age_hist.maxlen)
            assert w._att_age_refusal() == "", (n_ok, n_unk)
        # an all-unknown window refuses too
        w._att_age_hist = deque([(now(), None)] * 50, maxlen=w._att_age_hist.maxlen)
        assert "50 of the last 50 ticks" in w._att_age_refusal()
        w.teardown()


@pytest.mark.parametrize("mode", ["rl_pwm", "rl_pwm_7d"])
def test_an_attitude_already_stale_at_start_is_refused_after_a_full_healthy_window(mode, monkeypatch):
    """verify-review F1 (2026-10-15): the age at the START instant was ONE sample beside ~200 window ticks, so it
    could not move the p99 — 200 x 0.05 s + now 0.5 s was admitted and the hard rule tripped 0.2 s later. Now an age
    NOW over age_s (0.15 s, the runtime soft limit) refuses, after every other rule."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pwms, logs, events = _pwm(tmp, mode, monkeypatch)
        w.on_enable(True)
        for _ in range(200):                     # a full healthy window (10 s at 20 Hz), the healthy p50 0.05 s
            _feed(w, age=0.05)
            w.tick()
        assert len(w._att_age_hist) >= 200
        # now 0.5 s: the p99 of the window + now stays 0.05 s, the instant refuses — with both numbers
        _feed(w, age=0.5)
        w.set_engaged(True)
        assert not w.engaged and f"{mode}: the attitude is 0.5" in w.reason and "s old NOW" in w.reason \
            and "engage.per_thruster.age_s 0.15 s" in w.reason and "p99 0.05 s" in w.reason, w.reason
        # just over the limit refuses (the age read at START only grows past what was fed) ...
        _feed(w, age=0.16)
        w.set_engaged(True)
        assert not w.engaged and "s old NOW" in w.reason, w.reason
        # ... below it engages (0.12 s: the p99 limit, 30 ms under age_s)
        _feed(w, age=0.12)
        w.set_engaged(True)
        assert w.engaged, w.reason
        w.disengage("test over")
        # the earlier rules keep their place: a banner 5 s ago is named before the stale instant
        _feed(w, age=0.5, t_banner=now() - 5.0, n_banners=1)
        w.set_engaged(True)
        assert not w.engaged and "'ArduSub V' banner 5 s ago" in w.reason, w.reason
        w.teardown()


def test_the_csv_columns_and_the_meta(monkeypatch):
    from rov_gui.control.workers import CSV_HEADER
    from rov_gui.tests.test_control import _last_csv_row

    cols = CSV_HEADER.strip().split(",")
    i = cols.index("att_age_v_s")
    assert cols[i:i + 3] == ["att_age_v_s", "att_tb_ms", "banner_age_s"] and len(cols) == len(set(cols))
    # schema 22 (2026-10-16): the DVL selection's seven follow them (test_dvl pins their meaning)
    assert cols[-7:] == ["w_dvl", "dvl_code", "dvl_res_fast_m_s", "dvl_res_slow_m_s", "dvl_esc_s",
                         "dvl_align", "dvl_scale"]
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pwms, logs, events = _pwm(tmp, "rl_pwm", monkeypatch)
        _engage_healthy(w)
        _feed(w, age=0.05, t_banner=now() - 40.0, n_banners=2)
        w.tick()
        row = _last_csv_row(w)
        assert len(row) == len(cols) and float(row["att_age_v_s"]) == pytest.approx(0.05, abs=0.03)
        assert row["att_tb_ms"] == "4242000" and float(row["banner_age_s"]) == pytest.approx(40.0, abs=0.5)
        m = w._run_meta()
        assert m["schema_version"] == 22
        a = m["att_age"]
        assert a["estimator"]["drift_ppm"] == 300.0 and a["estimator"]["reboot_gap_s"] == 2.0
        assert a["pwm_gate"]["thresholds"]["age_engage_p99_s"] == 0.12 and a["pwm_gate"]["lows"] == ["rl_pwm",
                                                                                                    "rl_pwm_7d"]
        assert a["at_write"]["map_age_s"] == pytest.approx(60.0, abs=1.0) and a["at_write"]["banners_seen"] == 2
        assert a["at_write"]["last_banner_age_s"] == pytest.approx(40.0, abs=1.0)
        assert a["at_write"]["window_ticks"] >= 5 and a["at_write"]["window_p99_s"] is not None
        json.dumps(m["att_age"])
        w.disengage("test over")
        w.teardown()


def test_the_banner_is_a_mission_line_at_most_once_a_minute():
    from rov_gui.tests.test_control import _worker

    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _worker(tmp)
        events = []
        bus.mpc_event.connect(events.append)
        # idle (no run folder): the screen line goes out, the file line is HELD for the next events.log
        _feed(w, n_banners=1, t_banner=now())
        w.tick()
        lines = [e for e in events if BANNER_LINE in e]
        assert len(lines) == 1 and "(1 'ArduSub V' banner since the last line)" in lines[0], lines
        assert any(BANNER_LINE in p for p in w._pending_events) and not list(Path(tmp).rglob("events.log"))
        # four more inside the minute: no new line, counted
        _feed(w, n_banners=5, t_banner=now())
        w.tick()
        assert len([e for e in events if BANNER_LINE in e]) == 1 and w._banners_unsaid == 4
        # a minute later (the tick's clock), said once with the count
        w._banner_tick(w._banner_event_t + BANNER_WARN_EVERY_S)
        lines = [e for e in events if BANNER_LINE in e]
        assert len(lines) == 2 and "(4 'ArduSub V' banners since the last line)" in lines[1]
        # engaged: straight into the run's events.log, which the held line heads
        w.on_enable(True)
        _feed(w, n_banners=5)
        w.set_engaged(True)
        assert w.engaged, w.reason
        _feed(w, n_banners=6)
        w.tick()
        w._banner_tick(w._banner_event_t + BANNER_WARN_EVERY_S)
        ev = "\n".join(p.read_text(encoding="utf-8") for p in Path(tmp).rglob("events.log"))
        assert ev.count(BANNER_LINE) == 3, ev
        assert ev.index("(1 'ArduSub V' banner since") < ev.index("ENGAGED"), ev
        # a VehicleWorker restart (its count starts over) is followed, not read as -5 banners
        _feed(w, n_banners=1)
        w.tick()
        assert w._banners_seen == 1
        w.disengage("test over")
        w.teardown()


# =============================================================================
# 6. the replay tool
# =============================================================================
def _schema21(path, ages):
    """Turn a test_rl_pwm_interlocks fixture into a schema-21 recording with the given att_age_v_s cells."""
    p = Path(path)
    with open(p, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    cols = list(rows[0].keys())
    assert "att_age_v_s" in cols
    for r, a in zip(rows, ages):
        r["att_age_v_s"] = "" if a is None else f"{a:.3f}"
    with open(p, "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)
    mp = Path(str(p.with_suffix("")) + ".meta.json")
    meta = json.loads(mp.read_text(encoding="utf-8"))
    meta["schema_version"] = 21
    mp.write_text(json.dumps(meta), encoding="utf-8")
    return str(p)


def test_the_replay_judges_age_only_where_it_was_recorded():
    from rov_gui.tests.test_rl_pwm_interlocks import _write_fixture
    from rov_gui.tools import rl_pwm_interlock_replay as T

    quiet = QUIET["pwm_us"]
    thr = dict(PER_THRUSTER_DEFAULTS)
    with tempfile.TemporaryDirectory() as tmp:
        # a schema-20 file (no age ever logged): AGE is skipped, nothing trips on quiet data
        a = _write_fixture(tmp, "aaa", "rl_pwm", 40, pwm=quiet)
        ra = T.replay_file(a, thr, (120.0,))
        assert ra["age_judged"] is False and ra["trips"][120.0] is None
        # schema 21, 0.30 s on every row: the hard rule, 4 rows from the first (pulses from row 2 -> +0.05 s)
        b = _schema21(_write_fixture(tmp, "bbb", "rl_pwm", 40, pwm=quiet), [0.30] * 40)
        rb = T.replay_file(b, thr, (120.0,))
        assert rb["age_judged"] is True and rb["trips"][120.0]["name"] == "age"
        assert rb["trips"][120.0]["t_s"] == pytest.approx(0.05, abs=1e-6) and "0.30 s > 0.25 s" in \
            rb["trips"][120.0]["why"]
        # schema 21 with EMPTY cells (the map never came up): unknown = over, same timing
        c = _schema21(_write_fixture(tmp, "ccc", "rl_pwm", 40, pwm=quiet), [None] * 40)
        rc = T.replay_file(c, thr, (120.0,))
        assert rc["trips"][120.0]["name"] == "age" and "UNKNOWN" in rc["trips"][120.0]["why"]
        # schema 21, healthy 0.06 s: nothing
        d = _schema21(_write_fixture(tmp, "ddd", "rl_pwm", 40, pwm=quiet), [0.06] * 40)
        assert T.replay_file(d, thr, (120.0,))["trips"][120.0] is None


# =============================================================================
# runner
# =============================================================================
def main() -> int:
    return pytest.main([__file__, "-q", "-p", "no:cacheprovider"])


if __name__ == "__main__":
    raise SystemExit(main())
