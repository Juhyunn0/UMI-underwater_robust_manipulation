#!/usr/bin/env python3
"""test_mavlink_rates.py — the station's MAVLink IMU-stream request (2026-10-15), offline.

Source: RL_controller/logs/eval/pool_trips_1013_rootcause (fix-design station item 1; operator-approved fix 1c) and
the consumer table in RL_controller/logs/eval/station_fix_20261015/impl-rates/NOTES.md. Until 2026-10-15 the
station asked for RAW_IMU, SCALED_IMU2 and ATTITUDE all at --mavlink-rate (default 200): ~500 msg/s of IMU stream to
every BlueOS endpoint, which a parameter download on 10-13 queued behind. Now ATTITUDE 50 / RAW_IMU 20 /
SCALED_IMU2 20 Hz (rov_gui/mavlink_streams.py has why each), and --mavlink-rate N is the old request.

    cd <repo> && QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/robust/bin/python -m pytest \\
        rov_gui/tests/test_mavlink_rates.py -q -p no:cacheprovider

What is pinned:

1. rov_gui/mavlink_streams.resolve: the defaults (90 msg/s asked instead of 600), the legacy request (all three at N,
   accelerometer order unchanged), the refusals (legacy + a per-message flag; ATTITUDE or RAW_IMU off; a negative,
   boolean or non-finite value), and SCALED_IMU2 off (the accelerometer moves to RAW_IMU only).
2. c3_camera/mavlink_log.MavlinkLogger: the per-message request leaves as these SET_MESSAGE_INTERVAL intervals
   (50 Hz = 20000 us, 20 Hz = 50000 us, off = -1) with the rest of DEFAULT_RATES_HZ untouched; the legacy request is
   byte-identical to what the station sent before; c3_collect's one-rate path is unchanged (<= 0 = not requested,
   never "off"); summary() says whether the request went out and what was turned off — and a request whose IMU-stream
   sends raised is NOT recorded as sent (rates_failed; verify-review F2, 2026-10-15).
3. rov_gui/__main__: the four flags default to None (= mavlink_streams.DEFAULT_HZ, which the help states);
   check_mavlink_rates refuses, warns and stays silent as documented — the ATTITUDE age-floor warning on the
   RESOLVED rate, legacy --mavlink-rate N included (verify-review F3); the legacy help says N >= 1 is the old request
   and 0/negative is refused (F4) — and main() exits 2 on a refusal before anything opens.
4. hardware.VehicleWorker: setup() hands the logger the resolved request and logs it (sent / not sent / refused);
   _publish_imu reads the accelerometer from SCALED_IMU2 then RAW_IMU as before, and from RAW_IMU only with
   SCALED_IMU2 off — never a stale SCALED_IMU2 left in _latest; stream_meta() snapshots request, ack and arrivals.
5. MpcWorker run meta `mavlink_streams` (the injected snapshot, the flags-only fallback, a failing hook) and the
   *_rov.json sidecar's `stream_request` (its other keys unchanged).
6. The ATTITUDE default against the pwm LOWs' age gate: a lower rate adds at most one period to the tick age, and
   the default is not below the floor the recorded-link replay supports.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from c3_camera import mavlink_log as ML                                                          # noqa: E402
from c3_camera.mavlink_log import MavlinkLogger, MessageStat, VehicleClockMap                    # noqa: E402
from rov_gui import mavlink_streams as MS                                                         # noqa: E402
from rov_gui.state import now                                                                    # noqa: E402

MG = 9.80665 / 1000.0


# =============================================================================
# 1. the request
# =============================================================================
def test_the_defaults_ask_for_90_msg_per_s_and_keep_the_accelerometer_where_it_was():
    res = MS.resolve()
    assert res == {"policy": "per_message", "legacy_hz": None,
                   "rates_hz": {"ATTITUDE": 50, "RAW_IMU": 20, "SCALED_IMU2": 20}, "off": [],
                   "accel_source": ["SCALED_IMU2", "RAW_IMU"]}
    assert MS.DEFAULT_HZ == res["rates_hz"] and tuple(res["accel_source"]) == MS.ACCEL_SOURCE
    # 600 asked (and ~500 delivered, 166.6 Hz each) before; 90 now
    assert MS.requested_msg_per_s(res) == 90.0 and MS.requested_msg_per_s(MS.resolve(legacy_hz=200)) == 600.0
    # an absent attribute is "not given": a bare namespace resolves to the defaults
    assert MS.resolve_opts(types.SimpleNamespace()) == res and MS.typed(types.SimpleNamespace()) == []
    assert "ATTITUDE 50 Hz, RAW_IMU 20 Hz, SCALED_IMU2 20 Hz" in MS.describe(res)


def test_the_legacy_request_sets_all_three_and_changes_nothing_else():
    res = MS.resolve(legacy_hz=200)
    assert res["policy"] == "legacy" and res["legacy_hz"] == 200
    assert res["rates_hz"] == {"ATTITUDE": 200, "RAW_IMU": 200, "SCALED_IMU2": 200} and res["off"] == []
    assert tuple(res["accel_source"]) == MS.ACCEL_SOURCE
    assert "LEGACY --mavlink-rate 200" in MS.describe(res)
    assert MS.typed(types.SimpleNamespace(mavlink_rate=200)) == ["--mavlink-rate"]


@pytest.mark.parametrize("kw, needle", [
    (dict(legacy_hz=200, attitude_hz=50), "--mavlink-att-rate"),
    (dict(legacy_hz=200, raw_imu_hz=20, scaled_imu2_hz=0), "--mavlink-imu-rate, --mavlink-scaled-imu2-rate"),
    (dict(legacy_hz=0), "--mavlink-rate 0"),
    (dict(attitude_hz=0), "--mavlink-att-rate 0"),
    (dict(raw_imu_hz=0), "--mavlink-imu-rate 0"),
    (dict(scaled_imu2_hz=-1), "--mavlink-scaled-imu2-rate -1"),
    (dict(attitude_hz=True), "--mavlink-att-rate True"),
    (dict(raw_imu_hz=float("nan")), "--mavlink-imu-rate nan"),
    (dict(attitude_hz="50"), "--mavlink-att-rate '50'"),
])
def test_a_contradiction_or_a_value_below_its_floor_is_refused(kw, needle):
    with pytest.raises(ValueError) as e:
        MS.resolve(**kw)
    assert needle in str(e.value), str(e.value)


def test_scaled_imu2_off_asks_the_vehicle_to_stop_it_and_moves_the_accelerometer():
    res = MS.resolve(scaled_imu2_hz=0)
    assert res["rates_hz"]["SCALED_IMU2"] == 0 and res["off"] == ["SCALED_IMU2"]
    assert tuple(res["accel_source"]) == MS.ACCEL_SOURCE_SCALED_IMU2_OFF == ("RAW_IMU",)
    assert MS.requested_msg_per_s(res) == 70.0 and "SCALED_IMU2 off" in MS.describe(res)
    # a float rate is kept as given (argparse hands ints; an embedder may not)
    assert MS.resolve(attitude_hz=62.5)["rates_hz"]["ATTITUDE"] == 62.5
    assert MS.resolve(attitude_hz=100.0)["rates_hz"]["ATTITUDE"] == 100


# =============================================================================
# 2. what leaves the reader
# =============================================================================
class _Mav:
    def __init__(self):
        self.sent = []

    def command_long_send(self, *a):
        self.sent.append(a)


class _Master:
    target_system = 1
    target_component = 1

    def __init__(self):
        self.mav = _Mav()


def _intervals(lg) -> dict:
    """{message name: interval_us} of every SET_MESSAGE_INTERVAL the logger sent."""
    from pymavlink import mavutil

    ids = {getattr(mavutil.mavlink, f"MAVLINK_MSG_ID_{k}"): k for k in lg.rates_hz
           if hasattr(mavutil.mavlink, f"MAVLINK_MSG_ID_{k}")}
    out = {}
    for a in lg.master.mav.sent:
        assert a[2] == mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL and a[:2] == (1, 1), a
        out[ids[a[4]]] = a[5]
    return out


def _sent(lg):
    lg.master = _Master()
    lg._request_rates()
    return _intervals(lg)


def test_the_logger_asks_for_each_stream_at_its_own_rate():
    lg = MavlinkLogger(set_rates=True, stream_rates_hz=MS.resolve()["rates_hz"], verbose=False)
    s0 = lg.summary()
    assert lg.rates_sent_t is None and s0["rates_sent"] is False and s0["requested_off"] == []
    iv = _sent(lg)
    assert (iv["ATTITUDE"], iv["RAW_IMU"], iv["SCALED_IMU2"]) == (20_000, 50_000, 50_000), iv
    for k, hz in ML.DEFAULT_RATES_HZ.items():          # the rest of the reader's request is untouched
        if k not in MS.STREAMS:
            assert iv[k] == int(1_000_000 / hz), (k, iv[k])
    s = lg.summary()
    assert lg.rates_sent_t is not None and s["rates_sent"] is True
    assert s["requested_rates_hz"]["ATTITUDE"] == 50 and s["requested_off"] == []
    # SCALED_IMU2 off goes out as interval -1 (MAVLink: stop sending)
    off = MavlinkLogger(set_rates=True, stream_rates_hz=MS.resolve(scaled_imu2_hz=0)["rates_hz"], verbose=False)
    iv = _sent(off)
    assert iv["SCALED_IMU2"] == -1 and iv["ATTITUDE"] == 20_000 and iv["RAW_IMU"] == 50_000
    assert off.summary()["requested_off"] == ["SCALED_IMU2"]
    # not asked to set rates: requested_* say so, nothing was sent
    quiet = MavlinkLogger(set_rates=False, stream_rates_hz=MS.resolve()["rates_hz"], verbose=False)
    assert quiet.summary()["requested_rates_hz"] is None and quiet.summary()["requested_off"] is None


def test_the_legacy_request_is_the_old_one_byte_for_byte():
    # exactly what hardware.VehicleWorker.setup built before 2026-10-15 at the CLI default
    old = MavlinkLogger(set_rates=True, default_rate_hz=200, verbose=False)
    new = MavlinkLogger(set_rates=True, stream_rates_hz=MS.resolve(legacy_hz=200)["rates_hz"], verbose=False)
    assert new.rates_hz == old.rates_hz and new.stream_off == () == old.stream_off
    _sent(old)
    _sent(new)
    assert new.master.mav.sent == old.master.mav.sent and len(old.master.mav.sent) == len(ML.DEFAULT_RATES_HZ)


def test_c3_collects_one_rate_path_is_unchanged():
    for n in (50, 200):
        lg = MavlinkLogger(set_rates=True, default_rate_hz=n, verbose=False)
        assert all(lg.rates_hz[k] == n for k in MS.STREAMS) and lg.stream_off == ()
    for n in (0, -1):            # never "off" on this path: not requested, as it always was
        lg = MavlinkLogger(set_rates=True, default_rate_hz=n, verbose=False)
        iv = _sent(lg)
        assert not set(MS.STREAMS) & set(iv) and -1 not in iv.values(), iv
    # the REST transport is read-only: nothing sent, and summary() says so
    rest = MavlinkLogger(transport="rest", set_rates=True, stream_rates_hz=MS.resolve()["rates_hz"], verbose=False)
    rest._request_rates()
    assert rest.rates_sent_t is None and rest.summary()["rates_sent"] is False


class _FailMav(_Mav):
    """command_long_send raises for these message ids (None = every one), like a socket gone under the reader."""

    def __init__(self, fail_ids=None):
        super().__init__()
        self.fail_ids = fail_ids

    def command_long_send(self, *a):
        if self.fail_ids is None or a[4] in self.fail_ids:
            raise OSError("[Errno 101] Network is unreachable")
        self.sent.append(a)


def test_a_request_whose_imu_stream_sends_failed_is_not_recorded_as_sent(capsys):
    """verify-review F2 (2026-10-15): rates_sent_t was stamped even when every SET_MESSAGE_INTERVAL raised, so the
    summary, the run meta and the setup line said 'sent'."""
    from pymavlink import mavutil

    def mid(k):
        return getattr(mavutil.mavlink, f"MAVLINK_MSG_ID_{k}")

    def request(fail_ids):
        lg = MavlinkLogger(set_rates=True, stream_rates_hz=MS.resolve()["rates_hz"], verbose=False)
        lg.master = _Master()
        lg.master.mav = _FailMav(fail_ids)
        capsys.readouterr()
        lg._request_rates()
        return lg, lg.summary(), capsys.readouterr().out

    # every send raised: not sent, every name kept, and said
    lg, s, out = request(None)
    asked = sorted(k for k, hz in lg.rates_hz.items() if hasattr(mavutil.mavlink, f"MAVLINK_MSG_ID_{k}") and hz > 0)
    assert lg.rates_sent_t is None and s["rates_sent"] is False and sorted(s["rates_failed"]) == asked, s
    assert "NOT sent (send error on" in out and "ATTITUDE" in out, out
    # ONE IMU stream raised (ATTITUDE, the one the age gate needs): not sent either
    lg, s, out = request({mid("ATTITUDE")})
    assert lg.rates_sent_t is None and s["rates_sent"] is False and s["rates_failed"] == ["ATTITUDE"], s
    # another message raised: the IMU request went out -> sent, the failure kept and said
    lg, s, out = request({mid("VFR_HUD")})
    assert lg.rates_sent_t is not None and s["rates_sent"] is True and s["rates_failed"] == ["VFR_HUD"], s
    assert _intervals(lg)["ATTITUDE"] == 20_000 and "VFR_HUD" not in _intervals(lg)
    # a clean request: nothing failed (and a logger that never requested says so too)
    lg, s, out = request(set())
    assert lg.rates_sent_t is not None and s["rates_failed"] == [] and "NOT sent" not in out
    assert MavlinkLogger(verbose=False).summary()["rates_failed"] == []


# =============================================================================
# 3. the command line
# =============================================================================
def test_the_flags_default_to_the_module_and_the_launch_check(monkeypatch, capsys):
    from rov_gui import __main__ as M

    p = M.build_parser()
    a = p.parse_args(["--source", "hw"])
    assert [getattr(a, MS.LEGACY_ATTR)] + [getattr(a, MS.OPT_ATTR[k]) for k in MS.STREAMS] == [None] * 4
    assert MS.resolve_opts(a) == MS.resolve() and M.check_mavlink_rates(a) is None
    helps = {act.dest: act.help or "" for act in p._actions}
    for k in MS.STREAMS:                       # the help states the module's numbers, not its own copy
        assert f"default {MS.DEFAULT_HZ[k]} Hz" in helps[MS.OPT_ATTR[k]], helps[MS.OPT_ATTR[k]]
    assert "LEGACY" in helps[MS.LEGACY_ATTR] and "defaulted to 200" in helps[MS.LEGACY_ATTR]
    # F4: N >= 1 is the old request; 0/negative is refused (the old station turned 0 into 50 Hz)
    assert "N >= 1 reproduces that old request" in helps[MS.LEGACY_ATTR] and "REFUSED" in helps[MS.LEGACY_ATTR] \
        and "turned 0 into 50 Hz" in helps[MS.LEGACY_ATTR]

    def check(*argv):
        return M.check_mavlink_rates(p.parse_args(list(argv)))

    lvl, txt = check("--source", "hw", "--mavlink-rate", "200")
    assert lvl == "warn" and "LEGACY" in txt and "600 msg/s asked" in txt and "against 90" in txt, txt
    lvl, txt = check("--source", "hw", "--mavlink-rate", "200", "--mavlink-scaled-imu2-rate", "20")
    assert lvl == "refuse" and "--mavlink-scaled-imu2-rate" in txt, txt
    for bad in (["--mavlink-att-rate", "0"], ["--mavlink-imu-rate", "0"], ["--mavlink-scaled-imu2-rate", "-1"],
                ["--mavlink-rate", "0"]):
        assert check("--source", "hw", *bad)[0] == "refuse", bad
    assert check("--mavlink-att-rate", "0")[0] == "refuse"     # a bad flag is refused whatever the source
    lvl, txt = check("--source", "hw", "--mavlink-scaled-imu2-rate", "0")
    assert lvl == "warn" and "RAW_IMU" in txt and "never run on" in txt, txt
    lvl, txt = check("--source", "hw", "--mavlink-att-rate", "30")
    assert lvl == "warn" and "age gate" in txt, txt
    # F3: the floor is checked on the RESOLVED ATTITUDE rate — the legacy request below it warns too
    lvl, txt = check("--source", "hw", "--mavlink-rate", "20")
    assert lvl == "warn" and "LEGACY" in txt and "ATTITUDE at 20 Hz" in txt and "age gate" in txt, txt
    for n in ("50", "200"):                                    # at or above the floor: the legacy note only
        lvl, txt = check("--source", "hw", "--mavlink-rate", n)
        assert lvl == "warn" and "LEGACY" in txt and "age gate" not in txt, (n, txt)
    for bad in ("0", "-5"):                                    # F4: the old 0 -> 50 Hz / negative -> none, refused
        lvl, txt = check("--source", "hw", "--mavlink-rate", bad)
        assert lvl == "refuse" and ">= 1 Hz" in txt, (bad, txt)
    assert check("--source", "hw", "--mavlink-att-rate", "100", "--mavlink-imu-rate", "25") is None
    lvl, txt = check("--mavlink-att-rate", "100")                                   # demo
    assert lvl == "warn" and "does nothing" in txt, txt
    lvl, txt = check("--source", "hw", "--mavlink-transport", "none", "--mavlink-rate", "200")
    assert lvl == "warn" and "does nothing" in txt, txt
    lvl, txt = check("--source", "hw", "--no-mavlink-set-rates", "--mavlink-att-rate", "100")
    assert lvl == "warn" and "NOTHING is requested" in txt, txt
    # a land dry run transmits nothing either (resolve_defaults clears mavlink_set_rates)
    lvl, txt = check("--source", "hw", "--land-dry-run", "--mavlink-att-rate", "100")
    assert lvl == "warn" and "NOTHING is requested" in txt, txt

    # main() exits 2 on the refusal BEFORE the preflight or a window: both are made to fail loudly here
    def boom(*_a, **_k):
        raise AssertionError("main() went past a refused stream request")

    import faulthandler

    monkeypatch.setattr(M, "_preflight", boom)
    monkeypatch.setitem(sys.modules, "rov_gui.window", types.SimpleNamespace(build_and_run=boom))
    monkeypatch.setattr(faulthandler, "enable", lambda *a, **k: None)     # capsys's stderr has no fileno
    capsys.readouterr()
    assert M.main(["--source", "hw", "--mavlink-rate", "200", "--mavlink-att-rate", "50"]) == 2
    assert "REFUSED — --mavlink-rate 200 sets RAW_IMU" in capsys.readouterr().err
    assert M.main(["--mavlink-imu-rate", "0"]) == 2
    assert "REFUSED — --mavlink-imu-rate 0" in capsys.readouterr().err


# =============================================================================
# 4. the vehicle worker
# =============================================================================
class _RecLogger:
    """Stands in for MavlinkLogger in VehicleWorker.setup: records the kwargs, 'sends' when it would."""
    built: list = []

    def __init__(self, **kw):
        _RecLogger.built.append(kw)
        self.kw = kw
        self.transport = kw.get("transport", "udp")
        self.rates_sent_t = None
        self.rates_failed = []
        self.rate_acks = {}
        self.stats = {}

    def start(self, wait_s=6.0):
        conn = self.kw.get("connection")
        if conn == "senderr":                     # every IMU-stream send raised (MavlinkLogger._request_rates)
            self.rates_failed = ["ATTITUDE", "RAW_IMU", "SCALED_IMU2"]
        elif self.kw.get("set_rates") and self.transport == "udp" and conn != "silent":
            self.rates_sent_t = 123.0
            if conn == "senderr_other":           # only a non-IMU message raised: sent
                self.rates_failed = ["VFR_HUD"]
        return conn != "silent"


def _bare_worker(opts):
    from rov_gui.backends.hardware import VehicleWorker

    logs = []
    w = VehicleWorker.__new__(VehicleWorker)          # no link, no thread (the test_offline pattern)
    w.bus = types.SimpleNamespace(log=types.SimpleNamespace(emit=lambda *a: logs.append(a)))
    w.opts = opts
    w.logger = None
    w._log = None
    return w, logs


def _setup(monkeypatch, **opts):
    from rov_gui.backends.hardware import VehicleWorker

    _RecLogger.built.clear()
    monkeypatch.setattr(ML, "MavlinkLogger", _RecLogger)
    w, logs = _bare_worker(types.SimpleNamespace(**opts))
    VehicleWorker.setup(w)
    return w, logs, (_RecLogger.built[-1] if _RecLogger.built else None)


def test_setup_hands_the_logger_the_request_and_says_it(monkeypatch):
    w, logs, kw = _setup(monkeypatch, mavlink_transport="udp", mavlink_set_rates=True)
    assert kw["stream_rates_hz"] == {"ATTITUDE": 50, "RAW_IMU": 20, "SCALED_IMU2": 20} and kw["set_rates"] is True
    assert "default_rate_hz" not in kw and w._accel_msgs == MS.ACCEL_SOURCE and w._streams_error is None
    line = [m for lvl, m in logs if "IMU streams" in m]
    assert line == ["mavlink: IMU streams " + MS.describe(MS.resolve())], logs
    assert [lvl for lvl, m in logs if "IMU streams" in m] == ["info"]
    # the legacy request: the old dict, said as a warning
    w, logs, kw = _setup(monkeypatch, mavlink_transport="udp", mavlink_set_rates=True, mavlink_rate=200)
    assert kw["stream_rates_hz"] == {"ATTITUDE": 200, "RAW_IMU": 200, "SCALED_IMU2": 200}
    assert [(lvl, "LEGACY" in m) for lvl, m in logs if "IMU streams" in m] == [("warn", True)]
    # SCALED_IMU2 off: the accelerometer moves with it
    w, logs, kw = _setup(monkeypatch, mavlink_transport="udp", mavlink_set_rates=True, mavlink_scaled_imu2_rate=0)
    assert kw["stream_rates_hz"]["SCALED_IMU2"] == 0 and w._accel_msgs == ("RAW_IMU",)
    # nothing went out: said, with why
    for opts, why in ((dict(mavlink_transport="udp", mavlink_set_rates=False), "--no-mavlink-set-rates"),
                      (dict(mavlink_transport="rest", mavlink_set_rates=True), "REST transport is read-only"),
                      (dict(mavlink_transport="udp", mavlink_set_rates=True, mavlink="silent"),
                       "no telemetry within 3 s")):
        w, logs, kw = _setup(monkeypatch, **opts)
        said = [m for lvl, m in logs if "IMU streams" in m]
        assert len(said) == 1 and "NOT sent" in said[0] and why in said[0], (opts, said)
    # a refused combination (an embedder past the launch check): NOTHING requested, said as an error
    w, logs, kw = _setup(monkeypatch, mavlink_transport="udp", mavlink_set_rates=True, mavlink_rate=200,
                         mavlink_att_rate=50)
    assert kw["set_rates"] is False and kw["stream_rates_hz"] is None and w._streams is None
    assert any(lvl == "error" and "NOT sent" in m and "--mavlink-att-rate" in m for lvl, m in logs), logs
    assert w._accel_msgs == MS.ACCEL_SOURCE
    # --mavlink-transport none still builds nothing (test_offline pins the rest of that)
    w, logs, kw = _setup(monkeypatch, mavlink_transport="none", mavlink_set_rates=True)
    assert kw is None and w.logger is None


def test_setup_says_not_sent_when_the_imu_stream_sends_failed(monkeypatch):
    """verify-review F2 (2026-10-15): the setup line and stream_meta() follow the logger's rates_sent_t /
    rates_failed — a request whose IMU-stream sends raised reads 'NOT sent (send error ...)', sent false."""
    w, logs, kw = _setup(monkeypatch, mavlink_transport="udp", mavlink_set_rates=True, mavlink="senderr")
    said = [(lvl, m) for lvl, m in logs if "IMU streams" in m]
    assert len(said) == 1 and said[0][0] == "warn" and "NOT sent (send error on ATTITUDE, RAW_IMU, SCALED_IMU2)" \
        in said[0][1], said
    m = w.stream_meta()
    assert m["sent"] is False and m["failed"] == ["ATTITUDE", "RAW_IMU", "SCALED_IMU2"], m
    # a non-IMU message failed: the IMU request went out — sent, and the line names what did not
    w, logs, kw = _setup(monkeypatch, mavlink_transport="udp", mavlink_set_rates=True, mavlink="senderr_other")
    said = [(lvl, m) for lvl, m in logs if "IMU streams" in m]
    assert len(said) == 1 and said[0][0] == "warn" and "sent, except VFR_HUD: NOT sent (send error)" in said[0][1], \
        said
    m = w.stream_meta()
    assert m["sent"] is True and m["failed"] == ["VFR_HUD"], m
    # a clean request: info, failed []
    w, logs, kw = _setup(monkeypatch, mavlink_transport="udp", mavlink_set_rates=True)
    assert [lvl for lvl, m in logs if "IMU streams" in m] == ["info"] and w.stream_meta()["failed"] == []


def _imu_worker():
    from rov_gui.backends.hardware import VehicleWorker
    from rov_gui.bus import DataBus
    from rov_gui.leak import LeakMonitor
    from rov_gui.tests.test_control import _app

    _app()
    bus = DataBus()
    imus = []
    bus.vehicle_imu.connect(imus.append)
    w = VehicleWorker.__new__(VehicleWorker)
    w.bus, w.opts = bus, types.SimpleNamespace(mavlink_transport="udp")
    w._leak, w._leak_reported, w._press_reported = LeakMonitor(), 0, 0.0
    w.leak_cfg_fn = None
    w._latest, w.logger = {}, None
    return w, imus


def test_publish_imu_reads_the_accelerometer_it_was_told_to():
    from rov_gui.backends.hardware import VehicleWorker

    assert VehicleWorker._accel_msgs == MS.ACCEL_SOURCE      # the class default IS the order every build used
    w, imus = _imu_worker()
    t = now()
    w._latest["ATTITUDE"] = {"t_host": t, "roll": 0.1, "pitch": -0.2, "yaw": 0.3, "rollspeed": 0.0,
                             "pitchspeed": 0.0, "yawspeed": 0.0}
    # this vehicle: SCALED_IMU2 reads zeros, RAW_IMU is the live accelerometer (mg)
    w._latest["SCALED_IMU2"] = {"t_host": t - 0.01, "xacc": 0, "yacc": 0, "zacc": 0}
    w._latest["RAW_IMU"] = {"t_host": t - 0.02, "xacc": 12, "yacc": -3, "zacc": -998}
    w._publish_imu()
    v = imus[-1]
    assert (v.ax, v.ay, v.az) == (0.0, 0.0, 0.0) and v.t_imu == t - 0.01       # unchanged: SCALED_IMU2 first
    del w._latest["SCALED_IMU2"]
    w._publish_imu()
    v = imus[-1]                                                                 # unchanged: RAW_IMU when alone
    assert v.az == pytest.approx(-998 * MG) and v.ax == pytest.approx(12 * MG) and v.t_imu == t - 0.02
    # SCALED_IMU2 asked to stop: RAW_IMU ONLY, even with a SCALED_IMU2 from before the request still in _latest
    w._latest["SCALED_IMU2"] = {"t_host": t - 30.0, "xacc": 0, "yacc": 0, "zacc": 0}
    w._accel_msgs = tuple(MS.resolve(scaled_imu2_hz=0)["accel_source"])
    w._publish_imu()
    v = imus[-1]
    assert v.az == pytest.approx(-998 * MG) and v.t_imu == t - 0.02 and v.roll == 0.1
    # no accelerometer message at all: none published, as before
    w._latest = {"ATTITUDE": w._latest["ATTITUDE"]}
    w._publish_imu()
    assert imus[-1].ax is None and imus[-1].t_imu is None


def test_stream_meta_snapshots_the_request_the_ack_and_the_arrivals(monkeypatch):
    w, logs, kw = _setup(monkeypatch, mavlink_transport="udp", mavlink_set_rates=True)
    lg = w.logger
    st = MessageStat()
    t0 = __import__("time").monotonic() - 4.0
    for k in range(200):                                   # 50 Hz, the last one 0.02 s ago
        st.update(t0 + k * 0.02)
    lg.stats = {"ATTITUDE": st}
    lg.rate_acks = {"last": "MAV_RESULT_ACCEPTED"}
    m = w.stream_meta()
    assert m["policy"] == "per_message" and m["requested_hz"] == MS.DEFAULT_HZ and m["off"] == []
    assert m["sent"] is True and m["ack"] == "MAV_RESULT_ACCEPTED" and m["requested_msg_per_s"] == 90.0
    assert m["achieved_hz"]["ATTITUDE"] == pytest.approx(50.0, rel=1e-3) and m["achieved_hz"]["RAW_IMU"] is None
    assert m["last_arrival_age_s"]["ATTITUDE"] == pytest.approx(0.04, abs=0.05) and m["accel_source"] == list(
        MS.ACCEL_SOURCE)
    json.dumps(m)
    assert "achieved_hz" not in w.stream_meta(achieved=False)
    # a worker that never set up still answers (class defaults), and a broken one never raises
    bare, _ = _bare_worker(types.SimpleNamespace())
    m = bare.stream_meta()
    assert m["policy"] is None and m["sent"] is False and m["achieved_hz"] == dict.fromkeys(MS.STREAMS)
    bare.logger = types.SimpleNamespace(rates_sent_t=1.0, rate_acks=None, stats={})
    assert "error" in bare.stream_meta()


# =============================================================================
# 5. the records
# =============================================================================
def test_the_run_meta_records_the_request():
    from rov_gui.backends import hardware
    from rov_gui.tests.test_control import _worker

    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _worker(tmp)
        ms = w._run_meta()["mavlink_streams"]                  # demo / not wired: the flags only
        assert ms["link"] is None and set(ms["flags"]) == {
            "mavlink_rate", "mavlink_att_rate", "mavlink_imu_rate", "mavlink_scaled_imu2_rate",
            "mavlink_set_rates", "mavlink_transport"}
        snap = {"policy": "per_message", "requested_hz": dict(MS.DEFAULT_HZ), "sent": True}
        w.mavlink_streams_fn = lambda: snap
        m = w._run_meta()
        assert m["mavlink_streams"] == snap and m["schema_version"] == 21
        json.dumps(m["mavlink_streams"])

        def broken():
            raise RuntimeError("worker gone")
        w.mavlink_streams_fn = broken
        assert "RuntimeError: worker gone" in w._run_meta()["mavlink_streams"]["error"]
        w.teardown()
    # HardwareBackend wires the vehicle worker's snapshot into the MPC worker
    assert "self.mpc.mavlink_streams_fn = self.vehicle.stream_meta" in inspect.getsource(hardware.HardwareBackend)


def test_the_rov_sidecar_carries_the_request(monkeypatch):
    from rov_gui.sensorlog import SensorLog

    with tempfile.TemporaryDirectory() as tmp:
        plain = SensorLog(Path(tmp) / "a_rov.jsonl", "rov")
        plain.write("ATTITUDE", {"roll": 0.0})
        assert set(plain.close()) == {"file", "source", "duration_s", "records", "by_message", "hz_measured",
                                      "clock", "error"}
        req = {"policy": "per_message", "requested_hz": dict(MS.DEFAULT_HZ)}
        log = SensorLog(Path(tmp) / "b_rov.jsonl", "rov", extra={"stream_request": req, "records": "no"})
        log.write("ATTITUDE", {"roll": 0.0})
        m = log.close()
        side = json.loads((Path(tmp) / "b_rov.json").read_text())
        assert side["stream_request"] == req == m["stream_request"] and side["records"] == 1   # never overrides
        # through the vehicle worker's own toggle
        w, logs, kw = _setup(monkeypatch, mavlink_transport="udp", mavlink_set_rates=True, mavlink_rate=100)
        w.set_sensor_log(True, str(Path(tmp) / "take"))
        w.set_sensor_log(False, "")
        side = json.loads((Path(tmp) / "take_rov.json").read_text())
        sr = side["stream_request"]
        assert sr["policy"] == "legacy" and sr["requested_hz"] == {k: 100 for k in MS.STREAMS} and sr["sent"]
        assert "achieved_hz" not in sr


# =============================================================================
# 6. the ATTITUDE rate against the pwm LOWs' age gate
# =============================================================================
def _tick_ages(rate_hz, seconds=60.0, transit=0.010, jitter=0.010, phase=0.047, seed=0):
    """The station's chain on a synthetic healthy link: ATTITUDE generated at rate_hz on the vehicle clock, transit
    + uniform jitter, VehicleClockMap at arrival, a 20 Hz tick seeing the newest record parsed <= T - phase (the
    VehicleWorker republish, as in the fix-design replay). -> the tick ages after 5 s."""
    rng = np.random.default_rng(seed)
    n = int(seconds * rate_hz)
    tb = np.arange(n) / rate_hz
    # a FIFO link: jitter never reorders (0 backward time_boot_ms steps in ~1.3 M recorded ATTITUDEs)
    th = np.maximum.accumulate(100.0 + tb + transit + rng.uniform(0.0, jitter, n))
    m = VehicleClockMap()
    tv = np.array([m.update(th[i], tb[i] * 1000.0) for i in range(n)])
    assert m.resets == 0
    T = np.arange(th[0] + 5.0, th[-1], 0.05)
    j = np.searchsorted(th, T - phase, side="right") - 1
    return T - tv[j]


def test_a_lower_attitude_rate_adds_at_most_one_period_of_age():
    a200, a50, a20 = _tick_ages(200.0), _tick_ages(50.0), _tick_ages(20.0)
    for a, rate in ((a200, 200.0), (a50, 50.0), (a20, 20.0)):
        # never more than phase + one period + the jitter above the floor (the map sits on the least-delayed sample)
        assert a.max() <= 0.047 + 1.0 / rate + 0.010 + 1e-6, (rate, a.max())
    assert np.percentile(a50, 99) - np.percentile(a200, 99) <= 1.0 / 50.0
    assert np.percentile(a20, 99) - np.percentile(a200, 99) > 1.0 / 50.0     # 20 Hz costs more than one 50 Hz period
    # the default is not below the floor the recorded-link replay supports (impl-rates/out/age_at_rates.json:
    # 50 Hz -> rolling 10-s p99 <= 0.110 s on 21 healthy recordings; ~28 Hz -> 16 refusals) and the floor is
    # still the gate it was derived against
    from rov_gui.control.rl_pwm import PER_THRUSTER_DEFAULTS

    assert MS.DEFAULT_HZ["ATTITUDE"] >= MS.ATT_AGE_FLOOR_HZ == 50
    assert PER_THRUSTER_DEFAULTS["age_engage_p99_s"] == 0.12 and PER_THRUSTER_DEFAULTS["age_s"] == 0.15
    assert math.isfinite(float(np.percentile(a50, 99)))
