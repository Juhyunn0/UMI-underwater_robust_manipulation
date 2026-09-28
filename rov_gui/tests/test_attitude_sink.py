#!/usr/bin/env python3
"""test_attitude_sink.py — the command sink's 2026-09-26 safety-audit fixes.

Four defects an independent audit of the 6-DoF variant confirmed, each
pinned here (backends/hardware.py MavlinkCommandSink + VehicleWorker,
tools/attitude_axes_probe.py, backends/demo.py):

1. AUTOPILOT_VERSION / RC_CHANNELS are taken ONLY from the vehicle, the
   (system, component) pair the sink addresses — BlueOS routes every
   companion component's traffic to the endpoint and firmware_version is
   sticky, so a companion's answer must never become the gate's fact;
2. once an extension frame has gone out, every roll = pitch = 0 frame
   (neutral, E-STOP, deadman, teardown, a dropped K/M) clears s/t
   EXPLICITLY as ``(..., 0, 0b11, 0, 0)``; a session that never sent one
   keeps the six-positional call, byte-identical to the 4-DoF sink;
3. ``_degrade_attitude_axes`` writes ``degraded_at`` BEFORE ``degraded``
   (the telemetry thread snapshots both without a lock);
4. the demo's synthetic firmware string says "(sim)" and still parses to
   (4, 5, 1) through the worker's gate parser.

The fake master / mav / log / probe master are IMPORTED from
test_attitude_axes.py rather than duplicated.

    cd <repo> && QT_QPA_PLATFORM=offscreen python -m pytest \\
        rov_gui/tests/test_attitude_sink.py -q -p no:cacheprovider
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rov_gui.state import PilotInput  # noqa: E402
from rov_gui.tests.test_attitude_axes import (_Log, _Master,  # noqa: E402
                                              _ProbeMaster, _fake_clock,
                                              _sink)

FW_451 = (4 << 24) | (5 << 16) | (1 << 8)
FW_412 = (4 << 24) | (1 << 16) | (2 << 8)
NEUTRAL6 = (1, 0, 0, 500, 0, 0)                       # the pre-variant frame
NEUTRAL_CLEAR = (1, 0, 0, 500, 0, 0, 0, 0b11, 0, 0)   # explicit s = t = 0
RC_VEHICLE = tuple(1500 + i for i in range(1, 9))


def _app():
    from rov_gui.qt import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _SrcMsg:
    """A fake pymavlink message WITH the header accessors every real one has."""

    def __init__(self, t, sys=1, comp=1, **kw):
        self._t, self._sys, self._comp = t, sys, comp
        self.__dict__.update(kw)

    def get_type(self):
        return self._t

    def get_srcSystem(self):
        return self._sys

    def get_srcComponent(self):
        return self._comp

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


def _rc_msg(sys=1, comp=1, base=1500):
    return _SrcMsg("RC_CHANNELS", sys=sys, comp=comp,
                   **{f"chan{i}_raw": base + i for i in range(1, 19)})


def _sink_addressing(target_sysid: int):
    """Like test_attitude_axes._sink, with an explicit --target-sysid."""
    from rov_gui.backends.hardware import MavlinkCommandSink

    _app()
    bus = types.SimpleNamespace(log=_Log())
    opts = types.SimpleNamespace(cmd_sysid=255, mavlink_out="udpin:0.0.0.0:14552",
                                 target_sysid=target_sysid)
    s = MavlinkCommandSink(bus, opts)
    s.master = _Master(v2=True)
    s.listening = True
    return s, bus.log


# =============================================================================
# 1. AUTOPILOT_VERSION / RC_CHANNELS: the vehicle's only
# =============================================================================
def test_sink_drain_ignores_a_companions_autopilot_version_and_rc_channels():
    s, log = _sink()
    assert (s.target_sys, s.target_comp) == (1, 1)
    # a companion component (1/100) answers first: ignored, counted, said once
    s.master.queue = [_SrcMsg("AUTOPILOT_VERSION", comp=100, flight_sw_version=FW_412),
                      _rc_msg(comp=100, base=1000)]
    s._drain()
    assert s.firmware_version == "" and s.rc_chan_raw is None
    assert s.attitude_axes_foreign == 2
    assert sum(1 for _, m in log.lines if "ignored" in m) == 1, "said once"
    # another SYSTEM's autopilot component (2/1) is not the vehicle either
    s.master.queue = [_SrcMsg("AUTOPILOT_VERSION", sys=2, comp=1, flight_sw_version=FW_412),
                      _rc_msg(sys=2, comp=1, base=1000)]
    s._drain()
    assert s.firmware_version == "" and s.rc_chan_raw is None
    assert s.attitude_axes_foreign == 4
    # the vehicle (target_sys, target_comp): taken
    s.master.queue = [_SrcMsg("AUTOPILOT_VERSION", flight_sw_version=FW_451), _rc_msg()]
    s._drain()
    assert s.firmware_version == "4.5.1" and s.rc_chan_raw == RC_VEHICLE
    # sticky — and a later companion answer overwrites neither
    s.master.queue = [_SrcMsg("AUTOPILOT_VERSION", comp=100, flight_sw_version=FW_412),
                      _rc_msg(comp=100, base=1000)]
    s._drain()
    assert s.firmware_version == "4.5.1" and s.rc_chan_raw == RC_VEHICLE
    assert s.attitude_axes_foreign == 6
    assert sum(1 for _, m in log.lines if "ignored" in m) == 1
    # the state surface the telemetry producer reads carries the dropped counter
    st = s.attitude_axes_state()
    assert st["dropped"] == 0 and st["firmware_version"] == "4.5.1"


def test_sink_drain_addresses_the_configured_target_sysid():
    s, _ = _sink_addressing(7)
    assert (s.target_sys, s.target_comp) == (7, 1)
    s.master.queue = [_SrcMsg("AUTOPILOT_VERSION", sys=1, comp=1, flight_sw_version=FW_412),
                      _SrcMsg("AUTOPILOT_VERSION", sys=7, comp=1, flight_sw_version=FW_451)]
    s._drain()
    assert s.firmware_version == "4.5.1" and s.attitude_axes_foreign == 1


def test_vehicle_worker_fallback_keeps_only_the_vehicles_autopilot_version_and_rc():
    """The worker's own copy (the fallback when the sink has none, and the
    recording) applies the same rule at ingestion, on the logger record's
    _srcsys/_srccomp; absent ids (the REST transport) pass."""
    from rov_gui.backends.hardware import VehicleWorker

    _app()
    got: list = []
    bus = types.SimpleNamespace(
        telemetry=types.SimpleNamespace(emit=got.append),
        thrusters=types.SimpleNamespace(emit=lambda *_: None),
        log=_Log(), link=types.SimpleNamespace(emit=lambda *_: None))
    vw = VehicleWorker(bus, types.SimpleNamespace(blueos_host="127.0.0.1", target_sysid=1))

    class _Logger:
        connected = True
        transport = "udp"
        stats: dict = {}

        def __init__(self, recs):
            self.recs = list(recs)

        def drain(self):
            r, self.recs = self.recs, []
            return r

    def rec(t, sys, comp, **kw):
        return {"t_host": 0.0, "msg_type": t, "_srcsys": sys, "_srccomp": comp, **kw}

    vw._publish_thrusters = lambda: None
    now = time.monotonic()
    vw._last_link = now                    # no NIC sample in this test
    vw._last_tel = now                     # publish by hand below
    # a companion first
    vw.logger = _Logger([rec("AUTOPILOT_VERSION", 1, 100, flight_sw_version=FW_412),
                         rec("RC_CHANNELS", 1, 100, **{f"chan{i}_raw": 1000 for i in range(1, 9)})])
    vw.tick()
    assert "AUTOPILOT_VERSION" not in vw._latest and "RC_CHANNELS" not in vw._latest
    assert vw._aa_foreign == 2
    assert sum(1 for _, m in bus.log.lines if "ignored" in m) == 1
    vw._publish()
    assert got[-1].firmware_version == "" and got[-1].rc_chan_raw is None
    # the vehicle
    vw.logger = _Logger([rec("AUTOPILOT_VERSION", 1, 1, flight_sw_version=FW_451),
                         rec("RC_CHANNELS", 1, 1, **{f"chan{i}_raw": 1500 + i for i in range(1, 9)})])
    vw.tick()
    vw._publish()
    assert got[-1].firmware_version == "4.5.1" and got[-1].rc_chan_raw == RC_VEHICLE
    # a later companion / other-system answer does not overwrite the vehicle's
    vw.logger = _Logger([rec("AUTOPILOT_VERSION", 1, 100, flight_sw_version=FW_412),
                         rec("RC_CHANNELS", 2, 1, **{f"chan{i}_raw": 1000 for i in range(1, 9)})])
    vw.tick()
    vw._publish()
    assert got[-1].firmware_version == "4.5.1" and got[-1].rc_chan_raw == RC_VEHICLE
    assert vw._aa_foreign == 4
    # the REST transport carries no source ids (scoped by its URL): accepted
    vw.logger = _Logger([{"t_host": 0.0, "msg_type": "AUTOPILOT_VERSION",
                          "flight_sw_version": FW_412}])
    vw.tick()
    vw._publish()
    assert got[-1].firmware_version == "4.1.2"


# =============================================================================
# 2. the explicit s/t clear — only after an extension frame
# =============================================================================
def test_sink_clears_s_t_explicitly_only_after_an_extension_frame():
    s, _ = _sink()
    s.set_attitude_axes(True)
    # before any extension frame: the six-positional call, byte-identical
    s._send(PilotInput(), buttons=0)
    s._send(PilotInput(surge=0.5, stamp=1.0), buttons=7)
    assert s.master.mav.calls == [NEUTRAL6, (1, 500, 0, 500, 0, 7)]
    assert not s.attitude_axes_ext_sent
    # one extension frame ...
    s._send(PilotInput(roll=0.2, pitch=-0.3, stamp=1.0), buttons=0)
    assert s.master.mav.calls[-1] == (1, 0, 0, 500, 0, 0, 0, 0b11, -300, 200)
    assert s.attitude_axes_ext_sent
    # ... and every roll = pitch = 0 frame after it clears s/t explicitly.
    # PilotInput() is exactly what E-STOP, the deadman and teardown send.
    s._send(PilotInput(), buttons=0)
    assert s.master.mav.calls[-1] == NEUTRAL_CLEAR
    s._send(PilotInput(surge=0.5, sway=-0.25, heave=0.1, yaw=0.3, stamp=1.0), buttons=7)
    assert s.master.mav.calls[-1] == (1, 500, -250, 550, 300, 7, 0, 0b11, 0, 0)
    # the axes switched OFF afterwards (a worker disengage): K/M dropped and
    # counted, the clear still explicit
    s.set_attitude_axes(False)
    s._send(PilotInput(roll=0.4, stamp=1.0), buttons=0)
    assert s.master.mav.calls[-1] == NEUTRAL_CLEAR
    assert s.attitude_axes_dropped == 1
    st = s.attitude_axes_state()
    assert st["dropped"] == 1 and st["ext_sent"] and not st["degraded"]


def test_session_without_an_extension_frame_never_grows_the_frame():
    s, _ = _sink()
    s._send(PilotInput(roll=0.3, pitch=0.1, stamp=1.0), buttons=0)   # OFF: dropped
    s.set_attitude_axes(True)
    for _ in range(3):
        s._send(PilotInput(surge=0.1, stamp=1.0), buttons=1)
    s._send(PilotInput(), buttons=0)
    assert len(s.master.mav.calls) == 5 and all(len(c) == 6 for c in s.master.mav.calls)
    assert not s.attitude_axes_ext_sent and s.attitude_axes_dropped == 1


def test_explicit_clear_that_the_link_refuses_falls_back_and_degrades():
    """A TypeError on the clear frame (it cannot happen on a link that took
    the extension frame, but the loop must still terminate) re-sends the
    six-argument neutral and degrades — never raises out of _send."""
    s, log = _sink()
    s.set_attitude_axes(True)
    s._send(PilotInput(roll=0.2, stamp=1.0), buttons=0)
    assert s.attitude_axes_ext_sent
    s.master.mav.raise_on_ext = True
    s._send(PilotInput(), buttons=0)
    assert s.master.mav.calls[-1] == NEUTRAL6
    assert s.attitude_axes_degraded and s.attitude_axes_degraded_at is not None
    assert any(lvl == "error" and "DEGRADED" in m for lvl, m in log.lines)


def test_explicit_clear_is_on_the_wire_with_the_real_packer():
    """The clear frame is NOT zero-truncated away: enabled_extensions = 3
    survives packing (26 bytes vs the 23-byte plain neutral) and decodes
    back to (3, 0, 0)."""
    pytest.importorskip("pymavlink")
    from pymavlink.dialects.v20 import ardupilotmega as m20

    class Buf:
        def write(self, b):
            pass

    mav = m20.MAVLink(Buf(), srcSystem=255, srcComponent=190)
    mav.seq = 5
    plain = mav.manual_control_encode(*NEUTRAL6).pack(mav)
    mav.seq = 5
    clear = mav.manual_control_encode(*NEUTRAL_CLEAR).pack(mav)
    assert len(plain) == 23 and len(clear) == 26 and clear != plain
    back = mav.decode(bytearray(clear))
    assert (back.enabled_extensions, back.s, back.t) == (3, 0, 0)


# =============================================================================
# 3. degrade ordering
# =============================================================================
def test_degrade_writes_degraded_at_before_degraded():
    from rov_gui.backends.hardware import MavlinkCommandSink

    _app()
    order: list[tuple] = []

    class Spy(MavlinkCommandSink):
        def __setattr__(self, k, v):
            if k in ("attitude_axes_degraded", "attitude_axes_degraded_at"):
                # what a lock-free reader sees the instant before this write
                order.append((k, self.__dict__.get("attitude_axes_degraded"),
                              self.__dict__.get("attitude_axes_degraded_at")))
            super().__setattr__(k, v)

    bus = types.SimpleNamespace(log=_Log())
    s = Spy(bus, types.SimpleNamespace(cmd_sysid=255, mavlink_out="udpin:0.0.0.0:14552"))
    order.clear()
    s._degrade_attitude_axes("test")
    assert [k for k, *_ in order] == ["attitude_axes_degraded_at", "attitude_axes_degraded"]
    _, degraded_before, at_before = order[1]
    assert degraded_before is False and at_before is not None, \
        "degraded flips only once degraded_at is already readable"
    st = s.attitude_axes_state()
    assert st["degraded"] and st["degraded_at"] is not None
    # idempotent: a second degrade neither rewrites the stamp nor logs again
    at = s.attitude_axes_degraded_at
    n = len(bus.log.lines)
    s._degrade_attitude_axes("again")
    assert s.attitude_axes_degraded_at == at and len(bus.log.lines) == n


# =============================================================================
# 4. the demo's firmware string is synthetic on its face — and still gates
# =============================================================================
def test_demo_firmware_string_is_marked_synthetic_and_still_parses():
    from rov_gui.backends.demo import DemoVehicleWorker
    from rov_gui.control.geometry import parse_firmware_version
    from rov_gui.control.workers import MpcWorker

    _app()
    got: list = []
    bus = types.SimpleNamespace(
        log=_Log(),
        telemetry=types.SimpleNamespace(emit=got.append),
        thrusters=types.SimpleNamespace(emit=lambda *_: None),
        payload=types.SimpleNamespace(emit=lambda *_: None),
        link=types.SimpleNamespace(emit=lambda *_: None))
    w = DemoVehicleWorker(bus, opts=types.SimpleNamespace(mpc=False))
    st = w.attitude_axes_state()
    assert st["firmware_version"] == "4.5.1 (sim)"
    assert st["mavlink_wire_version"] == "2.0", "the wire string is compared exactly"
    assert MpcWorker._parse_fw(st["firmware_version"]) == (4, 5, 1)
    assert parse_firmware_version(st["firmware_version"]) == (4, 5, 1)
    # the Telemetry it publishes says the same
    w._publish([0.0] * 8, 1.0, 16.0, 80.0)
    t = got[-1]
    assert t.firmware_version == "4.5.1 (sim)" and t.mavlink_wire_version == "2.0"
    assert MpcWorker._parse_fw(t.firmware_version) == (4, 5, 1)
    assert MpcWorker._parse_fw(t.firmware_version) >= (4, 1, 2)


# =============================================================================
# the bench probe: same two rules
# =============================================================================
class _CompanionFirstProbeMaster(_ProbeMaster):
    """A companion component (1/100) answers AUTOPILOT_VERSION 4.1.2 and
    another system (2/1) posts flat RC_CHANNELS BEFORE the vehicle (1/1)
    does, every cycle — what the BlueOS endpoint fan-out looks like."""

    def __init__(self, vehicle_answers: bool = True, **kw):
        super().__init__(**kw)
        self.vehicle_answers = vehicle_answers

    def recv_msg(self):
        self._n += 1
        k = self._n % 5
        if k == 0 or (k == 3 and not self.vehicle_answers):
            return _SrcMsg("HEARTBEAT", base_mode=(128 if self.armed else 0), custom_mode=19)
        if k == 1:
            return _SrcMsg("AUTOPILOT_VERSION", comp=100, flight_sw_version=FW_412)
        if k == 2:
            return _SrcMsg("RC_CHANNELS", sys=2, comp=1,
                           **{f"chan{i}_raw": 1000 for i in range(1, 9)})
        if k == 3:
            return _SrcMsg("AUTOPILOT_VERSION", flight_sw_version=FW_451)
        gain = 0.4 if self.reads_st else 0.0
        return _SrcMsg("RC_CHANNELS", **{"chan1_raw": int(1500 + gain * self.s),
                                          "chan2_raw": int(1500 + gain * self.t),
                                          **{f"chan{i}_raw": 1500 for i in range(3, 9)}})


def test_probe_takes_firmware_and_rc_only_from_the_addressed_vehicle():
    from rov_gui.tools import attitude_axes_probe as P

    clock, sleep = _fake_clock()
    m = _CompanionFirstProbeMaster()
    res = P.run_probe(m, amplitude=300, hold_s=1.0, settle_s=0.5, rate_hz=10.0,
                      clock=clock, sleep=sleep, log=lambda *_: None)
    assert res["firmware_version"] == "4.5.1", "the companion's 4.1.2 did not win"
    assert res["pass"], "the other system's flat RC did not dilute the medians"
    assert res["foreign_msgs_ignored"] > 0 and res["target"] == {"sys": 1, "comp": 1}
    # only the companion answers AUTOPILOT_VERSION: the firmware stays
    # unread — "" is what the engage gate refuses on
    clock, sleep = _fake_clock()
    res2 = P.run_probe(_CompanionFirstProbeMaster(vehicle_answers=False), amplitude=300,
                       hold_s=1.0, settle_s=0.5, rate_hz=10.0,
                       clock=clock, sleep=sleep, log=lambda *_: None)
    assert res2["firmware_version"] == ""
    # addressing another system id: the 1/1 answers are the foreign ones
    clock, sleep = _fake_clock()
    res3 = P.run_probe(_CompanionFirstProbeMaster(), amplitude=300, hold_s=1.0,
                       settle_s=0.5, rate_hz=10.0, target_sys=9,
                       clock=clock, sleep=sleep, log=lambda *_: None)
    assert res3["firmware_version"] == "" and res3["target"] == {"sys": 9, "comp": 1}


def test_probe_clears_s_t_explicitly_then_ends_on_the_plain_neutral():
    from rov_gui.tools import attitude_axes_probe as P

    clock, sleep = _fake_clock()
    m = _ProbeMaster(reads_st=True)
    res = P.run_probe(m, amplitude=300, hold_s=1.0, settle_s=0.5, rate_hz=10.0,
                      clock=clock, sleep=sleep, log=lambda *_: None)
    assert res["pass"]
    f = m.frames
    first_ext = next(i for i, fr in enumerate(f) if len(fr) > 6)
    assert first_ext > 0 and all(fr == NEUTRAL6 for fr in f[:first_ext]), \
        "baseline: the pre-variant frame"
    assert all(len(fr) == 10 and fr[7] == 0b11 for fr in f[first_ext:-1]), \
        "once s/t went out every frame carries the extension explicitly"
    assert any(fr[8:] == (0, 0) for fr in f[first_ext:-1]), \
        "the neutral phases clear s/t explicitly"
    assert f[-2] == NEUTRAL_CLEAR and f[-1] == NEUTRAL6, "explicit clear, THEN the plain neutral"
    # the abort path (armed mid-probe) ends the same way
    clock, sleep = _fake_clock()
    m = _ProbeMaster(arm_after=8)
    with pytest.raises(P.ProbeAbort, match="ARMED during"):
        P.run_probe(m, hold_s=1.0, settle_s=0.5, clock=clock, sleep=sleep, log=lambda *_: None)
    assert any(len(fr) > 6 and fr[8:] != (0, 0) for fr in m.frames), "s/t had gone out"
    assert m.frames[-2] == NEUTRAL_CLEAR and m.frames[-1] == NEUTRAL6
    # armed at the start: refused before any frame — nothing to clear
    clock, sleep = _fake_clock()
    m = _ProbeMaster(armed=True)
    with pytest.raises(P.ProbeAbort, match="ARMED"):
        P.run_probe(m, clock=clock, sleep=sleep, log=lambda *_: None)
    assert m.frames == []
