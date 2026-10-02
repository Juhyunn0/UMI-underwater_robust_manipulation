#!/usr/bin/env python3
"""test_rl_pwm.py — LOW mode ``rl_pwm`` (2026-09-30): the per-thruster RL policy, offline.

    cd <repo> && QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/rovgui-pose/bin/python -m pytest \
        rov_gui/tests/test_rl_pwm.py -q -p no:cacheprovider

No vehicle. What is pinned, layer by layer:

1. control/rl_pwm.py — the wire frame (motors on RC 9..16 IN THAT ORDER, every other channel 65535, pulses always
   explicit 1100..1900), its bytes under the real pymavlink v2 dialect against the offsets the vehicle's Lua script
   reads, the throttle <-> pulse round trip, and the vehicle gate item by item, including that a parameter the vehicle
   said more than 30 s ago no longer counts.
2. control/rl_policy.HwRlPwm — builds from the shipped pwm export, 116-wide observation, pulses inside the cap, the
   APPLIED pulses fed back as the previous action, a broken state -> eight 1500s and a counted failure; HwRl still
   refuses a pwm export and HwRlPwm an axes one.
3. backends/hardware.MavlinkCommandSink — nothing on RC 9..16 unless the gate holds AND COMMAND ENABLE is on; its own
   cap; a gate lost mid-flow is LATCHED until the follower's stop; a stalled follower, E-STOP, disable and teardown each
   end in neutral pulses and then silence; a session that never used the path never sends an RC override, and neither
   does one that finished using it; parameters are asked for only while somebody wants the transport.
4. control/workers.MpcWorker — the mode exists, refuses without a gate / outside MANUAL / on a mismatched export, tells
   the sink it is selected, emits pulses + a NEUTRAL MANUAL_CONTROL while engaged, and drops the engagement (with eight
   1500s) when the gate is lost.

The Lua script on the vehicle and the thrusters themselves are NOT covered by anything here: that is the bench test.
"""

from __future__ import annotations

import os
import re
import struct
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from rov_gui.control.rl_pwm import (NEUTRAL_PULSES, PARAM_FRESH_S, PWM_CHANNELS_DEFAULT, PwmGate,  # noqa: E402
                                    pack_rc_override, pwm_to_action, throttles_to_pwm)
from rov_gui.state import Conn, PilotInput, PwmCommand, Telemetry, now  # noqa: E402

#: the shipped export's motor directions (rl_policies/pwm10_s1/obs_spec.json output.pwm_model.motor_direction)
MOT = [-1, -1, 1, 1, -1, 1, 1, -1]


# =============================================================================
# 1. the wire frame and the gate
# =============================================================================
def test_pack_puts_the_motors_on_rc_9_to_16_and_leaves_everything_else():
    pw = (1600, 1400, 1510, 1490, 1555, 1445, 1500, 1501)
    ch = pack_rc_override(pw)
    assert len(ch) == 18
    assert ch[:8] == [65535] * 8, "the joystick channels 1..8 are never touched"
    assert tuple(ch[8:16]) == pw
    assert ch[16:] == [65535, 65535]
    # 0 / 65535 / 65534 mean leave / leave / release on 9..16: a pulse is clamped into 1100..1900, never one of those
    ch = pack_rc_override((0, 65535, 65534, 1099, 1901, 1500, 1500, 1500))
    assert ch[8:13] == [1100, 1900, 1900, 1100, 1900]
    for bad in ((1500,) * 7, (1500,) * 9):
        with pytest.raises(ValueError):
            pack_rc_override(bad)


def test_the_channel_order_is_not_configurable_anywhere():
    """The vehicle's script maps RC channel 8+m to motor m. Any other order would pass every other check and swap
    thrusters under a closed-loop policy, so it is refused in the frame, in the gate and in the config."""
    from rov_gui.control.geometry import MpcConfig

    for bad in ((16, 15, 14, 13, 12, 11, 10, 9), (9, 10, 11, 12, 13, 14, 16, 15), (1, 2, 3, 4, 5, 6, 7, 8),
                (9, 10, 11, 12, 13, 14, 15)):
        with pytest.raises(ValueError, match="in that order"):
            pack_rc_override((1500,) * 8, bad)
        with pytest.raises(ValueError, match="in that order"):
            PwmGate(MOT, 255, bad)
    assert PwmGate(MOT, 255).channels == PWM_CHANNELS_DEFAULT
    src = (ROOT / "config" / "hw_mpc.yaml").read_text()
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(src.replace("  pwm_cap: 0.25", "  pwm_cap: 0.25\n  pwm_channels: [16, 15, 14, 13, 12, 11, 10, 9]", 1))
    try:
        with pytest.raises(ValueError, match="unknown key"):
            MpcConfig.load(fh.name)
    finally:
        os.unlink(fh.name)


def test_real_pymavlink_v2_bytes_match_the_offsets_the_lua_script_reads():
    """rl_pwm_override.lua reads target_system at payload byte 17 and chan(8+m) at byte 19 + 2(m-1) (1-based). Pack a
    frame with the REAL dialect and look at those bytes — the fake mav in the other tests cannot say this."""
    os.environ["MAVLINK20"] = "1"
    try:
        from pymavlink.dialects.v20 import ardupilotmega as mavlink
    except Exception as e:                                       # noqa: BLE001
        pytest.skip(f"pymavlink v2 dialect unavailable: {e}")
    pw = (1600, 1400, 1550, 1450, 1520, 1480, 1500, 1590)
    mav = mavlink.MAVLink(None, srcSystem=255, srcComponent=190)
    buf = mav.rc_channels_override_encode(1, 1, *pack_rc_override(pw)).pack(mav)
    assert buf[0] == 0xFD and buf[5] == 255, "MAVLink 2, from sysid 255"
    assert buf[7] | buf[8] << 8 | buf[9] << 16 == 70, "RC_CHANNELS_OVERRIDE"
    payload = buf[10:10 + buf[1]]
    assert len(payload) == 38, "nothing trimmed: chan17/18 are 65535, not 0"
    assert payload[16] == 1, "target_system where the script reads it"
    assert struct.unpack("<8H", payload[18:34]) == pw, "chan9..16 where the script reads them"
    assert struct.unpack("<8H", payload[0:16]) == (65535,) * 8, "chan1..8 = leave"


def test_throttle_pulse_round_trip_and_cap():
    thr = np.array([1.0, -1.0, 0.25, -0.25, 0.1, 0.0, 0.6, -0.6])
    assert throttles_to_pwm(thr) == (1900, 1100, 1600, 1400, 1540, 1500, 1740, 1260)
    capped = throttles_to_pwm(thr, cap=0.25)
    assert max(capped) == 1600 and min(capped) == 1400
    assert np.allclose(pwm_to_action(capped), np.clip(thr, -0.25, 0.25), atol=1e-9)
    assert np.allclose(pwm_to_action(NEUTRAL_PULSES), 0.0)
    with pytest.raises(ValueError):
        throttles_to_pwm([float("nan")] * 8)


def _good_gate(t=100.0, sysid=255) -> PwmGate:
    g = PwmGate(MOT, sysid)
    for n in range(1, 9):
        g.note_param(f"SERVO{n}_FUNCTION", 32 + n, t)
        g.note_param(f"MOT_{n}_DIRECTION", MOT[n - 1], t)
    for n in range(9, 17):
        g.note_param(f"SERVO{n}_FUNCTION", 0, t)
    for c in PWM_CHANNELS_DEFAULT:
        g.note_param(f"RC{c}_OPTION", 0, t)
    for k, v in (("SCR_ENABLE", 1), ("SCR_USER1", 1), ("RC_OPTIONS", 0), ("RC_OVERRIDE_TIME", 3.0),
                 ("SYSID_MYGCS", sysid)):
        g.note_param(k, v, t)
    g.note_heartbeat(0.0, t)
    return g


def test_gate_passes_only_when_every_item_holds_and_names_what_does_not():
    t = 100.0
    assert _good_gate(t).check(t) == (True, "ok")
    assert _good_gate(t).missing(t) == []

    def broken(fn, **kw):
        g = _good_gate(t)
        fn(g)
        ok, why = g.check(kw.get("now", t), kw.get("link_v2", True))
        assert not ok
        return why

    assert "MAVLink 1" in broken(lambda g: None, link_v2=False)
    assert "Motor1..8" in broken(lambda g: g.note_param("SERVO3_FUNCTION", 53, t))          # RCIN3 passthrough
    assert "follow a thruster" in broken(lambda g: g.note_param("SERVO13_FUNCTION", 59, t))  # lights on RCIN9
    assert "follow a thruster" in broken(lambda g: g.note_param("SERVO14_FUNCTION", 148, t))  # RCIN9Scaled
    assert "spin reversed" in broken(lambda g: g.note_param("MOT_4_DIRECTION", -1, t))
    assert "SCR_ENABLE" in broken(lambda g: g.note_param("SCR_ENABLE", 0, t))
    assert "SCR_USER1" in broken(lambda g: g.note_param("SCR_USER1", 0, t))
    assert "RC_OPTIONS" in broken(lambda g: g.note_param("RC_OPTIONS", 2, t))
    assert "RC_OVERRIDE_TIME" in broken(lambda g: g.note_param("RC_OVERRIDE_TIME", 0, t))
    assert "aux function" in broken(lambda g: g.note_param("RC11_OPTION", 7, t))
    assert "SYSID_MYGCS" in broken(lambda g: g.note_param("SYSID_MYGCS", 254, t))
    assert "stale" in broken(lambda g: None, now=t + 2.5)
    # never heard the script at all
    g = _good_gate(t)
    g.t_hb = None
    assert "not running" in g.check(t)[1]
    # a parameter the vehicle never answered (no scripting in the firmware)
    g = PwmGate(MOT, 255)
    ok, why = g.check(t)
    assert not ok and "not read" in why and len(g.missing(t)) == len(g.param_names()) == 16 + 8 + 5 + 8
    # a NaN answer is no answer, and a name the gate does not know is ignored
    g = _good_gate(t)
    g.note_param("SCR_USER1", float("nan"), t + 1.0)
    g.note_param("SOMETHING_ELSE", 5, t)
    assert g.params["SCR_USER1"] == 1 and "SOMETHING_ELSE" not in g.params


def test_gate_stops_trusting_a_parameter_the_vehicle_said_long_ago():
    """A reboot, a parameter file loaded in QGC, a flipped MOT_n_DIRECTION: none of them may be judged from a value
    cached at start-up. Past PARAM_FRESH_S every parameter counts as unread, and a re-read is what clears it."""
    t = 100.0
    g = _good_gate(t)
    assert g.check(t + PARAM_FRESH_S - 1.0)[0] or "stale" in g.check(t + PARAM_FRESH_S - 1.0)[1]   # only the heartbeat
    g.note_heartbeat(0.0, t + PARAM_FRESH_S + 1.0)
    ok, why = g.check(t + PARAM_FRESH_S + 1.0)
    assert not ok and "not read in the last" in why
    # the vehicle answers again — and this time one direction is different
    for n in range(1, 9):
        g.note_param(f"MOT_{n}_DIRECTION", MOT[n - 1] if n != 6 else -MOT[5], t + PARAM_FRESH_S + 2.0)
    for name in g.param_names():
        if not name.startswith("MOT_"):
            v = g.servo_functions[int(name[5:-9])] if name.startswith("SERVO") else g.params[name]
            g.note_param(name, v, t + PARAM_FRESH_S + 2.0)
    g.note_heartbeat(0.0, t + PARAM_FRESH_S + 2.0)
    ok, why = g.check(t + PARAM_FRESH_S + 2.0)
    assert not ok and "spin reversed" in why


def test_gate_requires_the_script_to_take_over_within_a_second_of_pulses():
    t = 100.0
    g = _good_gate(t)                              # heartbeat value 0: the script is idle, which is fine before pulses
    for k in range(11):
        g.note_pulses(t + 0.05 * k)                # 0.5 s of policy frames at 20 Hz
    assert g.check(t + 0.5)[0], "half a second in, an idle script is not a fault yet"
    for k in range(11, 25):
        g.note_pulses(t + 0.05 * k)
    ok, why = g.check(t + 1.2)
    assert not ok and "not forcing the outputs" in why
    g.note_heartbeat(1.0, t + 1.3)                 # ...and once it says 1 the gate holds, and the count starts over
    assert g.check(t + 1.3)[0] and g.unconfirmed_s == 0.0


def test_gate_counts_unconfirmed_pulse_time_across_pauses():
    """A flow that is interrupted more than once a second (a coast at the edge of the tags, a stalled tick) used to
    restart the grace timer every time and was never judged: pulses nobody was acting on, live again the moment the
    script's condition was met (safety re-review 2026-09-30). The time is accumulated; only the END of the
    engagement, or the script saying 1, resets it."""
    t = 100.0
    g = _good_gate(t)
    now_ = t
    for burst in range(4):                         # 4 x 0.4 s of pulses with pauses between them
        for k in range(9):
            g.note_pulses(now_ + 0.05 * k)
        now_ += 0.4
        g.note_paused()
        now_ += 0.6                                # the pause itself is not counted
        g.note_heartbeat(0.0, now_)
    assert g.unconfirmed_s > 1.0
    assert not g.check(now_)[0] and "not forcing" in g.check(now_)[1]
    g.note_stopped()                               # the engagement ends: the next one starts from zero
    assert g.unconfirmed_s == 0.0 and g.check(now_)[0]


def test_gate_asks_in_small_batches_and_refreshes_what_it_already_knows():
    g = PwmGate(MOT, 255)
    first = g.next_requests(10.0)
    assert len(first) == 8 and len(g.next_requests(10.05)) == 8
    assert not set(first) & set(g.next_requests(10.1)), "nothing is re-asked inside the retry interval"
    assert set(g.next_requests(10.6)) & set(first), "unanswered ones are asked again after it"
    g2 = _good_gate(100.0)
    assert g2.next_requests(105.0) == [], "fresh values are left alone"
    assert len(g2.next_requests(111.0)) == 8, "...and re-read every PARAM_REFRESH_S"


# =============================================================================
# 2. the controller
# =============================================================================
def test_hw_rl_pwm_builds_from_the_shipped_export_and_stays_inside_its_cap():
    from rov_gui.control.geometry import MpcConfig
    from rov_gui.control.rl_policy import HwRl, HwRlPwm

    said = []
    r = HwRlPwm(MpcConfig(), log=said.append)
    assert r.transport == "pwm" and r.mode == "rl_pwm" and r.n_act == 8
    assert r.policy.obs_dim == 116 and r.frame_dim == 29 and r.K == 4
    assert r.pwm_cap == 0.25, "the shipped default is the first-water cap"
    assert r.model.motor_direction == MOT and r.pwm_channels == PWM_CHANNELS_DEFAULT
    assert "UNTESTED" in said[0]
    r.set_target_ned([0.3, 0.0, -0.5], 0.0)
    u, info = r.step(np.zeros(6), np.zeros(6), np.zeros(6), 0.0)
    pw = info["pwm_us"]
    assert len(pw) == 8 and all(isinstance(p, int) for p in pw) and info["status"] == 0
    assert all(1400 <= p <= 1600 for p in pw), pw
    assert u.shape == (6,) and np.all(np.isfinite(u))
    assert u[0] > 0.0, "a goal 0.3 m ahead asks for forward thrust (nominal wrench estimate)"
    # the APPLIED pulses are the next previous action; the axes feedback is ignored
    r.note_applied(np.full(6, 99.0))
    assert np.allclose(r._prev_action, 0.0)
    r.note_applied_pwm(pw)
    assert np.allclose(r._prev_action, (np.array(pw) - 1500) / 400.0)
    assert r.observation(np.zeros(6), np.zeros(6), 0.05).shape == (116,)
    r.note_applied_pwm(None)
    assert np.allclose(r._prev_action, 0.0) and r.last_pwm_us == NEUTRAL_PULSES
    m = r.meta()
    assert m["type"] == "rl_pwm" and m["action_mode"] == "pwm" and m["transport"] == "pwm_lua_override"
    assert m["pwm_cap"] == 0.25 and m["motor_direction_expected"] == MOT and m["axis_cap"] is None
    # the axes controller is untouched and still an axes controller
    assert HwRl(MpcConfig(), log=lambda s: None).meta()["action_mode"] == "axes"


def test_hw_rl_pwm_answers_a_broken_state_with_neutral_and_a_counted_failure():
    from rov_gui.control.geometry import MpcConfig
    from rov_gui.control.rl_policy import HwRlPwm

    r = HwRlPwm(MpcConfig(), log=lambda s: None)
    r.set_target_ned([0.3, 0.0, -0.5], 0.0)
    n0 = r.n_fail
    eta = np.zeros(6)
    eta[0] = float("nan")
    u, info = r.step(eta, np.zeros(6), np.zeros(6), 0.0)
    assert info["pwm_us"] == NEUTRAL_PULSES and info["status"] == 1 and info["n_fail"] == n0 + 1
    assert np.allclose(u, 0.0)


def test_hw_rl_pwm_refuses_an_axes_export_and_a_cap_that_is_not_a_number_in_0_1():
    from rov_gui.control.geometry import MpcConfig
    from rov_gui.control.rl_policy import DEFAULT_POLICY_DIR, HwRlPwm

    cfg = MpcConfig()
    cfg.rl_pwm = {"policy_dir": DEFAULT_POLICY_DIR}          # the shipped AXES policy
    with pytest.raises(ValueError, match="rl_pwm.*'axes' policy.*pwm"):
        HwRlPwm(cfg, log=lambda s: None)
    for bad in (0.0, 1.5, float("nan"), True, "0.5"):
        cfg = MpcConfig()
        cfg.rl_pwm = {"pwm_cap": bad}
        with pytest.raises(ValueError, match="pwm_cap"):
            HwRlPwm(cfg, log=lambda s: None)
    cfg = MpcConfig()
    cfg.rl_pwm = {"pwm_cap": 1.0}
    assert HwRlPwm(cfg, log=lambda s: None).pwm_cap == 1.0


def test_config_loads_the_rl_pwm_block_and_rejects_unknown_keys():
    from rov_gui.control.geometry import MpcConfig

    cfg = MpcConfig.load("config/hw_mpc.yaml")
    assert cfg.rl_pwm["policy_dir"].endswith("pwm10_s1") and cfg.rl_pwm["pwm_cap"] == 0.25
    src = (ROOT / "config" / "hw_mpc.yaml").read_text()

    def load(text):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
            fh.write(text)
        try:
            return MpcConfig.load(fh.name)
        finally:
            os.unlink(fh.name)

    with pytest.raises(ValueError, match="unknown key"):
        load(src.replace("  pwm_cap: 0.25", "  pwm_cap: 0.25\n  pwm_capp: 0.9", 1))
    for bad in ("1.5", "true"):
        with pytest.raises(ValueError, match="pwm_cap"):
            load(src.replace("  pwm_cap: 0.25", f"  pwm_cap: {bad}", 1))
    assert re.search(r"^mode: \w+", src, re.M), "the shipped config names a mode at the start of a line"
    assert load(re.sub(r"^mode: \w+", "mode: rl_pwm", src, count=1, flags=re.M)).mode == "rl_pwm"


# =============================================================================
# 3. the command sink
# =============================================================================
class _Mav:
    def __init__(self, v2=True):
        self.rc: list = []
        self.manual: list = []
        self.params: list = []
        self.v2 = v2

    def rc_channels_override_send(self, *args):
        if not self.v2 and len(args) > 10:
            raise TypeError("rc_channels_override_send() takes 11 positional arguments")
        self.rc.append(tuple(args))

    def manual_control_send(self, *args):
        self.manual.append(tuple(args))

    def param_request_read_send(self, *args):
        self.params.append(tuple(args))

    def heartbeat_send(self, *a):
        pass

    def command_long_send(self, *a):
        pass


class _Master:
    def __init__(self, v2=True):
        self.mav = _Mav(v2)
        self.clients = {("1.2.3.4", 1)}
        self._v2 = v2
        self.queue: list = []

    def mavlink20(self):
        return self._v2

    def recv_msg(self):
        return self.queue.pop(0) if self.queue else None

    def close(self):
        pass


class _Log:
    def __init__(self):
        self.lines: list = []

    def emit(self, lvl, msg):
        self.lines.append((lvl, msg))


class _Msg:
    def __init__(self, t, sys=1, comp=1, **kw):
        self._t, self._sys, self._comp = t, sys, comp
        self.__dict__.update(kw)

    def get_type(self):
        return self._t

    def get_srcSystem(self):
        return self._sys

    def get_srcComponent(self):
        return self._comp


def _sink(v2=True):
    from rov_gui.backends.hardware import MavlinkCommandSink
    from rov_gui.qt import QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    bus = types.SimpleNamespace(log=_Log(), payload=types.SimpleNamespace(emit=lambda *_: None))
    opts = types.SimpleNamespace(cmd_sysid=255, mavlink_out="udpin:0.0.0.0:14552")
    s = MavlinkCommandSink(bus, opts)
    s.master = _Master(v2)
    s.listening = True
    s._setup_pwm_gate()
    assert s._pwm_gate is not None, s._pwm_gate_error
    assert s._pwm_cap == 0.25
    return s, bus.log


def _good_values(**over):
    vals = {f"SERVO{n}_FUNCTION": 32 + n for n in range(1, 9)}
    vals.update({f"SERVO{n}_FUNCTION": 0 for n in range(9, 17)})
    vals.update({f"MOT_{n}_DIRECTION": MOT[n - 1] for n in range(1, 9)})
    vals.update({f"RC{c}_OPTION": 0 for c in PWM_CHANNELS_DEFAULT})
    vals.update({"SCR_ENABLE": 1, "SCR_USER1": 1, "RC_OPTIONS": 0, "RC_OVERRIDE_TIME": 3.0, "SYSID_MYGCS": 255})
    vals.update(over)
    return vals


def _answer_all(s, hb=0.0, **over):
    """The vehicle answers every gate parameter (good values unless overridden) and the script heartbeats."""
    s.master.queue = [_Msg("PARAM_VALUE", param_id=k, param_value=float(v)) for k, v in _good_values(**over).items()]
    s.master.queue.append(_Msg("NAMED_VALUE_FLOAT", name="RLPWM", value=hb))
    for _ in range(4):                                # _drain is bounded per call
        s._drain()


PW = (1600, 1400, 1550, 1450, 1520, 1480, 1500, 1590)


def _motors(frame):
    return frame[10:18]


def test_sink_sends_nothing_on_rc_until_the_gate_holds_and_enable_is_on():
    s, log = _sink()
    s.enabled = True
    s.set_pwm(PwmCommand(pulses=PW, source="mpc"))
    assert s.master.mav.rc == [], "no parameters read yet: refused"
    assert sum(1 for lvl, m in log.lines if lvl == "error" and "NOT sent" in m) == 1
    s.set_pwm(PwmCommand(pulses=PW, source="mpc"))
    assert sum(1 for lvl, m in log.lines if lvl == "error" and "NOT sent" in m) == 1, "said once per reason"
    _answer_all(s)
    s.enabled = False
    s.set_pwm(PwmCommand(pulses=PW, source="mpc"))
    assert s.master.mav.rc == [] and not s._pwm_active
    assert any("COMMAND ENABLE" in m for _, m in log.lines)
    s.enabled = True
    s.set_pwm(PwmCommand(pulses=PW, source="mpc"))
    assert len(s.master.mav.rc) == 1 and s._pwm_active
    frame = s.master.mav.rc[0]
    assert frame[:2] == (1, 1), "addressed to the vehicle's autopilot"
    ch = frame[2:]
    assert len(ch) == 18 and ch[:8] == (65535,) * 8 and ch[8:16] == PW and ch[16:] == (65535, 65535)
    assert s.master.mav.manual == [], "set_pwm itself sends no MANUAL_CONTROL — the sink's tick does, neutral"


def test_sink_enforces_the_cap_itself_and_never_sends_a_frame_that_arrives_old():
    s, log = _sink()
    _answer_all(s)
    s.enabled = True
    s.set_pwm(PwmCommand(pulses=(1900, 1100, 1700, 1300, 1601, 1399, 1500, 1600), source="mpc"))
    assert _motors(s.master.mav.rc[-1]) == (1600, 1400, 1600, 1400, 1600, 1400, 1500, 1600), \
        "clamped to 1500 +- 400 x 0.25 whatever the follower asked"
    # ONE late frame (a GIL pause) is dropped, not sent, and the flow carries on with the next fresh one
    n = len(s.master.mav.rc)
    s.set_pwm(PwmCommand(pulses=PW, source="mpc", stamp=now() - 1.0))
    assert len(s.master.mav.rc) == n and s._pwm_active and not s._pwm_latch
    s.set_pwm(PwmCommand(pulses=PW, source="mpc"))
    assert _motors(s.master.mav.rc[-1]) == (1600, 1400, 1550, 1450, 1520, 1480, 1500, 1590) and s._pwm_late_run == 0
    # a sink that is late EVERY time is latched shut
    n = len(s.master.mav.rc)
    for _ in range(s.PWM_LATE_FRAMES):
        s.set_pwm(PwmCommand(pulses=PW, source="mpc", stamp=now() - 1.0))
    after = s.master.mav.rc[n:]
    assert after and all(_motors(f) == NEUTRAL_PULSES for f in after)
    assert not s._pwm_active and "late" in s._pwm_latch
    assert sum(1 for lvl, m in log.lines if lvl == "error" and "arrive older" in m) == 2, \
        "said once per run of late frames (the single one, then the three), not once per frame"


def test_sink_refuses_on_a_v1_link_a_companions_messages_and_a_bad_parameter():
    s, _ = _sink(v2=False)
    _answer_all(s)
    s.enabled = True
    s.set_pwm(PwmCommand(pulses=PW))
    assert s.master.mav.rc == [] and not s._pwm_active
    # the heartbeat must be the VEHICLE's: a companion's RLPWM does not arm the gate
    s, _ = _sink()
    _answer_all(s)
    s._pwm_gate.t_hb = None
    s.master.queue = [_Msg("NAMED_VALUE_FLOAT", comp=100, name="RLPWM", value=1.0)]
    s._drain()
    s.enabled = True
    s.set_pwm(PwmCommand(pulses=PW))
    assert s.master.mav.rc == []
    # ...and so must the parameters: a companion cannot write a motor direction into the gate
    s, _ = _sink()
    s.master.queue = [_Msg("PARAM_VALUE", comp=100, param_id=k, param_value=float(v))
                      for k, v in _good_values().items()]
    for _ in range(4):
        s._drain()
    assert len(s._pwm_gate.missing()) == len(s._pwm_gate.param_names())
    # one reversed motor direction on the vehicle: refused, with the parameter named
    s, log = _sink()
    _answer_all(s, MOT_2_DIRECTION=1)
    s.enabled = True
    s.set_pwm(PwmCommand(pulses=PW))
    assert s.master.mav.rc == [] and any("spin reversed" in m for _, m in log.lines)


def test_sink_ends_a_stalled_flow_with_neutral_frames_and_then_writes_nothing_more():
    s, log = _sink()
    _answer_all(s)
    s.enabled = True
    s.set_pwm(PwmCommand(pulses=PW))
    assert s._pwm_active and len(s.master.mav.rc) == 1
    t = time.monotonic()
    s._pwm_tick(t)                                   # fresh: nothing but the snapshot
    assert s._pwm_active and len(s.master.mav.rc) == 1
    assert s.pwm_gate_state()["ok"] and s.pwm_gate_state()["active"] is True
    # the follower stops emitting: after PWM_STALE_S the sink neutralises (a burst, then one per tick), then stops
    for k in range(8):
        s._pwm_tick(t + s.PWM_STALE_S + 0.05 * (k + 1))
    assert not s._pwm_active and not s._pwm_latch, "a quiet follower is not a gate failure"
    neutral = s.master.mav.rc[1:]
    assert len(neutral) == 2 * s.PWM_NEUTRAL_FRAMES and all(_motors(f) == NEUTRAL_PULSES for f in neutral)
    assert any("pulses stopped" in m for _, m in log.lines)
    assert s.pwm_gate_state()["active"] is False
    # the flow is over: this is again a sink that writes nothing on RC 9..16 — E-STOP in a later PID run included
    assert s._pwm_ever_sent is False
    n = len(s.master.mav.rc)
    s.estop()
    assert len(s.master.mav.rc) == n


def test_sink_latches_a_gate_lost_mid_flow_until_the_follower_stops():
    """The engage-grace failure used to clear itself: the sink stopped, the gate read ok again 50 ms later, the flow
    resumed, for ever, and the worker was never told (safety review 2026-09-30)."""
    s, log = _sink()
    _answer_all(s, hb=0.0)                           # the script is alive but idle
    s.enabled = True
    s.set_pwm(PwmCommand(pulses=PW))
    assert s._pwm_active
    t0 = time.monotonic()
    for k in range(1, 26):                           # 1.25 s of frames, and the script still says it is NOT forcing
        s._pwm_gate.note_pulses(t0 + 0.05 * k)
    s._pwm_gate.note_heartbeat(0.0, t0 + 1.25)
    s._pwm_last_rx = t0 + 1.25
    s._pwm_tick(t0 + 1.25)
    assert not s._pwm_active and "not forcing the outputs" in s._pwm_latch
    st = s.pwm_gate_state()
    assert st["ok"] is False and st["latched"] is True and "not forcing" in st["why"], \
        "the worker must SEE the failure in the snapshot"
    n = len(s.master.mav.rc)
    for _ in range(3):
        s.set_pwm(PwmCommand(pulses=PW))             # the follower keeps asking: refused
    assert all(_motors(f) == NEUTRAL_PULSES for f in s.master.mav.rc[n:]) and not s._pwm_active
    # an ENGAGED follower's own neutral (it waits out its grace window) does NOT clear it...
    s.set_pwm(PwmCommand(source="mpc"))
    assert s._pwm_latch and s._pwm_gate.unconfirmed_s > 1.0
    s._pwm_tick(time.monotonic())
    assert s.pwm_gate_state()["ok"] is False, "so the worker keeps seeing the failure until it disengages"
    # ...its disengage ("stop") does, and the engagement is over for the gate too; the next one starts clean
    s.set_pwm(PwmCommand(source="stop"))
    assert s._pwm_latch == "" and s._pwm_gate.unconfirmed_s == 0.0
    s._pwm_gate.note_heartbeat(1.0, time.monotonic())
    s.set_pwm(PwmCommand(pulses=PW))
    assert s._pwm_active and _motors(s.master.mav.rc[-1]) == PW


def test_sink_estop_disable_and_teardown_neutralise_the_pulses_first():
    for how in ("estop", "disable", "teardown", "neutral_command"):
        s, log = _sink()
        _answer_all(s)
        s.enabled = True
        s.set_pwm(PwmCommand(pulses=PW))
        n0 = len(s.master.mav.rc)
        master = s.master
        if how == "estop":
            s.estop()
        elif how == "disable":
            s.set_enabled(False)
        elif how == "teardown":
            s.teardown()
        else:
            s.set_pwm(PwmCommand(source="stop"))     # the worker's explicit stop: 1500 x 8
        after = master.mav.rc[n0:]
        assert len(after) == s.PWM_NEUTRAL_FRAMES and all(_motors(f) == NEUTRAL_PULSES for f in after), how
        assert not s._pwm_active, how
        if how in ("estop", "disable", "teardown"):
            assert master.mav.manual and master.mav.manual[-1][1:5] == (0, 0, 500, 0), how
        if how != "teardown":
            # ...and again on the next ticks (a short rx queue on the vehicle may drop a burst), even while disabled
            t = time.monotonic()
            for k in range(5):
                s._pwm_tick(t + 0.05 * k)
            tail = master.mav.rc[n0 + s.PWM_NEUTRAL_FRAMES:]
            assert len(tail) == s.PWM_NEUTRAL_FRAMES and all(_motors(f) == NEUTRAL_PULSES for f in tail), how


def test_a_session_that_never_used_rl_pwm_never_sends_an_rc_override():
    """RC 9..16 are lights / camera / gripper inputs on a stock vehicle. E-STOP, disable, teardown and the sink's own
    tick must not write them unless this session put policy pulses there first."""
    s, _ = _sink()
    _answer_all(s)
    s.enabled = True
    t = time.monotonic()
    s.set_pilot(PilotInput(surge=0.2, source="gamepad"))
    s.tick()
    s._pwm_tick(t + 5.0)
    s.set_pwm(PwmCommand(source="idle"))             # the worker's keepalive / an explicit neutral: nothing to do
    master = s.master
    s.estop()
    s.set_enabled(False)
    s.teardown()
    assert master.mav.rc == []
    assert master.mav.manual, "the axes path is untouched"


def test_sink_reads_the_vehicle_parameters_only_while_somebody_wants_the_transport():
    s, _ = _sink()
    t = time.monotonic()
    s._pwm_last_rx = t - 100.0
    s._pwm_tick(t)
    assert s.master.mav.params == [], "nobody selected RL_PWM: no parameter traffic at all"
    s.set_pwm(PwmCommand(source="idle"))             # the worker's "RL_PWM is selected" keepalive
    t = time.monotonic()
    s._pwm_tick(t)
    assert len(s.master.mav.params) == 8, "one small batch per tick"
    s._pwm_tick(t + 0.05)
    assert len(s.master.mav.params) == 16
    names = {a[2].decode() for a in s.master.mav.params}
    assert names <= set(s._pwm_gate.param_names())
    # once everything is answered the sink goes quiet, then refreshes after PARAM_REFRESH_S while still wanted
    _answer_all(s)
    s.master.mav.params.clear()
    t = time.monotonic()
    s._pwm_tick(t + 0.1)
    assert s.master.mav.params == []
    s._pwm_last_rx = t + 11.0
    s._pwm_tick(t + 11.0)
    assert len(s.master.mav.params) == 8


def test_sink_tick_survives_a_failure_in_the_rl_pwm_bookkeeping_and_latches():
    """teleop, PID and MPC sessions run this tick too: nothing in the rl_pwm lines may cost them the MANUAL_CONTROL,
    the deadman or the GCS heartbeat. And with that bookkeeping broken nobody watches a flow, so it is latched shut."""
    s, log = _sink()
    _answer_all(s)
    s.enabled = True
    s.set_pwm(PwmCommand(pulses=PW))
    assert s._pwm_active

    def boom(*a, **k):
        raise RuntimeError("synthetic")

    s._pwm_gate.state = boom
    s.set_pilot(PilotInput(surge=0.3, source="gamepad"))
    n = len(s.master.mav.rc)
    s.tick()
    assert s.master.mav.manual and s.master.mav.manual[-1][1] == 300, "the pilot's axis still went out"
    st = s.pwm_gate_state()
    assert st["ok"] is False and "tick failed" in st["why"] and st["latched"] is True
    assert not s._pwm_active and "tick failed" in s._pwm_latch
    assert all(_motors(f) == NEUTRAL_PULSES for f in s.master.mav.rc[n:]) and len(s.master.mav.rc) > n
    assert sum(1 for lvl, m in log.lines if lvl == "error" and "sink tick failed" in m) == 1
    s.tick()
    assert sum(1 for lvl, m in log.lines if lvl == "error" and "sink tick failed" in m) == 1, "said once"
    n = len(s.master.mav.rc)
    s.set_pwm(PwmCommand(pulses=PW))
    assert all(_motors(f) == NEUTRAL_PULSES for f in s.master.mav.rc[n:]), "no policy frame leaves while latched"


# =============================================================================
# 4. the worker
# =============================================================================
def _gate(ok=True, why="ok", age=0.0, mot=None, cap=0.25):
    return {"configured": True, "ok": ok, "why": why, "stamp": now() - age, "active": False,
            "motor_direction": list(MOT if mot is None else mot), "pwm_cap": cap}


def _pwm_worker(tmp):
    from rov_gui.tests.test_control import _worker

    w, bus, pilots, logs = _worker(tmp)
    pwms = []
    bus.cmd_pwm.connect(pwms.append)
    return w, bus, pilots, pwms, logs


def _feed(w, mode="MANUAL", gate=None):
    from rov_gui.tests.test_control import _fix, _imu

    t = now()
    w.on_nav_fix(_fix([-2.0, 0.0, 0.8], 0.0, t))
    w.on_vehicle_imu(_imu(depth=0.8, t=t))
    w.on_telemetry(Telemetry(armed=True, mode=mode, conn=Conn.ONLINE, pwm_gate=gate))


def _live(pwms):
    return [p for p in pwms if p.pulses != NEUTRAL_PULSES]


def test_worker_has_the_mode_and_refuses_it_without_a_gate_or_outside_manual():
    from rov_gui.control.workers import MpcWorker
    from rov_gui.widgets.trajectory import MODE_LABELS

    assert "rl_pwm" in MpcWorker.MODES and MODE_LABELS["rl_pwm"] == "RL_PWM"
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, pwms, logs = _pwm_worker(tmp)
        w.set_mode("rl_pwm")
        assert w.cfg.mode == "rl_pwm" and getattr(w.ctrl, "transport", "") == "pwm"
        w.on_enable(True)
        # a backend whose sink has no such transport (the demo): Telemetry carries no gate
        _feed(w, gate=None)
        w.set_engaged(True)
        assert not w.engaged and "no per-thruster transport" in w.reason
        # the gate says no, and its sentence is the reason
        _feed(w, gate=_gate(False, "SCR_USER1 != 1 (rl_pwm_override.lua is not armed)"))
        w.set_engaged(True)
        assert not w.engaged and "SCR_USER1" in w.reason
        # a silent sink is not consent
        _feed(w, gate=_gate(True, age=5.0))
        w.set_engaged(True)
        assert not w.engaged and "stale" in w.reason
        # the sink's gate was built for another export's motor directions
        _feed(w, gate=_gate(True, mot=[1] * 8))
        w.set_engaged(True)
        assert not w.engaged and "motor directions" in w.reason
        # ...or clamps at another cap than this follower believes
        _feed(w, gate=_gate(True, cap=0.5))
        w.set_engaged(True)
        assert not w.engaged and "caps |throttle|" in w.reason
        # STABILIZE is admitted by engage.require_mode for the axes followers — never for raw pulses
        _feed(w, mode="STABILIZE", gate=_gate(True))
        w.set_engaged(True)
        assert not w.engaged and "needs MANUAL" in w.reason
        assert _live(pwms) == [], "a refused engage emits no pulse"
        # ...and the axes followers never look at the gate, and never emit on cmd_pwm at all
        w.set_mode("pid")
        pwms.clear()
        _feed(w, gate=None)
        w.tick()
        w.set_engaged(True)
        assert w.engaged
        _feed(w, gate=None)
        w.tick()
        w.disengage("test over")
        assert pwms == [], "an axes follower emits nothing on cmd_pwm — not even a stop or a keepalive"


def test_worker_tells_the_sink_rl_pwm_is_selected_about_once_a_second():
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, pwms, logs = _pwm_worker(tmp)
        w.set_mode("rl_pwm")
        for _ in range(5):
            _feed(w, gate=_gate(True))
            w.tick()
        assert len(pwms) == 1 and pwms[0].pulses == NEUTRAL_PULSES and pwms[0].source == "idle"
        w._pwm_keepalive_t -= 2.0
        _feed(w, gate=_gate(True))
        w.tick()
        assert len(pwms) == 2


def test_worker_emits_pulses_and_a_neutral_manual_control_while_engaged_and_stops_with_1500s():
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, pwms, logs = _pwm_worker(tmp)
        w.set_mode("rl_pwm")
        w.on_enable(True)
        _feed(w, gate=_gate(True))
        w.set_engaged(True)
        assert w.engaged, w.reason
        pilots.clear()
        pwms.clear()
        for _ in range(4):
            _feed(w, gate=_gate(True))
            w.tick()
        live = _live(pwms)
        assert live, "an engaged rl_pwm follower emits pulses"
        cap = w.ctrl.pwm_cap
        for p in live:
            assert isinstance(p, PwmCommand) and len(p.pulses) == 8 and p.source == "mpc"
            assert all(1500 - 400 * cap <= v <= 1500 + 400 * cap for v in p.pulses), p.pulses
        assert pilots and all((a.surge, a.sway, a.heave, a.yaw, a.roll, a.pitch) == (0.0,) * 6 for a in pilots), \
            "MANUAL_CONTROL goes out NEUTRAL beside the pulses"
        assert w.ctrl.last_pwm_us == live[-1].pulses, "the applied pulses are fed back to the policy"
        path = w._csv_path
        w.disengage("test over")
        assert pwms[-1].pulses == NEUTRAL_PULSES and pwms[-1].source == "stop", \
            "disengage ends with eight explicit 1500s, marked as the stop"
        rows = [r.split(",") for r in Path(path).read_text().splitlines()]
        idx = {n: i for i, n in enumerate(rows[0])}
        last = rows[-1]
        assert all(float(last[idx[k]]) == 0.0 for k in ("ax_surge", "ax_sway", "ax_heave", "ax_yaw")), \
            "CSV ax_* are the axes that left: zero"
        assert last[idx["mode"]] == "rl_pwm"


def test_worker_sends_neutral_during_the_gate_grace_and_then_drops_the_engagement():
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, pwms, logs = _pwm_worker(tmp)
        w.set_mode("rl_pwm")
        w.on_enable(True)
        _feed(w, gate=_gate(True))
        w.set_engaged(True)
        assert w.engaged, w.reason
        _feed(w, gate=_gate(True))
        w.tick()
        assert _live(pwms)
        bad = "RLPWM heartbeat stale (2.4 s) — the Lua script stopped"
        pwms.clear()
        _feed(w, gate=_gate(False, bad))
        w.tick()                                     # inside the grace window: still engaged, but asks for NOTHING
        assert w.engaged and pwms and pwms[-1].pulses == NEUTRAL_PULSES and not _live(pwms)
        assert pwms[-1].source == "mpc", "an engaged follower's neutral is not a stop (it must not clear the latch)"
        assert w.ctrl.last_pwm_us == NEUTRAL_PULSES, "the policy is not told a refused frame was applied"
        w._pwm_gate_bad_since -= w.PWM_GATE_GRACE_S + 0.1
        _feed(w, gate=_gate(False, bad))
        w.tick()
        assert not w.engaged and "rl_pwm gate" in w.reason and "heartbeat stale" in w.reason
        assert pwms[-1].pulses == NEUTRAL_PULSES and not _live(pwms)
        n = len(_live(pwms))
        _feed(w, gate=_gate(False, "x"))
        w.tick()
        assert len(_live(pwms)) == n, "no pulse is emitted once disengaged"


# =============================================================================
# runner
# =============================================================================
def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except pytest.skip.Exception as e:
            print(f"  skip  {name}: {e}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:                                   # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
