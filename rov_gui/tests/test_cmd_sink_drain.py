"""MavlinkCommandSink._drain under a fast vehicle stream (2026-10-13).

On 2026-10-13 21:00-21:01 every RL_PWM engagement was disengaged with "rl_pwm_override.lua is not forcing the outputs
(RLPWM = 0) after 1.0 s of pulses" while the vehicle's SERVO_OUTPUT_RAW followed the policy pulses
[측정 data/20261013/1013_205916/mpc_2100*.csv pwm1..8], and the "button N = ..." / "SYSID_MYGCS = ..." lines kept
repeating. The sink read at most 40 messages per 20 Hz tick while the vehicle streamed ~480-565 msg/s
[측정 data/20261013/*/c3_fs_depth_*_rov.json hz_measured], so the 2 Hz RLPWM heartbeat and the parameter replies sat
behind a backlog. These tests pin the fix: drain until empty (time-bounded), and print each button / SYSID once.
"""
import time

from rov_gui.tests.test_rl_pwm import PW, _answer_all, _Msg, _sink
from rov_gui.backends.hardware import MavlinkCommandSink
from rov_gui.state import PwmCommand


def _flood(n, start_t=0):
    """n vehicle messages the sink does not care about (what 140 Hz ATTITUDE/RAW_IMU/SCALED_IMU2 look like to it)."""
    return [_Msg("ATTITUDE", roll=0.0, pitch=0.0) for _ in range(n)]


def test_one_tick_reads_far_more_than_40_messages_and_finds_the_heartbeat_behind_them():
    s, _ = _sink()
    _answer_all(s, hb=0.0)
    # a full second of today's traffic in front of the heartbeat: the old cap (40/tick) needed 15 ticks to reach it
    s.master.queue = _flood(600) + [_Msg("NAMED_VALUE_FLOAT", name="RLPWM", value=1.0)]
    s._drain()
    assert s.master.queue == [], "one tick drains the whole backlog"
    assert s._pwm_gate.script_engaged is True


def test_the_gate_confirms_the_script_within_its_grace_under_a_continuous_stream():
    """Pulses flow, the vehicle streams ~600 msg/s, the script answers RLPWM = 1 once: the gate must not trip."""
    s, log = _sink()
    _answer_all(s, hb=0.0)
    s.enabled = True
    t0 = time.monotonic()
    for k in range(30):                                  # 1.5 s of 20 Hz ticks
        s.master.queue.extend(_flood(30))                # ~600 msg/s
        if k == 8:                                       # the script's first "1" after 0.4 s of pulses
            s.master.queue.append(_Msg("NAMED_VALUE_FLOAT", name="RLPWM", value=1.0))
        s._drain()
        s.set_pwm(PwmCommand(pulses=PW, source="mpc"))
    assert time.monotonic() - t0 < 5.0
    assert not any("not forcing the outputs" in m for _, m in log.lines), [m for _, m in log.lines if lvl_err(m)]
    assert s._pwm_gate.script_engaged is True


def lvl_err(m):
    return "error" in m.lower()


def test_drain_is_bounded_by_time_and_warns_once_a_minute_about_a_backlog(monkeypatch):
    s, log = _sink()
    monkeypatch.setattr(MavlinkCommandSink, "DRAIN_MAX_MSGS", 100)
    s.master.queue = _flood(1000)
    s._drain()
    assert len(s.master.queue) == 900                    # stopped at the cap, the rest stays queued
    s._drain()
    warns = [m for lvl, m in log.lines if lvl == "warn" and "more are waiting" in m]
    assert len(warns) == 1 and s.drain_backlog_ticks == 2, "one warning per minute, every backlog tick counted"


def test_button_and_sysid_lines_are_printed_once_per_value():
    s, log = _sink()
    s.bit_grip_open = btn = 0                            # the station's default --btn-grip-open (the test sink has no opts for it)
    for _ in range(5):                                   # the gate's 10 s re-reads / other clients' fetches
        s.master.queue = [_Msg("PARAM_VALUE", param_id=f"BTN{btn}_FUNCTION", param_value=float(_want("gripper_open"))),
                          _Msg("PARAM_VALUE", param_id="SYSID_MYGCS", param_value=255.0)]
        s._drain()
    ok_lines = [m for _, m in log.lines if m.startswith(f"button {btn} =")]
    sys_lines = [m for _, m in log.lines if m.startswith("SYSID_MYGCS = 255")]
    assert len(ok_lines) == 1 and len(sys_lines) == 1, (ok_lines, sys_lines)
    # a CHANGED value is reported again (here: the button reassigned on the vehicle -> error, button switched off)
    s.master.queue = [_Msg("PARAM_VALUE", param_id=f"BTN{btn}_FUNCTION", param_value=float(_want("gripper_open") + 1))]
    s._drain()
    assert any(lvl == "error" and f"button {btn} is BTN{btn}_FUNCTION" in m for lvl, m in log.lines)
    assert s.bit_grip_open is None


def _want(key):
    from rov_gui.backends.hardware import BTN_FUNCTION
    return BTN_FUNCTION[key][0]
