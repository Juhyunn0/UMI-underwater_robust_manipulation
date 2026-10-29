#!/usr/bin/env python3
"""
test_disturbance_panel.py — the disturbance rig panel against whisker_vision's
simulated devices, headless. No hardware, no display, no network.

    QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/robust/bin/python src/tests/test_disturbance_panel.py
    QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/whisker/bin/python -m pytest src/tests/test_disturbance_panel.py -v

Skips when whisker_vision is not importable (./c3 test must pass on any machine).
Runtime budget: each window joins three sim link threads on close (fast in sim;
the 3 s in link.stop() is a timeout, not a sleep).
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import disturbance_core as core  # noqa: E402
import disturbance_panel as P  # noqa: E402

SKIP_REASON = P.WHISKER_IMPORT_ERROR
try:
    import pytest
    if SKIP_REASON:
        pytest.skip(SKIP_REASON, allow_module_level=True)
except ImportError:
    pytest = None

if not SKIP_REASON:
    from rov_gui.qt import Qt, QtWidgets  # noqa: E402
    from motion.fmc4030 import Axis  # noqa: E402
    from motion_control import GantryLink  # noqa: E402
    from motion_simulator import SimGantryController, SimThrusterSerial  # noqa: E402

_APP = None


def _app():
    global _APP
    if _APP is None:
        _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return _APP


def _pump(predicate, timeout=5.0):
    app = _app()
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return False


class _Window:
    """Context manager: a sim panel in a temp data root, closed on exit."""

    def __init__(self, *argv, builders=None):
        self.argv = list(argv)
        self.builders = builders

    def __enter__(self):
        _app()
        self.td = tempfile.TemporaryDirectory()
        cfg = core.load_rig_config()
        args = P.build_arg_parser(cfg).parse_args(
            ["--sim", "--no-preflight", "--ramp-interval", "0.05",
             "--data-root", self.td.name] + self.argv)
        self.settings = core.settings_from_args(args, cfg)
        self.options = core.options_from_args(args, cfg)
        self.w = P.DisturbanceWindow(self.settings, self.options, P.WHISKER_ROOT,
                                     preflight={"verdict": "ready"},
                                     builders=self.builders)
        return self

    def __exit__(self, *exc):
        self.w.close()
        self.td.cleanup()
        return False

    def connect_all(self):
        w = self.w
        w._toggle_gantry()
        w._toggle_motor()
        w._toggle_thruster()
        assert _pump(lambda: w._gantry.is_ready and w._motor.is_ready
                     and w._thruster.is_ready)
        # the motor zeroes itself once the thruster is up (~2 s of dwells)
        assert _pump(lambda: w._motor_auto is None, 10), w.status_label.text()
        _still(w)
        return w

    def enable_motor(self):
        """Connect zeroes and enables the motor by itself: wait for it."""
        w = self.w
        assert _pump(lambda: w._motor_auto is None and w._motor.is_enabled, 10), \
            w.status_label.text()

    def disable_motor(self):
        w = self.w
        self.enable_motor()
        w._toggle_enable()
        assert _pump(lambda: not w._motor.is_enabled)

    def run_dir(self):
        metas = list(Path(self.td.name).rglob("meta.json"))
        assert metas, "no run folder"
        return metas[0].parent


# ----------------------------------------------------------------- tests
def test_connect_readout_limits_and_autolog():
    with _Window() as s:
        w = s.connect_all()
        assert _pump(lambda: w._log is not None), "autolog did not start"
        assert w._log.dir.name.endswith("_disturb")
        lo = [w._target_spins[a].minimum() for a in (Axis.X, Axis.Y, Axis.Z)]
        hi = [w._target_spins[a].maximum() for a in (Axis.X, Axis.Y, Axis.Z)]
        assert lo == [0.0, 0.0, 0.0] and hi == [495.0, 320.0, 150.0]   # sim's valid fw limits
        assert "X 0.0..495.0" in w.limits_label.text()
        assert w._limit_labels[(Axis.X, "hi")].text().endswith("(fw)")
        assert _pump(lambda: w._live_labels[Axis.X].text() == "0.0 mm")
        assert w.thruster_live.text().startswith("+0 %")
        assert "µs" not in w.thruster_live.text()      # no pwm before the first command
        assert not w.send_btn.isEnabled()               # nothing edited yet
        assert w._jog_buttons[(Axis.X, 1)].isEnabled()
        assert _pump(lambda: w._teach_buttons[(Axis.X, "lo")].isEnabled())
        assert not w.restore_btn.isEnabled()            # nothing saved
        assert not w._clear_buttons[Axis.X].isEnabled()  # nothing taught


def _still(w, timeout=5.0):
    """Physically at rest AND the buttons refreshed (a press on a button the
    100 ms tick has not re-enabled yet is silently dropped by Qt)."""
    assert _pump(lambda: w._limits_applied and w._all_at_rest(), timeout)
    w._update_buttons()


def _settle(w, timeout=10):
    assert _pump(lambda: not w._gantry.busy and not w._gantry_cmd_in_flight, timeout)
    _still(w)


def _send(w, **targets):
    for name, value in targets.items():
        w._target_spins[getattr(Axis, name)].setValue(value)
    assert _pump(lambda: w.send_btn.isEnabled())
    w._send_targets()
    _settle(w)


def test_send_and_link_refusal():
    with _Window() as s:
        w = s.connect_all()
        _send(w, X=10.0)
        assert abs(w._gantry.positions_mm()[0] - 10.0) < 0.2
        sim = w._gantry._controller
        assert not w.send_btn.isEnabled()               # nothing edited since it settled
        _send(w, Y=7.5)
        pos = w._gantry.positions_mm()
        assert abs(pos[0] - 10.0) < 0.2 and abs(pos[1] - 7.5) < 0.2
        # a target the link refuses never reaches the controller
        n_jogs = sum(1 for e in sim.call_log if e[0] == "jog")
        try:
            w._gantry.request_move({Axis.X: 1000.0}, 20.0)
        except ValueError:
            pass
        assert sum(1 for e in sim.call_log if e[0] == "jog") == n_jogs


class _FactoryLimits(SimGantryController):
    """The real controller: min==max==200 units -> no limits + firmware stops."""

    def get_device_parameters(self):
        from motion_simulator import _SimDeviceParams
        return _SimDeviceParams(soft_limit_min=(200, 200, 200),
                                soft_limit_max=(200, 200, 200))


def _put(w, axis, mm):
    """Teleport the sim axis (a stand-in for jogging it there). A teleport is
    not a command, so it would trip the post-release drift watch: end it."""
    w._jog_watch_until = 0.0
    sim = w._gantry._controller
    units = mm / {Axis.X: 8.25, Axis.Y: 2.5, Axis.Z: 0.5}[axis]
    sim._pos[int(axis)] = sim._target[int(axis)] = units
    assert _pump(lambda: abs(w._gantry.positions_mm()[int(axis)] - mm) < 0.01, 2)
    _still(w)


def _released(w, timeout=4.0):
    ok = _pump(lambda: w._jog_release is None, timeout)
    if ok and not w._latched:
        _still(w)
    return ok


def test_teach_homes_become_the_soft_limits():
    """The user's flow on the real controller (no valid firmware limits): jog
    anywhere — also below the power-up 0 — press −X home there, jog, press
    +X home there; those two are X's soft limits from then on."""
    with _Window("--speed", "40", builders=_rig(_FactoryLimits)) as s:
        w = s.connect_all()
        w.show()
        assert w._gantry.soft_limits_mm() == ([None] * 3, [None] * 3)   # nothing yet
        assert w._limit_labels[(Axis.X, "lo")].text().startswith("--")
        assert not w._target_spins[Axis.X].isEnabled()                  # Send needs both
        btn = w._jog_buttons[(Axis.X, -1)]
        _press(btn)
        assert w._jog is not None and w._jog["end"] is None             # no − home: free
        assert _pump(lambda: w._gantry.positions_mm()[0] < -12.0, 4)    # below the old 0
        _release(btn)
        assert _released(w) and not w._latched
        _settle(w)
        lo_at = w._gantry.positions_mm()[0]
        w._teach(Axis.X, "lo")
        assert w._taught["X"]["lo"] == lo_at
        assert w._limit_labels[(Axis.X, "lo")].text() == f"{lo_at:.1f} mm"
        assert w._gantry.soft_limits_mm()[0][0] is None          # one side: panel-enforced
        _press(btn)                                              # at the − home now
        assert w._jog is None and "limit" in w.status_label.text()
        _release(btn)
        _put(w, Axis.X, lo_at + 50.0)
        w._teach(Axis.X, "hi")
        lo, hi = w._gantry.soft_limits_mm()
        assert (lo[0], hi[0]) == (lo_at, lo_at + 50.0)           # installed in the link
        spin = w._target_spins[Axis.X]           # 0.1 mm steps, rounded INSIDE the homes
        assert _pump(lambda: lo_at <= spin.minimum() < lo_at + 0.1
                     and lo_at + 49.9 < spin.maximum() <= lo_at + 50.0)
        saved = json.loads(w._options.limits_file.read_text())
        assert saved["axes"]["X"]["lo"] == lo_at and saved["axes"]["X"]["hi"] == lo_at + 50.0
        # toward a home the jog is ONE move that ends at the home
        _put(w, Axis.X, lo_at + 20.0)
        sim = w._gantry._controller
        n = len([e for e in sim.call_log if e[0] == "jog"])
        btn_p = w._jog_buttons[(Axis.X, 1)]
        _press(btn_p)
        assert w._jog is not None, w.status_label.text()
        assert w._jog["end"] == lo_at + 50.0
        assert _pump(lambda: w._jog is None, 4)                  # stopped AT the home
        _release(btn_p)
        assert _released(w)
        jogs = [e for e in sim.call_log if e[0] == "jog"][n:]
        assert len(jogs) == 1 and abs(jogs[0][2] * 8.25 - (30.0 - P.LIMIT_EDGE_MM)) < 0.05
        assert abs(w._gantry.positions_mm()[0] - (lo_at + 50.0)) < 0.2
        # a + home below the − home is refused
        _put(w, Axis.X, lo_at - 10.0)
        w._teach(Axis.X, "hi")
        assert "must be above" in w.status_label.text()
        assert w._taught["X"]["hi"] == lo_at + 50.0
        # Clear: back to no limit on X, file follows
        _put(w, Axis.X, lo_at + 10.0)
        w._clear_limits(Axis.X)
        assert "X" not in w._taught and w._gantry.soft_limits_mm()[0][0] is None
        assert "X" not in json.loads(w._options.limits_file.read_text())["axes"]
        kinds = [line.split(",")[2] for line in
                 (w._log.dir / "events.csv").read_text().splitlines()[1:]]
        assert kinds.count("teach") == 2 and "teach_clear" in kinds


def test_taught_limits_tighten_valid_firmware_limits():
    with _Window() as s:
        w = s.connect_all()
        _send(w, X=100.0)
        w._teach(Axis.X, "hi")
        assert w._effective_limits(Axis.X) == (0.0, 100.0)       # fw lo, taught hi
        assert w._target_spins[Axis.X].maximum() == 100.0
        sim = w._gantry._controller
        n = sum(1 for e in sim.call_log if e[0] == "jog")
        try:
            w._gantry.request_move({Axis.X: 200.0}, 20.0)          # the link has it too
        except ValueError:
            pass
        assert sum(1 for e in sim.call_log if e[0] == "jog") == n
        w.show()
        btn = w._jog_buttons[(Axis.X, 1)]
        _press(btn)
        assert w._jog is None and "limit" in w.status_label.text()
        _release(btn)


def test_firmware_stop_caps_the_jog_and_send():
    with _Window(builders=_rig(_FactoryLimits)) as s:
        w = s.connect_all()
        assert "Y 500" in w.limits_label.text()                  # firmware stop shown
        assert w._gantry.firmware_stops_mm()["Y"] == 500.0
        w._teach(Axis.Y, "lo")                                   # Y 0
        _put(w, Axis.Y, 600.0)
        w._teach(Axis.Y, "hi")                                   # beyond the fw stop
        _put(w, Axis.Y, 0.0)
        assert w._jog_end_mm(Axis.Y, +1) == 500.0
        sim = w._gantry._controller
        n_jogs = sum(1 for e in sim.call_log if e[0] == "jog")
        w._target_spins[Axis.Y].setValue(550.0)                  # inside the homes, past the stop
        assert _pump(lambda: w.send_btn.isEnabled())
        w._send_targets()
        _app().processEvents()
        assert "firmware stop" in w.status_label.text()
        assert sum(1 for e in sim.call_log if e[0] == "jog") == n_jogs


class _Persistent(_FactoryLimits):
    """A controller whose counter survives a reconnect (TCP only, no power
    cycle): every instance shares one position list. Reset ``shared`` first."""
    shared = None

    def __init__(self):
        super().__init__()
        if _Persistent.shared is None:
            _Persistent.shared = self._pos
        else:
            self._pos = _Persistent.shared
            self._target = list(self._pos)


def test_restore_saved_homes_after_a_restart():
    _Persistent.shared = None
    with tempfile.TemporaryDirectory() as td:
        lim = str(Path(td) / "limits.json")
        with _Window("--limits-file", lim, builders=_rig(_Persistent)) as s:
            w = s.connect_all()
            w._teach(Axis.Z, "lo")
            _put(w, Axis.Z, 40.0)
            w._teach(Axis.Z, "hi")
            _put(w, Axis.Z, 20.0)                                 # close saves this spot
        assert json.loads(Path(lim).read_text())["last_pos_mm"][2] == 20.0
        with _Window("--limits-file", lim, builders=_rig(_Persistent)) as s:
            w = s.connect_all()
            assert not w._taught                                  # never applied silently
            assert "Z 0.0..40.0" in w.saved_label.text()
            assert _pump(lambda: w.restore_btn.isEnabled())
            w._restore_saved()
            assert w._taught["Z"]["hi"] == 40.0
            lo, hi = w._gantry.soft_limits_mm()
            assert (lo[2], hi[2]) == (0.0, 40.0)
            assert "restored" in w.status_label.text()
            assert _pump(lambda: not w.restore_btn.isEnabled())
            assert "teach_restore" in _kinds(w)


def test_restore_refused_when_the_gantry_is_not_where_it_was_saved():
    """A power cycle restarts the counter at 0: the saved homes would be
    shifted by the old position. Restore must refuse, installing nothing."""
    _Persistent.shared = None
    with tempfile.TemporaryDirectory() as td:
        lim = str(Path(td) / "limits.json")
        with _Window("--limits-file", lim, builders=_rig(_Persistent)) as s:
            w = s.connect_all()
            _put(w, Axis.X, -300.0)
            w._teach(Axis.X, "lo")
            _put(w, Axis.X, 200.0)
            w._teach(Axis.X, "hi")
            _put(w, Axis.X, -100.0)
        _Persistent.shared = None                                 # "power cycle": counter 0
        with _Window("--limits-file", lim, builders=_rig(_Persistent)) as s:
            w = s.connect_all()
            assert w.restore_btn.isEnabled() or _pump(lambda: w.restore_btn.isEnabled())
            w._restore_saved()
            assert not w._taught and w._gantry.soft_limits_mm()[0][0] is None
            assert "NOT restored" in w.status_label.text()
            assert "teach_restore_refused" in _kinds(w)


def test_homes_survive_a_reconnect_only_where_they_were():
    _Persistent.shared = None
    with _Window(builders=_rig(_Persistent)) as s:
        w = s.connect_all()
        w._teach(Axis.X, "lo")
        _put(w, Axis.X, 30.0)
        w._teach(Axis.X, "hi")
        w._toggle_gantry()                                       # disconnect (TCP only)
        assert w._gantry is None
        w._toggle_gantry()                                       # connect again
        assert _pump(lambda: w._gantry.is_ready and w._limits_applied)
        lo, hi = w._gantry.soft_limits_mm()
        assert (lo[0], hi[0]) == (0.0, 30.0)                     # re-installed
        # now a power cycle while disconnected: the counter restarts at 0
        w._toggle_gantry()
        _Persistent.shared = None
        w._toggle_gantry()
        assert _pump(lambda: w._gantry.is_ready and w._limits_applied)
        assert not w._taught and w._gantry.soft_limits_mm()[0][0] is None
        assert "NOT re-applied" in w.status_label.text()
        assert "homes_dropped" in _kinds(w)


def test_motor_goto_and_interlock():
    with _Window() as s:
        w = s.connect_all()
        s.disable_motor()
        w.thruster_spin.setValue(30)
        w._set_thruster()                             # interlock: motor not holding
        _app().processEvents()
        assert "motor enabled and holding" in w.status_label.text()
        assert w._thruster.ramp_target() is None and w._thruster.throttle_pct() == 0
        assert _pump(lambda: "press Enable" in w.thruster_info.text()
                     and w.thruster_info.isVisibleTo(w))
        w._toggle_enable()
        assert _pump(lambda: w._motor.is_enabled)
        w.angle_spin.setValue(60.0)
        w._go_theta()
        assert _pump(lambda: not w._motor.busy and abs(w._motor.angle_deg() - 60.0) < 1.0, 10)
        w._set_thruster()
        assert _pump(lambda: w._thruster.throttle_pct() == 30)
        assert _pump(lambda: "1620 µs" in w.thruster_live.text())   # label follows the tick
        assert not w.thruster_info.isVisibleTo(w)      # nothing to explain while allowed
        w._zero_thruster()
        assert _pump(lambda: w._thruster.throttle_pct() == 0 and not w._thruster.busy)


def _motor_at(angle_deg):
    """A sim motor whose shaft starts at ``angle_deg`` (a previous session's)."""
    import math
    from motion_control import MotorLink
    from motion_simulator import SimDamiaoMotor
    sims = []

    def build(st):
        m = SimDamiaoMotor()
        m._pos_rad = m._target_rad = math.radians(angle_deg)
        sims.append(m)
        return MotorLink(lambda: m, st)
    return build, sims


def test_motor_connect_zeroes_where_it_stands_and_holds():
    """2026-10-26 user request: no Disable (sync) / Zero here / Enable by hand —
    Connect makes the current shaft angle 0 and holds it."""
    build, sims = _motor_at(110.0)
    with _Window(builders={"motor": build}) as s:
        w = s.connect_all()
        s.enable_motor()
        names = [e[0] for e in sims[0].call_log]
        i_zero = names.index("set_zero")
        assert names[:i_zero].count("disable") == 2          # no ack on CAN: sent twice
        assert "enable" in names[i_zero:] and "enable" not in names[:i_zero]
        assert abs(w._motor.angle_deg()) < 0.5
        assert w.angle_spin.value() == 0.0
        assert _pump(lambda: "holding" in w.motor_info.text())
        assert "motor_auto_zero" in _kinds(w)
        w.thruster_spin.setValue(10)
        assert _pump(lambda: w.thruster_set_btn.isEnabled())   # thrust allowed at once


def test_motor_is_not_made_limp_before_the_thruster_is_connected():
    """Review H1: a panel that died leaves the Arduino at its last throttle; only
    opening the port (Arduino reset) guarantees 0, so the motor sequence waits."""
    build, sims = _motor_at(30.0)
    with _Window(builders={"motor": build}) as s:
        w = s.w
        w._toggle_motor()
        assert _pump(lambda: w._motor.is_ready, 5)
        _pump(lambda: False, 1.0)
        assert w._motor_auto is not None and w._motor_auto["stage"] == "ready"
        assert not any(e[0] in ("disable", "set_zero") for e in sims[0].call_log)
        assert _pump(lambda: "connect the thruster" in w.motor_info.text())
        w._toggle_thruster()
        s.enable_motor()
        assert abs(w._motor.angle_deg()) < 0.5


def test_motor_that_moves_after_enable_is_disabled():
    """Review M2: a drive still chasing an old POS_VEL target (a lost Disable)
    shows up as motion after Enable — the sequence disables it and says so."""
    import math
    from motion_control import MotorLink
    from motion_simulator import SimDamiaoMotor

    class _Chaser(SimDamiaoMotor):
        def enable(self):
            super().enable()
            self._target_rad, self._vel_rad_s = math.radians(40.0), math.radians(30.0)

    sims = []

    def build(st):
        m = _Chaser()
        sims.append(m)
        return MotorLink(lambda: m, st)
    with _Window(builders={"motor": build}) as s:
        w = s.connect_all()
        assert _pump(lambda: w._motor_auto is None, 10)
        assert _pump(lambda: not w._motor.is_enabled, 3)
        assert "moved to" in w.status_label.text() and "Disable sent" in w.status_label.text()
        assert "motor_auto_failed" in _kinds(w) and not w._motor_zeroed
        assert _pump(lambda: not w.thruster_set_btn.isEnabled())       # next tick


def test_unknown_motor_state_disable_needs_the_jet_known_off():
    """Re-review N1: with no thruster link a dead panel's jet may still run —
    the unknown-state Disable must wait for the thruster (its reset = 0), and
    then works even while STOP ALL is latched."""
    build, sims = _motor_at(20.0)
    with _Window(builders={"motor": build}) as s:
        w = s.w
        w._toggle_motor()
        w._stop_all()                                    # the sequence stops before Disable
        assert _pump(lambda: w._motor_auto is None and w._motor.is_ready, 5)
        assert _pump(lambda: w.enable_btn.text() == "Disable")
        assert not w.enable_btn.isEnabled()
        assert _pump(lambda: "connect the thruster, then Disable" in w.motor_info.text())
        w._toggle_enable()                               # refused
        _pump(lambda: False, 0.3)
        assert ("disable",) not in sims[0].call_log
        assert "jet known OFF" in w.status_label.text()
        w._toggle_thruster()
        assert _pump(lambda: w.enable_btn.isEnabled(), 10)
        w._toggle_enable()
        assert _pump(lambda: ("disable",) in sims[0].call_log, 3)
        assert _pump(lambda: w.enable_btn.text() == "Enable")
        assert w._motor_auto is None                     # nothing restarts by itself


def test_manual_enable_after_a_failed_zero_says_not_zeroed():
    build, sims = _motor_at(25.0)
    with _Window(builders={"motor": build}) as s:
        w = s.w
        w._toggle_thruster()
        assert _pump(lambda: w._thruster.is_ready, 10)
        w._toggle_motor()
        w._stop_all()
        assert _pump(lambda: w._motor_auto is None and w._motor.is_ready, 5)
        w._reset_latch()
        assert _pump(lambda: not w._latched, 3)
        w._toggle_enable()                               # unknown -> Disable
        assert _pump(lambda: w.enable_btn.text() == "Enable" and w.enable_btn.isEnabled(), 5)
        w._toggle_enable()                               # Enable, never zeroed
        assert _pump(lambda: "NOT zeroed this connection" in w.motor_info.text(), 5)
        assert abs(w._motor.angle_deg() - 25.0) < 0.5


def test_motor_zero_at_connect_stops_on_stop_all():
    build, sims = _motor_at(45.0)
    with _Window(builders={"motor": build}) as s:
        w = s.w
        w._toggle_motor()
        w._stop_all()                                  # before the sequence finished
        assert _pump(lambda: w._motor_auto is None, 5)
        assert not w._motor.is_enabled
        assert "motor_auto_failed" in _kinds(w)
        assert not any(e[0] == "enable" for e in sims[0].call_log)


def test_no_motor_interlock_flag():
    with _Window("--no-motor-interlock") as s:
        w = s.connect_all()
        w.thruster_spin.setValue(20)
        w._set_thruster()
        assert _pump(lambda: w._thruster.throttle_pct() == 20)


def test_cut_thrust_does_not_reramp():
    with _Window("--no-motor-interlock", "--ramp-interval", "0.3") as s:
        w = s.connect_all()
        w.thruster_spin.setValue(60)
        w._set_thruster()
        assert _pump(lambda: w._thruster.throttle_pct() >= 10)
        w._cut_thrust()
        assert w._thruster.throttle_pct() == 0
        assert _pump(lambda: not w._cut_pending, 5)   # link reset once the ramp loop exited
        time.sleep(0.8)
        _app().processEvents()
        assert w._thruster.throttle_pct() == 0         # a cancelled ramp must not climb back
        assert w._thruster.ramp_target() is None
        assert w.thruster_set_btn.isEnabled()


def test_stop_all_immediate_zero_resend_latch_reset():
    with _Window("--no-motor-interlock") as s:
        w = s.connect_all()
        s.enable_motor()
        w.thruster_spin.setValue(50)
        w._set_thruster()
        assert _pump(lambda: w._thruster.throttle_pct() >= 10)
        port = w._thruster._driver
        n_before = sum(1 for e in port.call_log if e == ("set_throttle", 0))
        w._stop_all()
        assert w._latched and w._thruster.throttle_pct() == 0
        assert w.reset_btn.isEnabled() and not w.thruster_set_btn.isEnabled()
        assert not w._jog_buttons[(Axis.X, 1)].isEnabled() and not w.go_theta_btn.isEnabled()
        assert w._motor.is_enabled                  # STOP holds, it does not go limp
        assert _pump(lambda: sum(1 for e in port.call_log
                                 if e == ("set_throttle", 0)) >= n_before + 3, 3.0)
        w._reset_latch()
        assert not w._latched
        _app().processEvents()
        assert w.thruster_set_btn.isEnabled() and w._jog_buttons[(Axis.X, 1)].isEnabled()


def test_esc_key_is_stop_all():
    from PyQt5 import QtTest
    with _Window("--no-motor-interlock") as s:
        w = s.connect_all()
        w.show()
        _app().processEvents()
        w.thruster_spin.setValue(30)
        w._set_thruster()
        assert _pump(lambda: w._thruster.throttle_pct() >= 10)
        QtTest.QTest.keyClick(w, Qt.Key_Escape)
        assert _pump(lambda: w._latched, 3.0), "Esc did not reach STOP ALL"
        assert _pump(lambda: w._thruster.throttle_pct() == 0)


def test_disconnect_needs_thrust_zero_and_log_survives():
    with _Window("--no-motor-interlock") as s:
        w = s.connect_all()
        assert _pump(lambda: w._log is not None)
        w.thruster_spin.setValue(20)
        w._set_thruster()
        assert _pump(lambda: w._thruster.throttle_pct() == 20)
        w._toggle_gantry()                          # refused while thrusting
        _app().processEvents()
        assert w._gantry is not None and "Thrust 0" in w.status_label.text()
        w._cut_thrust()
        assert _pump(lambda: not w._cut_pending)
        w._toggle_gantry()
        assert w._gantry is None and w.gantry_btn.text() == "Connect gantry"
        assert w._log is not None                   # the panel-side log keeps running
        n = w._log.n_samples
        assert _pump(lambda: w._log.n_samples > n + 2)


def test_log_files_and_meta():
    with _Window("--no-motor-interlock", "--rig-note", "jet faces +X at 90") as s:
        w = s.connect_all()
        assert _pump(lambda: w._log is not None)
        w.thruster_spin.setValue(10)
        w._set_thruster()
        assert _pump(lambda: w._thruster.throttle_pct() == 10)
        assert _pump(lambda: w._log.n_samples >= 5)              # ~0.5 s of 10 Hz rows
        w._stop_log()
        run = s.run_dir()
        assert sorted(p.name for p in run.iterdir()) == ["events.csv", "meta.json",
                                                         "samples.csv", "thruster.csv"]
        meta = json.loads((run / "meta.json").read_text())
        for key in ("t0_epoch_s", "t0_monotonic_s", "hostname", "whisker_vision", "panel",
                    "settings", "preflight", "gantry", "motor", "pose_stop", "rig_notes"):
            assert key in meta, key
        assert meta["rig_notes"] == ["jet faces +X at 90"]
        assert meta["simulator"] is True and meta["preflight"] == {"verdict": "ready"}
        assert meta["settings"]["thruster_ramp_interval_s"] == 0.05
        samples = (run / "samples.csv").read_text().splitlines()
        assert samples[0].split(",") == list(core.SAMPLE_COLUMNS) and len(samples) > 3
        kinds = [line.split(",")[2] for line in (run / "events.csv").read_text().splitlines()[1:]]
        assert kinds[:2] == ["log", "ready"] and "thrust" in kinds
        thr = (run / "thruster.csv").read_text().splitlines()
        assert thr[0] == "host_epoch_s,throttle_pct,pwm_us" and len(thr) >= 3
        assert w._log is None and "stopped" in w.log_label.text()
        w._new_log()
        assert w._log is not None and w._log.dir != run


def test_no_autolog_flag():
    with _Window("--no-autolog") as s:
        w = s.connect_all()
        time.sleep(0.3)
        _app().processEvents()
        assert w._log is None
        assert not list(Path(s.td.name).rglob("meta.json"))
        w._new_log()
        assert w._log is not None


def test_failed_connect_is_reaped():
    def broken(_st):
        def factory():
            raise RuntimeError("no libFMC4030")
        return GantryLink(factory, _st)
    with _Window(builders={"gantry": broken}) as s:
        w = s.w
        w._toggle_gantry()
        assert _pump(lambda: w._gantry is None, 5.0)
        assert w.gantry_btn.text() == "Connect gantry"
        assert "no libFMC4030" in w.status_label.text()


def test_close_zeroes_thruster_and_stops_gantry_in_order():
    s = _Window("--no-motor-interlock").__enter__()
    try:
        w = s.connect_all()
        w.thruster_spin.setValue(20)
        w._set_thruster()
        assert _pump(lambda: w._thruster.throttle_pct() == 20)
        port = w._thruster._driver
        sim = w._gantry._controller
        w.close()
        assert w._gantry is None and w._motor is None and w._thruster is None
        names = [e[0] for e in port.call_log]
        assert names[-1] == "close" and ("set_throttle", 0) in port.call_log
        assert port.call_log.index(("set_throttle", 0)) < len(names) - 1
        g = sim.call_log
        # the rig STOP sequence: mode-1 stops, then mode-2 stops, then close
        assert (g.index(("stop_axis", Axis.X, 1)) < g.index(("stop_axis", Axis.X, 2))
                < g.index(("close",)))
        assert ("stop_run",) not in g
        w.close()                                   # idempotent
    finally:
        s.td.cleanup()


class _FailingThrusterPort(SimThrusterSerial):
    """Raises on the next write once armed — a USB CDC stall / write timeout."""
    fail_next = False

    def set_throttle(self, pct, reply_timeout=1.0):
        if self.fail_next and pct != 0:
            self.fail_next = False
            raise OSError("write timeout (simulated)")
        return super().set_throttle(pct, reply_timeout)


def _thruster_builder_with_failing_port(holder):
    from motion_control import ThrusterLink

    def build(st):
        port = _FailingThrusterPort()
        holder.append(port)
        return ThrusterLink(lambda: port, st)
    return build


def test_thruster_link_error_fails_closed():
    """A link error means the physical throttle is UNKNOWN: every thrust-0 gate
    closes, Cut stays available, the label says so, samples keep the cache."""
    ports = []
    with _Window("--no-motor-interlock",
                 builders={"thruster": _thruster_builder_with_failing_port(ports)}) as s:
        w = s.connect_all()
        w.thruster_spin.setValue(20)
        w._set_thruster()
        assert _pump(lambda: w._thruster.throttle_pct() == 20)
        ports[0].fail_next = True
        w.thruster_spin.setValue(40)
        w._set_thruster()                                  # the 30-step raises in the worker
        assert _pump(lambda: w._thruster.error is not None, 5)
        assert not w._thruster.is_ready
        assert not w._thrust_is_zero()                     # fail closed
        _app().processEvents()
        if not w._motor.is_enabled:
            assert not w.enable_btn.isEnabled()             # Enable needs thrust 0
        w._toggle_gantry()                                  # Disconnect refused
        assert w._gantry is not None
        assert w.stop_btn.isEnabled()
        assert _pump(lambda: "LINK ERROR" in w.thruster_live.text())
        assert _pump(lambda: w._log.n_samples > 0)
        last = (w._log.dir / "samples.csv").read_text().splitlines()[-1].split(",")
        assert last[list(core.SAMPLE_COLUMNS).index("thr_pct_cmd")] == "20"
        w._cut_thrust()                                      # still possible on an errored link
        assert ports[0].throttle == 0


def test_disable_is_blocked_while_thrusting():
    with _Window() as s:
        w = s.connect_all()
        s.enable_motor()
        w.thruster_spin.setValue(20)
        w._set_thruster()
        assert _pump(lambda: w._thruster.throttle_pct() == 20)
        assert _pump(lambda: not w.enable_btn.isEnabled())   # reads "Disable", but gated
        w._toggle_enable()
        _app().processEvents()
        assert w._motor.is_enabled                           # refused
        assert "jet known OFF" in w.status_label.text()
        w._cut_thrust()
        assert _pump(lambda: not w._cut_pending)
        assert _pump(lambda: w.enable_btn.isEnabled())
        w._toggle_enable()
        assert _pump(lambda: not w._motor.is_enabled)


def test_motor_telemetry_loss_cuts_thrust():
    """Supply dropped: the drive stops answering, the cached enabled flag stays
    True — the interlock must notice and cut the jet."""
    from motion_control import MotorLink
    from motion_simulator import SimDamiaoMotor

    class _Mute(SimDamiaoMotor):
        mute = False

        def request_status(self, timeout=0.2):
            if self.mute:
                return None
            return super().request_status(timeout)
    sims = []

    def build(st):
        m = _Mute()
        sims.append(m)
        return MotorLink(lambda: m, st)
    with _Window(builders={"motor": build}) as s:
        w = s.connect_all()
        s.enable_motor()
        w.thruster_spin.setValue(20)
        w._set_thruster()
        assert _pump(lambda: w._thruster.throttle_pct() == 20)
        sims[0].mute = True
        assert _pump(lambda: not w._motor.telemetry_fresh(), 3)
        assert _pump(lambda: "STALE" in w.motor_info.text())
        assert _pump(lambda: w._thruster.throttle_pct() == 0, P.INTERLOCK_GRACE_S + 3)
        kinds = [line.split(",")[2] for line in
                 (w._log.dir / "events.csv").read_text().splitlines()[1:]]
        assert "interlock_lost" in kinds
        assert not w.thruster_set_btn.isEnabled()           # motor not holding


def test_go_theta_takes_the_short_way():
    with _Window() as s:
        w = s.connect_all()
        s.enable_motor()
        w.angle_spin.setValue(350.0)                         # measured 0 -> -10, not +350
        w._go_theta()
        assert _pump(lambda: not w._motor.busy and w._motor.angle_deg() is not None, 10)
        assert abs(w._motor.angle_deg() - (-10.0)) < 1.0


def test_cut_resends_zero_and_reset_waits_for_zero():
    with _Window("--no-motor-interlock", "--ramp-interval", "0.3") as s:
        w = s.connect_all()
        port = w._thruster._driver
        w.thruster_spin.setValue(60)
        w._set_thruster()
        assert _pump(lambda: w._thruster.throttle_pct() >= 10)
        n0 = sum(1 for e in port.call_log if e == ("set_throttle", 0))
        w._cut_thrust()
        assert _pump(lambda: not w._cut_pending, 5)
        assert sum(1 for e in port.call_log if e == ("set_throttle", 0)) >= n0 + 2  # cut + finish


def test_close_records_the_zero_in_the_log():
    s = _Window("--no-motor-interlock").__enter__()
    try:
        w = s.connect_all()
        assert _pump(lambda: w._log is not None)
        w.thruster_spin.setValue(20)
        w._set_thruster()
        assert _pump(lambda: w._thruster.throttle_pct() == 20)
        assert _pump(lambda: w._log.n_samples >= 3)
        run = w._log.dir
        w.close()
        thr = (run / "thruster.csv").read_text().splitlines()
        assert thr[-1].split(",")[1] == "0", thr[-3:]
        samples = (run / "samples.csv").read_text().splitlines()
        last = samples[-1].split(",")
        cols = list(core.SAMPLE_COLUMNS)
        assert last[cols.index("thr_pct_cmd")] == "0" and last[cols.index("latched")] == "1"
    finally:
        s.td.cleanup()


def test_send_moves_only_edited_axes_and_fields_follow_the_rig():
    with _Window() as s:
        w = s.connect_all()
        assert _pump(lambda: len(w._gantry_targets_initialized) == 3)
        assert not w.send_btn.isEnabled()                 # nothing edited yet
        _send(w, X=10.0)
        assert _pump(lambda: abs(w._target_spins[Axis.X].value() - 10.0) < 0.2)  # re-seeded
        w._target_spins[Axis.Y].setValue(7.5)             # the only edit
        assert _pump(lambda: w.send_btn.isEnabled())
        w._send_targets()
        assert w._move_targets == {Axis.Y: 7.5}          # X NOT dragged back
        _settle(w)
        pos = w._gantry.positions_mm()
        assert abs(pos[0] - 10.0) < 0.2 and abs(pos[1] - 7.5) < 0.2
        assert not w._target_dirty and not w.send_btn.isEnabled()


def test_stop_log_keeps_thruster_rows_on_an_errored_link():
    ports = []
    with _Window("--no-motor-interlock",
                 builders={"thruster": _thruster_builder_with_failing_port(ports)}) as s:
        w = s.connect_all()
        assert _pump(lambda: w._log is not None)
        w.thruster_spin.setValue(20)
        w._set_thruster()
        assert _pump(lambda: w._thruster.throttle_pct() == 20)
        ports[0].fail_next = True
        w.thruster_spin.setValue(40)
        w._set_thruster()
        assert _pump(lambda: w._thruster.error is not None, 5)
        w._stop_log()
        run = s.run_dir()
        thr = (run / "thruster.csv").read_text().splitlines()
        assert len(thr) >= 3, thr                         # header + the 10/20 steps at least
        meta = json.loads((run / "meta.json").read_text())
        assert meta["links_at_stop"]["thruster"]["error"]


class _DeadZeroPort(SimThrusterSerial):
    """Every write fails once armed — the board is gone."""
    dead = False

    def set_throttle(self, pct, reply_timeout=1.0):
        if self.dead:
            raise OSError("port vanished (simulated)")
        return super().set_throttle(pct, reply_timeout)


def test_close_with_a_dead_port_says_the_zero_is_unconfirmed(capsys=None):
    from motion_control import ThrusterLink
    ports = []

    def build(st):
        port = _DeadZeroPort()
        ports.append(port)
        return ThrusterLink(lambda: port, st)
    s = _Window("--no-motor-interlock", builders={"thruster": build}).__enter__()
    try:
        w = s.connect_all()
        assert _pump(lambda: w._log is not None)
        w.thruster_spin.setValue(20)
        w._set_thruster()
        assert _pump(lambda: w._thruster.throttle_pct() == 20)
        assert _pump(lambda: w._log.n_samples >= 2)
        run = w._log.dir
        ports[0].dead = True
        w.close()
        kinds = [line.split(",")[2] for line in (run / "events.csv").read_text().splitlines()[1:]]
        assert "zero_unconfirmed" in kinds, kinds
        meta = json.loads((run / "meta.json").read_text())
        assert meta["links_at_stop"]["thruster"]["error"] or \
            meta["links_at_stop"]["thruster"]["last_command_error"]
    finally:
        s.td.cleanup()


def test_stop_all_cancels_pending_recover_and_disconnect_flags():
    with _Window() as s:
        w = s.connect_all()
        w._gantry_recover_pending = True
        w._motor_disconnect_after_disable = True
        w._stop_all()
        assert not w._gantry_recover_pending and not w._motor_disconnect_after_disable
        assert "disconnect cancelled" in w.status_label.text()
        _app().processEvents()
        assert not w.recover_gantry_btn.isEnabled() and not w.recover_motor_btn.isEnabled()
        _pump(lambda: False, 1.8)
        w._reset_latch()
        assert not w._latched


# ------------------------------------------------------------- hold-to-jog
def _press(btn):
    from PyQt5 import QtTest
    QtTest.QTest.mousePress(btn, Qt.LeftButton)


def _release(btn):
    from PyQt5 import QtTest
    QtTest.QTest.mouseRelease(btn, Qt.LeftButton)


def _stable(w, axis, dt=0.4):
    _pump(lambda: False, 0.3)          # the stop is asynchronous (its own thread)
    a = w._gantry.positions_mm()[int(axis)]
    _pump(lambda: False, dt)
    return abs(w._gantry.positions_mm()[int(axis)] - a) < 0.05


def _kinds(w):
    return [line.split(",")[2] for line in
            (w._log.dir / "events.csv").read_text().splitlines()[1:]]


def test_release_stops_without_latching_and_the_next_press_works():
    """2026-10-26 hardware: a release latched the link, the stop did not take,
    and the buttons stayed dead. Now: mode-1 jog stop, no latch, press again."""
    with _Window("--speed", "20") as s:
        w = s.connect_all()
        w.show()
        btn = w._jog_buttons[(Axis.X, 1)]
        sim = w._gantry._controller
        for round_ in range(2):
            _press(btn)
            assert w._jog is not None, w.status_label.text()
            assert _pump(lambda: w._gantry.positions_mm()[0] > 5.0 + 10 * round_, 3)
            assert btn.isEnabled()                       # the held button stays live
            assert not w._jog_buttons[(Axis.Y, 1)].isEnabled()
            n_stop_run = sim.call_log.count(("stop_run",))
            _release(btn)
            assert w._jog is None
            assert _released(w), w.status_label.text()
            assert not w._latched                        # NOT a STOP ALL
            assert sim.call_log.count(("stop_run",)) == n_stop_run   # no STOP sequence
            assert ("stop_axis", Axis.X, 1) in sim.call_log
            assert _stable(w, Axis.X)
            assert w._gantry.positions_mm()[0] < 495.0
            assert _pump(lambda: btn.isEnabled() and w._jog_buttons[(Axis.Y, 1)].isEnabled(), 2)
        kinds = _kinds(w)
        assert kinds.count("jog_start") == 2 and kinds.count("jog_stopped") == 2
        assert "stop_all" not in kinds


class _Decel(SimGantryController):
    """Mode 1 decelerates at 50 mm/s² (the link's acc), as the hardware's run-on
    says it does (5.5-27.2 mm at 50 mm/s, data/20261026/1026_140909_disturb/
    events.csv); the base sim stops dead."""
    A_MM = 50.0

    def __init__(self):
        super().__init__()
        self._dec = [None, None, None]

    def _advance(self):
        dt = time.monotonic() - self._last_tick
        for i in range(3):
            if self._dec[i] is not None:
                self._speed[i] = max(0.0, self._speed[i] - self._dec[i] * dt)
                if self._speed[i] == 0.0:
                    self._target[i], self._dec[i] = self._pos[i], None
        super()._advance()

    def stop_axis(self, axis, mode=2):
        self.call_log.append(("stop_axis", axis, mode))
        self._advance()
        i = int(axis)
        moving = abs(self._target[i] - self._pos[i]) > 1e-9 and self._speed[i] > 0
        if mode == 1 and moving:
            if self._dec[i] is None:
                self._dec[i] = self.A_MM / {0: 8.25, 1: 2.5, 2: 0.5}[i]
        else:
            self._target[i], self._speed[i], self._dec[i] = self._pos[i], 0.0, None


def test_another_axis_can_be_jogged_while_a_released_one_is_stopping():
    """2026-10-27 user request: after releasing +X, -Y must not stay dead for
    about a second. Only the stopping axis waits; its release watch runs on."""
    with _Window("--speed", "40", builders=_rig(_Decel)) as s:
        w = s.connect_all()
        w.show()
        _put(w, Axis.Y, 100.0)
        bx, by = w._jog_buttons[(Axis.X, 1)], w._jog_buttons[(Axis.Y, -1)]
        _press(bx)
        assert _pump(lambda: w._gantry.positions_mm()[0] > 5.0, 3)
        _release(bx)
        assert _pump(lambda: by.isEnabled(), 0.5)          # X still running on
        assert Axis.X in w._releasing_axes()
        assert not bx.isEnabled()                          # X itself waits
        _press(by)
        assert w._jog is not None and w._jog["axis"] == Axis.Y, w.status_label.text()
        assert _pump(lambda: w._gantry.positions_mm()[1] < 98.0, 3)
        _release(by)
        assert _released(w, 6), w.status_label.text()
        # review M1: Y's stop (mode 1 on every axis) also reached X while X was still
        # stopping — X's record says so, or an ignored first stop would look routine
        assert "stopped again by 1 later release" in (w._log.dir / "events.csv").read_text()
        assert not w._latched
        kinds = _kinds(w)
        assert kinds.count("jog_stopped") == 2 and "stop_all" not in kinds
        assert _pump(lambda: bx.isEnabled() and by.isEnabled(), 2)


class _DecelNoStopY(_Decel):
    """X decelerates normally; Y ignores every stop (a stop that never lands)."""
    def stop_axis(self, axis, mode=2):
        if int(axis) == 1:
            self.call_log.append(("stop_axis", axis, mode))
            return
        super().stop_axis(axis, mode)


def test_a_second_release_that_does_not_stop_still_escalates():
    """The per-axis watch must not let a stuck second axis hide behind the first."""
    with _Window("--speed", "20", builders=_rig(_DecelNoStopY)) as s:
        w = s.connect_all()
        w.show()
        _put(w, Axis.Y, 100.0)
        bx, by = w._jog_buttons[(Axis.X, 1)], w._jog_buttons[(Axis.Y, -1)]
        _press(bx)
        assert _pump(lambda: w._gantry.positions_mm()[0] > 5.0, 3)
        _release(bx)
        assert _pump(lambda: by.isEnabled(), 0.5)
        _press(by)
        assert _pump(lambda: w._gantry.positions_mm()[1] < 98.0, 3)
        _release(by)                                       # Y never stops
        assert _pump(lambda: w._latched, 6), w.status_label.text()
        events = (w._log.dir / "events.csv").read_text()
        assert "(not taken)" in events and (" Y moved " in events or " Y at " in events)


def test_an_ignored_stop_is_caught_by_its_speed_not_only_its_distance():
    """Review H1 (10-27): at 60 mm/s the slack is 5 + 2·(36 + 18) = 113 mm, about
    what is left of a 300 mm piece — a stop the controller ignores (10-26) would
    end the piece inside it. Still at cruise speed after the stop = caught."""
    with _Window("--speed", "60", builders=_rig(_NoStop)) as s:
        w = s.connect_all()
        w.show()
        bx = w._jog_buttons[(Axis.X, 1)]
        _press(bx)
        assert _pump(lambda: w._gantry.positions_mm()[0] > 20.0, 3)
        x0 = w._gantry.positions_mm()[0]
        _release(bx)
        assert _pump(lambda: w._latched, 4), w.status_label.text()
        assert "not decelerating" in (w._log.dir / "events.csv").read_text()
        assert w._gantry.positions_mm()[0] - x0 < 113.0          # before the slack


def test_z_is_capped_at_50_and_send_at_50():
    with _Window("--speed", "100") as s:
        w = s.connect_all()
        w.show()
        bz = w._jog_buttons[(Axis.Z, 1)]
        _press(bz)
        assert w._jog is not None and w._jog["speed"] == 50.0
        assert _pump(lambda: w._gantry.positions_mm()[2] > 2.0, 3)
        _release(bz)
        assert _released(w, 5)
        sim = w._gantry._controller
        z = [e for e in sim.call_log if e[0] == "jog" and e[1] == Axis.Z][-1]
        assert abs(z[3] - 50.0 / 0.5) < 1e-6                     # 50 mm/s in Z units
        _still(w)
        _send(w, X=30.0)
        x = [e for e in sim.call_log if e[0] == "jog" and e[1] == Axis.X][-1]
        assert x[4] is False and abs(x[3] - 50.0 / 8.25) < 1e-6
        assert "move" in _kinds(w) and "@ 50 mm/s" in (w._log.dir / "events.csv").read_text()


def test_stop_all_above_50_marks_the_homes_suspect():
    with _Window("--speed", "80") as s:
        w = s.connect_all()
        w.show()
        w._teach(Axis.X, "lo")
        bx = w._jog_buttons[(Axis.X, 1)]
        _press(bx)
        assert _pump(lambda: w._gantry.positions_mm()[0] > 5.0, 3)
        w._stop_all()
        assert "X" in w._suspect_axes
        assert "homes_suspect" in _kinds(w)
        _release(bx)


def test_speed_goes_to_100():
    with _Window("--speed", "150") as s:
        w = s.w
        assert w.speed_spin.maximum() == core.RIG_MAX_SPEED_MM_S == 100.0
        assert w.speed_spin.value() == 100.0
        assert w.pattern_speed_spin.maximum() == 100.0


def test_free_hold_is_one_continuous_move_not_10mm_hops():
    """No home on that side: ONE controller-relative move per piece (the
    link's own relative moves are capped at 10 mm)."""
    with _Window("--speed", "40", builders=_rig(_FactoryLimits)) as s:
        w = s.connect_all()
        w.show()
        btn = w._jog_buttons[(Axis.X, -1)]
        _press(btn)
        assert w._jog is not None and w._jog["end"] is None
        assert _pump(lambda: w._gantry.positions_mm()[0] < -25.0, 3)
        sim = w._gantry._controller
        jogs = [e for e in sim.call_log if e[0] == "jog" and e[1] == Axis.X]
        assert len(jogs) == 1, jogs                 # one command, no hops
        assert jogs[0][4] is True                   # controller-relative
        assert abs(jogs[0][2] * 8.25 + 300.0) < 0.5  # one 300 mm piece (yaml default)
        _release(btn)
        assert _released(w)
        assert _stable(w, Axis.X)
        x = w._gantry.positions_mm()[0]
        assert -300.0 < x < -25.0                    # stopped where it was released


def test_plus_jog_without_a_home_is_a_piece_not_a_run_to_the_firmware_stop():
    """Review CRITICAL-1: the + firmware stop (X 1650 counter mm) has no fixed
    physical place, so it may only cap a piece, never become the jog's end."""
    with _Window("--speed", "40", builders=_rig(_FactoryLimits)) as s:
        w = s.connect_all()
        w.show()
        btn = w._jog_buttons[(Axis.X, 1)]
        _press(btn)
        assert w._jog is not None and w._jog["end"] is None
        assert "NO software limit" in w.status_label.text()
        sim = w._gantry._controller
        assert _pump(lambda: any(e[0] == "jog" for e in sim.call_log), 2)
        jogs = [e for e in sim.call_log if e[0] == "jog" and e[1] == Axis.X]
        assert len(jogs) == 1 and abs(jogs[0][2] * 8.25 - 300.0) < 0.5
        _release(btn)
        assert _released(w)
        # within a piece of the stop: the piece shrinks to the stop
        _put(w, Axis.Y, 450.0)                     # Y firmware stop 500
        btn = w._jog_buttons[(Axis.Y, 1)]
        _press(btn)
        assert _pump(lambda: any(e[0] == "jog" and e[1] == Axis.Y for e in sim.call_log), 2)
        jogs = [e for e in sim.call_log if e[0] == "jog" and e[1] == Axis.Y]
        assert abs(jogs[-1][2] * 2.5 - (50.0 - P.LIMIT_EDGE_MM)) < 0.05
        _release(btn)
        assert _released(w)


def test_reset_is_refused_while_an_axis_still_moves():
    """Review HIGH-3: a STOP that did not take leaves a move running; Reset
    (and every new command) waits for the gantry to be physically still."""
    with _Window("--speed", "20", builders=_rig(_NoStop)) as s:
        w = s.connect_all()
        w.show()
        btn = w._jog_buttons[(Axis.X, 1)]
        _press(btn)
        assert _pump(lambda: w._gantry.positions_mm()[0] > 2.0, 3)
        w._stop_all()                                  # does nothing in _NoStop
        _release(btn)
        _pump(lambda: False, 1.6)                      # past the stop sequence
        w._reset_latch()
        assert w._latched and "Reset refused" in w.status_label.text()
        assert not w._jog_buttons[(Axis.Y, 1)].isEnabled()
        assert "reset_refused" in _kinds(w)


def test_drift_on_another_axis_is_watched_during_a_jog():
    """Review MEDIUM-6: a press on Y must not blind the drift watch for X."""
    with _Window("--speed", "20") as s:
        w = s.connect_all()
        w.show()
        bx = w._jog_buttons[(Axis.X, 1)]
        _press(bx)
        assert _pump(lambda: w._gantry.positions_mm()[0] > 3.0, 3)
        _release(bx)
        assert _released(w)
        by = w._jog_buttons[(Axis.Y, 1)]
        _press(by)
        assert w._jog is not None
        sim = w._gantry._controller
        sim._target[0] = sim._pos[0] + 40.0 / 8.25     # X creeps on its own
        sim._speed[0] = 10.0 / 8.25
        assert _pump(lambda: w._latched, 3), w.status_label.text()
        assert "still moving" in w.status_label.text()
        _release(by)


def test_free_jog_continues_past_one_piece_while_held():
    with _Window("--speed", "50", "--jog-piece", "40", builders=_rig(_FactoryLimits)) as s:
        w = s.connect_all()
        w.show()
        btn = w._jog_buttons[(Axis.X, -1)]
        _press(btn)
        assert _pump(lambda: w._gantry.positions_mm()[0] < -50.0, 6)
        jogs = [e for e in w._gantry._controller.call_log if e[0] == "jog" and e[1] == Axis.X]
        assert len(jogs) >= 2                         # the next piece followed
        _release(btn)
        assert _released(w)
        assert _stable(w, Axis.X)


def test_lost_release_is_caught_by_the_watchdog():
    with _Window() as s:
        w = s.connect_all()
        w.show()
        btn = w._jog_buttons[(Axis.Z, 1)]
        _press(btn)
        assert _pump(lambda: w._gantry.positions_mm()[2] > 1.0, 3)
        btn.setDown(False)                     # no released() signal at all
        assert _pump(lambda: w._jog is None, 1.0)
        assert _released(w)
        assert _stable(w, Axis.Z)


def test_jog_at_the_limit_does_not_start():
    with _Window() as s:
        w = s.connect_all()
        w.show()
        btn = w._jog_buttons[(Axis.X, -1)]       # X at 0 = the sim's lower limit
        _press(btn)
        assert w._jog is None
        assert "limit" in w.status_label.text()
        _release(btn)
        sim = w._gantry._controller
        assert not any(e[0] == "jog" for e in sim.call_log)


def test_outside_the_homes_jogs_only_back_inside():
    with _Window(builders=_rig(_FactoryLimits)) as s:
        w = s.connect_all()
        w.show()
        w._teach(Axis.X, "lo")
        _put(w, Axis.X, 100.0)
        w._teach(Axis.X, "hi")
        _put(w, Axis.X, 150.0)                    # beyond + (as after a power cycle)
        btn = w._jog_buttons[(Axis.X, 1)]
        _press(btn)
        assert w._jog is None and "beyond" in w.status_label.text()
        _release(btn)
        btn = w._jog_buttons[(Axis.X, -1)]
        _press(btn)
        assert w._jog is not None and w._jog["end"] == 0.0
        _release(btn)
        assert _released(w)


def test_stop_all_during_a_jog_latches_and_ends_it():
    with _Window() as s:
        w = s.connect_all()
        w.show()
        btn = w._jog_buttons[(Axis.X, 1)]
        _press(btn)
        assert _pump(lambda: w._gantry.positions_mm()[0] > 2.0, 3)
        w._stop_all()
        assert w._jog is None and w._latched
        _release(btn)
        assert _stable(w, Axis.X)
        assert not btn.isEnabled()               # latched until Reset
        _pump(lambda: False, 1.8)
        w._reset_latch()
        assert not w._latched
        assert _pump(lambda: btn.isEnabled())


def test_teach_is_refused_while_moving():
    with _Window() as s:
        w = s.connect_all()
        w.show()
        btn = w._jog_buttons[(Axis.Y, 1)]
        _press(btn)
        assert _pump(lambda: w._gantry.positions_mm()[1] > 1.0, 3)
        assert not w._teach_buttons[(Axis.Y, "hi")].isEnabled()
        w._teach(Axis.Y, "hi")
        assert "Y" not in w._taught and "at rest" in w.status_label.text()
        _release(btn)
        assert _released(w)
        assert _pump(lambda: w._teach_buttons[(Axis.Y, "hi")].isEnabled())


def test_jog_stops_when_the_axis_does_not_move():
    from motion_simulator import SimGantryController

    class _Stuck(SimGantryController):
        def jog_single_axis(self, axis, position_units, speed_units, acc_units,
                            dec_units, *, relative=True):
            self.call_log.append(("jog", axis, position_units, speed_units, relative))
            # accepted, but the axis never moves (limit switch / fault)

    with _Window("--jog-piece", "20", builders=_rig(_Stuck)) as s:
        w = s.connect_all()
        w.show()
        btn = w._jog_buttons[(Axis.Y, 1)]
        _press(btn)
        assert _pump(lambda: w._jog is None, 3)
        assert "did not move" in w.status_label.text()
        _release(btn)


# ------------------------------------------------- safety review 2026-10-26
from motion_simulator import SimGantryController as _SimG  # noqa: E402


class _BlindStop(_SimG):
    """is_axis_stopped raises once when armed: the link swallows it, clears
    busy, and the controller keeps running the move (review finding #1)."""
    blind = False

    def is_axis_stopped(self, axis):
        if self.blind:
            self.blind = False
            raise RuntimeError("FMC4030_Check_Axis_Is_Stop failed (simulated)")
        return super().is_axis_stopped(axis)


class _NoStop(_SimG):
    """Stop calls are accepted but do nothing (a stop that never lands)."""
    def stop_run(self):
        self.call_log.append(("stop_run",))

    def stop_axis(self, axis, mode=2):
        self.call_log.append(("stop_axis", axis, mode))


class _FailStop(_SimG):
    """Stop calls raise (TCP glitch): the release stop FAILS."""
    fail = False

    def stop_axis(self, axis, mode=2):
        if self.fail:
            self.call_log.append(("stop_axis_failed", axis, mode))
            raise RuntimeError("Stop_Single_Axis failed (simulated)")
        super().stop_axis(axis, mode)


class _Hw1026(_FactoryLimits):
    """A MODEL of the 2026-10-26 controller, half observed and half assumed:
    stop_run and the mode-2 stop were accepted while the jog ran on (observed,
    data/20261026/1026_131512_disturb/); mode 1 stopped 18/18 releases later
    that day (data/20261026/1026_140909_disturb/events.csv). Inside STOP ALL it is still unobserved — passing here is
    not hardware evidence for that part."""
    def stop_run(self):
        self.call_log.append(("stop_run",))

    def stop_axis(self, axis, mode=2):
        if mode == 1:
            super().stop_axis(axis, mode)
        else:
            self.call_log.append(("stop_axis", axis, mode))


def _rig(cls):
    return {"gantry": lambda st: core.rig_gantry_link(cls, st)}


def test_release_and_stop_all_stop_the_1026_controller():
    with _Window("--speed", "20", builders=_rig(_Hw1026)) as s:
        w = s.connect_all()
        w.show()
        btn = w._jog_buttons[(Axis.X, -1)]
        _press(btn)
        assert _pump(lambda: w._gantry.positions_mm()[0] < -5.0, 3)
        _release(btn)
        assert _released(w) and not w._latched, w.status_label.text()
        assert _stable(w, Axis.X)
        _press(btn)
        assert _pump(lambda: w._gantry.positions_mm()[0] < -15.0, 3)
        w._stop_all()                                     # Esc
        assert _stable(w, Axis.X)
        _pump(lambda: False, 1.5)
        assert "still moving" not in w.status_label.text()
        _release(btn)


class _EarlyStop(_SimG):
    """is_axis_stopped says 'stopped' once while the axis still runs: busy
    clears with no error at all."""
    early = False

    def is_axis_stopped(self, axis):
        if self.early:
            self.early = False
            return True
        return super().is_axis_stopped(axis)


def test_release_stops_even_when_the_link_says_idle():
    with _Window("--speed", "20", builders=_rig(_EarlyStop)) as s:
        w = s.connect_all()
        w.show()
        btn = w._jog_buttons[(Axis.X, 1)]
        _press(btn)
        assert _pump(lambda: w._gantry.positions_mm()[0] > 2.0, 3)
        sim = w._gantry._controller
        n_jogs = len([e for e in sim.call_log if e[0] == "jog"])
        sim.early = True
        assert _pump(lambda: not w._gantry.busy, 2)     # link thinks the move ended
        _pump(lambda: False, 0.5)
        assert sim._pos[0] != sim._target[0]            # the controller does not
        # not at rest: no second relative move stacked on the running one
        assert len([e for e in sim.call_log if e[0] == "jog"]) == n_jogs
        n = sim.call_log.count(("stop_axis", Axis.X, 1))
        _release(btn)
        assert _pump(lambda: sim.call_log.count(("stop_axis", Axis.X, 1)) > n, 2)
        assert _released(w)
        assert _stable(w, Axis.X)


def test_a_swallowed_link_error_ends_the_jog():
    """is_axis_stopped raises: the worker swallows it and clears busy while the
    controller runs on (review MEDIUM). The jog ends (stop sent) instead of
    queueing the next move."""
    with _Window("--speed", "40", builders=_rig(_BlindStop)) as s:
        w = s.connect_all()
        w.show()
        _put(w, Axis.X, 200.0)
        sim = w._gantry._controller
        btn = w._jog_buttons[(Axis.X, -1)]
        _press(btn)
        assert _pump(lambda: w._gantry.positions_mm()[0] < 195.0, 3)
        n = len([e for e in sim.call_log if e[0] == "jog"])
        sim.blind = True
        assert _pump(lambda: w._jog is None, 3)
        assert "link error" in w.status_label.text()
        assert len([e for e in sim.call_log if e[0] == "jog"]) == n
        _release(btn)
        assert _released(w)
        assert _stable(w, Axis.X)


def test_a_failed_release_stop_escalates_to_stop_all():
    with _Window(builders=_rig(_FailStop)) as s:
        w = s.connect_all()
        w.show()
        btn = w._jog_buttons[(Axis.X, 1)]
        _press(btn)
        assert _pump(lambda: w._gantry.positions_mm()[0] > 2.0, 3)
        w._gantry._controller.fail = True
        _release(btn)
        assert _pump(lambda: w._latched, 3), w.status_label.text()
        assert "release stop FAILED" in w.status_label.text()
        kinds = _kinds(w)
        assert "stuck" in kinds and "stop_all" in kinds


def test_a_stop_that_does_not_take_escalates_to_stop_all():
    with _Window("--speed", "20", builders=_rig(_NoStop)) as s:
        w = s.connect_all()
        w.show()
        btn = w._jog_buttons[(Axis.X, 1)]
        _press(btn)
        assert _pump(lambda: w._gantry.positions_mm()[0] > 2.0, 3)
        _release(btn)                       # the stop "succeeds" but the axis runs on
        assert _pump(lambda: w._latched, 4)
        assert "still moving" in w.status_label.text() or "STALE" in w.status_label.text()
        assert "(not taken)" in (w._log.dir / "events.csv").read_text()


def test_thruster_panel_is_lean_and_ramps_at_the_config_values():
    """2026-10-26 user request: no RAMP/EVERY fields, no Cut button, 'Turn off'
    instead of 'Thrust 0 (ramp)', no explanatory notes. The ramp is the YAML's."""
    with _Window("--no-motor-interlock") as s:
        w = s.connect_all()
        assert w.thruster_pill._label == "ARDUINO"     # the link is the Arduino, not the ESC
        for gone in ("thruster_cut_btn", "ramp_step_spin", "ramp_interval_spin", "esc_note",
                     "jet_hint", "sync_disable_btn", "zero_btn"):
            assert not hasattr(w, gone), gone
        assert w.thruster_zero_btn.text() == "Turn off"
        assert "ramps 10 % every" in w.thruster_set_btn.toolTip()
        w.thruster_spin.setValue(10)
        w._set_thruster()
        assert _pump(lambda: "1540 µs" in w.thruster_live.text(), 3), w.thruster_live.text()
        assert w.thruster_live.text().startswith("+10 %")


def test_clear_before_the_frame_check_cannot_launder_old_homes():
    """Re-review N1: Clear right after a reconnect used to overwrite the frame
    reference with the NEW counter reading, so the old-frame homes of the other
    axes passed the check."""
    _Persistent.shared = None
    with _Window(builders=_rig(_Persistent)) as s:
        w = s.connect_all()
        _put(w, Axis.X, 100.0)
        w._teach(Axis.X, "hi")
        _put(w, Axis.Y, 50.0)
        w._teach(Axis.Y, "hi")
        w._toggle_gantry()
        _Persistent.shared = None                      # power cycle: counter restarts
        w._toggle_gantry()
        w._clear_limits(Axis.X)                        # before the first look
        assert "X" in w._taught and not w._clear_buttons[Axis.X].isEnabled()
        assert _pump(lambda: w._limits_applied)
        assert not w._taught and "homes_dropped" in _kinds(w)


def test_recover_whose_stop_does_not_take_escalates():
    """Re-review N3: Recover used to wait for rest silently forever."""
    with _Window("--speed", "20", builders=_rig(_NoStop)) as s:
        w = s.connect_all()
        w._target_spins[Axis.X].setValue(300.0)
        assert _pump(lambda: w.send_btn.isEnabled())
        w._send_targets()
        assert _pump(lambda: w._gantry.positions_mm()[0] > 2.0, 3)
        w._recover_gantry()                            # its stop does nothing here
        assert _pump(lambda: w._latched, P.GANTRY_STOP_SETTLE_TIMEOUT_S + 3)
        assert "not at rest" in w.status_label.text() or "still" in w.status_label.text()
        assert "stuck" in _kinds(w)


def test_one_failing_stop_does_not_skip_the_others():
    """Re-review N5: an exception in the gantry stop skipped the thruster zero."""
    with _Window("--no-motor-interlock", "--ramp-interval", "0") as s:
        w = s.connect_all()
        w.thruster_spin.setValue(20)
        w._set_thruster()
        assert _pump(lambda: w._thruster.throttle_pct() == 20)
        real = w._gantry.stop_all

        def boom(wait=True):
            raise RuntimeError("can't start new thread")
        w._gantry.stop_all = boom
        try:
            w._stop_all()
        finally:
            w._gantry.stop_all = real
        assert w._latched and w._thruster.throttle_pct() == 0
        assert "FAILED" in w.status_label.text()
        assert "stop_failed" in _kinds(w)


def test_drift_is_watched_during_a_jog_started_after_the_window():
    """Re-review caveat to MEDIUM-6: a jog started long after the last settle
    opens the drift watch for the other axes itself."""
    with _Window("--speed", "20") as s:
        w = s.connect_all()
        w.show()
        w._jog_watch_until = 0.0                      # the 5 s window is long gone
        by = w._jog_buttons[(Axis.Y, 1)]
        _press(by)
        assert w._jog is not None
        _pump(lambda: False, 0.3)                     # reference taken
        sim = w._gantry._controller
        sim._target[0] = sim._pos[0] + 40.0 / 8.25    # X creeps on its own
        sim._speed[0] = 10.0 / 8.25
        assert _pump(lambda: w._latched, 3), w.status_label.text()
        _release(by)


def test_panel_never_calls_the_sdk_or_converts_units():
    """The safety-code-reviewer invariant: no SDK call and no mm<->unit math
    on the GUI thread — the links own both. Mechanically checked."""
    src = Path(P.__file__).read_text()
    forbidden = re.compile(r"\b(jog_single_axis|home_axis|stop_axis|stop_run|command_pos_vel"
                           r"|set_zero|get_status|SCALE_MM_PER_UNIT|units_to_mm|mm_to_units)\b")
    hits = [m.group(0) for m in forbidden.finditer(src)]
    assert not hits, hits
    assert "serial.Serial(" not in src
    core_src = Path(core.__file__).read_text()
    assert core_src.count("serial.Serial(") == 1          # only the thruster factory


# ----------------------------------------------------- patterns (2026-10-26)
def _start(w, mode, size, speed):
    w.pattern_combo.setCurrentIndex(w.pattern_combo.findData(mode))
    w.pattern_size_spin.setValue(size)
    w.pattern_speed_spin.setValue(speed)
    assert _pump(lambda: w.pattern_start_btn.isEnabled())
    w._start_pattern()
    assert w._pattern is not None, w.status_label.text()


def _track(w, seconds):
    """Every position the panel saw for ``seconds`` (100 ms readout)."""
    seen = []
    deadline = time.time() + seconds
    while time.time() < deadline:
        _app().processEvents()
        pos = w._gantry.positions_mm()
        if pos is not None:
            seen.append(pos)
        time.sleep(0.02)
    return seen


def test_circle_pattern_stays_in_its_box_and_stops_without_latch():
    with _Window() as s:
        w = s.connect_all()
        _put(w, Axis.Y, 100.0)
        _put(w, Axis.Z, 60.0)
        _start(w, "circle", 10.0, 30.0)
        assert not w._jog_buttons[(Axis.Y, 1)].isEnabled()
        assert not w.pattern_start_btn.isEnabled() and w.pattern_stop_btn.isEnabled()
        assert _pump(lambda: w._pattern is not None and w._pattern["lap"] >= 1, 15), \
            w.status_label.text()
        seen = _track(w, 0.5)
        assert w._pattern is not None and not w._latched
        kinds = _kinds(w)
        assert "pattern_start" in kinds and "pattern_lap" in kinds
        assert all(abs(p[0]) < 0.01 for p in seen)                     # X never moves
        assert all(89.9 <= p[1] <= 110.1 and 49.9 <= p[2] <= 70.1 for p in seen)
        # on the chords: never further out than r, never much inside (sagitta)
        radii = [((p[1] - 100.0) ** 2 + (p[2] - 60.0) ** 2) ** 0.5 for p in seen]
        assert max(radii) <= 10.1 and min(radii) >= 10.0 * 0.95
        w._end_pattern("stopped")
        assert _pump(lambda: w._jog_release is None, 5)
        assert not w._latched and w._pattern is None
        assert "pattern_stopped" in _kinds(w)
        assert _pump(lambda: w._jog_buttons[(Axis.Y, 1)].isEnabled(), 3)
        stops = [e for e in w._gantry._controller.call_log if e[0] == "stop_axis"]
        assert stops and {e[2] for e in stops} == {1}                   # non-latching mode 1
        assert [e[1] for e in stops[:2]] == [Axis.Y, Axis.Z]            # moving axes first


def test_y_back_and_forth_stays_within_its_range():
    with _Window() as s:
        w = s.connect_all()
        _put(w, Axis.Y, 100.0)
        _put(w, Axis.Z, 30.0)
        _start(w, "line_y", 40.0, 30.0)
        seen = _track(w, 4.5)                    # 20 mm out, then 40 back, then 40 out
        ys = [p[1] for p in seen]
        assert max(ys) > 118.0 and min(ys) < 82.0, (min(ys), max(ys))   # 50 ms polls
        assert 79.9 <= min(ys) and max(ys) <= 120.1
        targets = {round(e[2] * 2.5, 3) for e in w._gantry._controller.call_log
                   if e[0] == "jog" and e[1] == Axis.Y and e[4] is False}
        assert targets == {120.0, 80.0}, targets                      # absolute, both ends
        assert all(abs(p[0]) < 0.01 and abs(p[2] - 30.0) < 0.01 for p in seen)
        w._end_pattern("stopped")
        assert _pump(lambda: w._jog_release is None and not w._gantry.busy, 5)
        assert not w._latched


def test_pattern_refused_when_the_box_does_not_fit():
    """10-26 rig: Z stood at 99.9 mm, the + firmware stop is 100 mm — a circle
    there must be refused with how far to move, before anything is sent."""
    with _Window(builders=_rig(_FactoryLimits)) as s:
        w = s.connect_all()
        _put(w, Axis.Y, 100.0)
        _put(w, Axis.Z, 95.0)
        sim = w._gantry._controller
        n_jogs = sum(1 for e in sim.call_log if e[0] == "jog")
        _app().processEvents()
        w.pattern_combo.setCurrentIndex(w.pattern_combo.findData("circle"))
        w.pattern_size_spin.setValue(30.0)
        w._start_pattern()
        assert w._pattern is None
        assert "move Z down by 25.0 mm" in w.status_label.text(), w.status_label.text()
        assert sum(1 for e in sim.call_log if e[0] == "jog") == n_jogs


def test_pattern_respects_a_taught_home():
    with _Window(builders=_rig(_FactoryLimits)) as s:
        w = s.connect_all()
        _put(w, Axis.Y, 90.0)
        w._teach(Axis.Y, "lo")
        _put(w, Axis.Y, 100.0)
        w.pattern_combo.setCurrentIndex(w.pattern_combo.findData("line_y"))
        w.pattern_size_spin.setValue(40.0)
        w._start_pattern()
        assert w._pattern is None
        assert "− home" in w.status_label.text()


def test_stop_all_ends_a_pattern():
    with _Window() as s:
        w = s.connect_all()
        _put(w, Axis.Y, 100.0)
        _put(w, Axis.Z, 60.0)
        _start(w, "circle", 10.0, 30.0)
        _pump(lambda: False, 0.5)
        w._stop_all()
        assert w._pattern is None and w._latched
        assert _pump(lambda: not w._gantry.busy, 3)
        a = w._gantry.positions_mm()
        _pump(lambda: False, 0.5)
        b = w._gantry.positions_mm()
        assert max(abs(x - y) for x, y in zip(a, b)) < 0.01            # nothing resumes
        w._reset_latch()
        assert _pump(lambda: not w._latched, 3)


def test_a_pattern_whose_link_dies_is_ended_and_stopped():
    class _Flaky(_SimG):
        fail_after = None

        def jog_single_axis(self, axis, position_units, speed_units, acc_units, dec_units,
                            *, relative=True):
            if self.fail_after is not None:
                self.fail_after -= 1
                if self.fail_after < 0:
                    raise RuntimeError("FMC4030_Jog_Single_Axis failed (simulated)")
            super().jog_single_axis(axis, position_units, speed_units, acc_units, dec_units,
                                    relative=relative)

    with _Window(builders=_rig(_Flaky)) as s:
        w = s.connect_all()
        _put(w, Axis.Y, 100.0)
        _put(w, Axis.Z, 60.0)
        w._gantry._controller.fail_after = 6
        _start(w, "circle", 10.0, 30.0)
        assert _pump(lambda: w._pattern is None, 10)
        assert "link error" in (w._log.dir / "events.csv").read_text()
        assert w._latched                                # review M1: not the operator's Stop


class _AbsAsRel(_SimG):
    """A controller that reads an ABSOLUTE jog as RELATIVE — what bring-up step 1
    tests on hardware (KNOWN_ISSUES 10-26 ⓐ)."""
    def jog_single_axis(self, axis, position_units, speed_units, acc_units, dec_units,
                        *, relative=True):
        super().jog_single_axis(axis, position_units, speed_units, acc_units, dec_units,
                                relative=True)


def test_send_that_runs_past_its_range_is_stopped():
    """Re-review N2: a misread absolute Send must be caught while it runs."""
    with _Window("--speed", "20", builders=_rig(_AbsAsRel)) as s:
        w = s.connect_all()
        _put(w, Axis.Y, 100.0)
        _send_nowait = w._target_spins[Axis.Y]
        _send_nowait.setValue(105.0)
        assert _pump(lambda: w.send_btn.isEnabled())
        w._send_targets()                                # runs toward 205 instead
        assert _pump(lambda: w._latched, 5), w.status_label.text()
        assert "left its start..target range" in w.status_label.text()
        assert w._gantry.positions_mm()[1] < 115.0
        assert "send Y" in (w._log.dir / "events.csv").read_text()


def test_normal_send_does_not_trip_the_range_watch():
    with _Window("--speed", "20") as s:
        w = s.connect_all()
        _send(w, Y=30.0)
        assert not w._latched and abs(w._gantry.positions_mm()[1] - 30.0) < 0.2


class _Drift(_SimG):
    """Y jumps by ``kick`` units once armed (a slipped belt / a frame error)."""
    kick = None

    def get_axis_position(self, axis):
        if self.kick is not None and int(axis) == 1:
            self._pos[1] += self.kick
            self._target[1] = self._pos[1]
            self.kick = None
        return super().get_axis_position(axis)


def test_pattern_axis_off_its_box_latches():
    """Re-review: the worker's 3 mm envelope check (or the reach check) ends the
    pattern with an error, which latches like the panel's own 5 mm watch."""
    with _Window(builders=_rig(_Drift)) as s:
        w = s.connect_all()
        _put(w, Axis.Y, 100.0)
        _put(w, Axis.Z, 60.0)
        _start(w, "circle", 10.0, 20.0)
        _pump(lambda: False, 0.6)
        w._gantry._controller.kick = 14.0 / 2.5          # 14 mm: past the 10 mm box + 3
        assert _pump(lambda: w._latched, 5), w.status_label.text()
        assert w._pattern is None
        text = (w._log.dir / "events.csv").read_text()
        assert "outside the pattern envelope" in text or "from its vertex" in text \
            or "left its box" in text


class _NeverStops(_SimG):
    """Accepts a jog, never moves, never reports stopped (hung drive)."""
    def jog_single_axis(self, axis, position_units, speed_units, acc_units, dec_units,
                        *, relative=True):
        self.call_log.append(("jog", axis, position_units, speed_units, relative))

    def is_axis_stopped(self, axis):
        return False


def test_pattern_stall_latches():
    with _Window(builders=_rig(_NeverStops)) as s:
        w = s.connect_all()
        _put(w, Axis.Y, 100.0)
        _put(w, Axis.Z, 60.0)
        _start(w, "circle", 10.0, 20.0)
        w._pattern["stall_s"] = 0.5                       # instead of >= 10 s
        assert _pump(lambda: w._latched, 4), w.status_label.text()
        assert "no progress" in (w._log.dir / "events.csv").read_text()


def test_gantry_disconnect_is_refused_while_a_pattern_runs():
    with _Window() as s:
        w = s.connect_all()
        _put(w, Axis.Y, 100.0)
        _put(w, Axis.Z, 60.0)
        _start(w, "circle", 10.0, 30.0)
        w._toggle_gantry()
        assert w._gantry is not None and w._pattern is not None
        assert "Stop the gantry first" in w.status_label.text()
        w._end_pattern("stopped")
        w._toggle_gantry()                                 # still settling: refused too
        assert w._gantry is not None
        assert _pump(lambda: w._jog_release is None, 5)
        w._stop_all()                                      # latched: a dead link may hang busy
        w._toggle_gantry()
        assert w._gantry is None


def main() -> int:
    if SKIP_REASON:
        print(f"  skip  whisker_vision unavailable: {SKIP_REASON}")
        return 0
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    t0 = time.time()
    for fn in tests:
        t = time.time()
        try:
            fn()
            print(f"  ok    {fn.__name__}  ({time.time() - t:.1f} s)")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed  ({time.time() - t0:.1f} s)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
