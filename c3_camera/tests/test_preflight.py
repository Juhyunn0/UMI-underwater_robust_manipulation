#!/usr/bin/env python3
"""
test_preflight.py — the readiness checks, with no camera and no vehicle attached.

    ./c3 test                                        # runs this with the others
    python -m pytest c3_camera/tests/test_preflight.py -v

A preflight check that is itself wrong is worse than no check: it either blocks a
run that would have worked, or blesses one that will not. Everything fiddly is
covered here — the classification rules (free/owned/undiscovered), the MAVLink2Rest
payload shapes (bitfields that arrive as ints or as {"bits": N}, invalid sentinels,
missing messages), the udpin parse, and the gate's exit codes.

The live network paths (TCP, UDP discovery, HTTP) are exercised against stubs
rather than hardware; what is tested is the decision made from the answer, which
is the part that can be wrong in a way nobody notices.
"""

from __future__ import annotations

import argparse
import io
import json
import socket
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from c3_camera import config as C
from c3_camera import device as D
from c3_camera import preflight as PF


class _FakeFound:
    """The parts of device.Found the checks actually read."""

    def __init__(self, name="192.168.2.191", mxid=C.DEFAULT_MXID,
                 state="X_LINK_FLASH_BOOTED", free=True):
        self.name = name
        self.mxid = mxid
        self.state = state
        self.is_free = free


def _quiet():
    return io.StringIO()


# =============================================================================
# result aggregation
# =============================================================================
def test_report_counts_and_verdict():
    rep = PF.Report(checks=[
        PF.Check("a", PF.OK), PF.Check("b", PF.WARN), PF.Check("c", PF.SKIP)])
    assert rep.ok is True
    assert rep.blocking() == []
    # strict promotes the warning, and only the warning
    assert [c.name for c in rep.blocking(strict=True)] == ["b"]

    rep.checks.append(PF.Check("d", PF.FAIL))
    assert rep.ok is False
    assert [c.name for c in rep.blocking()] == ["d"]
    assert rep.count(PF.OK) == 1 and rep.count(PF.SKIP) == 1


def test_a_check_that_raises_becomes_a_failure_and_the_rest_still_run():
    def boom():
        raise RuntimeError("kaboom")

    rep = PF.run([("bad", boom), ("good", lambda: PF._ok("fine"))],
                 title="t", stream=_quiet())
    assert [c.status for c in rep.checks] == [PF.FAIL, PF.OK]
    assert "kaboom" in rep.checks[0].detail
    assert rep.checks[0].name == "bad"          # label filled in by the runner


def test_report_serialises_for_the_dataset_metadata():
    rep = PF.run([("x", lambda: PF._warn("degraded", "do this", port=1))],
                 title="t", stream=_quiet())
    blob = json.loads(json.dumps(rep.to_dict()))
    assert blob["verdict"] == "ready"           # a warning is not a failure
    assert blob["checks"][0]["fix"] == ["do this"]
    assert blob["checks"][0]["data"] == {"port": 1}


# =============================================================================
# camera classification
# =============================================================================
def test_owned_camera_fails_and_names_the_owner(monkeypatch):
    monkeypatch.setattr(D, "discover", lambda timeout_s=None: [_FakeFound(free=False)])
    monkeypatch.setattr(D, "find_c3", lambda ip=None, mxid=None:
                        _FakeFound(state="X_LINK_BOOTED", free=False))
    monkeypatch.setattr(PF, "local_c3_processes", lambda: [])
    c = PF.check_camera_owner("192.168.2.191", None, 1.0, {"tcp_reachable": True})
    assert c.status == PF.FAIL
    assert "OWNED" in c.detail
    assert any("madrona stop" in f for f in c.fix)


def test_owned_camera_blames_the_local_process_when_there_is_one(monkeypatch):
    monkeypatch.setattr(D, "discover", lambda timeout_s=None: [])
    monkeypatch.setattr(D, "find_c3", lambda ip=None, mxid=None:
                        _FakeFound(state="X_LINK_BOOTED", free=False))
    monkeypatch.setattr(PF, "local_c3_processes",
                        lambda: [(4242, "python -m c3_camera.c3_stream")])
    c = PF.check_camera_owner("192.168.2.191", None, 1.0, {"tcp_reachable": True})
    assert c.status == PF.FAIL
    assert any("pid 4242" in f for f in c.fix)
    # and it must NOT send the operator to disable a vehicle extension that is
    # not the culprit
    assert not any("madrona" in f for f in c.fix)


def test_undiscovered_but_reachable_is_only_a_warning(monkeypatch):
    """Broadcast can be blocked; connecting by --ip needs no discovery."""
    monkeypatch.setattr(D, "discover", lambda timeout_s=None: [])
    monkeypatch.setattr(D, "find_c3", lambda ip=None, mxid=None: None)
    c = PF.check_camera_owner("192.168.2.191", None, 1.0, {"tcp_reachable": True})
    assert c.status == PF.WARN


def test_undiscovered_with_a_dead_link_does_not_block_twice(monkeypatch):
    """One broken link is one blocking item, not two — the link check owns it."""
    monkeypatch.setattr(D, "discover", lambda timeout_s=None: [])
    monkeypatch.setattr(D, "find_c3", lambda ip=None, mxid=None: None)
    c = PF.check_camera_owner("192.168.2.191", None, 1.0, {"tcp_reachable": False})
    assert c.status == PF.WARN and "link failure above" in c.detail

    # but with no verdict from the link check at all, silence is not a pass
    c = PF.check_camera_owner("192.168.2.191", None, 1.0, {})
    assert c.status == PF.FAIL


def test_an_owned_camera_blocks_even_when_the_link_probe_failed(monkeypatch):
    monkeypatch.setattr(D, "discover", lambda timeout_s=None: [])
    monkeypatch.setattr(D, "find_c3", lambda ip=None, mxid=None:
                        _FakeFound(state="X_LINK_BOOTED", free=False))
    monkeypatch.setattr(PF, "local_c3_processes", lambda: [])
    assert PF.check_camera_owner("192.168.2.191", None, 1.0,
                                 {"tcp_reachable": False}).status == PF.FAIL


def test_a_different_camera_is_a_warning_not_a_pass(monkeypatch):
    other = _FakeFound(mxid="DEADBEEFDEADBEEF00")
    monkeypatch.setattr(D, "discover", lambda timeout_s=None: [other])
    monkeypatch.setattr(D, "find_c3", lambda ip=None, mxid=None: other)
    c = PF.check_camera_owner("192.168.2.191", None, 1.0, {"tcp_reachable": True})
    assert c.status == PF.WARN
    assert C.DEFAULT_MXID in c.detail


def test_free_expected_camera_is_ok(monkeypatch):
    found = _FakeFound()
    monkeypatch.setattr(D, "discover", lambda timeout_s=None: [found])
    monkeypatch.setattr(D, "find_c3", lambda ip=None, mxid=None: found)
    c = PF.check_camera_owner("192.168.2.191", None, 1.0, {"tcp_reachable": True})
    assert c.status == PF.OK and c.data["mxid"] == C.DEFAULT_MXID


def test_link_check_publishes_its_verdict_for_the_owner_check(monkeypatch):
    monkeypatch.setattr(D, "tcp_reachable",
                        lambda ip, timeout_s=2.0: (False, "TCP timed out"))
    state: dict = {}
    c = PF.check_camera_link("192.168.2.191", 1.0, state)
    assert c.status == PF.FAIL and state["tcp_reachable"] is False


def test_local_process_scan_excludes_this_process():
    """Whatever it finds, preflight must never report itself as an owner."""
    import os
    assert all(pid != os.getpid() for pid, _ in PF.local_c3_processes())


# =============================================================================
# vehicle
# =============================================================================
def test_vehicle_summary_reads_a_realistic_rest_payload():
    messages = {
        "HEARTBEAT": {"message": {"base_mode": {"bits": 209}, "custom_mode": 19},
                      "status": {"time": {"counter": 100}}},
        "SYS_STATUS": {"message": {"voltage_battery": 15800,
                                   "battery_remaining": 72}},
        "ATTITUDE": {"message": {"roll": 0.0, "pitch": 0.0, "yaw": 1.0}},
        "SCALED_PRESSURE2": {"message": {"press_abs": 1013.9}},
    }
    detail, facts = PF.summarize_vehicle(messages)
    assert facts["armed"] is True        # 209 & 128
    assert facts["battery_v"] == 15.8 and facts["battery_pct"] == 72
    assert facts["ext_press_hpa"] == 1013.9
    assert "ARMED" in detail


def test_vehicle_summary_accepts_a_plain_int_bitfield():
    detail, facts = PF.summarize_vehicle(
        {"HEARTBEAT": {"message": {"base_mode": 81, "custom_mode": 0}}})
    assert facts["armed"] is False       # 81 & 128 == 0
    assert "disarmed" in detail


def test_vehicle_summary_never_invents_state_it_does_not_have():
    detail, facts = PF.summarize_vehicle({})
    assert facts == {} and "no HEARTBEAT" in detail

    # invalid sentinels must not become readings
    _, facts = PF.summarize_vehicle(
        {"SYS_STATUS": {"message": {"voltage_battery": 65535,
                                    "battery_remaining": -1}}})
    assert "battery_v" not in facts and "battery_pct" not in facts

    # an unparseable bitfield is reported as unknown, not guessed as disarmed
    detail, facts = PF.summarize_vehicle(
        {"HEARTBEAT": {"message": {"base_mode": "MAV_MODE_FLAG_SAFETY_ARMED"}}})
    assert "armed" not in facts and "unknown" in detail


def test_missing_endpoint_warns_with_the_command_that_adds_it(monkeypatch):
    from c3_camera import blueos_endpoint as BE
    monkeypatch.setattr(BE, "list_endpoints", lambda host: [
        {"name": "GCS Client Link", "connection_type": "udpout",
         "place": "192.168.2.1", "argument": 14550, "enabled": True}])
    c = PF.check_mavlink_endpoint(14551, "192.168.2.2", "192.168.2.1", 1.0)
    assert c.status == PF.WARN
    assert any("--port 14551 --yes" in f for f in c.fix)


def test_present_endpoint_is_ok_even_when_the_port_arrives_as_a_string(monkeypatch):
    from c3_camera import blueos_endpoint as BE
    monkeypatch.setattr(BE, "list_endpoints", lambda host: [
        {"connection_type": "udpout", "place": "192.168.2.1",
         "argument": "14551", "enabled": True, "persistent": True}])
    assert PF.check_mavlink_endpoint(14551, "h", "192.168.2.1", 1.0).status == PF.OK


def test_disabled_endpoint_is_not_a_pass(monkeypatch):
    from c3_camera import blueos_endpoint as BE
    monkeypatch.setattr(BE, "list_endpoints", lambda host: [
        {"connection_type": "udpout", "place": "192.168.2.1",
         "argument": 14551, "enabled": False}])
    assert PF.check_mavlink_endpoint(14551, "h", "192.168.2.1", 1.0).status == PF.WARN


def test_parse_udpin():
    assert PF.parse_udpin("udpin:0.0.0.0:14551") == ("0.0.0.0", 14551)
    assert PF.parse_udpin("udp:192.168.2.1:14550") == ("192.168.2.1", 14550)
    for bad in ("udpout:192.168.2.1:14550", "/dev/ttyACM0",
                "tcp:1.2.3.4:5760", "udpin:0.0.0.0:notaport", "udpin:14551"):
        assert PF.parse_udpin(bad) is None, bad


def test_udp_listen_receives_a_real_datagram():
    """Bind, send to ourselves, and prove the check sees it."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()                       # hand the port back before the check binds

    import threading
    sender = threading.Timer(
        0.2, lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        .sendto(b"\xfd\x00", ("127.0.0.1", port)))
    sender.start()
    try:
        c = PF.check_mavlink_udp(f"udpin:127.0.0.1:{port}", wait_s=3.0)
    finally:
        sender.cancel()
    assert c.status == PF.OK and c.data["port"] == port


def test_udp_listen_warns_when_nothing_arrives():
    c = PF.check_mavlink_udp("udpin:127.0.0.1:0", wait_s=0.3)
    # port 0 binds to an ephemeral port nothing sends to
    assert c.status == PF.WARN
    assert any("--mavlink-transport rest" in f for f in c.fix)


def test_udp_listen_reports_a_port_someone_else_holds():
    held = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    held.bind(("127.0.0.1", 0))
    port = held.getsockname()[1]
    try:
        c = PF.check_mavlink_udp(f"udpin:127.0.0.1:{port}", wait_s=0.2)
    finally:
        held.close()
    assert c.status == PF.WARN
    assert "not bindable" in c.detail


def test_a_dark_blueos_costs_one_timeout_not_four(monkeypatch):
    """When the vehicle is off, the dependent checks must not each re-time-out."""
    calls = {"tcp": 0, "http": 0}

    def fake_tcp(ip, port=None, timeout_s=2.0):
        calls["tcp"] += 1
        return False, "TCP timed out"

    def fake_http(url, timeout_s):
        calls["http"] += 1
        raise AssertionError("must not be reached once BlueOS is known dark")

    monkeypatch.setattr(D, "tcp_reachable", fake_tcp)
    monkeypatch.setattr(PF, "_http_json", fake_http)

    specs = PF.vehicle_checks(transport="udp",
                              connection="udpin:127.0.0.1:0",
                              rest_url="http://192.168.2.2:6040",
                              timeout_s=1.0)
    results = [fn() for _, fn in specs]
    assert [c.status for c in results] == [PF.WARN] * 4
    assert calls == {"tcp": 1, "http": 0}
    # the port is still bound, so a conflict would still be caught
    assert "free to bind" in results[3].detail


def test_blueos_reachable_but_silent_is_distinguished_from_absent(monkeypatch):
    monkeypatch.setattr(D, "tcp_reachable",
                        lambda ip, port=None, timeout_s=2.0: (True, "open"))
    monkeypatch.setattr(PF, "vehicle_info",
                        lambda host, t: {"ardusub_version": None,
                                         "vehicle_type": None})
    state: dict = {}
    c = PF.check_blueos("192.168.2.2", 1.0, state)
    assert c.status == PF.WARN and "starting up" in c.detail
    assert state["blueos_up"] is True      # so the later checks still try


def test_transport_none_skips_every_vehicle_check():
    specs = PF.vehicle_checks(transport="none", connection="", rest_url="")
    assert [fn().status for _, fn in specs] == [PF.SKIP] * 3


def test_rest_transport_checks_no_udp_port():
    labels = [lbl for lbl, _ in PF.vehicle_checks(
        transport="rest", connection="udpin:0.0.0.0:14551", rest_url="http://h")]
    assert "MAVLink endpoint" not in labels


# =============================================================================
# storage and display
# =============================================================================
def test_storage_checks_the_nearest_existing_ancestor():
    with tempfile.TemporaryDirectory() as td:
        deep = Path(td) / "not" / "created" / "yet"
        assert PF.check_storage(deep, 0.0).status == PF.OK
        assert str(Path(td).resolve()) in PF.check_storage(deep, 0.0).detail


def test_storage_fails_below_the_floor():
    with tempfile.TemporaryDirectory() as td:
        c = PF.check_storage(Path(td), 10_000_000.0)      # 10 PB
        assert c.status == PF.FAIL and "floor" in c.detail


def test_storage_warns_within_three_times_the_floor():
    import shutil as _sh
    with tempfile.TemporaryDirectory() as td:
        free_gb = _sh.disk_usage(td).free / 1e9
        assert PF.check_storage(Path(td), free_gb * 0.5).status == PF.WARN
        assert PF.check_storage(Path(td), free_gb * 0.1).status == PF.OK


def test_no_output_directory_is_a_skip_not_a_pass():
    assert PF.check_storage(None, 10.0).status == PF.SKIP


def test_display_check(monkeypatch):
    assert PF.check_display(want_window=False).status == PF.SKIP
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    c = PF.check_display(want_window=True)
    assert c.status == PF.WARN and any("--no-display" in f for f in c.fix)


# =============================================================================
# the gate
# =============================================================================
def _ns(**kw) -> argparse.Namespace:
    base = dict(no_preflight=False, preflight_only=False, preflight_strict=False,
                preflight_timeout=1.0, preflight_json=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_gate_lets_a_clean_run_through(monkeypatch):
    monkeypatch.setattr(PF, "run", lambda specs, title, **kw: PF.Report(
        checks=[PF.Check("x", PF.OK)]))
    assert PF.gate(_ns(), title="t", specs=[]) is None


def test_gate_blocks_on_a_failure(monkeypatch):
    monkeypatch.setattr(PF, "run", lambda specs, title, **kw: PF.Report(
        checks=[PF.Check("x", PF.FAIL)]))
    assert PF.gate(_ns(), title="t", specs=[]) == PF.EXIT_PREFLIGHT


def test_gate_lets_a_warning_through_unless_strict(monkeypatch):
    monkeypatch.setattr(PF, "run", lambda specs, title, **kw: PF.Report(
        checks=[PF.Check("x", PF.WARN)]))
    assert PF.gate(_ns(), title="t", specs=[]) is None
    assert PF.gate(_ns(preflight_strict=True), title="t",
                   specs=[]) == PF.EXIT_PREFLIGHT


def test_preflight_only_exits_with_the_verdict(monkeypatch):
    monkeypatch.setattr(PF, "run", lambda specs, title, **kw: PF.Report(
        checks=[PF.Check("x", PF.OK)]))
    assert PF.gate(_ns(preflight_only=True), title="t", specs=[]) == 0
    monkeypatch.setattr(PF, "run", lambda specs, title, **kw: PF.Report(
        checks=[PF.Check("x", PF.FAIL)]))
    assert PF.gate(_ns(preflight_only=True), title="t",
                   specs=[]) == PF.EXIT_PREFLIGHT


def test_no_preflight_runs_nothing_at_all():
    ran = []
    assert PF.gate(_ns(no_preflight=True), title="t",
                   specs=[("x", lambda: ran.append(1) or PF._ok(""))]) is None
    assert ran == []


def test_gate_stashes_the_report_for_the_dataset_metadata(monkeypatch):
    monkeypatch.setattr(PF, "run", lambda specs, title, **kw: PF.Report(
        checks=[PF.Check("x", PF.OK)], title=title))
    a = _ns()
    PF.gate(a, title="c3 collect", specs=[])
    assert a.preflight_report.to_dict()["title"] == "c3 collect"


def test_standard_checks_skips_vehicle_for_a_tool_without_mavlink():
    labels = [lbl for lbl, _ in PF.standard_checks(_ns(ip="192.168.2.191",
                                                       mxid=None))]
    assert "ROV state" not in labels and "camera owner" in labels


def test_standard_checks_includes_vehicle_when_the_tool_has_a_transport():
    labels = [lbl for lbl, _ in PF.standard_checks(
        _ns(ip="192.168.2.191", mxid=None, mavlink_transport="udp",
            mavlink="udpin:0.0.0.0:14551", mavlink_rest_url="http://h"))]
    assert {"BlueOS", "ROV state", "MAVLink endpoint"} <= set(labels)


# =============================================================================
# what the dataset records
# =============================================================================
def test_metadata_txt_surfaces_the_checks_that_were_not_clean():
    from c3_camera.dataset import _render_preflight

    rep = PF.run([("telemetry path", lambda: PF._warn("nothing arrived", "fix me")),
                  ("disk", lambda: PF._ok("plenty"))],
                 title="t", stream=_quiet())
    text = "\n".join(_render_preflight(rep.to_dict()))
    assert "telemetry path" in text and "nothing arrived" in text
    assert "disk" not in text                # a clean check is not worth a line


def test_metadata_txt_says_so_when_a_clean_rig_or_no_preflight():
    from c3_camera.dataset import _render_preflight

    clean = PF.run([("disk", lambda: PF._ok("plenty"))], title="t", stream=_quiet())
    assert "all checks passed" in "\n".join(_render_preflight(clean.to_dict()))
    assert "not run" in "\n".join(_render_preflight({"ran": False}))


def test_metadata_txt_never_dies_on_an_unrecognised_report():
    """A dataset must still get its metadata if the report shape ever changes."""
    from c3_camera.dataset import _render_preflight

    for junk in (None, "a string", [], {"checks": "not a list"},
                 {"checks": [{"status": "warn"}]}, {"checks": [None, 7]}):
        _render_preflight(junk)             # must not raise


# =============================================================================
# wiring — every hardware entry point must accept the flags
# =============================================================================
def test_every_hardware_tool_accepts_the_preflight_flags():
    """A tool that forgot PF.add_preflight_args would fail here, not on the boat."""
    from c3_camera import (c3_bench, c3_collect, c3_depth_accuracy,
                           c3_encode_quality, c3_option_sweep, c3_stream)

    cases = [
        ("c3_collect", c3_collect.build_parser()),
        ("c3_encode_quality", c3_encode_quality.build_parser()),
    ]
    for name, parser in cases:
        dests = {act.dest for act in parser._actions}
        assert {"no_preflight", "preflight_only", "preflight_strict",
                "preflight_timeout"} <= dests, name

    # The rest build their parser inside main(), so probe them through --help,
    # which argparse renders from the same actions.
    import contextlib
    for mod in (c3_bench, c3_stream, c3_option_sweep, c3_depth_accuracy):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                (mod.parse_args if hasattr(mod, "parse_args") else mod.main)(["--help"])
            except SystemExit:
                pass
        assert "--no-preflight" in buf.getvalue(), mod.__name__


# =============================================================================
# bare runner, so this works without pytest installed
# =============================================================================
class _MonkeyPatch:
    """The two pytest fixtures used above, in twenty lines."""

    def __init__(self):
        self._undo = []

    def setattr(self, obj, name, value):
        self._undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def delenv(self, name, raising=True):
        import os
        if name in os.environ:
            old = os.environ[name]
            self._undo.append((os.environ, name, old))
            del os.environ[name]
        elif raising:
            raise KeyError(name)

    def undo(self):
        import os
        for obj, name, old in reversed(self._undo):
            if obj is os.environ:
                os.environ[name] = old
            else:
                setattr(obj, name, old)
        self._undo.clear()


def _main() -> int:
    import inspect
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        mp = _MonkeyPatch()
        try:
            if "monkeypatch" in inspect.signature(fn).parameters:
                fn(mp)
            else:
                fn()
            print(f"  PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        finally:
            mp.undo()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
