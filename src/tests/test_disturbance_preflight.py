#!/usr/bin/env python3
"""
test_disturbance_preflight.py — the readiness checks with injected `ip` / `ss`
answers; nothing here touches the network or a device.

    ~/miniforge3/envs/robust/bin/python src/tests/test_disturbance_preflight.py
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import disturbance_preflight as PF  # noqa: E402


def test_gantry_client_probe_outcomes():
    assert PF.check_gantry_client("1.2.3.4", 8088, sim=True).status == PF.SKIP
    held = PF.check_gantry_client("1.2.3.4", 8088, sim=False,
                                  probe=lambda ip, port: (True, "ESTAB ... 1.2.3.4:8088"))
    assert held.status == PF.FAIL and "ONE client" in held.detail
    free = PF.check_gantry_client("1.2.3.4", 8088, sim=False,
                                  probe=lambda ip, port: (False, "nothing"))
    assert free.status == PF.OK
    unknown = PF.check_gantry_client("1.2.3.4", 8088, sim=False,
                                     probe=lambda ip, port: (None, "no ss"))
    assert unknown.status == PF.WARN


def test_gantry_route_wifi_warns_wired_ok_none_fails():
    wifi = PF.check_gantry_route("1.2.3.4", False,
                                 route=[{"dev": "wlp13s0", "prefsrc": "192.168.0.180"}])
    assert wifi.status == PF.WARN and "WiFi" in wifi.detail
    wired = PF.check_gantry_route("1.2.3.4", False,
                                  route=[{"dev": "enxa0cec874d345", "prefsrc": "192.168.0.35"}])
    assert wired.status == PF.OK
    assert PF.check_gantry_route("1.2.3.4", False, route=[]).status == PF.FAIL
    assert PF.check_gantry_route("1.2.3.4", True).status == PF.SKIP


def _can(flags, state, bitrate):
    return [{"ifname": "can0", "flags": flags,
             "linkinfo": {"info_kind": "can",
                          "info_data": {"state": state,
                                        "bittiming": {"bitrate": bitrate}}}}]


def test_can_bus_states():
    assert PF.check_can("can0", True).status == PF.SKIP
    assert PF.check_can("can0", False, link=[]).status == PF.FAIL
    down_rule = PF.check_can("can0", False, link=_can(["NOARP"], "", None), sudo_rule=True)
    assert down_rule.status == PF.WARN and "DOWN" in down_rule.detail
    down_norule = PF.check_can("can0", False, link=_can(["NOARP"], "", None), sudo_rule=False)
    assert down_norule.status == PF.FAIL
    assert PF.check_can("can0", False, link=_can(["UP"], "ERROR-ACTIVE", 500000),
                        sudo_rule=True).status == PF.FAIL
    assert PF.check_can("can0", False, link=_can(["UP"], "BUS-OFF", 1000000),
                        sudo_rule=True).status == PF.FAIL
    assert PF.check_can("can0", False, link=_can(["UP"], "ERROR-PASSIVE", 1000000),
                        sudo_rule=True).status == PF.WARN
    ok = PF.check_can("can0", False, link=_can(["UP"], "ERROR-ACTIVE", 1000000),
                      sudo_rule=True)
    assert ok.status == PF.OK and "1000 kbps" in ok.detail


def test_thruster_port_and_data_root():
    assert PF.check_thruster_port("/dev/nope", True).status == PF.SKIP
    assert PF.check_thruster_port("/dev/definitely/not/here", False).status == PF.FAIL
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "ttyFAKE"
        p.write_text("")
        assert PF.check_thruster_port(str(p), False).status == PF.OK
        assert PF.check_data_root(Path(td) / "data").status == PF.OK
        assert (Path(td) / "data").is_dir()


def test_ld_library_path_and_display_env():
    saved = {k: os.environ.get(k) for k in
             ("LD_LIBRARY_PATH", "DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM")}
    try:
        os.environ["LD_LIBRARY_PATH"] = "/opt/openfoam12/lib:/usr/lib"
        assert PF.check_ld_library_path().status == PF.WARN
        os.environ["LD_LIBRARY_PATH"] = "/opt/other/lib"
        assert PF.check_ld_library_path().status == PF.OK
        for k in ("DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM"):
            os.environ.pop(k, None)
        assert PF.check_display().status == PF.FAIL
        os.environ["DISPLAY"] = ":0"
        assert PF.check_display().status == PF.OK
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        assert PF.check_display().status == PF.WARN
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_interpreter_reports_this_python():
    c = PF.check_interpreter(sim=True)
    assert c.status in (PF.OK, PF.FAIL)
    assert sys.executable in c.detail


def _device_specs(gantry, motor, thruster, display=PF.OK):
    mk = {PF.OK: lambda: PF._ok("fine"), PF.WARN: lambda: PF._warn("meh", "x"),
          PF.FAIL: lambda: PF._fail("missing", "plug it"), PF.SKIP: lambda: PF._skip("sim")}
    return [("interpreter", mk[PF.OK]),
            ("gantry client", mk[PF.OK] if gantry != PF.SKIP else mk[PF.SKIP]),
            ("gantry route", mk[gantry]), ("can bus", mk[motor]),
            ("thruster port", mk[thruster]), ("display", mk[display])]


def _gate(specs):
    a = argparse.Namespace(no_preflight=False, preflight_only=False, preflight_strict=False,
                           preflight_json=None)
    stream = io.StringIO()
    return PF.gate(a, title="t", specs=specs, stream=stream), a.preflight_report, \
        stream.getvalue()


def test_one_device_is_enough_to_open_the_panel():
    """2026-10-27 user request: open the UI unless ALL three devices fail."""
    rc, report, out = _gate(_device_specs(PF.FAIL, PF.FAIL, PF.OK))       # thruster only
    assert rc is None, out
    assert report.devices == {"available": ["thruster"], "unavailable": ["gantry", "motor"],
                              "relaxed": True}
    route = next(c for c in report.checks if c.name == "gantry route")
    assert route.status == PF.WARN and "Connect gantry" in route.detail
    assert "UNAVAILABLE" in out and "READY" in out and "NOT READY" not in out
    assert report.to_dict()["devices"]["unavailable"] == ["gantry", "motor"]
    rc, report, _ = _gate(_device_specs(PF.OK, PF.WARN, PF.FAIL))         # can0 DOWN = WARN
    assert rc is None and report.devices["available"] == ["gantry", "motor"]


def test_all_three_devices_down_still_blocks():
    rc, report, out = _gate(_device_specs(PF.FAIL, PF.FAIL, PF.FAIL))
    assert rc == PF.EXIT_PREFLIGHT
    assert not report.devices["relaxed"] and "ALL unavailable" in out
    assert all(c.status == PF.FAIL for c in report.checks if c.name in
               ("gantry route", "can bus", "thruster port"))


def test_a_non_device_failure_still_blocks():
    rc, _, out = _gate(_device_specs(PF.OK, PF.FAIL, PF.OK, display=PF.FAIL))
    assert rc == PF.EXIT_PREFLIGHT and "blocking: display" in out


def test_sim_skips_are_not_devices():
    rc, report, _ = _gate(_device_specs(PF.SKIP, PF.SKIP, PF.SKIP))
    assert rc is None and report.devices == {"available": [], "unavailable": [],
                                             "relaxed": False}


def test_run_and_gate_contract():
    def boom():
        raise RuntimeError("kaboom")
    specs = [("good", lambda: PF._ok("fine")), ("bad", boom),
             ("soft", lambda: PF._warn("meh", "do this"))]
    stream = io.StringIO()
    report = PF.run(specs, title="t", stream=stream)
    assert [c.status for c in report.checks] == [PF.OK, PF.FAIL, PF.WARN]
    assert "kaboom" in report.checks[1].detail
    assert "NOT READY" in stream.getvalue() and "blocking: bad" in stream.getvalue()
    d = report.to_dict()
    assert d["verdict"] == "not_ready" and d["counts"]["fail"] == 1

    ap = argparse.ArgumentParser()
    PF.add_preflight_args(ap)
    with tempfile.TemporaryDirectory() as td:
        js = Path(td) / "pf.json"
        a = ap.parse_args(["--preflight-json", str(js)])
        rc = PF.gate(a, title="t", specs=specs, stream=io.StringIO())
        assert rc == PF.EXIT_PREFLIGHT and js.is_file()
        a = ap.parse_args(["--no-preflight"])
        assert PF.gate(a, title="t", specs=specs, stream=io.StringIO()) is None and a.preflight_report is None
        a = ap.parse_args(["--preflight-only"])
        assert PF.gate(a, title="t", specs=specs, stream=io.StringIO()) == PF.EXIT_PREFLIGHT
        a = ap.parse_args(["--preflight-only"])
        assert PF.gate(a, title="t", specs=specs[:1], stream=io.StringIO()) == 0
        a = ap.parse_args(["--preflight-strict"])
        assert PF.gate(a, title="t", specs=[specs[2]], stream=io.StringIO()) == PF.EXIT_PREFLIGHT


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
