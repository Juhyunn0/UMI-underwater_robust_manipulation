#!/usr/bin/env python3
"""test_station_bridge.py — the tag-dropout ladder for a STATION hold.

Two halves. The LADDER tests are pure logic (no Qt, no hardware, no acados)
and pin the tier machine and the fault whitelist. The WORKER tests drive the
real :class:`MpcWorker` with a stub controller — the same harness
test_control.py uses — and pin the thing that actually matters: a tag dropout
during a station hold must no longer disengage, and every other interlock
must still fire.

    QT_QPA_PLATFORM=offscreen \\
      ~/miniforge3/envs/robust/bin/python rov_gui/tests/test_station_bridge.py
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rov_gui.control import station_bridge as SB       # noqa: E402


# ------------------------------------------------------------------- ladder
def test_the_tiers_go_none_imu_coast_and_come_straight_back():
    br = SB.StationBridge({"imu_hold_s": 1.0})
    assert br.tier == SB.TIER_NONE and not br.active
    assert br.note_lost(0.05) == SB.TIER_IMU
    for _ in range(18):                       # 0.95 s total
        assert br.note_lost(0.05) == SB.TIER_IMU
    assert br.note_lost(0.05) == SB.TIER_IMU  # exactly 1.00 s is still IMU
    assert br.note_lost(0.05) == SB.TIER_COAST
    assert br.active and br.elapsed > 1.0
    rec = br.note_fix(0.0)                    # ...and one fix ends it at once
    assert rec["tier"] == SB.TIER_COAST and rec["elapsed"] > 1.0
    assert br.tier == SB.TIER_NONE and not br.active
    assert br.n_entered == 1 and br.n_coast == 1


def test_a_fix_with_no_dropout_reports_nothing():
    br = SB.StationBridge()
    assert br.note_fix(0.0) is None
    assert br.n_entered == 0


def test_only_tag_faults_may_be_bridged():
    """The whitelist is the safety property. Bridging `imu stale` or
    `pressure depth stale` would mean flying on the very sensor that just
    died, and `no vehicle imu` means the autopilot stopped talking."""
    for ok in ("no tag fix", "tag fix stale (0.83s)"):
        assert SB.is_bridgeable(ok), ok
    for bad in ("imu stale (0.41s)", "pressure depth stale", "no vehicle imu",
                "state lost", "", None,
                "tag fix stale-ish but a different fault"):
        assert not SB.is_bridgeable(bad), bad


def test_coast_releases_surge_sway_and_yaw_and_keeps_heave():
    br = SB.StationBridge()
    u = br.release_horizontal([11.0, -7.0, -6.5, 1.0, 2.0, 3.0])
    assert list(u) == [0.0, 0.0, -6.5, 1.0, 2.0, 0.0]


def test_a_typo_in_the_block_raises():
    for bad in ({"enable": True}, {"imu_hold": 3.0}, {"xy": "imu"}):
        try:
            SB.resolve(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} was accepted")
    for bad in ({"imu_hold_s": -1.0}, {"xy_source": "gps"}):
        try:
            SB.resolve(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} was accepted")


def test_the_shipped_config_arms_the_ladder():
    from rov_gui.control.geometry import MpcConfig

    cfg = MpcConfig.load(str(ROOT / "config" / "hw_mpc.yaml"))
    r = SB.resolve(cfg.station_bridge)
    assert r["enabled"] is True
    assert 0.0 < r["imu_hold_s"] <= 10.0, r
    assert r["xy_source"] == "auto"


# ------------------------------------------------------------------- worker
def _harness():
    """The same stub-controller worker test_control.py uses."""
    import rov_gui.tests.test_control as TC
    return TC


def test_a_tag_dropout_during_a_station_hold_no_longer_disengages():
    """THE POINT OF THE FEATURE. Before 2026-08-18 this disengaged after
    tag_stale_s + tag_stale_hold_s (~1.1 s), which also dropped depth hold on
    a negatively buoyant vehicle."""
    TC = _harness()
    tmp = tempfile.mkdtemp(prefix="bridge_")
    w, _bus, _pilots, logs = TC._worker(tmp)
    w.cfg.engage["settle_s"] = 0.0
    TC._feed_good_state(w)
    w.on_enable(True)
    w.set_engaged(True)
    assert w.engaged, w.reason
    w._warmup_left = 0
    w.set_scenario({"shape": "station", "origin_tag": None,
                    "origin": "current", "yaw_map_deg": 0.0})
    w.set_traj(True)
    assert w.station is not None, w.reason
    assert w._bridge is not None and w._bridge.enabled

    for _ in range(3):                        # let it take an anchor
        w.tick()
    assert w.engaged and w._bridge_anchor is not None
    # ...now freeze the tag fix in the past: the assembler refuses it
    w.fix.t_capture -= 5.0
    for _ in range(60):                       # 3 s of ticks
        w.tick()
        if not w.engaged:
            break
    assert w.engaged, (
        f"disengaged on a tag dropout: {w.reason} "
        f"(tier {w._bridge.tier}, {w._bridge.elapsed:.2f}s)")
    assert w._bridge.active, w._bridge.tier
    w.teardown()


def test_the_coast_tier_sends_heave_and_nothing_else():
    """End to end, at the wire: after imu_hold_s the axes that actually leave
    the station must be surge=sway=yaw=0 with heave still live. Zeroing the
    wrench is not enough on its own — the allocation, the cap and the slew
    limit all sit between it and the vehicle."""
    TC = _harness()
    tmp = tempfile.mkdtemp(prefix="bridge_coast_")
    w, _bus, pilots, _logs = TC._worker(tmp)
    w.cfg.engage["settle_s"] = 0.0
    w.cfg.station_bridge = {"imu_hold_s": 0.2}      # reach coast quickly
    w._setup_station_bridge()
    TC._feed_good_state(w)
    w.on_enable(True)
    w.set_engaged(True)
    w._warmup_left = 0
    w.set_scenario({"shape": "station", "origin_tag": None,
                    "origin": "current", "yaw_map_deg": 0.0})
    w.set_traj(True)
    for _ in range(3):
        w.tick()
    w.fix.t_capture -= 5.0
    for _ in range(40):                             # 2 s: well past 0.2 s
        w.tick()
    assert w.engaged, w.reason
    assert w._bridge.tier == SB.TIER_COAST, w._bridge.tier
    a = pilots[-1]
    assert a.surge == 0.0 and a.sway == 0.0 and a.yaw == 0.0, a
    # ...and the run says so, in the CSV and on the panel
    assert "COAST" in w._bridge.detail().upper(), w._bridge.detail()
    w.teardown()


def test_recovery_reports_how_far_the_bridge_had_drifted():
    """The free measurement. Every dropout that ends produces the one number
    that turns the [예측] IMU budget into a measured one, so it has to reach
    the mission log rather than being discarded with the bridge state."""
    TC = _harness()
    tmp = tempfile.mkdtemp(prefix="bridge_rec_")
    w, _bus, _pilots, logs = TC._worker(tmp)
    w.cfg.engage["settle_s"] = 0.0
    TC._feed_good_state(w)
    w.on_enable(True)
    w.set_engaged(True)
    w._warmup_left = 0
    w.set_scenario({"shape": "station", "origin_tag": None,
                    "origin": "current", "yaw_map_deg": 0.0})
    w.set_traj(True)
    for _ in range(3):
        w.tick()
    w.fix.t_capture -= 5.0
    for _ in range(20):
        w.tick()
    assert w._bridge.active
    w.fix.t_capture = TC.now()                      # the tag comes back
    w.tick()
    assert not w._bridge.active, w._bridge.tier
    rec = w._bridge.last_recover
    assert rec is not None and rec["elapsed"] > 0.5, rec
    assert any("fix back after" in m for _l, m in logs), logs[-3:]
    w.teardown()


def test_the_other_interlocks_still_fire_while_bridging():
    """A ladder that swallowed the disarm interlock would be a much worse bug
    than the one it fixes."""
    TC = _harness()
    tmp = tempfile.mkdtemp(prefix="bridge_i_")
    w, _bus, _pilots, _logs = TC._worker(tmp)
    w.cfg.engage["settle_s"] = 0.0
    TC._feed_good_state(w)
    w.on_enable(True)
    w.set_engaged(True)
    w._warmup_left = 0
    w.set_scenario({"shape": "station", "origin_tag": None,
                    "origin": "current", "yaw_map_deg": 0.0})
    w.set_traj(True)
    for _ in range(3):
        w.tick()
    w.fix.t_capture -= 5.0
    for _ in range(10):
        w.tick()
    assert w.engaged, w.reason
    w.tel.armed = False                       # the vehicle disarmed itself
    for _ in range(10):
        w.tick()
    assert not w.engaged, "a disarm during a bridge must still disengage"
    w.teardown()


def test_a_dead_imu_is_not_bridged():
    """`imu stale` / `pressure depth stale` are exactly the sensors the bridge
    would fly ON, so they are not on the whitelist and must still disengage."""
    TC = _harness()
    tmp = tempfile.mkdtemp(prefix="bridge_dead_")
    w, _bus, _pilots, _logs = TC._worker(tmp)
    w.cfg.engage["settle_s"] = 0.0
    TC._feed_good_state(w)
    w.on_enable(True)
    w.set_engaged(True)
    w._warmup_left = 0
    w.set_scenario({"shape": "station", "origin_tag": None,
                    "origin": "current", "yaw_map_deg": 0.0})
    w.set_traj(True)
    # the autopilot goes quiet: ATTITUDE stops, so the assembler says
    # "imu stale" rather than "tag fix stale"
    w.imu.t_att -= 5.0
    w.imu.stamp -= 5.0
    for _ in range(30):
        w.tick()
        if not w.engaged:
            break
    assert not w.engaged, "a dead autopilot IMU must still disengage"
    assert "imu" in w.reason, w.reason
    w.teardown()


def test_the_bridge_is_station_only():
    """A path mission has a moving reference and a real velocity; the ladder
    assumes a vehicle nominally at rest over one point."""
    TC = _harness()
    tmp = tempfile.mkdtemp(prefix="bridge_path_")
    w, _bus, _pilots, _logs = TC._worker(tmp)
    w.cfg.engage["settle_s"] = 0.0
    TC._feed_good_state(w)
    w.on_enable(True)
    w.set_engaged(True)
    w._warmup_left = 0
    w.set_scenario({"shape": "square", "origin_tag": None,
                    "origin": "current", "size": 1.0, "size_y": 1.0,
                    "speed": 0.1, "yaw_map_deg": 0.0})
    w.set_traj(True)
    assert w.station is None, "this test needs a PATH mission"
    for _ in range(3):
        w.tick()
    w.fix.t_capture -= 5.0
    for _ in range(60):
        w.tick()
        if not w.engaged:
            break
    assert not w.engaged, "a square must still disengage on a tag dropout"
    w.teardown()


def test_disabling_the_ladder_restores_the_old_behaviour():
    TC = _harness()
    tmp = tempfile.mkdtemp(prefix="bridge_off_")
    w, _bus, _pilots, _logs = TC._worker(tmp)
    w.cfg.station_bridge = {"enabled": False}
    w._setup_station_bridge()
    w.cfg.engage["settle_s"] = 0.0
    TC._feed_good_state(w)
    w.on_enable(True)
    w.set_engaged(True)
    w._warmup_left = 0
    w.set_scenario({"shape": "station", "origin_tag": None,
                    "origin": "current", "yaw_map_deg": 0.0})
    w.set_traj(True)
    for _ in range(3):
        w.tick()
    w.fix.t_capture -= 5.0
    for _ in range(60):
        w.tick()
        if not w.engaged:
            break
    assert not w.engaged, "with the ladder off, a dropout must disengage"
    w.teardown()


def test_the_run_meta_records_the_ladder_and_its_dropouts():
    TC = _harness()
    tmp = tempfile.mkdtemp(prefix="bridge_meta_")
    w, _bus, _pilots, _logs = TC._worker(tmp)
    w.cfg.engage["settle_s"] = 0.0
    TC._feed_good_state(w)
    w.on_enable(True)
    w.set_engaged(True)
    w._warmup_left = 0
    w.set_scenario({"shape": "station", "origin_tag": None,
                    "origin": "current", "yaw_map_deg": 0.0})
    w.set_traj(True)
    for _ in range(3):
        w.tick()
    w.fix.t_capture -= 5.0
    for _ in range(40):
        w.tick()
    m = w._run_meta("test")
    assert m["schema_version"] >= 5, m["schema_version"]
    sb = m["station_bridge"]
    assert sb["enabled"] is True
    assert sb["dropouts"] >= 1, sb
    assert sb["worst_dropout_s"] > 0.0, sb
    # The wording gained a clause on 2026-08-21 when the object-follow
    # mission started demoting INTO this ladder rather than widening it.
    assert sb["scope"].startswith("station hold only"), sb["scope"]
    w.teardown()


def test_the_csv_carries_the_bridge_columns():
    from rov_gui.control.workers import CSV_HEADER

    cols = CSV_HEADER.strip().split(",")
    # By NAME, never by position: the ten obj_*/follow_* columns were appended
    # after these on 2026-08-21, which is exactly the append rule working.
    assert "bridge_s" in cols and "bridge_tier" in cols, cols[-6:]
    assert cols.index("bridge_tier") == cols.index("bridge_s") + 1
    assert cols.index("roll_deg") == cols.index("bridge_s") - 1


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
