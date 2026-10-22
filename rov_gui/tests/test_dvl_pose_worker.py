"""--dvl_pose in the MpcWorker (control/dvl_pose.py + workers.py) — the
worker's own slots on a stubbed clock (test_dvl._Sim), a stub controller.

Every motion here is SYNTHETIC: heading 0 and no rotation unless a test says
otherwise, tag fixes and DVL samples describing the same motion.
"""

from __future__ import annotations

import csv
import math
import tempfile
import time
from types import SimpleNamespace

import numpy as np

from rov_gui.control import dvl as DV
from rov_gui.control import dvl_pose as DVP
from rov_gui.state import Conn
from rov_gui.tests import test_dvl as TD


def _TC():
    from rov_gui.tests import test_control as TC
    return TC


class PoseSim(TD._Sim):
    """test_dvl._Sim plus what the DVL pose eats: the RAW_IMU integral
    (no rotation: constant, its record count advancing) and the Bar30."""

    def __init__(self, w, TC):
        super().__init__(w, TC)
        self.n_gyro = 0

    def step(self, v=(0.0, 0.0, 0.0), fix=True, dvl=True, valid=True,
             tick=True, dvl_v=None, f=None, imu=True, lat=0.0):
        w, TC = self.w, self.TC
        self.t += 0.05
        self.p = self.p + np.asarray(v, float) * 0.05
        if fix:
            w.on_nav_fix(TC._fix(list(self.p), 0.0, self.t - lat))
        if imu:
            self.n_gyro += 1
            im = TC._imu(depth=0.8, t=self.t, f=f)
            im.gyro_int, im.t_gyro, im.n_gyro = (0.0, 0.0, 0.0), self.t, \
                self.n_gyro
            im.depth_sp2_m, im.t_sp2 = float(self.p[2]), self.t
            w.on_vehicle_imu(im)
        w.on_telemetry(TC.Telemetry(armed=True, mode="MANUAL",
                                    conn=Conn.ONLINE, stamp=self.t))
        if dvl and w.dvl is not None:
            w.on_dvl(TD._sample(v if dvl_v is None else dvl_v, t=self.t,
                                valid=valid))
        if tick:
            w.tick()


def _worker(tmp, cli="control", z_source=None, **over):
    TC = _TC()
    w, bus, pilots, logs = TC._worker(tmp)
    if z_source is not None:
        # (the harness's nav YAML leaves the dataclass default, pressure;
        # the STATION's hw_nav.yaml says tag — set before the pose is built,
        # which takes it from the assembler)
        w.nav_cfg.z_source = z_source
        w.asm.z_source = z_source
    cfg = {"enabled": True}
    cfg.update(TD.MOUNT)
    w.cfg.dvl = cfg
    w._setup_dvl()
    dvp = {"bias_min_span_s": 5.0, "bias_min_n": 20}
    dvp.update(over)
    w.cfg.dvl_pose = dvp
    w.opts.dvl_pose = cli
    w._setup_dvl_pose()
    events = []
    bus.mpc_event.connect(lambda *a: events.append(a))
    return TC, w, bus, logs, events


def _rows(w):
    path = w._csv_path
    w._close_csv()
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _ready(sim, w, seconds=8.0):
    """Prove the DVL, switch it IN, and sit still with tags long enough for
    the anchor and the gyro bias."""
    sim.prove()
    w.set_dvl_use(True)
    assert w.dvl.use, w.reason
    for _ in range(int(seconds / 0.05)):
        sim.step()
    assert w.dvp.anchored
    assert w.dvp.head.bias_state == "ok", w.dvp.head.bias_info()


# =============================================================================
def test_without_dvl_pose_the_worker_is_what_it_was():
    TC = _TC()
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, _p, _l = TC._worker(tmp)
        assert w.dvp is None and not w.dvp_control
        sim = TD._Sim(w, TC)
        try:
            sim.engage()
            st = []
            bus.mpc_status.connect(st.append)
            for _ in range(5):
                sim.step(v=(0.05, 0.0, 0.0))
            assert st and all(s.dvp_mode == "" and s.p_dvp_flu is None
                              for s in st)
            w.set_dvl_pose_tag(False)
            assert "no DVL pose" in w.reason
            m = w._run_meta()
            assert m["dvl_pose"]["enabled"] is False
            rows = _rows(w)
            assert rows and all(r["tagfix"] == "" and r["dvp_src"] == ""
                                and r["dvp_ok"] == "" and r["dvp_px"] == "nan"
                                for r in rows)
        finally:
            sim.close()
    import rov_gui.control.workers as W
    n = len(W.DVP_CSV_COLUMNS)
    k = len(W.SN_CSV_COLUMNS)     # schema 24 (2026-10-19) appends the state noise's after them
    assert W.CSV_HEADER.strip().split(",")[-n - k:-k] == list(W.DVP_CSV_COLUMNS)


def test_dvl_pose_builds_the_dvl_and_yaml_control_alone_flies_shadow():
    opts = SimpleNamespace(dvl=False, dvl_pose="control")
    assert DV.resolve_with_opts({}, opts)["enabled"] is True
    assert DV.resolve_with_opts({}, SimpleNamespace(dvl=False,
                                                    dvl_pose=None))[
        "enabled"] is False
    with tempfile.TemporaryDirectory() as tmp:
        _TC_, w, _b, logs, _e = _worker(tmp, cli=None, mode="control")
        assert w.dvp is not None and w.dvp.mode == "shadow"
        assert not w.dvp_control
        assert any("only the command line" in m for _l, m in logs)
        w.teardown()


def test_a_tag_dropout_does_not_end_a_control_run_the_dvl_can_carry():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        sim = PoseSim(w, TC)
        try:
            _ready(sim, w)
            sim.engage()
            for _ in range(10):
                sim.step(v=(0.05, 0.0, 0.0))
            p_before = sim.p.copy()
            # 5 s with NO tag: the run goes on, on the DVL
            for _ in range(100):
                sim.step(v=(0.05, 0.0, 0.0), fix=False)
                assert w.engaged, w.reason
            assert w._dvp_last[1]["state"] == DVP.ST_DR
            # the pose followed the motion (datum frame: engage was here)
            m, _h = w._dvp_last
            moved = float(sim.p[0] - p_before[0])
            assert moved > 0.2
            # tags back: the pose returns to the tag state
            for _ in range(20):
                sim.step(v=(0.05, 0.0, 0.0))
            assert w.engaged
            assert w._dvp_last[1]["state"] in (DVP.ST_TAG, DVP.ST_BLEND)
            rows = _rows(w)
            dr = [r for r in rows if r["dvp_src"] == "dr"]
            # (the first ~0.5 s after the last fix the assembler still
            # bridges the tag state — engage.tag_stale_s — and the pose IS
            # that state: dvp_src tag)
            assert len(dr) >= 85, len(dr)
            # px is the TAG state, nan without one; dvp_px is what flew. The
            # tags' return is believed only once anchor_n (5) fixes agree:
            # the first four tagged rows still fly the dead reckoning.
            reacq = [r for r in dr if r["dvp_why"].startswith(
                "tags re-acquiring")]
            assert len(reacq) == 4 and all(r["px"] != "nan" for r in reacq)
            assert all(r["px"] == "nan" and r["dvp_ok"] == "1"
                       for r in dr if r not in reacq)
            assert float(dr[-1]["dvp_px"]) - float(dr[0]["dvp_px"]) > 0.2
            import rov_gui.control.workers as W
            width = len(W.CSV_HEADER.strip().split(","))
            assert all(len(r) == width for r in rows)
        finally:
            sim.close()


def test_without_a_usable_dvl_or_a_tag_the_run_ends_after_the_debounce():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        sim = PoseSim(w, TC)
        try:
            _ready(sim, w)
            sim.engage()
            for _ in range(5):
                sim.step()
            n = 0
            while w.engaged and n < 200:
                sim.step(fix=False, dvl=False)
                n += 1
            assert not w.engaged
            assert "dvl pose stale" in w.reason, w.reason
            # lost_s 1.5 + tag_stale_hold_s 0.4 [the shipped values]
            assert 1.5 / 0.05 <= n <= (1.5 + 0.4) / 0.05 + 3, n
        finally:
            sim.close()


def test_tag_fix_off_is_refused_until_the_pose_can_carry_the_vehicle():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        sim = PoseSim(w, TC)
        try:
            sim.step()
            w.set_dvl_pose_tag(False)
            assert "no anchor" in w.reason and w.dvp.tag_on
            for _ in range(10):
                sim.step()
            w.set_dvl_pose_tag(False)
            assert "switch is OUT" in w.reason, w.reason
            sim.prove()
            w.set_dvl_use(True)
            # control: samples taken while the switch was OUT were not usable
            # for the pose — the next one is
            w.set_dvl_pose_tag(False)
            assert "no usable DVL" in w.reason and w.dvp.tag_on
            # (prove() moved the vehicle 12 s WITHOUT ticks — the station
            # ticks at 20 Hz always — so the first usable sample shows the
            # pose that far off: a level change, believed after anchor_n
            # fixes and snapped, disengaged)
            for _ in range(10):
                sim.step()
            assert w.dvp.corrections[-1]["applied"] == "snap"
            # a pose without a learnt bias (as at the start of a session)
            w.dvp.head.bias_state = "learning"
            w.dvp.head.bias_t = None
            w.set_dvl_pose_tag(False)
            assert "gyro bias" in w.reason and w.dvp.tag_on
            for _ in range(int(8.0 / 0.05)):
                sim.step()
            w.set_dvl_pose_tag(False)
            assert not w.dvp.tag_on, w.reason
        finally:
            sim.close()


def test_the_switch_works_while_engaged_and_every_row_says_which_way():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        sim = PoseSim(w, TC)
        st = []
        bus.mpc_status.connect(st.append)
        try:
            _ready(sim, w)
            sim.engage()
            for _ in range(5):
                sim.step(v=(0.03, 0.0, 0.0))
            w.set_dvl_pose_tag(False)
            assert not w.dvp.tag_on and w.engaged
            for _ in range(40):
                sim.step(v=(0.03, 0.0, 0.0))
            assert w._dvp_last[1]["state"] == DVP.ST_DR
            assert st[-1].dvp_mode == "control" and not st[-1].dvp_tag_on
            assert st[-1].dvp_state == "dr" and st[-1].dvp_ok
            # with tags in view the drift is measured on every row
            assert st[-1].dvp_err_m is not None and st[-1].dvp_err_m < 0.01
            # ON from OFF is ONE correction (2026-10-18): the tags are a
            # re-acquisition, agreed, slewed, and OFF again by itself
            w.set_dvl_pose_tag(True)
            assert w.dvp.tag_on and w.dvp.oneshot is not None
            sim.step(v=(0.03, 0.0, 0.0))
            assert st[-1].dvp_oneshot and st[-1].dvp_tag_on
            seen = []
            for _ in range(60):
                sim.step(v=(0.03, 0.0, 0.0))
                seen.append(w._dvp_last[1]["state"])
                if not w.dvp.tag_on:
                    break
            assert not w.dvp.tag_on and w.dvp.oneshot is None, seen
            assert set(seen) & {DVP.ST_TAG, DVP.ST_BLEND}, seen
            sim.step(v=(0.03, 0.0, 0.0))
            assert w._dvp_last[1]["state"] == DVP.ST_DR
            assert not st[-1].dvp_oneshot and not st[-1].dvp_tag_on
            m = w._run_meta()
            tg = m["dvl_pose"]["toggles"]
            assert [x["on"] for x in tg] == [False, True, False], tg
            assert tg[1].get("oneshot") is True and tg[2]["auto"] == "done"
            c = m["dvl_pose"]["corrections"][-1]
            assert c["kind"] == "reacquire" and c["applied"] == "slew", c
            assert m["dvl_pose"]["at_engage"]["tag_on"] is True
            rows = _rows(w)
            flags = [r["tagfix"] for r in rows if r["engaged"] == "1"]
            k1 = flags.index("1", flags.index("0"))
            assert "0" in flags[k1:] and flags[-1] == "0", flags
            joined = [" ".join(map(str, e)) for e in events]
            assert any("TAG FIX ONCE" in e for e in joined), joined
            assert any("TAG FIX OFF (correction in)" in e for e in joined)
        finally:
            sim.close()


def test_off_a_dvl_false_lock_ends_a_control_run_through_the_dvls_own_hold():
    """TAG FIX OFF, tags in view, the DVL reading 4 cm/s the vehicle does not
    have: DvlAid's disagreement hold takes its samples out, so the pose is
    LOST and the stale debounce ends the run (review 2026-10-18: this test
    used to claim the divergence abort did it — that path is
    test_off_a_heading_divergence_ends_a_control_run_on_the_pose_fault)."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        sim = PoseSim(w, TC)
        try:
            _ready(sim, w)
            sim.engage()
            w.set_dvl_pose_tag(False)
            assert not w.dvp.tag_on, w.reason
            n = 0
            while w.engaged and n < 400:
                sim.step(v=(0.0, 0.0, 0.0), dvl_v=(0.04, 0.0, 0.0))
                n += 1
            assert not w.engaged
            assert "dvl pose stale" in w.reason and "dis_" in w.reason, \
                w.reason
        finally:
            sim.close()


def test_shadow_never_changes_what_the_controller_flies_on():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp, cli="shadow")
        sim = PoseSim(w, TC)
        try:
            assert w.dvp is not None and not w.dvp_control
            _ready(sim, w)
            sim.engage()
            n = 0
            while w.engaged and n < 100:
                sim.step(fix=False)
                n += 1
            # a tag dropout ends the run exactly as without --dvl_pose
            assert not w.engaged and "tag fix stale" in w.reason, w.reason
            rows = _rows(w)
            assert any(r["dvp_src"] for r in rows)
        finally:
            sim.close()


def test_a_control_engage_needs_the_dvl_in_and_proven():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        sim = PoseSim(w, TC)
        try:
            for _ in range(20):
                sim.step()
            w.on_enable(True)
            w.set_engaged(True)
            assert not w.engaged
            assert "cannot carry the pose" in w.reason, w.reason
        finally:
            sim.close()


def test_one_estimate_flies():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        try:
            assert w.dvp_control
            if w.ekf is not None:
                assert "dvl_pose" in w._ekf_control_refusal()
            w.set_estimator("ekf")
            assert not w.ekf_control
        finally:
            w.teardown()


def test_the_pwm_guard_is_told_the_pose_freshness_on_dead_reckoned_ticks():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        try:
            w.ctrl = SimpleNamespace(obs_guard=SimpleNamespace(
                cfg={"min_tags": 2}))
            w._dvp_last = (None, {"state": DVP.ST_DR, "gap_s": 0.12})
            assert w._dvp_fix_report({"tag_age": None}) == (2, 0.12)
            w._dvp_last = (None, {"state": DVP.ST_TAG})
            n, age = w._dvp_fix_report({"tag_age": 0.1})
            assert age == 0.1
        finally:
            w.teardown()


# =============================================================================
# review 2026-10-18 — ported mutation killers, then the fixes' own tests
# =============================================================================
from rov_gui.control.state_assembler import rot_zyx        # noqa: E402
import pytest                                              # noqa: E402


class YawSim(PoseSim):
    """PoseSim plus a heading: ``v`` is NED, the DVL reads it in the body
    frame, the gyro integral turns with ``r`` (roll = pitch = 0)."""

    def __init__(self, w, TC):
        super().__init__(w, TC)
        self.yaw = 0.0

    def step(self, v=(0.0, 0.0, 0.0), fix=True, dvl=True, valid=True,
             tick=True, dvl_v=None, f=None, imu=True, lat=0.0, r=0.0):
        w, TC = self.w, self.TC
        self.t += 0.05
        self.yaw += r * 0.05
        v = np.asarray(v, float)
        self.p = self.p + v * 0.05
        if fix:
            w.on_nav_fix(TC._fix(list(self.p), self.yaw, self.t - lat))
        if imu:
            self.n_gyro += 1
            im = TC._imu(yaw=self.yaw, r=r, depth=0.8, t=self.t, f=f)
            im.gyro_int, im.t_gyro, im.n_gyro = (0.0, 0.0, self.yaw), \
                self.t, self.n_gyro
            im.depth_sp2_m, im.t_sp2 = float(self.p[2]), self.t
            w.on_vehicle_imu(im)
        w.on_telemetry(TC.Telemetry(armed=True, mode="MANUAL",
                                    conn=Conn.ONLINE, stamp=self.t))
        if dvl and w.dvl is not None:
            vb = rot_zyx(0.0, 0.0, self.yaw).T @ (v if dvl_v is None
                                                  else np.asarray(dvl_v))
            w.on_dvl(TD._sample(tuple(vb), t=self.t, valid=valid))
        if tick:
            w.tick()


def test_off_a_heading_divergence_ends_a_control_run_on_the_pose_fault():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        sim = PoseSim(w, TC)
        try:
            _ready(sim, w)
            sim.engage()
            w.set_dvl_pose_tag(False)
            assert not w.dvp.tag_on, w.reason
            real, drift = w.on_vehicle_imu, [0.0]

            def imu(im):                 # 5 deg/s the vehicle does not turn
                drift[0] += math.radians(5.0) * 0.05
                im.gyro_int = (0.0, 0.0, drift[0])
                real(im)
            w.on_vehicle_imu = imu
            n = 0
            while w.engaged and n < 200:
                sim.step()
                n += 1
            assert not w.engaged and "diverged" in w.reason, w.reason
            assert n <= 3.5 / 0.05, n    # 10 deg at 5 deg/s + abort_hold_s
        finally:
            sim.close()


def test_the_pose_is_the_tag_state_in_the_datum_frame_and_the_hull_stays_the_tags():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        sim = YawSim(w, TC)
        st = []
        bus.mpc_status.connect(st.append)
        try:
            _ready(sim, w)
            sim.engage()
            for _ in range(60):          # a slow turn in place, tags in view
                sim.step(r=math.radians(3.0))
            for _ in range(20):          # a dropout: dead reckoning
                sim.step(fix=False)
            assert w._dvp_last[1]["state"] == DVP.ST_DR
            # the hull (p_flu) is the TAG state: none in a dropout
            assert st[-1].p_flu is None and st[-1].p_dvp_flu is not None
            rows = _rows(w)
            tag_rows = [r for r in rows if r["engaged"] == "1"
                        and r["dvp_src"] == "tag"]
            assert len(tag_rows) > 50
            assert max(abs(float(r["yaw_deg"])) for r in tag_rows) > 5.0
            for r in tag_rows:
                for a, b, tol in (("dvp_px", "px", 1e-4),
                                  ("dvp_py", "py", 1e-4),
                                  ("dvp_pz", "pz", 1e-4),
                                  ("dvp_yaw_deg", "yaw_deg", 1e-2)):
                    assert abs(float(r[a]) - float(r[b])) < tol, \
                        (a, r[a], b, r[b])
        finally:
            sim.close()


def test_a_station_hold_does_not_bridge_a_lost_dvl_pose():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        sim = PoseSim(w, TC)
        try:
            _ready(sim, w)
            sim.engage()
            w.set_scenario({"shape": "station", "origin_tag": None,
                            "origin": "current", "yaw_map_deg": 0.0})
            w.set_traj(True)
            assert w.station is not None, w.reason
            for _ in range(5):
                sim.step()
            n = 0
            while w.engaged and n < 100:
                sim.step(fix=False, dvl=False)
                n += 1
            assert not w.engaged and "dvl pose stale" in w.reason, \
                (w.reason, w._bridge.tier)
            assert n <= (1.5 + 0.4) / 0.05 + 3, n
        finally:
            sim.close()


def test_a_pose_code_fault_ends_a_control_run_and_never_falls_back():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        sim = PoseSim(w, TC)
        st = []
        bus.mpc_status.connect(st.append)
        try:
            _ready(sim, w)
            sim.engage()

            def boom(*a, **k):
                raise RuntimeError("synthetic")
            w.dvp.step = boom
            n = 0
            while w.engaged and n < 60:
                sim.step()               # the tags stay fresh all along
                n += 1
            assert not w.engaged and "code fault" in w.reason, w.reason
            assert n <= 0.4 / 0.05 + 2, n
            assert st[-1].dvp_state == DVP.ST_FAULT
        finally:
            sim.close()


def test_a_non_finite_pose_never_reaches_the_controller():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        sim = PoseSim(w, TC)
        try:
            _ready(sim, w)
            sim.engage()
            real = w.dvp.step

            def nan_step(*a, **k):
                m, h = real(*a, **k)
                if m is not None:
                    m = dict(m, eta=np.array(m["eta"], float))
                    m["eta"][0] = float("nan")
                return m, h
            w.dvp.step = nan_step
            seen, real_ctrl = [], w.ctrl.step

            def ctrl_step(eta, nu, nudot, t):
                seen.append(np.array(eta, float))
                return real_ctrl(eta, nu, nudot, t)
            w.ctrl.step = ctrl_step
            for _ in range(12):
                sim.step()
            assert all(np.all(np.isfinite(e)) for e in seen)
            assert not w.engaged and "non-finite" in w.reason, w.reason
        finally:
            sim.close()


def test_a_control_engage_during_a_tag_dropout_starts_from_the_pose():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        sim = PoseSim(w, TC)
        try:
            _ready(sim, w)
            for _ in range(15):          # disengaged, no tag: the DVL carries
                sim.step(fix=False)
            assert w._dvp_last[1]["state"] == DVP.ST_DR
            z_before = w.asm.z_offset
            w.on_enable(True)
            w.set_engaged(True)
            assert w.engaged, w.reason
            assert w._dvp_at_engage["state"] == DVP.ST_DR
            # no fresh fix at this engage: the assembler's depth offset is
            # NOT taken on the stale last one (review 2026-10-18) — under
            # z_source pressure (this harness) the frame it had is KEPT, so
            # no self-anchor happens mid-run (fix verification 2026-10-18)
            assert w.asm.z_source == "pressure"
            assert w.asm.z_offset == pytest.approx(z_before)
        finally:
            sim.close()


def test_a_control_engage_while_dead_reckoning_rechecks_what_off_needs():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        sim = PoseSim(w, TC)
        try:
            _ready(sim, w)
            w.set_dvl_pose_tag(False)
            assert not w.dvp.tag_on, w.reason
            for _ in range(12):          # 0.6 s without a DVL sample: HELD
                sim.step(dvl=False)
            assert w._dvp_last[1]["state"] == DVP.ST_HELD
            w.on_enable(True)
            w.set_engaged(True)
            assert not w.engaged
            assert "the pose is dead-reckoning (TAG FIX OFF) but" in w.reason, \
                w.reason
        finally:
            sim.close()


def test_a_control_engage_needs_an_anchored_pose():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        sim = TD._Sim(w, TC)        # no gyro integral: it can never anchor
        try:
            sim.prove()
            w.set_dvl_use(True)
            for _ in range(40):
                sim.step()
            assert not w.dvp.anchored
            assert w._dvp_last[1]["state"] == DVP.ST_TAG     # still the tags
            w.on_enable(True)
            w.set_engaged(True)
            assert not w.engaged and "not anchored" in w.reason, w.reason
        finally:
            sim.close()


def test_a_shadow_pose_dead_reckons_with_the_dvl_switch_out():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp, cli="shadow")
        sim = PoseSim(w, TC)
        try:
            assert not w.dvl.use and not w.dvp_control
            for _ in range(20):
                sim.step()
            assert w.dvp.n_usable > 0, (w.dvp.n_unusable,
                                        w.dvp.unusable_code)
        finally:
            sim.close()


def test_control_is_refused_beside_another_estimate_in_control():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        try:
            assert w.dvp_control
            w.ekf_control = True
            w._setup_dvl_pose()
            assert w.dvp is not None and w.dvp.mode == "shadow"
            assert not w.dvp_control
            assert any("control REFUSED" in m for _l, m in logs)
        finally:
            w.ekf_control = False
            w.teardown()


def test_the_policy_counts_dead_reckoned_ticks_as_bridged():
    """A9 as shipped (policy.allow_dr_pose false). With the knob
    (--policy-dr, 2026-10-19) a DR/HELD tick is an anchor instead; the
    tag-dropout ladder stays a bridge, and outside --dvl_pose control the
    knob has nothing to act on."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        try:
            assert w.cfg.policy["allow_dr_pose"] is False
            for st, want in ((DVP.ST_DR, True), (DVP.ST_HELD, True),
                             (DVP.ST_TAG, False), (DVP.ST_BLEND, False)):
                w._dvp_last = (None, {"state": st})
                assert w._policy_bridged() is want, st
            w.cfg.policy["allow_dr_pose"] = True
            assert w._policy_dr_pose()
            for st in (DVP.ST_DR, DVP.ST_HELD, DVP.ST_TAG, DVP.ST_BLEND):
                w._dvp_last = (None, {"state": st})
                assert w._policy_bridged() is False, st
                # only a dead-reckoned pose may ANCHOR a plan; a held one is
                # frozen (safety review 2026-10-19 #3)
                assert w._policy_dvp_dr(anchor=True) is (st == DVP.ST_DR), st
            old = w._bridge
            w._bridge = SimpleNamespace(tier="imu")
            assert w._policy_bridged() is True
            w._bridge = old
            w.dvp_control = False
            assert not w._policy_dr_pose()
        finally:
            w.teardown()


def test_policy_dr_pose_anchors_a_dead_reckoned_tick_on_the_flown_pose():
    """--policy-dr (2026-10-19): a dead-reckoned / held tick WITHOUT a tag fix
    is a fresh anchor — the flown pose, stamped with the tick — so the policy
    worker keys a proprio row on it (backends/policy.py reads st.fix_fresh).
    Without the knob the tick stays out (A9); so does a tick with no pose at
    all, and one under the tag-dropout ladder."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        try:
            got = []
            bus.policy_state.connect(got.append)
            flown = {"eta": np.array([1.1, 2.0, 0.5, 0.0, 0.0, 0.1]),
                     "nu": np.zeros(6), "nudot": np.zeros(6)}
            w.fix = None
            w._flown_meas = flown
            w._dvp_last = (flown, {"state": DVP.ST_DR})
            w._emit_policy_state(None, {})
            assert got[-1].fix_fresh is False and got[-1].t_fix is None
            w.cfg.policy["allow_dr_pose"] = True
            w._emit_policy_state(None, {})
            s = got[-1]
            assert s.fix_fresh is True and s.t_fix is not None
            assert s.eta[0] == pytest.approx(1.1)
            assert s.t_fix == pytest.approx(s.t)
            time.sleep(0.002)
            w._emit_policy_state(None, {})          # the stamp advances
            assert got[-1].t_fix > s.t_fix
            w._dvp_last = (flown, {"state": DVP.ST_HELD})
            w._emit_policy_state(None, {})           # HELD: frozen, no row
            assert got[-1].fix_fresh is False and got[-1].t_fix is None
            # a dr tick WITH a tag in view (TAG FIX OFF, tags recorded): the
            # same tick stamp and the flown pose, not t_capture (review #2 —
            # one stamp rule across the tagless / tagged seam)
            w._dvp_last = (flown, {"state": DVP.ST_DR})
            w.fix = TC._fix([1.0, 2.0, 0.5], 0.1, 100.0)
            tag = {"eta": np.array([1.0, 2.0, 0.5, 0.0, 0.0, 0.1]),
                   "nu": np.zeros(6), "nudot": np.zeros(6)}
            w._emit_policy_state(tag, {})
            s2 = got[-1]
            assert s2.fix_fresh is True and s2.t_fix == pytest.approx(s2.t)
            assert s2.eta[0] == pytest.approx(1.1)
            w.cfg.policy["allow_dr_pose"] = False     # A9: t_capture, flown eta
            w._emit_policy_state(tag, {})
            assert got[-1].t_fix == pytest.approx(100.0)
            w.cfg.policy["allow_dr_pose"] = True
            w.fix = None
            w._flown_meas = None                     # no pose: nothing flew
            w._dvp_last = (None, {"state": DVP.ST_LOST})
            w._emit_policy_state(None, {})
            assert got[-1].fix_fresh is False and got[-1].t_fix is None
            w._flown_meas = flown
            w._dvp_last = (flown, {"state": DVP.ST_DR})
            old = w._bridge
            w._bridge = SimpleNamespace(tier="imu")
            w._emit_policy_state(None, {})
            assert got[-1].fix_fresh is False
            w._bridge = old
        finally:
            w.teardown()


def test_policy_dr_pose_keeps_the_mission_through_a_tag_blackout():
    """The whole seam, both ways. A policy mission under --dvl_pose control
    loses the tags for 3 s (> policy.stale_s 1.5). Shipped: the A9 stop
    ends it ('bridge'). With --policy-dr it goes on — every dead-reckoned
    tick a fresh anchor, no skip_bridge, the record counting the ticks and
    the meta saying which gate was off."""
    from rov_gui.tests import test_policy as TP
    for knob in (False, True):
        with tempfile.TemporaryDirectory() as tmp:
            TC, w, bus, logs, events = _worker(tmp)
            sim = PoseSim(w, TC)
            try:
                w.cfg.policy["allow_dr_pose"] = knob
                w.policy_present = True
                w.set_scenario({"shape": "policy"})
                _ready(sim, w)
                got = []
                bus.policy_state.connect(got.append)
                TP._feed_status(w, stamp=sim.t)
                sim.engage()
                for _ in range(5):
                    TP._feed_status(w, stamp=sim.t)
                    sim.step()
                w.set_traj(True)
                assert w.traj_on, w.reason
                n0 = len(got)
                for _ in range(60):                  # 3 s without a tag
                    TP._feed_status(w, stamp=sim.t)
                    sim.step(v=(0.05, 0.0, 0.0), fix=False)
                dr_rows = [s for s in got[n0:]
                           if s.fix_fresh and s.t_fix == s.t]
                rp = w.replay if w.replay is not None else w._replay_last
                if knob:
                    assert w.traj_on and w.engaged, w.reason
                    assert rp.get("bridge_since") is None
                    assert rp.get("skip_bridge", 0) == 0
                    assert rp["dr_ticks"] >= 40, rp["dr_ticks"]
                    assert len(dr_rows) >= 40, len(dr_rows)
                    assert any("ON THE DVL POSE (--policy-dr)" in m
                               for _l, m in logs)
                    pm = w._policy_meta()
                    dp = pm["dr_pose"]
                    assert (dp["allowed"], dp["active"], dp["pinned"]) == \
                        (True, True, True)
                    assert dp["ticks"] == rp["dr_ticks"]
                    assert dp["plans"] == 0 and dp["installed"] == 0
                    assert dp["blind_bound_s"] == {"max_unaided_s": None,
                                                   "max_blind_s": None}
                    g = pm["gates_enforced"]
                    assert g["bridge_stop_dr_pose"] is False
                    # the station ladder never steps under --dvl_pose control
                    assert g["bridge_stop"] is False
                    assert g["bridge_stop_via"] == "dvl_pose"
                    assert any("ON THE DVL POSE" in m and "blind bound" in m
                               and "NONE" in m for _l, m in logs)
                else:
                    assert not w.traj_on
                    assert rp["end_reason"] == "bridge", rp.get("end_reason")
                    assert rp.get("dr_ticks", 0) == 0
                    assert not dr_rows
                    assert any("POLICY flew on a bridged/DR state" in m
                               for _l, m in logs)
                    pm = w._policy_meta()
                    assert pm["dr_pose"]["allowed"] is False
                    assert pm["dr_pose"]["pinned"] is False
                    assert pm["gates_enforced"]["bridge_stop_dr_pose"] is True
            finally:
                sim.close()


def test_policy_dr_setup_says_when_the_knob_is_inert_or_unbounded():
    """Safety review 2026-10-19 #6 / #1: the knob acts only under --dvl_pose
    control — a shadow pose gets an INERT error at setup; a control pose
    with no blind-flight bound (the shipped null max_unaided_s /
    max_blind_s) gets a warning naming what still bounds the run; a bound
    silences it."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp, cli=None, mode="control")
        try:
            assert not w.dvp_control
            w.cfg.policy["allow_dr_pose"] = True
            n = len(logs)
            w._policy_dr_setup_check()
            assert any(l == "error" and "INERT" in m for l, m in logs[n:])
            assert not w._policy_dr_pose()
        finally:
            w.teardown()
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        try:
            w.cfg.policy["allow_dr_pose"] = True
            n = len(logs)
            w._policy_dr_setup_check()
            assert any(l == "warn" and "NO blind-flight bound" in m
                       for l, m in logs[n:])
            assert w._policy_dr_blind_bound() == {"max_unaided_s": None,
                                                  "max_blind_s": None}
            w.cfg.policy["allow_dr_pose"] = False
            n = len(logs)
            w._policy_dr_setup_check()
            assert len(logs) == n                       # knob off: silent
        finally:
            w.teardown()
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp, max_blind_s=60.0)
        try:
            w.cfg.policy["allow_dr_pose"] = True
            n = len(logs)
            w._policy_dr_setup_check()
            assert not any("blind-flight bound" in m for _l, m in logs[n:])
            assert w._policy_dr_blind_bound()["max_blind_s"] == 60.0
        finally:
            w.teardown()


def test_the_policy_dr_flag_needs_the_policy_mission_and_the_control_pose():
    from rov_gui.__main__ import build_parser, check_policy
    p = build_parser()
    base = ["--source", "demo", "--mpc", "--policy", "--dvl_pose"]
    assert p.parse_args(base).policy_dr is False
    a = p.parse_args(base + ["--policy-dr"])
    assert a.policy_dr is True
    v = check_policy(a)
    assert not (v and v[0] == "refuse"), v
    for drop in ("--policy", "--mpc", "--dvl_pose"):
        args = [x for x in base if x != drop] + ["--policy-dr"]
        v = check_policy(p.parse_args(args))
        assert v and v[0] == "refuse" and "--policy-dr" in v[1], (drop, v)
    v = check_policy(p.parse_args(base[:-1] + ["--dvl_pose", "shadow",
                                               "--policy-dr"]))
    assert v and v[0] == "refuse" and "control" in v[1], v


def test_the_policy_is_told_the_flown_pose_while_a_correction_slews():
    """BLEND: the follower flies tag + off, so the policy's anchor history
    and proprio are that pose (review 2026-10-18)."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        try:
            got = []
            bus.policy_state.connect(got.append)
            tag = {"eta": np.array([1.0, 2.0, 0.5, 0.0, 0.0, 0.1]),
                   "nu": np.zeros(6), "nudot": np.zeros(6)}
            flown = {"eta": np.array([1.1, 2.0, 0.5, 0.0, 0.0, 0.1]),
                     "nu": np.zeros(6), "nudot": np.zeros(6)}
            w.fix = TC._fix([1.0, 2.0, 0.5], 0.1, 100.0)
            w._flown_meas = flown
            w._emit_policy_state(tag, {})
            assert got[-1].eta[0] == pytest.approx(1.1)
            # a fresh tag state but no pose this tick (a blind
            # re-acquisition): nothing flew, so neither the tag state nor a
            # fresh history row — the last flown pose (fix verification
            # 2026-10-18)
            w._flown_meas = None
            w._eta = np.array([0.9, 2.0, 0.5, 0.0, 0.0, 0.1])
            w._emit_policy_state(tag, {})
            assert got[-1].eta[0] == pytest.approx(0.9)
            assert got[-1].fix_fresh is False
            # without --dvl_pose control the tag state is the state
            w.dvp_control = False
            w._emit_policy_state(tag, {})
            assert got[-1].eta[0] == pytest.approx(1.0)
            w.dvp_control = True
        finally:
            w.teardown()


def test_a_follow_is_demoted_the_tick_the_pose_stops_being_the_tags():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        sim = PoseSim(w, TC)
        try:
            _ready(sim, w)
            sim.engage()
            calls = []

            def demote(why):
                calls.append((w._dvp_last[1].get("state"), why))
                w.follow = None
            w._follow_to_station = demote
            w._tick_follow = lambda t: None
            w.follow = {"stub": True}
            sim.step()
            assert calls == [] and w.follow is not None
            for _ in range(15):
                sim.step(fix=False)
            assert calls and calls[0] == (DVP.ST_DR, "tag fix lost"), calls
        finally:
            w.follow = None
            sim.close()


def test_the_pwm_guard_hears_the_dvl_age_on_a_dead_reckoned_tick():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        sim = PoseSim(w, TC)
        try:
            _ready(sim, w)
            sim.engage()
            for _ in range(15):
                sim.step(fix=False)
            assert w._dvp_last[1]["state"] == DVP.ST_DR
            heard = []
            w.ctrl.obs_guard = SimpleNamespace(cfg={"min_tags": 3})
            w.ctrl.note_fix = lambda n, age: heard.append((n, age))
            w._pwm_transport = lambda: True
            try:
                sim.step(fix=False)
            except Exception:            # noqa: BLE001 — the stub has no pulses
                pass
            assert heard, "note_fix not called"
            n, age = heard[0]
            assert n == 3 and age is not None and age < 0.3, heard
        finally:
            sim.close()


def test_dvl_pose_refuses_the_legacy_selection():
    TC = _TC()
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = TC._worker(tmp)
        cfg = {"enabled": True, "selection": "legacy"}
        cfg.update(TD.MOUNT)
        w.cfg.dvl = cfg
        w._setup_dvl()
        assert w.dvl is not None and not w._dvl_primary()
        w.cfg.dvl_pose = {}
        w.opts.dvl_pose = "control"
        w._setup_dvl_pose()
        assert w.dvp is None and not w.dvp_control
        assert any("needs dvl.selection primary" in m for _l, m in logs)
        w.teardown()


class Mount180Sim(PoseSim):
    """The DVL yawed 180 deg on the hull (x aft): it reports the body
    velocity in ITS frame — every other worker test uses an identity mount
    at the origin, where the DVL frame IS the body frame."""

    def step(self, v=(0.0, 0.0, 0.0), dvl_v=None, **kw):
        vv = np.asarray(v if dvl_v is None else dvl_v, float)
        return super().step(v=v, dvl_v=(-vv[0], -vv[1], vv[2]), **kw)


def test_the_pose_integrates_the_body_origin_velocity_not_the_dvls():
    TC = _TC()
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = TC._worker(tmp)
        w.cfg.dvl = {"enabled": True, "mount_rpy_deg": [0.0, 0.0, 180.0],
                     "mount_t_flu": [0.0, 0.0, 0.0]}
        w._setup_dvl()
        w.cfg.dvl_pose = {"bias_min_span_s": 5.0, "bias_min_n": 20}
        w.opts.dvl_pose = "control"
        w._setup_dvl_pose()
        assert w.dvp_control
        sim = Mount180Sim(w, TC)
        try:
            _ready(sim, w)
            sim.engage()
            for _ in range(10):
                sim.step(v=(0.05, 0.0, 0.0))
            for _ in range(15):          # the assembler's bridge first
                sim.step(v=(0.05, 0.0, 0.0), fix=False)
            assert w._dvp_last[1]["state"] == DVP.ST_DR
            x0, p0 = float(w._dvp_last[0]["eta"][0]), sim.p.copy()
            for _ in range(40):          # 2 s on the DVL alone
                sim.step(v=(0.05, 0.0, 0.0), fix=False)
                assert w.engaged, w.reason
            moved = float(w._dvp_last[0]["eta"][0]) - x0
            assert moved == pytest.approx(float(sim.p[0] - p0[0]), abs=0.01)
        finally:
            sim.close()


def test_an_engaged_tick_that_did_not_step_carries_the_sentinels():
    """A --dvl_pose row the controller did not step (the stale debounce
    with no pose) is written — and never as 'commanded zero, solver fine'."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        sim = PoseSim(w, TC)
        try:
            _ready(sim, w)
            sim.engage()
            for _ in range(5):
                sim.step()
            n = 0
            while w.engaged and n < 100:
                sim.step(fix=False, dvl=False)
                n += 1
            rows = [r for r in _rows(w) if r["engaged"] == "1"]
            lost = [r for r in rows if r["dvp_src"] == "lost"]
            assert lost
            assert all(r["uX"] == "nan" and r["solver_status"] == "-1"
                       for r in lost), lost[0]
            flown = [r for r in rows if r["dvp_src"] in ("tag", "dr")]
            assert flown and all(r["solver_status"] != "-1" for r in flown)
        finally:
            sim.close()


def test_a_shadow_fault_is_said_once_as_the_run_control_would_have_ended():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp, cli="shadow",
                                           max_blind_s=None)
        sim = PoseSim(w, TC)
        try:
            _ready(sim, w)
            sim.engage()
            w.dvp.set_tag_on(False, sim.t, engaged=True)
            real, drift = w.on_vehicle_imu, [0.0]

            def imu(im):
                drift[0] += math.radians(5.0) * 0.05
                im.gyro_int = (0.0, 0.0, drift[0])
                real(im)
            w.on_vehicle_imu = imu
            for _ in range(100):
                sim.step()
            said = [m for _l, m in logs if "would have ended" in m]
            assert len(said) == 1 and "diverged" in said[0], said
            # ...and disengaged it is what a START would be refused on, not a
            # run's end (fix verification 2026-10-18)
            w.disengage("released")
            logs.clear()
            w._dvp_fault_logged = ""
            for _ in range(5):
                sim.step()
            assert not [m for _l, m in logs if "would have ended" in m]
            said = [m for _l, m in logs if "START now would be refused" in m]
            # (shadow refuses no START on it: the line names control's)
            assert said and all("a control START" in m for m in said), said
        finally:
            sim.close()


def test_a_refusal_note_does_not_outlive_its_reason():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        sim = PoseSim(w, TC)
        st = []
        bus.mpc_status.connect(st.append)
        try:
            sim.step()
            w.set_dvl_pose_tag(False)                 # refused: no anchor
            sim.step()
            assert "refused" in st[-1].dvp_note
            for _ in range(int(7.0 / 0.05)):          # > the 6 s window
                sim.step()
            assert "refused" not in st[-1].dvp_note, st[-1].dvp_note
        finally:
            sim.close()


def test_the_meta_says_off_not_yamls_boolean():
    TC = _TC()
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = TC._worker(tmp)
        w.cfg.dvl_pose = {"mode": False}           # what `mode: off` loads as
        assert w._dvp_meta() == {"enabled": False, "mode": "off"}
        w.teardown()


def test_the_launch_line_names_every_limit_and_a_failure_to_say_it_disables():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        line = [m for _l, m in logs if "dead-reckoned pose" in m][-1]
        for want in ("the OFF divergence abort: 25 cm / 10 deg / 20 cm z "
                     "for 1 s",
                     "no time limit without tags",     # max_blind_s null
                     # (xy / yaw only since round 9 — z never ends ON — and
                     # the ON re-acquisition watch named: round 10b)
                     "a held jump's drift, and an ON re-acquisition's not "
                     "yet agreed, are watched in xy / yaw: 25 cm / 10 deg "
                     "for 1 s",
                     "no heading from a gyro more than 0.3 s late",
                     "TAG FIX pressed while OFF corrects ONCE",
                     "DVL <= 0.3 m/s while dead-reckoning", "CONTROL"):
            assert want in line, (want, line)
        assert "unaided <=" not in line                 # max_unaided_s null
        w.teardown()
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp, abort_err_m=None,
                                           abort_err_yaw_deg=None,
                                           abort_err_z_m=None,
                                           max_unaided_s=30.0)
        line = [m for _l, m in logs if "dead-reckoned pose" in m][-1]
        assert "the OFF divergence abort: off" in line
        assert "watched in xy / yaw: off" in line
        assert "unaided <= 30 s" in line
        # the launch line cannot be said: the pose does not fly (never the
        # worker — the setup's rule)
        w._dvp_banner = lambda: (_ for _ in ()).throw(ValueError("x"))
        w.opts.dvl_pose = "control"
        w._setup_dvl_pose()
        assert w.dvp is None and not w.dvp_control
        assert any("ValueError: x — DISABLED" in m for _l, m in logs)
        w.teardown()


# =============================================================================
# fix verification 2026-10-18 (round 3)
# =============================================================================
def _worker_z(tmp, z_source, **kw):
    return _worker(tmp, z_source=z_source, **kw)


class OffsetSim(PoseSim):
    """PoseSim whose tag fixes read ``fix_off`` (NED) off the truth, and
    whose map is tilted against gravity by ``tilt`` (tag z = z + tilt * y;
    the depth sensors read the true z)."""

    def __init__(self, w, TC, tilt=0.0):
        super().__init__(w, TC)
        self.fix_off = np.zeros(3)
        self.tilt = tilt

    def step(self, v=(0.0, 0.0, 0.0), fix=True, dvl=True, valid=True,
             tick=True, dvl_v=None, f=None, imu=True, lat=0.0):
        w, TC = self.w, self.TC
        self.t += 0.05
        self.p = self.p + np.asarray(v, float) * 0.05
        if fix:
            q = self.p + self.fix_off
            q[2] = q[2] + self.tilt * float(self.p[1])
            w.on_nav_fix(TC._fix(list(q), 0.0, self.t - lat))
        if imu:
            self.n_gyro += 1
            im = TC._imu(depth=float(self.p[2]), t=self.t, f=f)
            im.gyro_int, im.t_gyro, im.n_gyro = (0.0, 0.0, 0.0), self.t, \
                self.n_gyro
            im.depth_sp2_m, im.t_sp2 = float(self.p[2]), self.t
            w.on_vehicle_imu(im)
        w.on_telemetry(TC.Telemetry(armed=True, mode="MANUAL",
                                    conn=Conn.ONLINE, stamp=self.t))
        if dvl and w.dvl is not None:
            w.on_dvl(TD._sample(v if dvl_v is None else dvl_v, t=self.t,
                                valid=valid))
        if tick:
            w.tick()


def test_a_flickering_view_keeps_a_control_run_on_the_tags():
    """4 fixes in every 20 ticks, the DVL 4 mm/s wrong: the agreement
    collects its fixes across the gaps and the pose is re-anchored — it
    dead-reckoned uncorrected before (fix verification 2026-10-18)."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        sim = PoseSim(w, TC)
        try:
            _ready(sim, w)
            sim.engage()
            for k in range(int(30.0 / 0.05)):
                sim.step(fix=(k % 20) < 4, dvl_v=(0.004, 0.0, 0.0))
                assert w.engaged, w.reason
            rows = [r for r in _rows(w) if r["engaged"] == "1"
                    and r["dvp_err_m"] != "nan"]
            assert rows and max(float(r["dvp_err_m"]) for r in rows) < 0.03
            assert len([c for c in w.dvp.corrections
                        if c["kind"] == "reacquire"]) > 5
        finally:
            sim.close()


def test_a_flickering_view_still_ends_an_off_control_run_that_diverged():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        sim = PoseSim(w, TC)
        try:
            _ready(sim, w)
            sim.engage()
            w.set_dvl_pose_tag(False)
            assert not w.dvp.tag_on, w.reason
            n = 0
            while w.engaged and n < int(90.0 / 0.05):
                sim.step(fix=(n % 20) < 4, dvl_v=(0.004, 0.0, 0.0))
                n += 1
            assert not w.engaged and "diverged" in w.reason, w.reason
            # 25 cm at 4 mm/s: ~63 s, plus the hold
            assert 55.0 < n * 0.05 < 75.0, n * 0.05
        finally:
            sim.close()


def test_a_shadow_run_says_where_a_lost_pose_would_have_ended_control():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp, cli="shadow")
        sim = PoseSim(w, TC)
        try:
            _ready(sim, w)
            sim.engage()
            w.set_dvl_pose_tag(False)
            assert not w.dvp.tag_on, w.reason
            for _ in range(60):                       # the DVL silent 3 s
                sim.step(dvl=False)
            said = [m for _l, m in logs if "would have ended" in m
                    and "the pose was lost" in m]
            assert len(said) == 1, said
            faults = [x["why"] for x in w.dvp.meta()["faults"]]
            assert any(x.startswith("(shadow) the pose was lost") for x in
                       faults), faults
            assert w.engaged                          # shadow: tags fly on
            for _ in range(10):                       # ...and once per onset
                sim.step(dvl=False)
            assert len([m for _l, m in logs if "the pose was lost" in m]) == 1
        finally:
            sim.close()


def test_a_start_re_gates_the_pose_the_datum_is_taken_from():
    """_engage_refusal judges the previous tick's pose (the tag state);
    the tags drop before START, so the step START takes its datum from is
    dead-reckoned — with the bias not learnt, that START is refused."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp, bias_min_span_s=60.0,
                                           bias_min_n=200)
        sim = PoseSim(w, TC)
        try:
            sim.prove()
            w.set_dvl_use(True)
            for _ in range(40):
                sim.step()
            assert w.dvp.anchored and w.dvp.head.bias_state != "ok"
            assert w._dvp_engage_refusal(sim.t) == ""
            for _ in range(16):                       # 0.8 s, no fix, no tick
                sim.step(fix=False, tick=False)
            w.on_enable(True)
            w.set_engaged(True)
            assert not w.engaged
            assert "dead-reckoning (TAG FIX ON, no tag)" in w.reason \
                and "bias" in w.reason, w.reason
        finally:
            sim.close()


def test_a_start_in_a_dropout_under_on_rechecks_what_off_needs():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp, bias_min_span_s=60.0,
                                           bias_min_n=200)
        sim = PoseSim(w, TC)
        try:
            sim.prove()
            w.set_dvl_use(True)
            for _ in range(40):
                sim.step()
            for _ in range(20):                       # a dropout, ticked
                sim.step(fix=False)
            assert w._dvp_last[1]["state"] == DVP.ST_DR and w.dvp.tag_on
            w.on_enable(True)
            w.set_engaged(True)
            assert not w.engaged
            assert ("the pose is dead-reckoning (TAG FIX ON, no tag) but the "
                    "gyro bias is not learnt yet") in w.reason, w.reason
        finally:
            sim.close()


def test_a_pose_fault_ends_a_control_run_on_a_tick_with_no_pose():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        try:
            w._stale_ticks = 0
            w._dvp_last = (None, {"fault": "dvl pose and tags disagree by "
                                           "31 cm (blind, > 0.25 m)"})
            why = w._runtime_fault(None, {"why": "dvl pose stale: x"}, 1.0)
            assert "disagree" in why, why
            w._dvp_last = (None, {"fault": ""})
            assert w._runtime_fault(None, {"why": "dvl pose stale: x"},
                                    1.0) == ""         # debounced
        finally:
            w.teardown()


def test_a_dead_reckoned_row_carries_the_followers_path_errors():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        sim = PoseSim(w, TC)
        try:
            _ready(sim, w)
            sim.engage()
            w.traj_on = True
            w._path_err = (0.12, -0.03)
            w._flown_meas = {"eta": np.zeros(6), "nu": np.zeros(6),
                             "nudot": np.zeros(6)}
            w._stepped = True                         # a flown DR row
            w._write_row(None, {"why": ""}, np.zeros(6), {}, None, 1.0,
                         sim.t)
            w._stepped = False                        # a pose, not stepped:
            w._write_row(None, {"why": ""}, np.zeros(6), {}, None, 1.0,
                         sim.t)                       # no stale errors
            w._flown_meas = None                      # a LOST row: none
            w._write_row(None, {"why": ""}, np.zeros(6), {}, None, 1.0,
                         sim.t)
            rows = _rows(w)
            assert float(rows[-3]["e_along"]) == pytest.approx(0.12)
            assert float(rows[-3]["e_cross"]) == pytest.approx(-0.03)
            assert rows[-2]["e_along"] == "nan" and rows[-1]["e_along"] == "nan"
        finally:
            w.traj_on = False
            sim.close()


def test_the_tick_tells_the_policy_the_pose_it_flew_while_a_correction_slews():
    """End to end through tick(): the policy's proprio is the flown pose
    (tag + the slewing correction), set by the tick itself."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        sim = OffsetSim(w, TC)
        got = []
        bus.policy_state.connect(got.append)
        try:
            _ready(sim, w)
            sim.engage()
            w.policy_present = True
            sim.fix_off = np.array([0.15, 0.0, 0.0])   # a lasting new level
            seen = 0
            for _ in range(40):
                sim.step()
                if w._dvp_last[1].get("state") == DVP.ST_BLEND:
                    seen += 1
                    flown = w._flown_meas
                    assert flown is not None
                    assert got[-1].eta[0] == pytest.approx(
                        float(flown["eta"][0]))
                    # ...and not the tag state, which is 15 cm on
                    assert abs(got[-1].eta[0] - float(
                        w._datumize(w.asm.step(w.fix, w.imu, sim.t, 0.05)[0]
                                    ["eta"])[0])) > 0.01
            assert seen > 5
        finally:
            w.policy_present = False
            sim.close()


@pytest.mark.parametrize("z_source", ["pressure", "tag"])
def test_a_start_after_a_move_on_a_tilted_map_is_no_z_jump(z_source):
    """The map plane 5 cm/m off gravity (the 10-17 artifact's size): under
    z_source pressure every START re-takes the assembler's z offset — a
    frame change, which the pose now follows before the datum (it held a DR
    START and slewed 15 cm in at rest, or ended the run at 5 m). Under tag
    (the station's) the offset never enters the state."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker_z(tmp, z_source)
        sim = OffsetSim(w, TC, tilt=0.05)
        try:
            _ready(sim, w)
            for _ in range(int(5.0 / 0.1 / 0.05)):     # 5 m in map y
                sim.step(v=(0.0, 0.1, 0.0))
            for _ in range(20):
                sim.step()
            sim.engage()
            assert w._dvp_at_engage["state"] in (DVP.ST_TAG, DVP.ST_BLEND)
            zs = []
            for _ in range(100):                         # 5 s at rest
                sim.step()
                assert w.engaged, w.reason
                zs.append(float(w._dvp_last[0]["eta"][2]))
            assert max(abs(z) for z in zs) < 0.01, max(zs)
            assert not [c for c in w.dvp.corrections if c["engaged"]]
        finally:
            sim.close()


def test_a_tagless_start_under_pressure_keeps_the_z_frame():
    """...and a tagless START keeps the frame the assembler had, so there is
    no self-anchor mid-run for the pose to read as a z correction (it was
    slewed in, or refused past max_corr_z_m on a tilted map — fix
    verification 2026-10-18)."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker_z(tmp, "pressure")
        sim = OffsetSim(w, TC, tilt=0.05)
        try:
            _ready(sim, w)
            for _ in range(int(3.0 / 0.1 / 0.05)):
                sim.step(v=(0.0, 0.1, 0.0))
            for _ in range(20):
                sim.step(fix=False)                      # a dropout...
            sim.step(fix=False)                          # ...START tagless
            w.on_enable(True)                            # (not sim.engage():
            w.set_engaged(True)                          #  it ticks a fix in)
            assert w.engaged, w.reason
            assert w._dvp_at_engage["state"] == DVP.ST_DR
            assert w.asm.z_offset is not None             # the frame kept
            prev, worst = None, 0.0
            for _ in range(200):                          # tags back, 10 s
                sim.step()
                assert w.engaged, w.reason
                m = w._dvp_last[0]
                if m is None:                             # (agreeing)
                    continue
                z = float(m["eta"][2])
                if prev is not None:
                    worst = max(worst, abs(z - prev))
                prev = z
            assert worst < 0.02 * 0.05 + 1e-3, worst      # never stepped
            assert not [c for c in w.dvp.corrections if c["engaged"]
                        and abs(c["err_z_m"]) > 0.05], w.dvp.corrections[-3:]
        finally:
            sim.close()


def test_a_latched_dvl_and_one_late_frame_under_on_do_not_end_the_run():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        sim = PoseSim(w, TC)
        try:
            _ready(sim, w)
            sim.engage()
            for _ in range(10):
                sim.step()
            w.dvl._trip("synthetic evidence")            # DvlAid latch
            for _ in range(10):
                sim.step()
            assert w.engaged, w.reason
            sim.step(lat=0.8)                              # ONE late frame
            for _ in range(10):
                sim.step()
            assert w.engaged, w.reason
        finally:
            sim.close()


def test_one_late_frame_with_the_dvl_gone_does_not_end_the_run_at_10_hz():
    """The DVL gone, the tags the pose at 10 Hz: one late frame used to
    wait anchor_n fixes for a blind re-acquisition and outlast the debounce
    (there was no drift to judge)."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        sim = PoseSim(w, TC)
        try:
            _ready(sim, w)
            sim.engage()
            for k in range(40):                            # the DVL silent
                sim.step(dvl=False, fix=bool(k % 2 == 0))
            assert w.engaged, w.reason
            sim.step(dvl=False, lat=0.8)                   # ONE late frame
            for k in range(20):
                sim.step(dvl=False, fix=bool(k % 2 == 0))
                assert w.engaged, w.reason
        finally:
            sim.close()


class LeverSim(YawSim):
    """YawSim with the DVL at the SHIPPED mount's lever arm (rotation 0 —
    the device applies its own 180 deg — and mount_t_flu [-0.232, 0,
    -0.105], config/hw_mpc.yaml dvl): it reads the velocity of ITS point,
    v_body + omega x r (FRD)."""

    R_FRD = np.array([-0.232, 0.0, 0.105])

    def step(self, v=(0.0, 0.0, 0.0), fix=True, dvl=True, valid=True,
             tick=True, dvl_v=None, f=None, imu=True, lat=0.0, r=0.0):
        w, TC = self.w, self.TC
        self.t += 0.05
        self.yaw += r * 0.05
        v = np.asarray(v, float)
        self.p = self.p + v * 0.05
        if fix:
            w.on_nav_fix(TC._fix(list(self.p), self.yaw, self.t - lat))
        if imu:
            self.n_gyro += 1
            im = TC._imu(yaw=self.yaw, r=r, depth=0.8, t=self.t, f=f)
            im.gyro_int, im.t_gyro, im.n_gyro = (0.0, 0.0, self.yaw), \
                self.t, self.n_gyro
            im.depth_sp2_m, im.t_sp2 = float(self.p[2]), self.t
            w.on_vehicle_imu(im)
        w.on_telemetry(TC.Telemetry(armed=True, mode="MANUAL",
                                    conn=Conn.ONLINE, stamp=self.t))
        if dvl and w.dvl is not None:
            vb = rot_zyx(0.0, 0.0, self.yaw).T @ v
            vb = vb + np.cross(np.array([0.0, 0.0, r]), self.R_FRD)
            w.on_dvl(TD._sample(tuple(vb), t=self.t, valid=valid))
        if tick:
            w.tick()


def test_the_pose_takes_the_lever_arm_out_during_a_turn():
    """A turn in place at 20 deg/s with no tags: the DVL point moves ~8 cm/s
    on the shipped arm while the body origin stands still — integrating the
    DVL's own velocity would walk the pose ~16 cm in 2 s."""
    TC = _TC()
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = TC._worker(tmp)
        w.cfg.dvl = {"enabled": True, "mount_rpy_deg": [0.0, 0.0, 0.0],
                     "mount_t_flu": [-0.232, 0.0, -0.105]}
        w._setup_dvl()
        w.cfg.dvl_pose = {"bias_min_span_s": 5.0, "bias_min_n": 20}
        w.opts.dvl_pose = "control"
        w._setup_dvl_pose()
        assert w.dvp_control
        sim = LeverSim(w, TC)
        try:
            _ready(sim, w)
            sim.engage()
            for _ in range(10):
                sim.step()
            p0 = None
            for k in range(60):                    # 3 s turn, tags gone
                sim.step(fix=False, r=math.radians(20.0))
                m, h = w._dvp_last
                if h.get("state") == DVP.ST_DR:
                    xy = np.array(m["eta"][:2], float)
                    p0 = xy if p0 is None else p0
                    assert float(np.hypot(*(xy - p0))) < 0.03, (k, xy - p0)
            assert p0 is not None
        finally:
            sim.close()


# =============================================================================
# fix verification 2026-10-18 (round 4)
# =============================================================================
def test_the_pose_takes_its_z_rule_from_the_assemblers_z_source():
    for zs, want in (("tag", True), ("pressure", False)):
        with tempfile.TemporaryDirectory() as tmp:
            TC, w, bus, logs, events = _worker(tmp, z_source=zs)
            try:
                assert w.dvp.z_from_fix is want, zs
            finally:
                w.teardown()


def test_a_start_is_refused_while_the_tags_are_being_re_acquired():
    """A re-acquisition still agreeing would finish after the START, engaged
    — slewed or refused where it would have snapped (and the words say the
    tags are in view, held back, not 'no tag')."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp, z_source="tag")
        sim = PoseSim(w, TC)
        try:
            _ready(sim, w)
            for _ in range(20):                     # a dropout, disengaged
                sim.step(fix=False)
            sim.step()                              # one fix back
            assert w.dvp.pending_n(sim.t) >= 1
            assert w._dvp_last[1]["state"] == DVP.ST_DR
            w.on_enable(True)
            w.set_engaged(True)
            assert not w.engaged
            assert "being re-acquired" in w.reason \
                and "TAG FIX ON, re-acquiring" in w.reason \
                and "no tag" not in w.reason, w.reason
            # (the count is of fixes so far, not "n/5 agree": they are the
            # ones being compared — fix verification round 5)
            assert "must agree" in w.reason, w.reason
            for _ in range(10):                     # agreed: the tag state
                sim.step()
            sim.engage()
            assert w.engaged, w.reason
        finally:
            sim.close()


def test_a_runs_stale_debounce_starts_at_zero():
    """A run the stale debounce ended left the counter at its limit, and
    the NEXT run's first late frame ended it at once (every mission)."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        sim = PoseSim(w, TC)
        try:
            _ready(sim, w)
            w._stale_ticks = 8
            sim.engage()
            assert w._stale_ticks == 0
        finally:
            sim.close()


def test_a_pose_fault_does_not_end_a_policy_observe_run():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        try:
            w._dvp_last = (None, {"fault": "dvl pose: 20 s without a tag in "
                                           "view"})
            assert w._dvp_runtime_fault({}) != ""
            w._observe = True                       # (the LOW None bit)
            assert w._dvp_runtime_fault({}) == ""
        finally:
            w._observe = False
            w.teardown()


def test_a_shadow_observe_run_records_no_control_run_end():
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp, cli="shadow")
        sim = PoseSim(w, TC)
        try:
            _ready(sim, w)
            sim.engage()
            w.set_dvl_pose_tag(False)
            w._observe = True                        # (the LOW None bit)
            for _ in range(60):                      # the DVL silent 3 s
                sim.step(dvl=False)
            assert not [m for _l, m in logs if "would have ended" in m]
        finally:
            w._observe = False
            sim.close()


def test_a_sample_dvlaid_failed_on_is_not_dead_reckoned():
    """DvlAid raises half way through a sample (its newest sample already
    set, its flags not judged): the worker latches the DVL out — and that
    sample must not reach the track."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        sim = PoseSim(w, TC)
        try:
            _ready(sim, w)
            n0 = w.dvp.n_usable
            real = w.dvl.note_sample

            def half(sample, omega):
                real(sample, omega)                  # set, then...
                raise RuntimeError("synthetic")      # ...fail
            w.dvl.note_sample = half
            sim.t += 0.05
            w.on_dvl(TD._sample((0.0, 0.0, 0.0), t=sim.t))
            assert w._dvl_fault
            assert w.dvp.n_usable == n0, (w.dvp.n_usable, n0)
        finally:
            w.dvl.note_sample = real
            sim.close()


def test_a_shadow_pose_follows_the_starts_recalibration_too():
    """z_source pressure, a tilted map, 5 m from the last calibration: the
    START re-takes the assembler's offset — shadow's pose must follow it like
    control's, or it records a 25 cm z 'correction' (and a 'would have ended')
    a control run never makes."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp, cli="shadow",
                                           z_source="pressure")
        sim = OffsetSim(w, TC, tilt=0.05)
        try:
            _ready(sim, w)
            for _ in range(int(5.0 / 0.1 / 0.05)):
                sim.step(v=(0.0, 0.1, 0.0))
            for _ in range(20):
                sim.step()
            sim.engage()
            for _ in range(60):
                sim.step()
            assert not [c for c in w.dvp.corrections if c["engaged"]], \
                w.dvp.corrections[-2:]
            assert not [m for _l, m in logs if "would have ended" in m]
        finally:
            sim.close()


# =============================================================================
# fix verification round 6 (2026-10-18, wf_0e5903e0-a6d)
# =============================================================================
def test_a_start_is_refused_while_a_re_acquisition_past_a_correction_waits():
    """A sparse glimpse (fixes 1.5 s apart) of tags 30 cm off the dead
    reckoning: 1.2 s and more after the newest, START is still refused —
    that agreement can only end as a refusal once engaged (round 5 let the
    START through and the next fix ended the run); disengaged it snaps, and
    START then flies."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp, z_source="tag")
        sim = OffsetSim(w, TC)
        try:
            _ready(sim, w)
            for _ in range(20):                     # a dropout: DR
                sim.step(fix=False)
            sim.fix_off = np.array([0.30, 0.0, 0.0])
            for _ in range(3):                      # 3 fixes, 1.5 s apart
                sim.step()
                for _ in range(29):
                    sim.step(fix=False)
            p = w.dvp
            assert 0 < p._pending_live(sim.t) < 5 and p.pending_too_big()
            assert sim.t - p._pending_t[-1] > 1.2
            w.on_enable(True)
            w.set_engaged(True)
            assert not w.engaged
            assert "further from them than a correction may be" in w.reason, \
                w.reason
            for _ in range(10):                     # agreed, disengaged
                sim.step()
            assert w._dvp_last[1]["state"] in (DVP.ST_TAG, DVP.ST_BLEND)
            sim.engage()
            assert w.engaged, w.reason
        finally:
            sim.close()


def test_a_shadow_start_on_a_standing_fault_is_a_refused_start_not_a_runs_end():
    """SHADOW, TAG FIX OFF, the divergence abort standing while disengaged:
    a control START would have been REFUSED on it. The record says that —
    not 'would have ended' a run control would never have begun, and no
    second fault record of it as the run's end."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp, cli="shadow")
        sim = OffsetSim(w, TC)
        try:
            _ready(sim, w)
            w.set_dvl_pose_tag(False)
            assert not w.dvp.tag_on, w.reason
            sim.fix_off = np.array([0.30, 0.0, 0.0])
            for _ in range(40):
                sim.step()
            assert "diverged" in (w._dvp_last[1].get("fault") or "")
            n0 = len(w.dvp.faults)
            logs.clear()
            sim.engage()
            assert w.engaged, w.reason
            for _ in range(10):
                sim.step()
            said = [m for _l, m in logs]
            assert [m for m in said
                    if "a control START would have been refused here" in m], \
                said
            assert not [m for m in said
                        if "would have ended" in m and "diverged" in m], said
            new = [f for _t, f in w.dvp.faults[n0:]]
            assert any("a control START would have been refused" in f
                       for f in new), new
            assert not any(f.startswith("dvl pose diverged") for f in new), \
                new
        finally:
            sim.close()


# =============================================================================
# fix verification round 7 (2026-10-18, wf_0a38afc8-48f)
# =============================================================================
def test_a_shadow_start_on_a_lost_pose_is_a_refused_start_not_a_runs_end():
    """SHADOW, TAG FIX OFF, the DVL out past lost_s: a hole, the pose LOST.
    Control would have refused the START (no DVL pose); shadow starts — and
    says that, not 'would have ended' for the LOST stretch it started in."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp, cli="shadow")
        sim = PoseSim(w, TC)
        try:
            _ready(sim, w)
            w.set_dvl_pose_tag(False)
            assert not w.dvp.tag_on, w.reason
            for _ in range(40):                     # 2 s without the DVL
                sim.step(dvl=False)
            assert w._dvp_last[0] is None, w._dvp_last[1]
            logs.clear()
            sim.engage()
            assert w.engaged, w.reason
            for _ in range(20):
                sim.step(dvl=False)
            said = [m for _l, m in logs]
            assert [m for m in said
                    if "a control START would have been refused here" in m
                    and "no DVL pose" in m], said
            assert not [m for m in said if "would have ended" in m], said
        finally:
            sim.close()


# ============================================================ round 8: once
def test_a_one_shot_with_no_tags_goes_back_to_off_uncorrected():
    """TAG FIX pressed while OFF with no tag in view (2026-10-18): after
    oneshot_timeout_s the worker puts it back to OFF, says the pose was NOT
    corrected, and the toggle records why — the run flies on, on the DVL."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp, oneshot_timeout_s=2.0)
        sim = PoseSim(w, TC)
        st = []
        bus.mpc_status.connect(st.append)
        try:
            _ready(sim, w)
            sim.engage()
            w.set_dvl_pose_tag(False)
            assert not w.dvp.tag_on, w.reason
            for _ in range(10):
                sim.step(fix=False)
            w.set_dvl_pose_tag(True)
            assert w.dvp.tag_on and w.dvp.oneshot is not None
            for _ in range(int(1.5 / 0.05)):
                sim.step(fix=False)
            assert w.dvp.tag_on and st[-1].dvp_oneshot
            for _ in range(int(1.0 / 0.05)):
                sim.step(fix=False)
            assert not w.dvp.tag_on and w.dvp.oneshot is None
            assert w.dvp.toggles[-1]["auto"] == "timeout"
            assert any("NOT corrected" in m for _l, m in logs)
            assert any("TAG FIX OFF (no correction)" in " ".join(map(str, e))
                       for e in events)
            assert w.engaged and w._dvp_last[1]["state"] == DVP.ST_DR
        finally:
            sim.close()


def test_a_one_shot_whose_off_is_refused_stays_on_and_says_why_once():
    """The correction is in, but OFF's own refusal holds (no usable DVL
    sample): it stays ON — the tags are the pose — says why ONCE, and goes
    back to OFF by itself when the DVL is back."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        sim = PoseSim(w, TC)
        try:
            _ready(sim, w)
            w.set_dvl_pose_tag(False)
            assert not w.dvp.tag_on, w.reason
            for _ in range(10):
                sim.step()
            for _ in range(int(2.0 / 0.05)):          # the DVL gone: LOST
                sim.step(dvl=False)
            w.set_dvl_pose_tag(True)
            n0 = len(w.dvp.corrections)
            for _ in range(20):
                sim.step(dvl=False)
            assert len(w.dvp.corrections) > n0
            assert w.dvp.oneshot and w.dvp.oneshot["accepted"]
            assert w.dvp.tag_on
            waits = [m for _l, m in logs if "back to OFF waits" in m]
            assert len(waits) == 1 and "DVL" in waits[0], waits
            for _ in range(10):
                sim.step(dvl=False)
            assert len([m for _l, m in logs if "back to OFF waits" in m]) == 1
            for _ in range(40):
                sim.step()
                if not w.dvp.tag_on:
                    break
            assert not w.dvp.tag_on
            assert w.dvp.toggles[-1]["auto"] == "done"
        finally:
            sim.close()


# ============================================================ round 9: once
class _WobbleSim(PoseSim):
    """PoseSim whose fixes alternate +/- ``wob`` m across y (``wob`` None:
    clean): a view whose consecutive fixes never agree (> jump_m)."""

    def __init__(self, w, TC):
        super().__init__(w, TC)
        self.wob = None
        self.k = 0

    def step(self, v=(0.0, 0.0, 0.0), fix=True, **kw):
        if fix and self.wob is not None:
            self.k += 1
            s = 1.0 if self.k % 2 else -1.0
            p = list(self.p + np.asarray(v, float) * 0.05
                     + np.array([0.0, s * self.wob, 0.0]))
            self.w.on_nav_fix(self.TC._fix(p, 0.0, self.t + 0.05))
            fix = False
        super().step(v=v, fix=fix, **kw)


def test_a_one_shot_in_a_view_that_never_agrees_goes_off_at_its_timeout():
    """Fixes arriving all along that never agree held a due one-shot ON
    (OFF's refusal waited for them) for as long as they came — the docs
    promised OFF, uncorrected, after oneshot_timeout_s (fix verification
    wf_f143a3fa-24b, docs R8D-1). Due, it drops them."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp, oneshot_timeout_s=2.0)
        sim = _WobbleSim(w, TC)
        try:
            _ready(sim, w)
            sim.engage()
            w.set_dvl_pose_tag(False)
            assert not w.dvp.tag_on, w.reason
            for _ in range(10):
                sim.step()
            sim.wob = 0.06
            w.set_dvl_pose_tag(True)
            assert w.dvp.oneshot is not None
            for _ in range(int(2.5 / 0.05)):
                sim.step()
            assert not w.dvp.tag_on, w.dvp.pending_n(sim.t)
            assert w.dvp.toggles[-1]["auto"] == "timeout"
            assert any("NOT corrected" in m for _l, m in logs)
            assert w.engaged
        finally:
            sim.close()


def test_the_one_shots_own_off_keeps_the_reason_a_run_ended_on():
    """A one-shot still in progress at a disengage (E-STOP here) went OFF by
    itself while disengaged and wrote 'TAG FIX OFF (no correction)' over
    'disengaged: E-STOP' — the chip and the SENSORS row lost why the run
    ended (safety R8-3). The record lines stay; the reason is kept."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp, oneshot_timeout_s=2.0)
        sim = PoseSim(w, TC)
        try:
            _ready(sim, w)
            sim.engage()
            w.set_dvl_pose_tag(False)
            for _ in range(10):
                sim.step(fix=False)
            w.set_dvl_pose_tag(True)
            assert w.dvp.oneshot is not None
            w.estop()
            assert not w.engaged
            why = w.reason
            assert "E-STOP" in why, why
            for _ in range(int(3.0 / 0.05)):
                sim.step(fix=False)
            assert not w.dvp.tag_on                 # (it did go back OFF)
            assert w.reason == why, w.reason
        finally:
            sim.close()


def test_the_csv_tagfix_is_the_switch_the_rows_pose_was_computed_under():
    """The one-shot's own OFF comes after the step, in the same tick: the
    row read the switch after it — tagfix 0 on a row whose pose was the tag
    state (safety R8-4). The cell is the step's switch."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        sim = PoseSim(w, TC)
        try:
            _ready(sim, w)
            sim.engage()
            w.set_dvl_pose_tag(False)
            for _ in range(20):
                sim.step(v=(0.03, 0.0, 0.0))
            w.set_dvl_pose_tag(True)
            for _ in range(60):
                sim.step(v=(0.03, 0.0, 0.0))
                if not w.dvp.tag_on:
                    break
            assert not w.dvp.tag_on
            sim.step(v=(0.03, 0.0, 0.0))
            rows = [r for r in _rows(w) if r["engaged"] == "1"]
            bad = [r for r in rows if r["tagfix"] == "0"
                   and r["dvp_src"] in ("tag", "blend")]
            assert not bad, [(r["tagfix"], r["dvp_src"]) for r in bad]
            assert any(r["tagfix"] == "1" for r in rows)
        finally:
            sim.close()


# =============================================================================
# fix verification round 10 (wf_ce59a697-8bd)
# =============================================================================
def test_a_bench_one_shot_goes_off_with_its_own_reason_and_record_line():
    """Disengaged, TAG FIX OFF, the operator presses: the one-shot snaps its
    correction (or times out) and goes back to OFF by itself. Round 9 kept
    whatever reason was set — 'TAG FIX ONCE' stayed on the chip and the
    SENSORS row after the switch was OFF — and wrote that reason as the
    record line, so the next run's events.log said 'TAG FIX ONCE' twice and
    never the OFF (round 10, R10-1 / R9-3). The switch's own line now."""
    for fix, tail in ((True, " (correction in)"), (False, " (no correction)")):
        with tempfile.TemporaryDirectory() as tmp:
            TC, w, bus, logs, events = _worker(tmp, oneshot_timeout_s=2.0)
            st = []
            bus.mpc_status.connect(st.append)
            sim = PoseSim(w, TC)
            try:
                _ready(sim, w)
                w.set_dvl_pose_tag(False)
                for _ in range(10):
                    sim.step(fix=fix)
                w.set_dvl_pose_tag(True)
                assert w.dvp.oneshot is not None and w.reason == "TAG FIX ONCE"
                for _ in range(80):
                    sim.step(fix=fix)
                    if not w.dvp.tag_on:
                        break
                for _ in range(5):
                    sim.step(fix=fix)
                assert not w.dvp.tag_on and not w.engaged
                assert w.dvp.toggles[-1].get("auto") == (
                    "done" if fix else "timeout"), w.dvp.toggles[-1]
                assert w.reason == "TAG FIX OFF" + tail, w.reason
                assert st[-1].reason == "TAG FIX OFF" + tail, st[-1].reason
                lines = [ln[20:] for ln in w._pending_events]
                assert lines[-1] == "TAG FIX OFF" + tail, lines
                assert lines.count("TAG FIX ONCE") == 1, lines
            finally:
                sim.close()


def test_the_one_shots_own_off_after_an_estop_records_the_off():
    """...and a one-shot in progress at an E-STOP that times out while
    disengaged keeps 'disengaged: E-STOP' on the chip (round 9) but RECORDS
    its own OFF: round 9 wrote a second 'disengaged: E-STOP', stamped at
    the OFF's time, where 'TAG FIX OFF (no correction)' belonged (R9-3)."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp, oneshot_timeout_s=2.0)
        sim = PoseSim(w, TC)
        try:
            _ready(sim, w)
            sim.engage()
            w.set_dvl_pose_tag(False)
            for _ in range(10):
                sim.step(fix=False)
            w.set_dvl_pose_tag(True)
            w.estop()
            why = w.reason
            assert why.startswith("disengaged:") and "E-STOP" in why, why
            n0 = len(w._pending_events)
            for _ in range(int(3.0 / 0.05)):
                sim.step(fix=False)
            assert not w.dvp.tag_on
            assert w.reason == why, w.reason
            new = [ln[20:] for ln in w._pending_events[n0:]]
            assert new == ["TAG FIX OFF (no correction)"], new
        finally:
            sim.close()


# =============================================================================
# trust mode (2026-10-19, the operator's decision — dvl_pose.py TRUST MODE)
# =============================================================================
def test_trust_mode_puts_the_dvl_in_by_itself_without_a_proving_pass():
    """1019_211310 spent 21:11:45-21:12:07 on the proving pass ('0/60 moving
    pairs', 'motion along one body axis only') and OFF was refused until
    21:12:47 for the gyro bias. Trust: the switch goes IN at the first
    sample, and OFF is allowed as soon as the tags have anchored."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp, trust_dvl=True)
        sim = PoseSim(w, TC)
        try:
            assert w.dvp.cfg["trust_dvl"] and not w.dvl.use
            sim.step()
            assert w.dvl.use and not w.dvl.proven, w.reason
            assert "trust mode" in w.reason
            assert any("DVL IN by itself" in m for _l, m in logs)
            for _ in range(20):
                sim.step()
            assert w.dvp.anchored and w.dvp.head.bias_state != "ok"
            w.set_dvl_pose_tag(False)
            assert not w.dvp.tag_on, w.reason
            assert w.reason.startswith("TAG FIX OFF"), w.reason
            line = [m for _l, m in logs if "dead-reckoned pose" in m][-1]
            assert "TRUST MODE" in line and "strict gates" not in line
            # the button's own IN is the trust question too
            sim.engage()
            assert w._dvl_run_ok, w._dvl_run_why
        finally:
            sim.close()
    # ...and strict (the default) is what it was
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp)
        sim = PoseSim(w, TC)
        try:
            sim.step()
            assert not w.dvl.use
            w.set_dvl_use(True)
            assert not w.dvl.use and "not proven" in w.reason, w.reason
            line = [m for _l, m in logs if "dead-reckoned pose" in m][-1]
            assert "strict gates" in line and "TRUST MODE" not in line
        finally:
            sim.close()


def test_the_command_line_decides_trust_over_the_yaml():
    with tempfile.TemporaryDirectory() as tmp:
        TC = _TC()
        w, bus, pilots, logs = TC._worker(tmp)
        cfg = {"enabled": True}
        cfg.update(TD.MOUNT)
        w.cfg.dvl = cfg
        w._setup_dvl()
        w.cfg.dvl_pose = {"bias_min_span_s": 5.0, "bias_min_n": 20,
                          "trust_dvl": True}
        w.opts.dvl_pose = "control"
        w.opts.dvl_trust = False                   # --dvl-strict
        w._setup_dvl_pose()
        try:
            assert w.dvp is not None and w.dvp.cfg["trust_dvl"] is False
            assert not w._dvp_trust
        finally:
            w.teardown()


def test_trust_mode_keeps_dead_reckoning_through_a_dvlaid_hold():
    """A DVL reporting 10 cm/s against a still tag state: DvlAid holds it
    (a disagreement). Strict: no usable sample — the pose is HELD, then
    LOST, and an engaged run ends (1019_211310: 'no usable DVL for 1.88s
    (dis_fit)' 1.9 s into the run). Trust: the dead reckoning flies the DVL
    as reported; with tags in view the OFF divergence abort is what ends a
    pose that runs off."""
    for trust in (True, False):
        with tempfile.TemporaryDirectory() as tmp:
            TC, w, bus, logs, events = _worker(tmp, trust_dvl=trust)
            sim = PoseSim(w, TC)
            try:
                _ready(sim, w)
                w.set_dvl_pose_tag(False)
                assert not w.dvp.tag_on, w.reason
                for _ in range(10):
                    sim.step()
                n0 = w.dvp.n_unusable
                states = []
                for _ in range(40):
                    sim.step(v=(0.0, 0.0, 0.0), dvl_v=(0.10, 0.0, 0.0))
                    states.append(w._dvp_last[1].get("state"))
                why, code = w.dvl.disagree_hold(sim.t)
                assert code.startswith("dis_"), (why, code)   # the aid holds
                if trust:
                    assert all(s == DVP.ST_DR for s in states), states
                    assert w.dvp.n_unusable == n0
                    # ...and its verdict for the blend / panel / CSV is the
                    # trusted one (safety review 2026-10-19 #4/#5)
                    assert w.dvl.usable(sim.t)[2] == "ok"
                    assert w.dvl.state(sim.t, used=True)[0] == DV.ST_IN_USE
                else:
                    assert any(s in (DVP.ST_HELD, DVP.ST_LOST)
                               for s in states), states
                    assert w.dvp.n_unusable > n0
            finally:
                sim.close()


def test_the_gyro_bias_is_saved_for_the_next_session_and_seeded():
    """The bias fit (60 s of TAG FIX ON on the station) is written to
    data/dvl_pose_gyro_bias.json once ok; a session started within
    bias_seed_max_age_h seeds its heading with it ('saved' until its own
    fit is ok) — a restart no longer starts from nothing."""
    import json
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp, trust_dvl=True)
        sim = PoseSim(w, TC)
        try:
            _ready(sim, w)
            for _ in range(5):
                sim.step()
            path = w._dvp_bias_file()
            assert str(path).startswith(str(tmp)) and path.exists(), path
            rec = json.loads(path.read_text())
            assert abs(rec["bias_dps"]) < 0.05 and rec["n"] >= 20, rec
            assert w.dvp.head.seed is None
        finally:
            sim.close()
        TC2, w2, bus2, logs2, events2 = _worker(tmp, trust_dvl=True)
        sim2 = PoseSim(w2, TC2)
        try:
            assert w2.dvp.head.bias_state == "saved"
            assert w2.dvp.head.seed["dps"] == rec["bias_dps"]
            assert any("gyro bias seeded" in m for _l, m in logs2)
            sim2.step()
            m2 = w2._dvp_meta()
            assert m2["bias_seed"]["dps"] == rec["bias_dps"]
            assert m2["bias_seed_file"]["path"] == str(path)
        finally:
            sim2.close()
        # too old: not seeded
        rec["t_unix"] -= 24 * 3600.0
        path.write_text(json.dumps(rec))
        TC3, w3, bus3, logs3, events3 = _worker(tmp, trust_dvl=True)
        try:
            assert w3.dvp.head.bias_state == "none"
            assert any("not seeded" in m for _l, m in logs3)
        finally:
            w3.teardown()


def test_the_cli_flags_decide_trust_or_strict():
    from rov_gui.__main__ import build_parser
    p = build_parser()
    base = ["--source", "demo", "--mpc", "--dvl_pose"]
    assert p.parse_args(base).dvl_trust is None            # the YAML decides
    assert p.parse_args(base + ["--dvl-trust"]).dvl_trust is True
    assert p.parse_args(base + ["--dvl-strict"]).dvl_trust is False
    assert p.parse_args(base + ["--dvl-trust", "--dvl-strict"]).dvl_trust \
        is False                                              # the last wins


def test_trust_mode_lets_an_operators_out_stick():
    """The auto-IN is once: after the operator sets the switch OUT (the one
    manual way, under trust, to keep a gate-passing wrong DVL out of the
    dead reckoning) the next sample does not put it IN again (safety
    review 2026-10-19 #1). The operator's own IN needs no proof either."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp, trust_dvl=True)
        sim = PoseSim(w, TC)
        try:
            sim.step()
            assert w.dvl.use
            w.set_dvl_use(False)
            assert not w.dvl.use, w.reason
            for _ in range(10):
                sim.step()
            assert not w.dvl.use
            assert sum("DVL IN by itself" in m for _l, m in logs) == 1
            w.set_dvl_use(True)
            assert w.dvl.use and not w.dvl.proven, w.reason
        finally:
            sim.close()


def test_trust_mode_has_one_velocity_verdict_for_the_controller():
    """Under trust the blend (the assembler's u, v on a tag tick), the panel
    state and the CSV code are the trusted verdict the pose flies — with an
    unproven DVL the strict blend stayed on the tag finite difference while
    the dead reckoning flew the DVL, so the controller's velocity source
    flipped at every tag <-> DR transition (safety review 2026-10-19 #4)."""
    for trust in (True, False):
        with tempfile.TemporaryDirectory() as tmp:
            TC, w, bus, logs, events = _worker(tmp, trust_dvl=trust)
            sim = PoseSim(w, TC)
            try:
                if not trust:
                    w.set_dvl_use(True)         # refused: not proven
                for _ in range(40):
                    sim.step()
                tick = dict(w._dvl_tick)
                if trust:
                    assert w.dvl.trust and w.dvl.use and not w.dvl.proven
                    assert tick["code"] == "ok" and tick["w"] == 1.0, tick
                    assert tick["src"] == DV.VEL_DVL, tick
                    assert w.dvl.state(sim.t, used=True)[0] == DV.ST_IN_USE
                else:
                    assert not w.dvl.trust and not w.dvl.use
                    assert tick["src"] == DV.VEL_TAG_FD, tick
            finally:
                sim.close()


def test_the_bias_seed_is_trust_only_and_sanity_bounded():
    """Strict mode does not seed (its heading is what the reviews flew —
    safety review 2026-10-19 #6); trust mode refuses a saved bias fitted on
    another heading source, out of +-DVP_BIAS_SEED_MAX_DPS, or shorter
    than this session's own fit would be (#7)."""
    import json
    from pathlib import Path
    import time as _time

    def rec(**over):
        r = {"bias_dps": 0.03, "se_dps": 0.002, "span_s": 100.0, "n": 500,
             "heading": "raw_gyro", "t_unix": _time.time(), "saved": "x"}
        r.update(over)
        return r

    def build(tmp, r, trust=True):
        Path(tmp, "dvl_pose_gyro_bias.json").write_text(json.dumps(r))
        return _worker(tmp, trust_dvl=trust)

    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = build(tmp, rec(), trust=False)
        try:
            assert w.dvp.head.bias_state == "none"
            assert not any("seeded" in m for _l, m in logs)
        finally:
            w.teardown()
    for over, why in ((dict(heading="att_rate"), "fitted on heading"),
                      (dict(bias_dps=5.0), "outside"),
                      (dict(n=5), "less than this session's")):
        with tempfile.TemporaryDirectory() as tmp:
            TC, w, bus, logs, events = build(tmp, rec(**over))
            try:
                assert w.dvp.head.bias_state == "none", over
                assert any("not seeded" in m and why in m
                           for _l, m in logs), (over, [m for _l, m in logs
                                                       if "bias" in m])
            finally:
                w.teardown()
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = build(tmp, rec())
        try:
            assert w.dvp.head.bias_state == "saved"
            assert w.dvp.head.bias_info()["dps"] == 0.03
        finally:
            w.teardown()


def test_a_trusted_blend_with_a_strict_control_pose_is_refused():
    """(review 2026-10-19) YAML dvl.trust true + dvl_pose.trust_dvl false in
    control mode is a hybrid nobody designed — the pose would dead-reckon
    on the trusted verdict under its own strict gates. The pose is DISABLED
    at setup and says why; --dvl-trust sets both keys."""
    with tempfile.TemporaryDirectory() as tmp:
        TC = _TC()
        w, bus, pilots, logs = TC._worker(tmp)
        cfg = {"enabled": True, "trust": True}
        cfg.update(TD.MOUNT)
        w.cfg.dvl = cfg
        w._setup_dvl()
        try:
            assert w.dvl is not None and w.dvl.trust
            w.cfg.dvl_pose = {"bias_min_span_s": 5.0, "bias_min_n": 20,
                              "trust_dvl": False}
            w.opts.dvl_pose = "control"
            w._setup_dvl_pose()
            assert w.dvp is None and not w.dvp_control
            bad = [m for _l, m in logs if "dvl_pose.trust_dvl is false" in m]
            assert bad and "DISABLED" in bad[0], [m for _l, m in logs][-5:]
        finally:
            w.teardown()


def test_a_control_pose_under_trust_says_the_blend_is_trusted():
    """(review 2026-10-19) The reverse pair — dvl.trust false, a control
    pose under dvl_pose.trust_dvl — trusts the blend from setup on, and the
    log says so where it happens (the "dvl: built" line said strict)."""
    with tempfile.TemporaryDirectory() as tmp:
        TC, w, bus, logs, events = _worker(tmp, trust_dvl=True)
        try:
            assert w.dvp_control and w.dvl.trust
            built = [m for _l, m in logs if m.startswith("dvl: built")]
            assert built and "TRUST (" not in built[0]
            said = [m for _l, m in logs
                    if m.startswith("dvl: TRUST (dvl_pose.trust_dvl")]
            assert said and "from now on" in said[0], \
                [m for _l, m in logs if m.startswith("dvl:")][:6]
        finally:
            w.teardown()
