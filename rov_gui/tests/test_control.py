#!/usr/bin/env python3
"""
test_control.py — the closed-loop MPC stack, with no hardware and no acados.

    ~/miniforge3/envs/rovgui-pose/bin/python rov_gui/tests/test_control.py

Everything here runs on the Qt ``offscreen`` platform and a STUB controller
(the same interface as HwDobMpc, none of the mathematics), so the suite is
fast and needs neither casadi nor a camera. What it pins down:

* the PnP chain recovers a synthetic pose (frames: map -> camera -> body ->
  NED) to millimetres, including the single-tag IPPE case;
* the square-reference copy has not drifted from the simulator's values;
* the state assembler's signs (NED/FRD, gravity, transport term);
* the engage preconditions each refuse for their own stated reason;
* pilot input while engaged is a takeover (the window yields, exactly once);
* the CSV is a superset of the sim's 9-column schema and carries a meta
  sidecar.

The real solver's health is covered by ``rov_gui/control/smoke.py`` (P0) and
the demo closed loop (`--source demo --mpc`), not here.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from rov_gui.bus import DataBus
from rov_gui.control.allocation import axes_to_wrench, wrench_to_axes
from rov_gui.control.geometry import (NavConfig, WALL_PRESETS,
                                      R_flu_cam_from_xyaxes, S_FLU_FRD)
from rov_gui.control.reference import (make_square_ref, make_square_ref_world,
                                       square_setpoint)
from rov_gui.control.state_assembler import G_NED, StateAssembler, rot_zyx
from rov_gui.control.tagnav import (Detection, TagMap, TagNav,
                                    tag_object_points)
from rov_gui.control.workers import CSV_HEADER, MpcWorker
from rov_gui.qt import QtWidgets, QTimer
from rov_gui.state import Conn, NavFix, PilotInput, Telemetry, VehicleImu, now

ROOT = Path(__file__).resolve().parents[2]

_APP = None


def _app():
    global _APP
    _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return _APP


# =============================================================================
# tag navigation — synthetic PnP round trips
# =============================================================================
K_TEST = np.array([[770.0, 0.0, 480.0], [0.0, 770.0, 270.0], [0.0, 0.0, 1.0]])


def _project(obj_pts_map, R_cm, t_cm):
    import cv2
    rvec, _ = cv2.Rodrigues(R_cm)
    px, _ = cv2.projectPoints(obj_pts_map.astype(np.float64), rvec,
                              t_cm.reshape(3, 1), K_TEST, None)
    return px.reshape(-1, 2).astype(np.float32)


def _wall_nav(tag_map) -> TagNav:
    """A TagNav with the REAL forward-level extrinsic and the wall preset,
    so the test exercises the exact frame chain the pool run will."""
    cfg = NavConfig()                    # frozen c3_payload_frames defaults
    R_bc, t_bc = cfg.R_t_frd_cam()
    return TagNav(tag_map, 0.170, R_bc, t_bc, WALL_PRESETS["x_into_wall"],
                  max_reproj_px=3.0)


def _detections_for_body_pose(nav: TagNav, tag_map: TagMap, p_ned, R_ned_body):
    """Fabricate pixel corners for a vehicle at (p_ned, R_ned_body), by
    composing the same homogeneous chain TagNav inverts."""
    R_nm = nav.R_nm
    R_map_body = R_nm.T @ np.asarray(R_ned_body, float)
    t_map_body = R_nm.T @ np.asarray(p_ned, float)
    T_map_body = np.eye(4)
    T_map_body[:3, :3] = R_map_body
    T_map_body[:3, 3] = t_map_body
    T_body_cam = np.eye(4)
    T_body_cam[:3, :3] = nav.R_bc
    T_body_cam[:3, 3] = nav.t_bc
    T_map_cam = T_map_body @ T_body_cam
    T_cam_map = np.linalg.inv(T_map_cam)
    dets = []
    for tid, (R_mt, t_mt) in tag_map.poses.items():
        obj_map = (R_mt @ nav.obj_tag.T).T + t_mt
        px = _project(obj_map, T_cam_map[:3, :3], T_cam_map[:3, 3])
        dets.append(Detection(tid, px))
    return dets, T_cam_map


def test_multi_tag_pnp_recovers_pose():
    # Two wall tags, side by side, world = first tag's frame.
    tm = TagMap({25: (np.eye(3), np.zeros(3)),
                 26: (np.eye(3), np.array([0.6, 0.0, 0.0]))}, 25)
    nav = _wall_nav(tm)
    # Vehicle 2 m in front of the wall (x_into_wall => x = -2), level, small yaw.
    p_true = np.array([-2.0, 0.3, 0.4])
    R_true = rot_zyx(0.02, -0.03, 0.1)
    dets, _ = _detections_for_body_pose(nav, tm, p_true, R_true)
    sol = nav.solve(dets, K_TEST, None)
    assert sol is not None and sol.n_tags == 2
    assert np.linalg.norm(sol.p_ned - p_true) < 2e-3, sol.p_ned
    assert np.max(np.abs(sol.R_ned_body - R_true)) < 1e-3
    assert sol.reproj_rms_px < 0.5


def test_single_tag_ippe_recovers_pose_with_hint():
    tm = TagMap.single(25)
    nav = _wall_nav(tm)
    p_true = np.array([-1.8, -0.2, 0.5])
    R_true = rot_zyx(0.01, 0.02, -0.15)
    dets, _ = _detections_for_body_pose(nav, tm, p_true, R_true)
    sol = nav.solve(dets, K_TEST, None, rp_hint=(0.01, 0.02))
    assert sol is not None and sol.n_tags == 1
    assert np.linalg.norm(sol.p_ned - p_true) < 5e-3, sol.p_ned
    assert abs(math.atan2(sol.R_ned_body[1, 0], sol.R_ned_body[0, 0])
               - (-0.15)) < 5e-3


def test_single_tag_gravity_gate_rejects_wrong_attitude():
    """If the autopilot says we are level but the (only) PnP solution implies
    a 30-degree tilt, the frame must be rejected, not swallowed."""
    tm = TagMap.single(25)
    nav = _wall_nav(tm)
    p_true = np.array([-1.5, 0.0, 0.5])
    R_true = rot_zyx(0.5, 0.0, 0.0)       # actually rolled 28.6 deg
    dets, _ = _detections_for_body_pose(nav, tm, p_true, R_true)
    sol = nav.solve(dets, K_TEST, None, rp_hint=(0.0, 0.0))   # claims level
    assert sol is None


def test_floor_map_loads_and_anchor_is_identity():
    tm = TagMap.load(ROOT / "config" / "tag_map.yaml")
    assert len(tm) == 47 and tm.anchor_id == 25
    R, t = tm.poses[25]
    assert np.allclose(R, np.eye(3), atol=1e-6)
    assert np.allclose(t, 0.0, atol=1e-6)


def test_extrinsic_matches_payload_json():
    """NavConfig must pick the measured extrinsic up from the sim's JSON, and
    the optical-axis direction must come out +x-forward in FRD."""
    cfg = NavConfig()
    R_bc, t_bc = cfg.R_t_frd_cam()
    z_cv_in_body = R_bc[:, 2]              # optical forward, FRD body coords
    assert z_cv_in_body[0] > 0.999, z_cv_in_body
    assert abs(t_bc[0] - 0.23949) < 1e-6 and abs(t_bc[1] + 0.00547) < 1e-6


# =============================================================================
# reference — the copy must not drift from the simulator
# =============================================================================
def test_square_setpoint_matches_sim_values():
    # Hand-computed from the run_compare definition (CCW, origin corner).
    (x, y), (tx, ty) = square_setpoint(0.0, 1.0, 0.1)
    assert (x, y, tx, ty) == (0.0, 0.0, 1.0, 0.0)
    (x, y), (tx, ty) = square_setpoint(15.0, 1.0, 0.1)   # s=1.5: up the right edge
    assert abs(x - 1.0) < 1e-12 and abs(y - 0.5) < 1e-12 and (tx, ty) == (0.0, 1.0)
    (x, y), (tx, ty) = square_setpoint(25.0, 1.0, 0.1)   # s=2.5: back along the top
    assert abs(x - 0.5) < 1e-12 and abs(y - 1.0) < 1e-12 and (tx, ty) == (-1.0, 0.0)
    (x, y), (tx, ty) = square_setpoint(35.0, 1.0, 0.1)   # s=3.5: down the left edge
    assert abs(x - 0.0) < 1e-12 and abs(y - 0.5) < 1e-12 and (tx, ty) == (0.0, -1.0)


def test_square_ref_shapes_and_velocity():
    ref = make_square_ref(1.0, 0.1, -0.5, True, math.radians(60), 0.05, 60.0)
    ts = np.arange(0.0, 3.05, 0.05)
    p, yaw, v, r = ref(ts)
    assert p.shape == (3, ts.size) and v.shape == (3, ts.size)
    assert np.allclose(np.linalg.norm(v[:2], axis=0), 0.1)
    assert np.allclose(p[2], -0.5)
    # heading follows the first edge: yaw ~ 0 while moving +x
    assert abs(yaw[0]) < 1e-9


def test_square_ref_world_rotation_and_fixed_yaw():
    ref = make_square_ref_world(1.0, 0.1, -0.5, False, 1.0, 0.05, 60.0,
                                origin_xy=(2.0, 1.0), rot_rad=math.pi / 2,
                                yaw_fixed=0.7)
    p, yaw, v, r = ref(np.array([0.0, 5.0]))
    # t=0: origin corner; t=5 (s=0.5): half way along the first edge, which
    # rotation pi/2 turns from +x into +y.
    assert np.allclose(p[:2, 0], [2.0, 1.0], atol=1e-12)
    assert np.allclose(p[:2, 1], [2.0, 1.5], atol=1e-9)
    assert np.allclose(yaw, 0.7) and np.allclose(r, 0.0)


# =============================================================================
# allocation — sign map and cap
# =============================================================================
def test_wrench_axis_signs_and_cap():
    gains = {"surge_n": 60.0, "sway_n": 60.0, "heave_n": 60.0, "yaw_nm": 20.0}
    cmd = wrench_to_axes([30.0, -15.0, 30.0, 5.0, 5.0, 10.0], gains, cap=0.5)
    assert cmd.surge == 0.5                    # +X -> +surge
    assert cmd.sway == -0.25                   # -Y -> port
    assert cmd.heave == -0.5                   # +Z (down) -> heave DOWN
    assert cmd.yaw == 0.5                      # +N -> +yaw, and capped
    assert cmd.source == "mpc"
    back = axes_to_wrench(cmd, gains)
    assert back[0] == 30.0 and back[2] == 30.0 and back[3] == 0.0 == back[4]
    assert back[5] == 10.0


# =============================================================================
# state assembler — frames, gravity, staleness
# =============================================================================
def _fix(p, yaw, t_cap):
    R = rot_zyx(0.0, 0.0, yaw)
    return NavFix(t_capture=t_cap, n_tags=2, tag_ids=(25, 26),
                  p_ned=tuple(p), R_ned_body=tuple(R.ravel()), yaw_ned=yaw,
                  reproj_rms_px=0.3, conn=Conn.ONLINE, stamp=t_cap)


def _imu(roll=0.0, pitch=0.0, yaw=0.0, r=0.0, depth=0.8, t=None,
         f=None):
    t = now() if t is None else t
    f = (0.0, 0.0, -9.80665) if f is None else f
    return VehicleImu(roll=roll, pitch=pitch, yaw=yaw, p=0.0, q=0.0, r=r,
                      ax=f[0], ay=f[1], az=f[2], depth_m=depth,
                      t_att=t, t_imu=t, t_baro=t, conn=Conn.ONLINE, stamp=t)


def test_assembler_velocity_rotates_into_body():
    asm = StateAssembler(z_source="tag", vel_lp_alpha=1.0, nudot_source="fd")
    t0 = now()
    # vehicle yawed 90 deg (nose along +y_ned), moving +y_ned at 0.2 m/s
    yaw = math.pi / 2
    m1, h1 = asm.step(_fix([0.0, 0.0, 0.5], yaw, t0), _imu(yaw=yaw, t=t0),
                      t0 + 0.01, 0.05)
    assert m1 is not None and h1["ok"]
    m2, _ = asm.step(_fix([0.0, 0.02, 0.5], yaw, t0 + 0.1),
                     _imu(yaw=yaw, t=t0 + 0.1), t0 + 0.11, 0.05)
    # world +y motion with nose along +y => pure SURGE in the body frame
    assert abs(m2["nu"][0] - 0.2) < 1e-6, m2["nu"]
    assert abs(m2["nu"][1]) < 1e-9
    assert abs(m2["eta"][5] - yaw) < 1e-9


def test_assembler_pressure_z_and_gravity():
    asm = StateAssembler(z_source="pressure", nudot_source="imu")
    t0 = now()
    fix = _fix([-2.0, 0.0, 0.42], 0.0, t0)
    imu = _imu(depth=0.80, t=t0)
    m, h = asm.step(fix, imu, t0 + 0.01, 0.05)
    assert h["z_src"] == "pressure"
    assert abs(m["eta"][2] - 0.42) < 1e-9      # anchored to the tag world
    # deeper by 0.10 m on the pressure sensor moves z by exactly 0.10
    m2, _ = asm.step(fix, _imu(depth=0.90, t=t0 + 0.05), t0 + 0.06, 0.05)
    assert abs(m2["eta"][2] - 0.52) < 1e-9
    # level + stationary: specific force (0,0,-g) must cancel to ~zero accel
    assert np.linalg.norm(m2["nudot"][:3]) < 1e-6


def test_assembler_stale_gates():
    asm = StateAssembler()
    t0 = now()
    m, h = asm.step(_fix([0, 0, 0.5], 0.0, t0 - 2.0), _imu(t=t0), t0, 0.05)
    assert m is None and "stale" in h["why"]
    m, h = asm.step(_fix([0, 0, 0.5], 0.0, t0), _imu(t=t0 - 2.0), t0, 0.05)
    assert m is None and "stale" in h["why"]


# =============================================================================
# MpcWorker — engage discipline, CSV, geofence (stub controller, no acados)
# =============================================================================
class StubCtrl:
    """HwDobMpc's surface with a P-controller inside. Enough for the worker's
    logic to run; no casadi anywhere near it."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.mode = cfg.mode
        self.solver_kind = "stub"
        self.realtime_ok = True
        self.probe_ms = 0.1
        self.build_s = 0.0
        self.scenario = None
        self.w_hat = np.zeros(6)
        self.n_fail = 0
        self._target = (np.zeros(3), 0.0)

    def set_target_ned(self, p, yaw, v_ned=None, r_ned=0.0):
        self._target = (np.asarray(p, float).copy(), float(yaw))

    def set_square_ned(self, square, origin_xy, yaw_fixed, depth):
        self.scenario = {"kind": "square", "size": float(square["size"]),
                         "speed": float(square["speed"]),
                         "laps": int(square["laps"]),
                         "heading_follow": bool(square["heading_follow"]),
                         "origin_ned": [float(origin_xy[0]), float(origin_xy[1])],
                         "rot_deg": 0.0, "depth_ned": depth,
                         "yaw_fixed_ned_deg": math.degrees(yaw_fixed),
                         "yaw_rate_deg_s": 60.0,
                         "T_run_s": square["laps"] * 4 * square["size"]
                         / square["speed"]}
        return self.scenario

    def ref_ned_at(self, t):
        return self._target[0].copy(), self._target[1]

    def step(self, eta, nu, nudot, t):
        e = self._target[0] - np.asarray(eta[:3])
        u = np.zeros(6)
        u[:3] = 20.0 * e
        return u, {"w_hat": self.w_hat, "solve_ms": 0.2, "status": 0,
                   "n_fail": self.n_fail, "nis": 1.0}

    def note_applied(self, tau):
        self.applied = tau

    def reset(self):
        pass

    def meta(self):
        return {"type": self.mode, "solver": "stub"}


# Tests carry their OWN nav/mpc configs, decoupled from the repo's live
# mission files (config/hw_*.yaml change with every pool campaign — a test
# that reads them breaks whenever the geofence or square is retuned).
_TEST_NAV_YAML = """\
geometry: wall
wall_tag_id: 25
nav_source: main
datum: map
# ENGAGE-datum-relative box (the worker re-zeros at engage, so the vehicle
# is at the origin by definition when a run begins).
geofence_ned:
  x: [-2.0, 2.0]
  y: [-2.0, 2.0]
  z: [-1.0, 2.0]
  margin: 0.4
"""
_TEST_MPC_YAML = """\
mode: dobmpc
square:
  size: 1.0
  speed: 0.12
  laps: 3
  heading_follow: false
  origin: current
"""


def _test_opts(tmp) -> type:
    nav_path = Path(tmp) / "nav.yaml"
    mpc_path = Path(tmp) / "mpc.yaml"
    nav_path.write_text(_TEST_NAV_YAML)
    mpc_path.write_text(_TEST_MPC_YAML)
    # NOTE: the class attribute `mpc = True` shadows any same-named local in
    # the class body's scope — the paths are bound to distinct names first.
    nav_s, mpc_s = str(nav_path), str(mpc_path)

    class Opts:
        source = "demo"
        mpc = True
        nav_config = nav_s
        mpc_config = mpc_s
        nav_geometry = None
        mpc_mode = None
        rov_model = None
    return Opts


def _worker(tmp) -> tuple[MpcWorker, DataBus, list, list]:
    _app()
    bus = DataBus()
    w = MpcWorker(bus, _test_opts(tmp)(), controller_factory=StubCtrl)
    w.setup()
    w.cfg.log_dir = str(tmp)
    w.cfg.engage["warmup_s"] = 0.1
    pilots, logs = [], []
    bus.cmd_pilot.connect(pilots.append)
    bus.log.connect(lambda lvl, msg: logs.append((lvl, msg)))
    return w, bus, pilots, logs


def _feed_good_state(w):
    t = now()
    w.on_nav_fix(_fix([-2.0, 0.0, 0.8], 0.0, t))
    w.on_vehicle_imu(_imu(depth=0.8, t=t))
    w.on_telemetry(Telemetry(armed=True, mode="MANUAL", conn=Conn.ONLINE))


def test_engage_preconditions_each_refuse():
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _worker(tmp)
        # 1) COMMAND ENABLE off
        _feed_good_state(w)
        w.set_engaged(True)
        assert not w.engaged and "ENABLE" in w.reason
        # 2) disarmed
        w.on_enable(True)
        w.on_telemetry(Telemetry(armed=False, mode="MANUAL", conn=Conn.ONLINE))
        w.set_engaged(True)
        assert not w.engaged and "armed" in w.reason
        # 3) wrong flight mode
        w.on_telemetry(Telemetry(armed=True, mode="STABILIZE", conn=Conn.ONLINE))
        w.set_engaged(True)
        assert not w.engaged and "MANUAL" in w.reason
        # 4) stale fix
        w.on_telemetry(Telemetry(armed=True, mode="MANUAL", conn=Conn.ONLINE))
        w.fix = _fix([-2.0, 0.0, 0.8], 0.0, now() - 3.0)
        w.set_engaged(True)
        assert not w.engaged and "stale" in w.reason
        # 5) all good -> engaged, DP hold — and the engage pose becomes the
        # datum: ANY tag-frame position engages at (0,0,0), by design (the
        # geofence is relative to it; there is no absolute-position gate).
        _feed_good_state(w)
        w.on_nav_fix(_fix([5.0, 0.3, 0.8], 0.7, now()))
        w.set_engaged(True)
        assert w.engaged and w._csv is not None
        assert w._datum is not None
        assert abs(w._datum["p0"][0] - 5.0) < 1e-9
        assert abs(w._datum["yaw0"] - 0.7) < 1e-9
        assert np.linalg.norm(w._eta[:3]) < 1e-9      # datum frame: origin
        assert abs(w._eta[5]) < 1e-9                  # datum frame: yaw 0
        w.teardown()


def test_tick_emits_pilot_and_csv_superset():
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _worker(tmp)
        w.on_enable(True)
        _feed_good_state(w)
        w.set_engaged(True)
        assert w.engaged
        for _ in range(6):
            _feed_good_state(w)
            w.tick()
        app = _app()
        deadline = time.monotonic() + 2.0
        while not pilots and time.monotonic() < deadline:
            app.processEvents()
        assert pilots, "engaged ticks emitted no cmd_pilot"
        assert all(p.source == "mpc" for p in pilots)
        # trajectory start after warm-up
        w.set_traj(True)
        assert w.traj_on and w.ctrl.scenario is not None
        _feed_good_state(w)
        w.tick()
        path = w._csv_path
        w.disengage("test over")
        assert not w.engaged
        # CSV: sim 9-column prefix + extensions, plus the meta sidecar
        text = Path(path).read_text().splitlines()
        assert text[0] == CSV_HEADER.strip()
        assert text[0].startswith("t,px,py,pz,rx,ry,yaw_deg,pitch_deg,lap")
        assert len(text) >= 2
        first = text[1].split(",")
        # datum frame: the run STARTS at (0,0) whatever the tag-frame pose was
        assert abs(float(first[1])) < 0.05              # px (FLU x = NED x)
        assert abs(float(first[2])) < 0.05
        meta = json.loads(Path(path).with_suffix(".meta.json").read_text())
        assert meta["schema_version"] == 1
        assert meta["controller"]["solver"] == "stub"
        assert meta["hardware"]["axis_gain_provenance"].startswith("[예측]")
        # provenance back to absolute coordinates
        d = meta["hardware"]["datum_tag_frame"]
        assert d is not None and abs(d["p0"][0] - (-2.0)) < 1e-6
        w.teardown()


def test_start_traj_refuses_square_outside_geofence():
    """A square too big for the fence must refuse BEFORE the reference is
    armed, not be discovered by the mid-run fence abort (review 2026-08-12).
    In the datum frame the vehicle starts at (0,0), so a 5 m square against
    the ±2 m test fence is the spill case."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _worker(tmp)
        w.on_enable(True)
        _feed_good_state(w)
        w.set_engaged(True)
        assert w.engaged
        w._warmup_left = 0
        w.set_scenario({"size": 5.0})
        w.set_traj(True)
        assert not w.traj_on
        assert any("outside the geofence" in m for _l, m in logs)
        # shrink the square -> the same START must now succeed (nothing
        # latched by the refusal)
        w.set_scenario({"size": 0.5})
        w.set_traj(True)
        assert w.traj_on
        w.teardown()


def test_restart_after_stop_traj_and_after_estop():
    """The two field sequences that must not wedge (reported live
    2026-08-12): STOP TRAJ -> START again, and E-STOP -> re-enable ->
    START. Each engagement re-zeros the datum at the CURRENT pose."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _worker(tmp)
        w.on_enable(True)
        _feed_good_state(w)
        w.start_mission()
        assert w.engaged
        for _ in range(6):
            _feed_good_state(w)
            w.tick()
        assert w.traj_on
        # --- STOP TRAJ, then START again (still engaged, already warm)
        w.set_traj(False)
        assert w.engaged and not w.traj_on
        w.start_mission()
        assert w.traj_on, "START after STOP TRAJ did not restart the square"
        # --- E-STOP kills everything; re-enable + START must fully recover
        w.estop()
        assert not w.engaged and not w.traj_on
        w.on_enable(False)                 # the sink/UI drop ENABLE on E-STOP
        w.on_enable(True)
        # vehicle moved meanwhile: the NEW engage re-zeros the datum there
        t = now()
        w.on_nav_fix(_fix([-1.0, 0.5, 0.6], 0.4, t))
        w.on_vehicle_imu(_imu(depth=0.6, t=t))
        w.on_telemetry(Telemetry(armed=True, mode="MANUAL", conn=Conn.ONLINE))
        w.start_mission()
        assert w.engaged, f"re-engage after E-STOP failed: {w.reason}"
        assert abs(w._datum["p0"][0] - (-1.0)) < 1e-9
        for _ in range(6):
            t = now()
            w.on_nav_fix(_fix([-1.0, 0.5, 0.6], 0.4, t))
            w.on_vehicle_imu(_imu(depth=0.6, t=t))
            w.tick()
        assert w.traj_on, "square did not restart after E-STOP recovery"
        w.teardown()


def test_runtime_geofence_and_tag_loss_disengage():
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _worker(tmp)
        w.on_enable(True)
        _feed_good_state(w)
        w.set_engaged(True)
        assert w.engaged
        # geofence: teleport outside the margin
        w.on_nav_fix(_fix([5.0, 0.0, 0.8], 0.0, now()))
        w.on_vehicle_imu(_imu(depth=0.8))
        w.tick()
        assert not w.engaged and "geofence" in w.reason
        # re-engage, then lose the tag
        _feed_good_state(w)
        w.set_engaged(True)
        assert w.engaged
        w.fix = _fix([-2.0, 0.0, 0.8], 0.0, now() - 1.0)   # stale
        w.tick()
        assert not w.engaged and "stale" in w.reason
        w.teardown()


def test_disarm_mid_run_disengages():
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _worker(tmp)
        w.on_enable(True)
        _feed_good_state(w)
        w.set_engaged(True)
        assert w.engaged
        w.on_telemetry(Telemetry(armed=False, mode="MANUAL", conn=Conn.ONLINE))
        _feed = _fix([-2.0, 0.0, 0.8], 0.0, now())
        w.on_nav_fix(_feed)
        w.on_vehicle_imu(_imu(depth=0.8))
        w.tick()
        assert not w.engaged and "disarmed" in w.reason
        w.teardown()


def test_start_mission_flies_after_warmup():
    """The one-button flow: START = engage + auto-square once warm, with the
    CSV open from the engage. One refusal must not retry at 20 Hz."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _worker(tmp)
        w.on_enable(True)
        _feed_good_state(w)
        w.start_mission()
        assert w.engaged and not w.traj_on and w._csv is not None
        for _ in range(6):                 # warmup_s 0.1 @ 20 Hz = 2 ticks
            _feed_good_state(w)
            w.tick()
        assert w.traj_on, "square did not start by itself after warm-up"
        assert w.ctrl.scenario is not None
        w.teardown()


def test_tagnav_multi_source_toggles():
    """Two feeds, one worker: the calibrated feed localizes, the other is
    overlay-only; each toggle drives its own mailbox's wanted gate and
    toggle-off emits one explicit CLEAR."""
    from rov_gui.bus import RgbdMailbox
    from rov_gui.control.workers import TagNavWorker
    import numpy as np

    _app()
    bus = DataBus()
    mb_main, mb_second = RgbdMailbox(), RgbdMailbox()
    w = TagNavWorker(bus, {"main": mb_main, "second": mb_second},
                     types.SimpleNamespace(), nav_cfg=NavConfig())
    w.setup()
    overlays = []
    bus.tag_overlay.connect(overlays.append)
    fixes = []
    bus.nav_fix.connect(fixes.append)
    # defaults: the localizing feed ON, the extra OFF
    assert mb_main.wanted() and not mb_second.wanted()
    w.set_source_enabled("second", True)
    assert mb_second.wanted()
    # a frame with no tags in it still produces an (empty) overlay for its
    # panel, marked non-localizing for the uncalibrated feed
    gray = np.zeros((120, 160), np.uint8)
    mb_second.put(gray, None, None, now(), 0.0)
    w.tick()
    app = _app()
    deadline = time.monotonic() + 2.0
    while not overlays and time.monotonic() < deadline:
        app.processEvents()
    assert overlays and overlays[-1].panel == "second"
    assert overlays[-1].localizes is False and overlays[-1].quads == ()
    assert not fixes                       # overlay-only feed never localizes
    # toggle-off: wanted drops and one CLEAR (enabled=False) goes out
    overlays.clear()
    w.set_source_enabled("second", False)
    while not overlays and time.monotonic() < deadline:
        app.processEvents()
    assert not mb_second.wanted()
    assert overlays and overlays[-1].enabled is False
    w.teardown()


def test_floor_map_multi_tag_with_tilted_second_cam():
    """The tagslam-map regime: joint PnP over REAL tags from
    config/tag_map.yaml, seen by the default-RGB camera tilted 30 deg down —
    the 2026-08-13 mission. World = anchor tag 25's frame, exactly like the
    gantry runs."""
    tm = TagMap.load(ROOT / "config" / "tag_map.yaml")
    cfg = NavConfig()
    cfg.second_cam = dict(cfg.second_cam)
    cfg.second_cam["tilt_deg"] = 30.0
    R_bc, t_bc = cfg.R_t_frd_cam("second")
    # tilt sanity: optical forward dips 30 deg below level in FRD
    z_cv = R_bc[:, 2]
    assert abs(z_cv[2] - math.sin(math.radians(30))) < 1e-9   # +down in FRD
    assert abs(z_cv[0] - math.cos(math.radians(30))) < 1e-9
    nav = TagNav(tm, 0.170, R_bc, t_bc, np.eye(3),   # floor map: NED-like as-is
                 max_reproj_px=3.0, min_tags=2)
    # hover 0.8 m above the floor near tags 8/25 (map z is DOWN; above = -z),
    # pitched level, yawed 0.3 — the camera looks down-forward at the floor.
    p_true = np.array([-0.6, 0.2, -0.8])
    R_true = rot_zyx(0.02, -0.01, 0.3)
    dets, _ = _detections_for_body_pose(nav, tm, p_true, R_true)
    # keep only tags that actually project in front of the camera and near
    # the image (the synthetic projector does not cull) — take 4 nearby ids
    keep = {8, 25, 5, 4}
    dets = [d for d in dets if d.tag_id in keep]
    assert len(dets) >= 2
    sol = nav.solve(dets, K_TEST, None)
    assert sol is not None and sol.n_tags == len(dets)
    assert np.linalg.norm(sol.p_ned - p_true) < 5e-3, sol.p_ned
    assert abs(math.atan2(sol.R_ned_body[1, 0], sol.R_ned_body[0, 0])
               - 0.3) < 5e-3
    # min_tags=2 really gates: a single mapped detection is refused
    assert nav.solve(dets[:1], K_TEST, None) is None
    assert "unique tags" in nav.last_reject


def test_pid_controller_pushes_toward_target():
    """HwPid: sim gain structure in the datum NED frame — sign checks plus
    the slew limit on the first tick."""
    from rov_gui.control.geometry import MpcConfig
    from rov_gui.control.pid import HwPid

    cfg = MpcConfig()
    pid = HwPid(cfg, log=lambda m: None)
    pid.set_target_ned([1.0, 0.0, 0.0], 0.5)
    eta = np.zeros(6)
    nu = np.zeros(6)
    u, info = pid.step(eta, nu, np.zeros(6), 0.0)
    assert u[0] > 0.0                      # +x error -> +X force
    assert abs(u[0] - pid.slew * pid.dt) < 1e-9   # first tick is slew-limited
    assert u[5] > 0.0                      # +yaw error -> +N
    assert u[3] == 0.0 == u[4]             # K, M never commanded
    # integral winds only inside the gate, and stays clamped
    pid.reset()
    pid.set_target_ned([0.1, 0.0, 0.0], 0.0)
    for _ in range(200):
        u, _ = pid.step(eta, nu, np.zeros(6), 0.0)
    assert pid.ki[0] * pid._i_xyz[0] <= pid.i_max[0] + 1e-9
    # square sampler: ref moves along the first edge in the datum frame
    pid.reset()
    scen = pid.set_square_ned({"size": 0.2, "speed": 0.05, "laps": 1,
                               "heading_follow": False}, (0.0, 0.0), 0.0, 0.5)
    p0, y0 = pid.ref_ned_at(0.0)
    p1, y1 = pid.ref_ned_at(2.0)           # s = 0.1 m along +x
    assert np.allclose(p0, [0.0, 0.0, 0.5], atol=1e-9)
    assert abs(p1[0] - 0.1) < 1e-9 and abs(y1) < 1e-9
    assert scen["T_run_s"] == 16.0


def test_mode_switch_to_pid_and_back():
    """The mode combo swaps controller OBJECTS (pid <-> mpc family), refuses
    mid-engagement, and engages fine on the PID."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _worker(tmp)
        w.on_enable(True)
        w.set_mode("pid")
        assert w.ctrl is w._pid and w.cfg.mode == "pid"
        _feed_good_state(w)
        w.set_engaged(True)
        assert w.engaged
        w.set_mode("dobmpc")               # refused while engaged
        assert w.cfg.mode == "pid" and w.ctrl is w._pid
        for _ in range(3):
            _feed_good_state(w)
            w.tick()
        app = _app()
        deadline = time.monotonic() + 2.0
        while not pilots and time.monotonic() < deadline:
            app.processEvents()
        assert pilots and pilots[-1].source == "mpc"
        w.disengage("test")
        w.set_mode("dobmpc")
        assert w.ctrl is w._mpc_ctrl and w.cfg.mode == "dobmpc"
        w.teardown()


def test_assembler_velocity_propagation_bridges_fix_gaps():
    """Between camera frames the position ramps with the velocity estimate
    and yaw with the gyro, instead of holding the last fix."""
    asm = StateAssembler(z_source="tag", vel_lp_alpha=1.0, nudot_source="fd",
                         propagate=True, tag_stale_s=0.5)
    t0 = now()
    m, _ = asm.step(_fix([0.0, 0.0, 0.5], 0.0, t0 - 0.30),
                    _imu(t=t0 - 0.005, r=0.2), t0, 0.05)
    # no velocity estimate yet -> only the gyro bridges yaw over the 0.3 s age
    assert abs(m["eta"][5] - 0.2 * 0.30) < 1e-6
    # two fixes 0.1 s apart moving +x at 0.2 m/s -> v estimate = 0.2
    asm.reset()
    asm.step(_fix([0.0, 0.0, 0.5], 0.0, t0), _imu(t=t0), t0 + 0.01, 0.05)
    m2, _ = asm.step(_fix([0.02, 0.0, 0.5], 0.0, t0 + 0.1),
                     _imu(t=t0 + 0.1), t0 + 0.11, 0.05)
    # third step: SAME fix, 0.2 s older -> x bridged by v*age
    m3, _ = asm.step(_fix([0.02, 0.0, 0.5], 0.0, t0 + 0.1),
                     _imu(t=t0 + 0.3), t0 + 0.30, 0.05)
    assert abs(m3["eta"][0] - (0.02 + 0.2 * 0.20)) < 1e-6, m3["eta"][0]
    # propagation is capped at the stale gate
    asm_off = StateAssembler(z_source="tag", propagate=False)
    asm_off.step(_fix([0.0, 0.0, 0.5], 0.0, t0), _imu(t=t0), t0 + 0.01, 0.05)
    m4, _ = asm_off.step(_fix([0.02, 0.0, 0.5], 0.0, t0 + 0.1),
                         _imu(t=t0 + 0.3), t0 + 0.30, 0.05)
    assert abs(m4["eta"][0] - 0.02) < 1e-9        # off = plain ZOH


# =============================================================================
# window arbitration — pilot input while engaged is a takeover
# =============================================================================
def test_pilot_input_overrides_mpc():
    from rov_gui.tests.test_offline import Opts as BaseOpts
    from rov_gui.window import MainWindow

    class Opts(BaseOpts):
        mpc = True
        nav_config = str(ROOT / "config" / "hw_nav.yaml")
        nav_geometry = None

    _app()
    win = MainWindow(Opts())
    takeovers = []
    win.bus.cmd_mpc_engage.connect(takeovers.append)
    pilots = []
    win.bus.cmd_pilot.connect(pilots.append)
    win.teleop.enable.setChecked(True)
    pilots.clear()                       # the enable toggle may emit a frame
    win._mpc_engaged = True
    # neutral sticks: the pump must YIELD (no teleop frame, no takeover)
    win._pump_commands()
    assert not pilots and not takeovers
    # pilot pushes an axis -> exactly ONE takeover (edge fires through the
    # gate; the held key at 20 Hz must not spam), and NO teleop frame leaks —
    # not even from the edge-triggered pilot_changed path.
    from rov_gui.qt import Qt
    win.teleop.key_event(Qt.Key.Key_W, True, False)
    win._pump_commands()
    win._pump_commands()
    assert takeovers == [False]
    assert not pilots
    # the worker confirms the release via MpcStatus -> pump resumes
    from rov_gui.state import MpcStatus
    win._on_mpc_status(MpcStatus(engaged=False, reason="disengaged: pilot"))
    assert win._mpc_engaged is False
    assert win._mpc_takeover_sent is False     # latch cleared for next time
    # PILOT-initiated release: the held key is the pilot's intent and must
    # SURVIVE the handback (a takeover wiped by all_stop would leave the
    # vehicle coasting while the pilot holds reverse — review 2026-08-12).
    win._pump_commands()
    assert pilots and pilots[-1].surge > 0.0, \
        "held key was wiped during the takeover handback"
    win.teleop.key_event(Qt.Key.Key_W, False, False)
    # WORKER-initiated release (fault): all_stop() must still fire so stale
    # trim is not transmitted the moment the pump resumes.
    win._mpc_engaged = True
    win._on_mpc_status(MpcStatus(engaged=False, reason="disengaged: tag lost"))
    pilots.clear()
    win._pump_commands()
    assert pilots and not pilots[-1].any_axis
    win.shutdown()


def test_mpc_panel_lives_in_the_grid_and_fits_the_operator_screen():
    """The MPC panel is embedded in the BOTTOM ROW spanning columns 2-3 (the
    old PROPULSION/SENSORS slots — operator request 2026-08-14), with those
    two panels side by side under SYSTEM HEALTH in column 3. Stacking them
    vertically instead pushes the minimum height to 1040 — over the 1080p
    budget — which is why they sit abreast. Budget: the operator's actual
    screen (1920x1080): width keeps the base promise, height must fit 1080p.
    The base layout (no --mpc) is still pinned to 1366x768 by test_offline's
    layout test."""
    from rov_gui import theme
    from rov_gui.tests.test_offline import Opts as BaseOpts, _pump
    from rov_gui.window import MainWindow

    class Opts(BaseOpts):
        mpc = True
        nav_config = str(ROOT / "config" / "hw_nav.yaml")
        nav_geometry = None

    app = _app()
    theme.apply(app)
    win = MainWindow(Opts())
    assert win._traj_panel is not None
    # in the grid, column 3, below health — not a floating window, not a dock
    assert win._traj_panel.window() is win
    win.resize(1600, 1000)
    win.show()
    _pump(app, 250)
    hint = win.minimumSizeHint()
    assert hint.width() <= 1366, f"minimum width {hint.width()} > 1366"
    assert hint.height() <= 1000, f"minimum height {hint.height()} > 1000"
    # QComboBox builds an internal popup list view (a QAbstractScrollArea)
    # lazily; that is combo furniture, not a layout scroll area. Everything
    # NOT owned by a combo is still banned.
    def _under_combo(widget) -> bool:
        p = widget.parent()
        while p is not None:
            if isinstance(p, QtWidgets.QComboBox):
                return True
            p = p.parent()
        return False

    scrollers = [s for s in win.findChildren(QtWidgets.QAbstractScrollArea)
                 if not _under_combo(s)]
    assert not scrollers, f"a control station must not scroll: {scrollers}"
    win.close()


def test_tagnav_datum_first_fix_re_zeros_the_world():
    """datum=first_fix: the first accepted fix is (0,0,0)/yaw 0 and later
    poses are relative — position rotated into the datum heading, roll/pitch
    untouched."""
    tm = TagMap.single(25)
    cfg = NavConfig()
    R_bc, t_bc = cfg.R_t_frd_cam()
    nav = TagNav(tm, 0.170, R_bc, t_bc, WALL_PRESETS["x_into_wall"],
                 max_reproj_px=3.0, datum="first_fix")
    p1, yaw1 = np.array([-1.8, 0.4, 0.5]), 0.3
    dets, _ = _detections_for_body_pose(nav, tm, p1, rot_zyx(0.0, 0.0, yaw1))
    s1 = nav.solve(dets, K_TEST, None)
    assert s1 is not None
    assert np.linalg.norm(s1.p_ned) < 1e-2, s1.p_ned
    assert abs(math.atan2(s1.R_ned_body[1, 0], s1.R_ned_body[0, 0])) < 1e-2
    # move 0.2 m along the INITIAL heading and yaw a little
    d_body = np.array([0.2, 0.0, 0.0])
    p2 = p1 + rot_zyx(0.0, 0.0, yaw1) @ d_body
    dets, _ = _detections_for_body_pose(nav, tm, p2, rot_zyx(0.0, 0.0, yaw1 + 0.1))
    s2 = nav.solve(dets, K_TEST, None)
    assert s2 is not None
    assert np.allclose(s2.p_ned, [0.2, 0.0, 0.0], atol=5e-3), s2.p_ned
    assert abs(math.atan2(s2.R_ned_body[1, 0], s2.R_ned_body[0, 0]) - 0.1) < 5e-3
    # reset -> the next fix re-zeros
    nav.reset_datum()
    s3 = nav.solve(dets, K_TEST, None)
    assert np.linalg.norm(s3.p_ned) < 1e-2


def test_estop_releases_mpc_immediately():
    from rov_gui.tests.test_offline import Opts as BaseOpts
    from rov_gui.window import MainWindow

    class Opts(BaseOpts):
        mpc = True
        nav_config = str(ROOT / "config" / "hw_nav.yaml")
        nav_geometry = None

    _app()
    win = MainWindow(Opts())
    released = []
    win.bus.cmd_mpc_engage.connect(released.append)
    win._mpc_engaged = True
    win.estop()
    assert released == [False] and win._mpc_engaged is False
    win.shutdown()


def _floor_map_and_nav(duplicate_ids=(), min_tags=2):
    """A small flat tag grid + a downward-looking TagNav over it."""
    poses, pitch = {}, 0.2135
    for i, tid in enumerate((8, 9, 24, 25, 60, 61)):
        poses[tid] = (np.eye(3), np.array([0.0, i * pitch, 0.0]))
    tm = TagMap(poses, 25, "test grid")
    # Camera looking straight DOWN at the floor. MuJoCo xyaxes look along
    # -z_cam = -(x cross y): x=(0,-1,0), y=(1,0,0) gives x cross y = +Z_flu,
    # so the view direction is -Z_flu = down.
    R_bc = S_FLU_FRD @ R_flu_cam_from_xyaxes([0, -1, 0, 1, 0, 0])
    nav = TagNav(tm, 0.170, R_bc, np.zeros(3), np.eye(3),
                 max_reproj_px=3.0, min_tags=min_tags,
                 duplicate_ids=duplicate_ids, dup_confirm_px=6.0)
    return tm, nav


def test_duplicate_id_wrong_copy_is_dropped_not_trusted():
    """The pool mat reuses 12 ids (operator survey 2026-08-14). A detection of
    the OTHER copy must be RECOGNISED and dropped — the unique tags fix the
    pose, the duplicated tag is only kept if it lands where the map says."""
    tm, nav = _floor_map_and_nav(duplicate_ids=(60, 61))
    p, R = np.array([0.0, 0.4, -0.8]), rot_zyx(0.0, 0.0, 0.0)
    dets, _ = _detections_for_body_pose(nav, tm, p, R)
    by_id = {d.tag_id: d for d in dets}

    # (a) all tags are the SURVEYED copies -> everything is used
    sol = nav.solve(list(dets), K_TEST, None)
    assert sol is not None and sol.n_tags == 6, sol
    assert np.allclose(sol.p_ned, p, atol=2e-3), sol.p_ned
    assert not nav.last_dropped

    # (b) tag 60 is really the OTHER copy: its corners come from a tag one
    # grid pitch away. The pose must stay correct and 60 must be dropped.
    shifted = dict(tm.poses)
    shifted[60] = (np.eye(3), tm.poses[60][1] + np.array([0.2135, 0.0, 0.0]))
    dets_b, _ = _detections_for_body_pose(nav, TagMap(shifted, 25, ""), p, R)
    imposter = {d.tag_id: d for d in dets_b}[60]
    mixed = [by_id[t] for t in (8, 9, 24, 25, 61)] + [imposter]
    sol = nav.solve(mixed, K_TEST, None)
    assert sol is not None, nav.last_reject
    assert 60 not in sol.tag_ids, "the wrong copy was trusted"
    assert 61 in sol.tag_ids, "the RIGHT copy of a duplicated id was thrown away"
    assert np.allclose(sol.p_ned, p, atol=2e-3), sol.p_ned
    assert [t for t, _e in nav.last_dropped] == [60]
    assert "60" in sol.note

    # (c) without the declaration the imposter poisons the joint solve — this
    # is the 2026-08-13 pool symptom, and why the list has to be declared.
    _tm2, naive = _floor_map_and_nav(duplicate_ids=())
    assert naive.solve(mixed, K_TEST, None) is None
    assert "reproj" in naive.last_reject


def test_duplicated_ids_cannot_carry_a_fix_alone():
    """Two duplicated tags and nothing else: refused. They may confirm a pose
    the unique tags built, never build one — a pair from the wrong copy is
    self-consistent and would give a confident wrong answer."""
    tm, nav = _floor_map_and_nav(duplicate_ids=(60, 61))
    p, R = np.array([0.0, 0.4, -0.8]), rot_zyx(0.0, 0.0, 0.0)
    dets, _ = _detections_for_body_pose(nav, tm, p, R)
    only_dups = [d for d in dets if d.tag_id in (60, 61)]
    assert nav.solve(only_dups, K_TEST, None) is None
    assert "unique tags" in nav.last_reject and "dup held back" in nav.last_reject


def test_same_id_twice_in_one_frame_is_dropped():
    """Both copies in view at once: neither is usable, and the pair must not
    satisfy min_tags — before this, two detections of one id counted as two."""
    tm, nav = _floor_map_and_nav(duplicate_ids=(), min_tags=2)
    p, R = np.array([0.0, 0.4, -0.8]), rot_zyx(0.0, 0.0, 0.0)
    dets, _ = _detections_for_body_pose(nav, tm, p, R)
    d25 = next(d for d in dets if d.tag_id == 25)
    twin = Detection(25, d25.corners + 40.0)
    assert nav.solve([d25, twin], K_TEST, None) is None
    assert "seen twice" in nav.last_reject, nav.last_reject
    # ... and the rest of the frame still works with the pair removed
    sol = nav.solve([d for d in dets if d.tag_id != 25] + [d25, twin],
                    K_TEST, None)
    assert sol is not None and 25 not in sol.tag_ids


def test_build_tag_map_grows_outward_and_splits_a_duplicate():
    """One pass over a mat whose outer tags are unknown: every tag is
    recovered against the FROZEN anchors, and an id present at two physical
    places comes out as two instances rather than one averaged ghost."""
    import yaml

    from rov_gui.control.tagnav import quat_wxyz_to_R, tag_object_points
    from rov_gui.tools.build_tag_map import R_to_quat_wxyz, build

    obj = tag_object_points(0.170).astype(np.float64)
    pitch = 0.22
    truth = {}                                  # (r, c) -> (id, R, t)
    for r in range(6):
        for c in range(4):
            tid = 60 if (r, c) in ((1, 1), (4, 2)) else 300 + r * 4 + c
            yaw = math.pi if (r + c) % 2 else 0.0
            truth[(r, c)] = (tid, rot_zyx(0.0, 0.0, yaw),
                             np.array([c * pitch, r * pitch, 0.0]))
    # anchors = the inner block, minus the duplicated id (which the tool pulls
    # out for rediscovery, so the test hands `build` the same set it would get)
    anchor = {tid: [(R, t)] for (r, c), (tid, R, t) in truth.items()
              if 1 <= r <= 3 and 1 <= c <= 2 and tid != 60}
    K = np.array([[513.4, 0, 317.5], [0, 513.4, 179.8], [0, 0, 1.0]])
    frames, fi = {}, 0
    # a dense serpentine that runs slightly PAST the mat edges — an edge tag
    # only ever seen obliquely from inside is the one that stays unpromoted
    for py in np.linspace(-0.2, 5 * pitch + 0.2, 64):
        for px in np.linspace(-0.2, 3 * pitch + 0.2, 32):
            cam = np.array([px, py, -0.8])
            dets = []
            for tid, R, t in truth.values():
                pts = (R @ obj.T).T + t - cam
                uv = (K @ pts.T).T
                uv = uv[:, :2] / uv[:, 2:3]
                if (uv[:, 0].min() < 2 or uv[:, 0].max() > 637
                        or uv[:, 1].min() < 2 or uv[:, 1].max() > 357):
                    continue
                dets.append((tid, uv))
            if len(dets) >= 3:
                frames[fi] = {"t": fi * 0.067, "K": K, "dets": dets}
                fi += 1

    class Args:
        min_obs, max_spread, dup_sep, max_reproj_px = 5, 0.03, 0.10, 3.0
        # min_known=2 here only because the fixture's anchor block is 5 tags;
        # the tool defaults to 3 (a two-tag fit on a flat floor threw
        # metre-scale outliers on the real run).
        min_known, refine = 2, 1

    from rov_gui.tools.build_tag_map import refine as refine_pass

    frozen = set(anchor)
    known, _obs, _rounds = build(frames, dict(anchor), obj, Args(),
                                 log=lambda _m: None)
    known = refine_pass(frames, known, frozen, obj, Args(),
                        log=lambda _m: None)
    n_phys = len({(r, c) for r in range(6) for c in range(4)})
    assert sum(len(v) for v in known.values()) == n_phys, \
        f"recovered {sum(len(v) for v in known.values())} of {n_phys}"
    assert len(known[60]) == 2, "the duplicated id was not split in two"
    # 25 mm is the honest bound for a synthetic pass at this density: the
    # inner tags land sub-mm, the outermost ones are seen from fewer and more
    # oblique poses. The tool reports per-tag observation counts so a real run
    # can be judged the same way.
    for (r, c), (tid, _R, t) in truth.items():
        got = min(known[tid], key=lambda p: np.linalg.norm(p[1] - t))
        assert np.linalg.norm(got[1] - t) < 0.025, (tid, got[1], t)
    # frozen anchors came through untouched
    for tid, poses in anchor.items():
        if tid == 60:
            continue
        assert np.allclose(known[tid][0][1], poses[0][1]), tid
    # the quaternion round-trip the writer uses is exact
    for _tid, poses in list(known.items())[:8]:
        for R, _t in poses:
            assert np.allclose(quat_wxyz_to_R(R_to_quat_wxyz(R)), R, atol=1e-9)
    assert yaml is not None


def test_geofence_is_drawn_only_once_engaged():
    """The fence is DATUM-relative ('+-1.2 m around where START was pressed'),
    so before an engagement there is no datum and drawing it would put the box
    somewhere the fence will never be. Operator asked for it to appear on
    START; correctness says the same thing."""
    from rov_gui.state import MpcStatus
    from rov_gui.widgets.trajectory import TrajectoryView

    _app()
    v = TrajectoryView()
    v.set_geofence({"x": [-1.2, 1.2], "y": [-1.2, 1.2], "z": [-1, 1],
                    "margin": 0.3})
    drawn = []
    v._px = lambda x, y, _o=v._px: (drawn.append((x, y)) or _o(x, y))
    v.resize(400, 300)
    v.grab()                                  # paint with no status at all
    assert not any(abs(x) == 1.2 or abs(y) == 1.2 for x, y in drawn), \
        "the geofence was drawn before any engagement"
    drawn.clear()
    v.add_status(MpcStatus(engaged=True, p_flu=(0.0, 0.0, 0.0)))
    v.grab()
    assert any(abs(x) == 1.2 and abs(y) == 1.2 for x, y in drawn), \
        "the geofence did not appear once engaged"
    # and it goes away again on release
    drawn.clear()
    v.add_status(MpcStatus(engaged=False, p_flu=(0.0, 0.0, 0.0)))
    v.grab()
    assert not any(abs(x) == 1.2 and abs(y) == 1.2 for x, y in drawn)


def test_pool_border_follows_the_tag_map():
    """pool_margin_m: the wall is the outermost tag EDGES + the margin, and it
    is DERIVED, so a rebuilt map moves it instead of leaving a stale box."""
    cfg = NavConfig.load(ROOT / "config" / "hw_nav.yaml")
    assert cfg.pool_margin_m is not None and cfg.pool_ned is None
    tm = TagMap.load(cfg.tag_map_path)
    box = cfg.pool_from_tags(tm)
    P = np.array([t for poses in tm.instances.values() for (_R, t) in poses])
    pad = cfg.effective_tag_size() / 2.0 + cfg.pool_margin_m
    assert abs(box["x"][0] - (P[:, 0].min() - pad)) < 1e-9
    assert abs(box["y"][1] - (P[:, 1].max() + pad)) < 1e-9
    # every physical tag, edges included, is inside the wall
    h = cfg.effective_tag_size() / 2.0
    assert P[:, 0].min() - h >= box["x"][0] and P[:, 0].max() + h <= box["x"][1]
    assert P[:, 1].min() - h >= box["y"][0] and P[:, 1].max() + h <= box["y"][1]
    # ...with exactly the margin to spare
    assert abs((P[:, 0].min() - h) - box["x"][0] - cfg.pool_margin_m) < 1e-9
    # a map with no instances (wall geometry) derives nothing rather than crashing
    assert cfg.pool_from_tags(TagMap.single(104)) is not None
    cfg.pool_margin_m = None
    assert cfg.pool_from_tags(tm) is None


def test_localization_needs_no_original_survey_tag():
    """The anchor/new distinction lives ONLY in build_tag_map. Once the map
    exists every tag is equal: a frame showing only NEW tags localizes."""
    cfg = NavConfig.load(ROOT / "config" / "hw_nav.yaml")
    tm = TagMap.load(cfg.tag_map_path)
    old47 = set(TagMap.load(ROOT / "config" / "tag_map.yaml").instances)
    new_only = [t for t in tm.instances if t not in old47 and tm.is_unique(t)]
    assert len(new_only) >= 4
    R_bc, t_bc = cfg.R_t_frd_cam("main")
    nav = TagNav(tm, cfg.tag_size_m, R_bc, t_bc, np.eye(3), max_reproj_px=3.0,
                 min_tags=2, duplicate_ids=cfg.duplicate_ids)
    # pick 4 new tags that are near each other so one camera sees them all
    pts = {t: tm.instances[t][0][1] for t in new_only}
    seed = min(pts, key=lambda t: pts[t][1])
    near = sorted(pts, key=lambda t: np.linalg.norm(pts[t] - pts[seed]))[:4]
    centre = np.mean([pts[t] for t in near], axis=0)
    p_true = np.array([centre[0], centre[1], -0.8])
    R_true = rot_zyx(0.0, 0.0, 0.2)
    dets, _ = _detections_for_body_pose(nav, tm, p_true, R_true)
    dets = [d for d in dets if d.tag_id in near]
    sol = nav.solve(dets, K_TEST, None)
    assert sol is not None, nav.last_reject
    assert not (set(sol.tag_ids) & old47), "the fix leaned on an original tag"
    assert np.linalg.norm(sol.p_ned - p_true) < 5e-3, sol.p_ned


def test_nav_record_writes_map_and_raw_fixes():
    """REC NAV: map.json carries the locked map + pool, fixes.csv is RAW
    tag-frame (the engage datum must NOT leak into it) and keeps misses."""
    import csv as _csv

    from rov_gui.tests.test_offline import Opts as BaseOpts
    from rov_gui.window import MainWindow

    tmp = tempfile.mkdtemp(prefix="navrec_")

    class Opts(BaseOpts):
        mpc = True
        nav_config = str(ROOT / "config" / "hw_nav.yaml")
        nav_geometry = None
        nav_rec_dir = tmp

    _app()
    win = MainWindow(Opts())
    win._toggle_nav_record(True)
    assert win._nav_rec is not None
    win._mpc_datum = (1.0, 2.0, 0.0, math.pi / 2)   # active datum, on purpose
    win._on_nav_fix(NavFix(t_capture=10.0, n_tags=2, tag_ids=(25, 30),
                           p_ned=(0.5, 0.25, 0.4),
                           R_ned_body=tuple(np.eye(3).ravel()), yaw_ned=0.1,
                           reproj_rms_px=0.5, detect_ms=5.0, hz=18.0,
                           source="main", conn=Conn.ONLINE))
    win._on_nav_fix(NavFix(t_capture=10.1, note="no tags"))   # a miss, kept
    run_dir = win._nav_rec["dir"]
    win._toggle_nav_record(False)
    assert win._nav_rec is None
    meta = json.loads((run_dir / "map.json").read_text())
    # one entry per PHYSICAL tag: a duplicated id contributes two
    assert meta["anchor_tag_id"] == 25
    from rov_gui.control.geometry import NavConfig as _NC
    from rov_gui.control.tagnav import TagMap as _TM
    _tm = _TM.load(_NC.load(Opts.nav_config).tag_map_path)
    assert len(meta["tags"]) == sum(len(v) for v in _tm.instances.values())
    assert sorted(meta["duplicate_ids"]) == sorted(_tm.duplicate_ids)
    assert meta["pool_ned"] and abs(meta["tag_size_m"] - 0.17) < 1e-9
    with open(run_dir / "fixes.csv", newline="") as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["ok"] == "1" and rows[0]["tag_ids"] == "25;30"
    assert abs(float(rows[0]["x_ned"]) - 0.5) < 1e-9, \
        "fixes.csv must be RAW tag-frame — the datum transformed it"
    assert rows[1]["ok"] == "0" and rows[1]["note"] == "no tags"
    win.shutdown()


def test_nav_record_writes_raw_detections_for_map_building():
    """The corners + camera model are what build_tag_map consumes; a run that
    logged only solved fixes cannot grow the map (the 2026-08-13 gap)."""
    import csv as _csv

    from rov_gui.state import TagOverlay
    from rov_gui.tests.test_offline import Opts as BaseOpts
    from rov_gui.window import MainWindow

    tmp = tempfile.mkdtemp(prefix="navdet_")

    class Opts(BaseOpts):
        mpc = True
        nav_config = str(ROOT / "config" / "hw_nav.yaml")
        nav_geometry = None
        nav_rec_dir = tmp

    _app()
    win = MainWindow(Opts())
    win._toggle_nav_record(True)
    run_dir = win._nav_rec["dir"]
    quad = ((10.0, 20.0), (40.0, 22.0), (41.0, 51.0), (11.0, 49.0))
    win._on_tag_overlay(TagOverlay(
        panel="main", quads=(quad,), ids=(25,), mapped=(True,),
        src_w=640, src_h=360, localizes=True, t_capture=12.5,
        K=(513.4, 513.4, 317.5, 179.8)))
    # an overlay-only feed (no calibration) must NOT be recorded: its corners
    # belong to a different camera and would be read as this one's
    win._on_tag_overlay(TagOverlay(
        panel="second", quads=(quad,), ids=(9,), mapped=(True,),
        src_w=1920, src_h=1080, localizes=False, t_capture=12.5))
    win._toggle_nav_record(False)
    with open(run_dir / "detections.csv", newline="") as f:
        dets = list(_csv.DictReader(f))
    with open(run_dir / "frames.csv", newline="") as f:
        frames = list(_csv.DictReader(f))
    assert len(frames) == 1 and len(dets) == 1, (frames, dets)
    assert dets[0]["tag_id"] == "25"
    assert (float(dets[0]["x0"]), float(dets[0]["y0"])) == (10.0, 20.0)
    assert abs(float(frames[0]["fx"]) - 513.4) < 1e-6
    assert frames[0]["width"] == "640"
    # and build_tag_map can actually read what we just wrote
    from rov_gui.tools.build_tag_map import load_recording
    got = load_recording(run_dir)
    assert list(got) == [0] and got[0]["dets"][0][0] == 25
    assert abs(got[0]["K"][0, 0] - 513.4) < 1e-6
    win.shutdown()


# =============================================================================
# runner (same shape as test_offline.py — works with or without pytest)
# =============================================================================
def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
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
