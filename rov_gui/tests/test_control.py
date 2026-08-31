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
# MpcWorker — engage discipline, CSV, interlocks (stub controller, no acados)
# =============================================================================
class StubCtrl:
    """HwDobMpc's surface with a P-controller inside. Enough for the worker's
    logic to run; no casadi anywhere near it."""

    #: A stub that DOES accept a moving setpoint with a feedforward — it is
    #: modelled on HwDobMpc, which does. HwMpcc's False is what the follow
    #: refusal test pins.
    follow_ok = True

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
        self._target = (np.zeros(3), 0.0, np.zeros(3))
        self._path_plan = None

    def set_target_ned(self, p, yaw, v_ned=None, r_ned=0.0):
        v = np.zeros(3) if v_ned is None else np.asarray(v_ned, float).copy()
        self._target = (np.asarray(p, float).copy(), float(yaw), v)
        self._ref_traj = None
        self._path_plan = None

    def set_square_ned(self, square, origin_xy, yaw_fixed, depth):
        """Mirrors HwDobMpc: rot_deg is honoured and mirror_y is on, so the
        placement the worker hands down is the placement under test."""
        from rov_gui.control.reference import make_square_ref_world

        size = float(square["size"])
        size_y = square.get("size_y")
        size_y = size if size_y in (None, "") else float(size_y)
        speed = float(square["speed"])
        rot = float(square.get("rot_deg", 0.0))
        self.scenario = {"kind": "square", "size": size, "size_y": size_y,
                         "speed": speed,
                         "laps": int(square["laps"]),
                         "heading_follow": bool(square["heading_follow"]),
                         "origin_ned": [float(origin_xy[0]), float(origin_xy[1])],
                         "rot_deg": rot, "depth_ned": depth,
                         "yaw_fixed_ned_deg": math.degrees(yaw_fixed),
                         "yaw_rate_deg_s": 60.0,
                         "T_run_s": (int(square["laps"]) * 2
                                     * (size + size_y) / speed)}
        self._ref_traj = make_square_ref_world(
            size=size, speed=speed, depth=-float(depth),
            heading_follow=bool(square["heading_follow"]),
            yaw_rate=math.radians(60.0), dt=0.05,
            T_total=self.scenario["T_run_s"] + 10.0,
            origin_xy=(float(origin_xy[0]), -float(origin_xy[1])),
            rot_rad=-math.radians(rot), yaw_fixed=-float(yaw_fixed),
            size_y=size_y, mirror_y=True)
        self._path_plan = None
        return self.scenario

    def set_circle_ned(self, circle, origin_xy, yaw_fixed, depth):
        """Straight through ``place_circle_ned`` — which is exactly what all
        three real bridges do, so the stub cannot pass a placement the
        hardware would not fly."""
        from rov_gui.control.reference import place_circle_ned

        self._ref_traj, self.scenario = place_circle_ned(
            circle, origin_xy, yaw_fixed, depth, dt=0.05)
        self._path_plan = None
        return self.scenario

    def set_line_ned(self, line, origin_xy, yaw_fixed, depth):
        from rov_gui.control.reference import line_run_time, make_line_ref_world

        length = float(line.get("length", 1.0))
        speed = float(line.get("speed", 0.05))
        laps = int(line.get("laps", 5))
        ramp = float(line.get("ramp_s", 1.0))
        self.scenario = {
            "kind": "line", "length": length, "speed": speed, "laps": laps,
            "ramp_s": ramp, "dir_deg": float(line.get("dir_deg", 90.0)),
            "depth_ned": depth,
            "origin_ned": [float(origin_xy[0]), float(origin_xy[1])],
            "yaw_fixed_ned_deg": math.degrees(yaw_fixed),
            "T_run_s": line_run_time(length, speed, laps, ramp)}
        # A MOVING reference, not a constant: the path-following governor has
        # to have something to fall behind.
        self._ref_traj = make_line_ref_world(
            length=length, speed=speed, depth=-float(depth), dt=0.05,
            T_total=self.scenario["T_run_s"] + 10.0,
            origin_xy=(float(origin_xy[0]), -float(origin_xy[1])),
            dir_rad=-math.radians(float(line.get("dir_deg", 90.0))),
            yaw_fixed=-float(yaw_fixed), ramp_s=ramp)
        self._path_plan = None
        return self.scenario

    @property
    def path_plan_steps(self):
        return 1

    @property
    def path_plan_dt(self):
        return 0.05

    def set_path_plan_ned(self, plan):
        self._path_plan = plan

    def ref_ned_at(self, t):
        if self._path_plan is not None:
            return (self._path_plan.p_ned[:, 0].copy(),
                    float(self._path_plan.yaw_ned[0]),
                    self._path_plan.v_ned[:, 0].copy())
        if getattr(self, "_ref_traj", None) is None:
            return (self._target[0].copy(), self._target[1],
                    self._target[2].copy())
        p, yaw, v, _r = self._ref_traj(np.array([float(t)]))
        return (np.array([p[0, 0], -p[1, 0], -p[2, 0]]), -float(yaw[0]),
                np.array([v[0, 0], -v[1, 0], -v[2, 0]]))

    def step(self, eta, nu, nudot, t):
        p_ref, _yaw_ref, _v_ref = self.ref_ned_at(t)
        e = p_ref - np.asarray(eta[:3])
        u = np.zeros(6)
        u[:3] = 20.0 * e
        return u, {"w_hat": self.w_hat, "solve_ms": 0.2, "status": 0,
                   "n_fail": self.n_fail, "nis": 1.0}

    def note_applied(self, tau):
        self.applied = tau

    def reset(self):
        self._path_plan = None

    def meta(self):
        return {"type": self.mode, "solver": "stub"}


# Tests carry their OWN nav/mpc configs, decoupled from the repo's live
# mission files (config/hw_*.yaml change with every pool campaign — a test
# that reads them breaks whenever the square is retuned).
#
# The geofence_frame / geofence_ned keys below are DELIBERATELY still here
# after the fence was removed on 2026-08-14: they are what a real operator's
# hw_nav.yaml still contains, and this fixture is the regression test that a
# stale config keeps loading instead of raising (geometry.py: the keys are
# ignored, not validated).
_TEST_NAV_YAML = """\
geometry: wall
wall_tag_id: 25
nav_source: main
datum: map
geofence_frame: datum
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
    # 0 = arm the moment START is pressed if already over the origin, i.e. the
    # pre-approach behaviour these tests were written against. The approach
    # and the 10 s settle get their own test.
    w.cfg.engage["settle_s"] = 0.0
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
        # datum is re-zeroed at engage; there is no absolute-position gate).
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
        # The tail moves whenever a column is appended, and the equality above
        # already pins the whole header — so this only asserts the RULE that
        # matters: new columns go on the END, where an old reader keyed by
        # name never trips over them.
        # 08-21 follow, 08-24 tick, 08-30 replay's plan columns
        assert text[0].endswith("tick_ms,plan_id,ref_src,grip_cmd")
        assert ",path_s_m,ref_speed_m_s," in text[0]
        assert len(text) >= 2
        first = text[1].split(",")
        assert len(first) == len(text[0].split(","))
        # datum frame: the run STARTS at (0,0) whatever the tag-frame pose was
        assert abs(float(first[1])) < 0.05              # px (FLU x = NED x)
        assert abs(float(first[2])) < 0.05
        meta = json.loads(Path(path).with_suffix(".meta.json").read_text())
        assert meta["schema_version"] == 8      # 8: + plan_stream/replay
        # Written even with no replay flown, the imu_dr rule.
        assert meta["plan_stream"]["enabled"] is False
        # Written even when the estimator is off: an absent key cannot tell
        # "old build" from "switched off", and neither can the version.
        assert meta["imu_dr"] == {"enabled": False} or \
            meta["imu_dr"]["enabled"] is False
        assert meta["controller"]["solver"] == "stub"
        # The corner GATE was removed 2026-08-16; the PID now rides a
        # projection cursor on the same filleted curve the MPCC flies.
        assert meta["reference_clock"]["strategy"] == "arc_projection_cursor"
        assert meta["reference_clock"]["fillet_m"] > 0.0
        assert meta["hardware"]["axis_gain_provenance"].startswith("[예측]")
        # provenance back to absolute coordinates
        d = meta["hardware"]["datum_tag_frame"]
        assert d is not None and abs(d["p0"][0] - (-2.0)) < 1e-6
        # ...and the PLANT the controller was flown against (2026-08-14): a CSV
        # says what the vehicle did, this says what the controller thought it
        # was. The self-checks are the load-bearing part — the recorded C and D
        # generator matrices must reproduce fossen's own products, or the
        # record is describing a different model than the one that flew.
        plant = meta["plant"]
        assert plant["rigid_body"]["mass_kg"] > 0, plant
        assert len(plant["M"]["matrix"]) == 6
        assert plant["C"]["check_max_abs_err"] < 1e-9, plant["C"]
        assert plant["D"]["check_max_abs_err"] < 1e-9, plant["D"]
        # the mission as the PANEL had it, which survives even before START
        assert meta["mission"]["config_square"]["shape"]
        w.teardown()


def test_rec_button_writes_the_plant_and_the_gains_beside_the_data():
    """REC NAV / REC UI -> controller.json in the recording's own folder
    (operator request 2026-08-14: "PID square인데 Rec Nav 누르면 M, D, C도").

    Deliberately NOT engaged here: the whole point is that a hand-flown survey
    pass — which never opens a controller CSV — still records which controller
    was selected and what the vehicle model was."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _worker(tmp)
        out = Path(tmp) / "nav_120000"
        assert not w.engaged
        w.dump_run_meta(str(out))
        meta = json.loads((out / "controller.json").read_text())
        assert meta["meta_trigger"] == "rec_button"
        assert meta["run"]["engaged"] is False
        assert meta["controller"]["type"] == w.cfg.mode
        assert meta["plant"]["C"]["check_max_abs_err"] < 1e-9
        assert meta["plant"]["D"]["check_max_abs_err"] < 1e-9
        # M is diagonal-plus-zg on this plant, and must be the FULL matrix —
        # a diagonal alone cannot be told apart from a coupled one later.
        assert len(meta["plant"]["M"]["matrix"][0]) == 6
        assert meta["hardware"]["axis_gain"]["surge_n"] > 0
        w.teardown()


def test_launching_the_station_leaves_no_folder_behind():
    """Starting the GUI is not a run.

    On 2026-08-14 seven folders under sessions/low_level_controller_data/
    held nothing but the one-line build fingerprint setup() writes — the
    station had been launched and nothing was ever flown. The fingerprint is
    still worth having (it is what answers "am I on the current code?"), so it
    is DEFERRED: it reaches the mission log immediately and the file only when
    there is a real event to write beside it. It must head up EVERY run
    folder, not just the first — a session flies more than one run."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _worker(tmp)
        day = Path(tmp) / time.strftime("%Y%m%d")
        assert not day.exists() or not list(day.iterdir()), \
            f"setup() alone created {list(day.iterdir())}"

        w.on_enable(True)
        _feed_good_state(w)
        w.set_engaged(True)
        assert w.engaged
        runs = sorted(p for p in day.iterdir() if p.is_dir())
        assert len(runs) == 1, runs
        lines = (runs[0] / "events.log").read_text().splitlines()
        assert "ready:" in lines[0], lines[:2]
        assert any("ENGAGED" in ln for ln in lines), lines

        # A LATER run folder in the same session gets the banner too — a
        # session flies more than one run, and the second one must not be the
        # only file that cannot say which build produced it.
        w.disengage("test over")
        w._run_dir_pin = later = day / "0101_000000"
        later.mkdir(parents=True, exist_ok=True)   # run_dir() does this for real
        w._log_event("a later run")
        assert (later / "events.log").read_text().splitlines()[0].endswith(
            lines[0].split(" ", 2)[2]), "the build fingerprint did not repeat"
        w.teardown()


def test_pid_meta_records_every_knob_not_just_the_gains():
    """A gate or a slew limit changes a trajectory as surely as kp does, so
    the run record carries all of them (2026-08-14)."""
    from rov_gui.control.geometry import MpcConfig
    from rov_gui.control.pid import HwPid

    cfg = MpcConfig()
    cfg.pid = {"omega_derate": 0.6}
    m = HwPid(cfg, log=lambda _m: None).meta()
    for key in ("kp", "kd", "ki", "yaw_kp", "yaw_kd", "yaw_ki", "i_max_n",
                "yaw_i_max_nm", "e_gate_m", "yaw_gate_rad", "f_max",
                "mz_max", "slew_n_per_s", "omega_derate", "ctrl_hz"):
        assert key in m, f"pid meta lost {key}"
    # the recorded gains are the SCALED ones actually flown, not the design set
    assert abs(m["kp"][0] - 143.7 * 0.6 ** 2) < 1e-6, m["kp"]
    assert m["gains_provenance"].endswith("[예측]")


def test_start_traj_no_longer_has_a_geofence_to_refuse_against():
    """The geofence was REMOVED on 2026-08-14 at the operator's request.

    This test is the record of what that means, so nobody re-derives the old
    behaviour from a stale comment: a 5 m square that the ±2 m fence used to
    refuse now ARMS. That is intended, and the safety consequence is real —
    nothing in this package knows where the pool wall is any more.
    """
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _worker(tmp)
        w.on_enable(True)
        _feed_good_state(w)
        w.set_engaged(True)
        assert w.engaged
        w._warmup_left = 0
        w.set_scenario({"size": 5.0})
        w.set_traj(True)
        assert w.traj_on, "the removed geofence still refused a large square"
        assert not any("geofence" in m for _l, m in logs), \
            "a fence message survived the removal"
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


def test_runtime_tag_loss_disengages_but_leaving_the_old_fence_does_not():
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _worker(tmp)
        w.on_enable(True)
        _feed_good_state(w)
        w.set_engaged(True)
        assert w.engaged
        # Teleport far outside what USED to be the ±2 m fence: since
        # 2026-08-14 position alone never disengages (the fence is gone).
        w.on_nav_fix(_fix([5.0, 0.0, 0.8], 0.0, now()))
        w.on_vehicle_imu(_imu(depth=0.8))
        w.tick()
        assert w.engaged, "a position abort survived the geofence removal"
        # ...but a LOST TAG still does.
        _feed_good_state(w)
        # Stale fix: DEBOUNCED (tag_stale_hold_s) — one late frame is not a
        # lost localizer, and an undebounced gate killed both dobmpc runs on
        # 2026-08-14. It must still disengage once the staleness PERSISTS.
        w.fix = _fix([-2.0, 0.0, 0.8], 0.0, now() - 1.0)
        w.tick()
        assert w.engaged, "one stale sample disengaged (gate not debounced)"
        need = int(float(w.cfg.engage["tag_stale_hold_s"]) * w.cfg.ctrl_hz)
        for _ in range(need + 2):
            w.fix = _fix([-2.0, 0.0, 0.8], 0.0, now() - 1.0)
            w.tick()
        assert not w.engaged and "stale" in w.reason, w.reason
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
    p0, y0, v0 = pid.ref_ned_at(0.0)
    p1, y1, v1 = pid.ref_ned_at(2.0)       # s = 0.1 m along +x
    assert np.allclose(p0, [0.0, 0.0, 0.5], atol=1e-9)
    assert abs(p1[0] - 0.1) < 1e-9 and abs(y1) < 1e-9
    assert np.allclose(v0, [0.05, 0.0, 0.0], atol=1e-9)
    assert np.allclose(v1, [0.05, 0.0, 0.0], atol=1e-9)
    assert scen["T_run_s"] == 16.0


def test_pid_tracks_reference_velocity_in_body_frame():
    """Matching path and measured velocities produce no D force, including
    when world NED and body axes are separated by yaw."""
    from rov_gui.control.geometry import MpcConfig
    from rov_gui.control.pid import HwPid

    pid = HwPid(MpcConfig(), log=lambda m: None)
    yaw = math.pi / 2.0
    v_ref_ned = np.array([0.10, -0.04, 0.02])
    eta = np.array([1.0, 2.0, 0.5, 0.0, 0.0, yaw])
    nu = np.zeros(6)
    nu[:3] = rot_zyx(0.0, 0.0, yaw).T @ v_ref_ned
    pid.set_target_ned(eta[:3], yaw, v_ned=v_ref_ned)

    p_ref, yaw_ref, v_ref = pid.ref_ned_at(0.0)
    assert np.allclose(p_ref, eta[:3])
    assert yaw_ref == yaw
    assert np.allclose(v_ref, v_ref_ned)
    u, _ = pid.step(eta, nu, np.zeros(6), 0.0)
    assert np.allclose(u, 0.0, atol=1e-12), u


def test_mpc_ref_ned_at_returns_world_ned_velocity():
    """The public reference helper must not leak NMPC's body-frame velocity."""
    from rov_gui.control.mpc_bridge import HwDobMpc

    ctrl = HwDobMpc.__new__(HwDobMpc)
    xref = np.zeros((12, 1))
    xref[:3, 0] = [1.0, -2.0, 0.7]
    xref[3:6, 0] = [0.2, -0.1, 0.8]
    xref[6:9, 0] = [0.12, -0.04, 0.03]
    ctrl._xref_ned = lambda _t: xref

    p_ref, yaw_ref, v_ref_ned = ctrl.ref_ned_at(3.0)
    assert np.allclose(p_ref, xref[:3, 0])
    assert yaw_ref == xref[5, 0]
    assert np.allclose(v_ref_ned,
                       rot_zyx(*xref[3:6, 0]) @ xref[6:9, 0], atol=1e-12)


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

    # The MISSION LOG is the ONE exemption, added 2026-08-14 with the
    # operator's request to read back over older lines. The rule it is exempt
    # from is "a control station must not hide a CONTROL behind a scroll": a
    # log is not a control, nothing in it is clickable, and the newest line —
    # the only one that matters live — is always the visible one (payload.py
    # add_event sticks to the bottom unless the operator scrolled away).
    # Everything else stays banned.
    scrollers = [s for s in win.findChildren(QtWidgets.QAbstractScrollArea)
                 if not _under_combo(s) and s is not win.payload.log_view]
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


def test_rect_setpoint_reduces_to_the_verbatim_square():
    """The rectangle generalizes the sim's VERBATIM square_setpoint, so with
    equal sides it must agree exactly — otherwise the generalization has
    quietly forked from the simulator the copy exists to match."""
    from rov_gui.control.reference import rect_setpoint

    for size, speed in ((1.0, 0.12), (0.2, 0.05), (2.5, 0.3)):
        for t in np.linspace(0.0, 4.0 * size / speed * 2.2, 257):
            a_p, a_t = square_setpoint(t, size, speed)
            b_p, b_t = rect_setpoint(t, size, size, speed)
            assert np.allclose(a_p, b_p) and np.allclose(a_t, b_t), (t, size)


def test_line_profile_is_out_and_back_and_never_jumps():
    """The out-and-back line: reaches exactly `length`, comes exactly back,
    and — the point of the cosine ramps — has ZERO velocity at each
    turnaround, so the reference is something the vehicle can actually be."""
    from rov_gui.control.reference import (line_lap_of, line_profile,
                                           line_run_time)

    L, v, ramp = 2.0, 0.05, 1.0
    T_half = ramp + L / v
    assert abs(line_run_time(L, v, 5, ramp) - 5 * 2 * T_half) < 1e-9
    ts = np.arange(0.0, 2 * T_half * 3, 0.01)
    s = np.array([line_profile(t, L, v, ramp)[0] for t in ts])
    ds = np.array([line_profile(t, L, v, ramp)[1] for t in ts])
    assert -1e-9 <= s.min() and s.max() <= L + 1e-9, (s.min(), s.max())
    assert abs(s.max() - L) < 1e-3, s.max()          # really reaches the end
    assert abs(s[0]) < 1e-12                          # starts at the origin
    # turnarounds: speed passes through zero, and s is continuous everywhere
    assert abs(line_profile(T_half, L, v, ramp)[1]) < 1e-9
    assert abs(line_profile(2 * T_half, L, v, ramp)[1]) < 1e-9
    assert np.abs(np.diff(s)).max() < v * 0.01 * 1.001, "position jumped"
    assert np.abs(np.diff(ds)).max() < 0.02, "velocity stepped"
    assert abs(ds).max() <= v + 1e-9                  # never exceeds the cruise
    # laps count out-and-backs
    assert line_lap_of(T_half, L, v, ramp) == 0
    assert line_lap_of(2 * T_half + 0.01, L, v, ramp) == 1
    # the integral of speed is the distance covered.
    # np.trapezoid is the numpy 2.x spelling; `robust` is pinned to numpy
    # 1.26 because gtsam needs it (memory: environment-numpy-constraint), so
    # this suite has to accept both names or it cannot run in its own env.
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    assert abs(_trapz(np.abs(ds[:int(T_half / 0.01)]), dx=0.01) - L) < 0.02


def test_line_mission_is_placed_at_a_tag():
    """START on a LINE: the origin comes from the tag map (datumized) and the
    armed scenario says what ran. The endpoints used to be fence-checked too;
    the geofence was removed 2026-08-14, so the only refusals left here are
    about the TAG (ambiguous or unknown) — see the assertions below."""
    from rov_gui.control.tagnav import TagMap

    tmp = tempfile.mkdtemp(prefix="linemission_")
    w, _bus, _pilots, _logs = _worker(tmp)
    # the REAL mat map, so "tag 79" and "tag 60 is duplicated" mean what they
    # mean in the pool rather than in a fixture
    tm = TagMap.load(ROOT / "config" / "tag_map_full.yaml")
    w._tagmap = tm
    p79 = tm.instances[79][0][1]
    assert len(tm.instances[60]) == 2, "fixture assumption: 60 is duplicated"

    # engage right above tag 79 so the datum sits there
    w._datum = {"p0": np.array([p79[0], p79[1], 0.8]), "yaw0": 0.0,
                "Rz": rot_zyx(0.0, 0.0, 0.0)}
    w._eta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    w.engaged, w._warmup_left = True, 0
    w.set_scenario({"shape": "line", "origin_tag": 79, "length": 2.0})
    w.set_traj(True)
    assert w.traj_on, w.reason
    assert w.ctrl.scenario["path_min_duration_s"] > 0.0
    assert abs(w.ctrl.scenario["path_timeout_s"]
               - w.ctrl.scenario["path_min_duration_s"]
               * w.cfg.traj_timeout_factor) < 1e-9
    sc = w.ctrl.scenario
    assert sc["kind"] == "line" and sc["origin_tag"] == 79
    assert abs(sc["length"] - 2.0) < 1e-9 and sc["dir_deg"] == 90.0
    # origin is tag 79 expressed in the datum frame == (0, 0) here
    assert np.allclose(sc["origin_ned"], [0.0, 0.0], atol=1e-9), sc["origin_ned"]

    # A 9 m line out of a 4.4 m pool now ARMS: nothing checks the geometry
    # against a boundary any more. Pinned so the removal stays visible rather
    # than being rediscovered in the water.
    w.set_traj(False)
    w.set_scenario({"shape": "line", "origin_tag": 79, "length": 9.0})
    w.set_traj(True)
    assert w.traj_on, "something is still fencing the placed path"
    w.set_traj(False)

    # an ambiguous (duplicated) tag is refused rather than guessed
    w.set_scenario({"shape": "line", "origin_tag": 60, "length": 0.5})
    w.set_traj(True)
    assert not w.traj_on
    # ...and an unknown tag too
    w.set_scenario({"shape": "line", "origin_tag": 999, "length": 0.5})
    w.set_traj(True)
    assert not w.traj_on
    w.teardown()


def test_start_flies_to_the_tag_then_settles_before_the_path_runs():
    """START from far away no longer refuses (operator 2026-08-14): it takes
    station over the path origin under DP, holds ``settle_s``, and only then
    arms. The travelling is done by the SAME controller under the SAME
    interlocks — there is no separate, less-guarded move mode."""
    tmp = tempfile.mkdtemp(prefix="approach_")
    w, _bus, _pilots, logs = _worker(tmp)
    w.cfg.engage["settle_s"] = 2.0
    w.cfg.engage["start_err_max_m"] = 0.2
    _feed_good_state(w)
    w.on_enable(True)
    w.set_engaged(True)
    assert w.engaged, w.reason
    w._warmup_left = 0
    # the origin is 1.2 m away in y
    w.set_scenario({"shape": "line", "origin_tag": None,
                    "origin": [w._datum["p0"][0], w._datum["p0"][1] + 1.2],
                    "length": 0.4})
    w.set_traj(True)
    assert not w.traj_on, "the path armed without going to its origin first"
    assert "approaching" in w.reason, w.reason
    assert w._approach is not None
    # The setpoint STARTS at the vehicle and walks to the tag — a teleporting
    # setpoint is a full-authority step command, which is what made the
    # vehicle lunge and wallow in the pool (2026-08-14).
    assert abs(w.ctrl._target[0][1]) < 1e-6, w.ctrl._target
    w.cfg.engage["approach_speed_m_s"] = 0.10
    w.cfg.engage["approach_lead_m"] = 0.35

    t = now()
    w._approach["t_prev"] = t
    for k in range(1, 21):                     # 20 ticks x 50 ms = 1 s
        w._tick_approach(t + 0.05 * k)
    assert 0.08 < w.ctrl._target[0][1] < 0.12, w.ctrl._target   # ~0.10 m/s
    assert w._approach["t_in"] is None and not w.traj_on
    # ...and the setpoint never runs further than the leash ahead of the hull
    for k in range(1, 401):
        w._tick_approach(t + 1.0 + 0.05 * k)
    lead = math.hypot(w._approach["sp"][0] - w._eta[0],
                      w._approach["sp"][1] - w._eta[1])
    assert lead <= 0.35 + 1e-6, lead

    w._eta = np.array([0.0, 1.19, 0.8, 0.0, 0.0, 0.0])   # arrived
    w._tick_approach(t)
    assert w._approach["t_in"] is not None and not w.traj_on
    assert "settling" in w.reason, w.reason
    w._tick_approach(t + 1.0)                  # settle not finished
    assert not w.traj_on
    w._eta = np.array([0.0, 0.9, 0.8, 0.0, 0.0, 0.0])    # drifted back out
    w._tick_approach(t + 1.5)
    assert w._approach["t_in"] is None, "the settle timer survived a drift-out"
    w._eta = np.array([0.0, 1.2, 0.8, 0.0, 0.0, 0.0])
    w._tick_approach(t + 2.0)
    w._tick_approach(t + 4.1)                  # 2.1 s held -> go
    assert w.traj_on and w._approach is None, w.reason
    assert w.ctrl.scenario["kind"] == "line"

    # STOP cancels an approach in progress rather than leaving it armed
    w.set_traj(False)
    w.set_traj(True)
    assert w._approach is not None and not w.traj_on
    w.set_traj(False)
    assert w._approach is None
    # ...and so does a disengage
    w.set_traj(True)
    w.disengage("test")
    assert w._approach is None and not w.engaged
    assert any("heading for" in m for _l, m in logs)
    w.teardown()


def test_measured_operating_depth_band_is_still_on_record():
    """The geofence is gone (2026-08-14), so nothing pins the operating band
    any more — which is exactly why it is worth measuring here.

    This used to assert that hw_nav.yaml's fence z contained the measured
    envelope; getting the sign backwards silently refused every START. With no
    fence there is no such failure, and no guard either: the recorded band is
    now the only statement anyone has about where this vehicle actually flies.
    """
    import csv as _csv
    import glob as _glob

    # BOTH trees: pre-2026-08-14 runs are in sessions/nav_runs/<stamp>/ and
    # were deliberately not migrated; new ones are nav_<hhmmss>/ inside a dated
    # run folder. Globbing only the old one would let this test quietly become
    # a no-op the moment nothing new lands there (it passes on an empty set).
    runs = (_glob.glob(str(ROOT / "sessions" / "nav_runs" / "*" / ""))
            + _glob.glob(str(ROOT / "sessions" / "low_level_controller_data"
                             / "*" / "*" / "nav_*" / "")))
    seen = []
    for d in sorted(runs):
        try:
            with open(Path(d) / "fixes.csv", newline="") as f:
                seen += [float(r["z_ned"]) for r in _csv.DictReader(f)
                         if r["ok"] == "1"]
        except OSError:
            continue
    if not seen:
        return                                   # no recordings on this machine
    z = np.array(seen)
    band = np.percentile(z, 1), np.percentile(z, 99)
    # Tag-frame z: the floor tags are z=0 with +z DOWN, so a vehicle swimming
    # above the mat is NEGATIVE. A positive band would mean the vehicle was
    # under the floor, i.e. the map or the sign convention broke.
    assert band[1] < 0.0, (
        f"measured operating band {np.round(band, 2).tolist()} (n={len(z)}) "
        f"puts the vehicle BELOW the tag mat — check the map z convention")


def test_start_refusals_reach_the_panel_not_only_the_log():
    """A refused START used to leave the chip saying 'engaged (DP hold)', so
    the operator saw nothing happen and no reason why (pool, 2026-08-14)."""
    tmp = tempfile.mkdtemp(prefix="refuse_")
    w, _bus, _pilots, _logs = _worker(tmp)
    _feed_good_state(w)
    w.on_enable(True)
    w.set_engaged(True)
    assert w.engaged
    w._warmup_left = 5
    w.set_traj(True)
    assert w.reason.startswith("START refused: "), w.reason
    assert "warming up" in w.reason
    # A path that leaves the pool no longer refuses — the fence was removed
    # 2026-08-14 — so the remaining refusal to pin is the tag-map one, which
    # is still a refusal the operator must see on the PANEL and not only in
    # the log.
    w._warmup_left = 0
    w.set_scenario({"shape": "line", "origin_tag": 999, "length": 1.0})
    w.set_traj(True)
    assert not w.traj_on
    assert w.reason.startswith("START refused: "), w.reason
    assert "999" in w.reason, w.reason
    w.teardown()


def test_path_following_waits_for_a_lagging_vehicle():
    """The worker must install a projection-anchored plan, not wall time.

    A stationary hull may let the local acceleration ramp fill the configured
    lookahead, but stage 0 can never run farther away. A hull placed on stage
    0 advances spatial progress without needing a wall-time schedule.
    """
    tmp = tempfile.mkdtemp(prefix="pathfollow_")
    w, _bus, _pilots, _logs = _worker(tmp)
    w.cfg.engage["settle_s"] = 0.0
    w.cfg.path_following = True
    w.cfg.path_lead_m = 0.15
    _feed_good_state(w)
    w.on_enable(True)
    w.set_engaged(True)
    w._warmup_left = 0
    w._datum = {"p0": np.zeros(3), "yaw0": 0.0, "Rz": rot_zyx(0.0, 0.0, 0.0)}
    w._eta = np.zeros(6)
    w.set_scenario({"shape": "line", "origin_tag": None, "origin": [0.0, 0.0],
                    "length": 1.0, "dir_deg": 90.0, "speed": 0.05,
                    "laps": 1, "ramp_s": 0.0})
    w.set_traj(True)
    assert w.traj_on, w.reason

    # A vehicle that does NOT move at all: stage 0 is bounded by lookahead.
    meas = {"eta": np.zeros(6)}
    t = now()
    for k in range(1, 401):                      # 20 s of ticks
        w._advance_path_clock(t + 0.05 * k, meas)
    p_ref, _, _v_ref = w.ctrl.ref_ned_at(w._tau)
    lag = float(np.linalg.norm(np.asarray(p_ref)[:2]))
    assert lag <= w.cfg.path_lead_m + 1e-6, f"reference ran away: {lag:.3f} m"
    assert w._tau < 20.0, "spatial progress followed wall time"
    # The cursor is the thing that bounds it: progress is a PROJECTION of the
    # hull, so a hull that never moves never advances the setpoint past lead.
    assert w._path_cursor is not None
    assert w._path_cursor.theta <= w.cfg.path_lead_m + 1e-6

    # ...and a vehicle placed on stage 0 advances through the path.
    w2, _b2, _p2, _l2 = _worker(tempfile.mkdtemp(prefix="pathfollow2_"))
    w2.cfg.engage["settle_s"] = 0.0
    w2.cfg.path_following = True
    _feed_good_state(w2)
    w2.on_enable(True)
    w2.set_engaged(True)
    w2._warmup_left = 0
    w2._datum = {"p0": np.zeros(3), "yaw0": 0.0, "Rz": rot_zyx(0.0, 0.0, 0.0)}
    w2._eta = np.zeros(6)
    w2.set_scenario({"shape": "line", "origin_tag": None, "origin": [0.0, 0.0],
                     "length": 1.0, "dir_deg": 90.0, "speed": 0.05,
                     "laps": 1, "ramp_s": 0.0})
    w2.set_traj(True)
    t = now()
    for k in range(1, 201):                      # vehicle rides the reference
        tt = t + 0.05 * k
        p_ref, _, _v_ref = w2.ctrl.ref_ned_at(w2._tau)
        w2._advance_path_clock(tt, {"eta": np.array(
            [p_ref[0], p_ref[1], 0.0, 0.0, 0.0, 0.0])})
    # tau is now ARCLENGTH (m), not the sampler time it used to be: a vehicle
    # riding the reference for 10 s at 0.05 m/s covers ~0.5 m of a 1 m line.
    assert w2._tau > 0.35, f"a vehicle on the path made no progress: {w2._tau:.3f} m"

    # switching it off restores the wall clock exactly
    w2.cfg.path_following = False
    w2._t0_traj = t
    assert abs(w2._advance_path_clock(t + 3.0, {"eta": np.zeros(6)}) - 3.0) < 1e-9
    w.teardown()
    w2.teardown()


def test_line_direction_is_a_map_heading_not_a_datum_one():
    """dir_deg 90 means the POOL's +y, whatever heading the vehicle engaged
    on. Before this the line was laid along datum +y, so the same mission
    drew a different physical direction every run (operator: "+y is the
    pool's long side but it drew along -x", 2026-08-14)."""
    tmp = tempfile.mkdtemp(prefix="linedir_")
    w, _bus, _pilots, _logs = _worker(tmp)
    w.cfg.engage["settle_s"] = 0.0
    _feed_good_state(w)
    w.on_enable(True)
    w.set_engaged(True)
    w._warmup_left = 0
    for yaw0_deg in (0.0, 30.0, -115.0):
        w._datum = {"p0": np.array([0.0, 0.0, 0.8]),
                    "yaw0": math.radians(yaw0_deg),
                    "Rz": rot_zyx(0.0, 0.0, -math.radians(yaw0_deg))}
        w._eta = np.zeros(6)
        w.set_traj(False)
        w.set_scenario({"shape": "line", "origin_tag": None,
                        "origin": [0.0, 0.0], "length": 1.0, "dir_deg": 90.0})
        w.set_traj(True)
        assert w.traj_on, w.reason
        sc = w.ctrl.scenario
        # the armed direction is expressed in the DATUM frame...
        assert abs(_wrap_deg(sc["dir_deg"] - (90.0 - yaw0_deg))) < 1e-6, sc
        # ...and rotating it back by the engage yaw gives the MAP heading
        assert abs(_wrap_deg(sc["dir_deg"] + yaw0_deg - 90.0)) < 1e-6
        assert abs(sc["dir_map_deg"] - 90.0) < 1e-6
    w.teardown()


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _map_corners(w) -> list:
    """The armed rectangle's corners in the MAP frame — the same chain the
    trajectory widget draws with (TrajectoryView._square_corners + _to_map),
    so this asserts on the picture the operator actually sees."""
    from rov_gui.control.reference import rect_corners_world

    sc = w.ctrl.scenario
    ox, oy = sc["origin_ned"]
    flu = rect_corners_world(sc["size"], sc.get("size_y", sc["size"]),
                             (ox, -oy), -math.radians(sc.get("rot_deg", 0.0)),
                             mirror_y=True)
    return [w._to_map_xy(x, -y) for x, y in flu]


def test_square_is_axis_aligned_in_the_map_with_the_tag_at_the_min_corner():
    """The rectangle belongs to the POOL, not to whatever heading the vehicle
    happened to hold at START. Two properties, both from the operator's
    screenshot of a tilted square (2026-08-14):

      1. rot_deg 0 means the sides run along the MAP's x and y axes, for any
         engage yaw — the same fold the line's dir_deg already gets.
      2. the entered tag is the min-x / min-y corner (bottom LEFT of the
         top-down plot), and the first leg leaves it along +x.

    Property 2 needs the mirror: the sim's square grows into world-FLU +y,
    which the NED mirror turns into map -y, putting the tag at the corner
    with the LARGEST y.
    """
    tmp = tempfile.mkdtemp(prefix="sqrot_")
    w, _bus, _pilots, _logs = _worker(tmp)
    w.cfg.engage["settle_s"] = 0.0
    _feed_good_state(w)
    w.on_enable(True)
    w.set_engaged(True)
    w._warmup_left = 0
    for yaw0_deg in (0.0, 30.0, 91.26, -115.0):
        w._datum = {"p0": np.array([0.4, -0.3, 0.8]),
                    "yaw0": math.radians(yaw0_deg),
                    "Rz": rot_zyx(0.0, 0.0, -math.radians(yaw0_deg))}
        # sit ON the corner, so START arms the path instead of flying to it
        w._eta = np.zeros(6)
        o = w._datumize(np.array([0.9, 0.1, 0.0, 0.0, 0.0, 0.0]))
        w._eta[0:2] = o[0:2]
        w.set_traj(False)
        w.set_scenario({"shape": "square", "origin_tag": None,
                        "origin": [0.9, 0.1], "size": 2.0, "size_y": 0.5})
        w.set_traj(True)
        assert w.traj_on, w.reason
        sc = w.ctrl.scenario
        assert abs(_wrap_deg(sc["rot_deg"] + yaw0_deg)) < 1e-6, sc
        assert abs(sc["rot_map_deg"]) < 1e-9, sc

        c = _map_corners(w)
        for a, b, axis in ((c[0], c[1], 1), (c[1], c[2], 0),
                           (c[2], c[3], 1), (c[3], c[0], 0)):
            assert abs(a[axis] - b[axis]) < 1e-9, (yaw0_deg, c)
        assert abs(c[0][0] - 0.9) < 1e-9 and abs(c[0][1] - 0.1) < 1e-9, c
        xs = [p[0] for p in c]
        ys = [p[1] for p in c]
        assert abs(min(xs) - c[0][0]) < 1e-9 and abs(min(ys) - c[0][1]) < 1e-9
        assert abs((max(xs) - min(xs)) - 2.0) < 1e-9
        assert abs((max(ys) - min(ys)) - 0.5) < 1e-9
        # and the sampled reference agrees with the drawn corners: it starts
        # ON the tag and the first leg runs along map +x.
        p0, _, _v0 = w.ctrl.ref_ned_at(0.0)
        p1, _, _v1 = w.ctrl.ref_ned_at(2.0)
        m0 = w._to_map_xy(p0[0], p0[1])
        m1 = w._to_map_xy(p1[0], p1[1])
        assert abs(m0[0] - 0.9) < 1e-6 and abs(m0[1] - 0.1) < 1e-6, m0
        assert m1[0] > m0[0] + 0.05 and abs(m1[1] - m0[1]) < 1e-6, (m0, m1)
    w.teardown()


def test_path_split_separates_being_late_from_being_off_the_path():
    """`err 12 cm` on a line run was 11 cm of lag and 1 cm off the line
    (sessions/low_level_controller_data/20260814/2018/mpc_201843.csv). The
    deliberate stage-0 lookahead contributes along error, so the readout has
    to report the two axes apart or path following looks like it failed."""
    tmp = tempfile.mkdtemp(prefix="split_")
    w, _bus, _pilots, _logs = _worker(tmp)
    w.cfg.engage["settle_s"] = 0.0
    _feed_good_state(w)
    w.on_enable(True)
    w.set_engaged(True)
    w._warmup_left = 0
    w._datum = {"p0": np.zeros(3), "yaw0": 0.0, "Rz": np.eye(3)}
    w._eta = np.zeros(6)
    w.set_scenario({"shape": "line", "origin_tag": None, "origin": [0.0, 0.0],
                    "length": 2.0, "dir_deg": 90.0})
    w.set_traj(True)
    assert w.traj_on, w.reason
    t = 8.0
    p_ref, _, _v_ref = w.ctrl.ref_ned_at(t)  # somewhere along the outbound leg
    # travelling +y (east): 0.10 m behind it, and 0.02 m toward -x (north is
    # left of east in NED, so -x is the vehicle's RIGHT).
    meas = {"eta": np.array([p_ref[0] - 0.02, p_ref[1] - 0.10, p_ref[2],
                             0.0, 0.0, 0.0])}
    along, cross = w._path_split(meas, p_ref, t)
    assert abs(along - 0.10) < 1e-6, along      # + = late
    assert abs(cross + 0.02) < 1e-6, cross      # - = right of travel
    # mirror it: same point on the other side of the line reads + and the lag
    # is unchanged, so the two axes really are independent.
    meas2 = {"eta": np.array([p_ref[0] + 0.02, p_ref[1] - 0.10, p_ref[2],
                              0.0, 0.0, 0.0])}
    along2, cross2 = w._path_split(meas2, p_ref, t)
    assert abs(along2 - 0.10) < 1e-6 and abs(cross2 - 0.02) < 1e-6
    # station hold has no path, so there is nothing to split
    w.set_traj(False)
    assert w._path_split(meas, p_ref, t) == (None, None)
    w.teardown()


def test_pid_can_arm_every_shape_the_worker_offers():
    """A controller missing one shape's arming method looks exactly like
    'reached the start point and then nothing happened' — which is what PID +
    line did in the pool on 2026-08-14."""
    from rov_gui.control.geometry import MpcConfig
    from rov_gui.control.pid import HwPid

    cfg = MpcConfig.load(ROOT / "config" / "hw_mpc.yaml")
    for name in ("set_target_ned", "set_square_ned", "set_line_ned",
                 "set_path_plan_ned", "ref_ned_at", "step", "reset", "meta"):
        assert hasattr(HwPid, name), f"HwPid is missing {name}"
    pid = HwPid(cfg)
    sc = pid.set_line_ned({"length": 1.0, "speed": 0.05, "laps": 2,
                           "ramp_s": 0.5, "dir_deg": 90.0},
                          (0.0, 0.0), 0.0, 0.8)
    assert sc["kind"] == "line" and sc["T_run_s"] > 0
    p0, _y0, v0 = pid.ref_ned_at(0.0)
    p1, _y1, v1 = pid.ref_ned_at(sc["T_run_s"] / 8.0)
    assert p1[1] > p0[1] + 0.1, (p0, p1)      # moved along +y in NED
    assert np.linalg.norm(v0) < 1e-12          # cosine ramp starts at rest
    assert v1[1] > 0.0                         # outbound reference velocity


def test_pid_and_mpc_place_the_same_mission_on_the_same_geometry():
    """A controller difference must never be a GEOMETRY difference — the two
    runs exist to be compared. HwPid kept its own copy of the placement
    arithmetic and it drifted: it ignored `size_y`, so an operator entering
    2.00 x 0.50 m flew a rectangle under MPC and a 2.00 x 2.00 m square under
    PID (operator report, 2026-08-14). Both now delegate to
    reference.place_*_ned, and this walks the sampled path of each to prove
    it — a shared helper that only ONE of them called would still pass a
    method-presence check.

    HwDobMpc is built without __init__ (acados/casadi are not importable in
    the offline suites); the arming methods touch nothing but DT_CTRL, N and
    set_reference_traj. The comparison is therefore on the stored world-FLU
    sampler, which is the placement; everything past it is ported code the
    two share by construction.
    """
    from rov_gui.control.geometry import MpcConfig
    from rov_gui.control.mpc_bridge import HwDobMpc
    from rov_gui.control.pid import HwPid

    cfg = MpcConfig.load(ROOT / "config" / "hw_mpc.yaml")
    pid = HwPid(cfg)
    mpc = HwDobMpc.__new__(HwDobMpc)
    mpc.P = types.SimpleNamespace(DT_CTRL=1.0 / cfg.ctrl_hz)
    mpc.nmpc = types.SimpleNamespace(N=60)

    square = {"size": 2.0, "size_y": 0.5, "speed": 0.05, "laps": 2,
              "heading_follow": False, "yaw_rate_deg_s": 60.0,
              "rot_deg": 17.0}
    line = {"length": 1.5, "speed": 0.05, "laps": 2, "ramp_s": 1.0,
            "dir_deg": 90.0}
    for arm, spec in (("set_square_ned", square), ("set_line_ned", line)):
        a = getattr(pid, arm)(spec, (0.3, -0.2), 0.4, -0.9)
        b = getattr(mpc, arm)(spec, (0.3, -0.2), 0.4, -0.9)
        assert a == b, (arm, a, b)
        ts = np.linspace(0.0, a["T_run_s"], 97)
        for ga, gb in zip(pid._ref_traj(ts), mpc._ref_traj(ts)):
            assert np.allclose(ga, gb, atol=1e-9), (arm, ga, gb)
    # and the rectangle really is a rectangle under BOTH: 2.0 x 0.5, not 2 x 2
    flat = dict(square, rot_deg=0.0)
    for c in (pid, mpc):
        c.set_square_ned(flat, (0.3, -0.2), 0.0, -0.9)
        p, _yaw, _v, _r = c._ref_traj(np.linspace(0.0, 100.0, 2001))
        assert abs(np.ptp(p[0]) - 2.0) < 1e-6, np.ptp(p[0])
        assert abs(np.ptp(p[1]) - 0.5) < 1e-6, np.ptp(p[1])


def test_station_holds_the_z_it_had_when_start_was_pressed():
    """WHICH z, exactly. STATION commands all three axes, and the depth it
    regulates to is captured ONCE, at the START press — not at engage, and
    not re-read afterwards. Operator asked to keep that behaviour on
    2026-08-18, so it is worth a test rather than a reading of the source:
    the difference only shows when the vehicle has drifted in z between the
    two presses, which is exactly when it matters."""
    tmp = tempfile.mkdtemp(prefix="station_z_")
    w, _bus, _pilots, _logs = _worker(tmp)
    w.cfg.engage["settle_s"] = 0.0
    _feed_good_state(w)
    w.on_enable(True)
    w.set_engaged(True)
    assert w.engaged, w.reason
    w._warmup_left = 0
    # ENGAGE put the datum here, so the engage target's z is 0 by construction
    assert abs(float(w.ctrl._target[0][2])) < 1e-9, w.ctrl._target
    w._datum = {"p0": np.array([0.0, 0.0, 0.8]), "yaw0": 0.0,
                "Rz": rot_zyx(0.0, 0.0, 0.0)}
    # ...now let it sag 7 cm (NED z is down-positive) BEFORE START is pressed
    w._eta = np.array([0.0, 0.0, 0.07, 0.0, 0.0, 0.0])
    w.set_scenario({"shape": "station", "origin_tag": None,
                    "origin": [0.0, 0.0], "yaw_map_deg": 0.0})
    w.set_traj(True)
    assert w.station is not None, w.reason
    # THE CLAIM: the held depth is the START-press z (0.07), not the engage z
    assert abs(float(w.ctrl._target[0][2]) - 0.07) < 1e-9, w.ctrl._target
    assert abs(float(w.station["depth_ned"]) - 0.07) < 1e-9, w.station

    # ...and it is a HOLD, not a follow: more sag must not move the setpoint
    for z in (0.10, 0.15, 0.22):
        w._eta = np.array([0.0, 0.0, z, 0.0, 0.0, 0.0])
        w.tick()
        assert abs(float(w.ctrl._target[0][2]) - 0.07) < 1e-9, (
            f"the setpoint followed the vehicle to {w.ctrl._target[0][2]}")

    # A NUMBER in the config pins it instead, and then the START-press z is
    # irrelevant — this is the escape hatch the panel tooltip points at.
    w.set_traj(False)
    w.cfg.square["depth_ned"] = -0.30
    w._eta = np.array([0.0, 0.0, 0.42, 0.0, 0.0, 0.0])
    w.set_traj(True)
    assert abs(float(w.ctrl._target[0][2]) + 0.30) < 1e-9, w.ctrl._target
    w.teardown()


def test_station_skips_the_settle_but_a_path_does_not():
    """The settle holds the vehicle in the SLOWER regime — the setpoint is
    still leashed to approach_lead_m — and STATION has nothing to start
    afterwards, so arming early is what makes it arrive. A PATH still needs
    it: the square begins the instant it expires."""
    tmp = tempfile.mkdtemp(prefix="settle_")
    w, _bus, _pilots, _logs = _worker(tmp)
    assert w._settle_s("station") == 0.0
    assert w._settle_s("square") == w.cfg.engage["settle_s"]
    assert w._settle_s("line") == w.cfg.engage["settle_s"]
    assert w._settle_s(None) == w.cfg.engage["settle_s"]
    # ...unless imu_dr is on, whose static window IS the settle
    w.cfg.imu_dr = dict(w.cfg.imu_dr, enabled=True)
    assert w._settle_s("station") == w.cfg.engage["settle_s"]
    w.cfg.imu_dr = dict(w.cfg.imu_dr, enabled=False)

    # end to end: a station START within tolerance arms on the spot
    _feed_good_state(w)
    w.on_enable(True)
    w.set_engaged(True)
    w._warmup_left = 0
    w.set_scenario({"shape": "station", "origin_tag": None,
                    "origin": "current", "yaw_map_deg": 0.0})
    w.set_traj(True)
    assert w.station is not None and w._approach is None, w.reason
    w.teardown()


def test_the_approach_hands_the_controller_the_setpoints_own_speed():
    """Without this the approach settles wherever kp*lead balances drag —
    measured 0.084 m/s against a 0.10 command (2026-08-18). The path follower
    has always passed v_ned; the approach never did."""
    tmp = tempfile.mkdtemp(prefix="approach_v_")
    w, _bus, _pilots, _logs = _worker(tmp)
    _feed_good_state(w)
    w.on_enable(True)
    w.set_engaged(True)
    w._warmup_left = 0
    # a target far enough away that the approach runs
    w._datum = {"p0": np.array([0.0, 0.0, 0.8]), "yaw0": 0.0,
                "Rz": rot_zyx(0.0, 0.0, 0.0)}
    w._eta = np.zeros(6)
    w.set_scenario({"shape": "station", "origin_tag": None,
                    "origin": [3.0, 0.0], "yaw_map_deg": 0.0})
    w.set_traj(True)
    assert w._approach is not None, w.reason
    v_cmd = float(w.cfg.engage["approach_speed_m_s"])
    w._tick_approach(now())
    _p, _yaw, v = w.ctrl._target
    assert v[0] > 0.5 * v_cmd, f"no forward feedforward: {v}"
    assert abs(v[1]) < 1e-9 and abs(v[2]) < 1e-9, v
    assert v[0] <= v_cmd + 1e-9, v

    # ...and it TAPERS inside the controller's own preview, or an NMPC that
    # extrapolates along v_ref would be told to drive past the target.
    w._approach["sp"] = [2.99, 0.0]          # 1 cm to go
    w._tick_approach(now())
    _p2, _yaw2, v2 = w.ctrl._target
    assert v2[0] < v_cmd, f"feedforward did not taper near the target: {v2}"
    w.teardown()


def test_station_mode_holds_the_tag_with_an_absolute_map_heading():
    """STATION: no path at all — go to the tag, hold it, and face an ABSOLUTE
    map heading. yaw_map_deg is the point: a datum-relative angle means
    something different every time the vehicle engages."""
    tmp = tempfile.mkdtemp(prefix="station_")
    w, _bus, _pilots, logs = _worker(tmp)
    w.cfg.engage["settle_s"] = 0.0
    _feed_good_state(w)
    w.on_enable(True)
    w.set_engaged(True)
    assert w.engaged, w.reason
    w._warmup_left = 0
    # engaged facing 30 deg (map); ask to hold facing +y = 90 deg (map)
    w._datum = {"p0": np.array([0.0, 0.0, 0.8]), "yaw0": math.radians(30.0),
                "Rz": rot_zyx(0.0, 0.0, -math.radians(30.0))}
    w._eta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    w.set_scenario({"shape": "station", "origin_tag": None,
                    "origin": [0.0, 0.0], "yaw_map_deg": 90.0})
    w.set_traj(True)
    assert w.station is not None, w.reason
    assert not w.traj_on, "station must not arm a trajectory"
    assert w.phase == "station"
    # the DP target's yaw is the MAP heading expressed in the datum frame
    assert abs(w.ctrl._target[1] - math.radians(60.0)) < 1e-9, w.ctrl._target
    assert abs(w.station["yaw_map_deg"] - 90.0) < 1e-6
    # ...and it survives into the status the UI reads
    st = []
    w.bus.mpc_status.connect(st.append)
    w.tick()
    assert st and st[-1].phase == "station"
    assert (st[-1].scenario or {}).get("kind") == "station"
    # STOP leaves the station mode, not just the (absent) trajectory
    w.set_traj(False)
    assert w.station is None and w.phase == ""
    assert any("STATION" in m for _l, m in logs)
    w.teardown()


# =============================================================================
# IMU dead reckoning — the worker's half (the mechanization is test_imu_dr.py)
# =============================================================================
def _dr_worker(tmp, mode="shadow", control=False, attitude="gyro"):
    """A worker with the dead reckoner ON and a non-zero settle, since the
    anchor lands exactly on the settle->arm boundary."""
    w, bus, pilots, logs = _worker(tmp)
    w.opts.imu_dr = mode
    w.opts.imu_dr_attitude = attitude
    w.opts.imu_dr_control = control
    w.opts.c3_imu_rate = 200
    # Point at a path that cannot exist: whether the repo happens to carry a
    # real config/c3_imu_calib.json must not decide what these tests measure.
    w.cfg.imu_dr = {**w.cfg.imu_dr,
                    "calibration": str(Path(tmp) / "no_such_calib.json")}
    w._setup_imu_dr()
    return w, bus, pilots, logs


def _dr_samples(w, T=1.0, f=(0.0, 0.0, -9.80665), t0=None, hz=200.0):
    """Feed the worker a batch of synthetic C3 samples, as the backend would.

    The sample clock CONTINUES from the previous batch rather than restarting
    at ``now()``: ticks in a test run far faster than wall time, so restarting
    would hand the estimator overlapping timestamps and it would (correctly)
    reject them as out of order.
    """
    from rov_gui.state import ImuBatch

    n = int(round(T * hz)) + 1
    if t0 is None:
        t0 = getattr(w, "_test_imu_t", None)
        t0 = now() if t0 is None else t0 + 1.0 / hz
    arr = np.zeros((n, 7))
    arr[:, 0] = t0 + np.arange(n) / hz
    arr[:, 1:4] = np.asarray(f, float)
    w._test_imu_t = float(arr[-1, 0])
    w.on_camera_imu(ImuBatch(source="c3", samples=arr, n=n,
                             t_host_drain=arr[-1, 0],
                             t_device_last=float(arr[-1, 0]),
                             conn=Conn.ONLINE))
    return arr


def test_dead_reckoning_anchors_at_the_settle_to_arm_boundary():
    """Not at engage and not at START: at the END of the settle, which is the
    moment the vehicle is over the tag and stopped. And exactly once."""
    tmp = tempfile.mkdtemp(prefix="dranchor_")
    w, bus, _pilots, logs = _dr_worker(tmp)
    events = []
    bus.mpc_event.connect(events.append)
    w.cfg.engage["settle_s"] = 1.0
    _feed_good_state(w)
    w.on_enable(True)
    w.set_engaged(True)
    assert w.engaged, w.reason
    w._warmup_left = 0
    _dr_samples(w, T=1.0)
    w.tick()
    assert not w.dr.anchored, "must not anchor before the settle completes"
    # ...and the pre-anchor samples are being kept for the static window
    assert w.dr.static_window().shape[0] > 100

    w.set_scenario({"shape": "station", "origin_tag": None,
                    "origin": [w._datum["p0"][0], w._datum["p0"][1]],
                    "yaw_map_deg": 90.0})
    w.set_traj(True)                       # -> approach/settle, not armed yet
    assert w.station is None and w._approach is not None
    assert not w.dr.anchored, "still only settling"
    t = now()
    w._tick_approach(t)                    # arrives, settle timer starts
    assert w._approach["t_in"] is not None
    assert not w.dr.anchored, "anchored before the settle finished"
    w._tick_approach(t + 1.1)              # 1.1 s held > settle_s 1.0 -> go
    assert w.station is not None, "settle never completed"
    assert w.dr.anchored, "anchor must fire at _arm_path"
    assert w.dr.gyro_bias_source != "none"
    assert any("ANCHORED" in m for m in events), events
    t_anchor = w.dr.t_anchor
    for _ in range(5):
        _feed_good_state(w)
        w.tick()
    assert w.dr.t_anchor == t_anchor, "anchored more than once"
    w.teardown()


def test_shadow_mode_does_not_change_one_single_command():
    """The safety property that makes shadow free to leave on: with the
    estimator running but not driving, the commands are BIT-IDENTICAL.

    The worker's clock is STUBBED for this one. The tag state is
    velocity-bridged by ``t_now - fix.t_capture``, so on the real clock the
    extra microseconds the estimator costs would move the position a little
    and this test would be measuring scheduling jitter rather than behaviour.
    """
    import rov_gui.control.workers as W

    def fly(dr_on):
        tmp = tempfile.mkdtemp(prefix="drpar_")
        if dr_on:
            w, _b, pilots, _l = _dr_worker(tmp, mode="shadow")
        else:
            w, _b, pilots, _l = _worker(tmp)
        clock = {"t": 10_000.0}
        real_now, W.now = W.now, lambda: clock["t"]
        try:
            w.on_nav_fix(_fix([-2.0, 0.0, 0.8], 0.0, clock["t"]))
            w.on_vehicle_imu(_imu(depth=0.8, t=clock["t"]))
            w.on_telemetry(Telemetry(armed=True, mode="MANUAL",
                                     conn=Conn.ONLINE, stamp=clock["t"]))
            w.on_enable(True)
            w.set_engaged(True)
            assert w.engaged, w.reason
            w._warmup_left = 0
            for i in range(25):
                clock["t"] += 0.05
                t = clock["t"]
                w.on_nav_fix(_fix([-2.0 + 0.01 * i, 0.002 * i, 0.8], 0.0, t))
                w.on_vehicle_imu(_imu(depth=0.8, t=t))
                w.on_telemetry(Telemetry(armed=True, mode="MANUAL",
                                         conn=Conn.ONLINE, stamp=t))
                if dr_on:
                    _dr_samples(w, T=0.05, f=(0.3, 0.2, -9.6),
                                t0=t - 0.05 + 0.005)
                w.tick()
        finally:
            W.now = real_now
        out = [(p.surge, p.sway, p.heave, p.yaw) for p in pilots]
        w.teardown()
        return out

    off, on = fly(False), fly(True)
    assert off and len(off) == len(on), (len(off), len(on))
    assert off == on, "shadow mode perturbed the commands"


def test_control_mode_feeds_the_estimate_to_both_controller_and_path_cursor():
    """In control mode the controller AND the reference cursor must see the
    same (dead-reckoned) state. Advancing the path on truth while flying on
    the estimate would measure neither."""
    tmp = tempfile.mkdtemp(prefix="drctl_")
    w, _bus, _pilots, _logs = _dr_worker(tmp, mode="control", control=True)
    assert w.dr_control, "--imu-dr-control should have armed it"
    _feed_good_state(w)
    w.on_enable(True)
    w.set_engaged(True)
    w._warmup_left = 0
    # Anchor by hand at the origin, then push a large bias so the estimate
    # walks visibly away from the (stationary) tag solution.
    _dr_samples(w, T=1.0)
    w.tick()
    w._anchor_dr()
    assert w.dr.anchored
    seen = []
    real_step = w.ctrl.step
    w.ctrl.step = lambda eta, nu, nudot, t: (seen.append(np.array(eta)),
                                             real_step(eta, nu, nudot, t))[1]
    for _ in range(20):
        _feed_good_state(w)
        _dr_samples(w, T=0.05, f=(1.0, 0.0, -9.80665))
        w.tick()
    w.ctrl.step = real_step
    assert seen, "controller was never stepped"
    st = w._dr_last
    assert st["ok"], st["why"]
    # what the controller got IS the estimate, not the tag
    assert abs(seen[-1][0] - st["p_ned"][0]) < 1e-9, (seen[-1], st["p_ned"])
    assert st["p_ned"][0] > 0.001, "the injected bias should have moved it"
    # ...and z came from the barometer, not from the accelerometer
    assert abs(seen[-1][2] - w._eta[2]) < 1e-9
    w.teardown()


def test_a_config_file_alone_cannot_arm_the_closed_loop():
    """`--imu-dr control` on the command line IS explicit and arms it on its
    own. What must NOT arm it is hw_mpc.yaml left in that state with nobody
    asking — that is the case with no human in the moment."""
    tmp = tempfile.mkdtemp(prefix="drgate_")

    # 1) config says control, command line says nothing -> SHADOW
    w, _bus, _pilots, logs = _worker(tmp)
    w.opts.imu_dr = None
    w.opts.imu_dr_control = False
    w.cfg.imu_dr = {**w.cfg.imu_dr, "enabled": True, "mode": "control",
                    "attitude": "gyro"}
    w._setup_imu_dr()
    assert w.dr is not None, "the estimator should still run"
    assert not w.dr_control, "a config file alone must not arm it"
    assert any("--imu-dr control" in m for _l, m in logs), [m for _l, m in logs]

    # 2) ...and the operator opting in on the command line does arm it
    w.opts.imu_dr = "control"
    w._setup_imu_dr()
    assert w.dr_control, "--imu-dr control should be enough on its own"

    # 3) ...as does the explicit gate flag beside a control config
    w.opts.imu_dr = None
    w.opts.imu_dr_control = True
    w._setup_imu_dr()
    assert w.dr_control
    w.teardown()


def test_the_status_and_csv_carry_the_estimate_and_the_tag_stays_truth():
    tmp = tempfile.mkdtemp(prefix="drcsv_")
    w, bus, _pilots, _logs = _dr_worker(tmp)
    st = []
    bus.mpc_status.connect(st.append)
    _feed_good_state(w)
    w.on_enable(True)
    w.set_engaged(True)
    w._warmup_left = 0
    _dr_samples(w, T=1.0)
    w.tick()
    w._anchor_dr()
    for _ in range(10):
        _feed_good_state(w)
        _dr_samples(w, T=0.05, f=(0.5, 0.0, -9.80665))
        w.tick()
    s = st[-1]
    assert s.dr_ok and s.dr_mode == "shadow" and s.dr_source == "c3"
    assert s.p_dr_flu is not None and s.dr_err_m is not None
    assert s.dr_hz > 100.0, s.dr_hz
    assert s.dr_n > 100, s.dr_n
    # p_flu is still the TAG solution even with the estimator running
    assert abs(s.p_flu[0] - w._eta[0]) < 1e-9
    assert s.rp_residual_deg is not None
    path = w._csv_path
    w.disengage("test over")
    rows = Path(path).read_text().splitlines()
    cols = rows[0].split(",")
    body = [dict(zip(cols, r.split(","))) for r in rows[1:]]
    live = [r for r in body if r["dr_ok"] == "1"]
    assert live, "no row carried a live estimate"
    assert float(live[-1]["dr_t_s"]) > 0.0
    assert abs(float(live[-1]["dr_px"])) > 1e-6
    # dr_pz_imu is kept even though z came from the barometer
    assert live[-1]["dr_pz_imu"] not in ("", "nan")
    w.teardown()


def test_a_tag_dropout_still_writes_a_row_when_dead_reckoning():
    """Drift across a dropout is what such a run is FOR, so the gap in the
    file that the tag-only path would leave has to be filled."""
    tmp = tempfile.mkdtemp(prefix="drgap_")
    w, _bus, _pilots, _logs = _dr_worker(tmp)
    _feed_good_state(w)
    w.on_enable(True)
    w.set_engaged(True)
    w._warmup_left = 0
    _dr_samples(w, T=1.0)
    w.tick()
    w._anchor_dr()
    path = w._csv_path
    n0 = w._rows
    # A STALE fix, which is what a real dropout looks like: the localizer
    # keeps missing rather than vanishing, and the stale gate debounces for
    # tag_stale_hold_s before it disengages. Those are the ticks with no tag
    # state and a live estimate — exactly the ones that used to write nothing.
    w.fix = _fix([-2.0, 0.0, 0.8], 0.0, now() - 3.0)
    for _ in range(4):
        _dr_samples(w, T=0.05)
        w.tick()
    assert w.engaged, "the stale gate should still be debouncing"
    assert w._rows > n0, "no rows written across the dropout"
    w._close_csv()
    rows = Path(path).read_text().splitlines()
    cols = rows[0].split(",")
    last = dict(zip(cols, rows[-1].split(",")))
    assert last["px"] == "nan", "the tag columns must go nan, not stale"
    assert last["dr_ok"] == "1", "the estimate is still live"
    w.teardown()


def test_a_dead_sample_stream_reads_as_not_ok_not_as_a_perfect_hold():
    tmp = tempfile.mkdtemp(prefix="drdead_")
    w, bus, _pilots, _logs = _dr_worker(tmp)
    st = []
    bus.mpc_status.connect(st.append)
    _feed_good_state(w)
    w.on_enable(True)
    w.set_engaged(True)
    w._warmup_left = 0
    # Anchor on samples that are ALREADY old, so the very next tick is stale.
    _dr_samples(w, T=1.0, t0=now() - 5.0)
    w.tick()
    w._anchor_dr()
    _feed_good_state(w)
    w.tick()
    s = st[-1]
    assert not s.dr_ok, "a starved estimator must not report ok"
    assert "stale" in s.dr_note, s.dr_note
    assert s.p_dr_flu is None, "and it must not draw a marker"
    w.teardown()


def test_run_meta_records_the_estimator_as_a_record_boundary():
    tmp = tempfile.mkdtemp(prefix="drmeta_")
    w, _bus, _pilots, _logs = _dr_worker(tmp, attitude="ahrs")
    _feed_good_state(w)
    w.on_enable(True)
    w.set_engaged(True)
    _dr_samples(w, T=1.0)
    w.tick()
    w._anchor_dr()
    w.tick()
    m = w._run_meta("test")
    # 8 since 2026-08-30: + plan_stream (always written), the replay mission
    # shape and the CSV plan_id/ref_src/grip_cmd columns. 7 added tick_ms and
    # the follow boundary; 6 added object_nav and the ten obj_*/follow_*
    # columns; 5 added station_bridge (bridge_s / bridge_tier).
    assert m["schema_version"] == 8
    # Written even with no --pose, for the imu_dr reason: an absent key cannot
    # tell "this build had no object nav" from "it was switched off".
    assert m["object_nav"]["enabled"] is False
    d = m["imu_dr"]
    assert d["enabled"] and d["mode"] == "shadow"
    assert d["attitude"] == "ahrs" and d["z_source"] == "pressure"
    assert d["anchored"] and d["source"] == "c3"
    # no calibration file in a temp dir: that must be VISIBLE, not implied
    assert d["calibration"]["sha1"] is None
    assert d["gyro_bias_source"] != "none"
    assert d["requested_rate_hz"] == 200
    # ...and the camera tilt, the other new record boundary
    assert m["hardware"]["cam_tilt_deg"] == 0.0
    assert m["hardware"]["axis_cap_used"] == w.cfg.axis_cap
    w.teardown()


def test_a_typo_in_the_imu_dr_config_raises_instead_of_being_ignored():
    """The failure it prevents: the operator believes the estimator is on and
    configured, and flies a pool run with it off or on a default."""
    from rov_gui.control.geometry import MpcConfig

    tmp = tempfile.mkdtemp(prefix="drtypo_")
    p = Path(tmp) / "mpc.yaml"
    p.write_text("mode: dobmpc\nimu_dr:\n  enabled: true\n"
                 "  attitude_mode: ahrs\n")
    try:
        MpcConfig.load(str(p))
        raise AssertionError("an unknown imu_dr key was accepted")
    except ValueError as e:
        assert "attitude_mode" in str(e), str(e)
    p.write_text("mode: dobmpc\nimu_dr:\n  attitude: quaternion\n")
    try:
        MpcConfig.load(str(p))
        raise AssertionError("an invalid attitude was accepted")
    except ValueError as e:
        assert "gyro|ahrs|vehicle" in str(e), str(e)


def test_the_main_camera_tilt_rotates_the_extrinsic_and_defaults_to_level():
    """The C3 branch had no tilt term at all until 2026-08-17. The default
    MUST stay 0.0 — a non-zero default would reinterpret every recorded run."""
    cfg = NavConfig()
    assert cfg.cam_tilt_deg == 0.0
    R0, t0 = cfg.R_t_frd_cam("main")
    cfg.cam_tilt_deg = 40.0
    R40, t40 = cfg.R_t_frd_cam("main")
    assert np.allclose(t0, t40), "tilt is a pure rotation; t must not move"
    # the optical axis (camera +z in body FRD) dips by the tilt angle
    down0 = math.degrees(math.asin(float(R0[2, 2])))
    down40 = math.degrees(math.asin(float(R40[2, 2])))
    assert abs((down40 - down0) - 40.0) < 1e-6, (down0, down40)
    # and the SAME helper now serves the second camera it came from
    sc = NavConfig()
    sc.second_cam = {**sc.second_cam, "tilt_deg": 40.0}
    Rs, _ts = sc.R_t_frd_cam("second")
    assert abs(math.degrees(math.asin(float(Rs[2, 2]))) - 40.0) < 1e-6


def test_datum_to_map_transform_survived_the_geofence_removal():
    """_to_map_xy was written for the map-frame geofence and OUTLIVED it: the
    START-refusal messages and the plot both still turn datum coordinates into
    pool coordinates with it. Deleting the fence must not have taken it."""
    tmp = tempfile.mkdtemp(prefix="frame_")
    w, _bus, _pilots, _logs = _worker(tmp)
    # engaged at map (0.9, 0.0) facing +y: the datum frame is rotated 90 deg
    w._datum = {"p0": np.array([0.9, 0.0, 0.5]), "yaw0": math.pi / 2,
                "Rz": rot_zyx(0.0, 0.0, -math.pi / 2)}
    mx, my = w._to_map_xy(0.0, -0.5)
    assert abs(mx - 1.4) < 1e-9 and abs(my) < 1e-9, (mx, my)
    mx, my = w._to_map_xy(0.5, 0.0)
    assert abs(mx - 0.9) < 1e-9 and abs(my - 0.5) < 1e-9, (mx, my)
    assert not hasattr(w, "_fence_ok"), "the fence test came back"
    w.teardown()


def test_mission_fields_are_typeable_but_never_keep_the_pilot_keys():
    """The tag/distance fields accept typed digits — and hand the keyboard
    straight back on any key that is not part of typing a number, re-sending
    it to the window. Otherwise reaching for W after typing a tag id would do
    NOTHING: the field would swallow it and the vehicle would sit there with
    no visible cause (which is the reason everything else here is NoFocus)."""
    from rov_gui.qt import Qt, QtCore, QtGui
    from rov_gui.tests.test_offline import Opts as BaseOpts, _pump
    from rov_gui.window import MainWindow

    class Opts(BaseOpts):
        mpc = True
        nav_config = str(ROOT / "config" / "hw_nav.yaml")
        mpc_config = str(ROOT / "config" / "hw_mpc.yaml")
        nav_geometry = None

    app = _app()
    win = MainWindow(Opts())
    win.resize(1600, 1000)
    win.show()
    _pump(app, 200)
    panel = win._traj_panel
    box = panel.tag_box
    assert box.focusPolicy() == Qt.FocusPolicy.ClickFocus
    assert not box.keyboardTracking(), "one scenario per keystroke"

    def press(widget, key, text=""):
        ev = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, key,
                             Qt.KeyboardModifier.NoModifier, text)
        return QtWidgets.QApplication.sendEvent(widget, ev)

    box.setFocus(Qt.FocusReason.MouseFocusReason)
    _pump(app, 50)
    assert box.hasFocus()
    box.lineEdit().selectAll()
    for k, t in ((Qt.Key.Key_1, "1"), (Qt.Key.Key_0, "0"), (Qt.Key.Key_7, "7")):
        press(box, k, t)
    _pump(app, 50)
    assert box.lineEdit().text().endswith("107"), box.lineEdit().text()
    assert box.hasFocus(), "digits must stay in the field"

    # a PILOT key: focus is released AND the window gets the key
    win.teleop.enable.setChecked(True)
    press(box, Qt.Key.Key_W, "w")
    _pump(app, 50)
    assert not box.hasFocus(), "the field kept the keyboard after a pilot key"
    assert win.teleop.current().surge > 0.0, "W never reached the pilot handler"
    win.teleop.key_event(Qt.Key.Key_W, False, False)

    # Enter commits and releases too
    box.setFocus(Qt.FocusReason.MouseFocusReason)
    _pump(app, 50)
    press(box, Qt.Key.Key_Return)
    _pump(app, 50)
    assert not box.hasFocus()
    win.shutdown()


def test_no_geofence_box_is_drawn_on_engage():
    """Operator, 2026-08-14: the box that appeared on START is gone. Pinned
    because a plot that draws a boundary is promising a limit, and there is no
    longer a limit behind it — POOL is the only rectangle left, and it is a
    picture of the wall rather than something anything enforces."""
    from rov_gui.state import MpcStatus
    from rov_gui.widgets.trajectory import TrajectoryView

    _app()
    v = TrajectoryView()
    assert not hasattr(v, "set_geofence"), "the fence setter came back"
    drawn = []
    # z since the 3-D view: _px takes an optional depth now, and a two-arg
    # spy silently ate it as the keyword default and then called a float.
    v._px = lambda x, y, z=0.0, _o=v._px: (drawn.append((x, y)) or _o(x, y, z))
    v.resize(400, 300)
    v.add_status(MpcStatus(engaged=True, p_flu=(0.0, 0.0, 0.0)))
    v.grab()
    # With no pool set, _fit falls back to a +-2 m box; nothing may be drawn
    # at the old fence extents, and no dashed rectangle may appear at all.
    assert drawn, "the plot painted nothing at all"
    assert not any(abs(abs(x) - 1.2) < 1e-9 and abs(abs(y) - 1.2) < 1e-9
                   for x, y in drawn), "a geofence box was drawn on engage"


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


def test_axis_slew_limit_clips_thrashing_without_touching_normal_commands():
    """The rate limit must be a NET, not a filter: inactive when the controller
    is behaving, decisive when it is not.

    Sizing evidence (2026-08-17): with deadband compensation off the command's
    own |d(axis)/dt| p95 is 0.93/s in closed loop, so a 1.5/s limit never
    engages and tracking is bit-identical; the run that shook the vehicle
    (.../0817_124624) hit 10.51/s on yaw, which this clips to 1.5.
    """
    from rov_gui.control.allocation import slew_axes
    from rov_gui.state import PilotInput

    dt, rate = 0.05, 1.5
    step = rate * dt                     # 0.075 per tick

    def ax(v):
        return PilotInput(surge=v, sway=v, heave=v, yaw=v, source="mpc")

    # a full reversal is spread over ticks instead of arriving at once
    prev = (0.4, 0.4, 0.4, 0.4)
    out = slew_axes(ax(-0.4), prev, rate, dt)
    assert abs(out.yaw - (0.4 - step)) < 1e-9, out.yaw
    # ...a small, normal change passes untouched
    out = slew_axes(ax(0.44), prev, rate, dt)
    assert abs(out.surge - 0.44) < 1e-9
    # ...and 0 (or no history) disables it entirely
    assert slew_axes(ax(-0.4), prev, 0.0, dt).yaw == -0.4
    assert slew_axes(ax(-0.4), None, rate, dt).yaw == -0.4
    # it can never exceed the PilotInput clamp
    assert abs(slew_axes(ax(9.0), (1.0, 1.0, 1.0, 1.0), 100.0, dt).surge) <= 1.0


def test_waypoint_reference_stops_at_every_vertex():
    """Corner handling option A: an un-filleted vertex is taken at a stop.

    The reference brakes into the corner under a_long, crosses it at a crawl,
    and accelerates out — so the 90-degree direction change happens while the
    speed is essentially zero and the VELOCITY reference stays continuous.
    That is what makes a sharp rectangle realizable at all; the 2 cm error
    floor this repo measured on sharp corners was the reference asking for a
    velocity reversal in one sample.
    """
    from rov_gui.control.path_geometry import path_from_scenario

    scen = {"kind": "square", "size": 1.0, "size_y": 1.0, "speed": 0.20,
            "laps": 2, "origin_ned": [0.0, 0.0], "depth_ned": 0.0,
            "rot_deg": 0.0}
    p = path_from_scenario(scen, fillet_m=0.0)
    s_g, v_g = p.speed_profile(0.20, 0.05, 0.05)
    assert abs(v_g.max() - 0.20) < 1e-9, "the straights must reach the command"
    for vertex in (1.0, 2.0, 3.0, 4.0):
        i = int(np.argmin(np.abs(s_g - vertex)))
        assert v_g[i] < 0.05, f"vertex at {vertex} m runs at {v_g[i]:.3f} m/s"
        assert v_g[i] > 0.0, "an exact zero is a fixed point the cursor never leaves"
    mid = int(np.argmin(np.abs(s_g - 0.5)))
    assert v_g[mid] > 0.19, "the straight must not be slowed by the corners"


def test_waypoint_plan_gives_the_pid_exactly_what_it_had():
    """One stage of the plan must equal the setpoint the previous code sent,
    or 'PID vs MPC' silently becomes a geometry comparison."""
    from rov_gui.control.path_geometry import PathCursor, path_from_scenario

    scen = {"kind": "square", "size": 1.0, "size_y": 1.0, "speed": 0.20,
            "laps": 2, "origin_ned": [0.0, 0.0], "depth_ned": 0.0,
            "rot_deg": 0.0}
    p = path_from_scenario(scen, fillet_m=0.0)
    s_g, v_g = p.speed_profile(0.20, 0.05, 0.05)
    cur = PathCursor(p, lead_m=0.10, s_grid=s_g, v_grid=v_g)
    for _ in range(120):
        cur.step([0.40, 0.0, 0.5], 0.05)
    plan = cur.plan(1, 0.05, 0.5, 0.0, False)
    target, psi, _al, _cr, v_ref = cur.step([0.40, 0.0, 0.5], 0.05)
    assert plan.p_ned.shape == (3, 1) and plan.v_ned.shape == (3, 1)
    assert abs(plan.p_ned[0, 0] - target[0]) < 1e-9
    assert abs(plan.p_ned[1, 0] - target[1]) < 1e-9
    assert abs(plan.p_ned[2, 0] - 0.5) < 1e-9
    assert abs(np.hypot(plan.v_ned[0, 0], plan.v_ned[1, 0]) - v_ref) < 1e-9


def test_waypoint_horizon_brakes_into_the_corner_instead_of_driving_through():
    """The defect this replaces: because set_target_ned clears the trajectory
    sampler, the tracking NMPC was extrapolating ONE setpoint along a straight
    ray for all 61 stages and had no idea a corner existed. The plan must both
    bend and slow."""
    from rov_gui.control.path_geometry import PathCursor, path_from_scenario

    scen = {"kind": "square", "size": 1.0, "size_y": 1.0, "speed": 0.20,
            "laps": 2, "origin_ned": [0.0, 0.0], "depth_ned": 0.0,
            "rot_deg": 0.0}
    p = path_from_scenario(scen, fillet_m=0.0)
    s_g, v_g = p.speed_profile(0.20, 0.05, 0.05)
    cur = PathCursor(p, lead_m=0.10, s_grid=s_g, v_grid=v_g)
    cur.theta, cur._theta_cmd = 0.60, 0.70      # vertex is at 1.00 m
    plan = cur.plan(61, 0.05, 0.0, 0.0, False)
    speed = np.hypot(np.diff(plan.p_ned[0]), np.diff(plan.p_ned[1])) / 0.05
    assert speed[-1] < 0.4 * speed[0], (
        f"reference does not brake into the vertex: "
        f"{speed[0]:.3f} -> {speed[-1]:.3f} m/s")
    # ...and it never overshoots the corner it is braking for
    assert plan.p_ned[0].max() <= 1.0 + 1e-6


def test_mpc_mode_receives_a_horizon_not_a_bare_setpoint():
    """`set_target_ned` ends with `self._ref_traj = None`, so calling it every
    tick — which the worker used to do — left the NMPC with no path geometry
    at all. Assert the worker now installs a plan instead."""
    tmp = tempfile.mkdtemp(prefix="waypoint_")
    w, _bus, _pilots, _logs = _worker(tmp)
    w.cfg.engage["settle_s"] = 0.0
    w.cfg.path_following = True
    w.cfg.path_fillet_m = 0.0
    _feed_good_state(w)
    w.on_enable(True)
    w.set_engaged(True)
    w._warmup_left = 0
    w._datum = {"p0": np.zeros(3), "yaw0": 0.0, "Rz": rot_zyx(0.0, 0.0, 0.0)}
    w._eta = np.zeros(6)
    w.set_scenario({"shape": "square", "origin_tag": None, "origin": [0.0, 0.0],
                    "size": 1.0, "size_y": 1.0, "speed": 0.1, "laps": 1})
    w.set_traj(True)
    assert w.traj_on, w.reason

    # the stub declares 1 stage (it stands in for the PID); a real NMPC asks
    # for N+1. Subclass rather than patch — path_plan_steps is a property.
    steps = 61
    seen = {}
    real = w.ctrl

    class Horizon(type(real)):
        path_plan_steps = property(lambda self: steps)
        path_plan_dt = property(lambda self: 0.05)

        def set_path_plan_ned(self, plan):
            seen["plan"] = plan
            self._path_plan = plan

    real.__class__ = Horizon
    t = now()
    for k in range(1, 12):
        w._advance_path_clock(t + 0.05 * k, {"eta": np.zeros(6)})
    plan = seen.get("plan")
    assert plan is not None, "the worker still sends a bare setpoint"
    assert np.asarray(plan.p_ned).shape == (3, steps)
    assert np.asarray(plan.v_ned).shape == (3, steps)
    assert np.asarray(plan.yaw_ned).size == steps
    # the horizon must actually cover ground, not repeat one point
    assert np.ptp(plan.p_ned[0]) > 0.01, "the plan is a single repeated point"
    w.teardown()


# =============================================================================
# CIRCLE — the tag is on the RIM (operator, 2026-08-17), never the centre
# =============================================================================
def _circle_ned(scen_kw, origin_ned=(1.0, 2.0), yaw=0.0, depth=-0.4,
                n=721, laps=1):
    """Sample one placed circle in NED (the frame the pool is measured in)."""
    from rov_gui.control.reference import place_circle_ned

    kw = {"radius": 0.5, "speed": 0.05, "laps": laps}
    kw.update(scen_kw)
    fn, sc = place_circle_ned(kw, origin_ned, yaw, depth, dt=0.05)
    ts = np.linspace(0.0, sc["T_run_s"], n)
    p, yaw_flu, v, r = fn(ts)
    # world-FLU -> NED display mirror, S = diag(1,-1,-1); yaw and r negate
    return (sc, ts, np.vstack([p[0], -p[1], -p[2]]), -yaw_flu,
            np.vstack([v[0], -v[1], -v[2]]), -r)


def test_circle_puts_the_entered_tag_on_the_rim_not_at_the_centre():
    """THE requirement, in the frame the operator reads off the plot.

    The top-down plot draws screen up = +x_ned, so "the tag is the bottom of
    the circle" means the tag is the path's MINIMUM-x point and the centre
    sits one radius up-plot from it. A circle centred on the tag would pass
    every other test in this file — same radius, same speed, same closure —
    and be the one shape the operator explicitly did not ask for, so this is
    pinned on the centre and on the minimum, not on the radius.
    """
    R = 0.5
    sc, _ts, ned, _yaw, _v, _r = _circle_ned({"radius": R}, origin_ned=(1.0, 2.0))
    assert sc["kind"] == "circle" and abs(sc["radius"] - R) < 1e-12
    # starts AT the tag
    assert abs(ned[0, 0] - 1.0) < 1e-9 and abs(ned[1, 0] - 2.0) < 1e-9
    # ...which is the minimum-x point of the whole path
    assert abs(ned[0].min() - 1.0) < 1e-9, ned[0].min()
    assert abs(ned[0].max() - (1.0 + 2 * R)) < 1e-9
    # ...and the centre is one radius along +x from it
    cx, cy = 1.0 + R, 2.0
    assert np.abs(np.hypot(ned[0] - cx, ned[1] - cy) - R).max() < 1e-9
    # rot_deg swings the CENTRE around the tag; the tag itself never moves
    _sc2, _t2, ned2, _y2, _v2, _r2 = _circle_ned(
        {"radius": R, "rot_deg": 90.0}, origin_ned=(1.0, 2.0))
    assert abs(ned2[0, 0] - 1.0) < 1e-9 and abs(ned2[1, 0] - 2.0) < 1e-9
    assert np.abs(np.hypot(ned2[0] - 1.0, ned2[1] - (2.0 + R)) - R).max() < 1e-9


def test_circle_ref_holds_the_commanded_speed_and_closes_its_lap():
    """No vertex and no reversal, so unlike the square and the line the
    circle's reference speed never has to drop — that is the whole reason it
    is worth flying (memory: square-corner-error-floor)."""
    R, V = 0.4, 0.06
    sc, ts, ned, yaw, v, _r = _circle_ned({"radius": R, "speed": V}, laps=2)
    assert abs(sc["T_run_s"] - 2 * 2 * math.pi * R / V) < 1e-9
    sp = np.hypot(v[0], v[1])
    assert abs(sp.min() - V) < 1e-9 and abs(sp.max() - V) < 1e-9
    # velocity is tangent to the circle: radial component is zero
    cx, cy = 1.0 + R, 2.0
    radial = ((ned[0] - cx) * v[0] + (ned[1] - cy) * v[1]) / R
    assert np.abs(radial).max() < 1e-9, np.abs(radial).max()
    # the lap closes, and depth is held
    assert math.hypot(ned[0, -1] - ned[0, 0], ned[1, -1] - ned[1, 0]) < 1e-9
    assert np.abs(ned[2] + 0.4).max() < 1e-12
    # crab by default: heading is the fixed one, not the tangent
    assert np.abs(yaw).max() < 1e-12


def test_circle_heading_follow_faces_along_the_tangent():
    """With heading_follow the yaw command is the path tangent, slewed at
    yaw_rate_deg_s. A lap therefore accumulates a full turn, and the profile
    must NOT wrap while doing it — a preview that dropped 2*pi at the lap
    boundary would command a 360-degree spin the live loop never makes."""
    R, V = 0.5, 0.05
    # TWO laps sampled on a grid whose midpoint is exactly one lap in, so the
    # revolution can be measured between two SETTLED instants. Measuring it
    # from t=0 instead would be off by the initial slew: the command starts at
    # the heading the vehicle is holding, not already on the tangent.
    n = 721
    _sc, ts, _ned, yaw, v, r = _circle_ned(
        {"radius": R, "speed": V, "heading_follow": True,
         "yaw_rate_deg_s": 60.0}, origin_ned=(0.0, 0.0), yaw=0.0,
        n=n, laps=2)
    tangent = np.arctan2(v[1], v[0])
    d = np.abs(np.arctan2(np.sin(yaw - tangent), np.cos(yaw - tangent)))
    settled = ts > 3.0                       # the initial slew from yaw 0
    assert np.degrees(d[settled]).max() < 0.5, np.degrees(d[settled]).max()
    half = (n - 1) // 2                      # exactly one lap of samples
    assert abs(ts[half] - _sc["T_run_s"] / 2) < 1e-9
    assert abs((yaw[-1] - yaw[half]) - 2 * math.pi) < 1e-3, \
        yaw[-1] - yaw[half]
    # ...monotonically, ONCE settled. The first 1.5 s legitimately runs the
    # other way: the command starts at the held heading and slews DOWN to the
    # tangent 90 degrees away before the circle starts turning it back up.
    assert np.all(np.diff(yaw[settled]) >= -1e-9), "the heading profile wrapped"
    # ...and the yaw RATE is the circle's own, speed/R
    assert abs(np.median(r[settled]) - V / R) < 1e-3


def test_every_controller_can_arm_every_shape():
    """A shape the MPC can arm and the PID cannot is a mission that dies after
    the approach and the 10 s settle, with nothing on the panel to say why.
    This repo has paid for it twice — HwPid had no set_line_ned at all, and
    before that it ignored size_y and flew a square while the MPC flew the
    rectangle (reference.py's placement header). So the surface is pinned
    rather than trusted: whatever the worker can hand down, all three
    controllers take.

    Class-level, deliberately — building an HwDobMpc or an HwMpcc drags in
    acados, and the failure this guards against is a MISSING METHOD.
    """
    from rov_gui.control.mpc_bridge import HwDobMpc
    from rov_gui.control.mpcc_bridge import HwMpcc
    from rov_gui.control.pid import HwPid

    for cls in (HwDobMpc, HwMpcc, HwPid, StubCtrl):
        for name in ("set_target_ned", "set_square_ned", "set_line_ned",
                     "set_circle_ned"):
            assert callable(getattr(cls, name, None)), \
                f"{cls.__name__} cannot arm {name}"


def test_circle_arms_through_the_worker_and_counts_revolutions():
    """END TO END through MpcWorker: the panel's scenario override, the
    map->datum rotation, the armed geometry and the lap counter."""
    from rov_gui.control.workers import _scenario_lap

    with tempfile.TemporaryDirectory() as tmp:
        w, _bus, _pilots, _logs = _worker(tmp)
        _feed_good_state(w)
        w.on_enable(True)
        w.set_engaged(True)
        assert w.engaged, w.reason
        w._warmup_left = 0
        w._eta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        w.set_scenario({"shape": "circle", "origin_tag": None,
                        "origin": "current", "radius": 0.6, "speed": 0.05})
        w.set_traj(True)
        assert w.traj_on, w.reason
        sc = w.ctrl.scenario
        assert sc["kind"] == "circle" and abs(sc["radius"] - 0.6) < 1e-9
        assert np.allclose(sc["origin_ned"], [0.0, 0.0], atol=1e-9)
        assert w.phase == "circle", w.phase
        # the wall-clock backstop is sized on the FEASIBLE traversal, and a
        # circle's curvature limit is what makes that longer than 2*pi*R/v
        assert sc["path_min_duration_s"] > 0.0
        assert sc["path_timeout_s"] > sc["path_min_duration_s"]
        # a lap is one revolution
        per = 2 * math.pi * 0.6 / 0.05
        assert _scenario_lap(sc, per * 0.99) == 0
        assert _scenario_lap(sc, per * 1.01) == 1
        assert _scenario_lap(sc, per * 3.5) == 3
        w.teardown()


def test_circle_geometry_survives_the_engage_heading():
    """rot_deg is a MAP bearing, exactly like the rectangle's and the line's:
    the same number must mean the same physical circle however the vehicle
    happened to be pointing at START. Two engagements 90 degrees apart, one
    circle."""
    from rov_gui.control.workers import _wrap_pi

    with tempfile.TemporaryDirectory() as tmp:
        placed = []
        for yaw0 in (0.0, math.pi / 2):
            w, _bus, _pilots, _logs = _worker(tmp)
            _feed_good_state(w)
            w.on_enable(True)
            w.set_engaged(True)
            w._warmup_left = 0
            w._datum = {"p0": np.array([0.0, 0.0, 0.8]), "yaw0": yaw0,
                        "Rz": rot_zyx(0.0, 0.0, yaw0)}
            w._eta = np.zeros(6)
            w.set_scenario({"shape": "circle", "origin_tag": None,
                            "origin": "current", "radius": 0.5,
                            "speed": 0.05})
            w.set_traj(True)
            assert w.traj_on, w.reason
            sc = w.ctrl.scenario
            # datum-frame rot, mapped back to the MAP frame, is the 0 asked for
            placed.append(_wrap_pi(math.radians(sc["rot_deg"]) + yaw0))
            w.teardown()
        assert abs(_wrap_pi(placed[0] - placed[1])) < 1e-9, placed


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
