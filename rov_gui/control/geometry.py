#!/usr/bin/env python3
"""geometry.py — frames, extrinsics, geofence and the two config files.

Everything that turns "a tag in a camera image" into "the vehicle in a NED
world box" is decided HERE, from YAML, and nowhere else:

* which tag map defines the world (floor survey vs a single wall tag);
* the map->NED remap ``R_ned_map`` (presets below, overridable);
* the camera extrinsic ``T_body_cam`` (seeded from the sim's measured
  ``c3_payload_frames.json``; a re-mount writes its own numbers into YAML);
* the geofence box: START TRAJ refuses a square whose corners leave it, and
  the vehicle exiting box+margin at runtime disengages the controller.

Frame cheat-sheet (the repo has been burned twice by implied conventions):

    tag/map frame   +z INTO the tag face (tagslam object points, +y down)
                    -> floor tags print-up have +z DOWN: map is NED-like.
    camera optical  +x right, +y down, +z forward (OpenCV).
    body FLU        +x fwd, +y left, +z up (MuJoCo/sim, c3_payload_frames).
    body FRD / NED  +x fwd, +y right, +z down (Fossen, ArduSub, the MPC).
    FLU <-> FRD     S = diag(1,-1,-1), same S as dobmpc/frames.py.

Wall presets (tag upright on a vertical wall, print facing the pool):
    x_into_wall   NED x points INTO the wall (ROV in front has x < 0; facing
                  the wall is yaw 0)          R rows: x=[0,0,1] y=[1,0,0] z=[0,1,0]
    x_out_of_wall NED x points OUT into the pool (ROV has x > 0; facing the
                  wall is yaw pi)             R rows: x=[0,0,-1] y=[-1,0,0] z=[0,1,0]
Both keep z_ned = +y_tag = down for an upright tag; the stationary check in
the state assembler (tag roll/pitch vs ATTITUDE) is what catches a mounting
that violates that assumption.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

S_FLU_FRD = np.diag([1.0, -1.0, -1.0])     # same involution as dobmpc.frames.S

WALL_PRESETS = {
    "x_into_wall": np.array([[0.0, 0.0, 1.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0]]),
    "x_out_of_wall": np.array([[0.0, 0.0, -1.0],
                               [-1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0]]),
}

# Fallback extrinsic if the measured JSON is missing: the same numbers, frozen
# 2026-08-12 from bluerov2_mujoco_marinegym/meshes/c3_payload_frames.json
# (Onshape registration 2026-07-19; base_link FLU, origin = heavy_c3 COM).
_C3_T_FLU = (0.23949, 0.00547, -0.15537)
_C3_XYAXES = (-0.0, -1.0, -0.0, -0.0056, 0.0, 0.99998)


def R_flu_cam_from_xyaxes(xyaxes) -> np.ndarray:
    """MuJoCo camera xyaxes (x right, y UP, looks along -z) -> the FLU->optical
    rotation, columns = OpenCV camera axes expressed in FLU body:
    x_cv = cam_x, y_cv = -cam_y (optical y is DOWN), z_cv = x_cv x y_cv."""
    x = np.asarray(xyaxes[:3], float)
    y = np.asarray(xyaxes[3:6], float)
    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)
    x_cv = x
    y_cv = -y
    z_cv = np.cross(x_cv, y_cv)
    return np.column_stack([x_cv, y_cv, z_cv])


def geofence_contains(box: dict, p_ned, margin: float = 0.0) -> bool:
    x, y, z = (float(v) for v in p_ned)
    return (box["x"][0] - margin <= x <= box["x"][1] + margin
            and box["y"][0] - margin <= y <= box["y"][1] + margin
            and box["z"][0] - margin <= z <= box["z"][1] + margin)


def geofence_clamp(box: dict, p_ned) -> np.ndarray:
    p = np.asarray(p_ned, float).copy()
    for i, ax in enumerate(("x", "y", "z")):
        p[i] = min(max(p[i], box[ax][0]), box[ax][1])
    return p


@dataclass
class NavConfig:
    """config/hw_nav.yaml, resolved. See the file itself for field-by-field
    provenance notes; unknowns there are tagged [예측]/[스펙] per CLAUDE.md."""

    geometry: str = "wall"                    # "wall" | "floor"
    detector: str = "auto"
    tag_family: str = "tag36h11"
    tag_size_m: float = 0.170                 # config/config.yaml tags.tag_size_m
    tag_map_path: str = "config/tag_map.yaml"
    wall_tag_id: int = 25
    wall_tag_size_m: float | None = None      # None = tag_size_m
    wall_preset: str = "x_into_wall"
    min_tags: int = 1
    max_reproj_px: float = 3.0
    stale_s: float = 0.5
    # Ids that exist TWICE on the physical mat. They never anchor a solve;
    # each detection is kept only if it reprojects within dup_confirm_px of
    # the map pose, using the pose the UNIQUE tags established. See
    # tagnav.py's module docstring for why that is enough.
    duplicate_ids: tuple = ()
    dup_confirm_px: float = 6.0
    # Single-tag gravity gate: reject a fix whose implied roll/pitch differs
    # from ATTITUDE by more than this. Assumes the tag hangs VERTICAL and the
    # extrinsic (incl. camera tilt) is right — a leaning paper tag or an
    # un-levelled tilt mount eats every frame at the default 10.
    tilt_gate_deg: float = 10.0
    ambiguity_ratio: float = 1.5
    # AprilTag quad detection decimation: a float for every feed, or a dict
    # {panel: float}. 1.0 = full resolution; 2.0 samples a quarter of the
    # pixels for the quad SEARCH while corners are still refined at full
    # resolution. Rule of thumb: 1.0 for the 640x360 C3 stream (decimating a
    # small image loses far tags for no win), 2.0 for the 720p+ ROV RGB
    # (camera-rate detection).
    quad_decimate: object = None

    def decimate_for(self, panel: str) -> float:
        d = self.quad_decimate
        if isinstance(d, dict):
            return float(d.get(panel, d.get("default", 2.0)))
        if d is None:
            return 1.0 if panel == "main" else 2.0
        return float(d)
    z_source: str = "pressure"                # "pressure" | "tag"
    nav_stream: str = "color"                 # which C3 stream feeds detection
    # WHICH video feed localizes: "main" = the C3 colour stream (factory
    # underwater intrinsics ride every frame) or "second" = the ROV's own RGB
    # (needs the [예측] second_cam block below — there is no factory
    # calibration for that camera, so every number in it is an estimate until
    # someone calibrates it).
    nav_source: str = "main"
    # World datum: "map" = the tag/anchor frame as-is; "first_fix" = the
    # FIRST successful fix defines (0,0,0) and yaw 0 — positions, the square
    # and the geofence are all relative to where the run began. Re-zeroed by
    # toggling the localizing feed's TAG button off and on.
    datum: str = "map"
    cam_t_flu: tuple = _C3_T_FLU
    cam_xyaxes_flu: tuple = _C3_XYAXES
    # The ROV default-RGB camera model, used ONLY when nav_source: second.
    # [예측] defaults: tilt mount assumed LEVEL (mount_center), position a
    # rough tape estimate, fx from the vendor's in-air FOV — a wrong fx scales
    # every distance proportionally, so a 20 cm square stays square but not
    # exactly 20 cm. Calibrate before quoting numbers from this source.
    second_cam: dict = field(default_factory=lambda: {
        "t_flu": [0.18, 0.0, 0.0],
        "xyaxes_flu": [0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
        # The tilt-mount angle the extrinsic assumes, degrees DOWN from level.
        # The mount is open-loop (no angle feedback), so this must match what
        # the operator SET: press LEVEL for 0, or drive to the stop and enter
        # that angle here. A wrong tilt rotates the whole world estimate.
        "tilt_deg": 0.0,
        "fx": 1144.0, "fy": 1144.0, "cx": 960.0, "cy": 540.0,
        "width": 1920, "height": 1080, "dist": []})
    R_ned_map: np.ndarray = field(default_factory=lambda: np.eye(3))
    geofence: dict = field(default_factory=lambda: {
        "x": [-4.0, -0.3], "y": [-1.5, 1.5], "z": [0.2, 2.0], "margin": 0.4})
    pool_draw: tuple = (4.877, 1.8)           # display only
    # Pool boundary rectangle in the MAP/NED frame, {"x": [x0, x1],
    # "y": [y0, y1]} — DISPLAY ONLY (plot border + axis scale). None = the
    # plot falls back to fitting the geofence. Placement provenance lives in
    # hw_nav.yaml next to the numbers.
    pool_ned: dict | None = None
    # ...or DERIVE it: the outermost tag EDGES plus this margin, on every
    # side. Preferred over a hand-typed box because it follows the map — a
    # rebuilt or extended map moves the wall with it instead of leaving a
    # stale rectangle behind. See pool_from_tags().
    pool_margin_m: float | None = None
    raw: dict = field(default_factory=dict)

    # ------------------------------------------------------------ derived
    def R_t_frd_cam(self, source: str = "main") -> tuple[np.ndarray, np.ndarray]:
        """The extrinsic the PnP chain consumes: x_bodyFRD = R x_cam + t.
        ``source`` picks the camera: "main" = C3 (measured registration),
        "second" = the ROV RGB ([예측] block)."""
        if source == "second":
            sc = self.second_cam
            R_flu_cam = R_flu_cam_from_xyaxes(sc["xyaxes_flu"])
            # Tilt-mount pitch: rotate the LEVEL camera axes down by tilt_deg
            # about body +Y (FLU). R_y(+th) maps +X -> (cos, 0, -sin): the
            # optical axis dips toward the floor, which is what "tilt down"
            # means on this mount.
            th = math.radians(float(sc.get("tilt_deg", 0.0) or 0.0))
            if abs(th) > 1e-9:
                c, s = math.cos(th), math.sin(th)
                R_tilt = np.array([[c, 0.0, s],
                                   [0.0, 1.0, 0.0],
                                   [-s, 0.0, c]])
                R_flu_cam = R_tilt @ R_flu_cam
            return (S_FLU_FRD @ R_flu_cam,
                    S_FLU_FRD @ np.asarray(sc["t_flu"], float))
        R_flu_cam = R_flu_cam_from_xyaxes(self.cam_xyaxes_flu)
        return S_FLU_FRD @ R_flu_cam, S_FLU_FRD @ np.asarray(self.cam_t_flu, float)

    def second_K(self, w: int, h: int) -> tuple[np.ndarray, np.ndarray | None]:
        """The [예측] second-camera intrinsics, rescaled to the frame size
        actually received (the RTP stream can arrive at any negotiated size)."""
        sc = self.second_cam
        sx = float(w) / float(sc.get("width", w) or w)
        sy = float(h) / float(sc.get("height", h) or h)
        K = np.array([[float(sc["fx"]) * sx, 0.0, float(sc["cx"]) * sx],
                      [0.0, float(sc["fy"]) * sy, float(sc["cy"]) * sy],
                      [0.0, 0.0, 1.0]])
        dist = np.asarray(sc.get("dist", []) or [], float)
        return K, (dist if dist.size else None)

    def pool_from_tags(self, tag_map) -> dict | None:
        """The pool box implied by the map: outermost tag EDGES + the margin.

        Tag poses are CENTRES, so half a tag is added before the margin — the
        operator measures to the printed edge, not to an invisible centre.
        Returns None unless ``pool_margin_m`` is set; an explicit
        ``pool_ned`` in the YAML wins over this (the caller decides).
        """
        if self.pool_margin_m is None or not getattr(tag_map, "instances", None):
            return None
        P = np.array([t for poses in tag_map.instances.values()
                      for (_R, t) in poses], float)
        if P.size == 0:
            return None
        pad = self.effective_tag_size() / 2.0 + float(self.pool_margin_m)
        return {"x": [float(P[:, 0].min() - pad), float(P[:, 0].max() + pad)],
                "y": [float(P[:, 1].min() - pad), float(P[:, 1].max() + pad)]}

    def make_tag_map(self):
        from .tagnav import TagMap
        if self.geometry == "wall":
            m = TagMap.single(self.wall_tag_id)
        else:
            m = TagMap.load(self.tag_map_path)
        return m

    def effective_tag_size(self) -> float:
        if self.geometry == "wall" and self.wall_tag_size_m:
            return float(self.wall_tag_size_m)
        return float(self.tag_size_m)

    @classmethod
    def load(cls, path, repo_root: Path | None = None,
             geometry_override: str | None = None) -> "NavConfig":
        import yaml

        root = Path(repo_root) if repo_root else Path(path).resolve().parents[1]
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if geometry_override:
            raw = dict(raw)
            raw["geometry"] = str(geometry_override)
        cfg = cls(raw=raw)
        for key in ("geometry", "detector", "tag_family", "wall_preset",
                    "z_source", "nav_stream", "nav_source", "datum"):
            if key in raw:
                setattr(cfg, key, str(raw[key]))
        if cfg.nav_source not in ("main", "second"):
            raise ValueError(f"nav_source must be main|second, got "
                             f"{cfg.nav_source!r}")
        if cfg.datum not in ("map", "first_fix"):
            raise ValueError(f"datum must be map|first_fix, got {cfg.datum!r}")
        if "second_cam" in raw and raw["second_cam"]:
            cfg.second_cam = {**cfg.second_cam, **raw["second_cam"]}
        for key in ("tag_size_m", "max_reproj_px", "stale_s", "tilt_gate_deg",
                    "ambiguity_ratio", "dup_confirm_px"):
            if key in raw:
                setattr(cfg, key, float(raw[key]))
        if raw.get("duplicate_ids"):
            cfg.duplicate_ids = tuple(sorted(int(v) for v in
                                             raw["duplicate_ids"]))
        if "quad_decimate" in raw:
            qd = raw["quad_decimate"]
            cfg.quad_decimate = (dict(qd) if isinstance(qd, dict)
                                 else float(qd))
        if raw.get("wall_tag_size_m") is not None:
            cfg.wall_tag_size_m = float(raw["wall_tag_size_m"])
        for key in ("wall_tag_id", "min_tags"):
            if key in raw:
                setattr(cfg, key, int(raw[key]))
        if "tag_map" in raw:
            p = Path(raw["tag_map"])
            cfg.tag_map_path = str(p if p.is_absolute() else root / p)
        if "pool_draw_flu" in raw:
            cfg.pool_draw = tuple(float(v) for v in raw["pool_draw_flu"])
        if raw.get("pool_margin_m") is not None:
            cfg.pool_margin_m = float(raw["pool_margin_m"])
        if "pool_ned" in raw and raw["pool_ned"]:
            g = raw["pool_ned"]
            cfg.pool_ned = {"x": [float(g["x"][0]), float(g["x"][1])],
                            "y": [float(g["y"][0]), float(g["y"][1])]}
        if "geofence_ned" in raw:
            g = raw["geofence_ned"]
            cfg.geofence = {
                "x": [float(g["x"][0]), float(g["x"][1])],
                "y": [float(g["y"][0]), float(g["y"][1])],
                "z": [float(g["z"][0]), float(g["z"][1])],
                "margin": float(g.get("margin", 0.4)),
            }

        # ---- camera extrinsic: measured JSON first, YAML override wins
        js = raw.get("cam_extrinsic_json",
                     "bluerov2_mujoco_marinegym/meshes/c3_payload_frames.json")
        jp = Path(js)
        jp = jp if jp.is_absolute() else root / jp
        if jp.exists():
            try:
                d = json.loads(jp.read_text())
                cfg.cam_t_flu = tuple(float(v) for v in d["cam_center_bl"])
                cfg.cam_xyaxes_flu = tuple(float(v) for v in d["cam_xyaxes"])
            except (KeyError, ValueError, json.JSONDecodeError):
                pass                            # fall back to the frozen copy
        if raw.get("cam_t_flu") is not None:
            cfg.cam_t_flu = tuple(float(v) for v in raw["cam_t_flu"])
        if raw.get("cam_xyaxes_flu") is not None:
            cfg.cam_xyaxes_flu = tuple(float(v) for v in raw["cam_xyaxes_flu"])

        # ---- map -> NED remap
        if raw.get("R_ned_map") is not None:
            cfg.R_ned_map = np.asarray(raw["R_ned_map"], float).reshape(3, 3)
        elif cfg.geometry == "wall":
            try:
                cfg.R_ned_map = WALL_PRESETS[cfg.wall_preset].copy()
            except KeyError:
                raise ValueError(f"unknown wall_preset {cfg.wall_preset!r}; "
                                 f"pick one of {sorted(WALL_PRESETS)}")
        else:
            cfg.R_ned_map = np.eye(3)          # floor map is already NED-like
        d = float(np.linalg.det(cfg.R_ned_map))
        if abs(d - 1.0) > 1e-6:
            raise ValueError(f"R_ned_map must be a proper rotation (det={d:.4f})")
        return cfg


@dataclass
class MpcConfig:
    """config/hw_mpc.yaml, resolved. Axis gains and EAOB sigmas are [예측]
    until the P4 step-calibration / P3 residual fit replace them."""

    rov_model: str = "heavy_gripper"
    mode: str = "dobmpc"                      # "mpc" | "dobmpc" | "pid"
    ctrl_hz: float = 20.0
    # PID overrides (rov_gui/control/pid.py SIM_GAINS is the base) — most
    # importantly omega_derate, the hardware detune of the sim pole placement.
    pid: dict = field(default_factory=dict)
    # Bridge tag fixes with the velocity estimate + gyro between camera
    # frames (state_assembler): the position the EAOB/PID sees (and the plot
    # draws) moves at the control rate instead of holding the last fix.
    vel_propagation: bool = True
    axis_gain: dict = field(default_factory=lambda: {
        "surge_n": 60.0, "sway_n": 60.0, "heave_n": 60.0, "yaw_nm": 20.0})
    axis_cap: float = 0.5
    w_hat_clip: tuple = (15.0, 45.0, 45.0, 5.0, 5.0, 8.0)
    nudot_source: str = "imu"                 # "imu" | "fd"
    vel_lp_alpha: float = 0.35
    eaob_sigmas: dict = field(default_factory=dict)
    square: dict = field(default_factory=lambda: {
        "size": 1.0, "speed": 0.12, "laps": 3, "depth_ned": None,
        "heading_follow": False, "yaw_rate_deg_s": 60.0,
        "origin": "current", "rot_deg": 0.0, "yaw_fixed_deg": "current"})
    engage: dict = field(default_factory=lambda: {
        "require_mode": "MANUAL", "probe_ms_max": 25.0, "tag_stale_s": 0.5,
        "imu_stale_s": 0.3, "warmup_s": 1.5, "start_err_max_m": 0.3,
        "max_solver_fails": 3, "tick_overrun_ms": 100.0})
    log_dir: str = "sessions/mpc_runs"
    raw: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path) -> "MpcConfig":
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        cfg = cls(raw=raw)
        for key in ("rov_model", "mode", "nudot_source", "log_dir"):
            if key in raw:
                setattr(cfg, key, str(raw[key]))
        for key in ("ctrl_hz", "axis_cap", "vel_lp_alpha"):
            if key in raw:
                setattr(cfg, key, float(raw[key]))
        if "axis_gain" in raw:
            cfg.axis_gain.update({k: float(v) for k, v in raw["axis_gain"].items()})
        if "w_hat_clip" in raw:
            cfg.w_hat_clip = tuple(float(v) for v in raw["w_hat_clip"])
        if "eaob_sigmas" in raw and raw["eaob_sigmas"]:
            cfg.eaob_sigmas = dict(raw["eaob_sigmas"])
        if "square" in raw and raw["square"]:
            cfg.square.update(raw["square"])
        if "engage" in raw and raw["engage"]:
            cfg.engage.update(raw["engage"])
        if "pid" in raw and raw["pid"]:
            cfg.pid = dict(raw["pid"])
        if "vel_propagation" in raw:
            cfg.vel_propagation = bool(raw["vel_propagation"])
        if cfg.mode not in ("mpc", "dobmpc", "pid"):
            raise ValueError(f"mode must be mpc|dobmpc|pid, got {cfg.mode!r}")
        return cfg


def yaw_from_R(R: np.ndarray) -> float:
    """ZYX psi, the same extraction dobmpc.frames._euler_from_R uses."""
    return math.atan2(float(R[1, 0]), float(R[0, 0]))
