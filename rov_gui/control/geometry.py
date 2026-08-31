#!/usr/bin/env python3
"""geometry.py — frames, extrinsics and the two config files.

Everything that turns "a tag in a camera image" into "the vehicle in a NED
world box" is decided HERE, from YAML, and nowhere else:

* which tag map defines the world (floor survey vs a single wall tag);
* the map->NED remap ``R_ned_map`` (presets below, overridable);
* the camera extrinsic ``T_body_cam`` (seeded from the sim's measured
  ``c3_payload_frames.json``; a re-mount writes its own numbers into YAML).

There used to be a geofence here too — a NED box that START TRAJ checked the
placed path against and that disengaged the controller at runtime. It was
removed on 2026-08-14 at the operator's request; ``geofence_ned`` /
``geofence_frame`` keys left in an old YAML are now IGNORED, not an error, so
a stale config still loads (see :meth:`NavConfig.load`).

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

#: Every mission ``square.shape`` may name. The panel offers exactly these and
#: :meth:`MpcConfig.load` refuses anything else — see the note there.
#: ``replay`` re-flies a recorded handheld demonstration (poses extracted by
#: ``umi_handheld.extract_pose``) through the plan-stream seam — the same
#: install path a live policy will use later (control/plan_stream.py).
SHAPES = ("station", "line", "square", "circle", "follow", "replay")

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


def _tilt_flu(R_flu_cam: np.ndarray, tilt_deg: float) -> np.ndarray:
    """Rotate a LEVEL camera's axes DOWN by ``tilt_deg`` about body +Y (FLU).

    ``R_y(+th)`` maps +X -> (cos, 0, -sin): the optical axis dips toward the
    floor, which is what "tilt down" means on a mount. Shared by BOTH cameras
    — the C3 was re-pointed at the floor map for the 2026-08 mission and the
    ROV RGB rides an open-loop tilt servo, so neither extrinsic is level by
    assumption any more.

    A wrong angle rotates the whole world estimate, and it does NOT show up in
    reprojection error: solvePnP recovers the CAMERA pose, and the extrinsic
    only maps camera -> body afterwards. What it moves is (a) the reported body
    ATTITUDE by the full angle, and (b) the position by the camera-to-body
    lever arm only. The check is ``StateAssembler.rp_residual_deg`` (tag-implied
    roll/pitch vs ATTITUDE), which is reported on MpcStatus and in the CSV.
    """
    th = math.radians(float(tilt_deg or 0.0))
    if abs(th) <= 1e-9:
        return R_flu_cam
    c, s = math.cos(th), math.sin(th)
    R_tilt = np.array([[c, 0.0, s],
                       [0.0, 1.0, 0.0],
                       [-s, 0.0, c]])
    return R_tilt @ R_flu_cam


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
    # FIRST successful fix defines (0,0,0) and yaw 0 — positions and the
    # square are relative to where the run began. Re-zeroed by
    # toggling the localizing feed's TAG button off and on.
    datum: str = "map"
    cam_t_flu: tuple = _C3_T_FLU
    cam_xyaxes_flu: tuple = _C3_XYAXES
    # Tilt-mount angle for the MAIN (C3) camera, degrees DOWN from the axes
    # above. The measured Onshape registration in cam_xyaxes_flu is the
    # FORWARD-LEVEL mount (its implied pitch is 0.32 deg up); when the C3 is
    # re-pointed at the floor map, this is where that angle goes.
    #
    # The default MUST stay 0.0: it is what every recorded run was flown with,
    # and a non-zero default would retroactively reinterpret all of them.
    # ``_run_meta["hardware"]["cam_tilt_deg"]`` is the record boundary — do not
    # pool positions or yaw across runs whose value differs.
    #
    # NOTE this is applied as a PURE ROTATION about the camera's own origin, so
    # cam_t_flu is assumed unchanged by the re-mount. A hinge moves the lens
    # too; measure it and set cam_t_flu in hw_nav.yaml if that matters at the
    # centimetre level (the lever arm here is ~0.29 m).
    cam_tilt_deg: float = 0.0
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
    pool_draw: tuple = (4.877, 1.8)           # display only
    # Pool boundary rectangle in the MAP/NED frame, {"x": [x0, x1],
    # "y": [y0, y1]} — DISPLAY ONLY (plot border + axis scale). None = the
    # plot falls back to a 4 m box. Placement provenance lives in
    # hw_nav.yaml next to the numbers.
    pool_ned: dict | None = None
    # ...or DERIVE it: the outermost tag EDGES plus this margin, on every
    # side. Preferred over a hand-typed box because it follows the map — a
    # rebuilt or extended map moves the wall with it instead of leaving a
    # stale rectangle behind. See pool_from_tags().
    pool_margin_m: float | None = None
    # The vehicle's footprint (ALONG-heading, ACROSS), metres — display only,
    # drawn around the position on the trajectory plot.
    rov_footprint_m: tuple = (0.4318, 0.5334)
    raw: dict = field(default_factory=dict)

    # ------------------------------------------------------------ derived
    def R_t_frd_cam(self, source: str = "main") -> tuple[np.ndarray, np.ndarray]:
        """The extrinsic the PnP chain consumes: x_bodyFRD = R x_cam + t.
        ``source`` picks the camera: "main" = C3 (measured registration),
        "second" = the ROV RGB ([예측] block)."""
        if source == "second":
            sc = self.second_cam
            R_flu_cam = _tilt_flu(R_flu_cam_from_xyaxes(sc["xyaxes_flu"]),
                                  sc.get("tilt_deg", 0.0))
            return (S_FLU_FRD @ R_flu_cam,
                    S_FLU_FRD @ np.asarray(sc["t_flu"], float))
        R_flu_cam = _tilt_flu(R_flu_cam_from_xyaxes(self.cam_xyaxes_flu),
                              self.cam_tilt_deg)
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
                    "ambiguity_ratio", "dup_confirm_px", "cam_tilt_deg"):
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
        if raw.get("rov_footprint_m"):
            cfg.rov_footprint_m = tuple(float(v)
                                        for v in raw["rov_footprint_m"][:2])
        if raw.get("pool_margin_m") is not None:
            cfg.pool_margin_m = float(raw["pool_margin_m"])
        if "pool_ned" in raw and raw["pool_ned"]:
            g = raw["pool_ned"]
            cfg.pool_ned = {"x": [float(g["x"][0]), float(g["x"][1])],
                            "y": [float(g["y"][0]), float(g["y"][1])]}
        # geofence_ned / geofence_frame: DELIBERATELY not read. The fence
        # was removed on 2026-08-14; silently ignoring the keys means an old
        # hw_nav.yaml still loads rather than crashing the station at startup.
        # (They stay readable in cfg.raw for anyone auditing a past run.)

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
    # Rate limit on the axis command [1/s]; 0 = none (pre-2026-08-17).
    axis_slew_per_s: float = 0.0
    # PATH FOLLOWING (default) vs trajectory tracking. The worker projects the
    # measured vehicle onto the active segment and builds one shared spatial
    # plan for the PID. False = the legacy wall-clock reference.
    #
    # (The corner GATE this block used to configure — path_corner_tol_m /
    # _speed_m_s / _dwell_s — was removed 2026-08-16 at the operator's request.
    # It brought the vehicle to a full stop inside 5 cm of every vertex and
    # hid the next leg until it did, which is precisely why an MPC horizon
    # could never see round a corner. The mission is now a C1 curve
    # (path_geometry) that is simply flown.)
    path_following: bool = True
    path_lead_m: float = 0.15
    # MPCC path shaping. fillet rounds rectangle corners; turn_radius
    # optionally rounds a line's turnarounds into a stadium (0 = reverse in
    # place, keeping the mission's exact geometry). The two accelerations set
    # the feasible speed profile: v <= sqrt(a_lat * R) through a fillet, and
    # a_long limits how fast the reference may speed up or slow down.
    path_fillet_m: float = 0.06
    path_turn_radius_m: float = 0.0
    path_lat_accel_m_s2: float = 0.30
    path_long_accel_m_s2: float = 0.15
    path_creep_m_s: float = 0.02
    traj_timeout_factor: float = 3.0
    w_hat_clip: tuple = (15.0, 45.0, 45.0, 5.0, 5.0, 8.0)
    nudot_source: str = "imu"                 # "imu" | "fd"
    # 0.6 since 2026-08-14 — see hw_mpc.yaml for the measurement that moved it
    # off 0.35. Kept equal to the shipped config so a hand-built MpcConfig()
    # (a test, an embedder) filters the same way the station does.
    vel_lp_alpha: float = 0.6
    eaob_sigmas: dict = field(default_factory=dict)
    # ALONG/CROSS COST SPLIT for the ``mpc_tuned`` / ``dobmpc_tuned`` modes.
    # Empty = path_cost.DEFAULT_TUNE. Only these modes read it; ``mpc`` and
    # ``dobmpc`` fly the isotropic Q whatever is written here, so the block
    # can sit in the file while a baseline run is flown.
    mpc_tuned: dict = field(default_factory=dict)
    # STATION BRIDGE — carry a station hold across a tag dropout instead of
    # disengaging. Empty = station_bridge.DEFAULTS. STATION MODE ONLY.
    station_bridge: dict = field(default_factory=dict)
    # OBJECT FOLLOW — place the --pose tracked object in the tag-map frame and
    # (shape: follow) hold a captured relative pose on it. Empty =
    # object_nav.DEFAULTS. Needs --pose AND --mpc; see control/object_nav.py.
    object_nav: dict = field(default_factory=dict)
    # The mission. ``shape`` picks the path: "square" (a rectangle — size is
    # the x side, size_y the y side), "line" (out-and-back along dir_deg, one
    # lap = there AND back), or "circle" (radius, one lap = one revolution).
    # ``origin_tag`` anchors the path to a physical tag id, which is how the
    # operator names a place; without it the path starts wherever the vehicle
    # is at START. For every shape the tag is a point the path PASSES THROUGH,
    # never a centre it merely orbits: the rectangle's min-x/min-y corner, the
    # line's near end, the circle's minimum-x rim point. "follow" is the one
    # shape with no geometry at all — it holds a relative pose on the tracked
    # object and ignores origin_tag/size/speed entirely.
    square: dict = field(default_factory=lambda: {
        "shape": "square",           # station|line|square|circle|follow
        "size": 1.0, "size_y": None, "speed": 0.12, "laps": 3,
        "length": 2.0, "dir_deg": 90.0, "ramp_s": 1.0,
        "radius": 0.5,               # CIRCLE only; the tag is on the rim
        "depth_ned": None,
        "heading_follow": False, "yaw_rate_deg_s": 60.0,
        "origin": "current", "origin_tag": None, "rot_deg": 0.0,
        # Heading held during the mission. "current" = whatever the vehicle
        # had at START. A NUMBER in yaw_map_deg is an ABSOLUTE heading in the
        # tag-map frame (90 = facing +y), which is what "sit on the tag facing
        # +y" means and what a datum-relative angle cannot express.
        "yaw_fixed_deg": "current", "yaw_map_deg": None})
    # The approach/settle keys are HERE and not only in the YAML for the
    # vel_lp_alpha reason: a hand-built MpcConfig() (a test, an embedder) must
    # fly the same numbers the station does. approach_speed_m_s and
    # approach_lead_m in particular only mean anything TOGETHER — the leash is
    # what actually set the speed until 2026-08-18 (see hw_mpc.yaml).
    engage: dict = field(default_factory=lambda: {
        "require_mode": "MANUAL", "probe_ms_max": 25.0, "tag_stale_s": 0.5,
        "tag_stale_hold_s": 0.4,
        "imu_stale_s": 0.3, "warmup_s": 1.5, "start_err_max_m": 0.3,
        "settle_s": 10.0, "settle_station_s": 0.0,
        "approach_speed_m_s": 0.20, "approach_lead_m": 0.50,
        "approach_max_s": 180.0,
        # FOLLOW's feedforward ceiling — the vehicle's own top speed, so an
        # object estimate that claims to be moving faster cannot relocate the
        # far end of the MPC horizon (hw_mpc.yaml carries the derivation).
        "follow_ff_max_m_s": 0.30,
        "max_solver_fails": 3, "tick_overrun_ms": 100.0})
    # DEMO REPLAY (shape: replay) — re-fly one recorded handheld demonstration.
    # The session dir must hold poses.npy/poses.json (umi_handheld.extract_pose
    # output). The track is re-expressed relative to its own first pose and
    # anchored at the vehicle's pose when the mission arms, so a bench demo
    # replays wherever the vehicle happens to be — no tag id enters it.
    #
    # stream_period_s = 0 installs the whole track as ONE plan (M0a: the
    # minimal label→flight check); > 0 chops it into `horizon_s` windows
    # released every period through the PlanFilter/PlanStitcher (M0b: the
    # exact seam a live diffusion policy will feed later).
    #
    # gripper: OFF by default — the jaw path is open-loop (no position
    # feedback exists on the real vehicle) and early replays should prove the
    # trajectory before the jaw moves at all. The thresholds read the demo's
    # gripper_width channel (0 = closed): width below close_below drives
    # CLOSE, above open_above drives OPEN, in between holds; hold_max_s is the
    # anti-stall auto-neutral (UMI-U used 4 s).
    replay: dict = field(default_factory=lambda: {
        "session": "",                # sessions/demonstration_NNNN (poses.npy)
        "v_max_m_s": 0.12,            # [예측] time-dilation ceiling; replace
                                      # with the measured achievable speed
        "blend_s": 0.4,
        "stream_period_s": 0.0,       # 0 = one-shot plan; >0 = 1 Hz-style feed
        "horizon_s": 4.0,
        "anchor_max_m": 0.30,         # [예측] plan-vs-reference gates — see
        "jump_max_m": 0.20,           #   control/plan_stream.FilterLimits
        # The ONLY position-based protection left after the geofence removal
        # (2026-08-14): a streamed plan with any knot outside this datum-NED
        # box is REJECTED. None disables the check.
        "workspace_box_ned": None,    # [[xmin,ymin,zmin],[xmax,ymax,zmax]]
        "gripper": False,
        "gripper_close_below": 0.35,
        "gripper_open_above": 0.65,
        "gripper_hold_max_s": 4.0,
    })
    # IMU DEAD RECKONING — the "how far can the IMU carry us" experiment.
    # OFF by default, and off means the tick is byte-identical to a build
    # without it (pinned by test_control's shadow-parity test).
    #
    #   mode: shadow   the controller keeps flying on the tag state; the DR
    #                  integrates beside it and is only published/recorded.
    #         control  the controller consumes the DR state instead. The tag
    #                  stays alive as ground truth AND as the operator's
    #                  instrument. Needs --imu-dr-control as well: nothing
    #                  should enter a closed loop on dead reckoning because a
    #                  config file was edited.
    imu_dr: dict = field(default_factory=lambda: {
        "enabled": False,
        "mode": "shadow",              # "shadow" | "control"
        "source": "c3",                # the C3 BNO086 (camera-rigid)
        "attitude": "ahrs",            # "gyro" | "ahrs" | "vehicle"
        "ahrs_tau_s": 10.0,
        "accel_trust_m_s2": 0.15,
        "z_source": "pressure",        # "pressure" | "imu"
        "static_window_s": 8.0,
        "gyro_static_std_max": 0.02,
        "gyro_bias_sem_max": 0.005,
        "accel_static_sd_max": 0.15,
        "calibration": "config/c3_imu_calib.json",
        "max_dt_s": 0.1,
        # Automatic aborts, DISABLED by operator decision (2026-08-16): the
        # pilot stops the run by hand. The keys exist so turning one on later
        # is a config edit and not a code change. None = no limit.
        "abort_err_m": None,
        "abort_max_s": None,
        "axis_cap_dr": None,           # None = use axis_cap
    })
    # ROOT of the dated run tree (rov_gui/runstore.py), not the folder
    # written to: the CSV, its .meta.json and events.log land in
    # <log_dir>/YYYYMMDD/MMDD_HHMMSS/ beside the recordings of the same run.
    # The DEFAULT is load-bearing — a config file with no log_dir: key, or
    # a hand-built MpcConfig(), must not fall back into the old flat tree.
    log_dir: str = "sessions/low_level_controller_data"
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
        if "path_following" in raw:
            cfg.path_following = bool(raw["path_following"])
        for key in ("ctrl_hz", "axis_cap", "axis_slew_per_s",
                    "vel_lp_alpha", "path_lead_m",
                    "path_fillet_m", "path_turn_radius_m",
                    "path_lat_accel_m_s2", "path_long_accel_m_s2",
                    "path_creep_m_s", "traj_timeout_factor"):
            if key in raw:
                setattr(cfg, key, float(raw[key]))
        if "axis_gain" in raw:
            cfg.axis_gain.update({k: float(v) for k, v in raw["axis_gain"].items()})
        if "w_hat_clip" in raw:
            cfg.w_hat_clip = tuple(float(v) for v in raw["w_hat_clip"])
        if "eaob_sigmas" in raw and raw["eaob_sigmas"]:
            cfg.eaob_sigmas = dict(raw["eaob_sigmas"])
        if "object_nav" in raw and raw["object_nav"]:
            # Unknown keys RAISE, same rule as station_bridge and imu_dr.
            from .object_nav import resolve as _resolve_object

            _resolve_object(raw["object_nav"])
            cfg.object_nav = dict(raw["object_nav"])
        if "station_bridge" in raw and raw["station_bridge"]:
            # Unknown keys RAISE, same rule as imu_dr and mpc_tuned: a
            # silently ignored key here means the operator flies believing a
            # safety ladder is armed when it is not.
            from .station_bridge import resolve as _resolve_bridge

            _resolve_bridge(raw["station_bridge"])
            cfg.station_bridge = dict(raw["station_bridge"])
        if "mpc_tuned" in raw and raw["mpc_tuned"]:
            # Unknown keys and non-positive weights RAISE here, for the imu_dr
            # reason: a silently ignored `cross_sale:` means the run flies the
            # BASELINE cost while its meta says "tuned", and the two CSVs are
            # then indistinguishable from two runs of the same controller.
            from .path_cost import resolve_tune

            resolve_tune(raw["mpc_tuned"])
            cfg.mpc_tuned = dict(raw["mpc_tuned"])
        if "square" in raw and raw["square"]:
            cfg.square.update(raw["square"])
        # WHICH SHAPE, validated — AFTER the merge, so the name checked is the
        # one that will fly. Until 2026-08-21 `_arm_path`'s else branch was
        # "square", so a typo ("staton", "squrae") silently flew a RECTANGLE:
        # a mission of a completely different size and duration from the one
        # asked for, with nothing on screen or in the meta saying the name had
        # not been understood.
        shape = str(cfg.square.get("shape", "square")).lower()
        if shape not in SHAPES:
            raise ValueError(f"square.shape must be one of {list(SHAPES)}, "
                             f"got {cfg.square.get('shape')!r}")
        cfg.square["shape"] = shape
        if "engage" in raw and raw["engage"]:
            # The ONE safety block that used to merge blind. object_nav,
            # station_bridge, imu_dr and mpc_tuned all reject unknown keys for
            # the same reason, and this block holds the interlocks:
            # tick_overrun_ms, max_solver_fails, tag_stale_s,
            # follow_ff_max_m_s. A typo here reads as armed and is not.
            unknown = set(raw["engage"]) - set(cfg.engage)
            if unknown:
                raise ValueError(
                    f"unknown engage keys {sorted(unknown)}; known: "
                    f"{sorted(cfg.engage)}")
            cfg.engage.update(raw["engage"])
            # ...and a limit that is zero, negative or NaN is not a limit.
            # `follow_ff_max_m_s: 0` is exactly what an operator would type to
            # mean "no feedforward"; it used to mean "uncapped", which is the
            # 2026-08-23 failure verbatim.
            for k in ("follow_ff_max_m_s", "tick_overrun_ms",
                      "approach_speed_m_s", "approach_lead_m"):
                v = float(cfg.engage[k])
                if not (math.isfinite(v) and v > 0.0):
                    raise ValueError(f"engage.{k} must be finite and > 0, "
                                     f"got {cfg.engage[k]!r}")
                cfg.engage[k] = v
            if int(cfg.engage["max_solver_fails"]) < 1:
                raise ValueError("engage.max_solver_fails must be >= 1 "
                                 "(0 disengages on the first tick)")
        if "pid" in raw and raw["pid"]:
            cfg.pid = dict(raw["pid"])
        if "vel_propagation" in raw:
            cfg.vel_propagation = bool(raw["vel_propagation"])
        if "imu_dr" in raw and raw["imu_dr"]:
            # Unknown keys RAISE here, unlike everywhere else in this file.
            # The failure mode is what earns the exception: `imu-dr:` for
            # `imu_dr:`, or `attitude_mode:` for `attitude:`, is silently
            # ignored — and the operator flies a run believing the estimator
            # is on and configured when it is off or on a default. A crash at
            # startup costs a minute; a wasted pool session costs an evening.
            unknown = set(raw["imu_dr"]) - set(cfg.imu_dr)
            if unknown:
                raise ValueError(
                    f"unknown imu_dr keys {sorted(unknown)}; "
                    f"known: {sorted(cfg.imu_dr)}")
            cfg.imu_dr.update(raw["imu_dr"])
        if "replay" in raw and raw["replay"]:
            # Unknown keys RAISE, the imu_dr rule: `griper:` for `gripper:` or
            # `vmax:` for `v_max_m_s:` silently ignored means a replay flown
            # believing a limit is armed when it is not.
            unknown = set(raw["replay"]) - set(cfg.replay)
            if unknown:
                raise ValueError(
                    f"unknown replay keys {sorted(unknown)}; known: "
                    f"{sorted(cfg.replay)}")
            cfg.replay.update(raw["replay"])
            for k in ("v_max_m_s", "blend_s", "horizon_s",
                      "gripper_hold_max_s", "anchor_max_m", "jump_max_m"):
                # anchor/jump included on purpose: `anchor_max_m: .nan` makes
                # every filter comparison False, i.e. the gate silently
                # PASSES — and the box below is the only position-based
                # protection left (safety review 2026-08-30).
                v = float(cfg.replay[k])
                if not (math.isfinite(v) and v > 0.0):
                    raise ValueError(f"replay.{k} must be finite and > 0, "
                                     f"got {cfg.replay[k]!r}")
                cfg.replay[k] = v
            box = cfg.replay.get("workspace_box_ned")
            if box is not None:
                b = np.asarray(box, float)
                if b.shape != (2, 3) or not np.all(np.isfinite(b)):
                    raise ValueError(
                        f"replay.workspace_box_ned must be "
                        f"[[xmin,ymin,zmin],[xmax,ymax,zmax]] (finite), "
                        f"got {box!r}")
                if not np.all(b[0] < b[1]):
                    raise ValueError(
                        f"replay.workspace_box_ned min must be < max on "
                        f"every axis, got {box!r}")
                cfg.replay["workspace_box_ned"] = [
                    [float(v) for v in b[0]], [float(v) for v in b[1]]]
            sp = float(cfg.replay["stream_period_s"])
            if not (math.isfinite(sp) and sp >= 0.0):
                raise ValueError(f"replay.stream_period_s must be finite and "
                                 f">= 0, got {cfg.replay['stream_period_s']!r}")
            lo = float(cfg.replay["gripper_close_below"])
            hi = float(cfg.replay["gripper_open_above"])
            if not (0.0 <= lo < hi <= 1.0):
                raise ValueError(
                    f"replay gripper thresholds must satisfy 0 <= close_below "
                    f"< open_above <= 1, got {lo!r} / {hi!r}")
        if cfg.imu_dr["mode"] not in ("shadow", "control"):
            raise ValueError(f"imu_dr.mode must be shadow|control, got "
                             f"{cfg.imu_dr['mode']!r}")
        if cfg.imu_dr["attitude"] not in ("gyro", "ahrs", "vehicle"):
            raise ValueError(f"imu_dr.attitude must be gyro|ahrs|vehicle, got "
                             f"{cfg.imu_dr['attitude']!r}")
        if cfg.imu_dr["z_source"] not in ("pressure", "imu"):
            raise ValueError(f"imu_dr.z_source must be pressure|imu, got "
                             f"{cfg.imu_dr['z_source']!r}")
        if cfg.mode not in ("mpc", "dobmpc", "mpc_tuned", "dobmpc_tuned",
                            "pid"):
            raise ValueError(
                f"mode must be mpc|dobmpc|mpc_tuned|dobmpc_tuned|pid, "
                f"got {cfg.mode!r}")
        return cfg


def yaw_from_R(R: np.ndarray) -> float:
    """ZYX psi, the same extraction dobmpc.frames._euler_from_R uses."""
    return math.atan2(float(R[1, 0]), float(R[0, 0]))
