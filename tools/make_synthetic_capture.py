#!/usr/bin/env python3
"""make_synthetic_capture.py — render a fake handheld-UMI capture from known geometry.

Why this exists
---------------
No camera is connected for this work, so this is the only executable path through the
recording pipeline. It renders a scene whose geometry is known exactly, writes it in the
byte-for-byte layout that a real capture uses, and drops a sidecar of ground truth beside
it. Everything downstream (record.py --source synthetic, extract_gripper_width.py, the
warp stage, build_zarr.py) can then be run end to end and CHECKED, not just executed:
the sidecar says what the answer should be.

What is rendered
----------------
A ray is cast per pixel through the camera model in configs/source_camera_air.yaml and
intersected with a small analytic scene:

  - a checkerboard plane standing off at `--board-z` (the calibration-style backdrop,
    and the source of GT corner pixel coordinates)
  - a few axis-aligned boxes at grasp distance
  - a two-finger gripper stand-in whose jaws open and close over the take, each finger
    carrying a real cv2.aruco DICT_4X4_50 marker (ids 0 and 1)

Ray casting rather than rasterising is deliberate: the depth it produces is the exact
metric range to the surface, so the sidecar's Z is not an approximation of the image.

What is faked, and how faithfully
---------------------------------
  depth quantisation   On by default. Real StereoDepth reports Z = fx*B/d with d on an
                       integer disparity grid, so the depth PNG is quantised the same
                       way. The sidecar keeps the exact pre-quantisation Z.
  MinZ cutoff          On by default. Anything closer than fx*B/max_disparity cannot be
                       triangulated and is written as 0 (invalid), which is what makes
                       the "depth valid %" overlay worth having.
  stereo pair          Generated already rectified: both cameras carry the same model
                       and differ by a pure baseline translation. That is the analogue
                       of StereoDepth's rectifiedLeft/rectifiedRight, not of a raw pair.
  colour               Rendered from the LEFT camera pose, not through a separate CAM_A
                       model. The affordance pipeline does not use the colour path; this
                       exists so the output layout matches a real capture.
  photometrics         Lambert shading, no noise, no vignetting, no motion blur, no
                       rolling shutter (the OV9282 is global shutter anyway).

NOT MEASURED. Nothing in this file is a measurement of any hardware. The camera model
comes from configs/source_camera_air.yaml, which is itself synthesised and carries
verified: false; scene dimensions are placeholders chosen to be plausible, and the
gripper dimensions in particular are placeholders pending real numbers.

Usage
-----
    python tools/make_synthetic_capture.py --dry-run
    python tools/make_synthetic_capture.py --out captures/synthetic_0001
    python tools/make_synthetic_capture.py --out ... --frames 90 --size 640x400
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))          # so `umi_handheld` imports from any cwd

# Matches c3_camera/dataset.py:58 — metres = pixel_value / DEPTH_SCALE.
DEPTH_SCALE = 1000.0

# Fixed so repeated runs are byte-identical. Real captures use wall-clock.
DEFAULT_EPOCH = 1785000000.0


# ------------------------------------------------------------------ camera model

class CameraModel:
    """OpenCV pinhole + rational distortion, with the ray direction for every pixel.

    Forward projection is cv2's. The inverse (pixel -> ray) is done by inverting the
    radial mapping on a monotone table rather than by cv2.undistortPoints, whose
    iterative solver is not dependable at the >120 deg field of view this model has.
    """

    def __init__(self, cfg: dict, path: Path):
        self.path = path
        self.name = cfg["name"]
        self.verified = bool(cfg.get("verified", False))
        self.medium = cfg.get("medium", "unknown")
        self.width, self.height = (int(v) for v in cfg["image_size"])
        k = cfg["intrinsics"]
        self.fx, self.fy = float(k["fx"]), float(k["fy"])
        self.cx, self.cy = float(k["cx"]), float(k["cy"])
        self.dist = np.array(cfg["distortion"], dtype=np.float64).reshape(1, -1)
        st = cfg.get("stereo", {}) or {}
        self.baseline_m = float(st.get("baseline_m", 0.075))
        self.raw = cfg

    @classmethod
    def load(cls, path: os.PathLike) -> "CameraModel":
        path = Path(path)
        with open(path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        if cfg.get("schema") != "umi_camera_model/1":
            raise ValueError(f"{path}: unsupported schema {cfg.get('schema')!r}")
        m = cls(cfg, path)
        if not m.verified:
            print(
                f"WARNING: camera model '{m.name}' is UNVERIFIED "
                f"({path.relative_to(REPO) if path.is_absolute() else path}).\n"
                f"         No hardware has been measured; the values are synthesised. "
                f"Geometry produced from this model is provisional.",
                file=sys.stderr,
            )
        return m

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], float)

    def _radial_table(self, n: int = 4096):
        """r_undistorted -> r_distorted, over the model's monotone domain."""
        if abs(self.dist.ravel()[2]) > 1e-6 or abs(self.dist.ravel()[3]) > 1e-6:
            raise ValueError(
                f"{self.path}: tangential distortion p1/p2 are non-zero; the radial "
                f"inverse used for ray generation assumes p1 = p2 = 0."
            )
        th = np.deg2rad(np.linspace(0.0, 89.0, n))
        r_u = np.tan(th)
        pts = np.stack([r_u, np.zeros_like(r_u), np.ones_like(r_u)], axis=1)[:, None, :]
        uv, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3),
                                  np.eye(3), self.dist)          # K = I -> normalised
        r_d = uv.reshape(-1, 2)[:, 0]
        good = np.isfinite(r_d)
        r_d, r_u = r_d[good], r_u[good]
        keep = np.concatenate([[True], np.diff(r_d) > 0])         # monotone prefix only
        stop = int(np.argmin(keep)) if not keep.all() else len(keep)
        return r_u[:stop], r_d[:stop]

    def pixel_rays(self, width=None, height=None):
        """Unit ray direction per pixel, camera frame (x right, y down, z forward)."""
        W = width or self.width
        H = height or self.height
        sx, sy = W / self.width, H / self.height
        fx, fy = self.fx * sx, self.fy * sy
        cx, cy = self.cx * sx, self.cy * sy

        u, v = np.meshgrid(np.arange(W, dtype=np.float64),
                           np.arange(H, dtype=np.float64))
        xd = (u - cx) / fx
        yd = (v - cy) / fy
        rd = np.hypot(xd, yd)

        r_u_tab, r_d_tab = self._radial_table()
        ru = np.interp(rd, r_d_tab, r_u_tab, left=0.0, right=np.nan)
        valid = np.isfinite(ru)                                   # outside the model
        scale = np.divide(ru, rd, out=np.ones_like(rd), where=rd > 1e-12)
        d = np.stack([xd * scale, yd * scale, np.ones_like(rd)], axis=-1)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        return d, valid

    def max_ray_angle_deg(self) -> float:
        r_u_tab, _ = self._radial_table()
        return float(np.rad2deg(np.arctan(r_u_tab[-1])))


# ---------------------------------------------------------------------- geometry

@dataclass
class Quad:
    """Planar patch: points are p0 + a*e1 + b*e2 with a, b in [0, 1]."""
    name: str
    p0: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    albedo: float = 0.7
    colour: tuple = (0.75, 0.75, 0.75)
    texture: object = None            # f(a, b) -> multiplier in [0, 1], same shape

    def intersect(self, o, d):
        n = np.cross(self.e1, self.e2)
        n = n / np.linalg.norm(n)
        denom = d @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((self.p0 - o) @ n) / denom
        t = np.where(np.abs(denom) < 1e-12, np.inf, t)
        t = np.where(t > 1e-6, t, np.inf)
        p = o + t[..., None] * d
        rel = p - self.p0
        a = (rel @ self.e1) / (self.e1 @ self.e1)
        b = (rel @ self.e2) / (self.e2 @ self.e2)
        inside = (a >= 0) & (a <= 1) & (b >= 0) & (b <= 1) & np.isfinite(t)
        t = np.where(inside, t, np.inf)
        shade = np.full(t.shape, self.albedo)
        if self.texture is not None:
            # a, b are NaN wherever the ray missed; clip alone does not remove NaN and
            # .astype(int) on NaN is undefined, so zero them before sampling.
            a = np.clip(np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0), 0, 1)
            b = np.clip(np.nan_to_num(b, nan=0.0, posinf=1.0, neginf=0.0), 0, 1)
            shade = shade * self.texture(a, b)
        nrm = np.broadcast_to(n, d.shape)
        return t, shade, nrm


@dataclass
class Box:
    """Axis-aligned box in world coordinates."""
    name: str
    lo: np.ndarray
    hi: np.ndarray
    albedo: float = 0.6
    colour: tuple = (0.8, 0.7, 0.55)

    def intersect(self, o, d):
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / d
            t0 = (self.lo - o) * inv
            t1 = (self.hi - o) * inv
        tmin = np.minimum(t0, t1).max(axis=-1)
        tmax = np.maximum(t0, t1).min(axis=-1)
        hit = (tmax >= np.maximum(tmin, 0.0)) & np.isfinite(tmin)
        t = np.where(hit & (tmin > 1e-6), tmin, np.inf)
        p = o + np.where(np.isfinite(t), t, 0.0)[..., None] * d
        centre = (self.lo + self.hi) / 2.0
        half = (self.hi - self.lo) / 2.0
        rel = (p - centre) / np.maximum(half, 1e-9)
        axis = np.argmax(np.abs(rel), axis=-1)
        nrm = np.zeros_like(d)
        np.put_along_axis(nrm, axis[..., None], np.sign(
            np.take_along_axis(rel, axis[..., None], axis=-1)), axis=-1)
        return t, np.full(t.shape, self.albedo), nrm


def checker_texture(squares_x: int, squares_y: int, dark=0.18, light=1.0):
    def tex(a, b):
        i = np.floor(a * squares_x).astype(int) + np.floor(b * squares_y).astype(int)
        return np.where(i % 2 == 0, light, dark)
    return tex


QUIET_RATIO = 0.25      # white margin each side, as a fraction of the marker's edge


def aruco_texture(marker_id: int, dict_name: str = "DICT_4X4_50", px: int = 240,
                  quiet_ratio: float = QUIET_RATIO):
    """Real ArUco bits on a white backing plate.

    The quiet zone is not decoration. detectMarkers segments the marker by finding its
    black border against a lighter surround; printed straight onto the dark finger body
    the border has almost no outer contrast and detection collapses (measured on the
    first render of this scene: both tags found in 9 of 72 frames). The plate is
    `1 + 2*quiet_ratio` times the marker edge, and the marker's own size — the number
    solvePnP needs — stays `tag_size_m`.
    """
    ad = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    marker = cv2.aruco.generateImageMarker(ad, marker_id, px)     # uint8, 0/255
    pad = int(round(px * quiet_ratio))
    plate = np.full((px + 2 * pad, px + 2 * pad), 255, np.uint8)
    plate[pad:pad + px, pad:pad + px] = marker
    lut = (plate.astype(np.float64) / 255.0) * 0.92 + 0.04        # keep off pure 0/1
    n = plate.shape[0]

    def tex(a, b):
        yi = np.clip((b * (n - 1)).astype(int), 0, n - 1)
        xi = np.clip((a * (n - 1)).astype(int), 0, n - 1)
        return lut[yi, xi]
    return tex


# ------------------------------------------------------------------------- scene

@dataclass
class SceneSpec:
    """All scene dimensions. PLACEHOLDERS — no physical rig has been measured."""
    # A 127 deg camera sees a very wide cone, so the scene is enclosed. Without walls
    # most rays escape and the depth map comes back mostly invalid, which would make
    # the depth-valid-fraction overlay meaningless during development.
    room_back_z: float = 1.30           # m, back wall
    room_floor_y: float = 0.45          # +y is down
    room_ceil_y: float = -0.70
    room_half_x: float = 1.10

    board_z: float = 0.62               # m, checkerboard standoff
    board_w: float = 0.90
    board_h: float = 0.56
    board_squares: tuple = (9, 6)
    board_square_m: float = 0.05

    box_specs: tuple = (
        # (name, centre xyz, half-extent xyz)
        ("target_can",   (-0.055, 0.045, 0.335), (0.033, 0.052, 0.033)),
        ("clutter_left", (-0.180, 0.062, 0.455), (0.045, 0.035, 0.045)),
        ("clutter_right", (0.155, 0.055, 0.400), (0.038, 0.045, 0.038)),
    )

    # Gripper stand-in. Every number here is a PLACEHOLDER pending the real rig:
    # jaw travel, tag size, tag face and lens-to-tag distance are all unmeasured.
    jaw_open_m: float = 0.085           # separation between inner faces, fully open
    jaw_closed_m: float = 0.012         # fully closed
    finger_len: float = 0.075
    finger_thick: float = 0.016
    finger_z0: float = 0.145            # nearest finger surface to the lens
    # The board and box standoffs above were authored against a 145 mm gripper. When
    # main() reads a different standoff out of `gripper.nominal_tag_z_m`, the scene
    # travels with the gripper by the difference — otherwise the objects stay put while
    # the gripper walks into them, and `target_can` (z 0.302-0.368) ends up occluding
    # the left tag outright. Seen for real at finger_z0 = 0.332: detection fell to 0/72.
    authored_for_finger_z0: float = 0.145
    # Tag spec defaults, overridden from the pipeline config by main(). They are read
    # from `gripper.aruco` rather than fixed here so that a change to what the real rig
    # carries cannot leave the only end-to-end test rendering the old tags — which would
    # show up as "0 detections" and look like a detector bug.
    tag_size_m: float = 0.016           # OUTER BLACK SQUARE side, what solvePnP wants
    tag_dictionary: str = "DICT_4X4_50"
    tag_face: str = "outward"           # outward (toward lens) | inward
    tag_ids: tuple = (0, 1)             # left finger, right finger


def jaw_separation(frame: int, n_frames: int, spec: SceneSpec) -> float:
    """One open -> close -> open cycle across the take, smooth (cosine) in time."""
    phase = 2.0 * math.pi * frame / max(n_frames - 1, 1)
    u = 0.5 * (1.0 + math.cos(phase))                 # 1 at the ends, 0 in the middle
    return spec.jaw_closed_m + (spec.jaw_open_m - spec.jaw_closed_m) * u


def build_scene(spec: SceneSpec, frame: int, n_frames: int):
    """World-frame primitives for one frame, plus the GT bookkeeping for that frame."""
    prims, gt = [], {}

    # --- enclosure. Textured so CoTracker has something to hold on to. -----------
    zb, fy, cyr, hx = spec.room_back_z, spec.room_floor_y, spec.room_ceil_y, spec.room_half_x
    prims.append(Quad("wall_back", np.array([-hx, cyr, zb]),
                      np.array([2 * hx, 0, 0]), np.array([0, fy - cyr, 0]),
                      albedo=0.62, texture=checker_texture(14, 9, dark=0.55, light=0.72)))
    prims.append(Quad("floor", np.array([-hx, fy, -0.35]),
                      np.array([2 * hx, 0, 0]), np.array([0, 0, zb + 0.35]),
                      albedo=0.55, texture=checker_texture(11, 9, dark=0.42, light=0.62)))
    prims.append(Quad("ceiling", np.array([-hx, cyr, -0.35]),
                      np.array([2 * hx, 0, 0]), np.array([0, 0, zb + 0.35]),
                      albedo=0.70, texture=checker_texture(9, 7, dark=0.66, light=0.78)))
    for sgn, nm in ((-1, "wall_left"), (+1, "wall_right")):
        prims.append(Quad(nm, np.array([sgn * hx, cyr, -0.35]),
                          np.array([0, fy - cyr, 0]), np.array([0, 0, zb + 0.35]),
                          albedo=0.58,
                          texture=checker_texture(8, 11, dark=0.48, light=0.68)))
    gt["room"] = {"back_z_m": zb, "floor_y_m": fy, "ceiling_y_m": cyr,
                  "half_x_m": hx}

    # Objects were placed relative to the authored gripper standoff, so they move with
    # it. Without this the gripper walks into the scene as the standoff grows.
    dz = spec.finger_z0 - spec.authored_for_finger_z0
    sx, sy = spec.board_squares
    board_z = spec.board_z + dz
    p0 = np.array([-spec.board_w / 2, -spec.board_h / 2, board_z])
    e1 = np.array([spec.board_w, 0.0, 0.0])
    e2 = np.array([0.0, spec.board_h, 0.0])
    prims.append(Quad("checkerboard", p0, e1, e2, albedo=1.0, colour=(0.9, 0.9, 0.9),
                      texture=checker_texture(sx, sy)))
    corners = []
    for j in range(1, sy):
        for i in range(1, sx):
            corners.append((p0 + e1 * (i / sx) + e2 * (j / sy)).tolist())
    gt["checkerboard"] = {
        "squares": [sx, sy], "square_size_m": spec.board_square_m,
        "plane_z_m": board_z, "inner_corners_world_m": corners,
    }

    gt["boxes"] = []
    for name, c, h in spec.box_specs:
        c, h = np.array(c, float) + np.array([0.0, 0.0, dz]), np.array(h, float)
        prims.append(Box(name, c - h, c + h))
        gt["boxes"].append({"name": name, "centre_world_m": c.tolist(),
                            "half_extent_m": h.tolist(),
                            "nearest_z_m": float(c[2] - h[2])})

    sep = jaw_separation(frame, n_frames, spec)
    gt["gripper"] = {"jaw_separation_m": sep, "tag_ids": list(spec.tag_ids),
                     "tag_size_m": spec.tag_size_m,
                     "tag_dictionary": spec.tag_dictionary, "tag_face": spec.tag_face,
                     "dimensions_are_placeholders": True, "tags": []}
    for side, tag_id in zip((-1, +1), spec.tag_ids):
        inner_x = side * sep / 2.0
        outer_x = inner_x + side * spec.finger_thick
        lo = np.array([min(inner_x, outer_x), -0.030, spec.finger_z0])
        hi = np.array([max(inner_x, outer_x), 0.030, spec.finger_z0 + spec.finger_len])
        prims.append(Box(f"finger_{'L' if side < 0 else 'R'}", lo, hi, albedo=0.45))

        s = spec.tag_size_m
        plate = s * (1.0 + 2.0 * QUIET_RATIO)     # marker + its white quiet zone
        cxw = (inner_x + outer_x) / 2.0
        z_face = spec.finger_z0 - 0.0015 if spec.tag_face == "outward" else spec.finger_z0
        tp0 = np.array([cxw - plate / 2, -plate / 2, z_face])
        prims.append(Quad(f"tag_{tag_id}", tp0,
                          np.array([plate, 0.0, 0.0]), np.array([0.0, plate, 0.0]),
                          albedo=1.0,
                          texture=aruco_texture(tag_id, spec.tag_dictionary)))
        gt["gripper"]["tags"].append({
            "id": int(tag_id), "centre_world_m": [cxw, 0.0, z_face],
            "size_m": s,                          # the MARKER, not the plate
            "plate_size_m": plate, "quiet_ratio": QUIET_RATIO, "normal": [0, 0, -1],
        })
    gt["gripper"]["tag_centre_distance_m"] = float(
        abs(gt["gripper"]["tags"][1]["centre_world_m"][0]
            - gt["gripper"]["tags"][0]["centre_world_m"][0]))
    return prims, gt


def camera_pose(frame: int, n_frames: int):
    """Gentle handheld-looking sweep. Returns (R_world_cam, t_world_cam)."""
    u = frame / max(n_frames - 1, 1)
    t = np.array([0.055 * math.sin(2 * math.pi * u),
                  0.022 * math.sin(4 * math.pi * u) - 0.010,
                  -0.030 * u])
    yaw = math.radians(4.5 * math.sin(2 * math.pi * u))
    pitch = math.radians(2.5 * math.sin(2 * math.pi * u + 1.0))
    cy, sy_ = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    Ry = np.array([[cy, 0, sy_], [0, 1, 0], [-sy_, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return Ry @ Rx, t


# ----------------------------------------------------------------------- render

def render(prims, R_wc, t_wc, rays, valid, light_dir=np.array([-0.35, -0.55, -0.75])):
    """Cast the rays into the scene. Returns (grey float 0-1, depth_m, hit mask)."""
    d_world = rays @ R_wc.T
    o = t_wc
    L = light_dir / np.linalg.norm(light_dir)

    best_t = np.full(rays.shape[:2], np.inf)
    best_shade = np.zeros(rays.shape[:2])
    best_n = np.zeros_like(rays)
    for p in prims:
        t, shade, n = p.intersect(o, d_world)
        take = t < best_t
        best_t = np.where(take, t, best_t)
        best_shade = np.where(take, shade, best_shade)
        best_n = np.where(take[..., None], n, best_n)

    hit = np.isfinite(best_t) & valid
    lam = np.clip(-(best_n @ L), 0.0, 1.0)
    grey = np.clip(best_shade * (0.30 + 0.70 * lam), 0.0, 1.0)
    grey = np.where(hit, grey, 0.02)

    # depth along the optical axis, in the CAMERA frame
    z = best_t * rays[..., 2]
    depth_m = np.where(hit, z, 0.0)
    return grey, depth_m, hit


def quantise_depth(depth_m, fx, baseline_m, max_disparity, subpixel_bits=0):
    """Push metric depth through the disparity grid a real StereoDepth would use."""
    out = np.zeros_like(depth_m)
    nz = depth_m > 1e-6
    d = np.zeros_like(depth_m)
    d[nz] = fx * baseline_m / depth_m[nz]
    step = 1.0 / (2 ** subpixel_bits)
    dq = np.round(d / step) * step
    ok = nz & (dq >= 1.0) & (dq <= max_disparity)
    out[ok] = fx * baseline_m / dq[ok]
    return out, ok


# ------------------------------------------------------------------------ output

FRAME_FIELDS = (
    "idx", "t_unix", "t_monotonic",
    "rgb_file", "depth_file", "left_file", "right_file",
    "rgb_seq", "depth_seq", "left_seq", "right_seq",
    "rgb_t_device", "depth_t_device", "left_t_device", "right_t_device",
    "rgb_latency_ms", "depth_latency_ms", "skew_ms", "exposure_ms",
    "rgb_encoding", "rgb_frame_type",
)


def write_calibration_json(out: Path, cam: CameraModel, W: int, H: int):
    sx, sy = W / cam.width, H / cam.height
    intr = {"fx": cam.fx * sx, "fy": cam.fy * sy, "cx": cam.cx * sx, "cy": cam.cy * sy,
            "width": W, "height": H, "distortion": cam.dist.ravel().tolist()}
    doc = {
        "source": "SYNTHETIC — tools/make_synthetic_capture.py",
        "note": (
            "NOT a device calibration. Rendered with the camera model in "
            f"{cam.path.name}, which is itself synthesised (verified: "
            f"{str(cam.verified).lower()}). No hardware was measured."
        ),
        "camera_model_config": str(cam.path.relative_to(REPO)),
        "camera_model_verified": cam.verified,
        "medium": cam.medium,
        "streamed_intrinsics": {"left": intr, "right": intr, "depth": intr,
                                "color": intr},
        "device": {
            "stereo_left": "CAM_B", "stereo_right": "CAM_C",
            "baseline_cm": cam.baseline_m * 100.0,
            "intrinsics": {"CAM_B": {"default_size": [cam.width, cam.height],
                                     "fx": cam.fx, "fy": cam.fy,
                                     "cx": cam.cx, "cy": cam.cy,
                                     "distortion": cam.dist.ravel().tolist()}},
        },
    }
    (out / "calibration.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "captures/synthetic_0001")
    ap.add_argument("--camera", type=Path,
                    default=REPO / "configs/source_camera_air.yaml")
    ap.add_argument("--frames", type=int, default=72,
                    help="episodes shorter than 60 are skipped by the gripper pipeline "
                         "and shorter than 50 degrade goal sampling; 72 clears both")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--size", type=str, default=None,
                    help="render size WxH (default: the model's own image_size)")
    ap.add_argument("--start-time", type=float, default=DEFAULT_EPOCH)
    ap.add_argument("--max-disparity", type=int, default=190,
                    help="95 = standard, 190 = extended disparity")
    ap.add_argument("--subpixel-bits", type=int, default=0)
    ap.add_argument("--no-quantise", action="store_true",
                    help="write exact ray-cast depth instead of disparity-gridded depth")
    ap.add_argument("--tag-face", choices=["outward", "inward"], default="outward")
    ap.add_argument("--config", default="configs/pipeline.yaml",
                    help="pipeline config whose gripper.aruco block decides the rendered "
                         "dictionary, marker size and ids. Pass 'none' to keep this "
                         "file's own placeholder tag spec.")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cam = CameraModel.load(a.camera)
    W, H = (int(x) for x in a.size.lower().split("x")) if a.size \
        else (cam.width, cam.height)
    # Render the tags the pipeline is actually configured to look for. Without this the
    # only end-to-end test silently keeps rendering whatever was hardcoded here, and a
    # config change shows up as a detector failure instead of a stale-fixture failure.
    tag_kw = {}
    if str(a.config).lower() != "none":
        from umi_handheld.record import load_pipeline_config     # noqa: E402
        gcfg = load_pipeline_config(a.config)["gripper"]
        acfg = gcfg["aruco"]
        tag_kw = {"tag_dictionary": acfg["dictionary"],
                  "tag_size_m": float(acfg["marker_length_m"]),
                  "tag_ids": (int(acfg["left_tag_id"]), int(acfg["right_tag_id"])),
                  # Stand the fingers off at the distance the config's z gate expects.
                  # Left hardcoded, the gate rejects every synthetic detection the
                  # moment the real rig's working distance is measured and written in.
                  "finger_z0": float(gcfg["nominal_tag_z_m"])}
    spec = SceneSpec(tag_face=a.tag_face, **tag_kw)
    fx_s = cam.fx * (W / cam.width)
    min_z = fx_s * cam.baseline_m / a.max_disparity

    print(f"camera model : {a.camera.relative_to(REPO)}  "
          f"({cam.name}, medium={cam.medium}, verified={cam.verified})")
    print(f"render size  : {W}x{H}   fx={fx_s:.2f}  baseline={cam.baseline_m*1000:.0f} mm")
    print(f"frames       : {a.frames} @ {a.fps} fps")
    print(f"MinZ         : {min_z*1000:.1f} mm  (max_disparity={a.max_disparity})"
          f"{'  [derived, not measured]' if True else ''}")
    print(f"quantisation : {'off' if a.no_quantise else f'on, subpixel_bits={a.subpixel_bits}'}")
    print(f"max ray angle: {cam.max_ray_angle_deg():.1f} deg (model validity limit)")
    print(f"gripper jaws : {spec.jaw_closed_m*1000:.0f}-{spec.jaw_open_m*1000:.0f} mm, "
          f"tags {spec.tag_ids} @ {spec.tag_size_m*1000:.2f} mm {spec.tag_dictionary}, "
          f"face={spec.tag_face} "
          f"[PLACEHOLDER dimensions]")
    print(f"out          : {a.out}")
    if a.dry_run:
        print("\n--dry-run: config parsed, nothing rendered, nothing written.")
        return 0

    out = a.out
    for sub in ("rgb", "depth", "left", "right"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    rays, valid = cam.pixel_rays(W, H)
    png = [cv2.IMWRITE_PNG_COMPRESSION, 1]

    rgb_list, depth_list, assoc, frames_rows, gt_frames = [], [], [], [], []
    for i in range(a.frames):
        t_unix = a.start_time + i / a.fps
        stamp = f"{t_unix:.6f}"
        prims, gt = build_scene(spec, i, a.frames)
        R_wc, t_wc = camera_pose(i, a.frames)

        grey_l, depth_m, hit = render(prims, R_wc, t_wc, rays, valid)
        t_r = t_wc + R_wc @ np.array([cam.baseline_m, 0.0, 0.0])
        grey_r, _, _ = render(prims, R_wc, t_r, rays, valid)

        exact = depth_m.copy()
        if a.no_quantise:
            dq, ok = depth_m, depth_m > 1e-6
        else:
            dq, ok = quantise_depth(depth_m, fx_s, cam.baseline_m,
                                    a.max_disparity, a.subpixel_bits)
        dq = np.where(dq >= min_z, dq, 0.0)

        left8 = (grey_l * 255).astype(np.uint8)
        right8 = (grey_r * 255).astype(np.uint8)
        # rint, not astype: astype truncates, which biases every depth down by up to
        # 1 mm (fx*B/190 = 228.672 mm would store as 228) and would show up as a
        # systematic error in the step-5 depth round-trip test.
        depth16 = np.clip(np.rint(dq * DEPTH_SCALE), 0, 65535).astype(np.uint16)
        colour = cv2.applyColorMap((grey_l * 255).astype(np.uint8), cv2.COLORMAP_BONE)

        cv2.imwrite(str(out / "left" / f"{stamp}.png"), left8, png)
        cv2.imwrite(str(out / "right" / f"{stamp}.png"), right8, png)
        cv2.imwrite(str(out / "depth" / f"{stamp}.png"), depth16, png)
        cv2.imwrite(str(out / "rgb" / f"{stamp}.png"), colour, png)

        rgb_list.append(f"{stamp} rgb/{stamp}.png")
        depth_list.append(f"{stamp} depth/{stamp}.png")
        assoc.append(f"{stamp} rgb/{stamp}.png {stamp} depth/{stamp}.png")
        frames_rows.append({
            "idx": i + 1, "t_unix": stamp, "t_monotonic": f"{i / a.fps:.6f}",
            "rgb_file": f"rgb/{stamp}.png", "depth_file": f"depth/{stamp}.png",
            "left_file": f"left/{stamp}.png", "right_file": f"right/{stamp}.png",
            "rgb_seq": i, "depth_seq": i, "left_seq": i, "right_seq": i,
            "rgb_t_device": stamp, "depth_t_device": stamp,
            "left_t_device": stamp, "right_t_device": stamp,
            "rgb_latency_ms": "", "depth_latency_ms": "",     # 미측정 — synthetic
            "skew_ms": "0.000", "exposure_ms": "",            # perfectly synchronous
            "rgb_encoding": "raw", "rgb_frame_type": "",
        })

        valid_frac = float(np.count_nonzero(depth16) / depth16.size)
        nz = exact[exact > 1e-6]
        R_cw = R_wc.T
        gt_frames.append({
            "idx": i + 1, "t_unix": float(t_unix),
            "T_world_cam": {"R_world_cam": R_wc.tolist(),
                            "t_world_cam_m": t_wc.tolist()},
            "camera_position_world_m": t_wc.tolist(),
            "depth": {
                "valid_fraction": valid_frac,
                "exact_min_m": float(nz.min()) if nz.size else None,
                "exact_max_m": float(nz.max()) if nz.size else None,
                "quantised": not a.no_quantise,
                "min_z_cutoff_m": float(min_z),
            },
            "scene": gt,
            "object_distance_along_optical_axis_m": {
                b["name"]: float((R_cw @ (np.array(b["centre_world_m"]) - t_wc))[2])
                for b in gt["boxes"]
            },
            "gripper_tag_pixels": {
                str(tg["id"]): cv2.projectPoints(
                    (R_cw @ (np.array(tg["centre_world_m"]) - t_wc)).reshape(1, 1, 3),
                    np.zeros(3), np.zeros(3),
                    np.array([[fx_s, 0, cam.cx * W / cam.width],
                              [0, cam.fy * H / cam.height, cam.cy * H / cam.height],
                              [0, 0, 1]]), cam.dist)[0].ravel().tolist()
                for tg in gt["gripper"]["tags"]
            },
        })
        if (i + 1) % 12 == 0 or i == a.frames - 1:
            print(f"  frame {i+1}/{a.frames}  depth valid {valid_frac*100:5.1f}%  "
                  f"jaw {gt['gripper']['jaw_separation_m']*1000:5.1f} mm")

    hdr = "# recorded (SYNTHETIC) by tools/make_synthetic_capture.py\n# timestamp filename\n"
    (out / "rgb.txt").write_text("# color images\n" + hdr + "\n".join(rgb_list) + "\n")
    (out / "depth.txt").write_text("# depth maps\n" + hdr + "\n".join(depth_list) + "\n")
    (out / "associations.txt").write_text(
        "# rgb_timestamp rgb_file depth_timestamp depth_file\n" + "\n".join(assoc) + "\n")

    import csv
    with (out / "frames.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FRAME_FIELDS)
        w.writeheader()
        w.writerows(frames_rows)

    write_calibration_json(out, cam, W, H)

    gt_doc = {
        "generator": "tools/make_synthetic_capture.py",
        "synthetic": True,
        "warning": ("Ground truth for a RENDERED capture. No hardware was measured. "
                    "Scene and gripper dimensions are placeholders."),
        "camera_model": {
            "config": str(a.camera.relative_to(REPO)), "name": cam.name,
            "verified": cam.verified, "medium": cam.medium,
            "render_size": [W, H],
            "fx": fx_s, "fy": cam.fy * H / cam.height,
            "cx": cam.cx * W / cam.width, "cy": cam.cy * H / cam.height,
            "distortion": cam.dist.ravel().tolist(),
            "baseline_m": cam.baseline_m,
        },
        "depth_convention": {
            "units": "millimetre", "dtype": "uint16", "scale": DEPTH_SCALE,
            "metres": "png_value / 1000.0", "zero_means": "invalid",
            "quantised_to_disparity_grid": not a.no_quantise,
            "max_disparity": a.max_disparity, "subpixel_bits": a.subpixel_bits,
            "min_z_m": float(min_z),
            "min_z_provenance": "derived: fx*baseline/max_disparity — 미측정",
        },
        "scene_spec": {k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in spec.__dict__.items()},
        "frames": gt_frames,
    }
    (out / "ground_truth.json").write_text(json.dumps(gt_doc, indent=2), encoding="utf-8")

    (out / "metadata.txt").write_text(
        "SYNTHETIC CAPTURE — tools/make_synthetic_capture.py\n"
        "No hardware was connected. Every image is rendered.\n\n"
        f"camera model : {a.camera.relative_to(REPO)} "
        f"(verified={cam.verified}, medium={cam.medium})\n"
        f"frames       : {a.frames} @ {a.fps} fps, {W}x{H}\n"
        f"depth        : uint16 millimetres, metres = value/{DEPTH_SCALE:.1f}, "
        f"0 = invalid\n"
        f"MinZ         : {min_z*1000:.1f} mm [derived, not measured]\n"
        "ground truth : ground_truth.json\n", encoding="utf-8")

    print(f"\nwrote {a.frames} frames to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
