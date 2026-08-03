#!/usr/bin/env python3
"""
geometry.py — intrinsics and depth(mm) -> XYZ(m) deprojection.

This is the seam where RGB-D turns into something a manipulation policy or a
SLAM front end can use. Two things to keep straight:

* DepthAI's `depth` frame is uint16 **millimetres**, 0 meaning "no measurement"
  (not "zero distance"). Never let a 0 through as a valid point.
* Intrinsics must be read at the *resolution you actually receive*. DepthAI's
  `getCameraIntrinsics(socket, w, h)` rescales the factory calibration for you,
  which is why the resolved output size is threaded through here rather than
  assuming the calibration resolution.

Refraction: the factory calibration is an in-air calibration. Underwater, the
flat/dome port changes the effective focal length and adds radial error, so
these intrinsics are a starting point, not a final answer, for in-water metric
work. The existing refractive calibration work in `src/` applies here too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics for one image at one specific size."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: tuple[float, ...] = ()

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    def scaled_to(self, width: int, height: int) -> "Intrinsics":
        """Rescale for a resized image (uniform scaling only)."""
        sx, sy = width / self.width, height / self.height
        return Intrinsics(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy,
                          width, height, self.distortion)

    def to_dict(self) -> dict:
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
                "width": self.width, "height": self.height,
                "distortion": list(self.distortion)}


def intrinsics_from_calibration(calib, socket, width: int, height: int) -> Intrinsics:
    """Read factory intrinsics, rescaled to the size actually being streamed."""
    K = calib.getCameraIntrinsics(socket, width, height)
    try:
        dist = tuple(float(d) for d in calib.getDistortionCoefficients(socket))
    except Exception:  # noqa: BLE001
        dist = ()
    return Intrinsics(fx=float(K[0][0]), fy=float(K[1][1]),
                      cx=float(K[0][2]), cy=float(K[1][2]),
                      width=width, height=height, distortion=dist)


def depth_to_xyz(depth_mm: np.ndarray, intr: Intrinsics,
                 stride: int = 1,
                 min_mm: float = 100.0,
                 max_mm: float = 20000.0) -> np.ndarray:
    """Deproject a uint16 millimetre depth map to an (N, 3) float32 cloud in metres.

    Camera frame: +X right, +Y down, +Z forward — the OpenCV convention, which
    is also what ROS's `image_geometry` optical frame uses (`*_optical_frame`).
    Converting to a ROS body frame (+X forward, +Y left, +Z up) is a fixed
    rotation applied downstream, deliberately not baked in here.

    Invalid pixels (0, or outside [min_mm, max_mm]) are dropped, so N <= H*W.
    """
    if depth_mm.ndim != 2:
        raise ValueError(f"expected a 2-D depth map, got shape {depth_mm.shape}")

    d = depth_mm[::stride, ::stride]
    i = intr if (intr.width, intr.height) == (depth_mm.shape[1], depth_mm.shape[0]) \
        else intr.scaled_to(depth_mm.shape[1], depth_mm.shape[0])

    h, w = d.shape
    us = (np.arange(w, dtype=np.float32) * stride)
    vs = (np.arange(h, dtype=np.float32) * stride)
    uu, vv = np.meshgrid(us, vs)

    z = d.astype(np.float32) / 1000.0
    valid = (d > 0) & (d >= min_mm) & (d <= max_mm)
    z = z[valid]
    uu = uu[valid]
    vv = vv[valid]

    x = (uu - i.cx) * z / i.fx
    y = (vv - i.cy) * z / i.fy
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def rgbd_to_pointcloud(color_bgr: np.ndarray, depth_mm: np.ndarray,
                       intr: Intrinsics, stride: int = 1,
                       min_mm: float = 100.0,
                       max_mm: float = 20000.0) -> tuple[np.ndarray, np.ndarray]:
    """Coloured cloud: returns (xyz_m (N,3) float32, rgb (N,3) uint8).

    Requires colour and depth to be pixel-aligned, i.e. streamed with
    `--depth-align color`. If the two differ in size, colour is nearest-neighbour
    sampled onto the depth grid — cheap and adequate for visualisation, but
    resample properly if you are training on it.
    """
    if depth_mm.ndim != 2:
        raise ValueError(f"expected a 2-D depth map, got shape {depth_mm.shape}")

    dh, dw = depth_mm.shape
    ch, cw = color_bgr.shape[:2]
    d = depth_mm[::stride, ::stride]

    if (ch, cw) != (dh, dw):
        ys = (np.arange(d.shape[0]) * stride * ch // dh).clip(0, ch - 1)
        xs = (np.arange(d.shape[1]) * stride * cw // dw).clip(0, cw - 1)
        colour = color_bgr[np.ix_(ys, xs)]
    else:
        colour = color_bgr[::stride, ::stride]

    i = intr if (intr.width, intr.height) == (dw, dh) else intr.scaled_to(dw, dh)

    h, w = d.shape
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32) * stride,
                         np.arange(h, dtype=np.float32) * stride)
    valid = (d > 0) & (d >= min_mm) & (d <= max_mm)

    z = d.astype(np.float32)[valid] / 1000.0
    x = (uu[valid] - i.cx) * z / i.fx
    y = (vv[valid] - i.cy) * z / i.fy
    xyz = np.stack([x, y, z], axis=-1).astype(np.float32)
    rgb = colour[valid][:, ::-1].copy()  # BGR -> RGB
    return xyz, rgb
