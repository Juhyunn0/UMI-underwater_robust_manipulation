"""Camera models loaded from configs/*.yaml — never from hardcoded numbers.

One class serves both ends of the pipeline: the bench OAK-D W the demonstrations are
recorded on (in air) and the MarineSitu C3 the policy is deployed on (underwater). They
differ only by the YAML they are loaded from, which is the point: when the bench unit is
finally connected, one file changes and no code does.

The `verified` flag is load-bearing, not decorative. A model whose numbers were never
measured warns on load, is stamped into every artifact built from it, and makes the
training export refuse to run without --allow-unverified.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]

SCHEMA = "umi_camera_model/1"


class UnverifiedModelError(RuntimeError):
    """Raised when an unverified model reaches a gate that requires a measured one."""


class CameraModel:
    """OpenCV pinhole + rational distortion, plus a robust pixel -> ray inverse."""

    def __init__(self, cfg: dict, path: Path):
        self.path = Path(path)
        self.cfg = cfg
        self.name = cfg["name"]
        self.role = cfg.get("role", "unknown")
        self.medium = cfg.get("medium", "unknown")
        self.verified = bool(cfg.get("verified", False))
        self.rectified = bool(cfg.get("rectified", False))
        self.width, self.height = (int(v) for v in cfg["image_size"])
        k = cfg["intrinsics"]
        self.fx, self.fy = float(k["fx"]), float(k["fy"])
        self.cx, self.cy = float(k["cx"]), float(k["cy"])
        self.dist = np.asarray(cfg["distortion"], dtype=np.float64).reshape(1, -1)
        st = cfg.get("stereo", {}) or {}
        self.baseline_m = float(st.get("baseline_m", 0.075))
        # Which physical unit this model was read off, when it was read off one. More
        # than one DepthAI device is usually reachable — the bench OAK-D-W over USB and
        # the C3 over PoE — and dai.Device() with no argument takes whichever it finds
        # first. Recording the wrong one would file the wrong calibration with the
        # frames, so callers pin the device with this.
        self.device_mxid = (cfg.get("provenance") or {}).get("device_mxid")

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(cls, path) -> "CameraModel":
        path = Path(path)
        if not path.is_absolute():
            path = REPO / path
        with open(path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        if cfg.get("schema") != SCHEMA:
            raise ValueError(f"{path}: expected schema {SCHEMA!r}, got {cfg.get('schema')!r}")
        m = cls(cfg, path)
        if not m.verified:
            print(f"WARNING: camera model '{m.name}' is UNVERIFIED ({m.rel}). "
                  f"No hardware was measured; the values are synthesised.",
                  file=sys.stderr)
        return m

    @property
    def rel(self) -> str:
        try:
            return str(self.path.relative_to(REPO))
        except ValueError:
            return str(self.path)

    def require_verified(self, what: str, allow: bool = False) -> None:
        if self.verified or allow:
            return
        raise UnverifiedModelError(
            f"{what} refuses to run on the UNVERIFIED camera model '{self.name}' "
            f"({self.rel}). No hardware has been measured. Pass --allow-unverified to "
            f"proceed anyway; the flag is recorded in the output."
        )

    def provenance(self) -> dict:
        """The block every artifact carries, so a stale model cannot hide."""
        return {
            "config": self.rel, "name": self.name, "role": self.role,
            "medium": self.medium, "verified": self.verified,
            "rectified": self.rectified,
            "image_size": [self.width, self.height],
            "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
            "distortion": self.dist.ravel().tolist(),
            "baseline_m": self.baseline_m,
        }

    # ---------------------------------------------------------------- geometry

    def K(self, width=None, height=None) -> np.ndarray:
        sx = (width or self.width) / self.width
        sy = (height or self.height) / self.height
        return np.array([[self.fx * sx, 0, self.cx * sx],
                         [0, self.fy * sy, self.cy * sy],
                         [0, 0, 1]], dtype=np.float64)

    def _radial_table(self, n: int = 4096):
        """r_undistorted -> r_distorted over the monotone part of the model."""
        d = self.dist.ravel()
        if abs(d[2]) > 1e-6 or abs(d[3]) > 1e-6:
            raise ValueError(
                f"{self.rel}: p1/p2 are non-zero, so the radial inverse used here does "
                f"not apply. Use cv2.undistortPoints for this model instead.")
        th = np.deg2rad(np.linspace(0.0, 89.0, n))
        r_u = np.tan(th)
        pts = np.stack([r_u, np.zeros_like(r_u), np.ones_like(r_u)], 1)[:, None, :]
        uv, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), np.eye(3), self.dist)
        r_d = uv.reshape(-1, 2)[:, 0]
        ok = np.isfinite(r_d)
        r_u, r_d = r_u[ok], r_d[ok]
        keep = np.concatenate([[True], np.diff(r_d) > 0])
        stop = int(np.argmin(keep)) if not keep.all() else len(keep)
        return r_u[:stop], r_d[:stop]

    def max_ray_angle_deg(self) -> float:
        r_u, _ = self._radial_table()
        return float(np.rad2deg(np.arctan(r_u[-1])))

    def pixel_rays(self, width=None, height=None):
        """Unit ray per pixel (x right, y down, z forward) + a validity mask."""
        W, H = width or self.width, height or self.height
        K = self.K(W, H)
        u, v = np.meshgrid(np.arange(W, dtype=np.float64),
                           np.arange(H, dtype=np.float64))
        xd = (u - K[0, 2]) / K[0, 0]
        yd = (v - K[1, 2]) / K[1, 1]
        rd = np.hypot(xd, yd)
        r_u_tab, r_d_tab = self._radial_table()
        ru = np.interp(rd, r_d_tab, r_u_tab, left=0.0, right=np.nan)
        valid = np.isfinite(ru)
        scale = np.divide(ru, rd, out=np.ones_like(rd), where=rd > 1e-12)
        d = np.stack([xd * scale, yd * scale, np.ones_like(rd)], -1)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        return d, valid

    def unproject(self, pts_px: np.ndarray, width=None, height=None) -> np.ndarray:
        """Pixels -> unit rays. Uses cv2 so models with p1/p2 are handled too."""
        K = self.K(width, height)
        p = np.asarray(pts_px, np.float64).reshape(-1, 1, 2)
        n = cv2.undistortPoints(p, K, self.dist).reshape(-1, 2)
        d = np.concatenate([n, np.ones((len(n), 1))], 1)
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def project(self, pts_cam: np.ndarray, width=None, height=None) -> np.ndarray:
        """Camera-frame 3-D points -> pixels."""
        K = self.K(width, height)
        p = np.asarray(pts_cam, np.float64).reshape(-1, 1, 3)
        uv, _ = cv2.projectPoints(p, np.zeros(3), np.zeros(3), K, self.dist)
        return uv.reshape(-1, 2)

    def min_z_m(self, max_disparity: int, width=None) -> float:
        """Closest triangulable range: fx*B/d_max. Derived, never measured."""
        return float(self.K(width)[0, 0] * self.baseline_m / max_disparity)
