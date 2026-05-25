from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


POOL_EDGE_COLOR = "#4A7A9C"
POOL_WATER_COLOR = "#6FB3D9"
POOL_FLOOR_COLOR = "#315C78"
POOL_GRID_COLOR = "#CBD5DD"


@dataclass(frozen=True)
class PoolGeometry:
    floor: np.ndarray
    top: np.ndarray
    axis_limits: dict[str, tuple[float, float]]
    floor_z: float
    water_z: float
    long_axis: str

    def to_json(self) -> dict[str, Any]:
        return {
            "floor": self.floor.tolist(),
            "top": self.top.tolist(),
            "axis_limits": {
                axis: [float(values[0]), float(values[1])]
                for axis, values in self.axis_limits.items()
            },
            "floor_z": float(self.floor_z),
            "water_z": float(self.water_z),
            "long_axis": self.long_axis,
        }


def normalize_pool_config(pool_cfg: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(pool_cfg or {})
    cfg.setdefault("length_m", 4.877)
    cfg.setdefault("width_m", 2.438)
    cfg.setdefault("depth_m", 1.143)
    cfg.setdefault("water_depth_m", cfg["depth_m"])
    cfg.setdefault("pool_corner_offset_m", 0.10)
    cfg.setdefault("pool_long_axis", "y")
    cfg.setdefault("tag1_position_m", [4.77, 2.34, 0.0])
    cfg["pool_long_axis"] = str(cfg["pool_long_axis"]).strip('"').lower()
    if cfg["pool_long_axis"] not in {"x", "y"}:
        raise ValueError("pool.pool_long_axis must be 'x' or 'y'")
    cfg["tag1_position_m"] = [float(value) for value in cfg["tag1_position_m"]]
    for key in ("length_m", "width_m", "depth_m", "water_depth_m", "pool_corner_offset_m"):
        cfg[key] = float(cfg[key])
    return cfg


def _away_sign(value: float) -> float:
    return 1.0 if value >= 0.0 else -1.0


def compute_pool_geometry(pool_cfg: dict[str, Any] | None) -> PoolGeometry:
    cfg = normalize_pool_config(pool_cfg)
    tag1 = np.asarray(cfg["tag1_position_m"], dtype=np.float64)
    offset = float(cfg["pool_corner_offset_m"])
    long_axis = str(cfg["pool_long_axis"])
    short_axis = "y" if long_axis == "x" else "x"
    axis_index = {"x": 0, "y": 1}

    nearest_corner = tag1.copy()
    for axis in ("x", "y"):
        i = axis_index[axis]
        nearest_corner[i] += _away_sign(tag1[i]) * offset

    long_vec = np.zeros(3, dtype=np.float64)
    short_vec = np.zeros(3, dtype=np.float64)
    long_i = axis_index[long_axis]
    short_i = axis_index[short_axis]
    long_vec[long_i] = -_away_sign(nearest_corner[long_i]) * cfg["length_m"]
    short_vec[short_i] = -_away_sign(nearest_corner[short_i]) * cfg["width_m"]

    floor_z = float(tag1[2])
    water_z = floor_z - cfg["water_depth_m"]
    nearest_corner[2] = floor_z

    floor = np.vstack(
        (
            nearest_corner,
            nearest_corner + long_vec,
            nearest_corner + long_vec + short_vec,
            nearest_corner + short_vec,
        )
    )
    top = floor.copy()
    top[:, 2] = water_z

    points = np.vstack((floor, top))
    margin = 0.30
    axis_limits = {
        "x": (float(np.min(points[:, 0]) - margin), float(np.max(points[:, 0]) + margin)),
        "y": (float(np.min(points[:, 1]) - margin), float(np.max(points[:, 1]) + margin)),
        "z": (float(np.min(points[:, 2]) - margin), float(np.max(points[:, 2]) + margin)),
    }
    return PoolGeometry(
        floor=floor,
        top=top,
        axis_limits=axis_limits,
        floor_z=floor_z,
        water_z=water_z,
        long_axis=long_axis,
    )


def pool_geometry_json(pool_cfg: dict[str, Any] | None) -> dict[str, Any]:
    return compute_pool_geometry(pool_cfg).to_json()


def draw_pool(ax, pool_cfg: dict[str, Any] | None) -> PoolGeometry:
    geometry = compute_pool_geometry(pool_cfg)
    floor = geometry.floor
    top = geometry.top

    try:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        water = Poly3DCollection(
            [top],
            facecolors=POOL_WATER_COLOR,
            edgecolors="none",
            alpha=0.08,
        )
        ax.add_collection3d(water)
    except Exception:
        pass

    _draw_loop(ax, floor, POOL_FLOOR_COLOR, linewidth=1.8, linestyle="-", alpha=0.95)
    _draw_loop(ax, top, POOL_EDGE_COLOR, linewidth=1.4, linestyle="--", alpha=0.78)
    for i in range(4):
        ax.plot(
            [floor[i, 0], top[i, 0]],
            [floor[i, 1], top[i, 1]],
            [floor[i, 2], top[i, 2]],
            color=POOL_EDGE_COLOR,
            linewidth=0.9,
            alpha=0.58,
        )

    draw_floor_grid(ax, geometry)
    set_pool_axes(ax, geometry)
    return geometry


def draw_tag1_drift(ax, pool_cfg: dict[str, Any] | None, tag1_estimate) -> float | None:
    cfg = normalize_pool_config(pool_cfg)
    gt = np.asarray(cfg["tag1_position_m"], dtype=np.float64)
    if tag1_estimate is None:
        ax.scatter(gt[0], gt[1], gt[2], color="#222222", s=45, marker="x", label="Tag 1 GT (config)")
        return None

    est = np.asarray(tag1_estimate, dtype=np.float64)
    drift_m = float(np.linalg.norm(est - gt))
    ax.scatter(gt[0], gt[1], gt[2], color="#222222", s=45, marker="x", label="Tag 1 GT (config)")
    ax.scatter(est[0], est[1], est[2], color="#C46A3A", s=42, marker="o", label="Tag 1 ISAM2")
    ax.plot(
        [gt[0], est[0]],
        [gt[1], est[1]],
        [gt[2], est[2]],
        color="#C46A3A",
        linewidth=1.2,
        linestyle=":",
        alpha=0.85,
    )
    return drift_m


def draw_floor_grid(ax, geometry: PoolGeometry, divisions: int = 8) -> None:
    floor = geometry.floor
    p0, p1, p2, p3 = floor
    for i in range(1, divisions):
        u = i / divisions
        a = p0 * (1.0 - u) + p1 * u
        b = p3 * (1.0 - u) + p2 * u
        c = p0 * (1.0 - u) + p3 * u
        d = p1 * (1.0 - u) + p2 * u
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=POOL_GRID_COLOR, linewidth=0.55)
        ax.plot([c[0], d[0]], [c[1], d[1]], [c[2], d[2]], color=POOL_GRID_COLOR, linewidth=0.55)


def set_pool_axes(ax, geometry: PoolGeometry) -> None:
    limits = geometry.axis_limits
    ax.set_xlim(*limits["x"])
    ax.set_ylim(*limits["y"])
    ax.set_zlim(*limits["z"])
    try:
        ax.set_box_aspect(
            (
                limits["x"][1] - limits["x"][0],
                limits["y"][1] - limits["y"][0],
                limits["z"][1] - limits["z"][0],
            )
        )
    except Exception:
        pass


def _draw_loop(ax, points: np.ndarray, color: str, linewidth: float, linestyle: str, alpha: float) -> None:
    closed = np.vstack((points, points[0]))
    ax.plot(
        closed[:, 0],
        closed[:, 1],
        closed[:, 2],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
    )
