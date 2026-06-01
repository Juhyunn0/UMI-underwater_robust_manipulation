#!/usr/bin/env python3
"""rebuild_trajectory_html.py — regenerate trajectory_interactive.html for an
existing recording using a DIFFERENT R_gantry_to_slam, scale, or anchor tag,
WITHOUT re-running the experiment.

It reads the recording's per-stream CSVs (``gantry_telemetry.csv``,
``camera_trajectory.csv``, optional ``tag_poses.csv``) and re-runs the same
dashboard generator the live pipeline uses (``tagslam_core.
write_experiment_dashboard_html``), so the output is the identical 2-tab
(Trajectory + Velocity) dashboard — just transformed by the calibration you ask
for.

    # reproduce the recording's own dashboard (uses the calibration YAML as-is)
    python -m src.tools.rebuild_trajectory_html \
        --input-dir data/20260528/20260528_215858_recording

    # try a different gantry->SLAM rotation (9 numbers, row-major) and scale
    python -m src.tools.rebuild_trajectory_html --input-dir <dir> \
        --R 1 0 0 0 1 0 0 0 1 --scale 1.2 --output /tmp/test.html

    # re-anchor the whole view to a chosen tag's frame
    python -m src.tools.rebuild_trajectory_html --input-dir <dir> --anchor-tag 25

Re-anchoring (``--anchor-tag T``): the dashboard's ``anchor_id`` only HIGHLIGHTS
a tag; it does not re-frame coordinates. When ``--anchor-tag T`` names a tag
present in ``tag_poses.csv``, this tool computes the rigid transform that maps
the SLAM world frame into tag T's frame and passes it as ``extra_world_transform``
so the camera, the tags, and the gantry overlay are all re-expressed in tag T's
frame together (their mutual alignment is preserved). Tag orientation columns are
not rotated (positions/velocities are what re-anchored trajectory plots need).

NOTE: this tool imports ``tagslam_core`` for the HTML generator, which imports
cv2 (and gtsam) at module load. Those are present on the capture machine; on a
bare environment the tool exits with an actionable message rather than a raw
traceback. Reading the calibration YAML itself needs neither (minimal YAML
parse), so argument validation works everywhere.

Comments, logs, and report text are English by project convention.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ── repo wiring: this file lives in src/tools/, so src/ is the parent's parent.
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
_REPO_ROOT = _SRC_DIR.parent

DEFAULT_CALIB = _REPO_ROOT / "config" / "fisheye_calibration.yaml"
DEFAULT_CONFIG = _REPO_ROOT / "config" / "config.yaml"


# ── calibration (minimal, cv2-free) ──────────────────────────────────────────
def _parse_R_arg(values):
    """--R is either 9 inline floats (row-major) or a path to a YAML/text file
    containing an ``R_gantry_to_slam`` 3x3. Returns a 3x3 ndarray or None."""
    if not values:
        return None
    if len(values) == 1:  # a path
        import yaml
        data = yaml.safe_load(Path(values[0]).read_text(encoding="utf-8")) or {}
        R = np.asarray(data.get("R_gantry_to_slam"), dtype=np.float64)
        return R.reshape(3, 3)
    if len(values) == 9:
        return np.asarray([float(v) for v in values], dtype=np.float64).reshape(3, 3)
    raise SystemExit("error: --R takes either 9 numbers (row-major) or 1 file path")


def load_calibration(path: Path, args):
    """Return (R 3x3, scale float, offset_mm 3-vec|None, T_gc 4x4|None).

    Reads the calibration YAML with a minimal parser (no cv2), then applies any
    CLI overrides (--R / --scale / --gantry-anchor-offset)."""
    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    R = np.asarray(data.get("R_gantry_to_slam", np.eye(3)), dtype=np.float64)
    if R.shape != (3, 3):
        R = np.eye(3)
    scale = float(data.get("gantry_to_slam_scale", 1.0) or 1.0)
    offset = data.get("gantry_anchor_offset_mm")
    offset_mm = (np.asarray(offset, dtype=np.float64).reshape(3).tolist()
                 if offset is not None else None)
    T_gc = data.get("T_gantry_camera")
    T_gc = np.asarray(T_gc, dtype=np.float64).reshape(4, 4) if T_gc is not None else None

    R_override = _parse_R_arg(args.R)
    if R_override is not None:
        R = R_override
    if args.scale is not None:
        scale = float(args.scale)
    if args.gantry_anchor_offset is not None:
        offset_mm = [float(v) for v in args.gantry_anchor_offset]
    return R, scale, offset_mm, T_gc


# ── re-anchoring ──────────────────────────────────────────────────────────────
def _read_tag_pose(tag_csv: Path, tag_id: int):
    """Return (x_m, y_m, z_m, roll_deg, pitch_deg, yaw_deg) for tag_id, or None."""
    import csv
    if not tag_csv.exists():
        return None
    with tag_csv.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                if int(float(r["tag_id"])) == int(tag_id):
                    return (float(r["x_m"]), float(r["y_m"]), float(r.get("z_m", 0.0) or 0.0),
                            float(r.get("roll_deg", 0.0) or 0.0),
                            float(r.get("pitch_deg", 0.0) or 0.0),
                            float(r.get("yaw_deg", 0.0) or 0.0))
            except (KeyError, ValueError, TypeError):
                continue
    return None


def reanchor_transform(tag_csv: Path, tag_id: int, rpy_deg_to_matrix):
    """4x4 = inv(world_T_tagT): maps SLAM-world coords into tag T's frame.

    Uses the project's rpy_deg_to_matrix (R = Rz·Ry·Rx, the same convention the
    CSV roll/pitch/yaw were written with), so no Euler-convention mismatch."""
    pose = _read_tag_pose(tag_csv, tag_id)
    if pose is None:
        return None
    x, y, z, roll, pitch, yaw = pose
    R = rpy_deg_to_matrix(roll, pitch, yaw)
    world_T_tag = np.eye(4)
    world_T_tag[:3, :3] = R
    world_T_tag[:3, 3] = [x, y, z]
    return np.linalg.inv(world_T_tag)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, required=True,
                    help="Recording folder (gantry_telemetry.csv + camera_trajectory.csv [+ tag_poses.csv]).")
    ap.add_argument("--calib", type=Path, default=DEFAULT_CALIB,
                    help="Calibration YAML for R / scale / offset / T_gantry_camera.")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                    help="config.yaml for the pool outline.")
    ap.add_argument("--R", nargs="+", default=None,
                    help="Override R_gantry_to_slam: 9 numbers (row-major) OR a YAML/text path.")
    ap.add_argument("--scale", type=float, default=None,
                    help="Override gantry_to_slam_scale.")
    ap.add_argument("--gantry-anchor-offset", nargs=3, type=float, default=None,
                    metavar=("X", "Y", "Z"), help="Override gantry_anchor_offset_mm (mm).")
    ap.add_argument("--anchor-tag", type=int, default=1,
                    help="Tag id. If present in tag_poses.csv, re-frame the whole view into "
                         "this tag's frame; otherwise just highlight it.")
    ap.add_argument("--tag-size", type=float, default=0.17, help="Tag size (m) for the viewer.")
    ap.add_argument("--plot-z-scale", type=float, default=1.0, help="Vertical exaggeration for the 3D view.")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output HTML path (default <input-dir>/trajectory_interactive.html).")
    args = ap.parse_args(argv)

    input_dir = args.input_dir
    if not input_dir.is_dir():
        print(f"error: input directory not found: {input_dir}", file=sys.stderr)
        return 2
    calib_path = args.calib if args.calib.is_absolute() else (_REPO_ROOT / args.calib)
    if not calib_path.exists():
        print(f"error: calibration not found: {calib_path}", file=sys.stderr)
        return 2

    gantry_csv = input_dir / "gantry_telemetry.csv"
    camera_csv = input_dir / "camera_trajectory.csv"
    tag_csv = input_dir / "tag_poses.csv"
    if not gantry_csv.exists():
        print(f"error: gantry_telemetry.csv not found in {input_dir}", file=sys.stderr)
        return 2
    if not camera_csv.exists():
        print(f"error: camera_trajectory.csv not found in {input_dir}", file=sys.stderr)
        return 2

    # Calibration read needs no heavy deps.
    R, scale, offset_mm, T_gc = load_calibration(calib_path, args)
    output = args.output or (input_dir / "trajectory_interactive.html")

    # Heavy import (cv2/gtsam) only needed for the HTML generator itself.
    try:
        import tagslam_core as tc
        from tagslam.visualization import normalize_pool_config
    except Exception as exc:  # noqa: BLE001 — surface a friendly message
        print("error: could not import tagslam_core / tagslam.visualization "
              f"({exc.__class__.__name__}: {exc}).\n"
              "This tool needs the capture-machine dependencies (cv2, gtsam, "
              "pupil_apriltags) to render the dashboard. Run it where experiments "
              "run, or install those packages.", file=sys.stderr)
        return 2

    # Pool config (best-effort; an empty cfg still renders).
    pool_cfg: dict = {}
    try:
        cfg_path = args.config if args.config.is_absolute() else (_REPO_ROOT / args.config)
        if cfg_path.exists():
            runtime_cfg = tc.parse_simple_yaml(cfg_path.read_text(encoding="utf-8"))
            pool_cfg = normalize_pool_config(runtime_cfg.get("pool", {}))
    except Exception as exc:  # noqa: BLE001
        print(f"[rebuild] pool config load failed ({exc}); continuing without pool outline.",
              file=sys.stderr)

    # Optional re-anchor: map everything into the chosen tag's frame.
    M = None
    if tag_csv.exists():
        M = reanchor_transform(tag_csv, args.anchor_tag, tc.rpy_deg_to_matrix)
        if M is not None:
            print(f"[rebuild] re-anchoring view into tag {args.anchor_tag}'s frame.")
        else:
            print(f"[rebuild] tag {args.anchor_tag} not in tag_poses.csv — "
                  "highlight only, no re-frame.", file=sys.stderr)

    out = tc.write_experiment_dashboard_html(
        output,
        gantry_csv=gantry_csv,
        camera_csv=camera_csv,
        tag_poses_csv=(tag_csv if tag_csv.exists() else None),
        pool_cfg=pool_cfg,
        anchor_id=int(args.anchor_tag),
        T_gantry_camera=T_gc,
        gantry_anchor_offset_mm=offset_mm,
        R_gantry_to_slam=R,
        gantry_to_slam_scale=scale,
        extra_world_transform=M,
        run_name=input_dir.name,
        tag_size_m=float(args.tag_size),
        plot_z_scale=float(args.plot_z_scale),
    )
    if out is None:
        print("error: dashboard generation returned None (no usable rows?).", file=sys.stderr)
        return 1
    print(f"✓ Wrote {out}")
    print(f"   R={np.array2string(R, precision=4)}  scale={scale:.5f}  "
          f"anchor_tag={args.anchor_tag}  reframed={'yes' if M is not None else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
