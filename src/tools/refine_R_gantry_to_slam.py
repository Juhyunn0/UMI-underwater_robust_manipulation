#!/usr/bin/env python3
"""refine_R_gantry_to_slam.py — data-driven refinement of R_gantry_to_slam.

The dashboard overlays the gantry ground-truth trajectory on the camera (SLAM)
trajectory using a fixed rotation ``R_gantry_to_slam`` from the calibration YAML.
When that matrix is set by hand it is usually a clean 90° permutation, but the
real anchor tag can sit a few degrees off, leaving a residual yaw/tilt that bows
the two curves apart. This tool reads an existing recording, solves the optimal
rotation that maps the gantry path onto the camera path (orthogonal Procrustes),
composes it with the current R, and (optionally) writes the refined R back.

    python -m src.tools.refine_R_gantry_to_slam \
        --input-dir data/20260528/20260528_213015_recording \
        [--write] [--calib config/fisheye_calibration.yaml] \
        [--min-displacement-mm 50] [--rms-threshold-mm 5.0] [--max-angle-deg 15.0]

No SciPy dependency: Procrustes is solved with a NumPy SVD; the YAML is edited
with a targeted block replacement (no full re-dump) so K/D/T_gantry_camera/etc.
are preserved byte-for-byte. A timestamped backup is written before any --write.

Comments, logs, and report text are English by project convention.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ── repo wiring: this file lives in src/tools/, so src/ is the parent's parent.
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
_REPO_ROOT = _SRC_DIR.parent

DEFAULT_CALIB = _REPO_ROOT / "config" / "fisheye_calibration.yaml"


# ── CSV loading ─────────────────────────────────────────────────────────────
def _load_csv_columns(path: Path, columns: list[str]) -> dict[str, np.ndarray]:
    """Load the named float columns from a CSV. Rows with a non-parseable value
    in any requested column are skipped. Returns {col: np.ndarray}."""
    out: dict[str, list[float]] = {c: [] for c in columns}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in columns if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"error: {path.name} is missing columns {missing}; "
                             f"found {reader.fieldnames}")
        for row in reader:
            try:
                vals = [float(row[c]) for c in columns]
            except (TypeError, ValueError):
                continue
            for c, v in zip(columns, vals):
                out[c].append(v)
    return {c: np.asarray(v, dtype=np.float64) for c, v in out.items()}


def load_gantry(input_dir: Path):
    cols = _load_csv_columns(input_dir / "gantry_telemetry.csv",
                             ["timestamp_monotonic", "x_mm", "y_mm", "z_mm"])
    t = cols["timestamp_monotonic"]
    xyz_mm = np.column_stack([cols["x_mm"], cols["y_mm"], cols["z_mm"]])
    return t, xyz_mm


def load_camera(input_dir: Path):
    cols = _load_csv_columns(input_dir / "camera_trajectory.csv",
                             ["timestamp_monotonic", "x_m", "y_m", "z_m"])
    t = cols["timestamp_monotonic"]
    xyz_m = np.column_stack([cols["x_m"], cols["y_m"], cols["z_m"]])
    return t, xyz_m


# ── calibration YAML (minimal parse for R + offset; targeted write) ──────────
def load_calibration(path: Path):
    """Return (R_current 3x3, offset_mm 3-vec) using the project's loader."""
    from fisheye_gantry_tagslam import load_fisheye_calibration
    calib = load_fisheye_calibration(path)
    R = np.asarray(getattr(calib, "R_gantry_to_slam", np.eye(3)), dtype=np.float64)
    if R.shape != (3, 3):
        R = np.eye(3)
    offset = getattr(calib, "gantry_anchor_offset_mm", None)
    offset_mm = (np.asarray(offset, dtype=np.float64).reshape(3)
                 if offset is not None else np.zeros(3))
    return R, offset_mm


def _format_R_block(R: np.ndarray) -> list[str]:
    """3x3 matrix -> YAML block-list lines matching the existing file style."""
    lines: list[str] = []
    for i in range(3):
        lines.append(f"- - {float(R[i, 0]):.12g}")
        lines.append(f"  - {float(R[i, 1]):.12g}")
        lines.append(f"  - {float(R[i, 2]):.12g}")
    return lines


def write_refined_R(path: Path, R_refined: np.ndarray, source: str) -> Path:
    """Replace ONLY the R_gantry_to_slam block in the YAML (preserve everything
    else) and upsert metadata.R_refined_at. Returns the backup path."""
    backup = path.with_name(
        f"{path.stem}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{path.suffix}")
    shutil.copy2(path, backup)

    lines = path.read_text(encoding="utf-8").splitlines()

    # 1) Replace the R_gantry_to_slam block.
    try:
        hdr = next(i for i, ln in enumerate(lines)
                   if ln.rstrip() == "R_gantry_to_slam:")
    except StopIteration:
        raise SystemExit("error: 'R_gantry_to_slam:' key not found in calibration")
    end = hdr + 1
    while end < len(lines):
        s = lines[end]
        if s.strip() == "" or s[0] not in " \t-":   # blank or next top-level key
            break
        end += 1
    lines[hdr + 1:end] = _format_R_block(R_refined)

    # 2) Upsert metadata.R_refined_at (2-space indented under 'metadata:').
    stamp = f"'{datetime.now().isoformat(timespec='seconds')} (src={source})'"
    try:
        midx = next(i for i, ln in enumerate(lines) if ln.rstrip() == "metadata:")
        # search the metadata block for an existing R_refined_at
        j = midx + 1
        replaced = False
        while j < len(lines) and (lines[j].startswith(("  ", "\t")) or lines[j].strip() == ""):
            if lines[j].lstrip().startswith("R_refined_at:"):
                lines[j] = f"  R_refined_at: {stamp}"
                replaced = True
                break
            j += 1
        if not replaced:
            lines.insert(midx + 1, f"  R_refined_at: {stamp}")
    except StopIteration:
        lines.append("metadata:")
        lines.append(f"  R_refined_at: {stamp}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return backup


# ── alignment math ──────────────────────────────────────────────────────────
def interpolate_to(t_target: np.ndarray, t_src: np.ndarray,
                   xyz_src: np.ndarray) -> np.ndarray:
    """Linear-interpolate a 3-D source trajectory onto target timestamps."""
    order = np.argsort(t_src)
    t_s, xyz_s = t_src[order], xyz_src[order]
    return np.column_stack([np.interp(t_target, t_s, xyz_s[:, k]) for k in range(3)])


def procrustes_rotation(g_centered: np.ndarray, c_centered: np.ndarray) -> np.ndarray:
    """Optimal rotation R (left-multiply, column-vector convention) minimizing
    Σ||R·g_i − c_i||². For row-stacked centered clouds: H = Gᵀ·C, SVD H = U·S·Vᵀ,
    R = V·Uᵀ. (scipy.orthogonal_procrustes uses the right-multiply transpose.)"""
    H = g_centered.T @ c_centered
    U, _S, Vt = np.linalg.svd(H)
    return Vt.T @ U.T


def rotation_angle_axis(R: np.ndarray):
    """Angle (deg) and unit axis of a rotation matrix (numpy, no scipy)."""
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(axis)
    axis = axis / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
    return float(angle), axis


def _centered(a: np.ndarray) -> np.ndarray:
    return a - a.mean(axis=0)


def _rms_mm(diff_m: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(diff_m ** 2, axis=1)))) * 1000.0


def _per_axis_rms_mm(diff_m: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(diff_m ** 2, axis=0)) * 1000.0


# ── optional visualization ───────────────────────────────────────────────────
def save_check_png(path: Path, g_current_m, g_refined_m, cam_m) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    # center all three for a translation-free shape comparison
    g0, g1, c = _centered(g_current_m), _centered(g_refined_m), _centered(cam_m)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(g0[:, 0], g0[:, 1], "--", color="#999999", lw=1.4, label="gantry · current R")
    ax.plot(g1[:, 0], g1[:, 1], "-", color="#ff9800", lw=1.8, label="gantry · refined R")
    ax.plot(c[:, 0], c[:, 1], "-", color="#2ecc71", lw=1.8, label="camera (SLAM)")
    ax.set_aspect("equal", "datalim")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("R_gantry_to_slam refinement check (centroid-aligned, top-down)")
    ax.legend(loc="best", fontsize=9); ax.grid(True, alpha=0.3)
    try:
        fig.tight_layout(); fig.savefig(path, dpi=130)
    finally:
        plt.close(fig)
    return True


# ── report ────────────────────────────────────────────────────────────────────
def _fmt_R(R: np.ndarray, indent: str = "  ") -> str:
    rows = [", ".join(f"{v:7.3f}" for v in R[i]) for i in range(3)]
    return (f"{indent}[[{rows[0]}],\n"
            f"{indent} [{rows[1]}],\n"
            f"{indent} [{rows[2]}]]")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, required=True,
                    help="Recording folder (gantry_telemetry.csv + camera_trajectory.csv).")
    ap.add_argument("--calib", type=Path, default=DEFAULT_CALIB,
                    help="Calibration YAML to read (and write with --write).")
    ap.add_argument("--write", action="store_true",
                    help="Write the refined R back into the calibration YAML (in place).")
    ap.add_argument("--min-displacement-mm", type=float, default=50.0,
                    help="Skip early frames until the gantry has moved this far (warmup).")
    ap.add_argument("--rms-threshold-mm", type=float, default=5.0,
                    help="Only write if RMS improves by more than this.")
    ap.add_argument("--max-angle-deg", type=float, default=15.0,
                    help="Refuse to apply if the refinement rotation exceeds this.")
    ap.add_argument("--no-plot", action="store_true",
                    help="Skip the R_refinement_check.png visualization.")
    args = ap.parse_args(argv)

    input_dir = args.input_dir
    if not input_dir.is_dir():
        print(f"error: input directory not found: {input_dir}", file=sys.stderr)
        return 2
    calib_path = args.calib if args.calib.is_absolute() else (_REPO_ROOT / args.calib)
    if not calib_path.exists():
        print(f"error: calibration not found: {calib_path}", file=sys.stderr)
        return 2

    # 1) Load inputs.
    t_g, gantry_mm_all = load_gantry(input_dir)
    t_c, cam_m_all = load_camera(input_dir)
    R_current, offset_mm = load_calibration(calib_path)
    if t_c.size < 2 or t_g.size < 2:
        print("error: not enough samples in the recording", file=sys.stderr)
        return 2

    # 2) Interpolate gantry onto camera sample times, then drop the warmup window
    #    (early frames where the gantry has barely moved → rotation ill-constrained).
    gantry_mm = interpolate_to(t_c, t_g, gantry_mm_all)
    disp = np.linalg.norm(gantry_mm - gantry_mm[0], axis=1)
    keep = disp >= float(args.min_displacement_mm)
    if not np.any(keep):
        keep = np.ones(len(t_c), dtype=bool)  # gantry never moved enough; keep all
    cam_m = cam_m_all[keep]
    gantry_mm = gantry_mm[keep]
    n_used = int(keep.sum())

    path_traveled_m = float(np.sum(np.linalg.norm(np.diff(gantry_mm, axis=0), axis=1))) / 1000.0

    # Safety: enough data + enough motion.
    if n_used < 100:
        print(f"error: only {n_used} samples after warmup filter (< 100) — "
              "record a longer motion or lower --min-displacement-mm", file=sys.stderr)
        return 3
    if path_traveled_m < 0.200:
        print(f"⚠ WARNING: gantry path traveled only {path_traveled_m*1000:.0f} mm "
              "(< 200 mm) — rotation is weakly constrained; result may be unreliable.",
              file=sys.stderr)

    # 3) Gantry → current SLAM frame estimate (meters).
    g_m = (gantry_mm - offset_mm) / 1000.0
    g_current = (R_current @ g_m.T).T

    # 4) Procrustes on centered clouds.
    R_residual = procrustes_rotation(_centered(g_current), _centered(cam_m))

    # 6a) Reflection / orthogonality sanity.
    det = float(np.linalg.det(R_residual))
    if det < 0:
        print(f"error: Procrustes produced a reflection (det={det:+.3f}); refusing. "
              "Likely bad/degenerate data or a wrong starting R.", file=sys.stderr)
        return 4

    # 5) Compose: apply current R first, residual on top.
    R_refined = R_residual @ R_current

    ortho_err = float(np.max(np.abs(R_refined @ R_refined.T - np.eye(3))))
    det_refined = float(np.linalg.det(R_refined))

    # 6b) Angular change vs current R.
    angle_deg, axis = rotation_angle_axis(R_refined @ R_current.T)

    # 6c) RMS before/after (translation-free shape error, in mm).
    g_refined = (R_refined @ g_m.T).T
    diff_before = _centered(g_current) - _centered(cam_m)
    diff_after = _centered(g_refined) - _centered(cam_m)
    rms_before, rms_after = _rms_mm(diff_before), _rms_mm(diff_after)
    pax_before, pax_after = _per_axis_rms_mm(diff_before), _per_axis_rms_mm(diff_after)
    improvement_mm = rms_before - rms_after
    pct = (-100.0 * improvement_mm / rms_before) if rms_before > 1e-9 else 0.0

    meaningful = improvement_mm > float(args.rms_threshold_mm)
    within_angle = angle_deg <= float(args.max_angle_deg)
    proper = det_refined > 0 and ortho_err < 1e-6

    # 7) Report.
    line = "=" * 76
    print(line)
    print("R_gantry_to_slam Refinement Report")
    print(line)
    print(f"Source recording:  {input_dir}")
    print(f"Duration:          {t_c[-1] - t_c[0]:.1f} s")
    print(f"Gantry samples:    {t_g.size}")
    print(f"Camera samples:    {t_c.size} (used {n_used} after warmup filter)")
    print(f"Path traveled:     {path_traveled_m:.2f} m")
    print()
    print(f"Current R (from {calib_path.relative_to(_REPO_ROOT) if calib_path.is_relative_to(_REPO_ROOT) else calib_path}):")
    print(_fmt_R(R_current))
    print()
    print("Refined R (Procrustes alignment):")
    print(_fmt_R(R_refined))
    print()
    print(f"Refinement rotation: {angle_deg:.1f}° around axis "
          f"[{axis[0]:.2f}, {axis[1]:.2f}, {axis[2]:.2f}]")
    print()
    print("Trajectory RMS error:")
    print(f"  Before refinement: {rms_before:7.1f} mm")
    print(f"  After refinement:  {rms_after:7.1f} mm    ({pct:+.1f}%)")
    print()
    print("Per-axis RMS:")
    print("               X        Y        Z")
    print(f"  Before:  {pax_before[0]:6.1f} mm {pax_before[1]:6.1f} mm {pax_before[2]:6.1f} mm")
    print(f"  After:   {pax_after[0]:6.1f} mm {pax_after[1]:6.1f} mm {pax_after[2]:6.1f} mm")
    print()
    print(f"{'✓' if meaningful else '✗'} Refinement is "
          f"{'meaningful' if meaningful else 'NOT meaningful'} "
          f"(>{args.rms_threshold_mm:.0f}mm improvement: {improvement_mm:+.1f} mm)")
    print(f"{'✓' if within_angle else '✗'} Angular change "
          f"{'within' if within_angle else 'EXCEEDS'} safety bound "
          f"({angle_deg:.1f}° vs {args.max_angle_deg:.0f}°)")
    print(f"{'✓' if proper else '✗'} Result is a proper rotation "
          f"(det={det_refined:+.3f}, orthogonality err={ortho_err:.1e})")
    print()

    if not args.no_plot:
        png = input_dir / "R_refinement_check.png"
        if save_check_png(png, g_current, g_refined, cam_m):
            print(f"Saved visualization: {png}")

    # Decide whether to write.
    can_write = meaningful and within_angle and proper and rms_after < rms_before
    if not args.write:
        if can_write:
            print("Recommended: rerun with --write to apply.")
        else:
            print("Not recommended to write (see flags above) — already near-optimal "
                  "or refinement rejected.")
        print(line)
        return 0

    if not within_angle:
        print(f"✗ ABORT: refinement {angle_deg:.1f}° exceeds --max-angle-deg "
              f"{args.max_angle_deg:.0f}°. Not writing.")
        print(line)
        return 4
    if not proper:
        print("✗ ABORT: refined R is not a proper rotation. Not writing.")
        print(line)
        return 4
    if rms_after >= rms_before or not meaningful:
        print(f"✓ Already optimal — RMS improvement {improvement_mm:+.1f} mm does not "
              f"exceed --rms-threshold-mm {args.rms_threshold_mm:.0f}. No write.")
        print(line)
        return 0

    backup = write_refined_R(calib_path, R_refined, str(input_dir))
    print(f"✓ Updated {calib_path}")
    print(f"   Backup saved: {backup}")
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
