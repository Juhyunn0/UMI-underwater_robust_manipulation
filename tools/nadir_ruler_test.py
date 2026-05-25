#!/usr/bin/env python3
"""
Near-nadir raw AprilTag ruler test for the underwater ZED2 setup.

This is a read-only diagnostic: it opens the ZED source, detects one target tag,
and reports raw cv2.solvePnP distances before water scale, GTSAM, priors, or
trajectory fusion. The helper imports come from zed2_underwater_tagslam.py so
the camera open path, intrinsics, detector settings, and object-point convention
match the SLAM front-end.

README / examples
-----------------
Live camera, compare the current config tag size against a half-size test:
    python3 tools/nadir_ruler_test.py --target-tag-id 1 --tag-size 0.085 --tag-size 0.170 --true-distance-m 2.60

Recorded SVO, headless:
    python3 tools/nadir_ruler_test.py --svo data/example.svo --target-tag-id 1 --tag-size 0.170 --samples 200 --no-window --out-csv data/nadir_ruler_test.csv

How to read R:
    R = median(raw solvePnP distance) / tape-measured true distance.
    If the physical tag size is correct and the camera is near-nadir through a
    flat air-water interface, pure refraction often compresses raw distance
    toward roughly R ~= 0.77. A much different value points to tag-size,
    intrinsics, aiming angle, or non-flat water-surface effects.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
import time
import types

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def install_gtsam_import_stub() -> None:
    """Let this diagnostic import front-end helpers when GTSAM is not installed."""
    unavailable_message = (
        "GTSAM is not installed. The near-nadir ruler test does not use GTSAM, "
        "but TagSLAM backend classes are unavailable from this import stub."
    )

    class UnavailableGtsamObject:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(unavailable_message)

    def unavailable_function(*args, **kwargs):
        raise RuntimeError(unavailable_message)

    gtsam_stub = types.ModuleType("gtsam")
    for name in (
        "BetweenFactorPose3",
        "NonlinearFactorGraph",
        "Pose3",
        "PriorFactorPose3",
        "Point3",
        "Rot3",
        "Values",
    ):
        setattr(gtsam_stub, name, UnavailableGtsamObject)

    gtsam_stub.noiseModel = types.SimpleNamespace(
        Base=UnavailableGtsamObject,
        Diagonal=types.SimpleNamespace(Sigmas=unavailable_function),
        Robust=types.SimpleNamespace(Create=unavailable_function),
        mEstimator=types.SimpleNamespace(
            Huber=types.SimpleNamespace(Create=unavailable_function),
            Cauchy=types.SimpleNamespace(Create=unavailable_function),
            Tukey=types.SimpleNamespace(Create=unavailable_function),
        ),
    )
    symbol_stub = types.ModuleType("gtsam.symbol_shorthand")
    symbol_stub.L = unavailable_function
    symbol_stub.X = unavailable_function
    sys.modules["gtsam"] = gtsam_stub
    sys.modules["gtsam.symbol_shorthand"] = symbol_stub


def import_tagslam_module():
    try:
        return importlib.import_module("zed2_underwater_tagslam")
    except ModuleNotFoundError as exc:
        if exc.name != "gtsam":
            raise
        install_gtsam_import_stub()
        return importlib.import_module("zed2_underwater_tagslam")


tagslam = None


WINDOW_NAME = "Near-Nadir Raw PnP Ruler Test"


@dataclass(frozen=True)
class RawPnPSample:
    frame_index: int
    elapsed_s: float
    tag_size_m: float
    x_m: float
    y_m: float
    z_m: float
    norm_m: float
    off_nadir_deg: float
    lateral_ratio: float
    decision_margin: float
    hamming: int
    tag_area_px: float
    reprojection_error_px: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure raw single-tag solvePnP distance for a near-nadir ZED2 "
            "underwater ruler test. No water scale and no GTSAM are applied."
        )
    )
    parser.add_argument("--config", default="config/config.yaml", help="Runtime config used by TagSLAM.")
    parser.add_argument("--list", action="store_true", help="List connected ZED cameras and exit.")
    parser.add_argument("--camera-id", type=int, help="Open a specific local ZED camera ID.")
    parser.add_argument("--serial", type=int, help="Open a specific ZED serial number.")
    parser.add_argument("--svo", help="Read a ZED SVO file instead of a live camera.")
    parser.add_argument("--stream", help="Read a ZED network stream as IP or IP:PORT.")
    parser.add_argument(
        "--resolution",
        default="HD720",
        choices=["HD2K", "HD1200", "HD1080", "HD720", "SVGA", "VGA"],
        help="ZED camera resolution.",
    )
    parser.add_argument("--fps", type=int, default=30, help="ZED camera FPS.")
    parser.add_argument("--tag-family", default="tag36h11", help="AprilTag family.")
    parser.add_argument(
        "--target-tag-id",
        type=int,
        help="AprilTag ID placed directly under the camera for the ruler test.",
    )
    parser.add_argument(
        "--tag-size",
        type=float,
        action="append",
        default=None,
        help=(
            "Physical AprilTag edge length in meters. Repeat this flag to solve "
            "the same image with multiple tag sizes."
        ),
    )
    parser.add_argument("--samples", type=int, default=200, help="Target detections to collect.")
    parser.add_argument(
        "--duration-s",
        type=float,
        help="Optional wall-clock collection duration after the target tag is first seen.",
    )
    parser.add_argument(
        "--true-distance-m",
        type=float,
        help="Optional tape-measured lens-to-tag straight-line distance.",
    )
    parser.add_argument("--out-csv", type=Path, help="Optional CSV path for every raw sample.")
    parser.add_argument("--no-window", action="store_true", help="Run headless.")
    parser.add_argument("--display-width", type=int, default=1280, help="Maximum preview width; 0 is native.")
    parser.add_argument("--nthreads", type=int, default=2, help="pupil-apriltags worker threads.")
    parser.add_argument("--quad-decimate", type=float, default=1.0, help="AprilTag quad decimation.")
    parser.add_argument("--quad-sigma", type=float, default=0.0, help="Gaussian blur sigma for segmentation.")
    parser.add_argument(
        "--decode-sharpening",
        type=float,
        default=0.25,
        help="AprilTag decode sharpening parameter.",
    )
    parser.add_argument(
        "--warn-off-nadir-deg",
        type=float,
        default=5.0,
        help="Warn when median off-nadir angle is above this value.",
    )

    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if not args.list and args.target_tag_id is None:
        parser.error("--target-tag-id is required unless --list is used")
    if args.target_tag_id is not None and args.target_tag_id < 0:
        parser.error("--target-tag-id must be >= 0")
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if args.duration_s is not None and args.duration_s <= 0:
        parser.error("--duration-s must be positive")
    if args.true_distance_m is not None and args.true_distance_m <= 0:
        parser.error("--true-distance-m must be positive")
    if args.display_width < 0:
        parser.error("--display-width must be >= 0")
    if args.nthreads <= 0:
        parser.error("--nthreads must be positive")
    if args.quad_decimate <= 0:
        parser.error("--quad-decimate must be positive")
    if args.warn_off_nadir_deg <= 0:
        parser.error("--warn-off-nadir-deg must be positive")

    selected_inputs = sum(
        value is not None for value in (args.camera_id, args.serial, args.svo, args.stream)
    )
    if selected_inputs > 1:
        parser.error("choose only one of --camera-id, --serial, --svo, or --stream")
    return args


def resolve_requested_tag_sizes(args: argparse.Namespace) -> tuple[list[float], str]:
    if args.tag_size:
        sizes = [float(size) for size in args.tag_size]
        source = "cli"
    else:
        runtime_config = tagslam.load_runtime_config(args.config)
        size, source = tagslam.resolve_tag_size_m(runtime_config, None)
        sizes = [float(size)]

    for size in sizes:
        if not np.isfinite(size) or size <= 0:
            raise ValueError(f"tag size must be positive and finite, got {size}")
    return sizes, source


def solve_raw_pnp(
    detection,
    intrinsics: tagslam.CameraIntrinsics,
    object_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    image_points = np.asarray(detection.corners, dtype=np.float32).reshape(4, 2)
    method = (
        cv2.SOLVEPNP_IPPE_SQUARE
        if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE")
        else cv2.SOLVEPNP_ITERATIVE
    )
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
        flags=method,
    )
    if not success:
        return None
    raw_tvec_m = np.asarray(tvec, dtype=np.float64).reshape(3)
    reproj_error = tagslam.reprojection_error_px(
        object_points,
        image_points,
        rvec,
        raw_tvec_m,
        intrinsics,
    )
    return np.asarray(rvec, dtype=np.float64).reshape(3), raw_tvec_m, reproj_error


def choose_target_detection(detections, target_tag_id: int):
    matches = [detection for detection in detections if int(detection.tag_id) == target_tag_id]
    if not matches:
        return None
    return max(matches, key=lambda detection: float(detection.decision_margin))


def sample_from_detection(
    frame_index: int,
    elapsed_s: float,
    tag_size_m: float,
    detection,
    intrinsics: tagslam.CameraIntrinsics,
) -> RawPnPSample | None:
    object_points = tagslam.tag_object_points(tag_size_m)
    solved = solve_raw_pnp(detection, intrinsics, object_points)
    if solved is None:
        return None

    _rvec, raw_tvec_m, reproj_error = solved
    norm_m = float(np.linalg.norm(raw_tvec_m))
    lateral_m = float(np.hypot(raw_tvec_m[0], raw_tvec_m[1]))
    lateral_ratio = lateral_m / max(norm_m, 1e-9)
    image_points = np.asarray(detection.corners, dtype=np.float32).reshape(4, 2)
    return RawPnPSample(
        frame_index=frame_index,
        elapsed_s=elapsed_s,
        tag_size_m=float(tag_size_m),
        x_m=float(raw_tvec_m[0]),
        y_m=float(raw_tvec_m[1]),
        z_m=float(raw_tvec_m[2]),
        norm_m=norm_m,
        off_nadir_deg=tagslam.off_nadir_angle_deg(raw_tvec_m),
        lateral_ratio=float(lateral_ratio),
        decision_margin=float(detection.decision_margin),
        hamming=int(detection.hamming),
        tag_area_px=tagslam.quadrilateral_area_px(image_points),
        reprojection_error_px=float(reproj_error),
    )


def draw_preview(
    frame: np.ndarray,
    detections,
    target_detection,
    args: argparse.Namespace,
    used_frames: int,
    skipped_frames: int,
    latest_samples: list[RawPnPSample],
) -> None:
    for detection in detections:
        corners = np.round(np.asarray(detection.corners, dtype=np.float32)).astype(np.int32)
        is_target = target_detection is not None and detection is target_detection
        color = (0, 255, 0) if is_target else (150, 150, 150)
        thickness = 3 if is_target else 1
        cv2.polylines(frame, [corners], isClosed=True, color=color, thickness=thickness)
        center = tuple(np.round(np.asarray(detection.center, dtype=np.float64)).astype(int).tolist())
        cv2.drawMarker(
            frame,
            center,
            (0, 255, 255) if is_target else (120, 120, 120),
            markerType=cv2.MARKER_CROSS,
            markerSize=14 if is_target else 8,
            thickness=2 if is_target else 1,
            line_type=cv2.LINE_AA,
        )
        if is_target:
            tagslam.draw_text_box(
                frame,
                f"TARGET ID {args.target_tag_id}",
                tuple((corners[0] + np.array([0, -8])).tolist()),
                scale=0.55,
                bg_color=(0, 90, 0),
                thickness=2,
                padding=4,
            )

    state = "DETECTED" if target_detection is not None else "NOT FOUND"
    latest_text = "no raw sample yet"
    if latest_samples:
        latest = latest_samples[-1]
        latest_text = (
            f"size {latest.tag_size_m:.3f}m z={latest.z_m:.3f}m "
            f"norm={latest.norm_m:.3f}m off={latest.off_nadir_deg:.1f}deg"
        )
    tagslam.draw_text_box(
        frame,
        (
            f"Target {args.target_tag_id}: {state} | used {used_frames}/{args.samples} | "
            f"skipped {skipped_frames}"
        ),
        (12, 32),
        scale=0.62,
        bg_color=(15, 20, 26),
    )
    tagslam.draw_text_box(
        frame,
        latest_text,
        (12, 64),
        scale=0.55,
        text_color=(210, 235, 255),
        bg_color=(15, 20, 26),
        thickness=2,
        padding=4,
    )
    tagslam.draw_text_box(
        frame,
        "Aim near-nadir. Press Q or ESC to finish.",
        (12, 94),
        scale=0.55,
        text_color=(230, 230, 230),
        bg_color=(15, 20, 26),
        thickness=1,
        padding=4,
    )


def values_for(samples: list[RawPnPSample], attr: str) -> np.ndarray:
    return np.asarray([float(getattr(sample, attr)) for sample in samples], dtype=np.float64)


def stat_dict(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def fmt(row: list[str]) -> str:
        return "  ".join(cell.rjust(widths[index]) for index, cell in enumerate(row))

    lines = [fmt(headers), fmt(["-" * width for width in widths])]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


def summarize(
    samples: list[RawPnPSample],
    tag_sizes_m: list[float],
    true_distance_m: float | None,
    warn_off_nadir_deg: float,
    used_frames: int,
    skipped_frames: int,
    solve_failed_frames: int,
) -> int:
    if not samples:
        print("No valid raw PnP samples collected.", file=sys.stderr)
        return 1

    print()
    print("Raw solvePnP summary (NO water scale, NO graph, NO priors)")
    print(f"Frames used: {used_frames}; skipped/no target: {skipped_frames}; solve failed: {solve_failed_frames}")

    rows: list[list[str]] = []
    medians_by_size: dict[float, dict[str, float]] = {}
    for tag_size_m in tag_sizes_m:
        group = [sample for sample in samples if sample.tag_size_m == tag_size_m]
        if not group:
            continue
        x_stats = stat_dict(values_for(group, "x_m"))
        y_stats = stat_dict(values_for(group, "y_m"))
        z_stats = stat_dict(values_for(group, "z_m"))
        norm_stats = stat_dict(values_for(group, "norm_m"))
        off_stats = stat_dict(values_for(group, "off_nadir_deg"))
        lateral_stats = stat_dict(values_for(group, "lateral_ratio"))
        medians_by_size[tag_size_m] = {
            "z": z_stats["median"],
            "norm": norm_stats["median"],
        }
        r_norm = ""
        r_z = ""
        if true_distance_m is not None:
            r_norm = f"{norm_stats['median'] / true_distance_m:.4f}"
            r_z = f"{z_stats['median'] / true_distance_m:.4f}"

        rows.append(
            [
                f"{tag_size_m:.4f}",
                str(len(group)),
                f"{x_stats['median']:+.4f}",
                f"{y_stats['median']:+.4f}",
                f"{z_stats['mean']:+.4f}",
                f"{z_stats['median']:+.4f}",
                f"{z_stats['std']:.4f}",
                f"{z_stats['min']:+.4f}",
                f"{z_stats['max']:+.4f}",
                f"{norm_stats['mean']:.4f}",
                f"{norm_stats['median']:.4f}",
                f"{norm_stats['std']:.4f}",
                f"{norm_stats['min']:.4f}",
                f"{norm_stats['max']:.4f}",
                f"{off_stats['median']:.2f}",
                f"{lateral_stats['median']:.4f}",
                r_norm,
                r_z,
            ]
        )

        if off_stats["median"] > warn_off_nadir_deg:
            print(
                f"WARNING: tag_size={tag_size_m:.4f} median off-nadir "
                f"{off_stats['median']:.2f} deg exceeds {warn_off_nadir_deg:.2f} deg."
            )
        elif off_stats["max"] > warn_off_nadir_deg:
            print(
                f"WARNING: tag_size={tag_size_m:.4f} max off-nadir "
                f"{off_stats['max']:.2f} deg exceeds {warn_off_nadir_deg:.2f} deg."
            )

    headers = [
        "size_m",
        "n",
        "med_x",
        "med_y",
        "mean_z",
        "med_z",
        "std_z",
        "min_z",
        "max_z",
        "mean_norm",
        "med_norm",
        "std_norm",
        "min_norm",
        "max_norm",
        "med_off_deg",
        "med_lat",
        "R_norm",
        "R_z",
    ]
    print(format_table(headers, rows))

    if len(medians_by_size) >= 2:
        print()
        print("Empirical distance ratios between tag-size runs")
        ratio_rows: list[list[str]] = []
        sizes_with_data = [size for size in tag_sizes_m if size in medians_by_size]
        for i, size_a in enumerate(sizes_with_data):
            for size_b in sizes_with_data[i + 1 :]:
                expected = size_a / size_b
                observed_norm = medians_by_size[size_a]["norm"] / medians_by_size[size_b]["norm"]
                observed_z = medians_by_size[size_a]["z"] / medians_by_size[size_b]["z"]
                ratio_rows.append(
                    [
                        f"{size_a:.4f}/{size_b:.4f}",
                        f"{expected:.4f}",
                        f"{observed_norm:.4f}",
                        f"{observed_z:.4f}",
                    ]
                )
        print(format_table(["sizes", "expected", "norm_ratio", "z_ratio"], ratio_rows))

    if true_distance_m is not None:
        print()
        print(
            "R is raw/true. Near-nadir with correct tag size and flat water often trends "
            "toward roughly R ~= 0.77 from refraction alone."
        )
    return 0


def write_csv(path: Path, samples: list[RawPnPSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "frame_index",
        "elapsed_s",
        "tag_size_m",
        "x_m",
        "y_m",
        "z_m",
        "norm_m",
        "off_nadir_deg",
        "lateral_ratio",
        "decision_margin",
        "hamming",
        "tag_area_px",
        "reprojection_error_px",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            writer.writerow({field: getattr(sample, field) for field in fields})
    print(f"Wrote raw samples: {path}")


def main() -> int:
    global tagslam

    args = parse_args()

    try:
        tagslam = import_tagslam_module()
    except ModuleNotFoundError as exc:
        print(f"nadir ruler test failed: missing dependency {exc.name!r}", file=sys.stderr)
        return 1

    if args.list:
        tagslam.print_zed_cameras()
        return 0

    try:
        tag_sizes_m, size_source = resolve_requested_tag_sizes(args)
    except Exception as exc:
        print(f"nadir ruler test failed: {exc}", file=sys.stderr)
        return 1

    print(
        "Tag sizes to test: "
        + ", ".join(f"{size:.4f} m" for size in tag_sizes_m)
        + f" (source: {size_source})"
    )
    pnp_method = "SOLVEPNP_IPPE_SQUARE" if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE") else "SOLVEPNP_ITERATIVE"
    print(f"Raw PnP method: {pnp_method}; water scale is not applied.")

    zed = None
    try:
        zed = tagslam.open_zed(args)
        runtime = tagslam.sl.RuntimeParameters()
        image_mat = tagslam.sl.Mat()
        detector = tagslam.make_detector(args)

        if not args.no_window:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

        intrinsics: tagslam.CameraIntrinsics | None = None
        samples: list[RawPnPSample] = []
        latest_frame_samples: list[RawPnPSample] = []
        frame_index = 0
        used_frames = 0
        skipped_frames = 0
        solve_failed_frames = 0
        collection_start_s: float | None = None
        last_progress_s = 0.0

        while True:
            error = zed.grab(runtime)
            if error == tagslam.sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                print("Reached end of SVO.")
                break
            if error != tagslam.sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(image_mat, tagslam.sl.VIEW.LEFT_BGR, tagslam.sl.MEM.CPU)
            frame = tagslam.bgr_from_zed_mat(image_mat)
            if intrinsics is None:
                intrinsics = tagslam.get_left_intrinsics(zed, frame.shape)
                print(
                    "Left camera intrinsics: "
                    f"fx={intrinsics.camera_matrix[0, 0]:.2f}, "
                    f"fy={intrinsics.camera_matrix[1, 1]:.2f}, "
                    f"cx={intrinsics.camera_matrix[0, 2]:.2f}, "
                    f"cy={intrinsics.camera_matrix[1, 2]:.2f}"
                )

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = detector.detect(gray)
            target_detection = choose_target_detection(detections, args.target_tag_id)
            now_s = time.monotonic()

            if target_detection is None:
                skipped_frames += 1
            else:
                if collection_start_s is None:
                    collection_start_s = now_s
                    print(f"Target tag {args.target_tag_id} first detected; starting collection.")
                elapsed_s = now_s - collection_start_s
                frame_samples: list[RawPnPSample] = []
                for tag_size_m in tag_sizes_m:
                    sample = sample_from_detection(
                        frame_index,
                        elapsed_s,
                        tag_size_m,
                        target_detection,
                        intrinsics,
                    )
                    if sample is not None:
                        frame_samples.append(sample)

                if frame_samples:
                    samples.extend(frame_samples)
                    latest_frame_samples = frame_samples
                    used_frames += 1
                else:
                    solve_failed_frames += 1

            if now_s - last_progress_s > 1.0:
                last_progress_s = now_s
                if latest_frame_samples:
                    latest = latest_frame_samples[-1]
                    print(
                        f"used={used_frames}/{args.samples} skipped={skipped_frames} "
                        f"latest size={latest.tag_size_m:.4f} z={latest.z_m:.3f}m "
                        f"norm={latest.norm_m:.3f}m off={latest.off_nadir_deg:.2f}deg",
                        flush=True,
                    )
                else:
                    print(
                        f"used={used_frames}/{args.samples} skipped={skipped_frames}; "
                        f"target {args.target_tag_id} not sampled yet",
                        flush=True,
                    )

            if not args.no_window:
                draw_preview(
                    frame,
                    detections,
                    target_detection,
                    args,
                    used_frames,
                    skipped_frames,
                    latest_frame_samples,
                )
                display_scale = tagslam.get_display_scale(frame.shape, args.display_width)
                cv2.imshow(WINDOW_NAME, tagslam.resize_for_display(frame, display_scale))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            if collection_start_s is not None:
                elapsed_collection_s = now_s - collection_start_s
                if used_frames >= args.samples:
                    break
                if args.duration_s is not None and elapsed_collection_s >= args.duration_s:
                    break

            frame_index += 1

        if args.out_csv is not None and samples:
            write_csv(args.out_csv, samples)

        return summarize(
            samples,
            tag_sizes_m,
            args.true_distance_m,
            args.warn_off_nadir_deg,
            used_frames,
            skipped_frames,
            solve_failed_frames,
        )

    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    except Exception as exc:
        print(f"nadir ruler test failed: {exc}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()
        if zed is not None:
            zed.close()


if __name__ == "__main__":
    raise SystemExit(main())
