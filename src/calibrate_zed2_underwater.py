#!/usr/bin/env python3
"""
ZED2 left-camera underwater intrinsic calibration with a standard checkerboard.

The camera should be inside its underwater housing and submerged during capture.
Output is written as a YAML file that can be consumed by the TagSLAM pipeline.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys

import cv2
import numpy as np

try:
    import pyzed.sl as sl
except ImportError:  # pragma: no cover - handled in main for a clear message.
    sl = None


WINDOW_NAME = "ZED2 Underwater Checkerboard Calibration"
DEFAULT_OUTPUT_PATH = Path("config/zed2_underwater_calib.yaml")
DEFAULT_DATA_ROOT = Path("data/calib")
MIN_CALIBRATION_FRAMES = 10


@dataclass(frozen=True)
class CapturedFrame:
    image_path: Path
    corners_px: np.ndarray


@dataclass(frozen=True)
class CalibrationResult:
    rms_px: float
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rvecs: tuple[np.ndarray, ...]
    tvecs: tuple[np.ndarray, ...]
    per_image_errors_px: list[tuple[Path, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect underwater checkerboard images from the ZED2 left camera "
            "and calibrate OpenCV pinhole intrinsics."
        )
    )
    parser.add_argument("--list", action="store_true", help="List connected ZED cameras and exit.")
    parser.add_argument("--camera-id", type=int, help="Open a specific local ZED camera ID.")
    parser.add_argument("--serial", type=int, help="Open a specific ZED serial number.")
    parser.add_argument(
        "--resolution",
        default="HD720",
        choices=["HD2K", "HD1200", "HD1080", "HD720", "SVGA", "VGA"],
        help="ZED camera resolution.",
    )
    parser.add_argument("--fps", type=int, default=30, help="ZED camera FPS.")
    parser.add_argument("--pattern-cols", type=int, default=6, help="Checkerboard inner corners horizontally.")
    parser.add_argument("--pattern-rows", type=int, default=5, help="Checkerboard inner corners vertically.")
    parser.add_argument("--square-size", type=float, default=0.065, help="Checkerboard square size in meters.")
    parser.add_argument(
        "--target-frames",
        type=int,
        default=50,
        help="Suggested number of saved checkerboard views before calibration.",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=1280,
        help="Maximum display width; 0 displays native width.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="YAML output path.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory for saved calibration image runs.",
    )
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.pattern_cols <= 1 or args.pattern_rows <= 1:
        parser.error("--pattern-cols and --pattern-rows must both be > 1")
    if args.square_size <= 0:
        parser.error("--square-size must be positive")
    if args.target_frames < MIN_CALIBRATION_FRAMES:
        parser.error(f"--target-frames must be >= {MIN_CALIBRATION_FRAMES}")
    if args.display_width < 0:
        parser.error("--display-width must be >= 0")
    return args


def make_init_parameters(args: argparse.Namespace) -> sl.InitParameters:
    params = sl.InitParameters()
    params.camera_resolution = getattr(sl.RESOLUTION, args.resolution)
    params.camera_fps = args.fps
    params.coordinate_units = sl.UNIT.METER
    params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE

    if args.camera_id is not None:
        params.set_from_camera_id(args.camera_id)
    elif args.serial is not None:
        params.set_from_serial_number(args.serial)

    return params


def open_zed(args: argparse.Namespace) -> sl.Camera:
    zed = sl.Camera()
    error = zed.open(make_init_parameters(args))
    if error != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not open ZED camera: {error}")
    return zed


def print_zed_cameras() -> None:
    print(f"ZED SDK version: {sl.Camera().get_sdk_version()}")
    devices = sl.Camera.get_device_list()
    if not devices:
        print("ZED cameras: none found")
        return
    for device in devices:
        print(
            "ZED "
            f"id={device.id} model={device.camera_model} "
            f"serial={device.serial_number} state={device.camera_state} path={device.path}"
        )


def bgr_from_zed_mat(image_mat: sl.Mat) -> np.ndarray:
    image = image_mat.get_data()
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image.copy()
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    raise RuntimeError(f"Unsupported ZED image shape: {image.shape}")


def get_display_scale(frame_shape: tuple[int, int, int], max_width: int) -> float:
    width = frame_shape[1]
    if max_width == 0 or width <= max_width:
        return 1.0
    return max_width / width


def resize_for_display(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return frame
    width = max(1, int(frame.shape[1] * scale))
    height = max(1, int(frame.shape[0] * scale))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def create_object_points(pattern_cols: int, pattern_rows: int, square_size_m: float) -> np.ndarray:
    object_points = np.zeros((pattern_cols * pattern_rows, 3), np.float32)
    grid = np.mgrid[0:pattern_cols, 0:pattern_rows].T.reshape(-1, 2)
    object_points[:, :2] = grid.astype(np.float32) * np.float32(square_size_m)
    return object_points


def draw_text_panel(image: np.ndarray, lines: list[str], origin: tuple[int, int]) -> None:
    x, y = origin
    line_height = 22
    width = 0
    for line in lines:
        (text_width, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        width = max(width, text_width)
    height = line_height * len(lines) + 12

    x2 = min(image.shape[1] - 1, x + width + 16)
    y2 = min(image.shape[0] - 1, y + height)
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), (18, 24, 31), thickness=-1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0.0, dst=image)
    cv2.rectangle(image, (x, y), (x2, y2), (180, 190, 200), thickness=1)

    text_y = y + 22
    for line in lines:
        cv2.putText(
            image,
            line,
            (x + 8, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 242, 248),
            1,
            cv2.LINE_AA,
        )
        text_y += line_height


def draw_hud(
    image: np.ndarray,
    captured_count: int,
    target_frames: int,
    detected: bool,
    last_message: str,
) -> None:
    status = "DETECTED" if detected else "NOT FOUND"
    status_color = (45, 220, 90) if detected else (70, 120, 255)
    lines = [
        f"Captured: {captured_count}/{target_frames}",
        f"Detection: {status}",
        "SPACE save   D delete last   C calibrate   Q/ESC quit",
    ]
    if last_message:
        lines.append(last_message)
    draw_text_panel(image, lines, (12, 12))

    cv2.putText(
        image,
        status,
        (22, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        status_color,
        2,
        cv2.LINE_AA,
    )

    tip_lines = [
        "Tip: vary distance (0.5 - 2.5 m), tilt left/right/up/down,",
        "and screen position (center, corners). Aim for >= 30 frames.",
    ]
    panel_height = 22 * len(tip_lines) + 12
    draw_text_panel(image, tip_lines, (12, max(12, image.shape[0] - panel_height - 12)))


def find_checkerboard_live(
    gray: np.ndarray,
    pattern_size: tuple[int, int],
) -> tuple[bool, np.ndarray | None]:
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found:
        return False, None
    return True, corners


def refine_corners(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    refined = corners.astype(np.float32, copy=True)
    cv2.cornerSubPix(gray, refined, (11, 11), (-1, -1), criteria)
    return refined


def save_capture(
    run_dir: Path,
    frame_bgr: np.ndarray,
    corners_px: np.ndarray,
    captures: list[CapturedFrame],
) -> str:
    image_path = run_dir / f"calib_{len(captures):04d}.png"
    if not cv2.imwrite(str(image_path), frame_bgr):
        return f"Failed to save {image_path}"
    captures.append(CapturedFrame(image_path=image_path, corners_px=corners_px.copy()))
    return f"Saved {image_path.name}"


def delete_last_capture(captures: list[CapturedFrame]) -> str:
    if not captures:
        return "No saved frames to delete"
    last = captures.pop()
    try:
        last.image_path.unlink(missing_ok=True)
    except OSError as exc:
        return f"Removed from list, but could not delete image: {exc}"
    return f"Deleted {last.image_path.name}"


def per_image_reprojection_errors(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_paths: list[Path],
    rvecs: tuple[np.ndarray, ...],
    tvecs: tuple[np.ndarray, ...],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> list[tuple[Path, float]]:
    errors: list[tuple[Path, float]] = []
    for objp, imgp, path, rvec, tvec in zip(
        object_points,
        image_points,
        image_paths,
        rvecs,
        tvecs,
        strict=True,
    ):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        delta = imgp.reshape(-1, 2) - projected.reshape(-1, 2)
        rms = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
        errors.append((path, rms))
    return errors


def run_calibration(
    captures: list[CapturedFrame],
    object_points_template: np.ndarray,
    image_size: tuple[int, int],
) -> CalibrationResult:
    object_points = [object_points_template.copy() for _ in captures]
    image_points = [capture.corners_px.astype(np.float32, copy=True) for capture in captures]
    image_paths = [capture.image_path for capture in captures]

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    errors = per_image_reprojection_errors(
        object_points,
        image_points,
        image_paths,
        tuple(rvecs),
        tuple(tvecs),
        camera_matrix,
        dist_coeffs,
    )
    return CalibrationResult(
        rms_px=float(rms),
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs.reshape(-1),
        rvecs=tuple(rvecs),
        tvecs=tuple(tvecs),
        per_image_errors_px=errors,
    )


def quality_assessment(rms_px: float) -> tuple[str, str]:
    if rms_px < 0.5:
        return "EXCELLENT", "< 0.5 px"
    if rms_px < 1.0:
        return "GOOD", "0.5 - 1.0 px, usable"
    if rms_px < 2.0:
        return "MARGINAL", "1.0 - 2.0 px, recommend more data"
    return "POOR", "> 2.0 px, do not use"


def save_debug_data(
    run_dir: Path,
    captures: list[CapturedFrame],
    object_points_template: np.ndarray,
    result: CalibrationResult,
    image_size: tuple[int, int],
) -> None:
    debug_dir = run_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    image_points = np.asarray([capture.corners_px for capture in captures], dtype=np.float32)
    object_points = np.asarray([object_points_template for _ in captures], dtype=np.float32)
    image_filenames = np.asarray([capture.image_path.name for capture in captures])

    np.savez_compressed(
        debug_dir / "calibration_inputs.npz",
        image_points_px=image_points,
        object_points_m=object_points,
        image_filenames=image_filenames,
        image_width=np.int32(image_size[0]),
        image_height=np.int32(image_size[1]),
        camera_matrix=result.camera_matrix,
        dist_coeffs=result.dist_coeffs,
        rms_px=np.float64(result.rms_px),
    )

    with (debug_dir / "image_filenames.txt").open("w", encoding="utf-8") as file:
        for capture in captures:
            file.write(f"{capture.image_path.name}\n")

    with (debug_dir / "per_image_reprojection_errors.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["image_filename", "rms_reprojection_error_px"])
        for path, error in result.per_image_errors_px:
            writer.writerow([path.name, f"{error:.6f}"])


def yaml_float(value: float) -> str:
    return f"{float(value):.12g}"


def write_calibration_yaml(
    output_path: Path,
    result: CalibrationResult,
    image_size: tuple[int, int],
    args: argparse.Namespace,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    camera_matrix = result.camera_matrix
    coeffs = np.zeros(5, dtype=np.float64)
    coeff_count = min(5, result.dist_coeffs.size)
    coeffs[:coeff_count] = result.dist_coeffs[:coeff_count]
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

    lines = [
        "camera_model: pinhole",
        "resolution:",
        f"  width: {image_size[0]}",
        f"  height: {image_size[1]}",
        "camera_matrix:",
        "  - ["
        + ", ".join(yaml_float(value) for value in camera_matrix[0])
        + "]",
        "  - ["
        + ", ".join(yaml_float(value) for value in camera_matrix[1])
        + "]",
        "  - ["
        + ", ".join(yaml_float(value) for value in camera_matrix[2])
        + "]",
        "dist_coeffs: ["
        + ", ".join(yaml_float(value) for value in coeffs)
        + "]",
        f"reprojection_error_px: {yaml_float(result.rms_px)}",
        "calibration:",
        "  method: opencv_calibrateCamera",
        "  pattern: checkerboard",
        f"  pattern_cols: {args.pattern_cols}",
        f"  pattern_rows: {args.pattern_rows}",
        f"  square_size_m: {yaml_float(args.square_size)}",
        f"  num_frames_used: {len(result.per_image_errors_px)}",
        f"  date: {timestamp}",
        "  environment: underwater",
        '  notes: "ZED2 left camera in underwater housing, fresh water pool"',
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def print_calibration_report(result: CalibrationResult) -> tuple[str, str]:
    quality, description = quality_assessment(result.rms_px)
    print()
    print(f"RMS reprojection error: {result.rms_px:.4f} px")
    print(f"Quality: {quality} ({description})")
    print("Worst frames:")
    worst = sorted(result.per_image_errors_px, key=lambda item: item[1], reverse=True)[:3]
    for index, (path, error) in enumerate(worst, start=1):
        print(f"  {index}. {path.name}: {error:.4f} px")
    return quality, description


def should_write_yaml(output_path: Path, quality: str) -> bool:
    if quality not in {"MARGINAL", "POOR"} or not output_path.exists():
        return True
    print()
    print(f"Calibration quality is {quality}, and {output_path} already exists.")
    try:
        answer = input("Overwrite existing YAML anyway? [y/N] ").strip().lower()
    except EOFError:
        answer = ""
    return answer in {"y", "yes"}


def try_calibrate_now(
    captures: list[CapturedFrame],
    object_points_template: np.ndarray,
    image_size: tuple[int, int] | None,
    run_dir: Path,
    args: argparse.Namespace,
) -> tuple[bool, str]:
    if len(captures) < MIN_CALIBRATION_FRAMES:
        return (
            False,
            f"Need at least {MIN_CALIBRATION_FRAMES} saved frames; have {len(captures)}",
        )
    if image_size is None:
        return False, "No image size available yet"

    result = run_calibration(captures, object_points_template, image_size)
    save_debug_data(run_dir, captures, object_points_template, result, image_size)
    quality, _ = print_calibration_report(result)

    if should_write_yaml(args.output, quality):
        write_calibration_yaml(args.output, result, image_size, args)
        print(f"Saved calibration YAML: {args.output}")
        print(f"Saved calibration images/debug data: {run_dir}")
        return True, f"Saved {args.output}"

    print("YAML not overwritten. Captured images/debug data were still saved.")
    print(f"Calibration images/debug data: {run_dir}")
    return True, "Calibration done; YAML not overwritten"


def capture_loop(args: argparse.Namespace) -> int:
    pattern_size = (args.pattern_cols, args.pattern_rows)
    object_points_template = create_object_points(
        args.pattern_cols,
        args.pattern_rows,
        args.square_size,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.data_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    zed = open_zed(args)
    image_mat = sl.Mat()
    runtime_params = sl.RuntimeParameters()
    captures: list[CapturedFrame] = []
    last_message = f"Saving images to {run_dir}"
    target_notice_shown = False
    image_size: tuple[int, int] | None = None

    print("Controls: SPACE save detected board, D delete last, C calibrate, Q/ESC quit")
    print(f"Calibration images will be saved to: {run_dir}")

    try:
        while True:
            if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            frame_bgr = bgr_from_zed_mat(image_mat)
            image_size = (frame_bgr.shape[1], frame_bgr.shape[0])
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            found, corners = find_checkerboard_live(gray, pattern_size)

            preview = frame_bgr.copy()
            if found and corners is not None:
                cv2.drawChessboardCorners(preview, pattern_size, corners, found)

            if len(captures) >= args.target_frames and not target_notice_shown:
                last_message = "Target reached. Press C to calibrate, or keep collecting."
                print(last_message)
                target_notice_shown = True

            display_scale = get_display_scale(preview.shape, args.display_width)
            preview = resize_for_display(preview, display_scale)
            draw_hud(preview, len(captures), args.target_frames, found, last_message)
            cv2.imshow(WINDOW_NAME, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("Quit without calibrating.")
                return 0
            if key == ord(" "):
                if not found or corners is None:
                    last_message = "SPACE ignored: checkerboard not detected"
                    print(last_message)
                    continue
                refined = refine_corners(gray, corners)
                last_message = save_capture(run_dir, frame_bgr, refined, captures)
                print(last_message)
                continue
            if key in (ord("d"), ord("D")):
                last_message = delete_last_capture(captures)
                print(last_message)
                target_notice_shown = len(captures) >= args.target_frames
                continue
            if key in (ord("c"), ord("C")):
                calibrated, last_message = try_calibrate_now(
                    captures,
                    object_points_template,
                    image_size,
                    run_dir,
                    args,
                )
                if calibrated:
                    return 0
                print(last_message)
                continue
    finally:
        zed.close()
        cv2.destroyAllWindows()


def main() -> int:
    args = parse_args()
    if sl is None:
        print(
            "Could not import pyzed.sl. Install the Stereolabs ZED SDK Python wrapper.",
            file=sys.stderr,
        )
        return 1

    if args.list:
        print_zed_cameras()
        return 0

    try:
        return capture_loop(args)
    except RuntimeError as exc:
        print(f"ZED2 underwater calibration failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
