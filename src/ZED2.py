import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys

import cv2
import numpy as np
import pyzed.sl as sl


DEFAULT_DEPTH_MIN_M = 0.15
DEFAULT_DEPTH_MAX_M = 2.0
DEFAULT_DEPTH_RANGE_M = (DEFAULT_DEPTH_MIN_M, DEFAULT_DEPTH_MAX_M)
DEFAULT_DISPLAY_WIDTH = 1600
DATA_DIR = Path("data")
RECORDINGS_DIR = DATA_DIR
RAW_DATA_DIR = DATA_DIR
WINDOW_NAME = "ZED2 RGB + depth"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure depth with a ZED2/ZED stereo camera."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print connected ZED cameras and exit.",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        help="Open a specific local ZED camera ID.",
    )
    parser.add_argument(
        "--serial",
        type=int,
        help="Open a specific ZED camera serial number.",
    )
    parser.add_argument(
        "--svo",
        help="Read from a ZED SVO file instead of a live camera.",
    )
    parser.add_argument(
        "--stream",
        help="Read from a ZED network stream, as IP or IP:PORT.",
    )
    parser.add_argument(
        "--resolution",
        default="HD720",
        choices=["HD2K", "HD1200", "HD1080", "HD720", "SVGA", "VGA"],
        help="Camera resolution; default: HD720.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Camera FPS; default: 30.",
    )
    parser.add_argument(
        "--depth-mode",
        default="NEURAL",
        choices=["PERFORMANCE", "QUALITY", "ULTRA", "NEURAL", "NEURAL_LIGHT", "NEURAL_PLUS"],
        help="ZED depth mode; default: NEURAL.",
    )
    parser.add_argument(
        "--depth-min",
        type=float,
        default=DEFAULT_DEPTH_MIN_M,
        help=f"SDK minimum depth distance in meters; default: {DEFAULT_DEPTH_MIN_M}.",
    )
    parser.add_argument(
        "--depth-max",
        type=float,
        default=DEFAULT_DEPTH_MAX_M,
        help=f"SDK maximum depth distance in meters; default: {DEFAULT_DEPTH_MAX_M}.",
    )
    parser.add_argument(
        "--depth-range",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        default=DEFAULT_DEPTH_RANGE_M,
        help=(
            "Depth color display range in meters; values outside this range "
            f"are shown as black. Default: {DEFAULT_DEPTH_RANGE_M[0]} "
            f"{DEFAULT_DEPTH_RANGE_M[1]}."
        ),
    )
    parser.add_argument(
        "--auto-depth-range",
        action="store_true",
        help="Auto-scale the depth display from valid pixels in each frame.",
    )
    parser.add_argument(
        "--confidence",
        type=int,
        default=95,
        help="Runtime confidence threshold, 0-100; default: 95.",
    )
    parser.add_argument(
        "--texture-confidence",
        type=int,
        default=100,
        help="Runtime texture confidence threshold, 0-100; default: 100.",
    )
    parser.add_argument(
        "--fill",
        action="store_true",
        help="Enable SDK depth fill mode.",
    )
    parser.add_argument(
        "--record-fps",
        type=float,
        default=30.0,
        help="Recording frame rate; default: 30.",
    )
    parser.add_argument(
        "--raw-every",
        type=int,
        default=1,
        help="Save every Nth raw frame while recording; default: 1.",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=DEFAULT_DISPLAY_WIDTH,
        help=(
            "Maximum displayed window image width. Use 0 for native size. "
            f"Default: {DEFAULT_DISPLAY_WIDTH}."
        ),
    )
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.record_fps <= 0:
        parser.error("--record-fps must be positive")
    if args.depth_min <= 0:
        parser.error("--depth-min must be positive")
    if args.depth_max <= 0:
        parser.error("--depth-max must be positive")
    if args.depth_min >= args.depth_max:
        parser.error("--depth-min must be less than --depth-max")
    if args.depth_range[0] >= args.depth_range[1]:
        parser.error("--depth-range MIN_M must be less than MAX_M")
    if not 0 <= args.confidence <= 100:
        parser.error("--confidence must be in 0..100")
    if not 0 <= args.texture_confidence <= 100:
        parser.error("--texture-confidence must be in 0..100")
    if args.raw_every <= 0:
        parser.error("--raw-every must be positive")
    if args.display_width < 0:
        parser.error("--display-width must be >= 0")

    selected_inputs = sum(
        value is not None
        for value in (args.camera_id, args.serial, args.svo, args.stream)
    )
    if selected_inputs > 1:
        parser.error("choose only one of --camera-id, --serial, --svo, or --stream")

    return args


def print_camera_list():
    print(f"ZED SDK version: {sl.Camera().get_sdk_version()}")
    devices = sl.Camera.get_device_list()
    if not devices:
        print("ZED cameras: none found")
        return

    print("ZED cameras:")
    for device in devices:
        print(
            "  - "
            f"id={device.id}, "
            f"name={device.camera_name}, "
            f"model={device.camera_model}, "
            f"serial={device.serial_number}, "
            f"state={device.camera_state}, "
            f"path={device.path}"
        )


def parse_stream(value):
    if ":" not in value:
        return value, 30000
    host, port = value.rsplit(":", 1)
    return host, int(port)


def make_init_parameters(args):
    params = sl.InitParameters()
    params.camera_resolution = getattr(sl.RESOLUTION, args.resolution)
    params.camera_fps = args.fps
    params.depth_mode = getattr(sl.DEPTH_MODE, args.depth_mode)
    params.coordinate_units = sl.UNIT.METER
    params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    params.depth_minimum_distance = args.depth_min
    params.depth_maximum_distance = args.depth_max

    if args.camera_id is not None:
        params.set_from_camera_id(args.camera_id)
    elif args.serial is not None:
        params.set_from_serial_number(args.serial)
    elif args.svo:
        params.set_from_svo_file(args.svo)
    elif args.stream:
        host, port = parse_stream(args.stream)
        params.set_from_stream(host, port)

    return params


def make_runtime_parameters(args):
    params = sl.RuntimeParameters()
    params.enable_depth = True
    params.confidence_threshold = args.confidence
    params.texture_confidence_threshold = args.texture_confidence
    params.enable_fill_mode = args.fill
    return params


def open_camera(args):
    zed = sl.Camera()
    error = zed.open(make_init_parameters(args))
    if error != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not open ZED camera: {error}")
    return zed


def camera_label(info):
    model = safe_filename(str(info.camera_model).split(".")[-1])
    serial = getattr(info, "serial_number", None)
    if serial:
        return f"{model}_{serial}"
    return model or "ZED2"


def bgr_from_zed_mat(image_mat):
    image = image_mat.get_data()
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image.copy()
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    raise RuntimeError(f"Unsupported ZED image shape: {image.shape}")


def depth_from_zed_mat(depth_mat):
    depth = depth_mat.get_data()
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth.astype(np.float32, copy=False)


def depth_mm_from_zed_mat(depth_mm_mat):
    depth_mm = depth_mm_mat.get_data()
    if depth_mm.ndim == 3:
        depth_mm = depth_mm[:, :, 0]
    return depth_mm.astype(np.uint16, copy=False)


def colorize_depth(depth_m, depth_range_m=None, auto_range=False):
    h, w = depth_m.shape
    roi = depth_m[h // 2 - 10 : h // 2 + 10, w // 2 - 10 : w // 2 + 10]
    valid_roi = roi[np.isfinite(roi) & (roi > 0)]
    median_m = float(np.median(valid_roi)) if valid_roi.size else None

    raw_valid_mask = np.isfinite(depth_m) & (depth_m > 0)
    valid_depth = depth_m[raw_valid_mask]
    if valid_depth.size:
        if auto_range:
            min_m, max_m = np.percentile(valid_depth, [2, 98])
            if max_m - min_m < 0.05:
                min_m, max_m = DEFAULT_DEPTH_RANGE_M
        else:
            min_m, max_m = depth_range_m
    else:
        min_m, max_m = DEFAULT_DEPTH_RANGE_M

    valid_mask = raw_valid_mask & (depth_m >= min_m) & (depth_m <= max_m)
    clipped = np.clip(depth_m, min_m, max_m)
    scaled = np.zeros(depth_m.shape, dtype=np.uint8)
    scaled[valid_mask] = (
        (clipped[valid_mask] - min_m) * 255 / (max_m - min_m)
    ).astype(np.uint8)

    vis = cv2.applyColorMap(255 - scaled, cv2.COLORMAP_TURBO)
    vis[~valid_mask] = (0, 0, 0)
    return vis, median_m, (float(min_m), float(max_m))


def make_combined_frame(
    rgb,
    depth_vis,
    median_m,
    depth_range_m,
    cursor_depth,
    recording,
):
    if depth_vis.shape[:2] != rgb.shape[:2]:
        depth_vis = cv2.resize(depth_vis, (rgb.shape[1], rgb.shape[0]))

    combined = np.hstack((rgb, depth_vis))
    draw_text(combined, "RGB", (16, 34))
    draw_text(combined, "Depth", (rgb.shape[1] + 16, 34))

    if median_m is not None:
        draw_text(
            combined,
            f"center: {median_m:.3f} m / {median_m * 1000:.0f} mm",
            (rgb.shape[1] + 16, rgb.shape[0] - 20),
            scale=0.75,
        )

    draw_text(
        combined,
        f"scale: {depth_range_m[0]:.2f}-{depth_range_m[1]:.2f} m",
        (rgb.shape[1] + 16, rgb.shape[0] - 52),
        scale=0.7,
    )

    if cursor_depth is not None:
        draw_cursor_depth(combined, cursor_depth)

    if recording:
        draw_text(combined, "REC", (combined.shape[1] - 90, 34), color=(0, 0, 255))

    return combined


def draw_text(image, text, origin, scale=0.9, color=(255, 255, 255)):
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_cursor_depth(combined, cursor_depth):
    x = cursor_depth["combined_x"]
    y = cursor_depth["combined_y"]
    color = (0, 255, 255) if cursor_depth["depth_m"] is not None else (0, 0, 255)

    cv2.drawMarker(
        combined,
        (x, y),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=18,
        thickness=2,
        line_type=cv2.LINE_AA,
    )
    if cursor_depth["corresponding_x"] != x:
        cv2.drawMarker(
            combined,
            (cursor_depth["corresponding_x"], y),
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=14,
            thickness=1,
            line_type=cv2.LINE_AA,
        )

    if cursor_depth["depth_m"] is None:
        label = "cursor: no depth"
    else:
        depth_mm = cursor_depth["depth_m"] * 1000
        label = (
            f"cursor: {cursor_depth['depth_m']:.3f} m / {depth_mm:.0f} mm "
            f"({cursor_depth['raw_x']}, {cursor_depth['raw_y']})"
        )

    text_x = min(max(x + 12, 8), combined.shape[1] - 360)
    text_y = min(max(y - 12, 28), combined.shape[0] - 12)
    draw_text(combined, label, (text_x, text_y), scale=0.65, color=color)


class MouseDepthProbe:
    def __init__(self):
        self.mouse_x = None
        self.mouse_y = None

    def callback(self, event, x, y, _flags, _param):
        if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            self.mouse_x = x
            self.mouse_y = y

    def sample(self, depth_m, rgb_shape, display_scale):
        if self.mouse_x is None or self.mouse_y is None:
            return None

        rgb_h, rgb_w = rgb_shape[:2]
        combined_w = rgb_w * 2
        if display_scale <= 0:
            return None

        native_x = int(self.mouse_x / display_scale)
        native_y = int(self.mouse_y / display_scale)
        if native_x < 0 or native_x >= combined_w:
            return None
        if native_y < 0 or native_y >= rgb_h:
            return None

        depth_panel_x = native_x if native_x < rgb_w else native_x - rgb_w
        raw_x = int(depth_panel_x * depth_m.shape[1] / rgb_w)
        raw_y = int(native_y * depth_m.shape[0] / rgb_h)
        raw_x = min(max(raw_x, 0), depth_m.shape[1] - 1)
        raw_y = min(max(raw_y, 0), depth_m.shape[0] - 1)

        radius = 3
        roi = depth_m[
            max(raw_y - radius, 0) : min(raw_y + radius + 1, depth_m.shape[0]),
            max(raw_x - radius, 0) : min(raw_x + radius + 1, depth_m.shape[1]),
        ]
        valid = roi[np.isfinite(roi) & (roi > 0)]
        depth_value = float(np.median(valid)) if valid.size else None

        return {
            "combined_x": native_x,
            "combined_y": native_y,
            "corresponding_x": rgb_w + depth_panel_x if native_x < rgb_w else depth_panel_x,
            "raw_x": raw_x,
            "raw_y": raw_y,
            "depth_m": depth_value,
        }


def get_display_scale(frame_shape, max_width):
    width = frame_shape[1]
    if max_width == 0 or width <= max_width:
        return 1.0
    return max_width / width


def resize_for_display(frame, scale):
    if scale == 1.0:
        return frame
    width = max(1, int(frame.shape[1] * scale))
    height = max(1, int(frame.shape[0] * scale))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def safe_filename(value):
    value = value.strip() or "zed_camera"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def session_stem(camera_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{safe_filename(camera_name)}"


def session_output_directory(stem):
    date = stem[:8] if re.match(r"^\d{8}", stem) else datetime.now().strftime("%Y%m%d")
    output_dir = DATA_DIR / date / stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def recording_path(stem):
    return session_output_directory(stem) / f"{stem}.mp4"


def open_video_writer(path, frame_shape, fps):
    height, width = frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return writer


class RawSessionWriter:
    def __init__(self, stem, info, args):
        self.path = session_output_directory(stem)
        self.depth_dir = self.path / "depth_mm"
        self.rgb_dir = self.path / "rgb_bgr"
        self.depth_dir.mkdir(parents=True, exist_ok=True)
        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.frame_index = 0

        metadata = {
            "product_name": camera_label(info),
            "camera_model": str(info.camera_model),
            "serial_number": int(info.serial_number),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "zed_sdk_version": sl.Camera().get_sdk_version(),
            "depth_units": "millimeters",
            "depth_encoding": "uint16_png",
            "rgb_encoding": "bgr_png",
            "resolution": args.resolution,
            "fps": args.fps,
            "depth_mode": args.depth_mode,
            "depth_min": args.depth_min,
            "depth_max": args.depth_max,
            "depth_range": list(args.depth_range),
            "confidence": args.confidence,
            "texture_confidence": args.texture_confidence,
            "fill": args.fill,
            "raw_every": args.raw_every,
        }
        (self.path / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    def write(self, rgb, depth_mm):
        frame_name = f"{self.frame_index:06d}.png"
        depth_path = self.depth_dir / frame_name
        rgb_path = self.rgb_dir / frame_name

        if not cv2.imwrite(str(depth_path), depth_mm):
            raise RuntimeError(f"Could not write raw depth frame: {depth_path}")
        if not cv2.imwrite(str(rgb_path), rgb):
            raise RuntimeError(f"Could not write raw RGB frame: {rgb_path}")

        self.frame_index += 1


def main():
    args = parse_args()
    if args.list:
        print_camera_list()
        return 0

    zed = None
    try:
        zed = open_camera(args)
        info = zed.get_camera_information()
        name = camera_label(info)
        print(f"Connected: {name}")
        print("Depth values are reported in meters; raw depth PNGs are millimeters.")
        print("Press R to start/stop recording, q to quit.")

        runtime = make_runtime_parameters(args)
        image_mat = sl.Mat()
        depth_mat = sl.Mat()
        depth_mm_mat = sl.Mat()
        mouse_probe = MouseDepthProbe()

        writer = None
        raw_writer = None
        output_path = None
        raw_output_path = None
        record_frame_index = 0

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, mouse_probe.callback)

        while True:
            error = zed.grab(runtime)
            if error == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            if error != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(image_mat, sl.VIEW.LEFT_BGR, sl.MEM.CPU)
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH, sl.MEM.CPU)

            rgb = bgr_from_zed_mat(image_mat)
            depth_m = depth_from_zed_mat(depth_mat)
            display_scale = get_display_scale(
                (rgb.shape[0], rgb.shape[1] * 2),
                args.display_width,
            )
            cursor_depth = mouse_probe.sample(depth_m, rgb.shape, display_scale)
            depth_vis, median_m, depth_range_m = colorize_depth(
                depth_m,
                args.depth_range,
                args.auto_depth_range,
            )
            combined = make_combined_frame(
                rgb,
                depth_vis,
                median_m,
                depth_range_m,
                cursor_depth,
                writer is not None,
            )

            if writer is not None:
                writer.write(combined)
                if record_frame_index % args.raw_every == 0:
                    zed.retrieve_measure(depth_mm_mat, sl.MEASURE.DEPTH_U16_MM, sl.MEM.CPU)
                    raw_writer.write(rgb, depth_mm_from_zed_mat(depth_mm_mat))
                record_frame_index += 1

            cv2.imshow(WINDOW_NAME, resize_for_display(combined, display_scale))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("r"), ord("R")):
                if writer is None:
                    stem = session_stem(name)
                    output_path = recording_path(stem)
                    raw_writer = RawSessionWriter(stem, info, args)
                    raw_output_path = raw_writer.path
                    writer = open_video_writer(output_path, combined.shape, args.record_fps)
                    print(f"Recording started: {output_path}")
                    print(f"Raw data started: {raw_output_path}")
                    record_frame_index = 0
                else:
                    writer.release()
                    writer = None
                    print(f"Recording saved: {output_path}")
                    print(f"Raw data saved: {raw_output_path}")
                    raw_writer = None
                    output_path = None
                    raw_output_path = None

            if key == ord("q"):
                break

        if writer is not None:
            writer.release()
            print(f"Recording saved: {output_path}")
            print(f"Raw data saved: {raw_output_path}")
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()
        if zed is not None:
            zed.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
