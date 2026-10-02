#!/usr/bin/env python3
"""Pool-corner USB cameras: live view + recording, nothing more.

For watching what happens in the pool -- not a data stream for a policy or any
algorithm. Every pool camera plugged in gets a tile in one preview window, and SPACE
records all of them, each to its own H.264 MP4.

    python pool_cam/pool_cam.py             # every pool camera found, preview only
    python pool_cam/pool_cam.py --list      # USB cameras + the USB port each is on
    python pool_cam/pool_cam.py --record    # start recording right away
    python pool_cam/pool_cam.py --size 1280x720 --fps 30
    python pool_cam/pool_cam.py --device north=0 --device south=2   # pick and name cameras
    python pool_cam/pool_cam.py --no-preview --duration 3600        # headless: record 1 hour
    ./c3 poolcam ...                        # the same, in the station's interpreter

Inside the control station: `./c3 gui --pool-cams` runs this file as a child
process (--serve, below) and shows the cameras on a POOL CAMS tab; the station's
REC UI button records them too. That is the mode for pool sessions -- one clock,
one run folder. This standalone window is for setting the cameras up.

Keys (preview window focused):
    SPACE / r   start / stop recording (all cameras together)
    q / ESC     quit -- a recording in progress is finished and kept

Which cameras: by default, every camera of a pool model (POOL_MODELS below, or
--model), so another project's USB camera on this desktop is left alone;
--device takes any. Unnamed cameras are cam0, cam1, ... in USB-port order -- so
if one is missing at start, the rest shift. For fixed names, give --device
NAME=DEV per camera (DEV = N, /dev/videoN or a /dev/v4l/by-path link); a by-path
camera that is not plugged in yet just waits for it.

--slots N (with --serve): the station always shows N tiles. Cameras present at
start take the first ones; the rest are EMPTY SLOTS, and a pool camera plugged
in later takes the first empty slot without a restart. A slot then belongs to
that USB port: unplugged and plugged back into the same port, it comes back in
the same slot; plugged into another port, it is a new camera.

Output, in --outdir (default: the repo's dated data tree,
data/YYYYMMDD/MMDD_HHMMSS_poolcam/, made when recording first starts -- see
rov_gui/runstore.py; from the station: <run folder>/pool_HHMMSS/):
    cam0_2026-09-30_17-05-12.mp4    the video
    cam0_2026-09-30_17-05-12.json   device, USB port, settings, wall-clock start/end,
                                    t0_monotonic, frame counts ("complete": false
                                    until it is closed)

Lining a video up with anything else on this machine: frame k of a file shows
CLOCK_MONOTONIC time  t0_monotonic + k / fps  (to within 3/4 frame, below) -- the
clock behind time.monotonic(), so the station's host_stamp columns (nav
fixes.csv) and its UI recording's started_monotonic read on the same axis.

--serve (for rov_gui; not meant to be typed): no window; commands arrive on
stdin and status, log lines and preview frames leave on stdout, framed as in
protocol.py. The station starting and stopping recordings is the only thing
that crosses -- nothing here ever reaches its control or policy code. Closing
stdin (the station exiting, even by a crash) stops the recordings and ends the
process, with every file finished.

How the recording behaves:
  * Constant frame rate on the camera's own clock (the driver's per-frame kernel
    timestamps): a missing frame is covered by repeating the last one, a surplus
    one skipped. So an hour plays as an hour even though this camera really runs at
    ~30.1 fps, and frame N of every camera started together shows the same moment,
    to within about a frame.
  * If this program dies (crash, kill, closed terminal), ffmpeg still finishes the
    file. If ffmpeg or the machine dies, the MP4 is fragmented, so it plays up to
    its last 2-second fragment on disk -- after a power cut, up to ~30 s more can
    be missing (Linux writes files out with that delay).
  * ffmpeg does the encoding (NVENC on the GPU when it works, else libx264), so the
    Python side needs only cv2 + numpy and runs in any env here.
  * Unplugging a camera finishes its file; it is reopened when it comes back and,
    if recording is still on, a new file starts.
"""

import argparse
import fcntl
import glob
import json
import math
import os
import queue
import re
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if __package__ in (None, ""):
    # Run as a script. The repo root goes FIRST, so `pool_cam` below is the
    # package and not this file (whose folder Python put on the path), and
    # rov_gui -- a sibling package -- is importable at all.
    sys.path[:0] = [str(REPO)]

from pool_cam import protocol          # noqa: E402
from rov_gui import runstore           # noqa: E402  (stdlib only; no Qt)

DATA_ROOT = REPO / "data"              # the one dated data tree (rov_gui/runstore.py)
RUN_KIND = "poolcam"                   # standalone runs: data/YYYYMMDD/MMDD_HHMMSS_poolcam/
WINDOW = "pool cameras   SPACE: record/stop   q: quit"
# V4L2 card names that count as pool cameras (substring match; --model replaces):
POOL_MODELS = (
    "USB RGB Camera",       # Realtek 0bda:2a16, the first pool camera
    "SPCA2650 AV Camera",   # Sunplus SPCA2650 (also has a microphone), 2026-09-30
)


def is_pool(model, models=POOL_MODELS):
    return any(m in model for m in models)

# struct v4l2_capability (104 bytes): driver[16] card[32] bus_info[32] version
# capabilities device_caps reserved[3]
VIDIOC_QUERYCAP = 0x80685600
# struct v4l2_control (8 bytes): id, value -- _IOWR('V', 28, ...)
VIDIOC_S_CTRL = 0xC008561C
V4L2_CID_POWER_LINE_FREQUENCY = 0x00980918
# Anti-flicker: the camera's exposure follows the mains the room lights run on.
# These cameras power up at 50 Hz; the pool's mains is 60 Hz (KR and US both),
# and the mismatch rolls horizontal bands through the picture: frame-to-frame
# row-brightness std at 50 / 60 Hz, SPCA2650 3.75 / 0.59, the three USB RGB
# Cameras 1.48-1.92 / 0.16-0.25 [측정: pool_cam/bench_out/flicker_20261001.txt,
# pool_cam/bench_flicker.py].
POWER_LINE = {"off": 0, "50": 1, "60": 2, "auto": 3}
V4L2_CAP_VIDEO_CAPTURE = 0x00000001
V4L2_CAP_DEVICE_CAPS = 0x80000000

FONT = cv2.FONT_HERSHEY_SIMPLEX
RED, WHITE, AMBER = (40, 40, 255), (255, 255, 255), (0, 190, 255)


_say_hook = None     # --serve: where say() goes instead (the station's log)


def say(*parts, level="info"):
    """print() that cannot fail: a closed terminal or a dead `| tee` must not stop
    the recordings from being finished properly. Under --serve the line goes to
    the station instead, tagged with `level` (info / warn / error)."""
    if _say_hook is not None:
        _say_hook(level, " ".join(str(p) for p in parts))
        return
    try:
        print(*parts, flush=True)
    except (OSError, ValueError):
        sys.stdout = open(os.devnull, "w")    # gone for good; and exit won't trip on it


# ---------------------------------------------------------------- finding cameras

def natural_key(text):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", text)]


def query_cap(node):
    """(driver, card, usb_port, can_capture) of a /dev/video* node, or None."""
    buf = bytearray(104)
    try:
        fd = os.open(node, os.O_RDWR | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        fcntl.ioctl(fd, VIDIOC_QUERYCAP, buf)
    except OSError:
        return None
    finally:
        os.close(fd)
    text = lambda a, b: buf[a:b].split(b"\0", 1)[0].decode(errors="replace")
    caps, device_caps = struct.unpack_from("<II", buf, 84)
    if caps & V4L2_CAP_DEVICE_CAPS:
        caps = device_caps
    return text(0, 16), text(16, 48), text(48, 80), bool(caps & V4L2_CAP_VIDEO_CAPTURE)


def usb_cameras():
    """(path, model, usb_port) of every USB (uvcvideo) camera, in USB-port order.

    The path is a /dev/v4l/by-path link where udev made one: it names the USB port,
    so it survives replugging and reboots, unlike /dev/videoN. (Identical cameras can
    share a serial number -- this model reports "2022" -- so /dev/v4l/by-id can't
    tell them apart.) Each UVC camera also has a metadata node; it is skipped.
    """
    found, seen = [], set()
    for dev in (sorted(glob.glob("/dev/v4l/by-path/*"), key=natural_key)
                + sorted(glob.glob("/dev/video*"), key=natural_key)):
        node = os.path.realpath(dev)
        if node in seen:
            continue
        seen.add(node)
        info = query_cap(node)
        if info and info[0] == "uvcvideo" and info[3]:
            found.append((dev, info[1], info[2]))
    return found


def set_power_line(node, hz):
    """Set a camera's anti-flicker frequency ("off" / "50" / "60" / "auto").
    None on success, else why not. Its own short-lived fd: controls belong to
    the device, not to the fd, and the capture's fd is OpenCV's. The camera
    forgets the setting when it is unplugged, so this runs on every open."""
    buf = bytearray(struct.pack("<Ii", V4L2_CID_POWER_LINE_FREQUENCY, POWER_LINE[hz]))
    try:
        fd = os.open(node, os.O_RDWR | os.O_NONBLOCK)
    except OSError as ex:
        return str(ex)
    try:
        fcntl.ioctl(fd, VIDIOC_S_CTRL, buf)
    except OSError as ex:
        return str(ex)
    finally:
        os.close(fd)
    return None


def resolve_device(arg):
    """--device value -> path to open. A bare number N means /dev/videoN, and a node
    is swapped for its by-path link so a replugged camera is found again."""
    dev = f"/dev/video{arg}" if arg.isdigit() else arg
    node = os.path.realpath(dev)
    for link in sorted(glob.glob("/dev/v4l/by-path/*"), key=natural_key):
        if os.path.realpath(link) == node:
            return link
    return dev


def controller_of(usb_port):
    """'usb-0000:13:00.0-3' (V4L2 bus_info) -> '0000:13:00.0', the PCI USB
    controller the camera hangs off ('' if the form is unexpected)."""
    m = re.match(r"usb-([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7])-", usb_port or "")
    return m.group(1) if m else ""


def pc_port_of(usb_port):
    """'usb-0000:13:00.0-1.3' -> ('0000:13:00.0', '1'): the controller and the
    PC's own port the camera hangs off, through however many hubs (a dot in
    the path = a hub)."""
    m = re.match(r"usb-([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7])-(\d+)",
                 usb_port or "")
    return (m.group(1), m.group(2)) if m else ("", "")


def usb_controllers():
    """Every USB controller in the machine, by PCI address. Only symlinks are
    read: a wedged device's sysfs attributes can block forever (2026-10-01)."""
    out = set()
    for bus in glob.glob("/sys/bus/usb/devices/usb*"):
        pci = re.findall(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", os.path.realpath(bus))
        if pci:
            out.add(pci[-1])
    return sorted(out)


def camera_report(models=POOL_MODELS):
    """The --list text: every USB camera, the pool cameras per USB controller,
    and a warning where two share one PC port through a hub.

    Bandwidth is per PC port, not per controller: this machine's xHCI gives
    each of its own ports a 480 Mbit/s link, and on 2026-10-01 four cameras
    on four ports of ONE controller streamed 1080p30 together
    (pool_cam/bench_out/cost_20261001_real4_1080p30.txt) although each
    reserves 3072 B per microframe -- alt setting 7, read live from sysfs --
    and 2 x 3072 is past USB 2.0's 6000 B periodic cap. Behind a hub,
    cameras share the hub's ONE link: at 1080p30 only one of them fits. The
    controller grouping is still worth seeing: a controller wedged by a
    driver crash takes every camera on it along (KNOWN_ISSUES 2026-10-01)."""
    found = usb_cameras()
    lines = [] if found else ["no USB cameras found"]
    n, per, by_port = 0, {}, {}
    for dev, model, usb in found:
        ctl = controller_of(usb)
        if is_pool(model, models):
            tag, note, n = f"cam{n}", "", n + 1
            per.setdefault(ctl, []).append(tag)
            by_port.setdefault(pc_port_of(usb), []).append(tag)
        else:
            tag, note = "  - ", "   (not a pool camera: used only with --device)"
        lines.append(f"{tag:5s} {os.path.realpath(dev)}  {usb}  {model}{note}\n      {dev}")
    lines.append("")
    lines.append("pool cameras per USB controller (the PCI address inside usb-XXXX:XX:XX.X-N;"
                 " bandwidth is per PC port, so this is about faults, not speed):")
    for ctl in usb_controllers():
        cams = per.get(ctl, [])
        lines.append(f"  {ctl}  {len(cams)}  {' '.join(cams)}")
    shared = {k: v for k, v in by_port.items() if len(v) > 1}
    for (ctl, port), cams in sorted(shared.items()):
        lines.append(f"  !! {' '.join(cams)} share PC port {port} of {ctl} through a hub: "
                     "at 1080p30 only ONE of them can stream -- give each its own PC port")
    return "\n".join(lines)


def pool_devices(models=POOL_MODELS):
    """Device paths of the pool cameras plugged in now, in USB-port order."""
    return [dev for dev, model, _ in usb_cameras() if is_pool(model, models)]


def fill_slots(slots, found):
    """Where newly plugged cameras go: [(slot index, device)].

    `slots` is [(label, device or None)], None an empty slot; `found` the pool
    cameras plugged in now. A camera already in a slot -- the same link, or the
    same /dev/videoN -- is not new. New ones take the empty slots in order;
    any beyond the last empty slot are left alone."""
    taken = set()
    for _, dev in slots:
        if dev:
            taken.add(dev)
            if os.path.exists(dev):
                taken.add(os.path.realpath(dev))
    free = [i for i, (_, dev) in enumerate(slots) if dev is None]
    out = []
    for dev in found:
        if not free:
            break
        if dev in taken or os.path.realpath(dev) in taken:
            continue
        out.append((free.pop(0), dev))
        taken.update((dev, os.path.realpath(dev)))
    return out


def frame_time(cap):
    """When the frame was captured: the driver's kernel timestamp (CLOCK_MONOTONIC,
    the same clock as time.monotonic()), else the time we received it."""
    now = time.monotonic()
    t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    return t if 0.0 < now - t < 1.0 else now


def wall_clock(t_mono):
    """time.monotonic() seconds -> local datetime."""
    return datetime.fromtimestamp(time.time() - time.monotonic() + t_mono)


def hms(seconds):
    s = int(seconds)
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"


# ---------------------------------------------------------------- synthetic cameras

FAKE_PREFIX = "fake:"
PROBLEM_REPEAT_S = 60.0   # the same camera problem is said again after this long
DISK_CHECK_S = 10.0       # --serve: free space is checked this often while recording,
DISK_STOP_GB = 2.0        # and recording stops below this (the station shares the disk)
STUCK_S = 15.0            # opening / first frame this long = the driver has it, not us
SETTLE_S = 1.0            # --slots: a plug-in is acted on once /dev holds still this long     # a --device of this form is a FakeCapture, not hardware


class FakeCapture:
    """Stands in for cv2.VideoCapture on a "fake:N" device: a moving test
    pattern at the requested size and rate, stamped on CLOCK_MONOTONIC like the
    real driver. For tests and for `rov_gui --source demo`, which opens no
    hardware -- nothing it records is a measurement of anything."""

    def __init__(self, label, size, fps):
        self.w, self.h = size
        self.fps = float(fps)
        self.label = label
        self._t = time.monotonic()
        self._n = 0
        y, x = np.mgrid[0:self.h, 0:self.w]
        self._bg = np.dstack([(x * 255 // max(1, self.w - 1)).astype(np.uint8),
                              (y * 255 // max(1, self.h - 1)).astype(np.uint8),
                              np.full((self.h, self.w), 60, np.uint8)])

    def isOpened(self):
        return True

    def set(self, *_):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self._t * 1000.0
        if prop == cv2.CAP_PROP_FOURCC:
            return float(cv2.VideoWriter_fourcc(*"MJPG"))
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        return 0.0

    def read(self):
        self._t += 1.0 / self.fps
        delay = self._t - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            self._t = time.monotonic()        # fell behind: don't sprint to catch up
        self._n += 1
        img = self._bg.copy()
        x = int(self._n * 8) % self.w
        img[:, max(0, x - 6):x + 6] = 255
        put_text(img, f"{self.label} SYNTHETIC  frame {self._n}",
                 (10, max(20, self.h // 2)), WHITE, max(0.4, self.h / 900))
        return True, img

    def release(self):
        pass


# ---------------------------------------------------------------------- recording

def pick_encoder(ffmpeg, want):
    """'nvenc' if h264_nvenc really encodes on this machine, else 'x264'."""
    if want == "x264":
        return "x264"
    probe = [ffmpeg, "-hide_banner", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "gray", "-s", "256x256", "-framerate", "30",
             "-i", "pipe:0", "-c:v", "h264_nvenc", "-f", "null", "-"]
    try:
        ok = subprocess.run(probe, input=bytes(256 * 256 * 5), capture_output=True,
                            timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    if not ok and want == "nvenc":
        sys.exit(f"h264_nvenc does not work with {ffmpeg}")
    return "nvenc" if ok else "x264"


# Measured on this camera at 1080p30: these two give the same SSIM (0.981 vs a
# lossless reference) at ~6 Mbit/s. NVENC CQ 23 would be twice the size for 0.005.
DEFAULT_QUALITY = {"nvenc": 28, "x264": 23}


def ffmpeg_command(ffmpeg, encoder, quality, size, fps, path):
    w, h = size
    if encoder == "nvenc":
        # -b:v 0 = constant quality, but NVENC still caps it at ~20 Mbit/s unless
        # told otherwise -- which would blur the busiest moments at low CQ.
        codec = ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr",
                 "-cq", str(quality), "-b:v", "0", "-maxrate", "60M", "-bufsize", "120M"]
    else:
        codec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(quality)]
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostats", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
        "-framerate", f"{fps:g}", "-i", "pipe:0", "-an",
        # BGR -> BT.709 limited-range 4:2:0, tagged as such, so players show the
        # colours the camera saw instead of guessing the matrix
        "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
        *codec, "-g", str(max(1, round(2 * fps))),
        "-color_primaries", "bt709", "-color_trc", "bt709",
        "-colorspace", "bt709", "-color_range", "tv",
        # A keyframe + fragment every 2 s, each handed to the OS as soon as it is
        # done (by default ffmpeg holds 256 KiB, many seconds at a low bitrate):
        # a file that is never closed still plays up to its last fragment.
        "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
        "-flush_packets", "1",
        str(path),
    ]


class Recorder:
    """One MP4. Frames come in from the camera thread; a writer thread of its own
    pipes them into ffmpeg, so a slow encoder or disk never stalls the camera.

    The video is constant frame rate on the camera's clock: slot k of the file shows
    time t0 + k/fps, filled with the latest frame captured by then.
    """

    def __init__(self, path, size, fps, t0, info, ffmpeg, encoder, quality):
        self.path, self.size, self.fps, self.t0 = Path(path), size, fps, t0
        self.encoder, self.info = encoder, info
        self.frames = 0       # frames in the file: its length is frames / fps
        self.received = 0     # distinct camera frames that went in
        self.repeated = 0     # extra copies that covered a late or missing frame
        self.skipped = 0      # camera frames left out because the camera ran ahead
        self.dropped = 0      # camera frames lost because the writer fell behind
        self.error = None
        self._write_info(final=False)    # first, so an unwritable folder fails before ffmpeg runs
        self._queue = queue.Queue()      # unbounded, so stop() never blocks; push() caps it
        self._max_queued = max(2, round(fps))   # ~1 s of frames
        self._prev = None                # the last frame written: what fills a gap
        self._log = tempfile.TemporaryFile()
        # Own session: a Ctrl+C in the terminal must not reach ffmpeg (it would
        # stop with frames still queued). It ends when we close its stdin -- and
        # also if this process dies, so a crash still leaves a finished file.
        self._proc = subprocess.Popen(
            ffmpeg_command(ffmpeg, encoder, quality, size, fps, self.path),
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._log,
            start_new_session=True)
        # A daemon, so a dying program is not held up by it (ffmpeg then finishes
        # the file on its own); a normal exit waits for it in Camera.close().
        self._thread = threading.Thread(target=self._run, name=f"rec-{self.path.stem}",
                                        daemon=True)
        self._thread.start()

    @property
    def seconds(self):
        return self.frames / self.fps

    def push(self, image, ts):
        """Hand over a camera frame. Never blocks: if the writer is a full second
        behind, the frame is counted as dropped instead."""
        if self._queue.qsize() >= self._max_queued:
            self.dropped += 1
        else:
            self._queue.put((image, ts))

    def stop(self):
        """Finish the file. Returns at once; wait() blocks until it is closed."""
        self._queue.put(None)

    def wait(self, timeout=30):
        """Block until the file is closed. An ffmpeg that stopped reading (hung GPU,
        stalled disk) is killed after `timeout` s; the file keeps what it had."""
        self._thread.join(timeout)
        if self._thread.is_alive():
            self._proc.kill()
            self._thread.join()

    def _run(self):
        next_slot = 0
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                if self.error is None:            # after a failure: drain until stop()
                    try:
                        next_slot = self._write(*item, next_slot)
                    except Exception as ex:       # a bug must not end the file silently
                        self.error = f"writer error: {ex!r}"
        finally:
            self._close()

    def _write(self, image, ts, next_slot):
        # How late this frame is for the next slot, in frames. Timestamp jitter is
        # ~0.06 frame and drift ~0.003 frame per frame (the camera runs ~0.3% fast).
        # Acting only past 3/4 of a frame -- not at 1/2, like plain rounding --
        # means jitter can't flip frames back and forth between slots (a visible
        # stutter): there is one skip per surplus camera frame, one repeat per
        # missing one, and each frame lands within 3/4 frame of its own time.
        late = (ts - self.t0) * self.fps - next_slot
        if late <= -0.75:                          # camera ahead, or frame before t0
            if ts >= self.t0:
                self.skipped += 1
            return next_slot
        copies = 1 + max(0, math.ceil(late - 0.75))    # >1: frames came late or got lost
        # The slots it is too late for keep showing the previous frame -- the
        # newest one captured by then. (A file's very first frame has none.)
        fill = image if self._prev is None else self._prev
        try:
            for _ in range(copies - 1):
                self._proc.stdin.write(fill.data)
            self._proc.stdin.write(image.data)
        except OSError:                            # BrokenPipe: ffmpeg is gone
            self.error = "ffmpeg stopped"
            return next_slot
        self._prev = image
        self.received += 1
        self.frames += copies
        self.repeated += copies - 1
        return next_slot + copies

    def _close(self):
        self._prev = None
        try:
            self._proc.stdin.close()               # EOF: ffmpeg writes the last fragment
        except OSError:
            pass
        try:
            code = self._proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            code = self._proc.wait()
            self.error = self.error or "ffmpeg did not finish within 60 s; killed"
        if code != 0:
            self._log.seek(0)
            tail = self._log.read().decode(errors="replace").strip().splitlines()[-3:]
            self.error = " | ".join([f"{self.error or 'ffmpeg failed'} (exit code {code})"] + tail)
        self._log.close()
        try:
            self._write_info(final=True)
        except OSError as ex:
            self.error = self.error or f"cannot write {self.path.with_suffix('.json').name}: {ex}"
        mb = self.path.stat().st_size / 1e6 if self.path.exists() else 0.0
        notes = [f"{n} {what}" for n, what in ((self.repeated, "repeated"),
                 (self.skipped, "skipped"), (self.dropped, "dropped")) if n]
        say(f"saved {self.path}  ({hms(self.seconds)}, {mb:.0f} MB, {self.frames} frames"
            + (f"; {', '.join(notes)}" if notes else "") + ")"
            + (f"\n  ERROR: {self.error}" if self.error else ""),
            level="error" if self.error else "info")

    def _write_info(self, final):
        info = dict(self.info, video=self.path.name,
                    size=f"{self.size[0]}x{self.size[1]}", fps=self.fps,
                    encoder=self.encoder,
                    # Frame k shows CLOCK_MONOTONIC t0_monotonic + k/fps: the key
                    # to lining this file up with anything else on this machine.
                    t0_monotonic=round(self.t0, 6),
                    start=wall_clock(self.t0).isoformat(timespec="milliseconds"),
                    end=(wall_clock(self.t0 + self.seconds).isoformat(timespec="milliseconds")
                         if final else None),
                    complete=bool(final and not self.error),
                    frames=self.frames, seconds=round(self.seconds, 3),
                    camera_frames=self.received, repeated=self.repeated,
                    skipped=self.skipped, dropped=self.dropped, error=self.error)
        self.path.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")


# ------------------------------------------------------------------------- camera

class Camera:
    """One USB camera. A capture thread keeps the newest frame for the preview and,
    while recording is on, hands every frame to a Recorder. Losing the camera
    finishes the current file; it is reopened when it comes back."""

    def __init__(self, label, device, size, fps, outdir, ffmpeg, encoder, quality,
                 power_line="60"):
        self.label, self.device = label, device
        self.req_size, self.fps = size, fps
        self.power_line = power_line   # anti-flicker, set on every open (set_power_line)
        # Where the next file goes. start_recording() may name a folder per
        # recording (the station does: one per REC); None until one is given.
        self.outdir = Path(outdir) if outdir is not None else None
        self.rec_opts = dict(ffmpeg=ffmpeg, encoder=encoder, quality=quality)
        info = query_cap(os.path.realpath(device)) or ("", "?", "?", False)
        self.model, self.usb = info[1], info[2]
        self.size = None              # what the camera actually delivers
        self.connected = False
        self._status = ("opening", time.monotonic())   # (text, since): see .status
        self.recorder = None          # the Recorder being fed right now
        self._failed = None           # last Recorder that failed, or why one couldn't start
        self._reported = {}           # problem -> when last printed (retries don't repeat it)
        self._done = []               # stopped Recorders, possibly still flushing
        self._latest = None           # (image, timestamp, sequence number)
        self._times = deque(maxlen=31)
        self._lock = threading.Lock()
        self._want_rec = False
        self._rec_t0 = None           # set by start_recording(), taken by the thread
        self._quit = threading.Event()
        self._thread = threading.Thread(target=self._run, name=label, daemon=True)

    # -- control, from any thread

    def start(self):
        self._thread.start()
        return self

    def start_recording(self, t0=None, outdir=None):
        """Record from monotonic time t0. Give several cameras the same t0 and
        their files line up frame for frame. `outdir` changes where this and
        later files go (a file resumed after an unplug goes there too)."""
        with self._lock:
            if outdir is not None:
                self.outdir = Path(outdir)
            self._want_rec = True
            self._rec_t0 = time.monotonic() if t0 is None else t0
        self._failed = None

    def stop_recording(self):
        with self._lock:
            self._want_rec = False

    @property
    def status(self):
        """What the tile shows while not connected ("ok" while streaming)."""
        return self._status[0]

    @status.setter
    def status(self, text):
        if text != self._status[0]:
            self._status = (text, time.monotonic())

    @property
    def stuck_s(self):
        """Seconds spent opening / waiting for a first frame, or 0. OpenCV's own
        read times out after 10 s and says so; far past that, the thread is not
        coming back -- on 2026-10-01 the kernel's uvcvideo crashed inside this
        camera's STREAMON (a cable pulled mid-open) and killed it there."""
        text, since = self._status
        if self.connected or text not in ("opening", "starting"):
            return 0.0
        return time.monotonic() - since

    @property
    def node(self):
        """The /dev/videoN the device path points at right now."""
        if self.device.startswith(FAKE_PREFIX):
            return self.device
        return os.path.realpath(self.device)

    @property
    def wants_recording(self):
        return self._want_rec

    @property
    def rec_error(self):
        """Why the last recording failed, if it did. Read live: a failed file's
        error gains ffmpeg's own message once ffmpeg has exited."""
        failed = self._failed
        return failed.error if isinstance(failed, Recorder) else failed

    def latest(self):
        with self._lock:
            return self._latest

    def fps_now(self):
        t = list(self._times)
        if not self.connected or len(t) < 2 or t[-1] <= t[0]:
            return 0.0
        return (len(t) - 1) / (t[-1] - t[0])

    def close(self):
        """Stop capturing, then wait until every file this camera wrote is closed."""
        self._quit.set()
        self._thread.join(timeout=15)     # read() gives up after 10 s at worst
        if self._thread.is_alive():       # still stuck in read(): finish from here
            self._end_recording()
        for rec in list(self._done):
            rec.wait()

    # -- capture thread

    def _run(self):
        while not self._quit.is_set():
            cap = None
            try:
                cap = self._open()
                if cap is not None:
                    self._stream(cap)
            except Exception as ex:
                # e.g. cv2.error from a dying USB controller: report it, then
                # reopen as after an unplug instead of losing the camera for good
                lines = str(ex).strip().splitlines() or [type(ex).__name__]
                self._problem(f"error: {lines[-1][:80]}")
            finally:
                if cap is not None:
                    cap.release()
                self.connected = False
                self._end_recording()
            self._quit.wait(1.0)          # then retry: unplugged, busy, or not yet up

    def _problem(self, text):
        """Show a problem on the tile, and once in the terminal (not every retry)."""
        self.status = text
        # Once per problem per minute, not once per CHANGE: a camera that opens
        # but sends nothing alternates "busy" / "no frames" every retry, and
        # per-change reporting put a line a second into the station's log.
        now = time.monotonic()
        if now - self._reported.get(text, -1e9) > PROBLEM_REPEAT_S:
            self._reported[text] = now
            say(f"[{self.label}] {text}", level="warn")

    def _open(self):
        if self.device.startswith(FAKE_PREFIX):
            return FakeCapture(self.label, self.req_size, self.fps)
        node = os.path.realpath(self.device)
        if not os.path.exists(node):
            self._problem("not connected")
            return None
        cap = cv2.VideoCapture(node, cv2.CAP_V4L2)
        if not cap.isOpened():
            self._problem("busy (open in another program?)")
            cap.release()
            return None
        w, h = self.req_size
        # MJPG first: it is the only format this kind of camera streams at full
        # rate in HD (raw YUYV manages 5 fps at 1080p).
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        if self.power_line:
            # Before the first read (STREAMON), so the first frames are clean.
            err = set_power_line(node, self.power_line)
            if err:
                self._problem_once(f"anti-flicker {self.power_line} Hz not set: {err}")
        return cap

    def _problem_once(self, text):
        """A note that is not a fault: logged once, the tile is left alone."""
        if text not in self._reported:
            self._reported[text] = time.monotonic()
            say(f"[{self.label}] {text}", level="warn")

    def _stream(self, cap):
        self.status = "starting"
        first = True
        while not self._quit.is_set():
            ok, image = cap.read()
            if not ok:
                self._problem("no frames: USB bandwidth? (cameras sharing a USB link)"
                              if first else "stopped sending frames (unplugged?)")
                return
            ts = frame_time(cap)
            if first:
                first = False
                self._on_connect(cap, image)
            self._times.append(ts)
            with self._lock:
                seq = self._latest[2] + 1 if self._latest else 1
                self._latest = (image, ts, seq)
            self._feed(image, ts)

    def _on_connect(self, cap, image):
        self.size = (image.shape[1], image.shape[0])
        self._times.clear()
        info = (None if self.device.startswith(FAKE_PREFIX)
                else query_cap(os.path.realpath(self.device)))
        if info:                          # unknown if it was absent at startup
            self.model, self.usb = info[1], info[2]
        self.connected, self.status, self._reported = True, "ok", {}
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="replace")
        say(f"[{self.label}] streaming {self.size[0]}x{self.size[1]} {fourcc} "
            f"@ {cap.get(cv2.CAP_PROP_FPS):g} fps  ({self.node}, {self.usb})")
        if self.size != tuple(self.req_size):
            say(f"[{self.label}] note: asked for {self.req_size[0]}x{self.req_size[1]}, "
                f"the camera gives {self.size[0]}x{self.size[1]}")
        if fourcc != "MJPG":
            say(f"[{self.label}] note: camera refused MJPG; {fourcc} may be slow")

    def _feed(self, image, ts):
        with self._lock:
            want, t0, outdir = self._want_rec, self._rec_t0, self.outdir
            self._rec_t0 = None
        rec = self.recorder
        if rec is not None and rec.error:
            self._failed = rec            # e.g. disk full: stop, don't retry every frame
            want = False
            with self._lock:
                self._want_rec = False
        if rec is not None and (not want or t0 is not None or rec.error):
            self._end_recording()         # stopped, failed, or restarted by a new start
        if want and self.recorder is None:
            # A fresh start lines up on the shared t0 (a streaming camera delivers
            # its next frame well within 0.25 s). A file resumed after the camera
            # came back, or a start given while it was still opening, begins at
            # its first frame instead of opening on copies of it.
            self._begin_recording(t0 if t0 is not None and ts - t0 < 0.25 else ts,
                                  outdir)
        if self.recorder is not None:
            self.recorder.push(image, ts)

    def _begin_recording(self, t0, outdir):
        stamp = wall_clock(t0).strftime("%Y-%m-%d_%H-%M-%S")
        info = dict(camera=self.label, device=self.device, power_line=self.power_line,
                    node=self.node, usb_port=self.usb, model=self.model)
        try:
            if outdir is None:
                raise RuntimeError("no output folder given")
            outdir.mkdir(parents=True, exist_ok=True)
            path = outdir / f"{self.label}_{stamp}.mp4"
            n = 2
            while path.exists() or path.with_suffix(".json").exists():
                path = outdir / f"{self.label}_{stamp}_{n}.mp4"
                n += 1
            self.recorder = Recorder(path, self.size, self.fps, t0, info, **self.rec_opts)
        except Exception as ex:           # folder gone or unwritable, ffmpeg missing...
            self._failed = f"cannot start recording: {ex}"
            with self._lock:
                self._want_rec = False
            say(f"[{self.label}] {self._failed}", level="error")
            return
        say(f"[{self.label}] recording -> {path}")

    def _end_recording(self):
        rec, self.recorder = self.recorder, None
        if rec is not None:
            self._done.append(rec)
            rec.stop()


# ------------------------------------------------------------------------ preview

def put_text(img, text, org, color=WHITE, scale=0.55):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, 1, cv2.LINE_AA)


def fit(image, w, h):
    """Scale into w x h, aspect kept, letterboxed."""
    ih, iw = image.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(1, round(iw * s)), max(1, round(ih * s))
    small = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    if (nw, nh) == (w, h):
        return small
    out = np.zeros((h, w, 3), np.uint8)
    x, y = (w - nw) // 2, (h - nh) // 2
    out[y:y + nh, x:x + nw] = small
    return out


def draw_tile(cam, w, h, cache):
    item = cam.latest()
    if cam.connected and item is not None:
        image, _, seq = item
        if cache.get(cam.label, (None,))[0] != seq:       # resize only new frames
            cache[cam.label] = (seq, fit(image, w, h))
        tile = cache[cam.label][1].copy()
        put_text(tile, f"{cam.label}  {cam.size[0]}x{cam.size[1]}  "
                       f"{cam.fps_now():.1f} fps", (10, 24))
    else:
        tile = np.zeros((h, w, 3), np.uint8)
        put_text(tile, f"{cam.label}: {cam.status}", (10, 24), AMBER)
    rec = cam.recorder
    if rec is not None:
        try:
            mb = rec.path.stat().st_size / 1e6
        except OSError:
            mb = 0.0
        label = f"REC {hms(rec.seconds)}  {mb:.0f} MB"
        tw = cv2.getTextSize(label, FONT, 0.55, 1)[0][0]
        cv2.circle(tile, (w - tw - 26, 18), 7, RED, -1, cv2.LINE_AA)
        put_text(tile, label, (w - tw - 12, 24), RED)
        cv2.rectangle(tile, (0, 0), (w - 1, h - 1), RED, 2)
    if cam.rec_error:
        put_text(tile, f"recording failed: {cam.rec_error}"[:110], (10, h - 34), RED, 0.45)
    put_text(tile, f"{cam.node}  {cam.usb}", (10, h - 12), WHITE, 0.45)
    return tile


def compose(cams, width, cache):
    cols = math.ceil(math.sqrt(len(cams)))
    rows = math.ceil(len(cams) / cols)
    tw = width // cols
    th = tw * 9 // 16
    canvas = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for i, cam in enumerate(cams):
        r, c = divmod(i, cols)
        canvas[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = draw_tile(cam, tw, th, cache)
    return canvas


# --------------------------------------------------------------------------- main

class Stop:
    """Quit request, set by the signal handler. A plain attribute on purpose: the
    handler runs on the main thread, maybe while it is inside a threading.Event's
    wait() holding the Event's lock -- Event.set() would then block forever."""
    requested = False


def set_recording(cams, on, outdir=None):
    if on:
        t0 = time.monotonic()                 # one t0 for all: files line up
        for cam in cams:
            cam.start_recording(t0, outdir)
    else:
        for cam in cams:
            cam.stop_recording()


def quiet_opencv():
    """OpenCV prints a misleading warning every second while a camera can't be
    opened; the tile and the terminal already say what is wrong."""
    for set_level in (lambda: cv2.setLogLevel(2),             # OpenCV 4.x: 2 = ERROR
                      lambda: cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)):
        try:
            set_level()
            return
        except AttributeError:
            pass


def window_closed():
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:   # OpenCV 4.12/5.0 tear Qt down when the last window closes
        return True


def run_preview(cams, args, stop, deadline, rec_dir):
    cache, sized = {}, False
    while not stop.requested and (deadline is None or time.monotonic() < deadline):
        canvas = compose(cams, args.preview_width, cache)
        cv2.imshow(WINDOW, canvas)
        if not sized:
            cv2.resizeWindow(WINDOW, canvas.shape[1], canvas.shape[0])
            sized = True
        key = cv2.waitKey(15) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord(" "), ord("r")):
            on = not any(c.wants_recording for c in cams)
            set_recording(cams, on, rec_dir() if on else None)
        if window_closed():               # the window's X button
            break
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def run_headless(cams, stop, deadline):
    next_report = time.monotonic() + 5
    while not stop.requested and (deadline is None or time.monotonic() < deadline):
        time.sleep(0.2)
        if time.monotonic() >= next_report:
            next_report += 5
            for cam in cams:
                rec = cam.recorder
                say(f"[{cam.label}] {cam.status}, {cam.fps_now():.1f} fps"
                    + (f", REC {hms(rec.seconds)}" if rec else ""))


def parse_size(text):
    m = re.fullmatch(r"(\d+)[xX](\d+)", text)
    if not m:
        raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT, got {text!r}")
    return int(m.group(1)), int(m.group(2))


def parse_device(arg, index):
    """'NAME=DEV' or 'DEV' -> (name, path to open)."""
    name, sep, dev = arg.partition("=")
    if not sep:
        name, dev = f"cam{index}", arg
    if not re.fullmatch(r"[\w.-]+", name):
        sys.exit(f"--device {arg!r}: a name may only use letters, digits, _ . -")
    return name, resolve_device(dev)


# ----------------------------------------------------------------- --serve (rov_gui)

def shrink(image, w, h):
    """Scaled down to fit w x h, aspect kept, never enlarged and not letterboxed
    (the station centres it). Always a new array: the capture thread's frame is
    not held on to."""
    ih, iw = image.shape[:2]
    s = min(w / iw, h / ih, 1.0)
    nw, nh = max(1, round(iw * s)), max(1, round(ih * s))
    if (nw, nh) == (iw, ih):
        return image.copy()
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)


class Sender:
    """The one writer of the protocol stream. Messages go out in order; preview
    frames are conflated per camera (only the newest waits). So a station that
    reads slowly costs it previews -- never a log line -- and nothing here, least
    of all a camera or a recording, ever waits on the station."""

    def __init__(self, stream):
        self._stream = stream
        self._cond = threading.Condition()
        self._msgs = deque()
        self._frames = {}                 # label -> (msg, payload): the newest only
        self._closed = False
        self.broken = False               # the station stopped reading
        self._thread = threading.Thread(target=self._run, name="sender", daemon=True)
        self._thread.start()

    def send(self, msg):
        with self._cond:
            if not self._closed:
                self._msgs.append(msg)
                self._cond.notify()

    def frame(self, label, msg, payload):
        with self._cond:
            if not self._closed:
                self._frames[label] = (msg, payload)
                self._cond.notify()

    def close(self, timeout=5):
        """Send the messages already queued (not the frames), then stop."""
        with self._cond:
            self._closed = True
            self._frames.clear()
            self._cond.notify()
        self._thread.join(timeout)

    def _run(self):
        while True:
            with self._cond:
                while not (self._msgs or self._frames or self._closed):
                    self._cond.wait()
                if self._msgs:
                    msg, payload = self._msgs.popleft(), b""
                elif self._frames:
                    msg, payload = self._frames.pop(next(iter(self._frames)))
                else:                     # closed, and everything is out
                    return
            if self.broken:
                continue
            try:
                self._stream.write(protocol.pack(msg, payload))
                self._stream.flush()
            except (OSError, ValueError):
                self.broken = True


def take_stdout():
    """Keep the protocol stream for ourselves and point fd 1 at stderr, so a
    stray print -- ours, OpenCV's, ffmpeg's -- cannot land inside a frame."""
    sys.stdout.flush()
    stream = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    return stream


def serve(slots, sender, hello, new_camera=None, discover=None):
    """The station's side of the cameras: obey stdin, report on stdout.

    `slots` is [[label, device, Camera]], with device and Camera None for an
    EMPTY slot. With `discover` (-> pool camera devices plugged in now) and
    `new_camera(label, device)`, a camera plugged in later fills the first
    empty slot (fill_slots); without them the slots stay as they are.

    Recording is on while the station says so, for every camera it has not
    left out ("arm" off). A camera left out or let back in mid-recording stops
    its file, or starts one at its next frame; so does one that arrives in an
    empty slot. Ends at "quit" or when stdin closes -- which is also what a
    dying station looks like -- with every file finished first."""
    by_label = {slot[0]: slot for slot in slots}
    state = dict(rec=False, outdir=None, armed={slot[0]: True for slot in slots},
                 preview={}, hz=10.0)
    done = threading.Event()
    lock = threading.Lock()        # a slot filling vs. a command changing REC

    def cams():
        return [slot[2] for slot in slots if slot[2] is not None]

    def slot_status(slot):
        label, dev, cam = slot
        if cam is None:
            return dict(label=label, empty=True, device=None, connected=False,
                        status="empty slot -- plug a pool camera into a free USB port",
                        fps=0.0, size=None, usb=None, armed=state["armed"][label],
                        recording=False, rec_s=0.0, rec_mb=0.0, rec_file=None,
                        rec_error=None)
        rec = cam.recorder
        try:
            mb = rec.path.stat().st_size / 1e6 if rec is not None else 0.0
        except OSError:
            mb = 0.0
        status = cam.status
        stuck = cam.stuck_s
        if stuck > STUCK_S:
            status = (f"stuck {stuck:.0f} s getting a first frame -- USB/driver fault? "
                      "replug; if it persists, reboot (journalctl -k)")
            if label not in warned_stuck:
                warned_stuck.add(label)
                say(f"[{label}] {status}", level="error")
        elif label in warned_stuck and cam.connected:
            warned_stuck.discard(label)
        return dict(label=label, empty=False, device=dev, connected=cam.connected,
                    status=status, fps=round(cam.fps_now(), 2),
                    size=list(cam.size) if cam.size else None, usb=cam.usb,
                    armed=state["armed"][label], recording=rec is not None,
                    rec_s=round(rec.seconds, 1) if rec is not None else 0.0,
                    rec_mb=round(mb, 1), rec_file=str(rec.path) if rec is not None else None,
                    rec_error=cam.rec_error)

    def pump():
        """Previews at the asked rate, status twice a second, a disk check and
        the empty-slot scan. Never dies quietly: a silent pump looks exactly
        like a wedged process."""
        said = set()
        while not done.wait(1.0 / state["hz"]):
            try:
                pump_once()
            except Exception as ex:                       # noqa: BLE001
                if repr(ex) not in said:
                    said.add(repr(ex))
                    say(f"status/preview error: {ex!r}", level="error")

    last_seq, warned_stuck = {}, set()
    clock = dict(status=0.0, disk=0.0, look=0.0, nodes=None, settled=0.0, scanned=True)

    def plug_in(now):
        """Fill empty slots with cameras plugged in since the last look. Acts
        once the device list has held still for SETTLE_S: the kernel's
        /dev/videoN comes first and udev's by-path link a moment later, and
        the link -- the USB port -- is what a slot is tied to."""
        if discover is None or new_camera is None or now < clock["look"]:
            return
        clock["look"] = now + 0.5
        if all(slot[2] is not None for slot in slots):
            return
        nodes = tuple(sorted(glob.glob("/dev/video*") + glob.glob("/dev/v4l/by-path/*")))
        if nodes != clock["nodes"]:
            clock.update(nodes=nodes, settled=now + SETTLE_S, scanned=False)
            return
        if clock["scanned"] or now < clock["settled"]:
            return
        clock["scanned"] = True
        found = discover()
        with lock:
            for i, dev in fill_slots([(slot[0], slot[1]) for slot in slots], found):
                slot = slots[i]
                cam = new_camera(slot[0], dev)
                slot[1], slot[2] = dev, cam
                say(f"[{slot[0]}] camera plugged in: {cam.model} on {cam.usb} ({dev})")
                if state["rec"] and state["armed"][slot[0]]:
                    cam.start_recording(None, state["outdir"])   # from its first frame

    def pump_once():
        now = time.monotonic()
        if state["rec"] and now >= clock["disk"]:
            clock["disk"] = now + DISK_CHECK_S
            try:
                where = Path(state["outdir"])     # (made at a camera's first frame)
                where = next(p for p in (where, *where.parents) if p.exists())
                free = shutil.disk_usage(where).free / 1e9
            except (OSError, StopIteration):
                free = None
            if free is not None and free < DISK_STOP_GB:
                # The station's own records share this disk; they come first.
                with lock:
                    state["rec"] = False
                    set_recording(cams(), False)
                say(f"recording STOPPED: {free:.1f} GB free on the recording disk "
                    f"(< {DISK_STOP_GB:g} GB)", level="error")
        plug_in(now)
        for label, (w, h) in list(state["preview"].items()):
            cam = by_label[label][2]
            item = cam.latest() if cam is not None and cam.connected else None
            if item is None or last_seq.get(label) == item[2]:
                continue
            last_seq[label] = item[2]
            small = shrink(item[0], w, h)
            sender.frame(label, dict(t="frame", cam=label, w=small.shape[1],
                                     h=small.shape[0], seq=item[2], ts=item[1]),
                         small.tobytes())
        if time.monotonic() >= clock["status"]:
            clock["status"] = time.monotonic() + 0.5
            sender.send(dict(t="status", session=state["rec"],
                             outdir=state["outdir"],
                             cams=[slot_status(slot) for slot in slots]))

    def apply(msg):
        cmd = msg["cmd"]
        if cmd == "record":
            if not msg.get("on"):
                state["rec"] = False
                set_recording(cams(), False)
                return
            if not msg.get("outdir"):
                say("record: no output folder given -- not recording", level="error")
                return
            t0 = float(msg.get("t0") or time.monotonic())   # the station's REC moment
            state.update(rec=True, outdir=str(msg["outdir"]))
            for label, _dev, cam in slots:
                if cam is None:
                    continue
                if state["armed"][label]:
                    cam.start_recording(t0, state["outdir"])
                else:
                    cam.stop_recording()
        elif cmd == "arm":
            slot = by_label.get(msg.get("cam"))
            if slot is None:
                return
            label, on = slot[0], bool(msg.get("on"))
            if on == state["armed"][label]:
                return
            state["armed"][label] = on
            if state["rec"] and slot[2] is not None:
                if on:
                    slot[2].start_recording(None, state["outdir"])  # from its next frame
                else:
                    slot[2].stop_recording()
            say(f"[{label}] {'records with' if on else 'left out of'} REC")
        elif cmd == "preview":
            sizes = {}
            for label, wh in (msg.get("sizes") or {}).items():
                try:
                    w, h = (int(v) for v in wh)
                except (TypeError, ValueError):
                    continue
                if label in by_label and w >= 16 and h >= 16:
                    sizes[label] = (min(w, 1920), min(h, 1080))
            state["preview"] = sizes
            try:
                state["hz"] = min(30.0, max(1.0, float(msg.get("hz", state["hz"]))))
            except (TypeError, ValueError):
                pass

    sender.send(hello)
    threading.Thread(target=pump, name="pump", daemon=True).start()
    try:
        for line in sys.stdin.buffer:      # blocks; the loop ends when stdin closes
            msg = protocol.parse_command(line)
            if msg is None:
                continue
            if msg["cmd"] == "quit":
                break
            with lock:
                apply(msg)
    finally:
        # A second TERM/HUP must not cut this cleanup short (it would leave a
        # sidecar saying complete: false over a file ffmpeg did finish).
        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        done.set()
        with lock:                         # no slot fills from here on
            final = cams()
        if any(c.recorder for c in final):
            say("finishing recordings ...")
        set_recording(final, False)
        for cam in final:
            cam.close()
        sender.send(dict(t="bye"))
        sender.close()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true",
                   help="list USB cameras and which USB controller each is on, and exit")
    p.add_argument("--watch", action="store_true",
                   help="with --list: keep listing, redrawn on every change -- plug the "
                        "cameras in one at a time and see where each one lands")
    p.add_argument("--device", action="append", metavar="[NAME=]DEV",
                   help="camera to use: N, /dev/videoN, or a /dev/v4l/by-path link, "
                        "optionally named; repeat for several (default: every pool "
                        "camera found, named cam0, cam1, ...)")
    p.add_argument("--size", type=parse_size, default=(1920, 1080), metavar="WxH",
                   help="capture and recording size (default 1920x1080)")
    p.add_argument("--fps", type=float, default=30.0,
                   help="frame rate (default 30; this camera does 25/30/60)")
    p.add_argument("--outdir", type=Path, default=None,
                   help="where recordings go (default: data/YYYYMMDD/MMDD_HHMMSS_poolcam/ "
                        "in this repo, made when recording first starts)")
    p.add_argument("--record", action="store_true", help="start recording right away")
    p.add_argument("--duration", type=float, default=0, metavar="SEC",
                   help="quit after SEC seconds (of recording, with --record)")
    p.add_argument("--no-preview", action="store_true",
                   help="no window (e.g. over ssh): records from the start, until Ctrl+C "
                        "or --duration (for long runs over ssh, use tmux or nohup)")
    p.add_argument("--encoder", choices=["auto", "nvenc", "x264"], default="auto",
                   help="auto = GPU (h264_nvenc) if it works, else CPU (libx264)")
    p.add_argument("--quality", type=int, default=None,
                   help="lower = better and bigger. Default: CQ 28 for nvenc, CRF 23 for "
                        "x264 -- the same look and ~2.5 GB/hour at 1080p30 (the same "
                        "number does not mean the same quality on the two encoders)")
    p.add_argument("--preview-width", type=int, default=1280, metavar="PX")
    p.add_argument("--serve", action="store_true",
                   help="machine mode for rov_gui: protocol on stdin/stdout, no window "
                        "(see the top of this file)")
    p.add_argument("--power-line", choices=list(POWER_LINE), default="60",
                   help="anti-flicker for the room lights' mains frequency (default 60: "
                        "KR/US; 50 makes rolling horizontal bands under 60 Hz lights)")
    p.add_argument("--model", action="append", dest="models", metavar="NAME",
                   help="V4L2 card name (substring) that counts as a pool camera; "
                        f"repeat for several (default: {', '.join(POOL_MODELS)})")
    p.add_argument("--slots", type=int, default=0, metavar="N",
                   help="--serve: always N camera slots; empty ones are filled by "
                        "pool cameras plugged in later (see the top of this file)")
    p.add_argument("--fake", type=int, default=0, metavar="N",
                   help="N synthetic test-pattern cameras instead of hardware "
                        "(tests; rov_gui --source demo)")
    args = p.parse_args()
    args.models = tuple(args.models or POOL_MODELS)

    sender = None
    if args.serve:
        global _say_hook
        sender = Sender(take_stdout())
        _say_hook = lambda level, msg: sender.send(dict(t="log", level=level, msg=msg))
    try:
        run(args, sender)
    except SystemExit as ex:
        # A refusal (no camera, bad --device, no ffmpeg) is a message for the
        # operator -- in --serve mode the station's log is where they are looking.
        if sender is not None:
            if isinstance(ex.code, str):
                say(ex.code, level="error")
            sender.send(dict(t="bye"))
            sender.close()
        raise


def run(args, sender):
    if args.list:
        if not args.watch:
            print(camera_report(args.models))
            return
        # --watch: redraw on every change, so cameras can be plugged in one
        # at a time and each one's controller read off as it appears.
        last = None
        try:
            while True:
                report = camera_report(args.models)
                if report != last:
                    last = report
                    print(f"\n===== {time.strftime('%H:%M:%S')}  (Ctrl+C to stop)\n{report}",
                          flush=True)
                time.sleep(1.0)
        except KeyboardInterrupt:
            return

    if args.fake:
        specs = [(f"cam{i}", f"{FAKE_PREFIX}{i}") for i in range(args.fake)]
    elif args.device:
        specs = [parse_device(d, i) for i, d in enumerate(args.device)]
    else:
        specs = [(f"cam{i}", dev) for i, dev in enumerate(pool_devices(args.models))]
    if not specs and not (args.serve and args.slots):   # empty slots are fine
        sys.exit(f"no pool camera ({', '.join(args.models)}) found -- plugged in? "
                 f"(python pool_cam/pool_cam.py --list shows every USB camera)")
    if len({name for name, _ in specs}) < len(specs):
        sys.exit("camera names must be unique")
    for name, dev in specs:
        if dev.startswith(FAKE_PREFIX):
            continue
        node = os.path.realpath(dev)
        if not os.path.exists(node):
            if "/by-path/" in dev:        # a fixed USB port with nothing in it yet
                continue
            sys.exit(f"no such device: {dev}")
        info = query_cap(node)
        if info is None:
            sys.exit(f"cannot open {dev}")
        if not info[3]:
            sys.exit(f"{dev} is not a video capture node (each UVC camera also has a "
                     f"metadata node; see python pool_cam/pool_cam.py --list)")

    if args.serve:
        args.no_preview, args.record = True, False   # the station says when
    elif args.no_preview:
        args.record = True                # without a window there is no key to start one
    elif not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        sys.exit("no display for the preview window -- use --no-preview")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        sys.exit("ffmpeg not found on PATH")
    encoder = pick_encoder(ffmpeg, args.encoder)
    if args.quality is None:
        args.quality = DEFAULT_QUALITY[encoder]

    def rec_dir():
        """Where a recording starting now goes. Asked at each start, so a session
        that never records leaves no folder behind (runstore: a folder is a run)."""
        return args.outdir or runstore.run_dir(DATA_ROOT, kind=RUN_KIND)

    # OpenMP builds of OpenCV (the umi env's) leave every core busy-waiting after
    # each cv2.resize: measured 12 cores for one preview. One thread is plenty.
    cv2.setNumThreads(1)
    quiet_opencv()
    if not args.no_preview:
        # Before any camera or recording starts: if the window can't open, Qt
        # aborts the whole process, and better now than in the middle of a file.
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
                        | cv2.WINDOW_GUI_NORMAL)
    codec = (f"{'h264_nvenc (GPU), CQ' if encoder == 'nvenc' else 'libx264 (CPU), CRF'} "
             f"{args.quality}")
    say(f"{len(specs)} camera(s); encoder: {codec} via {ffmpeg}; recordings -> "
        + ("the station's run folder" if args.serve else
           str(args.outdir or f"{DATA_ROOT}/YYYYMMDD/MMDD_HHMMSS_{RUN_KIND}/")))

    if args.serve:
        # Below the station in the scheduler (ffmpeg inherits it): when cores
        # are short, the controller and the policy get them, and the cameras
        # drop frames -- which the sidecars count.
        try:
            os.nice(10)
        except OSError:
            pass
        if encoder == "x264" and args.encoder == "auto":
            say(f"h264_nvenc unavailable: encoding {len(specs)} camera(s) on the CPU "
                "(libx264), which competes with the station for cores", level="warn")
        # Its own session (the station starts it so), so a terminal Ctrl+C is
        # the station's to handle; a TERM/HUP sent here still finishes the files.
        def leave(signum, frame):
            raise SystemExit(128 + signum)
        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, leave)
        def new_camera(name, dev):
            return Camera(name, dev, args.size, args.fps, None, ffmpeg, encoder,
                          args.quality, args.power_line).start()

        labels = [name for name, _ in specs]
        n = 0
        while len(labels) < args.slots:            # the empty slots' names
            if f"cam{n}" not in labels:
                labels.append(f"cam{n}")
            n += 1
        slots = ([[name, dev, new_camera(name, dev)] for name, dev in specs]
                 + [[name, None, None] for name in labels[len(specs):]])
        hello = dict(t="hello", pid=os.getpid(), encoder=encoder, codec=codec,
                     size=list(args.size), fps=args.fps, models=list(args.models),
                     cams=[dict(label=name, device=dev,
                                usb=cam.usb if cam else None,
                                model=cam.model if cam else None)
                           for name, dev, cam in slots])
        if len(slots) > len(specs):
            say(f"{len(slots) - len(specs)} empty slot(s): "
                + ", ".join(labels[len(specs):])
                + ("" if not args.fake else " (synthetic cameras: no plug-in scan)"))
        # Synthetic runs never scan: a test must not pick up the real camera
        # that happens to be plugged into this desktop.
        serve(slots, sender, hello, new_camera,
              None if args.fake else (lambda: pool_devices(args.models)))
        return

    stop = Stop()

    def on_signal(signum, frame):
        if stop.requested:   # second Ctrl+C: leave now -- ffmpeg still closes the files
            os._exit(130)
        stop.requested = True

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        if signal.getsignal(sig) != signal.SIG_IGN:    # e.g. SIGHUP under nohup
            signal.signal(sig, on_signal)

    cams = [Camera(name, dev, args.size, args.fps, None, ffmpeg, encoder,
                   args.quality, args.power_line).start() for name, dev in specs]
    try:
        if args.record:
            # Start together once every camera streams (or after 5 s regardless),
            # so the files share one t0 and don't open on a frozen first frame.
            give_up = time.monotonic() + 5
            while (not all(c.connected for c in cams) and time.monotonic() < give_up
                   and not stop.requested):
                time.sleep(0.05)
            if not stop.requested:
                set_recording(cams, True, rec_dir())
        deadline = time.monotonic() + args.duration if args.duration > 0 else None
        if args.no_preview:
            run_headless(cams, stop, deadline)
        else:
            run_preview(cams, args, stop, deadline, rec_dir)
    finally:
        if any(c.recorder for c in cams):
            say("finishing recordings ...")
        for cam in cams:
            cam.stop_recording()
        for cam in cams:
            cam.close()


if __name__ == "__main__":
    main()
