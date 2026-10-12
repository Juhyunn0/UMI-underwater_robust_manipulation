# c3_collect — Collecting underwater RGB-D / visual SLAM datasets

> **English translation.** Source of truth: [`README_DATASET.md`](README_DATASET.md) (Korean). Snapshot of 2026-10-12, source blob `5585283596`; later edits to the original are not reflected here.
>
> Provenance tags are kept as in the original: `[측정 …]` measured (artifact cited) · `[유도 …]` derived · `[예측 …]` predicted, not measured · `[스펙 …]` vendor spec · `실측` measured · `미실측` not measured · `산출물 없음` no artifact found · `예시/유도값` example or derived value.

Grabs the C3 camera over a direct DepthAI connection (bypassing Madrona), records the BlueROV2's MAVLink
telemetry at the same time, and writes **everything onto a single timeline** in the TUM RGB-D layout.

The direct camera connection itself (latency 3000 ms → 35 ms) and the measured evidence behind it are in  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
[README.md](README.en.md). This document covers only the **dataset collection** part.

> Each dataset folder's `metadata.txt` is written **in English**. It holds all the depth_scale, timestamp and
> IMU caveats, so you can hand the whole folder to a colleague as is.

## 0. Before you start: check whether Madrona needs to be stopped

An OAK device allows **only one pipeline**. If Madrona is holding the camera, you get
`Could not connect to device with mx_id`. However, **the camera can be idle (`X_LINK_BOOTLOADER`)
even while the container is up** — in that case it connects without touching anything.

```bash
./c3 discover --probe                      # tells you explicitly: free / OWNED
```

> The examples below are written in the form `$V c3_camera/<tool>.py`, but `./c3 <tool>` does the same
> thing (`V=~/.venvs/c3-depthai/bin/python`). And instead of typing the flags every time, you can use a
> **profile**:
>
> ```bash
> ./c3 collect --profile research_near            # c3_camera/profiles/*.yaml
> ./c3 collect --profile research_near --dry-run  # check the values + their provenance
> ```

If it shows `OWNED`:

```bash
$V -m c3_camera.madrona stop --yes         # disable the BlueOS extension + wait for the camera to be released
# ... always restore it after capturing
$V -m c3_camera.madrona start
```

`stop` does not work without `--yes`. A BlueOS disable is a persistent setting, so it stays off even
after a reboot, and BlueOS/Cockpit keep showing no video. If you prefer the GUI, you can also turn it off on the BlueOS
Extensions page.

`c3_collect.py` detects this situation and prints the commands above as they are.

## 1. Using it at the same time as QGroundControl — must read

This problem was actually measured. BlueOS **pushes** MAVLink to `udpout → 192.168.2.1:14550`, and
QGroundControl holds that port on the desktop. **Two processes cannot both receive the same unicast UDP
datagram.**

- If you bind 14550 twice with `SO_REUSEPORT`, Linux **distributes the datagrams instead of
  duplicating them** → the recorder steals about half of the QGC packets, and the pilot link silently degrades.
  **Don't do this.**

Three options, in order of recommendation:

### (1) Add a second endpoint in BlueOS — recommended

```bash
$V -m c3_camera.blueos_endpoint list                    # check the current routing
$V -m c3_camera.blueos_endpoint add --port 14551 --yes   # udpout → 192.168.2.1:14551
$V c3_camera/c3_collect.py --mavlink udpin:0.0.0.0:14551 --mode research
```

You receive the full stream at the highest rate. `--persistent` is the default, so you only have to do this once.
However, do it **before the dive, because the QGC link can drop momentarily while the mavlink-router config
is reloaded**. (`c3_collect.py --setup-endpoint` can also do it for you.)

### (2) MAVLink2Rest — vehicle config unchanged

```bash
$V c3_camera/c3_collect.py --mavlink-transport rest --mode research
```

Polls the MAVLink2Rest (port 6040) that is already running on BlueOS. **It does not touch the vehicle config
at all, and there is no conflict with QGC either.** In exchange, because it is built to serve the *latest value* of each message,
it **samples** rather than capturing every message. That is enough for attitude/pressure/battery context,
but it is not a substitute for a high-rate IMU stream. (The IMU for VIO is the camera's
BNO086 anyway, so in practice this is not a big loss.)

### (3) Without telemetry

```bash
$V c3_camera/c3_collect.py --mavlink-transport none
```

## 2. How to use it

```bash
# compute the settings only (does not touch the camera, no 12 s boot either)
$V c3_camera/c3_collect.py --mode research --dry-run

# long exploration dive: mp4 + telemetry only
$V c3_camera/c3_collect.py --mode review

# SLAM dataset
$V c3_camera/c3_collect.py --mode research

# 20 fps RGB-D: RGB as on-device H.264, depth as lossless uint16 PNG
$V c3_camera/c3_collect.py --mode research --mavlink-transport rest \
    --streams color,depth --color-encode h264 --video-bitrate-kbps 4000 \
    --isp-scale 1/4 --depth-size 480x270 --fps 20 \
    --extract-encoded-rgb

# verify after recording (always, before handing it to a colleague)
$V c3_camera/c3_dataset_check.py c3_camera/datasets/dataset_20260729_161500
```

**SPACE** starts/stops recording → you can dive first and capture only the segment you want.
`q` quits; `Ctrl-C` is safe too (writer flush → mp4 finalize → metadata written).
Headless: `--no-display --record-now`, and stop with `Ctrl-C`.

### H.264/H.265 RGB + lossless depth at 20 fps

The research default of `c3_collect.py` sends RGB as raw NV12. Now, if you pass
`--color-encode h264|h265`, the C3's `VideoEncoder` hardware-compresses only the RGB, and depth is
**saved losslessly as uint16 millimetre PNG**, exactly as before.
The default for `--video-bitrate-kbps` is 4000, and by default a keyframe goes in every 1 s.

H.264/H.265 are inter-frame codecs, so during capture it records to `rgb.h264`/`rgb.h265` plus
`rgb_video.csv`, which holds the per-frame capture times. With `--extract-encoded-rgb`,
it uses FFmpeg after exit to create `rgb/<timestamp>.png`, so the existing TUM index points to them
directly. You can also do this manually later:

```bash
$V c3_camera/c3_extract_rgb_video.py c3_camera/datasets/dataset_YYYYMMDD_HHMMSS
```

**At 20 fps, `--streams color,depth` matters.** If the two raw left/right images are also
sent over XLink, those alone are about 82 Mbit/s, and with the 480x270 depth at about 41 Mbit/s added
on top, it already exceeds the PoE limit even if the RGB were compressed completely for free. Even without
sending left/right, StereoDepth keeps using the two mono cameras inside the device, so depth
works as before. What you give up is only the raw mono files for recomputing the stereo disparity later,
and RGB-D SLAM is not affected.

The budget for the setting that sends 480x270 depth @20 + H.264 4 Mbit/s + RGB only is about
**45 Mbit/s**. That leaves plenty of margin for the IMU. H.264/H.265 RGB is lossy, so
do not use it for photometric calibration data, and even for feature-based SLAM, if the bitrate is
lowered too far, block artifacts can damage the feature points.

### The two modes

| | review | research |
|---|---|---|
| Purpose | long exploration dive | SLAM dataset |
| Video | `rgb.mp4` only | 4 lossless PNG streams **+** `rgb.mp4` for reference |
| Default settings | 960x540 MJPEG @15 | 480x270 lossless @8, depth 1:1 |
| Telemetry | CSV | CSV |
| Storage | about 0.5 GB/hour | about 21 GB/hour [UNVERIFIED: 예시/유도값 — docs/MEASUREMENT_AUDIT.md] |

## 3. Measured performance (research mode)

Default settings, 480x270 colour+depth + 640x400 stereo @8 fps, measured over 20 s:  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]

| Item | Value |
|---|---|
| Frames | 8.0 fps requested → **8.0 measured, 0 dropped** |
| Images | 4 lossless PNGs per frame |
| Link | 62 Mbit/s (69% of the measured 91.5 ceiling) |
| Disk | about 11–13 MB/s (about 21 GB/hour) [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| Latency | colour p50 88 ms, depth 89 ms [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| Camera IMU | 200 Hz requested → **about 130 Hz measured** (see below) [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |

**The bottleneck is the PoE link, not the disk.** The brief raised a concern about the disk, but this
NVMe is tens of times faster than the ~11 MB/s the link can deliver. PNG encoding, too, takes 17 ms per frame
(all 4 images combined), so 3 writer threads are more than enough. The fps upper bound for the 4 lossless streams
is set by the link:

| Resolution (colour=depth, mono 640x400) | Link upper-bound fps |
|---|---|
| 480x270 | 11.8 |
| 640x360 | 8.7 [UNVERIFIED: 예시/유도값 — docs/MEASUREMENT_AUDIT.md] |
| 960x540 | 4.9 [UNVERIFIED: 예시/유도값 — docs/MEASUREMENT_AUDIT.md] |

If you need a higher fps, drop the stereo with `--streams color,depth` (14 fps),  [UNVERIFIED: 예시/유도값 — docs/MEASUREMENT_AUDIT.md]
or lower the resolution.

## 4. Output format

```
dataset_YYYYMMDD_HHMMSS/
├── rgb/     <timestamp>.png     lossless 8-bit BGR
├── depth/   <timestamp>.png     lossless 16-bit, millimetres, 1:1 aligned with RGB
├── left/    <timestamp>.png     lossless mono (raw)
├── right/   <timestamp>.png     lossless mono (raw)
├── rgb.txt          # timestamp filename   (TUM)
├── depth.txt        # timestamp filename   (TUM)
├── associations.txt rgb_ts rgb_file depth_ts depth_file
├── frames.csv       frame index + per-stream device/host time + skew + latency
├── imu_camera.csv   C3 BNO086  ← use this one for VIO
├── imu_rov.csv      ArduSub IMU (MAVLink)
├── telemetry.csv    attitude, pressure/depth, battery, servo, mode
├── control.csv      pilot input (RC_CHANNELS / MANUAL_CONTROL)
├── calibration.json on-device factory calibration
├── orbslam3_rgbd.yaml  draft ORB-SLAM3 config generated from the actual intrinsics
├── rgb.mp4          preview for reference (lossy compression, not data)
└── metadata.txt/.json  depth_scale, clock and IMU caveats + preflight at start time (English)
```

- **Depth never goes into video.** 16-bit PNG, lossless (`--record-depth-format
  npy` also works). Lossy compression ruins the millimetre values.
- **left/right are lossless PNG too.** A colleague has to be able to recompute the disparity or run
  stereo SLAM, and compression artifacts ruin the matching.
- Colour transmission is NV12 by default (1.5 B/px, 4:2:0 chroma). PNG storage is lossless, but
  **chroma is subsampled at the transmission stage** — luma is at full resolution, so feature-based SLAM
  is unaffected. If you need full chroma, `--color-wire bgr` (2× the bandwidth).

### depth_scale — correcting the number in the brief

The brief says "depth_scale is 5000.0 for millimetres", but **this is wrong.**
If used as is, every depth in the colleague's SLAM comes out **5× smaller**, and because it looks
plausible rather than broken, it does not even get noticed.

```
metres = png_pixel_value / 1000.0        ← our data (uint16 millimetres)
```

`5000` is the value TUM's own sequences use because they stored their Kinect data scaled that way;
it is not a convention. The parameter name per tool, and our value:

| Tool | Parameter | Value |
|---|---|---|
| ORB-SLAM3 | `RGBD.DepthMapFactor` | **1000.0** |
| Open3D | `depth_scale` (as a divisor) | **1000.0** |
| RTAB-Map | recognizes 16-bit mm by default, no separate parameter needed | — |

The generated `orbslam3_rgbd.yaml` already has it set to 1000.0.

One more thing: **`depth == 0` is not "distance 0" but "measurement failed"**. Mask it out.
And when reading, **`cv2.IMREAD_UNCHANGED` is mandatory** — leave it out and OpenCV silently reads
the image as 8-bit, and the values are off by 257×.

## 5. Timestamps — what to explain to your colleague

**Measurement shows that on this host `dai.Clock.now()` and `time.monotonic()` are the same clock
(CLOCK_MONOTONIC) and agree to within 8 microseconds.** Therefore camera frames,  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
camera IMU and MAVLink receive times are **already on one timeline, without any conversion**.

Every value written to the files is in **Unix epoch seconds, 6 decimal places**, converted with a constant
offset measured once at session start (`clock_offset_...` in `metadata.txt`):

```
t_unix = t_monotonic + clock_offset_monotonic_to_unix
```

`frames.csv` also keeps the raw monotonic values, so no information is lost.

**Be sure to tell them about one asymmetry:**

| | Meaning of the timestamp |
|---|---|
| `rgb/`, `depth/`, `left/`, `right/`, `imu_camera.csv` | **Capture time** (stamped on the device and kept in the host domain via timesync) |
| `imu_rov.csv`, `telemetry.csv`, `control.csv` | **Host arrival time** — later than the actual time of measurement by the link/routing latency (a few ms on this Ethernet path) |

If you need better precision than a few ms, each MAVLink row has ArduSub's own `time_boot_ms`, so
you can regress it against `t_unix` to recover the autopilot clock.

## 6. IMU — which one to use, and two warnings

**The C3 has an onboard IMU: BNO086** (confirmed with `getConnectedIMU()`, firmware
3.9.9). It is rigidly attached to the camera and its timestamps are on the same clock as the images, so
**this is the right one to use for VIO.** The ROV's Navigator IMU is tens of cm away and  [UNVERIFIED: 예시/유도값 — docs/MEASUREMENT_AUDIT.md]
arrives from a different board over UDP, stamped only with its arrival time — it is for context and cross-checking.

### Measured rate rules

| Setting | Measured |
|---|---|
| accel+gyro @500 requested | **486 Hz** |
| accel+gyro @200 | 194 Hz [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| accel+gyro @200 **+ rotvec @100** | 97 Hz (all halved) [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| accel+gyro @200 **+ mag @100** | 97 Hz (all halved) [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| accel+gyro @200 **+ rotvec @200** | **194 Hz** (no loss) [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |

**Rule 1: the rate of the slowest of the enabled sensors applies to all of them.** So
`--imu-rotation-vector` is free because it is requested at the same rate, but
`--imu-magnetometer` is capped at 100 Hz, so it **ties the whole IMU down to 100 Hz.**

**Rule 2: batching decides the loss.** While recording the 4 lossless streams at the same time:

| `--imu-batch` | Result of a 200 Hz request |
|---|---|
| 1 | 130 Hz received, **35% lost** [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 10 (default) | **199.5 Hz received, 0 lost** |

The cause was not bandwidth but **per-message XLink overhead**. With 1 message per 1 report,
it cannot keep up while competing with the images. Batch 1 has minimal latency, so it suits the live
view, but for a dataset completeness is what matters — **a hole in the IMU stream breaks VIO
preintegration.** That is why the `c3_collect.py` default is 10.

`metadata.txt` records **the measured values, not the requested values**, so the colleague does not
have to take it on trust that the request was honoured.

### ⚠ Warning 1: the IMU-camera extrinsic is not on the device

`getImuToCameraExtrinsics()` → `IMU calibration data is not available on device yet.`
The factory calibration covers only the cameras, so **T_imu_cam is unknown**. To run
VIO, the colleague has to calibrate it with a Kalibr-type tool or measure it mechanically.
(The camera intrinsics and stereo extrinsics are fine.)

### ⚠ Warning 2: the accelerometer scale is off

At rest, |a| = **11.8 m/s²** (about 20% high relative to gravity 9.81). raw/calibrated
are identical, and the accuracy flag is UNRELIABLE/MEDIUM. **Do not feed it into VIO without an IMU
intrinsic calibration.** The accuracy field is kept for every sample, so this does not stay hidden.

The axis orientation (relative to the camera's optical axis) is not in the vendor documentation either, and it was
not confirmed this time. Treat the rotation as unknown too, along with the translation.

All three of these are written into `metadata.txt`, because VIO results built on an uncalibrated IMU
fail in a way that looks like an "algorithm problem".

### What state the rig was in when recording started (`preflight`)

`c3_collect.py` runs readiness checks before connecting (`c3_camera/preflight.py`) and stores
the whole report under the `preflight` key of `metadata.json`. Checks that were not clean are
also written, one line each, into the `== preflight ==` section of `metadata.txt`:

```
== preflight ==
These readiness checks were NOT clean when this run started. They are
the first thing to suspect if something in this dataset looks wrong:
  [WARN] telemetry path: nothing arrived on UDP 0.0.0.0:14551 in 3 s — ...
```

The reason this exists is simple — months later, when only the folder is left, the answer to "why is
`telemetry.csv` empty?" is usually that line, and no other file records it.

## 7. Control — nothing is sent for now

This program **transmits nothing** (exception: SET_MESSAGE_INTERVAL when you turn on
`--mavlink-set-rates` yourself). Pilot with QGroundControl; the pilot input is
read from MAVLink and recorded in `control.csv`.

- It sends no heartbeat, so ArduSub's GCS failsafe cannot mistake us for the piloting
  GCS.
- It uses a source_system different from QGC's (255).

The place to attach a joystick or a vision policy later has been set aside as an interface in
[control.py](control.py) (`ControlCommand`, `ControlSource`, `MavlinkCommandSink`). For now it is
deliberately `NotImplementedError` — on a real vehicle, a control path that only half works is
more dangerous than having none at all.

**In every case, ArduSub is the flight controller.** Stabilization, the depth/attitude loops, arming and
failsafe all stay with ArduSub. The only thing that changes is **who takes over the "MAVLink command
source" role that QGroundControl used to fill**:

```
Now     QGroundControl → MANUAL_CONTROL → ArduSub → thrusters
Next    joystick       → MANUAL_CONTROL → ArduSub → thrusters
Then    vision policy  → MANUAL_CONTROL → ArduSub → thrusters
```

Caution when attaching a command source: **if two sources command at the same time**, ArduSub follows the
packet that arrived last, so the vehicle jitters between the two. Disconnect QGC piloting first.

## 8. Main CLI options

For the full list, `--help`. Camera options override the `--mode` defaults.

```
--mode research|review          --out DIR            --notes "..."
--dry-run                       print only the settings and bandwidth, then exit

--isp-scale 1/4                 downscale colour (field of view kept)
--fps 8                         --streams color,depth,left,right
--color-wire nv12|bgr           --mono-source raw|rectified
--depth-size WxH                (in research the default is the same as colour = 1:1)
--extended                      extended close-range disparity (~30 cm -> ~15 cm, theoretical value)

--imu-rate 200                  --imu-rotation-vector   (free, since it is the same rate)
--imu-magnetometer              (ties everything to 100 Hz)
--imu-calibrated                --no-camera-imu

--mavlink udpin:0.0.0.0:14551   --mavlink-transport udp|rest|none
--mavlink-set-rates             send SET_MESSAGE_INTERVAL (default off = fully passive)
--mavlink-rate 50               --setup-endpoint

--writer-threads 3              --png-compression 1     --no-mp4
--min-free-gb 10                --record-now            --no-display
```

## 9. Troubleshooting

| Symptom | Cause | Action |
|---|---|---|
| `Could not connect to device with mx_id` | Another pipeline holds it (usually Madrona) | `discover_c3.py` → `madrona stop --yes`, wait 5–10 s |
| 12 s to connect | Normal (PoE firmware upload + boot) | A 1-time cost per run |
| Nothing arrives on MAVLink | QGC holds 14550 / no endpoint | `blueos_endpoint add --port 14551 --yes` or `--mavlink-transport rest` |
| fps below the request, with drops | Link ceiling exceeded | Check with `--dry-run`, lower `--fps`/`--isp-scale`, leave out stereo |
| `dropped > 0` (writer) | Disk can't keep up (rare on this machine) | Raise `--writer-threads`, `--png-compression 0` |
| depth mostly 0 | Too little texture / too close | `--extended`, lighting. Check the valid % in the HUD |
| `Qt xcb` error | conda library conflict | After `conda deactivate`, run with the venv's absolute path, or `env -u LD_LIBRARY_PATH ...`, or `--no-display` |
| Run from conda base | base has depthai **3.5.0** (v3), so v2 code breaks about halfway and it may grab the camera | Refused automatically at import. Run via the venv path |

## 10. Files

| File | Role |
|---|---|
| `c3_collect.py` | Main collection program (deliverable 2) |
| `discover_c3.py` | Discovery/diagnostics + IMU presence + calibration dump (deliverable 1) |
| `c3_dataset_check.py` | Recording verification — run it before handing off |
| `dataset.py` | TUM layout writer (thread pool), metadata generation |
| `imu.py` | BNO086 host-side reading + extrinsic lookup |
| `mavlink_log.py` | Passive MAVLink logger |
| `blueos_endpoint.py` | MAVLink endpoint management for coexisting with QGC |
| `control.py` | Seam for future joystick/vision control (inactive for now) |
| `madrona.py` | Madrona extension stop/start |
