# rov_gui — BlueROV2 topside control dashboard (PyQt)

> **English translation.** Source of truth: [`README.md`](README.md) (Korean). Snapshot of 2026-10-12, source blob `8a9f2782b7`; later edits to the original are not reflected here.
>
> Provenance tags are kept as in the original: `[측정 …]` measured (artifact cited) · `[유도 …]` derived · `[예측 …]` predicted, not measured · `[스펙 …]` vendor spec · `실측` measured · `미실측` not measured · `산출물 없음` no artifact found · `예시/유도값` example or derived value.

One screen, no scrolling. It is a topside control station that puts 3 video feeds + telemetry + thrusters +
payload + teleop on a single fixed `QGridLayout`. The panels do not touch the hardware directly
— they only draw what a **backend** has put on the data bus. That is why the same UI runs unchanged on
synthetic data, on a real C3 + ArduSub, and on ROS 2 topics.

```
./c3 gui                             # use this. The wrapper picks the interpreter and cwd
./c3 gui --source hw                 # C3 + ArduSub (preflight before connecting)
./c3 gui --source demo               # synthetic data. Does not open any hardware
```

`./c3 gui` is recommended. There are two reasons.

**1. depthai version.** `--source hw` imports `c3_camera`, which requires **depthai 2.x**, but the
default python on this desktop is conda base (depthai 3.5.0), so merely creating a Pipeline can
grab the camera. The wrapper calls the interpreter by its absolute path.

**2. The GUI now runs in its own env** (since 2026-08-09). The camera tools in `./c3` use `robust`,
but `./c3 gui` uses **`rovgui-pose`** — because the station loads the SAM2 + FoundationPose object
tracker into the same process, and that stack cannot go into `robust`:

| | rovgui-pose (GUI) | robust (sim·acados·SLAM) |
|---|---|---|
| torch / sam2 | 2.11+cu128 / present | absent |
| numpy | **2.4.4** | 1.26.4 (<2 because of gtsam) |
| depthai · cv2 · Qt | 2.32 · 5.0 · 5.15.14 | 2.32 · 4.10 · 5.15.14 |

`rovgui-pose` is **a clone of `oakd` + PyQt5 + depthai 2.32**. The expensive part is FoundationPose's
compiled CUDA extensions (pytorch3d·kaolin·nvdiffrast·its own mycpp/mycuda), and those are already
built in `oakd`, so this direction is overwhelmingly cheaper than putting torch on `robust`.
**numpy must be 2.x** — torch 2.11 is compiled against numpy 2's C ABI, so on 1.26 it fails to
initialize regardless of the metadata.

Verification (2026-08-09): in that env `rov_gui` 37/37 and `c3_camera` 58/58 passed, Qt is the same
5.15.14 on both sides, and the minimum window size is identical too at 986×763 — the one-screen promise holds.

If `rovgui-pose` is missing, the wrapper **falls back to the original interpreter and says so.** The
station works correctly; only `--pose` (object tracking) cannot be used. At that point it also prints how to create it:

```bash
conda create -n rovgui-pose --clone oakd
P=~/miniforge3/envs/rovgui-pose/bin
$P/pip install --force-reinstall --no-deps numpy==2.4.4   # the clone breaks numpy
$P/pip install PyQt5==5.15.11 'depthai~=2.32.0' pymavlink
```

> `conda create --clone` **silently breaks numpy** — the pip metadata says 2.4.4 but the actual
> files become conda's 1.26.4. It does not show in `pip list`; it shows only with `import numpy`.
> Right after cloning, always check with `python -c "import numpy, torch"`.

`./c3 env` shows in two lines which interpreter will be used (`interpreter` = camera tools,
`gui` = station). It can be overridden with `$ROVGUI_PY`.

## Screen layout

```
        header    (full width: source, [STATION | POOL CAMS tabs], 4 status pills, clock, REC, E-STOP)
row 0   C3 MAIN (RGB)   | ROV CAM  | SYSTEM
row 1   (2 col × 2 row) | C3 DEPTH | HEALTH  (2-row span)
row 2   TELEOP | PAYLOAD | PROPULS. | SENSORS
        status bar (UI Hz, frames drawn, conflate count, recording status)
```

The header sits **outside and above** the grid (since 2026-09-30). With `--pool-cams` on, the header gains
tabs and everything below it becomes two pages, [STATION grid | POOL CAMS] — the header (E-STOP·REC·LEAK)
is visible on either tab. Without the flag the screen is the same as before.

The middle panel is chosen with `--panel2`: `rov` (default, the ROV's own RGB camera) / `stereo`
(C3 left mono) / `none`.

**Double-click** a video panel and that feed moves up into the large slot.

### C3 Depth panel — the cursor is the measuring tool

Hover the mouse over the depth panel and **the actual distance of that pixel** shows up next to the cursor (`0.45 m`).
Rather than converting the colour by eye, it reads **the raw uint16 millimetre map that rode the mailbox
together with** that frame on screen — by construction, the value and the picture cannot come from different frames.
Holes (stereo invalid, 0 mm) show as `no data`, because showing 0.00 m would mean the object is touching
the lens.

The colour bar got bigger too (up to half the panel height): besides the end labels 0.3 m / 6 m it has **1·2·4 m
ticks**, and the tick positions use the same linear formula as the colouring (`imaging.depth_to_bgr`) — with a
different formula the legend would lie. While the cursor is reading depth, that value is also shown as a marker
on the colour bar.

### What makes it fit on one screen is the size policy, not the layout

1. **Video widgets do not derive their size from their content.** With `QLabel.setPixmap` the
   sizeHint becomes the pixmap size, and a single 1920×1080 frame pushes the window off the
   screen. Here it is `QSizePolicy.Ignored` + drawing directly in `paintEvent`.
2. **Data panels are Preferred vertically and put a stretch at the end of the body.** They ask for
   only as much as they need and give the leftover height back to the video rows.
3. **There is not a single `QScrollArea`.** Whatever does not fit is squeezed or truncated with `…`. That
   is better than controls hiding where the pilot cannot see them.

Starting with 3 columns gave a minimum window height of **1026 px** — on a 1366×768 laptop the
promise was already broken. The sensor list was split off into a 4th column, so the minimum is now
**1152×749** (including the camera tilt section), and
`test_layout_fits_one_screen_and_has_no_scrollarea` holds that line.

## Pool corner cameras (CCTV) — `--pool-cams` (2026-09-30)

Attaches the USB cameras at the pool's four corners to the station **for monitoring and recording only**. They are not
an input to any controller, estimator or policy, and it is built so that no path by which they could become one even exists (see "Isolation" below).

```
./c3 gui --source hw --pool-cams     # POOL CAMS tab + REC UI also records the cameras
./c3 gui --pool-cams                 # demo: 4 synthetic cameras (no hardware opened)
./c3 cctv                            # cameras only, without ROV·C3: the same 4-slot tab + its own REC button
./c3 cctv --record                   # start recording as soon as all cameras are up
./c3 poolcam --list                  # plugged-in cameras, by-path links, count per USB controller
./c3 poolcam --list --watch          # plug in one at a time and watch live which controller each lands on
./c3 poolcam                         # low-level OpenCV window (for mounting and setting angles)
```

`./c3 cctv` (`rov_gui/cctv.py`, 2026-10-01) is the window to use when no vehicle is connected — the same camera process,
the same tiles and the same AUTO REC switches as the station's POOL CAMS tab, and one press of the REC button (Ctrl+R) is one folder:
`data/YYYYMMDD/MMDD_HHMMSS_poolcam/` (mp4+json per camera, `session.json`, `log.txt`). It is the `_poolcam` kind, so it does not get
mixed into water-run statistics. **It cannot be opened at the same time as the station (`--pool-cams`)** — a camera can be opened by only one process.

> **Do not unplug a camera while it is streaming (especially right after it starts).** On 2026-10-01 this PC's uvcvideo (the RealSense dkms-patched build)
> caused a kernel crash in that race condition, and that USB controller stayed tied up until a reboot (KNOWN_ISSUES). The station
> was unaffected, because its cameras are in a child process.

- **Press REC UI** and they record along with it. They are saved inside **the same run folder** as the UI recording, in
  `pool_<HHMMSS>/`: `<name>_<stamp>.mp4` + `.json` per camera, plus `session.json`
  (which cameras were recording targets and which were left out, the common t0).
- **Per-camera AUTO REC switch** (POOL CAMS tab, top right of each tile): when OFF, that camera does not record
  even when REC UI is pressed (its picture keeps showing). Flip it during a recording and only that camera's file is cut off,
  or a new file starts from the next frame.
- **Always 4 slots** (`config/pool_cams.yaml slots: 4`). Cameras plugged in at startup fill the first slots,
  the rest are **EMPTY SLOT**. Plug a camera in while the station is running and it goes into the first empty slot **without a restart**
  (after `/dev` has been quiet for 1 s — once udev has created the by-path link). A slot, once filled, belongs to that USB **port**:
  plugging back into the same port gives the same slot; a different port means a new camera in another empty slot. Empty slots are left out of the recording and
  are written to `empty_slots` in `session.json`. A camera plugged in during REC records from its first frame if its slot is
  AUTO REC ON.
- **Which cameras count as pool cameras**: they are picked by V4L2 name — `USB RGB Camera` (the first model) and
  `SPCA2650 AV Camera` (the second model, plugged in on 2026-09-30, includes a microphone). To use another model, add it to yaml `models:`.
  USB cameras not on this list (other projects' cameras) are left untouched.
- **Preventing horizontal stripes (light flicker)**: every time a camera is opened, its anti-flicker is set to **60 Hz**
  (`config/pool_cams.yaml power_line`). The cameras power up at 50 Hz, so under lighting on 60 Hz mains, horizontal bands
  crawled across the picture — going 50→60 Hz reduced the bands 6–12× [측정:
  `pool_cam/bench_out/flicker_20261001.txt`, how to measure: `python pool_cam/bench_flicker.py`].
  Unplugging and re-plugging the USB returns a camera to 50 Hz, so it is set again every time. `power_line` is kept in the recording json.
- **Double-click** a tile to show only that camera large; double-click again for the 4-way split.
- Camera names·ports·resolutions are in [`config/pool_cams.yaml`](../config/pool_cams.yaml).
  If left empty, all plugged-in pool cameras are named cam0..cam3 in USB port order, but if one is
  missing at startup the remaining names shift — **once all four are wired, fix the names via by-path**
  (`./c3 poolcam --list` prints the links). `--pool-cam NAME=DEV` (repeatable) replaces the file's list.

### Synchronization

All files start from **a single t0, the moment REC UI is pressed**. Camera frames are stamped with the driver's
CLOCK_MONOTONIC timestamp, and this clock is the same as `time.monotonic()` and **is the same even across
different processes**. So:

- `t0_monotonic` in the camera `.json` — frame k of that file = `t0_monotonic + k/fps` (±3/4 frame;
  dropped frames are filled by repeating the previous frame to keep the time axis: described at the top of pool_cam.py)
- `started_monotonic` in the UI recording sidecar — the same value (a test checks they agree within 1e-5 s:
  `test_window_rec_ui_records_the_pool_cameras_into_the_run_folder`)
- `host_stamp` in nav `fixes.csv` — the same clock

### Isolation — why a child process

The cameras, decoding and the encoder (ffmpeg, NVENC when possible) are owned by the `pool_cam/pool_cam.py --serve` **child
process**. The station sends only "record into this folder from t0" and the preview size, and receives only status, logs and
small preview images (`pool_cam/protocol.py`).

1. **The cameras cannot kill the station.** The four cameras on long USB cables are the equipment in this rig that comes
   unplugged most easily. A native crash in a capture driver kills the process it runs in — in a child, one tab
   dies and one line is left in the MISSION LOG (RESTART button).
2. **They do not share CPU·GIL with the GUI thread and the MPC/policy workers.** Sending commands also has its own thread, so
   even if the child hangs, the GUI thread (20 Hz command heartbeat) is not blocked
   (`test_a_wedged_camera_process_never_blocks_the_caller`).
3. **The OpenCV thread setting stays the station's own.** pool_cam calls `cv2.setNumThreads(1)`,
   and that is process-global.

The station-side code (`rov_gui/poolcam.py`, `rov_gui/widgets/poolcam.py`) does not receive the DataBus
(the only thing the window hands over is one log function) and does not import bus·state·backends·control·perception
— `test_poolcam_is_isolated_from_control` enforces this.

Lifetime: the child runs in its own session (a terminal Ctrl+C does not reach it) and, **when stdin closes**, stops recording,
finalizes the files and exits — the same whether the station exits normally or is killed by SIGKILL
(`test_a_station_killed_mid_recording_still_leaves_finished_files`). On station shutdown it waits for the cameras **after**
putting the vehicle in neutral, and after at most 10 s it leaves the child to finish up on its own.

### Keeping the tab from blocking piloting (2026-09-30 safety audit applied)

- Even on the POOL CAMS page the keyboard·joystick keep piloting the vehicle (the window is the key handler), and the header's
  E-STOP·REC·LEAK·status pills stay visible. The DISARM·MPC panel·MISSION LOG are hidden, though, so
  **on E-STOP, LEAK, or an MPC no-response release, it automatically returns to the STATION tab**.
- The tab update runs at the **very end** of the UI tick with its exceptions isolated — a bug in the tab cannot skip the MPC watchdog.
- If the child dies, RESTART shows up in the tab, and if REC UI was on, the log·status bar say **"pool cams NOT
  RECORDING"**. RESTART continues the recording into the same folder (new files, each with its own t0). If it is alive but silent for more than 2 s,
  it is shown as "not answering — tiles are frozen".
- Disk: pool-camera recording is refused if free space is under 5 GB at start, and stopped if it falls under 2 GB during recording —
  the station's own nav·controller·policy records are on the same disk. The child runs at `nice 10` (ffmpeg included), so
  when cores run short, control·policy get them first (drops on the camera side are counted by the sidecar).

### Cost

**4 hardware cameras** × 1080p30 MJPEG, NVENC: camera process+ffmpeg **0.87 cores** while recording, **0.02 cores** on the
station side while the tab is being viewed, 0 dropped in a 15 s recording [측정: `pool_cam/bench_out/cost_20261001_real4_1080p30.txt`].
How to measure: `python pool_cam/bench_cost.py --device NAME=DEV ...`.

**One port on the PC itself per camera.** It is fine to plug into the four ports of one controller because the bandwidth is
separate for each PC port. At 1080p30 one camera reserves about half of a USB 2.0 link (196.6 Mbit/s), so **put several on one
hub (multi-port adapter) and only one comes through** [유도]. `./c3 poolcam --list` warns about that case.
When the tab is not being viewed, previews are not made at all. NVENC is a separate encoder
block, not CUDA cores, but it is true that it shares the GPU with policy inference [유도].

## Threading and data binding

The GUI thread touches **only widgets**. It never touches sockets·devices·encoders.
There are three shapes of traffic, so there are three mechanisms (`bus.py`).

| Traffic | Mechanism | Why |
|---|---|---|
| Snapshots (telemetry·thrusters·link) | queued signal on `DataBus`, 10 Hz | Small and slow. The queued connection carries them safely across the thread boundary |
| Video frames | `FrameMailbox` (1-slot conflating buffer) | A signal per frame becomes an **unbounded queue** the moment the GUI falls behind. Latency is the problem more than memory — the pilot ends up piloting off a picture from 10 seconds ago |
| Absence (silence) | `Freshness` watchdog | No source emits "I died". Deadlock·cable cut·camera stall all look like silence |

### There are only two kinds of worker (`backends/base.py`)

* **`TimerWorker`** — ticks with a `QTimer` on its own thread's event loop.
  A worker that has to **receive** commands must be this kind (queued slots are delivered between ticks).
* **`LoopWorker`** — **blocks** in `while not stopping`. Things like waiting on a DepthAI queue,
  `recv_match`. Waking up on arrival has lower latency than polling. In exchange the event
  loop does not run, so it **cannot have slots** — stopping is via `threading.Event`.

Two traps are nailed down in the code:

* A `QTimer` belongs to **the thread it was created in**. Create it in `__init__` (GUI thread) and
  the "worker" timer runs on the GUI thread. The code looks multithreaded and the profiler says
  it is not. So every timer is created in `on_started()` (= after moveToThread).
* `QThread::quit()` is not a queued event; it ends the event loop **immediately**.
  So if you put a shutdown request on the queue and then call quit, the thread dies with its timers
  still alive, and later, when the main thread GCs the worker,
  `QObject::killTimer: Timers cannot be stopped from another thread` is printed.
  → `request_exit()` only posts `_shutdown`, and **`quit()` is called inside the worker thread**
  after the timers are stopped.

### Commands ride the same bus in the opposite direction

Widget → `bus.cmd_*` → (queued) → worker slot → socket. The GUI thread does not transmit.

**Command heartbeat**: the window re-emits the current pilot input at 20 Hz *even if it has not changed*.
It is not a duplicate of the sink's deadman (500 ms) but the device that **makes the deadman work**. If the GUI thread
stalls, no new stamp comes out → the sink sends neutral. If it emitted only on change,
"a key being held down" and "the GUI being stalled" could not be told apart.

### Frame path (worker → screen)

```
decode/generate (worker)
  → if depth, colourise (worker, cv2)
  → downscale to the panel size (worker; the panel writes its target size into the mailbox)
  → bgr_to_qimage(...).copy()          ← QImage only wraps the numpy buffer. Required
  → mailbox.put()                      ← drops the previous frame and conflated++
  ...
  → UI timer 30 Hz: mailbox.take() → blit in paintEvent
```

A `QImage` is safe to create outside the GUI thread (implicitly shared data).
A `QPixmap` is not (it can touch the window system) — workers hand over only QImages.

## What makes "Disconnected" trustworthy

* Every panel measures the age of its own data. Timeouts differ per source (for 15 fps video
  0.3 s means late; a 2 Hz BATTERY_STATUS is still normal after 3 s).
* **A frozen picture is marked destructively.** Dead video does not go black; it shows the last good
  frame forever — indistinguishable from a picture of hovering still in front of a structure.
  So when stale it is dimmed, diagonal hatching is drawn over it, and "STALE 3.2 s ago" is stamped on it.
* An unknown value is `--`, not `0` (`theme.fmt`). 0.0 V does not mean "unknown".
* Colour is for status only: green ONLINE / amber DEGRADED·STALE / grey OFFLINE /
  red FAULT. **Splitting grey from red is deliberate** — "not plugged in" and "broken" call
  for different responses, and if both are painted red, red ends up being ignored.

## Backends

| Source | What it opens | Notes |
|---|---|---|
| `demo` | nothing | **Every number is synthetic**. A `SIMULATED DATA` banner is always shown in the header. Exception: the tether panel reads the **real** NIC counters even in demo |
| `hw` | C3(DepthAI) + ArduSub(MAVLink) | Reuses `c3_camera.source.C3Source`, `c3_camera.mavlink_log.MavlinkLogger`. Preflight before connecting |
| `ros2` | one rclpy node | This machine has no rclpy, so it **could not be verified by running it**. Only the structure is complete — see KNOWN_ISSUES |

The tether panel's **from ROV / to ROV** is the actual traffic on the topside NIC (`rx`/`tx` turned into
pilot language). *from ROV* = what comes up (C3 video + depth + ROV camera +
telemetry), *to ROV* = what goes down (our commands, usually 1~2 Mb/s).

What else this panel watches (`net.py`, sysfs, no privileges needed):

* `carrier` / `operstate` → the Ethernet-side link of the fibre media converter
* `speed` → renegotiation detection (1000 → 100 is typical of a connector going bad)
* `rx_errors` / `rx_crc_errors` / `rx_dropped` growth rate → **the link is up but
  dirty**. It is the only early warning that comes up *before* the video cuts out
* TCP connect RTT to BlueOS → the path down the tether

## Latency (measured, 2026-08-06)

C3 colour **sensor→host 28.2 ms @ 30 fps** (1920×1080 window, with depth on). Four things
fixed to get there:

| Config (depth at 640×360) | Colour latency | Colour fps | Link |
|---|---|---|---|
| 15 fps colour + 15 fps depth | 33.7 ms | 15.0 | 61 Mb/s |
| **30 fps colour + 30 fps depth** | **128.0 ms** | 20.2 (33% dropped) | 81 Mb/s — saturated |
| 30 fps colour + 10 fps depth | 34.4 ms | 30.2 | 48 Mb/s |
| 30 fps colour + 10 fps depth (same config, complex scene) | 49.4 ms | 29.9 | 65.7 Mb/s |
| 30 fps colour, `--no-depth` | 33.1 ms | 30.0 | 11 Mb/s |

* **Colour and depth get their fps separately** (`--fps 30`, `--depth-fps 20`). Giving both 30
  does not make the picture faster; instead frames fall to 1/3 and latency becomes 4× — because it hits
  the PoE ~90 Mbit/s limit.

### To keep depth at 640×360@20, cut colour by **pixels and quality**, not by fps

The two `30+10` rows in the table above are **the same config** measured on different days, and they differ by 18 Mb/s.
That is not noise — colour is MJPEG, so depending on the scene it varies from 46~121 kB per frame.
**Depth is raw uint16, so it is fixed at `w·h·2·fps`** and independent of the scene (640×360@10 =
predicted 36.86, measured 36.9 Mb/s — the model fits exactly).

**640×360@20 on its own is 73.7 Mb/s, 82% of the link.** Only ~16 Mb/s is left for colour, and
960×540 q90 is 18.7 even at 20 fps, so it does not fit. **Cutting fps is the least efficient lever for this
problem** — 30→15 gives back half, but more than that is needed, and the cost is the thing the pilot
notices first (motion).

Quality and resolution are much cheaper. Below are values **measured by re-encoding 120 frames captured by the C3
itself** (`c3_camera/datasets/dataset_20260805_174544/rgb`, 960×540 MJPEG).
Host re-encoding looks untrustworthy, but at q90 it **gave 114.1 kB/frame, and the value the device
encoder left for the same frames is 114.2 kB/frame** — a 0.1% difference, so it can be used as a substitute.

| Colour | kB/frame | Mb/s @20 | Mb/s @30 | + depth 640×360@20 |
|---|---|---|---|---|
| 960×540 q90 | 114.1 | 18.7 | 28.0 | **103% @20** |
| 960×540 q75 | 64.6 | 10.6 | 15.9 | 94% @20 |
| 800×450 q90 | 80.2 | 13.1 | 19.7 | 97% @20 |
| **640×360 q80** | **38.0** | 6.2 | **9.3** | **92% @30** |
| 640×360 q85 | 44.8 | 7.3 | 11.0 | 90% @20 · 94% @30 |
| 480×270 q90 | 36.9 | 6.1 | 9.1 | 92% @30 |

The size exponent is `bytes ~ px^0.82` — halving the pixels does not bring the bytes down to half
(high frequencies saturate first). So **cutting quality first pays off**: q90→q80 gives back
1/3 of the bytes, while at panel size whiteboard writing is still readable.

The current default is this combination:

```
--fps 30 --isp-scale 1/3 --mjpeg-quality 80 --depth-fps 20 --depth-size 640x360
   colour 640×360 q80 @30 = 9.3    depth 640×360 @20 = 73.7      total 83.1 Mb/s (92%)
```

It is the combination that keeps depth at 640×360@20 **while holding colour at 30 fps**. For more headroom,
`--mjpeg-quality 75` (91%) or `--fps 20` (88%); for more picture,
`--isp-scale 1/2 --mjpeg-quality 75 --fps 20` (94%, no headroom).

The budget estimate now **responds to `--mjpeg-quality`** (`MJPEG_QUALITY_SCALE`, derived from the
measurements above). Before, it ignored it, so even after lowering to q80 the log reported the q90 value. Give a
combination that exceeds the budget and the startup log says so **at warning level** — exceeding it does not fail; it
drops frames and latency becomes 3~4×, so if nothing says so, it just looks like a slow camera.
`test_default_video_config_fits_the_c3_link` holds the default **below 95%** of the budget
(95%, not 100%, to leave room for the scene to change).

> The table's kB/frame values are measured; **the combination totals are [유도]** — they could not be measured directly on the link.
> During the 2026-08-06 sweep the C3 crashed and vanished from the network, so only `30+10` (the 65.7 Mb/s row above)
> was left.
* **Depth is colourised after downscaling** (`imaging.scale_depth`). The other way round it is
  4× more expensive, and that cost shows up as **latency in the colour feed**, which uses the same worker thread
  (57.8 → 32.2 ms). Being nearest-neighbour is deliberate too — interpolating depth creates distances at
  edges where there is nothing.
* **Near-1:1 resizes are skipped** (`NO_RESIZE_ABOVE = 0.85`). Shrinking 960×540 to
  933×525 scans every pixel and gains nothing (57.7 → 28.2 ms). The last few
  % is done by the blitting QPainter anyway.
* `--ui-fps 60` (default). At 30 Hz a frame waits up to 33 ms for the next paint.

**~90 Mbit/s is the limit of the C3's own link (camera→switch), not of the tether.** The fibre tether is
1000 Mbit/s and also carries the ROV's stock camera (~34 Mbit/s) and MAVLink. Only **the C3 stream**
goes into the budget calculation above.

## The SRG/SWY/HVE/YAW bars are **the command we send**, not the vehicle's motion

That is why in STABILIZE or DEPTH HOLD, PROPULSION moves while these bars sit at 0. That is not
wrong — in those modes **the thrust the autopilot produces does not pass through these bars.** The bars are the values
this station actually sent out after composing·scaling·clamping, and what the motors are actually doing is
shown by PROPULSION.

Why the bars themselves were not turned into vehicle motion: recovering the body-frame axes from the PWM of the 8 motors
requires **the vehicle's allocation matrix (AP_Motors6DOF)**, which we do not know. Borrowing the sim's
matrix gives plausible but wrong numbers — exactly the kind this repo goes as far as auditing
to prevent.

Instead it **says it in words.** When armed + command 0 + motors spinning, the deadman line shows:

```
TX 20 Hz   AUTOPILOT DRIVING (STABILIZE) — bars show YOUR command only
```

Hover over a bar and the same explanation shows up as a tooltip.

## OUTPUT defaults to 20%

It is a scale factor applied to every axis, and the first thing a pilot does right after connecting is to nudge
the stick to see "what moves". In a narrow pool or next to a diver, that one nudge must not be 1/3 of maximum
thrust.

> There was an actual bug here: the slider position·label·scale factor were hard-coded **in three separate places**, and
> the `valueChanged` connection came **after** `setValue`, so the handler did not run at startup. When the default was
> changed to 20, the slider showed 20%, the label 60%, and the actual scale factor became 0.60 — a state where what goes out is
> 3× the value the pilot reads. Now all three are set from a single `DEFAULT_OUTPUT_PCT` through the handler, and a test
> guards that the three agree.

## Object tracking + 6-DoF pose — `--pose`

Click an object in the C3 RGB feed and SAM2 keeps following it, and after that a 6-DoF pose comes
out as well. **It is off by default.** When it is on, the flags decide which of three branches it
takes:

| | On click | GPU time | Working distance |
|---|---|---|---|
| `--pose` | collect reference views → reconstruct → pose | collection + ~1–2 min | 0.3~0.8 m |
| `--pose --pose-mesh M.obj` | register directly against that mesh → pose | ~0.73 s | 0.3~0.8 m |
| `--pose --pose-no-build` | mask only | none | no limit |

```bash
./c3 gui --source hw --allow-command --pose
```

Turn on `TRACK` at the top right of the panel and **click the object once**. A double-click still
promotes the feed to the large slot.

Measured (2026-08-09, RTX 5090, `--pose-model tiny`): loading **1.1 s**, enters `tracking` after the
click, **36 Hz**. The model size is changed with `--pose-model` (tiny 84.5 fps/777 MB … large
37.1 fps/1523 MB, measured by the original project).

### Four things that matter in the design

* **Off means truly absent.** Without `--pose`, neither torch nor SAM2 is imported, and
  `C3VideoWorker` **does not even copy** the frame (`_tap_pose` sees that the mailbox is None and
  returns immediately).
* **Layout cost: 0.** The `TRACK` button is an absolutely positioned child of the panel, next to
  REC, and the mask and status chip are a paint pass in `paintEvent` — the same approach as
  `legend=`. There are only 5 px of vertical headroom, so a new row is impossible — and in fact the
  minimum window **stays at 1152×763** (the tests check both the on and the off state).
* **What crosses over is the contour, not the mask.** A 640×360 bool mask is 230 kB per frame, and
  the GUI thread would have to draw it pixel by pixel. `findContours` + `approxPolyDP` turn it into
  a ~1 kB polyline, which is drawn with `QPainterPath` — the same picture, 0 pixel work on the GUI
  thread.
* **Clicks are delayed by 400 ms.** Qt sends press+release **first** even for a double-click, so a
  plain `mousePressEvent` misfires a prompt on every promote gesture. The click is held back for
  the system's `doubleClickInterval` and cancelled if a double-click arrives. The delay is not
  noticeable — selecting an object is not an instantaneous action to begin with.

### The coordinate-transform trap

Panel click → **source pixel** passes through three coordinate systems, and the middle one is the
trap:

```
canvas px --(letterbox)--> displayed image px --(scale factor)--> source px
```

The displayed image is not the source (the worker shrinks it to the panel size before putting it
in). But `scale_to_fit` **returns the original as is** when the shrink ratio is 0.85 or more
(`NO_RESIZE_ABOVE`), so that scale factor switches between 1.0 and some other value depending on the
window size. Assume either one and it is silently wrong at some window size. A click on a letterbox
bar is rejected with `None`.

### 6-DoF pose — `--pose-mesh`

Given a mesh, the pose locks on **about 0.73 s** after the click, and from then on it is tracked at
40~60 Hz.

```bash
./c3 gui --source hw --allow-command --pose \
         --pose-mesh "~/Desktop/New Folder/ref_views/model/model.obj"
```

Measured (2026-08-09, replayed with the saved reference views + that mesh): mesh
`13444 verts 101×135×165 mm`, registration succeeded, `T[:3,3] = [0.058, 0.109, 0.413] m`
(distance **0.431 m**), axis projection correct, clean exit.

Shown on screen are the RGB axes (a 12-edge box with `--pose-box3d` if you want one) and a stage
chip at the top left:

```
POSE TRACKING  13 Hz    solve 210 ms    reg 1
```

**The three numbers answer different questions** (2026-08-23):

| Number | Meaning | Where from |
|---|---|---|
| `n Hz` | **how often a new pose actually arrives** — the very rate the overlay, object anchor and follow receive | **counted directly** by `_RateMeter` in `perception/session.py` |
| `solve n ms` | the time the GPU takes **to solve one frame** | the upstream tracker's `hz` converted back to ms |
| `reg n` | re-registration count = the number of times the pose slid off the object | FoundationPose |

Looking at the two together shows **where the slowness comes from**: if `solve` ≫ `1/rate`, the GPU
is the bottleneck; if `solve` ≪ `1/rate`, the camera or the mask is the bottleneck.

> **Never pool with records from before 2026-08-23.** Before then, the chip's `n Hz` was
> **`1/평균 계산시간`** ("1/mean compute time") — because both upstream trackers
> (`sam2_live/tracker.py:507-512`, `sam2_live/pose.py:357-365`) publish
> `1.0 / mean(직전 30개 job 소요시간)` ("1.0 / mean(time taken by the last 30 jobs)") under the
> name `hz`. A single 0.73 s registration and 20 ms tracking steps were mixed in
> **the same 30-sample window**, so even with the object sitting still the chip wobbled
> 5 Hz → 9 Hz → 5 Hz (pilot, 2026-08-23). The update rate could never have been read from that
> number in the first place. The names were changed in the files too, to set a boundary:
> `pose_hz` → **`pose_hz_measured`** + `solve_ms` (CSV), `sam_hz` → `sam_hz_measured` +
> `sam_solve_ms` (jsonl TRACK row). A reader that looked up `pose_hz` by name in old files
> **fails on new files instead of silently reading a different quantity.**

How it is measured: both trackers create a **new array object** on every solve and leave the
previous one in place on failure, so **object identity** (`is`), not value comparison, is the exact
test for "a new one came out" (an identical pose of an object sitting still is correctly counted as
a new arrival too). A sample is taken **once per frame handed over** — both trackers have a single
job slot, so there is never more than one result per frame and none can be missed, whereas sampling
in `poll` (the 10 Hz publish gate) would alias everything faster than that. The denominator is the
time up to **now**, not up to the last arrival, so when poses stop coming the number does not
freeze but falls to 0 over 2 s. `test_the_pose_rate_is_an_arrival_rate_not_one_over_compute_time`.

**The object's coordinates are not in this panel** (2026-08-23, removed at the pilot's request).
There used to be one more line at the bottom, `x +0.084  y +0.020  z +0.581 m  d 0.59 m`, but those
are **camera-frame** numbers — nothing else in this station works in that frame, so they could not
be compared with anything, nor could anything be flown on them. The object position now appears in
**one place** only, the trajectory plot's readout (see "Following an object" below). What is left
in the chip, `n Hz`/`reg n`, is **health**, not position.

### On-site reconstruction — when there is no mesh (default)

Give only `--pose`, without `--pose-mesh`, and **one click is all it takes**. The station collects
the reference views by itself, reconstructs a mesh and estimates the pose with that mesh. This is
the same as the original tool's `--pose` behaviour.

```bash
./c3 gui --source hw --allow-command --pose     # goes all the way to pose even without a mesh
```

### The screen always tells you the stage

**Two lines** appear at the top left of the C3 RGB panel. The top one says **what stage it is in
now**, the bottom one **why it is not moving on**. Both are needed — during collection, only the
bottom line tells you whether a counter that is not going up is normal or a fault.

```
LOADING SAM2 — 1s
READY — CLICK AN OBJECT
COLLECTING VIEWS  7/20    ORBIT 31/75°    0.45 m
    too close in angle to a view already taken (3 deg)
COLLECTING VIEWS  3/20    ORBIT 62/75°    1.24 m
    object too far (124 cm) — bring it to 30-80 cm
RECONSTRUCTING MESH — 42s
    BundleSDF is running in a child process
LOADING FOUNDATIONPOSE — 8s
REGISTERING POSE...
POSE TRACKING  13 Hz    solve 210 ms
TRACKING  13 Hz  solve 34 ms — no pose yet   ← pose is expected but not there yet
TRACKING  13 Hz  solve 34 ms — mask only     ← --pose-no-build. A mode with no pose by design
LOST — CLICK THE OBJECT AGAIN
STOPPED
    stopped after 74 deg with only 3 usable views (need 6) — turn the object
    more slowly, keep it 30-80 cm away, then press TRACK again
```

Why the distance is in the chip: **outside 0.3~0.8 m every frame is rejected because of stereo
noise**, and from the screen alone there is no way to tell how far away the object is.

**The colour says the same thing.** Blue (BUSY) = in progress (collection/reconstruction/registration),
green = pose locked on and tracking, red = stopped. Previously, when collection gave up it silently
went back to green and **could not be told apart from normal tracking** — that is what "it suddenly
turned green" really was; now it stays at `STOPPED` (red), says why, and is only cleared by pressing
`TRACK` again.

The same transitions are also left **in the LOG panel**, one line each. The overlay has no history,
so afterwards "did reconstruction even start?" can only be answered from the log.

**Messages are translated into English before being shown.** The original project talks to the
pilot in Korean (`sam2_live/capture.py`), and those strings are **the only explanation of why
collection is not working**, so the translation is done at the boundary
(`perception/session.py::english`) — since the original is read-only.

The artifacts are left in `data/<날짜>/<시각>_obj/` (`data/<date>/<time>_obj/`; `--pose-ref-dir`).
In the next session, passing that `model/model.obj` as `--pose-mesh` skips the 2 minutes.

**Turning `TRACK` off really aborts it** — it calls `terminate()` on the reconstruction child
process. A process that is piloting the vehicle must never be unable to stop a 2-minute GPU job, so
the original's `subprocess.call` was rewritten with `Popen` (everything else is identical to the
original).

Measured (2026-08-09, replaying 17 saved reference views, input `~/Desktop/New Folder/ref_views`):
collection ended automatically at **8 views / 75.3° rotation / ICP RMSE 6.2 mm** → BundleSDF
**25 s** → mesh **7993 verts, 51×94×167 mm**. The original mesh of the same object is
`13444 verts 101×135×165 mm`, so **the longest axis agrees within 2 mm** and the other two axes are
smaller — because it is a partial model that used only 8 of the 17 views. **This is a replay of
saved data, not a measurement of live collection.**

The two collection rules are backed by measurements, and both are caps:

* **Cumulative rotation 75°** (`--pose-max-arc`). The per-view camera poses are built with colored
  ICP, and beyond 80° the error does not degrade gradually but collapses like a cliff — per the
  original's synthetic experiment, 60° 8.7 mm, 80° 9.7 mm, **100° 87 mm, 120° 283 mm**. Raising it
  buys a wrong mesh, not coverage.
* **At least 6 views** (`--pose-max-views` caps at 20). With fewer than 6, reconstruction is
  **refused** and it tells you to do it again. 10 or more recommended; the reference implementation
  uses 16.

The price is that the mesh is **a partial model that covers only the range of angles captured**.
Seen from the opposite side, pose estimation fails, and the re-registration monitor catches that as
"the mesh is not this object".

If only the mask is needed, `--pose-no-build` turns off the whole reconstruction path — no
working-distance constraint, no 2 GPU minutes. All the pose fields are then blank; that is not the
tracker being broken, it is doing what it was told.

#### Re-registration monitor — the only recovery mechanism

FoundationPose's `track_one()` **does not use the mask.** So once the pose slides off the object,
it has no way of coming back by itself. Every frame the IoU of the mesh silhouette against the SAM2
mask is compared, and on a mismatch the pose is registered again (`POSE_MIN_IOU 0.25`, 8 consecutive
frames, minimum interval 1.5 s — because `register()` itself takes 0.73 s). If it still does not
fit after 4 retries, it does not keep clinging on but says so:
**"pose does not fit — wrong object, or the mesh is not this one"**.

#### Working distance 0.3~0.8 m

FoundationPose consumes depth. Per the original README,
**at 2.4 m the stereo noise exceeds 50 mm and registration itself fails**. For an ROV that is a very
close range. The mask-only mode (without a mesh) is not subject to this constraint, so if you just
want to mark a distant object, that mode is the right one.

The C3's HFOV of **63.9°** has to be considered as well — at 0.5 m the field of view is **62 cm**
wide, so the vehicle moving just 30 cm takes the object out of the frame.

**Two ways collection gets cancelled (2026-08-21, hardware)**. The log shows which one it was.

1. `stopped after N deg with only M usable views (need 6)` — the **arc budget** is used up.
   `--pose-max-arc` (default 75°) is consumed by **amount of rotation**, so turning fast lets the
   75° go by while most of the frames are being rejected. **Raising this value is not the answer**
   — in the upstream synthetic experiment the per-view error collapses **like a cliff**: 9.7 mm at
   80°, 87 mm at 100°, 283 mm at 120°. The answer is to turn more slowly.
2. `lost the object — lost — reacquiring` → `could not reacquire — click again` —
   **SAM2 lost the mask and failed to reacquire it**. In water it loses the mask far more often
   than on the bench, because of motion blur and the vehicle's own motion. The retry budget is
   `--pose-lost-timeout` (default 15 s), but until 2026-08-21 it **was not exposed on the CLI**.
   Now it is:

   ```bash
   ./c3 gui --source hw --mpc --pose --pose-lost-timeout 45
   ```

   While reacquiring, nothing is recorded and nothing is discarded, so the only cost of raising it
   is the waiting time. `--pose-lost-grace` (default 3 s) only sets when the overlay starts calling
   a **short** dropout "lost" — it is not a knob for riding out long dropouts.

### Seeing the object in 3-D — the trajectory panel's `3D` button (2026-08-21)

**The free-standing `MAP` window is gone.** That window drew the object in
**camera coordinates**, but the real question was "where in the **pool** is the object, and how
does it relate to the **vehicle**". Now the trajectory panel answers that question directly —
press the `3D` button and the same plot tilts.

* **Same picture, only depth added**. The pool boundary, tag map, trajectory, reference cross, DR
  ghost and object diamond are all still there, each drawn at its own **z**.
* **A dotted line drops from every marker to the tag plane (z=0)**. In an orthographic projection
  a marker suspended in space is ambiguous along the line of sight, so without this foot the depth
  cannot be read.
* Drag = orbit (pan in 2-D), wheel = zoom, double-click = reset. `az`/`el` appear at the top left.
* **At elevation 90° it is pixel-for-pixel identical to the top-down view** — meaning the world
  does not move when the button is pressed, and a test pins that down
  (`test_the_trajectory_3d_view_reduces_to_the_top_down_one`).
* It is an orthographic projection (no perspective). At pool scale perspective gains nothing and
  makes the same distance look different depending on position — and the judgement the pilot makes
  on this plot is precisely that distance judgement.

**Markers keep their depth even when engaged** (fixed 2026-08-23). The engage datum is a
**horizontal isometry with a z offset** (`_datumize` is `Rz @ (eta - p0)`), but `_to_map` undid
only x/y and left z as it was (the field name itself was `_z0`). So **the moment anything engaged**,
the hull, trajectory, reference cross and DR ghost were all drawn at z ≈ 0, i.e.
**lying flat on the tag mat** — only the object kept its true map z, and the chip's `z -1.05 m` was
right only because `_hold_z_text` added the offset back by hand in that one place. In top-down this
was invisible; with `3D` on, the robot was stuck to the floor. Now `_to_map_z` converts
**once, at the boundary**, so every z after that is map z, and the differences between them (`dz`,
the error line) keep working unchanged because the offset is common to all.
`test_the_engaged_marker_keeps_its_map_depth_instead_of_the_tag_plane`.

**The hull marker is always written from only one place** — `engaged` is the switch (fixed
2026-08-23).

| Status | Who writes the marker and trajectory | Source of z |
|---|---|---|
| not engaged | `add_fix` — the **raw** tag fix | the tag PnP's z |
| engaged | `add_status` — the assembled control state | **pressure depth + session offset**, x/y velocity-interpolated over the fix age, yaw gyro-interpolated |

`MpcWorker._publish` fills `p_flu` whenever there is a state, regardless of whether anything is
engaged, so previously **both sides wrote to the same marker at once while not engaged** — at
17 Hz and 20 Hz, and **different quantities**. The marker jumped between the two values every tick,
and the trajectory got one point from each side in turn. In 3-D, the screen-right vector of
`_basis3` has a z component of 0, so that alternation is drawn as **perfectly vertical lines**, and
once 90 seconds' worth piles up it becomes a comb-tooth curtain — this is exactly the picture the
pilot saw on 2026-08-23 as "it is sitting still, but z keeps going back and forth". Neither of the
two estimates was wrong; **drawing the two as one series was wrong.**
`test_only_one_source_writes_the_vehicle_marker_at_a_time` pins this.

### The vehicle marker is an ROV, not a hull box — gripper, thrusters, C3 (2026-09-03)

**Why it changed.** For a long time the marker was **a single rectangle**: the vehicle's footprint,
drawn at the position the tag solution reports. That position is the **vehicle COM**, but the point
the pilot actually cares about on this vehicle is **0.50 m ahead** of it — the **jaw**. The
diffusion policy's TCP is exactly that jaw (`hw_mpc.yaml policy.tcp_body_flu_m`), so every
reference the DP draws is composed there, yet the COM-centred rectangle was placing the drawing of
the vehicle half a metre away from that point.

Now `widgets/rov_shape.py` holds a **body-frame model**, and the panel projects it in two modes:

* **DP plans** are drawn as **thick red dashed lines** — the thick one is the current plan, the thin
  one the one just before, **exactly two** of them (2026-09-03 pilot request — in the first pool
  run, once 60 had piled up, the network's output could not be told apart from the vehicle's
  motion). The zoomed inset was removed: the request was that having everything in one plot is
  better, and with only two left the scale problem went away too. **Drawn last** — originally they
  were drawn even before the trajectory and so were hidden under the hull (0.43x0.53 m, a body with
  filled faces), and the plan is composed, of all places, exactly where that hull is (the jaw). That
  was the cause of the report "the reference is hidden behind the live ROV", and colour or
  thickness cannot fix it. The colour changed once more, **fluorescent yellow -> red**: yellow lost
  out both to the "now" end of the trajectory age ramp (yellow-green) and to the green hull.
  `theme.FAIL` (#ef4444) is already used by the 1 px tracking-error line, so the plan is a purer red
  (#ff1f1f) at **5× the thickness**, and the real distinguisher is the thickness. A `DP` swatch in
  the legend.
* **The trajectory trail is 3 s** (`TRAIL_AGE_S`, 2026-09-03 pilot request; originally 90 s). At
  90 s the vehicle's own history filled the plot and buried the ~1 s plan that actually needs to be
  seen. At 3 s, which way and how fast it is going still reads just the same, and it is on the same
  time scale as the plan lying beside it.
* **The `3D` depth gauge is fixed in the right margin** (2026-09-07 pilot request: "the z axis
  moves along with it ... I'd like it to be fixed"). The original ruler, added on 2026-09-03, was
  drawn next to the vehicle through the projection, so it **moved along in three ways**: (1) its
  anchor was 118 px to the screen-left of `p_act`, (2) because
  `u = (cos az sin el, sin az sin el, −cos el)`,
  **1 m horizontally moved it up and down tan(el)=1.43× more than 1 m of depth**, and (3) the span
  was re-centred on the live z band every frame, so **even the tick numbers** slid. All three were
  removed — fixing only the anchor catches only (1)(2), fixing only the span only (3), and with both
  fixed the scale can no longer be a projection of the scene (because the projection is a function
  of the viewpoint). So now it is **an instrument, not an axis**: a fixed z window sits on a fixed
  pixel strip, the vehicle is shown as a **needle** moving over it (green, printing its value
  directly), and the reference as a hollow blue needle.
  `_z_gauge()` reads only three_d, the widget size and the latched window, and
  **reads neither the pose nor the pan** (`test_the_z_gauge_does_not_move_with_the_vehicle` pins
  that property — `pan` is included because `follow` in `replay_run.py` pans to the vehicle every
  frame, so depending on pan amounts to depending on the pose).
  **The cost, plainly**: 1 m on the gauge is now not 1 m in the picture, so you cannot read it by
  moving your eye horizontally from the hull to the scale. Instead the needle marks the value and
  prints the number.
  The window starts at `Z_GAUGE_WINDOW_M = (-1.10, +0.40)` m and **only ever widens** (it does not
  shrink and does not re-centre; `clear()` resets it) — the pilot chose "fixed band + widen when
  exceeded".
  NED z is + downward, so **deeper is lower on the screen**, and the caption says so.
* **The current z is always shown at the top left** (`z -0.170 m   ref -0.200`). The chip's
  `_hold_z_text` appears only when there is a reference, and the ruler only in 3-D, but what a pilot
  flying by hand keeps asking is "how deep is it now".
* **Hull** box — the planform is as before (`hw_nav.yaml rov_footprint_m`), height 0.254 m, bottom
  face at z −0.2171. The top-down outline is **identical to before, vertex for vertex**, and a test
  pins that down (`test_footprint_is_exactly_the_rectangle_the_panel_always_drew`) — because the
  rectangle the pilot used to gauge wall clearance must not move by even 1 mm just because a gripper
  has appeared.
* **Newton gripper** tube + two jaws, and a **TCP ring** at the jaw centre. The gripper is attached
  47 mm **above** the bottom (it is wedged between the skids), so from above all you see is
  **the part sticking out** in front of the nose, and top-down draws exactly that much and no more.
* **8 T200s** — 4 vectored horizontal + 4 vertical. The four turned by 45° are drawn as their
  **actual silhouette (a rhombus), not a bounding box** (`silhouette_xy`); drawn as boxes, which way
  they push would be lost. When the hull gets smaller than 48 px the thrusters are dropped (at that
  size they are just specks).
* **C3** and its **dotted optical axis**. In a DP run the pilot's question is "is the object in the
  frame", and that the lens sits 0.105 m **above** the COM, looking 43.3° down, cannot be told from
  the hull outline alone. Why a **line** and not a cone: a cone becomes a claim about "what is
  visible", and this panel does not know the range. The angle is exactly the `cam_tilt_deg` the
  localizer uses.
* **Dotted heading line** (2026-09-07, pilot request). A dotted line drawn through the COM along the
  vehicle x axis, **0.55 m aft and 0.85 m forward** (`rov_shape.HEADING_AFT_M/HEADING_FWD_M`, for
  the drawing [예측]). Before, the only line drawn on the vehicle was the dotted optical axis, which
  starts at the lens (above the gripper tube) and ends 0.63 m ahead of the COM
  [유도: 0.306 + 0.45·cos 43.3°], so it read as "a heading line that exists only in front of the
  gripper"; and judging whether the plan, reference or object is **on** the vehicle axis or beside
  it needs an axis that runs through the body and extends both ways.
  **In top-down only this line is drawn** — seen from above it is the same straight line as the
  dotted optical axis (the 43.3° downward pitch is not visible), so drawing both on top of each
  other made the dots of the lens~0.63 m segment twice as dense, and they read as a marker. In `3D`
  both are drawn (the heading line is a horizontal line at COM height, the optical axis a line bent
  downward). In the plan-view path it is drawn **after the outline, fill and nose tick**, so it is
  not covered, and **the heading line remains** even at zooms where the hull shrinks below 14 px and
  the body is dropped (that zoom is exactly the one that needs a sight line). If the heading is
  unknown (yaw None) it is not drawn — an axis pointing to a default north would become a claim.
  The DR ghost draws neither line. The replay page (`replay_html`) follows the same rule.

**Press `3D` and it becomes a body, not a box.** Per-part back-face culling (exact within a part,
since each part is convex), then far→near sorting between parts = painter's algorithm. The only
shading is `|dot(법선, 시선)|` ("|dot(normal, line of sight)|") — no light source, no specular.
It is an instrument, and the shading's only job is to make a box read as a box.
**The gripper is not hidden behind the hull; it is always drawn on top**: physically the hull would
hide it from the default viewpoint, but the jaw is the policy's TCP and the whole reason for drawing
the gripper is to see its position relative to the reference — the picture is a **cutaway**, not a
photograph.

**Provenance of the numbers.** All of them are in `rov_shape.GEOMETRY_PROVENANCE`, and
`rov_geometry_meta()` hands them to the run meta (the `rov_drawn_geometry` key of
`controller.json`/`meta.json`) — this had existed since 2026-09-03 only as a promise with no caller;
the 2026-09-07 review found that and wired it in; the meta of earlier runs does not have this key.
Honestly put, **of the gripper geometry, the only thing measured on this vehicle is the jaw x**
(2026-09-08): in the frame where the bottle held in the jaw was placed on the centre of floor tag
58, the grip point is **0.196 m in front of the lens** (range 0.187–0.204) [측정: data/20260908/0908_180453_observe/policy_obs/rgb/000160.jpg
(tag PnP with the run's own intrinsics) + 0908_170428/policy_obs/rgb/000000.jpg(bottle height 0.084 (±0.006) m) +
0908_175151/policy_obs/rgb/000000.jpg(open-jaw ray)] — `LENS_TO_GRIP_FLU_M`. So the jaw is
**anchored to the lens** (`JAW_CENTRE_M` = `CAM_T_FLU_M` + [0.196, 0, −0.275] = (0.502, 0, −0.17),
meta `lens_to_grip_flu_m`). The rest is still estimated — tube and opening are from the vendor
[스펙], the tube position is the sim `GRIP_POS` shifted by the same +0.0855 as the jaw [유도], the
jaw box is [예측], the jaw y·z come from the CAD chain [유도]. The vertical chain is well pinned
down: the jaw is
47.075 mm above the bottom, the lens 275.014 mm above the jaw (`hw_nav.yaml` `cam_t_flu` block, the
pilot's CAD [측정]; the image also agrees within ~1 cm), overall height 0.254 m.
**That chain is the very chain that caught the C3 sign error**, but the image disproved the same
CAD's **horizontal** 110.664 mm (measured 196 mm) — so the x/z of `cam_t_flu` itself are suspect too
(KNOWN_ISSUES 2026-09-08 entry). `test_the_vertical_chain_closes_on_hw_nav_s_numbers` pins both the
vertical chain and the lens anchor. The jaw opening is
**not live** — this station has no jaw feedback (`PayloadState.gripper_fb` is None on
hardware). The drawn opening is a schematic constant.

Verification: 18 tests in `rov_gui/tests/test_rov_shape.py` — the two sign flips (FLU→NED) against
**hand-computed values**, whether the footprint is identical to before, whether the drawn TCP is
**the same point** as `policy.tcp_body_flu_m`, whether the camera matches `cam_t_flu`/`cam_tilt_deg`,
whether the normals point outward, whether the heading line passes through the COM, extends both
ways farther than the old optical-axis end and the tail, and turns with yaw, and whether every
constant has a provenance tag (`test_offline.py` pins the widget side — top-down only the heading
line, 3-D both, it survives even at a zoomed-out level and is absent when yaw is None). To see it by eye:
`rov_gui/tools/policy_observe_smoke.py --shots DIR`.

### The overlay goes into the recorded video

With `TRACK` on, **the mask, 3-D axes and stage lines are burned as is into the C3 RGB feed
recording (`REC`).** A recording of a tracking session in which the tracking is not visible is a
recording of the floor.

```
POSE TRACKING  13 Hz    solve 210 ms    reg 1
  (mask contour + RGB axes)
```

* It is **the same function** (`_paint_pose`) as the code that draws on screen — so that the video
  cannot show anything the pilot did not see. Only the `fit` argument differs (a recorded frame has
  no letterbox, so the image is everything).
* **It draws on a copy.** The canvas holds the same QImage and draws the live overlay on top of it
  again, so drawing on the original would draw every line twice.
* **When not recording, it does not even copy**; when `TRACK` is off, the frame passes through
  untouched.

The full-UI recording (the `REC` at the top) has always been exactly what is on screen, so this does
not apply to it.

### Logging — rides on the C3 Depth recording

Same trigger as `_rov.jsonl` / `_c3_imu.jsonl`. With the same stem as the video:

```
c3_depth_….mp4
c3_depth_…_pose.jsonl   CAMERA(1×) · MESH(1×) · PROMPT(click) · EVENT(transition) · TRACK · POSE
c3_depth_…_pose.json    manifest (by_message, hz_measured)
c3_depth_…_pose.csv     1 row per 1 pose — t, frame_seq, x/y/z_m, distance_m,
                        r00..r22(rotation rows), pose_hz, n_register, state
```

The CSV is **a second format, not a second source** — the same numbers from the same `T_cam_obj` as
the jsonl POSE rows, with the same capture time. It is just flattened out so that pandas/plot
scripts can consume it directly.

The `TRACK` rows hold **the contours themselves** — offline re-analysis is possible without running
SAM2 again. The timestamp is **the frame's capture time**, so it is aligned with the video.

The `CAMERA` row carries the intrinsics together with **their provenance**:

```json
{"msg":"CAMERA","fx":513.4,…,"medium":"water","rectified":false,
 "note":"UNDERWATER factory calibration (vendor-confirmed; measured HFOV
         matches the Snell-underwater prediction, not the in-air spec).
         Distances ARE metric in water. IN-AIR captures read ~1.33x LONG."}
```

The `MESH` row stamps **which 3-D model the pose is a statement about** — the path, `sha1`, whether
it was an on-site reconstruction, the reference-view directory, and the sanity-check lines right
after reconstruction (the scene size implied by `sc_factor`, the mesh bbox, the `bad_mask#` range).
An on-site reconstruction is a different model every time, so the path alone cannot identify it,
and the name `model.obj` is always the same.

The `POSE` row holds the 16 values of `T_cam_obj` (row-major) plus `pos_m` and `distance_m`, and the
manifest spells out **the convention in words** — left implicit, it gets misread 6 months later:

```
T_cam_obj maps OBJECT -> CAMERA. OpenCV optical axes: +X right, +Y DOWN,
+Z forward. Metres, row-major. NO camera->body transform is applied.
```

> **Mind the calibration direction — the factory calibration is an underwater value** (confirmed
> by the vendor + FOV measured: HFOV 63.7° agrees not with the in-air spec of 95° but with the
> Snell underwater prediction of 67.2°, `calib/fov_audit.py`, KNOWN_ISSUES 2026-08-04). So the
> direction is counter-intuitive:
> **distances in water are metric, and a desk test on land reads ~1.33× long.** A real 0.5 m
> showing up as `d ≈ 0.67 m` is normal. It does not matter when only the mask is used (that is just
> a pixel region).
> (2026-08-10 correction: this document and the CAMERA record were first written exactly the other
> way round, and that came from a stale comment in `c3_camera/geometry.py` that the audit had
> already marked as wrong.)

### Why there is no rectification

The original project moves between two coordinate systems, raw (1280×800) and rectified
(640×400), because the OAK-D W's 96° lens is off by **232 px** at the corners. The C3 does not have
such a lens — measured:

```
HFOV 63.9°   pinhole vs distortion: centre 0.13 px … corner max 2.08 px = 0.23° = 2.4 mm at 0.6 m
```

That is an order of magnitude smaller than the stereo depth noise. So
**RGB, depth, mask and clicks all live in one 640×360 coordinate system**. A whole class of bugs
disappears.

### The original is left untouched

`~/Desktop/New Folder` is **read-only**. `rov_gui/perception/session.py` imports that library
(`sam2_live.tracker` etc.) and rewrites only the orchestration — without the camera and the OpenCV
windows. Its `camera.py` (depthai 3.x) is **not imported**: the station already holds the camera,
and DepthAI allows only one pipeline per device.
The location is changed with `--pose-src` or `$SAM2_LIVE_ROOT`.

## Learned stereo depth — `--fstereo` (2026-09-02 **confirmed on hardware**)

The C3's **on-device depth is not metric in water**. It is 1.56x relative to the caliper and 1.71x
relative to the tag map, and the error **grows in proportion to distance**, so `--depth-scale 0.64`
is right at only one distance (KNOWN_ISSUES 2026-08-24). `--fstereo` runs **FoundationStereo**
(NVlabs) on the same mono pair and **replaces that run's depth source wholesale.**

```bash
./c3 gui --source hw --fstereo            # this run's depth is FoundationStereo (CUDA graph by default)
./c3 gui --source hw --fstereo --fstereo-size 224x224   # 7 ms faster, at the cost of one point of boundary accuracy
./c3 gui --source hw --fstereo --no-fstereo-graph       # the eager path that launches kernels one at a time
./c3 gui --source hw --fstereo --fstereo-process        # only the network runs in a child process (2026-10-12, unverified in water)
./c3 gui --source hw                      # this run's depth is the C3's own stereo
```

The checkout is the in-repo submodule `external/FoundationStereo`, and the weights are outside git —
on a first run, or if `FS FAULT … checkout not found` appears, start with the "FoundationStereo
checkout" section below.

### One instrument per run — there is no toggle

With the flag on, the depth panel, the cursor probe, the depth-vs-MAP check, `--pose`'s
FoundationPose and the recording **all** consume the learned depth. **The C3's own depth stream is
not even requested.**

That the GUI has no button for this is not an omission but the design:

* **Two instruments cannot mix in one run.** Two sensors spliced together in one recording file,
  one ratio line contaminated with the other sensor's numbers, or one `_pose.jsonl` holding two
  depths — these are **structurally impossible**, because there is only one depth in the process.
* **The race at the moment of switching disappears entirely.** Previously this needed an expiring
  ownership latch, an instrument label carried on each frame, and arbitration between two
  producers. Now there is one producer.
* **It is also the answer to why the first hardware run failed.** Back then the flag paid the whole
  pipeline cost while the model loaded only when a button was pressed; the pilot did not know of
  any reason to press it, and the log printed `rig ready`, so it looked normal.

To compare, **fly two runs** — that is a better experimental design in the first place. The meta's
`fstereo` block and `_pose.jsonl`'s `depth_source` stamp which run used which instrument.

### A failure does not silently fall back to another sensor

If the model does not load or inference fails 10 times in a row, it **stops and says so.** It does
not quietly slip over to the C3's own depth — partly because that stream does not exist at all, and
partly because silently swapping the instrument is something a measurement record must never do.

```
fstereo: STOPPED — checkpoint vanished. This run has no other depth source
(--fstereo does not stream the camera's own); restart without --fstereo to fly
on the C3's stereo.
```

### The panel always states its own status

One line at the top left of the depth panel, still visible after an image has been drawn:

```
FS LOADING — 3s
FOUNDATIONSTEREO  11.4 Hz  81 ms  valid 99%→95%
FOUNDATIONSTEREO  4.0 Hz  76 ms  valid 99%→95%  [only]
FS STARVED — no mono pair for 4.2 s
FS STOPPED — no depth this run
FS FAULT — FStereoError: no checkpoint
```

The `[yield]` / `[only]` label means `--policy-fs-schedule` (policy section below, 2026-10-01) is
on. In that case Hz is not the EMA of the arrival interval but **the number of frames finished in
the last 2 s / 2 s**, and during an `only` mission around 4 frames per second is normal
[유도: the policy takes 2 frames every period_s 0.5] — it is not a slowed-down network. For runs
without a gate (`free`) the line is the same as before. `[proc]` (or `[yield proc]`) means the
network runs in a **child process** via `--fstereo-process` (2026-10-12, records boundary — section
below).

> Why it is a separate paint pass rather than `_note`: `_note` is drawn **only in the NO SIGNAL
> branch**. Once even a single frame has been drawn, every explanation from the worker becomes
> invisible, and a feature that ran once and then stopped turns into **a STALE rectangle with no
> reason** — indistinguishable from "it was never on in the first place".

### The black stripes were a grid that was too big

On the first hardware screen the depth panel was covered in black lines (`valid 79%→46%`). It is
not the matcher's fault but **the projection grid's fault**. The rectified mono pair sees **82.6°**
while the colour lens sees only **63.9°** — **only 50.4%** of the mono samples fall inside the
colour frame. A 640×360 grid has 230,400 cells but only ~129,000 usable samples, so **in principle
it cannot exceed 56%.**

| align grid | Cells | Fill |
|---|---|---|
| 640×360 (old default) | 230,400 | **50.6%** ← stripes |
| 480×270 | 129,600 | 89.9% |
| **400×250 (current default)** | 100,000 | **100.0%** |

**Going bigger does not make it sharper; it adds holes.** The scatter lands on 400×250, and from
there it is **NEAREST-upscaled to the colour grid (640×360)** for output — matching the shape every
consumer already assumes, without creating distances that were not there. `session.py:891` refuses
reference-view collection when the depth and colour shapes differ, so without this match `--fstereo`
and `--pose` would be mutually exclusive.

### What still remained black was the **Z range cut** (fixed 2026-09-02)

Even after the grid was fixed, the hardware screen showed `valid 82%→88%` — meaning **the network
output itself** was only 82% valid, whereas the reference tool gave 100%. The network is the same,
so **our post-processing threw it away**. The culprit was the `(200, 15000) mm` range cut passed to
`depth_from_disparity`: the gripper handle 15 cm in front of the lens and the far tent wall were
being erased to 0. The reference tool has no such cut.

Checked on real stereo frames [측정: `/home/bdml/Desktop/data collection/videos/{61,66,12}`,
9 frames, 2026-09-02]:

| Cause of loss | Measured |
|---|---|
| Pixels where the network output disp ≤ 0.1 | **0.00%** |
| Colour-grid projection (border) | 0.2~0.35% |
| (200, 15000) mm cut | **all the rest** |

Removed the cut (`Z_RANGE_MM = (0, 65535)`). Values closer than 300 mm are merely painted in the
colour bar's bottom colour, which is exactly what the device depth was doing too. On real frames the
output fill is now **98.5~99.6%**.

### Frame rate — it has to be bought twice

The first hardware run was at 5 fps not because of inference but because the **input** was 5 fps.

**Input (link).** The mono pair is not compressed (2.05 Mbit/s per camera per fps). The pair's fps
is bought by **not receiving the device depth stream at all**:

```
colour + pair @12            =  59.0 Mbit/s ( 66%)
colour + pair @15            =  71.3        ( 79%)  ← current default
colour + pair @18            =  83.6        ( 93%)  tight
colour + pair @20            =  91.8        (102%)  OVER
colour + depth + pair @15    = 126.6        (141%)  OVER  ← impossible if the device depth is kept
```

**Compute (GPU) — the time goes into the kernel count, not the pixels.** 9 real 640×400 pairs,
eager, disparity mismatch relative to the reference setting (iters 16, 640×400) [측정 2026-09-02,
RTX 5090 idle: `rov_gui/tools/fstereo_bench_out/sweep.txt`, reproduce with
`python -m rov_gui.tools.fstereo_bench sweep`]:

| Input | iters | eager ms | Kernel count | med \|Δ\| | >1 px fraction |
|---|---|---|---|---|---|
| 640×400 (scale 1.0) | 8 | 113.3 | 6,854 | 0.029 px | 1.8% |
| 480×300 (0.75) | 8 | 74.9 | 6,839 | 0.099 | 7.0% |
| 320×200 (0.5) | 16 | 58.8 | 9,580 | 0.161 | 10.6% |
| **320×200 (0.5, default)** | **8** | **44.2** | 6,572 | 0.165 | 11.1% |
| 320×200 (0.5) | 4 | 36.8 | 5,068 | 0.184 | 11.3% |
| 256×160 (0.4) | 8 | 37.8 | 6,427 | 0.228 | 12.8% |
| 224×224 (`--fstereo-size 224x224`) | 8 | 39.1 | 6,347 | 0.222 | 12.3% |
| 224×140 (0.35) | 8 | 37.7 | 6,355 | 0.299 | 17.0% |
| 160×100 (0.25) | 8 | 37.1 | 6,167 | 1.028 | 50.6% |

How to read:

* Even cutting the pixels to 1/4 by going from width 320→160 only gets 44→37 ms. One forward pass
  is **6,572** kernels, and 44.2 ms/6,572 = **6.7 µs/kernel** [유도] — on the order of the CPU cost
  of launching one kernel. At this size the bottleneck is not the GPU but **the rate at which
  Python launches kernels one at a time**. nvidia-smi at 49~53% on the user's screen
  (2026-09-02 09:05) says the same thing: even during the solve, the GPU sat idle half the time.
* So **resolution is not the lever.** 224×224 buys 5 ms and pays one point of boundary accuracy
  (>1 px 11.1→12.3%). Isotropic at width 224 (224×140) is worse (17.0%) — even at the same width,
  having fewer rows shrinks the context of the 1/4-resolution cost volume. Width 160 collapses
  (50.6%).
* The accuracy lever is still scale (1.0 → 1.8%, 0.75 → 7.0%, 0.5 → 11.1%), and iters only cost
  time (at 0.5, 16 vs 8 = 0.161 vs 0.165 px, 58.8 vs 44.2 ms). With this baseline, 1 px at a
  distance of 1 m is ~3.7% of the range [유도].

**CUDA graph — ON by default, turned off with `--no-fstereo-graph`.** If launching kernels one at a
time is the bottleneck, the answer is to record once and replay. `FStereoSession` captures the
entire forward as a `torch.cuda.CUDAGraph` on the first frame, and after that only copies the input
in and calls `replay()` [측정: `fstereo_bench_out/graph.txt`, `session.txt`]:

| Input | iters | replay ms | eager ms | Output (9 frames) |
|---|---|---|---|---|
| 640×400 (1.0) | 8 | 105.2 | 114.9 | identical |
| 480×300 (0.75) | 8 | 67.2 | 74.9 | identical |
| 320×200 | 16 | 48.3 | 60.5 | identical |
| **320×200 (default)** | **8** | **36.8** | 45.4 | **identical** (`np.array_equal`) |
| 320×200 | 4 | 31.0 | 37.9 | identical |
| 224×224 | 8 | 30.5 | 38.7 | identical |
| 224×140 | 4 | 22.7 | 29.3 | identical |

ms is one call of resize + H2D + forward + D2H + resize-back. Capture on the first frame takes
**1,332 ms** in a fresh process (including 3 warm-ups, `session.txt`), and ~0.2~1 s from the second
size onward in the same process.

Over the whole station path (rectification → network → mm → 400×250 scatter → 640×360):
**graph 44.9 ms = 22.3 Hz**, eager 52.0 ms, 9 depth maps at **max |Δ| 0 mm**, native valid 100.00%
→ colour grid 99.64% (scatter loss 0.36%) (`session.txt`). 224×224 is 38.6 ms (26 Hz).

For capture to work, five host↔device round trips inside the forward had to be removed
(`perception/fstereo.py:_patch_for_capture` — the upstream checkout is left untouched, and the
imported modules are patched in this process only): the per-call CPU tensor in `normalize_image`,
`model.dx` (not a buffer, so `.cuda()` does not move it), the autocast weight cache, the
`torch.zeros(shape)` in timm EdgeNeXt's positional encoding, and, in `bilinear_sampler`, **the
`torch.unique` assert that runs on every GRU step** (which was synchronizing eight times per frame
even in eager). A capture failure is logged once and that run goes eager (same numbers, ~8 ms
slower). The meta's `fstereo.cuda_graph`/`cuda_graph_error` record it — a record written before the
first frame says `false`, but that is not a failure; it simply has not been stamped yet.

> **Why this matters on the hardware screen.** The 2026-09-02 09:05 screen
> (`--fstereo --pose --mpc`, no other process on the GPU): `11.4 Hz / 81 ms / 169.6 ms`. Idle
> standalone eager with the same settings is 44 ms. The 37 ms difference is the **GIL** — an eager
> forward is 6,500 Python calls, and the video worker, the MAVLink parser and the tag detector cut
> in between them. A graph replay is **a single call**, so there is no gap to cut into → inside the
> station too, something around ~45 ms is expected [예측, unconfirmed on hardware].

> **Breakdown of the 170 ms latency** [유도: screenshot + link calculation]: raw pair transfer
> ~46 ms (256 kB × 2 frames, 22.8 ms per frame at 90 Mbit/s) + mailbox wait ~40 ms (solve 81 >
> period 67, so a frame waits half a period on average) + solve 81 ms. If graph brings the solve
> below the period, the wait term disappears, giving **~100 ms** [예측]. The worker tick was also
> lowered 20 → 5 ms (in an input-limited state the tick period is directly the wait time). Below
> that is the transfer, and that is the next item.

> **20 Hz is still blocked by the link — the key is mono MJPEG, not implemented.** The raw pair is
> 82 Mbit/s at 20 fps. Mono is 8-bit, so JPEG can be applied with the device encoder
> [측정: `link.txt`, based on host libjpeg-turbo — the device encoder's ratio is [예측]]: **4.7×** at
> q90 (pair at 20 fps = 17.6 Mbit/s, decode 0.36 ms/frame, PSNR 42 dB). The worry about matching
> quality was measured [`jpeg.txt`]: the disparity change after a q90 round trip is **median
> 0.025 px, >1 px 0.4%** — buried under scale 0.5's own reference mismatch (11.1%). Besides, the
> land UMI dataset's left/right.mp4 are already **H.264 19 Mbit/s** (ffprobe,
> `data collection/videos/61/left.mp4`), and FS depth is being extracted from them and used. The
> pipeline has never attached an encoder to mono (`StreamConfig` has no `mono_encode`, which is
> exactly the KNOWN_ISSUES entry of 20 failing tests). Turning it on would bring the transfer
> 46 → ~5 ms [유도] and open up 20 fps. It is a change to
> `c3_camera/config.py`·`pipeline.py`·`hardware.py` and needs hardware confirmation on the camera, so it is
> **at the proposal stage**.

### When handing off to FoundationPose, pairs are matched by timestamp

The learned map arrives one inference (~45 ms) later than the mono pair, and at the input cap of
15 Hz against the colour's 30 fps. So the RGB-D tap **pairs by capture time and rejects anything over
250 ms** — the tracker knows how to handle "no depth this frame" but cannot deal with "confidently
wrong depth". `skew_ms` is recorded as is.

### The device's rectified pair is not used — **raw is rectified on the host**

The original reference tool takes the device's `rectifiedLeft/Right` and gets metres with
`getCameraIntrinsics(CAM_C)[0][0]`. That is the **unrectified fx**. DepthAI 2.32 has no API for the
rectified focal length, so using the EEPROM fx instead produces a **~0.4% systematic scale error** —
small enough not to look broken and large enough to ruin the map. So the raw pair is received and
rectified here with `cv2.stereoRectify` in `c3_camera/host_depth.py`.

```
raw left/right (640x400)
  -> rectify     host_depth  StereoRig.map_left/map_right
  -> disparity   FoundationStereo
  -> uint16 mm   host_depth  depth_from_disparity     (d<=0 -> 0, never inf)
  -> colour grid host_depth  warp_depth_to_color + resize_depth_nearest
```

`host_depth.py` already existed but was **code nobody used at runtime**. Thanks to reusing it, the
learned depth has a contract byte-for-byte identical to the device depth's (uint16 mm, 0 = no
measurement).

> **Fixed one cv2 5.0 trap.** `cv2.stereoRectify` needs T as `(3,1)`, but 4.x also accepts a flat
> `(3,)`. The GUI env is cv2 **5.0** and `robust` is 4.10, so a flat vector **passes the tests and
> dies only on the vehicle**. This module being dead code is why it stayed latent.

### The black cobwebs were the warp's **resampling gaps** — `--fstereo-fill` (2026-09-02)

Using 3 hardware frames, the black lines that remained in the panel (1~3 px curves running along
depth slopes) were attributed stage by stage to what they really were
[측정: `rov_gui/tools/fstereo_bench_out/hardware_20260902_fill.txt`]:

| Stage | Contribution | Verdict |
|---|---|---|
| Rectification (`cv2.remap`) margin | **0.00%** | alpha=0 crops to the valid region → rect-valid 100.00% |
| Network invalid (disp≤0 or over 65 m) | 3.06% | **all outside the colour FOV** — even with the source filled entirely with valid values, the output fill stays unchanged at 95.32% |
| Forward-scatter gaps | **4.68%** | 539 connected components, all thin curves. This is all of it |

The cause is simple. The scatter writes **one** destination pixel per source pixel. Where a surface
is tilted or the depth bends, two neighbouring source samples land more than 1 px apart at the
destination, and nobody writes the pixels in between. **It is not that the information is missing;
the samples just did not reach there**.

`c3_camera/host_depth.py:fill_scatter_gaps` writes a neighbour's actual mm **only into empty pixels**
(grayscale dilation applied to holes only). It does not use averaging, interpolation or inpainting —
so as not to create the "distance where there is nothing" that `resize_depth_nearest` forbids. There
is a rationale for the dilation using the **maximum** (the farthest neighbour), too: if the gap is a
resampling gap, the neighbours are all at the same distance, so it makes no difference, and if it is
a real disocclusion, the truth at that spot is the **background**, i.e. the far side. Using the near
side would smear the foreground object into places where it does not belong.

Result on the 3 hardware frames: in 2 passes, **fill 95.32% → 100.00%, 0.14 ms, 0 already-measured
pixels changed**.

> **The HUD does not mix the two.** The chip reads `valid 97%→95%+5` — 95 is the measurement actually
> projected, +5 is the filled share. Writing the sum as 100% would report a repair as a measurement,
> which is exactly what this repo forbids. With `--fstereo-fill 0` it comes out raw, as is.

> **Why it looked different from the reference viewer.** `oakd_foundation_stereo.py` draws depth in
> **the mono camera's own frame** — there is no warp, so these gaps cannot occur. The station
> reprojects onto the colour grid, because the cursor probe, the colour overlay and FoundationPose
> all assume "same pixel = same ray", and the price of that was these gaps.

### Hardware confirmation — 2026-09-02 11:30, C3 + ArduSub, no commands sent

Two runs of 75 s each with `--source hw --fstereo --mavlink-transport none` (no command sink, no
`--pose`/`--mpc`). The artifact is `rov_gui/tools/fstereo_bench_out/hardware_20260902.txt`.

| `--depth-fps` | FS delivery | solve | Panel latency | valid | Mono drops | C3 link |
|---|---|---|---|---|---|---|
| **15 (default)** | **fixed at 15.0 Hz** | 46–47 ms | 101–106 ms | 97%→95% | 0.3% | 70.5 Mb/s (78%) |
| 10 | fixed at 10.0 Hz | 46–47 ms | 105–112 ms | 97%→95% | 0.5% | 50.2 Mb/s (56%) |

How to read:

* **The CUDA graph is captured on hardware** — `forward captured as one CUDA graph in 1328 ms
  (input 320x224, iters 8)`, `eager=False`. Measured 46–47 ms against the idle bench's 44.9 ms
  prediction, so "~45 ms inside the station too" [예측] held. With the same settings, the measured
  value before graph was **81 ms**.
* **Latency 170 → 105 ms.** The solve dropped below the input period (67 ms), so the mailbox-wait
  term disappeared.
* **The link can handle 15 Hz** — mono pair drops 0.3%. What was written earlier, "tight at 79%",
  is correct, but frames are not actually lost. There is no reason to lower it to 10 fps.
* valid 97%→95% is for an indoor scene in air, and what makes it lower than the land recording
  (100%→99.6%) is the part on the left of the screen — a nearby metal shelf and cables — where the
  network could not give a disparity.

> **In the first run it once dropped 14.6 → 6.7 Hz** (11:23, the run with 5 screenshots). It was not
> reproduced in the two runs that followed, and the drop counter was not being logged at the time,
> so the cause is unknown. **It is left as cause unknown** — if it shows up again, look first at the
> fps and drop% of `c3[left]/c3[right]`.

### When turned on together with `--pose` — the two checkouts' `Utils` have the same name (fixed 2026-09-02)

FoundationStereo and FoundationPose each have a **top-level `Utils.py`** and import it bare
(`core/extractor.py:17` / `estimater.py:10` plus 8 other files, all at import time). In one process,
if whichever is imported first takes `sys.modules["Utils"]`, the later one **gets the wrong
module.** Under `--fstereo --pose` the stereo comes up first at startup and FoundationPose is
imported on the first click (registration), so FoundationPose's `from Utils import *` gets the
stereo side's 8 functions and its own 31 helpers hit NameError — **tracking dies on the first
registration, and the startup log shows no trace of it.** The 2026-09-02 09:05 run
(`--fstereo --pose --mpc`) was exactly this combination, and it went unseen because no registration
was done.

How it was fixed (`perception/upstream.py`, `fstereo._import_upstream`, `session.start_pose`): the
stereo import owns the bare name inside a shared lock **only for the duration of the import**, gives
it back, and then sends its own checkout to the **end** of `sys.path` (`core` has no deferred
imports, so after that it needs neither the name nor the path — checked with grep). FoundationPose
is imported up front inside the same lock, so that the worker uses the cached module. The tests cover
both orders with two fake checkouts
(`test_the_stereo_import_does_not_keep_foundationposes_utils_name`). **On hardware it has not yet
been taken as far as registration.**

### `--depth-scale 0.64` is **not applied** to FS

0.64 is a constant fitted to the device block matcher's bias. FS converts its own disparity with its
own fx_rect/baseline, so multiplying would **correct what is already corrected, a second time**. The
very question of this experiment is *"does 1.0 come out without correction?"*. `_pose.jsonl`'s
`depth_scale_applied: 1.0` and `depth_source:
"foundation_stereo"` are that boundary.

### Measured (2026-09-02, RTX 5090 idle, 9 real 640×400 pairs — not hardware)

| | | Evidence |
|---|---|---|
| Model load | **2.1–5.1 s** (vitl @ 23-51-11, 3.07 GiB; the first load is 5.1, 2.1–2.4 from the page cache) | `fstereo_bench_out/session.txt` |
| First inference (including CUDA graph capture) | **1,332 ms** | `session.txt` |
| End-to-end thereafter, defaults (graph, scale 0.5, iters 8) | **44.9 ms ≈ 22.3 Hz** | `session.txt` |
| Same settings, eager | 52.0 ms | `session.txt` |
| graph vs eager depth map | max \|Δ\| **0 mm** | `session.txt` |
| Colour-grid fill | native **100.00%** → **99.64%** (align 400×250, scatter loss 0.36%) | `session.txt` |
| Disparity vs reference (defaults) | med 0.165 px, >1 px 11.1% | `sweep.txt` |

> The frames are an OAK-D-W-97 land recording (same OV9282, different lens), so the absolute
> disparity is not the C3's. Cite only **comparisons** between settings. The synthetic planar
> fixture numbers (2026-09-01) were taken out of this table — a plane has no depth boundaries, so it
> cannot show the cost of scale. [[measurement-citation-rule]]

### Records boundary

`schema_version` in `controller.json` / `.meta.json` goes **8 → 9**, and the `fstereo` block is
always written, even when it is off (`iters`, `scale`, `infer_size`, `cuda_graph`, `cuda_graph_input`,
`cuda_graph_error`, ckpt sha1). The `CAMERA` row of `_pose.jsonl` gets `depth_source`.
**Do not pool distances that come from depth with other runs before checking their source.**

From 2026-10-01, `fstereo.schedule`: `null` = no gate (`--policy-fs-schedule free`, the default);
if present, a gate snapshot (mode, counters, wait·hold distributions). If the key is missing
altogether, it is a build from before this switch or a run without an FS worker. With a gate,
`measured_hz` is the 2 s window count at the moment the meta is written (with `only`, ~4 during a
mission and the free-running rate after the mission — `schedule.mission_active` says which). Do not
pool FS rate·plan age·infer_ms across runs with different schedules.

From 2026-10-12, `fstereo.process` is always written as well: `{"mode": "in-station"}` (default) or,
with `--fstereo-process`, `{"mode": "child", "pid", "alive", "exit_code", "rpc_ms", "ipc_ms",
"first_frame_ms", "rig_sends", "timeouts_s", …}`. `rpc_ms` is the round trip the worker waited for, and `ipc_ms` is
that minus the `solve_ms` measured by the child. `solve_ms` means the same in both placements
(the wall clock of rectification→network→warp→fill, measured inside the child). Every policy plan
line (plans.jsonl) carries `fs_process` ("child" / "in-station" / None if not FS). **Do not pool plan
age·infer_ms·FS rate across runs with different placements.** `schema_version` stays at 20 (keys
added only).

From 2026-09-30, `fstereo.repo`/`fstereo.ckpt` point to `…/external/FoundationStereo/…` (earlier
runs: `~/Desktop/FoundationStereo/FoundationStereo/…`). **The weights are the same** —
`ckpt_sha1_first_8mib` is `d277c7ff…` on both (checkout section below). Compare instruments across
runs by this sha1, not by the path.

### Environment

`rovgui-pose` needs only three more (torch 2.11+cu128 is already there, sm_120 included):

```bash
~/miniforge3/envs/rovgui-pose/bin/pip install timm einops huggingface_hub
```

The dinov2 hub archive and the timm `edgenext_small` weights are already in `~/.cache`, so **the
first run needs no network access** (if `~/.cache` is cleared, the first load needs network access).

### FoundationStereo checkout — `external/FoundationStereo` (since 2026-09-30)

The checkout is **a git submodule inside this repo**: NVlabs/FoundationStereo `6e88068`
(upstream master — no commits after 2025-12-18, `git ls-remote` 2026-09-30), `shallow = true`.
For a freshly cloned repo:

```bash
git submodule update --init external/FoundationStereo
```

Always give the path: without a path, `git submodule update --init`/`git submodule status` dies with
rc 128 because of pre-existing gitlinks that have no `.gitmodules` entry (`external/UMI_aquatic`,
`external/iPhUMI`). Also, `git clone --recurse-submodules` passes `--no-single-branch` to the
submodule clone, so even with `shallow = true` it fetches the upstream `website` branch (glb/mp4
assets) as well and ends up much larger than master alone [유도: the 2026-09-30 review reproduced
it with a toy repo; size artifact not kept] — to use it, also pass `--single-branch`, or do a plain
clone followed by the command above.

**The weights are not in git** — upstream `.gitignore` excludes `pretrained_models/*`. Put
`model_best_bp2.pth` and `cfg.yaml` into `external/FoundationStereo/pretrained_models/23-51-11/` by
hand (the Google Drive from the upstream readme; the gdown id of this file is
`1Yh_2o9QCUrVqZrnAXZ7RUr0zTp3JrMKe`, `~/Desktop/data collection/depth_job/pipeline.sh:20`).
Whether it is the same file is checked with the sha1 of the first 8 MiB — the same value as
`ckpt_sha1_first_8mib` in a run's meta, and all 423 GUI runs before 2026-09-30 recorded
`d277c7ff7de5c627328cde9eaf775e181685276a` (`fstereo.ckpt_sha1_first_8mib` in
`data/2026*/*/*.meta.json`):

```bash
python -c "import hashlib; print(hashlib.sha1(open('external/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth','rb').read(8<<20)).hexdigest())"
```

`11-33-40` (vits) is also there alongside, but nothing uses it. For a different checkout use
`--fstereo-repo DIR` or `$FOUNDATION_STEREO_REPO`; to change only the checkpoint, `--fstereo-ckpt`
(needs a `cfg.yaml` next to it).

**Why inside the repo.** The previous default was `~/Desktop/FoundationStereo/FoundationStereo`,
**outside** the repo, and that path existed only in the code (`DEFAULT_REPO`) — it was in no
document. So it looked unneeded and was deleted on 2026-09-27 (at 22:13 that day the depth job
failed with "checkout not found": `~/Desktop/data collection/depth_job/run.log:3`), and the GUI
depth panel showed `FS FAULT — FStereoError: FoundationStereo checkout not found at …`. The current
weights are a byte-identical copy taken from
`~/Desktop/data collection/FoundationStereo/pretrained_models/` (`cmp`, 2026-09-30). That copy is
the 2026-09-28 re-download, and over 699 frames of ep7 its depth was bit-for-bit identical to the
depth made with the pre-deletion weights (`~/Desktop/data collection/depth_job/verify_result.txt`).
**Do not delete `~/Desktop/data collection/FoundationStereo`** — it is the default for
`UMI_Underwater`'s `replay_foundation_stereo.py`·`oakd_foundation_stereo.py` and for
`depth_job/*.sh`.

## Closed-loop MPC — `--mpc` (AprilTag localization + acados NMPC)

```bash
./c3 gui --source hw --allow-command --mpc                  # hardware (wall geometry)
./c3 gui --source hw --allow-command --mpc --nav-geometry floor
./c3 gui --source demo --mpc                                 # synthetic plant closed loop
~/miniforge3/envs/rovgui-pose/bin/python -m rov_gui.control.smoke   # P0 bench gate
```

This is the closed loop that attaches the sim's DOB-MPC (`bluerov2_mujoco_marinegym/dobmpc/`, which has
no MuJoCo dependency and is therefore **imported as is**) to the hardware. The whole pipeline lives
inside this process — because the C3 and the command sink each have a single owner:

```
C3 colour frame ─► TagNavWorker (pupil_apriltags + solvePnP, fixed map, no GTSAM)
                     └► bus.nav_fix  (vehicle pose, NED tag world)
ArduSub MAVLink ─► VehicleWorker (20 Hz tick with --mpc) ─► bus.vehicle_imu
                     └► StateAssembler: eta/nu/nudot (NED/FRD, 20 Hz)
                          └► EAOB + acados NMPC (mpc_bridge, DT_CTRL=50 ms)
                               └► wrench → MANUAL_CONTROL 4 axes (K/M discarded)
                                    └► existing MavlinkCommandSink — inherits all the gates
```

The config is `config/hw_nav.yaml` (tags/geometry/extrinsic) and
`config/hw_mpc.yaml` (mode/axis gains/EAOB sigma/square). Every value from before measurement carries
the `[예측]` tag — in particular, **the axis gains are estimates until the P4 step calibration**.

Operating discipline (the MPC panel is **inside the main grid** — since 2026-08-14 it occupies **all
of columns 2·3 of the bottom row** (the old PROPULSION·SENSORS spot, ~790×376 px), and PROPULSION and
SENSORS go **side by side** under SYSTEM HEALTH in column 3 (stacked vertically, the minimum height
would exceed the 1080p budget — pinned by `test_mpc_panel_lives_...`). The plot is an NED top-down
view with axes·ticks in m, and **the pool boundary (`hw_nav.yaml: pool_ned`) is both the solid border
and the basis of the axis scale**. Because of this layout, the minimum window height with `--mpc`
exceeds 768 — the default layout's one-screen (1366×768) promise is unchanged, and `--mpc` presumes
a 1080p operating screen):

* **The mission is set in the top row of the panel** (2026-08-14; circle 2026-08-17):
  `[station|line|square|circle|follow] from tag [id] [facing tag [id]] [거리|반지름] [속도]`
  ("[station|line|square|circle|follow] from tag [id] [facing tag [id]] [distance|radius] [speed]").
  **Fields that are not used disappear** — station does not show distance/speed at all, and follow does
  not show `from tag` and distance at all (follow uses **not a single** tag id: its reference is the
  vehicle pose and the object pose at the moment of arm. Before 2026-08-23 they stayed grey and looked
  like "fields that still do something").
  **station** = go above that tag and **stay still**
  (no path, bearing held too) — **fly this first**.
  **`facing tag`** (2026-09-11, shown only for station) = stay above that tag while **looking toward
  another tag**: with `from tag 47, facing tag 97` the heading is the 47→97 bearing in the map. The two
  ids mean the same physical direction every time, so "which way was it facing" does not vary from run
  to run (`hw_mpc.yaml square.heading_tag`, `station.heading_tag` in the run meta). With 0 it is
  `yaw_map_deg`/`yaw_fixed_deg`, as before. If the heading tag is not in the map, is in two places, or
  is the same as the origin tag, START is refused with that reason. It is also refused **when engaged
  in STABILIZE** (`engage.stabilize.yaw_axis:
  hold` — the station sends 0 on the yaw axis and the autopilot's heading hold holds the heading): so
  that a heading that cannot be flown is not written into meta as "kept". Only **engaging in MANUAL**
  makes facing tag actually turn the vehicle. A vehicle that cannot hold a single point
  cannot follow a line either, and if you run line in that state, every number that comes out ends up
  measuring that. **line** = start directly above that tag, go the entered distance in the `dir_deg`
  direction (default 90° = map +y) and **come back: a round trip** (1 lap = 1 round trip, the count is
  `hw_mpc.yaml: laps`). **square** = a **rectangle** with that tag as its first corner
  (x side × y side entered separately). **circle** = a circle with that tag as **the bottom point of
  the circle** (the distance field becomes the **radius**, with an `r` prefix; 1 lap = 1 revolution).
  For all four shapes, the tag is **a point the path passes through**, not a centre circled from afar —
  the circle's centre is placed one radius away from the tag toward map +x (up on the plot), and
  `rot_deg` rotates that centre around the tag. Tag id 0 = "from where it is now".
  **Speed (m/s)** is the target path speed for line·square·circle and is hidden for station
  (added 2026-08-16). The actual
  path mode projects the vehicle onto the currently active curve and gives the bounded lookahead ahead
  of that point to PID and MPC alike. For polygons, the input speed is **the target for the straight
  segments**, and at corners the speed profile slows down in advance. **circle is the opposite**: the
  curvature applies over the whole lap, so the input speed is only a **cap**, which the geometry may
  not allow (the item right above).
  These fields are written over the `square:` block of hw_mpc.yaml via `cmd_mpc_scenario` (the file
  values are the panel's starting values), and the lap count·depth·ramp come from the file.
  **The mission is snapshotted the moment START is pressed**, so the fields are locked not only while
  the path is being flown but also during the approach·settling phases — because values edited then
  would not actually be flown.
  * **The bearing is `yaw_map_deg`** (default 90 = facing map +y), and **line's direction of travel
    `dir_deg` is also an absolute map bearing** (90 = the pool's long direction, +y). Had either been
    datum-relative, it would mean a different physical direction every time the engage heading
    changed — that was in fact the cause of `dir_deg: 90` being drawn as −x on 2026-08-14.
  * **The chip says which phase it is in now**: `WARMING UP` → `GOING TO START 0.83 m
    to go` → `SETTLING 6 s` → `STATION HOLD` / `LINE` / `SQUARE`. Previously it showed
    `DP HOLD` even during the approach, so the phases could not be told apart.
  * **The approach ramps the setpoint** (`engage.approach_speed_m_s` 0.10,
    `approach_lead_m` 0.35): teleporting the target point by 2 m is, to the position controller, a
    full-authority step command, so the vehicle lurches out and then sloshes back and forth. A hard
    clamp is also applied so that the setpoint never gets more than 0.35 m ahead of the vehicle.
  * **START is not refused even when far away** (2026-08-14): it flies with DP to above that tag,
    waits at rest for `engage.settle_s` (default 10 s), and only then does the path start. The transit
    also uses the same controller·same interlocks, so no separate, looser transit mode comes into
    being. The arrival criterion is `engage.start_err_max_m` (0.3 m), and if the vehicle is pushed
    outside that while waiting, the timer resets. Once `approach_max_s` (180 s) has passed, it gives up
    and holds in place. STOP/DISENG also cancels an approach in progress. With `settle_s: 0` it starts
    immediately, as before (when it is above the origin).
  * **A duplicate id cannot be used as the origin** (it is in two places, so "above that tag" is
    ambiguous). It is refused.
  * The turnaround point of line gets a **cosine speed ramp** (`ramp_s`, default 1 s). Reversing the
    velocity within 1 sample, as a triangle wave does, puts a floor under the tracking error for the
    same reason as the square corners (memory: square-corner-error-floor) — only when the reference is
    feasible does reference-vs-actual become a number that measures the controller.
  * **There is no geofence** (removed 2026-08-14, pilot request). Previously, before START, the line's
    two endpoints / the square's four corners were checked against a box, and START was refused if
    they fell outside it. Now **nothing at all prevents path placement** — even a 9 m line that goes out
    of the pool is armed as is. The only remaining protections are E-STOP·DISENG·sink deadman
    (500 ms)·the engage gate (ARMED / MANUAL / fresh tag fix·telemetry)·automatic release on disarm
    while running·telemetry stop·mode exit·tag dropout, and the axis authority caps.
* **START (hold) is a one-button flow**: engage → warm-up → square starts automatically, and CSV
  recording runs from the moment of engage. The preconditions are all still checked: COMMAND
  ENABLE·ARMED·flight mode MANUAL·acados real-time (probe <25 ms)·tag fix fresh (<0.5 s)·IMU fresh.
  If even one is missing, it states the reason and refuses. (The geofence check was dropped on
  2026-08-14.)
* **HOLD (hold) is a DP-only engage** (for calibration/station-keeping). STOP TRAJ·DISENG·
  E-STOP are all single clicks — stopping has no gate.
* **AprilTag detection is switched on and off with the per-feed TAG button**. **Which feed handles
  localization is decided by `hw_nav.yaml: nav_source`**: `main` = C3 (factory underwater calibration
  built in), `second` = the ROV's default RGB (`second_cam` block — **all [예측]**; if fx is wrong, every
  distance is scaled by that ratio). Tags on the localization feed get a solid green outline, those on
  a non-localization feed a faint outline. Turning off TAG on the localization feed disengages safely
  via stale-fix, and **turning it back on is the datum re-zero** gesture (below).
* **(0,0)·yaw 0 = the pose at the moment START (engage) is pressed** (mission datum, always on).
  It is re-zeroed on the spot at every engage, and square·geofence·plot·CSV all live in this relative
  coordinate frame — EAOB/NMPC are also born in this frame, so they do not jump midway.
  The datum for recovering absolute (tag frame) coordinates is kept in meta.json
  `hardware.datum_tag_frame`. `fix N Hz · detect M ms` at the bottom left of the plot is the
  localization update rate — if it is slow, the more you raise `hw_nav.yaml: quad_decimate`
  (default 2.0), the faster it gets, while the corner precision drops slightly. Current mission setup
  (2026-08-13): **the gantry tagslam's 47-tag floor map** (`config/tag_map.yaml`, anchor = **tag 25** =
  map origin) with multi-tag joint PnP on **C3 RGB** (`geometry: floor`, `nav_source: main`,
  `min_tags: 1`) — the C3 has the factory underwater calibration built in, so the scale is correct.
  With 2 or more tags it is the path with neither flip ambiguity nor a gravity gate, and **with only 1
  it is solved with IPPE + the gravity gate** (the flip is judged from ATTITUDE, `tilt_gate_deg`) — on
  2026-09-07 `min_tags` was lowered from 2 to 1 (operator decision). Which feed is localizing is shown
  in real time by `loc C3 RGB · fix N Hz · detect M ms` at the bottom left of the plot and by the
  `TagNav [C3] …` line in SENSORS. quad_decimate is per feed (main 1.0 / second 2.0 — the 640×360 C3
  loses distant tags when decimated). **The 47 map tags are laid out, as in the old tagslam
  visualization, as squares of their actual size·orientation**, and **only the tags that made the
  current fix are painted green** (they go dark if the fix drops out for 1 s or more — a dead
  localization cannot keep tags lit). While not engaged, the fix drives the marker directly (it moves
  even on the bench without ArduSub telemetry). The wall single-tag (104) mode is restored with
  `geometry: wall`. 20 cm square @ 0.05 m/s, yaw fixed at the start heading.
* **The actual trajectory is coloured by age (time)**: a viridis ramp (dark purple = old → yellow =
  now), with a `-90s → now` legend at the bottom right. Points older than 90 s (`trajectory.py:
  TRAIL_AGE_S`) gradually turn transparent and are erased — the time order stays readable even on
  crossing paths, and the old trajectory does not cover the screen. The reference trajectory stays
  solid blue.
* **REC NAV (button on the plot panel) = raw localization record**. Regardless of engage (manually
  piloted surveys included), it leaves four files in `nav_<hhmmss>/` inside the run folder: `map.json`
  (the locked tag map id→x,y,z,yaw + tag size + pool + geofence), `fixes.csv` (**raw fixes in the tag
  frame** — before the datum transform; failed frames are also recorded, **with the tag ids** and the
  reason), and `detections.csv` + `frames.csv` (**raw observations** — the 4 corner points of every
  detected tag and that frame's camera model). The latter two are the material for rebuilding or
  extending the map — if only the solved fixes were kept, the corners would be thrown away and nothing
  could be done.
  `python -m rov_gui.tools.plot_nav_run <런폴더>/nav_<hhmmss>`
  ("python -m rov_gui.tools.plot_nav_run <run folder>/nav_<hhmmss>") redraws the same picture
  (tag squares + time-coloured path + pool boundary, matplotlib).
* **Duplicate tag ids** (measured 2026-08-14): the pool mat is a 20×7=140-cell custom layout, and
  **12 ids are in two places** (`hw_nav.yaml: duplicate_ids`). The map holds only one pose per id, so
  if the corners of "the other copy" get mixed in, the joint PnP blows up (measured 59–131 px). The
  mitigation is **confirm-or-drop**: first establish the pose with the unique-ID tags, then accept a
  duplicate tag only when it lands within `dup_confirm_px` (6 px) of its map position. **The key is
  that the other copy's position need not be known** (it only has to be recognized), so no re-survey
  is needed. If the same id is picked up twice in one frame, both are discarded — an undeclared
  duplicate reveals itself, and `min_tags` ends up counting only unique ids. **It works the same even
  with only 1 unique-ID tag** (2026-09-07): that 1 tag establishes the pose, the duplicate tags attach
  to it via confirm-or-drop, and the result is a 3-tag fix. Before that, a view with only one
  unique-ID tag, like `{60, 65, 141}`, was thrown away entirely — that accounts for **372** of the 390
  rejections in `data/20260906/0906_192348/nav_192348`, and it created a 10.9 s continuous fix gap.
* **The default RGB camera fills the C3's gaps** (2026-09-07, operator request;
  `control/nav_fusion.py`, `hw_nav.yaml: fallback`). The C3 is the primary localizer, and the ROV's
  default RGB (`second_cam` — position·fx·tilt all [예측]) also **solves separately** every frame in
  which tags are visible. The two cameras are in different positions, so the two answers differ — that
  difference is **a constant transform E in the body frame** (`T_rgb = T_true · E`, independent of the
  vehicle pose, as long as the mount does not move), so it is learned from time pairs (`pair_tol_s`)
  whenever both produce a fix at the same time, and when the C3 is silent for `after_s` (0.2 s), **the
  RGB fix with E divided out** is handed over. RGB fixes from before learning are **withheld** (default
  `allow_unaligned: false`) — a brief gap is better than the MPC flying off after a 5–20 cm step. E
  also **measures the actual tilt** of the RGB mount (`measured:tags` in PAYLOAD). A handed-over fix
  has `NavFix.source: second`, the SENSORS line shows `[RGB↩]` + yellow, and the note shows the
  alignment status (`aligned n=… offset … mm`). The TAG button of the second panel is ON by default
  (orange border = standby fallback), and the learning state is kept in `controller.json`
  `nav_fallback`. Synthetic verification: an RGB with a planted 40° tilt + 5 cm error is recovered to within
  5 mm after 8 pairs (test_control). **First hardware run (2026-09-07 14:52)**: the alignment worked
  (`offset 67 mm spread 11 mm`, 63 fixes), and **what blocked it was the RGB camera model** —
  `second_cam` had no distortion terms, so multi-tag RGB frames all died at the 3 px gate and only
  single tags got through (`0907_145206/nav_145240/fixes.csv`, 58 of the 63 RGB fixes had n_tags 1).
  So (a) `fallback.max_reproj_px: 8.0` ([예측]) is a **separate gate** for the RGB solver, and if the
  RGB fails while the C3 is silent, the reason appears on screen as `[RGB] N seen: …`; (b) REC NAV
  saves the fallback feed's raw observations separately as `frames_second.csv` /
  `detections_second.csv`; and (c) **`python -m
  rov_gui.tools.calibrate_second_cam <run>/nav_HHMMSS --yaml`** uses them to fit the RGB intrinsics
  (fx, 5 distortion terms) — the mat itself is the calibration target. By default the principal
  point·aspect ratio are fixed (to prevent planar-target degeneracy); the self-check `--feed main`
  recovers the C3's known fx to **+0.62%** (same run, 159 frames, RMS 1.92 px). **Run on 2026-09-08**:
  the operator recorded REC NAV while viewing the mat with both cameras
  (`20260908/0908_105822/nav_111010`, RGB 3,463 frames) → the tool gave fx **713.8** (estimate 762.7,
  −6.4%), 5 distortion terms, fit RMS 2.00 px; the same alignment measured the RGB lens position
  `t_flu [0.2663, −0.0051, −0.0189]` (8.6 cm further forward·1.9 cm lower than the tape estimate) and
  the tilt **+15.9°** → all written into `second_cam`. Reproduction verification: the old model accepted 43.5%
  of RGB at the 3 px gate, the new model **3463/3463** (n_tags p50 5); with the new extrinsic added as
  well, the difference between the two solvers is **offset 2 mm / spread 1.7 mm**. The gate is
  `fallback.max_reproj_px: 4.0` (the RGB pixel pitch is 2× the C3's, so the same map error is 2× the
  px [유도]; 99.97% pass even at 3.0).
* **The camera tilt is tracked** (2026-09-07, `control/tilt_tracker.py`). The mount is open loop
  (no angle report), so the CAMERA TILT line in PAYLOAD prints **the estimated angle·source·±**:
  vehicle report (MOUNT_STATUS/servo) > tag measurement (`measured:tags`) > operator input (SET) >
  button-hold integration (`dead-reckoned`, rate `tilt_rate_deg_s` **[예측] 30°/s** — measure one sweep
  and replace it; the ± grows with the amount integrated). LEVEL is the 0° anchor. Type the actual
  angle into the spin box and press **SET**, and it counts from there. If it differs by more than 3°
  from the angle the RGB localization assumes, the line warns in yellow. When a command moves the
  mount (`tilt_epoch`), the RGB extrinsic moves to the new angle and the C3-RGB alignment is learned
  again.
* **1 outlier tag does not kill the whole frame** (2026-09-07,
  `tagnav.py: _drop_outliers`). The joint reprojection RMS mixes all corners with equal weight, so
  **one** tag with broken corners pushes a frame whose remaining tags agree to 1 px outside the gate.
  Now, when it exceeds `max_reproj_px`, it **drops the worst tag and refits**. Three safeguards:
  `outlier_max_frac` (up to 30% of the anchors), the `min_tags` lower bound, and **drop only if it
  stands out** (`outlier_min_ratio` 1.8 = 1.8× the median of the rest) — if the whole frame disagrees
  with the map, nothing stands out, so the rejection stands. The rejection message now also names
  **the culprit tag and its px**. A quad clipped off the image edge (`min_border_px`) has extrapolated
  corners, so it never becomes an anchor in the first place. A/B on hardware records, 6 runs, 6,278
  frames: acceptance **88.9% → 99.5%**, pose change on frames accepted by both p50 0.00 mm
  (p99 27.6 mm), and the continuity of newly accepted fixes, relative to the existing fixes within
  0.5 s, is p50 3–13 mm.
* **The mirror pose of planar tags is rejected** (2026-09-14, `tagnav.py:
  _all_in_front`). Tags on a single plane, like the floor, have **exactly two** PnP solutions: the
  pose with the camera coordinates of every point sign-flipped (R·Rz(180°), −t = the camera reflected
  in the tag plane and yaw turned by 180°) also fits the pixels just as well — it is just that all the
  tags are **behind** the camera, and the reprojection error cannot see that. Hardware record
  `data/20260914/0914_163103/nav_163112`: 42 frames with 2 tags (t 88.5–92.1 s) passed at 2.2 px
  while 0.44 m **below** the floor·0.7 m to the side·180° reversed, the warm start latched onto that
  solution and kept carrying it, and the NMPC chased it at ±30 N until the engagement was released by a solver
  failure. Now every starting point of the joint fit (warm/cold/IPPE seeds) can become best or pass
  only when **all the tags are in front of the camera** (z_cam > 0), and single-tag IPPE candidates go
  through the same check. No knob — an imaged tag is in front of the lens. Replay of the same record
  (original corners + recorded K + C3 14-coefficient distortion): mirror 42 → 0, 42/42 recovered to
  the correct pose (z −0.165, rms p50 1.85 / max 2.29 px), the remaining 366 accepted frames
  bit-for-bit identical, newly rejected frames 0. Replay of the RGB fallback feed (configured tilt
  15.9°) likewise: mirror 153 → 0, accepted 2955 unchanged.
* **The plot is always in the MAP (tag world) frame** (2026-08-14): the mat is always aligned with
  the axes, and the pool rectangle is in the same place every run. The controller runs in the
  ENGAGE-datum frame (START pose is the origin, start heading is +x), so `view.set_datum()` maps the
  positions coming in via MpcStatus back into the map. **Before that the transform was inverted, so the
  whole mat snapped around the moment START was pressed.**
* **The ROV on the plot is drawn at its actual hull size** (`hw_nav.yaml: rov_footprint_m`,
  default 0.4318 × 0.5334 m = **17 × 21 inch**, 17 inch along the heading
  [측정: operator 2026-08-14]). It is a rectangle, not a dot, so you can see whether it grazes a wall
  or a tag. The small dot in the centre is for when you are zoomed out.
* **The operating depth band is measured at −1.30 to −0.54 m** (`data/*/*/nav_*/fixes.csv`, z_ned of
  accepted fixes). It is tag-frame z, so the floor tags are at z=0 and +z is down — a vehicle swimming
  above the mat is **negative**. Previously a test pinned that the geofence z contained this band, but
  the fence is gone, so this number is now **a record that enforces nothing**.
  `test_measured_operating_depth_band_is_still_on_record` guards only the sign (positive would mean the
  vehicle is below the floor = the map or the convention is broken).
* **Refusal reasons appear on the panel**: previously they went only to a log line and the chip kept
  showing `engaged (DP hold)`, so the pilot did not even know START had been swallowed. Now it appears
  as `START refused: …`.
* **The geofence is entirely gone, both the drawing and the rules** (2026-08-14, pilot request). The
  dashed orange `GEOFENCE` box that appeared on engage, the path check before START, and the automatic
  release on leaving it while running — all three were removed. The `geofence_ned` /
  `geofence_frame` keys in `hw_nav.yaml` are **not read** (if they remain, they are ignored rather
  than being an error — so that old config files still load as is). The `geofence_ok` column of the
  MPC CSV is also gone, so **CSVs from before 2026-08-14 have that column**: read it by name, not by
  position. The only rectangle left on the plot is `POOL`, and it is a **drawing** of the walls, not a
  limit.
* **Only minimal text on the plot** (2026-08-14): all that remains is `off … lag … lap N` and
  dobmpc's `w_hat`. Position·localization status·solver time·tag count·refusal reasons were moved,
  respectively, to the TagNav line in SENSORS, the chip above the plot, and the **MISSION LOG** under
  PAYLOAD.
* **The error is shown split into `off` (cross-track) and `lag` (along-track)** (2026-08-14). Stage 0
  of path following is `path_lead_m` ahead of the projection point on the current segment, so looking
  only at the combined `err`, it reads like a failure even when the vehicle is exactly on the path —
  in fact, the `err 12.5 cm` of the 20:18 MPC line run was **lag 11 cm + off 1 cm**
  (`data/20260814/2018/mpc_201843.csv`).
  Sign convention: `lag` + means **behind**, `off` + means **left of the direction of travel**. The
  CSV also got the two columns `e_along` / `e_cross` (runs before that do not have them).
* **circle has no infeasible segments** (2026-08-17). It has neither 90° vertices
  (memory: square-corner-error-floor) nor 180° reversals, so the curvature is constant. Instead, the
  curvature limit `v <= sqrt(path_lat_accel_m_s2 * R)` applies **over the whole lap** — a limit that,
  on polygons, only bit at the corners. With the deployed value `path_lat_accel_m_s2: 0.05`, the cap
  is 0.158 m/s for r 0.5 m and 0.122 m/s for r 0.3 m, so even a larger value entered in the speed field
  will not be flown as is — **[유도]** `v = sqrt(a_lat·R)`, pinned by
  `rov_gui/tests/test_mpcc.py ::
  test_circle_speed_is_curvature_limited_over_the_whole_lap`.
  It is not a value measured in water (the circle has not been flown even once yet). In that case the
  worker logs a warning (`curvature caps this path at ... m/s`) — so as not to create a state where
  the geometry silently ignores the number the pilot entered.
* **square is placed in the MAP frame** (2026-08-14). With `rot_deg 0`, the sides are parallel to the
  tag map's x·y axes, the entered tag is **the min-x / min-y corner** (bottom left of the plot), and
  the first side goes out from there toward +x. Previously it was placed in the datum frame, so the
  whole square was tilted by the angle the vehicle was facing at START — the same kind of bug as
  line's `dir_deg`. The tag ends up at the bottom left thanks to `mirror_y` (it undoes what the
  FLU→NED mirror was doing: flipping y and sending the tag to the corner with the **largest** y).
* **The tag number·distance can be clicked and typed** (2026-08-14). Only the three fields of this
  panel are `ClickFocus` and everything else is `NoFocus` (because the window is the only key
  handler), and when a non-digit key arrives, **the field immediately releases focus and resends that
  key to the window** — even if you type a tag number and press W right away, the vehicle moves.
  Without this, the field would swallow the W, leading to a state where "piloting does not work for no
  reason". Enter = commit + release focus.
* **MISSION LOG** (under CAMERA TILT in the PAYLOAD panel): keeps mission events and **every log
  line** together with the **wall-clock time** (`18:29:06  going to tag 79 (1.24 m)`). It uses the
  wall clock rather than monotonic so that it can be matched later against video·notes.
  * **It scrolls** (2026-08-14, pilot request "so I can see the older ones too"). Previously the
    QLabel held only the last 12 lines and discarded the rest, so a refusal reason from 2 minutes
    earlier — the very line you most want to see when a run goes wrong — simply vanished. Now it holds
    the whole session (up to 5000 lines) and follows new lines **only when it is at the bottom**: if
    you are scrolled back up reading, the 20 Hz status lines do not drag the view down.
  * It is the **only exception** to the rule "never hide controls behind a scroll". The log is not a
    control, has nothing to press, and the latest lines, which are what matters now, are always
    visible.
  * **The level carries weight** (2026-08-23). Previously everything was the same dim grey, so
    `ENGAGED (mpc)`, `pose: tracking` and a CUDA warning looked typographically identical, and to find
    out whether a run had started you had to read the whole panel.

    | Level | Display |
    |---|---|
    | `mission` | **bold + accent colour + a horizontal rule above** — the lifecycle of the run itself (ENGAGED / STATION / FOLLOW / DISENGAGED / refusal) |
    | `error` / `warn` | red / amber |
    | `info` | the original dim grey |
    | `debug` | **not shown on the panel** — stdout only |

  * **Identical consecutive lines are counted into one line**: `16:34:21 (x4)  [WARN] mpc: engage
    refused: ...`. The count goes right after the timestamp because this panel does not wrap
    lines — appended at the end, it would run off-screen on a long line and not be seen. They are
    merged **only when consecutive**, so if something else happened in between, it stays a new event.
  * **One line per event.** Previously engage/disengage was printed **twice**, as one mission-event
    line + one `[WARN]` log line. Now the mission line carries the hold point as well, and the log
    copy is demoted to `debug` and remains only on stdout. For the same reason BundleSDF's 3
    mesh-check lines are **merged into one line** (`pose: mesh check — ... | ...`), and on a
    tracker-stage line, when the note repeats the stage word, that part is cut off (`pose: lost the object —
    lost — reacquiring` → `pose: lost the object — reacquiring`).
  * **It is also saved to a file**: when a recording (REC UI / REC NAV) stops or the window closes,
    the whole session is written to `mission_log.txt` in that run folder. Even if they stop at
    different times, it always rewrites **everything up to that point**, so it does not get split
    into fragments.
  * Mission events also go into `events.log`, so the screen and the file cannot tell different
    stories.
  * **2026-08-14: it grew to nearly twice the size** (pilot request "reduce the space a bit and make
    the log bigger"). The space was made by shrinking the three blocks GRIPPER / LIGHTS / CAMERA TILT
    from 5 rows to 3 rows each — the status notes ("no position feedback", "draw 0.42 A") give up
    their own lines and move up to the right of the caption line, where they are elided; the lights
    slider shares one row with OFF/¼/½/MAX; and the button height is 26→22 px. **Nothing was removed.**
    The window's minimum height is unchanged (994 px, --mpc budget 1000), so the layout tests are
    unchanged too. At 1920×1080 the log height goes 109 → **192 px**. To make it bigger still, space
    has to be taken from inside this panel again.
* **Mouse on the plot**: drag to move (pan), wheel to zoom in/out, double-click to return to the
  original view. This is not baked onto the screen as text but lives in the widget tooltip
  (2026-08-14 request). **The geofence (dashed orange line) is drawn only while engaged** — the
  fence is datum-relative ("±1.2 m from the spot where START was pressed"), and before engage there
  is no datum, so drawing it would place a box somewhere it will never actually be.
* **The pool border is derived from the map**: `hw_nav.yaml: pool_margin_m` (default 0.10) =
  **this much on every side beyond the printed edge of the outermost tags**. The map holds centre
  coordinates, so half a tag (0.085 m) is added first. Rebuild the map and the walls move with it
  (no hand-written rectangle is left behind to go stale). If an explicit setting is needed,
  `pool_ned: {x: [..], y: [..]}` takes precedence. With the current map: **2.025 × 4.437 m**.
* **The screen says which copy is used**: a duplicate id has two squares on the floor, and
  **only the copy** that actually contributed to the fix is painted green (`NavFix.tag_insts`).
  Measured verification: cases where both copies were accepted in the same frame at once: **0**.
* **Expanding the map**: `python -m rov_gui.tools.build_tag_map
  <run_folder>/nav_<hhmmss> --anchor config/tag_map.yaml -o
  config/tag_map_full.yaml`. The 47 surveyed tags are kept as **frozen anchors**; in each frame the
  camera pose is fixed from the known tags, and then the unknown tags in the same frame are solved.
  New tags are promoted only between rounds (`--min-obs` observations agreeing within
  `--max-spread`), so **errors do not accumulate along the path** — the very structure by which
  graph SLAM gets warped is simply not there. Duplicate ids are left out of the anchors, rediscovered
  and recorded as **two instances**. One capture rule: **when an unknown tag comes into view, a known tag must be visible along with it.**
  Tags that could not be promoted appear in the report together with their observation counts.
* **The robot marker on the plot moves at 20 Hz**: the state assembler bridges between camera
  fixes with the velocity estimate + gyro (`hw_mpc.yaml: vel_propagation`, on by default — double
  integration of acceleration is deliberately not used because of the BNO086 scale/bias), and the
  plot draws the MpcStatus stream, not the fixes. The fix Hz is shown separately, so there is no
  way to confuse "the state the controller sees" with "the localization source's rate".
* **PID mode**: `PID` in the **LOW** combo of the trajectory panel (labels since 2026-09-11: `None | PID | MPC |
  MPC_Tuned | DOBMPC | DOBMPC_Tuned | MPCC | DOBMPCC`, later `RL`, `RL_PWM`, 2026-10-03 `PID_7D`/`MPC_Tuned_7D`/`RL_7D` —
  see the "7d LOW followers" section; the keys used by the worker·CSV·meta,
  `pid`/`dobmpc`/…, are unchanged. Before that it was a "mode combo" that showed the lowercase keys as is).
  `None` is not a controller but teleop — see the "LOW level None" section below. The PID gains are the sim's
  pole-placed `GAINS_HEAVY_GRIPPER` (controller.py) × `hw_mpc.yaml: pid.omega_derate`
  (default 0.6, kp·d²/kd·d/ki·d³) [예측]. Even if the acados build fails, PID is always
  available. It is recorded with the same CSV schema, so a PID vs MPC vs DOB-MPC A/B works right
  away, as in the sim (the mode is in meta.json `controller.type`).
* **RL mode (2026-09-21)**: LOW combo `RL` (key `rl`, `--mpc-mode rl`, `hw_mpc.yaml: mode: rl`).
  `HwRl` in `rov_gui/control/rl_policy.py` — it inherits `HwPid`, reuses its reference-trajectory API (square·line·circle·path
  plan·follow) as is and swaps only `step` for the trained policy, so it flies **the same placement geometry with the same
  missions**. The policy is numpy weights exported from `/home/bdml/Desktop/RL_controller` (Isaac Lab PPO, a reproduction of
  "Learning to Swim"; observation = tag pose·IMU·reference point·reference-velocity feedforward·previous command, output =
  MANUAL_CONTROL 4 axes) and lives in `rov_gui/control/rl_policies/<name>/` together with `obs_spec.json`
  (`hw_mpc.yaml: rl.policy_dir`). To leave the worker plumbing unchanged, it returns the policy's axis outputs as a
  **pseudo-wrench** multiplied by `axis_gain` (`wrench_to_axes` recovers the axes exactly, `axis_cap` applied), and
  `note_applied` inverse-transforms the axes actually sent and feeds them back as the "previous command" of the next
  observation (same as in training). Training had no slew, so the controller overrides `axis_slew_per_s` with 0. Assumptions:
  vehicle `JS_GAIN_DEFAULT` 0.5 (`obs_spec.pilot_gain_assumed`), MANUAL mode, 20 Hz. The coordinate transform is the
  NED/FRD ↔ z-up/FLU mirror `diag(1,−1,−1)` (the same convention as the reference sampler in pid.py). Verification: 4 tests in
  `test_control.py` (harness surface·pseudo-wrench round trip·observation frame·mode switch·axis cap), `demo_e2e.py rl square`.
  The first hardware test (2026-09-21, `data/20260921/0921_182926`) completed Square 1 m @0.10 m/s for 5 laps with
  `axis_cap 0.5` (xy error RMS 0.042 m, p95 0.057 m [측정]); the approach segment's maximum of 0.45 m/s was too fast, so the
  cap was lowered to `rl.axis_cap 0.3`.
  **2026-09-27**: a direct-thruster output policy (`action_mode: pwm`, 8 throttles, mixer bypassed) was added to
  RL_controller. The station's transport is only the 4 MANUAL_CONTROL axes, so `HwRl` **refuses such a policy at
  construction** and leaves the reason (it does not force 8 outputs onto 4 axes;
  `test_control.py::test_rl_refuses_a_pwm_policy_with_the_reason`). A pwm policy flies only via
  `RL_controller/deploy/run_policy.py` (transport `pwm_lua_override`: the Navigator Lua `rl_pwm_override.lua` forces
  RC_CHANNELS_OVERRIDE ch 9..16 onto the motor outputs with a 100 ms timeout; `SERVO1..8` stay Motor1..8 — RCIN passthrough
  bypasses ARM/failsafe and was **discarded**, RL_controller README §10.3; this is not the station's reserved `rc_override`
  (ch 1..6 pilot axes)); `meta()["action_mode"]` is kept in the run meta.
* **RL_PWM mode (2026-09-30, unverified on hardware)**: LOW combo `RL_PWM` (key `rl_pwm`, `--mpc-mode rl_pwm`,
  `hw_mpc.yaml: rl_pwm:`). From the station it flies an RL policy that sends pulses directly to the 8 thrusters
  (`action_mode: pwm`, default `rl_policies/pwm10_s1`) — it **bypasses** the ArduSub mixer. The observation and
  reference-trajectory API are the same as `RL` (`rl_policy.py` `HwRlPwm(HwRl)`, observation 116 = 29×4); only the output
  differs: the controller puts 8 pulses in `info["pwm_us"]`, and the worker sends them as `bus.cmd_pwm` (`state.PwmCommand`)
  while sending MANUAL_CONTROL at **neutral**. The sink (`backends/hardware.py`) puts the pulses on RC_CHANNELS_OVERRIDE
  channels 9..16 (1..8·17·18 are 65535 = left as is), and the vehicle's Lua script `rl_pwm_override.lua` (a copy is in the
  policy folder) forces them onto the motor outputs only under the conditions armed + MANUAL + fresh frame (100 ms)
  (transport `pwm_lua_override`; RCIN passthrough is not used because it bypasses ARM/failsafe). When the frames stop, the
  script lets go and AP_Motors returns to the neutral MANUAL_CONTROL — DISARM·E-STOP·failsafe still take effect as before.
  The motor 1..8 ↔ RC 9..16 order is **fixed** (the script reads it that way; it cannot be changed by config).
  **Gate** (`control/rl_pwm.py` `PwmGate`, judged on values read back from the vehicle; if even one is off, engagement is
  refused + the reason is shown): `SERVO1..8_FUNCTION` = 33..40, no RCIN9..16 passthrough on `SERVO1..16` (so the
  lights·camera·gripper do not follow the thruster pulses), `MOT_1..8_DIRECTION` = the export's `motor_direction`,
  `SCR_ENABLE` 1, `SCR_USER1` 1, `RC_OPTIONS` bit1 0, `RC_OVERRIDE_TIME` > 0, `RC9..16_OPTION` 0, `SYSID_MYGCS` = station
  sysid, script heartbeat `RLPWM` within 2 s, `RLPWM` = 1 within 1 s after the pulses start, MAVLink 2 link, flight mode
  MANUAL (STABILIZE not allowed). For parameters, **only values the vehicle reported within 30 s** are accepted — while RL_PWM is
  selected, the worker notifies the sink every 1 s (a neutral `PwmCommand`; nothing goes out on the wire) and the sink
  re-reads them every 10 s; a session that has not selected it sends not a single parameter request. If the gate breaks
  mid-stream, the sink **latches** (refuses until the worker's stop command arrives, with the reason in the snapshot) —
  meanwhile the worker sends neutral pulses and disengages after 1 s. The latch is cleared only by the worker's release
  (`source="stop"`) or its not-engaged idle signal (`"idle"`). "Time during which pulses went out without the script having
  taken over" is **accumulated** across interruptions, and exceeding 1 s means refusal.
  **Stop path**: disengage·pilot-input handover·E-STOP·ENABLE off·teardown·controller silence (0.2 s) all send 1500×8
  immediately for 3 frames + 1 frame on each of the next 3 ticks, then stop the override. A session that does not use
  this path (or has used it up) writes nothing to RC 9..16. During station bridge coast (tag dropout) only neutral pulses
  are sent — **in rl_pwm there is no depth hold during coast**.
  **Output cap** `rl_pwm.pwm_cap` — the value at the first water entry was 0.25 (1400..1600 µs); that run saturated and
  limit-cycled (`data/20261001/1001_172910`), so by operator decision on 2026-10-01 it is **1.0** (the trained value) — the
  controller and the sink each apply it. This value is **the only cap on the pwm transport**, so `rl_pwm_7d` reads the same
  key too (2026-10-03). The pulses after the cap are fed back as the "previous command" of the next observation.
  **2026-10-10**: the cap is not a per-thruster clip but **a single scale factor in thrust space**
  (`rl_pwm.pwm_cap_mode: scale`, `rov_gui/control/rl_pwm.py scale_throttles` — the allocation direction is kept, the
  largest |throttle| is exactly the cap; `clip` is for A/B). Pulses over the cap are not clipped anywhere but rejected
  (follower neutral + n_fail, sink rejection + latch). If `pwm_cap` is lower than the export's trained cap, creating the
  follower is refused (a recorded bypass only with `rl_pwm.allow_caps_below_trained: true`; `rl_pwm_7d` has its own key).
  If the per-thruster sign probe artifact (`rov_gui/tools/thruster_probe.py` → `data/*/*_probe/thruster_sign_probe.json`)
  is missing, or is a dry-run · older than 30 days · for a different sigma/MOT_n_DIRECTION · did not pass, engagement is
  refused (`rl_pwm.require_thruster_probe`). Appended to the CSV: the predicted pack current of the sent pulses, `i_pred_a`,
  and the null-space `null_v_us`/`null_h_us` (schema 19; only on pwm LOW rows). Evidence·numbers: the KNOWN_ISSUES
  'LOW RL_PWM' 2026-10-10 paragraph, `RL_controller/logs/eval/pwm_cocontraction_analysis_20261010/REPORT.md`.
  **2026-10-10 (b)**: a pwm-LOW-only runtime interlock `engage.per_thruster` (tilt 20°/0.25 s, rate 40 deg/s/0.15 s, pack
  50 A/1 s + 13.5 V, null 80 µs/0.5 s, pinned ≥ 3/1 s — all [예측]; the first proposal, 60 deg/s·120 µs/1 s, missed 2 of
  14 events in a replay on the same day and was replaced; `rov_gui/control/rl_pwm.py PwmInterlocks` — the worker and the
  offline replay tool `rov_gui/tools/rl_pwm_interlock_replay.py` share the same implementation) and a policy-side
  observation guard `rl_pwm.obs_guard` (velocity slew, fix-jump rejection, tag gate → HOLD pulses; only inside
  `HwRlPwm.step`) were added, and a `pwm_guard` column (schema 20) is appended to the CSV. `i_pred_a` switched to the
  physical-thrust convention (T200 × batt_v/20.5 V). Replay results·limits: the KNOWN_ISSUES 'LOW RL_PWM' 2026-10-10 (b)
  paragraph.
  **Tilt ceiling (2026-10-03)**: until this date rl_pwm had no tilt interlock (the Lua script has no attitude logic
  either) — now, if |roll| or |pitch| exceeds `engage.attitude_axes.abort_deg` (35°) for 0.25 s, it disengages (neutral
  pulses, passive restoring). CSV `ax_*` are 0, and `uX..uN` are a **nominal wrench estimate** of the throttles actually
  sent (T200 curve × voltage factor × B), so they are not to be compared directly with other controllers' command columns;
  `pwm1..8` are the vehicle's SERVO_OUTPUT_RAW.
  **Before putting it in the water**: (1) upload the Lua script to the Navigator's `scripts/`, set `SCR_ENABLE` 1 (reboot)
  and `SCR_USER1` 1; (2) engage on the bench with the propellers removed and check that the script takes over
  (`RLPWM: engaged`), that it returns to 1500 on E-STOP/release, and that `RLPWM: released` is not printed mid-engagement
  (frame period 50 ms vs script stale 100 ms); (3) check the per-thruster directions (RL_controller/README.md §10.3).
  The Lua script has never run on the vehicle [예측] — the `mavlink:init` argument order is also unconfirmed.
  Verification: 26 tests in `tests/test_rl_pwm.py` (byte positions in real pymavlink v2 frames·refusal per gate
  item·controller·sink latch/stop path·worker gate/handover) — the script and the thrusters are not covered.
* **RL_PWM_7D mode (2026-10-03, 0 water runs)**: LOW combo `RL_PWM_7D` (key `rl_pwm_7d`, `hw_mpc.yaml: rl_pwm_7d:`).
  The same transport as RL_PWM (8 pulses, Lua override, MANUAL only, gate read from the vehicle, cap `rl_pwm.pwm_cap` — this
  block has no `pwm_cap` key), and only the policy is a pwm export **trained with roll/pitch references** (RL_controller
  `att7pwm_s*`: preset `traj10att`, **still water — no waves or current**) (`rl_policy.py HwRlPwm7d`). The plan's
  roll/pitch go into `q_des` (clipped to the training envelope and rate-limited to the training reference rate
  `rate_max_rad_s`; engagement starts level); when a 7-dim policy is flown on this LOW they are tracked, and when it is
  flown on RL_PWM they are dropped-and-logged. roll/pitch go out **as pulses**, so the s/t gate (`engage.attitude_axes`, firmware·probe·sign) does
  not apply — instead the worker's pulse bits (`_attitude_pulses`, `rp_track_live`) turn on the tilt ceiling and div_rp.
  The s/t-only sat-ineffective (iii)·degrade (v) do not exist on this LOW (a pulse version of (iii) is not implemented,
  KNOWN_ISSUES). Refusals (fail-closed): no export·level-trained·axes6·layout·missing envelope/`rate_max_rad_s`
  keys·`rl_pwm.pwm_cap` < trained cap (a recorded bypass with `allow_caps_below_trained`). Records: CSV `rp_track` **2** =
  "tracking on ticks where the policy's pulses actually went out" (`ax_roll/ax_pitch` nan; neutral ticks during coast·gate
  grace are 0), meta `run.attitude_axes.rp_transport: pwm_pulses`·`ceiling_armed`·`rp_track_live`,
  `controller.type: rl_pwm_7d`; the station bridge coast sentence for pwm LOW was corrected to "all 8 at 1500 µs — no
  depth·attitude hold".
  Verification: `tests/test_rl_pwm_7d.py` (build refusal·cap provenance·q_des parity·ceiling·truth table·records). There
  is no path to run the pwm closed loop offline inside the station (the demo has no pulse transport) — tracking
  performance is seen only with RL_controller `scripts/eval_attitude.py`.
* **Pilot input always wins.** When a stick/key moves, that frame does not go to the vehicle but
  becomes an MPC release request (the window's `_pilot_gate` — both paths, pump and edge, pass
  through the same gate). E-STOP (from the header·MPC panel·Esc, anywhere)·DISARM·ENABLE off·tag
  dropout >0.5 s·3 consecutive solver failures all disengage automatically.
  **Position no longer does** — the geofence-excursion release was removed on 2026-08-14. Even if
  the MpcWorker dies, the sink deadman (500 ms) sends neutral, and the window's mpc watchdog (1.5 s)
  takes back control on its own — the same last line of defence as when the pilot dies.
* **When circle is run on the wall geometry, the tag can leave the field of view** [예측, unverified].
  It crabs, so the nose does not turn, but the **position** moves relative to the tag by range 0..2R
  and lateral ±R (with the deployed r 0.5 m: 1.0 m and ±0.5 m). In a geometry that holds on with a
  single wall tag, this is a harsher demand than a corner — if the tag is lost, it auto-disengages
  mid-lap after the `engage.tag_stale_hold_s` debounce. Fly it on the floor geometry (several tags)
  first.
* On the wall geometry, `heading_follow: false` (crab square) is **mandatory**. Turning the nose at a
  corner makes the wall tag disappear from view.
  On speed [측정 data/20260921/0921_182926]: the policy has no explicit speed limit. At rest, a goal offset of 0.1 m →
  command 0.5, 0.2 m or more → 1.0 (saturated), and the only things that set the actual speed are (1) the GUI trajectory
  speed (reference), (2) `axis_cap` (0.5), (3) `rl.leash_m` (the distance at which a far goal is presented). Approach
  segment max 0.45 m/s.
  Decision 2026-09-21: `rl.axis_cap: 0.3` (min(axis_cap, 0.3) in RL mode only; by the speed law, steady state ≈ 0.14 m/s
  [유도]); a 0.10 m/s reference needs an axis value ≈ 0.24, so under this cap keep the GUI trajectory speed at 0.10 m/s or
  below. More fundamentally: retrain with a speed cap in the training reward (RL_controller README).
  Since 2026-10-01, `vehicle_net_buoyancy_n` (B−W) is also added to the PID's z force as a constant feedforward (before
  `f_max`, meta `controller.heave_trim`). Before that, only the clamped heave integral (i_max 5 N) took up the buoyancy, so
  with the peg gripped pid also sat on the floor (`data/20261001/1001_155105/mpc_154825.csv` 0.43→0.19 m). At 0 it
  behaves exactly as before.

Records: on engage a CSV is always opened (`…/mpc_<hhmmss>.csv` — see "Run folder" below, or it
piggybacks on the ongoing recording's stem as `_mpc.csv`). The first 9 columns have the same schema as the sim's
`runs/traj_*.csv` (`t,px,py,pz,rx,ry,yaw_deg,pitch_deg,lap`, world FLU), so existing analysis tools read it as is, and
after them come `w_hat×6·u×6·솔버 status/solve_ms·태그 수·reproj px·축 명령` ("w_hat×6·u×6·solver status/solve_ms·tag count·reproj px·axis commands")
and more. The `.meta.json` sidecar records even the tag-map sha1, the extrinsic and the gain provenance.

**Physical properties and constants are recorded too (2026-08-14, pilot request).** The CSV says only
*what the vehicle did*; it does not say *what the controller believed the vehicle to be*. So a `plant`
block goes into `meta.json`, and **pressing REC also drops `controller.json` into that recording folder**
(`rov_gui/control/plant.py` + `MpcWorker.dump_run_meta`).

* `plant` — all of `M nu_dot + C(nu) nu + D(nu) nu + g(eta) = tau + w` (NED/FRD):
  mass·inertia·added mass and the **6×6 M matrix**, the coefficients that build C and, evaluated at that run's nu,
  `C_rb_at_nu`/`C_a_at_nu`, D's linear·2nd-order coefficients and `D_at_nu`, `g_at_eta`,
  `u_max`. The source is not a copy but the very
  `dobmpc/params.py`+`fossen.py` that the controller actually imports.
  * **C and D are state-dependent**, so the matrices are *samples*, and the state they were taken at is
    written alongside in `evaluated_at`. They do not pretend to be constants.
  * Right before writing, the recorded generated matrices are checked against `fossen`'s product, and the residual
    is kept as `check_max_abs_err` (currently 0.0). The record cannot silently drift away from the model.
* `controller` — for PID, kp/kd/ki **plus** `omega_derate`, i_max, e_gate,
  yaw_gate, f_max, mz_max, and slew too. For MPC/DOB-MPC, N·horizon·**Q/QN/R**·u_max·
  w_hat_clip and the EAOB tuning (tau_dist, all the sigmas). A single gate or slew changes the trajectory
  as much as kp does, so "the things I tune" all go in.
* `mission` — the shape/tag/distance the panel was requesting. `trajectory` (armed scenario) is
  null before START, so this keeps the mission from being entirely empty when REC was pressed first.
* It is written even without an engage. Even a hand-flown survey pass that never engaged MPC must keep
  "which controller was selected and what the vehicle model was".

Verification status (2026-08-12): `rov_gui/control/smoke.py` — acados in rovgui-pose (numpy 2): SQP-RTI
build 1.3 s, EAOB+solve p50 3.1 ms/p99 6.4 ms (not in `data/`, console output); 1 lap of square completed
in the demo closed loop (0 solver failures). **Still unverified on hardware and in water** — see
KNOWN_ISSUES.md. The pool procedure must keep the order P3 (record localization only) → P4 (DP + axis-gain
step calibration) → P5 (square, mpc) → P6 (dobmpc A/B). **circle after P5** — a crab circle demands sway
for the whole lap, so it shakes the not-yet-calibrated axis gains harder than square does [예측].

### The panel says which depth it is holding right now (2026-08-18)

station mode **had been holding z all along** — `_arm_path` sets all 3 axes with
`set_target_ned((x, y, depth), yaw)`, and when `hw_mpc.yaml: square.depth_ned` is null, `depth` is
**the z at the very moment START was pressed** (`workers.py:989`). The problem was that **that number
was nowhere on screen**: `err_xy` is **horizontal** by definition, so a hold that had sunk 20 cm and was
hovering there read as a perfect hold. So the depth was added to the MPC panel chip and the plot's
error line.

```
ENGAGED [pid] STATION HOLD · err 1 cm · 0.01 m/s · z -1.05 m (+3 cm)
ENGAGED [mpc_tuned] SQUARE · off -2 cm · lag +10 cm · 0.05/0.10 m/s · z -1.05 m (+3 cm)
Plot:  off  -2.3 cm   lag +10.5 cm   dz  +3.0 cm   lap 1
```

* `z` is in the **MAP frame** (the engage datum's z is added back). It compares directly with the
  measured operating band −1.30 to −0.54 m. Without a datum it is renamed `z_d` to **make clear that it
  is datum-relative** — it does not silently put out a wrong number pretending to be map.
* NED z is **positive down**, so a vehicle swimming above the floor tags is **negative**, and the
  `(+3 cm)` in parentheses means **3 cm deeper** than the setpoint. The chip's tooltip writes this
  convention down (a sign that is not written down is a coin toss).
* To **nail the depth down to a number**, put a value in `hw_mpc.yaml: square.depth_ned`. Left
  empty, it uses the z at each START, as now (pilot decision 2026-08-18).
* `rov_gui/tests/test_offline.py :: test_the_panel_says_what_depth_is_being_held`
  pins the two conventions (map frame · positive down).

### A cost function that suppresses corner cutting — `mpc_tuned` / `dobmpc_tuned` (2026-08-18)

**Problem**: the tracking NMPC penalizes position error with a **world-frame diagonal Q**
(Q[0]=Q[1]=300 in `params.MPC_Q`). That means isotropic, so 1 cm **off** the path and 1 cm **behind**
on the path cost the same. At a corner the optimizer sells the cheap one to buy the expensive one —
i.e. it **cuts the corner**. That is the optimal behaviour for that cost function, not a bug
(memory: square-corner-error-floor).

**Mode**: the two modes with the `_tuned` suffix take the same 2×2 block and, at every stage,
**rotate it into the path frame**.

```
cost = q_along * (t̂·e)²  +  q_cross * (n̂·e)²
     = eᵀ W e,      W = Rz(ψ_path) diag(q_along, q_cross) Rz(ψ_path)ᵀ
```

The rotation is an orthogonal transform, so with `along_scale == cross_scale` it is **identical to the
isotropic diagonal to machine precision** — that is the parity that makes the A/B trustworthy, and
`rov_gui/tests/test_path_cost.py` pins it both algebraically and in closed loop. acados accepts
`cost_set(k, "W", W_k)` at runtime on the already-generated `LINEAR_LS` OCP, so **`mpc` and `mpc_tuned`
are the same compiled solver** — no second codegen, no waiting for a build.

* The rotation angle is **the path tangent** (`NedPlan.psi_path`), not the vehicle heading.
  With `heading_follow: false` the vehicle crabs, so the two are unrelated, and the reference **velocity**
  cannot stand in for it either — because at exactly the corners this feature exists for, `speed_profile`
  cuts the speed down to a creep.
* **Without a path plan it automatically falls back to isotropic.** DP hold·approach·the 10 s settle
  have no path, and "hold this point, but only in one direction" is not what station promised.
* Switching the mode back also reverts the rotated W (`_restore_base_weights`). If it lingered,
  **the next baseline run would be silently contaminated** — solver status is 0 either way.
* The tuning is the `mpc_tuned:` block of `config/hw_mpc.yaml`, and **a typo raises at
  startup** (the same rule as `imu_dr`: if it were silently ignored, the run would fly on the baseline
  cost while meta says tuned). `cost_frame` and `path_cost` remain in the run meta, and those two
  fields are the **boundary** between tuned/baseline records — the `Q` row of a tuned run is the
  isotropic baseline the split was derived from, not the value the solver actually used, so **do not pool**.

**Offline sweep** (builds once and runs the whole grid):

```bash
python -m rov_gui.tools.sweep_path_cost --fillet 0.15 --speed 0.10
python -m rov_gui.tools.sweep_path_cost --along 0.25 1 --cross 1 4 16 --plot figures/x.png
```

The plant is **the controller's own prediction model**, so it has no deadband and no tether, and its
horizontal drag is 8–12× too low (the whole-loop fit in `hw_mpc.yaml`). It ranks geometry honestly but
cannot rank anything the thrusters decide — all outputs are **[예측]**.
(Since 2026-09-08 **the hardware config flies with that drag corrected** — see the `plant:` item below.
This sweep tool builds with the params defaults, so it still has the old drag.)

### MPC_Tuned profiles — save a tuning to a file and pick it like a DP ckpt (2026-10-04)

The values mpc_tuned actually flies with used to be scattered across several hw_mpc.yaml blocks and
params.py. Now they are gathered in **one profile file** (`config/mpc_tuned_profiles/NAME.yaml`) and,
when LOW = `MPC_Tuned` (or `MPC_Tuned_7D`), picked from `tune 이름 상태 …` ("tune name status …") on the LOW row. The default is
`hw_mpc.yaml mpc_tuned_profile: 20261004_default` — a snapshot of the values exactly as they were on 2026-10-04.

| Section | What the station does |
|---|---|
| `apply` | **Applied** (only in mpc_tuned·mpc_tuned_7d, no rebuild): `Q/QN/R`, `path_cost` (along/cross), `policy.q_scale/along_scale`, `vehicle_net_buoyancy_n`, `axis_cap`, `axis_slew_per_s`. If even one key is missing, the load is refused |
| `compare` | **Not applied**: the solver·vehicle at creation time (rov_model, N, dt, u_max, plant damping, axis_gain). If they differ from the current station, the status is `WARN` (list of differences in the tooltip), meta `controller.mpc_tune.warnings` |
| `record` | For reference (prediction-model constants, path shaping, the hw_mpc.yaml blocks at save time). Not read |

* **mpc / dobmpc / dobmpc_tuned / pid are unaffected** — these LOWs keep using the hw_mpc.yaml values.
  Switching LOW while holding a profile immediately returns the solver's baseline W·heave trim to the hw_mpc.yaml values.
* **The buoyancy trim is a profile value too**: in mpc_tuned it is the profile's `apply.vehicle_net_buoyancy_n` that
  flies, not hw_mpc.yaml `vehicle_net_buoyancy_n`. 20261004_default is **peg mounted, −12 N** — when flying without
  the peg, use a profile with that value changed, or `--mpc-tune none`. If the two values differ, the tune status shows
  **WARN** ("is the payload the same?") — to catch the case where the peg was taken off and only hw_mpc.yaml was edited.
* `axis_cap`·`axis_slew_per_s` can **only be tightened** relative to the station values (if looser: WARN, and it flies
  with the station value). slew 0 (no limit) is refused at load. A profile whose position·attitude weights (Q/QN[0:6])
  are 0 is also refused.
* If the file name and `name:` differ, the load is refused (this prevents a copied-and-edited file from showing up under
  the old name).
* Picking is **refused while engaged·while a REC CSV is open** (one run = one tuning). The refusal reason is shown in red
  for 6 s.
* If a named profile fails to load at launch, it does not fly on hw_mpc.yaml instead but **refuses mpc_tuned
  engagement** (status `ERROR`) — picking one in the panel clears it.
* Records: meta `controller.mpc_tune` (profile·path·**sha1**·active·apply·warnings),
  `controller.weights_source`, `trajectory.follow_knobs_source`. Editing a profile file changes the sha1, so save
  **a new tuning under a new name**.

```bash
./c3 gui --source hw --mpc --mpc-tune 20261004_default   # set at launch (none = the hw_mpc.yaml blocks)
python -m rov_gui.tools.mpc_tune save NAME --description "what this tuning is for"   # the values mpc_tuned flies with now (the held profile)
python -m rov_gui.tools.mpc_tune save NEW --from OLD     # copy a profile → edit apply: in NEW.yaml
python -m rov_gui.tools.mpc_tune save NEW --from-config  # hw_mpc.yaml blocks + params.py (the values flown with --mpc-tune none)
python -m rov_gui.tools.mpc_tune list
python -m rov_gui.tools.mpc_tune show NAME     # + differences from the current hw_mpc.yaml, WARN on this station
python -m rov_gui.tools.mpc_tune diff A B
```

Order for making a new tuning: (a) `save NEW --from 20261004_default` → edit `apply:` in `NEW.yaml` and
select it in the panel, or (b) edit the hw_mpc.yaml values, try flying with `--mpc-tune none`, then
`save NEW --from-config`. The default `save` saves the profile hw_mpc.yaml points to as is — because while a
profile is held, the hw_mpc.yaml blocks are values mpc_tuned **did not fly with**.
Tests: `python -m pytest -q rov_gui/tests/test_mpc_tune.py`.

### The thrust the vehicle actually produced — per-thruster PWM (schema 14, 2026-09-08)

The left half of the run CSV is **what the station requested** (`uX..uN`, `ax_*`), while
**what the vehicle actually did** had so far been a single cell, `pwm_dev_us` — the **mean** of |PWM−1500|
over the 8 thrusters. That mean cannot answer "did even one thruster cross the deadband?". Because a
horizontal axis spins only 4 of the 8 thrusters, a mean of 5 µs fits both "nothing spun" and "two thrusters
spun at 20 µs". In fact, a cross-check against the 0817_114759 raw data shows that
**in 15.1 % of records the mean was under 25 µs while at least one thruster exceeded 25 µs**
[측정: `.../20260817/0817_114701/mpc_114759_rov.jsonl` join].

So eleven columns were appended **at the end**.

```
pwm1..pwm8   SERVO_OUTPUT_RAW microseconds as reported by the vehicle (1500 = neutral).
             blank = that output is unconfigured or has no report yet — not 1500.
pwm_age_s    age of that report. bus.thrusters is 10 Hz but the CSV is 20 Hz, so the same value
             repeats over two rows, and if the feed dies the last value repeats forever
             (a trap fixed in the panel on 2026-08-07). If the age keeps growing, it is a repeat.
batt_v/a     pack voltage·current. T200 thrust is a function of voltage even at the same PWM, so a
             newton value computed without voltage is an estimate. The station keeps the **measured value (µs)** and leaves the curve to the reader.
```

`pwm_dev_us` stays as it is (there is code that reads it). Not a single existing column was moved, and runs
with schema < 14 have no per-thruster records at all.

**A richer record does already exist**: `SensorLog` writes SERVO_OUTPUT_RAW together with the battery, as is,
to `<stem>_rov.jsonl` (10 Hz). But it switches on **only while the depth feed is being recorded or with `--imu-dr`**,
so the 2026-09-07/08 runs have none of it. The columns above are written unconditionally, on the same row
as the control tick.

### The prediction model's drag set to the hardware value — `plant:` (2026-09-08)

After the ballast weights were attached, the vehicle held depth but got only **0.006 m in 54 s** horizontally
[측정: `data/20260908/0908_133049/mpc_133049.csv`].
The cause was not a lack of force but a **threshold**: the solver's plant reckoned that moving at 0.05 m/s
takes 0.25 N, so it requested only |uX| p50 2.16 N, which is inside the ESC deadband, so the thrust is 0.

This is the spot that the `axis_gain` comment in `hw_mpc.yaml` has been pointing to since 2026-08-17.
Inverting the measured whole-loop law `speed = 0.692 (|axis| − 0.096)` through the axis gain gives

```
D_surge = axis_gain / 0.692 = 60 / 0.692 = 86.7 N·s/m      (model 4.03 → 21.5×)
```

and the 0.096 knee is **deadband, not drag**, so it is kept separate. Hence the new config is

```yaml
plant:
  linear_damping: [86.7, 133.8, 5.18, 0.07, 0.07, 0.07]
```

sway gets the same factor (preserving the model's anisotropy); heave and the angular axes stay as they are,
since there is no measurement for them. It is applied by `mpc_bridge.apply_plant_overrides` changing
`params.DL` **before the acados build** (the damping lives inside the generated dynamics), and the damping is
part of `_guard_stale_solver`'s rebuild key, so a cached `.so` cannot silently keep the old values. The
records boundary is meta `controller.plant_overrides` (`{}` = the old plant).

Effect and limits [측정: hardware solver A/B, 2 processes, `diag/drag_sweep.py`].
The benefit exists **only while the reference is moving** — drag does not enter the LQ position stiffness
sqrt(q/r), so it does nothing for a stationary reference (4.7 cm error, at rest: 3.40 → 3.54 N).

| Error / reference velocity | Model 4.03 | Measured 86.7 |
|---|---|---|
| 0 cm, 0.047 m/s | 2.56 N (0.043) | 4.49 N (0.075) |
| 4.7 cm, 0.047 m/s | 5.97 N (0.099) | 8.04 N (0.134) |
| 4.7 cm, 0.047 m/s, **hold_frac 0.885** | 4.62 N (0.077) | **4.79 N (0.080)** |

The third row is the hardware case: in 0908_133049 the plan covered only **12 %** of the 3 s horizon and the
reference stood still for the rest, so this block's benefit was masked down to **+3.7 %**. The drag value
itself is now the vehicle's own, but the key to getting the vehicle moving is the **hold tail / plan age**.
The deadband (axis 0.096) is not drag, so this block cannot fill it. **Unverified on hardware**.

| Mission | baseline cut | 1/16 tuned | p95 cross | lap time |
|---|---|---|---|---|
| fillet 0.15, 0.10 m/s | 4.8 mm | **1.2 mm (−74 %)** | 4.4 → 1.0 mm | 41.1 → 41.0 s |
| fillet 0.0 (waypoint) | 3.6 mm | 3.1 mm (−15 %) | 1.6 → 1.0 mm | 45.7 → 45.7 s |

Two things read straight off this table [예측, offline]:
1. **The lever is `cross_scale` alone**. Lowering `along_scale` barely reduces the corner cut (at fixed
   cross, 0.1↔1.0 differ by <0.1 mm) and only adds lap time and lag.
   On the waypoint mission it is **actually worse** (+57 % at 0.25/1).
2. **It only takes effect with a fillet.** The error left at fillet 0 is not a cut chosen by the optimizer
   but a floor that arises because **the reference itself is infeasible**, so weights cannot move it.

**Unverified on hardware** — not flown in water even once yet. See KNOWN_ISSUES.md.

### The approach was slow because of the leash, not thrust (2026-08-18)

**Measured** (2026-08-18, 4 approaches): during the approach `|axis|` p50 **0.18**, ticks that touched the
0.50 cap **0.0 %**. It was not using even 1/3 of its authority. Speed by remaining distance:

| Remaining distance | Speed | \|axis\| |
|---|---|---|
| 0.5–1.0 m | 0.102 m/s | 0.18 |
| 0.2–0.5 m | 0.093 m/s | 0.17 |
| 0.05–0.2 m | **0.037 m/s** | **0.066** |
| 0–0.05 m | 0.015 m/s | 0.12 |

**What set the speed was the leash, not `approach_speed_m_s`.** Because `_tick_approach` was calling
`set_target_ned` **without a velocity feedforward** (the path follower has always passed one), the steady
state settles like this:

```
kp·lead = 51.7 × 0.35 = 18.1 N,   measured whole-loop F = 86.7·v + 5.76
  → v = 0.084 m/s      [유도]   ← measured 0.079~0.102 m/s
```

So **all three were fixed together**. Changing only one would repeat the mistake that wasted a run on
2026-08-17 ("a speed number that never entered the loop").

* `_tick_approach` passes the setpoint's own velocity as `v_ned`, but
  **tapered by the controller's lookahead distance** — the NMPC extrapolates the reference with `v_ref` to the end
  of the horizon, so full-speed FF within one horizon of the target means "go past it".
  PID (1 stage, 0.05 s) is unaffected by the taper, so the comparison between the two still holds.
* `approach_speed_m_s: 0.10 → 0.20`, `approach_lead_m: 0.35 → 0.50`.
  By the same formula the leash upper bound becomes 0.232 m/s, so 0.20 becomes feasible. The maximum command is
  kp×0.50 = 25.9 N = axis 0.43, still inside the cap.
* **station skips the settle** (`settle_station_s: 0.0`). This is counter-intuitive:
  during the settle the setpoint is **still tied to the leash**, and arming is what releases that leash —
  i.e. the settle does not help arrival, it **holds the vehicle in the slow state for another 10 s.** Moreover,
  station has nothing to start after the settle (armed target = the very point the approach was walking toward).
  path missions keep it as is (a square must not start 30 cm off).
  If `imu_dr` is on, a static window is needed, so **the original settle comes back automatically.**

Expected effect [예측 — derived from the measured relation above, needs re-confirmation in water]:

| Move | Before (measured move + settle) | After |
|---|---|---|
| 0.4 m | 13 s + 10 s | ~2 s + 0 |
| 1.3 m | 22 s + 10 s | ~7 s + 0 |

**The last 20 cm is still slow** — `|axis|` dropping below the 0.096 deadband threshold there is a gain
problem, and this change did not touch it (pilot decision). In station, skipping the settle releases the
leash early, so that segment gets a little faster too.

### Losing the tags does not switch it off — STATION BRIDGE (2026-08-18)

**Symptom**: when a single disturbance makes the camera lose the tags, the loop **disengages** about 1.1 s
later (`tag_stale_s` 0.7 + `tag_stale_hold_s` 0.4). That is worse than it sounds — **depth hold switches off too**,
and this vehicle is not neutral (the model sinks at −5.7 N; on 2026-09-07 the hardware floated up —
`vehicle_net_buoyancy_n`), so it starts drifting off according to the day's trim. With attitude and view
changing, the odds of reacquisition actually **drop**. And the pilot has to ARM again.

**Even turning on `imu_dr` does not fix it**, and that is the crux. What `MpcWorker._runtime_fault` consumes is
`meas` (the tag solution, deliberately kept as ground truth), not the `meas_ctrl` that the controller
actually consumes. So whatever the estimator does, the interlock fires all the same. What has to be
separated is not "tag vs IMU" but **"the estimate dropped out" and "give up on control"**.

**Ladder** (`rov_gui/control/station_bridge.py`, **station mode only**). How long each axis can be carried
along is not one question but four — because the sensors behind them differ:

| Axis | Source without tags | Drift |
|---|---|---|
| z | pressure sensor | **none** (absolute) |
| roll/pitch | ArduSub AHRS | **none** (absolute) |
| yaw | gyro integration | slow (0.02 rad/s = 70 °/min) |
| x, y | double integration of acceleration | t², fast |

```
fix fresh          as usual
0 ~ imu_hold_s     all axes on the bridge estimate, authority same as usual
after imu_hold_s   x/y/yaw released, depth+attitude continue — unlimited
```

The last stage is **deliberately not a disengage**. Pushing horizontally on a drifting estimate actively
drives the vehicle somewhere wrong, and there is no longer any means of stopping it on position grounds
(geofence removed 2026-08-14). Hands off, it merely drifts with the water, and at pool speeds that is
slow. Meanwhile the **two drift-free axes** hold the vehicle at depth and level, which is the state with the
highest odds of reacquiring the tags. Hence it is left unlimited. The axis actually sent out in coast is
**heave alone** (K/M have no axis in MANUAL_CONTROL and are dropped anyway) —
`test_the_coast_tier_sends_heave_and_nothing_else` pins this on the **on-the-wire value** after
allocation·cap·slew.

* **The x/y source depends on the calibration** (`xy_source: auto`). With a calibration file it
  integrates the accelerometer; without one it **pins x/y to the last fix**. The RAW C3 has a measured
  1.8 m/s² bias, so 3 s makes 8 m — auto is not a convenience feature but a safety device.
  The pinned path **works even with no C3 IMU at all** (it uses only the pressure sensor + autopilot).
* **All other interlocks stay live**: disarm, leaving the flight mode, telemetry loss, solver
  failure, tick overrun, E-STOP. And the two faults in which **the very sensor the bridge would ride on is dead**
  (`imu stale`, `pressure depth stale`), plus `no vehicle imu`, are not on the whitelist,
  so they still disengage. The whitelist matches on **word boundaries**, not prefixes
  — with `startswith` alone, new fault strings would be silently inherited.
* **The panel says it in red**: `ENGAGED [pid] NO TAG — COASTING`. coast is unlimited, so
  it is a human who judges "when to stop", and for that the screen has to speak in colour.
* **Every recovery yields a free measurement**: at the moment the tags come back, how far the estimate
  had drifted apart is written to the mission log (`fix back after 1.35s (tier imu, IMU
  was 4.2 cm off)`). It is the number that turns the IMU budget, which existed only as [예측], into [측정].
* **Records boundary**: two columns, `bridge_s` / `bridge_tier`, are appended to the CSV and the meta schema
  became **5**. Rows with `bridge_s > 0` were **not flown on tags** — px/py are the IMU
  estimate (or the pinned anchor), and in coast rows, whatever the controller asked for, surge/sway/yaw
  went out as 0. **Never pool** them with clean ticks.

Config is `station_bridge:` in `config/hw_mpc.yaml` (typos raise at startup).
**Unverified on hardware** — see KNOWN_ISSUES.md.

## Following an object — follow (2026-08-21)

Turn on `--pose --mpc` **together** and the clicked object is lifted into pool coordinates, and the mission
shape `follow` **keeps holding the relative pose** to that object.
`rov_gui/control/object_nav.py`.

### Why it did not exist

The station had two 6-DoF estimators running that knew nothing of each other. `--mpc`'s `TagNavWorker`
outputs the map coordinates of the **vehicle** (`bus.nav_fix`), and `--pose`'s `PoseWorker` outputs the
**camera-relative** pose of the **object** (`T_cam_obj` in `bus.pose`). There was not a single line of code
connecting the two, so the pose log stated it outright — *"NO camera->body transform
is applied; the vehicle's own pose is not in this file."* So nobody knew where in the pool the object
was, and doing anything relative to the object was impossible.

### The frame chain, and why the extrinsic cancels out

```
T_cam_obj (16 floats, row-major, object->camera, OpenCV optical, m)
  R_cam_obj = M[0:3,0:3],  t_cam_obj = (T[3], T[7], T[11])

R_map_cam = R_ned_body · R_frd_cam        <- no extrinsic
p_map_cam = p_ned + R_ned_body · t_frd_cam
R_map_obj = R_map_cam · R_cam_obj
p_map_obj = p_map_cam + R_map_cam · t_cam_obj
```

`TagNav._solution` builds the body pose by **dividing** the extrinsic out of the PnP camera pose, and this
side **multiplies it back** in. For the **same camera frame** the two are exact inverses, so the extrinsic
cancels out entirely.

Why this matters: `cam_tilt_deg: 43.3` is applied **only as a pure rotation**, and the translational
component that the real hinge moves along with it (`cam_t_flu`, lever arm 0.2855 m
[측정: `bluerov2_mujoco_marinegym/meshes/c3_payload_frames.json`, Onshape 2026-07-19]) is
**still not measured** (KNOWN_ISSUES 2026-08-17). Same-frame pairing neutralizes that unknown.

If the frames are mismatched the cancellation breaks, and the residual is `|t_frd_cam| · 2·sin(θ/2)` —
2.5 cm for a 5° yaw between frames. That is why **`pair_exact` has a permanent slot** (the panel's corner
readout, the `Object` row in SENSORS, `obj_pair_exact` in the CSV). It not being 1 is the **only** warning
that the cancellation has stopped.

**Composition uses only the raw `NavFix`. Never `meas["eta"]`.** What `StateAssembler` returns is a
hybrid — z from the pressure sensor, roll/pitch from the ArduSub AHRS, x/y propagated by velocity between
frames, and on top of that it datumizes. Composing with that breaks the cancellation in z·roll·pitch and
brings the lever arm back to life. And it happens **silently**.

### What is held — the object-frame offset

The relative pose is captured at **the very moment** START is pressed. The offset is stored in the **object's yaw frame**.

```
Rz_T  = Rz(yaw_obj)^T
d_obj = Rz_T · (p_veh_map - p_obj_map)
dyaw  = wrap_pi(yaw_veh_map - yaw_obj)
```

Result: **object translates → vehicle translates / object yaws → vehicle orbits (keeps looking at the same face) / object rolls·pitches → no response.**
The last one is deliberate — a tilting object must not make the vehicle translate, and MANUAL_CONTROL has
no K/M axes at all, so roll/pitch could not be held anyway (a fact `station_bridge.release_horizontal`
already records).

**Arming never moves the vehicle.** The offset is captured from the vehicle's **current** pose, so the
target at arm time is the vehicle's own position. No approach and no settle — there is nowhere to go. That
is a property, not an omission, and a test pins it
(`test_arming_follow_does_not_move_the_vehicle`).

The object's heading axis is set by `object_nav.yaw_axis`. `auto`
**picks the most horizontal axis once, at the first lock, and keeps it fixed for the anchor's whole lifetime**
(if it changed midway, the heading reference would jump in a 90° step). On an on-site reconstruction mesh,
which axis is "x" is arbitrary, so `geometry.yaw_from_R` (ZYX psi) is not used — it diverges as the
object's +x approaches vertical.
**With `yaw_axis: none` the object's yaw is not used at all and the offset is held in the map frame. A whole failure class disappears, so it is recommended as the mode for the first bench and pool runs.**

### Four layers of safeguards

1. **Excursion clamp** — if the target point leaves the `max_excursion_m` (default 1.5 m) sphere around the
   arm point, it is clamped back inside and the panel shows `EXCURSION LIMIT`.
   **This is not the geofence revived**: it neither refuses nor stops, it only **clamps the reference**.
   It is the form allowed after the fence was removed on 2026-08-14.
2. **Rate-limited walk** — the setpoint only walks at the rate in the speed field (`speed`). Even if the
   object estimate jumps, the target follows at a walk.
3. **Hard leash — applied to the result, not the step.** If `|sp - p_veh| > approach_lead_m`, it is scaled
   down by that ratio. If only the step were limited, one long tick would overshoot the leash (a lesson
   `_tick_approach` already learned).
4. **Feedforward** — the target point's own velocity (including the orbit term) + a catch-up tapered by
   the controller lookahead distance. **Without this layer the leash sets the speed**: measured
   `kp·lead = 51.7 × 0.35 = 18.1 N` vs `F = 86.7·v + 5.76` → `v ≈ 0.084 m/s`
   (2026-08-18, memory: approach-speed-leash-limited). follow without FF cannot keep up with an object
   faster than that.

### Freshness ladder — no silent green

| Condition | Behaviour | Panel |
|---|---|---|
| `age ≤ max_extrap_s` **and** `PoseTrack.state == "tracking"` | extrapolation + full FF | `FOLLOWING` |
| otherwise | **FREEZE**: re-publish the last setpoint, FF 0 | `OBJECT STALE 1.2s` (amber) |
| freeze exceeds `hold_s` | **demoted to station hold** | `OBJECT LOST — HOLDING` (red) |

The `PoseTrack.state` check is the key: `PoseWorker` publishes at 10 Hz whatever the pipeline is doing, so
even a 30-second `registering` has a perfectly fresh stamp. **Looking at age alone will never catch it.**

Why it is **a demotion, not a disengage** has already been argued at length in `station_bridge.py` — a
disengage also switches off depth hold on a vehicle that is not neutral (model −5.7 N, varies day to day on
hardware), and a vehicle that drifts off changes its view, so the odds of reacquisition **drop**.

**On tag loss: demote first, then bridge.** Three reasons: (1) the bridge's premise is "at rest over one
point", whereas follow is structurally a moving mission; (2) **when the tags are lost, the object cannot be seen either**
— `T_map_obj` is composed from `NavFix`, so without a fix there is nothing to follow in the first place, and
carrying the vehicle on the IMU while leaving the target unobservable means subtracting two drifting
quantities; (3) the argument that not switching off is still right is the same. In
`station_bridge.py`, only the wording changed.

`STOP TRAJ` can be pressed during follow too (follow, like station, does not turn on `traj_on`).
`set_traj(False)` already retargets to `self._eta`, so "stop following, hold right here" comes for free.
The engage gate and the runtime interlocks are **all unchanged**.

### Refusals before arm — each reason has its own sentence

* launched without `--pose`
* `nav_source` in `hw_nav.yaml` is not `main` → the object and the fix ride on **different cameras**, so the
  extrinsic cancellation breaks (the ROV RGB extrinsics are all `[예측]`)
* the controller is `mpcc`/`dobmpcc` → `HwMpcc.set_target_ned`, with `del v_ned, r_ned`,
  **throws away the velocity feedforward** and rebuilds the path on every call. `follow_ok = False`,
  fail-closed via `getattr(..., False)`
* no object lock (including `registering`, out of range, unpaired)
* the heading axis is vertical, so yaw is undefined (only when `yaw_axis != none`)

A **loose pair**, one that is not `pair_exact`, is a **warning**, not a refusal — it is a degradation, not an
error, but it still has to say that at that moment the lever arm has entered the error budget.

### Display

* The object is a **diamond** (tags are squares, the hull a rectangle, the reference a cross — deliberately
  a fourth shape). Filled when live, hollow amber when stale, hollow faint when lost.
  **No estimate, no confident marker** — the same rule `p_dr`·`p_ref` already follow.
* Heading tick 0.15 m; a `?` glyph if there is no `yaw_map`.
* The object's own trajectory is a solid 1 px line (not viridis — one age ramp per plot is enough).
* Hull → object dashed line + cm label. The number the pilot actually judges by (is it inside the
  0.3~0.8 m band?).
* Corner readout (extended 2026-08-23) — **the only place the object position appears**, and the frame is
  MAP (the same frame as `px/py/pz` above, so you can subtract by eye):

  ```
  obj  x +0.31  y -0.09  z -1.02 m
       live  d 0.87 m  yaw +147  age 0.08 s  pair 0 ms
  ```

  While following, one more line: `follow following  err 4.1 cm  hold 0.55 m / +12`.
* **`depth-vs-MAP  1.02x  (+/-0.04)`** — above the trajectory readout, right below `tag-vs-ATTITUDE`.
  It is the median + p10~p90 spread of comparing the **expected Z**, obtained by intersecting a pixel grid
  with the tag plane (z=0), against **the Z the C3 depth map reports** (`window._check_depth_scale`, 2026-08-23).
  **If the spread exceeds 0.15, do not cite the median as a constant** — it means the depth is mis-shaped
  rather than mis-scaled, or that something other than the mat is filling the view.

  > The first version read **a 5x5 at the tag center**, but the tag center is a black square — the one
  > patch in this scene that stereo cannot match. The smaller the tags got (the higher the vehicle), the
  > more holes and edge bleeding, so the number moved **with the tag count** (3 tags 0.94x / 8 tags 1.17x /
  > 14 tags 1.46x, 2026-08-23). The mat is dense texture, fills the view, and the tag map knows it is at z=0
  > — so now it samples hundreds of points across the whole mat.

  **Why this exists**: the depth path is **the only quantity on this station with no witness to verify it**.
  Tag localization is monocular PnP + a known 0.17 m tag and uses **no depth at all**, so the ratio of the
  two values is the absolute scale of the depth.

  | Reading | Meaning |
  |---|---|
  | `1.00x` | depth is metric. If the object pose is wrong, the **mesh** is to blame |
  | `~1.3x` | **The depth path reads long.** The reconstructed mesh and the tracking distance both inflate with it; fix this before touching anything else |

  On 2026-08-23 the object showed up **0.45 m below** the mat and the rays were 1.45× long, but "the mesh is
  oversized" and "the depth reads long" **produce the identical symptom**, so they could not be told apart.
  This one line is that fork in the road. No worker, no extra copy, no new signal — the corners, the depth
  map and the tag map are all already on the GUI thread. With fewer than 3 tags it writes nothing (one tag
  is an anecdote).

* **The rejection reason is always visible.** If the state is not `live`, the anchor's `note` is appended
  verbatim as one line, and **the line appears even when `cold` (never locked)**:

  ```
  obj  cold  pair --
       object at 1.53 m, outside 0.15-1.20 m
  ```

  Before 2026-08-23, nothing was drawn when `cold`, and once a position had been obtained even once the
  reason was hidden — i.e. **the plot was silent in the two cases that most needed an explanation.**
  If the object is clearly visible on screen but no diamond appears, this line says why (usually the
  distance gate in `object_nav`).
* An `Object` row in SENSORS (state · range · pair if the pair is loose).
* The follow target is drawn **for free** — the blue cross that `ctrl.ref_ned_at()` fills in simply moves
  along with it. No new field, no new drawing.

### Config and records

`object_nav:` in `config/hw_mpc.yaml` (typos raise at startup).
**Every number in `object_nav:` is `[예측]` until pool stage 1.**
0.3~0.8 m working / 2.4 m registration failure are [측정: KNOWN_ISSUES 2026-08-09, saved data],
~1.33× in air is [측정: `calib/FOV_AUDIT.md`], lever arm 0.2855 m is
[측정: `c3_payload_frames.json`, Onshape 2026-07-19].

**Records boundary**: 10 columns (`obj_px,obj_py,obj_pz,obj_yaw_deg,obj_age_s,
obj_pair_dt_ms,obj_pair_exact,obj_state,follow_state,follow_err_m`) are appended to the CSV and the meta
schema became **6**. `obj_p*` is in the same **datum-frame world FLU** as `px/py/pz`·`dr_p*`, so the
relative vector comes straight out as `obj_px - px` (the bus's `ObjectFix` is in the MAP frame, and the
conversion happens in one place only, `_obj_row`). **Only rows with `obj_pair_exact == 1` are extrinsic-free**
— never pool object-position statistics across this column. The `object_nav` block is **always** recorded,
even without `--pose` (`{"enabled": false}`) — without the key, "this build had no follow" and "follow was
switched off" cannot be told apart.

### Checking offline

```bash
./c3 gui --source demo --mpc --pose --demo-object drift   # still | drift | orbit
python rov_gui/tests/demo_e2e.py pid follow orbit
QT_QPA_PLATFORM=offscreen python rov_gui/tests/test_object_nav.py
```

The demo's object is **placed in the MAP frame and back-projected through the real `R_t_frd_cam("main")`**,
and it **shares the capture stamp** with the synthetic `NavFix`, so `pair_exact` is actually checked offline.
**What the demo cannot prove**: everything about real object estimation. SAM2 mask quality, FoundationPose
latency, dropout statistics, depth noise — none of them show up.

**Unverified on hardware**, and it inherits wholesale the risk that `--pose` itself has never once run on
hardware (KNOWN_ISSUES 2026-08-09). The bench → pool sequence is in KNOWN_ISSUES.md.

### If the gantry carries the target (2026-08-23)

For an experiment in which the target object hangs from the FMC4030 gantry and is moved by it, the gantry is
not put into this GUI but launched as a **separate process**:

```bash
./c3 gantry          # left window — gantry panel (robust env)
./c3 gui --pose --mpc   # right window — station (rovgui-pose env)
```

There are three reasons not to embed it, each sufficient on its own: the vendor `.so` enforces 1 connection
per process, the two envs are mutually exclusive (torch vs pyqtgraph/gtsam), and only as a separate process
does the gantry get its own Ctrl-C emergency stop.

The gantry panel's **Tag map position** card shows **where in anchor-25 tag-map coordinates** the gantry's
fisheye camera is right now — the same frame as `fixes.csv`. That it is the same frame is no coincidence: the
card uses `TagDetector`+`TagNav` from `rov_gui/control/tagnav.py` **imported as is**
(`src/gantry_map_pose.py`). So the gantry and the ROV can be viewed side by side in tag-map coordinates.

Two cautions: what the card shows is the **lens** position, not the position of the hanging target (lever
arm not measured), and the old path that lifts gantry **encoder** coordinates into the tag map
(`R_gantry_to_slam`/`gantry_to_slam_scale`) is broken in three places and must not be used. For details,
see "Operating alongside the ROV — tag-map frame position" in
[README_fisheye_gantry.md](../README_fisheye_gantry.en.md), and KNOWN_ISSUES.

## How far the IMU alone gets you — `--imu-dr` (2026-08-17)

**Question**: if the tags are cut off and the state is updated from the C3's BNO086 alone, how long does it
stay usable?
**Method**: anchor to the tag solution at the moment the mission settle (`engage.settle_s`, 10 s) ends, and
from then on integrate only the IMU. AprilTag localization keeps running and becomes
**ground truth as well as the second marker on screen**. The rate at which the two markers move apart is the
measurement.

No new mission was made — `station | line | square | circle` with `origin_tag`, unchanged. The only thing
that changes is who produces the state.

```
./c3 gui --source hw --allow-command --mpc --imu-dr shadow
./c3 gui --source hw --allow-command --mpc --imu-dr control
```

* **shadow** — the controller keeps flying on the tags. The estimate is only drawn and recorded.
  Even with it on, the commands are **bit-for-bit identical** (a test pins this).
* **control** — the controller **consumes the estimate**. Typing this on the command line is explicit in
  itself, so this one line is all it takes. `--imu-dr-control` is a separate gate, needed only when `hw_mpc.yaml`
  is already at `mode: control` — because a closed loop that nobody asked for must not come
  up just because the config file was left in that state.

**Plot**: solid green hull = tags (truth), dashed amber ghost = IMU, the solid line joining them is the
error, and top left `DR 42.0 cm  23 s [c3/ahrs]`. **Automatic abort is off** (operator decision) — there is
no geofence either, so the only thing that stops the vehicle on position grounds is the pilot's E-STOP, and
that number is its instrument. To turn it on, just put a number in `imu_dr.abort_err_m` in `hw_mpc.yaml`.

**Attitude defaults to AHRS**: gyro integration + accelerometer leveling of roll/pitch only; yaw is
gyro-only (the magnetometer is unusable in a steel pool). It is still 100% IMU, and it bounds the term in
which attitude error grows as t³ down to t². There are also `gyro` (pure integration) and `vehicle`
(roll/pitch only from the autopilot).

**Depth is not from the IMU** — it is the pressure sensor as is (operator decision). If z were integrated,
its error would turn straight into a heave command and the vehicle would ram the mat or float up to the
surface. The IMU-integrated z is also kept, as `dr_pz_imu`, so the after-the-fact comparison is free.

### Calibration first — without it the estimate itself does not hold

This camera's accelerometer has a **1.80 m/s² (0.184 g) bias** (almost entirely on IMU x), and the
IMU→camera extrinsic is **absent** from the device (`KNOWN_ISSUES.md`). Without correction,
`0.5·1.80·t²` = **90 m** in 10 seconds [유도] — not approximately wrong, but unusable.
(What the repo long recorded as "scale +20%" was a **misreading**: it was `|a|=11.8` measured in a single
upright pose, and in that pose gravity is nearly parallel to IMU x, so the bias adds on directly. Measured
in 6 poses, the scale is within 0.9% of 1. The distinction matters because
**a fixed bias is wiped out completely by the settle-window static correction, regardless of attitude**.)

```
# 1. accel scale/bias — out of the water, slowly roll the vehicle for at least 2 minutes (the whole sphere)
python -m rov_gui.tools.calib_c3_imu <run>/..._c3_imu.jsonl --fit accel \
    -o config/c3_imu_calib.json
# 2. IMU->vehicle rotation — 60 s of hand shaking, all 3 axes, with ArduSub left on
python -m rov_gui.tools.calib_c3_imu <run>/..._c3_imu.jsonl --fit rotation \
    --rov <run>/..._rov.jsonl -o config/c3_imu_calib.json
```

The point of step 2 is that it goes straight to **IMU→vehicle**: the IMU→camera extrinsic that the device
lacks is no longer needed, and the camera's 40° tilt is absorbed automatically. The tool **rejects**
tumbles that did not cover the sphere and single-axis wiggles — because confidently giving a wrong answer is
the real failure mode of this calibration. Without a calibration, the run meta's `calibration_sha1` stays
`null` and a loud warning is given at startup.

### Records and post-hoc analysis

At the CSV tail: `dr_px,dr_py,dr_pz,dr_pz_imu,dr_yaw_deg,dr_err_m,dr_err_z_m,dr_t_s,
dr_hz,dr_n,dr_ok,rp_residual_deg,roll_deg`. `dr_px` follows the **same convention** as `px`
(world FLU, datum frame), so simply subtracting gives the drift. The `imu_dr` block in meta is the records
boundary, and even when off it writes `{"enabled": false}` — without the key, "old build" and
"switched off" cannot be told apart. Throughout the run, `*_c3_imu.jsonl` (raw samples) is kept too.

```
python -m rov_gui.tools.plot_imu_dr <run folder>
python -m rov_gui.tools.plot_imu_dr <run> --from-jsonl --attitude gyro,ahrs --restart 20
```

4 figures (map overlay / error vs elapsed time / along·cross·vertical decomposition / health) and
`imu_dr_summary.json` (time to reach 5·10·25·50·100 cm, fits of `0.5·b·t²` and `(1/6)·g·β·t³` + R²).
Which R² is higher tells you **whether to chase the accelerometer or the gyro**.

`--from-jsonl` is the whole reason the raw log is kept: the same pool session can be re-estimated
**with a different attitude mode, a different calibration, a different anchor**. In particular
`--restart N` re-anchors every N seconds, turning one continuous curve (n=1) into dozens of independent
segments — in flight, per the operator decision, the anchor is set **only once**, and the p50/p95
statistics are drawn from here.

**Unverified on hardware** — verification status and order are in `KNOWN_ISSUES.md`.

## DVL as a third sensor — `--dvl` (2026-10-01, **unverified on hardware**)

> **The DVL is not on the vehicle yet.** The code in this section was written from nothing but the Water Linked DVL A50 protocol documentation and
> a 12-second capture from the vendor's public demo unit (`rov_gui/tests/fixtures/dvl_a50_demo_capture_20261001.jsonl`).
> Every number below is either [스펙](a value read from the docs) or [예측](an unfounded first value), and
> **not one** of them was measured on this vehicle or in this pool. The first pool session replaces the [예측] values.

**What changes.** State estimation is currently AprilTag + IMU. Position (x, y, yaw) comes from tag PnP, z·roll·pitch·angular
rates from the autopilot, and **the linear velocity `nu[0:3]` is the tag position, finite-differenced and low-pass filtered**
(`state_assembler.py` — the term whose lag the code itself notes as "the estimator's dominant error"). The DVL measures that
velocity directly. So the fusion is deliberately small (operator decision 2026-10-01: not a new EKF but **adding one velocity
source** to the existing assembler):

| Quantity | Now | When the DVL is IN and usable |
|---|---|---|
| x, y, yaw | tag PnP | unchanged |
| z, roll, pitch, p·q·r | barometer / ArduSub | unchanged |
| `nu[0:2]` (u, v) | tag finite difference + LPF | **measured directly by the DVL** |
| `nu[2]` (w) | tag finite difference + LPF | **unchanged** — see "Horizontal only" below |
| Position interpolation between camera frames | velocity × fix age | **integration** of the DVL velocity (fix capture time → now) |
| Tag dropout (station hold only) | IMU double integration or freeze | DVL velocity integration (`bridge_tier: dvl`) |

This is the same division of roles that "Learning to Swim" (arXiv:2410.00120v2) used on CUREE — AprilTag for global position,
DVL+IMU for "faster position feedback" and for "when the tags are not visible" (that paper's Fig. 5 caption). The paper only
says it fused them with an EKF and does not disclose the state vector, the noise or the latency handling. The approach in this
section does not use an EKF — the paper-style EKF runs alongside it in the [next section](#alongside-a-paper-style-ekf----ekf-2026-10-02-unverified-on-hardware) (`--ekf`).

```
./c3 gui --source hw --allow-command --mpc --dvl          # vehicle (TCP 192.168.2.95:16171)
./c3 gui --source demo --mpc --dvl                        # practise the wiring and toggles with a synthetic DVL
QT_QPA_PLATFORM=offscreen python rov_gui/tests/demo_e2e_dvl.py [wrong]   # the whole sequence in a real window
```

**Horizontal only.** The "proof" below looks only at surge·sway. A bracket mounted at a tilt, a wrong vz sign convention and a
wrong lever arm are not caught by that proof, so to keep unproven numbers out of the heave velocity, `nu[2]` stays on its
existing source (tag finite difference). The DVL's w is only recorded, in CSV `dvl_w` (safety review 2026-10-02).

### There are three switches

| Switch | Question it answers | Where it is changed |
|---|---|---|
| `dvl.enabled` / `--dvl` | Is a DVL client **created** | At launch. When off, nothing is created and the tick is bit-for-bit identical |
| `dvl.use` / panel **DVL** button | May the estimator **consume** it | Row 2 of the trajectory panel. **Only while disengaged + the controller CSV is closed** (same rule as the LOW combo: one run = one estimator) |
| (automatic) | Can **this tick**'s sample be used | If not, only that tick drops back to the tag finite difference. CSV `vel_src` states it for every row |

**OUT is shadow.** Even with the button at `DVL OUT`, the stream is still received, passed through the gates, compared with the
tag velocity and recorded. The commands are bit-for-bit identical to having no DVL (pinned by `test_dvl.py`). The point is to
see, in the same run, "what the DVL would have said" during the test phase.

The button only sends a **request**. What it shows is decided by `MpcStatus.dvl_use` — if the worker refuses (engaged, CSV open,
mount not measured, not proven), the button stays as it was and the reason appears on the chip. Colours: grey = OUT, green = IN
and currently flying on the DVL, amber = IN but currently on the tag velocity, **red `DVL TRIPPED`** = the trip described below
(the tooltip gives reason·altitude·fom·Hz·residual).

### Since 2026-10-16: `dvl.selection: primary` — a disagreement with the tag velocity is a **HOLD** that returns by itself

Operator decision (2026-10-15): the DVL is the station's **main horizontal velocity sensor**. The tag finite difference is the
worse of the two velocities (RMS against a non-causal tag fit: DVL 2.3–2.9 vs tag FD 5.5–7.0 cm/s), and the old difference
trip fired ~18 times an hour on replay [측정: RL_controller logs/eval/vel_source_20261015]. So under the default `primary`
(design and evidence: RL_controller `logs/eval/dvl_trip_fix_20261015/SPEC.md`):

- **The velocity is the trained blend.** Exactly what the pwm11att13 policy was trained on (`dvl.DvlBlend` = RL_controller
  `deploy/obs_builder.py DvlBlendNp`): when the newest sample cannot be used, the last good one is held for 0.45 s, then the
  DVL weight w fades to the tag FD over 0.5 s, and back up over 0.5 s once it can be used [스펙: export `velocity_source`].
  **No hard switch.** The re-entry wait, the innovation gate and "unchecked" are not on this path. Every CSV row carries
  `w_dvl` (the DVL weight) and `dvl_code` (why) (schema 22). Heave stays on the tag FD.
- **A sample is not used** on a data gate (below), on a **quality flag** — fom > 3 mm/s, an altitude jump > 0.10 m, < 4
  beams, a lock re-acquired (the report before it failed a gate: a flickering false lock) — or on a **disagreement with the
  tag velocity** (slow difference > 5 cm/s, the alignment verdict, a refuting fit). A disagreement is a **HOLD**: the fade
  to the tag FD starts on that tick, and the DVL comes back **by itself** once the disagreement and its evidence are gone. The
  proof is kept. **There is no button to press.**
- **Panel words**: `held` = not used for a moment, returns by itself (a flag, a disagreement, gross); `not used this run` =
  not armed at engage / a code fault (also when the lock drops in such a run); `no tag state` = this tick had no tag state.
  A lost lock reads `… — returns by itself` only while the DVL is IN and proven (in shadow, or unproven, the plain words).
  The chip gives the weight, the direction and why: `v:DVL`, `v:DVL 60%↓ (flag fom)` (fading out), `v:DVL 40%↑` (fading
  in), `v:TAG (DVL held: disagrees)`. The button face says `DVL TRIPPED` **only for a latch**. In the log every
  `dvl: HELD (not latched)` line has its `dvl: hold released` line (a hold that starts within 2 s of the previous HELD line
  is announced when those 2 s are up, or not at all if it ends sooner).
- **What still TRIPS (latches)** — two things: (a) the device's own configuration (rotation, sound speed) changing
  (unchanged), (b) **escalation** — a disagreement that **lasts** (4 s of accumulated evidence, `esc_latch_s`) or **keeps
  coming back** (3 separate episodes within 60 s, each with difference, fit or scale evidence — alignment evidence alone is
  held but not counted; a warning at the second). On the 0.735 h of 10-13/10-15 recordings it never fired (bucket max
  1.43 s, no counted episode); a +10 cm/s bias, a reversed sway, 180° and a 2× scale held for 30 s latched in 34/35,
  35/35, 35/35 and 31/35 windows, and **one short glitch, even a large one, does not latch** (0 of 850 injected
  +0.3–0.8 m/s × 0.3–1.5 s glitches; fix round 1) [측정: same folder, design/out, impl/out, fix1/out]. A latch discards
  the proof: **proving pass (surge AND sway, tags in view) → OUT → IN** — the log line says exactly that.
- **EKF**: it is given a DVL sample only while the blend may use it (not on a flag, a disagreement hold, a gate or a latch);
  its control-velocity interlock compares with the tag FD.
- **RL exports**: an export whose obs_spec says `velocity_source.trained: true` (pwm11att13 and later) loads only when the
  station runs the same blend (primary, equal constants), and flies **only on the TAG estimator** (engage is refused with EKF
  or imu_dr in CONTROL). The `obs_guard` block is checked the same way. Exports without the blocks (every shipped one) are
  unchanged.
- **Rollback**: `dvl.selection: legacy` = the 2026-10-01..15 ladder (re-entry, innovation gate, latching trips), bit for bit
  on replay. But (a) legacy **refuses** every export trained on the blend (RL_PWM / RL_PWM_7D become unavailable, warn in the
  log); (b) `stale_s` is also the blend's stale clock: a value other than 0.6 refuses those exports in primary too; (c)
  `selection` is read at startup and the proof is per process — switching = a restart + a new proving pass; (d) after the
  yaml patch (`RL_controller logs/eval/dvl_trip_fix_20261015/hw_mpc_dvl_patch.diff`) is applied, undo it first
  (`patch -p1 -R`) before reverting the code, or setup dies on the unknown dvl keys.
- **Remaining costs and blind spots**: KNOWN_ISSUES, "DVL `selection: primary`".

The "Three rules" below and the trip / innovation-gate / re-entry text after them describe **`legacy`** (under primary, rule
2's latches are only the two above; everything else is a hold).

### Three rules — the automatic logic only ever **removes** the DVL

1. **Whether a run may use the DVL is decided once, at engage.** If switch IN + proof complete + not tripped,
   that run is "armed". Otherwise that run stays on the tag velocity to the end — even if the proof completes mid-mission, the velocity source does not
   change (from the next engage on). meta `dvl.armed_at_engage` / `not_armed_why`.
2. **A trip latches.** Once **evidence that the DVL is wrong** appears — the fit refutes the mount, a persistent disagreement with the tag velocity,
   the directions of the two velocities do not match, the device's rotation or speed-of-sound setting changes — the DVL is dropped from the estimator and
   does not come back **until the operator presses the button OUT → IN again**. And once this session's data has said "wrong" even
   once, IN requires **a proof built after that** — even with `require_proven: false`.
   That opt-out exempts only "the first proof of a DVL with no counter-evidence at all" (re-audit 2026-10-02: under the opt-out the DVL went back IN without any grounds 1 second after a trip,
   and a refutation seen in shadow vanished together with the 60 s window). A trip discards the proof and
   the pairs that built it. A device config change does the same **in any state** — the pairs so far belong to a stream
   that no longer exists. A proof rebuilt while the latch is set likewise collapses
   when the next evidence appears.
3. **A data gap is not a trip.** Loss of bottom lock, a link stall, or an outlier in a single sample drops only that tick to the tag
   velocity, and the DVL comes back on its own once it has been good continuously for `reentry_s` (0.5 s).
4. **A trip only when it is "wrong in a lasting way".** There are two such ways:

   | | What | Which monitor |
   |---|---|---|
   | Mount | The tags **see almost all** of the motion the DVL reports (80 % or more, `COHERENT_MIN`) and place it in a different direction | the alignment monitor, the fit's (coherent) refutation |
   | false lock | Whatever the tags saw, the DVL sits more than `agree_trip_m_s` (8 cm/s) away from the tag velocity for about `agree_trip_s` (1 s) | the difference-vector monitor. A DVL that reports nonexistent motion for 1 second cannot be flown on — the message gets `a false bottom lock rather than the mount` appended |

   **Not a trip**: a glitch a few reports long (whether once or every few seconds), DVL noise with a mean of 0, a fit refutation
   that arises because only the DVL "moved". These drop the DVL without a latch, **only while they last**, and it comes back on its own once they pass:

   | hold | When | When it is released |
   |---|---|---|
   | innovation hold | A single sample deviates from the tag velocity by more than 15 cm/s (widened by as much as the fix has aged) | Only when the tag velocity of a **new camera frame** that arrived afterwards agrees. The limit widening as the same frame ages does not release it. Rejected samples are not used in the fix-age integration (position interpolation between camera frames) even after the release |
   | alignment hold | The alignment monitor reads "the DVL's motion is going in a direction the tags do not show" — however much of that motion the tags see | When that verdict disappears (it lingers for at most 2 s) |
   | refutation hold | The fit is currently refuting (15 or more in a row) and the last refutation evaluation is within 3 s (`REFUTE_HOLD_S`) | When the refutation clears, or when there has been no new refutation evaluation for 3 s |

   These thresholds were found by the re-audits over three rounds: without the "the tags are seeing it" condition, a correct,
   proven mount latched on two reports 12 cm/s off, or on 3 cm/s noise while at rest. With the condition at half (0.5), it
   latched on DVL noise of 2–2.5 cm/s, on a 4-report glitch every few seconds (which dragged the fit down to "scale 1.3×") and
   on a single 4-report glitch at a square corner. Without the innovation hold, as the tags disappeared and the fix aged, the
   spurious 0.25 m/s the gate had been rejecting got back in, and the bridge rung carried it 0.6 m. With the hold set and
   released on a "fresh frame" basis, tags 0.25 s or more late meant the hold either never released (DVL 0 % for 55 s) or
   was never set at all, and a 4–6 report glitch during a 0.3 m/s round trip shook the regression in a direction with almost no
   motion and tripped a correct mount (fifth audit). Even after the hold released, the samples that had been rejected stayed in
   the fix-age integration, putting 9–11 cm into the estimate, and when the tags disappeared the DVL rung carried it along
   (sixth audit, all reproduced). Conversely, any trip at all means the proof has to be redone — even one tripped by a false
   lock (even when the mount itself is fine); this is not distinguished for now.

### IN switches on only after the "proof"

A DVL mounted in the wrong orientation **does not show it at rest** (both read 0). The moment the vehicle moves, the sign of
the velocity feedback flips. So IN is refused until two things are in place:

1. `hw_mpc.yaml dvl.mount_rpy_deg` / `mount_t_flu` **filled in with measured values** (shipped value null).
2. **Proven with this session's data** — the 2×2 least-squares map between the DVL body velocity and the tag finite difference,
   `v_tag ≈ A·v_dvl`. The verdict is **three-way** and carries a standard error (σ = residual scatter / (√n · rms velocity on the weak axis)):

   | Verdict | Condition | Meaning |
   |---|---|---|
   | proven | max\|A − I\| + 3σ ≤ `agree_tol`(0.25) | **inside** the tolerance |
   | refuted | max\|A − I\| − 3σ > 0.25 | **outside** it (with a sentence on what is wrong) |
   | insufficient | otherwise | pairs < 60, motion along one axis only, or not enough data yet |

In practice: with the tags in view, drive manually fore-aft and side to side, and the log shows
`dvl: mount PROVEN …`. If it is wrong, it says what is wrong — "the DVL is rotated +90° from the config
(add that much to yaw)", "one axis is flipped (frame convention error)", "scale 1.5×". If you only move forward or stay in
place, **it issues no verdict at all** (forward motion alone cannot tell the y sign, and standing still tells nothing).
[예측: synthetic prototype — assuming tag noise 0.8 mm@15 Hz and DVL noise 3 mm/s, moving at 8~12 cm/s for 8 s gives
A ±0.01, yaw ±0.3°. These are not values for this DVL.]

A one-frame tag jump (a spurious velocity on the order of m/s once finite-differenced) is filtered out by one trimmed refit. If
fewer than 60 pairs remain after trimming, the verdict is **withheld** (it does not say "rotated 27°" from a contaminated fit).
For the fit to overturn an already proven mount, a refuted verdict has to come out **15 times in a row**. A stretch where the
DVL disagreed with the tags stays in the window for `agree_window_s` (60 s), so it cannot be re-proven during that time — this
is the intended behaviour.

**What this proof cannot see**: the vertical axis, the bracket's roll/pitch tilt (a tilt of e° only scales one row by cos e),
the lever arm (its error is orthogonal to the regressor). These three come down to the **tape measure**.

**`lag_s` — do the two sensors' clocks agree.** The same fit gives one more number: **how much later** the tag velocity
describes the motion than the DVL velocity with the same stamp (`v_tag(t) = v_dvl(t − lag)`; the residual regressed on the DVL
acceleration). If both sensors stamp "the time it was true" correctly, it is 0. It is not a gate but a **commissioning number**,
printed in the `mount PROVEN` log as `tag velocity +NN ms behind the DVL's by their stamps`. Why this value matters: both the
proof and the in-flight monitors pair the two sensors **by their stamps**. If the stamps are L seconds apart while the velocity
vector rotates at ω rad/s, that is indistinguishable from the mount being rotated by ω·L [유도]. Two things on how to read it:
because the DVL acceleration is a 2-point difference and noisy, the estimate is **biased toward 0** — read it as a lower bound
[예측: in a synthetic simulation, stamp errors of 50 / 100 / 150 ms read as +33 / +49 / +73 ms,
`rov_gui/tests/fixtures/sim_dvl_monitors_20261002.txt` A4]. And in an elliptical motion that **mixes fore-aft and side to side
at the same frequency**, the lag is absorbed whole into the map A and reads as 0, and the mount instead comes out rotated by a
few degrees (`test_dvl.py::test_the_fit_reports_how_far_apart_the_two_sensors_stamps_are`) — measure it with straight passes.

### In-flight monitors — two trips, gates and holds

| What | Condition | Result |
|---|---|---|
| Difference vector | Magnitude of the low-pass-filtered (time constant `agree_trip_s` 1 s) `v_dvl − v_tag` (horizontal) > `agree_trip_m_s` 8 cm/s | **trip** — a sustained offset: moving at speed with a flipped or rotated mount, a scale error, a long-lasting false lock. Even a flipped mount is invisible at 4 cm/s or less [유도: difference = 2·speed], and an oscillation cannot be seen because its mean is 0. The difference of a single pair is **clipped at a fixed `innov_max_m_s` (15 cm/s)** — independent of tag noise. So a single burst does not trip it, and a DVL that is badly wrong on every pair trips in about 0.8 s (re-audit: with the clip value set at 4σ of the proof scatter, that came to 36 cm/s with a 3 mm tag, and a single glitch of 3–4 reports tripped it in 0.4 s). A false lock while at rest: trips when +14 cm/s lasts 1 s or +10 cm/s lasts 2 s (table A5b below) |
| Alignment | For each **direction the vehicle actually moved** e over the last `agree_trip_s` (a principal axis of the DVL velocity; only those with rms velocity > `agree_ratio_v_min_m_s` 2 cm/s that account for 7 % or more of the motion — in smaller directions the regression is unstable and a single glitch shakes it), `1 − \|A·e − e\|`, with A = the least-squares map tag ← DVL. "Bad" if this value + 3σ < `agree_ratio_min` 0.5 | If bad, **alignment hold** (no latch). And if it persists for 0.4 s while the tags **see almost all** of that motion (`\|A·e\|` ≥ 0.8), **trip** — 1 for the same motion, −1 if flipped, −0.4 if rotated 90°. This is the item that catches a station hold **oscillating** on wrong-sign feedback. σ comes from the residuals of that regression itself, so slow or noisy motion becomes "no evidence". A mount that passes the proof (yaw 14.5° = deviation 0.25) cannot be caught [유도]. It works even when tags arrive at 5 Hz or less (a single pair's weight is capped at 1/7) |
| innovation | A sample deviates from the **most recent** tag velocity by `innov_max_m_s` 15 cm/s or more. That tag velocity serves as the reference even when up to 0.25 s old (the limit widens by as much as it has aged); a **tick that has had no tag velocity** for longer than that (the first tick right after a real tag gap) goes unused as `DVL unchecked` | That tick unused (not a trip; CSV `vel_src` is `tag_fd`). Then the innovation hold — unused until the tag velocity of a new camera frame that arrived afterwards agrees. It blocks large outliers such as a false lock on a wall reflection, and their return while the tags are disappearing |
| Fit refuting | The 60 s window's fit is "refuted" (`agree_refute_n` 15 or more in a row) and the last refutation evaluation is within `REFUTE_HOLD_S` (3 s) | Unused meanwhile (not a trip). A refutation of the mount would already have tripped; one due to noise or a false lock gets caught here |
| jump | Jumps by `a_max·dt + 5 cm/s` or more from the previous good sample (`a_max_m_s2` 2.0 [유도: 30 N / 19 kg ≈ 1.6]) | That sample unused — a spike, not a maneuver |

The two trip monitors consume **the same evidence as the proof**: the "time-aligned pair" produced at each new tag fix — the raw
tag 2-point difference, and the DVL velocity interpolated to its midpoint time. They do not compare the tick's (low-pass-filtered)
tag velocity with the latest DVL sample as is, because when the two sensors' latencies differ, that whole difference shows up as
"disagreement" — the first design did exactly that, and in a synthetic simulation a correct mount oscillating at 0.5 Hz or more
tripped once the two latencies differed by ~0.15 s or more [예측: an observation made during the design on 2026-10-02; A3 of the
script below now runs that condition]. Looking at alignment per **principal axis of motion** rather than per body axis
(surge/sway) is a fix from the same day: when the mount is rotated a few degrees, surge leaks into the DVL's sway, and a per-axis
slope reads that leakage as a "sway disagreement".

No tag fix, no pairs, no evidence — the monitors stop during a tag dropout (which is why the DVL rung is bounded by speed and
displacement caps, below). All thresholds [예측].

#### What the synthetic simulation says — **not a measurement**

`rov_gui/tests/sim_dvl_monitors.py` runs the real `DvlAid` + `StateAssembler` code with **assumed** timing (tags
15 Hz·latency 0.13 s·noise 0.8 mm, DVL 12 Hz·latency 0.19 s·noise 3 mm/s·arm 0.2 m, tick 20 Hz). The motion is a predefined
trajectory (open loop — the controller does not react to a wrong velocity) and the sensor numbers are all assumptions. Below is
the output of the 2026-10-02 run (`rov_gui/tests/fixtures/sim_dvl_monitors_20261002.txt`), all of it [예측]:

| Question | Result |
|---|---|
| Does a correct mount trip — 9 motions × 12 seeds (A1), with tag jumps (A2), 5 latency combinations (A3) | 0 trips |
| DVL glitches and noise (A5): a single 4-report ±14 cm/s (including at a square corner), 3 mm tag + 3–4 reports +25–30 cm/s, 1–4 report glitches repeating every 3–10 s, DVL noise 1–5 cm/s, glitches with the tags 0.28·0.35 s late — 21 conditions × 4 motions (including a 0.3 m/s round trip) × 6 seeds | **0 trips**. Instead the DVL drops out meanwhile (hold): with a one-off glitch, ticks flown on the DVL are 97–100 %; every 10 s, 85–100 %; every 3 s, 39–60 % at rest·76–100 % while moving. Even with late tags the hold releases after the glitch: 100 %. Noise of 2.5 cm/s gives 48 % at rest; at 3 cm/s or more it is almost never proven and stays OUT |
| **A long-lasting false lock** — here the trip is **intended** (A5b) | 6/6 trip at +14 cm/s × 1 s and +10 cm/s × 2 s (whether once or every 10 s). Shorter or weaker than that, 0/6 — neither +10 cm/s × 1 s nor +14 cm/s × 0.5 s trips |
| opt-out + correct mount, 300 s near the monitors' speed floor (C) | 0 trips |
| Flipped axis·90° mount put IN via opt-out (B) | Trips in 0.8–1.9 s. Except that sway reversal in 5+2 cm/s diagonal motion takes 8.6 s (because of the rule that does not judge directions under 7 % of the motion, sway reversal in a 10 cm/s oscillation goes 1.5 → 1.9 s) |
| **What it does not catch** (B's NEVER — meaningful only under opt-out; the % in the cell is the fraction of ticks flown on the wrong DVL) | ① every wrong mount in station keeping (1–2 cm/s), and a flipped axis moving at under ~2 cm/s rms (98 %). ② a flipped axis in **straight-line motion within 14.5° of the other axis** — independent of speed (20+3.5 cm/s straight line 98 %, 30+7.5 cm/s in phase at 0.5 Hz 90 %). In straight-line motion an axis flip is indistinguishable from "a rotation by 2× the angle of the motion direction" [유도], and within 7° it is the same as a rotation inside the proof's tolerance. ③ a 30° rotation below ~15 cm/s, a 2× scale below ~8 cm/s (98 %) — this comes from the difference-vector threshold [유도: 2·v·sin 15°, v must exceed 8 cm/s]. ④ a small (~3 cm/s) secondary motion axis flipped on top of a fast primary motion (0.15–0.3 m/s) — because the alignment monitor does not judge directions under 7 % of the motion (KNOWN_ISSUES). Even without a trip, the DVL drops out when the fit refutes: a 6·10 cm/s oscillation at 2× scale gives 15–16 % |
| The same table with DVL noise 2 cm/s (B2) | Flipped axis·90°: mostly 0.8–3.4 s (sway reversal at 5+2 cm/s 17.1 s). The station column is all NEVER, but with the alignment hold the ticks flown on the wrong DVL are 66–86 % (30°: 98 %); surge reversal in a 3 cm/s oscillation 2 %, 90° 24 % |
| A mount that **passed** the proof, going at speed (A6) | Residual yaw 12° or scale 1.2×: 0 trips up to 0.3 m/s, **trips at 0.4 m/s and above** (3/4, 4/4). 5°: 0 trips up to 0.5 m/s. Because the proof tolerance is relative (0.25 of the speed) while the difference-vector threshold is absolute (8 cm/s) — the proof log also prints "with this residual alone it trips above X m/s; correct yaw by +D°" |
| Wrong tag capture stamp + a 0.5–1.5 Hz elliptical oscillation filling the 60 s window (A4) | Even a correct mount trips: 50 ms late 2/36, 50 ms early 3/36, 100 ms 8/36, 150 ms 8/36 — the time offset looks like an apparent rotation. 0 trips in straight-line motion (square, reversals) |
| Wrong lever arm + 0.3 rad/s yaw sweep (A4) | 3 cm and 10 cm errors both 0/36 — **no monitor catches it**. The spurious velocity a wrong arm creates is motion only the DVL sees, and it falls short of 8 cm/s [유도: 0.3 rad/s × 0.1 m = 3 cm/s]. The tape measure is the only defence |

How to read it: on the default path (`require_proven: true`), everything under "what it does not catch" **is caught by the
prior proof** — ①③ get refuted (station keeping alone leaves the verdict withheld), and for ② a proof never forms in the first
place (the proof requires data in which surge and sway moved **separately**, so straight-line motion alone gives "verdict
withheld"). The monitors bear the responsibility alone only under opt-out. Then ①③ mean a velocity error of a few cm/s or less,
but **② does not**: the error on a flipped axis is 2× that axis's velocity (amplitude 15 cm/s in the 30+7.5 cm/s example).
Whether, in closed loop, the wrong-sign feedback grows the motion, breaks the straightness and so reveals itself, this
open-loop simulation does not tell us. **The opt-out is a convenience for skipping the proof pass, not a safety device.**

### Mount — axes and lever arm

* Device frame [스펙: docs.waterlinked.com/dvl/axes]: **x forward (LED side), y starboard, z down**. Mounted with the LED
  forward and the transducer down, it matches body FRD → `mount_rpy_deg: [0, 0, 0]`.
* The device can also rotate its own output (the web GUI's mounting rotation offset). Use **only one of the two** —
  with yaw in both it is applied twice, and the station logs an error when it sees the `get_config` response.
  The station asks `get_config` on connect and again every `config_poll_s` (10 s) after that, and
  **trips on a change** to the rotation offset or the speed-of-sound setting (since the proof is about the stream before the change).
* `mount_t_flu` is the position of the DVL measurement origin (A50: 28 mm **above** the backplate [스펙]), same as `cam_t_flu`:
  FLU metres relative to the body origin. It is a lever arm: yaw 0.3 rad/s with a 0.2 m arm gives an apparent velocity of 6 cm/s
  [유도: v = ω × r] — the same size as this vehicle's cruise speed, so it cannot be left out. The angular rate used for the correction
  is interpolated from the ATTITUDE history at **the time the sample was true** (~0.2 s before arrival). If the arm is not 0 and
  the history has no angular rate for that time (telemetry gap), that sample is **not used** — an uncorrected value is not the body
  origin velocity (`no body rate for the DVL lever arm`).
* The device does not correct roll/pitch mounting errors [스펙] → the first two fields of `mount_rpy_deg`.

### Sample gates (`hw_mpc.yaml dvl:`)

| Key | Value | Evidence |
|---|---|---|
| `stale_s` | 0.6 s | [예측] 1 vendor example report was already 0.19 s old at the time of transmission [스펙 example], 15 Hz at 0.6 m or less·10 Hz at 3 m or less [스펙], periodic cycling drops a few every 10 s [스펙] |
| `fom_max` | 0.05 m/s | [예측] the velocity accuracy the device reports. What it comes out to in this pool is unknown |
| `min_altitude_m` | 0.05 m | [스펙] A50 altitude range starts at 5 cm. **With the hull on the floor it loses lock** |
| `v_max_m_s` | 1.5 m/s | [예측] sanity. Applied to both the raw value and the value after lever-arm correction |
| `reentry_s` | 0.5 s | [예측] it has to be good continuously for this long before it is used (again). The count also restarts after a silence longer than `stale_s` |

A report that claims lock but **lacks fom or altitude counts as a failure** (fail-closed).

Latency: **do not assume it shrinks**. A single example in the device documentation is `time_of_transmission − time_of_validity = 189 ms`.
That is the same size as the tag path's latency (≈130 ms + LPF 44 ms), so the DVL's benefit is likely to be **noise**, not
latency — until it is measured, neither is claimed. The sample time is obtained by moving the device stamp onto the station
clock: the `도착 − tot` ("arrival − tot") of the report that **arrived fastest** over the last 30 s is taken as the offset, and
`t_valid = tov + offset`. The device clock needs no synchronization (its boot time is arbitrary [스펙]), and even when TCP
stalls and the backed-up reports arrive all at once, each keeps its **true age** (stamping them with the arrival time would make a
velocity several seconds old look brand new).

### When the tags are lost — the station bridge's DVL rung

When the tags cut out during a station hold, the existing ladder holds on with IMU/freeze for `imu_hold_s` (3 s) and then lets
go of x/y/yaw (COAST). If the DVL is **IN and usable without a break from the first dropout tick**, one more rung appears above
that: it carries x/y on by integrating the DVL velocity from the last fix, and holds all axes for up to `dvl_hold_s` (10 s [예측]).
If the DVL drops out for even a single tick, the existing ladder decides **with the same elapsed time**, and there is no return
to the DVL rung within the same dropout (because the position would then have an unknown hole).

On this rung there are no tags to cross-check the DVL against. So it is bounded by the premise **"a station hold is almost at
rest over one point"**: if the DVL reports a speed above `dvl_max_speed_m_s` (0.3), or the integrated position moves farther
than `dvl_max_disp_m` (0.5) from the last fix, the DVL rung is closed for that dropout
(`station_bridge: DVL rung closed — …`). Both [예측].

What follows depends on why the rung closed:

| Why it closed | Position the rung below (IMU/freeze) takes over |
|---|---|
| The DVL **went away** (lock lost, link stall, `dvl_hold_s` expired) | The position the DVL last integrated — it was trustworthy up to then |
| The DVL's story **makes no sense** (speed·displacement cap) | **Back to the last fix, at rest** — if it kept holding the integral of a DVL it has just judged untrustworthy, the controller would push out that spurious displacement (up to 0.5 m) until `imu_hold_s` ends (reproduced in the re-audit: an estimate error of 0.34 m persisted up to 3 s) |

If even one tick could not be integrated (a blind tick the bridge could not ride out, a fault the bridge does not cover), that
position now has a hole in it, so the DVL rung does not reopen until a new fix arrives.

And this rung opens **only if the vehicle was actually flying on the DVL at the last tick that had tags**. A DVL that the
innovation gate had been keeping out because the tags were refuting it does not become trustworthy just because those tags
disappeared — the same goes for the order in which tags actually disappear (fixes simply stop coming, and the gate limit widens
while the assembler keeps using the aging fix up to `tag_stale_s`): a DVL rejected once stays out under the innovation hold
until a new frame agrees (reproduced in the re-audit: in the second audit it was taken up at the first dropout tick and walked
0.5 m; in the third, once the fix was 0.35 s old, it passed the widened gate and walked 0.6 m). When the rung closes on a cap and
goes back "to the last fix", that fix is **the fix's own position** — the estimate at the last tag tick may already contain that
DVL's integral (third audit: it went back to a spot 0.125 m off).

The cost: if the vehicle is kicked hard (~1 m/s²) while the tags are visible, even an honest DVL ends up more than 15 cm/s apart
from the tag velocity of that moment (which lags) and a hold is set; if the tags then disappear, the existing ladder takes over
instead of the DVL rung — the same behaviour as with no DVL.

What the DVL rung hands down to the rung below (IMU) is **position only**. On every DVL-rung tick, the IMU rung's dead reckoner
is re-anchored **at rest** to the position the DVL integrated, and on ticks with tags it is anchored with the **tag velocity**,
even on ticks flown on the DVL (the same value as before the DVL). When a false lock simply ended without hitting a cap (lock
lost, link silence, return to the original value), the IMU rung inherited that velocity and integrated it for the whole of
`imu_hold_s` — 0.67–1.2 m on a stationary vehicle (fifth audit).

When the tags come back, the log gets `DVL dead reckoning was N cm off, gyro yaw M deg off` **only if the DVL carried it all
the way** — the number that will move `dvl_hold_s` from [예측] to measured. If it was cut off midway, the drift after that is
mixed in, so only "the DVL carried the first X s" is logged and no error is written. **Station only** (operator decision).
Path, policy and replay missions disengage on a stale fix, as they do now.

> Two pre-existing defects fixed in the same change (they apply regardless of the DVL): ① the panel's `NO TAG — COASTING` chip
> appeared only with `--imu-dr` on (`MpcStatus.bridge_tier` was filled only inside the dead-reckoning block) —
> now it always appears. ② disengage wrote the closing meta **after** clearing the ladder counters, so on disk
> `station_bridge.dropouts` / `reached_coast` / `last_recovery` were always 0 / 0 / null — now they are
> reset to 0 only at engage.

### Records (schema 17)

* **The last 15 CSV columns**: `vel_src, nu_u, nu_v, nu_w, fd_u, fd_v, fd_w, dvl_u, dvl_v, dvl_w, dvl_ok,
  dvl_valid, dvl_fom, dvl_alt_m, dvl_age_s`. Velocities are **body FRD m/s** (u forward, v starboard, w down —
  the frame of `uX..uN`, not the world FLU of `px/py`). This is the first time this CSV holds body velocities.
  `vel_src` says where that row's **horizontal** velocity came from: `tag_fd` | `dvl` | `bridge`. **Do not pool statistics
  derived from velocity across `vel_src`** — they are two instruments with different noise and latency. `fd_*` and `dvl_*` are
  always recorded together, whichever one was flown on (`nu_w` is always `fd_w`).
* **meta `dvl`** (always recorded): `enabled`, `use`, `use_at_engage`, `armed_at_engage`,
  `not_armed_why`, `proven`, `proof` (A at the time of the proof, the yaw correction, scale, σ, sample count),
  `agreement_now`, gate values, `device_config` and its change history, `rows` (row counts for `dvl`/`tag_fd` and
  **row counts per fallback reason**; rows bridged below the DVL rung are `bridge: tier imu|coast`; holds are
  `fallback: DVL innovation hold` / `DVL alignment hold` / `DVL held back`), `tripped`,
  `against` (the counter-evidence this session's data has left against the DVL — the proof clears it), `innovation_hold` (the reason
  for the innovation hold in effect at closing time), `trips`, `innovation_rejects` (whole session; per run: `rows`), `monitor_pairs` /
  `gross_pairs` (pairs the monitors saw / of those, pairs 15 cm/s or more apart), `monitor_now` (at closing time, the difference-vector
  magnitude and the per-direction alignment value·σ). `proof` and `agreement_now` also contain `lag_s` / `lag_sigma_s`.
* **Raw stream**: while the controller CSV is open, `<stem>_dvl.jsonl` gets the lines exactly as the device sent them
  (velocity, position_local, response) + the station arrival time. If the mount is re-measured, it can be reprocessed.
* The device's own dead reckoning (`position_local`) is **not used** — it is the device IMU integrated in the device frame
  and knows nothing of the tag world. It is kept only in the raw log.

The station **only reads** the device. The only thing it sends is `get_config` (itself a read); it does no config changes, dead
reckoning resets or ping triggers (because those would change the instrument for other clients too). The TCP server accepted
two clients at once (observed on the vendor's **demo** unit, 2026-10-01:
`rov_gui/tests/fixtures/dvl_a50_demo_capture_20261001.README.md` — unconfirmed on hardware) → so it is expected to be readable
alongside the BlueOS DVL extension [예측].

If the DVL code itself throws an exception on the tick path, the DVL is ignored for that session (one `CODE FAULT` log line)
and the control tick keeps running on the tag velocity — a new sensor's bookkeeping cannot abort a tick in the engaged state.

### Caution for the RL followers

`rl` / `rl_pwm` were trained **with the tag finite-difference estimator simulated**
(`control/rl_policies/*/RL_controller_README.md`: "no DVL"). DVL velocity is an observation distribution never seen in
training. It is not blocked; at engage it is left as a warn log line and in meta `dvl.use_at_engage`.

### Turning it on in the pool for the first time

1. After mounting, fill in `mount_rpy_deg` and `mount_t_flu` **measured with a ruler**. Leave the evidence in a comment. The proof
   only looks at horizontal rotation·sign·scale — for tilt and the lever arm, this tape measure is all there is.
2. `./c3 gui … --dvl`, with the button left at `DVL OUT`. Check that the **DVL** line in SENSORS reports Hz and altitude
   (out of water it overheats [스펙: 55 ℃ shutdown] — always switch it on submerged). In the log's
   `dvl: device config {...}`, check that `speed_of_sound` matches the pool water temperature and that `mounting_rotation_offset`
   is 0.
3. Where the tags are visible, drive manually **in straight lines**, once fore-aft and once side to side → `mount PROVEN` log.
   If it does not appear, the tooltip/log says why. (An elliptical trajectory mixing fore-aft and side to side creates an apparent rotation
   when the two instruments' clocks are offset [유도] — straight lines are safe.) If that log's `tag velocity +NN ms behind`
   (`lag_s`) **exceeds a few tens of ms, stop before going IN**: it means the two sensors' stamps are offset (camera
   capture time or the DVL `time_of_validity` mapping), and in that state the monitors trip a correct mount during
   oscillating motion (table A4 above).
4. After flying **one run in shadow**, plot the CSV's `dvl_u/v` vs `fd_u/v` — noise, latency (cross-correlation), fom and
   Hz per altitude get measured here for the first time. Replace `stale_s` / `fom_max` / `innov_max_m_s` /
   `agree_*` with these values.
5. Then go `DVL IN`, starting with station hold.

## Alongside a paper-style EKF — `--ekf` (2026-10-02, **unverified on hardware**)

"Learning to Swim" tied AprilTag + IMU + DVL together with an **EKF** (50 Hz, ROS). Neither the paper nor the
public code (`warplab/isaac-auv-env`, simulator only) gives the EKF details, but the description of that lab's
state-estimation repository (`warplab/robot-state-estimation`, empty) reads "state estimation for marine robots
using the ROS localization package". So "the paper's method" here is **a line-by-line port of ROS
`robot_localization`'s `ekf_localization_node`** (`rov_gui/control/rl_ekf.py`, original noetic-devel @ 9ef26a57):
a 15-state kinematic EKF, the r_l default process noise, Joseph form, the measurement queue and the 50 Hz timer,
delayed-measurement handling, lever arm (the angular velocity **of the state**), gravity removal (the attitude
**of the IMU message**). It runs in NED/FRD, and the only sign that changes is gravity's.

**It is on by default and runs in shadow.** Every tick it runs beside the assembler and is only recorded — so
every run carries comparison data. Which state the controller receives is decided by the panel's **EST** button.

```
./c3 gui --source hw --allow-command --mpc                 # EKF shadow (default)
./c3 gui --source hw --allow-command --mpc --dvl           # + DVL fusion (only when the mount has been measured)
./c3 gui ... --mpc --ekf control                           # fly on the EKF from the start
./c3 gui ... --mpc --ekf off                               # no EKF (CSV columns are nan/0)
python -m rov_gui.tools.ekf_replay <run>/<stem>_mpc.csv \
    --set smooth_lagged_data=true --set history_length_s=1.0   # the same run with different settings
python rov_gui/tests/sim_ekf_compare.py                    # synthetic comparison (table below)
```

| Switch | What it decides | When it changes |
|---|---|---|
| `ekf.enabled` / `--ekf off\|shadow\|control` | whether the EKF is **built** | at launch (on by default) |
| panel **EST** button (`EST TAG` / `EST EKF`) | **which state** the controller receives | only when **disengaged + controller CSV closed** (the DVL button rule). To EKF only when it is ready (initialized·finite·tags fused·output fresh). It cannot start in control from YAML — `--ekf control` only |

**What does not change when flying on the EKF**: the green trajectory in the plot, CSV `px/py` and all
interlocks are the **tag state** (the same rule as imu_dr control). The EKF is drawn alongside as a teal
dashed trajectory + a ring (when engaged). A run flown on the EKF **ends rather than silently falling back
to the tags**: an EKF code fault; an input the EKF fuses going silent (tags > `engage.tag_stale_s`, IMU
attitude·angular velocity > `engage.imu_stale_s`, acceleration > 1 s (because SCALED_IMU2 can be at 2–3 Hz),
depth > 1.5 s — r_l predicts to "now" and stamps that even with no input, so the output stamp cannot tell);
a **station bridge tick** (tag dropout — the bridge does not carry EKF runs: the problem of uncorrected IMU
integration + coast retarget writing the drift into the target, safety audit 2026-10-02 critical). These two
cases are debounced by `engage.tag_stale_hold_s` (0.4 s) like the tag path's stale gate, and **no command is
sent in the meantime** (it does not fly on the bridge state instead) — if the tags come back within that time,
it keeps flying on the EKF; or departing from the **fresh** tag state by more than xy·z 15 cm, yaw·roll/pitch
10°, velocity 15 cm/s (the difference vector put through a 0.5 s low-pass — because of tag FD noise) for 0.25 s
(`control_max_*` [예측]). To switch EST to EKF or to engage on the EKF, the two states must **agree** within
5 cm / 3° (`control_engage_*`) — because the "here" targets (engage hold, mission origin) are set from the tag
state.

**The DVL's gates apply to the EKF too**: the EKF fuses the DVL only when the DVL button is **IN**, the
mount-proof gate is passed (`DvlAid.refusal` — refused before PROVEN even when started with YAML
`dvl.use: true`) and there is no trip (r_l has no such gates, but proving the mount sign is not r_l's job — a
wrong-signed velocity turns the D term into negative damping even while the position is tied to the tags).
Every report stays in the log (`fed` flag), and `ekf_replay --dvl-all` shows the ungated EKF offline.

**Input selection (not in the paper → chosen)**: tags give x, y, yaw (+ z if hw_nav `z_source: tag`), the IMU
gives roll, pitch, angular velocity, acceleration (compass yaw excluded), pressure depth gives z with the same
offset as the assembler, the DVL gives (vx, vy). That is, it gets **the same information as the assembler**, so
that any difference comes only from "the fusion method". IMU noise is the mavros imu plugin defaults
(acceleration 0.0003, gyro 0.02°/s, attitude 1.0) — the values a BlueROV2+ROS user gets.

**Records**: the last 16 CSV columns (schema 18) — `est` (the state the controller received on this row:
tag|ekf|dr), `ekf_ok`, `ekf_px/py/pz`·`ekf_roll/pitch/yaw_deg` (the same world FLU datum convention as px/py),
`ekf_u/v/w` (the same body FRD as nu_u), `ekf_sig_xy_m`, `ekf_age_s` (r_l publishes the state at the **time of
the last measurement**), `ekf_tag_late_s`, `ekf_n_tag/n_dvl` (cumulative over the session). `est = hold` is a
tick on which the EKF run was debouncing and so **sent no command** (u/w are nan, solver_status −1 — the same
marker as observe). The meta `ekf` block (always), and `<stem>_ekf_inputs.jsonl` — every input the EKF received
(arrival time included), the engage datum, and the 50 Hz timer phase. `ekf_replay` uses it to re-run the same
run with different settings (with the same settings, within 1 mm of the live columns after convergence —
`test_rl_ekf`).

### What the synthetic comparison says [예측: synthetic·open loop — `rov_gui/tests/sim_ekf_compare.py`, output `rov_gui/tests/fixtures/sim_ekf_compare_20261002.txt`]

Assumptions: tags 15 Hz·0.13 s latency, IMU 25 Hz, depth 10 Hz, DVL 12 Hz·0.19 s latency, tick 20 Hz. Scores
are against the **true value at tick time** (what the controller receives at that instant).

| 0.3 m/s round trip (`passes`) | position RMS | (u,v) RMS | yaw RMS |
|---|---|---|---|
| assembler (tag FD) | 5.0 mm | 3.57 cm/s | 0.30° |
| assembler + DVL | 3.7 mm | 3.18 cm/s | 0.30° |
| **EKF (r_l defaults)** | **33.1 mm** | 2.62 cm/s | 0.28° |
| EKF, smooth_lagged_data | 6.2 mm | **0.72 cm/s** | 0.28° |

- **The r_l defaults (no smoothing) correct the _current_ state with a late tag frame.** The position lags by velocity × tag
  latency (the 33 mm above), and yaw likewise lags by 1.7° while turning, for the same reason (`yawing`). Since the paper's launch
  file has not been made public, it was left at the most likely setting = the defaults.
- r_l's own fix, `smooth_lagged_data: true` (rewind and re-fuse), removes that lag, and with even the accelerometer fused on time,
  **its velocity is the best of all the estimators**. The few mm of position error that remain are because r_l outputs not "now" but
  the state at the time of the last measurement (`predict_to_current_time: false`).
- The r_l default process noise (0.05 m²/s on x, y) grows the variance by 1e-3 m² every 20 ms → the EKF accepts tag frames
  almost as they are, and even at rest its own 1σ is a few cm.
- The mavros acceleration noise values trust the accelerometer almost completely: with an acceleration bias of 0.1 m/s² and no DVL,
  the velocity error is 15 cm/s. With a DVL it is masked.

Verification: `rov_gui/tests/test_rl_ekf.py` (port: Jacobian = the model's derivative, a delayed measurement is
corrected without prediction, Mahalanobis, Joseph, smoothing = the answer from processing in order, DVL mount
rotation·lever arm·no-TF, gravity sign / worker: shadow changes not a single command, switch-refusal rules, EKF
flight·divergence·fault termination, CSV/meta/log, replay reproduction / panel). 0 water runs.

## Battery remaining — the screen says which of the three you are looking at

If `BATT_CAPACITY` and the current monitor are not configured, ArduSub sends `battery_remaining` as
**−1 (unknown)**. That is why the demo showed 95% while the hardware showed only `--`.

Now it is filled in by source priority:

| Source | Display | Reliability |
|---|---|---|
| vehicle `BATTERY_STATUS.battery_remaining` | `battery 58%` | the autopilot's own value |
| coulomb counting `(capacity − consumed)/capacity` | `battery 58%  est mAh` | **[유도]** integral of the measured current |
| voltage curve | `battery 58%  est V` | **[유도]** reads low under load |

Giving `--battery-capacity-mah 18000` (stock Li-ion) turns on coulomb counting, and the tile changes from
`USED` to **`LEFT` (mAh remaining)** — mid-dive, the remaining amount should be visible at a glance, not
worked out in your head.

Use the voltage estimate knowing its two limits: **the pack sags under load** (with 8 thrusters pulling it
drops by nearly 1 V, so it reads lowest when it is working hardest), and **the curve is almost flat at
3.9–3.6 V** (most of the capacity is there, so a 0.05 V error is worth about 10%).

For the cell count, **do not estimate it; pin it with `--battery-cells`** (default 4 = stock 4S). From
voltage alone, **an empty 4S (12.0 V) and a full 3S cannot be told apart**, and getting it wrong in that
direction makes a discharged battery read 82%. So if the specified cell count is plausible it wins, and the
estimate runs only outside that range (a regression test guards 12.0 V → 0%).

**The proper fix is on the vehicle side**: configure `BATT_CAPACITY` and the current monitor and ArduSub sends
`battery_remaining` itself; this panel then drops the estimate and uses that value.

## Two IMUs — and two lines in SENSORS too

`IMU (ROV)` and `IMU (C3)` are **different devices, different clocks, different rates**. Merged into one
line, you cannot tell which one went quiet, and worse, you cannot tell which one the downstream is fusing.

| | `IMU (ROV)` | `IMU (C3)` |
|---|---|---|
| Device | autopilot (Navigator) | C3 onboard BNO086 |
| Path | MAVLink `RAW_IMU` / `SCALED_IMU2` | DepthAI IMU queue |
| Clock | vehicle `time_boot_ms` | **the same clock as the images** (`dai.Clock`) |
| Default request | per message: `ATTITUDE` 50 · `RAW_IMU` 20 · `SCALED_IMU2` 20 Hz (2026-10-15; all three at 200 Hz before) | 200 Hz (`--c3-imu-rate`) |
| Measured ceiling | **~208 Hz** | **~486 Hz** (standalone) |

* **ROV IMU** (measured 2026-08-06, `SET_MESSAGE_INTERVAL`): default 2-3 Hz · request 50 → 62 ·
  100 → 125 · 200 → **208** · 400/1000 → still 208. The cost is about 0.2 Mbit/s.
  Raising the rates is **on by default** (off with `--no-mavlink-set-rates`) — the ArduSub
  default of 2-3 Hz is good for nothing but dashboard lights, and a sensor log kept beside a depth recording
  is only worth keeping at a proper rate. When on, it **transmits** to the vehicle.
* **Per-message request since 2026-10-15** (`rov_gui/mavlink_streams.py`, operator-approved fix 1c): the old
  request (all three at `--mavlink-rate` 200) came out at ~166.6 Hz each, ~500 of the ~552 msg/s the vehicle sends
  to **every** BlueOS endpoint [measured, 27 recordings, `RL_controller/logs/eval/station_fix_20261015/impl-rates/out/
  rov_stream_scan.json`], and on 10-13 a vehicle-side queue formed on top of it at every parameter download by
  another client. Now `--mavlink-att-rate` 50 · `--mavlink-imu-rate` 20 · `--mavlink-scaled-imu2-rate` 20 Hz
  (90 msg/s asked); `--mavlink-rate N` is **the old request unchanged** (all three at N; giving it together with a
  per-message flag is refused at launch). Why 50 Hz ATTITUDE, and why SCALED_IMU2 (whose accel/gyro are all zero on
  this vehicle) stays on: `rov_gui/mavlink_streams.py` and KNOWN_ISSUES 2026-10-15. The request and the rates that
  arrived are recorded in the run meta `mavlink_streams` and the `*_rov.json` `stream_request`.
* **C3 IMU**: using only accel+gyro, standalone ~486 Hz (`c3_camera/pipeline.py`, measured).
  The batch threshold is 10 — at this rate, a threshold of 1 leaks samples (`c3_collect.py`, measured).
  It can be turned off with `--no-c3-imu`.
  **The default request went down from 500 → 200 (2026-08-17)**: ask for more and you actually get less.
  With a 500 request, this station measured **234.9 Hz**
  (`data/20260809/0809_202407/c3_depth_20260809_202409_c3_imu.json`); with a 200 request,
  the measurement at `pipeline.py:67` is **194 Hz**. In this regime, the more you raise the request, the more throughput you lose.
  Also, this paragraph was **wrong** for a long time — it said "turning on `ROTATION_VECTOR` makes every stream
  collapse to ~40 Hz", but the measurement table in the same file (`pipeline.py:71`)
  says **there is no penalty when requested at the same rate** (accel+gyro @200 +
  ROTATION_VECTOR @200 → 194 Hz). What drags it down is the magnetometer, and a rotation vector requested **slower**
  than accel/gyro.
* The `IMU (C3)` line keeps updating when the C3 is alive, even if the ROV is absent. For checking the camera IMU
  before a dive, a panel that has to wait until the vehicle answers is useless.
* The rate cell shows the **rate**. Before, the detail string won, so where `486 Hz` should have been,
  `BNO086` was showing. The only exception is when the detail already contains a rate
  (the REST transport's `200 Hz sampled`), because the caveat that it is **our polling
  rate**, not the vehicle's transmit rate, must not be stripped off.

### The ROV camera is received through GStreamer (the same pipeline as QGC)

The FFmpeg path was **measured on every metric, and all of them were clean** — kernel receive queue 0, drops 0,
20-second drift +1 ms, frame interval max 48 ms, no B-frames, chain capacity ~430 ms measured with a stall
test. Yet the picture was still **about 2 seconds late, while BlueOS's own viewer and QGC showed the same
stream without delay**. When every measurement on a component says it is healthy and the component is still
wrong, the right move is not to defend that component but **to switch to an implementation that actually
works on this machine**.

```
udpsrc ! rtpjitterbuffer latency=0 drop-on-latency=true
       ! rtph264depay ! h264parse ! avdec_h264
       ! videoscale ! videoconvert ! video/x-raw,format=BGR ! fdsink sync=false
```

The two lines that matter: `rtpjitterbuffer latency=0` (removes the jitter buffer itself — **FFmpeg has no
counterpart.** The RTP reader thread fills a FIFO several MB in size that from the outside can be neither sized
nor inspected. That is why `/proc/net/udp` looked like 0) and `fdsink sync=false` (output frames as they are
decoded instead of holding them back to their timestamps). Scaling is done in the pipeline (shrinking in I420
moves only half the bytes of shrinking after the BGR conversion). Measured 19-20 fps, interval p50 48 ms /
max 80 ms.

If `gst-launch-1.0` is present it is selected automatically; `--rov-cam-backend ffmpeg` switches back.

**The resolution follows the panel** (`--rov-cam-size auto`, the default). The pipeline's output size is fixed
at launch, so when the panel grows enough (e.g. moved into a large slot with a double-click) the pipeline is
relaunched — 464x262 → **940x528**, ~0.5 s. Hysteresis (1.25×/0.6×) and a 3-second cooldown keep it from
restarting over and over while the window is dragged, and it does not exceed the source's 1920x1080 or its
aspect ratio (it does not invent pixels that are not there). If you want maximum detail even in a small panel,
fix the size, e.g. `--rov-cam-size 1280x720` — the panel's downscaling makes it slightly sharper, at the cost
of more bytes flowing through the pipe.

What was fixed in the earlier FFmpeg path (still in place): the real culprit was the **stream probe buffer**.
With the default `probesize`/`analyzeduration`, `open()` takes **1.7 seconds**, video piles up meanwhile, and
the decoder starts that far behind. On top of that, 1080p decoding cost more than the 33 ms budget, so relative
to the stream's own PTS it **slipped a further 300 ms every 15 seconds** (measured). Switching to
`probesize;32` `analyzeduration;0` `max_delay;0` gives open in 0.1 s and **+1 ms of drift over 20 seconds**.
The cost is 26 → 20 fps — the socket drops the frames we cannot keep up with, and for a live view that is the
right trade (fewer frames received, but the latency does not grow). If 30 fps is needed, lower the stream to
**1280×720** in BlueOS Video Streams.

On top of that, it **also flushes the decoder backlog** (`_grab_latest`): `read()` plays back the frames piled
up in the buffer in order, so once it falls behind during startup, that latency **persists forever** (because
from then on it consumes at the same rate as frames are produced). At the live edge a grab takes one frame
interval, while a frame coming out of the buffer returns much faster than that — that tells them apart, and it
skips ahead to the newest frame without colour conversion (`grab` only). FFmpeg runs with `threads;1`, because
frame threading holds (threads−1) frames (3 frames at 30 fps = 100 ms). Even with 1 thread it decodes this
1080p stream at 31.0 fps (measured).

Even so, the ROV camera (`--panel2 rov`) is **not a low-latency path** — it is the same stream as QGC's,
H.264-encoded on the Pi and arriving over RTP. It is for situational awareness; piloting is done on the C3
feed. RTP timestamps are not synchronized with the host clock, so it **does not display** a latency number
(it would be a made-up number).

## Sending commands — two gates

It is exactly the principle of `c3_camera/control.py`: ArduSub is always the flight controller, and **two
command sources are the danger** (ArduSub follows whichever MANUAL_CONTROL arrived last, so sending at the same
time as QGC makes the vehicle shake between the two inputs).

* **Commands go out over `udpin:`** (default `udpin:0.0.0.0:14552`). It binds the port that BlueOS pushes to
  and **fires back at the sender** — exactly what QGC does.
  `udpout:192.168.2.2:14550` was the original default and **silently did nothing**:
  a 2026-08-06 probe found the vehicle's UDP 14550 ICMP port-unreachable (BlueOS's inbound
  "GCS Server Link" is disabled; outbound leaves from an ephemeral port). So an
  endpoint must exist: `./c3 blueos_endpoint add --port 14552 --yes`.
  If nothing has arrived yet, the CMD pill shows red + `no vehicle on udp/14552 — nowhere to
  send`. The original problem was that this failure **was not visible**.
* **`--cmd-sysid` must equal the vehicle's `SYSID_MYGCS`** (default 255). ArduPilot accepts
  pilot input **from exactly one GCS only**:

  ```c
  void GCS_MAVLINK::handle_manual_control(const mavlink_message_t &msg) {
      if (msg.sysid != sysid_my_gcs()) { return; }   // silently dropped
  ```

  No NAK, no error. But **COMMAND_LONG (=ARM) does not go through this check** — which is why
  on 2026-08-06 this station could arm, yet every thruster sat at 1500 µs (vehicle
  SYSID_MYGCS=255, sink 245). The gripper/lights are button bits of MANUAL_CONTROL, so they get dropped along with it as well.
  Now the sink **reads** `SYSID_MYGCS` at boot, and if it differs, the CMD pill shows red +
  `SYSID_MYGCS=255 != 245 — sticks ignored`.
* **The arm indicator trusts only the heartbeat of component 1 (the autopilot).** At least three
  systems send heartbeats on this link — the autopilot (1/1), an onboard component (1/194), the GCS (255/240).
  1/194 uses `base_mode = 0x80`, which happens to be exactly the SAFETY_ARMED bit, so when any heartbeat at all
  was accepted, **a disarmed vehicle was shown as MOTORS ARMED** (measured 2026-08-06). An arm indicator that
  can be wrong is worse than none. `c3_camera/mavlink_log.py` was fixed as well.
* The default is `NullCommandSink` — it transmits nothing.
* Only with `--allow-command` is the MAVLink sink **created**, and only once the UI's
  `COMMAND ENABLE` switch is then turned on does it **transmit**.
* **arm / disarm are implemented** (2026-08-06, user request). In exchange, two asymmetric rules:
  - **ARM is a 1.2-second hold-to-confirm**, requires `COMMAND ENABLE` to be on,
    and at the moment of pressing sends the command **after setting every axis to 0** (arm with a key held
    down and the vehicle moves the instant the motors come alive). A modal "Are you sure?" window is not used
    because it steals keyboard focus from the window that is flying the vehicle, and pilots learn to close it
    without reading it.
  - **DISARM is a single click** and goes out even when `COMMAND ENABLE` is off.
    Hiding the stop behind a switch is the more dangerous mistake.
  - The `MOTORS ARMED` indicator shows **the vehicle's HEARTBEAT** as is. It is not painted armed just because
    we sent the command — that way an arm refused by the pre-arm checks shows exactly as "nothing
    happened". If telemetry is stale it is `MOTORS ?` (yellow).
* **The station changes the flight mode** (ARM requests MANUAL first; the teleop panel's MANUAL|STAB|DEPTH buttons; the "Flight modes" section below).
  Which of MANUAL / STABILIZE / DEPTH HOLD is active changes the meaning of the same axis input — the pre-2026-08-07 sentence "leave it to QGC" is withdrawn.
* E-STOP goes only as far as **neutral + transmit disabled**; it does not disarm. Underwater, disarming
  means losing depth hold and the vehicle drifting, so it is not always safer either. If the motors really
  have to be killed, use the DISARM button next to it (or QGC).
* While enabled, it sends a 1 Hz HEARTBEAT (because the ArduSub GCS failsafe is the mechanism that stops the
  vehicle when this station dies). On disable, the heartbeat stops immediately.
* Deadman: if the input is older than `--deadman-ms` (default 500), it **sends neutral**.
  "Not sending" is not a stop command.
* The MANUAL_CONTROL `z`-axis convention is **stated explicitly** with `--z-convention`. Depending on the ArduSub
 version it is either `0..1000` or `-1000..1000`, and if it is wrong, the vehicle descends when neutral was intended.
* **The gripper/lights go out as ArduSub joystick button functions, and the two functions mean different things.**
  Read them from QGC > Vehicle Setup > Joystick > Button Assignment and pass them in:

  This vehicle's values are **baked in as the defaults** — no need to pass them every time:

  | Button | `BTNn_FUNCTION` | ArduSub function | Meaning | Flag (default) |
  |---|---|---|---|---|
  | 0 | 77 | `servo_1_max_momentary` | opens **while held** | `--btn-gripper-open 0` |
  | 15 | 76 | `servo_1_min_momentary` | closes **while held** | `--btn-gripper-close 15` |
  | 13 | 33 | `lights1_dimmer` | one step **per press** | `--btn-lights-down 13` |
  | 14 | 32 | `lights1_brighter` | one step **per press** | `--btn-lights-up 14` |

  Even as defaults, **they are checked against the vehicle at startup**: the sink reads each `BTNn_FUNCTION`,
  and if it differs from the numbers above, it **does not use** that bit and logs the reason. This is so that
  the wrong button never gets pressed on a different vehicle. To leave a function completely untouched, give `-1`.

  So the gripper is sent as a **hold**, and the lights as a **pulse (press)**. Moving the lights slider
  **queues up button presses** for the difference and sends them out at 100 ms on / 100 ms off
  (ArduSub looks at the *change* between consecutive messages, so a button must be pressed and released before the next press registers).
  The number of steps has to match the vehicle's `JS_LIGHTS_STEPS`; `--lights-steps` is that value.
  Swap the two — send a level and the lights go one step and stop; send pulses and the jaw
  just twitches. If no bit number is given, it **sends nothing** and the panel shows OFFLINE +
  `no button bit mapped`.
* **The light level is handled as an absolute value.** One slider step = one vehicle step, and the step count is
  matched to `JS_LIGHTS_STEPS`, **read from the vehicle** (this vehicle has 8, not the 10 that used to be the code
  default — this was half of "it's at 70% but looks off"). The other half is accumulated
  error: lose even a single press and our model is permanently out of step, with no feedback to bring it back.
  So **OFF/MAX are resync points** — pressing twice past the end brings the model back to the truth.
  Given `--lights-servo N`, it reads that servo output and shows the **measured level**
  (on this vehicle, SERVO13 is 1100 µs when off).
* **The gripper has no slider.** That is because a momentary servo has no position to command.
  It is driven while CLOSE/OPEN is held, and the panel shows the direction and how long it has been held.

## Keyboard

```
W A S D / ←→↑↓     surge, sway   (arrows: surge, yaw)
Q E                yaw
R F / PgUp PgDn    heave
space              ALL STOP
G H                gripper close / open (while held)
[ ]                lights down / up
, . /              camera tilt down / level / up (`,` and `/` while held)
Ctrl+R             UI recording start/stop   F11 full screen   Esc  ALL STOP + transmit disabled
```

arm/disarm **deliberately have no keyboard shortcuts.** This is so as not to create a path where a slipped
finger brings the motors to life — ARM has to be held with the mouse for 1.2 seconds.

All buttons·sliders are `NoFocus`. If even one widget takes focus, the arrows move a slider instead of the ROV
and space presses the last button again — the pilot silently loses the keyboard. Auto-repeat is filtered out
at the window boundary (otherwise the "held" state flickers at the OS key-repeat rate). When the window loses
focus it loses the key-release events too, so that is treated as a **full stop**.

## Joystick

Reads `/dev/input/js*` **directly from the kernel** (no package needed, `joystick.py`). It sweeps the
8-byte events non-blocking, so it polls at 50 Hz from a GUI timer with no worker thread — there is
no way for a dead joystick to freeze the UI. If the joystick is unplugged, this is detected automatically,
and it looks for the joystick again every 2 seconds.

* **Always read and displayed, regardless of COMMAND ENABLE or the arm state.** That way the
  mapping can be checked in a state where nothing can move. Only transmission is gated.
* Stick and keyboard are **summed, then clamped**. Either one alone reaches full deflection.
* The teleop panel's `JOY` line shows **axis numbers and pressed button numbers live**. Set the
  mapping with `--js-axis-*` / `--js-btn-*` while watching it (the axes are usually auto-detected
  correctly, so normally there should be nothing to touch — see below).
* **An analog trigger rests at an end, not at the centre.** On this desktop's Xbox Wireless
  Controller, axes 4·5 output `-1.000` with no input — with that axis mapped to heave, **heave was
  stuck at +0.60** as soon as the GUI started (a command the pilot never gave). So the first value
  an axis reports becomes that axis's 0. But **only when it is not near the centre (|v| ≤ 0.5)** —
  a stick that really is centred keeps a real 0. Axes whose 0 point was set automatically are
  listed in the startup log.
* **Axis numbers change with the cable — so they are not hard-coded; the gamepad is asked.**
  The same Xbox gamepad numbers its axes differently depending on the driver:
  BT (hid-generic) is `0,1` left · `2,3` right · `4,5` triggers,
  **USB (xpad) is `0,1` left · `3,4` right · `2,5` triggers**.
  So the old fixed defaults (`yaw 2`, `heave -3`) were right only on wireless, and on 2026-08-17,
  when the gamepad was **plugged in by cable, the right stick's left-right (=yaw) drove heave and yaw
  died** (yaw was hooked to axis 2 = the left trigger, and since triggers get the automatic 0-point
  treatment, it stayed 0 forever).
  Now `JSIOCGAXMAP` **reads each axis's `ABS_*` code** and the right stick is found by name (if both
  `ABS_RX`·`ABS_RY` exist, that is the right stick and `Z/RZ` are the triggers; otherwise `Z/RZ` is
  the right stick). The startup log prints what it found:
  `joystick axes (read off the device): surge=-1, sway=0, heave=-4, yaw=3`.
  A number given to `--js-axis-*` wins (marked with `*` in the log).
  If the ioctl fails, it falls back to the old BT numbers — better than not being able to fly.
* The 0.08 deadzone is **rescaled**, so full deflection still reaches 1.0 — otherwise the pilot
  could never use the last 8% shown on the bar.
* **PROPULSION also shows the update rate.** The vehicle worker re-publishes the last value every
  100 ms whether or not a new `SERVO_OUTPUT_RAW` has arrived, and it used to stamp it with `now()`
  each time. The watchdog then fed on **our own heartbeat instead of vehicle data**, so even with
  the feed completely cut, the pill stayed green and the pre-arm 1500 µs stayed as it was — the
  panel said "not moving" while the motors were actually spinning (2026-08-07 hardware report).
  Now it stamps with the record's `t_host`, so silence ages properly, and the summary line shows
  `10 Hz` or `NOT UPDATING — reading is old`. 2 regression tests guard this.
* If you push the stick and the vehicle does not move, the bottom of the teleop panel says why:
  `TX 20 Hz  MOTORS DISARMED — nothing will move`. When the bars are full and transmission is
  normal but the vehicle sits still, this is the only line that tells you which of the two
  switches is off.

### Button mapping is the same as QGC — because there is only one map

**The gamepad's button bitmask is passed through to the vehicle** (`MANUAL_CONTROL.buttons`; can be
turned off with `--no-js-passthrough`). Then **which button does what is decided by the vehicle's
`BTNn_FUNCTION` parameters** — exactly the same contract as QGC. Functions that have no widget in
this station (mode_manual/depth_hold/stabilize, gain_inc/dec, trim, shift, input_hold_set,
arm/disarm) also work exactly as configured in QGC.

#### Kernel button number ≠ vehicle button number (not knowing this caused an accident)

Before passing them on, **the numbers are translated** (off with `--no-js-translate`). The kernel
numbers buttons in HID descriptor order, and **QGC, reading through SDL2, reorders them into a fixed
logical layout called `SDL_GameControllerButton`.** The vehicle's `BTNn_FUNCTION` is set in those
SDL numbers.

What happens if the kernel button numbers are passed through unchanged was confirmed on hardware on
2026-08-07: **the pilot pressed camera tilt (left bumper, kernel 6), the vehicle read
`BTN6_FUNCTION = arm`, and the motors came alive.** All 4 hardware observations are explained by
the table below.

| Physical | Kernel (BT) | Kernel (USB) | → Vehicle (SDL) | QGC function |
|---|---|---|---|---|
| A / B / X / Y | 0 / 1 / 3 / 4 | 0 / 1 / 2 / 3 | 0 / 1 / 2 / 3 | gripper open · manual · depth_hold · stabilize |
| LB / RB | 6 / 7 | 4 / 5 | **9 / 10** | mount_tilt_down / up |
| View / Menu | 10 / 11 | 6 / 7 | **4 / 6** | **disarm / arm** |
| Xbox | 12 | 8 | 5 | shift |
| Stick click L/R | 13 / 14 | 9 / 10 | 7 / 8 | mount_center / input_hold_set |
| **D-pad** | **axes 6,7 (hat)** | **axes 6,7 (hat)** | **11–14** | gain_inc/dec · lights dimmer/brighter |

**Kernel button numbers, like the axes, differ with the cable** — on BT 2/5/8/9 are empty, while
USB is packed 0..10 (both measured). So one fixed table looked up by gamepad name cannot cover both.
Now `JSIOCGBTNMAP` **reads each button's `BTN_*` code** and maps it to the SDL number
(`BTN_TL` → 9, `BTN_SELECT` → 4 …). Running the BT gamepad's btnmap through this rule reproduces the
table above, which was verified on hardware, **exactly, down to the blank cells** (a regression
test guards this). The name table remains only as the fallback for when the ioctl fails.

**The D-pad is not buttons but a hat (axes 6,7).** So pressing it showed nothing in the input
display, and nothing went to the vehicle either. SDL turns that hat into buttons 11–14, so we do
the same.

A gamepad with no map sends the kernel button numbers unchanged, and **the startup log warns.** A button that
is not in the map is **not sent at all** — guessing at an unknown number is pressing an unknown
function on the vehicle.

Check it on the teleop panel's `JOY` line. How to read it:

```
btn 3>2     left 3 = kernel button number, right 2 = vehicle button number. The kernel counts the X button as 3,
            the vehicle (SDL/QGC) as 2. The number on the right decides the function.
btn 1       no arrow means the two numbers are the same (B button: kernel 1 = vehicle 1).
btn 11      D-pad. In the kernel it is an axis, not a button, so there is no number to put on the left.
```

The point is that you must be able to check on this line before pressing a button that can bring the motors alive.

Previously the station even carried **its own second gamepad map** (`--js-btn-*`, defaults 0/3/4/5).
Now `--js-btn-*` only decides "which chip on screen lights up when that button is pressed", and its
defaults are **the same numbers** as `--btn-*` (= the vehicle's values). With passthrough on, the
chips only display and issue no commands — otherwise one press would go out twice (the lights
moving two steps at a time).

This vehicle's assignment (QGC > Joystick > Button Assignment):

| # | Function | Physical | # | Function | Physical |
|---|---|---|---|---|---|
| 0 | `servo_1_max_momentary` gripper open | A | 8 | `input_hold_set` | RS click |
| 1 | `mode_manual` | B | 9 | `mount_tilt_down` | LB |
| 2 | `mode_depth_hold` | X | 10 | `mount_tilt_up` | RB |
| 3 | `mode_stabilize` | Y | 11 | `gain_inc` | D-pad ↑ |
| 4 | `disarm` | View | 12 | `gain_dec` | D-pad ↓ |
| 5 | `shift` | Xbox | 13 | `lights1_dimmer` | D-pad ← |
| 6 | `arm` | Menu | 14 | `lights1_brighter` | D-pad → |
| 7 | `mount_center` | LS click | 15 | `servo_1_min_momentary` gripper close | **not on this gamepad** |

`MANUAL_CONTROL.buttons` is a uint16, so **buttons 16 and above cannot be carried** (if such a
button is pressed, it is ignored with a one-time warning).

#### If there is a function the gamepad cannot press: `--js-remap`

**Gripper close is 15, but 15 is SDL's `MISC1` ("Share"), and for this gamepad the kernel reports only
buttons 0–14** — i.e. there was no way at all to press it from the gamepad. (The UI's CLOSE button and
the `G` key work fine. We can set bit 15 ourselves; only the *gamepad* cannot press it.)

So **the number the gamepad sends is rewritten just before it is sent**:

```
--js-remap 11:0,12:15      (default)  D-pad ↑ → gripper open, ↓ → gripper close
```

**Only the remapped number** goes out, instead of the original (sending both would fire two
functions from one press). It is applied right after translation, so from then on there is only
one set of numbers — the bitmask, the chips and the `JOY` line all use the same numbers.

> **Do not change `--btn-gripper-*` instead.** That value records "where this vehicle's `BTNn_FUNCTION` is";
> it is not a preference. On 2026-08-07 it was changed to 11/12, and the UI's gripper
> button then pressed `gain_inc` instead of the gripper, leaving **neither the gamepad nor the UI working**.
> To change a gamepad button, use `--js-remap`; if the vehicle's assignment changed, use `--btn-*`.

Gamepad buttons **go through the COMMAND ENABLE gate just the same.** With the switch off, even the
gamepad's `arm` (Menu button) does not go out. But with the switch on, **the gamepad arms with a single
button press** — the on-screen ARM button is a 1.2 s hold-to-confirm, but the gamepad is not. That is
the same behaviour as QGC, so it was left that way.

## Re-flying a demo — replay (2026-08-30, unverified on hardware)

The sixth shape in the trajectory panel (HIGH combo label `Replay`, key `replay`). The vehicle
re-flies a pose trajectory extracted from one demonstration captured with the handheld, **taking
the pose it is in now as the origin**. It is the M0 gate on the way to diffusion policy: label
extraction → coordinate transform → feasibility → MPC tracking → (optional) gripper execution are
all verified with 1 demo, and it walks, ahead of time, **the same install path**
(plan_stream → `set_path_plan_ned`) that the policy will use later. Plan document:
`docs/DP_TRAJECTORY_PLAN.ko.md`.

```bash
# 1) extract pose labels from the demonstration session (the environment tags must be visible in the frames)
python -m umi_handheld.extract_pose data/<YYYYMMDD>/demonstration_00NN
# 2) replay at the station
./c3 gui --mpc --replay-session data/<YYYYMMDD>/demonstration_00NN
#    → pick Replay in the HIGH combo (before 2026-09-11: replay in the shape combo; the key is still replay) → START (hold)
```

Sequence of operation and rules:

* **Arm does not move the vehicle** — the same property as follow. The track is converted to
  coordinates relative to its own first pose and is anchored to the vehicle pose at the moment of
  START. No tag id goes in anywhere (localization itself is still done by the tags).
* **Speed is cut by time dilation** — if the demo is faster than `replay.v_max_m_s` (default
  0.12 [예측]), the geometry is left as is and only the clock is stretched by a factor of α. By what
  factor it was cut is kept in one log line and in meta (`plan_stream.run.time_dilation`).
* **PlanFilter is the last gate in front of the controller** (`control/plan_stream.py`, pure
  numpy): anchor/jump gates, kinematic limits, and `replay.workspace_box_ned` — now that the
  geofence has been removed, **this box is the only position-based protection**. Every verdict
  leaves its margin numbers in the run folder's `plans.jsonl` (for planner-vs-tracker
  attribution).
* **Two modes**: `replay.stream_period_s: 0` (default) installs the whole track as one plan (M0a);
  with `> 0` it is cut into `horizon_s` windows and streamed at that period (M0b) — a mode that
  tests, with real data, the same path as the stream the policy will later emit at 1 Hz.
  PlanStitcher joins the seams with commit-then-blend (cosine, `blend_s`), and beyond the last knot
  it is an **endpoint v=0 hold** (do not extrapolate a constant-velocity ray — the lesson of the
  path-mode corner accident).
* **The gripper is OFF by default** (`replay.gripper: false`). When on, the demo's gripper-width
  channel is replayed onto the momentary drive (-1/0/+1) through a threshold + hysteresis. This
  drive is a **level that latches** in the sink, so every path that ends the mission (STOP TRAJ /
  DISENG / E-STOP / re-arm) fires 0.0 to release the jaw. The hardware gripper has no feedback
  (open-loop), so neither the records nor the screen tell you "whether it grasped" — early
  hardware runs should verify only the trajectory and fly with the jaw off.
* **mpcc is refused** — contouring does not consume streamed plans (the same fail-closed rule as
  `follow_ok`). Fly with dobmpc/mpc/*_tuned/pid.
* **START also works under LOW None (teleop, 2026-09-11)** — the pilot flies the vehicle and the
  track is only drawn (only replay and Diffusion Policy can START under None). Such a run lands in
  `data/*/*_observe/` and its meta `controller.type` is `none` (the tree rule in the "LOW level
  None" section below) — do not pool it with replay tracking results.

Records boundary: meta `schema_version` **8** (replay landed 2026-08-30; 9 = fstereo block added,
10 = policy landed 2026-09-02 — a `policy` block is always attached, see the policy section below;
the schema ladder comment in `workers.py` is the original), `trajectory.kind: replay`,
`reference_clock.strategy: plan_stream_replay`, an always-recorded `plan_stream` block, three
columns `plan_id,ref_src,grip_cmd` at the end of the CSV (from schema 10 on, `grip_w_est,hold_frac`
are appended after them). **Never pool replay runs with geometric mission runs** — their references
come from a different family of sources. Offline verification: `rov_gui/tests/test_replay.py`
(+`test_plan_stream.py`); E2E is `demo_e2e.py dobmpc replay` (the real acados solver, passed
2026-08-30).

## The policy flies — policy (2026-09-02, unverified on hardware)

The seventh shape in the trajectory panel (HIGH combo label `Diffusion Policy`, key `policy`). The
UMI diffusion policy (`DiffusionTransformerTimmPolicy`, depth observation + relative pose → a
1.07 s action chunk of 16 knots; since 2026-09-07 knot = `[dx, dy, dz, dyaw, width]` 5-dim
`pos_yaw_width`, while the 2026-09-01 checkpoints' `[pos, rot6d, width]` 10-dim `pose10d` remains
only for reading old records — see the "5-dim action" boundary paragraph below) drives the vehicle
through **the same install path as replay** (`control/plan_stream.py` PlanFilter → PlanStitcher →
`set_path_plan_ned`). The worker is `rov_gui/backends/policy.py`, the network
`rov_gui/perception/dp_policy.py`, the depth observation `rov_gui/perception/policy_obs.py`, the
frame algebra `rov_gui/control/policy_frames.py`. The design spec is DP_LIVE_POLICY_SPEC(_V2).md in
the session scratchpad; the plan is `docs/DP_TRAJECTORY_PLAN.ko.md`.

```bash
# hardware (parity path): FoundationStereo depth + MPC + policy. All three flags are required.
./c3 gui --source hw --allow-command --mpc --fstereo --policy
#    → pick Diffusion Policy in the HIGH combo (before 2026-09-11: policy in the shape combo; the key is still policy)
#    → ckpt field: hw_mpc.yaml policy.ckpt is shown there as the seed. Pick another model with "…"
#      (the --policy-ckpt flag was removed 2026-09-11 — see the "The checkpoint is picked in the panel" section below)
#    → pick the follower in the LOW combo (PID / MPC / MPC_Tuned / DOBMPC / …; None = teleop, see the "LOW level None" section below)
#    → START (policy is a single click with no hold, 2026-09-02)
# offline dry-run (loads the real checkpoint onto the GPU, 40 s over a synthetic scene):
QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/rovgui-pose/bin/python \
    rov_gui/tools/policy_dryrun.py            # records to: data/YYYYMMDD/MMDD_HHMMSS_dryrun/
# wiring only, without torch (stub session):
QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/rovgui-pose/bin/python \
    rov_gui/tools/policy_dryrun.py --ckpt stub
```

**Read this first — the first full run is a pipeline smoke test, not a grasp test.** The
handheld's TCP sits 0.346 m forward / 22° down from its own camera axis [유도: dataset config
offset_m [0.0355, 0.1293, 0.3186]], so it is **visible in the frame**. This ROV's TCP is its own
jaw (`policy.tcp_body_flu_m: [0.502, 0, -0.17]` = `cam_t_flu` + lens→grip [0.196, 0, −0.275]; x
measured 2026-09-08 — 0.196 m in front of the lens in the frame where the bottle held in the jaw
was placed on the centre of floor tag 58, range 0.187–0.204 [측정: data/20260908/0908_180453_observe/policy_obs/rgb/000160.jpg
+ 0908_170428/policy_obs/rgb/000000.jpg(bottle height 0.084 (±0.006) m) + 0908_175151/policy_obs/rgb/000000.jpg(open jaw)];
z is CAD vertical [유도], y=0 [예측]), and it sits 0.338 m from the lens / about 11° below the
optical axis [유도], so even under the 43.3° tilt it is **visible in the frame** (the open jaw is at
rows 265–360). Runs before 2026-09-08 used the sim jaw 0.4165 (CAD chain), and that jaw projects
below the frame (row ~420) — records boundary (meta `policy.tcp_body_flu_m`). What remains not
measured is the jaw's y·z and the suspected x/z of `cam_t_flu` itself (KNOWN_ISSUES 2026-09-08);
if the jaw is measured directly in the C3 frame by stereo triangulation
(`measure_rig.py --pick-tcp`-style), it can be put into `tcp_offset_cam_m` so that the lens
position cancels out.

Behaviour and decisions (spec v2 A1–A24, outcome of the 5-lens review):

* **Clock (A1)**: `PolicyPlan.obs_t` is on the same monotonic `now()` clock as every `t_capture`.
  The **only** place that converts it to the mission clock is `MpcWorker._tick_policy_intake`, and
  for a live source, a plan without `obs_t` is rejected before the filter (`require_obs_t`) — this closed
  the replay-era trap of "if obs_t=None, skip the freshness gate".
* **TCP-space composition (A2)**: the action is composed in the TCP frame and only the body yaw is
  taken; roll/pitch are discarded (the NMPC levels). The magnitude of the discarded attitude is
  kept per plan as `dropped_rp_deg`. With the 5-dim action (2026-09-07) that drop **moved up into
  the training stage**: the label itself is `dyaw = yaw_of(R_bt·R_rel·R_btᵀ)` (the body yaw a level
  vehicle carrying the C3 mount must turn to reproduce the relative TCP rotation), so
  `compose_plan` only does `yaw_k = anchor yaw + dyaw_k`, `p_tcp_k = p_a + R_ned_tcp_a·dp_k`, and
  `dropped_rp_deg` is structurally 0 (comparing `dropped_rp_deg` between old and new runs is
  meaningless). At a level anchor the two decodes agree to 1e-14.
* **Anchor (A4)**: the design is `policy.anchor: leash` — measured pose + reference offset,
  clipped to `anchor_leash_m` 0.05 m [예측]. A measured anchor resets the tracking error at every
  replan, and a pure reference anchor moves away from the vehicle. **From 2026-09-07, both configs
  set to `measured` on a trial basis** (operator decision): since each plan starts at the
  vehicle's obs_t position, the red knot-0 dot sits on the hull; the cost is that at every replan
  the reference jumps back by the amount of lag · the low-level loop cannot build up the error
  needed to punch through the deadband · steady disturbances seep into the reference (the config
  comment above). If it does not work well, revert to `leash`. Records boundary: tell apart by
  meta `policy.anchor_mode` and do not pool. **2026-09-14: back to leash** (measured wrote off the
  error every 0.5 s, so the hull could not get past the ESC deadband, `0914_172420`).
* **Yaw reference filter (2026-09-14, `policy.yaw_ref_filter`)**: why the yaw wobble did not go
  away however the anchor was chosen came to light from giving yaw steps of 10~26° in station
  without the policy (`data/20260914/0914_190226/mpc_190335`·`mpc_190532`): the NMPC yaw loop is
  **stable but underdamped** (overshoot 45/18 %, ringing period ~1.4–2 s, settling after one or two
  cycles, quiet tail) [측정]. In policy runs, the plans arriving every 0.5 s injected components
  near that period into the yaw reference — with yaw leash 0 it was re-pinned to the obs_t (0.5 s
  stale) measured yaw, copying the hull's own wobble with a 0.8 s delay (`0914_181425` run 2 was in
  antiphase); with leash 10 the policy's dyaw, jumping back and forth, passed straight through.
  So now the yaw knots made by `compose_plan` are swapped out by `policy_frames.filter_plan_yaw`:
  it **starts from the reference yaw that was being flown at plan t0** (not the measured yaw),
  turns at the chunk's requested mean turn rate low-passed with `tau_s` (2 s) (`rate_deg_s` 3°/s
  cap), and cannot go outside `guard_deg` (15°) of the measured yaw (the only point where the
  measurement enters). The body knots are re-derived from the TCP knots with the filtered yaw, so
  **the TCP goes where the policy asked and only the heading is filtered**. If the same turn keeps
  being requested, it reaches that turn rate (at most the cap) after ~tau; if the sign flips from
  plan to plan, it is pressed down to around a mean of 0. The turn-rate state lives in
  `rp["yaw_filt"]` (reset at every ARM) and advances **at intake time** (even if a gate rejects the
  plan later — the next plan continues from the reference being flown anyway). With `null`, the
  old behaviour (yaw_k = anchor yaw + dyaw_k). Records boundary: meta
  `policy.config.yaw_ref_filter`·`run.yaw_ref_filter` (plan count, last turn rate, guard/rate fire
  counts), and per plan record `yaw_ref_filter` (requested/filtered dyaw, the two turn rates, start
  yaw, guard clamp, body displacement). At every ARM the log prints `YAW REF FILTER …`. On the same
  day the plant yaw damping was also changed, `hw_mpc.yaml plant.linear_damping[5]` 0.07 → 4.0
  [유도: open-loop reconstruction from the same step data; the shipped model swings ±60–75° for the
  same command]. **Both unverified on hardware** — before flying the policy, first check with the
  same yaw-step whether the overshoot went down.
* **Observation viewpoint shift (2026-09-30, `policy.obs_view_forward_m`, temporary, default
  `null`, unverified in water)**: the depth obs the policy sees is redrawn from a virtual viewpoint
  this distance ahead of the C3 along body x (the gripper axis) (`perception/policy_obs.py`
  `_shift_view`: back-project with the training rays → translate → reproject; each sample writes
  2×2 pixels with the nearer point winning, and holes are filled with the farther of the nearest
  measured values above/below/left/right). **Only the observation** changes; the action transform
  (`tcp_body_flu_m`) stays as it is. A temporary workaround for the peg checkpoint reading the peg held in the
  jaw as "after release" and retreating, because the ROV jaw sits about 5 cm further forward along
  the axis than the handheld in the peg demos (peg root 0.254 vs 0.208 m [측정:
  `data/20260930/0930_183320_observe/diag/rig_geometry.json`]). At 0.05 m the dz<0 fraction goes
  0.90 → 0.31, and 0.18 if the proprio width 0.007 m is given as well [예측:
  `.../diag/validate_builder_shift.json`, 322 plans of obs reconstructed from the UI recording].
  Build 0.4 → 14.7 ms/frame [측정: same file] — two frames per plan, so the obs age grows by about
  29 ms [유도], and late plans increase by as much. **It is a process-wide switch, so it also
  applies to a can checkpoint picked in the panel — turn it on only for peg sessions and set it
  back to `null` when done.** Records boundary: at every launch and ARM the log prints
  `OBS VIEW SHIFT …` (also in events.log), meta
  `policy.config.obs_view_forward_m`·`policy.worker.obs.view_shift`. `depth_valid` in plans.jsonl
  is the value after filling, so in shifted runs it is stamped 1.0.
* **0.2 s grid (A5)**: the 16 original points at 66.7 ms are window-averaged into 6 on the
  `knot_dt_s` 0.2 s grid (knot 0 is fixed). The gain of the finite-difference acceleration gate
  goes 551·σ → 61·σ [유도]. `jitter_rms_mm` is recorded.
* **proprio is fix-based (A6)**: a row is stacked only when PolicyState's `t_fix` advances
  (stacking at every 20 Hz tick would quantize the 66.7 ms motion cue into {0,1,2} fixes). If the
  two observation rows collapse onto the same fix, **inference is skipped** (`skip_degenerate`).
* **epoch (A8)**: `epoch` goes up at every ENGAGE·ARM and the worker clears its history; plans
  from a previous epoch are dropped by the consumer. Plans that arrive between STOP→START are not
  installed.
* **Nothing is taken in during bridge/DR (A9)**: while the tag dropout ladder is active or the
  vehicle is flying on DR, intake is skipped (`skip_bridge`), and once `stale_s` is exceeded the
  mission ends. The worker also does not run inference unless the latest row is a fresh fix.
* **hold tail = cost mask (A10)**: stages beyond the end of the plan get position/linear-velocity
  weight 0 (`policy.hold_tail: mask`; heading·leveling are kept). Tracking a v=0 hold would hit
  the brakes 0.3–0.4 s before the end of every plan [유도].
* **blend gate (A11)** + yaw-jump gate (8° [유도]), **stale status (A13)**: PolicyStatus carries a
  stamp, and after 2 s of silence START is refused.
* **depth grid (A16)**: `--fstereo`'s `depth_native` (host rectified-left 640×400) is unrectified
  onto the training target grid (raw CAM_B 640×400, `configs/target_camera_underwater.yaml`) to
  reproduce the training recipe (centre crop 400 → 224 nearest → inverse-depth u8 + validity)
  **bit for bit**. If the live K/D differ from the yaml, it rejects, naming the recalibration, and
  if the 224 observation coverage is below `min_obs_coverage` 0.985 [유도: 0.985 sits at the 0.06 % tail of the training obs validity (per-frame min 0.9777, p1 1.000, 99.94 % of frames ≥ 0.985) [측정: rov_gui/tools/dp_policy_out/training_validity_20260902.txt]]
  it rejects — so under `--policy`, `--fstereo-alpha` moves to 0.5 (typing a value below 0.5 makes
  `check_policy` warn), and a rejected grid shows up as is in the controller's arm refusal, as
  `PolicyStatus.grid_ok=False` / `grid_why` (previously it armed with ready=True and plans never
  came — verified 2026-09-02). **Device depth is refused by default**: `--source hw --policy`
  without `--fstereo` is a **launch refusal** (decided before whether `--mpc` is present or not),
  opened only by `--policy-allow-device-depth`, and even then a "bench experiment, recorded in
  meta" warning is raised (the CAM_A-aligned 63.7° field of view covers only the part that falls
  inside the policy's CAM_B 85.6° [유도] + the ×0.64 matcher). `--policy` without `--mpc` is not
  refused but only gets a **warning** (the worker produces plans nobody consumes).
* **FS instrument parity (A17)**: the training depth store is iters 16 / scale 1.0 [측정: `~/Desktop/data
  collection/depth/0/depth.zarr/.zattrs`]. `--policy`'s FS defaults **deliberately** depart from it:
  iters 8 / scale 0.75 / alpha 0.5 (`POLICY_DEPENDENT_DEFAULTS`, 2026-09-02~, a speed trade-off —
  `FS_RATE_NOTE`); instrument parity is `--fstereo-iters 16 --fstereo-scale 1.0`. `check_policy`
  reads `.hydra/config.yaml` next to the checkpoint → the dataset zip's
  `.zattrs['depth_source']` without torch and compares them, and if they differ it **warns** (until
  2026-09-02 it was a launch refusal — the call-site comment in `check_policy`; now
  `--policy-allow-fs-mismatch` is only recorded in meta). What is compared is not the typed
  `--fstereo-ckpt` but the checkpoint **that will actually be loaded** (defaults included), and
  since 2026-09-30 the comparison is **by content, not by path** (`same_fstereo_ckpt`: if both
  exist, the sha1 of the first 8 MiB of each; if the store-side file has been deleted, the
  remaining file against the upstream public release's sha1 `fstereo.UPSTREAM_CKPT_SHA1`) — the
  stores record the deleted `~/Desktop/FoundationStereo/…` path or a `data collection` copy path,
  so with a path comparison the same weights read as a different model once the checkout has
  moved. And **in the 2026-09-14~09-30 runs this check did not run at all**:
  `data/checkpoints/*.ckpt` are symlinks into run folders, and `hydra_config_for` did not follow
  the link, so meta `policy.worker.fstereo.training.status` is `no_hydra_config` in all 178
  non-stub records since 09-14 (245 `ok` before that) [측정: `data/2026*/*/*.meta.json`,
  2026-09-30 tally]. Since 09-30 the link target is resolved first — the iters/scale mismatch
  **warning** reappearing on a default launch is due to this fix and is an intended trade-off
  (`FS_RATE_NOTE`). With the parity setting (16 / 1.0), FoundationStereo runs at 7.26 Hz, solve
  137 ms, panel latency 243 ms [측정: rov_gui/tools/fstereo_bench_out/policy_bench_20260902_141224.json, C3 hardware, alpha 0.5, scale 1.0, iters 16, 330 frames, GPU shared with a concurrent test run] — so
  the two depth images of one observation are 138 ms apart in frame spacing = **2.07×** the
  training stride of 66.7 ms (recorded per plan as `obs_pair_dt_s`). The default (8 / 0.75)
  accepts the instrument mismatch in exchange for ~13 Hz [측정:
  `rov_gui/tools/fstereo_bench_out/sweep.txt`, 74.9 ms eager — `FS_RATE_NOTE`].
* **depth pairing (A18)**: the latest frame + the frame closest to `t_d − obs_dt` (±0.6·obs_dt);
  if there is none, an older frame within **3.0·obs_dt** (200 ms; with 2.5 the margin at the 138 ms
  spacing above was only 29 ms, so a single jitter meant a skip) (the spacing honestly recorded as
  `obs_pair_dt_s`); if there is none of those either, skip — except that the first inference after
  START may duplicate the latest frame (training also pads only the episode's first frame).
  **The two proprio rows are taken at the same spacing as the depth pair** (previously 138 ms images
  went with 66.7 ms proprio, so image motion was 2× the proprio's — a combination absent from
  training). If the depth stamp is ahead of the latest fix by 0.5·obs_dt or more, `skip_fix_lag`
  (skip, instead of the rows being clamped to the fix and the motion cue silently shrinking);
  `obs_rows_t`/`obs_fix_t` in the records are the times of the rows actually used. An inference
  attempt needs a frame that **arrived** since the last attempt. But if even one frame comes in
  during the period (0.5 s), that mark stays set, so in practice it fires on the first tick (20 ms)
  after the period ends with **the latest frame already in the ring**, and it waits for an
  arrival only when not a single frame came in meanwhile (corrected 2026-10-01 — the earlier doc
  said "fires on arrival". The meta `pairing.trigger` string is unchanged for record compatibility,
  and `pairing.trigger_detail` describes the actual behaviour; per plan,
  `trigger_age_s − depth_ready_age_s` is how long that frame waited in the ring). A skip does not
  burn the period; it retries on the next frame. The intake freshness
  `policy.obs_max_age_s` is 0.6 s [유도: measured depth chain 243 ms + inference 33 ms +
  tick 25 ms ≈ 0.30–0.46 s at intake [측정: policy_bench_20260902_141224.json,
  dryrun_20260902_140513]; plan life at install ≥ 0.4 s ≥ blend_s]. That derivation's "inference
  33 ms" was wrong: the policy forward alone is 19.3 ms, but on hardware, sharing the GPU with
  FoundationStereo, it was 207–273 ms p50 [측정: data/20260930/0930_220212/diag/
  policy_vs_fstereo_contention.json, plan_timing.json] — see the `--policy-fs-schedule` section
  below. In the working tree, `config/hw_mpc.yaml` has 0.7 (operator, 2026-09-30).
* **START without a hold (2026-09-02)**: a single click, for `policy` (`Diffusion Policy` in the
  HIGH combo) only. The other shapes keep the 1 s hold (`test_offline` pins this in both
  directions).
* **gripper (A12)**: **ON** since 2026-09-07 (`policy.gripper: true`; runs before that only
  computed it and threw it away — records boundary). In the first hardware approach (0907_145206)
  the policy emitted close (g<0.30) in 53/161 plans, but 0 commands went out [측정: data/20260907/0907_145206/
  plans.jsonl, mpc_145239.csv grip_cmd]. **Trap — sample time**: the stitcher reads `g` only at
  "now" (`ts[0]`), but the policy's intent to close sits at the end of every chunk (knot 4–5/6),
  and knot 0 follows the estimated width the policy observes (which stays open as long as there
  is no command), so with plans replaced every 0.45 s the "now" value drops below the threshold
  only when a plan handover is late (7–59 % of the ticks after 32 s in that run) — a deadlock in
  which the policy and the station wait on each other, released once the first CLOSE edge lowers
  the estimated width. Sampling `t+la` with `gripper_lookahead_s` (default 0.0 = old behaviour)
  puts 70–100 % of the ticks after 32 s in the same run below the threshold at 0.6 s [유도: offline
  replay of plans.jsonl]. However, the policy's intent to close comes "~1 s before the final pose"
  at demo speed (≈0.3 m out), so on a vehicle 5–10× slower it may close well before arriving —
  see KNOWN_ISSUES. The CSV `grip_g` column (schema 13) is the value the hysteresis saw on that
  tick. The width estimate is an open-loop integrator (`GripperWidthEstimator`, integrating all
  pilot and policy drives over the process lifetime, `cmd_gripper_drive` →
  `MpcWorker.on_gripper_drive`; the gripper buttons of the gamepad passthrough go only to the
  vehicle, so `bus.jaw_drive_seen` reports the same level to the estimator alone, 2026-10-01).
  **At run START the jaw state is not filled in from the estimate; it is "neutral"** — the
  policy's first open/close threshold crossing goes out as a drive whatever the estimate says
  (2026-10-01: an accident in which the estimate, stuck at "open", swallowed the open requests of
  three runs, `data/20261001/1001_151431`). The levels are open 0.069 / closed 0.042 m
  [측정: /home/bdml/Desktop/data collection/slam/actions_summary.json reports[*].gripper_command — open_level_m p10–p90 0.055–0.070 (median 0.061), closed_level_m p10–p90 0.034–0.043 (median 0.040); 0.069/0.042 chosen near the upper tail [예측]].
* **Safety defaults (A21)**: v_max 0.08 [유도], a_max 0.20, r_max 0.50, anchor_max 0.15,
  jump_max 0.06, div_max 0.25 (the policy's own divergence limit), workspace box
  [[-1,-1,-0.5],[1,1,0.5]] **required** (refused if null — unlike replay), max_run_s 500
  [decision: operator 2026-09-11 — 120 → 500 s; not a measurement. `path_timeout_s` = max_run_s ×
  `traj_timeout_factor` (3.0) also becomes 1500 s along with it, while `observe_max_run_s` 1800 is
  unchanged; `test_policy_ckpt_paths.py` pins that the four definition sites —
  hw_mpc.yaml·land_dp.yaml·`geometry.default_policy_block`·`backends/policy._FALLBACK_POLICY_BLOCK` —
  hold this one value],
  clip_ratio_max 3.0 [예측: 3.0 chosen from the gate — GT chunks need p50 1.50 / p90 3.44 at these caps on the 0.2 s grid [측정: rov_gui/tools/dp_policy_out/gate_20260902_clip3.log]].
  Every number in the `config/hw_mpc.yaml policy:` block carries a provenance tag.

Records boundary: meta `schema_version` **10**, `trajectory.kind: policy`,
`reference_clock.strategy: plan_stream_policy`, an always-recorded `policy` block (config +
provenance, TCP geometry, worker meta — session describe / obs recipe / pairing·skip counters /
FS settings compared with the training settings / `synthetic`), `kind`·`obs_t_rel`·`age_at_intake_s`·
`action_raw`·`anchor_*`·`dropped_rp_deg`·`jitter_rms_mm`·`obs_dt_s`·`obs_dt_cfg_s` in `plans.jsonl`,
and `grip_w_est,hold_frac` at the end of the CSV. `hold_frac` is a **recorded value**: the mask is passed
down to ctrl only while the plan is alive (`t_rel < t_end`) and not latched, so `1.000` means "the whole
horizon is past the plan = tracking the end-point hold at **full weight**", not a horizon without
weights (fixed during verification 2026-09-02; before that, same-day dry-run meta masked the whole
horizon after the plan expired). Runs with `synthetic: true` (demo source or stub session): **do not
cite as measurements**. **Never pool** policy runs with replay or geometric-mission runs.

**Why the MPC follower did not follow the policy plans — the hold-tail mask (2026-09-07)**: flying policy
with mpc/mpc_tuned left the vehicle in place (net displacement 0.05–0.4 m/50–90 s; PID 0.2–0.9 m). The
reference sat consistently 7.5–10 cm ahead of the body, with yaw error ~1° and sign flips 0–0.4/s, so it
was **not wobble but not pushing**. The #1 cause is `hold_tail: mask`: a plan (1 s) is
already ~0.5 s old when it arrives, so only 0.25–0.4 s of the 3 s NMPC horizon keeps any position weight
(`hold_frac` p50 0.87–0.92), and under that cost the NMPC's optimal surge is 0.05–0.2 N on a fresh plan
and 0.6–1.9 N during a blend — below the T200 deadband (axis ≈0.096 ≈ 5.8 N; the knee in this record is
axis 0.05–0.07 ≈ 3–4 N), so no thrust comes out. **In the same runs, the ticks where the plan had expired
and the mask was released (`ref_src hold`) put out 3.3–6.9 N and moved** (PID ~6.5 N regardless of
ref_src) [측정: data/20260907/0907_164659/
mpc_16{4716,4835,5209}.csv, 0907_152730/mpc_153134.csv vs 0907_145206/mpc_145239.csv; ref_src×|uX|].
Secondary causes: the 0.05 m leash saturates on 95–99 % of replans, so the error cannot grow;
mpc/mpc_tuned have no integral or disturbance term (EAOB unused, intended); `mpc_tuned` is softer still
along the direction of travel with `along_scale 0.25`. Mitigation (in single-variable A/B order):
① `policy.hold_tail: track` (default since 2026-09-07; no mask, the horizon past the plan tracks the
end-point hold — the behaviour already observed on hold ticks), ② if still below the deadband,
`policy.q_scale` (NMPC position weights ×s during policy plans, `HwDobMpc.set_plan_cost_scale` runtime
cost_set, restored to 1.0 on STOP or a mode switch, recorded in meta `trajectory.q_scale`/`plan_q_scale`;
default 1.0, stiffness ~√q [유도]), ③ raise `anchor_leash_m`. Why the deadband is not compensated in the
axis domain: see the THRUSTER DEADBAND comment in hw_mpc.yaml. Both unverified on hardware.

**With `track` it did not move either — extrapolating the hold tail and reverting the along weight for
policy missions (2026-09-08)**: `0908_165517`, which flew mpc_tuned (MANUAL·STAB, 3 missions) and PID
(STAB) in alternation on the same scene: PID moved 0.73 m in 40 s (uX p50 3.6 / p90 8.9 N, 30 % of ticks
above the 0.096 knee), mpc_tuned 0.01 m in 45 s (uX p10–p90 2.5–3.5 N **pinned**, 0 % above, PWM deviation
of the 8 thrusters p50 ≤ 36 µs) [측정: `data/20260908/0908_165517/mpc_165753.csv` vs `mpc_165700.csv`,
`diag/cmp4.py`]. Three mechanisms overlap. ① `anchor: measured` pulls the reference back to the measured
pose every 0.5 s, so the tracking error cannot exceed "one plan's displacement" (3–9 cm); ② with
`hold_tail: track`, a 1 s plan that arrives 0.5 s old leaves 87 % of the 3 s horizon as "stopped at the
end point" (hold_frac p50 0.87), so the velocity feedforward that the drag compensation (86.7 N·s/m)
would multiply is almost 0; and ③ mpc_tuned's `along_scale 0.25` makes the stiffness along the direction
of travel √(75/0.05) ≈ 39 N/m. Result: uX ≈ 40 N/m × 0.086 m = 3.1 N, and the LQ optimum is "don't push,
fall behind" (a horizon reconstruction reproduces CSV rx/ry to 0.0 mm) [측정: `diag/fwd_A_out.txt`].
PID moved not because of stiffness but because of **pulses**: kd·v_ref on the stage-0 velocity
(59.7 N·s/m × 0.10–0.14 = 6–8 N) + the integrator saturated at 4 N produce knee-exceeding bursts at
73 times/min, phase-locked to the 0.55 s plan period, and 89 % of the displacement lies inside those
windows. PID too sat stalled in exactly the same way at 1.5–2 N for the first 10 s (the stretch where the
policy was putting out 0.029 m/s plans). The three mpc_tuned missions have different plan speeds
(0.086 / 0.048 / 0.037 m/s), so they must not be pooled as a same-plan comparison.

Mitigation (2026-09-08, code + config, **unverified on hardware**): (1) **`policy.hold_tail: extrapolate`** —
horizon stages beyond the last knot of the latest plan keep walking at that plan's **last-segment
velocity** (1.44× the plan mean; a chunk that ends decelerating extrapolates gently, a chunk that has
stopped is not extrapolated). Clipped by `v_max_m_s`, only up to `hold_tail_extrap_max_m` (0.3 m [예측]),
inside the work box, behind the same live-only gate as the mask (an expired stream holds the end point).
The stitcher itself, `_replay_ref_now`, the divergence guard and the jaw sample are unchanged
(`MpcWorker._extrapolate_tail`, `PlanStitcher.end_velocity`). (2) **`policy.along_scale: 1.0`** — returns
the path-frame follower's along weight to the baseline (300) only while a policy plan is installed
(`HwDobMpc.set_plan_along_scale`, same lifecycle as q_scale: applied at START, restored on STOP·mode
switch·disengage·reset; meta `controller.plan_along_scale`/`plan_along_scale_flown`, run `extrap_ticks`).
`q_scale 4` gives the same surge too, but it scales x·y·z all by 4×, so sway 1200→4800·heave
150→600 fight the STAB attitude loop; hence this knob, which touches only surge, was chosen. The evidence
is an offline closed-loop A/B that ran the actual acados solver, a plant with the measured deadband law
and the 88 plans of that run [측정: `diag/fwd_C_out.txt`, `fwd_C_harness.py`; reproduction: the flown
setting gives 0.000 m·uX 3.24 N vs the recorded 0.004 m·3.18 N]:

| Setting (anchor measured, plan 0.085 m/s) | Travel in 45 s | Mean speed | Error p50 | Sign flips/s |
|---|---|---|---|---|
| track, along 0.25, q_scale 1 / 4 / 8 | 0.00 / 0.24 / 0.77 m | 0.000 / 0.005 / 0.017 m/s | 6.7 cm | — |
| extrapolate, along 0.25, q 1 | 1.01 m | 0.022 | 4.4 cm | 0.16 |
| **extrapolate, along 1.0, q 1 (deployed)** | **2.47 m** | **0.055 (64 %)** | **1.3 cm** | **0.16** |
| extrapolate, along 0.25, q 4 | 2.41 m | 0.053 | 1.4 cm | 0.09 |
| PID (same plant) | 1.21 m | 0.027 | 3.9 cm | 2.26 |

Limits: on a 0.04 m/s plan the feedforward is only 2.5 N, so the same setting only reaches
0.008–0.02 m/s (the structure in which the error has to grow before the knee is crossed remains), and if
the effective knee is 0.13 rather than 0.096 (25 µs per thruster, ~190 µs/axis), the 0.085 plan still
holds 0.042 m/s but the 0.04 plan drops to ~0. The next lever in that case is **`anchor: leash` +
`anchor_leash_m 0.10`** (allows the error up to 10 cm; harness 0.076 / 0.036 m/s, and 0.075 / 0.035 even
at knee 0.13) — but it reverses the 2026-09-07 `measured` decision (the knot0 point leaves the hull; plans
are composed 10 cm ahead of the pose the policy saw), so it is a matter for operator decision [측정: `diag/fwd_R_leash_knee_out.txt`].
In STAB, heading hold pre-loads T1/T3 at +25/+36 µs and T2/T4 have the opposite sign, so below surge axis
~0.16 only one diagonal pair pushes and heading hold has to absorb that yaw torque (not in the 1-D
harness; the effective threshold in this run appears to be ≥36 µs, as T3 sat at 32–36 µs for 39 s
without moving). Reverting is two lines: `hold_tail: track` + `along_scale: null`. Tests: test_policy
63/63 (3 added: extrapolation, along lifecycle, verification), test_path_cost 28/28 (bridge along hook),
test_control 106/106, test_plan_stream 26/26, test_policy_frames 22/22, `demo_e2e.py mpc_tuned policy`
OK (stub policy, 22 plans; the "own → 1 → own (restored)" log confirmed).

**Why the MPC follower rose under the policy's descending plans — the plant buoyancy constant
(2026-09-07)**: the 5 mpc_tuned policy missions of 0907_180038 rose 0.27–0.54 m even though the policy
was requesting +0.055 m per 1 s (p50, NED down). Regressing the heave command gives
`uZ ≈ −5.6 N − 60 N/m·ez` in all five missions (R² 0.54–0.94): the **intercept −5.6 N (up)** is
`dobmpc/params.py NET_BUOYANCY = −5.71 N` (a CAD·vendor composite value saying heavy_gripper sinks;
0 measurements) — the nominal NMPC cancels its own model's gravity, so at error 0 it pushes 5.7 N
**upward**, and adding the 5.8 N deadband, downward thrust only starts at ez ≈ −0.19 m, while the 0.05 m
leash plus the plan's descent step bind the error to −0.10…−0.15 m → thrust 0 (|ax_heave| < 0.096 on
88–100 % of ticks), and the vehicle rises by as much as the hardware buoyancy. The hardware B−W swung
between −2…>+3 N that same day and was never −5.7 N [측정: data/20260907/0907_180038/mpc_18*.csv,
plans.jsonl; 0907_16*/mpc_*.csv]. Mitigation: `hw_mpc.yaml`/`land_dp.yaml` **`vehicle_net_buoyancy_n`**
(hardware B−W, + = buoyant, default 0.0, |v| ≤ 15; mpcc ignores it and says so in meta) →
`HwDobMpc.heave_trim_wrench` puts (model − hardware) into the acados parameter p (= w) as a **constant
heave wrench** (runtime input, no rebuild, not an observer; **mpc/mpc_tuned only** — not applied to
dobmpc, because its EAOB estimates the same residual; `reset()` does not clear it, since it is a config
constant). The records boundary is meta `controller.heave_trim` (runs without it = runs flown with the
−5.7 N upward FF); CSV `w0..w5` remain the EAOB columns. Model-plant prediction: holding a neutral
vehicle, it comes to rest +12.2 cm high uncorrected → 0.00 cm corrected [예측: `rov_gui/tests/test_path_cost.py::test_a_neutral_vehicle_holds_depth_only_with_the_trim`,
plant without a deadband]. Being a constant, it cannot follow the hardware buoyancy, which moves ±8 N
within a day, so measure the value each session (the steady `uZ` of a PID depth hold, if it is outside
the deadband), and the residual error of ~0.1 m inside the deadband shrinks only with `policy.q_scale`
[유도]. The 1st-pass `hold_tail: track` A/B in the same run: uX p50 0.4–0.6 → 2.5–3.6 N, but still below
the deadband (>0.096 on 0–2 % of ticks) → next lever `q_scale` 2–3. **Unverified on hardware**. Tests:
test_path_cost 21/21.

**Log tag `ctrl[<mode>]:` (2026-09-07)**: mission log lines from the low-level control worker
(`MpcWorker`) now carry the **name of the follower actually selected**, as in `ctrl[pid]: …` /
`ctrl[dobmpc]: …` (attached in one place, `MpcWorker._log`). Before that they were fixed to `mpc:`
regardless of mode, so "mpc: policy plan late" appeared even while flying on PID and confused the pilot.
`station_bridge:`, `imu_dr:` and the solver's own logs are unchanged.

**Low-level controller = the LOW combo (confirmed 2026-09-07; named LOW since 2026-09-11, before that the
"mode combo")**: the follower of a policy mission is the controller chosen in the trajectory panel's LOW
combo (`mode_requested` → `bus.cmd_mpc_mode` → `MpcWorker.set_mode` → `self.ctrl`; the combo shows the
labels `PID | MPC | MPC_Tuned | DOBMPC | DOBMPC_Tuned |
MPCC | DOBMPCC` and sends the keys `pid`/`mpc`/… — `currentData()` is the key, the text is for display).
The stream hands `self.ctrl.path_plan_steps` worth of NedPlan to `set_path_plan_ned`, so pid (1 stage),
mpc/dobmpc and *_tuned (N+1 stages, path-frame cost via `psi_path`) all follow the same plan;
mpcc/dobmpcc contour their own path and ignore the stream, so arm is refused (left as is). **`None` is
not a follower** — it is teleop, and the policy only infers and draws (the "LOW level None" section
below). The mode is chosen **before START** — while engaged the combo is disabled and `set_mode`
refuses; since 2026-09-11 it also refuses while a controller CSV is open via REC (because the open CSV
pins the run folder to the old tree; `MpcStatus.csv_open` locks the combo), and if the worker is still
building it refuses with "controller still building" (previously silently ignored). For a refused or
dropped request, the combo **reverts** to the worker's actual mode on the next MpcStatus (honesty rule:
the combo shows the worker, not the request). Hardware evidence: of the 24 policy missions on
2026-09-07, 23 flew on `pid` and 1 (0907_152730/mpc_153134) on `mpc_tuned` (the file default is
dobmpc) [측정: data/20260907/*/mpc_*.csv `mode` column]. Offline:
`demo_e2e.py {pid|mpc|dobmpc|mpc_tuned|dobmpc_tuned} policy` all OK, and the 4
`test_policy_follows_on_the_selected_*` tests in `test_policy.py` pin it.

**5-dim action boundary (2026-09-07)**: meta `schema_version` **12**. The checkpoint contract declares
`action_repr` (`pos_yaw_width` = `[dx, dy, dz, dyaw, width]`, training repo
`umi/common/yaw_action.py`; `pose10d` = the old 10-dim), and it is carried as-is into the worker status,
every line of plans.jsonl, the run counters and meta `policy.action_repr`. The representation the station
flies is a single one, pinned at ARM for each mission (config `policy.action_repr`; since 2026-10-03 the
deployed value is `auto` = the contract of the checkpoint held at that moment — the 6-DoF section below),
and **both the arm gate and the intake gate check it**: a `policy.ckpt` pointing at an old 10-dim ckpt is
not silently decoded but ends in an arm refusal (`_policy_refusal` "action_repr mismatch") / a plan
rejection (`reject_action_repr`, no strike). The mount constant is cross-checked too: if ckpt
`shape_meta.action.yaw_axis_R_frd_cam` does not equal the C3 rotation in `hw_nav.yaml` (`cam_tilt_deg`
43.3) within 1e-6, arm is refused with `PolicyStatus.mount_ok=false` (because the yaw axis of the labels
was defined by that mount). From then on `plans.jsonl action_raw` rows have **5 columns** (10 columns
before); `render_run_scene` recomputes both generations by column width (it branches on width, not on the
meta version). The contract's `action_range` (normalizer input_stats) is the actual upper bound of the network
output, so the `|dyaw| ≤ π` gate never fires. Old (16,10) runs and new (16,5) runs are **not the same
policy family** — do not pool them. The checkpoint path is derived from a single place,
`rov_gui/state.py POLICY_CKPT_DEFAULT`, and `rov_gui/tests/test_policy_ckpt_paths.py` pins its agreement
with the two YAML locations.

**6-DoF variant — roll/pitch tracking `pos_rpy_width` (2026-09-26, 0 water runs)**: meta `schema_version` **16**.
The full design and the first-flight order are in `docs/DP_6DOF_PLAN.ko.md` (for operators); the list of
unresolved items is in the KNOWN_ISSUES entry of the same date. Summary:

* **`policy.action_repr: auto` (2026-10-03, operator request "if I set the checkpoint to 7d, set things up
  for 7d on its own")** — the deployed value (hw_mpc.yaml·land_dp.yaml). The contract of the checkpoint the
  policy worker **is holding** at START (`PolicyStatus.action_repr`, from `shape_meta.action.action_repr`
  or inferred from the width) is fixed at ARM as that mission's pin — switching between 5-dim/7-dim in the
  panel needs no YAML edit or restart. If there is no contract (`""`) or it is `pose10d`, it is
  **refused** (not replaced by the default); 7-dim still requires the mount keys. The pin is decided only
  once, from **the very `PolicyStatus` object** the refusal check saw (if another object slips in between,
  ARM is refused), and intake and meta do not re-interpret it afterwards. The explicit values
  `pos_yaw_width`/`pos_rpy_width` pin the session byte-identically to before (a different contract gives
  "action_repr mismatch"). The code default (no key) is still `pos_yaw_width`. LOW does not change
  automatically — whether it tracks follows the truth table below unchanged. **Caution**: under auto + the
  deployed veto `attitude_track: true`, picking 7-dim and picking LOW `RL_PWM_7D` makes a single START
  **track** roll/pitch (no YAML step); a new 7-dim network goes on a 4-DoF LOW first (dropped-and-logged).
  The panel ckpt field shows `READY · 5-dim | 7-dim` (red `· not flyable` / `· ≠ config X` /
  `· MOUNT MISMATCH` if START would refuse the checkpoint), the mission line shows before START what the
  current LOW will do with roll/pitch (`TRACKED on …` in the warning colour /
  `DROPPED-AND-LOGGED (… is 4-DoF; …_7D tracks)` / `holds roll/pitch level`), and the chip after ARM shows
  `· 7d rp TRACKED|DROPPED` (5-dim keeps the old wording).
* **Selection is by config + checkpoint; the defaults are all OFF.** All three switches are stamped into
  meta: `policy.action_repr` (`auto` is the deployed value — above; an explicit value must equal the
  checkpoint contract `shape_meta.action.action_repr` at ARM and be in `state.POLICY_ACTION_REPRS_FLYABLE`
  — a 5-dim config with a 7-dim ckpt is refused with "action_repr mismatch", and vice versa),
  `policy.attitude_track` (false = **dropped-and-logged**: even when flying a 7-dim ckpt, the roll/pitch
  columns are discarded and only recorded — `plans.jsonl rp_raw`, and `dropped_rp_deg` becomes meaningful
  again; true = the NMPC tracks them), `engage.attitude_axes.enabled` (sends K/M as the MANUAL_CONTROL
  extension axes s = pitch / t = roll, with `enabled_extensions=0b11`; when false, allocation discards
  them as before). When OFF, the control output, xref, MAVLink frames and resolved config are
  **byte-identical** to 4-DoF (pinned by tests).
* **7-dim contract**: `[dx, dy, dz, dyaw, droll, dpitch, width]` — columns 0:4 and 6 are bit-identical to
  5-dim; (droll, dpitch, dyaw) are the ZYX Euler angles of the same `R_bt·R_rel·R_btᵀ`
  (`policy_frames.encode_pos_rpy` = training `yaw_action.encode_pos_rpy`). The ckpt must declare
  `shape_meta.action.rpy_convention {order: zyx, frame: body_frd_via_R_bt, columns: [...]}`, and the mount
  check compares it against `dp_policy.RPY_CONVENTION_STATION`; in this representation the `yaw_axis_*`
  keys are **required** (without them `mount_ok=false`, no lenient path). `action_raw` rows have 7
  columns; `render_run_scene` recomputes all three generations by width.
* **ARM truth table** (`_policy_refusal`; since 2026-10-03 the LOW decides — the "7d LOW followers" section
  below takes precedence): tracking happens only when `*_7d` LOW AND `action_repr == pos_rpy_width` AND
  `attitude_track` (veto), and then it requires all of `attitude_axes` live AND MANUAL AND the mount keys
  AND the firmware gate (mavlink20, AUTOPILOT_VERSION ≥ `min_firmware` 4.1.2, **parsing** the artifact of
  the bench probe `python -m rov_gui.tools.attitude_axes_probe` — `tool == rov_gui.tools.attitude_axes_probe`,
  `pass: true`, `mavlink_wire_version "2.0"`, `firmware_version` matching the vehicle — plus
  `sign_proven: true` in the `sign_probe` artifact) AND `rp_residual_deg ≤ 10°`; if even one is missing,
  **arm is refused** (no silent demotion). With `engage.attitude_axes.require_sign_probe: false` (default
  true) a run is allowed even with the sign unproven, but `first_water_caps [0.1, 0.15]` are enforced and
  `sign_proven: false` + `caps_in_force` are stamped into meta (as long as the sign is not proven, the
  effective caps are always the first-water caps). (Since 2026-10-03 a 5-dim ckpt flies level instead of
  being refused.) **STABILIZE + attitude_axes → refused** (s/t are reinterpreted as lean-angle targets).
* **Cap hierarchy**: T0 label range (normalizer + DDIM clip) → T1 `compose_plan` clip `rp_max_deg` 20 → T2
  PlanFilter reject `rp_reject_deg` 30 + `pq_max_rad_s` 0.35 dilation + `rp_jump_max_deg` 5 soft → T3
  `set_path_plan_ned` ValueError → T4 wire cap `cap_roll 0.2 / cap_pitch 0.3` (first water
  `first_water_caps [0.1, 0.15]`; enforced when `sign_proven: false`). All [예측].
* **Interlocks**: div_rp 15° for 0.5 s → demote the policy to a level DP hold (K/M keep flowing: active
  leveling), `abort_deg` 35° for 0.25 s → disengage, an axis stuck at the cap for `sat_ineffective_s` 1 s
  with the error not shrinking → disengage, extended-frame TypeError → demote to the 6-argument frame and
  `attitude_axes_degraded` → disengage on the next tick. On every hold entry the attitude reference ramps
  to level inside HwDobMpc at `pq_max` (not a step).
* **Records**: at the end of the mission CSV `rroll_deg,rpitch_deg,ax_roll,ax_pitch,rp_track` (nan/0 when
  OFF), at the end of `policy_plan.csv` `roll_deg,pitch_deg` (after reason; nan if absent), plans.jsonl
  `rp_tracked` always + `rp_raw` (if 7-dim)/`raw.rp` (only when tracked)/`anchor.rp_*`/`rp_clipped_*`,
  meta `run.attitude_axes{… sign_proven, caps_in_force, sat_ticks}`, `trajectory.attitude_track`,
  `trajectory.action_repr` (the ARM pin — pooling key), `policy.action_repr` (the pin; under auto, if the
  record contains no policy mission, the value that CSV last pinned, and if there is none either, null —
  then use `policy.status.action_repr`·plans.jsonl `action_repr`), `policy.config.action_repr` (`auto`
  as-is), `controller.attitude_ref{tracked, source, …}` (`source` is the value at the time meta is
  written; the pooling key is `tracked`), `controller.allocation.attitude`. **Runs that differ in
  `attitude_track`·`attitude_axes.enabled`·`action_repr`·flight mode·`caps_in_force` must not be
  pooled.** Offline tools: `dp_policy_offline.py gate|replay --action-repr pos_rpy_width` (roll/pitch
  RMS·zero-predictor·corr rows on held-out + GT |drp| percentiles),
  `policy_session_replay.py --ckpt D7=… [--attitude-track]` (pre-flight OOD probe), and stub sessions
  `stub7` (7-dim level) / `stub_rp` (pitch 5° ramp) to check the wiring without torch.
* **Offline closed loop (variant ON)**: `QT_QPA_PLATFORM=offscreen python rov_gui/tests/demo_e2e.py mpc_tuned_7d policy rp`
  (since 2026-10-03 only `pid_7d`/`mpc_tuned_7d`; the "7d LOW followers" section below) — runs the real
  PolicyWorker + `stub_rp` + a temporary copy of the config (`policy.action_repr: pos_rpy_width`,
  `attitude_track: true`, `engage.attitude_axes.enabled: true`, `require_sign_probe: false`; the `probe`
  placeholder file is a **valid probe JSON** that the gate parses and passes (tool·pass·
  `mavlink_wire_version "2.0"`·firmware_version matching), and `sign_probe` is a `sign_proven: false`
  placeholder → the demo's effective caps are `first_water_caps`, confirmed via meta `caps_in_force`)
  through compose → PlanFilter → stitcher → `NedPlan.rp_ned` → HwDobMpc xref[3:5] →
  `wrench_to_axes(attitude=True)` → the demo pendulum plant in one process, and checks plans.jsonl
  `rp_tracked`/`raw.rp`, CSV `rp_track`/`ax_pitch`/`rpitch_deg`, `MpcStatus.axes_rp`, `PolicyPlanViz.rp`,
  meta `run.attitude_axes`/`trajectory.attitude_track`/`controller.allocation.attitude` (passed
  2026-09-26; every plan rp_tracked; the records are in a temporary folder under /tmp, so they cannot be
  cited). Followers are mpc/mpc_tuned only (dobmpc is refused by `dobmpc_allowed: false` → automatically
  mpc). The stub's "+5° every chunk" is integrated through the leash anchor and climbs up to the
  `rp_max_deg` 20° clip (the T1 clip and M-cap saturation are deliberately exercised), so this is a wiring
  check, not a performance number. The deployed config (`config/hw_mpc.yaml`) has been `action_repr: auto`
  since 2026-10-03: choosing 7-dim gives dropped-and-logged on a 4-DoF LOW and tracking on a `*_7d` LOW —
  both `demo_e2e.py mpc_tuned policy auto` (deployed config as-is, `stub_rp`, CSV rp_track all 0·plans
  rp_tracked false·meta pin pos_rpy_width·`policy.config.action_repr` auto) and `pid_7d policy rp auto`
  (auto in a temporary config, rp_track 1 on every row) check this [측정: demo (synthetic plant, wiring
  check): `rov_gui/tests/fixtures/demo_e2e_mpc_tuned_policy_auto_20261004.log` 304 rows rp_track 0,
  `demo_e2e_pid_7d_policy_rp_auto_20261004.log` 308/308 rows rp_track 1].
* **First-flight order** (none of it run yet): ① bench probe (disarmed) → ② in-water sign probe (armed,
  no policy, caps 0.1/0.15) → ③ plain mpc attitude STEP (no policy; `roll_nm/pitch_nm` calibration) →
  ④ 7-dim ckpt **dropped-and-logged** flight (`attitude_track: false`, K/M not sent) + tagnav acceptance
  rate vs measured pitch → ⑤ active leveling (`attitude_axes.enabled` only) → ⑥ `attitude_track: true`.
  ④ may come before ①–③ (a matter of allocating hardware time, DP_6DOF_PLAN §7).
* **7-dim training entry points**: task yaml `external/UMI_aquatic/diffusion_policy/config/task/umi_depth_7d.yaml` /
  `umi_rgbd_7d.yaml` (`shape_meta.action {shape: [7], action_repr: pos_rpy_width, rpy_convention: …}`; the
  rest is the same as 5d),
  `tools/run_ablations.sh depth_7d_E_baseline | depth_7d_dinov3b | rgbd_7d_dinov3b | depth_7d_dinov3b_9426_9926`
  (run leaf `<stamp>_train_umi_depth_7d_<exp>`, `rp_label_stats.txt` inside the folder,
  `*_action_mse_error_rp` in logs, `action_repr`/`action_shape` in the best-ckpt sidecar json). To see only
  the label statistics: `external/UMI_aquatic/tests/print_rp_label_stats.py <zarr>…` (artifact
  `data/20260926/rp_label_stats_*.txt`). Attitude-tracking verification on the sim side is
  `bluerov2_mujoco_marinegym/verify/verify_attitude_hold.py` (see the README table). As of 2026-09-26
  there were **0** 7-dim training runs, and on 2026-10-03 the first 7-dim checkpoint came in
  (`data/checkpoints/20260929_peg7d_vitclip.ckpt`, peg 9/27+9/28, trained on bdml-server; offline gate and
  OOD replay not run).

**7d LOW followers — `PID_7D` / `MPC_Tuned_7D` / `RL_7D` (2026-10-03, 0 water runs)**: "who controls
roll/pitch" in the 6-DoF variant above is **decided by the LOW combo, not by a config switch.** Each
`*_7D` is the same x/y/z/yaw control as its base follower, plus roll/pitch.

* **Keys and bases**: `pid_7d` → `control/pid.py HwPid7d` (pid gains unchanged + roll/pitch PID),
  `mpc_tuned_7d` → the same HwDobMpc in mode `"mpc_tuned"` (the worker maps it; the key itself is not a
  HwDobMpc mode — `"mpc_tuned_7d".endswith("_tuned")` is False, so passing it through would silently fly
  with an isotropic cost), `rl_7d` → `control/rl_policy.py HwRl7d` (axes6 policy, the plan's roll/pitch in
  `q_des`). The combo order puts each right after its base
  (`None, PID, PID_7D, MPC, MPC_Tuned, MPC_Tuned_7D, …, RL, RL_7D, RL_PWM`).
* **Condition for K/M to go out on the wire = `*_7d` LOW AND `engage.attitude_axes.enabled`** (now a
  "station capability" — the sink reads it once at launch). With the capability off, `*_7d` **refuses** a
  hardware engage ("… restart the station with it, or pick pid"); with it on, every other LOW (pid, mpc,
  mpc_tuned, dobmpc*, rl, rl_pwm, mpcc) **engages as 4-DoF** (K/M discarded). From 2026-09-26 to 10-02,
  turning on just the capability made plain mpc/mpc_tuned emit K/M, and pid/rl/rl_pwm/mpcc were refused —
  there were 0 water runs in that period, so there are no records to reinterpret. `*_7d` goes through all
  the existing gates unchanged (MANUAL, firmware ≥ 4.1.2, MAVLink 2, probe JSON parse, sign probe or
  `first_water_caps`, interlocks ceiling/sat-ineffective/div_rp/degraded/coast).
* **Tracking decision**: flying a 7-dim ckpt (`pos_rpy_width`) on a `*_7d` LOW **tracks** roll/pitch;
  flying it on a 4-DoF LOW gives **dropped-and-logged** (the recommended first flight).
  `policy.attitude_track` is now a **veto** (default true; when false, dropped-and-logged + active
  leveling even on `*_7d`). With a 5-dim ckpt or a shape mission, `*_7d` **actively holds level**. In a
  land dry-run, `*_7d` + 7-dim is refused at ARM (no K/M path). Records: meta
  `trajectory.attitude_track_low/attitude_track_veto`, `run.attitude_axes.low/low_attitude/decided_by`,
  `controller.type` = the LOW key (mpc_tuned_7d also carries `base_type: mpc_tuned` — so plot_runs does
  not mix it with 4-DoF).
* **PID_7D loop** (`hw_mpc.yaml pid_7d:`, unknown keys raise an exception): τ = k_ff·[cosθr·sinφr, sinθr] +
  Kp·e + Kd·(T⁻¹(reference angles)·reference Euler rate − ATTITUDE p,q) + I. Kp/Kd = 3rd-order pole
  placement (ωn **0.5 rad/s**, ζ 0.9, α 0.2, heavy_gripper inertia; **omega_derate not applied**), Ki 0.15
  set separately, k_ff 1.35 (NMPC ZG·W, the lower bound of the restoring stiffness), I is stored in N·m,
  clamped to `0.8 × 선 토크 캡` ("0.8 × wire torque cap") and integrated conditionally on the actually sent
  K/M that the worker reports (integration stops in the saturating direction). The heave trim is rotated
  by the measured attitude; the yaw D term is the Euler yaw rate. When the plan cuts out, it returns to
  level with the same D5 ramp (`pq_max`) as HwDobMpc; the plan attitude goes through T3 verification
  (`rp_reject_deg`). **Margin [유도]** (control-theory-advisor consult + recomputation with
  `rov_gui/tests/sim_pid7d_attitude.py`, artifact `rov_gui/tests/fixtures/sim_pid7d_attitude_20261003.txt`):
  at a total delay of 0.2 s and actuation gain ×0.4~×2.5, PM ≥ ~50° holds only when the restoring
  stiffness k ≤ ~2.3 (roll)/3.6 (pitch) N·m/rad; at k ≈ 20 (the log-fit value on the RL side) PM is
  3–41° — hull resonance + delay, which gains cannot fix. So a **pulse-and-release test before closing
  the loop in water** (open-loop 0.15 axis pulse → sign·delay·g/I·k/I·damping) comes first. Closed loop
  in the same sim (the real HwPid7d, first-water caps): at k 1.35 a 10° pitch hold has steady-state error
  −0.03° and overshoot 2.7°; at k 20 the caps are insufficient, so 4–9° steady-state error remains while
  the axis is **below** the cap. Hence HwPid7d reports "authority exhausted" (command clipped at the cap /
  integral at its limit and error > 1° / error outside the 20° integration gate) via
  `attitude_authority_exhausted()`, and interlock (iii) judges it exactly like being pinned at the cap
  (safety audit 2026-10-03 HIGH: with the wire sign flipped, the PID sat at a wrong tilt below the cap and
  tripped nothing — pinned by `test_low_7d.py`). **Hardware engage gate**: `pid_7d.pulse_test`
  (pulse-and-release result JSON, `free_decay_period_s [roll, pitch]` ≥ `min_free_decay_period_s` 3.2 s)
  or the recorded bypass `require_pulse_test: false` (meta `controller.rp_loop.pulse_test`). Non-finite
  attitude → K/M 0 + n_fail (blocks the path where NaN became `max(-c, min(c, nan))` = +cap).
* **RL_7D**: accepts only axes6 exports with `obs_spec.json attitude_reference.trained: true` (fail-closed —
  the layout is the same as a level-trained export, so this key is the only discriminator).
  `q_des = quat_from_euler_xyz(roll_ref, −pitch_ref, −yaw_ref)` is clipped to the training envelope and
  limited every tick by the trained reference rate (`attitude_reference.rate_max_rad_s`; the build is
  refused without it), the attitude slew (1.5/s) is applied to roll/pitch too, K/M are clipped so they do
  not exceed the trained cap (`attitude_cap_trained`), the reference that the CSV and interlocks see is the
  clipped attitude (the one actually given), the build is refused if the envelope keys are missing,
  non-finite → neutral + n_fail. Wire sign: a4/a5 follow the physical convention (+a4 = +M nose-up, the
  training mixer), so the sink's `engage.attitude_axes.sign` maps physical→wire exactly as for the
  model-based followers. Engage is refused with a cap lower than the trained cap (sign unproven =
  first-water) (recorded bypass `rl_7d.allow_caps_below_trained`; the evidence is the limit cycle of the
  2026-10-01 rl_pwm cap-0.25 run). **No trained export** — RL_controller `scripts/_pipeline_att7_seed.sh`
  (preset `traj10att`) is only prepared, and training is deferred by operator decision (2026-10-03).
  `rl` (LOW RL) now refuses axes6 (that path was unreachable on hardware). **RL_PWM (pwm10_s1) was trained
  only with a level q_des, so it cannot track the roll/pitch of a 7-dim policy** (active level holding
  only).
* **Verification**: `rov_gui/tests/test_low_7d.py` (19), updated test_control/test_policy/test_replay/test_attitude_axes/test_offline,
  `demo_e2e.py pid_7d policy rp` / `mpc_tuned_7d policy rp` both OK (logs `rov_gui/tests/fixtures/demo_e2e_*_policy_rp_20261003.log`;
  the demo pendulum is a fake plant — a wiring check, not a performance figure). Giving the `rp` row a
  4-DoF follower now **fails** (previously it was silently switched to mpc).

**Deployed weights = `model`, epoch 195 (2026-09-07)**: one checkpoint holds two state dicts — `model`
(the original trained weights) and `ema_model` (the EMA copy) — but the EMA does not average buffers, so
the BatchNorm running stats stay at the ImageNet values (a pre-existing defect in KNOWN_ISSUES.md,
inherited by this retraining). So the deployment uses `model`: `policy.weights: model` in
`config/hw_mpc.yaml`·`config/land_dp.yaml` (single definition `rov_gui/control/geometry.py default_policy_block`),
overridden only by `--policy-weights`. 219-window held-out (`get_val_mask(75, 0.05, 42)` episodes
6/32/48/56, valid start stride 4, DDIM 16, de-normalised, all four rows with the same tool
`rov_gui/tools/dp_policy_offline.py replay --val-only` and the same windows):

| ckpt / weights | pos RMSE | yaw RMS (zero-predictor 2.56°) | width RMSE | Artifact |
|---|---|---|---|---|
| **5-dim ep195 / `model`** — deployed | **17.3 mm** | **2.20°** (corr +0.556) | **7.3 mm** | `heldout_pos_yaw_width_ep0195_model_stride4.log` |
| 10-dim 2026-09-01 ep50 / `model` | 21.0 mm | 2.24° (corr +0.551) | 7.6 mm | `heldout_pose10d_old_model_stride4.log` |
| 5-dim ep10 / `ema_model` — discarded first choice | 31.3 mm | 3.36° (corr −0.063) | 10.4 mm | `heldout_pos_yaw_width_ep0010_stride4.log` |
| 10-dim ep50 / `ema_model` | 29.8 mm | 3.44° (corr +0.045) | 8.9 mm | `heldout_pose10d_old_2026-09-01_ep50_stride4.log` |

[측정: all four logs are in `/home/bdml/Desktop/umi_underwater_robust_control/data/20260907/0907_113747_train_umi_depth_5d_can_grasp_depth_5d_v0/`, the summary is `checkpoints/selected.json`, and the epoch-sweep figure is `train_curves_20260907.png`].
Measured with the deployed weights, the 5-dim retraining is ~18% better in position and tied in yaw, and
**yaw is learned** (below the zero-predictor 2.56° and corr ≈ +0.55) — the earlier reading that "yaw is
sampler noise" was an artifact created by the EMA weights. One trap: the 32-window val curve in
`logs.json.txt` is **computed with the EMA weights**, so it must not be used to rank this checkpoint
family — that curve's minimum is at epoch 10, but measured with the deployed weights epoch 10 is the
**worst** of those measured (26.7 mm [측정: `heldout_pos_yaw_width_ep0010_model_stride4.log`]) and
position is flat from ep100 on (17.2–17.6 mm [측정: same directory `..._ep{0100,0150,0175,0190}_model_stride4.log`],
within the 219-window noise). The first choice, picked with this curve, stood the whole run on its head.
Runs with different weights must **not be pooled** with each other (meta `policy.weights` is the
boundary).

Verification: `rov_gui/tests/test_policy_frames.py` (frame algebra + the 5-dim encoder's upstream bit
parity·sign·mount·legacy equivalence, robust+rovgui-pose, 22),
`test_policy_obs.py` (depth recipe bit-parity, rect_left inverse rectification, 9),
`test_dp_policy.py` (session; without a GPU only the stub half, 17 — 2026-09-11 `ckpt=` kwarg precedence),
`test_policy.py` (MpcWorker side A1–A22 + OBSERVE + action_repr/mount gates + 2026-09-11 LOW None·ckpt
pin, 79), `test_policy_worker.py` (worker: pairing·skip·status·wiring·CLI·FS tap·mount check + 2026-09-11
`set_ckpt`·parser, 28), `test_policy_ckpt_paths.py` (single source for ckpt paths + a single max_run_s
value, 4), `test_run_scene.py` (recomputing old and new action_raw, 19), `demo_e2e.py dobmpc policy`
(stub, real acados), and `rov_gui/tools/policy_dryrun.py` (real checkpoint, synthetic scene, PASS/FAIL
criteria: worker ready < 30 s, plans ≥ 20, infer_ms p50 reported, 0 worker errors, reject ratio
**reported**). Parity tool `rov_gui/tools/dp_policy_parity.py`, offline gate
`rov_gui/tools/dp_policy_offline.py`.

### Sharing the GPU with FoundationStereo — `--policy-fs-schedule` (2026-10-01, unverified in water)

**Problem.** The policy and FoundationStereo (FS) use the same GPU in one process. At the `--policy` defaults
(iters 8, scale 0.75), FS takes 75.7 ms per frame, longer than the 66.7 ms camera interval, so it runs without a
break, and next to it the policy forward grows from 19.3 ms alone to p50 152.8 ms [측정: data/20260930/
0930_220212/diag/policy_vs_fstereo_resolution.json, policy_vs_fstereo_contention.json, offline].
On hardware (2026-09-30) infer_ms was p50 207–273 ms, and this accounts for 0.21 s of the plan age p50 0.56 s
[측정: same folder plan_timing.json, latency_breakdown.json]. Even though the CUDA stream the policy uses is a
separate one, the GPU is still shared (`perception/dp_policy.py` "Timing and threads").

**How to choose — the default stays as it is; a flag changes it.**

```bash
# existing (free): no coordination at all — every run before 2026-10-01
./c3 gui --source hw --allow-command --mpc --fstereo --policy
# method 1a (yield): FS starts no new frame, but only while the policy is inferring
./c3 gui --source hw --allow-command --mpc --fstereo --policy --policy-fs-schedule yield
# method 1b (only): during a policy mission FS computes only the 2 frames the policy uses
./c3 gui --source hw --allow-command --mpc --fstereo --policy --policy-fs-schedule only
# method 2 (resolution): the code path stays as it is, the FS input goes to 320x200
./c3 gui --source hw --allow-command --mpc --fstereo --policy --fstereo-scale 0.5
# the two may be used together (e.g. --fstereo-scale 0.5 --policy-fs-schedule yield)
# method 3 (process split, 2026-10-12): only the FS network runs in a child process — independent of the schedule, may be used together
./c3 gui --source hw --allow-command --mpc --fstereo --policy --fstereo-process
```

* **yield**: once the policy starts an attempt (past the three pre-forward skips), FS does not start a
  new frame, and right before the forward the policy waits **at most 150 ms** for the FS frame in
  progress to finish (`FsGate.wait_s`) and then has the GPU to itself. The trigger and the frames it
  picks are the same as under `free`. The depth panel draws about one frame fewer per plan.
* **only**: while a mission is running and a forward is possible (when there is a fresh fix, proprio
  history, a grid and a start pose), the policy requests 2 frames shortly before the period ends; FS
  computes only those 2 and discards the pairs in between without computing them. The two are
  **consecutive camera frames** (as long as one FS frame takes less than twice the camera
  interval). A frame whose computation failed, or that did not get into the policy mailbox, does
  not use up the request's quota. While a forward is impossible (tag dropout etc.) and outside a
  mission, FS free-runs. During a mission the depth panel shows around 4 frames per second, and
  `[only]` is appended to the end of the chip. Used together with `--pose`, FoundationPose gets no
  depth for most colour frames [예측: 250 ms pairing limit] — it warns about this at launch.
* **resolution (`--fstereo-scale 0.5`)**: one FS frame takes 45.7 ms, so it keeps up with 15 fps
  and has idle time left over. The cost is a mismatch with the training depth: pixels off by more
  than 1 px in disparity 7.0 % → 11.1 % [측정: rov_gui/tools/fstereo_bench_out/sweep.txt]. Whether
  the policy tolerates this difference is unverified.

**There is no way to stop FS for good.** The FS tick does not wait at the gate (it asks and returns at once),
and each of the following releases by itself: the policy's hold after its 0.5 s lease, and `only` at the policy tick's
heartbeat (1.0 s), at 1.5 s with no request, at 0.6 s after a request, and at 2.5 s of an unfinished-frame mark
(`perception/fs_gate.py` header). The wait on the policy side is at most 150 ms in real time. Under `free` the
gate object is not created at all.

**Offline measurement — the two real workers, the two real networks** [측정: rov_gui/tools/fstereo_bench_out/
policy_fs_schedule_20261001_115008.json, `rov_gui/tools/policy_fs_schedule_bench.py`, 30 s per mode,
RTX 5090, peg ViT-CLIP ckpt 8 step]. The input is synthetic: 9 land stereo pairs are replayed at 15 fps and the
put time is used as the capture time, so **the camera transfer latency (~0.1 s on hardware) is missing**, the
vehicle state is synthetic too, and there is no GUI or controller. During the measurement another test was using
the CPU (1-min load 0.9→7.5 / 16 cores, recorded in the JSON).

| Mode | Policy forward p50/p90 | FS wait p50/p90 | Plan age (emit) p50/p90 | Consecutive frame pairs | FS during a mission |
|---|---|---|---|---|---|
| free, scale 0.75 (existing) | 132.7 / 172.0 ms | — | 328.8 / 370.9 ms | 82.5 % | 12.93 /s |
| yield, scale 0.75 | 19.4 / 19.9 ms | 0.0 / 19.5 ms | 224.6 / 256.3 ms | 86.0 % | 11.93 /s |
| only, scale 0.75 | 19.4 / 19.6 ms | 0.0 / 0.0 ms | 154.5 / 165.0 ms | 100 % | 3.73 /s |
| free, scale 0.5 | 30.8 / 56.5 ms | — | 134.5 / 191.7 ms | 100 % | 15.0 /s |
| yield, scale 0.5 | 19.6 / 20.1 ms | 0.0 / 6.5 ms | 145.6 / 166.5 ms | 100 % | 15.0 /s |

The plan interval was p50 0.51–0.53 s in every mode (period_s 0.5 is the lower bound). On hardware the plan age
adds the camera transfer and the hardware-side load on top of this — on 2026-09-30, hardware with the existing
scheme had an intake age p50 of 0.556 s.

**Records (do-not-pool boundary).** run meta `policy.worker.fs_schedule` (requested / effective / why / worker
counters / `only_lead_s` / gate snapshot), `policy.run.fs_schedule` (at ARM time), `fstereo.schedule`
(null = no gate), `policy.worker.pairing.trigger_detail`; at ARM, if not `free`, one warn line and
events.log `FS SCHEDULE …`; `fstereo.policy_fs_schedule` in the `stereo/meta.json` of `--record-stereo`.
**On every policy plan line** (plans.jsonl): `fs_schedule`, `fs_input` ("scale 0.75 it8"), `fs_wait_ms`,
`fs_wait_timeout`, `depth_ready_age_s`, `trigger_age_s`, `emit_age_s` (seconds, relative to the observation's
capture time; None = not stamped — these keys appear in `free` runs too). `age_at_intake_s − emit_age_s` is the
bus·inbox·control-tick segment. Do not pool plan age, infer_ms or FS rate across runs with different schedules.
`only` also changes the observation pair spacing (`obs_pair_dt_s`, `pairing.near/fallback`) and the trigger, so
do not pool **policy behaviour** with `free`/`yield` runs either. `schema_version` stays at 16 (keys added only).

**Before putting it in the water for the first time**: first look at `fs_wait_ms`·`fs_wait_timeout`·
`policy.worker.fs_schedule.worker.burst_timeouts`·`pairing` with a LOW None (`--policy-observe`) run.
Verification: `rov_gui/tests/test_fs_gate.py` (the gate alone), `test_policy_worker.py` (two workers·CLI·wiring),
`test_policy.py` (record keys), bench `rov_gui/tools/policy_fs_schedule_bench.py`
(`--arms free@0.75,yield@0.75,only@0.75,free@0.5,yield@0.5`).

**Method 3 — `--fstereo-process` (2026-10-12, unverified in water).** Only the FS network runs in a child process
(`python -m rov_gui.perception.fstereo_proc`, `rov_gui/perception/fstereo_proc.py`). The camera, the mono pair and its
capture time, the depth panel, `--record-stereo`, FoundationPose, the policy tap and FsGate stay in the station as they
are, and one pair at a time makes a synchronous round trip — so `--policy-fs-schedule yield|only` means the same thing
even when used together with it. The output is **bit-for-bit identical** to the in-station session: 9 land pairs × 2 passes,
at both scale 0.5 and 0.75 `depth_native`·`depth_mm` match 18/18, max |Δ| 0 mm. The IPC — the round trip minus the FS
computation — is p50 0.33–0.48 ms, and a real child shutdown takes 0.51 s
[측정: rov_gui/tools/fstereo_bench_out/proc_parity_20261012_120951.json, tool `rov_gui/tools/fstereo_proc_parity.py`].

A comparison measured with the two real workers (the same tool as the bench above, arm syntax `<schedule>[+proc]@<scale>`; synthetic pairs, no camera or GUI):

| Mode | Policy forward p50/p90 | Plan age (emit) p50/p90 | FS during a mission |
|---|---|---|---|
| free@0.5 | 30.6–31.4 / 56.6–57.0 ms | 133.0–133.5 / 191.0–192.4 ms | 15.0 /s |
| yield@0.5 | 19.5 / 20.1–20.2 ms | 142.9–146.7 / 163.8–166.5 ms | 15.0 /s |
| free+proc@0.5 | 29.5–32.6 / 39.2–42.9 ms | 129.1–132.9 / 170.7–179.2 ms | 15.0 /s |
| yield+proc@0.5 | 19.5 / 19.9–20.0 ms | 143.8–145.0 / 163.5–164.3 ms | 15.0 /s |
| free@0.75 | 121.6 / 174.8 ms | 321.2 / 365.5 ms | 12.97 /s |
| yield@0.75 | 19.4 / 20.1 ms | 233.3 / 264.0 ms | 11.9 /s |
| free+proc@0.75 | 40.8 / 48.9 ms | 222.3 / 275.3 ms | 12.8 /s |

[측정: rov_gui/tools/fstereo_bench_out/policy_fs_schedule_20261012_100908.json (all modes, 1 pass),
policy_fs_schedule_20261012_121018.json (the four scale 0.5 modes, 2 passes in ABCD order — the ranges are the values of
the two passes)]. How to read: the split does not cut the forward as much as `yield` does, but the plan age is similar.
`yield` has a 12–15 ms longer trigger age and so loses part of its forward benefit. These figures leave out the camera
transfer (~0.1 s) and the station load, so **the underwater ranking cannot be decided from them** — the station forward
was around 3× slower than the bench with the same placement (KNOWN_ISSUES "Policy inference takes 19 ms on its own but").

**A failure means something different.** If the child dies or does not answer (5 s per frame, 10 s for the first frame,
120 s for loading), that run has no depth to the end — FAULT, no fallback, no restart within the run (relaunch the
station). The in-station session's infer has no timeout at all. The child dies together with the station when the
station dies (socket EOF + parent-pid monitor, during loading too).

**Underwater A/B within the same session.** Differences between sessions are larger than differences between placements:
the same free@0.5 had 3.9–7.4 % expiring before the next plan on 10-01, and 35.4 % on 10-03
[측정: data/20261012/1012_fs_server_review_offline/plan_timing_by_schedule.json]. Fix the scale (0.5), and with the same
ckpt alternate `free` / `--policy-fs-schedule yield` / `--fstereo-process` ABAB, keeping the recording flags the same on every arm.
If recording is on, wait **more than 90 s** between relaunches (KNOWN_ISSUES "Relaunching the station within 90 s"). Reading: `python3 data/20261012/1012_fs_server_review_offline/
plan_timing_by_schedule.py data/<date>/<run> …` — splits by (fs_schedule, fs_process) and reports plan age·infer·late·expiry.
Verification: `rov_gui/tests/test_fstereo_proc.py` (25 tests with a real child + a stub network: round trip, left/right distinction,
crash·hang·load failure·send deadline, shutdown limit, child shutdown on station death, setup branches, record keys, CLI refusal).

### The checkpoint is picked in the panel — the ckpt field (2026-09-11, unverified on hardware)

**`--policy-ckpt` is gone** (operator request). The checkpoint is picked in the panel's **ckpt field**, which is visible
only when HIGH = `Diffusion Policy` (immediately to the right of the HIGH combo, visible only when HIGH is `Diffusion Policy` — it takes
no row of its own): `ckpt` label · file name (basename, long ones shortened, the full path in the tooltip) ·
status (`loading…` / `READY` / `ERROR: …`; since 2026-10-03, `READY · 5-dim`/`READY · 7-dim` when there is a contract,
and a red `· not flyable`/`· ≠ config X`/`· MOUNT MISMATCH` for a checkpoint that START would refuse) · the `…` button.
`config/hw_mpc.yaml policy.ckpt` (land_dp.yaml has the same line too) is only the **startup seed** — at launch that file goes
onto the GPU and its name appears in the row, but after that the name the row shows comes only from `PolicyStatus.ckpt`
(the session the worker **holds** right now). A file picked with `…` is shown as the request `→ <이름>` ("→ <name>")
until it is confirmed; if the worker accepts it, `loading…` → `READY`; if it refuses, the reason appears **in red text
for 6 s** (`PolicyStatus.ckpt_note`; also one warn line in the log — a refusal that used to stay only in the log went
unseen next to the name silently reverting). The path by which tests/tools plug in `Opts.policy_ckpt = "stub"` directly
is still alive (only the argparse argument was removed; typing `--policy-ckpt` makes the parser refuse it).

The file dialog is opened by the window (`MainWindow._pick_policy_ckpt`), not by the panel — for the same reason as the
recording-name field. The moment it opens, it releases every pressed key (`teleop.all_stop`) and prints the warning
"choosing a checkpoint — the keyboard is NOT flying the vehicle", and **Esc closes the dialog and is an E-STOP at the
same time** (`CkptDialog.estop_requested → estop`; so that the rule that Esc is an E-STOP everywhere is not broken
inside the dialog alone). **The joystick keeps flying even while the dialog is up** (the reader is a timer, so it stays alive
in the modal loop too). If launched without `--policy`, the button does not open the dialog and only gives a "no policy
worker" warning — so as not to leave hanging on the row a request that nobody will take. The button is disabled while
engaged.

The swap (`PolicyWorker.set_ckpt`; a queue slot on the worker thread, so it does not compete with inference) **refuses**
in the following order, and each refusal reaches the panel as a warn log + `ckpt_note` ("checkpoint swap REFUSED — …"):

| Order | Condition | Result |
|---|---|---|
| 1 | the worker is not running (stopping / torn down / before setup) | refuse `the policy worker is not running` |
| 2 | empty path | silently ignored |
| 3 | the same path as the file already held (symlinks are resolved before comparing — `selected.ckpt` and its target are the same file) | no swap, "already the loaded checkpoint", only the status is republished |
| 4 | the controller's PolicyState silent for more than 1 s / `engaged` | refuse `the controller is silent … wait` / `DISENG first (a mission may be armed …)`; if there is no state at all (no controller·not ready yet), allowed — there is nothing to be armed |
| 5 | the held session is still `loading` | refuse `still loading <이름>; wait for READY or ERROR` ("still loading <name>; wait for READY or ERROR") (a second loader takes twice the GPU memory) |
| 6 | neither `stub` nor a file | refuse `not a file: <경로>` ("not a file: <path>") |
| 7 | FS-vs-training parity mismatch under `--fstereo` | **warning** only (the same rule as `check_policy` — the default `--policy` settings themselves differ from the store, so refusing would make any real pick impossible in a default launch; the design was to refuse, and it was changed in the implementation); also a warning when the store cannot be read |

If the swap passes, the old session is closed (torch released, CUDA cache emptied), the per-session state (ready·sha1·
mount check·pairing wait) is reset, `pc["ckpt"]` is changed, and then the new session is brought up asynchronously. If
the factory throws an exception (after the old session is already closed), the worker honestly shows `ERROR` while
holding **no** network at all, and recovers with the next pick. If `--record-depth` is on, the recorder stays armed and
writes the swap down as a seq boundary (`policy_obs/meta.json extra.ckpt_swaps` — `extra.ckpt` is a value captured once
at the first frame, so reading only that gives the old network's name).

**The checkpoint is pinned at ARM.** `MpcWorker._arm_policy` writes `PolicyStatus.ckpt` / `ckpt_sha1`, not the config,
into `scen["ckpt"]` / `scen["ckpt_sha1"]` (the same pattern as obs_dt), and intake rejects a plan carrying a different
sha1 without a strike (`reject_ckpt` counter, one line in `plans.jsonl`, one log line; lenient only when both are empty —
stub / hand-made plans). If the worker is loading again under an armed mission (`loading and not ready`), the mission ends
with `_policy_end("worker_reloading", …)`. So a pick during a mission is refused by the worker (No. 4), and the button is
also disabled while engaged.

Records boundary: meta `policy.ckpt_loaded` (what the worker held at write time) / `ckpt_config` (the YAML seed) /
`ckpt_source` (`launch` | `panel`) / `config_note`, `trajectory.ckpt` + `ckpt_sha1` (the ARM-time pin — this is what to
read for which checkpoint **did the inference**), run counter `reject_ckpt`, worker meta `ckpt_swaps`.
Never pool runs with different weights with each other (the same reason as the `policy.weights` boundary above).

Verification: `test_offline.py` 6 tests (the ckpt-field truth rule·`ckpt_note` 6 s·dialog Esc·window wiring and the YAML
seed·HIGH/LOW combos), `test_policy_worker.py` 6 tests (swap·same-path no-op·refusal reasons·factory exception·the parser
refuses `--policy-ckpt`), `test_policy.py` 3 tests (pin/reject, config fallback, `worker_reloading`), `test_dp_policy.py`
1 test (`ckpt=` kwarg precedence). A panel swap has not yet been tried on hardware or in the pool.

### RGB checkpoints — feeding them the C3 centre lens (2026-10-03, unverified on hardware)

An RGB-only checkpoint (`shape_meta` image key `camera0_rgb`, e.g. `umi_rgb_5d_peg5d_rgb_dinov3b_…`) is also simply
picked in the same ckpt field — **there is no new flag.** On the first tick at which the session becomes READY, the
worker reads the image key of the checkpoint's contract (`PolicyWorker._adopt_obs_contract`) and decides the input:

| Contract image key | Input | Notes |
|---|---|---|
| `camera0_depth` | the same as before (FoundationStereo/device depth → training grid) | no behaviour change |
| `camera0_rgb` | **C3 colour frame (centre lens, the same frame as tag detection)** → square centre crop → 224 `INTER_AREA` → BGR→RGB (`policy_obs.ColorObsBuilder`, the same recipe as training's `build_dataset.py square()`) | the FS schedule gate is not used |
| anything else, or RGB but the training store's channel order is not written as `RGB…` (`.zattrs rgb.stored_channel_order`)·`image_shape` is not (3, R, R)·no C3 colour supply on this backend | none — panel `ERROR: unsupported ckpt obs …`, ARM refused | previously a KeyError on every tick after READY, 0 plans (2026-10-03 `data/20261003/1003_151227`). The status becomes READY only **after** the contract is applied |

**The input is naive: its optics are not matched.** The training RGB is the handheld colour camera (in air) cropped
without a warp, while the underwater C3 colour has a narrower field of view, so the same scene appears about **1.8×
magnified** [유도: focal length at 224, training ~178 px vs C3 ~319 px — `policy_obs.RGB_OPTICS_NOTE`]. By operator
decision (2026-10-03) the already-trained checkpoints are tested first as they are; the proper fix is retraining on RGB
warped to the C3 optics (`umi_handheld/warp.py`). Records boundary: meta `policy.worker.obs_mode` (`depth` | `rgb`)·
`obs_image_key`·`obs.optics`, per plan `depth_src` = `c3_rgb` — **never pool with warped-RGB runs.**

**C3-warp RGB checkpoints (2026-10-04)**: a store built with `umi_handheld/build_dp_rgb_c3warp_zarr.py` reprojects the
land RGB into the C3 colour optics (`configs/target_camera_c3_cam_a_640x360.yaml`) and applies **the same crop as this
station** — the input path is exactly the same as above, and only the records change. The station reads the ckpt
training store's `.zattrs rgb.optics` as the contract's `image_optics` and writes meta `obs.optics_kind` as `c3_warped`
(only when that store **declares** the C3 warp) or `naive` (a store that says nothing = every RGB store before
2026-10-04), and writes the `obs.optics` sentence and the warn log at ARM to match (`policy_obs.rgb_optics`). It is read
again on every ckpt change. Split the two kinds of runs by `optics_kind`.

Image·proprio pairing: each attempt is built on **the very colour frame of the most recent tag fix**
(`PolicyWorker._anchor`; the stamps of the two taps are `now() - age_ms` of the same frame, so they differ by
microseconds, `COLOR_FIX_MATCH_S` 5 ms). The newest colour frame is usually one whose fix has not come out yet, so using
it would make the proprio "now" row lag behind the image. If every frame is newer than the latest fix, `skip_fix_lag`;
if the fix stalls and the anchor frame stays the same, the same obs is not planned again (`skip_same_obs`). A colour
frame whose size changed or that is not uint8 is discarded and counted (`color_refused`, status DEGRADED — cleared when a
frame of the original size arrives). The FS schedule does not apply to RGB checkpoints (meta `fs_schedule.effective` =
free; FoundationStereo keeps running for the panel).

With an RGB checkpoint these **depth settings are not used**: `policy.obs_view_forward_m` (depth re-render — the worker
does not apply it, and at ARM there is one info line "configured but NOT applied"), `min_obs_coverage`·`z_near_m`/`z_far_m`
(depth only). `--record-depth` does not record RGB obs (one warn line). The offline tools (`policy_session_replay.py`,
`dp_policy_offline.py`) are still depth-only (KNOWN_ISSUES). On a swap, the previous network's `plan_hz`·`infer_ms` are
reset (previously the RGB run's meta kept the values of the depth run just before it).

```bash
# offline check — demo backend + the real MpcWorker + an RGB-contract stub
QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/rovgui-pose/bin/python \
  rov_gui/tests/demo_e2e.py mpc_tuned policy rgb
```

Verification: `test_policy_worker.py` 9 tests (C3 colour obs·pairing anchored on the fix frame·`skip_fix_lag`·frame
size/dtype rejection and recovery·an unsupported key = ERROR·channel order/shape refusal·`skip_same_obs`·no READY before
the contract is applied·FS parity skipped for an RGB ckpt·depth↔RGB swap),
`test_policy_obs.py` 2 tests (bit-identical to the training recipe·size/dtype/greyscale rejection),
`test_dp_policy.py` (`stub_rgb`), `demo_e2e.py mpc_tuned policy rgb` (synthetic scene·stub: all 24 plans `c3_rgb`·24 installed [측정: `rov_gui/tests/fixtures/demo_e2e_mpc_tuned_policy_rgb_20261003.log`]). 0 water runs.

## Watching only the policy — LOW level None (teleop; started 2026-09-03 as `--policy-observe`, a runtime choice in the panel since 2026-09-11, unverified on hardware)

**A mode that takes the low-level controller out and checks only the diffusion policy.** The DP infers as usual, and its
action chunk is, as usual, composed·filtered·stitched into a datum-NED plan and **drawn in the trajectory panel**. There
is only one difference: `ctrl.step()` is never called at all, and there are **no bytes at all** going from this worker
to the vehicle. Control for the whole session is **the pilot's joystick** — the panel does not take the stick away, so
you can fly around the bottle in the water yourself and see from many positions "what trajectory is the network drawing
right now".

**Since 2026-09-11 this is not a launch flag but a choice in the panel.** `None` in the trajectory panel's **LOW** combo
(the first item — the only LOW that cannot move the vehicle on its own) is this mode, and there is a single source of
truth: `MpcWorker.observe` is True when LOW is `none`, and only then (a read-only property, written only by `set_mode`;
`cfg.mode` is `none` too). `--policy-observe` remains as a **launch alias** that pre-selects LOW None (for scripts; the
same as `--mpc-mode none`, and `resolve_defaults` fills `mpc_mode` with `none`). Giving the alias together with a
different `--mpc-mode X` is a contradiction, so **the launch is refused**; the alias is still refused without `--policy`
and `--mpc`; `--mpc-mode none` alone is not refused (a harmless idle station on which every START is refused).

```bash
./c3 gui --source hw --allow-command --mpc --fstereo --policy
#    → None in the LOW combo, Diffusion Policy in the HIGH combo → START
#    → when the chip shows "OBSERVE · DP NOT COMMANDING", it is open loop
#    → fly directly with the joystick. Moving the stick does not break off the mission.
# the alias that pre-selects the same thing at launch (for scripts; the only way before that):
./c3 gui --source hw --allow-command --mpc --fstereo --policy --policy-observe
```

**Frozen for each engagement.** LOW is picked before START: while engaged the combo is disabled and `set_mode` refuses,
and it also refuses while a controller CSV is open via REC (the open CSV pins the run folder to the old tree — the tree
rule below). Every time it is switched to None, the worker prints **the DISARM list as a WARN log** again, the same list
as the launch banner, and leaves the event "LOW None (teleop) selected" (it does not open a folder — that is put off
until there is a run worth opening one for); switching back to a controller prints "the station commands again". The
HIGHs that can START under None are **only Diffusion Policy and Replay** — the approach·settling·corner gates of
Station/Line/Square/Circle/Follow all rest on the premise "the vehicle is being driven onto the reference", so `set_traj`
refuses with "LOW None (teleop) runs only Diffusion Policy / Replay — pick a low-level controller (PID/MPC/…) for
<shape>", and the panel's mission label says the same thing before you press. A bare ENGAGE (HOLD) works under None too —
the datum·CSV·PolicyState feed is everything observe needs. What **holds** the reference is the PID
(`_ctrl_for("none")` → `self._pid`; it is always built, `path_plan_steps`·`ref_ned_at` make the drawn reference the
same, and it has no cost-scale API, so the meta cannot claim "a scale that was never flown"). It is not `step`ped
even once, and `_mpc_ctrl.mode` is not touched (`HwDobMpc.MODES` has no `none`). Of the engage gates, the ones on the
command path (including the real-time re-probe) are skipped under None — for the same reason as the five guards in the
table below.

**Why `engaged` and `commanding` were split.** Previously the single bit `MpcStatus.engaged` meant both "the mission
machine is running" and "the station is driving the vehicle", and at least four places read it as the latter (teleop
pump/gate, the command bar, the chip's tracking error). Making observe `engaged=False` kills the policy mission itself —
`set_traj` refuses, `_tick_replay` sits inside `if self.engaged`, the proprio history does not fill up, and
`PolicyWorker._active()` requires `PolicyState.engaged`. So `engaged` was narrowed to the former, and **`commanding`
(= `engaged and not observe`)** was newly created to take over the latter. All code that means actuation reads
`commanding`.

**The five automatic stops that presumed a follower are switched off.** The pilot moving away from the reference
**deliberately** is this experiment, so each of them would end the session just by doing its own job:

| Guard | Original behaviour | observe |
|---|---|---|
| divergence (`div_max_m` 0.25 m) | mission ends after more than 0.5 s | computed·recorded only (`run.div_max_seen_m`) |
| escalation latch (3 rejects in a row) | halt → GPU idle | count +5 s-periodic warning, no latch |
| bridge-too-long (`stale_s`) | end | warning only (plans keep being skipped) |
| workspace box (datum ±1 m) | hard reject | margin computed·sentence recorded, no reject |
| tag-loss disengage | disengage → **datum replaced → plot reset** | only that tick is skipped |

The box matters especially: the datum is the ENGAGE point, so flying only 1.2 m away gets every plan rejected, and only a
faint dotted line is left on screen, which reads as "the network died" — exactly the misreading that
`_emit_policy_plan_viz_refused` was built to prevent. The jaw is forced OFF regardless of `policy.gripper`, and the run
clock uses `policy.observe_max_run_s` (1800 s).

**Tree rule (2026-09-11) — under LOW None every engagement, regardless of shape, lands in
`data/*/*_observe/`**: a policy run, a replay or a bare HOLD without a mission alike
(`MpcWorker._run_tree`; since 2026-09-14 not a separate tree but the **leaf suffix `_observe`** inside the same date
folder). This is because the pooling guard is the kind, not the shape — `plot_runs.py` looks only at underwater leaves
(no suffix) and reads `controller.type`, and under None that value is `none`, so the run goes into no controller's pool.
The 90 s JOIN window bites **harder** here than for a land dry-run: an observe run naturally starts right after a
closed-loop attempt ("that didn't go well, let's just watch"), and within that window `plans.jsonl`·`policy_plan.csv`
get **appended** to the previous run folder, and since the lines of both runs are all `kind: "policy"`, they cannot be
separated later, even in principle. That is why a LOW change is refused while a REC CSV is open. If `--slam` is on as
well, the SLAM child follows the `_run_tree` at the time it starts (passed as a callable since 2026-09-11; a launch-time
`runstore.Tree` only when there is `--slam` and no `--mpc`).

**Records boundary — five layers, and any one of them alone tells the runs apart.** ① The run lands in a **separate kind
(leaf suffix)**, `data/*/*_observe/` (runstore JOINs only a same-kind folder within 90 s, so without a separate kind it
would be appended to the preceding control run's `plans.jsonl` — word for word the same as the landdry argument in the
`_run_tree` docstring). ② An `observe` column in the CSV (schema 11); since 2026-09-11 meta `schema_version`
**15** — a top-level `mode` (attached to every run), under None `controller = {type: "none", role: "reference holder — never stepped (LOW level None / teleop)", holder: <PID meta>}`,
the `mode` value `none` in the CSV·meta, and `none` in `MpcStatus.solver` and in the CSV `solver` cell (so that the
holder's `pid` does not leak out). ③ In those rows `uX..uN`/`w*`/`ax_*`/
`solve_ms`/`tick_ms`/`e_along`/`e_cross`/`ref_speed_m_s` are **nan** and `solver_status` is
**-1** (a value the solver cannot produce — 0 is "converged" and would read as a silent failure); only `px,py` (the
manually flown trajectory) and `rx,ry` (the reference the network requested) remain real numbers. ④ A POLICY OBSERVE
clause in the meta's `source`, `policy_observe: true`, `policy.synthetic: true`, `policy.gates_enforced` (which guards
were actually on), and an `observe_` prefix on `end_reason`. ⑤ `follower: observe|mpc` in `plans.jsonl`·`policy_plan.csv`
— because the same `status: accept` means "the NMPC tracked it" in a control run and "it was drawn" in an observe run.

**No number from these runs is a closed-loop result.** The vehicle went where the pilot took it, and nobody followed
the reference. Even the accept/reject ratio is the result of dragging the anchor around by hand, so **never pool** it
with the control-run `plans.jsonl` that `hw_mpc.yaml` cites as the [측정] evidence for
`num_inference_steps`/`v_max_m_s`.

Verification: `test_observe_*` in `test_policy.py` (79) (the 9 from 2026-09-03 — no commands·no step·status bits·the five
guards·jaw forced off·run clock·records boundary·end_reason prefix·no change when off — and the additions after that) +
the 14 of the 2026-09-11 LOW None family (`test_low_none_*`: the holder is the PID·engage with the vehicle DISARMed/ENABLE
off·shape refusal·mode-change refusal while engaged, then back to a controller; `test_mode_change_is_refused_while_a_rec_csv_is_open`,
`test_replay_under_none_lands_in_the_observe_tree`, `test_run_meta_under_none_marks_the_holder_and_the_solver_none`,
`test_none_never_joins_the_bridge_modes_and_never_touches_its_mode`, `test_observe_is_a_read_only_property`,
`test_policy_observe_opts_alias_still_yields_mode_none`, `test_observe_engages_even_when_the_holder_reports_not_realtime`,
…), `test_control.py` 2 tests (`mode: none` load·normalization, `csv_open`/`none` status publishing), `test_offline.py`
(the LOW combo's resync·lock, mission label), `test_policy_worker.py` (parser: `--policy-observe` → `mpc_mode none`,
`--policy-observe --mpc-mode dobmpc` refused, `--mpc-mode none` alone allowed). `--policy-observe` is still **refused**
without `--policy` and `--mpc`, and when launched with LOW None (whether by the alias or by `--mpc-mode none`) the stderr
banner says what was switched off — every time None is picked in the panel, the same list is printed again as a WARN log.
Smoke: `rov_gui/tools/policy_observe_smoke.py` (15/15; leaves a stub run in the real `data/*/*_observe/` tree). Changing
LOW from the panel has not yet been tried on hardware.

## Saving underwater depth — `--record-depth` (2026-09-06, unverified on hardware)

**No run before 2026-09-06 has a single frame of underwater depth.** Neither the 224-px observations, nor
the uint16 mm maps, nor even the **raw stereo pairs** that produced them went to disk
(`rov_gui/backends/hardware.py:3244-3352`). So they can neither be inspected after the fact nor
recomputed with different settings. The only visual trace, `ui_*.mp4`, is a recording of **a screen
already coloured and shrunk to widget size** — not a picture of the depth but a picture of a picture.
(In runs before 2026-09-06 that colouring was **per-frame auto-range + per-frame 64-knot equalization**,
so the same colour did not even mean the same distance. Now the default is a fixed rule, so
colour→distance can be recovered, but it is still a recording of the screen, not depth. See
`--fstereo-palette` below.)

```bash
./c3 gui --source hw --mpc --fstereo --policy --policy-observe --record-depth
# since 2026-09-11, launching without --policy-observe and picking LOW = None in the panel is the same
# (the flag is still parsed as an alias; in control runs --record-depth is used the same way too)
```

It lands in `<run>/policy_obs/` as **lossless PNG**. A lossy codec turns "no measurement" into "some
distance" along every hole edge, and that is the one distinction the observation's validity channel is
there to protect.

```
index.csv          seq, t_capture, obs_png, depth_png, rgb_jpg, rgb_dt_ms
meta.json          obs recipe (z_near/z_far, crop, resolution, grid) + builder describe() + counters
obs/000123.png     224x224x3 uint8 — exactly the tensor the policy consumed
depth/000123.png   640x400 uint16 mm, 0 = no measurement (off with `--record-depth-obs-only`)
rgb/000123.jpg     the C3 colour frame from when that observation was built (2026-09-07, ON by default)
```

**Colour is saved too (2026-09-07).** For every observation, `--record-depth` also writes the C3 colour
frame of that moment to `rgb/` — `export_run_html`'s **C3 RGB column is turned on by this** (no flag
needed; it finds `<run>/policy_obs` by itself). Three things are deliberate asymmetries:

- **JPEG** (q85, `--record-depth-rgb-quality`). The obs and the mm map are **measurements**, so they are
  lossless, but this is a picture for the eye, and the policy does not even look at colour. Lossless
  would be ~1 MB per frame [예측: 미실측 on this camera], so 2854 frames would fill the disk.
- **`rgb_dt_ms`**: colour (~30 Hz) and the recorded depth (~5 Hz) are **only nearly simultaneous, not
  simultaneous.** The difference between their own capture stamps is kept as a column, so that no claim
  lumps the two together under one stamp. **On the first hardware run this column caught a design
  defect**: at first the "latest colour" was attached, but the depth is a frame captured **before**
  FoundationStereo runs, so the latest colour was systematically later — median **+195 ms**
  (p5 +86 / p95 +365) [측정 2026-09-07,
  data/20260907/0907_133358_observe/policy_obs/index.csv, 514 rows],
  which is 2 cm at 0.1 m/s. Now the camera worker holds **~0.5 s of history** (`bus.LatestFrame`, capped
  at 16 frames) and the recorder picks the colour **closest** to the depth stamp → within one 30 Hz
  frame interval (~33 ms). **0907_133358 is a run from before the fix, so it still has +195 ms** (that
  value is written on every row).
- **The tap copies not a single byte when nobody is recording** (`bus.LatestFrame.wanted()`). It is the
  same contract as the camera worker's other optional taps, and if it fails, the pilot's screen and the
  depth recording are unaffected. Turn it off with `--no-record-depth-rgb`.

Given without `--policy`, it is **refused**. The recorder lives inside the policy worker, so it would
become a silent no-op, and you would find out only after finishing the whole dive that nothing was
saved — exactly the failure this feature exists to eliminate.

### Five safety rules (read before changing anything)

This code runs inside the policy worker's 20 ms tick while the vehicle is armed. Every rule is a defect
that the 2026-09-06 safety audit caught in the draft, and the top of
`rov_gui/perception/depth_record.py` carries the same list.

1. **Not a single filesystem call on the worker thread.** The writer thread resolves and creates the run
   folder **on the first frame**. The draft did the mkdir in `setup()`, which not only blocks the tick
   but — `runstore.run_dir` JOINs only within 90 s — puts the recording in **a different folder from
   the CSV** if the preflight runs long.
2. **Bounded queue, drop rather than block.** A recorder that puts backpressure on the control loop is a
   recorder that can hurt someone.
3. **A failure here does not end the dive.** Even the construction at the call site is wrapped, so a
   read-only mount cannot block the policy worker's startup.
4. **A failed write is not counted as a success.** When the disk is full, `cv2.imwrite` **returns
   False** instead of raising. A missing frame always shows up as missing and is never mistaken for a
   hole in the depth.
5. **Shutdown is bounded.** `close` waits ~1 s, not 6 s. Exceeding the worker stop budget parks the
   thread, and station shutdown freezes **with the vehicle still in the water**.

There is **deliberately no** frame-decimation (every-Nth) option: the worker submits the two frames of an
observation pair per inference, so "every N" phase-locks to one side of the pair, and the operator asks
for half and gets a **biased half**. Size is controlled with `--record-depth-max` and
`--record-depth-obs-only`.

### Side by side with the land training observations — `depth_compare.py`

```bash
python -m rov_gui.tools.depth_compare pair --run data/20260910/0910_141530_observe \
    --land-frame 250 --water-frame 88 --out /tmp/pair
```

There is **one** renderer. Fixed endpoints (read from the training zarr's `.zattrs`), no auto-range, no
histogram equalization, warm=NEAR, invalid is black and distinct from z_far. The colour rule is baked
into the image and saved alongside as `colormap.json` — the failure this tool exists to prevent is
comparing two images months later with nobody able to say what scale either one had.

**Compare in the default, `--domain obs`.** It is the only axis that the training builder
(`umi_handheld/build_dp_depth_zarr.py`) and the live builder (`rov_gui/perception/policy_obs.py`)
produce bit-identically, so equal pixel values meaning equal distances is guaranteed **by
construction**. `--domain metric` is for a **different question**, "do the numbers differ too?", and
uses a millimetre-linear ramp there.

`--strip` (default) · `--video` · `--frames`, and `stats` when you want only the numbers instead of images.

### The live panel in the same colours too — `--fstereo-palette` (2026-09-06)

The `--fstereo` depth panel is coloured **with the same function as the land videos**
(`rov_gui/depth_colour.py`, default `--fstereo-palette umi`):

```
TURBO · warm = NEAR · fixed 0.20-3.00 m · inverse (obs) spacing · invalid = black
red = 0.2 m,  dark blue = 3.0 m,  no per-frame adaptation
```

It is exactly the rule that `data_collection/make_depth_trajectory_video.py` (3-panel videos of land
episodes) and `depth_compare` use, and there is **one** implementation — the purpose of this default is
to set the station screen and the land videos side by side and compare them by eye. The colour bar is
drawn with the same formula too (NEAR at the bottom; the ticks 0.30/0.50/0.75/1.00/1.50/2.00 m are
identical to `depth_compare`'s ticks). The rule is written in the HUD note and in `fstereo.panel` of the
run meta.

| Flag | Meaning |
|---|---|
| `--fstereo-palette umi` (default) | the fixed rule above |
| `--fstereo-palette adaptive` | the image of 2026-09-02~09-06: JET + per-frame 2-98 percentile range + 64-knot equalization. Good contrast on flat scenes, but **the same distance gets a different colour in every frame**, so it cannot be compared with anything |
| `--fstereo-domain obs` (default) / `metric` | ramp spacing. `obs`=inverse (same spacing as policy channel 0), `metric`=metre-linear. Match it to the `--domain` of the land image you put beside it |
| `--no-fstereo-equalize` / `--no-fstereo-autorange` | meaningful only with `adaptive` |

**Only the image and the colour bar change.** The millimetres the cursor reads, the policy input,
FoundationPose and the `--record-depth` PNGs are identical either way.

### Reproducing a run as video — `render_run_scene.py` (2026-09-06)

Turns one recorded run back into a video of **a 3-D scene on the left + depth on the right**. Nothing
is re-simulated; only what the run left behind is joined **on a single clock**.

```bash
~/miniforge3/envs/fstereo/bin/python -m rov_gui.tools.render_run_scene \
    --run data/20260906/0906_191935_observe --speed 4
# -> <run>/scene.mp4  (1664x720)
```

| Panel | Content | Source |
|---|---|---|
| Left | AprilTag map (3-D) + ROV (actual geometry) + DP reference trajectory + zoom inset | `config/tag_map_full.yaml` (hash-checked) · `mpc_*.csv` · `policy_plan.csv`/`plans.jsonl` |
| Right | FoundationStereo depth, **shared colour rule** | `policy_obs/depth/*.png` (`--record-depth`) |

**Three frames differ, and that is the heart of this tool.**

```
CSV       datum frame · world FLU   (px, +py=left, +pz=up, +yaw=CCW)
plan      datum frame · NED         (x_ned, y_ned, z_ned)
tag map   anchor tag frame · NED    (+z DOWN, mat at z≈0)
vehicle   body FLU                  (rov_shape.body_to_map is the only sign flip)
```

CSV→plan is a **y·z·yaw sign flip** [측정 2026-09-06, this run, t 294.83: CSV
(0.34335, +0.47377, +0.11847) yaw −41.727° vs the anchor eta at the same instant (0.3318, −0.4782,
−0.1194) yaw +41.43° → residuals 4.5 mm / 1.0 mm / 0.3°]. Leave this out and the vehicle is drawn **below
the mat**, **0.93 m** away from its own reference (the first render actually did that). After the fix,
the knot0–COM distance is **0.055 m**, matching the configured anchor leash of 0.05 m — in effect,
physics cross-checked the frame transform. And the altitude above the mat is p50 **0.50 m**
(p5 0.21 / p95 0.75), which makes physical sense.

The clock is not assumed but **measured from the run**: a plan carries both `obs_t_mono` (the depth
frame's monotonic clock) and `obs_t` (the run-relative clock), so the median of their difference is the
offset, and the spread is printed alongside (this run: **0.0 ms**, 1426/1426 exact matches).

- The tag map is **not copied into the run but recorded only as path+sha1**, so a hash mismatch is
  **refused** (forcing it with `--allow-map-mismatch` marks every frame).
- The reference is the **body-origin trajectory** (the policy plans at the TCP, and the lever arm is
  removed upstream). The inset is centred on the body origin; the TCP is an amber dot.
- The vehicle is drawn with **yaw only** applied (same as the station panel). roll/pitch only as
  numbers.
- A plan is ~5 cm while the pool is 4 m, so it is not visible in the 3-D view alone → the
  `--inset-box` (default 1.0 m) zoom inset takes care of that difference in scale.
- **One depth recording spans several engagements.** The recorder is armed per station **process** while
  the CSV is per engagement, so in `0907_164659` 1160 frames over 343 s cover epochs 2/4/6/8/10. So the
  selection is not "whose plans fall inside the depth window" (all of them do) but **"how many frames fall
  inside this engagement's window"**, and frames outside the window are **dropped** (left in, `pose_at`
  would clamp them to the edge row and draw *a frozen vehicle under moving depth*). It prints how many
  were dropped, and `--engagement <조각>` ("<fragment>") exports each of the other engagements separately.
- **A run folder can hold several engagements, and `plans.jsonl`/`policy_plan.csv` are shared.**
  0906_191935 is one (19:19:35 epoch 2 + 20:00:50 epoch 4). The tool picks **the engagement whose plans
  fall inside the recorded depth window** and filters by `meta.policy.epoch` + the CSV `plan_id` column;
  specify it explicitly with `--engagement 191935`. Without that filter, **another run's reference** is
  drawn in the t<230 s window.
- The ROV **history line is not drawn** (current pose only), so that the only line on screen is the
  policy's reference. Tag ids are ON by default (off with `--no-tag-ids`).
- Verification: 13 tests in `rov_gui/tests/test_run_scene.py` (datum inverse transform · FLU↔NED flip ·
  **2 for engagement separation** · plan lookup · clock fallback · map-hash refusal · tag squares ·
  2 projection · 2 HTML template).

### Turning it around in the browser — `export_run_html.py` (2026-09-06)

Exports the same run as a **page you drag to turn around**. The browser draws the 3D, so the viewpoint belongs to the reader.

```bash
./c3 scene-html --run data/20260906/0906_191935_observe --embed --every 2
# -> <run>/scene_embedded.html  (one file, about 30 MB)

./c3 scene-html --run data/20260906/0906_191935_observe --every 2
# -> <run>/scene_html/index.html  (page 1.3 MB + frames/ 43 MB)
```

**Running via `./c3` is the default**, and the interpreter is **the same env** as `gui` (`rovgui-pose`)
— a tool only gets used if you can export straight from the environment the run was flown in. The export
uses neither PyAV nor zarr. The mp4 (`./c3 scene`) is the only exception: that env has no PyAV, so it
**falls back to mp4v and says so**; if you need h264,
`C3_SCENE_PY=~/miniforge3/envs/fstereo/bin/python ./c3 scene ...`.

```
+-----------+------------------+-----------------------------+
|  C3 RGB   | FoundationStereo | trajectory (drag=rotate)    |
+-----------+------------------+-----------------------------+
[ play | speed | frame slider | tag id | top-down | grid ]
```

Left-drag = orbit, right-drag (or shift) = pan, wheel = zoom, `R` = reset, space = play, ←/→ = one
frame. Deep links: a specific frame with `?f=200`, and even tag marking with `?mark=25,60`. Tag ids
appear in both the 3-D view and the top-down. The top-down inset **scales with the panel size** (25 % of
the short side, adjusted with `size` s/m/l) and **moves to the next corner when clicked** — which spot
gets covered depends on where the vehicle is, and that changes during a run. Tag ids are drawn in the
inset **only when there is room** (tag ≥ 15 px); smaller than that, only the outline remains (better
than overlapping numbers). **The top-down defaults to a 2 m box centred on the ROV** (the surrounding
tags and their ids are visible with it) — 1 m / 3 m / the full map can also be picked, but the full map
buries the vehicle among 140 labels, so it is not the default. The mp4 inset is the same box
(`--inset-box`, default 2 m) and draws the surrounding tags too.

**The reference is drawn in two layers.** The policy emits **16 steps** (≈1.07 s) at the training
stride of 66.7 ms, and the plan that goes to the controller is **6 knots resampled from those onto a
0.2 s grid** (spec A5) — which is why `policy_plan.csv` holds only 6 and the image showed 6 dots. The
16 steps are not saved anywhere, so they are **recomputed** from `action_raw` + the anchor +
`T_body_tcp` with the very `policy_frames.compose_plan` the run used. The evidence that this is a
reconstruction, not a re-derivation: the recomputed 6 knots match the recorded 6 knots to **0.000 µm**
[측정 2026-09-06, this run, 383 plans]. On screen, thin line + small dots = the policy's 16 steps, thick
line + large dots = the 6 knots actually sent down (off with the `raw 16-step` checkbox; the mp4 draws
them too).

**Marking objects on tags.** Pick an id in the `mark tag` dropdown or **click a tag in the 3-D view**,
and a small cylinder (∅80 mm × 150 mm) stands on that tag — it stands along the tag's own normal, so it
sits on the mat, and it also appears as a dot in the top-down. Clear with `clear marks`. **No CDN and no
framework** (the data is inline in the page — from `file://` sibling JSON cannot be fetched), and the
frames are JPEGs in the folder next to it.

**For a single file, `--embed`.** It puts the frames inside the page as data URIs, so depth shows up
with just the one `index.html` (`<run>/scene_embedded.html`).

```bash
... --embed --every 2      # 1427 frames, about 26 MB, single file
... --embed --every 4      # 714 frames, about 14 MB
```

| Form | Page | Needed alongside |
|---|---|---|
| default (folder) | 0.8 MB | `frames/` 45 MB — **the images only appear if you copy the whole folder** |
| `--embed` | about 18 kB per frame (webp q80 13.8 kB + base64 1/3) + hover 64 kB | none |

**Hover the mouse over the depth and the distance at that pixel appears** (2026-09-08). A chip that
follows the cursor writes `0.601 m  601 mm · px 268,218 · depth/000000.png` — the value, the pixel and
even **the file the value came from**. It hides over the header and the colour bar.

That number is **not read back from the colour.** For every frame the page carries a second image, a
**lossless WebP** that puts the recorded mm map's high byte in R and its low byte in G
(`encode_values`); the browser reads just that one pixel from a canvas. The colour is not inverted
because the obs palette is 8 bit and `dz/dt = 4.667 z²`, so **one step is 0.16 m at 3 m**
[유도: `depth_colour.palette_pos`], and lossy compression is piled on top of that — a number read that way
becomes a measurement of the renderer, not depth. `0` is written as `no measurement` (it does not mean
far; same rule as the legend).

The value layer is **48 kB** per frame [측정 2026-09-08, `0908_180453` 640×400], about 64 kB as base64
with `--embed`, so the page grows roughly **3.5×**. If you do not need it, `--no-hover`.

```bash
./c3 scene-html --run <RUN> --embed --every 1              # with hover (default)
./c3 scene-html --run <RUN> --embed --every 2 --no-hover   # lighter, without the value layer
```

**In the folder form, hover does not work from `file://`** — the browser blocks reading the pixels of
sibling images (canvas tainting). The value files (`frames/v*.webp`) are still used as they are, so
serving the folder over http works, and when it does not work, the chip writes the reason. If you need
one file, `--embed`.

The geometry is **carried along, not assumed** (`depth_geom`): `depth_panel.fit_into` rounds `nw`/`nh`
separately per axis, so a 320×240 grid ends up with different scale factors, 533/320 horizontally and
400/240 vertically — inverting with one shared scale factor is right only on the 2026-09 grid
(640×400, 1:1), and on the rest it **points at the wrong column while looking fine**.
`test_the_hover_geometry_inverts_the_panel_layout` checks **every pixel** of the image strip against
the `cv2.resize` result for four grids.

Either way, **the 3-D scene · tag map · ROV · DP reference · all readouts are inside the page** — open
`index.html` on its own and all of that still appears; only the image columns stay empty, with the
reason written (confirmed by actually trying it). Viewing **reads none** of the original run data
(`policy_obs/`, CSV, plans.jsonl, the tag-map yaml); it was consumed at export time.

**Runs without depth are drawn too.** A run flown without `--record-depth` loses only the right-hand
column; the pool, vehicle and reference are in the CSV and the plans, so they come out as usual — the
timeline then uses **the pose log thinned to ~5 Hz**, and every frame says "this run has no depth". If
the folder has **0 pose rows** (that is what the stream recorder's companion CSV looks like), it
refuses and **also lists the runs in the same date folder that do have content**.

**The C3 RGB column is empty for this run.** `--record-depth` saves only the observations and the mm
maps, and the station has no writer that saves the colour stream **with frame stamps** (`ui_*.mp4` ·
`c3_*.mp4` have only a start time and a nominal fps, so they cannot be joined). Once runs save colour,
it attaches to the same clock via `--rgb-dir` (policy_obs layout: `t_capture` in `index.csv` + images).

### Saving the raw stereo pairs — `--record-stereo` (2026-09-06, unverified on hardware)

`--record-depth` saves FoundationStereo's **output** (observation + mm map). Settings cannot be changed
from the output, so with that alone the run is **forever tied** to the `iters`/`scale`/checkpoint at
capture time. This flag saves the **input** (the rectifiable left/right mono pairs) and unties that
knot — it gives underwater runs the property land collection had from the start (`oakd_record.py:524`,
lossless mono pairs → `replay_foundation_stereo.py`).

```bash
./c3 gui --source hw --mpc --fstereo --record-stereo            # every pair FoundationStereo computed
#   (during a policy mission with --policy-fs-schedule only, just the 2 frames/cycle the policy used — the pairs in between
#    are discarded without being computed, leaving holes in frame_seq; fstereo.policy_fs_schedule in stereo/meta.json marks this)
./c3 gui --source hw --mpc --fstereo --record-stereo \
         --record-stereo-every 2.0 --record-stereo-max 2000     # sample long dives
```

```
<run>/stereo/
  index.csv          seq, t_capture, frame_seq, left_png, right_png
  meta.json          rig (rectification geometry + calib provenance) + the FS settings the run **actually used**
                     + counters + mono geometry
  left/000123.png    RAW mono — exactly as the camera delivered it
  right/000123.png   its partner from the same **device exposure** (frame_seq is the proof)
```

- **Saved RAW (unrectified).** Rectification is a function of the calibration, so baking the current
  calibration in now would lose the **ability to apply an underwater recalibration** later. The rig used
  at the time is in the meta, so that run's depth reproduces as it was, and a new calibration can also
  be applied to the same pixels.
- **Lossless PNG.** With a lossy codec the regenerated disparity would become a function of the codec,
  and a settings comparison would turn into a compression comparison.
- **Size is the lever.** A 640x400 pair is **296 KiB** [측정 2026-09-06, 10 pairs actually recorded
  with this recorder]; at 7.26 Hz that is **~2.1 MB/s**. The default cap of 6000 pairs ≈ 1.7 GB ≈ 14 min.
- **The cost of `--record-stereo-every`**: a policy observation is built from **two consecutive depth
  frames**, so with sampling, individual frames can be regenerated but **the observation the policy
  actually consumed cannot be reconstructed**. For a run you will check against `policy_obs/`, keep
  every pair.
- Refusal rules: refused without `--fstereo` (there is no worker supplying the pairs), refused without
  `--mpc` (nothing knows the run folder), and `--record-stereo-max <= 0` · a negative interval are
  refused as well.
- The 5 producer-side safety rules are **shared** with `--record-depth` (`perception/frame_record.py`;
  `test_depth_record.py` pins both sides together). No filesystem calls on the worker thread, drop
  instead of block when the queue is full, a failure does not end the dive, a failed write is not
  counted as a success, shutdown capped at 1 s. Verification: 19 tests in
  `rov_gui/tests/test_stereo_record.py`.

**Not there yet**: an **offline tool** that reads `<run>/stereo/` and rebuilds the depth with different
settings. The format is self-describing (see `regenerate_with` in the meta) and `infer()` in
`perception/fstereo.py` is the chain as is, but the script has not been written yet.

## Flight modes — arm goes into MANUAL

If you arm in a mode other than MANUAL (STABILIZE / DEPTH HOLD / POSHOLD), **the autopilot spins the
thrusters by itself to hold attitude and depth.** It does so even when the pilot input is 0. That is
the reason the mode exists, and out of the water it looks like "the motors suddenly spin on their own".

ArduSub **keeps whatever mode it was last in.** So if the mode is left untouched, the ARM button
inherits that mode, and COMMAND ENABLE + ARM alone spins the thrusters (hardware report, 2026-08-07).

So **the ARM button first requests MANUAL right before arming** (`--arm-mode`, default `MANUAL`; with
`keep` it inherits as before). MANUAL is the only mode that stays still when armed with an input of 0.

**The mode is not locked.** It can be changed at any time, before or after arming, with the teleop
panel's `MANUAL | STAB | DEPTH` buttons — that is, **STABILIZE is entered only when the pilot presses
it**.

* What is highlighted is **the mode the vehicle reported via HEARTBEAT**, not the mode we requested.
  ArduSub refuses some transitions while armed; had we displayed the request, the screen would say
  MANUAL while the vehicle is in STABILIZE — exactly the state in which ARM spins the motors. Same rule
  as `MOTORS ARMED`. When telemetry drops, the highlight goes off too.
* When a mode other than MANUAL is active, the button shows in a **warning colour**. It means "right
  now this vehicle can spin the thrusters by itself".
* On a disarm→arm transition the mode is printed to the log:

```
[warn] ARMED in STABILIZE — the autopilot will run the thrusters by itself
       to hold attitude/depth, with no stick input.
```

> This was originally a **feature deliberately left unimplemented** ("the station is a command source,
> not an authority; the mode is QGC's job"). Hardware overturned that — a station that can arm but
> cannot decide what armed means is not safer, just less predictable.
>
> **The gamepad's `arm` button is the exception.** It goes straight to the vehicle as a bitmask, so we
> cannot intercept it, and it therefore inherits the mode (same as QGC). If you are going to arm from the
> gamepad, check the mode first.

### Engaging in STABILIZE — the `engage.require_mode` list (2026-09-08, unverified on hardware)

> **The second blocker the same day (fixed)**: `engage refused: solver not real-time (acados, probe 79.7 ms)`.
> The probe was the wall-clock time of **1 cold solve** at controller creation (right after the pid →
> mpc_tuned switch, while FoundationStereo was capturing its CUDA graph), and that value stayed frozen
> for the whole session and refused START 9 times. Now the probe is the **median** of 5 solves after
> 1 warm-up (the cold value is kept separately in meta `controller.probe_ms_first`), and right before
> refusing it **measures again** and refuses only if it is still slow (`HwDobMpc.reprobe`,
> `test_control.py::test_engage_reprobes_a_slow_solver_before_refusing`).
> **A bug that blocked the first hardware attempt on 2026-09-08 (fixed)**: after the STAB button was
> pressed, START was refused with `flight mode is mode -1,
> need MANUAL|STABILIZE`. ArduSub's STABILIZE is custom_mode **0**, but the HEARTBEAT parsing was
> `custom_mode or -1`, which turned 0 into −1 and read it as "mode -1" (`backends/hardware.py _arm_state`).
> The on-screen `ARMED mode -1` · `AUTOPILOT DRIVING (mode -1)` were the same misreading, and the
> 4 "mode -1" refusals of 2026-08-23 were also STABILIZE. Now 0 is read as STABILIZE
> (`test_offline.py::test_heartbeat_custom_mode_zero_is_stabilize_not_unknown`).

Until now, if the vehicle was in STABILIZE when you picked POLICY (or any mission), picked the
low-level controller and pressed START, it was refused with `flight mode is STABILIZE, need MANUAL`.
The gate key takes a **list**, and its default is `[MANUAL, STABILIZE]`: press MANUAL or STAB in the
teleop panel and START, and the low-level controller runs under the mode the vehicle reports (the
design review recommended a MANUAL-only default; the operator chose both — 2026-09-08):

```yaml
engage:
  require_mode: [MANUAL, STABILIZE]   # default since 2026-09-08 (operator decision) — can be narrowed to MANUAL alone
  mode_settle_s: 2.0                  # START is refused for this long after the mode changes [예측]
  stabilize:
    yaw_axis: hold                    # hold | torque  (rate is not implemented → refused)
```

Behaviour rules (all in `rov_gui/control/workers.py`; tests `test_control.py::test_engage_mode_gate_*`,
`test_stabilize_yaw_axis_*`, `test_dobmpc_is_refused_outside_manual`):

* **Pinned**: a run is tied to the mode it engaged in. Even a change to a mode that is on the list disengages with `flight mode left
  STABILIZE (now MANUAL)` — because the yaw handling and the records boundary are set at engage time.
* **dobmpc/dobmpc_tuned are MANUAL-only** (regardless of the list). In STABILIZE, ArduSub's attitude
  and yaw-rate loops drive the thrusters as well, and the EAOB absorbs that torque as a disturbance
  (w_hat K/M/N).
* **`mode_settle_s`**: when entering STABILIZE from MANUAL while armed, ArduSub does not relax the
  attitude target or the rate-PID integrators (`control_stabilize.cpp stabilize_init()`, 4.1/4.5
  source) → there is a 2 s wait to avoid the kick right after the switch. 2.0 is [예측] and is to be confirmed
  with ATTITUDE/Des* logs.
* **yaw axis**: in STABILIZE, ArduSub reads MANUAL_CONTROL's r as a **yaw-rate command**. |r| ≤ 100
  counts (= station axis 0.10 = N 2 N·m @ yaw_nm 20) is a dead zone, so it is ignored, and after 250 ms
  the current heading is held (`ATC_ANG_YAW_P 6`, slew 60°/s); above that it is a rate servo (~68°/s at
  cap 0.5 with PilotGain 1, ~32°/s with the boot default of 0.5; a rate PID with an integrator). The
  station's NMPC/PID computes N as a **torque**, so the plant assumption is wrong wholesale. Hence the
  default is `hold`: the yaw axis is sent as 0 (CSV `ax_yaw` 0, `uN` keeps the solver value,
  `note_applied` has N=0) and the autopilot's heading hold holds the heading → **heading_follow ·
  circle · the policy's dyaw cannot turn the vehicle** (warning at engage). `torque` is an experimental
  mode that sends N as is; hunting at the dead-zone boundary is predicted. Note alongside that the yaw
  command p50 in the hardware records is 12–43 counts, so even in MANUAL it was effectively inside the
  T200 deadband.
* **surge/sway/heave are byte-identical to MANUAL** (`control_stabilize.cpp`: forward/lateral/throttle
  pass-through, not depth hold) → `axis_gain` and `vehicle_net_buoyancy_n` stay valid as they are. But
  the 4 vertical thrusters are shared with the roll/pitch PID (IMAX 0.444, no anti-windup while armed)
  with no priority, so in phases with a pitch moment, such as gripper–rock contact, heave authority
  drops and a pitch kick comes the moment it lets go.
* **roll/pitch is a benefit**: in MANUAL the station discards K/M, leaving only the pure restoring
  force, whereas in STABILIZE ArduSub holds the vehicle level with its IMU loop (target 0°, since
  RC1/RC2 are not sent). The C3 is fixed tilted 43.3° down, so vehicle pitch directly means a shift in
  the policy's observation distribution, and the 5-dim labels (`dp-action-5d`) also assume a "level
  vehicle", so this favours the policy side. The reference is AHRS level (`AHRS_TRIM`), not tag-PnP
  level — the delta between the two is to be measured in the first STABILIZE session.
* **Records boundary**: meta `run.flight_mode_at_engage / flight_modes_seen / require_mode / mode_settle_s /
  stabilize{yaw_axis, yaw_axis_held}`, events `ENGAGED (...) vehicle flight mode STABILIZE`, warning
  lines. Old runs without these fields = MANUAL-only gate. **Never pool STABILIZE runs with MANUAL
  runs** (yaw means something different).
* **Incompatible with the 6-DoF variant (2026-09-26)**: with `engage.attitude_axes.enabled`, a
  STABILIZE engage is **refused** — in STABILIZE, MANUAL_CONTROL's s/t are lean-angle targets
  (`get_pilot_desired_lean_angles`), so the NMPC's torque model would be reinterpreted as an angle
  servo. `stabilize.rp_axis: hold` is the only value (ArduSub holds the vehicle level and the station
  does not send K/M).
* **E-STOP is not a 'stop' in STABILIZE**: sending neutral = level and heading hold with the thrusters
  alive, and only the position drifts. A real stop is DISARM. (Changing what E-STOP means is a matter
  for a separate safety review.)
* **PilotGain**: ArduSub multiplies every axis by the joystick gain (0.5 at boot), but the station
  does not read it (KNOWN_ISSUES). The yaw-rate figures and the axis authority are all proportional to
  that value.

First STABILIZE session, in order (none of it measured): ① on the bench, dump the firmware version
(exclude 4.1.0 — there the STABILIZE attitude target locks to the attitude at the last yaw input) ·
PilotGain · `RC4_DZ` · `ACRO_YAW_P` · `ATC_*` → ② in an armed hover, the MANUAL↔STABILIZE switch kick
(ATTITUDE 3 s) → ③ with the station not engaged, send r steps of 0/±80/±120/±200/±500 counts for 3 s
each and measure the dead-zone boundary and °/s per count → ④ in the order PID → mpc, compare station
keeping and a heading step, each paired with a MANUAL run from the same session (heading RMS, `ax_yaw`,
pitch RMS, heave deadband). [Source: a reading of the ArduSub source Sub-4.1/4.5 + 2-lens verification
24/30, workflow stabilize-mode-llc 2026-09-08]

## Camera tilt

Tilts the mount of the ROV's default RGB camera up and down. It follows **the same pattern as the gripper**, not
the lights: ArduSub handles `mount_tilt_up/down` in the button **repeat** path, so the mount keeps moving
for as long as the bit is set (unlike the lights' "one step per press"). Hence press-and-hold buttons and
the `,` / `/` keys, plus `LEVEL` (`.`, `mount_center`), which returns to level with a single press.

The angle readout is a number **only when the vehicle sends one**:

* If `MOUNT_STATUS.pointing_a` arrives, that value is used, in degrees (the angle the vehicle believes).
* If not, then only if `--tilt-servo <ch>` was given, that channel's PWM is scaled into the `--tilt-min-deg` /
  `--tilt-max-deg` range and shown, **marked `[유도]` ("derived")** — PWM is not an angle unless the
  servo range is known.
* If neither is available, it writes `no angle reported (open loop)` and **does not draw the command as if
  it were an angle** (the same rule as the gripper).

The button numbers (9/10/7) come from this vehicle's QGC button assignment, and the function numbers
(23/22/21) are the ArduSub `JSButton` enum **[스펙]**. On connect, `BTNn_FUNCTION` is read and
cross-checked; on a mismatch, **that button is not pressed**, and both possibilities (the vehicle's
assignment differs, or our table is wrong) are logged together. This has not been checked on hardware
yet.

## Screen recording

Captures the **UI widget tree** with `QWidget.grab()` (not the X11 root window). No other window can end
up on top, there are no Wayland capture-permission issues, and it also runs offscreen, so it can be
tested. In exchange it renders the widgets one more time, so it is not free → 12 fps by default.
grab runs on the GUI thread, encoding on a separate thread, and the queue between them is **bounded and
drops on overflow** (freezing the piloting screen to protect the recording is not an acceptable trade).
The number of dropped frames is recorded in the `.json` sidecar — because if a file labelled 12 fps was
actually 7 fps, every time measured from that file is 40% wrong.

### The codec is H.264 — opens directly in VS Code (2026-09-30)

`ui_*.mp4` and the per-feed `c3_*.mp4` are written **as H.264 by piping into the system `ffmpeg`
(libx264)**. They used to be OpenCV `VideoWriter` `mp4v` (MPEG-4 Part 2), but the ffmpeg bundled with
VS Code has an h264 decoder and no mpeg4 decoder, so the preview ended in *"An error occurred
while loading the video file"*. pip cv2 has no H.264 encoder and cannot open `avc1`, so ffmpeg is
called directly. If no ffmpeg with libx264 is found, it falls back to `mp4v`; those files do open in
VLC/mpv. The sidecar `"codec"` (`h264` | `mpeg4`) records which one it was. A sidecar without this
field belongs to a file from before 2026-09-30, and that file is `mp4v`.

To view an old file in VS Code, re-encode it (the frame count stays the same):

```bash
f=data/<date>/<run>/ui_XXXX.mp4
ffmpeg -i "$f" -an -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p \
  -colorspace smpte170m -color_primaries smpte170m -color_trc smpte170m \
  -movflags +faststart "${f%.mp4}.h264.mp4"
```

### Per-feed recording

**The recording toggle is on each video panel** (`REC` → when pressed, `● REC C3 RGB`). With all four
in the header, fitting them into 42 px clipped the "ROV" lettering — the place where a feed's name fits
is on the feed itself. Only the whole-window recording (`REC UI`) remains in the header.

The name is the same in three places: panel title · button · file name (`c3_rgb_*.mp4`, `c3_depth_*.mp4`,
`default_rgb_*.mp4`). The window recording and the feed recordings are each **independent files**. The UI
is the record of what the pilot saw and did; a feed is the picture itself — they are different things,
so they are switched on separately.

### Naming UI recordings (2026-08-14)

Type a name at any time into the **input field to the left of** `REC UI` in the header, and the next
recording is saved as `ui_20260814_190532_squaretest.mp4`. Leave it empty and it is
`ui_20260814_190532.mp4`, as before. The sidecar `.json` gets the string **exactly as typed, not the
slug**, as `"name"` (the slug loses spaces, dots and slashes).

* **An always-present input field rather than a modal popup** is a safety decision. This window is the
  only key handler (see "Keyboard" below), so a popup appearing while driving would swallow W/A/S/D
  outright. The field takes focus **only when clicked** (`ClickFocus`; Tab never brings focus to it),
  and the moment it takes focus it **drops every axis to 0** (the key releases cannot be received, so
  this keeps an axis from being left stuck on) — the same pattern as the number fields in the
  trajectory panel.
* **Esc is always E-STOP**. QLineEdit normally swallows Esc as edit-cancel, which would make the very
  key the pilot presses in an emergency do nothing, depending on where the caret is. An event filter
  drops the focus and **passes the key straight through to the window's E-STOP**
  (`test_recording_name_field_never_takes_escape_away_from_estop`).
* Enter commits + releases focus. While recording, the field is locked — the name is baked into the
  file name at start time, so editing it midway would make it refer to something other than that file.
* The field is **not cleared** when a recording ends: a session is usually several takes of the same
  thing, and having to retype it every time makes the take names drift apart.

### Making sure no white widget is left (2026-08-14)

Some of Qt's default widgets are **light-themed widgets**, so if left untouched they become the
brightest objects on this palette. A white bar next to a screen built for reading murky video in a dark
cabin breaks dark adaptation — so three things are written out explicitly in `theme.py` (pointed out
by the pilot: "the Recording name entry, the log scroll and the like are white bars"):

* **`QLineEdit`** — the recording-name field. It follows the same language as the combo boxes and spin
  boxes (panel fill + the same border + accent on hover).
* **`QScrollBar`** — the MISSION LOG scrollbar. This **has to be in the app stylesheet**: the scrollbar
  is a child widget, so the widget's own stylesheet does not reach it, and the log view has a stylesheet
  of its own. A thin bar with no arrow buttons — the log is scrolled with the wheel.
* **`QPlainTextEdit`** — the log body is a **dark well** one step darker than the panel (the same
  background as the progress bars and plots), so the scrolling area does not read as yet another
  embossed control.

Calling `setStyleSheet` on a single widget is still avoided — for that widget it **replaces** the app
sheet and falls back to the Qt default look (the trajectory panel got bitten by this once). If a size is
needed, use `setFixedHeight`/`setMaximumWidth`; if only the padding differs, solve it inside the app
sheet with an `objectName` (`#Compact`) rule.

### One data root, by date (2026-09-14)

`data/` is the only root. There used to be two — gantry/ZED/survey captures and training artifacts in
`data/`, everything the station writes in `sessions/`, and inside `sessions/` nine inconsistently named
trees (`low_level_controller_data`, `policy_observe`, `land_dryruns`, `dryruns`,
`nav_runs`, `mpc_runs`, `ui_recordings`, `pose_meshes`, `demonstration_NNNN`). At the operator's
request (2026-09-14) they were folded into **one folder, by date**:

```
data/
  YYYYMMDD/
    MMDD_HHMMSS                  water run (unchanged)
    MMDD_HHMMSS_observe          LOW None / --policy-observe (formerly sessions/policy_observe)
    MMDD_HHMMSS_landdry          --land-dry-run (formerly sessions/land_dryruns)
    MMDD_HHMMSS_dryrun           policy_dryrun.py synthetic run (formerly sessions/dryruns)
    MMDD_HHMMSS_obj              on-site object reconstruction (formerly sessions/pose_meshes/obj_*)
    MMDD_HHMMSS_train_<task>_<exp>   DP training run (formerly data/outputs/Y.M.D/H.M.S_*)
    train_logs/                  training stdout (tools/run_ablations.sh)
    demonstration_NNNN           umi_handheld take
    YYYYMMDD_HHMMSS_<suffix>     gantry/ZED/survey capture (unchanged)
  checkpoints/                   one folder holding the best ckpt of every training run (symlink + .json)
  MIGRATION_20260914.json        full mapping, old path → new path
```

* **The kind is the leaf suffix.** The pooling guard that the old separate trees (policy_observe /
  land_dryruns) used to provide — "a bench run cannot join a water-run folder even if it starts within
  90 s" — is now kept by `runstore.run_dir(base, kind=...)` **JOINing only within the same kind**.
  `MpcWorker._run_tree()` returns `runstore.Tree(log_dir, kind)`, and a water run is kind `""` (no
  suffix). Scanners tell the kinds apart with `runstore.leaf_kind(name)` — `plot_runs.py` draws only
  water leaves (the `kind:` key can pick other kinds), and the policy tools (`policy_session_replay`,
  `policy_yaw_vs_can`) find the `_observe` leaf even when given only `0908_173106`.
* A tool that needs **one execution = one folder** (`policy_dryrun.py`) opens a new folder with
  `run_dir(join=False)`, and the writers that follow join that folder (since it is the most recent
  one). The old way (a sub-root `dryrun_<stamp>/` per call) is gone.
* **The flat trees from before runstore (before 2026-08-14)** were folded with the same rule: the
  `ui_*`/`nav_runs/*`/`mpc_runs/*.csv` that follow one another within 90 s were gathered into a single
  `MMDD_HHMMSS/` named after the earliest stamp (nav into a `nav_HHMMSS/` subfolder), and file names
  were left as they were.
* **Every old citation was rewritten to the new path** (`tools/rewrite_data_citations_20260914.py`,
  292 lines of config comments · docs · journal · memory). This is because of the citation rule for
  measured values (the path must actually exist); the mapping is `data/MIGRATION_20260914.json`. The
  move script is `tools/migrate_data_layout_20260914.py` (`os.rename` only; file contents were not
  touched).
* **One folder for the DP best checkpoints** — `data/checkpoints/<YYYYMMDD_HHMMSS>_<task>_<exp>.ckpt`
  is a relative symlink to the file inside that run's `checkpoints/`, and the `.json` next to it records
  how it was chosen (`auto`: the workspace updated the train_loss top-1 during training; `selected`:
  picked by a person with `tools/publish_ckpt.py <run> --epoch N --why ...` — auto does not overwrite
  it afterwards). The topk/latest inside the run folder are exactly as before.
  `hw_mpc.yaml policy.ckpt`, `state.POLICY_CKPT_DEFAULT` and the ckpt picker's start folder all look
  here.

### Everything one run produces goes into one folder (2026-08-14)

A single START used to scatter files across three places — `sessions/ui_recordings` (the name before
2026-09-14), `sessions/nav_runs/<stamp>/`, `sessions/mpc_runs` (also before 2026-09-14). The same run in
three places, and the only way to see how they related was to match the timestamps by eye. Now:

```
data/20260814/0814_184148/
    ui_20260814_184149_squaretest.mp4   what the pilot saw (+ .json sidecar)
    c3_depth_20260814_184150.mp4        feed recording (+ sensor JSONL, same stem)
    mission_log.txt                     the whole MISSION LOG, wall clock
    nav_184148/                         map.json fixes.csv detections.csv frames.csv
                                        controller.json  ← M/C/D + that run's gains
    mpc_184151.csv  mpc_184151.meta.json  the controller's record
    events.log                          engage / refuse / disengage
```

* Why the leaf folder is named `MMDD_HHMMSS` (2026-08-14, two operator requests). **Date**:
  run folders are often **taken out of** the date directory (attached to a message, a plot-script
  argument, backups), and `1841` alone does not tell you when it is from. The parent still
  carries `YYYYMMDD`, so the duplication is free. **Seconds**: a pool session makes several runs in 1
  minute, and minute resolution silently merged them.
  **`HHMM/` folders created before that are left as they are** — they are not moved. One side effect:
  joining (below) only happens between new names, so an old `1841/` is never reopened for a new run
  (the safe direction).

* The roots are `--rec-dir` (default `data`) and
  `hw_mpc.yaml: log_dir`. Both are **the root of the tree, not the folder that gets written**.
* The three writers (the screen recording on the GUI thread, the window's nav recording, the MpcWorker
  CSV on its own thread) have to reach the same answer without knowing about each other. Passing a
  shared run id would take a handshake, and a handshake has a failure mode where "one writer misses it
  and its file silently lands somewhere else".
* **What changed when seconds were added**: with minute buckets the three writers each *computed*
  `1841`, and the join window existed only for the case of a run that straddled a minute boundary. At second
  resolution :48, :49 and :51 make different names, so `run_dir()` no longer computes; it **looks for the
  run in progress in today's folder and joins it**, and creates a new one only when there is none. The
  filesystem is still the only state and there is still no handshake, but **`JOIN_WINDOW_S` (90 s) now
  does the job of holding every run together**. A writer that starts more than 90 seconds after the
  preceding writer gets its own folder (the same thing that used to happen at minute boundaries in the
  minute-bucket days, just in more directions).
  It only ever merges; it never splits or moves.
* **The "in progress" test** is the **later** of the folder's **name time** (when the run started)
  and the **directory mtime** (updated when a file is *created*). Going by the name alone, a run would
  be over after 90 seconds while writers are still arriving; going by mtime alone, calls that pass
  `when=` (tests · replay) get a nonsensical answer, because the actual mkdir time and the time being
  asked about differ.
* **Why `nav_<hhmmss>/` is a subfolder**: `plot_nav_run` · `build_tag_map` take a "run directory" and
  look for the `map.json`/`fixes.csv` inside it. Flattening it would break both tools for every past and
  future record.
* **Old data was not moved (as of 2026-08-14)** — config comments · docs · memory cited those paths as
  **the provenance of measured values**, so moving the paths would have turned the citations into dead
  references. In the 2026-09-14 single-root migration the data was moved and the citations were
  rewritten along with it (section above).

### Folders with only an `events.log` — why they appeared, and why they no longer do

On 2026-08-14, seven folders under `20260814/` (`2009 2011 2037 2039 2047 2053 2111 2118`) were just a
one-line `events.log`. That line is, in every case, the **build fingerprint** left by
`MpcWorker.setup()`:

```
2026-08-14 21:18:46 ready: dobmpc, path-following, no geofence, mission=station @tag79
```

That is, it means **"the station was launched at that time, and nothing was flown"** (no engage, no
recording). `_log_event` calls `run_dir()`, and `run_dir()` creates the folder, so that single line
created an entire folder. `2111` holds the same line 72 times in 4 bursts — meaning `setup()` ran that
many times at that moment (the shape you get from launching the process repeatedly), and it is **a
record of executions, not data**. Deleting them loses nothing.

The fingerprint itself is worth keeping (on 2026-08-14 a whole pool session was spent on "a refusal
already removed from the source" — what was running was simply an old process, and nothing on screen
said so), so **it was deferred instead of removed**: it appears in the MISSION LOG immediately, and
goes into the file as the first line of that run folder's `events.log` **once a real event to write
alongside it occurs**. A session flies several runs, so it is attached again **for every new folder**
(`_log_event(defer=True)`).

### Trajectory figures for presentations — just give it the date (`plot_runs.py`, 2026-08-15)

```bash
./c3 plots                              # the dates in config/traj_plots.yaml
./c3 plots --dates 20260814 --list      # first just see what would be drawn
./c3 plots --dates 20260814_212021      # only that one CSV
./c3 plots --dates 20260814 --theme dark
python -m rov_gui.tools.plot_runs -c my_plots.yaml       # same thing without the wrapper
```

**Only dates** go into `config/traj_plots.yaml`. You can choose the resolution —
`20260814` (that whole day) / `20260814_2120` (that run folder) / `20260814_212021` (that one
CSV) / `all` / `latest`. Figures land one per run at

```
figures/trajectories/<date>/<date>_<run>_<controller>_<hhmmss>.png
```

The figure carries only **the vehicle + reference (black dashed line, on top) + tracked object (if
any) + one error + a key**, and no date — the date is in the file name, so stamping it onto the figure
as well would fight with the slide caption.

**Colour carries two things at once.** The *hue* says which track (blue = vehicle, orange = tracked
object), and the *lightness* within one hue says when (the start of the run is light, the end dark). So
the strip below is neither a legend nor a colour bar but both — one gradient row per track, sharing a
single elapsed-time axis. The reference gets no ramp: it is not a measurement but a **command**, and
tinting it would imply that it, too, flowed in time
(with `time_color: false`, a single colour + an ordinary legend, as before).

Two things make it readable (2026-08-23: in the first version the colours were too similar to read
the time from).

* **The ramp is wide and even.** Instead of sticking to one hue, it walks out into analogous
  hues — blue from cyan-blue to indigo, orange from amber to a deep red-brown. And `_cmap`
  **reparameterizes the ramp by OKLab arc length**, so that 1 minute shows the same amount of
  colour change wherever it falls. The old orange ramp spent its first half almost standing still
  (segment ΔE 5.7, 5.1 / 13.1, 13.8) — meaning the first 2 minutes of a run were one colour.
  It is now pinned **by measurement** at end-to-end ~53 / 49 and a quartile spacing of ~14.
* **Round-time dots on the path** (one every 60 s) **are labelled with their seconds.** When a
  trajectory goes around the same spot for several laps, the last lap covers the previous four,
  and no palette can fix that. A ringed dot survives that overdraw, and once there is a label,
  **there is no need to judge colour at all** (bonus: the spacing is the speed). A tick with no
  sample nearby (a released stretch, `segment: engaged`) is **dropped, not pulled** to a nearby
  row — putting the 120 s label at the 141 s point writes a wrong time in a spot that looks
  right.
  `time_ticks: false` turns them off, and `time_bands: N` quantizes the ramp into N steps.

**The controller in the file name comes from `meta.json`.** The station writes the log as
`mpc_<hhmmss>.csv` whether it is PID or MPC (the prefix refers to the **worker**, not the controller) —
using the CSV stem as is would ship a PID result as `..._mpc_....png`.

Three things are the reason this tool exists.

* **Frame.** The CSV is in **that run's datum frame** — the pose at the moment of ENGAGE becomes
  (0,0,0)/yaw 0 (`workers.py _datumize`) and the columns are mirrored to world-FLU. Overlay two runs
  from the same afternoon as they are, and two squares that were really on opposite sides of the pool
  land on top of each other. So every run is **lifted back into the map (tag) frame** with
  `hardware.datum_tag_frame` from `meta.json` (the same rigid transform as `_to_map_xy`). A run with no
  datum is not silently drawn in some plausible spot; it is **skipped with the reason stated**
  (forcing it with `frame: datum` stamps a "DATUM FRAME" watermark onto the figure).
* **Axes.** Every figure is locked to **the same pool box**. There is no autoscale anywhere.
  The box comes from the same derivation the station uses — outermost tag **edges** + `pool_margin_m`
  (`geometry.py pool_from_tags`) — which gives, from `config/tag_map_full.yaml`,
  x [-0.723, +1.301] y [-2.120, +2.317] = 2.02 × 4.44 m. If a run's
  `nav_*/map.json` records a different box, a **warning is printed** instead of changing the figure.
* **Tracked object (`--pose`, `object_nav.py`).** `obj_px/py/pz` are in **the same datum-frame world
  FLU** as `px/py/pz` (workers.py:341), so they go onto the map with exactly the same transform as the
  vehicle — which makes the two comparable down to the mm. Drawn as **dots**, not a line: the tracker
  reseeds, so drawing a line would draw motion that never happened. All three filters exist because of
  what the log actually does, and how many rows each one removed is **printed to the console**
  (nothing is dropped silently) —
  `live_only` (stale/cold/lost are holding the last value, not a measurement),
  `pair_exact_only` (**the camera extrinsic cancels out only when the object pose and the tag fix come
  from the same frame**; the CSV calls `obj_pair_exact == 0` a record boundary), `min_step_m` (log
  20 Hz / tracker ~9 Hz, so most rows repeat the previous value).
  Since the axes are the pool, the number of object points that go **outside** the box is printed too.
* **Which single error to show.** The default for path-following runs is **cross-track
  RMS**. The spatial follower puts stage 0 up to `path_lead_m` ahead of the vehicle's projection onto
  the current segment (`_path_split`), so even a vehicle riding exactly on the line shows a radial |p − ref|
  equal to that lead — that is why run 2055 has 3.3 cm cross-track but 12.3 cm radial.
  For radial, use `--error radial`. **A station hold has no path**, so it falls back to radial even when
  `path_following` is on ("cross-track" relative to a fixed point is a meaningless direction). A 5-lap
  reference has **only its first lap** drawn (five sets of dashed lines at different phases fill in
  each other's gaps and look like a solid line).

**Which segment to draw is decided by `segment: auto`.** For square/line/circle the trajectory window
is the result itself, but station hold and object follow **never once turn on** `traj_on` (meta
`run.traj_on: false`) — asking for the trajectory there is asking for nothing. So auto falls back in
the order trajectory → engagement → whole log. For the strict view, `--segment traj`.

**The controller in the title is also decided by the CSV.** `meta.controller.type` is the controller
the run *ended* with, and the panel can be switched mid-run (on 2026-08-23 there are runs that went
dobmpc → pid → mpc_tuned within one CSV). The per-row `mode` column is the truth, so when the modes in
the drawn segment differ, the title lists them **in the order they were flown**, like
`PID → MPC path-frame`.

When there is no data, it says so. A missing date, an `_시분초` ("_hhmmss") that matches nothing, a
header-only CSV, a run with fewer usable samples than `min_points` — each comes out as one
`[none] <무엇>: <왜>` (`[none] <what>: <why>`) line, and a total of how many did not come out is added
at the end. Provenance was taken off the figure and moved to **the file name and the console** — one
line `[ok] 20260814/2055/mpc_205520 → 20260814_2055_mpc_205520.png` is printed per figure, so the CSV
is still one lookup away (CLAUDE.md citation rule).

Tests: `rov_gui/tests/test_plot_runs.py` (synthetic run folders, no hardware, no Qt).

### **Where the policy was trying to look** — time as colour (`plot_policy_map.py`, 2026-09-08)

```bash
P=~/miniforge3/envs/robust/bin/python
$P -m rov_gui.tools.plot_policy_map 0908_112101                     # map
$P -m rov_gui.tools.plot_policy_map 0908_112101 --plan-scale 0.25   # only 1/4 of the plans
$P -m rov_gui.tools.plot_policy_map 0908_112101 --object 0.147,0.379 --panels both
$P -m rov_gui.tools.plot_policy_map 0908_111456 --mission 111831 --span 3
$P -m rov_gui.tools.plot_policy_map 20260908/0908_105822 --list     # mission list
```

Where `plot_runs.py` draws **tracking error**, this draws **the policy's intent**. All you give it is
the run folder's date/time (`0908_112101`, `20260908/0908_112101`, or a path — all three are accepted),
and if one folder holds several engagements, it writes one figure per policy mission (the plan log is
shared within the folder, so only that mission's plans are picked, using the meta
`run.started`–`written` window).

How to read: **the dot colour on the ROV path and the colour of the policy's plan-end yaw rays are the
same clock**. When the vehicle was at a dot of a given colour, the policy wanted to be looking along the
ray of that colour 1 s later, so you can see at a glance when the requested heading starts to diverge
from the object's bearing (give MAP coordinates with `--object x,y` to also get the object, and with
`--panels yaw|both` the object-bearing curve as well). With hundreds of plans they overlap and cannot be
read, so thin them out with `--plan-scale 0.25` (only 1/4, evenly) or `--plan-every N`. Translational
intent arrows are deliberately not drawn — forward motion dominates, so they would only hide the path.

The map window is **at least a 2 m square** even for a small run (`--min-span`), and it is pulled inward
into the map so that the tag map fills the frame (the run is never cropped). Wider: `--span 3`; the
whole map (4.1 × 1.7 m): `--view tags`. The yaw-ray length follows the window at 0.14× (fix it with
`--yaw-len`).

**Clock trap**: `t_rel` in `plans.jsonl` is not the CSV `t` but `t_traj` (reset to 0 at mission
start). The two clocks are 1.478 s apart in run 0908_112101 [측정:
`data/20260908/0908_112101/diag/verifierC_redo_out.txt`].
This tool takes the offset as `median(t − t_traj)` over the rows with `t_traj > 0.5`, puts the two on
the same clock, and prints that value to the console — the colours only line up if this is right.

## Leak alarm (2026-08-14)

A feature QGC had and this station did not. QGC has no leak **widget** either — what it has is the
vehicle message stream, and ArduSub shouts into it. So the job was not to draw an indicator light but
**to receive the messages this station had been throwing away**.

**ArduSub has exactly one channel for reporting a leak**: `STATUSTEXT`, the string
`"Leak Detected"`, severity `MAV_SEVERITY_CRITICAL` (2), `Sub::failsafe_leak_check()`.
There is no dedicated message and no SYS_STATUS bit — the MAVLink spec's
`MAV_SYS_STATUS_SENSOR_LEAK` goes **the other way** (an input by which a companion computer tells the
autopilot about a remote leak sensor), so no amount of polling SYS_STATUS will ever produce it.

Three consequences of this decided the implementation (`rov_gui/leak.py`):

* **On the wire it is a repeat, not a latch.** While wet, it is sent again **every 20 seconds**. So the
  topside state is a hold with a timeout (`LEAK_HOLD_S = 50 s`) — one missed repeat must not become
  "the water has drained".
* **There is no "leak cleared" message.** The vehicle silently clears its own failsafe (3-second
  detector cooldown, dataflash log only). Clearing can be inferred **only from silence**.
* **Silence is ambiguous.** `FS_LEAK_ENABLE=0` blocks not just the surface action but **the warning
  itself**, and with `LEAK1_PIN=-1` (the ArduSub default) the detector backend is never even created.
  With either one, **a flooding vehicle is completely silent.** So silence while the settings have not
  been read is not called `dry` — `Telemetry.leak` is 3-state, and `None` when unknown
  (state.py rule: "what is unknown is None, not a plausible default").

The string is compared **exactly** (`text.strip() == "Leak Detected"` + severity 2).
ArduSub has two more strings containing "leak" (`"Leak detector 1 error. Please
set SERVO..."`, `"Leak detector 1 pin (servo 10) auto-set to GPIO"` — lowercase d,
boot-time setup notices), and substring matching would turn **every boot into a flood alarm**.

On screen:

* A **red `LEAK DETECTED` banner** appears in the header and blinks. Why blinking rather than steady:
  the header already has an orange banner and four coloured pills, so a steady red is just one more
  colour, and what grabs an eye that is watching video is **motion** (the only cue that survives even
  colour-vision deficiency).
* The **`Leak` row** in SENSORS (moved to the top — if a row that means "abort the dive" can only be
  seen by scrolling, nobody looks at it), and below it the **`Enclosure` row** (internal pressure, hPa).
* One `[ERROR]` line in the MISSION LOG. **One line per event** — printing the repeat that comes every
  20 seconds as is would fill the log with the same sentence, and nobody would read it.
* The command sink **reads in** `FS_LEAK_ENABLE` / `LEAK1_PIN` (parameter reading is a path that
  already existed for `SYSID_MYGCS`). If the detector is not enabled, the log says so at boot and the
  `Leak` row becomes `detector DISABLED (FS_LEAK_ENABLE=0) — a flood would be silent`.

**Internal pressure is the second signal.** A separate failsafe sends `"Internal pressure critical!"`
(WARNING, every 30 seconds, `FS_PRESS_MAX` default 105000 Pa = 1050 hPa). Why watch it as well: the
leak pad is a **late** signal (it only gets wet once water reaches that spot). Internal pressure moves
from the moment the seal lets go. The absolute value is read from `SCALED_PRESSURE.press_abs` (id 29 =
the **internal** barometer; `SCALED_PRESSURE2` (137) is the external Bar30 = depth).

**A leak does not turn the thrusters off.** This vehicle is negatively buoyant by design, so cutting
thrust makes it sink — a leak is a reason **to bring it up to the surface and recover it**, not a reason
to stop. ArduSub's default (`FS_LEAK_ENABLE=1` = warn only) makes the same call. Alert loudly and leave
the decision to the pilot.

An alarm you have never actually seen cannot be trusted, so the demo has `--demo-leak-after S`
(demo only — there is no route from this flag into the hardware path).

### Recording C3 Depth also records the sensors

A behaviour attached **only to the depth feed**. A depth video alone does not make a dataset; what makes
it one is "how the camera was moving when each frame was captured", and that lives in the IMU ·
barometer · compass, which are much faster than the video. The colour feeds are the picture the pilot
looks at, not geometry, so nothing tags along with them (logging per feed would just triple the files).

Two more files appear with **the same stem** as the video (`sensorlog.py`):

```
c3_depth_20260806_225127.mp4          video
c3_depth_20260806_225127_rov.jsonl    ArduSub: RAW_IMU/SCALED_IMU2, SCALED_PRESSURE(2),
                                      ATTITUDE, AHRS2, VFR_HUD, BATTERY_STATUS,
                                      SYS_STATUS, EKF_STATUS_REPORT, MOUNT_STATUS,
                                      SERVO_OUTPUT_RAW, LOCAL/GLOBAL position
c3_depth_20260806_225127_c3_imu.jsonl C3 BNO086: t_device, seq, ax/ay/az, gx/gy/gz
c3_depth_..._rov.json / _c3_imu.json  the manifest for each
```

**A separate file per source** is intended. The ROV IMU and the C3 IMU are different devices with
different clocks — merging them into one means picking one clock, and the moment you do, you are lying
about the other. Each worker writes its own file on its own thread.

> **2026-08-09 correction**: between 2026-08-06~09, `_c3_imu.jsonl` was **never once created on
> hardware.** `C3VideoWorker` is a `LoopWorker` but was receiving requests via `@Slot`, and a LoopWorker's
> `run()` blocks and never reaches the event loop, so **queued slots are never delivered.**
> `_rov.jsonl` was fine because `VehicleWorker` is a `TimerWorker`, and the demo backend is a TimerWorker
> too, so the tests did not catch it — hence "half a dataset" came out silently.
> It is now a plain thread-safe method (`request_sensor_log`) + `DirectConnection`, guarded by
> `test_c3_sensor_log_request_reaches_a_loopworker` and an AST-scan test that forbids LoopWorker
> slots. **Depth recordings made in that period have no C3 IMU.**

Why **JSONL** and not CSV: five or six kinds of messages with different fields arrive mixed together, so
a CSV would be either a union schema full of blanks or one file per type, and a union schema **silently
drops** fields the firmware adds. It reads in one line: `pandas.read_json(path, lines=True)`.

Clocks: `t` is topside `time.monotonic()` (**the same clock** as the video recorder, so it is the value
to use when matching against frames), `t_unix` is wall clock. The sensors' own timestamps
(`time_boot_ms`, `t_device`) are **preserved as is** inside each record, and time-critical work should
use those — `t` includes the link latency.

Look at the manifest's `hz_measured` first, then decide whether to trust it. If an IMU requested at
200 Hz shows up there as 47 Hz, it was **the link**, not the sensor, that set the rate.

Feed recordings save **exactly what the panel received** (at the displayed size). The worker hands the
already decoded and downscaled QImage to the encoder thread, so there is no GUI-thread cost, but it is
**not the sensor original.** For datasets that need the original resolution + timestamps,
`c3_camera/c3_collect.py` is the tool for that.

VideoWriter needs a fixed fps, but frames arrive at whatever pace the link allows. So it **opens at the
actual fps at start time, and writes what actually happened into the sidecar**:

```json
{"requested_fps": 30.0, "frames_written": 116, "frames_dropped": 0,
 "duration_s": 3.89, "effective_fps": 29.81, "size": [960, 540],
 "source": "rov_gui panel 'main', as displayed"}
```

To record the whole desktop without burdening the GUI, use the following instead of this class:

```bash
ffmpeg -f x11grab -framerate 15 -i :0.0 -c:v libx264 -preset veryfast out.mp4
```

## Tests

```bash
QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/robust/bin/python rov_gui/tests/test_offline.py
# or
QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/robust/bin/python -m pytest rov_gui/tests/test_offline.py -v
```

14 tests. Runs the demo backend for 2 seconds with the real bus · real widgets and checks the result:
whether frames reach the panels, whether the watchdog turns silence red, whether the minimum window
size still fits on one screen, whether the tree has no scroll areas, whether the recording file
contains frames, whether the QImage↔numpy round trip respects the stride (padded rows are the classic
bug that crops the picture askew).

DVL (`--dvl`) also has its own file — it runs **with no device**. Line-by-line parsing of a capture from
the vendor's demo unit, mount rotation · lever arm, the "proof" fit (a correct mount passes; rotation,
mirror-image and scale errors are rejected by name; at rest or forward-only, the verdict is withheld),
**whether, with no DVL, the assembler is bit-for-bit identical to the old arithmetic**, whether shadow
changes not a single bit of the command, whether the switch is frozen while engaged, the station
bridge's DVL rung, the TCP client against a local fake server (torn lines · reconnects · unreadable
lines), the panel buttons' honesty rule, and what the 2026-10-02 review found — whether the trip stays
alive even with opt-out, whether a trip latches, whether a run's eligibility is pinned at engage, the
DVL rung's velocity · displacement caps, whether heave velocity does not come from the DVL — and the
properties of the in-flight monitors: whether the result is the same no matter which stream arrives how
late, whether a single tag jump is not recorded as the DVL's fault, whether a mount within the proof
tolerance does not trip during diagonal motion, whether the first pairs of a new motion do not become a
verdict, whether `lag_s` reads the stamp error — and what the re-audit (2026-10-02) found: whether a
proof rebuilt under the latch collapses on the next evidence, whether proof is still required after
counter-evidence even with opt-out, whether a device config change discards the evidence in every
state, whether DVL noise · glitches are not a trip, whether the alignment monitor runs even when tags
arrive slowly, whether a tick without tag velocity does not use the DVL, whether a DVL rung closed by a
cap returns to the last fix, whether the bridge rung does not accept a DVL that the gate was rejecting,
whether the DVL drops out without a latch while the fit refutes it, whether a false lock lasting 1
second is a trip and a 0.4-second one is not, whether an exception in the DVL bookkeeping cannot abort
engage · the switch · meta — and those from the fourth audit: whether a DVL rejected by a fresh tag
does not come back even as the fix goes stale (innovation hold), whether such a DVL does not open the
bridge rung when the tags disappear in their real order while a healthy DVL does, whether rejected
samples do not remain in the fix-age integration, whether a mapping that has only shrunk and an
alignment the tags only partly see are a hold rather than a trip, whether, even with large tag noise,
a single burst does not trip the difference monitor, whether a refutation hold is released within 3 s
once the evidence ends, whether a closed rung returns to the fix
position rather than the estimate — and those from the fifth audit: whether a hold is released and
applied by new frames regardless of tag latency, whether a rejected DVL does not open the bridge rung
even when late tags disappear, whether the IMU rung does not inherit the DVL velocity, whether the panel
shows the reason that tick received exactly as it is. 96 tests.

```bash
QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/robust/bin/python rov_gui/tests/test_dvl.py
# how the monitors behave under real timing (the two sensors differ in rate · latency, and ticks are asynchronous) — synthetic, about 5 minutes:
~/miniforge3/envs/robust/bin/python rov_gui/tests/sim_dvl_monitors.py
```

Run `demo_e2e_dvl.py` **in the `rovgui-pose` environment**. If it is run with `robust`, then, as with
`test_mpcc.py`, the generated files under `rov_gui/control/_mpcc_gen/` are rebuilt against that
environment's paths and git-tracked files change.

Learned depth (`--fstereo`) has its own file. It runs **with no GPU, no weights and no camera** — the
network itself is not run here; only the parts that can be silently wrong are pinned: whether the
feature is really off, whether the link budget still closes with the mono pair added, **whether a pair
with mismatched seq is never matched** (a mismatched pair does not fail; it yields a plausible but wrong
disparity), and whether the depth ratio is **never drawn without a provenance label**.

```bash
QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/rovgui-pose/bin/python rov_gui/tests/test_fstereo.py
```

14 tests. To actually run the model without a camera, do this (needs the weights in
`external/FoundationStereo` — see the "FoundationStereo checkout" section — and CUDA):

```bash
~/miniforge3/envs/rovgui-pose/bin/python -c "
import sys; sys.path.insert(0,'.')
from c3_camera.host_depth import bench_rig, synthetic_pair
from rov_gui.perception.fstereo import FStereoSession
import time, numpy as np
rig = bench_rig((640,400), with_color=True, color_size=(640,360))
s = FStereoSession(); s.start_async(on_log=lambda l,m: print(m))
while s.loading: time.sleep(0.2)
l, r = synthetic_pair((640,400), 24, seed=1)
o = s.infer(l, r, rig); print(o['solve_ms'], 'ms', o['valid_out'], '% valid')
s.close()"
```

Live policy (`--policy`) has five files + tools (the policy section above):

```bash
P=~/miniforge3/envs/rovgui-pose/bin/python
QT_QPA_PLATFORM=offscreen $P rov_gui/tests/test_policy_frames.py    # 22 tests, also runs under robust
QT_QPA_PLATFORM=offscreen $P rov_gui/tests/test_policy_obs.py       # 9 tests
QT_QPA_PLATFORM=offscreen $P rov_gui/tests/test_dp_policy.py        # 17 tests, only the stub half if there is no GPU
QT_QPA_PLATFORM=offscreen $P rov_gui/tests/test_policy.py           # 79 tests, the MpcWorker side (real acados; includes LOW None · ckpt pin)
QT_QPA_PLATFORM=offscreen $P rov_gui/tests/test_policy_worker.py    # 28 tests, no torch (includes set_ckpt · parser)
$P rov_gui/tests/test_policy_ckpt_paths.py                           # 4 tests, ckpt paths + max_run_s single source
$P -m pytest rov_gui/tests/test_run_scene.py -q                      # 19 tests (cv2+pytest: rovgui-pose)
QT_QPA_PLATFORM=offscreen $P rov_gui/tools/policy_dryrun.py --ckpt stub
QT_QPA_PLATFORM=offscreen $P rov_gui/tools/policy_bench_check.py --seconds 45   # hardware C3 (no commands sent)
```

The offline re-plotter is a separate file and **needs no Qt**:

```bash
~/miniforge3/envs/robust/bin/python rov_gui/tests/test_plot_runs.py
```

32 tests. Builds synthetic run folders and checks whether the datum lift-back is the same arithmetic as
`_to_map_xy` (whether it is rigid), whether the pool box has the same value as the real
`geometry.pool_from_tags`, whether two runs' PNGs come out at the same dimensions, whether each of the
three resolutions of a date entry (date / run / CSV) catches only what matches it, whether a PID run's
file name says `pid`, **whether the object rides the same transform as the vehicle** (whether a fixed
offset in the CSV stays the same distance on the map), whether each of the three filters removes the
right rows, whether old runs with no obj_* columns at all still open, whether `segment: auto` falls back
to engagement for station/follow, whether hhmmss is extracted correctly from feed recording names too
(`c3_depth_..._mpc`), and **whether missing data is reported as a single line with a reason**. The time
ramp is checked **in numbers**, not by eye — OKLab span (> 45) and the evenness of the quartile spacing
(deviation < 2.5), plus the contrast at the surface end (>= 2:1).

## Environment notes

* Qt bindings are searched in the order PyQt5 → PyQt6 → PySide6 (can be pinned with `ROV_GUI_QT_API`).
  The verified combination is **PyQt5 5.15.11 / Qt 5.15.14** in the `robust` env.
### Two causes of the xcb platform plugin dying (both guarded against in code)

The symptom is **the same** single line below in both cases, ending in `Aborted (core dumped)`, but the
causes are completely different. Both were hit on the first `--source hw` run on 2026-08-06.

```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "..." even though it was found.
```

1. **cv2 hijacks `QT_QPA_PLATFORM_PLUGIN_PATH`.** `cv2/config-3.py` overwrites this variable at
   import time with cv2's own Qt5 plugin directory, **unconditionally**. That folder holds only one file,
   `platforms/libqxcb.so`, built for cv2's Qt, so it does not load into PyQt5's Qt. On the hw path this
   happens *before* the QApplication is created, because **preflight's `check_python_env` imports cv2 to
   read its version**.
   → `rov_gui.qt.sanitize_plugin_path()` restores the path to our binding's right before the
   QApplication. cv2 is imported only via `import_cv2()` (which restores the original value).
   `c3_camera.viz` does the exact opposite (`repair_qt_plugin_path`), which is right since it is for `cv2.imshow`
   — but it must not be imported before the QApplication is created.
   **Offscreen tests will never catch it**: the cv2 folder has no `libqoffscreen.so`, so Qt silently
   falls back to its own folder. It dies only on X11.

2. **`LD_LIBRARY_PATH` pulls in the system Qt first.** In this desktop shell, OpenFOAM 12
   plugs `/usr/lib/x86_64-linux-gnu` into `LD_LIBRARY_PATH`. PyQt5's extension modules use
   `DT_RPATH` (takes precedence over LD_LIBRARY_PATH), so `libQt5Core` resolves correctly, but the plugin
   `Qt5/plugins/platforms/libqxcb.so` uses `DT_RUNPATH` (LD_LIBRARY_PATH wins), so
   `libQt5XcbQpa.so.5` resolves to the Ubuntu system Qt and it dies:
   `undefined symbol: _ZN23QPlatformVulkanInstance22presentAboutToBeQueuedEP7QWindow`.
   → `rov_gui.qt.preload_platform_libs()` dlopens our binding's Qt libraries first, by absolute path,
   with `RTLD_GLOBAL`. Later `dlopen` calls attach to the SONAMEs already loaded, so
   no search happens at all. (Removing `LD_LIBRARY_PATH` from `os.environ` does not help
   — glibc settles the search path at process start.)
