# c3_camera — Connecting directly to the MarineSitu C3 with DepthAI

> **English translation.** Source of truth: [`README.md`](README.md) (Korean). Snapshot of 2026-10-12, source blob `9a45635b5b`; later edits to the original are not reflected here.
>
> Provenance tags are kept as in the original: `[측정 …]` measured (artifact cited) · `[유도 …]` derived · `[예측 …]` predicted, not measured · `[스펙 …]` vendor spec · `실측` measured · `미실측` not measured · `산출물 없음` no artifact found · `예시/유도값` example or derived value.

> **To collect a SLAM dataset** → [README_DATASET.md](README_DATASET.en.md)
> (`c3_collect.py`: TUM RGB-D layout + camera IMU (BNO086) + MAVLink telemetry,
> all on one timeline). This document covers the direct camera connection, low latency and performance measurements.

This package bypasses the BlueOS/Madrona H264 + RTSP re-streaming path and connects from the desktop
**directly** to the camera over XLink to receive RGB + depth. The goal is to take the Raspberry Pi
out of the video path entirely.

```
Before: C3 → GigaBlox → RPi(Madrona: H264 encoding + RTSP re-serving) → GigaBlox → fiber → Desktop
Now:    C3 → GigaBlox → fiber → Desktop        (DepthAI/XLink, no Pi in between)
```

## Confirmed hardware (measured, not guessed)

Values that `discover_c3.py --probe` read directly from the camera:

| Item | Value |
|---|---|
| product | **OAK-D-W-POE** (wide-FOV OAK-D PoE) |
| mx_id | `19443010315B0B2F00` @ 192.168.2.191 |
| bootloader | 0.0.28 |
| CAM_A | IMX378 colour, native 4056x3040, **fixed focus** (no autofocus) |
| CAM_B / CAM_C | OV9282 mono (left/right), native 1280x800 |
| stereo baseline | 7.5 cm, factory calibration in EEPROM |
| IR projector | none (no dot projector / flood LED fitted) |
| Supported colour modes | 1920x1080@60, 2024x1520@85, 1352x1012@52, 3840x2160@42, 4056x3040@30 |
| Supported mono modes | 640x400@255, 1280x720@143, 1280x800@129 |

The resolution table in `config.py` comes from the measured values above, and at runtime
`validate_against_device()` cross-checks it again.

## Installation and runtime interpreter

```bash
bash c3_camera/setup_env.sh          # create ~/.venvs/c3-depthai + install + verify
./c3 env                             # check what got picked up and which versions
```

From then on, just call every tool through the wrapper — the wrapper takes care of both the interpreter and the working directory:

```bash
./c3 collect --profile research_near
./c3 stream
./c3 test
```

Two verified combinations:

| | python | depthai | opencv | numpy | Status |
|---|---|---|---|---|---|
| conda `robust` | 3.12.13 | 2.32.0.0 | 4.10.0 | 1.26.4 | **default** |
| `~/.venvs/c3-depthai` | 3.10.12 | 2.32.0.0 | 4.11.0 | 1.26.4 | fallback |

Since 2026-08-06 everything has been consolidated into `robust` alone — the main code (sim/SLAM) and
the camera use the same interpreter. Until then `robust` was python 3.14, and in that combination numpy
silently overwrote the operands of operations, which is why the venv was the #1 choice. Dropping to 3.12
installed numpy 1.26.4 from an official wheel rather than a source build, and that problem went away.
The last line of `./c3 env`, `numpy elide`, is that diagnostic, and the old environment is preserved as
`robust314_bad` (`KNOWN_ISSUES.md`: results produced during that period have not been audited yet).

The venv was not deleted. It can be used any time with `C3_PY=~/.venvs/c3-depthai/bin/python ./c3 ...`.

`viz.py` automatically fixes the preview window (`cv2.imshow`) not appearing in the conda environment —
cv2 overwrites its own Qt plugin path at import time, and in `robust` that directory has been switched
off as `plugins.disabled`, which is what causes it.

### ⚠ Why it must not just be run in conda base (important)

This desktop's conda base is **python 3.12.11 and already has `depthai 3.5.0`
installed.** v3's API differs from v2's, and, of all things, it differs only **partially**:

| | conda base (depthai 3.5.0) |
|---|---|
| `dai.node.ColorCamera` | present (shim) |
| `dai.node.XLinkOut` | **absent** |
| `Device.getOutputQueue` | **absent** |
| `dai.RawStereoDepthConfig` | **absent** |

In other words, v2 code breaks about halfway through building the pipeline. Worse, in v3
`dai.Pipeline(createImplicitDevice=True)` is the **default**, so merely
constructing a Pipeline finds and opens a device — on this network that
**means it can unintentionally grab the C3 that Madrona is using.**

So `c3_camera` **refuses right away** at import time if depthai is not 2.x:

```
c3_camera.DepthAIVersionError: c3_camera targets the depthai 2.x API but
found depthai 3.5.0 at /home/bdml/miniforge3/lib/python3.12/...
  Use the dedicated venv instead:
      ~/.venvs/c3-depthai/bin/python <script>
```

If you see this message, just run it via the venv path.

### If conda libraries clash

The second reason to use the venv is a history of conflicts with the system libraries, because conda
carries its own libstdc++/libGL/Qt (usually `cv2.imshow` is the first to die). The venv links against
the system libraries the X session already uses, so the display path just works.

Using the venv's absolute path is mostly fine even with conda activated. If you still get something like
`Could not load the Qt platform plugin "xcb"`:

```bash
conda deactivate                     # do this first
~/.venvs/c3-depthai/bin/python c3_camera/c3_stream.py
```

If that still doesn't work:

```bash
unset QT_QPA_PLATFORM_PLUGIN_PATH    # remove the Qt path that conda planted
QT_DEBUG_PLUGINS=1 ~/.venvs/c3-depthai/bin/python c3_camera/c3_stream.py  # find the cause
```

Measurement still works even if you give up on the display altogether: `--no-display`.
(The `libxcb-cursor0`/`libxcb-xinerama0` this environment needs are already installed, and it was
confirmed that a window actually appears in the X11 session (`DISPLAY=:1`).)

### DepthAI environment variables set automatically on import

depthai **caches environment variables on first read**, so changing them after `import depthai`
does nothing. That is why `c3_camera/__init__.py` lays them down with `setdefault` before any
submodule (an explicitly set value, if there is one, takes precedence):

| Variable | Value | Reason |
|---|---|---|
| `DEPTHAI_PROTOCOL` | `tcpip` | PoE camera, so USB discovery is skipped |
| `DEPTHAI_BOOTUP_TIMEOUT` | `30000` ms | measured boot is about 12 s while the default is 15 s, so the margin is too thin [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| `DEPTHAI_CONNECT_TIMEOUT` | `15000` ms | default 5 s |
| `XLINK_LEVEL` | `warn` | the default is FATAL, so PoE TCP errors are **hidden** |

When digging into connection failures, turn on both: `DEPTHAI_LEVEL=debug XLINK_LEVEL=debug`
(with only `DEPTHAI_LEVEL` on, XLink socket errors don't show). Note that
`DEPTHAI_LEVEL=fatal` **is not a valid value, so import throws** —
`fatal` exists only for `XLINK_LEVEL`.

## Order of use

```bash
# 0. Is the rig ready — camera/ROV/disk/display on one screen (opens nothing)
./c3 preflight

# 1. Find the camera, down to sensors and calibration (from here on, the camera gets opened)
./c3 discover --probe

# 2. Live stream (RGB + depth, fps/latency HUD)
./c3 stream

# 3. Sweep resolution/fps combinations → CSV
./c3 bench --out bench.csv

# 4. When done, restore Madrona (if you stopped it at step 2)
./c3 madrona start
```

To go without the wrapper, pick the interpreter yourself and run from the repo root — the form is the same:

```bash
V=~/.venvs/c3-depthai/bin/python
$V -m c3_camera.c3_stream
```

## preflight — tells you first where things went wrong (`preflight.py`)

From the pilot's seat every failure in this stack looks the same — "it's not working" — but in reality
they live in five different places, and some only show up after waiting out the 12 s PoE boot. **Every
entry point that touches the hardware runs this check first, before connecting.** Failures are printed
together with the command that fixes them.

| Check | What it catches |
|---|---|
| `python + deps` | wrong interpreter (conda base = depthai 3.x), numpy operand-corruption probe |
| `host route` | traffic leaking out via a VPN or the wrong NIC |
| `camera link` | PoE · switch · `192.168.2.1` — is TCP 11490 open |
| `camera owner` | **who** is holding it (Madrona / the pid of a c3 process in another terminal), is the mx_id really that C3 |
| `BlueOS` · `ROV state` | is the vehicle alive right now — armed/mode/voltage, whether the HEARTBEAT counter is actually ticking |
| `MAVLink endpoint` | does BlueOS push to this recorder's port (QGC holds 14550) |
| `telemetry path` | can that port be bound, are packets actually arriving |
| `disk` · `display` | free space, can `imshow` open a window |

```bash
./c3 preflight                              # report only
./c3 preflight --json-out pf.json           # machine-readable form
./c3 collect --preflight-only               # with that tool's own settings, run the checks only and exit
./c3 collect --no-preflight                 # skip
./c3 collect --preflight-strict             # unattended run: WARN is also a reason to abort
```

Three design decisions matter:

* **It never opens the camera.** Everything stays at the TCP / UDP discovery level, so it takes milliseconds
  and never takes the camera away from anything. Opening means a firmware upload + boot (`device.TYPICAL_POE_BOOT_S`)
  plus occupying a pipeline slot, so a check that did that would be slower than what it checks and **would
  itself become the reason the next tool fails**. If you need the sensor list and calibration as well,
  `./c3 discover --probe` is what does that.
* **It sends nothing to the vehicle.** The MAVLink checks only read.
* **A cause is counted only once.** If BlueOS is dead, the checks below it don't each time out on the
  same cause; they end immediately with "see above". When the link is dead, `camera owner` doesn't
  become a second abort reason either.

`WARN` does not block progress — capturing only video, without telemetry, is a legitimate run too. The
only thing that blocks is `FAIL` (camera occupied · link down · not enough disk), and then the exit code is `5`.

`c3_collect` **keeps this report verbatim in the dataset's `metadata.json`**, and also writes the checks
that were not clean into the `== preflight ==` section of `metadata.txt`. Months later, when only the
folder is left, the answer to "why is telemetry.csv empty?" is usually there.

## Profiles — YAML instead of a wall of flags

`c3_collect` takes `--profile`. Profiles live in `c3_camera/profiles/*.yaml`; just give the
name:

```bash
./c3 collect --profile research_near              # replaces 12 flags
./c3 collect --profile research_near --fps 30     # flags beat the file
./c3 collect --profile research_near --dry-run    # print every value + where it came from
```

Precedence, lowest first, is **StreamConfig defaults < `--mode` defaults < profile (parent→child,
`extends:`) < command line**. `--dry-run` prints, for every setting, which of `default` / `mode:research` /
`profile:<파일>` ("profile:<file>") / `cli` it came from, so there is no need to guess.

Profiles can also set `StreamConfig` fields the CLI does not expose (`median`, `subpixel`,
`confidence`, `depth_preset`, `lr_check`, `pair_mode` …). Typos, wrong sections, and values
that YAML reads as booleans, such as `median: off`, are refused at load time together with the
file name.

## Option sweep — leaves a SLAM dataset for every combination (`c3_option_sweep.py`)

Because `c3_bench.py` **keeps no pixels**, it only answers as far as "how many fps does this setting
give". What you actually want to know comes next — *does SLAM run on data captured with that
setting?* `c3_option_sweep.py` sweeps the same axes, but **for every cell it writes a complete TUM
RGB-D dataset with the same layout as `datasets/dataset_*`**
(`rgb/ depth/ left/ right/`, `rgb.txt depth.txt associations.txt frames.csv`,
`calibration.json`, ORB-SLAM3 yaml, `imu_camera.csv`, MAVLink CSV,
`metadata.json/txt`). So a setting can be judged by the **map**, not by fps.

The four axes being swept are four questions:

| Question | Option |
|---|---|
| What colour resolution? | `--color-res` × `--isp-scale` |
| Compress or not? | `--color-encode none,mjpeg,h264,h265` (+ `--mjpeg-quality`, `--video-bitrate-kbps`) |
| Include depth or not? | `--depth on,off` |
| What depth resolution? | `--depth-size match-color,derived,WxH` |

```bash
V=~/.venvs/c3-depthai/bin/python

# 0. Plan only — look at combinations, bandwidth and disk first, without touching the camera
$V c3_camera/c3_option_sweep.py --dry-run

# 1. Default ladder: 2 colour sizes × raw/MJPEG × depth on/off × 2 depth grids = 12 cells
$V c3_camera/c3_option_sweep.py

# 2. A specific question, such as "compression or not at 4K"
$V c3_camera/c3_option_sweep.py \
    --color-res 4k,1080p --isp-scale 2/3 --color-encode none,mjpeg \
    --mjpeg-quality 90 --depth on --depth-size match-color --fps 20

# 3. How far can the depth resolution be cut
$V c3_camera/c3_option_sweep.py --depth on --color-encode mjpeg \
    --depth-size match-color,derived,320x180
```

Results land under `c3_camera/sweeps/sweep_<타임스탬프>/` ("c3_camera/sweeps/sweep_<timestamp>/"):
a directory per cell + `results.csv` + `sweep_meta.json`. Each cell directory can go into SLAM as is.

Three things to watch when reading the results (also in the script docstring):

- `*_mbps` is a **last 120 frames** window; `*_mbps_run` is the average over the whole capture. Compare
  across cells with `*_mbps_run`.
- The "measured" bandwidth is a real measurement **only for compressed streams**. NV12 for `--color-encode none`
  is `w*h*1.5` and depth16 is `w*h*2`, i.e. **computed** values — so when looking at the compression axis,
  the raw rows are exact by definition and only the MJPEG/H.26x rows are measured.
- The `check` column is the `c3_dataset_check.py` verdict, and that is an RGB-D checker. `--depth off`
  cells are reported as `pass-mono`/`problems-mono`, leaving out the 2 depth-only failures, and cells that
  deliberately put depth outside the colour grid are reported with `-offgrid` appended.

### Do you really have to stop Madrona?

**Only when the camera is actually holding a pipeline.** On this unit, even with the Madrona
container running, the camera is sometimes in the `X_LINK_BOOTLOADER` (= idle) state, and in
that state you simply connect without touching the extension. `discover_c3.py` explicitly reports `free` /
`OWNED`, so decide based on that.

If `OWNED`:

```bash
$V -m c3_camera.madrona stop --yes   # disable the extension + wait until the camera is released
```

`stop` **does not run without `--yes`**, because a BlueOS disable is a persistent setting: it stays
off across reboots, and the BlueOS/Cockpit side keeps getting no video. The only way back is
`madrona start`, so don't forget it. If you'd rather not touch the vehicle settings from a script,
you can also switch it off by hand on the BlueOS Extensions page.

## Measured performance

Below are values measured directly with `c3_bench.py` on this desktop–GigaBlox–C3 path
(only the steady segment after `--warmup` is tallied).

### Default-settings result — what switching to this path gained

| | Madrona (H264/RTSP, via the Pi) | **DepthAI direct (default settings)** |
|---|---|---|
| colour latency (p50) | about **3000 ms** | **38 ms** [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| depth latency (p50) | — | **82 ms** |
| fps | 10 requested → 6–8 measured | **15 requested → 15.00 measured** [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| frame drops | — | **0 %** |

Default settings = `1080p → 960x540 MJPEG + 640x360 aligned depth @ 15 fps`,
420 frames measured back to back, measured link 71.6 Mbit/s.  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]

**Colour latency went from about 3000 ms → 38 ms.** As hypothesized, the Pi round trip was the bottleneck.

### Effective PoE link bandwidth ≈ 91.5 Mbit/s — this is the only real constraint

Three different configurations all hit the ceiling at the same throughput:

| Configuration | measured fps | x frame size | = throughput |
|---|---|---|---|
| colour only 960x540 NV12 | 13.8 | 778 kB | 86 Mbit/s [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 960x540 NV12 + depth | 9.25 | 1.24 MB | **91.7** Mbit/s [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 1920x1080 NV12 + depth | 3.19 | 3.57 MB | **91.1** Mbit/s [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 480x270 NV12 + depth | 17.5 | 655 kB | **91.6** Mbit/s [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |

Completely different pipelines converging on the same number is the evidence that this is **a link
ceiling, not a compute limit**. The PHY is gigabit, but the Myriad X's XLink-over-TCP path can't
reach that. So it is this formula, not the requested fps, that sets the actual fps:

```
actual fps ≈ 91.5 Mbit/s / (frame_bytes x 8)
```

`POE_BUDGET_MBPS = 90` in `config.py` is this value (safety margin included), and
`--dry-run` and the HUD show the requested settings against this budget. The HUD displays **measured
Mbit/s, not an estimate**.

### raw vs MJPEG — the largest single improvement

Same resolution, same 20 fps request, with depth 640x360 alongside:

| colour output | encode | measured fps | colour p50 | depth p50 | drop |
|---|---|---|---|---|---|
| 480x270 | raw | 17.5 | 123 ms | 121 ms | 12.5 % [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 480x270 | **mjpeg** | **20.0** | **57.6 ms** | 107 ms | **0 %** [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 960x540 | raw | 9.1 | 250 ms | 405 ms | 53.5 % [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 960x540 | **mjpeg** | **19.1** | **123 ms** | 124 ms | 5 % [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 1920x1080 | raw | 3.2 | 353 ms | 495 ms | 83.5 % [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 1920x1080 | **mjpeg** | **13.1** | **147 ms** | 154 ms | 34.8 % [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |

At 960x540, fps 2.1× and latency 1/2. At full 1080p, fps 4.1× and latency 1/2.4.  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
That is why **MJPEG is the default.** If you need lossless pixels,
use `--color-encode none` (about 9 fps in exchange).  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]

The measured MJPEG compression ratio at q90 is **about 6:1** (960x540 is 112–139 kB/frame). At first  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
it was estimated at 10:1, which underestimated the bandwidth by nearly half, so the measured value is
now hard-coded into `MJPEG_RATIO`. Turbid or dark water has fewer high-frequency components and
compresses better, while a textured seabed in clear water compresses less — the estimate is for
planning, and decisions are made on the HUD's measured values.

### Order for cutting bandwidth (biggest effect first)

1. **`--color-encode mjpeg`** — encodes on the device, measured about 1/6. Host decoding takes
   a few ms. As the table above shows, this has the biggest effect.
2. **`--isp-scale`** — downscales in the ISP, so the sensor stays in its native mode
   (= **the field of view is kept**). Cropping to a smaller sensor mode narrows the field of view.
3. **Match `--fps` to the actual throughput** — frames that can't get through are never produced at all,
   so the latency spread (p95) shrinks. Rather than receiving 9 fps at 38% drops, it is better to request 9 fps  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
   and get 0% drops.
4. **Cut down the depth side** — with the default settings depth is 55 Mbit/s, 3.5× the colour (16).
   640x360 depth16 = 461 kB, while MJPEG colour is about 130 kB. If you want to push higher,
   set `--depth-size 480x270`, or set `--mono-fps` lower than the colour.

### Latency = fixed overhead + transfer time + queuing

Gathering all the measurements, colour latency tracks **utilization relative to the ceiling** far
more cleanly than it tracks frame size:

| Configuration | colour kB/f | measured Mbit/s | vs ceiling | colour p50 | p95 |
|---|---|---|---|---|---|
| 640x360 mjpeg @12 + depth 480x270 | 67 | 31.8 | 35 % | **35.5 ms** | 36.1 [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 640x360 mjpeg @20 + depth 480x270 | 67 | 52.0 | 58 % | **35.0 ms** | 35.8 [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 960x540 mjpeg @15 + depth 640x360 **(default)** | 131 | 71.4 | 78 % | 39 ms | 80 [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 1920x1080 mjpeg @12 + depth 480x270 | 377 | 61.9 | 68 % | 77 ms | 84 [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 480x270 mjpeg @20 + depth 640x360 | ~30 | ~77 | 84 % | 58 ms | 58 [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 960x540 mjpeg @30 + depth **480x270** | 131 | 90.4 | 99 % | 84 ms | 96 [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 960x540 mjpeg @30 + depth 640x360 | 131 | 88.2 | 96 % | 120 ms | — [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 1920x1080 mjpeg @20 | 377 | saturated | >100 % | 147 ms | 170 [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 960x540 raw @20 | 778 | saturated | >100 % | 250 ms | 284 [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| 1920x1080 raw @20 | 3111 | saturated | >100 % | 353 ms | 395 [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |

Utilization alone doesn't explain it — 68 % (1080p, 77 ms) is slower than 78 % (default, 39 ms).  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
Viewed as the **sum** of three terms, everything fits:

```
colour latency ≈ 30 ms          (fixed: sensor readout + ISP + encoding)
               + frame_kB x 8 / 91.5 Mbit/s   (transfer: time it costs even when the link is empty)
               + queuing        (surges as the ceiling gets closer)
```

Sanity check: 640x360 (67 kB) → 30 + 6 = 36 vs measured 35.0. 960x540 (131 kB) → 30 + 11 = 41  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
vs measured 39. 1080p (377 kB) → 30 + 33 = 63; adding the queuing at 68 % utilization gives the measured 77.

So there are two practical rules:

> 1. **Making frames smaller** shrinks the transfer term — a benefit regardless of utilization.
> 2. **Below 60 % of the ceiling** (about 55 Mbit/s) the queuing term vanishes and the jitter goes away too
>    (p95−p50 < 1 ms). Above that is the zone where you trade fps against latency.  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
>
> The colour latency floor is **about 30–35 ms**, and it can't go below that.
> Preview with `--dry-run`, and confirm with the HUD's measured values.

### Recommended settings by purpose

All of these are actually measured values (not predictions):  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]

| Purpose | Command (appended after `c3_stream.py`) | Measured |
|---|---|---|
| **Balanced (default)** | *(none)* | 15.0 fps, colour **39 ms**, drop 0 % [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| **Lowest latency + no jitter** | `--isp-scale 1/3 --fps 20 --depth-size 480x270` | 20.0 fps, colour **35.0 ms** (p95 35.8, max 36.1), drop 0 %, ceiling 58 % [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| **Highest fps** | `--fps 30 --depth-size 480x270` | **28.5 fps**, 84 ms, drop 5 %, ceiling 99 % [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| **Max image quality (full 1080p)** | `--isp-scale none --fps 12 --depth-size 480x270` | 12.0 fps, **77 ms**, drop 0 %, ceiling 68 % [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |
| **Lossless pixels** | `--color-encode none --fps 9` | about 9 fps, raw NV12 [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md] |

For visual servoing/teleoperation use option 2, for data collection option 4, and if registration
matters add `--pair-mode timestamp`.

### About the latency table in the Luxonis docs

The starting point, "PoE OAK: 150 ms at 4K 8 fps, 530 ms at 10 fps", has the right direction
(lowering resolution/fps improves latency and fps together), but **its numbers were not carried
over as is.** That table's 4K-row bandwidth values disagree with the same page's own formula
(530/663 vs 796/846 Mbps), and the measurement conditions (whether encoded, ISP output size) are not
specified. So all the numbers above are **values measured directly with `c3_bench.py` on this camera
and this network**. If the cable, switch or optical link changes, just measure again.

### Good to know

- **Connecting takes about 12 s.** That is the time DepthAI takes to upload firmware over PoE and boot,  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
  a 1-time cost per run. It is unrelated to frame latency.
- **depth latency > colour latency** (82 vs 38 ms) is normal. Stereo computation is involved,
  and the frame is also 3.5× larger.
- **Colour/depth skew about 66 ms** (≈ 1 frame). Colour and the mono pair are not hardware-synchronized,
  so they can drift apart by up to one frame interval. If registration matters,
  use `--pair-mode timestamp` (at the cost of waiting up to one frame).
- **In an in-air indoor test, depth valid was ≈ 21%.** That was due to textureless walls/backlighting,  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
  and in water it is different again. The HUD's `depth valid %` keeps showing this.

## What was actually done for low latency

| Item | Call | Reason |
|---|---|---|
| Send colour as NV12 | `ColorCamera.video` (not `preview`) | 1.5 B/px vs 3.0 — half the transfer for the same pixels. The BGR conversion is done on the host anyway |
| Disable XLink chunking | `pipeline.setXLinkChunkSize(0)` | Hands each frame over in one go instead of splitting it |
| Device-side queue 1 + non-blocking | `XLinkOut.input.setQueueSize(1)`, `setBlocking(False)` | When the link backs up, **old frames are dropped**. Latency does not accumulate |
| Host-side queue 1 + non-blocking | `getOutputQueue(name, 1, False)` | Same reason, on the host |
| Discard the backlog | `tryGetAll()`, then decode only the last one | Doesn't replay piled-up frames; looks only at the current one |
| Wait without spinning | `Device.getQueueEvents(...)` | Wakes up at arrival time. No polling latency, no wasted CPU |
| ISP downscaling | `setIspScale(n, d)` | Cuts transfer while keeping the field of view |
| Pin the depth unit | `initialConfig.setDepthUnit(MILLIMETER)` | States the contract in code instead of relying on the default |
| Stronger timesync | `setTimesync(500ms, 10, True)` | The precondition for the latency numbers to mean anything |

## How latency is measured

```
latency = dai.Clock.now() - frame.getTimestamp()
```

DepthAI timestamps the frame on the device, and timesync aligns the device clock to the
host `dai.Clock` domain. So this is the **one-way (not round-trip)**
sensor→host segment.

- **Included**: sensor readout, ISP, (for depth, the stereo computation), device-side queue,
  PoE XLink/TCP transfer, host queue wait.
- **Not included**: our own colour conversion, colorize, `cv2.imshow`. This remainder is
  covered by `Frame.age_ms()`, which also appears on the HUD.
- `getTimestamp()` is referenced to the **middle of the exposure** by default. When exposure gets
  longer in dark water, the number moves by half the exposure time. That is why the HUD also shows
  the exposure time.

**Drop verdict**: sequence numbers are assigned on the device before transfer, so a hole in the
numbers means that frame was produced and then discarded (the link couldn't keep up). This value
tells apart the case where "it's doing 30 fps" is actually the result of discarding half.

## CLI parameters

`c3_stream.py` and `c3_bench.py` share the same argument surface (`add_stream_args` in
`config.py`). For the full list, see `--help`.

Main arguments:

```
--color-res  1080p | 1352x1012 | 2024x1520 | 4k | 12mp      (IMX378 sensor modes)
--isp-scale  N/D or none           e.g. 1/2 → 1080p becomes 960x540
--color-encode mjpeg | none        default mjpeg (measured about 1/6). none is lossless but about 9 fps
--mono-res   400p | 720p | 800p    (OV9282)
--fps / --mono-fps
--streams    color,depth[,left,right]
--depth-align color | right | left | center | none
--depth-preset robotics | high_accuracy | high_density | ...
--subpixel / --extended / --no-lr-check / --median / --confidence
--pair-mode  latest | timestamp    latest never waits (lowest latency)
--queue-size / --device-queue-size / --xlink-chunk / --small-pools
```

`c3_stream.py` keys: `q` quit, **`v` start/stop recording**, `s` snapshot (PNG + depth
16-bit PNG + NPZ), `c` point cloud (PLY), `o` switch view (side/overlay/color/depth),
`p` stats, `r` reset stats, `[` `]` depth colour range, `h` toggle HUD.

## Recording

```bash
# record from the moment it starts
c3py c3_camera/c3_stream.py --record

# start/stop with the v key while watching (when capturing several separate segments)
c3py c3_camera/c3_stream.py
```

The default save location is `c3_camera/recordings/<YYYYMMDD_HHMMSS>/` (change it with `--record-dir`).

```
recordings/20260729_161500/
  meta.json      settings·device·intrinsics·calibration·clock offset
  frames.csv     per-frame index → filename + capture time + seq + latency
  color/000001.jpg    with MJPEG, the original as is, without re-encoding
  depth/000001.png    16-bit PNG, **lossless millimetres** (0 = no measurement)
```

What the design holds to:

- **Never block the stream.** 16-bit PNG encoding takes a few ms, so doing it inline would
  raise latency and make the device drop frames — and the reason this client was built would be gone.
  So writing is done by a **separate writer thread**, and when its queue fills it drops frames but
  **counts them and reports the count** (in a research recording, a silent hole is the worst case).
- **Depth is lossless.** It is uint16 millimetres, so lossy compression would silently ruin the measurements.
  To write faster, use `--record-depth-format npy` (uncompressed, but about 2× the size).
- **MJPEG colour is not re-encoded.** It already arrives as a JPEG bitstream, so it is written to a
  file as is. No decoding/re-encoding → almost 0 CPU, no additional quality loss.

The size is reported at startup. With the recommended settings (640x360 mjpeg @20 + depth 480x270) it is
roughly on the order of **150 MB/min = 9 GB/hour**.  [UNVERIFIED: 예시/유도값 — docs/MEASUREMENT_AUDIT.md]

### Checking · playing back a recording

**Run `--info` first.** It cross-checks the recording against itself — whether every file the
index points to exists, whether depth really is uint16 millimetres, and whether there are no holes
in the device seq:

```bash
c3py c3_camera/c3_replay.py c3_camera/recordings/20260729_161500 --info
```

```bash
# play back at the recorded speed (space pause, . next, , previous)
c3py c3_camera/c3_replay.py c3_camera/recordings/20260729_161500

# MP4 for sharing (colour+depth side by side). But the depth in the MP4 is a colour map, not data
c3py c3_camera/c3_replay.py c3_camera/recordings/20260729_161500 --export-mp4 out.mp4
```

### Reading a recording in code

```python
import csv, json, cv2
from pathlib import Path

d = Path("c3_camera/recordings/20260729_161500")
meta = json.loads((d / "meta.json").read_text())
rows = list(csv.DictReader((d / "frames.csv").open()))

for r in rows:
    colour = cv2.imread(str(d / r["color_file"]), cv2.IMREAD_COLOR)
    depth  = cv2.imread(str(d / r["depth_file"]), cv2.IMREAD_UNCHANGED)  # uint16 mm
    t      = float(r["color_t_device"])        # capture time (dai.Clock domain)
```

`cv2.IMREAD_UNCHANGED` is **mandatory**. Leave it out and OpenCV silently reads 8-bit, so every
depth value is off by 257×.

`t_device` is a monotonic clock, so it loses its absolute-time meaning once the process ends. That is
why the offset is stored alongside it under `clock` in `meta.json`:
`절대시각 = t_device - dai_now_s + time_time_s` ("absolute time = t_device - dai_now_s + time_time_s").

## Using RGB-D in code

```python
from c3_camera.config import StreamConfig
from c3_camera.source import C3Source

cfg = StreamConfig(color_res="1080p", isp_scale=(1, 2), mono_res="400p",
                   fps=10, depth_align="color", pair_mode="timestamp")

with C3Source(cfg) as src:
    for bundle in src.stream():
        colour = bundle.color.image        # HxWx3 uint8 BGR
        depth  = bundle.depth.image        # HxW  uint16, **millimetre**, 0 = no measurement
        skew   = bundle.skew_ms            # colour/depth capture-time difference
        xyz, rgb = bundle.pointcloud(stride=4)   # (N,3) float32 m, (N,3) uint8
```

`depth == 0` means "measurement failed", not "distance 0". Never let it through as a valid value.

The point-cloud coordinate frame is the OpenCV optical frame (+X right, +Y down, +Z forward).
Going to the ROS body frame (+X forward, +Y left, +Z up) takes just one fixed rotation, but it was
deliberately not baked in here.

## Hooking up ROS

`Frame` carries only what a ROS publisher needs, and has no ROS dependency:

| Frame field | ROS |
|---|---|
| `image` | `sensor_msgs/Image` (`bgr8`, `16UC1` for depth) |
| `t_device` | `header.stamp` |
| `seq` | device sequence number (for hole detection) |
| `src.intrinsics[...]` | `sensor_msgs/CameraInfo` (K, D at the streaming resolution) |

`header.stamp` must get **`t_device` (capture time)**, not the host arrival time. Whether the TF
tree holds up when the link stalls once is decided right here.

## Underwater caveats

- The factory calibration is an **in-air** calibration. In water the port (flat/dome)  [UNVERIFIED: 예시/유도값 — docs/MEASUREMENT_AUDIT.md]
  changes the effective focal length and adds radial distortion. For metric work in water it is only
  a starting point, and the refractive calibration work in this repo's `src/` applies directly.
- baseline 7.5 cm + no IR projector → in textureless, turbid water stereo
  leaves many holes. You can check this right away with the HUD's `depth valid %`.
- Colour is fixed focus. Having no focus-related knobs is normal.

## Troubleshooting

| Symptom | Cause | Action |
|---|---|---|
| `Could not connect to device with mx_id` / `X_LINK_DEVICE_NOT_FOUND` | Another pipeline is holding it (usually Madrona) | `madrona status` → if needed `madrona stop --yes`, wait 5–10 s |
| Connecting takes about 12 s | Normal. The time to upload firmware over PoE and boot | A 1-time cost per run, not a per-frame cost |
| Fails when reopened right after closing | The PoE device is coming back to the bootloader after a reset | Retry after 5–10 s (`open_device` retries automatically) |
| discovery fails but TCP is open | A switch/firewall/unrelated VPN is blocking broadcast | Connect directly with `--ip` (no broadcast needed) |
| fps below the request, and many drops | Link bandwidth limit (~90 Mbit/s) | Downscale with `--isp-scale`, `--color-encode mjpeg`, lower `--fps` |
| depth mostly 0 | Not enough texture / too close | `--extended` (close range), raise `--confidence`, lighting |
| `dropped > 0` in a recording | The disk can't keep up | `--record-depth-format npy`, raise `--record-queue`, lower `--fps`, a faster disk. Find where the holes are from seq jumps in `frames.csv` |
| Recorded depth values look odd | `IMREAD_UNCHANGED` not passed to `cv2.imread` | Read as 8-bit, so the values are off by 257×. Always use `cv2.IMREAD_UNCHANGED` |
| `Qt platform plugin "xcb"` | conda library conflict | Re-run after `conda deactivate`; work around it with `--no-display` |

## Files

| File | Role |
|---|---|
| `config.py` | `StreamConfig` + measured capability table + bandwidth budget + CLI arguments |
| `device.py` | discovery / connection (direct IP, no broadcast needed) + diagnostics |
| `pipeline.py` | `StreamConfig` → device pipeline |
| `source.py` | `C3Source` → `Bundle`/`Frame` (programmatic interface, ROS seam) |
| `metrics.py` | fps / latency / drop tally |
| `geometry.py` | intrinsics + depth(mm) → XYZ(m) |
| `viz.py` | depth colorize + HUD |
| `madrona.py` | BlueOS extension stop/start |
| `recorder.py` | recording on a background writer thread (does not block the stream) |
| `discover_c3.py` | discovery script (deliverable 1) |
| `c3_stream.py` | live viewer + recording (deliverable 2·3·4) |
| `c3_bench.py` | resolution/fps sweep → CSV (keeps no pixels) |
| `c3_option_sweep.py` | **option sweep → 1 SLAM dataset per combination + comparison CSV** (see below) |
| `c3_replay.py` | recording verification (`--info`) · playback · MP4 export |
| `tests/test_offline.py` | 47 regression tests that run without a camera (back-projection math, resolution/bandwidth calculation, CLI parsing, H.264 extraction, recording losslessness) |
| `setup_env.sh` | venv creation + install + verification |

```bash
# verification you can run any time, no camera needed
~/.venvs/c3-depthai/bin/python c3_camera/tests/test_offline.py
```


<!-- MEASUREMENT AUDIT (2026-08-04): the following numbers appear inside code
     blocks above and could not be annotated inline without corrupting them. -->

> **[UNVERIFIED]** The figures on the line below were classified as unverified by the provenance audit — details in [docs/MEASUREMENT_AUDIT.md](docs/MEASUREMENT_AUDIT.md).
>
> - `c3_camera/README.md:306` — 예시/유도값 — `컬러 지연 ≈ 30 ms          (고정: 센서 readout + ISP + 인코딩)` ("colour latency ≈ 30 ms (fixed: sensor readout + ISP + encoding)")
> - `c3_camera/README.md:405` — 산출물 없음 — `--color-encode mjpeg | none        기본 mjpeg (실측 약 1/6). none은 무손실 대신 약 9 fps` ("--color-encode mjpeg | none — default mjpeg (measured about 1/6); none is lossless but about 9 fps")
