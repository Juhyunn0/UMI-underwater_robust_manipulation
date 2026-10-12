# C3 Camera: Operating Manual for the 3 Main Tests (TESTING.md)

> **English translation.** Source of truth: [`TESTING.md`](TESTING.md) (Korean). Snapshot of 2026-10-12, source blob `2c3309a1c3`; later edits to the original are not reflected here.
>
> Provenance tags are kept as in the original: `[측정 …]` measured (artifact cited) · `[유도 …]` derived · `[예측 …]` predicted, not measured · `[스펙 …]` vendor spec · `실측` measured · `미실측` not measured · `산출물 없음` no artifact found · `예시/유도값` example or derived value.

This document is a runbook that answers three questions about the MarineSitu C3 (= Luxonis
OAK-D-W-POE). Just run it from top to bottom as written.

| # | Question | Tool | Hardware occupancy | Physical setup |
|---|---|---|---|---|
| 1 | Which resolutions/fps actually hold 15-20 fps? | `c3_bench.py` | ~7 min (+ optional 1.6 min) | None |
| 2 | How accurate is depth, and what is the minimum distance? | `c3_depth_accuracy.py` | ~12 min (+ ~40 min of moving it by hand) | Flat wall + tape measure |
| 3 | How much image quality does the encoder cost? | `c3_encode_quality.py` | ~11 min | Static textured scene |

**Why this order**

1. **Test 1 is the cheapest and the most informative.** It needs no physical setup at all, is done in 7 minutes,
   and sets the operating point (resolution · fps · depth size) for the other two tests.
2. **Test 1's `mono_res` choice directly constrains Test 2's minimum distance.**
   `MinZ = fx * baseline / max_disparity`, and fx is proportional to the mono resolution.
   At 400p MinZ is **300 mm** (measured), at 720p/800p it is 600.2 mm[유도] — **double**. If in Test 1
   you decide "raise mono to 720p for depth detail", Test 2's close-range
   rungs (130/185/280 mm) all fail structurally. So mono_res has to be decided first,
   and Test 2 designed around that value.
3. **Tests 2 and 3 use the same wall.** The wall with Test 2's AprilTag sheets on it
   is reused as-is as Test 3's "static textured scene". To avoid setting up the rig twice,
   the order is 2 → 3.
4. **Test 3 loses nothing by going last.** It is the only test that can be re-scored any number of times via `--offline`
   without the camera, so once the capture is done, the analysis can be revised and
   repeated later.

---

## Before you start

**1. The interpreter must be this one.** The wrapper picks it for you:

```bash
./c3 env          # prints interpreter / python / depthai / numpy elide
```

The commands below are written with `$V`, which is the same interpreter the wrapper picks:

```bash
V=~/.venvs/c3-depthai/bin/python
```

Since 2026-08-06, the 1st choice of `./c3` is conda `robust` (python 3.12). Before that, robust was
3.14, and numpy overwrote the operands of operations — the `numpy elide` line of `./c3 env` is the
diagnostic for that; if it shows `CORRUPTS OPERANDS`, do not measure with that interpreter.

**2. conda base is dangerous.** conda base has depthai 3.x installed, and in v3
`Pipeline(createImplicitDevice=True)` **opens the device just by being constructed** — it snatches a
C3 that another process is using. The package has an import guard that refuses 3.x, but do not call
these scripts from a `conda activate` state in the first place. Check:

```bash
$V -c "import depthai; print(depthai.__version__)"   # must be 2.x
```

**3. See in one go whether the rig is ready.** It prints interpreter · network path · camera
ownership · ROV · disk · display on one screen. It does not open the device, so it is safe even
while another process is streaming, and it finishes in a few seconds:

```bash
./c3 preflight; echo "exit=$?"
```

If `READY`, just proceed. If `NOT READY`, the last `blocking:` line is the cause, and the command to
run is attached under that check as `fix:`. Exit codes: **`0` = ready**, `5` = a blocking check is
present.

To look at camera ownership on its own, `discover_c3.py` still does that job —
**`0` = found and connectable**, `3` = owned by another pipeline (`X_LINK_BOOTED` — stop the
stream on the Madrona/BlueOS side first), `4` = discovery failed,
`5` = not visible on the network at all.

> **Do not add** `--probe`. It connects briefly without a pipeline, so it occupies the device for 1-2 s.
> Do not use it while a streamer is running. `./c3 preflight` is different in that respect —
> it opens nothing.

**4. First check that the regression tests are green** (the venv has no pytest, so run the scripts
directly — `python -m pytest` fails with `No module named pytest`):

```bash
./c3 test                                   # all six, in order (recommended)

$V c3_camera/tests/test_offline.py          # 58/58
$V c3_camera/tests/test_depth_accuracy.py   # 58/58
$V c3_camera/tests/test_encode_quality.py   # 36/36
$V c3_camera/tests/test_host_depth.py       # 45/45
$V c3_camera/tests/test_option_sweep.py     # 32/32
$V c3_camera/tests/test_preflight.py        # 43/43
```

**5. The link ceiling is ~91.5 Mbit/s** (`C.POE_BUDGET_MBPS = 90.0`). If you changed the cable or
the switch, this number is the first thing to become invalid — Test 1's Stage 1a is the pass/fail
check for it.

---

# Test 1 — Resolution x fps frontier

**Question:** Which colour/depth resolution combinations actually **sustain** 15-20 fps?

**Physical setup:** None. The camera just has to be on and looking at any scene.

**Total time:** 13 combo x 32 s = **~7 min** (+1.6 min if the optional Stage 4 is chosen).
32 s per combo = `--duration 10 + --warmup 3 + --settle 5 + 14` (12 s boot + PoE reset).

## Stage 0 — Pre-screening without using the camera (0 min, free)

Screen the combinations first, before burning 32 s of hardware time on each.

```bash
V=~/.venvs/c3-depthai/bin/python
cd /home/bdml/Desktop/umi_underwater_robust_control

$V c3_camera/c3_stream.py --dry-run --color-res 1080p --isp-scale 2/3 \
    --color-encode mjpeg --mono-res 400p --fps 15 --depth-size 640x360
```

Two things to check:

* `colour output width ... is not a multiple of 32` — **there must be no such warning.**
  If there is one, VideoEncoder refuses it and the whole 32 s run is lost.
* The % on the `budget` line. But **do not drop 1080p from the ladder because it says `OVER BUDGET`** —
  `est_mbps` uses a fixed `MJPEG_RATIO = 6.0`, so at 1080p it overestimates by about 1.4× relative to
  the measurement (measured 8.25:1). The actual frontier only shows up in the `color_kb_frame` column.  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]

Values checked in advance (all pass the 32-alignment):

```
isp 1/2 (960x540)  + depth 480x270 @15 =  52% [ok]     @20 =  69% [tight]
isp 1/2 (960x540)  + depth 640x360 @15 =  79% [tight]  @20 = 105% [OVER]
isp 2/3 (1280x720) + depth 480x270 @15 =  65% [tight]  @20 =  87% [tight]
isp 2/3 (1280x720) + depth 576x324 @15 =  80% [tight]
isp 2/3 (1280x720) + depth 640x360 @15 =  92% [tight]  @20 = 123% [OVER]
isp none (1920x1080) + depth 480x270 @15 = 104% [OVER]
isp none (1920x1080) + depth 576x324 @15 = 119% [OVER]
```

## Stage 1 — Depth resolution ladder (6 combo, 3.2 min)

Depth is uncompressed, so its bandwidth is determined by arithmetic (`w*h*2*fps*8/1e6`). So the
sweep's job is not to find the ceiling but to see **how it collapses at the knee**.

```bash
mkdir -p c3_camera/bench

# 1a — anchor. This combo is the only measured boundary row in the README bandwidth table:
#      "960x540 mjpeg @15 + depth 640x360" = 71.4 Mbit/s (78%), p50 39 ms, p95 80 ms.
#      The @20 row in the encoder table is 19.1 fps / drop 5%.
#      If measured_mbps agrees with 71.4 within 5%, the link is unchanged,
#      and all predictions of the later stages become valid. If it is off, the cable/switch has changed.
$V c3_camera/c3_bench.py \
    --color-res 1080p --isp-scale 1/2 --color-encode mjpeg --mono-res 400p \
    --fps 15,20 --depth-size 640x360 \
    --out c3_camera/bench/s1a_anchor.csv

# 1b — try lowering depth
$V c3_camera/c3_bench.py \
    --color-res 1080p --isp-scale 1/2 --color-encode mjpeg --mono-res 400p \
    --fps 15,20 --depth-size 480x270,320x180 \
    --out c3_camera/bench/s1b_depth.csv
```

**Early-stop condition:** if 1a's `measured_mbps` agrees with 71.4 within 5%, and both 480x270 rows
of 1b pass the `hold(f)` verdict formula below at both 15 and 20, then `depth_size = 480x270` is
confirmed as the safe default. If you are in a hurry, you may stop here.

## Stage 2 — Colour resolution ladder (4 combo, 2.1 min)

Fix depth at the 480x270 decided in Stage 1 and raise only the colour.

```bash
$V c3_camera/c3_bench.py \
    --color-res 1080p --isp-scale 2/3,none --color-encode mjpeg --mono-res 400p \
    --fps 15,20 --depth-size 480x270 \
    --out c3_camera/bench/s2_colour.csv
```

`2/3` = 1280x720 (never measured), `none` = 1920x1080 @15 (the README has only @12 and @20).
`1/3` (640x360) and `1/2` (960x540) are already measured in the README's bandwidth table, so do
not include them.

> Build the colour ladder **only with `--isp-scale`, within a single sensor mode**.
> isp reduction preserves the field of view (downscale), but stepping down to a smaller sensor mode
> crops it and narrows the field of view. Mixing sensor modes mixes the resolution comparison up with a field-of-view comparison.

## Stage 3 — The corner of both axes (3 combo, 1.6 min)

Measuring each axis's upper limit separately leads to the wrong conclusion that "colour 1280x720
works and depth 640x360 works too". The budget is a sum, so together they take 92%.

```bash
$V c3_camera/c3_bench.py \
    --color-res 1080p --isp-scale 2/3 --color-encode mjpeg --mono-res 400p \
    --fps 15 --depth-size 576x324,640x360 \
    --out c3_camera/bench/s3a_corner.csv

$V c3_camera/c3_bench.py \
    --color-res 1080p --isp-scale none --color-encode mjpeg --mono-res 400p \
    --fps 15 --depth-size 576x324 \
    --out c3_camera/bench/s3b_1080p_corner.csv
```

## Stage 4 (optional) — Sensor modes above 1080p / 4:3 full field of view (3 combo, 1.6 min)

This stage tests **ISP/sensor compute**, not the link.

```bash
$V c3_camera/c3_bench.py --color-res 4k --isp-scale 1/3 --color-encode mjpeg \
    --mono-res 400p --fps 15 --depth-size 480x270 \
    --out c3_camera/bench/s4a_4k.csv

$V c3_camera/c3_bench.py --color-res 12mp --isp-scale 6/19 --color-encode mjpeg \
    --mono-res 400p --fps 15 --depth-size 480x270 \
    --out c3_camera/bench/s4b_12mp.csv

$V c3_camera/c3_bench.py --color-res 2024x1520 --isp-scale 12/23 \
    --color-encode mjpeg --mono-res 400p --fps 15 --depth-size 480x270 \
    --out c3_camera/bench/s4c_2024.csv
```

`4k 1/3` produces the same 1280x720 pixels as `1080p 2/3`, from a different sensor mode — placed
side by side, that is a field-of-view experiment in itself. **Do not mix it into the Stage 2 table.**

## Run checklist

| ☐ | Stage | Command | combo | Time | What it answers |
|---|---|---|---|---|---|
| ☐ | 0 | `c3_stream.py --dry-run ...` | - | 0 | 32-alignment + budget pre-screen |
| ☐ | 1a | `s1a_anchor.csv` | 2 | 64 s | Is the link still 91.5 Mbit/s? |
| ☐ | 1b | `s1b_depth.csv` | 4 | 128 s | Depth 480x270 / 320x180 knee |
| ☐ | 2 | `s2_colour.csv` | 4 | 128 s | Colour 1280x720 / 1920x1080 |
| ☐ | 3a | `s3a_corner.csv` | 2 | 64 s | 1280x720 + depth 576/640 together |
| ☐ | 3b | `s3b_1080p_corner.csv` | 1 | 32 s | 1920x1080 + depth 576 together |
| ☐ | 4a-c | `s4a/b/c_*.csv` (optional) | 3 | 96 s | 4:3 full field of view / ISP compute limit |

## Which column is the answer

**Verdict columns:** `color_fps`, `depth_fps` (both are `fps_lifetime` after warmup).
**Trap columns:** `color_drop_rate`, `depth_drop_rate`, `color_dropped`, `color_frames`.

The verdict formula is exactly this:

```
hold(f) = color_fps      >= 0.98 * f
      AND depth_fps      >= 0.98 * f
      AND color_drop_rate <= 0.005
      AND depth_drop_rate <= 0.005
      AND (color_lat_p95 - color_lat_p50) <= 10 ms
```

Why `depth_fps` must always be checked as well: the pairing rate is `min(color_fps, depth_fps)`, and
if only one side drops, the colour/depth skew widens.

Cross-check: comparing `color_frames` with `requested_fps * duration` also catches a late metrics
reset.

## How to read bad results

* **`color_fps` 19.1 but `color_drop_rate` 0.05 → not a success.** drop is counted as **holes** in
  the sequence numbers the device assigns before transmission. I.e. it produced 20 frames and threw 1
  away, and the stream has holes in it. For SLAM/teleoperation this is **worse than a uniform 15 fps.**
  In that case the right move is to lower fps to 15 and get a stream without holes.
* **link-limited signature:** `measured_mbps` stuck at 88-92 + `est_mbps >= measured_mbps`
  + both streams drop **simultaneously**, in proportion to their byte share + `color_lat_p95`
  spreads far above `p50`. → You have to lower the resolution or the depth size.
* **compute-limited signature (3 branches):** when `measured_mbps < 75` but fps falls short,
  * `drop_rate ~ 0` → **sensor clamp.** The frames were never produced in the first place (4k tops out at 42 fps,
    12mp at 30 fps). If it shows up in Stage 4, that is normal.
  * `depth_fps < color_fps` and only `depth_drop_rate` is > 0 → **stereo node.**
    (`--subpixel` + `--depth-align color` are the known culprits)
  * `loop_hz < min(color_fps, depth_fps)` → **host.** Shut down other processes and rerun.
* **Several `ok=0` rows that all look identical:** look at the `depth_out` column. Failed
  rows also carry the requested depth size, so you can tell which rung of the ladder each one is.

## What Test 1 cannot tell you

* **`--extended` and `--mono-fps` do not exist in c3_bench.** `--extended` is the only knob that halves
  MinZ, but it cannot be measured with the bench → that is Test 2's job.
  The same goes for `--mono-fps` (the 4th-priority bandwidth knob that decouples depth fps from colour).
* **`est_mbps` now responds to `--mjpeg-quality`** (fixed 2026-08-06).
  `MJPEG_QUALITY_SCALE` scales relative to q90=1.00 (q80 0.71, q75 0.58) —
  the values come from measurements made by re-encoding 120 frames that the C3 itself captured
  (`c3_camera/datasets/dataset_20260805_174544/rgb`), and at q90 the re-encode's 114.1 kB/f vs
  the device encoder's 114.2 kB/f agreed within 0.1%. Still, **an estimate is an estimate**: the scene
  dependence (46~121 kB/f) is something no model can capture, so the verdict uses the measured `color_kb_frame` /
  `color_mbps` columns.
* **Do not compare `drop_rate` and latency directly between encoder rows.** The queue policy
  is asymmetric (`pipeline.py:120-130`): h264/h265 use device+host blocking queues
  1.5 s deep, none/mjpeg non-blocking depth-1. Under saturation, none/mjpeg shows the pressure
  as `drop_rate`, h26x as latency.
* **2 duplicates that fold does not catch:** (1) with mono 400p + 16:9 colour, `--depth-size 640x360` and
  `derived` are the same pipeline but are treated as different tuples. (2) `--fps 30
  --video-keyframe-frequency 0,30` is also the same pipeline (0 → round(fps)).
* **`--mjpeg-quality` is not range-checked.** Even 150 or -5 gets through and fails only at device  [UNVERIFIED: 예시/유도값 — docs/MEASUREMENT_AUDIT.md]
  start time = it burns one run.
* **`--append` is refused if the column set differs** (rc=2, existing file left unmodified). That is why
  all the commands above use a different `--out` per stage. Merge the CSVs when reading them later.
* **It cannot measure the field of view.** The actual gain from Stage 4's 4:3 modes has to be
  checked separately by eye, comparing a still scene.
* **Valid only for this cable and this switch.** 91.5 Mbit/s is a property of the path, not of the
  camera.
* **This frontier was measured on an in-air scene.** MJPEG compression ratio is scene-dependent (with the same settings,
  it swung between 4.05:1 and 6.35:1 over 6 sessions), and turbid or dark water has fewer high-frequency components, so  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
  it compresses **better**. I.e. underwater, the colour bandwidth is likely to come out lower than what was measured
  here, and in that sense this ladder is **conservative**. But only the direction is known, not the magnitude,
  so do not assume that settings tuned in the pool will hold as-is over a reef.

---

# Test 2 — Depth accuracy and minimum distance

**Question:** How wrong is the C3's depth as a function of distance, and down to how many cm is it
actually valid? What does it take to reach the 18 cm the vendor claims?

**Premise:** `mono_res` must have been decided in Test 1. The ladder below is **for 400p**
(MinZ **300 mm**, **150 mm** with `--extended` — both measured, see 'What we already know' below).
If Test 1 picked 720p/800p, MinZ is 600.2 mm[유도], so **delete all of the 130/185/280/330 mm rungs** and
start from 660 mm.

## Physical setup

* **The target is "a fixed flat wall", not "a moving board".** Tile tag36h11
  sheets onto the wall on **matte** paper — do not use coated/glossy paper (specular reflection
  kills stereo matching).
* Fix a tape measure to the floor **perpendicular** to the wall, and move the camera along that tape.
  Using a wall guarantees flatness for free (a board bends and tilts), and at far rungs
  the trap of "the target is smaller than the analysis region" disappears structurally.
* Measure the tape from a **mechanical datum surface** you can put a square against (e.g. the housing front face).
  Always write down in `--datum` what that surface is.
* **Lighting: indoor ambient.** Do not use the onboard lamp — lamp illuminance changes as 1/Z², so the exposure
  conditions differ per rung. The wall is Lambertian, so under ambient light it has the same luminance regardless of distance.
* The rig must not move while a rung is being captured. The script
  detects it with `motion_madiff` and refuses.

## Required wall size

The ROI (central 40%) covers `0.499*Z x 0.280*Z`, the full frame covers `1.247*Z x 0.701*Z`.
The 2 m rung needs 997x561 mm for the ROI alone, and 2494x1402 mm for the full frame.

## Step 0 — Check the geometry first, without the camera (0 min)

```bash
V=~/.venvs/c3-depthai/bin/python
cd /home/bdml/Desktop/umi_underwater_robust_control

$V c3_camera/c3_depth_accuracy.py --dry-run --truth-mm 300
$V c3_camera/c3_depth_accuracy.py --dry-run --truth-mm 185 --extended
```

For each rung it prints MinZ, the disparity step size, the predicted overlap band, and the ROI
margin in pixels. **`--dry-run` never opens the device.**

## Step 1 — Shakedown (1 rung, ~1 min)

The capture path has **never once been run on hardware.** Before committing to the whole ladder,
briefly capture just one rung at a convenient distance.

```bash
$V c3_camera/c3_depth_accuracy.py --truth-mm 1000 --frames 15 \
    --label shakedown --datum "housing front face"
```

If it ends with `ok=1` and `rho_mm` is near 1000, proceed. If `ok=0`, first fix the printed
`refused` reason (usually aiming or lighting).

## Step 2 — CORE ladder, arm A/B (12 rung, ~7 min machine + moving time)

Distance is the **outer loop** (a person moves the camera and enters the tape-measure reading), arm
is the **inner loop**. At each station, run the two lines below back to back, then move to the next
distance.

```bash
# --- station: 130 mm (negative control — it is normal only if everything fails) ---
$V c3_camera/c3_depth_accuracy.py --truth-mm 130 --label default \
    --datum "housing front face" --median off --confidence 245
$V c3_camera/c3_depth_accuracy.py --truth-mm 130 --label ext --extended \
    --datum "housing front face" --median off --confidence 245

# --- then repeat the same two lines at 185, 280, 330, 1000, 2000 mm ---
```

At each station, just change the one `T=` line and paste the three lines below:

```bash
T=185
$V c3_camera/c3_depth_accuracy.py --truth-mm $T --label default \
    --datum "housing front face" --median off --confidence 245
$V c3_camera/c3_depth_accuracy.py --truth-mm $T --label ext --extended \
    --datum "housing front face" --median off --confidence 245
```

> Why `--median off` and `--confidence 245` are **pinned explicitly**: the `robotics`
> preset's confidence 245 means 245 out of 255, which is almost fully permissive, so much of the fill
> may be garbage, and the median filter fills holes and **lowers** the residual, making the camera look
> better than it really is. Without pinning them, a conclusion like "extended cut the fill" may be a preset
> artifact.

## Step 3 — Control arms C, D (5 rung, ~4 min)

```bash
# C — 800p far arm. At 2 m, 400p has a disparity step of 141 mm, so
#     the quantisation-limited warning appears. 800p is half that (70.4 mm).
#     Why --fps 2: 800p depth is 1280x720, so at 5 fps it is 133% OVER BUDGET.
for T in 660 1000 2000; do
  $V c3_camera/c3_depth_accuracy.py --truth-mm $T --label far800 \
      --mono-res 800p --fps 2 --timeout 60 --median off --confidence 245 \
      --datum "housing front face"
done

# D — overlap band control. Without --depth-align none the band cannot be seen.
for T in 185 280; do
  $V c3_camera/c3_depth_accuracy.py --truth-mm $T --label band --extended \
      --depth-align none --median off --confidence 245 \
      --datum "housing front face"
done
```

> **Why arm D is needed separately:** `--depth-align color` crops the stereo field of view from H 80.4° →
> 63.9°, so the close-range overlap-loss band is **cut completely out of the frame above 338 mm.**
> Looking only at align=color data, it looks as if this phenomenon does not
> exist. `--dry-run` itself prints it like this: *"the delivered frame is as wide as
> the stereo pair, so the whole band stays visible — this is the control
> configuration for measuring it."*

## Step 4 — Analysis (no camera, repeatable any number of times)

```bash
$V c3_camera/c3_depth_accuracy.py --analyze
```

## Run checklist

| ☐ | Arm | rung (mm) | Settings | Runs | What it answers |
|---|---|---|---|---|---|
| ☐ | shakedown | 1000 | default, `--frames 15` | 1 | Does the capture path run on hardware? |
| ☐ | A `default` | 130,185,280,330,1000,2000 | 400p, ext off | 6 | The actual valid range of the current defaults |
| ☐ | B `ext` | 130,185,280,330,1000,2000 | 400p, `--extended` | 6 | Does it really reach 18 cm? |
| ☐ | C `far800` | 660,1000,2000 | 800p, `--fps 2` | 3 | Does halving the quantisation reduce the far-range bias? |
| ☐ | D `band` | 185,280 | ext + `--depth-align none` | 2 | Does the overlap band come out as the formula says? |
| ☐ | analyze | - | `--analyze` | 0 | Scale a, origin c, ext off/on comparison |

Why each rung exists: **130** = below every MinZ → a negative control that must fail entirely.
**185** = the vendor's claimed 18 cm, the answer the user actually wants. **280/330** = a bracket
below/above the current default's 300 mm cliff. **1000/2000** = far anchors for fitting the
intercept c.

## Which column is the answer

* **"Down to how many cm does it work"** → `fill_roi_inlier` (≥ 0.5 means "it works here"),
  `z_min_mm`, `min_z_pred_mm`. Looking at the three side by side tells you whether prediction and measurement agree.
* **"How wrong is it"** → `bias_mm`, `bias_pct`. But look at the values after `--analyze` has fitted
  the origin offset.
* **"How noisy is it"** → `resid_rms_mm` (primary metric: plane-residual RMS, the pure surface
  reconstruction error with tilt removed). `patch_std_mm` is for comparison only, not a noise metric.
* **Band** → `hole_left` / `hole_centre` / `hole_right` / `hole_asym`, and
  the predictions `band_pred_px` / `band_pred_frac` / `band_delivered_frac`.
* **Accepted or not** → `ok`, `warn`, `refused`. `--analyze` drops rows with `ok=0` from the fit.

## How to read bad results

* **The 130 mm rung "succeeds" (`fill_roi` > 0.5)** → this is the worst outcome. Below MinZ
  there is physically no stereo information, so getting valid depth means median /
  hole-fill is **making values up**, and then **the fill numbers of every rung are
  invalid.** The script warns about this explicitly.
* **`quantisation-limited` in `warn`** → means the residual is smaller than `dZ/6`, a sign that the
  whole wall fell inside one disparity bin. `rho` is the **bin**, not the wall, and
  the bias carries the whole ±dZ/2. At 2 m/400p, dZ = 141 mm — larger than the origin  [UNVERIFIED: 예시/유도값 — docs/MEASUREMENT_AUDIT.md]
  offset (~25 mm) being fitted. Mitigation: use the 800p arm (arm C), or deliberately turn the wall a few
  degrees to dither the grid.
* **`fill_roi` is between 0.30 and 0.50 but `warn` is empty** → a known blind spot.
  Even if 40% is reported valid below MinZ, it passes silently. Thanks to `--fit-min-mm 450`
  it does not enter the fit, but it stays in the table, so filter it out by eye.
* **`--analyze` warns of a `c` mismatch between configs** → the optical centre cannot depend on
  the mono settings, so it means something other than the origin is wrong. First check whether `--datum` differed between rungs.
* **`refused: exposure`** → exposure exceeded 20 ms. Raise the lighting. The script
  **cannot fix the exposure, it can only detect it** (see the limitations below).

## What Test 2 cannot tell you

* **★ This test measures the sensor, not the water.** It is all **in-air** measurement. In water,
  refraction changes the whole scale (about 1.33× uncorrected). The coefficient that `--analyze` extracts,
  `a`, is exactly where that shows up — so if this experiment is repeated in water, `a` becomes the
  refraction scale — **but it has not been measured in water yet.** Do not cite the in-air bias
  as underwater performance.
* **The mono exposure cannot be fixed manually.** `StreamConfig` has no `mono_exposure_us` /
  `mono_iso`, and `pipeline.py` has no `MonoCamera.initialControl.setManualExposure()`
  path. Consequence: different rungs/arms **may be compared at different exposures.**
  The script puts gates on exposure level · exposure spread · motion and **only detects** it. This is the most
  valuable follow-up work.
* **There is no `--disparity-shift`.** It is an alternative that reaches 15 cm more cheaply than extended disparity,
  but it is not implemented, so that trade-off cannot be judged.
* **There is no `--tag-check`.** There is no independent second opinion on the fitted origin offset `c`
  — the cross-config consistency check is the only verification.
* **The delivered-band prediction models CAM_A as being at the same position as the stereo reference camera.**
  In reality they are ~37.5 mm apart, so at 185 mm that translation is 104 px (16.3% of the width),  [UNVERIFIED: 예시/유도값 — docs/MEASUREMENT_AUDIT.md]
  **larger** than the predicted 94 px band. The band is not clipped but **shifted**, so  [UNVERIFIED: 예시/유도값 — docs/MEASUREMENT_AUDIT.md]
  `band_delivered_frac` is unreliable precisely at close range. That is why arm D (`align none`)
  is needed.
* **At 400p / 1 m or more, `fixed_rms_mm` is dominated by quantisation, not by a "fixed pattern".**
  Measured example at 1 m: resid 10.01, temporal 5.19, fixed 8.56 — this happens even though it was synthesized with true sensor noise of 1 mm  [UNVERIFIED: 예시/유도값 — docs/MEASUREMENT_AUDIT.md]
  and ripple 0 (dZ/sqrt(12) = 10.2 leaks in).
  Read it with `quant_dz_pred_mm` subtracted. There is also a ~0.9 mm blind spot, so a true fixed pattern
  may read as exactly 0.  [UNVERIFIED: 예시/유도값 — docs/MEASUREMENT_AUDIT.md]
* **`temporal_rms_mm` uses only pixels that are valid in every frame.** It is optimistically biased,
  and with 30% holes/20 frames only 0.08% of the ROI remains, and at 40% the cell becomes empty altogether — exactly
  at the close-range holey rungs this study cares about.
* ~~**Unresolved whether MinZ is 298.9 or 300.1.**~~ **Resolved (2026-08-05): CAM_C.**
  An exhaustive survey of the recorded depth PNGs shows the floor is exactly **300 mm**, and with `--extended` exactly
  **150 mm**, with **not a single pixel** below that
  `[측정: c3_camera/datasets/*/depth/, c3_camera/recordings/*/depth/ — 47,270 frames,
  3.6e9 valid pixels, 2026-08-05]`. For the floor to read exactly 300, fx*B/95 ∈ [300, 301),
  i.e. fx_rect ∈ [380.0, 381.3), is required; CAM_C's 380.14 is inside that, CAM_B's 378.56 is
  outside (it would have read 298). That 300:150 is exactly 2:1 also cannot be explained except by disparity 95→190
  (this pipeline has neither a threshold filter nor disparityShift).
  It also agrees within 0.04% with the ~380.00 that `host_depth.py` measured from host-side rectification.
* **`--save-npy`'s `run_id` has 1 s resolution.** Harmless in a human-speed protocol, but
  when run from a script, the file names collide.
* **The analysis artifacts are only text and CSV.** Without `--save-npy` (off by default), you cannot later
  re-analyze from the raw data; only what the CSV holds remains.

---

# Test 3 — Encoder image quality

**Question:** Within a given Mbit/s budget, which (codec, rate) combination gives the most
**usable** detail per second?

> **Know this first:** the hypothesis that "H.264/H.265 solves the bandwidth problem" is **wrong at the
> current operating point.** In a measurement (dataset_20260803_105221), colour MJPEG at 6.8 Mbit/s is  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
> **only 14%** of the link, while depth uint16 at 41.5 Mbit/s eats **86%**. Even a perfect codec that makes colour 0 bytes
> recovers only 7.4% of the ceiling, and the total is already 53%, so **the fps gain is  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
> exactly 0**. The encoder question becomes binding **only when colour resolution is raised**.

**Physical setup:** reuse Test 2's wall as is. Two requirements:

* It must be **static** (rig fixed, scene not moving). Nothing may move while one setting is
  being measured.
* It must be **richly textured**. AprilTag sheets meet this condition. If the reference has fewer than
  500 ORB keypoints, the script flags the whole capture as `under_textured` and
  excludes it from the conclusions (`--min-ref-keypoints`).

**Total time:** 20 settings x 32 s = **~11 min**.

## Step 0 — Plan the ladder without the camera (0 min)

```bash
V=~/.venvs/c3-depthai/bin/python
cd /home/bdml/Desktop/umi_underwater_robust_control

$V c3_camera/c3_encode_quality.py --dry-run \
    --codec none,mjpeg,h264,h265 --mjpeg-quality 80,90,97 \
    --bitrate-kbps 2000,4000,8000,16000 --keyframe 0,1
```

If the output says 20 settings / ~10.7 min, it is going to plan.

## Step 1 — Real capture (11 min, **opens the device**)

```bash
$V c3_camera/c3_encode_quality.py \
    --codec none,mjpeg,h264,h265 --mjpeg-quality 80,90,97 \
    --bitrate-kbps 2000,4000,8000,16000 --keyframe 0,1 \
    --streams color --fps 20 --target-fps 19 \
    --scene-label static \
    --out c3_camera/encode_quality/run1
```

> Why `--streams color`: 640x360 uint16 depth is ~74 Mbit/s, so it swallows the whole 2-30 Mbit/s
> colour ladder and the encoder axis becomes invisible.
>
> `--codec` must include `none`. The raw arm is the fidelity baseline and is not
> optional. At 960x540@20 raw is 138% of the link ceiling, so it drops to ~9 fps; this is
> **normal**, and in the ranking it is disqualified for falling short of `--target-fps`. Producing the reference is  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
> that arm's job.

## Step 2 — Re-scoring (no camera, any number of times)

```bash
$V c3_camera/c3_encode_quality.py --offline c3_camera/encode_quality/run1 \
    --target-fps 19 --budget-mbps 25
```

Running it many times with different `--budget-mbps` values is the whole point of this script. Also
check whether the ranking flips when you switch the deciding metric with `--rank-by ssim_y`.

## Step 3 (optional) — 1080p arm (4 settings, ~2 min)  [UNVERIFIED: 예시/유도값 — docs/MEASUREMENT_AUDIT.md]

The only operating point where the encoder question actually becomes binding. **Resolution is not a
sweep axis (the reference median stack is valid at only one resolution), so you must use a separate
`--out` directory, and later merge the `runs.csv` files by hand when reading them.**

```bash
$V c3_camera/c3_encode_quality.py \
    --codec none,mjpeg,h265 --isp-scale none --mjpeg-quality 90 \
    --bitrate-kbps 8000,16000 --keyframe 1 \
    --streams color --fps 20 --target-fps 19 \
    --duration 20 --reference-frames 15 \
    --out c3_camera/encode_quality/run2_1080p
```

> `--duration 20 --reference-frames 15` is mandatory. 1080p raw gets ~3.8 fps over the link, so  [UNVERIFIED: 예시/유도값 — docs/MEASUREMENT_AUDIT.md]
> with the default `--duration 10` there are only ~38 frames, and `--reference-frames 25` eats almost
> all of them, so the score stage stops with an explicit error.

## Run checklist

| ☐ | Step | Command | Settings | Time | What it answers |
|---|---|---|---|---|---|
| ☐ | 0 | `--dry-run` | 20 | 0 | Ladder · time plan |
| ☐ | 1 | `--out .../run1` | 20 | 10.7 min | rate-quality curve at 960x540 |
| ☐ | 2 | `--offline .../run1 --budget-mbps 25` | - | 0 | Winner per budget |
| ☐ | 2b | `--offline ... --rank-by ssim_y` | - | 0 | Does the ranking flip with a different metric? |
| ☐ | 3 | `--out .../run2_1080p` (optional) | 4 | 2.1 min | 1080p H.265 vs 540p MJPEG |
| ☐ | 4 | Rerun Step 1 with `--scene-label motion` (optional) | 20 | 10.7 min | P-frame penalty |

## Which column is the answer

* **Deciding metric: `orb_inlier_rate`** (the default `--rank-by`). It is what the SLAM front end actually
  consumes, and it separates settings that PSNR scores as a tie. The spread is `orb_inlier_sd`.
* **For reference:** `ssim_y`, `psnr_y_db` (+ `_sd`, `_p05`).
* **Never use as a deciding metric:** Laplacian variance (`lap_var`). Compression can actually
  **raise** the value, so its direction flips depending on the scene. `RANKABLE_METRICS`
  deliberately leaves it out.
* **AprilTag:** look at `tags_known` (detections against the known ID map) and `tags_false` **separately**.
  Do not look at `tags_detected` alone — compression artifacts create false-positive detections, so
  the detection count **goes up** (measured: 11 → 13). If `tags_false` is not 0, that rate is  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
  unfit for the dataset path.
* **Link cost:** `measured_mbps`, `budget_frac`, `color_kb_frame`,
  `color_kb_frame_p95`, `burstiness` (= p95/mean; whether keyframe peaks fill the queue and
  create latency spikes).
* **Integrity:** `extraction_ok`, `frames_extracted` vs `frames_saved`. Even with good SSIM,
  a codec whose extraction fails 1 time in 5 is **not a better codec.**
* **One-line conclusion:** the `DECISION:` at the very bottom of the output and the margin line under it. If the margin is
  within the per-frame `sd`, it prints `UNRESOLVED` — then it is a coin toss.

## How to read bad results

* **`UNRESOLVED` under `DECISION`** → the margin between 1st and 2nd place is smaller than the per-frame spread.
  Either one is fine to pick; if you really want to separate them, increase `--duration` to collect more frames, or
  compare again with a more widely spaced rate pair.
* **`under_textured = 1`** → the reference has fewer than 500 ORB keypoints. That is the scene's fault,
  not the codec's. Move to a wall with more texture and recapture.
* **`SCENE WAS NOT STATIC (or the sensor is very noisy)`** → the rig moved, or
  the gain is too high. The 0.7 threshold is uncalibrated, so even on a static rig, 12 DN of sensor noise  [UNVERIFIED: 예시/유도값 — docs/MEASUREMENT_AUDIT.md]
  drops the value as low as 0.682 and a false alarm fires. Check the mount, then look at gain/exposure.
* **`extraction_ok = 0`** → the h26x stream cannot be read back. However good the quality numbers are,
  it is disqualified.
* **Rows at the top bitrates (8000-16000 kbps) bunch together and cannot be told apart** → you are reading the
  **host decoder's upper bound**, not the encoder (see the limitations below). Do not cite
  MJPEG-vs-H.26x rankings in this range.

## What Test 3 cannot tell you

* **★ The reference is not the same frame.** The DepthAI pipeline is fixed at boot, so
  each setting is a separate connection, and therefore reference and test are **different exposures
  of a static scene**. Folding the raw arm into a temporal median stack to remove shot noise is
  the best that can be done, but it is still not same-frame PSNR. **Use these numbers only to RANK
  settings, and never cite them as a fidelity claim.** That is why `CORRESPONDENCE_NOTE` is printed alongside every table,
  `report.md`, and `meta.json`.
* **h26x rows have an upper bound that has nothing to do with the codec.** raw / mjpeg / h26x each take a different host
  decode path (OpenCV NV12 / libjpeg / ffmpeg→PNG→imread). Measured directly on this host:
  feeding a **bit-exact H.264 stream with exactly 0 loss** through the same scoring path gives
  PSNR **45.60 dB** / SSIM **0.9962** / ORB inlier **0.9156**. I.e. h26x  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
  rows cannot exceed those values however high the bitrate goes. **At the top of the ladder (8000-16000
  kbps), an MJPEG-vs-H.26x comparison reads the decoder, not the encoder.**
  In the same experiment MJPEG agreed with the true loss within 0.1 dB, so it has no such penalty.  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
* **The codec axis is confounded with elapsed time.** The run order is always
  `none → mjpeg → h264 → h265`, and the reference is built from the raw run, which goes first.
  With 20 settings, the last h265 row is **~10 min** away from the reference, and slow drift in lighting · exposure · scene
  is all charged to the settings that ran late. How to bracket it: run once more with the `--codec` order
  reversed, or capture one more raw arm at the end and compare the two raw rows.
* **A static scene is the best case, the most favourable one for inter-frame codecs.** The H.264/H.265 rows
  are overestimated. The real risk — P-frames collapsing under surge + backscatter —
  only becomes visible when rerun with `--scene-label motion` (checklist Step 4).
* **H.264 is fixed to BASELINE** (`pipeline.py:179`). It measures only the CABAC-less lower bound,
  and since depthai 2.32 has H264_MAIN / H264_HIGH, adding a profile axis
  may change the results.
* **Resolution is not a sweep axis.** Because the reference median stack is valid at only one
  resolution. To get a per-resolution upper envelope, run each resolution with a separate `--out`, then
  merge the `runs.csv` files (Step 3).
* **AprilTag corner σ (subpixel precision) is not implemented.** All that is implemented is the detection count and
  the false-positive ID count.
* **The h26x winner cannot be carried over to `c3_collect.py` yet.** `source.py:429`
  calls `msg.getFrameType()`, which `dai.ImgFrame` does not have, so `frame_type` is always
  an empty string and the first-I-frame gate in `dataset.py:341-344` does not open. This script  [UNVERIFIED: 예시/유도값 — docs/MEASUREMENT_AUDIT.md]
  worked around it with its own `annexb_frame_type()`, but **c3_collect's h26x dataset capture
  still writes 0 bytes.** To adopt an encoder for the dataset path, `source.py` has to be fixed
  first.
* **The caveat is not attached to `runs.csv` / `scores.csv` / the per-run `meta.json`.**
  It goes only into the tables · `report.md` · the top-level `meta.json`. What gets carried into figures or papers
  is usually `runs.csv`, so be careful.

---

# How to use the results — candidate config.py defaults

These are the **values to change** when the results come out well, and the **conditions for changing
them**. If a condition is not met, do not touch the value.

| Location | Current value | Value to change to | Condition (which test approves it) |
|---|---|---|---|
| `config.py` `StreamConfig.depth_size` | `None` (derived → 640x360 at 400p) | `(480, 270)` | When in T1 Stage 1b 480x270 passes `hold()` at both 15/20 fps and 640x360@20 does not pass |
| `config.py` `StreamConfig.isp_scale` | `(1, 2)` → 960x540 | `(2, 3)` → 1280x720 | When in T1 Stage 2, 2/3 @15 passes `hold(15)` and the Stage 3a corner also passes |
| `config.py` `StreamConfig.mono_res` | `"400p"` | Leave as is | **If T2 wants close range, keep 400p.** 800p doubles MinZ, 300 → 600.2 mm. Reconsider only if arm C shows a large far-range bias improvement |
| `config.py` `StreamConfig.extended` | `False` | `True` | When T2 arm B's 185 mm rung is `ok=1`, and the `fill_roi_inlier` loss relative to arm A at ≥ 300 mm is acceptable |
| `config.py` `StreamConfig.confidence` | `None` (→ preset 245) | An explicit value (e.g. `200`) | When T2 produces evidence that 245 is making up fill (`fill_roi` of the 130 mm rung) |
| `config.py` `StreamConfig.median` | `"5x5"` | `"off"` on the measurement path | When in T2 5x5 only lowers the residual and gives no fill gain. **Keep 5x5 on the preview path** |
| `config.py` `StreamConfig.mjpeg_quality` | `90` | `80` | When in T3 the `orb_inlier_rate` difference between q80 and q90 is within `orb_inlier_sd` (= `UNRESOLVED`) |
| `config.py` `MJPEG_RATIO` | `6.0` (scalar) | A per-resolution table | When T1's `color_kb_frame` reconfirms the 8:1 range at 1080p. For now the 1080p estimate is ~1.4× conservative |
| `config.py` `POE_BUDGET_MBPS` | `90.0` | The measured value | Only when the T1 Stage 1a anchor deviates from the README's 91.5 by more than 5% (meaning the cable/switch has changed) |
| `c3_collect.py` `RESEARCH_DEFAULTS["color_encode"]` | `"none"` | `"h265"` or `"mjpeg"` | When T3 shows it is indistinguishable from raw at the operating point **and at the same time** the `getFrameType` bug at `source.py:429` has been fixed. If either one does not hold, keep `none` |
| `c3_collect.py:223` `--min-mm` default | `300.0` | `100.0` | If `extended` has been made the default. 300 wipes the whole 149-300 mm interval that extended opens up off the display |

**Two rules when changing them**

1. **Do not change them without measured evidence.** In particular, the encoder default of `c3_collect`
   is defensible only if the supporting numbers (e.g. `ORB inlier 0.94 vs raw, tags_false 0`) are recorded alongside it in `metadata.json`.
   A dataset labelled only `"H.265"` is not defensible.
2. **Never encode depth.** Passing 16 bit depth through an 8 bit video codec
   is a category error. The repo has never done so, and it should stay that way.

**What you can get right away (independent of the test results)**

The current dataset operating point (colour 480x270 + depth 480x270 @20 fps) uses **only 53%** of the
link. Raising colour to 960x540 MJPEG q90 gives 21.5 + 41.5 = 62.9 Mbit/s =  [UNVERIFIED: 예시/유도값 — docs/MEASUREMENT_AUDIT.md]
**69%** of the ceiling, so you can get **4× the pixel area without changing a single codec**, right now.
69% is above the jitter-free zone (≤ 60%), but not in the drop zone (≥ 85%) — T1 Stage 1b is exactly
this confirmation run.


<!-- MEASUREMENT AUDIT (2026-08-04): the following numbers appear inside code
     blocks above and could not be annotated inline without corrupting them. -->

> **[UNVERIFIED]** The figures on the line below were classified as unverified by the provenance audit — details in [docs/MEASUREMENT_AUDIT.md](docs/MEASUREMENT_AUDIT.md).
>
> - `c3_camera/TESTING.md:130` — 산출물 없음 — `#      "960x540 mjpeg @15 + depth 640x360" = 71.4 Mbit/s (78%), p50 39 ms, p95 8`
> - `c3_camera/TESTING.md:131` — 산출물 없음 — `#      @20 행은 인코더 표에서 19.1 fps / drop 5%.` ("the @20 row in the encoder table is 19.1 fps / drop 5%.")
