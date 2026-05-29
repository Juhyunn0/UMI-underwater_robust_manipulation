# fisheye_gantry_tagslam — usage notes

Four scripts live under `src/`:

| script                          | purpose                                                                                       |
|---------------------------------|-----------------------------------------------------------------------------------------------|
| `gantry_panel.py`               | **PyQt5 live control panel** — connect, jog, home, move, record, run waypoint sequences.       |
| `gantry_runner.py`              | Headless CLI: drive the FMC4030 to XYZ targets in mm; log telemetry CSV at ~100 Hz.            |
| `fisheye_gantry_tagslam.py`     | Fisheye camera → undistort → AprilTag → GTSAM iSAM2, alongside gantry motion + telemetry.     |
| `zed2_underwater_tagslam.py`    | Original ZED2 TagSLAM (unchanged CLI/output; now imports from the shared `tagslam_core`).     |

Reusable backend: `tagslam_core.py` (detection, refractive PnP, iSAM2, CSV/HTML/plot writers, OpenCV visualizer).

Running `python src/gantry_runner.py` **with no arguments** launches the PyQt5 panel
directly (it just calls `gantry_panel.main([])`). The CLI mode is unchanged when you
pass any flag.

---

## Frame & sign conventions to confirm per rig

* Gantry XYZ in millimeters in the controller's native frame. Per-axis unit
  scaling lives in `gantry_runner.SCALE_MM_PER_UNIT` (X=8.25, Y=2.5, Z=0.5
  mm/unit — copied from `src/gantry/demos/whisker_dragging.py`).
* `T_gantry_camera` (4×4) in the fisheye calibration YAML maps a point in
  the camera body frame to a point in the gantry frame. **Translation column
  is in METERS** (matches SLAM units). Any +Z-up vs +Z-down convention flip
  is absorbed by the rotation block — the runtime makes no axis-sign assumption.
* SLAM world = anchor AprilTag pose. Gantry world = controller XYZ. Without
  a separate anchor↔gantry calibration, `translation_error_mm` is the
  literal `||p_est − T_gc·p_gantry||` and includes the unknown frame
  offset. For drift-only error, subtract the first sample in post-processing.

---

## Gantry Control Panel (PyQt5)

`src/gantry_panel.py` is the day-to-day operator UI. It binds the move /
log / sequence primitives in `gantry_runner.py` to live widgets with a
dark theme and a worker-per-task threading model.

> **Removed by user request:** the panel no longer exposes a soft-limit
> editor, the live soft-limit watchdog, or physical limit-switch
> calibration ("Calibrate X/Y/Z to Limit"). The Setup tab is now reduced
> to **Set Current as Home Reference** + Axis Direction toggles. Whatever
> soft limits are stored on the firmware are still enforced by the
> controller; the panel just doesn't read/write them anymore. Use
> `gantry_runner.py`'s CLI flags (`--soft-limit-min-mm` /
> `--soft-limit-max-mm`) if you need to control firmware-side limits
> from the command line.

> **Units:** Velocities and accelerations are displayed and entered in **cm/s**
> and **cm/s²**. Positions remain in **mm**. Calibration source:
> `SCALE_MM_PER_UNIT` in `gantry_runner.py` (X = 8.25, Y = 2.5, Z = 0.5
> mm/unit). The panel converts cm/s → controller units/s per axis at the
> callsite using those constants. Pool dimensions for the workspace map are
> read from `config/config.yaml` (`pool.length_m`, `pool.width_m`,
> `pool.depth_m`). The physical orientation of the pool in the gantry frame
> is set independently in the **Setup tab → Pool Orientation** group (default:
> pool long axis = gantry X), persisted to `~/.umi_gui_state.json`.

### Launching

```bash
python src/gantry_panel.py            # real controller
python src/gantry_panel.py --mock     # in-process simulation, safe to click around
python src/gantry_panel.py --light    # skip the dark theme
python src/gantry_runner.py           # same as the first one (no-args dispatch)
```

### Optional dependencies

All graceful-fallback. Install the ones you want:

```bash
pip install PyQt5            # required
pip install pyqtgraph        # optional, enables the live position plot
pip install qdarkstyle       # optional, replaces the built-in dark stylesheet
pip install qtawesome        # optional, adds icons on the action buttons
```

Without `pyqtgraph` you get a "Install pyqtgraph for live plots" notice in
place of the plot. Without `qdarkstyle` the panel uses its own hand-written
dark stylesheet. Without `qtawesome` the buttons are text-only.

### Layout sketch (split: left pane always pinned, right pane is tabbed)

```
┌── Menu: File · View · Help ─────────────────────────────────────────────┐
├── Connection: IP · Port · ID · [Connect] · [X][Y][Z] · ●Connected ──────┤
├──────────────────────────────┬──────────────────────────────────────────┤
│  LEFT PANE  (always)         │  RIGHT PANE  (QTabWidget)               │
│                              │ ┌──────────────────────────────────────┐ │
│  ┌─ Live Status ──────────┐  │ │ [Control] [Sequences] [Setup] [Rec] │ │
│  │ X  card  Y card  Z card│  │ ├──────────────────────────────────────┤ │
│  │  +123.4   +50.0  +10.0 │  │ │                                      │ │
│  │  vel/acc  vel/acc vel  │  │ │  (tab content fills the rest)        │ │
│  │  [▓▓░] mm [░▓░] [▓▓░]  │  │ │                                      │ │
│  └────────────────────────┘  │ │                                      │ │
│  ┌─ Workspace Map ────────┐  │ │                                      │ │
│  │ ☐ Trail ☐ Target ☐ Fit │  │ │                                      │ │
│  │  ╔══ Top-down (XY) ══╗ │  │ │                                      │ │
│  │  ║   • dot · ◯ tgt   ║ │  │ │                                      │ │
│  │  ║   ╌  trail  ╌╌╌╌  ║ │  │ │                                      │ │
│  │  ╚═══════════════════╝ │  │ │                                      │ │
│  │  ╔══ Side (XZ) ══════╗ │  │ │                                      │ │
│  │  ║   • dot · ◯ tgt   ║ │  │ │                                      │ │
│  │  ╚═══════════════════╝ │  │ │                                      │ │
│  └────────────────────────┘  │ └──────────────────────────────────────┘ │
├──────────────────────────────┴──────────────────────────────────────────┤
│  [🔄 Refresh] [Pause] [Resume] [Stop Run] ────── [⚠ EMERGENCY STOP ALL] │
├─────────────────────────────────────────────────────────────────────────┤
│  ●Connected ── Idle ──────────────────────── ● RECORDING <file.csv> ────┘
```

The right pane's four tabs are:

| Tab | Contents |
|---|---|
| **Control** | Per-Axis Control (jog + Move Abs + per-axis blue Go Home cards) + the combined Move to Target panel. |
| **Sequences** | The waypoint `QTableWidget` plus Add Row / Remove Selected / Load CSV / Save CSV / Run Sequence / Stop Sequence. |
| **Setup** | Home Reference group (Set Current as Home Reference + per-axis Axis Direction toggles). |
| **Recording** | Start/Stop Recording, the clickable CSV path label, and the 30 s rolling per-axis position plot. |

The left pane stays visible no matter which tab is active, so you always see live position and the workspace map. The splitter between left/right is draggable; default ratio 38/62, window default `1500 × 900`, minimum `1280 × 800`. The last-active tab is persisted to `~/.umi_gui_state.json` under `gantry_panel.active_tab` and restored on next launch.

The **Workspace Map** renders two stacked 2D plots: top-down `XY` (aspect-locked square) and side `XZ` (shares the same X axis as the top view). Each shows:
- Light-blue dashed **pool outline** (dimensions from `config.yaml`, orientation from Pool Orientation setting)
- Gray dotted **soft-limit envelope** (when limits are loaded)
- Current-position dot (filled green) with crosshairs
- Target marker (hollow blue circle) when a Move-to-Target is active
- A 200-sample trail

The **Fit** dropdown (top of map) has three modes:
| Mode | Behaviour |
|---|---|
| **Fit: Pool** *(default)* | View locked to the pool boundary. XY shows ≈ 4877 × 1800 mm or 1800 × 4877 mm depending on Pool Orientation. |
| **Fit: Soft Limits** | View expands to the controller's loaded soft-limit envelope. |
| **Fit: Trail + Target** | View auto-fits to current trail ∪ target with 50 mm margin. |

Tick spacing adapts to the visible range (~6–10 major ticks at any zoom). Implementation is `pyqtgraph` when installed; falls back to a hand-painted `QPainter` widget otherwise. All updates ride the 10 Hz status-poll — no extra SDK calls.

### Per-Axis Control (jog + Move Abs + per-axis Home)

Between the Homing group and the combined Move-to-Target panel, the **Per-Axis
Control** section gives you direct one-axis control of X, Y, and Z. Three
side-by-side cards, one shared parameter row at the top:

```
┌─ Per-Axis Control ──────────────────────────────────────────────────────┐
│  Jog/Move Speed [10.00] cm/s   Accel [5.00] cm/s²   Decel [5.00] cm/s² │
│  ┌─ X ──────────┐  ┌─ Y ──────────┐  ┌─ Z ──────────┐                   │
│  │  +123.456 mm │  │  +50.000 mm  │  │   +0.000 mm  │  ← live readout   │
│  │  Vel: +0.00  │  │  Vel: +0.00  │  │  Vel: +0.00  │                   │
│  │  [X+]  [X-]  │  │  [Y+]  [Y-]  │  │  [Z+]  [Z-]  │  ← hold to jog    │
│  │  [_____] Abs │  │  [_____] Abs │  │  [_____] Abs │  ← Move Abs (mm)  │
│  │  Home X      │  │  Home Y      │  │  Home Z      │  ← shortcut       │
│  └──────────────┘  └──────────────┘  └──────────────┘                   │
└─────────────────────────────────────────────────────────────────────────┘
```

* **Jog buttons (`X+/X-/Y+/Y-/Z+/Z-`)** — hold to jog continuously in that
  direction at the shared Jog/Move Speed; release for an SDK soft stop
  (`stop_axis(mode=1)`). Same pattern as `manual_pad.py`'s `_start_jog` /
  `_stop_jog`. Speed is entered in cm/s; converted to controller units/s
  per axis via `SCALE_MM_PER_UNIT` at the callsite.
* **Move Abs spinbox + button** — absolute move on this one axis only. The
  spinbox range is clamped to the loaded soft limits per axis (in mm), and the
  click re-validates against them. Runs on its own background
  `AxisAbsMoveThread` (one per axis, never blocks the GUI).
* **Per-axis Home button** — pure UI shortcut into the existing
  `_home_single(axis)` path. The shared homing parameters from the Homing
  group above (Home Speed in cm/s, Accel/Decel, Fall Step in mm, Direction) are
  re-used verbatim, so there's exactly one place to tune homing.
* **Shared jog/move parameter row** — `Jog/Move Speed`, `Accel`, `Decel` in
  cm/s and cm/s². Independent from the combined Move-to-Target panel's own
  speed/accel — each panel keeps its own values.
* **Safety**: every jog / Move Abs / per-axis Home goes through the same
  `_controller_lock` as everything else; per-axis controls disable when
  disconnected, when any homing or sequence is in progress, or when an
  absolute move is in flight on any axis. Emergency Stop stays enabled
  unconditionally.
* **Recording auto-toggle**: same rule as Move-to-Target — if recording isn't
  already manually on, jog and Move Abs auto-start the
  `GantryTelemetryLogger` and auto-stop ~500 ms after the axis reports
  stopped.

### Home reference workflow

The Setup tab now has a single section: **Home Reference**.

1. Manually jog the gantry to where you want home to be.
2. Click **Set Current as Home Reference**. The panel snapshots the current
   absolute machine-frame XYZ from the SDK and stores it as the per-axis
   home offset. User-frame readouts on each Control-tab axis card jump to
   `+0.000` mm.
3. **Axis Direction** toggles (`+1` / `-1` per axis) live in the same
   section. If clicking X+ on the panel moves the gantry in what you
   consider the negative direction, flip that axis to `-1`. The panel
   inverts the user-facing direction (display + commanded jog/move) without
   touching the firmware counter. Persisted to `~/.umi_gui_state.json`
   under `gantry_panel.axis_sign`.

The blue **Go Home** button on each Control-tab axis card does a Move Abs
to user-frame 0 mm for that axis (i.e. back to the captured reference).
This is the everyday "return to home" operation.

### Frame convention (user-frame mm everywhere)

Every mm value you see in the panel is in **user-frame**:

- **HOME-RELATIVE** — zero is wherever you last captured the home reference
  via Setup → Set Current as Home Reference.
- **SIGN-FLIPPED** — when an axis has Axis Direction = −1, the panel's `+`
  matches your physical intuition even if the firmware counter decreases.

All UI fields share this frame: the per-axis position readout, the
Move-to-Target spinboxes, the per-axis card's target spin, the waypoints
table, the workspace map. Internal absolute machine-frame mm is only used
at the SDK boundary and is exposed for diagnostics in the position tooltip
(`abs +XXX.XXX mm`) and in `run_metadata.json` (the
`home_reference_abs_mm` field lets you reconstruct user-frame from the
absolute samples in `gantry_telemetry.csv` post-hoc).

### Emergency Stop semantics

The big red **EMERGENCY STOP ALL (Esc)** button at the bottom of the window
and the **Esc** keyboard shortcut both call the same handler, which is
designed to **never block on the SDK lock**: it `acquire(timeout=0.05)`'s
the controller lock, then issues `controller.stop_axis(axis, mode=2)` on
every axis **whether or not the lock was acquired**, releasing in `finally`
only if it was. Rationale: the controller serializes incoming commands in
its own queue, so a stop going through during another in-flight call is
strictly safer than waiting for the lock and never stopping. Smoke-test
timing (mock): click → handler return in **~3 ms**, halt distance after
click **≈ 0 mm**. The handler also sets a panel-scoped
`self._abort_event = threading.Event()` (in addition to the module-level
`EMERGENCY_STOP` from `gantry_runner.py`); `HomingThread`, `SequenceThread`,
and `AxisAbsMoveThread` all check both at every poll iteration and exit
early. The Esc shortcut uses `Qt.ApplicationShortcut` context so it fires
even when keyboard focus is in a spinbox, combobox, or table cell (verified
on offscreen Qt with focus on a `QDoubleSpinBox`: handler ran in **~3 ms**).

When an E-Stop fires, a **yellow banner** appears across the top of the
window (`EMERGENCY STOP triggered at HH:MM:SS — click Reset to resume`)
with a **Reset E-Stop** button. The banner stays up until the user
explicitly clicks Reset; while it's up, the panel refuses to start new
motion (move / jog / Move Abs / Home / Run Sequence) and pops a warning
telling the user to reset first. A non-blocking `QMessageBox.information`
dialog is also shown summarising per-axis stop results and any SDK error
codes — failures surface, they aren't swallowed.

- **Ctrl+C in terminal** — same effect; the SIGINT handler bounces the work
  back onto the Qt thread via `QTimer.singleShot(0, …)` then quits the app.
- **Window close** — stops the timer, stops the logger, interrupts every
  worker thread, closes the controller. Same path as `gantry_runner.py`'s
  CLI cleanup.

### Home reference (Δ home column + workspace map marker)

Each Live Status card carries a third readout line — `Δ home: +73.45 mm`
in yellow — showing the axis's current position relative to its captured
home reference (gray "—" until a reference is set). The reference is
captured automatically when a `home_axis(...)` operation completes
successfully for that axis, or manually via the **Set Current as Home
Reference** button in the Setup tab → Homing group (which captures all
three axes from the latest poll snapshot without commanding any motion).
The reference is persisted to `~/.umi_gui_state.json` under
`gantry_panel.home_position_mm` and restored on next launch. The
Workspace Map shows the home position as a yellow star marker in both the
XY and XZ views, with a dashed yellow line from home to the current
position dot, and a header row above the plots — `Home: X=… Y=… Z=…
(Δ from current: … mm)` — that updates live. A `Polling: ✓ N ms ago`
indicator at the top of the left pane (green ≤ 1000 ms, red + "STALE"
beyond) makes it impossible to be confused about whether the live readout
is updating.

### Per-axis Home Direction

Each axis homes toward a different end-stop depending on the physical
limit-switch wiring — for a downward-mounted tool, typical defaults are
**X = Positive limit, Y = Positive limit, Z = Negative limit**. The Setup
tab → Homing group has three per-axis direction dropdowns (matching the
two-item combo from `manual_pad.py`: `Negative limit` (False) and
`Positive limit` (True)) below the shared Home Speed / Accel / Fall Step
inputs. The selected direction is passed as
`controller.home_axis(..., positive_limit=...)`; per-axis Home buttons on
the Per-Axis Control cards consume the same combo for that axis; the
shared Home All button respects each axis's combo in the chosen order.
Selections are persisted to `~/.umi_gui_state.json` under
`gantry_panel.home_direction.{X,Y,Z}` and restored on launch. The
confirmation dialog spells out the direction:

```
Home Z toward NEGATIVE limit at 5.00 cm/s (≈ 100.00 units/s on Z → clamped to 20.0).
Make sure the path is clear. Continue?
```

### Recording

- **Auto**: any Move-to-Target or Run Sequence starts a fresh CSV in
  `data/YYYYMMDD/<ts>_gantry_run/` if no recording is active. Auto-stops
  ~500 ms after the motion completes.
- **Manual**: click **● Start Recording** before any motion. The same CSV
  captures multiple consecutive moves; click again to stop. The auto-stop
  is suppressed while in manual mode.
- The CSV path under the button is clickable — it opens the run folder in
  your OS file manager.

### Mock mode for smoke tests

```bash
python src/gantry_panel.py --mock
```

`MockFMC4030Controller` lives in `gantry_panel.py`. Each axis simulates
linear motion at the commanded speed toward its target; `get_status()`
interpolates from monotonic time so the live readouts, the progress bars,
and the plot all animate as if you were on real hardware. Soft-limit
apply, homing, sequence, recording (writes a real CSV), and emergency stop
all work end-to-end against the mock.

---

## Fisheye Calibration Tool

`src/calibrate_fisheye.py` is a PyQt5 GUI that walks you through fisheye
intrinsic calibration and writes a YAML file directly loadable by
`load_fisheye_calibration()` in `fisheye_gantry_tagslam.py`.

### Launching

```bash
# Real camera (device 0, 1280×720 default)
python -m src.calibrate_fisheye

# Specific device / output path
python -m src.calibrate_fisheye --device 1 --output config/my_calib.yaml

# Mock camera — synthetic chessboard frames, no hardware needed
python -m src.calibrate_fisheye --mock-camera
```

### Workflow (Record / Stop + automatic frame selection)

1. **Connect** — choose device, resolution, FPS; click **Connect**.
   The live preview starts immediately with corner overlay.
   With `--mock-camera` a synthetic 9 × 6 chessboard animates so every
   region of the image is covered automatically.

2. **Pattern** — set inner corners (cols × rows), square size, and the three
   calibration flags.  Defaults: 9 × 6, 25 mm/square.  The selection summary
   line shows what the algorithm will do before you start recording:
   `4×4 grid · 2 frames/cell · target ≈ 32 frames · ≥12 cells required`.

3. **Record** — click **● Record** (or press **R**).
   Move the calibration board continuously across the image.
   Watch the **live 4 × 4 coverage grid**: cells turn green as the board
   visits each region.  Watch **Live sharpness**: keep it green (> 150) —
   slow, steady motion is better than fast waving.  The button changes to
   **■ Stop & Calibrate** while recording.

   **Coverage tips:**
   - Move the board to *all four corners and all edges* of the image —
     fisheye distortion is strongest near the edges, so edge coverage
     matters most for `k3`/`k4`.
   - Tilt the board ± 30–45° in multiple directions; don't stay face-on.
     The algorithm rewards tilt diversity (30 % of the per-frame score).
   - Vary distance: one close pass (board fills ≥ 50 % of the frame) and
     one far pass.
   - Record at least 10–20 seconds; 30 s gives ≥ 16 cells easily.
   - Keep sharpness green.  If it's yellow/red, slow down or improve
     lighting before recording.

4. **Stop & Calibrate** — click **■ Stop & Calibrate** (or press **R** again).
   A progress dialog runs the automatic selection pipeline:

   | Step | What happens |
   |------|-------------|
   | Hard gate | Drops frames with sharpness < 50 and duplicates within 200 ms |
   | Group by cell | Divides the image into a 4 × 4 grid; groups survivors by cell |
   | Preliminary calibration | Runs `cv2.fisheye.calibrate` on one frame per cell |
   | Score | Rates each survivor: sharpness 50 % + tilt 30 % + reproj 20 % |
   | Top-K per cell | Picks the 2 highest-scoring frames from each occupied cell |

5. **Coverage report** — a modal shows how many frames were picked and the
   coverage map.  If ≥ 14 cells are covered the default action is
   **Proceed**; if < 12 cells are covered the default is **Record again**
   with a strong warning.  Right-click any picked thumbnail → **Why this
   frame?** to see sharpness, tilt angle, reprojection error, and composite
   score.

6. **Results** — RMS reprojection error displayed in colour:
   - Green  < 0.5 px — excellent
   - Yellow  0.5–1.0 px — acceptable for most applications
   - Red  > 1.0 px — record again with better coverage / lighting

7. **Save YAML** — browse to the output path (default
   `config/fisheye_calibration.yaml`).  Choose a **T_gantry_camera** source:
   - *Identity* — saves a 4 × 4 identity matrix; measure and edit later.
   - *Load from YAML…* — copies `T_gantry_camera` from an existing file.
   - *Edit 4×4 manually…* — spin-box grid with R^T·R ≈ I validation.

   After writing, the tool calls `load_fisheye_calibration()` immediately
   to verify the round-trip, then shows a success dialog.

### Frame selection scoring (for tuning)

Constants at the top of `calibrate_fisheye.py` are exposed for easy tuning:

```python
MIN_SHARPNESS       = 50.0   # Laplacian-variance floor; lower in dim environments
COVERAGE_GRID       = (4, 4) # Increase to (5, 5) for higher spatial resolution
TARGET_PER_CELL     = 2      # Increase to 3 for more robust calibration
MIN_CELLS_COVERED   = 12     # Minimum before showing strong re-record warning
SCORE_WEIGHT_SHARPNESS = 0.50
SCORE_WEIGHT_TILT      = 0.30  # Raise in underwater (diffuse light lowers sharpness)
SCORE_WEIGHT_REPROJ    = 0.20
```

### Smoke test (mock mode)

```bash
python -m src.calibrate_fisheye --mock-camera
```

1. Click **Connect** → preview shows an animated 9×6 chessboard,
   "Detected ✓" overlay appears within ≈ 2 frames.
2. Click **● Record** → watch the 4 × 4 grid light up progressively as the
   mock board sweeps across the image.
3. After ~15 s click **■ Stop & Calibrate**.
4. The progress dialog should show "16 / 16 cells covered, 32 frames picked".
5. Click **Proceed** → calibration completes; RMS < 0.5 px in mock.
6. Click **Save YAML** → verified round-trip.
7. Drag the splitter handle (8 px wide, turns blue on hover) — both panes
   should resize smoothly; position is saved across restarts.

### Generating a static mock pattern image (optional)

```bash
python tools/make_mock_pattern.py          # writes assets/calib_pattern_mock.png
python tools/make_mock_pattern.py --cols 9 --rows 6 --square 60
```

`calibrate_fisheye.py` does **not** require this file at runtime.

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| R | Record / Stop & Calibrate toggle |
| Ctrl+S | Save YAML |
| Esc | Quit |

### Integrating the calibration into the Experiment pipeline

1. Run the calibration tool; save to `config/fisheye_calibration.yaml`.
2. In the Gantry Control Panel → Connection bar → **Calib:** field, browse to
   that file.
3. Start the Experiment from the **Experiment** tab — the fisheye pipeline
   will load `K`, `D`, and `T_gantry_camera` from the YAML automatically.

---

## Fisheye calibration YAML

```yaml
# fisheye_calib.yaml
image_size: [1920, 1080]
K:
  - [928.5, 0.0,   960.1]
  - [0.0,   928.7, 540.4]
  - [0.0,   0.0,   1.0]
D: [-0.012, -0.001, 0.0008, -0.0002]   # cv2.fisheye distortion (k1,k2,k3,k4)
T_gantry_camera:                        # camera-frame point → gantry-frame point
  - [1.0,  0.0,  0.0,  0.000]           # translation in METERS
  - [0.0,  1.0,  0.0,  0.000]
  - [0.0,  0.0,  1.0,  0.150]           # e.g. camera 150 mm above the tool point
  - [0.0,  0.0,  0.0,  1.000]
```

Calibrate with OpenCV's fisheye routines (`cv2.fisheye.calibrate`). The
runtime uses `cv2.fisheye.estimateNewCameraMatrixForUndistortRectify` +
`initUndistortRectifyMap` + `remap`, then runs AprilTag detection + solvePnP
on the rectified pinhole image.

---

## Waypoints CSV

```csv
x_mm,y_mm,z_mm,speed_mm_s,dwell_s
100.0,50.0,10.0,20.0,0.5
150.0,50.0,10.0,30.0,0.0
150.0,80.0,10.0,15.0,1.0
```

Columns `speed_mm_s` and `dwell_s` are optional (fall back to `--speed-mm-s`
and 0 respectively). The validator refuses any waypoint outside the device
soft limits (or your `--soft-limit-min-mm`/`--soft-limit-max-mm` overrides).

---

## Example commands

### Gantry alone — dry run (no motion)

```bash
python src/gantry_runner.py \
    --dry-run \
    --x-mm 0 --y-mm 0 --z-mm 0 \
    --gantry-ip 192.168.0.30
```

Prints current pos, version, soft limits; writes
`data/YYYYMMDD/<ts>_gantry_run/{gantry_telemetry.csv (header only),
waypoints.csv, run_metadata.json}`.

### Gantry alone — single move with telemetry

```bash
python src/gantry_runner.py \
    --x-mm 100 --y-mm 50 --z-mm 10 \
    --speed-mm-s 20 --acc-mm-s2 50 --dec-mm-s2 50 \
    --log-hz 100
```

### Gantry alone — multi-waypoint

```bash
python src/gantry_runner.py \
    --waypoints-csv waypoints.csv \
    --mode line --log-hz 200
```

### Full pipeline — passive (camera + tags, no motion)

```bash
python src/fisheye_gantry_tagslam.py \
    --camera-device 0 \
    --camera-resolution 1920 1080 \
    --fisheye-calib config/fisheye_calib.yaml \
    --tag-size 0.170 --anchor-tag-id 1 \
    --no-gantry --record-trajectory
```

### Full pipeline — camera + gantry motion

```bash
python src/fisheye_gantry_tagslam.py \
    --camera-device 0 \
    --camera-resolution 1920 1080 --camera-fps 30 \
    --fisheye-calib config/fisheye_calib.yaml \
    --tag-size 0.170 --anchor-tag-id 1 \
    --waypoints-csv waypoints.csv \
    --speed-mm-s 20 --acc-mm-s2 50 --dec-mm-s2 50 \
    --log-hz 100 \
    --record-trajectory
```

Output folder layout:

```
data/YYYYMMDD/YYYYMMDD_HHMMSS_fisheye_gantry/
  gantry_telemetry.csv      # 100 Hz, 23 columns, monotonic + unix timestamps
  waypoints.csv             # planned waypoints (reference copy)
  run_metadata.json         # CLI args, K/D/T_gantry_camera, soft limits, t0/t1
  camera_trajectory.csv     # AprilTag-estimated poses + timestamp_unix/timestamp_monotonic
                            # + gantry_x_mm,gantry_y_mm,gantry_z_mm,translation_error_mm
  tag_poses.csv             # final optimized tag positions
  trajectory_interactive.html
  trajectory_plot.png
  frames/                   # only when --record-trajectory is set
```

---

## Tag Survey GUI — one-stop interactive surveying (`python -m src.survey_tags`)

Run **`python -m src.survey_tags`** with **no arguments** to open the interactive
survey GUI (1500×950): connect camera + gantry, jog the gantry around the working
area by hand, and watch the tag map build itself live — then finalize to
`config/tag_map.yaml` without ever leaving the window. (Any argument, e.g.
`--input-dir …`, runs the headless CLI below instead; the CLI imports no PyQt.)

**Window layout** — top connection bar (camera / gantry / calib / anchor), then a
3-pane splitter: **left** live undistorted camera with AprilTag-corner overlay;
**center** a top-down **live tag map** (custom `QPainter`: pool outline, numbered
tag circles, viridis camera-trail, anchor ★, drag-pan / wheel-zoom / double-click
fit / hover tooltips); **right** a gantry mini-panel (speed/accel/decel, hold-to-jog
X±/Y±/Z±, Move Abs, live position + velocity). Below: a quality-metrics panel
(state, frame counts, last backend jump, median residual, worst tag, scrollable
tag-observation table) and a global control bar (Start / Pause / Stop & Finalize /
Save / **EMERGENCY STOP (Esc)** / Reset E-Stop). Splitter sizes and connection
settings persist in `~/.umi_gui_state.json`.

**Two-stage SLAM.** During the survey, a `DetectionSlamWorker` runs **live
incremental iSAM2** (≤15 fps, drop-oldest frame queue) so you get immediate
feedback; on **Stop & Finalize** the exact CLI **batch** optimization
(`survey_tags.optimize`, Levenberg-Marquardt) re-solves all observations at once
for the accurate, low-uncertainty final map — the live map is fast but greedy, the
batch pass is globally consistent.

**Tag-map colors** (by translational uncertainty): green `< 5 mm`, yellow
`5–15 mm`, red `≥ 15 mm`, gray (dashed) = fewer than `MIN_OBSERVATIONS_PER_TAG`
observations → will be dropped. The anchor is a cyan ★.

**Anchor** — `auto` (default) picks the tag nearest the image center on the first
frame with a detection; or type a specific tag id. If the chosen anchor is not
seen within 30 s the survey aborts with a clear message.

**Jump warnings.** Every backend update compares the camera's actual motion to a
constant-velocity prediction; `residual = ‖p_curr−p_prev‖ − ‖v_prev‖·dt`. Residual
> 20 mm → 3 s yellow banner; > 100 mm → red persistent banner ("consider
re-recording this region more slowly"). These flag the moments a newly added tag
yanked the solution — exactly where the live map is least trustworthy.

**Survey robustness (graph hygiene).**
- **Single-tag frames are rejected** from the graph (need ≥2 simultaneous tags
  for a triangulation constraint); the count is logged
  `[survey] rejected N single-tag frames (no triangulation constraint)`.
- **Tightened detection gates** before the backend: reprojection ≤ 3 px,
  tag area ≥ 200 px², off-nadir ≤ 30°. AprilTag PnP uses
  `cv2.SOLVEPNP_IPPE_SQUARE` (planar 180°-ambiguity-aware).
- **Floor co-planarity prior ON by default** in survey mode: every tag is pinned
  to the z = 0 plane (through the anchor, +z normal) at σ = 5 mm — live
  (`TagSlamBackend`, strict co-planar) and in the batch finalize.
- **Warmup (first 30 s):** a tag may only enter the graph once it has been
  confirmed by ≥ 5 frames that each saw ≥ 3 simultaneous tags (the anchor is
  exempt). This keeps bad early triangulation from being baked in.
- **Suspect tags:** a newly-added tag whose addition jumped the camera > 10 mm is
  ringed in **magenta** with a `!`. **Right-click** any tag to exclude/include it
  — excluded tags (red ✕) stop receiving constraints live and are dropped from the
  final batch map.

**Gantry control has no soft-limit validation** (per rig preference): jog and Move
Abs commands are sent as-is. All SDK calls share one `RLock` with the 10 Hz status
poll, the 100 Hz telemetry logger, and the motion worker.

When finalized, the tag map switches to the **batch-refined** positions with the
**live positions ghosted** semi-transparent, plus a per-tag Δ (mm) table, then
**Save** writes `config/tag_map.yaml` + `config/tag_map_layout.png`.

**Drift control (long surveys).** Pure incremental iSAM2 lets well-observed early
tags numerically "lock in," so as the camera travels >~1 m from the anchor the
global map can squeeze/bend even while per-frame residuals stay low. Two fixes:
- **Periodic batch re-optimization.** Every 200 frames *or* 30 s the worker runs a
  full Levenberg-Marquardt pass over the accumulated graph between frames. If any
  tag moves ≥ 5 mm it **rebuilds iSAM2 from scratch** with the batch-refined values
  as the new linearization point (factors preserved, lock-in reset). A 200 ms blue
  flash on the tag map + a status line `Last batch: MM:SS · max tag shift X mm`
  mark each event; stderr logs `[periodic-batch] …`.
- **Tighter relinearization:** `relinearizeThreshold 0.01 → 0.001` in
  `tagslam_core.TagSlamBackend` (10× more eager).
- **Loop-closure hint:** if the camera goes > 60 s without coming within 30 cm of a
  previously-visited spot, a dismissible yellow hint suggests revisiting an earlier
  area to close a loop. Actual revisits are logged to `loop_closure_events.csv`.

### Diagnostics ("black box") — every survey run is fully reconstructable

When SURVEYING begins, a daemon-thread recorder (`survey_diagnostics.py`, never
blocks the SLAM hot path, flushes every 1 s, ~1–2 % CPU, ~50 MB/hour) writes these
files into `data/YYYYMMDD/<ts>_survey/` alongside the usual CSVs:

| file | what it captures |
|------|------------------|
| `survey_diagnostics.csv` | per-frame backend state (≤15 Hz): tag counts, residuals, jump, camera pose, gantry pose+velocity, warmup flag, `backend_update_ms` |
| `tag_history.csv` | sparse per-tag events: `first_seen`/`promoted`/`observation` (every 50th)/`shifted`/`suspect_set`/`excluded` |
| `batch_events.csv` | every periodic batch: iterations, errors, max/median tag shift, whether iSAM2 rebuilt, anchor drift |
| `user_actions.csv` | every UI click / state transition (start, pause/resume, jog, move, exclude, e-stop, anchor change) with JSON detail + gantry pos |
| `anchor_stability.csv` | anchor pose drift from t0 (should stay ~0) sampled at each batch + at finalize |
| `loop_closure_events.csv` | camera revisits (within 30 cm of a ≥30 s-old position) + co-observed tag count |
| `slam_internals.csv` | iSAM2 health every 60 frames: variable/factor counts, per-update p50/p95 ms |
| `tag_snapshots/snapshot_tNNNs.yaml` | full tag-pose snapshot every 30 s — replay the map's evolution |
| `user_notes.csv` | **timestamped notes you type in the UI** (the "Add Note" field / `Ctrl+N`) with camera+gantry pos |
| `diagnostics_summary.json` | written at finalize: run metadata, summary stats, per-tag summary, and a **drift-onset estimate** (first frame where residual/jump exceeded baseline+3σ for ≥5 frames) |

**Add Note** is the highest-value annotation: type "map starting to squeeze ←" the
instant you see a problem and future debugging can jump straight to that frame.

Inspect a run offline (text summary; plots are TODO):
```bash
python tools/replay_survey.py data/YYYYMMDD/<ts>_survey/
```
If the squeeze/warp reappears, share the **whole** `<ts>_survey/` folder — especially
`user_notes.csv` + `tag_snapshots/` + `survey_diagnostics.csv`.

---

## Tag Survey — build a reusable tag map (`survey_tags.py` CLI)

The headless post-processing path (used by the GUI's Finalize step and available
standalone). A **two-step** workflow that turns a hand-jogged recording into
`config/tag_map.yaml`, a refined tag layout you can later inject as a SLAM prior
(PnP-only localization, no per-run tag-init jump).

**Step 1 — record a survey (panel).** Connect the camera + gantry, click
**● Start Recording** (Recording tab), then **manually jog** the gantry through
the working area from the Control tab (per-axis jog / Move Abs). Recommended jog
pattern: **move slowly with brief dwells**, view **every tag from several angles,
distances, and heights**, and **cover all tag regions** of the board. ~3–5 min is
plenty. Click **■ Stop Recording** — this writes
`data/YYYYMMDD/<ts>_recording/` with `camera_trajectory.csv`, `tag_poses.csv`,
`gantry_telemetry.csv`, and `frames/` (if frame saving was on).

**Step 2 — post-process (CLI, no hardware).** Batch-optimize all observations:

```bash
python -m src.survey_tags --input-dir data/20260527/<ts>_recording \
                          --output config/tag_map.yaml \
                          [--anchor-tag-id 70] \
                          [--min-observations 10] \
                          [--max-iterations 200] \
                          [--use-frames] [--tag-size 0.17]
```

Outputs `config/tag_map.yaml` + `config/tag_map_layout.png` and a stdout report
(per-tag observation counts + uncertainty, dropped tags, optimizer convergence).

**Observation source — two modes:**

- **CSV-only (default, fast — seconds).** Reconstructs `camera_T_tag` from the
  recorded `camera_trajectory.csv` poses + `tag_poses.csv` (the `detected_tags`
  column says which tags were active each frame). These are consistent with the
  recorded poses by construction, so the optimizer confirms consistency and
  computes per-tag uncertainty from observation counts. **Scale is inherited from
  the recording** (no `--tag-size` needed).
- **`--use-frames` (slower — ~10–90 ms/frame; a 3-min/~5000-frame run ≈ 1–7 min).**
  Re-detects AprilTags in the saved frames and runs solvePnP for an *independent*
  `camera_T_tag` per detection → genuine batch refinement (large error reduction).
  ⚠ The recorder saves **undistorted, downscaled JPEG** frames, so re-detection is
  bounded by that resolution, and you **must pass the correct `--tag-size`** (the
  tag edge length in metres; the recording used whatever `tag_size` was active).
  If `frames/` is missing it falls back to CSV-only with a warning.

**Map structure:**

```yaml
anchor_tag_id: 70
tags:
  70:
    position_m: [0.0, 0.0, 0.0]
    quaternion_wxyz: [1.0, 0.0, 0.0, 0.0]
    n_observations: 343
    uncertainty_mm: 0.0          # anchor is pinned at the origin
  71:
    position_m: [0.000034, 0.213834, -0.0026]
    quaternion_wxyz: [1.0, 0.002, 0.003, 0.0126]
    n_observations: 258
    uncertainty_mm: 6.2          # sqrt(trace(Σ_translation)) · 1000
metadata: { source, used_frames_redetection, n_frames_processed, n_tags_qualified,
            n_tags_dropped, min_observations, optimizer_iterations, initial_error,
            final_error, converged, fisheye_calib_path, tool_version, created_at }
```

The anchor is auto-detected as the tag nearest the origin in `tag_poses.csv`
(override with `--anchor-tag-id`) and pinned at identity; all poses are reframed
so the anchor starts at the origin. Noise model matches the live pipeline
(`tag_rot_sigma=0.08 rad`, `tag_trans_sigma=0.04 m`); anchor prior `σ=1e-6`.

**Limitations.** Tags seen **fewer than `--min-observations` times are dropped**
(the report tells you which — drive the gantry through their region next time).
If the anchor tag is not present in the recording the tool exits with code 2.
Output is deterministic (re-runs are byte-identical except `created_at`).

### Using the tag map — PnP-only localization (locked tags)

Once you have `config/tag_map.yaml`, run SLAM in **PnP-only mode**: every tag is
pinned to its surveyed pose (hard prior, σ=1e-6, treated as ground truth) and the
live per-tag bootstrap is skipped, so the camera localizes by PnP against known
tags **from frame one** — no tag-initialization jump.

- **Panel:** Experiment tab → **Use tag map** checkbox + file field
  (`config/tag_map.yaml`). Active only when the toggle is **on AND the file
  exists**; otherwise SLAM runs the normal bootstrap flow (unchanged behavior).
- **CLI (standalone live tool):**
  ```bash
  python -m src.fisheye_gantry_tagslam --fisheye-calib config/fisheye_calibration.yaml \
                                       --tag-map config/tag_map.yaml  ...
  ```

**Runtime-anchor re-mapping.** The map is stored in the *survey* anchor frame
(e.g. tag 70), but each experiment picks its own runtime anchor (the tag nearest
image center at start). At startup, after the runtime anchor is chosen, every map
pose is transformed into the runtime anchor frame:

```
T_runtime_to_X = T_map_to_runtime_anchor^-1 @ T_map_to_X
```

so the runtime anchor lands on identity and all other tags follow. Startup logs
(stderr) confirm it:

```
[tag-map] Loaded 24 tags from tag map
[tag-map] Survey anchor: tag 70
[tag-map] Runtime anchor: tag 65 (selected from image center at frame 0)
[tag-map] Transformed 24 tag poses into runtime anchor frame
[tag-map] PnP-only mode active — tag positions locked
```

**Failure mode.** If the runtime anchor is **not in the map** (it was dropped for
low observations, or never surveyed), the backend fails loudly:
`Runtime anchor tag {id} is not in the loaded tag map. Either re-run survey to
include this tag, or pre-select a known anchor via CLI.` Fix by re-surveying that
region or forcing a known anchor with `--anchor-tag-id`. In PnP-only mode the
floor co-planarity prior is disabled (the locked priors are the ground truth) and
tags not present in the map are ignored.

---

## Time sync & joining the two CSVs

Both CSVs share the run-start monotonic clock and write `timestamp_monotonic`
on every row. Join post-hoc with pandas using `merge_asof`:

```python
import pandas as pd

gantry = pd.read_csv("data/.../gantry_telemetry.csv").sort_values("timestamp_monotonic")
cam    = pd.read_csv("data/.../camera_trajectory.csv").sort_values("timestamp_monotonic")

joined = pd.merge_asof(
    cam, gantry,
    on="timestamp_monotonic",
    direction="nearest",
    tolerance=0.05,                 # 50 ms; relax for low FPS captures
    suffixes=("_cam", "_gantry"),
)

# Drift-only error magnitude (cancels the constant frame offset):
ref_est = joined[["x_m", "y_m", "z_m"]].iloc[0].to_numpy() * 1000.0
ref_g   = joined[["gantry_x_mm", "gantry_y_mm", "gantry_z_mm"]].iloc[0].to_numpy()
delta_est    = joined[["x_m", "y_m", "z_m"]].to_numpy() * 1000.0 - ref_est
delta_gantry = joined[["gantry_x_mm", "gantry_y_mm", "gantry_z_mm"]].to_numpy() - ref_g
joined["delta_error_mm"] = ((delta_est - delta_gantry) ** 2).sum(axis=1) ** 0.5
```

If you only care about the live-recorded literal error, the
`translation_error_mm` column in `camera_trajectory.csv` is already
populated per frame (no joining needed) — it just includes the constant
SLAM-vs-gantry frame offset described above.

`timestamp_unix` is present on both CSVs as a sanity check / cross-machine
fallback if the monotonic clocks ever diverge across processes (they won't
within one run on one host).

---

## Experiment workflow (GUI end-to-end)

The **Experiment** tab in the Gantry Control Panel (`src/gantry_panel.py`) provides a
single-click workflow that coordinates gantry motion, fisheye+TagSLAM recording, and
post-run comparison plots.

### Quick-start

```bash
# Real hardware
python src/gantry_panel.py

# Mock gantry + mock camera (no hardware needed — end-to-end smoke test)
python src/gantry_panel.py --mock --mock-camera
```

### Camera mode (full pipeline vs gantry-only)

The Experiment tab has a **Camera Mode** selector at the top:

- 🎥 **With fisheye (full pipeline)** — default. Camera + calibration required;
  the run writes the full output set listed below.
- 🚫 **Gantry only (no camera)** — bypasses the fisheye pipeline entirely.
  The Camera and Fisheye Calibration checklist rows show "— (skipped)" and
  do not block Start. The run skips `FisheyeWorkerThread` and tag detection
  while idle. Live experiment stats hide tag-related counters. The fisheye
  preview panel shows "Camera not in use for this experiment". Outputs are
  reduced to `gantry_telemetry.csv`, `waypoints.csv`,
  `gantry_pose_velocity_acceleration.png` (3×3 grid of gantry GT pose/vel/acc),
  and `run_metadata.json` (with `"camera_mode": "gantry_only"`). The choice
  is persisted to `~/.umi_gui_state.json` under `gantry_panel.camera_mode`.

The Start button tooltip reflects what will run:
"Will record gantry telemetry only. Camera disabled." vs
"Will record gantry telemetry + fisheye AprilTag SLAM."

### Checklist → Countdown → Run → Outputs

1. **Pre-flight checklist** — all indicators must turn green (or "— (skipped)"
   in gantry-only mode):
   - Controller connected (connect from the Connection bar first)
   - Home reference set for X, Y, Z (home all axes on the Setup tab)
   - Soft limits loaded (auto-loaded on connect; reload from Setup if needed)
   - Path defined (add waypoints on the Sequences tab, or load a CSV)
   - Fisheye camera reachable (click **Test Camera** on the Experiment tab)
   - Fisheye calibration YAML loaded (browse to your `fisheye_calibration.yaml`)
   - Tag size configured (default 0.170 m)

2. **Path source** — choose *Sequences tab waypoints* (default) or a CSV file
   (`x_mm,y_mm,z_mm,speed_mm_s,dwell_s`).  From the Sequences tab you can click
   **→ Experiment** to copy waypoints and jump to the Experiment tab.

3. **Experiment parameters**:
   - *Pre-motion countdown* (default 2 s): countdown before motion starts.
     The fisheye+TagSLAM detector starts during the countdown when
     *Tag detection during countdown* is checked (default ON), giving the SLAM
     backend a few frames to bootstrap before motion begins.
   - *Post-motion settle time* (default 2 s): keep recording after the last
     waypoint to capture overshoot/oscillation.
   - *Output folder name*: leave blank for an auto-generated timestamp.

4. Click **▶ Start Experiment**.  The state label cycles:
   `COUNTDOWN T−2.0s` → `MOTION (waypoint 3/12)` → `SETTLE` → `POSTPROCESS` → `DONE`

5. Click **■ Stop Experiment** (or hit **⚠ EMERGENCY STOP ALL**) at any time to
   abort cleanly.  Partial data is post-processed, `run_metadata.json` gains
   `"aborted": true`.

### Output files in `data/YYYYMMDD/<timestamp>_experiment/`

Full-pipeline mode (`camera_mode: fisheye`):

| File | Content |
|------|---------|
| `gantry_telemetry.csv` | 100 Hz gantry pose/vel/acc (unchanged schema) |
| `waypoints.csv` | Snapshot of the waypoints that ran (user-frame mm) |
| `camera_trajectory.csv` | TagSLAM camera poses + `gantry_x/y/z_mm`, `translation_error_mm` |
| `tag_poses.csv` | Optimized tag positions |
| `trajectory_interactive.html` | **★ Primary viewer — two tabs (Trajectory + Velocity)**, one self-contained HTML, no external deps. **Trajectory:** the full interactive **3D viewer** (mouse orbit / wheel zoom / shift-drag pan), embedded as an `<iframe srcdoc>`, with play/pause + time slider **on top** (real-time playback, 1×/2×/4×), reference **X/Y/Z axes** (red/green/blue, labeled), translucent pool floor, AprilTag markers + triads, camera trajectory (viridis, time-coded) **and** gantry GT (plasma, time-coded, rotated to the SLAM frame via `R_gantry_to_slam`), current-time camera/gantry markers + dashed Δ line with `|Δ| mm` label, and layer-toggle buttons (Camera / Gantry / Tags / Pool / Markers). The left card shows the live fisheye/ZED frame. **Velocity:** stacked Vx/Vy/Vz (cm/s), gantry derived (blue) vs camera derived (dashed red), shared time cursor, pan/zoom. |
| `comparison_topdown.png` | Top-down overlay: camera traj (viridis) + gantry GT (orange) — paper figure |
| `comparison_plot.png` | 3×3 grid: pose/vel/acc × X/Y/Z, gantry vs AprilTag — paper figure |
| `run_metadata.json` | CLI args, timing, output paths, alignment note, `camera_mode`, `axis_sign`, `soft_limits` |
| `frames/` | Raw fisheye frames (if recording was active) |

Gantry-only mode (`camera_mode: gantry_only`):

| File | Content |
|------|---------|
| `gantry_telemetry.csv` | 100 Hz gantry pose + SDK/derived velocity + derived accel |
| `waypoints.csv` | Snapshot of the waypoints that ran |
| `trajectory_interactive.html` | **★ Primary viewer** — gantry-only: the Trajectory tab falls back to the 2D top-down canvas (pool + gantry path only; the 3D viewer needs a camera trajectory), Velocity shows the gantry curves only, header carries a *Gantry-only run (no camera)* badge |
| `gantry_pose_velocity_acceleration.png` | 3×3 grid: pose/vel/acc × X/Y/Z, gantry GT only |
| `run_metadata.json` | Timing, output paths, `camera_mode: gantry_only`, `axis_sign`, `soft_limits` |

Files no longer generated: `pose_velocity_acceleration.html` (replaced by the Velocity tab) and `run_dashboard.html` (folded into `trajectory_interactive.html`).

**`gantry_telemetry.csv` velocity columns (schema change)**

- `vx_mm_s_sdk` / `vy_…` / `vz_…` — firmware-reported velocity. May be zero/unreliable in the 664 "axis not enabled / not homed" state; kept for diagnostics only.
- `vx_mm_s_derived` / … and `ax_mm_s2_derived` / … — **position-derived** via a Savitzky-Golay smooth derivative (`SMOOTHING_WINDOW_S=0.25 s`, `SMOOTHING_POLYORDER=2`), filled by a post-pass in `stop()`. Downstream visualization uses the `*_derived` columns.
- **Backward compatibility:** old recordings (single `vx_mm_s` column) still visualize — the dashboard reads them as SDK velocity and shows a *Legacy CSV — SDK velocity only* banner.

**`trajectory_interactive.html` notes**

- The Trajectory tab reuses the **same 3D viewer** the standalone zed2 pipeline produces (`_build_trajectory_viewer_html`), embedded via `<iframe srcdoc>`. The viewer reads camera rows + a parallel `DATA.gantry` array (SLAM-frame, camera-aligned). Camera = viridis, gantry = plasma, both time-coded; play advances the slider in real time (1 s recording = 1 s playback at 1×).
- Camera **and** gantry velocity are both rotated by `R_gantry_to_slam` (velocity is a vector — rotation only, no translation offset) and derived/smoothed with the **same Savitzky-Golay window/order** (`SMOOTHING_WINDOW_S=0.25 s`, `SMOOTHING_POLYORDER=2`) — a fair comparison. Gantry positions get `R` + `gantry_anchor_offset_mm`; header shows `Alignment: gantry_anchor_offset_mm` when the offset is present, else `first-sample-zeroed (approximate)`.
- **Velocity diagnostics (stderr).** On generation the dashboard prints: a one-sample velocity transform trace (`gantry mm/s --R--> slam mm/s --/10--> cm/s`), the per-axis camera-vs-gantry RMS divergence, a *legacy SDK velocity column* note when `*_derived` is missing, a **warning** if divergence exceeds 50 cm/s, and — most useful — a **`R_gantry_to_slam may be TRANSPOSED`** warning when applying `Rᵀ` would align the curves substantially better. The latter catches the classic mistake of storing `R` the wrong way round.
- Self-contained; the only part needing the sibling `frames/` directory is the live fisheye/ZED frame in the viewer's left card. (`srcdoc` resolves relative frame paths against the parent HTML's folder, so frames render in-place and degrade to a placeholder when the file is moved.)

**`R_gantry_to_slam` (optional calibration field)**

3×3 rotation, gantry frame → SLAM (anchor-tag) frame. Fill it when gantry +X shows up as motion along a different SLAM axis. Example — gantry +X is SLAM +Y, gantry +Y is SLAM −X:

```yaml
R_gantry_to_slam:
  - [0,  1, 0]    # +X_gantry -> +Y_slam
  - [-1, 0, 0]    # +Y_gantry -> -X_slam
  - [0,  0, 1]    # +Z_gantry -> +Z_slam
```

Absent → identity. A non-orthonormal / det≠+1 matrix logs a warning and falls back to identity.

> **Verifying R.** Generate a run's `trajectory_interactive.html` and watch stderr. If the Velocity tab's camera (red) does not track the gantry (blue) and you see `R_gantry_to_slam may be TRANSPOSED`, replace `R` with its transpose (swap the off-diagonal signs / transpose the 3×3). The known-good value for this rig is the example above (`[[0,1,0],[-1,0,0],[0,0,1]]`); its transpose `[[0,-1,0],[1,0,0],[0,0,1]]` flips Vx and makes the curves diverge.

### Aligning the two CSVs in pandas

Both CSVs share the column `elapsed_s = timestamp_monotonic − t0` where `t0` is
the monotonic time at the start of the countdown.  To merge them:

```python
import pandas as pd

gantry = pd.read_csv("gantry_telemetry.csv")
camera = pd.read_csv("camera_trajectory.csv")

# Nearest-neighbour join within one camera frame interval
merged = pd.merge_asof(
    camera.sort_values("elapsed_s"),
    gantry.sort_values("elapsed_s"),
    on="elapsed_s",
    direction="nearest",
    tolerance=1.0 / 30,   # one 30-fps frame
    suffixes=("_cam", "_gantry"),
)
```

### Calibrating `gantry_anchor_offset_mm`

By default, the overlay plot aligns the two trajectories at their first sample
("first-sample-zeroed").  For a metrically correct alignment, measure the vector
from the gantry's home origin to the anchor AprilTag and add it to your
calibration YAML:

```yaml
# fisheye_calibration.yaml
K: ...
D: ...
image_size: [1280, 720]
T_gantry_camera:
  - [1,0,0,0]
  - [0,1,0,0]
  - [0,0,1,-0.17]
  - [0,0,0,1]
gantry_anchor_offset_mm: [1250.0, 800.0, 0.0]   # X, Y, Z from gantry home to anchor tag
```

When this key is present, the overlay plot and `run_metadata.json` report
`"alignment": "gantry_anchor_offset_mm"` instead of the fallback warning.

---

## Safety notes

* `Ctrl+C` triggers a SIGINT that calls `controller.stop_axis(mode=2)` on
  every axis, joins the logger, closes the controller, and exits.
* `--dry-run` validates the connection + soft limits + waypoint list, writes
  empty CSV headers, and exits without commanding any motion.
* Soft limits: the runtime reads `controller.get_device_parameters()` first,
  then applies `--soft-limit-{min,max}-mm` overrides. If both min and max
  read 0 on an axis it's treated as unconfigured (no validation on that
  axis); use the CLI overrides to enforce a manual envelope.
* Acceleration column in `gantry_telemetry.csv` is a 5-sample SMA-smoothed
  central finite difference of velocity — the FMC4030 has no acceleration
  readout. Treat it as a smoothed estimate, not a sensor value.
