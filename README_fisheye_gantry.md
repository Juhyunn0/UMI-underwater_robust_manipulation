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

`src/gantry_panel.py` is the day-to-day operator UI. It binds every primitive
in `gantry_runner.py` (move, log, soft limits, homing, sequence) to live
widgets with a dark theme and a worker-per-task threading model.

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
| **Control** | Per-Axis Control (jog + Move Abs + per-axis Home cards) + the combined Move to Target panel. |
| **Sequences** | The waypoint `QTableWidget` plus Add Row / Remove Selected / Load CSV / Save CSV / Run Sequence / Stop Sequence. |
| **Setup** | Software Limits group + the shared Homing group. "Rarely touched after initial setup." |
| **Recording** | Start/Stop Recording, the clickable CSV path label, and the 30 s rolling per-axis position plot. |

The left pane stays visible no matter which tab is active, so you always see live position and the workspace map. The splitter between left/right is draggable; default ratio 38/62, window default `1500 × 900`, minimum `1280 × 800`. The last-active tab is persisted to `~/.umi_gui_state.json` under `gantry_panel.active_tab` and restored on next launch.

The **Workspace Map** renders two stacked 2D plots: top-down `XY` (aspect-locked square) and side `XZ` (shares the same X axis as the top view). Each shows the soft-limit rectangle, a faint 100 mm grid, the current-position dot (filled green) with crosshairs to the axes, the target marker (hollow blue circle) when a Move-to-Target is active, and a 200-sample trail. The map's bounding box auto-fits to the loaded soft limits by default; toggle `Auto-fit to Soft Limits` off to fit it to current position ∪ target ∪ trail with a 50 mm margin. Implementation is `pyqtgraph` when installed; falls back to a hand-painted `QPainter` widget otherwise (it logs once to stderr if the fallback engages). All map updates ride the existing 10 Hz status-poll signal — no extra SDK calls.

### Per-Axis Control (jog + Move Abs + per-axis Home)

Between the Homing group and the combined Move-to-Target panel, the **Per-Axis
Control** section gives you direct one-axis control of X, Y, and Z. Three
side-by-side cards, one shared parameter row at the top:

```
┌─ Per-Axis Control  (mm) ────────────────────────────────────────────────┐
│  Jog/Move Speed [20.00] mm/s   Accel [50.00] mm/s²   Decel [50.00] mm/s²│
│  ┌─ X ──────────┐  ┌─ Y ──────────┐  ┌─ Z ──────────┐                   │
│  │  +123.456 mm │  │  +50.000 mm  │  │   +0.000 mm  │  ← live readout   │
│  │  raw: +14.96 │  │  raw: +20.00 │  │  raw:  +0.00 │                   │
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
  `_stop_jog`, but the mm→units conversion is exact (single-axis = always uses
  that axis's own `SCALE_MM_PER_UNIT` factor).
* **Move Abs spinbox + button** — absolute move on this one axis only. The
  spinbox range is clamped to the loaded soft limits per axis, and the click
  re-validates against them. Runs on its own background `AxisAbsMoveThread`
  (one per axis, never blocks the GUI).
* **Per-axis Home button** — pure UI shortcut into the existing
  `_home_single(axis)` path. The shared homing parameters from the Homing
  group above (Home Speed in units/s, Accel/Decel, Fall Step, Direction) are
  re-used verbatim, so there's exactly one place to tune homing.
* **Shared jog/move parameter row** — `Jog/Move Speed`, `Accel`, `Decel` in
  mm/s and mm/s². Independent from the combined Move-to-Target panel's own
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

### Homing procedure

1. Click **Connect** — soft limits are auto-loaded into the spinboxes and into
   the per-axis progress bars.
2. Set the Homing parameters (in **units** — the SDK's `home_axis` takes raw
   units; the field labels and tooltip remind you):
   - **Home Speed (units/s)**: default 5.0 (≈ 41 mm/s on X, 12.5 mm/s on Y,
     2.5 mm/s on Z given the per-axis mm-per-unit scaling). Clamped to 20.0
     units/s by `HOME_SPEED_LIMIT_UNITS` regardless of UI input.
   - **Home Accel/Decel (units/s²)**: default 20.0.
   - **Fall Step (units)**: default 5.0.
   - **Direction**: Positive limit (default) or Negative limit.
3. Click **Home X** / **Home Y** / **Home Z** for a single axis, or **Home
   All** with the order dropdown (default `Z → X → Y` so the tool lifts before
   any XY motion).
4. Confirmation dialog → confirm. Status bar turns yellow with `Homing …`.
   Every motion / soft-limit / record button is disabled during homing;
   **Emergency Stop stays enabled**.
5. On completion the panel auto-refreshes position and re-reads soft limits
   (homing can change the origin).

### Soft-limit workflow

| Action | What happens |
|---|---|
| **Connect** | Reads `controller.get_device_parameters()`, converts units → mm via `SCALE_MM_PER_UNIT`, populates per-axis Min/Max spinboxes, clamps Move-target spinboxes to that range, and configures each axis card's progress bar. |
| **Apply X / Y / Z** | Worker thread reads current `DeviceParameters` under the controller lock, mutates the one axis's `soft_limit_min[idx]` / `soft_limit_max[idx]` in raw units, writes back, re-reads to confirm. No confirmation dialog — you just typed those numbers. |
| **Apply All** | Confirmation dialog shows a **per-axis diff** (current vs proposed in mm). On confirm, single worker writes all three axes in one read-mutate-write block. |
| **Load from Controller** | Re-reads device state without writing — useful after homing or out-of-band controller changes. |

The read-mutate-write happens entirely inside one `with self._controller_lock:`
block. The status-polling thread uses the same RLock, so the worst that
happens during an apply is one 100 ms status tick being delayed.

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
Home Z toward NEGATIVE limit at 5.00 units/s.
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
