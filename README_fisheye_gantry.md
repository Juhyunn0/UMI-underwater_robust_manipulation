# fisheye_gantry_tagslam — usage notes

Three scripts live under `src/`:

| script                          | purpose                                                                                       |
|---------------------------------|-----------------------------------------------------------------------------------------------|
| `gantry_runner.py`              | Drive the FMC4030 gantry to XYZ targets in mm; log telemetry CSV at ~100 Hz.                  |
| `fisheye_gantry_tagslam.py`     | Fisheye camera → undistort → AprilTag → GTSAM iSAM2, alongside gantry motion + telemetry.     |
| `zed2_underwater_tagslam.py`    | Original ZED2 TagSLAM (unchanged CLI/output; now imports from the shared `tagslam_core`).     |

Reusable backend: `tagslam_core.py` (detection, refractive PnP, iSAM2, CSV/HTML/plot writers, OpenCV visualizer).

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
