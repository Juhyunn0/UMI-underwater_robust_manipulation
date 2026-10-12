# access GUI : `./c3 gui --source hw --allow-command --pose`

> **English translation.** Source of truth: [`command.md`](command.md) (Korean). Snapshot of 2026-10-12, source blob `8c7ee0e8dd`; later edits to the original are not reflected here.
>
> Provenance tags are kept as in the original: `[측정 …]` measured (artifact cited) · `[유도 …]` derived · `[예측 …]` predicted, not measured · `[스펙 …]` vendor spec · `실측` measured · `미실측` not measured · `산출물 없음` no artifact found · `예시/유도값` example or derived value.

./c3 gui --source hw --allow-command --mpc --fstereo --policy --mpc-mode rl_pwm

# Method 1a: only at the moment the policy runs inference, FoundationStereo does not start a new frame
./c3 gui --source hw --allow-command --mpc --fstereo --policy --mpc-mode rl_pwm --policy-fs-schedule yield


# 
./c3 gui --source hw --pool-cams
# Method 1b: during a mission, FoundationStereo computes only the 2 frames the policy uses (the "only when inferring" approach you described)
./c3 gui --source hw --allow-command --mpc --fstereo --policy --mpc-mode rl_pwm --policy-fs-schedule only

# Method 2: lower the FoundationStereo resolution
./c3 gui --source hw --allow-command --mpc --fstereo --policy --mpc-mode rl_pwm --fstereo-scale 0.5

# Method 3 : 2 process
./c3 gui --source hw --allow-command --mpc --fstereo --policy --fstereo-scale 0.5 --fstereo-process




./c3 gui --source hw --allow-command --mpc --fstereo --policycd /home/bdml/Desktop/RL_controller && DISPLAY=:1 /home/bdml/miniforge3/envs/env_isaaclab/bin/python -u scripts/play.py --task ROV-PoseHold-v0 --num_envs 64 --checkpoint logs/rsl_rl/rov_posehold/2026-09-21_16-53-50_traj5_s2/model_1199.p

# → panel: HIGH = Diffusion Policy, pick the checkpoint with the "…" in the ckpt field (hw_mpc.yaml policy.ckpt is

# only the startup seed; the --policy-ckpt flag was removed 2026-09-11), LOW = follower (PID / MPC / …) → START

# diffusion policy check image :

~/miniforge3/envs/robust/bin/python -m rov_gui.tools.plot_policy_map data/20260908/0908_170428_observe --plan-scale 0.25

# depth map html :

./c3 scene-html --run data/20260908/0908_180453_observe --embed --every 1

## Underwater DP-only verification (no low-level controller, flying directly by joystick) — LOW None in the panel

Since 2026-09-11 this run is chosen in the panel from the **same launch command**: HIGH = `Diffusion Policy`,
LOW = `None` (teleop) → START. `--policy-observe` keeps working as a launch alias that pre-selects LOW None
(the command below, unchanged; it is the same as `--mpc-mode none`, and given together with another `--mpc-mode` the launch is

refused). LOW is fixed for the duration of an engagement, so to go back to a controller, pick PID/MPC/… in LOW
after STOP TRAJ / DISENG (refused while a REC CSV is open).

```bash
./c3 gui --source hw --allow-command --mpc --fstereo --policy --policy-observe \
         --policy-weights model --record-depth
# or launch without --policy-observe, then LOW = None in the panel
```

# train diffusion policy :

conda activate umi2
cd /home/bdml/Desktop/umi_underwater_robust_control/external/UMI_aquatic

python train.py
  --config-name=train_diffusion_transformer_umi_depth_workspace
  task=umi_depth_5d
  task.dataset.action_repr=pos_yaw_width
  dataloader.batch_size=32
  val_dataloader.batch_size=32
  training.num_epochs=200
  training.checkpoint_every=5
  checkpoint.topk.k=100
  logging.project=umi_land2water
  exp_name=can_grasp_depth_5d_v0

# for train, data should be at  :

/home/bdml/Desktop/data collection/dataset_depth.zarr.zip

# checkpoints :

/home/bdml/Desktop/umi_underwater_robust_control/data/          # 2026-09-14: single dated root
  20260907/0907_113747_train_umi_depth_5d_can_grasp_depth_5d_v0/
    checkpoints/
      epoch=0000..0195-train_loss=*.ckpt   ← every 5 epochs, 704 MB each (40 of them)
      latest.ckpt
      selected.ckpt  →  epoch=0195-train_loss=0.028.ckpt   ← what the hardware uses (hw_mpc.yaml policy.ckpt's
                                                            startup seed; other epochs via the "…" in the panel's ckpt field)
      selected.json                                        ← why 195 was picked
    logs.json.txt, normalizer.pkl, .hydra/, heldout_*.log, train_curves_*.png
  train_logs/train_5d_20260907.log

**From 2026-09-06 on, add the last two flags.**

`--policy-weights model` loads the non-EMA state dict inside the checkpoint. The default
`ema_model` is **1.8× worse** on held-out action MSE (0.001535 vs 0.000842, pos RMSE
30.7 vs 20.5 mm) [측정 2026-09-06: validation episodes 6/32/48/56, 219 windows, DDIM 8]. The cause is
not the EMA averaging itself but the BatchNorm running statistics being carried over unchanged as the timm
ImageNet values; even swapping just those 108 buffers recovers 92% of the gap. **Do not pool runs that ran
on different weights**, and the chosen value is recorded in the run meta.

`--record-depth` saves the depth the policy actually consumed to `<run>/policy_obs/` as lossless PNGs.
Without it, that run keeps **not a single frame** of depth, and since there is no raw stereo pair either,
it cannot even be recomputed later. To view it side by side with the land training observations under the
same colour rule:

```bash
python -m rov_gui.tools.depth_compare pair --run data/<date>/<time>_observe \
        --episode 12 --out /tmp/pair            # align the grasp moment with --land-frame/--water-frame
python -m rov_gui.tools.depth_compare stats --run data/<date>/<time>_observe
```

## Direct capture with the land camera — `depth_capture_umi` (take the instrument out of the variables)

If the very OAK-D-W (`14442C10716DBCD600`) that recorded the land demos is mounted on the vehicle, you can
capture live depth **exactly through the training pipeline**, without going through the C3. Same sensor, same
EEPROM calibration, same rectifier object, same FoundationStereo settings (iters 16 / scale 1.0), same warp.
The only remaining differences are scene, medium and mount, so this is the cleanest comparison.

```bash
python -m rov_gui.tools.depth_capture_umi --out /tmp/rovcap --frames 30 --medium air
python -m rov_gui.tools.depth_compare pair  --run /tmp/rovcap --episode 12 --out /tmp/pair
python -m rov_gui.tools.depth_compare stats --run /tmp/rovcap --episode 12
```

At startup it checks two things by itself and **refuses** on a mismatch: a K comparison between the live EEPROM
and `calibration.json` (measured 2026-09-06: max|dK| **0.0000 px**), and a cross-check of the fx/fy/cx/cy that
the rectifier produces against the training store's `.zattrs['camera_model_source']`.

Be sure to look at the **near-field row** in `stats`. It is the row centroid of the nearest 8% of pixels, with
0 at the top of the image and 1 at the bottom. Land training was 0.895 (the fingers come in from below), the
2026-09-06 bench capture was 0.338 — **there is no overlapping range**. No amount of colour matching fixes this.

**`--medium` is a label, not a correction.** This camera's calibration is in-air, so in water the depth reads
long by the refractive index (the opposite direction of what the C3 goes through in air). Nothing is scaled
automatically.

The DP runs and its trajectory is drawn in the trajectory panel, but `ctrl.step()` is not called at all and
**not a single byte** goes from the station to the vehicle. Piloting is by **joystick** throughout, so you can
fly around the bottle yourself and see what the network draws from various positions.
`--allow-command` is needed here **so the pilot can fly** (the opposite of land-dry-run).

Steps:

1. LOW combo `None` (already selected if you launched with the alias) → ENGAGE → HIGH combo
   `Diffusion Policy` → START TRAJ. When the chip shows `OBSERVE · DP NOT COMMANDING`, it is open
   loop. Under None, the only HIGH options that START are Diffusion Policy and Replay (Station/Line/…
   are refused with a reason), and under None even a bare HOLD·replay lands in `data/*/*_observe/`.
2. Direct flight by joystick. Moving the sticks does not cut off the mission (in a control run it is read as a
   takeover and disengages). Losing the tags for a moment does not end the run; only that tick is skipped.
3. The DP plan appears on the main plot as a **thick red dashed line** — the thick one is the current one, the
   thin one the one just before it. **Just two** remain, and they are drawn **on top** so the hull does not hide them. The trajectory trail is
   only **3 s** (it used to be 90 s, which buried the plan). A `DP` swatch in the lower-right legend.

The five automatic stops that presupposed a follower (divergence·escalation latch·bridge-too-long·
workspace box·tag-loss disengage) are **turned off** — because the pilot deliberately moving away from the
reference is this experiment. The jaw is forced OFF; the run clock is `policy.observe_max_run_s` (1800 s).

### Watching it again later — `./c3 run-replay`

```bash
./c3 run-replay data/<date>/<time>_observe          # replay in a window
./c3 run-replay <RUN_DIR> --speed 2 --start 40 --3d
./c3 run-replay <RUN_DIR> --shots out/ --n-shots 6            # stills only, no window
```

Rewinds a recorded run for viewing: **floor AprilTag map + robot position + the reference trajectory at every
DP update**. Space = play/stop, arrows = one tick at a time, drag = pan, wheel = zoom. The `follow` checkbox is
on by default so it follows the vehicle, and the starting zoom is 6× because a 1 s plan spans **4 cm**.

What draws it is **the same `TrajectoryView` widget as live** — it only reconstructs `MpcStatus` and
`PolicyPlanViz` from the CSV and feeds them in; it does no coordinate transforms of its own. So if the replay
picture is wrong, **the picture actually seen that day was wrong in the same way**.

Files read (all of them already written by an ordinary run): `mpc_*.csv` (20 Hz vehicle pose),
`hardware.datum_tag_frame` in `mpc_*.meta.json` (it only sits in the map frame if this is present),
`policy_plan.csv` (one line per plan knot, **rejected ones included** — in these runs they are the majority and
more interesting), `nav_*/map.json` (REC NAV; if absent, the tag map the meta points to is used instead).

### When citing numbers from this run

**These are not closed-loop results.** The vehicle went where the pilot took it, and nobody followed the
reference. The artifacts are collected separately in `data/<날짜>/<시각>_observe/` (`data/<date>/<time>_observe/`);
the CSV `observe` column is 1, `uX..uN`/`w*`/`ax_*`/`e_along`/`e_cross` are nan, `solver_status` is -1, and
`plans.jsonl`·`policy_plan.csv` carry `follower: observe`. The accept/reject ratio, too, is the result of
dragging the anchor around by hand, so **never pool with control runs**.

## On-land DP verification (ROV held in hand, DP trajectory on the ORB-SLAM3 map)

```bash
./c3 gui --source hw --slam --mpc --fstereo --policy \
         --land-dry-run --mpc-config config/land_dp.yaml
```

`--allow-command` is **absent**. If it is present, execution is refused (`__main__.check_slam`) — because
that combination sends thruster commands to a vehicle a person is holding. So in this run the path to the
thrusters is not even built (`NullCommandSink`).

Three windows appear: the rov_gui station, `ORB-SLAM3: Map Viewer` (Pangolin — map points +
DP trajectory), `ORB-SLAM3: Current Frame`.

Steps:

1. Keep the can in the C3's view and hold it facing the room. ORB-SLAM3 only starts building a map once a
   single frame has more than 500 ORB keypoints, and if it does not get past that it **says nothing at all** — you only see an
   empty map. Point it at a textured 1–5 m scene and turn the lights on. When the station log prints
   `slam: tracking — first pose after …`, it has started.
2. At the station: ENGAGE → HIGH combo `Diffusion Policy` (LOW is seeded by `land_dp.yaml mode: pid`;
   the checkpoint comes from the ckpt field) → START TRAJ.
3. Watch the Pangolin window. Solid teal line = plans that reached the controller (accept/clip), dashed grey line =
   plans the station rejected, orange = the raw network output before the safety filter. The latest 12 are kept,
   and the older a plan, the fainter it is. The latest plan gets a 6 cm axis triad.
4. When one reach is done, re-arm with STOP TRAJ → START. The training demos had a median of 6.7 s and
   at most 52.9° of rotation relative to the episode start [측정: "data collection"/dataset_depth.zarr
   75 ep / 15,746 frame]. If you leave it on and walk a full lap around the room, `rot_axis_angle_wrt_start`
   leaves the training distribution, and predictions from then on are not the policy's fault.

### When citing numbers from this run

`config/land_dp.yaml` differs from `hw_mpc.yaml` in 6 values, and of those, `obs_max_age_s` and
`reject_escalate` directly move the accept/clip/reject distribution. **Never pool these with pool runs.**
The artifacts (`slam/orbslam.yaml`, `slam/traj_frames.tum`, `plans.jsonl`, `policy_plan.csv`) are collected in
`data/<날짜>/<시각>_landdry/` (`data/<date>/<time>_landdry/`).

The C3 factory calibration is an underwater value, so in air the distortion is non-uniform in the radial
direction. **Figures in metres from this run cannot be cited without a measured scale** (`KNOWN_ISSUES.md`).

### When it doesn't work

| Symptom                      | Cause                                                                          |
| ---------------------------- | ------------------------------------------------------------------------------ |
| Map stays empty              | 500-keypoint gate. Texture, lighting, distance. `--slam-ini-fast 12 --slam-min-fast 5` |
| `NEW MAP (session N)` log    | Tracking was lost, so ORB-SLAM3 made a new map → the origin moved. STOP, then re-arm |
| No trajectory drawn in Pangolin | Before ENGAGE there is no datum, so coordinates cannot be placed. ENGAGE first |
| Only dashed grey lines appear | The network is running but the station is rejecting everything. status/reason in `plans.jsonl` |

### Checking the whole path without a camera

```bash
cd "$HOME/Desktop/data collection/UMI_Underwater/slam"
~/miniforge3/envs/umi2/bin/python oakx_replay.py "$HOME/Desktop/data collection" \
  --episode 0 \
  --driver "'$PWD/build/oakd_live_slam' $HOME/ORB_SLAM3/Vocabulary/ORBvoc.txt rect.yaml /tmp/ep0"
```

Streams a recorded episode into the driver and injects a synthetic spiral as KNOTs. If the spiral is drawn in
the map window, the protocol, overlay and drawing are all alive.
