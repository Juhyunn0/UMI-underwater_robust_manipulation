# Diffusion Policy Trajectory Generation — Plan (v0.2, 2026-08-30)

> **English translation.** Source of truth: [`DP_TRAJECTORY_PLAN.ko.md`](DP_TRAJECTORY_PLAN.ko.md) (Korean). Snapshot of 2026-10-12, source blob `8369016d64`; later edits to the original are not reflected here.
>
> Provenance tags are kept as in the original: `[측정 …]` measured (artifact cited) · `[유도 …]` derived · `[예측 …]` predicted, not measured · `[스펙 …]` vendor spec · `실측` measured · `미실측` not measured · `산출물 없음` no artifact found · `예시/유도값` example or derived value.

> **Status: draft with the decisions applied.** D1–D5 + demonstration source + gripper + order of work + UI integration approach
> were decided by the user on 2026-08-30 (§1). Switching to the English version awaits the user's signal.
> **Implementation status (2026-08-30)**: M0 critical-path code **landed** — `umi_handheld/extract_pose.py`
> (A0), `rov_gui/control/plan_stream.py` (B1), shape `replay` (B2, one-shot M0a + streamed
> M0b), gripper execution path (B5, OFF by default), plans.jsonl + meta schema 8 (B4), part of the B3 failure-mode
> tests (test_replay 9/9 + test_plan_stream 19/19), demo_e2e ran to completion with the real solver.
> safety-code-reviewer audit FAIL→all items fixed (dead-man held-drive release, divergence guard, etc. —
> .claude/journal/reviews.md 2026-08-30). **Remaining M0 precondition = capture 1 demonstration in front of the tag array**
> (§7, first item). PolicyWorker (live inference) and Track A1 onward are not started (as of 2026-08-30;
> PolicyWorker landed on 2026-09-02 — next paragraph).
> Research evidence: 2026-08-30 workflow (3 code readers + UMI_underwater repo verification + 2 advisors)
> + 2 explorations (handheld pose-label path / gripper command path).
> **Implementation status (2026-09-02)**: **PolicyWorker (live inference) landed** — shape `policy`,
> `rov_gui/backends/policy.py` (worker) + `perception/dp_policy.py` (2026-09-01 checkpoint
> session, DDIM transcribed) + `perception/policy_obs.py` (depth observation, bit-parity with the training recipe) +
> `control/policy_frames.py` (TCP algebra) + MpcWorker policy mission (meta schema 10; after the 2026-09-07
> switch to the 5-dim action, schema 12).
> Decisions A1–A23 of the 5-lens design review applied as is: obs_t monotonic clock + single intake
> conversion (A1), TCP-space composition (A2), TCP = the ROV's own jaw (A3, **not measured**), leash anchor (A4),
> 0.2 s grid (A5), fix-based proprio (A6), escalation latch + late kept separate (A7), epoch (A8),
> reject during bridge (A9), hold-tail cost mask (A10), blend·yaw-jump gate (A11), dataset
> gripper level (A12), status freshness (A13), demo/e2e parity via a stub session (A14), provenance
> recording (A15), depth grid rule + reject device depth (A16), FS instrument parity (A17),
> pairing (A18), eval `center` (A19), upstream import discipline (A20), safe defaults (A21),
> end_reason message text (A22), intake inside the guard (A23). Offline verification: 5 test files +
> `policy_dryrun.py` (stub and real checkpoint, synthetic scenes). **Unverified on hardware**; the first hardware run is
> §5 Phase 6-1 (overlay only, no tracking), and what the README calls "first full run = pipeline
> smoke test" is Phase 6-2 (low-speed, narrow-box tracking); the grasp precondition is measuring the jaw center (KNOWN_ISSUES
> 2026-09-02). Verification follow-up 2026-09-02: FS measured 7.26 Hz (depth pair 138 ms =
> 2.07× the training stride), pair cap 3.0·obs_dt, proprio rows aligned to the pair spacing, frame-arrival trigger,
> obs_max_age_s 0.6 — README policy section.
> The number-notation rule is the measurement-citation rule in the repo's CLAUDE.md — numbers without a path are `[예측]`/`[스펙]`/`[유도]`.

## 0. One-line summary

Add **a diffusion policy trained on UMI handheld demonstrations** as a new mission in the GUI trajectory
panel: obs = (depth image, proprioception) → generates a future 1.07 s trajectory + gripper at ~2 Hz
(action 16 knots × [Δx, Δy, Δz, Δψ, g] = 80-dim, retrained 2026-09-07; the 2026-09-01 ckpt is
16×10 rot6d) → through PlanFilter/Stitcher, the existing NMPC (`HwDobMpc`) tracks it at 20 Hz,
and the gripper is sent over the existing MANUAL_CONTROL button path. 1st task: **rock approach + grasp**.

Order of work (decided): **① 1 demo → label extraction → hardware replay gate (M0) → ② bulk collection (100–150)
→ ③ training → ④ live policy integration**. ② starts only once M0 succeeds — replay involves no training, so
1 demo is enough, and that one demo verifies label reconstruction, coordinate transforms, feasibility, MPC tracking and
gripper execution all at once (only generalization is not verified — that is the policy's job).

The essence of the architecture is the **UMI-on-Air pattern**: the policy outputs only embodiment-independent geometry
(the trajectory), and the MPC follower handles the vehicle dynamics/constraints. This architecture is itself the reason
a handheld (a device, not a vehicle) demonstration becomes valid. **However, the only thing borrowed is this division of
roles — there is no MPC inside the policy** (a serial separation: pure generative model → post-hoc PlanFilter → NMPC).
UMI-on-Air's embodiment-aware guidance (steering samples with the controller cost during denoising) is not adopted in
the round-1 design — if filter rejections/clips are frequent on hardware, it is a v2 upgrade candidate. This design sits where
the sentence the UMI-U paper wrote in its Limitations gets carried out: "replacing PID with a dynamics-aware controller
would reduce overshoot and target-switch failures". Baselines for a paper write-up:
(a) UMI-U-style velocity-command DP, (b) geometric mission + NMPC (current), (c) proposed: trajectory-DP + NMPC.

## 1. Log of confirmed decisions (2026-08-30, user)

| # | Decision | Content | Deferred / accompanying risks |
|---|---|---|---|
| D1 | Goal channel **excluded** | Start with pure depth+proprio. SAM2 is possible later via offline processing of saved video | Single-rock scenes mitigate target confusion, but it **reignites** on a multi-object extension (UMI-U Table II: goal-agnostic is 0% under background shift) — a v2 item to reconsider |
| D2 | Task = **rock approach+grasp**, 1 model per task | Includes the grasp → gripper output needed (see G) | grasp is a contact task — interaction between tracking error and the g threshold (§6-②) |
| D3 | **Yaw included in training** | Δψ per knot (unwrapped scalar) | the yaw DoF may conflict with tag visibility — demonstrations need a "tag-facing heading" habit (memory: station-tag-dropout-bridge) |
| D4 | Demonstration = **UMI handheld rig** | teleop/gantry excluded | handheld↔ROV domain gap (§6-①); a pose-label pipeline must be newly built (§5-A0) |
| D5 | **w_hat excluded** | Simplifies proprio | adding it later means a new dataset boundary + retraining |
| G | Gripper = **included in the policy action** | 1 channel of scalar g per knot, thresholded at execution (UMI-U style) | the hardware gripper has no feedback at all → open-loop (§5-B5) |
| M0 | **Replay first** | Bulk collection after passing the 1-demo replay gate | — (a decision that reduces risk) |
| UI | **Integrated as Trajectory panel SHAPES entries** | two entries `replay`/`policy`, reusing the follow pattern | — (easy to take out and put back; `trajectory.kind` gives the records boundary for free) |

## 2. UMI-U verified spec (reference)

Facts verified in the UMI_underwater repo (commit fb99783) + the RSS 2026 paper — the basis for our design:

- Policy: Chi et al. diffusion_policy family `DiffusionUnetTimmPolicy` + `TimmObsEncoder`
  (CLIP ViT-B/16). Obs horizon 2, action horizon 16, execute 8. **DDIM inference 16 steps**,
  the denoiser is a 1D conditional UNet.
- Action: body-frame **velocity command** 6-dim + gripper 2-dim threshold (>0.40, 4 s
  auto-neutral). Execution 10 Hz, inference ~1 Hz (asynchronous receding horizon + time-alignment skip).
- Training data: 233 successful episodes from self-supervised underwater collection (policy), 800 on-land demos (affordance).
- In the code, proprio is [compass, pitch, roll]/360 — the paper's "vehicle depth" is inaccurate (the code is authoritative).

That is, the "1 Hz inference + chunk execution" timing structure is the same as UMI-U. What changes is the meaning of the
action (velocity → trajectory+g) and what closes the loop between replans (ZOH → 20 Hz NMPC).

## 3. Runtime architecture

```
C3VideoWorker (existing)                           MpcWorker (existing, 50 ms tick)
  color 640x360@30 ──► nav_mb ──► TagNav (existing PnP)  ▲            │
  depth 640x360@20 ─┐                                    │ set_reference_traj(fn)
  (×0.64 corr 1 time,├─► policy_mb (RgbdMailbox 3rd tap) │            │ g threshold
   hardware.py:504) ─┘        │                          │            ▼
                              ▼                          │   bus.cmd_gripper_drive
                    PolicyWorker (new, TimerWorker, 1 Hz)│   (existing path, §5-B5)
                      obs assembly: depth + proprio      │
                      DDIM 8–16 steps (rovgui-pose env,  │
                      RTX 5090, expected 15–40 ms [예측])│
                              │ raw plan (16×66.7 ms + g)│
                              ▼                          │
                    PlanFilter → PlanStitcher (new, pure numpy)
                      gate/clip/reject + blend — makes the 1 Hz seams
                      invisible to the controller.  ◄── ReplaySource(file) uses the same entry too
```

Key design decisions:

- **Panel integration**: add `replay`·`policy` entries to `SHAPES` (rov_gui/control/geometry.py:50).
  follow is the precedent for a streaming reference (workers.py:1941-2226), so arm/refusal, START hold,
  the 4 mission-release points and the meta `trajectory.kind` record are all reused. Orthogonal to the controller choice.
  **replay and policy share the same install seam** — M0 verifies the integration point in advance.
- **The follower is NMPC (`HwDobMpc`), not MPCC**. The DP output is a time-parameterized trajectory (timing =
  learned speed), whereas MPCC is a spatial path + its own progress, so it discards the timing, and in a 1 Hz stream
  the end tangent-ray extrapolation (path_geometry.py:131-162) dominates half the horizon. For MPCC,
  refuse via the `follow_ok` pattern (fail-closed).
- **Plug-in point**: a `fn(ts) -> (p, yaw, v, r)` sampler over the stitch buffer goes into
  `HwDobMpc.set_reference_traj(fn)` (mpc_bridge.py:305-308) — minimally invasive.
  pid/*_tuned compatibility goes through the `NedPlan` + `set_path_plan_ned` (mpc_bridge.py:404-418) path.
- **Inference inside the GUI process**: in the rovgui-pose env (torch 2.11+cu128, RTX 5090 — confirmed running),
  SAM2+FoundationPose already run. Copy the PoseWorker pattern (TimerWorker + RgbdMailbox tap,
  copy cost 0 when the consumer is off thanks to the want-gate, hardware.py:562-588; torch import in setup();
  shutdown via the join protocol session.py:499-529).
- Clamp **the horizon tail to an end-point v=0 hold** (do not use constant-velocity extrapolation/tangent ray).

## 4. Policy spec

```
Obs (horizon 2 frames, 0.25–0.5 s apart):
  depth   : uint16 mm → ×0.64 correction → inverse-depth normalized to [0,1] (near=large,
            reusing umi_handheld/build_zarr.py:59-69) → resize (e.g. 224²)
            + validity mask channel (invalid=0 ≠ far, prevents confusing the two)
            [domain bridge: for training data, in-air OAK-D depth is reprojected with warp.py onto the C3 underwater
             optics — warp processes both left+depth (warp.py:177-189)]
  proprio : [z, roll, pitch, u,v,w,r (body)] — on the handheld, values derived from the pose track
            (pose=PnP, velocity=finite difference); in deployment, the state assembler output.
            **matching semantics on both sides is the convention** — absolute yaw and absolute xy are excluded (frame invariance).
            to [-1,1] with per-dim dataset stats.

Action = future trajectory + gripper, expressed relative to the latest obs TCP (decided 2026-09-07, `action_repr: pos_yaw_width`):
  16 knots × 66.7 ms (stride 2 @ 30 fps) = 1.07 s  →  (Δx, Δy, Δz, Δψ, g) × 16 = 80-dim
  Δp  = translation of inv(T_tcp_now) @ T_tcp_k — in the TCP frame (= camera optical frame x right/y down/z forward,
        still tilted). Bit-identical to columns 0:3 of the 2026-09-01 ckpt's 10-dim [pos, rot6d, width].
  Δψ  = yaw_of(R_bt · R_rel_k · R_btᵀ) — the body yaw that a level vehicle carrying the C3 mount (R_bt, tilted 43.3° down)
        must turn to reproduce the relative TCP rotation. NED sign (clockwise seen from above = +), per knot
        wrap(-π, π] (not a cumulative unwrap, knot 0 = 0). Uses neither gravity nor the world frame.
        (an optical-axis azimuth-change definition was rejected because hand roll leaks in scaled by tan|tilt| — correlation 0.688 vs 0.986)
  g   = the jaw-width [m] label as is; in deployment, clip to [0,1] via w_open/w_closed, then threshold (§5-B5).
  normalization: all 5 columns range → [-1,1] (DDIM clip_sample). Encoder umi/common/yaw_action.py.
  deployment decode (policy_frames.compose_plan): p_tcp_k = p_a + R_ned_tcp_a Δp_k, yaw_k = anchor yaw + Δψ_k,
  roll/pitch = the anchor's measured values; resample to the 0.2 s grid (A5), then PlanFilter/Stitcher.
  (the old design's 4 s/0.25 s is dropped: the stitcher covers 1 s replan + MPC preview with the hold-tail, A10.)

6-DoF variant (2026-09-26, `action_repr: pos_rpy_width`, selected via config + ckpt, OFF by default — docs/DP_6DOF_PLAN.ko.md):
  16 knots × (Δx, Δy, Δz, Δψ, Δφ, Δθ, g) = 112-dim. Columns 0:4 and 6 are bit-identical to the 5-dim above (5-dim plan = the 7-dim's
  columns [0:4, 6]); (Δφ, Δθ, Δψ) = ZYX Euler angles of the same M = R_bt·R_rel_k·R_btᵀ — ψ = atan2(M10, M00) (same formula as 5-dim),
  θ = asin(−M20), φ = atan2(M21, M22). FRD sign: +Δθ nose-up, +Δφ starboard-down. Relative to the latest obs row (not cumulative),
  all 7 columns range → [-1,1]; encoder umi/common/yaw_action.py encode_pos_rpy / decode_pos_rpy, and the task yaml
  declares shape_meta.action.rpy_convention {order: zyx, frame: body_frd_via_R_bt, columns: [...]}.
  The deployment decode does **body-z composition from the anchor attitude**: R_body_k = R_body_a · rot_zyx(Δφ_k, Δθ_k, Δψ_k), (φ_k, θ_k, ψ_k) = rpy_of(R_body_k)
  — bit-identical to the 5-dim decode at a level anchor; at a tilted anchor it differs by O(rp·Δψ), hence a separate branch.
  Two sub-modes (policy.attitude_track): false = dropped-and-logged (5-dim arithmetic on columns 0:4 + anchor leveling; Δφ/Δθ are
  only recorded as rp_raw / dropped_rp_deg (raw.rp only when tracked); the **mandatory first flight** of a 7-dim net), true = after clipping at rp_max_deg, p_body = p_tcp − R_body_k·t_bt
  (p_tcp exactly as the policy requested, only the hull tilts) + NMPC tracks roll/pitch via PlanMsg.rp, K/M are sent on the MANUAL_CONTROL
  extension axes (s = pitch, t = roll). The label is the wrist and on hardware it is the hull — tilt spends thrust and moves the C3's view of the tag floor
  (tag dropout at 20° nose-up [예측]), so do not choose it for training without a label-distribution artifact (rp_label_stats / dp_policy_offline gate).

Model:
  encoder  : ResNet-18 + GroupNorm + spatial softmax (~11 M) as priority 1 — for depth input,
             CLIP ViT pretraining gives almost no benefit; "similar architecture" is satisfied by inheriting
             the obs composition·global-cond fusion.
  head     : 1D CNN UNet (down_dims 256/512/1024). Reconsider DiT when the data reaches thousands of episodes.
  diffusion: train DDPM 50–100, inference DDIM 8–16 steps — UMI-U runs a heavier stack
             (Depth-Anything+affordance+DP) at 1 Hz on the same RTX 5090 [스펙: paper].
  training : batch 64–128, ~150 epochs, EMA, bf16 (following UMI-U's public config).
```

**Do not use absolute NED waypoints** (dependent on the tag-map origin + SLAM bias gets baked into the labels).
**Velocity integration is second-best** (drift; the plan interface requires position).

## 5. Work tracks

Track A (data/training) and Track B (control/integration) proceed in parallel and **merge for the 1st time at M0**.

### M0 — single-demo replay gate (top-priority milestone)

Order: capture **1** demo with the new protocol → A0 label extraction → hardware replay → on success, bulk collection.

- **M0(a) one-shot replay**: build a time-parameterized sampler from the extracted pose trajectory (+g timing)
  and install it once into `set_reference_traj` (the same path as square) — with minimal code,
  verify label reconstruction, coordinate transforms, feasibility, MPC tracking and gripper execution.
- **M0(b) streamed replay**: cut the same demo into 4 s windows and stream them at 1 Hz to verify the PlanFilter/Stitcher
  path — doubling as Track B's integration test, as a real-data mock planner.
- Pass criteria (proposed): trajectory-tracking RMS at the level of the existing geometric missions, gripper actuates within ±1 s [예측]
  of the intended time, 0 filter rejections.

### Track A — data → training

**A0. Handheld pose-label pipeline** (new `umi_handheld/extract_pose.py` — currently absent)
~40 lines of glue will do. 4 parts already exist:
1. `CameraModel.K(w,h)` (umi_handheld/camera_model.py:104-109) — rectified-left is a pinhole with 0 distortion
   (configs/source_camera_air.yaml:37-46) → no undistort needed, `dist=None`.
2. `TagDetector` + `TagNav.solve(dets, K, None)` (rov_gui/control/tagnav.py:162-223,
   421-426) — numpy+cv2 only, no gtsam needed. Precedent for offline folder replay =
   src/gantry_map_pose.py + src/tests/test_gantry_map_pose.py:213-244
   (replay accuracy of the same code on real recordings p50 9–16 mm, README_fisheye_gantry.md:672-682).
3. **Giving TagNav the C3 extrinsic yields the camera→body transform in one step**:
   `NavConfig.R_t_frd_cam("main")` (rov_gui/control/geometry.py:213-227;
   c3_payload_frames.json cam_center_bl/cam_xyaxes; cam_tilt_deg 43.3 measured —
   record boundary, config/hw_nav.yaml:149-158).
4. zarr extension = copy the `gripper_width.npy` sidecar pattern (build_zarr.py:120-126, 157-161)
   → `data/pose`·`data/action` arrays + tag-map/scale provenance fields in session.json.

**Traps (mandatory rules)**:
- **Tag id collision**: the gripper tags = tag36h11 **id 0/1** (21.37 mm), but tag_map_full.yaml
  has id 0/1 (170 mm) → a silent ~8× scale blow-up. **Use tag_map.yaml (47 tags, no 0/1)
  + explicitly exclude id 0/1 from the pose computation.**
- PnP on the **pre-warp** left frame (the same rule as extract_gripper_width.py:1-6);
  warp only as the obs domain bridge.
- No IMU on the handheld → the single-tag IPPE ambiguity cannot be arbitrated (tagnav.py:537-557) →
  enforce **min_tags≥2**.
- **The existing 25 demos are presumably unusable for trajectory labels** (whether the environment tags were visible
  was not recorded; 0001–0020 also use the tag family DICT_4X4_50) → demoted to obs-only/spare; labels come only from new captures.

**A1. Bulk demonstration collection** (after passing M0; D2 task = rock approach+grasp)
- Target **100–150 episodes** [스펙: DP-family hardware standard 50–200, UMI-U 233]. Diversity over quantity:
  start pose/distance/approach angle/rock placement.
- Protocol: environment tag array always in view (≥2 per frame), record the tag family, measured size and tag map used
  in session.json, mimic the C3 mount attitude (tilt), include the grasp motion (gripper-width labels via the
  existing extractor), minimize the at-rest head/tail.
- Exclusion rules: segments with poor tag detection, at-rest segments (if not cut, it learns hover).

**A2. Dataset build**
- Slicing (actual implementation = reuse upstream `UmiDataset`): for each obs time t,
  `label = pose over [t, t+1.07 s], 66.7 ms stride, inv(T_tcp_t) 상대 변환 → (Δp, Δψ) + width` ("label = pose over [t, t+1.07 s], 66.7 ms stride, inv(T_tcp_t) relative transform → (Δp, Δψ) + width")
  (no action array in the zarr — the loader composes it from pos/rotvec/width every time; the `action_repr` flag selects the encoding).
- Labels are the **reconstructed EXECUTED trajectory** — smoothing spline/SavGol smoothing before resampling
  (otherwise the MPC faithfully tracks the PnP jitter).
- Unify the depth convention: record the training (in-air→warp) / deployment (C3 ×0.64) pipelines and their boundary fields
  in the dataset meta. 0.64 is a scalar stopgap that is accurate only near 1.0–1.1 m (the error depends on distance,
  hardware.py:118-126) — on an underwater recalibration, reset the dataset boundary + retrain.
- **replay overlay unit test (gate before training)**: logged obs → decode the GT label trajectory →
  automatically verify a top-down plot overlaid on the actual future trajectory. In a repo with three coordinate frames,
  even one wrong sign still trains fine and drives into the wall on hardware (NED↔FLU S-mirror, r_ned=−r_flu
  mpc_bridge.py:365, body-at-t0 transform).

**A3. Training + offline evaluation**
- env: rovgui-pose (has torch) or a separate training env — robust has no torch.
- Whether to reuse the upstream diffusion_policy codebase is decided once the A2 schema is confirmed.
- Open-loop trajectory error on held-out episodes + an offline closed-loop rehearsal with the M0(b) infrastructure.

### Track B — control/integration (can proceed without a model; mock planner + M0(b) are the test bed)

**B1. PlanFilter + PlanStitcher** (pure numpy, do not import solvers — the same discipline as path_cost.py)

Filter, in execution order:
1. Schema/finite values/monotonic timestamps.
2. Freshness: arrival latency relative to the obs time > 0.7 s [예측] → shift the time origin or reject.
3. **Anchor gate**: |p_plan(t_now) − r_now| ≤ 0.3 m [예측]; over the whole overlap window
   max|p_new−p_cur| ≤ 0.2 m [예측] (closes the "starts close, diverges fast" hole).
4. Kinematics: |v| ≤ v_max (based on the measured achieved speed — **needs measurement**, not U_MAX), |a|,
   |dψ/dt| after unwrap. Mild violations are slowed down with time dilation t'=α·t (geometry preserved); large ones are rejected.
5. **Workspace box**: now that the geofence has been removed completely (memory: rov-gui-run-folder),
   **this filter is the only position-based protection** — mandatory.
6. Consecutive-rejection escalation: 1 time → keep the current plan; exhausted → end-point DP hold;
   3 consecutive times [예측] → DP hold + operator alert.
7. **Do not accept plans that arrive during bridge (tag dropout)** — output conditioned on a frozen/DR state is a
   hallucination. After recovery, the first plan must pass the anchor gate.
Hysteresis on every threshold (a borderline planner's 1 Hz track↔hold toggling = excitation).

Stitcher — **commit-then-blend**:
```
p_ref(t) = (1-w(t))·p_old(t) + w(t)·p_new(t),  w: cosine ramp, T_blend 0.3–0.5 s
v_ref    = (1-w)·v_old + w·v_new + w_dot·(p_new − p_old)   ← cross term required
```
- **A new plan is anchored to the current reference r(t_now), not to the measured state** — a measured anchor resets the
  tracking error every 1 Hz cycle → invalidates EAOB/integrator + ratchet drift combined with the leash and deadband.
  Exception: while |r−x_meas| is saturated beyond the leash (0.10 m), anchor at the leash edge
  (in the 0818 run, leash saturation = 83.6 % of engaged ticks — path_cost.py header,
  data/20260818/0818_143802/mpc_143938.csv).
  - **2026-09-07 trial switch [decision: operator]**: `policy.anchor: measured` (both hw_mpc/land_dp). Of the 4,048
    hardware plans on 0907, 94 % had the leash saturated (always 5 cm ahead), and the operator decided to accept the reset
    cost above and observe the "actual-position anchor" first. If it does not work after observing, revert to leash.
    Records boundary = meta `policy.anchor_mode`.
- ACT-style temporal ensembling rejected — with diffusion's multimodal outputs it averages across modes
  (the average of dodge-left/dodge-right = a head-on charge). Option: mode-consistency (the candidate closest to the previous plan).
- yaw is blended after unwrap.

**B2. GUI/worker wiring**
- Add `replay`·`policy` to `SHAPES` (geometry.py:50 + MpcConfig verification + panel
  `_shape_changed`/`_emit_scenario` branches + plot branch).
- `policy_traj` Signal on the bus + backend wiring (next to hardware.py:3148-3153 / demo.py:750-755).
- MpcWorker: `on_policy_traj` slot (on_pose pattern — latest-wins, integrated in the tick),
  mission state cleared at all 4 release points, `_policy_refusal()` (one sentence per reason),
  `_tick_policy` (next to _tick_follow inside the tick).
- Stale stream (> ~1.5 s): freeze the current plan + end-point zero-FF → demote to station
  (follow's STALE/`_follow_to_station` pattern, workers.py:2079-2275).

**B3. Mock planner offline/HIL verification** — failure-mode tests (demo backend + test_control style):
1. Alternating mirror-image plans (1 Hz boundary oscillation) → assert an upper bound on the blended ref's 0.5–1 Hz spectrum.
2. Slow ramp plan → commands inside the T200 deadband → check the minimum effective speed floor (quantized to hold at or below it).
3. ±π crossing / a 90° yaw in one step → assert unwrap + r clip.
4. Plan starting outside 0.5 m (reproduces a re-registration jump) → assert reject+hold.
5. Plan latency/jitter, short plan (tail v=0), planner death (no plan for >T s → hold ladder),
   blending during leash saturation.
6. **M0(b)**: integration test streaming a real demo at 1 Hz (see M0 above).

**B4. Logging/records boundary** (planner vs tracker attribution)
- Run folder `plans.jsonl` (1 line per plan): plan_id, checkpoint hash, inference latency, conditioning obs
  time, anchor snapshot (eta/nu/r_now/leash/bridge_tier), **raw plan verbatim**, installed plan
  (clip/α/blend), filter verdict + **margin values**, **g-channel raw value and threshold verdict**.
- Added CSV columns: active plan_id, plan-relative time, blend w, reference source (plan/blend/hold/
  bridge), gripper command state. The "stage-0 the solver actually consumed" is the existing `ref_ned_at` column —
  do not repeat the 0823 lesson (the incident where the CSV follow_err was not the actual reference).
- meta: `schema_version` 7→8 (inline comment), an always-recorded `policy` block (checkpoint hash,
  inference rate, horizon, splice policy, FF cap, gripper threshold), `replay`/`policy` in `trajectory.kind`,
  a new value in `reference_clock.strategy`. **Never pool new runs with existing runs.**

**B5. Gripper execution path** (exploration done — the send path already exists, no new wiring needed)
- Path: `bus.cmd_gripper_drive.emit(±1.0/0.0)` → `set_gripper_drive`
  (hardware.py:1958-1960, **latched level** — it stops only when 0.0 is sent) → `_buttons()` bits
  76/77 (hardware.py:2422-2425; hardware BTN0/15 = 77/76, measured 2026-08-06 __main__.py:250-251)
  → `manual_control_send` (hardware.py:2436-2452). axes and buttons are orthogonal paths, so it is
  **not blocked even during MPC engage** (window.py:1371-1384 `_pilot_gate` covers axes only).
  Injection point = g threshold → drive emit, next to `cmd_pilot.emit` at workers.py:2788.
- g execution convention (proposed): g>0.6 → close hold, g<0.4 → open hold, in between = neutral (0.0) [예측 —
  UMI-U's 0.40 threshold + a hysteresis variant]; like UMI-U, auto-neutral after a maximum hold time (e.g. 4 s).
- **6 gaps (to apply during implementation)**:
  1. policy↔pilot jaw: **no arbitration** (grip_drive is last-writer-wins) → introduce a new arbitration rule
     (policy takes priority during engage + stop the policy gripper when the pilot intervenes, etc.).
  2. **No feedback at all** (gripper_fb=None hard-coded hardware.py:2135, open-loop servo) →
     no way to know "whether it grabbed"; g via an open-loop threshold+hold-duration.
  3. The demo/ROS2 backends **silently drop** `cmd_gripper_drive` (demo.py:765 connects
     only cmd_gripper) → add the drive path to the demo backend before HIL tests.
  4. The hardware gripper servo channel/PWM range is not recorded (servo9=Newton exists only as prose in dataset.py:904)
     [스펙 unconfirmed].
  5. The sim heavy_gripper is position ctrl[8] (0–0.031 m) — a different interface from the hardware momentary hold,
     no mapping, and currently nobody uses ctrl[8].
  6. Safety: `set_enabled(False)`/`estop()` already set grip_drive=0 — additionally, make **freezing the policy gripper
     command on a hold/bridge demotion** explicit in the filter rules.

### Phase 6 — hardware (in stages, after M0)
1. Plan visualization only, at rest over the tags (no tracking — policy output overlay). First hardware run.
2. Low speed, narrow workspace box + operator standing by to intervene immediately (existing double-gate/DISARM convention,
   option to disable the gripper in early runs). = the README's "first full run (pipeline smoke test)".
3. Widen the box/speed in stages → enable grasp. Planner-vs-tracker attribution from plans.jsonl on every run.

## 6. Risks (ranked)

1. **Handheld↔ROV domain gap** — obs: in-air OAK depth (warp correction) vs underwater C3 depth
   (×0.64 stopgap, distance-dependent residual); viewpoint distribution (hand vs mount); dynamics: handheld trajectories
   do not know the ROV authority (actually delivered ~1/11, memory: sim-vs-hw-square-corner) → the filter's speed
   clamp/reprofile + M0 are the 1st-pass verification.
2. **Open-loop gripper in the grasp phase** — no feedback at all + interaction between tracking error and the g threshold
   (the kind where the vehicle is 5 cm short but g flips to close) → verify timing in M0, hysteresis on the
   threshold, an option to disable the gripper in early hardware runs.
3. **1 Hz plan stitching + inference latency** → no measured anchor + blend + anchor/jump gates (§5-B1).
4. **Frame/sign/scale conventions** — 3 coordinate frames + tag id 0/1 collision (~8× scale trap) →
   replay overlay gate + the tag_map.yaml usage rule.
5. **Depth train/deploy mismatch** (where 0.64 is applied·range-dependence·invalid=far confound)
   → §5-A2 single convention + validity mask + meta boundary.

(Runners-up: goal loss — deferred because it is a single rock, reignites with multi-object (D1); hover collapse —
trim at-rest segments; prev_plan copycat — dropout if introduced.)

## 7. Items not settled

- Location/tag-array layout for capturing the new demonstrations (which tags and how many sheets, which map for the measured layout —
  reuse the pool floor mat in air? a separate bench board + a new map?).
- v_max/measured achieved speed (the basis for the filter thresholds — currently only [예측]).
- Check the hardware gripper servo parameters (SERVOn_FUNCTION/PWM range).
- Reuse the upstream diffusion_policy codebase vs our own minimal implementation (decide at A3).
- Underwater depth recalibration schedule (once done: reset the dataset boundary + retrain).
- Whether pid/*_tuned should also consume the policy source (low cost via the NedPlan path, wider verification scope).
- When to switch to the English version (the user's signal).
