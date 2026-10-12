# DP 6-DoF variant — roll/pitch tracking (`pos_rpy_width` + `engage.attitude_axes`) operating plan (2026-09-26)

> **English translation.** Source of truth: [`DP_6DOF_PLAN.ko.md`](DP_6DOF_PLAN.ko.md) (Korean). Snapshot of 2026-10-12, source blob `14d0d28024`; later edits to the original are not reflected here.
>
> Provenance tags are kept as in the original: `[측정 …]` measured (artifact cited) · `[유도 …]` derived · `[예측 …]` predicted, not measured · `[스펙 …]` vendor spec · `실측` measured · `미실측` not measured · `산출물 없음` no artifact found · `예시/유도값` example or derived value.

> This document is for operators: what changes, how to turn it on, **in what order to fly it the first time**, which gates block it,
> which numbers are [예측] and their grounds, and the three decisions the design handed to the user. Code boundary:
> the "6-DoF variant" paragraph of `rov_gui/README.md`; unresolved list: the `KNOWN_ISSUES.md` 2026-09-26 entry; original design: the session
> scratchpad `design_final.json` (D1–D13). **0 water runs — every attitude figure below is [예측] unless it has an artifact path.**

## 0. One-line summary

The policy emits a 7-dim `[dx, dy, dz, dyaw, droll, dpitch, width]` (bit-identical to the 5-dim in columns 0:4 and 6 + two roll/pitch columns),
the low-level NMPC tracks the roll/pitch reference, and its K/M torques are sent on the **extension axes** of ArduSub MANUAL_CONTROL (s = pitch, t = roll,
`enabled_extensions = 0b11`). **All defaults are OFF**, and when OFF the 4-DoF path (control output, xref, MAVLink frame,
resolved config) is **byte-identical** to before (pinned by tests). Only the record files gain, with schema 15 → 16, trailing columns/keys that are always appended.

## 0.5 2026-10-03 update — **the LOW decides** (`PID_7D` / `MPC_Tuned_7D` / `RL_7D`)

An operator request ("pid_7d, mpc_tuned_7d, RL_7d versions") changed the meaning of the switches in §1 below. **This section takes precedence over the tables and the truth table in §1–§2.**

* K/M go out **only when the LOW is `pid_7d` | `mpc_tuned_7d` | `rl_7d`**. `engage.attitude_axes.enabled` is now a "station capability"
  (the s/t path that the sink reads at launch + the §4 gates): when it is off, `*_7d` is refused at engage; when it is on, the other LOWs (pid, mpc,
  mpc_tuned, dobmpc*, rl, rl_pwm, mpcc) **engage as 4-DoF** (before: plain mpc/mpc_tuned sent K/M, pid/rl/rl_pwm/mpcc were refused).
* Tracking: a 7-dim ckpt flown on `*_7d` is tracked; flown on a 4-DoF LOW it is dropped-and-logged. `policy.attitude_track` is a **veto** (default
  `true`; with `false`, dropped-and-logged + active leveling even on `*_7d`). With a 5-dim ckpt or a shape mission, `*_7d` does an active level hold.
* §2 ④ "LOW `MPC` or `MPC_Tuned`" → the tracking/leveling stages use **LOW `MPC_Tuned_7D` (or `PID_7D`)**. For §3 ④ dropped-and-logged, just fly on a 4-DoF LOW
  (`MPC_Tuned`) (no need to turn the switch off). §3 ③ "plain mpc attitude STEP" → the station still has **no** attitude-step mission;
  the only current stimuli are `stub_rp` (policy stub) on `MPC_Tuned_7D`/`PID_7D` or `replay.track_attitude` (its guards are weaker than the policy's, KNOWN_ISSUES).
* **Before closing the loop, PID_7D needs 1 pulse-and-release (open loop) first**: if the restoring stiffness k exceeds ~2–4 N·m/rad (free-decay period < 3.2 s),
  the phase margin collapses at 0.2 s delay [유도: `rov_gui/tests/fixtures/sim_pid7d_attitude_20261003.txt`]. Procedure and verdict: the
  KNOWN_ISSUES 2026-10-03 entry.
* `RL_7D`: needs an axes6 export trained on roll/pitch references (none exists; RL_controller `scripts/_pipeline_att7_seed.sh`, training deferred).
* Code boundary: the "7d LOW followers" section of `rov_gui/README.md`, methodology log 2026-10-03.
* **`policy.action_repr: auto` (same day, operator request "if the checkpoint is 7d, set up 7d by itself")**: the deployed value (hw_mpc.yaml, land_dp.yaml).
  The contract of the checkpoint held at START (5-dim `pos_yaw_width` | 7-dim `pos_rpy_width`) is pinned to that mission at ARM — in §2 below,
  step 2 ("`policy.action_repr: pos_rpy_width`") is **not needed**. A missing contract or `pose10d` is refused (no default substituted), 7-dim requires the mount keys,
  and an explicit value is pinned for the session as before. So for §3 ④ (dropped-and-logged), pick 7-dim and START with a **4-DoF LOW** (`MPC_Tuned`, `RL_PWM` …).
  Conversely, **under auto + `attitude_track: true` (deployed), 7-dim + `RL_PWM_7D` (pulses, no capability switch needed) tracks on a single
  START** — until a ④ record exists, pick a 4-DoF LOW (before START the panel's mission line says `TRACKED on …` (warning colour) / `DROPPED-AND-LOGGED`,
  and after ARM the chip shows `· 7d rp TRACKED|DROPPED`). Records: meta `trajectory.action_repr` (pinned, pooling key), `policy.action_repr`
  (pinned or null), `policy.config.action_repr` (`auto`).

## 1. What the variant is — three switches and their combinations

| Switch | Location | Default | Meaning |
|---|---|---|---|
| `policy.action_repr` | `config/hw_mpc.yaml` | `auto` (deployed value as of 2026-10-03; code default `pos_yaw_width`) | The checkpoint contract this mission flies. `auto` = the contract of the ckpt held at START is pinned at ARM (§0.5). With an explicit value, it **must equal** the ckpt's `shape_meta.action.action_repr` at ARM and must be in `POLICY_ACTION_REPRS_FLYABLE = (pos_yaw_width, pos_rpy_width)`. `pos_rpy_width` on a 5-dim ckpt, `pos_yaw_width` on a 7-dim ckpt → both refused at arm ("action_repr mismatch"). |
| `policy.attitude_track` | 〃 | `false` | `false` = **dropped-and-logged**: even with a 7-dim ckpt the roll/pitch columns are discarded (leveling with the anchor attitude) and only recorded in `plans.jsonl rp_raw` (top-level, decoded absolute roll/pitch (2,K)) and `dropped_rp_deg` (`raw.rp` only when tracked). `true` = the absolute roll/pitch clipped by compose is carried in `PlanMsg.rp` and the NMPC tracks it. |
| `engage.attitude_axes.enabled` | 〃 | `false` | Whether the NMPC's K/M are sent on the wire. With `false`, allocation discards them as before (`roll = pitch = 0`). With `true`, `wrench_to_axes(..., attitude=True)` → `s/t` sent. Turning on only this without flying a policy gives an "active leveling" run. |

Truth table of the combinations (`workers._policy_refusal` + the engage gate, all stamped into meta):

* `attitude_track: true` ⇒ `action_repr == pos_rpy_width` **AND** `attitude_axes.enabled` **AND** flight mode MANUAL **AND**
  LOW ∈ {`mpc`, `mpc_tuned`} (`dobmpc*` only when `attitude_axes.dobmpc_allowed: true`; `pid`/`rl`/`mpcc` have no K/M path and are refused)
  **AND** mount keys (`yaw_axis_*`) + `rpy_convention` match **AND** the 4 firmware gates (§4; probe JSON parsed + `sign_proven`) **AND** `rp_residual_deg ≤ 10°` [예측].
  If any one is missing, **arm is refused** — there is no silent demotion.
* `attitude_axes.enabled` + STABILIZE ⇒ engage **refused**. In STABILIZE, s/t are lean-angle targets, so the NMPC's torque model is reinterpreted as an angle servo
  (the cascade is not implemented in this cut, §8 trim (b)).
* 7-dim ckpt + `attitude_track: false` = the **mandatory first flight** of every 7-dim network (§3 ④).

## 2. How to select it (operating procedure)

1. Checkpoint: in the ckpt field of the trajectory panel, pick `checkpoints/selected.ckpt` of the training run `data/<날짜>/<stamp>_train_umi_depth_7d_<exp>` ("data/<date>/<stamp>_train_umi_depth_7d_<exp>")
   (sha1 pinned at ARM). The ckpt must declare `shape_meta.action {shape [7], action_repr pos_rpy_width, yaw_axis_cam_tilt_deg 43.3,
   yaw_axis_R_frd_cam, rpy_convention {order: zyx, frame: body_frd_via_R_bt, columns: [dx,dy,dz,dyaw,droll,dpitch,width]}}`.
   To check only the wiring without torch, put `stub7` (7-dim level) or `stub_rp` (pitch 5° ramp [예측]) in the ckpt field.
2. ~~`config/hw_mpc.yaml`: `policy.action_repr: pos_rpy_width`.~~ Not needed from 2026-10-03 on, because the deployed value is `auto` (§0.5). The first flight goes
   on a 4-DoF LOW (dropped-and-logged).
3. Only at the stage that sends K/M: `engage.attitude_axes.enabled: true` + fill in the `probe`/`sign_probe` artifact paths (§4). The engage gate
   **parses the contents**, rather than checking that the path exists: `probe` must have `tool == rov_gui.tools.attitude_axes_probe`, `pass: true`,
   `mavlink_wire_version "2.0"` and a `firmware_version` equal to the vehicle's AUTOPILOT_VERSION, and `sign_probe` must carry `sign_proven: true`.
   If there is no sign probe yet, `engage.attitude_axes.require_sign_probe: false` (default `true`) can allow the run,
   but then **`first_water_caps [0.1, 0.15]` is enforced** regardless of the `cap_roll/cap_pitch` configured values, and meta gets `sign_proven: false` +
   `caps_in_force` stamped — every run whose sign is not proven flies on the first-water caps.
4. `./c3 gui --source hw --allow-command --mpc --fstereo --policy`, HIGH `Diffusion Policy`, LOW `MPC` or `MPC_Tuned`, MANUAL.

## 3. First-flight order — the two kinds that must come before a tracking flight

The order the design requires (the design's needs_user_decision (a) left open only whether ①–③ or ④ comes first, §7):

| Stage | What | Policy | K/M sent | Artifact (if missing, do not proceed to the next stage) |
|---|---|---|---|---|
| ① Bench probe | `python -m rov_gui.tools.attitude_axes_probe` — **DISARMED**, s=+300 1 s → t=+300 1 s @10 Hz, RC_CHANNELS chan1/chan2 compared before/after; preflight first, refused if armed | None | Extension frames only | `data/<YYYYMMDD>/<run>_probe/attitude_axes_probe.json` (pass/fail, firmware_version, wire version) → `engage.attitude_axes.probe`; engage parses the contents and requires a match on `tool`, `pass: true`, `mavlink_wire_version "2.0"` and `firmware_version` |
| ② In-water sign probe | armed, follower disengaged, roll +0.10 1 s → pitch +0.10 1 s, record ATTITUDE + SERVO_OUTPUT_RAW, abort at \|rp\| > 10° or 2 s | None | Caps 0.1/0.15 | `engage.attitude_axes.sign_probe` path + `sign {roll, pitch}` confirmed; the gate passes only if the JSON has `sign_proven: true` (`require_sign_probe: true` by default). **There is no mission kind for it in this cut** (§8 (c)) — build it by hand. If none exists yet, the run is allowed with `require_sign_probe: false`, but with `first_water_caps` enforced + meta `sign_proven: false`, `caps_in_force` |
| ③ plain mpc attitude STEP | Without a policy, roll/pitch reference steps (0 → 10~15°) and ramps (0.35 rad/s) in station — the attitude version of the 0914 yaw step test | None | ON (first-water caps) | Calibrate `roll_nm/pitch_nm` from steady-state M_cmd vs θ·k; overshoot, settling, damping → decide `attitude_q_scale`/`rp_ref_filter` |
| ④ dropped-and-logged 7-dim flight | (from 2026-10-03) keep `action_repr: auto` as is, pick 7-dim in the panel + **4-DoF LOW** (`MPC_Tuned`/`RL_PWM` …); the old settings were `action_repr: pos_rpy_width`, `attitude_track: false`, `attitude_axes.enabled: false` | 7-dim | **None** | `plans.jsonl rp_raw`, `dropped_rp_deg` distributions; tagnav acceptance rate vs measured pitch → rationale for choosing `rp_max_deg` |
| ⑤ Active leveling | `attitude_axes.enabled: true`, no policy or 5-dim | Optional | ON | Tracking quality of `rroll_deg/rpitch_deg = 0`, meta `run.attitude_axes.sat_ticks`, whether there is a deadband limit cycle |
| ⑥ Tracking flight | `attitude_track: true` | 7-dim | ON | The first run is a smoke test (the whole §1 truth table passes) |

④ is independent of the hardware actuation chain, so it may be run before ①–③. **⑥ requires all of ①, ②, ③ and ④**.

Offline, before flying (hardware time 0):
* `dp_policy_offline.py gate --action-repr pos_rpy_width` — percentiles of |droll|/|dpitch| in the demo labels and the attitude-gate pass rate.
  First artifact (0901 land demos, 1721 windows): `data/20260926/0926_175145_offline/gate_pos_rpy_width.log` — |dpitch| p50/p90/p99/p99.9/max
  1.17/5.53/13.07/18.31/27.12°, |droll| 0.70/2.56/5.39/8.17/10.78°, peak |p|,|q| p90 0.453 rad/s (max with both axes pooled, over the 0.35 cap → on the 200 ms grid 2/1721
  chunks go accept→clip), rp_mag rejects 0. I.e., the pitch column of this data is not SLAM jitter (p90 5.5°), and the 20° clip is below p99.9.
* `dp_policy_offline.py replay --val-only --ckpt <7-dim>` — held-out gets additional roll/pitch RMS, zero-predictor and corr rows.
  Regression gate: pos_rmse/yaw_rms/yaw_corr of columns [:, :4] within the noise of the 5-dim baseline (D/A 21.0/21.1 mm, corr .67/.65; `data/20260915/0915_131900_heldout_D_vs_A/`), the rp columns
  below the zero-predictor + corr > 0.5 [예측]. **Select with the `model` weights only** (EMA/BN defect).
* `policy_session_replay.py <수중 런> --ckpt D7=<7-dim> [--attitude-track]` ("policy_session_replay.py <water run> --ckpt D7=<7-dim> [--attitude-track]") — feeds existing underwater observations to the 7-dim and looks at the tilt the policy
  requests (requested − measured roll/pitch RMS). Underwater OOD was already a left bias in yaw (2026-09-08); now there are two more axes.

## 4. Gates — what blocks it

**Firmware/link (engage, D8)**: (1) `master.mavlink20()` — with a v1 dialect the extension arguments raise a TypeError; (2) `AUTOPILOT_VERSION`
`flight_sw_version ≥ engage.attitude_axes.min_firmware 4.1.2` (Sub 4.1.2 22-Feb-2024 "additional joystick axis for roll/pitch";
4.1.0/4.1.1 **silently ignore** s/t [스펙: ArduSub ReleaseNotes Sub 4.1.2, joystick.cpp source not read]); refused if no version was received ("");
(3) with `require_probe: true`, the `probe` artifact is **parsed** and must have `tool == rov_gui.tools.attitude_axes_probe`, `pass: true`,
`mavlink_wire_version "2.0"`, `firmware_version == 기체 AUTOPILOT_VERSION` ("firmware_version == vehicle AUTOPILOT_VERSION") (refused if only the path exists and the contents differ);
(4) with `require_sign_probe: true` (default), the `sign_probe` artifact must carry `sign_proven: true`. With `require_sign_probe: false`, arm is allowed
with the sign unproven, but **`first_water_caps [0.1, 0.15]` is enforced** and meta `run.attitude_axes` gets `sign_proven: false`,
`caps_in_force` stamped — as long as the sign is not proven, the effective caps are always the first-water caps.
Telemetry `firmware_version / mavlink_wire_version / attitude_axes_enabled / attitude_axes_degraded / rc_chan_raw` is stamped into meta
`run.attitude_axes`.

**Cap tiers (D3)**, being distinct mechanisms, survive a single bug:

| Tier | Where | Value | Behaviour |
|---|---|---|---|
| T0 | Label space | normalizer range + DDIM clip | The network cannot output anything outside the demo range |
| T1 | `compose_plan` | `rp_max_deg` 20 [예측] | Per-axis **clip** of the decoded absolute roll/pitch, records `rp_clipped_deg/n`, `p_tcp` unchanged (the policy's request), only the hull attitude and `p_body` re-derived |
| T2 | `PlanFilter` | `rp_reject_deg` 30 / `pq_max_rad_s` 0.35 / `rp_jump_max_deg` 5 [예측] | hard reject / `need` (dilation) term / soft (recorded) |
| T3 | `set_path_plan_ned` | `rp_reject_rad` | ValueError on a schema violation |
| T4 | Wire | `cap_roll 0.2 / cap_pitch 0.3`, first water `first_water_caps [0.1, 0.15]` [예측] — with the sign unproven (`sign_proven: false`) the first-water caps are enforced regardless of the configured values (meta `caps_in_force`) | `clip(u[3]/roll_nm)`, `clip(u[4]/pitch_nm)` |
| — | NMPC | soft \|φ\|,\|θ\| ≤ 1.2 rad | Unchanged |

**Interlocks (D9, based on the measured state)**: (i) `div_max_rp_deg` 15 [예측], 0.5 s debounce → the policy is **demoted to a level DP hold** (K/M keep flowing:
active leveling; why not disengage: losing the leveling torque on a tilted hull is worse); (ii) `abort_deg` 35 [예측] 0.25 s → disengage
(neutral = s=t=0 → passive restoring); (iii) sat-ineffective: if an axis sits at its cap for `sat_ineffective_s` 1 s and the corresponding attitude error **does not shrink**
→ disengage (catches sign flips, ignored s/t and saturation without progress alike; the "monotonic increase" condition was rejected because it cannot catch a constant saturated error); (iv) at ARM
`rp_residual_deg > 10°` → refused; (v) extension-frame TypeError → resend as a 6-argument frame (K/M 0) + `attitude_axes_degraded` → disengage on the next
tick. The pilot has no roll/pitch input — the only means of overriding an attitude command are disengage / E-STOP (MANUAL only, so neutral = torque 0).

**Hold transition (D5)**: instead of the 11 `set_target_ned` callers, `_rp_hold` inside `HwDobMpc` ramps from the last attitude to level at `pq_max`
(a step would mean a Q=80 torque spike). The hold tail keeps the end-point attitude (bounded by T1).

## 5. [예측] numbers and their grounds

| Value | Grounds | Promotion condition |
|---|---|---|
| `roll_nm 13.2 / pitch_nm 7.2` N·m | [유도] heave_n 60 N [예측] = vertical 4×15 N × lever arm y ±0.22 / x ±0.12 m (bluerov_heavy.xml) | M_cmd vs θ·k from the ③ step test |
| `cap_roll 0.2 / cap_pitch 0.3` (first water 0.1/0.15) | cap×gain 2.6/2.2 N·m ≤ U_MAX 8; the first-water caps cannot capsize the vehicle against restoring ≥ 1.35 N·m/rad [유도] | ②·③ |
| `rp_max_deg 20` | (a) heave-trim surge residual 5.81·sin 20° ≈ 2.0 N [유도] ≈ T200 knee, (b) the C3, tilted 43.3° down, keeps the floor tags in view at 20° nose-up [예측], (c) hold moment ≤ pitch cap, (d) far from the T(η) singularity at 1.2 rad; below the label p99.9 of 18.3° (§3 gate artifact) | ④ tagnav acceptance rate vs pitch |
| `rp_reject_deg 30`, `pq_max_rad_s 0.35`, `rp_jump_max_deg 5` | one rung above the clip, 70 % of yaw `r_max 0.5`, the attitude version of `yaw_jump_max_deg 8` | gate/held-out reject rates, pq peak distribution |
| `anchor_leash_rp_deg 3`, `div_max_rp_deg 15`, `abort_deg 35`, `sat_ineffective_s 1.0` | The attitude version of the 2026-09-14 yaw wobble mechanism (re-anchoring onto a 0.5 s stale measurement); sim S4 div_rp firing [예측 1–2 s] → measured 0.45 s (mpc, dobmpc, `bluerov2_mujoco_marinegym/recordings/20260926/attitude_hold_211917/results.csv`, `dropped_*` rows; reference 20° = `rp_max_deg`, threshold 15° = `div_max_rp_deg`; 0.45 s = 10 consecutive samples over the threshold = the station's 0.5 s debounce, the hull does not approach 20°; the dropped rows of the earlier 174607 had threshold = reference size and are degenerate, do not cite) | sim S4/S5 + ⑤ |
| `RP_RANGE_MIN_RAD 0.10` (training-build guard) | If a column's range is smaller than this, near-zero targets | `rp_label_stats.txt` |

The rotational plant is all placeholder (damping 0.07, added mass 0.12, ZG 0.01 vs coBM 0.01625, payload pitch moment ~1.4 N·m [스펙] not modelled)
— that is why ③ comes before policy flights. The first-water pitch cap is 1.08 N·m < ~1.4 N·m, so active leveling may carry a constant error from M saturation
(sat-ineffective has to catch it). pitch deadband knee ≈ 0.096 × 7.2 ≈ 0.7 N·m [유도] ≈ 20° hold moment → risk of an attitude version of the yaw dither,
with sim S5 as the first evidence. Solver `U_MAX[3:5]` 8 N·m > wire cap → plain mpc lags in large transients (the EAOB sees the truth via `note_applied`), recorded in meta
`controller.attitude_ref.u_max_wire_nm`.

## 6. Records boundary (do-not-pool criteria)

* meta `schema_version` 16; at the end of the mission CSV `rroll_deg, rpitch_deg, ax_roll, ax_pitch, rp_track` (nan/0 when OFF/observe; the reference columns
  have the same FLU sign as the measured `roll_deg/pitch_deg`); the `policy_plan.csv` header is always `…,reason,roll_deg,pitch_deg` (nan when there is no rp);
  plans.jsonl: `rp_tracked` always, `rp_raw` (always with 7-dim: the decoded raw attitude (2,K)), `raw.rp` (only when tracked: the carried (2,J)), `anchor.rp_meas / anchor.rp_used / rp_clipped_deg / rp_clipped_n` when present.
* meta `run.attitude_axes {enabled, transport, roll_nm, pitch_nm, cap_*, slew_per_s, sign, sign_proven, caps_in_force,
  mavlink_wire_version, firmware_version, probe, sign_probe, rc_trim_at_engage, enabled_extensions_sent, degraded_at, sat_ticks}`,
  `trajectory.attitude_track`, `policy.action_repr` (pinned), `controller.attitude_ref {tracked, source plan|hold_ramp|level,
  rate_source none|stitcher_rate_T_inv, …}` — `source` is the value at the time meta is written (even a tracking run's closing meta can end with `level`); the pooling key is
  `tracked`; for `rate_source`, the station (`stitcher_rate_T_inv`: the Euler rate sampled by the stitcher, through T⁻¹) and the sim `attitude_meta`
  (`fd_horizon_T_inv`: np.gradient over the sampled horizon) are **different methods, different labels**, so never pool them under one rate_source;
  `controller.allocation.attitude`. Counters: saturated ticks in meta `run.attitude_axes.sat_ticks`; on the policy side, `policy.run`'s
  `reject_rp_mag / clip_rp_rate / div_rp_max_seen_deg`.
* **Do not pool**: runs that differ in `attitude_track`, `attitude_axes.enabled/transport`, `run.attitude_axes.sign_proven/caps_in_force` (runs with different effective caps), `action_repr`, flight mode, `controller.allocation.attitude`
  (the meaning of dobmpc `w_hat[3:5]` changes), or CSV `rp_track`.
* Caution for external readers: counting columns or comparing bytes will break — only this repo's name-based readers are guaranteed.

## 7. Three things that need a user decision (design `needs_user_decision`)

1. **First-flight order**: does (a) the policy-less plain `mpc` attitude STEP (after ①·②) come first, or (b) the dropped-and-logged 7-dim flight
   with no K/M sent? Both precede the tracking flight, but which comes first decides whether the next hardware time goes to the actuation chain or to the network.
   The design recommends (a).
2. **pose10d rot6d ckpt smoke**: before the 7-dim retraining finishes, whether to fly an existing pose10d checkpoint with the same `attitude_track` flag
   as a training-free smoke test of the follower/actuation chain (this means accepting the EMA/BN defect + the identity-normalizer refusal, and needs a change to the arm allow-list).
3. **Axis gains**: whether to fly the first water with `roll_nm/pitch_nm` at the lever-arm derived values (13.2/7.2 [유도]), or, as for the yaw axis, to require a sim/bench calibration artifact
   first.

## 8. Scope trims in this cut (what was not implemented)

(a) `transport: rc_override` (FALLBACK-A) — config validation refuses it with "not implemented"; (b) STABILIZE lean-angle cascade —
with `attitude_axes.enabled`, STABILIZE is refused; (c) armed in-water sign-probe **mission kind** — the arm gate requires only `sign_proven: true` in the `sign_probe` JSON
(with `require_sign_probe: false` it is bypassed by enforcing the first-water caps), so §3 ② is a manual procedure; (d) ~~PID roll/pitch loop (S11)~~ (implemented on 2026-10-03 as `pid_7d`, §0.5), `run_compare` attitude scenario, `plot_policy_map` rp panel,
teleop panel attitude bar; (e) sim verification up to the S2 hold grid (heavy, mpc/dobmpc, NONE) is mandatory; S3–S7 were run with heavy, NONE
(`bluerov2_mujoco_marinegym/recordings/20260926/attitude_hold_174607`, all scenarios — but the steady-state values of step/ramp/rate/dropped are not citable because the window
includes the transient → cite step from `attitude_hold_211813`, ramp/rate from `attitude_hold_211921`, dropped from `attitude_hold_211917`);
the C/CDW hold cells, the heavy_gripper deadband and the PID rp loop were not run.
