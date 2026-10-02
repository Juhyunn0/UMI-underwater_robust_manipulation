#!/usr/bin/env python3
"""geometry.py — frames, extrinsics and the two config files.

Everything that turns "a tag in a camera image" into "the vehicle in a NED
world box" is decided HERE, from YAML, and nowhere else:

* which tag map defines the world (floor survey vs a single wall tag);
* the map->NED remap ``R_ned_map`` (presets below, overridable);
* the camera extrinsic ``T_body_cam`` (seeded from the sim's measured
  ``c3_payload_frames.json``; a re-mount writes its own numbers into YAML).

There used to be a geofence here too — a NED box that START TRAJ checked the
placed path against and that disengaged the controller at runtime. It was
removed on 2026-08-14 at the operator's request; ``geofence_ned`` /
``geofence_frame`` keys left in an old YAML are now IGNORED, not an error, so
a stale config still loads (see :meth:`NavConfig.load`).

Frame cheat-sheet (the repo has been burned twice by implied conventions):

    tag/map frame   +z INTO the tag face (tagslam object points, +y down)
                    -> floor tags print-up have +z DOWN: map is NED-like.
    camera optical  +x right, +y down, +z forward (OpenCV).
    body FLU        +x fwd, +y left, +z up (MuJoCo/sim, c3_payload_frames).
    body FRD / NED  +x fwd, +y right, +z down (Fossen, ArduSub, the MPC).
    FLU <-> FRD     S = diag(1,-1,-1), same S as dobmpc/frames.py.

Wall presets (tag upright on a vertical wall, print facing the pool):
    x_into_wall   NED x points INTO the wall (ROV in front has x < 0; facing
                  the wall is yaw 0)          R rows: x=[0,0,1] y=[1,0,0] z=[0,1,0]
    x_out_of_wall NED x points OUT into the pool (ROV has x > 0; facing the
                  wall is yaw pi)             R rows: x=[0,0,-1] y=[-1,0,0] z=[0,1,0]
Both keep z_ned = +y_tag = down for an upright tag; the stationary check in
the state assembler (tag roll/pitch vs ATTITUDE) is what catches a mounting
that violates that assumption.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..state import (POLICY_ACTION_REPR, POLICY_ACTION_REPRS_FLYABLE,  # noqa: F401
                     POLICY_CKPT_DEFAULT, POLICY_UMI_REPO)

S_FLU_FRD = np.diag([1.0, -1.0, -1.0])     # same involution as dobmpc.frames.S

#: Every mission ``square.shape`` may name. The panel offers exactly these and
#: :meth:`MpcConfig.load` refuses anything else — see the note there.
#: ``replay`` re-flies a recorded handheld demonstration (poses extracted by
#: ``umi_handheld.extract_pose``) through the plan-stream seam — the same
#: install path a live policy uses. ``policy`` (2026-09-02) IS that live
#: policy: a diffusion policy's action chunks (backends/policy.py) streamed
#: through the same PlanFilter/PlanStitcher into set_path_plan_ned. Needs
#: ``--policy`` (the worker) and ``--mpc``; refused otherwise.
SHAPES = ("station", "line", "square", "circle", "follow", "replay", "policy")


#: Provenance of every number in ``MpcConfig.policy`` (CLAUDE.md rule: a
#: number without an artifact is not a measurement). Mirrored verbatim into
#: config/hw_mpc.yaml's ``policy:`` block and written into the run meta
#: beside the values, so a reader of a recorded run sees which knobs were
#: measured and which were guessed. Keys absent here are paths/enums.
_GRIPPER_LEVELS_PROVENANCE = (
    "[측정: /home/bdml/Desktop/data collection/slam/actions_summary.json "
    "reports[*].gripper_command — open_level_m p10–p90 0.055–0.070 (median "
    "0.061), closed_level_m p10–p90 0.034–0.043 (median 0.040); 0.069/0.042 "
    "chosen near the upper tail [예측]]")

POLICY_PROVENANCE = {
    "dataset_fps": "[측정: dataset .zattrs fps 30 (spec v1 §0)]",
    "num_inference_steps": "[스펙: training cfg num_inference_steps 16]",
    "period_s": "[유도: n_action_steps 8 x obs_dt 66.7 ms = 0.533 s (spec v1 D4)]",
    "knot_dt_s": "[예측] (v2 A5: FD-accel gain 61·σ at 0.2 s vs 551·σ at 1/15 s)",
    "obs_max_age_s": "[유도: measured depth chain 243 ms + inference 33 ms + "
                     "tick 25 ms ≈ 0.30–0.46 s at intake [측정: rov_gui/tools/"
                     "fstereo_bench_out/policy_bench_20260902_141224.json, "
                     "rov_gui/tools/dp_policy_out/dryrun_20260902_140513]; plan "
                     "life at install ≥ 0.4 s ≥ blend_s (v2 A7; was 0.4: zero "
                     "margin against the measured chain)]",
    "stale_s": "[예측]",
    "max_run_s": "[결정: operator 2026-09-11 — 120 -> 500 s]",
    "observe_max_run_s": "[예측] (POLICY OBSERVE / LOW level None only; "
                         "nothing is commanded)",
    "anchor_leash_m": "[예측: below the 62 mm jaw opening (gen_gripper_variant"
                      ".py:47 vendor spec), above the ~0.04 m one-period lag at "
                      "0.08 m/s (v2 A4)]",
    "anchor_leash_yaw_deg": "[예측]",
    "yaw_ref_filter": "[예측 2026-09-14: tau 2 s sits above the 1.4-2 s yaw "
                      "ring period and the 0.5 s plan period [측정: station "
                      "yaw-step test data/20260914/0914_190226/mpc_190335, "
                      "mpc_190532 — overshoot 45/18 %, settles in 1-2 cycles]; "
                      "rate 3 deg/s and guard 15 deg are guesses. Cuts the "
                      "reference's copy of the hull's own yaw swing (0.8 s "
                      "delayed, anti-phase in 0914_181425 run 2) and the "
                      "policy's alternating dyaw (0914_174815). RECORD "
                      "BOUNDARY: meta policy.config.yaw_ref_filter, every "
                      "plan record's yaw_ref_filter]",
    "follower_owns_dynamics": "[결정 2026-09-02 operator: the plan doc's "
                              "division of labour taken literally — the policy "
                              "emits embodiment-agnostic geometry, the NMPC "
                              "owns dynamics. Dynamics and continuity gates "
                              "(anchor, overlap/yaw jump, blend, clip band) "
                              "measure and RECORD but no longer reject; "
                              "freshness, the workspace box and div_max_m stay "
                              "hard. A RECORD BOUNDARY — accept rates are not "
                              "comparable across it]",
    "v_max_m_s": "[측정 2026-09-02: raised 0.08 -> 0.15 because 0.08 sat below "
                 "the distribution it gated — policy chunk peaks 0.090-0.152 m/s, "
                 "all time-dilated alpha 1.23-1.91 [data/20260902/0902_215928/"
                 "plans.jsonl]; human demos need p50 "
                 "alpha 1.50, peak speed p50 0.118 m/s [rov_gui/tools/dp_policy_"
                 "out/gate_20260902_clip3.log]. Best speed the hull has ACHIEVED "
                 "is still 0.058 m/s [유도: config/hw_mpc.yaml axis_gain] (v2 A21)]",
    "a_max_m_s2": "[예측]",
    "r_max_rad_s": "[예측]",
    "anchor_max_m": "[예측]",
    "jump_max_m": "[예측]",
    "yaw_jump_max_deg": "[유도: r_max·blend_s·2/π ≈ 0.13 rad (v2 A11)]",
    "div_max_m": "[예측]",
    "blend_s": "[예측]",
    "blend_v_max_m_s": "[예측: 2·v_max (v2 A11)]",
    "blend_a_max_m_s2": "[예측]",
    "hold_tail_taper_s": "[예측] 0 = hard mask",
    "hold_tail_extrap_max_m": "[예측: 0.3 = v_max 0.15 x 2 s; the offline "
                              "closed-loop A/B that chose `extrapolate` ran "
                              "uncapped, so the cap only trims the far horizon "
                              "at fast plans (data/"
                              "20260908/0908_165517/diag/fwd_C_out.txt)]",
    "along_scale": "[측정 2026-09-08 offline closed-loop A/B on the real acados "
                   "solver + the measured deadband law: hold_tail track never "
                   "moves the vehicle at any q_scale (q8 0.017 m/s); extrapolate "
                   "+ along 1.0 reaches 0.055 m/s (64 % of the 0.085 m/s plan) "
                   "with 1.3 cm error and 0.16 sign flips/s; q_scale 4 matches "
                   "it but also x4 on sway/heave (data/20260908/0908_165517/"
                   "diag/fwd_C_out.txt, fwd_B_out.txt)]",
    "workspace_box_ned": "[예측] datum-relative; None -> REFUSAL for shape policy",
    "tcp_body_flu_m": "[유도: cam_t_flu (hw_nav.yaml) + lens->grip [0.196, 0, "
                      "-0.275] = [0.502, 0, -0.17] (v2 A3; anchored to the lens "
                      "2026-09-08). The lens->grip x is [측정 2026-09-08: a jar "
                      "gripped in the jaws and set on floor tag 58's centre — "
                      "data/20260908/0908_180453_observe/policy_obs/"
                      "rgb/000160.jpg (tag PnP, the run's own intrinsics) + "
                      "0908_170428/policy_obs/rgb/000000.jpg (jar height 0.084 (±0.006) "
                      "m) + 0908_175151/policy_obs/rgb/000000.jpg (open jaws); "
                      "range 0.187-0.204]; z = the CAD vertical; y = 0 [예측]. "
                      "NOT the sim JAW_POS 0.4165 any more — that chain is "
                      "contradicted by the image. Re-derive if cam_t_flu "
                      "changes (KNOWN_ISSUES 2026-09-08)]",
    "tcp_offset_cam_m": "None = derived from tcp_body_flu_m through the C3 "
                        "extrinsic (v2 A3)",
    "obs_view_forward_m": "None = off. [측정: peg root 0.254 m (ROV) vs 0.208 m "
                          "(handheld 9/28) along the gripper axis, "
                          "data/20260930/0930_183320_observe/diag/"
                          "rig_geometry.json] -> 0.05 for the peg checkpoint; "
                          "effect on the plans is [예측: .../diag/"
                          "validate_builder_shift.json], not flown",
    "min_obs_coverage": "[유도: 0.985 sits at the 0.06 % tail of the training "
                        "obs validity (per-frame min 0.9777, p1 1.000, 99.94 % "
                        "of frames ≥ 0.985) [측정: rov_gui/tools/dp_policy_out/"
                        "training_validity_20260902.txt]]",
    "clip_ratio_max": "[예측: 3.0 chosen from the gate — GT chunks need p50 "
                      "1.50 / p90 3.44 at these caps on the 0.2 s grid [측정: "
                      "rov_gui/tools/dp_policy_out/gate_20260902_clip3.log]]",
    "z_near_m": "[스펙: dataset depth recipe normalise_depth z_near (spec v1 §0)]",
    "z_far_m": "[스펙: dataset depth recipe normalise_depth z_far (spec v1 §0)]",
    "obs_res": "[스펙: shape_meta camera0_depth 224 (spec v1 §0)]",
    "gripper_close_below": "[유도: = width 0.050 m on the dataset levels (v2 A12)]",
    "gripper_open_above": "[유도: = width 0.060 m on the dataset levels (v2 A12)]",
    "gripper_hold_max_s": "[예측: UMI-U used 4 s]",
    # Both jaw levels carry the SAME sentence: the 2026-09-02 verify read the
    # cited file and the earlier "≈ 0.062–0.069 / 0.040–0.045" was not what it
    # says (open p10–p90 0.055–0.070, closed 0.034–0.043); the chosen
    # 0.069 / 0.042 sit near the upper tail, which is a choice, not a
    # measurement.
    "gripper_width_open_m": _GRIPPER_LEVELS_PROVENANCE,
    "gripper_width_closed_m": _GRIPPER_LEVELS_PROVENANCE,
    "gripper_width_init_m": "[가정: jaw OPEN at process start — logged, not "
                            "sensed (v2 A12)]",
    "gripper_travel_s": "[예측]",
    "gripper_lookahead_s": "[유도: 0.0 = the pre-2026-09-07 sample-at-now; "
                           "0.6–0.8 s replayed offline from "
                           "data/20260907/0907_145206/"
                           "plans.jsonl (mission mpc_145239) — set by the "
                           "operator after the first jaw-on pool run]",
    # 6-DoF VARIANT (2026-09-26, design_final D3/D9/D12/D13). Every number is
    # [예측] until the sim S2-S7 artefacts and the offline gate stats exist.
    "action_repr": "[결정: the checkpoint contract this mission flies; pinned "
                   "per mission at ARM (POLICY_ACTION_REPRS_FLYABLE)]",
    "attitude_track": "[결정: False = dropped-and-logged (the mandatory first "
                      "flight of any 7-dim network); True needs "
                      "engage.attitude_axes.enabled + every arm gate]",
    "rp_max_deg": "[예측: T1 compose clip — heave-trim surge residual "
                  "5.81·sin 20° ≈ 2.0 N ≈ the T200 knee [유도: mpc_bridge.py "
                  "heave_trim_wrench + hw_mpc.yaml vehicle_net_buoyancy_n]; "
                  "C3 43.3° mount keeps floor tags at 20° nose-up [예측]]",
    "rp_reject_deg": "[예측: T2 PlanFilter hard reject + T3 set_path_plan_ned "
                     "ValueError; far below the T(eta) singularity 1.2 rad]",
    "pq_max_rad_s": "[예측: attitude Euler-rate ceiling; dilation repairs it "
                    "like yaw rate; also the HwDobMpc hold-ramp rate]",
    "rp_jump_max_deg": "[예측: soft overlap-jump gate, recorded not rejected]",
    "anchor_leash_rp_deg": "[예측: 0 = measured (the 2026-09-14 yaw-wobble "
                           "mechanism on the attitude channel)]",
    "div_max_rp_deg": "[예측: 0.5 s debounce -> policy dropped to level DP "
                      "hold, K/M keep flowing (D9 i)]",
    "blend_rp_rate_max": "[예측: preview_install rp_rate_peak gate, rad/s]",
    "heave_trim_attitude_rotated": "[예측: sim S6 decides the default (D13)]",
    "rp_ref_filter": "[예측: null = off; the yaw_ref_filter's attitude twin, "
                     "reserved until an attitude step test shows ringing]",
    "attitude_q_scale": "[예측: 1.0 = the generated Q[3:5]; a knob, not a "
                        "measurement]",
}

#: The handheld UMI gripper's TCP offset in ITS camera optical frame, from the
#: dataset config (spec v1 §0). A COMPARISON value for the run meta only —
#: never a live input: the ROV's TCP is its own jaw (v2 A3).
HANDHELD_TCP_OFFSET_CAM_M = (0.0355, 0.1293, 0.3186)

#: down_sample_steps of the trained policy [스펙: ckpt cfg]; obs_dt =
#: POLICY_DOWN_SAMPLE_STEPS / policy.dataset_fps.
POLICY_DOWN_SAMPLE_STEPS = 2
#: The action representation the station flies (state.POLICY_ACTION_REPR,
#: re-exported here beside its sibling): the worker's checkpoint contract must
#: match it to arm, every plan is checked at intake (workers._policy_refusal /
#: _tick_policy_intake) -- a stale policy.ckpt is refused, not decoded.
POLICY_ACTION_REPR = POLICY_ACTION_REPR


def default_policy_block() -> dict:
    """The ``policy:`` defaults (v2 §B), one definition — a hand-built
    ``MpcConfig()`` and the station fly the same numbers."""
    return {
        "ckpt": POLICY_CKPT_DEFAULT,          # ONE source: rov_gui/state.py
        "repo": POLICY_UMI_REPO,
        # Which state dict inside the checkpoint to fly (--policy-weights
        # overrides). "model" = the raw trained weights, "ema_model" = the EMA
        # copy. The EMA copy carries the BatchNorm defect (KNOWN_ISSUES.md: EMA
        # does not average buffers, so its running stats stay at timm ImageNet
        # values) and is measurably worse on BOTH checkpoint families
        # [측정 2026-09-07, 219-window held-out, dp_policy_offline replay
        # --val-only --weights: 5-dim ep195 pos RMSE 17.3 mm (model) vs 31.3 mm
        # (ema_model at the ep10 the EMA-based val picked); 10-dim 2026-09-01
        # 21.0 vs 29.8 mm], so "model" is the default here. Runs on different
        # weights must not be pooled — meta records it.
        "weights": "model",
        "target_model": "configs/target_camera_underwater.yaml",
        "dataset_fps": 30.0,
        "num_inference_steps": 16,
        "eval_transforms": "center",        # center | train (v2 A19)
        "period_s": 0.5,
        "knot_dt_s": 0.2,
        # TEMPORARY (2026-09-12, operator: "오늘만"): pin every knot's z of the
        # composed policy plan to a fixed height ABOVE THE TAG FLOOR (metres,
        # positive = up, i.e. map z = -value), replacing the policy's own dz.
        # null = off (the policy's z flies). A yaw-response test does not want
        # the handheld demos' descend-to-grasp dz pulling the reference into
        # the floor. RECORD BOUNDARY: meta policy.z_hold_above_floor_m and
        # every plan's `z_hold_applied` say so; do not pool with runs where
        # the policy flew its own z. Delete the key (or set null) to revert.
        "z_hold_above_floor_m": None,
        "obs_max_age_s": 0.6,               # == config/hw_mpc.yaml + backends/policy.py fallback
        "stale_s": 1.5,
        # [결정: operator 2026-09-11 — 120 -> 500 s] ("Policy 시간 120s에서
        # 500s으로 늘려줘"). Not a measurement. path_timeout_s (= max_run_s x
        # traj_timeout_factor, the generic wall-clock backstop) grows with it.
        "max_run_s": 500.0,
        # POLICY OBSERVE's own wall clock (LOW level None; --policy-observe
        # is the launch alias). max_run_s is sized for ONE closed-loop
        # attempt; an observe run is the pilot flying to several places and
        # watching what the network draws at each, and ending that on the
        # mission clock just re-arms the whole rig (a new datum, a cleared
        # plot, a new run folder) mid-check.
        "observe_max_run_s": 1800.0,        # [예측]
        "anchor": "leash",                  # leash | reference | measured (A4)
        "anchor_leash_m": 0.05,
        "anchor_leash_yaw_deg": 10.0,
        # YAW REFERENCE FILTER (2026-09-14). The plan's yaw knots are replaced
        # by a reference that starts at the FLOWN reference yaw (never the
        # measured yaw), turns at a low-passed (tau_s) version of the chunk's
        # requested turn rate capped at rate_deg_s, and stays within guard_deg
        # of the measured yaw; the body knots are re-derived from the TCP
        # knots so the TCP still goes where the policy asked
        # (policy_frames.filter_plan_yaw). null = off (the pre-2026-09-14
        # behaviour: yaw knots = anchor yaw + dyaw_k, anchor per `anchor`).
        "yaw_ref_filter": {"rate_deg_s": 3.0, "tau_s": 2.0, "guard_deg": 15.0},
        "follower_owns_dynamics": False,
        "v_max_m_s": 0.15,
        "a_max_m_s2": 0.20,
        "r_max_rad_s": 0.50,
        "clip_ratio_max": 3.0,              # time-dilation band (A5; replay = 1.5)
        # Consecutive rejects before the mission LATCHES (PlanFilter._reject,
        # plan_stream.py:216-221 -> _policy_halt). Exposed because the latch is
        # indistinguishable on screen from "the network stopped predicting":
        # PolicyState.active goes False, the worker stops inferring and nothing
        # is drawn again until STOP TRAJ / START. A bench run where the vehicle
        # cannot move wants a higher number than a run where it can.
        "reject_escalate": 3,
        "anchor_max_m": 0.15,
        "jump_max_m": 0.06,
        "yaw_jump_max_deg": 8.0,
        "div_max_m": 0.25,                  # the policy's OWN divergence limit
        "blend_s": 0.4,
        "blend_v_max_m_s": 0.16,
        "blend_a_max_m_s2": 1.0,
        "hold_tail": "mask",                # mask | track | extrapolate (A10; 2026-09-08)
        "hold_tail_taper_s": 0.0,
        "hold_tail_extrap_max_m": 0.3,      # extrapolate: continue at most this far past the plan
        "workspace_box_ned": [[-1.0, -1.0, -0.5], [1.0, 1.0, 0.5]],
        "tcp_body_flu_m": [0.502, 0.0, -0.17],   # = cam_t_flu + [0.196, 0, -0.275] (2026-09-08)
        "tcp_offset_cam_m": None,
        # TEMPORARY (2026-09-30, operator: "잠시 오프셋으로"): re-render the depth
        # obs from a viewpoint this many metres FORWARD of the C3 along the
        # body x axis (= the gripper axis), so the held peg lands where the
        # handheld demos have it. null = off (the obs is the training recipe,
        # byte for byte). It changes what the policy SEES and nothing about
        # how its action is flown (tcp_body_flu_m is untouched). RECORD
        # BOUNDARY: meta policy.config.obs_view_forward_m and
        # policy.worker.obs.view_shift; do not pool with unshifted runs.
        "obs_view_forward_m": None,
        "min_obs_coverage": 0.985,
        "z_near_m": 0.20,
        "z_far_m": 3.00,
        "obs_res": 224,
        "gripper": False,
        "gripper_close_below": 0.30,
        "gripper_open_above": 0.67,
        "gripper_hold_max_s": 4.0,
        "gripper_width_open_m": 0.069,
        "gripper_width_closed_m": 0.042,
        "gripper_width_init_m": 0.069,
        "gripper_travel_s": 2.0,
        # WHEN on the streamed plan the jaw hysteresis samples the width
        # channel `g`: now + gripper_lookahead_s [s]. 0.0 == today's
        # behaviour (the sample AT now, i.e. plan knot 0). Why it is a knob:
        # on the 2026-09-07 pool run (data/
        # 20260907/0907_145206, mission mpc_145239) the policy put its
        # "close" at the FAR end of every chunk (first knot with g < 0.30
        # was knot 4-5 of 6 in the last 15 plans; knot 0 stayed ~0.95),
        # because the width it observes (the open-loop
        # GripperWidthEstimator) never changed — and a plan is replaced
        # every ~0.45 s, so the sample at now crossed close_below only
        # during plan GAPS (7-59 % of ticks from t~32 s) while each fresh
        # plan's knot 0 (~0.95 > open_above 0.67) would have re-emitted
        # OPEN: a deadlock. A look-ahead sample (now + 0.6 s) was below
        # 0.30 on 70-100 % of ticks from t~32 s [유도: replayed offline
        # from that run's plans.jsonl]. The sample is clamped inside the
        # stitcher's covered span (an endpoint hold, never an
        # extrapolation); <= 3.0 s, the NMPC horizon. Recorded per tick as
        # the CSV `grip_g` column (schema 13).
        "gripper_lookahead_s": 0.0,
        # POSITION-weight multiplier the MPC-family follower applies while a
        # policy plan is installed (HwDobMpc.set_plan_cost_scale; PID ignores it).
        # WHY: on the vehicle the NMPC commands ~1 N for the ~8-10 cm error the
        # leash allows (its plant has no thruster deadband), and the T200s
        # produce nothing below axis ~0.096 (~5.8 N at surge_n 60) — 92-100 % of
        # MPC/mpc_tuned ticks sat below it while PID (6-7 N) cleared it half the
        # time and made way [측정 2026-09-07: data/
        # 20260907/0907_164659/mpc_16{4716,4835,5209}.csv ax_surge/uX vs
        # 0907_145206/mpc_145239.csv]. Stiffness grows ~sqrt(q), so 16 ~ 4x.
        # 1.0 = the untouched diagonal (pre-2026-09-07 behaviour). Restored to
        # 1.0 at STOP / mode change; recorded in the run meta (trajectory.q_scale).
        "q_scale": 1.0,
        # ALONG-track weight scale for a path-frame (mpc_tuned) follower while a
        # policy plan is installed; None = the controller's own split
        # (mpc_tuned.along_scale, 0.25 for geometric path missions). Surge-only:
        # sway / heave / R untouched, unlike q_scale. Restored at STOP / mode
        # change like q_scale. Recorded in meta controller.plan_along_scale_flown.
        "along_scale": None,
        # ---- 6-DoF VARIANT (2026-09-26). All OFF / [예측] by default; see
        # POLICY_PROVENANCE. action_repr is the checkpoint contract this
        # mission flies (pinned at ARM); attitude_track True makes the NMPC
        # follow the policy's roll/pitch (needs engage.attitude_axes).
        "action_repr": POLICY_ACTION_REPR,
        "attitude_track": False,
        "rp_max_deg": 20.0,
        "rp_reject_deg": 30.0,
        "pq_max_rad_s": 0.35,
        "rp_jump_max_deg": 5.0,
        "anchor_leash_rp_deg": 3.0,
        "div_max_rp_deg": 15.0,
        "blend_rp_rate_max": 0.5,
        "heave_trim_attitude_rotated": False,
        "rp_ref_filter": None,
        "attitude_q_scale": 1.0,
    }


def validate_policy_block(pc: dict) -> dict:
    """Normalise + validate a merged ``policy:`` block IN PLACE and return it.

    The imu_dr/replay rule: a limit that is zero, negative or NaN is not a
    limit (``anchor_max_m: .nan`` makes every filter comparison False, i.e.
    the gate silently PASSES), so every numeric knob is coerced and checked
    here, at load, where a mistake costs a minute instead of a pool session.
    Unknown keys are the CALLER's check (it knows the known set).
    """
    for k in ("period_s", "knot_dt_s", "obs_max_age_s", "stale_s",
              "max_run_s", "observe_max_run_s",
              "v_max_m_s", "a_max_m_s2", "r_max_rad_s",
              "anchor_max_m", "jump_max_m", "yaw_jump_max_deg", "div_max_m",
              "blend_s", "blend_v_max_m_s", "blend_a_max_m_s2",
              "gripper_hold_max_s", "gripper_travel_s", "dataset_fps",
              "z_near_m", "z_far_m"):
        v = float(pc[k])
        if not (math.isfinite(v) and v > 0.0):
            raise ValueError(f"policy.{k} must be finite and > 0, "
                             f"got {pc[k]!r}")
        pc[k] = v
    v = float(pc["clip_ratio_max"])
    if not (math.isfinite(v) and v >= 1.0):
        raise ValueError(f"policy.clip_ratio_max must be finite and >= 1 "
                         f"(1 = no dilation allowed), got {pc['clip_ratio_max']!r}")
    pc["clip_ratio_max"] = v
    for k in ("anchor_leash_m", "anchor_leash_yaw_deg", "hold_tail_taper_s"):
        v = float(pc[k])
        if not (math.isfinite(v) and v >= 0.0):
            raise ValueError(f"policy.{k} must be finite and >= 0, "
                             f"got {pc[k]!r}")
        pc[k] = v
    yf = pc.get("yaw_ref_filter")
    if yf is not None:
        if not isinstance(yf, dict):
            raise ValueError("policy.yaw_ref_filter must be null or a mapping "
                             "{rate_deg_s, tau_s, guard_deg}, got "
                             f"{yf!r}")
        want = {"rate_deg_s", "tau_s", "guard_deg"}
        if set(yf) != want:
            raise ValueError(f"policy.yaw_ref_filter keys must be exactly "
                             f"{sorted(want)}, got {sorted(yf)}")
        clean = {}
        for k in sorted(want):
            try:
                v = float(yf[k])
            except (TypeError, ValueError):
                v = float("nan")
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError(f"policy.yaw_ref_filter.{k} must be finite and "
                                 f"> 0, got {yf[k]!r}")
            clean[k] = v
        if not (1.0 <= clean["guard_deg"] <= 180.0):
            # below ~1 deg the filter degenerates into "anchor measured +
            # hard clamp", i.e. the wobble amplifier it replaces
            raise ValueError("policy.yaw_ref_filter.guard_deg must be in "
                             f"[1, 180] deg, got {clean['guard_deg']!r}")
        r_max_deg = math.degrees(float(pc["r_max_rad_s"]))
        if clean["rate_deg_s"] > r_max_deg:
            raise ValueError(f"policy.yaw_ref_filter.rate_deg_s "
                             f"{clean['rate_deg_s']!r} exceeds policy.r_max_rad_s "
                             f"({r_max_deg:.1f} deg/s) — the kinematic gate would "
                             f"dilate every plan")
        pc["yaw_ref_filter"] = clean
    else:
        pc["yaw_ref_filter"] = None
    for k in ("num_inference_steps", "obs_res", "reject_escalate"):
        v = int(pc[k])
        if v < 1:
            raise ValueError(f"policy.{k} must be >= 1, got {pc[k]!r}")
        pc[k] = v
    if not (pc["z_near_m"] < pc["z_far_m"]):
        raise ValueError("policy.z_near_m must be < policy.z_far_m")
    zh = pc.get("z_hold_above_floor_m")
    if zh is not None:
        # A height, so it must be a finite positive number: 0 is the tag
        # plane itself and a negative value is below the floor. The 1.5 m cap
        # is the pool's depth [스펙: hw_nav.yaml pool] — nothing above it is
        # a place the vehicle can be.
        v = float(zh)
        if not (math.isfinite(v) and 0.0 < v <= 1.5):
            raise ValueError(f"policy.z_hold_above_floor_m must be finite, > 0 "
                             f"and <= 1.5 m above the tag floor, or null; "
                             f"got {zh!r}")
        pc["z_hold_above_floor_m"] = v
    ovf = pc.get("obs_view_forward_m")
    if ovf is not None:
        # 0.15 m is the builder's own cap (policy_obs.VIEW_SHIFT_MAX_M): past
        # it a single depth view has too little behind the jaw to re-render.
        # 0 is "off" spelled as a number, and is stored as None so the record
        # never shows a shift that did nothing.
        # FORWARD only: a viewpoint behind the lens shrinks the scene and
        # leaves an empty rim no fill can close.
        v = float(ovf)
        if not (math.isfinite(v) and 0.0 <= v <= 0.15):
            raise ValueError(f"policy.obs_view_forward_m must be finite and in "
                             f"[0, 0.15] m, or null; got {ovf!r}")
        pc["obs_view_forward_m"] = v if v != 0.0 else None
    cov = float(pc["min_obs_coverage"])
    if not (math.isfinite(cov) and 0.0 <= cov <= 1.0):
        raise ValueError(f"policy.min_obs_coverage must be in [0, 1], "
                         f"got {pc['min_obs_coverage']!r}")
    pc["min_obs_coverage"] = cov
    if pc["anchor"] not in ("leash", "reference", "measured"):
        raise ValueError(f"policy.anchor must be leash|reference|measured, "
                         f"got {pc['anchor']!r}")
    if pc["hold_tail"] not in ("mask", "track", "extrapolate"):
        raise ValueError(f"policy.hold_tail must be mask|track|extrapolate, "
                         f"got {pc['hold_tail']!r}")
    v = pc.get("hold_tail_extrap_max_m", 0.3)
    try:
        v = float(v)
    except (TypeError, ValueError):
        v = float("nan")
    if not (math.isfinite(v) and v > 0.0):
        raise ValueError(f"policy.hold_tail_extrap_max_m must be finite and > 0, "
                         f"got {pc.get('hold_tail_extrap_max_m')!r}")
    pc["hold_tail_extrap_max_m"] = v
    al = pc.get("along_scale", None)
    if al is not None:
        try:
            al = float(al)
        except (TypeError, ValueError):
            al = float("nan")
        if not (math.isfinite(al) and 0.05 <= al <= 100.0):
            raise ValueError(f"policy.along_scale must be null or a finite number "
                             f"in [0.05, 100], got {pc.get('along_scale')!r}")
    pc["along_scale"] = al
    if pc["eval_transforms"] not in ("center", "train"):
        raise ValueError(f"policy.eval_transforms must be center|train, "
                         f"got {pc['eval_transforms']!r}")
    box = pc.get("workspace_box_ned")
    if box is not None:
        b = np.asarray(box, float)
        if b.shape != (2, 3) or not np.all(np.isfinite(b)):
            raise ValueError(
                f"policy.workspace_box_ned must be "
                f"[[xmin,ymin,zmin],[xmax,ymax,zmax]] (finite), got {box!r}")
        if not np.all(b[0] < b[1]):
            raise ValueError(
                f"policy.workspace_box_ned min must be < max on every "
                f"axis, got {box!r}")
        pc["workspace_box_ned"] = [[float(v) for v in b[0]],
                                   [float(v) for v in b[1]]]
    for k in ("tcp_body_flu_m", "tcp_offset_cam_m"):
        v = pc.get(k)
        if v is None:
            if k == "tcp_body_flu_m":
                raise ValueError("policy.tcp_body_flu_m must be a 3-vector")
            continue
        a = np.asarray(v, float).ravel()
        if a.size != 3 or not np.all(np.isfinite(a)):
            raise ValueError(f"policy.{k} must be a finite 3-vector, "
                             f"got {v!r}")
        pc[k] = [float(x) for x in a]
    lo = float(pc["gripper_close_below"])
    hi = float(pc["gripper_open_above"])
    if not (0.0 <= lo < hi <= 1.0):
        raise ValueError(
            f"policy gripper thresholds must satisfy 0 <= close_below "
            f"< open_above <= 1, got {lo!r} / {hi!r}")
    pc["gripper_close_below"], pc["gripper_open_above"] = lo, hi
    wo = float(pc["gripper_width_open_m"])
    wc = float(pc["gripper_width_closed_m"])
    wi = float(pc["gripper_width_init_m"])
    if not (math.isfinite(wo) and math.isfinite(wc) and math.isfinite(wi)
            and 0.0 <= wc < wo):
        raise ValueError(
            f"policy gripper widths must satisfy 0 <= closed < open, "
            f"got closed {wc!r} / open {wo!r}")
    pc["gripper_width_open_m"], pc["gripper_width_closed_m"] = wo, wc
    pc["gripper_width_init_m"] = wi
    pc["gripper"] = bool(pc["gripper"])
    # 0 is a valid value here (it IS the pre-2026-09-07 behaviour), so the
    # "> 0" loop above cannot host it; the ceiling is the NMPC horizon.
    try:
        la = float(pc["gripper_lookahead_s"])
    except (TypeError, ValueError):
        la = float("nan")
    if not (math.isfinite(la) and 0.0 <= la <= 3.0):
        raise ValueError(
            f"policy.gripper_lookahead_s must be a finite number in "
            f"[0, 3.0] s (0 = sample the jaw channel at now; 3.0 = the NMPC "
            f"horizon), got {pc['gripper_lookahead_s']!r}")
    pc["gripper_lookahead_s"] = la
    for k in ("ckpt", "repo", "target_model"):
        pc[k] = str(pc[k])
    qs = pc.get("q_scale", 1.0)
    try:
        qs = float(qs)
    except (TypeError, ValueError):
        qs = float("nan")
    if not (0.1 <= qs <= 100.0):
        raise ValueError(f"policy.q_scale must be a finite number in [0.1, 100], got {pc.get('q_scale')!r}")
    pc["q_scale"] = qs
    w = str(pc.get("weights", "model"))
    if w not in ("model", "ema_model"):
        raise ValueError(f"policy.weights must be 'model' or 'ema_model', got {w!r}")
    pc["weights"] = w
    _validate_policy_attitude(pc)
    return pc


def _validate_policy_attitude(pc: dict) -> None:
    """The 6-DoF variant's policy keys (2026-09-26), in place. Missing keys
    take the defaults so a pre-variant hand-built block still resolves."""
    d = default_policy_block()
    ar = str(pc.get("action_repr", d["action_repr"]))
    if ar not in POLICY_ACTION_REPRS_FLYABLE:
        raise ValueError(f"policy.action_repr must be one of "
                         f"{list(POLICY_ACTION_REPRS_FLYABLE)}, got {ar!r}")
    pc["action_repr"] = ar
    pc["attitude_track"] = bool(pc.get("attitude_track", False))
    pc["heave_trim_attitude_rotated"] = bool(
        pc.get("heave_trim_attitude_rotated", False))
    for k, hi in (("rp_max_deg", 60.0), ("rp_reject_deg", 60.0),
                  ("rp_jump_max_deg", 60.0), ("div_max_rp_deg", 90.0)):
        try:
            v = float(pc.get(k, d[k]))
        except (TypeError, ValueError):
            v = float("nan")
        # the T(eta) singularity is at 90 deg and the NMPC's own soft bound
        # is 1.2 rad (69 deg); nothing above it is a reference
        if not (math.isfinite(v) and 0.0 < v <= hi):
            raise ValueError(f"policy.{k} must be finite and in (0, {hi:g}] deg, "
                             f"got {pc.get(k)!r}")
        pc[k] = v
    if pc["rp_max_deg"] > pc["rp_reject_deg"]:
        raise ValueError(f"policy.rp_max_deg ({pc['rp_max_deg']:g}) must not "
                         f"exceed policy.rp_reject_deg ({pc['rp_reject_deg']:g}): "
                         f"the clip tier sits inside the reject tier")
    for k in ("pq_max_rad_s", "blend_rp_rate_max", "attitude_q_scale"):
        try:
            v = float(pc.get(k, d[k]))
        except (TypeError, ValueError):
            v = float("nan")
        if not (math.isfinite(v) and v > 0.0):
            raise ValueError(f"policy.{k} must be finite and > 0, got {pc.get(k)!r}")
        pc[k] = v
    try:
        v = float(pc.get("anchor_leash_rp_deg", d["anchor_leash_rp_deg"]))
    except (TypeError, ValueError):
        v = float("nan")
    if not (math.isfinite(v) and v >= 0.0):
        raise ValueError(f"policy.anchor_leash_rp_deg must be finite and >= 0 "
                         f"(0 = measured), got {pc.get('anchor_leash_rp_deg')!r}")
    pc["anchor_leash_rp_deg"] = v
    rf = pc.get("rp_ref_filter")
    if rf is not None:
        # The yaw filter's shape, so the day it is implemented the config
        # already validates; until then the worker treats it as off.
        if not isinstance(rf, dict):
            raise ValueError("policy.rp_ref_filter must be null or a mapping "
                             f"{{rate_deg_s, tau_s, guard_deg}}, got {rf!r}")
        want = {"rate_deg_s", "tau_s", "guard_deg"}
        if set(rf) != want:
            raise ValueError(f"policy.rp_ref_filter keys must be exactly "
                             f"{sorted(want)}, got {sorted(rf)}")
        clean = {}
        for k in sorted(want):
            try:
                v = float(rf[k])
            except (TypeError, ValueError):
                v = float("nan")
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError(f"policy.rp_ref_filter.{k} must be finite and "
                                 f"> 0, got {rf[k]!r}")
            clean[k] = v
        pc["rp_ref_filter"] = clean
    else:
        pc["rp_ref_filter"] = None

WALL_PRESETS = {
    "x_into_wall": np.array([[0.0, 0.0, 1.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0]]),
    "x_out_of_wall": np.array([[0.0, 0.0, -1.0],
                               [-1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0]]),
}

# Fallback extrinsic if the measured JSON is missing: the same numbers, frozen
# 2026-08-12 from bluerov2_mujoco_marinegym/meshes/c3_payload_frames.json
# (Onshape registration 2026-07-19; base_link FLU, origin = heavy_c3 COM).
# DEAD IN PRACTICE: config/hw_nav.yaml `cam_t_flu` OVERRIDES this on every run
# since 2026-09-02 (this is the pre-re-mount, sign-flipped value — compute
# nothing from it).
_C3_T_FLU = (0.23949, 0.00547, -0.15537)
_C3_XYAXES = (-0.0, -1.0, -0.0, -0.0056, 0.0, 0.99998)


def _tilt_flu(R_flu_cam: np.ndarray, tilt_deg: float) -> np.ndarray:
    """Rotate a LEVEL camera's axes DOWN by ``tilt_deg`` about body +Y (FLU).

    ``R_y(+th)`` maps +X -> (cos, 0, -sin): the optical axis dips toward the
    floor, which is what "tilt down" means on a mount. Shared by BOTH cameras
    — the C3 was re-pointed at the floor map for the 2026-08 mission and the
    ROV RGB rides an open-loop tilt servo, so neither extrinsic is level by
    assumption any more.

    A wrong angle rotates the whole world estimate, and it does NOT show up in
    reprojection error: solvePnP recovers the CAMERA pose, and the extrinsic
    only maps camera -> body afterwards. What it moves is (a) the reported body
    ATTITUDE by the full angle, and (b) the position by the camera-to-body
    lever arm only. The check is ``StateAssembler.rp_residual_deg`` (tag-implied
    roll/pitch vs ATTITUDE), which is reported on MpcStatus and in the CSV.
    """
    th = math.radians(float(tilt_deg or 0.0))
    if abs(th) <= 1e-9:
        return R_flu_cam
    c, s = math.cos(th), math.sin(th)
    R_tilt = np.array([[c, 0.0, s],
                       [0.0, 1.0, 0.0],
                       [-s, 0.0, c]])
    return R_tilt @ R_flu_cam


def R_flu_cam_from_xyaxes(xyaxes) -> np.ndarray:
    """MuJoCo camera xyaxes (x right, y UP, looks along -z) -> the FLU->optical
    rotation, columns = OpenCV camera axes expressed in FLU body:
    x_cv = cam_x, y_cv = -cam_y (optical y is DOWN), z_cv = x_cv x y_cv."""
    x = np.asarray(xyaxes[:3], float)
    y = np.asarray(xyaxes[3:6], float)
    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)
    x_cv = x
    y_cv = -y
    z_cv = np.cross(x_cv, y_cv)
    return np.column_stack([x_cv, y_cv, z_cv])


@dataclass
class NavConfig:
    """config/hw_nav.yaml, resolved. See the file itself for field-by-field
    provenance notes; unknowns there are tagged [예측]/[스펙] per CLAUDE.md."""

    geometry: str = "wall"                    # "wall" | "floor"
    detector: str = "auto"
    tag_family: str = "tag36h11"
    tag_size_m: float = 0.170                 # config/config.yaml tags.tag_size_m
    tag_map_path: str = "config/tag_map.yaml"
    wall_tag_id: int = 25
    wall_tag_size_m: float | None = None      # None = tag_size_m
    wall_preset: str = "x_into_wall"
    min_tags: int = 1
    max_reproj_px: float = 3.0
    stale_s: float = 0.5
    # Ids that exist TWICE on the physical mat. They never anchor a solve;
    # each detection is kept only if it reprojects within dup_confirm_px of
    # the map pose, using the pose the UNIQUE tags established. See
    # tagnav.py's module docstring for why that is enough.
    duplicate_ids: tuple = ()
    dup_confirm_px: float = 6.0
    # Outlier rescue (tagnav.py _drop_outliers): a frame that fails
    # max_reproj_px drops its worst tag and refits rather than dying, bounded
    # by these two so a uniformly-bad frame is rejected instead of peeled.
    outlier_max_frac: float = 0.30            # of the anchors, always >= 1
    outlier_min_ratio: float = 1.8            # worst / median of the others
    # A detection whose quad comes within this many pixels of the image edge
    # does not anchor (0.0 = only quads that actually cross it; null = off).
    min_border_px: float | None = 0.0
    # Single-tag gravity gate: reject a fix whose implied roll/pitch differs
    # from ATTITUDE by more than this. Assumes the tag hangs VERTICAL and the
    # extrinsic (incl. camera tilt) is right — a leaning paper tag or an
    # un-levelled tilt mount eats every frame at the default 10.
    tilt_gate_deg: float = 10.0
    ambiguity_ratio: float = 1.5
    # AprilTag quad detection decimation: a float for every feed, or a dict
    # {panel: float}. 1.0 = full resolution; 2.0 samples a quarter of the
    # pixels for the quad SEARCH while corners are still refined at full
    # resolution. Rule of thumb: 1.0 for the 640x360 C3 stream (decimating a
    # small image loses far tags for no win), 2.0 for the 720p+ ROV RGB
    # (camera-rate detection).
    quad_decimate: object = None

    def decimate_for(self, panel: str) -> float:
        d = self.quad_decimate
        if isinstance(d, dict):
            return float(d.get(panel, d.get("default", 2.0)))
        if d is None:
            return 1.0 if panel == "main" else 2.0
        return float(d)
    z_source: str = "pressure"                # "pressure" | "tag"
    nav_stream: str = "color"                 # which C3 stream feeds detection
    # WHICH video feed localizes: "main" = the C3 colour stream (factory
    # underwater intrinsics ride every frame) or "second" = the ROV's own RGB
    # (needs the [예측] second_cam block below — there is no factory
    # calibration for that camera, so every number in it is an estimate until
    # someone calibrates it).
    nav_source: str = "main"
    # World datum: "map" = the tag/anchor frame as-is; "first_fix" = the
    # FIRST successful fix defines (0,0,0) and yaw 0 — positions and the
    # square are relative to where the run began. Re-zeroed by
    # toggling the localizing feed's TAG button off and on.
    datum: str = "map"
    cam_t_flu: tuple = _C3_T_FLU
    cam_xyaxes_flu: tuple = _C3_XYAXES
    # Tilt-mount angle for the MAIN (C3) camera, degrees DOWN from the axes
    # above. The measured Onshape registration in cam_xyaxes_flu is the
    # FORWARD-LEVEL mount (its implied pitch is 0.32 deg up); when the C3 is
    # re-pointed at the floor map, this is where that angle goes.
    #
    # The default MUST stay 0.0: it is what every recorded run was flown with,
    # and a non-zero default would retroactively reinterpret all of them.
    # ``_run_meta["hardware"]["cam_tilt_deg"]`` is the record boundary — do not
    # pool positions or yaw across runs whose value differs.
    #
    # NOTE this is applied as a PURE ROTATION about the camera's own origin, so
    # cam_t_flu is assumed unchanged by the re-mount. A hinge moves the lens
    # too; measure it and set cam_t_flu in hw_nav.yaml if that matters at the
    # centimetre level (the lever arm here is ~0.29 m).
    cam_tilt_deg: float = 0.0
    # The ROV default-RGB camera model, used ONLY when nav_source: second.
    # [예측] defaults: tilt mount assumed LEVEL (mount_center), position a
    # rough tape estimate, fx from the vendor's in-air FOV — a wrong fx scales
    # every distance proportionally, so a 20 cm square stays square but not
    # exactly 20 cm. Calibrate before quoting numbers from this source.
    second_cam: dict = field(default_factory=lambda: {
        "t_flu": [0.18, 0.0, 0.0],
        "xyaxes_flu": [0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
        # The tilt-mount angle the extrinsic assumes, degrees DOWN from level.
        # The mount is open-loop (no angle feedback), so this must match what
        # the operator SET: press LEVEL for 0, or drive to the stop and enter
        # that angle here. A wrong tilt rotates the whole world estimate.
        "tilt_deg": 0.0,
        # Mount slew while UP/DOWN is held, deg/s [예측] — what the PAYLOAD
        # panel dead-reckons the angle with between anchors. Time a sweep
        # against the tag-measured tilt and replace it.
        "tilt_rate_deg_s": 30.0,
        "fx": 1144.0, "fy": 1144.0, "cx": 960.0, "cy": 540.0,
        "width": 1920, "height": 1080, "dist": []})
    # SECOND-CAMERA FALLBACK (control/nav_fusion.py). The C3 localizes; when
    # it has had no fix for ``after_s`` the ROV RGB's fix stands in — but
    # only once the constant body-frame offset between the two solvers has
    # been LEARNED from ``min_pairs`` time-paired fixes (``pair_tol_s``),
    # unless ``allow_unaligned`` says a raw RGB pose is acceptable. A tracked
    # mount-tilt change bigger than ``tilt_reset_deg`` throws the alignment
    # away (the extrinsic it corrected for no longer exists).
    fallback: dict = field(default_factory=lambda: {
        "enabled": False, "after_s": 0.2, "pair_tol_s": 0.08,
        "min_pairs": 10, "window": 60, "allow_unaligned": False,
        "tilt_reset_deg": 2.0,
        # The fallback solver's OWN reprojection gate, px. None = the main
        # max_reproj_px. An uncalibrated second camera (no distortion model)
        # cannot fit several tags across a wide frame to 3 px, and until it
        # is calibrated (rov_gui/tools/calibrate_second_cam.py) a looser gate
        # is the difference between a fix and none at all.
        "max_reproj_px": None})
    R_ned_map: np.ndarray = field(default_factory=lambda: np.eye(3))
    pool_draw: tuple = (4.877, 1.8)           # display only
    # Pool boundary rectangle in the MAP/NED frame, {"x": [x0, x1],
    # "y": [y0, y1]} — DISPLAY ONLY (plot border + axis scale). None = the
    # plot falls back to a 4 m box. Placement provenance lives in
    # hw_nav.yaml next to the numbers.
    pool_ned: dict | None = None
    # ...or DERIVE it: the outermost tag EDGES plus this margin, on every
    # side. Preferred over a hand-typed box because it follows the map — a
    # rebuilt or extended map moves the wall with it instead of leaving a
    # stale rectangle behind. See pool_from_tags().
    pool_margin_m: float | None = None
    # The vehicle's footprint (ALONG-heading, ACROSS), metres — display only,
    # drawn around the position on the trajectory plot.
    rov_footprint_m: tuple = (0.4318, 0.5334)
    raw: dict = field(default_factory=dict)

    # ------------------------------------------------------------ derived
    def R_t_frd_cam(self, source: str = "main",
                    tilt_deg: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """The extrinsic the PnP chain consumes: x_bodyFRD = R x_cam + t.
        ``source`` picks the camera: "main" = C3 (measured registration),
        "second" = the ROV RGB ([예측] block). ``tilt_deg`` overrides the
        configured mount angle — the RGB's extrinsic follows the TRACKED
        tilt at runtime (control/tilt_tracker.py)."""
        if source == "second":
            sc = self.second_cam
            R_flu_cam = _tilt_flu(R_flu_cam_from_xyaxes(sc["xyaxes_flu"]),
                                  sc.get("tilt_deg", 0.0)
                                  if tilt_deg is None else float(tilt_deg))
            return (S_FLU_FRD @ R_flu_cam,
                    S_FLU_FRD @ np.asarray(sc["t_flu"], float))
        R_flu_cam = _tilt_flu(R_flu_cam_from_xyaxes(self.cam_xyaxes_flu),
                              self.cam_tilt_deg)
        return S_FLU_FRD @ R_flu_cam, S_FLU_FRD @ np.asarray(self.cam_t_flu, float)

    def second_K(self, w: int, h: int) -> tuple[np.ndarray, np.ndarray | None]:
        """The [예측] second-camera intrinsics, rescaled to the frame size
        actually received (the RTP stream can arrive at any negotiated size)."""
        sc = self.second_cam
        sx = float(w) / float(sc.get("width", w) or w)
        sy = float(h) / float(sc.get("height", h) or h)
        K = np.array([[float(sc["fx"]) * sx, 0.0, float(sc["cx"]) * sx],
                      [0.0, float(sc["fy"]) * sy, float(sc["cy"]) * sy],
                      [0.0, 0.0, 1.0]])
        dist = np.asarray(sc.get("dist", []) or [], float)
        return K, (dist if dist.size else None)

    def pool_from_tags(self, tag_map) -> dict | None:
        """The pool box implied by the map: outermost tag EDGES + the margin.

        Tag poses are CENTRES, so half a tag is added before the margin — the
        operator measures to the printed edge, not to an invisible centre.
        Returns None unless ``pool_margin_m`` is set; an explicit
        ``pool_ned`` in the YAML wins over this (the caller decides).
        """
        if self.pool_margin_m is None or not getattr(tag_map, "instances", None):
            return None
        P = np.array([t for poses in tag_map.instances.values()
                      for (_R, t) in poses], float)
        if P.size == 0:
            return None
        pad = self.effective_tag_size() / 2.0 + float(self.pool_margin_m)
        return {"x": [float(P[:, 0].min() - pad), float(P[:, 0].max() + pad)],
                "y": [float(P[:, 1].min() - pad), float(P[:, 1].max() + pad)]}

    def make_tag_map(self):
        from .tagnav import TagMap
        if self.geometry == "wall":
            m = TagMap.single(self.wall_tag_id)
        else:
            m = TagMap.load(self.tag_map_path)
        return m

    def effective_tag_size(self) -> float:
        if self.geometry == "wall" and self.wall_tag_size_m:
            return float(self.wall_tag_size_m)
        return float(self.tag_size_m)

    @classmethod
    def load(cls, path, repo_root: Path | None = None,
             geometry_override: str | None = None) -> "NavConfig":
        import yaml

        root = Path(repo_root) if repo_root else Path(path).resolve().parents[1]
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if geometry_override:
            raw = dict(raw)
            raw["geometry"] = str(geometry_override)
        cfg = cls(raw=raw)
        for key in ("geometry", "detector", "tag_family", "wall_preset",
                    "z_source", "nav_stream", "nav_source", "datum"):
            if key in raw:
                setattr(cfg, key, str(raw[key]))
        if cfg.nav_source not in ("main", "second"):
            raise ValueError(f"nav_source must be main|second, got "
                             f"{cfg.nav_source!r}")
        if cfg.datum not in ("map", "first_fix"):
            raise ValueError(f"datum must be map|first_fix, got {cfg.datum!r}")
        if "second_cam" in raw and raw["second_cam"]:
            cfg.second_cam = {**cfg.second_cam, **raw["second_cam"]}
        if "fallback" in raw and raw["fallback"]:
            fb = {**cfg.fallback, **dict(raw["fallback"])}
            fb["enabled"] = bool(fb["enabled"])
            fb["allow_unaligned"] = bool(fb["allow_unaligned"])
            for k in ("after_s", "pair_tol_s", "tilt_reset_deg"):
                fb[k] = float(fb[k])
            fb["max_reproj_px"] = (None if fb.get("max_reproj_px") is None
                                   else float(fb["max_reproj_px"]))
            for k in ("min_pairs", "window"):
                fb[k] = int(fb[k])
            cfg.fallback = fb
        for key in ("tag_size_m", "max_reproj_px", "stale_s", "tilt_gate_deg",
                    "ambiguity_ratio", "dup_confirm_px", "cam_tilt_deg",
                    "outlier_max_frac", "outlier_min_ratio"):
            if key in raw:
                setattr(cfg, key, float(raw[key]))
        if "min_border_px" in raw:              # null = keep clipped quads
            v = raw["min_border_px"]
            cfg.min_border_px = None if v is None else float(v)
        if raw.get("duplicate_ids"):
            cfg.duplicate_ids = tuple(sorted(int(v) for v in
                                             raw["duplicate_ids"]))
        if "quad_decimate" in raw:
            qd = raw["quad_decimate"]
            cfg.quad_decimate = (dict(qd) if isinstance(qd, dict)
                                 else float(qd))
        if raw.get("wall_tag_size_m") is not None:
            cfg.wall_tag_size_m = float(raw["wall_tag_size_m"])
        for key in ("wall_tag_id", "min_tags"):
            if key in raw:
                setattr(cfg, key, int(raw[key]))
        if "tag_map" in raw:
            p = Path(raw["tag_map"])
            cfg.tag_map_path = str(p if p.is_absolute() else root / p)
        if "pool_draw_flu" in raw:
            cfg.pool_draw = tuple(float(v) for v in raw["pool_draw_flu"])
        if raw.get("rov_footprint_m"):
            cfg.rov_footprint_m = tuple(float(v)
                                        for v in raw["rov_footprint_m"][:2])
        if raw.get("pool_margin_m") is not None:
            cfg.pool_margin_m = float(raw["pool_margin_m"])
        if "pool_ned" in raw and raw["pool_ned"]:
            g = raw["pool_ned"]
            cfg.pool_ned = {"x": [float(g["x"][0]), float(g["x"][1])],
                            "y": [float(g["y"][0]), float(g["y"][1])]}
        # geofence_ned / geofence_frame: DELIBERATELY not read. The fence
        # was removed on 2026-08-14; silently ignoring the keys means an old
        # hw_nav.yaml still loads rather than crashing the station at startup.
        # (They stay readable in cfg.raw for anyone auditing a past run.)

        # ---- camera extrinsic: measured JSON first, YAML override wins
        js = raw.get("cam_extrinsic_json",
                     "bluerov2_mujoco_marinegym/meshes/c3_payload_frames.json")
        jp = Path(js)
        jp = jp if jp.is_absolute() else root / jp
        if jp.exists():
            try:
                d = json.loads(jp.read_text())
                cfg.cam_t_flu = tuple(float(v) for v in d["cam_center_bl"])
                cfg.cam_xyaxes_flu = tuple(float(v) for v in d["cam_xyaxes"])
            except (KeyError, ValueError, json.JSONDecodeError):
                pass                            # fall back to the frozen copy
        if raw.get("cam_t_flu") is not None:
            cfg.cam_t_flu = tuple(float(v) for v in raw["cam_t_flu"])
        if raw.get("cam_xyaxes_flu") is not None:
            cfg.cam_xyaxes_flu = tuple(float(v) for v in raw["cam_xyaxes_flu"])

        # ---- map -> NED remap
        if raw.get("R_ned_map") is not None:
            cfg.R_ned_map = np.asarray(raw["R_ned_map"], float).reshape(3, 3)
        elif cfg.geometry == "wall":
            try:
                cfg.R_ned_map = WALL_PRESETS[cfg.wall_preset].copy()
            except KeyError:
                raise ValueError(f"unknown wall_preset {cfg.wall_preset!r}; "
                                 f"pick one of {sorted(WALL_PRESETS)}")
        else:
            cfg.R_ned_map = np.eye(3)          # floor map is already NED-like
        d = float(np.linalg.det(cfg.R_ned_map))
        if abs(d - 1.0) > 1e-6:
            raise ValueError(f"R_ned_map must be a proper rotation (det={d:.4f})")
        return cfg


# ArduSub flight-mode names the engage gate may be configured with
# (backends/hardware.py MODE_NUMBERS carries the custom_mode numbers). The
# gate compares HEARTBEAT-reported names, so a typo here would silently
# refuse every engage — hence the vocabulary check.
ARDUSUB_MODE_NAMES = ("MANUAL", "STABILIZE", "ALT_HOLD", "ACRO", "POSHOLD",
                      "SURFACE")


def normalize_require_mode(v) -> tuple:
    """``engage.require_mode`` -> tuple of upper-case ArduSub mode names.

    Accepts one name (``MANUAL``), a list (``[MANUAL, STABILIZE]``), or a
    ``|``/``,``-separated string. Empty means NO flight-mode gate (the
    pre-2026-09-07 meaning of ``require_mode: ""``). Unknown names raise:
    a name the vehicle never reports would refuse every engage."""
    if v is None:
        # `require_mode:` with no value is a YAML accident, not a decision.
        # No gate at all has to be spelled out: [] (or the string "").
        raise ValueError("engage.require_mode is empty (null); write "
                         "require_mode: [] to remove the flight-mode gate "
                         "on purpose, or name the modes, e.g. [MANUAL, STABILIZE]")
    if isinstance(v, str):
        items = [t.strip() for t in v.replace("|", ",").split(",")]
    elif isinstance(v, (list, tuple)):
        items = [str(t).strip() for t in v]
    else:
        raise ValueError(f"engage.require_mode must be a mode name or a list "
                         f"of them, got {v!r}")
    out = tuple(dict.fromkeys(t.upper() for t in items if t))
    bad = [m for m in out if m not in ARDUSUB_MODE_NAMES]
    if bad:
        raise ValueError(f"engage.require_mode names unknown ArduSub mode(s) "
                         f"{bad}; known: {list(ARDUSUB_MODE_NAMES)}")
    return out


STABILIZE_YAW_AXIS_MODES = ("hold", "torque")
STABILIZE_RP_AXIS_MODES = ("hold",)


# ---------------------------------------------------------------------------
# engage.attitude_axes — the 6-DoF variant's actuation switch (2026-09-26)
# ---------------------------------------------------------------------------
#: The only transport implemented: MANUAL mode, K/M as the MANUAL_CONTROL v2
#: extension axes s (pitch) / t (roll), enabled_extensions = 0b11. The
#: design's FALLBACK-A (``rc_override``: RC_CHANNELS_OVERRIDE ch1..ch6 with
#: MANUAL_CONTROL stopped) is a SEPARATE transport with its own deadman
#: semantics and is deliberately NOT implemented in this cut — naming it
#: raises "not implemented" rather than flying a half-built path.
ATTITUDE_AXES_TRANSPORTS = ("manual_control_ext",)
ATTITUDE_AXES_TRANSPORTS_RESERVED = ("rc_override",)


def default_attitude_axes_block() -> dict:
    """``engage.attitude_axes`` defaults: everything OFF, every number [예측]
    or [유도] (design_final D4/D7/D8/D9). One definition — a hand-built
    ``MpcConfig()`` and the station resolve the same block."""
    return {
        "enabled": False,
        "transport": "manual_control_ext",
        # |axis| ceilings for roll / pitch (the SINK sends axis*1000):
        # 0.2 x 13.2 = 2.6 N·m, 0.3 x 7.2 = 2.2 N·m [예측] — the 20 deg hold
        # moment 0.47-0.73 N·m [유도] plus the ~1.4 N·m static payload pitch
        # moment [스펙 BlueROVHeavyGripper.yaml]. Solver U_MAX[3:5] 8 N·m is
        # UNCHANGED, so the optimiser may plan more than the wire carries;
        # the mismatch is recorded (controller.attitude_ref.u_max_wire_nm).
        "cap_roll": 0.2,
        "cap_pitch": 0.3,
        # What the FIRST water session should fly instead [예측]: below the
        # static moment, so an active-levelling run may sit saturated on M —
        # the sat-ineffective interlock is what stops it. ENFORCED (safety
        # audit 2026-09-26): while the wire sign is not proven (no sign_probe
        # artefact with sign_proven true) these — elementwise min with
        # cap_roll / cap_pitch — are the caps on the wire; null here with an
        # unproven sign REFUSES the engage (nothing would bound an inverted
        # sign).
        "first_water_caps": [0.1, 0.15],
        "slew_per_s": 1.5,            # roll/pitch axis slew, 1/s (= axis_slew_per_s)
        # Wire sign of s / t relative to the NED K / M convention [가정 until
        # the armed in-water sign probe writes its artefact]. A reversed sign
        # is positive feedback: the sat-ineffective interlock and the caps
        # are what bound it.
        "sign": {"roll": 1.0, "pitch": 1.0},
        "abort_deg": 35.0,            # |roll| or |pitch| ceiling, 0.25 s -> disengage [예측]
        "sat_ineffective_s": 1.0,     # axis pinned at cap this long with no progress -> disengage [예측]
        "require_manual": True,       # MANUAL only (STABILIZE reinterprets s/t as angles)
        # The bench probe artefact must PARSE at engage (safety audit
        # 2026-09-26): a JSON from rov_gui.tools.attitude_axes_probe with
        # pass true, mavlink_wire_version "2.0" and a firmware_version whose
        # a.b.c equals the vehicle's AUTOPILOT_VERSION — a file that merely
        # exists is not a probe.
        "require_probe": True,
        # The armed in-water sign probe artefact must carry sign_proven
        # true. Nothing writes one in this cut, so on hardware the variant
        # arms ONLY through the explicit, RECORDED opt-out
        # require_sign_probe: false — and then the wire caps in force are
        # first_water_caps, never cap_roll / cap_pitch (meta
        # run.attitude_axes.caps_in_force + sign_proven say which).
        "require_sign_probe": True,
        "probe": None,                # data/<day>/<run>_probe/attitude_axes_probe.json (bench, disarmed)
        "sign_probe": None,           # armed in-water sign probe artefact (mission kind: not in this cut)
        "dobmpc_allowed": False,      # K/M credited to the EAOB only after gain calibration (D10)
        "min_firmware": "4.1.2",      # Sub 4.1.2 (2024-02-22): s/t consumed in MANUAL
    }


ATTITUDE_AXES_KEYS = tuple(default_attitude_axes_block())


def parse_firmware_version(s) -> tuple | None:
    """``"4.1.2"`` (with any trailing text) -> (4, 1, 2); None when unparsable.
    An EMPTY string is None too — a firmware nobody read must refuse."""
    import re

    m = re.match(r"\s*(\d+)\.(\d+)\.(\d+)", str(s or ""))
    if not m:
        return None
    return tuple(int(g) for g in m.groups())


def validate_engage_attitude_axes(engage: dict) -> dict:
    """Merge + validate ``engage.attitude_axes`` IN PLACE and return it.

    Unknown keys RAISE (the imu_dr rule); ``transport`` other than
    ``manual_control_ext`` raises "not implemented"; ``probe`` /
    ``sign_probe``, when set, must be existing file paths — an artefact path
    that does not resolve is a run that would arm on a probe nobody ran.
    Their CONTENT is judged at engage, against the live vehicle
    (``MpcWorker._attitude_artefacts``: tool / pass / wire / firmware match,
    sign_proven), not here — the firmware to match is not known at load."""
    raw = engage.get("attitude_axes")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"engage.attitude_axes must be a mapping, got {raw!r}")
    unknown = set(raw) - set(ATTITUDE_AXES_KEYS)
    if unknown:
        raise ValueError(f"unknown engage.attitude_axes keys {sorted(unknown)}; "
                         f"known: {sorted(ATTITUDE_AXES_KEYS)}")
    aa = default_attitude_axes_block()
    aa.update(raw)
    aa["enabled"] = bool(aa["enabled"])
    tr = str(aa["transport"]).strip().lower()
    if tr in ATTITUDE_AXES_TRANSPORTS_RESERVED:
        raise ValueError(f"engage.attitude_axes.transport {tr!r} is not "
                         f"implemented (design FALLBACK-A: needs its own deadman "
                         f"semantics and a bench rc_parity artefact); use "
                         f"{ATTITUDE_AXES_TRANSPORTS[0]!r}")
    if tr not in ATTITUDE_AXES_TRANSPORTS:
        raise ValueError(f"engage.attitude_axes.transport must be one of "
                         f"{list(ATTITUDE_AXES_TRANSPORTS)}, got {aa['transport']!r}")
    aa["transport"] = tr
    for k in ("cap_roll", "cap_pitch"):
        try:
            v = float(aa[k])
        except (TypeError, ValueError):
            v = float("nan")
        if not (math.isfinite(v) and 0.0 < v <= 1.0):
            raise ValueError(f"engage.attitude_axes.{k} must be in (0, 1], got {aa[k]!r}")
        aa[k] = v
    fw = aa.get("first_water_caps")
    if fw is not None:
        try:
            fwv = [float(x) for x in fw]
        except (TypeError, ValueError):
            fwv = []
        if len(fwv) != 2 or not all(math.isfinite(x) and 0.0 < x <= 1.0 for x in fwv):
            raise ValueError(f"engage.attitude_axes.first_water_caps must be "
                             f"[roll, pitch] in (0, 1], got {fw!r}")
        aa["first_water_caps"] = fwv
    for k in ("slew_per_s", "abort_deg", "sat_ineffective_s"):
        try:
            v = float(aa[k])
        except (TypeError, ValueError):
            v = float("nan")
        if not (math.isfinite(v) and v > 0.0):
            raise ValueError(f"engage.attitude_axes.{k} must be finite and > 0, "
                             f"got {aa[k]!r}")
        aa[k] = v
    if aa["abort_deg"] > 89.0:
        raise ValueError(f"engage.attitude_axes.abort_deg must be < 89 deg "
                         f"(the T(eta) singularity), got {aa['abort_deg']!r}")
    sg = aa.get("sign")
    if not isinstance(sg, dict) or set(sg) != {"roll", "pitch"}:
        raise ValueError("engage.attitude_axes.sign must be {roll: +-1, "
                         f"pitch: +-1}}, got {sg!r}")
    clean = {}
    for k in ("roll", "pitch"):
        try:
            v = float(sg[k])
        except (TypeError, ValueError):
            v = float("nan")
        if v not in (1.0, -1.0):
            raise ValueError(f"engage.attitude_axes.sign.{k} must be +1 or -1, "
                             f"got {sg[k]!r}")
        clean[k] = v
    aa["sign"] = clean
    for k in ("require_manual", "require_probe", "require_sign_probe",
              "dobmpc_allowed"):
        aa[k] = bool(aa[k])
    for k in ("probe", "sign_probe"):
        v = aa.get(k)
        if v is None:
            continue
        v = str(v)
        if not Path(v).is_file():
            raise ValueError(f"engage.attitude_axes.{k} names {v!r}, which is "
                             f"not an existing file — pin the artefact the "
                             f"probe wrote, or null")
        aa[k] = v
    mf = aa.get("min_firmware", "4.1.2")
    if parse_firmware_version(mf) is None:
        raise ValueError(f"engage.attitude_axes.min_firmware must read "
                         f"'a.b.c', got {mf!r}")
    aa["min_firmware"] = str(mf).strip()
    if aa["enabled"] and not aa["require_manual"]:
        # Not a safety knob to turn off by config: outside MANUAL the wire
        # axes are lean-angle targets, never torques (design D7).
        raise ValueError("engage.attitude_axes.require_manual must stay true "
                         "while enabled (STABILIZE reinterprets s/t as angles)")
    engage["attitude_axes"] = aa
    return aa


def check_engage_stabilize(engage: dict) -> None:
    """Validate ``engage.mode_settle_s`` and ``engage.stabilize`` in place."""
    v = engage.get("mode_settle_s", 2.0)
    try:
        v = float(v)
    except (TypeError, ValueError) as e:
        raise ValueError(f"engage.mode_settle_s must be a number, got {v!r}") from e
    if not (math.isfinite(v) and v >= 0.0):
        raise ValueError(f"engage.mode_settle_s must be finite and >= 0, got {v!r}")
    engage["mode_settle_s"] = v
    st = engage.get("stabilize") or {}
    if not isinstance(st, dict):
        raise ValueError(f"engage.stabilize must be a mapping, got {st!r}")
    unknown = set(st) - {"yaw_axis", "rp_axis"}
    if unknown:
        raise ValueError(f"unknown engage.stabilize keys {sorted(unknown)}; "
                         f"known: ['yaw_axis', 'rp_axis']")
    ya = str(st.get("yaw_axis", "hold")).strip().lower()
    if ya == "rate":
        raise ValueError("engage.stabilize.yaw_axis: 'rate' is not implemented "
                         "— inverting ArduSub's yaw deadzone needs the runtime "
                         "PilotGain, which the station does not read yet")
    if ya not in STABILIZE_YAW_AXIS_MODES:
        raise ValueError(f"engage.stabilize.yaw_axis must be one of "
                         f"{list(STABILIZE_YAW_AXIS_MODES)}, got {st.get('yaw_axis')!r}")
    out = {"yaw_axis": ya}
    if "rp_axis" in st:
        # The station's ROLL/PITCH axes under STABILIZE. `hold` (send 0; the
        # autopilot levels) is the ONLY value: in STABILIZE s/t are lean-angle
        # targets (Attitude.cpp get_pilot_desired_lean_angles), so the NMPC's
        # torque model would be reinterpreted as angles — the angle cascade
        # is a documented future run kind, not implemented (design D7).
        # engage.attitude_axes is refused outside MANUAL for the same reason.
        # Written into the resolved dict only when the file spells it, so a
        # pre-variant block resolves to exactly what it did before.
        ra = str(st.get("rp_axis", "hold")).strip().lower()
        if ra not in STABILIZE_RP_AXIS_MODES:
            raise ValueError(f"engage.stabilize.rp_axis must be one of "
                             f"{list(STABILIZE_RP_AXIS_MODES)} (the STABILIZE "
                             f"lean-angle cascade is not implemented), got "
                             f"{st.get('rp_axis')!r}")
        out["rp_axis"] = ra
    engage["stabilize"] = out


VEHICLE_NET_BUOYANCY_MAX_N = 15.0


def check_vehicle_net_buoyancy(v) -> float:
    """Validate hw_mpc.yaml ``vehicle_net_buoyancy_n`` (B - W of the real
    vehicle, + floats). Shared by MpcConfig.load and HwDobMpc.__init__."""
    try:
        v = float(v)
    except (TypeError, ValueError) as e:
        raise ValueError(f"vehicle_net_buoyancy_n must be a number, got {v!r}") from e
    if not np.isfinite(v) or abs(v) > VEHICLE_NET_BUOYANCY_MAX_N:
        raise ValueError(
            f"vehicle_net_buoyancy_n must be a finite B-W within "
            f"+/-{VEHICLE_NET_BUOYANCY_MAX_N:.0f} N, got {v!r}")
    return v


@dataclass
class MpcConfig:
    """config/hw_mpc.yaml, resolved. Axis gains and EAOB sigmas are [예측]
    until the P4 step-calibration / P3 residual fit replace them."""

    rov_model: str = "heavy_gripper"
    # "none" | "mpc" | "dobmpc" | "mpc_tuned" | "dobmpc_tuned" | "pid" | "rl" | "rl_pwm" (the
    # file / CLI vocabulary, validated in load()). "none" = LOW level None,
    # TELEOP (2026-09-11): the station commands nothing.
    mode: str = "dobmpc"
    ctrl_hz: float = 20.0
    # PID overrides (rov_gui/control/pid.py SIM_GAINS is the base) — most
    # importantly omega_derate, the hardware detune of the sim pole placement.
    pid: dict = field(default_factory=dict)
    # RL follower (rov_gui/control/rl_policy.py): policy_dir (repo-relative), axis_slew_per_s override (default 0).
    rl: dict = field(default_factory=dict)
    # LOW mode rl_pwm (2026-09-30): the per-thruster RL policy (control/rl_policy.HwRlPwm) — policy_dir, pwm_cap.
    # Empty = the shipped default (rl_policies/pwm10_s1, cap 0.25).
    rl_pwm: dict = field(default_factory=dict)
    # Bridge tag fixes with the velocity estimate + gyro between camera
    # frames (state_assembler): the position the EAOB/PID sees (and the plot
    # draws) moves at the control rate instead of holding the last fix.
    vel_propagation: bool = True
    # roll_nm / pitch_nm (2026-09-26, the 6-DoF variant): the K / M the
    # verticals produce at full deflection [유도: heave_n 60 N = 4 x 15 N per
    # vertical x lever arms |y| 0.22 / |x| 0.12 m, bluerov_heavy.xml:116-119].
    # Read only under engage.attitude_axes (allocation.wrench_to_axes).
    axis_gain: dict = field(default_factory=lambda: {
        "surge_n": 60.0, "sway_n": 60.0, "heave_n": 60.0, "yaw_nm": 20.0,
        "roll_nm": 13.2, "pitch_nm": 7.2})
    axis_cap: float = 0.5
    # Rate limit on the axis command [1/s]; 0 = none (pre-2026-08-17).
    axis_slew_per_s: float = 0.0
    # PATH FOLLOWING (default) vs trajectory tracking. The worker projects the
    # measured vehicle onto the active segment and builds one shared spatial
    # plan for the PID. False = the legacy wall-clock reference.
    #
    # (The corner GATE this block used to configure — path_corner_tol_m /
    # _speed_m_s / _dwell_s — was removed 2026-08-16 at the operator's request.
    # It brought the vehicle to a full stop inside 5 cm of every vertex and
    # hid the next leg until it did, which is precisely why an MPC horizon
    # could never see round a corner. The mission is now a C1 curve
    # (path_geometry) that is simply flown.)
    path_following: bool = True
    path_lead_m: float = 0.15
    # MPCC path shaping. fillet rounds rectangle corners; turn_radius
    # optionally rounds a line's turnarounds into a stadium (0 = reverse in
    # place, keeping the mission's exact geometry). The two accelerations set
    # the feasible speed profile: v <= sqrt(a_lat * R) through a fillet, and
    # a_long limits how fast the reference may speed up or slow down.
    path_fillet_m: float = 0.06
    path_turn_radius_m: float = 0.0
    path_lat_accel_m_s2: float = 0.30
    path_long_accel_m_s2: float = 0.15
    path_creep_m_s: float = 0.02
    traj_timeout_factor: float = 3.0
    w_hat_clip: tuple = (15.0, 45.0, 45.0, 5.0, 5.0, 8.0)
    # NET BUOYANCY OF THE VEHICLE ACTUALLY FLOWN, B - W in newtons (+ floats,
    # - sinks). The NMPC's plant (dobmpc/params.py NET_BUOYANCY: -5.71 N for
    # heavy_gripper, a CAD/vendor composition never measured on this vehicle)
    # is baked into the generated solver, and the solver cancels ITS OWN
    # model's gravity with a constant up-force at zero depth error. On
    # 2026-09-07 the pool vehicle floated instead, so every mpc / mpc_tuned
    # tick carried a wrong-signed 5.7 N heave feedforward and the policy
    # follower could not push down (fit uZ = -5.6 N - 60 N/m * ez, R^2
    # 0.54-0.94: data/20260907/0907_180038/
    # mpc_18*.csv). HwDobMpc feeds (model - this) to the mpc / mpc_tuned
    # solver as a constant heave wrench through the disturbance parameter;
    # the dobmpc modes ignore it because the EAOB estimates the same residual
    # online. 0.0 = neutral — the honest single number for a trim that moved
    # between about -2 N and > +3 N across the 2026-09-07 sessions.
    # |value| <= VEHICLE_NET_BUOYANCY_MAX_N (15 N), see check_vehicle_net_buoyancy.
    vehicle_net_buoyancy_n: float = 0.0
    nudot_source: str = "imu"                 # "imu" | "fd"
    # 0.6 since 2026-08-14 — see hw_mpc.yaml for the measurement that moved it
    # off 0.35. Kept equal to the shipped config so a hand-built MpcConfig()
    # (a test, an embedder) filters the same way the station does.
    vel_lp_alpha: float = 0.6
    eaob_sigmas: dict = field(default_factory=dict)
    # PREDICTION-MODEL DAMPING as measured on THIS vehicle, positive
    # coefficients [surge sway heave roll pitch yaw]; empty = fly marinegym's
    # BlueROV.yaml values. Keys: linear_damping, quadratic_damping. Applied by
    # mpc_bridge.apply_plant_overrides BEFORE the acados build (the damping is
    # inside the generated dynamics), and recorded in the run meta as
    # controller.plant_overrides. See hw_mpc.yaml for the measurement.
    plant: dict = field(default_factory=dict)
    # ALONG/CROSS COST SPLIT for the ``mpc_tuned`` / ``dobmpc_tuned`` modes.
    # Empty = path_cost.DEFAULT_TUNE. Only these modes read it; ``mpc`` and
    # ``dobmpc`` fly the isotropic Q whatever is written here, so the block
    # can sit in the file while a baseline run is flown.
    mpc_tuned: dict = field(default_factory=dict)
    # STATION BRIDGE — carry a station hold across a tag dropout instead of
    # disengaging. Empty = station_bridge.DEFAULTS. STATION MODE ONLY.
    station_bridge: dict = field(default_factory=dict)
    # OBJECT FOLLOW — place the --pose tracked object in the tag-map frame and
    # (shape: follow) hold a captured relative pose on it. Empty =
    # object_nav.DEFAULTS. Needs --pose AND --mpc; see control/object_nav.py.
    object_nav: dict = field(default_factory=dict)
    # The mission. ``shape`` picks the path: "square" (a rectangle — size is
    # the x side, size_y the y side), "line" (out-and-back along dir_deg, one
    # lap = there AND back), or "circle" (radius, one lap = one revolution).
    # ``origin_tag`` anchors the path to a physical tag id, which is how the
    # operator names a place; without it the path starts wherever the vehicle
    # is at START. For every shape the tag is a point the path PASSES THROUGH,
    # never a centre it merely orbits: the rectangle's min-x/min-y corner, the
    # line's near end, the circle's minimum-x rim point. "follow" is the one
    # shape with no geometry at all — it holds a relative pose on the tracked
    # object and ignores origin_tag/size/speed entirely.
    square: dict = field(default_factory=lambda: {
        "shape": "square",           # station|line|square|circle|follow
        "size": 1.0, "size_y": None, "speed": 0.12, "laps": 3,
        "length": 2.0, "dir_deg": 90.0, "ramp_s": 1.0,
        "radius": 0.5,               # CIRCLE only; the tag is on the rim
        "depth_ned": None,
        "heading_follow": False, "yaw_rate_deg_s": 60.0,
        "origin": "current", "origin_tag": None, "rot_deg": 0.0,
        # Heading held during the mission. "current" = whatever the vehicle
        # had at START. A NUMBER in yaw_map_deg is an ABSOLUTE heading in the
        # tag-map frame (90 = facing +y), which is what "sit on the tag facing
        # +y" means and what a datum-relative angle cannot express.
        "yaw_fixed_deg": "current", "yaw_map_deg": None,
        # STATION only: face THIS tag id from origin_tag — the map bearing
        # origin -> tag, so "sit on 47 looking at 97" is the same physical
        # heading every run. Takes priority over yaw_map_deg. None = off.
        "heading_tag": None})
    # The approach/settle keys are HERE and not only in the YAML for the
    # vel_lp_alpha reason: a hand-built MpcConfig() (a test, an embedder) must
    # fly the same numbers the station does. approach_speed_m_s and
    # approach_lead_m in particular only mean anything TOGETHER — the leash is
    # what actually set the speed until 2026-08-18 (see hw_mpc.yaml).
    engage: dict = field(default_factory=lambda: {
        # MANUAL or STABILIZE (operator decision 2026-09-08; the shipped
        # YAML says the same so a hand-built config gates like the station).
        "require_mode": ("MANUAL", "STABILIZE"),
        "probe_ms_max": 25.0, "tag_stale_s": 0.5,
        "tag_stale_hold_s": 0.4,
        # Refuse START this long after an observed flight-mode change: ArduSub
        # does not relax its attitude target / rate-PID integrators on an armed
        # MANUAL->STABILIZE entry [예측: ArduSub control_stabilize.cpp
        # stabilize_init, unmeasured on this vehicle].
        "mode_settle_s": 2.0,
        # What the station does with its YAW axis when it engaged in
        # STABILIZE, where ArduSub reads that axis as a yaw-RATE demand
        # (|axis| <= 0.10 ignored, heading hold 250 ms later):
        #   hold   - send 0; the autopilot's heading hold owns heading (DEFAULT)
        #   torque - send the follower's N unchanged (experiment only)
        #   rate   - reserved: a deadzone-inverting remap needs PilotGain
        "stabilize": {"yaw_axis": "hold"},
        "imu_stale_s": 0.3, "warmup_s": 1.5, "start_err_max_m": 0.3,
        "settle_s": 10.0, "settle_station_s": 0.0,
        "approach_speed_m_s": 0.20, "approach_lead_m": 0.50,
        "approach_max_s": 180.0,
        # FOLLOW's feedforward ceiling — the vehicle's own top speed, so an
        # object estimate that claims to be moving faster cannot relocate the
        # far end of the MPC horizon (hw_mpc.yaml carries the derivation).
        "follow_ff_max_m_s": 0.30,
        "max_solver_fails": 3, "tick_overrun_ms": 100.0,
        # THE 6-DoF VARIANT'S ACTUATION SWITCH (2026-09-26): K/M on the
        # MANUAL_CONTROL v2 extension axes. OFF here and in the shipped
        # YAML; with it off every control output, xref and wire frame is
        # byte-identical to the 4-DoF station (test_attitude_axes.py).
        "attitude_axes": default_attitude_axes_block()})
    # DEMO REPLAY (shape: replay) — re-fly one recorded handheld demonstration.
    # The session dir must hold poses.npy/poses.json (umi_handheld.extract_pose
    # output). The track is re-expressed relative to its own first pose and
    # anchored at the vehicle's pose when the mission arms, so a bench demo
    # replays wherever the vehicle happens to be — no tag id enters it.
    #
    # stream_period_s = 0 installs the whole track as ONE plan (M0a: the
    # minimal label→flight check); > 0 chops it into `horizon_s` windows
    # released every period through the PlanFilter/PlanStitcher (M0b: the
    # exact seam a live diffusion policy will feed later).
    #
    # gripper: OFF by default — the jaw path is open-loop (no position
    # feedback exists on the real vehicle) and early replays should prove the
    # trajectory before the jaw moves at all. The thresholds read the demo's
    # gripper_width channel (0 = closed): width below close_below drives
    # CLOSE, above open_above drives OPEN, in between holds; hold_max_s is the
    # anti-stall auto-neutral (UMI-U used 4 s). gripper_lookahead_s is WHEN
    # on the plan the channel is sampled (now + this; 0 = at now) — see
    # default_policy_block for the 2026-09-07 deadlock it exists for; the
    # same key in both blocks because MpcWorker._stream_cfg reads whichever
    # kind is flying.
    replay: dict = field(default_factory=lambda: {
        "session": "",                # data/<YYYYMMDD>/demonstration_NNNN (poses.npy)
        "v_max_m_s": 0.12,            # [예측] time-dilation ceiling; replace
                                      # with the measured achievable speed
        "blend_s": 0.4,
        "stream_period_s": 0.0,       # 0 = one-shot plan; >0 = 1 Hz-style feed
        "horizon_s": 4.0,
        "anchor_max_m": 0.30,         # [예측] plan-vs-reference gates — see
        "jump_max_m": 0.20,           #   control/plan_stream.FilterLimits
        # The ONLY position-based protection left after the geofence removal
        # (2026-08-14): a streamed plan with any knot outside this datum-NED
        # box is REJECTED. None disables the check.
        "workspace_box_ned": None,    # [[xmin,ymin,zmin],[xmax,ymax,zmax]]
        "gripper": False,
        "gripper_close_below": 0.35,
        "gripper_open_above": 0.65,
        "gripper_hold_max_s": 4.0,
        "gripper_lookahead_s": 0.0,   # 0 = sample at now (pre-2026-09-07)
        # 2026-09-26: re-fly the demo's roll/pitch too (ReplayTrack.rp
        # through PlanMsg.rp). False = level (every replay on record).
        "track_attitude": False,
    })
    # LIVE DIFFUSION POLICY (shape: policy, 2026-09-02, HARDWARE-UNVERIFIED).
    # The trained UMI policy's action chunks (backends/policy.py ->
    # MpcWorker) through the SAME PlanFilter/PlanStitcher seam as replay.
    # Every number carries its provenance in POLICY_PROVENANCE (mirrored in
    # config/hw_mpc.yaml); the filter limits are ALL explicit here (v2 A5),
    # the box is REQUIRED (None refuses the shape, unlike replay — v2 A21),
    # and the TCP is the ROV's own jaw in body FLU, never the handheld's
    # camera offset (v2 A3). Unknown keys RAISE, the replay rule.
    policy: dict = field(default_factory=default_policy_block)
    # IMU DEAD RECKONING — the "how far can the IMU carry us" experiment.
    # OFF by default, and off means the tick is byte-identical to a build
    # without it (pinned by test_control's shadow-parity test).
    #
    #   mode: shadow   the controller keeps flying on the tag state; the DR
    #                  integrates beside it and is only published/recorded.
    #         control  the controller consumes the DR state instead. The tag
    #                  stays alive as ground truth AND as the operator's
    #                  instrument. Needs --imu-dr-control as well: nothing
    #                  should enter a closed loop on dead reckoning because a
    #                  config file was edited.
    imu_dr: dict = field(default_factory=lambda: {
        "enabled": False,
        "mode": "shadow",              # "shadow" | "control"
        "source": "c3",                # the C3 BNO086 (camera-rigid)
        "attitude": "ahrs",            # "gyro" | "ahrs" | "vehicle"
        "ahrs_tau_s": 10.0,
        "accel_trust_m_s2": 0.15,
        "z_source": "pressure",        # "pressure" | "imu"
        "static_window_s": 8.0,
        "gyro_static_std_max": 0.02,
        "gyro_bias_sem_max": 0.005,
        "accel_static_sd_max": 0.15,
        "calibration": "config/c3_imu_calib.json",
        "max_dt_s": 0.1,
        # Automatic aborts, DISABLED by operator decision (2026-08-16): the
        # pilot stops the run by hand. The keys exist so turning one on later
        # is a config edit and not a code change. None = no limit.
        "abort_err_m": None,
        "abort_max_s": None,
        "axis_cap_dr": None,           # None = use axis_cap
    })
    # ROOT of the dated run tree (rov_gui/runstore.py), not the folder
    # written to: the CSV, its .meta.json and events.log land in
    # <log_dir>/YYYYMMDD/MMDD_HHMMSS/ beside the recordings of the same run.
    # The DEFAULT is load-bearing — a config file with no log_dir: key, or
    # a hand-built MpcConfig(), must not fall back into the old flat tree.
    log_dir: str = "data"
    raw: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path) -> "MpcConfig":
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        cfg = cls(raw=raw)
        for key in ("rov_model", "mode", "nudot_source", "log_dir"):
            if key in raw:
                setattr(cfg, key, str(raw[key]))
        if "path_following" in raw:
            cfg.path_following = bool(raw["path_following"])
        for key in ("ctrl_hz", "axis_cap", "axis_slew_per_s",
                    "vel_lp_alpha", "path_lead_m",
                    "path_fillet_m", "path_turn_radius_m",
                    "path_lat_accel_m_s2", "path_long_accel_m_s2",
                    "path_creep_m_s", "traj_timeout_factor"):
            if key in raw:
                setattr(cfg, key, float(raw[key]))
        if "axis_gain" in raw:
            # Unknown keys RAISE (2026-09-26; update() was silent before): a
            # `rol_nm:` typo would fly the [유도] default while the file — and
            # the run meta copied from it — says otherwise.
            unknown = set(raw["axis_gain"] or {}) - set(cfg.axis_gain)
            if unknown:
                raise ValueError(f"unknown axis_gain keys {sorted(unknown)}; "
                                 f"known: {sorted(cfg.axis_gain)}")
            cfg.axis_gain.update({k: float(v) for k, v in raw["axis_gain"].items()})
            for k, v in cfg.axis_gain.items():
                if not (math.isfinite(v) and v > 0.0):
                    raise ValueError(f"axis_gain.{k} must be finite and > 0, got {v!r}")
        if "w_hat_clip" in raw:
            cfg.w_hat_clip = tuple(float(v) for v in raw["w_hat_clip"])
        if "vehicle_net_buoyancy_n" in raw:
            cfg.vehicle_net_buoyancy_n = float(raw["vehicle_net_buoyancy_n"])
        # |v| <= VEHICLE_NET_BUOYANCY_MAX_N: at zero error the solver
        # commands ~v N of heave, so a typo here is a constant feedforward.
        # 15 N leaves half the heave authority (U_MAX 30 N = heave_n 60 x
        # axis_cap 0.5) to the position term and bounds a wrong-signed
        # entry to a ~0.25 m parking error at 60 N/m; every 2026-09-07
        # session implied |B-W| < 8 N. The bridge repeats the check so a
        # hand-built MpcConfig cannot bypass it.
        check_vehicle_net_buoyancy(cfg.vehicle_net_buoyancy_n)
        if "eaob_sigmas" in raw and raw["eaob_sigmas"]:
            cfg.eaob_sigmas = dict(raw["eaob_sigmas"])
        if "plant" in raw and raw["plant"]:
            # Unknown keys and bad shapes RAISE (station_bridge rule): a
            # silently ignored damping override means the run flies the SIM's
            # drag while its meta and this file say it flies the vehicle's.
            from .mpc_bridge import resolve_plant

            cfg.plant = resolve_plant(raw["plant"])
        if "object_nav" in raw and raw["object_nav"]:
            # Unknown keys RAISE, same rule as station_bridge and imu_dr.
            from .object_nav import resolve as _resolve_object

            _resolve_object(raw["object_nav"])
            cfg.object_nav = dict(raw["object_nav"])
        if "station_bridge" in raw and raw["station_bridge"]:
            # Unknown keys RAISE, same rule as imu_dr and mpc_tuned: a
            # silently ignored key here means the operator flies believing a
            # safety ladder is armed when it is not.
            from .station_bridge import resolve as _resolve_bridge

            _resolve_bridge(raw["station_bridge"])
            cfg.station_bridge = dict(raw["station_bridge"])
        if "mpc_tuned" in raw and raw["mpc_tuned"]:
            # Unknown keys and non-positive weights RAISE here, for the imu_dr
            # reason: a silently ignored `cross_sale:` means the run flies the
            # BASELINE cost while its meta says "tuned", and the two CSVs are
            # then indistinguishable from two runs of the same controller.
            from .path_cost import resolve_tune

            resolve_tune(raw["mpc_tuned"])
            cfg.mpc_tuned = dict(raw["mpc_tuned"])
        if "square" in raw and raw["square"]:
            cfg.square.update(raw["square"])
        # WHICH SHAPE, validated — AFTER the merge, so the name checked is the
        # one that will fly. Until 2026-08-21 `_arm_path`'s else branch was
        # "square", so a typo ("staton", "squrae") silently flew a RECTANGLE:
        # a mission of a completely different size and duration from the one
        # asked for, with nothing on screen or in the meta saying the name had
        # not been understood.
        shape = str(cfg.square.get("shape", "square")).lower()
        if shape not in SHAPES:
            raise ValueError(f"square.shape must be one of {list(SHAPES)}, "
                             f"got {cfg.square.get('shape')!r}")
        cfg.square["shape"] = shape
        if "engage" in raw and raw["engage"]:
            # The ONE safety block that used to merge blind. object_nav,
            # station_bridge, imu_dr and mpc_tuned all reject unknown keys for
            # the same reason, and this block holds the interlocks:
            # tick_overrun_ms, max_solver_fails, tag_stale_s,
            # follow_ff_max_m_s. A typo here reads as armed and is not.
            unknown = set(raw["engage"]) - set(cfg.engage)
            if unknown:
                raise ValueError(
                    f"unknown engage keys {sorted(unknown)}; known: "
                    f"{sorted(cfg.engage)}")
            cfg.engage.update(raw["engage"])
            # ...and a limit that is zero, negative or NaN is not a limit.
            # `follow_ff_max_m_s: 0` is exactly what an operator would type to
            # mean "no feedforward"; it used to mean "uncapped", which is the
            # 2026-08-23 failure verbatim.
            for k in ("follow_ff_max_m_s", "tick_overrun_ms",
                      "approach_speed_m_s", "approach_lead_m"):
                v = float(cfg.engage[k])
                if not (math.isfinite(v) and v > 0.0):
                    raise ValueError(f"engage.{k} must be finite and > 0, "
                                     f"got {cfg.engage[k]!r}")
                cfg.engage[k] = v
            if int(cfg.engage["max_solver_fails"]) < 1:
                raise ValueError("engage.max_solver_fails must be >= 1 "
                                 "(0 disengages on the first tick)")
            # a mode name, a list of them, or empty; validated vocabulary
            cfg.engage["require_mode"] = normalize_require_mode(
                cfg.engage.get("require_mode", "MANUAL"))
            check_engage_stabilize(cfg.engage)
            validate_engage_attitude_axes(cfg.engage)
        if "pid" in raw and raw["pid"]:
            cfg.pid = dict(raw["pid"])
        if "rl" in raw and raw["rl"]:
            cfg.rl = dict(raw["rl"])
        if "rl_pwm" in raw and raw["rl_pwm"]:
            cfg.rl_pwm = dict(raw["rl_pwm"])
            # NO pwm_channels key: the vehicle's script maps RC channel 8+m to motor m and nothing else, so a
            # configurable order could only ever swap thrusters silently (safety review 2026-09-30).
            unknown = set(cfg.rl_pwm) - {"policy_dir", "pwm_cap", "leash_m", "yaw_rate_ff", "yaw_rate_max"}
            if unknown:
                # raw thruster pulses: a misspelt pwm_cap silently falling back to a default is not acceptable here
                raise ValueError(f"rl_pwm: unknown key(s) {sorted(unknown)}")
            cap = cfg.rl_pwm.get("pwm_cap")
            if cap is not None and (isinstance(cap, bool) or not isinstance(cap, (int, float))
                                    or not (math.isfinite(float(cap)) and 0.0 < float(cap) <= 1.0)):
                # `pwm_cap: true` is YAML for 1.0 = full throttle; a number or nothing
                raise ValueError(f"rl_pwm.pwm_cap must be a number in (0, 1], got {cap!r}")
        if "vel_propagation" in raw:
            cfg.vel_propagation = bool(raw["vel_propagation"])
        if "imu_dr" in raw and raw["imu_dr"]:
            # Unknown keys RAISE here, unlike everywhere else in this file.
            # The failure mode is what earns the exception: `imu-dr:` for
            # `imu_dr:`, or `attitude_mode:` for `attitude:`, is silently
            # ignored — and the operator flies a run believing the estimator
            # is on and configured when it is off or on a default. A crash at
            # startup costs a minute; a wasted pool session costs an evening.
            unknown = set(raw["imu_dr"]) - set(cfg.imu_dr)
            if unknown:
                raise ValueError(
                    f"unknown imu_dr keys {sorted(unknown)}; "
                    f"known: {sorted(cfg.imu_dr)}")
            cfg.imu_dr.update(raw["imu_dr"])
        if "replay" in raw and raw["replay"]:
            # Unknown keys RAISE, the imu_dr rule: `griper:` for `gripper:` or
            # `vmax:` for `v_max_m_s:` silently ignored means a replay flown
            # believing a limit is armed when it is not.
            unknown = set(raw["replay"]) - set(cfg.replay)
            if unknown:
                raise ValueError(
                    f"unknown replay keys {sorted(unknown)}; known: "
                    f"{sorted(cfg.replay)}")
            cfg.replay.update(raw["replay"])
            for k in ("v_max_m_s", "blend_s", "horizon_s",
                      "gripper_hold_max_s", "anchor_max_m", "jump_max_m"):
                # anchor/jump included on purpose: `anchor_max_m: .nan` makes
                # every filter comparison False, i.e. the gate silently
                # PASSES — and the box below is the only position-based
                # protection left (safety review 2026-08-30).
                v = float(cfg.replay[k])
                if not (math.isfinite(v) and v > 0.0):
                    raise ValueError(f"replay.{k} must be finite and > 0, "
                                     f"got {cfg.replay[k]!r}")
                cfg.replay[k] = v
            box = cfg.replay.get("workspace_box_ned")
            if box is not None:
                b = np.asarray(box, float)
                if b.shape != (2, 3) or not np.all(np.isfinite(b)):
                    raise ValueError(
                        f"replay.workspace_box_ned must be "
                        f"[[xmin,ymin,zmin],[xmax,ymax,zmax]] (finite), "
                        f"got {box!r}")
                if not np.all(b[0] < b[1]):
                    raise ValueError(
                        f"replay.workspace_box_ned min must be < max on "
                        f"every axis, got {box!r}")
                cfg.replay["workspace_box_ned"] = [
                    [float(v) for v in b[0]], [float(v) for v in b[1]]]
            sp = float(cfg.replay["stream_period_s"])
            if not (math.isfinite(sp) and sp >= 0.0):
                raise ValueError(f"replay.stream_period_s must be finite and "
                                 f">= 0, got {cfg.replay['stream_period_s']!r}")
            lo = float(cfg.replay["gripper_close_below"])
            hi = float(cfg.replay["gripper_open_above"])
            if not (0.0 <= lo < hi <= 1.0):
                raise ValueError(
                    f"replay gripper thresholds must satisfy 0 <= close_below "
                    f"< open_above <= 1, got {lo!r} / {hi!r}")
            try:
                la = float(cfg.replay["gripper_lookahead_s"])
            except (TypeError, ValueError):
                la = float("nan")
            if not (math.isfinite(la) and 0.0 <= la <= 3.0):
                raise ValueError(
                    f"replay.gripper_lookahead_s must be a finite number in "
                    f"[0, 3.0] s, got {cfg.replay['gripper_lookahead_s']!r}")
            cfg.replay["gripper_lookahead_s"] = la
            cfg.replay["track_attitude"] = bool(cfg.replay.get("track_attitude", False))
        if "policy" in raw and raw["policy"]:
            # Unknown keys RAISE, the replay/imu_dr rule: `v_max:` for
            # `v_max_m_s:` or `anchor_leash:` for `anchor_leash_m:` silently
            # ignored means a policy flown believing a gate is armed when the
            # default is what is flying. Every numeric knob is then coerced
            # and range-checked (validate_policy_block).
            unknown = set(raw["policy"]) - set(cfg.policy)
            if unknown:
                raise ValueError(
                    f"unknown policy keys {sorted(unknown)}; known: "
                    f"{sorted(cfg.policy)}")
            cfg.policy.update(raw["policy"])
        validate_policy_block(cfg.policy)
        if cfg.imu_dr["mode"] not in ("shadow", "control"):
            raise ValueError(f"imu_dr.mode must be shadow|control, got "
                             f"{cfg.imu_dr['mode']!r}")
        if cfg.imu_dr["attitude"] not in ("gyro", "ahrs", "vehicle"):
            raise ValueError(f"imu_dr.attitude must be gyro|ahrs|vehicle, got "
                             f"{cfg.imu_dr['attitude']!r}")
        if cfg.imu_dr["z_source"] not in ("pressure", "imu"):
            raise ValueError(f"imu_dr.z_source must be pressure|imu, got "
                             f"{cfg.imu_dr['z_source']!r}")
        # MODE VOCABULARY. The file and `--mpc-mode` accept {none, mpc,
        # dobmpc, mpc_tuned, dobmpc_tuned, pid}; the trajectory panel's LOW
        # combo (MpcWorker.MODES) also offers mpcc / dobmpcc — a pre-existing
        # asymmetry, unchanged. "none" joined on 2026-09-11 (operator
        # request: LOW level None = TELEOP; `--policy-observe` is its launch
        # alias) — it is NOT a member of HwDobMpc.MODES and must not become
        # one (that setter raises for non-members).
        if cfg.mode not in ("none", "mpc", "dobmpc", "mpc_tuned",
                            "dobmpc_tuned", "pid", "rl", "rl_pwm"):
            raise ValueError(
                f"mode must be none|mpc|dobmpc|mpc_tuned|dobmpc_tuned|pid|rl|"
                f"rl_pwm, got {cfg.mode!r}")
        return cfg


def yaw_from_R(R: np.ndarray) -> float:
    """ZYX psi, the same extraction dobmpc.frames._euler_from_R uses."""
    return math.atan2(float(R[1, 0]), float(R[0, 0]))
