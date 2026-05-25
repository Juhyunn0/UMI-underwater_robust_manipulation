CLAUDE.md — UMI Underwater Robust Control: Refractive TagSLAM
Context file for Claude Code. Summarizes the project, the validated diagnosis chain, current code state, and the exact next step. Read this fully before changing anything.
1. Project goal
Long-term: energy-efficient, robust underwater manipulation in dynamic ocean currents (follow-up to "UMI-Underwater: Learning Underwater Manipulation without Underwater Teleoperation").
This repo's immediate scope: build a relative pose/velocity estimator whose ground truth comes from AprilTags on the pool floor. Before the estimator can be trusted, the AprilTag-based ground truth itself must be metrically correct. That is what this codebase (src/zed2_underwater_tagslam.py) currently does: single-camera AprilTag SLAM (ZED2 left image → AprilTag → PnP → GTSAM iSAM2).
Physical setup: ZED2 camera in air, ~0.17 m above the water surface, looking down through the (flat, calm) water surface at AprilTags tiled across the pool floor. Pool ≈ 4.877 m × 2.438 m × 1.143 m (45 in) deep. Water level varies slightly day to day; it is set in config. Camera is NOT yet in a waterproof housing (that is future work).
2. Code layout
src/zed2_underwater_tagslam.py — main pipeline (latest is the file the user uploaded as zed2_underwater_tagslam__4_.py; treat the in-repo src/ file as the source of truth).
tools/nadir_ruler_test.py — standalone raw-PnP ruler diagnostic (read-only; reuses the main module's intrinsics/detector/solvePnP).
config/config.yaml — pool geometry, water surface, tag size, etc.
Key symbols in the main file: detect_observations, estimate_tag_observation, project_refractive, refine_refractive_pose_lm (reference-only now), interface_plane_from_camera_pose, fallback_near_nadir_interface_from_raw_pose, water_up_axis_world, refractive_pose_from_raw_pnp, corrected_underwater_tvec, TagSlamBackend (_initialize_when_anchor_visible, _update_incremental, _apply_floor_plane_priors, camera_pose_hint_for_measurement), run_refractive_self_test, write_interactive_trajectory_html.
3. Validated diagnosis chain (do NOT re-litigate these — they are settled)
Tag size was a 2× error. Config tags.tag_size_m is visualization-only; SLAM uses the resolved tag size. Real printed AprilTag black-square edge = 0.170 m (ruler-measured). A code change now makes the config tag size drive the SLAM tag size (precedence: explicit CLI --tag-size > config > built-in default; startup logs SLAM tag size: ... (source: ...)).
Residual after tag-size fix is pure single flat-interface refraction. Confirmed by a near-nadir ruler test + first-principles physics:
Camera in air h_air ≈ 0.17 m, water column d_water, n_water ≈ 1.333.
Predicted R = (h_air + d_water/n) / (h_air + d_water).
Measured (correct tag size 0.170, mode none): R ≈ 0.80 at d_water ≈ 1.0 m vs predicted ≈ 0.79. Intrinsics are fine (ZED factory fx=fy≈534.88 HD720; no hidden extra scale, else R would deviate from physics).
A single global --water-scale is WRONG because the correction is geometry-dependent (air gap, water column, off-nadir angle). The old --water-scale 3.6 was tag-size-2× × refraction × over-tuning; it warps the floor and amplifies tracking noise.

Refractive PnP implemented and validated. New --water-correction-mode refractive (modes: none/scalar/trust-region/ refractive; refractive is the one to use for real runs). Flat air→water interface + Snell, water surface is config-driven (water.surface_height_m, n_water, n_air), per-frame air gap derived from the tracked camera pose.
M5 synthetic self-test passes (--refractive-self-test): rms ≈ 1e-5 px, trans ≈ 0 mm, rot ≈ 2e-4 deg.
M6 real data: D_true ≈ 1.2 m → none ≈ 1.0 m (R ≈ 0.83), refractive ≈ 1.2 m (R ≈ 1.0). PASS.

Refractive solver made real-time. The first refractive implementation was a per-tag finite-difference LM with a 48-iter inner bisection → fps collapsed (~10 → ~1 when tags appear, scaling with per-frame tag count; note: a normal frame shows at most ~10 tags, NOT 20 — the overlay "N observed / M in graph" is cumulative, not per-frame). Optimized to a fixed-point cv2.solvePnP loop + Newton refraction point + per-tag warm-start + batched correction. The old LM is kept only for --refractive-regression-check. Synthetic benchmark ≈ 41× speedup.
--refractive-self-test: rms ≈ 2.1e-5 px (unchanged-level). PASS.
--refractive-regression-check: max_trans_delta ≈ 0.0003 mm, max_rot_delta ≈ 0.00037 deg (bounds 0.1 mm / 0.01 deg). PASS.
New knobs: --refractive-max-iterations, --refractive-convergence-tol-m, --refractive-convergence-tol-deg, --refractive-ray-max-iterations, --refractive-ray-tol, --refractive-regression-check, --refractive-benchmark. New latency log line: Refractive PnP latency: last=... ms/frame (... tags), avg=..., ms/tag, fps=....

4. Open problem (the current focus)
After the refractive fix, two original problems remain partially:
Problem 1 (floor not flat): the reconstructed tag floor is a tilted "ramp" — only the anchor tag (tag 1) sits on the floor; every other tag rises with distance from it. Tag Z span did NOT shrink (refractive even showed ~64.9 cm vs scalar ~49.9 cm). Scalar additionally overshoots the pool box (constant scale inflates distant tags); refractive no longer overshoots (scale is correct) but the ramp shape persists.
Problem 2 (trajectory noise): residual per-frame jitter (deferred; to be addressed only after Problem 1 and after a clean refractive run).
Root cause of the ramp (diagnosed, not yet fixed): the world is hard-pinned to a single, possibly-oblique tag-1 observation (_initialize_when_anchor_visible: world_T_camera = anchor_obs.camera_T_tag.inverse()), and both the floor co-planarity prior (_apply_floor_plane_priors) and the refractive interface normal (interface_plane_from_camera_pose / fallback_near_nadir_interface_from_raw_pose / water_up_axis_world) are referenced to that tilted anchor frame, not true gravity. Compounded by near-stationary capture (distant tags weakly constrained). Refraction was one bias source; removing it exposed this structural one. Robust kernel / floor prior cannot fix it (systematic, not sparse outliers; prior re-levels a plane, it cannot un-tilt a ramp).
Note: pinning tag 0 and tag 1 does NOT help — they are physically adjacent, so the baseline is too short to constrain tilt (≈ same as one pin).
5. NEXT STEP — IMU gravity integration (decided, prompt ready)
Decision: skip further verification runs; go straight to using the ZED2 built-in IMU gravity vector as the single source of "up", read frame-synchronously (one IMU read per camera frame, NO separate IMU thread / sample-rate handling — smoothing window is in camera frames, so the IMU operating frequency is irrelevant by design).
Use IMU gravity to:
G2: make the refractive interface normal = true gravity (not anchor frame).
G3: reference the floor prior plane normal to true gravity.
G4: at init, gravity-align the world frame's roll/pitch only (anchor still defines position and yaw — IMU cannot observe yaw/heading).
Hard constraints for the IMU change:
Do NOT change the validated refraction physics, the fast solver, the config-driven water surface, modes, or CSV schema.
Use IMU gravity (roll/pitch) only; never IMU yaw/heading.
--use-imu-gravity (default ON); OFF or IMU-absent ⇒ byte-for-byte current behavior (safe revert). Frame-synchronous read; never block the loop.
--refractive-self-test and --refractive-regression-check must still pass unchanged (synthetic; must be unaffected). M6 must stay: none ≈ 0.83·D_true, refractive ≈ D_true.
The full IMU implementation prompt (G1 frame-synchronous variant + G2/G3/G4 + constraints + acceptance) has been written in chat; implement that. Acceptance: on the same pool run, --use-imu-gravity ON vs OFF → Tag Z span drops substantially and the front-view tag sheet is flat (no ramp); add a debug log of the gravity vector and its angle vs anchor-frame up (quantifies the anchor tilt).
6. Standard validation sequence after ANY refractive/IMU change
python3 src/zed2_underwater_tagslam.py --refractive-self-test → rms ~1e-5 px, "passed".
python3 src/zed2_underwater_tagslam.py --refractive-regression-check → max_trans_delta < 0.1 mm, max_rot_delta < 0.01 deg, "passed".
M6 real data (near-nadir static, tape-measure D_true in meters, set config water.surface_height_m to the session water column): compare per-tag ID x z= label (NOT the "Global camera pose" line) for --water-correction-mode none vs refractive: none ≈ 0.83·D_true, refractive ≈ D_true.
Real moving pool run: compare scalar vs refractive (and IMU ON vs OFF): fps + latency log, Tag Z span (Problem 1), trajectory noise (Problem 2).
7. Conventions / gotchas
Run shell commands on ONE line (multi-line \ continuations have broken before due to trailing spaces).
Per-tag ID <id> z= overlay label = direct single-tag measurement (use for M6). "Global camera pose from tag1" line = graph-fused (anchor/floor-prior contaminated) — do NOT use it for M6.
Config tags.tag_size_m historically was viz-only; the resolved SLAM tag size now logs its source at startup — always check it is 0.170 (source config/cli).
Pool depth is 1.143 m (45 in), not 3.048 m (older config was wrong); fix config depth fields if not already corrected.
ZED depth/stereo is NOT used for pose (left mono only); depth_mode is not disabled so the SDK still computes unused depth — optional perf cleanup (sl.DEPTH_MODE.NONE), no effect on results.
Camera is in air through a flat calm surface; ZED stereo depth would itself be refraction-distorted, so the AprilTag refractive-PnP path is the metrically correct one for this setup.
8. Tone / working agreement
Diagnose with code-grounded evidence; do not force results to fit a prior hypothesis. Distinguish bias (constant offset) vs noise (jitter).
Make fixes configurable and reversible (default-on new mode, OFF = old behavior) and always provide acceptance checks + a self-test/regression guard before trusting accuracy.
Keep the validated chain in §3 intact; new work must not regress self-test / regression / M6.