# KNOWN_ISSUES — things not fixed yet

> **English translation.** Source of truth: [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) (Korean). Snapshot of 2026-10-12, source blob `52553fa4c1`; later edits to the original are not reflected here.
>
> Provenance tags are kept as in the original: `[측정 …]` measured (artifact cited) · `[유도 …]` derived · `[예측 …]` predicted, not measured · `[스펙 …]` vendor spec · `실측` measured · `미실측` not measured · `산출물 없음` no artifact found · `예시/유도값` example or derived value.

> A living list of bugs, traps and deferred items found during Claude Code sessions but **not yet fixed in
> code**. Rule: **when you fix one, delete its entry** (git history keeps the record of the fix).
> When you find a new one, add it with the date. Entry format: Symptom → Workaround → Proper fix.

## ⚠️ Operational safety

### Since the MPC_Tuned profiles were introduced, **the peg buoyancy trim lives in two places** — fix only hw_mpc.yaml and mpc_tuned stays at −12 N (2026-10-04)
- **Symptom**: while it holds a profile, mpc_tuned (·_7d) flies on `apply.vehicle_net_buoyancy_n` (default `20261004_default` =
  **−12 N, peg mounted**), while PID·mpc fly on hw_mpc.yaml `vehicle_net_buoyancy_n`. If you take the peg off and change only
  hw_mpc.yaml to 0.1, **PID is right but mpc_tuned still outputs a feedforward that pushes up by the
  peg's weight (~12 N)** — per the hw_mpc.yaml comment, it holds about 6 cm high [유도: that comment, 미실측].
  For the same reason, the hw_mpc.yaml `mpc_tuned:` block and `policy.q_scale/along_scale` have no effect on mpc_tuned either.
- **Workaround**: when the two values differ, the tune row shows a **WARN** ("vehicle_net_buoyancy_n … ≠ hw_mpc.yaml … — is
  the payload the same?") (applied from the 2026-10-04 safety review) — it is only a warning and does not block. When you
  change whether the peg is mounted, pick a profile saved in that state, or launch with `--mpc-tune none`. Write the payload
  into the profile name (e.g. `..._nopeg`). It is recorded in meta `controller.mpc_tune.warnings`·`heave_trim`.
- **Proper fix**: move the payload (peg or no peg) out of the profile into **session state** (a panel toggle + a value in
  one place) so that PID·mpc·mpc_tuned read the same trim, or require confirmation at START when the profile's trim differs
  from hw_mpc.yaml. The operator chose "apply the trim through the profile too" on 2026-10-04, so changing this needs an
  operator decision.
- **Related test failure (pre-existing)**: `test_path_cost.py::test_net_buoyancy_is_validated_and_recorded` asserts that
  `|vehicle_net_buoyancy_n| ≤ 2` and that it is the same value as land_dp.yaml — it has kept failing since the 2026-10-01
  peg value of −12 N (land_dp.yaml has 0.1). Fixing the test assertion to match the peg state, or separating the payload as
  above, resolves it as well.

### The paper-style EKF (`rl_ekf.py`, panel EST button) is **unverified on hardware** — "paper = robot_localization" is an inference, and the noise and input choices are [예측] (2026-10-02)
- **Status**: `rov_gui/control/rl_ekf.py` is a port of robot_localization `ekf_localization_node` (noetic @ 9ef26a57), and
  it is on by default, in shadow. `test_rl_ekf.py` 38/38, `demo_e2e_ekf.py` (real window), the synthetic comparison
  `sim_ekf_compare.py` and the cross-check against an independent re-transcription of r_l (state difference ≤ 5.5e-14) pin
  only the **logic**. 0 water runs. Safety audit FAIL → fixes → 2 re-audits, PASS_WITH_FIXES (see README).
- **What the comparison means**: the paper's launch file is not public. This EKF is not "the paper's EKF" but "r_l
  defaults + this repo's choices (tag x,y,yaw / IMU roll,pitch,angular velocity,acceleration / depth z / DVL vx,vy, mavros
  IMU noise)". Do not cite the results as a reproduction of the paper.
- **With r_l's default lag handling (smooth_lagged_data false), the position lags behind by velocity × tag latency**
  [예측: synthetic `rov_gui/tests/fixtures/sim_ekf_compare_20261002.txt`, 33 mm rms at 0.3 m/s]. Flying on the EST EKF puts
  this lag into the control loop. For comparison, also try flying `smooth_lagged_data: true, history_length_s: 1.0` (or
  `ekf_replay --set`).
- **The mavros IMU noise trusts the accelerometer almost completely** (stdev 0.0003): a 0.1 m/s² bias with no DVL gives a
  velocity error of 15 cm/s [예측: same file]. The bias of the vehicle's SCALED_IMU2 is not measured. The velocity-divergence
  interlock (15 cm/s, 0.25 s) ends the run, but before that it gets into the D term·EAOB.
- **The IMU·depth stamps are host arrival times** (not the ATTITUDE measurement time — the same limitation as the
  assembler). The tens of ms of lateness are not corrected.
- **On tag dropout (station hold) the EKF run ends** — the bridge does not carry EKF runs (a critical fix from the safety
  audit). A negatively buoyant vehicle sinks after disengage. Bridging with IMU+DVL as the paper does needs a separate
  design·audit.
- **The "here" target is set from the tag state** (engage hold, mission origin, replay fallback). Engage/switch requires a
  5 cm / 3° agreement, but a new origin in the middle of a mission becomes a step input of the EKF−tag difference at that
  moment (up to 15 cm / 10°).
- **Policy (DP) missions**: the policy's proprio·anchor keep being computed from the tag state and can disagree at the cm
  level with the follower flying on the EKF (`_policy_bridged` does not treat EKF flight as bridged).
- **EKF flight has no separate authority cap** (something like imu_dr's `axis_cap_dr`). All interlock limits are [예측] —
  in particular the 15 cm/s velocity limit is the same value as the yaml's own prediction (bias 0.1, no DVL → 15 cm/s), so
  it can trip even at a normal bias, and the velocity difference contains heave (the FD along the down-looking camera's
  optical axis = the axis where PnP is worst) (re-audit 2026-10-02). Before going in the water, run recorded runs through
  `ekf_replay`, compute the per-channel distribution of `_ekf_vs_tag`, and set the limits from that artifact. A false
  positive errs on the safe side (disengage) but loses depth hold.
- **Even if the accelerometer dies, the EKF stays ok for up to 1 s** (between the attach rule's 0.3 s and the freshness
  1.0 s — a value chosen so that the gate does not flicker when SCALED_IMU2 runs at 2–3 Hz). Meanwhile the acceleration
  state is held at its last value and integrated into velocity, and the only thing that catches it is the velocity gate
  (LP 0.5 s) (a 0.3 m/s error in ~0.6 s [유도]). Look at the measured SCALED_IMU2 rate and adjust `acc_s`.
- **The DVL's per-sample verdicts are not applied to the EKF**: the EKF looks only at the per-run eligibility
  (`_dvl_run_ok` / `refusal`) and the trip, not at the innovation hold·stale·lock verdicts the assembler makes every tick
  (only `valid`, like r_l). A false lock under 0.15 m/s gets in until the trip (~1 s sustained). Latent, since the DVL is
  not mounted. The real device's covariance diagonal at rest is ≈ 8e-7 m²/s² (σ ≈ 0.9 mm/s) [측정:
  data/20261011/1011_155235_dvl_probe/station_client_live.json first_sample] and the `dvl_rejection_threshold` default is
  None, so in opt-out (mount values set + `require_proven: false`) the ~0 velocity of a DVL that is not attached to the
  vehicle, or is in false lock, is fused with **a very high weight**.
- **Transient right after engage**: `asm.reset()` in `set_engaged` restarts the assembler velocity from 0 for ~2 frames,
  so the velocity channel reads |nu_ekf| (it trips only at an engage velocity of ~0.37 m/s or more [유도]); with z_source
  pressure, the z offset recalibration puts the EKF z off for 1–2 samples.
- **replay is a cold start** (from the first CSV row), and because CSV `t` is rounded to 0.1 ms, a row occasionally
  differs — compare variants against "replay (live settings)".
- **Proper fix**: fly the first water session in shadow and check the `ekf_*` columns and `est` → measure the SCALED_IMU2
  at-rest bias → compare smoothing on/off·IMU noise variants with `ekf_replay` → after that, a short station hold on the
  EST EKF.

### The entire DVL path (`--dvl`) is **unverified on hardware** — mount values null, all thresholds [예측], latency and noise not measured (2026-10-01)
- **Status**: `rov_gui/control/dvl.py` · `backends/dvl.py` · the station bridge's `dvl` rung were written **before** the
  DVL A50 was mounted, from nothing but the protocol document and a capture from the vendor's public demo unit
  (`rov_gui/tests/fixtures/dvl_a50_demo_capture_20261001.jsonl`).
  `rov_gui/tests/test_dvl.py` 96/96, `demo_e2e_dvl.py` and the synthetic timing simulation `sim_dvl_monitors.py` only pin
  the logic; they are not evidence about this vehicle·this pool.
  It is OFF by default (`hw_mpc.yaml dvl.enabled: false`), so unless it is turned on, the estimation path is bit-for-bit
  identical to before.
- **Unknowns**: (1) The mount `mount_rpy_deg` / `mount_t_flu` — null; until they are measured, IN is refused. (2) `stale_s 0.6`,
  `fom_max 0.05`, `reentry_s 0.5`, `a_max_m_s2 2.0`, `innov_max_m_s 0.15`, `agree_*`, `station_bridge.dvl_hold_s 10`
  / `dvl_max_speed_m_s 0.3` / `dvl_max_disp_m 0.5` — all [예측]. (3) The actual latency and velocity noise. In the first
  device read on 2026-10-11 (at rest, in a container of water, altitude 0.083 m), `tot − tov` was 80.5–113.9 ms [측정: data/20261011/1011_155235_dvl_probe/
  dvl_lines_A.jsonl] — but the `time` field matches the **next** tov interval, and tot[i] > tov[i+1] in every consecutive
  pair, so tov may be a stamp that is one ping (~80 ms) late (then the actual processing latency is 3–30 ms and
  `DeviceClock`'s t_valid·age are ~80 ms too large) [유도: same file, analysis/ in that folder]. Tell these apart in a
  motion test with the proof's `lag_s` (relative to the tags). The noise while moving is not measured (only fom ≈ 1 mm/s
  at rest). (4) Whether bottom lock holds in a shallow pool (altitude < 1 m), over the tag-mat floor, with wall reflections
  (multipath) and thruster bubbles, and whether false lock is filtered out by `velocity_valid`/fom — the at-rest capture in
  the container passed the sample gate 150/150 (a wall·multipath lock is also "good data" at the sample level; only the
  proof makes the call).
  (5) **With the hull on the floor, the altitude is < 5 cm and lock is lost** — this overlaps with the problem of policy
  runs sitting on the floor (2026-09-14). (6) The device's
  `mounting_rotation_offset` is **180** (2026-10-11, api/config.json in the same folder) — the device outputs velocity·DR
  rotated 180° about the Z axis. The README "Mount" section's "LED forward → `mount_rpy_deg: [0, 0, 0]`" presupposes a
  device offset of 0, so an LED-forward mount + device 180 + that recipe flips **both** the surge and sway signs, and the
  config-stage check (duplicate-yaw error) cannot catch it (only the proof catches it).
  The "x forward = the LED edge" in the `state.py` DvlSample docstring is also wrong under this setting. Step 2 of the
  README "Turning it on in the pool for the first time" also says to check that `mounting_rotation_offset` in the log is
  **0** (station convention = rotation goes in `mount_rpy_deg`). But the BlueOS extension → ArduSub path uses only the
  device offset, so for an LED-backward mount, device 180 + `mount_rpy_deg` yaw 0 gets both paths right
  (the station raises an error only when neither is 0 — `workers.py` `on_dvl_info`). For an LED-forward mount,
  device 0 + yaw 0 keeps both simple.
  The device does not correct for a roll/pitch mount [스펙].
  The speed of sound defaults to 1475 m/s (≈ 17.7 °C fresh water) — distance·velocity scale by c_set/c_true, so set it to
  the value for the pool water temperature [유도].
  (7) The behaviour when `rl` / `rl_pwm` are fed DVL velocity (outside the training distribution).
  (8) The BlueOS DVL extension uses the same DVL to **aid ArduSub's EKF3** (VISION_POSITION_DELTA 12 Hz, "EKF3 IMU0 started
  relative aiding" — mavlink2rest_*.json in the same folder). Until the DVL is fixed to the vehicle·aligned, ArduSub's
  position·altitude-based modes (POSHOLD, panel DEPTH = ALT_HOLD) use a velocity that has nothing to do with the vehicle.
  Turning the extension off may make prearm refuse with "VisOdom: not healthy" etc. [예측: SYS_STATUS VISION_POSITION bit —
  AP_Arming unconfirmed].
- **Design limitations (confirmed over 6 reviews, 2026-10-02; unresolved)**:
  - **A long false lock trips, and that trip, too, requires a fresh proof.** +14 cm/s × 1 s or +10 cm/s × 2 s trips
    (whether once or every 10 s); +10 cm/s × 1 s, +14 cm/s × 0.5 s, and 4 reports or fewer do not trip [예측: `rov_gui/tests/fixtures/
    sim_dvl_monitors_20261002.txt` A5b]. This is intended behaviour (you cannot fly on a DVL that reports nonexistent
    motion for 1 s). However, such a trip discards the proof and demands a new proof pass even though it has nothing to
    do with the mount — if false locks are frequent near the pool walls, you have to prove the mount again every time. To
    close this, split the trip in two so that a "DVL-only motion" difference trip only sets the latch and keeps the proof
    (not implemented). The boundary values are all [예측].
  - **Glitches and noise take the DVL out with a hold instead of a trip — if they are frequent, it flies without the DVL
    for that much of the time.** With a 4-report glitch every 3 s, the ticks flown on the DVL at rest are 39–60 %; with
    2.5 cm/s DVL noise, 48 % at rest; at 3 cm/s or more it is almost never proven and stays OUT [예측:
    same output file A5]. Each time a hold is set and released, the velocity source switches (the difference between the
    two values is at the noise level). There is no latch.
  - **Vertical axis unproven → heave velocity does not use the DVL.** Because the proof looks only at surge·sway. To
    close this, add a 1-D regression of baro depth rate (or tag z differencing) ↔ DVL vz, and replace `nu[2]` only after
    confirming sign·scale with an up-down pass.
  - **Neither the proof nor the monitors can see the mount tilt (roll/pitch) or the lever arm.** A pitch error e only
    scales one row by cos e, so it passes up to ~37° [유도], and a lever-arm error δr enters as ω × δr and shows up only
    during yaw. `mount_t_flu: [0, 0, 0]` also passes as "measured". The tape measure is the only defence. The spurious
    velocity a wrong arm produces is motion that only the DVL sees, so the "DVL-only motion is not proof of the mount" rule
    (`COHERENT_MIN`) deliberately ignores it: in the synthetic simulation, during a 0.3 rad/s yaw sweep, arm errors of 3 cm
    and 10 cm both gave trip 0/36 [예측: `rov_gui/tests/fixtures/sim_dvl_monitors_20261002.txt` A4] — with a 10 cm error,
    a spurious sway of 3 cm/s during yaw goes straight into the estimator [유도: 0.3 rad/s × 0.1 m]. If r_z is added as a
    regressor in an in-place yaw pass, the two horizontal arm components are identifiable (not implemented).
  - **The proof fit ignores DVL noise.** When the weak-axis excitation is near the threshold (2.5 cm/s) and the DVL noise
    is ≥ 1.25 cm/s, the slope shrinks below 0.75 and a correct mount can be refuted as "scale 1.33×" [유도]. In the
    synthetic simulation, DVL noise of 1–2 cm/s is no problem; at 3 cm/s the proof almost never succeeds (1 run out of 6,
    at rest·5 cm/s motion), and at 5 cm/s it is 0 [예측: same
    output file A5] — a DVL that noisy effectively never gets IN (the outcome is safe, but the reason shown is "scale",
    which invites misunderstanding). Subtracting the noise variance using the covariance the device sends (parsed only,
    unused) or the differences between consecutive DVL samples would solve it structurally. For now the mitigation is
    "move faster, in a straight line". If the noise drags the mapping below 0.8 (`gain_min` <
    `COHERENT_MIN`), it is classified as "says nothing about the mount" and gets a hold instead of a trip — **there is no
    monitor that catches DVL noise itself with a trip**, and the only things filtering it are `fom_max`, a [예측] value,
    and the hold. For the same reason, **a pure scale error of 1.25× or more** (e.g. a wrong speed-of-sound setting) is
    also a hold, not a trip.
  - **Both the proof and the monitors are sensitive to the stamp offset δ between the two instruments.** Because both pair
    tag differencing with DVL velocity by stamp. In motion where the velocity vector rotates at ω (an ellipse in which
    surge·sway have a phase difference), δ becomes an apparent rotation ω·δ [유도].
    In the synthetic simulation, a 50 ms tag-stamp error + 0.5–1.5 Hz elliptical oscillation **tripped a correct mount**
    (50 ms 2–3/36, 100 ms 8/36, 150 ms 8/36; 0 in straight-line motion) [예측: same output file A4]. Mitigation: do the
    proof pass as two straight lines, and if `lag_s` in the `mount PROVEN` log (an estimated lower bound on δ — biased
    toward 0) exceeds tens of ms, fix the stamps first, before IN. Which one is wrong — the camera `t_capture` or the DVL
    `time_of_validity` mapping (DeviceClock's minimum-latency assumption) — cannot be told apart from `lag_s` alone.
  - **Wrong mounts that the in-flight monitors cannot catch under opt-out (`require_proven: false`)**: ① every wrong
    mount in station keeping (1–2 cm/s), and a reversed axis moving at under ~2 cm/s rms (with 2 cm/s DVL noise, the
    alignment hold only brings the ticks flown on the wrong DVL down to 66–86 %), ② **a reversed axis under straight-line
    motion within 14.5° of the other axis — regardless of speed** (20+3.5 cm/s straight line and 30+7.5 cm/s in phase at
    0.5 Hz are both NEVER; in straight-line motion an axis reversal is indistinguishable from a rotation of 2× the
    motion-direction angle [유도]), ③ a rotation of 30° or less under ~15 cm/s, and a 2× scale under ~8 cm/s [예측: the
    NEVER cells of B in the same output file; ③ is 유도 from the 8 cm/s difference-vector threshold]. With the default
    (`true`), all of them are caught by the up-front proof (for ②, the proof never succeeds at all). The velocity error in
    ② is 2× the reversed axis's velocity, so it is not small. That simulation is open loop, so it is not known whether, in
    closed loop, the wrong-sign feedback grows the motion, breaks the straightness and shows up in the monitors. opt-out
    is a convenience for skipping the proof pass, not a safety mechanism. Even without a trip, if the fit refutes it, the
    DVL drops out without a latch (`DVL held back (the fit refutes it)`): in the same table, 2× scale·6 cm/s oscillation
    has 16 % of ticks flown on the wrong DVL. Most NEVER cells are motion on which the fit cannot even make a call (slow or
    in one direction), so they stay at 98 %.
  - **Even a mount that passed the proof can trip once it picks up speed.** The proof tolerance is relative (0.25 of the
    speed), the difference-vector threshold is absolute (8 cm/s). A mount proven with a residual yaw of 12° or a scale of
    1.2× trips at 0.4 m/s and above (3/4, 4/4); 5° gives 0 up to 0.5 m/s
    [예측: same output file A6]. Mitigation: if the warn right after `mount PROVEN` (`… above X m/s that alone trips the difference
    monitor … Correct dvl.mount_rpy_deg yaw by +D deg`) appears, correct the mount yaw by that amount and restart.
  - **The innovation hold is released only when a new camera frame agrees.** So if the vehicle is kicked hard (~1 m/s²)
    while the tags are visible, even an honest DVL diverges from the delayed tag velocity by more than 15 cm/s and a hold
    is set; if the tags disappear at that moment, the existing IMU/freeze ladder takes over instead of the DVL rung — the
    same behaviour as when there is no DVL (a false lock and a real disturbance cannot be told apart at that moment).
    Of the same kind: the first tick right after the tag velocity has been absent for more than 0.25 s always flies on
    the tag velocity (`DVL unchecked`). And on the tick where the hold is released, the samples the gate accepted go in
    over the fix age — if the tags are 0.3 s late and that sample is 0.14 m/s, that is 4 cm in one tick
    [유도: 0.14 × 0.30], gone from the next tick on. Samples the gate rejected are left out of the fix-age integration
    (the fix from audit round 6).
  - **The later the tags, the wider the innovation gate** — 0.15 + a_max × (fix age − 0.25 s): 0.25 m/s if the tags are
    0.3 s late, 0.45 m/s at 0.4 s [유도]. A false lock smaller than that passes the gate even while the tags are visible,
    and the difference monitor catches it only ~1 s later. If the tags disappear in the meantime, the DVL rung carries it
    up to the cap (0.5 m) and then returns to the fix.
    For the same reason, **a false lock that starts while the fix is going stale (the last 0.25–0.45 s)** also passes the
    widened gate and the DVL rung carries it up to the cap (R5-W2 of the fifth audit; on a vehicle at rest an estimated
    0.49 m, about 2 s). In both cases the cap bounds it.
    The `tag_stale_s` comment in the engage block of `config/hw_mpc.yaml` gives tag age p50 0.26 s / p95 0.40 s in
    dobmpc (2026-08-14; that comment has no artifact path) — if those values hold, the first case is within the actual
    operating range.
  - **The alignment monitor does not judge directions that make up less than 7 % of the motion** (`ALIGN_MIN_SHARE`).
    A regression along such a direction was shaken by a single glitch and tripped a correct mount (0.3 m/s round trip +
    4–6-report surge glitch: 20/128 at 0, 0/128 at 0.07). The cost, under opt-out, is a reversed axis in **a small
    secondary motion on top of a fast primary motion** — not only straight-line motion: with a sway-reversed mount, mixing
    a 3 cm/s sway oscillation into a 0.15–0.3 m/s surge, what tripped in 2.0–3.5 s at 0 flies 98 % of its ticks on the
    wrong DVL for 30 s at 0.07, with neither a trip nor a hold; with a 4–8 cm/s sway, 1.2–1.9 s → 5.2 s (8 cm/s + round
    trip: 1.0 s for both). Then only the fit (60 pairs + 15 refutations) is left [예측: synthetic, audit round 6
    experiment — the artifact is not in the repo]. With the default (`require_proven:
    true`), the up-front proof refutes such a mount.
  - **On the first tick after the tag state has been cut off for 3–4 ticks and comes back, the DVL is accepted with
    widened limits.** A false lock acquired in the meantime can get in for one tick (50 ms) — the fresh rejection on the
    next tick sets the innovation hold. Preventing this would require making the first tick after a gap "unchecked", but
    then, when the tags are slow, like 4 Hz·0.25 s latency, the velocity source would switch once per fix period
    (a finding of the third audit). The effect of one tick is small (estimated 3 cm, a D-term kick of 0.04 m/s [유도]), so
    it is left as is.
  - **If the device clock steps backward by less than 10 s, the sample times are wrong by that much for up to 30 s**
    (`DeviceClock` re-syncs immediately only on steps of more than 10 s). A step of more than 0.6 s drops out as "stale"
    and is safe; if smaller than that, the pairing goes off and a false-alarm trip can occur. This only happens if the
    device's NTP steps the clock during operation, so it is left as is.
  - **(Unrelated to the DVL, pre-existing behaviour) The station bridge restarts the hold clock on a blind tick.** If,
    while there are no tags, even the autopilot ATTITUDE or the barometer is one tick late, the ladder treats that tick as
    "end of dropout" and from the next tick counts `imu_hold_s` again from the start — when such ticks are mixed in, the
    horizontal hold lasts longer than the configured value (with `imu_hold_s` 3.0, one blind tick at 2.5 s → COAST at
    5.60 s instead of 3.05 s [유도: ladder arithmetic, confirmed by running it in the re-audit]). During the DVL work it was
    once changed to "keep the clock running", then reverted, because that made the ladder of runs without a DVL differ
    from the committed code. Whether to fix it is a separate decision.
  - **The DVL rung's hold time·integration dt are based on tick counts (nominal 0.05 s)**, so when ticks slip the actual
    hold is longer (the same property as the existing ladder).
- **Workaround**: the first session flies only in `DVL OUT` (shadow). Step 5 of the README "Turning it on in the pool for
  the first time".
- **Proper way to close it**: measure the latency·noise·fom distributions from the CSV of one shadow run (`dvl_u/v` vs
  `fd_u/v`, `dvl_age_s`, `dvl_fom`, `dvl_alt_m`) and `<stem>_dvl.jsonl` and replace the [예측] values, then cover the
  tags during a station hold and measure `dvl_hold_s` from the `DVL dead reckoning was N cm off` log (it is printed only
  when the dropout ended on the DVL rung). Close each of the "Design limitations" above separately. When all of them are
  closed, delete this entry.

### Of the vehicle's STATUSTEXTs, **EMERGENCY (severity 0) does not show up on the station** — `int(sev or 6)` (2026-10-11)
- **Symptom**: `rov_gui/backends/hardware.py:1870` `elif text and int(sev or 6) <= 4:` — severity 0 is falsy, so it is
  turned into 6 (INFO) and dropped. ALERT(1)~WARNING(4) do show up. What gets dropped: the BlueOS DVL extension's "DVL is
  too hot (T C). Please cool it down." (every 30 s while the DVL is above 45 °C, severity EMERGENCY, header 1/1 [스펙:
  extension master `dvl.py:418-431`, helper `:332` — copy at data/20261011/1011_155235_dvl_probe/analysis/wf/src_master/])
  and ArduSub's own EMERGENCY messages. leak/pressure/config are classified first, in the branch above, so they are not
  affected. There is a verifier report that on the mavlink2rest (REST) transport severity is a dict, so `int(dict)` fails
  the tick with a TypeError (not reproduced).
- **Workaround**: watch for DVL overheating on the SENSORS DVL row (status bit0 → `HIGH TEMPERATURE`, only while there is
  lock) and on the QGC/BlueOS screens.
- **Proper fix**: `int(6 if sev is None else sev)` + normalization of the REST path, and a regression test that a severity
  0 STATUSTEXT shows up as a warn.

### The ATTITUDE rate request is sent only **1 time, at connect** — if an ArduSub reboot or the BlueOS DVL extension's 5 Hz request overrides it, it is 5–10 Hz for the rest of that session (2026-10-11)
- **Symptom**: in `c3_camera/mavlink_log.py` start, the station sends `_request_rates()` **once**, when it receives the
  first message (ATTITUDE `--mavlink-rate` default 200 Hz; if there is no telemetry after start, it never sends it at
  all). When ArduSub reboots (battery swap etc.) this request is gone, and the BlueOS DVL extension
  requests ATTITUDE·VFR_HUD at **5 Hz** when it looks for the DVL — in the 2026-10-11 boot, ATTITUDE was 5.0 Hz after the
  request and ≈ 9.6 Hz before it, AHRS2 10 Hz [측정: data/20261011/1011_155235_dvl_probe/mavlink2rest_225506Z.json,
  mavlink2rest_225619Z.json counter difference]. If the extension's request goes out later than the station's request
  (DVL found late, container restart), it overrides the 200 Hz. With `engage.imu_stale_s 0.3` at 5 Hz the margin is
  100 ms, so a single lost ATTITUDE means imu stale [유도]. The health `Compass` row is max(ATTITUDE, AHRS2), so AHRS2 at
  10 Hz masks this degradation. The actual ATTITUDE rate over the station link has not been measured yet.
- **Workaround**: if you rebooted the vehicle with the station running, or connected the DVL late, restart the station too
  (that way the 200 Hz request is the last one).
- **Proper fix**: monitor the measured ATTITUDE Hz and re-request when it is well below the requested value (periodic
  re-request), and show ATTITUDE separately in health.

### LOW `RL_PWM_7D` (RL that tracks roll/pitch with pulses): the policy `att7pwm_s0` is enabled while short of the sim promotion criteria, and **2 water runs (2026-10-03 DP following, 2026-10-05 station hold), both failed** — vertical co-contraction, sinking, and a high-gain response to tag jumps (2026-10-03, updated 10-05)
- **What**: `rl_pwm_7d` = the same transport as RL_PWM (8 pulses, Lua override, MANUAL, a gate read from the vehicle, cap `rl_pwm.pwm_cap`) with a
  policy trained on roll/pitch references (`HwRlPwm7d`). Training: RL_controller `att7pwm_s0..2`, preset `traj10att`, **still water** (no waves or
  current — the 0–4 N tether-emulating external force stays as in the traj10 recipe), 34–35 min each [측정: `RL_controller/logs/train_att7pwm_s*.log`].
- **Sim evaluation (still water, Isaac, nominal dynamics, 3 worlds × 15 programs)** [측정: `RL_controller/logs/eval/ATT_COMPARE_20261003.md`,
  `logs/eval/att_att7pwm_s{0,1,2}/metrics.csv`, comparison `att_pwm10_s1_ood/`]: the retraining is effective — level-world pitch RMS (mean) s0 1.46°
  vs level-trained pwm10_s1 10.93°, pd20 steady-state error +1.1° vs +27.3°. **However**: (a) in the trim worlds (cob −0.029/−0.035) s0's
  steady-state pitch error is up to −7.1/−8.4° (pu20), roll up to −4.9°; (b) even during level hold, roll e RMS 2.8° (pwm10_s1 0.6°) — wobble;
  (c) vertical |throttle| 0.70 (pwm10_s1 0.30), saturated ticks 24–38 % (s1 57–83 %, s2 98–100 %); (d) all three seeds fall short of the design's
  promotion criteria [예측] (worst |ss pitch| ≤ 2°, |ss roll| ≤ 1.5°). Standard level evaluation: square position RMS s0 0.041 m · saturation 16 %
  vs pwm10_s1 0.036 m · 1 % [측정: `logs/eval/traj_att7pwm_s0_square/metrics.csv`, `traj_pwm10_s1_square/metrics.csv`]. So
  `rov_gui/control/rl_policies/att7pwm_s0/` was first shipped as a disabled candidate, and **was enabled on 2026-10-03 at the operator's request**
  (`config/hw_mpc.yaml rl_pwm_7d.policy_dir: rov_gui/control/rl_policies/att7pwm_s0`; `null` to disable). (a)–(d) above remain open problems — the
  first flight in water should start from a level reference (station hold or a 5-dim policy) [예측].
- **First water run 2026-10-03 20:27 — failed** [측정: `data/20261003/1003_202721/`]: RL_PWM_7D + 7-dim `20260929_peg7d_vitclip`,
  3 engagements of 1.7 / 6.3 / 2.5 s, all released by the pilot (`events.log`). Causes (2026-10-04 diagnosis, recomputed from the records):
  (1) **vertical co-contraction** — even in level hold, T5/T8 up · T6/T7 down by +230~+340 µs (`mpc_2027*.csv` pwm5..8, mean of the pre-plan
  rows), the same pattern as the sim level hold (`RL_controller/logs/eval/att_att7pwm_s0/timeseries.csv` level, t≥5 s: +281/+289/+278/+280 µs)
  = a learned habit, not a deployment bug (offline replay of the recorded states, residual ≤0.18 N, a `/tmp` diagnostic script — not preserved).
  The sign is right: in a regression on the mpc_tuned (mixer) flights in the same folder, up = T5/T8 PWM>1500, T6/T7 PWM<1500 → the policy's
  (+,−,−,+) is, as intended, 5/8 up · 6/7 down. The cost: current 22~42 A mean (`batt_a`; same-day mpc_tuned 3.6~4.5 A), voltage 15.4→14.2 V
  (UI video HUD), and because the net vertical force is the difference of two ~50 N forces, model error turns directly into net-force error.
  (2) **It actually sank** — despite a nominal +19 N upward command, ~5 cm/s sinking during level hold; pressure depth 0.66→0.74 m (before the
  first plan)→0.83 m (`ui_20261003_202732.mp4` HUD DEPTH, 20:27:35→37→40). peg −12 N is outside the training DR (−4.9~+6.9 N).
  (3) **±90 N vertical-force reversals** (engagements 1 and 3) — the policy has a high gain on the differenced velocity nu_w (1 tick +0.1 m/s →
  about −28 N [유도: replay]); 9 fixes in which the tag z jumped by 20~47 mm (engagement 3, 33 ms apart; engagement 2 had 0, max 12 mm) + a
  72 mm z mismatch of the auxiliary-camera fix (engagement 1) (`mpc_2027*_ekf_inputs.jsonl` fix differences). Sign-flip rate 0.41/tick
  (engagement 3) vs 0.14 (engagement 2), corr(uZ, nu_w) −0.71/−0.50 (CSV). (4) The DP is not the main cause — engagement 1 had 0 plans, and
  engagement 3's pitch +10.8° · roll −7.5° came before the first plan (`plan_id` nan); in engagement 2 the DP requested nose-up (NED +8~+12°)
  and the vehicle went over to +14.8°, then stayed at +13~+14.5° — judging from the origin height 0.197~0.216 m (tags) and the depth stalled
  at 0.83 m, likely the stern on the floor [유도: drawn geometry underside −0.217 m, not measured]. The control cycle (20 Hz) and the tag
  latency (p50 0.12~0.15 s) are the same as mpc_tuned → the pipeline is normal.
- **Second water run 2026-10-05 21:22 — the same failure even with STATION hold (tag 47) alone, without the DP** [측정: `data/20261005/1005_212023/`
  `mpc_212239.csv`·`mpc_212257.csv`·`mpc_212304.csv`, analysis `data/20261005/1005_rlpwm7d_station_diag_offline/{diag.py,report.txt}`]:
  3 engagements of 1.6 / 1.7 / 2.8 s, all released by the pilot, 0 plans (`plan_id` nan, rroll/rpitch 0) → the "DP is not the main cause" of
  (4) above is confirmed by experiment. (1) From the first pulse tick, all 4 verticals co-contract by +150~+300 µs (T5..T8 means
  +178/+175/+159/+227, +263/+247/+191/+298, +251/+247/+200/+290 µs; mutually cancelling share 166~374 µs, net upward +71~+124 µs — less than the
  same evening's mpc_tuned mixer net upward +163 µs) → current mean 26~29 A · max 71 A (mixer mode 3.3~4.6 A), 16.0→14.4 V, saturated ticks
  19~33 %, sinking +4.5~+6.1 cm/s (engagements 2 and 3, `*_ekf_inputs.jsonl` fix z). (2) 0.4~0.6 s after the pulses, a sharp attitude change
  (pitch −7.9→+3.8°, 0.35 s) takes the C3 tag count 23→15→1→0 (`nav_212237/fixes.csv`; in the video `1005_210544/ui_20261005_211603.mp4` the
  21:23:06 frame shows motion blur · 3 tags) — with `hw_nav.yaml min_tags: 1`, single-tag fixes that jumped +59/−100/+98 mm get through (host
  0.56 s) → differenced velocity nu_u −0.94 / nu_w +0.89 m/s → `state_assembler.py` extrapolates at that velocity across the 0.15 s fix gap
  (px −0.25→−0.38 m) → policy heave reverses −130 → +167 N (pwm6 1100 / pwm5 1900), roll −14.7° · pitch ±10° (engagement 1). The same
  evening's mixer mode also went through tag count ≤2 on 8~99 % of ticks and velocity jumps of up to 5.1 m/s, but kept flying with |uZ| p95
  11~59 N → the deciding factor is not a sensor defect but the policy's gain. The other warnings in the log are not the cause: MPC_TUNED_7D
  engage refused ×4 (`engage.attitude_axes.enabled` false — needs a station restart), "commands are leaving … every thruster sits at neutral"
  (all in the same second as ENGAGED — false alarms from the SERVO_OUTPUT_RAW report latency, normal flight afterwards), EST switch refused
  (EKF z 24.8 cm mismatch — the gate working correctly).
- **Improvement candidates [예측]**: stronger action-magnitude and per-thruster L1 penalties (to suppress vertical co-contraction), narrowing the
  trim range to the measured trim, more training iterations, a separate roll-error term. None of these has been tried.
- **Unresolved on the station side**: (1) no pulse version of sat-ineffective (iii) — there are no axes, so the s/t version is structurally
  blind. Only the tilt ceiling (35°) and div_rp exist. (2) There is no path to run the pwm closed loop offline inside the station (the demo has
  no pulse transport) — tracking performance is seen only through the RL_controller evaluation. (3) The sink gate is built from the
  `rl_pwm.policy_dir` export — if that export breaks, rl_pwm_7d is refused too (fail-closed). (4) div_rp judges against the stitcher (T1 20°
  clip), not against the follower's envelope (roll 9.7°) clip — conservative. (5) The per-thruster sign probe has still not been done (same as
  RL_PWM). (6) replay `track_attitude` flies level on this LOW (the replay attitude guard is s/t-only). (7) The tilt ceiling `abort_deg` 35° /
  0.25 s debounce uses as-is the values set for the s/t axes (K/M caps) — they were not re-set for the roll/pitch authority of the pulse
  transport (|throttle| ≤ `rl_pwm.pwm_cap`). If the pulses can tilt faster, the vehicle goes further over during the debounce [예측, 미실측]
  (review 2026-10-03: not a defect, recorded only).
- **2026-10-10 correction (re-analysis `RL_controller/logs/eval/pwm_cocontraction_analysis_20261010/REPORT.md`, workflow wf_412fe90b)** — the
  paragraphs above are left as they are and corrected here. (1) **CSV sign**: the station CSV's `pitch_deg` is `-pitch_NED`
  (`rov_gui/control/workers.py:10836`; checked against the same engagement's `*_ekf_inputs.jsonl` ATTITUDE: CSV −15.9° ↔ ATTITUDE +16.0°) —
  **negative = nose up**. The angles above must be read with this sign, and the comment "negative = nose UP" in the 10-05 `diag.py` was right.
  (2) **The training world's trim sign was wrong**: `traj10att`'s `cob_trim_x_range_m [-0.035, 0.005]` (0.15 m arm) is a world from 13° nose
  **down** to 2° nose up, whereas the hardware was +10.6° nose-up on 09-21 (jaw empty, mid-water), and in October (10) +0~+3° holding the peg,
  +7.5° nose-up without it [측정: station CSV]. The restoring stiffness too: sim 20.2 N·m/rad (0.15 m arm) vs measured 4–11 (static balance
  3.7–7.6, peg step 8.4, free period 2.6–3.5 s) — 3–5× over-stiff. (3) **What att7pwm_s0 actually did**: the null space of the 8-thruster
  allocation matrix B (6×8) = span{n_v = the same +PWM on all 4 verticals (FR up · FL down · RR down · RL up), n_h = front horizontals forward ·
  rear horizontals reverse}, and all 6 engagements produced **n_v +250~+290 µs + n_h −125~−140 µs** — 85–96 % of the thrust is wrench 0,
  25–43 N of vertical thrust cancels out, 26–44 A (of which 28–42 A is the wrench-0 share), 16.0→14.0 V, 4.5–6 cm/s sinking under a nominal
  net upward command of +9~+16 N [측정: same folder, `scratch/hw-decomp/report.txt`]. The "vertical co-contraction" in (1) above is right, but
  the horizontal pair n_h was missing. The tag collapse came only **after** attitude changes (0 times before them), and it is decided by
  altitude (P(n_tags≤2) 0.8–0.95 at <0.3 m) and attitude rate, not by the pitch angle. The reason the sim learns this is not latency but the
  convex T200 curve with a deadband (a constant bias = free gain; the only cost, exp(−|a|²), is already 74–96 % used up). Action taken:
  retraining `pwm11` (correct trim sign · 0.05 m arm · null-space penalty, RL_controller README §13), a pre-flight probe block; the station cap
  and gate changes are a separate entry on the same day. **att7pwm_s0..2 are not to be flown again at any cap.**

### LOW `RL_PWM` (direct-thruster pulses) **saturates and oscillates** at the 0.25 cap — first water run 2026-10-01 17:29 (listed 2026-09-30)
- **What**: the `rl_pwm` mode (`rov_gui/control/rl_pwm.py`, `rl_policy.py` `HwRlPwm`, `backends/hardware.py` `set_pwm`) sends the 8 pulses of
  the policy `pwm10_s1` on RC_CHANNELS_OVERRIDE ch 9..16, and the vehicle's `rl_pwm_override.lua` forces them onto the motor outputs. It
  bypasses the ArduSub mixer.
- **What is confirmed**: 26 offline tests (`rov_gui/tests/test_rl_pwm.py`) — the bytes of real pymavlink v2 frames match the positions the
  script reads, refusal per gate item, sink latch and stop paths, worker gate/stop. On the demo backend it is refused with "no transport path".
  (After the operator changed `pwm_cap` to 1.0 on 2026-10-01, 11 of these were **failing** because they pinned the shipped value 0.25 — fixed
  on 2026-10-03 not to depend on the shipped value, 26/26.)
- **What changed on 2026-10-03**: (a) **a tilt ceiling was added** — until then rl_pwm had no tilt interlock (the ceiling applied only to the
  s/t path, and the Lua script has no attitude logic either). Now if |roll|/|pitch| > `abort_deg` 35° lasts 0.25 s, it disengages (neutral
  pulses). The 10-01 run's pitch −9~−18° is a value that does not trip it. To revert, change `self.attitude_actuated` in
  `workers._runtime_fault` back to `self._attitude_axes`. (b) The cap is now a single per-transport key `rl_pwm.pwm_cap` (`rl_pwm_7d` reads it
  too). (c) `HwRlPwm.reset` sets `n_fail` to 0 — a latent defect in which the previous engagement's failure count was stamped on the next
  engagement's first tick as a false "solver FAILURE" (fail-streak 1). (d) For pwm, meta `station_bridge.released_in_coast` reads "all 8 at
  1500 µs — no depth or attitude hold" (the earlier sentence, "heave and attitude kept", was false).
- **First water run (2026-10-01 17:29–17:30, `data/20261001/1001_172910`, 3 policy missions of 42/8/24 s)**: the pulses went out
  (`rl_pwm: policy pulses are going out on RC 9..16`, `pwm_age_s` mean 0.14 s). Motion: roll −8~+6° (period about 3 s), pitch biased to
  −9~−18° with a 1–4° wobble, yaw wobble up to 34° (std 4–10°); position tracking itself rms 2–4 cm · max 15 cm [측정:
  `mpc_172910/172957/173011.csv` `roll_deg`·`pitch_deg`·`yaw_deg`, px−rx etc.]. (The same day's round-1 report, "pitch ±5–10° · yaw −16° ·
  depth 0.1–0.4 m", was the reference-position columns `ry`/`rz` misread as attitude — retracted. The depth change is mostly the reference
  going down.) **The cause is the cap**: the thrust one thruster produces at the 0.25 cap is 5.8 N, versus 64 N in training (1.0) — the policy
  was trained assuming 11× more authority [유도: `rl_pwm.py PwmModel` T200 curve]. Thrusters 3·4·6 sit on the cap 61–91 % of the time and 1·5
  44–64 %, and since the cap is **a clip, not a scale** (`rl_policy.py:350`, `rl_pwm.py:79`), when several thrusters exceed 0.25 at once the
  ratios between thrusters (the allocation) collapse. It is not latency (tag 0.13 s, 1 late plan per run). **Evidence for the direction**: in
  the same runs the roll · pitch · yaw angular accelerations and the vertical acceleration agree in sign with the model's uK · uM · uN · uZ (all
  three runs; weak, correlation 0.1–0.4) [측정: regression on the same CSVs]. Command→pulse has been cross-checked against the ArduSub mixer
  with hardware logs (`RL_controller/README.md` line 28, `scripts/fit_mixer_from_logs.py`). A one-thruster-at-a-time sign probe (§10.3 (4)) was
  not done. **2026-10-01 operator decision**: `pwm_cap` 1.0 (in one step). It has not been run at 1.0 yet.
- **What changed on 2026-10-10** (evidence: `RL_controller/logs/eval/pwm_cocontraction_analysis_20261010/REPORT.md` §1 cause 4 "deployment cap
  as a per-thruster CLIP" · §3 step 5): (a) **the cap became a scale factor in thrust space instead of a clip** — `rl_pwm.pwm_cap_mode: scale`
  (default; `clip` is for A/B) `rov_gui/control/rl_pwm.py scale_throttles`: multiplies the 8 thrusts by a single scale factor s and maps them
  back through the closed-form inverse curve (`T200CurveNp.inverse`, positive- and negative-branch round-trip error 1.4e-8 N). The allocation
  direction is kept (the att7pwm_s0 pool pattern [−115,−125,+125,+120,+260,+220,+200,+270] µs at cap 0.5: s 0.578, cos 1.000, max|throttle|
  0.500; a clip of the same pattern gives cos < 0.999) [측정: `rov_gui/tests/test_rl_pwm_scale.py`]. Pulses over the cap are now clipped
  nowhere but **refused** (follower `throttles_to_pwm(strict=True)` → neutral + n_fail; sink `set_pwm` → refuse + latch + neutral). Limits
  [측정: same test, seed 20261010]: when the scale factor pushes a thruster below the deadband, the rule "keep a thruster that is on at the edge
  (+1.44/−1.10 N)" bends the direction — over 10⁴ uniform random vectors, cap 0.5 worst cos 0.98, cap 0.25 (cap-1.0 policy, s≈0.09) worst
  0.46. That is, **a cap far below the training cap is not safe even with scale** (hence (b)). (b) **A configured cap below the training cap
  is refused at construction** — if `rl_pwm.pwm_cap` < the export's `pwm_cap_trained`, `HwRlPwm` refuses, citing both numbers, and only
  `rl_pwm.allow_caps_below_trained: true` (recorded, meta `controller.allow_caps_below_trained`) lifts it; `rl_pwm_7d` uses the key in its own
  block. The 10-01 run (cap 1.0 policy + 0.25 clip) was exactly this case. (c) **Per-thruster sign probe artifact gate**
  `rl_pwm.require_thruster_probe: true` (default): it engages only if the `data/*/*_probe/thruster_sign_probe.json` (schema 1) written by
  `rov_gui/tools/thruster_probe.py` is not a dry_run · within `thruster_probe_max_age_days` (30 [예측]) · for this export's sigma · for the
  vehicle's current MOT_1..8_DIRECTION · all_pass on all 8 (worker `_pwm_refusal` + sink `PwmGate.check`,
  `rov_gui/control/rl_pwm.py ThrusterProbeCheck`; a missing reader module `rov_gui/control/thruster_probe_artifact.py` or a missing file is
  also refused). **No probe has been taken yet, not even once** → with the shipped config, RL_PWM engagement is refused (intended). (d) CSV
  schema 19: the **predicted pack current `i_pred_a`** of the sent pulses (2.1 + Σ 0.0584|F|^1.526 / 0.0537|F|^1.673, F is the export T200
  curve × batt_v/14.8, 0.72 if there is no voltage [유도: fit to the 15 V column of the public T200 datasheet; REPORT 202735 pattern 42.3 A
  predicted vs 41.6 A measured mean]) and the **null-space monitors `null_v_us` = mean(pwm5..8)−1500, `null_h_us` = ((pwm1+pwm2)−(pwm3+pwm4))/4**
  [유도 REPORT §0] — only on pwm LOW rows, blank for other LOWs. Recorded only; the trip thresholds (100–150 µs, 50 A) are a follow-up. Tests:
  `test_rl_pwm.py` 26 + `test_rl_pwm_7d.py` 14 + new `test_rl_pwm_scale.py`. 0 water runs, uncommitted.
- **What changed on 2026-10-10 (b)** (REPORT §3 step 5, latter half: "interlocks … policy-side tag gate → hold wrench, replay-tested on the 14
  engagements"): (a) **runtime interlocks only for the direct-thruster-pulse LOWs** `engage.per_thruster:` (`rov_gui/control/rl_pwm.py PwmInterlocks`,
  judged by `workers._runtime_fault` every tick right after the 35° ceiling, with the same tick-counter debounce as the ceiling; trip = disengage
  (neutral pulses) + mission log/events.log "PER-THRUSTER INTERLOCK <name> — disengaged" + a sentence carrying the measured values + meta
  `controller.interlocks.last_trip`): **shipped values (v2, replaced the same day based on the replay results)** tilt |roll|/|pitch| > 20° for
  0.25 s, rate |p|/|q| > **40** deg/s for 0.15 s [측정 REPORT §1 cause 2: cascade 33–65 deg/s; the first draft's 60 missed 2 of 14 in the
  replay], pack batt_a > 50 A for 1.0 s (BATTERY_STATUS ~1 Hz latest value) [스펙 BR 18 Ah 60 A continuous] / batt_v < 13.5 V immediately,
  null on the sent pulses: |null_v|/|null_h| > **80 µs for 0.5 s** [측정 REPORT: mixer 0, pwm10_s1 +82..+107, att7pwm +250..+290 µs; the first
  draft's 120 µs/1.0 s does not catch the pwm10_s1 squeeze and missed 2 engagements of 1.3–1.7 s], pinned: ≥ 3 thrusters on the cap for 1.0 s.
  All [예측]. Other LOWs do not read the block (existing suite green). (b) **Policy-side observation guard** `rl_pwm.obs_guard:` (`PwmObsGuard`,
  only inside `HwRlPwm.step`; the shared state_assembler and the CSV state columns are unchanged): a per-axis slew of 2.0 m/s² on the body
  velocity the policy sees [유도: policy max acceleration ≤ 4.5]; when n_tags < **2** (min_tags; the first draft's 3 would have sent 5–99 % of
  ticks to HOLD even in the mixer runs [측정 replay]), a fix jump of more than 0.15 m in one tick is rejected (the previous fix is kept); tag
  gate: n_tags < 2 or fix age > 0.3 s for 0.2 s or more → instead of the policy pulses, **HOLD pulses** (Z −4.0 N NED = 4 N up [측정 REPORT §1
  cause 5: October (10) net weight with peg 2.5–5.4 N] via the export's minimum-norm vertical allocation + the T200 inverse curve; on the
  pwm10_s1 export [1500,1500,1500,1500,1537,1452,1470,1530] µs ≈ 3.7 N up [유도], horizontals neutral; the prev_action fed back is the hold
  that went out), returning once the tags have been back for 0.2 s. CSV schema **20**: `pwm_guard` column (0 none/1 slew/2 jump/3 hold; pwm
  rows only). (c) The thrust convention of `i_pred_a` is now the same **physical convention** as RL_controller's: F = T200(thr) × batt_v/20.5 V
  (fit 1.0 = the ~20 V curve, 14.8 V → 0.72 = export voltage_scale, 0.72 if there is no voltage) [스펙 rov_rl/thrusters.py; rov_env
  `_nominal_voltage` 0.72] — the previous one (×batt_v/14.8) overstated thrust 1.4×. Under this convention the att7pwm_s0 pool pattern is
  predicted at 14.8 V 25.9 A / 16 V 28.9 A vs a measured mean of 41.6 A (202735) — **the datasheet proxy underpredicts the co-contraction
  pattern by ~1.6×** (pack sag · ESC losses · floor proximity [예측]); that is why the pack interlock watches `batt_a`. (d) **Offline replay**
  `rov_gui/tools/rl_pwm_interlock_replay.py` (the same `PwmInterlocks`; inputs: the CSV's ATTITUDE angles · BATTERY · vehicle-reported pwm1..8 (not the sent pulses but
  SERVO_OUTPUT_RAW, ~0.14 s late) · gyro from `*_ekf_inputs.jsonl` pqr (10-03/05) or angle finite differences (10-01)) → `data/20261010/
  1010_interlock_replay_offline/report.txt` (first draft 60/120/1.0 s, null 80·100·120 compared), `report_sensitivity_rate40_null05s.txt`,
  **`report_shipped_v2.txt` (shipped v2)**). **First-draft results [측정: report.txt]**: mixer mode, 91 files, 0 trips; of the 14 pwm
  engagements, 12 trip with null 120 — 10-01 cap-0.25 pwm10_s1, all 5 engagements on **pinned** (+0.96..+2.89 s; at rate 24–38 deg/s rate does
  not fire, and at a squeeze of 95–100 µs null is **not caught** at 100 or at 120), 10-03 cap-1.0 pwm10_s1, 3 engagements: tilt at
  +1.00/+1.25 s and **1 with no trip** (180854, a 1.3 s run: rate 62 deg/s does not last the 0.15 s, null 139 µs does not last the 1 s),
  att7pwm, 6 engagements: null +0.95 s ×5 and **1 with no trip** (212239, a 1.6 s run: null 297 µs drops out for one tick after 0.95 s) — the
  expectation "within ~0.5 s via rate or null" [예측] does not hold with the first draft (the 1.0 s debounce dominates). **Shipped v2 results
  [측정: report_shipped_v2.txt]**: with null 80 µs/0.5 s, **14/14 all trip on null, +0.43..+0.60 s** (10-01 5 engagements +0.44..+0.60,
  10-03 pwm10_s1 3 engagements +0.45, att7pwm 6 engagements +0.43..+0.45), mixer mode still **0/91**; on the same data, null 100 gives 14/14
  (null 8 · pinned 5 · tilt 1) and 120 gives 13/14. That is, at the shipped values null is the first trip and tilt/rate/pinned are backstop
  protection (rate 40 fired first in none of the 14 replayed engagements — it is later than or equal to the earliest null). **Therefore
  pwm10_s1 / att7pwm_s* end within 0.5 s under the shipped interlocks** (consistent with REPORT §3 step 6, "do not fly them again"; in the
  offline harness too, pwm10_s1's at-rest-frame output null_h ≈ +144 µs trips in just 0.5 s [측정 test_rl_pwm_interlocks]). Limits: the
  replay's pulses are the vehicle's reports (SERVO_OUTPUT_RAW, ~0.14 s latency), not the sent pulses; the 10-01 angular rates are angle finite
  differences. Tests: new `test_rl_pwm_interlocks.py` 13 + existing 57 + test_thruster_probe 18 green; schema pin updated 19→20. 0 water
  runs, uncommitted.
- **Installed 2026-10-01**: `rov_gui/tools/install_rl_pwm_lua.py --install --yes` set `SCR_ENABLE` 0→1 and `SCR_USER1` 0→1 and uploaded the
  script via MAVLink FTP to `scripts/rl_pwm_override.lua` (read back, byte-identical), and after a reboot the vehicle sent
  `rl_pwm_override.lua loaded` — the script **does load** (initialization, including the `mavlink:init` argument order, passes)
  [측정: `data/20261001/1001_145846_vehicle_setup/rl_pwm_lua_install.json`]. The changed values and the previous values are in that file.
  The same day, `SERVO14_FUNCTION` 60 (RCIN10 passthrough, output 0 µs — nothing sends a value on RC10) → 0 (unused), at the operator's request
  [측정: `data/20261001/1001_150534_vehicle_setup/servo14_function.json`]. All the gate's parameter items now pass
  (`install_rl_pwm_lua.py` read-only check). What remains is confirming the script heartbeat and engagement — which can only be seen in an
  actual engagement.
- **What is not confirmed [예측]**: whether the script actually takes over the pulses (`set_output_pwm_chan_timeout` behaviour,
  `RLPWM: engaged`), the margin between the 50 ms frame period and the script's 100 ms stale limit, per-thruster direction (the gate checks
  that `MOT_n_DIRECTION` matches, but not the physical direction), whether the policy flies at `pwm_cap` 0.25 (trained at 1.0).
- **Known limits**: (1) on this vehicle, if lights · camera · gripper are mapped to RCIN9..16 passthrough, the gate refuses — they have to be
  moved before it can be used. (2) The CSV has no commanded-pulse columns (`pwm1..8` are vehicle-reported values, `uX..uN` nominal
  estimates). (3) The "commands are going out but the thrusters are at neutral" monitor (`_watch_actuation`) is axis-based, so it is off in
  this mode — the script's `RLPWM` value stands in for it. (4) During a station bridge coast it emits only neutral pulses (no depth hold).
  (5) The sink does not check arm/MANUAL itself (the worker and the script do); it does not read `SYSID_THISMAV`. (6) During the tag-dropout
  debounce (up to 0.4 s) there are neither pulses nor updates of the previous command.
- **Procedure before water entry**: `rov_gui/control/rl_policies/pwm10_s1/README.md` — install the script → bench with the propellers removed
  (check engagement · E-STOP · handover · `RLPWM: released` blinking) → per-thruster direction → in water, `pwm_cap` 0.25, in a shallow spot
  where it can be observed.
- **Review**: safety-code-reviewer round 1 FAIL (3 gate holes: channel permutations allowed, grace failure not latched, parameter session
  cache) → fixed → round 2 WARN ("first water run at cap 0.25 allowed after passing the bench", 2 recommendations before water: the grace
  timer resets on every interruption, no latch on a sink tick exception) → both fixed. **The last round of fixes was not re-reviewed.** The
  reviewer could not run the code. Remaining recommendations: on the bench with QGC attached for at least 10 minutes — parameter age, the
  frequency of spurious latches from late frames, a cross-check of the `RCINnScaled` numbers (139+n). Recorded in `.claude/journal/reviews.md`.
- **2026-10-10 correction (re-analysis `RL_controller/logs/eval/pwm_cocontraction_analysis_20261010/REPORT.md`)** — the paragraphs above are
  left as they are and corrected here. (1) **CSV sign**: `pitch_deg` is `-pitch_NED` (`workers.py:10836`), so the "pitch −9~−18°" above is
  **9–18° nose-up** (negative = nose up). (2) **Trim sign error in the training world**: pwm10_s1's world (`traj10`, `cob_trim_x_m -0.029`) is a
  10.6° nose-**down** world built by reading the 09-21 CSV value −10.6 as nose-down, whereas the hardware was +10.6° nose-up (09-21, jaw
  empty), and in October (10) +0~+3° holding the peg [측정: station CSV]; the restoring stiffness too: sim 20.2 N·m/rad vs measured 4–11 (3–5×
  over-stiff). The policy learned a +2.8~+3.5 N·m nose-up feedforward moment for the nose-down world, and in all 8 engagements (10-01 cap 0.25,
  5 times, 119 s + 10-03 18:08 cap 1.0, 3 times, 2.9/7.3/1.3 s, `data/20261003/1003_180545/mpc_1808{20,35,54}.csv`) it pitched nose-up
  **in the commanded direction** (max +13~+25°, gyro I·α = 73–97 % of the reconstructed moment) — not a sign flip or swapped wiring (3
  independent verifications rejected that). The per-thruster sign probe (§10.3 (4)) is still mandatory. (3) **At cap 0.25 there was no
  vertical co-contraction** (c_v −17~−43 µs); instead a wrench-0 squeeze along the horizontal null direction n_h (front forward · rear
  reverse) of **+82~+107 µs = 12–17 N** pinned T3/T4/T6 to the cap (48–94 % of ticks); current 6.8–7.1 A mean (cap 1.0: 9–12 A).
  (4) **Floor-contact contamination**: most of the 10-01 holds sank to the floor (hull-on-the-floor attitude +7~+10° nose-up; 172910 started on
  the floor), and the attitude and tracking figures above include floor contact. Reproduction on the sim side: re-evaluated in a 0.05 m arm ·
  nose-up trim world, pwm10_s1 pitches +14~+16° nose-up [측정: `RL_controller/logs/eval/sweep_rl_pwm10_s1_{hwlike,nominal05}`]. The action
  taken is the same as in the correction to the RL_PWM_7D entry above (`pwm11`, probe block, RL_controller README §13); the cap implementation
  and gate changes are a separate entry on the same day. **pwm10_s1 · pwm10_s2 are not to be flown again at any cap.**

### Policy inference takes **19 ms** on its own but 0.21–0.27 s in the station — because it shares the GPU with FoundationStereo (2026-10-01)
- **Measurement** [`data/20260930/0930_220212/diag/policy_vs_fstereo_contention.json` + `policy_alone_timing.json`, script in the same folder; RTX 5090, peg ViT-CLIP ckpt, `model` weights, no GUI·cameras]:
  policy forward alone: 8 step 19.3 ms / 16 step 33.5 ms / 4 step 12.2 ms. Running FoundationStereo (iters 8, scale 0.75, CUDA graph) non-stop in the same process gives
  8 step p50 152.8 ms (p90 210.0), 4 step 104.4 ms. Running FS intermittently at 7.5 Hz gives 8 step p50 40.3 ms (p90 86.3).
  On hardware (6 engagements at 22:02·22:16): infer_ms p50 207–273 ms [`plan_timing.json`], FS solve p50 99 ms (76 ms in the offline bench where it ran non-stop beside the policy — there is no FS-alone measurement in that file)·9.7–11.2 Hz [`latency_breakdown.json` — 10 frames of the UI recording read by eye, `depth_hud_montage.png`].
- **Composition of the plan age** (rl run `mpc_220407`, p50 0.556 s) [`latency_breakdown.json`]: capture→depth complete 0.196 s (FS solve 0.099 + camera transfer·wait 0.098 [유도: difference]),
  policy forward 0.207 s, remainder 0.153 s [유도: difference — building the observation from 2 frames·trigger·bus·20 Hz tick; since 2026-10-01 it is split by `depth_ready_age_s`/`trigger_age_s`/`emit_age_s` in plans.jsonl: obs build+pack p50 0.046–0.081 s [유도: emit−trigger−infer, `data/20261012/1012_fs_server_review_offline/plan_timing_by_schedule.json`], the camera transfer ~0.098 s is still an unstamped difference].
- **Meaning**: the bulk of the latency is not the diffusion computation but waiting for the GPU. `num_inference_steps` 8→4 gives −7 ms alone and −48 ms (−32 %) when overlapping with FS, so it does not halve the time.
- **Workaround (implemented, OFF by default, 2026-10-01; water results in the 2026-10-12 item below)**: `--policy-fs-schedule yield|only` (default `free` = the state above, unchanged) — with yield, FS does not start a new frame while the policy
  is running inference (the policy waits up to 150 ms for the frame in progress); with only, during a policy mission FS computes only the 2 frames the policy uses.
  Another knob that needs no code change is `--fstereo-scale 0.5` (FS 45.7 ms, so it keeps up with 15 fps and rests in between). Offline bench measured with the actual two workers·two networks
  [측정: `rov_gui/tools/fstereo_bench_out/policy_fs_schedule_20261001_115008.json`, synthetic input so no camera transfer latency, CPU load 0.9→7.5/16 cores during the measurement]:
  policy forward p50 132.7 ms (free) → 19.4 (yield) / 19.4 (only) / 30.8 (scale 0.5) / 19.6 ms (yield+0.5); plan age (emit) p50 328.8 → 224.6 / 154.5 / 134.5 / 145.6 ms;
  consecutive frame pairs 82.5 % → 86.0 / 100 / 100 / 100 %; FS during the mission 12.93 → 11.93 / 3.73 / 15.0 / 15.0 frames/s. Usage·record keys·pooling boundary: `rov_gui/README.md`
  section "Sharing the GPU with FoundationStereo".
  The prototype figures written earlier [`data/20260930/0930_220212/diag/policy_fs_schedules.json`] come from a simple loop, not the workers, and that JSON's note
  "the wait is ~0 in the station" was an assumption, not a measurement (the trigger is the period phase, not an arrival, so the wait is the remaining time of the frame in progress — prototype
  p50 43.1 / p90 69.9 ms, worker bench p50 0.0 / p90 19.5 ms). The "panel stale threshold 1.5 s" of (A) was also wrong: the `--fstereo` depth panel is warn 0.75 s /
  stale 2.5 s (window.py `expect = min(expect, 4.0)` → widgets/video.py Freshness) — the conclusion (no warning appears) is the same.
  With `emit_age_s`·`trigger_age_s`·`depth_ready_age_s`·`fs_wait_ms` in `plans.jsonl`, the "remainder 0.15 s" can now be split (on every policy plan line, `free` included).
- **Water check + 'separate process' bench (2026-10-12 review)**: 10-01 in water (same ckpt, `data/20261012/1012_fs_server_review_offline/plan_timing_by_schedule.json`)
  free@0.75 age p50 0.53 s / infer 212 ms / expired before the next install 47 % → yield@0.75 0.36 s / 44–46 ms / 0.3 % → only@0.75 0.28–0.31 s / 27–35 ms / 2 %;
  free@0.5 0.35–0.40 s / 113–133 ms / 4–7 %. The user went back to free@0.5 from 16:48, and **the reason was not recorded** (gate timeouts 0). yield@0.5: 0 runs.
  The lab's proposal 'FS in a separate process', in a bench [`rov_gui/tools/fstereo_bench_out/two_process_20261011_235457.json`, no IPC·cameras·GUI, MPS not used],
  only reduces forward 151.7→**47.2 ms** (p90 209→49) and does not get back to the standalone 19.4 ms (same-GPU time-slicing; FS p90 77→88.5 ms); it falls short of yield (19.4).
  **Unattributed**: the station's forward is ~3× slower than the bench with the same arrangement (free@0.5 113–165 vs 30–42 ms, yield 44–46 vs 19.4) — unknown whether it is GPU-bound or CPU launch-bound.
- **Method 3 `--fstereo-process` (2026-10-12, 0 water runs)**: only the FS network runs in a child process. Output bit-identical (18/18, |Δ| 0 mm)
  [측정: `rov_gui/tools/fstereo_bench_out/proc_parity_20261012_120951.json`]. Real-worker bench: forward p50 at scale 0.75 121.6→40.8 ms
  (yield 19.4), plan age (emit) p50 321.2→222.3 ms (yield 233.3); at scale 0.5, forward 30.6–31.4→29.5–32.6 ms, plan age p50
  133→129–133 ms (yield 143–147), p90 191–192→171–179 ms (yield 164–167) [측정: `policy_fs_schedule_20261012_100908.json`,
  `policy_fs_schedule_20261012_121018.json`]. If the 3× gap under 'Unattributed' above is FS's CPU/GIL share, the split may do better in water; if it is the GPU's share, it does not.
- **Proper fix**: first look at the hardware `fs_wait_ms`·`burst_timeouts`·plan age with a LOW None (observe) run, then pick one of free / yield / only / scale 0.5 by an in-water A/B
  and make it the default (or delete the switch). From 2026-10-12, `--fstereo-process` also goes into the candidates (ABAB in the same session). The accuracy cost of scale 0.5 is to be checked by an offline A/B after collecting raw underwater data with `--record-stereo` — not run yet.

### Policy plans **end before the next plan arrives** — the reference stands at the end point 26–78 % of the time (2026-10-01)
- **Found**: 2026-10-01, while checking against the code·records the question "shouldn't the leading actions be discarded by the amount of the inference latency?".
- **The time alignment itself is correct (code)**: the plan's time axis is fixed to the observation time (`workers.py` `compose_plan(..., t0=obs_t_rel)`, raw action k = observation time + k·66.7 ms),
  and the stitcher samples that axis at the "now" time (`plan_stream._PlanRef.sample`) — the leading part covering the latency is skipped automatically. raw action 0 is the pose at the observation time
  (|action 0| p50 0.3–0.9 mm).
- **Measurement** [`data/20260930/0930_220212/diag/plan_timing.json`, script `plan_timing.py` in the same folder; 6 engagements at 22:02·22:16]:
  plan age p50 0.56–0.66 s (pre-inference chain 0.35–0.39 + inference 0.21–0.27) → raw action index at which the reference sits at install time p50 **7.4–8.3** (the first 8 are never flown).
  Remaining life after install p50 0.45–0.52 s (plans without slow-down), time to the next install p50 0.55–2.55 s → plans that ended before the next install **54–94 %**.
  CSV `ref_src`: hold is 26 % in the rl run (`mpc_220407`) and 38–78 % in the 5 mpc_tuned engagements; pure plan is 3–21 %, the rest is blend.
- **Cause**: the chunk length of 1.0 s (16 × 66.7 ms) has no margin. Inference runs overlapped with execution (observation interval p50 0.533 s; cycles in which the next observation was taken before the previous plan arrived 73–96 %).
  Looking only at installed plans, age p50 0.52–0.57 s and remaining life p50 0.45–0.52 s, so **if the next plan arrives normally, the empty window is only p50 0.0–0.10 s (p90 ≤ 0.14 s)**,
  but **if the next plan is dropped as late, p50 0.5–1.2 s (worst run 3.9 s)** goes empty [측정: `data/20260930/0930_220212/diag/plan_hole.json`] — most of the hold time comes from late drops.
  After expiry `hold_tail: extrapolate` switches off and it becomes an end-point stop (`_extrapolate_tail` only while the plan is live).
- **In passing**: in the same records, plans whose observation frame-pair interval is 66.7 ms and plans where it is 133 ms are half and half (rl run 234 : 182) — the same problem as the 2026-09-02 "depth pair" entry below.
  `--policy-fs-schedule only` and `--fstereo-scale 0.5` gave 100 % consecutive frame pairs in the offline worker bench (free 82.5 %) [측정: `rov_gui/tools/fstereo_bench_out/policy_fs_schedule_20261001_115008.json`].
- **How to fix**: reduce the age — the switch in the entry right above (implemented, OFF by default, unverified in water; bench plan age p50 328.8 → 134.5–224.6 ms). Other options: reduce `period_s` to
  about 0.35 s [예측: the GPU is shared with FS, so the age may grow, untested], continue the extrapolation after expiry (code), or retrain with a longer action horizon (not applied).

### Even when the policy outputs "open gripper", the station does not open it — turning it on also carries a misfire risk (2026-09-30)
- **Fact**: on 2026-09-30 `policy.gripper: false`, so the width channel was discarded (true from 10-01, lookahead 0.8 — first policy OPEN executed in the 10-01 15:19 rl run,
  `data/20261001/1001_151431/mpc_151814.csv` grip_cmd). The remaining problems are **misfires** and the lack of jaw feedback. The peg policy outputs open at the moment of insertion: 6 plans of rl run `mpc_212757` at t=77–84 s
  output a width of 0.068–0.085 m 1 s ahead, and at that time the peg was inside the hole (`data/20260930/0930_211846/diag/open_moment.png`, the pilot opened it manually at 85 s).
- **Misfire**: the same day, 5 plans of `mpc_213330` at t=9–20 s also output open, but the peg was outside the hole (`data/20260930/0930_211846/diag/open_moment2.png`) [측정: `data/20260930/0930_211846/diag/plan_age_and_open_requests.json`].
- **When turning it on**: `gripper: true` alone does not open it — open sits at the end of the chunk (about 0.9–1.0 s ahead) and knot 0 is the current width, so `gripper_lookahead_s` 0.0 (sampling "now")
  cannot cross the threshold (width 0.063 m). 0.6–0.8 s is needed (the 09-07 comment in the same file). There is no jaw feedback, and a misfire drops the peg. Unverified in water.
- **Re-check at 22 h**: in rl run `0930_220212/mpc_220407`, while the peg was in the hole (`data/20260930/0930_220212/diag/ui_t303.png`, `ui_t318.png`), at t=176–214 s
  17 plans output open (width up to 0.090 m), and CSV `grip_cmd` is 0 over the whole span. Applying `gripper_lookahead_s` to the installed plans, 0.0 s crosses the threshold 0 times and 0.8 s 9 times.
  In the 176 s before the insertion and in the 5 mpc_tuned engagements of the same night, 0 open requests (no misfire) [측정: `data/20260930/0930_220212/diag/open_requests.json`].

### The peg-in-hole policy (`peg5d_vitclip_9_27_9_28`) outputs a **retreat even while holding the peg** in water — the jaw+peg in the ROV image differ in position·shape from the demos (2026-09-30)
- **Symptom**: in all 7 engagements of the 2026-09-30 policy runs, the plans point back+up. In the observe run `data/20260930/0930_183320_observe/plans.jsonl`, over its 1079 plans
  the last knot (1.0 s ahead) has mean TCP dz −0.048 m, dz<0 fraction 0.958; in the tracking runs the hull was actually pushed back 0.10–0.50 m
  [측정: `data/20260930/0930_183320_observe/diag/action_direction_stats.json`, `mpc_*.csv` px/py of each run]. The jaw was holding the peg, and the peg is also visible in the depth
  [측정: `data/20260930/0930_182033/ui_20260930_182035.mp4` C3 RGB·C3 DEPTH panels].
- **Cause (partly identified)**: reconstructed the policy input from the depth panel of the UI recording and re-ran inference (reconstructed obs MAE 1.4–2.1/255 [측정: `data/20260930/0930_183320_observe/diag/recon_fit_0921.json`];
  dz correlation with the recorded plans 0.91·sign agreement 0.95, 322 plans [측정: `data/20260930/0930_183320_observe/diag/final_tests.json` `_check`]). On the reconstructed obs the dz<0 fraction is
  0.90 unmodified → **0.31 with the jaw+peg erased** → 0.66 with only the jaw+peg moved 42 px down (to the demos' position) → 0.53 with the proprio width 0.007 m also given on top of that
  [측정: `data/20260930/0930_183320_observe/diag/final_tests.json`]. That is, what produces the retreat is **the jaw+peg captured in the ROV image itself**. On the ROV the jaw is 0.067 m below the optical axis, on the handheld 0.129 m, so
  [configured value: `mpc_183320.meta.json policy.tcp`] in the 224 px obs the jaw+peg appear about 43 px higher [유도]. The effect of the proprio width alone is small (0.90 → 0.83).
- **What differs (measurement)**: fitting the peg axis in the depth, the camera–gripper axis tilt is the same on both rigs (ROV 40.8°, handheld 9/28 40.8°, 9/27 35.6°).
  The perpendicular distance from the camera to the peg axis is also the same (0.213 vs 0.215–0.217 m [유도: root coordinates in the same file]). What differs is the **distance along the axis**: on the ROV the peg root sits
  4.6–5.9 cm further forward [측정: `data/20260930/0930_183320_observe/diag/rig_geometry.json`; assuming both depths are correct in metres]. Computing it separately from the configured TCP offset also gives 3.9 cm forward [유도: meta `policy.tcp`].
  So it is not a calibration (intrinsics·warp) problem but a difference in the fore-aft camera–jaw distance. Needs checking with a ruler (not measured).
- **Would fixing it work (prediction)**: reprojecting the reconstructed obs in 3D to make a viewpoint with the camera moved forward along the gripper axis gives a dz<0 fraction of 0.90 → 0.66 (3 cm) → 0.23 (5 cm) → 0.22 (7 cm),
  and with the proprio width 0.007 m also given, at 5 cm 0.16·dz +0.018 m·width output 0.008 m (keeps holding) [예측: `data/20260930/0930_183320_observe/diag/virtual_camera.json`, 322 plans; an approximation, since it is a single-viewpoint depth reprojection].
- **Retracted**: the inference "the jaw was empty" from the same day's round-1 diagnosis. `policy.estimator.n_drives` 0 does not count joystick-button actuation, so it cannot serve as evidence.
- **Workaround (implemented, OFF by default, unverified in water)**: `config/hw_mpc.yaml policy.obs_view_forward_m` — redraws the depth obs the policy sees from a virtual viewpoint that is that distance ahead of the C3
  along body x (`rov_gui/perception/policy_obs.py` `_shift_view`; the action transform is unchanged). Re-running inference on that day's 322 plans with the actual builder gives
  dz<0 fraction 0.90 → 0.31 (0.05 m), and 0.18·dz +0.018 m with the proprio width 0.007 m also given [예측: `data/20260930/0930_183320_observe/diag/validate_builder_shift.json`]; with width 0.007 included,
  0.03 m 0.37, 0.04 m 0.23, 0.06 m 0.16, 0.07 m 0.18 [예측: `data/20260930/0930_183320_observe/diag/validate_builder_shift_sweep.json`]. Build time 0.4 → 14.7 ms/frame
  [측정: same file `timing_ms`, 97 depth frames from the 0921 records]. The retry runs at 21:03–21:07 (`0930_210312`, `0930_210502_observe`, etc.) ran **with the switch off**
  (meta `policy.config.obs_view_forward_m` null, no `OBS VIEW SHIFT` log) — recorded dz<0 fraction 0.86. Turning the switch on for the obs of those 237 plans gives 0.32 (0.05 m),
  and 0.15 with the width 0.007 m also given [예측: `data/20260930/0930_183320_observe/diag/replay_2103.json`]. Water runs with it on: still 0.
- **Know before turning it on**: (1) it is **per process** — if you pick the can checkpoint in the panel while it is on, it applies there too. `0.05` only for peg sessions, `null` when they are over
  (`test_policy_obs.py` checks that the shipped config is `null`). Boundary: log/events.log `OBS VIEW SHIFT` at every launch·ARM, meta `policy.worker.obs.view_shift`.
  (2) obs age per plan grows by about 29 ms [유도: 2 frames × 14.7 ms]. In the 09-30 tracking runs 5–10 % were already late, and adding 29 ms to the same distribution gives 8–18 %
  [유도: `data/20260930/0930_183320_observe/diag/late_fraction_estimate.json`] — measure `age_at_intake_s` on the first run. (3) `depth_valid` in plans.jsonl is the post-fill value, so it is stamped as 1.0,
  and the per-frame fill count is not recorded. (4) `policy_session_replay.py`·`policy_yaw_vs_can.py`·`depth_compare.py` do not know about this switch — in shifted runs
  `obs/*.png` are the shifted images and `depth/*.png` are pre-shift. (5) The occluded area behind is a value filled with the background distance, not a measurement. Do not pool shifted and unshifted runs.
- **Settings that must go with it to take effect (not applied)**: proprio width levels for the peg ckpt — `gripper_width_open_m` 0.090, `gripper_width_closed_m`·`gripper_width_init_m` about 0.007
  (the current 0.069/0.042 differ from the peg data's width range 0–0.090 m). These values are shared with the can ckpt, so they were left untouched.
- **How to revert**: `obs_view_forward_m: null`. To remove the switch itself, delete the view shift block in `policy_obs.py`, the keys in `geometry.py`·`backends/policy.py`·`workers.py`,
  section 7 of `test_policy_obs.py`, and the README entry. If the route of matching it physically is taken (C3 5 cm forward or the gripper back, reference figure `data/20260930/0930_183320_observe/diag/jaw_position_reference.png`), this switch is not needed.

### Unplugging a USB camera during stream start **crashes the kernel uvcvideo and ties up that USB controller until reboot** (2026-10-01)
- **Symptom**: at 18:25:40, the moment the `./c3 cctv` camera process opened 3 cameras, the port 3-6 camera disconnected from USB (`usb 3-6: USB
  disconnect`), and 2 s later STREAMON on that device hit `BUG: kernel NULL pointer dereference` — `usb_ifnum_to_if ←
  usb_hcd_alloc_bandwidth ← usb_set_interface ← uvc_video_start_transfer`, Comm `python` (the camera process's capture thread)
  [측정: `journalctl -k` 2026-10-01 18:25:42]. After that `kworker/…+usb_hub_wq` was in D state, and the other camera(s) on the same controller (PCI `0000:13:00.0`,
  Bus 03) went into **permanent D state** in STREAMOFF (`usb_set_interface`)·format query (`uvc_v4l2_get_format`);
  even reading `/sys/bus/usb/devices/3-6/product` hung. The disconnected camera was picked up again 58 s later on another controller (Bus 01, `0000:11:00.0` port 2)
  (whether it was moved by hand or dropped off on its own at stream start is unknown).
- **Cause (partial)**: the module that crashed is not the stock kernel driver but the patched uvcvideo of `librealsense2-dkms` 1.3.28
  (`/lib/modules/6.8.0-138-generic/updates/dkms/uvcvideo.ko`, version `1.1.1-realsense-1.3.28`). In the race between the disconnect and STREAMON,
  the thread that died appears to have vanished while holding the controller's bandwidth lock [유도: D-state wchan]. Whether the stock uvcvideo has the same race
  is unconfirmed. This repo's code does not use RealSense, but librealsense2·pyrealsense2 are installed on the system (possibly for another project).
- **Impact on the station**: none — the cameras are a child process, so only that process hung, and the window shut down normally after the `stop()`
  timeout. Had it been the same process, the station itself would have become unkillable.
- **Mitigation**: reboot (D state is not released by kill). **Do not unplug cameras while streaming·right after start**. If there is no first frame for more than 15 s,
  the tile says "stuck … USB/driver fault?" (`pool_cam.py STUCK_S`).
- **Proper fix (not run)**: if RealSense is not used on this PC, consider removing `librealsense2-dkms` to go back to the stock uvcvideo
  (sudo; check the other projects first). If it is used, move to the latest dkms version.

### On station shutdown `cmd_estop` goes out **after the recorder joins** — if MPC is engaged, commands can keep going out in the meantime (2026-09-30, found in a safety audit)
- **Symptom (code reading, not reproduced)**: `rov_gui/window.py` `MainWindow.shutdown()` stops `cmd_timer`, then does the UI recorder join (≤5 s,
  `recorder.py` `ScreenRecorder.stop`), a join for each feed recorder that is on (≤5 s each), the mission log save and closing the nav records, and
  **after that** `cmd_estop.emit()` + `backend.stop()`. The teleop pump has stopped, so the sink deadman (500 ms) brings it to neutral, but
  if MPC is engaged, MpcWorker can keep sending `cmd_pilot` during those few seconds and feed the deadman [유도; the MpcWorker side was not read].
  The pool camera (`--pool-cams`) wait is already at the very end (after neutral), so it is unrelated to this problem.
- **Workaround**: press E-STOP (Esc) before closing the window — `estop()` sends MPC disengage + `cmd_estop` immediately.
- **Proper fix**: at the very top of `shutdown()`, `cmd_mpc_engage(False)` if `self._mpc_engaged`, then `cmd_estop.emit()` +
  `backend.stop()` **before** the recorder cleanup. The recorders·nav·mission log are independent of the backend, so only the order needs to change (needs checking with a test).

### If you fly while a training run is using the GPU, the policy plans are **all dropped as late** (2026-09-14)
- **Symptom**: in the three engagements of `data/20260914/0914_151901/`, the 329 `plans.jsonl` entries were **all `late`** (obs age at intake 0.84 s >
  `policy.obs_max_age_s` 0.60 s), `mpc_152047.meta.json policy.run`: received 75 / installed 0. Inference `infer_ms` p50 **629 ms**
  (p95 707, max 952) — 2.4× the 262 ms on hardware on 09-13. The reference never moved from the hold point (`rz` 0, `plan_id` NaN)
  and the DP did not drive the vehicle.
- **Cause**: the `tools/run_ablations.sh D_dinov3b` training (started 12:00:50) was occupying the same RTX 5090 at 99 % / 14.7 GB
  while the station inference (rovgui-pose) ran [측정: `nvidia-smi` 15:30, `ps` etimes]. The station does not detect or warn about this.
- **Countermeasure (not implemented)**: stop training before a pool session, or WARN at station launch if another CUDA process exists (since the `policy: READY` line
  already has the first infer_ms measurement, the minimal fix is to have it name GPU sharing as a candidate cause after 3 consecutive lates).

### [Temporary — to be deleted on 2026-09-13] The policy plan's z is pinned by `policy.z_hold_above_floor_m: 0.20` (2026-09-12)
- **What**: while `config/hw_mpc.yaml policy.z_hold_above_floor_m: 0.20` is on, **every knot z of the policy plan is pinned at 0.20 m above the tag floor** and the policy's own dz is discarded. Pilot request ("just for today, delete tomorrow") — so that in the yaw response test the descent (dz) of the handheld demonstrations does not drag the reference down to the floor.
- **Where**: `rov_gui/control/workers.py` policy intake (right after `compose_plan`, `msg.p_ned[2, :] = z_ned`), `rov_gui/control/geometry.py` default·validation (0 < h ≤ 1.5 m or null), `hw_mpc.yaml`.
- **Boundary**: log `ctrl: Z HOLD …` at every arm, `z_hold_applied` on every plan record, meta `policy.config.z_hold_above_floor_m`. **Do not pool the z metrics of runs with this key on and runs with it off.**
- **How to revert**: `z_hold_above_floor_m: null` in `hw_mpc.yaml` (or delete the line) → delete this entry. The code is harmless to keep (does nothing when null), but it can be deleted together with the test `test_policy_z_hold_pins_every_knot_and_says_so`.
- **Closed-loop caution**: in observe mode only the drawing changes, but flying without `--policy-observe` makes the NMPC actually track this z. The height's frame is "above the tag floor, positive = up", and the datum conversion is an offset by p0[2] (pinned by a test: p0 z −0.205 → datum +0.005).


### When a single-tag fix is consumed right after a fallback fix (Δt 0.8 ms), the velocity estimate spikes to 47.9 m/s, the state jumps 2 m and the wrench saturates at ±30 N (2026-09-14)
- **Symptom**: `data/20260914/0914_202238/mpc_202610.csv` rows 891–929 (t 44.6–46.5 s, 20:26:55): the moment the nav source switched from second (RGB fallback)→main (C3),
  state px jumped 0.214→1.369 m, 2.36 m/s coast → uX/uY/uN ±30/±30/±10 saturated for 29 ticks, solver status 4 (1 time), pwm 1351–1695 µs,
  batt_a 4.5–5.0 A, the hull was physically kicked (yaw +8/−16°, sway 0.27 m), followed by ~8 s of ringing [측정]. Smaller cases: `mpc_202951.csv` t 53.69 s
  (1-tag fix Δt 2.4 ms → 5.3 m/s, 0.24 m jump), `0914_181425/mpc_181425.csv` t 18.28–18.43 s (4 ticks ±30 N).
  The recorded fixes themselves (`nav_202611/fixes.csv` rows 1286–1288, x_ned) are normal; the jump exists only in the consumed STATE.
- **Cause (estimated, [유도] — the code path checks out, the numbers were not reproduced)**: `rov_gui/control/state_assembler.py:193-197`
  `v_new = (p_tag − prev_p) / dtc`, where dtc = the t_capture difference of the two fixes. The fallback fix's t_capture is the host arrival time, so when it comes in at almost the same
  time as the main fix but at a different position (1-tag PnP offset of a few cm), dividing by dtc≈1 ms makes v tens of m/s, and after the alpha 0.6 EMA
  `p += v·age` (velocity bridging, :200-204) moves the state by metres in one tick. Occurs only on a source switch/1-tag.
- **Workaround**: none — the longer the stretches of a policy run in which the C3 cannot see the tags (in 202610/202718, 56/64 % of the ticks were fallback), the greater the exposure. In fallback stretches,
  `tag_age_s` 0.03 s is based on the host arrival time, so do not trust it as a freshness indicator.
- **Proper fix**: (1) reset `_v_ned` on the first fix after a source change (or re-seed prev_p on the new source), (2) skip the velocity update when dtc is below a lower bound (e.g. 0.5·nominal
  frame interval), (3) clamp |v_new| at a physical upper bound (e.g. 0.5 m/s) + event log, (4) do not use 1-tag fixes for velocity updates.
  Regression test: inject a second→main switch with Δt 1 ms + a 4 cm offset and check that the state jump < 5 cm.

### Thruster outputs **froze for 1.4 s** and the vehicle turned 55° even under maximum counter-torque — a command-path hiccup (2026-09-14)
- **Symptom**: `data/20260914/0914_190226/mpc_190230.csv` t 3.62–5.03 s: the 8 reported PWMs stayed at `1554/1500/1400/1654/…`, **identical for 1.4 s** (normally updated every 0.149 s,
  telemetry fresh with `pwm_age_s` 0.11–0.22 s), while in the meantime the command axes moved `ax_yaw` +0.40→−0.50, `ax_sway` −0.38→+0.06 and `uN` was −10 N·m
  saturated. The vehicle yaw swung +14°→−42° (55°, up to 45°/s), then recovered as the PWM updated at 5.1 s [측정]. It was during the approach walk (sway −22 N), so the position was also pushed 0.5 m.
- **Frequency**: over the whole session that day, 6 windows of "PWM identical ≥0.35 s + axis command change >0.05" (0.35–0.75 s: `mpc_171212` 52.0 s, `mpc_172142` 1.9 s,
  `mpc_175628` 24.5 s, `mpc_175912` 18.6 s, `mpc_175227` 1.8 s; the 1.4 s one is this single case). None in the three 18:14 policy runs.
- **Cause (estimated, [유도])**: telemetry kept arriving and only the servo outputs did not change, so ArduSub did not receive new MANUAL_CONTROL and held the last input
  (tether/UDP drop or a stalled send thread). The control tick (`tick_ms` 2–6 ms) was normal, so it is not on the worker side. Had the 500 ms deadman fired, the output should have become 1500 (neutral),
  but the values were held.
- **Workaround**: none. If it hits during large horizontal thrust, as in the approach walk, the vehicle turns a lot. Run large-step experiments while a human hand is on the E-STOP.
- **Proper fix**: a "sent axes vs reported PWM" mismatch monitor on the sending side (e.g. WARN·event record when PWM is frozen ≥0.3 s + the axes change), and log MANUAL_CONTROL
  send timestamps/intervals in the CSV to tell whether a drop is on the send side or the receive side.

### The `--record-stereo` recorder is **pinned to the first engagement's tree** — after a LOW runtime switch, recordings get mixed into another tree (2026-09-11)
- **Symptom**: the fstereo worker of `HardwareBackend` arms `stereo_rec` only once (`_rec_started`, `rov_gui/backends/hardware.py`
  ~3402), and the folder is resolved once, at the first frame, through `run_dir_fn = mpc._run_dir`. Since 2026-09-11 LOW can be changed from the panel, so
  `_run_tree()` alternates between the in-water leaf ↔ the `_observe` leaf between engagements (a suffix since 2026-09-14; a separate tree before that), yet the raw stereo pairs keep
  piling up in the first folder — crossing the records boundary whose pooling guard is the kind separation. The depth recorder (`--record-depth`), which had the same defect, was fixed the same day
  with `PolicyWorker._rotate_recorder_if_moved` (closes and opens a new one when the folder changes, `meta policy.depth_record.previous`).
- **Workaround**: in `--record-stereo` sessions do not change LOW (if it must change, relaunch the station); if it was changed, separate the data by cross-checking the t_capture in `stereo/`
  against the times of each engagement's `mpc_*.csv`.
- **Proper fix**: give the fstereo worker a "new engagement" trigger (there is no PolicyState epoch, so the rising edge of `MpcStatus.engaged` or a bus
  signal), and add the same folder-compare rotation as the depth recorder. Re-arming in the same folder overwrites index.csv with `w`, so **only when the folder differs**.

### On hardware the DP is **unresponsive** to an object offset to the right, and on out-of-distribution scenes the yaw head drifts to a **left default of −1.4~−2.6°/plan** (2026-09-08)
- **Symptom**: in 0908_112101 (object +2.5~3.3° right of the heading, 0.9 m ahead) the policy turned slightly right for the first 18 s, then from 27 s on requested left −2°/plan,
  ending 11° left of the object bearing; in all 15 runs today the run-mean yaw intent is left (−2.0 ± 0.4°/plan)
  [측정: data/20260908/0908_112101/plans.jsonl, diag/out_regress.txt in the same folder].
- **Cause (ablation on top of an offline reproduction with r 0.92)**: (1) moving the can blob left by −20/−40/−60 px gives −2.6/−5.1/−6.3°, moving it right by +20/+40 px gives
  +0.2/−0.1° — a **one-sided response**, and the baseline is not the image centre but the jaw post (u≈150/224) [측정: diag/ablate_out2.txt]. (2) proprio all-identity, width, pairing,
  pitch, depth scale and left-right flip all give |Δ|≤0.3°, and any out-of-distribution image (even a left-right-flipped **training** image gives −2.2°) drifts to −1.4~−2.6° =
  the yaw head's OOD default; live images weaken the directional sensitivity ~3.5× relative to training [측정: diag/taskD_out_ablation_analysis.txt]. (3) In 112101
  the trigger was the vehicle floating up 0.3 m so that the floor receded (inverse-depth slope 0.52→0.26) — a synthetic plane alone, with no can or gripper, reproduces the same time series
  [측정: diag/ablate_timeseries.png]. The Newton jaw being in the image contributes ~0.5~0.7°. (4) The training set does contain steering (bearing→dyaw r 0.55),
  but it aims at the **fingertip (+4.1° = the handheld TCP lateral offset 0.0355 m)**, and the can-bearing p99 is +8.5°, so the +9~13° the ROV showed has 0 examples
  [측정: diag/taskB_out.txt]. The deployed ep195/model reaches yaw corr +0.64 and self-consistency +0.94 on held-out (it did learn; do not cite the ep10 EMA's "≈0").
- **Workaround**: keep the vehicle low and level (STAB, or resolve the deadband entry below) so the floor appearance stays within the training range, and start with the object near the fingertip set-point (+4°)
  or slightly left of it; never pool hardware yaw statistics with STAB runs (policy yaw not run).
- **Proper fix**: collect demonstrations with a large lateral offset (±15°) and observations in which the ROV jaw is visible, or mask the jaw region consistently; regenerate the labels with the TCP lateral offset matched to the ROV geometry (y=0);
  gate the policy output with out-of-distribution detection (floor slope, close-range blob position).
- **2026-09-08 verification correction (the part that passed the 2 refutation lenses)**: (1) "unresponsive to the right" is correct, but **it is not the hardware images' fault** —
  the same one-sided response comes out on training images too (the entry below, "policy yaw response is one-sided"). (2) "Out-of-distribution default" is only half
  right: the left bias at bearing 0° is **already in the demonstration labels themselves, at −1.85~−2.17°/plan** (the fingertip is 3.55 cm right of the lens),
  and the hardware images stack another −2.2~−2.7 **on top** of that. (3) **Nose-aligned is not out of distribution** — bearing 0° is at the 16.5th percentile of the training approach
  distribution (19.1 % within ±2°), whereas the +12° that was recommended as the aim point is at the 99.4th percentile, and above +13.6° there are
  0 training windows [측정: `.../diag/taskB_windows.npz` recomputed, `q1_refuteX_recompute_out.txt`].

### In policy missions the follower commands sit **inside the ESC deadband on all three axes**, so the vehicle drifts — `anchor: measured` cannot let the error grow (2026-09-08)
- **Symptom**: 0908_112101 ground speed 0.008 m/s (requested 0.047), rose +0.30 m (the policy requested descent in every plan); surge |ax|<0.096 100 % (uX p50 1.4 N),
  heave 92 % (uZ ≤5.9 N), yaw 96 %; vehicle-reported pwm_dev p50 10 µs, max 20 µs < ±25 µs [스펙] [측정: 0908_112101/mpc_112101.csv, diag/taskE_out.txt].
  The rise was not "against thrust" but a drift with thrust 0 + B−W ≈ +0.1 N, and the same day's MANUAL runs, with the same sub-deadband commands, floated at
  scattered rates of −0.006~+0.029 m/s (cannot be attributed to buoyancy, trim or the policy).
- **Cause**: with anchor measured + 1 s plans (0.5 s stale on arrival) the reference is re-anchored to the measured pose every 0.5 s → the tracking error stays bound to the plan displacement (xy p50 3 cm, z ≤10 cm),
  and with follower stiffness uX ≈ 35.6 N/m·e, uZ ≈ 64.9 N/m·e, reaching the 5.8~7.5 N threshold needs an xy error of 0.15~0.2 m, so it is never crossed. hold_tail track
  only raised |uX| in the plan segment from 0.05→1.21 N (1/4 of the deadband). In STAB the attitude loop wakes only the vertical thrusters, to 34~37 µs, so the rise is smaller (horizontally it still drifts).
- **2026-09-08 action taken (half)**: raised the prediction model's horizontal drag to the hardware value — `hw_mpc.yaml`
  `plant.linear_damping [86.7, 133.8, ...]` (= `axis_gain/0.692`, 21.5× the model). At a 0.05 m/s reference with error 0,
  uX 2.74 → 4.80 N (axis 0.046 → 0.080) [측정: offline solver].
  **But that benefit exists only while the reference is moving** — LQ position stiffness is sqrt(q/r), so drag
  does not enter it. Re-measured with the hardware geometry (error 4.7 cm, v_ref 0.047, **hold_frac 0.885**)
  it is only 4.62 → **4.79 N (+3.7 %)** [측정: .../0908_112101/diag/drag_sweep.py]: the plan covers only 12 % of the 3 s
  horizon and for the rest the reference stands still, which masks the feedforward. So the drag value
  is right now, but **the key to making the vehicle move is the hold tail / plan age**, and the deadband
  (axis 0.096 = 5.8 N) is not a drag matter — this block cannot fill it. Unverified on hardware.
- **A/B #2 (0908_165517, same scene, PID↔mpc_tuned)**: PID 0.73 m/40 s (uX p90 8.9 N, 30 % over the knee) vs mpc_tuned 0.01 m/45 s
  (uX fixed at 2.5–3.5 N, 0 %). The stall is the **proportional spring**, not the feedforward: uX ≈ 40 N/m × (displacement of one plan, 0.086 m) = 3.1 N is
  the LQ optimum, and the velocity feedforward is 0.56 N because 87 % of the horizon is v=0 hold. PID moved on kd·v_ref (6–8 N)+4 N integrator pulses at 73 per minute, and
  in the first 10 s, while it was receiving 0.029 m/s plans, it sat still just the same [측정: `.../0908_165517/diag/fwd_A_out.txt`].
- **Landed (2026-09-08, unverified on hardware)**: `policy.hold_tail: extrapolate` (extrapolates beyond the plan end at the last segment's velocity, clipped to v_max, 0.3 m and the box,
  live-only) + `policy.along_scale: 1.0` (along weight 75→300 only during policy missions, surge only). Offline closed loop (actual acados + the measured deadband
  law, that run's 88 plans): track gives 0.017 m/s even at q_scale 8; the deployed setting gives 0.055 m/s (64 % of the plan), error 1.3 cm, reversals 0.16/s
  [측정: `.../0908_165517/diag/fwd_C_out.txt`]. Remaining: on 0.04 m/s plans only 0.008–0.02 m/s (FF 2.5 N), and the STAB heading hold's T2/T4
  opposite-sign preload (until surge axis ~0.16 only one diagonal pair pushes) is not in the harness — a pool A/B should look at per-thruster pwm and yaw together.
- **Workaround (alternatives, operator decision needed)**: `anchor: leash` + `anchor_leash_m 0.10` (lets the error grow to 10 cm; refuter harness 0.050 m/s@0.085,
  0.029@0.040, holds up even at knee 0.13 — but the knot0 point leaves the hull and the plan is composed 10 cm ahead of the pose the policy saw, which reverts the 2026-09-07 measured
  decision), `policy.q_scale 4` (same surge, with sway 1200→4800 and heave 150→600 coming along), STAB engage (vertical only).
- **Proper fix**: deadband compensation at the allocation stage (minimum-PWM feedforward or a measured thrust map), or a per-thruster interface.
- **Unexplained observation**: the same day, MANUAL 110419 turned left at 1.6°/s with yaw command ~0 (|ax_yaw| p50 0.032), and STAB 105945 also turned left at 0.15°/s — there is an uncommanded left-turn
  torque (tether, thruster asymmetry?) that may contribute up to 3.6~6° of 112101's net 13° [측정: 0908_105822/mpc_110419.csv, diag/taskE_yaw_out.txt].

### The camera optical axis and the jaw are **laterally offset by 3.4~5 cm** — the CAD chain does not have this (2026-09-08)
- **Symptom**: in obs, the column centre of the close-range gripper blob is **obs u 150.7 / 150.8 / 151.3** in all three records
  (sd 0.2~1.3), whereas the CAD chain (`cam_t_flu [0.30584,0,0.10501]`, jaw `[0.4165,0,-0.17]` (x 0.502 since 2026-09-08, but only y=0 enters this column prediction), both y=0)
  predicts **optical-axis column 116.8**. The difference is +34.5 px = +10.3°. The camera roll measured from the floor plane (+3.0~5.9°) explains only
  4~8 px (12~23 %) of it, and the remaining +26~30 px implies a **lateral offset of 34~50 mm**
  [측정: `data/20260908/0908_112101/diag/q1_jaw_out.txt`, `q1_roll_out.txt`].
- **Why it matters**: the plan→NMPC conversion runs on the premise that `policy.tcp_offset_cam_m` is x=0 (jaw on the optical axis),
  and the left/right interpretation of the policy observation also rests on this premise. 3.4~5 cm exceeds half of the 62 mm jaw opening.
- **Not yet confirmed**: whether the blob centre is the jaw itself (or has the arm/mount mixed in) or a distortion of the FS depth has not been separated.
- **Next step**: measure it with a ruler held in the jaw, or put a target at a known position on the optical axis and check the obs column.
- **Correction to the 2026-09-06 entry**: "the gripper is on the opposite side (top) from land" has been **disproved** — in that capture the vehicle
  was tilted on the bench (the entry itself warned about it), and in the first real in-water frames the close-range row centre is
  **live 0.900 ± 0.020 vs train 0.898 ± 0.081, 100 % within the training range** (same
  `depth_compare.py:408-425` near_row) [측정: `.../diag/q1_ood_report_out.txt`]. The only remaining difference is not up/down
  but that it is **1.8× larger and 14 px to the right**, and that is the entry above.
- **Cross-reference (2026-09-08, a candidate explanation that came out of the jaw measurement)**: the obs grid is the rect_LEFT mono grid, and that camera sits
  37.5 mm away from the colour camera (data/20260908/0908_180453_observe/mpc_180453.meta.json
  `fstereo.rig.color.t_rectleft_to_color_mm` = [−37.5, …]; the same run's policy_obs/meta.json does not have this key); the axis of the bottle held in the jaw,
  measured by colour-camera PnP, is **0.011 m to the right** of the vehicle axis [측정: data/20260908/0908_180453_observe/policy_obs/rgb/000160.jpg,
  same in frames 0/80 of the same run]. 37.5 + 11 ≈ 49 mm fits the 34~50 mm above — only a candidate, not settled (the sign of the baseline
  direction and whether "blob = jaw" are unconfirmed).
- **2026-10-05 update — the baseline sign has been confirmed (1 vote from a verification agent, [유도]).**
  - The ColorRig convention in `c3_camera/host_depth.py:298-305` is `P_color = R·P_rectleft + t`, with t_x −37.55 mm. So the colour centre is
    **37.5 mm to the right** of rect-left. The q1 analysis placed the rect-left centre at the colour lens (cam_t_flu), so most of the 34~50 mm above is this
    half-baseline. Removing it, the jaw referenced to colour is about 1.1–1.5 cm right of the lens (bottle-axis PnP; −0.4~+1.3 cm depending on the FS scale assumption).
  - So the error of `tcp_body_flu_m` y = 0 [예측] is of the same order of magnitude as the lateral margin of about ±11 mm.
  - The same frame mixing is also in the jaw mismatch relative to the handheld. The handheld `tcp.offset_m` is a rect-LEFT frame value
    (`~/Desktop/data collection/slam/grasp_9_4_26/actions_summary.json` config.tcp._comment), while the ROV jaw is a colour frame value.
    Aligned to rect-left, the jaw is about 1.3–1.7 cm **to the right** of the fingertip. That is the opposite direction from the prediction 'first grasp misses about 3.6 cm to the left' [유도].
  - Remaining check: clamp a small tag or cylinder in the closed jaw and measure the lens→jaw lateral offset with the same colour-frame PnP over several frames.

### The x/z of the C3 extrinsic cam_t_flu are suspected — the lens may be ~9 cm further back and 5 cm higher (2026-09-08)
- **How it was found**: in 0908_180453 (POLICY OBSERVE, vehicle at rest for 44 s, over the mat, nose 5.3° up) the pilot placed the bottle held in the jaw
  on the centre of floor tag 58, but the on-screen jaw ring was drawn 0.138 m behind tag 58 along the heading
  (yaw-only marker; 0.123 m in 3-D with the nose-up included) [측정: data/20260908/0908_180453_observe/
  mpc_180453.meta.json datum + re-analysis of policy_obs/rgb/000160.jpg, config/tag_map_full.yaml tag 58; the re-analysis scripts, projection figures and outputs are in the same run's diag/ (solve_frame.py, jar_geom.py, pixskeptic_refined.py, proj160_zoom.png, *_out.txt)].
- **The measurement stands; the attribution is open**: the tag map, PnP (rms 2.0 px; stable to 1 cm along the heading even with the tag subsets {48,59}/{59}),
  tilt (rp_residual 2.4°), vertical (lens 0.343 m above the mat vs CAD chain + nose angle 0.350 m, 7 mm) and even the marker
  transform (body_to_map) all reproduce [측정: data/20260908/0908_180453_observe/mpc_180453.csv
  `pnp_rms_px`, `rp_residual_deg`, `n_tags`; the subset re-analysis and the lens height 0.343 m are a re-analysis of the same run's policy_obs/rgb/000160.jpg,
  0.350 m = CAD 0.322 + 0.306·sin 5.33° [유도]]. What was disproved is **a single CAD offset, "the lens is 110.66 mm behind the jaw centre"**
  — that jaw projects to row ~420 of the 360-row frame, but the jaw is visible at rows 265~360, and the axis of the held bottle is 0.196 m in front of the lens
  (frames 0/80/160: 0.196/0.194/0.196, range 0.187~0.204) [측정: the frame above + 0908_170428/policy_obs/rgb/000000.jpg
  (bottle height 0.084 ± 0.006 m) + 0908_175151/policy_obs/rgb/000000.jpg (open-jaw rays: fingertip 0.215~0.234, palm 0.165~0.180)].
  There is also a height-independent verification: the apparent width ratio of the held bottle's cap (60 px vs 35~36 px at a distance = 1.67~1.71) matches the hypothesis from the measurement (1.66)
  and rejects the CAD jaw (2.38). But this reads two ways — **(a)** `cam_t_flu` is right and the jaw is at 0.502 ahead of the COM;
  **(a')** `cam_t_flu.x` is ~0.09 m too large (actual lens x ~0.21~0.23), z is ~0.05 m too low (actual ~0.155), and the sim jaw
  0.4165 is roughly right (0.22 + 0.196 = 0.42). Both readings put the jaw at **lens + [0.196, 0, −0.275]**, so the jaw/TCP was
  fixed by anchoring it to the lens (`rov_shape.LENS_TO_GRIP_FLU_M`, `policy.tcp_body_flu_m` 0.502), and this entry is **the remaining half**.
- **Circumstantial evidence pointing to (a')** (all indirect; nothing has measured this key): (1) `second_cam.t_flu [0.2663, …]` is a value BodyAlign measured
  **relative to the C3 chain**, while in the sim hull mesh (base_link frame) the dome apex is at x 0.230~0.231 and the end of the electronics
  tube at ~0.176 [유도: vertex ranges of bluerov2_mujoco_marinegym/meshes/rov_body_white.obj / rov_body_black.obj]
  — the dome camera lens cannot be 3.5~4.5 cm ahead of the dome apex, and the pilot's tape gave 0.18 (the
  "8.6 cm short" comment in hw_nav.yaml). (2) In pure yaw pivots gated on pilot commands (per-thruster PWM, surge/sway ≈ 0) **the reported
  body origin is 0.117~0.130 m behind the yaw pivot** [측정: data/20260908/0908_170428_observe/mpc_170428.csv
  n=192, 0908_175151/mpc_175151.csv n=43; script and output data/20260908/0908_180453_observe/diag/skeptic_lever_pwm2.py, yaw_pivot_out.txt]; if the pivot is the composite COM (+0.035), (a') predicts −0.13
  and the current value predicts −0.035. (3) Vertical: BodyAlign's dome camera z −0.019 vs the mesh's electronics tube axis ~+0.03,
  the chain's underside −0.217 vs the mesh's −0.165~−0.167 (white/black) [유도: same mesh] — the chain is ~5 cm lower, and its anchor (jaw z −0.17) is a sim
  estimate. (4) Under (a'), the CAD's 110.66 mm reads as the distance to a point ~9 cm behind the grip point, i.e. to the jaw **base/pivot**, which makes it not a "wrong
  number" but a "different point".
- **Consequences if (a') is right**: the reported body origin is ~0.09 m behind the actual one along the heading (0.085 if the sim jaw 0.4165
  is exact; 0.076~0.096 if lens x is 0.21~0.23 [유도]) and ~0.05 m below → **every recorded
  position** is shifted by that much, the gap between the point the NMPC regulates and the plant COM is not the known 0.043 m but ~0.13 m
  [유도], and the marker's hull is drawn ~9 cm behind where it actually is (the jaw is right). The sim GRIP_POS/JAW_POS are roughly right under (a') and,
  under (a), 8.5 cm behind the lens-anchored jaw — the plant composition was not touched.
- **Why it was not changed / Workaround**: this value is the records boundary for every recorded position (hw_nav.yaml comment), and the circumstantial evidence above is all
  indirect. Only the jaw/TCP was anchored to the lens (that is all). When citing absolute positions at the cm level, cite this entry alongside.
- **Proper fix**: two tape-measure readings on the vehicle — (1) frame front end (dome apex) → C3 lens: under the current value the lens is ~7.5 cm **ahead** of the dome
  apex, under (a') at the dome apex ±2 cm; (2) C3 lens → gripper palm. If it is changed, in the same commit
  re-derive `policy.tcp_body_flu_m = cam_t_flu + [0.196, 0, −0.275]` (or set `tcp_offset_cam_m` so that the lens position
  cancels out), and meta `hardware.cam_t_flu` records the boundary.
- **2026-09-09 CAD re-export evidence (on the (a') side)**: the user's Onshape document "BlueROV2", tab "mujoco" (with the camera, upper mount and Newton
  gripper), was re-exported with onshape-to-robot and placed in base_link with the July (07) registration (R0, C_ASM; the hull part poses are byte-identical to July (07))
  [유도: assets/CAD files/onshape_export_20260909/payload_frames_20260909.json; the script uses
  the same registration constants (tools/process_c3_mesh.py)]. What the CAD says: **lens-face centre [0.194, 0.006, +0.133]**, optical axis 40.0° down,
  baseline horizontal; **jaw-pair centre [0.364, 0.006, −0.127]** (jaw x 0.329~0.404); lens→jaw centre [+0.170, 0, −0.260].
  Relative to the current `cam_t_flu` [0.306, 0, 0.105] the lens is **x 0.11 m back, z 0.03 m up**, and relative to `tcp_body_flu_m` 0.502 the jaw is
  0.10~0.14 m back — the same direction as the (a') prediction (lens x ~0.21~0.23, jaw ~0.42), and the lens-jaw relative value (0.17~0.21 m ahead)
  agrees with the image measurement 0.196. Limitation: the origin is the COM of the July (07) sim-skin registration, and the CAD's gripper/mount placement is the user's
  modelling (instance Unfix/Drag edit history in the mujoco tab until 20 minutes before the export), so it **does not replace a tape-measure measurement**.
  - **Re-checking the 0908 evidence against the CAD** [유도: assets/CAD files/onshape_export_20260909/reanalysis_20260909.json]: (1) yaw pivot —
    the CAD predicts that the reported origin is **0.111 m behind** the COM; measured 0.117~0.130 (within 0.6~1.9 cm). (2) Dome camera — BodyAlign 0.266 −
    0.111 = 0.155, within 2~3 cm of the CAD hull's foremost dome-region x 0.208 (overall foremost 0.226) and the tape's 0.18. (3) Jaw — lens +0.196 = 0.39,
    CAD jaw centre 0.364, tip 0.404, sim 0.4165 → the current 0.502 is 0.11 m ahead. (4) **Vertical goes the opposite way**: from the CAD hull's lowest point z −0.164
    (matching the sim mesh −0.165~−0.167) the lens is 0.297 above the floor at rest and 0.325 m with the nose 5.33° up, but the 0908 measurement is 0.343
    (0.359 if the floor is at z −0.016) — the CAD is 2~3.5 cm **lower** (the old chain's 0.350 is 0.7 cm off). So the lens z may be +0.15~0.16
    [유도], higher than the CAD's +0.133, and either the CAD placement of the upper mount was drawn that much too low or it is a mat/pitch assumption problem. Conclusion: horizontal converges to (a')
    (x −0.11), vertical is undecided between +0.03~+0.05 — **decide with a tape measure**: dome apex→lens; under the CAD the lens is 1~3 cm
    **behind** the apex, under the current chain 8~10 cm **ahead**.

### The policy's yaw response is **one-sided** — an error to the right can never be corrected (2026-09-08)
- **Symptom**: the knot-15 dyaw measured while moving the can left and right responds at **+0.15~0.25°/° on the left** of the aim point
  but dies to **+0.02~0.05°/° on the right**. A flat region with |dyaw| ≤ 1° runs from +8° to at least +20°.
  Running the same sweep on **training images** gives the same shape (left +0.146, right +0.020), so it is not the hardware images' fault but
  **a property of the policy** [측정: `.../diag/q1_sweep1_out.txt`, `q1_sweep_train_out.txt`].
- **An added constant**: at bearing 0° (nose aligned) the demonstration labels themselves are −1.85~−2.17°/plan left (because the demonstrator's fingertip being 3.55 cm right
  of the lens = +4.1° is the labels' zero), the policy on training images gives −2.32°, and **on hardware images −3.7~−4.97°**
  — i.e. the hardware images stack another −2.2~−2.7°/plan on top [측정: `taskB_windows.npz` recomputed, `q1_sweep_train_out.txt`,
  `q1_sweep1_out.txt`].
- **Trap**: the equilibrium of the hardware closed loop (command 0) is at **bearing +12~18°**, but that interval has **0** training windows (bearing > +13.6°).
  The policy is quiet there not because it is right but **because it has stopped responding**. Conversely, nose-aligned (bearing 0°) is
  **at the 16.5th percentile, inside** the training distribution — putting the can on the right is not a repair but a **numerical cancellation** of the bias.
- **Workaround**: for the first test put the can at **bearing +4°**, the centre of the demonstration distribution (5~7 cm to the right at 0.7~1.0 m),
  and hold altitude to keep the can within the observation band (0.2~3.0 m). Even then, about −3.7°/plan remains.
- **Proper fix**: (a) regenerate the labels with the ROV's TCP (jaw on the optical axis, including the measured offset from the entry above) and retrain,
  (b) collect demonstrations with a large left/right error (±15°), (c) as a stopgap, shift the intake crop by +34.5 px (at bearing 0, −4.66 → −0.98°/plan,
  but since it responds only to the absolute column, it only buys as much as it is shifted [측정: `.../diag/q1_sweep_shift2_out.txt`]).

### The deployed EMA weights are **1.9× worse** on held-out — the cause is BatchNorm running stats (2026-09-06)
- **Symptom**: on hardware on 2026-09-03 the DP reference was indecisive. The `state_dicts.ema_model` loaded by that checkpoint
  is **much worse on the validation set than the non-EMA `state_dicts.model`**
  [측정: 4 validation episodes (6/32/48/56, `get_val_mask(75, 0.05, 42)`), 219 windows, DDIM 16 steps]:

  | Weights | action MSE | pos MSE | pos RMSE |
  |---|---|---|---|
  | `ema_model` (**currently deployed**) | 0.001582 | 0.000956 | 30.9 mm |
  | `model` (non-EMA) | 0.000828 | 0.000435 | 20.8 mm |

  The same holds in the hardware setting (DDIM 8 steps): 0.001535 vs 0.000842. That is, **1.82–1.91×**.
- **A decomposition experiment confirmed the cause**: keeping the EMA copy's **parameters as they are and swapping in trained values for only the 108 BatchNorm
  buffers** takes 0.001535 → **0.000896** (pos RMSE 20.9 mm),
  **recovering 92%** of the gap [측정 2026-09-06, same 219 windows]. So **the EMA averaging itself
  is fine, and the statistics were everything**. The reading "the EMA trained badly" was wrong.
- **Cause (confirmed)**: `EMAModel.step` iterates only over `named_parameters()`, so it **does not average
  the buffers**. And because of `if use_group_norm and not pretrained:` in `transformer_obs_encoder.py:120`,
  `pretrained: true` meant that, even with `use_group_norm: true`, **the GroupNorm substitution was silently
  skipped and BatchNorm survived**. As a result, the EMA copy's `running_mean/var` are
  **still the timm ImageNet values** [측정: `num_batches_tracked` — timm `resnet34.a1_in1k`
  pretrained **374,981**, ckpt `ema_model` **374,983** (+2), ckpt `model` **395,128**
  (= +20,145 = global_step+1)]. The maximum model↔ema relative difference in the BN buffers is **9847×**.
  In eval() BN uses the running stats, so **the vision encoder is normalizing the depth observations with ImageNet RGB
  statistics**.
- **Workaround (implemented 2026-09-06)**: `--policy-weights model`. The default stays `ema_model`, so
  earlier records reproduce, and the chosen value is left in `describe()` and the run meta — **do not pool runs that ran
  on different weights**.
- **Proper fix**: (1) ~~expose the weight source via config/CLI~~ done; (2) when retraining, actually replace the encoder's
  BatchNorm with GroupNorm (= remove the `pretrained` gate) or make the EMA copy the buffers
  too; (3) redo the selection on the **full validation set** — the supporting numbers in the current `selected.json` come from only
  one batch of `next(iter(val_dataloader))` (32 windows), which was 1.4× more optimistic than the full set.
- **The 2026-09-07 5-dim action retraining (`umi_depth_5d`) left (2) untouched by user decision and so inherited the defect,
  which made the same-epoch evidence sharper**: at epoch 195, the 219-window held-out pos RMSE is
  **ema_model 33.0 mm vs model 17.3 mm** (MSE 3.6×), yaw RMS 2.87° vs 2.20°
  [측정: `<run>/heldout_pos_yaw_width_ep0195{,_model}_stride4.log`]. The old 10-dim ckpt also: 29.8 vs 21.0 mm.
  → **From 2026-09-07 the deployed default weights were switched to `model`** (`geometry.default_policy_block()['weights']`,
  `policy.weights: model` in `config/hw_mpc.yaml` and `land_dp.yaml`; can be overridden with `--policy-weights`).
  The parity fixture is also rebuilt with the deployed weights (`dp_policy_reference.py --weights`, whose default is the deployed choice).
  **Do not pool** with earlier runs that ran on `ema_model` — that still stands.

### The NMPC plant believes the vehicle **sinks with 5.7 N**, but on hardware it floats up — mpc/mpc_tuned keep rising under the policy's descending plans (2026-09-07)
- **Symptom**: all 5 `mpc_tuned` + policy missions drifted upward (pz rose 0.27–0.54 m, vz p50 +0.015…+0.035 m/s) — the policy
  requested +0.055 m (p50) **downward** per 1 s (NED +down, 154 plans), and with an anchor z offset (reference−measured) of p50 +0.108 m the reference was always below the vehicle
  [측정: data/20260907/0907_180038/mpc_18*.csv, plans.jsonl].
- **Cause**: regressing the heave command gives `uZ ≈ −5.6 N − 60 N/m·ez` for all five missions (R² 0.54–0.94; ez = rz − pz, FLU). **The intercept −5.6 N (upward)**
  matches `dobmpc/params.py NET_BUOYANCY = −5.71 N` (a CAD/vendor composite value saying heavy_gripper sinks; 0 measurements) — the nominal NMPC cancels its own model's
  gravity, so at error 0 it pushes 5.7 N **upward**. Add the T200 deadband (~5.8 N) on top of that, and getting downward thrust needs ez ≈ −0.19 m,
  but leash 0.05 m + the plan's descent span bind the error at −0.10…−0.15 m → thrust 0 (|ax_heave| < 0.096 on 88–100 % of ticks), and it rises by whatever the hardware buoyancy
  is. The first 1.4 s of a mission (station hold) also has uZ −5.1…−6.1 N (up). The hardware B−W moved within the same day from −2 N (16:53 mpc hold, −6.5 N command, flat depth)
  to > +3 N (16:07, rising 0.25–0.49 m even with a PID +8 N command) and was never −5.7 N [측정: 0907_16*/mpc_*.csv, 0907_15*/].
  The same mechanism is also present in 4 mpc_tuned missions at 16:11–16:13 (0907_160750/mpc_1611{15,49}, 1612{38}, 1613{15}: +0.22…+0.53 m).
- **Mitigation (2026-09-07, unverified on hardware)**: `hw_mpc.yaml`/`land_dp.yaml` **`vehicle_net_buoyancy_n: 0.0`** (hardware B−W, + = buoyant) → `HwDobMpc`
  puts (model − hardware) = −5.71 N into the acados parameter p (= w) as a **constant heave wrench** (runtime input, no rebuild, not an observer; mpc/mpc_tuned only —
  not applied to dobmpc, because its EAOB estimates the same residual). The meta `controller.heave_trim` block is the records boundary (a run without it = a run flown with the −5.7 N upward FF);
  CSV `w2` stays the EAOB column. Model-plant prediction: a neutral vehicle in station hold comes to rest +12.2 cm high uncorrected, 0.00 cm corrected
  [예측: rov_gui/tests/test_path_cost.py::test_a_neutral_vehicle_holds_depth_only_with_the_trim, plant without a deadband]. Verify by whether the intercept
  of CSV `uZ` (mean over at-rest ticks) is near 0, and by the pz drift in policy missions.
- **Measured (2026-09-08)**: fitting the plant heave model to a free-rise window in which the heave command is inside the deadband (0908_105822/mpc_110419.csv 0–11 s, pwm_dev ≤ 22 µs)
  gives B−W = **+0.12 N** (model drag), +0.04 N (drag 0), +0.28 N (drag 3×) [측정+유도: 0908_105822/buoyancy_fit_20260908.txt] → the vehicle is
  effectively neutral, and the config was set to `vehicle_net_buoyancy_n: 0.1`. The same day's MANUAL runs still floated up 0.25 m/15 s, but that is not the buoyancy constant but
  the **deadband** (thrust 0 until the error reaches ~0.1 m, where the position term becomes ~6 N). By contrast, 2 STABILIZE runs (mpc_105945/110306) held depth
  within ±7 cm with a heave command of ~0 — the attitude loop keeps the vertical thrusters standing outside the deadband (pwm_dev p50 35 vs MANUAL 10 µs).
- **Remaining**: the trim is a **constant**, so it cannot follow the hardware buoyancy (tether, trim), which moves ±8 N within a day — the mpc family has no integrator or observer (user decision),
  so the only option is to measure the value every session (the steady uZ of the PID depth hold, if it is outside the deadband), and a buoyancy error inside the 5.8 N deadband still remains
  as a ~0.1 m offset (it shrinks if `policy.q_scale` raises the stiffness [유도]). Irrelevant under `--policy-observe` (not transmitting).

### Engaging in STABILIZE — wired but **0 water runs**, and yaw is `hold`, so the station cannot turn the heading (2026-09-08)
- `engage.require_mode` takes a list and can be opened up to `[MANUAL, STABILIZE]` (pinned, `mode_settle_s`, dobmpc refused,
  `stabilize.yaw_axis: hold`; README "Engaging in STABILIZE"). The verdicts of all three consults (control theory, underwater operation, policy observation) are
  **conditional**: roll/pitch is a benefit, yaw is a loss because its meaning changes from torque to rate. The review recommended MANUAL-only as the default, but
  **by operator decision the default is `[MANUAL, STABILIZE]`** (2026-09-08): STAB button → START runs as-is under STAB.
- Unresolved: (1) under `hold`, heading_follow, circle and policy dyaw do not work, and with `torque`, hunting at the deadzone (axis ≤ 0.10) boundary is
  [예측]; doing it properly needs an N→yaw-rate reinterpretation + deadzone inverse compensation, but that is not implemented because PilotGain (the entry above) cannot be read (`rate` refused).
  (2) `mode_settle_s 2.0` is [예측], and the gamepad's mode buttons go straight to the vehicle where the station cannot see them, so a switch over that path is noticed late, by the HEARTBEAT (1 Hz)
  delay (the panel path stamps the clock at request time). (3) The neutral that E-STOP/deadman sends is not a stop in STABILIZE (level and heading
  hold, thrusters active) — on E-STOP it only reports via an error log, and an automatic switch to DISARM/MANUAL is a matter for a separate safety review.
  (4) Firmware version unconfirmed — on 4.1.0 the STABILIZE attitude target locks to 'the attitude at the last yaw input', breaking the premise. (5) The vertical thrusters
  are shared with the roll/pitch PID, reducing heave authority in the contact phase [유도]. (6) No STAB status chip/tooltip in the station UI (only the refusal message).
- The first-session procedure and the measurement items are in the corresponding README section. Until those artifacts exist, do not pool STABILIZE-run numbers with MANUAL runs or cite them.

### 7d LOW followers (`PID_7D` / `MPC_Tuned_7D` / `RL_7D`) have **0 water runs** — the PID_7D gains rest on unmeasured delay and restoring stiffness, and RL_7D has no trained policy (2026-10-03)
- **What**: three LOWs that also control roll/pitch (README section "7d LOW followers"). K/M are sent only with a `*_7d` LOW + `engage.attitude_axes.enabled`.
  Everything left unconfirmed in the 6-DoF variant entry below — firmware s/t, sign, plant — applies here unchanged.
- **The PID_7D gains are [예측], and the stability margin hinges on the restoring stiffness k**: ωn 0.5 rad/s, Kp/Kd pole placement, Ki 0.15 and FF 1.35
  give PM ≥ ~50° at a total delay of 0.2 s and an actuation gain of ×0.4 to ×2.5 only when k ≤ ~2.3 (roll)/3.6 (pitch) N·m/rad; with k ≈ 20 (fit to RL-side logs),
  PM is 3–41° [유도: `rov_gui/tests/fixtures/sim_pid7d_attitude_20261003.txt`]. The actual k, delay, actuation gain and damping are not measured.
  **Workaround**: before closing the loop in water, do 1 pulse-and-release (open-loop pitch 0.15 axis for 0.5–1 s → hold → release; initial direction = wire sign,
  time to the first gyro response = delay, initial angular acceleration = g/I, static tilt = g/k, free-decay period · logarithmic decrement = k/I · d/I). If the free-decay period
  is ≥ 3.2 s, the current set; if 1.9–3.2 s, after gain correction; if < 1.9 s, Kp = Kd = 0 + FF 0.8·k̂ + a small Ki (control-theory-advisor consult,
  `.claude/journal/consults.md` 2026-10-03). **Proper fix**: re-set `pid_7d.rp_k_model/rp_k_ff/rp_wn` from that measurement.
- **PID_7D's "authority exhausted" verdict rests on [예측] thresholds** (residual of the HIGH fix from the 2026-10-03 safety audit): the integral stops at 0.8× the wire torque,
  so the axis does not reach the cap; HwPid7d therefore reports "authority exhausted" (command clip / integral limit + error > `rp_pin_err_rad` 1° / error ≥ gate 20°)
  and interlock (iii) treats it the same as being pinned at the cap — a flipped sign or an unreachable tilt (stiff hull) disengages after 1 s (intended).
  Remaining risk: a **false disengage** when tracking is normal but a large static moment drives the integral to its limit and the error slightly exceeds 1° (the 1° is [예측];
  does not occur on the demo pendulum, unconfirmed on hardware). Conversely, a lack of authority in which the error stays at or below 1° is not caught.
- **A pid_7d hardware engage requires a pulse record**: the minimum of `pid_7d.pulse_test` (a JSON the operator writes during the test, `free_decay_period_s
  [roll, pitch]`) ≥ 3.2 s, or the recorded bypass `require_pulse_test: false`. No tool writes this record (manual). Measurements other than the period
  (delay, g/I, damping) are not read by the gate — re-setting `rp_wn/rp_k_ff` from those values is a human's job.
- **The reference attitude is stamped in the record one tick early** (same ordering as HwDobMpc): the D5 ramp decays inside step and the worker reads `ref_attitude_ned_at`
  after that, so during the ramp the CSV `rroll/rpitch` and the reference of (iii) are one tick (≈1°) ahead of the value actually used. Fix: snapshot before step.
- **RL_7D's roll/pitch slew (1.5/s) is an actuator that was absent in training**: applying the first-frame ±cap guard (2026-09-26 audit fix) to RL too
  is a safe-side choice, but the previous action fed back may be out of distribution (same family as the rl_pwm cap-clip limit cycle [예측]). To be checked in the first test after training.
- **Verdict mismatch when flying on EKF/DR (not reviewed)**: with `ekf_control`/`dr_control` the 7d loop closes on the `meas_ctrl` attitude, while the ceiling and (iii) judge on `meas`.
  Since roll/pitch are IMU-based in both, the difference is expected to be small, but it was outside the audit's scope.
- **RL_7D has no trained export** (operator decision: code only, training deferred). `config/hw_mpc.yaml rl_7d.policy_dir: null`, so the mode
  is refused with a reason. Training: RL_controller `scripts/_pipeline_att7_seed.sh <seed>` (preset `traj10att`, axes6, 1200 it;
  3 seeds in parallel ≈ 32 min [유도: 3 traj10 runs in parallel 32:06–32:14, `RL_controller/logs/train_traj10_s*.log`]) → export → station
  `rov_gui/control/rl_policies/<name>/`. Even after training: (a) with first-water caps lower than the training caps (0.5/0.75 in policy units = 0.2/0.3 at the station),
  engage is refused (get the full caps with a sign probe, or use the recorded bypass `rl_7d.allow_caps_below_trained`), (b) the RL-side evaluation scripts (eval_sweep/
  eval_trajectory/summarize_results) still have no roll/pitch reference steps or metrics — there is only the training curve `/att/err_full_deg`.
- **RL_PWM (pwm10_s1) cannot track a roll/pitch reference**: it was trained only with a level q_des (active level hold only). To use it with a 7-dim policy,
  the att7 recipe has to be trained in pwm mode, and the station's `HwRlPwm` also has to carry the attitude in q_des like HwRl7d (not implemented).
- **Yaw-rate FF spike in the existing RL followers (common to rl/rl_pwm/rl_7d, found in the 2026-10-03 investigation)**: `HwRl.observation`
  updates `_yaw_ref_prev` only on ticks where r_ref == 0, so on the first hold tick after an r ≠ 0 plan segment goal_yaw_rate jumps to −0.6 rad/s for one tick
  (code behaviour, not a measurement). Fix: update `_yaw_ref_prev` every tick — kept out of this change so that rl/rl_pwm records stay comparable.
- **Replay attitude tracking (`replay.track_attitude`) has weaker guards than the policy** (pre-existing): no T1 clip, rp FilterLimits or div_rp; only T3 (30°).
  Applies when a replay carries attitude on a `*_7d` LOW.
- **Meaning change = records boundary (2026-10-03)**: `engage.attitude_axes.enabled` means "station capability", not "send K/M", and `policy.attitude_track` is
  a veto, not a switch (default true). Runs that ran with this switch from 09-26 to 10-02 (0 in water, sim/demo only) have no meta `run.attitude_axes.low`.

### The 6-DoF variant (roll/pitch tracking, `pos_rpy_width` + `engage.attitude_axes`) has **0 water runs** — firmware s/t consumption, sign and plant all unconfirmed (2026-09-26)
- **What**: the variant in which the policy outputs the 7-dim `[dx,dy,dz,dyaw,droll,dpitch,width]` and the NMPC tracks the roll/pitch reference, sending K/M torques
  over the MANUAL_CONTROL extension axes (s = pitch, t = roll, `enabled_extensions=0b11`). All defaults are OFF
  (`engage.attitude_axes.enabled: false`; since 2026-10-03 K/M are sent only from a `*_7d` LOW and `policy.attitude_track` is a veto,
  default true — 7d LOW entry above; since the same day `policy.action_repr` has the deployed value `auto` = the contract of the checkpoint picked in the panel is pinned at ARM,
  while the explicit values `pos_yaw_width`/`pos_rpy_width` are pinned per session as before),
  and when OFF the 4-DoF path is byte-identical (pinned by tests). Design and first-flight order: `docs/DP_6DOF_PLAN.ko.md`. Below is what remains before turning it on.
- **Unconfirmed whether the firmware reads s/t, and with which sign**: the ArduSub version of this ROV is itself unconfirmed (STABILIZE entry above, (4)), and s/t are
  consumed only from Sub 4.1.2 (2024-02-22) on — on 4.1.0/4.1.1 they are **silently ignored**. The original source of the s/t lines in `joystick.cpp` (channel index, rpyScale, sign,
  pitchTrim/rollTrim latch) has not been read [스펙 unconfirmed]. The premise of the engage gate (mavlink20 + AUTOPILOT_VERSION ≥ 4.1.2 + parsing the artifact JSON of the bench probe
  `python -m rov_gui.tools.attitude_axes_probe`: `tool`, `pass: true`, `mavlink_wire_version "2.0"`, `firmware_version` matching
  the vehicle) and of the disarmed probe, "RC_CHANNELS chan1/2 in the disarmed state reflect the MANUAL_CONTROL
  override", is also [스펙 unconfirmed] — if that does not hold, the probe has to be run armed, in water.
- **Extension axes stated as 0 in neutral frames**: after an extension frame (`enabled_extensions=0b11`) has been sent even once, subsequent neutral frames are sent with `enabled_extensions=0b11`
  + `s = t = 0` stated **explicitly** (whether the firmware keeps the last s/t or reads them as 0 in a frame without the bits is [스펙 unconfirmed], so stating 0 explicitly
  was chosen). The firmware behaviour when the bits are absent is still [스펙 unconfirmed].
- **K/M sign [assumption]**: `engage.attitude_axes.sign {roll: 1, pitch: 1}` is an assumption. If it is flipped, positive feedback up to U_MAX 8 N·m
  (restoring ~2.1 N·m/rad [유도] → capsize possible). The lines of defence are the first-water caps `first_water_caps [0.1, 0.15]` (≤ 1.35 N·m/rad restoring, against which capsize is impossible
  [유도]) + the sat-ineffective interlock (disengage if the axis stays stuck at the cap for 1 s and the error does not shrink; constants [예측]) + an **armed in-water sign probe**.
  The sign-probe mission kind (`attitude_sign_probe`) is **not in this cut**: the arm gate requires, in the `engage.attitude_axes.sign_probe` JSON,
  `sign_proven: true` (`require_sign_probe: true` by default), so for now that file has to be made by hand (record ATTITUDE + SERVO_OUTPUT_RAW,
  roll +0.10 for 1 s → pitch +0.10 for 1 s, abort at |rp| > 10° or 2 s) with `sign_proven: true` written into it. Bypassing with `require_sign_probe: false`
  lets arm go through, but `first_water_caps [0.1, 0.15]` is enforced, and `sign_proven: false` and `caps_in_force` are left in meta `run.attitude_axes`
  — the effective caps of a run whose sign has not been proven are always the first-water caps.
- **The rotational plant is all placeholder**: `hw_mpc.yaml plant.linear_damping` roll/pitch 0.07, rotational added mass 0.12
  (memory heavy-added-mass-provenance), ZG fixed at 0.01 (params.py) vs heavy_gripper coBM 0.01625, and the static payload pitch moment ~1.4 N·m
  [스펙 BlueROVHeavyGripper.yaml] is not modelled. The attitude loop has never been closed on hardware → the first-water test must be a policy-free **plain mpc attitude
  step** (the attitude version of the 0914 yaw step test), and until then do not touch `attitude_q_scale` or `rp_ref_filter`.
- **Axis gains [유도] + solver U_MAX vs wire cap mismatch**: `axis_gain.roll_nm 13.2 / pitch_nm 7.2` are values derived from heave_n 60 N [예측] × lever arm
  (y ±0.22 / x ±0.12 m, bluerov_heavy.xml). The solver U_MAX[3:5] = 8 N·m, but the most the wire carries is cap×gain =
  2.6 / 2.2 N·m (first water 1.3 / 1.08 N·m) → the optimizer can plan torques the wire cannot carry. EAOB is told the capped value via `note_applied`,
  so the observer sees the truth, but plain mpc shows tracking lag in large attitude transients. The first-water pitch cap of 1.08 N·m is
  **below** the static payload moment of ~1.4 N·m, so an active-leveling run can sit with M saturated, carrying a constant error (sat-ineffective has to catch it).
  Later fix: rebuild acados with U_MAX[3:5] = cap×gain (the sim plant binary changes along with it). Recorded in meta `controller.attitude_ref.u_max_wire_nm`.
- **Pitch-axis deadband limit-cycle risk**: knee ≈ 0.096 × 7.2 ≈ 0.7 N·m [유도] ≈ the 20° hold moment (0.47–0.73 N·m [유도]) —
  the attitude version of the yaw dither (2026-09-14). Whether the vertical thrusters are already above the knee during depth hold depends on the buoyancy assumption, and two memories
  conflict (hull negatively buoyant vs fit +0.1 N, neutral) → the sim S5 deadband plugin is the first evidence.
- **`transport: rc_override` (FALLBACK-A) not implemented**: `validate_engage_attitude_axes` refuses it with "not implemented". It is the alternative path for when s/t are ignored on firmware below 4.1.2,
  but it changes the deadman semantics (RC_OVERRIDE_TIME [스펙 unconfirmed] vs 500 ms) and the 4-DoF path itself, so
  do not enable it without its own bench-parity artifact. The STABILIZE lean-angle cascade is not implemented either (with attitude_axes.enabled, STABILIZE is refused).
- **The attitude numbers are all [예측]**: rp_max 20°, rp_reject 30°, pq_max 0.35 rad/s, rp_jump 5°, leash 3°, div_rp 15°, abort 35°,
  caps 0.2/0.3. The first artifact for the label distribution is `data/20260926/0926_175145_offline/gate_pos_rpy_width.log` (0901 land demos, 1721 windows:
  |dpitch| p50/p90/p99/p99.9 1.17/5.53/13.07/18.31°, |droll| 0.70/2.56/5.39/8.17°; peak |p|,|q| p90 0.453 rad/s > the 0.35 cap, so on a 200 ms
  grid 2/1721 chunks go accept→clip) — the held-out gate of a 7-dim training run (`dp_policy_offline.py replay --val-only`) and
  `policy_session_replay.py --ckpt D7=…` (water-run OOD probe) have not been run yet.
- **The records boundary is broken deliberately**: from schema 16 on, 5 columns at the end of the mission CSV (`rroll_deg,rpitch_deg,ax_roll,ax_pitch,rp_track`),
  `roll_deg,pitch_deg` at the end of `policy_plan.csv`, and plans.jsonl `rp_tracked` are always appended. This repo's tools, which read by name, are unaffected, but
  external readers that count columns or compare bytes break. **Do not pool** runs that differ in `attitude_track`, `attitude_axes.enabled`, `action_repr`, flight mode or
  `controller.allocation.attitude` (the meaning of dobmpc w_hat[3:5] changes).
- **Gaps in the sim evidence**: (i) [resolved] the 174607 `dropped_*` rows for "dobmpc does not exceed `div_rp 15°` when K/M are silently dropped"
  are degenerate because threshold = reference size (15°) — do not cite. In the non-degenerate rerun (reference 20° = `rp_max_deg`, threshold 15° = `div_max_rp_deg`) mpc and dobmpc
  both tripped at 0.45 s (10 consecutive samples above the threshold = the station's 0.5 s debounce; the hull did not approach 20°), and dobmpc, on K/M that were credited but
  never delivered, drifted to pitch 6.83° with rad_max 14.5 cm (`bluerov2_mujoco_marinegym/recordings/20260926/attitude_hold_211917/
  results.csv`, `dropped_*` rows) — the reason for keeping `dobmpc_allowed: false` is this drift, not the threshold. (ii) The S2 hold grid ran NONE only; the C/CDW cells were not run (`verify_attitude_hold.py --modes C,CDW`);
  (iii) the S5 deadband plugin ran heavy (+1.1 N, near-neutral) only; the limit cycle of heavy_gripper (−5.7 N, sinking) was not run;
  (iv) the first `pos_rpy_width` checkpoint now exists (2026-10-03: `data/checkpoints/20260929_peg7d_vitclip.ckpt` → `data/20260929/peg7d_vitclip/`,
  trained on bdml-server, epoch 190, contract keys `action_repr`/`yaw_axis_*`/`rpy_convention` present; labels |dpitch| p95 3.28° p99 6.03°,
  |droll| p95 2.94° p99 5.02° — `data/20260929/peg7d_vitclip/rp_label_stats.txt`) → but held-out acceptance, OOD replay and the
  `dp_policy_reference.py` parity fixture are still not run. **Since `action_repr: auto` (2026-10-03), picking this checkpoint and picking LOW
  `RL_PWM_7D` means roll/pitch are tracked on a single START, with no YAML step** (the RL_PWM_7D policy also falls short of the sim promotion criteria, entry above) —
  the panel says so before START with `TRACKED on RL_PWM_7D` (warning colour), and the chip after ARM with `· 7d rp TRACKED`, but it is not blocked. ④ A dropped-and-logged
  record (4-DoF LOW) comes first; (v) because of the pre-existing NU=4 vs 6 failure, `tests/test_dobmpc.py main()` is still
  red; the 6 new attitude tests are run individually by name.

### The ArduSub **joystick gain** multiplies every axis the station sends, yet it is neither read nor recorded (2026-09-07)
- ArduSub puts MANUAL_CONTROL's x/y/r into the RC override as `PWM = 1500 + 0.4·gain·값` ("PWM = 1500 + 0.4·gain·value"), and z with a `0.8·gain·JS_THR_GAIN` scale
  (`ArduSub/joystick.cpp transform_manual_control_to_rc_override`, Sub-4.1 :55-150, Sub-4.5 identical). At boot `gain` is
  `JS_GAIN_DEFAULT` (default **0.5**, range 0.25–1.0), and the gamepad's `gain_inc/dec` buttons (D-pad 11–14 in this station's gamepad table)
  change it on the spot. So station calibrations such as `axis_gain.surge_n/heave_n 60 N`, the deadband threshold axis ≈0.096 and the slew 1.5/s
  are all **values that presume gain 0.5**, and a single D-pad press silently scales them 2×. The vehicle sends this value periodically as `NAMED_VALUE_FLOAT name="PilotGain"`
  (`GCS_Mavlink.cpp` Sub-4.1 :145) [provenance: reading the ArduSub source 2026-09-07, workflow stabilize-mode-llc; the hardware parameter is unconfirmed].
- Impact: even if the effective thrust scale differs from run to run, meta has no trace of it, so every numeric comparison involving the deadband or axis gains is conditional.
- Fix: `hardware.py` puts `PilotGain` into Telemetry and `MpcWorker` writes it to meta `run.pilot_gain` on engage;
  if gain ≠ 0.5, refuse engage or warn. Until then, check the gain display in QGC/BlueOS (STATUSTEXT `#Gain: NN%`) at the start of each session.

### On the manual REC path, meta `controller` **may not be the controller that flew** (2026-09-07, found in the safety audit)
- If REC is pressed before engage, `_csv_auto=False` → disengage does not close the CSV → `set_mode` is allowed → on REC stop
  `_write_meta()` writes the `meta()` of the **new** controller (`rov_gui/control/workers.py` `_write_meta`/`set_mode`). `type`, Q and
  `heave_trim.applied` may all be saved out of step with the actual flight. Automatic REC (the path that engage opens and closes) is unaffected.
- Fix: take a `ctrl.meta()` snapshot at engage time and write that at close. Until then, read the meta `controller` of manual-REC runs
  against the `ENGAGED (<mode>)` line in `events.log`.

### MPC-family followers under policy plans **cannot produce thrust because of the hold-tail mask** — the round-1 `track` A/B is not enough, `q_scale` unverified (2026-09-07)
- **Symptom**: when the policy is flown with `mpc`/`mpc_tuned` as the low-level controller, the vehicle barely moves forward (net displacement
  0.05–0.4 m/50–90 s; PID 0.2–0.9 m). The reference sits consistently 7.5–10 cm ahead of the body, yaw error ~1°, uX sign flips
  0–0.4/s → not wobble but **lack of thrust**: |uX| p50 0.5–1.8 N, axis_surge 0.01–0.03, so below the deadband
  on 92–100 % of ticks (PID 6–7 N, 37–52 %) [측정: data/20260907/0907_164659/
  mpc_16{4716,4835,5209}.csv, 0907_152730/mpc_153134.csv vs 0907_145206/mpc_145239.csv].
- **Cause (in order of evidence)**: (1) `hold_tail: mask` — plans are already ~0.5 s stale when they arrive (obs age p50 0.5 s),
  so only 0.25–0.4 s of the 3 s horizon is position-weighted (`hold_frac` p50 0.87–0.92); the optimal surge for that cost is
  0.05–0.2 N on a fresh plan, 0.6–1.9 N on blend. **In the same runs, `ref_src hold` ticks, where the mask is released, give 3.3–6.9 N**
  (mpc 164716 6.5 / 165209 6.9 / mpc_tuned 164835 4.5 / 153134 3.3 N; PID ~6.5 N regardless of ref_src).
  (2) The 0.05 m leash saturates on 95–99 % of replans → the error cannot grow (the policy requests 3–9 m per epoch, the anchor
  moves only 0.05–0.48 m). (3) mpc/mpc_tuned have no integral or disturbance term (EAOB unused, intended) → they cannot learn the DC loss;
  `mpc_tuned` is softer still with `along_scale 0.25`. The deadband is a physical threshold (knee in this record: axis 0.05–0.07 ≈ 3–4 N).
- **A/B round-1 result (2026-09-07 18:00, `hold_tail: track`)**: in 4 mpc_tuned missions uX p50 went from 0.4–0.6 N (mask, 16:11–16:40) →
  **2.5–3.6 N**, about 5× higher, but is still below the deadband (|ax_surge| > 0.096 on 0–2 % of ticks), net displacement 0.07–0.24 m/16–28 s
  [측정: 0907_180038/mpc_18{0058,0132,0202,0253}.csv vs 0907_161726/, 0907_163702/]. The same runs drifted upward from the buoyancy FF problem above, so
  the forward-progress verdict is to be redone after `vehicle_net_buoyancy_n` is applied. The next lever is `policy.q_scale` (2–3).
- **Workaround (2026-09-07, unverified)**: hw_mpc/land_dp `policy.hold_tail: mask → track` (single-variable A/B, priority 1;
  watch for the "slowdown at every chunk end" that A10 worried about) + secondary lever `policy.q_scale` (position weights ×s during policy plans,
  runtime cost_set, restored on STOP/mode switch; default 1.0, stiffness ~√q [유도]) + raise `anchor_leash_m`. Judge by comparing
  the CSV `ax_surge > 0.096` tick fraction and net displacement against PID runs. dobmpc left untouched, as requested.
- **A/B round 2 (2026-09-08, `track` + `anchor: measured`)**: in the plan segments |uX| rose from 0.05–0.22 → 1.21 N (blend 1.46, hold 0.80), but that is still
  1/4 of the deadband, and all three axes have pwm_dev ≤20 µs; ground speed 0.008 m/s. The bottleneck is now not the mask but the "all three axes in the ESC deadband" entry above
  [측정: 0908_112101/mpc_112101.csv, diag/taskE_out.txt]. Today the measured anchor changed at the same time, so do not cite this as the effect of track alone.
- **Proper fix**: (a) cut the observation age (inference 232 ms, GPU shared with FS) to win back plan lifetime, or base the mask
  on arrival time instead of plan start time, (b) put a measured thrust map (deadband) into the plant, or per-thruster
  compensation (needs a direct-thruster interface), (c) re-derive the hardware Q/R, leash and v_max for the approach phase.

### Policy jaw channel: "now" sampling deadlocks, and intent sampling can close **before arrival** (2026-09-07)
- **Symptom**: on the first hardware approach (0907_145206, mission mpc_145239, 120 plans) the vehicle reached the object, but the jaw
  did not close. The 1st-order cause was `policy.gripper: false` (A12, OFF by default) → switched to `true` on 2026-09-07. But
  going back over the record, it would not have closed immediately even with the switch on: in every chunk the policy outputs g≈0.95 (open) at knot 0
  and g<0.30 (closed) at knots 4–5 (all of the last 15 plans), but the station looks only at the stitcher's "now" value (`ts[0]`)
  and plans are replaced every 0.45 s, so only 7–59 % of ticks after 32 s are below the threshold, and knot 0 of a new plan (0.95 >
  open_above 0.67) can fire an OPEN edge again [유도: offline replay of plans.jsonl, this session].
  Knot 0 stays at open because the width the policy observes is an open-loop estimate that does not change without a command
  (the policy waits for the jaw to move, the station waits for g to drop).
- **Deeper problem**: the policy's intent to close appears from t≈32 s, **0.35 m out** from the final position (plan g[-1] mean
  0.26). In the demos the hand starts closing ~1 s (≈0.3 m) before the final pose, so the policy is reproducing that **spatial** timing,
  but the vehicle is 5–10× slower than the demos (0.03–0.2 m/s) and takes 30 s for those 0.35 m. If it closes as intended, the jaw (travel
  2 s) is fully closed 28 s before arrival. On top of that, the deployment observations are out of distribution ("gripper on the opposite side of the image" entry above), so
  the reliability of the width prediction itself is also unconfirmed.
- **Workaround (2026-09-07)**: a `gripper_lookahead_s` knob (default 0.0 = old behaviour; at 0.6 s, in the same run, 70–100 % of the ticks after 32 s
  are below the threshold) + a CSV `grip_g` column (schema 13), so that the next run records the value the hysteresis saw.
  After the first CLOSE edge the estimated width drops and the policy follows, so the deadlock is expected to clear by itself [예측].
  Pilot G/H edges still take precedence (last-writer-wins, replay entry above).
- **Proper fix**: (a) tie the jaw decision to **path progress** rather than time — close when the vehicle reaches the position of the knot
  where the plan's g drops (applying the stitcher's time dilation to g as well does this naturally; for now it needs checking whether g
  is carried onto the knot grid as is); (b) AND with an arrival check (reference at rest + intent g<threshold); (c) fundamentally,
  tracking matched to the demo speed (v_max ↑, time dilation ↓), or relabelling the width channel as "distance-based", independent of speed.
  Either way, **the `grip_g`/`grip_cmd`/events record of 1 hardware run with the jaw enabled comes first**.

### `demo_e2e.py policy` is **red** with the deployed config — temporary z-hold + demo plant, unrelated to today's change (2026-09-11)
- **Symptom**: `QT_QPA_PLATFORM=offscreen python rov_gui/tests/demo_e2e.py policy` gives exit 1. The deployed
  `hw_mpc.yaml policy.z_hold_above_floor_m: 0.20` (temporary entry above), relative to the demo datum, puts every knot z outside the box
  (~0.3 m), so 3 consecutive rejects → escalated, installed 0. Even with z-hold set to null, under `hold_tail: extrapolate` +
  `anchor: measured` (both pre-existing uncommitted diffs) the reference outruns the synthetic plant and it ends `diverged`
  [측정: 3 runs by the review agent, log in the scratchpad e2e_policy_shipped.log — gone after the session ended; reproduce with the command above].
  The HIGH/LOW, ckpt picker and max_run_s 500 change (2026-09-11) is not on the failure path, but there is **no green e2e baseline** for that change.
- **Workaround**: checked instead with the offline unit tests (offline 119, policy 84, control 112, policy_worker 30) and
  `rov_gui/tools/policy_observe_smoke.py` 15/15. For e2e, only other shapes such as `demo_e2e.py pid follow …` are green.
- **Proper fix**: after deleting the z-hold entry as planned, retune the demo plant or the e2e expectations for `hold_tail: extrapolate`/`anchor: measured`
  (the synthetic plant's response relative to `div_max_m` 0.25) to make `demo_e2e.py policy` green again, and at that point make the LOW None variant
  (`Opts.mpc_mode = "none"`) runnable through a driver argument too (currently main() accepts only mpc/dobmpc/…/pid).

### The training repo's val curve is **based on the EMA weights**, so it steers checkpoint selection backwards (2026-09-07)
- **Symptom**: in the 2026-09-07 5-dim retraining, ranking by the 32-window `val_action_mse_error_pos` in `logs.json.txt` (= the workspace's
  `sample_every` batch) put **epoch 10** in 1st place, but re-measuring the 219-window held-out set with the deployed weights (`model`) shows
  epoch 10 is the **worst of the epochs measured** (26.7 mm) and the late epochs are the best (ep150 17.2 / ep195 17.3 mm)
  [측정: `<run>/heldout_pos_yaw_width_ep*_model_stride4.log`, run =
  umi_underwater_robust_control/data/20260907/0907_113747_train_umi_depth_5d_can_grasp_depth_5d_v0].
  The workspace's val samples with `policy` (= the EMA copy), so with the curve of weights contaminated by the BatchNorm defect (entry below)
  it is in effect **ranking the healthy weights**. At the same epoch 195, ema 33.0 mm vs model 17.3 mm
  (MSE 3.6×), and the epoch rankings of the two curves are inverted relative to each other.
- **Workaround**: select with `dp_policy_offline.py replay --val-only --weights model` (219 windows), and look at the val curve
  only "for checking convergence". This protocol and both curves are written down in `selected.json` of the 2026-09-07 run.
- **Proper fix**: (a) fix the BatchNorm/EMA defect ((2) of the entry below) so that both sets of weights point to the same thing, or
  (b) also run the workspace's val sampling with `model` and log both series together, or (c) revive the commented-out held-out val
  loop (`train_diffusion_transformer_timm_workspace.py:273-288`) and record on the full validation set.

### All runs before 2026-09-06 have **not a single frame of underwater depth** (2026-09-06)
- **Symptom**: hardware runs do not keep depth on disk — not the 224 obs, not the uint16 mm maps, **not even the raw
  stereo pairs**. So neither after-the-fact checking nor offline regeneration is possible
  [측정: data/20260903/0903_183555/ = controller.json,
  mission_log.txt, ui_20260903_183555.{mp4,json} — no depth].
  The only visual trace, `ui_*.mp4`, is a recording of a **screen already colourized with per-frame auto-range + 64-knot histogram equalization
  and downscaled to widget size**, so the same colour is not the same distance.
- **Workaround (from now on)**: `--record-depth` saves lossless PNGs to `<run>/policy_obs/`.
  `python -m rov_gui.tools.depth_compare pair --run <run>` puts them on the **same colour rule** as the land training observations
  for comparison.
- **~~Proper fix: the raw stereo pairs must be saved too~~ → implemented 2026-09-06**:
  `--record-stereo` saves **RAW (unrectified) mono pairs + rig + that run's FS settings** to `<run>/stereo/`
  losslessly (`rov_gui/perception/stereo_record.py`). Now, by changing iters/scale/checkpoint
  or by re-rectifying with a new calibration, **depth can be regenerated without diving again**.
  Remaining gap: **there is no offline regeneration tool yet** — the storage format is self-describing, but
  the script that reads `<run>/stereo/` and reruns FoundationStereo has not been written yet.

### Unifying the depth renderers — **only half** done (updated 2026-09-06)
- **The fixed part**: the underwater FoundationStereo panel and the land FS video now use **the same function**
  (`rov_gui/depth_colour.py`: TURBO / warm=NEAR / fixed 0.20–3.00 m / obs inverse spacing /
  invalid black). The station defaults to `--fstereo-palette umi`; the old scene-adaptive rendering
  (JET + per-frame percentiles + 64-knot equalization) appears only with `--fstereo-palette adaptive`.
  On land, `data_collection/make_depth_trajectory_video.py` and `depth_compare` import the same
  module. The rule is written in the HUD, the colour bar and the run meta (`fstereo.panel`).
- **The remaining two (not fixed)**: the land recorder preview, TURBO/linear mm/**warm=FAR**
  (`umi_handheld/record.py:529-536`), and the underwater device-stereo panel, TURBO/linear mm/**warm=FAR**
  (`rov_gui/imaging.py:105-125`, `depth_to_bgr`). Both have the opposite polarity, so placed side by side with an FS panel screenshot,
  **the near side is the opposite colour**.
- **Bonus (still valid)**: the old `umi_ep0_foundationstereo.mp4` at the repo root has the same endpoints as obs
  (0.2–3.0 m), but its mapping is **linear** — 1.00 m is palette 0.29 in that video and 0.14 in obs.
  The new 3-panel video uses obs spacing, so do not mix it with that old mp4 when viewing.
- **Proper fix**: move the remaining two renderers to `depth_colour.depth_mm_to_bgr` as well
  (for the device panel there is also the option of keeping the 0.3–6 m range and matching only the polarity), or at least have each panel
  write its own rule on screen.

### One run folder holds several engagements, and **the plan log is shared** (2026-09-06)
- **Symptom**: `data/20260906/0906_191935_observe/` contains **two** engagements —
  `mpc_191935` (19:19:35, 14,918 rows, epoch 2, plan 1..1426) and, 41 minutes later, `mpc_200050`
  (20:00:50, 2,391 rows, epoch 4, plan 1991..2203). The CSV and meta are separate files, but
  **the two runs share `plans.jsonl` (1,639 rec) and `policy_plan.csv` (1,639 plans)**, and
  the second run's `t_rel` **starts again from 0**.
- **Why it goes silently wrong**: if plans are looked up by time, then in the t < 230 s window **the second engagement's
  reference gets picked at the first one's times**. That is what the first render actually did, and the clock offset
  spread measured with the two epochs mixed came out as **2,475 s instead of 0 ms**.
- **Workaround (implemented)**: `render_run_scene.py` / `export_run_html.py` **pick the engagement
  from the data** (the one whose plans fall inside the recorded depth window) + filter the plans by `meta.policy.epoch` and
  the CSV's `plan_id` column. Explicit choice with `--engagement <이름조각>` ("name fragment").
  Regression test `test_run_scene.py::test_two_engagements_in_one_folder_are_not_mixed`.
- **Proper fix**: have the station open a new folder for each engagement (it appears the join window reopened after the
  `_run_dir` pin was released), or at least add an `epoch` column to `policy_plan.csv` so that
  the file states its own ownership. Right now **you cannot tell which run it belongs to from the file name alone.**

### The top arch and right-hand strip of the FoundationStereo panel are **red (≤0.2 m)** — the network hallucinates the alpha 0.5 rectification margin as close range, and nothing filters it out (2026-09-14)
- **Symptom**: at the top of the `--fstereo` (native view) depth panel, a deep-red arch, thickest in the middle (rect rows 0~10),
  a ~6 px vertical strip at the far right (cols 634~639), nothing on the left. Values: a 205→315 mm ramp at the top, **72 mm** on the right
  (disparity 357 px @640). The HUD says `valid 100%` — that 100 % is the symptom (junk counted as valid).
- **Cause (confirmed; 3 refutation lenses, 0 refutations)**: `--policy` raises `--fstereo-alpha` 0→**0.5**
  (`rov_gui/__main__.py:1127`, because of the 224 obs crop coverage gate). In the alpha 0.5 left rectification map
  (`initUndistortRectifyMap(K_left_live, D_left_live, R1, P1)`), **2.89 %** of the grid samples outside the raw sensor
  (row 0 spans raw y −10.4~+5.9 → only the middle is outside = the arch; col 639 spans raw x 638.6~643.9; col 0 spans x 7.4~13.3 = inside,
  so no left strip). `cv2.remap`'s default BORDER_CONSTANT 0 fills those pixels **black** (`rov_gui/perception/fstereo.py:840-841`),
  FoundationStereo is a softmax regression, so it cannot output "invalid" and spits out a positive disparity (`core/submodule.py:427-431`;
  the clip is only in `run_hierachical`), `depth_from_disparity` sets only d≤0 to 0 (`c3_camera/host_depth.py:168-226`,
  `Z_RANGE_MM=(0,65535)`), and `stereoRectify`'s `validPixROI` is thrown away (`host_depth.py:503`). There is no mask anywhere.
  [측정: near(<300 mm) fraction inside the out-of-sensor mask 0.811 vs outside 0.015, 168/168 frames, bleed 1 px —
  the scratchpad `margin_check.py`/`ring_bleed.txt` results can be reproduced from data/20260914/0914_152913/policy_obs/depth/*.png +
  mpc_152951.meta.json `fstereo.rig`; identical in all 52/52 runs that have native PNGs]
- **Impact**: (1) the panel and the cursor probe (they show the arch at 0.17~0.31 m and the strip at 0.07~0.09 m as a "measurement"); (2) in the colour
  grid map the top and bottom margins fall outside the FOV, but the right-hand 72 mm strip **lands as a wavy curve at cols 298~399 and covers the real
  surface in the z-buffer** (0.6~1.0 % per frame) → if the `--pose` FoundationPose mask reaches into the right 1/4, the `_frame_depth_quality` p5
  may wobble; (3) the junk is recorded as is in the `--record-depth` native PNGs (52 runs); (4) **policy obs**: the right-hand strip is outside the
  crop (x0=120); the top arch enters the obs only **as a dotted line on the single row 0 (~36 %/frame, 0.16~0.20 % of the pixels)**, yet disguised as
  valid=255 — identical in 53/53 runs, a pattern absent from the land training obs (alpha 0, using only source rows 81~369). The effect on the plans is
  **not measured** (comparing against obs with row 0 overwritten by row 1 via `policy_session_replay.py` gives the answer; needs a GPU).
- **Workaround**: ignore those two spots on the panel. Compute offline statistics after filtering with the out-of-sensor mask (the rig is in the meta).
- **Proper fix (priority 1: A)**: in `StereoRig.from_stereo_pair` (`host_depth.py:517-519`), build
  `valid_left = cv2.remap(np.full((h,w),255,np.uint8), maps_l[0], maps_l[1], cv2.INTER_LINEAR) == 255` and
  **dilate** invalid by 1~2 px (do not erode valid — the col 0 side has no margin), add a `StereoRig.valid_left` field +
  provenance `valid_left_pct`/`margin_dilate_px` (meta boundary) → right after `fstereo.py:844`, `depth_native[~valid_left] = 0`.
  The HUD's `valid_native` becomes honest at ~97 %. **Do not apply** a right-image mask (for 8.8 % of the grid the match point falls in the right
  margin, but there is no near junk, 0.016 vs 0.015). As a secondary option, consider alpha 0.2~0.3 [유도: coverage 99.1~99.6 %, margin 0.31~0.74 %],
  but the network d≤0 of 3.06 % from the alpha 0 days (`fstereo_bench_out/hardware_20260902_fill.txt`) may come back (not modelled).
  Precondition: **record K_right/D_right/R2** in the run meta `fstereo.rig` (`hardware.py:708-715` writes only the left side) — without them, all
  offline verification is [유도]. A range-limit cut (E) is not the answer (precedent: it erased the 15 cm gripper, `fstereo.py:107-119`). Without
  `--record-stereo`, no BORDER_REPLICATE / pre-network crop A/B is possible.


### The deployed FoundationStereo settings differ from the training data — deliberate, but unresolved (2026-09-06)
- **Status**: the training depth is iters **16** / scale **1.0**
  [측정: `~/Desktop/data collection/dataset_depth.zarr.zip` `.zattrs['depth_source']`],
  while the 0903 hardware run is iters **8** / scale **0.75**
  [측정: data/20260903/0903_183555/controller.json
  `fstereo.iters` / `fstereo.scale`]. It is an **intended trade-off** made to buy time-axis parity (matching the observation-pair spacing to the
  training stride of 66.7 ms), and `FS_RATE_NOTE` in `__main__.py` says as much.
- **Unresolved**: **nobody has run the A/B** on which mismatch costs more. `--fstereo-scale 1.0
  --fstereo-iters 16` restores instrument parity but drops to ~7 Hz.
- **Proper fix**: use `--record-depth` to leave one run with each of the two settings and compare them against the land observations
  with `depth_compare` — that comparison is now possible.


### The two upstream workspaces of `--pose` (SAM2 + FoundationPose) **are not on disk** — turning it on stops at load (2026-09-30)
- **Symptom**: the defaults `SAM2_LIVE_ROOT` = `~/Desktop/New Folder` (sam2_live, `rov_gui/perception/session.py:57`) and
  `FOUNDATIONPOSE_ROOT` = `~/Desktop/poseEstimation/FoundationPose` (`session.py:1492`) are both missing (`ls`, 2026-09-30;
  the trash is empty). Turning on `--pose` gives `PoseSessionError: perception source not found at …/New Folder`, and mesh reconstruction gives
  `reconstruction script not found …/run_nerf_single.py`. It is the same class as FoundationStereo (a tree outside the repo), and `New Folder` was
  deleted **even though it was documented** in the "The original is left untouched" section of `rov_gui/README.md` — documentation alone is not
  enough; it has to live inside the repo (FS was restored to `external/FoundationStereo` on 2026-09-30). The FoundationPose root existed only in the
  code and in `oakd.sh`. The usual launch line (`--fstereo --policy`, without `--pose`) is unaffected.
- **Workaround**: do not use `--pose`. If a copy exists, point to it with `--pose-src`/`$SAM2_LIVE_ROOT` and `$FOUNDATIONPOSE_ROOT`
  (`etc/conda/activate.d/oakd.sh` in `rovgui-pose` also exports the old paths, but `./c3` does not activate the env, so it does not matter).
- **Proper fix**: find or re-download the two trees, put them inside the repo (`external/`) like FoundationStereo, and make the defaults
  repo-relative paths. sam2_live is in-house code, so there may be no upstream — start by checking where the original was.


### Loading FoundationStereo **asks GitHub every time** — on a 429/5xx that run has no depth even though a cache exists (2026-09-30)
- **Symptom (latent, has not happened yet)**: upstream `external/FoundationStereo/depth_anything/dpt.py:157`
  calls `torch.hub.load('facebookresearch/dinov2', …)` without a ref, and torch 2.11's `torch/hub.py:205-212`
  (`_parse_repo_info`) opens `https://github.com/facebookresearch/dinov2/tree/main/` on every load. If there is no network
  at all (URLError) it falls through to `~/.cache/torch/hub/facebookresearch_dinov2_main`, but **an HTTPError other than 404 (429, 5xx)
  is raised as is** → `FS FAULT` → FS does not fall back to another sensor, so that run has no depth. It is `urlopen` without a timeout, so on a
  half-broken network `FS LOADING` may drag on [유도].
- **Workaround**: if the internet at the pool is unstable, disconnect it entirely before launching (URLError path → cache).
- **Proper fix**: during the load only, redirect the dinov2 hub call to the cache directory with `source='local'` — upstream
  left exactly that line as a comment at `dpt.py:155`. The rule is not to touch the checkout, so do it the way of the existing monkey patches in
  `fstereo.py` (like `_patch_for_capture`).


### Gripper geometry of the vehicle marker: **only the jaw x is measured**, the rest is still estimated (2026-09-03, updated 2026-09-08)
- **Status**: `--policy-observe` and the new vehicle marker (hull+gripper+thrusters+C3, 3-D, dashed heading line fore and aft) have run in
  the pool (data/20260903/0903_183405_observe, 20260908/0908_180453, etc.). The marker's first check against the real thing was
  the 2026-09-08 "bottle held in the jaw onto the centre of tag 58" test, and it exposed an error in the jaw position (below). Whether it keeps going
  without cutting out through tag dropout, leaving the box and long runs, and whether the joystick stays alive to the end, **has never been checked separately**.
- **Evidence grade of the marker geometry (updated 2026-09-08)**: the jaw x is now **measured relative to the lens** — in the frame where the bottle
  held in the jaw was placed on the centre of floor tag 58, the grip point is 0.196 m ahead of the lens (range 0.187~0.204) [측정: data/*/*_observe/
  20260908/0908_180453/policy_obs/rgb/000160.jpg (tag PnP with the run's own intrinsics) + 0908_170428/policy_obs/rgb/
  000000.jpg (bottle height 0.084 (±0.006) m) + 0908_175151/policy_obs/rgb/000000.jpg (open-jaw ray)]. The CAD chain's 110.664 mm
  (→ jaw 0.4165) has been disproved (that jaw projects to row ~420 of a 360-row frame, but the jaw is seen at rows 265~360).
  `rov_shape.LENS_TO_GRIP_FLU_M = (0.196, 0, −0.275)`, `JAW_CENTRE_M = policy.tcp_body_flu_m = (0.502, 0, −0.17)`,
  meta `rov_drawn_geometry.lens_to_grip_flu_m`. The meta of runs before 2026-09-08 carries 0.4165 (records boundary).
  **Not yet measured**: jaw z (anchored to the CAD 275.014 mm vertical; agrees with the image within ~1 cm but is not an independent measurement
  [유도]), jaw y (assumed 0 [예측]; the colour camera sees the axis of the held bottle 0.011 m to the right of the vehicle axis), finger length and
  opening (jaw box [예측] / vendor [스펙]), tube position (sim `GRIP_POS` pushed by the same +0.0855 as the jaw [유도]), footprint/shroud (below).
  And **the x/z of the lens itself are suspected** (entry "The x/z of the C3 extrinsic cam_t_flu are suspected") — the jaw is anchored to the
  lens, so either way the jaw is right, but the hull drawing may be drawn ~9 cm further back than it really is.
- **The thruster shroud radius 0.0467 m is a circular derivation**: it is the value obtained by assuming "the vertical thrusters set the vehicle
  width" and subtracting from the footprint 0.5334, so **there is no independent verification**. On top of that, a CAD width of 0.5749 is floating
  around separately (which would give 0.0675), and which one is right is undecided.
- **Proper fix**: (1) in a pool `--policy-observe` run, explicitly check tag dropout, leaving the box, long runs and joystick
  survival; (2) measure the jaw y/z and the gripper tube on the vehicle (stereo triangulation or a ruler) and
  update `rov_shape` + `policy.tcp_body_flu_m` (or `tcp_offset_cam_m`) together; (3) settle the footprint on either CAD or the tape measure
  and re-derive the shroud radius.

### The C3 underwater depth error **grows in proportion to distance** — a constant scale factor cannot fix it, in principle (2026-08-24)
- **Symptom**: a single `--depth-scale` is right at only one distance. The measured raw depth-vs-MAP value
  moves from **1.28** at low height to **1.56** at ~0.9 m (confirmed by the pilot).
- **Conclusive evidence that it is not a sampling artifact** (a witness the diagnostic function plays no part in):
  the three axes of one mesh are **143 x 166 x 187 mm** while the caliper measurement is 119.73 mm — per axis 1.19 /
  1.39 / 1.56, **anisotropy 1.31**
  [측정: data/20260823/0823_210304/mission_log.txt 21:02:29].
  A constant scale error stretches **every axis equally**. A cube coming out as a brick means that the error
  changes with distance.
- **Model**: the error lives in the **disparity domain** (disparity is under-reported → depth reads long,
  getting worse in proportion to distance). `r(Z) ≈ 0.977 + 0.538·Z`
  [유도: regression on 108 × 10⁴ samples from c3_camera/datasets/* — **a different stereo config** (extended ON,
  MinZ 150), so it needs refitting under the flight config (extended OFF, MinZ 300). Do not cite as a measured value].
  As Z→0 the coefficient is 0.98±0.04 = **metric at close range**; the whole error is proportional to distance.
- **Ruled out** (each by a specific number): pure scale (1.56× would mean fx=244 px → half-FOV 52.7°, which
  exceeds the Snell in-water limit of 48.75° = a camera that cannot exist) · flat-port refraction (the sign is opposite and it is
  10~30× smaller) · integer disparity (the truncation upper bound is +1.8% but +56% is needed; moreover, the actual
  output is dense at 1 mm spacing, so no quantization grid is visible at all) · "the stereo extrinsics are
  in-air values" (a quantity that does not change with the medium, so it has no physical content).
- **Workaround**: `--depth-scale 0.64` is **exact only at Z ≈ 1.09 m**, and the residual changes sign on
  either side (0.5 m −20%, 1.5 m +14%, 2.0 m +31% [유도]). If you keep it, **fly at a height of 1.0~1.1 m**.
  The depth-derived distances (mesh dimensions · FoundationPose distance · object_nav distance) — **none of them is
  currently metric**. The colour AprilTag path is unaffected and is the only reference.
- **Proper fix**: (1) **record** the hundreds of pairs that `_check_depth_scale` makes per frame
  (right now everything except the median is thrown away, window.py:801-811) → `depth_check.csv` in the nav folder;
  (2) once in the pool: a 5-rung distance ladder (0.4~1.8 m) + a **pitch fan of ±15°** (without it, z and
  the FOV term are collinear and the coefficients turn into garbage) + a yaw fan; (3) regress in disparity space and
  apply `Z_corr = Z/(1 + c·Z)`. **Two prerequisite checks that work without getting in the water**:
  dump `calib.getFov(sock, useSpec=True/False)` and `getStereoLeft/RightRectificationRotation()`
  (if the rectification rotations have a 1~2° relative yaw, the offset hypothesis is confirmed), and run the same
  diagnostic in air to see **whether the distance dependence remains** (if it does, refraction and the medium are ruled out completely as causes).
- **Not settled**: nobody has confirmed the cause (rectification zero-point / homography rectification / matching bias) on the
  device. **1.7 m and above is not calibrated** — all the data are 0.10~0.76 m.
  If the FOV-dependent term (`+0.474·Z·tan²θ`) is real, **no r(Z) curve can fix it**, and underwater
  stereo recalibration is needed.

### If the object shows up in the tag map **under the mat / two cells over**, nothing stops it — no map-frame plausibility check (2026-08-24)
- **Symptom**: in both runs on the morning of 2026-08-24, the object (on top of tag 58) was reported **51~53 cm below the mat**,
  with the nearest tag 11/10/52, and follow armed as is (231 cm / 256 cm).
  [측정: data/20260824/{0824_101807,0824_101251},
  reprojected with nav_*/map.json — overscale 1.55x / 1.57x]
- **That day's cause was fixed separately** (`--depth-scale` left unspecified → 0.64 promoted to the default). The remaining defect is
  **that there is no guard that works regardless of the cause** — an object below the floor is physically impossible, yet
  `object_nav.update` looks only at pose_state / camera distance / jump and does not look at the map-frame coordinates.
- **Workaround**: check by eye in the trajectory plot that the object marker is near the mat plane.
- **Proper fix**: add to `ObjectAnchor.update` a z band relative to the tag-map plane (e.g. reject if outside
  −0.1~+1.5 m above the mat) and rejection outside the pool volume. Derive the thresholds from the z distribution of tag_map.

### FoundationPose: **the re-registration pose itself still has no continuity gate** (2026-08-24, partially fixed)
- **What was fixed**: the `object_nav` jump gate still auto-reseeds after 5 rejections as before, but
  **if a reseed happens during follow, it demotes to STATION** (`workers.py:_tick_follow`, holding depth and heading;
  not a disengage). The path by which the 1.324 m ghost snap of run1 on 2026-08-23 dragged the vehicle has been cut.
  If the object's apparent speed exceeds 3× the FF cap for 1 second, the same demotion applies (a ghost that slides
  smoothly does not trip the jump gate, so a separate guard was needed).
- **Remaining defect**: SAM2 loss→global register **still does no comparison at all against the last-known pose**
  (`rov_gui/perception/session.py:1012,1018,1045`). The "pose does not fit" give-up counter also
  counts only watchdog-triggered re-registrations and is reset by 1 normal frame (`session.py:1038-1041`) —
  run1 used up 2 bad snaps and still fired only ~5–6 s later. The distance gate looks **only at the camera range**,
  and there is no map-frame plausibility check (pool volume, near the floor) at all.
- **Workaround**: do not trust the obj pose in the window right after stale→re-registration. (On the mesh-quality side,
  a gate was added on 2026-08-24 — frames with a depth smear ratio above 2.0 are not collected, and
  reconstruction itself is rejected for such captures. `SMEAR_MAX_RATIO_*`, session.py.)
- **Proper fix**: put a position/yaw gate against the last-known pose on the re-registration pose, on the perception side,
  and make the give-up counter also count recovered-triggered re-registrations.

### FoundationPose poses carry the time of **the last submitted frame, not the frame that was solved**, and during re-registration the old pose is republished as `tracking` — `pair_exact` only means "the labels match" (2026-10-05)
- **Symptom**: `PoseWorker.tick` puts the `t_capture` of the colour frame it last took from the mailbox into `_last_t_cap`, and attaches that value
  to the published pose (`rov_gui/backends/hardware.py:4732-4739`). But the pose content is the solution FpWorker most recently **finished**
  (`rov_gui/perception/session.py:1336-1343`). The comment right above (4724-4731) says "it is solved inline inside submit", but that does not
  match the data. In the three 0823 water runs, publishing continued at 10 Hz even during re-registrations lasting 0.6–1.4 s, and `frame_seq`
  went up by +2 per row. 11–17 % of consecutive `tracking` row pairs had bit-for-bit identical position and rotation, and the long freezes lasting
  7 rows or more mostly ended with `n_register`+1
  [computed in review, not saved: `data/20260823/0823_{182628,195927,210304}/c3_depth_*_pose.csv`]. `poll()` reports `registering` only when T is
  None (`session.py:1371-1375`). Under `--fstereo`, if the FS map is more than 250 ms older than the colour frame, the FP submission itself is skipped
  (`session.py:648`), and even then a new label is attached.
- **Consequence**: `object_nav.pick_fix` compares only float equality. So even with `pair_exact`=1 there is no guarantee that it is the same frame,
  and the extrinsic cancellation breaks. During a freeze the object's MAP position moves with the camera (translation gain 1), the FD velocity follows
  the vehicle velocity, and the freshness measured by label age stays LIVE. The reference moves with the vehicle, so the div guard structurally cannot
  see this. The size is vehicle speed × freeze time ≈ 0.023–0.055 m/s × 0.8–1.4 s ≈ 2–8 cm [유도]. However, these freeze statistics come from the
  failure regime of a non-metric mesh and distances of 1.5–2.5 m.
- **Two timing errors on the same path**:
  (1) Under `--fstereo`, for the FP depth, the latest not-yet-consumed FS map is attached to the latest colour frame. So 250 ms is the upper bound, not the
  tail, and the standing skew is about 0.06–0.18 s [유도: HUD latency difference + FS period]. The recorded skew of the same 'latest vs latest' pairing
  had a median of 195 ms (p5 86, p95 365 ms; `data/20260907/0907_133358_observe/policy_obs/index.csv` `rgb_dt_ms`, n 514).
  (2) The `pair_tol_s` derivation at `config/hw_mpc.yaml:645` (lever arm 0.2855 m × 0.5 rad/s ≈ 1.1 cm) left out the dominant term, the
  line-of-sight rotation ρ·r·τ. At object distances ρ of 0.5–1.5 m this term is a few cm [유도].
- **Workaround**: today `--pose` does not load at all (the 2026-09-30 entry above). Even after it is restored, do not trust `obj_*` and the follow FF
  right after a re-registration or LOST. Do not run new features that put the object position into the reference path (e.g. DP plan correction for
  moving objects) in closed loop until the fix below.
- **Proper fix**:
  - When rebuilding FpWorker (inside the deleted sam2_live), return the input frame's `(t_capture, seq, solve_id)` with every solution. If there is
    no new solution, do not publish, or mark it "no new solve".
  - The receiving side drops duplicate `solve_id`s and bit-identical T, and composes only with **the fix of the solved frame**.
  - Even if an old T remains, publish `registering` while re-registering and STALE when there is no depth. Record `solved_seq` in the pose jsonl.
  - Pair the FS depth with the frame in the colour history closest to the FS stamp (the `LatestFrame.nearest` approach).
  - Regression test: a fake FpWorker running asynchronously at 150 ms + a moving camera + a stationary object → change in the object's MAP position 0.

### `ObjectAnchor`'s object velocity comes out **2×** at constant velocity — it differences the raw position against the EMA-filtered position (2026-10-05)
- **Symptom**: `ObjectAnchor.update` computes `v_raw = (p_new − self.p)/dt`, but `self.p` is already the position filtered with `pos_lp_alpha`
  (`rov_gui/control/object_nav.py:411`; the update is at 418). At constant velocity the filtered position lags behind by v·dt, so at steady state
  v̂ = v/α. At the deployed value `pos_lp_alpha 0.5` (`config/hw_mpc.yaml:657`) it is **2v** [유도; 2.00v with an in-memory replica]. The tests that
  check velocity all use α 1.0 (`rov_gui/tests/test_object_nav.py:300, 325-326, 338-339, 486`), so they do not catch this bias.
- **Impact**: all of the following are off by the same factor.
  - The follow mission's target velocity FF (`v_track`) becomes 2×.
  - `follow_ff_max_m_s 0.30` actually saturates at about 0.15 m/s.
  - The `predict()` extrapolation is off.
  - The threshold 'demote to STATION if the object speed exceeds 3×cap for 1 s' is off.
  The FF noise grows even on a stationary object. The FF published during the 0824 stationary segment was p90 0.062–0.110 m/s
  [computed in review, not saved: `data/20260824/0824_105751/mpc_*.csv`].
- **Workaround**: follow has had 0 water runs since 08-24 and `--pose` does not load either, so there is no harm right now. Fix it after the
  restore, before flying follow.
- **Proper fix**: difference raw positions against each other, or switch to a CV Kalman (α-β) based on capture time. Add a constant-velocity ramp
  test with α<1.

### Same-ID copies in the tag map (54, 65) cause wrong-copy matches during aggressive maneuvers (2026-08-24)
- **Symptom**: `nav_213749/fixes.csv` has "wrong-copy tag(s) dropped: 54, 65" (csv_t
  48.66–48.79, 49.76–50.13) — 0 occurrences in the 650-row baseline of normal tracking; it happens only during sharp yaw
  swings. In the same window a full-frame rejection (reproj 3.0 px > 3, t=48.49) and a 6.5 cm fix
  step came with it and contributed to the x0 perturbation late in run3 (not the main cause — 2.3 s
  downstream of the object pose collapse). The code does detect and drop them, but the very fact that the map has the same ID in two places is
  a latent risk that bites on every aggressive maneuver (the existence of the copies is derived from the log message text — needs a physical check).
- **Workaround**: remove/cover one of the physical copies of 54/65, or exclude those IDs from the map.
- **Proper fix**: reject duplicate IDs in the map build, or always exclude duplicate IDs at runtime.

### `object_nav.max_distance_m: 10` contradicts its own comment and **is breaking a test** (2026-08-24)
- **Symptom**: the value at `config/hw_mpc.yaml:312` is `10`, but the comment right next to it is
  "[측정: KNOWN_ISSUES 2026-08-09 — 0.3-0.8 m works, 2.4 m fails to register]", and
  the shipped default in `object_nav.py:102` is **1.20**. `test_object_nav.py ::
  test_the_shipped_config_resolves` asserts `max_distance_m <= 3.0`, so it is **currently red**
  (35/36). This gate is meant to filter out "absurd values", but at 10 m it is wider than the whole pool and
  effectively filters out nothing.
- **Why it was not fixed**: it is an operating limit that the pilot widened deliberately, so silently reverting it would change the
  workflow. However, **the 2026-08-23 ghost (camera 0.44 m) passes whether the value is 1.20 or 10**, so this value
  is not the cause of that failure.
- **Proper fix**: set it to the maximum distance actually in use (1.5–3 m) and make it match the comment, or
  write in the comment why it has to be wider and raise the test limit along with it.

### `path_fillet_m: 0.0` is breaking the MPCC fillet test (2026-08-24)
- **Symptom**: `test_mpcc.py :: test_the_fillet_must_exceed_the_cross_track_error` is red
  (24/25). The value that the **PID path-following vertex deadlock** (2026-08-18) entry above names as its workaround
  is exactly `path_fillet_m: 0.15`, but the current working tree has `0.0` (waypoint mode).
- **Workaround / fix**: see that entry. This entry records only **why the test suite is red** — do not misread it as
  something newly broken.

### The gantry encoder coordinates are **not registered** to the tag map (2026-08-23)
- **Symptom**: using the gantry→map transform in `config/fisheye_calibration.yaml` gives wrong answers.
  Three places are broken at the same time.
  1. `R_gantry_to_slam` is a value that `src/tools/refine_R_gantry_to_slam.py`
     fit on `data/20260528/20260528_215858_recording`, but **that run's anchor was
     tag 67** (in that run's `tag_poses.csv` tag 67 is the origin, 1e-10). The map's anchor is
     25. The two differ by **179.81°** (derived from tag 67's pose in `config/tag_map.yaml`).
     The cause is that `_exp_autopick_anchor_tag` in `src/gantry_panel.py` **re-picks the anchor on every run
     as the tag closest to the image centre**, and `_setup_tag_map` in `src/tagslam_core.py`
     re-expresses the map in that frame — i.e. every gantry run lands in a
     different world.
  2. **There is no translation calibration.** The `gantry_anchor_offset_mm` key does not exist in the YAML at all, so
     the dashboard runs off to the "first-sample-zeroed" fallback. The raw ‖p_est − p_gantry‖ is
     p50 **1688.4 mm** (recomputed from the above run's CSV).
  3. `gantry_to_slam_scale: 1.03574608` is **anisotropic** — per-axis stretch
     [0.9891, 1.0465, 1.0440], spread 0.057 (recomputed from the above run's CSV). Refraction and tag size are both
     isotropic, so neither is the culprit. Moreover, that fit had **0 mm of Z travel** and a camera/gantry path ratio of
     1.78× (the tool's own warning threshold is 1.3×).
- **Workaround**: **use the camera instead of the encoders.** The gantry panel's `Tag map position`
  card (`src/gantry_map_pose.py`) does PnP directly against the tag map via `rov_gui/control/tagnav.py` and outputs
  anchor-25 coordinates — none of the three values above is on that path. **`R_gantry_to_slam` and
  `gantry_to_slam_scale` are for visualization only; do not cite them as the basis for any number.**
  (Fortunately, neither has ever gone into the SLAM solution, `config/tag_map.yaml` or `rov_gui/`, so
  the map itself is not contaminated.)
- **Proper fix**: after fixing the anchor at 25 (the entry below), refit on a trajectory that moves **≥3 m on all 3 axes**,
  and trust a uniform scale only when the per-axis stretches converge to isotropic. Before that,
  separate the two suspects for the anisotropy one at a time — `SCALE_MM_PER_UNIT` X=8.25 mm/unit
  (`src/gantry_runner.py` confesses it was "copied verbatim from whisker_dragging.py") with
  a 1 m commanded move + a tape measure, and `fx/fy = 1.0411` (with square pixels it should be 1.000) with
  a recalibration using more frames. The translation comes out by reading the card and the encoders at the same instant and
  subtracting — a 10-minute procedure that needs just one parking.

### The gantry Experiment tab re-picks the anchor **automatically on every run** (2026-08-23)
- **Symptom**: `_exp_autopick_anchor_tag` (`src/gantry_panel.py`) takes "the tag closest to
  the image centre" as the anchor, so the `camera_trajectory.csv` of two runs are in **different
  world coordinates**, and that fact is not written anywhere in the files. Also, the default anchor value is `1`
  (`gantry_panel.py`'s spin box, `tagslam_core.DEFAULT_ANCHOR_TAG_ID`), yet **tag 1 does not
  exist in `config/tag_map.yaml`**. On top of that,
  `tagslam_core.DEFAULT_TAG_SIZE_M = 0.085` is **exactly half** of the actual 0.170, so if
  `--tag-size` is left out on the CLI, a 0.5×-scale map comes out silently.
- **Workaround**: to compare trajectories of existing runs, find each run's anchor in `tag_poses.csv`
  (the tag at the origin), map it onto anchor-25 with `world_T_anchor` from `config/tag_map.yaml`,
  then compare. `src/tests/test_gantry_map_pose.py` does exactly that.
  If you need the live position, look at the `Tag map position` card, not the Experiment tab
  (that path does not use the anchor at all — `TagMap` uses the file's `anchor_tag_id: 25` as is).
- **Proper fix**: in PnP-only mode, turn auto-pick off and force the map's `anchor_tag_id`.
  `DEFAULT_ANCHOR_TAG_ID`/`DEFAULT_TAG_SIZE_M` are a value absent from the map and a half value,
  which makes them dangerous as defaults, so remove them or have them read from the map.

### Gantry hold-to-jog commands an **infinite distance** (2026-08-23)
- **Symptom**: `_start_jog` (`src/gantry_panel.py`) fires
  `jog_single_axis(..., position_units=999999.0 × dir, relative=True)`.
  X is 8.25 mm/unit, so that is **about 8.25 km**. The only thing that stops it is the button's `released` →
  `stop_axis(mode=1)`, and **both issuing and stopping run synchronously on the GUI thread**.
  If the release signal is swallowed or the event loop stalls, the axis travels to the firmware soft limit or the physical
  limit switch. The panel no longer manages soft limits.
- **Workaround**: jog in short, separate presses. **While the ROV is under the gantry, use a waypoint CSV
  instead of jogging** (`./c3 gantry --waypoints-csv …`).
- **Proper fix**: bound the travel with a `JOG_MAX_TRAVEL_MM` constant (e.g. 200 mm) and
  arm a `QTimer.singleShot(travel/speed + margin, stop)` watchdog. The meaning of hold
  changes from "as long as it is pressed" to "at most N mm at a time", so the pilot's agreement is needed. Moving issuing · stopping
  to a worker thread is **not recommended** — adding a fourth concurrent caller to a `.so` with no internal mutex
  is worse than the problem being fixed.

### `c3_camera/tests/test_option_sweep.py` fails 20/32 — API drift (2026-08-23)
- **Symptom**: in `./c3 test`, `TypeError: StreamConfig.__init__() got an unexpected
  keyword argument 'mono_encode'` breaks 20 tests. In `c3_camera/config.py`,
  `StreamConfig` has no `mono_encode`, yet the tests keep passing it. This is the committed state, and it is
  unrelated to the gantry work (both files are identical to HEAD).
- **Workaround**: this file's results in `./c3 test` are currently not a signal. The other files
  (host_depth 45/45, offline 58/58, preflight 43/43, src/tests 21/21) are fine, so
  judge by those alone.
- **Proper fix**: check in the git log when and why `mono_encode` was dropped, then
  settle it one way or the other: delete it from the tests or restore it in `StreamConfig`.


### The geofence was **removed entirely** — nothing knows about the pool walls (2026-08-14)
- **What was removed** (explicit pilot request): the orange dashed `GEOFENCE` box on the plot,
  the path check before START (both ends of a line / the four corners of a square — now that circle exists,
  this check cannot be expressed as a vertex set and the rim would have to be sampled), and the automatic
  disengage on leaving the box mid-run. `geofence_ned`·`geofence_frame` in `hw_nav.yaml` are no longer
  read, and the `geofence_ok` column of the MPC CSV is gone too.
- **What was lost**: now **nothing at all stops the vehicle because of its position.** A line that goes
  9 m outside the pool still arms as is, and if the vehicle is pushed toward a wall mid-run, the controller keeps
  going. **The circle (2026-08-17) is the shape that goes farthest from the entered tag** —
  the centre is R from the tag and the opposite rim 2R (1.0 m at the deployed radius of 0.5 m), so if you set the vehicle next to the tag
  and press START, it travels to a point 1 m away. Other than a person checking the placement,
  there is still no means of preventing this. The remaining protections are E-STOP (Esc · header · MPC panel) · DISENG · sink deadman (500 ms) ·
  the engage gate (ARMED / MANUAL / fresh tag fix · telemetry) · disarm mid-run ·
  telemetry stopping · leaving the mode · automatic release on tag dropout · axis authority cap.
- **Workaround**: in pool runs, **a person checks the path when placing it**, and the pilot
  keeps a hand on E-STOP. The solid `POOL` line on the plot still draws the walls, but
  **it is only a drawing and enforces nothing**.
- **Proper fix**: if it is needed again, rather than reviving the fence, it is better to **clamp the reference**
  instead of refusing (`geofence_clamp` existed for that purpose) — that keeps the setpoint from crossing the wall
  without blocking the placement. git: the state just before this commit.

### PID path following **deadlocks permanently 10 cm before a vertex** — leash × corner brake × deadband (2026-08-18)
- **Symptom**: on a square with `--mpc` panel `pid` + `path_fillet_m: 0.0` (waypoint_vertex_stop),
  the vehicle stops 10 cm short of one vertex and **halts forever without any warning.** solver_status 0,
  21 tags · reproj 1.85 px · ambig 0 · tag_age 0.15 s · axis saturation 0 % — every instrument reads normal.
  In the 2026-08-18 pool session it happened **at the same vertex in two consecutive runs**; each stood still for
  40 s / 45 s respectively, until the pilot released it by hand
  (`data/20260818/0818_143802/mpc_143802.csv` stopped at s=1.93,
  `.../mpc_143938.csv` stopped at s=5.92 — both at the s≡2.0 m vertex of the path anchored on tag 37, i.e.
  the corner diagonally opposite the origin tag. Hull wander while stopped p95 1.9 cm).
- **Mechanism (holds only when all three coincide)**:
  1. `PathCursor.step`'s leash `cmd = min(cmd, theta + lead_m)` — when the hull stops, the setpoint
     stops too. → the position error has a **hard cap** at `path_lead_m` = 0.10 m
     (`e_along` p50 0.099; **83.6 %** of engaged ticks exceed 0.095 m = leash constantly saturated).
  2. `speed_profile` brakes v_ref down to creep 0.02 at a vertex with no fillet → once the setpoint passes the vertex
     by 2 cm, **the tangent turns 90° onto the next side.** PID's velocity FF `kd·v_ref_b`
     **vanishes to 0** on the surge axis (kd_x = 59.7 N·s/m × 0.10 m/s = 5.97 N evaporates entirely).
  3. The remaining upper bound: `kp·0.10 + i_max` = 51.73×0.103 + 4.0 = **9.33 N** [유도] — measured uX mean
     9.47 N (sd 0.97). While moving on straight segments it was 11.3–11.6 N.
     9.47 N / `axis_gain.surge_n` 60 = ax_surge **0.158**, `pwm_dev_us` mean **26 µs**
     — just above both the T200/Basic ESC null zone of ±25 µs [스펙] that hw_mpc.yaml notes itself and
     the threshold of the measured whole-loop `speed = 0.692·(|axis| − 0.096)`. Actual speed 0.000 m/s.
  → **The hull cannot move, so the reference cannot move; the reference cannot move, so the command cannot grow; the command cannot grow, so
     the hull cannot move.** It is a stable fixed point, so it can never get out on its own.
- **Background condition**: in this run the leash was saturated from the start — commanded 0.100 m/s, actual 0.051 m/s
  (5.92 m / 116 s). That is, the leash, not the speed box, was effectively the control law, and the corner FF loss of
  2 N directly decided which side of the threshold it fell on.
- **Workaround**: reverting to `path_fillet_m: 0.15` in `config/hw_mpc.yaml` (the value used by the best run on record, 0817_110145)
  removes the vertex itself, so both v_ref and the tangent become continuous. Along with that, lower the speed box
  to 0.05–0.06 m/s (what the vehicle actually achieves).
- **Proper fix**: two things are needed, separately.
  (a) The integral clamp `pid.i_max_n` = [4.0, 5.0, 5.0] N is **lower than the force needed to escape the deadband** —
      the textbook cause of "permanent stop on a steady-state error". Raise it to at least 8–10 N, or use a conditional clamp
      that releases only while the leash is saturated.
  (b) **There is no deadlock watchdog.** If `e_along ≥ 0.95·lead_m` && `|v| < ε` persists for N seconds, it should
      warn/abort. Right now it fails silently, exactly like the 2 MPCC deadlocks (memory: mpcc-contouring-control),
      with solver status 0.

### Miss reasons are logged only at 1 Hz, so the CSV under-reports dropouts 15.6× (2026-09-06)
- **Symptom**: because of [`workers.py:290`](rov_gui/control/workers.py#L290)'s
  `if t - self._last_miss_note > 1.0:`, rejected frames leave only **one line per second**
  in `fixes.csv`. Measured: in `data/20260906/
  0906_192348/nav_192348`, **390 rejections were recorded as 25 rows** (15.6×).
- **Why it is dangerous**: **computing the acceptance rate from the row ratio of `fixes.csv` gives a wrong answer**
  (accepted rows are all there, rejected rows were thinned to 1 Hz). The 2026-09-06 diagnosis actually
  fell into this trap, and the true number comes only from a reproduction that re-solves
  `frames.csv` + `detections.csv`.
- **Workaround**: compute the acceptance rate by reproduction
  (feeding the recorded corners unchanged into `rov_gui/control/tagnav.py` reproduces accept/reject
  100% — the accepted frames' `reproj_rms_px` agree within a max |Δ| of 0.0051 px).
- **Proper fix**: keep the message text at 1 Hz, but update a per-reason **tally counter**
  every frame and leave it as one line at the end of the run (or add a reason column to `frames.csv`).


### **Relaunching the station within 90 s** joins the same run folder, and `--record-stereo`/`--record-depth` recordings **overwrite the previous run's** (2026-10-12)
- **Symptom**: `runstore.run_dir` joins a folder of the same kind if that folder started within `JOIN_WINDOW_S` 90 s
  (`rov_gui/runstore.py:144`). But after opening the folder, the recorder opens `index.csv` **with `"w"`** and rewrites frame numbers from 0
  (`rov_gui/perception/frame_record.py:320`, shared by StereoRecorder · the depth recorder). So if you change a flag and relaunch right away
  (exactly what a `--policy-fs-schedule`/`--fstereo-process` A/B does), the new run's recording overwrites the previous run's `index.csv`·`000000.png…`·`meta.json`.
  `mission_log.txt` is also replaced by the new session's log (`window.py` session log). The 2026-10-12 review critic reproduced it in scratch by running StereoRecorder twice
  (only the second run's rows remain in index.csv — a scratchpad artifact, so not citable; the code evidence is the two lines above).
- **Workaround**: between runs with recording on, **wait more than 90 s** before relaunching. Check from the launch log · the `data/<날짜>/` (`data/<date>/`) folder names that a new folder
  was created.
- **Proper fix**: when the recorder opens a folder that already has an `index.csv`, have it go to a new subfolder (`stereo_2/`, a launch id suffix) or append with `"a"`
  and continue the seq numbering. Or have `run_dir` hand out a different leaf per process (launch) (joining only among writers of the same launch).

## 🐛 Test / script traps

### `filter_plan_yaw` rebuilds `PlanMsg` by listing its fields — a new trailing field **silently disappears** on the live path where the yaw filter is on (2026-10-05)
- **Symptom**: `rov_gui/control/policy_frames.py:1329-1331` builds a new object with `PlanMsg(plan_id=…, obs_t=…, arrival_t=…, rp=rp_msg)`.
  `PlanFilter`'s dilation preserves every field via `dataclasses.replace`, but this function carries over only the fields written out.
  `policy.yaw_ref_filter` is on (`config/hw_mpc.yaml:846-849`).
- **Workaround**: when adding a field to `PlanMsg`, fix this function along with it.
- **Proper fix**: switch to `dataclasses.replace(msg, p_ned=…, yaw=…, rp=…)` and test that every field is preserved.

### `detect_ms` in `fixes.csv` is not the detection time but the **PnP solve time** (2026-10-05)
- **Symptom**: `NavFix.detect_ms` of the main feed is the `perf_counter` elapsed time measured from the start of `TagNav._solve_raw` (`rov_gui/control/tagnav.py:673, 765, 881`).
  So it is logged as 0.1–1.6 ms. The actual detection time (`rov_gui/control/workers.py:410-412`) goes only to the overlay. Real detection values such as 4–6 ms
  show up only in the RGB fallback rows.
- **Workaround**: do not cite `detect_ms` of main rows as the detection time.
- **Proper fix**: rename the field to `solve_ms` and record the detection time in a separate column. Mark the day of the change as a records boundary.

### `slam/9_9_26/` became `slam/grasp_9_9_26/`, but the old path remains, so **the ablation runner silently skips runs** (2026-10-01)
- **Symptom**: `~/Desktop/data collection/slam/` now holds only `grasp_9_4_26`, `grasp_9_9_26`, `peg_in_hole_9_27_26`, `peg_in_hole_9_28_26`
  (2026-10-01 `ls`; `data/20260915/0915_131900_heldout_D_vs_A/D_dinov3b_ep195.log:2` loaded from the old path, so the rename came after that).
  Places that point at the old path: `tools/run_ablations.sh:44-45` (`DS_9926_DEPTH`·`DS_9926_RGBD` → all of the E/A/C/D, rgbd_*, depth_7d_*/rgbd_7d_* specs),
  `external/UMI_aquatic/diffusion_policy/config/task/umi_rgbd_5d.yaml:115`, `umi_rgbd_7d.yaml:127`, `umi_handheld/build_dp_rgbd_zarr.py:5`
  (docstring), `external/UMI_aquatic/tests/print_rp_label_stats.py:6,48`, the `.zattrs` of `dataset_rgbd.zarr.zip`. At `:97` the runner
  prints "!! <name>: no dataset … — skipped" and **moves on to the next run** (no crash) — start an overnight batch and in the morning those runs are missing.
- **Workaround**: right after starting a batch, check the runner log for "skipped", or symlink `slam/9_9_26 → grasp_9_9_26`.
- **Proper fix**: update the paths above to `grasp_9_9_26` (or keep the symlink deliberately and record that here), and have the runner treat a missing dataset
  as a non-zero exit rather than a skip.

### The "new can 0913 closed-loop baseline" in `docs/DINOV3_ENCODER_STUDY.ko.md` is **mostly not the can** (2026-10-01)
- **Symptom**: `:25`, `:134` (original figures in `docs/measurements/dinov3_encoder_study_20260913/` followup_baseline-0913-yaw.json) group the 8 folders of 0913
  as "new can" and set A's r +0.116 and the pooled r +0.034 of the 3 L/R-balanced runs (163921/170213/173702) as the baseline DINOv3 has to beat.
  But 5 of the 8 folders (161733/163921/165748/170213/173702) are a black long-necked doll (teal head · orange feet), and **all 3 balanced runs are the doll**
  [측정: `data/20260913/0913_163921/policy_obs/rgb/000768.jpg` doll, `data/20260913/0913_155559/policy_obs/rgb/000037.jpg` orange-taped can —
  checked directly 2026-10-01; the remaining runs were checked by a workflow verifier the same day, 1–2 images per run]. The underwater records showing the orange can are 6 runs
  (0911_171926_observe/173045/175144, 0913_155559/160850/161350), only 5,330 JPEGs, and in the three 0913 runs the can is bunched to the left
  (L/C/R 37/1/0, 9/0/2, 10/0/0 — same json). The 0914–0921 water runs are the doll.
- **Impact**: those r values are not a "new can" baseline but mostly a doll baseline. There are still no L/R-balanced runs with the orange can by which to judge the yaw response.
- **Proper fix**: correct the labels on those two lines of the doc + capture new observe runs with `--record-depth`, with the orange can placed alternately left and right.

### Running `rov_gui/tests` overwrites the git-tracked generated MPCC solver **with the model drag (4.03)** (2026-09-30)
- **Symptom**: `rov_gui/control/_mpcc_gen/heavy_gripper/` (`.c/.o/.so` + `acados_ocp_mpcc.json`) is committed generated code, and
  HEAD was generated with the drag on the real vehicle, `-86.7 / -133.8`. The tests (e.g. `HwMpcc(MpcConfig(...))` in `test_path_cost.py:677`)
  build `AcadosMPCC()` (`build=True, generate=True`) without a `plant:` override and rewrite the same folder with `-4.03 / -6.22`
  (confirmed 2026-09-30 by comparing the json `f_expl_expr` after the full test run). If you `git add .` out of habit, a solver with the wrong drag gets committed.
- **Harmless for flight**: on every GUI launch, `MpcWorker` **regenerates** in the order `HwDobMpc` (drag override applied) → `HwMpcc`
  (`rov_gui/control/workers.py` 1170-1203). A headless regeneration in the same order brings `.c/.o/.so` back byte-identical to HEAD.
- **Workaround**: if `_mpcc_gen` shows up in `git status` after the tests, run `git checkout -- rov_gui/control/_mpcc_gen/` before committing, or
  launch the GUI once. A difference only in the env paths (`Makefile` · the json's `code_gen_opts`) is one that a GUI launch produces anyway.
- **Proper fix**: have the tests generate into a temporary directory (inject `GEN_DIR`), or take the generated files out of git (.gitignore) and leave them to the launch-time build.


### depthai has to be initialized **before** cv2/torch for the USB camera to be enumerated (2026-09-06)
- **Symptom**: if a process imports `cv2` and FoundationStereo (=torch+CUDA) first,
  then in that process only, the USB OAK-D-W **does not show up in `getAllAvailableDevices()` for more than 30 seconds**.
  Another Python launched at the same time sees it immediately. The C3, being PoE, is fine, so it reads as "the USB camera is broken"
  [측정 2026-09-06: inside the tool, only the C3 throughout 20 × 1.5 s retries; a separate interpreter saw
  both].
- **Mitigation**: `import depthai` → enumerate → **open both devices first** → then the heavy imports.
  `rov_gui/tools/depth_two_cameras.py` and `depth_live_view.py` are written in that order.
- **Proper fix**: narrow the cause down to the XLink initialization order (presumably), and if needed enforce it with a helper that
  enumerates once at the entry point.

### `kill -9` on a camera tool leaves the device booted, and the watchdog does not revive it (2026-09-06)
- **Symptom**: after SIGKILL the USB PID changes from `03e7:2485` (unbooted) → `03e7:f63b` (booted),
  `getAllAvailableDevices()` returns an empty array, and a direct connection gives `X_LINK_DEVICE_ALREADY_IN_USE`.
  It did not recover on its own for more than 2 minutes.
- **Trap**: a process launched from a heredoc has the command line `python -`, so **`pkill -f <스크립트명>` (`pkill -f <script name>`)
  does not catch it**. So it looks like "nobody is holding it, yet it does not work".
- **Mitigation**: find it by the interpreter path and **SIGTERM** it. A normal exit gives the device back.
  ```
  ps aux | grep 'rovgui-pose/bin/python'
  kill -TERM <pid>
  ```
  Measured: 6 seconds after SIGTERM the USB PID was back to `03e7:2485`; no need to unplug and replug.
- **Proper fix**: give tools that open a camera a SIGTERM handler that guarantees `device.close()`,
  and do not `kill -9` the tools.


### If `test_object_nav.py` runs first, 23 tests in `test_policy.py` break — order dependence (2026-09-06)
- **Symptom**: as a file on its own, `test_policy.py` **passes 42/42**. In the full suite, 26 fail.
  Minimal reproduction: `pytest rov_gui/tests/test_object_nav.py rov_gui/tests/test_policy.py`
  → **23 failed, 55 passed**. `pytest rov_gui/tests/test_replay.py
  rov_gui/tests/test_policy.py` → 51 passed. So the source of contamination is `test_object_nav.py`.
- **It shows up in two forms**: (1) the plan intake dies —
  `w.replay["installed"] == 0`, `rp["late"] == 0` and so on: `MpcWorker` receives no plans at all;
  (2) wall-clock dependence — `engage refused: imu stale (0.44s)`. The latter depends on the wall-clock time the previous file
  consumed, so if unrelated code shifts by just a few ms, fail/pass flips.
- **Confirmed it is not caused by my change**: on 2026-09-06, with all 4 `--record-depth`-related files reverted,
  running the same command still **fails the same 23** (23 in the reverted state, 24 before reverting — the extra
  1 is the imu-stale timing case of (2) above).
- **Workaround**: judge the policy tests by running the file on its own. Do not use the full suite's failure count as a regression
  signal.
- **Proper fix**: find the global state that `test_object_nav.py` leaves behind and isolate it with a fixture, and
  switch the clock-dependent checks to monotonic injection.


### At 1280×800, **the colour stream drops 3.3% of the device frames** (2026-08-04)
- **Found**: 2026-08-04, while verifying a hardware take after fixing maxres from 8.4 → 29 fps.
- **Symptom (exact values counted by sequence number, same pipeline as `bench_stereo`, 10 s/row)**:

  | | age | drops | fps |
  |---|---|---|---|
  | colour ON | 153.0 ms | **3.67%** | 28.90 |
  | colour OFF | 161.5 ms | **0.33%** | 29.90 |

  It does not appear with the default profile (320×200) or in near mode (19.7 fps) — the only combination with no headroom
  is 1280×800 · 29 fps.
- **It is not the disk**: raising the writer from 6→12 threads + the queue from 128→256 did not improve it.
  It looks like XLink/USB bandwidth, but **the cause is not settled**. For reference, the device
  connects as `usb=SUPER` even when `maxUsbSpeed=SUPER_PLUS` is requested.
- **Workaround**: removing `color` from `record.streams:` is enough (0.33%). Frames are recorded with timestamps,
  so the take itself stays valid even with losses.
- **Proper fix**: lower the colour MJPEG quality/resolution (97 → 80, or 720p instead of 800p)
  to free up bandwidth, and re-measure with `umi_handheld/bench_stereo.py`. Until something downstream actually
  starts using the colour, there is no basis for deciding — at present nothing reads it.

### At 1280×800, the **150 ms residual latency is the resolution itself**, so settings cannot reduce it (2026-08-04)
- **Symptom**: even after fixing the stereo input queue from 8 → 1 to strip out ~95 ms, frames have already aged
  **153 ms** when they arrive at the host (~4.4 frames at 29 fps). The preview shows up late by that much.
- **All candidate causes were ruled out** (all measured, device held fixed): not the host loop (with no
  display, 640 kHz; `draw_overlay` 3.0 ms) · not XLink bandwidth (removing right/color from `streams`
  still gives 153 → 159 ms, no change) · `setXLinkChunkSize(0)` has no effect · host queue size has no effect ·
  post-processing filters have almost no effect (152.8 ms with median/temporal all off). The only remaining variable is the resolution:
  with the same filter chain, **640×400 gives 36.1 ms**.
- **Workaround**: for latency-sensitive work (precision teaching, etc.), use `configs/pipeline.yaml` (320×200
  depth). As long as 1280×800 is used, the ~150 ms comes with it.
- **Proper fix**: it is the pipeline depth inside the device, so depthai settings cannot touch it.
  Reducing it means shortening the frame period (latency ≈ number of stages × period), but at 800p the stereo core
  tops out at 29–30 fps. At 720p (same MinZ, 10% fewer rows) 21.4 fps has been measured, so
  not much can be expected — do not conclude definitively without re-measuring.

### `record.py` is depthai **2.x only**, but the base conda environment has 3.5.0 (2026-08-04)
- **Symptom**: running `--source device` in base (`/home/bdml/miniforge3`) immediately gives
  `AttributeError: module 'depthai.node' has no attribute 'XLinkOut'`. `dai.node.XLinkOut`,
  `dai.RawStereoDepthConfig`, `Device.getOutputQueue` all disappeared in 3.x
  (record.py:154-159, 177, 197, 216, 403-409).
- **Workaround**: run it only with an interpreter that has depthai 2.32.0.0 —
  `~/.venvs/c3-depthai/bin/python`, or conda `robust` (installed 2026-08-05, but read the
  numpy entry above first). `test_oak_depth.py` in the same folder
  branches on `DEPTHAI_MAJOR_VERSION` for both sides, so it also runs in base, and therefore the two
  scripts run on **different stacks**.
- **Progress (2026-08-05)**: the `c3_camera` side was resolved by the repo-root `./c3` wrapper, which pins the interpreter and the working
  directory. `umi_handheld/record.py` is **still** as it was —
  it has neither a wrapper nor a version branch, so it depends on which python `python -m umi_handheld.record`
  picks up.
- **Proper fix**: give record.py the same version branch as test_oak_depth.py, or
  put a wrapper like `./c3` in umi_handheld too, to pin the interpreter.

### `xlink_out_queue` is applied but **not measured yet** (2026-08-06)
- **What was done**: `build_device_pipeline` was leaving the input queues of the `XLinkOut` nodes at the DepthAI default
  (8, blocking) — the only buffering stage in the chain with neither a setting nor a measurement.
  In the same way as `stereo_input_queue`, a `record.xlink_out_queue: {size: 1, blocking: false}`
  key was created and applied to all of left/depth/right/color.
- **Why it is promising**: when the stereo **input** queue was changed from 8→1, latency dropped from 252 → 153 ms
  (record.py:283-297). The output queue is the same kind of spot, yet it was the only one left untouched.
- **Unverified**: the bench camera was not connected at the time of the work
  (`dai.Device.getAllAvailableDevices()` → none), so no before/after comparison could be made. **Do not claim
  it has an effect.**
- **How to measure**: `python -m umi_handheld.bench_stereo --config configs/pipeline.yaml --seconds 10`
  — the `age` (frame age on reaching the host, median in ms) and `drop%` (missing sequence numbers) columns were added this time,
  so it can be measured now. Log it side by side with a row reverted to `xlink_out_queue: {size: 8, blocking: true}`
  and compare (valid only with the same scene · the same device).
- **Expectation**: as in the `KNOWN_ISSUES` entry below, the residual ~150 ms at 1280×800 is the resolution itself.
  The 640×400 baseline is 36 ms, so the share that can be stripped out here is smaller than that.

### The output directory `zarr/` in the repo root hijacks `import zarr` (2026-08-06)
- **Symptom**: running `python -m umi_handheld.build_zarr` in `robust` gives
  `AttributeError: module 'zarr' has no attribute 'open'`. The repo root is `sys.path[0]`, and
  the artifact directory `zarr/` sits there, so **when the real zarr is not installed** that directory
  is picked up as a namespace package (`import zarr` → `_NamespacePath(['<repo>/zarr'])`).
- **Worse, it ruins the diagnosis**: since `import zarr` succeeds from the repo root,
  it is misread as "this environment has zarr". **`robust` has no zarr** (checked with `pip list`:
  robust none, `~/.venvs/c3-depthai` 2.18.3). An installed regular package wins over the namespace
  portion, so in the venv it works correctly even from the repo root.
- **Workaround**: run `build_zarr.py` with `~/.venvs/c3-depthai/bin/python`
  (measured: from the repo root `zarr 2.18.3` resolves correctly; the session_syn 72-frame build succeeded).
  Check the environment from **outside** the repo — `cd /tmp && <python> -c "import zarr; print(zarr.__version__)"`.
- **Proper fix**: install zarr in `robust` (the numpy<2 constraint needs checking), or rename the artifact
  directory to something that does not collide with the module name, like `zarr_out/`. The latter is the root fix, but
  all 7 existing dataset paths change.

### The C3 factory calibration is an **underwater** value, but code · docs assert "IN-AIR" → depth captured in air reads ~1.33× too large (2026-08-04)
- **Found**: 2026-08-04 (while planning the OAK-D W handheld pipeline; re-verified after the user pointed it out)
- **Fact 1 (vendor)**: the MarineSitu C3 **ships with an individual underwater calibration per unit**.
  Blue Robotics product page: *"Each unit is individually calibrated for underwater
  operation"*; on the forum, Tony White: *"the calibration Marine Situ performs takes
  place entirely underwater. It uses the standard checkerboard approach with openCV"*.
- **Fact 2 (measured on our EEPROM)**: the 10 `datasets/*/calibration.json` are all one identical set, and
  measuring the actual field of view by **forward-projecting** the K+distortion (rational, 8 coefficients) they hold gives
  **CAM_A 63.7° / CAM_B 85.6° / CAM_C 85.1°** (HFOV). The in-air OAK-D-W spec is
  **95 / 127 / 127°**, and the flat-port Snell prediction is **67.2 / 84.3 / 84.3°** → all three cameras are
  within 3.5° of the underwater prediction and up to 41.9° off the in-air spec.
  (artifact: `python calib/fov_audit.py --audit`; method · control experiment [calib/FOV_AUDIT.md](calib/FOV_AUDIT.en.md))
  The three cameras independently give the same conclusion. That is, EEPROM = **underwater** calibration.
- **Symptom**: capturing **in air** with this calibration gives `Z_reported = (fx_water/fx_air)·Z_true
  ≈ 1.33 · Z_true` — i.e. **depth comes out about 33% too far**. Rectification is also off because the medium
  differs. `c3_camera/datasets/*` and `recordings/*` were all captured indoors (in air), so
  **the depth scale of the existing C3 RGB-D datasets carries a systematic error**. Conversely, in actual water
  this calibration is **correct**.
- **Where the wrong wording is baked in** (all of them our own comments, hardcoded without evidence):
  `c3_collect.py:330` (this string **is recorded as the calibration.json `note` of every dataset**),
  `dataset.py:593, 649, 909`, `geometry.py:15`, `TESTING.md` (describes T2 as "all in air;
  underwater, the scale is ~1.33× when uncorrected" — the sign is reversed).
- **Workaround**: read C3 depth captured in air as **÷1.33**, referenced to the centre. However,
  **it is not a clean scalar** — the medium mismatch extends to the distortion model, so an extra radial
  error builds up toward the periphery. The MinZ table (400p **300 mm**) is based on the underwater fx, so
  **the actual minimum distance in air is ≈225 mm, and that is reported as 300 mm**.
  The reported floor of 300 mm (and 150 mm with `--extended`) is now measured
  `[측정: c3_camera/datasets/*/depth/, recordings/*/depth/ — exhaustive, 47,270 frames,
  floor exactly 300/150 mm, 0 pixels below it]`. The in-air physical distance of ≈225 mm is still
  `[유도: substitute fx_air = fx_water/1.333. 미실측 — depth_accuracy/rungs.csv absent,
  a tape-measure check has never been run]`.
- **Proper fix**: (a) correct the wording in the 5 places above to the facts and, in `calibration.json`, state
  `medium: "water"` + the evidence; (b) when the camera is back on the network, read the EEPROM metadata
  (`batchTime`/`boardCustom`/`productName` from `getEepromData()`) to nail down the provenance;
  (c) take a separate in-air calibration for in-air work with a checkerboard, so it can be selected per medium.
  (a) alone removes most of this entry's risk, so priority is high. Delete this entry once fixed.

### The water surface · pool depth in `config/config.yaml` **contradict the ROV hardware records** (2026-08-23)
- **Symptom**: `water.surface_height_m: 0.8255` / `pool.depth_m: 1.143` have not changed even once since the 2026-05-25
  initial commit (`git log -- config/config.yaml`), yet the ROV hardware records exceed
  both.
  1. **A lower bound with no pressure sensor involved**: the lowest `z_ned` among accepted fixes is **−1.296 m**
     (p50 −1.138, n=22094 — `data/*/*/nav_*/fixes.csv`, tallied 2026-08-13/14).
     The floor tag mat is z=0 and +z is down, so this means the vehicle was 1.296 m above the mat, **submerged**,
     while looking at the floor tags → the water column above the mat is at least 1.3 m.
  2. **From pressure**: at every engage, `StateAssembler.calibrate_z_offset`
     sets `z_off = z_tag − depth` (`rov_gui/control/state_assembler.py:104-108`).
     Reconstructing `−z_off` (= water-surface height above the mat) from the `datum p0` z in events.log and
     `GLOBAL_POSITION_INT.relative_alt` in the same run's `*_rov.jsonl` gives
     2026-08-18 = 1.386 / 1.409 / 1.415 / 1.418 / 1.440 m,
     2026-08-17 = 1.641 / 1.654 / 1.664 / 1.686 / 1.722 m.
  That is, the config is ~0.6 m too shallow, and even the 1.143 m of the "45 in pool" is less than the 1.3 m lower bound.
  (`docs/MEASUREMENT_AUDIT.md` has already stamped the pool-dimension line in `claude.md` as UNVERIFIED, and
  the width written there also disagrees with the config.)
- **Side finding — same water, yet two sessions differ by 0.24 m**: this is because ArduSub's depth 0 point is
  **reset at every boot** (depth = (ground_pressure − p)/9800/`BARO_SPEC_GRAV`,
  ground_pressure is re-acquired at boot · at the preflight baro cal). Within one day it is consistent to ±3–4 cm,
  so **a water-surface height derived from pressure is valid only within that session**. The Bar30's absolute
  accuracy is ±200 mbar (≈±2.04 m of fresh water), so absolute pressure cannot settle anything
  [스펙: bluerobotics.com/store/.../bar-depth-pressure-sensor/].
- **Workaround**: the two config values — **do not cite them as the basis for any number**. Since the refraction correction
  uses `water.surface_height_m` (`src/tagslam_core.py:880`), the gantry/ZED refraction results also
  hinge on this value — do not trust refraction recomputations for water-filled days.
  The fallback at `rov_gui/backends/hardware.py:1337` (absolute pressure − 1013.25 hPa) swallows the Bar30's absolute
  offset whole, so it **must not be used as evidence for the water level** (while GLOBAL_POSITION_INT is alive
  that path is not taken, but if it dies, it silently switches over).
- **Proper fix**: measure the pool depth and that day's water level with a tape measure (±3 mm, 10 min) and write them
  into the config with the date. On the ROV side, if ① the vehicle is booted **out of the water** so the 0 point is set at atmospheric pressure,
  and ② `z_off` and depth at engage are recorded in `*.meta.json` (currently reconstructable only from the events.log +
  jsonl combination), the water level becomes a quantity recorded per run. The Bar30's in-hull z
  offset (electronics-enclosure end cap) has no measurement anywhere in the repo, so **±5–10 cm is the limit for the absolute value** —
  a **change** measured with the vehicle seated in the same spot has the lever arm cancel out and comes out at the mm level.

### The `dataset.py` telemetry CSV schema is "first-message winner-takes-all" — the columns differ per session (2026-08-03)
- **Found**: 2026-08-03 (API-mapping workflow while writing `c3_option_sweep.py`)
- **Symptom**: per group (telemetry/imu_rov/control), `DatasetWriter.mavlink_rows` **locks in the header as the
  field set of the single message that arrives first** (dataset.py:510-517), and since it is
  `DictWriter(extrasaction="ignore")` (dataset.py:195), fields of other types that come later
  are **silently dropped**. Measured: `datasets/dataset_20260803_105221/telemetry.csv` has only
  the 6 columns `altitude,lat,lng,pitch,roll,yaw` because its first record was AHRS2, and
  **the 1169 VFR_HUD rows mixed into the same file are blank in every column** (11 message types / 7935 rows).
- **Why it is worse in a sweep**: which message arrives first is a **race**, so even with identical
  vehicle traffic the telemetry.csv schema differs from cell to cell and **cross-cell comparison breaks**.
- **Workaround**: when using telemetry, filter by `msg_type` first, and interpret an empty column
  not as "that message has no such field" but as "it was cut from the header". Cross-check the original
  against `mavlink.message_counts` in `metadata.json`.
- **Proper fix**: settle each group's header in advance as the union of the fields of `ALL_MESSAGES`, or
  split the files by message type (`telemetry_VFR_HUD.csv` …). Delete this entry once fixed.

### The `mpc_tuned` meta does not record **whether the rotation was actually applied** — even when flown isotropic, it stays "tuned" (2026-08-25)
- **Found**: 2026-08-25 (workflow: mpc_tuned formulation extraction + adversarial verification of 12 claims)
- **Symptom**: `HwDobMpc.meta()` (mpc_bridge.py:641-643) looks at `self.tuned` (whether the mode name ends in `_tuned`)
  **alone** and stamps `cost_frame: "path (along/cross split)"` + a `path_cost` block
  (q_along 75 / q_cross 1200). But the condition under which the weight rotation actually reaches the solver
  is narrower than that — `_apply_stage_weights` (mpc_bridge.py:554-558) rotates **only when a path plan is installed**,
  and the plan's only producer (workers.py:1634-1641) is reached only when `cfg.path_following` is true and
  a `PathCursor` exists (workers.py:1613, 2468). That is, if, with `path_following: false`,
  `mpc_tuned` is flown, the meta says tuned **even though the entire mission was flown at isotropic 300/300**.
  This is exactly the failure mode that path_cost.py:70-72 guards against ("the run flies baseline but the meta says tuned"),
  and the `_w_tuned` flag, which knows whether the rotation was actually applied, is not recorded anywhere.
- Runtime check (heavy_gripper, prebuilt `heavy_gripper_rti`, read back with `solver.cost_get`):
  built `[[300,0],[0,300]]` / tuned tick with a plan `[[75,0],[0,1200]]` / station tick (plan=None)
  `[[300,0],[0,300]]`. **Artifact not preserved** — to reproduce: `cost_get(0,"W")[:2,:2]` in the three states above.
- **Incidental defect**: the third no-op documented at mpc_bridge.py:550 ("if the plan does not carry `psi_path`,
  don't guess, fall back") is **dead code**. `NedPlan.__post_init__` (path_geometry.py:336-342)
  always fills `psi_path`, and for a legacy 4-argument construction it puts in `yaw_ned` instead — rather than reverting to isotropic,
  it **silently rotates to the vehicle heading** (with `heading_follow: false`, an angle unrelated to the path tangent).
- **Workaround**: before citing a tuned run, check `path_following: true` in that run's `controller.json`.
  The current config is true (hw_mpc.yaml:105), so it is on the normal path, and there are still 0 tuned runs at all.
- **Proper fix**: have `meta()` also record `_w_tuned` (or the number of ticks on which the rotation was actually used), and
  refuse, or at least warn, when a `_tuned` mode is armed together with `path_following: false`.
  Change the `psi_path` fallback in `NedPlan` to an exception instead of a silent yaw substitution. Delete this entry once fixed.

### 📊 compare sweep awaiting re-run — after the MPC surge box change 8 → 30 N (records boundary)
- **Found/changed**: 2026-07-24 (benchmark unfairness found in the wave-crossover workflow → fixed the same day)
- **Symptom**: since `params.U_MAX[0]` was changed from 8 → 30 N (= PID `f_max`), **all mpc/dobmpc results in the existing
  recordings were measured under the stale 8 N box**. In particular, what was observed in `compare_20260724_160210` and others,
  "PID is better than MPC in strong waves", is **a benchmark artifact; do not cite it** (ablation:
  storm PID−MPC −4.24 → +8.60 cm, a sign reversal; the crossover vanishes).  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
- **Workaround**: new run meta records `controller.u_max`, so new/old records can be told apart
  (if the key is missing, it is the stale 8 N). Do not pool results across the boundary.
- **Proper fix**: after re-running the full sweep (`experiments/run_compare.py`, 6 sea state × 5 mode × 3 ctrl),
  delete this entry. The re-run is up to the user.

### w_hat ±50 clip — a single scalar serves both N and N·m + silent firing + duplicated constant
- **Found**: 2026-07-22 (mpc_acados review workflow: control-theory ×2 + simulation-advisor,
  verifier numeric verification 11/12 verified)
- **Symptom**: regarding `np.clip(w_hat, ±50)` (mpc_acados.py:170, mpc.py:198) —
  (1) the force-channel 50 N is above this environment's physical upper bound (realistic X≈35 / Y≈37 / Z≈3 N), so it effectively
  does not bite, but at the conservative sway stack (thrust-limit speed into a wave-crest backflow transient) ≈85–100 N it does bite,  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
  and this region is exactly the acados blowup regime; (2) the torque-channel 50 N·m is 5–6× the authority (8–10 N·m)
  and 5× the physical upper bound (Munk mismatch ≤10 N·m) — the rotational effective inertia is 0.45–0.76 kg·m², so
  if estimate spikes in the 40 N·m range pass through, 66–111 rad/s² is predicted → slack/RTI blowup;
  (3) no firing counter/log + the logged wh0-2 are pre-clip · FLU-world, so whether the body-frame clip
  fired cannot be told (the ‖wh‖ 50–86.6 N band is ambiguous, only >86.6 N is confirmed); (4) the constant is
  hardcoded in duplicate in two files.
- **Workaround**: none (in normal missions the force clip almost never bites; the observer state is not clipped, so
  the estimate itself is not contaminated).
- **2026-07-23 update**: EAOB innovation gating was **implemented, but after measurement it was decided to be OFF by default**
  (`params.EAOB_GATE_ON=False`; the mechanism · `n_gated` counter · meta records are kept) — in a square
  seed-0 A/B (at tau_dist=0.5), the χ²(0.999,18) gate — of the frames that constantly violate w_dot=0 in waves/corners —
  rejected 25–76%, blocking wave tracking itself (CDW radRMS 4.5→15.5 cm, CD 1.35→1.43 cm);  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
  the vicious cycle of rejection in turn growing the innovation contaminated the NIS statistics too (137 vs gate-off 25). At the final default  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
  tau_dist=0.2, the gate does not fire at all in the same scenario (0/1067). The consistency gate is  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
  structurally incompatible with the wave-band model violation → spike defence still has to be handled by a single per-axis
  `W_HAT_CLIP=[15,45,45,5,5,8]` definition + clip-firing counter logging (the duplicated hardcoding at mpc_acados.py:170 ·
  mpc.py, ±50, stays as is). The root fix is harmonic-EAOB.

### `tests/test_dobmpc.py` — premised on rank-5 (NU=4/option-b), cannot run because bluerov2 was removed (deferred)
- **Found**: 2026-07-06, **updated 2026-07-21** (bluerov2 variant removed)
- **Symptom**: the trim/option-(b) assertions in `tests/test_dobmpc.py` (6 N ≈ 23° pitch, `[Fx,0,0,0]` NU=4
  input) are based on the removed rank-5 `bluerov2` plant. The previously documented `ROV_MODEL=bluerov2` bypass
  now gives ValueError (no such variant). With the default heavy it is NU=6, so the option-(b) assertions do not hold.
- **Workaround**: none — the dobmpc cleanup has been postponed, so this test is deferred along with it.
- **To fix properly**: do it **together with the postponed dobmpc NU=6-only cleanup** — remove the NU=4/
  option-b dead path in params.py + rewrite test_dobmpc's rank-5 trim assertions for heavy (NU=6).

### `verify/verify_hydro.py` — a fixture mixing the bluerov2 rigid body + heavy coefficients, 39 FAIL (2026-08-13)
- **Found**: 2026-08-13, during the self-verification of the `for_jaden/` handover package.
- **Symptom**: `python verify/verify_hydro.py --no-plot` → **39 FAIL / 13 PASS**
  (widespread: added mass T5, restoring pendulum T4, transfer-function TL axes, T7-R2 buoyancy, etc.).
  It is not a defect in hydro.py but a **fixture mismatch**:
  - verify_hydro.py:32 loads `bluerov.xml` (the legacy rank-5 BlueROV2), and
    the ground-truth constants at verify_hydro.py:35-45 are BlueROV2 values too (MASS 11.2, VOLUME 0.0113459).
  - But `H.Hydrodynamics(model, disturbance=None)` at verify_hydro.py:61 reads the coefficient-YAML
    default = `RM.YAML_PATH`, and the `ROV_MODEL` default is heavy, so **BlueROVHeavy.yaml**
    (VOLUME 0.0116499) is what gets picked up.
  - Confirmed exactly via buoyancy: expected 997·9.81·0.0113459 = **110.969 N** (test constant) vs
    measured 997·9.81·0.0116499 = **113.943 N** (the coefficients actually loaded). Agrees to the third decimal place.
  - On 2026-07-21 the `bluerov2` variant was removed from the registry, so the `ROV_MODEL=bluerov2` bypass is not possible either
    (ValueError) — a leftover of the same family as the `tests/test_dobmpc.py` entry above.
- **Workaround**: none. Do not cite hydro numbers as verified by this script.
  `tests/test_hydro.py` (neutral buoyancy / self-righting / drag terminal velocity) passes normally, so it is currently
  the only valid hydro smoke test.
- **Proper fix**: have `make_sim()` **specify** the coefficients via `H.Hydrodynamics(model, disturbance=None,
  coeff_path=<marinegym_assets/BlueROV.yaml>)`, pairing them with the bluerov.xml
  fixture (the ground-truth constants stay valid as they are). Or move the whole fixture to heavy
  and rewrite the top constants with the BlueROVHeavy.yaml values. `verify_hydro_precise.py` also
  builds its sim through the same default-coefficient path at verify_hydro_precise.py:438, so it is likely to have
  the same mismatch — **unconfirmed**; check it at the same time.

### dobmpc NU=4 / option-(b) dead path (unreachable after the bluerov2 removal, deferred)
- **Found**: 2026-07-21 (bluerov2 variant removed)
- **Symptom**: all remaining variants are rank-6 (NU=6), so the NU=4 branch in `dobmpc/params.py` and
  the option-(b) surge→pitch coupling path in `mpc.py`/`mpc_acados.py` have become dead code.
- **Workaround**: harmless (never reached) — no functional impact.
- **To fix properly**: per the user's instruction, simplify everything in one batch **later** (NU=6 fixed). Together with the test_dobmpc
  rewrite.
- **2026-07-22 update**: **mpc.py cleanup done** — tau if/else removed (`tau = u`), the pitch bound's
  PITCH_AWARE ternary → fixed at 1.2 (behaviour identical, since PITCH_AWARE=False), stale comments/docstring corrected
  (verified: dynamics Δ=0 vs git HEAD over 2000 random samples + IPOPT NMPC smoke).
  **Remaining**: the NU=4 branch · PITCH_AWARE/THETA_MAX/SURGE_PITCH_COUPLING in params.py,
  the option-(b) pitch-bound ternary at mpc_acados.py:109 (+docstring "u=[X,Y,Z,N] (4)"),
  the rank-5 branch at dobmpc_controller.py:269-280, the test_dobmpc rewrite.

### `tools/gen_pool_apriltags.py --selftest` — overwrites `tag_floor.xml`
- **Found**: 2026-07-06 (still valid after the `--tag-mode plane` rework — `run_selftest`
  writes tag_floor.xml as a version with 2 test tiles at tools/gen_pool_apriltags.py:475)
- **Symptom**: after selftest, the POOL_TAGS scene loads with a 2-tile floor.
- **Workaround**: after selftest, re-run the `python tools/gen_pool_apriltags.py` full build
  (README §7 has a warning).
- **To fix properly**: have selftest write to a separate temporary file and delete it on exit
  (like the existing `_selftest_scene.xml`).

### `tools/plot_wave_spreading.py` — a hardcoded copy of config/base.yaml
- **Found**: 2026-07-06
- **Symptom**: Hs/Tp/gamma/s/h/N_omega/N_beta are copied in as script constants (the yaml is not
  read) → if the config is changed, the slide figure **silently** diverges from the actual experiment.
- **To fix properly**: read the yaml directly with `disturbance.config.load_config`.

### `water_viz.py` hfield axes are **row↔column transposed** — the rendered wave surface is mirrored about the x=y diagonal (2026-08-19)
- **Symptom**: `water_viz.py:79-84` is commented and implemented as "row = X, col = Y", but when, in MuJoCo
  (3.9.0), a 4×4 synthetic hfield is built and probed directly with `mj_ray`, raising the last **row**
  makes +Y rise and raising the last **column** makes +X rise — the actual convention is row→Y, col→X. As a result,
  the on-screen propagation direction of the wave surface · current is **mirrored about the x=y diagonal** relative to the physical direction. The 48-sample/
  0.1771 m axis is physically Y, the 96-sample/0.0885 m axis is X.
- **Impact**: **cosmetic only**. hx = hy = 4.25, so the geometry still matches, and the hfield is
  `contype=0 conaffinity=0`, so there is no dynamics path (`tests/test_water_viz.py` Δ=0 unaffected).
  However, checking "the waves come from this side" in the video against the physics gives a mismatch.
- **Workaround**: do not claim the wave **direction** from the preview video (intensity · texture only).
- **Proper fix**: in `water_viz.py:79-84`, hook X to ncol and Y to nrow,
  match the `meshgrid(..., indexing="ij")` axis order, then add a synthetic hfield + `mj_ray` regression test.

### The `water_viz.py` preview looks **calmer** than the actual sea — 120-component trim + eta clipping (2026-08-19)
- **2 symptoms** (both render-only, not recorded anywhere):
  (a) `MAX_MODERN_COMPONENTS = 120` (`water_viz.py:40,138-150`) draws, out of M = 1260–1848 components,
  only the top 120 by amplitude → elevation variance retention **very_rough 74.7% (eta std 0.375 →
  0.324 m) / moderate 91.1% / gentle 96.8%** [유도: recomputed this session, not saved — 산출물 없음].
  (b) `d = 0.5 + eta/elev` is clipped to [0,1] (`water_viz.py:105-107`), and with the shipped
  `elev = 0.60` (`bluerov2_mujoco_marinegym/tag_floor.xml:10`) it is **flat-topped at |eta| > 0.30 m**.
  very_rough's post-trim eta std is ~0.32 m, so a substantial part of the preview gets clipped. Yet
  `tools/gen_pool_apriltags.py:231,569-570` calls this value "half-range / max|eta| headroom"
  (the actual headroom is elev/2).
- **Workaround**: for strong sea-state previews, regenerate with `--water-hf-elev` at 2× (e.g., 1.2 or more).
- **Proper fix**: (a) log the post-trim retained variance once in `update()`; (b) so that `elev` is used as
  the actual half-range, change it to `d = 0.5 + eta/(2*elev)`, or rename the argument.

### `water_viz.py` **ignores the layer gating** of the modern env — ripples show in C/CD modes (2026-08-19)
- **Symptom**: `_eta_modern`/`_current_vec` read `field.waves` directly and
  **do not check** `field.use_waves` (`water_viz.py:113-136,183-198`). The caller passes only the master
  `enabled`. So even in the modes with waves off (C, CD), the water surface undulates.
  `experiments/wave_preview.py:200` exposes `--mode C/CD` on the CLI, and `:151`
  hardcodes `enabled=True`. (legacy `disturbances.py` gates `enabled` properly, so
  the problem does not exist on the teleop path.)
- **Impact**: visual only, but it **invites misreading** (mistaking a C-mode video for a wave video).
- **Proper fix**: check `getattr(field, "use_waves", True)` in `_eta`, and
  gate advection with `use_current` as well.

### `tools/analyze_square3.py` / `tools/analyze_acados_vs_before.py` — hardcoded paths
- **Found**: 2026-07-06
- **Symptom**: specific CSV file names under `recordings/20260615/` are hardcoded (`DIR`/`RUNS`/`PAIRS`
  constants) → crash if that recording is absent; no CLI flags.
- **Workaround**: to use them on another run, edit the top constants (stated in README §5).
- **To fix properly**: make it a `--dir` argument. Low priority (one-off analysis scripts).

### `experiments/plot_trajectories.py` — docstring/code mismatch
- **Found**: 2026-07-06
- **Symptom**: the docstring mentions an error-vs-time panel, but the current code does not have it.
  The laps[-2:] RMS (`rms_lastlaps`) is only computed internally and is not displayed anywhere.
  (On 2026-07-28 the *displayed* RMS was unified onto the same steady window as run_compare —
  this entry is limited to the remaining docstring cleanup + the undisplayed `rms_lastlaps` item.)
- **To fix properly**: clean up the docstring, or add the laps[-2:] RMS to a panel/legend.

### `verify/verify_acados.py` gate is marginally exceeded on heavy_gripper (0.2717 > 0.25 N)
- **Found**: 2026-07-12 (while verifying the heavy_gripper variant)
- **Symptom**: the acados RTI vs IPOPT worst-case |Δu| gate of 0.25 N is calibrated on heavy;
  on heavy_gripper (13.7 kg) it is 0.2717 N — ~0.9% of the 30 N sway authority, so effectively the same optimum,  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
  and the closed loop is verified too (DP hold 1.3 cm). heavy is still PASS.
- **To fix properly**: scale the gate per variant, or make it relative to ‖u‖.

### In wave modes, acados solver failures (n_fail) infrequently trigger attitude/depth blowup
- **Found**: 2026-07-21 (`compare_20260720_221845` analysis workflow; scale confirmed with n=200
  `compare_20260720_230025`)
- **Symptom**: n=200 census — 289 failures in total in 217 of 3000 runs, all in the MPC family, and
  89–99% in CW/CDW (run failure rate **mpc 14.0% vs dobmpc 7.7%** — the EAOB FF, if anything,
  stabilizes the OCP). In failed runs, coupled extreme depth/pitch excursions — mpc up to 181.8 cm (pitch up to
  79.5°), dobmpc up to 134.7 cm (|pz| 97 cm). **dobmpc radial_max>40 cm: 23/23 are fail runs**
  (clean-run upper bound 37.8 cm) → this class dominates the worst-case statistics. The trigger is the seed-0 shared
  wave group's lap-7/8 V3 turn event, and the failure is an amplifier (not the cause; only a run-level association can be confirmed).
- **Workaround**: in analysis, treat the radial_max of `n_fail>0` runs separately (the RMS tally is robust:
  excluding them shifts the mean by only −3~−8%).
- **To fix properly**: (1) log the failure **time** (currently there is only a per-run count, so tick-level causality
  cannot be confirmed), (2) review the fallback strategy on failure (the mpc_acados failure path), (3) add
  w_hat · solver-status records to the traj CSV.
- **2026-07-21 update**: after reference preview was introduced, the failure rate dropped sharply (on the shared 50 headings, dobmpc
  10–13→1–2 runs, mpc 16→7/12→4) and the dobmpc >40 cm tail vanished — but the issue itself remains
  (a new 210 cm blowup in mpc CDW; the three fixes above are still valid).

### hydro.py implicitly assumes body_iquat=identity (diagonal inertia)
- **Found**: 2026-07-12 (found while tracing the root cause of the heavy_gripper NMPC divergence)
- **Symptom**: `mj_objectVelocity(mjOBJ_BODY, local=1)` is referenced to the **inertial (principal-axis) frame**, but
  hydro treats it as the body frame and applies drag with `xmat`. When `fullinertia` sorts · permutes the principal
  axes (Iyy>Izz>Ixx etc.), the drag axes get scrambled → **energy injection → explosion** (reproduced: a torque-free
  kick of 0.5 rad/s → |q|>60 rad/s within 1.5 s).  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
- **Workaround**: the generated heavy_gripper XML forces diaginertia + `tests/test_heavy_gripper.py`
  guards `body_iquat==identity` against regression. Existing variants are diagonal to begin with, so no symptoms.
- **To fix properly**: have hydro measure with `mjOBJ_XBODY` (body frame), or convert with ximat —
  it is a physics-file change, so it needs separate verification (confirm the existing variants are byte-identical).
- **2026-07-19 update**: moving the C3 to its measured position (front-bottom) grew the discarded Ixz from
  −0.0016 (0.4%) → **heavy_gripper +0.064 kg·m² (16.8% of Ixx) / heavy_c3 +0.046 (12.4%)**.
  This means the roll-yaw product of inertia that would exist on the real vehicle is absent from the plant — until hydro is fixed to
  the body frame, it structurally cannot be put in. The priority of the "To fix properly" above goes up.

### EAOB perf consistency breaks in CDW after the 2026-07-23 base.yaml wave strengthening (regardless of source)
- **Found**: 2026-07-23 (during the verify_state_source A/B verification; at 19:28 the wave block was strengthened to
  Hs 0.75→1.2 m, Tp 12→6 s, γ 5→2, s 30→10, ω_max 1.6→3.0)
- **Symptom**: in CDW of the new sea state, regardless of mpc_state_source, NIS 80 (truth)/71 (estimate)
  (DP), 44/42 (square) — far outside the target range (14–24) verified with τ_dist=0.2, and radRMS also  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
  worsened, DP 1.5→9.6 cm / square 3.5→15.7 cm. It is w_dot=0 + τ_dist=0.2 failing to keep up with the new wave band
  (ω_p≈1.05, energy ~3 rad/s); with the old wave config, PASS on every item is reproduced.
- **Workaround**: none (not a divergence — n_fail 0, finite). `verify/verify_eaob.py` ·
  `verify_state_source.py` signal it with exit 1 under the current config.
- **2026-07-24(3) update**: in the surge box 8→30 N change A/B (same command `verify_eaob --no-plot`,
  CDW/T=60/seed 0), **NIS 245.8→100.0, NEES 398.9→122.0, a 2.5–3.3× improvement** (still  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
  over the gate 24 = FAIL). That is, part of this consistency collapse was **large tracking error caused by insufficient authority**
  (the box contaminated even the EAOB's pseudo-measurement channel), and the rest is, as originally diagnosed, τ_dist
  failing to keep up with the new wave band. Re-sweep on the 30 N box (do not tune on the old 8 N numbers).
- **To fix properly**: re-sweep EAOB_TAU_DIST for the new sea state (0.05–0.1 expected) or
  harmonic-EAOB — either way it is an experiment-design decision, so the user's judgment is needed.

### STATION BRIDGE (tag dropout ladder) is **unverified on hardware** (2026-08-18)
- **What**: in station mode it does not disengage even when the tags are lost. From 0 to `imu_hold_s` (3 s),
  all axes run on the bridge estimate; after that, x/y/yaw are released and only depth+attitude are held **indefinitely**.
  `rov_gui/control/station_bridge.py`, config `config/hw_mpc.yaml: station_bridge:`.
- **What is verified**: ladder logic · whitelist · coast's axis values on the wire (through allocation/cap/slew)
  · other interlocks surviving · station-only scope · and even the meta/CSV boundary, in `test_station_bridge.py` 15/15,
  no regressions in the existing suite (test_control 79/79, test_offline 79/79, test_imu_dr 25/25).
- **What is not verified**: **It has never been run in water.** In particular, (1) how many seconds an actual dropout lasts
  is not known yet — in the 8 runs of 2026-08-18 the worst was 0.55 s, within the existing 1.1 s budget,
  i.e. **no log in which the problem reproduced has been seen yet**; (2) the acceleration-integration path that `xy_source: auto` picks
  depends entirely on the quality of the C3 BNO086 calibration (`config/c3_imu_calib.json`),
  and that calibration itself is unverified ([[imu-dead-reckoning-experiment]], 40° tilt not entered);
  (3) how far it actually drifts during coast is not known.
- **Workaround**: in the first pool session, **the pilot keeps a hand on the E-STOP**, and once the panel turns red with
  `NO TAG — COASTING`, from then on a human makes the call. There is no geofence, so
  if it drifts into a wall during coast, nothing stops it. To revert, the one line `station_bridge.enabled: false`
  returns it exactly to the old behaviour (pinned by a test).
- **Proper fix**: (a) from the CSV of a run where the problem occurred, first measure the dropout-length distribution with `tag_age_s`/`n_tags`
  and fit `imu_hold_s` to it. (b) Collect the recovery log's
  "IMU was N cm off" a few times, and whether to trust `xy_source: imu` is decided **by measurement**
  — the current 3 s/9 cm is [유도]. (c) What to do automatically when coast gets long
  has not been decided yet (currently: hold forever + human).

### Object following (`follow` + `object_nav`) is **unverified on hardware** and inherits the `--pose` risks wholesale (2026-08-21)
- **What**: with `--pose --mpc` both on, the clicked object comes up in tag-map coordinates (panel diamond,
  `bus.object_fix`), and the mission shape `follow` holds the relative pose at the moment of START (3D position + yaw).
  `rov_gui/control/object_nav.py`, config `config/hw_mpc.yaml: object_nav:`,
  doc: "Following an object — follow" in `rov_gui/README.md`.
- **What is verified**: extrinsic cancellation (the object position is unchanged to 1e-9 even with a wrong extrinsic) ·
  lever-arm cost of a frame mismatch · offset round trip and orbit properties · fixed yaw axis · jump rejection/reseed ·
  t_capture-based velocity · extrapolation clamp · pairing priority · five arm refusals ·
  arm does not move the vehicle · leash holds even at a 1 s tick · excursion clamp · freeze→station demotion ·
  bridge handover after tag-dropout demotion · lifecycle in four places · CSV 10 columns/schema 6 —
  `test_object_nav.py` 28/28, no regressions in the existing suites (test_control 81/81, test_offline 79/79,
  test_station_bridge 15/15, test_imu_dr 27/27), `demo_e2e.py pid follow {still,drift,orbit}`
  all pass (pair_exact 100 %).
- **What is not verified**: **never run even once, neither in water nor in air.** And this feature
  **inherits wholesale** the risks of the entry below, "`rov_gui --pose`: object tracking/pose estimation has never run on hardware (2026-08-09)"
  — if `--pose` cannot deliver `TRACKING` on an object on a desk,
  this feature might as well not exist. All the demo proves is the plumbing and the algebra: the demo's
  `T_cam_obj` is built from the demo vehicle state, so the composition round-trips **structurally**,
  and none of SAM2 mask quality, FoundationPose latency, dropout statistics or depth noise shows up.
  The thresholds in `object_nav:` are **all [예측]**.
- **Cheapest go/no-go (it may be a placement problem, not a code problem)**: with `cam_tilt_deg: 43.3`,
  the C3 **looks down** at the mat. The object has to be in **one frame** with ≥2 tags, at 0.3~0.8 m range,
  and within the HFOV of 63.9° (62 cm field-of-view width at 0.5 m) for the composition to hold.
  **If the tags and the object cannot coexist in one frame, this feature cannot fly on this mount, and the answer is
  not code but a second camera or a re-tilt.** Check it with a tape measure first.
- **Workaround (bench → pool, in order; each stage earns the next)**:
  Bench (in air, COMMAND ENABLE off, not mounted): ① check the placement above → ② click and watch the diamond.
  Push the object 20 cm by hand and it must move in the plot too. **In-air distance reads ~1.33×
  long** (the C3 factory calibration is an **underwater** value — `calib/FOV_AUDIT.md`, the 2026-08-04 entry below).
  **Do not "fix" this in air** — the by-hand translation check is valid **as a ratio**, and what to record
  is that ratio (if a real 20 cm is drawn as 26.6 cm, the whole chain is right and only the scale is a known offset).
  → ③ check whether the panel's `pair` slot is effectively 0 ms. If not, independent conflation in the two mailboxes
  has defeated the shared stamp, the extrinsic cancellation stops, and the 0.2855 m lever arm enters the error budget
  — you need to know this **before the pool**. → ④ confirm that the 4 refusals (mode mpcc / no lock /
  `nav_source: second` / no `--pose`) fire.
  Pool: ① **HOLD only, object in view, 2 min** — scatter · `pair_exact` ratio · `obj_age_s` p50/p95 ·
  how often `PoseTrack.state` leaves `tracking`. This is the run in which every [예측] in `object_nav:` becomes [측정],
  and whether follow is worth arming is decided here. → ② **follow on a stationary object**:
  the vehicle **must not move** (the offset was taken from the current pose). You see it within 1 second. →
  ③ an object moved slowly by hand along a 1 m straight line at ~0.05 m/s; compare the chip's `speed`/`ref_speed`
  (if the FF is not taking effect, it shows up as a lag of 2× or more). **Hand on the E-STOP — the only thing that knows about the walls is the new
  `max_excursion_m`, and that is unverified.** → ④ rotation in place (the orbit case, where a wrong
  `yaw_axis` shows up most clearly) → only after that, fast moves and long excursions.
  Do the first bench and pool sessions with `object_nav.yaw_axis: none` — not using the object yaw removes a whole
  failure class. To revert, set the shape to anything other than `follow`; to use only the display
  half, simply do not arm (there is no new authority over the vehicle at all).
- **2026-08-23 first hardware attempt — caught by the go/no-go above (not code; distance)**: the tags and the object
  did make it into one frame, but **the object was at 1.2~1.56 m**, so `object_nav.max_distance_m:
  1.20` rejected almost all observations. Evidence: `pose: collecting reference views — object is
  1556 mm away` and, at mesh capture, `distance 1195-1256 mm`
  (`data/20260823/0823_162548/mission_log.txt`),
  in the three 16:34~16:36 runs `obj_state` is **`cold` on every row** (never locked once), and the 16:38 run,
  after locking just once, is **`lost` on every row**, with `obj_age_s` growing 9.6 s→202 s
  (`0823_163414/mpc_163512.csv`, `0823_163707/mpc_163809.csv`). So the diamond hardly ever appears
  in the plot, and `follow` cannot even be armed ("no object lock").
  **1.2~1.5 m is an [unverified] range** — 0.3~0.8 m works [측정: KNOWN_ISSUES 2026-08-09],
  2.4 m fails [측정], and nobody has ever measured in between. Bringing the object to 0.3~0.8 m comes
  **first**; only when that is impossible given the placement, raise `max_distance_m`, but mark that run
  as flown in a new [예측] range.
- **Proper fix**: once pool ①~④ pass, delete this entry and, in `object_nav:`, turn the
  `[예측]` into measured values. If ② fails, the cause is usually the placement or `--pose` itself.

### Object pose **distance is 1.42~1.50× long** — cause not settled (mesh scale vs depth calibration) (2026-08-23)
- **Symptom**: an object lying on the mat shows up **0.42~0.50 m below** the tag plane, and the hull→object ray is
  **1.42~1.50×** the distance to the tag plane (p10~p90 1.37~1.60). The direction is right; only the length stretches.
  [측정: `data/20260823/0823_174602/mpc_174638.csv`,
  `mpc_174657.csv`, `obj_state=live` rows; tag-plane position from each run's meta
  `hardware.datum_tag_frame.p0`]
- **It is not refraction (the sign is opposite)**: at a flat port, water magnifies the image 1.33×, which makes distances read
  **short**. An in-air calibration used in water reads 1.33× short, and an underwater calibration used in air
  reads 1.33× long ([[c3-calibration-is-underwater]], `calib/FOV_AUDIT.md`). No combination
  reads long in water. Tag PnP independently disproves it too: at engage, tag z −1.002 m +
  pressure depth 0.41 m = water column above the mat **1.41 m**, matching the actual record of ~1.4 m
  (at 1.45× it would have to be 1.86 m).
- **The two remaining candidates cannot be told apart with this data** — both produce "the mesh is big and the distance is far":
  1. **Reconstructed mesh scale**. That day's mesh was 8 views / arc 101° (the log warns `only 8 views (10+
     recommended); the mesh may be poor`) → `88 x 175 x 239 mm`. FoundationPose
     pushes the distance out until the rendered mesh matches the observed size, so **distance ∝ mesh size**.
     For reference, the mesh of the same family of object from the 16:32 session is `101 x 114 x 171 mm` (239/171 = 1.40).
  2. **The depth map is not metric**. Stereo depth comes from a mono pair + baseline, so it is a
     **separate calibration path** from the RGB fx, and **the tags do not use depth at all**. If this is 1.4×, the BundleSDF
     reconstruction and the FoundationPose tracking both inherit the same scale error, and only the tags stay fine.
- **A discriminator has been added**: the **`depth-vs-TAG n.nnx (N tags)`** line in the trajectory readout
  (`window._check_depth_scale`). For each mapped tag in view it compares the depth-map value with the **geometric
  camera→tag distance from the fix**, and takes the median. At `1.00x` depth is metric and the culprit is
  the mesh; at `~1.3x` it is the depth path. `test_the_depth_map_is_cross_checked_against_the_tag_pnp`.
- **The 2026-08-23 18:26 run disproved the hypothesis of a "constant scale error".** 12178 live rows
  [측정: `0823_182628/c3_depth_20260823_182645_mpc.csv`]:

  | | median | p10 | p90 |
  |---|---|---|---|
  | object z (MAP NED, 0 = mat) | **+0.53 m** | −0.84 | +0.77 |
  | ray's mat-crossing scale factor | **1.52x** | 0.51 | 1.75 |
  | reported hull→object distance | 2.41 m | | |

  The factor ranges 0.5~1.75, **swinging by more than 3×** — no calibration constant can produce that. The object z also
  swings between 84 cm above the mat and 77 cm below it. **That is, the pose estimation itself is not finding a valid solution**:
  in the same run the `reg` count climbs 67→86 and the log is plastered with `pose does not fit — wrong object` and
  `lost the object`. A mesh of a glossy underwater object built from 8 views is candidate cause #1.
- **2026-08-23 21:1x final confirmation: stereo depth is 1.56~1.71× long in water — the pilot's
  calibration hypothesis was right.** Conclusive evidence: the mesh of a clean capture (arc 79°, 0.88~0.91 m, inside the gate),
  `143×166×187 mm`, vs **caliper-measured 119.73 mm** = **1.562×**
  [측정: `0823_210304/mission_log.txt` 21:02:29 + pilot's calipers]. The close-capture meshes are also
  all **consistent** at 1.43~1.63× (171/195 mm), and the floor sweep `depth-vs-MAP` reads **1.71x**
  (at longer range — the error may be range-dependent). Colour PnP is mm-level in the same water, so
  **the colour calibration is fine, and only the stereo depth path is non-metric**.
  The "swallows what is not the object" analysis below is **downgraded to a partial cause**: the growth of scatter with distance (smear) is
  a measured fact, but the dominant term is this scale. 470 mm mesh = scale × smear.
- **2026-10-05 correction — the 'conclusive evidence' above is an in-air capture.** The reference view of the 187 mm mesh,
  `data/20260823/0823_210135_obj/rgb/000000.png`, is a shot of a hand-held PVC elbow taken on the lab floor (checked directly).
  The C3 factory calibration is an underwater value, so in-air depth inherently reads about 1.33× long (the 2026-08-04 entry below).
  So this 1.562× is not cited as evidence for '1.56× in water'. Two pieces of in-water evidence remain.
  - Object reprojection overscale of 1.55/1.57× in the two 0824 runs (`data/20260824/{0824_101807,0824_101251}`, `.claude/journal/consults.md` 2026-08-24).
  - Floor sweep depth-vs-MAP 1.71×. Read off the screen; not recorded to a file.
- **Mitigation (added 2026-08-23)**: `--depth-scale K` — multiplies the depth millimeters **once, at the source**, so
  every consumer (cursor probe · capture · FoundationPose · depth-vs-MAP) sees the same corrected stream.
  Starting value **0.64** (=1/1.562). Once applied, depth-vs-MAP should come down to ~1.0x (that line becomes the verifier of the
  correction), and **the mesh must be re-captured** (old meshes are 1.56× and do not match the corrected depth).
  It is recorded in the pose CAMERA record as `depth_scale_applied`. **The real repair is an underwater stereo
  recalibration** (work on the c3_camera side, not started).
- **Boundary**: every depth-derived distance before this date is 1.4~1.7× long — object position · hold_m · mesh
  dimensions · reference-view distance: do not cite any of them.
- **(Below: record of past analysis)** What was the candidate cause as of 2026-08-23 20:00: the reconstruction swallows what is not
  the object (it is not a scale error).**
  The same object reconstructed eight times [측정: `data/*/*_obj/model/model.obj`, oriented bbox]:

  | verts | bbox (mm) |
  |---|---|
  | 18,548 | 101 × 114 × **171** |
  | 30,147 | 87 × 112 × **195** |
  | 59,542 | 88 × 175 × **239** |
  | 101,428 | 108 × 172 × **470** |

  **The two short sides are stable** (87~108 × 104~175 mm) and **only the long side** grows. The reconstruction
  **is based on metric RGB-D, so it knows the size** — what is wrong is not the scale.

  **The culprit is not the mask but the depth inside the mask, and the deciding variable is distance.**
  Measured the in-mask depth p5~p95 in every saved reference capture
  [측정: `data/*/*_obj/{depth_enhanced,mask}`]:

  | capture distance | in-mask depth scatter | resulting mesh length |
  |---|---|---|
  | 510 mm | **102 mm** | 182 mm |
  | 626 mm | 142 mm | 239 mm |
  | 1210 mm | 284 mm | 171 mm |
  | 1369 mm | 265 mm | **470 mm** |
  | 1358 mm | **714 mm** | (discarded) |

  The object is about 110 mm. **At 0.5 m the scatter ≈ the object size** (normal), but **at 1.4 m it is 2~6×**, so
  the point cloud being fused is a smeared blob, and the mesh lengthens in the direction the smear points.
  The 470 mm mesh in question was **captured at 1083~1479 mm** (mask normal, 18 frames/79°).
  The station already prompts "30-80 cm" during capture, but the upstream view gate accepts up to 1.5 m
  — the gap in between is where this accident happens.
- **So a uniform rescale and mask tuning are both the wrong remedy.** On 2026-08-23 a rescale was
  built once by mistake and then reverted (it ends up shrinking the two correct sides). The mask was actually fine.
- **Mitigation (added 2026-08-23)**: the moment capture ends, `PoseSession.depth_quality`
  prints **in-mask depth scatter vs the object size implied by the mask area**. Above 1.5× it
  warns: "The mask is fine and the depth inside it is smear — recapture at 30-80 cm."
- **Heading dependence**: an object on tag 56 shows up at 57 (+0.22 m y) with the vehicle facing +y, and near 55
  (+0.29 m x) with it facing +x — the error follows **the direction the vehicle is looking**. It is not an extrinsic bug
  (`TagNav._solution` divides by `t_cb` and `compose_map_pose` multiplies it back, so it cancels exactly).
  A constant error in the camera frame (= a distance error along the optical axis) rides on `R_map_cam` and rotates,
  and it is one and the same defect as "the object goes below the mat".
- **Mitigation (added 2026-08-23)**: `--pose-object-size MM`. Measure the object's longest side and pass it, and **right after the build**
  it compares that with the mesh dimensions and warns (not a rescale). If more than 25% longer: "do not fly with this mesh".
  The downstream symptoms below are all consequences of this, so do not cite runs flown with a mesh that raises this warning.
- **Do not cite the earlier `depth-vs-TAG 1.44x` — that check had a sampling bias.**
  It read a 5x5 at the tag center, but the tag center is a black square, i.e. **the one patch in this scene that stereo
  cannot match**. So the smaller the tags got (= the higher the vehicle), the more holes and edge bleed there were, and
  the number moved with the tag count: 3 tags 0.94x / 8 tags 1.17x / 14 tags 1.46x
  [측정: 4 pilot screenshots, 18:30~18:35]. **On 2026-08-23, replaced with a method that sweeps the whole mat**
  (`window._check_depth_scale`: pixel grid → intersection with the tag plane z=0 → depth Z relative to the expected Z,
  median of hundreds of samples + p10~p90 spread). The new line is `depth-vs-MAP n.nnx (+/-s.ss)`, and
  **if the spread exceeds 0.15, that number is not a constant** (= meaning it is a shape problem rather than a scale problem, or
  that something other than the mat is in view). **No run has been read with the new method yet.**
- **A check you can do right away (no code)**: put the cursor probe of the C3 DEPTH panel **on the object**, read
  the mm, and compare it with the camera→object distance the pipeline reports at the same moment (SENSORS `Object` row).
  If the probe is short, it is the mesh; if it is long too, it is depth.

### `HwMpcc.set_target_ned` **discards** the velocity feedforward — why `follow` is refused on mpcc (2026-08-21)
- **Symptom**: `set_target_ned` in `mpcc_bridge.py` starts with `del v_ned, r_ned`.
  On every call it builds a new `ArcPath`+`speed_profile`+`window_for` and clears the scenario with `scenario = None`.
  Given a setpoint moving at 20 Hz, it **rebuilds the path every tick with no FF**.
- **Why it matters**: without FF the leash becomes the de facto control law — measured
  `kp·lead = 51.7 × 0.35 = 18.1 N` vs `F = 86.7·v + 5.76` → `v ≈ 0.084 m/s`
  (2026-08-18, memory: approach-speed-leash-limited). Harmless in DP hold, but
  unusable in any mode that chases a moving target.
- **Workaround**: `HwMpcc.follow_ok = False` — the `follow` mission is **refused** on
  mpcc/dobmpcc (fail-closed via `getattr(..., False)`). Fly with dobmpc/mpc/pid.
- **Proper fix**: contouring on a moving path. In a structure where theta is solver state,
  theta loses its meaning if the path itself changes every tick, so this is a design change, not
  a flag. Until then, refusing is correct.

### `r_ned` of `HwDobMpc.set_target_ned` is **effectively a no-op** in the default controller (2026-08-21)
- **Symptom**: `set_target_ned(p, yaw, v_ned, r_ned)` passes `r_ref`, but
  because it calls `set_target(..., yaw_target=None)`, `yaw_target = yaw_ref` applies,
  and in `_xref_ned` `delta = psi_t - psi0 = 0` → `xref[11,:]` is **forced to 0**.
  So the yaw-rate feedforward is alive only in `HwPid`.
- **Why it matters**: the two controllers take the same argument and **ignore it differently**. An A/B
  in this state is no longer a comparison of controllers.
- **Workaround**: `follow` **does not pass** `r_ned` (`_issue_follow_target`). The heading
  rate limit is applied identically to both controllers on the worker side.
- **Proper fix**: when `r_ned != 0`, `set_target_ned` should also compute and pass `yaw_target`
  (e.g. `yaw + r*preview`), or else drop the argument and use only path planning.
  Taking it and silently discarding it, as now, is the worst option.

### `mpc_tuned` / `dobmpc_tuned` (along/cross cost split) is **unverified on hardware** (2026-08-18)
- **What**: a mode that rotates the tracking NMPC's 2×2 position weights into the path frame at every stage
  and penalizes longitudinal (along) and lateral (cross) error separately. `rov_gui/control/path_cost.py`,
  tuning in `config/hw_mpc.yaml: mpc_tuned:`.
- **What is verified**: the algebra (rotation = error split; if isotropic, identical to baseline to machine precision) and,
  further, offline closed-loop parity and no contamination across mode switches — `rov_gui/tests/test_path_cost.py`
  15/15. The corner-cut reduction is confirmed **only on the offline prediction-model plant**
  (4.8 → 1.2 mm at fillet 0.15 · 0.10 m/s, `rov_gui/tools/sweep_path_cost.py`).
- **What is not verified**: never flown in water, not once. That plant has neither the ESC deadband nor the tether,
  and its horizontal drag is 8~12× too low — none of the dominant hardware constraints confirmed in the 2026-08-18 diagnosis
  (actual speed 0.000 m/s at ax_surge 0.158) is captured in it. That is,
  **"corner cut −74 %" is a geometric prediction, not a claim about hardware performance.**
- **Workaround**: the deployed defaults `along_scale 0.25 / cross_scale 4.0` are a **[예측]
  first estimate**. The default mode is still `dobmpc`; tuned is opt-in.
  In waypoint missions with fillet 0 the effect drops to −15 %, and lowering along_scale
  actually makes it worse (+57 %), so **use it together with fillet 0.15**.
- **Proper fix**: after P5/P6, fly one baseline `mpc` ↔ `mpc_tuned` pair in the same
  session and compare the cross-track over the corner segments. The run meta's
  `controller.cost_frame` / `controller.path_cost` is the boundary between the two records —
  the `Q` row of a tuned run is the isotropic baseline the split was derived from, not the value the solver used.

### `rov_gui --pose`: object tracking/pose estimation has **never run on hardware** (2026-08-09)
SAM2 tracking (stage 1), mesh-based 6-DoF (stage 2) and on-site reconstruction (stage 3) were all implemented,
and verified **only with saved data and the demo**. The C3 was not on the network, and the ROV was disconnected too.

- **Done**: SAM2 loading 1.1 s / 36 Hz (saved frames), FoundationPose registration succeeded
  (`ref_views` replay, distance 0.431 m), the whole on-site reconstruction pipeline works in the `rovgui-pose` interpreter
  (saved reference-view replay → collection 8 frames/75.3°/RMSE 6.2 mm → BundleSDF 25 s → 7993 verts
  51×94×167 mm), overlay · click coordinate transform · logging, 61 tests.
- **Not tried**: real C3 live frames, SAM2 mask quality on underwater scenes, whether registration works with underwater stereo
  depth, the effect of the 30 fps tap on C3 worker latency, and **live collection**
  (the 25 s above is a saved-view replay and does not include the time a person actually spends circling the object).
- **Order of checks** (each later step is more expensive):
  1. `./c3 gui --source hw --pose --pose-no-build` — in air, an object on a desk. Turn TRACK on and
     click → do you get `TRACKING` + a mask? If it fails here, there is no need to look at the rest.
  2. Same conditions + `--pose-mesh .../ref_views/model/model.obj`; put **the physical object** of that mesh
     at 0.5 m and click → do the axes appear after 0.7 s with `d ≈ 0.67 m`? (it is an underwater calibration, so on land
     reading ~1.33× long is **normal** — if it comes out as 0.5 m, investigate that instead).
  3. Without `--pose-no-build` (= on-site reconstruction). Click → slowly turn the object at 0.3~0.8 m:
     does `COLLECTING VIEWS` climb → `RECONSTRUCTING` → pose? Do the sanity-check lines (`bad_mask#`,
     mesh bbox) appear in the log, and do the dimensions match the physical object?
  4. Turn on C3 Depth recording and look at `hz_measured` in `_pose.jsonl` — the number shown there is
     the rate actually achieved.
  5. Only then, in water.
- **Know in advance**: working distance **0.3~0.8 m** (registration fails at 2.4 m), HFOV **63.9°** (at 0.5 m, a field of view
  62 cm wide), and the intrinsics are an **underwater calibration** (the 2026-08-04 entry above), so in a desk test on land
  depth reads **~1.33× long** — a physical 0.5 m showing up as `d ≈ 0.67 m` is
  normal, and once in water the metric is right. (2026-08-10 correction: this entry was originally
  written the other way round.)
- **The real unknown in stage 3 is operation, not code.** In water, the object has to be circled on a **75° orbit**
  while holding 0.3~0.8 m (original table: 9.7 mm at 80°, **87 mm at 100°**). Whether that is
  a maneuver the vehicle can make can only be found out from 3~5 above. If not, building the mesh beforehand on land
  and bringing it in with `--pose-mesh` is the realistic way to operate.

### C3 tilt correction: the **translation component is not measured yet** (2026-08-17)
- **What is resolved**: `cam_tilt_deg: 43.3` is in and was confirmed on hardware —
  `rp_residual` dropped **43.3° → roll −0.8 / pitch −0.2**
  `[측정: 2026-08-17, 기체 정지·매트 위, rov_gui GUI 판독]` ("vehicle at rest · on the mat, rov_gui GUI readout").
  Also consistent with two independent corroborations (the IMU→vehicle rotation fit gives 40.7 / 41.3° over axis permutations).
- **Remaining**: the tilt is applied only as a **pure rotation** — it rotates about the camera origin, and
  `cam_t_flu` (lever arm 0.2855 m) is still the pre-remount value. Tilting down on a hinge actually moves the lens
  center too, so that amount (estimated at a few cm or less, **not measured**) remains in the position as a constant
  offset. `rp_residual` looks only at rotation, so it cannot catch this.
- **Workaround**: none. Before drawing any conclusion where cm-level matters, measure the lens center with a tape measure
  and put it into `hw_nav.yaml: cam_t_flu`.
- **Records boundary**: `meta.json hardware.cam_tilt_deg`. Do not pool position/heading across runs where this value
  differs — every record before 2026-08-17 was flown with a level extrinsic and carries up to
  0.21 m of offset + yaw contamination.

### IMU dead reckoning (`--imu-dr`) is **unverified on hardware** (2026-08-17)
- The estimator that updates the state from the C3 BNO086 alone (`rov_gui/control/imu_dr.py`) and its
  closed-loop mode are verified **only up to bench and demo**: 24 unit tests (constant-velocity and constant-acceleration
  integration exact to 1e-9, the 0.5bt²·(1/6)gβt³ law, AHRS steady-state tilt βτ, Kabsch
  rotation+delay recovery), demo-backend closed loop (injected bias comes back as predicted,
  both shadow/control), offline re-estimation round trip.
- **Not tried**: real BNO086 samples, the real mounting rotation (calibration not run →
  shows up as `calibration_sha1: null`), underwater, and **actually driving the vehicle in closed loop**.
- **Most dangerous item**: with `--imu-dr control` the controller **actively drives** the vehicle toward a
  drifting estimate. There is no geofence (entry above), so the only means of stopping it for position reasons
  is the pilot's E-STOP — by operator decision, the automatic abort is off
  (`imu_dr.abort_err_m/abort_max_s: null`; to turn it on, just put in numbers).
- **Order**: the 2 bench calibrations → on a trolley out of the water, check whether the DR trajectory points in **the same direction**
  as the tags → pool station shadow → pool station control → line/square/circle.

### `demo_e2e.py line` does not finish within the driver budget (60 s) (2026-08-17, pre-existing defect)
- **Symptom**: `demo_e2e.py pid line` → `FAIL: line never completed`. **Unrelated to DR**
  (reproduces without `dr` too). station and square pass. circle is not in the driver
  yet — if it is added, the budget must be set not as `2*pi*R*laps/speed` but as
  `2*pi*R*laps/min(speed, sqrt(a_lat*R))`, so as not to reproduce this defect as is
  (at small radii the curvature cap dominates).
- **Estimated cause**: the demo toy plant's tracking error is large (err p95 14 cm), so the governed
  path clock slows down sharply → a 21 s path does not finish within 60 s. That is, the governor
  behaved as designed, and the driver budget is too short.
- **Workaround**: none. line regressions are covered by `test_control.py`'s
  `test_line_mission_is_placed_at_a_tag` and others.
- **Proper fix**: derive the 60 s cap of `demo_e2e` from the path length, or make the demo
  `SHAPES["line"]` shorter. Delete this entry once fixed.

### DP dataset: `extract_pose` composes a tilt 0 extrinsic, but the C3 on hardware is tilted 43.3° down (2026-09-01)
- **Symptom**: on the default `--extrinsic c3` path, `umi_handheld/extract_pose.py:149` composes
  a **cam_tilt_deg = 0 (forward-level)** registration. The hardware value is
  `config/hw_nav.yaml:161` `cam_tilt_deg: 43.3` [측정 2026-08-17].
  The label is "where the BODY (FRD) origin would have been if this camera were on the C3 mount", and
  if that mount pose is wrong, **every position label is off by the attitude-coupling error**.
  The repo has already priced the same mismatch at "up to 0.21 m offset + yaw contamination"
  (the KNOWN_ISSUES 'C3 tilt correction' entry). It does not show up in the reprojection error.
- **Workaround**: none. So far only one session (0026) has had labels made, and it has fix 0, so
  there is no contaminated artifact.
- **Proper fix**: expose `--cam-tilt-deg` in `extract_pose`, or
  reuse `NavConfig.R_t_frd_cam` from `rov_gui/control/geometry.py` (it already composes the tilt correctly),
  and stamp the value used into `poses.json` · `session.json` **as a boundary field**.
  Also physically fix the rig at the same angle with a jig. The tilt translation component is still not measured, so
  measure `cam_t_flu` at the same opportunity. **Fix this before mass capture — afterwards it means relabeling everything.**

### DP dataset: the capture stereo settings and the flight stereo settings differ across the board, and the warp target grid is not the deployment grid either (2026-09-01)
- **Symptom**: the two paths share effectively no knob.
  Capture (`configs/pipeline.yaml` stereo_depth): extended **true**, subpixel **true** (3 bit),
  median **off**, confidence 245, `depth_align: rectified_left`,
  decimation 2 + speckle + temporal + spatial(hole_filling 3) + threshold 100~3000 mm.
  Flight (`rov_gui/backends/hardware.py:425-447` overrides nothing when it builds `StreamConfig`
  → `c3_camera/config.py:179-192` defaults): extended **False**, subpixel **False**,
  median **5x5**, preset robotics, `depth_align: "color"`, post-processing stages: **none**.
- **Consequences**: (a) MinZ 300 mm (flight) vs ~112 mm (capture) — the segment just before the grasp physically
  does not exist for the deployed sensor; (b) training depth is hole-filled · temporally smoothed · cut at 3 m, deployed depth is raw;
  (c) the warp target grid is the **raw CAM_B** of `configs/target_camera_underwater.yaml`, but
  what deployment serves is **CAM_A (colour)-aligned** depth.
- **Workaround**: none. Capturing now would set the whole dataset in stone with this mismatch.
- **Proper fix**: decide on **one** config, enforce it on both sides, and stamp its hash in every session.
  For the grid, one of two — switch flight to `depth_align=rectified_left` (but the rectified model has
  no extrinsics in EEPROM, so an underwater recalibration is needed), or build a **CAM_A target model** from the same EEPROM dump
  and point the warp at it. `zarr.z_near_m: 0.20` also has to be raised above the deployed MinZ.

### DP dataset: the pose-label yield of the 25 existing handheld demos was **measured at 0%** (2026-09-01)
- **Symptom**: `data/20260831/demonstration_0026/poses.json` = n_frames 576, **n_fix 0**,
  reject_counts `{"no tags": 576}`. Independently re-detecting all 25 sessions (10 frames per session,
  pupil_apriltags tag36h11) also gives **0 detections** other than id 0/1 (gripper fingers).
  The cause is not the "tag family" the plan document wrote down but **indoor office scenes with no environment tags at all**.
  The two hedges "not recorded/estimated" in the plan document `docs/DP_TRAJECTORY_PLAN.ko.md:180-183` should be replaced
  by this measurement. On top of that, those 25 are not in the working tree (2026-08-31 filter-branch accident;
  they exist only in the `refs/original` backup ref — do not `git gc` until they are recovered).
- **Workaround**: none. Data usable for labels: **0**.
- **Proper fix**: a new capture. `record.py`'s `TagProbe` currently looks only at the gripper finger tags
  (`record.py:714-715` builds it from `cfg['gripper']['aruco']`), so extend it to show the **number of mapped environment tags**
  live and write it to `frames.csv`, and only then capture. Otherwise this accident repeats
  (0026 was discovered only 24 days later).

### DP dataset: `build_zarr` has no pose/action arrays, and the gripper channel is a constant 0.0 in every store (2026-09-01)
- **Symptom**: the arrays `umi_handheld/build_zarr.py` writes are only the four camera / camera_1 /
  camera0_main_depth / tracked_gripper_pct — **neither pose nor action** (it is the schema for the prior work's
  affordance pipeline). In all 7 stores under `zarr/`, `tracked_gripper_pct` is
  the single value 0.0 and `gripper_dimensions_are_placeholders=true`, and
  `meta/episode_ends` has shape (1,) = **just 1 episode**, so multi-episode slicing
  has never been run even once.
- **Incidental defect**: the `data/` rule at `.gitignore:34` is not anchored, so `zarr/*.zarr/data`
  is ignored wholesale → `git ls-files zarr` gives only 28 entries (.zgroup/.zarray/episode_ends), yet
  `git status` says nothing. A fresh clone gets **a zarr with no pixels**.
- **Proper fix**: when building the slicer (A2), hook up `poses.npy`+`gripper_width.npy`
  and add the arrays. Anchor `.gitignore` as `/data/`.

### cv2 5.0's "thick black + thin white" outline leaves the tail of long labels as a ghost (2026-09-06)
- **Symptom**: in OpenCV 5.0.0, the glyph advance width of `putText` varies with thickness
  [측정 2026-09-06, `cv2.getTextSize`, the same 79-character string @scale 0.40: **thickness 1 → 408 px,
  thickness 2/3/4 → 435 px**]. So the idiom of overdrawing black 3 → white 1 at the same org
  pushes the outline progressively to the right of the fill, and at the end of the string, offset by 27 px, **the tail characters
  remain as a shadow**. The longer the label, the more visible it is; on short labels it does not show.
- **Affected code (not fixed)**: `rov_gui/tools/depth_compare.py:272-277` `_put`,
  `c3_camera/viz.py:133-143`. Both run in envs with cv2 5.0.0 (`fstereo`/`oakd`/`rovgui-pose`).
  It does not reproduce on cv2 4.x (`robust` 4.10, `umi` 4.7).
- **Workaround**: `text()` in `data_collection/make_depth_trajectory_video.py` fixes the thickness at 1 and
  draws the outline as 4 copies at ±1 px offsets — the advance widths become equal, so there is no ghost.
- **Proper fix**: do the two places above the same way. It is a purely cosmetic issue and does not affect measured values.


## 📌 Known limitations (no plan to fix right away; do not forget)

### Pool CCTV cameras (`--pool-cams` / `./c3 cctv`): 4 units confirmed on hardware, **`--source hw` station run and hub connection unconfirmed** (2026-10-01)
- **What is confirmed**: After a reboot, the 4 cameras (1 `SPCA2650` + 3 `USB RGB Camera`) were plugged one each into PC ports 1·2·9·10
  of controller `0000:13:00.0` and recorded 1080p30 simultaneously for 15 s — 4 files `complete: true`, dropped 0, 29.5–30.1 fps; while
  recording, the camera process + ffmpeg used 0.87 cores and the station side 0.02 cores [측정: `pool_cam/bench_out/cost_20261001_real4_1080p30.txt`].
  While streaming, a `USB RGB Camera` uses alt 7 = 3072 B/µframe (196.6 Mbit/s reserved) [측정: sysfs `bAlternateSetting`·descriptors 2026-10-01 19:00] —
  just two of them together already exceed the USB 2.0 periodic-transfer limit (6000 B/µframe), yet 4 ran on one controller, so the bandwidth
  is **separate for each PC port** [유도].
- **Unknowns**: (a) **Hub (multi-port adapter)**: cameras behind a hub share the hub's single link, so at 1080p30 only one will
  fit [유도, not tested] — `--list` warns about cameras that share the same PC port. (b) A full `./c3 gui --source hw --pool-cams` run
  (together with C3·ROV·policy GPU). (c) All 4 are on one controller, so a failure like the uvcvideo crash above stops all 4 together —
  splitting them onto ports of other controllers (`0000:11:00.0` etc.) splits the failure scope (not because of bandwidth).

### Paths inside records the app wrote before 2026-09-14 still point to `sessions/…` (2026-09-14)
- **Symptom**: When the data root was folded into the single `data/YYYYMMDD/<run>[_kind]/` (`tools/migrate_data_layout_20260914.py`,
  485 moves), the citations in config comments·docs·journal·memory were rewritten to the new paths (292 lines), but **the records the app
  wrote at runtime** were not touched — the `path`/`ref_views_dir` of MESH rows in `_pose.jsonl` (old `sessions/pose_meshes/obj_*`), `log_dir`
  in old dryrun `meta.json` (`rov_gui/tools/dp_policy_out/*/`), `rov_gui/tools/fstereo_bench_out/*.json`, and the bodies of each run's
  `events.log`/`mission_log.txt`. Those paths no longer exist.
- **Resolution path**: Read them by resolving through `moves[]` (from→to) in `data/MIGRATION_20260914.json`. The reason the records are not
  rewritten is the same as the measurement-citation rule — records must stay as they were at the time.
- For the same reason, citations of folders that were already gone at migration time, such as `sessions/demonstration_0015/0019`,
  `sessions/grippercalibration_0004`, `sessions/session_syn` (`configs/pipeline*.yaml`, `umi_handheld/extract_gripper_width.py`, journal),
  were left as they are — they were dead references from the start, and where they went has not been confirmed.

### The z ruler in `replay-html` has diverged from the live panel (2026-09-07)
- **Symptom**: The depth scale of the live `3D` panel became a **fixed right-margin gauge** on 2026-09-07
  (`_z_gauge`/`_paint_z_axis`), but the HTML that `./c3 replay-html` bakes uses an **independent JS port**,
  `rov_gui/tools/replay_html_page.html:411 drawZAxis()`, and that one still anchors to **the hull's
  projected left edge − 34 px**. So if the same run is viewed in the two tools, the ruler stays still in one
  and follows the vehicle around in the other.
- **Why it was left**: It is a separate implementation in another language and has no tests. What the user asked about was the live
  GUI.
- **Workaround**: Read the ruler in the replay HTML as "the old ruler attached beside the vehicle". Do not cite the
  scale positions of the two tools as the same thing.
- **Proper fix**: Port the rules of `_z_gauge()` (fixed window → fixed pixel band, independent of attitude·pan,
  expand-only latch) into `drawZAxis()` as is. For the window constant, it is better to inject the Python-side
  `Z_GAUGE_WINDOW_M` when the HTML is generated, so that the number is not written twice in two places.

### The fallback pose of `policy_observe_smoke` is below the mat (2026-09-07)
- **Symptom**: When there is no fix, `rov_gui/tools/policy_observe_smoke.py:86, :204` falls back to
  `_set_p_act((0.0, 0.0, 0.35))`. Map z is **+ downward**, so +0.35 is **below** the tag
  plane — the 44,548 accepted fixes in the hardware records all lie in the −0.856 .. −0.165 m range
  [측정: `data/2026090*/*/nav_*/fixes.csv`, 32 files] and never went below
  the mat. The new depth gauge has started marking that value with its needle, so the smoke shots now show
  "a depth that never actually existed".
- **Workaround**: Do not cite the depth numbers in the smoke shots.
- **Proper fix**: Change both places to `-0.22`, the measured p50. It is a one-liner, but it was outside the scope of
  this change, so it was left untouched.

### Two-camera fallback + tilt tracking is **unverified on hardware**; the tilt rate is [예측] (2026-09-07)
- **What**: `hw_nav.yaml: fallback` — when the C3 is silent for `after_s` (0.2 s), the Default RGB camera's
  fix takes its place (`control/nav_fusion.py`). The RGB camera's [예측] extrinsic error is learned as a body-frame
  constant E and factored out, and RGB fixes are deferred until it is learned. The CAMERA TILT line in PAYLOAD
  tracks the mount angle (`control/tilt_tracker.py`).
- **What is verified**: Synthetic only. An RGB camera with a planted 40° tilt + 5 cm error is recovered to within 5 mm
  after 8 pairs, and the measured tilt comes out at 40.0° (`rov_gui/tests/test_control.py`
  `test_second_camera_fallback_*`, `test_nav_fusion.py`).
- **What was learned on the first hardware run (2026-09-07 14:52, `0907_145206/nav_145240`)**: Alignment worked
  (`offset 67 mm spread 11 mm`, RGB fixes 63) — what blocked it was the **RGB camera model**. With no distortion
  terms, multi-tag RGB frames all die at the 3 px gate, and of the 63 that passed, 58 were single-tag
  (self-residual 2.1~2.4 px vs C3 0.17). Temporarily `fallback.max_reproj_px: 8.0`
  [예측]; properly: REC NAV (both cameras seeing the mat, several positions·headings) →
  `rov_gui/tools/calibrate_second_cam.py --yaml` → paste into `second_cam` →
  restore the gate. **Done 2026-09-08**: from `nav_111010` (RGB 3,463 frames), fx 713.8 + 5 distortion
  terms, lens position·tilt 15.9° measured → entered into `second_cam`; in the reproduction, RGB acceptance 43.5% → 100%,
  offset between the two solvers 2 mm. The gate is 4.0 (2× the RGB pixel pitch [유도]). Remaining: 1 and 3 below.
- **What needs checking on hardware**:
  1. `second_cam.tilt_rate_deg_s: 30` **[예측]** — never measured even once. Press and hold the mount from
     LEVEL to DOWN for 2 s and read PAYLOAD's `measured:tags` value; that by itself is a
     measurement (the tags must be visible to both cameras). Until then, the ± of a dead-reckoned
     value grows by half of the integrated angle.
  2. The RGB feed's `t_capture` is the **arrival time** (RTP h264 latency 100~300 ms not corrected,
     `hardware.py _publish`). Alignment pairs are formed with C3 fixes within `pair_tol_s` 0.08, so
     the actual time offset within a pair is as large as that latency. **Measured 2026-09-08**: in the reproduction of
     that run, spread 1.7 mm (207 pairs) — negligible at that speed. On runs with fast motion,
     look at `spread_mm` again.
  3. The actual step size at the moment of handover (controller.json `nav_fallback.offset_mm`
     vs `spread_mm`), and whether a handover moves the vehicle while MPC is engaged.
  4. Whether `fallback.min_pairs: 10` fills quickly enough at the real frame rates (with C3 15 Hz +
     RGB 32 fps, ~1 s [예측]).
- **Known limitation (by design)**: The two cameras are **not combined into one PnP** (the RGB calibration is
  all [예측], so it would only contaminate the good C3 solution). When the mount moves, E is discarded and
  relearned, so if the C3 drops out right after a move, that interval is a gap (this can be changed with
  `allow_unaligned: true`, but that means accepting the step).

### replay mission (plan_stream): **unverified on hardware** + no jaw arbitration (2026-08-30, updated 2026-09-02)
- **Unverified on hardware**: The whole of shape `replay` (re-flying a recorded demonstration, `control/plan_stream.py` →
  `set_path_plan_ned`) has been verified only offline (9/9 + 19/19) and up to demo_e2e (actual acados,
  honest-completion verdict). The CRITICAL (deadman did not clear latched button bits)·HIGH (no divergence guard)
  findings of the safety-code-reviewer audit (2026-08-30) are **fixed**.
- **No jaw arbitration (remaining)**: Pilot G/H and replay/policy write the same `cmd_gripper_drive`
  last-writer-wins. The audit judged the 3-fold mitigation (stream missions emit only on state-change edges + default
  `gripper: false` + the deadman now releases a held drive) to be enough for M0,
  but if a worker edge overwrites while the pilot is **holding the button down**, the pilot's intent is pushed aside until
  release. Properly: a jaw-specific arbitration in the sink (pilot priority + a window that ignores worker emissions)
  — before using `replay.gripper: true` / `policy.gripper: true` routinely.
  (Since 2026-09-02 every drive is also integrated into the open-loop width estimator in `MpcWorker.on_gripper_drive`,
  so once arbitration exists, the estimator must be fed from the same place.)

### Live diffusion policy (shape `policy`): TCP/jaw geometry — **only x measured**, y·z not measured (2026-09-02, updated 2026-09-08)
- **Status**: The `--policy` stack has been running in the pool since 2026-09-07 (data/20260907/0907_145206
  onward); the problems that surfaced on hardware are in this file's separate 2026-09-07/08 entries (jaw channel, follower
  deadband, yaw left bias). It is still too early to attribute grasp outcomes to the policy (below).
- **TCP = jaw geometry**: `policy.tcp_body_flu_m: [0.502, 0, -0.17]` = `cam_t_flu` + lens→grip [0.196, 0, −0.275]
  (2026-09-08; x [측정: data/20260908/0908_180453_observe/policy_obs/rgb/000160.jpg — a bottle held in the jaw placed at the
  center of tag 58, + 0908_170428/policy_obs/rgb/000000.jpg bottle height 0.084 (±0.006) m + 0908_175151/policy_obs/rgb/000000.jpg
  open jaw; range 0.187~0.204], z from the CAD vertical [유도], y=0 [예측]). The 0.4165 (sim JAW_POS) of the 2026-09-02~07
  runs was disproved by the images and is a **records boundary** (meta `policy.tcp_body_flu_m`). At 0.338 m from the lens / about
  11° below the optical axis [유도], the jaw is **visible** in the lower part of the frame (rows 265~360) — closer to the optical axis
  than on the handheld (0.346 m ahead along the camera axis / 22° below [유도: dataset config
  offset_m [0.0355, 0.1293, 0.3186]]). **Mismatch vector vs the handheld (not corrected)**: jaw − fingertip,
  in the C3 optical frame (x right, y down, z forward) = [−0.036, −0.062, +0.012] m (|0.072|; the old 0.4165 gave
  [−0.036, −0.004, −0.050], |0.062|) = body FLU forward +0.051·left +0.036·up +0.037 [유도: `NavConfig.R_t_frd_cam` + `tcp_offset_from_body`].
  dp is camera-relative, so the camera trajectory is independent of this key (t_bt cancels out exactly when yaw is constant), and
  this vector is "how far the jaw ends up off the object when the policy places the camera as trained" → the expected first
  grasp failure is **~5 cm overshoot·~4 cm high·~3.6 cm to the left**, not "falling short" (the old value gave 3.4 cm short with
  the same left/up; if the jaw really is 0.011 m right of the axis, then 2.4 cm left). Do not override this with `tcp_offset_cam_m`
  — the jaw is there; a correction is a separate decision·a separate records boundary (e.g. a plan-level offset). Remaining: jaw
  y·z not measured, x/z of `cam_t_flu` itself suspected (separate entry; the jaw is anchored to the lens, so the TCP is right
  either way).
- **Properly**: Measure the jaw center in the C3 frame by stereo triangulation (`measure_rig.py --pick-tcp`-style) and
  put it into `tcp_offset_cam_m`; then the lens position cancels out. Until then, do not attribute grasp outcomes to the policy.

### policy: device depth is **refused**, FoundationStereo is the only parity path (2026-09-02)
- Without `--fstereo`, `--source hw --policy` **refuses** to start (a missing `--mpc` is
  only a warning, and the refusal is decided before the warning). The C3's own depth is CAM_A-aligned (63.7°
  FOV), so it covers only the part that falls inside the policy FOV (CAM_B 85.6°) [유도: FOV comparison, area ratio
  not computed], and it is an on-device matcher with the ×0.64 stopgap applied. `--policy-allow-device-depth` is
  for bench experiments; along with a warning, `depth_scale_applied: 0.64` and the coverage are left in meta.
- `--fstereo` also warns if it differs from the training store's settings (iters 16 / scale 1.0 [측정: `~/Desktop/data
  collection/depth/0/depth.zarr/.zattrs`]) (until 2026-09-02 this was
  refused without `--policy-allow-fs-mismatch`; now that flag is only recorded in meta).
  The checkpoint is compared not by the typed value but down to the default that will actually be loaded, and since 2026-09-30 not
  by path but by content (sha1 of the first 8 MiB; if the store-side file has been deleted, against the sha1 of the upstream public
  release) (for runs from 2026-09-14~09-30 this check did not run because of the `data/checkpoints` symlink — fixed). Measured
  with this config: 7.26 Hz, solve 137 ms, panel latency 243 ms [측정: rov_gui/tools/fstereo_bench_out/policy_bench_20260902_141224.json, C3 hardware, alpha 0.5, scale 1.0, iters 16, 330 frames, GPU shared with a concurrent test run].

### policy: at FS 7.26 Hz the depth pair is **2.07×** the training stride — instrument parity and time parity conflict (2026-09-02)
- Training observations are two frames 66.7 ms apart, but FoundationStereo at the `--policy` defaults (iters 16 / scale 1.0 / alpha 0.5)
  runs at 7.26 Hz, solve 137 ms, panel latency 243 ms [측정: rov_gui/tools/fstereo_bench_out/policy_bench_20260902_141224.json, C3 hardware, alpha 0.5, scale 1.0, iters 16, 330 frames, GPU shared with a concurrent test run], so the two depth frames of one observation are a frame interval of 138 ms = 2.07×
  apart (`obs_pair_dt_s` recorded per plan; the pair cap is 3.0·obs_dt = 200 ms, so they are not
  skipped). The two proprio rows are now taken at **the same 138 ms**, so the image–proprio baselines match
  (verification fix 2026-09-02), but the fact remains that the network sees a 2× baseline it never saw in training.
  `--fstereo-scale 0.5` + `--policy-allow-fs-mismatch` gets ~15 Hz pairs (≈ inside the near
  window) at the price of an instrument mismatch (training scale 1.0) — which mismatch is cheaper is not known
  before a hardware A/B [예측]. To fix it: a 20 Hz FS (mono MJPEG, fstereo memory) or a scale 0.5 depth store
  with retraining.

### policy: proprio SNR — the 66.7 ms motion cue is the same order of magnitude as the tag-fix noise (2026-09-02)
- The policy's relative-pose observation is the difference of two rows (66.7 ms in training; on hardware 138 ms, the same as the
  depth-pair interval — entry above). At 0.08 m/s that displacement is ~5 mm over 66.7 ms and ~11 mm over 138 ms
  [유도], while the position noise of single-tag PnP is on the order of ~1 cm [예측: same order of magnitude as the demo noise
  model; for the hardware residual fit see item 3 (EAOB sigmas) of the entry "`rov_gui --mpc`: the whole closed-loop MPC stack is
  unverified on hardware and in water" below]. Fix-based history (A6) and degenerate /
  fix-lag skips prevent **quantization** and cues that silently shrink, but they cannot raise the SNR itself.
  `dp_policy_offline.py noise` is the tool that measures the sensitivity; read the hardware values from the first run's
  plans.jsonl `lowdim`.

### policy: hold-tail cost mask (A10) is **unverified on hardware** (2026-09-02)
- `policy.hold_tail: mask` sets the position/linear-velocity weights of the stages beyond the end of the plan to 0
  (`HwDobMpc._apply_stage_weights`) — but **only while the plan is alive and not latched**
  (the `live` gate in `_tick_replay`, verification fix 2026-09-02: before that, even after plan expiry·halt the whole
  horizon was masked and the x/y/z·heave weights were 0; after expiry, `w_stage=None`,
  full endpoint-hold weights). Offline, only that StubCtrl receives w_stage and that
  acados runs to completion have been checked. Whether "hitting the brakes early" really
  disappears on hardware, and whether the mask leaves the solver condition number undisturbed, are unconfirmed. If suspected,
  A/B against `hold_tail: track` (the previous behaviour).

### policy: validation error is ~36 mm RMS — not enough for precise grasping (2026-09-02)
- Position RMS ~36 mm, [유도] from the val action pos MSE in selected.json. Relative to the 62 mm jaw [스펙],
  the margin is thin. The expectation for the first pool run extends to "the policy pushes the vehicle in a mostly
  right direction"; grasp success rate is a matter for after more demonstrations and retraining.

### policy/fstereo: **2 native crashes** when three model loads overlap — cause unattributed, mitigated by serialization (2026-09-02)
- **Symptom (both on the same day, same process configuration)**: (a) with `rov_gui/tools/policy_dryrun.py` and the actual
  checkpoint, 1 of 4 runs core-dumped right after `DP policy: loading selected.ckpt` (an acados build had just
  finished; FoundationStereo not involved); (b) the 1st run of `rov_gui/tools/policy_bench_check.py` hit, at t≈2.4 s,
  `malloc(): invalid size (unsorted)` — the moment when the DP ckpt load + acados mpcc build + FoundationStereo
  load overlapped. (b) wrote its JSON (FS 7.27 Hz, 220 frames, grid coverage 1.0) but, with the session at
  ready=False, hung at shutdown and was killed by the 400 s timeout
  [측정: rov_gui/tools/fstereo_bench_out/policy_bench_20260902_150726.json]. The retry (151527) and
  the reference run (141224) were normal (3.5~4.3 s load).
- **Cause unattributed**: There is no stack trace (faulthandler was not set up). All that is known is that it is the combination
  of two torch.load calls + dlopen of a just-built solver `.so` overlapping across three threads.
- **Workaround (applied)**: `rov_gui/perception/upstream.MODEL_LOAD_LOCK` **serializes** the FoundationStereo
  load (`FStereoSession._load`) and the policy load (`DpPolicySession.load`), and
  `rov_gui/__main__.main` calls `faulthandler.enable()` so that the next crash dumps the threads. The acados
  build is still parallel (`MpcWorker.setup`).
- **Proper fix**: If it recurs, identify the culprit from the faulthandler stack, and if needed move the acados build behind
  the same lock too (startup ~+3 s). To try reproducing it, run `policy_bench_check.py` under `timeout -s KILL 150`
  — if a hung process keeps hold of the camera, the next run dies with DeviceBusy.

### Hardware Newton gripper: servo channel·PWM range·sustained-close behaviour **not recorded** (2026-08-30)
- The only measurements in the repo are two button function numbers (BTN0/15 = 77/76, 2026-08-06,
  `rov_gui/__main__.py:250`). servo9=Newton is a single line of prose (c3_camera/dataset.py:904), and
  SERVOn_FUNCTION/MIN/MAX/TRIM appear nowhere [스펙 unconfirmed]. Stall current·heat·clutch behaviour under a sustained
  close bit is also unconfirmed — the 4 s auto-neutral of the replay gripper's `gripper_hold_max_s`
  is UMI-U practice, not a measurement of this device [예측].
- Properly: Dump the SERVO parameters with QGC/mavproxy and record them here, measure the sustained
  close current once on the bench, and replace hold_max with that number.

### `rov_gui --mpc`: the whole closed-loop MPC stack is **unverified on hardware and in water** (2026-08-12)
The AprilTag PnP → EAOB+acados NMPC → MANUAL_CONTROL closed loop (`rov_gui/control/`) has
been verified only up to the bench: acados smoke (rovgui-pose, numpy 2, p50 3.1 ms), offline
tests 18/18, demo-backend closed-loop square 1 lap completed (0 solver failures). What necessarily
remains before hardware:
1. **All 4 axis gains are [예측]** (`config/hw_mpc.yaml: axis_gain`). They were ballparked from the T200
   curve + mixer geometry, and they also depend on the pilot gain setting in MANUAL mode.
   → Replace them via the P4 step calibration and update this item. Even when wrong it is at least
   safe (axis_cap 0.5 + geofence + deadman); only performance collapses.
2. **The tag-axis assumption of the wall preset (`x_into_wall`) is unconfirmed** — the assumption that the tag is mounted
   upright on a vertical wall. If it is off, it is caught by `rp_residual_deg` (tag pose vs ATTITUDE), which
   state_assembler reports every tick → check at rest in P3.
3. **The EAOB sigmas are also [예측]** — fit the real PnP/velocity-differentiation noise from P3 records and
   update `hw_mpc.yaml: eaob_sigmas`.
4. **MANUAL_CONTROL has no roll/pitch axes, so the MPC's K/M outputs are discarded** — this relies on the heavy's
   passive restoring moment. It should be fine in a calm pool, but that is [예측].
5. The camera extrinsic origin is relative to the heavy_c3 COM (`c3_payload_frames.json`) — the COM moves when the
   gripper is mounted. On a downward remount (floor geometry), re-measuring the extrinsic is mandatory.
6. **EAOB/NMPC assume a fixed dt=50 ms, but the QTimer tick has jitter** — the observer
   prediction and the reference sampling use a fixed grid rather than the wall clock, so when the GUI process is busy
   (video decode, screen recording) the effective tick stretches and the model error leaks into the EAOB's w by that
   much. On the bench solve p99 is 6.4 ms, so the margin is large, but in a water session check the CSV's
   `solve_ms`·row spacing. If it is bad, add propagation of the measured dt to MpcWorker.
   (Raised by the 2026-08-12 multi-agent review, unverified because of the session limit — an item with judgement deferred.)

### `rov_gui`: camera tilt · joystick passthrough · depth 20 fps are **unverified on hardware** (2026-08-06)
All three for the same reason — on the day they were implemented, the C3 crashed and vanished from the network, and the ROV
was also disconnected. The code has been verified only with the demo backend and 27 offline tests.

1. **The camera-tilt function numbers are [스펙].** In `BTN_FUNCTION`,
   `mount_tilt_up=22 / mount_tilt_down=23 / mount_center=21` come from the ArduSub `JSButton`
   enum and **have never been confirmed on this vehicle** (the repo has no ArduPilot source either).
   The button numbers 9/10/7 come from the user's capture of the QGC button assignment, so they have evidence behind them.
   - **Mitigated**: On connect, `BTNn_FUNCTION` is read and cross-checked, and on a mismatch the button is **not
     pressed** and both possibilities are written to the log. So the symptom when it is wrong is "tilt silently does nothing +
     red log", not "the wrong function gets pressed".
   - **How to check**: After connecting the ROV, in the `./c3 gui --source hw --allow-command` log, check the
     three `button 10 = mount_tilt_up — tilt_up ok` lines. On a mismatch, use the actual values the log reports
     to fix `BTN_FUNCTION`, and delete item 1 from this entry.
2. **The joystick button translation has not been run on hardware.** On 2026-08-07, sending the kernel numbers
   as is caused **an accident in which the tilt button armed the vehicle and the motors came alive**; it was fixed by adding the SDL
   translation (matches 4 hardware observations, `test_pad_buttons_are_translated_to_the_vehicles_numbering`).
   The translation itself has not yet been confirmed on the vehicle.
   - **How to check**: With only COMMAND ENABLE on and **while DISARMED**, press the buttons one at a time and
     look at the `btn 커널>기체` ("btn kernel>vehicle") value in the `JOY` line. **The left of the arrow (kernel number) depends on
     the cable** — over BT, LB/RB `6>9`,`7>10` · View `10>4` · Menu `11>6`;
     over USB, LB/RB `4>9`,`5>10` · View `6>4` · Menu `7>6` (both measured 2026-08-17).
     **It is correct only when the right side (vehicle number) is the same for both**, and the D-pad is `>11..14`.
     Only after that, arm.
     The lights must move **one step at a time** (two steps means the chip is issuing the command twice →
     check the `notify=False` path).
   - **Caution**: A gamepad arm is a single button press (only the on-screen ARM is a 1.2 s hold). It is a deliberate choice to match
     QGC, but **a gamepad arm inherits the mode** — only the on-screen ARM requests MANUAL first
     (the bitmask goes straight to the vehicle, so it cannot be intercepted). If arming from the gamepad, check the mode first.
4. **Flight-mode control is unverified on hardware.** MANUAL (19) etc. are requested with `MAV_CMD_DO_SET_MODE`,
   and the ARM button sends MANUAL first, right before arming. Verified only with the demo backend
   (`test_arm_puts_the_vehicle_in_manual_first`).
   - **How to check**: After ARM, whether the teleop `MANUAL` button lights up (= the vehicle reports MANUAL
     via HEARTBEAT), and whether the thrusters sit still at 1500 µs when armed + input 0. Only when `STAB` is
     pressed should it start moving. If ArduSub refuses a switch while armed, it shows up as the button
     highlight not changing (because it draws the vehicle's report, not the request).
3. **The video-default combination is [유도].** `컬러 640×360 q80 @30 + depth 640×360 @20` ("colour 640×360 q80 @30 + depth 640×360 @20") =
   83.1 Mb/s (92%). The components are all measured (depth formula + re-encoding the C3's own frames), but
   **the combination has never been measured on the link.** 92% is not a large margin.
   - **How to check**: After the camera recovers, whether the HUD shows depth 20.0 fps / drop 0%, colour 30 fps /
     drop 0%, and a total Mb/s below 85. If drops appear, lower in order: `--mjpeg-quality 75` (91%) →
     `--fps 20` (88%).
- **Side finding**: During that sweep the C3 died with `ping was missed → Device likely crashed but did not
  reboot` and vanished from the network until power came back. Reproduction conditions unknown —
  if it shows up again, raise it as a separate entry.

### `rov_gui`: the ROS 2 backend has **never been run even once** (2026-08-06)
- **Fact**: This desktop has no `/opt/ros`, and the `robust` env has no `rclpy` either (checked 2026-08-06).
  `rov_gui/backends/ros2.py` is complete only in structure and **verified only up to the import guard** —
  topic names, QoS choices, the encoding handling in `image_to_bgr`, and the publish timer are all code that has never been run.
- **Impact**: It would be no surprise if `--source ros2` fails the first time it runs. The demo/hw paths are
  covered by `rov_gui/tests/test_offline.py` (14 tests), but nothing covers ros2.
- **Workaround**: Even if it fails, the window comes up and every panel stays OFFLINE (worker exception → `bus.log`).
  So it does not silently draw wrong values.
- **Proper fix**: Run it once against real topics in an interpreter with ROS 2 sourced, check three things —
  (a) `sensor_msgs/Image` step padding, (b) whether `SensorDataQoS` actually conflates,
  (c) whether the header stamp is synced to the host clock (otherwise the latency reads as a constant offset) —
  and then delete this entry. Mixing a conda env with ROS python causes an ABI crash on rclpy's first spin,
  so pick a single interpreter.

### C3 BNO086 IMU: no extrinsic + accelerometer scale +20% → calibration required to use VIO (2026-07-29)
- **Fact 1**: `getImuToCameraExtrinsics(CAM_A)` → `IMU calibration data is not available
  on device yet.` The factory calibration contains only the cameras, so **T_imu_cam is unknown**.
  (Camera intrinsics·stereo extrinsics are fine, so RGB-D/stereo are not affected.)
- **Fact 2 — corrected 2026-08-17: "scale +20%" was a misreading. It is actually an x-axis bias.**
  The original record was "at rest |a| = 11.8 m/s² (+20% relative to gravity) → scale not corrected", but
  that was a value measured **in only one pose (standing upright)**. Re-measured with a 6-pose tumble:

  | | Value |
  |---|---|
  | raw \|a\| (per pose) | **8.37 ~ 11.69** m/s², mean 10.63 |
  | fitted scale | **[1.0013, 1.0071, 1.0092]** — **within 0.9%** of 1 |
  | fitted bias | **[1.803, 0.007, −0.064]** m/s² = **0.184 g, almost all IMU x** |
  | \|a\| after correction | **9.806 ± 0.050** (gravity 9.807) |

  `[측정: data/20260817/0817_101511/ +
  0817_100139/ *_c3_imu.jsonl, 2 takes, 21985 static samples, 6 poses;
  config/c3_imu_calib.json sha1 7081ff43]`

  A scale error cannot produce the pattern in which \|a\| swings 8.4↔11.7 depending on pose. In the upright
  pose the gravity direction is almost parallel to IMU x (gravity dir ≈ [0.83, 0, 0.56]), so the
  1.8 m/s² bias was added as is, and that read as 11.8.
- **Why this distinction matters**: **A fixed bias is removed completely by every run's settling-window static calibration,
  regardless of attitude.** Had it been a scale error, it would have leaked back in as `s·g·sinΔ` every time the attitude
  changed (that term had been used to compute "3.4 m in 10 s at 2° pitch", but the
  term itself does not exist). The dead-reckoning error budget is better than feared.
- **Still true**: the accuracy flag is UNRELIABLE, and the axis directions are not documented by the vendor —
  but `R_frd_imu` has now been measured (below).
- **Impact**: In visual-inertial SLAM (VIO), the scale and the gravity alignment go off. Worse,
  it **fails "as if the algorithm just doesn't fit"**, so it is hard to suspect the IMU as the cause.
  If only RGB-D SLAM / stereo SLAM will be used, it is irrelevant.
- **Workaround**: `c3_collect.py` states both facts in every dataset's `metadata.txt`,
  and leaves an accuracy field on every sample. `c3_dataset_check.py` also warns that the extrinsic is missing.
  → No colleague will use it without knowing.
- **2026-08-17: run completed. `config/c3_imu_calib.json` sha1 `0aae3e4d`.**
  accel scale/bias from a 6-pose tumble, 2 takes (21985 static samples); `R_frd_imu` from 60~100 s
  wiggles, 2 takes. The results are in the "Fact 2" table above.
  **Do not read `R_fit.rms_deg` (6~8°) as the error of R** — that is the noise floor of comparing the C3 raw gyro vs
  ArduSub's **filtered** ATTITUDE, and it does not shrink even with harder shaking
  (7.57 → 6.20). The actual accuracy of R is that **2 independent takes agree within 0.81°**,
  and that is recorded in the JSON's `R_fit.repeatability`.
  Bonus: both takes came out **40.7° / 41.3°** away from an axis permutation — independent corroboration that the camera
  really is tilted by ~40° (the tag-side `cam_tilt_deg` is still not filled in).
- **(History) How the remedy became a tool**.
  `rov_gui/tools/calib_c3_imu.py` measures both —
  (a) accel scale/bias by an ellipsoid fit of an out-of-water tumble (`--fit accel`),
  (b) the **IMU→vehicle** rotation by Kabsch alignment of the C3 gyro vs ArduSub ATTITUDE (`--fit rotation`).
  (b) is the point: **the IMU→camera extrinsic is no longer needed at all** (the device does not have one), and
  the camera tilt is absorbed automatically. The result is `config/c3_imu_calib.json`, and its sha1 is stamped into the run
  meta. Verified on synthetic data (scale/bias recovered exactly, rotation <0.5°, latency
  <2 ms; `rov_gui/tests/test_imu_dr.py`), **not run with the real camera**.
  The tool **rejects** tumbles with insufficient coverage and single-axis wiggles — because confidently giving a wrong
  answer is the real failure mode of this calibration.
- **Proper fix**: Run the two commands above once on hardware, inject the result into
  `c3_collect.py`'s `calibration.json` as well, and then delete this entry.

### C3 direct connection (`c3_camera/`): colour/depth capture-time skew of about 1 frame — whether there is hardware sync is unconfirmed (2026-07-29)
- **Fact**: In the `c3_stream.py` default (`--pair-mode latest`), the `getTimestamp()` difference between colour (CAM_A) and depth
  (CAM_B/C stereo) is observed to be **about 66 ms (≈1 frame interval at 15 fps)**. This appears to be because, inside the same
  pipeline, the colour camera and the mono pair **are not tied together by hardware sync (FSYNC)**.
- **Impact**: While the camera or the scene is moving, the RGB-D registration is off by up to one frame. It is harmless for
  perceiving things at rest, but **it can actually be a problem for collecting·training on moving-manipulation data**
  (66 ms × 0.2 m/s = 1.3 cm misalignment).
- **Workaround**: With `--pair-mode timestamp` (+`--pair-tolerance-ms`), a bundle is emitted only when the capture times of the
  two streams fall within the tolerance. The cost is up to one frame of latency.
  `Bundle.skew_ms` reports the actual skew every frame, so it can be monitored in the HUD.
- **Proper fix**: It is **unconfirmed whether the OAK-D-W-POE supports an FSYNC hardware trigger for CAM_A and the mono
  pair** (depends on whether this board has FFC/FSYNC wiring). If it does, turn on sync in the pipeline.
  If not, consider the depthai `Sync` node (device-side alignment, more latency) or host-side interpolation.
  Do not delete this entry before a verdict.

### The square reference is dynamically infeasible at the corners → an ineradicable ~2 cm corner-error floor (2026-07-24)
- **2026-08-16 hardware path mode mitigation**: `rov_gui`'s `path_following: true` now uses
  active-segment projection + a corner gate; at each vertex it brings the reference velocity to 0 and reveals the next side
  only after confirming the actual vehicle's position/velocity/dwell capture. So the problem below
  **still applies to the legacy wall-clock trajectory mode and the simulator benchmark**, but in the new hardware
  geometric-path mode the cause of cutting corners via the next-leg preview has been blocked.
- **Fact**: The position path of `square_setpoint` is a sharp-cornered square, so at a corner the reference velocity flips 90°
  within one sample (0.05 s) (|Δv|=0.212 m/s → the required acceleration is effectively infinite). On top of that, the yaw reference
  slews at 60°/s (1.5 s for 90°), so it and the position reference **contradict each other** — the turn rate needed to actually round the corner,
  126°/s (lateral force 8 N)~474°/s (30 N), far exceeds the slew cap of 60°/s. Even with infinite force the nose cannot follow  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
  the path. (The slew itself is stated in the `slew_heading` docstring as an intended design.)
- **Quantified** (dobmpc, gentle, **NONE mode = disturbance 0**, lap 2–10 folded): corner 2.00/1.92/2.04/2.39 cm
  vs straight 0.22–0.28 cm (≈10×). **Unchanged to the last decimal when the surge box goes 8→30 N** (the commanded surge at the corners  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
  averages 0.9–1.4 N, nowhere near the box; active on 0.7% of ticks) → **a reference-geometry problem, not authority**.
  The MPC, with preview + a quadratic (2nd-order) cost, turns early and cuts inside the corner (corner-cutting) — an **optimal trade-off**,
  not a malfunction. Under waves the box also contributes in part, but it shrinks corners·straights by almost the same ratio
  (33% vs 37%), so it is not a corner-specific effect; what the box actually improves is the **recovery right after** the corner.  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
- **Impact**: For DOB-MPC the straights are almost perfect at 0.6–1.3 cm, so this corner floor dominates the error budget and
  stands out conspicuously in the trajectory_compare figures (in absolute terms it is the smallest of the 3 controllers: gentle CDW corner  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
  PID 24.6 / MPC 13.2 / DOB 3.8 cm). It cannot be removed by controller tuning.
- **To reduce it**: fix the reference design — a corner fillet (arc/spline, with the radius matched to the achievable lateral force and
  the yaw slew rate) or a corner deceleration profile, or raise the yaw slew cap. Either way the benchmark
  definition changes, so it cannot be compared with existing records → needs the user's judgment.

### heavy rotational added mass = isotropic placeholder
- `[0.12, 0.12, 0.12]` is a temporary value — the literature evidence is weak (von Benzon reports 30–100% error,
  and a competing 0.40 set exists). HOLD until an in-house system ID.

### hydro does not run in MJX (confirmed with the `bluerov.xml` fixture)
- hydro is a CPU passive callback, so MJX does not support it — the bonus check in `verify/verify_gpu_mjx.py` confirmed this,
  non-gating, by loading `bluerov.xml` (now a verification fixture). hydro needs an
  MJX port before the RL phase.

### The C3-BR mount bracket mass is not included in the composite inertia (2026-07-19)
- The bracket of heavy_gripper·heavy_c3 (`meshes/c3_mount.stl`) is **visual-only** — its material/mass is
  unknown, so it is left out of the `compute_payload_inertia.py` composition (estimated at a few hundred g, vs the camera's 1.7 kg).
- Once the user provides the real bracket's mass (or material), add it to the composition like the C3.

### The Newton gripper has been added in Onshape (2026-09-09 export), but the sim GRIP_POS/JAW_POS are still estimates (2026-07-20)
- User request: reflect only what is in the Onshape assembly (vehicle body + C3). The gripper is excluded from
  `heavy_c3` until it is added in CAD. The `heavy_gripper` variant is kept as a config for when the gripper gets added, but
  its current GRIP_POS=[0.25,0,−0.17] is still an **estimate** (unverified against Onshape).
- When the gripper is added in Onshape: rerun the export → update GRIP_POS to the measured position, as for the bracket →
  regenerate heavy_gripper.
- **2026-09-09**: Re-exported the mujoco tab containing the gripper·upper mount·C3 (tilted 40° down)
  (`assets/CAD files/onshape_export_20260909/`; part poses→base_link in payload_frames_20260909.json [유도]).
  CAD jaw-pair centre x 0.364 (tip 0.404), housing x 0.127~0.316, z −0.127 → the cylinder is ~2.5 cm forward·4 cm up from the
  sim GRIP_POS 0.25/z −0.17. **The sim variants·hw config have not been updated yet** (awaiting the user's decision).
- **2026-09-08**: The hardware jaw was measured at 0.196 m in front of the C3 lens (entry "Gripper geometry of the vehicle marker"), so the
  station's lens-anchored jaw became 0.502 ahead of the COM (`rov_shape.JAW_CENTRE_M`, `policy.tcp_body_flu_m`; the tube moved by the same
  +0.0855). The sim's GRIP_POS 0.25 / JAW_POS 0.4165 is **8.5 cm behind** that if `cam_t_flu` is right, and
  **roughly right** if `cam_t_flu` is wrong by ~9 cm toward the front (entry "The x/z of the C3 extrinsic cam_t_flu are suspected"). The
  plant composition (composite COM/inertia) was not touched — an 8.5 cm shift moves the composite COM on the order of a few mm [예측,
  not computed]. Once a tape-measure measurement settles the attribution, update GRIP_POS/JAW_POS at that point.

### The direction sweep shares one seed-0 wave realization — the worst-vertex statistics are a single-realization artifact
- Found 2026-07-21 (`compare_20260720_230025` corner-geometry analysis): every (current, wave) heading-pair
  run sees **the same seed-0 wave time series** (the wave-group **envelope** recurs every 264.8 s ≈ the run
  length 266.7 s) →
  the wave-group at a particular absolute time hits the same lap/vertex in every run (of 400 dobmpc runs, 181 peak at
  t=200–210 s; 66% of the worst vertices are V3). Draw direction-dependent conclusions only from **per-passage relative-angle statistics**;
  per-vertex·per-time claims cannot be published before a multi-seed rerun.
- **Wording correction (2026-08-19)**: "the realization **repeats** every 264.8 s" is **wrong**. In `waves.py:97`,
  `omega = linspace(omega_min, omega_max, N)` gives omega_i = omega_min + i·dOmega, and since omega_min is
  not an integer multiple of dOmega (gentle 0.2/0.0237288 = 59/7 = 8.4286), shifting by T = 2π/dOmega rotates every
  component phase by **the same constant omega_min·T = 154.29°** → eta(t+T) ≈ −0.9·eta(t)
  (gentle reproduction: corr −0.899, max|Δ| 0.583 m vs eta std ~0.13 m [유도: recomputed this session,
  not saved — 산출물 없음]). The exact repeat is 7T = 1853.5 s. What recurs is the **group envelope**
  (|hilbert| corr ≈ 0.995), and the observation above (the same wave-group hits the same vertex) remains
  valid. When citing, do not write "the waves repeat".
- Found 2026-07-19 (during C3 position registration): skin bbox = vendor dimensions × 1.0233 (uniform on all three axes).
- The C3/payload placement is based on **measured metric** (COM anchor), so the dynamics·camera are exact, but
  in the render the payload may look ~3–5 mm off relative to the skin (cosmetic).

### 4 stale/wrong citations from the DP frame audit ((2)(5)(6)(7)) + a 43.4 mm COM gap (2026-09-03; (1) resolved 2026-09-08)
- Found: the 6-reader frame audit (`.claude/journal/research.md` 2026-09-03 entry). The frame chain itself is
  sound (policy=TCP; `policy_frames.py:578` subtracts the lever arm, so the NMPC gets the body-origin trajectory). The problem is
  that **the surrounding docs·comments keep carrying retired values**.
- **(2) "body origin = heavy_c3 COM" (the `geometry.py:331` comment) is wrong.** It is actually **the bare heavy vehicle-body
  COM** (`bluerov2_mujoco_marinegym/compute_payload_inertia.py:45-46` `COM_VEH = np.zeros(3)`).
  heavy_c3 re-origins to its own composite COM, so it differs by 32.5 mm.
- **(3) The point the NMPC regulates and the plant model's COM are 43.4 mm apart.** The station eta is
  the pose of the bare vehicle-body COM (base_link), but the plant the NMPC integrates is `heavy_gripper`
  (`config/hw_mpc.yaml:7`), and that MJCF **resets its origin to the composite COM**
  (`compute_payload_inertia.py:118-122`). Gap (0.03489, 0.00099, −0.02579) m [유도].
  It is exactly the kind of error that `compute_payload_inertia.py:118-122` itself warns about ("moving the origin off the COM makes unmodeled
  rotation-translation coupling destabilize the NMPC"), yet **no config key and no
  meta field records this boundary**. Workaround: none (hardware impact not measured). Proper
  fix: first decide whether it is deliberate → if intended, record it in meta; if not, apply a shift to eta or
  regenerate the plant with the base_link origin.
- **(5) The two-row obs spacing "66.7 ms" is false at runtime.** The actual spacing is the depth pair spacing
  (`rov_gui/backends/policy.py:684` `spacing = pair_dt if how != "dup" else obs_dt`,
  ~138 ms measured). 66.7 ms is the **training stride** (down_sample_steps 2 / fps 30). It is a training-vs-deployment
  motion-cue timescale mismatch, so separately from correcting the wording, the actual impact needs to be assessed.
- **(6) Deployed TCP: the x circularity was broken on 2026-09-08; the z circularity remains.** `tcp_body_flu_m` is now
  `cam_t_flu` + [0.196, 0, −0.275], and its x was measured independently of the CAD chain (0.196 m in front of the lens, in frames where a bottle
  held in the jaw was placed on the centre of tag 58 [측정: data/20260908/0908_180453_observe/policy_obs/rgb/000160.jpg
  + 0908_170428/policy_obs/rgb/000000.jpg + 0908_175151/policy_obs/rgb/000000.jpg]) — the CAD's 110.664 mm has been
  disproved. **The remaining circularity is z**: "the lens is 0.275 m above the jaw" is still the very CAD figure that `cam_t_flu` was derived from
  (the images confirm it within ~1 cm, but it is not an independent measurement), and there are still 0 stereo triangulations. Moreover,
  the x/z of `cam_t_flu` itself are suspected (separate entry) — the jaw is anchored to the lens, so the TCP is right either way, but the body origin
  is not. Proper fix: stereo-triangulate the jaw in the C3 frame and put it directly into `tcp_offset_cam_m`
  (the lens position cancels out).
- **(7) The C3 mount's roll/yaw have not been re-measured since the remount.** `cam_xyaxes_flu` in `hw_nav.yaml` is
  commented out, so the `c3_payload_frames.json` values stay in effect as they are (only the pitch was updated, to 43.3°). Since `R_bt = R_bc`,
  a roll/yaw error **rotates the entire policy action frame**. Moreover, the lens position is from the 2026-09-02
  CAD and the tilt from the 2026-08-17 measurement — the two halves are fixed at different mount times.

### `test_object_nav.test_the_shipped_config_resolves` fails on the deployed config (found 2026-09-07)
- The test asserts `object_nav.max_distance_m ≤ 3.0`, but `config/hw_mpc.yaml` has **10.0**, so it
  drops to 35/36 (the `min_distance_m 0.15 < max 10.0 ≤ 3.0` assertion). It was found while running the suite that calls `_run_meta` during the
  heading dotted-line work, and it is unrelated to that change (no distance key in the config diff =
  an already committed value). Whether to loosen the test cap or revert the config to 3 m is the user's decision —
  10 m is longer than the pool diagonal, so it is effectively no cap.

### RGB checkpoints: since 2026-10-03 the station feeds C3 colour (naive optics · 0 water runs) — RGB+D is still not possible, the offline tools are depth-only (2026-09-23, 2026-10-01, updated 2026-10-03)
- **What was fixed (2026-10-03, removed from here)**: the RGB-only ckpt making 0 plans after READY because of a KeyError on every tick — reproduced on hardware
  [측정: `data/20261003/1003_151227/events.log` RGB START 3 times·`plans.jsonl` 0 RGB plan rows, meta `infer_errors` cumulative 39→158].
  The input is now picked by the contract's image key (README "RGB checkpoints — feeding them the C3 centre lens"). Fixed along with it: a key that cannot be
  supplied·unknown channel order·a non-square `image_shape` give ERROR + ARM refusal instead of READY; `plan_hz`/`infer_ms`/pairing reset on a ckpt swap;
  removed the false OBS VIEW SHIFT notice·FS parity warning for RGB ckpts; no re-planning on the same obs when the fix stops (`skip_same_obs`). Safety audit
  PASS_WITH_FIXES·correctness review 0 bugs (`.claude/journal/reviews.md` 2026-10-03).
- **Remaining**:
  - **RGB+D two-stream ckpt** (`umi_rgbd_5d_*`, 2 image keys) goes to ERROR at station load with `ValueError` ("exactly one image key",
    `dp_policy.py` `_obs_shapes_from_meta`) — there is no path that supplies two images to one obs at the same time.
  - **The offline tools are depth-only**: `rov_gui/tools/policy_session_replay.py` (fixed to `{"camera0_depth": img}`) and
    `rov_gui/tools/dp_policy_offline.py` (held-out) cannot take `camera0_rgb`. `--record-depth` does not record RGB obs either
    (a single warn line) — RGB water runs currently have no raw data to replay·re-evaluate (`policy_obs/rgb/*.jpg` is the C3 original 640x360 JPEG q85, not the 224 obs).
  - **The optics are naive** (operator decision 2026-10-03): per the figures below, the underwater C3 obs is magnified about 1.8× relative to training [유도]. meta
    `policy.worker.obs_mode = rgb`·`obs.optics` is the boundary — never pool with warped RGB runs. **In progress 2026-10-04**: retraining on a store warped with the
    C3 optics (`slam/dataset_rgb_c3warp_peg_9_27_9_28.zarr.zip`, `umi_handheld/build_dp_rgb_c3warp_zarr.py`; the crop is the same as the station's, so no station
    change is needed) (`data/20261004/1004_025349_train_umi_rgb_c3warp_5d_…`) — once it finishes, the naive sentences in this entry do not apply to that ckpt.
    From 2026-10-04 the run records follow the ckpt: meta `obs.optics_kind` `c3_warped` | `naive` (from the training store's `.zattrs rgb.optics`).
  - **FS schedule**: RGB ckpts do not use the gate (`effective` = free) — FoundationStereo keeps running for the panel and shares the GPU with the
    policy forward (the contention figures for depth runs are in the [[policy-plan-latency-budget]] memo, `data/20260930/0930_220212/diag/`). Giving RGB a
    `yield`-style hold too is not implemented (correctness review R1).
  - **0 water runs**: make the first water run with LOW None (observe) — check in plans.jsonl that `obs_t − obs_fix_t[-1]` is on the order of µs and that
    `skip_fix_lag`/`skip_same_obs` are low, then go closed loop (safety audit recommendation).
- **What the training RGB really is**: OAK-D-W colour CAM_A `rgb.mp4` (1280x720), centre crop 224, no rectification·warp
  (`data collection/UMI_Underwater/demonstration_processing/build_dataset.py:83-92,146-150`). The source store is **BGR**, and
  the merged store (`slam/grasp_9_9_26/dataset_rgbd.zarr.zip` — old name `slam/9_9_26/`, the 🐛 path entry below) was saved flipped to RGB
  [측정: `umi_handheld/build_dp_rgbd_zarr.py` channel-order proof, stored-vs-BGR 0.0000 / stored-vs-RGB 10.2925]. It is
  **not pixel-aligned** with depth (colour camera vs the rectified-left grid, warped with the C3 optics) — irrelevant to training because it is two-stream, but
  this data must not be used for 4-channel early fusion.
- **Size of the optics difference [유도, 2026-10-01]**: the training RGB 224 is about 72°×72° with the land CAM_A (OV9782, in air; `grasp_9_9_26/calibration.json`
  socket 0 fx 573.08 @1280x800, 14 coefficients), and the C3 colour cut with the same recipe (IMX378, underwater EEPROM fx 3080.35 @3840 → 513.4 at 640x360;
  `calib/FOV_AUDIT.md:57`) is about 38.6°×38.6° → an image-centre magnification of about 1.8× (an object at the same distance is 1.8× larger in the underwater obs),
  barrel distortion of 13.5 % at the land-crop edge. Training augmentation (crop 0.95, ±5°) cannot cover it. `umi_handheld/warp.py` `WarpStage` remaps camera model
  to camera model without depth (`:74-84`), so RGB can also be warped to C3 CAM_A — but the source is the 1280x720 CAM_A model (fx unchanged, cy−40;
  going through the resolution-scaling path of `CameraModel.K` skews fy by 10 %); the target was created on 2026-10-04 as `configs/target_camera_c3_cam_a_640x360.yaml`
  (converted from the run meta `fstereo.rig.color`, matches the K in `data/20260908/0908_180453_observe/diag/solve_frame.py:22`). After the warp, an asymmetry arises:
  the handheld fingertips (about 22° below the optical axis) fall outside the VFOV ±19.3° while the ROV jaw (about 11.5°) stays visible [유도].

---
*Last updated: 2026-10-05*

### Running the policy+GUI tests back to back in one long pytest process hangs in `pupil_apriltags` (heap corruption) — reproduced at HEAD too (2026-10-03)
- **Symptom**: after the `test_policy_*`·`test_plan_stream`·`test_replay` bundle, running `test_control.py` in the same process hangs forever early in `test_control`
  (TagNavWorker construction → `pupil_apriltags.Detector`). Once, glibc printed `malloc(): mismatching next->prev_size
  (unsorted)` — earlier native code corrupted the heap, and a later malloc/free hung while holding the lock.
  [측정: `rov_gui/tests/fixtures/pytest_chain_hang_worktree_20261003.txt` (malloc message),
  `…_worktree_tagnav_20261003.txt` (stopped at the glob in Detector.__init__), `…_HEAD5c50f99_20261003.txt` (the same spot even in a worktree of
  commit HEAD 5c50f99, which has none of today's changes — stopped in `bindings.py:450 __del__`)].
- **What it is not**: it is not caused by the 2026-10-03 RGB obs change (reproduced at HEAD). Per-file runs·two-file combinations (`test_policy_worker`+`test_control` etc.) pass.
- **Work around it**: run the suite file by file. The culprit library (which of acados·cv2·pupil_apriltags corrupts the heap) is not settled —
  catching the first corruption point with `PYTHONMALLOC=malloc_debug` or `MALLOC_CHECK_=3` is the next step.

---
*Last updated: 2026-10-03*


### Two logging defects: plans.jsonl `follower` is always "mpc", and the IMU `acc` in `*_ekf_inputs.jsonl` is always 0 (found 2026-10-04)
- **Symptom**: (a) the plans.jsonl line in `rov_gui/control/workers.py` is hard-coded as `"follower": ("observe" if self.observe else "mpc")` —
  "mpc" even in rl_pwm_7d·pid runs (`data/20261003/1003_202721/plans.jsonl`). (b) the `acc` of the imu records in `mpc_*_ekf_inputs.jsonl` is
  all `[0.0, 0.0, 0.0]` (same folder) — acceleration does not get into the EKF replay.
- **Impact**: when diagnosing·pooling, read the follower from meta `controller.type`/CSV `mode`. Acceleration fusion in the EKF replay cannot be reproduced from these records.
- **Proper fix**: (a) use `self.cfg.mode` (LOW), (b) check whether the IMU acceleration path (ArduSub SCALED_IMU/RAW_IMU) is carried into the state.

### `demo_e2e.py dobmpc policy` gets 0 plans because of `skip_fix_lag` — in the demo process the dobmpc tick is 70–140 ms later than the fix (2026-09-26)
- **Symptom**: `rov_gui/tests/demo_e2e.py dobmpc policy` (the docstring's default follower) ends with "no PolicyPlan ever reached the bus", because the
  PolicyWorker drops 12/12 pairs as `skip_fix_lag`. `mpc policy`·`mpc station`·`pid policy` of the same driver pass.
- **Measured**: `fix_lag_s` = depth stamp − latest proprio row, as seen through temporary logging (since reverted): dobmpc 0.068–0.141 s (all > FIX_LAG_TOL·obs_dt = 33 ms),
  mpc 0.000–0.067 s (`scratchpad/demo_dobmpc_diag.txt`, `demo_mpc_diag.txt`, 2026-09-26; a session scratchpad, so reproduce with the same temporary logging).
  That is, the gate works as designed, and the cause is that the MpcWorker tick misses the fix stamp in the single demo process (offscreen Qt + acados dobmpc +
  EAOB + synthetic plant) [유도]. Unrelated to the 6-DoF variant (variant OFF, 4-DoF path).
- **Boundary**: this is the row already recorded in the 2026-09-11 journal as "no green baseline for `demo_e2e.py policy`". Hardware dobmpc policy runs have a
  different process layout, so do not carry this over to them as is.
- **Follow-up**: measure the dobmpc tick time in the demo (EAOB first-tick cost / plant RK4), or pin the default follower of demo_e2e's policy row to mpc.
- **Re-observed 2026-10-03**: `demo_e2e.py policy` (default LOW = deployed `dobmpc`) gave the same FAIL in both of 2 runs (`skip_fix_lag` 11–12 times, 0 plans). The
  same with a copy of the deployed config changed to `action_repr: pos_yaw_width` → unrelated to the same day's `action_repr: auto` [측정: session scratch log,
  not kept]. `mpc_tuned policy`·`mpc_tuned policy auto` pass.

### `test_trajectory_panel_ckpt_widgets_share_the_high_row_and_add_no_height` fails when **run alone** — passes with the whole file (found 2026-10-03)
- **Symptom**: `pytest rov_gui/tests/test_offline.py -k share_the_high_row` alone → `policy must add no row`, `minimumSizeHint`
  height 223 vs 224 (policy is 1 px **shorter** than station). In a run of the whole file (122 tests) it passes.
- **Boundary**: run alone in a copy of HEAD (`git archive HEAD`) too, it fails the same way with 223 vs 224 → an order dependence unrelated to the 2026-10-03 change
  (panel 7-dim display) (it appears to be font/style state created by earlier tests [유도]).
- **Proper fix**: have the test pin the font·style before measuring, or change the comparison to "policy adds no row" (≤).

### `demo_e2e.py dobmpc square` is red in this worktree — the config's `plant.linear_damping` override vs the demo plant, and the driver budget (2026-09-26)
- **Symptom**: `rov_gui/tests/demo_e2e.py dobmpc square` → `[error] ctrl[dobmpc]: w_hat has been PINNED to its clip for 3 s (X=15, Y=45)`
  → `FAIL: START did not reach the square. last: approaching the start point (0.37 m)`; after the FAIL the driver calls `app.quit()` without a
  disengage and aborts with "Qt has caught an exception thrown from an event handler … terminate called without an active exception"
  (exit 134). `mpc station`·`mpc policy`·`pid policy`·`mpc replay`·`mpc policy rp` of the same driver pass.
- **Cause [유도]**: `config/hw_mpc.yaml plant.linear_damping [86.7, 133.8, …]` (2026-09-08 hardware value, 21.5× the model's) goes only into the **controller model**,
  while the demo plant (`rov_gui/control/plant.py`) runs on the marinegym original drag (4.03) → the EAOB estimates that difference as a disturbance and
  sticks to the clip during the approach. Rerun with a temporary config without the override, the approach works, but the 10 s settling pushes it past the driver's
  20 s budget and it FAILs at `settling over the start point (5 s)` (the same class as the budget defect of `demo_e2e.py line`). Unrelated to the 6-DoF variant
  (variant OFF; the dobmpc control output is covered by the byte-identical checks of `test_attitude_axes`/`test_control`).
- **Boundary**: the journal has no record of the dobmpc square demo row passing since 2026-08-14 (the last green was before the damping override). Do not carry this over to hardware dobmpc runs.
- **Follow-up**: make the demo plant also receive the `plant.linear_damping` override (if the demo's purpose is "controller model = plant") or raise the driver budget
  by SETTLE_S, and fix the FAIL branch to send `cmd_mpc_engage(False)` before quitting.
