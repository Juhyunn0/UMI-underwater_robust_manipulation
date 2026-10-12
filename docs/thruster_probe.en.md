# Thruster probe block — `rov_gui/tools/thruster_probe.py` operating document (2026-10-10)

> **English translation.** Source of truth: [`thruster_probe.md`](thruster_probe.md) (Korean). Snapshot of 2026-10-12, source blob `6851124400`; later edits to the original are not reflected here.
>
> Provenance tags are kept as in the original: `[측정 …]` measured (artifact cited) · `[유도 …]` derived · `[예측 …]` predicted, not measured · `[스펙 …]` vendor spec · `실측` measured · `미실측` not measured · `산출물 없음` no artifact found · `예시/유도값` example or derived value.

> For operators. The purpose, order, safety checklist, how to read the artifacts, and what to do when a motor FAILs, for the
> five sequences (A–E) run in the pool **before any rl_pwm flight**. Evidence: `RL_controller/logs/eval/pwm_cocontraction_analysis_20261010/REPORT.md` §0–3
> (the 2026-10-10 pool-failure re-analysis) and `RL_controller/README.md` §10.3 item 4 (per-thruster sign measurement — **never performed, not even once**).
> **0 water runs**: every threshold below is [예측] until an artifact path exists. The same tag also sits next to the constants in the code.

## 0. One-line summary

```
P=~/miniforge3/envs/rovgui-pose/bin/python
$P -m rov_gui.tools.thruster_probe --dry-run                 # check the whole logic on the synthetic vehicle, no hardware (A)
$P -m rov_gui.tools.thruster_probe --seq A                   # vehicle: ARMED + MANUAL, propellers ON, pool centre, tether slack
$P -m rov_gui.tools.thruster_probe --seq E,A,B,D --note "jaw empty, 16.1 V"
$P -m rov_gui.tools.thruster_probe --seq C                   # open-loop stiffness (±1 N·m)
$P -m rov_gui.tools.thruster_probe --seq C --free-decay 30   # free-decay record (neutral pulses only)
```

Drives the thrusters one at a time over **the very transmission path** the policy flies on (RC_CHANNELS_OVERRIDE ch 9..16 →
`rl_pwm_override.lua` on the Navigator, every PwmGate item in `rov_gui/control/rl_pwm.py` passed, script heartbeat `RLPWM` fresh)
and judges each one by the vehicle's own ATTITUDE angular rates. The artifact of sequence A, `thruster_sign_probe.json`, is read by
the rl_pwm engage gate (`rov_gui/control/rl_pwm.py ThrusterProbeCheck` → `rov_gui/control/thruster_probe_artifact.load_and_check`).
**Unless `all_pass: true`, not a single rl_pwm pulse goes out.** Artifacts from the synthetic vehicle (`--dry-run`) are marked
`dry_run: true`, and the gate refuses them.

## 1. Why this tool exists

Conclusion of the 2026-10-10 re-analysis (REPORT.md §1): the nose-up of pwm10_s1 is a **trim sign error** in the training world
(−10.6° was read as nose-down but is actually nose-up [측정 data/20260921/0921_182926/mpc_182926.csv]) + **hydrostatic stiffness
3–5× too high** (sim 20.2 vs hardware 4–11 N·m/rad [측정 data/2026100{1,3,5}]), and the runaway current draw of att7pwm_s0 is
co-contraction in the allocation null space n_v = [0,0,0,0,1,1,1,1]. Three verifiers **disproved** a thruster sign flip from the
logs, but the four vertical thrusters are verified only through their heave sum in mixer mode, so **even if one is flipped, the
4-axis policy does not notice and only the pwm policy runs away** (README §10.3 item 4). That single-thruster test is this tool's
A. B/C/D/E are the four numbers that re-derive, by measurement, the ranges for step 1 of the fix plan (rebuilding the training
world): n_v leakage, stiffness k, ESC rest→start-up delay, and stable trim and net weight in water.

## 2. Sequences and recommended order

The recommended order is **E → A → B → D → C** (quietest first; `--seq E,A,B,D`, then `--seq C` separately). Every sequence
sends **neutral (8 channels at 1500 µs)** between steps and on exit, exception, Ctrl-C and abort conditions. The neutral between
steps of A, B (no hold) and C is **extended until the motion dies down** (free attitude period 2.6–3.5 s [측정 REPORT §1] and yaw
decay τ ≈ 2 s mean that with only 3 s the previous pulse's motion remains in the next window; the extension time is written to the
step meta `settle_extra_s` and to `detail.settle_after`).

| Sequence | What | Pulses | Verdict/output | Artifact |
|---|---|---|---|---|
| **E** rest trim | 60 s of neutral pulses with the jaw **empty** | 1500 ×8 | roll/pitch mean and standard deviation (NED, +pitch = nose **up**), depth drift (sign of the net weight in water), current (hotel load) | `rest_trim.json` |
| **A** sign probe | motors m=1..8: +100 µs 1.0 s → neutral 3 s (+ until the gyro has been quiet below 2 °/s for 0.5 s, at most 10 s extra [예측]) → −100 µs 1.0 s → the same neutral | ±100 µs, one motor at a time | dominant-axis (**computed at run time** from the export's B_flu and sigma: vertical → roll, horizontal → yaw) mean angular rate **change** (mean over the 0.15–0.55 s window after the first non-neutral SERVO_OUTPUT_RAW report − 0.5 s baseline before the command) | `thruster_sign_probe.json` |
| **B** n_v leakage | common mode on the four verticals c_v ∈ {0,+100,+200,+300} µs, 5 s each, two altitudes (moved by the operator) | T5..T8 identical | regress depth rate, pitch, roll and current on c_v; slope 0 is nominal | `nv_leak.json` |
| **C** open-loop stiffness | ±1 N·m pitch, ±1 N·m roll, 4 s each (4 s neutral in between) — minimum-norm allocation over the four verticals F = B_v⁺[0,K,M] + T200 inverse | pitch ±61, roll ±34 µs [유도] | steady-state angle change → k = \|τ\|/Δθ; `--free-decay N` sends neutral only, and the decay period after the operator tilts the vehicle ~15° and releases it → k = I(2π/T)² | `stiffness.json` / `free_decay.json` |
| **D** ESC delay | one motor (default T1): 1500→1600 (from rest), 1540→1640 (deadband edge), 1600→1700 (spinning), 3 times, 1.5 s on / 3 s pre-level | ≤ +200 µs | pulse change → gyro onset (\|Δrate\| > 5 °/s) delay, 10–90 % rise time; if the start from rest is ≥ 100 ms longer, the BLHeli_S start-up penalty is real | `esc_latency.json` |

Design notes (where this differs from the spec; all of them surfaced in dry runs):

* **B's common mode is sent without a hold pattern by default** (`--nv-hold-us 0`). The spec's "c_v on top of the mixer depth-hold pattern (+41/−41/−41/+41 [측정 mixer
  mode hold, REPORT §3])" is **not wrench-free in pulse space**: the T200 curve is convex, so the thrust that T5/T8 gain at +41+100 µs
  is much larger than the thrust that T6/T7 lose at −41+100 µs, which gave a 13.7 N net upward force and pitch −3.5° on the synthetic vehicle [유도 dry run]. The null space
  n_v is a direction in **thrust space**. So with hold 0, pulse common mode = thrust common mode (exactly as in the spec); with a hold (`--nv-hold-us 41`
  or `auto`), c_v is added **in thrust units** and the pulses are recovered through the T200 inverse (requested/achieved thrust is written to the artifact). `auto` is a level linearly interpolated from the depth rates of 2.5 s of neutral and
  2.5 s of the 41 µs pattern (30..80 µs, 0 if below the deadband), meant for the case of being 2.5–5.4 N heavy while holding the peg [측정 REPORT §1].
  With a hold, the pause between steps is the hold pattern (c_v 0) — because a heavy vehicle leaves the depth band during the pause.
* **D's default motor is T1 (horizontal, yaw)**. The spec's T5 has the same Basic ESC, but a single vertical at the 1600/1700 levels lifts the vehicle ~0.3 m per repetition,
  so it leaves the depth band before the third repetition, and at 1700 it tilts ~26° (at k 6.7) and trips the 25° abort [유도 dry run]. `--esc-motor 5`
  lets you run it as specified, and a "reposition to the pool centre" prompt appears between levels. Yaw has no restoring moment, so the "steady angular rate" is an actual steady state.
* C's roll pulse ±34 µs is 4 µs above the model deadband (±30 µs). Because the actual ESC deadband is ±25 µs [스펙 BlueRobotics Basic ESC] and
  the small-signal gain near the deadband is half of the datasheet's [측정 REPORT §1 item 5], read the roll k as an **upper bound**; if needed,
  `--stiffness-nm 1.5`.

## 3. Safety checklist (before and during the run)

What the operator must secure before running:

1. Vehicle **ARMED, MANUAL**, propellers **mounted**, pool centre (COM ≥ 0.6 m above the floor), tether slack, **nobody in the water** (do the free-decay tilting
   with a pole; do not put hands near the propellers).
2. Pack ≥ 15.5 V (the tool refuses otherwise); B's +300 µs (~25 A [예측]) only at 15.0 V or above.
3. Lua override installed (`rov_gui/tools/install_rl_pwm_lua.py`; 2026-10-01 install record `data/20261001/1001_145846_vehicle_setup/`),
   QGC closed (port 14550), station GUI kept off (do not have two senders on the same vehicle).
4. Check the net weight in water from E's result, **ballast to near neutral**, then run B (hold 0 by default). Keep the jaw empty.

Conditions the tool checks every tick (20 Hz) and on which it sends **5 neutral frames + a neutral MANUAL_CONTROL** and then exits
(code 3) [예측 thresholds]:

| Condition | Limit | Flag |
|---|---|---|
| \|roll\| or \|pitch\| | > 25° | `--tilt-abort-deg` |
| Depth | outside 0.2..0.9 m, or missing for 2 s or more | `--depth-min-m`, `--depth-max-m` |
| Battery current | > 30 A | `--batt-a-max` |
| SERVO_OUTPUT_RAW age | > 0.5 s | `--servo-age-max-s` |
| ATTITUDE age | > 0.5 s | `--att-age-max-s` |
| Autopilot HEARTBEAT | lost for 3 s | `--hb-age-max-s` |
| DISARMED / not MANUAL / PwmGate failure (parameters, RLPWM heartbeat, script not forcing) | immediately | — |
| Operator Ctrl-C, exception | immediately | — |

Preflight (before any pulse; nothing is sent on RC 9..16): HEARTBEAT, ATTITUDE, SERVO_OUTPUT_RAW, depth and voltage received; all
gate parameters read (MOT_1..8_DIRECTION = the export's `motor_direction`, SERVO1..8 = Motor1..8, SCR_ENABLE/SCR_USER1 1, RC_OPTIONS
bit1 0, RC_OVERRIDE_TIME > 0, RC9..16_OPTION 0, SYSID_MYGCS = 255); RLPWM heartbeat present; every limit above passed. After that,
the first non-neutral pulse goes out only once the script reports `RLPWM = 1` (forcing) during 1.5 s of neutral frames. The tool
**does not write parameters**; the pulses it sends are at most ±300 µs (B), otherwise ≤ ±200 µs. Message rates: it requests
ATTITUDE 25 Hz and SERVO_OUTPUT_RAW 20 Hz via MAV_CMD_SET_MESSAGE_INTERVAL (at the default 4 Hz, a 0.4 s window holds 1–2
samples).

## 4. How to read the artifacts

Folder: `data/<YYYYMMDD>/<MMDD_HHMMSS>_probe/` (`_probedry` for dry runs). Common header `schema 1, written, tool, dry_run, vehicle{sysid,
fw, mot_direction}, export{policy_dir, sigma, B_flu, obs_spec_sha1}`. Every tick goes to `probe_ticks.csv` (t, seq, step, pwm1..8 sent,
servo1..8 reported, roll/pitch/yaw deg NED, p/q/r deg/s, depth m, batt V/A). `probe_run.json` says what ran and what it was aborted by.

### `thruster_sign_probe.json` (the file the gate reads — schema pinned)

```json
{"schema": 1, "written": "...", "tool": "rov_gui/tools/thruster_probe.py", "dry_run": false,
 "vehicle": {"sysid": 1, "fw": "4.5.1", "mot_direction": [-1,-1,1,1,-1,1,1,-1]},
 "export": {"policy_dir": "...", "sigma": [1,1,-1,-1,1,-1,-1,1], "B_flu": [[6x8]], "obs_spec_sha1": "..."},
 "pulse_us": 100, "on_s": 1.0, "off_s": 3.0,
 "motors": [{"motor": 5, "axis": "p", "expected_sign_pos": -1, "mean_rate_pos_dps": -9.2, "mean_rate_neg_dps": 8.1,
             "verdict": "PASS", "n_samples_pos": 10, "n_samples_neg": 10, "detail": {...}}, ...],
 "all_pass": true, "operator_notes": ""}
```

* `axis`/`expected_sign_pos`: the axis and sign of the FRD angular rate that a +pulse must produce. With the pwm10_s1 export: T1/T3 → r−, T2/T4 → r+,
  T5/T6 → p− (starboard up = negative roll), T7/T8 → p+ [유도 obs_spec B_flu; the test recomputes this independently from thrust_pos×thrust_dirs].
* Verdict: **PASS** = sign matches on both pulses **and** \|mean\| ≥ 5 °/s **and** window samples ≥ 3; **WEAK** = sign right but weak / too few samples;
  **FAIL** = sign mismatch, 0, empty window (SERVO_OUTPUT_RAW does not show the pulse), not performed (aborted).
* `detail.pos/neg`: command time, servo report time and latency (a measure of link latency), the window's raw mean (`mean_raw`) and baseline (`baseline_*_dps`), and the
  baseline-subtracted means on all three axes (cross-coupling diagnosis). Why the baseline is subtracted: yaw has no restoring moment, so it decays slowly with τ = I/c ≈ 2 s (N_|r|r 1.55
  [스펙 Wu 2018 BlueROV2]), and the rotation from the previous +pulse is still present in the −pulse window 3 s later and gets read as is [유도 reproduced in a dry run].
* Expected magnitudes [유도]: +100 µs ≈ 4.2 N, −100 µs ≈ 3.4 N (voltage_scale 0.72); roll moment of a single vertical 0.9 N·m, I_roll 0.5–1.05 kg·m²
  → 20–40 °/s after 0.4 s without damping. The synthetic vehicle gave 8–12 °/s. **The horizontal thrusters' −100 µs is borderline**: yaw moment 3.4 N × 0.14–0.20 m
  = 0.49–0.68 N·m, I_yaw ≈ 0.73 kg·m² → window mean ~5 °/s [유도], the same as the threshold. SERVO_OUTPUT_RAW reports come at 20 Hz, so if the window jitters by 50 ms,
  PASS/WEAK can go either way (on the synthetic vehicle, T2 −4.8 vs −5.1 °/s). A WEAK on the negative pulse of T1..T4 is likely a threshold problem, not a sign problem —
  check the sign with the three-axis means in `detail` and retry; the four verticals (0.9 N·m, roll) have margin.

### `nv_leak.json`

Per altitude pass: `steps[]` (c_v, pulses, requested/achieved thrust, depth start/end/rate, pitch/roll mean, current mean and max,
voltage min, servo mean) and `regression`: `depth_rate_mps_per_100us`, `pitch_deg_per_100us`, `roll_deg_per_100us`,
`batt_a_per_100us`, plus the converted `leak_n_per_100us_down_positive` (= depth-rate slope × 15 N·s/m [예측 rov_params heave linear
damping]) and `leak_pitch/roll_nm_per_100us` (= angle slope × 6.7 N·m/rad [유도 cob_arm 0.05 m]). **Nominal prediction = slope 0.**
If `findings` records \|leakage\| ≥ 5 N/100 µs or ≥ 1 N·m/100 µs, that is a finding, and the null-space penalty and the B model of
step 1 of the fix plan have to be revisited. The slope of the current (synthetic vehicle 8 A/100 µs) is the cost of co-contraction itself.

### `stiffness.json` / `free_decay.json`

Per step: `angle_before_deg → angle_steady_deg` (mean over the last 1.5 s of the step), `delta_deg`, `peak_deg` and `t_peak_s`
(overshoot → damping hint), `k_est_nm_per_rad`, `sign_as_expected` (false means the same sign problem as in A). Expected: ~13° if k
is 4.5, 2.8° if it is 20.3 [유도]. `k_pitch/k_roll` are **upper bounds** based on nominal T200×0.72 thrust (if the actual thrust near
the deadband is half, k is half too). Free decay: per axis `period_s` (decay period), `zeta` (logarithmic decrement),
`omega_n_rad_s`, `k_lo/k_hi_nm_per_rad` (I_pitch 0.87–1.25, I_roll 0.49–1.05 kg·m² [유도 rov_params + added-inertia placeholder
0.12]), `extrema`. Fewer than 3 alternating extrema means no period — tilt further or redo it.

### `esc_latency.json`

`cases[]` (pre_us→on_us, predicted steady tilt, per repetition `onset_s_from_cmd`/`onset_s_from_servo`/`peak_dps`/`rise_10_90_s`),
`from_rest_penalty_s` = start delay from rest (referenced to the servo report) − mean start delay while spinning; if ≥ 0.100 s,
`verdict` "BLHeli_S start-up penalty REAL" [스펙 BR forum/BLHeli_S manual 110–130 ms]. A delay referenced to the servo report has
the link latency removed. The synthetic vehicle (0.12 s injected) gave 0.19 s.

### `rest_trim.json`

`roll/pitch_mean_deg ± std` (NED, +pitch = nose up — note that `pitch_deg` in the station CSV has the opposite sign [측정 REPORT §0]),
`depth_rate_mps` and `net_weight_sign` (heavy/buoyant/neutral within resolution, threshold 2 mm/s [예측]), `net_weight_n_est_down_positive`
(× 15 N·s/m [예측]), `batt_a_mean` (hotel load), `batt_v_mean`. Write the jaw-empty and ballast state into `operator_notes` (`--note`).

## 5. If a motor FAILs

1. **Stop everything.** Do not fly rl_pwm / rl_pwm_7d at any cap — the gate refuses on `all_pass: false`, but that is the last
   line of defence, not the procedure. Write the situation in `--note` and leave the artifact folder path in KNOWN_ISSUES.
2. Isolate what is wrong: if the FAIL is an **opposite sign** and \|mean\| ≥ 5 °/s, it is one of (a) the export's `sigma`/`B_flu` (the direction the training world assumed)
   or (b) wiring/propeller/`MOT_n_DIRECTION`. If both of a diagonal pair (5↔8, 6↔7) FAIL at the same time, the wiring is crossed [유도 the dry run caught a 5↔8 swap
   in that form]; if only one, that motor's direction parameter or its propeller (CW/CCW). Commanding +heave in mixer mode (mpc_tuned) while watching `SERVO_OUTPUT_RAW` and
   the vehicle's motion lets you cross-check the mixer's reference direction (2026-10-03 regression fit: up = T5/T8 > 1500, T6/T7 < 1500 [측정 data/20261003]).
3. Decide which side to fix. **The rule is to change the export, not the vehicle** (the training world and the mixer fit `logs/mixer_fit_report.md` are fitted with the vehicle
   as the reference). Flipping one sigma entry flips that entire column of B_flu — fix RL_controller `configs/rov_params.yaml`
   `mixer.motor_direction`/`thrusters.axes`; **retraining and re-export** are required (the policy was trained with that sign). If the wiring is fixed,
   the gate checks that `MOT_n_DIRECTION` matches the export again.
4. **Run A again** and get 8/8 PASS. WEAK (sign right, weak) is not PASS: check the pack voltage, the ATTITUDE rate (whether the 25 Hz request took effect:
   `probe_run.json msg_counts`), and whether the vehicle touched the floor or a wall, then retry. `n_samples` of 1–2 means a message-rate problem.
5. Only then go on to B/D/E/C, and then to the README §10.3 order (station-keeping at cap 0.25).

## 6. Verification status

* Offline: `rov_gui/tests/test_thruster_probe.py` 18 tests + `test_rl_pwm.py` 26 tests passed (2026-10-10). All sequences run in a dry run,
  the 5↔8 swap and reverse rotation detected, and the abort conditions, neutral burst, refusal codes and `load_and_check` refusal items all pinned.
* **Hardware: 0 runs [예측]** — the thresholds (5 °/s, window 0.15–0.55 s, 25°, 0.2–0.9 m, 30 A), whether the message-rate request takes effect over the BlueOS path, the Lua script
  heartbeat timing, the SERVO_OUTPUT_RAW report latency, and whether the tilting during free decay does not trip the 25° abort — the first artifact will tell.
