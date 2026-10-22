#!/usr/bin/env python3
"""dvl_pose.py — the vehicle POSE from the DVL: dead reckoning, with AprilTag
fixes as optional corrections (``--dvl_pose``).

HARDWARE-UNVERIFIED. Written 2026-10-18. Every threshold below is tagged
[예측] unless it names the recording it came from; the replay numbers are
an OPEN-LOOP replay of runs that were flown on the tags, not a flight on this
estimator.

WHY. Today the pose is the tag pose: state_assembler.StateAssembler.step
returns no state at all without a fresh fix, so a tag dropout ends the
engagement after engage.tag_stale_s + tag_stale_hold_s — 15 of the 54
disengages of 2026-10-17 [측정: data/20261017/*/events.log, "DISENGAGED:
tag fix stale"], and the DVL, which measures the motion directly, is only
ever a velocity source (control/dvl.py). This module integrates that
velocity into a pose of its own, in the tag-map frame, so the DVL can carry
the vehicle where the tags cannot.

WHAT EACH AXIS COMES FROM (dead reckoning):

    x, y         the DVL's body-origin velocity (DvlAid: mount rotation +
                 lever arm at the sample's own body rate), rotated into the
                 map by this module's heading and integrated at the
                 samples' own validity times
    yaw          the RAW_IMU gyro, Euler-rate integrated with ArduSub's
                 roll/pitch, its bias learned from the tags while TAG FIX is
                 ON (GyroHeading)
    z            the Bar30 (SCALED_PRESSURE2) + an offset to the map's z,
                 the offset taken on the FIX itself (Z, below)
    roll, pitch  ArduSub ATTITUDE (gravity-referenced, as everywhere)
    u, v         the newest usable DVL sample, held <= hold_s
    w            the Bar30's finite difference on its records' own
                 (vehicle-clock) stamps, bursts skipped, clamped, low-passed
    p, q, r      ArduSub ATTITUDE (as everywhere)

THE HEADING IS THE HARD PART, AND NOT THE COMPASS'S. ArduSub fuses the
compass (EK3_SRC1_YAW 1 [측정: data/20261011/1011_200824_t8_motortest/
analysis/software_config/00000046_parm_initial.tsv]) and the field in this
pool bends with position: on the 2026-10-17 1 x 1 m squares the ArduSub yaw
error ran -9.3..-9.6 deg per metre of x [측정: data/20261017/
1017_dvl_pose_feasibility/sources/heading_lens/out/s4_gyro_mag.txt]. Its
ATTITUDE rates are no way out: they carry EKF3's gyro-bias estimate, learnt
from that same compass — the integrated ATTITUDE rate followed the compass
error (correlation p50 +0.54 over 28 runs [측정: data/20261017/
1017_dvl_pose_feasibility/verify/out/heading_checks.txt]). The RAW_IMU gyro
does not: 60 s heading error p50/p95 0.36/1.20 deg with a same-day bias,
against 2.02/6.94 for the integrated ATTITUDE rate and 3.71/12.55 for the
ArduSub yaw [측정: data/20261017/1017_dvl_pose_feasibility/sources/
heading_lens/out/s5_horizon_rawgyro.txt]. Unbiased it drifts ~2 deg/min
(1.93 deg p50 at 60 s, same file), so its bias is LEARNT from the tags
while TAG FIX is ON, kept for the session, and frozen while OFF.

THE TAG FIX SWITCH (``tag_on``):

    ON   while a FRESH tag state exists (the assembler's, n_tags >= min_tags)
         the pose IS that state — the controller sees what it sees without
         --dvl_pose — and the dead reckoning is re-anchored to it every tick
         (the bridge's per-fix re-anchor, station_bridge.py). In a tag
         dropout the dead reckoning carries the pose. When the tags come
         back (or ON is pressed after OFF), they are believed only after
         ``anchor_n`` fixes agree with each other — collected across the
         gaps of a flickering view, within ``agree_window_s``; the
         difference is then slewed in at corr_rate_* while engaged (a step
         of the accumulated drift would be a setpoint step, the 2026-08-14
         approach-teleport lesson), snapped while not. Only a correction
         AFTER dead reckoning (a re-acquisition) measures the drift: larger
         than max_corr_* in x, y or yaw it ends a control run. A level
         change with the tags in view all along (a held jump) and any z
         difference are slewed whatever their size — the station without
         --dvl_pose steps to them (fix verification 2026-10-18).
    OFF  dead reckoning only. The tags are still detected, recorded and
         compared (``err_m``) — that comparison is the first measurement of
         this estimator's drift, and it is the divergence interlock — but
         they never move the pose. Refused until the dead reckoning can
         carry the vehicle (:meth:`DvlPose.tag_refusal`).
    ONCE pressed while OFF (the operator's use, 2026-10-18: clear the DVL's
         drift now and then on a long run): ON until ONE correction is in —
         agreed, accepted and slewed out — then OFF again by itself, its
         fixes still pending dropped, once :meth:`DvlPose.oneshot_refusal`
         lets it (an anchor, no slew left, a dead reckoning that can fly —
         OFF's refusal minus the agreement in progress, round 9;
         :meth:`DvlPose.start_oneshot`, ``oneshot_due``).
         No correction within ``oneshot_timeout_s``: OFF again, uncorrected.
         The session's first ON (``tag_on_at_start``: the anchor and the
         gyro bias, 60 s) stays ON until it is pressed.

TRUST MODE (``trust_dvl``, 2026-10-19 — the operator's decision after
data/20261019/1019_211310: 96 s of gates before OFF was allowed, then a
DvlAid fit hold ended the engaged run in 1.9 s with the vehicle on the
floor): the DVL is believed as the device reports it, the way a dead
reckoning on a DVL alone is flown elsewhere. The tags are used to ANCHOR
(the first position and heading, ``anchor_n`` fixes) and to CORRECT (TAG
FIX ONCE) — and, since the operator's second decision that evening ("the
diffusion policy must run even when the DVL and the tags differ a lot"),
NEVER to end a run or refuse a press: the OFF divergence past abort_* is
the warning "— TRUST: recorded, not a stop" and one fault record "(trust)
would have ended: ..." per onset; a correction past max_corr_* while
engaged is slewed in at corr_rate_* with the record "(trust) would have
refused: ..."; the ON press and a START are not refused for it
(tag_refusal, pending_too_big). What strict mode had, trust mode drops:
  * the DVL switch needs no proving pass — it goes IN by itself when the
    first sample arrives (the worker; the mount from hw_mpc.yaml is taken
    as measured);
  * DvlAid's tag-velocity holds (dis_diff / dis_align / dis_fit) and its
    latch do not keep a sample from the dead reckoning — only the DVL's
    own data gates do (no lock, altitude under min_altitude_m, fom, beams,
    a velocity jump) and the R3 flags;
  * a DVL gap makes no HOLE — the pose resumes from where the gap found
    it, and the motion the gap may have hidden (``unobserved_m``: gap time
    beyond hold_s x the speed before it, never below
    unobserved_v_floor_m_s) is warned until the tags re-anchor. It is
    bounded once: past max_corr_m it is more than a correction may be while
    engaged — the ONCE the warning asks for would end the run — so it ends
    an engaged run / refuses a START there (the ``unobserved`` stop,
    _watch; safety review 2026-10-19), and a TAG FIX ONCE with tags in
    view, disengaged, snaps it away;
  * OFF does not wait for the gyro bias: it is learnt whenever TAG FIX is
    ON with tags in view, used from the moment it is "ok", and seeded from
    the last session's saved fit (the worker, ``bias_seed_max_age_h``).
What stays: lost_s (no sample for 1.5 s — plus the worker's 0.4 s stale
debounce — ends an engaged run: nothing to fly on; a gap that closes
inside that is flown across), max_dr_speed_m_s, the OFF divergence abort,
the correction limit (max_corr_*), the heading rules (a gyro late past the
cap, a gyro hole). The worker sets DvlAid.trust with it, so the blend's
verdict (the assembler's u, v on a tag tick), the panel's DVL state and
the CSV's dvl_code are the same trusted verdict the pose flies — one
velocity source, not one per tick kind.
A wrong DVL (mount, scale, a false lock the device does not flag) is caught
only when tags are in view, by the divergence abort — the trade the mode
makes. [운용자 결정]

A TAG JUMP IS NOT PASSED THROUGH while the dead reckoning is valid. Between
two fixes the tag-minus-DR difference moves by the dead reckoning's error
over ~66 ms (millimetres); a jump of a fix moves it by the jump (0.83 m on
2026-10-15 [측정: data/20261015/1015_191132/mpc_191521.csv, t ~ 93.5 s]).
A change larger than ``jump_m`` holds the dead reckoning until ``anchor_n``
fixes agree on the new level; a one-frame jump never reaches the state.
With no usable DVL there is nothing to compare against, and the tag state
passes through exactly as without --dvl_pose.

Z. Under z_source tag (the station's) the assembler carries the tag z
between fixes with the TAG finite-difference heave x the fix's age (its u, v
are the DVL's): one 2-tag fix put 18 cm into it on 10-17, and a dead
reckoning anchored to that every tick read the next fix as a jump and its
correction as a fault — 13 of the 111 ON runs of a replay ended so, of
18 the pose ended [측정: data/20261017/1017_dvl_pose_feasibility/
rules_replay/out_round3/on_summary.json].
So the z offset is taken, and the z difference judged, on the FIX at its
capture time against the Bar30 then; the pose shown under ON keeps the
assembler's z (the station's). Where the two part at a dropout the pose
STEPS to the dead reckoning's z — the Bar30 on the fix's offset, the truth
the bridged z drifted from — as the station bridge steps to its pressure
depth. (Round 4 slewed that split in through the correction offset, and
every held jump ratcheted the flown z further: fix verification 2026-10-18,
round 5.) An accepted correction starts its slew from the pose the
controller saw — the offset takes up the STATE's difference — while the
fix's z only decides whether it is a jump. A Bar30 record further from the
last accepted one than 0.5 m/s x the interval + 2 cm is a spike: refused
until the new level has held 1 s.

THE GYRO'S LATENESS. A queued downlink delays the RAW_IMU records with the
integral intact: up to GYRO_EXTRAP_S of it is covered by extrapolating at
the newest rate; beyond that the heading is wrong by the unknown change of
the rate. The pose stops when the uncovered lateness x the newest rate
passes gyro_late_deg (a pivot) or the uncovered lateness itself passes
gyro_late_max_s — 0.1 s, i.e. 0.3 s late: a queued link has no dead
reckoning at all (round 8; the healthy days never pass 0.13 s). The rate
is taken over >= 20 ms (a burst of queued records read tens of times the
rate) and never above 2.5 rad/s. The lateness is
measured on the VEHICLE clock: once a record has been placed on it, a
record the clock map cannot place is lateness UNKNOWN — the heading is
stale until one is placed again (a queue past the reader's 2 s window used
to fall back to the arrival time and read 'late 0.02 s'). And the heading
is ANCHORED where the history already reaches: at a fix's own capture time,
on that fix's own yaw (with the correction in force when it was the pose),
once the gyro history covers it — never at t_now through the extrapolation,
which put the rotation still in the queue into the offset for good (fix
verification round 6: on 10-13's queued link the raw the anchors used was
off by p99 20 deg against the raw the records later showed) — and only when
that fix agrees with the heading then within jump_yaw_deg (round 8: one
tag-yaw outlier anchored so made every clean fix after it a jump). The bias
pairs come from the same covered fixes. The YAW DIFFERENCE is judged the
same way, on the newest fix the history covers (YAW_EVIDENCE_S), for every
check — the gate, the agreement, the correction limit, the OFF abort, the
vouching; with none, on the state at t_now. (Round 6 put the expected difference in
its place, and on a queued link every yaw check read 0: round 7.) A Bar30
whose stamps trail a queued link is carried at its heave to the capture
while it still arrives, so z stays judged on the fix too. What stays
unknowable: a turn that STARTED within the lateness — its records have not
arrived, the newest rate is still the old one — leaves the heading behind by
up to rate x lateness until they do (KNOWN_ISSUES).

GAPS. A usable DVL sample is integrated from the previous one (trapezoid)
if they are at most ``max_interp_s`` apart. In real time the newest velocity
is extrapolated for ``hold_s`` (the trained blend's hold, dvl.blend_hold_s),
then the position is HELD ("held": velocity zero). A gap longer than
``max_interp_s`` is committed the same way (hold_s of the old velocity,
then nothing), so the committed track is the one the controller was shown.
Filling a long gap with the last velocity instead is worse: on the replay it
put 207-273 cm tails on the 60-120 s error [측정: data/20261017/
1017_dvl_pose_feasibility/out/summary.json sensitivity, REPORT.md §6]. With
no usable sample for ``lost_s`` the pose is LOST: no state ("dvl pose
stale", which the worker debounces like a stale tag fix). Motion left
unintegrated since the track's anchor — each gap's time beyond what was shown,
at the speed before it (never below unobserved_v_floor_m_s) — past
``max_unobserved_m``, or one gap after the anchor past ``lost_s``, is a HOLE:
the position is unknown until the tags re-anchor it.

ALL OF THE ABOVE ONLY ON USABLE SAMPLES, as DvlAid judges them (its data
gates, R3 flags, disagreement holds, and — for a pose the controller flies
on — the switch IN and the session's proof). The proof matters MORE here
than for the velocity: a wrong mount is a position feedback of the wrong
sign. Its tolerance (agree_tol 0.25 = ~14 deg or 25 % scale) is loose for
a position [유도]; the residual mount yaw and scale it reports belong in
dvl.mount_rpy_deg before long DVL-only flights.

No Qt, no sockets: plain numpy, tested bare (rov_gui/tests/test_dvl_pose.py).
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from .state_assembler import G_NED, rot_zyx

#: How far past its newest sample the gyro's yaw is extrapolated at that
#: sample's rate (GyroHeading.raw_at) — the link latency it covers. [예측]
GYRO_EXTRAP_S = 0.2
#: How long a fix waits, queued, for the gyro history to reach its capture
#: time (_anchor_heading) — longer than any lateness the late rule lets fly
#: (GYRO_EXTRAP_S + gyro_late_max_s: 0.3 s since round 8, 1.2 s at a
#: gyro_late_max_s of 1.0 — the queued-link tests' setting). [예측]
YAW_QUEUE_S = 3.0
#: How old the newest JUDGED fix (its capture covered by the gyro history)
#: may be for its yaw difference to stand in for this tick's: the late rule
#: lets a heading fly up to GYRO_EXTRAP_S + gyro_late_max_s late (0.3 s
#: since round 8; 1.2 s at 1.0), plus the fix's own capture age. [예측]
YAW_EVIDENCE_S = 1.5


def _fin(v):
    """float(v), or None (None / not a number / not finite)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None

#: ``mode`` values; "off" = not built.
MODES = ("off", "shadow", "control")
#: ``state`` vocabulary (MpcStatus.dvp_state, CSV dvp_src)
ST_NO_ANCHOR = "no_anchor"   # no tag has anchored the dead reckoning yet
ST_TAG = "tag"               # the pose is the tag state
ST_BLEND = "blend"           # the tag state + a correction still slewing in
ST_DR = "dr"                 # DVL dead reckoning
ST_HELD = "held"             # dead reckoning, no usable DVL > hold_s: held
ST_LOST = "lost"             # no pose (DVL / gyro / attitude / depth gone)
ST_FAULT = "fault"           # code fault (the worker's latch)
STATES = (ST_NO_ANCHOR, ST_TAG, ST_BLEND, ST_DR, ST_HELD, ST_LOST, ST_FAULT)
#: ``heading`` values
HEADINGS = ("raw_gyro", "att_rate")

DEFAULTS = {
    # "off" | "shadow" | "control". The command line (--dvl_pose) overrides;
    # a YAML "control" without it is flown as shadow (the imu_dr rule: a
    # config file left in that state must not arm a closed loop on dead
    # reckoning with nobody asking for it).
    "mode": "off",
    # The TAG FIX switch at startup: ON, so the first pose is the tags'.
    "tag_on_at_start": True,
    # ---- heading ----------------------------------------------------------
    # raw_gyro: VehicleImu.gyro_int (RAW_IMU). att_rate: ArduSub ATTITUDE
    # rates — compass-coupled (module doc), the fallback only by choice.
    "heading": "raw_gyro",
    # No new gyro integral for this long = no heading (the backend leaves an
    # interval out of the integral above 0.5 s: VehicleImu.n_gyro_gaps). [예측]
    "gyro_stale_s": 0.5,
    # A LATE gyro (records arriving, each one old — a queued downlink): the
    # newest value is extrapolated over up to GYRO_EXTRAP_S of its age; past
    # that the heading lags by rate x the rest. More than gyro_late_deg of
    # that, or any rate more than gyro_late_max_s past the extrapolation, is
    # no heading ('stale', debounced). A station hold in a queue flies on; a
    # turn does not (10-13's queued link: the gyro-late OFF stops were 0.50-
    # 0.90 s late [측정: data/20261017/1017_dvl_pose_feasibility/
    # rules_replay/out_round5/off_pass.json]; a 20 deg/s pivot flown that
    # late without the rule leaves 20 deg/s x (L - 0.2 s) of heading [유도],
    # 8-14 deg at L 0.6-0.9 s). 3 deg is under what the tag gate itself
    # calls one level (jump_yaw_deg 5): at 1 deg, 20 of the 23 10-13 OFF
    # replay anchors stopped on it (median 3.6 s) [측정: …/out_round5/
    # off_pass_late1deg.json], 18 of them on 2.4-5.9 deg/s station
    # corrections at 1.0-2.6 deg [측정: …/out_round6/evidence/
    # exact_late_round5code.out]. [예측]
    "gyro_late_deg": 3.0,
    # ...and a gyro this far past the extrapolation at ANY rate (0.1 s =
    # 0.3 s late): a turn that STARTS inside the undelivered stretch is in
    # no record yet, so neither rule can see it — the heading is out by rate
    # x lateness, 6 deg at a 20 deg/s pivot 0.3 s late [유도]. The healthy
    # days seen never come near it: RAW_IMU lateness max 0.123 s on 10-15
    # (one segment, ~37 s engaged) and 0.13 s on 10-17 (16 segments, ~46
    # min; 10-16 has no RAW_IMU record), while 10-13's queue put 63 % of
    # its engaged ticks past 0.3 s [측정: data/20261017/
    # 1017_dvl_pose_feasibility/rules_replay/out_round8/evidence/
    # gyro_lateness.json] — a banner every ~2 s [측정: data/20261013/
    # 1013_param_banner_scan/], read as another client re-asking the full
    # parameter list [유도: the same README]. So a queued link has no dead
    # reckoning at all: at 1.0 s, of 56,455 simulated tag losses (the camera
    # stopped after each fix of the real records), 142 — all on 10-13's
    # queued link — were carried more than 3 deg off (10 past 10 deg, max
    # 30-32 deg [측정: …/out_round8/evidence/bridged_dropout_round7code.txt]),
    # and judged a re-acquisition's yaw
    # on the fixes before the
    # dropout (round 8, fix verification wf_21856d6c-69a, docs F1 and
    # estimator F4: …/out_round8/evidence/wf_21856d6c-69a_result.json). [예측]
    "gyro_late_max_s": 0.1,
    # THE BIAS: least-squares slope of (gyro yaw - tag yaw) over the newest
    # bias_window_s of TAG FIX ON fixes. "learning" until the window spans
    # bias_min_span_s with bias_min_n fixes; then "ok". A RAW_IMU bias
    # estimated from the preceding 60 s of tags gave 0.66/1.21 deg (p50/p90)
    # at 60 s and 0.86/1.72 at 120 s on the 10-17 squares [측정: data/
    # 20261017/1017_dvl_pose_feasibility/verify/out/rawgyro_causal.txt]; the
    # bias moved between days (10-15 +1.46, 10-17 -1.28..-2.38 deg/min [측정:
    # .../sources/heading_lens/out/s4_gyro_mag.txt, s5_horizon_rawgyro.txt]).
    "bias_window_s": 120.0,
    "bias_min_span_s": 60.0,
    "bias_min_n": 200,
    # A fix whose (gyro - tag) sits further than this from the fit is left
    # out (a mirror PnP solution, a duplicate-tag mix-up). [예측]
    "bias_outlier_deg": 2.0,
    # control mode: OFF is refused until the bias is "ok". [유도: unbiased
    # RAW_IMU 120 s p50 3.78 deg vs 0.50 with a 120-s tag bias, s5 above]
    # (not under trust_dvl)
    "require_gyro_bias": True,
    # ---- trust mode (module docstring, TRUST MODE) ------------------------
    # True: the DVL is believed as reported — no proving pass, no DvlAid
    # tag-velocity hold or latch on the dead reckoning, no HOLE from a gap,
    # OFF without waiting for the bias. False: the strict gates of rounds
    # 1-10b — the shipped hw_mpc.yaml's value too; --dvl-trust turns it on
    # (operator decision 2026-10-19, data/20261019/1019_211310),
    # --dvl-strict off, whatever the YAML says.
    "trust_dvl": False,
    # A gyro bias a previous session saved (data/dvl_pose_gyro_bias.json,
    # written by the worker whenever the fit is "ok") seeds this session's
    # heading while the file is younger than this (hours); 0 = never. The
    # bias moved between days (10-15 +1.46, 10-17 -1.28..-2.38 deg/min,
    # s4/s5 above), so a day is the longest worth trusting. [예측]
    "bias_seed_max_age_h": 12.0,
    # ---- the DVL ----------------------------------------------------------
    # = dvl.blend_hold_s, the hold the RL exports were trained with [스펙:
    # RL_controller rov_rl/velocity_source.py HOLD_S, via dvl.py DEFAULTS]
    "hold_s": 0.45,
    # = dvl.stale_s: two usable samples at most this far apart are joined
    # (trapezoid); further apart, the gap is committed as hold_s then held.
    "max_interp_s": 0.6,
    # No usable DVL sample for this long = no pose. 10-17: the longest gap
    # per run p50 0.85 s, max 8.78 s [측정: data/20261017/
    # 1017_dvl_pose_feasibility/out/summary.json availability_by_day]. [예측]
    "lost_s": 1.5,
    # ---- tags -------------------------------------------------------------
    # A fix is used (pass-through, anchor, correction, bias, comparison)
    # only with this many tags. 1 = every fix the LOCALIZER accepted — it
    # gates fixes itself (hw_nav.yaml min_tags: 1 since 2026-09-07, its
    # duplicate-tag refit making most of them multi-tag; a wall tag is a
    # single-tag fix by design, and so is the demo's), and a pose
    # that refused a tag state the assembler built would not be "the state
    # without --dvl_pose" while TAG FIX is ON (found by demo_e2e_dvl_pose,
    # 2026-10-18). Raise it to demand more of a correction than of a fix.
    "min_tags": 1,
    # Consecutive fixes that must agree (within jump_m) before a NEW level
    # is believed: the first anchor, a re-acquisition, a jump. [예측: the
    # M-of-N rule, control-theory review 2026-10-18]
    "anchor_n": 5,
    # ...collected over at most this long. The fixes being agreed on (and
    # the divergence abort's evidence) survive the gaps of a FLICKERING view
    # — cleared per tag-less tick, a view that was in sight 0.25 s of every
    # second never re-anchored and never aborted (fix verification
    # 2026-10-18) — but not forever. [예측: 5 fixes at 1 fix/s fit twice]
    "agree_window_s": 10.0,
    # A ONE-SHOT TAG FIX (pressed while OFF: one correction, then OFF again
    # by itself — the operator's way to clear the drift now and then,
    # 2026-10-18) with no correction accepted this long gives up and goes
    # back to OFF, the pose NOT corrected: no tag in view, or fixes that
    # never agree. = agree_window_s, the oldest fix an agreement uses. [예측]
    "oneshot_timeout_s": 10.0,
    # The tag-minus-pose difference may move this much between two fixes
    # and still be the same level; more = a jump. [예측]
    "jump_m": 0.10,
    "jump_yaw_deg": 5.0,
    # ...and z (the Bar30 under the pose vs the tag z; the map plane is ~3
    # deg off gravity, 43-57 mm per metre of tag y [측정: data/20261017/
    # 1017_dvl_pose_feasibility/sources/heading_lens/out/s6_depth_lever.txt],
    # which moves the difference slowly, never between two fixes). [예측]
    "jump_z_m": 0.10,
    # ---- corrections while engaged ---------------------------------------
    # A re-correction is slewed in at most this fast; inside the EAOB's
    # velocity noise (0.05 m/s, hw_mpc.yaml eaob R) [예측, control review].
    "corr_rate_m_s": 0.02,
    "corr_rate_deg_s": 1.0,
    # ...and a correction AFTER dead reckoning (reacquire, blind) larger than
    # this in x, y or yaw, ENGAGED, is not slewed: control mode ends the run
    # (the dead reckoning drifted further than it may be trusted), shadow
    # logs it. A held tag jump and z are slewed whatever their size (_accept).
    # Disengaged, any size is snapped. [예측]
    "max_corr_m": 0.25,
    "max_corr_yaw_deg": 10.0,
    # (the EKF's control_max_z_divergence_m is 0.15; the map tilt above
    # alone reaches ~0.2 m across the pool's 4.4 m, so a little more): a z
    # correction larger than this is WARNED (it is slewed whatever its size
    # — z never ends a run on a correction, the dead reckoning's z IS the
    # Bar30), and it caps abort_err_z_m. NOT a bound of the z change at a
    # tag dropout: that is a STEP to the Bar30-based dead reckoning, as large
    # as the bridging error of that moment (round 5/6). [예측]
    "max_corr_z_m": 0.20,
    # ---- OFF: the divergence interlock -------------------------------------
    # |pose - fresh tag state| with >= min_tags: a warning above warn_*, and
    # in control mode the end of the run above abort_* for abort_hold_s.
    # null abort = off. Each abort may not exceed its max_corr_* (resolve):
    # otherwise a band opens where OFF flies on and pressing ON ends the run
    # (review 2026-10-18). [예측: underwater review 10/30 cm, control 10/25]
    "warn_err_m": 0.10,
    "abort_err_m": 0.25,
    "abort_err_yaw_deg": 10.0,
    "abort_err_z_m": 0.20,
    "abort_hold_s": 1.0,
    # Time without a correction: a panel warning (10-17: p90 > 10 cm after
    # 15-19 s [측정: data/20261017/1017_dvl_pose_feasibility/verify/out/
    # agg_curves_perm_drift.txt]); max_unaided_s ends a control run (null =
    # never). [예측]
    "warn_unaided_s": 20.0,
    "max_unaided_s": None,
    # BLIND = no fresh tag state at all (neither used nor compared): nothing
    # can see the pose drift, and a DVL that is wrong in a way its own gates
    # pass (a slow false lock, a mount inside the proof's 14 deg / 25 %) is
    # integrated at full authority. max_blind_s ends a control run blind
    # that long; null = never, the DEFAULT since 2026-10-18 by the operator's
    # choice: this mode is for long tagless stretches re-fixed by a one-shot
    # TAG FIX now and then (the panel warns after warn_unaided_s). What is
    # left against a DVL wrong that way: max_dr_speed_m_s — a DVL that reads
    # faster than this vehicle's missions fly (<= 0.15 m/s) over a 1 s mean,
    # or in ONE reading at twice it (round 5) — and, with tags in view, the
    # divergence watch. [예측: underwater review "<= 20 s or <= 3.7 m" for a
    # cap; 10-17 p90 passed 10 cm after 15-19 s, data/20261017/
    # 1017_dvl_pose_feasibility/verify/out/agg_curves_perm_drift.txt; 0.3
    # m/s = station_bridge dvl_max_speed_m_s]
    "max_blind_s": None,
    "max_dr_speed_m_s": 0.30,
    # Motion that went unintegrated since the track's anchor, in METRES:
    # each gap's time beyond what the output showed (hold_s), times the
    # speed before it (never below unobserved_v_floor_m_s), summed. Past
    # this, or any single gap after the anchor past lost_s, the track has a
    # HOLE — the position is unknown until the tags re-anchor it (the gyro
    # hole rule). A budget in seconds ended most real OFF runs within
    # minutes whatever the vehicle was doing (KM median 73 s over the
    # 10-13..10-17 sessions, 0 of 747 OFF anchors reached 300 s — fix
    # verification 2026-10-18, scratch replay, not a repo artifact); in
    # metres a station hold pays almost nothing for a gap and a moving
    # mission pays what it could have moved. [예측]
    "max_unobserved_m": 0.10,
    "unobserved_v_floor_m_s": 0.02,
    # ---- z / w --------------------------------------------------------------
    # Older than this = no Bar30 (the assembler's 1.5 s rule).
    "depth_stale_s": 1.5,
    # Recursive low-pass of the Bar30 finite difference (w). [예측]
    "vz_lp_alpha": 0.3,
    # The DVL altitude band the replay measured as good: residual p50 0.37
    # cm/s at 0.2-0.5 m vs 1.16 below 0.10 and 1.18 above 0.50 [측정: data/
    # 20261017/1017_dvl_pose_feasibility/out/concentration.json]. Outside it
    # = a warning, not a refusal.
    "alt_band_m": [0.10, 0.60],
}
KEYS = tuple(DEFAULTS)


def _num(key, v, *, allow_zero=False, allow_none=False):
    if v is None and allow_none:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        f = float("nan")
    if isinstance(v, bool) or not math.isfinite(f) \
            or (f < 0.0 if allow_zero else f <= 0.0):
        raise ValueError(f"dvl_pose.{key} must be a finite number "
                         f"{'>= 0' if allow_zero else '> 0'}"
                         f"{' or null' if allow_none else ''}, got {v!r}")
    return f


def resolve(raw: dict | None) -> dict:
    """DEFAULTS overlaid with ``raw``; unknown keys and bad values RAISE (a
    typo in a safety number must not quietly become its default)."""
    raw = dict(raw or {})
    bad = sorted(set(raw) - set(DEFAULTS))
    if bad:
        raise ValueError(f"dvl_pose: unknown key(s) {bad}")
    c = dict(DEFAULTS)
    c.update(raw)
    c["mode"] = str(c["mode"] or "off").strip().lower()
    if c["mode"] not in MODES:
        raise ValueError(f"dvl_pose.mode must be one of {MODES}, got "
                         f"{c['mode']!r}")
    c["heading"] = str(c["heading"]).strip().lower()
    if c["heading"] not in HEADINGS:
        raise ValueError(f"dvl_pose.heading must be one of {HEADINGS}, got "
                         f"{c['heading']!r}")
    for k in ("tag_on_at_start", "require_gyro_bias", "trust_dvl"):
        if not isinstance(c[k], bool):
            raise ValueError(f"dvl_pose.{k} must be true or false")
    c["bias_seed_max_age_h"] = _num("bias_seed_max_age_h",
                                    c["bias_seed_max_age_h"], allow_zero=True)
    for k in ("gyro_stale_s", "gyro_late_deg", "gyro_late_max_s",
              "bias_window_s", "bias_min_span_s",
              "bias_outlier_deg", "hold_s", "max_interp_s", "lost_s",
              "jump_m", "jump_yaw_deg", "jump_z_m", "corr_rate_m_s",
              "corr_rate_deg_s", "max_corr_m", "max_corr_yaw_deg",
              "max_corr_z_m", "warn_err_m", "abort_hold_s",
              "warn_unaided_s", "depth_stale_s", "max_unobserved_m",
              "unobserved_v_floor_m_s", "agree_window_s",
              "oneshot_timeout_s"):
        c[k] = _num(k, c[k])
    for k in ("abort_err_m", "abort_err_yaw_deg", "abort_err_z_m",
              "max_unaided_s", "max_blind_s", "max_dr_speed_m_s"):
        c[k] = _num(k, c[k], allow_none=True)
    for ab, mx in (("abort_err_m", "max_corr_m"),
                   ("abort_err_yaw_deg", "max_corr_yaw_deg"),
                   ("abort_err_z_m", "max_corr_z_m")):
        if c[ab] is not None and c[ab] > c[mx]:
            # (x, y, yaw: between them OFF would fly on while pressing ON
            # ended the run; z: a difference OFF still flies that the
            # warning above already called too big — kept for symmetry)
            raise ValueError(f"dvl_pose.{ab} ({c[ab]:g}) must not exceed "
                             f"{mx} ({c[mx]:g}): between them OFF would fly "
                             f"on a difference a correction may not be")
    for k in ("bias_min_n", "min_tags", "anchor_n"):
        v = c[k]
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"dvl_pose.{k} must be an integer >= 1, got "
                             f"{v!r}")
    a = float(c["vz_lp_alpha"])
    if not (0.0 < a <= 1.0):
        raise ValueError(f"dvl_pose.vz_lp_alpha must be in (0, 1], got {a!r}")
    c["vz_lp_alpha"] = a
    band = c["alt_band_m"]
    try:
        lo, hi = (float(band[0]), float(band[1]))
    except (TypeError, ValueError, IndexError):
        raise ValueError(f"dvl_pose.alt_band_m must be [low, high], got "
                         f"{band!r}") from None
    if not (0.0 <= lo < hi):
        raise ValueError(f"dvl_pose.alt_band_m must be 0 <= low < high, got "
                         f"{band!r}")
    c["alt_band_m"] = [lo, hi]
    if c["hold_s"] > c["max_interp_s"] or c["max_interp_s"] > c["lost_s"]:
        raise ValueError("dvl_pose: need hold_s <= max_interp_s <= lost_s")
    if c["bias_min_span_s"] > c["bias_window_s"]:
        raise ValueError("dvl_pose: bias_min_span_s must not exceed "
                         "bias_window_s")
    return c


def mode_from(cfg_mode: str, opt) -> tuple:
    """-> (mode, note): the session's mode from hw_mpc.yaml dvl_pose.mode
    and the command line (``--dvl_pose`` None | off | shadow | control).
    The command line wins; a YAML control without it flies shadow."""
    cli = None if opt is None else str(opt).strip().lower()
    if cli in MODES:
        return cli, (f"--dvl_pose {cli}" if cli != "off" else "")
    m = str(cfg_mode or "off").strip().lower()
    if m == "control":
        return "shadow", ("hw_mpc.yaml dvl_pose.mode is control, but only "
                          "the command line arms a closed loop on dead "
                          "reckoning (--dvl_pose): flying SHADOW")
    return (m if m in MODES else "off"), ""


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def yaw_of(R) -> float:
    """Yaw of a body->NED rotation (Rz Ry Rx), the assembler's formula."""
    R = np.asarray(R, float).reshape(3, 3)
    return math.atan2(float(R[1, 0]), float(R[0, 0]))


# =============================================================================
# heading
# =============================================================================
class GyroHeading:
    """Yaw from a gyro, with a bias learnt from the tags.

    ``raw`` is the Euler yaw the gyro alone has turned through since the
    first snapshot (unwrapped, no bias correction) — the quantity the bias
    fit and the history are kept in, and the one an anchor never touches.
    The heading the pose uses is

        psi(t) = raw(t) + off - b * (t - t_anchor)

    with ``off`` set at each anchor (so psi(t_anchor) = the anchor's yaw)
    and ``b`` the bias in force at that anchor (frozen until the next one,
    so OFF flies on one fixed bias). The yaw rate is the Euler one,

        psi_dot = (q sin(phi) + r cos(phi)) / cos(theta)

    from the gyro's body rotation between two snapshots and ArduSub's
    roll/pitch — the body rate r alone under-turns by cos(theta), 1.3-3.1 %
    at this vehicle's 9-14 deg trim [유도: hardware review 2026-10-18].
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.source = cfg["heading"]
        self.raw = 0.0
        self.t: float | None = None         # stamp of the newest raw
        self.rate = 0.0                     # d raw / dt over the newest step
        # ...and when that record ARRIVED (host clock): freshness is judged
        # on it — a queued downlink delays arrivals with the integral intact,
        # and the vehicle-clock age read 42 % of 10-13's engaged ticks as
        # stale (fix verification 2026-10-18, scratch replay)
        self.t_rx: float | None = None
        # ONE CLOCK per history (fix verification round 6): once a record
        # has been placed on the vehicle clock (t_gyro_v / t_att_v) — or from
        # the start when need_vclk (the station's hardware source) — a
        # record that cannot be placed is one of UNKNOWN lateness: it is
        # integrated, not stamped, and the heading is stale until one is
        # placed again. (A queue past the reader's 2 s window fell back to
        # the arrival time, 'late 0.02 s', while the hazard grew.)
        self.need_vclk = False
        self.vclk = False                   # the history is the vehicle's
        self.unplaced = False               # the newest record could not be
        self.t_v: float | None = None       # the newest record's vehicle time
        self._prev_int = None               # raw_gyro: previous gyro_int
        self._prev_n = None
        self._prev_gaps = None
        self._prev_t_att = None             # att_rate: previous ATTITUDE
        self._prev_t_att_v = None           # ...its vehicle time
        self.n_holes = 0                    # intervals with no rotation known
        self.hole_t: float | None = None    # newest hole (a heading break)
        self._hist: deque = deque(maxlen=1200)    # (t, raw), ~60 s at 20 Hz
        # the bias fit
        self._pairs: deque = deque()        # (t, d = raw - psi_tag), unwrapped
        self.bias = 0.0                     # rad/s, in force for new anchors
        self.bias_se = None
        self.bias_state = "none"
        self.bias_span_s = 0.0
        self.bias_n = 0
        self.bias_t: float | None = None    # when the fit last updated
        self.seed: dict | None = None       # a saved bias seeded (seed_bias)
        # the anchor
        self.off = 0.0
        self.t_anchor: float | None = None
        self.b_anchor = 0.0

    # -------------------------------------------------------------- inputs
    def note_imu(self, imu) -> None:
        """One VehicleImu snapshot: advance ``raw`` by the rotation since
        the previous one."""
        if imu is None or imu.roll is None or imu.pitch is None:
            return
        phi, th = float(imu.roll), float(imu.pitch)
        if not (math.isfinite(phi) and math.isfinite(th)):
            # one NaN here would make `raw` NaN for the rest of the session
            # (review 2026-10-18): this snapshot's rotation is not taken
            return
        ct = math.cos(th)
        if abs(ct) < 0.2:                   # > ~78 deg of pitch: not here
            return
        if self.source == "raw_gyro":
            gi = getattr(imu, "gyro_int", None)
            # the newest record's moment on the VEHICLE's clock when the
            # reader could place it (a queued link delays arrivals, not
            # samples — review 2026-10-18), and its host arrival
            tv = _fin(getattr(imu, "t_gyro_v", None))
            trx = _fin(getattr(imu, "t_gyro", None))
            if gi is None or (tv is None and trx is None):
                return
            n = int(getattr(imu, "n_gyro", 0) or 0)
            gaps = int(getattr(imu, "n_gyro_gaps", 0) or 0)
            gi = np.asarray(gi, float).reshape(3)
            if not np.all(np.isfinite(gi)):
                return
            if self._prev_int is None:
                self._prev_int, self._prev_n, self._prev_gaps = gi, n, gaps
                if trx is not None:
                    self.t_rx = trx
                self._place(tv, trx)
                return
            if n == self._prev_n:
                return                      # no new record since the last
            if trx is not None:
                self.t_rx = trx
            if n < self._prev_n or gaps != self._prev_gaps:
                # The reader restarted its integral, or left an interval out
                # of it: the rotation across that interval is UNKNOWN.
                self._hole(tv if tv is not None else trx)
            else:
                d = gi - self._prev_int
                self.raw += (math.sin(phi) * float(d[1])
                             + math.cos(phi) * float(d[2])) / ct
            self._prev_int, self._prev_n, self._prev_gaps = gi, n, gaps
            self._place(tv, trx)
        else:
            ta = _fin(imu.t_att if imu.t_att is not None else imu.stamp)
            if imu.q is None or imu.r is None or ta is None:
                return
            if not (math.isfinite(float(imu.q)) and math.isfinite(float(imu.r))):
                return
            # (the ATTITUDE's own moment on the vehicle clock, as RAW_IMU's:
            # stamped with the parse time the lateness rule could not fire —
            # fix verification round 6)
            tv = _fin(getattr(imu, "t_att_v", None))
            if self._prev_t_att is None:
                self._prev_t_att, self._prev_t_att_v = ta, tv
                self.t_rx = ta
                self._place(tv, ta)
                return
            # the interval on the vehicle clock when both ends have it (the
            # samples' own spacing), else on arrivals
            dt = (tv - self._prev_t_att_v
                  if tv is not None and self._prev_t_att_v is not None
                  else ta - self._prev_t_att)
            if dt <= 0.0:
                return
            self.t_rx = ta
            if dt > self.cfg["gyro_stale_s"]:
                self._hole(tv if tv is not None else ta)
            else:
                self.raw += (float(imu.q) * math.sin(phi)
                             + float(imu.r) * math.cos(phi)) / ct * dt
            self._prev_t_att, self._prev_t_att_v = ta, tv
            self._place(tv, ta)

    def _place(self, tv, ta) -> None:
        """Stamp the newest raw into the history on ONE clock (__init__):
        the vehicle's once a record has been placed on it (the arrival
        stamps before are later by the link — dropped), else the arrival."""
        if tv is not None:
            if not self.vclk:
                self._hist.clear()
                self.vclk = True
            self.t_v = tv
            self.unplaced = False
            self._push(tv)
            return
        self.t_v = None
        if self.vclk or self.need_vclk:
            self.unplaced = True             # integrated, not stamped
            return
        if ta is not None:
            self._push(ta)

    def _push(self, t: float) -> None:
        self.t = t
        if self._hist and t <= self._hist[-1][0]:
            self._hist[-1] = (self._hist[-1][0], self.raw)
        else:
            if self._hist:
                t0, r0 = self._hist[-1]
                if t - t0 >= 0.02:
                    # (over a real interval only, and never faster than the
                    # vehicle turns: a burst of queued records, or a clock
                    # map that moved, gave tens of times the rate — fix
                    # verification 2026-10-18)
                    self.rate = max(-2.5, min(2.5, (self.raw - r0) / (t - t0)))
            self._hist.append((t, self.raw))

    def _hole(self, t: float) -> None:
        self.n_holes += 1
        self.hole_t = t

    # ------------------------------------------------------------- queries
    def stale(self, t_now: float) -> float | None:
        """Seconds since the newest rotation ARRIVED (its host time; the
        sample's own stamp when no arrival is known), or None (never had
        one)."""
        t = self.t_rx if self.t_rx is not None else self.t
        return None if t is None else max(0.0, float(t_now) - t)

    def raw_at(self, t: float) -> float | None:
        """``raw`` at station time ``t``, interpolated; past the newest
        stamp, extrapolated at the newest rate for up to 0.2 s (the newest
        RAW_IMU value is one link latency old: held, it put yaw rate x
        latency into an anchor made at t_now — ~1 deg after a 20 deg/s
        vertex pivot, fix verification 2026-10-18 — while extrapolated, an
        anchor is exact through a steady turn and after it); None before the
        oldest."""
        h = self._hist
        if not h:
            return None
        t = float(t)
        if t >= h[-1][0]:
            return h[-1][1] + self.rate * min(t - h[-1][0], GYRO_EXTRAP_S)
        if t < h[0][0]:
            return None
        lo, hi = 0, len(h) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if h[mid][0] <= t:
                lo = mid
            else:
                hi = mid
        t0, r0 = h[lo]
        t1, r1 = h[hi]
        w = (t - t0) / max(t1 - t0, 1e-9)
        return r0 + w * (r1 - r0)

    def heading(self, t: float) -> float | None:
        """The pose's yaw at ``t`` (wrapped), or None (never anchored, or
        no history reaching ``t``)."""
        if self.t_anchor is None:
            return None
        r = self.raw_at(t)
        if r is None:
            return None
        return wrap(r + self.off - self.b_anchor * (float(t) - self.t_anchor))

    def broken_since_anchor(self) -> bool:
        """A hole after the anchor: the rotation across it is unknown, so
        the heading is not this gyro's any more."""
        return (self.hole_t is not None and self.t_anchor is not None
                and self.hole_t > self.t_anchor)

    # ------------------------------------------------------------- anchors
    def anchor(self, t: float, psi: float) -> bool:
        """psi(t) := ``psi``. False when the history cannot place ``t``."""
        r = self.raw_at(t)
        if r is None:
            return False
        self.off = float(psi) - r
        self.t_anchor = float(t)
        self.b_anchor = float(self.bias)
        return True

    # ------------------------------------------------------------ the bias
    def note_tag_yaw(self, t_capture: float, psi_tag: float) -> None:
        """One tag fix's yaw at its capture time (TAG FIX ON only — the
        caller's rule): a pair for the bias fit."""
        r = self.raw_at(t_capture)
        if r is None:
            return
        d = r - float(psi_tag)
        if self._pairs:
            # unwrap against the previous pair: d changes slowly
            prev = self._pairs[-1][1]
            d = prev + wrap(d - prev)
        t = float(t_capture)
        if self._pairs and t <= self._pairs[-1][0]:
            return
        if self.hole_t is not None and self._pairs \
                and self._pairs[-1][0] < self.hole_t <= t:
            # the gyro's own integral broke between these two: the old pairs
            # describe a different raw
            self._pairs.clear()
        self._pairs.append((t, d))
        lo = t - self.cfg["bias_window_s"]
        while self._pairs and self._pairs[0][0] < lo:
            self._pairs.popleft()
        self._fit(t)

    def _fit(self, t_now: float) -> None:
        c = self.cfg
        if len(self._pairs) < 3:
            if self.bias_t is None:          # never fitted: nothing to keep
                self.bias_state = "learning" if self._pairs else "none"
            self.bias_n = len(self._pairs)
            return
        a = np.asarray(self._pairs, float)
        tt, dd = a[:, 0] - a[0, 0], a[:, 1]
        keep = np.ones(len(tt), bool)
        lim = math.radians(c["bias_outlier_deg"])
        slope = 0.0
        for _ in range(3):
            if int(keep.sum()) < 3:
                break
            A = np.vstack([tt[keep], np.ones(int(keep.sum()))]).T
            sol, *_ = np.linalg.lstsq(A, dd[keep], rcond=None)
            slope, icpt = float(sol[0]), float(sol[1])
            res = dd - (slope * tt + icpt)
            new = np.abs(res) <= lim
            if np.array_equal(new, keep):
                break
            keep = new
        n = int(keep.sum())
        span = float(tt[keep].max() - tt[keep].min()) if n >= 2 else 0.0
        se = None
        if n >= 3 and span > 0.0:
            A = np.vstack([tt[keep], np.ones(n)]).T
            sol, *_ = np.linalg.lstsq(A, dd[keep], rcond=None)
            res = dd[keep] - A @ sol
            s2 = float(res @ res) / max(n - 2, 1)
            sxx = float(((tt[keep] - tt[keep].mean()) ** 2).sum())
            se = math.sqrt(s2 / sxx) if sxx > 0.0 else None
            slope = float(sol[0])
        self.bias_n, self.bias_span_s, self.bias_se = n, span, se
        if n >= c["bias_min_n"] and span >= c["bias_min_span_s"]:
            self.bias = slope
            self.bias_state = "ok"
            self.bias_t = float(t_now)
        elif self.bias_t is None:
            # Too little data is not an estimate (no value is used yet). A
            # window that shrank AFTER a good fit (OFF for a while, a gyro
            # hole) keeps that fit: a same-day bias from another run gave
            # 0.36/1.20 deg (p50/p95) at 60 s [측정: data/20261017/
            # 1017_dvl_pose_feasibility/sources/heading_lens/out/
            # s5_horizon_rawgyro.txt] — the bias is the gyro's, not the run's.
            self.bias_state = "learning"

    def seed_bias(self, bias_dps: float, se_dps, t_now: float,
                  age_h: float) -> None:
        """A bias another session fitted (the worker's saved file): in
        force for new anchors from now, state "saved" until this session's
        own fit is "ok" (_fit keeps a state that has a bias_t). [운용자 결정
        2026-10-19: trust mode flies OFF before the 60 s fit]"""
        self.bias = math.radians(float(bias_dps))
        self.bias_se = (None if se_dps is None
                        else math.radians(float(se_dps)))
        self.bias_state = "saved"
        self.bias_t = float(t_now)
        self.seed = {"dps": float(bias_dps), "se_dps": se_dps,
                     "age_h": round(float(age_h), 2)}

    def bias_info(self) -> dict:
        return {"state": self.bias_state,
                "dps": (math.degrees(self.bias)
                        if self.bias_state in ("ok", "saved") else None),
                "se_dps": (None if self.bias_se is None
                           else math.degrees(self.bias_se)),
                "span_s": round(self.bias_span_s, 1), "n": int(self.bias_n),
                "in_use_dps": math.degrees(self.b_anchor)}


# =============================================================================
# the pose
# =============================================================================
class DvlPose:
    """DVL dead reckoning + tag corrections, in the TAG-MAP NED frame (the
    frame NavFix and the state assembler report in; the worker datumizes
    the output exactly as it datumizes the tag state).

    Inputs, from the worker's own slots: :meth:`note_imu` (every VehicleImu
    snapshot), :meth:`note_dvl` (every DVL sample, after DvlAid has judged
    it), and per control tick :meth:`step`, which is handed the tick's TAG
    state (the assembler's ``meas`` — None when there is no fresh fix) and
    returns this pose's ``meas`` and a health dict. One instance per
    session: the dead reckoning, the heading's bias and the z offset survive
    engagements (an engagement starts from wherever the pose is).

    WHAT ENDS A DEAD RECKONING (review 2026-10-18 — every one of these used
    to let the pose fly on): no usable DVL sample for ``lost_s``; a HOLE in
    the track (one gap past ``lost_s``, or ``max_unobserved_m`` of motion
    left unintegrated since the anchor) or in the heading (a gyro gap) —
    both cleared only by the tags re-anchoring the track; the DVL latched
    OUT by DvlAid; and, in control mode, ``max_blind_s`` without a tag in
    view (when set — off by default since 2026-10-18), a DVL reading faster
    than ``max_dr_speed_m_s``, the OFF divergence from the tags (xy, yaw or
    z), the drift of an ON re-acquisition that has not agreed yet (xy or
    yaw), the drift during a held tag jump (xy or yaw), and a re-correction
    too large to slew (xy or yaw).
    """

    def __init__(self, cfg: dict | None = None, mode: str = "control",
                 nudot_source: str = "imu", imu_stale_s: float = 0.3,
                 z_from_fix: bool = True, need_vclk: bool = False):
        self.cfg = resolve(cfg)
        # z_source tag (the station's): the tag state's z between fixes is
        # the fix's z bridged with the TAG finite-difference heave x age
        # (state_assembler; u, v are the DVL's) — one 2-tag fix put 18 cm
        # into it, and anchoring the dead reckoning's z to that made the next
        # fix a held 'jump' and its correction a fault: 13 of the 111 ON
        # runs of a replay (fix verification 2026-10-18 [측정: data/20261017/
        # 1017_dvl_pose_feasibility/rules_replay/out_round3/on_summary.json,
        # fault:correction_too_big 0/5/1/7]). So the z offset
        # is anchored, and the z difference judged, on the FIX at its own
        # capture time against the Bar30 at that time. z_source pressure:
        # the state's z IS the pressure depth + an offset — anchored on it.
        self.z_from_fix = bool(z_from_fix)
        if mode not in ("shadow", "control"):
            raise ValueError(f"DvlPose mode must be shadow|control, got "
                             f"{mode!r}")
        self.mode = mode
        self.nudot_source = nudot_source
        self.imu_stale_s = float(imu_stale_s)
        self.tag_on = bool(self.cfg["tag_on_at_start"])
        # a ONE-SHOT TAG FIX in progress (start_oneshot): {"t0": press time,
        # "accepted": a correction is in} — None: the switch is the
        # operator's (the session's first ON, or OFF)
        self.oneshot: dict | None = None
        self.head = GyroHeading(self.cfg)
        # the station's hardware: the gyro must be on the vehicle clock from
        # its first record (GyroHeading.need_vclk; round 6)
        self.head.need_vclk = bool(need_vclk)
        self.note = ""                       # the newest refusal / warning
        self.note_t: float | None = None     # ...and when it was said
        self.reset_track()
        # the record (run meta): everything a later reader needs to tell
        # which rows flew on what
        self.toggles: list = []              # (t, on, engaged)
        self.corrections: list = []          # dicts, see _accept
        self.faults: list = []               # (t, why), one per onset
        self.n_jumps_held = 0
        # covered fixes not taken as the heading's anchor: further from the
        # heading then than jump_yaw_deg (_anchor_heading, round 8)
        self.n_anchor_refused = 0
        self.n_anchor = 0
        self.n_xy_holes = 0
        self.rows: dict = {}                 # state -> ticks

    def reset_track(self) -> None:
        """Cold: no anchor, no DVL history (the heading and its bias are
        kept — they are the gyro's, not the track's)."""
        # `anchored` = the tags have PLACED the track in the map (anchor_n
        # agreeing fixes, once per session) and it may carry the pose. Until
        # then the track is still re-anchored to every tag tick — it just
        # may not stand in for one.
        self.anchored = False
        self._anchor_ready = False
        self.p = np.zeros(2)                 # committed x, y at t_p
        self.t_p: float | None = None        # newest usable sample's t_valid
        self.v_last = None                   # its body velocity (3,)
        self.att_last = (0.0, 0.0)           # its roll, pitch
        self.alt_last = None                 # its DVL altitude
        self.t_usable_last: float | None = None
        self.n_usable = 0
        self.n_unusable = 0
        self.unusable_code = ""
        self._gap_code = ""                  # ...since the newest usable one
        # a sample DvlAid REFUSED ON EVIDENCE (a disagreement hold, a latch)
        # after t_p: the last good velocity is not extrapolated past it
        # (DvlBlend expires its own hold on the same codes)
        self._stop_t: float | None = None
        self._latched = ""                   # the DVL is latched OUT
        # a stretch of unknown motion after the anchor (_xy_why)
        self._xy_hole_t: float | None = None
        self._xy_hole_why = ""
        self.z_off = None                    # map z - Bar30 depth
        self._z_prev = None                  # (t_sp2, depth) for w
        self._z_rej_t0 = None                # a spike refused since
        # a Bar30 LEVEL change was taken (the 1 s hold) and the z offset has
        # not been re-taken on a fix since: the z jump test is off until it
        # is (round 6 — under ON the new level was flown as a held z 'jump'
        # and slewed back at 2 cm/s)
        self._z_retake = False
        # the fixes the pose took, (capture time, their own yaw), until the
        # gyro history covers them (_anchor_heading)
        self._yaw_q: deque = deque()
        self._yaw_last_tc: float | None = None   # ...the newest ever queued
        # EVERY fresh fix, (capture, its own yaw), until the gyro history
        # covers it — the yaw judgment (_yaw_innovation), any TAG FIX state
        self._yj_q: deque = deque()
        self._yaw_ev = None                  # the newest covered one
        self._yaw_judged = True              # this tick's yaw on a covered fix
        # the dead reckoning's heading LAG behind the tags at t_now, measured
        # on a tag tick (the state's yaw difference minus the judged one):
        # (t, lag rad) — a turn that started within the gyro's lateness, its
        # records not yet arrived (round 7)
        self._yaw_lag = None
        # the capture where an accepted tag LEVEL began (its first pending
        # fix): judged yaw on fixes before it still shows the old level, so
        # its difference from the state's is that change, not a lag
        self._level_t: float | None = None
        self._level_t0: float | None = None  # ...of the pending one
        # a SNAP moved the pose to a new tag level: the heading needs an
        # anchor on that level (a fix captured at or after this) before the
        # dead reckoning may carry it (round 7)
        self._relevel_t: float | None = None
        self._z_rej_v = None                 # the level a refusal refused
        self._blind_from_jump = False
        # a HELD JUMP's own reference: the first anchor_n of its
        # innovations and their median — what the tags' difference was when
        # the hold began, so the hold's DRIFT can be watched (round 8)
        self._hold_items: list = []
        self._hold_ref = None
        # ...kept while a held jump goes blind (the DR dropped out of it),
        # back when the DR returns as the jump it was (round 9)
        self._hold_keep = None
        self._e_now = None                   # this tick's judged innovation
        # ...and off / off_jump as they stood when it was taken (before this
        # tick's slew moved the track)
        self._off_e = np.zeros(4)
        self._offj_e = np.zeros(4)
        self._stops_now: set = set()         # _watch's stops on this tick
        self._div_now = False                # ...and the divergence's
        self._zh: deque = deque(maxlen=64)   # (t, depth) of the pinned source
        self.vz = 0.0                        # NED, low-passed
        self.state = ST_NO_ANCHOR
        # Why the pose was NOT the tag state on the last tick: "" (it was),
        # "start" (never output), "jump" / "reacquire" (tags in view, held
        # back), "dropout" (TAG FIX ON, no tag), "off", "lost" (no pose).
        self._carry = "start"
        self.off = np.zeros(4)               # correction: x, y, z, yaw
        # ...the part of it a held JUMP put in flight (the tags' level moved:
        # a place the tags may come back on with no drift); the rest is a
        # re-acquisition's — the error the pose is flying (round 9)
        self.off_jump = np.zeros(4)
        self._e_ref = None                   # innovation expected next
        self._pending: list = []             # innovations awaiting agreement
        self._pending_t: list = []           # ...the tick each one came in
        self._pending_raw: list = []         # ...and the raw tag pose then
        self._pending_off: list = []         # ...off then
        self._pending_offj: list = []        # ...off_jump then
        self._spd_hist: deque = deque()      # (t, DVL speed) while DR, 1 s
        self._pending_kind = ""              # "jump" | "reacquire" | "blind"
        # a dead-reckoned (DR / HELD) pose went out since the last tag one:
        # only then is a blind re-acquisition judged against _shown (after a
        # pure tag dropout with no DR, _shown IS the tag state)
        self._dr_shown = False
        self._last_fix_t = None              # newest fix capture time seen
        self.t_fix = None                    # newest accepted tag tick
        self.t_seen = None                   # newest tick with a fresh tag
        self.dist_m = 0.0                    # horizontal path since t_fix
        self.unobserved_m = 0.0              # gap motion not integrated
        self._t_ta: float | None = None      # the track's newest anchor
        self._div_t0 = None                  # divergence above abort since
        self._div_n = 0                      # ...new fixes over it since then
        self._div_last_t = None              # ...the newest of them
        self._div_faulted = False            # ...and already recorded
        self._watch_on: set = set()          # blind/speed/unaided stops raised
        self._gap_warn = ""                  # trust mode: the newest gap's
        self._gap_warn_t: float | None = None   # ...warning (until a re-anchor)
        self.n_gaps_trusted = 0              # ...gaps flown across, no hole
        self._lat = 0.0                      # newest usable sample's latency
        self._unobs_now = 0.0                # unobserved_m + the open gap's
        self._z_src = ""                     # depth source z_off was taken on
        self._z_anchored = False             # z_off re-taken on this anchor
        self._z_prev_src = ""                # ...and the one _z_prev holds
        self._prev_gyro = None
        self._prev_nu = None
        self.err_m = None
        self.err_yaw_deg = None
        self.err_z_m = None
        self.max_err_m = 0.0
        self.max_since_fix_s = 0.0
        self.max_dist_m = 0.0
        self.max_blind_s = 0.0
        # the newest pose OUTPUT (map frame), kept through ticks without one:
        # what the controller last saw, which a blind re-acquisition is
        # judged against (_accept_blind)
        self._shown = None
        self._shown_dz = None                # its z - the pinned depth then
        self._last_depth = None              # this tick's pinned depth

    # ================================================================ inputs
    def note_imu(self, imu) -> None:
        self.head.note_imu(imu)

    def note_dvl(self, t_valid: float, v_body, usable: bool, code: str = "",
                 roll: float | None = None, pitch: float | None = None,
                 altitude: float | None = None,
                 t_rx: float | None = None) -> None:
        """One DVL sample, as DvlAid judged it (``usable``; ``code`` = why
        not). ``roll``/``pitch`` = the attitude at its validity time;
        ``t_rx`` = when it reached the station (its delivery latency puts an
        OPEN gap on the same clock as a committed one, step())."""
        code = str(code or "")
        try:
            t = float(t_valid)
        except (TypeError, ValueError):
            t = float("nan")
        if not usable or v_body is None:
            self.n_unusable += 1
            self.unusable_code = code or "unusable"
            self._gap_code = self.unusable_code
            if (code == "latched" or code.startswith("dis_")) \
                    and math.isfinite(t) \
                    and (self.t_p is None or t > self.t_p) \
                    and self._stop_t is None:
                self._stop_t = t
            if code == "latched":
                self._latched = self.unusable_code
            return
        v = np.asarray(v_body, float).reshape(3)
        if not np.all(np.isfinite(v)) or not math.isfinite(t):
            self.n_unusable += 1
            self.unusable_code = "non-finite"
            self._gap_code = self.unusable_code
            return
        if self.t_p is not None and t <= self.t_p:
            return                           # out of order / duplicate
        phi = float(roll) if roll is not None else self.att_last[0]
        th = float(pitch) if pitch is not None else self.att_last[1]
        if not (math.isfinite(phi) and math.isfinite(th)):
            phi, th = self.att_last
        c = self.cfg
        if self.t_p is not None and self.v_last is not None:
            gap = t - self.t_p
            psi0 = self._psi(self.t_p)
            psi1 = self._psi(t)
            have_psi = (psi0 is not None and psi1 is not None
                        and math.isfinite(psi0) and math.isfinite(psi1))
            stop = (self._stop_t - self.t_p
                    if self._stop_t is not None and self._stop_t > self.t_p
                    else None)
            step = None
            hole = ""
            if gap <= c["max_interp_s"] and stop is None:
                if have_psi:
                    vw0 = rot_zyx(self.att_last[0], self.att_last[1],
                                  psi0) @ self.v_last
                    vw1 = rot_zyx(phi, th, psi1) @ v
                    step = 0.5 * (vw0[:2] + vw1[:2]) * gap
            else:
                # what the real-time output showed: the old velocity for
                # hold_s (or up to a refusal), then held (module doc, GAPS)
                shown = min(gap, c["hold_s"] if stop is None
                            else min(c["hold_s"], stop))
                if have_psi:
                    vw0 = rot_zyx(self.att_last[0], self.att_last[1],
                                  psi0) @ self.v_last
                    step = vw0[:2] * shown
            post, unobs = self._gap_share(t, have_psi)
            self.unobserved_m += unobs * self._v_charge()
            if post > c["lost_s"]:
                hole = f"a DVL gap of {post:.1f} s"
            elif not have_psi and post > 0.0:
                hole = ("no heading history over a DVL gap of "
                        f"{post:.1f} s")
            elif self.unobserved_m > c["max_unobserved_m"]:
                hole = (f"{self.unobserved_m * 100:.0f} cm of unintegrated "
                        f"motion")
            # (the hole is judged whether or not the heading history still
            # reaches back over the gap: a gap longer than the history — it
            # holds ~60 s — used to skip this, and the track resumed from the
            # pre-outage position with no hole; fix verification 2026-10-18.
            # A hole from BEFORE the newest anchor is no hole and does not
            # stand in the way of a new one — it used to, and blocked every
            # later hole until the next anchor.)
            if hole and self.anchored and c["trust_dvl"]:
                # TRUST MODE: no hole — the pose resumes from where the gap
                # found it; what the gap may have hidden is said and kept
                # (unobserved_m, the CSV's dvp_unobs_m), and a TAG FIX ONCE
                # with tags in view takes it out
                self._gap_warn = (f"dvl pose: {hole} — flown across (trust "
                                  f"mode): position may be off, TAG FIX ONCE "
                                  f"with tags in view to correct it")
                self._gap_warn_t = t
                self.n_gaps_trusted += 1
            elif hole and self.anchored and not self._hole_live():
                self._xy_hole_t = t
                self._xy_hole_why = hole
                self.n_xy_holes += 1
            if step is not None:
                self.p = self.p + step
                self.dist_m += float(np.hypot(step[0], step[1]))
        if t_rx is not None:
            try:
                lat = float(t_rx) - t
            except (TypeError, ValueError):
                lat = float("nan")
            if math.isfinite(lat):
                self._lat = min(0.5, max(0.0, lat))
        self.t_p = t
        self.v_last = v
        self.att_last = (phi, th)
        self.alt_last = None if altitude is None else float(altitude)
        self.t_usable_last = t
        self.n_usable += 1
        self._stop_t = None
        self._latched = ""
        self._gap_code = ""

    def _depth_at(self, t: float):
        """The pinned depth source at station time ``t`` (interpolated in
        its history; the newest past the newest stamp, as the Bar30 trails
        the fix capture by less than its interval; None before the oldest or
        with no history)."""
        h = self._zh
        if not h:
            return None
        t = float(t)
        if t >= h[-1][0]:
            # Past the newest stamp: carried at the heave. A Bar30 still
            # ARRIVING (this tick's depth fresh) whose stamps trail the fix
            # by a queued link's lateness is good for depth_stale_s past its
            # stamp — the 0.5 s of round 6 left 15.5-47.6 % of the fixes of
            # 10-13's queued segments with no Bar30 (30 % pooled over all 7
            # of its rov segments) [측정: data/20261017/
            # 1017_dvl_pose_feasibility/rules_replay/out_round8/evidence/
            # fix_z_coverage_pinned.txt, round6 rows], and their z fell back
            # to the bridged state z, a held z 'jump' that ended a replay run
            # (fix verification round 7).
            # One that stopped is no depth for a fix captured later.
            gap = t - h[-1][0]
            lim = (self.cfg["depth_stale_s"] if self._last_depth is not None
                   else 0.5)
            if gap > lim:
                return None
            return h[-1][1] + self.vz * gap
        if t < h[0][0]:
            return None
        for k in range(len(h) - 1, 0, -1):
            t0, d0 = h[k - 1]
            t1, d1 = h[k]
            if t0 <= t <= t1:
                w = (t - t0) / max(t1 - t0, 1e-9)
                return d0 + w * (d1 - d0)
        return None

    @staticmethod
    def _fix_yaw(fix):
        """(the fix's capture time, its own yaw) or (None, None)."""
        tc = _fin(getattr(fix, "t_capture", None)) if fix is not None else None
        R = getattr(fix, "R_ned_body", None) if fix is not None else None
        if tc is None or R is None:
            return None, None
        try:
            psi = yaw_of(R)
        except (TypeError, ValueError):
            return None, None
        if not math.isfinite(psi):
            return None, None
        return tc, psi

    def _queue_judge(self, fix) -> None:
        """A fresh fix (any TAG FIX state) into the yaw judgment queue."""
        tc, psi = self._fix_yaw(fix)
        if tc is None:
            return
        q = self._yj_q
        if not q or tc > q[-1][0] + 1e-9:
            q.append((tc, psi))
        while q and tc - q[0][0] > YAW_QUEUE_S:
            q.popleft()

    def _judge_pop(self) -> None:
        """The fixes the gyro history now covers: the newest is the yaw
        evidence (_yaw_ev). A fix across a gyro hole is none (its raw is
        another integral)."""
        hd = self.head
        q = self._yj_q
        while q and hd.t is not None and q[0][0] <= hd.t + 1e-9:
            tc, psi = q.popleft()
            if hd.hole_t is not None and tc <= hd.hole_t:
                continue
            self._yaw_ev = (tc, psi)

    def _yaw_innovation(self, t_now: float):
        """The yaw difference JUDGED on the newest fix the gyro history
        covers — its own yaw at its own capture against the dead reckoning's
        heading then (no extrapolation of either), the current fix's itself
        on a healthy link, one up to the link's lateness old on a queued
        one, never older than YAW_EVIDENCE_S. None: no such fix — the
        caller keeps the state's difference at t_now, a measured yaw with
        the queued rotation in it. (Round 6 put the EXPECTED difference in
        its place: every yaw check — the OFF divergence abort, the vouching
        behind the blind stop, the correction limit, the ON refusal — then
        read 0 on a queued link, and a 12-15 deg heading error flew with
        nothing to stop it in the round-7 review's synthetic scenarios
        [유도: 합성 시나리오, fix verification round 7].)"""
        ev = self._yaw_ev
        hd = self.head
        if ev is None or float(t_now) - ev[0] > YAW_EVIDENCE_S:
            return None
        if hd.hole_t is not None and ev[0] <= hd.hole_t:
            return None
        psi_dr = hd.heading(ev[0])
        if psi_dr is None or not math.isfinite(psi_dr):
            return None
        return wrap(ev[1] - psi_dr)

    def _fix_z(self, fix):
        """(fix z at its capture, the pinned depth then) or None."""
        if not self.z_from_fix or fix is None:
            return None
        p = getattr(fix, "p_ned", None)
        tc = getattr(fix, "t_capture", None)
        if p is None or tc is None or len(p) < 3:
            return None
        z = float(p[2])
        d = self._depth_at(float(tc))
        if d is None or not math.isfinite(z):
            return None
        return z, d

    def _gap_share(self, t_end: float, have_psi: bool = True):
        """The interval from the newest usable sample (t_p) to ``t_end`` (a
        committed gap, or the open one) as the TRACK sees it -> (its length
        after the track's newest anchor, the part of that the output did not
        integrate). Only what lies after the anchor counts: a stretch the
        tags re-anchored is not unobserved (fix verification 2026-10-18 —
        a tag-filled gap was charged to the next one's budget)."""
        c = self.cfg
        gap = float(t_end) - self.t_p
        stop = (self._stop_t - self.t_p
                if self._stop_t is not None and self._stop_t > self.t_p
                else None)
        if gap <= c["max_interp_s"] and stop is None and have_psi:
            unobs = 0.0                      # joined: nothing unobserved
            credited = gap
        else:
            credited = (min(gap, c["hold_s"] if stop is None
                            else min(c["hold_s"], stop)) if have_psi else 0.0)
        t0 = self.t_p
        ta = self._t_ta
        if ta is not None and ta > t0:
            credited = max(0.0, credited - (ta - t0))
            t0 = ta
        post = max(0.0, float(t_end) - t0)
        if gap <= c["max_interp_s"] and stop is None and have_psi:
            return post, 0.0
        unobs = max(0.0, post - credited)
        return post, unobs

    def _v_charge(self) -> float:
        """The speed a gap's unobserved time is charged at: the newest usable
        sample's horizontal speed, never below unobserved_v_floor_m_s."""
        v = 0.0
        if self.v_last is not None:
            v = float(math.hypot(self.v_last[0], self.v_last[1]))
        return max(v, self.cfg["unobserved_v_floor_m_s"])

    def _hole_live(self) -> bool:
        """A hole AFTER the track's newest anchor (one before it is spent)."""
        return (self._xy_hole_t is not None
                and (self._t_ta is None
                     or self._xy_hole_t > self._t_ta - 1e-9))

    def _psi(self, t: float):
        """Heading for integration: the anchored heading, or — before the
        first anchor — the gyro's raw (relative) one, so the track exists
        (in an arbitrary frame) before the tags place it."""
        if self.head.t_anchor is not None:
            return self.head.heading(t)
        return self.head.raw_at(t)

    # ============================================================ the switch
    def corr_m(self) -> float:
        return float(math.hypot(self.off[0], self.off[1]))

    def tag_refusal(self, on: bool, t_now: float, dvl_ok: bool = True,
                    dvl_why: str = "", engaged: bool = False) -> str:
        """Why the TAG FIX switch may NOT go this way ("" = it may).

        ON only adds information — except while ENGAGED in control with the
        pose already further from the tags than a correction may be: the
        re-acquisition would end the run, so the press is refused in words
        (disengage first: then it snaps). OFF needs a dead reckoning that
        can carry the vehicle NOW, and no correction still slewing in."""
        c = self.cfg
        if on:
            # (z is slewed whatever its size — _accept — so it refuses nothing)
            if (engaged and self.mode == "control" and self.err_m is not None
                    and (self.err_m > c["max_corr_m"]
                         or (self.err_yaw_deg or 0.0) > c["max_corr_yaw_deg"])):
                return (f"the pose is {self.err_m * 100:.0f} cm / "
                        f"{self.err_yaw_deg or 0.0:.1f} deg from the tags — "
                        f"more than a correction may be while engaged "
                        f"(max_corr_m / max_corr_yaw_deg; z is slewed), so ON "
                        f"would end the run: disengage first (it then "
                        f"snaps), or stay OFF")
            return ""
        if not self.anchored:
            return ("no anchor yet — keep TAG FIX ON with tags in view "
                    f"until {self.cfg['anchor_n']} fixes agree")
        left = []
        if self.corr_m() > 1e-3:
            left.append(f"{self.corr_m() * 100:.1f} cm")
        if abs(float(self.off[3])) > 1e-4:
            left.append(f"{math.degrees(abs(float(self.off[3]))):.2f} deg")
        if abs(float(self.off[2])) > 1e-3:
            left.append(f"{abs(float(self.off[2])) * 100:.1f} cm z")
        n_pend = self.pending_n(t_now)
        if left or n_pend:
            # (what is left, axis by axis: the z and yaw slews outlast the
            # xy one, and "0.0 cm left" read as no reason at all)
            return ("a tag correction is still being applied ("
                    + ", ".join(
                        ([" / ".join(left) + " left"] if left else [])
                        # (fixes so far, not "n/5 agree": they are the
                        # ones being compared, and 5 kept can disagree)
                        + ([f"{n_pend} fix{'es' if n_pend > 1 else ''} "
                            f"waiting to agree, {c['anchor_n']} must"]
                           if n_pend else []))
                    + ") — wait until the pose is the tag state")
        return self.dr_refusal(t_now, dvl_ok, dvl_why)

    def dr_refusal(self, t_now: float, dvl_ok: bool = True,
                   dvl_why: str = "") -> str:
        """Why the dead reckoning cannot carry the vehicle NOW ("" = it
        can): what OFF needs beyond its own rule (no correction in flight),
        and what a START that would dead-reckon needs. A START under TAG FIX
        ON abandons nothing in flight — a pending agreement or a slew goes on
        after it — so it is judged on this alone (fix verification
        2026-10-18)."""
        c = self.cfg
        if not self.anchored:
            return ("no anchor yet — keep TAG FIX ON with tags in view "
                    f"until {c['anchor_n']} fixes agree")
        if self.mode == "control" and not dvl_ok:
            return f"the DVL cannot carry the pose ({dvl_why or 'not usable'})"
        gap = None if self.t_usable_last is None \
            else float(t_now) - self.t_usable_last
        if gap is None or gap > c["hold_s"] or self._stop_t is not None:
            return ("no usable DVL sample"
                    + ("" if gap is None else f" for {gap:.2f} s")
                    + (f" ({self._gap_code})" if self._gap_code
                       else ""))
        why = self._xy_why() or self._heading_why(t_now)
        if why:
            return why
        if (self.mode == "control" and self.cfg["require_gyro_bias"]
                and not self.cfg["trust_dvl"]
                and self.head.bias_state != "ok"):
            b = self.head.bias_info()
            return (f"the gyro bias is not learnt yet ({b['span_s']:.0f}/"
                    f"{self.cfg['bias_min_span_s']:.0f} s, {b['n']}/"
                    f"{self.cfg['bias_min_n']} fixes of TAG FIX ON with "
                    f"tags in view)")
        return ""

    def set_tag_on(self, on: bool, t: float, engaged: bool,
                   auto: str = "") -> None:
        """The switch. ``auto``: a one-shot's own way back to OFF ("done" |
        "timeout"), recorded on the toggle."""
        on = bool(on)
        if on == self.tag_on:
            return
        self.tag_on = on
        self._hold_keep = None
        if not on:
            self.oneshot = None
        rec = {"t": float(t), "on": on, "engaged": bool(engaged)}
        if auto:
            rec["auto"] = str(auto)
        self.toggles.append(rec)
        # ON after OFF is a re-acquisition: the next fixes must agree first
        self._unpend()
        # (fixes the pose took before: their yaw is no bias pair or heading
        # anchor once OFF — the bias is frozen while OFF)
        self._yaw_q.clear()
        self.note, self.note_t = "", None

    def start_oneshot(self, t: float, engaged: bool) -> None:
        """TAG FIX pressed while OFF: ON until ONE correction is in, then the
        worker turns it OFF again (oneshot_due) — the operator's way to clear
        the drift now and then (2026-10-18). The tags returning after OFF are
        a re-acquisition, so the correction is the usual one: anchor_n fixes
        agree, then it is slewed (engaged) or snapped; one larger than
        max_corr_* while engaged in control ends the run, as any."""
        self.set_tag_on(True, t, engaged)
        if self.tag_on:
            self.oneshot = {"t0": float(t), "accepted": False}
            if self.toggles:
                self.toggles[-1]["oneshot"] = True

    def oneshot_due(self, t: float) -> str:
        """A one-shot TAG FIX that should go back to OFF now: "done" (its
        correction is accepted and slewed out) | "timeout" (none accepted in
        oneshot_timeout_s) | "" (not yet, or none in progress). The caller
        drops what is still pending (drop_pending: the one correction asked
        for is in, or no longer wanted — a glimpse's leftover fixes kept it
        ON for as long as the tags stayed away, a view that never agrees
        held it past its timeout: round 9) and turns it OFF once
        oneshot_refusal says nothing."""
        o = self.oneshot
        if o is None or not self.tag_on:
            return ""
        if o["accepted"]:
            if (self.corr_m() <= 1e-3 and abs(float(self.off[2])) <= 1e-3
                    and abs(float(self.off[3])) <= 1e-4):
                return "done"
            return ""
        if float(t) - o["t0"] >= self.cfg["oneshot_timeout_s"]:
            return "timeout"
        return ""

    def oneshot_refusal(self, t_now: float, dvl_ok: bool = True,
                        dvl_why: str = "") -> str:
        """Why a DUE one-shot may not go back to OFF ("" = it may, after
        drop_pending): OFF's refusal minus the agreement in progress — the
        one correction asked for is in (or no longer wanted). A correction
        still slewing and the dead reckoning's own ability still hold it."""
        if not self.anchored:
            return ("no anchor yet — keep TAG FIX ON with tags in view "
                    f"until {self.cfg['anchor_n']} fixes agree")
        if (self.corr_m() > 1e-3 or abs(float(self.off[3])) > 1e-4
                or abs(float(self.off[2])) > 1e-3):
            return "a tag correction is still being applied"
        return self.dr_refusal(t_now, dvl_ok, dvl_why)

    def drop_pending(self) -> None:
        """Forget an agreement in progress (a one-shot that is done or timed
        out: OFF must not wait for it, nor apply it later)."""
        self._unpend()
        self._hold_keep = None

    def say(self, note: str, t: float) -> None:
        """A refusal / remark for the panel, shown for a while (the chip's
        START-refusal rule) — never as a standing state."""
        self.note, self.note_t = str(note), float(t)

    def note_now(self, t: float, keep_s: float = 6.0) -> str:
        if self.note and self.note_t is not None \
                and float(t) - self.note_t <= keep_s:
            return self.note
        return ""

    # ================================================================== step
    def _heading_why(self, t_now: float) -> str:
        st = self.head.stale(t_now)
        name = "RAW_IMU gyro" if self.head.source == "raw_gyro" \
            else "ATTITUDE rate"
        if st is None:
            return f"dvl pose stale: no {name} yet (heading)"
        if st > self.cfg["gyro_stale_s"]:
            return f"dvl pose stale: no {name} for {st:.2f}s (heading)"
        hd = self.head
        if hd.unplaced:
            # (unknown is never fresh — fix verification round 6)
            return (f"dvl pose stale: the {name} cannot be placed on the "
                    f"vehicle clock (lateness unknown) — the heading cannot "
                    f"be carried (heading)")
        if hd.t_v is not None and hd.t is not None:
            # late on the VEHICLE clock; covered = what raw_at extrapolates
            # past the history's newest stamp (all of it for att_rate, whose
            # history is stamped on the vehicle clock too since round 6)
            late = float(t_now) - hd.t_v
            covered = min(max(0.0, float(t_now) - hd.t), GYRO_EXTRAP_S)
            rest = late - covered
            if rest > 0.0 and (
                    abs(hd.rate) * rest
                    > math.radians(self.cfg["gyro_late_deg"])
                    or rest > self.cfg["gyro_late_max_s"]):
                return (f"dvl pose stale: the {name} is {late:.2f}s late at "
                        f"{math.degrees(abs(hd.rate)):.0f} deg/s — "
                        f"the heading cannot be carried (heading)")
        if self.head.broken_since_anchor():
            return (f"dvl pose stale: {name} gap after the anchor (heading "
                    f"unknown — TAG FIX ON to re-anchor)")
        if self._relevel_t is not None:
            # (a snap moved the pose to a new tag level; the heading is on
            # the old one until a fix of the new level is covered — round 7)
            return ("dvl pose stale: the heading is not re-anchored on the "
                    "new tag level yet (heading)")
        lag = self._yaw_lag
        if (lag is not None and lag[1] > math.radians(self.cfg["gyro_late_deg"])
                and (hd.t is None or hd.t < lag[0] - 1e-9)
                and not self._pending):
            # The tags showed the heading behind them by more than the late
            # rule allows, and the gyro history has not reached that moment
            # yet: a turn that STARTED within the lateness — the newest rate
            # is still the old one, so the late rule is silent (10-13: the
            # dropout yaw step was p99 16.7, max 40 deg, the rule silent on
            # 291 of 640 ticks over 3 deg; round 7). Until it has arrived.
            # Not while an agreement is pending: a tag LEVEL change reads
            # the same for the lateness, and dropping the dead reckoning
            # there sent a held jump blind for the whole stale debounce.
            return (f"dvl pose stale: the heading lags the tags by "
                    f"{math.degrees(lag[1]):.0f} deg (a turn still in the "
                    f"gyro's queue) — the heading cannot be carried "
                    f"(heading)")
        return ""

    def _xy_why(self) -> str:
        """Why the TRACK cannot carry the pose ("" = it can)."""
        if self._latched:
            if self.tag_on:
                # TAG FIX ON: the tags are the pose whenever they are fresh,
                # and one late frame is not a lost localizer — debounced like
                # the station without --dvl_pose (fix verification
                # 2026-10-18: this ended the run on a single late frame)
                return (f"dvl pose stale: the DVL is latched OUT "
                        f"({self._latched}) — no dead reckoning across a tag "
                        f"dropout")
            # OFF: not 'stale' — DvlAid has EVIDENCE the DVL is wrong, a
            # latch cannot clear while engaged, and OFF has nothing else to
            # fly on: waiting gains nothing
            return (f"dvl pose: the DVL is latched OUT ({self._latched}) — "
                    f"the dead reckoning cannot go on")
        if self._hole_live():
            return (f"dvl pose stale: {self._xy_hole_why} after the anchor "
                    f"(position unknown — TAG FIX ON to re-anchor)")
        return ""

    def _depth(self, imu, t_now: float):
        """-> (depth m positive down, stamp for its finite difference, src)
        from the Bar30, or the GPI-derived depth when the backend has no
        SCALED_PRESSURE2 field; (None, None, why) when neither is fresh.
        The source z_off was taken on is asked FIRST (it used to be the
        first fresh one, so a z pinned on the GPI stayed invalid for good
        once the Bar30 came back — fix verification 2026-10-18). Freshness
        is the record's arrival; the stamp handed on for w is its own sample
        moment on the vehicle clock when the backend could place it (bursts
        of queued records arrive ~1 ms apart and made a 0.6 m/s heave)."""
        lim = self.cfg["depth_stale_s"]
        srcs = (("depth_sp2_m", "t_sp2", "t_sp2_v", "sp2"),
                ("depth_m", "t_baro", None, "gpi"))
        if self._z_src == "gpi":
            srcs = srcs[::-1]
        for dk, tk, tvk, src in srcs:
            d, ts = getattr(imu, dk, None), getattr(imu, tk, None)
            if d is None or ts is None:
                continue
            d, ts = float(d), float(ts)
            if math.isfinite(d) and math.isfinite(ts) and t_now - ts <= lim:
                tv = getattr(imu, tvk, None) if tvk else None
                tv = (float(tv) if tv is not None and math.isfinite(float(tv))
                      else ts)
                return d, tv, src
        return None, None, "pressure depth stale"

    def _dr_xy(self, t_now: float):
        """-> (x, y at t_now, held: bool, gap s) from the committed track."""
        if self.t_p is None or self.v_last is None:
            return None, True, None
        gap = max(0.0, float(t_now) - self.t_p)
        lim = self.cfg["hold_s"]
        if self._stop_t is not None and self._stop_t > self.t_p:
            lim = min(lim, self._stop_t - self.t_p)
        ext = min(gap, lim)
        psi = self._psi(self.t_p)
        if psi is None or not math.isfinite(psi):
            return None, True, gap
        vw = rot_zyx(self.att_last[0], self.att_last[1], psi) @ self.v_last
        return self.p + vw[:2] * ext, gap > lim, gap

    def _anchor_to(self, eta, t_now: float, depth, zsrc: str = "",
                   fix=None, qok: bool = False) -> bool:
        """Re-anchor the track, the heading and the z offset so the pose at
        ``t_now`` is ``eta`` (map NED). Clears a track hole: the tags have
        placed the position again."""
        # THE HEADING at a fix's own capture, once the gyro history reaches
        # it (_anchor_heading) — not at t_now through the extrapolation
        if (not self._anchor_heading(eta, t_now, qok)
                and self.head.t_anchor is None):
            return False                     # (the first one waits)
        xy, _held, _gap = self._dr_xy(t_now)
        target = np.asarray(eta[:2], float)
        if xy is not None and np.all(np.isfinite(xy)) \
                and np.all(np.isfinite(self.p)):
            self.p = self.p + (target - xy)
        elif xy is not None and np.all(np.isfinite(xy - self.p)):
            self.p = target - (xy - self.p)
        else:
            # no DVL yet (or a track poisoned): the track starts here
            self.p = target.copy()
        self._z_anchored = False
        fz = self._fix_z(fix)
        if fz is not None:
            # (the pose is tag + off: the offset carries off's z too; the
            # depth it pairs with is the HISTORY's source)
            self.z_off = fz[0] + float(self.off[2]) - fz[1]
            self._z_src = self._z_prev_src or self._z_src
            self._z_anchored = True
            self._z_retake = False           # (a Bar30 level change: taken)
        elif depth is not None and (not self.z_from_fix
                                    or self.z_off is None):
            self.z_off = float(eta[2]) - float(depth)
            self._z_src = str(zsrc or "")
            self._z_anchored = True
            # (re-taken on the state: z_source pressure, where no fix z is
            # judged — a Bar30 level change is taken here too, or the z jump
            # test stayed off for good; round 6 self-review)
            self._z_retake = False
        self._xy_hole_t, self._xy_hole_why = None, ""
        self.unobserved_m = 0.0
        self._unobs_now = 0.0
        self._gap_warn, self._gap_warn_t = "", None    # (trust: re-anchored)
        self._t_ta = float(t_now)
        if (self._anchor_ready and not self.anchored and self.z_off is not None
                and self.head.t_anchor is not None):
            self.n_anchor += 1
            self.anchored = True
            self._e_ref = np.zeros(4)
            # (the anchoring branch's raw tag POSES are no innovations: left
            # pending they held the tags on the first dead-reckoned tick or
            # were taken as one — a 141 cm 'jump' record [측정: data/
            # 20261017/1017_dvl_pose_feasibility/rules_replay/out_round6/
            # on_pass2.json, 1017_220707/mpc_221744 err_m 1.4116]; round 7)
            self._unpend()
        return True

    def _queue_yaw(self, fix) -> bool:
        """A fix the pose took (TAG FIX ON, the pose its tag state): its own
        yaw at its own capture time, held until the gyro history covers that
        moment. False: no usable capture time / yaw on this fix."""
        tc, psi = self._fix_yaw(fix)
        if tc is None:
            return False
        q = self._yaw_q
        if self._yaw_last_tc is None or tc > self._yaw_last_tc + 1e-9:
            # (with the correction's yaw IN FORCE then: the pose this fix
            # was — anchored with a later one, a correction accepted in
            # between put its whole size into the heading; round 7)
            q.append((tc, psi, float(self.off[3])))
            self._yaw_last_tc = tc
        while q and tc - q[0][0] > YAW_QUEUE_S:
            q.popleft()
        return True

    def _anchor_heading(self, eta, t_now: float, qok: bool) -> bool:
        """The heading's anchor on a tag tick: the newest queued fix the gyro
        history already covers — its own yaw at its own capture, against the
        raw interpolated there, exact whatever the link's latency, with the
        correction's yaw on top. Fixes the history has not reached stay
        queued: anchoring at t_now through the extrapolation put the rotation
        still in the queue into the offset for good (fix verification round
        6 — on 10-13's queued link the raw the anchors used was off by p99
        20 deg, max 53, against the raw the records later showed). The bias
        pairs come from the same covered fixes, in order. A fix across a
        gyro hole is no anchor (its raw is another integral). A fix without
        a capture time or an attitude (``qok`` False): at t_now on the tag
        state, as before. -> True when anchored on this tick."""
        hd = self.head
        q = self._yaw_q
        if not qok:
            return hd.anchor(t_now, float(eta[5]))
        best = None
        # (Judged before it may anchor: the pose this fix was, against the
        # heading the earlier anchors give at its capture — further apart
        # than the tag gate calls one level and it is no anchor. One tag-yaw
        # outlier passed as the pose before the gyro covered it, became the
        # heading once it did, and every clean fix after it read as a jump,
        # slewed, re-anchored: 10-20 deg flown with no fault on a queued link
        # in a synthetic test and an outlier injected into 10-13's records,
        # and one ~8 deg excursion of ~6 s in 5 of 276 injections into
        # 10-17's — round 8, fix verification wf_21856d6c-69a, estimator F1
        # [측정: 주입 시험, …/rules_replay/out_round8/evidence/
        # wf_21856d6c-69a_result.json]. Free while there is no heading to
        # judge by: none yet, one across a gyro hole, or a snap's new level.)
        free = (hd.t_anchor is None or hd.broken_since_anchor()
                or self._relevel_t is not None)
        lim = math.radians(self.cfg["jump_yaw_deg"])
        while q and hd.t is not None and q[0][0] <= hd.t + 1e-9:
            tc, psi, off3 = q.popleft()
            if hd.hole_t is not None and tc <= hd.hole_t:
                continue
            if hd.raw_at(tc) is None:
                continue
            hd.note_tag_yaw(tc, psi)
            if self._relevel_t is not None and tc < self._relevel_t - 1e-9:
                continue                     # (the old level: no anchor)
            if not free:
                h_tc = hd.heading(tc)
                if (h_tc is not None and math.isfinite(h_tc)
                        and abs(wrap(psi + off3 - h_tc)) > lim):
                    self.n_anchor_refused += 1
                    continue
            best = (tc, psi, off3)
        if best is None:
            return False
        ok = hd.anchor(best[0], wrap(best[1] + best[2]))
        if ok and self._relevel_t is not None:
            self._relevel_t = None
        return ok

    def step(self, t_now: float, imu, tag_meas, fix, engaged: bool,
             dt: float):
        """-> (meas | None, health) for this tick, in the MAP frame.

        ``tag_meas`` = the assembler's state this tick (None = no fresh
        fix); ``fix`` = the NavFix it came from (its n_tags, capture time
        and own yaw are what a correction / the bias fit is judged on)."""
        c = self.cfg
        t_now = float(t_now)
        h = {"ok": False, "why": "", "state": "", "tag_on": self.tag_on,
             "mode": self.mode, "anchored": self.anchored, "fault": "",
             "warn": "", "err_m": None, "err_yaw_deg": None, "err_z_m": None,
             "since_fix_s": None, "dist_m": None, "gap_s": None,
             "corr_m": None, "unobserved_m": self.unobserved_m,
             "blind_s": None, "z_src": "", "alt_m": self.alt_last,
             "heading": self.head.source}
        bi = self.head.bias_info()
        h["bias_state"] = bi["state"]
        h["bias_dps"] = bi["in_use_dps"] if self.anchored else bi["dps"]
        if imu is None or imu.roll is None or imu.p is None:
            return self._none(h, "no vehicle imu")
        t_att = imu.t_att if imu.t_att is not None else imu.stamp
        if t_now - float(t_att) > self.imu_stale_s:
            return self._none(h, f"imu stale ({t_now - float(t_att):.2f}s)")
        phi, th = float(imu.roll), float(imu.pitch)
        gyro = np.array([float(imu.p), float(imu.q), float(imu.r)])
        if not (math.isfinite(phi) and math.isfinite(th)
                and np.all(np.isfinite(gyro))):
            return self._none(h, "imu stale (a non-finite attitude)")
        depth, t_d, zsrc = self._depth(imu, t_now)
        h["z_src"] = zsrc if depth is not None else ""
        depth = self._note_depth(depth, t_d, zsrc)
        # (the depth this tick USES: a refused spike is not it — _shown_dz
        # paired the shown z with the spike, fix verification round 6)
        self._last_depth = depth
        # the dead reckoning's z is pinned to the source z_off was taken on:
        # a per-tick switch (Bar30 -> GPI) would step z by the offset between
        # the two, which nothing measures (fix verification 2026-10-18)
        depth_dr = depth
        if (depth is not None and self._z_src and zsrc != self._z_src):
            depth_dr = None

        n_tags = int(getattr(fix, "n_tags", 0) or 0) if fix is not None else 0
        tag_ok = (tag_meas is not None and n_tags >= c["min_tags"]
                  and np.all(np.isfinite(np.asarray(tag_meas["eta"], float))))
        new_fix = False
        if tag_ok:
            if fix is not None:
                tc = float(fix.t_capture)
                new_fix = (self._last_fix_t is None
                           or tc > self._last_fix_t + 1e-6)
                if new_fix:
                    self._last_fix_t = tc
        if tag_ok and new_fix and fix is not None:
            self._queue_judge(fix)
        self._judge_pop()
        if not self.tag_on:
            self._unpend()
        elif not tag_ok and self._pending_kind == "blind":
            # a blind re-acquisition that came from a JUMP is one no more:
            # the tags were not in view all along (round 6) — nor its drift
            # reference
            self._blind_from_jump = False
            self._hold_keep = None
        elif not tag_ok and self._pending_kind == "jump":
            # A held jump does not survive a dropout AS A JUMP: the pose is
            # carried now, so the tags coming back are a re-acquisition, never
            # the jump's "back at the old level" pass-through (review
            # 2026-10-18). The fixes collected so far stay — a flickering
            # view must be able to agree across its gaps (agree_window_s).
            # (Nor its drift reference, or the divergence count of it.)
            self._pending_kind = "reacquire"
            self._hold_items, self._hold_ref = [], None
            self._div_t0, self._div_n, self._div_faulted = None, 0, False
        xy, held, gap = self._dr_xy(t_now)
        h["gap_s"] = gap
        # THE OPEN GAP'S SHARE of the unobserved budget, on the committed
        # clock (the sample that will close it is _lat late): the budget is
        # judged now, not only when that sample arrives (fix verification
        # 2026-10-18 — a HELD pose flew up to lost_s - hold_s past it)
        self._unobs_now = self.unobserved_m
        if gap is not None and self.t_p is not None and self.v_last is not None:
            _post, unobs = self._gap_share(t_now - self._lat)
            self._unobs_now += unobs * self._v_charge()
        h["unobserved_m"] = self._unobs_now
        hw = self._heading_why(t_now) or self._xy_why()
        dr_valid = (self.anchored and xy is not None and not hw
                    and gap is not None and gap <= c["lost_s"])
        psi_dr = self.head.heading(t_now) if self.anchored else None
        if psi_dr is None or not math.isfinite(psi_dr):
            dr_valid = False
        eta_dr = None
        if dr_valid and depth_dr is not None and self.z_off is not None:
            eta_dr = np.array([xy[0], xy[1], depth_dr + self.z_off, phi, th,
                               psi_dr])
            if not np.all(np.isfinite(eta_dr)):
                eta_dr, dr_valid = None, False
        elif dr_valid:
            dr_valid = False
            hw = hw or ("pressure depth stale" if depth is None else
                        f"pressure depth stale: the depth source changed "
                        f"({self._z_src} -> {zsrc}) since the anchor")

        # ---- the tag side: the innovation, gated
        use_tag = False
        self._e_now = None
        if tag_ok:
            eta_t = np.asarray(tag_meas["eta"], float)
            if eta_dr is not None:
                e = np.array([eta_t[0] - eta_dr[0], eta_t[1] - eta_dr[1],
                              eta_t[2] - eta_dr[2],
                              wrap(eta_t[5] - eta_dr[5])])
                # (the state's difference: what a correction must take up for
                # the pose to stay where it was on the accept tick)
                e_state = e.copy()
                fz = self._fix_z(fix)
                if fz is not None and self.z_off is not None:
                    # z on the FIX at its capture against the Bar30 then
                    # (the bridged state's z carries the tag FD heave x age)
                    e[2] = fz[0] - (fz[1] + self.z_off)
                # ...and yaw on the newest FIX the gyro history covers, at
                # its capture against the dead reckoning's heading then (no
                # extrapolation of either): the state's yaw difference at
                # t_now lacks the rotation still in a queued link, and that
                # read as a yaw 'jump' held against the right tags — 13 deg
                # on 10-13 (fix verification round 6). With no covered fix
                # in YAW_EVIDENCE_S, the state's difference stands: measured,
                # if late — never the expected one (round 7).
                ey = self._yaw_innovation(t_now)
                judged = ey is not None
                if judged:
                    # (what the state's difference at t_now has beyond the
                    # judged one is the rotation the gyro history has not
                    # delivered yet: the heading's lag behind the tags now —
                    # unless the judged fix predates a tag level accepted
                    # since, whose change that difference then is)
                    if (self._level_t is not None
                            and self._yaw_ev[0] < self._level_t - 1e-9):
                        self._yaw_lag = None
                    else:
                        self._level_t = None
                        self._yaw_lag = (t_now,
                                         abs(wrap(float(e_state[3]) - ey)))
                    e[3] = ey
                self._yaw_judged = judged
                h["yaw_judged"] = judged
                self._e_now = e.copy()
                self._off_e = self.off.copy()
                self._offj_e = self.off_jump.copy()
                self.err_m = float(math.hypot(e[0], e[1]))
                self.err_yaw_deg = math.degrees(abs(e[3]))
                self.err_z_m = float(e[2])
                h["err_m"], h["err_yaw_deg"] = self.err_m, self.err_yaw_deg
                h["err_z_m"] = self.err_z_m
                self.max_err_m = max(self.max_err_m, self.err_m)
            else:
                e = e_state = None
                self.err_m = self.err_yaw_deg = self.err_z_m = None
            if self.tag_on:
                use_tag = self._gate(e, new_fix, eta_t, t_now, engaged, h,
                                     fix=fix, e_state=e_state)
        else:
            self.err_m = self.err_yaw_deg = self.err_z_m = None
        # BLIND = time since the tags last VOUCHED for the pose: they were it,
        # or they were compared with it within the abort limits. A fix that
        # only showed the pose to be wrong, or that is still waiting for its
        # agreement beyond them, vouches for nothing — glimpses of a few
        # fixes every 10-20 s used to refresh this and kept every stop away
        # while the pose walked metres off (fix verification 2026-10-18).
        if use_tag or (tag_ok and self._vouches()):
            self.t_seen = t_now
        blind = None if self.t_seen is None else t_now - self.t_seen
        h["blind_s"] = blind

        if use_tag:
            # the pose IS the tag state (+ a correction slewing out)
            if not engaged:
                self.off[:] = 0.0
                self.off_jump[:] = 0.0
            eta = np.asarray(tag_meas["eta"], float).copy()
            eta[0] += self.off[0]
            eta[1] += self.off[1]
            eta[2] += self.off[2]
            eta[5] = wrap(eta[5] + self.off[3])
            nu = np.asarray(tag_meas["nu"], float).copy()
            nudot = np.asarray(tag_meas["nudot"], float).copy()
            corr = self.corr_m()
            state = ST_BLEND if (corr > 1e-4 or abs(self.off[2]) > 1e-4
                                 or abs(self.off[3]) > 1e-5) else ST_TAG
            qok = self._queue_yaw(fix)
            if not self._anchor_to(eta, t_now, depth, zsrc, fix=fix, qok=qok):
                h["warn"] = h["warn"] or (
                    self._heading_why(t_now)
                    or "no gyro history over the fixes yet: the dead "
                       "reckoning cannot be anchored")
            self.t_fix = t_now
            self.dist_m = 0.0
            self._div_t0, self._div_n, self._div_faulted = None, 0, False
            self._spd_hist.clear()
            z_before = float(self.off[2])
            self._decay(engaged, dt)
            if not self._z_anchored and self.z_off is not None:
                # no depth to re-take the offset on this tick: it follows
                # off's z as the slew goes on, or the dead reckoning would
                # step by what off gave up (fix verification 2026-10-18)
                self.z_off += float(self.off[2]) - z_before
            h["corr_m"] = corr
            self._carry = ""
            self._dr_shown = False
            return self._out_ok(h, state, eta, nu, nudot, gyro, t_now)

        # ---- dead reckoning (or nothing)
        carry = ("off" if not self.tag_on else
                 (self._pending_kind or "reacquire") if tag_ok else "dropout")
        if not self.anchored:
            return self._none(h, ("dvl pose: no anchor yet — TAG FIX ON with "
                                  "tags in view"), ST_NO_ANCHOR)
        if (dr_valid and self._unobs_now > c["max_unobserved_m"]
                and not self._hole_live() and not c["trust_dvl"]):
            # the open gap has used up the budget: a hole now, on the tick
            # the pose would have dead-reckoned through it
            self._xy_hole_t = t_now
            self._xy_hole_why = (f"{self._unobs_now * 100:.0f} cm of "
                                 f"unintegrated motion (a DVL gap still open"
                                 + (f", {self._gap_code}"
                                    if self._gap_code else "") + ")")
            self.n_xy_holes += 1
            dr_valid, hw = False, self._xy_why()
        if not dr_valid:
            if hw:
                return self._none(h, hw, ST_LOST)
            return self._none(
                h, (f"dvl pose stale: no usable DVL for "
                    f"{gap if gap is not None else float('inf'):.2f}s"
                    + (f" ({self._gap_code})" if self._gap_code
                       else "")), ST_LOST)
        # (No z split carried over from the tags: the first dead-reckoned
        # tick's z is the Bar30 on the FIX's offset — the truth the bridged
        # tag z drifted from — and steps there, as the station bridge steps
        # to its pressure depth. Carrying the split into off ratcheted the
        # flown z metres away with every held jump: fix verification
        # 2026-10-18, round 4.)
        # (the slew below moves the track: this tick's pose moves with it)
        d_slew = self._slew_track(engaged, dt)
        if d_slew is not None:
            # ...and the difference the tags will show on the level they were
            # last on moves with the track: carried, a held jump's 'back at
            # the old level' read the slew's progress as a new jump (round 9)
            if self._e_ref is not None:
                self._e_ref = -self.off.copy()
            eta_dr = eta_dr.copy()
            eta_dr[0] += d_slew[0]
            eta_dr[1] += d_slew[1]
            eta_dr[2] += d_slew[2]
            eta_dr[5] = wrap(eta_dr[5] + d_slew[3])
        R = rot_zyx(phi, th, float(eta_dr[5]))
        if held:
            u_v = np.zeros(2)
        else:
            u_v = self.v_last[:2].copy()
        # w from the Bar30: the third row of v_ned = R nu, solved for w
        if abs(R[2, 2]) > 0.2:
            w = (self.vz - R[2, 0] * u_v[0] - R[2, 1] * u_v[1]) / R[2, 2]
        else:
            w = 0.0
        nu_lin = np.array([u_v[0], u_v[1], w])
        nu = np.concatenate([nu_lin, gyro])
        nudot = self._nudot(R, nu_lin, gyro, imu, dt)
        since = None if self.t_fix is None else t_now - self.t_fix
        h["since_fix_s"], h["dist_m"] = since, self.dist_m
        if since is not None:
            self.max_since_fix_s = max(self.max_since_fix_s, since)
        if blind is not None:
            self.max_blind_s = max(self.max_blind_s, blind)
        self.max_dist_m = max(self.max_dist_m, self.dist_m)
        spd = None
        if not held:
            vw = rot_zyx(self.att_last[0], self.att_last[1],
                         float(eta_dr[5])) @ self.v_last
            self._spd_hist.append((t_now, float(math.hypot(vw[0], vw[1]))))
        while self._spd_hist and t_now - self._spd_hist[0][0] > 1.0:
            self._spd_hist.popleft()
        if (len(self._spd_hist) >= 2
                and self._spd_hist[-1][0] - self._spd_hist[0][0] >= 0.5):
            # over a second, not one sample: an RL_PWM run's oscillation read
            # 0.31-0.36 m/s for a sample with the DVL right (fix verification
            # 2026-10-18); a false lock lasts
            spd = float(np.mean([v for _t, v in self._spd_hist]))
        if (self._spd_hist and c["max_dr_speed_m_s"] is not None
                and self._spd_hist[-1][1] > 2.0 * c["max_dr_speed_m_s"]):
            # ...but a reading twice the cap is a stop at once: the mean
            # would integrate ~0.3 m of a false lock before it spoke
            spd = float(self._spd_hist[-1][1])
        self._watch(h, tag_ok, new_fix, t_now, since, blind, spd, engaged)
        lo, hi = c["alt_band_m"]
        if self.alt_last is not None and not (lo <= self.alt_last <= hi):
            h["warn"] = h["warn"] or (f"DVL altitude {self.alt_last:.2f} m "
                                      f"outside {lo:.2f}-{hi:.2f} m")
        out = self._out_ok(h, ST_HELD if held else ST_DR, eta_dr, nu, nudot,
                           gyro, t_now)
        self._carry = carry
        return out

    # ------------------------------------------------------------- pieces
    def _gate(self, e, new_fix: bool, eta_t, t_now: float, engaged: bool,
              h, fix=None, e_state=None) -> bool:
        """TAG FIX ON with a fresh tag state: is THIS tick's tag state the
        pose? Records a correction when a new level is accepted."""
        c = self.cfg
        carried = self._carry
        if e is None:
            if not self.anchored:
                # nothing placed yet: the tag state passes through — the
                # state without --dvl_pose — and the first anchor_n agreeing
                # fixes PLACE the track (it is re-anchored to every tag tick
                # anyway; this only decides when it may stand in for one)
                if self._anchor_ready:
                    return True              # (agreed: waits for the heading)
                if new_fix:
                    self._pend(eta_t, t_now)
                if self._agree_raw():
                    self._anchor_ready = True
                    self._unpend()
                else:
                    h["warn"] = (f"anchoring: {len(self._pending)}/"
                                 f"{c['anchor_n']} fixes agree")
                return True
            if carried and carried != "start" and self._dr_shown:
                # The pose was CARRIED by dead reckoning and the tags are
                # back with no dead reckoning to judge them by (the DVL or
                # the gyro is gone since): not passed through — judged
                # against what the controller last saw (review 2026-10-18:
                # this used to step the whole drift in, unslewed and
                # unbounded). After a pure tag dropout with no dead reckoning
                # shown, the last pose WAS the tag state and the tags pass
                # through as without --dvl_pose (fix verification: waiting
                # for anchor_n fixes there ended runs the station survives).
                return self._accept_blind(eta_t, new_fix, t_now, engaged, h,
                                          fix=fix)
            # the DVL gone, the tags in view: the tag state (+ a correction
            # still slewing) passes through and the track is re-anchored to
            # it, so the next difference is -off, not 0 (fix verification:
            # 0 read a slew's remainder as a jump when the DVL came back)
            # (disengaged, off is zeroed this very tick: the reference with it)
            self._e_ref = -self.off.copy() if engaged else np.zeros(4)
            return True
        was_jump = False
        if self._pending_kind == "blind":
            # a blind re-acquisition (raw tag POSES) was under way and the
            # dead reckoning is back: start over on innovations — as the JUMP
            # it was, if it came from one with the tags in view all along
            # (fix verification round 6: it restarted as a re-acquisition,
            # and a jump bigger than max_corr ended the run)
            was_jump = bool(self._blind_from_jump)
            keep = self._hold_keep
            self._unpend()
            self._hold_keep = None
            if was_jump:
                carried = "jump"
                if keep is not None:
                    # (the hold goes on: its drift since it BEGAN, not since
                    # the DR came back — reset, one DR-invalid tick a hold
                    # forgave all the drift so far: round 9)
                    self._hold_items, self._hold_ref = keep
        reacq = carried in ("dropout", "off", "lost", "reacquire")
        er = self._e_ref
        jump = (er is not None and (
            math.hypot(e[0] - er[0], e[1] - er[1]) > c["jump_m"]
            # (z not while a Bar30 level change waits for its re-take)
            or (abs(float(e[2] - er[2])) > c["jump_z_m"]
                and not self._z_retake)
            or abs(wrap(e[3] - er[3])) > math.radians(c["jump_yaw_deg"])))
        if not self._pending:
            if not reacq and not jump:
                # (the dead reckoning is re-anchored to this tick's pose, so
                # the next difference is -off — not this one: that made a
                # sub-gate glitch's return a 2x 'jump' and kept the glitch
                # level, round after round; fix verification 2026-10-18)
                # (disengaged, off is zeroed this very tick: the reference
                # with it — a disengage mid-slew read as a held jump, round 6)
                self._e_ref = (-self.off.copy() if engaged
                               else np.zeros(4))
                if self._pending_kind or was_jump:
                    # (a resumed hold, back: over — and a hold restored on
                    # THIS tick, whose kind the blind stretch's unpend had
                    # already cleared: its reference stayed behind, and the
                    # next jump, at any later time, was judged against that
                    # dead level — a clean 30 cm jump refused as 60 cm of
                    # drift with every sensor right: round 10, R10D-1)
                    self._unpend()
                return True
            # a re-acquisition (the pose was dead reckoning: a dropout, or
            # OFF) or a jump: the new level is believed only once anchor_n
            # fixes agree on it
            fresh = not was_jump and self._pending_kind != "jump"
            self._pending_kind = "reacquire" if reacq else "jump"
            if fresh:
                # (once per hold: a resumed one — the DR back from a blind
                # stretch, its first tick with no fix — is the same hold)
                if self._pending_kind == "jump":
                    self.n_jumps_held += 1
                    # (a new hold builds its own reference — never one left
                    # behind by an earlier hold: round 10)
                    self._hold_items, self._hold_ref = [], None
                self._level_t0 = self._last_fix_t   # (where this level began)
        elif self._pending_kind == "jump" and not jump:
            # the jump is over: the difference is back at the level it had
            # before it (the dead reckoning carried the frames in between,
            # with the tags in view all along — a dropout turns a held jump
            # into a re-acquisition, step())
            self._unpend()
            self._e_ref = -self.off.copy() if engaged else np.zeros(4)
            return True
        if new_fix:
            self._pend(e, t_now, raw=eta_t)
            if (self._pending_kind == "jump"
                    and len(self._hold_items) < c["anchor_n"]):
                # (what the difference was as the hold began, with the
                # correction in flight added back — the slew moves the track
                # and the difference together, and read as drift it ended
                # runs with a perfect DVL: round 9 — its drift is watched
                # against this, _watch, round 8)
                self._hold_items.append(self._plus(e, self.off))
                self._hold_ref = self._median_e(self._hold_items)
        if not self._agree():
            h["warn"] = (f"tags re-acquiring: {len(self._pending)}/"
                         f"{c['anchor_n']} agree"
                         if self._pending_kind == "reacquire" else
                         f"tag jump held ({len(self._pending)}/"
                         f"{c['anchor_n']} agree)")
            return False
        e_new = self._pending[-1]
        hold = booked = None
        if self._pending_kind == "jump":
            hold = self._hold_shift()
            if self._hold_ref is not None:
                m = self._median_e([self._plus(p, o) for p, o in
                                    zip(self._pending, self._pending_off)])
                booked = self._hold_booked(m, self._hold_ref)
        return self._accept(e_new, t_now, engaged, h, self._pending_kind,
                            e_cont=e_state, yaw_judged=self._yaw_judged,
                            hold_drift=hold, booked=booked)

    def _accept_blind(self, eta_t, new_fix: bool, t_now: float,
                      engaged: bool, h, fix=None) -> bool:
        """Tags back after a carried / lost pose, with no dead reckoning to
        compare: anchor_n agreeing tag poses, then the difference from the
        pose the controller LAST SAW (``_shown``) is slewed, snapped or
        refused exactly like any other correction."""
        c = self.cfg
        if self._pending_kind != "blind":
            # A re-acquisition the dead reckoning dropped out of goes on as a
            # blind one on the raw poses it already has, instead of starting
            # over — anchor_n more fixes of no pose outlasted the debounce
            # (fix verification 2026-10-18). A held JUMP stays one: the tags
            # were in view all along, whatever the dead reckoning did.
            from_jump = self._pending_kind == "jump"
            raws = [(r, tt) for r, tt in zip(self._pending_raw,
                                              self._pending_t)
                    if r is not None]
            keep = ((self._hold_items, self._hold_ref)
                    if from_jump and self._hold_ref is not None else None)
            self._unpend()
            self._hold_keep = keep
            self._pending_kind = "blind"
            self._blind_from_jump = from_jump
            for r, tt in raws:
                self._pend(r, tt, raw=r)
        if new_fix:
            self._pend(eta_t, t_now, raw=eta_t)
        if not self._agree_raw():
            h["warn"] = (f"tags re-acquiring with no dead reckoning to "
                         f"compare: {len(self._pending)}/{c['anchor_n']} "
                         f"agree")
            return False
        s = self._shown
        t = np.asarray(eta_t, float)
        e = (np.zeros(4) if s is None else
             np.array([t[0] - s[0], t[1] - s[1], t[2] - s[2],
                       wrap(t[5] - s[5])]))
        e_cont = e.copy()
        fz = self._fix_z(fix)
        if s is not None and fz is not None and self._shown_dz is not None:
            # z on the FIX against the z offset the shown pose stood at (the
            # bridged state's z carries the tag FD heave x age)
            e[2] = (fz[0] - fz[1]) - self._shown_dz
        kind = "jump" if getattr(self, "_blind_from_jump", False) else "blind"
        hold = booked = None
        if kind == "jump" and self._hold_keep is not None:
            x = self._plus(e, self.off)
            if (math.hypot(float(x[0]), float(x[1])) <= c["jump_m"]
                    and abs(wrap(float(x[3])))
                    <= math.radians(c["jump_yaw_deg"])):
                # The raw poses agreed on the level the jump LEFT (inside
                # the jump gate, x-y and yaw — the test _gate's 'the jump is
                # over' makes with the dead reckoning in): the tags back on
                # it while the dead reckoning was out. Bounded as the hold's
                # drift, the whole jump read as drift and ended a run with
                # every sensor right (round 10, R9-1). It is a blind
                # re-acquisition of the old level, judged as drift like any.
                kind = "blind"
            else:
                # (a held jump's drift is bounded at its accept, blind or
                # not — the shown pose is the dead reckoning it was held
                # against)
                ref = np.asarray(self._hold_keep[1], float)
                hold = self._hold_vs(x, ref)
                booked = self._hold_booked(x, ref)
        self._blind_from_jump = False
        self._unpend()
        self._hold_keep = None
        return self._accept(e, t_now, engaged, h, kind, e_cont=e_cont,
                            yaw_judged=True, hold_drift=hold, booked=booked)

    def _agree(self) -> bool:
        c = self.cfg
        p = self._pending
        if len(p) < c["anchor_n"]:
            return False
        ref = p[-1]
        lim_y = math.radians(c["jump_yaw_deg"])
        return all(math.hypot(q[0] - ref[0], q[1] - ref[1]) <= c["jump_m"]
                   and abs(float(q[2] - ref[2])) <= c["jump_z_m"]
                   and abs(wrap(q[3] - ref[3])) <= lim_y for q in p)

    def _agree_raw(self) -> bool:
        """Agreement of raw TAG POSES (no dead reckoning to compare with):
        anchor_n fixes within jump_m / jump_z_m / jump_yaw_deg of the newest
        — a vehicle at rest or slow."""
        c = self.cfg
        p = self._pending
        if len(p) < c["anchor_n"]:
            return False
        ref = p[-1]
        lim_y = math.radians(c["jump_yaw_deg"])
        return all(math.hypot(q[0] - ref[0], q[1] - ref[1]) <= c["jump_m"]
                   and abs(float(q[2] - ref[2])) <= c["jump_z_m"]
                   and abs(wrap(q[5] - ref[5])) <= lim_y for q in p)

    def _accept(self, e, t_now: float, engaged: bool, h, kind: str,
                e_cont=None, yaw_judged: bool = True,
                hold_drift=None, booked=None) -> bool:
        """A new tag level is believed: record how far the pose was off,
        then either slew it in (engaged, small enough), snap it (not
        engaged), or refuse it (engaged, too large — a FAULT in control; in
        shadow the refusal is recorded once and the pose snaps, so it keeps
        describing what a control run would have flown up to here).
        ``hold_drift`` (xy m, yaw deg): how far a held jump's difference
        moved while it was held — the dead reckoning's drift then.
        ``booked`` (x, y, z, yaw): the part of a held jump's measured shift
        beyond the jump gate (_hold_booked) — the dead reckoning's drift over
        the hold, kept out of the jump's part of the correction."""
        c = self.cfg
        e = np.asarray(e, float)
        on_target = False
        if (kind or "jump") in ("reacquire", "blind"):
            # (judged as DRIFT: against the level a correction still in
            # flight was slewing to as well as the pose shown — _drift_of)
            e, on_target = self._drift_of(e)
        err = float(math.hypot(e[0], e[1]))
        err_yaw = math.degrees(abs(e[3]))
        since = None if self.t_fix is None else t_now - self.t_fix
        rec = {"t": t_now, "err_m": round(err, 4),
               "err_yaw_deg": round(err_yaw, 2),
               "err_z_m": round(float(e[2]), 4),
               "since_fix_s": None if since is None else round(since, 2),
               "dist_m": round(self.dist_m, 3),
               "kind": kind or "jump", "engaged": bool(engaged)}
        if not yaw_judged:
            # (no fix the gyro history covers: yaw judged on the state at
            # t_now, the queued rotation in it — a blind correction compares
            # the state with the shown pose and is judged: round 7)
            rec["yaw_judged"] = False
        if on_target:
            rec["judged_on"] = "slew_target"
        hold_big = False
        if hold_drift is not None:
            rec["hold_drift_m"] = round(float(hold_drift[0]), 4)
            rec["hold_drift_yaw_deg"] = round(float(hold_drift[1]), 2)
            # a held JUMP is slewed whatever its size — but not the drift
            # the dead reckoning made while it was held (round 8, R7-1)
            hold_big = engaged and (hold_drift[0] > c["max_corr_m"]
                                    or hold_drift[1] > c["max_corr_yaw_deg"])
        big_xy = err > c["max_corr_m"] or err_yaw > c["max_corr_yaw_deg"]
        big_z = abs(float(e[2])) > c["max_corr_z_m"]
        # Only a correction AFTER dead reckoning (reacquire, blind) measures
        # the drift, and only in x, y, yaw: past max_corr it ends a control
        # run. A JUMP (the tags in view all along, their level moved and
        # held) is the tag stream's change — the station without --dvl_pose
        # steps to it, so slewing it is never worse; and z is the Bar30 under
        # the pose, so a z difference is the tags' z against it, not a drift.
        # Refusing those ended most of the real ON runs the pose ended (fix
        # verification 2026-10-18: 13 of the 111 ON runs of a replay of
        # 10-13..10-17, of 18 the pose ended [측정: data/20261017/
        # 1017_dvl_pose_feasibility/rules_replay/out_round3/on_summary.json];
        # 2 of 111 with the round-5 code, both gyro-late pivots, none a
        # correction [측정: …/out_round5/on_summary.txt]). Slewed, and said.
        too_big = engaged and big_xy and (kind or "jump") != "jump"
        if engaged and (big_xy or big_z) and not too_big and not hold_big:
            h["warn"] = (f"the tags differ by {err * 100:.0f} cm / "
                         f"{err_yaw:.1f} deg / {abs(float(e[2])) * 100:.0f} cm "
                         f"z ({kind or 'jump'}) — slewed in, not refused")
        self._unpend()
        self._hold_keep = None
        # (a lag measured on the way in was this level change — taken up by
        # the correction now, or refused: no lag left to wait out — and the
        # judged yaw shows the old level until it reaches the new one)
        self._yaw_lag = None
        self._level_t = (self._level_t0 if self._level_t0 is not None
                         else self._last_fix_t)
        self._level_t0 = None
        if too_big or hold_big:
            why = (f"dvl pose drifted {hold_drift[0] * 100:.0f} cm / "
                   f"{hold_drift[1]:.1f} deg while a tag jump was held (> "
                   f"{c['max_corr_m']:g} m / {c['max_corr_yaw_deg']:g} deg "
                   f"while engaged) — re-anchor while disengaged"
                   if hold_big and not too_big else
                   f"dvl pose and tags disagree by {err * 100:.0f} cm / "
                   f"{err_yaw:.1f} deg ({rec['kind']}, > "
                   f"{c['max_corr_m']:g} m / {c['max_corr_yaw_deg']:g} deg "
                   f"while engaged; z is slewed, never refused) — re-anchor "
                   f"while disengaged")
            h["fault"] = why
            self._fault(t_now, why)
            if self.mode == "control":
                rec["applied"] = "refused"
                self._log_corr(rec)
                return False
            rec["applied"] = "refused (shadow: snapped)"
            self._log_corr(rec)
            self.off[:] = 0.0
            self.off_jump[:] = 0.0
            self._e_ref = np.zeros(4)
            self._relevel_t = self._last_fix_t
            h["correction"] = rec
            self._oneshot_took()
            return True
        if engaged:
            # the pose stays where it was this tick (tag + off = the pose it
            # was showing: off takes up the STATE's difference, e_cont — the
            # fix-based z of e decides, it does not place), and the tag level
            # is reached at corr_rate_* as off decays to 0. The next
            # difference is -off, so that is the next jump's reference.
            cont = (np.asarray(e_cont, float) if e_cont is not None
                    else e)
            prev = self.off.copy()
            self.off = -cont.copy()
            self._e_ref = -self.off.copy()
            # What of it is a tag LEVEL change: a jump's all of it; a
            # re-acquisition judged on the slew's target keeps the jump part
            # still in flight (the rest of off is drift — the flown error,
            # and the next re-acquisition is judged with it in: round 9 —
            # discounting all of off let a DVL wrong faster than the slew
            # walk the pose 1.5 m off at 22 cm per re-acquisition, synthetic,
            # fix verification wf_f143a3fa-24b R8E-1); judged on the pose,
            # none (the level was left).
            if (kind or "jump") == "jump":
                # ...and of a jump, only what THIS accept changed (off -
                # off_prev: the correction already in flight, a
                # re-acquisition's leftover, stays what it was), less the dead
                # reckoning's drift over the hold as far as the fixes can tell
                # it from their own noise (booked). off_jump = off laundered
                # the leftover and the hold's drift into a level — the R8E-1
                # walk came back with a tag level change in each view (round
                # 10, R10E-1: 1.6-1.9 m in 120 s with no stop, synthetic);
                # the level taken from the hold's first fixes carried their
                # noise in as drift, misjudging the next re-acquisition by
                # 5-12 cm either way (round 10b, R11E-1, synthetic).
                inc = self.off - prev
                if booked is not None:
                    inc = inc + np.asarray(booked, float)
                self.off_jump = self.off_jump + inc
            elif not on_target:
                self.off_jump[:] = 0.0
            rec["applied"] = "slew"
            # what the slew MOVES (the state's difference) beside err_* (the
            # fix-based one that judged it): they part in z (round 6 — a
            # replay summary read err_z_m as the slewed z)
            rec["slew_m"] = round(float(math.hypot(cont[0], cont[1])), 4)
            rec["slew_yaw_deg"] = round(math.degrees(abs(float(cont[3]))), 2)
            rec["slew_z_m"] = round(float(cont[2]), 4)
        else:
            self.off[:] = 0.0
            self.off_jump[:] = 0.0
            self._e_ref = np.zeros(4)
            rec["applied"] = "snap"
            # (the pose is on the new level now; the heading is not until a
            # fix of it is covered — round 7)
            self._relevel_t = self._last_fix_t
        self._log_corr(rec)
        h["correction"] = rec
        self._oneshot_took()
        return True

    def _oneshot_took(self) -> None:
        """A correction was applied (slewed, snapped, or — shadow — snapped
        after a refusal): a one-shot TAG FIX in progress has its one."""
        if self.oneshot is not None:
            self.oneshot["accepted"] = True

    def _corr_score(self, e) -> tuple:
        """A difference in units of what a correction may be — (the larger
        of x-y and yaw: what ends a run; their sum: the tie-break when the
        larger one ties; z last) — which of two readings of it is the
        smaller. z as an equal partner in the max made a re-acquisition back
        on a jump's x-y level 'judged on the pose' whenever a map z
        difference dominated both readings, and the jump's x-y leftover was
        then judged as drift (round 10, R10E-2); a yaw both readings carry
        tied the max the same way (round 10b, R11E-2)."""
        c = self.cfg
        a = math.hypot(float(e[0]), float(e[1])) / c["max_corr_m"]
        b = math.degrees(abs(wrap(float(e[3])))) / c["max_corr_yaw_deg"]
        return (round(max(a, b), 9), round(a + b, 9),
                abs(float(e[2])) / c["max_corr_z_m"])

    def _drift_of(self, e, off_jump=None):
        """A re-acquisition's difference as the dead reckoning's DRIFT: the
        better of the two places the tags can come back to with no drift at
        all — the pose shown (e = 0), or the LEVEL a held jump's correction
        still in flight was slewing to (e = -off_jump). A tag dropout in the
        middle of a big jump's slew left its remainder in e, judged as
        drift: the run ended with nothing wrong (round 8, fix verification
        wf_21856d6c-69a, estimator F3 / safety R7-2). Only the JUMP part:
        a re-acquisition's own leftover is the error the pose is flying, and
        discounting it let the next one judge only the newest dropout's
        drift — a DVL wrong faster than the slew walked the pose 1.5 m off
        with no stop (round 9, wf_f143a3fa-24b R8E-1). -> (the difference to
        judge, True = the slew's target)."""
        e = np.asarray(e, float).copy()
        oj = self.off_jump if off_jump is None else np.asarray(off_jump, float)
        if not np.any(np.abs(oj) > 1e-9):
            return e, False
        alt = self._plus(e, oj)
        if self._corr_score(alt) < self._corr_score(e):
            return alt, True
        return e, False

    @staticmethod
    def _plus(e, off):
        """e + off (yaw wrapped): a difference with a correction added back."""
        x = np.asarray(e, float) + np.asarray(off, float)
        x[3] = wrap(float(x[3]))
        return x

    def _jerr(self):
        """(err_m, err_yaw_deg, err_z_m) as the stops judge them: under TAG
        FIX ON as the drift (_drift_of — a re-acquisition after dead
        reckoning is the only ON tick they judge), with err_z_m None (z is
        slewed, never a stop, under ON); under OFF the difference itself (no
        correction is in flight there)."""
        if not self.tag_on or self._e_now is None or self.err_m is None:
            return self.err_m, self.err_yaw_deg, self.err_z_m
        e, _ = self._drift_of(self._e_now, self._offj_e)
        # (no z under ON: z never ends an ON run — the dead reckoning's z IS
        # the Bar30, a z difference is the tags'. A re-acquisition's own z
        # slew left over, or a map z 25 cm off seen sparsely, ended runs on
        # this watch: round 10, R10Z-1. OFF keeps its z abort.)
        return (float(math.hypot(e[0], e[1])),
                math.degrees(abs(wrap(float(e[3])))), None)

    @staticmethod
    def _median_e(items):
        """The median innovation of ``items`` (yaw unwrapped about the
        first)."""
        a = np.asarray(items, float).reshape(-1, 4)
        m = np.median(a, axis=0)
        y0 = float(a[0, 3])
        m[3] = wrap(y0 + float(np.median([wrap(float(v) - y0)
                                          for v in a[:, 3]])))
        return m

    def _hold_drift(self):
        """(xy m, yaw deg, None) the tags' difference — with the correction
        in flight added back — has moved since a held jump began, on this
        tick's innovation; (None, None, None) without one. No z: the dead
        reckoning's z IS the Bar30, so a z move is the tags' (slewed, as
        any z), and a cluster's map z read as a z 'drift' ended runs with a
        right pose (round 9)."""
        e, ref = self._e_now, self._hold_ref
        if e is None or ref is None:
            return None, None, None
        xy, yaw = self._hold_vs(self._plus(e, self._off_e), ref)
        return xy, yaw, None

    def _hold_booked(self, x, ref):
        """A held jump's measured shift (x - ref, both e + off) beyond what
        its fixes can tell from their own noise — the jump gate (jump_m in
        x-y, jump_yaw_deg in yaw), the test that calls two fixes one level:
        the dead reckoning's drift over the hold, booked as drift (x, y, 0,
        yaw; z is never judged under ON)."""
        c = self.cfg
        x = np.asarray(x, float)
        ref = np.asarray(ref, float)
        d = np.zeros(4)
        dx, dy = float(x[0] - ref[0]), float(x[1] - ref[1])
        n = math.hypot(dx, dy)
        if n > c["jump_m"]:
            k = (n - c["jump_m"]) / n
            d[0], d[1] = dx * k, dy * k
        dyaw = wrap(float(x[3]) - float(ref[3]))
        ly = math.radians(c["jump_yaw_deg"])
        if abs(dyaw) > ly:
            d[3] = math.copysign(abs(dyaw) - ly, dyaw)
        return d

    @staticmethod
    def _hold_vs(x, ref):
        """(xy m, yaw deg) from a hold's reference to x (both e + off)."""
        x = np.asarray(x, float)
        ref = np.asarray(ref, float)
        return (float(math.hypot(x[0] - ref[0], x[1] - ref[1])),
                math.degrees(abs(wrap(float(x[3]) - float(ref[3])))))

    def _hold_shift(self):
        """(xy m, yaw deg) from a held jump's reference to the median of
        the fixes now agreeing on its level (each with the correction in
        flight when it came in) — the dead reckoning's drift over the hold,
        judged at the accept (None: no reference)."""
        if self._hold_ref is None or not self._pending:
            return None
        m = self._median_e([self._plus(p, o) for p, o in
                            zip(self._pending, self._pending_off)])
        return self._hold_vs(m, self._hold_ref)

    def _vouches(self) -> bool:
        """The compared fix is within the abort limits (max_corr_* where an
        abort is off)."""
        c = self.cfg
        if self.tag_on and self._pending_kind == "jump" \
                and self._hold_ref is not None:
            # (a held jump: the tags vouch for the dead reckoning when it has
            # kept to the level they held — round 8)
            em, ey, ez = self._hold_drift()
        else:
            em, ey, ez = self._jerr()
        if em is None:
            return False
        lx = c["abort_err_m"] if c["abort_err_m"] is not None \
            else c["max_corr_m"]
        ly = c["abort_err_yaw_deg"] if c["abort_err_yaw_deg"] is not None \
            else c["max_corr_yaw_deg"]
        lz = c["abort_err_z_m"] if c["abort_err_z_m"] is not None \
            else c["max_corr_z_m"]
        return (em < lx and (ey or 0.0) < ly and abs(ez or 0.0) < lz)

    def pending_n(self, t_now: float) -> int:
        """Fixes waiting for their agreement, inside agree_window_s — while
        they are still COMING IN (the newest under 1 s old): a sparse view
        whose agreement may never complete refused START and OFF for as long
        as it lasted (fix verification 2026-10-18) — and, at any age inside
        the window, while a RE-ACQUISITION's newest fix already shows the
        pose further from the tags than a correction may be: that agreement
        can only end as a refusal once engaged (fix verification round 6:
        START went through 1.2 s after such a fix, and the next one ended
        the run; disengaged it snaps)."""
        live = self._pending_live(t_now)
        if not live:
            return 0
        if float(t_now) - self._pending_t[-1] <= 1.0:
            return live
        return live if self.pending_too_big() else 0

    def pending_too_big(self) -> bool:
        """A re-acquisition pending whose newest innovation is past
        max_corr_m / max_corr_yaw_deg (x, y, yaw — what would end a run).
        Never under trust: such a correction is slewed in there, so it ends
        nothing and refuses nothing."""
        if self._pending_kind != "reacquire" or not self._pending:
            return False
        e = np.asarray(self._pending[-1], float).reshape(-1)
        if e.shape[0] != 4 or not np.all(np.isfinite(e)):
            return False
        oj = (self._pending_offj[-1] if self._pending_offj
              else self.off_jump)
        e, _ = self._drift_of(e, oj)         # (as the accept will judge it)
        c = self.cfg
        return (math.hypot(e[0], e[1]) > c["max_corr_m"]
                or math.degrees(abs(e[3])) > c["max_corr_yaw_deg"])

    def _pend(self, item, t_now: float, raw=None) -> None:
        """One fix into the agreement window: the newest anchor_n, none older
        than agree_window_s (a flickering view agrees across its gaps, not
        across minutes). ``raw`` = that fix's tag pose, kept so a
        re-acquisition the dead reckoning drops out of can go on as a blind
        one without starting over."""
        c = self.cfg
        self._pending.append(np.asarray(item, float).copy())
        self._pending_t.append(float(t_now))
        self._pending_raw.append(None if raw is None
                                 else np.asarray(raw, float).copy())
        # (the correction in flight when it came in, and its jump part)
        self._pending_off.append(self.off.copy())
        self._pending_offj.append(self.off_jump.copy())
        lo = float(t_now) - c["agree_window_s"]
        keep = [i for i, tt in enumerate(self._pending_t) if tt >= lo]
        keep = keep[-c["anchor_n"]:]
        self._pending = [self._pending[i] for i in keep]
        self._pending_t = [self._pending_t[i] for i in keep]
        self._pending_raw = [self._pending_raw[i] for i in keep]
        self._pending_off = [self._pending_off[i] for i in keep]
        self._pending_offj = [self._pending_offj[i] for i in keep]

    def _unpend(self) -> None:
        self._pending, self._pending_t, self._pending_kind = [], [], ""
        self._pending_raw = []
        self._pending_off, self._pending_offj = [], []
        self._blind_from_jump = False
        self._hold_items, self._hold_ref = [], None

    def _pending_live(self, t_now: float) -> int:
        """How many pending fixes are still inside agree_window_s."""
        lo = float(t_now) - self.cfg["agree_window_s"]
        return sum(1 for tt in self._pending_t if tt >= lo)

    def shift_z(self, dz: float) -> None:
        """The tag state's z FRAME moved by ``dz`` (the assembler re-took
        its pressure z offset: z_source pressure, every START): move the
        dead reckoning's z and the last shown pose with it, so a frame change
        is not read as a z jump (fix verification 2026-10-18)."""
        dz = float(dz)
        if not math.isfinite(dz) or dz == 0.0:
            return
        if self.z_off is not None:
            self.z_off += dz
        if self._shown is not None:
            self._shown[2] += dz

    def note_start_fault(self, t: float, why: str) -> None:
        """SHADOW: a START on a standing stop control would have REFUSED the
        START on — recorded as that, and the standing stops are marked so
        the first engaged tick does not record them again as the end of a
        run control would never have begun (fix verification round 6)."""
        self.faults.append((float(t), f"(shadow) a control START would have "
                                       f"been refused: {why}"))
        self._watch_on |= set(self._stops_now)
        if self._div_now:
            self._div_faulted = True

    def note_fault(self, t: float, why: str) -> None:
        """A stop the worker judged (a shadow run's 'a control run would
        have ended here' on a lost pose) into meta faults."""
        self._fault(t, why)

    def _log_corr(self, rec: dict) -> None:
        self.corrections.append(rec)
        if len(self.corrections) > 500:
            del self.corrections[:-500]

    def _fault(self, t: float, why: str) -> None:
        self.faults.append((float(t), why))
        if len(self.faults) > 200:
            del self.faults[:-200]

    def _slew_track(self, engaged: bool, dt: float):
        """A dead-reckoned tick's share of a correction still slewing: off
        decays as on a tag tick, and the TRACK takes what off gives up (the
        track was anchored to tag + off, so the correction is in it). A
        correction cut short by a dropout keeps converging — on a flickering
        view it would otherwise advance only on the tag ticks — and when off
        reaches 0 nothing of it is left unapplied, so OFF is safe then (fix
        verification 2026-10-18). -> the (dx, dy, dz, dyaw) moved, or None."""
        before = self.off.copy()
        self._decay(engaged, dt)
        d = self.off - before
        if not np.any(d):
            return None
        self.p = self.p + d[:2]
        if self.z_off is not None:
            self.z_off += float(d[2])
        self.head.off += float(d[3])
        return d

    def _decay(self, engaged: bool, dt: float) -> None:
        """off toward 0 at corr_rate_*; off_jump shrinks by the same factor
        per axis group (the slew takes the two parts out together)."""
        if not engaged:
            self.off[:] = 0.0
            self.off_jump[:] = 0.0
            return
        c = self.cfg
        dxy = float(math.hypot(self.off[0], self.off[1]))
        lim = c["corr_rate_m_s"] * float(dt)
        if dxy > lim:
            k = (dxy - lim) / dxy
            self.off[0:2] *= k
            self.off_jump[0:2] *= k
        else:
            self.off[0:2] = 0.0
            self.off_jump[0:2] = 0.0
        dz = abs(float(self.off[2]))
        if dz <= lim:
            self.off[2] = self.off_jump[2] = 0.0
        else:
            self.off_jump[2] *= (dz - lim) / dz
            self.off[2] = math.copysign(dz - lim, self.off[2])
        ly = math.radians(c["corr_rate_deg_s"]) * float(dt)
        dy = abs(float(self.off[3]))
        if dy <= ly:
            self.off[3] = self.off_jump[3] = 0.0
        else:
            self.off_jump[3] *= (dy - ly) / dy
            self.off[3] = math.copysign(dy - ly, self.off[3])

    def _watch(self, h, tag_ok: bool, new_fix: bool, t_now: float, since,
               blind, spd, engaged: bool = True) -> None:
        """A dead-reckoned tick: the warnings and the stops. Each stop is
        h["fault"] in BOTH modes — control ends the run on it, shadow logs
        it as the run a control session would have ended (fix verification
        2026-10-18: shadow used to skip the blind and speed stops) — and is
        recorded in meta faults once per onset — while ENGAGED (disengaged,
        it is what a START would be refused on, not a run's end)."""
        c = self.cfg
        stops = {}
        if since is not None and since > c["warn_unaided_s"]:
            h["warn"] = (f"{since:.0f} s since the last tag correction "
                         f"({self.dist_m:.2f} m)")
        if self._gap_warn:
            # (trust mode, a gap flown — said until the tags re-anchor,
            # beside the unaided warning, never over it)
            h["warn"] = ((h["warn"] + " · ") if h["warn"] else "") \
                + self._gap_warn
        if c["trust_dvl"] and self._unobs_now > c["max_corr_m"]:
            # (the one bound trust mode keeps on a gap: more motion may be
            # hidden than a correction may take in while engaged — the ONCE
            # the warning asks for would end the run; safety review
            # 2026-10-19. Disengaged it is a START refusal that a ONCE with
            # tags in view, which snaps, takes away.)
            stops["unobserved"] = (
                f"dvl pose: {self._unobs_now * 100:.0f} cm of unintegrated "
                f"motion since the anchor (DVL gaps; > dvl_pose.max_corr_m "
                f"{c['max_corr_m']:g} m, more than a correction may be while "
                f"engaged) — TAG FIX ONCE with tags in view while disengaged")
        mx = c["max_unaided_s"]
        if mx is not None and since is not None and since > mx:
            stops["unaided"] = (f"dvl pose: {since:.0f} s without a tag "
                                f"correction (> dvl_pose.max_unaided_s "
                                f"{mx:g})")
        mb = c["max_blind_s"]
        if mb is not None and blind is not None and blind > mb:
            stops["blind"] = (f"dvl pose: {blind:.0f} s without a tag in "
                              f"view (> dvl_pose.max_blind_s {mb:g}) — "
                              f"nothing can see the drift")
        ms = c["max_dr_speed_m_s"]
        if ms is not None and spd is not None and spd > ms:
            stops["speed"] = (f"dvl pose: the DVL reads {spd:.2f} m/s while "
                              f"dead-reckoning (> dvl_pose.max_dr_speed_m_s "
                              f"{ms:g} — a false lock?)")
        self._stops_now = set(stops)
        for kind in ("unaided", "blind", "speed", "unobserved"):
            if kind in stops:
                h["fault"] = stops[kind]
                if kind not in self._watch_on and engaged:
                    self._fault(t_now, stops[kind])
                    self._watch_on.add(kind)
            else:
                self._watch_on.discard(kind)
        if self.tag_on and self._pending_kind == "jump":
            # A held JUMP, the tags in view all along, is slewed whatever its
            # size (a sparse view held one past abort_hold_s and ended the
            # run on 'diverged': round 7) — so it is the DRIFT since the hold
            # began that is watched, on the same rule. Unwatched, the hold
            # was bounded by max_blind_s alone: 0.69-0.77 m flown in a
            # synthetic sparse noisy hold [측정: 합성 시험, …/out_round8/
            # evidence/wf_21856d6c-69a_result.json, safety R7-1], and
            # nothing with max_blind_s null (round 8).
            self._watch_divergence(h, tag_ok, new_fix, t_now, engaged,
                                   errs=self._hold_drift(),
                                   what="from a held tag jump's level")
        else:
            # (the OFF abort; under ON a re-acquisition after DR is judged
            # by it too, as drift — _jerr)
            self._watch_divergence(h, tag_ok, new_fix, t_now, engaged)

    def _watch_divergence(self, h, tag_ok: bool, new_fix: bool,
                          t_now: float, engaged: bool = True, errs=None,
                          what: str = "") -> None:
        """The OFF divergence abort (and the warning before it). Its
        evidence is the FRESH FIXES over abort_*: a tick with no tag does
        not reset it — a view that flickers in and out for less than
        abort_hold_s at a time must still end the run (fix verification
        2026-10-18) — but evidence older than agree_window_s expires, and
        the stop needs anchor_n fixes over the line, so two stray frames far
        apart are not a divergence. ``errs`` (xy m, yaw deg, z m) + ``what``
        judge another difference on the same rule — a held jump's drift
        (round 8); default: _jerr."""
        c = self.cfg
        self._div_now = False
        em, eyd, ez = errs if errs is not None else self._jerr()
        if not tag_ok or em is None:
            if (self._div_t0 is not None and (
                    self._div_last_t is None
                    or t_now - self._div_last_t > c["agree_window_s"])):
                self._div_t0, self._div_n, self._div_faulted = None, 0, False
            return
        if em >= c["warn_err_m"]:
            h["warn"] = (f"dvl pose {'drifted ' if what else ''}"
                         f"{em * 100:.0f} cm / {eyd:.1f} deg "
                         f"{what or 'from the tags'}")
        ab, aby, abz = (c["abort_err_m"], c["abort_err_yaw_deg"],
                        c["abort_err_z_m"])
        over = ((ab is not None and em >= ab)
                or (aby is not None and eyd >= aby)
                or (abz is not None and abs(ez or 0.0) >= abz))
        if not over:
            self._div_t0, self._div_n, self._div_faulted = None, 0, False
            return
        if self._div_t0 is None:
            self._div_t0, self._div_n = t_now, 0
        self._div_last_t = t_now
        if new_fix:
            self._div_n += 1
        if (t_now - self._div_t0 >= c["abort_hold_s"]
                and self._div_n >= c["anchor_n"]):
            head = ("drifted " + what) if what else "diverged from the tags"
            # (z named only where it was judged — the held jump's and the ON
            # watch's errors carry none: round 10, R9-4)
            zt = "" if ez is None else f" / {abs(ez) * 100:.0f} cm z"
            why = (f"dvl pose {head}: {em * 100:.0f} cm / {eyd:.1f} deg"
                   f"{zt} for {t_now - self._div_t0:.1f} s")
            h["fault"] = why
            self._div_now = True
            if not self._div_faulted and engaged:   # one record per onset
                self._fault(t_now, why)
                self._div_faulted = True

    def _note_depth(self, depth, t_d, src: str = ""):
        """One depth reading into the heave filter and the depth history ->
        the depth to USE this tick (None: none). A reading further from the
        last accepted one than this vehicle can move (0.5 m/s x the interval
        + 2 cm) is a spike: refused — the last accepted depth stands — until
        the new level has held for 1 s (a real step: then it is the depth;
        under ON the next fix captured on it re-takes the offset, the z jump
        test off until then — round 6: the new level was flown as a held z
        'jump' and slewed back at 2 cm/s; under OFF the dead reckoning steps
        with it). Round 4 guarded w only; the spike went into the
        dead-reckoned z (fix verification 2026-10-18)."""
        if depth is None or t_d is None:
            return None
        depth, t_d = float(depth), float(t_d)
        if src != self._z_prev_src:
            # another sensor: no finite difference across the two (their
            # offset would read as a heave velocity), and no history
            self._z_prev, self._z_prev_src = None, src
            self._zh.clear()
            self._z_rej_t0, self._z_rej_v = None, None
        if self._z_prev is not None and t_d > self._z_prev[0] + 1e-6:
            dtz = t_d - self._z_prev[0]
            if dtz < 0.02:
                # closer than any rate the Bar30 runs at (10 Hz here): a
                # burst of queued records, not a measurement interval — wait
                # for the next one (differenced against the older record)
                return self._z_prev[1]
            if dtz <= self.cfg["depth_stale_s"]:
                jump = abs(depth - self._z_prev[1])
                if jump > 0.5 * dtz + 0.02:
                    if self._z_rej_t0 is None:
                        self._z_rej_t0 = t_d
                    self._z_rej_v = depth
                    if t_d - self._z_rej_t0 < 1.0:
                        return self._z_prev[1]           # the spike: refused
                    return self._z_level(t_d, depth)     # held for a second
                if (self._z_rej_t0 is not None and self._z_rej_v is not None
                        and abs(depth - self._z_rej_v)
                        < abs(depth - self._z_prev[1])):
                    # taken now only because the time since the last
                    # accepted reading has grown to the vehicle's reach — the
                    # readings between were this level already: a LEVEL
                    # change, not a heave (round 6)
                    return self._z_level(t_d, depth)
                self._z_rej_t0, self._z_rej_v = None, None
                v = (depth - self._z_prev[1]) / dtz
                # (no heave this vehicle flies reaches 0.5 m/s)
                v = max(-0.5, min(0.5, v))
                a = self.cfg["vz_lp_alpha"]
                self.vz = (1.0 - a) * self.vz + a * v
        if not self._zh or t_d > self._zh[-1][0] + 1e-6:
            self._zh.append((t_d, depth))
        if self._z_prev is None or t_d > self._z_prev[0] + 1e-6:
            self._z_prev = (t_d, depth)
        return depth

    def _z_level(self, t_d: float, depth: float):
        """A Bar30 LEVEL change taken (no heave): the history restarts on it,
        and under ON the next fix captured on it re-takes the z offset, the
        z jump test off until then (fix verification round 6: the new level
        was flown as a held z 'jump' and slewed back at 2 cm/s)."""
        self._z_prev, self._z_rej_t0, self._z_rej_v = (t_d, depth), None, None
        self._zh.clear()
        self._zh.append((t_d, depth))
        self._z_retake = True
        return depth

    def _nudot(self, R, nu_lin, gyro, imu, dt: float) -> np.ndarray:
        """The state assembler's nudot, verbatim in form (state_assembler.py
        step(), '---- nudot'), so a controller sees the same quantity from
        either pose."""
        if self._prev_gyro is None:
            ang_acc = np.zeros(3)
        else:
            ang_acc = (gyro - self._prev_gyro) / dt
        if self.nudot_source == "imu" and imu.ax is not None:
            f = np.array([float(imu.ax), float(imu.ay), float(imu.az)])
            a_lin = f + R.T @ G_NED - np.cross(gyro, nu_lin)
        else:
            if self._prev_nu is None:
                a_lin = np.zeros(3)
            else:
                a_lin = (nu_lin - self._prev_nu[:3]) / dt
        return np.concatenate([a_lin, ang_acc])

    def _out_ok(self, h, state, eta, nu, nudot, gyro, t_now):
        self._prev_gyro = np.asarray(gyro, float).copy()
        self._prev_nu = np.asarray(nu, float).copy()
        self.state = state
        self.rows[state] = self.rows.get(state, 0) + 1
        h["ok"], h["state"] = True, state
        h["anchored"] = self.anchored
        h["unobserved_m"] = self._unobs_now
        if state in (ST_DR, ST_HELD):
            self._dr_shown = True
        if self.t_fix is not None and h["since_fix_s"] is None:
            h["since_fix_s"] = t_now - self.t_fix
        if h.get("dist_m") is None:
            h["dist_m"] = self.dist_m
        self._shown = np.asarray(eta, float).copy()
        if self._last_depth is not None:
            self._shown_dz = float(eta[2]) - float(self._last_depth)
        return {"eta": np.asarray(eta, float), "nu": np.asarray(nu, float),
                "nudot": np.asarray(nudot, float)}, h

    def _none(self, h, why: str, state: str = ST_LOST):
        self._carry = "lost"
        self.state = state
        self.rows[state] = self.rows.get(state, 0) + 1
        h["ok"], h["why"], h["state"] = False, why, state
        self._prev_gyro = None
        self._prev_nu = None
        return None, h

    # ============================================================== record
    def meta(self) -> dict:
        return {"mode": self.mode, "tag_on": bool(self.tag_on),
                "heading": self.head.source,
                "bias": self.head.bias_info(),
                "anchored": bool(self.anchored), "n_anchor": self.n_anchor,
                "toggles": list(self.toggles),
                "corrections": list(self.corrections),
                "faults": [{"t": t, "why": w} for t, w in self.faults],
                "n_jumps_held": int(self.n_jumps_held),
                "n_anchor_refused": int(self.n_anchor_refused),
                "gyro_holes": int(self.head.n_holes),
                "xy_holes": int(self.n_xy_holes),
                "trust_dvl": bool(self.cfg["trust_dvl"]),
                "gaps_trusted": int(self.n_gaps_trusted),
                "bias_seed": self.head.seed,
                "n_dvl_usable": int(self.n_usable),
                "n_dvl_unusable": int(self.n_unusable),
                "max_err_m": round(self.max_err_m, 4),
                "max_since_fix_s": round(self.max_since_fix_s, 2),
                "max_blind_s": round(self.max_blind_s, 2),
                "max_dist_m": round(self.max_dist_m, 3),
                "rows": dict(self.rows),
                "config": dict(self.cfg)}
