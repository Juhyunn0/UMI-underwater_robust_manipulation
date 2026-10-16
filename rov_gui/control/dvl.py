#!/usr/bin/env python3
"""dvl.py — a DVL as the estimator's THIRD sensor: AprilTag + IMU + DVL.

HARDWARE-UNVERIFIED. Written 2026-10-01, before a DVL was mounted on this
vehicle, against the Water Linked DVL A50 protocol pages. Every threshold
below is tagged [예측] and every device fact [스펙] with the page it was read
from; NOTHING here has been measured on this vehicle. The first pool session
with the device replaces the [예측] values, not this docstring's claims.

WHAT THE DVL IS FOR. The state assembler (state_assembler.py) builds the body
velocity nu[0:3] by finite-differencing tag positions and low-passing the
result. That works, and its own comment names the cost: the low-pass lag is
"the estimator's dominant error at this vehicle's speeds". A DVL measures
that velocity directly. So the fusion is deliberately small (operator
decision 2026-10-01 — "기존 assembler에 속도원 추가", not a new EKF):

    position x, y, yaw   AprilTag PnP           unchanged
    z, roll, pitch, pqr  barometer / ArduSub    unchanged
    nu[0:2]  u, v        tag finite difference  -> DVL when it is IN and usable
    nu[2]    w           tag finite difference  unchanged (see below)
    fix-age bridging     that same velocity     -> the DVL's, integrated

and, for a STATION hold only, the tag-dropout ladder may integrate the DVL
velocity from the last fix instead of freezing x/y (station_bridge.py tier
``dvl``). That is the same division of labour the "Learning to Swim" paper
describes for CUREE (AprilTag = global position, DVL + IMU = faster feedback
and the bridge when the tag is not seen; arXiv:2410.00120v2 Fig. 5) without
its EKF, whose details that paper does not give.

ONLY THE HORIZONTAL PAIR IS CONSUMED. The proof below sees surge and sway and
nothing else, so a tilted bracket, a wrong vz sign convention or a wrong
lever arm would put an unverified number into the heave rate with every gate
silent (safety review 2026-10-02). Until a vertical check exists, w stays on
its old source; the DVL's own w is still recorded (CSV ``dvl_w``).

THREE SWITCHES, and they answer three different questions:

    dvl.enabled / --dvl   is a DVL client built at all?  Off = nothing is
                          constructed and the tick is byte-identical.
    dvl.use / GUI button  may the estimator CONSUME it?  OUT = "shadow": the
                          stream is still received, gated, compared with the
                          tag velocity and recorded — which is the point of a
                          test phase. Changeable only while disengaged
                          (operator decision 2026-10-01: one run = one
                          estimator), so a CSV never straddles the switch.
    usable (automatic)    is THIS TICK's sample fit to fly on?  Not usable
                          while IN = that tick falls back to the tag finite
                          difference, and the CSV row says so (``vel_src``).

WHY "PROVEN". A wrong mount is invisible at rest — both velocities read zero
— and becomes velocity feedback of the wrong sign the moment the vehicle
moves. So IN is refused until this session's own data has shown that the DVL
body velocity and the tag finite difference describe the same motion: a 2x2
least-squares map between them, accepted only with motion in BOTH surge and
sway and only when the map is within tolerance of identity by three standard
errors (:class:`Agreement`). The same fit reports the mount's apparent yaw
correction and scale, which is the commissioning number the operator needs.
``require_proven: false`` is a recorded opt-out of THAT proof only.

AUTOMATION MAY ONLY TAKE THE DVL OUT. Evidence that the DVL is WRONG — the
fit refuting the mount, a standing disagreement with the tag velocity, the
device's own configuration changing — is a TRIP: latched, whatever
``require_proven`` says, until the operator sets the switch IN again. And
once this session's data has said the DVL is wrong, IN needs a PROOF made
after that — also under ``require_proven: false``, which waives the proof a
DVL with nothing against it would need, not the one evidence demands. A
trip throws away the proof and the pairs it was made of; so does a change
of the device's configuration, in every state (the pairs describe a stream
that no longer exists). A missing bottom lock or a stale link is not a
trip; it is a tick without a DVL, and the DVL comes back by itself after
``reentry_s`` of good samples. And whether an ENGAGEMENT may use the DVL at
all is decided once, at engage (MpcWorker), so a proof that completes
mid-run cannot change the estimator under a mission.

WHAT IS A TRIP, WHAT IS ONLY A TICK WITHOUT THE DVL. A trip needs the DVL
to be wrong in a way that LASTS, and there are two such ways:

    the mount     the tags SEE the motion the DVL reports (at least 80 % of
                  it, COHERENT_MIN) and put it somewhere else — the
                  alignment monitor, and a coherent refutation by the fit
    a false lock  the DVL stands apart from the tag velocity by more than
                  agree_trip_m_s for about agree_trip_s, whatever the tags
                  saw — the difference monitor. A DVL that reports for a
                  second a motion that is not there is not fit to fly on.

Neither is: a glitch of a few reports, zero-mean DVL noise, a fit that
refutes only because the DVL "moves" and the tags do not. Those are held
back without a latch, and the DVL returns by itself when they pass:

    innovation hold   a sample further than innov_max_m_s from the tag
                      velocity; the DVL then stays out until a tag velocity
                      from a NEWER camera frame agrees — the same frame
                      growing older (its limit widening) does not
                      rehabilitate it
    alignment hold    the alignment monitor reads the DVL's motion as going
                      somewhere the tags do not show, however little of it
                      the tags see
    refutation hold   the fit currently refutes the DVL, within the last
                      REFUTE_HOLD_S of evidence

(Reviews 2026-10-02: without the "tags see it" condition a right, proven
mount was latched out by two reports 12 cm/s off, by 3 cm/s of DVL noise at
rest, and — with the condition at one half — by repeated short glitches
that dilute the fit's slope to 0.6-0.75; without the innovation hold, a
false lock the gate was rejecting went in as soon as the fix grew old.)

SELECTION (2026-10-16, ``dvl.selection``). DTF below = RL_controller
logs/eval/dvl_trip_fix_20261015 (SPEC.md, DESIGN_DECISIONS.md and the
replays of the 10-13 / 10-15 pool recordings behind every [측정] number of
the 2026-10-16 keys). Operator, 2026-10-15: the DVL is the station's MAIN
horizontal velocity sensor, and disagreeing with the tag velocity must not
latch it out — the tag finite difference is the WORSE of the two (DVL
2.3-2.9 vs tag 5.5-7.0 cm/s RMS against a non-causal tag fit [측정
RL_controller logs/eval/vel_source_20261015]); the shipped difference trip
fired ~18 times an hour on replay, and the innovation gate threw away the
closer of the two velocities on 80 % of its rejections [측정 same].

    primary  (the default) the DVL IS nu[0:2]. Each tick blends it with the
             tag finite difference exactly as the pwm11att13 policy was
             trained (DvlBlend = RL_controller deploy/obs_builder.py
             DvlBlendNp): a newest sample that is not usable is replaced by
             the last usable one for blend_hold_s, then the weight fades to
             the tag velocity over blend_xfade_s, and back — no hard
             switch, nothing to press. Not usable = a data gate above; an
             R3 quality flag (fom, altitude jump, beams, lock re-acquired:
             the report's own word that it may be a false lock); or a
             DISAGREEMENT with the tag velocity — the slow difference, the
             alignment monitor, a refuting fit — which is a HOLD through
             that same fade, released by itself once it has cleared and its
             evidence has drained (disagree_hold). The re-entry wait, the
             innovation gate and "unchecked" are not on this path. Two
             things still LATCH: the device's own configuration changing
             (unchanged) and ESCALATION — a disagreement that LASTS
             (esc_latch_s of evidence in a leaky count) or keeps COMING
             BACK (esc_repeat_n episodes within esc_repeat_window_s, each
             with difference, fit or scale evidence: EPISODE_KINDS). Like
             every trip, a latch throws the proof away: re-prove, then OUT
             and IN.
    legacy   the 2026-10-01..15 ladder described above, bit for bit — the
             rollback. An RL export trained on the blend refuses to load
             under it (rl_policy.check_velocity_source).

Above, under primary, read "trip" as "hold" except for those two latches.

No Qt, no sockets, no dobmpc: plain numpy, so it is tested bare
(rov_gui/tests/test_dvl.py). The TCP client lives in backends/dvl.py.
"""

from __future__ import annotations

import json
import math
from collections import Counter, deque

import numpy as np

from .state_assembler import rot_zyx

S_FLU_FRD = np.diag([1.0, -1.0, -1.0])     # same involution as geometry.py

#: ``MpcStatus.dvl_state`` vocabulary ("" = no DVL was built).
ST_NO_LINK = "no_link"      # built, but no (fresh) report
ST_NO_LOCK = "no_lock"      # reports arrive; this one fails a data gate
ST_SHADOW = "shadow"        # good data, operator has it OUT: recorded only
ST_UNPROVEN = "unproven"    # operator has it IN, the mount is not proven yet
ST_TRIPPED = "tripped"      # evidence it is wrong; latched until re-armed
ST_FALLBACK = "fallback"    # IN, good data, held back this tick (legacy)
ST_IN_USE = "in_use"        # nu[0:2] is the DVL's
# 2026-10-16, selection primary (which never answers ST_FALLBACK, so "held"
# always means self-releasing):
ST_HELD = "held"            # IN, a quality flag / disagreement / gross:
#                             held back, returns by ITSELF
ST_NOT_ARMED = "not_armed"  # IN, but not used this run (not armed at
#                             engage, a code fault)
ST_NO_TAG = "no_tag"        # this tick had no tag state (the blend waits)
STATES = (ST_NO_LINK, ST_NO_LOCK, ST_SHADOW, ST_UNPROVEN, ST_TRIPPED,
          ST_FALLBACK, ST_IN_USE, ST_HELD, ST_NOT_ARMED, ST_NO_TAG)

VEL_TAG_FD = "tag_fd"
VEL_DVL = "dvl"
#: ``dvl.selection`` values (2026-10-16): the DVL-primary blend, or the
#: 2026-10-01..15 ladder kept for a rollback.
SELECTIONS = ("primary", "legacy")

DEFAULTS = {
    "enabled": False,
    # The switch's position at startup. False = shadow. The panel button
    # changes it between engagements; the run meta records what was flown.
    "use": False,
    # [스펙: bluerobotics.com DVL A50 integration guide — the address the
    # BlueROV2 guide assigns; the device's own fallback is 192.168.194.95,
    # docs.waterlinked.com/dvl/networking]
    "host": "192.168.2.95",
    # [스펙: docs.waterlinked.com/dvl/dvl-json-protocol — "TCP server on port
    # 16171", newline-delimited JSON]
    "port": 16171,
    "connect_timeout_s": 2.0,
    "reconnect_s": 2.0,
    # How often the client re-asks the device for its configuration
    # (get_config — read-only). The device rotates its own output by its
    # "mounting rotation offset"; if someone changes that in the web GUI the
    # proof no longer describes the stream, and this is how the station
    # finds out. 0 = once per connection only. [예측]
    "config_poll_s": 10.0,
    # THE MOUNT, both null until measured on the vehicle. IN is refused while
    # either is null; shadow still runs (identity rotation, zero lever arm)
    # so the agreement fit can say what the rotation actually is.
    #   mount_rpy_deg  [roll, pitch, yaw] of the DVL output frame in body FRD.
    #                  [0, 0, 0] = LED edge forward, transducers down, AND the
    #                  device's own mounting_rotation_offset left at 0. The
    #                  device rotates its output by that offset itself, so a
    #                  yaw entered in BOTH places is applied twice.
    #   mount_t_flu    the DVL's measurement origin in body FLU metres, from
    #                  the same body origin cam_t_flu uses. The origin is
    #                  28 mm ABOVE the back plate on the A50 [스펙: axes page].
    "mount_rpy_deg": None,
    "mount_t_flu": None,
    # ---- per-sample gates -------------------------------------------------
    # The docs' own example report is 0.19 s old at transmission
    # (time_of_transmission - time_of_validity = 189 ms at 0.49 m altitude
    # [스펙 예시 1개]); reports come at 10-15 Hz below 3 m [스펙: range-mode
    # page] and periodic cycling drops some every 10 s. 0.6 s keeps two
    # missed reports usable. [예측]
    "stale_s": 0.6,
    # fom is the device's own velocity accuracy in m/s (docs example at rest:
    # 0.00016). What it reads in THIS pool is unknown. [예측]
    "fom_max": 0.05,
    # [스펙: overview page — A50 altitude range starts at 5 cm]
    "min_altitude_m": 0.05,
    # Sanity only. [스펙] the A50 tracks to 3.75 m/s; this vehicle has never
    # exceeded 0.5 m/s. [예측]
    "v_max_m_s": 1.5,
    # Seconds of uninterrupted good samples before the DVL is used (again).
    # Stops a marginal bottom lock from flipping the velocity source every
    # tick. [예측] LEGACY ONLY: primary has no re-entry wait — the fade
    # (blend_xfade_s) does that job without a hard switch, and a lock that
    # flickers is flagged (flag_reacquired). The setup log names it.
    "reentry_s": 0.5,
    # The fastest this vehicle can change its velocity: 30 N (axis_cap 0.5 x
    # 60 N) over ~19 kg with added mass is 1.6 m/s^2 [유도], rounded up. A
    # sample further than a_max*dt + 5 cm/s from the last good one is a
    # SPIKE, not a manoeuvre, and is not used. [예측]
    "a_max_m_s2": 2.0,
    # With a current tag velocity to compare against, a sample further than
    # this from it (horizontal) is not used — the gross-outlier gate, e.g. a
    # false bottom lock off a wall. Widened by the proof's own measure of the
    # tag-velocity noise, and by a_max x (how stale the tag velocity is
    # beyond 0.25 s). Once rejected, the DVL stays out until a tag velocity
    # from a NEWER camera frame agrees ("innovation hold"). The slow, small
    # disagreements are the trips' business below. [예측]
    # That gate is LEGACY: replayed on 10-15 it threw away the closer of the
    # two velocities 80 % of the time [측정 RL_controller logs/eval/
    # vel_source_20261015 accuracy/out/trips_replay.json]. PRIMARY uses the
    # number for the station bridge's DVL rung: a tagged tick opens it only
    # with its sample within this (widened the same way) of the tag velocity
    # (rung_admit). BOTH modes: the monitors' gross-pair floor and the
    # per-pair clip of the 1-s residual.
    "innov_max_m_s": 0.15,
    # ---- the agreement with the tag finite difference -----------------------
    "require_proven": True,
    # Synthetic prototype 2026-10-01 (NOT a measurement; tag noise 0.8 mm at
    # 15 Hz, DVL noise 3 mm/s): 8 s of 8-12 cm/s surge AND sway identifies the
    # 2x2 map to +-0.01 and the mount yaw to +-0.3 deg; surge-only or a
    # station hold correctly does not. All [예측].
    "agree_min_pairs": 60,
    "agree_min_excitation_m_s": 0.025,
    "agree_tol": 0.25,
    "agree_v_min_m_s": 0.02,
    "agree_window_s": 60.0,
    # A PROVEN mount is un-proven by the fit only after this many consecutive
    # refuting evaluations (one per new tag fix): one evaluation on a window
    # that happens to hold a tag jump must not undo a proof. [예측]
    # PRIMARY: that many in a row make the refutation CURRENT, which holds
    # the DVL back (self-releasing, any coherence); a coherent one is also
    # escalation evidence, and only an escalation latch un-proves.
    "agree_refute_n": 15,
    # LEGACY: the runtime monitors below each latch a trip. PRIMARY: neither
    # trips — the alignment verdict holds the DVL back and, when coherent,
    # feeds the escalation; the difference's role passes to the slow
    # residual (agree_slow_s, 2026-10-16 keys); agree_trip_m_s is LEGACY ONLY
    # and agree_trip_s stays the alignment window and the 1-s residual's
    # (recorded) time constant. All three are fed by the SAME
    # evidence as the proof — one TIME-ALIGNED pair per new tag fix (the raw
    # tag difference against the DVL velocity interpolated to its midpoint) —
    # so none of them depends on how late either sensor's data arrives
    # (simulation 2026-10-02: fed by the tick's low-passed tag velocity
    # instead, a right mount tripped whenever the two latencies differed by
    # more than ~0.15 s and the vehicle oscillated at 0.5 Hz or faster):
    #  * the low-passed horizontal DIFFERENCE vector (time constant
    #    agree_trip_s) above trip_m_s. Catches a standing offset: a reversed
    #    or rotated mount at speed, a scale error, a false lock that lasts.
    #    Blind to an oscillation, which averages to zero. Every pair counts
    #    as at most innov_max_m_s apart — a FIXED clip, whatever the tag
    #    noise: a tag fix that jumps is one pair of metres per second, and a
    #    burst of a few pairs must not be able to trip, while a DVL that is
    #    grossly wrong pair after pair gets there in about 0.8 s. (Clipped at
    #    4 sigma of the proof's scatter instead, 3 mm of tag noise let ONE
    #    glitch of 3-4 reports trip it in 0.4 s — review 2026-10-02.) Pairs
    #    beyond that 4-sigma limit are GROSS: left out of the alignment sums.
    #  * ALIGNMENT, the proof's own test in miniature over the last
    #    agree_trip_s: along each direction the vehicle is actually moving in
    #    (the principal axes of the DVL velocity, each judged only above
    #    ratio_v_min of rms speed and when it carries >= ALIGN_MIN_SHARE of
    #    the motion), 1 - |A e - e| for the least-squares map
    #    A: tag <- DVL. 1 = the same motion, -1 = reversed, -0.4 = a quarter
    #    turn. It reads BAD below ratio_min by THREE standard errors (the
    #    regression's own residual), so slow or noisy motion is simply "no
    #    evidence". While it reads bad the DVL is held back; it TRIPS only
    #    when the tags also see at least COHERENT_MIN (80 %) of that motion
    #    and it has read bad for 0.4 s on end. This is the monitor that
    #    survives the case the difference is blind to: a station hold
    #    OSCILLATING on wrong-signed feedback. A mount the proof accepts
    #    cannot trip it (agree_tol 0.25 is at most a 0.25-0.5 deviation;
    #    14.5 deg of yaw is 0.25).
    "agree_trip_m_s": 0.08,
    "agree_trip_s": 1.0,
    "agree_ratio_min": 0.5,
    "agree_ratio_v_min_m_s": 0.02,
    # ==== 2026-10-16: SELECTION (module doc; DTF SPEC.md §2). Twenty NEW
    # keys — none of the 27 above is re-purposed, and the shipped hw_mpc.yaml
    # names none of these, so it runs on exactly these values. ============
    # "primary" | "legacy". [가정: operator decision 2026-10-15, the DVL is
    # the main horizontal velocity sensor; legacy = the rollback]
    "selection": "primary",
    # THE BLEND (primary): the pwm11att13 export's velocity_source.hold_s /
    # xfade_s / gross_m_s / gross_tau_s [스펙: RL_controller
    # rov_rl/velocity_source.py HOLD_S, XFADE_S, GROSS_M_S, GROSS_TAU_S] —
    # stale_s above is its stale clock (export stale_s 0.6 [스펙]). An RL
    # export trained on the blend refuses to load if any of the five differ
    # (rl_policy.check_velocity_source). blend_gross_m_s 0 = no gross check.
    "blend_hold_s": 0.45,
    "blend_xfade_s": 0.5,
    "blend_gross_m_s": 0.5,
    "blend_gross_tau_s": 1.0,
    # R3 QUALITY FLAGS (primary): the report itself says it may be a false
    # lock. A flagged sample is not FLOWN (hold, then fade) — it stays in
    # the pairs, the monitors and the fit (as data gates they cut the 17:45
    # pairs from 1258 to 561 and an injected +10 cm/s bias never latched
    # [측정 DTF map/out/package_replay2.log]) and does not become the jump
    # gate's reference. Never a trip. 0 / false = off. Share of the > 8 cm/s
    # wrong 10-15 reports flagged / of the good ones flagged [측정 DTF
    # design/out/flag_recall.json]: fom 20.3 / 1.9 %, altitude 20.4 / 5.4 %,
    # beams 9.1 / 0.8 %, re-acquired 20.4 / 4.3 %, all four 51.3 / 11.1 %.
    # A firmware without `transducers` reports 0 beams: then EVERY sample is
    # flagged and the panel names the flag (fail-safe).
    "flag_fom_m_s": 0.003,       # valid fom p95 2.6 mm/s [측정 RL_controller
    #                              logs/eval/vel_source_20261015 accuracy/out/
    #                              dvl_detectors.json]
    "flag_alt_jump_m": 0.10,     # |altitude - median of the unflagged reports
    "flag_alt_window_s": 1.0,    # of the last 1.0 s (>= 3 of them)| [측정 DTF]
    "flag_min_beams": 4,         # 3-beam error p50/p90 6.9/16.0 vs 1.4/6.1
    #                              cm/s with 4 [측정 vel_source_20261015
    #                              pipeline/out/a16_beams_fom.json]
    "flag_reacquired": True,     # the report before it failed a data gate:
    #                              a lock that flickers never flies (1:1
    #                              flicker, DVL weight 10.0 -> 0.58 s of 10 s
    #                              [측정 DTF design/out/synth_v2.json])
    # DISAGREEMENT (primary) — a HOLD, never a trip by itself: the time-
    # aligned difference (the monitors' pairs), each pair clipped to
    # agree_slow_clip_m_s (its own key: tied to innov_max_m_s, a clip of 0.30
    # gave 5.5 false trips/h [측정 vel_source_20261015 pipeline/out/
    # a12b_clip_coupling.json]), low-passed over agree_slow_s; above
    # agree_hold_m_s it holds the DVL back until it is below
    # agree_release_m_s AND the escalation below has drained. Largest on the
    # 58 recorded CSVs (0.735 h): 4.06 cm/s, 0 holds [측정 DTF design/out/
    # clean_ev.json]; the 2:1 hysteresis is [가정].
    "agree_slow_s": 4.0,
    "agree_slow_clip_m_s": 0.15,
    "agree_hold_m_s": 0.05,
    "agree_release_m_s": 0.025,
    # ESCALATION (primary) — the one latch on disagreement: a bucket of
    # seconds of evidence (the slow difference above agree_hold_m_s, the
    # alignment monitor's coherent verdict, a coherent refutation by the fit,
    # or the DVL reading > esc_scale_ratio x the rms motion the tags see
    # while they see >= esc_scale_v_min_m_s of it), drained at esc_leak x the
    # time without any, moving only while armed and not latched, capped at
    # esc_latch_s. Latched at esc_latch_s ("persistently") or on
    # esc_repeat_n counted episodes within esc_repeat_window_s
    # ("repeatedly"; an episode counts only with difference, fit or scale
    # evidence — EPISODE_KINDS). Recorded CSVs: bucket max 1.43 s, no counted
    # episode (4 of alignment alone), scale evidence 0 s; 786 transient
    # faults (zero locks <= 3 s, offsets <= 1.1 s, silences, flicker): 0
    # latches, bucket max 3.11 s; 850 single glitches (+0.3-0.8 m/s for
    # 0.3-1.5 s, steps / ramps / fom-flagged): 0 latches, bucket max 1.29 s;
    # +10 cm/s bias / reversed sway / 180 deg / 2x scale held 30 s: 34 / 35
    # / 35 / 31 of 35 latched [측정 DTF fix1/out/clean_ev_impl.json,
    # f_compare_final.json]. Leak 0.25-1.0 changes none of those [측정 DTF
    # design/v1/out_v1/esc_choose.json]; the leak itself is [가정].
    "esc_latch_s": 4.0,
    "esc_leak": 0.5,
    "esc_scale_ratio": 1.5,      # 0 = off; 1.4 gives 1.1 s of evidence on the
    "esc_scale_v_min_m_s": 0.05,  # recordings [측정 DTF design/out/scale_rules.json]
    "esc_repeat_n": 3,           # 0 = off
    "esc_repeat_window_s": 60.0,  # = agree_window_s, the fit's own [가정]
}

KEYS = tuple(DEFAULTS)
#: The keys that arrived on 2026-10-16 (selection primary): the shipped
#: hw_mpc.yaml spells out the 27 before them and none of these until the
#: operator applies DTF hw_mpc_dvl_patch.diff (test_dvl's config test).
KEYS_20261016 = KEYS[KEYS.index("selection"):]
#: The shipped keys `primary` does not use, and what governs instead — named
#: in the setup log (with a warning when one is not its default: tuning it at
#: the pool would change nothing) and in the meta. innov_max_m_s is NOT here:
#: primary uses it (DEFAULTS).
INACTIVE_IN_PRIMARY = {
    "reentry_s": ("blend_xfade_s",),
    "agree_trip_m_s": ("agree_hold_m_s", "agree_release_m_s", "esc_latch_s"),
}

#: Margin of the jump gate on top of a_max*dt (m/s): two samples' worth of
#: noise at the [예측] 3 mm/s plus room for the lever-arm term.
JUMP_MARGIN_M_S = 0.05
#: The tag velocity is "current" for the innovation gate up to this fix age.
FD_FRESH_S = 0.25
#: ...and the gate's reference (the tag velocity of the newest tick that had
#: one) may itself be this old before a tick counts as UNCHECKED. One or two
#: ticks without a tag state — a fix age that touches tag_stale_s once per
#: cycle at 4 Hz tags, a late tick — still check against it, with the limit
#: widened for its age; a real tag gap does not (review 2026-10-02: clearing
#: it on every such tick made the velocity source flap once per fix cycle).
FD_REF_MAX_S = 0.25
#: Two aligned pairs further apart than this are not one stretch of
#: evidence: the monitors' clock restarts (the same limit note_fix puts on a
#: tag difference).
MONITOR_GAP_S = 0.5
#: The alignment monitor's standard error means nothing on a handful of
#: pairs; it is silent below this effective count (~0.7 s at 15 Hz).
ALIGN_MIN_PAIRS = 10.0
#: A disagreement is evidence about the MOUNT only when the tag velocity
#: shows at least this fraction of the motion the DVL reports. A reversed or
#: rotated mount reads 1 (the same motion, elsewhere); DVL noise, a glitch
#: or a false lock at rest read ~0 (the tags saw nothing); motion the DVL
#: reports only part of the time — repeated short glitches, noise on top of
#: slow motion — reads 0.55-0.75 (the slope of the tags on a noisy regressor
#: is pulled toward zero); a DVL reading 1.25x the true speed reads 0.8.
#: Reviews 2026-10-02: without the condition a RIGHT mount was tripped by
#: two reports 12 cm/s off and by 3 cm/s of DVL noise at rest; at one half,
#: by 2-2.5 cm/s of noise and by glitches of 4 reports every 3-10 s (a
#: "scale 1.3x" refutation) and by one 4-report glitch at a square's corner
#: (an alignment trip). Below it the disagreement still holds the DVL back.
COHERENT_MIN = 0.8
#: A refutation by the fit holds the DVL back only while it is RECENT: the
#: newest refuting evaluation no older than this. The window keeps old
#: pairs for agree_window_s, and with the vehicle at rest no new pair is
#: admitted to replace them — a DVL that went quiet again stayed held back
#: for ~52 s (review 2026-10-02). Motion brings new evaluations, which renew
#: the hold at once if the window still refutes.
REFUTE_HOLD_S = 3.0
#: The largest weight one pair may carry in the alignment sums. The weight
#: is dt / agree_trip_s, and the effective pair count settles at
#: (2 - a) / a — 9 at one pair per 0.2 s, which never reaches
#: ALIGN_MIN_PAIRS: the monitor was silently off whenever the tags came at
#: 5 Hz or slower (review 2026-10-02). Capped, the memory stretches in TIME
#: instead and the count settles at 13.
ALIGN_MAX_GAIN = 1.0 / 7.0
#: ...and a direction carrying less than this share of the motion (its
#: second moment against the sum of both) is not judged at all. Along such
#: a direction the regression is ill-conditioned: a few DVL-only reports
#: (a glitch) correlated with whatever the strong direction was doing at
#: that moment read as "the tags see this motion going elsewhere" — on
#: 0.3 m/s passes one 4-6 report surge glitch tripped a right mount through
#: the minor direction (round 5: 44 of 906 synthetic runs at 0, 9 of 162 at
#: 0.03, none at 0.05-0.10; 0.10 lost a reversed sway on a 10 cm/s
#: oscillation that 0.07 still trips in 1.9 s). A reversed axis carrying
#: this little of the motion is the straight-line blind spot already
#: documented.
ALIGN_MIN_SHARE = 0.07
#: ...and its verdict has to STAND this long before it trips. When motion
#: starts along a new direction, the first pair or two ARE that direction's
#: whole evidence; if the two sensors' stamps are a few tens of ms apart, in
#: those pairs one has stepped and the other has not — a slope of zero on
#: almost no residual. Two or three more pairs outweigh them (simulation
#: 2026-10-02: 50 ms of stamp error tripped a right mount at the corners of
#: a 10 cm/s square without this).
ALIGN_HOLD_S = 0.4
#: 2026-10-16 (primary). Pairs further apart than this restart the SLOW
#: monitors (the slow residual, the slow powers, the escalation bucket); the
#: 1-s residual and the alignment sums keep MONITOR_GAP_S. With the tags in
#: bursts (0.3 s on, 0.7 s off) the 0.5-s restart froze them: a 20-s zero
#: lock at 0.12 m/s was first held after 11.25 s and never latched; with
#: this, 4.25 s and 10.15 s [측정 DTF design/out/synth_v2.json]. 2.0 [가정]
SLOW_GAP_S = 2.0
#: ...the interval one pair gives the slow EMAs is capped at this: 0.5
#: raised the clean slow-residual maximum from 4.06 to 4.78 cm/s (tag noise
#: right after gaps at high tilt) [측정 DTF design/out/cap_choose.json] —
SLOW_DT_CAP_S = 0.25
#: ...and the interval one pair gives the escalation bucket at this, which
#: only counts time that already has evidence (clean bucket max 1.21 ->
#: 1.43 s, still 0 latches [측정 same]).
ESC_DT_CAP_S = MONITOR_GAP_S
#: (fix round 1) The slow POWERS behind scale_ratio leave out a GLITCH pair:
#: one that is gross AND further from the tag velocity than a DVL reading up
#: to this many times the motion could be at that speed ((this - 1) x |v_tag|
#: + JUMP_MARGIN_M_S). A scale error is then seen whole up to 3x at any speed,
#: and a glitch larger than twice the speed never feeds the scale evidence
#: [가정 3x; 측정 effect DTF fix1/out/scale_faults_aid_m2.json,
#: glitch_slowmotion_aid_m2.json].
SCALE_PAIR_MAX = 3.0
#: An escalation EPISODE (the bucket rising from 0) counts toward
#: esc_repeat_n once its bucket has reached this: 4 counted on the 0.735 h
#: of recordings, never more than 2 within 60 s [측정 DTF design/out/
#: clean_ev.json, episode_choose.json] (0.25 would also latch 10 of 23
#: recorded repeated 3-s zero locks instead of 5 — a module constant a
#: later session may lower).
EPISODE_MIN_S = 0.5
#: ...and only once it has carried one of these kinds of evidence (fix
#: round 1): an episode of ALIGNMENT evidence alone is still held (dis_align)
#: and still fills the bucket toward the persistent latch, but it does not
#: count toward the repeated one. All 4 counted episodes on the recordings
#: were alignment-only, two of them 55 s apart on mpc_192708 ("one more
#: latches it"), and in the second the DVL was the better sensor (1.8 vs
#: 8.8 cm/s RMS against the reference) [측정 DTF verify/out/
#: episodes_real.json, verify/REPLAY.md §3.1]: tag-side disagreement about
#: direction, the operator's 19:29:18 complaint.
EPISODE_KINDS = frozenset(("difference", "fit", "scale"))
#: A disagreement HOLD, once its own condition has cleared, releases only
#: when the bucket is below this: a reversed axis or a scale error stops
#: showing whenever the vehicle slows or leaves that axis, and a hold that
#: released then flew the wrong velocity at full weight again (reversed
#: sway, 5 cm/s oscillation: DVL weight 5.58 -> 1.03 s [측정 DTF
#: design/out/synth_v2.json]). = EPISODE_MIN_S, so "a hold sticks" and "an
#: episode counts" are one level; release at 0 put a station on a 73 cm/s
#: tag-FD spike for 0.5 s [측정 DTF design/out/selection.json NEW-Dz].
HOLD_DRAIN_S = EPISODE_MIN_S

#: The usage-reason CODE vocabulary (CSV dvl_code; DTF SPEC.md §6.3): the
#: data gates by the phrase before " (" ...
GATE_CODES = (("no DVL data", "no_data"), ("DVL has no bottom lock", "no_lock"),
              ("DVL is in water-tracking mode", "water_track"),
              ("DVL report lacks fom or altitude", "no_fom_alt"),
              ("DVL fom too high", "fom_max"),
              ("DVL altitude too low", "low_alt"),
              ("DVL velocity implausible", "implausible"),
              ("DVL velocity jump", "jump"),
              ("no body rate for the DVL lever arm", "no_rate"),
              ("DVL stale", "stale"))
#: ...the R3 quality flags, the disagreement holds...
FLAG_CODES = ("flag_fom", "flag_alt", "flag_beams", "flag_reacq")
DIS_CODES = ("dis_diff", "dis_align", "dis_fit")
#: ...and the codes the station bridge's DVL rung carries on its last usable
#: sample for blend_hold_s (the blend's own hold, on the rung): a data gate
#: or a flag. A latch, a disagreement, OUT or not armed refuse at once.
RUNG_BRIDGE_CODES = frozenset([c for _p, c in GATE_CODES] + list(FLAG_CODES))
#: Phrase -> code for everything else a tick can be told, the legacy
#: ladder's words included (the meta counts rows by the phrase).
_REASON_PHRASES = (
    ("DVL is OUT (shadow)", "out"),
    ("DVL latched OUT (", "latched"),
    ("DVL tripped before engage", "not_armed"),
    ("DVL not armed at engage", "not_armed"),
    ("DVL tripped (", "latched"),
    ("DVL mount not measured", "no_mount"),
    ("DVL not proven against the tag velocity", "unproven"),
    ("DVL velocity not finite", "implausible"),
    ("DVL code fault", "fault"),
    ("DVL quality flag: fom", "flag_fom"),
    ("DVL quality flag: altitude", "flag_alt"),
    ("DVL quality flag: beams", "flag_beams"),
    ("DVL quality flag: lock re-acquired", "flag_reacq"),
    ("DVL disagrees with the tag velocity: alignment", "dis_align"),
    ("DVL disagrees with the tag velocity: the fit", "dis_fit"),
    ("DVL disagrees with the tag velocity", "dis_diff"),
    ("DVL disagreement releasing", "dis_diff"),
    ("DVL gross", "gross"),
    ("no tag state", "no_tag"),
    ("station bridge on tier", "bridge"),
    ("DVL re-entry", "reentry"),
    ("DVL innovation hold", "innov_hold"),
    ("DVL innovation (", "innovation"),
    ("DVL unchecked", "unchecked"),
    ("DVL held back", "refute_hold"),
    ("DVL alignment hold", "align_hold"),
)


def reason_code(why) -> str:
    """A tick's reason (the phrase before " (" + numbers) -> its CSV
    ``dvl_code``: "ok" for none, "other" for a phrase nobody mapped (a test
    pins that every reason the ladders and the worker give has a code)."""
    w = str(why or "")
    if not w:
        return "ok"
    if w.startswith("held: "):                   # the rung's hold (rung_velocity)
        return reason_code(w[len("held: "):])
    for phrase, code in _REASON_PHRASES:
        if w.startswith(phrase):
            return code
    for phrase, code in GATE_CODES:
        if w.startswith(phrase):
            return code
    return "other"


def reason_class(code) -> str:
    """A usage code -> the panel chip's short word ("" = nothing to say)."""
    c = str(code or "")
    if c in ("", "ok", "rung"):
        return ""
    if c.startswith("dis_"):
        return "disagrees"
    return {"flag_fom": "flag fom", "flag_alt": "flag altitude",
            "flag_beams": "flag beams", "flag_reacq": "flag re-lock",
            "gross": "gross", "no_lock": "no lock", "stale": "stale",
            "latched": "latched", "not_armed": "not this run",
            "fault": "code fault", "no_tag": "no tag state",
            "unproven": "not proven", "no_mount": "no mount",
            "out": "OUT"}.get(c, c.replace("_", " "))


def resolve(raw: dict | None) -> dict:
    """Merge a ``dvl:`` block over the defaults, rejecting typos.

    The station_bridge / imu_dr rule: a silently ignored key here means the
    operator flies believing a gate is armed when the default is."""
    cfg = dict(DEFAULTS)
    if raw:
        unknown = set(raw) - set(KEYS)
        if unknown:
            raise ValueError(f"unknown dvl keys {sorted(unknown)}; "
                             f"known: {sorted(KEYS)}")
        cfg.update(raw)
    for k in ("enabled", "use", "require_proven", "flag_reacquired"):
        if not isinstance(cfg[k], bool):
            raise ValueError(f"dvl.{k} must be true or false, got {cfg[k]!r}")
    sel = cfg["selection"]
    if not isinstance(sel, str) or sel not in SELECTIONS:
        raise ValueError(f"dvl.selection must be 'primary' or 'legacy', "
                         f"got {sel!r}")
    cfg["host"] = str(cfg["host"])
    port = cfg["port"]
    if isinstance(port, bool) or not isinstance(port, int) \
            or not 1 <= port <= 65535:
        raise ValueError(f"dvl.port must be an integer 1..65535, got {port!r}")
    for k in ("connect_timeout_s", "reconnect_s", "stale_s", "fom_max",
              "v_max_m_s", "a_max_m_s2", "innov_max_m_s",
              "agree_min_excitation_m_s", "agree_tol", "agree_v_min_m_s",
              "agree_window_s", "agree_trip_m_s", "agree_trip_s",
              "agree_ratio_v_min_m_s"):
        cfg[k] = _pos(k, cfg[k])
    for k in ("min_altitude_m", "reentry_s", "config_poll_s"):
        cfg[k] = _pos(k, cfg[k], allow_zero=True)
    cfg["agree_ratio_min"] = _pos("agree_ratio_min", cfg["agree_ratio_min"])
    if cfg["agree_ratio_min"] >= 1.0:
        raise ValueError("dvl.agree_ratio_min must be below 1 (a perfect "
                         "mount reads 1)")
    for k, lo in (("agree_min_pairs", 4), ("agree_refute_n", 1)):
        n = cfg[k]
        if isinstance(n, bool) or not isinstance(n, int) or n < lo:
            raise ValueError(f"dvl.{k} must be an integer >= {lo}, got {n!r}")
    # ---- the 2026-10-16 keys (selection primary)
    for k in ("blend_hold_s", "blend_xfade_s", "blend_gross_tau_s",
              "flag_alt_window_s", "agree_slow_s", "agree_slow_clip_m_s",
              "agree_hold_m_s", "agree_release_m_s", "esc_latch_s",
              "esc_scale_v_min_m_s", "esc_repeat_window_s"):
        cfg[k] = _pos(k, cfg[k])
    for k in ("blend_gross_m_s", "flag_fom_m_s", "flag_alt_jump_m",
              "esc_leak", "esc_scale_ratio"):          # 0 = off / no leak
        cfg[k] = _pos(k, cfg[k], allow_zero=True)
    nb = cfg["flag_min_beams"]
    if isinstance(nb, bool) or not isinstance(nb, int) or not 0 <= nb <= 4:
        raise ValueError(f"dvl.flag_min_beams must be an integer 0..4 (0 = "
                         f"off; the A50 has 4 beams), got {nb!r}")
    n = cfg["esc_repeat_n"]
    if isinstance(n, bool) or not isinstance(n, int) or (n != 0 and n < 2):
        raise ValueError(f"dvl.esc_repeat_n must be 0 (off) or an integer "
                         f">= 2, got {n!r}")
    if cfg["agree_release_m_s"] >= cfg["agree_hold_m_s"]:
        raise ValueError(f"dvl.agree_release_m_s {cfg['agree_release_m_s']!r} "
                         f"must be below dvl.agree_hold_m_s "
                         f"{cfg['agree_hold_m_s']!r} (a hold must be able to "
                         f"end)")
    if 0.0 < cfg["esc_scale_ratio"] <= 1.0:
        raise ValueError(f"dvl.esc_scale_ratio must be 0 (off) or above 1 (the "
                         f"DVL reading MORE motion than the tags), got "
                         f"{cfg['esc_scale_ratio']!r}")
    for k, what in (("mount_rpy_deg", "[roll, pitch, yaw] degrees"),
                    ("mount_t_flu", "[x, y, z] metres, body FLU")):
        v = cfg[k]
        if v is None:
            continue
        try:
            a = np.asarray(v, float).reshape(-1)
        except (TypeError, ValueError):
            a = np.zeros(0)
        if a.shape != (3,) or not np.all(np.isfinite(a)):
            raise ValueError(f"dvl.{k} must be null or {what}, got {v!r}")
        cfg[k] = [float(x) for x in a]
    return cfg


def resolve_with_opts(raw: dict | None, opts) -> dict:
    """``resolve`` plus the command-line overrides (``--dvl``, ``--dvl-host``,
    ``--dvl-port``). ONE function for the two readers — MpcWorker (the
    estimator side) and the backend's TCP client — so they cannot disagree
    about whether a DVL exists or where it is."""
    r = dict(raw or {})
    if bool(getattr(opts, "dvl", False)):
        r["enabled"] = True
    host = getattr(opts, "dvl_host", None)
    if host:
        r["host"] = str(host)
    port = getattr(opts, "dvl_port", None)
    if port:
        r["port"] = int(port)
    return resolve(r)


def _pos(key: str, v, allow_zero: bool = False) -> float:
    """A limit that is zero, negative or NaN is not a limit (the engage
    block's rule)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        f = float("nan")
    if isinstance(v, bool) or not math.isfinite(f) \
            or (f < 0.0 if allow_zero else f <= 0.0):
        raise ValueError(f"dvl.{key} must be a finite number "
                         f"{'>= 0' if allow_zero else '> 0'}, got {v!r}")
    return f


# =============================================================================
# the wire: one JSON object per line
# =============================================================================
def parse_line(line) -> dict | None:
    """One line of the TCP stream -> the report dict, or None.

    The stream interleaves three kinds, told apart by ``type``: ``velocity``
    (the one this station uses), ``position_local`` (the device's own dead
    reckoning — recorded raw, never consumed: it integrates the device's IMU
    in the device's frame and knows nothing about the tag world) and
    ``response`` (the answer to a command). Lines end ``\\r\\n`` on the wire
    (vendor demo unit capture, rov_gui/tests/fixtures/dvl_a50_demo_capture_
    20261001.jsonl); a torn or non-JSON line is None, never an exception —
    one bad line must not end the reader."""
    if isinstance(line, (bytes, bytearray)):
        try:
            line = line.decode("utf-8")
        except UnicodeDecodeError:
            return None
    line = str(line).strip()
    if not line or line[0] != "{":
        return None
    try:
        d = json.loads(line)
    except (ValueError, RecursionError):
        # RecursionError: a deeply nested line. Not something the device
        # sends — but the reader thread must outlive whatever arrives.
        return None
    return d if isinstance(d, dict) and "type" in d else None


def _finite(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError):
        return None
    return f if math.isfinite(f) else None


def _int(v) -> int | None:
    """An integer stamp/flag off the wire, or None for anything that is not
    one (``Infinity``, a string, a list...)."""
    if isinstance(v, bool):
        return None
    try:
        return int(v)
    except (TypeError, ValueError, OverflowError):
        return None


class DeviceClock:
    """Device stamps -> the station clock, without synchronising anything.

    ``offset`` is the SMALLEST ``t_rx - time_of_transmission`` seen in the
    last ``window_s``: the one report that crossed the network fastest. A
    stamp then maps as ``device_time + offset``. Against the simpler
    ``t_rx - (tot - tov)`` this matters in exactly one case, and it is the
    dangerous one: after a TCP stall the backlog arrives in one burst, every
    line with the same arrival time, and the simple rule would stamp
    seconds-old velocities as fresh. Here they keep their true age and the
    staleness gate sees them for what they are.

    The device boots at an arbitrary time and may be stepped by NTP mid-run
    [스펙: docs.waterlinked.com/dvl/time]; a jump of more than ``step_s`` in
    the offset starts the estimate over (``n_steps`` counts them).
    """

    def __init__(self, window_s: float = 30.0, step_s: float = 10.0):
        self.window_s = float(window_s)
        self.step_s = float(step_s)
        self._d: deque = deque()            # (t_rx, t_rx - tot)
        self.offset: float | None = None
        self.n_steps = 0

    def note(self, t_rx: float, tot_us: int) -> None:
        d = float(t_rx) - float(tot_us) * 1e-6
        if self.offset is not None and abs(d - self.offset) > self.step_s:
            self._d.clear()
            self.n_steps += 1
        self._d.append((float(t_rx), d))
        while self._d and t_rx - self._d[0][0] > self.window_s:
            self._d.popleft()
        self.offset = min(x[1] for x in self._d)

    def to_station(self, dev_us: int) -> float | None:
        return (None if self.offset is None
                else float(dev_us) * 1e-6 + self.offset)


def sample_fields(rep: dict, t_rx: float, clock: DeviceClock | None = None
                  ) -> dict | None:
    """A ``velocity`` report -> the keyword arguments of ``state.DvlSample``
    (a dict, so this module needs no rov_gui.state import and the client can
    build the dataclass where it emits). None for any other report type.

    Field meanings [스펙: docs.waterlinked.com/dvl/dvl-json-protocol, read
    2026-10-01]: vx/vy/vz m/s in the device output frame; ``fom`` m/s;
    ``altitude`` m along Z; ``velocity_valid`` = the device has a lock;
    ``time_of_validity`` / ``time_of_transmission`` Unix MICROSECONDS on the
    device's own clock; ``time`` ms since the previous report.

    ``t_valid`` — when the velocity was TRUE, on the station clock — comes
    from ``clock`` when one is given (see :class:`DeviceClock`), else from
    ``t_rx - (tot - tov)``. Either way it never lands after ``t_rx``.

    A report that claims validity with a non-finite velocity is demoted to
    invalid here: nothing downstream may ever see a NaN that says "valid".
    Never raises, whatever the line held."""
    kind = str(rep.get("type", ""))
    if kind not in ("velocity", "velocity_water"):
        return None
    v = tuple(_finite(rep.get(k)) for k in ("vx", "vy", "vz"))
    ok_v = all(x is not None for x in v)
    note = ""
    tov, tot = _int(rep.get("time_of_validity")), _int(
        rep.get("time_of_transmission"))
    t_valid = float(t_rx)
    if tov is not None and tot is not None \
            and max(abs(tov), abs(tot)) > 10 ** 18:
        # Microseconds: 1e18 is the year 33658. json parses a 400-digit
        # integer happily, and float() of one raises.
        tov = tot = None
    if tov is None or tot is None:
        tov = tot = None
        note = "no device stamps — t_valid = arrival"
    else:
        lat_us = tot - tov
        if not 0 <= lat_us <= 2_000_000:
            # Garbage, or the device clock stepped between the two stamps.
            # Arrival time is the honest fallback; the age gate still
            # applies to it.
            note = (f"device latency {lat_us} us out of range — "
                    f"t_valid = arrival")
        else:
            t_valid = float(t_rx) - lat_us * 1e-6
            if clock is not None:
                clock.note(t_rx, tot)
                tc = clock.to_station(tov)
                if tc is not None:
                    t_valid = min(float(t_rx), tc)
    cov = rep.get("covariance")
    try:
        cov_t = tuple(float(x) for row in cov for x in row)
        if len(cov_t) != 9 or not all(math.isfinite(x) for x in cov_t):
            cov_t = None
    except (TypeError, ValueError, OverflowError):
        cov_t = None
    n_beams = 0
    beams = rep.get("transducers")
    if isinstance(beams, (list, tuple)):
        n_beams = sum(1 for b in beams
                      if isinstance(b, dict) and b.get("beam_valid") is True)
    status = _int(rep.get("status")) or 0
    return {
        "t_valid": t_valid, "t_rx": float(t_rx),
        "v_dvl": (v if ok_v else None),
        "valid": bool(rep.get("velocity_valid") is True and ok_v),
        "fom": _finite(rep.get("fom")),
        "altitude": _finite(rep.get("altitude")),
        "covariance": cov_t, "status": status, "n_beams": int(n_beams),
        "dt_ms": _finite(rep.get("time")),
        "tov_us": tov, "tot_us": tot,
        "kind": kind, "note": note,
    }


# =============================================================================
# geometry
# =============================================================================
def mount_from_cfg(cfg: dict) -> tuple:
    """-> (R_frd_dvl 3x3, r_frd 3, measured: bool).

    ``measured`` is False while either mount key is null; the rotation is
    then identity and the lever arm zero, which is what SHADOW runs on (so
    the agreement fit can measure the real rotation) and what IN refuses."""
    rpy, t = cfg.get("mount_rpy_deg"), cfg.get("mount_t_flu")
    measured = rpy is not None and t is not None
    R = np.eye(3) if rpy is None else rot_zyx(*(math.radians(float(a))
                                                for a in rpy))
    r = np.zeros(3) if t is None else S_FLU_FRD @ np.asarray(t, float)
    return R, r, measured


def body_velocity(v_dvl, R_frd_dvl, r_frd, omega) -> np.ndarray:
    """Velocity of the BODY ORIGIN in body FRD, from the DVL's reading.

        v_dvl_point = v_origin + omega x r     (rigid body)
        v_origin    = R_frd_dvl @ v_dvl - omega x r

    ``omega`` is the body rate (p, q, r) rad/s AT THE SAMPLE'S OWN TIME, or
    None for no correction. At 0.3 rad/s of yaw and a 0.2 m arm the
    correction is 6 cm/s — more than this vehicle's cruise speed in some
    missions, so it is not optional; and a rate taken 0.19 s late (the
    report's latency) leaves centimetres per second of it behind during a
    turn, which is why the caller looks the rate up in its history."""
    v = np.asarray(R_frd_dvl, float) @ np.asarray(v_dvl, float).reshape(3)
    if omega is not None:
        v = v - np.cross(np.asarray(omega, float).reshape(3),
                         np.asarray(r_frd, float).reshape(3))
    return v


def integrate_xy(p_xy, v_body, roll: float, pitch: float, yaw: float,
                 dt: float) -> np.ndarray:
    """One dead-reckoning step of the horizontal position (NED x, y)."""
    v_w = rot_zyx(roll, pitch, yaw) @ np.asarray(v_body, float).reshape(3)
    return np.asarray(p_xy, float).reshape(2) + v_w[:2] * float(dt)


# =============================================================================
# agreement: is the DVL describing the same motion the tags are?
# =============================================================================
class Agreement:
    """Least squares ``v_fd ~= A @ v_dvl`` over (surge, sway) pairs.

    The DVL is the regressor because it is the cleaner of the two: noise on
    the regressor biases the slope toward zero (errors in variables), and a
    raw two-point tag difference carries ~17 mm/s of it (0.8 mm at 15 Hz)
    against the DVL's few mm/s. Pairs are admitted only above ``v_min`` of
    DVL speed for the same reason — a pile of station-keeping samples would
    otherwise drag every slope to ~0.7 and call a good mount bad (seen in
    the 2026-10-01 synthetic prototype).

    ``v_dvl`` here is the BODY velocity computed with the CONFIGURED mount.
    If the device really sits rotated by D more (clockwise seen from above,
    i.e. about +z down) than ``mount_rpy_deg`` says, and reads s times the
    true speed, then v_dvl = s Rot(-D) v_true and A = Rot(D) / s, so

        mount_yaw_fix_deg = atan2(A10 - A01, A00 + A11)   = D: ADD this to
                                                          mount_rpy_deg yaw
        scale             = 1 / sqrt(det A)               (DVL reads this x true)

    and det A < 0 is a mirrored axis — a frame-convention mistake, not a
    rotation.

    THE VERDICT IS A THREE-WAY ONE, with a standard error. ``dev`` is the
    largest element of |A - I| and ``sigma`` the standard error of an element
    of A (residual scatter over sqrt(n) times the weaker axis's rms speed):

        proven        dev + 3 sigma <= tol     it IS within tolerance
        refuted       dev - 3 sigma >  tol     it is NOT
        insufficient  otherwise — too few pairs, motion along one body axis
                      only, or simply not enough data yet to say which

    A refutation also says whether it is ``coherent``: whether the tags see
    the DVL's motion at all (the smaller singular value of A, ``gain_min``,
    at least COHERENT_MIN). A wrong mount is a coherent refutation. An
    incoherent one — the DVL reports motion the tags do not show — is noise
    or a false lock: it still keeps the switch from going IN, but it is not
    evidence about the mount and does not trip.

    so a mount is neither proven nor refuted by a lucky or unlucky handful
    of noisy pairs (review 2026-10-02: the point estimate alone proved a
    mount at the minimum pair count on 1 cm tag noise).

    What it does NOT see: the vertical axis, a tilted bracket (a pitch or
    roll error of e degrees only scales one row by cos e) and the lever arm
    (its error is orthogonal to the regressor unless the vehicle arcs).
    Those rest on the tape measure.

    ``lag_s`` is a commissioning number, not a gate: how much LATER the tag
    velocity describes the motion than the DVL velocity carrying the same
    stamp (v_tag(t) = v_dvl(t - lag)), from the residual's slope on the
    DVL's acceleration. Zero when both sensors stamp their data with the
    moment it was true. It matters because everything here pairs the two by
    their stamps: a lag of L seconds on a velocity that turns at w rad/s
    looks like a mount rotated by w L (simulation 2026-10-02: 50 ms at 1 Hz
    reads 16 degrees). The DVL acceleration is a two-sample difference and
    noisy, which biases the estimate toward zero — read it as a floor.
    """

    def __init__(self, min_pairs: int, min_excitation: float, tol: float,
                 v_min: float, window_s: float):
        self.min_pairs = int(min_pairs)
        self.min_excitation = float(min_excitation)
        self.tol = float(tol)
        self.v_min = float(v_min)
        self.window_s = float(window_s)
        self._pairs: deque = deque()        # (t, vd0, vd1, vf0, vf1, ad0, ad1)
        self.n_seen = 0                     # pairs offered, incl. too slow
        self._cache = None

    def clear(self) -> None:
        self._pairs.clear()
        self._cache = None

    def add(self, t: float, v_dvl_xy, v_fd_xy, a_dvl_xy=None) -> bool:
        """``a_dvl_xy`` = the DVL's acceleration at the pair (for ``lag_s``
        only; None = unknown)."""
        self.n_seen += 1
        vd = np.asarray(v_dvl_xy, float).reshape(2)
        vf = np.asarray(v_fd_xy, float).reshape(2)
        ad = (np.full(2, np.nan) if a_dvl_xy is None
              else np.asarray(a_dvl_xy, float).reshape(2))
        while self._pairs and t - self._pairs[0][0] > self.window_s:
            self._pairs.popleft()
            self._cache = None
        if not (np.all(np.isfinite(vd)) and np.all(np.isfinite(vf))):
            return False
        if float(np.hypot(vd[0], vd[1])) < self.v_min:
            return False
        self._pairs.append((float(t), vd[0], vd[1], vf[0], vf[1],
                            ad[0], ad[1]))
        self._cache = None
        return True

    @property
    def n(self) -> int:
        return len(self._pairs)

    def fit(self) -> dict:
        """The current verdict, with every number that led to it."""
        if self._cache is not None:
            return self._cache
        out = {"n": self.n, "verdict": "insufficient", "why": "",
               "A": None, "excitation_m_s": None, "mount_yaw_fix_deg": None,
               "scale": None, "mirrored": None, "rms_m_s": None,
               "sigma_A": None, "dev": None, "n_outliers": 0,
               "lag_s": None, "lag_sigma_s": None, "gain_min": None,
               "coherent": None}
        self._cache = out
        if self.n < self.min_pairs:
            out["why"] = (f"{self.n}/{self.min_pairs} moving pairs — fly a "
                          f"short pass in surge AND sway with tags in view")
            return out
        P = np.asarray(self._pairs, float)
        vd, vf, acc = P[:, 1:3], P[:, 3:5], P[:, 5:7]
        A = res = exc = None
        for attempt in (0, 1):
            S_dd = vd.T @ vd
            lam = np.linalg.eigvalsh(S_dd / len(vd))
            exc = np.sqrt(np.maximum(lam, 0.0))
            out["excitation_m_s"] = [float(exc[0]), float(exc[1])]
            if exc[0] < self.min_excitation:
                out["why"] = (f"motion along one body axis only (weak axis "
                              f"{exc[0] * 100:.1f} cm/s rms < "
                              f"{self.min_excitation * 100:.1f}) — add a "
                              f"sideways or a forward pass")
                return out
            A = (vf.T @ vd) @ np.linalg.inv(S_dd)
            res = vf - vd @ A.T
            if attempt:
                break
            # ONE trimmed refit. A tag fix that jumps (a re-acquired or
            # mis-solved frame) is a finite difference of metres per second;
            # a single such pair outweighs a hundred honest ones at
            # 10 cm/s. Pairs further than 4x the median residual (and at
            # least 3 cm/s) from the first fit are set aside — which leaves
            # a genuinely wrong mount untouched: a rotation or a mirror is
            # EXPLAINED by the linear map, so its residuals are small.
            r = np.hypot(res[:, 0], res[:, 1])
            keep = r <= max(4.0 * float(np.median(r)), 0.03)
            n_drop = int(np.count_nonzero(~keep))
            if not n_drop:
                break
            out["n_outliers"] = n_drop
            if int(np.count_nonzero(keep)) < self.min_pairs:
                # The outliers ARE in the first fit, so it must not be
                # judged: with them it can call a good mount "rotated 27
                # degrees" the moment the window shrinks to the minimum
                # (review 2026-10-02). Wait for more pairs instead.
                out["why"] = (f"{int(np.count_nonzero(keep))}/"
                              f"{self.min_pairs} pairs after setting aside "
                              f"{n_drop} outlier(s) — keep moving")
                return out
            vd, vf, acc = vd[keep], vf[keep], acc[keep]
        n_used = len(vd)
        det = float(np.linalg.det(A))
        rms = float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))))
        has = np.all(np.isfinite(acc), axis=1)
        if int(np.count_nonzero(has)) >= self.min_pairs:
            # v_tag(t) = A v_dvl(t - lag) ~ A v_dvl(t) - lag * A a_dvl(t)
            g = acc[has] @ A.T
            gg = float(np.sum(g * g))
            if gg > 1e-9:
                out["lag_s"] = float(-np.sum(res[has] * g) / gg)
                out["lag_sigma_s"] = float((rms / math.sqrt(2.0))
                                           / math.sqrt(gg))
        # Standard error of one element of A: per-axis residual scatter over
        # sqrt(n) x the rms speed along the weaker excited direction.
        sigma = (rms / math.sqrt(2.0)) / (math.sqrt(n_used)
                                          * max(float(exc[0]), 1e-9))
        dev = float(np.max(np.abs(A - np.eye(2))))
        # How much of the DVL's motion the tags see along the direction they
        # see LEAST of it (the smaller singular value of A). ~1 for any
        # rotation or mirror; ~0 when the DVL "moves" and the tags do not.
        gain_min = float(np.linalg.svd(A, compute_uv=False)[-1])
        out["gain_min"] = gain_min
        out["coherent"] = bool(gain_min >= COHERENT_MIN)
        out["A"] = [[float(A[0, 0]), float(A[0, 1])],
                    [float(A[1, 0]), float(A[1, 1])]]
        out["rms_m_s"] = rms
        out["sigma_A"] = float(sigma)
        out["dev"] = dev
        out["mirrored"] = bool(det < 0.0)
        out["mount_yaw_fix_deg"] = float(math.degrees(
            math.atan2(A[1, 0] - A[0, 1], A[0, 0] + A[1, 1])))
        out["scale"] = (float(1.0 / math.sqrt(det)) if det > 1e-9 else None)
        if dev + 3.0 * sigma <= self.tol:
            out["verdict"] = "proven"
        elif dev - 3.0 * sigma > self.tol:
            out["verdict"] = "refuted"
            if not out["coherent"]:
                what = (f"the tag velocity shows only {gain_min:.2f} of the "
                        f"motion the DVL reports along one direction — DVL "
                        f"noise, a false bottom lock or a gross scale "
                        f"error; this says nothing about the mount")
            elif det < 0.0:
                what = ("one axis is MIRRORED — a frame-convention error "
                        "(is the device upside down, or y taken as port?)")
            elif abs(out["mount_yaw_fix_deg"]) > 10.0:
                what = (f"the DVL looks mounted "
                        f"{out['mount_yaw_fix_deg']:+.0f} deg (clockwise from "
                        f"above) away from dvl.mount_rpy_deg yaw — add that "
                        f"to it, or set the device's mounting rotation "
                        f"offset, not both")
            else:
                what = (f"scale {out['scale']:.2f}x — the DVL reads that "
                        f"much more (or less) motion than the tags: a scale "
                        f"error, or motion only the DVL reports part of the "
                        f"time (repeated false locks)" if out["scale"]
                        else "no consistent map")
            out["why"] = (f"DVL and tag velocity disagree "
                          f"(max |A - I| {dev:.2f} > {self.tol:.2f}): {what}")
        else:
            out["why"] = (f"not decided yet: max |A - I| {dev:.2f} +- "
                          f"{3.0 * sigma:.2f} against {self.tol:.2f} — keep "
                          f"moving (more pairs, or faster)")
        return out


# =============================================================================
# the DVL-primary blend (2026-10-16, selection primary)
# =============================================================================
class DvlBlend:
    """The velocity blend the pwm11att13 policy was trained on — RL_controller
    deploy/obs_builder.py ``DvlBlendNp`` (= rov_rl/velocity_source.py
    VelocitySelector, mode dvl_primary), transcribed line for line [스펙];
    plus two additions that are no-ops on the twin's own traces (DTF
    design/out/check_proto.json, test_dvl N1):

        expire_hold()   a disagreement / latch makes the held sample
                        ineligible, so the fade starts on that tick (the
                        twin's own law for a held sample older than hold_s)
                        instead of re-flying the faulty sample for 0.45 s
        step(n_skip=)   the ticks that had no tag state (the station does not
                        step the blend without one) are replayed as fade
                        steps toward the current target; tag_lp does not
                        move across them

    Per tick: the gross check's reference tag_lp is a 1-s EMA of THIS tick's
    tag finite difference (a = dt / gross_tau_s); ``ok`` = the newest sample
    is usable, fresh (stale_s on its capture time) and within gross_m_s of
    tag_lp. Then the last ok sample is flown for hold_s of its capture time
    (src "hold"), and the weight w moves dt / xfade_s per tick toward its
    target (1 or 0; it snaps to the first target after reset(), at engage).
    u, v = w * v_dvl + (1 - w) * fd; heave stays the tag's (dvl_w false).
    Frames: the station feeds FRD; the step is linear and its only norm is of
    a (u, v) difference, so FRD in gives the FRD image of the FLU result
    [유도]. Sources: "dvl", "hold", "blend" (0 < w < 1), "tag_fd" (w = 0)."""

    def __init__(self, dt=0.05, hold_s=0.45, xfade_s=0.5, gross_m_s=0.5,
                 gross_tau_s=1.0, stale_s=0.6, dvl_w=False, snap_tol=1e-6):
        self.dt, self.hold_s, self.xfade_s = float(dt), float(hold_s), float(xfade_s)
        self.gross, self.gross_tau = float(gross_m_s), float(gross_tau_s)
        self.stale_s = float(stale_s)
        self.dvl_w, self.snap_tol = bool(dvl_w), float(snap_tol)
        self.reset()

    def reset(self) -> None:
        """At engage, together with the assembler's reset."""
        self.w = 0.0
        self.v_hold = None
        self.t_hold = None
        self.tag_lp = None
        self.snap = True
        self.src = "tag_fd"
        self.target = 0.0                   # (logging only)
        self.gross_rejected = False         # (logging only)

    def expire_hold(self) -> None:
        """The held sample is no longer eligible: the next not-ok step starts
        the fade at once."""
        if self.t_hold is not None:
            self.t_hold = -math.inf

    def step(self, t_now, fd_b, dvl_v=None, dvl_t_valid=None, dvl_ok=False,
             n_skip=0):
        """-> (v_b (3,), w). ``fd_b`` = this tick's tag finite difference
        (body); ``dvl_v`` / ``dvl_t_valid`` = the NEWEST sample (passed every
        tick until a newer one replaces it); ``dvl_ok`` = the station allows
        it; ``n_skip`` = ticks without a tag state since the last step."""
        fd = np.asarray(fd_b, float).reshape(3)
        a = min(1.0, self.dt / self.gross_tau)
        self.tag_lp = (fd[:2].copy() if self.tag_lp is None
                       else self.tag_lp + a * (fd[:2] - self.tag_lp))
        ok = (bool(dvl_ok) and dvl_v is not None and dvl_t_valid is not None
              and (t_now - float(dvl_t_valid) <= self.stale_s))
        v = None if dvl_v is None else np.asarray(dvl_v, float).reshape(3)
        self.gross_rejected = False
        if ok and self.gross > 0.0 and float(
                np.hypot(*(v[:2] - self.tag_lp))) > self.gross:
            ok = False
            self.gross_rejected = True
        if ok:
            self.v_hold, self.t_hold = v.copy(), float(dvl_t_valid)
            target, self.src = 1.0, "dvl"
        elif self.v_hold is not None and t_now - self.t_hold <= self.hold_s:
            target, self.src = 1.0, "hold"
        else:
            target, self.src = 0.0, "tag_fd"
        self.target = target
        v_src = self.v_hold if self.v_hold is not None else fd
        if self.snap:
            self.w, self.snap = target, False
        stp = self.dt / self.xfade_s
        # The skipped ticks replay only until w can have reached the target:
        # the target is fixed here, so every step after that changes nothing
        # (bit-identical) — unbounded, an hour without tags while disengaged
        # (72 000 ticks) cost one 2-3 ms tick [측정 DTF fix1, review F1].
        n_rep = 1 + min(max(0, int(n_skip)), int(math.ceil(1.0 / stp)) + 1)
        for _ in range(n_rep):
            # the trained per-tick law, once per tick — the skipped ones too
            if abs(target - self.w) <= stp + self.snap_tol:
                self.w = target
            else:
                self.w = self.w + stp if target > self.w else self.w - stp
        if self.w <= 0.0:
            return fd.copy(), self.w
        out = self.w * v_src + (1.0 - self.w) * fd
        if not self.dvl_w:
            out[2] = fd[2]
        if 0.0 < self.w < 1.0:
            self.src = "blend"
        return out, self.w


# =============================================================================
# the estimator-side object
# =============================================================================
class DvlAid:
    """Everything between ``bus.dvl`` and ``StateAssembler.step(vel_xy=)``.

    One instance per worker, for the life of the process: the proof of the
    mount is a property of the SESSION (the hardware does not change between
    engagements), so it survives disengage. ``set_use`` is the operator's
    switch; the caller enforces "only while disengaged".

    ``fd_lp_gain`` is how much the assembler's low-pass shrinks the raw
    tag-difference noise: alpha / sqrt(2 - alpha) for its recursive filter
    acting on a two-point DIFFERENCE of white position noise (0.507 at
    alpha 0.6 — not the white-input sqrt(alpha / (2 - alpha)) = 0.655,
    review 2026-10-02). It turns the proof's residual scatter into the
    noise of the velocity the innovation gate compares against.

    ``dt`` (2026-10-16) = the control period the blend steps at (the worker
    passes 1 / ctrl_hz; the trained export is 0.05 s [스펙]). Under
    ``selection: primary`` the tick asks :meth:`usable` and
    :meth:`blend_step`, the bridge rung :meth:`rung_admit` /
    :meth:`rung_velocity`; :meth:`velocity`, :meth:`displacement` and
    :meth:`note_tag_velocity` serve ``legacy`` only (unchanged).
    """

    def __init__(self, cfg: dict | None = None, fd_lp_gain: float = 0.507,
                 dt: float = 0.05):
        self.cfg = resolve(cfg)
        self.R, self.r, self.mount_measured = mount_from_cfg(self.cfg)
        self.fd_lp_gain = float(fd_lp_gain)
        self.use = bool(self.cfg["use"])
        self.agree = Agreement(self.cfg["agree_min_pairs"],
                               self.cfg["agree_min_excitation_m_s"],
                               self.cfg["agree_tol"],
                               self.cfg["agree_v_min_m_s"],
                               self.cfg["agree_window_s"])
        self.proven = False
        self.proof: dict | None = None      # the fit that proved it
        #: WHY the DVL was taken out on evidence ("" = not tripped). Latched
        #: until the operator sets the switch IN again.
        self.tripped = ""
        #: What this session's data has said AGAINST the DVL ("" = nothing).
        #: Stickier than ``tripped``: set by every trip and by a refuting fit
        #: seen in shadow, cleared only by a proof made afterwards. While it
        #: is set, IN needs that proof whatever ``require_proven`` says.
        self.against = ""
        self.n_trips = 0
        self.last_trip: str = ""
        self.device_config: dict | None = None   # newest get_config answer
        self.device_config_changes: list = []    # what changed, and to what
        self.source = ""
        self._s = None                      # newest state.DvlSample
        self._s_ok = False                  # ...did it pass the data gates
        self._s_why = "no DVL data"
        self._v_body = None                 # ...its body-origin velocity
        self._hist: deque = deque(maxlen=128)    # (t_valid, v_body, ok)
        self._last_good = None              # (t_valid, v_body) last good one
        self._good_since: float | None = None
        self._prev_fix = None               # (t_capture, p_ned)
        self._fd_pending: deque = deque(maxlen=32)   # (t_mid, v_fd_body)
        #: (t, v_fd_xy, tag_age, fix_t): the newest tick's tag velocity and
        #: the capture time of the camera frame it was made from
        self._fd_ref = None
        #: Why the innovation gate is holding the DVL out ("" = it is not):
        #: set by a rejection, cleared only by an acceptance against a tag
        #: velocity made from a NEWER frame than the newest rejecting one
        #: (``_innov_hold_fix``).
        self._innov_hold = ""
        self._innov_hold_fix: float | None = None
        #: t_valid of the newest sample the innovation gate rejected: no
        #: integral (displacement) may use it or anything before it.
        self._innov_reject_t: float | None = None
        self._refuted_run = 0
        self._refuted_t: float | None = None    # newest refuting evaluation
        self.n_gross = 0                    # pairs kept out of the monitors
        self.n_monitor = 0                  # aligned pairs the monitors saw
        # ---- 2026-10-16: selection (module doc). The slow monitors are
        # computed in BOTH modes (recorded); everything else below acts in
        # primary only, and its counters stay 0 under legacy.
        self.selection = str(self.cfg["selection"])
        self.dt = float(dt)
        c = self.cfg
        self.blend = DvlBlend(self.dt, hold_s=c["blend_hold_s"],
                              xfade_s=c["blend_xfade_s"],
                              gross_m_s=c["blend_gross_m_s"],
                              gross_tau_s=c["blend_gross_tau_s"],
                              stale_s=c["stale_s"], dvl_w=False)
        #: R3: the newest sample's quality flag (text, code; "" = none) and
        #: the (t_valid, altitude) of the recent UNflagged good reports
        self._s_flag, self._s_flag_code = "", ""
        self._alt_ref: deque = deque(maxlen=64)
        self._refuted_coh = False           # the newest refutation's coherence
        self._esc_onsets: deque = deque()   # counted episodes in the window
        self.n_hold = Counter()             # stage-1 hold onsets by kind
        self.n_flag = Counter()             # flagged samples by flag
        self.n_esc = Counter()              # latches: persistent / repeated
        self.esc_max = 0.0                  # largest escalation bucket (s)
        self.r_slow_max = 0.0               # largest slow residual (m/s)
        self.ep_log: list = []              # onset times of counted episodes
        #: onset times of episodes that reached EPISODE_MIN_S on alignment
        #: evidence alone (not counted toward esc_repeat_n; EPISODE_KINDS)
        self.ep_log_align: list = []
        self.warn_log: list = []            # (t, text) "disagreed again"
        self._warn_new: list = []           # ...not yet taken (take_warnings)
        self.esc_log: list = []             # (t, latch text)
        #: the station bridge's DVL rung: (t_valid, v_body) of its last
        #: usable sample, and the lost ticks it carried on it
        self._rung_hold = None
        self.n_rung_held = 0
        #: per-pair evidence for offline replays (DTF design/ harness: a
        #: list = record (t, dt_bucket, vd, vf, r_slow, alignment bad, fit
        #: coherent, alignment off, slow gap, scale, bucket)); None = off
        self.ev_log = None
        self._monitor_reset()
        self.n_samples = 0
        self.n_valid = 0
        self.n_out_of_order = 0
        self.n_innovation = 0
        self.t_first: float | None = None

    # ------------------------------------------------------------ the switch
    def refusal(self, on: bool) -> str:
        """Why the operator may NOT set the switch this way ("" = may)."""
        if not on:
            return ""
        if not self.mount_measured:
            return ("DVL mount not measured — set dvl.mount_rpy_deg and "
                    "dvl.mount_t_flu in hw_mpc.yaml")
        if self._s is None:
            return "no DVL data yet"
        f = self.agree.fit()
        if f["verdict"] == "refuted":
            # Whatever require_proven says: the opt-out waives the PROOF, it
            # does not overrule evidence that the mount is wrong.
            return f"the tag velocity REFUTES this DVL mount — {f['why']}"
        if not self.proven and self.against:
            # ...and that evidence does not expire with the fit's window or
            # with the trip that cleared it (review 2026-10-02: under the
            # opt-out a tripped DVL could be re-armed one second later, on
            # nothing).
            return (f"this session's data has shown the DVL wrong "
                    f"({self.against}) — it has to be PROVEN against the "
                    f"tag velocity before it goes IN again"
                    + (f": {f['why']}" if f["why"] else ""))
        if self.cfg["require_proven"] and not self.proven:
            return ("DVL not proven against the tag velocity yet — "
                    + (f["why"] or "fly a short pass with tags in view"))
        return ""

    def set_use(self, on: bool) -> str:
        """The operator's switch. Setting it IN is also the one thing that
        clears a trip — and only if ``refusal`` lets it through. (primary:
        an IN that clears a latch also starts a new escalation bucket.)"""
        was = bool(self.tripped)
        why = self.refusal(bool(on))
        if not why:
            self.use = bool(on)
            if on:
                self.tripped = ""
                if was and self.selection == "primary":
                    self._esc_reset()
        return why

    def arm_refusal(self) -> str:
        """(primary) May THIS engagement fly on the DVL? ``refusal(True)``
        without its refuted-fit branch: a refutation current at engage is
        the per-tick ``dis_fit`` hold's business, which releases by itself —
        it no longer unarms the whole run (the shipped verdicts lasted up to
        ~52 s after an episode; REFUTE_HOLD_S). The switch IN, no code fault
        and not latched are the worker's questions, as before."""
        if not self.mount_measured:
            return ("DVL mount not measured — set dvl.mount_rpy_deg and "
                    "dvl.mount_t_flu in hw_mpc.yaml")
        if self._s is None:
            return "no DVL data yet"
        f = self.agree.fit()
        if not self.proven and self.against:
            return (f"this session's data has shown the DVL wrong "
                    f"({self.against}) — it has to be PROVEN against the "
                    f"tag velocity before it goes IN again"
                    + (f": {f['why']}" if f["why"] else ""))
        if self.cfg["require_proven"] and not self.proven:
            return ("DVL not proven against the tag velocity yet — "
                    + (f["why"] or "fly a short pass with tags in view"))
        return ""

    # --------------------------------------------------------------- inputs
    def note_sample(self, s, omega=None) -> None:
        """One ``state.DvlSample``. ``omega`` = the body rates (p, q, r) at
        the sample's ``t_valid``, for the lever arm. (primary: then the R3
        quality flags, :meth:`_flag_sample`.)"""
        lg_before = self._last_good
        self._note_sample(s, omega)
        if self.selection == "primary" and self._s is s:
            self._flag_sample(s, lg_before)

    def _flag_sample(self, s, lg_before) -> None:
        """R3 (primary): the report's OWN word that it may be a false lock,
        on every sample that became the newest and passed the data gates. A
        flag is usage only: the sample stays data-good in ``_hist`` (the
        pairs, the monitors, the fit) but is not flown, and it does not
        become the jump gate's reference (``_last_good``) — the twin's
        detected report never moves its own [스펙 RL_controller
        rov_rl/velocity_source.py]. An out-of-order sample never gets here,
        so the newest sample's flag stands."""
        if not self._s_ok:
            self._s_flag, self._s_flag_code = "", ""
            return
        c = self.cfg
        t = float(s.t_valid)
        hits = []
        fom = getattr(s, "fom", None)
        if c["flag_fom_m_s"] > 0.0 and fom is not None \
                and float(fom) > c["flag_fom_m_s"]:
            hits.append(("flag_fom", f"DVL quality flag: fom "
                                     f"({float(fom) * 1e3:.1f} > "
                                     f"{c['flag_fom_m_s'] * 1e3:.1f} mm/s)"))
        alt = getattr(s, "altitude", None)
        if c["flag_alt_jump_m"] > 0.0 and alt is not None:
            ref = [a for (tt, a) in self._alt_ref
                   if t - c["flag_alt_window_s"] <= tt < t]
            if len(ref) >= 3:
                med = float(np.median(ref))
                if abs(float(alt) - med) > c["flag_alt_jump_m"]:
                    hits.append(("flag_alt", (
                        f"DVL quality flag: altitude jump ({float(alt):.2f} m "
                        f"against {med:.2f} m over the last "
                        f"{c['flag_alt_window_s']:g} s)")))
        nb = getattr(s, "n_beams", None)
        if c["flag_min_beams"] > 0 and nb is not None \
                and int(nb) < c["flag_min_beams"]:
            hits.append(("flag_beams", f"DVL quality flag: beams ({int(nb)} "
                                       f"< {c['flag_min_beams']})"))
        if c["flag_reacquired"] and len(self._hist) >= 2:
            h = self._hist[-2]                  # the entry before its own
            dtg = t - float(h[0])
            if not h[2] and 0.0 <= dtg <= c["stale_s"]:
                hits.append(("flag_reacq", (
                    f"DVL quality flag: lock re-acquired (the report "
                    f"{dtg:.2f} s before it failed a data gate)")))
        for code, _w in hits:
            self.n_flag[code] += 1
        if hits:
            self.n_flag["flagged"] += 1
            self._s_flag_code, self._s_flag = hits[0]
            self._last_good = lg_before
        else:
            self._s_flag, self._s_flag_code = "", ""
            self._alt_ref.append((t, float(alt)))

    def _note_sample(self, s, omega=None) -> None:
        """The shipped note_sample body (both modes)."""
        self.n_samples += 1
        self.source = str(getattr(s, "source", "") or self.source)
        if self.t_first is None:
            self.t_first = float(s.t_rx)
        if self._s is not None and float(s.t_valid) < float(self._s.t_valid):
            # Older than what we already hold (a re-ordered burst). It cannot
            # become "the newest sample", and the history is kept in time
            # order for the pairing below.
            self.n_out_of_order += 1
            return
        v_body = None
        if s.v_dvl is not None:
            v_body = body_velocity(s.v_dvl, self.R, self.r, omega)
        ok, why = self._sample_ok(s, v_body)
        if ok and omega is None and bool(np.any(np.abs(self.r) > 1e-9)):
            # With a lever arm, a reading that could not be corrected for
            # the body rate is NOT the body-origin velocity (0.3 rad/s on a
            # 0.2 m arm is 6 cm/s of it). Fail closed: this tick falls back
            # and the sample is a gap for the pairing and the integral.
            ok, why = False, "no body rate for the DVL lever arm"
        self._s, self._s_ok, self._s_why, self._v_body = s, ok, why, v_body
        t = float(s.t_valid)
        if ok:
            self.n_valid += 1
            if self._good_since is None or (
                    self._last_good is not None
                    and t - self._last_good[0] > self.cfg["stale_s"]):
                # ...or a SILENCE longer than stale_s with no bad sample in
                # it (a link stall): the hysteresis starts over there too.
                self._good_since = t
            self._last_good = (t, v_body.copy())
            self._hist.append((t, v_body.copy(), True))
        else:
            self._good_since = None
            # A gap marker: nothing may interpolate across a dropout.
            self._hist.append((t, None, False))
        self._pair()

    def note_fix(self, fix, imu) -> None:
        """One NEW tag fix: form the raw two-point finite difference in the
        body frame, valid at the midpoint of the two capture stamps, and
        queue it for pairing. Raw rather than the assembler's low-passed
        velocity: the low-pass lag would bias the fit, noise only widens it.
        """
        if fix is None or not getattr(fix, "ok", False) or imu is None \
                or imu.roll is None:
            return
        tc = float(fix.t_capture)
        p = np.asarray(fix.p_ned, float).reshape(3)
        prev = self._prev_fix
        if prev is not None and tc <= prev[0] + 1e-6:
            return                           # the same camera frame again
        self._prev_fix = (tc, p)
        if prev is None:
            return
        dt = tc - prev[0]
        if not 0.02 <= dt <= 0.5:
            return                           # a gap is not a derivative
        R_tag = np.asarray(fix.R_ned_body, float).reshape(3, 3)
        psi = math.atan2(R_tag[1, 0], R_tag[0, 0])
        R = rot_zyx(float(imu.roll), float(imu.pitch), psi)
        v_fd_b = R.T @ ((p - prev[1]) / dt)
        if float(np.linalg.norm(v_fd_b)) > self.cfg["v_max_m_s"]:
            return                           # a tag JUMP, not a velocity
        self._fd_pending.append((0.5 * (tc + prev[0]), v_fd_b))
        self._pair()

    def _pair(self) -> None:
        """Match queued tag differences with the DVL velocity interpolated
        at their midpoint. A pair needs two GOOD samples bracketing it no
        more than 0.4 s apart — otherwise it is dropped, not extrapolated."""
        if not self._fd_pending or not self._hist:
            return
        newest = self._hist[-1][0]
        keep = deque(maxlen=self._fd_pending.maxlen)
        added = False
        for t_mid, v_fd in self._fd_pending:
            if t_mid > newest:
                if t_mid - newest < 2.0:
                    keep.append((t_mid, v_fd))   # the DVL has not got there
                continue
            a = b = None
            for h in self._hist:
                if h[0] <= t_mid:
                    a = h
                elif b is None:
                    b = h
                    break
            if a is None or b is None or not (a[2] and b[2]) \
                    or b[0] - a[0] > 0.4 or b[0] - a[0] <= 0.0:
                continue
            w = (t_mid - a[0]) / (b[0] - a[0])
            v_d = (1.0 - w) * a[1] + w * b[1]
            acc = (b[1] - a[1])[:2] / (b[0] - a[0])
            if self.agree.add(t_mid, v_d[:2], v_fd[:2], acc):
                added = float(t_mid)
            self._monitor(t_mid, v_d[:2], v_fd[:2])
        self._fd_pending = keep
        if added is not False:
            self._judge(added)

    def _armed(self) -> bool:
        """Is there something for a trip to take out? A proof — also one
        re-made while a trip is still latched, or it could never be undone
        again — or the switch IN with nothing latched yet."""
        return bool(self.proven or (self.use and not self.tripped))

    def _judge(self, t: float | None = None) -> None:
        """One evaluation of the fit after new evidence (the newest pair's
        time ``t``). Proving is immediate (the verdict already carries its
        3-sigma margin); UN-proving takes ``agree_refute_n`` refuting
        evaluations in a row."""
        if self.selection == "primary":
            self._judge_primary(t)
            return
        f = self.agree.fit()
        if f["verdict"] == "proven":
            self._refuted_run = 0
            if not self.proven:
                self.proven, self.proof = True, dict(f)
                self.against = ""        # positive evidence, made afterwards
        elif f["verdict"] == "refuted":
            self._refuted_run += 1
            if t is not None:
                self._refuted_t = float(t)
            if self._refuted_run < self.cfg["agree_refute_n"]:
                return
            if self._armed():
                # Only a COHERENT refutation is about the mount (see
                # COHERENT_MIN); a noisy DVL at rest is not a trip.
                if f["coherent"]:
                    self._trip(f["why"])
            elif f["coherent"] and not self.against:
                # In shadow, with nothing to take out: remember it. The
                # window forgets in agree_window_s; this does not. (Only
                # what would have tripped — a noisy DVL at rest is not
                # evidence about the mount in shadow either.)
                self.against = f["why"]
        else:
            self._refuted_run = 0

    def _judge_primary(self, t: float | None = None) -> None:
        """(primary) The same counting, with NO trip: ``agree_refute_n``
        refutations in a row make the refutation current — a ``dis_fit``
        hold through the blend while it lasts, any coherence; a coherent one
        is also escalation evidence (_monitor), and only an escalation latch
        un-proves. A proof starts a new escalation bucket. In shadow with
        nothing armed, a coherent refutation is still remembered (shipped)."""
        f = self.agree.fit()
        if f["verdict"] == "proven":
            self._refuted_run = 0
            if not self.proven:
                self.proven, self.proof = True, dict(f)
                self.against = ""
                self._esc_reset()
        elif f["verdict"] == "refuted":
            self._refuted_run += 1
            if t is not None:
                self._refuted_t = float(t)
            self._refuted_coh = bool(f["coherent"])
            if self._refuted_run < self.cfg["agree_refute_n"]:
                return
            if not self._armed() and f["coherent"] and not self.against:
                self.against = f["why"]
        else:
            self._refuted_run = 0

    def _fit_current(self, t: float) -> bool:
        """The shipped refutation-hold condition (any coherence): enough
        refutations in a row, the newest within REFUTE_HOLD_S, and the
        verdict still refuted."""
        return bool(self._refuted_run >= self.cfg["agree_refute_n"]
                    and self._refuted_t is not None
                    and float(t) - self._refuted_t <= REFUTE_HOLD_S
                    and self.agree.fit()["verdict"] == "refuted")

    def _trip(self, why: str) -> None:
        """Evidence that the DVL is wrong. Latched (see the module doc)."""
        self.tripped = str(why)
        self.against = str(why)
        self.n_trips += 1
        self.last_trip = str(why)
        self._invalidate()

    def _invalidate(self, stream_changed: bool = False) -> None:
        """Throw away what has been learnt about the stream: the proof, the
        pairs it was made of, the monitors' memory. ``stream_changed`` = the
        device now rotates or scales its output differently, so nothing may
        pair or integrate ACROSS this moment either."""
        self.proven = False
        self.proof = None
        self.agree.clear()
        self._refuted_run = 0
        self._refuted_t = None
        self._innov_hold = ""
        self._innov_hold_fix = None
        self._monitor_reset()
        if stream_changed:
            self._fd_pending.clear()
            if self._hist:
                self._hist.append((self._hist[-1][0], None, False))
            self._good_since = None
            self._last_good = None

    def _monitor_reset(self) -> None:
        self._res_vec = None                # low-passed (v_dvl - v_tag)_xy
        self._res_lp: float | None = None   # ...and its magnitude
        # Exponentially weighted second moments of the aligned pairs, for
        # the alignment test (un-normalised: divide by _al_w1).
        self._al_S = np.zeros((2, 2))       # v_dvl v_dvl^T
        self._al_M = np.zeros((2, 2))       # v_tag v_dvl^T
        self._al_F = np.zeros(2)            # v_tag^2, per component
        self._al_w1 = 0.0                   # sum of weights
        self._al_w2 = 0.0                   # sum of squared weights
        self._al_bad_t0: float | None = None    # alignment verdict "bad" since
        #: the newest alignment verdict that reads bad, coherent or not —
        #: (the direction's dict) or None — and the time of that pair
        self._al_hold: dict | None = None
        self._al_hold_t: float | None = None
        self._mon_t: float | None = None    # the newest pair's time
        self._pw_d = 0.0                    # low-passed |v_dvl|^2, every pair
        self._pw_f = 0.0                    # low-passed |v_tag|^2, every pair
        # 2026-10-16: the slow monitors (both modes, recorded; primary acts)
        self._slow_vec = None               # slow LP of (v_dvl - v_tag)_xy
        self._r_slow: float | None = None   # ...its magnitude
        self._pd_s = 0.0                    # slow LP of |v_dvl|^2
        self._pf_s = 0.0                    # slow LP of |v_tag|^2
        self._slow_t: float | None = None   # the newest pair's time (slow)
        self._al_min: float | None = None   # lowest align judged, newest pair
        self._hold_diff = False             # the difference hold (hysteresis)
        self._fit_was = False               # rising edges, for the counters
        self._fit_inc_was = False
        self._dis_code = ""                 # the sticky disagreement code
        self._esc_reset()

    def _esc_reset(self) -> None:
        """(primary) The escalation bucket and its episode history start
        again: at a latch or a config change (through _monitor_reset), at a
        proof (_judge_primary) and at the IN that clears a latch (set_use).
        A latch is therefore never re-fired on evidence from before it."""
        self._esc_s = 0.0
        self._esc_ev = ""
        self._ep_active = False
        self._ep_t0: float | None = None
        self._ep_peak = 0.0
        self._ep_counted = False
        self._ep_kinds: set = set()         # evidence kinds of this episode
        self._esc_onsets.clear()

    def note_device_config(self, info: dict) -> str:
        """The device's own ``get_config`` answer. A change in what rotates
        or scales its OUTPUT — the mounting rotation offset, the speed of
        sound — after a first answer means everything learnt so far was
        learnt on a different stream: it is thrown away, and with a proof or
        the switch IN that is a trip. Returns what changed ("" = nothing)."""
        prev = self.device_config
        self.device_config = dict(info)
        if prev is None:
            return ""
        diff = []
        for k in ("mounting_rotation_offset", "speed_of_sound"):
            a, b = _finite(prev.get(k)), _finite(info.get(k))
            if a != b:
                diff.append(f"{k} {prev.get(k)!r} -> {info.get(k)!r}")
        if not diff:
            return ""
        what = "; ".join(diff)
        self.device_config_changes.append(what)
        # The evidence so far describes a stream that no longer exists —
        # in EVERY state: in shadow and not yet proven, 59 old pairs plus
        # one new one would otherwise complete a proof of the old stream;
        # under a latched trip, a proof re-made since would survive the
        # change (review 2026-10-02, both reproduced).
        if self.proven or self.use:
            self._trip(f"the device's configuration changed ({what})")
        self._invalidate(stream_changed=True)
        return what

    # ---------------------------------------------------------------- gates
    def _sample_ok(self, s, v_body) -> tuple:
        """The per-sample DATA gates (no clock, no operator, no proof).
        Reasons are a stable phrase, then the numbers in parentheses — the
        run meta counts fallback rows by the phrase."""
        c = self.cfg
        if s is None:
            return False, "no DVL data"
        if s.v_dvl is None or not s.valid or v_body is None:
            return False, "DVL has no bottom lock"
        if getattr(s, "kind", "velocity") != "velocity":
            return False, "DVL is in water-tracking mode"
        # FAIL-CLOSED: a report that claims a lock but carries no fom or no
        # altitude has not told us what these gates need to know.
        if s.fom is None or s.altitude is None:
            return False, "DVL report lacks fom or altitude"
        if s.fom > c["fom_max"]:
            return False, f"DVL fom too high ({s.fom:.3f} > {c['fom_max']:.3f} m/s)"
        if s.altitude < c["min_altitude_m"]:
            return False, (f"DVL altitude too low ({s.altitude:.2f} < "
                           f"{c['min_altitude_m']:.2f} m)")
        if not np.all(np.isfinite(v_body)) or max(
                float(np.linalg.norm(s.v_dvl)),
                float(np.linalg.norm(v_body))) > c["v_max_m_s"]:
            return False, "DVL velocity implausible"
        lg = self._last_good
        if lg is not None:
            dtv = float(s.t_valid) - lg[0]
            if 0.0 < dtv <= c["stale_s"]:
                lim = c["a_max_m_s2"] * dtv + JUMP_MARGIN_M_S
                jump = float(np.linalg.norm(v_body - lg[1]))
                if jump > lim:
                    return False, (f"DVL velocity jump ({jump:.2f} > "
                                   f"{lim:.2f} m/s)")
        return True, ""

    def data_ok(self, t_now: float) -> tuple:
        """Is the newest sample good data RIGHT NOW (fresh + the gates)?"""
        s = self._s
        if s is None:
            return False, "no DVL data"
        age = float(t_now) - float(s.t_valid)
        if age > self.cfg["stale_s"]:
            return False, f"DVL stale ({age:.2f}s)"
        return self._s_ok, self._s_why

    def _fd_sigma_xy(self) -> float:
        """Noise (2-D rms, m/s) of the assembler's low-passed tag velocity,
        as the proof measured it; 0 before a proof exists."""
        if not self.proof or self.proof.get("rms_m_s") is None:
            return 0.0
        return float(self.proof["rms_m_s"]) * self.fd_lp_gain

    def velocity(self, t_now: float, need_ref: bool = True,
                 count: bool = True) -> tuple:
        """-> (v_body (3,) | None, why). The ONE question the tick asks: may
        this tick's nu[0:2] be the DVL's? None = use the tag finite
        difference, and ``why`` says what stood in the way. (The caller
        consumes the horizontal pair only — see the module doc.)

        ``need_ref``: a tick that has no CURRENT tag velocity to check the
        sample against (none within FD_REF_MAX_S) does not use it — the
        first tick after a tag gap is exactly when a false lock picked up
        during the gap would otherwise go in unchecked. False only for the
        station bridge's DVL rung, which has no tag by definition and is
        bounded by its own caps instead.
        ``count``: a display-only question (``state``) must not be booked as
        another rejected tick."""
        if not self.use:
            return None, "DVL is OUT (shadow)"
        if self.tripped:
            return None, f"DVL tripped ({self.tripped})"
        if not self.mount_measured:
            return None, "DVL mount not measured"
        ok, why = self.data_ok(t_now)
        if not ok:
            return None, why
        if (self.cfg["require_proven"] or self.against) and not self.proven:
            return None, "DVL not proven against the tag velocity"
        t_now = float(t_now)
        if self._refuted_run >= self.cfg["agree_refute_n"] \
                and self._refuted_t is not None \
                and t_now - self._refuted_t <= REFUTE_HOLD_S \
                and self.agree.fit()["verdict"] == "refuted":
            # This session's RECENT evidence refutes the DVL. A coherent
            # refutation has tripped by now (_judge); an incoherent one —
            # DVL noise, false locks — does not trip, but it is not flown on
            # either (review 2026-10-02: it was, for the rest of the
            # engagement, while refusal() would have refused IN on the same
            # data). No latch: it ends when the evidence stops.
            return None, "DVL held back (the fit refutes it)"
        if self._al_hold is not None and self._al_hold_t is not None \
                and t_now - self._al_hold_t <= 2.0 * self.cfg["agree_trip_s"]:
            # The alignment monitor's newest verdict: the DVL's motion is
            # going somewhere the tags do not show. Not necessarily the
            # mount (that is the trip's question) — but not a velocity to
            # fly on while it lasts.
            al = self._al_hold
            return None, (f"DVL alignment hold (bearing "
                          f"{al['along_deg']:+.0f} deg: alignment "
                          f"{al['align']:+.2f}, the tags see "
                          f"{al['gain']:.2f} of it)")
        held = float(self._s.t_valid) - float(self._good_since
                                              if self._good_since is not None
                                              else self._s.t_valid)
        if held < self.cfg["reentry_s"]:
            return None, (f"DVL re-entry "
                          f"({held:.2f}/{self.cfg['reentry_s']:.2f}s)")
        ref = self._fd_ref
        ref_age = None if ref is None else t_now - ref[0]
        if ref_age is not None and 0.0 <= ref_age <= FD_REF_MAX_S:
            # THE INNOVATION GATE: a current tag velocity exists, so a DVL
            # sample far from it is not used this tick. Not a trip — one
            # bad sample is an outlier, a second of them is the monitor's.
            # The reference is last tick's (one tick old in the normal
            # case); anything older than that widens the limit like a
            # stale fix does.
            tag_age = float(ref[2] or 0.0)
            fix_t = float(ref[3])
            g = self._innov_limit(tag_age + max(0.0, ref_age - 0.075))
            d = float(np.hypot(*(self._v_body[:2] - ref[1])))
            if d > g:
                why = f"DVL innovation ({d:.2f} > {g:.2f} m/s)"
                if count:
                    self.n_innovation += 1
                    # Evidence: the DVL stays out until a tag velocity from
                    # a NEWER camera frame agrees. The same frame growing
                    # older — its limit widening — must not let it back in
                    # (review 2026-10-02, reproduced: as the fix aged the
                    # limit widened past a false 0.25 m/s the gate had been
                    # rejecting, and the station bridge then carried it
                    # 0.6 m). "Newer frame", not "fresh frame": with the
                    # tags 0.25 s or more late no frame is ever fresh, and a
                    # hold keyed on freshness was then either never set or
                    # never released (round 5).
                    self._innov_hold = why
                    self._innov_hold_fix = (
                        fix_t if self._innov_hold_fix is None
                        else max(self._innov_hold_fix, fix_t))
                    # ...and the rejected sample stays out of the fix-age
                    # integral too: the gate judges the NEWEST sample only,
                    # and once it releases, displacement() over the fix's
                    # age still held the samples it had rejected (round 6,
                    # reproduced: 9-11 cm into eta with the tags in view,
                    # then carried by the station bridge's DVL rung).
                    tv = float(self._s.t_valid)
                    self._innov_reject_t = (
                        tv if self._innov_reject_t is None
                        else max(self._innov_reject_t, tv))
                return None, why
            if count and self._innov_hold and (
                    self._innov_hold_fix is None
                    or fix_t > self._innov_hold_fix + 1e-6):
                self._innov_hold = ""
                self._innov_hold_fix = None
        elif need_ref:
            return None, "DVL unchecked (no current tag velocity)"
        if self._innov_hold:
            return None, (f"DVL innovation hold (until a fresh tag velocity "
                          f"agrees; it was {self._innov_hold[len('DVL '):]})")
        return self._v_body.copy(), ""

    def displacement(self, t0: float, t1: float):
        """Integral of the body velocity over [t0, t1] (body frame, (3,)),
        from the good samples in the history — or None when the interval is
        not covered by an unbroken run of them.

        The velocity is taken as piecewise linear between samples and held
        flat before the first and after the last. The assembler rotates the
        result by the CURRENT attitude; over the fraction of a second this
        is used for, the heading moves by hundredths of a radian.

        A sample the innovation gate rejected — and every one before it —
        counts as a gap here: the integral never contains what the gate
        kept out of the tick (None then, and the caller falls back to
        velocity x age with a sample the gate accepted)."""
        t0, t1 = float(t0), float(t1)
        if not t1 > t0:
            return np.zeros(3)
        stale = self.cfg["stale_s"]
        rej = self._innov_reject_t
        knots = []
        gap_before = False
        for t, v, ok in self._hist:
            if t < t0 - stale or t > t1:
                continue
            if not ok or (rej is not None and t <= rej):
                if t >= t0:
                    return None             # a dropout inside the interval
                knots = []                  # ...or before it: start after it
                gap_before = True
                continue
            knots.append((t, v))
        if not knots or t1 - knots[-1][0] > stale:
            return None
        if knots[0][0] > t0 and (gap_before or knots[0][0] - t0 > stale):
            return None                     # the start is not covered
        for a, b in zip(knots, knots[1:]):
            if b[0] - a[0] > stale:
                return None
        total = np.zeros(3)

        def v_at(t):
            if t <= knots[0][0]:
                return knots[0][1]
            if t >= knots[-1][0]:
                return knots[-1][1]
            for a, b in zip(knots, knots[1:]):
                if a[0] <= t <= b[0]:
                    w = (t - a[0]) / max(b[0] - a[0], 1e-9)
                    return (1.0 - w) * a[1] + w * b[1]
            return knots[-1][1]

        cuts = [t0] + [k[0] for k in knots if t0 < k[0] < t1] + [t1]
        for a, b in zip(cuts, cuts[1:]):
            total = total + 0.5 * (v_at(a) + v_at(b)) * (b - a)
        return total

    def _innov_limit(self, tag_age=None) -> float:
        """How far a DVL sample may sit from a CURRENT tag velocity
        (horizontal, m/s): the configured gate, widened to 4 sigma of the
        tag-velocity noise the proof measured and — when given — by what the
        vehicle could have done since the tag velocity was current."""
        g = max(self.cfg["innov_max_m_s"], 4.0 * self._fd_sigma_xy())
        if tag_age is not None:
            g += self.cfg["a_max_m_s2"] * max(0.0, float(tag_age) - FD_FRESH_S)
        return g

    def note_tag_velocity(self, v_fd_body, t_now: float, tag_age=None) -> None:
        """Once per tick: this tick's tag velocity (the assembler's
        low-passed one), or None when there is no tag state. It is the
        innovation gate's reference for the NEXT call of :meth:`velocity` and
        nothing else — the monitors that can TRIP are fed by time-aligned
        pairs (:meth:`_monitor`), not by this. A tick without a tag state
        leaves the last reference to age out by itself (FD_REF_MAX_S)."""
        if v_fd_body is None:
            return
        v_fd = np.asarray(v_fd_body, float).reshape(3)[:2].copy()
        t_now = float(t_now)
        # The frame it was made from: what tells a new camera frame from the
        # same one growing older.
        fix_t = t_now - float(tag_age) if tag_age is not None else t_now
        self._fd_ref = ((t_now, v_fd, tag_age, fix_t)
                        if np.all(np.isfinite(v_fd)) else None)

    # ------------------------------------------------- usage (primary only)
    def usable(self, t_now: float) -> tuple:
        """-> (ok, why, code): may the blend use the NEWEST sample this tick
        (its ``dvl_ok``)? The first hit answers: the switch, a latch, the
        mount, the data gates (fresh within stale_s), the proof, an R3 flag,
        a disagreement hold. ``why`` keeps the shipped style (a digit-free
        phrase, then the numbers in parentheses); ``code`` is the CSV's
        dvl_code. No side effect but disagree_hold's sticky code, which is a
        function of the aid's state: asking twice in one tick answers the
        same."""
        if not self.use:
            return False, "DVL is OUT (shadow)", "out"
        if self.tripped:
            return False, f"DVL latched OUT ({self.tripped})", "latched"
        if not self.mount_measured:
            return False, "DVL mount not measured", "no_mount"
        ok, why = self.data_ok(t_now)
        if not ok:
            return False, why, reason_code(why)
        if (self.cfg["require_proven"] or self.against) and not self.proven:
            return False, "DVL not proven against the tag velocity", "unproven"
        if self._s_flag:
            return False, self._s_flag, self._s_flag_code
        why, code = self.disagree_hold(t_now)
        if why:
            return False, why, code
        return True, "", "ok"

    def disagree_hold(self, t_now: float) -> tuple:
        """-> (why, code) of the stage-1 hold, or ("", ""): the first of its
        own conditions that holds — the slow difference (with hysteresis),
        the alignment verdict (the shipped alignment hold, 2 x agree_trip_s
        of life), a CURRENT fit refutation of any coherence. Once set, a hold
        releases only when its own condition has cleared AND the escalation
        bucket is below HOLD_DRAIN_S: a reversed axis or a scale error stops
        showing whenever the vehicle slows, and at rest nothing can be judged
        at all — so not "until it agrees again" (that would keep the DVL out
        through a whole station hold), but at most 2 x (bucket - 0.5) s
        after its evidence ends at esc_leak 0.5 [유도]."""
        c = self.cfg
        t_now = float(t_now)
        own = None
        if self._hold_diff:
            r = float(self._r_slow or 0.0)
            if r > c["agree_hold_m_s"]:
                own = (f"DVL disagrees with the tag velocity (slow residual "
                       f"{r * 100:.1f} > {c['agree_hold_m_s'] * 100:.1f} "
                       f"cm/s)", "dis_diff")
            else:
                own = (f"DVL disagreement releasing (slow residual "
                       f"{r * 100:.1f} cm/s; it returns below "
                       f"{c['agree_release_m_s'] * 100:.1f})", "dis_diff")
        elif self._al_hold is not None and self._al_hold_t is not None \
                and t_now - self._al_hold_t <= 2.0 * c["agree_trip_s"]:
            al = self._al_hold
            own = (f"DVL disagrees with the tag velocity: alignment (bearing "
                   f"{al['along_deg']:+.0f} deg: alignment "
                   f"{al['align']:+.2f}, the tags see {al['gain']:.2f} of "
                   f"it)", "dis_align")
        elif self._fit_current(t_now):
            if self._refuted_coh:
                own = ("DVL disagrees with the tag velocity: the fit refutes "
                       "the mount (the tags see the motion elsewhere)",
                       "dis_fit")
            else:
                g = self.agree.fit().get("gain_min") or 0.0
                own = (f"DVL disagrees with the tag velocity: the fit refutes "
                       f"it (the tags see only {g:.2f} of the motion it "
                       f"reports)", "dis_fit")
        if own:
            self._dis_code = own[1]
            return own
        if self._dis_code and self._esc_s >= HOLD_DRAIN_S:
            return (f"DVL disagreement releasing (waiting for "
                    f"{self._esc_s:.1f} s of disagreement evidence to "
                    f"drain)", self._dis_code)
        self._dis_code = ""
        return "", ""

    def blend_reset(self) -> None:
        """At engage, with the assembler's reset: the weight snaps to its
        first target on the next step."""
        self.blend.reset()

    def blend_step(self, t_now: float, fd_b, ok: bool, code: str,
                   n_skip: int = 0) -> tuple:
        """One tick of the blend on the NEWEST sample -> (v_b (3,), w, src,
        code, why). ``ok`` / ``code`` = this tick's :meth:`usable` (or the
        worker's not-armed / fault). A disagreement or a latch expires the
        held sample first, so the fade starts on this tick instead of
        re-flying the sample the disagreement is about; a data gate or a flag
        keeps it (its last good sample IS good). When the ladder said ok but
        the gross check refused, code / why become "gross"."""
        code = str(code or "")
        if code.startswith("dis_") or code == "latched":
            self.blend.expire_hold()
        s = self._s
        tv = None if s is None else float(s.t_valid)
        v_b, w = self.blend.step(float(t_now), fd_b, self.v_body, tv,
                                 bool(ok), n_skip=n_skip)
        why = ""
        if ok and self.blend.gross_rejected:
            dd = float(np.hypot(*(np.asarray(self._v_body, float)[:2]
                                  - self.blend.tag_lp)))
            code = "gross"
            why = (f"DVL gross (|v_dvl - LP(v_tag)| {dd:.2f} > "
                   f"{self.blend.gross:.2f} m/s)")
        return v_b, w, self.blend.src, code, why

    def rung_admit(self, t_now: float, fd_b, tag_age=None) -> tuple:
        """-> (ok, why): may a tagged tick that FLEW the DVL (vel_src "dvl")
        open the station bridge's DVL rung? Its sample within the shipped
        innovation limit of THIS tick's tag velocity, and no disagreement
        evidence pending. The shipped station never let the rung take a DVL
        the tags were contradicting on the last tagged ticks (review
        2026-10-02: a false 0.25 m/s walked the estimate 0.5 m); the blend
        flies such a sample for a moment, the rung must not carry it. On
        admission the sample becomes the rung's hold reference."""
        if self._v_body is None or fd_b is None or self._s is None:
            return False, "no sample / no tag velocity"
        d = float(np.hypot(*(np.asarray(self._v_body, float)[:2]
                             - np.asarray(fd_b, float).reshape(-1)[:2])))
        g = self._innov_limit(tag_age)
        if d > g:
            return False, f"DVL innovation ({d:.2f} > {g:.2f} m/s)"
        if self._r_slow is not None \
                and self._r_slow >= self.cfg["agree_release_m_s"]:
            return False, (f"DVL slow residual pending "
                           f"({self._r_slow * 100:.1f} cm/s)")
        if self._esc_s > 0.0:
            return False, (f"DVL escalation evidence pending "
                           f"({self._esc_s:.1f} s)")
        self._rung_hold = (float(self._s.t_valid),
                           np.asarray(self._v_body, float).copy())
        return True, ""

    def rung_velocity(self, t_now: float) -> tuple:
        """-> (v_body (3,) | None, why) for a LOST tick (no tag state) on the
        station bridge's DVL rung: the newest sample if usable and within the
        gross limit of the last tag low-pass; across a data gate or an R3
        flag, the last rung-usable sample while it was captured <=
        blend_hold_s ago (the blend's own hold, on the rung); else None at
        once (a latch, a disagreement, OUT). The blend is NOT stepped."""
        ok, why, code = self.usable(t_now)
        if ok:
            v = np.asarray(self._v_body, float)
            lp = self.blend.tag_lp
            if lp is not None and self.blend.gross > 0.0:
                dd = float(np.hypot(*(v[:2] - lp)))
                if dd > self.blend.gross:
                    return None, (f"DVL gross (|v_dvl - LP(v_tag)| {dd:.2f} "
                                  f"> {self.blend.gross:.2f} m/s, the last "
                                  f"tag low-pass)")
            self._rung_hold = (float(self._s.t_valid), v.copy())
            return v.copy(), ""
        hold = self._rung_hold
        if code in RUNG_BRIDGE_CODES and hold is not None \
                and float(t_now) - hold[0] <= self.cfg["blend_hold_s"]:
            self.n_rung_held += 1
            return hold[1].copy(), "held: " + why
        return None, why

    def take_warnings(self) -> list:
        """The "disagreed again" warnings since the last call [(t, text)],
        for the worker's log."""
        out, self._warn_new = list(self._warn_new), []
        return out

    @property
    def residual_slow_m_s(self) -> float | None:
        return self._r_slow

    @property
    def escalation_s(self) -> float:
        return float(self._esc_s)

    @property
    def scale_ratio(self) -> float | None:
        """sqrt(slow |v_dvl|^2 / slow |v_tag|^2) while the tags see at least
        esc_scale_v_min_m_s of rms motion, else None (the DVL's side from
        the clipped pair, _slow_monitors)."""
        pf = self._pf_s
        if pf <= 0.0 or math.sqrt(pf) < self.cfg["esc_scale_v_min_m_s"]:
            return None
        return math.sqrt(max(self._pd_s, 0.0) / pf)

    @property
    def episodes_recent(self) -> int:
        """Counted episodes within esc_repeat_window_s of the newest pair.
        The bucket prunes its own list only while it is live; frozen (not
        armed, latched) the window still moves for the panel (fix round 1:
        it showed onsets older than the window)."""
        t = self._mon_t
        if t is None:
            return len(self._esc_onsets)
        win = float(self.cfg["esc_repeat_window_s"])
        return sum(1 for t0 in self._esc_onsets if t - t0 <= win)

    # ------------------------------------------------------------- monitors
    def _gross_limit(self) -> float:
        """How far apart one aligned pair may be before it is GROSS: the
        innovation gate's base, or 4 sigma of the raw pair scatter the proof
        measured if that is larger."""
        rms = float(self.proof["rms_m_s"]) if (
            self.proof and self.proof.get("rms_m_s") is not None) else 0.0
        return max(self.cfg["innov_max_m_s"], 4.0 * rms)

    def _monitor(self, t: float, vd, vf) -> None:
        """One time-aligned pair -> the runtime monitors (DEFAULTS describes
        them). ``vd`` is the DVL body velocity interpolated to the midpoint
        ``t`` of the raw tag difference ``vf`` (horizontal pairs).

        The monitors always run — the residual is shown and recorded in
        shadow too — but they only TRIP while there is something to take
        out: a proof, or the switch IN.

        2026-10-16: under ``selection: primary`` neither of the two trips
        below exists; the same verdicts feed the self-releasing holds and the
        escalation instead (:meth:`_monitor_primary`). The slow residual and
        the slow powers are computed in BOTH modes (CSV dvl_res_slow_m_s /
        dvl_scale); legacy decides nothing on them."""
        cfg = self.cfg
        t = float(t)
        last, self._mon_t = self._mon_t, t
        vd = np.asarray(vd, float).reshape(2)
        vf = np.asarray(vf, float).reshape(2)
        if not (np.all(np.isfinite(vd)) and np.all(np.isfinite(vf))):
            return
        self.n_monitor += 1
        # After a gap the first pair only restarts the clock: letting it in
        # at full weight would hand the filters to one noisy pair.
        gap = last is None or not 0.0 < t - last <= MONITOR_GAP_S
        a = 0.0 if gap else min(1.0, (t - last) / cfg["agree_trip_s"])
        d = vd - vf
        dn = float(np.hypot(d[0], d[1]))
        gross = dn > self._gross_limit()
        if gross:
            self.n_gross += 1
        d_raw = d                       # (the slow residual clips it itself)
        clip = float(cfg["innov_max_m_s"])
        if dn > clip:
            d = d * (clip / dn)
        if self._res_vec is None:
            self._res_vec = np.zeros(2)
        self._res_vec = self._res_vec + a * (d - self._res_vec)
        self._res_lp = float(np.hypot(self._res_vec[0], self._res_vec[1]))
        self._pw_d += a * (float(vd @ vd) - self._pw_d)
        self._pw_f += a * (float(vf @ vf) - self._pw_f)
        dtb, gap_s = self._slow_monitors(t, d_raw, dn, vd, vf, gross and dn > (
            (SCALE_PAIR_MAX - 1.0) * float(np.hypot(vf[0], vf[1]))
            + JUMP_MARGIN_M_S))
        if not gross:
            aa = min(a, ALIGN_MAX_GAIN)
            self._al_S += aa * (np.outer(vd, vd) - self._al_S)
            self._al_M += aa * (np.outer(vf, vd) - self._al_M)
            self._al_F += aa * (vf * vf - self._al_F)
            self._al_w1 += aa * (1.0 - self._al_w1)
            self._al_w2 = (1.0 - aa) ** 2 * self._al_w2 + aa * aa
        # The DVL's motion along a direction goes somewhere the tags do not
        # show it (beyond three standard errors): hold the DVL back...
        als = self.alignment()
        self._al_min = min(al["align"] for al in als) if als else None
        off = [al for al in als
               if al["align"] + 3.0 * al["sigma"] < cfg["agree_ratio_min"]]
        primary = self.selection == "primary"
        if primary and off and self._al_hold is None:
            self.n_hold["alignment"] += 1
        self._al_hold = off[0] if off else None
        self._al_hold_t = t if off else None
        # ...and if the tags SEE that motion (gain), it is evidence about the
        # mount: the trip below, once it has stood for ALIGN_HOLD_S.
        bad = next((al for al in off if al["gain"] >= COHERENT_MIN), None)
        if bad is None:
            self._al_bad_t0 = None
        elif self._al_bad_t0 is None or gap:
            self._al_bad_t0 = t
        if primary:
            self._monitor_primary(t, vd, vf, bad, bool(off), gap_s, dtb)
            return
        if not self._armed():
            return
        if self._res_lp > cfg["agree_trip_m_s"]:
            # Say which of the two it looks like: when the tags moved less
            # than half as much as the DVL says, the operator's problem is
            # the DVL's lock, not the mount. (The tag side's own noise power,
            # as the proof measured it, is not motion.)
            rms = float(self.proof["rms_m_s"]) if (
                self.proof and self.proof.get("rms_m_s") is not None) else 0.0
            dvl_only = max(0.0, self._pw_f - rms * rms) < 0.25 * self._pw_d
            self._trip(f"|v_dvl - v_tag| sustained ({self._res_lp * 100:.0f} "
                       f"> {cfg['agree_trip_m_s'] * 100:.0f} cm/s)"
                       + (" — the tags did not show that motion: a false "
                          "bottom lock rather than the mount"
                          if dvl_only else ""))
            return
        if bad is not None and t - self._al_bad_t0 >= ALIGN_HOLD_S:
            self._trip(f"DVL and tag velocity disagree along body bearing "
                       f"{bad['along_deg']:+.0f} deg (alignment "
                       f"{bad['align']:+.2f} +- {3.0 * bad['sigma']:.2f} < "
                       f"{cfg['agree_ratio_min']:.2f})")

    def _slow_monitors(self, t: float, d, dn: float, vd, vf,
                       glitch: bool = False) -> tuple:
        """2026-10-16, both modes: the SLOW residual — the pair difference
        ``d`` (|d| = ``dn``, unclipped) clipped to agree_slow_clip_m_s and
        low-passed over agree_slow_s — and the slow powers |v_dvl|^2,
        |v_tag|^2 behind ``scale_ratio`` (fix round 1: a ``glitch`` pair —
        gross AND beyond any scale error up to SCALE_PAIR_MAX at its speed —
        is left out of both; the DVL's power takes the pair clipped like the
        residual). A pair gap up to SLOW_GAP_S keeps them running, the
        interval counted at most SLOW_DT_CAP_S (the bucket's, returned, at
        most ESC_DT_CAP_S). -> (bucket dt, gap)."""
        c = self.cfg
        last, self._slow_t = self._slow_t, t
        gap_s = last is None or not 0.0 < t - last <= SLOW_GAP_S
        dts = 0.0 if gap_s else min(t - last, SLOW_DT_CAP_S)
        dtb = 0.0 if gap_s else min(t - last, ESC_DT_CAP_S)
        a_s = min(1.0, dts / c["agree_slow_s"])
        c2 = float(c["agree_slow_clip_m_s"])
        d2 = d * (c2 / dn) if dn > c2 else d
        if self._slow_vec is None:
            self._slow_vec = np.zeros(2)
        self._slow_vec = self._slow_vec + a_s * (d2 - self._slow_vec)
        self._r_slow = float(np.hypot(self._slow_vec[0], self._slow_vec[1]))
        self.r_slow_max = max(self.r_slow_max, self._r_slow)
        # The slow POWERS (the scale evidence) leave a GLITCH pair out and
        # take the DVL's velocity clipped like the residual (vf + d2) on the
        # rest. Fed every pair unclipped, ONE short large glitch (+0.5-0.8
        # m/s for 0.5-1 s, which the blend's gross check never flew) held
        # scale_ratio above esc_scale_ratio for ~4 s (the 4-s memory) and
        # latched a right DVL "persistently": 20 of 32 aid-level glitch runs,
        # 12 of 30 real-session worker replays [측정 DTF fix1/out/
        # repro_glitch_aid_before.json, verify/primary_replay/
        # glitch_spot.txt]. The clip alone still left a 3.15-s bucket (0.79 of
        # the latch) for a 1.5-s glitch where the tags see just over 5 cm/s
        # rms; leaving out every gross pair made a 3x scale in a 24/16 cm/s
        # oscillation never latch; this rule: bucket 0.0 on all 240 slow-
        # motion glitch runs, scale latches as with the clip [측정 DTF fix1/
        # out/glitch_slowmotion_aid*.json, scale_faults_aid_*.json].
        if not glitch:
            vdc = vd if dn <= c2 else np.asarray(vf, float).reshape(2) + d2
            self._pd_s += a_s * (float(vdc @ vdc) - self._pd_s)
            self._pf_s += a_s * (float(vf @ vf) - self._pf_s)
        return dtb, gap_s

    def _monitor_primary(self, t: float, vd, vf, bad, off: bool,
                         gap_s: bool, dtb: float) -> None:
        """(primary) What the pair means, without a trip on disagreement:

            stage 1  the HOLDS' own conditions — the slow residual above
                     agree_hold_m_s (released below agree_release_m_s), the
                     alignment verdict (``_al_hold``, set above) and a CURRENT
                     fit refutation — which :meth:`disagree_hold` turns into
                     a self-releasing hold through the blend;
            stage 2  the ESCALATION bucket: + the pair interval while there is
                     lasting-disagreement evidence (the slow residual above
                     the hold level, the alignment's COHERENT verdict, a
                     coherent current refutation, the DVL reading more than
                     esc_scale_ratio x the motion the tags see), - esc_leak x
                     it otherwise; live only while armed and not latched
                     (frozen otherwise), capped at esc_latch_s. It LATCHES
                     (the shipped _trip: the proof is thrown away) at
                     esc_latch_s, or on esc_repeat_n counted episodes within
                     esc_repeat_window_s — the second one queues a warning.
                     An episode counts once its bucket has reached
                     EPISODE_MIN_S AND it has carried difference, fit or
                     scale evidence (EPISODE_KINDS; alignment alone is
                     recorded in ep_log_align, fix round 1).

        Only an over-reading DVL is scale evidence: an under-reading one
        refutes coherently (A = 2 I has gain 2) and reads alignment-bad, and
        an under-read ratio is what a zero lock looks like, which must stay
        transient."""
        c = self.cfg
        r = float(self._r_slow or 0.0)
        if not self._hold_diff and r > c["agree_hold_m_s"]:
            self._hold_diff = True
            self.n_hold["difference"] += 1
        elif self._hold_diff and r < c["agree_release_m_s"]:
            self._hold_diff = False
        fit_any = self._fit_current(t)
        fit_coh = bool(fit_any and self._refuted_coh)
        if fit_coh and not self._fit_was:
            self.n_hold["fit"] += 1
        if fit_any and not fit_coh and not self._fit_inc_was:
            self.n_hold["fit_incoherent"] += 1
        self._fit_was, self._fit_inc_was = fit_coh, bool(fit_any and not fit_coh)
        ev = []
        if r > c["agree_hold_m_s"]:
            ev.append("difference")
        if bad is not None:
            ev.append("alignment")
        if fit_coh:
            ev.append("fit")
        k = self.scale_ratio
        if c["esc_scale_ratio"] > 0.0 and k is not None \
                and k > c["esc_scale_ratio"]:
            ev.append("scale")
        live = bool(self._armed() and not self.tripped)
        n_rep, win = int(c["esc_repeat_n"]), float(c["esc_repeat_window_s"])
        if live:
            if ev:
                if self._esc_s <= 0.0:          # an episode starts
                    self._ep_active, self._ep_t0 = True, t
                    self._ep_peak, self._ep_counted = 0.0, False
                    self._ep_kinds = set()
                self._esc_s = min(float(c["esc_latch_s"]), self._esc_s + dtb)
                self._esc_ev = "+".join(ev)
                self._ep_kinds.update(ev)
            else:
                self._esc_s = max(0.0, self._esc_s - c["esc_leak"] * dtb)
            newly = False
            if self._ep_active:
                self._ep_peak = max(self._ep_peak, self._esc_s)
                if not self._ep_counted and self._ep_peak >= EPISODE_MIN_S \
                        and self._ep_kinds & EPISODE_KINDS:
                    self._ep_counted = True
                    self._esc_onsets.append(self._ep_t0)
                    self.ep_log.append(self._ep_t0)
                    newly = True
                if self._esc_s <= 0.0:
                    self._ep_active = False
                    if not self._ep_counted \
                            and self._ep_peak >= EPISODE_MIN_S:
                        self.ep_log_align.append(self._ep_t0)
            while self._esc_onsets and t - self._esc_onsets[0] > win:
                self._esc_onsets.popleft()
            if newly and n_rep > 1 and len(self._esc_onsets) == n_rep - 1:
                w_ = (t, f"the DVL disagreed with the tag velocity again "
                         f"({len(self._esc_onsets)} times in the last "
                         f"{win:g} s)")
                self.warn_log.append(w_)
                self._warn_new.append(w_)
        self.esc_max = max(self.esc_max, self._esc_s)
        if self.ev_log is not None:
            self.ev_log.append((t, dtb, float(vd[0]), float(vd[1]),
                                float(vf[0]), float(vf[1]), r,
                                float(bad is not None), float(fit_coh),
                                float(off), float(gap_s),
                                float("scale" in ev), self._esc_s))
        if not live:
            return
        if self._esc_s >= c["esc_latch_s"]:
            why = (f"DVL disagreed with the tag velocity persistently "
                   f"({self._esc_s:.1f} s of evidence; now: "
                   + self._evidence_text(ev, r, bad, k) + ")")
            self.n_esc["persistent"] += 1
            self.esc_log.append((t, why))
            self._trip(why)
        elif n_rep > 0 and len(self._esc_onsets) >= n_rep and ev:
            why = (f"DVL disagreed with the tag velocity repeatedly "
                   f"({len(self._esc_onsets)} times in the last {win:g} s; "
                   f"now: " + self._evidence_text(ev, r, bad, k) + ")")
            self.n_esc["repeated"] += 1
            self.esc_log.append((t, why))
            self._trip(why)

    @staticmethod
    def _evidence_text(ev, r, bad, scale) -> str:
        """The latch text names only the evidence on the latching pair."""
        what = []
        if "difference" in ev:
            what.append(f"residual {r * 100:.1f} cm/s")
        if "alignment" in ev and bad is not None:
            what.append(f"alignment {bad['align']:+.2f} along "
                        f"{bad['along_deg']:+.0f} deg")
        if "fit" in ev:
            what.append("the fit refutes the mount")
        if "scale" in ev and scale is not None:
            what.append(f"the DVL reads {scale:.1f}x the motion the tags see")
        return ", ".join(what)

    def alignment(self) -> list:
        """The fast agreement test's numbers, strongest direction first.

        Over the last ``agree_trip_s`` of aligned pairs (exponentially
        weighted), for each principal direction ``e`` of the DVL velocity
        whose rms speed is above ``agree_ratio_v_min_m_s``:

            b      = sum(v_tag (e.v_dvl)) / sum((e.v_dvl)^2)
                     — where the least-squares map tag <- DVL sends e
            align  = 1 - |b - e|       1 the same motion, -1 reversed,
                                       1 - sqrt(2) a quarter turn
            sigma  the standard error of |b - e|, from the residual of the
                   two-direction least-squares fit (so tag noise, a jump
                   below the gross gate, anything the linear map does not
                   explain WIDENS it instead of moving the verdict)

        Principal directions rather than body axes: a mount a few degrees
        out leaks surge into the DVL's sway, and a per-axis slope reads that
        leak as a disagreement along sway; along the directions the DVL
        velocity is uncorrelated in, a rotation by D reads 2 sin(D/2)
        whichever way the vehicle moves.

        ``gain`` = |b|: how much of the DVL's motion along e the tags see
        at all. 1 for any rotation or mirror; ~0 when only the DVL "moved"
        (its noise, a glitch, a false lock at rest) — which reads align ~ 0
        too, and is why the trip asks for gain >= COHERENT_MIN as well
        (a low align alone holds the DVL back without a latch).

        -> [{"along_deg", "align", "gain", "sigma", "speed_m_s", "n_eff"}];
        empty while there are too few pairs or no direction is moving
        enough."""
        w1, w2 = self._al_w1, self._al_w2
        if w1 <= 0.0 or w2 <= 0.0:
            return []
        n_eff = w1 * w1 / w2
        if n_eff < ALIGN_MIN_PAIRS:
            return []
        S, M, F = self._al_S / w1, self._al_M / w1, self._al_F / w1
        lam, E = np.linalg.eigh(S)
        # The principal directions are uncorrelated in v_dvl, so the
        # two-regressor least squares is two one-regressor ones; what
        # neither explains is the residual (summed over both components).
        B = np.zeros((2, 2))                # column k: where the map sends e_k
        explained = np.zeros(2)
        for k in (0, 1):
            if lam[k] > 1e-12:
                B[:, k] = (M @ E[:, k]) / lam[k]
                explained += B[:, k] ** 2 * lam[k]
        s2 = float(np.sum(np.maximum(F - explained, 0.0)))
        floor2 = self.cfg["agree_ratio_v_min_m_s"] ** 2
        share_min = ALIGN_MIN_SHARE * float(max(lam[0], 0.0) + max(lam[1], 0.0))
        out = []
        for k in (1, 0):
            lk = float(lam[k])
            if not lk > floor2 or lk < share_min:
                continue
            e, b = E[:, k], B[:, k]
            sig = math.sqrt(s2 / ((n_eff - 2.0) * lk))
            deg = math.degrees(math.atan2(e[1], e[0]))
            if deg > 90.0:
                deg -= 180.0
            elif deg <= -90.0:
                deg += 180.0
            out.append({"along_deg": float(deg),
                        "align": float(1.0 - np.hypot(*(b - e))),
                        "gain": float(np.hypot(b[0], b[1])),
                        "sigma": float(sig), "speed_m_s": math.sqrt(lk),
                        "n_eff": float(n_eff)})
        return out

    # ------------------------------------------------------------- reporting
    @property
    def sample(self):
        return self._s

    @property
    def v_body(self):
        """Body-origin velocity of the newest sample, or None."""
        return None if self._v_body is None else self._v_body.copy()

    @property
    def residual_m_s(self) -> float | None:
        return self._res_lp

    def state(self, t_now: float, used: bool, why: str | None = None,
              code: str | None = None, w: float | None = None) -> tuple:
        """-> (dvl.STATES member, note) for the panel. ``why`` = the reason
        the tick itself was given for not flying on the DVL, if the caller
        has it; asking :meth:`velocity` again here could answer differently
        (the reference has moved on since — review 2026-10-02).
        (primary: ``code`` / ``w`` = the tick's usage code and blend weight;
        see :meth:`_state_primary`.)"""
        if self.selection == "primary":
            return self._state_primary(t_now, used, why, code, w)
        s = self._s
        if s is None or float(t_now) - float(s.t_rx) > max(
                2.0, self.cfg["stale_s"]):
            return ST_NO_LINK, "no reports"
        if self.tripped:
            f = self.agree.fit()
            return ST_TRIPPED, (
                self.tripped
                + (f"; now: {f['why']}" if f["verdict"] == "refuted" else "")
                + " — set the switch OUT then IN to re-arm; that needs a "
                  "proof made after this trip")
        ok, data_why = self.data_ok(t_now)
        if not ok:
            return ST_NO_LOCK, data_why
        if used:
            return ST_IN_USE, ""
        if not self.use:
            f = self.agree.fit()
            return ST_SHADOW, (
                "proven" if self.proven else
                (f["why"] or "not proven yet")
                + (f" — shown wrong earlier: {self.against}"
                   if self.against and f["verdict"] != "refuted" else ""))
        if (self.cfg["require_proven"] or self.against) and not self.proven:
            return ST_UNPROVEN, (self.agree.fit()["why"] or "not proven yet")
        if why is None:
            _v, why = self.velocity(t_now, count=False)
        return ST_FALLBACK, why

    def _state_primary(self, t_now: float, used: bool, why, code, w) -> tuple:
        """(primary) The panel's words. Never ST_FALLBACK: "held" is only
        ever a hold that RETURNS BY ITSELF (a flag, a disagreement, gross),
        and the button face says TRIPPED only for a latch. A lost lock says
        "returns by itself" only when the DVL would be used again with the
        lock back (IN, mount measured, proven or not required): not in
        shadow, not unproven (fix round 1). A run the worker did not arm is
        the worker's to say (_publish_dvl_primary: not_armed)."""
        s = self._s
        if s is None or float(t_now) - float(s.t_rx) > max(
                2.0, self.cfg["stale_s"]):
            return ST_NO_LINK, "no reports"
        if self.tripped:
            f = self.agree.fit()
            return ST_TRIPPED, (
                self.tripped
                + (f"; now: {f['why']}" if f["verdict"] == "refuted" else "")
                + " — latched: fly a proving pass in surge AND sway with tags "
                  "in view, then set the switch OUT then IN")
        ok, data_why = self.data_ok(t_now)
        if not ok:
            back = bool(self.use and self.mount_measured and not (
                (self.cfg["require_proven"] or self.against)
                and not self.proven))
            return ST_NO_LOCK, (f"{data_why} — returns by itself" if back
                                else data_why)
        if used:
            return ST_IN_USE, ("" if w is None or w >= 1.0 else
                               f"fading in ({w * 100:.0f} % DVL)")
        if not self.use:
            f = self.agree.fit()
            return ST_SHADOW, (
                "proven" if self.proven else
                (f["why"] or "not proven yet")
                + (f" — shown wrong earlier: {self.against}"
                   if self.against and f["verdict"] != "refuted" else ""))
        if (self.cfg["require_proven"] or self.against) and not self.proven:
            return ST_UNPROVEN, (self.agree.fit()["why"] or "not proven yet")
        if code is None:
            _ok, why2, code = self.usable(t_now)
            why = why or why2
        code = str(code or "")
        if code.startswith("flag_") or code.startswith("dis_") \
                or code == "gross":
            note = f"{why or code} — returns by itself"
            if w is not None and 0.0 < w < 1.0:
                note += f" ({w * 100:.0f} % DVL still)"
            n = self.episodes_recent
            if n >= 1 and int(self.cfg["esc_repeat_n"]) > 0:
                note += (f"; {n} of {self.cfg['esc_repeat_n']} in the last "
                         f"{self.cfg['esc_repeat_window_s']:g} s")
            return ST_HELD, note
        if code in ("no_tag", "bridge"):
            return ST_NO_TAG, (why or "no tag state this tick (the blend "
                                      "waits for it)")
        # not armed this run, a code fault — and anything else that does not
        # return by itself within the run (an unmeasured mount)
        return ST_NOT_ARMED, str(why or code)

    def meta(self) -> dict:
        m = self._meta_shipped()
        c = self.cfg
        primary = self.selection == "primary"
        m["monitor_now"].update({
            "residual_slow_m_s": self._r_slow,
            "escalation_s": float(self._esc_s) if primary else None,
            "align_min": self._al_min, "scale_ratio": self.scale_ratio,
            "episodes_recent": self.episodes_recent})
        m.update({
            # 2026-10-16 (DTF SPEC.md §6.2). RECORD BOUNDARY: never pool
            # velocity statistics across `selection`.
            "selection": self.selection,
            # not inside `gates`: a replay rebuilds a dvl: block from gates +
            # agreement_cfg, and resolve() refuses an unknown key
            "reentry_s_used": not primary,
            "blend": {"hold_s": c["blend_hold_s"], "xfade_s": c["blend_xfade_s"],
                      "gross_m_s": c["blend_gross_m_s"],
                      "gross_tau_s": c["blend_gross_tau_s"],
                      "stale_s": c["stale_s"], "dvl_w": False, "dt_s": self.dt},
            "flags": {"fom_m_s": c["flag_fom_m_s"],
                      "alt_jump_m": c["flag_alt_jump_m"],
                      "alt_window_s": c["flag_alt_window_s"],
                      "min_beams": c["flag_min_beams"],
                      "reacquired": bool(c["flag_reacquired"]),
                      "counts": {k: int(self.n_flag.get(k, 0)) for k in (
                          "flag_fom", "flag_alt", "flag_beams", "flag_reacq",
                          "flagged")}},
            "disagreement": {
                "slow_s": c["agree_slow_s"], "clip_m_s": c["agree_slow_clip_m_s"],
                "hold_m_s": c["agree_hold_m_s"],
                "release_m_s": c["agree_release_m_s"],
                "slow_gap_s": SLOW_GAP_S, "slow_dt_cap_s": SLOW_DT_CAP_S,
                "hold_drain_s": HOLD_DRAIN_S,
                "hold_onsets": {k: int(self.n_hold.get(k, 0)) for k in (
                    "difference", "alignment", "fit", "fit_incoherent")},
                "slow_residual_max_m_s": float(self.r_slow_max)},
            "escalation": {
                "latch_s": c["esc_latch_s"], "leak": c["esc_leak"],
                "scale_ratio": c["esc_scale_ratio"],
                "scale_v_min_m_s": c["esc_scale_v_min_m_s"],
                "repeat_n": c["esc_repeat_n"],
                "repeat_window_s": c["esc_repeat_window_s"],
                "episode_min_s": EPISODE_MIN_S, "dt_cap_s": ESC_DT_CAP_S,
                # fix round 1: what makes an episode count toward repeat_n,
                # and the episodes alignment evidence alone made (not counted)
                "episode_kinds": sorted(EPISODE_KINDS),
                "max_s": float(self.esc_max),
                "latches": {k: int(self.n_esc.get(k, 0))
                            for k in ("persistent", "repeated")},
                "episodes": len(self.ep_log),
                "episodes_alignment_only": len(self.ep_log_align),
                "warnings": len(self.warn_log)},
            "rung_held_ticks": int(self.n_rung_held),
            "inactive_keys": ({k: c[k] for k in INACTIVE_IN_PRIMARY}
                              if primary else {}),
            "governed_by": ({k: list(v) for k, v in
                             INACTIVE_IN_PRIMARY.items()} if primary else {}),
        })
        return m

    def _meta_shipped(self) -> dict:
        f = self.agree.fit()
        c = self.cfg
        return {
            "use": bool(self.use),
            "source": self.source or None,
            "host": c["host"], "port": c["port"],
            "consumes": ("nu[0:2] (surge, sway) and the fix-age bridging; "
                         "nu[2] (heave rate) stays on the tag finite "
                         "difference — the proof is horizontal only"),
            "mount_measured": bool(self.mount_measured),
            "mount_rpy_deg": c["mount_rpy_deg"],
            "mount_t_flu": c["mount_t_flu"],
            "mount_frame": ("R_frd_dvl = Rz(yaw) Ry(pitch) Rx(roll); device "
                            "frame x fwd (LED), y starboard, z down [스펙]; "
                            "lever arm in body FLU from the cam_t_flu origin"),
            "device_config": self.device_config,
            "device_config_changes": list(self.device_config_changes),
            "gates": {k: c[k] for k in ("stale_s", "fom_max",
                                        "min_altitude_m", "v_max_m_s",
                                        "reentry_s", "a_max_m_s2",
                                        "innov_max_m_s")},
            "require_proven": bool(c["require_proven"]),
            "proven": bool(self.proven),
            "proof": self.proof,
            "agreement_now": f,
            "agreement_cfg": {k: c[k] for k in (
                "agree_min_pairs", "agree_min_excitation_m_s", "agree_tol",
                "agree_v_min_m_s", "agree_window_s", "agree_refute_n",
                "agree_trip_m_s", "agree_trip_s", "agree_ratio_min",
                "agree_ratio_v_min_m_s")},
            "tripped": self.tripped or None,
            "against": self.against or None,
            "innovation_hold": self._innov_hold or None,
            "trips": int(self.n_trips),
            "last_trip": self.last_trip or None,
            "samples": int(self.n_samples),
            "samples_valid": int(self.n_valid),
            "samples_out_of_order": int(self.n_out_of_order),
            # Ticks the innovation gate held back, engaged or not, over the
            # whole SESSION. The per-run count is rows["fallback: DVL
            # innovation"] in the worker's meta.
            "innovation_rejects": int(self.n_innovation),
            "monitor_pairs": int(self.n_monitor),
            "gross_pairs": int(self.n_gross),
            "monitor_now": {"residual_m_s": self._res_lp,
                            "alignment": self.alignment()},
            "provenance": ("HARDWARE-UNVERIFIED (written 2026-10-01 before a "
                           "DVL was mounted); thresholds [예측], device facts "
                           "[스펙] — see control/dvl.py"),
        }
