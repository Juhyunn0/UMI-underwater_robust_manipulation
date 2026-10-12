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

No Qt, no sockets, no dobmpc: plain numpy, so it is tested bare
(rov_gui/tests/test_dvl.py). The TCP client lives in backends/dvl.py.
"""

from __future__ import annotations

import json
import math
from collections import deque

import numpy as np

from .state_assembler import rot_zyx

S_FLU_FRD = np.diag([1.0, -1.0, -1.0])     # same involution as geometry.py

#: ``MpcStatus.dvl_state`` vocabulary ("" = no DVL was built).
ST_NO_LINK = "no_link"      # built, but no (fresh) report
ST_NO_LOCK = "no_lock"      # reports arrive; this one fails a data gate
ST_SHADOW = "shadow"        # good data, operator has it OUT: recorded only
ST_UNPROVEN = "unproven"    # operator has it IN, the mount is not proven yet
ST_TRIPPED = "tripped"      # evidence it is wrong; latched until re-armed
ST_FALLBACK = "fallback"    # IN, good data, held back this tick
ST_IN_USE = "in_use"        # nu[0:2] is the DVL's
STATES = (ST_NO_LINK, ST_NO_LOCK, ST_SHADOW, ST_UNPROVEN, ST_TRIPPED,
          ST_FALLBACK, ST_IN_USE)

VEL_TAG_FD = "tag_fd"
VEL_DVL = "dvl"

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
    # tick. [예측]
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
    "agree_refute_n": 15,
    # Runtime monitors, each latching a trip. All three are fed by the SAME
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
}

KEYS = tuple(DEFAULTS)

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
    for k in ("enabled", "use", "require_proven"):
        if not isinstance(cfg[k], bool):
            raise ValueError(f"dvl.{k} must be true or false, got {cfg[k]!r}")
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
    """

    def __init__(self, cfg: dict | None = None, fd_lp_gain: float = 0.507):
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
        clears a trip — and only if ``refusal`` lets it through."""
        why = self.refusal(bool(on))
        if not why:
            self.use = bool(on)
            if on:
                self.tripped = ""
        return why

    # --------------------------------------------------------------- inputs
    def note_sample(self, s, omega=None) -> None:
        """One ``state.DvlSample``. ``omega`` = the body rates (p, q, r) at
        the sample's ``t_valid``, for the lever arm."""
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
        out: a proof, or the switch IN."""
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
        clip = float(cfg["innov_max_m_s"])
        if dn > clip:
            d = d * (clip / dn)
        if self._res_vec is None:
            self._res_vec = np.zeros(2)
        self._res_vec = self._res_vec + a * (d - self._res_vec)
        self._res_lp = float(np.hypot(self._res_vec[0], self._res_vec[1]))
        self._pw_d += a * (float(vd @ vd) - self._pw_d)
        self._pw_f += a * (float(vf @ vf) - self._pw_f)
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
        off = [al for al in als
               if al["align"] + 3.0 * al["sigma"] < cfg["agree_ratio_min"]]
        self._al_hold = off[0] if off else None
        self._al_hold_t = t if off else None
        # ...and if the tags SEE that motion (gain), it is evidence about the
        # mount: the trip below, once it has stood for ALIGN_HOLD_S.
        bad = next((al for al in off if al["gain"] >= COHERENT_MIN), None)
        if bad is None:
            self._al_bad_t0 = None
        elif self._al_bad_t0 is None or gap:
            self._al_bad_t0 = t
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

    def state(self, t_now: float, used: bool, why: str | None = None
              ) -> tuple:
        """-> (dvl.STATES member, note) for the panel. ``why`` = the reason
        the tick itself was given for not flying on the DVL, if the caller
        has it; asking :meth:`velocity` again here could answer differently
        (the reference has moved on since — review 2026-10-02)."""
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

    def meta(self) -> dict:
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
