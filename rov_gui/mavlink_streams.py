#!/usr/bin/env python3
"""mavlink_streams.py — which autopilot IMU streams this station asks ArduSub for, and how fast (2026-10-15).

Operator-approved station fix 1c of RL_controller/logs/eval/pool_trips_1013_rootcause (= RC; fix-design station
item 1). Paths below: impl-rates = RL_controller/logs/eval/station_fix_20261015/impl-rates.

Until 2026-10-15 the station asked for RAW_IMU, SCALED_IMU2 and ATTITUDE all at ``--mavlink-rate`` (default 200).
The vehicle delivered ~166.6 Hz of each: ~500 of the ~552 msg/s it pushes to EVERY BlueOS endpoint (this station's
telemetry port, its command port, QGC's) [측정: 27 recordings = 28 *_rov.jsonl minus one offline copy, 2026-08-09..10-13, impl-rates/out/
rov_stream_scan.json]. On 2026-10-13 a vehicle-side downlink queue formed on top of that load whenever another
client pulled the parameter list, and the rl_pwm policy flew on an ATTITUDE 0.43 s old at the median [측정:
RC/synth/out/synth_checks.json C1_age, C4_queue].

The request is now per message. Every consumer of the three streams in this station, and what it needs, is tabled
in impl-rates/NOTES.md; in short:

  ATTITUDE 50 Hz — read ONCE per 20 Hz VehicleImu snapshot (hardware.VehicleWorker._publish_imu) by every LOW
      (state_assembler roll/pitch/p/q/r, the RL observations, PID 7d, the pwm interlocks), the EKF shadow (its IMU
      input is that 20 Hz snapshot), imu_dr's settle window and 'vehicle' attitude, the DVL lever-arm rate history
      and the tag solver's roll/pitch hint; Telemetry reads it at 10 Hz. Every ATTITUDE also feeds the
      vehicle-clock map behind the pwm LOWs' AGE gate (mavlink_log.VehicleClockMap), whose age INCLUDES the
      sample's own age. Replaying the 21 healthy recordings (1.82 h) with ATTITUDE decimated to ~47-50 Hz: tick age
      p99 69-86 ms (57-78 as recorded), the rolling 10-s p99 at most 0.110 s against the 0.12 s engage refusal,
      0 trips, 0 p99 refusals; at ~28 Hz healthy links start being refused (16 checks), at 20 Hz 122
      [유도: impl-rates/out/age_at_rates.json, the station's own VehicleClockMap + PwmInterlocks]. Higher buys
      little: 100 Hz tops out at 0.106 s — the link's transit jitter, not the sample age, dominates.
  RAW_IMU 20 Hz — the autopilot's LIVE accelerometer (|a| p50 999-1001 mg [측정: rov_stream_scan.json]):
      the recording beside a depth/controller run (sensorlog *_rov.jsonl), the IMU (ROV) health row (>= 10 Hz),
      and VehicleImu.ax..az only when SCALED_IMU2 is off. 20 Hz = the snapshot that would read it.
  SCALED_IMU2 20 Hz — its accel AND gyro are ZERO in every record on this vehicle (27/27 recordings) [측정:
      rov_stream_scan.json], yet it is what VehicleImu.ax..az have always been read from (``ACCEL_SOURCE`` below):
      the EKF's accelerometer, the dobmpc EAOB's nudot and the station bridge's DVL rung. Turning it off — the fix
      design's proposal — would hand those consumers a live accelerometer they have never run on. Replayed on two
      10-13 runs, the EKF's heave error goes 0.74-0.76 m -> 3-6 mm (p50), but its horizontal speed only
      1.76 -> 1.03 and 0.17 -> 0.13 m/s (p50), against 0.01 and 0.11 m/s with no accelerometer fused [유도:
      impl-rates/out/ekf_accel_source.json]. All 7 dobmpc hardware runs that flew (2026-08-14..08-23) held the
      EAOB heave estimate at its +45 N clip (83-96 % of rows), and all 60 098 imu records of the 71
      *_ekf_inputs.jsonl carry acc 0 [측정: impl-rates/out/accel_consumers_evidence.json]. Changing what those
      consumers are fed is a tuning decision for the operator, not a stream-rate side effect, so SCALED_IMU2 stays
      ON at the snapshot rate; ``--mavlink-scaled-imu2-rate 0`` is the deliberate switch (it moves the
      accelerometer to RAW_IMU, see ``ACCEL_SOURCE``).

Requested total: 90 msg/s of IMU stream instead of 600 requested / ~500 delivered. The vehicle sends about what is
asked at these rates: the 10 Hz requests arrive 100 ms apart on the vehicle clock on a healthy link [측정:
impl-rates/out/send_spacing.json], and a 50 Hz request measured 62 Hz on 2026-08-06 (rov_gui/README.md). [예측] ~5x
less stream traffic keeps one parameter download from filling the vehicle-side queue; pool procedure step 0c
(RC fix-design) is the check.

``--mavlink-rate N`` keeps the OLD request (all three at N, the accelerometer read as before); giving it together
with a per-message flag is refused at launch. The resolved request is logged at setup and recorded in the run meta
(``mavlink_streams``) and the *_rov.json sidecar (``stream_request``).

Pure Python, no Qt, no depthai: rov_gui/__main__.py validates the flags with it at launch for every source.
"""

from __future__ import annotations

import math

#: The three messages this module decides. Every other rate the reader asks for stays
#: c3_camera/mavlink_log.DEFAULT_RATES_HZ.
STREAMS = ("ATTITUDE", "RAW_IMU", "SCALED_IMU2")

#: Hz per message when its flag is not given (the reasons are the module docstring).
DEFAULT_HZ = {"ATTITUDE": 50, "RAW_IMU": 20, "SCALED_IMU2": 20}

#: The lowest value each flag accepts. ATTITUDE never off (every LOW flies on it); RAW_IMU never off (the live
#: accelerometer, and the only one once SCALED_IMU2 is off); SCALED_IMU2 0 = ask the vehicle to stop sending it.
MIN_HZ = {"ATTITUDE": 1, "RAW_IMU": 1, "SCALED_IMU2": 0}

#: ATTITUDE below this and the pwm LOWs' AGE gate loses its healthy-link margin (replay above: ~28 Hz refuses).
ATT_AGE_FLOOR_HZ = 50

#: Command-line flag and the namespace attribute argparse makes of it.
FLAGS = {"ATTITUDE": "--mavlink-att-rate", "RAW_IMU": "--mavlink-imu-rate",
         "SCALED_IMU2": "--mavlink-scaled-imu2-rate"}
OPT_ATTR = {"ATTITUDE": "mavlink_att_rate", "RAW_IMU": "mavlink_imu_rate",
            "SCALED_IMU2": "mavlink_scaled_imu2_rate"}
LEGACY_FLAG, LEGACY_ATTR = "--mavlink-rate", "mavlink_rate"

#: Where VehicleImu.ax..az come from: the first of these that has arrived (hardware.VehicleWorker._publish_imu).
#: The order every --mpc build has used. With SCALED_IMU2 off it is RAW_IMU ONLY — never a SCALED_IMU2 record left
#: in the worker's latest-of-each-type table from before the request (or from another client's stream).
ACCEL_SOURCE = ("SCALED_IMU2", "RAW_IMU")
ACCEL_SOURCE_SCALED_IMU2_OFF = ("RAW_IMU",)


def _hz(flag: str, v) -> int | float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise ValueError(f"{flag} {v!r} is not a rate in Hz")
    return int(v) if float(v).is_integer() else float(v)


def resolve(legacy_hz=None, attitude_hz=None, raw_imu_hz=None, scaled_imu2_hz=None) -> dict:
    """The station's IMU-stream request from its four flags (None = not given).

    -> ``{"policy": "per_message" | "legacy", "legacy_hz": N | None,
    "rates_hz": {"ATTITUDE": hz, "RAW_IMU": hz, "SCALED_IMU2": hz (0 = off)}, "off": [names asked to stop],
    "accel_source": [the VehicleImu.ax..az order]}``.

    Raises ValueError, whose text is the launch refusal, on ``--mavlink-rate`` given together with a per-message flag
    or on a value below its floor. Never guesses: a refused request is better than one nobody asked for."""
    per = {"ATTITUDE": attitude_hz, "RAW_IMU": raw_imu_hz, "SCALED_IMU2": scaled_imu2_hz}
    if legacy_hz is not None:
        typed = [FLAGS[k] for k in STREAMS if per[k] is not None]
        if typed:
            raise ValueError(
                f"{LEGACY_FLAG} {legacy_hz} sets RAW_IMU, SCALED_IMU2 and ATTITUDE together (the request before "
                f"2026-10-15) and cannot be combined with {', '.join(typed)}. Give one or the other.")
        n = _hz(LEGACY_FLAG, legacy_hz)
        if n < 1:
            raise ValueError(f"{LEGACY_FLAG} {legacy_hz}: the legacy request needs >= 1 Hz (it sets ATTITUDE, which "
                             f"every LOW flies on).")
        return {"policy": "legacy", "legacy_hz": n, "rates_hz": {k: n for k in STREAMS}, "off": [],
                "accel_source": list(ACCEL_SOURCE)}
    rates, off = {}, []
    for k in STREAMS:
        hz = DEFAULT_HZ[k] if per[k] is None else _hz(FLAGS[k], per[k])
        if hz < MIN_HZ[k]:
            why = {"ATTITUDE": "every LOW flies on ATTITUDE; it cannot be off",
                   "RAW_IMU": "RAW_IMU is the live accelerometer and the IMU (ROV) health row; it cannot be off",
                   "SCALED_IMU2": "0 turns it off; a rate cannot be negative"}[k]
            raise ValueError(f"{FLAGS[k]} {per[k]}: below {MIN_HZ[k]} Hz — {why}.")
        rates[k] = hz
        if hz == 0:
            off.append(k)
    accel = ACCEL_SOURCE_SCALED_IMU2_OFF if "SCALED_IMU2" in off else ACCEL_SOURCE
    return {"policy": "per_message", "legacy_hz": None, "rates_hz": rates, "off": off, "accel_source": list(accel)}


def resolve_opts(opts) -> dict:
    """:func:`resolve` on a station namespace (an attribute that is absent counts as not given)."""
    return resolve(legacy_hz=getattr(opts, LEGACY_ATTR, None),
                   attitude_hz=getattr(opts, OPT_ATTR["ATTITUDE"], None),
                   raw_imu_hz=getattr(opts, OPT_ATTR["RAW_IMU"], None),
                   scaled_imu2_hz=getattr(opts, OPT_ATTR["SCALED_IMU2"], None))


def typed(opts) -> list:
    """The stream flags given on this namespace (not None), as command-line flags."""
    out = [LEGACY_FLAG] if getattr(opts, LEGACY_ATTR, None) is not None else []
    return out + [FLAGS[k] for k in STREAMS if getattr(opts, OPT_ATTR[k], None) is not None]


def requested_msg_per_s(res: dict) -> float:
    """What the request asks for in total, msg/s (an off stream counts 0)."""
    return float(sum(max(0.0, float(v)) for v in res["rates_hz"].values()))


def describe(res: dict) -> str:
    """One line for the log: the three rates and which rule set them."""
    rates = ", ".join(f"{k} {'off' if res['rates_hz'][k] == 0 else str(res['rates_hz'][k]) + ' Hz'}"
                      for k in STREAMS)
    if res["policy"] == "legacy":
        return (f"{rates} — LEGACY {LEGACY_FLAG} {res['legacy_hz']} (the request before 2026-10-15; at 200 it was "
                f"~500 msg/s of IMU stream, the 10-13 load)")
    tail = "2026-10-15 per-message request"
    if res["off"]:
        tail += f"; {', '.join(res['off'])} asked to STOP, accelerometer from {'/'.join(res['accel_source'])}"
    return f"{rates} ({tail}; {LEGACY_FLAG} N restores the old one)"


__all__ = ["STREAMS", "DEFAULT_HZ", "MIN_HZ", "ATT_AGE_FLOOR_HZ", "FLAGS", "OPT_ATTR", "LEGACY_FLAG", "LEGACY_ATTR",
           "ACCEL_SOURCE", "ACCEL_SOURCE_SCALED_IMU2_OFF", "resolve", "resolve_opts", "typed",
           "requested_msg_per_s", "describe"]
