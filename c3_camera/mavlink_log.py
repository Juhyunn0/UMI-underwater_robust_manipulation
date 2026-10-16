#!/usr/bin/env python3
"""
mavlink_log.py — log ArduSub telemetry alongside the camera, without touching the vehicle.

Posture: PASSIVE by default
---------------------------
The user pilots with QGroundControl while this records, so the logger's first duty
is to not perturb the vehicle. By default it therefore:

* never sends a heartbeat — it does not present itself as a GCS, so it cannot be
  mistaken for one by ArduSub's GCS-loss failsafe, and cannot participate in
  RC-override arbitration;
* uses a distinct source_system so that even the packets it does send (only if
  --mavlink-set-rates is passed) are not confused with QGC's sysid 255;
* transmits nothing at all unless --mavlink-set-rates is explicitly requested.

`--mavlink-set-rates` is the one exception, and it is opt-in for a reason: raising
message rates IS a transmission to the autopilot. It is safe in normal operation,
but "safe in normal operation" is not the same as "guaranteed harmless mid-dive",
so the choice is the pilot's.

The port conflict, and why there are two transports
---------------------------------------------------
BlueOS pushes MAVLink with an endpoint `udpout -> 192.168.2.1:14550`, and QGC binds
UDP 14550 on the desktop. **Two processes cannot both receive the same unicast UDP
datagrams**; SO_REUSEPORT does not duplicate them, it load-balances, which would
silently steal packets from QGC. So there are two honest options:

  udp     — bind a second local port (default 14551) that BlueOS also pushes to.
            Full rate, lowest latency. Needs one extra BlueOS endpoint, which
            `--setup-endpoint` can add (see blueos_endpoint.py).
  rest    — poll MAVLink2Rest on the Pi (port 6040). Needs no vehicle
            configuration at all and cannot conflict with QGC, but it serves the
            *latest* value of each message, so it samples rather than captures
            every message. Fine for attitude/pressure/battery context; not a
            substitute for a real IMU stream.

Default is `udp`, because a dataset wants every message; `rest` exists so a dive
is never blocked on reconfiguring the vehicle.
"""

from __future__ import annotations

import json
import math
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field

# Messages we log, with the CSV group each lands in. Grouping keeps the output
# files small and readable rather than one giant sparse table.
IMU_MESSAGES = ("RAW_IMU", "SCALED_IMU2", "SCALED_IMU3", "HIGHRES_IMU")
TELEM_MESSAGES = ("ATTITUDE", "SCALED_PRESSURE", "SCALED_PRESSURE2", "VFR_HUD",
                  "SYS_STATUS", "BATTERY_STATUS", "SERVO_OUTPUT_RAW",
                  "HEARTBEAT", "GLOBAL_POSITION_INT", "LOCAL_POSITION_NED",
                  "AHRS2", "EKF_STATUS_REPORT")
CONTROL_MESSAGES = ("RC_CHANNELS", "RC_CHANNELS_RAW", "MANUAL_CONTROL",
                    "SERVO_OUTPUT_RAW")
# STATUSTEXT is the vehicle TALKING, not a periodic sample, and on ArduSub it
# is the ONLY way a leak is reported ("Leak Detected", MAV_SEVERITY_CRITICAL —
# see rov_gui/leak.py). Dropping it here is what made the station blind to
# water in the enclosure. It is rare and small, so passing it through costs
# nothing; consumers that only want periodic telemetry can ignore it by type.
EVENT_MESSAGES = ("STATUSTEXT",)

ALL_MESSAGES = tuple(dict.fromkeys(IMU_MESSAGES + TELEM_MESSAGES
                                   + CONTROL_MESSAGES + EVENT_MESSAGES))

# The vehicle-clock map (2026-10-15, RL_controller/logs/eval/pool_trips_1013_rootcause,
# fix-design station item 2a). `t_host` is when THIS process parsed a message, so
# it cannot see a queue that forms on the VEHICLE side of the link: on 2026-10-13
# the station's imu_age (tick - t_host) read ~0.05 s while the ATTITUDE the rl_pwm
# policy flew on was 0.43 s old at the median (0-0.9 s sawtooth), measured on the
# autopilot's own clock. VehicleClockMap puts ATTITUDE's time_boot_ms on the host
# clock instead; see its docstring for the estimator and its limits.
#: How fast the map's offset may RISE, per second of host time. The vehicle-vs-host
#: drift measured 81-240 ppm [측정: pool_trips_1013_rootcause pool-sysid / obs-pipeline
#: caveats]; 300 ppm covers it. A falling offset is followed at once (min()).
VCLK_DRIFT_PER_S = 300e-6
#: Reset rule 2 (a reboot whose new time_boot_ms is not BELOW the last one): host time
#: advanced this much more than time_boot_ms between two consecutive ATTITUDEs. The
#: largest such excess on record is 0.222 s (the 10-13 queue onsets; the 23.6-s
#: outage of 0809_202049 is 0.046 s: the vehicle clock ran on) [측정: 27 recordings,
#: RL_controller/logs/eval/station_fix_20261015/impl-age/out/host_vs_vehicle_steps.json].
VCLK_REBOOT_GAP_S = 2.0
#: Only the autopilot's clock is mapped (MAV_COMP_ID_AUTOPILOT1). Every ATTITUDE on
#: record came from 1/1 [측정: same 27 recordings, attitude_stream_scan.json].
AUTOPILOT_COMPONENT = 1


class VehicleClockMap:
    """time_boot_ms (the autopilot's clock) -> this host's ``time.monotonic()``,
    causally, one ATTITUDE at a time in ARRIVAL order.

    A rate-limited LOWER ENVELOPE of ``t_host - time_boot``::

        off_k = min(off_{k-1} + VCLK_DRIFT_PER_S * (t_host_k - t_host_{k-1}),
                    t_host_k - time_boot_k)
        t_valid_k = time_boot_k + off_k      # when the sample was TRUE, host clock

    Every sample's transit is >= 0, so ``t_host - time_boot`` is an upper bound of
    the true offset and the envelope settles on the least-delayed sample (its
    transit, ~ms, is the one thing this cannot see: ages are lower bounds by it).
    Replayed causally over 21 healthy recordings (1.82 h) this agreed with the
    offline floor fit within -3.0..+1.9 ms (p1..p99) [측정: RL_controller/logs/eval/
    pool_trips_1013_rootcause/fix-design/out/fixdesign_summary.json "replay"].

    RESET (a new map, ``t_start`` = that sample's arrival) when the source
    changes, when time_boot_ms goes BACK (a reboot; 0 backward steps in ~1.3 M
    recorded ATTITUDEs, so reordering does not trigger it), or when host time ran
    VCLK_REBOOT_GAP_S ahead of the vehicle's between two samples (a reboot that
    landed above the old time_boot_ms).

    LIMITS, by construction: a map started while a queue is standing under-reads
    the age until the queue first drains (by up to 0.77 s for the first 6.8 s of
    1013_211913 [측정: fix-design/scripts/check_online_err.py]) — hence the rl_pwm
    engage gate wants the map >= engage.per_thruster.age_map_min_s old; and a queue
    that NEVER drains is invisible to any lower envelope. Not thread-safe: one
    reader thread feeds it (MavlinkLogger)."""

    def __init__(self, drift_per_s: float = VCLK_DRIFT_PER_S,
                 reboot_gap_s: float = VCLK_REBOOT_GAP_S):
        self.drift_per_s = float(drift_per_s)
        self.reboot_gap_s = float(reboot_gap_s)
        self.n = 0                    # samples placed, all maps
        self.resets = 0               # maps started after the first
        self.last_reset: dict | None = None
        self._clear()

    def _clear(self) -> None:
        self.off: float | None = None
        self.t_start: float | None = None
        self.src = None
        self._t_prev: float | None = None
        self._tb_prev: float | None = None

    def update(self, t_host, time_boot_ms, src=None) -> float | None:
        """One sample -> its ``t_valid`` (host clock), or None when it cannot be
        placed (missing / non-numeric stamp: nothing changes)."""
        try:
            th = float(t_host)
            tb = float(time_boot_ms) / 1000.0
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(th) and math.isfinite(tb)):
            return None
        if self.off is not None:
            why = None
            if src != self.src:
                why = f"source {self.src} -> {src}"
            elif tb < self._tb_prev:
                why = (f"time_boot_ms went back {self._tb_prev:.3f} -> {tb:.3f} s "
                       f"(reboot)")
            elif (th - self._t_prev) - (tb - self._tb_prev) > self.reboot_gap_s:
                why = (f"host clock ran {(th - self._t_prev) - (tb - self._tb_prev):.1f}"
                       f" s ahead of time_boot_ms between two samples (reboot)")
            if why is not None:
                self.resets += 1
                self.last_reset = {"t_host": th, "why": why}
                self._clear()
        cand = th - tb
        if self.off is None:
            self.off, self.t_start, self.src = cand, th, src
        else:
            self.off = min(self.off + self.drift_per_s * max(0.0, th - self._t_prev),
                           cand)
        self._t_prev, self._tb_prev = th, tb
        self.n += 1
        return tb + self.off

    def summary(self) -> dict:
        return {"estimator": "rate-limited lower envelope of t_host - time_boot_ms "
                             "(ATTITUDE, autopilot only)",
                "drift_ppm": round(self.drift_per_s * 1e6, 3),
                "reboot_gap_s": self.reboot_gap_s,
                "samples": self.n, "resets": self.resets,
                "last_reset": self.last_reset}

# Rates we ask for when --mavlink-set-rates is given. ArduSub's defaults are far
# too low for anything time-sensitive. The ROV IMU is NOT the primary VIO IMU here
# (the camera's BNO086 is), so these are for context and cross-checking, and there
# is no reason to push the link hard. The first three are always replaced: by
# default_rate_hz (one rate for all three), or per message by stream_rates_hz —
# which is how rov_gui asks since 2026-10-15 (rov_gui/mavlink_streams.py).
#: The IMU streams whose request decides ``rates_sent_t`` (a send error on any
#: of them = the request did not go out; MavlinkLogger._request_rates).
IMU_STREAMS = ("RAW_IMU", "SCALED_IMU2", "ATTITUDE")

DEFAULT_RATES_HZ = {
    "RAW_IMU": 50,
    "SCALED_IMU2": 50,
    "ATTITUDE": 50,
    "SCALED_PRESSURE2": 10,
    "VFR_HUD": 10,
    "SERVO_OUTPUT_RAW": 10,
    "RC_CHANNELS": 10,
    "MANUAL_CONTROL": 10,
    "BATTERY_STATUS": 2,
    "SYS_STATUS": 2,
}


@dataclass
class MessageStat:
    count: int = 0
    first_t: float | None = None
    last_t: float | None = None
    _recent: list = field(default_factory=list, repr=False)

    def update(self, t: float) -> None:
        self.count += 1
        if self.first_t is None:
            self.first_t = t
        self.last_t = t
        self._recent.append(t)
        if len(self._recent) > 200:
            del self._recent[: len(self._recent) - 200]

    @property
    def hz(self) -> float:
        if len(self._recent) < 2:
            return 0.0
        span = self._recent[-1] - self._recent[0]
        return (len(self._recent) - 1) / span if span > 0 else 0.0


class MavlinkLogger:
    """Background MAVLink reader. Start it, drain records, close it.

    Records are dicts with a `t_host` (time.monotonic, the same clock the camera's
    host timestamps use) plus `msg_type` and the message fields verbatim. Fields
    are kept in their native MAVLink units and NOT converted, so nothing is lost
    to a conversion assumption; the unit for every field is written into the
    dataset metadata instead.
    """

    def __init__(self, connection: str = "udpin:0.0.0.0:14551",
                 transport: str = "udp",
                 rest_url: str = "http://192.168.2.2:6040",
                 source_system: int = 200,
                 set_rates: bool = False,
                 rates_hz: dict | None = None,
                 default_rate_hz: int = 50,
                 verbose: bool = True,
                 stream_rates_hz: dict | None = None):
        self.connection = connection
        self.transport = transport
        self.rest_url = rest_url.rstrip("/")
        self.source_system = source_system
        self.set_rates = set_rates
        self.rates_hz = dict(DEFAULT_RATES_HZ)
        if rates_hz:
            self.rates_hz.update(rates_hz)
        # Messages this logger asks the autopilot to STOP sending (interval -1).
        # Only ever from stream_rates_hz: on the default_rate_hz path a rate
        # <= 0 still means "not requested", as it always has.
        self.stream_off: tuple = ()
        if stream_rates_hz is not None:
            # Per message (rov_gui since 2026-10-15, rov_gui/mavlink_streams.py):
            # Hz, and 0 = off. default_rate_hz is not used.
            self.rates_hz.update(stream_rates_hz)
            self.stream_off = tuple(sorted(
                k for k, hz in stream_rates_hz.items()
                if hz is not None and hz <= 0))
        else:
            # --mavlink-rate raises the time-critical ones together.
            for k in ("RAW_IMU", "SCALED_IMU2", "ATTITUDE"):
                self.rates_hz[k] = default_rate_hz
        # time.monotonic() when _request_rates last transmitted, None = never
        # (rates off, REST, no traffic within start()'s wait, or a send error
        # on one of the IMU streams — rates_failed): the vehicle then keeps
        # whatever it was last asked for, by anyone.
        self.rates_sent_t: float | None = None
        # the names whose SET_MESSAGE_INTERVAL raised on the last request
        # (2026-10-15; a request every send of which failed used to read
        # 'sent' — verify-review F2, RL_controller/logs/eval/
        # station_fix_20261015)
        self.rates_failed: list[str] = []
        self.verbose = verbose

        self.master = None
        self.stats: dict[str, MessageStat] = defaultdict(MessageStat)
        self.connected = False
        self.autopilot_version: str = ""
        self.armed: bool | None = None
        self.mode: str = ""
        self.rate_acks: dict[str, str] = {}

        self._records: list[dict] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._first_msg_at: float | None = None
        # ATTITUDE time_boot_ms -> host clock, fed by the reader thread only
        # (_stamp_vehicle_clock). Every ATTITUDE record leaves with `t_valid`
        # and `vclk_t0` (None when it could not be placed).
        self.vclock = VehicleClockMap()

    # ------------------------------------------------------------------ start
    def start(self, wait_s: float = 6.0) -> bool:
        """Connect and begin reading. Returns True if traffic was seen."""
        if self.transport == "rest":
            self._thread = threading.Thread(target=self._run_rest,
                                            name="mavlink-rest", daemon=True)
            self._thread.start()
        else:
            try:
                from pymavlink import mavutil
            except ImportError:
                print("  MAVLink: pymavlink is not installed "
                      "(pip install pymavlink); telemetry will not be logged")
                return False
            try:
                # source_system is ours, distinct from QGC's 255. We do not send
                # heartbeats, so ArduSub never counts us as a GCS.
                self.master = mavutil.mavlink_connection(
                    self.connection, source_system=self.source_system,
                    source_component=192)   # MAV_COMP_ID_USER1-ish, not a GCS id
            except Exception as e:  # noqa: BLE001
                print(f"  MAVLink: could not open {self.connection}: {e}")
                return False
            self._thread = threading.Thread(target=self._run_udp,
                                            name="mavlink-udp", daemon=True)
            self._thread.start()

        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if self._first_msg_at is not None:
                self.connected = True
                if self.verbose:
                    print(f"  MAVLink: receiving on {self._describe_source()}")
                if self.set_rates:
                    self._request_rates()
                return True
            time.sleep(0.2)

        if self.verbose:
            self._explain_silence(wait_s)
        return False

    def _describe_source(self) -> str:
        return (f"MAVLink2Rest {self.rest_url}" if self.transport == "rest"
                else self.connection)

    def _explain_silence(self, wait_s: float) -> None:
        print(f"  MAVLink: nothing received in {wait_s:g}s on "
              f"{self._describe_source()}")
        if self.transport == "rest":
            print(f"    Check that MAVLink2Rest is up:  "
                  f"curl -s {self.rest_url}/v1/mavlink | head -c 200")
            return
        print("    Most likely BlueOS is not pushing to this port. BlueOS sends")
        print("    MAVLink to 192.168.2.1:14550 by default, and QGroundControl")
        print("    already holds that port — two processes cannot receive the")
        print("    same UDP datagrams. Add a second endpoint:")
        print("        python -m c3_camera.blueos_endpoint add --port 14551")
        print("    or use the no-configuration path instead:")
        print("        --mavlink-transport rest")

    # ------------------------------------------------------------------- read
    def _run_udp(self) -> None:
        while not self._stop.is_set():
            try:
                msg = self.master.recv_match(blocking=True, timeout=0.5)
            except Exception:  # noqa: BLE001 - keep logging through a bad packet
                continue
            if msg is None:
                continue
            mtype = msg.get_type()
            if mtype == "BAD_DATA":
                continue
            t = time.monotonic()
            if self._first_msg_at is None:
                self._first_msg_at = t
            self.stats[mtype].update(t)

            if mtype == "HEARTBEAT":
                self._note_heartbeat(msg)
            elif mtype == "COMMAND_ACK":
                self._note_ack(msg)

            if mtype in ALL_MESSAGES:
                # Who sent it. to_dict() drops the MAVLink header, and on this
                # vehicle the link carries heartbeats from at least three
                # systems (autopilot 1/1, an onboard component 1/194, and the
                # GCS 255/240 — observed 2026-08-06). Without the source, a
                # consumer cannot tell the vehicle's arm state from someone
                # else's flags.
                rec = {"t_host": t, "msg_type": mtype,
                       "_srcsys": msg.get_srcSystem(),
                       "_srccomp": msg.get_srcComponent()}
                rec.update(msg.to_dict())
                rec.pop("mavpackettype", None)
                if mtype == "ATTITUDE":
                    self._stamp_vehicle_clock(rec)
                with self._lock:
                    self._records.append(rec)

    def _stamp_vehicle_clock(self, rec: dict) -> None:
        """ATTITUDE only, in the reader thread, at arrival: ``t_valid`` = when
        the sample was TRUE on the vehicle, on this host's clock (VehicleClockMap),
        and ``vclk_t0`` = the arrival of the first sample of the map in force (its
        age gates the rl_pwm engage). Both None for a record the map cannot place
        — another component's clock, or no usable time_boot_ms — which a consumer
        must read as "age unknown", never as fresh."""
        comp = rec.get("_srccomp")
        if comp is not None and comp != AUTOPILOT_COMPONENT:
            rec["t_valid"] = rec["vclk_t0"] = None
            return
        t_valid = self.vclock.update(rec.get("t_host"), rec.get("time_boot_ms"),
                                     src=(rec.get("_srcsys"), comp))
        rec["t_valid"] = t_valid
        rec["vclk_t0"] = self.vclock.t_start if t_valid is not None else None

    def _note_heartbeat(self, msg) -> None:
        """Arm state and mode, from the AUTOPILOT's heartbeat only.

        Measured on this vehicle 2026-08-06: component 194 also heartbeats, with
        ``base_mode = 0x80`` — which is exactly the SAFETY_ARMED bit. Taking any
        heartbeat made a disarmed vehicle report ARMED, and the mode read
        ``Mode(0x00000080)``. An arm indicator that can be wrong is worse than
        no arm indicator, so this now ignores everything except component 1.
        """
        try:
            from pymavlink import mavutil
            if msg.get_srcComponent() != mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1:
                return
            self.armed = bool(msg.base_mode &
                              mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            self.mode = mavutil.mode_string_v10(msg) or ""
        except Exception:  # noqa: BLE001
            pass

    def _note_ack(self, msg) -> None:
        try:
            from pymavlink import mavutil
            if msg.command == mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL:
                names = {v: k for k, v in vars(mavutil.mavlink).items()
                         if k.startswith("MAV_RESULT_")}
                self.rate_acks["last"] = names.get(msg.result, str(msg.result))
        except Exception:  # noqa: BLE001
            pass

    def _run_rest(self) -> None:
        """Poll MAVLink2Rest. Samples the latest value; does not capture every message."""
        path = f"{self.rest_url}/v1/mavlink/vehicles/1/components/1/messages"
        seen_counters: dict[str, int] = {}
        while not self._stop.is_set():
            try:
                with urllib.request.urlopen(path, timeout=2.0) as r:
                    payload = json.loads(r.read().decode("utf-8", "replace"))
            except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
                time.sleep(0.5)
                continue
            t = time.monotonic()
            if self._first_msg_at is None:
                self._first_msg_at = t
            msgs = payload if isinstance(payload, dict) else {}
            for mtype, entry in msgs.items():
                if mtype not in ALL_MESSAGES:
                    continue
                body = (entry or {}).get("message") or {}
                counter = ((entry or {}).get("status") or {}).get(
                    "time", {}).get("counter")
                # Only record when MAVLink2Rest's own counter advanced, so a
                # stale value is not logged repeatedly as if it were new.
                if counter is not None and seen_counters.get(mtype) == counter:
                    continue
                seen_counters[mtype] = counter
                self.stats[mtype].update(t)
                rec = {"t_host": t, "msg_type": mtype}
                rec.update({k: v for k, v in body.items() if k != "type"})
                if mtype == "ATTITUDE":
                    # t_host is the POLL time here, so the map's floor carries
                    # the best-case poll delay too (same lower-bound caveat)
                    self._stamp_vehicle_clock(rec)
                with self._lock:
                    self._records.append(rec)
                if mtype == "HEARTBEAT":
                    self.armed = None
            time.sleep(0.02)

    # ------------------------------------------------------------------ rates
    def _request_rates(self) -> None:
        """Raise message rates via MAV_CMD_SET_MESSAGE_INTERVAL.

        param1 = message id, param2 = interval in MICROseconds (0 = default,
        -1 = disable). This is the only thing this module ever transmits.
        A name in ``stream_off`` is sent -1 (stop); any other rate <= 0 is not
        requested at all.

        ``rates_sent_t`` is stamped only when at least one request went out and
        none of the IMU streams' (``IMU_STREAMS``) raised; the names whose send
        raised are kept in ``rates_failed`` either way.
        """
        if self.master is None:
            print("  MAVLink: cannot set rates over the REST transport "
                  "(it is read-only); leaving vehicle rates untouched")
            return
        from pymavlink import mavutil

        target_sys = self.master.target_system or 1
        target_comp = self.master.target_component or 1
        if self.verbose:
            print(f"  MAVLink: requesting rates from system {target_sys} "
                  f"(this transmits to the autopilot)")
        sent: list[str] = []
        failed: list[str] = []
        for name, hz in sorted(self.rates_hz.items()):
            msg_id = getattr(mavutil.mavlink, f"MAVLINK_MSG_ID_{name}", None)
            if msg_id is None:
                continue
            if name in self.stream_off:
                interval_us = -1
            elif hz is None or hz <= 0:
                continue
            else:
                interval_us = int(1_000_000 / hz)
            try:
                self.master.mav.command_long_send(
                    target_sys, target_comp,
                    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                    msg_id, interval_us, 0, 0, 0, 0, 0)
                sent.append(name)
            except Exception as e:  # noqa: BLE001
                failed.append(name)
                print(f"    could not request {name}: {e}")
        self.rates_failed = failed
        imu_failed = [k for k in failed if k in IMU_STREAMS]
        if sent and not imu_failed:
            self.rates_sent_t = time.monotonic()
        else:
            why = (f"send error on {', '.join(failed)}" if failed
                   else "nothing to request")
            print(f"  MAVLink: rate request NOT sent ({why}): the vehicle "
                  f"keeps the rates it was last asked for, by anyone")
            return
        if self.verbose:
            print(f"    requested: " + ", ".join(
                f"{k} off" if k in self.stream_off else f"{k}@{v}Hz"
                for k, v in sorted(self.rates_hz.items())
                if k not in failed))
            if failed:
                print(f"    NOT sent (send error): {', '.join(failed)}")
            print("    achieved rates are printed in the live stats and written "
                  "to metadata, so you can verify rather than assume")

    # ----------------------------------------------------------------- drain
    def drain(self) -> list[dict]:
        with self._lock:
            out = self._records
            self._records = []
        return out

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self.master is not None:
            try:
                self.master.close()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------- HUD
    def hud_lines(self, top: int = 4) -> list[str]:
        if not self.stats:
            return ["mav    (no telemetry)"]
        interesting = [m for m in ("RAW_IMU", "SCALED_IMU2", "ATTITUDE",
                                   "SCALED_PRESSURE2", "RC_CHANNELS", "HEARTBEAT")
                       if m in self.stats]
        parts = [f"{m.split('_')[0].lower()[:6]} {self.stats[m].hz:5.1f}"
                 for m in interesting[:top]]
        state = ""
        if self.armed is not None:
            state = f"  {'ARMED' if self.armed else 'disarmed'}"
            if self.mode:
                state += f"/{self.mode}"
        total = sum(s.count for s in self.stats.values())
        return [f"mav    {' '.join(parts)} Hz  {total} msgs{state}"]

    def summary(self) -> dict:
        return {
            "transport": self.transport,
            "source": self._describe_source(),
            "connected": self.connected,
            "armed_last_seen": self.armed,
            "mode_last_seen": self.mode,
            "set_rates_requested": self.set_rates,
            "requested_rates_hz": self.rates_hz if self.set_rates else None,
            # asked to STOP (interval -1); their requested_rates_hz entry is 0
            "requested_off": list(self.stream_off) if self.set_rates else None,
            # did the request actually go out (see rates_sent_t)
            "rates_sent": self.rates_sent_t is not None,
            # names whose SET_MESSAGE_INTERVAL raised on the last request
            "rates_failed": list(self.rates_failed),
            "set_message_interval_ack": self.rate_acks.get("last"),
            "achieved_rates_hz": {m: round(s.hz, 2)
                                  for m, s in sorted(self.stats.items())},
            "message_counts": {m: s.count for m, s in sorted(self.stats.items())},
            "vehicle_clock": self.vclock.summary(),
        }
