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

ALL_MESSAGES = tuple(dict.fromkeys(IMU_MESSAGES + TELEM_MESSAGES + CONTROL_MESSAGES))

# Rates we ask for when --mavlink-set-rates is given. ArduSub's defaults are far
# too low for anything time-sensitive. The ROV IMU is NOT the primary VIO IMU here
# (the camera's BNO086 is), so these are for context and cross-checking, and there
# is no reason to push the link hard.
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
                 verbose: bool = True):
        self.connection = connection
        self.transport = transport
        self.rest_url = rest_url.rstrip("/")
        self.source_system = source_system
        self.set_rates = set_rates
        self.rates_hz = dict(DEFAULT_RATES_HZ)
        if rates_hz:
            self.rates_hz.update(rates_hz)
        # --mavlink-rate raises the time-critical ones together.
        for k in ("RAW_IMU", "SCALED_IMU2", "ATTITUDE"):
            self.rates_hz[k] = default_rate_hz
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
                with self._lock:
                    self._records.append(rec)

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
        for name, hz in sorted(self.rates_hz.items()):
            msg_id = getattr(mavutil.mavlink, f"MAVLINK_MSG_ID_{name}", None)
            if msg_id is None or hz <= 0:
                continue
            interval_us = int(1_000_000 / hz)
            try:
                self.master.mav.command_long_send(
                    target_sys, target_comp,
                    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                    msg_id, interval_us, 0, 0, 0, 0, 0)
            except Exception as e:  # noqa: BLE001
                print(f"    could not request {name}: {e}")
        if self.verbose:
            print(f"    requested: " + ", ".join(
                f"{k}@{v}Hz" for k, v in sorted(self.rates_hz.items())))
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
            "set_message_interval_ack": self.rate_acks.get("last"),
            "achieved_rates_hz": {m: round(s.hz, 2)
                                  for m, s in sorted(self.stats.items())},
            "message_counts": {m: s.count for m, s in sorted(self.stats.items())},
        }
