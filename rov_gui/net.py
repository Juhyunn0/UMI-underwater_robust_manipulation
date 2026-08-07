#!/usr/bin/env python3
"""
net.py — tether / fiber link health, measured rather than assumed.

Everything here reads the *topside* end of the link from the kernel, which is
free, needs no privileges, no extra dependency (no psutil), and — unlike a ping
that only says up/down — distinguishes the three failure modes that matter:

    the converter lost link        carrier == 0, operstate == "down"
    the link renegotiated slower   speed_mbps drops (1000 -> 100 is a classic
                                   symptom of a marginal connector or a
                                   partially failed pair)
    the link is up but dirty       rx_errors / rx_crc_errors / rx_dropped climb
                                   while carrier stays 1

The third one is why the counters are here at all: a fiber tether that is
losing frames shows perfect "connected" indicators everywhere else in the
system, and the only early warning is the error rate rising before the video
starts stuttering.

Throughput is byte-counter deltas over wall time, so it is the interface's real
traffic — the sum of video, MAVLink, and anything else sharing the tether — not
a per-stream estimate. Compare it against the per-stream numbers the camera
reports (``c3_camera.metrics``) to see who is spending the budget.

On this desktop (2026-08-06) the tether NIC is the USB-Ethernet adapter holding
192.168.2.1/24; :func:`iface_for_subnet` finds it by address rather than by
name, because the name (``enx0000bad00249``) is derived from that adapter's MAC
and changes if the adapter does.
"""

from __future__ import annotations

import fcntl
import os
import socket
import struct
import threading
import time
from pathlib import Path

from .state import Conn, LinkStat

SYS_NET = Path("/sys/class/net")
SIOCGIFADDR = 0x8915

# The vehicle side of a stock BlueROV2 tether. 192.168.2.2 is BlueOS on the Pi;
# port 80 is its web UI, which is the cheapest thing on the vehicle that answers
# a TCP connect without us having to speak any protocol.
DEFAULT_PEER = ("192.168.2.2", 80)


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _read_int(path: Path) -> int | None:
    text = _read(path)
    try:
        return int(text)
    except ValueError:
        return None


def interfaces() -> list[str]:
    if not SYS_NET.is_dir():
        return []
    return sorted(p.name for p in SYS_NET.iterdir() if p.name != "lo")


def ipv4_of(iface: str) -> str | None:
    """The interface's IPv4 address, or None. Linux-only, no root needed."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        packed = fcntl.ioctl(s.fileno(), SIOCGIFADDR,
                             struct.pack("256s", iface[:15].encode()))
        return socket.inet_ntoa(packed[20:24])
    except OSError:
        return None
    finally:
        s.close()


def iface_for_subnet(prefix: str = "192.168.2.") -> str | None:
    """Which NIC currently holds an address on the tether subnet."""
    for name in interfaces():
        addr = ipv4_of(name)
        if addr and addr.startswith(prefix):
            return name
    return None


def tcp_probe(host: str, port: int, timeout_s: float = 1.0) -> float | None:
    """Round-trip time of a TCP connect, in ms, or None if it did not connect.

    A TCP connect rather than ICMP because ping needs either root or a
    cap_net_raw binary, and a connect exercises the same path. It costs one
    SYN/SYN-ACK; do not run it faster than about 1 Hz.
    """
    t0 = time.monotonic()
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return (time.monotonic() - t0) * 1000.0
    except OSError:
        return None


class NicMonitor:
    """Samples one interface's counters and turns deltas into rates.

    Stateful on purpose: rates need two samples. The first :meth:`sample` after
    construction has no previous reading, so its rate fields are None rather
    than 0.0 — a zero there would be indistinguishable from a genuinely idle
    link, and on a control station "no traffic" and "I do not know yet" must not
    look the same.
    """

    _COUNTERS = ("rx_bytes", "tx_bytes", "rx_errors", "tx_errors",
                 "rx_dropped", "rx_crc_errors")

    def __init__(self, iface: str | None = None,
                 peer: tuple[str, int] = DEFAULT_PEER,
                 probe_interval_s: float = 2.0,
                 subnet: str = "192.168.2."):
        self.subnet = subnet
        self.iface = iface or iface_for_subnet(subnet)
        self.peer = peer
        self.probe_interval_s = probe_interval_s
        self._prev: dict[str, int] = {}
        self._prev_t: float | None = None
        self._last_probe_t = 0.0
        self._last_rtt: float | None = None
        self._probe_lock = threading.Lock()
        self._probing = False
        self._rtt_t: float | None = None      # when _last_rtt was measured

    # ------------------------------------------------------------------ read
    def _counters(self, iface: str) -> dict[str, int]:
        base = SYS_NET / iface / "statistics"
        out = {}
        for name in self._COUNTERS:
            v = _read_int(base / name)
            if v is not None:
                out[name] = v
        return out

    # ----------------------------------------------------------------- probe
    def _start_probe(self, t: float) -> None:
        """Kick off the RTT probe on its OWN thread and return immediately.

        This used to be a plain ``tcp_probe()`` call inside :meth:`sample`, and
        that was a latency bug with teeth. ``sample`` is called from the worker
        that also owns the command slots (ARM, DISARM, E-STOP, gripper), and a
        TCP connect to a host that is not answering blocks for its full timeout.
        So the moment the vehicle became unreachable — precisely when the pilot
        is most likely to be reaching for DISARM — every queued command sat
        behind a blocking socket. Measured on this desktop with the ROV
        unplugged: queued-command latency jumped from 20 ms to 500-620 ms
        (rov_gui/tests/test_offline.py::test_link_probe_never_blocks_the_caller).

        One probe in flight at a time. If the previous one has not come back,
        skip this round rather than pile threads up against a dead peer.
        """
        with self._probe_lock:
            if self._probing:
                return
            self._probing = True
        self._last_probe_t = t

        def _run() -> None:
            rtt = tcp_probe(*self.peer, timeout_s=1.0)
            with self._probe_lock:
                self._last_rtt = rtt
                self._rtt_t = time.monotonic()
                self._probing = False

        threading.Thread(target=_run, name="link-probe", daemon=True).start()

    def sample(self, probe: bool = True) -> LinkStat:
        """One reading. Never blocks: the RTT probe runs on its own thread."""
        # Re-detect if the adapter was unplugged and came back with a new name.
        if not self.iface or not (SYS_NET / self.iface).is_dir():
            self.iface = iface_for_subnet(self.subnet)
        if not self.iface:
            return LinkStat(iface="", conn=Conn.OFFLINE, peer=f"{self.peer[0]}",
                            note=f"no interface on {self.subnet}0/24")

        base = SYS_NET / self.iface
        carrier = _read_int(base / "carrier")
        operstate = _read(base / "operstate")
        speed = _read_int(base / "speed")
        if speed is not None and speed < 0:
            speed = None                       # -1 = link down, not "0 Mbit/s"

        counters = self._counters(self.iface)
        t = time.monotonic()
        rates: dict[str, float] = {}
        if self._prev and self._prev_t is not None:
            dt = t - self._prev_t
            if dt > 0:
                for k, v in counters.items():
                    prev = self._prev.get(k)
                    # A counter that went backwards means the interface was
                    # reset; drop the interval rather than emit a negative rate.
                    if prev is not None and v >= prev:
                        rates[k] = (v - prev) / dt
        self._prev, self._prev_t = counters, t

        if probe and (t - self._last_probe_t) >= self.probe_interval_s:
            self._start_probe(t)
        with self._probe_lock:
            rtt, probed = self._last_rtt, self._rtt_t is not None

        up = bool(carrier) and operstate in ("up", "unknown")
        stat = LinkStat(
            iface=self.iface,
            up=up,
            speed_mbps=speed,
            rx_mbps=(rates["rx_bytes"] * 8 / 1e6) if "rx_bytes" in rates else None,
            tx_mbps=(rates["tx_bytes"] * 8 / 1e6) if "tx_bytes" in rates else None,
            rx_err_per_s=rates.get("rx_errors"),
            tx_err_per_s=rates.get("tx_errors"),
            rx_drop_per_s=rates.get("rx_dropped"),
            rtt_ms=rtt,
            peer=f"{self.peer[0]}:{self.peer[1]}",
        )
        if not probed:
            # The first probe has not come back yet. "No answer" and "have not
            # asked yet" are different facts and the panel must not show the
            # first as the second — a red tether light during the second after
            # launch would train the pilot to ignore a red tether light.
            stat.conn, stat.note = Conn.CONNECTING, f"probing {stat.peer}"
        else:
            stat.conn, stat.note = self._verdict(stat)
        return stat

    # --------------------------------------------------------------- verdict
    @staticmethod
    def _verdict(s: LinkStat) -> tuple[Conn, str]:
        """Turn the readings into one state plus the reason for it.

        The reason string is not decoration: "DEGRADED" on its own sends the
        pilot hunting, "DEGRADED — 12.0 rx err/s" sends them to the connector.
        """
        if not s.up:
            return Conn.FAULT, "media converter link down (carrier 0)"
        errs = (s.rx_err_per_s or 0.0) + (s.tx_err_per_s or 0.0)
        if errs > 1.0:
            return Conn.DEGRADED, f"{errs:.1f} link err/s"
        if (s.rx_drop_per_s or 0.0) > 10.0:
            return Conn.DEGRADED, f"{s.rx_drop_per_s:.0f} rx drop/s"
        if s.speed_mbps is not None and s.speed_mbps < 1000:
            return Conn.DEGRADED, f"negotiated {s.speed_mbps} Mbit/s"
        if s.rtt_ms is None:
            # The NIC is fine; something further down the tether is not.
            return Conn.DEGRADED, f"no answer from {s.peer}"
        return Conn.ONLINE, ""


def describe_host() -> str:
    """One line for the log: which interface we picked and what it is."""
    iface = iface_for_subnet()
    if not iface:
        return "tether: no interface on 192.168.2.0/24 (demo/offline)"
    return (f"tether: {iface} {ipv4_of(iface)} "
            f"speed={_read(SYS_NET / iface / 'speed')} "
            f"carrier={_read(SYS_NET / iface / 'carrier')}")


if __name__ == "__main__":                       # quick manual check
    print(describe_host())
    mon = NicMonitor()
    for _ in range(3):
        time.sleep(1.0)
        print(mon.sample())
    os._exit(0)
