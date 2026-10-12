#!/usr/bin/env python3
"""dvl.py — the Water Linked DVL's TCP stream, onto the bus.

HARDWARE-UNVERIFIED (2026-10-01): written against the vendor protocol page
and a 12 s capture of the vendor's public DEMO unit
(rov_gui/tests/fixtures/dvl_a50_demo_capture_20261001.jsonl), before a DVL was
mounted on this vehicle. The demo unit showed the framing (one JSON object
per line, ``\\r\\n``), accepted two TCP clients at once — so the BlueOS DVL
extension and this station can PROBABLY both read the real one [예측] — and
answered a command only on the socket that sent it (the observations are
written down beside the capture, tests/fixtures/dvl_a50_demo_capture_
20261001.README.md). It says nothing about this vehicle's network, mount or
pool, and a demo unit is not the product.

Two pieces, the VehicleWorker shape (backends/hardware.py):

``DvlClient`` — a plain reader THREAD. It blocks on the socket, so the
arrival stamp is taken the moment the bytes are there, and it reconnects on
its own. No Qt, so it is tested against a local fake server.

``DvlWorker`` — a TimerWorker that drains the client, turns velocity reports
into ``state.DvlSample`` on ``bus.dvl``, keeps the SENSORS row honest and
writes the verbatim stream beside a controller run. A TimerWorker and not a
LoopWorker because it has to RECEIVE (the raw-log request is a queued slot,
backends/base.py rule).

This module only ever READS the device. The one thing it sends is
``get_config`` — itself a read: once per connection and then every
``dvl.config_poll_s``, because the device rotates and scales its own output
by settings anyone can change in its web GUI, and a proof made before such a
change says nothing about the stream after it. It never sets a
configuration, never resets the device's dead reckoning and never triggers a
ping — those change the instrument for every other client.
"""

from __future__ import annotations

import socket
import threading
import time
from collections import deque
from pathlib import Path

from ..control import dvl as DV
from ..qt import Slot
from ..sensorlog import SensorLog
from ..state import Conn, DvlSample, SensorStat, now
from .base import TimerWorker

#: A line longer than this with no newline is not a report (the longest in
#: the demo capture is 912 bytes); the buffer is dropped rather than grown.
MAX_LINE = 65536
#: No bytes for this long = a dead link, whatever TCP thinks: reconnect.
SILENCE_S = 5.0


def dvl_config(opts) -> dict | None:
    """The resolved ``dvl:`` block for these options, or None when no DVL is
    wanted (or the block does not parse — MpcWorker reports that one, in
    words; a backend that died on it would take the cameras down too)."""
    try:
        from ..control.geometry import MpcConfig

        raw = MpcConfig.load(getattr(opts, "mpc_config",
                                     "config/hw_mpc.yaml")).dvl
        cfg = DV.resolve_with_opts(raw, opts)
    except Exception:                                            # noqa: BLE001
        return None
    return cfg if cfg["enabled"] else None


class DvlClient:
    """Reader thread for the DVL's newline-delimited JSON over TCP.

    ``drain()`` returns ``[(t_rx, report_dict, raw_line, conn_id), ...]``
    oldest first, where ``t_rx`` is ``clock()`` when the chunk holding the
    line was received and ``conn_id`` counts connections (a consumer that
    keeps per-connection state — the device-clock map — starts over when it
    changes). The queue is bounded: a consumer that stalls loses the OLDEST
    reports and ``dropped`` counts them (bus.py's rule — a stale velocity is
    worth less than a current one).

    The reader thread survives anything a line can contain: a report that
    cannot be parsed is counted and skipped, and an unexpected exception
    ends the CONNECTION, not the thread.
    """

    def __init__(self, host: str, port: int, connect_timeout_s: float = 2.0,
                 reconnect_s: float = 2.0, clock=time.monotonic,
                 ask_config: bool = True, config_poll_s: float = 0.0):
        self.host, self.port = str(host), int(port)
        self.connect_timeout_s = float(connect_timeout_s)
        self.reconnect_s = float(reconnect_s)
        self.clock = clock
        self.ask_config = bool(ask_config)
        self.config_poll_s = float(config_poll_s)
        self._t_ask = 0.0
        self._q: deque = deque(maxlen=512)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.state = Conn.OFFLINE
        self.error = ""
        self.n_lines = 0
        self.n_bad = 0
        self.n_connects = 0
        self.dropped = 0
        self.t_last_rx: float | None = None

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="dvl-reader",
                                        daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 3.0) -> None:
        self._stop.set()
        t, self._thread = self._thread, None
        if t is not None:
            t.join(timeout_s)
        self.state = Conn.OFFLINE

    def drain(self) -> list:
        with self._lock:
            out = list(self._q)
            self._q.clear()
        return out

    # --------------------------------------------------------------- reader
    def _push(self, item) -> None:
        with self._lock:
            if len(self._q) == self._q.maxlen:
                self.dropped += 1
            self._q.append(item)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.state = Conn.CONNECTING
            sock = None
            try:
                sock = socket.create_connection(
                    (self.host, self.port), timeout=self.connect_timeout_s)
                sock.settimeout(0.5)
                self.n_connects += 1
                self.error = ""
                if self.ask_config:
                    # The ONE thing this client ever sends (a read). The
                    # answer comes back as a `response` line on this socket.
                    sock.sendall(self._ASK)
                    self._t_ask = self.clock()
                self._read(sock)
            except Exception as e:                               # noqa: BLE001
                # Not only OSError: whatever went wrong, the thread's job is
                # to come back. A reader that died once would leave the
                # station showing its last state forever.
                self.error = f"{type(e).__name__}: {e}"
            finally:
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
            self.state = Conn.OFFLINE
            self._stop.wait(self.reconnect_s)

    _ASK = b'{"command":"get_config"}\n'

    def _read(self, sock) -> None:
        buf = b""
        t_last = self.clock()
        conn_id = self.n_connects
        while not self._stop.is_set():
            if self.ask_config and self.config_poll_s > 0.0 \
                    and self.clock() - self._t_ask >= self.config_poll_s:
                sock.sendall(self._ASK)
                self._t_ask = self.clock()
            try:
                chunk = sock.recv(8192)
            except socket.timeout:
                if self.clock() - t_last > SILENCE_S:
                    self.error = f"silent for {SILENCE_S:.0f} s"
                    return
                continue
            if not chunk:
                self.error = "closed by the DVL"
                return
            t_rx = self.clock()
            t_last = t_rx
            self.t_last_rx = t_rx
            self.state = Conn.ONLINE
            buf += chunk
            while True:
                i = buf.find(b"\n")
                if i < 0:
                    break
                line, buf = buf[:i], buf[i + 1:]
                self.n_lines += 1
                rep = DV.parse_line(line)
                if rep is None:
                    if line.strip():
                        self.n_bad += 1
                    continue
                self._push((t_rx, rep, line, conn_id))
            if len(buf) > MAX_LINE:
                buf = b""
                self.n_bad += 1


class DvlWorker(TimerWorker):
    """Drains a :class:`DvlClient` at 50 Hz and publishes what it said."""

    def __init__(self, bus, opts, client: DvlClient | None = None):
        super().__init__("dvl", interval_ms=20)
        self.bus = bus
        self.opts = opts
        self.client = client
        self.cfg: dict | None = None
        self._log: SensorLog | None = None
        self._t_stat = 0.0
        self._arrivals: deque = deque(maxlen=64)   # t_rx of velocity reports
        self._last: DvlSample | None = None
        self._was_online = False
        self._clock = DV.DeviceClock()             # device stamps -> station
        self._conn_id = None                       # ...per connection
        self._cfg_seen: dict | None = None         # last config answered
        self.n_bad_reports = 0

    def setup(self) -> None:
        if self.client is None:
            from ..control.geometry import MpcConfig

            raw = MpcConfig.load(getattr(self.opts, "mpc_config",
                                         "config/hw_mpc.yaml")).dvl
            self.cfg = DV.resolve_with_opts(raw, self.opts)
            self.client = DvlClient(self.cfg["host"], self.cfg["port"],
                                    self.cfg["connect_timeout_s"],
                                    self.cfg["reconnect_s"], clock=now,
                                    config_poll_s=self.cfg["config_poll_s"])
        self.client.start()
        self.bus.log.emit(
            "info", f"dvl: connecting to {self.client.host}:"
                    f"{self.client.port} (read-only; get_config per "
                    f"connection and every "
                    f"{self.client.config_poll_s:.0f} s)")

    def _hz(self, t: float) -> float | None:
        """Measured ARRIVAL rate over the last 2 s — a count, so a link that
        stopped reads as falling toward 0 rather than holding its last EMA."""
        while self._arrivals and t - self._arrivals[0] > 2.0:
            self._arrivals.popleft()
        n = len(self._arrivals)
        return None if n < 2 else (n - 1) / max(1e-3,
                                                self._arrivals[-1]
                                                - self._arrivals[0])

    def tick(self) -> None:
        c = self.client
        if c is None:
            return
        for t_rx, rep, raw, conn_id in c.drain():
            try:
                self._handle(t_rx, rep, conn_id)
            except Exception as e:                               # noqa: BLE001
                # ONE report is lost, not the rest of the drain: the reports
                # behind it in this batch are just as real.
                self.n_bad_reports += 1
                if self.n_bad_reports == 1:
                    self.bus.log.emit(
                        "warn", f"dvl: a report could not be handled "
                                f"({type(e).__name__}: {e}) — skipped; "
                                f"further ones are only counted")
        t = now()
        online = c.state is Conn.ONLINE
        if online != self._was_online:
            self._was_online = online
            self.bus.log.emit(
                "info" if online else "warn",
                "dvl: link up" if online else
                f"dvl: link down ({c.error or 'not connected'}) — retrying "
                f"every {c.reconnect_s:.0f} s")
        if t - self._t_stat >= 0.5:
            self._t_stat = t
            self._publish_stat(t)

    def _handle(self, t_rx: float, rep: dict, conn_id) -> None:
        kind = str(rep.get("type", ""))
        if self._log is not None:
            # VERBATIM, plus the station's arrival stamp: the record a
            # better estimator can be re-run over afterwards.
            self._log.write(kind or "unknown", rep, t=t_rx)
        if conn_id != self._conn_id:
            # A new connection may be a rebooted device with a new clock.
            self._conn_id = conn_id
            self._clock = DV.DeviceClock()
        if kind == "response":
            if rep.get("response_to") == "get_config" \
                    and rep.get("success") is True \
                    and isinstance(rep.get("result"), dict):
                info = dict(rep["result"])
                info["format"] = rep.get("format")
                self.bus.dvl_info.emit(info)
                if info != self._cfg_seen:
                    # Said when it is news, not every poll.
                    self.bus.log.emit(
                        "info" if self._cfg_seen is None else "warn",
                        f"dvl: device config {info}"
                        + ("" if self._cfg_seen is None else " (CHANGED)"))
                    self._cfg_seen = info
            return
        f = DV.sample_fields(rep, t_rx, clock=self._clock)
        if f is None:
            return                        # position_local etc.: logged only
        self._arrivals.append(t_rx)
        s = DvlSample(hz=self._hz(t_rx), source="a50",
                      conn=(Conn.ONLINE if f["valid"] else Conn.DEGRADED),
                      stamp=t_rx, **f)
        self._last = s
        self.bus.dvl.emit(s)

    def _publish_stat(self, t: float) -> None:
        c, s = self.client, self._last
        hz = self._hz(t)
        if c.state is not Conn.ONLINE or s is None or t - s.t_rx > 2.0:
            self.bus.sensor_stat.emit(SensorStat(
                "DVL", None,
                Conn.CONNECTING if c.state is Conn.CONNECTING else Conn.OFFLINE,
                f"no link ({c.error})" if c.error else "no reports"))
            return
        if not s.valid:
            self.bus.sensor_stat.emit(SensorStat(
                "DVL", hz, Conn.DEGRADED, "no bottom lock"))
            return
        detail = (f"{hz:.0f} Hz alt {s.altitude:.2f} m"
                  if hz is not None and s.altitude is not None else "")
        if s.status & 1:
            self.bus.sensor_stat.emit(SensorStat(
                "DVL", hz, Conn.DEGRADED, "HIGH TEMPERATURE — keep submerged"))
            return
        self.bus.sensor_stat.emit(SensorStat("DVL", hz, Conn.ONLINE, detail))

    @Slot(bool, str)
    def set_raw_log(self, on: bool, stem: str) -> None:
        """``<stem>_dvl.jsonl`` — every report of the run, verbatim."""
        if on and self._log is None and stem:
            self._log = SensorLog(Path(f"{stem}_dvl.jsonl"), "dvl")
            self.bus.log.emit("info", f"dvl raw log -> {self._log.path}")
        elif not on and self._log is not None:
            meta = self._log.close()
            self._log = None
            self.bus.log.emit("info", f"dvl raw log closed: "
                                      f"{meta['records']} records")

    def teardown(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None
        if self.client is not None:
            self.client.stop()
