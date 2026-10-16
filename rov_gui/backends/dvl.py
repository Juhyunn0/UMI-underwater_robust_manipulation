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

``DvlTempPoller`` (2026-10-13) — a second plain thread for the device's
TEMPERATURE, which is not on the TCP stream at all: that carries only bit 0
of a velocity report's ``status`` ("high temperature, thermal shutdown
soon"). The number is on the device's web API, the page its own GUI shows.
A thread of its own so a slow web server can never hold up a velocity
report's arrival stamp.

This module only ever READS the device. The one thing it sends is
``get_config`` — itself a read: once per connection and then every
``dvl.config_poll_s``, because the device rotates and scales its own output
by settings anyone can change in its web GUI, and a proof made before such a
change says nothing about the stream after it — plus the temperature poller's
HTTP GET. It never sets a configuration, never resets the device's dead
reckoning and never triggers a ping — those change the instrument for every
other client.
"""

from __future__ import annotations

import http.client
import json
import math
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

# ---------------------------------------------------------------- temperature
# Vendor copies cited below live under data/20261011/1011_155235_dvl_probe/
# analysis/wf/vendor_dl/ ("VENDOR_DL").
#: The device's own status page: ``{"cpu_load": .., "disk_free": {..},
#: "temperature": <deg C>}``. The vendor GUI prints that value as
#: "Temperature … C" [스펙: VENDOR_DL/demo_gui/main.js]; WHICH sensor inside
#: the housing it is, the docs do not say. Port 80, GET only.
STATUS_PATH = "/api/v1/about/status"
#: The vendor GUI asks every 3 s [스펙: VENDOR_DL/demo_gui/main.js,
#: refreshInterval], the BlueOS DVL extension every 30 s [스펙: VENDOR_DL/gh/
#: blueos_ext/dvl.py:72]. The A50 here warmed 0.26-0.29 °C/min with acoustics
#: on [측정: data/20261013/1013_194334_dvl_temp/ — 0.26 end to end over 3.5
#: min, 0.29 least squares over 60 s; that it was out of the water then is
#: [유도, 운영자 미확인]] — 5 s misses nothing.
TEMP_POLL_S = 5.0
TEMP_TIMEOUT_S = 1.5
#: The page is 76 bytes [측정: data/20261011/1011_155235_dvl_probe/api/
#: about_status.json]; anything this long is not it.
MAX_STATUS_BYTES = 16384
#: A reading older than this is not shown — three missed polls. A temperature
#: that stopped updating is the frozen number SensorPanel must never show.
TEMP_STALE_S = 20.0
#: Outside this the number is not a temperature in °C (a unit change in a
#: firmware, a broken sensor): "no temperature", never a fake HOT. [예측]
TEMP_SANE_C = (-40.0, 150.0)
#: HOT from here: the BlueOS DVL extension's "DVL is too hot" threshold —
#: it says so ABOVE 45, this station AT 45 [스펙: VENDOR_DL/gh/blueos_ext/
#: dvl.py:73, :427].
TEMP_WARN_C = 45.0
#: ...and back below this before the station calls it cool again (the log
#: and the row both use the band). [예측]
TEMP_COOL_C = 44.0
#: The vendor: a warning (status bit 0) before thermal shutdown at 55 °C, and
#: an automatic restart once it cools below 50 °C [스펙: VENDOR_DL/
#: p_how-to-diagnose/page.txt:237].
TEMP_SHUTDOWN_C = 55.0
TEMP_RESTART_C = 50.0
#: A reading this close to the shutdown counts as "hot when the link went",
#: as bit 0 does. [예측 — when the device raises bit 0 is undocumented]
TEMP_NEAR_SHUTDOWN_C = 54.0
#: Bit 0 is said at once but called cleared only after it has stayed clear
#: this long, so a flag flickering at its threshold is one warning, not a
#: warn/info pair per report. [예측 — its hysteresis is undocumented]
BIT0_CLEAR_S = 10.0
#: A status-page request still in flight after this many timeouts is
#: reported as stuck: the socket timeout bounds each recv, not the request,
#: and a server dripping bytes can hold one open indefinitely.
TEMP_STUCK_X = 3.0


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


class DvlTempPoller:
    """The DVL's temperature from its web API, on a reader THREAD of its own.

    ``GET http://<host>/api/v1/about/status`` every ``period_s``, or at once
    after ``poke()``. ``latest()`` is ``(temp_c, t_rx, error)``: the newest
    SANE reading and ``clock()`` when it arrived (whether it is still fresh
    is the caller's question), and the last failure in words — "" after a
    success, "… stuck N s" while a request has hung longer than ``stuck_s``
    (``TEMP_STUCK_X`` timeouts). ``last_try()`` is ``(t_done, answered)``
    of the newest COMPLETED attempt: ``answered`` is whether ANY response
    came back — a 404 or a page that is not JSON says the device is up as
    surely as a temperature does; only a refused, reset or timed-out request
    says it may be down. ``drain()`` hands over every page received since
    the last call, verbatim, for the raw log.

    A GET is a read, and it is the only thing this ever sends. No Qt, so it
    is tested against a local fake server; whatever a request does — refused,
    timed out, a 404 from an older firmware, a page that is not JSON — is a
    reported state, never an exception out of the thread.
    """

    def __init__(self, host: str, port: int = 80,
                 period_s: float = TEMP_POLL_S,
                 timeout_s: float = TEMP_TIMEOUT_S, clock=time.monotonic):
        self.host, self.port = str(host), int(port)
        self.period_s = float(period_s)
        self.timeout_s = float(timeout_s)
        self.stuck_s = TEMP_STUCK_X * self.timeout_s
        self.clock = clock
        self._q: deque = deque(maxlen=64)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._temp_c: float | None = None
        self._t_temp: float | None = None
        self._error = ""
        self._t_req: float | None = None           # a request in flight since
        self._t_done: float | None = None          # the newest completed one
        self._answered = False                     # ...did anything answer
        self.n_ok = 0
        self.n_fail = 0

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="dvl-temp",
                                        daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 0.5) -> None:
        """Short join on purpose: a request in flight ends at its next
        socket timeout, the thread is a daemon holding nothing, and the
        station's shutdown must not wait on a web page."""
        self._stop.set()
        self._wake.set()
        t, self._thread = self._thread, None
        if t is not None:
            t.join(timeout_s)

    def poke(self) -> None:
        """Poll now rather than at the end of the period (the worker asks
        when the velocity stream drops while the DVL was hot)."""
        self._wake.set()

    def latest(self) -> tuple:
        with self._lock:
            err = self._error
            if self._t_req is not None:
                age = self.clock() - self._t_req
                if age > self.stuck_s:
                    err = f"status page request stuck {age:.0f} s"
            return self._temp_c, self._t_temp, err

    def last_try(self) -> tuple:
        with self._lock:
            return self._t_done, self._answered

    def drain(self) -> list:
        with self._lock:
            out = list(self._q)
            self._q.clear()
        return out

    # --------------------------------------------------------------- reader
    def fetch(self) -> dict:
        """One GET of the status page -> its JSON object. Raises on anything
        else. http.client rather than urllib: urllib would route the request
        through an http_proxy set in the environment, away from the tether.
        ``Connection: close`` frames it the way the urllib probe that read
        the real A50 did (data/20261013/1013_194334_dvl_temp/scripts/)."""
        conn = http.client.HTTPConnection(self.host, self.port,
                                          timeout=self.timeout_s)
        try:
            conn.request("GET", STATUS_PATH,
                         headers={"Accept": "application/json",
                                  "Connection": "close"})
            r = conn.getresponse()
            body = r.read(MAX_STATUS_BYTES + 1)
            if r.status != 200:
                raise RuntimeError(f"HTTP {r.status} for {STATUS_PATH}")
        finally:
            conn.close()
        if len(body) > MAX_STATUS_BYTES:
            raise ValueError(f"status page longer than {MAX_STATUS_BYTES} B")
        page = json.loads(body)
        if not isinstance(page, dict):
            raise ValueError(f"status page is a {type(page).__name__}, "
                             f"not an object")
        return page

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                self._t_req = self.clock()
            try:
                page = self.fetch()
                t_rx = self.clock()
                temp = page.get("temperature")
                if isinstance(temp, bool) or not isinstance(temp, (int, float)) \
                        or not math.isfinite(temp) \
                        or not TEMP_SANE_C[0] <= temp <= TEMP_SANE_C[1]:
                    raise ValueError(f"temperature {temp!r} is not a "
                                     f"temperature in °C")
                with self._lock:
                    self._temp_c, self._t_temp = float(temp), t_rx
                    self._error = ""
                    self._q.append((t_rx, page))
                    self._t_req, self._t_done = None, t_rx
                    self._answered = True
                    self.n_ok += 1
            except Exception as e:                               # noqa: BLE001
                # Not only OSError: the thread's job is to come back next
                # period. The last good reading stays, with its own stamp —
                # the consumer stops showing it when it is too old.
                with self._lock:
                    self._error = f"{type(e).__name__}: {e}"
                    self._t_req, self._t_done = None, self.clock()
                    # OSError = no answer at all (refused, reset, timed out,
                    # unreachable); anything else came back from a server.
                    self._answered = not isinstance(e, OSError)
                    self.n_fail += 1
            self._wake.wait(self.period_s)
            self._wake.clear()


class DvlWorker(TimerWorker):
    """Drains a :class:`DvlClient` at 50 Hz and publishes what it said."""

    def __init__(self, bus, opts, client: DvlClient | None = None,
                 temp: DvlTempPoller | None = None):
        super().__init__("dvl", interval_ms=20)
        self.bus = bus
        self.opts = opts
        self.client = client
        # Built beside the client in setup() when the client is; an injected
        # client comes with whatever poller it is given (None = none).
        self.temp = temp
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
        # The temperature's say in the SENSORS row and the log. Each flag is
        # what was last SAID, so every change is logged once, not per stat.
        self._temp_said: bool | None = None        # "temperature N °C" said
        self._hot = False                          # HOT (45/44 band), row + log
        self._bit0 = False                         # bit 0, held BIT0_CLEAR_S
        self._t_bit0: float | None = None          # bit 0 last seen set
        # The last stat that had fresh reports, and whether the DVL was hot
        # at it — what a later link loss is read against.
        self._t_linked: float | None = None
        self._hot_at_link = False
        self._c_at_link: float | None = None       # fresh reading then, or None
        self._was_linked = False
        self._shutdown_said = False                # ...once per outage
        self._shutdown_withdrawn = False

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
            if self.temp is None and TEMP_POLL_S > 0.0:
                self.temp = DvlTempPoller(self.cfg["host"], clock=now)
        self.client.start()
        if self.temp is not None:
            self.temp.start()
        self.bus.log.emit(
            "info", f"dvl: connecting to {self.client.host}:"
                    f"{self.client.port} (read-only; get_config per "
                    f"connection and every "
                    f"{self.client.config_poll_s:.0f} s"
                    + ("" if self.temp is None else
                       f"; temperature GET http://{self.temp.host}"
                       f"{STATUS_PATH} every {self.temp.period_s:.0f} s")
                    + ")")

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
        if self.temp is not None:
            for t_rx, page in self.temp.drain():
                if self._log is not None:
                    # Beside the reports, so a run that lost its DVL can be
                    # read for whether it was cooking (msg "about_status").
                    self._log.write("about_status", page, t=t_rx)
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

    def _temperature(self, t: float) -> float | None:
        """The DVL's temperature if FRESH (``TEMP_STALE_S``), else None."""
        if self.temp is None:
            return None
        temp_c, t_temp, _err = self.temp.latest()
        if temp_c is None or t_temp is None or t - t_temp > TEMP_STALE_S:
            return None
        return temp_c

    def _say_temperature(self, t: float, temp: float | None, linked: bool,
                         bit0_now: bool) -> None:
        """Each change in what the temperature means goes to the log ONCE,
        and the two states the row shows are kept here: HOT on the 45/44 °C
        band, and bit 0 held until it has been clear for ``BIT0_CLEAR_S``."""
        p = self.temp
        if p is not None:
            have = temp is not None
            # A shown reading that went stale is said even when no request
            # has FAILED — a hung one never does (latest() says "stuck").
            if have != self._temp_said \
                    and (have or p.n_fail > 0 or self._temp_said):
                self._temp_said = have
                if have:
                    self.bus.log.emit(
                        "info", f"dvl: temperature {temp:.1f} °C (HOT at "
                                f"{TEMP_WARN_C:.0f} °C; the vendor shuts the "
                                f"DVL down at {TEMP_SHUTDOWN_C:.0f} °C)")
                else:
                    self.bus.log.emit(
                        "warn", f"dvl: no temperature "
                                f"({p.latest()[2] or 'no fresh reading'}) — "
                                f"retrying every {p.period_s:.0f} s")
        if temp is not None:
            if not self._hot and temp >= TEMP_WARN_C:
                self._hot = True
                host = self.temp.host if self.temp is not None else "?"
                self.bus.log.emit(
                    "warn", f"dvl: HOT {temp:.1f} °C — out of the water? "
                            f"Submerge it, or turn its acoustics OFF (web GUI "
                            f"http://{host}); the vendor shuts it down at "
                            f"{TEMP_SHUTDOWN_C:.0f} °C")
            elif self._hot and temp < TEMP_COOL_C:
                self._hot = False
                self.bus.log.emit(
                    "info", f"dvl: temperature back to {temp:.1f} °C")
        if linked:
            if bit0_now:
                self._t_bit0 = t
                if not self._bit0:
                    self._bit0 = True
                    self.bus.log.emit(
                        "warn", "dvl: the DVL itself reports HIGH TEMPERATURE "
                                "(status bit 0) — its thermal shutdown comes "
                                "next")
            elif self._bit0 and self._t_bit0 is not None \
                    and t - self._t_bit0 >= BIT0_CLEAR_S:
                self._bit0 = False
                self.bus.log.emit(
                    "info", "dvl: the DVL's high-temperature flag cleared")

    def _shutdown_likely(self) -> bool:
        """A LIKELY thermal shutdown: the velocity stream is gone, the DVL
        was hot at the last moment it was there (its own bit 0, or a reading
        at ``TEMP_NEAR_SHUTDOWN_C``), AND the newest status-page request
        COMPLETED since then got no answer at all — the shutdown takes the
        whole device down (the vendor says it restarts). Any answer, even a
        404, says the device is up, and the call is withdrawn; another
        silence makes it again. The vendor GUI makes the same call from its own
        temperature_high warning when its web connection drops
        [스펙: VENDOR_DL/demo_gui/main.js, setConnectionClosed]; the rule
        here is the station's."""
        if not self._hot_at_link or self._t_linked is None \
                or self.temp is None:
            return False
        t_done, answered = self.temp.last_try()
        return t_done is not None and t_done > self._t_linked \
            and not answered

    def _publish_stat(self, t: float) -> None:
        c, s = self.client, self._last
        hz = self._hz(t)
        temp = self._temperature(t)
        linked = (c.state is Conn.ONLINE and s is not None
                  and t - s.t_rx <= 2.0)
        self._say_temperature(t, temp, linked, linked and bool(s.status & 1))
        deg = "" if temp is None else f" {temp:.1f}°C"
        hot = temp is not None and self._hot
        if linked:
            self._t_linked = t
            self._hot_at_link = self._bit0 or (
                temp is not None and temp >= TEMP_NEAR_SHUTDOWN_C)
            self._c_at_link = temp
        elif self._was_linked and self._hot_at_link and self.temp is not None:
            self.temp.poke()                  # the verdict in one request
        self._was_linked = linked
        if not linked:
            if self._shutdown_likely():
                if not self._shutdown_said:
                    self._shutdown_said = True
                    why = " + ".join(
                        ([] if not self._bit0 else ["its bit 0"])
                        + ([] if self._c_at_link is None else
                           [f"{self._c_at_link:.1f} °C"]))
                    self.bus.log.emit(
                        "error", f"dvl: velocity stream lost while hot ({why})"
                                 f" and its web page stopped answering "
                                 f"({self.temp.latest()[2]}) — LIKELY THERMAL "
                                 f"SHUTDOWN. The vendor restarts it below "
                                 f"{TEMP_RESTART_C:.0f} °C; out of the water "
                                 f"it needs its acoustics OFF, or water")
                # No number: the last one is minutes old by the time the
                # device is back — the log line above has it, with its words.
                self.bus.sensor_stat.emit(SensorStat(
                    "DVL", None, Conn.FAULT, "THERMAL SHUTDOWN?"))
                return
            if self._shutdown_said and not self._shutdown_withdrawn:
                self._shutdown_withdrawn = True
                self.bus.log.emit(
                    "info", "dvl: its web page answers again — not a thermal "
                            "shutdown (the velocity stream is still gone)")
            if temp is None:
                detail = f"no link ({c.error})" if c.error else "no reports"
            else:
                # Bounded: the error text is already in the "link down" log
                # line, and a row this long squeezes PROPULSION beside it.
                detail = (("no link" if c.error else "no reports")
                          + (" HOT" if hot else "") + deg)
            self.bus.sensor_stat.emit(SensorStat(
                "DVL", None,
                Conn.DEGRADED if hot else
                Conn.CONNECTING if c.state is Conn.CONNECTING else Conn.OFFLINE,
                detail))
            return
        self._shutdown_said = self._shutdown_withdrawn = False
        # The temperature comes BEFORE the lock: out of the water — the only
        # place this DVL overheats — there is never a lock, and until
        # 2026-10-13 bit 0 was checked behind "no bottom lock", a warning
        # nobody could ever see.
        rate = "" if hz is None else f"{hz:.0f} Hz "
        if self._bit0 or hot:
            self.bus.sensor_stat.emit(SensorStat(
                "DVL", hz, Conn.DEGRADED,
                f"{rate}{'HIGH TEMP' if self._bit0 else 'HOT'}{deg}"))
            return
        if not s.valid:
            self.bus.sensor_stat.emit(SensorStat(
                "DVL", hz, Conn.DEGRADED, f"{rate}no lock{deg}"))
            return
        # "N Hz" leads whenever there is a rate, so SensorRow (which shows a
        # detail only if it carries "Hz") never drops the °C behind it.
        alt = "" if s.altitude is None else f"alt {s.altitude:.2f} m"
        detail = " ".join(x for x in (rate.strip(), alt, deg.strip()) if x)
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
        if self.temp is not None:
            self.temp.stop()
        if self._log is not None:
            self._log.close()
            self._log = None
        if self.client is not None:
            self.client.stop()
