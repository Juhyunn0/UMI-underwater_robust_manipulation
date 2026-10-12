"""FoundationStereo in its OWN process — ``--fstereo-process`` (2026-10-12).

Why this exists
---------------
By default the learned-depth network runs inside the station, beside the
diffusion policy, on one GPU, in one CUDA context and one Python interpreter.
Beside a back-to-back FoundationStereo the policy forward went from 19.4 ms
alone to 121.6 ms p50 on the station's real workers; with the network in its
own process it was 40.8 ms, and with ``--policy-fs-schedule yield`` (in-station)
19.4 ms [측정: rov_gui/tools/fstereo_bench_out/policy_fs_schedule_20261012_
100908.json, FS scale 0.75, synthetic pairs, no camera/GUI]. The plan age that
reached the controller was close between the two remedies there (emit age p50
222.3 vs 233.3 ms), so this flag exists to fly the two layouts against each
other in water, not because either is known to be better (memory
fs-server-split-review).

What moves and what stays
-------------------------
Only :meth:`FStereoSession.infer` moves — the same class, the same code, in a
child interpreter (``python -m rov_gui.perception.fstereo_proc``, launched with
the station's own ``sys.executable`` so it sees the same torch and checkout,
and handed the station's RESOLVED repo/checkpoint paths). Everything around it
stays in the station, unchanged:

* the camera, the mono pair, its ``t_capture`` stamp and the StereoMailbox —
  the stamp never crosses the process boundary, so it cannot be re-stamped;
* :class:`~rov_gui.backends.hardware.FStereoWorker` and its tick, the depth
  panel, ``--record-stereo``, the FoundationPose return leg, the policy tap;
* :class:`~rov_gui.perception.fs_gate.FsGate` (``yield`` / ``only`` keep their
  meaning because the round trip is synchronous: one pair in flight, the
  worker waits for its result inside ``fs_begin`` ... ``fs_end``).

:class:`RemoteFStereoSession` is a drop-in for ``FStereoSession`` as the worker
uses it (start_async / ready / loading / error / load_seconds / graph /
graph_requested / iters / scale / infer_size / infer / describe / close), so
the worker code path is the same in both layouts.

The wire
--------
Two AF_UNIX socketpairs, one per direction, ``multiprocessing.connection``
framing (a 4-byte length, then a pickle; numpy arrays ride as pickled
buffers). Station -> child: ``("spec", dict)`` once, ``("rig", id, rig)`` once
per rig OBJECT (~3 MB of rectification maps; the station builds one rig per
device open), ``("infer", req, left, right, rig_id, out_size)`` per pair,
``("close",)``. Child -> station: ``("log", level, text)`` at any time,
``("ready", describe, load_s, graph)`` or ``("load_error", text)`` once, then
``("result", req, out, describe, graph)`` or ``("infer_error", req, text,
describe, graph)`` per pair. ``rect_left`` (no production reader) is not sent
back. The round trip minus the child's own solve time was 0.40-0.48 ms p50,
max ~19 ms [측정: same JSON, ``fstereo_process_meta.ipc_ms``].

Two sockets, not one, because the station SENDS on a non-blocking socket with
one deadline per message (``SEND_TIMEOUT_S``) and RECEIVES on a blocking one
in a reader thread; one socket has one blocking mode. Only the worker thread
sends, and only :meth:`close` (the worker thread) ever closes a descriptor —
every other path only shuts sockets down and kills the child, so no thread can
write into a descriptor number another thread already freed.

Failure is fail-CLOSED, the same two paths the in-process session has:

* a frame that raises in the child comes back as ``infer_error`` and is raised
  here, so the worker counts it exactly like an in-process exception
  (``INFER_FAIL_LIMIT`` in a row disarms);
* a child that exits, stops answering (``INFER_TIMEOUT_S``; the first frame,
  which captures the CUDA graph, ``FIRST_INFER_TIMEOUT_S``), cannot be written
  to within ``SEND_TIMEOUT_S``, never finishes loading (``LOAD_TIMEOUT_S``) or
  sends garbage sets :attr:`error` permanently — the worker then disarms on its
  next tick and the run has no depth. It never falls back to the in-station
  network or to the camera's own depth (one instrument per run: memory
  fstereo-learned-depth).

The child dies with the station: its loop ends on EOF of the socket, and a
watchdog thread exits it when its parent pid changes (a SIGKILLed station,
also mid-load). It runs in its own session (``start_new_session``), so a
Ctrl+C in the station's terminal reaches the station, which then closes it —
not a CUDA call mid-flight in the child.

Shutdown bound: :meth:`close` itself takes at most ~3.5 s (ask, SIGTERM,
SIGKILL, reader join). The worker's stop is a queued slot, so a tick that is
waiting on a round trip finishes that wait first — at most ``INFER_TIMEOUT_S``
(``FIRST_INFER_TIMEOUT_S`` on the first frame) plus ``SEND_TIMEOUT_S``. The
in-station session has no bound there at all: a hung CUDA call blocks forever.

RECORD BOUNDARY: ``fstereo.process`` in the run meta (always written, also
``{"mode": "in-station"}``) and ``fs_process`` on every policy plan line of
plans.jsonl. Do not pool plan age, infer_ms or FS rate across it.
"""

from __future__ import annotations

import argparse
import os
import select
import socket
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from multiprocessing.connection import Connection
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MODULE = "rov_gui.perception.fstereo_proc"

#: Wire version. The child refuses a spec that carries another one, so a
#: station and a child from two checkouts cannot half-understand each other.
PROTOCOL = 2

#: The first frame also captures the CUDA graph: 1.36-1.46 s in the child
#: [측정: rov_gui/tools/fstereo_bench_out/policy_fs_schedule_20261012_100908
#: .json, fstereo_process_meta.first_frame_ms]. A slow first frame is not a
#: fault; ~7x that is.
FIRST_INFER_TIMEOUT_S = 10.0
#: Every later round trip. FS solves in 46-99 ms live [측정: fstereo_bench_out/
#: hardware_20260902.txt; data/20260930/0930_220212/diag/latency_breakdown.json]
#: and a training job on the same GPU slowed it to ~6 Hz (memory
#: policy-plan-latency-budget). A child silent for this long is hung, and a hung
#: child holds the network's VRAM: it is stopped, not waited for.
INFER_TIMEOUT_S = 5.0
#: Model load. 4.7-10.5 s in the child [측정: same JSON, load_s_child]; a cold
#: disk and a second CUDA context get headroom.
LOAD_TIMEOUT_S = 120.0
#: One deadline for one whole message to the child (not per write call).
SEND_TIMEOUT_S = 5.0
#: close(): ask, then SIGTERM, then SIGKILL, then the reader. ~3.5 s at most,
#: inside the worker's 4000 ms stop budget [스펙: backends/base.py].
CLOSE_ASK_S = 1.5
CLOSE_TERM_S = 1.0
CLOSE_KILL_S = 0.5
CLOSE_JOIN_S = 0.5
#: How often the child checks that its parent is still the station.
PPID_POLL_S = 0.5
#: Round trips kept for the meta's rpc/ipc statistics.
STATS_WINDOW = 2000


def _fs_error():
    from .fstereo import FStereoError
    return FStereoError


def _q(values) -> Optional[dict]:
    a = np.asarray(list(values), float)
    if a.size == 0:
        return None
    return {"p50": round(float(np.percentile(a, 50)), 2),
            "p90": round(float(np.percentile(a, 90)), 2),
            "max": round(float(a.max()), 2), "n": int(a.size)}


# =============================================================================
# station side
# =============================================================================
class RemoteFStereoSession:
    """``FStereoSession``'s surface, with the network in a child process.

    Construction validates and resolves exactly like the in-process session
    (it builds one, which imports no torch), so a bad ``scale``/``size``
    raises here, synchronously, where ``FStereoWorker.setup`` expects it, and
    the child is handed absolute repo/checkpoint paths (the ones the station's
    FS-parity check hashed) rather than re-deriving them in its own cwd.

    Threads: :meth:`start_async`, :meth:`infer` and :meth:`close` are called
    from the worker thread, which is the only sender and the only closer; a
    daemon reader thread owns every ``recv``; ``ready`` / ``error`` /
    ``describe()`` / :meth:`process_meta` are mirrored values, safe to read
    from the MPC and policy threads (``FStereoWorker.meta``) without a round
    trip.
    """

    process_mode = "child"

    def __init__(self, repo=None, ckpt=None, *,
                 iters: int = 8, scale: float = 1.0,
                 z_range_mm=None, graph: bool = True, fill: int = 2,
                 size=None, python: Optional[str] = None,
                 stub: Optional[dict] = None,
                 infer_timeout_s: float = INFER_TIMEOUT_S,
                 first_infer_timeout_s: float = FIRST_INFER_TIMEOUT_S,
                 load_timeout_s: float = LOAD_TIMEOUT_S,
                 send_timeout_s: float = SEND_TIMEOUT_S):
        from .fstereo import FStereoSession, Z_RANGE_MM
        spec = FStereoSession(repo=repo, ckpt=ckpt, iters=iters, scale=scale,
                              z_range_mm=(z_range_mm if z_range_mm is not None
                                          else Z_RANGE_MM),
                              graph=graph, fill=fill, size=size)
        self.repo = Path(spec.repo).resolve()
        self.ckpt = Path(spec.ckpt).resolve()
        self.iters, self.scale = spec.iters, spec.scale
        self.infer_size = spec.infer_size
        self.z_range_mm = spec.z_range_mm
        self.fill = spec.fill
        self.graph = spec.graph
        self.graph_requested = spec.graph_requested
        # The in-process describe() before any load: the same keys, honest
        # placeholders (model/sha None, cuda_graph False) until the child says.
        self._describe = dict(spec.describe())
        self._spec = {
            "protocol": PROTOCOL,
            "repo": str(self.repo), "ckpt": str(self.ckpt),
            "iters": int(self.iters), "scale": float(self.scale),
            "size": list(self.infer_size) if self.infer_size else None,
            "z_range_mm": [float(v) for v in self.z_range_mm],
            "graph": bool(self.graph), "fill": int(self.fill),
            # `is not None`: an empty stub config is still the stub
            "stub": dict(stub) if stub is not None else None,
        }
        self.python = python or sys.executable
        self.infer_timeout_s = float(infer_timeout_s)
        self.first_infer_timeout_s = float(first_infer_timeout_s)
        self.load_timeout_s = float(load_timeout_s)
        self.send_timeout_s = float(send_timeout_s)

        self._lock = threading.Lock()
        self._on_log = None
        self._proc: Optional[subprocess.Popen] = None
        self._tx: Optional[socket.socket] = None      # station -> child
        self._rx: Optional[socket.socket] = None      # child -> station
        self._reader: Optional[threading.Thread] = None
        self._loading = False
        self._ready = False
        self._closing = False
        self._closed = False
        self._error = ""
        self._t_load0 = 0.0
        self._load_s = 0.0
        self._child_load_s = None
        self._solve_ms = 0.0
        self._req = 0
        self._pending: Optional[dict] = None
        self._rig_obj = None
        self._rig_id = 0
        self._rig_sends = 0
        self._n_ok = 0
        self._n_err = 0
        self._first_ms = None
        self._rpc_ms = deque(maxlen=STATS_WINDOW)
        self._ipc_ms = deque(maxlen=STATS_WINDOW)
        self._exit_code = None

    # ------------------------------------------------------------- lifecycle
    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def loading(self) -> bool:
        if self._loading and self._t_load0 and not self._error \
                and time.monotonic() - self._t_load0 > self.load_timeout_s:
            # Polled every 5 ms by the worker, so the bound needs no thread.
            self._fail(f"FoundationStereo process did not finish loading in "
                       f"{self.load_timeout_s:.0f} s — stopped it")
            self._kill()
        return self._loading

    @property
    def error(self) -> str:
        return self._error

    @property
    def solve_ms(self) -> float:
        return self._solve_ms

    @property
    def load_seconds(self) -> float:
        if self._ready:
            return self._load_s
        if self._loading and self._t_load0:
            return time.monotonic() - self._t_load0
        return 0.0

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc is not None else None

    def start_async(self, on_log=None) -> None:
        """Launch the child and return at once; it loads the network itself
        and the reader thread flips :attr:`ready` when it says so."""
        if self._proc is not None or self._closing:
            return
        self._on_log = on_log
        self._loading = True
        self._t_load0 = time.monotonic()
        tx, tx_child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        rx, rx_child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            argv = [self.python, "-m", MODULE,
                    "--rx-fd", str(tx_child.fileno()),
                    "--tx-fd", str(rx_child.fileno()),
                    "--ppid", str(os.getpid())]
            env = dict(os.environ)
            env.setdefault("PYTHONUNBUFFERED", "1")
            self._proc = subprocess.Popen(
                argv, cwd=str(REPO), env=env, stdin=subprocess.DEVNULL,
                pass_fds=(tx_child.fileno(), rx_child.fileno()),
                start_new_session=True)
        except Exception as e:                                   # noqa: BLE001
            for s in (tx, tx_child, rx, rx_child):
                s.close()
            self._loading = False
            self._fail(f"could not start the FoundationStereo process: "
                       f"{type(e).__name__}: {e}")
            return
        tx_child.close()                    # the child holds its own copies
        rx_child.close()
        tx.setblocking(False)               # sends are deadline-bounded
        self._tx, self._rx = tx, rx
        # The reader owns a DUP of the receive side: shutdown() on `rx` wakes
        # its blocking recv (closing a descriptor another thread is reading
        # does not), and the reader closes only its own dup.
        rconn = Connection(os.dup(rx.fileno()), writable=False)
        self._reader = threading.Thread(target=self._read_loop, args=(rconn,),
                                        name="fstereo-proc-rx", daemon=True)
        self._reader.start()
        if not self._send(("spec", self._spec)):
            return
        self._say("info", f"fstereo: the network runs in its OWN process "
                          f"(pid {self._proc.pid}, --fstereo-process); the "
                          f"camera, the depth panel and the policy tap stay "
                          f"in the station")

    def close(self) -> None:
        """Stop the child within ~3.5 s, whatever state it is in. Worker
        thread only — the one place that closes descriptors."""
        if self._closed:
            return
        self._closing = True
        self._ready = False
        self._loading = False
        p = self._proc
        if p is not None and p.poll() is None:
            self._send(("close",), quiet=True)
            try:
                p.wait(CLOSE_ASK_S)
            except subprocess.TimeoutExpired:
                p.terminate()
                try:
                    p.wait(CLOSE_TERM_S)
                except subprocess.TimeoutExpired:
                    p.kill()
                    try:
                        p.wait(CLOSE_KILL_S)
                    except subprocess.TimeoutExpired:
                        pass
        if p is not None:
            self._exit_code = p.poll()
        self._shutdown_channel()
        r = self._reader
        if r is not None and r is not threading.current_thread():
            r.join(CLOSE_JOIN_S)
        for s in (self._tx, self._rx):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass
        self._closed = True

    # ------------------------------------------------------------- the work
    def infer(self, left_raw: np.ndarray, right_raw: np.ndarray, rig,
              out_size=None) -> dict:
        """One synchronous round trip; the same dict ``FStereoSession.infer``
        returns, minus ``rect_left``."""
        FStereoError = _fs_error()
        if self._error:
            raise FStereoError(self._error)
        if not self._ready:
            raise FStereoError("FoundationStereo process is not ready")
        t0 = time.perf_counter()
        if rig is not self._rig_obj:
            # By identity: the rig is unhashable and has no cheap equality,
            # and the station builds exactly one per device open.
            self._rig_id += 1
            if not self._send(("rig", self._rig_id, rig)):
                raise FStereoError(self._error)
            self._rig_obj = rig
            self._rig_sends += 1
        self._req += 1
        slot = {"id": self._req, "ev": threading.Event(), "msg": None}
        with self._lock:
            # Under the lock _fail also takes: either it already ran (the
            # error is visible here) or it will find this slot and wake it —
            # never a child that died unseen while this waits out a timeout.
            if self._error:
                raise FStereoError(self._error)
            self._pending = slot
        if not self._send(("infer", self._req,
                           np.ascontiguousarray(left_raw),
                           np.ascontiguousarray(right_raw), self._rig_id,
                           tuple(int(v) for v in out_size) if out_size
                           else None)):
            raise FStereoError(self._error)
        first = self._n_ok == 0
        timeout = self.first_infer_timeout_s if first else self.infer_timeout_s
        if not slot["ev"].wait(timeout):
            self._fail(f"FoundationStereo process did not answer a frame in "
                       f"{timeout:.1f} s — stopped it (a hung child holds the "
                       f"GPU)")
            self._kill()
            raise FStereoError(self._error)
        msg = slot["msg"]
        if msg is None:                     # woken by the child going away
            raise FStereoError(self._error or "FoundationStereo process exited")
        rtt = (time.perf_counter() - t0) * 1000.0
        if msg[0] == "infer_error":
            self._n_err += 1
            raise FStereoError(str(msg[2]))
        out = msg[2]
        solve = float(out.get("solve_ms", 0.0) or 0.0)
        self._solve_ms = solve
        if first:
            # The graph capture, recorded on its own and kept out of the
            # steady-state statistics it would otherwise dominate.
            self._first_ms = round(rtt, 1)
        else:
            self._rpc_ms.append(rtt)
            self._ipc_ms.append(max(0.0, rtt - solve))
        self._n_ok += 1
        return out

    # ------------------------------------------------------------- record
    def describe(self) -> dict:
        """The child's own ``FStereoSession.describe()`` as last reported —
        a fresh dict every call (FStereoWorker.meta updates it in place)."""
        with self._lock:
            return dict(self._describe)

    def process_meta(self) -> dict:
        """``fstereo.process`` in the run meta."""
        p = self._proc
        return {
            "mode": "child",
            "pid": p.pid if p is not None else None,
            "alive": bool(p is not None and p.poll() is None),
            "exit_code": (p.poll() if p is not None else self._exit_code),
            "python": self.python,
            "transport": "two AF_UNIX socketpairs, multiprocessing.connection "
                         "framing (pickled numpy)",
            "protocol": PROTOCOL,
            "load_s_child": self._child_load_s,
            "load_s_station": round(self._load_s, 2) if self._load_s else None,
            "frames_ok": int(self._n_ok),
            "frames_error": int(self._n_err),
            "first_frame_ms": self._first_ms,
            # round trip as the worker waited for it, and that minus the
            # child's own solve_ms (pickling + two socket hops + wake-ups)
            "rpc_ms": _q(self._rpc_ms),
            "ipc_ms": _q(self._ipc_ms),
            "stats_scope": (f"the last <= {STATS_WINDOW} round trips of this "
                            f"STATION SESSION when the meta was written, frame "
                            f"1 excluded (first_frame_ms) — not this CSV's "
                            f"window alone"),
            "rig_sends": int(self._rig_sends),
            "timeouts_s": {"first_infer": self.first_infer_timeout_s,
                           "infer": self.infer_timeout_s,
                           "load": self.load_timeout_s,
                           "send_per_message": self.send_timeout_s},
        }

    # ------------------------------------------------------------- internals
    def _say(self, level: str, text: str) -> None:
        cb = self._on_log
        if cb is not None and not self._closing:
            try:
                cb(level, text)
            except Exception:                                    # noqa: BLE001
                pass

    def _fail(self, text: str) -> None:
        """Permanent: the worker disarms on a non-empty ``error``."""
        first = False
        with self._lock:
            if not self._error:
                self._error = text
                first = True
            self._ready = False
            self._loading = False
            slot, self._pending = self._pending, None
        if slot is not None:
            slot["ev"].set()                # msg stays None: infer raises now
        if first:
            self._say("error", f"fstereo: {text}")

    def _send(self, msg, quiet: bool = False) -> bool:
        """One message, one deadline. Worker thread only."""
        tx = self._tx
        if tx is None:
            return False
        try:
            buf = ForkingPickler.dumps(msg)
            n = len(buf)
            if n > 0x7FFFFFFF:
                raise ValueError(f"message of {n} bytes is too large")
            views = (memoryview(struct.pack("!i", n)), memoryview(buf).cast("B"))
            deadline = time.monotonic() + self.send_timeout_s
            poller = select.poll()
            poller.register(tx.fileno(), select.POLLOUT)
            for view in views:
                while len(view):
                    left = deadline - time.monotonic()
                    if left <= 0.0:
                        raise TimeoutError(f"the child read nothing for "
                                           f"{self.send_timeout_s:.1f} s")
                    if not poller.poll(max(1, int(left * 1000.0))):
                        continue
                    try:
                        k = tx.send(view)
                    except (BlockingIOError, InterruptedError):
                        continue
                    view = view[k:]
            return True
        except Exception as e:                                   # noqa: BLE001
            if not quiet and not self._closing:
                self._fail(f"lost the FoundationStereo process (send: "
                           f"{type(e).__name__}: {e})")
                self._kill()
            return False

    def _kill(self) -> None:
        """Any thread. Kills and shuts down; never closes a descriptor."""
        p = self._proc
        if p is not None and p.poll() is None:
            try:
                p.kill()
                p.wait(CLOSE_KILL_S)
            except Exception:                                    # noqa: BLE001
                pass
        if p is not None:
            self._exit_code = p.poll()
        self._shutdown_channel()

    def _shutdown_channel(self) -> None:
        for s in (self._tx, self._rx):
            if s is not None:
                try:
                    s.shutdown(socket.SHUT_RDWR)   # wakes the reader's recv
                except OSError:
                    pass

    def _read_loop(self, conn: Connection) -> None:
        try:
            while True:
                try:
                    msg = conn.recv()
                except (EOFError, OSError):
                    break
                except Exception as e:                           # noqa: BLE001
                    # A message that does not unpickle is a desync; nothing
                    # after it can be trusted.
                    self._fail(f"garbled message from the FoundationStereo "
                               f"process ({type(e).__name__}: {e})")
                    self._kill()
                    break
                if not isinstance(msg, tuple) or not msg:
                    continue
                self._dispatch(msg)
        finally:
            p = self._proc
            if not self._closing:
                rc = p.poll() if p is not None else None
                if rc is None and p is not None:
                    try:
                        rc = p.wait(0.5)
                    except Exception:                            # noqa: BLE001
                        rc = None
                self._exit_code = rc
                self._fail(f"FoundationStereo process exited (code {rc}) — "
                           f"this run has no depth from here on")
            try:
                conn.close()                 # the reader's own dup, only
            except OSError:
                pass

    def _dispatch(self, msg: tuple) -> None:
        kind = msg[0]
        if kind == "log" and len(msg) >= 3:
            self._say(str(msg[1]), str(msg[2]))
        elif kind == "ready":
            with self._lock:
                self._describe = dict(msg[1] or {})
                self._child_load_s = (round(float(msg[2]), 2)
                                      if msg[2] is not None else None)
                self.graph = bool(msg[3])
                self._load_s = time.monotonic() - self._t_load0
                if not self._error and not self._closing:
                    self._ready = True
                self._loading = False
        elif kind == "load_error":
            self._fail(str(msg[1]))
        elif kind in ("result", "infer_error"):
            with self._lock:
                slot = self._pending
                if slot is None or slot["id"] != msg[1]:
                    return          # the answer to a request already given up on
                self._pending = None
                if len(msg) >= 5:
                    self._describe = dict(msg[3] or {})
                    self.graph = bool(msg[4])
            slot["msg"] = msg
            slot["ev"].set()


# =============================================================================
# child side
# =============================================================================
class _StubSession:
    """FStereoSession's surface with no torch — TESTS ONLY (spec ``stub``).

    Deterministic output from BOTH images of the pair (``left*10 + right``, so
    a swapped or duplicated image shows), plus scripted misbehaviour by frame
    number (1-based): ``raise_on`` (an exception, like a bad frame),
    ``crash_on`` (the process dies), ``hang_on`` (stops answering),
    ``load_error`` / ``load_s`` (the load), ``ignore_close`` (does not exit
    when asked). The station never sends a stub spec.
    """

    def __init__(self, cfg: dict, iters: int, scale: float, graph: bool):
        self.cfg = dict(cfg)
        self.iters, self.scale, self.infer_size = iters, scale, None
        self.graph = self.graph_requested = bool(graph)
        self.ready = False
        self.loading = False
        self.error = ""
        self.load_seconds = 0.0
        self.n = 0

    def start_async(self, on_log=None) -> None:
        self.loading = True
        time.sleep(float(self.cfg.get("load_s", 0.0)))
        if self.cfg.get("load_error"):
            self.error = str(self.cfg["load_error"])
        else:
            self.ready = True
            if on_log is not None:
                on_log("info", "fstereo: stub ready")
        self.load_seconds = float(self.cfg.get("load_s", 0.0))
        self.loading = False

    def infer(self, left, right, rig, out_size=None) -> dict:
        self.n += 1
        if self.n == self.cfg.get("crash_on"):
            os._exit(7)
        if self.n == self.cfg.get("hang_on"):
            time.sleep(3600.0)
        if self.n == self.cfg.get("raise_on"):
            raise RuntimeError(f"stub frame {self.n} failed")
        time.sleep(float(self.cfg.get("infer_s", 0.0)))
        native = (left.astype(np.uint16) * 10
                  + right.astype(np.uint16)).astype(np.uint16)
        w, h = (out_size or (left.shape[1], left.shape[0]))
        depth_mm = np.full((int(h), int(w)), int(self.n), np.uint16)
        return {"depth_mm": depth_mm, "depth_native": native,
                "rect_left": left, "valid_native": 97.0, "valid_out": 90.0,
                "filled_out": 1.0, "solve_ms": 2.5,
                "rig_k": float(getattr(rig, "k", 0.0))}

    def describe(self) -> dict:
        return {"depth_source": "stub", "stub": True, "iters": self.iters,
                "scale": self.scale, "frames": self.n,
                "cuda_graph": bool(self.graph)}

    def close(self) -> None:
        if self.cfg.get("ignore_close"):
            time.sleep(3600.0)


def _watch_parent(ppid: int) -> None:
    """Exit when the station is gone (reparented), even mid-load."""
    while True:
        time.sleep(PPID_POLL_S)
        if os.getppid() != ppid:
            os._exit(3)


def child_main(argv=None) -> int:
    import faulthandler
    faulthandler.enable()        # a native crash here should name its stack
    ap = argparse.ArgumentParser(prog=MODULE, description=(
        "FoundationStereo server child of the rov_gui station "
        "(--fstereo-process). Not meant to be started by hand."))
    ap.add_argument("--rx-fd", type=int, required=True)
    ap.add_argument("--tx-fd", type=int, required=True)
    ap.add_argument("--ppid", type=int, required=True)
    a = ap.parse_args(argv)
    if os.getppid() != a.ppid:
        return 3                            # the station is already gone
    threading.Thread(target=_watch_parent, args=(a.ppid,),
                     name="fstereo-proc-ppid", daemon=True).start()
    rx = Connection(a.rx_fd, writable=False)
    tx = Connection(a.tx_fd, readable=False)
    send_lock = threading.Lock()

    def send(msg) -> bool:
        try:
            with send_lock:
                tx.send(msg)
            return True
        except Exception:                                        # noqa: BLE001
            return False

    try:
        kind, spec = rx.recv()
    except Exception:                                            # noqa: BLE001
        return 0
    if kind != "spec" or not isinstance(spec, dict) \
            or spec.get("protocol") != PROTOCOL:
        got = spec.get("protocol") if isinstance(spec, dict) else spec
        send(("load_error", f"FoundationStereo process: protocol mismatch "
                            f"(station sent {got!r}, child speaks {PROTOCOL})"))
        return 2

    def on_log(level, text) -> None:
        send(("log", str(level), str(text)))

    if spec.get("stub") is not None:
        session = _StubSession(spec["stub"], spec["iters"], spec["scale"],
                               spec["graph"])
    else:
        from rov_gui.perception.fstereo import FStereoSession
        session = FStereoSession(
            repo=spec["repo"], ckpt=spec["ckpt"], iters=spec["iters"],
            scale=spec["scale"],
            z_range_mm=tuple(spec["z_range_mm"]), graph=spec["graph"],
            fill=spec["fill"],
            size=tuple(spec["size"]) if spec.get("size") else None)
    session.start_async(on_log=on_log)
    while session.loading:
        time.sleep(0.02)
    if session.error or not session.ready:
        send(("load_error", session.error or "FoundationStereo did not load"))
        return 2
    if not send(("ready", session.describe(), float(session.load_seconds),
                 bool(session.graph))):
        return 0
    rigs: dict = {}
    while True:
        try:
            msg = rx.recv()
        except Exception:                                        # noqa: BLE001
            break                           # EOF: the station closed or died
        kind = msg[0]
        if kind == "rig":
            rigs.clear()                    # one live rig at a time
            rigs[msg[1]] = msg[2]
        elif kind == "infer":
            _, rid, left, right, rig_id, out_size = msg
            try:
                if rig_id not in rigs:
                    raise RuntimeError(f"no rig {rig_id} on the child side")
                out = session.infer(left, right, rigs[rig_id],
                                    out_size=out_size)
                out.pop("rect_left", None)  # no production reader; 256 kB
                ok = send(("result", rid, out, session.describe(),
                           bool(session.graph)))
            except Exception as e:                               # noqa: BLE001
                ok = send(("infer_error", rid, f"{type(e).__name__}: {e}",
                           session.describe(), bool(session.graph)))
            if not ok:
                break
        elif kind == "close":
            break
    try:
        session.close()
    except Exception:                                            # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(child_main())
