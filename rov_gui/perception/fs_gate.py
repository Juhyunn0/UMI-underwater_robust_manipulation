#!/usr/bin/env python3
"""fs_gate.py — who gets the GPU: FoundationStereo or the diffusion policy.

WHY THIS EXISTS. Both networks run in one process on one GPU, each on its own
thread (``FStereoWorker`` and ``PolicyWorker``). At the ``--policy`` defaults
FoundationStereo takes longer per frame than the camera's frame interval, so
it runs back to back and the policy's forward waits behind it: the same
forward that takes 19.3 ms alone takes 152.8 ms p50 with FoundationStereo
beside it [측정: data/20260930/0930_220212/diag/
policy_vs_fstereo_contention.json, offline, RTX 5090] and 207-273 ms p50 on
the vehicle [측정: data/20260930/0930_220212/diag/plan_timing.json]. A separate
CUDA stream does not help — the two share the device, not a queue.

This object is the ONE thing the two workers share to settle that. It never
touches the GPU, a mailbox or a network: it only answers "may FoundationStereo
start a frame now" and lets the policy wait, bounded, for the frame in flight.

Three schedules (``--policy-fs-schedule``):

  free   today's behaviour. The backend builds NO gate; both workers see
         ``fs_gate is None`` and make no call into this module.
  yield  FoundationStereo runs as always, except that while the policy is
         inferring it does not START a frame. The policy calls :meth:`hold`
         when an attempt begins, :meth:`wait_idle` just before its forward
         (the frame in flight is allowed to finish) and :meth:`release` after.
         The same depth frames are chosen as under ``free``; only the forward's
         timing changes.
  only   while a policy mission is active FoundationStereo computes ONLY the
         frames the policy asks for — :meth:`request_burst` of two per period —
         and discards the pairs in between (:data:`FS_DRAIN`), so the first
         frame of a burst is a pair that arrived after the request and the
         second is the next camera frame (as long as one frame takes less
         than two camera intervals). A burst is counted in frames DELIVERED
         to the policy (:meth:`fs_delivered`), not frames computed: a frame
         whose inference failed, or that never reached the policy mailbox,
         does not use it up, and the stereo network simply runs the next
         pair. Outside a mission it runs free.

NOTHING HERE CAN STOP FoundationStereo FOR GOOD. This run has no other depth
source, a worker's tick exception is swallowed by ``TimerWorker._tick``, and
the policy worker stops before the stereo worker at shutdown — so every way of
holding it expires by itself:

  * a :meth:`hold` is a LEASE (``lease_s``): older than that, it is ignored;
  * ``only`` needs a HEARTBEAT (``heartbeat_s``): the policy worker calls
    :meth:`set_active` every tick, and when the calls stop the gate is open;
  * ``only`` with a mission active but nobody asking for a frame for
    ``starve_s`` opens too (a stuck requester must not darken the panel);
  * an in-flight mark older than ``inflight_max_s`` is stale (the stereo tick
    raised between :meth:`fs_begin` and :meth:`fs_end`) and nobody waits on it.

And nothing here blocks the stereo thread: :meth:`fs_begin` returns at once
(its tick must keep returning to the event loop, or a queued shutdown is never
delivered). Only :meth:`wait_idle` blocks, on the policy thread, for at most
``wait_s`` — far inside the 2 s after which the controller calls the policy
worker silent.

Pure ``threading`` + stdlib: no Qt, no torch, so the state machine is testable
alone (rov_gui/tests/test_fs_gate.py). Every method is safe from any thread;
the lock is held for bookkeeping only, never across a GPU call, so
:meth:`snapshot` (called from the controller's thread for the run meta) cannot
stall a control tick.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Optional, Tuple

#: The values of ``--policy-fs-schedule``. ``free`` never reaches this module
#: as a gate (the backend builds none); it is listed for the CLI and the meta.
SCHEDULES = ("free", "yield", "only")

#: What :meth:`FsGate.fs_begin` tells the stereo worker.
FS_GO = "go"          # start a frame; it is now in flight — call fs_end()
FS_HOLD = "hold"      # do not start; leave the waiting pair in the mailbox
FS_DRAIN = "drain"    # do not start, and discard the waiting pair

#: Frames per policy observation: the newest depth frame and the one before
#: it (``PolicyWorker._pick_pair``).
BURST_FRAMES = 2


class FsGate:
    """See the module docstring. One instance per run, built by
    ``HardwareBackend`` and handed to both workers."""

    def __init__(self, mode: str, *,
                 clock: Callable[[], float] = time.monotonic,
                 lease_s: float = 0.5,
                 wait_s: float = 0.15,
                 inflight_max_s: float = 2.5,
                 heartbeat_s: float = 1.0,
                 burst_timeout_s: float = 0.6,
                 starve_s: float = 1.5):
        mode = str(mode)
        if mode not in SCHEDULES or mode == "free":
            raise ValueError(f"FsGate mode must be 'yield' or 'only', got "
                             f"{mode!r} ('free' builds no gate)")
        self.mode = mode
        self._clock = clock
        #: A hold older than this is ignored by fs_begin. [결정] The forward
        #: alone is 19.3 ms (module docstring) and the bounded wait 150 ms,
        #: so a healthy hold is far shorter; this only bounds a hold whose
        #: release never came.
        self.lease_s = float(lease_s)
        #: wait_idle's default bound. The policy holds at an arbitrary phase
        #: of the frame in flight, so the wait is whatever is left of that
        #: frame once the observation is built — not zero. One frame at the
        #: --policy defaults is 75.8 ms p50 offline (back to back, beside the
        #: policy) [측정: data/20260930/0930_220212/diag/
        #: policy_vs_fstereo_contention.json fs_back_to_back.fs_ms] and ~99 ms
        #: on the vehicle [측정: data/20260930/0930_220212/diag/
        #: latency_breakdown.json]; 137 ms at the training store's setting
        #: [측정: rov_gui/tools/fstereo_bench_out/
        #: policy_bench_20260902_141224.json]. 150 ms outlasts all three; a
        #: timeout means a longer frame still (the first frame's graph
        #: capture, a GPU shared with something else), and the forward then
        #: runs beside the tail of that frame — today's behaviour.
        self.wait_s = float(wait_s)
        #: [결정] > the first frame's CUDA-graph capture, which runs inline in
        #: the stereo tick (1.3 s [측정: rov_gui/tools/fstereo_bench_out/
        #: session.txt]) — a mark older than this is a tick that raised.
        self.inflight_max_s = float(inflight_max_s)
        #: [결정] the policy worker ticks every 20 ms and its PolicyState goes
        #: stale at 1.0 s (backends/policy.py STATE_STALE_S): same horizon.
        self.heartbeat_s = float(heartbeat_s)
        #: [결정] one period (0.5 s) plus slack: a burst not delivered by then
        #: is void and the policy asks again.
        self.burst_timeout_s = float(burst_timeout_s)
        #: [결정] < FStereoWorker.IDLE_S (2.0 s), so a schedule alone never
        #: makes the panel say "FS STARVED".
        self.starve_s = float(starve_s)

        self._cv = threading.Condition(threading.Lock())
        # the frame in flight (stereo thread)
        self._in_flight = False
        self._t_begin = 0.0
        # the policy's hold (policy thread)
        self._hold_t: Optional[float] = None
        self._lease_said = False
        # `only`: the mission heartbeat and the burst allowance
        self._active = False
        self._t_beat = 0.0
        self._t_grant = 0.0          # last request_burst / activation
        self._allow = 0
        self._t_request = 0.0
        self._starving = False
        # the record
        self._n = {
            "frames": 0,             # inferences that returned a map
            "delivered": 0,          # ...and reached the policy mailbox
            "held_polls": 0,         # stereo ticks refused (5 ms polls, not frames)
            "drained": 0,            # pairs discarded unprocessed (`only`)
            "holds": 0,
            "lease_expired": 0,      # holds that outlived lease_s
            "waits": 0,
            "wait_timeouts": 0,
            "inflight_reset": 0,     # fs_begin found a stale in-flight mark
            "bursts": 0,             # request_burst granted
            "burst_frames": 0,       # deliveries charged to a burst
            "burst_void": 0,         # allowances that expired unused
            "starve_free": 0,        # times the starve watchdog opened the gate
            "activations": 0,
        }
        self._wait_ms: deque = deque(maxlen=1000)
        self._hold_ms: deque = deque(maxlen=1000)

    # ------------------------------------------------------- stereo thread
    def fs_begin(self) -> str:
        """May FoundationStereo start a frame NOW? Never blocks.

        :data:`FS_GO` marks a frame in flight and MUST be followed by exactly
        one :meth:`fs_end` (``ran=False`` when no pair was there to start).
        The check and the mark are one critical section, so a policy that
        calls :meth:`hold` either lands before the mark (this returns
        HOLD/DRAIN) or after it (:meth:`wait_idle` then sees the frame).
        """
        with self._cv:
            t = self._clock()
            if self._in_flight:
                # The previous frame never reported its end: the stereo tick
                # raised between begin and end. One thread calls this, so the
                # mark cannot be anyone else's.
                self._n["inflight_reset"] += 1
                self._in_flight = False
                self._cv.notify_all()
            mission = self._mission_active(t)
            if self._held(t):
                self._n["held_polls"] += 1
                return FS_DRAIN if (self.mode == "only" and mission) else FS_HOLD
            if self.mode == "only" and mission:
                if self._allow > 0 and (t - self._t_request) > self.burst_timeout_s:
                    self._allow = 0
                    self._n["burst_void"] += 1
                if self._allow <= 0:
                    if (t - self._t_grant) <= self.starve_s:
                        self._starving = False
                        self._n["held_polls"] += 1
                        return FS_DRAIN
                    # Active, and nobody has asked for a frame for starve_s:
                    # run free until someone does.
                    if not self._starving:
                        self._starving = True
                        self._n["starve_free"] += 1
            self._in_flight = True
            self._t_begin = t
            return FS_GO

    def fs_end(self, ran: bool = True) -> None:
        """The frame :meth:`fs_begin` allowed is over: the GPU is free.
        ``ran=False`` when no map came out of it (there was no pair to start,
        or the inference raised). Wakes a policy waiting in
        :meth:`wait_idle`. Does NOT use up a burst — see
        :meth:`fs_delivered`."""
        with self._cv:
            self._in_flight = False
            if ran:
                self._n["frames"] += 1
            self._cv.notify_all()

    def fs_delivered(self) -> None:
        """A depth frame was put into the policy's mailbox. Under ``only``,
        during a mission, this is what a burst is counted in — so the gate
        and the policy worker (which counts the frames that ARRIVE after its
        request) count the same thing, whichever came first: a frame in
        flight when the request was made is the first of the burst, and so is
        one that had finished but not yet been handed over."""
        with self._cv:
            self._n["delivered"] += 1
            if (self.mode == "only" and self._allow > 0
                    and self._mission_active(self._clock())):
                self._allow -= 1
                self._n["burst_frames"] += 1

    def fs_drained(self, n: int = 1) -> None:
        """The stereo worker discarded ``n`` pairs on :data:`FS_DRAIN`."""
        with self._cv:
            self._n["drained"] += int(n)

    # ------------------------------------------------------- policy thread
    def hold(self) -> None:
        """FoundationStereo must not START a frame until :meth:`release`
        (or ``lease_s``, whichever comes first). The frame in flight is not
        interrupted — :meth:`wait_idle` waits for it."""
        with self._cv:
            self._hold_t = self._clock()
            self._lease_said = False
            self._n["holds"] += 1

    def wait_idle(self, timeout: Optional[float] = None) -> Tuple[float, bool]:
        """Block until no FoundationStereo frame is in flight, at most
        ``timeout`` (default ``wait_s``) seconds. Returns ``(waited_ms,
        timed_out)``; on a timeout the caller goes ahead anyway — that is
        today's behaviour, never a failure."""
        limit = self.wait_s if timeout is None else max(0.0, float(timeout))
        # The bound is REAL time (Condition.wait sleeps in real time): with an
        # injected clock that does not advance, a bound measured on that clock
        # would never expire. The injected clock only judges staleness.
        t0 = time.monotonic()
        timed_out = False
        with self._cv:
            while self._flying(self._clock()):
                left = limit - (time.monotonic() - t0)
                if left <= 0.0:
                    timed_out = True
                    break
                self._cv.wait(left)
            ms = max(0.0, (time.monotonic() - t0) * 1e3)
            self._n["waits"] += 1
            if timed_out:
                self._n["wait_timeouts"] += 1
            self._wait_ms.append(ms)
        return ms, timed_out

    def release(self) -> None:
        """End of the policy's hold. Idempotent; safe without a hold."""
        with self._cv:
            if self._hold_t is not None:
                self._hold_ms.append(
                    max(0.0, (self._clock() - self._hold_t) * 1e3))
            self._hold_t = None

    def set_active(self, active: bool) -> None:
        """``only``: the policy mission is running (the heartbeat — call it
        every tick) or is not. Inactive, or silent for ``heartbeat_s``, the
        gate is open. A no-op under ``yield``."""
        if self.mode != "only":
            return
        with self._cv:
            t = self._clock()
            if active:
                if not self._mission_active(t):
                    self._n["activations"] += 1
                    self._t_grant = t
                    self._starving = False
                self._active = True
                self._t_beat = t
            else:
                self._active = False
                self._allow = 0

    def request_burst(self, n: int = BURST_FRAMES) -> bool:
        """``only``: let FoundationStereo run until ``n`` frames have been
        delivered to the policy. False when a burst is already pending (or
        under ``yield``, where there is nothing to ask for). A frame in
        flight when this is called counts as the first."""
        if self.mode != "only":
            return False
        with self._cv:
            t = self._clock()
            if self._allow > 0 and (t - self._t_request) <= self.burst_timeout_s:
                return False
            if self._allow > 0:
                self._n["burst_void"] += 1
            self._allow = max(1, int(n))
            self._t_request = t
            self._t_grant = t
            self._starving = False
            self._n["bursts"] += 1
            return True

    # ---------------------------------------------------------------- record
    def snapshot(self) -> dict:
        """Plain, JSON-serialisable state and counters (run meta). Safe from
        any thread; the lock is held only to copy."""
        with self._cv:
            t = self._clock()
            wait = sorted(self._wait_ms)
            hold = sorted(self._hold_ms)
            out = {
                "mode": self.mode,
                "lease_s": self.lease_s, "wait_s": self.wait_s,
                "inflight_max_s": self.inflight_max_s,
                "heartbeat_s": self.heartbeat_s,
                "burst_timeout_s": self.burst_timeout_s,
                "starve_s": self.starve_s,
                "burst_frames_per_request": BURST_FRAMES,
                "counters": dict(self._n),
                "held": bool(self._held(t, count=False)),
                "in_flight": bool(self._flying(t)),
                "mission_active": bool(self._mission_active(t)),
                "burst_allowance": int(self._allow),
            }
        out["wait_ms"] = _summary(wait)
        out["hold_ms"] = _summary(hold)
        return out

    # --------------------------------------------------------------- private
    def _held(self, t: float, count: bool = True) -> bool:
        if self._hold_t is None:
            return False
        if (t - self._hold_t) <= self.lease_s:
            return True
        if count and not self._lease_said:
            self._lease_said = True
            self._n["lease_expired"] += 1
        return False

    def _flying(self, t: float) -> bool:
        return self._in_flight and (t - self._t_begin) <= self.inflight_max_s

    def _mission_active(self, t: float) -> bool:
        return self._active and (t - self._t_beat) <= self.heartbeat_s


def _summary(sorted_ms) -> dict:
    """{p50, p90, max, n} of an already-sorted list of milliseconds; the
    values are None when empty (a 0.0 would read as "waited, and it was
    nothing")."""
    n = len(sorted_ms)
    if n == 0:
        return {"p50": None, "p90": None, "max": None, "n": 0}
    return {"p50": round(float(sorted_ms[n // 2]), 2),
            "p90": round(float(sorted_ms[min(n - 1, int(0.9 * n))]), 2),
            "max": round(float(sorted_ms[-1]), 2), "n": n}
